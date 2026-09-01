#!/usr/bin/env python3
"""Five-fold frozen Nucleotide Transformer layer-wise probing.

One frozen NT forward pass feeds independent, identically initialized
attention-pooling heads for embedding layer 0, transformer layers 1--12, and a
learned scalar mixture of layers 1--12.  Each candidate has its own optimizer,
scheduler, early-stopping state, and checkpoint.  The input data, grouped folds,
raw 20-bin target, objective, and pooling head match train_nt_heads.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer


SOURCE_DIR = Path("/root/sc-motif-open/R")
sys.path.insert(0, str(SOURCE_DIR))

from train_nt_heads import (  # noqa: E402
    AttentionPoolingHead,
    FOLDS_FILE,
    IndexDataset,
    MODEL_PATH,
    SequenceCollator,
    load_folds,
    set_seed,
    tokenize_batch,
)


LAYER_KEYS = [f"layer_{index:02d}" for index in range(13)]
WEIGHTED_KEY = "weighted_01_12"
ALL_KEYS = LAYER_KEYS + [WEIGHTED_KEY]


class FrozenLayerwiseNT(nn.Module):
    def __init__(self, nt_model: nn.Module):
        super().__init__()
        self.nt = nt_model
        for parameter in self.nt.parameters():
            parameter.requires_grad = False
        self.nt.eval()

        # Every layer starts from exactly the same head initialization.
        base_head = AttentionPoolingHead(input_dim=512, hidden_dim=256, dropout=0.1)
        self.heads = nn.ModuleDict({key: copy.deepcopy(base_head) for key in ALL_KEYS})
        self.layer_mix_logits = nn.Parameter(torch.zeros(12))

    def train(self, mode: bool = True):
        super().train(mode)
        self.nt.eval()
        return self

    def candidate_parameters(self, key: str):
        parameters = list(self.heads[key].parameters())
        if key == WEIGHTED_KEY:
            parameters.append(self.layer_mix_logits)
        return parameters

    def candidate_state(self, key: str):
        state = {"head": copy.deepcopy(self.heads[key].state_dict())}
        if key == WEIGHTED_KEY:
            state["layer_mix_logits"] = self.layer_mix_logits.detach().cpu().clone()
        return state

    def load_candidate_state(self, key: str, state: dict):
        self.heads[key].load_state_dict(state["head"])
        if key == WEIGHTED_KEY:
            self.layer_mix_logits.data.copy_(state["layer_mix_logits"].to(self.layer_mix_logits.device))

    def forward(self, input_ids, attention_mask, keys):
        with torch.no_grad():
            outputs = self.nt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states

        if len(hidden_states) != 13:
            raise RuntimeError(f"Expected embedding + 12 hidden layers, found {len(hidden_states)}")

        valid_mask = attention_mask.bool().clone()
        valid_mask[:, 0] = False
        predictions = {}
        for key in keys:
            if key == WEIGHTED_KEY:
                weights = torch.softmax(self.layer_mix_logits, dim=0)
                representation = sum(weights[index] * hidden_states[index + 1] for index in range(12))
            else:
                representation = hidden_states[int(key.rsplit("_", 1)[1])]
            predictions[key] = self.heads[key](representation, valid_mask)
        return predictions


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, keys):
    model.eval()
    accumulators = {
        key: {
            "sse": np.zeros(20, dtype=np.float64),
            "sum_y": np.zeros(20, dtype=np.float64),
            "sum_y2": np.zeros(20, dtype=np.float64),
            "n": 0,
        }
        for key in keys
    }
    for seqs, target in loader:
        target = target.to(device, non_blocking=True)
        encoded = tokenize_batch(tokenizer, seqs, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(encoded["input_ids"], encoded["attention_mask"], keys)
        target64 = target.double()
        sum_y = target64.sum(0).cpu().numpy()
        sum_y2 = target64.square().sum(0).cpu().numpy()
        for key, prediction in predictions.items():
            difference = prediction.float().double() - target64
            item = accumulators[key]
            item["sse"] += difference.square().sum(0).cpu().numpy()
            item["sum_y"] += sum_y
            item["sum_y2"] += sum_y2
            item["n"] += target.shape[0]

    metrics = {}
    for key, item in accumulators.items():
        sst = item["sum_y2"] - np.square(item["sum_y"]) / item["n"]
        r2_bins = 1.0 - item["sse"] / np.maximum(sst, 1e-12)
        mse = item["sse"].sum() / (item["n"] * 20)
        metrics[key] = {
            "n": int(item["n"]),
            "mse": float(mse),
            "rmse": float(math.sqrt(mse)),
            "mean_r2": float(r2_bins.mean()),
            "r2_bins": r2_bins.tolist(),
        }
    return metrics


def train(args):
    set_seed(42 + args.fold)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    output = Path(args.output_root) / f"fold{args.fold + 1}"
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    nt_model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    model = FrozenLayerwiseNT(nt_model).to(device)
    del nt_model

    train_idx, val_idx = load_folds(FOLDS_FILE, args.fold)
    train_loader = DataLoader(
        IndexDataset(train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=SequenceCollator(),
    )
    val_loader = DataLoader(
        IndexDataset(val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=SequenceCollator(),
    )

    optimizers = {}
    schedulers = {}
    for key in ALL_KEYS:
        parameters = model.candidate_parameters(key)
        optimizers[key] = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        schedulers[key] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[key], mode="max", factor=0.5, patience=2
        )

    config = {
        **vars(args),
        "gpu": torch.cuda.get_device_name(0),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "candidates": ALL_KEYS,
        "selection_metric": "validation_mean_bin_r2",
        "target": "raw_20_bin_accessibility",
        "backbone": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "backbone_frozen": True,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    active = set(ALL_KEYS)
    best_score = {key: -float("inf") for key in ALL_KEYS}
    best_epoch = {key: 0 for key in ALL_KEYS}
    waits = {key: 0 for key in ALL_KEYS}
    best_states = {}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums = {key: 0.0 for key in ALL_KEYS}
        train_n = 0
        for key in active:
            optimizers[key].zero_grad(set_to_none=True)
        for step, (seqs, target) in enumerate(train_loader, start=1):
            target = target.to(device, non_blocking=True)
            encoded = tokenize_batch(tokenizer, seqs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(encoded["input_ids"], encoded["attention_mask"], sorted(active))
                losses = {key: F.mse_loss(prediction.float(), target) for key, prediction in predictions.items()}
                total_loss = sum(losses.values()) / args.accum_steps
            total_loss.backward()
            update = step % args.accum_steps == 0 or step == len(train_loader)
            if update:
                for key in list(active):
                    torch.nn.utils.clip_grad_norm_(model.candidate_parameters(key), args.clip_grad)
                    optimizers[key].step()
                    optimizers[key].zero_grad(set_to_none=True)
            for key in list(active):
                train_sums[key] += losses[key].item() * target.shape[0]
            train_n += target.shape[0]
            if step % args.log_every == 0:
                message = {key: train_sums[key] / train_n for key in sorted(active)}
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_mse={json.dumps(message)}", flush=True)

        validation = evaluate(model, val_loader, tokenizer, device, sorted(active))
        epoch_record = {"epoch": epoch, "active": sorted(active), "validation": validation}
        history.append(epoch_record)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"EPOCH {epoch} VALIDATION {json.dumps(validation)}", flush=True)

        for key in list(active):
            score = validation[key]["mean_r2"]
            schedulers[key].step(score)
            if score > best_score[key] + args.min_delta:
                best_score[key] = score
                best_epoch[key] = epoch
                waits[key] = 0
                best_states[key] = model.candidate_state(key)
                torch.save(best_states[key], output / f"{key}.pt")
            else:
                waits[key] += 1
                if waits[key] >= args.patience:
                    active.remove(key)
                    print(f"EARLY_STOP {key} epoch={epoch} best_epoch={best_epoch[key]}", flush=True)
        if not active:
            break

    for key in ALL_KEYS:
        if key not in best_states:
            best_states[key] = torch.load(output / f"{key}.pt", map_location="cpu", weights_only=True)
        model.load_candidate_state(key, best_states[key])
    final = evaluate(model, val_loader, tokenizer, device, ALL_KEYS)
    for key in ALL_KEYS:
        final[key]["best_epoch"] = best_epoch[key]
        final[key]["best_validation_mean_r2"] = best_score[key]
    if WEIGHTED_KEY in final:
        final[WEIGHTED_KEY]["layer_weights"] = (
            torch.softmax(model.layer_mix_logits.detach().float(), dim=0).cpu().tolist()
        )
    (output / "metrics.json").write_text(json.dumps(final, indent=2))
    (output / "COMPLETE").write_text("Frozen NT layer-wise probing complete.\n")
    print(f"FINAL {json.dumps(final)}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output-root", default="/root/autodl-tmp/nt_layerwise5")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
