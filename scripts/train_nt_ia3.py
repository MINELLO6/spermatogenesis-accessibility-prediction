#!/usr/bin/env python3
"""IA3 parameter-efficient fine-tuning for NT-v2-50M.

The pretrained weights remain frozen.  Learned multiplicative vectors rescale
the key, value and feed-forward activations in every transformer block, matching
the IA3 formulation used by the Nucleotide Transformer study.  Only development
folds are read; the locked held-out set is never accessed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

BASE_DIR = Path(str(DATA_ROOT))
from scripts.training import train_nt_heads as base


class OutputScale(nn.Module):
    def __init__(self, module: nn.Module, width: int):
        super().__init__()
        self.module = module
        self.ia3_scale = nn.Parameter(torch.ones(width))

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output * self.ia3_scale.to(output.dtype)


def add_ia3(nt: nn.Module):
    for parameter in nt.parameters():
        parameter.requires_grad = False
    layers = nt.esm.encoder.layer
    for layer in layers:
        layer.attention.self.key = OutputScale(layer.attention.self.key, 512)
        layer.attention.self.value = OutputScale(layer.attention.self.value, 512)
        # EsmIntermediate applies the v2 GLU internally and returns 2048 channels.
        layer.intermediate = OutputScale(layer.intermediate, 2048)
    return nt


class IA3Regressor(nn.Module):
    def __init__(self, nt: nn.Module):
        super().__init__()
        self.nt = add_ia3(nt)
        self.head = base.AttentionPoolingHead()

    def load_adapter(self, state):
        """Require every learned scale and head weight, while retaining the frozen backbone."""
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        missing = expected - state.keys()
        unexpected = state.keys() - expected
        if missing or unexpected:
            raise ValueError(
                f"Invalid adapter checkpoint; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return self.load_state_dict(state, strict=False)

    def forward(self, input_ids, attention_mask):
        return self.encoder_forward(input_ids, attention_mask)

    def encoder_forward(self, input_ids, attention_mask):
        output = self.nt.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        valid = attention_mask.bool().clone()
        valid[:, 0] = False
        return self.head(output.last_hidden_state, valid)


def make_loader(indices, batch_size: int, shuffle: bool):
    return DataLoader(
        base.IndexDataset(indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=base.SequenceCollator(),
    )


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    n = 0
    sse = np.zeros(20, np.float64)
    sy = np.zeros(20, np.float64)
    sy2 = np.zeros(20, np.float64)
    for seqs, y_cpu in data_loader:
        y = y_cpu.to(device, non_blocking=True)
        encoded = base.tokenize_batch(tokenizer, seqs, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.encoder_forward(encoded["input_ids"], encoded["attention_mask"])
        truth = y.float()
        sse += (prediction.float() - truth).square().sum(0).cpu().numpy()
        sy += truth.sum(0).cpu().numpy()
        sy2 += truth.square().sum(0).cpu().numpy()
        n += len(y)
    sst = sy2 - np.square(sy) / n
    r2 = 1.0 - sse / np.maximum(sst, 1e-12)
    mse = sse.sum() / (n * 20)
    return {
        "n": n,
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mean_r2": float(r2.mean()),
        "r2_bins": r2.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--output-root", default=str(RUN_ROOT / "nt_ia3"))
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.accum_steps) < 1:
        parser.error("epochs, patience, batch-size and accum-steps must be positive")

    base.set_seed(9300 + args.fold)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    output = Path(args.output_root) / f"fold{args.fold + 1}"
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}; choose a new --output-root")
    output.mkdir(parents=True, exist_ok=True)
    train_idx, val_idx = base.load_folds(base.FOLDS_FILE, args.fold)
    if args.max_train:
        train_idx = train_idx[: args.max_train]
    if args.max_val:
        val_idx = val_idx[: args.max_val]
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Training and validation splits must both be nonempty")

    tokenizer = AutoTokenizer.from_pretrained(
        base.MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    nt = AutoModelForMaskedLM.from_pretrained(
        base.MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    model = IA3Regressor(nt).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    train_loader = make_loader(train_idx, args.batch_size, True)
    val_loader = make_loader(val_idx, args.batch_size, False)

    config = {
        **vars(args),
        "seed": 9300 + args.fold,
        "train_n": len(train_idx),
        "val_n": len(val_idx),
        "method": "IA3_key_value_feedforward_scaling",
        "backbone_original_weights_frozen": True,
        "head": "final_layer_attention_pooling",
        "trainable_parameters": sum(p.numel() for p in trainable),
        "effective_batch_size": args.batch_size * args.accum_steps,
        "selection_metric": "development_fold_mean_bin_r2",
        "heldout_accessed": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    best = -float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen = 0
        for step, (seqs, y_cpu) in enumerate(train_loader, 1):
            y = y_cpu.to(device, non_blocking=True)
            encoded = base.tokenize_batch(tokenizer, seqs, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.encoder_forward(encoded["input_ids"], encoded["attention_mask"])
                raw_loss = F.mse_loss(prediction.float(), y)
                loss = raw_loss / args.accum_steps
            loss.backward()
            total_loss += raw_loss.item() * len(y)
            seen += len(y)
            if step % args.accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 1000 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} mse={total_loss / seen:.3f}",
                    flush=True,
                )
        metrics = evaluate(model, val_loader, tokenizer, device)
        history.append({"epoch": epoch, "train_mse": total_loss / seen, **metrics})
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print("EPOCH", epoch, json.dumps(metrics), flush=True)
        if metrics["mean_r2"] > best + 1e-5:
            best = metrics["mean_r2"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if "ia3_scale" in key or key.startswith("head.")
            }
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    final = {
        "fold": args.fold + 1,
        "best_mean_r2": best,
        "best_epoch": best_epoch,
        "heldout_accessed": False,
    }
    torch.save(best_state, output / "ia3_best.pt")
    (output / "metrics.json").write_text(json.dumps(final, indent=2))
    (output / "COMPLETE").write_text("NT IA3 development-fold training complete.\n")
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
