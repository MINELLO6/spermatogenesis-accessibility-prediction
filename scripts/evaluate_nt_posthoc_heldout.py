#!/usr/bin/env python3
"""One-time post-hoc held-out evaluation for weighted-layer and IA3 NT models.

The models and checkpoints are fixed by grouped development-fold validation.
This script performs inference only and records that the comparison is post hoc.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

SOURCE = Path(str(DATA_ROOT))
WORK = Path(str(RUN_ROOT))
FINAL = WORK / "final_analysis"
OUTPUT = WORK / "nt_posthoc_heldout"


from scripts import train_nt_ia3 as ia3
from scripts.training import train_nt_heads as base


class WeightedCandidate(nn.Module):
    def __init__(self, state: dict):
        super().__init__()
        self.head = base.AttentionPoolingHead(input_dim=512, hidden_dim=256, dropout=0.1)
        self.head.load_state_dict(state["head"])
        self.register_buffer("layer_mix_logits", state["layer_mix_logits"].float())

    def forward(self, hidden_states, valid_mask):
        weights = torch.softmax(self.layer_mix_logits, dim=0)
        representation = sum(weights[index] * hidden_states[index + 1] for index in range(12))
        return self.head(representation, valid_mask)


def heldout_loader(batch_size: int):
    indices = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    return indices, DataLoader(
        base.IndexDataset(indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=base.SequenceCollator(),
    )


def metrics_from_prediction(prediction_path: Path):
    indices = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    prediction = np.load(prediction_path, mmap_mode="r")
    with open(base.TOTMAT_SHAPE) as handle:
        n_regions, n_bins = map(int, handle.read().split())
    targets = np.memmap(
        base.TOTMAT_BIN, dtype="<f8", mode="r", shape=(n_regions, n_bins), order="F"
    )
    sse = np.zeros(20, dtype=np.float64)
    sum_y = np.zeros(20, dtype=np.float64)
    sum_y2 = np.zeros(20, dtype=np.float64)
    for start in range(0, len(indices), 100_000):
        selected = indices[start : start + 100_000]
        truth = np.asarray(targets[selected], dtype=np.float64)
        pred = np.asarray(prediction[start : start + len(selected)], dtype=np.float64)
        sse += np.square(pred - truth).sum(0)
        sum_y += truth.sum(0)
        sum_y2 += np.square(truth).sum(0)
    sst = sum_y2 - np.square(sum_y) / len(indices)
    r2_bins = 1.0 - sse / np.maximum(sst, 1e-12)
    mse = sse.sum() / (len(indices) * 20)
    return {
        "n": int(len(indices)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mean_r2": float(r2_bins.mean()),
        "r2_bins": r2_bins.tolist(),
    }


def evaluate_weighted(batch_size: int):
    output = OUTPUT / "weighted_layers_5fold_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists():
        print((output / "metrics.json").read_text())
        return
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        base.MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    backbone = (
        AutoModelForMaskedLM.from_pretrained(
            base.MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        .to(device)
        .eval()
    )
    candidates = []
    for fold in range(1, 6):
        state = torch.load(
            WORK / "nt_layerwise5_v2" / f"fold{fold}" / "weighted_01_12.pt",
            map_location="cpu",
            weights_only=True,
        )
        candidates.append(WeightedCandidate(state).to(device).eval())
    indices, loader = heldout_loader(batch_size)
    prediction_path = output / "heldout_predictions_f32.npy"
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(indices), 20)
    )
    offset = 0
    with torch.inference_mode():
        for step, (sequences, _) in enumerate(loader, 1):
            encoded = base.tokenize_batch(tokenizer, sequences, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = backbone(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                valid = encoded["attention_mask"].bool().clone()
                valid[:, 0] = False
                batch_prediction = (
                    sum(candidate(result.hidden_states, valid).float() for candidate in candidates)
                    / 5.0
                )
            values = batch_prediction.cpu().numpy()
            predictions[offset : offset + len(values)] = values
            offset += len(values)
            if step % 100 == 0:
                print(f"weighted step={step}/{len(loader)} regions={offset}", flush=True)
    predictions.flush()
    metrics = metrics_from_prediction(prediction_path)
    metrics.update(
        {
            "name": "frozen_nt_weighted_layers_5fold_ensemble",
            "aggregation": "equal prediction mean",
            "post_hoc_heldout_evaluation": True,
            "model_selected_by": "grouped_development_folds_only",
            "heldout_evaluations_for_this_model": 1,
        }
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "COMPLETE").write_text("Post-hoc held-out weighted-layer ensemble complete.\n")
    print(json.dumps(metrics, indent=2), flush=True)


def evaluate_ia3_fold(fold: int, batch_size: int):
    output = OUTPUT / "ia3_fold_predictions" / f"fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists():
        print(f"IA3 fold {fold} already complete")
        return
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        base.MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    nt = AutoModelForMaskedLM.from_pretrained(
        base.MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    model = ia3.IA3Regressor(nt).to(device)
    state = torch.load(
        WORK / "nt_ia3" / f"fold{fold}" / "ia3_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_adapter(state)
    model.eval()
    indices, loader = heldout_loader(batch_size)
    prediction_path = output / "heldout_predictions_f32.npy"
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(indices), 20)
    )
    offset = 0
    with torch.inference_mode():
        for step, (sequences, _) in enumerate(loader, 1):
            encoded = base.tokenize_batch(tokenizer, sequences, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_prediction = model.encoder_forward(
                    encoded["input_ids"], encoded["attention_mask"]
                ).float()
            values = batch_prediction.cpu().numpy()
            predictions[offset : offset + len(values)] = values
            offset += len(values)
            if step % 100 == 0:
                print(f"ia3 fold={fold} step={step}/{len(loader)} regions={offset}", flush=True)
    predictions.flush()
    fold_metrics = metrics_from_prediction(prediction_path)
    fold_metrics.update(
        {
            "name": f"ia3_fold{fold}",
            "post_hoc_heldout_evaluation": True,
            "heldout_evaluations_for_this_model": 1,
        }
    )
    (output / "metrics.json").write_text(json.dumps(fold_metrics, indent=2) + "\n")
    (output / "COMPLETE").write_text("Post-hoc held-out IA3 fold prediction complete.\n")
    print(json.dumps(fold_metrics, indent=2), flush=True)


def aggregate_ia3():
    output = OUTPUT / "ia3_5fold_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        OUTPUT / "ia3_fold_predictions" / f"fold{fold}" / "heldout_predictions_f32.npy"
        for fold in range(1, 6)
    ]
    markers = [path.parent / "COMPLETE" for path in paths]
    if not all(path.is_file() and marker.is_file() for path, marker in zip(paths, markers)):
        raise SystemExit("Not all IA3 fold predictions are complete")
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    ensemble_path = output / "heldout_predictions_f32.npy"
    ensemble = np.lib.format.open_memmap(
        ensemble_path, mode="w+", dtype=np.float32, shape=arrays[0].shape
    )
    for start in range(0, arrays[0].shape[0], 100_000):
        stop = min(start + 100_000, arrays[0].shape[0])
        ensemble[start:stop] = (
            sum(np.asarray(array[start:stop], dtype=np.float64) for array in arrays) / 5.0
        )
    ensemble.flush()
    metrics = metrics_from_prediction(ensemble_path)
    metrics.update(
        {
            "name": "nt_ia3_final_layer_5fold_ensemble",
            "aggregation": "equal prediction mean",
            "post_hoc_heldout_evaluation": True,
            "model_selected_by": "grouped_development_folds_only",
            "heldout_evaluations_for_this_model": 1,
        }
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "COMPLETE").write_text("Post-hoc held-out IA3 ensemble complete.\n")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["weighted", "ia3-fold", "aggregate-ia3"], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.mode == "weighted":
        evaluate_weighted(args.batch_size)
    elif args.mode == "ia3-fold":
        if args.fold is None:
            parser.error("--fold is required for ia3-fold")
        evaluate_ia3_fold(args.fold, args.batch_size)
    else:
        aggregate_ia3()


if __name__ == "__main__":
    main()
