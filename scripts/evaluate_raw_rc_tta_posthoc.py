#!/usr/bin/env python3
"""Post-hoc held-out reverse-complement averaging for the five Raw+RC folds.

Use ``--fold`` to generate one reverse-complement prediction array. After all
five folds complete, use ``--aggregate`` to average them, combine that result
with the pre-existing forward-orientation ensemble, and calculate metrics.
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("/root/sc-motif-open/R")
FINAL = Path("/root/autodl-tmp/final_analysis")
OUTPUT = Path("/root/autodl-tmp/rc_tta_posthoc_heldout")


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + residual)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def branch(kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(4, 64, kernel, padding=kernel // 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )

        self.branch7 = branch(7)
        self.branch15 = branch(15)
        self.branch31 = branch(31)
        self.project = nn.Sequential(
            nn.Conv1d(192, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.blocks = nn.Sequential(*(Block(128, d) for d in (1, 2, 4, 8)))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 20))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(
            torch.cat((self.branch7(x), self.branch15(x), self.branch31(x)), dim=1)
        )
        x = self.blocks(x)
        return self.head(torch.cat((x.mean(2), x.max(2).values), dim=1))


def dataset():
    with open(ROOT / "totmat_shape.txt") as handle:
        n_regions, n_outputs = map(int, handle.read().split())
    targets = np.memmap(
        ROOT / "totmat_f64.bin",
        dtype="<f8",
        mode="r",
        shape=(n_regions, n_outputs),
        order="F",
    )
    sequences = np.memmap(ROOT / "fullseqs.txt", dtype="S202", mode="r", shape=(n_regions,))
    heldout = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    return targets, sequences, heldout


def encode(sequences, indices: np.ndarray) -> torch.Tensor:
    raw = np.ascontiguousarray(np.asarray(sequences[indices]))
    bases = raw.view(np.uint8).reshape(len(indices), 202)[:, :201]
    array = np.stack(
        (bases == ord("A"), bases == ord("C"), bases == ord("G"), bases == ord("T")),
        axis=1,
    )
    return torch.from_numpy(array.astype(np.float32, copy=False))


def predict_fold(fold: int, batch_size: int) -> None:
    _, sequences, heldout = dataset()
    fold_dir = OUTPUT / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = fold_dir / "heldout_rc_predictions_f32.npy"
    if prediction_path.exists():
        raise SystemExit(f"Refusing to overwrite {prediction_path}")

    device = torch.device("cuda:0")
    checkpoint = FINAL / "github_weights" / "raw_rc" / f"raw_rc_fold{fold}.pt"
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(heldout), 20)
    )
    with torch.inference_mode():
        for start in range(0, len(heldout), batch_size):
            indices = heldout[start : start + batch_size]
            x = encode(sequences, indices).to(device, non_blocking=True)
            x_rc = torch.flip(x[:, (3, 2, 1, 0), :], dims=(2,))
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(x_rc).float().cpu().numpy()
            predictions[start : start + len(indices)] = prediction
            if start % (batch_size * 40) == 0:
                print(f"fold={fold} processed={start:,}/{len(heldout):,}", flush=True)
    predictions.flush()
    (fold_dir / "COMPLETE").write_text("Held-out reverse-complement prediction complete.\n")


def calculate_metrics(predictions: np.ndarray, targets, indices: np.ndarray) -> dict:
    n_outputs = predictions.shape[1]
    sse = np.zeros(n_outputs, dtype=np.float64)
    sum_y = np.zeros(n_outputs, dtype=np.float64)
    sum_y2 = np.zeros(n_outputs, dtype=np.float64)
    for start in range(0, len(indices), 100_000):
        selected = indices[start : start + 100_000]
        y = np.asarray(targets[selected], dtype=np.float64)
        p = np.asarray(predictions[start : start + len(selected)], dtype=np.float64)
        sse += np.square(p - y).sum(axis=0)
        sum_y += y.sum(axis=0)
        sum_y2 += np.square(y).sum(axis=0)
    sst = sum_y2 - np.square(sum_y) / len(indices)
    r2_bins = 1.0 - sse / sst
    mse = float(sse.sum() / (len(indices) * n_outputs))
    return {
        "mean_r2": float(r2_bins.mean()),
        "r2_bins": r2_bins.tolist(),
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def aggregate() -> None:
    targets, _, heldout = dataset()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fold_paths = [OUTPUT / f"fold{fold}" / "heldout_rc_predictions_f32.npy" for fold in range(1, 6)]
    missing = [str(path) for path in fold_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    forward_path = (
        FINAL / "heldout_ensembles" / "raw_rc_5fold_ensemble" / "heldout_predictions_f32.npy"
    )
    reverse_path = OUTPUT / "reverse_5fold_ensemble_f32.npy"
    average_path = OUTPUT / "forward_reverse_average_5fold_ensemble_f32.npy"
    reverse = np.lib.format.open_memmap(
        reverse_path, mode="w+", dtype=np.float32, shape=(len(heldout), 20)
    )
    average = np.lib.format.open_memmap(
        average_path, mode="w+", dtype=np.float32, shape=(len(heldout), 20)
    )
    fold_arrays = [np.load(path, mmap_mode="r") for path in fold_paths]
    forward = np.load(forward_path, mmap_mode="r")
    direction_abs_sum = 0.0
    direction_sq_sum = 0.0
    for start in range(0, len(heldout), 100_000):
        stop = min(start + 100_000, len(heldout))
        rc = sum(np.asarray(array[start:stop], dtype=np.float64) for array in fold_arrays) / 5.0
        fw = np.asarray(forward[start:stop], dtype=np.float64)
        av = 0.5 * (fw + rc)
        reverse[start:stop] = rc
        average[start:stop] = av
        difference = fw - rc
        direction_abs_sum += np.abs(difference).sum()
        direction_sq_sum += np.square(difference).sum()
    reverse.flush()
    average.flush()

    forward_metrics = calculate_metrics(forward, targets, heldout)
    reverse_metrics = calculate_metrics(reverse, targets, heldout)
    average_metrics = calculate_metrics(average, targets, heldout)
    n_values = len(heldout) * 20
    result = {
        "status": "complete",
        "evaluation_status": "secondary_posthoc",
        "heldout_accessed": True,
        "n_regions": int(len(heldout)),
        "aggregation": "mean of five folds within orientation, then equal mean of forward and reverse orientations",
        "forward": forward_metrics,
        "reverse_complement": reverse_metrics,
        "forward_reverse_average": average_metrics,
        "delta_mean_r2_average_minus_forward": average_metrics["mean_r2"] - forward_metrics["mean_r2"],
        "delta_rmse_average_minus_forward": average_metrics["rmse"] - forward_metrics["rmse"],
        "forward_reverse_mean_absolute_difference": direction_abs_sum / n_values,
        "forward_reverse_rmse": math.sqrt(direction_sq_sum / n_values),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2))
    (OUTPUT / "COMPLETE").write_text("Post-hoc held-out RC averaging evaluation complete.\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, choices=range(1, 6))
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    else:
        predict_fold(args.fold, args.batch_size)


if __name__ == "__main__":
    main()
