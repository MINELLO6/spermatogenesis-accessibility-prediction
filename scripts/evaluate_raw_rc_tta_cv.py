#!/usr/bin/env python3
"""Development-fold evaluation of reverse-complement test-time averaging.

The checkpoints were trained with 50% random reverse-complement augmentation.
For one grouped validation fold, this script compares predictions from the stored
orientation, its reverse complement, and their equal-weight prediction mean.
No held-out observations are loaded.
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
OUTPUT = Path("/root/autodl-tmp/rc_tta_cv")


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


def metrics(sse: np.ndarray, sst: np.ndarray, n: int) -> dict:
    r2_bins = 1.0 - sse / sst
    mse = float(sse.sum() / (n * len(sse)))
    return {
        "mean_r2": float(r2_bins.mean()),
        "r2_bins": r2_bins.tolist(),
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    output = OUTPUT / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "metrics.json"
    if result_path.exists():
        raise SystemExit(f"Refusing to overwrite {result_path}")

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
    with open(ROOT / "folds.pkl", "rb") as handle:
        folds = pickle.load(handle)
    validation_indices = np.asarray(folds[args.fold - 1][1], dtype=np.int64)

    def encode(indices: np.ndarray) -> torch.Tensor:
        raw = np.ascontiguousarray(np.asarray(sequences[indices]))
        bases = raw.view(np.uint8).reshape(len(indices), 202)[:, :201]
        array = np.stack(
            (bases == ord("A"), bases == ord("C"), bases == ord("G"), bases == ord("T")),
            axis=1,
        )
        return torch.from_numpy(array.astype(np.float32, copy=False))

    device = torch.device("cuda:0")
    checkpoint = FINAL / "github_weights" / "raw_rc" / f"raw_rc_fold{args.fold}.pt"
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    sse_forward = np.zeros(n_outputs, dtype=np.float64)
    sse_reverse = np.zeros(n_outputs, dtype=np.float64)
    sse_average = np.zeros(n_outputs, dtype=np.float64)
    sum_y = np.zeros(n_outputs, dtype=np.float64)
    sum_y2 = np.zeros(n_outputs, dtype=np.float64)
    direction_abs_sum = 0.0
    direction_sq_sum = 0.0

    with torch.inference_mode():
        for start in range(0, len(validation_indices), args.batch_size):
            indices = validation_indices[start : start + args.batch_size]
            x = encode(indices).to(device, non_blocking=True)
            x_rc = torch.flip(x[:, (3, 2, 1, 0), :], dims=(2,))
            with torch.autocast("cuda", dtype=torch.float16):
                pred_forward = model(x).float()
                pred_reverse = model(x_rc).float()
            pred_average = 0.5 * (pred_forward + pred_reverse)

            pf = pred_forward.cpu().numpy().astype(np.float64, copy=False)
            pr = pred_reverse.cpu().numpy().astype(np.float64, copy=False)
            pa = pred_average.cpu().numpy().astype(np.float64, copy=False)
            y = np.asarray(targets[indices], dtype=np.float64)

            sse_forward += np.square(pf - y).sum(axis=0)
            sse_reverse += np.square(pr - y).sum(axis=0)
            sse_average += np.square(pa - y).sum(axis=0)
            sum_y += y.sum(axis=0)
            sum_y2 += np.square(y).sum(axis=0)
            difference = pf - pr
            direction_abs_sum += np.abs(difference).sum()
            direction_sq_sum += np.square(difference).sum()

            if start % (args.batch_size * 40) == 0:
                print(f"fold={args.fold} processed={start:,}/{len(validation_indices):,}", flush=True)

    sst = sum_y2 - np.square(sum_y) / len(validation_indices)
    forward = metrics(sse_forward, sst, len(validation_indices))
    reverse = metrics(sse_reverse, sst, len(validation_indices))
    average = metrics(sse_average, sst, len(validation_indices))
    n_values = len(validation_indices) * n_outputs
    result = {
        "status": "complete",
        "scope": "grouped_development_validation_only",
        "heldout_accessed": False,
        "fold": args.fold,
        "n_regions": int(len(validation_indices)),
        "checkpoint": str(checkpoint),
        "training_rc_probability": 0.5,
        "forward": forward,
        "reverse_complement": reverse,
        "forward_reverse_average": average,
        "delta_mean_r2_average_minus_forward": average["mean_r2"] - forward["mean_r2"],
        "delta_rmse_average_minus_forward": average["rmse"] - forward["rmse"],
        "forward_reverse_mean_absolute_difference": direction_abs_sum / n_values,
        "forward_reverse_rmse": math.sqrt(direction_sq_sum / n_values),
    }
    result_path.write_text(json.dumps(result, indent=2))
    (output / "COMPLETE").write_text("Development-fold RC averaging evaluation complete.\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
