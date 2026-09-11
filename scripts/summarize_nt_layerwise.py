#!/usr/bin/env python3
"""Summarise finalized five-fold frozen-NT layer-probing metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "metrics" / "nt_layerwise"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination; archived inputs are never overwritten",
    )
    args = parser.parse_args()
    metric_files = sorted(args.metrics_dir.glob("fold*/metrics.json"))
    if len(metric_files) != 5:
        raise RuntimeError(f"Expected five fold metrics files, found {len(metric_files)}")
    folds = [json.loads(path.read_text()) for path in metric_files]
    names = [f"layer_{index:02d}" for index in range(13)] + ["weighted_01_12"]
    summary = {}
    print("representation\tmean_r2\tsample_sd\tfold_values")
    for name in names:
        values = [fold[name]["mean_r2"] for fold in folds]
        summary[name] = {
            "mean_r2": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "fold_values": values,
        }
        shown = ",".join(f"{value:.4f}" for value in values)
        print(f"{name}\t{summary[name]['mean_r2']:.4f}\t{summary[name]['sample_sd']:.4f}\t{shown}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
