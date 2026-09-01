#!/usr/bin/env python3
"""Summarise finalized five-fold frozen-NT layer-probing metrics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "nt_layerwise_metrics"


def main() -> None:
    metric_files = sorted(ROOT.glob("fold*/metrics.json"))
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
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
