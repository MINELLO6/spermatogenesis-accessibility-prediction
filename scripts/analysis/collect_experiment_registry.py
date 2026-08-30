#!/usr/bin/env python3
"""Collect completed motif experiments into machine-readable registries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOTS = [Path("/root/autodl-tmp"), Path("/root/sc-motif-open/R")]
OUT = Path("/root/autodl-tmp/experiment_registry")
OUT.mkdir(parents=True, exist_ok=True)


def scalar(v: Any) -> Any:
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    if isinstance(v, np.generic):
        return v.item()
    return v


def first(d: dict[str, Any], *keys: str):
    for k in keys:
        if k in d and np.ndim(d[k]) == 0:
            return scalar(d[k])
    return ""


def row_for(path: Path, payload: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "source": str(path),
        "kind": kind,
        "experiment": str(path.parent),
        "variant": first(payload, "variant", "name", "model", "objective"),
        "fold": first(payload, "fold"),
        "mean_r2": first(payload, "mean_r2", "mean_bin_r2", "r2"),
        "oof_r2": first(payload, "oof_r2", "oof_mean_r2"),
        "magnitude_r2": first(payload, "magnitude_r2", "scalar_magnitude_r2"),
        "pearson": first(payload, "pearson", "pearson_mean", "mean_pearson"),
        "rmse": first(payload, "rmse", "count_rmse", "magnitude_rmse"),
        "parameters": first(payload, "parameters", "n_parameters"),
        "input": first(payload, "input"),
        "mtime": path.stat().st_mtime,
    }


rows: list[dict[str, Any]] = []
payloads: list[dict[str, Any]] = []
seen: set[str] = set()

for root in ROOTS:
    for path in root.rglob("metrics.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        rows.append(row_for(path, payload, "metrics_json"))
        payloads.append({"source": str(path), "payload": payload})
        seen.add(str(path.resolve()))

for root in ROOTS:
    for path in root.rglob("*.npz"):
        if path.stat().st_size > 2_000_000:
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                payload = {k: scalar(z[k]) for k in z.files if z[k].ndim == 0}
        except Exception:
            continue
        if not payload:
            continue
        rows.append(row_for(path, payload, "small_npz_summary"))
        payloads.append({"source": str(path), "payload": payload})

fields = [
    "source", "kind", "experiment", "variant", "fold", "mean_r2", "oof_r2",
    "magnitude_r2", "pearson", "rmse", "parameters", "input", "mtime"
]
rows.sort(key=lambda r: (r["experiment"], str(r["fold"]), r["source"]))
with open(OUT / "experiments.csv", "w", newline="") as h:
    writer = csv.DictWriter(h, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
with open(OUT / "experiments_full.json", "w") as h:
    json.dump(payloads, h, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

summary = {
    "n_registry_rows": len(rows),
    "n_metrics_json": sum(r["kind"] == "metrics_json" for r in rows),
    "n_small_npz_summaries": sum(r["kind"] == "small_npz_summary" for r in rows),
    "roots": [str(r) for r in ROOTS],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
