"""Validate file alignment and all five split boundaries before running models."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from developmental_accessibility.data import overlap_groups
from developmental_accessibility.paths import DATA_ROOT


def validate(root: Path, check_groups: bool = False):
    n, t = map(int, (root / "totmat_shape.txt").read_text().split())
    if n < 2 or t != 20:
        raise ValueError("Expected at least two rows and 20 target bins")
    for filename, size in [("totmat_f64.bin", n * t * 8), ("fullseqs.txt", n * 202)]:
        if (root / filename).stat().st_size != size:
            raise ValueError(
                f"Wrong byte size: {filename}; check row alignment and LF line endings"
            )
    seq = np.memmap(root / "fullseqs.txt", mode="r", dtype=np.uint8, shape=(n, 202))
    target = np.memmap(root / "totmat_f64.bin", mode="r", dtype="<f8", shape=(n, t), order="F")
    for start in range(0, n, 100000):
        if not (seq[start : start + 100000, 201] == 10).all():
            raise ValueError("Sequences must contain exactly 201 bases followed by LF")
        if not np.isfinite(target[start : start + 100000]).all():
            raise ValueError("Nonfinite response values")
    with (root / "folds.pkl").open("rb") as handle:
        folds = pickle.load(handle)
    if isinstance(folds, dict):
        folds = folds["folds"]
    if len(folds) != 5:
        raise ValueError("Expected five development folds")
    splits = []
    for fold in folds:
        if isinstance(fold, dict):
            fold = (fold.get("train_idx", fold.get("train")), fold.get("val_idx", fold.get("val")))
        tr, va = (np.asarray(x) for x in fold)
        for x in [tr, va]:
            if x.ndim != 1 or not np.issubdtype(x.dtype, np.integer) or not len(x):
                raise ValueError("Fold indices must be nonempty integer vectors")
            if x.min() < 0 or x.max() >= n or len(np.unique(x)) != len(x):
                raise ValueError("Out-of-range or duplicate fold indices")
        if np.intersect1d(tr, va).size:
            raise ValueError("Training/validation overlap")
        splits.append((tr, va))
    all_val = np.concatenate([va for _, va in splits])
    dev = np.unique(all_val)
    if len(dev) != len(all_val):
        raise ValueError("Validation folds are not disjoint")
    for tr, va in splits:
        if not np.array_equal(np.union1d(tr, va), dev):
            raise ValueError("Folds do not partition the same development set")
    heldout = np.setdiff1d(np.arange(n), dev)
    if check_groups:
        peaks = pd.read_csv(
            root / "peaks.tsv", sep="\t", usecols=["chr", "median_position"], dtype={"chr": str}
        )
        if len(peaks) != n:
            raise ValueError("Coordinate row count differs from target row count")
        groups = overlap_groups(peaks["chr"].astype(str), peaks["median_position"])
        for tr, va in [*splits, (dev, heldout)]:
            if np.intersect1d(groups[tr], groups[va]).size:
                raise ValueError("Genomic overlap group crosses a split boundary")
    return {
        "regions": n,
        "bins": t,
        "development": len(dev),
        "heldout": len(heldout),
        "group_boundaries_checked": check_groups,
        "heldout_index_sha256": hashlib.sha256(heldout.astype("<i8").tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--check-groups", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root, args.check_groups), indent=2))


if __name__ == "__main__":
    main()
