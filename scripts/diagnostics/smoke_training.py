"""Run one-epoch GPU integration checks on synthetic, non-biological data.

Each invocation creates a fresh directory under --output-root. It never reads
the dissertation's development or test observations. NT checks additionally
need a cached backbone specified through ACCESSIBILITY_NT_PATH.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-backbone", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="training_smoke_", dir=args.output_root))
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(19)
    n = 40
    targets = rng.uniform(1, 20, size=(n, 20))
    (data / "totmat_shape.txt").write_text(f"{n} 20\n")
    targets.astype("<f8").ravel(order="F").tofile(data / "totmat_f64.bin")
    sequences = ["".join(rng.choice(list("ACGT"), 201)) for _ in range(n)]
    (data / "fullseqs.txt").write_bytes(("\n".join(sequences) + "\n").encode("ascii"))
    indices = np.arange(n)
    folds = [(indices[indices % 5 != k], indices[indices % 5 == k]) for k in range(5)]
    with (data / "folds.pkl").open("wb") as handle:
        pickle.dump(folds, handle)
    (data / "peaktest.tsv").write_text("p.value\n" + "0.0001\n" * n)
    # Original allmot columns: one-based region ID, score, position, motif ID.
    hits = np.array([[i + 1, 0.5, i % 10 + 1, i % 100 + 1] for i in range(1, n)])
    (data / "allmot_shape.txt").write_text(f"{len(hits)} 4\n")
    hits.astype("<f8").ravel(order="F").tofile(data / "allmot_f64.bin")
    env = dict(
        os.environ,
        ACCESSIBILITY_DATA_ROOT=str(data),
        ACCESSIBILITY_RUN_ROOT=str(root / "runs"),
        OMP_NUM_THREADS="2",
    )
    jobs = [
        (
            "raw_no_rc",
            "scripts.training.train_models",
            ["--model", "raw", "--fold", "0", "--epochs", "1"],
        ),
        ("convnext", "scripts.training.train_convnext", ["--fold", "0", "--epochs", "1"]),
        (
            "raw_rc_attn",
            "scripts.training.train_raw_rc_attn",
            ["--fold", "0", "--epochs", "1", "--batch-size", "8"],
        ),
        ("sparse_prepare", "scripts.training.train_sparse_motif", ["--mode", "prepare"]),
        (
            "sparse_train",
            "scripts.training.train_sparse_motif",
            ["--mode", "train", "--model", "transformer", "--epochs", "1"],
        ),
        (
            "raw_rc",
            "scripts.training.train_raw_rc",
            ["--fold", "0", "--epochs", "1", "--batch-size", "8"],
        ),
        (
            "dense",
            "scripts.training.train_dense_motif_magnitude",
            ["--variant", "conv", "--epochs", "1", "--batch-size", "8"],
        ),
        (
            "dense_nopos",
            "scripts.training.train_dense_motif_magnitude",
            ["--variant", "conv_nopos", "--epochs", "1", "--batch-size", "8"],
        ),
        (
            "dense_presence",
            "scripts.training.train_motif_transformer_allregions",
            ["--fold", "0", "--epochs", "1", "--batch-size", "8"],
        ),
    ]
    for modality, backbone in [
        ("raw", "cnn"),
        ("motif", "cnn"),
        ("motif", "transformer"),
        ("motif_nopos", "cnn"),
        ("motif", "linear"),
    ]:
        name = f"shape_{modality}_{backbone}"
        jobs.append(
            (
                name,
                "scripts.training.train_shape_shared_trunk",
                [
                    "--modality",
                    modality,
                    "--backbone",
                    backbone,
                    "--objective",
                    "equal_ce",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--output-root",
                    str(root / name),
                ],
            )
        )
    if args.include_backbone:
        jobs.extend(
            [
                (
                    "frozen_nt",
                    "scripts.training.train_nt_heads",
                    ["--fold", "0", "--epochs", "1", "--batch-size", "8"],
                ),
                (
                    "ia3",
                    "scripts.train_nt_ia3",
                    ["--fold", "0", "--epochs", "1", "--batch-size", "8", "--accum-steps", "1"],
                ),
                (
                    "layer_probe",
                    "scripts.train_nt_layerwise",
                    [
                        "--fold",
                        "0",
                        "--epochs",
                        "1",
                        "--batch-size",
                        "8",
                        "--accum-steps",
                        "1",
                        "--workers",
                        "0",
                    ],
                ),
            ]
        )
    report = []
    for name, module, options in jobs:
        log = root / f"{name}.log"
        with log.open("w") as handle:
            result = subprocess.run(
                [sys.executable, "-m", module, *options],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        report.append({"name": name, "returncode": result.returncode})
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(name, "PASS" if result.returncode == 0 else "FAIL", flush=True)
        if result.returncode:
            print(log.read_text()[-6000:])
            raise SystemExit(1)
    print(f"Reports: {root}")


if __name__ == "__main__":
    main()
