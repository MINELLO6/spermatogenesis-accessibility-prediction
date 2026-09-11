def main():
    import argparse

    argparse.ArgumentParser(description="bootstrap heldout ensembles").parse_args()
    #!/usr/bin/env python3
    import json
    from pathlib import Path

    import numpy as np

    from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

    ROOT = Path(str(DATA_ROOT))
    FINAL = Path(str(RUN_ROOT / "final_analysis"))
    OUT = FINAL / "bootstrap"
    OUT.mkdir(parents=True, exist_ok=True)
    models = {
        "raw_rc": FINAL / "heldout_ensembles/raw_rc_5fold_ensemble/heldout_predictions_f32.npy",
        "raw_no_rc": FINAL
        / "heldout_ensembles/raw_no_rc_5fold_ensemble/heldout_predictions_f32.npy",
        "frozen_nt": FINAL
        / "heldout_ensembles/frozen_nt_attnpool_5fold_ensemble/heldout_predictions_f32.npy",
        "motif": FINAL
        / "heldout_ensembles/motif_transformer_5fold_ensemble/heldout_predictions_f32.npy",
    }
    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    heldout = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    all_groups = np.load(FINAL / "region_groups.npy", mmap_mode="r")
    group_ids, inverse = np.unique(np.asarray(all_groups[heldout]), return_inverse=True)
    G = len(group_ids)
    with open(ROOT / "totmat_shape.txt") as h:
        n, t = map(int, h.read().split())
    targets = np.memmap(ROOT / "totmat_f64.bin", dtype="<f8", mode="r", shape=(n, t), order="F")
    preds = {k: np.load(v, mmap_mode="r") for k, v in models.items()}

    ng = np.zeros(G, np.float64)
    sy = np.zeros((G, 20), np.float64)
    sy2 = np.zeros((G, 20), np.float64)
    sse = {k: np.zeros((G, 20), np.float64) for k in models}
    chunk = 100000
    for start in range(0, len(heldout), chunk):
        sl = slice(start, min(start + chunk, len(heldout)))
        inv = inverse[sl]
        y = np.asarray(targets[heldout[sl]], dtype=np.float64)
        np.add.at(ng, inv, 1)
        np.add.at(sy, inv, y)
        np.add.at(sy2, inv, y * y)
        for k, p in preds.items():
            d = np.asarray(p[sl], dtype=np.float64) - y
            np.add.at(sse[k], inv, d * d)
        print("aggregate", sl.stop, "/", len(heldout), flush=True)

    names = list(models)
    stats = np.concatenate([ng[:, None], sy, sy2] + [sse[k] for k in names], axis=1)
    B = 20
    R = 2000
    rng = np.random.default_rng(20260829)
    values = {k: np.empty(R, np.float64) for k in names}
    rmse = {k: np.empty(R, np.float64) for k in names}
    for start in range(0, R, B):
        b = min(B, R - start)
        w = rng.poisson(1.0, size=(b, G)).astype(np.float32)
        z = w @ stats
        nrep = z[:, 0]
        sumy = z[:, 1:21]
        sumy2 = z[:, 21:41]
        sst = sumy2 - sumy * sumy / nrep[:, None]
        offset = 41
        for j, k in enumerate(names):
            es = z[:, offset + 20 * j : offset + 20 * (j + 1)]
            r2 = 1 - es / sst
            values[k][start : start + b] = r2.mean(1)
            rmse[k][start : start + b] = np.sqrt(es.sum(1) / (nrep * 20))
        print("bootstrap", start + b, "/", R, flush=True)

    np.savez(
        OUT / "poisson_group_bootstrap_2000.npz",
        **{f"r2_{k}": v for k, v in values.items()},
        **{f"rmse_{k}": v for k, v in rmse.items()},
    )
    summary = {
        "method": "Poisson(1) cluster multiplier bootstrap",
        "unit": "200-bp genomic overlap group",
        "replicates": R,
        "seed": 20260829,
        "n_groups": int(G),
        "comparisons": {},
    }
    for other in names[1:]:
        d = values["raw_rc"] - values[other]
        summary["comparisons"][f"raw_rc_minus_{other}"] = {
            "mean_difference": float(d.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "probability_positive": float(np.mean(d > 0)),
        }
    with open(OUT / "summary.json", "w") as h:
        json.dump(summary, h, indent=2)
    with open(OUT / "LOCKED_COMPLETE", "w") as h:
        h.write("2000-replicate grouped bootstrap complete.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
