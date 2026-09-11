def main():
    import argparse

    from developmental_accessibility.models import MotifTransformer

    argparse.ArgumentParser(description="evaluate motif ensemble").parse_args()
    #!/usr/bin/env python3
    import json
    import math
    from pathlib import Path

    import numpy as np
    import torch

    from developmental_accessibility.paths import (
        DATA_ROOT,
        RUN_ROOT,
        WEIGHTS_ROOT,
    )

    ROOT = Path(str(DATA_ROOT))
    FINAL = Path(str(RUN_ROOT / "final_analysis"))
    OUT = FINAL / "heldout_ensembles/motif_transformer_5fold_ensemble"
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "LOCKED_COMPLETE").exists():
        raise SystemExit("Already complete")
    with open(ROOT / "totmat_shape.txt") as h:
        n, t = map(int, h.read().split())
    targets = np.memmap(ROOT / "totmat_f64.bin", dtype="<f8", mode="r", shape=(n, t), order="F")
    heldout = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    CSR = ROOT / "motif_csr"
    offsets = np.load(CSR / "offsets.npy", mmap_mode="r")
    mids = np.load(CSR / "motif_ids.npy", mmap_mode="r")
    pbin = np.load(CSR / "position_bins.npy", mmap_mode="r")
    scores = np.load(CSR / "scores.npy", mmap_mode="r")

    def motif_batch(ids, max_tokens=168):
        ids = np.asarray(ids, dtype=np.int64)
        counts = offsets[ids + 1] - offsets[ids]
        L = int(min(max_tokens, max(1, counts.max())))
        B = len(ids)
        mo = np.zeros((B, L), np.int64)
        po = np.zeros((B, L), np.int64)
        sc = np.zeros((B, L), np.float32)
        ma = np.zeros((B, L), bool)
        for i, rid in enumerate(ids):
            a = int(offsets[rid])
            b = int(offsets[rid + 1])
            k = b - a
            if k == 0:
                continue
            if k <= L:
                take = slice(a, b)
            else:
                local = np.asarray(scores[a:b], dtype=np.float32)
                sel = np.argpartition(local, -L)[-L:]
                take = a + sel
                k = L
            mo[i, :k] = mids[take]
            po[i, :k] = pbin[take]
            sc[i, :k] = scores[take]
            ma[i, :k] = True
        return tuple(torch.from_numpy(x) for x in (mo, po, sc, ma))

    device = torch.device("cuda:0")
    nets = []
    for fold in range(1, 6):
        net = MotifTransformer().to(device)
        net.load_state_dict(
            torch.load(
                WEIGHTS_ROOT / "motif" / f"motif_transformer_fold{fold}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        net.eval()
        nets.append(net)
    pred = np.lib.format.open_memmap(
        OUT / "heldout_predictions_f32.npy", mode="w+", dtype=np.float32, shape=(len(heldout), 20)
    )
    batch = 1024
    with torch.inference_mode():
        for start in range(0, len(heldout), batch):
            idx = heldout[start : start + batch]
            mo, po, sc, ma = [x.to(device, non_blocking=True) for x in motif_batch(idx)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = sum(net(mo, po, sc, ma).float() for net in nets) / 5
            pred[start : start + len(idx)] = p.cpu().numpy()
            if start % (batch * 50) == 0:
                print(start, "/", len(heldout), flush=True)
    pred.flush()
    sse = np.zeros(20)
    sy = np.zeros(20)
    sy2 = np.zeros(20)
    for start in range(0, len(heldout), 100000):
        idx = heldout[start : start + 100000]
        y = np.asarray(targets[idx], dtype=np.float64)
        p = np.asarray(pred[start : start + len(idx)], dtype=np.float64)
        sse += np.square(p - y).sum(0)
        sy += y.sum(0)
        sy2 += np.square(y).sum(0)
    sst = sy2 - sy**2 / len(heldout)
    r2 = 1 - sse / sst
    mse = sse.sum() / (len(heldout) * 20)
    m = {
        "name": "motif_transformer_5fold_ensemble",
        "folds": 5,
        "aggregation": "equal prediction mean",
        "heldout_evaluations": 1,
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mean_r2": float(r2.mean()),
        "r2_bins": r2.tolist(),
    }
    with open(OUT / "metrics.json", "w") as h:
        json.dump(m, h, indent=2)
    with open(OUT / "LOCKED_COMPLETE", "w") as h:
        h.write("Five-fold heldout ensemble complete.\n")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
