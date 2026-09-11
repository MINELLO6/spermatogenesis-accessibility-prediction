def main():
    #!/usr/bin/env python3
    """Fold-1 magnitude screen using ONLY the 1443 x 10 motif score grid."""
    import argparse
    import copy
    import json
    import pickle
    import random
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm

    from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=[
            "conv",
            "conv_dual",
            "conv_smooth",
            "conv_nopos",
            "conv_pca3",
            "conv_pca6",
            "conv_ple",
            "conv_wide",
            "convflat",
            "transformer",
            "mlp",
            "conv_log",
        ],
        required=True,
    )
    ap.add_argument("--fold", type=int, default=0, choices=range(5))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--rc-prob", type=float, default=0.0)
    ap.add_argument("--output-root", default=str(RUN_ROOT / "dense_motif_magnitude"))
    a = ap.parse_args()
    if min(a.epochs, a.patience, a.batch_size, a.accum_steps) < 1:
        ap.error("Training counts must be positive")
    R = DATA_ROOT
    O = Path(a.output_root) / a.variant / f"fold{a.fold + 1}"
    if O.exists() and any(O.iterdir()):
        ap.error(f"Output directory is not empty: {O}")
    O.mkdir(parents=True, exist_ok=True)
    NM, NP, NO = 1443, 10, 20
    seed = 7200 + a.fold + a.seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda:0")
    nr, nb = map(int, open(R / "totmat_shape.txt").read().split())
    assert nb == NO
    Y = np.memmap(R / "totmat_f64.bin", dtype="<f8", mode="r", shape=(nr, nb), order="F")
    folds = pickle.load(open(R / "folds.pkl", "rb"))
    tr, va = [np.asarray(x, dtype=np.int64) for x in folds[a.fold]]
    M = R / "motif_csr"
    off = np.load(M / "offsets.npy", mmap_mode="r")
    mid = np.load(M / "motif_ids.npy", mmap_mode="r")
    pos = np.load(M / "position_bins.npy", mmap_mode="r")
    score = np.load(M / "scores.npy", mmap_mode="r")

    def grid(idx):
        idx = np.asarray(idx, dtype=np.int64)
        B = len(idx)
        st = np.asarray(off[idx], dtype=np.int64)
        en = np.asarray(off[idx + 1], dtype=np.int64)
        cnt = en - st
        nh = int(cnt.sum())
        out = np.zeros((B, NP, NM), np.float32)
        if not nh:
            return torch.from_numpy(out)
        cum = np.cumsum(cnt)
        prev = np.r_[0, cum[:-1]]
        row = np.repeat(np.arange(B), cnt)
        hi = np.arange(nh) - np.repeat(prev, cnt) + np.repeat(st, cnt)
        mi = np.asarray(mid[hi], dtype=np.int64) - 1
        po = np.asarray(pos[hi], dtype=np.int64) - 1
        sc = np.asarray(score[hi], dtype=np.float32)
        ok = (mi >= 0) & (mi < NM) & (po >= 0) & (po < NP) & np.isfinite(sc)
        np.maximum.at(out.reshape(B, -1), (row[ok], po[ok] * NM + mi[ok]), sc[ok])
        return torch.from_numpy(out)

    def target_stats(idx, log=False, chunk=200000):
        n = 0
        s = np.zeros(NO)
        ss = np.zeros(NO)
        for j in range(0, len(idx), chunk):
            y = np.asarray(Y[idx[j : j + chunk]], np.float64)
            y = np.log1p(y) if log else y
            n += len(y)
            s += y.sum(0)
            ss += (y * y).sum(0)
        mu = s / n
        sd = np.sqrt(np.maximum(ss / n - mu * mu, 1e-6))
        return mu.astype("f4"), sd.astype("f4")

    use_log = a.variant == "conv_log"
    mu, sd = target_stats(tr, use_log)
    MU = torch.tensor(mu, device=dev)
    SD = torch.tensor(sd, device=dev)
    if a.variant == "conv_dual":
        n_total = 0
        s_total = 0.0
        ss_total = 0.0
        for j in range(0, len(tr), 200000):
            lt = np.log1p(np.asarray(Y[tr[j : j + 200000]], np.float64).sum(1))
            n_total += len(lt)
            s_total += lt.sum()
            ss_total += (lt * lt).sum()
        log_total_mu = s_total / n_total
        log_total_sd = max((ss_total / n_total - log_total_mu**2) ** 0.5, 1e-6)
    else:
        log_total_mu = 0.0
        log_total_sd = 1.0
    LTMU = torch.tensor(log_total_mu, device=dev)
    LTSD = torch.tensor(log_total_sd, device=dev)
    OUTDIM = 3 if a.variant == "conv_pca3" else 6 if a.variant == "conv_pca6" else NO
    if OUTDIM < NO:
        cov = np.zeros((NO, NO), np.float64)
        nobs = 0
        for j in range(0, len(tr), 200000):
            z = (np.asarray(Y[tr[j : j + 200000]], np.float64) - mu) / sd
            cov += z.T @ z
            nobs += len(z)
        ev, Vnp = np.linalg.eigh(cov / nobs)
        order = np.argsort(ev)[::-1]
        ev = ev[order]
        Vnp = Vnp[:, order[:OUTDIM]].astype("f4")
        V = torch.tensor(Vnp, device=dev)
    else:
        ev = np.ones(NO)
        V = None

    def decode(z):
        return z if V is None else z @ V.T

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c1 = nn.Conv1d(128, 128, 3, padding=d, dilation=d)
            self.n1 = nn.BatchNorm1d(128)
            self.c2 = nn.Conv1d(128, 128, 3, padding=d, dilation=d)
            self.n2 = nn.BatchNorm1d(128)

        def forward(self, x):
            return F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x))))) + x)

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Linear(NM, 128, bias=False)
            self.n = nn.BatchNorm1d(128)
            self.b = nn.Sequential(*[Block(x) for x in (1, 2, 4)])
            self.h = nn.Sequential(
                nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, OUTDIM)
            )

        def forward(self, x):
            x = F.gelu(self.n(self.p(x).transpose(1, 2)))
            x = self.b(x)
            return self.h(torch.cat([x.mean(2), x.max(2).values], 1))

    class ConvPLE(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Linear(NM * 6, 128, bias=False)
            self.n = nn.BatchNorm1d(128)
            self.b = nn.Sequential(*[Block(x) for x in (1, 2, 4)])
            self.h = nn.Sequential(
                nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, OUTDIM)
            )

        def forward(self, x):
            x = torch.cat([x] + [F.relu(x - q) for q in (0.25, 0.40, 0.55, 0.70, 0.85)], 2)
            x = F.gelu(self.n(self.p(x).transpose(1, 2)))
            x = self.b(x)
            return self.h(torch.cat([x.mean(2), x.max(2).values], 1))

    class ConvDual(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Linear(NM, 128, bias=False)
            self.n = nn.BatchNorm1d(128)
            self.b = nn.Sequential(*[Block(x) for x in (1, 2, 4)])
            self.h = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1))
            self.direct = nn.Linear(256, NO)
            self.total = nn.Linear(256, 1)
            self.profile = nn.Linear(256, NO)

        def forward(self, x):
            x = F.gelu(self.n(self.p(x).transpose(1, 2)))
            x = self.b(x)
            h = self.h(torch.cat([x.mean(2), x.max(2).values], 1))
            return self.direct(h), self.total(h).squeeze(1), self.profile(h)

    class WideBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c1 = nn.Conv1d(256, 256, 3, padding=d, dilation=d)
            self.n1 = nn.BatchNorm1d(256)
            self.c2 = nn.Conv1d(256, 256, 3, padding=d, dilation=d)
            self.n2 = nn.BatchNorm1d(256)

        def forward(self, x):
            return F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x))))) + x)

    class ConvWide(nn.Module):
        def __init__(self, flat=False):
            super().__init__()
            self.flat = flat
            self.p = nn.Linear(NM, 256, bias=False)
            self.n = nn.BatchNorm1d(256)
            self.b = nn.Sequential(*[WideBlock(x) for x in (1, 2, 4, 8)])
            din = NP * 256 if flat else 512
            self.h = nn.Sequential(
                nn.Linear(din, 512),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, OUTDIM),
            )

        def forward(self, x):
            x = self.b(F.gelu(self.n(self.p(x).transpose(1, 2))))
            z = x.flatten(1) if self.flat else torch.cat([x.mean(2), x.max(2).values], 1)
            return self.h(z)

    class Tx(nn.Module):
        def __init__(self):
            super().__init__()
            d = 256
            self.p = nn.Linear(NM, d, bias=False)
            self.pos = nn.Parameter(torch.randn(1, NP, d) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d, 8, 1024, 0.1, "gelu", batch_first=True, norm_first=True
            )
            self.t = nn.TransformerEncoder(layer, 3, nn.LayerNorm(d))
            self.h = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, OUTDIM))

        def forward(self, x):
            x = self.t(self.p(x) + self.pos)
            return self.h(torch.cat([x.mean(1), x.max(1).values], 1))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Linear(NM, 128, bias=False)
            self.h = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(NP * 128),
                nn.Linear(NP * 128, 512),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, OUTDIM),
            )

        def forward(self, x):
            return self.h(self.p(x))

    class NoPos(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.Sequential(
                nn.Linear(NM, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, OUTDIM),
            )

        def forward(self, x):
            return self.h(x.amax(1))

    ctors = {
        "conv": Conv,
        "conv_dual": ConvDual,
        "conv_smooth": Conv,
        "conv_nopos": NoPos,
        "conv_pca3": Conv,
        "conv_pca6": Conv,
        "conv_ple": ConvPLE,
        "conv_log": Conv,
        "conv_wide": lambda: ConvWide(False),
        "convflat": lambda: ConvWide(True),
        "transformer": Tx,
        "mlp": MLP,
    }
    net = ctors[a.variant]().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def ybatch(ii):
        y = torch.as_tensor(np.asarray(Y[ii], np.float32), device=dev)
        yt = torch.log1p(y) if use_log else y
        return y, (yt - MU) / SD

    @torch.no_grad()
    def evaluate(idx, save=False):
        net.eval()
        sy = np.zeros(NO)
        sy2 = np.zeros(NO)
        se = np.zeros(NO)
        n = 0
        preds = []
        ids = []
        for j in tqdm(range(0, len(idx), a.batch_size), desc="eval"):
            ii = idx[j : j + a.batch_size]
            x = grid(ii).to(dev)
            y, _ = ybatch(ii)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = net(x)
                z = decode(raw[0] if a.variant == "conv_dual" else raw)
            ph = z.float() * SD + MU
            ph = torch.expm1(ph).clamp_min(0) if use_log else ph
            yy = y.cpu().numpy()
            pp = ph.cpu().numpy()
            n += len(ii)
            sy += yy.sum(0)
            sy2 += (yy * yy).sum(0)
            se += ((yy - pp) ** 2).sum(0)
            if save:
                preds.append(pp.astype("f4"))
                ids.append(ii)
        r2 = 1 - se / np.maximum(sy2 - sy * sy / n, 1e-9)
        m = {
            "mean_r2": float(r2.mean()),
            "r2_bins": r2.tolist(),
            "rmse": float(np.sqrt(se.sum() / (n * NO))),
            "n": n,
        }
        if save:
            np.savez_compressed(
                O / "val_predictions.npz", indices=np.concatenate(ids), pred=np.vstack(preds)
            )
        return m

    cfg = vars(a) | {
        "seed": seed,
        "train_n": len(tr),
        "val_n": len(va),
        "target_mean": mu.tolist(),
        "target_std": sd.tolist(),
        "log_total_mean": log_total_mu,
        "log_total_std": log_total_sd,
        "output_rank": OUTDIM,
        "pca_explained_fraction": float(ev[:OUTDIM].sum() / ev.sum()) if OUTDIM < NO else 1.0,
        "parameters": sum(p.numel() for p in net.parameters()),
        "input": "motif_score_only_1443x10",
    }
    (O / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg))
    best = -1e9
    state = None
    be = 0
    wait = 0
    for ep in range(1, a.epochs + 1):
        net.train()
        order = np.random.permutation(tr)
        opt.zero_grad(set_to_none=True)
        bar = tqdm(range(0, len(order), a.batch_size), desc=f"epoch {ep}")
        for step, j in enumerate(bar, 1):
            ii = order[j : j + a.batch_size]
            x = grid(ii).to(dev)
            if a.rc_prob > 0:
                mask = torch.rand(len(x), device=dev) < a.rc_prob
                if mask.any():
                    x[mask] = torch.flip(x[mask], dims=[1])
            y, yt = ybatch(ii)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = net(x)
                pred = raw[0] if a.variant == "conv_dual" else raw
                loss = F.mse_loss(decode(pred), yt)
                if a.variant == "conv_smooth":
                    residual = pred - yt
                    loss = loss + 0.05 * F.mse_loss(
                        residual[:, 1:] - residual[:, :-1], torch.zeros_like(residual[:, 1:])
                    )
                if a.variant == "conv_dual":
                    total = y.sum(1)
                    lt = (torch.log1p(total) - LTMU) / LTSD
                    loss = loss + 0.03 * F.mse_loss(raw[1], lt)
                    nz = total > 0
                    if nz.any():
                        loss = loss + 0.02 * (
                            -((y[nz] / total[nz, None]) * F.log_softmax(raw[2][nz], 1))
                            .sum(1)
                            .mean()
                        )
                loss = loss / a.accum_steps
            scaler.scale(loss).backward()
            if step % a.accum_steps == 0 or j + a.batch_size >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        met = evaluate(va)
        print("VAL", ep, json.dumps(met))
        if met["mean_r2"] > best + 1e-5:
            best = met["mean_r2"]
            state = copy.deepcopy(net.state_dict())
            be = ep
            wait = 0
        else:
            wait += 1
        if wait >= a.patience:
            break
    net.load_state_dict(state)
    torch.save({"state_dict": state, "config": cfg, "best_epoch": be}, O / "model.pt")
    met = evaluate(va, True) | {"best_epoch": be}
    (O / "metrics.json").write_text(json.dumps(met, indent=2))
    print("FINAL", json.dumps(met))


if __name__ == "__main__":
    main()
