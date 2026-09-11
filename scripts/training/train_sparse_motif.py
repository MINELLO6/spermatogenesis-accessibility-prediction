import argparse
import json
import math
import pickle
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from developmental_accessibility.models import MotifBagMLP, MotifTransformer

# ============================================================
# Paths
# ============================================================
from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

ROOT = Path(str(DATA_ROOT))

ALLMOT_BIN = ROOT / "allmot_f64.bin"
ALLMOT_SHAPE = ROOT / "allmot_shape.txt"

TOTMAT_BIN = ROOT / "totmat_f64.bin"
TOTMAT_SHAPE = ROOT / "totmat_shape.txt"

FOLDS_FILE = ROOT / "folds.pkl"

CSR_DIR = ROOT / "motif_csr"
RESULT_DIR = RUN_ROOT / "results"


# ============================================================
# Utilities
# ============================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_shape(path):
    txt = Path(path).read_text()
    nums = list(map(int, re.findall(r"\d+", txt)))
    return tuple(nums[:2])


def load_folds(path, fold):

    with open(path, "rb") as f:
        obj = pickle.load(f)

    # common structure:
    # [(train_idx, val_idx), ...]
    if isinstance(obj, dict) and "folds" in obj:
        obj = obj["folds"]

    entry = obj[fold]

    if isinstance(entry, dict):
        train_idx = entry.get("train_idx") if "train_idx" in entry else entry.get("train")
        val_idx = entry.get("val_idx") if "val_idx" in entry else entry.get("val")
    else:
        train_idx, val_idx = entry

    return (np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64))


# ============================================================
# PART 1:
# Convert allmot -> CSR
# ============================================================


def prepare_csr(chunk_size=2_000_000, force=False):

    CSR_DIR.mkdir(parents=True, exist_ok=True)
    offsets_path = CSR_DIR / "offsets.npy"
    motif_path = CSR_DIR / "motif_ids.npy"
    pos_path = CSR_DIR / "position_bins.npy"
    score_path = CSR_DIR / "scores.npy"
    stats_path = CSR_DIR / "stats.json"

    if (
        offsets_path.exists()
        and motif_path.exists()
        and pos_path.exists()
        and score_path.exists()
        and not force
    ):
        print("CSR files already exist. Skip preprocessing.")
        return

    n_hits, n_cols = read_shape(ALLMOT_SHAPE)
    n_peaks, n_bins = read_shape(TOTMAT_SHAPE)

    print("allmot:", n_hits, "x", n_cols)
    print("regions:", n_peaks)

    allmot = np.memmap(ALLMOT_BIN, dtype="<f8", mode="r", shape=(n_hits, n_cols), order="F")

    # --------------------------------------------------------
    # PASS 1: count motifs per region
    # --------------------------------------------------------

    counts = np.zeros(n_peaks, dtype=np.int64)

    print("\nPASS 1: counting motifs per region")

    for start in range(0, n_hits, chunk_size):
        end = min(start + chunk_size, n_hits)

        peak_id = np.asarray(allmot[start:end, 0], dtype=np.int64) - 1

        if peak_id.min() < 0 or peak_id.max() >= n_peaks:
            raise ValueError("Invalid peak IDs found.")

        counts += np.bincount(peak_id, minlength=n_peaks)

        print(f"{end:,}/{n_hits:,}", end="\r")

    print()

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]

    stats = {}

    print("\nMotifs per region:")
    print("mean  =", counts.mean())

    for p in percentiles:
        v = float(np.percentile(counts, p))
        stats[f"p{p}"] = v
        print(f"p{p:<4} = {v:.1f}")

    print("max   =", counts.max())

    # recommendation
    p995 = np.percentile(counts, 99.5)

    recommended_max_tokens = int(min(192, max(96, math.ceil(p995 / 8) * 8)))

    print("\nRecommended MAX_TOKENS =", recommended_max_tokens)

    stats["mean"] = float(counts.mean())
    stats["max"] = int(counts.max())
    stats["recommended_max_tokens"] = recommended_max_tokens
    stats["n_regions"] = int(n_peaks)
    stats["n_hits"] = int(n_hits)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # --------------------------------------------------------
    # offsets
    # --------------------------------------------------------

    offsets = np.zeros(n_peaks + 1, dtype=np.int64)

    np.cumsum(counts, out=offsets[1:])

    np.save(offsets_path, offsets)

    assert offsets[-1] == n_hits

    # --------------------------------------------------------
    # Allocate compressed arrays
    # --------------------------------------------------------

    motif_ids = np.lib.format.open_memmap(motif_path, mode="w+", dtype=np.uint16, shape=(n_hits,))

    position_bins = np.lib.format.open_memmap(pos_path, mode="w+", dtype=np.uint8, shape=(n_hits,))

    scores = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float16, shape=(n_hits,))

    cursor = offsets[:-1].copy()

    # --------------------------------------------------------
    # PASS 2: fill CSR
    # --------------------------------------------------------

    print("\nPASS 2: writing CSR")

    for start in range(0, n_hits, chunk_size):
        end = min(start + chunk_size, n_hits)

        peak_id = np.asarray(allmot[start:end, 0]).astype(np.int64) - 1

        score = np.asarray(allmot[start:end, 1], dtype=np.float32)

        pos = np.asarray(allmot[start:end, 2], dtype=np.uint8)

        motif = np.asarray(allmot[start:end, 3], dtype=np.uint16)

        # sort current chunk by region
        order = np.argsort(peak_id, kind="stable")

        p = peak_id[order]
        s = score[order]
        po = pos[order]
        m = motif[order]

        # groups inside current chunk
        change = np.empty(len(p), dtype=bool)

        change[0] = True
        change[1:] = p[1:] != p[:-1]

        first = np.flatnonzero(change)
        uniq = p[first]

        group_counts = np.diff(np.append(first, len(p)))

        # position inside each region group
        repeated_first = np.repeat(first, group_counts)

        within = np.arange(len(p), dtype=np.int64) - repeated_first

        dest = cursor[p] + within

        motif_ids[dest] = m
        position_bins[dest] = po
        scores[dest] = s.astype(np.float16)

        cursor[uniq] += group_counts

        print(f"{end:,}/{n_hits:,}", end="\r")

    print()

    motif_ids.flush()
    position_bins.flush()
    scores.flush()

    if not np.array_equal(cursor, offsets[1:]):
        raise RuntimeError("CSR construction failed.")

    print("\nCSR preprocessing complete.")
    print("Saved to:", CSR_DIR)


# ============================================================
# Dataset
# ============================================================


class IndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


class MotifCollator:
    def __init__(self, max_tokens):

        self.max_tokens = max_tokens

        self.offsets = np.load(CSR_DIR / "offsets.npy", mmap_mode="r")

        self.motif_ids = np.load(CSR_DIR / "motif_ids.npy", mmap_mode="r")

        self.position_bins = np.load(CSR_DIR / "position_bins.npy", mmap_mode="r")

        self.scores = np.load(CSR_DIR / "scores.npy", mmap_mode="r")

        n_peaks, n_bins = read_shape(TOTMAT_SHAPE)

        self.totmat = np.memmap(
            TOTMAT_BIN, dtype="<f8", mode="r", shape=(n_peaks, n_bins), order="F"
        )

    def __call__(self, ids):

        ids = np.asarray(ids, dtype=np.int64)

        counts = self.offsets[ids + 1] - self.offsets[ids]

        L = int(min(self.max_tokens, max(1, counts.max())))

        B = len(ids)

        motif = np.zeros((B, L), dtype=np.int64)

        pos = np.zeros((B, L), dtype=np.int64)

        score = np.zeros((B, L), dtype=np.float32)

        mask = np.zeros((B, L), dtype=bool)

        for i, rid in enumerate(ids):
            start = int(self.offsets[rid])

            end = int(self.offsets[rid + 1])

            n = end - start

            if n == 0:
                continue

            if n <= L:
                take = slice(start, end)

                m = self.motif_ids[take]
                p = self.position_bins[take]
                s = self.scores[take]

                k = n

            else:
                local_scores = np.asarray(self.scores[start:end], dtype=np.float32)

                sel = np.argpartition(local_scores, -L)[-L:]

                m = self.motif_ids[start + sel]

                p = self.position_bins[start + sel]

                s = self.scores[start + sel]

                k = L

            motif[i, :k] = m
            pos[i, :k] = p
            score[i, :k] = s
            mask[i, :k] = True

        y = np.asarray(self.totmat[ids, :], dtype=np.float32)

        return (
            torch.from_numpy(motif),
            torch.from_numpy(pos),
            torch.from_numpy(score),
            torch.from_numpy(mask),
            torch.from_numpy(y),
        )


# ============================================================
# Motif token encoder
# ============================================================


# ============================================================
# Model 1:
# Motif pooled baseline
# ============================================================


# ============================================================
# Model 2:
# GET-inspired motif Transformer
# ============================================================


# ============================================================
# Metrics
# ============================================================


@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    n = 0

    sse = np.zeros(20, dtype=np.float64)

    sum_y = np.zeros(20, dtype=np.float64)

    sum_y2 = np.zeros(20, dtype=np.float64)

    for batch in loader:
        motif, pos, score, mask, y = batch

        motif = motif.to(device, non_blocking=True)

        pos = pos.to(device, non_blocking=True)

        score = score.to(device, non_blocking=True)

        mask = mask.to(device, non_blocking=True)

        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(motif, pos, score, mask)

        pred = pred.float()

        diff = pred - y

        sse += (diff**2).sum(dim=0).cpu().numpy()

        sum_y += y.sum(dim=0).cpu().numpy()

        sum_y2 += (y**2).sum(dim=0).cpu().numpy()

        n += y.shape[0]

    sst = sum_y2 - (sum_y**2) / n

    r2_bins = 1 - sse / sst

    mse = sse.sum() / (n * 20)

    rmse = math.sqrt(mse)

    mean_r2 = r2_bins.mean()

    return {"mse": mse, "rmse": rmse, "r2_bins": r2_bins, "mean_r2": mean_r2}


# ============================================================
# Training
# ============================================================


def train_one(fold, model_name, max_tokens, epochs, patience):

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42 + fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nDevice:", device)

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    train_idx, val_idx = load_folds(FOLDS_FILE, fold)

    print(f"\nFold {fold + 1}")

    print("Train:", len(train_idx))

    print("Val:", len(val_idx))

    collate_train = MotifCollator(max_tokens)

    collate_val = MotifCollator(max_tokens)

    if model_name == "bag":
        batch_size = 4096

        model = MotifBagMLP(d_model=128)

        lr = 5e-4

    elif model_name == "transformer":
        batch_size = 1024

        model = MotifTransformer(d_model=128, n_heads=4, n_layers=4, ff_dim=512, dropout=0.1)

        lr = 3e-4

    else:
        raise ValueError(model_name)

    model = model.to(device)

    print("\nModel:", model_name)

    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    print("Batch size:", batch_size)

    print("Max motifs:", max_tokens)

    train_loader = DataLoader(
        IndexDataset(train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_train,
    )

    val_loader = DataLoader(
        IndexDataset(val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_val,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val = float("inf")
    best_epoch = 0
    wait = 0

    model_file = RESULT_DIR / f"motif_{model_name}_fold{fold + 1}.pt"

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_n = 0

        for batch in train_loader:
            motif, pos, score, mask, y = batch

            motif = motif.to(device, non_blocking=True)

            pos = pos.to(device, non_blocking=True)

            score = score.to(device, non_blocking=True)

            mask = mask.to(device, non_blocking=True)

            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(motif, pos, score, mask)

                loss = F.mse_loss(pred.float(), y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            B = y.shape[0]

            total_loss += loss.item() * B

            total_n += B

        train_mse = total_loss / total_n

        metrics = evaluate(model, val_loader, device)

        val_mse = metrics["mse"]

        scheduler.step(val_mse)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE {train_mse:.2f} | "
            f"Val MSE {val_mse:.2f} | "
            f"RMSE {metrics['rmse']:.2f} | "
            f"Mean R² {metrics['mean_r2']:.4f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            wait = 0

            torch.save(model.state_dict(), model_file)

        else:
            wait += 1

            if wait >= patience:
                print("Early stopping.")
                break

    # --------------------------------------------------------
    # final best model
    # --------------------------------------------------------

    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))

    final = evaluate(model, val_loader, device)

    output = RESULT_DIR / f"motif_{model_name}_fold{fold + 1}.npz"

    np.savez(
        output,
        mse=final["mse"],
        rmse=final["rmse"],
        mean_r2=final["mean_r2"],
        r2_bins=final["r2_bins"],
        best_epoch=best_epoch,
        max_tokens=max_tokens,
    )

    print("\n==========================")

    print(f"{model_name} Fold {fold + 1}")

    print(f"Best epoch = {best_epoch}")

    print(f"MSE = {final['mse']:.2f}")

    print(f"RMSE = {final['rmse']:.2f}")

    print(f"Mean R² = {final['mean_r2']:.4f}")

    print("==========================\n")


# ============================================================
# Summary
# ============================================================


def summarize(model_name):

    all_r2 = []
    all_rmse = []
    all_mse = []
    all_bins = []
    epochs = []

    for fold in range(1, 6):
        path = RESULT_DIR / f"motif_{model_name}_fold{fold}.npz"

        x = np.load(path)

        all_r2.append(float(x["mean_r2"]))

        all_rmse.append(float(x["rmse"]))

        all_mse.append(float(x["mse"]))

        all_bins.append(x["r2_bins"])

        epochs.append(int(x["best_epoch"]))

        print(
            f"Fold {fold} | "
            f"Epoch {int(x['best_epoch'])} | "
            f"MSE {float(x['mse']):.2f} | "
            f"RMSE {float(x['rmse']):.2f} | "
            f"R² {float(x['mean_r2']):.4f}"
        )

    print(f"\n===== Motif {model_name} =====")

    print(f"Mean R² = {np.mean(all_r2):.4f} ± {np.std(all_r2, ddof=1):.4f}")

    print(f"RMSE = {np.mean(all_rmse):.2f} ± {np.std(all_rmse, ddof=1):.2f}")

    print(f"MSE = {np.mean(all_mse):.2f} ± {np.std(all_mse, ddof=1):.2f}")

    bins = np.asarray(all_bins)

    print("\nPer-bin R²:")

    for i in range(20):
        print(f"Bin {i + 1:02d}: {bins[:, i].mean():.4f} ± {bins[:, i].std(ddof=1):.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["prepare", "train", "summary"], required=True)

    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--model", choices=["bag", "transformer"], default="transformer")

    parser.add_argument("--max-tokens", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_csr(force=args.force)

    elif args.mode == "train":
        if args.max_tokens == 0:
            stats_path = CSR_DIR / "stats.json"

            with open(stats_path) as f:
                stats = json.load(f)

            max_tokens = int(stats["recommended_max_tokens"])

        else:
            max_tokens = args.max_tokens

        train_one(
            fold=args.fold,
            model_name=args.model,
            max_tokens=max_tokens,
            epochs=args.epochs,
            patience=args.patience,
        )

    elif args.mode == "summary":
        summarize(args.model)
