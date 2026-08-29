#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Motif-only Transformer Fold-1 screening on all development regions (no p-value filter).

Input per genomic region:
    10 position bins x 1,443 HOCOMOCO motif models.
    Missing motif-position cells = 0.
    Duplicate (position, motif) hits are aggregated by maximum score.

Model:
    motif_transformer
    score 1443->256 + presence 1443->256 for each of 10 position tokens
    + learned position embedding
    -> 2-layer Transformer (d=256, 8 heads, FF=1024)
    -> mean/max/learned-query gated pooling
    -> 256 -> 128 -> 20

The original grouped folds.pkl is used unchanged. No p-value filtering is applied.
The untouched 10% final test set therefore remains untouched; Fold 1 uses the
original development-set train/validation indices only.

Outputs:
    /root/sc-motif-open/R/results_motif_allregions/motif_transformer/fold1.npz
"""

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
from sklearn.metrics import r2_score
from tqdm import tqdm


# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0, help="0-based; 0 = Fold 1")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--accum-steps", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.10)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--rc-prob", type=float, default=0.5)
args = parser.parse_args()


# ============================================================
# Paths
# ============================================================
ROOT = Path("/root/sc-motif-open/R")
MOTIF_DIR = ROOT / "motif_csr"
RESULT_ROOT = ROOT / "results_motif_allregions"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

variant_id = 2
variant_name = "motif_transformer"
fold_id = args.fold

RESULT_DIR = RESULT_ROOT / variant_name
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = RESULT_DIR / f"fold{fold_id + 1}.npz"
MODEL_FILE = RESULT_DIR / f"fold{fold_id + 1}.pt"
CONFIG_FILE = RESULT_DIR / f"fold{fold_id + 1}_config.json"


# ============================================================
# Reproducibility
# ============================================================
seed = 42 + fold_id
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0")


# ============================================================
# Target data + original grouped split
# ============================================================
with open(ROOT / "totmat_shape.txt") as f:
    n_regions, n_timepoints = map(int, f.read().split())

if n_timepoints != 20:
    raise ValueError(f"Expected 20 target bins, got {n_timepoints}")

totmat = np.memmap(
    ROOT / "totmat_f64.bin",
    dtype="<f8",
    mode="r",
    shape=(n_regions, n_timepoints),
    order="F",
)

with open(ROOT / "folds.pkl", "rb") as f:
    folds = pickle.load(f)

train_idx_all, val_idx_all = folds[fold_id]
train_idx_all = np.asarray(train_idx_all, dtype=np.int64)
val_idx_all = np.asarray(val_idx_all, dtype=np.int64)


# ============================================================
# Use the original grouped Fold split unchanged
# ============================================================
train_idx = train_idx_all
val_idx = val_idx_all

# ============================================================
# Motif CSR-like arrays
# ============================================================
N_MOTIFS = 1443
N_POS = 10

required = {
    "offsets": MOTIF_DIR / "offsets.npy",
    "motif_ids": MOTIF_DIR / "motif_ids.npy",
    "position_bins": MOTIF_DIR / "position_bins.npy",
    "scores": MOTIF_DIR / "scores.npy",
}
missing = [str(p) for p in required.values() if not p.exists()]
if missing:
    raise FileNotFoundError("Missing motif files:\n" + "\n".join(missing))

offsets = np.load(required["offsets"], mmap_mode="r")
motif_ids = np.load(required["motif_ids"], mmap_mode="r")
position_bins = np.load(required["position_bins"], mmap_mode="r")
scores = np.load(required["scores"], mmap_mode="r")

if offsets.shape != (n_regions + 1,):
    raise ValueError(f"offsets shape {offsets.shape} != {(n_regions + 1,)}")
if not (len(motif_ids) == len(position_bins) == len(scores)):
    raise ValueError("motif_ids / position_bins / scores lengths differ")


def motif_batch(idx):
    """Reconstruct B x 10 x 1443 using maximum score per cell."""
    idx = np.asarray(idx, dtype=np.int64)
    B = len(idx)
    starts = np.asarray(offsets[idx], dtype=np.int64)
    ends = np.asarray(offsets[idx + 1], dtype=np.int64)
    counts = ends - starts
    total_hits = int(counts.sum())

    out = np.zeros((B, N_POS, N_MOTIFS), dtype=np.float32)
    if total_hits == 0:
        return torch.from_numpy(out)

    cumulative = np.cumsum(counts, dtype=np.int64)
    previous = np.concatenate((np.array([0], dtype=np.int64), cumulative[:-1]))
    row_ids = np.repeat(np.arange(B, dtype=np.int64), counts)
    hit_indices = (
        np.arange(total_hits, dtype=np.int64)
        - np.repeat(previous, counts)
        + np.repeat(starts, counts)
    )

    mids = np.asarray(motif_ids[hit_indices], dtype=np.int64) - 1
    pos = np.asarray(position_bins[hit_indices], dtype=np.int64) - 1
    vals = np.asarray(scores[hit_indices], dtype=np.float32)

    valid = (
        (mids >= 0) & (mids < N_MOTIFS)
        & (pos >= 0) & (pos < N_POS)
        & np.isfinite(vals)
    )
    row_ids = row_ids[valid]
    mids = mids[valid]
    pos = pos[valid]
    vals = vals[valid]

    flat = out.reshape(B, N_POS * N_MOTIFS)
    cols = pos * N_MOTIFS + mids
    np.maximum.at(flat, (row_ids, cols), vals)
    return torch.from_numpy(out)


def reverse_position_augment(M, p=0.5):
    """Flip motif position bins 1<->10, 2<->9, ... for a random subset."""
    if p <= 0:
        return M
    mask = torch.rand(M.shape[0], device=M.device) < p
    if mask.any():
        M[mask] = torch.flip(M[mask], dims=[1])
    return M


# ============================================================
# Models
# ============================================================
class LearnedQueryPool(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.10):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        q = self.query.expand(x.shape[0], -1, -1)
        z, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(z[:, 0])


class GatedPooling(nn.Module):
    def __init__(self, d_model=256, dropout=0.10):
        super().__init__()
        self.query = LearnedQueryPool(d_model, 8, dropout)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, x):
        mean = x.mean(dim=1)
        maxv = x.max(dim=1).values
        query = self.query(x)
        summaries = torch.stack([mean, maxv, query], dim=1)
        weights = torch.softmax(
            self.gate(torch.cat([mean, maxv, query], dim=1)), dim=1
        )
        return (summaries * weights.unsqueeze(-1)).sum(dim=1)


class MotifTransformer(nn.Module):
    def __init__(self, dropout=0.10):
        super().__init__()
        self.score_proj = nn.Linear(N_MOTIFS, 256, bias=False)
        self.presence_proj = nn.Linear(N_MOTIFS, 256, bias=False)
        self.input_norm = nn.LayerNorm(256)
        self.pos = nn.Parameter(torch.randn(1, N_POS, 256) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(256)
        )
        self.pool = GatedPooling(256, dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 20),
        )

    def forward(self, M):
        presence = (M > 0).to(M.dtype)
        x = self.score_proj(M) + self.presence_proj(presence)
        x = F.gelu(self.input_norm(x))
        x = x + self.pos
        x = self.transformer(x)
        z = self.pool(x)
        return self.head(z)


net = MotifTransformer(args.dropout).to(device)
n_params = sum(p.numel() for p in net.parameters())

print("=" * 92)
print("Motif-only Transformer | all development regions | Fold screening")
print("=" * 92)
print("Fold:", fold_id + 1)
print("GPU:", torch.cuda.get_device_name(0))
print("All regions:", f"{n_regions:,}")
print("Fold train:", f"{len(train_idx):,}")
print("Fold val:", f"{len(val_idx):,}")
print("NOTE: untouched final 10% test set is not used here")
print("Input: 10 x 1443 motif scores + deterministic presence")
print("Duplicate aggregation: max score")
print("Position RC/reversal augmentation probability:", args.rc_prob)
print("Parameters:", f"{n_params:,}")
print("Batch:", args.batch_size, "Accum:", args.accum_steps)
print("LR:", args.lr, "Weight decay:", args.weight_decay)
print("=" * 92)

config = {
    "variant": variant_id,
    "variant_name": variant_name,
    "fold": fold_id + 1,
    "seed": seed,
    "region_filter": "none_within_original_fold",
    "train_n": int(len(train_idx)),
    "val_n": int(len(val_idx)),
    "n_motifs": N_MOTIFS,
    "n_position_bins": N_POS,
    "duplicate_aggregation": "max",
    "score_and_presence": True,
    "rc_position_probability": args.rc_prob,
    "epochs": args.epochs,
    "patience": args.patience,
    "batch_size": args.batch_size,
    "accum_steps": args.accum_steps,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "dropout": args.dropout,
    "n_parameters": n_params,
}
with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=2)


# ============================================================
# Training
# ============================================================
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(
    net.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
scaler = torch.cuda.amp.GradScaler()

best_val = np.inf
best_state = None
best_epoch = 0
no_improve = 0

for epoch in range(args.epochs):
    net.train()
    shuffled = np.random.permutation(train_idx)
    starts = list(range(0, len(shuffled), args.batch_size))
    optimizer.zero_grad(set_to_none=True)
    train_loss_sum = 0.0

    bar = tqdm(
        enumerate(starts),
        total=len(starts),
        desc=f"{variant_name} Fold {fold_id+1} Epoch {epoch+1:02d}",
    )

    for batch_num, start in bar:
        idx = shuffled[start:start + args.batch_size]
        M = motif_batch(idx).to(device, non_blocking=True)
        M = reverse_position_augment(M, args.rc_prob)
        y = torch.as_tensor(
            np.asarray(totmat[idx]), dtype=torch.float32, device=device
        )

        with torch.cuda.amp.autocast():
            pred = net(M)
            raw_loss = loss_fn(pred, y)
            loss = raw_loss / args.accum_steps

        scaler.scale(loss).backward()

        boundary = ((batch_num + 1) % args.accum_steps == 0)
        last_batch = (batch_num + 1 == len(starts))
        if boundary or last_batch:
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss_sum += raw_loss.item() * len(idx)
        bar.set_postfix(mse=f"{raw_loss.item():.1f}")

    train_mse = train_loss_sum / len(train_idx)

    net.eval()
    val_loss_sum = 0.0
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(val_idx), args.batch_size),
            desc="Validation", leave=False
        ):
            idx = val_idx[start:start + args.batch_size]
            M = motif_batch(idx).to(device, non_blocking=True)
            y = torch.as_tensor(
                np.asarray(totmat[idx]), dtype=torch.float32, device=device
            )
            with torch.cuda.amp.autocast():
                pred = net(M)
                loss = loss_fn(pred, y)
            val_loss_sum += loss.item() * len(idx)

    val_mse = val_loss_sum / len(val_idx)
    print(
        f"\nEpoch {epoch+1:02d} | train MSE: {train_mse:.2f} | "
        f"val MSE: {val_mse:.2f}", flush=True
    )

    if val_mse < best_val:
        best_val = val_mse
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        best_epoch = epoch + 1
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= args.patience:
        print(f"Early stopping at epoch {epoch+1}", flush=True)
        break

if best_state is None:
    raise RuntimeError("No best state recorded")

net.load_state_dict(best_state)
net.to(device)
net.eval()

preds, trues = [], []
with torch.inference_mode():
    for start in tqdm(range(0, len(val_idx), args.batch_size), desc="Final evaluation"):
        idx = val_idx[start:start + args.batch_size]
        M = motif_batch(idx).to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            pred = net(M)
        preds.append(pred.float().cpu().numpy())
        trues.append(np.asarray(totmat[idx], dtype=np.float32))

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
mse = float(np.mean((preds - trues) ** 2))
rmse = float(np.sqrt(mse))
r2_bins = np.array(
    [r2_score(trues[:, i], preds[:, i]) for i in range(20)],
    dtype=np.float64,
)
mean_r2 = float(r2_bins.mean())

print("\n" + "=" * 92)
print(f"{variant_name} | Fold {fold_id+1} | all development regions")
print("Best epoch:", best_epoch)
print("Best val MSE:", best_val)
print("Final MSE:", mse)
print("Final RMSE:", rmse)
print("Mean R²:", mean_r2)
print("R² each bin:", r2_bins)
print("=" * 92)

np.savez(
    RESULT_FILE,
    variant=variant_id,
    variant_name=variant_name,
    fold=fold_id + 1,
    region_filter="none_within_original_fold",
    train_n=len(train_idx),
    val_n=len(val_idx),
    best_epoch=best_epoch,
    best_val_mse=best_val,
    mse=mse,
    rmse=rmse,
    mean_r2=mean_r2,
    r2_bins=r2_bins,
    n_parameters=n_params,
)
torch.save(best_state, MODEL_FILE)

print("Saved metrics:", RESULT_FILE)
print("Saved model:  ", MODEL_FILE)
print("Saved config: ", CONFIG_FILE)