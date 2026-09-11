import argparse
import math
import pickle
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ============================================================
# Paths
# ============================================================
from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

ROOT = Path(str(DATA_ROOT))

FULLSEQS = ROOT / "fullseqs.txt"

TOTMAT_BIN = ROOT / "totmat_f64.bin"
TOTMAT_SHAPE = ROOT / "totmat_shape.txt"

FOLDS_FILE = ROOT / "folds.pkl"

RESULT_DIR = RUN_ROOT / "results"

from developmental_accessibility.paths import NT_MODEL_PATH

MODEL_PATH = NT_MODEL_PATH


# ============================================================
# Utils
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

    if isinstance(obj, dict) and "folds" in obj:
        obj = obj["folds"]

    entry = obj[fold]

    if isinstance(entry, dict):
        if "train_idx" in entry:
            train_idx = entry["train_idx"]
        else:
            train_idx = entry["train"]

        if "val_idx" in entry:
            val_idx = entry["val_idx"]
        else:
            val_idx = entry["val"]

    else:
        train_idx, val_idx = entry

    return (np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64))


# ============================================================
# Dataset
# ============================================================


class IndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


class SequenceCollator:
    def __init__(self):

        n_peaks, n_bins = read_shape(TOTMAT_SHAPE)

        self.n_peaks = n_peaks

        self.fullseqs = np.memmap(FULLSEQS, dtype="S202", mode="r", shape=(n_peaks,))

        self.totmat = np.memmap(
            TOTMAT_BIN, dtype="<f8", mode="r", shape=(n_peaks, n_bins), order="F"
        )

    def __call__(self, ids):

        ids = np.asarray(ids, dtype=np.int64)

        seqs = [self.fullseqs[i].decode("ascii").strip() for i in ids]

        y = np.asarray(self.totmat[ids, :], dtype=np.float32)

        return (seqs, torch.from_numpy(y))


# ============================================================
# Attention Pooling
# ============================================================


class AttentionPoolingHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.1):

        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

        self.attn = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 20),
        )

    def forward(self, x, mask):

        # x:
        # B x L x 512

        x = self.proj(x)

        # learned attention score
        score = self.attn(x).squeeze(-1)

        score = score.masked_fill(~mask, -1e4)

        weights = torch.softmax(score, dim=1)

        attn_pool = torch.sum(x * weights.unsqueeze(-1), dim=1)

        # Max pooling gives strong local signal
        max_pool = x.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values

        z = torch.cat([attn_pool, max_pool], dim=-1)

        return self.head(z)


# ============================================================
# Small Transformer Head
# ============================================================


class SmallTransformerHead(nn.Module):
    def __init__(self, input_dim=512, d_model=128, n_heads=4, n_layers=2, ff_dim=512, dropout=0.1):

        super().__init__()

        self.proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))

        self.region_token = nn.Parameter(torch.zeros(1, 1, d_model))

        nn.init.normal_(self.region_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        self.attn_pool = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1))

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 20),
        )

    def forward(self, x, mask):

        x = self.proj(x)

        B = x.shape[0]

        cls = self.region_token.expand(B, 1, -1)

        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)

        full_mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=~full_mask)

        x = self.norm(x)

        region_repr = x[:, 0]

        token_x = x[:, 1:]

        scores = self.attn_pool(token_x).squeeze(-1)

        scores = scores.masked_fill(~mask, -1e4)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(token_x * weights.unsqueeze(-1), dim=1)

        z = torch.cat([region_repr, pooled], dim=-1)

        return self.head(z)


# ============================================================
# Full model
# ============================================================


class NTRegressor(nn.Module):
    def __init__(self, nt_model, head_type):

        super().__init__()

        self.nt = nt_model

        # Freeze NT
        for p in self.nt.parameters():
            p.requires_grad = False

        self.nt.eval()

        if head_type == "attnpool":
            self.head = AttentionPoolingHead()

        elif head_type == "smalltx":
            self.head = SmallTransformerHead()

        else:
            raise ValueError(head_type)

    def train(self, mode=True):

        super().train(mode)

        # NT must always remain eval
        self.nt.eval()

        return self

    def forward(self, input_ids, attention_mask, special_tokens_mask):

        # IMPORTANT:
        # use no_grad, NOT inference_mode
        with torch.no_grad():
            out = self.nt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            x = out.hidden_states[-1]

        # Remove CLS / PAD / other special tokens
        # Keep real DNA tokens, remove padding and the leading <CLS>.
        # Do not use tokenizer special_tokens_mask because its length
        # can differ from input_ids for this remote tokenizer.
        valid_mask = attention_mask.bool().clone()
        valid_mask[:, 0] = False

        return self.head(x, valid_mask)


# ============================================================
# Tokenization
# ============================================================


def tokenize_batch(tokenizer, seqs, device):

    enc = tokenizer(
        seqs, padding=True, truncation=True, return_tensors="pt", return_special_tokens_mask=True
    )

    return {k: v.to(device, non_blocking=True) for k, v in enc.items()}


# ============================================================
# Evaluation
# ============================================================


@torch.no_grad()
def evaluate(model, loader, tokenizer, device):

    model.eval()

    n = 0

    sse = np.zeros(20, dtype=np.float64)

    sum_y = np.zeros(20, dtype=np.float64)

    sum_y2 = np.zeros(20, dtype=np.float64)

    for seqs, y in loader:
        y = y.to(device, non_blocking=True)

        enc = tokenize_batch(tokenizer, seqs, device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(enc["input_ids"], enc["attention_mask"], enc["special_tokens_mask"])

        pred = pred.float()

        diff = pred - y

        sse += (diff**2).sum(dim=0).cpu().numpy()

        sum_y += y.sum(dim=0).cpu().numpy()

        sum_y2 += (y**2).sum(dim=0).cpu().numpy()

        n += y.shape[0]

    sst = sum_y2 - (sum_y**2) / n

    r2_bins = 1.0 - sse / sst

    mse = sse.sum() / (n * 20)

    rmse = math.sqrt(mse)

    return {"mse": mse, "rmse": rmse, "mean_r2": float(r2_bins.mean()), "r2_bins": r2_bins}


# ============================================================
# Train
# ============================================================


def train_one(fold, head_type, epochs, patience, batch_size):

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42 + fold)

    device = torch.device("cuda:0")

    print("Device:", device, flush=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )

    print("Loading NT model...", flush=True)

    nt_model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )

    model = NTRegressor(nt_model, head_type).to(device)

    train_idx, val_idx = load_folds(FOLDS_FILE, fold)

    print(f"\nFold {fold + 1}", flush=True)

    print(f"Train: {len(train_idx):,}", flush=True)

    print(f"Val:   {len(val_idx):,}", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Head: {head_type}", flush=True)

    print(f"Trainable parameters: {trainable:,}", flush=True)

    print(f"Batch size: {batch_size}", flush=True)

    collator_train = SequenceCollator()
    collator_val = SequenceCollator()

    train_loader = DataLoader(
        IndexDataset(train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collator_train,
    )

    val_loader = DataLoader(
        IndexDataset(val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collator_val,
    )

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_mse = float("inf")
    best_epoch = 0
    wait = 0

    ckpt = RESULT_DIR / f"nt_{head_type}_fold{fold + 1}.pt"

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_n = 0

        for step, (seqs, y) in enumerate(train_loader, 1):
            y = y.to(device, non_blocking=True)

            enc = tokenize_batch(tokenizer, seqs, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(enc["input_ids"], enc["attention_mask"], enc["special_tokens_mask"])

                loss = F.mse_loss(pred.float(), y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.0)

            optimizer.step()

            B = y.shape[0]

            total_loss += loss.item() * B

            total_n += B

            # show progress every 500 batches
            if step % 500 == 0:
                print(
                    f"Epoch {epoch:02d} | "
                    f"step {step:04d}/{len(train_loader)} | "
                    f"Train MSE "
                    f"{total_loss / total_n:.2f}",
                    flush=True,
                )

        train_mse = total_loss / total_n

        metrics = evaluate(model, val_loader, tokenizer, device)

        scheduler.step(metrics["mse"])

        print(
            f"\nEpoch {epoch:02d} DONE | "
            f"Train MSE {train_mse:.2f} | "
            f"Val MSE {metrics['mse']:.2f} | "
            f"RMSE {metrics['rmse']:.2f} | "
            f"Mean R² {metrics['mean_r2']:.4f}\n",
            flush=True,
        )

        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            best_epoch = epoch
            wait = 0

            torch.save(model.head.state_dict(), ckpt)

        else:
            wait += 1

            if wait >= patience:
                print("Early stopping.", flush=True)

                break

    # ========================================================
    # Reload best head
    # ========================================================

    model.head.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    final = evaluate(model, val_loader, tokenizer, device)

    np.savez(
        RESULT_DIR / f"nt_{head_type}_fold{fold + 1}.npz",
        mse=final["mse"],
        rmse=final["rmse"],
        mean_r2=final["mean_r2"],
        r2_bins=final["r2_bins"],
        best_epoch=best_epoch,
    )

    print("\n============================", flush=True)

    print(f"NT {head_type} Fold {fold + 1}", flush=True)

    print(f"Best epoch = {best_epoch}", flush=True)

    print(f"MSE = {final['mse']:.2f}", flush=True)

    print(f"RMSE = {final['rmse']:.2f}", flush=True)

    print(f"Mean R² = {final['mean_r2']:.4f}", flush=True)

    print("============================", flush=True)


# ============================================================
# Summary
# ============================================================


def summary(head_type):

    r2 = []
    rmse = []
    mse = []
    bins = []
    epochs = []

    for fold in range(1, 6):
        x = np.load(RESULT_DIR / f"nt_{head_type}_fold{fold}.npz")

        r2.append(float(x["mean_r2"]))

        rmse.append(float(x["rmse"]))

        mse.append(float(x["mse"]))

        bins.append(x["r2_bins"])

        epochs.append(int(x["best_epoch"]))

        print(
            f"Fold {fold} | "
            f"Best epoch {epochs[-1]} | "
            f"MSE {mse[-1]:.2f} | "
            f"RMSE {rmse[-1]:.2f} | "
            f"Mean R² {r2[-1]:.4f}"
        )

    print(f"\n===== NT {head_type} =====")

    print(f"Mean R² = {np.mean(r2):.4f} ± {np.std(r2, ddof=1):.4f}")

    print(f"RMSE = {np.mean(rmse):.2f} ± {np.std(rmse, ddof=1):.2f}")

    print(f"MSE = {np.mean(mse):.2f} ± {np.std(mse, ddof=1):.2f}")

    bins = np.asarray(bins)

    print("\nPer-bin R²:")

    for i in range(20):
        print(f"Bin {i + 1:02d}: {bins[:, i].mean():.4f} ± {bins[:, i].std(ddof=1):.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, choices=range(5))

    parser.add_argument("--head", choices=["attnpool", "smalltx"], default="attnpool")

    parser.add_argument("--epochs", type=int, default=25)

    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--batch-size", type=int, default=1024)

    parser.add_argument("--summary", action="store_true")

    args = parser.parse_args()
    if not args.summary and args.fold is None:
        parser.error("--fold is required for training")
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch-size must be positive")

    if args.summary:
        summary(args.head)

    else:
        train_one(
            fold=args.fold,
            head_type=args.head,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
