
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import r2_score


# =========================================================
# 1. 参数
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["nt", "raw"],
    required=True
)

parser.add_argument(
    "--fold",
    type=int,
    required=True
)

args = parser.parse_args()

model_type = args.model
fold_id = args.fold

# CUDA_VISIBLE_DEVICES 会让每个进程只看到一张卡
device = "cuda:0"

ROOT = "/root/sc-motif-open/R"

print("=" * 50)
print("Model:", model_type)
print("Fold:", fold_id + 1)
print("GPU:", torch.cuda.get_device_name(0))
print("=" * 50)


# =========================================================
# 2. 数据
# =========================================================

with open(f"{ROOT}/totmat_shape.txt") as f:
    n_peaks, n_timepoints = map(
        int,
        f.read().split()
    )

totmat = np.memmap(
    f"{ROOT}/totmat_f64.bin",
    dtype="<f8",
    mode="r",
    shape=(n_peaks, n_timepoints),
    order="F"
)

fullseqs = np.memmap(
    f"{ROOT}/fullseqs.txt",
    dtype="S202",
    mode="r",
    shape=(n_peaks,)
)

with open(f"{ROOT}/folds.pkl", "rb") as f:
    folds = pickle.load(f)

train_idx, val_idx = folds[fold_id]

print("Train:", len(train_idx))
print("Val:", len(val_idx))


# =========================================================
# 3. 通用 Dilated Residual Block
# =========================================================

class DilatedResBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation,
        kernel_size=3,
        use_bn=False
    ):

        super().__init__()

        padding = (
            dilation * (kernel_size - 1) // 2
        )

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.use_bn = use_bn

        if use_bn:
            self.bn1 = nn.BatchNorm1d(channels)
            self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):

        residual = x

        x = self.conv1(x)

        if self.use_bn:
            x = self.bn1(x)

        x = F.relu(x)

        x = self.conv2(x)

        if self.use_bn:
            x = self.bn2(x)

        return F.relu(x + residual)


# =========================================================
# 4. Model A
# Frozen NT + Dilated ResCNN
# =========================================================

class NTResCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # 512 → 128
        self.proj = nn.Linear(
            512,
            128
        )

        self.blocks = nn.Sequential(

            DilatedResBlock(
                128,
                dilation=1
            ),

            DilatedResBlock(
                128,
                dilation=2
            ),

            DilatedResBlock(
                128,
                dilation=4
            ),

            DilatedResBlock(
                128,
                dilation=8
            )
        )

        # avg + max = 256
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)
        )

    def forward(
        self,
        x,
        mask
    ):

        # [B, T, 512]
        x = self.proj(x)

        # [B, 128, T]
        x = x.transpose(1, 2)

        x = self.blocks(x)

        # [B, T, 128]
        x = x.transpose(1, 2)

        mask3 = (
            mask.unsqueeze(-1)
            .to(x.dtype)
        )

        # ---------- mean pooling ----------

        avg_pool = (
            (x * mask3).sum(dim=1)
            /
            mask3.sum(dim=1).clamp_min(1)
        )

        # ---------- max pooling ----------

        valid = mask.unsqueeze(-1).bool()

        max_input = x.masked_fill(
            ~valid,
            torch.finfo(x.dtype).min
        )

        max_pool = max_input.max(
            dim=1
        ).values

        # [B, 256]
        x = torch.cat(
            [avg_pool, max_pool],
            dim=1
        )

        return self.head(x)


# =========================================================
# 5. Model B
# Raw DNA + Multi-scale Dilated ResCNN
# =========================================================

class RawMultiScaleResCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # -------------------------
        # Multi-scale stem
        # kernel 7 / 15 / 31
        # -------------------------

        self.branch7 = nn.Sequential(
            nn.Conv1d(
                4, 64,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.branch15 = nn.Sequential(
            nn.Conv1d(
                4, 64,
                kernel_size=15,
                padding=7
            ),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.branch31 = nn.Sequential(
            nn.Conv1d(
                4, 64,
                kernel_size=31,
                padding=15
            ),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 64 × 3 = 192 → 128
        self.project = nn.Sequential(
            nn.Conv1d(
                192,
                128,
                kernel_size=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # -------------------------
        # Dilated residual blocks
        # -------------------------

        self.blocks = nn.Sequential(

            DilatedResBlock(
                128,
                dilation=1,
                kernel_size=3,
                use_bn=True
            ),

            DilatedResBlock(
                128,
                dilation=2,
                kernel_size=3,
                use_bn=True
            ),

            DilatedResBlock(
                128,
                dilation=4,
                kernel_size=3,
                use_bn=True
            ),

            DilatedResBlock(
                128,
                dilation=8,
                kernel_size=3,
                use_bn=True
            )
        )

        # avg + max = 256
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)
        )

    def forward(self, x):

        # x = [B, 4, 201]

        x7 = self.branch7(x)
        x15 = self.branch15(x)
        x31 = self.branch31(x)

        x = torch.cat(
            [x7, x15, x31],
            dim=1
        )

        x = self.project(x)

        x = self.blocks(x)

        # Global average
        avg_pool = x.mean(dim=2)

        # Global max
        max_pool = x.max(
            dim=2
        ).values

        x = torch.cat(
            [avg_pool, max_pool],
            dim=1
        )

        return self.head(x)


# =========================================================
# 6. 快速 DNA one-hot
# =========================================================

def one_hot_encode(idx):

    s = np.asarray(
        fullseqs[idx]
    )

    s = np.ascontiguousarray(s)

    # 每条记录 S202:
    # 201 bp + newline
    b = (
        s.view(np.uint8)
        .reshape(len(idx), 202)
    )

    # 只取前 201 bp
    b = b[:, :201]

    X = np.stack(
        [
            b == ord("A"),
            b == ord("C"),
            b == ord("G"),
            b == ord("T")
        ],
        axis=1
    )

    return torch.from_numpy(
        X.astype(
            np.float32,
            copy=False
        )
    )


# =========================================================
# 7. 创建模型
# =========================================================

if model_type == "nt":

    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM
    )

    MODEL_PATH = (
        "/root/.cache/huggingface/hub/"
        "models--InstaDeepAI--"
        "nucleotide-transformer-v2-50m-multi-species/"
        "snapshots/"
        "81b29e5786726d891dbf929404ef20adca5b36f1"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )

    nt_model = (
        AutoModelForMaskedLM
        .from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True
        )
        .to(device)
    )

    nt_model.eval()

    for p in nt_model.parameters():
        p.requires_grad = False

    net = NTResCNN().to(device)

    # 32GB GPU
    batch_size = 1024


else:

    net = (
        RawMultiScaleResCNN()
        .to(device)
    )

    # Raw model显存更省
    batch_size = 4096


print(net.__class__.__name__)
print("Batch size:", batch_size)


# =========================================================
# 8. Training settings
# =========================================================

max_epochs = 30
patience = 5
lr = 1e-3

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=lr
)

best_val = np.inf
best_state = None
best_epoch = 0

no_improve = 0


# =========================================================
# 9. Training
# =========================================================

for epoch in range(max_epochs):

    net.train()

    shuffled = np.random.permutation(
        train_idx
    )

    train_loss_sum = 0.0


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    train_bar = tqdm(
        range(
            0,
            len(shuffled),
            batch_size
        ),
        desc=(
            f"{model_type.upper()} "
            f"Fold {fold_id+1} "
            f"Epoch {epoch+1:02d}"
        )
    )


    for start in train_bar:

        idx = shuffled[
            start:start + batch_size
        ]

        y = torch.as_tensor(
            np.asarray(
                totmat[idx]
            ),
            dtype=torch.float32,
            device=device
        )


        # =========================
        # NT
        # =========================

        if model_type == "nt":

            seqs = [
                fullseqs[i]
                .decode()
                .strip()
                for i in idx
            ]

            inputs = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            # NT frozen
            with torch.no_grad():

                token_emb = (
                    nt_model.base_model(
                        **inputs
                    )
                    .last_hidden_state
                )

            mask = (
                inputs["attention_mask"]
                .clone()
            )

            # remove <cls>
            mask[:, 0] = 0

            pred = net(
                token_emb,
                mask
            )


        # =========================
        # RAW DNA
        # =========================

        else:

            X = (
                one_hot_encode(idx)
                .to(device)
            )

            pred = net(X)


        # -------------------------
        # SAME LOSS FOR BOTH
        # -------------------------

        loss = loss_fn(
            pred,
            y
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()


        train_loss_sum += (
            loss.item()
            * len(idx)
        )

        train_bar.set_postfix(
            mse=f"{loss.item():.1f}"
        )


    train_mse = (
        train_loss_sum
        / len(train_idx)
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    net.eval()

    val_loss_sum = 0.0


    with torch.inference_mode():

        for start in range(
            0,
            len(val_idx),
            batch_size
        ):

            idx = val_idx[
                start:start + batch_size
            ]

            y = torch.as_tensor(
                np.asarray(
                    totmat[idx]
                ),
                dtype=torch.float32,
                device=device
            )


            # ---------------- NT ----------------

            if model_type == "nt":

                seqs = [
                    fullseqs[i]
                    .decode()
                    .strip()
                    for i in idx
                ]

                inputs = tokenizer(
                    seqs,
                    return_tensors="pt",
                    padding=True
                )

                inputs = {
                    k: v.to(device)
                    for k, v in inputs.items()
                }

                token_emb = (
                    nt_model.base_model(
                        **inputs
                    )
                    .last_hidden_state
                )

                mask = (
                    inputs[
                        "attention_mask"
                    ]
                    .clone()
                )

                mask[:, 0] = 0

                pred = net(
                    token_emb,
                    mask
                )


            # ---------------- RAW ----------------

            else:

                X = (
                    one_hot_encode(idx)
                    .to(device)
                )

                pred = net(X)


            loss = loss_fn(
                pred,
                y
            )

            val_loss_sum += (
                loss.item()
                * len(idx)
            )


    val_mse = (
        val_loss_sum
        / len(val_idx)
    )


    print(
        f"\nEpoch {epoch+1:02d} | "
        f"train MSE: {train_mse:.2f} | "
        f"val MSE: {val_mse:.2f}"
    )


    # =====================================================
    # 10. Early stopping
    # =====================================================

    if val_mse < best_val:

        best_val = val_mse

        best_state = copy.deepcopy(
            net.state_dict()
        )

        best_epoch = epoch + 1

        no_improve = 0

    else:

        no_improve += 1


    if no_improve >= patience:

        print(
            f"Early stopping at "
            f"epoch {epoch+1}"
        )

        break


# =========================================================
# 11. Load best model
# =========================================================

net.load_state_dict(
    best_state
)

net.eval()


# =========================================================
# 12. Final validation evaluation
# =========================================================

preds = []
trues = []


with torch.inference_mode():

    eval_bar = tqdm(
        range(
            0,
            len(val_idx),
            batch_size
        ),
        desc="Final evaluation"
    )


    for start in eval_bar:

        idx = val_idx[
            start:start + batch_size
        ]


        # ---------------- NT ----------------

        if model_type == "nt":

            seqs = [
                fullseqs[i]
                .decode()
                .strip()
                for i in idx
            ]

            inputs = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            token_emb = (
                nt_model.base_model(
                    **inputs
                )
                .last_hidden_state
            )

            mask = (
                inputs[
                    "attention_mask"
                ]
                .clone()
            )

            mask[:, 0] = 0

            pred = net(
                token_emb,
                mask
            )


        # ---------------- RAW ----------------

        else:

            X = (
                one_hot_encode(idx)
                .to(device)
            )

            pred = net(X)


        preds.append(
            pred.cpu().numpy()
        )

        trues.append(
            np.asarray(
                totmat[idx],
                dtype=np.float32
            )
        )


preds = np.concatenate(
    preds
)

trues = np.concatenate(
    trues
)


# =========================================================
# 13. Metrics
# =========================================================

mse = np.mean(
    (preds - trues) ** 2
)

rmse = np.sqrt(mse)

r2_bins = np.array([
    r2_score(
        trues[:, i],
        preds[:, i]
    )
    for i in range(20)
])

mean_r2 = r2_bins.mean()


print("\n")
print("=" * 55)

print(
    f"{model_type.upper()} "
    f"| Fold {fold_id+1}"
)

print("=" * 55)

print("Best epoch:", best_epoch)
print("Best val MSE:", best_val)
print("Final MSE:", mse)
print("RMSE:", rmse)
print("Mean R²:", mean_r2)

print("\nR² each bin:")
print(r2_bins)

print("=" * 55)


# =========================================================
# 14. Save
# =========================================================

RESULT_DIR = (
    f"{ROOT}/results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


np.savez(
    f"{RESULT_DIR}/"
    f"{model_type}_fold{fold_id+1}.npz",

    fold=fold_id + 1,
    best_epoch=best_epoch,

    val_mse=best_val,
    final_mse=mse,
    rmse=rmse,

    mean_r2=mean_r2,
    r2_bins=r2_bins
)


torch.save(
    best_state,

    f"{RESULT_DIR}/"
    f"{model_type}_fold{fold_id+1}.pt"
)


print("Result saved.")
