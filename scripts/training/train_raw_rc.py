def main():

    import argparse
    import copy
    import os
    import pickle
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import r2_score
    from tqdm import tqdm

    # ============================================================
    # Arguments
    # ============================================================
    from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--batch-size", type=int, default=4096)

    parser.add_argument("--summary", action="store_true")

    args = parser.parse_args()

    ROOT = str(DATA_ROOT)
    RESULT_DIR = str(RUN_ROOT / "results")

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ============================================================
    # Summary
    # ============================================================

    def summary():

        all_r2 = []
        all_rmse = []
        all_mse = []
        all_bins = []
        best_epochs = []

        for fold in range(1, 6):
            x = np.load(f"{RESULT_DIR}/raw_rc_fold{fold}.npz")

            all_r2.append(float(x["mean_r2"]))

            all_rmse.append(float(x["rmse"]))

            all_mse.append(float(x["mse"]))

            all_bins.append(x["r2_bins"])

            best_epochs.append(int(x["best_epoch"]))

            print(
                f"Fold {fold} | "
                f"Best epoch = {int(x['best_epoch'])} | "
                f"MSE = {float(x['mse']):.2f} | "
                f"RMSE = {float(x['rmse']):.2f} | "
                f"Mean R² = {float(x['mean_r2']):.4f}"
            )

        print("\n===== Raw Multi-scale Dilated ResCNN + RC =====")

        print(f"Mean R² = {np.mean(all_r2):.4f} ± {np.std(all_r2, ddof=1):.4f}")

        print(f"RMSE = {np.mean(all_rmse):.2f} ± {np.std(all_rmse, ddof=1):.2f}")

        print(f"MSE = {np.mean(all_mse):.2f} ± {np.std(all_mse, ddof=1):.2f}")

        print("Best epochs =", best_epochs)

        bins = np.asarray(all_bins)

        print("\nPer-bin R²:")

        for i in range(20):
            print(f"Bin {i + 1:02d}: {bins[:, i].mean():.4f} ± {bins[:, i].std(ddof=1):.4f}")

    if args.summary:
        summary()
        raise SystemExit

    # ============================================================
    # Reproducibility
    # ============================================================

    fold_id = args.fold

    seed = 42 + fold_id

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA_VISIBLE_DEVICES makes this process see one GPU
    device = "cuda:0"

    print("=" * 60)
    print("Model: Raw Multi-scale Dilated ResCNN + RC")
    print("Fold:", fold_id + 1)
    print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 60)

    # ============================================================
    # Data
    # ============================================================

    with open(f"{ROOT}/totmat_shape.txt") as f:
        n_peaks, n_timepoints = map(int, f.read().split())

    totmat = np.memmap(
        f"{ROOT}/totmat_f64.bin", dtype="<f8", mode="r", shape=(n_peaks, n_timepoints), order="F"
    )

    fullseqs = np.memmap(f"{ROOT}/fullseqs.txt", dtype="S202", mode="r", shape=(n_peaks,))

    with open(f"{ROOT}/folds.pkl", "rb") as f:
        folds = pickle.load(f)

    train_idx, val_idx = folds[fold_id]

    train_idx = np.asarray(train_idx, dtype=np.int64)

    val_idx = np.asarray(val_idx, dtype=np.int64)

    print("Train:", len(train_idx))

    print("Val:", len(val_idx))

    # ============================================================
    # Dilated Residual Block
    # ============================================================

    class DilatedResBlock(nn.Module):
        def __init__(self, channels, dilation):

            super().__init__()

            self.conv1 = nn.Conv1d(
                channels, channels, kernel_size=3, padding=dilation, dilation=dilation
            )

            self.conv2 = nn.Conv1d(
                channels, channels, kernel_size=3, padding=dilation, dilation=dilation
            )

            self.bn1 = nn.BatchNorm1d(channels)

            self.bn2 = nn.BatchNorm1d(channels)

        def forward(self, x):

            residual = x

            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)

            return F.relu(x + residual)

    # ============================================================
    # Raw Multi-scale Dilated ResCNN
    # SAME architecture as original model
    # ============================================================

    class RawMultiScaleResCNN(nn.Module):
        def __init__(self):

            super().__init__()

            # ----------------------------------------------------
            # Multi-scale stem
            # ----------------------------------------------------

            self.branch7 = nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU()
            )

            self.branch15 = nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=15, padding=7), nn.BatchNorm1d(64), nn.ReLU()
            )

            self.branch31 = nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=31, padding=15), nn.BatchNorm1d(64), nn.ReLU()
            )

            # 64 x 3 = 192 -> 128
            self.project = nn.Sequential(
                nn.Conv1d(192, 128, kernel_size=1), nn.BatchNorm1d(128), nn.ReLU()
            )

            # ----------------------------------------------------
            # Dilated residual blocks
            # ----------------------------------------------------

            self.blocks = nn.Sequential(
                DilatedResBlock(128, dilation=1),
                DilatedResBlock(128, dilation=2),
                DilatedResBlock(128, dilation=4),
                DilatedResBlock(128, dilation=8),
            )

            # mean + max = 256
            self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 20))

        def forward(self, x):

            # x: B x 4 x 201

            x7 = self.branch7(x)
            x15 = self.branch15(x)
            x31 = self.branch31(x)

            x = torch.cat([x7, x15, x31], dim=1)

            x = self.project(x)

            x = self.blocks(x)

            avg_pool = x.mean(dim=2)

            max_pool = x.max(dim=2).values

            x = torch.cat([avg_pool, max_pool], dim=1)

            return self.head(x)

    # ============================================================
    # Fast one-hot encoding
    # ============================================================

    def one_hot_encode(idx):

        s = np.asarray(fullseqs[idx])

        s = np.ascontiguousarray(s)

        # S202 = 201 bases + newline
        b = s.view(np.uint8).reshape(len(idx), 202)

        b = b[:, :201]

        X = np.stack([b == ord("A"), b == ord("C"), b == ord("G"), b == ord("T")], axis=1)

        return torch.from_numpy(X.astype(np.float32, copy=False))

    # ============================================================
    # Reverse-complement augmentation
    # ============================================================

    def reverse_complement_augment(X):
        """
        X: B x 4 x 201

        Channel order:
        A, C, G, T

        Reverse complement:
        A <-> T
        C <-> G

        Applied independently to 50% of training sequences.
        """

        B = X.shape[0]

        flip_mask = torch.rand(B, device=X.device) < 0.5

        if flip_mask.any():
            selected = X[flip_mask]

            # complement:
            # A,C,G,T -> T,G,C,A
            selected = selected[:, [3, 2, 1, 0], :]

            # reverse sequence direction
            selected = torch.flip(selected, dims=[2])

            X[flip_mask] = selected

        return X

    # ============================================================
    # Model
    # ============================================================

    net = RawMultiScaleResCNN().to(device)

    print(net.__class__.__name__)

    print("Batch size:", args.batch_size)

    print("RC probability: 0.5")

    # ============================================================
    # Training settings
    # ============================================================

    batch_size = args.batch_size
    max_epochs = args.epochs
    patience = args.patience

    lr = 1e-3

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_val = np.inf
    best_state = None
    best_epoch = 0

    no_improve = 0

    # ============================================================
    # Train
    # ============================================================

    for epoch in range(max_epochs):
        net.train()

        shuffled = np.random.permutation(train_idx)

        train_loss_sum = 0.0

        train_bar = tqdm(
            range(0, len(shuffled), batch_size),
            desc=(f"RC Fold {fold_id + 1} Epoch {epoch + 1:02d}"),
        )

        for start in train_bar:
            idx = shuffled[start : start + batch_size]

            X = one_hot_encode(idx).to(device, non_blocking=True)

            # ============================================
            # RC augmentation ONLY during training
            # ============================================

            X = reverse_complement_augment(X)

            y = torch.as_tensor(np.asarray(totmat[idx]), dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)

            pred = net(X)

            loss = loss_fn(pred, y)

            loss.backward()

            optimizer.step()

            train_loss_sum += loss.item() * len(idx)

            train_bar.set_postfix(mse=f"{loss.item():.1f}")

        train_mse = train_loss_sum / len(train_idx)

        # ========================================================
        # Validation
        # NO reverse complement augmentation here
        # ========================================================

        net.eval()

        val_loss_sum = 0.0

        with torch.inference_mode():
            for start in range(0, len(val_idx), batch_size):
                idx = val_idx[start : start + batch_size]

                X = one_hot_encode(idx).to(device, non_blocking=True)

                y = torch.as_tensor(np.asarray(totmat[idx]), dtype=torch.float32, device=device)

                pred = net(X)

                loss = loss_fn(pred, y)

                val_loss_sum += loss.item() * len(idx)

        val_mse = val_loss_sum / len(val_idx)

        print(
            f"\nEpoch {epoch + 1:02d} | train MSE: {train_mse:.2f} | val MSE: {val_mse:.2f}",
            flush=True,
        )

        # ========================================================
        # Early stopping
        # ========================================================

        if val_mse < best_val:
            best_val = val_mse

            best_state = copy.deepcopy(net.state_dict())

            best_epoch = epoch + 1

            no_improve = 0

        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)

            break

    # ============================================================
    # Restore best model
    # ============================================================

    net.load_state_dict(best_state)

    net.eval()

    # ============================================================
    # Final validation
    # ============================================================

    preds = []
    trues = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(val_idx), batch_size), desc="Final evaluation"):
            idx = val_idx[start : start + batch_size]

            X = one_hot_encode(idx).to(device, non_blocking=True)

            pred = net(X)

            preds.append(pred.cpu().numpy())

            trues.append(np.asarray(totmat[idx], dtype=np.float32))

    preds = np.concatenate(preds)

    trues = np.concatenate(trues)

    # ============================================================
    # Metrics
    # ============================================================

    mse = np.mean((preds - trues) ** 2)

    rmse = np.sqrt(mse)

    r2_bins = np.array([r2_score(trues[:, i], preds[:, i]) for i in range(20)])

    mean_r2 = r2_bins.mean()

    print("\n")
    print("=" * 60)

    print(f"Raw ResCNN + RC | Fold {fold_id + 1}")

    print("=" * 60)

    print("Best epoch:", best_epoch)

    print("MSE:", mse)

    print("RMSE:", rmse)

    print("Mean R²:", mean_r2)

    print("\nR² each bin:")

    print(r2_bins)

    print("=" * 60)

    # ============================================================
    # Save results
    # ============================================================

    np.savez(
        f"{RESULT_DIR}/raw_rc_fold{fold_id + 1}.npz",
        fold=fold_id + 1,
        best_epoch=best_epoch,
        mse=mse,
        rmse=rmse,
        mean_r2=mean_r2,
        r2_bins=r2_bins,
    )

    torch.save(best_state, f"{RESULT_DIR}/raw_rc_fold{fold_id + 1}.pt")

    print(f"\nSaved: raw_rc_fold{fold_id + 1}.npz")


if __name__ == "__main__":
    main()
