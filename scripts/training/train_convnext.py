def main():

    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import argparse
    import copy
    import pickle

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import r2_score
    from tqdm import tqdm

    # =========================================================
    # 1. Arguments
    # =========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    fold_id = args.fold

    # CUDA_VISIBLE_DEVICES控制实际GPU
    device = "cuda:0"

    from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

    ROOT = str(DATA_ROOT)

    print("=" * 60)
    print("Model: ConvNeXt-dCNN")
    print("Fold:", fold_id + 1)
    print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 60)

    # =========================================================
    # 2. Load data
    # =========================================================

    with open(f"{ROOT}/totmat_shape.txt") as f:
        n_peaks, n_timepoints = map(int, f.read().split())

    totmat = np.memmap(
        f"{ROOT}/totmat_f64.bin", dtype="<f8", mode="r", shape=(n_peaks, n_timepoints), order="F"
    )

    fullseqs = np.memmap(f"{ROOT}/fullseqs.txt", dtype="S202", mode="r", shape=(n_peaks,))

    with open(f"{ROOT}/folds.pkl", "rb") as f:
        folds = pickle.load(f)

    train_idx, val_idx = folds[fold_id]

    print("Train:", len(train_idx))
    print("Val:", len(val_idx))

    # =========================================================
    # 3. ConvNeXt 1D Block
    # =========================================================

    class ConvNeXt1DBlock(nn.Module):
        def __init__(self, dim, kernel_size=7, expansion=4):

            super().__init__()

            padding = kernel_size // 2

            # ---------------------------------
            # spatial modelling
            # 每个channel独立做卷积
            # ---------------------------------

            self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)

            # ---------------------------------
            # channel mixing
            # ---------------------------------

            self.norm = nn.LayerNorm(dim)

            self.pw1 = nn.Linear(dim, expansion * dim)

            self.pw2 = nn.Linear(expansion * dim, dim)

            self.act = nn.GELU()

            # ConvNeXt layer scale
            self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

        def forward(self, x):

            residual = x

            # [B,C,L]
            x = self.dwconv(x)

            # [B,L,C]
            x = x.transpose(1, 2)

            x = self.norm(x)

            x = self.pw1(x)

            x = self.act(x)

            x = self.pw2(x)

            x = self.gamma * x

            # [B,C,L]
            x = x.transpose(1, 2)

            return residual + x

    # =========================================================
    # 4. Dilated Residual Block
    # =========================================================

    class DilatedResBlock(nn.Module):
        def __init__(self, channels, dilation):

            super().__init__()

            self.conv1 = nn.Conv1d(
                channels, channels, kernel_size=3, padding=dilation, dilation=dilation
            )

            self.conv2 = nn.Conv1d(
                channels, channels, kernel_size=3, padding=dilation, dilation=dilation
            )

            self.norm1 = nn.BatchNorm1d(channels)

            self.norm2 = nn.BatchNorm1d(channels)

        def forward(self, x):

            residual = x

            x = self.conv1(x)
            x = self.norm1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.norm2(x)

            return F.relu(x + residual)

    # =========================================================
    # 5. ConvNeXt-dCNN
    # =========================================================

    class ConvNeXtDilatedCNN(nn.Module):
        def __init__(self):

            super().__init__()

            # ---------------------------------
            # Initial DNA feature extraction
            # ---------------------------------

            self.stem = nn.Sequential(
                nn.Conv1d(4, 128, kernel_size=15, padding=7), nn.BatchNorm1d(128), nn.GELU()
            )

            # ---------------------------------
            # ConvNeXt feature extractor
            # ---------------------------------

            self.convnext = nn.Sequential(
                ConvNeXt1DBlock(128, kernel_size=7),
                ConvNeXt1DBlock(128, kernel_size=7),
                ConvNeXt1DBlock(128, kernel_size=7),
            )

            # ---------------------------------
            # Dilated CNN
            # ---------------------------------

            self.dilated = nn.Sequential(
                DilatedResBlock(128, dilation=1),
                DilatedResBlock(128, dilation=2),
                DilatedResBlock(128, dilation=4),
                DilatedResBlock(128, dilation=8),
            )

            # ---------------------------------
            # Prediction head
            # avg + max = 256
            # ---------------------------------

            self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 20))

        def forward(self, x):

            # [B,4,201]

            x = self.stem(x)

            # [B,128,201]

            x = self.convnext(x)

            x = self.dilated(x)

            # ---------------------------------
            # Global average pooling
            # ---------------------------------

            avg_pool = x.mean(dim=2)

            # ---------------------------------
            # Global max pooling
            # ---------------------------------

            max_pool = x.max(dim=2).values

            # [B,256]

            x = torch.cat([avg_pool, max_pool], dim=1)

            return self.head(x)

    # =========================================================
    # 6. Fast one-hot encoding
    # =========================================================

    def one_hot_encode(idx):

        s = np.asarray(fullseqs[idx])

        s = np.ascontiguousarray(s)

        b = s.view(np.uint8).reshape(len(idx), 202)

        # 201bp
        b = b[:, :201]

        X = np.stack([b == ord("A"), b == ord("C"), b == ord("G"), b == ord("T")], axis=1)

        return torch.from_numpy(X.astype(np.float32, copy=False))

    # =========================================================
    # 7. Model
    # =========================================================

    net = ConvNeXtDilatedCNN().to(device)

    print(net.__class__.__name__)

    # =========================================================
    # 8. Training settings
    # =========================================================

    # ConvNeXt中间会expand 128→512，
    # 所以比刚才Raw模型更吃显存
    batch_size = 2048

    max_epochs = args.epochs

    patience = args.patience

    lr = 1e-3

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_val = np.inf

    best_state = None

    best_epoch = 0

    no_improve = 0

    print("Batch size:", batch_size)

    # =========================================================
    # 9. Training
    # =========================================================

    for epoch in range(max_epochs):
        net.train()

        shuffled = np.random.permutation(train_idx)

        train_loss_sum = 0.0

        # =====================================================
        # Train
        # =====================================================

        train_bar = tqdm(
            range(0, len(shuffled), batch_size),
            desc=(f"ConvNeXt Fold {fold_id + 1} Epoch {epoch + 1:02d}"),
        )

        for start in train_bar:
            idx = shuffled[start : start + batch_size]

            X = one_hot_encode(idx).to(device)

            y = torch.as_tensor(np.asarray(totmat[idx]), dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)

            pred = net(X)

            loss = loss_fn(pred, y)

            loss.backward()

            optimizer.step()

            train_loss_sum += loss.item() * len(idx)

            train_bar.set_postfix(mse=f"{loss.item():.1f}")

        train_mse = train_loss_sum / len(train_idx)

        # =====================================================
        # Validation
        # =====================================================

        net.eval()

        val_loss_sum = 0.0

        val_bar = tqdm(
            range(0, len(val_idx), batch_size), desc=(f"ConvNeXt Fold {fold_id + 1} Val")
        )

        with torch.inference_mode():
            for start in val_bar:
                idx = val_idx[start : start + batch_size]

                X = one_hot_encode(idx).to(device)

                y = torch.as_tensor(np.asarray(totmat[idx]), dtype=torch.float32, device=device)

                pred = net(X)

                loss = loss_fn(pred, y)

                val_loss_sum += loss.item() * len(idx)

        val_mse = val_loss_sum / len(val_idx)

        print(f"\nEpoch {epoch + 1:02d} | train MSE: {train_mse:.2f} | val MSE: {val_mse:.2f}")

        # =====================================================
        # 10. Early stopping
        # =====================================================

        if val_mse < best_val:
            best_val = val_mse

            best_state = copy.deepcopy(net.state_dict())

            best_epoch = epoch + 1

            no_improve = 0

        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")

            break

    # =========================================================
    # 11. Restore best model
    # =========================================================

    net.load_state_dict(best_state)

    net.eval()

    # =========================================================
    # 12. Final evaluation
    # =========================================================

    preds = []

    trues = []

    with torch.inference_mode():
        eval_bar = tqdm(range(0, len(val_idx), batch_size), desc="Final evaluation")

        for start in eval_bar:
            idx = val_idx[start : start + batch_size]

            X = one_hot_encode(idx).to(device)

            pred = net(X)

            preds.append(pred.cpu().numpy())

            trues.append(np.asarray(totmat[idx], dtype=np.float32))

    preds = np.concatenate(preds)

    trues = np.concatenate(trues)

    # =========================================================
    # 13. Metrics
    # =========================================================

    mse = np.mean((preds - trues) ** 2)

    rmse = np.sqrt(mse)

    r2_bins = np.array([r2_score(trues[:, i], preds[:, i]) for i in range(20)])

    mean_r2 = r2_bins.mean()

    print("\n")
    print("=" * 60)

    print(f"ConvNeXt-dCNN | Fold {fold_id + 1}")

    print("=" * 60)

    print("Best epoch:", best_epoch)

    print("Best val MSE:", best_val)

    print("Final MSE:", mse)

    print("RMSE:", rmse)

    print("Mean R²:", mean_r2)

    print("\nR² each bin:")

    print(r2_bins)

    print("=" * 60)

    # =========================================================
    # 14. Save
    # =========================================================

    RESULT_DIR = str(RUN_ROOT / "results")

    os.makedirs(RESULT_DIR, exist_ok=True)

    np.savez(
        f"{RESULT_DIR}/convnext_fold{fold_id + 1}.npz",
        fold=fold_id + 1,
        best_epoch=best_epoch,
        val_mse=best_val,
        final_mse=mse,
        rmse=rmse,
        mean_r2=mean_r2,
        r2_bins=r2_bins,
    )

    torch.save(best_state, f"{RESULT_DIR}/convnext_fold{fold_id + 1}.pt")

    print("Result saved.")


if __name__ == "__main__":
    main()
