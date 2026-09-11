def main():
    import argparse

    argparse.ArgumentParser(description="interpret final raw").parse_args()
    #!/usr/bin/env python3
    """Locked held-out interpretation for the final Raw+RC model.

    Runs only after final training/evaluation is complete. It selects a balanced set
    of held-out regions by the target peak pseudotime bin, computes integrated
    gradients for the corresponding peak-bin output, and validates a smaller subset
    with exhaustive single-base substitutions.
    """
    import json
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from developmental_accessibility.paths import DATA_ROOT, RUN_ROOT

    ROOT = Path(str(DATA_ROOT))
    FINAL = Path(str(RUN_ROOT / "final_analysis"))
    MODEL_DIR = FINAL / "final_models/raw_rc"
    OUT = FINAL / "interpretation/raw_rc_v2"
    OUT.mkdir(parents=True, exist_ok=True)

    if not (MODEL_DIR / "LOCKED_COMPLETE").exists():
        raise RuntimeError("Final Raw+RC model has not completed held-out evaluation")
    if (OUT / "LOCKED_COMPLETE").exists():
        raise RuntimeError("Interpretation already completed; refusing to overwrite")

    class DilatedResBlock(nn.Module):
        def __init__(self, channels, dilation):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(channels)
            self.bn2 = nn.BatchNorm1d(channels)

        def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + residual)

    class RawMultiScaleResCNN(nn.Module):
        def __init__(self):
            super().__init__()

            def branch(k):
                return nn.Sequential(
                    nn.Conv1d(4, 64, k, padding=k // 2), nn.BatchNorm1d(64), nn.ReLU()
                )

            self.branch7, self.branch15, self.branch31 = branch(7), branch(15), branch(31)
            self.project = nn.Sequential(nn.Conv1d(192, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
            self.blocks = nn.Sequential(*[DilatedResBlock(128, d) for d in (1, 2, 4, 8)])
            self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 20))

        def forward(self, x):
            x = self.project(torch.cat([self.branch7(x), self.branch15(x), self.branch31(x)], 1))
            x = self.blocks(x)
            return self.head(torch.cat([x.mean(2), x.max(2).values], 1))

    device = torch.device("cuda:0")
    checkpoint = torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=False)
    net = RawMultiScaleResCNN()
    net.load_state_dict(checkpoint["model_state"])
    net.to(device).eval()

    with open(ROOT / "totmat_shape.txt") as handle:
        n_regions, n_outputs = map(int, handle.read().split())
    targets = np.memmap(
        ROOT / "totmat_f64.bin", dtype="<f8", mode="r", shape=(n_regions, n_outputs), order="F"
    )
    seqs = np.memmap(ROOT / "fullseqs.txt", dtype="S202", mode="r", shape=(n_regions,))
    heldout = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    pred = np.load(MODEL_DIR / "heldout_predictions_f32.npy", mmap_mode="r")
    truth = np.asarray(targets[heldout], dtype=np.float32)

    def encode(global_indices):
        raw = np.ascontiguousarray(np.asarray(seqs[global_indices]))
        b = raw.view(np.uint8).reshape(len(global_indices), 202)[:, :201]
        x = np.stack([b == ord("A"), b == ord("C"), b == ord("G"), b == ord("T")], 1)
        return torch.from_numpy(x.astype(np.float32, copy=False))

    # Balanced sampling: 25 low-error and 25 high-error regions per target peak bin.
    peak_bin = np.argmax(truth, axis=1)
    region_mse = np.mean(np.square(np.asarray(pred) - truth), axis=1)
    selected_local = []
    selection_rows = []
    for b in range(20):
        candidates = np.flatnonzero(peak_bin == b)
        if len(candidates) == 0:
            continue
        ordered = candidates[np.argsort(region_mse[candidates])]
        k = min(25, len(ordered) // 2)
        if k == 0:
            continue
        chosen = np.concatenate([ordered[:k], ordered[-k:]])
        selected_local.extend(chosen.tolist())
        selection_rows.extend(
            [
                {
                    "local_index": int(x),
                    "global_index": int(heldout[x]),
                    "peak_bin": b + 1,
                    "region_mse": float(region_mse[x]),
                    "error_stratum": "low" if j < k else "high",
                }
                for j, x in enumerate(chosen)
            ]
        )
    selected_local = np.asarray(selected_local, dtype=np.int64)
    selected_global = heldout[selected_local]
    selected_bins = peak_bin[selected_local]

    # Integrated gradients against a neutral 0.25-per-base baseline.
    steps = 32
    batch_size = 32
    all_full = []
    all_position = []
    for start in range(0, len(selected_global), batch_size):
        idx = selected_global[start : start + batch_size]
        out_bins = torch.as_tensor(selected_bins[start : start + batch_size], device=device)
        x = encode(idx).to(device)
        baseline = torch.full_like(x, 0.25)
        grad_sum = torch.zeros_like(x)
        for alpha in torch.linspace(0, 1, steps, device=device):
            z = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
            score = net(z).gather(1, out_bins[:, None]).sum()
            grad = torch.autograd.grad(score, z, retain_graph=False)[0]
            grad_sum += grad
        full_attr = (x - baseline) * grad_sum / steps
        pos_attr = full_attr.sum(dim=1)
        all_full.append(full_attr.detach().cpu().numpy().astype(np.float32))
        all_position.append(pos_attr.detach().cpu().numpy().astype(np.float32))

    full_attr = np.concatenate(all_full)
    position_attr = np.concatenate(all_position)
    np.save(OUT / "selected_local_indices.npy", selected_local)
    np.save(OUT / "selected_global_indices.npy", selected_global)
    np.save(OUT / "integrated_gradients_full.npy", full_attr)
    np.save(OUT / "integrated_gradients_position.npy", position_attr)

    # Exhaustive substitution scan for five low-error representatives per peak bin.
    low_rows = [r for r in selection_rows if r["error_stratum"] == "low"]
    mut_rows = []
    for b in range(1, 21):
        mut_rows.extend([r for r in low_rows if r["peak_bin"] == b][:5])
    mut_global = np.asarray([r["global_index"] for r in mut_rows], dtype=np.int64)
    mut_bins = np.asarray([r["peak_bin"] - 1 for r in mut_rows], dtype=np.int64)
    mutation_delta = np.full((len(mut_global), 201, 4), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for r, (gidx, obin) in enumerate(zip(mut_global, mut_bins)):
            original = encode(np.asarray([gidx])).to(device)
            base_pred = float(net(original)[0, obin])
            original_base = original[0].argmax(dim=0).cpu().numpy()
            mutants, keys = [], []
            for pos in range(201):
                for alt in range(4):
                    if alt == original_base[pos]:
                        continue
                    m = original.clone()
                    m[0, :, pos] = 0
                    m[0, alt, pos] = 1
                    mutants.append(m)
                    keys.append((pos, alt))
            for start in range(0, len(mutants), 256):
                m = torch.cat(mutants[start : start + 256], dim=0)
                vals = net(m)[:, obin].float().cpu().numpy() - base_pred
                for (pos, alt), value in zip(keys[start : start + 256], vals):
                    mutation_delta[r, pos, alt] = value

    np.save(OUT / "mutagenesis_global_indices.npy", mut_global)
    np.save(OUT / "mutagenesis_delta.npy", mutation_delta)
    with open(OUT / "selection.json", "w") as handle:
        json.dump(selection_rows, handle, indent=2)
    with open(OUT / "summary.json", "w") as handle:
        json.dump(
            {
                "n_integrated_gradients": int(len(selected_global)),
                "ig_steps": steps,
                "ig_baseline": "0.25 in each A/C/G/T channel",
                "attributed_output": "target peak pseudotime bin for each region",
                "n_full_mutagenesis": int(len(mut_global)),
            },
            handle,
            indent=2,
        )
    with open(OUT / "LOCKED_COMPLETE", "w") as handle:
        handle.write("Integrated gradients and mutagenesis completed.\n")
    print(json.dumps(json.load(open(OUT / "summary.json")), indent=2))


if __name__ == "__main__":
    main()
