#!/usr/bin/env python3
import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/root/sc-motif-open/R')
OUTROOT = Path('/root/autodl-tmp/final_analysis/final_models')

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['raw_rc', 'raw_no_rc'], required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch-size', type=int, default=4096)
parser.add_argument('--seed', type=int, default=2026)
args = parser.parse_args()

out = OUTROOT / args.variant
out.mkdir(parents=True, exist_ok=True)
if (out / 'LOCKED_COMPLETE').exists():
    raise SystemExit(f'{args.variant} already completed; refusing to overwrite')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:0')

with open(ROOT / 'totmat_shape.txt') as handle:
    n_regions, n_outputs = map(int, handle.read().split())
targets = np.memmap(ROOT / 'totmat_f64.bin', dtype='<f8', mode='r',
                    shape=(n_regions, n_outputs), order='F')
seqs = np.memmap(ROOT / 'fullseqs.txt', dtype='S202', mode='r', shape=(n_regions,))
with open(ROOT / 'folds.pkl', 'rb') as handle:
    folds = pickle.load(handle)
development = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for _, v in folds]))
heldout = np.load('/root/autodl-tmp/final_analysis/heldout_test_idx_LOCKED.npy')
assert len(development) == 4_366_231 and len(heldout) == 485_229
assert np.intersect1d(development, heldout).size == 0

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
            return nn.Sequential(nn.Conv1d(4, 64, k, padding=k // 2), nn.BatchNorm1d(64), nn.ReLU())
        self.branch7 = branch(7)
        self.branch15 = branch(15)
        self.branch31 = branch(31)
        self.project = nn.Sequential(nn.Conv1d(192, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.blocks = nn.Sequential(*[DilatedResBlock(128, d) for d in (1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 20))
    def forward(self, x):
        x = self.project(torch.cat([self.branch7(x), self.branch15(x), self.branch31(x)], dim=1))
        x = self.blocks(x)
        return self.head(torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1))

def encode(indices):
    raw = np.ascontiguousarray(np.asarray(seqs[indices]))
    bases = raw.view(np.uint8).reshape(len(indices), 202)[:, :201]
    x = np.stack([bases == ord('A'), bases == ord('C'), bases == ord('G'), bases == ord('T')], axis=1)
    return torch.from_numpy(x.astype(np.float32, copy=False))

def augment(x, probability):
    if probability <= 0:
        return x
    mask = torch.rand(x.shape[0], device=x.device) < probability
    if mask.any():
        x[mask] = torch.flip(x[mask][:, [3, 2, 1, 0], :], dims=[2])
    return x

net = RawMultiScaleResCNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda')
rc_prob = 0.5 if args.variant == 'raw_rc' else 0.0
history = []

print(json.dumps({'variant': args.variant, 'epochs': args.epochs, 'batch_size': args.batch_size,
                  'development_n': len(development), 'heldout_n': len(heldout),
                  'rc_probability': rc_prob, 'gpu': torch.cuda.get_device_name(0)}, indent=2), flush=True)

start_all = time.time()
for epoch in range(1, args.epochs + 1):
    net.train()
    shuffled = np.random.permutation(development)
    sum_loss = 0.0
    seen = 0
    started = time.time()
    for start in range(0, len(shuffled), args.batch_size):
        idx = shuffled[start:start + args.batch_size]
        x = augment(encode(idx).to(device, non_blocking=True), rc_prob)
        y = torch.as_tensor(np.asarray(targets[idx]), dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            pred = net(x)
            loss = F.mse_loss(pred.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sum_loss += float(loss) * len(idx)
        seen += len(idx)
    row = {'epoch': epoch, 'train_mse': sum_loss / seen, 'seconds': time.time() - started}
    history.append(row)
    print(json.dumps(row), flush=True)

# The held-out outcomes are accessed here for the first and only evaluation.
net.eval()
pred_path = out / 'heldout_predictions_f32.npy'
preds = np.lib.format.open_memmap(pred_path, mode='w+', dtype=np.float32,
                                  shape=(len(heldout), n_outputs))
sse = np.zeros(n_outputs, dtype=np.float64)
sum_y = np.zeros(n_outputs, dtype=np.float64)
sum_y2 = np.zeros(n_outputs, dtype=np.float64)
with torch.inference_mode():
    for start in range(0, len(heldout), args.batch_size):
        idx = heldout[start:start + args.batch_size]
        x = encode(idx).to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            p = net(x).float().cpu().numpy()
        truth = np.asarray(targets[idx], dtype=np.float32)
        preds[start:start + len(idx)] = p
        diff = p.astype(np.float64) - truth.astype(np.float64)
        sse += np.square(diff).sum(axis=0)
        sum_y += truth.sum(axis=0, dtype=np.float64)
        sum_y2 += np.square(truth.astype(np.float64)).sum(axis=0)
preds.flush()
sst = sum_y2 - np.square(sum_y) / len(heldout)
r2_bins = 1 - sse / sst
mse = sse.sum() / (len(heldout) * n_outputs)
metrics = {'variant': args.variant, 'epochs': args.epochs, 'seed': args.seed,
           'mse': float(mse), 'rmse': float(math.sqrt(mse)),
           'mean_r2': float(r2_bins.mean()), 'r2_bins': r2_bins.tolist(),
           'training_seconds': time.time() - start_all, 'history': history,
           'heldout_evaluations': 1}
torch.save({'model_state': net.state_dict(), 'variant': args.variant,
            'epochs': args.epochs, 'seed': args.seed}, out / 'model.pt')
with open(out / 'metrics.json', 'w') as handle:
    json.dump(metrics, handle, indent=2)
with open(out / 'LOCKED_COMPLETE', 'w') as handle:
    handle.write('Final development training and single heldout evaluation complete.\n')
print(json.dumps(metrics, indent=2), flush=True)
