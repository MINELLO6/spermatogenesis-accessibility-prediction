#!/usr/bin/env python3
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/root/sc-motif-open/R')
OUT = Path('/root/autodl-tmp/final_analysis')
OUT.mkdir(parents=True, exist_ok=True)

N = 4_851_460
T = 20
CHUNK = 250_000

print('Loading genomic positions and reconstructing groups...', flush=True)
peaks = pd.read_csv(ROOT / 'peaks.tsv', sep='\t', usecols=['chr', 'median_position'])
peaks['median_position'] = pd.to_numeric(peaks['median_position'])
ordered = peaks.reset_index().sort_values(['chr', 'median_position']).reset_index(drop=True)
gap = ordered.groupby('chr')['median_position'].diff()
new_group = (
    ordered['chr'].ne(ordered['chr'].shift()) |
    gap.isna() |
    gap.gt(200)
)
ordered_groups = new_group.cumsum().to_numpy(dtype=np.int64)
groups = np.empty(N, dtype=np.int64)
groups[ordered['index'].to_numpy(dtype=np.int64)] = ordered_groups

with open(ROOT / 'folds.pkl', 'rb') as handle:
    folds = pickle.load(handle)
fold_val_union = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for _, v in folds]))
all_idx = np.arange(N, dtype=np.int64)
dev_idx = fold_val_union
test_idx = np.setdiff1d(all_idx, dev_idx, assume_unique=True)
if np.intersect1d(dev_idx, test_idx).size:
    raise RuntimeError('Development/test overlap detected')
group_overlap = np.intersect1d(np.unique(groups[dev_idx]), np.unique(groups[test_idx])).size
if group_overlap:
    raise RuntimeError(f'Development/test grouping overlap detected: {group_overlap}')
np.save(OUT / 'heldout_test_idx_LOCKED.npy', test_idx)
np.save(OUT / 'region_groups.npy', groups)

print('Computing development-only target diagnostics...', flush=True)
y = np.memmap(ROOT / 'totmat_f64.bin', dtype='<f8', mode='r', shape=(N, T), order='F')

def summarize(indices):
    count = 0
    total = np.zeros(T, dtype=np.float64)
    total2 = np.zeros(T, dtype=np.float64)
    zeros = np.zeros(T, dtype=np.int64)
    minimum = np.full(T, np.inf)
    maximum = np.full(T, -np.inf)
    for start in range(0, len(indices), CHUNK):
        block = np.asarray(y[indices[start:start + CHUNK]], dtype=np.float64)
        finite = np.isfinite(block)
        safe = np.where(finite, block, 0.0)
        count += block.shape[0]
        total += safe.sum(axis=0)
        total2 += np.square(safe).sum(axis=0)
        zeros += np.sum(block == 0, axis=0)
        minimum = np.minimum(minimum, np.nanmin(block, axis=0))
        maximum = np.maximum(maximum, np.nanmax(block, axis=0))
    mean = total / count
    var = np.maximum(total2 / count - mean ** 2, 0)
    return dict(n=int(count), mean=mean, variance=var, sd=np.sqrt(var),
                zero_fraction=zeros / count, minimum=minimum, maximum=maximum)

dev_summary = summarize(dev_idx)
fold_summaries = [summarize(np.asarray(v, dtype=np.int64)) for _, v in folds]

print('Computing development target correlation...', flush=True)
# Exact cross-products require only a 20x20 accumulator.
sum_vec = np.zeros(T, dtype=np.float64)
cross = np.zeros((T, T), dtype=np.float64)
n_corr = 0
for start in range(0, len(dev_idx), CHUNK):
    block = np.asarray(y[dev_idx[start:start + CHUNK]], dtype=np.float64)
    block = np.nan_to_num(block)
    sum_vec += block.sum(axis=0)
    cross += block.T @ block
    n_corr += block.shape[0]
cov = (cross - np.outer(sum_vec, sum_vec) / n_corr) / max(n_corr - 1, 1)
den = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
corr = np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)
np.save(OUT / 'development_target_correlation.npy', corr)

print('Computing fold composition diagnostics...', flush=True)
pvals = pd.read_csv(ROOT / 'peaktest.tsv', sep='\t', usecols=['p.value'])['p.value'].to_numpy()
seqs = np.memmap(ROOT / 'fullseqs.txt', dtype='S202', mode='r', shape=(N,))

rows = []
for fold_id, (_, val) in enumerate(folds, start=1):
    val = np.asarray(val, dtype=np.int64)
    chr_counts = peaks.iloc[val]['chr'].astype(str).value_counts(normalize=True)
    # Exact GC fraction, processed in chunks.
    gc_count = 0
    base_count = 0
    for start in range(0, len(val), CHUNK):
        raw = np.ascontiguousarray(seqs[val[start:start + CHUNK]])
        bases = raw.view(np.uint8).reshape(-1, 202)[:, :201]
        gc_count += int(np.sum((bases == ord('G')) | (bases == ord('C'))))
        base_count += bases.size
    s = fold_summaries[fold_id - 1]
    rows.append({
        'fold': fold_id,
        'n': len(val),
        'n_groups': int(np.unique(groups[val]).size),
        'gc_fraction': gc_count / base_count,
        'p_lt_0.001_fraction': float(np.mean(np.isfinite(pvals[val]) & (pvals[val] < 0.001))),
        'target_grand_mean': float(np.mean(s['mean'])),
        'target_mean_sd_across_bins': float(np.std(s['mean'], ddof=1)),
        'target_grand_variance': float(np.mean(s['variance'])),
        'largest_chr_fraction': float(chr_counts.iloc[0]),
        'largest_chr': str(chr_counts.index[0]),
    })

fold_df = pd.DataFrame(rows)
fold_df.to_csv(OUT / 'fold_diagnostics.csv', index=False)

target_df = pd.DataFrame({
    'bin': np.arange(1, T + 1),
    'mean': dev_summary['mean'],
    'variance': dev_summary['variance'],
    'sd': dev_summary['sd'],
    'zero_fraction': dev_summary['zero_fraction'],
    'minimum': dev_summary['minimum'],
    'maximum': dev_summary['maximum'],
})
target_df.to_csv(OUT / 'development_target_diagnostics.csv', index=False)

with open(OUT / 'split_audit.json', 'w') as handle:
    json.dump({
        'n_total': N,
        'n_groups': int(np.unique(groups).size),
        'n_development': int(len(dev_idx)),
        'n_heldout_locked': int(len(test_idx)),
        'development_matches_folds': True,
        'group_overlap': int(group_overlap),
        'test_targets_inspected': False,
        'split_source': 'strict complement of union of folds.pkl validation indices',
    }, handle, indent=2)

plt.figure(figsize=(7.2, 4.4))
plt.plot(target_df['bin'], target_df['mean'], marker='o', label='Mean')
plt.fill_between(target_df['bin'],
                 target_df['mean'] - target_df['sd'],
                 target_df['mean'] + target_df['sd'], alpha=0.2, label='Mean +/- SD')
plt.xlabel('Pseudotime bin')
plt.ylabel('Accessibility')
plt.xticks(range(1, 21))
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT / 'development_target_mean_sd.png', dpi=220)
plt.close()

plt.figure(figsize=(6.2, 5.2))
plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(label='Pearson correlation')
plt.xlabel('Pseudotime bin')
plt.ylabel('Pseudotime bin')
plt.xticks(range(20), range(1, 21), fontsize=7)
plt.yticks(range(20), range(1, 21), fontsize=7)
plt.tight_layout()
plt.savefig(OUT / 'development_target_correlation.png', dpi=220)
plt.close()

print('\nSPLIT AUDIT')
print(json.dumps(json.load(open(OUT / 'split_audit.json')), indent=2))
print('\nFOLD DIAGNOSTICS')
print(fold_df.to_string(index=False))
print('\nTARGET DIAGNOSTICS')
print(target_df.to_string(index=False))
print('\nOutputs:', OUT, flush=True)
