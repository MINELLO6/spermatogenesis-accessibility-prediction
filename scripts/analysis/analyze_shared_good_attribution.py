#!/usr/bin/env python3
"""Compare raw-model attribution patterns inside the raw/motif shape-good intersection."""

from __future__ import annotations

from collections import Counter
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("/root/sc-motif-open/R")
INTERP = Path("/root/autodl-tmp/final_analysis/interpretation/raw_rc_v2")
REGION = Path("/root/autodl-tmp/region_error_analysis")
OUT = REGION / "shared_good_attribution"
OUT.mkdir(parents=True, exist_ok=True)
(REGION / "figures").mkdir(exist_ok=True)


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(REGION / "figures" / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(REGION / "figures" / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


shared = set()
with gzip.open(REGION / "region_metrics.csv.gz", "rt") as h:
    next(h)
    for line in h:
        c = line.rstrip().split(",")
        if c[-2] == "1":
            shared.add(int(c[1]))

global_ids = np.load(INTERP / "selected_global_indices.npy")
ig = np.asarray(np.load(INTERP / "integrated_gradients_full.npy", mmap_mode="r"), dtype=np.float32)
is_shared = np.array([int(x) in shared for x in global_ids], dtype=bool)
if is_shared.sum() < 30:
    raise RuntimeError("Too few precomputed attribution regions in the shared-good intersection")

with open(DATA / "totmat_shape.txt") as h:
    n, _ = map(int, h.read().split())
seqs = np.memmap(DATA / "fullseqs.txt", dtype="S202", mode="r", shape=(n,))
importance_raw = np.abs(ig).sum(axis=1)
importance = importance_raw / np.maximum(importance_raw.mean(axis=1, keepdims=True), 1e-8)
mean_shared = importance[is_shared].mean(axis=0)
mean_other = importance[~is_shared].mean(axis=0)

def top_kmers(mask, k=5):
    counts = Counter()
    values = []
    half = k // 2
    for row in np.where(mask)[0]:
        seq = seqs[int(global_ids[row])].decode("ascii", errors="ignore")[:201]
        valid = np.arange(half, 201 - half)
        best = valid[np.argmax(importance[row, valid])]
        mer = seq[best-half:best+half+1]
        if len(mer) == k and set(mer) <= set("ACGT"):
            counts[mer] += 1
            values.append((mer, float(importance[row, best]), int(global_ids[row]), int(best)))
    return counts, values

shared_k, shared_rows = top_kmers(is_shared)
other_k, other_rows = top_kmers(~is_shared)
n_s = int(is_shared.sum()); n_o = int((~is_shared).sum())
all_kmers = set(shared_k) | set(other_k)
enrichment = []
for mer in all_kmers:
    ps = (shared_k[mer] + 0.5) / (n_s + 1)
    po = (other_k[mer] + 0.5) / (n_o + 1)
    enrichment.append((np.log2(ps / po), mer, shared_k[mer], other_k[mer], ps, po))
enrichment.sort(reverse=True)

with open(OUT / "top_attributed_5mer_enrichment.csv", "w") as h:
    h.write("kmer,shared_count,other_count,shared_prevalence,other_prevalence,log2_enrichment\n")
    for e, mer, cs, co, ps, po in enrichment:
        h.write(f"{mer},{cs},{co},{ps:.8g},{po:.8g},{e:.8g}\n")

with open(OUT / "top_attributed_windows.csv", "w") as h:
    h.write("group,kmer,absolute_attribution,global_region,centre_position\n")
    for group, rows in (("shared_shape_good", shared_rows), ("other_selected", other_rows)):
        for mer, value, rid, pos in rows:
            h.write(f"{group},{mer},{value:.8g},{rid},{pos}\n")

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
x = np.arange(1, 202)
axes[0].plot(x, mean_shared, lw=2, color="#2A9D8F", label=f"shared shape-good (n={n_s})")
axes[0].plot(x, mean_other, lw=2, color="#7566A0", label=f"other IG sample (n={n_o})")
axes[0].set_xlabel("sequence position")
axes[0].set_ylabel("relative absolute integrated gradient\n(each region mean-normalised)")
axes[0].legend(frameon=False, fontsize=8)
axes[0].spines[["top", "right"]].set_visible(False)

top = [row for row in enrichment if row[2] >= 4 and row[2] + row[3] >= 6][:15][::-1]
axes[1].barh([r[1] for r in top], [r[0] for r in top], color="#3B82A0")
axes[1].axvline(0, color="#7A858B", lw=0.8)
axes[1].set_xlabel("log2 enrichment of top-attribution 5-mer")
axes[1].set_ylabel("sequence centred on maximum attribution")
axes[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "raw_attribution_shared_good_patterns")

(OUT / "summary.txt").write_text(
    f"Precomputed IG regions: {len(global_ids)}\n"
    f"Shared raw/motif shape-good IG regions: {n_s}\n"
    f"Other IG regions: {n_o}\n"
    "K-mer enrichment is descriptive and is not a motif-discovery significance test.\n"
)
print((OUT / "summary.txt").read_text())
