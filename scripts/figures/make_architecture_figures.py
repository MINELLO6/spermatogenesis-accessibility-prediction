#!/usr/bin/env python3
"""Create paper-quality vector architecture figures with a real held-out example."""

from __future__ import annotations

import gzip
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


DATA = Path("/root/sc-motif-open/R")
FINAL = Path("/root/autodl-tmp/final_analysis")
ANALYSIS = Path("/root/autodl-tmp/region_error_analysis")
OUT = ANALYSIS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3976A8"
TEAL = "#2A9D8F"
ORANGE = "#E07A3F"
PURPLE = "#7566A0"
LIGHT = "#F4F6F8"
DARK = "#263238"


def box(ax, xy, wh, text, color=BLUE, fontsize=9, sub=None):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.25, edgecolor=color, facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.57 if sub else 0.50), text, ha="center", va="center", fontsize=fontsize, color=DARK, weight="semibold")
    if sub:
        ax.text(x + w / 2, y + h * 0.22, sub, ha="center", va="center", fontsize=7.2, color="#59636A")
    return patch


def arrow(ax, a, b, color="#69747B"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.15, color=color))


def load_example():
    heldout = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    with open(DATA / "totmat_shape.txt") as h:
        n, t = map(int, h.read().split())
    y_mm = np.memmap(DATA / "totmat_f64.bin", dtype="<f8", mode="r", shape=(n, t), order="F")
    raw = np.load(FINAL / "heldout_ensembles/raw_rc_5fold_ensemble/heldout_predictions_f32.npy", mmap_mode="r")
    motif_pred = np.load(FINAL / "heldout_ensembles/motif_transformer_5fold_ensemble/heldout_predictions_f32.npy", mmap_mode="r")
    good = []
    with gzip.open(ANALYSIS / "region_metrics.csv.gz", "rt") as h:
        next(h)
        for line in h:
            cols = line.rstrip().split(",")
            if cols[-2] == "1":
                good.append((float(cols[5]) + float(cols[11]), int(cols[0])))
    row = min(good)[1] if good else 0
    rid = int(heldout[row])
    y = np.maximum(np.asarray(y_mm[rid], dtype=np.float64), 0)
    rp = np.maximum(np.asarray(raw[row], dtype=np.float64), 0)
    mp = np.maximum(np.asarray(motif_pred[row], dtype=np.float64), 0)
    seqs = np.memmap(DATA / "fullseqs.txt", dtype="S202", mode="r", shape=(n,))
    seq = seqs[rid].decode("ascii", errors="ignore")[:201]

    csr = DATA / "motif_csr"
    offsets = np.load(csr / "offsets.npy", mmap_mode="r")
    mids = np.load(csr / "motif_ids.npy", mmap_mode="r")
    pbin = np.load(csr / "position_bins.npy", mmap_mode="r")
    score = np.load(csr / "scores.npy", mmap_mode="r")
    a, b = int(offsets[rid]), int(offsets[rid + 1])
    m = np.asarray(mids[a:b], dtype=np.int64)
    p = np.asarray(pbin[a:b], dtype=np.int64)
    s = np.asarray(score[a:b], dtype=np.float32)
    return rid, seq, y, rp, mp, m, p, s


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def one_hot_inset(fig, rect, seq):
    ax = fig.add_axes(rect)
    bases = "ACGT"
    arr = np.stack([[base == x for x in seq] for base in bases]).astype(float)
    ax.imshow(arr, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(4), bases, fontsize=6)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def profile_inset(fig, rect, y, pred, label, color):
    ax = fig.add_axes(rect)
    x = np.arange(1, 21)
    ax.plot(x, y, color="black", lw=1.5, label="observed")
    ax.plot(x, pred, color=color, lw=1.5, label=label)
    ax.set_xticks([1, 10, 20])
    ax.tick_params(labelsize=6, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")


def main_workflow(rid, seq, y, raw, motif_pred, mids, pbin, scores):
    fig = plt.figure(figsize=(14.2, 8.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.035, 0.955, "Sequence and motif models for developmental accessibility prediction", fontsize=16, weight="bold", color=DARK)
    ax.text(0.035, 0.925, f"Real locked held-out example: genomic region {rid}", fontsize=9, color="#59636A")

    ax.text(0.035, 0.845, "A  Raw sequence model", fontsize=12, weight="bold", color=BLUE)
    snippet = seq[86:115]
    ax.text(0.045, 0.774, snippet, family="monospace", fontsize=8, color=DARK)
    ax.text(0.045, 0.745, "201 bp DNA", fontsize=7, color="#59636A")
    one_hot_inset(fig, [0.035, 0.61, 0.15, 0.10], snippet)
    ax.text(0.11, 0.595, "4 × 201 one-hot", ha="center", fontsize=7, color="#59636A")
    arrow(ax, (0.19, 0.66), (0.225, 0.66))
    box(ax, (0.225, 0.61), (0.12, 0.11), "Multi-scale CNN", BLUE, sub="kernels 7, 15, 31\n3 × (64 × 201)")
    arrow(ax, (0.345, 0.665), (0.375, 0.665))
    box(ax, (0.375, 0.61), (0.10, 0.11), "Concatenate", BLUE, sub="192 × 201")
    arrow(ax, (0.475, 0.665), (0.505, 0.665))
    box(ax, (0.505, 0.61), (0.105, 0.11), "1 × 1 projection", BLUE, sub="128 × 201")
    arrow(ax, (0.61, 0.665), (0.64, 0.665))
    box(ax, (0.64, 0.61), (0.12, 0.11), "Residual blocks", BLUE, sub="dilations 1, 2, 4, 8")
    arrow(ax, (0.76, 0.665), (0.79, 0.665))
    box(ax, (0.79, 0.61), (0.09, 0.11), "Mean + max", BLUE, sub="256 features")
    arrow(ax, (0.88, 0.665), (0.91, 0.665))
    box(ax, (0.91, 0.61), (0.065, 0.11), "MLP", BLUE, sub="20 bins")
    profile_inset(fig, [0.79, 0.40, 0.185, 0.15], y, raw, "raw", BLUE)
    arrow(ax, (0.942, 0.61), (0.88, 0.555))

    ax.plot([0.035, 0.975], [0.365, 0.365], color="#D8DEE2", lw=1)
    ax.text(0.035, 0.325, "B  Sparse motif Transformer", fontsize=12, weight="bold", color=PURPLE)
    if len(scores):
        take = np.argsort(scores)[-5:][::-1]
        token_text = "\n".join([f"({int(mids[i])}, {int(pbin[i])}, {scores[i]:.2f})" for i in take])
    else:
        token_text = "no motif hits"
    box(ax, (0.04, 0.12), (0.13, 0.15), "Motif-hit tokens", PURPLE, sub="(motif ID, position, score)\nup to 168 tokens")
    ax.text(0.105, 0.095, token_text, ha="center", va="top", fontsize=6.2, family="monospace", color="#59636A")
    arrow(ax, (0.17, 0.195), (0.21, 0.195))
    box(ax, (0.21, 0.14), (0.14, 0.11), "Token embedding", PURPLE, sub="motif + position\n+ score MLP; 128-D")
    arrow(ax, (0.35, 0.195), (0.39, 0.195))
    box(ax, (0.39, 0.14), (0.13, 0.11), "Transformer", PURPLE, sub="4 layers, 4 heads")
    arrow(ax, (0.52, 0.195), (0.56, 0.195))
    box(ax, (0.56, 0.14), (0.13, 0.11), "Region pooling", PURPLE, sub="CLS + mean + max\n384 features")
    arrow(ax, (0.69, 0.195), (0.73, 0.195))
    box(ax, (0.72, 0.14), (0.08, 0.11), "MLP", PURPLE, sub="20 bins")
    profile_inset(fig, [0.835, 0.10, 0.14, 0.16], y, motif_pred, "motif", PURPLE)
    arrow(ax, (0.80, 0.195), (0.835, 0.195))
    ax.text(0.04, 0.025, "All performance estimates use grouped data partitions; the plotted region was not used to fit any fold model.", fontsize=7.4, color="#59636A")
    save(fig, "architecture_raw_and_sparse_motif_with_example")


def dense_factorized(rid, y, motif_pred, mids, pbin, scores):
    fig = plt.figure(figsize=(13.4, 6.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.04, 0.93, "Dense motif matrix and magnitude–shape factorisation", fontsize=16, weight="bold", color=DARK)
    ax.text(0.04, 0.885, f"Real motif scores and profile from held-out region {rid}", fontsize=9, color="#59636A")

    unique = np.unique(mids)
    ranked = []
    for m in unique:
        ranked.append((scores[mids == m].max(), int(m)))
    chosen = [m for _, m in sorted(ranked, reverse=True)[:12]] or list(range(1, 13))
    grid = np.zeros((10, len(chosen)), dtype=float)
    lookup = {m: j for j, m in enumerate(chosen)}
    for m, p, s in zip(mids, pbin, scores):
        if int(m) in lookup and 1 <= int(p) <= 10:
            grid[int(p) - 1, lookup[int(m)]] = max(grid[int(p) - 1, lookup[int(m)]], float(s))
    hm = fig.add_axes([0.04, 0.26, 0.20, 0.50])
    hm.imshow(grid, aspect="auto", cmap="magma", interpolation="nearest")
    hm.set_xlabel("example motif IDs", fontsize=7); hm.set_ylabel("position bin", fontsize=7)
    hm.set_xticks(range(len(chosen)), chosen, rotation=90, fontsize=5.5); hm.set_yticks(range(10), range(1, 11), fontsize=6)
    ax.text(0.14, 0.79, "10 × 1443 score matrix", ha="center", fontsize=9, weight="semibold", color=DARK)
    arrow(ax, (0.25, 0.51), (0.30, 0.51))
    box(ax, (0.30, 0.42), (0.15, 0.18), "Matrix encoder", TEAL, sub="CNN / Transformer\nposition-aware interactions")
    arrow(ax, (0.45, 0.51), (0.50, 0.63), TEAL)
    arrow(ax, (0.45, 0.51), (0.50, 0.37), ORANGE)
    box(ax, (0.50, 0.57), (0.14, 0.13), "Magnitude head", TEAL, sub="Softplus → M ≥ 0")
    box(ax, (0.50, 0.30), (0.14, 0.13), "Shape head", ORANGE, sub="Softmax → q; Σq = 1")
    arrow(ax, (0.64, 0.635), (0.70, 0.54), TEAL)
    arrow(ax, (0.64, 0.365), (0.70, 0.46), ORANGE)
    box(ax, (0.70, 0.43), (0.10, 0.14), "λ = M × q", PURPLE, sub="20 non-negative rates")
    arrow(ax, (0.80, 0.50), (0.84, 0.50), PURPLE)
    profile_inset(fig, [0.84, 0.34, 0.14, 0.27], y, motif_pred, "prediction", PURPLE)
    ax.text(0.31, 0.18, "Loss options", fontsize=9, weight="bold", color=DARK)
    ax.text(0.31, 0.135, "equal-CE: shape-focused", fontsize=8, color=ORANGE)
    ax.text(0.50, 0.135, "Poisson NLL: count-focused", fontsize=8, color=TEAL)
    ax.text(0.70, 0.135, "balanced dual head: joint", fontsize=8, color=PURPLE)
    ax.text(0.04, 0.04, "The dense matrix contains only motif scores; no raw sequence or genomic annotation enters this model.", fontsize=7.5, color="#59636A")
    save(fig, "architecture_dense_motif_magnitude_shape_with_example")


def representation_comparison(rid, mids, pbin, scores):
    fig = plt.figure(figsize=(11.8, 5.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.05, 0.91, "Two representations of exactly the same motif-score input", fontsize=15, weight="bold", color=DARK)
    ax.text(0.05, 0.86, f"Held-out region {rid}", fontsize=9, color="#59636A")
    box(ax, (0.06, 0.52), (0.20, 0.19), "Dense grid", TEAL, sub="10 position bins × 1443 motifs\n14,430 numeric cells")
    box(ax, (0.06, 0.18), (0.20, 0.19), "Sparse token set", PURPLE, sub="only non-zero cells\n(motif, position, score)")
    arrow(ax, (0.26, 0.615), (0.34, 0.615), TEAL)
    arrow(ax, (0.26, 0.275), (0.34, 0.275), PURPLE)
    box(ax, (0.34, 0.52), (0.20, 0.19), "Matrix operations", TEAL, sub="position CNN, projection,\nor matrix Transformer")
    box(ax, (0.34, 0.18), (0.20, 0.19), "Token operations", PURPLE, sub="embedding, self-attention,\nset/region pooling")
    arrow(ax, (0.54, 0.615), (0.62, 0.615), TEAL)
    arrow(ax, (0.54, 0.275), (0.62, 0.275), PURPLE)
    box(ax, (0.62, 0.52), (0.16, 0.19), "Dense experiments", TEAL, sub="preserve all zero locations;\nregular grid inductive bias")
    box(ax, (0.62, 0.18), (0.16, 0.19), "Primary motif model", PURPLE, sub="top-score truncation;\nefficient attention")
    arrow(ax, (0.78, 0.615), (0.84, 0.45), "#69747B")
    arrow(ax, (0.78, 0.275), (0.84, 0.45), "#69747B")
    box(ax, (0.84, 0.37), (0.11, 0.16), "20-bin output", BLUE, sub="same target and splits")
    n_hits = len(scores)
    ax.text(0.05, 0.07, f"This example contains {n_hits:,} motif hits before any sparse-token truncation.", fontsize=8, color="#59636A")
    save(fig, "motif_dense_vs_sparse_representation")


def main():
    rid, seq, y, raw, motif_pred, mids, pbin, scores = load_example()
    main_workflow(rid, seq, y, raw, motif_pred, mids, pbin, scores)
    dense_factorized(rid, y, motif_pred, mids, pbin, scores)
    representation_comparison(rid, mids, pbin, scores)
    (ANALYSIS / "ARCHITECTURE_FIGURES_COMPLETE").write_text("Three vector architecture figures complete.\n")


if __name__ == "__main__":
    main()
