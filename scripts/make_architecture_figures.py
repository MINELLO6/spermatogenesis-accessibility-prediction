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
    nt_pred = np.load(FINAL / "heldout_ensembles/frozen_nt_attnpool_5fold_ensemble/heldout_predictions_f32.npy", mmap_mode="r")
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
    npred = np.maximum(np.asarray(nt_pred[row], dtype=np.float64), 0)
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
    return rid, seq, y, rp, npred, mp, m, p, s


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


def profile_inset(fig, rect, y, pred, label, color, ylim=None):
    ax = fig.add_axes(rect)
    x = np.arange(1, 21)
    ax.plot(x, y, color="black", lw=1.5, label="observed")
    ax.plot(x, pred, color=color, lw=1.5, label=label)
    ax.set_xticks([1, 10, 20])
    if ylim is not None:
        ax.set_ylim(*ylim)
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


def three_method_overview_legacy(rid, seq, y, raw, nt_pred, motif_pred, mids, pbin, scores):
    """Create a compact three-row overview of the principal model classes."""
    fig = plt.figure(figsize=(15.2, 9.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.035, 0.958, "Three routes from local DNA to a developmental accessibility profile",
            fontsize=17, weight="bold", color=DARK)
    ax.text(0.035, 0.926,
            f"The same real locked held-out region ({rid}) is shown in every row; coloured lines are model predictions.",
            fontsize=9, color="#59636A")

    rows = [
        (0.665, BLUE, "A", "Task-specific raw sequence", "end-to-end trainable"),
        (0.380, TEAL, "B", "Pretrained Nucleotide Transformer", "frozen backbone + trainable head"),
        (0.095, PURPLE, "C", "Explicit motif representation", "motif annotations + trainable encoder"),
    ]
    for y0, color, letter, title, status in rows:
        panel = FancyBboxPatch((0.025, y0), 0.95, 0.245,
                               boxstyle="round,pad=0.010,rounding_size=0.018",
                               linewidth=1.15, edgecolor=color, facecolor=color + "0D")
        ax.add_patch(panel)
        ax.text(0.042, y0 + 0.210, letter, fontsize=13, weight="bold", color="white",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=color, edgecolor=color))
        ax.text(0.070, y0 + 0.210, title, fontsize=12.2, weight="bold", color=color, va="center")
        ax.text(0.298, y0 + 0.210, status, fontsize=7.5, color="#59636A", va="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, linewidth=0.8))

    # A: task-specific raw sequence model.
    y0 = rows[0][0]
    one_hot_inset(fig, [0.045, y0 + 0.055, 0.115, 0.105], seq[84:117])
    ax.text(0.102, y0 + 0.035, "4 × 201 one-hot DNA", fontsize=7, ha="center", color="#59636A")
    arrow(ax, (0.165, y0 + 0.108), (0.195, y0 + 0.108))
    box(ax, (0.195, y0 + 0.055), (0.125, 0.105), "Multi-scale CNN", BLUE,
        sub="kernels 7, 15, 31\n192 × 201")
    arrow(ax, (0.320, y0 + 0.108), (0.345, y0 + 0.108))
    box(ax, (0.345, y0 + 0.055), (0.115, 0.105), "1 × 1 projection", BLUE,
        sub="128 × 201")
    arrow(ax, (0.460, y0 + 0.108), (0.485, y0 + 0.108))
    box(ax, (0.485, y0 + 0.055), (0.130, 0.105), "Residual stack", BLUE,
        sub="dilations 1, 2, 4, 8")
    arrow(ax, (0.615, y0 + 0.108), (0.640, y0 + 0.108))
    box(ax, (0.640, y0 + 0.055), (0.105, 0.105), "Pool + MLP", BLUE,
        sub="mean ∥ max → 20")
    arrow(ax, (0.745, y0 + 0.108), (0.775, y0 + 0.108))
    profile_inset(fig, [0.780, y0 + 0.035, 0.175, 0.135], y, raw, "raw CNN", BLUE)

    # B: pretrained NT with explicit representation selection.
    y0 = rows[1][0]
    token_ax = fig.add_axes([0.045, y0 + 0.060, 0.115, 0.098]); token_ax.axis("off")
    token_ax.set_xlim(0, 1); token_ax.set_ylim(0, 1)
    token_labels = ["<s>", seq[0:6], seq[6:12], "…", seq[192:198], "</s>"]
    widths = [0.13, 0.18, 0.18, 0.10, 0.18, 0.13]
    x = 0.01
    for label, width in zip(token_labels, widths):
        token_ax.add_patch(FancyBboxPatch((x, 0.30), width, 0.42,
                           boxstyle="round,pad=0.015", facecolor="#EAF5F3", edgecolor=TEAL, lw=0.8))
        token_ax.text(x + width / 2, 0.51, label, ha="center", va="center", fontsize=5.5,
                      family="monospace", color=DARK)
        x += width + 0.025
    ax.text(0.102, y0 + 0.035, "6-mer tokens + mask", fontsize=7, ha="center", color="#59636A")
    arrow(ax, (0.165, y0 + 0.108), (0.195, y0 + 0.108))
    box(ax, (0.195, y0 + 0.055), (0.125, 0.105), "NT-v2-50M", TEAL,
        sub="12 transformer blocks\nbackbone frozen")
    ax.text(0.257, y0 + 0.170, "FROZEN", ha="center", va="center", fontsize=6.3,
            weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#607D8B", edgecolor="none"))
    arrow(ax, (0.320, y0 + 0.108), (0.345, y0 + 0.108))
    box(ax, (0.345, y0 + 0.055), (0.115, 0.105), "Layer selection", TEAL,
        sub="layers 1–12 mixture\nor layer 6")
    arrow(ax, (0.460, y0 + 0.108), (0.485, y0 + 0.108))
    box(ax, (0.485, y0 + 0.055), (0.130, 0.105), "Attention pooling", TEAL,
        sub="weighted mean ∥ max")
    arrow(ax, (0.615, y0 + 0.108), (0.640, y0 + 0.108))
    box(ax, (0.640, y0 + 0.055), (0.105, 0.105), "Regression MLP", TEAL,
        sub="256 → 20 bins")
    arrow(ax, (0.745, y0 + 0.108), (0.775, y0 + 0.108))
    profile_inset(fig, [0.780, y0 + 0.035, 0.175, 0.135], y, nt_pred, "frozen NT", TEAL)

    # C: sparse motif Transformer derived from the complete motif scan.
    y0 = rows[2][0]
    motif_ax = fig.add_axes([0.045, y0 + 0.050, 0.115, 0.115]); motif_ax.axis("off")
    motif_ax.set_xlim(0, 1); motif_ax.set_ylim(0, 1)
    take = np.argsort(scores)[-4:][::-1] if len(scores) else []
    for row, index in enumerate(take):
        motif_ax.text(0.02, 0.87 - row * 0.22,
                      f"ID {int(mids[index]):4d}  pos {int(pbin[index]):2d}  score {scores[index]:.2f}",
                      fontsize=5.8, family="monospace", color=DARK)
        motif_ax.plot([0.02, 0.96], [0.79 - row * 0.22, 0.79 - row * 0.22],
                      color="#E4E0EF", lw=0.7)
    ax.text(0.102, y0 + 0.035, "real motif-hit tokens", fontsize=7, ha="center", color="#59636A")
    arrow(ax, (0.165, y0 + 0.108), (0.195, y0 + 0.108))
    box(ax, (0.195, y0 + 0.055), (0.125, 0.105), "Token embedding", PURPLE,
        sub="motif + position\n+ score; 128-D")
    arrow(ax, (0.320, y0 + 0.108), (0.345, y0 + 0.108))
    box(ax, (0.345, y0 + 0.055), (0.115, 0.105), "Self-attention", PURPLE,
        sub="4 layers × 4 heads")
    arrow(ax, (0.460, y0 + 0.108), (0.485, y0 + 0.108))
    box(ax, (0.485, y0 + 0.055), (0.130, 0.105), "Region summary", PURPLE,
        sub="CLS ∥ mean ∥ max")
    arrow(ax, (0.615, y0 + 0.108), (0.640, y0 + 0.108))
    box(ax, (0.640, y0 + 0.055), (0.105, 0.105), "Regression MLP", PURPLE,
        sub="384 → 20 bins")
    arrow(ax, (0.745, y0 + 0.108), (0.775, y0 + 0.108))
    profile_inset(fig, [0.780, y0 + 0.035, 0.175, 0.135], y, motif_pred, "motif", PURPLE)

    ax.text(0.035, 0.025,
            "All rows use the same grouped partitions and 20-bin target. The dense 10 × 1,443 motif matrix and magnitude–shape objectives are detailed separately.",
            fontsize=7.5, color="#59636A")
    save(fig, "architecture_three_method_overview_with_example")


def three_method_overview(rid, seq, y, raw, nt_pred, motif_pred, mids, pbin, scores):
    """Comparison-oriented overview with explicit tensors and internal operations."""
    fig = plt.figure(figsize=(12.8, 9.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def panel(x, color, letter, title, subtitle):
        ax.add_patch(FancyBboxPatch(
            (x, 0.075), 0.295, 0.775,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            linewidth=1.5, edgecolor=color, facecolor="white"
        ))
        ax.add_patch(FancyBboxPatch(
            (x, 0.790), 0.295, 0.060,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            linewidth=0, facecolor=color
        ))
        ax.text(x + 0.023, 0.820, letter, color="white", fontsize=15, weight="bold",
                ha="center", va="center")
        ax.text(x + 0.052, 0.826, title, color="white", fontsize=12.0, weight="bold",
                ha="left", va="center")
        ax.text(x + 0.052, 0.803, subtitle, color="white", fontsize=8.8,
                ha="left", va="center")

    def stage_label(x, y0, number, text, color):
        ax.text(x, y0, str(number), ha="center", va="center", fontsize=8.5,
                color="white", weight="bold",
                bbox=dict(boxstyle="circle,pad=0.25", facecolor=color, edgecolor="none"))
        ax.text(x + 0.020, y0, text, ha="left", va="center", fontsize=9.0,
                color=DARK, weight="bold")

    def small_box(x, y0, w, h, text, color, fill="white", fs=8.6, lw=1.1):
        ax.add_patch(FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.006,rounding_size=0.008",
            linewidth=lw, edgecolor=color, facecolor=fill
        ))
        ax.text(x + w / 2, y0 + h / 2, text, ha="center", va="center",
                fontsize=fs, color=DARK)

    def down_arrow(x, y1, y2, color="#6F7A80"):
        arrow(ax, (x, y1), (x, y2), color)

    # Shared input and branch point.
    ax.add_patch(FancyBboxPatch(
        (0.075, 0.925), 0.85, 0.055,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.2, edgecolor="#8A959B", facecolor="#F7F8F9"
    ))
    ax.text(0.095, 0.952, "SHARED INPUT  ·  one genomic region", fontsize=9.5, weight="bold", color=DARK, va="center")
    ax.text(0.335, 0.952, seq[55:135] + "…", family="monospace", fontsize=8.2,
            color="#37474F", va="center")
    ax.text(0.895, 0.952, "201 bp", fontsize=9.0, weight="bold", color="#59636A",
            ha="right", va="center")
    for xpos in (0.1775, 0.5000, 0.8225):
        arrow(ax, (0.500, 0.922), (xpos, 0.858), "#8A959B")

    xraw, xnt, xmot = 0.030, 0.3525, 0.675
    panel(xraw, BLUE, "A", "RAW SEQUENCE", "task-specific; trained end to end")
    panel(xnt, TEAL, "B", "NUCLEOTIDE TRANSFORMER", "pretrained backbone; frozen features")
    panel(xmot, PURPLE, "C", "MOTIF MODEL", "explicit motif identity, score and position")

    # ---------------- Raw column ----------------
    stage_label(xraw + 0.020, 0.755, 1, "One-hot encoding", BLUE)
    one_hot_inset(fig, [xraw + 0.040, 0.660, 0.215, 0.070], seq[80:116])
    ax.text(xraw + 0.1475, 0.642, "input tensor  4 × 201", fontsize=8.8,
            ha="center", color="#59636A")
    down_arrow(xraw + 0.1475, 0.636, 0.610)

    stage_label(xraw + 0.020, 0.592, 2, "Parallel motif-scale convolutions", BLUE)
    branch_y = 0.526
    labels = [("k = 7", "64 × 201"), ("k = 15", "64 × 201"), ("k = 31", "64 × 201")]
    for j, (kernel, dims) in enumerate(labels):
        bx = xraw + 0.025 + j * 0.087
        small_box(bx, branch_y, 0.077, 0.047, kernel + "\n" + dims, BLUE, "#EEF5FA", fs=7.7)
    ax.plot([xraw + 0.063, xraw + 0.237], [0.514, 0.514], color=BLUE, lw=1.0)
    for j in range(3):
        bx = xraw + 0.0635 + j * 0.087
        ax.plot([bx, bx], [branch_y, 0.514], color=BLUE, lw=1.0)
    down_arrow(xraw + 0.1475, 0.514, 0.486, BLUE)
    small_box(xraw + 0.066, 0.444, 0.163, 0.040,
              "concatenate → 192 × 201", BLUE, "white", fs=8.2)
    down_arrow(xraw + 0.1475, 0.442, 0.417)

    stage_label(xraw + 0.020, 0.398, 3, "Projection and residual context", BLUE)
    small_box(xraw + 0.025, 0.333, 0.095, 0.047,
              "1 × 1 Conv\n128 × 201", BLUE, "#EEF5FA", fs=7.8)
    arrow(ax, (xraw + 0.120, 0.356), (xraw + 0.143, 0.356), BLUE)
    small_box(xraw + 0.143, 0.323, 0.120, 0.066,
              "4 residual blocks\nk = 3;  d = 1, 2, 4, 8", BLUE, "#EEF5FA", fs=7.7)
    ax.add_patch(FancyArrowPatch(
        (xraw + 0.154, 0.329), (xraw + 0.251, 0.329),
        connectionstyle="arc3,rad=-0.45", arrowstyle="-|>", mutation_scale=8,
        lw=0.9, color=BLUE
    ))
    down_arrow(xraw + 0.1475, 0.319, 0.291)

    stage_label(xraw + 0.020, 0.274, 4, "Region summary and output", BLUE)
    small_box(xraw + 0.025, 0.205, 0.105, 0.050,
              "mean ∥ max\n256 features", BLUE, "white", fs=7.8)
    arrow(ax, (xraw + 0.130, 0.230), (xraw + 0.149, 0.230), BLUE)
    small_box(xraw + 0.149, 0.205, 0.114, 0.050,
              "MLP 256 → 128 → 20", BLUE, "white", fs=7.8)
    shared_ylim = (0, 1.05 * max(float(y.max()), float(raw.max()),
                                 float(nt_pred.max()), float(motif_pred.max())))
    profile_inset(fig, [xraw + 0.050, 0.095, 0.205, 0.085], y, raw, "raw", BLUE,
                  ylim=shared_ylim)

    # ---------------- NT column ----------------
    stage_label(xnt + 0.020, 0.755, 1, "Tokenisation", TEAL)
    tok_ax = fig.add_axes([xnt + 0.038, 0.665, 0.220, 0.062]); tok_ax.axis("off")
    tok_ax.set_xlim(0, 1); tok_ax.set_ylim(0, 1)
    token_labels = ["<s>", seq[0:6], seq[6:12], "…", seq[192:198], "</s>"]
    widths = [0.12, 0.19, 0.19, 0.10, 0.19, 0.12]
    xpos = 0.01
    for label, width in zip(token_labels, widths):
        tok_ax.add_patch(FancyBboxPatch((xpos, 0.20), width, 0.58,
                           boxstyle="round,pad=0.01", facecolor="#EAF7F5",
                           edgecolor=TEAL, lw=0.9))
        tok_ax.text(xpos + width / 2, 0.49, label, ha="center", va="center",
                    fontsize=6.7, family="monospace", color=DARK)
        xpos += width + 0.025
    ax.text(xnt + 0.1475, 0.642, "at most 37 tokens", fontsize=8.8,
            ha="center", color="#59636A")
    down_arrow(xnt + 0.1475, 0.636, 0.610)

    stage_label(xnt + 0.020, 0.592, 2, "Frozen 12-layer backbone", TEAL)
    # Draw a compact transformer tower whose depth is visually explicit.
    for j in range(6):
        yy = 0.523 + j * 0.009
        ax.add_patch(FancyBboxPatch(
            (xnt + 0.053 + j * 0.004, yy), 0.112, 0.042,
            boxstyle="round,pad=0.003,rounding_size=0.004",
            linewidth=0.85, edgecolor="#78909C", facecolor="#ECEFF1"
        ))
    ax.text(xnt + 0.118, 0.557, "NT-v2-50M", ha="center", va="center",
            fontsize=9.0, weight="bold", color=DARK)
    ax.text(xnt + 0.118, 0.538, "12 Transformer blocks", ha="center", va="center",
            fontsize=7.2, color="#59636A")
    ax.text(xnt + 0.120, 0.503, "FROZEN", ha="center", va="center", fontsize=7.0,
            weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#607D8B", edgecolor="none"))
    arrow(ax, (xnt + 0.180, 0.548), (xnt + 0.207, 0.548), TEAL)
    small_box(xnt + 0.207, 0.521, 0.063, 0.058,
              "13 tensors\nL × 512", TEAL, "#EAF7F5", fs=7.5)
    down_arrow(xnt + 0.1475, 0.496, 0.459)

    stage_label(xnt + 0.020, 0.440, 3, "Choose transferable depth", TEAL)
    small_box(xnt + 0.025, 0.360, 0.115, 0.058,
              "single layer\n(best: layer 6)", TEAL, "white", fs=7.7)
    ax.text(xnt + 0.151, 0.389, "or", fontsize=8.5, color="#59636A", ha="center")
    small_box(xnt + 0.165, 0.350, 0.105, 0.078,
              "Σ softmax(αₗ) Hₗ\nlearned mix, layers 1–12", TEAL, "#EAF7F5", fs=7.4)
    ax.text(xnt + 0.1475, 0.334, "selected token tensor  L × 512", fontsize=8.2,
            color="#59636A", ha="center")
    down_arrow(xnt + 0.1475, 0.328, 0.298)

    stage_label(xnt + 0.020, 0.279, 4, "Attention pooling and output", TEAL)
    small_box(xnt + 0.025, 0.205, 0.105, 0.055,
              "512 → 256\ntoken attention", TEAL, "white", fs=7.7)
    arrow(ax, (xnt + 0.130, 0.232), (xnt + 0.149, 0.232), TEAL)
    small_box(xnt + 0.149, 0.195, 0.121, 0.075,
              "weighted mean ∥ max\n512 → 256 → 128 → 20", TEAL, "white", fs=7.4)
    profile_inset(fig, [xnt + 0.050, 0.095, 0.205, 0.085], y, nt_pred, "NT", TEAL,
                  ylim=shared_ylim)

    # ---------------- Motif column ----------------
    stage_label(xmot + 0.020, 0.755, 1, "Motif scan as a regular matrix", PURPLE)
    unique = np.unique(mids)
    ranked = sorted([(scores[mids == m].max(), int(m)) for m in unique], reverse=True)
    chosen = [m for _, m in ranked[:12]] or list(range(1, 13))
    grid = np.zeros((10, len(chosen)), dtype=float)
    lookup = {m: j for j, m in enumerate(chosen)}
    for m, p, score in zip(mids, pbin, scores):
        if int(m) in lookup and 1 <= int(p) <= 10:
            grid[int(p) - 1, lookup[int(m)]] = max(grid[int(p) - 1, lookup[int(m)]], float(score))
    hm = fig.add_axes([xmot + 0.055, 0.655, 0.185, 0.082])
    hm.imshow(grid, aspect="auto", cmap="Purples", vmin=0, vmax=max(1.0, float(grid.max())),
              interpolation="nearest")
    hm.set_xticks([]); hm.set_yticks([0, 4, 9], [1, 5, 10], fontsize=6.5)
    hm.set_ylabel("position", fontsize=6.5, labelpad=1)
    for spine in hm.spines.values(): spine.set_visible(False)
    ax.text(xmot + 0.1475, 0.642, "shown: 12 motifs  ·  model input: 10 × 1,443",
            fontsize=8.5, ha="center", color="#59636A")
    down_arrow(xmot + 0.1475, 0.636, 0.610)

    stage_label(xmot + 0.020, 0.592, 2, "Sparse motif-hit tokens", PURPLE)
    take = np.argsort(scores)[-4:][::-1] if len(scores) else []
    token_y = 0.555
    for row, index in enumerate(take):
        text_value = f"({int(mids[index])}, {int(pbin[index])}, {scores[index]:.2f})"
        small_box(xmot + 0.025 + (row % 2) * 0.125,
                  token_y - (row // 2) * 0.045, 0.115, 0.034,
                  text_value, PURPLE, "#F3F0F8", fs=7.0, lw=0.9)
    ax.text(xmot + 0.1475, 0.465, "token = (motif ID, position bin, score); up to 168 hits",
            fontsize=7.7, ha="center", color="#59636A")
    down_arrow(xmot + 0.1475, 0.458, 0.430)

    stage_label(xmot + 0.020, 0.412, 3, "Embed and model motif interactions", PURPLE)
    small_box(xmot + 0.025, 0.337, 0.105, 0.055,
              "ID + position\n+ score → N × 128", PURPLE, "white", fs=7.5)
    arrow(ax, (xmot + 0.130, 0.364), (xmot + 0.149, 0.364), PURPLE)
    small_box(xmot + 0.149, 0.327, 0.121, 0.075,
              "4 Transformer blocks\n4-head self-attention", PURPLE, "#F3F0F8", fs=7.5)
    down_arrow(xmot + 0.1475, 0.320, 0.292)

    stage_label(xmot + 0.020, 0.274, 4, "Region summary and output", PURPLE)
    small_box(xmot + 0.025, 0.205, 0.105, 0.050,
              "CLS ∥ mean ∥ max\n384 features", PURPLE, "white", fs=7.6)
    arrow(ax, (xmot + 0.130, 0.230), (xmot + 0.149, 0.230), PURPLE)
    small_box(xmot + 0.149, 0.205, 0.121, 0.050,
              "MLP 384 → 128 → 20", PURPLE, "white", fs=7.6)
    profile_inset(fig, [xmot + 0.050, 0.095, 0.205, 0.085], y, motif_pred, "motif", PURPLE,
                  ylim=shared_ylim)

    # Legend and output semantics.
    ax.text(0.032, 0.041, "Notation", fontsize=8.5, weight="bold", color=DARK)
    ax.add_patch(FancyBboxPatch((0.100, 0.029), 0.035, 0.019, boxstyle="round,pad=0.003",
                               facecolor="#ECEFF1", edgecolor="#78909C", lw=0.8))
    ax.text(0.140, 0.039, "frozen", fontsize=7.6, color="#59636A", va="center")
    ax.add_patch(FancyBboxPatch((0.200, 0.029), 0.035, 0.019, boxstyle="round,pad=0.003",
                               facecolor="white", edgecolor=BLUE, lw=0.9))
    ax.text(0.240, 0.039, "trainable", fontsize=7.6, color="#59636A", va="center")
    ax.text(0.330, 0.039, "C × L = channels/features × sequence or token length",
            fontsize=7.6, color="#59636A", va="center")
    ax.text(0.965, 0.039,
            "Output in every panel: 20 accessibility values across pseudotime",
            fontsize=7.8, color=DARK, weight="bold", ha="right", va="center")

    save(fig, "architecture_three_method_overview_with_example")


def detailed_raw_architecture(rid, seq, y, raw):
    fig = plt.figure(figsize=(14.2, 8.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.035, 0.952, "A  Task-specific multi-scale residual CNN", fontsize=17,
            weight="bold", color=BLUE)
    ax.text(0.035, 0.918,
            "Complete 201-bp sequence is learned end to end; dimensions below exclude the batch axis.",
            fontsize=9.5, color="#59636A")

    # Main flow.
    one_hot_inset(fig, [0.035, 0.675, 0.115, 0.120], seq[76:121])
    ax.text(0.0925, 0.648, "one-hot DNA\n4 × 201", ha="center", fontsize=8.5, color=DARK)
    arrow(ax, (0.155, 0.735), (0.188, 0.735), BLUE)

    ax.add_patch(FancyBboxPatch((0.188, 0.635), 0.195, 0.200,
                               boxstyle="round,pad=0.010", facecolor="#EEF5FA",
                               edgecolor=BLUE, lw=1.4))
    ax.text(0.2855, 0.807, "parallel convolutional stem", ha="center",
            fontsize=10.5, weight="bold", color=DARK)
    for j, (k, pad) in enumerate(((7, 3), (15, 7), (31, 15))):
        yy = 0.748 - j * 0.050
        box(ax, (0.205, yy), (0.160, 0.039), f"Conv1D  4→64  k={k}  pad={pad}", BLUE, fontsize=7.7)
    ax.text(0.2855, 0.648, "each branch: BN → ReLU → 64 × 201",
            ha="center", fontsize=7.6, color="#59636A")
    arrow(ax, (0.383, 0.735), (0.416, 0.735), BLUE)

    box(ax, (0.416, 0.675), (0.115, 0.120), "Concatenate", BLUE, fontsize=9.2,
        sub="192 × 201")
    arrow(ax, (0.531, 0.735), (0.558, 0.735), BLUE)
    box(ax, (0.558, 0.675), (0.115, 0.120), "1 × 1 Conv", BLUE, fontsize=9.2,
        sub="192→128; BN; ReLU\n128 × 201")
    arrow(ax, (0.673, 0.735), (0.700, 0.735), BLUE)

    ax.add_patch(FancyBboxPatch((0.700, 0.635), 0.125, 0.200,
                               boxstyle="round,pad=0.010", facecolor="#EEF5FA",
                               edgecolor=BLUE, lw=1.4))
    ax.text(0.7625, 0.807, "residual context", ha="center", fontsize=9.8,
            weight="bold", color=DARK)
    for j, d in enumerate((1, 2, 4, 8)):
        yy = 0.748 - j * 0.040
        box(ax, (0.718, yy), (0.089, 0.030), f"ResBlock  d={d}", BLUE, fontsize=7.2)
    ax.text(0.7625, 0.650, "128 × 201", ha="center", fontsize=7.6, color="#59636A")
    arrow(ax, (0.825, 0.735), (0.850, 0.735), BLUE)
    box(ax, (0.850, 0.675), (0.115, 0.120), "Global pooling", BLUE, fontsize=9.0,
        sub="mean ∥ max\n256 features")
    arrow(ax, (0.907, 0.675), (0.907, 0.620), BLUE)
    box(ax, (0.850, 0.510), (0.115, 0.100), "Regression head", BLUE, fontsize=8.8,
        sub="256 → 128 → 20")
    profile_inset(fig, [0.815, 0.305, 0.165, 0.155], y, raw, "prediction", BLUE,
                  ylim=(0, 1.05 * max(float(y.max()), float(raw.max()))))
    ax.text(0.8975, 0.282, f"real held-out region {rid}", ha="center", fontsize=7.5,
            color="#59636A")

    # Detailed residual block inset.
    ax.add_patch(FancyBboxPatch((0.035, 0.080), 0.735, 0.445,
                               boxstyle="round,pad=0.012", facecolor="#FAFBFC",
                               edgecolor="#B0BEC5", lw=1.2))
    ax.text(0.055, 0.490, "Inside one dilated residual block", fontsize=11.5,
            weight="bold", color=DARK)
    ax.text(0.055, 0.462, "The sequence length and 128 channels are preserved throughout.",
            fontsize=8.3, color="#59636A")
    yb = 0.355
    details = [
        (0.060, 0.090, "input\n128 × 201"),
        (0.180, 0.115, "Conv1D\nk=3, dilation d"),
        (0.325, 0.090, "BN\nReLU"),
        (0.445, 0.115, "Conv1D\nk=3, dilation d"),
        (0.590, 0.070, "BN"),
        (0.690, 0.055, "+"),
    ]
    for x0, w, label in details:
        box(ax, (x0, yb), (w, 0.075), label, BLUE, fontsize=7.8)
    for a, b in ((0.150, 0.180), (0.295, 0.325), (0.415, 0.445), (0.560, 0.590),
                 (0.660, 0.690)):
        arrow(ax, (a, yb + 0.037), (b, yb + 0.037), BLUE)
    ax.add_patch(FancyArrowPatch((0.105, yb), (0.717, yb),
                                connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                                mutation_scale=10, lw=1.4, color=BLUE))
    arrow(ax, (0.745, yb + 0.037), (0.755, yb + 0.037), BLUE)
    ax.text(0.754, yb + 0.037, "ReLU", fontsize=7.8, color=DARK, va="center")

    ax.text(0.060, 0.285, "dilated sampling around one output position", fontsize=8.8,
            weight="bold", color=DARK)
    centers = np.arange(11)
    for row, d in enumerate((1, 2, 4, 8)):
        yy = 0.245 - row * 0.040
        ax.text(0.065, yy, f"d={d}", fontsize=7.7, color="#59636A", va="center")
        for col in centers:
            color = BLUE if col in {5 - d if 5 - d >= 0 else 5, 5,
                                    5 + d if 5 + d < len(centers) else 5} else "#E3E8EB"
            ax.add_patch(plt.Circle((0.125 + col * 0.025, yy), 0.0075,
                                    facecolor=color, edgecolor="none"))
    ax.text(0.440, 0.250, "Training-time reverse-complement augmentation", fontsize=8.8,
            weight="bold", color=DARK)
    ax.text(0.440, 0.215, "A ↔ T,  C ↔ G,  position order reversed", fontsize=8.1,
            color="#59636A")
    ax.text(0.440, 0.175, "Objective: MSE on all 20 bins", fontsize=8.1, color="#59636A")
    ax.text(0.440, 0.140, "No genomic coordinates or annotations", fontsize=8.1,
            color="#59636A")
    save(fig, "architecture_raw_sequence_detailed")


def detailed_nt_architecture(rid, seq, y, nt_pred):
    fig = plt.figure(figsize=(14.2, 8.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.035, 0.952, "B  Frozen Nucleotide Transformer with layer selection",
            fontsize=17, weight="bold", color=TEAL)
    ax.text(0.035, 0.918,
            "The pretrained 50M-parameter backbone is frozen; layer mixing, token pooling and regression are trained.",
            fontsize=9.5, color="#59636A")

    # Main path.
    ax.text(0.045, 0.790, seq[30:105] + "…", family="monospace", fontsize=7.8,
            color=DARK)
    ax.text(0.045, 0.758, "201-bp DNA", fontsize=8.0, color="#59636A")
    arrow(ax, (0.145, 0.735), (0.185, 0.735), TEAL)
    box(ax, (0.185, 0.675), (0.120, 0.120), "6-mer tokenizer", TEAL, fontsize=9.2,
        sub="special tokens + mask\nL ≤ 37")
    arrow(ax, (0.305, 0.735), (0.340, 0.735), TEAL)

    ax.add_patch(FancyBboxPatch((0.340, 0.625), 0.180, 0.220,
                               boxstyle="round,pad=0.010", facecolor="#ECEFF1",
                               edgecolor="#78909C", lw=1.4))
    ax.text(0.430, 0.818, "NT-v2-50M  ·  FROZEN", fontsize=10.0, weight="bold",
            ha="center", color=DARK)
    for j in range(12):
        yy = 0.650 + j * 0.012
        fill = "#D7EEEA" if j == 5 else "white"
        edge = TEAL if j == 5 else "#90A4AE"
        ax.add_patch(FancyBboxPatch((0.363 + j * 0.004, yy), 0.120, 0.034,
                                   boxstyle="round,pad=0.003", facecolor=fill,
                                   edgecolor=edge, lw=0.8))
    ax.text(0.430, 0.633, "embedding + 12 Transformer blocks", fontsize=7.5,
            ha="center", color="#59636A")
    arrow(ax, (0.520, 0.735), (0.555, 0.735), TEAL)
    box(ax, (0.555, 0.655), (0.135, 0.160), "Hidden-state stack", TEAL, fontsize=9.0,
        sub="$H_0, H_1, …, H_{12}$\neach $L × 512$")
    arrow(ax, (0.690, 0.735), (0.725, 0.735), TEAL)
    box(ax, (0.725, 0.655), (0.130, 0.160), "Select depth", TEAL, fontsize=9.0,
        sub="layer 6\nor learned 1–12 mix")
    arrow(ax, (0.855, 0.735), (0.885, 0.735), TEAL)
    box(ax, (0.885, 0.655), (0.090, 0.160), "Pooling + MLP", TEAL, fontsize=8.4,
        sub="token → region\n20 outputs")
    profile_inset(fig, [0.805, 0.405, 0.175, 0.165], y, nt_pred, "prediction", TEAL,
                  ylim=(0, 1.05 * max(float(y.max()), float(nt_pred.max()))))
    ax.text(0.8925, 0.382, f"real held-out region {rid}", ha="center", fontsize=7.5,
            color="#59636A")

    # Layer-selection inset.
    ax.add_patch(FancyBboxPatch((0.035, 0.085), 0.365, 0.480,
                               boxstyle="round,pad=0.012", facecolor="#FAFBFC",
                               edgecolor="#B0BEC5", lw=1.2))
    ax.text(0.055, 0.530, "1. Layer selection", fontsize=11.5, weight="bold", color=DARK)
    ax.text(0.055, 0.500, "Identical attention-pooling heads compare all frozen depths.",
            fontsize=8.2, color="#59636A")
    layer_labels = [1, 3, 5, 6, 9, 12]
    for j in range(6):
        yy = 0.420 - j * 0.050
        left = f"$H_{{{layer_labels[j]}}}$"
        value = [".3735", ".4455", ".4533", ".4930", ".4722", ".4744"][j]
        ax.text(0.065, yy, left, fontsize=8.2, color=DARK, va="center")
        ax.add_patch(FancyBboxPatch((0.105, yy - 0.016), 0.170 * float(value) / 0.50,
                                   0.032, boxstyle="round,pad=0.002",
                                   facecolor=TEAL if j == 3 else "#B9DDD7", edgecolor="none"))
        ax.text(0.300, yy, value, fontsize=7.7, color="#59636A", va="center")
    ax.text(0.065, 0.125, "five-fold mean per-bin $R^2$", fontsize=7.8,
            color="#59636A")
    box(ax, (0.215, 0.105), (0.155, 0.080), "Learned mixture", TEAL, fontsize=8.5,
        sub=r"$H^*=\sum_{l=1}^{12}\mathrm{softmax}(\alpha)_lH_l$" + "\n" + r"$R^2=0.4948$")

    # Pooling inset.
    ax.add_patch(FancyBboxPatch((0.425, 0.085), 0.345, 0.480,
                               boxstyle="round,pad=0.012", facecolor="#FAFBFC",
                               edgecolor="#B0BEC5", lw=1.2))
    ax.text(0.445, 0.530, "2. Trainable token-to-region head", fontsize=11.5,
            weight="bold", color=DARK)
    box(ax, (0.455, 0.430), (0.105, 0.070), "$H^*$\n$L × 512$", TEAL, fontsize=8.3)
    arrow(ax, (0.560, 0.465), (0.590, 0.465), TEAL)
    box(ax, (0.590, 0.415), (0.135, 0.100), "Linear 512→256\nGELU + LayerNorm", TEAL, fontsize=8.0)
    arrow(ax, (0.657, 0.415), (0.570, 0.350), TEAL)
    arrow(ax, (0.657, 0.415), (0.695, 0.350), TEAL)
    box(ax, (0.455, 0.270), (0.120, 0.080), "attention scores\n256→128→1\nsoftmax over tokens", TEAL, fontsize=7.4)
    box(ax, (0.625, 0.270), (0.120, 0.080), "token-wise max\n256 features", TEAL, fontsize=7.7)
    arrow(ax, (0.515, 0.270), (0.575, 0.215), TEAL)
    arrow(ax, (0.685, 0.270), (0.625, 0.215), TEAL)
    box(ax, (0.540, 0.155), (0.120, 0.070), "concatenate\n512 features", TEAL, fontsize=7.8)
    arrow(ax, (0.660, 0.190), (0.690, 0.190), TEAL)
    box(ax, (0.690, 0.145), (0.060, 0.090), "MLP\n256→128\n→20", TEAL, fontsize=7.4)
    ax.text(0.445, 0.110, "Mask excludes special tokens and padding from both pooling operations.",
            fontsize=7.6, color="#59636A")

    ax.text(0.805, 0.285, "Trainable", fontsize=8.5, weight="bold", color=TEAL)
    ax.text(0.805, 0.253, r"layer-mix logits $\alpha$", fontsize=8.0, color="#59636A")
    ax.text(0.805, 0.220, "projection, attention and MLP", fontsize=8.0, color="#59636A")
    ax.text(0.805, 0.170, "Frozen", fontsize=8.5, weight="bold", color="#607D8B")
    ax.text(0.805, 0.138, "all 50M backbone parameters", fontsize=8.0, color="#59636A")
    save(fig, "architecture_nt_detailed")


def detailed_motif_architecture(rid, y, motif_pred, mids, pbin, scores):
    fig = plt.figure(figsize=(14.2, 8.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.035, 0.952, "C  Motif-only models from explicit scan scores", fontsize=17,
            weight="bold", color=PURPLE)
    ax.text(0.035, 0.918,
            "No nucleotide sequence, GC content, coordinate or genomic annotation enters after motif scanning.",
            fontsize=9.5, color="#59636A")

    # Construct a real displayed slice of the dense matrix.
    unique = np.unique(mids)
    ranked = sorted([(scores[mids == m].max(), int(m)) for m in unique], reverse=True)
    chosen = [m for _, m in ranked[:18]] or list(range(1, 19))
    grid = np.zeros((10, len(chosen)), dtype=float)
    lookup = {m: j for j, m in enumerate(chosen)}
    for m, p, score in zip(mids, pbin, scores):
        if int(m) in lookup and 1 <= int(p) <= 10:
            grid[int(p) - 1, lookup[int(m)]] = max(grid[int(p) - 1, lookup[int(m)]], float(score))
    hm = fig.add_axes([0.035, 0.655, 0.155, 0.155])
    hm.imshow(grid, aspect="auto", cmap="Purples", vmin=0,
              vmax=max(1.0, float(grid.max())), interpolation="nearest")
    hm.set_yticks([0, 4, 9], [1, 5, 10], fontsize=6.5); hm.set_xticks([])
    hm.set_xlabel("18 displayed motifs", fontsize=7.0); hm.set_ylabel("position bin", fontsize=7.0)
    for spine in hm.spines.values(): spine.set_visible(False)
    ax.text(0.1125, 0.628, "model input: 10 × 1,443 score matrix", ha="center",
            fontsize=8.0, color="#59636A")
    arrow(ax, (0.195, 0.735), (0.230, 0.735), PURPLE)

    ax.add_patch(FancyBboxPatch((0.230, 0.635), 0.170, 0.200,
                               boxstyle="round,pad=0.010", facecolor="#F3F0F8",
                               edgecolor=PURPLE, lw=1.4))
    ax.text(0.315, 0.807, "sparse hit extraction", ha="center", fontsize=10.2,
            weight="bold", color=DARK)
    take = np.argsort(scores)[-4:][::-1] if len(scores) else []
    for row, index in enumerate(take):
        ax.text(0.250, 0.765 - row * 0.033,
                f"({int(mids[index]):4d}, {int(pbin[index]):2d}, {scores[index]:.2f})",
                family="monospace", fontsize=7.0, color=DARK)
    ax.text(0.315, 0.647, "token = (motif ID, position, score)\nretain up to 168 highest-score hits",
            ha="center", va="top", fontsize=6.9, color="#59636A")
    arrow(ax, (0.400, 0.735), (0.435, 0.735), PURPLE)
    box(ax, (0.435, 0.655), (0.135, 0.160), "Token encoder", PURPLE, fontsize=9.2,
        sub="motif embedding\n+ position embedding\n+ score MLP → N × 128")
    arrow(ax, (0.570, 0.735), (0.605, 0.735), PURPLE)
    box(ax, (0.605, 0.655), (0.135, 0.160), "Motif Transformer", PURPLE, fontsize=9.2,
        sub="prepend region token\n4 blocks × 4 heads\nFFN width 512")
    arrow(ax, (0.740, 0.735), (0.775, 0.735), PURPLE)
    box(ax, (0.775, 0.655), (0.105, 0.160), "Region pooling", PURPLE, fontsize=8.8,
        sub="CLS ∥ mean ∥ max\n384 features")
    arrow(ax, (0.880, 0.735), (0.905, 0.735), PURPLE)
    box(ax, (0.905, 0.655), (0.070, 0.160), "MLP", PURPLE, fontsize=9.0,
        sub="384→128\n→20")
    profile_inset(fig, [0.805, 0.405, 0.175, 0.165], y, motif_pred, "prediction", PURPLE,
                  ylim=(0, 1.05 * max(float(y.max()), float(motif_pred.max()))))
    ax.text(0.8925, 0.382, f"real held-out region {rid}", ha="center", fontsize=7.5,
            color="#59636A")

    # Token construction inset.
    ax.add_patch(FancyBboxPatch((0.035, 0.085), 0.345, 0.480,
                               boxstyle="round,pad=0.012", facecolor="#FAFBFC",
                               edgecolor="#B0BEC5", lw=1.2))
    ax.text(0.055, 0.530, "1. What one motif token contains", fontsize=11.5,
            weight="bold", color=DARK)
    comp = [(0.060, "motif ID", "embedding table\n1,443 identities"),
            (0.160, "position", "embedding table\n10 coarse bins"),
            (0.260, "score", "two-layer MLP\ncontinuous value")]
    for x0, title, detail in comp:
        box(ax, (x0, 0.380), (0.085, 0.105), title, PURPLE, fontsize=8.5, sub=detail)
    ax.text(0.153, 0.335, "+", fontsize=15, color=PURPLE, weight="bold")
    ax.text(0.253, 0.335, "+", fontsize=15, color=PURPLE, weight="bold")
    arrow(ax, (0.305, 0.335), (0.305, 0.285), PURPLE)
    box(ax, (0.130, 0.210), (0.155, 0.075), "LayerNorm\n128-dimensional token", PURPLE,
        fontsize=8.3)
    ax.text(0.055, 0.145,
            "Self-attention can compare motif identities while using position embeddings to represent spacing.",
            fontsize=7.8, color="#59636A", wrap=True)

    # Dense magnitude-shape branch inset.
    ax.add_patch(FancyBboxPatch((0.405, 0.085), 0.365, 0.480,
                               boxstyle="round,pad=0.012", facecolor="#FAFBFC",
                               edgecolor="#B0BEC5", lw=1.2))
    ax.text(0.425, 0.530, "2. Dense matrix branch for magnitude and shape", fontsize=11.5,
            weight="bold", color=DARK)
    box(ax, (0.430, 0.425), (0.100, 0.075), "10 × 1,443\nscore matrix", PURPLE, fontsize=8.0)
    arrow(ax, (0.530, 0.462), (0.558, 0.462), PURPLE)
    box(ax, (0.558, 0.410), (0.105, 0.105), "shared encoder", PURPLE, fontsize=8.5,
        sub="position CNN\nmean ∥ max")
    arrow(ax, (0.663, 0.462), (0.700, 0.415), TEAL)
    arrow(ax, (0.663, 0.462), (0.700, 0.315), ORANGE)
    box(ax, (0.680, 0.385), (0.070, 0.075), "magnitude", TEAL, fontsize=7.7,
        sub="Softplus\n$M≥0$")
    box(ax, (0.680, 0.285), (0.070, 0.075), "shape", ORANGE, fontsize=7.7,
        sub="Softmax\n" + r"$\sum q=1$")
    ax.text(0.445, 0.330, r"$\widehat{\lambda}_t=\widehat M\,\widehat q_t$", fontsize=12,
            color=DARK)
    ax.text(0.425, 0.235, "Count-focused", fontsize=8.0, weight="bold", color=TEAL)
    ax.text(0.425, 0.205, "Poisson NLL on 20 rates", fontsize=7.6, color="#59636A")
    ax.text(0.425, 0.165, "Shape-focused", fontsize=8.0, weight="bold", color=ORANGE)
    ax.text(0.425, 0.135, "equal-region CE on row-normalised $q$", fontsize=7.6,
            color="#59636A")
    ax.text(0.580, 0.235, "Reported separately", fontsize=8.0, weight="bold", color=PURPLE)
    ax.text(0.580, 0.200, "magnitude $R^2$", fontsize=7.6, color="#59636A")
    ax.text(0.580, 0.170, "shape $R^2$, Pearson, JS, Wasserstein", fontsize=7.6,
            color="#59636A")
    ax.text(0.580, 0.140, "peak-time error and accuracy", fontsize=7.6, color="#59636A")

    ax.text(0.805, 0.290, "Primary locked comparison", fontsize=8.7, weight="bold",
            color=PURPLE)
    ax.text(0.805, 0.255, "sparse-token Transformer", fontsize=8.0, color="#59636A")
    ax.text(0.805, 0.205, "Secondary shape analysis", fontsize=8.7, weight="bold",
            color=PURPLE)
    ax.text(0.805, 0.170, "complete 10 × 1,443 matrix", fontsize=8.0, color="#59636A")
    save(fig, "architecture_motif_detailed")


def main():
    rid, seq, y, raw, nt_pred, motif_pred, mids, pbin, scores = load_example()
    main_workflow(rid, seq, y, raw, motif_pred, mids, pbin, scores)
    dense_factorized(rid, y, motif_pred, mids, pbin, scores)
    representation_comparison(rid, mids, pbin, scores)
    three_method_overview(rid, seq, y, raw, nt_pred, motif_pred, mids, pbin, scores)
    detailed_raw_architecture(rid, seq, y, raw)
    detailed_nt_architecture(rid, seq, y, nt_pred)
    detailed_motif_architecture(rid, y, motif_pred, mids, pbin, scores)
    (ANALYSIS / "ARCHITECTURE_FIGURES_COMPLETE").write_text("Three vector architecture figures complete.\n")


if __name__ == "__main__":
    main()
