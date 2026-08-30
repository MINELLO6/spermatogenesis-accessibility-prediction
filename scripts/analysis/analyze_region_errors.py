#!/usr/bin/env python3
"""Post-hoc held-out error stratification and raw-versus-motif overlap analysis.

This script is CPU-only.  It uses the already locked held-out predictions and
does not train, select, or tune a model.  All thresholds and definitions are
written to analysis_definitions.json for reproducibility.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("/root/sc-motif-open/R")
FINAL = Path("/root/autodl-tmp/final_analysis")
OUT = Path("/root/autodl-tmp/region_error_analysis")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

EPS = 1e-8
PEAK_PROMINENCE = 0.03
PEAK_MIN_HEIGHT = 0.08
LOCAL_PEAK_EXCESS = 0.025
TIME_SHIFT_BINS = 2
FLAT_RATIO = 0.50


def row_shape(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    magnitude = x.sum(axis=1)
    shape = (x + EPS) / (magnitude[:, None] + EPS * x.shape[1])
    return magnitude, shape


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log((p + EPS) / (m + EPS)), axis=1) + 0.5 * np.sum(
        q * np.log((q + EPS) / (m + EPS)), axis=1
    )


def row_pearson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pc = p - p.mean(axis=1, keepdims=True)
    qc = q - q.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(pc * pc, axis=1) * np.sum(qc * qc, axis=1))
    return np.divide(np.sum(pc * qc, axis=1), den, out=np.zeros(len(p)), where=den > EPS)


def terciles(error: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cuts = np.quantile(error, [1 / 3, 2 / 3])
    labels = np.where(error <= cuts[0], "good", np.where(error <= cuts[1], "middle", "poor"))
    return labels, cuts.tolist()


def smooth_profile(q: np.ndarray) -> np.ndarray:
    padded = np.pad(q, ((0, 0), (1, 1)), mode="edge")
    return 0.25 * padded[:, :-2] + 0.50 * q + 0.25 * padded[:, 2:]


def local_peak_count(q: np.ndarray) -> np.ndarray:
    q = smooth_profile(q)
    med = np.median(q, axis=1, keepdims=True)
    core = q[:, 1:-1]
    hit = (core > q[:, :-2]) & (core >= q[:, 2:]) & (core >= PEAK_MIN_HEIGHT) & (core >= med + LOCAL_PEAK_EXCESS)
    edge0 = (q[:, 0] > q[:, 1]) & (q[:, 0] >= PEAK_MIN_HEIGHT) & (q[:, 0] >= med[:, 0] + LOCAL_PEAK_EXCESS)
    edge1 = (q[:, -1] > q[:, -2]) & (q[:, -1] >= PEAK_MIN_HEIGHT) & (q[:, -1] >= med[:, 0] + LOCAL_PEAK_EXCESS)
    return hit.sum(axis=1) + edge0 + edge1


def error_types(true_q: np.ndarray, pred_q: np.ndarray) -> np.ndarray:
    true_s = smooth_profile(true_q)
    pred_s = smooth_profile(pred_q)
    true_prom = true_s.max(axis=1) - np.median(true_s, axis=1)
    pred_prom = pred_s.max(axis=1) - np.median(pred_s, axis=1)
    true_peaked = (true_prom >= PEAK_PROMINENCE) & (true_s.max(axis=1) >= PEAK_MIN_HEIGHT)
    pred_peaked = (pred_prom >= PEAK_PROMINENCE) & (pred_s.max(axis=1) >= PEAK_MIN_HEIGHT)
    shift = np.abs(true_s.argmax(axis=1) - pred_s.argmax(axis=1))
    true_npeak = local_peak_count(true_q)
    pred_npeak = local_peak_count(pred_q)
    result = np.full(len(true_q), "other", dtype="U24")
    true_range = np.ptp(true_q, axis=1)
    pred_range = np.ptp(pred_q, axis=1)
    result[(true_range >= PEAK_PROMINENCE) & (pred_range < FLAT_RATIO * true_range)] = "over_flat"
    result[~true_peaked & pred_peaked] = "false_peak"
    result[true_peaked & ~pred_peaked] = "missed_peak"
    result[true_peaked & pred_peaked & (shift >= TIME_SHIFT_BINS)] = "peak_time_shift"
    result[(true_npeak >= 2) & (pred_npeak < true_npeak)] = "multi_peak_failure"
    return result


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def motif_enrichment(global_ids: np.ndarray, selected: np.ndarray) -> None:
    csr = DATA / "motif_csr"
    offsets = np.load(csr / "offsets.npy", mmap_mode="r")
    motif = np.load(csr / "motif_ids.npy", mmap_mode="r")
    pos = np.load(csr / "position_bins.npy", mmap_mode="r")
    n_motif, n_pos = 1443, 10
    all_m = np.zeros(n_motif + 1, dtype=np.int64)
    sel_m = np.zeros(n_motif + 1, dtype=np.int64)
    all_mp = np.zeros((n_motif + 1, n_pos + 1), dtype=np.int64)
    sel_mp = np.zeros((n_motif + 1, n_pos + 1), dtype=np.int64)
    for j, rid in enumerate(global_ids):
        a, b = int(offsets[rid]), int(offsets[rid + 1])
        if b <= a:
            continue
        mids = np.asarray(motif[a:b], dtype=np.int64)
        pbin = np.asarray(pos[a:b], dtype=np.int64)
        um = np.unique(mids[(mids >= 1) & (mids <= n_motif)])
        all_m[um] += 1
        cells = np.unique(mids * (n_pos + 1) + pbin)
        cm, cp = cells // (n_pos + 1), cells % (n_pos + 1)
        ok = (cm >= 1) & (cm <= n_motif) & (cp >= 1) & (cp <= n_pos)
        all_mp[cm[ok], cp[ok]] += 1
        if selected[j]:
            sel_m[um] += 1
            sel_mp[cm[ok], cp[ok]] += 1

    n_all, n_sel = len(global_ids), int(selected.sum())
    bg_m = all_m - sel_m
    n_bg = n_all - n_sel
    sel_prev = (sel_m[1:] + 0.5) / (n_sel + 1.0)
    bg_prev = (bg_m[1:] + 0.5) / (n_bg + 1.0)
    log2e = np.log2(sel_prev / bg_prev)
    order = np.argsort(log2e)[::-1]
    with open(OUT / "motif_enrichment_intersection.csv", "w") as h:
        h.write("motif_id,selected_regions,background_regions,selected_prevalence,background_prevalence,log2_enrichment\n")
        for k in order:
            h.write(f"{k+1},{sel_m[k+1]},{bg_m[k+1]},{sel_prev[k]:.8g},{bg_prev[k]:.8g},{log2e[k]:.8g}\n")

    bg_mp = all_mp - sel_mp
    sel_prev_mp = (sel_mp[1:, 1:] + 0.5) / (n_sel + 1.0)
    bg_prev_mp = (bg_mp[1:, 1:] + 0.5) / (n_bg + 1.0)
    log2e_mp = np.log2(sel_prev_mp / bg_prev_mp)
    flat_order = np.argsort(log2e_mp.ravel())[::-1]
    with open(OUT / "motif_position_enrichment_intersection.csv", "w") as h:
        h.write("motif_id,position_bin,selected_regions,background_regions,selected_prevalence,background_prevalence,log2_enrichment\n")
        for z in flat_order:
            m, p = np.unravel_index(z, log2e_mp.shape)
            h.write(f"{m+1},{p+1},{sel_mp[m+1,p+1]},{bg_mp[m+1,p+1]},{sel_prev_mp[m,p]:.8g},{bg_prev_mp[m,p]:.8g},{log2e_mp[m,p]:.8g}\n")

    top = order[:20][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh([f"motif {i+1}" for i in top], log2e[top], color="#3B82A0")
    ax.set_xlabel("log2 prevalence enrichment\n(raw-good ∩ motif-good versus remaining held-out regions)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "shared_good_motif_enrichment")


def main() -> None:
    heldout = np.load(FINAL / "heldout_test_idx_LOCKED.npy")
    with open(DATA / "totmat_shape.txt") as h:
        n, t = map(int, h.read().split())
    targets_mm = np.memmap(DATA / "totmat_f64.bin", dtype="<f8", mode="r", shape=(n, t), order="F")
    y = np.asarray(targets_mm[heldout], dtype=np.float32)
    raw = np.load(FINAL / "heldout_ensembles/raw_rc_5fold_ensemble/heldout_predictions_f32.npy", mmap_mode="r")
    motif = np.load(FINAL / "heldout_ensembles/motif_transformer_5fold_ensemble/heldout_predictions_f32.npy", mmap_mode="r")

    true_mag, true_q = row_shape(y)
    metrics: dict[str, dict[str, np.ndarray | list[float]]] = {}
    for name, pred in (("raw", raw), ("motif", motif)):
        mag, q = row_shape(pred)
        mag_err = np.abs(np.log1p(mag) - np.log1p(true_mag))
        shape_js = js_divergence(true_q, q)
        shape_r = row_pearson(true_q, q)
        mag_class, mag_cuts = terciles(mag_err)
        shape_class, shape_cuts = terciles(shape_js)
        metrics[name] = {
            "magnitude": mag,
            "magnitude_log_error": mag_err,
            "magnitude_class": mag_class,
            "magnitude_cuts": mag_cuts,
            "shape_js": shape_js,
            "shape_pearson": shape_r,
            "shape_class": shape_class,
            "shape_cuts": shape_cuts,
            "error_type": error_types(true_q, q),
            "shape": q,
        }

    raw_shape_good = metrics["raw"]["shape_class"] == "good"
    motif_shape_good = metrics["motif"]["shape_class"] == "good"
    raw_mag_good = metrics["raw"]["magnitude_class"] == "good"
    motif_mag_good = metrics["motif"]["magnitude_class"] == "good"
    shape_intersection = raw_shape_good & motif_shape_good
    mag_intersection = raw_mag_good & motif_mag_good

    definitions = {
        "magnitude": "sum of non-negative 20-bin values",
        "magnitude_error": "absolute difference in log1p magnitude",
        "shape": "non-negative 20-bin profile divided by its row sum",
        "shape_error": "Jensen-Shannon divergence",
        "strata": "held-out within-model error terciles; low error=good, middle=middle, high=poor",
        "peak_prominence": PEAK_PROMINENCE,
        "peak_minimum_normalised_height": PEAK_MIN_HEIGHT,
        "local_peak_excess_over_row_median": LOCAL_PEAK_EXCESS,
        "peak_time_shift_bins": TIME_SHIFT_BINS,
        "over_flat_predicted_to_true_range_ratio": FLAT_RATIO,
        "raw_magnitude_cuts": metrics["raw"]["magnitude_cuts"],
        "motif_magnitude_cuts": metrics["motif"]["magnitude_cuts"],
        "raw_shape_js_cuts": metrics["raw"]["shape_cuts"],
        "motif_shape_js_cuts": metrics["motif"]["shape_cuts"],
    }
    with open(OUT / "analysis_definitions.json", "w") as h:
        json.dump(definitions, h, indent=2)

    overlap = {
        "n_heldout": len(heldout),
        "shape_good_raw": int(raw_shape_good.sum()),
        "shape_good_motif": int(motif_shape_good.sum()),
        "shape_good_intersection": int(shape_intersection.sum()),
        "shape_good_jaccard": float(shape_intersection.sum() / (raw_shape_good | motif_shape_good).sum()),
        "magnitude_good_raw": int(raw_mag_good.sum()),
        "magnitude_good_motif": int(motif_mag_good.sum()),
        "magnitude_good_intersection": int(mag_intersection.sum()),
        "magnitude_good_jaccard": float(mag_intersection.sum() / (raw_mag_good | motif_mag_good).sum()),
    }
    with open(OUT / "overlap_summary.json", "w") as h:
        json.dump(overlap, h, indent=2)

    with gzip.open(OUT / "region_metrics.csv.gz", "wt") as h:
        h.write("heldout_row,global_index,true_magnitude,raw_magnitude_log_error,raw_magnitude_class,raw_shape_js,raw_shape_pearson,raw_shape_class,raw_error_type,motif_magnitude_log_error,motif_magnitude_class,motif_shape_js,motif_shape_pearson,motif_shape_class,motif_error_type,shape_good_intersection,magnitude_good_intersection\n")
        for i in range(len(heldout)):
            h.write(
                f"{i},{heldout[i]},{true_mag[i]:.8g},{metrics['raw']['magnitude_log_error'][i]:.8g},{metrics['raw']['magnitude_class'][i]},"
                f"{metrics['raw']['shape_js'][i]:.8g},{metrics['raw']['shape_pearson'][i]:.8g},{metrics['raw']['shape_class'][i]},{metrics['raw']['error_type'][i]},"
                f"{metrics['motif']['magnitude_log_error'][i]:.8g},{metrics['motif']['magnitude_class'][i]},"
                f"{metrics['motif']['shape_js'][i]:.8g},{metrics['motif']['shape_pearson'][i]:.8g},{metrics['motif']['shape_class'][i]},{metrics['motif']['error_type'][i]},"
                f"{int(shape_intersection[i])},{int(mag_intersection[i])}\n"
            )

    with open(OUT / "error_type_summary.csv", "w") as h:
        h.write("model,error_type,n,fraction\n")
        for name in ("raw", "motif"):
            vals, counts = np.unique(metrics[name]["error_type"], return_counts=True)
            for v, c in zip(vals, counts):
                h.write(f"{name},{v},{c},{c/len(heldout):.8g}\n")

    categories = ["peak_time_shift", "over_flat", "false_peak", "missed_peak", "multi_peak_failure"]
    exemplars: dict[str, dict[str, list[int]]] = {}
    for name in ("raw", "motif"):
        exemplars[name] = {}
        for category in categories:
            candidates = np.where(metrics[name]["error_type"] == category)[0]
            order = candidates[np.argsort(metrics[name]["shape_js"][candidates])[::-1]]
            exemplars[name][category] = order[:5].tolist()
    with open(OUT / "exemplars.json", "w") as h:
        json.dump(exemplars, h, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8), sharex=True, sharey=True)
    x = np.arange(1, 21)
    for row, name in enumerate(("raw", "motif")):
        for col, stratum in enumerate(("good", "middle", "poor")):
            cand = np.where(metrics[name]["shape_class"] == stratum)[0]
            idx = cand[np.argsort(metrics[name]["shape_js"][cand])][len(cand) // 2]
            axes[row, col].plot(x, true_q[idx], color="black", lw=2, label="observed")
            axes[row, col].plot(x, metrics[name]["shape"][idx], color="#D95F02", lw=2, label=name)
            axes[row, col].set_title(f"{name}: {stratum}")
            axes[row, col].text(0.03, 0.95, f"region {heldout[idx]}\nJS={metrics[name]['shape_js'][idx]:.3f}", transform=axes[row, col].transAxes, va="top", fontsize=8)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("pseudotime bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("row-normalised accessibility")
    fig.tight_layout()
    save_figure(fig, "shape_good_middle_poor_examples")

    fig, axes = plt.subplots(2, 5, figsize=(14.0, 5.2), sharex=True, sharey=True)
    for row, name in enumerate(("raw", "motif")):
        for col, category in enumerate(categories):
            ids = exemplars[name][category]
            ax = axes[row, col]
            if ids:
                idx = ids[0]
                ax.plot(x, true_q[idx], color="black", lw=1.8)
                ax.plot(x, metrics[name]["shape"][idx], color="#7570B3", lw=1.8)
                ax.text(0.03, 0.95, f"region {heldout[idx]}", transform=ax.transAxes, va="top", fontsize=7)
            ax.set_title(category.replace("_", "\n"), fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("raw\nnormalised signal")
    axes[1, 0].set_ylabel("motif\nnormalised signal")
    for ax in axes[-1]:
        ax.set_xlabel("bin")
    fig.tight_layout()
    save_figure(fig, "shape_error_taxonomy_examples")

    motif_enrichment(heldout, shape_intersection)
    (OUT / "COMPLETE").write_text("CPU-only held-out error analysis complete.\n")
    print(json.dumps(overlap, indent=2))


if __name__ == "__main__":
    main()
