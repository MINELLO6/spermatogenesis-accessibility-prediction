def main():
    import argparse

    argparse.ArgumentParser(description="make interpretation figures").parse_args()
    #!/usr/bin/env python3
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr

    from developmental_accessibility.paths import RUN_ROOT

    ROOT = Path(str(RUN_ROOT / "final_analysis"))
    INP = ROOT / "interpretation/raw_rc_v2"
    OUT = ROOT / "figures"
    OUT.mkdir(exist_ok=True)

    ig = np.load(INP / "integrated_gradients_position.npy")
    full = np.load(INP / "integrated_gradients_full.npy")
    mut = np.load(INP / "mutagenesis_delta.npy")
    rows = json.load(open(INP / "selection.json"))
    peak = np.asarray([r["peak_bin"] for r in rows])
    stratum = np.asarray([r["error_stratum"] for r in rows])
    positions = np.arange(-100, 101)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Aggregate positional sensitivity, contrasting well- and poorly-predicted regions.
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for label, colour in [("low", "#16697A"), ("high", "#DB6400")]:
        values = np.abs(ig[stratum == label])
        mean = values.mean(0)
        se = values.std(0, ddof=1) / np.sqrt(len(values))
        ax.plot(
            positions, mean, color=colour, lw=1.8, label=f"{label.capitalize()} prediction error"
        )
        ax.fill_between(
            positions, mean - 1.96 * se, mean + 1.96 * se, color=colour, alpha=0.18, linewidth=0
        )
    ax.axvline(0, color="0.55", ls="--", lw=1)
    ax.set(
        xlabel="Position relative to region centre (bp)",
        ylabel="Mean absolute integrated gradient",
        title="Sequence sensitivity is distributed across the 201-bp window",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"integrated_gradients_by_position.{ext}", dpi=300)
    plt.close(fig)

    # Pseudotime-bin-specific attribution heat map.
    heat = np.stack([np.mean(np.abs(ig[peak == b]), axis=0) for b in range(1, 21)])
    heat = heat / np.maximum(heat.mean(1, keepdims=True), 1e-8)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = ax.imshow(
        heat,
        aspect="auto",
        cmap="magma",
        extent=(-100, 100, 20.5, 0.5),
        vmin=np.percentile(heat, 2),
        vmax=np.percentile(heat, 98),
    )
    ax.set(
        xlabel="Position relative to region centre (bp)",
        ylabel="Target peak pseudotime bin",
        title="Relative integrated-gradient magnitude by target peak bin",
    )
    fig.colorbar(im, ax=ax, label="Attribution / bin-wise mean")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"integrated_gradients_heatmap.{ext}", dpi=300)
    plt.close(fig)

    # Comparison with exhaustive substitutions. Five low-error representatives were
    # selected from every target peak bin by interpret_final_raw.py.
    low_idx = np.concatenate(
        [np.flatnonzero((stratum == "low") & (peak == b))[:5] for b in range(1, 21)]
    )
    ig_sensitivity = np.abs(ig[low_idx])
    mut_sensitivity = np.nanmean(np.abs(mut), axis=2)
    rho_all = float(spearmanr(ig_sensitivity.ravel(), mut_sensitivity.ravel()).statistic)
    rho_region = np.asarray(
        [spearmanr(a, b).statistic for a, b in zip(ig_sensitivity, mut_sensitivity)]
    )

    rng = np.random.default_rng(20260829)
    flat_i, flat_m = ig_sensitivity.ravel(), mut_sensitivity.ravel()
    take = rng.choice(len(flat_i), min(12000, len(flat_i)), replace=False)
    fig, ax = plt.subplots(figsize=(5.7, 5.0))
    ax.hexbin(flat_i[take], flat_m[take], gridsize=45, mincnt=1, cmap="viridis", bins="log")
    ax.set(
        xlabel="Absolute integrated gradient",
        ylabel="Mean absolute substitution effect",
        title=f"Limited agreement between attribution methods ($\\rho$={rho_all:.2f})",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ig_mutagenesis_agreement.{ext}", dpi=300)
    plt.close(fig)

    # Direct perturbation sensitivity across the input window.
    mut_mean = np.nanmean(np.abs(mut), axis=(0, 2))
    mut_se = np.nanstd(np.nanmean(np.abs(mut), axis=2), axis=0, ddof=1) / np.sqrt(len(mut))
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.plot(positions, mut_mean, color="#5B2C6F", lw=1.8)
    ax.fill_between(
        positions,
        mut_mean - 1.96 * mut_se,
        mut_mean + 1.96 * mut_se,
        color="#5B2C6F",
        alpha=0.18,
        linewidth=0,
    )
    ax.axvline(0, color="0.55", ls="--", lw=1)
    ax.set(
        xlabel="Position relative to region centre (bp)",
        ylabel="Mean absolute change in predicted accessibility",
        title="Single-base substitution sensitivity across the input window",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"mutagenesis_by_position.{ext}", dpi=300)
    plt.close(fig)

    stats = {
        "n_ig_regions": int(len(ig)),
        "n_mutagenesis_regions": int(len(mut)),
        "spearman_all_region_positions": rho_all,
        "median_within_region_spearman": float(np.nanmedian(rho_region)),
        "iqr_within_region_spearman": [
            float(np.nanpercentile(rho_region, 25)),
            float(np.nanpercentile(rho_region, 75)),
        ],
        "mean_absolute_ig": float(np.mean(np.abs(ig))),
        "mean_absolute_mutation_effect": float(np.nanmean(np.abs(mut))),
        "centre_41bp_fraction_absolute_ig": float(np.abs(ig[:, 80:121]).sum() / np.abs(ig).sum()),
    }
    with open(INP / "figure_statistics.json", "w") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
