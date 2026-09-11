"""Reproduce the approved comparison plot from archived summary values."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from developmental_accessibility.paths import REPOSITORY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=REPOSITORY / "results/figure_data/model_comparison.json"
    )
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "figures")
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(args.input.read_text())["overview"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
    fig.subplots_adjust(left=0.35, right=0.97, top=0.9, bottom=0.15, wspace=0.22)
    y = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12])
    names = [
        "CNN",
        "CNN + RC augmentation",
        "CNN + RC + strand averaging",
        "CNN + RC + attention pooling",
        "ConvNeXt",
        "NT: final layer",
        "NT: layer 6",
        "NT: learned layer mixture",
        "NT: IA3 adaptation",
        "Sparse motif model",
    ]
    for a, key, sd, title, ticks, lim in zip(
        ax,
        ["mean_r2", "rmse"],
        ["sd_r2", "sd_rmse"],
        [r"(a) Mean per-bin $R^2$", "(b) RMSE"],
        [[0.3, 0.4, 0.5, 0.6], [40, 50, 60, 70]],
        [(0.28, 0.68), (37, 70)],
    ):
        for yy in y:
            a.axhline(yy, color=".94", linewidth=0.5, zorder=0)
        a.errorbar(
            d[key],
            y,
            xerr=d[sd],
            fmt="o",
            markersize=3.8,
            color="#284b63",
            ecolor="#718491",
            elinewidth=0.85,
            capsize=2,
            capthick=0.7,
            zorder=3,
        )
        a.set_xlim(*lim)
        a.set_xticks(ticks)
        a.set_ylim(12.65, -0.5)
        a.set_title(title, loc="left", fontsize=10, pad=10)
        a.spines[["top", "right", "left"]].set_visible(False)
        a.tick_params(axis="y", length=0)
        a.tick_params(axis="x", labelsize=8)
        for boundary in [5.65, 10.65]:
            a.axhline(boundary, color=".72", linewidth=0.55)
    ax[0].set_yticks(y, names, fontsize=8.3)
    ax[1].tick_params(axis="y", labelleft=False)
    for yy, title in [
        (0, "Raw-sequence models"),
        (6, "Nucleotide Transformer"),
        (11, "Motif representation"),
    ]:
        ax[0].text(
            -0.07,
            yy,
            title,
            transform=ax[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            clip_on=False,
        )
    ax[0].set_xlabel("Higher is better", fontsize=8, labelpad=7)
    ax[1].set_xlabel("Lower is better", fontsize=8, labelpad=7)
    fig.text(0.35, 0.035, "Points: five-fold mean; intervals: ±1 sample SD.", fontsize=8)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT / f"model_overview_academic.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08
        )
    plt.close(fig)
    print(OUT / "model_overview_academic.pdf")


if __name__ == "__main__":
    main()
