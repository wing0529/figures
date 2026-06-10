"""Workload-wise DRAM throughput comparison for seq traces."""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path("/home/wing02/arialnarrow_bold.ttf")
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    FONT_NAME = fm.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT_NAME = "DejaVu Sans"

SCALE = 2.0
plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.weight": "bold",
        "font.size": 10 * SCALE,
        "axes.labelsize": 10 * SCALE,
        "xtick.labelsize": 9.5 * SCALE,
        "ytick.labelsize": 9.5 * SCALE,
        "legend.fontsize": 10 * SCALE,
        "axes.linewidth": 0.6 * SCALE,
        "xtick.major.width": 0.5 * SCALE,
        "ytick.major.width": 0.5 * SCALE,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 2 * SCALE,
        "ytick.major.pad": 2 * SCALE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


MODELS = ["Llama 3.2-1B", "OPT-2.7B", "ResNet-50"]
SERIES = ["Freya", "PCM", "Approx"]
VALUES = np.array(
    [
        [1.142, 0.352, 0.059],
        [1.142, 0.350, 0.058],
        [1.136, 0.347, 0.059],
    ],
    dtype=float,
)


def main():
    with (OUT / "arch_compare_seq_values.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "Freya", "PCM", "Approx_RowSplitController_weighted"])
        for model, row in zip(MODELS, VALUES):
            writer.writerow([model, *[f"{v:.6f}" for v in row]])

    colors = ["#382955", "#936C8E", "#E4AFCF"]
    x = np.arange(len(MODELS), dtype=float)
    bar_w = 0.19
    offsets = np.array([-bar_w - 0.035, 0.0, bar_w + 0.035])

    fig, ax = plt.subplots(figsize=(6* SCALE, 3 * SCALE))
    for i, (name, color) in enumerate(zip(SERIES, colors)):
        xpos = x + offsets[i]
        ax.bar(
            xpos,
            VALUES[:, i],
            bar_w,
            color=color,
            edgecolor="black",
            linewidth=0.7 * SCALE,
            label=name,
            zorder=5,
        )
        for xi, v in zip(xpos, VALUES[:, i]):
            ax.text(
                xi,
                v + 0.025,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8.8 * SCALE,
                fontweight="bold",
                zorder=6,
            )

    ax.axhline(1.0, color="black", linewidth=0.6 * SCALE, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Normalized throughput", fontweight="bold")
    ax.set_ylim(0, 1.42)
    ax.set_yticks(np.arange(0, 1.31, 0.25))
    ax.yaxis.grid(True, linewidth=0.35, linestyle=":", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.78),
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.5, rect=[0.0, 0.0, 1.0, 0.70])
    fig.savefig(OUT / "arch_compare_seq.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
