"""Edge seq accelerator breakdown with baseline bars."""

from pathlib import Path

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
        "font.size": 11 * SCALE,
        "axes.labelsize": 11 * SCALE,
        "xtick.labelsize": 11 * SCALE,
        "ytick.labelsize": 11 * SCALE,
        "legend.fontsize": 11 * SCALE,
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


DNNS = ["Llama 3.2-1B", "OPT-2.7B", "ResNet-50"]
PRECISIONS = ["Base", "FP16", "BF16", "FP8"]
COLOR_DM = "#E4AFCF"
COLOR_ST = "#BE6C91"
COLOR_SY = "#382955"

EDGE_BASE = np.array(
    [
        [6939647, 6464200, 2106189],
        [10751025, 10249675, 3454320],
        [4491231, 2193445, 155160],
    ],
    dtype=float,
)
EDGE_FP16 = np.array(
    [
        [6866191, 6399639, 2041628],
        [10678439, 10185726, 3390371],
        [4132823, 2070291, 32006],
    ],
    dtype=float,
)
EDGE_BF16 = np.array(
    [
        [6901982, 6435336, 2077325],
        [10675419, 10182649, 3387294],
        [4120521, 2058027, 19742],
    ],
    dtype=float,
)
EDGE_FP8 = np.array(
    [
        [6911926, 6438906, 2080895],
        [10681696, 10188814, 3393459],
        [4166872, 2104299, 66014],
    ],
    dtype=float,
)


def decompose(data, base):
    base_total = base[:, 0]
    total, compute, stall = data[:, 0], data[:, 1], data[:, 2]
    return (total - compute) / base_total, stall / base_total, (compute - stall) / base_total


def main():
    raws = {"Base": EDGE_BASE, "FP16": EDGE_FP16, "BF16": EDGE_BF16, "FP8": EDGE_FP8}
    bar_w = 0.18
    gap = 0.035
    step = bar_w + gap
    offsets = {"Base": -1.5 * step, "FP16": -0.5 * step, "BF16": 0.5 * step, "FP8": 1.5 * step}

    x = np.arange(len(DNNS), dtype=float)
    fig, ax = plt.subplots(figsize=(6.8 * SCALE, 3.55 * SCALE))
    all_totals = []

    for precision in PRECISIONS:
        dm, st, sy = decompose(raws[precision], EDGE_BASE)
        xpos = x + offsets[precision]
        ax.bar(xpos, sy, bar_w, color=COLOR_SY, edgecolor="black", linewidth=0.8 * SCALE, zorder=5)
        ax.bar(xpos, st, bar_w, bottom=sy, color=COLOR_ST, edgecolor="black", linewidth=0.8 * SCALE, zorder=5)
        ax.bar(xpos, dm, bar_w, bottom=sy + st, color=COLOR_DM, edgecolor="black", linewidth=0.8 * SCALE, zorder=5)

        totals = dm + st + sy
        all_totals.extend(totals.tolist())
        for xi, total in zip(xpos, totals):
            ax.text(
                xi,
                -0.015,
                precision,
                ha="center",
                va="top",
                fontsize=9.2 * SCALE,
                rotation=45,
                transform=ax.get_xaxis_transform(),
                zorder=4,
            )
            ax.text(
                xi,
                total + 0.008,
                f"{total:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.2 * SCALE,
                rotation=90,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(DNNS)
    ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True, pad=5)
    ax.set_xlim(-0.55, len(DNNS) - 0.45)
    ax.set_ylim(0, max(1.14, max(all_totals) + 0.12))
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Normalized cycles", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=":", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLOR_SY, ec="none", label="Systolic compute"),
        plt.Rectangle((0, 0), 1, 1, fc=COLOR_ST, ec="none", label="Stall"),
        plt.Rectangle((0, 0), 1, 1, fc=COLOR_DM, ec="none", label="Data staging"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.5, rect=[0.0, 0.0, 1.0, 0.70])
    fig.savefig(OUT / "accel_perf_edge_seq_v2.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
