# -*- coding: utf-8 -*-
"""Generate matched-style edge figures without overwriting existing PDFs."""

from pathlib import Path
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "outputs"
OUT.mkdir(exist_ok=True)
SIM_ROOT = SCRIPT_DIR.parents[0] / "SCALE-SIMv3_Ramulator2" / "SCALE-Sim"
CSV_PATH = SIM_ROOT / "workload_config_results_sa64_ch1_seq.csv"

SCALE = float(os.environ.get("FIG_SCALE", "5.0"))

def S(x):
    return x * SCALE

def scaled_figsize(w, h):
    return (w * SCALE, h * SCALE)

font_path = SCRIPT_DIR / "arialnarrow_bold.ttf"
font_name = "DejaVu Sans"
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_name = fm.FontProperties(fname=str(font_path)).get_name()

plt.rcParams.update({
    "font.family": font_name,
    "font.weight": "bold",
    "font.size": S(11),
    "axes.labelsize": S(11),
    "xtick.labelsize": S(11),
    "ytick.labelsize": S(11),
    "legend.fontsize": S(11),
    "axes.linewidth": S(0.7),
    "xtick.major.width": S(0.5),
    "ytick.major.width": S(0.5),
    "xtick.major.size": S(0),
    "ytick.major.size": S(3),
    "xtick.major.pad": S(2),
    "ytick.major.pad": S(2),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODELS = ["llama", "opt", "resnet"]
LABELS = ["Llama 3.2-1B", "OPT-2.7B", "ResNet-50"]
MODELS = ["llama", "opt", "resnet"]
LABELS = ["Llama 3.2-1B", "OPT-2.7B", "ResNet-50"]

PRECS = ["FP16", "BF16", "FP8"]
CONFIG = {"FP16": "fp16", "BF16": "bf16", "FP8": "fp8"}
COL = {"FP16": "#6A5C94", "BF16": "#8E7EA8", "FP8": "#C99AB8"}
TH_COL = {"FP16": "#3E2E5E", "BF16": "#71618B", "FP8": "#BE6C91"}
C_SY, C_ST, C_DM = "#382955", "#BE6C91", "#E4AFCF"

rows = {}
with CSV_PATH.open(newline="") as f:
    for row in csv.DictReader(f):
        rows[(row["workload"], row["config"])] = {
            "cycles": float(row["cycles"]),
            "tot_energy": float(row["tot_energy"]),
            "ref_energy": float(row["ref_energy"]),
        }

def metric_values(kind):
    vals = {}
    for prec, cfg in CONFIG.items():
        arr = []
        for model in MODELS:
            base = rows[(f"{model}_edge", "baseline")]
            cur = rows[(f"{model}_edge", cfg)]
            if kind == "throughput":
                arr.append(base["cycles"] / cur["cycles"])
            elif kind == "total":
                arr.append(cur["tot_energy"] / base["tot_energy"])
            elif kind == "refresh":
                arr.append(cur["ref_energy"] / base["ref_energy"])
        vals[prec] = np.array(arr)
    return vals


def setup_panel(ax, ylabel, ylim, yticks=None):
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=S(1.5))
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=S(0.35), linestyle=":", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)


def save_fixed(fig, out_name):
    fig.savefig(OUT / out_name)
    plt.close(fig)
    print(f"Saved {OUT / out_name}")


def draw_grouped(vals, ylabel, out_name, colors, ylim, yticks=None, value_suffix="", baseline=False):
    x = np.arange(len(LABELS))
    width = 0.22
    offs = [-width, 0, width]
    fig, ax = plt.subplots(figsize=scaled_figsize(4.0, 2.0))
    for off, prec in zip(offs, PRECS):
        v = vals[prec]
        ax.bar(x + off, v, width, color=colors[prec], edgecolor="black", linewidth=S(0.8), zorder=3)
        for xi, yi in zip(x + off, v):
            ax.text(xi, yi + (ylim[1] - ylim[0]) * 0.018, f"{yi:.2f}{value_suffix}",
                    ha="center", va="bottom", fontsize=S(11), rotation=90, zorder=4)
    if baseline:
        ax.axhline(1.0, color="#555555", linewidth=S(0.8), linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.55, len(LABELS) - 0.45)
    setup_panel(ax, ylabel, ylim, yticks)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[p], ec="none", label=p) for p in PRECS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.78)
    save_fixed(fig, out_name)


# Accelerator cycle data: [total, compute, stall]
edge_base = np.array([
    [6939647, 6464200, 2106189],
    [10751025, 10249675, 3454320],
    [4491231, 2193445, 155160],
], dtype=float)
edge_fp16 = np.array([
    [6866191, 6399639, 2041628],
    [10678439, 10185726, 3390371],
    [4132823, 2070291, 32006],
], dtype=float)
edge_bf16 = np.array([
    [6901982, 6435336, 2077325],
    [10675419, 10182649, 3387294],
    [4120521, 2058027, 19742],
], dtype=float)
edge_fp8 = np.array([
    [6911926, 6438906, 2080895],
    [10681696, 10188814, 3393459],
    [4166872, 2104299, 66014],
], dtype=float)

def decompose(data, base):
    bt = base[:, 0]
    total, compute, stall = data[:, 0], data[:, 1], data[:, 2]
    return (total - compute) / bt, stall / bt, (compute - stall) / bt


def draw_cycles():
    x = np.arange(len(LABELS), dtype=float)
    width = 0.22
    offs = {"FP16": -width, "BF16": 0.0, "FP8": width}
    raws = {"FP16": edge_fp16, "BF16": edge_bf16, "FP8": edge_fp8}
    fig, ax = plt.subplots(figsize=scaled_figsize(6.0, 3.0))
    all_totals = []
    for prec in PRECS:
        dm, st, sy = decompose(raws[prec], edge_base)
        xpos = x + offs[prec]
        ax.bar(xpos, sy, width, color=C_SY, edgecolor="black", linewidth=S(0.8), zorder=3)
        ax.bar(xpos, st, width, bottom=sy, color=C_ST, edgecolor="black", linewidth=S(0.8), zorder=3)
        ax.bar(xpos, dm, width, bottom=sy + st, color=C_DM, edgecolor="black", linewidth=S(0.8), zorder=3)
        totals = dm + st + sy
        all_totals.extend(totals)
        for xi, t in zip(xpos, totals):
            ax.text(xi, -0.055, prec, ha="center", va="top", fontsize=S(10), rotation=45,
                    transform=ax.get_xaxis_transform(), zorder=4)
            ax.text(xi, t + 0.008, f"{t:.3f}", ha="center", va="bottom", fontsize=S(10),
                    rotation=90, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True, pad=S(2))
    ax.set_xlim(-0.55, len(LABELS) - 1.9)
    setup_panel(ax, "Normalized cycles", (0, max(1.12, max(all_totals) + 0.12)), [0.00, 0.25, 0.50, 0.75, 1.00])
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_SY, ec="none", label="Systolic compute"),
        plt.Rectangle((0, 0), 1, 1, fc=C_ST, ec="none", label="Stall"),
        plt.Rectangle((0, 0), 1, 1, fc=C_DM, ec="none", label="Data staging"),
    ]
    fig.legend(handles=handles, loc="center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.76)
    save_fixed(fig, "accel_perf_edge_matched.pdf")


def draw_cycles_v2():
    """Alternative accel panel: model labels at the bottom to reduce top crowding."""
    x = np.arange(len(LABELS), dtype=float)
    width = 0.22
    offs = {"FP16": -width, "BF16": 0.0, "FP8": width}
    raws = {"FP16": edge_fp16, "BF16": edge_bf16, "FP8": edge_fp8}
    fig, ax = plt.subplots(figsize=scaled_figsize(4.0, 2.0))
    all_totals = []
    bar_positions = []
    bar_labels = []
    for prec in PRECS:
        dm, st, sy = decompose(raws[prec], edge_base)
        xpos = x + offs[prec]
        bar_positions.extend(xpos.tolist())
        bar_labels.extend([prec] * len(xpos))
        ax.bar(xpos, sy, width, color=C_SY, edgecolor="black", linewidth=S(0.8), zorder=3)
        ax.bar(xpos, st, width, bottom=sy, color=C_ST, edgecolor="black", linewidth=S(0.8), zorder=3)
        ax.bar(xpos, dm, width, bottom=sy + st, color=C_DM, edgecolor="black", linewidth=S(0.8), zorder=3)
        totals = dm + st + sy
        all_totals.extend(totals)
        for xi, t in zip(xpos, totals):
            ax.text(xi, t + 0.008, f"{t:.3f}", ha="center", va="bottom", fontsize=S(10),
                    rotation=90, zorder=4)

    # Minor-looking tick labels identify precision; model names are centered below groups.
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right")
    ax.tick_params(axis="x", bottom=False, top=False, pad=S(1.5))
    for xi, label in zip(x, LABELS):
        ax.text(xi, -0.4, label, ha="center", va="top", fontsize=S(11), fontweight="bold",
                transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlim(-0.55, len(LABELS) - 0.45)
    setup_panel(ax, "Normalized cycles", (0, max(1.12, max(all_totals) + 0.12)), [0.00, 0.25, 0.50, 0.75, 1.00])
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_SY, ec="none", label="Systolic compute"),
        plt.Rectangle((0, 0), 1, 1, fc=C_ST, ec="none", label="Stall"),
        plt.Rectangle((0, 0), 1, 1, fc=C_DM, ec="none", label="Data staging"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.30, top=0.78)
    save_fixed(fig, "accel_perf_edge_matched_v2.pdf")


def main():
    draw_grouped(metric_values("total"), "Normalized energy", "energy_norm_edge_total_legend_matched.pdf",
                 COL, (0, 1.0), [0.00, 0.25, 0.50, 0.75, 1.00])
    draw_grouped(metric_values("refresh"), "Normalized energy", "energy_norm_edge_refresh_legend_matched.pdf",
                 {"FP16": "#8A4E7E", "BF16": "#BA7EAC", "FP8": "#DCAECE"},
                 (0, 0.75), [0.00, 0.25, 0.50, 0.75])
    draw_grouped(metric_values("throughput"), "Normalized throughput", "throughput_edge_matched.pdf",
                 TH_COL, (0.95, 1.22), [1.0, 1.1, 1.2], value_suffix="x", baseline=True)
    draw_cycles()
    draw_cycles_v2()


if __name__ == "__main__":
    main()
