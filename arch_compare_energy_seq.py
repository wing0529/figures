"""Workload-wise normalized DRAM energy comparison for seq prior-work comparison."""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim")
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

COMPARISON_CSV = ROOT / "analysis_results/evaluation_pcm_approx_seqstrict/pcm_approx_comparison.csv"
PER_LAYER_CSV = ROOT / "analysis_results/evaluation_pcm_approx_seqstrict/pcm_approx_per_layer.csv"
LLAMA_ROWSPLIT_CSV = ROOT / "analysis_results/evaluation_pcm_approx_rowsplit_seq/llama_rowsplit_controller_per_layer.csv"
OPT_RESNET_ROWSPLIT_CSV = ROOT / "analysis_results/evaluation_pcm_approx_rowsplit_seq/opt_resnet_rowsplit_controller_per_layer.csv"

MODEL_KEYS = ["llama", "opt", "resnet"]
MODEL_LABELS = ["Llama 3.2-1B", "OPT-2.7B", "ResNet-50"]
SCHEMES = ["Freya", "PCM", "Approx"]

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


def load_repeat_factors():
    repeats = {}
    with PER_LAYER_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["scheme"] != "approx_row_1ch":
                continue
            repeats[(row["model"], int(row["layer"]))] = int(float(row["repeat"]))
    return repeats


def load_baseline_and_direct_schemes():
    baseline_1ch = {}
    rows = {}
    with COMPARISON_CSV.open() as f:
        for row in csv.DictReader(f):
            model = row["model"]
            comparison = row["comparison"]
            if comparison == "frcam_fp16_vs_baseline_1ch":
                baseline_1ch[model] = {
                    "total": float(row["baseline_total_energy"]),
                    "ref": float(row["baseline_total_ref_energy"]),
                }
                rows[(model, "Freya")] = {
                    "baseline_total": float(row["baseline_total_energy"]),
                    "baseline_ref": float(row["baseline_total_ref_energy"]),
                    "config_total": float(row["config_total_energy"]),
                    "config_ref": float(row["config_total_ref_energy"]),
                }
            elif comparison == "pcm_8c_vs_baseline_8ch":
                rows[(model, "PCM")] = {
                    "baseline_total": float(row["baseline_total_energy"]),
                    "baseline_ref": float(row["baseline_total_ref_energy"]),
                    "config_total": float(row["config_total_energy"]),
                    "config_ref": float(row["config_total_ref_energy"]),
                }
    return baseline_1ch, rows


def add_rowsplit_approx(rows, baseline_1ch, repeats):
    approx = {}
    with LLAMA_ROWSPLIT_CSV.open() as f:
        for row in csv.DictReader(f):
            model = "llama"
            repeat = repeats[(model, int(row["layer"]))]
            acc = approx.setdefault(model, {"config_total": 0.0, "config_ref": 0.0})
            acc["config_total"] += float(row["total_energy"]) * repeat
            acc["config_ref"] += float(row["total_ref_energy"]) * repeat

    with OPT_RESNET_ROWSPLIT_CSV.open() as f:
        for row in csv.DictReader(f):
            model = row["model"]
            repeat = repeats[(model, int(row["layer"]))]
            acc = approx.setdefault(model, {"config_total": 0.0, "config_ref": 0.0})
            acc["config_total"] += float(row["total_energy"]) * repeat
            acc["config_ref"] += float(row["total_ref_energy"]) * repeat

    for model, vals in approx.items():
        rows[(model, "Approx")] = {
            "baseline_total": baseline_1ch[model]["total"],
            "baseline_ref": baseline_1ch[model]["ref"],
            "config_total": vals["config_total"],
            "config_ref": vals["config_ref"],
        }


def build_values():
    repeats = load_repeat_factors()
    baseline_1ch, rows = load_baseline_and_direct_schemes()
    add_rowsplit_approx(rows, baseline_1ch, repeats)

    total_norm = np.zeros((len(MODEL_KEYS), len(SCHEMES)), dtype=float)
    refresh_norm = np.zeros_like(total_norm)
    csv_rows = []

    for i, model in enumerate(MODEL_KEYS):
        for j, scheme in enumerate(SCHEMES):
            vals = rows[(model, scheme)]
            norm_total = vals["config_total"] / vals["baseline_total"]
            norm_ref = vals["config_ref"] / vals["baseline_ref"]
            total_norm[i, j] = norm_total
            refresh_norm[i, j] = norm_ref
            csv_rows.append(
                {
                    "model": model,
                    "scheme": scheme,
                    "baseline_total_energy": vals["baseline_total"],
                    "config_total_energy": vals["config_total"],
                    "baseline_ref_energy": vals["baseline_ref"],
                    "config_ref_energy": vals["config_ref"],
                    "normalized_total_energy": norm_total,
                    "normalized_refresh_energy": norm_ref,
                }
            )

    with (OUT / "arch_compare_energy_seq_values.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    return total_norm, refresh_norm


def draw_grouped(values, ylabel, out_name, value_fmt="{:.2f}", ylim_pad=0.8, ytick_step=None, cap=None):
    x = np.arange(len(MODEL_LABELS), dtype=float)
    bar_w = 0.19
    offsets = np.array([-bar_w - 0.035, 0.0, bar_w + 0.035])
    colors = ["#382955", "#936C8E", "#DEC9A3"]

    fig, ax = plt.subplots(figsize=(5.8 * SCALE, 3.25 * SCALE))
    for j, scheme in enumerate(SCHEMES):
        xpos = x + offsets[j]
        ax.bar(
            xpos,
            np.minimum(values[:, j], cap) if cap is not None else values[:, j],
            bar_w,
            color=colors[j],
            edgecolor="black",
            linewidth=0.7 * SCALE,
            label=scheme,
            zorder=5,
        )
        drawn_values = np.minimum(values[:, j], cap) if cap is not None else values[:, j]
        for xi, value, drawn in zip(xpos, values[:, j], drawn_values):
            label = value_fmt.format(value)
            if cap is not None and value > cap:
                ax.text(
                    xi,
                    cap - 0.18,
                    label,
                    ha="center",
                    va="top",
                    rotation=90,
                    fontsize=8.8 * SCALE,
                    fontweight="bold",
                    zorder=8,
                )
            else:
                ax.text(
                    xi,
                    drawn + (ylim_pad * 0.08),
                    label,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8.8 * SCALE,
                    fontweight="bold",
                    zorder=6,
                )

    #ax.axhline(1.0, color="black", linewidth=0.6 * SCALE, linestyle="", zorder=2)
    if cap is not None:
        ax.axhline(cap, color="black", linewidth=0.75 * SCALE, linestyle="-", zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel(ylabel, fontweight="bold")
    ymax = (cap if cap is not None else float(values.max())) + ylim_pad
    ax.set_ylim(0, ymax)
    if ytick_step is not None:
        ax.set_yticks(np.arange(0, ymax + 1e-9, ytick_step))
    ax.yaxis.grid(True, linewidth=0.35, linestyle=":", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.785),
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.5, rect=[0.0, 0.0, 1.0, 0.72])
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)


def main():

    total, refresh = build_values()
    draw_grouped(
        total,
        "Normalized total DRAM energy",
        "arch_compare_total_energy_seq.pdf",
        value_fmt="{:.2f}",
        ylim_pad=0.35,
        ytick_step=1.0,
        cap=6.0,
    )
    draw_grouped(
        refresh,
        "Normalized refresh energy",
        "arch_compare_refresh_energy_seq.pdf",
        value_fmt="{:.2f}",
        ylim_pad=0.35,
        ytick_step=1.0,
        cap=6.0,
    )


if __name__ == "__main__":
    main()
