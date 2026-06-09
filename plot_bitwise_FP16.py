#!/usr/bin/env python3
"""
Combined bit-sensitivity plotting script.
Produces three figures per run:
  1. baseline_bars  – absolute metric values per bit, bars vs baseline (from script A)
  2. vs_bit         – degradation (Δ%) per bit, bar chart            (from script B)
  3. intuition      – heatmap + worst-case bar summary               (from script B)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from bitwise_input import as_bar_rows, load_metric_rows

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURE_ROOT = Path(__file__).resolve().parent          # /home/wing02/figures/
ROOT        = FIGURE_ROOT.parent                       # /home/wing02/
FIGURE_DIR  = Path(__file__).resolve().parent

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL             = "llama"
DEFAULT_METRIC            = "auto"
DEFAULT_TREFI             = [4096]
DEFAULT_YMAX_FACTOR       = 1.45
DEFAULT_CATASTROPHIC_RATIO = 100.0

METRIC_CHOICES = [
    "auto", "ppl", "cross_entropy_loss",
    "negative_log_likelihood", "logit_margin",
    "top1_accuracy", "top5_accuracy",
]
CSV_ROOTS = {
    "ppl": FIGURE_ROOT / "analysis_csv",
    "acc": FIGURE_ROOT / "analysis_acc_csv",
}
BASELINE_FILES = {
    "opt":      ROOT / "results" / "baseline"          / "baseline_fp16_opt.harness.json",
    "llama":    ROOT / "results" / "baseline"          / "baseline_fp16_llama.harness.json",
    "resnet50": ROOT / "results" / "baseline_resnet50" / "baseline_resnet50.harness.json",
}
METRIC_DEFAULTS = {
    "opt":      "ppl",
    "llama":    "ppl",
    "resnet50": "cross_entropy_loss",
}
import matplotlib.font_manager as fm
# ── 폰트 등록  ──────────────────────────────────────────────────────────
ARIAL_NARROW_BOLD = 'arialnarrow_bold.ttf'
fm.fontManager.addfont(ARIAL_NARROW_BOLD)
_font_prop = fm.FontProperties(fname=ARIAL_NARROW_BOLD)

plt.rcParams.update({
    'font.family':      _font_prop.get_name(),
    'font.weight':      'bold',
    'font.size':        15,
    'axes.titlesize':   15,
    'axes.labelsize':   15,
    'xtick.labelsize':  15,
    'ytick.labelsize':  15,
    'axes.linewidth':   0.6,
    'xtick.major.width':0.5,
    'ytick.major.width':0.5,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.major.pad':  2,
    'ytick.major.pad':  2,
})

# ── Shared utilities ──────────────────────────────────────────────────────────

def _coerce_float(value: Any) -> float | None:
    if value in (None, "", "null", "None", "N/A"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:  return math.inf
    if text in {"-inf", "-infinity"}:                     return -math.inf
    if text == "nan":                                      return math.nan
    return float(value)


def infer_family_from_metric(metric: str) -> str:
    return "ppl" if metric == "ppl" else "acc"


def choose_metric(model: str, metric: str) -> str:
    return METRIC_DEFAULTS.get(model, "cross_entropy_loss") if metric == "auto" else metric


def metric_label(metric: str) -> str:
    return {
        "ppl":                    "Perplexity (PPL)",
        "cross_entropy_loss":     "Cross-Entropy Loss",
        "negative_log_likelihood":"Negative Log-Likelihood",
        "logit_margin":           "Logit Margin",
        "top1_accuracy":          "Top-1 Accuracy",
        "top5_accuracy":          "Top-5 Accuracy",
    }.get(metric, metric)


def metric_label_short(metric: str) -> str:
    return {
        "ppl":               "PPL",
        "cross_entropy_loss":"Cross-Entropy Loss",
        "top1_accuracy":     "Top-1 Accuracy",
        "top5_accuracy":     "Top-5 Accuracy",
    }.get(metric, metric)


def default_csv(model: str, metric: str, dram: str = "lpddr5") -> Path:
    root       = CSV_ROOTS[infer_family_from_metric(metric)]
    candidates = sorted(root.glob(f"{model}__fp16__bitwise16__*__{dram}__*.csv"))
    if not candidates:
        candidates = sorted(root.glob(f"{model}__fp16__bitwise16__*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No bitwise16 CSV for model={model} under {root}")
    return candidates[0]


def resolve_baseline(model: str, metric: str, rows: list[dict]) -> float:
    path = BASELINE_FILES.get(model)
    if path and path.exists():
        payload = json.loads(path.read_text("utf-8"))
        for src in (payload, payload.get("harness_result", {})):
            v = _coerce_float(src.get(metric)) if isinstance(src, dict) else None
            if v is not None and math.isfinite(v):
                return v
    finite = [r["value"] for r in rows
              if isinstance(r.get("value"), float) and math.isfinite(r["value"])]
    if not finite:
        raise RuntimeError(f"No finite values for metric={metric}")
    return max(finite) if metric in {"top1_accuracy", "top5_accuracy", "logit_margin"} else min(finite)


def build_maps(rows: list[dict]) -> tuple[dict, dict]:
    values: dict[int, dict[int, float | None]] = {}
    bers:   dict[int, float] = {}
    for r in rows:
        values.setdefault(r["trefi"], {})[r["bit"]] = r["value"]
        bers[r["trefi"]] = r["ber"]
    return values, bers


def metric_delta_pct(value: float | None, baseline: float, metric: str) -> float | None:
    if value is None or math.isnan(value):
        return None
    delta = (baseline - value) if metric in {"top1_accuracy","top5_accuracy","logit_margin"} \
            else (value - baseline)
    denom = abs(baseline) if baseline else 1.0
    return (delta / denom) * 100.0


def sensitivity_score(delta_pct: float | None) -> float | None:
    if delta_pct is None or math.isnan(delta_pct):  return None
    if math.isinf(delta_pct):                        return 12.0
    return math.log10(max(delta_pct, 1e-3))


def fmt_delta_pct(delta_pct: float | None) -> str:
    if delta_pct is None or math.isnan(delta_pct):  return "NaN"
    if math.isinf(delta_pct):                        return ">100x"
    if delta_pct < 0.05:   return "0"
    if delta_pct < 10.0:   return f"+{delta_pct:.1f}%"
    if delta_pct < 100.0:  return f"+{delta_pct:.0f}%"
    ratio = delta_pct / 100.0
    return ">100x" if ratio > 100.0 else f"{ratio:.1f}x"


def label_value(value: float | None, baseline: float, cat_ratio: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    if math.isinf(value):
        return f">{cat_ratio:.0f}x"
    ratio = value / baseline if baseline else math.inf
    if ratio >= cat_ratio:
        return f">{cat_ratio:.0f}x"
    return f"{value:.3f}" if value < 1000 else f"{ratio:.1f}x"


def save_fig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {path}")


def add_field_bg(ax, zorder=0):
    ax.axvspan(-0.5,  0.5, alpha=0.07, color="#C00000", zorder=zorder)
    ax.axvspan( 0.5,  5.5, alpha=0.07, color="#ED7D31", zorder=zorder)
    ax.axvspan( 5.5, 15.5, alpha=0.05, color="#1565c0", zorder=zorder)


def add_field_labels_axes(ax, y=0.93):
    for x, label, color in [(0.04,"sign","#C00000"),(0.22,"exponent","#ED7D31"),(0.68,"mantissa","#1565c0")]:
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=20, fontweight="bold", color=color,
                bbox={"facecolor":"white","edgecolor":"none","alpha":0.72,"pad":1.5}, zorder=10)


def parse_trefi_list(text: str | None) -> list[int]:
    if not text:
        return DEFAULT_TREFI
    return [int(t) for t in text.replace(",", " ").split()]


# ── Figure 1: Baseline bars ───────────────────────────────────────────────────

def plot_baseline_bars(model, metric, csv_path, trefi_list, output, ymax, cat_ratio):
    """Absolute metric values per bit, with baseline reference line."""
    rows     = as_bar_rows(load_metric_rows(csv_path, model, metric))
    baseline = resolve_baseline(model, metric, rows)
    values, bers = build_maps(rows)
    trefi_sub = [t for t in trefi_list if t in values]
    if not trefi_sub:
        raise RuntimeError(f"No matching tREFI in data: {trefi_list}")

    bits    = list(range(15, -1, -1))
    x       = np.arange(len(bits))
    bar_w   = min(0.78 / len(trefi_sub), 0.34)
    palette = ["#2f6690","#57986a","#d4873b","#9a4b7a","#666666","#1b9e77","#d95f02"]

    finite_under = [v for t in trefi_sub for v in values[t].values()
                    if isinstance(v, float) and math.isfinite(v) and baseline and v/baseline < cat_ratio]
    if ymax is None:
        ymax = max(max(finite_under + [baseline]) * DEFAULT_YMAX_FACTOR, baseline * 1.2)

    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    add_field_bg(ax)

    for i, trefi in enumerate(trefi_sub):
        color  = palette[i % len(palette)]
        offset = (i - (len(trefi_sub) - 1) / 2) * bar_w
        for xi, bit in enumerate(bits):
            value  = values[trefi].get(bit)
            xpos   = x[xi] + offset
            if value is None or (isinstance(value, float) and math.isnan(value)):
                ax.bar(xpos, ymax, width=bar_w*1.5, color=color, alpha=0.38,
                       edgecolor="white", linewidth=0.5, hatch="//", zorder=3)
            elif math.isinf(value) or (baseline and value/baseline >= cat_ratio):
                ax.bar(xpos, ymax, width=bar_w*1.5, color=color, alpha=0.9,
                       edgecolor="white", linewidth=0.5, zorder=3)
            else:
                ax.bar(xpos, min(float(value), ymax), width=bar_w*1.5, color=color,
                       alpha=0.9, edgecolor="white", linewidth=0.5, zorder=3)

            text   = label_value(value, baseline, cat_ratio)
            text_y = min(max(ymax * 0.92, baseline * 1.02), ymax * 0.92)
            ax.text(xpos, text_y * 0.92, text, ha="center", va="top",
                    fontsize=7.2 if len(trefi_sub) <= 2 else 6.2,
                    color="white", fontweight="bold",
                    rotation=90 if len(trefi_sub) > 2 else 0, zorder=5)

    ax.axhline(baseline, color="#c44e52", linewidth=1.4, linestyle="--", zorder=8)
    ax.text(10, baseline + 1.5, f" baseline {baseline:.3f}",
            ha="left", va="center", fontsize=10, color="#9d2f33", fontweight="bold")

    add_field_labels_axes(ax)
    ax.set_xticks(x);  ax.set_xticklabels([str(b) for b in bits], fontsize=9)
    ax.set_xlim(-0.7, 15.7);  ax.set_ylim(0.0, ymax)
    ax.set_xlabel("FP16 Bit Index", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_label(metric), fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.set_title("Bit Sensitivity Analysis — Absolute Values", fontsize=15, fontweight="bold", pad=20)
    subtitle = (f"Model: {model} | DRAM: LPDDR5 | Metric: {metric_label(metric)} | "
                f"tREFW: {', '.join(str(t) for t in trefi_sub)} ms")
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#666666")

    handles = [plt.Line2D([0],[0], color="#c44e52", linewidth=1.4, linestyle="--",
                           label=f"Baseline ({baseline:.3f})")]
    for i, t in enumerate(trefi_sub):
        handles.append(mpatches.Patch(facecolor=palette[i % len(palette)], alpha=0.9,
                                      label=f"tREFI={t} ms (BER={bers[t]:.1e})"))
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    save_fig(fig, output)
    plt.close(fig)


# ── Figure 2: Δ% vs bit ──────────────────────────────────────────────────────

def plot_vs_bit(model, metric, csv_path, trefi_list, output):
    """Degradation (Δ%) per bit position, bar chart."""
    rows     = as_bar_rows(load_metric_rows(csv_path, model, metric))  # ← as_bar_rows 추가
    baseline = resolve_baseline(model, metric, rows)
    raw, bers = build_maps(rows)
    trefi_sub = [t for t in trefi_list if t in raw] or sorted(raw)

    bits_x  = list(range(15, -1, -1))
    x       = np.arange(len(bits_x))
    bar_w   = 0.72 / max(1, len(trefi_sub))
    palette = ["#08306b","#2171b5","#6baed6","#fdd49e","#fdbb84","#e34a33","#99000d"]
    colors  = {t: palette[i % len(palette)] for i, t in enumerate(trefi_sub)}

    finite_dpcts = [
        (1e9 if math.isinf(d) else max(d, 0.0))
        for t in trefi_sub for b in bits_x
        if (d := metric_delta_pct(raw[t].get(b), baseline, metric)) is not None and not math.isnan(d)
    ]
    nan_h = max(finite_dpcts + [1.0])

    fig, ax = plt.subplots(figsize=(15, 6.2))
    for i, t in enumerate(trefi_sub):
        offsets = x + (i - (len(trefi_sub) - 1) / 2) * bar_w
        for ox, b in zip(offsets, bits_x):
            dpct = metric_delta_pct(raw[t].get(b), baseline, metric)
            if dpct is None or math.isnan(dpct):
                ax.bar(ox, nan_h, width=bar_w, color="#6b6b6b", alpha=0.72,
                       edgecolor="white", linewidth=0.35, hatch="//", zorder=3)
                ax.text(ox, nan_h*0.5, "NaN", ha="center", va="center",
                        fontsize=7, rotation=90, color="white", fontweight="bold", zorder=5)
            else:
                if math.isinf(dpct): dpct = 1e9
                ax.bar(ox, max(dpct, 0.0), width=bar_w, color=colors[t],
                       alpha=0.88, edgecolor="white", linewidth=0.35, zorder=3)

    ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--", zorder=10)
    add_field_bg(ax)
    ymax_cur = ax.get_ylim()[1] or baseline * 1.5
    for xpos, label, color in [(0, "Sign", "#C00000"),(3, "Exponent","#ED7D31"),(10.5,"Mantissa","#1565c0")]:
        ax.text(xpos, ymax_cur * 0.92, label, ha="center", va="bottom",
                fontsize=20, color=color, fontweight="bold")

    ax.set_xticks(x);  ax.set_xticklabels([str(b) for b in bits_x], fontsize=9)
    ax.set_xlabel("FP16 Bit Index  (15 = MSB, 0 = LSB)", fontsize=12)
    ax.set_ylabel(f"Degradation vs Baseline ({metric_label_short(metric)} Δ%)", fontsize=12)
    ax.set_title(f"{model} FP16 Bit Sensitivity — Δ% per Bit", fontsize=14)
    ax.set_xlim(-0.6, len(bits_x) - 0.4)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4, zorder=0)

    handles = [mpatches.Patch(facecolor=colors[t], alpha=0.85,
                               label=f"tREFI={t} ms  (BER≈{bers[t]:.1e})") for t in trefi_sub]
    handles.append(plt.Line2D([0],[0], color="black", linewidth=1.2, linestyle="--",
                               label=f"Baseline Δ=0  ({metric_label_short(metric)}={baseline:.3f})"))
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, output)
    plt.close(fig)


# ── Figure 3: 히트맵만 ────────────────────────────────────────────────────────

def plot_heatmap_only(model, metric, csv_path, trefi_list, output):
    rows      = as_bar_rows(load_metric_rows(csv_path, model, metric))
    baseline  = resolve_baseline(model, metric, rows)
    raw, bers = build_maps(rows)
    trefi_sub = [t for t in trefi_list if t in raw] or sorted(raw)
    bits_x    = list(range(15, -1, -1))

    heat   = np.full((len(trefi_sub), len(bits_x)), np.nan)
    labels: list[list[str]] = []
    for ri, t in enumerate(trefi_sub):
        row_lbls = []
        for ci, b in enumerate(bits_x):
            dpct  = metric_delta_pct(raw[t].get(b), baseline, metric)
            score = sensitivity_score(dpct)
            heat[ri, ci] = np.nan if score is None else score
            row_lbls.append(fmt_delta_pct(dpct))
        labels.append(row_lbls)

    cmap = plt.get_cmap("YlOrRd").copy();  cmap.set_bad("#4d4d4d")
    norm = mcolors.Normalize(vmin=-2.0, vmax=4.0)

    fig, ax = plt.subplots(figsize=(15, 3.2 + len(trefi_sub) * 0.6))
    im = ax.imshow(heat, cmap=cmap, norm=norm, aspect="auto", zorder=2)

    for ri in range(len(trefi_sub)):
        for ci in range(len(bits_x)):
            txt = labels[ri][ci]
            val = heat[ri, ci]
            fg  = "white" if (not math.isnan(val) and val > 2.0) or txt == "NaN" else "black"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=8.2, fontweight="bold", color=fg, zorder=3)

    for ax_ in [ax]:
        ax_.axvspan(-0.5,  0.5, alpha=0.07, color="#C00000", zorder=2.2)
        ax_.axvspan( 0.5,  5.5, alpha=0.07, color="#ED7D31", zorder=2.2)
        ax_.axvspan( 5.5, 15.5, alpha=0.05, color="#1565c0", zorder=2.2)

    ax.set_yticks(np.arange(len(trefi_sub)))
    ax.set_yticklabels([f"{t} ms\n{bers[t]:.1e}" for t in trefi_sub], fontsize=9)
    ax.set_xticks(np.arange(len(bits_x)))
    ax.set_xticklabels([str(b) for b in bits_x], fontsize=9)
    ax.set_xlabel("FP16 Bit Index", fontsize=12)
    ax.set_ylabel("LPDDR5 tREFW / BER", fontsize=11)
    ax.set_title(f"{model} FP16 Bit Sensitivity — {metric_label(metric)} Degradation (log10 Δ%)", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("log10(Δ%);  darker = more sensitive", fontsize=9)

    fig.tight_layout()
    save_fig(fig, output)
    plt.close(fig)


# ── Figure 4: Worst-case bars만 ───────────────────────────────────────────────

def plot_worstcase_bars(model, metric, csv_path, trefi_list, output):
    import matplotlib.font_manager as fm

    ARIAL_NARROW_BOLD = '/home/wing02/.fonts/arialnarrow_bold.ttf'
    rows      = as_bar_rows(load_metric_rows(csv_path, model, metric))
    baseline  = resolve_baseline(model, metric, rows)
    raw, bers = build_maps(rows)
    trefi_sub = [t for t in trefi_list if t in raw] or sorted(raw)
    bits_x    = list(range(15, -1, -1))
    x         = np.arange(len(bits_x))

    field_colors = {b: ("#C00000" if b == 15 else "#ED7D31" if b >= 10 else "#1565c0")
                    for b in range(16)}
    worst: dict[int, float] = {}
    nan_cnt: dict[int, int] = {}
    for b in bits_x:
        vals, nc = [], 0
        for t in trefi_sub:
            dpct = metric_delta_pct(raw[t].get(b), baseline, metric)
            if dpct is None or math.isnan(dpct):
                nc += 1
            else:
                vals.append(1e9 if math.isinf(dpct) else max(dpct, 0.0))
        worst[b]   = max(vals) if vals else math.nan
        nan_cnt[b] = nc

    finite_worst = [v for v in worst.values() if not math.isnan(v)]
    # nan_h = max(finite_worst + [1.0])

    fig, ax = plt.subplots(figsize=(15, 5.0))

   
    #cap_h = max(finite_worst + [1.0])  # NaN/INF 바 고정 높이 (일반 최대값의 20배)
    cap_h = 1e4
    for xi, b in enumerate(bits_x):
        pct = worst[b]

        # ── NaN (측정 불가) ──────────────────────────────────────
        if math.isnan(pct):
            ax.bar(xi, cap_h, width=0.58, color="#6b6b6b", alpha=0.72,
                edgecolor="white", linewidth=0.6,  zorder=3)
            ax.text(xi, cap_h * 0.01, "NaN", ha="center", va="bottom",
                    color="white", fontsize=20, fontweight="bold", rotation=90, zorder=6)
            continue

        # ── >100x (발산 수준) ─────────────────────────────────────
        if math.isinf(pct) or pct >= 1e6:
            ax.bar(xi, cap_h, width=0.58, color=field_colors[b], alpha=0.78,
                edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(xi, cap_h * 0.01, ">100x", ha="center", va="bottom",
                    color="white", fontsize=20, fontweight="bold", rotation=90, zorder=6)
            if nan_cnt[b] > 0:
                ax.text(xi, cap_h * 0.5, f"NaN×{nan_cnt[b]}",
                        ha="center", va="center", color="white",
                        fontsize=20, fontweight="bold", rotation=90, zorder=6)
            continue

        # ── 일반 값 ──────────────────────────────────────────────
        ax.bar(xi, pct, width=0.58, color=field_colors[b], alpha=0.78,
            edgecolor="white", linewidth=0.6, zorder=3)
        if pct >= 10.0:
            ax.text(xi, pct * 0.005, fmt_delta_pct(pct),
                    ha="center", va="bottom", color="white", fontsize=20,
                    fontweight="bold", rotation=90, zorder=6)
        # elif pct >= 1.0:
            # ax.text(xi, pct * 0.02, fmt_delta_pct(pct),
            #         ha="center", va="bottom", color="white", fontsize=20,
            #         fontweight="bold", rotation=90, zorder=6)
        elif nan_cnt[b] > 0:
            ax.text(xi, max(pct, 0.2), f"NaN×{nan_cnt[b]}",
                    ha="center", va="bottom", color="#4d4d4d",
                    fontsize=20, fontweight="bold", rotation=90, zorder=6)

# y축 범위도 cap_h 기준으로
    
    add_field_bg(ax)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", zorder=5)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=ax.get_ylim()[0], top=1e4)
    #ax.set_ylim(0.0, max(finite_worst + [1.0]) * 5.0)
    ax.set_xticks(x);  ax.set_xticklabels([str(b) for b in bits_x], fontsize=15)
    ax.set_xlabel("FP16 Bit Index", fontsize=20)
    ax.set_ylabel("PPL Increase (%)", fontsize=20)
    #ax.set_title("Worst-case degradation by bit position", fontsize=15)
    ax.set_xlim(-0.6, len(bits_x) - 0.4)
    
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.35, zorder=0)
    add_field_labels_axes(ax)

    fig.tight_layout()
    save_fig(fig, output)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bit-sensitivity analysis: 3 figures in one run")
    p.add_argument("--model",  default=DEFAULT_MODEL, help="llama | opt | resnet50")
    p.add_argument("--metric", default=DEFAULT_METRIC, choices=METRIC_CHOICES)
    p.add_argument("--dram",   default="lpddr5")
    p.add_argument("--csv",    type=Path, default=None, help="Override CSV path")
    p.add_argument("--trefi",  default=None, help="tREFI ms list, e.g. '2048 4096'")
    p.add_argument("--ymax",   type=float, default=None, help="Manual y-max for figure 1")
    p.add_argument("--catastrophic-ratio", type=float, default=DEFAULT_CATASTROPHIC_RATIO)
    p.add_argument("--out-bars",      type=Path, default=None)
    p.add_argument("--out-vs-bit",    type=Path, default=None)
    p.add_argument("--out-intuition", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    metric     = choose_metric(args.model, args.metric)
    csv_path   = args.csv or default_csv(args.model, metric, args.dram)
    trefi_list = parse_trefi_list(args.trefi)

    tag = f"{args.model}_{args.dram}_{metric}"
    out_bars      = args.out_bars      or FIGURE_DIR / f"{tag}_baseline_bars.png"
    out_vs_bit    = args.out_vs_bit    or FIGURE_DIR / f"{tag}_vs_bit.png"
    out_heatmap   = FIGURE_DIR / f"{tag}_heatmap.png"       # ← 신규
    out_worstcase = FIGURE_DIR / f"{tag}_worstcase.png"     # ← 신규

    # print(f"\n=== Figure 1/4: baseline bars  →  {out_bars}")
    # plot_baseline_bars(args.model, metric, csv_path, trefi_list,
    #                    out_bars, args.ymax, args.catastrophic_ratio)

    # print(f"\n=== Figure 2/4: Δ%  vs  bit   →  {out_vs_bit}")
    # plot_vs_bit(args.model, metric, csv_path, trefi_list, out_vs_bit)

    # print(f"\n=== Figure 3/4: heatmap       →  {out_heatmap}")
    # plot_heatmap_only(args.model, metric, csv_path, trefi_list, out_heatmap)

    print(f"\n=== Figure 4/4: worst-case    →  {out_worstcase}")
    plot_worstcase_bars(args.model, metric, csv_path, trefi_list, out_worstcase)

    print("\nAll done.")


if __name__ == "__main__":
    main()