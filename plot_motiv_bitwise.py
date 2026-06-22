#!/usr/bin/env python3
"""
Produces:
  - outputs/fig_bitwise.pdf/png         : bitwise worst-case bar     (standalone)
"""

from __future__ import annotations
import json, math, os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from bitwise_input import as_bar_rows, load_metric_rows

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURE_DIR  = Path(__file__).resolve().parent
FIGURE_ROOT = FIGURE_DIR
ROOT        = FIGURE_DIR.parent
OUT         = FIGURE_DIR / "outputs"
OUT.mkdir(exist_ok=True)

# ── Font ──────────────────────────────────────────────────────────────────────
_FONT_PATH = '/home/wing02/arialnarrow_bold.ttf'
if Path(_FONT_PATH).exists():
    fm.fontManager.addfont(_FONT_PATH)
    _FONT_NAME = fm.FontProperties(fname=_FONT_PATH).get_name()
else:
    _FONT_NAME = 'DejaVu Sans'

plt.rcParams.update({
    'font.family':       _FONT_NAME,
    'font.weight':       'bold',
    'font.size':         11,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'axes.linewidth':    0.7,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.major.pad':   2,
    'ytick.major.pad':   2,
})

# ── Bitwise config ────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "llama"
DEFAULT_METRIC = "ppl"
DEFAULT_TREFI  = [4096]
DEFAULT_CAT    = 100.0

CSV_ROOTS = {
    "ppl": FIGURE_ROOT / "analysis_csv",
    "acc": FIGURE_ROOT / "analysis_acc_csv",
}
BASELINE_FILES = {
    "llama":    ROOT / "results" / "baseline" / "baseline_fp16_llama.harness.json",
    "opt":      ROOT / "results" / "baseline" / "baseline_fp16_opt.harness.json",
    "resnet50": ROOT / "results" / "baseline_resnet50" / "baseline_resnet50.harness.json",
}
METRIC_DEFAULTS = {"llama": "ppl", "opt": "ppl", "resnet50": "cross_entropy_loss"}

FORMAT_ALIASES = {
    "fp16": "fp16",
    "bf16": "bf16",
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "e4m3": "fp8_e4m3",
}
FORMAT_SPECS = {
    "fp16": {
        "file_tag": "fp16",
        "label": "FP16",
        "bits": 16,
        "fields": [
            ("Sign", 15, 15, "#3E2E5E"),
            ("Exponent", 14, 10, "#3E2E5E"),
            ("Mantissa", 9, 0, "#3E2E5E"),
        ],
    },
    "bf16": {
        "file_tag": "bf16",
        "label": "BF16",
        "bits": 16,
        "fields": [
            ("Sign", 15, 15, "#3E2E5E"),
            ("Exponent", 14, 7, "#3E2E5E"),
            ("Mantissa", 6, 0, "#3E2E5E"),
        ],
    },
    "fp8_e4m3": {
        "file_tag": "fp8_e4m3",
        "label": "FP8 E4M3",
        "bits": 8,
        "fields": [
            ("Sign", 7, 7, "#3E2E5E"),
            ("Exponent", 6, 3, "#3E2E5E"),
            ("Mantissa", 2, 0, "#3E2E5E"),
        ],
    },
}

# ── Shared helpers ────────────────────────────────────────────────────────────

def _coerce_float(v: Any) -> float | None:
    if v in (None, "", "null", "None", "N/A"): return None
    if isinstance(v, (int, float)):            return float(v)
    t = str(v).strip().lower()
    if t in {"inf", "+inf"}:  return math.inf
    if t in {"-inf"}:         return -math.inf
    if t == "nan":            return math.nan
    return float(v)

def canonical_format(fmt):
    key = (fmt or "fp16").lower().replace("-", "_")
    if key not in FORMAT_ALIASES:
        supported = ", ".join(sorted(FORMAT_ALIASES))
        raise ValueError(f"Unsupported format={fmt!r}; expected one of: {supported}")
    return FORMAT_ALIASES[key]


def choose_metric(model, metric):
    return METRIC_DEFAULTS.get(model, "ppl") if metric == "auto" else metric


def csv_search_roots(metric):
    primary = CSV_ROOTS["ppl" if metric == "ppl" else "acc"]
    roots = []
    for root in (primary, CSV_ROOTS["ppl"], CSV_ROOTS["acc"]):
        if root not in roots:
            roots.append(root)
    return roots


def default_csv(model, metric, fmt="fp16", dram="lpddr5"):
    fmt = canonical_format(fmt)
    spec = FORMAT_SPECS[fmt]
    tag = spec["file_tag"]
    bits = spec["bits"]
    patterns = [
        f"{model}__{tag}__bitwise{bits}__trefi*__{dram}__*.csv",
        f"{model}__{tag}__bitwise{bits}__*__{dram}__*.csv",
        f"{model}__{tag}__bitwise{bits}__*.csv",
        f"{model}__{tag}__bitwise*__*{dram}*.csv",
        f"{model}__{tag}__bitwise*__*.csv",
        f"*{model}*{tag}*bitwise*.csv",
    ]
    searched = []
    for root in csv_search_roots(metric):
        searched.append(str(root))
        for pat in patterns:
            cs = sorted(root.glob(pat))
            if cs:
                return cs[0]
    raise FileNotFoundError(
        f"No CSV for model={model}, format={fmt}, metric={metric} under "
        f"{', '.join(searched)}"
    )


def bit_order(fmt):
    spec = FORMAT_SPECS[canonical_format(fmt)]
    return list(range(spec["bits"] - 1, -1, -1))


def format_field_spans(fmt):
    spec = FORMAT_SPECS[canonical_format(fmt)]
    nbits = spec["bits"]
    for label, high_bit, low_bit, color in spec["fields"]:
        left = (nbits - 1 - high_bit) - 0.5
        right = (nbits - 1 - low_bit) + 0.5
        yield left, right, label, color


def metric_axis_label(metric):
    if metric in {"top1_accuracy", "top5_accuracy", "logit_margin"}:
        return "Accuracy loss (%)"
    if metric == "cross_entropy_loss":
        return "Cross-entropy increase (%)"
    return "PPL increase (%)"

def resolve_baseline(model, metric, rows, baseline_override=None):
    if baseline_override is not None:
        return float(baseline_override)
    p = BASELINE_FILES.get(model)
    if p and p.exists():
        payload = json.loads(p.read_text("utf-8"))
        for src in (payload, payload.get("harness_result", {})):
            v = _coerce_float(src.get(metric)) if isinstance(src, dict) else None
            if v is not None and math.isfinite(v): return v
    finite = [r["value"] for r in rows
              if isinstance(r.get("value"), float) and math.isfinite(r["value"])]
    if not finite: raise RuntimeError(f"No finite values for {metric}")
    return max(finite) if metric in {"top1_accuracy","top5_accuracy","logit_margin"} else min(finite)

def build_maps(rows):
    vals, bers = {}, {}
    for r in rows:
        vals.setdefault(r["trefi"], {})[r["bit"]] = r["value"]
        bers[r["trefi"]] = r["ber"]
    return vals, bers

def metric_delta_pct(value, baseline, metric):
    if value is None or math.isnan(value): return None
    delta = (baseline - value) if metric in {"top1_accuracy","top5_accuracy","logit_margin"} \
            else (value - baseline)
    return (delta / abs(baseline)) * 100.0 if baseline else None

def fmt_delta_pct(d):
    if d is None or math.isnan(d): return "NaN"
    if math.isinf(d):               return ">100x"
    if d < 0.05:  return "0"
    if d < 10.0:  return f"+{d:.1f}%"
    if d < 100.0: return f"+{d:.0f}%"
    r = d / 100.0
    return ">100x" if r > 100.0 else f"{r:.1f}x"

def save_fig(fig, path: Path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {path}  +  {path.with_suffix('.pdf')}")


# ── Panel B: bitwise worst-case bars ─────────────────────────────────────────

def draw_bitwise(ax, model=DEFAULT_MODEL, metric=DEFAULT_METRIC,
                 csv_path=None, trefi_list=None, fmt="fp16", baseline=None):
    if trefi_list is None: trefi_list = DEFAULT_TREFI
    fmt = canonical_format(fmt)
    if csv_path   is None: csv_path   = default_csv(model, metric, fmt=fmt)

    rows      = as_bar_rows(load_metric_rows(csv_path, model, metric))
    baseline  = resolve_baseline(model, metric, rows, baseline_override=baseline)
    raw, bers = build_maps(rows)
    trefi_sub = [t for t in trefi_list if t in raw] or sorted(raw)

    spec   = FORMAT_SPECS[fmt]
    bits_x = bit_order(fmt)
    x      = np.arange(len(bits_x))

    worst, nan_cnt = {}, {}
    for b in bits_x:
        vals, nc = [], 0
        for t in trefi_sub:
            d = metric_delta_pct(raw[t].get(b), baseline, metric)
            if d is None or math.isnan(d): nc += 1
            else: vals.append(1e9 if math.isinf(d) else max(d, 0.0))
        worst[b]   = max(vals) if vals else math.nan
        nan_cnt[b] = nc

    cap_h = 1e4

    for xi, b in enumerate(bits_x):
        pct = worst[b]
        if math.isnan(pct):
            ax.bar(xi, cap_h, width=0.58, color='#3E2E5E', alpha=0.72,
                   edgecolor="black", lw=S(1), zorder=3)
            ax.text(xi, cap_h * 0.01, "NaN", ha="center", va="bottom",
                    color="white", fontsize=S(11), fontweight="bold", rotation=90, zorder=6)
        elif math.isinf(pct) or pct >= 1e6:
            ax.bar(xi, cap_h, width=0.58, color='#3E2E5E', alpha=0.78,
                   edgecolor="black", lw=S(1), zorder=3)
            ax.text(xi, cap_h * 0.005, ">100x", ha="center", va="bottom",
                    color="white", fontsize=S(11), fontweight="bold", rotation=90, zorder=6)
        else:
            ax.bar(xi, pct, width=0.58, color='#3E2E5E', alpha=0.78,
                   edgecolor="black", lw=S(1), zorder=3)
            lbl = fmt_delta_pct(pct)
            # if pct >= 10.0:
            #     ax.text(xi, pct * 0.005, lbl, ha="center", va="bottom",
            #             color="white", fontsize=S(11), fontweight="bold", rotation=90, zorder=6)
            #elif 0.05 <= pct < 10.0:
                # ax.text(xi, pct * 1.3, lbl, ha="center", va="bottom",
                #         color="black", fontsize=S(9), fontweight="bold", rotation=90, zorder=6)

    # field background
    for left, right, _, _ in format_field_spans(fmt):
        ax.axvspan(left, right, alpha=0.08, color='#b0b0b0', zorder=0)

    # ax.axvspan(-0.5,  0.5, alpha=0.07, color="#C00000", zorder=0)
    # ax.axvspan( 0.5,  5.5, alpha=0.07, color="#ED7D31", zorder=0)
    # ax.axvspan( 5.5, 15.5, alpha=0.05, color="#1565c0", zorder=0)


    # field labels
    # for xf, lbl, col in [(0.04,"Sign","#C00000"),(0.22,"Exponent","#ED7D31"),(0.68,"Mantissa","#1565c0")]:
    #     ax.text(xf, 0.93, lbl, transform=ax.transAxes, ha="center", va="center",
    #             fontsize=12, fontweight="bold", color=col,
    #             bbox={"facecolor":"white","edgecolor":"none","alpha":0.72,"pad":1.5}, zorder=10)

    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0, top=cap_h)
    ax.set_xticks(x);  ax.set_xticklabels([str(b) for b in bits_x], fontsize=S(10))
    ax.set_xlim(-0.6, len(bits_x) - 0.4)
    ax.set_xlabel(f"{spec['label']} bit index", fontweight='bold')
    ax.set_ylabel(metric_axis_label(metric), fontweight='bold')
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.75, zorder=4)
    ax.axhline(0.0, color="black", linewidth=S(0.8), linestyle="--", zorder=7)
    return ax


def draw_bitwise_2(ax, model=DEFAULT_MODEL, metric=DEFAULT_METRIC,
                   csv_path=None, trefi_list=None, fmt="fp16", baseline=None):
    """Bitwise worst-case bars with explicit floating-point field annotations."""
    if trefi_list is None: trefi_list = DEFAULT_TREFI
    fmt = canonical_format(fmt)
    if csv_path   is None: csv_path   = default_csv(model, metric, fmt=fmt)

    rows      = as_bar_rows(load_metric_rows(csv_path, model, metric))
    baseline  = resolve_baseline(model, metric, rows, baseline_override=baseline)
    raw, _    = build_maps(rows)
    trefi_sub = [t for t in trefi_list if t in raw] or sorted(raw)

    spec   = FORMAT_SPECS[fmt]
    bits_x = bit_order(fmt)
    x      = np.arange(len(bits_x))
    # field_color = {
    #     "sign": "#C00000",
    #     "exponent": "#ED7D31",
    #     "mantissa": "#1565c0",
    # }
    field_color = {
        "sign": "#3E2E5E",
        "exponent": "#3E2E5E",
        "mantissa": "#3E2E5E",
    }
    field_bg = {
        "sign": "#FFFFFF",
        "exponent": "#FFFFFF",
        "mantissa": "#FFFFFF",
    }

    worst, nan_cnt = {}, {}
    for b in bits_x:
        vals, nc = [], 0
        for t in trefi_sub:
            d = metric_delta_pct(raw[t].get(b), baseline, metric)
            if d is None or math.isnan(d): nc += 1
            else: vals.append(1e9 if math.isinf(d) else max(d, 0.0))
        worst[b]   = max(vals) if vals else math.nan
        nan_cnt[b] = nc

    cap_h = 1e4

    field_spans = [
        (left, right, label, color, "#FFFFFF")
        for left, right, label, color in format_field_spans(fmt)
    ]
    for left, right, _, _, _ in field_spans:
        ax.axvspan(left, right, alpha=0.05, color='#000000', zorder=0)

    for xi, b in enumerate(bits_x):
        pct = worst[b]
        if math.isnan(pct):
            ax.bar(xi, cap_h, width=0.58, color='#3E2E5E', alpha=0.72,
                   edgecolor="black", lw=S(0.8), zorder=3)
            ax.text(xi, cap_h * 0.01, "NaN", ha="center", va="bottom",
                    color="white", fontsize=S(10), fontweight="bold", rotation=90, zorder=6)
        elif math.isinf(pct) or pct >= 1e6:
            ax.bar(xi, cap_h, width=0.58, color='#3E2E5E', alpha=0.78,
                   edgecolor="black", lw=S(0.8), zorder=3)
            ax.text(xi, cap_h * 0.005, ">100x", ha="center", va="bottom",
                    color="white", fontsize=S(10), fontweight="bold", rotation=90, zorder=6)
        else:
            ax.bar(xi, pct, width=0.58, color='#3E2E5E', alpha=0.78,
                   edgecolor="black", lw=S(0.8), zorder=3)

    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0, top=cap_h)
    ax.set_xticks(x);  ax.set_xticklabels([str(b) for b in bits_x], fontsize=S(11))
    ax.set_xlim(-0.6, len(bits_x) - 0.4)
    ax.set_xlabel(f"{spec['label']} bit index", fontweight='bold', labelpad=3)
    ax.set_ylabel(metric_axis_label(metric), fontweight='bold', labelpad=3)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.65, zorder=2)
    ax.axhline(0.0, color="black", linewidth=S(0.8), linestyle="--", zorder=7)

    # Put field names above the plotting area so labels stay readable on
    # log-scale bars and do not collide with data.
    trans = ax.get_xaxis_transform()
    for left, right, label, color, _ in field_spans:
        cx = (left + right) / 2
        ax.text(cx, 1.055, label, transform=trans, ha="center", va="bottom",
                fontsize=S(11), fontweight="bold", color=color, clip_on=False)
        ax.plot([left + 0.06, right - 0.06], [1.035, 1.035], transform=trans,
                color=color, linewidth=S(1.6), solid_capstyle="round", clip_on=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

FIG_SCALE = 2.0

def S(x):
    return x * FIG_SCALE

def scaled_figsize(w, h):
    return (w * FIG_SCALE, h * FIG_SCALE)
plt.rcParams.update({
    'font.family':       _FONT_NAME,
    'font.weight':       'bold',
    'font.size':         S(11),
    'axes.labelsize':    S(11),
    'axes.titlesize':    S(11),
    'xtick.labelsize':   S(11),
    'ytick.labelsize':   S(11),
    'axes.linewidth':    S(0.7),
    'xtick.major.width': S(0.5),
    'ytick.major.width': S(0.5),
    'xtick.major.size':  S(3),
    'ytick.major.size':  S(3),
    'xtick.major.pad':   S(2),
    'ytick.major.pad':   S(2),
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def main(model=DEFAULT_MODEL, metric=DEFAULT_METRIC,
         csv_path=None, trefi_list=None, fmt="fp16", dram="lpddr5", baseline=None):
    metric    = choose_metric(model, metric)
    fmt       = canonical_format(fmt)
    csv_path  = csv_path or default_csv(model, metric, fmt=fmt, dram=dram)
    trefi_list = trefi_list or DEFAULT_TREFI

    # # ── 2. Bitwise standalone ────────────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=scaled_figsize(5.5, 2.5))
    fontsize=S(9)
    linewidth=S(0.8)
    labelpad=S(3)
    draw_bitwise_2(
        ax_b,
        model=model,
        metric=metric,
        csv_path=csv_path,
        trefi_list=trefi_list,
        fmt=fmt,
        baseline=baseline,
    )
    fig_b.tight_layout()

    out_pdf = OUT / f"fig_bitwise_2_{model}_{fmt}.pdf"
    fig_b.savefig(out_pdf, bbox_inches="tight")
    if model == DEFAULT_MODEL and fmt == "fp16":
        fig_b.savefig(OUT / "fig_bitwise_2.pdf", bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig_b)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--metric", default="auto")
    p.add_argument("--format", "--dtype", dest="fmt", default="fp16")
    p.add_argument("--dram",   default="lpddr5")
    p.add_argument("--csv",    type=Path, default=None)
    p.add_argument("--baseline", type=float, default=None)
    p.add_argument("--trefi",  default=None)
    args = p.parse_args()
    trefi_list = [int(t) for t in args.trefi.replace(",", " ").split()] if args.trefi else None
    main(model=args.model,
         metric=choose_metric(args.model, args.metric),
         csv_path=args.csv or None,
         trefi_list=trefi_list,
         fmt=args.fmt,
         dram=args.dram,
         baseline=args.baseline)