#!/usr/bin/env python3
"""
Produces:
  - outputs/fig_energy.pdf/png          : refresh energy stacked bar (standalone)
  - outputs/fig_bitwise.pdf/png         : bitwise worst-case bar     (standalone)
  - outputs/fig_combined.pdf/png        : both side-by-side
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

# ── Shared helpers ────────────────────────────────────────────────────────────

def _coerce_float(v: Any) -> float | None:
    if v in (None, "", "null", "None", "N/A"): return None
    if isinstance(v, (int, float)):            return float(v)
    t = str(v).strip().lower()
    if t in {"inf", "+inf"}:  return math.inf
    if t in {"-inf"}:         return -math.inf
    if t == "nan":            return math.nan
    return float(v)

def choose_metric(model, metric):
    return METRIC_DEFAULTS.get(model, "ppl") if metric == "auto" else metric

def default_csv(model, metric, dram="lpddr5"):
    root = CSV_ROOTS["ppl" if metric == "ppl" else "acc"]
    for pat in (f"{model}__fp16__bitwise16__*__{dram}__*.csv",
                f"{model}__fp16__bitwise16__*.csv",
                f"*{model}*bitwise*.csv"):
        cs = sorted(root.glob(pat))
        if cs: return cs[0]
    raise FileNotFoundError(f"No CSV for model={model} under {root}")

def resolve_baseline(model, metric, rows):
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

# ── Panel A: energy stacked bar ───────────────────────────────────────────────

# def draw_energy(ax):
#     densities   = ['8 Gb', '16 Gb', '32 Gb']
#     refresh     = [14, 22, 34]
#     non_refresh = [86, 78, 66]
    
#     #fcols  = {b: ("#C00000" if b == 15 else "#ED7D31" if b >= 10 else "#1565c0") for b in range(16)}

#     # C_REF  = '#3d3d3d'
#     # C_NREF = '#b0b0b0'
#     C_REF  =  "#C00000" # 진한 빨강  (Refresh, 위)
#     C_NREF =  '#ED7D31'  # 진한 주황  (Non-Refresh, 아래)
#     #C_REF  = '#5B3A8A'   # 진한 보라  (Refresh, 아래)
#     #C_NREF = '#D4A8C7
#     x      = range(len(densities))
#     bw     = 0.52

#     br = ax.bar(x, refresh,     width=bw, color=C_REF,  edgecolor='white', lw=0.6, label='Refresh',zorder=3)
#     bn = ax.bar(x, non_refresh, width=bw, color=C_NREF, edgecolor='white', lw=0.6, label='Non-Refresh',
#                 bottom=refresh, alpha=0.55,  zorder=3)


#     for i, (nr, r) in enumerate(zip(non_refresh, refresh)):
#         ax.text(i, r / 2,      f'{r}%',  ha='center', va='center', fontsize=11, fontweight='bold', color='white',  zorder=5)
#         ax.text(i, r + nr / 2, f'{nr}%', ha='center', va='center', fontsize=11, fontweight='bold', color='black',  zorder=5)

#     ax.set_xticks(list(x));  ax.set_xticklabels(densities, fontweight='bold')
#     ax.set_yticks(range(0, 101, 20));  ax.set_ylim(0, 100)
#     ax.set_ylabel('Proportion (%) of\nActive Energy', fontweight='bold')
#     ax.set_xlabel('Density', fontweight='bold')
#     ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
#     ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
#     ax.legend(handles=[bn, br], labels=['Non-Refresh', 'Refresh'],
#               loc='upper right', fontsize=9, frameon=True, framealpha=0.9,
#               edgecolor='#cccccc', ncol=2, bbox_to_anchor=(1.02, 1.10))
#     return ax

def draw_energy(ax):
    densities   = ['8 Gb', '16 Gb', '32 Gb']
    refresh     = [14, 22, 34]
    non_refresh = [86, 78, 66]
    
    # ==========================================================================
    # [선택] 원하시는 옵션 하나만 남기고 주석을 해제하여 사용하세요.
    
    # Option A: 세련된 딥 모브 & 소프트 실버 (추천)
    C_REF  = '#544D7B' # Refresh (강조군 - 딥 퍼플)
    C_NREF = '#B0B7BD'  # Non-Refresh (대조군 - 실버 그레이)
    TEXT_COLOR_NREF = 'black'
    TEXT_COLOR_REF  = 'white'

    
    # #Option B: 화사한 로즈 핑크 & 클래식 네이비
    # C_REF  = '#BC6991'  # Refresh (강조군 - 로즈 핑크)
    # C_NREF = '#544D7B'  # Non-Refresh (대조군 - 네이비 퍼플)
    # TEXT_COLOR_NREF = 'white'
    # TEXT_COLOR_REF  = 'white'


    # C_REF  = '#3E2E5E' #(딥 퍼플)
    # C_NREF = '#DEAAC8' #(라이트 파스텔 핑크)
    # TEXT_COLOR_NREF = 'black'
    # TEXT_COLOR_REF  = 'white'
    
    # C_REF  = '#2a2a2a'  # Refresh (강조군 - 다크 그레이)
    # C_NREF = '#909090'  # Non-Refresh (대조군 - 실버 그레이)
    # TEXT_COLOR_NREF = 'black'
    # TEXT_COLOR_REF  = 'white'


    # ==========================================================================

    x   = range(len(densities))
    bw  = 0.4

    # 바 차트 그리기 (zorder를 주어 그리드 뒤로 가도록 설정)
    br = ax.bar(x, refresh, width=bw, color=C_REF, edgecolor='black', linewidth=0.8, label='Refresh', zorder=3)
    bn = ax.bar(x, non_refresh, width=bw, color=C_NREF, edgecolor='black', linewidth=0.8, label='Non-Refresh',
                bottom=refresh, zorder=3)

    # 텍스트 레이블 추가
    for i, (nr, r) in enumerate(zip(non_refresh, refresh)):
        # Refresh 텍스트 (하단 블록의 중앙)
        ax.text(i, r / 2, f'{r}%', ha='center', va='center', 
                fontsize=9, fontweight='bold', color=TEXT_COLOR_REF, zorder=5)
        # Non-Refresh 텍스트 (상단 블록의 중앙)
        ax.text(i, r + nr / 2, f'{nr}%', ha='center', va='center', 
                fontsize=9, fontweight='bold', color=TEXT_COLOR_NREF, zorder=5)

    # 축 및 스타일 레이아웃 설정
    ax.set_xticks(list(x))
    ax.set_xticklabels(densities, fontweight='bold',fontsize=9)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f'{y}%' for y in range(0, 101, 20)], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Proportion (%) of\nActive Energy', fontweight='bold')
    ax.set_xlabel('Density', fontweight='bold')

    
    ax.set_xticks(list(x))
    ax.set_xticklabels(densities, fontweight='bold',fontsize=9)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f'{y}%' for y in range(0, 101, 20)], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Proportion (%) of\nActive Energy', fontweight='bold')
    ax.set_xlabel('Density', fontweight='bold')
    
    # 테두리 정리
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
    
    # 범례 설정 (깔끔하게 Non-Refresh가 먼저 오도록 순서 매칭)
    # ax.legend(handles=[bn, br], labels=['Non-Refresh', 'Refresh'],
    #           fontsize=9, frameon=True, framealpha=0.9,
    #           edgecolor='#cccccc', ncol=2, bbox_to_anchor=(1.02, 1.10))
    
    # ax.legend(handles=[bn, br], labels=['Non-Refresh', 'Refresh'],
    #         fontsize=9, frameon=True, framealpha=0.9,
    #         edgecolor='#cccccc', 
    #         ncol=1,  
    #         loc='center left',             
    #         # x축을 1.02에서 1.05로 늘려 오른쪽 여백을 주고,
    #         # y축을 0.5에서 0.45~0.48로 살짝 내려 밸런스를 맞춥니다.
    #         bbox_to_anchor=(1.05, 0.47))

    ax.legend(handles=[bn, br], labels=['Non-Refresh', 'Refresh'],
          fontsize=9, 
          frameon=False,               # ★ True에서 False로 변경 (테두리 및 배경 제거)
          ncol=2,                      
          loc='lower center',          
          bbox_to_anchor=(0.5, 1.02))
    return ax


# ── Panel B: bitwise worst-case bars ─────────────────────────────────────────

def draw_bitwise(ax, model=DEFAULT_MODEL, metric=DEFAULT_METRIC,
                 csv_path=None, trefi_list=None):
    if trefi_list is None: trefi_list = DEFAULT_TREFI
    if csv_path   is None: csv_path   = default_csv(model, metric)

    rows      = as_bar_rows(load_metric_rows(csv_path, model, metric))
    baseline  = resolve_baseline(model, metric, rows)
    raw, bers = build_maps(rows)
    trefi_sub = [t for t in trefi_list if t in raw] or sorted(raw)

    bits_x = list(range(15, -1, -1))
    x      = np.arange(len(bits_x))
    fcols  = {b: ("#C00000" if b == 15 else "#ED7D31" if b >= 10 else "#1565c0") for b in range(16)}

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
                   edgecolor="black", lw=1, zorder=3)
            ax.text(xi, cap_h * 0.01, "NaN", ha="center", va="bottom",
                    color="white", fontsize=11, fontweight="bold", rotation=90, zorder=6)
        elif math.isinf(pct) or pct >= 1e6:
            ax.bar(xi, cap_h, width=0.58, color='#3E2E5E', alpha=0.78,
                   edgecolor="black", lw=1, zorder=3)
            ax.text(xi, cap_h * 0.01, ">100x", ha="center", va="bottom",
                    color="white", fontsize=11, fontweight="bold", rotation=90, zorder=6)
        else:
            ax.bar(xi, pct, width=0.58, color='#3E2E5E', alpha=0.78,
                   edgecolor="black", lw=1, zorder=3)
            lbl = fmt_delta_pct(pct)
            # if pct >= 10.0:
            #     ax.text(xi, pct * 0.005, lbl, ha="center", va="bottom",
            #             color="white", fontsize=11, fontweight="bold", rotation=90, zorder=6)
            #elif 0.05 <= pct < 10.0:
                # ax.text(xi, pct * 1.3, lbl, ha="center", va="bottom",
                #         color="black", fontsize=9, fontweight="bold", rotation=90, zorder=6)

    # field background
    ax.axvspan(-0.5,  0.5, alpha=0.08, color='#b0b0b0' , zorder=0)
    ax.axvspan( 0.5,  5.5, alpha=0.08, color='#b0b0b0' , zorder=0)
    ax.axvspan( 5.5, 15.5, alpha=0.08, color='#b0b0b0' , zorder=0)

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
    ax.set_xticks(x);  ax.set_xticklabels([str(b) for b in bits_x], fontsize=10)
    ax.set_xlim(-0.6, len(bits_x) - 0.4)
    ax.set_xlabel("FP16 Bit Index", fontweight='bold')
    ax.set_ylabel("PPL Increase (%)", fontweight='bold')
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.75, zorder=4)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=7)
    return ax

# ── Main: save individual + combined ─────────────────────────────────────────

def main(model=DEFAULT_MODEL, metric=DEFAULT_METRIC,
         csv_path=None, trefi_list=None):
    metric    = choose_metric(model, metric)
    csv_path  = csv_path or default_csv(model, metric)
    trefi_list = trefi_list or DEFAULT_TREFI

    # ── 1. Energy standalone ──────────────────────────────────────────────────
    fig_e, ax_e = plt.subplots(figsize=(4, 2))
    draw_energy(ax_e)
    fig_e.tight_layout()
    save_fig(fig_e, OUT / "fig_energy.png")
    plt.close(fig_e)




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--metric", default="auto")
    p.add_argument("--dram",   default="lpddr5")
    p.add_argument("--csv",    type=Path, default=None)
    p.add_argument("--trefi",  default=None)
    args = p.parse_args()
    trefi_list = [int(t) for t in args.trefi.replace(",", " ").split()] if args.trefi else None
    main(model=args.model,
         metric=choose_metric(args.model, args.metric),
         csv_path=args.csv or None,
         trefi_list=trefi_list)