# -*- coding: utf-8 -*-
"""
Grouped bar chart: accelerator performance breakdown (Large & Edge).
Style mirrors energy.py: color = precision, bar style = cycle component.
Each DNN group has 3 precision clusters (fp16, bf16, fp8); within each
cluster 3 adjacent bars show data movement (solid), stall (///), and
systolic execution (xxx), all normalised to baseline total cycles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)

# ── Raw data: [total, compute, stall] summed across all layers ────────────────
# Layout: array shape (n_dnns, 3) — dnn order: LLaMA, OPT, ResNet

# Large-scale config (LLaMA real; OPT & ResNet dummy = base)
large_base = np.array([
    [34198+17241+619500+182985+182985,
     10604+11321+581570+131875+131875,
     381+1098+254403+50084+50084],
    [1000000, 800000, 200000],
    [500000,  420000,  80000],
], dtype=float)

large_fp16 = np.array([
    [29706+12768+567434+148019+148019,
     10223+11302+529522+110100+110100,
     0+1079+202355+28309+28309],
    [1000000, 800000, 200000],
    [500000,  420000,  80000],
], dtype=float)

large_bf16 = large_base.copy()   # dummy — not yet available

large_fp8 = np.array([
    [33867+16136+605463+176358+176358,
     11383+11302+567564+126801+126801,
     1160+1079+240397+45010+45010],
    [1000000, 800000, 200000],
    [500000,  420000,  80000],
], dtype=float)

# Edge-device config — all dummy = base for now
edge_base = np.array([
    [1036909, 867245, 356050],
    [1000000, 800000, 200000],
    [500000,  420000,  80000],
], dtype=float)
edge_fp16 = edge_base.copy()
edge_bf16 = edge_base.copy()
edge_fp8  = edge_base.copy()


# ── Decompose helper ──────────────────────────────────────────────────────────
def decompose(data, base):
    """Return (data_mov, stall, systolic) each normalised by base total."""
    bt      = base[:, 0]
    total, compute, stall = data[:, 0], data[:, 1], data[:, 2]
    return (total - compute) / bt, stall / bt, (compute - stall) / bt


# ── Colors & style ───────────────────────────────────────────────────────────
DNNS       = ['Llama', 'OPT', 'Resnet']
PRECISIONS = ['FP16', 'BF16', 'FP8']

# Colors distinguish cycle components within a bar
COLOR_DM  = '#D94A64'   # data movement
COLOR_ST  = '#8C1F6F'   # stall
COLOR_SY  = '#2D1040'   # systolic execution

plt.rcParams.update({
    'font.family':       ['Liberation Sans Narrow', 'Arial Narrow', 'DejaVu Sans'],
    'font.weight':       'bold',
    'font.size':         11,
    'axes.titlesize':    11,
    'axes.labelsize':    11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  0,
    'ytick.major.size':  0,
    'xtick.major.pad':   2,
    'ytick.major.pad':   2,
})

# ── Layout parameters ─────────────────────────────────────────────────────────
BW       = 0.22   # bar width
PREC_GAP = 0.04   # gap between precision bars within one DNN group

fp16_off = -BW - PREC_GAP
bf16_off =  0.0
fp8_off  =  BW + PREC_GAP
PREC_OFFS = {'FP16': fp16_off, 'BF16': bf16_off, 'FP8': fp8_off}


# ── Helper ────────────────────────────────────────────────────────────────────
def draw_subfigure(ax, base, fp16_raw, bf16_raw, fp8_raw):
    x    = np.arange(len(DNNS), dtype=float)
    raws = {'FP16': fp16_raw, 'BF16': bf16_raw, 'FP8': fp8_raw}

    for prec in PRECISIONS:
        off = PREC_OFFS[prec]
        dm, st, sy = decompose(raws[prec], base)
        xpos = x + off

        # Stack: systolic (bottom) → stall → data movement (top)
        ax.bar(xpos, sy, BW,
               color=COLOR_SY, edgecolor='white', linewidth=0.5, zorder=3)
        ax.bar(xpos, st, BW, bottom=sy,
               color=COLOR_ST, edgecolor='white', linewidth=0.5, zorder=3)
        ax.bar(xpos, dm, BW, bottom=sy + st,
               color=COLOR_DM, edgecolor='white', linewidth=0.5, zorder=3)

        totals = dm + st + sy
        for xi, t in zip(xpos, totals):
            # precision label just below x-axis tick area, rotated
            ax.text(xi, -0.04, prec,
                    ha='center', va='top', fontsize=8, rotation=45,
                    transform=ax.get_xaxis_transform(), zorder=4)
            ax.text(xi, t + 0.005, f'{t:.2f}',
                    ha='center', va='bottom', fontsize=8, rotation=90, zorder=4)

    ax.axhline(1.0, color='#555555', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(DNNS)
    ax.tick_params(axis='x', pad=28)  # make room for precision labels below
    ax.set_xlim(-0.5, len(DNNS) - 0.5)
    ax.set_ylabel('Normalized cycles', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)


def make_legend():
    return [
        plt.Rectangle((0,0), 1, 1, fc=COLOR_SY, ec='none', label='systolic compute'),
        plt.Rectangle((0,0), 1, 1, fc=COLOR_ST, ec='none', label='stall'),
        plt.Rectangle((0,0), 1, 1, fc=COLOR_DM, ec='none', label='prefetch + drain'),
    ]


def save_fig(base, fp16_raw, bf16_raw, fp8_raw, out_path):
    all_vals = np.concatenate([
        np.concatenate(decompose(r, base)) for r in [fp16_raw, bf16_raw, fp8_raw]
    ])
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    draw_subfigure(ax, base, fp16_raw, bf16_raw, fp8_raw)
    ymax = all_vals.max() * 1.35
    ax.set_ylim(0, ymax)
    # Build ticks that always include 0, 1.0, and ymax
    import numpy as _np
    step = 0.2 if ymax <= 1.6 else 0.5
    ticks = sorted(set([round(v, 10) for v in _np.arange(0, ymax, step)] + [1.0]))
    ax.set_yticks(ticks)
    fig.legend(handles=make_legend(), loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(pad=0.5, rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# ── Generate both figures ─────────────────────────────────────────────────────
save_fig(large_base, large_fp16, large_bf16, large_fp8, 'outputs/accel_perf_large.pdf')
save_fig(edge_base,  edge_fp16,  edge_bf16,  edge_fp8,  'outputs/accel_perf_edge.pdf')
