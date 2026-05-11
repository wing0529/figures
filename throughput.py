# -*- coding: utf-8 -*-
"""
Grouped bar chart: DRAM throughput improvement (normalized speedup) over baseline.
Two subfigures: large-scale config (left) and edge-device config (right).
Each subfigure shows 3 DNN workloads x 3 precisions (fp16, bf16, fp8).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)


# ── Data  (baseline_cycles / config_cycles) ───────────────────────────────────
# Large-scale config
large_dnns   = ['LLaMA', 'OPT', 'ResNet']
large_fp16   = np.array([1.1381, 1.1412, 1.1372])
large_bf16   = np.array([1.1109, 1.1412, 1.1372])
large_fp8    = np.array([1.0813, 1.1412, 1.1372])

# Edge-device config  (opt_small not measured → dummy 1.0)
edge_dnns    = ['LLaMA', 'OPT', 'ResNet']
edge_fp16    = np.array([1.1358, 1.0,    1.1387])
edge_bf16    = np.array([1.1088, 1.0,    1.1387])
edge_fp8     = np.array([1.0791, 1.0,    1.1387])


# ── Colours & style ──────────────────────────────────────────────────────────
COLOR_FP16  = '#D94A64'
COLOR_BF16  = '#8C1F6F'
COLOR_FP8   = '#2D1040'
HATCH_DUMMY = '////'      # visual indicator for dummy (placeholder) data

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans Mono',
    'font.size'         : 8,
    'axes.titlesize'    : 9,
    'axes.labelsize'    : 8,
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'legend.fontsize'   : 7.5,
    'figure.titlesize'  : 9,
    'axes.linewidth'    : 0.6,
    'xtick.major.width' : 0.5,
    'ytick.major.width' : 0.5,
    'xtick.major.size'  : 0,
    'ytick.major.size'  : 0,
    'xtick.major.pad'   : 2,
    'ytick.major.pad'   : 2,
})


# ── Helper ────────────────────────────────────────────────────────────────────
def draw_subfigure(ax, dnns, fp16, bf16, fp8, dummy_mask=None):
    """
    Draw one grouped-bar subfigure.
    dummy_mask: boolean array of length len(dnns); True means placeholder data.
    """
    if dummy_mask is None:
        dummy_mask = np.zeros(len(dnns), dtype=bool)

    n      = len(dnns)
    x      = np.arange(n)
    width  = 0.22
    offsets = [-width, 0, width]
    precisions = [('fp16', fp16, COLOR_FP16),
                  ('bf16', bf16, COLOR_BF16),
                  ('fp8',  fp8,  COLOR_FP8 )]

    bar_handles = []
    for offset, (label, values, color) in zip(offsets, precisions):
        for i, (v, is_dummy) in enumerate(zip(values, dummy_mask)):
            hatch = HATCH_DUMMY if is_dummy else None
            alpha = 0.45 if is_dummy else 1.0
            b = ax.bar(x[i] + offset, v, width,
                       color=color, alpha=alpha, hatch=hatch,
                       edgecolor='white', linewidth=0.4, zorder=3)
        # Invisible bar for legend (use first bar returned)
        h = ax.bar([], [], width, color=color, label=label, zorder=3)
        bar_handles.append(h)

    # Value labels above each bar
    for offset, (_, values, _) in zip(offsets, precisions):
        for i, (v, is_dummy) in enumerate(zip(values, dummy_mask)):
            if is_dummy:
                continue
            ax.text(x[i] + offset, v + 0.003,
                    f'{v:.3f}x',
                    ha='center', va='bottom',
                    fontsize=5.5, rotation=90, zorder=4)

    # Baseline reference line
    ax.axhline(1.0, color='#444444', linewidth=0.8, linestyle='--', zorder=2)
    ax.text(n - 0.45, 1.002, 'baseline', fontsize=6, color='#444444', va='bottom')

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(dnns)
    ax.set_ylabel('Norm. Speedup', fontweight='bold')
    ax.set_xlim(-0.55, n - 0.45)

    ymin = 0.95
    ymax = max(fp16.max(), bf16.max(), fp8.max()) * 1.10
    ax.set_ylim(ymin, ymax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)

    return bar_handles


def legend_patches():
    return [
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP16, label='fp16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_BF16, label='bf16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP8,  label='fp8'),
        plt.Rectangle((0,0), 1, 1, color='#aaaaaa', alpha=0.45,
                      hatch=HATCH_DUMMY, label='placeholder (no data)'),
    ]


# ── Large-scale figure ────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(3.6, 2.6))
draw_subfigure(ax1, large_dnns, large_fp16, large_bf16, large_fp8)
fig1.legend(handles=legend_patches(), loc='lower center',
            ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig1.tight_layout(pad=0.5, rect=[0, 0.08, 1, 1])
fig1.savefig('outputs/throughput_large.pdf', bbox_inches='tight')
plt.close(fig1)

# ── Edge-device figure ────────────────────────────────────────────────────────
edge_dummy = np.array([False, True, False])
fig2, ax2 = plt.subplots(figsize=(3.6, 2.6))
draw_subfigure(ax2, edge_dnns, edge_fp16, edge_bf16, edge_fp8, dummy_mask=edge_dummy)
fig2.legend(handles=legend_patches(), loc='lower center',
            ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig2.tight_layout(pad=0.5, rect=[0, 0.08, 1, 1])
fig2.savefig('outputs/throughput_edge.pdf', bbox_inches='tight')
plt.close(fig2)

print('Saved outputs/throughput_large.pdf and outputs/throughput_edge.pdf')
