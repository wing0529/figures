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


# ── Data  (baseline_cycles / config_cycles) ── placeholder ───────────────────
# Large-scale config
large_dnns   = ['LLaMA', 'OPT', 'ResNet']
large_fp16   = np.array([1.1409, 1.1412, 1.1372])
large_bf16   = np.array([1.0, 1.077, 1.0])
large_fp8    = np.array([1.0772, 1.0774, 1.0])

# Edge-device config
edge_dnns    = ['LLaMA', 'OPT', 'ResNet']
edge_fp16    = np.array([1.1391, 1.1385, 1.1387])
edge_bf16    = np.array([1.0, 1.0750, 1.0])
edge_fp8     = np.array([1.0756, 1.0750, 1.0])


# ── Colours & style ──────────────────────────────────────────────────────────
COLOR_FP16  = '#D94A64'
COLOR_BF16  = '#8C1F6F'
COLOR_FP8   = '#2D1040'


plt.rcParams.update({
    'font.family'       : 'DejaVu Sans Mono',
    'font.size'         : 11,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 11,
    'xtick.labelsize'   : 11,
    'ytick.labelsize'   : 11,
    'legend.fontsize'   : 11,
    'figure.titlesize'  : 11,
    'axes.linewidth'    : 0.6,
    'xtick.major.width' : 0.5,
    'ytick.major.width' : 0.5,
    'xtick.major.size'  : 0,
    'ytick.major.size'  : 0,
    'xtick.major.pad'   : 2,
    'ytick.major.pad'   : 2,
})


# ── Helper ────────────────────────────────────────────────────────────────────
def draw_subfigure(ax, dnns, fp16, bf16, fp8):
    n      = len(dnns)
    x      = np.arange(n)
    width  = 0.22
    offsets = [-width, 0, width]
    precisions = [('fp16', fp16, COLOR_FP16),
                  ('bf16', bf16, COLOR_BF16),
                  ('fp8',  fp8,  COLOR_FP8 )]

    for offset, (label, values, color) in zip(offsets, precisions):
        for i, v in enumerate(values):
            ax.bar(x[i] + offset, v, width,
                   color=color, edgecolor='white', linewidth=0.4, zorder=3)
        ax.bar([], [], width, color=color, label=label, zorder=3)

    # Value labels above each bar
    for offset, (_, values, _) in zip(offsets, precisions):
        for i, v in enumerate(values):
            ax.text(x[i] + offset, v + 0.003,
                    f'{v:.2f}x',
                    ha='center', va='bottom',
                    fontsize=11, rotation=90, zorder=4)

    # Baseline reference line
    ax.axhline(1.0, color='#444444', linewidth=0.8, linestyle='--', zorder=2)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(dnns)
    ax.set_ylabel('Normalized throughput', fontweight='bold')
    ax.set_xlim(-0.55, n - 0.45)

    ymin = 0.95
    ymax = max(fp16.max(), bf16.max(), fp8.max()) * 1.10
    ax.set_ylim(ymin, ymax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)


def legend_patches():
    return [
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP16, label='fp16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_BF16, label='bf16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP8,  label='fp8'),
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
fig2, ax2 = plt.subplots(figsize=(3.6, 2.6))
draw_subfigure(ax2, edge_dnns, edge_fp16, edge_bf16, edge_fp8)
fig2.legend(handles=legend_patches(), loc='lower center',
            ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig2.tight_layout(pad=0.5, rect=[0, 0.08, 1, 1])
fig2.savefig('outputs/throughput_edge.pdf', bbox_inches='tight')
plt.close(fig2)

print('Saved outputs/throughput_large.pdf and outputs/throughput_edge.pdf')
