# -*- coding: utf-8 -*-
"""
Grouped bar chart: DRAM throughput improvement (normalized speedup) over baseline.
Two suBFigures: large-scale config (left) and edge-device config (right).
Each suBFigure shows 3 DNN workloads x 3 precisions (FP16, BF16, FP8).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)


# ── Data  (baseline_cycles / config_cycles) ── placeholder ───────────────────
# Large-scale config
large_dnns   = ['Llama 3.2-1B', 'OPT-2.7B', 'Resnet50']
large_FP16   = np.array([1.1409, 1.1412, 1.1372])
large_FP8    = np.array([1.0772, 1.0774, 1.1371])
large_BF16   = np.array([1.0772, 1.0774, 1.0831])

# Edge-device config
edge_dnns    = ['Llama 3.2-1B', 'OPT-2.7B', 'Resnet50']
edge_FP16    = np.array([1.1391, 1.1385, 1.1387])
edge_FP8     = np.array([1.0756, 1.0750, 1.1384])
edge_BF16    = np.array([1.0756, 1.0750, 1.0837])



# ── Colours & style ──────────────────────────────────────────────────────────
# COLOR_FP16  = '#D94A64'
# COLOR_BF16  = '#8C1F6F'
# COLOR_FP8   = '#2D1040'
COLOR_FP16  = '#92658E'
COLOR_BF16  = '#BE6C91'
COLOR_FP8   = '#E4AFCF'

COLOR_FP16 = '#3E2E5E' #(진한 다크 퍼플)
COLOR_BF16 = '#71618B' #(미디엄 모브)
COLOR_FP8 = '#BE6C91' #92658E' #(상단 (Layer 3))  


FIGURE_DIR  = Path(__file__).resolve().parent
_FONT_PATH = '/home/wing02/arialnarrow_bold.ttf'
if Path(_FONT_PATH).exists():
    fm.fontManager.addfont(_FONT_PATH)
    _FONT_NAME = fm.FontProperties(fname=_FONT_PATH).get_name()

plt.rcParams.update({
    'font.family'       : _FONT_NAME, 
    'font.weight'       : 'bold',
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
def draw_suBFigure(ax, dnns, FP16, BF16, FP8):
    n      = len(dnns)
    x      = np.arange(n)
    width  = 0.22
    offsets = [-width, 0, width]
    precisions = [('FP16', FP16, COLOR_FP16),
                  ('BF16', BF16, COLOR_BF16),
                  ('FP8',  FP8,  COLOR_FP8 )]

    for offset, (label, values, color) in zip(offsets, precisions):
        for i, v in enumerate(values):
            ax.bar(x[i] + offset, v, width,
                   color=color, edgecolor='black', linewidth=0.8, zorder=3)
        ax.bar([], [], width, color=color, label=label, zorder=3)

    # Value labels above each bar
    for offset, (_, values, _) in zip(offsets, precisions):
        for i, v in enumerate(values):
            ax.text(x[i] + offset, v + 0.003,
                    f'{v:.3f}x',
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
    ymax = max(FP16.max(), BF16.max(), FP8.max()) * 1.10
    ax.set_ylim(ymin, ymax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)


def legend_patches():
    return [
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP16, label='FP16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_BF16, label='BF16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP8,  label='FP8'),
    ]


# ── Large-scale figure ────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(3.6, 2.6))
draw_suBFigure(ax1, large_dnns, large_FP16, large_BF16, large_FP8)

fig1.legend(handles=legend_patches(), loc='lower center',
            ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig1.tight_layout(pad=0.5, rect=[0, 0.08, 1, 1])
fig1.savefig('outputs/throughput_large.pdf', bbox_inches='tight')
plt.close(fig1)

# ── Edge-device figure ────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(3.6, 2.6))
draw_suBFigure(ax2, edge_dnns, edge_FP16, edge_BF16, edge_FP8)
fig2.legend(handles=legend_patches(), loc='lower center',
            ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig2.tight_layout(pad=0.5, rect=[0, 0.08, 1, 1])
fig2.savefig('outputs/throughput_edge.pdf', bbox_inches='tight')
plt.close(fig2)

# ── Combined figure (large + edge side by side) ───────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8,2), sharey=True)

draw_suBFigure(ax_l, large_dnns, large_FP16, large_BF16, large_FP8)
draw_suBFigure(ax_r, edge_dnns,  edge_FP16,  edge_BF16,  edge_FP8)

ax_l.set_title('Datacenter-scale', fontweight='bold')
ax_r.set_title('Edge-device', fontweight='bold')

# sharey=True 쓰면 오른쪽 y축 label 중복되므로 제거
ax_r.set_ylabel('')

fig.legend(handles=legend_patches(), loc='lower center',
           ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09))  # -0.06 → -0.02
fig.tight_layout(pad=0.5, rect=[0, 0.04, 1, 1])                 # 0.08 → 0.05
fig.savefig('outputs/throughput_combined.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved outputs/throughput_combined.pdf')

print('Saved outputs/throughput_large.pdf and outputs/throughput_edge.pdf and Saved outputs/throughput_combined.pdf')
