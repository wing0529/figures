# -*- coding: utf-8 -*-
"""
Grouped bar chart: Energy saving vs. baseline (%).
Two separate output files: energy_saving_small.pdf and energy_saving_large.pdf.
For each workload x precision pair the total energy bar and the
refresh energy bar are placed side by side.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)


# -- Data ----------------------------------------------------------------------
DNNS       = ['Llama', 'OPT', 'Resnet']
PRECISIONS = ['FP16', 'BF16', 'FP8']

small_total   = {'FP16': np.array([16.97, 15.94, 18.44]),
                 'BF16': np.array([ 0.00 , 8.44,  0.00]),
                 'FP8':  np.array([ 9.47,  8.44, 18.44])}

large_total   = {'FP16': np.array([16.05, 16.16, 16.35]),
                 'BF16': np.array([-0.37,  9.88, 16.35]),
                 'FP8':  np.array([ 9.84,  9.88,  0.00])}

small_refresh = {'FP16': np.array([66.66, 66.64, 66.82]),
                 'BF16': np.array([ 0.00, 41.59,  0.00]),
                 'FP8':  np.array([41.63, 41.59,  0.00])}

large_refresh = {'FP16': np.array([66.97, 66.98, 66.53]),
                 'BF16': np.array([ 0.26, 41.84, 66.53]),
                 'FP8':  np.array([41.89, 41.84, 66.53])}


# -- Colors & style ------------------------------------------------------------
COLORS = {'FP16': '#D94A64', 'BF16': '#8C1F6F', 'FP8': '#2D1040'}

plt.rcParams.update({
    'font.family':       ['Liberation Sans Narrow', 'Arial Narrow', 'DejaVu Sans'],
    'font.weight':       'bold',
    'font.size':         11,
    'axes.titlesize':    11,
    'axes.labelsize':    11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  0,
    'ytick.major.size':  0,
    'xtick.major.pad':   2,
    'ytick.major.pad':   2,
})


# -- Layout parameters ---------------------------------------------------------
BW        = 0.09   # bar width
INNER_GAP = 0.01   # gap between total and refresh within one precision pair
PREC_GAP  = 0.04   # gap between different precision pairs

PAIR_W    = 2 * BW + INNER_GAP           # 0.19
CLUSTER_W = 3 * PAIR_W + 2 * PREC_GAP   # 0.65

FP16_ctr  = -CLUSTER_W / 2 + PAIR_W / 2
BF16_ctr  = FP16_ctr + PAIR_W + PREC_GAP
FP8_ctr   = BF16_ctr + PAIR_W + PREC_GAP
PAIR_CTRS = {'FP16': FP16_ctr, 'BF16': BF16_ctr, 'FP8': FP8_ctr}

TOT_OFF = -(BW / 2 + INNER_GAP / 2)
REF_OFF =  (BW / 2 + INNER_GAP / 2)


# -- Helper --------------------------------------------------------------------
def draw_subfigure(ax, total, refresh):
    x = np.arange(len(DNNS), dtype=float)

    for prec in PRECISIONS:
        pc = PAIR_CTRS[prec]
        for i, (tv, rv) in enumerate(zip(total[prec], refresh[prec])):
            tot_x = x[i] + pc + TOT_OFF
            ref_x = x[i] + pc + REF_OFF

            # Total -- solid
            ax.bar(tot_x, tv, BW, color=COLORS[prec], edgecolor='none', zorder=3)
            lbl_y = tv + 0.8 if tv >= 0 else tv - 0.8
            va    = 'bottom' if tv >= 0 else 'top'
            ax.text(tot_x, lbl_y, f'{tv:.1f}',
                    ha='center', va=va, fontsize=6, rotation=90, zorder=4)

            # Refresh -- hatched
            ax.bar(ref_x, rv, BW,
                   facecolor='white', hatch='///', edgecolor=COLORS[prec],
                   linewidth=0.8, zorder=3)
            ax.text(ref_x, rv + 0.8, f'{rv:.1f}',
                    ha='center', va='bottom', fontsize=6, rotation=90, zorder=4)

    ax.axhline(0, color='#555555', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(DNNS)
    ax.set_xlim(-0.5, len(DNNS) - 0.5)
    ax.set_ylabel('Saved energy (%)', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)


def make_legend():
    handles = []
    for prec in PRECISIONS:
        handles.append(plt.Rectangle((0,0), 1, 1, fc=COLORS[prec], ec='none',
                                     label=f'{prec} total'))
    for prec in PRECISIONS:
        handles.append(plt.Rectangle((0,0), 1, 1, fc='white', hatch='///',
                                     ec=COLORS[prec], lw=0.8,
                                     label=f'{prec} refresh'))
    return handles


def save_fig(total, refresh, out_path):
    all_vals = np.concatenate([v for d in [total, refresh] for v in d.values()])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    draw_subfigure(ax, total, refresh)
    ax.set_ylim(all_vals.min() - 5, all_vals.max() * 1.38)
    fig.legend(handles=make_legend(), loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.8, rect=[0, 0.20, 1, 1])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# -- Generate both figures -----------------------------------------------------
save_fig(small_total, small_refresh, 'outputs/energy_saving_small.pdf')
save_fig(large_total, large_refresh, 'outputs/energy_saving_large.pdf')
