# -*- coding: utf-8 -*-
"""
Energy saving vs. baseline (%) — Small accelerator scale.
For each workload × precision pair the total energy bar and the
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


# ── Data ──────────────────────────────────────────────────────────────────────
DNNS       = ['LLaMA', 'OPT', 'ResNet']
PRECISIONS = ['fp16', 'bf16', 'fp8']

total   = {'fp16': np.array([16.97, 15.94, 18.44]),
           'bf16': np.array([-2.88,  8.44, 18.44]),
           'fp8':  np.array([ 9.47,  8.44, 18.44])}

refresh = {'fp16': np.array([66.66, 66.64, 66.82]),
           'bf16': np.array([ 0.05, 41.59, 66.82]),
           'fp8':  np.array([41.63, 41.59, 66.82])}


# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {'fp16': '#D94A64', 'bf16': '#8C1F6F', 'fp8': '#2D1040'}


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       ['Liberation Sans Narrow', 'Arial Narrow', 'DejaVu Sans'],  # Liberation Sans Narrow = Arial Narrow on Linux
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


# ── Layout parameters ─────────────────────────────────────────────────────────
# Each workload cluster: 3 precisions × 2 bars (total + refresh)
# Within a precision pair: total bar | refresh bar (side by side)
BW         = 0.09    # bar width
INNER_GAP  = 0.01    # gap between total and refresh within one precision pair
PREC_GAP   = 0.04    # gap between different precision pairs

PAIR_W     = 2 * BW + INNER_GAP                             # 0.19
CLUSTER_W  = 3 * PAIR_W + 2 * PREC_GAP                      # 0.65

# Pair centres relative to workload centre
fp16_ctr = -CLUSTER_W/2 + PAIR_W/2                           # -0.23
bf16_ctr = fp16_ctr + PAIR_W + PREC_GAP                      #  0.00
fp8_ctr  = bf16_ctr + PAIR_W + PREC_GAP                      # +0.23
PAIR_CTRS = {'fp16': fp16_ctr, 'bf16': bf16_ctr, 'fp8': fp8_ctr}

# Bar offsets relative to pair centre
TOT_OFF = -(BW / 2 + INNER_GAP / 2)                          # -0.05
REF_OFF =  (BW / 2 + INNER_GAP / 2)                          # +0.05


# ── Drawing ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.2))
x = np.arange(len(DNNS), dtype=float)

for prec in PRECISIONS:
    pc = PAIR_CTRS[prec]
    for i, (tv, rv) in enumerate(zip(total[prec], refresh[prec])):
        tot_x = x[i] + pc + TOT_OFF
        ref_x = x[i] + pc + REF_OFF

        # Total — solid
        ax.bar(tot_x, tv, BW, color=COLORS[prec], edgecolor='none', zorder=3)
        lbl_y = tv + 0.8 if tv >= 0 else tv - 0.8
        va    = 'bottom' if tv >= 0 else 'top'
        ax.text(tot_x, lbl_y, f'{tv:.1f}',
                ha='center', va=va, fontsize=6, rotation=90, zorder=4)

        # Refresh — hatched
        ax.bar(ref_x, rv, BW,
               facecolor='white', hatch='///', edgecolor=COLORS[prec],
               linewidth=0.8, zorder=3)
        ax.text(ref_x, rv + 0.8, f'{rv:.1f}',
                ha='center', va='bottom', fontsize=6, rotation=90, zorder=4)

# Baseline
ax.axhline(0, color='#555555', linewidth=0.8, linestyle='--', zorder=2)

# Axes
ax.set_xticks(x)
ax.set_xticklabels(DNNS)
ax.set_xlim(-0.5, len(DNNS) - 0.5)
ax.set_ylabel('Energy saving (%)', fontweight='bold')

all_vals = np.concatenate([v for d in [total, refresh] for v in d.values()])
ax.set_ylim(all_vals.min() - 5, all_vals.max() * 1.38)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
ax.set_axisbelow(True)


# ── Legend ────────────────────────────────────────────────────────────────────
handles = []
for prec in PRECISIONS:
    handles.append(plt.Rectangle((0,0), 1, 1, fc=COLORS[prec], ec='none',
                                  label=f'{prec} total'))
for prec in PRECISIONS:
    handles.append(plt.Rectangle((0,0), 1, 1, fc='white', hatch='///',
                                  ec=COLORS[prec], lw=0.8,
                                  label=f'{prec} refresh'))

fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=0.8, rect=[0, 0.20, 1, 1])
fig.savefig('outputs/energy_saving_small.pdf', bbox_inches='tight')
plt.close(fig)

print('Saved outputs/energy_saving_small.pdf')
