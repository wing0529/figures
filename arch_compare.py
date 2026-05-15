"""
Bar chart: speedup from refresh reduction when allocating bits
to different levels of the DRAM hierarchy (FRCAM, Channel, Row).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)


# ── Data ─────────────────────────────────────────────────────────────────
levels   = ['FRCAM', 'Channel', 'Row']
throughputs = np.array([1.140750375, 1.003298042, 0.2305128284])


# ── Colours ──────────────────────────────────────────────────────────────
COLOR_BAR = '#2D1040'
colors = [COLOR_BAR] * len(throughputs)


# ── Font / style settings ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : ['Arial Narrow', 'Arial', 'DejaVu Sans'],
    'font.weight'       : 'bold',
    'font.size'         : 8,
    'axes.titlesize'    : 8,
    'axes.labelsize'    : 8,
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'legend.fontsize'   : 8,
    'figure.titlesize'  : 8,
    'axes.linewidth'    : 0.6,
    'xtick.major.width' : 0.5,
    'ytick.major.width' : 0.5,
    'xtick.minor.width' : 0.3,
    'ytick.minor.width' : 0.3,
    'xtick.major.size'  : 0,
    'ytick.major.size'  : 0,
    'xtick.major.pad'   : 2,
    'ytick.major.pad'   : 2,
})


# ── Figure & axes ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(2.8, 2.4))

x         = np.arange(len(levels))
bar_width = 0.5

bars = ax.bar(x, throughputs, bar_width, color=colors, zorder=3)

# Baseline = 1.0
ax.axhline(1.0, color='black', linewidth=0.6, linestyle='--', zorder=2)

# Value labels
for i, v in enumerate(throughputs):
    offset = 0.015
    va     = 'bottom'
    ax.text(
        x[i], v + offset,
        f'{v:.3f}',
        ha='center', va=va,
        fontsize=7, fontweight='bold',
        zorder=4,
    )


# ── Axes styling ─────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel('Normalized throughput', fontweight='bold')

# Scale: start from 0, top gives clear room above FRCAM (1.14)
ax.set_ylim(0, 1.30)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ── Save ─────────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.4)
plt.savefig('outputs/arch_compare.pdf', bbox_inches='tight')
plt.savefig('outputs/arch_compare.png', dpi=300, bbox_inches='tight')
print("Saved outputs/arch_compare.pdf and .png")
