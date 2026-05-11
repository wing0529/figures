"""
Bar chart showing DRAM refresh throughput loss (%)
for 8 Gb, 16 Gb, and 32 Gb configurations.

Design principles adapted from Andrey Churkin's beautiful-figure example.
"""

import numpy as np
import matplotlib.pyplot as plt

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# ── Data ─────────────────────────────────────────────────────────────────
configurations = ['8 Gb', '16 Gb', '32 Gb']

# Throughput loss due to refresh (%)
refresh_loss_pct = np.array([17.1, 17.1, 17.6])


# ── Colours ──────────────────────────────────────────────────────────────
color_refresh = '#b0b0b0'   # light grey


# ── Font settings ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans Mono',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.minor.size': 0,
    'ytick.minor.size': 0,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
})


# ── Figure & axes ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(2.2, 2.0))

x = np.arange(len(configurations))
bar_width = 0.5

bars = ax.bar(
    x, refresh_loss_pct, bar_width,
    color=color_refresh,
    zorder=3,
)


# ── Percentage labels above each bar ─────────────────────────────────────
for i in range(len(configurations)):
    ax.text(
        x[i], refresh_loss_pct[i] + 0.3,
        f'{refresh_loss_pct[i]:.1f}%',
        ha='center', va='bottom',
        fontsize=8, fontweight='bold',
        zorder=4,
    )


# ── Axes styling ─────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(configurations)
ax.set_ylabel('Throughput Loss (%)', fontweight='bold')
ax.set_xlabel('Density', fontweight='bold')

ax.set_ylim(0, max(refresh_loss_pct) * 1.2)



# Remove top & right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ── Save & show ──────────────────────────────────────────────────────────
fig.tight_layout(pad=0.4)
os.makedirs('../output_figures', exist_ok=True)
plt.savefig('../output_figures/refresh_throughput.png', dpi=300, bbox_inches='tight')
plt.savefig('../output_figures/refresh_throughput.pdf', bbox_inches='tight')
plt.savefig('../output_figures/refresh_throughput.svg', bbox_inches='tight')

plt.show()
