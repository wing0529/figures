"""
Stacked bar chart showing DRAM energy consumption breakdown:
  Refresh Power vs Other Active Power vs Background Power
for 8 GB, 16 GB, and 32 GB configurations.

Design principles adapted from Andrey Churkin's beautiful-figure example.
"""

import numpy as np
import matplotlib.pyplot as plt

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# ── Data ─────────────────────────────────────────────────────────────────
configurations = ['8 Gb', '16 Gb', '32 Gb']

# Energy values
refresh_power     = np.array([92762,  157159, 285951])
non_refresh_power = np.array([551351, 551349, 551355])

total_power = non_refresh_power + refresh_power

# Convert to percentages (all bars sum to 100%)
non_refresh_pct = non_refresh_power / total_power * 100
refresh_pct     = refresh_power     / total_power * 100


# ── Colours ──────────────────────────────────────────────────────────────
color_background = '#b0b0b0'   # light grey  (non-refresh)
color_active     = '#7e7e7e'   # neutral grey
color_refresh    = '#4a4a4a'   # dark grey   (refresh)

edge_background  = '#909090'
edge_active      = '#4e4e4e'
edge_refresh     = '#2a2a2a'


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

# Stacked bars (bottom-up: refresh → non-refresh)
bars_ref = ax.bar(
    x, refresh_pct, bar_width,
    label='Refresh',
    color=color_refresh,
    zorder=3,
)

bars_nr = ax.bar(
    x, non_refresh_pct, bar_width,
    bottom=refresh_pct,
    label='Non-Refresh',
    color=color_background,
    zorder=3,
)


# ── Percentage labels inside each segment ────────────────────────────────
for i in range(len(configurations)):
    segments = [
        (refresh_pct[i],     0,              'white'),
        (non_refresh_pct[i], refresh_pct[i], '#2a2a2a'),
    ]
    for pct, bottom, clr in segments:
        if pct > 5:  # only label segments tall enough to read
            ax.text(
                x[i], bottom + pct / 2,
                f'{round(pct)}%',
                ha='center', va='center',
                fontsize=8, fontweight='bold', color=clr,
                zorder=4,
            )


# ── Axes styling ─────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(configurations)
ax.set_ylabel('Proportion (%) of\nActive Energy', fontweight='bold')
ax.set_xlabel('Density', fontweight='bold')

# All bars are 100%
ax.set_ylim(0, 105)



# Remove top & right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ── Legend ────────────────────────────────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
# Reverse so the legend order matches the visual stack (top → bottom)
ax.legend(
    handles[::-1], labels[::-1],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.14),
    ncol=2,
    frameon=False,
    handlelength=1.0,
    handletextpad=0.4,
    columnspacing=1.0,
)


# ── Save & show ──────────────────────────────────────────────────────────
fig.tight_layout(pad=0.4)
os.makedirs('../output_figures', exist_ok=True)
plt.savefig('../output_figures/refresh_power.png', dpi=300, bbox_inches='tight')
plt.savefig('../output_figures/refresh_power.pdf', bbox_inches='tight')
plt.savefig('../output_figures/refresh_power.svg', bbox_inches='tight')

plt.show()
