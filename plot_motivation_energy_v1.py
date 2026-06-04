#!/usr/bin/env python3
"""
Stacked bar chart: Refresh vs Non-Refresh energy proportion by LPDDR5 die density.
Reproduces the style of Fig. 2 in the paper.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ── Font ──────────────────────────────────────────────────────────────────────
ARIAL_NARROW_BOLD = '/home/wing02/.fonts/arialnarrow_bold.ttf'
if os.path.exists(ARIAL_NARROW_BOLD):
    fm.fontManager.addfont(ARIAL_NARROW_BOLD)
    font_name = fm.FontProperties(fname=ARIAL_NARROW_BOLD).get_name()
else:
    font_name = 'DejaVu Sans'

plt.rcParams.update({
    'font.family':       font_name,
    'font.weight':       'bold',
    'font.size':         11,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
})

# ── Data ──────────────────────────────────────────────────────────────────────
densities = ['8 Gb', '16 Gb', '32 Gb']
refresh     = [14, 22, 34]          # % Refresh energy
non_refresh = [86, 78, 66]          # % Non-Refresh energy

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_REFRESH     = '#3d3d3d'   # dark gray  (Refresh)
COLOR_NON_REFRESH = '#b0b0b0'   # light gray (Non-Refresh)
EDGE_COLOR        = 'white'

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.2))

x = range(len(densities))
bar_w = 0.52
# Refresh (bottom) ← 순서 바꿈
bars_r = ax.bar(x, refresh, width=bar_w,
                color=COLOR_REFRESH, edgecolor=EDGE_COLOR,
                linewidth=0.6, label='Refresh', zorder=3)

# Non-Refresh (top, stacked) ← bottom=refresh 로 변경
bars_nr = ax.bar(x, non_refresh, width=bar_w, bottom=refresh,
                 color=COLOR_NON_REFRESH, edgecolor=EDGE_COLOR,
                 linewidth=0.6, label='Non-Refresh', zorder=3)



# ── Text labels inside bars ───────────────────────────────────────────────────
for i, (nr, r) in enumerate(zip(non_refresh, refresh)):
    # Refresh label (center of bottom segment)
    ax.text(i, r / 2, f'{r}%',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=5)
    # Non-Refresh label (center of top segment)
    ax.text(i, r + nr / 2, f'{nr}%',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='black', zorder=5)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(list(x))
ax.set_xticklabels(densities, fontweight='bold')
ax.set_yticks(range(0, 101, 20))
ax.set_ylim(0, 100)
ax.set_ylabel('Proportion (%) of\nActive Energy', fontweight='bold')
ax.set_xlabel('Density', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(
    handles=[bars_nr, bars_r],
    labels=['Non-Refresh', 'Refresh'],
    loc='upper right',
    fontsize=9,
    frameon=True,
    framealpha=0.9,
    edgecolor='#cccccc',
    ncol=2,
    bbox_to_anchor=(1.02, 1.08),
)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    path = os.path.join(out_dir, f'refresh_power.{ext}')
    dpi  = 600 if ext == 'png' else None
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f'Saved: {path}')

plt.close(fig)