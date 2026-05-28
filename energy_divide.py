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
import matplotlib.font_manager as fm
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)



FIGURE_DIR  = Path(__file__).resolve().parent
_FONT_PATH = '/home/wing02/arialnarrow_bold.ttf'
if Path(_FONT_PATH).exists():
    fm.fontManager.addfont(_FONT_PATH)
    _FONT_NAME = fm.FontProperties(fname=_FONT_PATH).get_name()


# -- Data ----------------------------------------------------------------------
DNNS       = ['Llama 3.2-1B', 'OPT-2.7B', 'Resnet50']
PRECISIONS = ['FP16', 'BF16', 'FP8']

small_total   = {'FP16': np.array([16.97, 15.94, 18.44]),
                 'BF16': np.array([ 9.47,  8.44, 11.55]),
                 'FP8':  np.array([ 9.47,  8.44, 18.41])}

large_total   = {'FP16': np.array([16.05, 16.16, 16.35]),
                 'BF16': np.array([ 9.84,  9.88, 10.18]),
                 'FP8':  np.array([ 9.84,  9.88, 16.34])}

small_refresh = {'FP16': np.array([66.66, 66.64, 66.82]),
                 'BF16': np.array([41.63, 41.59, 42.06]),
                 'FP8':  np.array([41.63, 41.59, 66.69])}

large_refresh = {'FP16': np.array([66.97, 66.98, 66.53]),
                 'BF16': np.array([41.89, 41.84, 41.74]),
                 'FP8':  np.array([41.89, 41.84, 66.50])}


# -- Mode ----------------------------------------------------------------------
# True  → Normalized energy (0–1, baseline = 1.0)
# False → Saved energy (%)
NORMALIZED = True

# -- Colors & style ------------------------------------------------------------
COLORS ={
    'FP16': '#6A5C94',
    'BF16': '#8E7EA8',
    'FP8':  '#C99AB8'
}
REF_COLORS = {'FP16_refresh': '#8A4E7E',
              'BF16_refresh': '#BA7EAC',
              'FP8_refresh':  '#DCAECE'}
#COLORS = {'FP16': '#D94A64', 'BF16': '#8C1F6F', 'FP8': '#2D1040'}

plt.rcParams.update({
    'font.family':       _FONT_NAME, 
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
BW        = 0.1   # bar width
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
def draw_subfigure(ax, total, refresh, draw_mode='both', normalized=False):
    x = np.arange(len(DNNS), dtype=float)
    bw = BW if draw_mode == 'both' else BW * 2

    def conv(v):
        return 1 - v / 100 if normalized else v

    if draw_mode == 'both':
        bar_ctrs = PAIR_CTRS
    else:
        # 3 bars placed touching, centered on each DNN position
        cluster_w = 3 * bw
        fp16_ctr = -cluster_w / 2 + bw / 2
        bar_ctrs = {
            'FP16': fp16_ctr,
            'BF16': fp16_ctr + bw,
            'FP8':  fp16_ctr + bw * 2,
        }

    for prec in PRECISIONS:
        pc = bar_ctrs[prec]
        for i in range(len(DNNS)):
            if draw_mode == 'both':
                tot_x = x[i] + pc + TOT_OFF
                ref_x = x[i] + pc + REF_OFF
            else:
                tot_x = x[i] + pc
                ref_x = x[i] + pc

            if draw_mode in ('both', 'total') and total is not None:
                tv = conv(total[prec][i])
                ax.bar(tot_x, tv, bw, color=COLORS[prec], edgecolor='black', zorder=3)
                if normalized:
                    ax.text(tot_x, tv + 0.01, f'{tv:.2f}',
                            ha='center', va='bottom', fontsize=11, rotation=90, zorder=1)
                else:
                    lbl_y = tv + 0.8 if tv >= 0 else tv - 0.8
                    va    = 'bottom' if tv >= 0 else 'top'
                    ax.text(tot_x, lbl_y, f'{tv:.1f}',
                            ha='center', va=va, fontsize=11, rotation=90, zorder=1)

            if draw_mode in ('both', 'refresh') and refresh is not None:
                rv = conv(refresh[prec][i])
                bar_w = 0.1 if draw_mode == 'both' else bw
                ax.bar(ref_x, rv, bar_w, color=REF_COLORS[f'{prec}_refresh'], edgecolor='black', zorder=3)
                if normalized:
                    ax.text(ref_x, rv + 0.01, f'{rv:.2f}',
                            ha='center', va='bottom', fontsize=11, rotation=90, zorder=1)
                else:
                    ax.text(ref_x, rv + 0.9, f'{rv:.1f}',
                            ha='center', va='bottom', fontsize=11, rotation=90, zorder=1)

    ref_line = 1.0 if normalized else 0
    #ax.axhline(ref_line, color='#555555', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(DNNS)
    ax.set_xlim(-0.5, len(DNNS) - 0.5)
    if normalized:
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel('Normalized energy' if normalized else 'Saved energy (%)', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)


def make_legend(draw_mode='both'):
    handles = []
    if draw_mode in ('both', 'total'):
        for prec in PRECISIONS:
            handles.append(plt.Rectangle((0,0), 1, 1, fc=COLORS[prec], ec='none',
                                         label=f'{prec}'))
    if draw_mode in ('both', 'refresh'):
        for prec in PRECISIONS:
            handles.append(plt.Rectangle((0,0), 1, 1, fc=REF_COLORS[f'{prec}_refresh'], hatch='///',
                                         ec=REF_COLORS[f'{prec}_refresh'], lw=0.8,
                                         label=f'{prec}'))
    return handles


def save_fig(total, refresh, out_path):
    all_vals = np.concatenate([v for d in [total, refresh] for v in d.values()])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    draw_subfigure(ax, total, refresh)
    ax.set_ylim(0, all_vals.max() * 1.38)
    ax.legend(handles=make_legend(), loc='upper left', ncol=2, frameon=True,
              framealpha=0.85, edgecolor='#cccccc', fontsize=8,
              borderpad=0.5, labelspacing=0.3, handlelength=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

def save_combined_fig(small_total, small_refresh, large_total, large_refresh, out_path, normalized=False):
    if normalized:
        ymax = 1.15
    else:
        all_vals = np.concatenate([
            v for d in [small_total, small_refresh, large_total, large_refresh]
            for v in d.values()
        ])
        ymax = all_vals.max() * 1.38

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 2.8), sharey=True,
                                      gridspec_kw={'wspace': 0})

    draw_subfigure(ax_r, small_total, small_refresh, normalized=normalized)
    draw_subfigure(ax_l, large_total, large_refresh, normalized=normalized)

    ax_l.set_title(y=-0.28, label='Datacenter-scale', fontweight='bold')
    ax_r.set_title(y=-0.28, label='Edge-device', fontweight='bold')
    ax_r.set_ylabel('')
    ax_r.spines['left'].set_visible(False)

    ax_l.set_ylim(0, ymax)

    fig.tight_layout(pad=0.8, rect=[0, 0.10, 1, 0.90])
    fig.subplots_adjust(wspace=0)
    fig.legend(handles=make_legend(), loc='upper center',
               ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.savefig(out_path, bbox_inches='tight')

    plt.close(fig)
    print(f'Saved {out_path}')




def save_combined_fig_total(small_total, large_total, out_path, normalized=False):
    """Generate combined figure showing only total energy saving (solid bars)."""
    if normalized:
        ymax = 1.15
    else:
        all_vals = np.concatenate([v for d in [small_total, large_total] for v in d.values()])
        ymax = all_vals.max() * 1.38

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 2.8), sharey=True,
                                      gridspec_kw={'wspace': 0})

    draw_subfigure(ax_r, small_total, None, draw_mode='total', normalized=normalized)
    draw_subfigure(ax_l, large_total, None, draw_mode='total', normalized=normalized)

    ax_l.set_title(y=-0.28, label='Datacenter-scale', fontweight='bold')
    ax_r.set_title(y=-0.28, label='Edge-device', fontweight='bold')
    ax_r.set_ylabel('')
    ax_r.spines['left'].set_visible(False)

    ax_l.set_ylim(0, ymax)

    fig.tight_layout(pad=0.8, rect=[0, 0.10, 1, 0.90])
    fig.subplots_adjust(wspace=0)
    fig.legend(handles=make_legend(draw_mode='total'), loc='upper center',
               ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.savefig(out_path, bbox_inches='tight')

    plt.close(fig)
    print(f'Saved {out_path}')


def save_combined_fig_refresh(small_refresh, large_refresh, out_path, normalized=False):
    """Generate combined figure showing only refresh energy saving (hatched bars)."""
    if normalized:
        ymax = 1.15
    else:
        all_vals = np.concatenate([v for d in [small_refresh, large_refresh] for v in d.values()])
        ymax = all_vals.max() * 1.38

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 2.8), sharey=True,
                                      gridspec_kw={'wspace': 0})

    draw_subfigure(ax_r, None, small_refresh, draw_mode='refresh', normalized=normalized)
    draw_subfigure(ax_l, None, large_refresh, draw_mode='refresh', normalized=normalized)

    ax_l.set_title(y=-0.28, label='Datacenter-scale', fontweight='bold')
    ax_r.set_title(y=-0.28, label='Edge-device', fontweight='bold')
    ax_r.set_ylabel('')
    ax_r.spines['left'].set_visible(False)

    ax_l.set_ylim(0, ymax)

    fig.tight_layout(pad=0.8, rect=[0, 0.10, 1, 0.90])
    fig.subplots_adjust(wspace=0)
    fig.legend(handles=make_legend(draw_mode='refresh'), loc='upper center',
               ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.savefig(out_path, bbox_inches='tight')

    plt.close(fig)
    print(f'Saved {out_path}')


def save_panel(data, out_path, draw_mode, ymax, normalized=False):
    """Single-panel figure (no legend) for one scale × one type."""
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    if draw_mode == 'total':
        draw_subfigure(ax, data, None, draw_mode='total', normalized=normalized)
    else:
        draw_subfigure(ax, None, data, draw_mode='refresh', normalized=normalized)
    ax.set_ylim(0, ymax)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def save_legend_strip(draw_mode, out_path):
    """Standalone legend strip for total or refresh."""
    fig, ax = plt.subplots(figsize=(4.0, 0.45))
    ax.set_visible(False)
    fig.legend(handles=make_legend(draw_mode=draw_mode), loc='center',
               ncol=3, frameon=False, fontsize=10,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# -- Generate figures ----------------------------------------------------------
_suffix = 'norm' if NORMALIZED else 'saving'

save_combined_fig(small_total, small_refresh,
                  large_total, large_refresh,
                  f'outputs/energy_{_suffix}_combined.pdf',
                  normalized=NORMALIZED)

save_combined_fig_total(small_total, large_total,
                        f'outputs/energy_{_suffix}_total.pdf',
                        normalized=NORMALIZED)

save_combined_fig_refresh(small_refresh, large_refresh,
                          f'outputs/energy_{_suffix}_refresh.pdf',
                          normalized=NORMALIZED)

# -- 4 single panels + 2 legend strips (6 PDFs) --------------------------------
if NORMALIZED:
    _ymax = 1.15
else:
    _all = np.concatenate([v for d in [small_total, small_refresh,
                                       large_total, large_refresh]
                           for v in d.values()])
    _ymax = _all.max() * 1.38

save_panel(large_total,   f'outputs/energy_{_suffix}_datacenter_total.pdf',   'total',   _ymax, NORMALIZED)
save_panel(large_refresh, f'outputs/energy_{_suffix}_datacenter_refresh.pdf', 'refresh', _ymax, NORMALIZED)
save_panel(small_total,   f'outputs/energy_{_suffix}_edge_total.pdf',         'total',   _ymax, NORMALIZED)
save_panel(small_refresh, f'outputs/energy_{_suffix}_edge_refresh.pdf',       'refresh', _ymax, NORMALIZED)
save_legend_strip('total',   f'outputs/energy_{_suffix}_legend_total.pdf')
save_legend_strip('refresh', f'outputs/energy_{_suffix}_legend_refresh.pdf')


# -- Generate both figures -----------------------------------------------------
# save_fig(small_total, small_refresh, 'outputs/energy_saving_small.pdf')
# save_fig(large_total, large_refresh, 'outputs/energy_saving_large.pdf')


# -- v2: separate panels + shared legend (for LaTeX \subfigure (a)(b)) ---------
def save_v2_figs():
    all_vals = np.concatenate([v for d in [small_total, small_refresh,
                                           large_total, large_refresh]
                               for v in d.values()])
    ymax = all_vals.max() * 1.38

    # (a) Datacenter-scale
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    draw_subfigure(ax, large_total, large_refresh)
    ax.set_ylim(0, ymax)
    fig.tight_layout(pad=0.8)
    fig.savefig('outputs/energy_datacenter_v2.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved outputs/energy_datacenter_v2.pdf')

    # (b) Edge-device
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    draw_subfigure(ax, small_total, small_refresh)
    ax.set_ylim(0, ymax)
    fig.tight_layout(pad=0.8)
    fig.savefig('outputs/energy_edge_v2.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved outputs/energy_edge_v2.pdf')

    # shared legend strip
    fig, ax = plt.subplots(figsize=(8.0, 0.45))
    ax.set_visible(False)
    fig.legend(handles=make_legend(), loc='center', ncol=6, frameon=False,
               fontsize=10, handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    fig.savefig('outputs/energy_legend_v2.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved outputs/energy_legend_v2.pdf')


save_v2_figs()
save_combined_fig_total(small_total, large_total, 'outputs/energy_saving_combined_total.pdf')
save_combined_fig_refresh(small_refresh, large_refresh, 'outputs/energy_saving_combined_refresh.pdf')