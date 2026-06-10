# -*- coding: utf-8 -*-
"""
Grouped bar chart: normalized DRAM energy relative to baseline.
Two separate output groups are emitted for edge-device and datacenter-scale runs.
For each workload x precision pair, total energy and refresh energy can be
plotted as separate panels or side by side.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.font_manager as fm
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)

FIGURE_DIR  = Path(__file__).resolve().parent
_FONT_PATH = 'arialnarrow_bold.ttf'
if Path(_FONT_PATH).exists():
    fm.fontManager.addfont(_FONT_PATH)
    _FONT_NAME = fm.FontProperties(fname=_FONT_PATH).get_name()


# -- Data ----------------------------------------------------------------------
SIM_ROOT = Path(__file__).resolve().parents[1] / 'SCALE-SIMv3_Ramulator2' / 'SCALE-Sim'
DNNS       = ['Llama 3.2-1B', 'OPT-2.7B', 'ResNet-50']
PRECISIONS = ['FP16', 'BF16', 'FP8']
MODEL_ORDER = ['llama', 'opt', 'resnet']
CONFIG_FOR_PREC = {'FP16': 'fp16', 'BF16': 'bf16', 'FP8': 'fp8'}

FIG_SCALE = 2.0

def S(x):
    return x * FIG_SCALE

def scaled_figsize(w, h):
    return (w * FIG_SCALE, h * FIG_SCALE)

plt.rcParams.update({
    'font.family':       _FONT_NAME,
    'font.weight':       'bold',
    'font.size':         S(11),
    'axes.labelsize':    S(11),
    'axes.titlesize':    S(11),
    'xtick.labelsize':   S(11),
    'ytick.labelsize':   S(11),
    'axes.linewidth':    S(0.7),
    'xtick.major.width': S(0.5),
    'ytick.major.width': S(0.5),
    'xtick.major.size':  S(3),
    'ytick.major.size':  S(3),
    'xtick.major.pad':   S(2),
    'ytick.major.pad':   S(2),
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def savings_to_normalized(data):
    return {prec: 1.0 - values / 100.0 for prec, values in data.items()}


def load_normalized_energy(csv_path, variant, fallback_total, fallback_refresh):
    """Return total/ref normalized energy from evaluate_workloads*.py CSV."""
    if not csv_path.exists():
        print(f'Using fallback normalized DRAM energy data: {csv_path} not found')
        return fallback_total, fallback_refresh

    rows = {}
    with csv_path.open(newline='') as f:
        for row in csv.DictReader(f):
            rows[(row['workload'], row['config'])] = {
                'tot_energy': float(row['tot_energy']),
                'ref_energy': float(row['ref_energy']),
            }

    total = {}
    refresh = {}
    for prec, config in CONFIG_FOR_PREC.items():
        total_vals = []
        refresh_vals = []
        for model in MODEL_ORDER:
            workload = f'{model}_{variant}'
            base = rows[(workload, 'baseline')]
            cur = rows[(workload, config)]
            total_vals.append(cur['tot_energy'] / base['tot_energy'])
            refresh_vals.append(cur['ref_energy'] / base['ref_energy'])
        total[prec] = np.array(total_vals)
        refresh[prec] = np.array(refresh_vals)

    print(f'Loaded normalized DRAM energy data from {csv_path}')
    return total, refresh


# Current paper-facing mobile result: sa64/ch1 seq trace.
# Values below are normalized energy ratios, i.e., config_energy / baseline_energy.
small_total_fallback = {
    'FP16': np.array([0.8343, 0.8348, 0.8418]),
    'BF16': np.array([0.8971, 0.8973, 0.9033]),
    'FP8':  np.array([0.8971, 0.8973, 0.9033]),
}
small_refresh_fallback = {
    'FP16': np.array([0.3316, 0.3318, 0.3359]),
    'BF16': np.array([0.5818, 0.5819, 0.5879]),
    'FP8':  np.array([0.5818, 0.5819, 0.5879]),
}

large_total_fallback = savings_to_normalized({
    'FP16': np.array([16.05, 16.16, 16.35]),
    'BF16': np.array([ 9.84,  9.88, 10.18]),
    'FP8':  np.array([ 9.84,  9.88, 16.34]),
})
large_refresh_fallback = savings_to_normalized({
    'FP16': np.array([66.97, 66.98, 66.53]),
    'BF16': np.array([41.89, 41.84, 41.74]),
    'FP8':  np.array([41.89, 41.84, 66.50]),
})

small_total, small_refresh = load_normalized_energy(
    SIM_ROOT / 'workload_config_results_sa64_ch1_seq.csv',
    'edge',
    small_total_fallback,
    small_refresh_fallback,
)
large_total, large_refresh = load_normalized_energy(
    SIM_ROOT / 'workload_config_results_sa256_ch16_seq.csv',
    'server',
    large_total_fallback,
    large_refresh_fallback,
)


# -- Mode ----------------------------------------------------------------------
# True  → Normalized energy (0–1, baseline = 1.0)
# False → Saved energy (%) computed from normalized energy
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

# plt.rcParams.update({
#     'font.family':       _FONT_NAME, 
#     'font.weight':       'bold',
#     'font.size':         11,
#     'axes.titlesize':    11,
#     'axes.labelsize':    11,
#     'xtick.labelsize':   11,
#     'ytick.labelsize':   11,
#     'legend.fontsize':   10,
#     'axes.linewidth':    0.6,
#     'xtick.major.width': 0.5,
#     'ytick.major.width': 0.5,
#     'xtick.major.size':  0,
#     'ytick.major.size':  0,
#     'xtick.major.pad':   2,
#     'ytick.major.pad':   2,
# })


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
        return v if normalized else (1 - v) * 100

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
                ax.bar(tot_x, tv, bw, color=COLORS[prec], edgecolor='black', zorder=3,lw=S(0.5))
                if normalized:
                    ax.text(tot_x, tv + 0.01, f'{tv:.2f}',
                            ha='center', va='bottom', fontsize=S(11), rotation=90, zorder=1)
                else:
                    lbl_y = tv + 0.8 if tv >= 0 else tv - 0.8
                    va    = 'bottom' if tv >= 0 else 'top'
                    ax.text(tot_x, lbl_y, f'{tv:.1f}',
                            ha='center', va=va, fontsize=S(11), rotation=90, zorder=1)

            if draw_mode in ('both', 'refresh') and refresh is not None:
                rv = conv(refresh[prec][i])
                bar_w = 0.1 if draw_mode == 'both' else bw
                ax.bar(ref_x, rv, bar_w, color=REF_COLORS[f'{prec}_refresh'], edgecolor='black',linewidth=S(0.5), zorder=3)
                if normalized:
                    ax.text(ref_x, rv + 0.01, f'{rv:.2f}',
                            ha='center', va='bottom', fontsize=S(11), rotation=90, zorder=1)
                else:
                    ax.text(ref_x, rv + 0.9, f'{rv:.1f}',
                            ha='center', va='bottom', fontsize=S(11), rotation=90, zorder=1)

    ref_line = 1.0 if normalized else 0
    #ax.axhline(ref_line, color='#555555', linewidth=0.8, linestyle='--', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(DNNS)
    ax.set_xlim(-0.5, len(DNNS) - 0.5)
    if normalized:
        ax.set_yticks([0.00, 0.25, 0.50, 0.75,1.00])
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
                                         ec=REF_COLORS[f'{prec}_refresh'], lw=S(0.8),
                                         label=f'{prec}'))
    return handles


def save_fig(total, refresh, out_path):
    all_vals = np.concatenate([v for d in [total, refresh] for v in d.values()])
    fig, ax = plt.subplots(figsize=scaled_figsize(4.5, 3.2))
    draw_subfigure(ax, total, refresh)
    ax.set_ylim(0, all_vals.max() * 1.38)
    ax.legend(handles=make_legend(), loc='upper left', ncol=2, frameon=True,
              framealpha=0.85, edgecolor='#cccccc', fontsize=S(8),
              borderpad=S(0.5), labelspacing=S(0.3), handlelength=S(1.2), handletextpad=S(0.4))
    fig.tight_layout(pad=S(0.8))
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

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=scaled_figsize(8.0, 2.8), sharey=True,
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

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=scaled_figsize(8.0, 2.8), sharey=True,
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
        ymax = 0.75
    else:
        all_vals = np.concatenate([v for d in [small_refresh, large_refresh] for v in d.values()])
        ymax = all_vals.max() * 1.38

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=scaled_figsize(8.0, 2.8), sharey=True,
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


def save_panel(data, out_path, draw_mode, ymax, normalized=False, legend=False):
    """Single-panel figure (no legend) for one scale × one type."""
    fig, ax = plt.subplots(figsize=scaled_figsize(4.0, 2))
    if draw_mode == 'total':
        draw_subfigure(ax, data, None, draw_mode='total', normalized=normalized)
    else:
        draw_subfigure(ax, None, data, draw_mode='refresh', normalized=normalized)
    ax.set_ylim(0, ymax)
    if legend:
        ax.legend(handles=make_legend(draw_mode=draw_mode), 
                  loc='lower center',             # 범례의 하단 기준점을
                  bbox_to_anchor=(0.5, 1.05),     # 그래프의 상단 밖에 배치
                  ncol=3, 
                  frameon=False, 
                  fontsize=S(10),
                  handlelength=S(1.2), 
                  handletextpad=S(0.4), 
                  columnspacing=S(1.0))
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')



def normalized_panel_ymax(data_groups, floor):
    """Leave room for rotated value labels in normalized single panels."""
    if not NORMALIZED:
        return floor
    max_val = max(float(np.max(values)) for data in data_groups for values in data.values())
    return max(floor, min(1.0, max_val + 0.12))

def save_legend_strip(draw_mode, out_path):
    """Standalone legend strip for total or refresh."""
    fig, ax = plt.subplots(figsize=scaled_figsize   (4.0, 0.45))
    ax.set_visible(False)
    fig.legend(handles=make_legend(draw_mode=draw_mode), loc='center',
               ncol=3, frameon=False, fontsize=S(10),
               handlelength=S(1.2), handletextpad=S(0.4), columnspacing=S(1.0))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# -- Generate figures ----------------------------------------------------------
_suffix = 'norm' if NORMALIZED else 'saving'

# -- 4 single panels + 2 legend strips (6 PDFs) --------------------------------
if NORMALIZED:
    _ymax = 1.00
else:
    _all = np.concatenate([v for d in [small_total, small_refresh,
                                       large_total, large_refresh]
                           for v in d.values()])
    _ymax = _all.max() * 1.38

#save_panel(large_total,   f'outputs/energy_{_suffix}_datacenter_total.pdf',   'total',   _ymax, NORMALIZED,legend=False)
#save_panel(small_total,   f'outputs/energy_{_suffix}_edge_total.pdf',         'total',   _ymax, NORMALIZED,legend=False)

if NORMALIZED:
    _ymax = normalized_panel_ymax([large_refresh, small_refresh], 0.75)
#save_panel(large_refresh, f'outputs/energy_{_suffix}_datacenter_refresh.pdf', 'refresh', _ymax, NORMALIZED,legend=False)
#save_panel(small_refresh, f'outputs/energy_{_suffix}_edge_refresh.pdf',       'refresh', _ymax, NORMALIZED,legend=False)
#save_legend_strip('total',   f'outputs/energy_{_suffix}_legend_total.pdf')
#save_legend_strip('refresh', f'outputs/energy_{_suffix}_legend_refresh.pdf')

# -- panels with legends --------------------------------
if NORMALIZED:
    _ymax = 1.00
else:
    _all = np.concatenate([v for d in [small_total, small_refresh,
                                       large_total, large_refresh]
                           for v in d.values()])
    _ymax = _all.max() * 1.38

#save_panel(large_total,   f'outputs/energy_{_suffix}_datacenter_total_legend.pdf',   'total',   _ymax, NORMALIZED,legend=True)
save_panel(small_total,   f'outputs/energy_{_suffix}_edge_total_legend.pdf',         'total',   _ymax, NORMALIZED,legend=True)


if NORMALIZED:
    _ymax = normalized_panel_ymax([large_refresh, small_refresh], 0.75)
#save_panel(large_refresh, f'outputs/energy_{_suffix}_datacenter_refresh_legend.pdf', 'refresh', _ymax, NORMALIZED,legend=True)
save_panel(small_refresh, f'outputs/energy_{_suffix}_edge_refresh_legend.pdf',       'refresh', _ymax, NORMALIZED,legend=True)



'''

sa54_ch1_seq.csv 결과:

'''

