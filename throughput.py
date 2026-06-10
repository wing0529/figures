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
import csv
import matplotlib.font_manager as fm
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('outputs', exist_ok=True)


SIM_ROOT = Path(__file__).resolve().parents[1] / 'SCALE-SIMv3_Ramulator2' / 'SCALE-Sim'
WORKLOAD_LABELS = ['Llama 3.2-1B', 'OPT-2.7B', 'ResNet-50']
MODEL_ORDER = ['llama', 'opt', 'resnet']


def load_speedups(csv_path, variant, fallback):
    """Return workload labels and speedup arrays from evaluate_workloads*.py CSV."""
    if not csv_path.exists():
        print(f'Using fallback throughput data: {csv_path} not found')
        return fallback

    rows = {}
    with csv_path.open(newline='') as f:
        for row in csv.DictReader(f):
            rows[(row['workload'], row['config'])] = float(row['cycles'])

    values = {}
    for config in ['fp16', 'bf16', 'fp8']:
        speedups = []
        for model in MODEL_ORDER:
            workload = f'{model}_{variant}'
            base = rows[(workload, 'baseline')]
            cycles = rows[(workload, config)]
            speedups.append(base / cycles)
        values[config] = np.array(speedups)

    print(f'Loaded throughput data from {csv_path}')
    return WORKLOAD_LABELS, values['fp16'], values['bf16'], values['fp8']


# ── Data  (baseline_cycles / config_cycles) ──────────────────────────────────
# Current paper-facing mobile result: sa64/ch1 seq trace.
edge_fallback = (
    WORKLOAD_LABELS,
    np.array([1.1423, 1.1424, 1.1357]),
    np.array([1.0784, 1.0785, 1.0741]),
    np.array([1.0784, 1.0785, 1.0741]),
)

large_fallback = (
    WORKLOAD_LABELS,
    np.array([1.1409, 1.1412, 1.1372]),
    np.array([1.0772, 1.0774, 1.0831]),
    np.array([1.0772, 1.0774, 1.1371]),
)

large_dnns, large_FP16, large_BF16, large_FP8 = load_speedups(
    SIM_ROOT / 'workload_config_results_sa256_ch16_seq.csv',
    'server',
    large_fallback,
)
edge_dnns, edge_FP16, edge_BF16, edge_FP8 = load_speedups(
    SIM_ROOT / 'workload_config_results_sa64_ch1_seq.csv',
    'edge',
    edge_fallback,
)



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
_FONT_PATH = 'arialnarrow_bold.ttf'
_FONT_NAME = 'DejaVu Sans'
if Path(_FONT_PATH).exists():
    fm.fontManager.addfont(_FONT_PATH)
    _FONT_NAME = fm.FontProperties(fname=_FONT_PATH).get_name()

FIG_SCALE = float(os.environ.get('FIG_SCALE', '2.0'))

def S(x):
    return x * FIG_SCALE

def scaled_figsize(w, h):
    return (w * FIG_SCALE, h * FIG_SCALE)

plt.rcParams.update({
    'font.family'       : _FONT_NAME,
    'font.weight'       : 'bold',
    'font.size'         : S(11),
    'axes.titlesize'    : S(11),
    'axes.labelsize'    : S(11),
    'xtick.labelsize'   : S(11),
    'ytick.labelsize'   : S(11),
    'legend.fontsize'   : S(11),
    'figure.titlesize'  : S(11),
    'axes.linewidth'    : S(0.6),
    'xtick.major.width' : S(0.5),
    'ytick.major.width' : S(0.5),
    'xtick.major.size'  : S(0),
    'ytick.major.size'  : S(0),
    'xtick.major.pad'   : S(2),
    'ytick.major.pad'   : S(2),
    'pdf.fonttype'      : 42,
    'ps.fonttype'       : 42,
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
                   color=color, edgecolor='black', linewidth=S(0.8), zorder=3)
        ax.bar([], [], width, color=color, label=label, zorder=3)

    # Value labels above each bar
    for offset, (_, values, _) in zip(offsets, precisions):
        for i, v in enumerate(values):
            ax.text(x[i] + offset, v + 0.003,
                    f'{v:.3f}x',
                    ha='center', va='bottom',
                    fontsize=S(11), rotation=90, zorder=4)

    # Baseline reference line
    ax.axhline(1.0, color='#444444', linewidth=S(0.8), linestyle='--', zorder=2)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(dnns)
    ax.set_ylabel('Normalized throughput', fontweight='bold', labelpad=S(1.5))
    ax.set_xlim(-0.55, n - 0.45)

    ymin = 0.95
    ymax = max(FP16.max(), BF16.max(), FP8.max()) * 1.10
    ax.set_ylim(ymin, ymax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=S(0.35), linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)


def legend_patches():
    return [
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP16, label='FP16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_BF16, label='BF16'),
        plt.Rectangle((0,0), 1, 1, color=COLOR_FP8,  label='FP8'),
    ]


# ── Large-scale figure ────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=scaled_figsize(4, 2))
draw_suBFigure(ax1, large_dnns, large_FP16, large_BF16, large_FP8)

fig1.legend(handles=legend_patches(), loc='upper center',
            ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig1.tight_layout(pad=0.5, rect=[0.08, 0.08, 1, 0.98])
fig1.savefig('outputs/throughput_large.pdf', bbox_inches='tight')
plt.close(fig1)

# ── Edge-device figure ────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=scaled_figsize(4, 2))
draw_suBFigure(ax2, edge_dnns, edge_FP16, edge_BF16, edge_FP8)
fig2.legend(handles=legend_patches(), loc='upper center',
            ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig2.tight_layout(pad=0.5, rect=[0.08, 0.08, 1, 0.98])
fig2.savefig('outputs/throughput_edge.pdf', bbox_inches='tight')
plt.close(fig2)

# ── Combined figure (large + edge side by side) ───────────────────────────────
# fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True,
#                                   gridspec_kw={'wspace': 0})

# draw_suBFigure(ax_l, large_dnns, large_FP16, large_BF16, large_FP8)
# draw_suBFigure(ax_r, edge_dnns,  edge_FP16,  edge_BF16,  edge_FP8)

# ax_l.set_title(y=-0.28, label='Datacenter-scale', fontweight='bold')
# ax_r.set_title(y=-0.28, label='Edge-device', fontweight='bold')

# # sharey=True 쓰면 오른쪽 y축 label 중복되므로 제거
# ax_r.set_ylabel('')
# ax_r.spines['left'].set_visible(False)

# fig.tight_layout(pad=0.8, rect=[0, 0.10, 1, 0.90])
# fig.subplots_adjust(wspace=0)
# fig.legend(handles=legend_patches(), loc='upper center',
#            ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
# fig.savefig('outputs/throughput_combined.pdf', bbox_inches='tight')
# plt.close(fig)
# print('Saved outputs/throughput_combined.pdf')

# print('Saved outputs/throughput_large.pdf and outputs/throughput_edge.pdf and Saved outputs/throughput_combined.pdf')


'''
sa54_ch1_seq.csv 결과:
========================================================================
  RESULTS: DRAM Cycles
========================================================================
Workload                      baseline              fp16               fp8              bf16
--------------------------------------------------------------------------------------------
llama_edge               2,238,478,238     1,959,629,801     2,075,689,170     2,075,689,170
opt_edge                 6,546,243,908     5,730,055,473     6,069,545,094     6,069,545,094
resnet_edge                317,342,812       279,419,983       295,439,354       295,439,354

========================================================================
  RESULTS: Throughput Improvement vs. Baseline  (baseline_cycles / config_cycles)
========================================================================
Workload                          fp16               fp8              bf16
--------------------------------------------------------------------------
llama_edge                      1.1423x            1.0784x            1.0784x
opt_edge                        1.1424x            1.0785x            1.0785x
resnet_edge                     1.1357x            1.0741x            1.0741x

========================================================================
  RESULTS: Total Energy (nJ)
========================================================================
Workload                      baseline              fp16               fp8              bf16
--------------------------------------------------------------------------------------------
llama_edge              327,210,435.82    272,976,108.93    293,529,058.09    293,529,058.09
opt_edge                961,820,960.65    802,938,597.13    863,055,661.01    863,055,661.01
resnet_edge              46,194,722.08     38,885,940.50     41,729,891.98     41,729,891.98

========================================================================
  RESULTS: Energy Saving vs. Baseline (%)
========================================================================
Workload                          fp16               fp8              bf16
--------------------------------------------------------------------------
llama_edge                      16.57%            10.29%            10.29%
opt_edge                        16.52%            10.27%            10.27%
resnet_edge                     15.82%             9.67%             9.67%

========================================================================
  RESULTS: Refresh Energy Saving vs. Baseline (%)
========================================================================
Workload                          fp16               fp8              bf16
--------------------------------------------------------------------------
llama_edge                      66.84%            41.82%            41.82%
opt_edge                        66.82%            41.81%            41.81%
resnet_edge                     66.41%            41.21%            41.21%

Raw results saved to: /gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim/workload_config_results_sa64_ch1_seq.csv
========================================================================
root@ccecfbe77b40:/gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim# 

========================================================================
  RESULTS: DRAM Cycles
========================================================================
Workload                      baseline              fp16               fp8              bf16
--------------------------------------------------------------------------------------------
llama_server               504,552,167     1,481,159,150     1,568,793,958     1,568,793,958
opt_server               1,467,684,592     4,330,760,742     4,587,050,846     4,587,050,846
resnet_server               70,306,273       211,395,963       223,348,866       223,348,866

========================================================================
  RESULTS: Throughput Improvement vs. Baseline  (baseline_cycles / config_cycles)
========================================================================
Workload                          fp16               fp8              bf16
--------------------------------------------------------------------------
llama_server                    0.3406x            0.3216x            0.3216x
opt_server                      0.3389x            0.3200x            0.3200x
resnet_server                   0.3326x            0.3148x            0.3148x

========================================================================
  RESULTS: Total Energy (nJ)
========================================================================
Workload                      baseline              fp16               fp8              bf16
--------------------------------------------------------------------------------------------
llama_server            575,284,367.83    735,369,845.32    983,876,147.26    983,876,147.26
opt_server            1,685,939,512.65  2,157,290,612.44  2,884,003,236.86  2,884,003,236.86
resnet_server            75,495,664.95    103,706,123.03    137,684,328.47    137,684,328.47

========================================================================
  RESULTS: Energy Saving vs. Baseline (%)
========================================================================
Workload                          fp16               fp8              bf16
--------------------------------------------------------------------------
llama_server                   -27.83%           -71.02%           -71.02%
opt_server                     -27.96%           -71.06%           -71.06%
resnet_server                  -37.37%           -82.37%           -82.37%

========================================================================
  RESULTS: Refresh Energy Saving vs. Baseline (%)
========================================================================
Workload                          fp16               fp8              bf16
--------------------------------------------------------------------------
llama_server                   -12.25%           -97.10%           -97.10%
opt_server                     -12.54%           -97.41%           -97.41%
resnet_server                  -29.75%          -126.35%          -126.35%

Raw results saved to: /gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim/workload_config_results_sa256_ch16_seq.csv
========================================================================
root@ccecfbe77b40:/gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim# 

'''
