# -*- coding: utf-8 -*-
"""
Grouped bar chart: accelerator performance breakdown (Large & Edge).
Style mirrors energy.py: color = precision, bar style = cycle component.
Each DNN group has 3 precision clusters (fp16, bf16, fp8); within each
cluster 3 adjacent bars show data movement (solid), stall (///), and
systolic execution (xxx), all normalised to baseline total cycles.
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
else:
    _FONT_NAME = 'DejaVu Sans'

# ── Raw data: [total, compute, stall] summed across all layers ────────────────
# Layout: array shape (n_dnns, 3) — dnn order: LLaMA, OPT, ResNet

# OPT large-scale uses the valid dim-aware rerun. The older
# sweep_results/opt_dimaware/ run is diagnostic-only because it reused
# fixed-layout R2 traces with a dim-aware layout.
OPT_LARGE_SOURCE = (
    '/gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim/'
    'sweep_results/opt_dimaware_valid/opt_tpuv5/'
    'cycle_breakdown_summary_vs_config_opt_tpuv5.csv'
)
OPT_LARGE_BASE = [1568320, 1417990, 625635]
OPT_LARGE_FP16 = [1372191, 1256010, 463655]
OPT_LARGE_BF16 = [1366210, 1249197, 456842]
OPT_LARGE_FP8 = [1366069, 1248945, 456590]


# ResNet50 uses the valid dim-aware rerun generated from layouts/resnet_dimaware.csv.
RESNET_SOURCE = (
    '/gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim/'
    'sweep_results/resnet_dimaware_valid/all_cycle_breakdown_summary.csv'
)
RESNET_EDGE_BASE = [6664393, 6019166, 90311]
RESNET_EDGE_FP16 = [6371102, 5971221, 42366]
RESNET_EDGE_BF16 = [6357886, 5957832, 28977]
RESNET_EDGE_FP8 = [6371509, 5971550, 42695]
RESNET_LARGE_BASE = [1107098, 517054, 94299]
RESNET_LARGE_FP16 = [814882, 461549, 38794]
RESNET_LARGE_BF16 = [802573, 447211, 24456]
RESNET_LARGE_FP8 = [822144, 468733, 45978]

large_base = np.array([
    # llama 3.2-1B
    [34198+17241+619500+182985+182985,
     10604+11321+581570+131875+131875,
     381+1098+254403+50084+50084],
    # opt-2.7B
    OPT_LARGE_BASE,
    # resnet50
    RESNET_LARGE_BASE,
], dtype=float)

large_fp16 = np.array([
    # llama 3.2-1B
    [29740+12768+567434+148019+148019,
     10223+11302+529522+110100+110100,
     0+1079+202355+28309+28309],
    # opt-2.7B
    OPT_LARGE_FP16,
    # resnet50
    RESNET_LARGE_FP16,
], dtype=float)

large_bf16 = np.array([
    # llama 3.2-1B
    [29746+12768+567704+148289+148289,
     10223+11302+529792+110370+110370,
     0+1079+202625+28579+28579],
    # opt-2.7B
    OPT_LARGE_BF16,
    # resnet50
    RESNET_LARGE_BF16,
], dtype=float)

large_fp8 = np.array([
    # llama 3.2-1B
    [29746+12768+567704+148289+148289,
     10223+11302+529792+110370+110370,
     0+1079+202625+28579+28579],
    # opt-2.7B
    OPT_LARGE_FP8,
    # resnet50
    RESNET_LARGE_FP8,
], dtype=float)

# Edge-device config
edge_base = np.array([
    # llama 3.2-1B
    [54373+49425+9944361+2497555+2497555,
     39420+39881+9929597+2483069+2483069,
     637+1098+894+894+894],
    # opt-2.7B
    [59147+59725+15530121+3895123+3895123,
     49116+49362+15515325+3880125+3880125,
     637+883+1726+1726+1726],
    # resnet50
    RESNET_EDGE_BASE,
], dtype=float)

edge_fp16 = np.array([
    # llama 3.2-1B
    [49050+45151+9940172+2493651+2493651,
     38783+39862+9930011+2483483+2483483,
     0+1079+1308+1308+1308],
    # opt-2.7B
    [57119+56734+15528073+3892143+3892143,
     49542+49537+15515044+3879844+3879844,
     1063+1058+1445+1445+1445],
    # resnet50
    RESNET_EDGE_FP16,
], dtype=float)

edge_bf16 = np.array([
    # llama 3.2-1B
    [49055+45151+9940172+2493651+2493651,
     38783+39862+9930011+2483483+2483483,
     0+1079+1308+1308+1308],
    # opt-2.7B
    [58617+58766+15528720+3893637+3893637,
     49542+49537+15515322+3880122+3880122,
     1063+1058+1723+1723+1723],
    # resnet50
    RESNET_EDGE_BF16,
], dtype=float)

edge_fp8 = np.array([
    # llama 3.2-1B
    [49056+45151+9940172+2493651+2493651,
     38783+39862+9930011+2483483+2483483,
     0+1079+1308+1308+1308],
    # opt-2.7B
    [58617+58766+15528720+3893637+3893637,
     49542+49537+15515322+3880122+3880122,
     1063+1058+1723+1723+1723],
    # resnet50
    RESNET_EDGE_FP8,
], dtype=float)


# ── Decompose helper ──────────────────────────────────────────────────────────
def decompose(data, base):
    """Return (data_mov, stall, systolic) each normalised by base total."""
    bt      = base[:, 0]
    total, compute, stall = data[:, 0], data[:, 1], data[:, 2]
    return (total - compute) / bt, stall / bt, (compute - stall) / bt


# ── Colors & style ───────────────────────────────────────────────────────────
DNNS       = ['Llama 3.2-1B', 'OPT-2.7B', 'Resnet50']
PRECISIONS = ['FP16', 'BF16', 'FP8']

# # Colors distinguish cycle components within a bar
# COLOR_DM  = '#D94A64'   # data movement
# COLOR_ST  = '#8C1F6F'   # stall
# COLOR_SY  = '#2D1040'   # systolic execution

COLOR_DM  = '#E4AFCF'   # data movement
COLOR_ST  = '#BE6C91'   # stall
COLOR_SY  = '#382955'   # systolic execution


'''
Dark Purple, #382955,
Muted Slate Purple, #554E77
Medium Mauve, #74668C
Mauve Pink,#936C8E
Deep Rose Pink),#BE6C91,
Soft Rose Pink),#D489AE,
Light Pastel Pink,#E4AFCF,
Pale Lilac Pink,#EFD4E8

'''


plt.rcParams.update({
    'font.family':       _FONT_NAME, 
    'font.weight':       'bold',
    'font.size':         11,
    'axes.titlesize':    11,
    'axes.labelsize':    11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  0,
    'ytick.major.size':  0,
    'xtick.major.pad':   2,
    'ytick.major.pad':   2,
})

# ── Layout parameters ─────────────────────────────────────────────────────────
BW       = 0.22   # bar width
PREC_GAP = 0.04   # gap between precision bars within one DNN group

fp16_off = -BW - PREC_GAP
bf16_off =  0.0
fp8_off  =  BW + PREC_GAP
PREC_OFFS = {'FP16': fp16_off, 'BF16': bf16_off, 'FP8': fp8_off}


# ── Helper ────────────────────────────────────────────────────────────────────
def draw_subfigure(ax, base, fp16_raw, bf16_raw, fp8_raw):
    x    = np.arange(len(DNNS), dtype=float)
    raws = {'FP16': fp16_raw, 'BF16': bf16_raw, 'FP8': fp8_raw}

    for prec in PRECISIONS:
        off = PREC_OFFS[prec]
        dm, st, sy = decompose(raws[prec], base)
        xpos = x + off

        # Stack: systolic (bottom) → stall → data movement (top)
        ax.bar(xpos, sy, BW,
               color=COLOR_SY, edgecolor='black', linewidth=0.8, zorder=5)
        ax.bar(xpos, st, BW, bottom=sy,
               color=COLOR_ST, edgecolor='black', linewidth=0.8, zorder=5)
        ax.bar(xpos, dm, BW, bottom=sy + st,
               color=COLOR_DM, edgecolor='black', linewidth=0.8, zorder=5)

        totals = dm + st + sy
        for xi, t in zip(xpos, totals):
            # precision label just below x-axis tick area, rotated
            ax.text(xi, -0.02, prec,
                    ha='center', va='top', fontsize=10, rotation=45,
                    transform=ax.get_xaxis_transform(), zorder=4)
            ax.text(xi, t + 0.008, f'{t:.2f}',
                    ha='center', va='bottom', fontsize=10, rotation=90, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(DNNS)
    ax.tick_params(axis='x', bottom=False, top=False,
                   labelbottom=False, labeltop=True, pad=5)
    ax.set_xlim(-0.5, len(DNNS) - 0.5)
    ax.set_ylabel('Normalized cycles', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.35, linestyle=':', color='#cccccc', zorder=0)
    ax.set_axisbelow(True)


def make_legend():
    return [
        plt.Rectangle((0,0), 1, 1, fc=COLOR_SY, ec='none', label='systolic compute'),
        plt.Rectangle((0,0), 1, 1, fc=COLOR_ST, ec='none', label='stall'),
        plt.Rectangle((0,0), 1, 1, fc=COLOR_DM, ec='none', label='prefetch + drain'),
    ]


def save_fig(base, fp16_raw, bf16_raw, fp8_raw, out_path, top_pad=0.08):
    totals = np.concatenate([r[:, 0] / base[:, 0]
                             for r in [fp16_raw, bf16_raw, fp8_raw]])
    fig, ax = plt.subplots(figsize=(5.5, 3))
    draw_subfigure(ax, base, fp16_raw, bf16_raw, fp8_raw)
    ymax = max(1.06, totals.max() + top_pad)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def save_legend(out_path):
    fig, ax = plt.subplots(figsize=(4.8, 0.35))
    ax.axis('off')
    fig.legend(handles=make_legend(), loc='center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.5))
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved {out_path}')

def save_combined_fig():
    totals = np.concatenate([
        r[:, 0] / large_base[:, 0] for r in [large_fp16, large_bf16, large_fp8]
    ] + [
        r[:, 0] / edge_base[:, 0] for r in [edge_fp16, edge_bf16, edge_fp8]
    ])
    ymax = max(1.06, totals.max() + 0.08)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    draw_subfigure(ax_l, large_base, large_fp16, large_bf16, large_fp8)
    draw_subfigure(ax_r, edge_base,  edge_fp16,  edge_bf16,  edge_fp8)

    ax_r.set_ylabel('')

    ax_l.set_ylim(0, ymax)
    ax_l.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

    # ax_l.text(0.5, -0.22, 'Datacenter-scale', transform=ax_l.transAxes,
    #           ha='center', va='top', fontweight='bold', fontsize=11)
    # ax_r.text(0.5, -0.22, 'Edge-device', transform=ax_r.transAxes,
    #           ha='center', va='top', fontweight='bold', fontsize=11)

    fig.legend(handles=make_legend(), loc='upper center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(pad=0.5, rect=[0, 0.20, 1, 0.83])
    fig.savefig('outputs/accel_perf_combined.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved outputs/accel_perf_combined.pdf')

# ── Generate figures ──────────────────────────────────────────────────────────
save_legend('outputs/accel_perf_legend.pdf')
save_fig(large_base, large_fp16, large_bf16, large_fp8, 'outputs/accel_perf_large.pdf',top_pad=0.4)
save_fig(edge_base,  edge_fp16,  edge_bf16,  edge_fp8,  'outputs/accel_perf_edge.pdf', top_pad=0.4)


# -- Generate combined figure --------------------------------------------------
save_combined_fig()
