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

OPT_LARGE_SOURCE = (
    '/gem5/SCALE-SIMv3_Ramulator2/SCALE-Sim/'
    'sweep_results/scaleaware_memory_wall_main/opt_sa256_ch16/opt_tpuv5/'
    'cycle_breakdown_summary_vs_config_opt_tpuv5.csv'
)
OPT_LARGE_BASE = [981811, 806259, 13904]
OPT_LARGE_FP16 = [943855, 796447, 4092]

large_base = np.array([
    [758729, 519828, 8633],  # LLaMA
    [981811, 806259, 13904], # OPT baseline only
    [819235, 484786, 62031], # ResNet
], dtype=float)

large_fp16 = np.array([
    [744731, 518570, 7375],
    [943855, 796447, 4092],
    [620018, 441863, 19108],
], dtype=float)

large_bf16 = np.array([
    [743583, 515381, 4186],
    [942765, 795351, 2996],
    [640956, 447907, 25152],
], dtype=float)

large_fp8 = np.array([
    [743583, 515381, 4186],
    [942753, 795339, 2984],
    [643764, 451219, 28464],
], dtype=float)

# large_base = np.array([
#     # llama 3.2-1B
#     [34198+17241+619500+182985+182985,
#      10604+11321+581570+131875+131875,
#      381+1098+254403+50084+50084],
#     # opt-2.7B
#     OPT_LARGE_BASE,
#     # resnet50
#     RESNET_LARGE_BASE,
# ], dtype=float)

# large_fp16 = np.array([
#     # llama 3.2-1B
#     [29740+12768+567434+148019+148019,
#      10223+11302+529522+110100+110100,
#      0+1079+202355+28309+28309],
#     # opt-2.7B
#     OPT_LARGE_FP16,
#     # resnet50
#     RESNET_LARGE_FP16,
# ], dtype=float)

# large_bf16 = np.array([
#     # llama 3.2-1B
#     [29746+12768+567704+148289+148289,
#      10223+11302+529792+110370+110370,
#      0+1079+202625+28579+28579],
#     # opt-2.7B
#     OPT_LARGE_BF16,
#     # resnet50
#     RESNET_LARGE_BF16,
# ], dtype=float)

# large_fp8 = np.array([
#     # llama 3.2-1B
#     [29746+12768+567704+148289+148289,
#      10223+11302+529792+110370+110370,
#      0+1079+202625+28579+28579],
#     # opt-2.7B
#     OPT_LARGE_FP8,
#     # resnet50
#     RESNET_LARGE_FP8,
# ], dtype=float)

# Edge-device config
edge_base = np.array([
    [6939647, 6464200, 2106189],   # LLaMA
    [10751025, 10249675, 3454320], # OPT
    [4491231, 2193445, 155160],    # ResNet
], dtype=float)

edge_fp16 = np.array([
    [6866191, 6399639, 2041628],
    [10678439, 10185726, 3390371],
    [4132823, 2070291, 32006],
], dtype=float)

edge_bf16 = np.array([
    [6901982, 6435336, 2077325],
    [10675419, 10182649, 3387294],
    [4120521, 2058027, 19742],
], dtype=float)

edge_fp8 = np.array([
    [6911926, 6438906, 2080895],
    [10681696, 10188814, 3393459],
    [4166872, 2104299, 66014],
], dtype=float)
# edge_base = np.array([
#     # llama 3.2-1B
#     [54373+49425+9944361+2497555+2497555,
#      39420+39881+9929597+2483069+2483069,
#      637+1098+894+894+894],
#     # opt-2.7B
#     OPT_EDGE_BASE,
#     # resnet50
#     RESNET_EDGE_BASE,
# ], dtype=float)

# edge_fp16 = np.array([
#     # llama 3.2-1B
#     [49050+45151+9940172+2493651+2493651,
#      38783+39862+9930011+2483483+2483483,
#      0+1079+1308+1308+1308],
#     # opt-2.7B
#     OPT_EDGE_FP16,
#     # resnet50
#     RESNET_EDGE_FP16,
# ], dtype=float)

# edge_bf16 = np.array([
#     # llama 3.2-1B
#     [49055+45151+9940172+2493651+2493651,
#      38783+39862+9930011+2483483+2483483,
#      0+1079+1308+1308+1308],
#     # opt-2.7B
#     OPT_EDGE_BF16,
#     # resnet50
#     RESNET_EDGE_BF16,
# ], dtype=float)

# edge_fp8 = np.array([
#     # llama 3.2-1B
#     [49056+45151+9940172+2493651+2493651,
#      38783+39862+9930011+2483483+2483483,
#      0+1079+1308+1308+1308],
#     # opt-2.7B
#     OPT_EDGE_FP8,
#     # resnet50
#     RESNET_EDGE_FP8,
# ], dtype=float)


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
            if not np.isfinite(t):
                ax.text(xi, 0.03, 'TBD',
                        ha='center', va='bottom', fontsize=8, rotation=90,
                        zorder=4)
                continue
            # precision label just below x-axis tick area, rotated
            ax.text(xi, -0.015, prec,
                    ha='center', va='top', fontsize=10, rotation=45,
                    transform=ax.get_xaxis_transform(), zorder=4)
            ax.text(xi, t + 0.008, f'{t:.3f}',
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
    fig, ax = plt.subplots(figsize=(6,3))
    draw_subfigure(ax, base, fp16_raw, bf16_raw, fp8_raw)
    fig.legend(handles=make_legend(), loc='center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
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
#save_legend('outputs/accel_perf_legend.pdf')
save_fig(large_base, large_fp16, large_bf16, large_fp8, 'outputs/accel_perf_large.pdf',top_pad=0.4)
save_fig(edge_base,  edge_fp16,  edge_bf16,  edge_fp8,  'outputs/accel_perf_edge.pdf', top_pad=0.4)


# -- Generate combined figure --------------------------------------------------
#save_combined_fig()
