import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
import os

OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)

NAN = 999
INF = 998

trefw_labels = ['x8', 'x16', 'x32', 'x64', 'x100', 'x128']
n_bits_labels = list(range(7, 17))

llama_fp16 = np.array([
    [0.01,  0.01,  0.01,  0.02,  0.29, 0.38, 0.18, 0.14, NAN,  NAN ],
    [0.01,  0.01,  0.01,  0.02,  0.29, 0.38, 0.18, 0.14, NAN,  NAN ],
    [0.01,  0.01,  0.11,  0.45,  INF,  7.47, 9.11, 6.59, NAN,  NAN ],
    [0.02,  0.03,  0.30,  5.73,  INF,  5.7,  7.3,  INF,  NAN,  NAN ],
    [0.02,  0.10,  1.07, 28.10,  INF,  INF,  NAN,  INF,  NAN,  NAN ],
    [0.02,  0.20,  1.80, 82.40,  INF,  INF,  NAN,  INF,  NAN,  NAN ],
])

opt_fp16 = np.array([
    [0.01,  0.01,  0.01,  0.01,  0.01, 0.01, 0.06,  0.06, INF,  INF ],
    [0.01,  0.01,  0.01,  0.01,  0.01, 0.01, 0.06,  0.06, INF,  INF ],
    [0.01,  0.01,  0.01,  0.01,  0.01, 0.01, 0.06,  0.06, INF,  INF ],
    [0.01,  0.01,  0.00,  0.01,  0.01, 0.03, 9.6,   INF,  NAN,  NAN ],
    [0.00,  0.00,  0.01,  0.01,  0.01, 0.18, 11.0,  INF,  NAN,  NAN ],
    [0.00, -0.01,  0.01,  0.03,  0.01, 0.34, INF,   INF,  NAN,  NAN ],
])

resnet_fp16 = np.array([
    [1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  4.92 ],
    [1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64, -100., -100. ],
    [1.64,  1.64,  1.64,  1.64, -1.64,  4.92, -100.,  1.64, -100., -100. ],
    [1.64,  1.64,  1.64,  1.64, -2.46, -0.82,-98.36,-55.74, -100., -100. ],
    [0.00,  0.00,  0.00,  0.00, -2.46,-45.90,-98.36, -100., -100., -100. ],
])
resnet_trefw_labels = ['x16', 'x32', 'x64', 'x100', 'x128']

# ── FP8 E4M3 datasets (% delta from baseline, NaN=999 for not measured) ───────
# x-axis: n lower bits reduced, 8→1 (left to right)
fp8_trefw_labels  = ['x8', 'x32', 'x64', 'x100', 'x128']
fp8_n_bits_labels = [1, 2, 3, 4, 5, 6, 7, 8]

# LLaMA FP8 PPL % delta  (baseline PPL = 13.042068)
llama_fp8 = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, NAN, NAN, NAN], # x8
    [0.00, 0.00, 0.01, 0.01, 0.01, NAN, NAN, NAN], # x32
    [0.00, 0.00, 0.00, 0.01, NAN, NAN, NAN, NAN],  # x64
    [0.00, 0.01, NAN, NAN, NAN, NAN, NAN, NAN],    # x100
    [0.01, 0.01, NAN, NAN, NAN, NAN, NAN, NAN],    # x128
])

# OPT FP8 PPL % delta  (baseline PPL = 14.407216)
opt_fp8 = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, NAN, NAN, NAN], # x8
    [0.00, 0.00, 0.00, 0.00, NAN, NAN, NAN, NAN],  # x32
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],      # x64
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],      # x100
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],      # x128
])

# ResNet-50 FP8 Top-5 accuracy % delta  (baseline acc = 0.9375)
resnet_fp8 = np.array([
    [ 0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05],  # x8
    [ 0.05,  0.05,  0.05,  0.05,  0.05,-100.0,  0.80,  0.80],  # x32
    [ 0.05,  0.05,  0.05,  0.05,  0.05,-100.0,-100.0, -0.80],  # x64
    [ 0.05,  0.05,  0.05,  0.80,  0.80,-100.0,-100.0,-100.0],  # x100
    [ 0.05,  0.05,  0.05,  0.80, -0.80,-100.0,-100.0,-100.0],  # x128
])

# ── Color scheme ─────────────────────────────────────────────────────────────
# Good zone: solid green. Bad zone: continuous gradient amber → red → crimson.
_GREEN_FILL = '#4a7c52'   # muted forest green
_GREEN_TEXT = '#ffffff'
_NA_FILL    = '#2d2d2d'   # near-black charcoal (INF / NaN)
_NA_TEXT    = '#ffffff'

# unsaturated: warm amber → muted red → dark crimson
_BAD_CMAP = LinearSegmentedColormap.from_list(
    'muted_bad',
    ['#c4883a', '#a84f2a', '#8c2e20', '#6e1a14', '#4a0f0f']
)

def _bad_color(t):
    """t in [0, 1]: 0 = mildly bad, 1 = catastrophic."""
    t = max(0.0, min(1.0, t))
    fill = mcolors.to_hex(_BAD_CMAP(t))
    return fill, '#ffffff'

def cell_color_ppl(val, threshold=3.0):
    if val >= 500:          return _NA_FILL, _NA_TEXT       # N/A
    if val >= 200:          return _bad_color(1.0)           # INF
    if val < threshold:     return _GREEN_FILL, _GREEN_TEXT  # safe
    t = min(1.0, math.log(val / threshold) / math.log(100.0 / threshold))
    return _bad_color(t)

def cell_color_acc(val, threshold=3.0):
    if val <= -500 or val >= 500: return _NA_FILL, _NA_TEXT  # N/A
    if val > -threshold:          return _GREEN_FILL, _GREEN_TEXT  # safe
    # linear scale: -threshold → 0.0, -100 → 1.0
    t = min(1.0, (-val - threshold) / (100.0 - threshold))
    return _bad_color(t)

def format_val_ppl(val):
    if val >= 500:  return 'NaN'              # 999 = not measured
    if val >= 200:  return 'INF'              # 998 = diverged
    if val >= 10:   return f'+{val:.1f}%'
    return f'{val:+.2f}%'

def format_val_acc(val):
    if val >= 500 or val <= -500: return 'NaN'
    return f'{val:+.2f}%'

def plot_heatmap(data, trefw_labels, n_bits_labels, title,
                 color_fn, format_fn, threshold, filename, selected=None,
                 mantissa_bits=None):
    nrows, ncols = data.shape
    cell_in  = 0.22   # physical inches per cell — guarantees square
    left_in  = 0.60   # y-label + ytick labels
    right_in = 0.08   # right padding
    top_in   = 0.35   # title
    bot_in   = 0.45   # x-label + xtick labels
    cbar_w   = 0.10   # colorbar width
    cbar_gap = 0.06   # gap between heatmap and colorbar
    cbar_rpad = 0.40  # space for colorbar tick labels + cb label

    axes_w = ncols * cell_in
    axes_h = nrows * cell_in
    fig_w  = left_in + axes_w + cbar_gap + cbar_w + cbar_rpad
    fig_h  = top_in  + axes_h + bot_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    ax = fig.add_axes([
        left_in / fig_w, bot_in / fig_h,
        axes_w  / fig_w, axes_h / fig_h
    ])
    ax.set_facecolor('white')

    cbar_ax = fig.add_axes([
        (left_in + axes_w + cbar_gap) / fig_w, bot_in / fig_h,
        cbar_w / fig_w, axes_h / fig_h
    ])

    is_ppl = (color_fn is cell_color_ppl)

    for r in range(nrows):
        for c in range(ncols):
            val = data[r, c]
            fc, tc = color_fn(val, threshold)
            rect = plt.Rectangle(
                (c * cell_in, r * cell_in), cell_in, cell_in,
                facecolor=fc, edgecolor=(0, 0, 0, 0.18), linewidth=0.4, zorder=1
            )
            ax.add_patch(rect)

    if selected:
        for (tlab, nb) in selected:
            if tlab not in trefw_labels or nb not in n_bits_labels:
                print(f'  Warning: ({tlab}, {nb}) not found, skipping star')
                continue
            r = trefw_labels.index(tlab)
            c = n_bits_labels.index(nb)
            ax.text(
                c * cell_in + cell_in / 2,
                r * cell_in + cell_in * 0.5,
                u'\u2605',
                ha='center', va='center',
                fontsize=7, color='#b8860b', zorder=10
            )

    # ── Mantissa / exponent divider ─────────────────────────────────────────
    if mantissa_bits is not None:
        mb_next = mantissa_bits + 1
        if mantissa_bits in n_bits_labels and mb_next in n_bits_labels:
            idx_m   = n_bits_labels.index(mantissa_bits)
            idx_exp = n_bits_labels.index(mb_next)
            line_x  = (min(idx_m, idx_exp) + 1) * cell_in
            ax.axvline(line_x, color='white', linewidth=1.2, zorder=5)

            # Labels above the heatmap
            y_lbl = nrows * cell_in + 0.03
            # idx_m < idx_exp means ascending labels: mantissa on left
            if idx_m < idx_exp:
                mant_mid = line_x / 2
                exp_mid  = (line_x + ncols * cell_in) / 2
            else:
                mant_mid = (line_x + ncols * cell_in) / 2
                exp_mid  = line_x / 2
            ax.text(mant_mid, y_lbl, 'mantissa', ha='center', va='bottom',
                    fontsize=4.5, color='#444444', clip_on=False)
            ax.text(exp_mid,  y_lbl, 'exponent/sign', ha='center', va='bottom',
                    fontsize=4.5, color='#444444', clip_on=False)

    ax.set_xlim(0, ncols * cell_in)
    ax.set_ylim(0, nrows * cell_in)
    ax.set_xticks([c * cell_in + cell_in / 2 for c in range(ncols)])
    ax.set_xticklabels([str(nb) for nb in n_bits_labels], color='black', fontsize=5.5)
    ax.set_yticks([r * cell_in + cell_in / 2 for r in range(nrows)])
    ax.set_yticklabels(trefw_labels, color='black', fontsize=5.5)
    ax.set_xlabel('# bits with reduced refresh (from LSB)', color='black', fontsize=6, labelpad=4)
    ax.set_ylabel('tREFW multiplier', color='black', fontsize=6, labelpad=4)
    ax.tick_params(colors='black', length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)
    ax.set_title(title, color='black', fontsize=7, fontweight='bold', pad=6)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    # Both PPL and acc use the same visual orientation: green (safe) at the
    # bottom, bad gradient in the middle, NaN/INF at the top.
    # For acc the tick labels are negated to reflect the drop direction.
    nan_band = 14.0  # virtual units reserved for NaN/INF band
    vmin_cb, vmax_cb = 0.0, 100.0 + nan_band
    span = vmax_cb - vmin_cb
    tg = threshold / span
    tb = 100.0 / span
    cmap_colors = [
        (0.0,  _GREEN_FILL),
        (tg,   _GREEN_FILL),
        (tg,   '#c4883a'),
        (tb,   '#4a0f0f'),
        (tb,   _NA_FILL),
        (1.0,  _NA_FILL),
    ]
    cb_ticks = [threshold, 100, 100 + nan_band / 2]
    if is_ppl:
        cb_label = 'PPL increase (%)'
        cb_tlbls = [str(int(threshold)), '100', 'NaN/INF']
    else:
        cb_label = 'Top-5 accuracy loss (%)'
        cb_tlbls = [f'\u2212{int(threshold)}', '\u2212100', 'NaN/INF']

    cmap = LinearSegmentedColormap.from_list('_cb', cmap_colors)
    norm = mcolors.Normalize(vmin=vmin_cb, vmax=vmax_cb)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_tlbls)
    cbar.ax.tick_params(labelsize=5, length=2)
    cbar.outline.set_visible(False)
    cbar.set_label(cb_label, fontsize=5.5, labelpad=3)

    plt.savefig(filename, bbox_inches='tight', transparent=False)
    plt.close()
    print(f'Saved {filename}')

# ── Generate ──────────────────────────────────────────────────────────────────
plot_heatmap(
    data=llama_fp16, trefw_labels=trefw_labels, n_bits_labels=n_bits_labels,
    title='LLaMA FP16',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_llama_fp16.pdf'),
    selected=[('x32', 10)],
    mantissa_bits=10,
)

plot_heatmap(
    data=opt_fp16, trefw_labels=trefw_labels, n_bits_labels=n_bits_labels,
    title='OPT FP16',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_opt_fp16.pdf'),
    selected=[('x128', 10)],
    mantissa_bits=10,
)

plot_heatmap(
    data=resnet_fp16, trefw_labels=resnet_trefw_labels, n_bits_labels=n_bits_labels,
    title='ResNet50 FP16',
    color_fn=cell_color_acc, format_fn=format_val_acc,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_resnet50_fp16.pdf'),
    selected=[('x128', 10)],
    mantissa_bits=10,
)

plot_heatmap(
    data=llama_fp8, trefw_labels=fp8_trefw_labels, n_bits_labels=fp8_n_bits_labels,
    title='LLaMA FP8',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_llama_fp8.pdf'),
    selected=[('x64', 3)],
    mantissa_bits=3,
)

plot_heatmap(
    data=opt_fp8, trefw_labels=fp8_trefw_labels, n_bits_labels=fp8_n_bits_labels,
    title='OPT FP8',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_opt_fp8.pdf'),
    selected=[('x32', 3)],
    mantissa_bits=3,
)

plot_heatmap(
    data=resnet_fp8, trefw_labels=fp8_trefw_labels, n_bits_labels=fp8_n_bits_labels,
    title='ResNet50 FP8',
    color_fn=cell_color_acc, format_fn=format_val_acc,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_resnet50_fp8.pdf'),
    selected=[('x128', 3)],
    mantissa_bits=3,
)

print('All done.')