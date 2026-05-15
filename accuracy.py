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

plt.rcParams.update({
    'font.family': 'DejaVu Sans Mono',
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
})

OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)

NAN = 999
INF = 998

trefw_labels = [ 'x16', 'x32', 'x64', 'x128']  # x100 removed
n_bits_labels     = list(range(16, 6, -1))  # [16,15,14,13,12,11,10,9,8,7] high→low
n_bits_labels_rev = list(range(7,  17))      # [7,8,...,16] low→high for FP16/BF16

# Llama FP16 PPL % delta — columns: n=16 (bit 15) → n=7 (bits 6-0), rows: x8→x128
Llama_fp16 = np.array([  # x8
    [NAN,  NAN,  0.07, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # x16
    [NAN,  NAN,  14.63, 0.45, 0.01, 0.02, 0.00, 0.01, 0.01, 0.01], # x32
    [NAN,  NAN,  INF,  6.12, 0.16, 0.06, 0.03, 0.02, 0.01, 0.01],  # x64
    [NAN,  NAN,  INF,  130.6, 1.61, 0.30, 0.05, 0.04, 0.01, 0.01], # x128
])

# OPT FP16 PPL % delta
opt_fp16 = np.array([  # x8
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00, 0.01],  # x16
    [NAN,  NAN,  0.57, 0.09, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # x32
    [NAN,  NAN,  INF,  INF,  0.03, 0.00, 0.00, 0.01, 0.01, 0.01],  # x64
    [NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  0.02, 0.01, 0.01, 0.01],  # x128
])

# ResNet-50 FP16 Top-5 accuracy % delta (negative = accuracy drop)
resnet_fp16 = np.array([

    [  0.00,   0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # x16
    [-100.0, -100.0,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # x32
    [-100.0, -100.0,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # x64
    [-100.0, -100.0, -2.46, -1.64, -1.64,  0.82,  0.00,  0.00,  0.00,  0.00],  # x128
])
resnet_trefw_labels = ['x16', 'x32', 'x64', 'x128']  # x100 removed

# Llama BF16 (E8M7) PPL % delta — baseline PPL = 12.894
# cols: n=16→7 (high→low), rows: x16(512ms)→x128(4096ms)
Llama_bf16 = np.array([
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # x16 (512ms)
    [NAN,  NAN,  4.00, 3.98, 3.99, 3.95, 0.41, 0.02, 0.02, 0.02],  # x32 (1024ms)
    [NAN,  NAN,  INF,  INF,  INF,  INF,  5.94, 0.12, 0.02, 0.02],  # x64 (2048ms)
    [NAN,  NAN,  INF,  INF,  INF,  INF,  104.8, 1.02, 0.14, 0.02], # x128 (4096ms)
])

# ResNet BF16 (E8M7) Top-5 accuracy % delta — baseline acc = 0.961
# cols: n=16→7 (high→low), rows: x16(512ms)→x128(4096ms)
resnet_bf16 = np.array([
    [-0.83, -0.83, -0.83, -0.83, -0.83, -0.83, -0.83, -0.83, -0.83, -0.83],  # x16 (512ms)
    [-100.0, -100.0, -0.83, -0.83, -0.83, -0.83, -0.83, -0.83, -0.83, -0.83],  # x32 (1024ms)
    [-100.0, -100.0, -0.83, -0.83, -2.39, -1.66, -0.83, -0.83, -0.83, -0.83],  # x64 (2048ms)
    [-100.0, -100.0, -4.06, -4.06, -3.23, -2.39, -2.39, -1.66, -1.66, -0.62],  # x128 (4096ms)
])
bf16_trefw_labels        = trefw_labels
resnet_bf16_trefw_labels = resnet_trefw_labels

# OPT BF16 (E8M7) PPL % delta — baseline PPL = 14.384
# cols: n=16→7 (high→low), rows: x16→x128
# Large PPL values (63125, 5.53e6, 1159, 27867) treated as INF
opt_bf16 = np.array([
    [NAN,  NAN,  0.03, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01,0.01],  # x16
    [NAN,  NAN,  0.03, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01,0.01],  # x32
    [NAN,  NAN,  INF,  INF,  INF,  INF,  196.7, 0.29, 0.10,0.01],  # x64  (tREFW=3600ms)
    [NAN,  NAN,  INF,  INF,  INF,  INF, INF,1.43, 0.05, 0.00],  # x128
])

# Column-flipped variants for reversed (low→high) FP16/BF16 x-axis
Llama_fp16_r  = np.fliplr(Llama_fp16)
opt_fp16_r    = np.fliplr(opt_fp16)
resnet_fp16_r = np.fliplr(resnet_fp16)
Llama_bf16_r  = np.fliplr(Llama_bf16)
opt_bf16_r    = np.fliplr(opt_bf16)
resnet_bf16_r = np.fliplr(resnet_bf16)

# FP8 E4M3 datasets (% delta from baseline, NaN=999 for not measured)
# x-axis: n lower bits reduced, 8->1 (left to right)
fp8_trefw_labels  = ['x16','x32', 'x64', 'x128']  # x100 removed
fp8_n_bits_labels = [8, 7, 6, 5, 4, 3, 2, 1]

# Llama FP8 PPL % delta  (cols: n=8→1 high-to-low, rows: x8→x128)
Llama_fp8 = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # x8
    [NAN,  NAN,  3.40, 0.40, 0.00, 0.00, 0.00, 0.00],  # x32
    [NAN,  NAN,  471., 7.20, 0.10, 0.00, 0.00, 0.00],  # x64
    [NAN,  NAN,  INF,  92.,  2.40, 0.30, 0.10, 0.00],  # x128
])

# OPT FP8 PPL % delta  (cols: n=8→1 high-to-low, rows: x8→x128)
opt_fp8 = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # x8
    [NAN,  NAN,  0.70, 0.10, 0.00, 0.00, 0.00, 0.00],  # x32
    [NAN,  NAN,  INF,  0.90, 0.00, 0.00, 0.00, 0.00],  # x64
    [NAN,  NAN,  INF,  182., 0.40, 0.10, 0.00, 0.00],  # x128
])

# ResNet-50 FP8 Top-5 accuracy % delta — baseline acc = 94.5% (lpddr5, Vendor-A)
# cols: n=8→1 (high→low), rows: x16(512ms)→x128(4096ms)
resnet_fp8 = np.array([
    [ -0.8,  -0.8,  -0.8,  -0.8,  -0.8,  -0.8,  -0.8,  -0.8],  # x16 (512ms)
    [-100.0, -100.0, -0.8,  -0.8,  -0.8,  -0.8,  -0.8,  -0.8],  # x32 (1024ms)
    [-100.0, -100.0, -0.8,  -0.8,   0.0,  -0.8,  -0.8,  -0.8],  # x64 (2048ms)
    [-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -82.6, -0.8],  # x128 (4096ms)
])

# Color scheme
# Good zone: solid green. Bad zone: continuous gradient amber -> red -> crimson.
_GREEN_FILL = '#4a7c52'   # muted forest green
_GREEN_TEXT = '#ffffff'
_NA_FILL    = '#2d2d2d'   # near-black charcoal (INF / NaN)
_NA_TEXT    = '#ffffff'

# unsaturated: warm amber -> muted red -> dark crimson
_BAD_CMAP = LinearSegmentedColormap.from_list(
    'muted_bad',
    ['#c4883a', '#a84f2a', '#8c2e20', '#6e1a14', '#4a0f0f']
)

def _bad_color(t):
    t = max(0.0, min(1.0, t))
    fill = mcolors.to_hex(_BAD_CMAP(t))
    return fill, '#ffffff'

def cell_color_ppl(val, threshold=1.0):
    if val >= 500:          return _NA_FILL, _NA_TEXT       # N/A
    if val >= 200:          return _bad_color(1.0)           # INF
    if val < threshold:     return _GREEN_FILL, _GREEN_TEXT  # safe
    t = min(1.0, math.log(val / threshold) / math.log(100.0 / threshold))
    return _bad_color(t)

def cell_color_acc(val, threshold=1.0):
    if val <= -500 or val >= 500: return _NA_FILL, _NA_TEXT  # N/A
    if val > -threshold:          return _GREEN_FILL, _GREEN_TEXT  # safe
    t = min(1.0, (-val - threshold) / (100.0 - threshold))
    return _bad_color(t)

def format_val_ppl(val):
    if val >= 500:  return 'NaN'
    if val >= 200:  return 'INF'
    if val >= 10:   return f'{val:.1f}%'
    s = f'{val:.2f}%'
    return s.replace('0.', '.').replace('-0.', '-.')

def format_val_acc(val):
    if val >= 500 or val <= -500: return 'NaN'
    s = f'{val:.2f}%'
    return s.replace('0.', '.').replace('-0.', '-.')

CELL_IN = 0.30  # physical inches per cell (shared constant)

def _draw_heatmap_cells(ax, data, trefw_labels, n_bits_labels,
                        color_fn, format_fn, threshold, mantissa_bits, selected,
                        show_bit_sections=False):
    """Draw cells, annotations, divider onto an existing Axes. Returns (nrows, ncols)."""
    nrows, ncols = data.shape
    cell_in = CELL_IN
    is_ppl  = (color_fn is cell_color_ppl)

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
                continue
            r = trefw_labels.index(tlab)
            c = n_bits_labels.index(nb)
            val = data[r, c]
            _, tc = color_fn(val, threshold)
            lbl = format_fn(val)
            ax.text(
                c * cell_in + cell_in / 2,
                r * cell_in + cell_in * 0.18,
                lbl,
                ha='center', va='center',
                fontsize=9, fontweight='bold', color=tc, zorder=2,
            )
            ax.text(
                c * cell_in + cell_in / 2,
                r * cell_in + cell_in * 0.68,
                u'\u2605',
                ha='center', va='center',
                fontsize=28, color='#FFD700', zorder=10
            )

    if mantissa_bits is not None:
        mb_next = mantissa_bits + 1
        if mantissa_bits in n_bits_labels and mb_next in n_bits_labels:
            idx_m   = n_bits_labels.index(mantissa_bits)
            idx_exp = n_bits_labels.index(mb_next)
            line_x  = (min(idx_m, idx_exp) + 1) * cell_in
            ax.axvline(line_x, color='white', linewidth=1.2, zorder=5)

    ax.set_xlim(0, ncols * cell_in)
    ax.set_ylim(0, nrows * cell_in)
    ax.set_xticks([c * cell_in + cell_in / 2 for c in range(ncols)])
    ax.set_xticklabels([str(nb) for nb in n_bits_labels], color='black', fontsize=12)
    ax.set_yticks([r * cell_in + cell_in / 2 for r in range(nrows)])
    ax.set_yticklabels(trefw_labels, color='black', fontsize=12)
    ax.tick_params(colors='black', length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)

    if show_bit_sections and mantissa_bits is not None:
        total_bits_val = max(n_bits_labels)   # works for both high→low and low→high axes
        sign_cols = [c for c, n in enumerate(n_bits_labels) if n == total_bits_val]
        exp_cols  = [c for c, n in enumerate(n_bits_labels)
                     if mantissa_bits < n < total_bits_val]
        mant_cols = [c for c, n in enumerate(n_bits_labels) if n <= mantissa_bits]
        sections = []
        if sign_cols:
            sections.append(('Sign', sign_cols[0], sign_cols[-1]))
        if exp_cols:
            sections.append(('Exponent', exp_cols[0], exp_cols[-1]))
        if mant_cols:
            sections.append(('Mantissa', mant_cols[0], mant_cols[-1]))
        # Some mantissa bits may be off the edge if min shown n > 1
        mant_n_vals    = [n for n in n_bits_labels if n <= mantissa_bits]
        min_mant_n     = min(mant_n_vals) if mant_n_vals else 1
        mant_truncated = min_mant_n > 1
        # mant_left: mantissa is on the left side (reversed / low→high axis)
        mant_left = bool(sign_cols and mant_cols and sign_cols[0] > mant_cols[-1])
        y_arr  = nrows * cell_in + 0.04
        y_text = nrows * cell_in + 0.10
        for (sec_label, c0, c1) in sections:
            x0 = c0 * cell_in + 0.02
            x1 = (c1 + 1) * cell_in - 0.02
            xmid = (x0 + x1) / 2
            if sec_label == 'Mantissa' and mant_truncated:
                if mant_left:
                    # mant on left, truncated bits further left → arrowhead at x0
                    ax.annotate('', xy=(x0, y_arr), xytext=(x1, y_arr),
                                arrowprops=dict(arrowstyle='<-', color='#444444', lw=0.9),
                                clip_on=False, annotation_clip=False)
                else:
                    # mant on right, truncated bits further right → arrowhead at x1
                    ax.annotate('', xy=(x1, y_arr), xytext=(x0, y_arr),
                                arrowprops=dict(arrowstyle='<-', color='#444444', lw=0.9),
                                clip_on=False, annotation_clip=False)
            else:
                ax.annotate('', xy=(x1, y_arr), xytext=(x0, y_arr),
                            arrowprops=dict(arrowstyle='<->', color='#444444', lw=0.9),
                            clip_on=False, annotation_clip=False)
            ax.text(xmid, y_text, sec_label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#222222', clip_on=False)

    return nrows, ncols


def _make_colorbar_sm(threshold, is_ppl, show_nan_band=True,
                      green_frac=0.125, nan_frac=0.15):
    """Build a ScalarMappable using fixed visual fractions (0–1 norm).
    green_frac  – fraction of bar height dedicated to the safe (green) zone
    nan_frac    – fraction dedicated to NaN/INF band (only when show_nan_band)
    The remainder goes to the bad (amber→crimson) gradient.
    Tick positions are returned in the same 0–1 coordinate space.
    """
    if show_nan_band:
        bad_frac = 1.0 - green_frac - nan_frac
        cmap_colors = [
            (0.0,                          _GREEN_FILL),
            (green_frac,                   _GREEN_FILL),
            (green_frac,                   '#c4883a'),
            (green_frac + bad_frac,        '#4a0f0f'),
            (green_frac + bad_frac,        _NA_FILL),
            (1.0,                          _NA_FILL),
        ]
        cb_ticks = [green_frac,
                    green_frac + bad_frac,
                    green_frac + bad_frac + nan_frac / 2]
        if is_ppl:
            cb_label = 'PPL increase (%)'
            cb_tlbls = [f'{int(threshold)}%', '100%', 'NaN/INF']
        else:
            cb_label = 'Top-5 accuracy loss (%)'
            cb_tlbls = [f'\u2212{int(threshold)}%', '\u2212100%', 'NaN/INF']
    else:
        bad_frac = 1.0 - green_frac
        cmap_colors = [
            (0.0,         _GREEN_FILL),
            (green_frac,  _GREEN_FILL),
            (green_frac,  '#c4883a'),
            (1.0,         '#4a0f0f'),
        ]
        cb_ticks = [green_frac, 1.0]
        if is_ppl:
            cb_label = 'PPL increase (%)'
            cb_tlbls = [f'{int(threshold)}%', '100%']
        else:
            cb_label = 'Top-5 accuracy loss (%)'
            cb_tlbls = [f'\u2212{int(threshold)}%', '\u2212100%']
    cmap = LinearSegmentedColormap.from_list('_cb', cmap_colors)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm, cb_ticks, cb_tlbls, cb_label


def plot_heatmap(data, trefw_labels, n_bits_labels, title,
                 color_fn, format_fn, threshold, filename, selected=None,
                 mantissa_bits=None):
    """Standalone single-panel heatmap saved to its own file."""
    nrows, ncols = data.shape
    cell_in   = CELL_IN
    left_in   = 0.85
    top_in    = 0.10
    bot_in    = 0.65
    cbar_w    = 0.12
    cbar_gap  = 0.08
    cbar_rpad = 0.55

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

    _draw_heatmap_cells(ax, data, trefw_labels, n_bits_labels,
                        color_fn, format_fn, threshold, mantissa_bits, selected)
    ax.set_xlabel('# bits with reduced refresh (from LSB)', color='black', fontsize=13, fontweight='bold', labelpad=4)
    ax.set_ylabel('tREFW multiplier', color='black', fontsize=13, fontweight='bold', labelpad=4)

    is_ppl = (color_fn is cell_color_ppl)
    sm, cb_ticks, cb_tlbls, cb_label = _make_colorbar_sm(threshold, is_ppl)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_tlbls)
    cbar.ax.tick_params(labelsize=11, length=2)
    cbar.outline.set_visible(False)
    cbar.set_label(cb_label, fontsize=11, labelpad=3)

    plt.savefig(filename, bbox_inches='tight', transparent=False)
    plt.close()
    print(f'Saved {filename}')

# Generate
plot_heatmap(
    data=Llama_fp16, trefw_labels=trefw_labels, n_bits_labels=n_bits_labels,
    title='Llama FP16',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_Llama_fp16.pdf'),
    selected=[('x128', 10)],
    mantissa_bits=10,
)

plot_heatmap(
    data=opt_fp16, trefw_labels=trefw_labels, n_bits_labels=n_bits_labels,
    title='OPT FP16',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_opt_fp16.pdf'),
    selected=[('x128', 10)],
    mantissa_bits=10,
)

plot_heatmap(
    data=resnet_fp16, trefw_labels=resnet_trefw_labels, n_bits_labels=n_bits_labels,
    title='Resnet FP16',
    color_fn=cell_color_acc, format_fn=format_val_acc,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_Resnet_fp16.pdf'),
    selected=[('x128', 10)],
    mantissa_bits=10,
)

plot_heatmap(
    data=Llama_bf16, trefw_labels=bf16_trefw_labels, n_bits_labels=n_bits_labels,
    title='Llama BF16 (E8M7)',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_Llama_bf16.pdf'),
    selected=[('x128', 7)],
    mantissa_bits=7,
)

plot_heatmap(
    data=opt_bf16, trefw_labels=bf16_trefw_labels, n_bits_labels=n_bits_labels,
    title='OPT BF16 (E8M7)',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_opt_bf16.pdf'),
    selected=[('x128', 7)],
    mantissa_bits=7,
)

plot_heatmap(
    data=resnet_bf16, trefw_labels=resnet_bf16_trefw_labels, n_bits_labels=n_bits_labels,
    title='Resnet BF16 (E8M7)',
    color_fn=cell_color_acc, format_fn=format_val_acc,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_Resnet_bf16.pdf'),
    selected=[('x128', 7)],
    mantissa_bits=7,
)

plot_heatmap(
    data=Llama_fp8, trefw_labels=fp8_trefw_labels, n_bits_labels=fp8_n_bits_labels,
    title='Llama FP8',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_Llama_fp8.pdf'),
    selected=[('x128', 3)],
    mantissa_bits=3,
)

plot_heatmap(
    data=opt_fp8, trefw_labels=fp8_trefw_labels, n_bits_labels=fp8_n_bits_labels,
    title='OPT FP8',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_opt_fp8.pdf'),
    selected=[('x128', 3)],
    mantissa_bits=3,
)

plot_heatmap(
    data=resnet_fp8, trefw_labels=fp8_trefw_labels, n_bits_labels=fp8_n_bits_labels,
    title='Resnet FP8',
    color_fn=cell_color_acc, format_fn=format_val_acc,
    threshold=1.0,
    filename=os.path.join(OUT, 'heatmap_Resnet_fp8.pdf'),
    selected=[('x64', 3)],
    mantissa_bits=3,
)

print('Individual heatmaps done.')

# ── Shared layout constants ───────────────────────────────────────────────────
fp16_w = len(n_bits_labels)     * CELL_IN   # 10 cols
fp8_w  = len(fp8_n_bits_labels) * CELL_IN   #  8 cols
col_widths = [fp16_w, fp16_w, fp8_w]

left_m    = 0.85   # y-labels
hgap      = 0.55   # gap between panels horizontally
vgap      = 0.70   # gap between rows
bot_m     = 0.65   # bottom margin
top_m     = 0.35   # top margin (col titles)
cbar_gap  = 0.10
cbar_w    = 0.12
cbar_rpad = 0.55

total_cells_w = sum(col_widths) + hgap * 2

col_x = [left_m,
         left_m + col_widths[0] + hgap,
         left_m + col_widths[0] + hgap + col_widths[1] + hgap]
col_titles = ['FP16 (E5M10)', 'BF16 (E8M7)', 'FP8 (E4M3)']


def _add_cbar(fig, fig_w, fig_h, x0, y0, h, threshold, is_ppl, show_nan_band=True):
    cbar_ax = fig.add_axes([x0 / fig_w, y0 / fig_h, cbar_w / fig_w, h / fig_h])
    sm, cb_ticks, cb_tlbls, cb_label = _make_colorbar_sm(threshold, is_ppl, show_nan_band)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_tlbls)
    cbar.ax.tick_params(labelsize=11, length=2)
    cbar.outline.set_visible(False)
    cbar.set_label(cb_label, fontsize=11, labelpad=3)


def _build_figure(row_specs, outfile):
    """
    row_specs: list of dicts, one per row (top to bottom):
        panels  – list of (col_idx, data, trefw_lbls, col_lbls, color_fn, format_fn, thr, mb, sel)
        height  – cell-area height in inches
        title   – row label string
        is_ppl  – bool for colorbar type
        bottom_labels – bool, show x-axis labels on this row
    """
    row_heights = [rs['height'] for rs in row_specs]
    n_rows = len(row_specs)
    fig_w = left_m + total_cells_w + cbar_gap + cbar_w + cbar_rpad
    fig_h = top_m + sum(row_heights) + vgap * (n_rows - 1) + bot_m

    # y origins (from bottom) for each row, top row first
    row_y = []
    y = bot_m
    for h in reversed(row_heights):
        row_y.insert(0, y)
        y += h + vgap

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    for ri, rs in enumerate(row_specs):
        for (pc, data, trefw_lbls, col_lbls, color_fn, format_fn, thr, mb, sel) in rs['panels']:
            nrows, ncols = data.shape
            w  = ncols * CELL_IN
            h  = nrows * CELL_IN
            ax = fig.add_axes([col_x[pc] / fig_w, row_y[ri] / fig_h,
                               w / fig_w, h / fig_h])
            ax.set_facecolor('white')
            _draw_heatmap_cells(ax, data, trefw_lbls, col_lbls,
                                color_fn, format_fn, thr, mb, sel,
                                show_bit_sections=True)
            if rs['bottom_labels']:
                ax.set_xlabel('# bits (from LSB)', color='black', fontsize=13, fontweight='bold', labelpad=3)
            else:
                ax.set_xticklabels([])
            if pc == 0:
                ax.set_ylabel('tREFW multiplier', color='black', fontsize=13, fontweight='bold', labelpad=3)
            else:
                ax.set_yticklabels([])

        # per-row colorbar
        _add_cbar(fig, fig_w, fig_h, left_m + total_cells_w + cbar_gap, row_y[ri], rs['height'], 1.0, rs['is_ppl'], rs.get('show_nan_band', True))

        # row title: single-row figures get a top title; multi-row get rotated left label
        if n_rows == 1:
            x_center = (left_m + total_cells_w / 2) / fig_w
            y_top    = (row_y[0] + rs['height'] + 0.68) / fig_h
            fig.text(x_center, y_top, rs['title'],
                     ha='center', va='bottom', fontsize=16, fontweight='bold', color='black')
        else:
            y_center = (row_y[ri] + rs['height'] / 2) / fig_h
            fig.text((left_m - 0.10) / fig_w, y_center, rs['title'],
                     ha='right', va='center', fontsize=14, fontweight='bold',
                     color='black', rotation=90)

    # column titles above the top row (leave room for bit-section arrows ~0.22in)
    col_title_y = row_y[0] + row_heights[0] + 0.27
    for pc, title in enumerate(col_titles):
        x_center = (col_x[pc] + col_widths[pc] / 2) / fig_w
        y_top    = col_title_y / fig_h
        fig.text(x_center, y_top, title,
                 ha='center', va='bottom', fontsize=15, fontweight='bold', color='black')

    plt.savefig(outfile, bbox_inches='tight', transparent=False)
    plt.close()
    print(f'Saved {outfile}')


# ── Figure 1: Llama + OPT — all 6 panels horizontal ────────────────────────
# Layout: [Llama: FP16 | BF16 | FP8 | cbar]  [OPT: FP16 | BF16 | FP8 | cbar]

ppl_lm        = 0.85   # y-label space before each group
ppl_hgap      = hgap
ppl_model_gap = 0.85   # space between Llama group and OPT group
ppl_cbar_gap  = cbar_gap
ppl_cbar_w    = cbar_w
ppl_cbar_rpad = cbar_rpad
ppl_top_m     = 0.50   # room for col titles + group labels
ppl_bot_m     = bot_m

ppl_fp16_w = len(n_bits_labels)     * CELL_IN   # 10 cols
ppl_fp8_w  = len(fp8_n_bits_labels) * CELL_IN   #  8 cols
ppl_fp16_h = len(trefw_labels)      * CELL_IN   # 5 rows (tallest)
ppl_fp8_h  = len(fp8_trefw_labels)  * CELL_IN   # 4 rows
ppl_row_h  = ppl_fp16_h

ppl_group_w = ppl_fp16_w + ppl_hgap + ppl_fp16_w + ppl_hgap + ppl_fp8_w

# x origins of each panel — no cbar between groups, just a gap
llx = [ppl_lm,
        ppl_lm + ppl_fp16_w + ppl_hgap,
        ppl_lm + ppl_fp16_w + ppl_hgap + ppl_fp16_w + ppl_hgap]

opt_lm = ppl_lm + ppl_group_w + ppl_model_gap
olx = [opt_lm,
        opt_lm + ppl_fp16_w + ppl_hgap,
        opt_lm + ppl_fp16_w + ppl_hgap + ppl_fp16_w + ppl_hgap]

# Single colorbar at far right after OPT group
ppl_cbar_x = opt_lm + ppl_group_w + ppl_cbar_gap
ppl_fig_w  = ppl_cbar_x + ppl_cbar_w + ppl_cbar_rpad
ppl_fig_h  = ppl_top_m + ppl_row_h + ppl_bot_m

fig_ppl = plt.figure(figsize=(ppl_fig_w, ppl_fig_h))
fig_ppl.patch.set_facecolor('white')

ppl_panels = [
    # (group_x_list, w, h, data, trefw_lbls, col_lbls, mb, sel, show_yticks)
    (llx, 0, ppl_fp16_w, ppl_fp16_h, Llama_fp16, trefw_labels,      n_bits_labels,     10, [('x128',  10)], True),
    (llx, 1, ppl_fp16_w, ppl_fp16_h, Llama_bf16, bf16_trefw_labels,  n_bits_labels,      7, [('x128',   7)], False),
    (llx, 2, ppl_fp8_w,  ppl_fp8_h,  Llama_fp8,  fp8_trefw_labels,   fp8_n_bits_labels,  3, [('x128',   3)], False),
    (olx, 0, ppl_fp16_w, ppl_fp16_h, opt_fp16,   trefw_labels,      n_bits_labels,     10, [('x128', 10)], False),
    (olx, 1, ppl_fp16_w, ppl_fp16_h, opt_bf16,   bf16_trefw_labels,  n_bits_labels,      7, [('x128',  7)], False),
    (olx, 2, ppl_fp8_w,  ppl_fp8_h,  opt_fp8,    fp8_trefw_labels,   fp8_n_bits_labels,  3, [('x128',   3)], False),
]

for (gx, ci, pw, ph, data, trefw_lbls, col_lbls, mb, sel, show_y) in ppl_panels:
    ax = fig_ppl.add_axes([gx[ci] / ppl_fig_w, ppl_bot_m / ppl_fig_h,
                           pw / ppl_fig_w, ph / ppl_fig_h])
    ax.set_facecolor('white')
    _draw_heatmap_cells(ax, data, trefw_lbls, col_lbls,
                        cell_color_ppl, format_val_ppl, 1.0, mb, sel,
                        show_bit_sections=True)
    ax.set_xlabel('# bits (from LSB)', color='black', fontsize=13, fontweight='bold', labelpad=3)
    if show_y:
        ax.set_ylabel('tREFW multiplier', color='black', fontsize=13, fontweight='bold', labelpad=3)
    else:
        ax.set_yticklabels([])

# Single colorbar on the far right
sm_p, tks, tlbls, lbl = _make_colorbar_sm(1.0, True)
cb_ax = fig_ppl.add_axes([ppl_cbar_x / ppl_fig_w, ppl_bot_m / ppl_fig_h,
                           ppl_cbar_w / ppl_fig_w, ppl_row_h / ppl_fig_h])
cb = fig_ppl.colorbar(sm_p, cax=cb_ax)
cb.set_ticks(tks); cb.set_ticklabels(tlbls)
cb.ax.tick_params(labelsize=11, length=2)
cb.outline.set_visible(False)
cb.set_label(lbl, fontsize=11, labelpad=3)

# Column titles and group labels
for gx, group_name in [(llx, 'Llama'), (olx, 'OPT')]:
    for ci, (ctitle, cw) in enumerate(zip(col_titles, [ppl_fp16_w, ppl_fp16_w, ppl_fp8_w])):
        xc = (gx[ci] + cw / 2) / ppl_fig_w
        fig_ppl.text(xc, (ppl_bot_m + ppl_row_h + 0.27) / ppl_fig_h,
                     ctitle, ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
    # group label centered over the 3 panels
    gxc = (gx[0] + ppl_group_w / 2) / ppl_fig_w
    fig_ppl.text(gxc, (ppl_bot_m + ppl_row_h + 0.55) / ppl_fig_h,
                 group_name, ha='center', va='bottom', fontsize=16, fontweight='bold', color='black')

out_ppl = os.path.join(OUT, 'accuracy_heatmap_ppl.pdf')
plt.savefig(out_ppl, bbox_inches='tight', transparent=False)
plt.close()
print(f'Saved {out_ppl}')

# ── Figure 2: Resnet (1 row) ───────────────────────────────────────────────
resnet_h = len(resnet_trefw_labels) * CELL_IN   # 4 rows

_build_figure(
    row_specs=[
        dict(
            panels=[
                (0, resnet_fp16, resnet_trefw_labels,       n_bits_labels,     cell_color_acc, format_val_acc, 1.0, 10, [('x128', 10)]),
                (1, resnet_bf16, resnet_bf16_trefw_labels,  n_bits_labels,     cell_color_acc, format_val_acc, 1.0,  7, [('x128',  7)]),
                (2, resnet_fp8,  fp8_trefw_labels,          fp8_n_bits_labels, cell_color_acc, format_val_acc, 1.0,  3, [('x64',  3)]),
            ],
            height=resnet_h, title='Resnet', is_ppl=False, bottom_labels=True, show_nan_band=False,
        ),
    ],
    outfile=os.path.join(OUT, 'accuracy_heatmap_acc.pdf'),
)

print('All done.')
