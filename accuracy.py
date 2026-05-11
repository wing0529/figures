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

trefw_labels = ['x8', 'x16', 'x32', 'x64', 'x128']  # x100 removed
n_bits_labels = list(range(7, 17))

llama_fp16 = np.array([
    [0.01,  0.01,  0.01,  0.02,  0.29, 0.38, 0.18, 0.14, NAN,  NAN ],  # x8
    [0.01,  0.01,  0.01,  0.02,  0.29, 0.38, 0.18, 0.14, NAN,  NAN ],  # x16
    [0.01,  0.01,  0.11,  0.45,  INF,  7.47, 9.11, 6.59, NAN,  NAN ],  # x32
    [0.02,  0.03,  0.30,  5.73,  INF,  5.7,  7.3,  INF,  NAN,  NAN ],  # x64
    [0.02,  0.20,  1.80, 82.40,  INF,  INF,  NAN,  INF,  NAN,  NAN ],  # x128
])

opt_fp16 = np.array([
    [0.01,  0.01,  0.01,  0.01,  0.01, 0.01, 0.06,  0.06, INF,  INF ],  # x8
    [0.01,  0.01,  0.01,  0.01,  0.01, 0.01, 0.06,  0.06, INF,  INF ],  # x16
    [0.01,  0.01,  0.01,  0.01,  0.01, 0.01, 0.06,  0.06, INF,  INF ],  # x32
    [0.01,  0.01,  0.00,  0.01,  0.01, 0.03, 9.6,   INF,  NAN,  NAN ],  # x64
    [0.00, -0.01,  0.01,  0.03,  0.01, 0.34, INF,   INF,  NAN,  NAN ],  # x128
])

resnet_fp16 = np.array([
    [1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  4.92 ],  # x16
    [1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64,  1.64, -100., -100. ],  # x32
    [1.64,  1.64,  1.64,  1.64, -1.64,  4.92, -100.,  1.64, -100., -100. ],  # x64
    [0.00,  0.00,  0.00,  0.00, -2.46,-45.90,-98.36, -100., -100., -100. ],  # x128
])
resnet_trefw_labels = ['x16', 'x32', 'x64', 'x128']  # x100 removed

# BF16 dummy data (copied from FP16)
llama_bf16               = llama_fp16.copy()
opt_bf16                 = opt_fp16.copy()
resnet_bf16              = resnet_fp16.copy()
bf16_trefw_labels        = trefw_labels
resnet_bf16_trefw_labels = resnet_trefw_labels

# FP8 E4M3 datasets (% delta from baseline, NaN=999 for not measured)
# x-axis: n lower bits reduced, 8->1 (left to right)
fp8_trefw_labels  = ['x8', 'x32', 'x64', 'x128']  # x100 removed
fp8_n_bits_labels = [8, 7, 6, 5, 4, 3, 2, 1]

# LLaMA FP8 PPL % delta  (baseline PPL = 13.042068)
llama_fp8 = np.array([
    [NAN, NAN, NAN, 0.00, 0.00, 0.00, 0.00, 0.00], # x8
    [NAN, NAN, NAN, 0.01, 0.01, 0.01, 0.00, 0.00], # x32
    [NAN, NAN, NAN, NAN, 0.01, 0.00, 0.00, 0.00],  # x64
    [NAN, NAN, NAN, NAN, NAN, NAN, 0.01, 0.01],    # x128
])

# OPT FP8 PPL % delta  (baseline PPL = 14.407216)
opt_fp8 = np.array([
    [NAN, NAN, NAN, 0.00, 0.00, 0.00, 0.00, 0.00], # x8
    [NAN, NAN, NAN, NAN, 0.00, 0.00, 0.00, 0.00],  # x32
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],      # x64
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN],      # x128
])

# ResNet-50 FP8 Top-5 accuracy % delta  (baseline acc = 0.9375)
resnet_fp8 = np.array([
    [   0.05,   0.05,  0.05,   0.05,  0.05,  0.05,  0.05,  0.05],  # x8
    [   0.80,-100.0, -100.0,   0.05,  0.05,  0.05,  0.05,  0.05],  # x32
    [  -0.80,-100.0, -100.0,   0.05,  0.05,  0.05,  0.05,  0.05],  # x64
    [-100.0, -100.0, -100.0,  -0.80,  0.80,  0.05,  0.05,  0.05],  # x128
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

def cell_color_ppl(val, threshold=3.0):
    if val >= 500:          return _NA_FILL, _NA_TEXT       # N/A
    if val >= 200:          return _bad_color(1.0)           # INF
    if val < threshold:     return _GREEN_FILL, _GREEN_TEXT  # safe
    t = min(1.0, math.log(val / threshold) / math.log(100.0 / threshold))
    return _bad_color(t)

def cell_color_acc(val, threshold=3.0):
    if val <= -500 or val >= 500: return _NA_FILL, _NA_TEXT  # N/A
    if val > -threshold:          return _GREEN_FILL, _GREEN_TEXT  # safe
    t = min(1.0, (-val - threshold) / (100.0 - threshold))
    return _bad_color(t)

def format_val_ppl(val):
    if val >= 500:  return 'NaN'
    if val >= 200:  return 'INF'
    if val >= 10:   return f'+{val:.1f}%'
    return f'{val:+.2f}%'

def format_val_acc(val):
    if val >= 500 or val <= -500: return 'NaN'
    return f'{val:+.2f}%'

CELL_IN = 0.30  # physical inches per cell (shared constant)

def _draw_heatmap_cells(ax, data, trefw_labels, n_bits_labels,
                        color_fn, format_fn, threshold, mantissa_bits, selected):
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
            lbl = format_fn(val)
            ax.text(
                c * cell_in + cell_in / 2,
                r * cell_in + cell_in / 2,
                lbl,
                ha='center', va='center',
                fontsize=5.5, color=tc, zorder=2,
            )

    if selected:
        for (tlab, nb) in selected:
            if tlab not in trefw_labels or nb not in n_bits_labels:
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
    ax.set_xticklabels([str(nb) for nb in n_bits_labels], color='black', fontsize=8)
    ax.set_yticks([r * cell_in + cell_in / 2 for r in range(nrows)])
    ax.set_yticklabels(trefw_labels, color='black', fontsize=8)
    ax.tick_params(colors='black', length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)
    return nrows, ncols


def _make_colorbar_sm(threshold, is_ppl):
    nan_band = 14.0
    vmin_cb, vmax_cb = 0.0, 100.0 + nan_band
    span = vmax_cb - vmin_cb
    tg = threshold / span
    tb = 100.0 / span
    cmap_colors = [
        (0.0, _GREEN_FILL),
        (tg,  _GREEN_FILL),
        (tg,  '#c4883a'),
        (tb,  '#4a0f0f'),
        (tb,  _NA_FILL),
        (1.0, _NA_FILL),
    ]
    cmap = LinearSegmentedColormap.from_list('_cb', cmap_colors)
    norm = mcolors.Normalize(vmin=vmin_cb, vmax=vmax_cb)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb_ticks = [threshold, 100, 100 + nan_band / 2]
    if is_ppl:
        cb_label = 'PPL increase (%)'
        cb_tlbls = [str(int(threshold)), '100', 'NaN/INF']
    else:
        cb_label = 'Top-5 accuracy loss (%)'
        cb_tlbls = [f'\u2212{int(threshold)}', '\u2212100', 'NaN/INF']
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
    ax.set_xlabel('# bits with reduced refresh (from LSB)', color='black', fontsize=9, labelpad=4)
    ax.set_ylabel('tREFW multiplier', color='black', fontsize=9, labelpad=4)

    is_ppl = (color_fn is cell_color_ppl)
    sm, cb_ticks, cb_tlbls, cb_label = _make_colorbar_sm(threshold, is_ppl)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_tlbls)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)
    cbar.set_label(cb_label, fontsize=7.5, labelpad=3)

    plt.savefig(filename, bbox_inches='tight', transparent=False)
    plt.close()
    print(f'Saved {filename}')

# Generate
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
    data=llama_bf16, trefw_labels=bf16_trefw_labels, n_bits_labels=n_bits_labels,
    title='LLaMA BF16 (dummy)',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_llama_bf16.pdf'),
    selected=[('x32', 7)],
    mantissa_bits=7,
)

plot_heatmap(
    data=opt_bf16, trefw_labels=bf16_trefw_labels, n_bits_labels=n_bits_labels,
    title='OPT BF16 (dummy)',
    color_fn=cell_color_ppl, format_fn=format_val_ppl,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_opt_bf16.pdf'),
    selected=[('x128', 7)],
    mantissa_bits=7,
)

plot_heatmap(
    data=resnet_bf16, trefw_labels=resnet_bf16_trefw_labels, n_bits_labels=n_bits_labels,
    title='ResNet50 BF16 (dummy)',
    color_fn=cell_color_acc, format_fn=format_val_acc,
    threshold=3.0,
    filename=os.path.join(OUT, 'heatmap_resnet50_bf16.pdf'),
    selected=[('x128', 7)],
    mantissa_bits=7,
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

print('Individual heatmaps done.')

# ── Combined 3x3 figure ───────────────────────────────────────────────────────
# Layout: rows = LLaMA / OPT / ResNet50,  cols = FP16 / BF16 / FP8
# Each panel uses the same styled cells as the individual plots.
# One shared colorbar per row (PPL rows share one style, ResNet row another).

panels = [
    # (row, col, data, trefw_lbls,             col_lbls,          color_fn,        format_fn,      threshold, mantissa_bits, selected)
    (0, 0, llama_fp16,  trefw_labels,             n_bits_labels,     cell_color_ppl, format_val_ppl, 3.0, 10, [('x32',  10)]),
    (0, 1, llama_bf16,  bf16_trefw_labels,         n_bits_labels,     cell_color_ppl, format_val_ppl, 3.0,  7, [('x32',   7)]),
    (0, 2, llama_fp8,   fp8_trefw_labels,          fp8_n_bits_labels, cell_color_ppl, format_val_ppl, 3.0,  3, [('x64',   3)]),
    (1, 0, opt_fp16,    trefw_labels,             n_bits_labels,     cell_color_ppl, format_val_ppl, 3.0, 10, [('x128', 10)]),
    (1, 1, opt_bf16,    bf16_trefw_labels,         n_bits_labels,     cell_color_ppl, format_val_ppl, 3.0,  7, [('x128',  7)]),
    (1, 2, opt_fp8,     fp8_trefw_labels,          fp8_n_bits_labels, cell_color_ppl, format_val_ppl, 3.0,  3, [('x32',   3)]),
    (2, 0, resnet_fp16, resnet_trefw_labels,       n_bits_labels,     cell_color_acc, format_val_acc, 3.0, 10, [('x128', 10)]),
    (2, 1, resnet_bf16, resnet_bf16_trefw_labels,  n_bits_labels,     cell_color_acc, format_val_acc, 3.0,  7, [('x128',  7)]),
    (2, 2, resnet_fp8,  fp8_trefw_labels,          fp8_n_bits_labels, cell_color_acc, format_val_acc, 3.0,  3, [('x128',  3)]),
]

# Each panel width/height in inches (cells only)
fp16_w  = len(n_bits_labels)     * CELL_IN   # 10 cols
fp8_w   = len(fp8_n_bits_labels) * CELL_IN   #  8 cols
row0_h  = len(trefw_labels)             * CELL_IN   # 5 rows (LLaMA)
row1_h  = len(trefw_labels)             * CELL_IN   # 5 rows (OPT)
row2_h  = len(resnet_trefw_labels)      * CELL_IN   # 4 rows (ResNet)

col_widths  = [fp16_w, fp16_w, fp8_w]
row_heights = [row0_h, row1_h, row2_h]

# Fixed margins in figure-inches
left_m   = 0.85   # y-labels
hgap     = 0.55   # horizontal gap between panels
vgap     = 0.70   # vertical gap between rows (x-tick + x-label space)
bot_m    = 0.65   # bottom margin (x-label + ticks for last row)
top_m    = 0.35   # top margin (col titles)
cbar_gap = 0.10
cbar_w   = 0.12
cbar_rpad= 0.55

total_cells_w = sum(col_widths) + hgap * 2
fig_w = left_m + total_cells_w + cbar_gap + cbar_w + cbar_rpad
fig_h = top_m + sum(row_heights) + vgap * 2 + bot_m

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('white')

# Compute panel origins (bottom-left in figure-inches, y measured from bottom)
col_x = [left_m,
          left_m + col_widths[0] + hgap,
          left_m + col_widths[0] + hgap + col_widths[1] + hgap]
# rows from top to bottom: row0 is at top
row_y_from_bottom = [
    bot_m + row_heights[2] + vgap + row_heights[1] + vgap,  # row 0
    bot_m + row_heights[2] + vgap,                           # row 1
    bot_m,                                                   # row 2
]

col_titles = ['FP16', 'BF16 (dummy)', 'FP8 E4M3']
row_titles = ['LLaMA', 'OPT', 'ResNet50']

for (pr, pc, data, trefw_lbls, col_lbls, color_fn, format_fn, thr, mb, sel) in panels:
    nrows, ncols = data.shape
    w = ncols * CELL_IN
    h = nrows * CELL_IN
    x0 = col_x[pc]
    y0 = row_y_from_bottom[pr]

    ax = fig.add_axes([x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h])
    ax.set_facecolor('white')
    _draw_heatmap_cells(ax, data, trefw_lbls, col_lbls,
                        color_fn, format_fn, thr, mb, sel)

    # x-axis label only on bottom row
    if pr == 2:
        ax.set_xlabel('# bits (from LSB)', color='black', fontsize=8, labelpad=3)
    else:
        ax.set_xticklabels([])

    # y-axis label only on left column
    if pc == 0:
        ax.set_ylabel('tREFW', color='black', fontsize=8, labelpad=3)
    else:
        ax.set_yticklabels([])

# Column titles (above top row)
for pc, title in enumerate(col_titles):
    x_center = (col_x[pc] + col_widths[pc] / 2) / fig_w
    y_top    = (row_y_from_bottom[0] + row_heights[0] + 0.08) / fig_h
    fig.text(x_center, y_top, title,
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Row titles (left of left column)
for pr, title in enumerate(row_titles):
    y_center = (row_y_from_bottom[pr] + row_heights[pr] / 2) / fig_h
    fig.text((left_m - 0.10) / fig_w, y_center, title,
             ha='right', va='center', fontsize=9, fontweight='bold', color='black',
             rotation=90)

# Two colorbars: one for PPL rows (rows 0-1), one for acc row (row 2)
def _add_cbar(fig, fig_w, fig_h, y0, h, threshold, is_ppl):
    cax_x = (left_m + total_cells_w + cbar_gap) / fig_w
    cax_y = y0 / fig_h
    cax_w = cbar_w / fig_w
    cax_h = h / fig_h
    cbar_ax = fig.add_axes([cax_x, cax_y, cax_w, cax_h])
    sm, cb_ticks, cb_tlbls, cb_label = _make_colorbar_sm(threshold, is_ppl)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_tlbls)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)
    cbar.set_label(cb_label, fontsize=7.5, labelpad=3)

# PPL colorbar spans rows 0+1
ppl_y0 = row_y_from_bottom[1]
ppl_h  = row_heights[0] + vgap + row_heights[1]
_add_cbar(fig, fig_w, fig_h, ppl_y0, ppl_h, 3.0, is_ppl=True)

# Acc colorbar spans row 2
_add_cbar(fig, fig_w, fig_h, row_y_from_bottom[2], row_heights[2], 3.0, is_ppl=False)

out_combined = os.path.join(OUT, 'accuracy_heatmap.pdf')
plt.savefig(out_combined, bbox_inches='tight', transparent=False)
plt.close()
print(f'Saved {out_combined}')
print('All done.')
