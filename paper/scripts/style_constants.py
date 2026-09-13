"""Shared style constants for all benchmark figure scripts.

Import this at the top of every plot_*.py script to ensure consistency.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global rcParams (IEEE two-column: columnwidth=3.5in, textwidth=7.16in) ──
RCPARAMS = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

def apply_style():
    plt.rcParams.update(RCPARAMS)

# ── Architecture colors (used in line plots, grouped bars) ──
ARCH_COLORS = {
    "Proposed-1": "#10B981",     # emerald
    "Proposed-2": "#3B82F6",     # royal blue
    "Azure-Lily": "#F97316",     # coral/orange
}

ARCH_MARKERS = {
    "Proposed-1": "o",
    "Proposed-2": "s",
    "Azure-Lily": "D",
}

ARCH_LINESTYLES = {
    "Proposed-1": "-",
    "Proposed-2": "--",
    "Azure-Lily": ":",
}

# Azure-Lily baseline line style
BASELINE_COLOR = "#94A3B8"
BASELINE_LW = 1.0
BASELINE_LS = ":"
BASELINE_ALPHA = 0.5

# ── Breakdown colors (bert_analysis panel b) ──
BREAKDOWN_COLORS = {
    "DIMM": "#6366F1",        # vivid indigo
    "Proj_FFN": "#F59E0B",    # bright amber
    "Other": "#CBD5E1",       # soft gray
}

# ── Dual-identity colors (dual_identity figure) ──
DUAL_COLORS = {
    "dual": "#8B5CF6",        # vivid violet
    "regular": "#EC4899",     # hot pink
}

# ── Layer colors (used in right panels / breakdown charts) ──
# CNN layers
CNN_LAYER_COLORS = {
    "Conv": "#7C3AED",          # purple
    "FC": "#A78BFA",            # light purple
    "Activation": "#F59E0B",    # amber
    "MaxPool": "#EC4899",       # pink
    "Residual": "#06B6D4",      # cyan
    "Other": "#94A3B8",         # gray
}

# BERT layers
BERT_LAYER_COLORS = {
    "Q/K/V Proj": "#7C3AED",       # purple
    "O Proj": "#A78BFA",            # light purple
    "FFN": "#C4B5FD",               # lighter purple
    "QK\u1d40 DIMM": "#F59E0B",    # amber
    "Softmax": "#FBBF24",           # yellow
    "S\u00d7V DIMM": "#F97316",    # orange
    "LayerNorm": "#EC4899",         # pink
    "Residual": "#06B6D4",          # cyan
    "Embed+LN": "#94A3B8",          # gray
}

# ── Annotation style ──
ANNOT_FONTSIZE = 6
ANNOT_FONTWEIGHT = "bold"

# ── Single-column annotation style (for figures rendered at columnwidth) ──
ANNOT_FONTSIZE_SC = 7
ANNOT_FONTWEIGHT_SC = "bold"

# ── Figure sizes (IEEE two-column) ──
FIG_SINGLE = (3.5, 2.4)           # single column width
FIG_DUAL = (7.16, 2.8)            # double column (textwidth)
FIG_DUAL_RATIO = {'width_ratios': [1, 1.2]}  # left:right ratio

# ── Single-column rcParams (for dual-panel figures rendered at columnwidth) ──
RCPARAMS_SC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

def apply_style_sc():
    """Apply single-column style (larger fonts for figures shown at columnwidth)."""
    plt.rcParams.update(RCPARAMS_SC)

# ── FPGA constants ──
FPGA_GRID = 150
CLB_TILE_UM2 = 2239
FPGA_AREA_MM2 = FPGA_GRID * FPGA_GRID * CLB_TILE_UM2 / 1e6
