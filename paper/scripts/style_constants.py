"""Shared style constants for all benchmark figure scripts.

Import this at the top of every plot_*.py script to ensure consistency.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global rcParams ──
RCPARAMS = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

def apply_style():
    plt.rcParams.update(RCPARAMS)

# ── Architecture colors (used in line plots, grouped bars) ──
ARCH_COLORS = {
    "NL-DPE Proposed": "#10B981",     # emerald
    "NL-DPE AL-Matched": "#3B82F6",   # royal blue
    "Azure-Lily": "#F97316",          # coral/orange
}

ARCH_MARKERS = {
    "NL-DPE Proposed": "o",
    "NL-DPE AL-Matched": "s",
    "Azure-Lily": "D",
}

ARCH_LINESTYLES = {
    "NL-DPE Proposed": "-",
    "NL-DPE AL-Matched": "--",
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
ANNOT_FONTSIZE = 7
ANNOT_FONTWEIGHT = "bold"

# ── Figure sizes ──
FIG_SINGLE = (5.5, 3.8)           # single panel
FIG_DUAL = (10, 4.2)              # two panels side by side
FIG_DUAL_RATIO = {'width_ratios': [1, 1.2]}  # left:right ratio

# ── FPGA constants ──
FPGA_GRID = 150
CLB_TILE_UM2 = 2239
FPGA_AREA_MM2 = FPGA_GRID * FPGA_GRID * CLB_TILE_UM2 / 1e6
