#!/usr/bin/env python3
"""3-panel figure: BERT-Tiny energy convergence story (seq_len sweep).

Left:   % breakdown bars (DIMM vs Proj+FFN vs Other) — both archs, across N
Middle: DIMM-internal split (DPE analog vs Digital DSP+CLB) — both archs, across N
Right:  Line plot: overall energy ratio + DIMM-only ratio vs N

Input: benchmarks/results/bert_tiny_seqlen_fixed_fmax.csv
Output: paper/figures/benchmarks/bert_seqlen_energy_3panel.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from style_constants import (apply_style, ARCH_COLORS, BREAKDOWN_COLORS,
                              BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE, ANNOT_FONTWEIGHT)
apply_style()

CSV_PATH = ROOT / "benchmarks" / "results" / "bert_tiny_seqlen_fixed_fmax.csv"
OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "bert_seqlen_energy_3panel.pdf"

# ── Load data ────────────────────────────────────────────────────────────
raw = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        raw[key] = {k: float(v) for k, v in row.items()
                    if k not in ("arch", "seq_len") and v.replace('.','',1).replace('-','',1).isdigit()}

SEQ_LENS = [128, 256, 512, 1024, 2048]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

def _get(arch, N, key):
    return raw[(arch, N)][key]

# ── Derived: per-operation % ─────────────────────────────────────────────
ALL_ARCHS = ("proposed", "al_like", "azurelily")
pct = {}
for arch in ALL_ARCHS:
    for cat in ("dimm", "proj_ffn", "other"):
        vals = []
        for N in SEQ_LENS:
            total = _get(arch, N, "energy_pj")
            if cat == "dimm":
                vals.append(_get(arch, N, "energy_dimm_pj") / total * 100)
            elif cat == "proj_ffn":
                vals.append(_get(arch, N, "energy_proj_ffn_pj") / total * 100)
            else:
                vals.append(_get(arch, N, "energy_other_pj") / total * 100)
        pct[(arch, cat)] = vals

# DIMM-internal: DPE (analog) vs Digital (DSP+CLB)
dimm_dpe_pct = {}
for arch in ALL_ARCHS:
    dimm_dpe_pct[arch] = []
    for N in SEQ_LENS:
        dimm_total = _get(arch, N, "energy_dimm_pj")
        dpe = _get(arch, N, "energy_dpe_pj")
        dimm_dpe = min(dpe, dimm_total)
        dimm_dpe_pct[arch].append(dimm_dpe / dimm_total * 100 if dimm_total > 0 else 0)

# Energy ratios: Azure-Lily / Proposed
energy_ratio_p1 = [_get("azurelily", N, "energy_pj") / _get("proposed", N, "energy_pj")
                   for N in SEQ_LENS]
energy_ratio_p2 = [_get("azurelily", N, "energy_pj") / _get("al_like", N, "energy_pj")
                   for N in SEQ_LENS]
dimm_ratio_p1 = [_get("azurelily", N, "energy_dimm_pj") / _get("proposed", N, "energy_dimm_pj")
                 for N in SEQ_LENS]
dimm_ratio_p2 = [_get("azurelily", N, "energy_dimm_pj") / _get("al_like", N, "energy_dimm_pj")
                 for N in SEQ_LENS]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.6),
                                     gridspec_kw={'width_ratios': [1, 1, 0.9]})
fig.subplots_adjust(wspace=0.45, top=0.78, bottom=0.18)

# ── Panel left: DIMM vs Proj+FFN vs Other stacked bars ──────────────────
bar_w = 0.25
arch_list = ["proposed", "al_like", "azurelily"]
arch_hatch = {"proposed": None, "al_like": "...", "azurelily": "///"}
arch_display = {"proposed": "P1", "al_like": "P2", "azurelily": "AL"}
cat_colors = ["#7EC8E3", "#FFB3BA", "#D5D5D5"]  # light blue, pink, light gray
cat_labels = ["DIMM", "Proj+FFN", "Other"]
cat_keys = ["dimm", "proj_ffn", "other"]

for ai, arch in enumerate(arch_list):
    offset = (ai - 1) * bar_w
    hatch = arch_hatch[arch]
    for i in range(n_groups):
        bottom = 0
        for ci, ck in enumerate(cat_keys):
            val = pct[(arch, ck)][i]
            ax1.bar(x[i] + offset, val, bar_w * 0.9, bottom=bottom,
                    color=cat_colors[ci], edgecolor='white', linewidth=0.5,
                    hatch=hatch, zorder=3)
            # Only annotate Proj+FFN for N=128,256,512 (first 3)
            if ck == "proj_ffn" and val > 2 and i < 3:
                ax1.text(x[i] + offset, bottom + val / 2, f'{val:.0f}%',
                         ha='center', va='center', fontsize=3.5,
                         fontweight='bold', color='black', zorder=5)
            bottom += val

ax1.set_xticks(x)
ax1.set_xticklabels([str(N) for N in SEQ_LENS])
ax1.set_xlabel('Sequence Length (N)')
ax1.set_ylabel('BERT-Tiny Energy Breakdown (%)')
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.08, axis='y', zorder=0)

# Left panel legend
a_handles = [
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, label='Proposed-1'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, hatch='...', label='Proposed-2'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, hatch='///', label='Azure-Lily'),
]
c_handles = [mpatches.Patch(facecolor=cat_colors[ci], edgecolor='white',
             label=cat_labels[ci]) for ci in range(3)]
left_handles = a_handles + c_handles
LEGEND_Y = 0.93
bbox_left = (ax1.get_position().x0 + ax1.get_position().width / 2, LEGEND_Y)
fig.legend(handles=left_handles, loc='upper center', ncol=3, fontsize=4.5,
           frameon=True, framealpha=0.9, bbox_to_anchor=bbox_left,
           columnspacing=0.5, handletextpad=0.3)

# ── Panel middle: DIMM-internal DPE (analog) vs Digital ──────────────────
DPE_COLOR = "#B5EAD7"     # mint green
FABRIC_COLOR = "#FFDAC1"  # peach/yellow

for ai, arch in enumerate(arch_list):
    offset = (ai - 1) * bar_w
    hatch = arch_hatch[arch]
    dpe = dimm_dpe_pct[arch]
    fab = [100 - d for d in dpe]

    for i in range(n_groups):
        ax2.bar(x[i] + offset, dpe[i], bar_w * 0.9, color=DPE_COLOR,
                edgecolor='white', linewidth=0.5, hatch=hatch, zorder=3)
        ax2.bar(x[i] + offset, fab[i], bar_w * 0.9, bottom=dpe[i],
                color=FABRIC_COLOR, edgecolor='white', linewidth=0.5,
                hatch=hatch, zorder=3)
        if dpe[i] > 30:
            ax2.text(x[i] + offset, dpe[i] / 2, f'{dpe[i]:.0f}%',
                     ha='center', va='center', fontsize=3.5,
                     fontweight='bold', color='black', zorder=5)

ax2.axhline(y=50, color='black', linewidth=1.0, linestyle='--', alpha=0.7, zorder=5)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.set_xticks(x)
ax2.set_xticklabels([str(N) for N in SEQ_LENS])
ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('DIMM Energy Breakdown (%)')
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.08, axis='y', zorder=0)

# Middle panel legend
d_handles = [
    mpatches.Patch(facecolor=DPE_COLOR, edgecolor='white', label='IMC block'),
    mpatches.Patch(facecolor=FABRIC_COLOR, edgecolor='white', label='FPGA fabric'),
]
mid_handles = a_handles + d_handles
bbox_mid = (ax2.get_position().x0 + ax2.get_position().width / 2, LEGEND_Y)
fig.legend(handles=mid_handles, loc='upper center', ncol=3, fontsize=4.5,
           frameon=True, framealpha=0.9, bbox_to_anchor=bbox_mid,
           columnspacing=0.5, handletextpad=0.3)

# ── Panel right: Energy ratio lines (AL / P1 and AL / P2) ───────────────
COLOR_P1 = '#2E86AB'
COLOR_P2 = '#E05263'

ax3.plot(x, energy_ratio_p1, color=COLOR_P1, linewidth=2,
         marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.8,
         label='Overall', zorder=4)
ax3.plot(x, dimm_ratio_p1, color=COLOR_P1, linewidth=1.5, linestyle='--',
         marker='s', markersize=3, markeredgecolor='white', markeredgewidth=0.8,
         label='DIMM only', zorder=3)
ax3.plot(x, energy_ratio_p2, color=COLOR_P2, linewidth=2,
         marker='^', markersize=5, markeredgecolor='white', markeredgewidth=0.8,
         zorder=4)
ax3.plot(x, dimm_ratio_p2, color=COLOR_P2, linewidth=1.5, linestyle='--',
         marker='D', markersize=3, markeredgecolor='white', markeredgewidth=0.8,
         zorder=3)

# Asymptotic lines + annotations
ax3.axhline(y=1.4, color='#555', linewidth=0.8, linestyle=':', alpha=0.5)
ax3.axhline(y=1.6, color='#555', linewidth=0.8, linestyle=':', alpha=0.5)
ax3.text(0.3, 1.4, '1.4\u00d7', fontsize=7, color='#555', fontweight='bold', fontstyle='italic', va='bottom')
ax3.text(1.0, 1.6, '1.6\u00d7', fontsize=7, color='#555', fontweight='bold', fontstyle='italic', va='bottom')

ax3.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax3.set_xticks(x)
ax3.set_xticklabels([str(N) for N in SEQ_LENS])
ax3.set_xlabel('Sequence Length (N)')
ax3.set_ylabel('Normalized Energy Consumption')
all_ratios = energy_ratio_p1 + energy_ratio_p2 + dimm_ratio_p1 + dimm_ratio_p2
ymin = min(min(all_ratios), 1.0) * 0.9
ymax = max(all_ratios) * 1.1
ax3.set_ylim(ymin, ymax)

# Right panel legend: 2 columns — arch colors + line styles
import matplotlib.lines as mlines
arch_handles = [
    mlines.Line2D([], [], color=COLOR_P1, linewidth=2, label='Proposed-1'),
    mlines.Line2D([], [], color=COLOR_P2, linewidth=2, label='Proposed-2'),
]
style_handles = [
    mlines.Line2D([], [], color='black', linewidth=1.5, linestyle='-', label='Overall'),
    mlines.Line2D([], [], color='black', linewidth=1.5, linestyle='--', label='DIMM only'),
]
all_handles = arch_handles + style_handles
bbox_right = (ax3.get_position().x0 + ax3.get_position().width / 2, LEGEND_Y)
fig.legend(handles=all_handles, loc='upper center', ncol=2, fontsize=5,
           frameon=True, framealpha=0.9, bbox_to_anchor=bbox_right,
           columnspacing=0.8, handletextpad=0.3)
ax3.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# Print summary
print(f"\n{'N':>5} {'AL/P1':>7} {'AL/P2':>7} {'DIMM/P1':>8} {'DIMM/P2':>8} {'P1 DIMM%':>9} {'P2 DIMM%':>9} {'AL DIMM%':>9}")
print("-" * 65)
for i, N in enumerate(SEQ_LENS):
    print(f"{N:>5} {energy_ratio_p1[i]:>6.2f}\u00d7 {energy_ratio_p2[i]:>6.2f}\u00d7 "
          f"{dimm_ratio_p1[i]:>7.2f}\u00d7 {dimm_ratio_p2[i]:>7.2f}\u00d7 "
          f"{pct[('proposed','dimm')][i]:>8.1f}% {pct[('al_like','dimm')][i]:>8.1f}% "
          f"{pct[('azurelily','dimm')][i]:>8.1f}%")
