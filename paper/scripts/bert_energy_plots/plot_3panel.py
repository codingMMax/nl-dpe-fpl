#!/usr/bin/env python3
"""3-panel figure: BERT-Tiny energy convergence story.

Left:   % breakdown bars (DIMM vs Proj+FFN+Other) — both archs, across N
Middle: DIMM-internal split (DPE vs fabric %) — both archs, across N
Right:  Line plot: overall energy ratio + DIMM-only ratio vs N

Shared legend above left+middle panels.

Output: paper/figures/benchmarks/bert_energy_3panel.pdf
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
ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from style_constants import (apply_style, ARCH_COLORS, BREAKDOWN_COLORS,
                              BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE, ANNOT_FONTWEIGHT)
apply_style()

CSV_PATH = SCRIPT_DIR / "bert_energy_sweep.csv"
OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "bert_energy_3panel.pdf"

# ── Load data ────────────────────────────────────────────────────────────
data = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [256, 512, 1024, 1536, 2048]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

# ── Derived metrics ──────────────────────────────────────────────────────
pct = {}
for arch in ("Proposed", "Azure-Lily"):
    for cat in ("dimm", "projffn", "other"):
        vals = []
        for N in SEQ_LENS:
            d = data[(arch, N)]
            total = d["total_pj"]
            if cat == "dimm":
                vals.append(d["dimm_pj"] / total * 100)
            elif cat == "projffn":
                vals.append((d["proj_pj"] + d["ffn_pj"]) / total * 100)
            else:
                vals.append(d["other_pj"] / total * 100)
        pct[(arch, cat)] = vals

dimm_dpe_pct = {}
for arch in ("Proposed", "Azure-Lily"):
    dimm_dpe_pct[arch] = []
    for N in SEQ_LENS:
        d = data[(arch, N)]
        dtotal = d["dimm_dpe_pj"] + d["dimm_fabric_pj"]
        dimm_dpe_pct[arch].append(d["dimm_dpe_pj"] / dtotal * 100 if dtotal > 0 else 0)

energy_ratio = [data[("Azure-Lily", N)]["total_pj"] / data[("Proposed", N)]["total_pj"]
                for N in SEQ_LENS]
dimm_ratio = [data[("Azure-Lily", N)]["dimm_pj"] / data[("Proposed", N)]["dimm_pj"]
              for N in SEQ_LENS]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.6),
                                     gridspec_kw={'width_ratios': [1, 1, 0.9]})
fig.subplots_adjust(wspace=0.45, top=0.78, bottom=0.18)

# ── Panel left: DIMM vs Proj+FFN+Other stacked bars ─────────────────────
bar_w = 0.35
arch_hatch = {"Proposed": None, "Azure-Lily": "///"}
# Lightweight pastels: easy to distinguish, black text readable
cat_colors = ["#7EC8E3", "#FFB3BA", "#D5D5D5"]  # light blue, pink, light gray
cat_labels = ["DIMM (QK\u1d40+S\u00d7V)", "Proj+FFN", "Other"]
cat_keys = ["dimm", "projffn", "other"]

for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
    offset = (ai - 0.5) * bar_w
    hatch = arch_hatch[arch]
    for i in range(n_groups):
        bottom = 0
        for ci, ck in enumerate(cat_keys):
            val = pct[(arch, ck)][i]
            ax1.bar(x[i] + offset, val, bar_w, bottom=bottom,
                    color=cat_colors[ci], edgecolor='white', linewidth=0.5,
                    hatch=hatch, zorder=3)
            if val > 10:
                ax1.text(x[i] + offset, bottom + val / 2, f'{val:.0f}%',
                         ha='center', va='center', fontsize=4.5,
                         fontweight='bold', color='black', zorder=5)
            bottom += val

ax1.set_xticks(x)
ax1.set_xticklabels([str(N) for N in SEQ_LENS])
ax1.set_xlabel('Sequence Length (N)')
ax1.set_ylabel('Energy Composition (%)')
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.08, axis='y', zorder=0)

# ── Panel middle: DIMM-internal DPE vs Fabric ────────────────────────────
# Lightweight: green for DPE, yellow for fabric
DPE_COLOR = "#B5EAD7"     # mint green
FABRIC_COLOR = "#FFDAC1"  # peach/yellow

for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
    offset = (ai - 0.5) * bar_w
    hatch = arch_hatch[arch]
    dpe = dimm_dpe_pct[arch]
    fab = [100 - d for d in dpe]

    for i in range(n_groups):
        ax2.bar(x[i] + offset, dpe[i], bar_w, color=DPE_COLOR,
                edgecolor='white', linewidth=0.5, hatch=hatch, zorder=3)
        ax2.bar(x[i] + offset, fab[i], bar_w, bottom=dpe[i],
                color=FABRIC_COLOR, edgecolor='white', linewidth=0.5,
                hatch=hatch, zorder=3)
        if dpe[i] > 10:
            ax2.text(x[i] + offset, dpe[i] / 2, f'{dpe[i]:.0f}%',
                     ha='center', va='center', fontsize=4.5,
                     fontweight='bold', color='black', zorder=5)
        if fab[i] > 10:
            ax2.text(x[i] + offset, dpe[i] + fab[i] / 2, f'{fab[i]:.0f}%',
                     ha='center', va='center', fontsize=4.5,
                     fontweight='bold', color='black', zorder=5)

ax2.set_xticks(x)
ax2.set_xticklabels([str(N) for N in SEQ_LENS])
ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('DIMM Energy Composition (%)')
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.08, axis='y', zorder=0)

# ── Shared legend above left + middle panels ─────────────────────────────
# Architecture handles
a_handles = [
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   label='K=2 (Proposed)'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   hatch='///', label='Azure-Lily'),
]
# Left panel category handles
c_handles = [mpatches.Patch(facecolor=cat_colors[ci], edgecolor='white',
             label=cat_labels[ci]) for ci in range(3)]
# Middle panel category handles
d_handles = [
    mpatches.Patch(facecolor=DPE_COLOR, edgecolor='white', label='DPE (crossbar)'),
    mpatches.Patch(facecolor=FABRIC_COLOR, edgecolor='white', label='FPGA fabric (CLB/DSP)'),
]

all_handles = a_handles + c_handles + d_handles
# Center the legend above ax1 and ax2 (left two panels)
# Get the midpoint between ax1 left and ax2 right in figure coords
bbox_mid = ((ax1.get_position().x0 + ax2.get_position().x1) / 2, 0.97)
fig.legend(handles=all_handles, loc='upper center', ncol=4, fontsize=5.5,
           frameon=True, framealpha=0.9, bbox_to_anchor=bbox_mid,
           columnspacing=0.8, handletextpad=0.3)

# ── Panel right: Energy ratio lines ──────────────────────────────────────
ax3.plot(x, energy_ratio, color='#2E86AB', linewidth=2,
         marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.8,
         label='Overall (AL/K=2)', zorder=4)
ax3.plot(x, dimm_ratio, color='#E05263', linewidth=2, linestyle='--',
         marker='s', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
         label='DIMM only (AL/K=2)', zorder=4)

# Annotate all points
for i, (er, dr) in enumerate(zip(energy_ratio, dimm_ratio)):
    ax3.annotate(f'{er:.2f}\u00d7', xy=(x[i], er), xytext=(0, 8),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                 color='#2E86AB')
    y_off = -10 if i > 0 else 8
    va = 'top' if y_off < 0 else 'bottom'
    ax3.annotate(f'{dr:.2f}\u00d7', xy=(x[i], dr), xytext=(0, y_off),
                 textcoords='offset points', ha='center', va=va,
                 fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                 color='#E05263')

ax3.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax3.set_xticks(x)
ax3.set_xticklabels([str(N) for N in SEQ_LENS])
ax3.set_xlabel('Sequence Length (N)')
ax3.set_ylabel('Energy Ratio (AL / Proposed)')
ax3.set_ylim(0.9, 1.85)
ax3.legend(fontsize=5.5, loc='upper right', frameon=True)
ax3.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# Print summary
print(f"\n{'N':>5} {'E_ratio':>8} {'DIMM_ratio':>11} {'gap':>6}")
print("-" * 35)
for i, N in enumerate(SEQ_LENS):
    print(f"{N:>5} {energy_ratio[i]:>7.2f}\u00d7 {dimm_ratio[i]:>10.2f}\u00d7 "
          f"{energy_ratio[i] - dimm_ratio[i]:>5.2f}")
