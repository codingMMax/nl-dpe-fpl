#!/usr/bin/env python3
"""2-panel figure with bracket annotation.

(a) Clean 3-segment stacked bars (DIMM, Proj+FFN, Other).
    A curly-brace annotation on one K=2 bar shows the DPE/Fabric split.
(b) Convergence line plot.

Output: paper/figures/benchmarks/bert_energy_2panel_bracket.pdf
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
OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "bert_energy_2panel_bracket.pdf"

# ── Load data ────────────────────────────────────────────────────────────
data = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [256, 512, 1024, 2048]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

# ── Derived metrics ──────────────────────────────────────────────────────
cat_pct = {}
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
        cat_pct[(arch, cat)] = vals

dpe_of_dimm = {}
for arch in ("Proposed", "Azure-Lily"):
    vals = []
    for N in SEQ_LENS:
        d = data[(arch, N)]
        dtotal = d["dimm_dpe_pj"] + d["dimm_fabric_pj"]
        vals.append(d["dimm_dpe_pj"] / dtotal * 100 if dtotal > 0 else 0)
    dpe_of_dimm[arch] = vals

dpe_of_total = {}
for arch in ("Proposed", "Azure-Lily"):
    vals = []
    for N in SEQ_LENS:
        d = data[(arch, N)]
        vals.append(d["dimm_dpe_pj"] / d["total_pj"] * 100)
    dpe_of_total[arch] = vals

energy_ratio = [data[("Azure-Lily", N)]["total_pj"] / data[("Proposed", N)]["total_pj"]
                for N in SEQ_LENS]
dimm_ratio = [data[("Azure-Lily", N)]["dimm_pj"] / data[("Proposed", N)]["dimm_pj"]
              for N in SEQ_LENS]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8),
                                gridspec_kw={'width_ratios': [1.3, 1]})
fig.subplots_adjust(wspace=0.35, top=0.82)

ax1.text(-0.12, 1.10, '(a)', transform=ax1.transAxes, fontsize=10,
         fontweight='bold', va='top')
ax2.text(-0.15, 1.10, '(b)', transform=ax2.transAxes, fontsize=10,
         fontweight='bold', va='top')

# ── Panel (a): clean 3-segment stacked bars ──────────────────────────────
bar_w = 0.35
arch_hatch = {"Proposed": None, "Azure-Lily": "///"}
cat_colors = [BREAKDOWN_COLORS["DIMM"], BREAKDOWN_COLORS["Proj_FFN"],
              BREAKDOWN_COLORS["Other"]]
cat_labels_display = ["DIMM (QK\u1d40+S\u00d7V)", "Proj+FFN", "Other"]
cat_keys = ["dimm", "projffn", "other"]

for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
    offset = (ai - 0.5) * bar_w
    hatch = arch_hatch[arch]

    for i in range(n_groups):
        bottom = 0
        for ci, ck in enumerate(cat_keys):
            val = cat_pct[(arch, ck)][i]
            ax1.bar(x[i] + offset, val, bar_w, bottom=bottom,
                    color=cat_colors[ci], edgecolor='white', linewidth=0.5,
                    hatch=hatch, zorder=3,
                    label=cat_labels_display[ci] if (i == 0 and ai == 0) else None)
            if val > 10:
                ax1.text(x[i] + offset, bottom + val / 2, f'{val:.0f}%',
                         ha='center', va='center', fontsize=5.5,
                         fontweight='bold', color='white', zorder=5)
            bottom += val

# ── Bracket annotation on K=2 bar at N=2048 (rightmost, tallest) ─────────
# Show DPE/Fabric split within the DIMM bar
bi = 3  # N=2048 index
k2_bar_x = x[bi] - 0.5 * bar_w  # center of K=2 bar
dimm_h = cat_pct[("Proposed", "dimm")][bi]
dpe_h = dpe_of_total["Proposed"][bi]
fab_h = dimm_h - dpe_h

# Bracket line to the left of the bar
bx = k2_bar_x - bar_w * 0.72  # x position of bracket

# Vertical lines for bracket
ax1.plot([bx, bx], [0, dimm_h], color='#555', linewidth=0.8, zorder=6,
         clip_on=False)
# Horizontal ticks
tick_len = bar_w * 0.12
for y_pos in [0, dpe_h, dimm_h]:
    ax1.plot([bx, bx + tick_len], [y_pos, y_pos], color='#555',
             linewidth=0.8, zorder=6, clip_on=False)

# Labels
ax1.text(bx - 0.04, dpe_h / 2, f'DPE\n{dpe_of_dimm["Proposed"][bi]:.0f}%',
         ha='right', va='center', fontsize=5, fontweight='bold',
         color='#7C3AED', zorder=6)
ax1.text(bx - 0.04, dpe_h + fab_h / 2, f'Fabric\n{100 - dpe_of_dimm["Proposed"][bi]:.0f}%',
         ha='right', va='center', fontsize=5, fontweight='bold',
         color='#4338CA', zorder=6)

# Also annotate Azure-Lily DIMM at N=2048: "100% Fabric"
al_bar_x = x[bi] + 0.5 * bar_w
al_dimm_h = cat_pct[("Azure-Lily", "dimm")][bi]
ax1.annotate('100%\nFabric', xy=(al_bar_x + bar_w * 0.55, al_dimm_h / 2),
             fontsize=5, fontweight='bold', color='#4338CA',
             ha='left', va='center')

# Legend
a_handles = [
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   label='K=2 (Proposed)'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   hatch='///', label='Azure-Lily'),
]
c_handles = [mpatches.Patch(facecolor=cat_colors[ci], edgecolor='white',
             label=cat_labels_display[ci]) for ci in range(3)]
ax1.legend(handles=a_handles + c_handles, fontsize=5.5, loc='center right',
           ncol=1, frameon=True, framealpha=0.9)

ax1.set_xticks(x)
ax1.set_xticklabels([str(N) for N in SEQ_LENS])
ax1.set_xlabel('Sequence Length (N)')
ax1.set_ylabel('Energy Composition (%)')
ax1.set_ylim(0, 108)
ax1.grid(True, alpha=0.08, axis='y', zorder=0)

# ── Panel (b): Convergence lines ────────────────────────────────────────
ax2.plot(x, energy_ratio, color=ARCH_COLORS["Proposed"], linewidth=2,
         marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.8,
         label='Overall (AL / K=2)', zorder=4)
ax2.plot(x, dimm_ratio, color='#E11D48', linewidth=2, linestyle='--',
         marker='D', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
         label='DIMM only (AL / K=2)', zorder=4)

for i, (er, dr) in enumerate(zip(energy_ratio, dimm_ratio)):
    ax2.annotate(f'{er:.2f}\u00d7', xy=(x[i], er), xytext=(0, 8),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                 color=ARCH_COLORS["Proposed"])
    y_off = -10 if i > 0 else 8
    va = 'top' if y_off < 0 else 'bottom'
    ax2.annotate(f'{dr:.2f}\u00d7', xy=(x[i], dr), xytext=(0, y_off),
                 textcoords='offset points', ha='center', va=va,
                 fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                 color='#E11D48')

ax2.annotate('', xy=(x[-1] + 0.15, dimm_ratio[-1]),
             xytext=(x[-1] + 0.15, energy_ratio[-1]),
             arrowprops=dict(arrowstyle='<->', color='#666', lw=1))
ax2.text(x[-1] + 0.3, (energy_ratio[-1] + dimm_ratio[-1]) / 2,
         'converges', fontsize=5, color='#666', ha='left', va='center',
         fontstyle='italic')

ax2.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax2.set_xticks(x)
ax2.set_xticklabels([str(N) for N in SEQ_LENS])
ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('Energy Ratio (Azure-Lily / Proposed)')
ax2.set_ylim(0.9, 1.85)
ax2.legend(fontsize=6, loc='upper right', frameon=True)
ax2.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
