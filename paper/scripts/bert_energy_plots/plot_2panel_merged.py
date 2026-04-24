#!/usr/bin/env python3
"""2-panel merged figure: BERT-Tiny energy convergence.

(a) 4-segment stacked bars: DPE(DIMM) + Fabric(DIMM) + Proj+FFN + Other
    Shows DIMM dominance AND DPE/fabric split in one view.
(b) Line plot: overall energy ratio + DIMM-only ratio → convergence.

Output: paper/figures/benchmarks/bert_energy_2panel_merged.pdf
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
OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "bert_energy_2panel_merged.pdf"

# ── Load data ────────────────────────────────────────────────────────────
data = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [256, 512, 1024, 2048]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

# ── Derive 4-segment breakdown (% of total) ─────────────────────────────
# Segments: DPE(DIMM), Fabric(DIMM), Proj+FFN, Other
seg_names = ["DPE\n(crossbar)", "FPGA fabric\n(DIMM)", "Proj+FFN", "Other"]
seg_colors = ["#8B5CF6", "#6366F1", "#F59E0B", "#CBD5E1"]  # violet, indigo, amber, gray

seg_pct = {}  # (arch, seg_idx) -> [vals over N]
for arch in ("Proposed", "Azure-Lily"):
    for si in range(4):
        vals = []
        for N in SEQ_LENS:
            d = data[(arch, N)]
            total = d["total_pj"]
            if si == 0:    # DPE portion of DIMM
                vals.append(d["dimm_dpe_pj"] / total * 100)
            elif si == 1:  # Fabric portion of DIMM
                vals.append(d["dimm_fabric_pj"] / total * 100)
            elif si == 2:  # Proj+FFN
                vals.append((d["proj_pj"] + d["ffn_pj"]) / total * 100)
            else:          # Other
                vals.append(d["other_pj"] / total * 100)
        seg_pct[(arch, si)] = vals

# Energy ratios
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

# ── Panel (a): 4-segment stacked bars ────────────────────────────────────
bar_w = 0.35
arch_hatch = {"Proposed": None, "Azure-Lily": "///"}

for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
    offset = (ai - 0.5) * bar_w
    hatch = arch_hatch[arch]

    for i in range(n_groups):
        bottom = 0
        for si in range(4):
            val = seg_pct[(arch, si)][i]
            ax1.bar(x[i] + offset, val, bar_w, bottom=bottom,
                    color=seg_colors[si], edgecolor='white', linewidth=0.5,
                    hatch=hatch, zorder=3,
                    label=seg_names[si] if (i == 0 and ai == 0) else None)
            if val > 6:
                ax1.text(x[i] + offset, bottom + val / 2, f'{val:.0f}%',
                         ha='center', va='center', fontsize=5,
                         fontweight='bold', color='white', zorder=5)
            bottom += val

# Legend
a_handles = [
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   label='K=2 (Proposed)'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   hatch='///', label='Azure-Lily'),
]
s_handles = [mpatches.Patch(facecolor=seg_colors[si], edgecolor='white',
             label=seg_names[si].replace('\n', ' '))
             for si in range(4)]
ax1.legend(handles=a_handles + s_handles, fontsize=5.5, loc='lower right',
           ncol=2, frameon=True, framealpha=0.9, columnspacing=0.8,
           handletextpad=0.4)

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

# Annotate all points
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

# Convergence arrow
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

# ── Print summary ────────────────────────────────────────────────────────
print(f"\n{'N':>5} {'Arch':>12} {'DPE%':>6} {'Fab%':>6} {'P+F%':>6} {'Oth%':>6}")
print("-" * 50)
for N in SEQ_LENS:
    for arch in ("Proposed", "Azure-Lily"):
        vals = [seg_pct[(arch, si)][SEQ_LENS.index(N)] for si in range(4)]
        print(f"{N:>5} {arch:>12} {vals[0]:>5.1f}% {vals[1]:>5.1f}% "
              f"{vals[2]:>5.1f}% {vals[3]:>5.1f}%")
    print()

print(f"{'N':>5} {'E_ratio':>8} {'DIMM_ratio':>11} {'gap':>6}")
print("-" * 35)
for i, N in enumerate(SEQ_LENS):
    print(f"{N:>5} {energy_ratio[i]:>7.2f}\u00d7 {dimm_ratio[i]:>10.2f}\u00d7 "
          f"{energy_ratio[i] - dimm_ratio[i]:>5.2f}")
