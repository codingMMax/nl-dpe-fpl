#!/usr/bin/env python3
"""2-panel dual-Y-axis figure: compact story of BERT-Tiny energy convergence.

(a) Left Y: DIMM % of total (bars). Right Y: energy ratio Azure-Lily/Proposed (line).
    Shows: as DIMM dominates, overall benefit drops.
(b) Left Y: DIMM-internal DPE vs fabric % (bars). Right Y: DIMM energy ratio (line).
    Shows: constant DPE/fabric split → constant DIMM ratio = asymptotic floor.

Output: paper/figures/benchmarks/bert_energy_2panel.pdf
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
OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "bert_energy_2panel.pdf"

# ── Load data ────────────────────────────────────────────────────────────
data = {}  # (arch, N) -> row dict
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [256, 512, 1024, 2048]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

# ── Derived metrics ──────────────────────────────────────────────────────
# DIMM % of total
dimm_pct = {}
for arch in ("Proposed", "Azure-Lily"):
    dimm_pct[arch] = [data[(arch, N)]["dimm_pj"] / data[(arch, N)]["total_pj"] * 100
                      for N in SEQ_LENS]

# Energy ratio (AL / Proposed)
energy_ratio = [data[("Azure-Lily", N)]["total_pj"] / data[("Proposed", N)]["total_pj"]
                for N in SEQ_LENS]

# DIMM-internal DPE %
dimm_dpe_pct = {}
for arch in ("Proposed", "Azure-Lily"):
    dimm_dpe_pct[arch] = []
    for N in SEQ_LENS:
        d = data[(arch, N)]
        dtotal = d["dimm_dpe_pj"] + d["dimm_fabric_pj"]
        dimm_dpe_pct[arch].append(d["dimm_dpe_pj"] / dtotal * 100 if dtotal > 0 else 0)

# DIMM energy ratio (AL DIMM / Proposed DIMM)
dimm_ratio = [data[("Azure-Lily", N)]["dimm_pj"] / data[("Proposed", N)]["dimm_pj"]
              for N in SEQ_LENS]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
fig.subplots_adjust(wspace=0.45, top=0.82)

ax1.text(-0.14, 1.10, '(a)', transform=ax1.transAxes, fontsize=10,
         fontweight='bold', va='top')
ax2.text(-0.17, 1.10, '(b)', transform=ax2.transAxes, fontsize=10,
         fontweight='bold', va='top')

# ── Panel (a): DIMM % bars + energy ratio line ──────────────────────────
bar_w = 0.3
colors_a = {"Proposed": ARCH_COLORS["Proposed"], "Azure-Lily": ARCH_COLORS["Azure-Lily"]}

for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
    offset = (ai - 0.5) * bar_w
    ax1.bar(x + offset, dimm_pct[arch], bar_w, color=colors_a[arch],
            edgecolor='white', linewidth=0.5, zorder=3,
            label=f'{arch} DIMM%')
    # Annotate %
    for i, v in enumerate(dimm_pct[arch]):
        ax1.text(x[i] + offset, v - 4, f'{v:.0f}%', ha='center', va='top',
                 fontsize=5.5, fontweight='bold', color='white', zorder=5)

ax1.set_ylabel('DIMM Share of Total Energy (%)')
ax1.set_ylim(0, 115)
ax1.set_xticks(x)
ax1.set_xticklabels([str(N) for N in SEQ_LENS])
ax1.set_xlabel('Sequence Length (N)')
ax1.grid(True, alpha=0.08, axis='y', zorder=0)

# Right Y: energy ratio line
ax1r = ax1.twinx()
ax1r.plot(x, energy_ratio, color='#E11D48', linewidth=2, marker='D',
          markersize=4, markeredgecolor='white', markeredgewidth=0.8,
          zorder=5, label='Energy ratio (AL/K=2)')
for i, r in enumerate(energy_ratio):
    ax1r.annotate(f'{r:.2f}×', xy=(x[i], r), xytext=(0, 8),
                  textcoords='offset points', ha='center', va='bottom',
                  fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                  color='#E11D48')
ax1r.set_ylabel('Energy Ratio (Azure-Lily / Proposed)', color='#E11D48')
ax1r.tick_params(axis='y', colors='#E11D48')
ax1r.set_ylim(1.0, 2.0)

# Combined legend
bars_handles = [
    mpatches.Patch(facecolor=ARCH_COLORS["Proposed"], label='K=2 DIMM%'),
    mpatches.Patch(facecolor=ARCH_COLORS["Azure-Lily"], label='AL DIMM%'),
]
line_handle = ax1r.get_legend_handles_labels()[0]
ax1.legend(handles=bars_handles + line_handle, fontsize=6, loc='upper left',
           ncol=1, frameon=True, framealpha=0.9)

# ── Panel (b): DIMM internal split bars + DIMM ratio line ───────────────
DPE_COLOR = "#8B5CF6"
FABRIC_COLOR = "#F59E0B"
bar_w2 = 0.3
arch_hatch = {"Proposed": None, "Azure-Lily": "///"}

for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
    offset = (ai - 0.5) * bar_w2
    hatch = arch_hatch[arch]
    dpe = dimm_dpe_pct[arch]
    fab = [100 - d for d in dpe]

    for i in range(n_groups):
        ax2.bar(x[i] + offset, dpe[i], bar_w2, color=DPE_COLOR,
                edgecolor='white', linewidth=0.5, hatch=hatch, zorder=3,
                label='DPE (crossbar)' if (i == 0 and ai == 0) else None)
        ax2.bar(x[i] + offset, fab[i], bar_w2, bottom=dpe[i],
                color=FABRIC_COLOR, edgecolor='white', linewidth=0.5,
                hatch=hatch, zorder=3,
                label='FPGA fabric' if (i == 0 and ai == 0) else None)
        if dpe[i] > 8:
            ax2.text(x[i] + offset, dpe[i] / 2, f'{dpe[i]:.0f}%',
                     ha='center', va='center', fontsize=5.5, fontweight='bold',
                     color='white', zorder=5)
        if fab[i] > 8:
            ax2.text(x[i] + offset, dpe[i] + fab[i] / 2, f'{fab[i]:.0f}%',
                     ha='center', va='center', fontsize=5.5, fontweight='bold',
                     color='white', zorder=5)

ax2.set_ylabel('DIMM Energy Composition (%)')
ax2.set_ylim(0, 115)
ax2.set_xticks(x)
ax2.set_xticklabels([str(N) for N in SEQ_LENS])
ax2.set_xlabel('Sequence Length (N)')
ax2.grid(True, alpha=0.08, axis='y', zorder=0)

# Right Y: DIMM ratio line
ax2r = ax2.twinx()
ax2r.plot(x, dimm_ratio, color='#E11D48', linewidth=2, marker='D',
          markersize=4, markeredgecolor='white', markeredgewidth=0.8,
          zorder=5, label='DIMM ratio (AL/K=2)')
for i, r in enumerate(dimm_ratio):
    y_off = 8 if i != len(SEQ_LENS) - 1 else -12
    va = 'bottom' if y_off > 0 else 'top'
    ax2r.annotate(f'{r:.2f}×', xy=(x[i], r), xytext=(0, y_off),
                  textcoords='offset points', ha='center', va=va,
                  fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                  color='#E11D48')
ax2r.set_ylabel('DIMM Ratio (Azure-Lily / Proposed)', color='#E11D48')
ax2r.tick_params(axis='y', colors='#E11D48')
ax2r.set_ylim(1.0, 1.8)

# Combined legend
arch_handles = [
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   label='K=2 (Proposed)'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   hatch='///', label='Azure-Lily'),
]
cat_handles = [
    mpatches.Patch(facecolor=DPE_COLOR, label='DPE (crossbar)'),
    mpatches.Patch(facecolor=FABRIC_COLOR, label='FPGA fabric'),
]
line_handle2 = ax2r.get_legend_handles_labels()[0]
ax2.legend(handles=arch_handles + cat_handles + line_handle2, fontsize=5.5,
           loc='center left', ncol=1, frameon=True, framealpha=0.9)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# Print summary
print(f"\n{'N':>5} {'DIMM%(K2)':>10} {'DIMM%(AL)':>10} {'E_ratio':>8} {'DIMM_ratio':>11}")
print("-" * 50)
for i, N in enumerate(SEQ_LENS):
    print(f"{N:>5} {dimm_pct['Proposed'][i]:>9.1f}% {dimm_pct['Azure-Lily'][i]:>9.1f}% "
          f"{energy_ratio[i]:>7.2f}× {dimm_ratio[i]:>10.2f}×")
