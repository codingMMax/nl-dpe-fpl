#!/usr/bin/env python3
"""BERT-Tiny Latency 2-panel: (1) DIMM fraction, (2) Speedup breakdown."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
from style_constants import (apply_style, ARCH_COLORS, BASELINE_COLOR,
                              BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE, ANNOT_FONTWEIGHT)
apply_style()

CSV_PATH = SCRIPT_DIR / "bert_energy_plots" / "bert_energy_sweep.csv"
OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "bert_latency_2panel.pdf"

# Load data
data = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [1024, 2048, 4096, 6144, 8192]
TICK_X = [2048, 4096, 6144, 8192]

# Metrics
dimm_frac_p = [data[("Proposed", N)]["dimm_ns"] / data[("Proposed", N)]["total_ns"] * 100 for N in SEQ_LENS]
dimm_frac_a = [data[("Azure-Lily", N)]["dimm_ns"] / data[("Azure-Lily", N)]["total_ns"] * 100 for N in SEQ_LENS]

overall_speedup = [data[("Azure-Lily", N)]["total_ns"] / data[("Proposed", N)]["total_ns"] for N in SEQ_LENS]
dimm_speedup = [data[("Azure-Lily", N)]["dimm_ns"] / data[("Proposed", N)]["dimm_ns"] for N in SEQ_LENS]
nondimm_speedup = []
for N in SEQ_LENS:
    p_nd = data[("Proposed", N)]["total_ns"] - data[("Proposed", N)]["dimm_ns"]
    a_nd = data[("Azure-Lily", N)]["total_ns"] - data[("Azure-Lily", N)]["dimm_ns"]
    nondimm_speedup.append(a_nd / p_nd)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))
fig.subplots_adjust(wspace=0.30, top=0.78, bottom=0.18)

COLOR_P = ARCH_COLORS["Proposed-1"]
COLOR_A = ARCH_COLORS["Azure-Lily"]

# ── Panel 1: DIMM latency fraction ──
ax1.plot(SEQ_LENS, dimm_frac_p, color=COLOR_P, linewidth=1.5, linestyle='-',
         marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.5,
         label='Proposed-1')
ax1.plot(SEQ_LENS, dimm_frac_a, color=COLOR_A, linewidth=1.5, linestyle='--',
         marker='D', markersize=4, markeredgecolor='white', markeredgewidth=0.5,
         label='Azure-Lily')

for frac, color, dy, va in [(dimm_frac_p, COLOR_P, -6, 'top'), (dimm_frac_a, COLOR_A, 6, 'bottom')]:
    ax1.annotate(f'{frac[-1]:.0f}%', xy=(SEQ_LENS[-1], frac[-1]),
                 xytext=(-6, dy), textcoords='offset points', ha='right', va=va,
                 fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT, color=color)

ax1.set_xticks(TICK_X)
ax1.set_xticklabels([str(n) for n in TICK_X])
ax1.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.1)
ax1.set_xlabel('Sequence Length (N)')
ax1.set_ylabel('DIMM Latency Proportion (%)')
ax1.set_ylim(0, 100)
ax1.legend(fontsize=6, loc='center left', bbox_to_anchor=(0.02, 0.30),
           ncol=1, frameon=True, fancybox=True, framealpha=0.9, edgecolor='#CCC')
ax1.grid(True, alpha=0.1)

# ── Panel 2: Speedup breakdown ──
ax2.plot(SEQ_LENS, overall_speedup, color='#333', linewidth=2, linestyle='-',
         marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.8,
         label='Overall', zorder=4)
ax2.plot(SEQ_LENS, dimm_speedup, color='#2E86AB', linewidth=1.5, linestyle='--',
         marker='s', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
         label='DIMM only', zorder=3)
ax2.plot(SEQ_LENS, nondimm_speedup, color='#E05263', linewidth=1.5, linestyle=':',
         marker='^', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
         label='Non-DIMM', zorder=3)

# Asymptotic line for DIMM
ax2.axhline(y=14.0, color='#555', linewidth=1.0, linestyle='--', alpha=0.5)
mid_x = SEQ_LENS[len(SEQ_LENS) // 2]
ax2.text(SEQ_LENS[-2], 14.0 + 0.3, '14\u00d7', ha='center', va='bottom',
         fontsize=7, color='#555', fontweight='bold')
ax2.text(mid_x, nondimm_speedup[-1] + 0.3, f'{nondimm_speedup[-1]:.1f}\u00d7',
         ha='center', va='bottom', fontsize=7, color='#E05263', fontweight='bold')

ax2.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS, alpha=BASELINE_ALPHA)
ax2.set_xticks(TICK_X)
ax2.set_xticklabels([str(n) for n in TICK_X])
ax2.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.15)
ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('Speedup over Azure-Lily', labelpad=6)
ax2.set_ylim(0, max(dimm_speedup) * 1.15)
ax2.legend(fontsize=5.5, loc='center right')
ax2.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
plt.close()
