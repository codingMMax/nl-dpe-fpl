#!/usr/bin/env python3
"""3-panel figure: BERT-Tiny latency convergence story.

Left:   % breakdown bars (DIMM vs Proj+FFN+Other) -- both archs, across N
Middle: Absolute latency comparison (Proposed vs Azure-Lily) across N
Right:  Latency ratio (AL / Proposed) for overall and DIMM-only

Output: paper/figures/benchmarks/bert_latency_3panel.pdf
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
OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "bert_latency_3panel.pdf"

# ── Load data ────────────────────────────────────────────────────────────
data = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

# ── Derived metrics ──────────────────────────────────────────────────────
pct = {}
for arch in ("Proposed", "Azure-Lily"):
    for cat in ("dimm", "projffn", "other"):
        vals = []
        for N in SEQ_LENS:
            d = data[(arch, N)]
            total = d["total_ns"]
            if cat == "dimm":
                vals.append(d["dimm_ns"] / total * 100)
            elif cat == "projffn":
                vals.append((d["proj_ns"] + d["ffn_ns"]) / total * 100)
            else:
                vals.append(d["other_ns"] / total * 100)
        pct[(arch, cat)] = vals

latency_ratio = [data[("Azure-Lily", N)]["total_ns"] / data[("Proposed", N)]["total_ns"]
                 for N in SEQ_LENS]
dimm_lat_ratio = [data[("Azure-Lily", N)]["dimm_ns"] / data[("Proposed", N)]["dimm_ns"]
                  if data[("Proposed", N)]["dimm_ns"] > 0 else 0
                  for N in SEQ_LENS]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.6),
                                     gridspec_kw={'width_ratios': [1, 1, 0.9]})
fig.subplots_adjust(wspace=0.45, top=0.78, bottom=0.18)

# ── Panel left: DIMM % of latency — line plot for both archs ────────────
cat_colors = {"Proposed": ARCH_COLORS["Proposed"], "Azure-Lily": ARCH_COLORS["Azure-Lily"]}
cat_markers = {"Proposed": "o", "Azure-Lily": "D"}
cat_styles = {"Proposed": "-", "Azure-Lily": "--"}

for arch in ["Proposed", "Azure-Lily"]:
    dimm_pcts = pct[(arch, "dimm")]
    ax1.plot(SEQ_LENS, dimm_pcts, color=cat_colors[arch], linewidth=1.5,
             linestyle=cat_styles[arch], marker=cat_markers[arch], markersize=4,
             markeredgecolor="white", markeredgewidth=0.5, label=arch)

# Annotate endpoints: Azure-Lily above (higher %), Proposed below
for arch in ["Proposed", "Azure-Lily"]:
    dimm_pcts = pct[(arch, "dimm")]
    r = dimm_pcts[-1]
    dy = -8 if arch == "Proposed" else 6
    va = 'top' if arch == "Proposed" else 'bottom'
    ax1.annotate(f'{r:.0f}%', xy=(SEQ_LENS[-1], r),
                 xytext=(-6, dy), textcoords='offset points',
                 ha='right', va=va, fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                 color=cat_colors[arch])

TICK_LENS = [512, 2048, 4096, 6144, 8192]
ax1.set_xticks(TICK_LENS)
ax1.set_xticklabels([str(n) for n in TICK_LENS], fontsize=7)
ax1.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.1)
ax1.set_xlabel('Sequence Length (N)')
ax1.set_ylabel('DIMM Latency Proportion (%)')
ax1.set_ylim(0, 100)
ax1.legend(fontsize=6, loc='center right')
ax1.grid(True, alpha=0.1)

# ── Panel middle: Normalized throughput (1/latency, norm to Azure-Lily) ─
for arch in ["Proposed", "Azure-Lily"]:
    norm_tput = []
    for N in SEQ_LENS:
        tput = 1.0 / data[(arch, N)]["total_ns"]
        tput_al = 1.0 / data[("Azure-Lily", N)]["total_ns"]
        norm_tput.append(tput / tput_al)
    ax2.plot(SEQ_LENS, norm_tput, color=cat_colors[arch], linewidth=1.5,
             linestyle=cat_styles[arch], marker=cat_markers[arch], markersize=4,
             markeredgecolor="white", markeredgewidth=0.5, label=arch)
    # Annotate Proposed only (Azure-Lily is always 1.0×)
    if arch == "Proposed":
        r = norm_tput[-1]
        ax2.annotate(f'{r:.1f}\u00d7', xy=(SEQ_LENS[-1], r),
                     xytext=(-6, -8), textcoords='offset points',
                     ha='right', va='top', fontsize=ANNOT_FONTSIZE,
                     fontweight=ANNOT_FONTWEIGHT, color=cat_colors[arch])

ax2.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax2.set_xticks(TICK_LENS)
ax2.set_xticklabels([str(n) for n in TICK_LENS], fontsize=7)
ax2.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.1)
ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('Norm. Throughput\n(vs Azure-Lily)')
ax2.legend(fontsize=6, loc='upper left', frameon=True)
ax2.grid(True, alpha=0.1)

# ── Panel right: Latency ratio lines ───────────────────────────────────
ax3.plot(SEQ_LENS, latency_ratio, color='#2E86AB', linewidth=2,
         marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
         label='Overall (AL/K=2)', zorder=4)
ax3.plot(SEQ_LENS, dimm_lat_ratio, color='#E05263', linewidth=2, linestyle='--',
         marker='s', markersize=3, markeredgecolor='white', markeredgewidth=0.8,
         label='DIMM only (AL/K=2)', zorder=4)

# Single asymptotic line (overall and DIMM converge close together)
asymp = (latency_ratio[-1] + dimm_lat_ratio[-1]) / 2
ax3.axhline(y=asymp, color='#666', linewidth=1, linestyle=':', alpha=0.7, zorder=2)
ax3.annotate(f'{asymp:.1f}\u00d7', xy=(SEQ_LENS[-1], asymp),
             xytext=(-4, 5), textcoords='offset points', ha='right', va='bottom',
             fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT, color='#444')

ax3.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax3.set_xticks(TICK_LENS)
ax3.set_xticklabels([str(n) for n in TICK_LENS], fontsize=7)
ax3.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.1)
ax3.set_xlabel('Sequence Length (N)')
ax3.set_ylabel('Latency Ratio (AL / Proposed)')
all_ratios = latency_ratio + dimm_lat_ratio
ax3.set_ylim(min(all_ratios) * 0.7, max(all_ratios) * 1.15)
ax3.legend(fontsize=5.5, loc='upper right', frameon=True)
ax3.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# Print summary
print(f"\n{'N':>5} {'L_ratio':>8} {'DIMM_ratio':>11}")
print("-" * 30)
for i, N in enumerate(SEQ_LENS):
    print(f"{N:>5} {latency_ratio[i]:>7.2f}\u00d7 {dimm_lat_ratio[i]:>10.2f}\u00d7")
