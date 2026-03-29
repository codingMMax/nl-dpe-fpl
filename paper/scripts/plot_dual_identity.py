#!/usr/bin/env python3
"""Dual-Identity vs Regular Mapping: Energy Efficiency vs Sequence Length.

Normalized to Azure-Lily = 1.0×. Shows the improvement from dual-identity
packing on QK^T DIMM operations.

Output: paper/figures/benchmarks/dual_identity_vs_regular.pdf
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from style_constants import (apply_style, BASELINE_COLOR, BASELINE_LS,
                              BASELINE_ALPHA, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT,
                              DUAL_COLORS, FIG_SINGLE)
apply_style()

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "figures" / "benchmarks"

# Energy constants
e_analoge = 3.89; e_digital = 0.171445; k_vmm = 8
e_dsp_mac = 1.2; e_clb_add = 0.085
C = 128; d_head = 64

ep_nl = k_vmm * e_analoge + e_digital * C   # 53.1 pJ per NL-DPE pass
ep_al = k_vmm * 2.33 * C                     # 2385.9 pJ per Azure-Lily pass

seq_lens = [64, 128, 256, 512, 1024, 2048]

ratios_dual = []
ratios_regular = []

for N in seq_lens:
    proj_nl = 3 * N * ep_nl
    proj_al = 3 * N * ep_al
    n_qkt = N * N
    e_qkt_regular = n_qkt * (d_head * e_clb_add + 1 * ep_nl + (d_head - 1) * e_clb_add)
    e_qkt_dual = n_qkt * (d_head * e_clb_add + 0.5 * ep_nl + (d_head - 1) * e_clb_add)
    e_qkt_al = n_qkt * d_head * e_dsp_mac
    e_soft = N * N * ep_nl
    e_soft_al = N * N * 2 * e_dsp_mac
    n_sv = N * d_head
    e_sv_nl = n_sv * (N * e_clb_add + math.ceil(N / C) * ep_nl + (N - 1) * e_clb_add)
    e_sv_al = n_sv * N * e_dsp_mac

    total_regular = proj_nl + e_qkt_regular + e_soft + e_sv_nl
    total_dual = proj_nl + e_qkt_dual + e_soft + e_sv_nl
    total_al = proj_al + e_qkt_al + e_soft_al + e_sv_al

    ratios_dual.append(total_al / total_dual)
    ratios_regular.append(total_al / total_regular)

fig, ax = plt.subplots(figsize=FIG_SINGLE)

ax.plot(seq_lens, ratios_dual, color=DUAL_COLORS["dual"], linewidth=2,
        marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8,
        label="Dual-identity mapping")

ax.plot(seq_lens, ratios_regular, color=DUAL_COLORS["regular"], linewidth=2,
        linestyle="--", marker="s", markersize=4, markeredgecolor="white", markeredgewidth=0.8,
        label="Regular mapping")

ax.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
           alpha=BASELINE_ALPHA)

ax.fill_between(seq_lens, ratios_regular, ratios_dual, alpha=0.12,
                color=DUAL_COLORS["dual"])

# Annotate first and last points only to reduce clutter
for i, N in enumerate(seq_lens):
    if i == 0 or i == len(seq_lens) - 1:
        y = ratios_dual[i]
        ax.annotate(f"{y:.2f}\u00d7", xy=(N, y),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                    color=DUAL_COLORS["dual"])
        y_r = ratios_regular[i]
        # First point: regular below; last point: regular above to avoid axis clip
        r_yoff = -12 if i == 0 else 8
        r_va = 'top' if i == 0 else 'bottom'
        ax.annotate(f"{y_r:.2f}\u00d7", xy=(N, y_r),
                    xytext=(0, r_yoff), textcoords='offset points',
                    ha='center', va=r_va,
                    fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                    color=DUAL_COLORS["regular"])

ax.set_xlabel("Sequence Length (N)")
ax.set_ylabel("Energy Efficiency\n(norm. to Azure-Lily = 1.0\u00d7)")
ax.set_xscale("log", base=2)
ax.set_xticks(seq_lens)
ax.set_xticklabels([str(n) for n in seq_lens])
ax.set_ylim(bottom=0.9, top=max(ratios_dual) + 0.2)
ax.set_xlim(50, 2500)
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.1)

out = OUT_DIR / "dual_identity_vs_regular.pdf"
fig.savefig(out)
print(f"Saved: {out}")
plt.close()
