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

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1})

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

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(seq_lens, ratios_dual, color="#059669", linewidth=2.5,
        marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1,
        label="Dual-identity mapping")

ax.plot(seq_lens, ratios_regular, color="#2563EB", linewidth=1.8,
        linestyle="--", marker="s", markersize=6, markeredgecolor="white", markeredgewidth=1,
        label="Regular mapping")

ax.axhline(y=1.0, color="#EF4444", linewidth=1.5, linestyle=":", alpha=0.6)

ax.fill_between(seq_lens, ratios_regular, ratios_dual, alpha=0.08, color="#059669")

# Annotate below data points
for i, N in enumerate(seq_lens):
    y = ratios_dual[i]
    ax.text(N, y - 0.04, f"{y:.2f}\u00d7", ha='center', va='top',
            fontsize=7, fontweight='bold', color='#059669')

# Baseline label on the left
ax.text(55, 1.03, "Azure-Lily baseline", fontsize=7, color="#EF4444",
        ha="left", fontstyle="italic")

ax.set_xlabel("Sequence Length (N)", fontsize=10)
ax.set_ylabel("Energy Efficiency\n(normalized, Azure-Lily = 1.0\u00d7)", fontsize=9)
ax.set_title("Attention Head: Dual-Identity vs Regular Mapping",
             fontsize=10, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(seq_lens)
ax.set_xticklabels([str(n) for n in seq_lens])
ax.set_ylim(bottom=0.9, top=max(ratios_dual) + 0.15)
ax.set_xlim(50, 2500)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.1)

out = OUT_DIR / "dual_identity_vs_regular.pdf"
fig.savefig(out)
print(f"Saved: {out}")
plt.close()
