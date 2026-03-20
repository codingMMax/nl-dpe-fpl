#!/usr/bin/env python3
"""Generate attention DSE plots for paper.

Plot 1: Config ranking bar chart (tput/mm² and tput/J)
Plot 2: Energy breakdown stacked bar (DPE vs FPGA CLB vs Memory)

Outputs:
    dse/results/attention_ranking.pdf
    dse/results/attention_energy_breakdown.pdf
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "attention_results.csv"

# ── Load data ────────────────────────────────────────────────────────────

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

# Sort by throughput/mm² descending (best first)
rows.sort(key=lambda r: -float(r["throughput_per_mm2"]))

configs = [r["config"] for r in rows]
tput_mm2 = [float(r["throughput_per_mm2"]) for r in rows]
tput_J = [float(r["throughput_per_J"]) for r in rows]
fmax = [float(r["fmax_mhz"]) for r in rows]
e_dpe = [float(r["e_dpe_grouped"]) for r in rows]
e_fpga = [float(r["e_fpga_grouped"]) for r in rows]
e_mem = [float(r["e_mem_grouped"]) for r in rows]
total_dpes = [int(r["total_dpes"]) for r in rows]
h_proj = [int(r["H_proj"]) for r in rows]

# ── Shared style ─────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLOR_MM2 = "#2563EB"   # blue
COLOR_J   = "#F59E0B"   # amber
COLOR_DPE  = "#3B82F6"  # blue
COLOR_FPGA = "#EF4444"  # red
COLOR_MEM  = "#8B5CF6"  # purple

# ═════════════════════════════════════════════════════════════════════════
# PLOT 1: Config Ranking Bar Chart (Tput/mm² and Tput/J)
# ═════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(5.5, 3.8))

n = len(configs)
y_pos = np.arange(n)
bar_h = 0.35

# Normalize to best
best_mm2 = max(tput_mm2)
best_J = max(tput_J)
norm_mm2 = [v / best_mm2 for v in tput_mm2]
norm_J = [v / best_J for v in tput_J]

# Plot bars (reversed so #1 is at top)
ax1.barh(y_pos + bar_h / 2, norm_mm2[::-1], bar_h,
         color=COLOR_MM2, alpha=0.85, label="Tput/mm\u00B2")
ax1.barh(y_pos - bar_h / 2, norm_J[::-1], bar_h,
         color=COLOR_J, alpha=0.85, label="Tput/J")

# Labels
configs_rev = configs[::-1]
fmax_rev = fmax[::-1]
dpes_rev = total_dpes[::-1]
for i, cfg in enumerate(configs_rev):
    rank = n - i
    label = f"#{rank}  {cfg}  ({dpes_rev[i]} DPEs, {fmax_rev[i]:.0f} MHz)"
    ax1.text(-0.02, i, label, ha="right", va="center", fontsize=8,
             fontweight="bold" if rank <= 3 else "normal",
             transform=ax1.get_yaxis_transform())

ax1.set_yticks([])
ax1.set_xlim(0, 1.08)
ax1.set_xlabel("Normalized Score (best = 1.0)")
ax1.set_title("Attention Head: Config Ranking (N=128, d=128)",
              fontweight="bold", pad=10)
ax1.legend(loc="lower right", framealpha=0.9)
ax1.axvline(x=1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
ax1.xaxis.grid(True, alpha=0.3, linestyle="--")
ax1.set_axisbelow(True)

fig1.tight_layout(rect=[0.30, 0, 1, 1])
fig1.savefig(RESULTS_DIR / "attention_ranking.pdf")
print(f"Saved: {RESULTS_DIR / 'attention_ranking.pdf'}")

# ═════════════════════════════════════════════════════════════════════════
# PLOT 2: Energy Breakdown Stacked Bar
# ═════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(6, 4))

x_pos = np.arange(n)
bar_w = 0.6

# Convert to thousands of pJ (kpJ) for readability
e_dpe_k = [v / 1e3 for v in e_dpe]
e_fpga_k = [v / 1e3 for v in e_fpga]
e_mem_k = [v / 1e3 for v in e_mem]

bars_dpe = ax2.bar(x_pos, e_dpe_k, bar_w, color=COLOR_DPE, alpha=0.85,
                   label="DPE (Q/K/V projections)")
bars_mem = ax2.bar(x_pos, e_mem_k, bar_w, bottom=e_dpe_k, color=COLOR_MEM,
                   alpha=0.85, label="Memory (SRAM)")
bars_fpga = ax2.bar(x_pos, e_fpga_k, bar_w,
                    bottom=[d + m for d, m in zip(e_dpe_k, e_mem_k)],
                    color=COLOR_FPGA, alpha=0.85,
                    label="FPGA CLB (DIMM + softmax)")

# Percentage annotations on each bar
for i in range(n):
    total = e_dpe[i] + e_fpga[i] + e_mem[i]
    pct_fpga = e_fpga[i] / total * 100
    pct_dpe = e_dpe[i] / total * 100
    # Annotate FPGA percentage on top of bar
    top_y = (e_dpe_k[i] + e_mem_k[i] + e_fpga_k[i])
    ax2.text(i, top_y + 15, f"{pct_fpga:.0f}%\nCLB",
             ha="center", va="bottom", fontsize=7, color=COLOR_FPGA,
             fontweight="bold")
    # Annotate DPE percentage inside DPE bar
    ax2.text(i, e_dpe_k[i] / 2, f"{pct_dpe:.1f}%",
             ha="center", va="center", fontsize=7, color="white",
             fontweight="bold")

ax2.set_xticks(x_pos)
ax2.set_xticklabels(configs, rotation=45, ha="right")
ax2.set_ylabel("Energy (kpJ)")
ax2.set_title("Attention Energy Breakdown: DPE vs CLB vs Memory\n"
              "(N=128, d=128 — CLB DIMM/softmax dominates)",
              fontweight="bold", fontsize=10, pad=10)
ax2.legend(loc="upper right", framealpha=0.9)
ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
ax2.set_axisbelow(True)

fig2.tight_layout()
fig2.savefig(RESULTS_DIR / "attention_energy_breakdown.pdf")
print(f"Saved: {RESULTS_DIR / 'attention_energy_breakdown.pdf'}")

print("Done.")
