#!/usr/bin/env python3
"""Generate Round 1 DSE plots for paper.

Plot 1: Config ranking bar chart (geomean tput/mm² and tput/J)
Plot 2: Config × Workload heatmap (normalized tput/mm², annotated with V)

Outputs:
    dse/results/round1_ranking.pdf
    dse/results/round1_heatmap.pdf
"""

import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "round1_results.csv"
TOP3_PATH = RESULTS_DIR / "top3_configs.json"

# ── Load data ────────────────────────────────────────────────────────────

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

with open(TOP3_PATH) as f:
    top3 = json.load(f)

ranking = top3["ranking"]
top_configs = top3["top_configs"]

# Sorted config names by rank (best first)
configs_ranked = [e["config"] for e in ranking]
workloads_ordered = ["fc_64_64", "fc_128_128", "fc_512_128", "fc_256_512", "fc_512_512", "fc_2048_256"]
workload_labels = {
    "fc_64_64": "FC 64×64",
    "fc_128_128": "FC 128×128",
    "fc_512_128": "FC 512×128",
    "fc_256_512": "FC 256×512",
    "fc_512_512": "FC 512×512",
    "fc_2048_256": "FC 2048×256",
}

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
COLOR_TOP = "#059669"    # emerald accent

# ═════════════════════════════════════════════════════════════════════════
# PLOT 1: Config Ranking Bar Chart
# ═════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(5.5, 3.8))

n = len(configs_ranked)
y_pos = np.arange(n)
bar_h = 0.35

gm_mm2 = [e["geomean_tput_mm2"] for e in ranking]
gm_J   = [e["geomean_tput_J"]   for e in ranking]

# Plot bars (reversed so #1 is at top)
bars_mm2 = ax1.barh(y_pos + bar_h / 2, gm_mm2[::-1], bar_h,
                     color=COLOR_MM2, alpha=0.85, label="Geomean Tput/mm²")
bars_J   = ax1.barh(y_pos - bar_h / 2, gm_J[::-1],   bar_h,
                     color=COLOR_J,   alpha=0.85, label="Geomean Tput/J")

# Labels
labels_reversed = configs_ranked[::-1]
for i, cfg in enumerate(labels_reversed):
    is_top = cfg in top_configs
    weight = "bold" if is_top else "normal"
    rank = n - i
    label = f"#{rank}  {cfg}"
    if is_top:
        label += "  *"  # top-3 marker
    ax1.text(-0.02, i, label, ha="right", va="center", fontsize=9,
             fontweight=weight, transform=ax1.get_yaxis_transform())

ax1.set_yticks([])
ax1.set_xlim(0, 1.05)
ax1.set_xlabel("Normalized Geomean Score (best = 1.0)")
ax1.set_title("Round 1: Crossbar Configuration Ranking", fontweight="bold", pad=10)
ax1.legend(loc="lower right", framealpha=0.9)
ax1.axvline(x=1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)

# Light grid
ax1.xaxis.grid(True, alpha=0.3, linestyle="--")
ax1.set_axisbelow(True)

fig1.tight_layout(rect=[0.18, 0, 1, 1])
fig1.savefig(RESULTS_DIR / "round1_ranking.pdf")
# fig1.savefig(RESULTS_DIR / "round1_ranking.png")
print(f"Saved: {RESULTS_DIR / 'round1_ranking.pdf'}")

# ═════════════════════════════════════════════════════════════════════════
# PLOT 2: Config × Workload Heatmap
# ═════════════════════════════════════════════════════════════════════════

# Build index: (config, workload) -> row
idx = {(r["config"], r["workload"]): r for r in rows}

# Per-workload best tput/mm² for normalization
best_mm2 = {}
for wl in workloads_ordered:
    vals = [float(idx[(c, wl)]["throughput_per_mm2"])
            for c in configs_ranked if (c, wl) in idx]
    best_mm2[wl] = max(vals) if vals else 1.0

# Build matrix (configs_ranked × workloads_ordered)
n_cfg = len(configs_ranked)
n_wl  = len(workloads_ordered)
mat       = np.zeros((n_cfg, n_wl))
V_mat     = np.zeros((n_cfg, n_wl), dtype=int)
acam_mat  = np.zeros((n_cfg, n_wl), dtype=bool)

for i, cfg in enumerate(configs_ranked):
    for j, wl in enumerate(workloads_ordered):
        r = idx.get((cfg, wl))
        if r:
            norm = float(r["throughput_per_mm2"]) / best_mm2[wl]
            mat[i, j] = norm
            V_mat[i, j] = int(r["V"])
            acam_mat[i, j] = r["acam_eligible"] == "True"

# Colormap: white (0) -> green (1)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "wg", ["#FFFFFF", "#D1FAE5", "#34D399", "#059669", "#064E3B"], N=256
)

fig2, ax2 = plt.subplots(figsize=(6, 4.2))
im = ax2.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1.0)

# Annotate cells
for i in range(n_cfg):
    for j in range(n_wl):
        v = V_mat[i, j]
        score = mat[i, j]
        is_acam = acam_mat[i, j]

        # Text color: dark on light cells, light on dark cells
        text_color = "white" if score > 0.6 else "#1F2937"

        # Main annotation: V value
        label = f"V={v}"
        if is_acam:
            label += "\nACAM"
        ax2.text(j, i, label, ha="center", va="center",
                 fontsize=8, color=text_color, fontweight="bold" if is_acam else "normal")

        # ACAM border
        if is_acam:
            rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                  linewidth=1.5, edgecolor=COLOR_TOP,
                                  facecolor="none", linestyle="-")
            ax2.add_patch(rect)

# Axes
ax2.set_xticks(range(n_wl))
ax2.set_xticklabels([workload_labels[w] for w in workloads_ordered], rotation=30, ha="right")
ax2.set_yticks(range(n_cfg))

# Config labels with rank
ylabels = []
for i, cfg in enumerate(configs_ranked):
    is_top = cfg in top_configs
    ylabels.append(f"#{i+1}  {cfg}" + ("  *" if is_top else ""))
ax2.set_yticklabels(ylabels, fontsize=9)
for i, cfg in enumerate(configs_ranked):
    if cfg in top_configs:
        ax2.get_yticklabels()[i].set_fontweight("bold")

ax2.set_title("Normalized Throughput/mm² by Config × Workload\n"
              "(ACAM = eligible for in-DPE activation, green border = V=1)",
              fontweight="bold", fontsize=10, pad=10)

# Colorbar
cbar = fig2.colorbar(im, ax=ax2, shrink=0.85, pad=0.02)
cbar.set_label("Normalized Tput/mm²\n(per-workload best = 1.0)", fontsize=9)

fig2.tight_layout()
fig2.savefig(RESULTS_DIR / "round1_heatmap.pdf")
# fig2.savefig(RESULTS_DIR / "round1_heatmap.png")
print(f"Saved: {RESULTS_DIR / 'round1_heatmap.pdf'}")

print("Done.")
