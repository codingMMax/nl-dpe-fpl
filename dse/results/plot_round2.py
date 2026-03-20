#!/usr/bin/env python3
"""Generate Round 2 DSE plots for paper.

Plot 1: CLB replacement ratio vs throughput/mm² per config (faceted by workload type)
Plot 2: CLB replacement ratio vs throughput/J per config
Plot 3: Resource utilization: wc_count and clb_count vs ratio
Plot 4: Geomean comparison across ratios per config

Outputs:
    dse/results/round2_clb_tput_mm2.pdf
    dse/results/round2_clb_tput_J.pdf
    dse/results/round2_geomean.pdf
"""

import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "round2_results.csv"

# ── Load data ────────────────────────────────────────────────────────────

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

# Group by config
configs = sorted(set(r["config"] for r in rows))
ratios = sorted(set(float(r["clb_replace_ratio"]) for r in rows))
workloads = sorted(set(r["workload"] for r in rows))

# Separate FC and attention workloads
fc_workloads = sorted(w for w in workloads if w.startswith("fc_"))
attn_workloads = [w for w in workloads if w == "attention"]

workload_labels = {
    "fc_64_64": "FC 64x64",
    "fc_128_128": "FC 128x128",
    "fc_512_128": "FC 512x128",
    "fc_256_512": "FC 256x512",
    "fc_512_512": "FC 512x512",
    "fc_2048_256": "FC 2048x256",
    "attention": "Attention",
}

# Index: (config, ratio, workload) -> row
idx = {}
for r in rows:
    key = (r["config"], float(r["clb_replace_ratio"]), r["workload"])
    idx[key] = r

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

CONFIG_COLORS = {
    "512x128": "#2563EB",  # blue
    "512x64": "#F59E0B",   # amber
    "512x256": "#059669",  # emerald
}
CONFIG_MARKERS = {
    "512x128": "o",
    "512x64": "s",
    "512x256": "D",
}

# ═════════════════════════════════════════════════════════════════════════
# PLOT 1: CLB Replacement Ratio vs Throughput/mm² (faceted by workload)
# ═════════════════════════════════════════════════════════════════════════

all_wl = fc_workloads + attn_workloads
n_wl = len(all_wl)
n_cols = 4
n_rows_grid = math.ceil(n_wl / n_cols)

fig1, axes1 = plt.subplots(n_rows_grid, n_cols, figsize=(12, 3.2 * n_rows_grid),
                            sharex=True)
axes1_flat = axes1.flatten() if n_wl > 1 else [axes1]

ratio_pcts = [r * 100 for r in ratios]

for wi, wl in enumerate(all_wl):
    ax = axes1_flat[wi]
    for cfg in configs:
        x_pts, y_pts = [], []
        for ratio in ratios:
            r = idx.get((cfg, ratio, wl))
            if r:
                x_pts.append(ratio * 100)
                y_pts.append(float(r["throughput_per_mm2"]))

        ax.plot(x_pts, y_pts,
                marker=CONFIG_MARKERS.get(cfg, "o"),
                color=CONFIG_COLORS.get(cfg, "#666"),
                linewidth=1.5, markersize=5,
                label=cfg)

    ax.set_title(workload_labels.get(wl, wl), fontsize=10, fontweight="bold")
    ax.set_ylabel("Tput/mm\u00B2" if wi % n_cols == 0 else "")
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

# Remove empty subplots
for i in range(n_wl, len(axes1_flat)):
    axes1_flat[i].set_visible(False)

# Shared x label and legend
for ax in axes1_flat[max(0, n_wl - n_cols):n_wl]:
    ax.set_xlabel("CLB Replace Ratio (%)")
axes1_flat[0].legend(loc="best", framealpha=0.9)

fig1.suptitle("Round 2: Throughput/mm\u00B2 vs CLB Replacement Ratio\n"
              "(Fixed 106\u00D7106 grid, top-3 configs)",
              fontweight="bold", fontsize=12, y=1.02)
fig1.tight_layout()
fig1.savefig(RESULTS_DIR / "round2_clb_tput_mm2.pdf")
print(f"Saved: {RESULTS_DIR / 'round2_clb_tput_mm2.pdf'}")

# ═════════════════════════════════════════════════════════════════════════
# PLOT 2: CLB Replacement Ratio vs Throughput/J (faceted by workload)
# ═════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(n_rows_grid, n_cols, figsize=(12, 3.2 * n_rows_grid),
                            sharex=True)
axes2_flat = axes2.flatten() if n_wl > 1 else [axes2]

for wi, wl in enumerate(all_wl):
    ax = axes2_flat[wi]
    for cfg in configs:
        x_pts, y_pts = [], []
        for ratio in ratios:
            r = idx.get((cfg, ratio, wl))
            if r:
                x_pts.append(ratio * 100)
                y_pts.append(float(r["throughput_per_J"]))

        ax.plot(x_pts, y_pts,
                marker=CONFIG_MARKERS.get(cfg, "o"),
                color=CONFIG_COLORS.get(cfg, "#666"),
                linewidth=1.5, markersize=5,
                label=cfg)

    ax.set_title(workload_labels.get(wl, wl), fontsize=10, fontweight="bold")
    ax.set_ylabel("Tput/J" if wi % n_cols == 0 else "")
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

for i in range(n_wl, len(axes2_flat)):
    axes2_flat[i].set_visible(False)

for ax in axes2_flat[max(0, n_wl - n_cols):n_wl]:
    ax.set_xlabel("CLB Replace Ratio (%)")
axes2_flat[0].legend(loc="best", framealpha=0.9)

fig2.suptitle("Round 2: Throughput/J vs CLB Replacement Ratio\n"
              "(Fixed 106\u00D7106 grid, top-3 configs)",
              fontweight="bold", fontsize=12, y=1.02)
fig2.tight_layout()
fig2.savefig(RESULTS_DIR / "round2_clb_tput_J.pdf")
print(f"Saved: {RESULTS_DIR / 'round2_clb_tput_J.pdf'}")

# ═════════════════════════════════════════════════════════════════════════
# PLOT 3: Geomean Throughput/mm² and Tput/J across ratios per config
# ═════════════════════════════════════════════════════════════════════════

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4))

for cfg in configs:
    x_pts = []
    gm_mm2 = []
    gm_J = []
    for ratio in ratios:
        vals_mm2 = []
        vals_J = []
        for wl in all_wl:
            r = idx.get((cfg, ratio, wl))
            if r:
                vals_mm2.append(float(r["throughput_per_mm2"]))
                vals_J.append(float(r["throughput_per_J"]))
        if vals_mm2:
            x_pts.append(ratio * 100)
            gm_mm2.append(math.exp(sum(math.log(v) for v in vals_mm2) / len(vals_mm2)))
            gm_J.append(math.exp(sum(math.log(v) for v in vals_J) / len(vals_J)))

    ax3a.plot(x_pts, gm_mm2,
              marker=CONFIG_MARKERS.get(cfg, "o"),
              color=CONFIG_COLORS.get(cfg, "#666"),
              linewidth=2, markersize=6, label=cfg)
    ax3b.plot(x_pts, gm_J,
              marker=CONFIG_MARKERS.get(cfg, "o"),
              color=CONFIG_COLORS.get(cfg, "#666"),
              linewidth=2, markersize=6, label=cfg)

ax3a.set_xlabel("CLB Replace Ratio (%)")
ax3a.set_ylabel("Geomean Tput/mm\u00B2")
ax3a.set_title("Area Efficiency", fontweight="bold")
ax3a.legend(framealpha=0.9)
ax3a.xaxis.grid(True, alpha=0.3, linestyle="--")
ax3a.yaxis.grid(True, alpha=0.3, linestyle="--")
ax3a.set_axisbelow(True)

ax3b.set_xlabel("CLB Replace Ratio (%)")
ax3b.set_ylabel("Geomean Tput/J")
ax3b.set_title("Energy Efficiency", fontweight="bold")
ax3b.legend(framealpha=0.9)
ax3b.xaxis.grid(True, alpha=0.3, linestyle="--")
ax3b.yaxis.grid(True, alpha=0.3, linestyle="--")
ax3b.set_axisbelow(True)

fig3.suptitle("Round 2: Geomean Performance vs CLB Replacement Ratio",
              fontweight="bold", fontsize=12)
fig3.tight_layout()
fig3.savefig(RESULTS_DIR / "round2_geomean.pdf")
print(f"Saved: {RESULTS_DIR / 'round2_geomean.pdf'}")

print("Done.")
