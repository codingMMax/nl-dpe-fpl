#!/usr/bin/env python3
"""FlexScore scalability plots (v3 — proportional budget).

Plot 1 (main): Dual-Y — Budget% vs Latency (left) and FlexScore (right)
Plot 2 (scalability): Per-tile-group FlexScore vs budget
Plot 3 (per-benchmark): Per-benchmark normalized Fmax vs budget
"""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path(__file__).resolve().parent
DATA_DIR = RESULTS_DIR.parent / "data"
OUTPUT_DIR = RESULTS_DIR.parent / "figures"
RAW_CSV = DATA_DIR / "flexscore_raw_results.csv"
SUMMARY_CSV = DATA_DIR / "flexscore_summary.csv"
DL_CSV = DATA_DIR / "flexscore_dl_results.csv"

BUDGET_LEVELS = [0, 10, 20, 30, 40, 50]
BENCHMARKS = ["bgm", "LU8PEEng", "stereovision1", "arm_core"]
BENCH_COLORS = {"bgm": "#2563EB", "LU8PEEng": "#DC2626",
                "stereovision1": "#059669", "arm_core": "#D97706"}
BENCH_MARKERS = {"bgm": "o", "LU8PEEng": "s",
                 "stereovision1": "^", "arm_core": "D"}

CONFIGS = ["512x128", "1024x128", "512x64", "512x256", "1024x256"]
CONFIG_LABELS = {
    "512x128": "512\u00d7128", "1024x128": "1024\u00d7128", "512x64": "512\u00d764",
    "512x256": "512\u00d7256", "1024x256": "1024\u00d7256",
}
# Per-config FlexScore (tg = config name in CSV)
CONFIG_TG = {
    "512x128": "512x128", "1024x128": "1024x128", "512x64": "512x64",
    "512x256": "512x256", "1024x256": "1024x256",
}
CONFIG_COLORS = {
    "512x128": "#10B981", "1024x128": "#3B82F6", "512x64": "#8B5CF6",
    "512x256": "#F97316", "1024x256": "#EC4899",
}
TG_LABELS = {c: CONFIG_LABELS[c] for c in CONFIGS}
TG_COLORS = {c: CONFIG_COLORS[c] for c in CONFIGS}
DL_WORKLOADS = ["fc_512_128", "fc_512_512", "fc_2048_256"]


def load_raw():
    rows = []
    if not RAW_CSV.exists():
        return rows
    with open(RAW_CSV) as f:
        for r in csv.DictReader(f):
            r["budget_pct"] = int(r["budget_pct"])
            r["fmax_mhz"] = float(r["fmax_mhz"])
            rows.append(r)
    return rows


def load_summary():
    fs = {}
    if not SUMMARY_CSV.exists():
        return fs
    with open(SUMMARY_CSV) as f:
        for r in csv.DictReader(f):
            fs[(r["tg"], int(r["budget_pct"]))] = float(r["flexscore"])
    return fs


def load_dl():
    rows = []
    if not DL_CSV.exists():
        return rows
    with open(DL_CSV) as f:
        for r in csv.DictReader(f):
            if r["status"].startswith("OK"):
                r["budget_pct"] = int(r["budget_pct"])
                r["eff_latency_ns"] = float(r["eff_latency_ns"])
                r["dpe_area_pct"] = float(r["dpe_area_pct"])
                rows.append(r)
    return rows


def compute_dl_geomean(dl_rows):
    """Geomean latency per (config, budget)."""
    by_point = defaultdict(dict)
    for r in dl_rows:
        by_point[(r["config"], r["budget_pct"])][r["workload"]] = r["eff_latency_ns"]
    result = {}
    for key, wl_lats in by_point.items():
        vals = [wl_lats[wl] for wl in DL_WORKLOADS if wl in wl_lats]
        valid = [v for v in vals if 0 < v < 1e5]
        if len(valid) == len(DL_WORKLOADS):
            gm = math.exp(sum(math.log(v) for v in valid) / len(valid))
            result[key] = gm
    return result


def plot_main(fs_lookup, dl_rows):
    """Dual-Y: Budget% vs Latency + FlexScore, one line per config."""
    dl_geomean = compute_dl_geomean(dl_rows)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    for cfg in CONFIGS:
        tg = CONFIG_TG[cfg]
        color = CONFIG_COLORS[cfg]
        label = CONFIG_LABELS[cfg]

        budgets, lats, fscores = [], [], []
        for b in BUDGET_LEVELS:
            if b == 0:
                continue
            key = (cfg, b)
            fs_key = (tg, b)
            if key in dl_geomean and fs_key in fs_lookup:
                budgets.append(b)
                lats.append(dl_geomean[key])
                fscores.append(fs_lookup[fs_key])

        if budgets:
            ax1.plot(budgets, lats, color=color, marker="o", markersize=5,
                     linewidth=1.5, label=f"{label} (lat)")
            ax2.plot(budgets, fscores, color=color, marker="s", markersize=4,
                     linewidth=1.5, linestyle="--", alpha=0.7)

    for tg, tg_color in TG_COLORS.items():
        if (tg, 0) in fs_lookup:
            ax2.axhline(y=fs_lookup[(tg, 0)], color=tg_color, linewidth=0.5,
                        linestyle=":", alpha=0.4)

    ax1.set_xlabel("Area Budget (%)", fontsize=11)
    ax1.set_ylabel("Geomean Eff. Latency (ns/inf, solid)", fontsize=11, color="#333")
    ax2.set_ylabel("FlexScore (dashed)", fontsize=11, color="#666")
    ax1.set_title("DL Latency vs Non-DL FlexScore \u2014 Proportional Budget Sweep",
                  fontweight="bold")
    ax1.set_xticks(BUDGET_LEVELS)
    ax2.set_ylim(0, 1.05)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.15)
    ax1.legend(fontsize=8, loc="upper left", ncol=2)

    out = RESULTS_DIR / "round2_flexscore_main.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_scalability(fs_lookup):
    """FlexScore vs budget per tile group."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for tg, color in TG_COLORS.items():
        budgets, scores = [], []
        for b in BUDGET_LEVELS:
            if (tg, b) in fs_lookup:
                budgets.append(b)
                scores.append(fs_lookup[(tg, b)])
        if budgets:
            ax.plot(budgets, scores, color=color, marker="o", markersize=6,
                    linewidth=2, label=TG_LABELS.get(tg, tg))

    ax.set_xlabel("Area Budget (%)", fontsize=11)
    ax.set_ylabel("FlexScore", fontsize=11)
    ax.set_title("FlexScore vs Area Budget by Tile Group", fontweight="bold")
    ax.set_xticks(BUDGET_LEVELS)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color="gray", linewidth=0.8, linestyle="--", alpha=0.5,
               label="FS = 0.95")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    out = RESULTS_DIR / "round2_flexscore_scalability.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_per_benchmark(raw_rows):
    """Per-benchmark normalized Fmax vs budget, one subplot per tile group."""
    baselines = {}
    for r in raw_rows:
        if r["budget_pct"] == 0 and r["status"].startswith("OK"):
            baselines[(r["tg"], r["benchmark"])] = r["fmax_mhz"]

    tgs = sorted(set(r["tg"] for r in raw_rows))
    n_tg = len(tgs)
    fig, axes = plt.subplots(1, n_tg, figsize=(5 * n_tg, 4.5), sharey=True)
    if n_tg == 1:
        axes = [axes]

    for idx, tg in enumerate(tgs):
        ax = axes[idx]
        for bench in BENCHMARKS:
            color = BENCH_COLORS[bench]
            marker = BENCH_MARKERS[bench]
            bl = baselines.get((tg, bench), 0)
            if bl <= 0:
                continue

            budgets, ratios = [], []
            for b in BUDGET_LEVELS:
                matches = [r for r in raw_rows
                           if r["tg"] == tg and r["budget_pct"] == b
                           and r["benchmark"] == bench and r["status"].startswith("OK")]
                if matches:
                    budgets.append(b)
                    ratios.append(matches[0]["fmax_mhz"] / bl)
                else:
                    if b > 0:
                        budgets.append(b)
                        ratios.append(0.0)

            if budgets:
                ax.plot(budgets, ratios, color=color, marker=marker,
                        markersize=5, linewidth=1.5, label=bench)

        ax.set_xlabel("Area Budget (%)")
        if idx == 0:
            ax.set_ylabel("Normalized Fmax (ratio to baseline)")
        ax.set_title(TG_LABELS.get(tg, tg), fontsize=10, fontweight="bold")
        ax.set_xticks(BUDGET_LEVELS)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.axhline(y=0.0, color="red", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.15)

    fig.suptitle("Per-Benchmark Fmax Degradation vs Area Budget",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = RESULTS_DIR / "round2_flexscore_per_benchmark.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def main():
    raw_rows = load_raw()
    fs_lookup = load_summary()
    dl_rows = load_dl()

    if fs_lookup:
        plot_scalability(fs_lookup)
    if raw_rows:
        plot_per_benchmark(raw_rows)
    if fs_lookup and dl_rows:
        plot_main(fs_lookup, dl_rows)

    if not fs_lookup and not dl_rows:
        print("No data found. Run flexscore_dse.py --nondl --dl first.")


if __name__ == "__main__":
    main()
