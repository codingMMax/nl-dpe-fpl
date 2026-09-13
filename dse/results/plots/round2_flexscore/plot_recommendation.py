#!/usr/bin/env python3
"""FlexScore DSE v3: DPE Area Recommendation Plot.

X = Budget %, two Y-axes: throughput (left) + FlexScore (right).
Shows the sweet spot where throughput saturates before FlexScore drops.
"""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path(__file__).resolve().parent
DL_CSV = RESULTS_DIR / "flexscore_dl_gemv_results.csv"
SUMMARY_CSV = RESULTS_DIR / "flexscore_summary.csv"

BUDGET_LEVELS = [0, 10, 20, 30, 40, 50]
DL_WORKLOADS = ["fc_512_128", "fc_512_512", "fc_2048_256"]
CONFIGS = ["512x128", "1024x128", "512x64", "512x256", "1024x256"]
CONFIG_LABELS = {
    "512x128": "512\u00d7128", "1024x128": "1024\u00d7128", "512x64": "512\u00d764",
    "512x256": "512\u00d7256", "1024x256": "1024\u00d7256",
}
CONFIG_TG = {
    "512x128": "512x128", "1024x128": "1024x128", "512x64": "512x64",
    "512x256": "512x256", "1024x256": "1024x256",
}
CONFIG_COLORS = {
    "512x128": "#10B981", "1024x128": "#3B82F6", "512x64": "#8B5CF6",
    "512x256": "#F97316", "1024x256": "#EC4899",
}


def main():
    fs_lookup = {}
    if SUMMARY_CSV.exists():
        with open(SUMMARY_CSV) as f:
            for r in csv.DictReader(f):
                fs_lookup[(r["tg"], int(r["budget_pct"]))] = float(r["flexscore"])

    by_point = defaultdict(dict)
    if DL_CSV.exists():
        with open(DL_CSV) as f:
            for r in csv.DictReader(f):
                if r["status"].startswith("OK"):
                    key = (r["config"], int(r["budget_pct"]))
                    by_point[key][r["workload"]] = float(r["eff_latency_ns"])

    if not fs_lookup or not by_point:
        print("No data. Run flexscore_dse.py --nondl --dl first.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    for cfg in CONFIGS:
        tg = CONFIG_TG[cfg]
        color = CONFIG_COLORS[cfg]
        label = CONFIG_LABELS[cfg]

        budgets, tputs, fscores = [], [], []
        for b in BUDGET_LEVELS:
            if b == 0:
                continue
            key = (cfg, b)
            fs_key = (tg, b)
            if key in by_point and fs_key in fs_lookup:
                wl_lats = by_point[key]
                vals = [wl_lats[wl] for wl in DL_WORKLOADS if wl in wl_lats]
                valid = [v for v in vals if 0 < v < 1e5]
                if len(valid) == len(DL_WORKLOADS):
                    gm_lat = math.exp(sum(math.log(v) for v in valid) / len(valid))
                    tput = 1.0 / gm_lat  # inferences per ns
                    budgets.append(b)
                    tputs.append(tput)
                    fscores.append(fs_lookup[fs_key])

        if budgets:
            ax1.plot(budgets, tputs, color=color, marker="o", markersize=5,
                     linewidth=1.5, label=f"{label}")
            ax2.plot(budgets, fscores, color=color, marker="s", markersize=4,
                     linewidth=1.5, linestyle="--", alpha=0.6)

            # Find sweet spot: highest throughput where FS >= 0.90
            best_b, best_t = None, -1
            for b, t, fs in zip(budgets, tputs, fscores):
                if fs >= 0.90 and t > best_t:
                    best_b, best_t = b, t
            if best_b is not None:
                idx = budgets.index(best_b)
                ax1.annotate(f"{label}\nBudget={best_b}%",
                             xy=(best_b, best_t),
                             xytext=(best_b + 3, best_t * 1.15),
                             fontsize=7, fontweight="bold", color=color,
                             arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax1.set_xlabel("Area Budget (%)", fontsize=11)
    ax1.set_ylabel("Geomean Throughput (inf/ns, solid)", fontsize=11)
    ax2.set_ylabel("FlexScore (dashed)", fontsize=11, color="#666")
    ax1.set_title("DPE Area Recommendation: Throughput vs FlexScore",
                  fontweight="bold")
    ax1.set_xticks(BUDGET_LEVELS)
    ax2.set_ylim(0, 1.05)
    ax1.set_ylim(bottom=0)
    ax2.axhline(y=0.90, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.15)
    ax1.legend(fontsize=8, loc="upper left", ncol=2)

    out = RESULTS_DIR / "round2_flexscore_recommendation.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
