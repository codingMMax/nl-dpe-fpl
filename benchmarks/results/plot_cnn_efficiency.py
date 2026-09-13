#!/usr/bin/env python3
"""CNN Benchmark: Area & Energy Efficiency bar chart (normalized to Azure-Lily).

Input:  benchmarks/results/imc_benchmark_results.csv
Output: benchmarks/results/cnn_efficiency.pdf
"""
import csv, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.05})

RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "imc_benchmark_results.csv"
AREA_MM2 = 150 * 150 * 2239 / 1e6  # 150×150 grid, CLB tile = 2239 µm²

MODELS = ["ResNet-9", "VGG-11"]
ARCHS = ["NL-DPE Proposed", "NL-DPE AL-Matched", "Azure-Lily"]
COLORS = {"NL-DPE Proposed": "#059669", "NL-DPE AL-Matched": "#2563EB", "Azure-Lily": "#D1D5DB"}


def main():
    imc = {r['label']: r for r in csv.DictReader(open(CSV_PATH))}

    data = {}
    for label, r in imc.items():
        m = label.split(" NL-DPE")[0].split(" Azure")[0]
        if m not in MODELS:
            continue
        a = "NL-DPE Proposed" if "Proposed" in label else ("NL-DPE AL-Matched" if "AL-Matched" in label else "Azure-Lily")
        e, l = float(r['total_energy_pj']), float(r['total_latency_ns'])
        data[(m, a)] = {'tput_mm2': (1e9 / l) / AREA_MM2, 'tput_j': 1e12 / e}

    # Normalize to Azure-Lily
    norm = {}
    for m in MODELS:
        for a in ARCHS:
            norm[(m, a)] = {k: data[(m, a)][k] / data[(m, "Azure-Lily")][k] for k in ['tput_mm2', 'tput_j']}

    # Geomean
    for a in ARCHS:
        for k in ['tput_mm2', 'tput_j']:
            vals = [norm[(m, a)][k] for m in MODELS]
            norm.setdefault(("Geomean", a), {})[k] = math.exp(sum(math.log(v) for v in vals) / len(vals))

    cats = MODELS + ["Geomean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    fig.subplots_adjust(wspace=0.25)

    for ax, (metric, ylabel) in zip([ax1, ax2], [('tput_mm2', 'Area Eff.'), ('tput_j', 'Energy Eff.')]):
        x = np.arange(len(cats))
        w = 0.25
        max_val = 0

        for i, a in enumerate(ARCHS):
            vals = [norm[(m, a)][metric] for m in cats]
            max_val = max(max_val, max(vals))
            ax.bar(x + (i - 1) * w, vals, w, color=COLORS[a],
                   edgecolor="black" if a == "Azure-Lily" else "white",
                   linewidth=0.5, zorder=3)

            if a != "Azure-Lily":
                for j, v in enumerate(vals):
                    ax.text(x[j] + (i - 1) * w, v + max_val * 0.01, f"{v:.1f}×",
                            ha='center', va='bottom', fontsize=6.5, fontweight='bold',
                            color=COLORS[a], rotation=30)

        ax.axhline(y=1.0, color='#EF4444', linewidth=1, linestyle='--', alpha=0.5, zorder=2)
        ax.set_ylabel(f"Normalized {ylabel}\n(Azure-Lily = 1.0×)", fontsize=8)
        ax.set_xticks(x)
        lbls = ax.set_xticklabels(cats, fontsize=9)
        lbls[-1].set_fontweight('bold')
        ax.grid(True, alpha=0.08, axis="y", zorder=0)
        ax.set_ylim(bottom=0, top=max_val * 1.2)

    handles = [Patch(facecolor=COLORS[a], label=a.replace("NL-DPE ", "")) for a in ARCHS]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    out = RESULTS_DIR / "cnn_efficiency.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
