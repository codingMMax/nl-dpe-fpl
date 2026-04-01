#!/usr/bin/env python3
"""CNN Benchmark: Area & Energy Efficiency bar chart (normalized to Azure-Lily).

Input:  imc_benchmark_results.csv (via symlink)
Output: paper/figures/benchmarks/cnn_efficiency.pdf
"""
import csv, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
from style_constants import apply_style_sc, ARCH_COLORS, ANNOT_FONTSIZE_SC, ANNOT_FONTWEIGHT_SC, FPGA_AREA_MM2
apply_style_sc()

DATA_DIR = SCRIPT_DIR.parent / "data"
CSV_PATH = DATA_DIR / "imc_benchmark_results.csv"
OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "cnn_efficiency.pdf"

MODELS = ["ResNet-9", "VGG-11"]
ARCHS = ["Proposed-1", "Proposed-2", "Azure-Lily"]


def main():
    imc = {r['label']: r for r in csv.DictReader(open(CSV_PATH))}

    data = {}
    for label, r in imc.items():
        m = label.split(" NL-DPE")[0].split(" Azure")[0]
        if m not in MODELS:
            continue
        a = "Proposed-1" if "Proposed" in label else (
            "Proposed-2" if ("AL-Matched" in label or "AL-like" in label) else "Azure-Lily")
        e, l = float(r['total_energy_pj']), float(r['total_latency_ns'])
        data[(m, a)] = {'tput_mm2': (1e9 / l) / FPGA_AREA_MM2, 'tput_j': 1e12 / e}

    norm = {}
    for m in MODELS:
        for a in ARCHS:
            norm[(m, a)] = {k: data[(m, a)][k] / data[(m, "Azure-Lily")][k]
                            for k in ['tput_mm2', 'tput_j']}
    for a in ARCHS:
        for k in ['tput_mm2', 'tput_j']:
            vals = [norm[(m, a)][k] for m in MODELS]
            norm.setdefault(("Geomean", a), {})[k] = math.exp(
                sum(math.log(v) for v in vals) / len(vals))

    cats = MODELS + ["Geomean"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.subplots_adjust(wspace=0.18)

    for ax, (metric, ylabel) in zip([ax1, ax2],
            [('tput_mm2', 'Normalized Inference/s/mm\u00b2'),
             ('tput_j', 'Normalized Inference/J')]):
        x = np.arange(len(cats))
        w = 0.25
        max_val = 0

        for i, a in enumerate(ARCHS):
            vals = [norm[(m, a)][metric] for m in cats]
            max_val = max(max_val, max(vals))
            ax.bar(x + (i - 1) * w, vals, w, color=ARCH_COLORS[a],
                   edgecolor="white", linewidth=0.5, zorder=3)

            if a != "Azure-Lily":
                for j, v in enumerate(vals):
                    ax.text(x[j] + (i - 1) * w, v + max_val * 0.02,
                            f"{v:.1f}\u00d7", ha='center', va='bottom',
                            fontsize=ANNOT_FONTSIZE_SC, fontweight=ANNOT_FONTWEIGHT_SC,
                            color=ARCH_COLORS[a], rotation=0)

        ax.axhline(y=1.0, color='#94A3B8', linewidth=1, linestyle=':', alpha=0.5, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        lbls = ax.set_xticklabels(cats)
        lbls[-1].set_fontweight('bold')
        ax.grid(True, alpha=0.08, axis="y", zorder=0)
        ax.set_ylim(bottom=0, top=max_val * 1.3)

    # ── Secondary axis on energy panel: per-pass energy ratio ──
    # Proposed (C=128): 53.0 pJ/pass, AL-like (C=256): 84.7 pJ/pass, Azure-Lily: 2385.9 pJ/pass
    ax2r = ax2.twinx()
    ratio_proposed = 2385.9 / 53.0    # ≈ 45×
    ratio_allike = 2385.9 / 84.7      # ≈ 28× (C=256 → more ACAM energy)

    ax2r.axhline(y=ratio_proposed, color="#B91C1C", linewidth=1.5,
                 linestyle="--", alpha=0.7, zorder=1)
    ax2r.text(len(cats) - 0.6, ratio_proposed + 0.8,
              f"Proposed-1 / Azure-Lily: {ratio_proposed:.0f}\u00d7",
              ha="right", va="bottom", fontsize=7, fontweight="bold",
              color="#B91C1C", fontstyle="italic")

    ax2r.axhline(y=ratio_allike, color="#475569", linewidth=1.5,
                 linestyle="--", alpha=0.7, zorder=1)
    ax2r.text(len(cats) - 0.6, ratio_allike + 0.8,
              f"Proposed-2 / Azure-Lily: {ratio_allike:.0f}\u00d7",
              ha="right", va="bottom", fontsize=7, fontweight="bold",
              color="#475569", fontstyle="italic")

    ax2r.set_ylabel("Energy Ratio per DPE Pass", color="#555")
    ax2r.tick_params(axis="y", labelcolor="#555", labelsize=7)
    ax2r.set_ylim(0, max_val * 1.3)

    handles = [Patch(facecolor=ARCH_COLORS[a], label=a) for a in ARCHS]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, 1), frameon=False)

    fig.savefig(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
