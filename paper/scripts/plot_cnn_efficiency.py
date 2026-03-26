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
from style_constants import apply_style, ARCH_COLORS, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT, FPGA_AREA_MM2
apply_style()

CSV_PATH = SCRIPT_DIR / "imc_benchmark_results.csv"
OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "cnn_efficiency.pdf"

MODELS = ["ResNet-9", "VGG-11"]
ARCHS = ["NL-DPE Proposed", "NL-DPE AL-Matched", "Azure-Lily"]


def main():
    imc = {r['label']: r for r in csv.DictReader(open(CSV_PATH))}

    data = {}
    for label, r in imc.items():
        m = label.split(" NL-DPE")[0].split(" Azure")[0]
        if m not in MODELS:
            continue
        a = "NL-DPE Proposed" if "Proposed" in label else (
            "NL-DPE AL-Matched" if "AL-Matched" in label else "Azure-Lily")
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.subplots_adjust(wspace=0.25)

    for ax, (metric, title) in zip([ax1, ax2],
            [('tput_mm2', '(a) Area Efficiency (inf/s/mm\u00b2)'),
             ('tput_j', '(b) Energy Efficiency (inf/J)')]):
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
                            fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                            color=ARCH_COLORS[a], rotation=0)

        ax.axhline(y=1.0, color='#94A3B8', linewidth=1, linestyle=':', alpha=0.5, zorder=2)
        ax.set_ylabel("Normalized (Azure-Lily = 1.0\u00d7)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        lbls = ax.set_xticklabels(cats, fontsize=9)
        lbls[-1].set_fontweight('bold')
        ax.grid(True, alpha=0.08, axis="y", zorder=0)
        ax.set_ylim(bottom=0, top=max_val * 1.3)

    handles = [Patch(facecolor=ARCH_COLORS[a], label=a) for a in ARCHS]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.savefig(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
