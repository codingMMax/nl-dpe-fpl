#!/usr/bin/env python3
"""Combined efficiency bar chart: all 3 models normalized to Azure-Lily.

Includes Geomean. Used for the main results overview figure.

Input:  benchmarks/results/imc_benchmark_results.csv
Output: benchmarks/results/efficiency_normalized.pdf
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
AREA_MM2 = 150 * 150 * 2239 / 1e6

MODELS = ["ResNet-9", "VGG-11", "BERT-Tiny"]
ARCHS = ["NL-DPE Proposed", "NL-DPE AL-Matched", "Azure-Lily"]
COLORS = {"NL-DPE Proposed": "#059669", "NL-DPE AL-Matched": "#2563EB", "Azure-Lily": "#94A3B8"}
HATCHES = {"NL-DPE Proposed": "", "NL-DPE AL-Matched": "//", "Azure-Lily": ""}


def main():
    imc = {r['label']: r for r in csv.DictReader(open(CSV_PATH))}

    data = {}
    for label, r in imc.items():
        m = label.split(" NL-DPE")[0].split(" Azure")[0]
        a = "NL-DPE Proposed" if "Proposed" in label else ("NL-DPE AL-Matched" if "AL-Matched" in label else "Azure-Lily")
        e, l = float(r['total_energy_pj']), float(r['total_latency_ns'])
        data[(m, a)] = {'tput_mm2': (1e9 / l) / AREA_MM2 if l > 0 else 0, 'tput_j': 1e12 / e if e > 0 else 0}

    norm = {}
    for m in MODELS:
        for a in ARCHS:
            norm[(m, a)] = {k: data[(m, a)][k] / data[(m, "Azure-Lily")][k] if data[(m, "Azure-Lily")][k] > 0 else 0
                           for k in ['tput_mm2', 'tput_j']}

    cats = MODELS + ["Geomean"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.20)
    x = np.arange(len(cats))
    w = 0.24

    for ax, metric, title in [(ax1, 'tput_mm2', '(a) Area Efficiency (inf/s/mm²)'),
                               (ax2, 'tput_j', '(b) Energy Efficiency (inf/J)')]:
        mv = 0
        av = {}
        for i, a in enumerate(ARCHS):
            pm = [norm[(m, a)][metric] for m in MODELS]
            gm = math.exp(sum(math.log(v) for v in pm if v > 0) / len([v for v in pm if v > 0])) if any(v > 0 for v in pm) else 0
            vals = pm + [gm]
            mv = max(mv, max(vals))
            av[a] = vals
            for j, v in enumerate(vals):
                ig = j == len(vals) - 1
                ax.bar(x[j] + i * w, v, w, color=COLORS[a], hatch=HATCHES[a],
                       edgecolor="black" if ig else ("white" if a != "Azure-Lily" else "#94A3B8"),
                       linewidth=1.5 if ig else 0.8, zorder=3)

        for i, a in enumerate(ARCHS):
            if a == "Azure-Lily":
                continue
            for j, v in enumerate(av[a]):
                ax.text(x[j] + i * w, v + mv * 0.02, f"{v:.1f}×", ha='center', va='bottom',
                        fontsize=7, fontweight='bold', color=COLORS[a], rotation=45)

        ax.axhline(y=1.0, color='#DC2626', linewidth=1, linestyle='--', alpha=0.4, zorder=2)
        ax.axvline(x=x[-1] - 0.35, color='#CBD5E1', linewidth=1, linestyle='-', alpha=0.6, zorder=1)
        ax.set_ylabel("Normalized (Azure-Lily = 1.0×)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x + w)
        labels = ax.set_xticklabels(cats, fontsize=9)
        labels[-1].set_fontweight('bold')
        ax.grid(True, alpha=0.1, axis="y", zorder=0)
        ax.set_ylim(bottom=0, top=mv * 1.45)

    fig.legend(handles=[Patch(facecolor=COLORS[a], hatch=HATCHES[a],
               label=f"{a} ({'1024×128' if 'Proposed' in a else '1024×256' if 'AL' in a else '512×128, baseline'})")
               for a in ARCHS], loc='upper center', ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.03), frameon=False)

    out = RESULTS_DIR / "efficiency_normalized.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
