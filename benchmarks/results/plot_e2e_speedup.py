#!/usr/bin/env python3
"""End-to-end speedup: NL-DPE Proposed vs Azure-Lily for CNN and BERT-Tiny.

Inputs:
  - benchmarks/results/cnn_benchmark_results.csv
  - benchmarks/results/bert_tiny_final_results.csv
Output:
  - benchmarks/results/e2e_speedup.pdf
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path(__file__).resolve().parent
CNN_CSV = RESULTS_DIR / "cnn_benchmark_results.csv"
BERT_CSV = RESULTS_DIR / "bert_tiny_final_results.csv"

COLOR = "#059669"


def _geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def load_cnn_speedup():
    rows = list(csv.DictReader(open(CNN_CSV)))
    lat = {}
    for r in rows:
        model = r["model"]
        arch = r["arch"]
        if model not in ("ResNet-9", "VGG-11"):
            continue
        if arch not in ("proposed", "azurelily"):
            continue
        lat[(model, arch)] = float(r["latency_ns"])

    speedups = []
    for model in ("ResNet-9", "VGG-11"):
        p = lat.get((model, "proposed"), 0.0)
        a = lat.get((model, "azurelily"), 0.0)
        speedups.append(a / p if p > 0 else 0.0)

    return _geomean(speedups)


def load_bert_speedup():
    rows = list(csv.DictReader(open(BERT_CSV)))
    lat = {}
    for r in rows:
        arch = r["arch"]
        if arch not in ("proposed", "azurelily"):
            continue
        lat[arch] = float(r["latency_ns"])

    p = lat.get("proposed", 0.0)
    a = lat.get("azurelily", 0.0)
    return a / p if p > 0 else 0.0


def main():
    cnn_speedup = load_cnn_speedup()
    bert_speedup = load_bert_speedup()

    labels = ["CNN (geomean)", "BERT-Tiny"]
    values = [cnn_speedup, bert_speedup]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=COLOR, width=0.55, zorder=3)
    for i, b in enumerate(bars):
        v = values[i]
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.03,
                f"{v:.2f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=1.0, color="#DC2626", linewidth=1, linestyle="--", alpha=0.5, zorder=2)
    ax.set_ylabel("Speedup vs Azure-Lily (latency)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0, top=max(values) * 1.25)
    ax.grid(True, axis="y", alpha=0.12, zorder=0)

    out = RESULTS_DIR / "e2e_speedup.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
