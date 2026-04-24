#!/usr/bin/env python3
"""End-to-end speedup: NL-DPE Proposed vs Azure-Lily (CNN + BERT-Tiny).

Inputs:
  - benchmarks/results/cnn_benchmark_results.csv
  - benchmarks/results/bert_tiny_final_results.csv

Output:
  - benchmarks/results/end_to_end_speedup.pdf
"""
import csv
import math
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
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


def _load_cnn_latency():
    lat = {}
    with open(CNN_CSV, newline="") as f:
        for r in csv.DictReader(f):
            lat[(r["model"], r["arch"])] = float(r["latency_ns"])
    return lat


def _load_bert_latency():
    lat = {}
    with open(BERT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            lat[r["arch"]] = float(r["latency_ns"])
    return lat


def main():
    cnn = _load_cnn_latency()
    bert = _load_bert_latency()

    speedups = {
        "ResNet-9": cnn[("ResNet-9", "azurelily")] / cnn[("ResNet-9", "proposed")],
        "VGG-11": cnn[("VGG-11", "azurelily")] / cnn[("VGG-11", "proposed")],
        "BERT-Tiny": bert["azurelily"] / bert["proposed"],
    }

    # CNN geomean, reported in stdout for convenience.
    cnn_geo = math.exp(
        sum(math.log(speedups[m]) for m in ["ResNet-9", "VGG-11"]) / 2
    )
    print(f"CNN geomean speedup (proposed vs Azure-Lily): {cnn_geo:.3f}x")

    labels = list(speedups.keys())
    values = [speedups[k] for k in labels]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.55, color="#059669", edgecolor="white", zorder=3)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(values) * 0.03,
            f"{v:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#065F46",
        )

    ax.axhline(y=1.0, color="#EF4444", linewidth=1, linestyle="--", alpha=0.5, zorder=2)
    ax.set_ylabel("Speedup vs Azure-Lily (end-to-end latency)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(bottom=0, top=max(values) * 1.25)
    ax.set_title("End-to-End Speedup: NL-DPE Proposed vs Azure-Lily", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.1, axis="y", zorder=0)

    out = RESULTS_DIR / "end_to_end_speedup.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
