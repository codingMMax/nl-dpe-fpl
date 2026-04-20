#!/usr/bin/env python3
"""Phase 2.1 GEMM DSE Pareto plots.

Reads ``dse/gemm_phase2_1/results/gemm_dse_smoke.csv`` and produces:

- ``gemm_pareto_{workload}.pdf`` — one per workload (x = CLB count,
  y = latency_ns, markers coloured by regime).
- ``gemm_pareto_combined.pdf`` — 2x2 subplot combining all four
  workloads.

Usage:
    python3 dse/gemm_phase2_1/plot_gemm_pareto.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DSE_DIR = Path(__file__).resolve().parent
CSV_PATH = DSE_DIR / "results" / "gemm_dse_smoke.csv"
OUT_DIR = DSE_DIR / "results"

REGIME_COLOR = {
    "feed-bound": "#1f77b4",   # blue
    "drain-bound": "#d62728",  # red
    "balanced": "#2ca02c",     # green
}


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def plot_one(ax, rows, workload):
    wl_rows = [r for r in rows if r["workload"] == workload]
    xs = [int(r["clb_count"]) for r in wl_rows]
    ys = [float(r["latency_ns"]) for r in wl_rows]
    labels = [r["config"] for r in wl_rows]
    regimes = [r["regime"] for r in wl_rows]
    colors = [REGIME_COLOR.get(rg, "#888888") for rg in regimes]

    ax.scatter(xs, ys, c=colors, s=52, edgecolors="black", linewidths=0.5,
               zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=6,
                    textcoords="offset points", xytext=(4, 3))

    # Plot analytical prediction overlay
    xs_pred = xs
    ys_pred = [float(r["T_pred_ns"]) for r in wl_rows]
    ax.scatter(xs_pred, ys_pred, marker="x", c="grey", s=25,
               alpha=0.7, zorder=2, label="analytical T_pred")

    # Highlight the sweet spot (lowest sim latency)
    if wl_rows:
        best = min(wl_rows, key=lambda r: float(r["latency_ns"]))
        ax.scatter(
            [int(best["clb_count"])], [float(best["latency_ns"])],
            facecolors="none", edgecolors="gold", s=180, linewidths=2,
            zorder=4, label=f"sweet spot {best['config']}",
        )

    if wl_rows:
        r0 = wl_rows[0]
        ax.set_title(
            f"{workload}  (M={r0['M']}, K={r0['K']}, N={r0['N']})",
            fontsize=10,
        )
    ax.set_xlabel("CLB count")
    ax.set_ylabel("Sim latency (ns)")
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=7, loc="best")


def main():
    rows = load_rows()
    workloads = sorted({r["workload"] for r in rows})
    print(f"Loaded {len(rows)} rows, {len(workloads)} workloads: {workloads}")

    # Per-workload
    for wl in workloads:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_one(ax, rows, wl)
        # Legend handles for regimes
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                   markeredgecolor="black", markersize=8, label=rg)
            for rg, col in REGIME_COLOR.items()
        ]
        handles.append(Line2D([0], [0], marker="x", color="grey",
                              linestyle="", markersize=7, label="T_pred (analytical)"))
        ax.legend(handles=handles, fontsize=7, loc="best")
        fig.tight_layout()
        out_path = OUT_DIR / f"gemm_pareto_{wl}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  wrote {out_path}")

    # Combined 2×2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, wl in zip(axes.flat, workloads):
        plot_one(ax, rows, wl)
    fig.suptitle("Phase 2.1 GEMM Pareto — CLB vs latency (48 configs)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / "gemm_pareto_combined.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
