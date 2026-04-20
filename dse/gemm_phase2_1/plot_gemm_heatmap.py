#!/usr/bin/env python3
"""Phase 2.1 GEMM DSE heatmap plots.

For each workload, draw a 4x3 heatmap of analytical T_pred_cyc across the
(R, C) grid with the regime annotated as F / D / B.

Usage:
    python3 dse/gemm_phase2_1/plot_gemm_heatmap.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DSE_DIR = Path(__file__).resolve().parent
CSV_PATH = DSE_DIR / "results" / "gemm_dse_smoke.csv"
OUT_DIR = DSE_DIR / "results"

DSE_R = (128, 256, 512, 1024)
DSE_C = (64, 128, 256)

REGIME_SHORT = {
    "feed-bound": "F",
    "drain-bound": "D",
    "balanced": "B",
}


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def plot_one(wl: str, rows: list):
    wl_rows = [r for r in rows if r["workload"] == wl]
    if not wl_rows:
        print(f"  [skip] no rows for {wl}")
        return
    grid = np.full((len(DSE_R), len(DSE_C)), np.nan, dtype=float)
    regime = np.full((len(DSE_R), len(DSE_C)), "", dtype=object)
    for r in wl_rows:
        i = DSE_R.index(int(r["rows"]))
        j = DSE_C.index(int(r["cols"]))
        grid[i, j] = float(r["T_pred_cyc"])
        regime[i, j] = REGIME_SHORT.get(r["regime"], "?")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")

    for i, rsz in enumerate(DSE_R):
        for j, csz in enumerate(DSE_C):
            v = grid[i, j]
            if np.isnan(v):
                continue
            # Choose annotation colour for contrast
            txt_color = "white" if v > np.nanmean(grid) else "black"
            ax.text(j, i, f"{int(v)}\n{regime[i, j]}",
                    ha="center", va="center",
                    color=txt_color, fontsize=8)

    ax.set_xticks(range(len(DSE_C)))
    ax.set_xticklabels([f"C={c}" for c in DSE_C])
    ax.set_yticks(range(len(DSE_R)))
    ax.set_yticklabels([f"R={r}" for r in DSE_R])
    ax.set_xlabel("Crossbar cols (C)")
    ax.set_ylabel("Crossbar rows (R)")

    r0 = wl_rows[0]
    ax.set_title(f"{wl}  (M={r0['M']}, K={r0['K']}, N={r0['N']})\n"
                 f"Analytical T_pred cycles, F=feed-/D=drain-/B=balanced",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="T_pred (cycles)")
    fig.tight_layout()
    out_path = OUT_DIR / f"gemm_heatmap_{wl}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    rows = load_rows()
    workloads = sorted({r["workload"] for r in rows})
    print(f"Loaded {len(rows)} rows, {len(workloads)} workloads: {workloads}")
    for wl in workloads:
        plot_one(wl, rows)


if __name__ == "__main__":
    main()
