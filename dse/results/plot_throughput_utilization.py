#!/usr/bin/env python3
"""Round 2 Prototype: Throughput scalability heatmap.

Measures actual throughput scalability as DPE replaces DSP and CLB resources.

For each (DSP%, CLB%) configuration, VTR gives us P replicas at Fmax:
  - T(d,c) = P × Fmax          (actual throughput, from VTR)
  - T₀     = 1 × f₀            (baseline: P=1 at measured f₀)
  - Speedup = T/T₀ = P×f/f₀   (actual vs baseline)
  - Ideal   = P                 (linear scaling, no degradation)
  - Efficiency = (T/T₀)/P = f/f₀  (how close to linear)

Heatmap:
  X-axis: CLB replacement ratio
  Y-axis: DSP replacement ratio
  Color:  Throughput efficiency = (T/T₀) / P
  Annotations: P (replicas), T/T₀ (measured speedup), efficiency

Input:  dse/results/round2_proto_results.csv
Output: dse/results/round2_throughput_utilization.pdf
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent
CSV_PATH = RESULTS_DIR / "round2_proto_results.csv"


def main():
    # ── Load data ────────────────────────────────────────────────────────
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    data = defaultdict(dict)
    for r in rows:
        d = float(r['dsp_ratio'])
        c = float(r['clb_ratio'])
        data[d][c] = r

    dsp_ratios = sorted(data.keys())
    clb_ratios = sorted(set(float(r['clb_ratio']) for r in rows))

    # Drop c=80% since it's identical to c=60% (wc column saturation)
    clb_ratios = [c for c in clb_ratios if c <= 0.6]

    # ── Baseline: P=1 (d=20% c=0%), measured by VTR ─────────────────────
    ref_row = data[0.20][0.0]
    f0 = float(ref_row['fmax_mhz'])
    t0 = 1 * f0  # T₀ = 1 replica × f₀
    print(f"Baseline (measured by VTR): d=20% c=0%, P=1, f\u2080 = {f0:.1f} MHz")
    print(f"T\u2080 = 1 \u00d7 {f0:.1f} = {t0:.1f} MHz (proportional to inferences/s)\n")

    # ── Build matrices ───────────────────────────────────────────────────
    n_dsp = len(dsp_ratios)
    n_clb = len(clb_ratios)
    eff_matrix = np.zeros((n_dsp, n_clb))
    speedup_matrix = np.zeros((n_dsp, n_clb))
    p_matrix = np.zeros((n_dsp, n_clb), dtype=int)

    for i, d in enumerate(dsp_ratios):
        for j, c in enumerate(clb_ratios):
            if c in data[d]:
                r = data[d][c]
                p = int(r['P'])
                fn = float(r['fmax_mhz'])
                tn = p * fn               # actual throughput
                speedup = tn / t0          # T_n / T_0
                efficiency = speedup / p   # (T_n/T_0) / P = fn / f0
                eff_matrix[i, j] = efficiency
                speedup_matrix[i, j] = speedup
                p_matrix[i, j] = p

    # ── Print summary ────────────────────────────────────────────────────
    print("=" * 90)
    print("Throughput Scalability: T\u2080 = 1\u00d7f\u2080,  T\u2099 = n\u00d7f\u2099,  "
          "Speedup = T\u2099/T\u2080,  Ideal = P")
    print("=" * 90)
    print(f"{'d%':>5}  {'c%':>5}  {'P=n':>5}  {'f\u2099 (MHz)':>9}  "
          f"{'T\u2099/T\u2080':>8}  {'Ideal=n':>8}  {'Eff':>6}  {'Note':>12}")
    print("-" * 90)
    for i, d in enumerate(dsp_ratios):
        for j, c in enumerate(clb_ratios):
            if c in data[d]:
                r = data[d][c]
                p = p_matrix[i, j]
                fn = float(r['fmax_mhz'])
                sp = speedup_matrix[i, j]
                ef = eff_matrix[i, j]
                note = ""
                if ef < 0.80:
                    note = "<< degraded"
                elif ef > 0.95:
                    note = "~ linear"
                print(f"{d:>5.0%}  {c:>5.0%}  {p:>5}  {fn:>9.1f}  "
                      f"{sp:>8.2f}  {p:>8}  {ef:>6.1%}  {note:>12}")

    # ── Plot ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    # Color: throughput efficiency, centered at 0.85
    norm = TwoSlopeNorm(vmin=0.60, vcenter=0.85, vmax=1.05)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(eff_matrix, cmap=cmap, norm=norm, aspect="auto",
                   origin="lower")

    # Annotate each cell: P, measured speedup T/T0, efficiency
    for i in range(n_dsp):
        for j in range(n_clb):
            p = p_matrix[i, j]
            sp = speedup_matrix[i, j]
            ef = eff_matrix[i, j]
            text_color = "white" if ef < 0.72 else "black"
            ax.text(j, i,
                    f"n={p}  gain={sp:.1f}\u00d7\n"
                    f"util={ef:.0%}",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color=text_color,
                    linespacing=1.4)

    # Axis labels
    ax.set_xticks(range(n_clb))
    ax.set_xticklabels([f"{c:.0%}" for c in clb_ratios])
    ax.set_yticks(range(n_dsp))
    ax.set_yticklabels([f"{d:.0%}" for d in dsp_ratios])
    ax.set_xlabel("CLB Replacement Ratio")
    ax.set_ylabel("DSP Replacement Ratio")

    ax.set_title("DPE Throughput Scalability\n"
                 "(fc_2048_256, 512\u00d7128 config)",
                 fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Utilization", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Footnote: explain n
    fig.text(0.01, 0.01, "n = GEMV replicas;  gain = T\u2099/T\u2080;  "
             "util = gain/n (ideal = 100%)",
             fontsize=7.5, color="gray", style="italic")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    pdf_path = RESULTS_DIR / "round2_throughput_utilization.pdf"
    fig.savefig(pdf_path)
    print(f"\nSaved: {pdf_path}")


if __name__ == "__main__":
    main()
