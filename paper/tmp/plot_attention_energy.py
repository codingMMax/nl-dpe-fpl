#!/usr/bin/env python3
"""Attention Head Energy Breakdown: NL-DPE vs Azure-Lily.

Two plots for paper introduction motivation:
1. DPE-level energy breakdown (crossbar vs ACAM vs ADC) — per DPE pass
2. End-to-end attention energy breakdown (%) — NL-DPE vs Azure-Lily

Usage:
    python paper/plot_attention_energy.py
"""

import sys, os, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "azurelily"))
sys.path.insert(0, str(ROOT / "azurelily" / "IMC"))
sys.path.insert(0, str(ROOT / "nl_dpe"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

OUTDIR = ROOT / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

R, C = 512, 128
d = 128


def plot_dpe_energy_breakdown():
    """Plot 1: Total DPE energy per pass — NL-DPE vs Azure-Lily."""

    k_bit = 8

    # NL-DPE: crossbar (analog accum) + ACAM
    e_nl_total = k_bit * 3.89 + 0.171445 * C   # 53.06 pJ
    # Azure-Lily: crossbar + digital accum + ADC (bundled)
    e_al_total = k_bit * 2.33 * C               # 2385.92 pJ

    ratio = e_al_total / e_nl_total

    fig, ax = plt.subplots(figsize=(5, 4.5))

    archs = ["NL-DPE", "Azure-Lily"]
    colors = ["#059669", "#DC2626"]
    x = np.arange(len(archs))
    bar_width = 0.45

    bars = ax.bar(x, [e_nl_total, e_al_total], bar_width,
                  color=colors, edgecolor="white", linewidth=0.8)

    # Labels inside bars
    ax.text(x[0], e_nl_total / 2,
            f"{e_nl_total:.1f} pJ\n\nCrossbar + analog\naccum + ACAM",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.text(x[1], e_al_total / 2,
            f"{e_al_total:.0f} pJ\n\nCrossbar + digital\naccum + ADC",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # Ratio annotation
    ax.annotate(f"{ratio:.0f}\u00d7",
                xy=(0.5, e_al_total * 0.85),
                fontsize=16, fontweight="bold", color="#333",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#999", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy per DPE Pass (pJ)", fontsize=11)
    ax.set_title(f"DPE Energy per Pass (8-bit, {R}\u00d7{C} crossbar)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.1, axis="y")
    ax.set_ylim(0, e_al_total * 1.15)

    out = OUTDIR / "dpe_energy_per_pass.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


def run_attention_breakdown(cfg_name, N, d, use_dpe=True, use_log=False):
    """Run attention head, return component-level energy breakdown."""
    from imc_core.config import Config
    from imc_core.imc_core import IMCCore
    from peripherals.fpga_fabric import FPGAFabric
    from peripherals.memory import MemoryModel
    from scheduler_stats.stats import Stats

    cfg = Config(str(ROOT / "azurelily" / "IMC" / "configs" / f"{cfg_name}.json"))
    cfg.rows = R; cfg.cols = C
    stats = Stats()
    mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)

    bd = {"crossbar": 0, "acam_or_adc": 0, "clb": 0, "dsp": 0}
    total_e = 0

    def snapshot():
        return dict(stats.energy_breakdown)

    def accum_delta(old, new):
        for k in new:
            delta = new[k] - old.get(k, 0)
            if delta <= 0:
                continue
            if "vmm" in k or "analoge" in k:
                bd["crossbar"] += delta
            elif "digital" in k or "conversion" in k or "imc_dimm" in k:
                bd["acam_or_adc"] += delta
            elif "clb" in k:
                bd["clb"] += delta
            elif "dsp" in k:
                bd["dsp"] += delta

    # Projections
    for _ in range(3):
        old = snapshot()
        _, ei = imc.run_gemm(N, d, d) if use_dpe else fpga.gemm_dsp(N, d, d)
        total_e += ei
        accum_delta(old, snapshot())

    # DIMM-1
    old = snapshot()
    if use_log and cfg.analoge_nonlinear_support:
        _, ei = fpga.gemm_log(N, d, N, n_parallel_dpes=N)
    else:
        _, ei = fpga.gemm_dsp(N, d, N)
    total_e += ei
    accum_delta(old, snapshot())

    # Softmax
    if use_log and cfg.analoge_nonlinear_support:
        _, ee = imc.dimm_nonlinear(N, "exp", False)
        total_e += ee * N; bd["acam_or_adc"] += ee * N
        _, _, es, _, em = fpga.norm_fpga(N)
        _, el2 = imc.dimm_nonlinear(1, "log", False)
        total_e += (es + el2 + em) * N
        bd["clb"] += es * N; bd["acam_or_adc"] += el2 * N; bd["dsp"] += em * N
    else:
        _, ee = fpga.exp_fpga(N)
        total_e += ee * N; bd["clb"] += ee * N
        _, en, es, ei_inv, em = fpga.norm_fpga(N)
        total_e += en * N
        bd["clb"] += (es + ei_inv) * N; bd["dsp"] += em * N

    # DIMM-2
    old = snapshot()
    if use_log and cfg.analoge_nonlinear_support:
        _, el3 = imc.dimm_nonlinear(N, "log", False)
        total_e += el3 * N; bd["acam_or_adc"] += el3 * N
        _, ei = fpga.gemm_log(N, N, d, n_parallel_dpes=d)
        total_e += ei
        new = snapshot()
        for k in new:
            delta = new[k] - old.get(k, 0)
            if delta <= 0: continue
            if "imc_dimm" in k: bd["acam_or_adc"] += delta
            elif "clb" in k: bd["clb"] += delta
    else:
        _, ei = fpga.gemm_dsp(N, N, d)
        total_e += ei
        new = snapshot()
        for k in new:
            delta = new[k] - old.get(k, 0)
            if delta <= 0: continue
            if "dsp" in k: bd["dsp"] += delta

    return total_e, bd


def plot_attention_breakdown(N_show=128):
    """Plot 2: End-to-end attention energy breakdown (%), NL-DPE vs Azure-Lily only.
    Excludes memory. Shows ratio between the two."""

    e_nl, bd_nl = run_attention_breakdown("nl_dpe", N_show, d, use_dpe=True, use_log=True)
    e_al, bd_al = run_attention_breakdown("azure_lily", N_show, d, use_dpe=True, use_log=False)

    # Exclude memory — compute only
    compute_nl = sum(bd_nl.values())
    compute_al = sum(bd_al.values())

    archs = ["NL-DPE", "Azure-Lily"]
    computes = [compute_nl, compute_al]
    breakdowns = [bd_nl, bd_al]

    components = ["crossbar", "acam_or_adc", "clb", "dsp"]
    comp_labels = ["DPE Crossbar", "ACAM / ADC", "FPGA CLB", "FPGA DSP"]
    comp_colors = ["#059669", "#DC2626", "#3B82F6", "#F59E0B"]

    fig, ax = plt.subplots(figsize=(5, 5))

    x = np.arange(len(archs))
    bar_width = 0.45
    bottoms = np.zeros(len(archs))

    for comp, label, color in zip(components, comp_labels, comp_colors):
        vals_pct = np.array([bd[comp] / c * 100 for bd, c in zip(breakdowns, computes)])

        ax.bar(x, vals_pct, bar_width, bottom=bottoms,
               color=color, label=label, edgecolor="white", linewidth=0.8)

        for j, v in enumerate(vals_pct):
            if v > 5:
                y_pos = bottoms[j] + v / 2
                label_text = f"{v:.0f}%" if v >= 1 else f"{v:.1f}%"
                ax.text(x[j], y_pos, label_text,
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if v > 20 else "black")

        bottoms += vals_pct

    # Ratio annotation
    ratio = compute_al / compute_nl
    mid_y = 50
    ax.annotate(f"{ratio:.1f}\u00d7", xy=(0.5, mid_y),
                fontsize=14, fontweight="bold", color="#333",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#999", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=11, fontweight="bold")
    ax.set_ylabel("Compute Energy Breakdown (%)", fontsize=11)
    ax.set_title(f"Attention Head Compute Energy (N={N_show}, d={d})\n"
                 f"Excluding memory — DPE + FPGA fabric only",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=True)
    ax.grid(True, alpha=0.1, axis="y")

    out = OUTDIR / "attention_compute_breakdown.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # Print table
    print(f"\nCompute Energy Breakdown at N={N_show}, d={d} (excl. memory):")
    print(f"{'Component':<16} | {'NL-DPE':>12} {'Azure-Lily':>12}")
    print("-" * 45)
    for comp, label in zip(components, comp_labels):
        pcts = [bd[comp] / c * 100 for bd, c in zip(breakdowns, computes)]
        print(f"{label:<16} | {pcts[0]:>10.1f}% {pcts[1]:>10.1f}%")
    print(f"{'TOTAL (nJ)':<16} | {compute_nl/1e3:>11.1f} {compute_al/1e3:>11.1f}")
    print(f"{'Ratio':<16} | {'1.0x':>12} {ratio:>11.1f}x")


if __name__ == "__main__":
    print("Plot 1: DPE-level energy (crossbar vs ACAM vs ADC)")
    plot_dpe_energy_breakdown()

    print("\nPlot 2: Attention end-to-end compute breakdown")
    plot_attention_breakdown(N_show=128)

    print("\nDone.")
