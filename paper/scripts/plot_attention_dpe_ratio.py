#!/usr/bin/env python3
"""Attention Head: DPE energy contribution vs sequence length.

Comparable to Azure-Lily's Figure R-2 from TACO reviewer response.
Shows NL-DPE keeps DPE contribution at ~50% while Azure-Lily drops to ~1%.

Output: benchmarks/results/attention_dpe_ratio_vs_seqlen.pdf
"""
import sys, math
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / "azurelily"))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / "azurelily" / "IMC"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.attention import attention_model

plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.05})

RESULTS_DIR = Path(__file__).resolve().parent

SEQ_LENS = [64, 128, 256, 512, 1024, 2048]
D_HEAD = 64  # standard transformer d_head

CONFIGS = [
    ("NL-DPE (dual-identity)", "nl_dpe", 1024, 128),
    ("Azure-Lily", "azure_lily", 512, 128),
]
COLORS = {"NL-DPE (dual-identity)": "#059669", "Azure-Lily": "#EF4444"}
MARKERS = {"NL-DPE (dual-identity)": "o", "Azure-Lily": "D"}
STYLES = {"NL-DPE (dual-identity)": "-", "Azure-Lily": "--"}


def run_attention(cfile, R, C, N):
    cfg = Config(f"azurelily/IMC/configs/{cfile}.json")
    cfg.rows = R; cfg.cols = C; cfg.freq = 150
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    layers, _ = attention_model(1, 1, N, D_HEAD, False, False)
    for layer in layers:
        sched.run_layer(layer)
    e_bd = dict(stats.energy_breakdown)
    total_e = sum(e_bd.values())
    e_dpe = sum(v for k, v in e_bd.items() if "imc" in k)
    return total_e, e_dpe


def main():
    results = {}
    for N in SEQ_LENS:
        for cname, cfile, R, C in CONFIGS:
            total_e, e_dpe = run_attention(cfile, R, C, N)
            results[(N, cname)] = {
                "total_e": total_e, "dpe_e": e_dpe,
                "dpe_pct": e_dpe / total_e * 100 if total_e > 0 else 0,
            }

    # Print table
    print(f"{'N':>5} | {'NL-DPE DPE%':>12} {'Total(K)':>10} | {'AL DPE%':>10} {'Total(K)':>10}")
    print("-" * 55)
    for N in SEQ_LENS:
        nl = results[(N, "NL-DPE (dual-identity)")]
        al = results[(N, "Azure-Lily")]
        print(f"{N:>5} | {nl['dpe_pct']:>11.1f}% {nl['total_e']/1e3:>9.0f} | "
              f"{al['dpe_pct']:>9.1f}% {al['total_e']/1e3:>9.0f}")

    # Plot: single panel — DPE Energy %
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for cname, _, _, _ in CONFIGS:
        ys = [results[(N, cname)]["dpe_pct"] for N in SEQ_LENS]
        ax.plot(SEQ_LENS, ys, color=COLORS[cname], linewidth=2.5,
                linestyle=STYLES[cname], marker=MARKERS[cname],
                markersize=7, markeredgecolor="white", markeredgewidth=1,
                label=cname)

    ax.axhline(y=50, color="#888", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Sequence Length (N)", fontsize=10)
    ax.set_ylabel("DPE Energy Contribution (%)", fontsize=10)
    ax.set_title("Attention Head: DPE Energy Ratio vs Sequence Length\n(d_head=64)",
                 fontsize=10, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels([str(n) for n in SEQ_LENS], fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.1)

    out = RESULTS_DIR / "attention_dpe_ratio_vs_seqlen.pdf"
    fig.savefig(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
