#!/usr/bin/env python3
"""Transformer (BERT-Tiny): Energy efficiency vs sequence length.

Sweeps seq_len = 128, 256, 512, 1024 using IMC simulator.
Plots normalized energy efficiency (vs Azure-Lily) as line chart.

Output: benchmarks/results/transformer_efficiency_scaling.pdf
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
from models.bert_tiny import bert_tiny_model

plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.05})

RESULTS_DIR = Path(__file__).resolve().parent

SEQ_LENS = [128, 256, 512, 1024]
CONFIGS = [
    ("NL-DPE Proposed", "nl_dpe", 1024, 128, 141.5),
    ("NL-DPE AL-Matched", "nl_dpe", 1024, 256, 140.1),
    ("Azure-Lily", "azure_lily", 512, 128, 124.9),
]
LINE_COLORS = {"NL-DPE Proposed": "#059669", "NL-DPE AL-Matched": "#2563EB", "Azure-Lily": "#EF4444"}
MARKERS = {"NL-DPE Proposed": "o", "NL-DPE AL-Matched": "s", "Azure-Lily": "D"}
STYLES = {"NL-DPE Proposed": "-", "NL-DPE AL-Matched": "--", "Azure-Lily": ":"}


def run_bert_tiny(cfile, R, C, fmax, N):
    cfg = Config(f"azurelily/IMC/configs/{cfile}.json")
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    stats = Stats(); mem = MemoryModel(cfg, stats)
    ic = IMCCore(cfg, mem, stats); fpga = FPGAFabric(cfg, mem, stats, imc_core=ic)
    sched = Scheduler(cfg, stats, ic, fpga)
    md, _ = bert_tiny_model(1, 1, N, 128, False, False)
    for layer in md["embedding"]:
        sched.run_layer(layer)
    for block in md["blocks"]:
        for layer in block["qkv_proj"]:
            sched.run_layer(layer)
        for hi in range(block["num_heads"]):
            for layer in block["head_attention"]:
                sched.run_layer(layer)
        for layer in block["post_attn"]:
            sched.run_layer(layer)
        for layer in block["ffn"]:
            sched.run_layer(layer)
    return sum(stats.energy_breakdown.values())


def main():
    results = {}
    for N in SEQ_LENS:
        for cname, cfile, R, C, fmax in CONFIGS:
            results[(N, cname)] = run_bert_tiny(cfile, R, C, fmax, N)

    # Normalize to Azure-Lily at each N
    fig, ax = plt.subplots(figsize=(5, 3.2))

    for cname, _, _, _, _ in CONFIGS:
        ys = [results[(N, "Azure-Lily")] / results[(N, cname)] for N in SEQ_LENS]
        ax.plot(SEQ_LENS, ys, color=LINE_COLORS[cname], linewidth=2,
                linestyle=STYLES[cname], marker=MARKERS[cname],
                markersize=7, markeredgecolor="white", markeredgewidth=1, label=cname)

        if cname == "NL-DPE Proposed":
            for j, (N, v) in enumerate(zip(SEQ_LENS, ys)):
                ax.annotate(f"{v:.2f}×", xy=(N, v), xytext=(0, 10),
                            textcoords='offset points', ha='center',
                            fontsize=7.5, fontweight='bold', color=LINE_COLORS[cname])

    ax.axhline(y=1.0, color='#94A3B8', linewidth=1, linestyle='-', alpha=0.5, zorder=1)
    ax.text(SEQ_LENS[-1] * 1.05, 1.02, "Azure-Lily baseline", fontsize=7,
            color='#94A3B8', ha='right', fontstyle='italic')

    ax.set_xlabel("Sequence Length", fontsize=10)
    ax.set_ylabel("Energy Efficiency Ratio\n(vs Azure-Lily)", fontsize=9)
    ax.set_title("Transformer (BERT-Tiny): Scaling with Sequence Length",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels([str(n) for n in SEQ_LENS])
    ax.set_ylim(bottom=0.8, top=2.0)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.1)

    out = RESULTS_DIR / "transformer_efficiency_scaling.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
