#!/usr/bin/env python3
"""BERT-Tiny Efficiency: Area and Energy efficiency vs sequence length.

Input: IMC simulator (azurelily/)
Output: paper/figures/benchmarks/bert_efficiency.pdf
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
sys.path.insert(0, str(ROOT_DIR / "azurelily"))
sys.path.insert(0, str(ROOT_DIR / "azurelily" / "IMC"))

import matplotlib.pyplot as plt
from style_constants import (apply_style_sc, ARCH_COLORS, ARCH_MARKERS, ARCH_LINESTYLES,
                              BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE_SC, ANNOT_FONTWEIGHT_SC, FPGA_AREA_MM2)
apply_style_sc()

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler

OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "bert_efficiency.pdf"


def run_bert(cfile, R, C, fmax, N):
    cfg = Config(str(ROOT_DIR / "azurelily" / "IMC" / "configs" / f"{cfile}.json"))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats); fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    from models.bert_tiny import bert_tiny_model
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
    return sum(stats.energy_breakdown.values()), sum(stats.latency_breakdown.values())


SEQ_LENS = [128, 256, 512, 1024]
CONFIGS = [
    ("Proposed", "nl_dpe", 1024, 128, 141.5),
    ("AL-like", "nl_dpe", 1024, 256, 140.1),
    ("Azure-Lily", "azure_lily", 512, 128, 124.9),
]

# Collect data
data = {}
for N in SEQ_LENS:
    for cname, cfile, R, C, fmax in CONFIGS:
        e, l = run_bert(cfile, R, C, fmax, N)
        data[(N, cname)] = {"energy": e, "latency": l}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
fig.subplots_adjust(wspace=0.28)

for ax, metric_fn, ylabel in [
    (ax1, lambda N, c: (1e9 / data[(N, c)]["latency"]) / FPGA_AREA_MM2,
     "Normalized Throughput/mm\u00b2"),
    (ax2, lambda N, c: 1e12 / data[(N, c)]["energy"],
     "Normalized Inference/J"),
]:
    for cname in ["Proposed", "AL-like", "Azure-Lily"]:
        ys = []
        for N in SEQ_LENS:
            val = metric_fn(N, cname)
            bl = metric_fn(N, "Azure-Lily")
            ys.append(val / bl)

        ax.plot(SEQ_LENS, ys, color=ARCH_COLORS[cname], linewidth=1.5,
                linestyle=ARCH_LINESTYLES[cname],
                marker=ARCH_MARKERS[cname], markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=cname)

        if "Proposed" in cname:
            for i, (N, r) in enumerate(zip(SEQ_LENS, ys)):
                # Place annotations above the line, offset to avoid marker
                ax.annotate(f"{r:.2f}\u00d7", xy=(N, r),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=ANNOT_FONTSIZE_SC, fontweight=ANNOT_FONTWEIGHT_SC,
                            color=ARCH_COLORS[cname])

    ax.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
               alpha=BASELINE_ALPHA)
    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels([str(n) for n in SEQ_LENS])
    # Ensure headroom for annotations
    all_proposed = [metric_fn(N, "Proposed") / metric_fn(N, "Azure-Lily")
                    for N in SEQ_LENS]
    ax.set_ylim(bottom=0.8, top=max(all_proposed) * 1.15)
    ax.set_xlim(105, 1250)
    ax.grid(True, alpha=0.1)

# Shared legend above both panels
from matplotlib.patches import Patch
handles = [Patch(facecolor=ARCH_COLORS[a], label=a)
           for a in ["Proposed", "AL-like", "Azure-Lily"]]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, 1), frameon=False)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
