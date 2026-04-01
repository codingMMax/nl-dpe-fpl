#!/usr/bin/env python3
"""K-Identity Mapping Analysis.

Panel (a): End-to-end BERT-Tiny energy: DIMM vs Proj+FFN+Other proportion
           K=2 (Proposed) vs Azure-Lily, stacked bar across sequence lengths.
Panel (b): Complete DIMM energy efficiency vs sequence length (K=2, K=4).

Output: paper/figures/benchmarks/k_identity.pdf
"""
import math
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style_constants import (apply_style, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT,
                              BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA)
apply_style()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "k_identity.pdf"

sys.path.insert(0, str(ROOT / "azurelily"))
sys.path.insert(0, str(ROOT / "azurelily" / "IMC"))

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.attention import attention_model
from models.bert_tiny import bert_tiny_model

d_head = 64
seq_lens = [1024, 2048, 4096, 6144, 8192]

# VTR ground truth resources
VTR_AVAILABLE = {
    "proposed":  {"DSPs": 4,   "CLBs": 453,  "BRAMs": 172, "DPEs": 274},
    "al_like":   {"DSPs": 4,   "CLBs": 441,  "BRAMs": 172, "DPEs": 78},
    "azurelily": {"DSPs": 326, "CLBs": 188,  "BRAMs": 16,  "DPEs": 18},
    "baseline":  {"DSPs": 312, "CLBs": 121,  "BRAMs": 0,   "DPEs": 0},
}


def _functional_dpes(rows, cols):
    D_MODEL, D_FF = 128, 512
    per_block = (4 * math.ceil(D_MODEL/rows) * math.ceil(D_MODEL/cols)
                 + math.ceil(D_MODEL/rows) * math.ceil(D_FF/cols)
                 + math.ceil(D_FF/rows) * math.ceil(D_MODEL/cols))
    return per_block * 2


# ── BERT-Tiny breakdown (for panel a) ─────────────────────────────────────

def run_bert_breakdown(cfg_file, R, C, fmax, N, avail_key):
    """Run BERT-Tiny, return {DIMM: energy, non_DIMM: energy}."""
    cfg = Config(str(ROOT / "azurelily" / "IMC" / "configs" / cfg_file))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    avail = VTR_AVAILABLE.get(avail_key, {})
    if avail.get("DSPs") is not None: cfg.total_dsp = avail["DSPs"]
    if avail.get("CLBs") is not None: cfg.total_clb = avail["CLBs"]
    if avail.get("BRAMs") is not None: cfg.total_mem = avail["BRAMs"]
    dpe_used = avail.get("DPEs", 0)
    cfg.total_dimm_dpes = max(0, dpe_used - _functional_dpes(R, C))

    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)

    md, _ = bert_tiny_model(1, 1, N, 128, False, False)
    groups = defaultdict(float)

    def run_group(layers, group_name):
        e_before = sum(stats.energy_breakdown.values())
        for layer in layers:
            sched.run_layer(layer)
        groups[group_name] += sum(stats.energy_breakdown.values()) - e_before

    run_group(md["embedding"], "Other")
    for block in md["blocks"]:
        run_group(block["qkv_proj"], "Proj+FFN")
        for hi in range(block["num_heads"]):
            run_group(block["head_attention"], "DIMM")
        run_group([block["post_attn"][0]], "Proj+FFN")
        run_group(block["post_attn"][1:], "Other")
        run_group(block["ffn"][:2], "Proj+FFN")
        run_group(block["ffn"][2:], "Other")

    return {"DIMM": groups["DIMM"], "Proj+FFN+Other": groups["Proj+FFN"] + groups["Other"]}


# ── Attention DIMM (for panel b) ──────────────────────────────────────────

def run_dimm_total(cfg_file, R, C, N, avail_key):
    """Run attention head, return total DIMM energy."""
    cfg = Config(str(ROOT / "azurelily" / "IMC" / "configs" / cfg_file))
    cfg.rows = R; cfg.cols = C; cfg.freq = 200
    avail = VTR_AVAILABLE.get(avail_key, {})
    if avail.get("DSPs") is not None: cfg.total_dsp = avail["DSPs"]
    if avail.get("CLBs") is not None: cfg.total_clb = avail["CLBs"]
    dpe_used = avail.get("DPEs", 0)
    cfg.total_dimm_dpes = max(0, dpe_used - _functional_dpes(R, C))
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    layers, _ = attention_model(1, 1, N, d_head, False, False)
    for layer in layers:
        sched.run_layer(layer)
    return sum(stats.energy_breakdown.values())


# ── Configs ────────────────────────────────────────────────────────────────

BERT_CONFIGS = [
    ("K=2 (Proposed-1)", "nl_dpe.json",     1024, 128, 135.7, "proposed"),
    ("Azure-Lily",       "azure_lily.json",  512, 128,  45.3, "azurelily"),
]

DIMM_CONFIGS = [
    ("K=2",  "nl_dpe.json",   1024, 128, "proposed"),
    ("K=4",  "nl_dpe.json",   1024, 256, "al_like"),
    ("DSP",  "baseline.json",    1,   1, "baseline"),
]


# ── Collect data ──────────────────────────────────────────────────────────

bert_data = {}
for label, cfg_file, R, C, fmax, avail_key in BERT_CONFIGS:
    for N in seq_lens:
        bert_data[(N, label)] = run_bert_breakdown(cfg_file, R, C, fmax, N, avail_key)

dimm_data = {}
dimm_seq_lens = [1024, 2048, 4096, 6144, 8192]
for label, cfg_file, R, C, avail_key in DIMM_CONFIGS:
    for N in dimm_seq_lens:
        dimm_data[(N, label)] = run_dimm_total(cfg_file, R, C, N, avail_key)


# ── Plot ──────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
fig.subplots_adjust(wspace=0.35)

# Panel labels
ax1.text(-0.12, 1.08, '(a)', transform=ax1.transAxes, fontsize=10,
         fontweight='bold', va='top')
ax2.text(-0.12, 1.08, '(b)', transform=ax2.transAxes, fontsize=10,
         fontweight='bold', va='top')

# ── Panel (a): DIMM vs Proj+FFN+Other — side by side ─────────────────────
n_groups = len(seq_lens)
bar_w = 0.35
# Architecture-colored DIMM bars for instant visual distinction
arch_dimm_color = {"K=2": "#3B82F6", "AL": "#F97316"}   # blue vs orange
non_dimm_color = "#E2E8F0"                                # light gray

for bi, (label, *_) in enumerate(BERT_CONFIGS):
    short = "K=2" if "K=2" in label else "AL"
    offset = (bi - 0.5) * bar_w

    dimm_pcts = []
    non_dimm_pcts = []
    for N in seq_lens:
        d = bert_data[(N, label)]
        total = d["DIMM"] + d["Proj+FFN+Other"]
        dimm_pcts.append(d["DIMM"] / total * 100)
        non_dimm_pcts.append(d["Proj+FFN+Other"] / total * 100)

    x = np.arange(n_groups) + offset
    ax1.bar(x, dimm_pcts, bar_w, color=arch_dimm_color[short],
            edgecolor='white', linewidth=0.5, zorder=3,
            label=f'{short} DIMM')
    ax1.bar(x, non_dimm_pcts, bar_w, bottom=dimm_pcts,
            color=non_dimm_color,
            edgecolor='white', linewidth=0.5, zorder=3,
            label='Proj+FFN+Other' if bi == 0 else None)

    # Annotate DIMM %
    for i, (dp, ndp) in enumerate(zip(dimm_pcts, non_dimm_pcts)):
        ax1.text(x[i], dp / 2, f'{dp:.0f}%', ha='center', va='center',
                 fontsize=5, fontweight='bold', color='white', zorder=5)
        if ndp > 10:
            ax1.text(x[i], dp + ndp / 2, f'{ndp:.0f}%', ha='center', va='center',
                     fontsize=5, fontweight='bold', color='#475569', zorder=5)

ax1.set_xticks(np.arange(n_groups))
ax1.set_xticklabels([str(N) for N in seq_lens])
ax1.set_xlabel('Sequence Length (N)')
ax1.set_ylabel('Energy Proportion (%)')
ax1.set_ylim(0, 108)
ax1.legend(fontsize=6, loc='upper right', ncol=1, framealpha=0.9)
ax1.grid(True, alpha=0.08, axis='y', zorder=0)

# ── Panel (b): Complete DIMM energy efficiency vs N ───────────────────────
for label, cfg_file, R, C, avail_key in DIMM_CONFIGS:
    if label == "DSP":
        continue
    color = '#3B82F6' if label == 'K=2' else '#10B981'
    marker = 'o' if label == 'K=2' else '^'

    ratios = []
    for N in dimm_seq_lens:
        e_nl = dimm_data[(N, label)]
        e_dsp = dimm_data[(N, "DSP")]
        ratios.append(e_dsp / e_nl if e_nl > 0 else 0)

    ax2.plot(dimm_seq_lens, ratios, color=color, linewidth=2, marker=marker,
             markersize=5, markeredgecolor='white', markeredgewidth=0.8,
             label=label)

    # Annotate first and last only — stagger to avoid overlap at N=1024
    for i, (N, r) in enumerate(zip(dimm_seq_lens, ratios)):
        if i == 0:
            y_off = 8
            ax2.annotate(f'{r:.2f}\u00d7', xy=(N, r), xytext=(0, y_off),
                         textcoords='offset points', ha='center', va='bottom',
                         fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                         color=color)
        elif i == len(dimm_seq_lens) - 1:
            # Stagger: K=2 above-left, K=4 below-left (keep inside plot area)
            y_off = 8 if label == 'K=2' else -14
            ax2.annotate(f'{r:.2f}\u00d7', xy=(N, r), xytext=(-8, y_off),
                         textcoords='offset points', ha='right',
                         va='bottom' if y_off > 0 else 'top',
                         fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                         color=color)

ax2.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax2.text(1100, 1.03, 'DSP baseline', fontsize=6, color='#666', ha='right',
         fontstyle='italic')

ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('DIMM Energy Efficiency\n(norm. to DSP baseline)')
ax2.set_xticks(dimm_seq_lens)
ax2.set_xticklabels([str(n) for n in dimm_seq_lens])
ax2.set_xlim(dimm_seq_lens[0] * 0.9, dimm_seq_lens[-1] * 1.05)
ax2.set_ylim(bottom=0.9)
ax2.legend(fontsize=7, loc='upper left')
ax2.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\nBERT-Tiny Energy Breakdown:")
print(f"{'Config':>20} {'N':>5} {'DIMM%':>7} {'Other%':>7} {'Total(MpJ)':>11}")
print("-" * 55)
for N in seq_lens:
    for label, *_ in BERT_CONFIGS:
        d = bert_data[(N, label)]
        total = d["DIMM"] + d["Proj+FFN+Other"]
        short = "K=2" if "K=2" in label else "Azure-Lily"
        print(f"{short:>20} {N:>5} {d['DIMM']/total*100:>6.1f}% "
              f"{d['Proj+FFN+Other']/total*100:>6.1f}% {total/1e6:>10.3f}")
    print()

print(f"DIMM Efficiency (norm. to DSP):")
print(f"{'Config':>10} {'N':>5} {'Ratio':>8}")
print("-" * 25)
for N in dimm_seq_lens:
    for label, *_ in DIMM_CONFIGS:
        if label == "DSP":
            continue
        r = dimm_data[(N, "DSP")] / dimm_data[(N, label)]
        print(f"{label:>10} {N:>5} {r:>7.2f}×")
    print()
