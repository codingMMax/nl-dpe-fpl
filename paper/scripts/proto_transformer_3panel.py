#!/usr/bin/env python3
"""PROTOTYPE: Transformer 3-panel — (1) DIMM fraction, (2) Speedup breakdown, (3) K-identity."""
import csv
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
sys.path.insert(0, str(ROOT_DIR / "azurelily"))
sys.path.insert(0, str(ROOT_DIR / "azurelily" / "IMC"))

from style_constants import (apply_style, ARCH_COLORS, BASELINE_COLOR,
                              BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE, ANNOT_FONTWEIGHT)
apply_style()

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.attention import attention_model
from scheduler_stats.common import configure_logging
configure_logging(False)

CSV_PATH = SCRIPT_DIR / "bert_energy_plots" / "bert_energy_sweep.csv"
OUT_PATH = SCRIPT_DIR / "proto_transformer_3panel.pdf"

# ── Panel 1 & 2 data (from CSV) ──
csv_data = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        csv_data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [1024, 2048, 4096, 6144, 8192]

dimm_frac_p = [csv_data[("Proposed", N)]["dimm_ns"] / csv_data[("Proposed", N)]["total_ns"] * 100 for N in SEQ_LENS]
dimm_frac_a = [csv_data[("Azure-Lily", N)]["dimm_ns"] / csv_data[("Azure-Lily", N)]["total_ns"] * 100 for N in SEQ_LENS]

overall_speedup = [csv_data[("Azure-Lily", N)]["total_ns"] / csv_data[("Proposed", N)]["total_ns"] for N in SEQ_LENS]
dimm_speedup = [csv_data[("Azure-Lily", N)]["dimm_ns"] / csv_data[("Proposed", N)]["dimm_ns"] for N in SEQ_LENS]
nondimm_speedup = []
for N in SEQ_LENS:
    p_nd = csv_data[("Proposed", N)]["total_ns"] - csv_data[("Proposed", N)]["dimm_ns"]
    a_nd = csv_data[("Azure-Lily", N)]["total_ns"] - csv_data[("Azure-Lily", N)]["dimm_ns"]
    nondimm_speedup.append(a_nd / p_nd)

# ── Panel 3 data (K-identity, from simulator) ──
ARCHS = {
    "proposed":  ("nl_dpe.json",     1024, 128, 133.4, 274, 4,   453, 172),
    "al_like":   ("nl_dpe.json",     1024, 256, 139.7, 78,  4,   441, 172),
    "azurelily": ("azure_lily.json", 512,  128, 127.9, 18,  326, 188, 16),
}
D_HEADS = [64, 128]
KID_SEQ = [1024, 2048, 4096, 6144, 8192]

def _functional_dpes(rows, cols):
    D_MODEL, D_FF = 128, 512
    per_block = (4 * math.ceil(D_MODEL / rows) * math.ceil(D_MODEL / cols)
                 + math.ceil(D_MODEL / rows) * math.ceil(D_FF / cols)
                 + math.ceil(D_FF / rows) * math.ceil(D_MODEL / cols))
    return per_block * 2

def run_attention_head(arch, N, d_head):
    cfg_file, R, C, fmax, dpes, dsps, clbs, brams = ARCHS[arch]
    cfg = Config(str(ROOT_DIR / "azurelily" / "IMC" / "configs" / cfg_file))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    cfg.total_dsp = dsps; cfg.total_clb = clbs; cfg.total_mem = brams
    cfg.total_dimm_dpes = max(0, dpes - _functional_dpes(R, C))
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    layers, _ = attention_model(1, 1, N, d_head, False, False)
    dimm_layers = [l for l in layers
                   if l.type in ("mac_qk", "softmax_exp", "softmax_norm", "mac_sv")]
    for layer in dimm_layers:
        sched.run_layer(layer)
    return sum(stats.energy_breakdown.values())

kid_data = {}
for d in D_HEADS:
    for N in KID_SEQ:
        for arch in ARCHS:
            kid_data[(N, arch, d)] = run_attention_head(arch, N, d)

# ── Plot ──
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.6),
                                      gridspec_kw={'width_ratios': [1, 1, 1]})
fig.subplots_adjust(wspace=0.45, top=0.78, bottom=0.18)

COLOR_P = ARCH_COLORS["Proposed-1"]
COLOR_A = ARCH_COLORS["Azure-Lily"]

# Legend will be inside panel 1

TICK_X = [2048, 4096, 6144, 8192]

# ── Panel 1: DIMM latency fraction ──
ax1.plot(SEQ_LENS, dimm_frac_p, color=COLOR_P, linewidth=1.5, linestyle='-',
         marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.5,
         label='Proposed-1')
ax1.plot(SEQ_LENS, dimm_frac_a, color=COLOR_A, linewidth=1.5, linestyle='--',
         marker='D', markersize=4, markeredgecolor='white', markeredgewidth=0.5,
         label='Azure-Lily')

for frac, color, dy, va in [(dimm_frac_p, COLOR_P, -6, 'top'), (dimm_frac_a, COLOR_A, 6, 'bottom')]:
    ax1.annotate(f'{frac[-1]:.0f}%', xy=(SEQ_LENS[-1], frac[-1]),
                 xytext=(-6, dy), textcoords='offset points', ha='right', va=va,
                 fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT, color=color)

ax1.set_xticks(TICK_X)
ax1.set_xticklabels([str(n) for n in TICK_X])
ax1.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.1)
ax1.set_xlabel('')
ax1.set_ylabel('DIMM Latency Proportion (%)')
ax1.set_ylim(0, 100)
ax1.legend(fontsize=6, loc='center left', bbox_to_anchor=(0.02, 0.30),
           ncol=1, frameon=True, fancybox=True, framealpha=0.9, edgecolor='#CCC')
ax1.grid(True, alpha=0.1)

# ── Panel 2: Speedup breakdown ──
ax2.plot(SEQ_LENS, overall_speedup, color='#333', linewidth=2, linestyle='-',
         marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.8,
         label='Overall', zorder=4)
ax2.plot(SEQ_LENS, dimm_speedup, color='#2E86AB', linewidth=1.5, linestyle='--',
         marker='s', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
         label='DIMM only', zorder=3)
ax2.plot(SEQ_LENS, nondimm_speedup, color='#E05263', linewidth=1.5, linestyle=':',
         marker='^', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
         label='Non-DIMM', zorder=3)

# Asymptotic line for DIMM
ax2.axhline(y=14.0, color='#555', linewidth=1.0, linestyle='--', alpha=0.5)
mid_x = SEQ_LENS[len(SEQ_LENS) // 2]
ax2.text(SEQ_LENS[-2], 14.0 + 0.3, '14\u00d7', ha='center', va='bottom',
         fontsize=7, color='#555', fontweight='bold')
ax2.text(mid_x, nondimm_speedup[-1] + 0.3, f'{nondimm_speedup[-1]:.1f}\u00d7',
         ha='center', va='bottom', fontsize=7, color='#E05263', fontweight='bold')

ax2.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS, alpha=BASELINE_ALPHA)
ax2.set_xticks(TICK_X)
ax2.set_xticklabels([str(n) for n in TICK_X])
ax2.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.15)
ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('Speedup over Azure-Lily', labelpad=6)
ax2.set_ylim(0, max(dimm_speedup) * 1.15)
ax2.legend(fontsize=5.5, loc='center right')
ax2.grid(True, alpha=0.1)

# ── Panel 3: K-identity (from proto_kid_combined) ──
kid_lines = [
    ("proposed", 64,  "#B91C1C", "-",  "o", "Proposed-1 d=64 (K=2)"),
    ("proposed", 128, "#B91C1C", "--", "D", "Proposed-1 d=128 (K=1)"),
    ("al_like",  64,  "#2563EB", "-",  "s", "Proposed-2 d=64 (K=4)"),
    ("al_like",  128, "#2563EB", "--", "^", "Proposed-2 d=128 (K=2)"),
]

kid_cache = {}
for arch, d, color, ls, marker, label in kid_lines:
    ys = []
    for N in KID_SEQ:
        e_nl = kid_data[(N, arch, d)]
        e_al = kid_data[(N, "azurelily", d)]
        ys.append(e_al / e_nl)
    kid_cache[(arch, d)] = ys
    ax3.plot(KID_SEQ, ys, color=color, linewidth=1.8, linestyle=ls,
             marker=marker, markersize=4, markeredgecolor="white",
             markeredgewidth=0.8, label=label)

# Shaded K-identity gain
ax3.fill_between(KID_SEQ, kid_cache[("proposed", 128)], kid_cache[("proposed", 64)],
                 alpha=0.08, color="#B91C1C")
ax3.fill_between(KID_SEQ, kid_cache[("al_like", 128)], kid_cache[("al_like", 64)],
                 alpha=0.06, color="#2563EB")

# Asymptotic lines
ASYMP_COLOR = "#555555"
for val in [1.4, 1.7, 2.1]:
    ax3.axhline(y=val, color=ASYMP_COLOR, linewidth=0.7, linestyle="--", alpha=0.5)
    if val == 2.1:
        ax3.text(KID_SEQ[1], val - 0.03, f"{val:.1f}\u00d7", ha="center", va="top",
                 fontsize=7, color=ASYMP_COLOR, fontstyle="italic")
    else:
        ax3.text(KID_SEQ[-2], val + 0.02, f"{val:.1f}\u00d7", ha="center", va="bottom",
                 fontsize=7, color=ASYMP_COLOR, fontstyle="italic")

ax3.set_xlabel("")
ax3.set_ylabel("Normalized Inference/J", labelpad=6)
ax3.set_xticks(TICK_X)
ax3.set_xticklabels([str(n) for n in TICK_X])
ax3.set_xlim(KID_SEQ[0] * 0.85, KID_SEQ[-1] * 1.05)
ax3.set_ylim(bottom=1.2, top=2.2)
ax3.legend(fontsize=5.5, loc="center right", ncol=1, frameon=True,
           fancybox=True, framealpha=0.9, edgecolor="#CCC",
           bbox_to_anchor=(0.98, 0.72), columnspacing=0.8, handletextpad=0.4)
ax3.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
plt.close()
