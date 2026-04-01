#!/usr/bin/env python3
"""PROTOTYPE: Combined K-identity plot — single panel, inf/J, shaded K-id gap."""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
sys.path.insert(0, str(ROOT_DIR / "azurelily"))
sys.path.insert(0, str(ROOT_DIR / "azurelily" / "IMC"))

from style_constants import apply_style_sc, BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA, FIG_SINGLE
apply_style_sc()

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.attention import attention_model
from scheduler_stats.common import configure_logging
configure_logging(False)

OUT_PATH = SCRIPT_DIR / "proto_kid_combined.pdf"

ARCHS = {
    "proposed":  ("nl_dpe.json",     1024, 128, 133.4, 274, 4,   453, 172),
    "al_like":   ("nl_dpe.json",     1024, 256, 139.7, 78,  4,   441, 172),
    "azurelily": ("azure_lily.json", 512,  128, 127.9, 18,  326, 188, 16),
}

D_HEADS = [64, 128]
SEQ_LENS = [1024, 2048, 4096, 6144, 8192]


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


# ── Collect data ────────────────────────────────────────────────────
data = {}
for d in D_HEADS:
    for N in SEQ_LENS:
        for arch in ARCHS:
            data[(N, arch, d)] = run_attention_head(arch, N, d)

# ── Plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SINGLE)

# Line configs: (arch, d_head, color, linestyle, marker, label)
lines = [
    ("proposed", 64,  "#B91C1C", "-",  "o", "Proposed-1 d=64 (K=2)"),
    ("proposed", 128, "#B91C1C", "--", "D", "Proposed-1 d=128 (K=1)"),
    ("al_like",  64,  "#2563EB", "-",  "s", "Proposed-2 d=64 (K=4)"),
    ("al_like",  128, "#2563EB", "--", "^", "Proposed-2 d=128 (K=2)"),
]

ys_cache = {}
for arch, d, color, ls, marker, label in lines:
    ys = []
    for N in SEQ_LENS:
        e_nl = data[(N, arch, d)]
        e_al = data[(N, "azurelily", d)]
        ys.append(e_al / e_nl)
    ys_cache[(arch, d)] = ys

    ax.plot(SEQ_LENS, ys, color=color, linewidth=1.8, linestyle=ls,
            marker=marker, markersize=5, markeredgecolor="white",
            markeredgewidth=0.8, label=label)

# Shaded K-identity gain regions
ys_k2 = ys_cache[("proposed", 64)]
ys_k1 = ys_cache[("proposed", 128)]
ax.fill_between(SEQ_LENS, ys_k1, ys_k2, alpha=0.08, color="#B91C1C")

ys_k4 = ys_cache[("al_like", 64)]
ys_k2al = ys_cache[("al_like", 128)]
ax.fill_between(SEQ_LENS, ys_k2al, ys_k4, alpha=0.06, color="#2563EB")

# 3 asymptotic reference lines (dark gray, dashed)
mid_x = SEQ_LENS[len(SEQ_LENS) // 2]
ASYMP_COLOR = "#555555"
right_x = SEQ_LENS[-2]  # second to last point
for val in [1.4, 1.7, 2.1]:
    ax.axhline(y=val, color=ASYMP_COLOR, linewidth=0.7, linestyle="--", alpha=0.5)
    if val == 2.1:
        ax.text(SEQ_LENS[0] * 1.2, val - 0.01, f"{val:.1f}\u00d7", ha="center", va="top",
                fontsize=7, color=ASYMP_COLOR, fontstyle="italic")
    else:
        ax.text(right_x, val + 0.02, f"{val:.1f}\u00d7", ha="center", va="bottom",
                fontsize=7, color=ASYMP_COLOR, fontstyle="italic")

ax.set_xlabel("Sequence Length (N)")
ax.set_ylabel("Normalized Inference/J")
ax.set_xticks(SEQ_LENS)
ax.set_xticklabels([str(n) for n in SEQ_LENS])
ax.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.05)
ax.set_ylim(bottom=1.2, top=2.2)
ax.legend(fontsize=5.5, loc="center right", ncol=2, frameon=True,
          fancybox=True, framealpha=0.9, edgecolor="#CCC",
          bbox_to_anchor=(0.98, 0.72), columnspacing=0.8, handletextpad=0.4)
ax.grid(True, alpha=0.1)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
plt.close()
