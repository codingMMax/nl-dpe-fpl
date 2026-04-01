#!/usr/bin/env python3
"""Attention Head K-identity Packing: Energy and Throughput vs Sequence Length.

Runs ONLY the attention DIMM (QK^T + softmax + Score×V) for multiple
d_head values to show how K_id packing scales with head dimension.

Normalized to Azure-Lily = 1.0× at each (N, d_head).

Output: paper/figures/benchmarks/attention_kid_efficiency.pdf
"""
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

from style_constants import (apply_style, ARCH_COLORS, BASELINE_COLOR,
                              BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE, ANNOT_FONTWEIGHT, FIG_DUAL)
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

OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "attention_kid_efficiency.pdf"

# ── Architecture configs (from VTR used resources) ───────────────────────
ARCHS = {
    "proposed":  ("nl_dpe.json",     1024, 128, 133.4, 274, 4,   453, 172),
    "al_like":   ("nl_dpe.json",     1024, 256, 139.7, 78,  4,   441, 172),
    "azurelily": ("azure_lily.json", 512,  128, 127.9, 18,  326, 188, 16),
}

DISPLAY = {"proposed": "Proposed-1", "al_like": "Proposed-2"}

D_HEADS = [64, 128]
LINE_STYLES = {64: "-", 128: "--"}
MARKERS = {64: "o", 128: "D"}
MARKER_SIZES = {64: 4, 128: 5}
LINE_WIDTHS = {64: 2, 128: 1.5}

# Colors: violet (Proposed), pink (AL-like) — matches dual_identity style
PLOT_COLORS = {
    "proposed": "#8B5CF6",   # vivid violet
    "al_like":  "#EC4899",   # hot pink
}


def _functional_dpes(rows, cols):
    D_MODEL, D_FF = 128, 512
    per_block = (4 * math.ceil(D_MODEL / rows) * math.ceil(D_MODEL / cols)
                 + math.ceil(D_MODEL / rows) * math.ceil(D_FF / cols)
                 + math.ceil(D_FF / rows) * math.ceil(D_MODEL / cols))
    return per_block * 2


def run_attention_head(arch, N, d_head):
    """Run one attention head DIMM only, return (energy, latency)."""
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

    return sum(stats.energy_breakdown.values()), sum(stats.latency_raw.values())


# ── Collect data ─────────────────────────────────────────────────────────
SEQ_LENS = [1024, 2048, 4096, 6144, 8192]

data = {}
for d in D_HEADS:
    for N in SEQ_LENS:
        for arch in list(ARCHS.keys()):
            e, l = run_attention_head(arch, N, d)
            data[(N, arch, d)] = {"energy": e, "latency": l}

# Print K_id table
print("K-identity packing factors:")
print(f"{'d_head':>6}  {'Proposed (C=128)':>18}  {'AL-like (C=256)':>18}")
for d in D_HEADS:
    k_p = 128 // d
    k_a = 256 // d
    print(f"{d:>6}  {'K_id=' + str(k_p):>18}  {'K_id=' + str(k_a):>18}")
print()

# Print energy ratios
print(f"{'d_head':>6} {'N':>5}  {'E ratio P':>10} {'E ratio AL':>10}  {'T ratio P':>10} {'T ratio AL':>10}")
print("-" * 60)
for d in D_HEADS:
    for N in [1024, 2048, 8192]:
        ez = data[(N, "azurelily", d)]["energy"]
        lz = data[(N, "azurelily", d)]["latency"]
        ep = data[(N, "proposed", d)]["energy"]
        lp = data[(N, "proposed", d)]["latency"]
        ea = data[(N, "al_like", d)]["energy"]
        la = data[(N, "al_like", d)]["latency"]
        print(f"{d:>6} {N:>5}  {ez/ep:>9.2f}x {ez/ea:>9.2f}x  {lz/lp:>9.2f}x {lz/la:>9.2f}x")
    print()

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DUAL)
fig.subplots_adjust(wspace=0.32, top=0.85, bottom=0.18)

TICK_LENS = [1024, 2048, 4096, 6144, 8192]

for ax, get_ratio, ylabel, panel_label in [
    (ax1,
     lambda N, a, d: data[(N, "azurelily", d)]["energy"] / data[(N, a, d)]["energy"],
     "Normalized Energy Efficiency\n(vs Azure-Lily = 1.0\u00d7)",
     "(a)"),
    (ax2,
     lambda N, a, d: (1.0 / data[(N, a, d)]["latency"]) / (1.0 / data[(N, "azurelily", d)]["latency"]),
     "Normalized Throughput\n(vs Azure-Lily = 1.0\u00d7)",
     "(b)"),
]:
    for arch in ["proposed", "al_like"]:
        cname = DISPLAY[arch]
        C = ARCHS[arch][2]

        for d in D_HEADS:
            K_id = C // d
            ys = [get_ratio(N, arch, d) for N in SEQ_LENS]
            color = PLOT_COLORS[arch]

            ax.plot(SEQ_LENS, ys, color=color, linewidth=LINE_WIDTHS[d],
                    linestyle=LINE_STYLES[d],
                    marker=MARKERS[d], markersize=MARKER_SIZES[d],
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=f"{cname} d={d} (K$_{{id}}$={K_id})")

    ax.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
               alpha=BASELINE_ALPHA)

    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(TICK_LENS)
    ax.set_xticklabels([str(n) for n in TICK_LENS], fontsize=7)
    ax.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.1)
    ax.legend(fontsize=5.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.1)

    ax.text(-0.12, 1.08, panel_label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')

fig.savefig(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
plt.close()
