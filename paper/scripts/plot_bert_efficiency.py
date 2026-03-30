#!/usr/bin/env python3
"""BERT-Tiny Efficiency: Area and Energy efficiency vs sequence length.

Auto-reads VTR results from bert_tiny_final_results.csv for Fmax and resources.
Runs IMC simulator for each (arch, seq_len) point.

Input: paper/data/bert_tiny_final_results.csv (Fmax + resources from VTR)
Output: paper/figures/benchmarks/bert_efficiency.pdf
"""
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR.parent / "data"
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
sys.path.insert(0, str(ROOT_DIR / "azurelily"))
sys.path.insert(0, str(ROOT_DIR / "azurelily" / "IMC"))

import matplotlib.pyplot as plt
from style_constants import (apply_style_sc, ARCH_COLORS, ARCH_MARKERS, ARCH_LINESTYLES,
                              BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE_SC, ANNOT_FONTWEIGHT_SC)
apply_style_sc()

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.bert_tiny import bert_tiny_model

OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "bert_efficiency.pdf"

CLB_TILE_UM2 = 2239  # includes routing

# ── Load VTR results ─────────────────────────────────────────────────────
VTR_CSV = DATA_DIR / "bert_tiny_final_results.csv"

# Config mapping: arch_name → (config_json, rows, cols, dpe_tile_w, dpe_tile_h)
ARCH_MAP = {
    "proposed":  ("nl_dpe.json",      1024, 128, 3, 7),
    "al_like":   ("nl_dpe.json",      1024, 256, 5, 8),
    "azurelily": ("azure_lily.json",   512, 128, 6, 5),
    "baseline":  ("baseline.json",       1,   1, 0, 0),
}

DISPLAY_NAMES = {
    "proposed": "Proposed",
    "al_like": "AL-like",
    "azurelily": "Azure-Lily",
    "baseline": "Baseline",
}


def _functional_dpes(rows, cols):
    """Non-DIMM DPEs: Q/K/V/O + FFN1 + FFN2, × 2 blocks."""
    import math as _m
    D_MODEL, D_FF = 128, 512
    per_block = (4 * _m.ceil(D_MODEL/rows) * _m.ceil(D_MODEL/cols)
                 + _m.ceil(D_MODEL/rows) * _m.ceil(D_FF/cols)
                 + _m.ceil(D_FF/rows) * _m.ceil(D_MODEL/cols))
    return per_block * 2


def load_vtr_results():
    """Load Fmax and resource counts from bert_tiny_final_results.csv."""
    results = {}
    with open(VTR_CSV) as f:
        for r in csv.DictReader(f):
            results[r["arch"]] = {
                "fmax": float(r["fmax_avg"]),
                "dpe_used": int(r["dpe_used"]),
                "dsp_used": int(r["dsp_used"]),
                "bram_used": int(r["bram_used"]),
                "clb_used": int(r["clb_used"]),
                "used_area_mm2": float(r["used_area_mm2"]),
            }
    return results


def run_bert(arch_name, vtr, N):
    """Run IMC simulator for one (arch, seq_len) point."""
    cfg_file, R, C, tw, th = ARCH_MAP[arch_name]
    v = vtr[arch_name]

    cfg = Config(str(ROOT_DIR / "azurelily" / "IMC" / "configs" / cfg_file))
    cfg.rows = R; cfg.cols = C; cfg.freq = v["fmax"]
    cfg.total_dsp = v["dsp_used"]
    cfg.total_clb = v["clb_used"]
    cfg.total_mem = v["bram_used"]
    cfg.total_dimm_dpes = max(0, v["dpe_used"] - _functional_dpes(R, C))

    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)

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


def compute_used_area(arch_name, vtr):
    """Compute area from USED resources."""
    v = vtr[arch_name]
    _, _, _, tw, th = ARCH_MAP[arch_name]
    dpe_cells = v["dpe_used"] * tw * th
    dsp_cells = v["dsp_used"] * 1 * 4
    bram_cells = v["bram_used"] * 1 * 2
    clb_cells = v["clb_used"] * 1 * 1
    return (dpe_cells + dsp_cells + bram_cells + clb_cells) * CLB_TILE_UM2 / 1e6


# ── Main ──────────────────────────────────────────────────────────────────
vtr = load_vtr_results()

# Print VTR summary
print("VTR Results (from bert_tiny_final_results.csv):")
print(f"{'Arch':>12} {'Fmax':>8} {'DPEs':>6} {'DSPs':>6} {'BRAMs':>6} {'CLBs':>7} {'Area(mm²)':>10}")
print("-" * 60)
for arch in ["baseline", "proposed", "al_like", "azurelily"]:
    if arch not in vtr:
        continue
    v = vtr[arch]
    area = compute_used_area(arch, vtr)
    print(f"{DISPLAY_NAMES[arch]:>12} {v['fmax']:>7.1f} {v['dpe_used']:>6} {v['dsp_used']:>6} "
          f"{v['bram_used']:>6} {v['clb_used']:>7} {area:>9.2f}")
print()

# Collect data for all seq_lens
SEQ_LENS = [1024, 2048, 4096, 6144, 8192]
PLOT_ARCHS = ["proposed", "al_like", "azurelily"]

data = {}
for N in SEQ_LENS:
    for arch in PLOT_ARCHS:
        if arch not in vtr:
            continue
        e, l = run_bert(arch, vtr, N)
        data[(N, arch)] = {"energy": e, "latency": l}

# Print results table
print(f"{'Arch':>12} {'N':>5} {'Energy(MpJ)':>12} {'Lat(µs)':>10} {'Tput/mm²':>10} {'inf/s/J':>12}")
print("-" * 70)
for N in SEQ_LENS:
    for arch in PLOT_ARCHS:
        if (N, arch) not in data:
            continue
        e = data[(N, arch)]["energy"]
        l = data[(N, arch)]["latency"]
        area = compute_used_area(arch, vtr)
        tput = 1e9 / l
        tput_mm2 = tput / area if area > 0 else 0
        tput_s_j = tput / (e / 1e12) if e > 0 else 0  # inf/s per joule
        print(f"{DISPLAY_NAMES[arch]:>12} {N:>5} {e/1e6:>11.3f} {l/1e3:>9.1f} {tput_mm2:>9.0f} {tput_s_j:>11.0f}")
    print()

# Normalized table (vs Azure-Lily)
print("Normalized (Azure-Lily = 1.0×):")
print(f"{'Arch':>12} {'N':>5} {'Tput/mm² (norm)':>16} {'inf/s/J (norm)':>15}")
print("-" * 55)
for N in SEQ_LENS:
    if (N, "azurelily") not in data:
        continue
    al_e = data[(N, "azurelily")]["energy"]
    al_l = data[(N, "azurelily")]["latency"]
    al_area = compute_used_area("azurelily", vtr)
    al_tput_mm2 = (1e9 / al_l) / al_area if al_area > 0 else 1
    al_tput_s_j = (1e9 / al_l) / (al_e / 1e12) if al_e > 0 else 1
    for arch in PLOT_ARCHS:
        if (N, arch) not in data:
            continue
        e = data[(N, arch)]["energy"]
        l = data[(N, arch)]["latency"]
        area = compute_used_area(arch, vtr)
        tput_mm2 = (1e9 / l) / area if area > 0 else 0
        tput_s_j = (1e9 / l) / (e / 1e12) if e > 0 else 0
        print(f"{DISPLAY_NAMES[arch]:>12} {N:>5} {tput_mm2/al_tput_mm2:>15.2f}× {tput_s_j/al_tput_s_j:>14.2f}×")
    print()

# ── Plot ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
fig.subplots_adjust(wspace=0.28)

for ax, metric_fn, ylabel in [
    (ax1, lambda N, a: (1e9 / data[(N, a)]["latency"]) / compute_used_area(a, vtr),
     "Normalized Throughput/mm\u00b2"),
    (ax2, lambda N, a: (1e9 / data[(N, a)]["latency"]) / (data[(N, a)]["energy"] / 1e12),
     "Normalized Inference/s/J"),
]:
    for arch in PLOT_ARCHS:
        cname = DISPLAY_NAMES[arch]
        ys = []
        for N in SEQ_LENS:
            val = metric_fn(N, arch)
            bl = metric_fn(N, "azurelily")
            ys.append(val / bl)

        ax.plot(SEQ_LENS, ys, color=ARCH_COLORS[cname], linewidth=1.5,
                linestyle=ARCH_LINESTYLES[cname],
                marker=ARCH_MARKERS[cname], markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=cname)

        # Store final values for asymptotic lines
        if arch == "proposed":
            _proposed_ys = ys
        elif arch == "al_like":
            _al_like_ys = ys

    # Asymptotic lines at convergence values
    mid_x = SEQ_LENS[len(SEQ_LENS) // 2]
    if ax == ax1:
        # Left panel: 2 asymptotic lines (Proposed + AL-like)
        for arch, ys_ref in [("proposed", _proposed_ys), ("al_like", _al_like_ys)]:
            cname = DISPLAY_NAMES[arch]
            asymp = ys_ref[-1]
            ax.axhline(y=asymp, color="black", linewidth=0.8,
                       linestyle=":", alpha=0.4)
            dy = 0.08 if arch == "proposed" else -0.08
            ax.text(mid_x, asymp + dy, f"{asymp:.2f}\u00d7",
                    ha="center", va="bottom" if arch == "proposed" else "top",
                    fontsize=10, color="black", fontstyle="italic")
    else:
        # Right panel: 1 asymptotic line (Proposed only)
        asymp = _proposed_ys[-1]
        ax.axhline(y=asymp, color="black", linewidth=0.8,
                   linestyle=":", alpha=0.4)
        ax.text(mid_x, asymp - 0.08, f"{asymp:.2f}\u00d7",
                ha="center", va="top",
                fontsize=10, color="black", fontstyle="italic")

    ax.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
               alpha=BASELINE_ALPHA)
    ax.set_xlabel("Sequence Length (N)")
    ax.set_ylabel(ylabel)
    TICK_LENS = [512, 2048, 4096, 6144, 8192]
    ax.set_xticks(TICK_LENS)
    ax.set_xticklabels([str(n) for n in TICK_LENS], fontsize=9)
    all_vals = []
    for arch in PLOT_ARCHS:
        for N in SEQ_LENS:
            v = metric_fn(N, arch) / metric_fn(N, "azurelily")
            all_vals.append(v)
    ax.set_ylim(bottom=min(all_vals) * 0.85, top=max(all_vals) * 1.15)
    ax.set_xlim(SEQ_LENS[0] * 0.85, SEQ_LENS[-1] * 1.1)
    ax.grid(True, alpha=0.1)

# Shared legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=ARCH_COLORS[DISPLAY_NAMES[a]], label=DISPLAY_NAMES[a])
           for a in PLOT_ARCHS]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, 1.01), frameon=False)

fig.savefig(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
