#!/usr/bin/env python3
"""BERT-Tiny End-to-End Energy Breakdown vs Sequence Length.

(a) Stacked bars: DIMM vs Proj+FFN vs Other, Proposed vs Azure-Lily across N.
(b) DIMM-internal split: DPE-crossbar vs FPGA-fabric within DIMM operations.

Output: paper/figures/benchmarks/bert_energy_breakdown.pdf
"""
import sys
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
sys.path.insert(0, str(ROOT_DIR / "azurelily"))
sys.path.insert(0, str(ROOT_DIR / "azurelily" / "IMC"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from style_constants import (apply_style, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT,
                              BREAKDOWN_COLORS, ARCH_COLORS)
apply_style()

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.bert_tiny import bert_tiny_model

OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "bert_energy_breakdown.pdf"

# VTR ground truth resources
VTR_AVAILABLE = {
    "proposed":  {"DSPs": 222, "CLBs": 13806, "BRAMs": 518},
    "azurelily": {"DSPs": 333, "CLBs": 11262, "BRAMs": 740},
}

CONFIGS = [
    ("Proposed",   "nl_dpe.json",     1024, 128, 135.7, "proposed"),
    ("Azure-Lily", "azure_lily.json",  512, 128,  45.3, "azurelily"),
]

SEQ_LENS = [256, 512, 1024, 2048]

# Keys that represent DPE/IMC energy (crossbar, ADC, ACAM, DIMM nonlinear)
DPE_KEYS = {"imc_vmm", "imc_conversion", "imc_digital_post",
            "imc_dimm_exp", "imc_dimm_log"}

# 3 high-level categories
CAT_KEYS = {
    "DIMM\n(QK\u1d40+S\u00d7V)": ["DIMM"],
    "Proj+FFN":                    ["Proj", "FFN"],
    "Other":                       ["Other"],
}
CAT_NAMES = list(CAT_KEYS.keys())
CAT_COLORS = [BREAKDOWN_COLORS["DIMM"], BREAKDOWN_COLORS["Proj_FFN"],
              BREAKDOWN_COLORS["Other"]]


def _snapshot_breakdown(stats):
    """Return a copy of the current energy_breakdown dict."""
    return dict(stats.energy_breakdown)


def _delta_breakdown(before, after):
    """Compute per-key energy delta between two breakdown snapshots."""
    all_keys = set(before) | set(after)
    return {k: after.get(k, 0) - before.get(k, 0) for k in all_keys}


def run_breakdown(cfg_file, R, C, fmax, N, avail_key):
    """Run BERT-Tiny, return:
      - groups: {Proj, FFN, DIMM, Other} -> total energy
      - dimm_dpe: total DPE energy within DIMM operations
      - dimm_fabric: total FPGA-fabric energy within DIMM operations
    """
    cfg = Config(str(ROOT_DIR / "azurelily" / "IMC" / "configs" / cfg_file))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    avail = VTR_AVAILABLE.get(avail_key, {})
    if avail.get("DSPs") is not None: cfg.total_dsp = avail["DSPs"]
    if avail.get("CLBs") is not None: cfg.total_clb = avail["CLBs"]
    if avail.get("BRAMs") is not None: cfg.total_mem = avail["BRAMs"]

    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)

    md, _ = bert_tiny_model(1, 1, N, 128, False, False)
    groups = defaultdict(float)
    dimm_dpe = 0.0
    dimm_fabric = 0.0

    def run_group(layers, group_name):
        nonlocal dimm_dpe, dimm_fabric
        snap_before = _snapshot_breakdown(stats)
        for layer in layers:
            sched.run_layer(layer)
        snap_after = _snapshot_breakdown(stats)
        delta = _delta_breakdown(snap_before, snap_after)
        group_total = sum(delta.values())
        groups[group_name] += group_total

        # Track DPE vs fabric split for DIMM operations
        if group_name == "DIMM":
            dpe_e = sum(delta.get(k, 0) for k in DPE_KEYS)
            dimm_dpe += dpe_e
            dimm_fabric += (group_total - dpe_e)

    run_group(md["embedding"], "Other")
    for block in md["blocks"]:
        run_group(block["qkv_proj"], "Proj")
        for hi in range(block["num_heads"]):
            run_group(block["head_attention"], "DIMM")
        run_group([block["post_attn"][0]], "Proj")
        run_group(block["post_attn"][1:], "Other")
        run_group(block["ffn"][:2], "FFN")
        run_group(block["ffn"][2:], "Other")

    return dict(groups), dimm_dpe, dimm_fabric


# ── Collect data ──────────────────────────────────────────────────────────
data = {}       # (N, label) -> {cat: energy}
dimm_split = {} # (N, label) -> {"DPE": e, "Fabric": e}

for label, cfg_file, R, C, fmax, avail_key in CONFIGS:
    for N in SEQ_LENS:
        groups, dpe_e, fab_e = run_breakdown(cfg_file, R, C, fmax, N, avail_key)
        data[(N, label)] = groups
        dimm_split[(N, label)] = {"DPE": dpe_e, "Fabric": fab_e}

# Merge into 3 high-level categories
merged = {}
for key, raw in data.items():
    m = {}
    for cat, detail_keys in CAT_KEYS.items():
        m[cat] = sum(raw.get(dk, 0) for dk in detail_keys)
    merged[key] = m

# Normalization reference: Proposed at N=256
ref_total = sum(merged[(256, "Proposed")].values())

# ── Plot: (a) energy breakdown, (b) DIMM internal split ─────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8),
                                gridspec_kw={'width_ratios': [1.2, 1]})
fig.subplots_adjust(wspace=0.35, top=0.82)

# Panel labels
ax1.text(-0.12, 1.10, '(a)', transform=ax1.transAxes, fontsize=10,
         fontweight='bold', va='top')
ax2.text(-0.15, 1.10, '(b)', transform=ax2.transAxes, fontsize=10,
         fontweight='bold', va='top')

# ── Panel (a): stacked bars, DIMM vs Proj+FFN vs Other ───────────────────
bar_w = 0.35
arch_labels = ["Proposed", "Azure-Lily"]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

arch_hatch = {"Proposed": None, "Azure-Lily": "///"}

for ai, arch in enumerate(arch_labels):
    offset = (ai - 0.5) * bar_w
    hatch = arch_hatch[arch]
    for ni, N in enumerate(SEQ_LENS):
        bottom = 0
        m = merged[(N, arch)]
        own_total = sum(m.values())
        for ci, cat in enumerate(CAT_NAMES):
            val = m[cat] / ref_total * 100
            ax1.bar(x[ni] + offset, val, bar_w, bottom=bottom,
                    color=CAT_COLORS[ci], edgecolor='white', linewidth=0.5,
                    hatch=hatch, zorder=3,
                    label=cat if (ni == 0 and ai == 0) else None)
            own_pct = m[cat] / own_total * 100
            if own_pct > 8:
                ax1.text(x[ni] + offset, bottom + val / 2, f'{own_pct:.0f}%',
                         ha='center', va='center', fontsize=5.5,
                         fontweight='bold', color='white', zorder=5)
            bottom += val

        # Total label on top
        total_norm = sum(m.values()) / ref_total * 100
        ax1.text(x[ni] + offset, total_norm * 1.12, f'{total_norm:.0f}',
                 ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                 color='black', zorder=5)

arch_handles = [
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   label='K=2 (Proposed)'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5,
                   hatch='///', label='Azure-Lily'),
]
cat_handles = [mpatches.Patch(facecolor=CAT_COLORS[ci], edgecolor='white',
               label=CAT_NAMES[ci].replace('\n', ' '))
               for ci in range(len(CAT_NAMES))]
ax1.legend(handles=arch_handles + cat_handles, fontsize=6, loc='upper left',
           ncol=1, frameon=True)

ax1.set_xticks(x)
ax1.set_xticklabels([f'N={N}' for N in SEQ_LENS])
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Energy (norm. to Proposed @ N=256)')
ax1.set_yscale('log')
ax1.set_ylim(50, 15000)
ax1.grid(True, alpha=0.08, axis='y', zorder=0)

# ── Panel (b): DIMM-internal: DPE vs Fabric, % stacked bars ─────────────
DPE_COLOR = "#8B5CF6"      # violet for DPE/crossbar
FABRIC_COLOR = "#F59E0B"   # amber for FPGA fabric (CLB/DSP)

bar_w2 = 0.35
for ai, arch in enumerate(arch_labels):
    offset = (ai - 0.5) * bar_w2
    hatch = arch_hatch[arch]
    dpe_pcts = []
    fab_pcts = []
    for N in SEQ_LENS:
        ds = dimm_split[(N, arch)]
        dimm_total = ds["DPE"] + ds["Fabric"]
        dpe_pct = ds["DPE"] / dimm_total * 100 if dimm_total > 0 else 0
        fab_pct = ds["Fabric"] / dimm_total * 100 if dimm_total > 0 else 0
        dpe_pcts.append(dpe_pct)
        fab_pcts.append(fab_pct)

    for ni in range(n_groups):
        ax2.bar(x[ni] + offset, dpe_pcts[ni], bar_w2,
                color=DPE_COLOR, edgecolor='white', linewidth=0.5,
                hatch=hatch, zorder=3,
                label='DPE (crossbar)' if (ni == 0 and ai == 0) else None)
        ax2.bar(x[ni] + offset, fab_pcts[ni], bar_w2, bottom=dpe_pcts[ni],
                color=FABRIC_COLOR, edgecolor='white', linewidth=0.5,
                hatch=hatch, zorder=3,
                label='FPGA fabric\n(CLB/DSP)' if (ni == 0 and ai == 0) else None)

        # Annotate DPE %
        if dpe_pcts[ni] > 8:
            ax2.text(x[ni] + offset, dpe_pcts[ni] / 2, f'{dpe_pcts[ni]:.0f}%',
                     ha='center', va='center', fontsize=5.5, fontweight='bold',
                     color='white', zorder=5)
        if fab_pcts[ni] > 8:
            ax2.text(x[ni] + offset, dpe_pcts[ni] + fab_pcts[ni] / 2,
                     f'{fab_pcts[ni]:.0f}%',
                     ha='center', va='center', fontsize=5.5, fontweight='bold',
                     color='white', zorder=5)

# Legend for panel (b)
b_handles = [
    mpatches.Patch(facecolor=DPE_COLOR, edgecolor='white', label='DPE (crossbar)'),
    mpatches.Patch(facecolor=FABRIC_COLOR, edgecolor='white', label='FPGA fabric (CLB/DSP)'),
]
ax2.legend(handles=arch_handles + b_handles, fontsize=6, loc='lower right',
           ncol=1, frameon=True)

ax2.set_xticks(x)
ax2.set_xticklabels([f'N={N}' for N in SEQ_LENS])
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('DIMM Energy Composition (%)')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.08, axis='y', zorder=0)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# ── Print tables ─────────────────────────────────────────────────────────
print(f"\n{'N':>5} {'Config':>12} {'DIMM%':>7} {'Proj+FFN%':>10} {'Other%':>7} {'Norm Total':>11}")
print("-" * 60)
for N in SEQ_LENS:
    for arch in arch_labels:
        m = merged[(N, arch)]
        total = sum(m.values())
        norm = total / ref_total * 100
        print(f"{N:>5} {arch:>12} "
              f"{m[CAT_NAMES[0]]/total*100:>6.1f}% "
              f"{m[CAT_NAMES[1]]/total*100:>9.1f}% "
              f"{m[CAT_NAMES[2]]/total*100:>6.1f}% "
              f"{norm:>10.1f}")
    print()

print(f"\nDIMM Internal Split (DPE vs Fabric):")
print(f"{'N':>5} {'Config':>12} {'DPE%':>7} {'Fabric%':>9} {'DIMM total (pJ)':>16}")
print("-" * 55)
for N in SEQ_LENS:
    for arch in arch_labels:
        ds = dimm_split[(N, arch)]
        dtotal = ds["DPE"] + ds["Fabric"]
        print(f"{N:>5} {arch:>12} "
              f"{ds['DPE']/dtotal*100:>6.1f}% "
              f"{ds['Fabric']/dtotal*100:>8.1f}% "
              f"{dtotal:>15.1f}")
    print()
