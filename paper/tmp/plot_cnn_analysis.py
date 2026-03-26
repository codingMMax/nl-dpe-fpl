#!/usr/bin/env python3
"""CNN Analysis: (a) End-to-end efficiency bars, (b) VGG-11 energy breakdown.

Input: benchmarks/results/imc_benchmark_results.csv + IMC simulator
Output: paper/figures/benchmarks/cnn_analysis.pdf
"""
import sys, csv, math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
sys.path.insert(0, str(ROOT_DIR / "azurelily"))
sys.path.insert(0, str(ROOT_DIR / "azurelily" / "IMC"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.05})

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler

CSV_PATH = SCRIPT_DIR / "imc_benchmark_results.csv"
OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "cnn_analysis.pdf"

def run_cnn(model_name, cfile, R, C, fmax):
    cfg = Config(str(ROOT_DIR / "azurelily" / "IMC" / "configs" / f"{cfile}.json"))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats); fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    if model_name == "resnet":
        from models.resnet import resnet_model
        layers, _ = resnet_model(1, 1, 128, 128, False, False)
    else:
        from models.vggnet import vgg_model
        layers, _ = vgg_model(1, 1, 128, 128, False, False)
    for layer in layers:
        sched.run_layer(layer)
    bd = dict(stats.energy_breakdown)
    total = sum(bd.values())
    groups = {
        "DPE Projections": sum(v for k, v in bd.items() if k in ["imc_vmm", "imc_digital_post", "imc_conversion"]),
        "CLB (activation)": sum(v for k, v in bd.items() if k in ["clb_compare", "clb_activation"]),
        "CLB (reduction)": sum(v for k, v in bd.items() if k in ["clb_reduction", "clb_add"]),
        "Memory": sum(v for k, v in bd.items() if "sram" in k or "bram" in k),
    }
    accounted = sum(groups.values())
    if abs(total - accounted) > 1:
        groups["Other"] = total - accounted
    return total, groups

# ── Left: Normalized efficiency bars ──
imc_csv = {r['label']: r for r in csv.DictReader(open(CSV_PATH))}
cnn_models = ["ResNet-9", "VGG-11"]
archs = ["NL-DPE Proposed", "NL-DPE AL-Matched", "Azure-Lily"]
colors_bar = {"NL-DPE Proposed":"#059669","NL-DPE AL-Matched":"#2563EB","Azure-Lily":"#D1D5DB"}

norm = {}
for m in cnn_models:
    bl = float(imc_csv[f"{m} Azure-Lily"]['total_energy_pj'])
    for a in archs:
        norm[(m,a)] = bl / float(imc_csv[f"{m} {a}"]['total_energy_pj'])
for a in archs:
    vals = [norm[(m,a)] for m in cnn_models]
    norm[("Geomean",a)] = math.exp(sum(math.log(v) for v in vals)/len(vals))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1.3]})
fig.subplots_adjust(wspace=0.35)

cats = cnn_models + ["Geomean"]
x = np.arange(len(cats)); w = 0.25
for i, a in enumerate(archs):
    vals = [norm[(m,a)] for m in cats]
    ax1.bar(x + (i-1)*w, vals, w, color=colors_bar[a], edgecolor="white", linewidth=0.5, zorder=3)
    if a != "Azure-Lily":
        for j, v in enumerate(vals):
            ax1.text(x[j]+(i-1)*w, v+0.5, f"{v:.0f}\u00d7", ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=colors_bar[a], rotation=30)

ax1.axhline(y=1.0, color='#EF4444', linewidth=1, linestyle='--', alpha=0.5, zorder=2)
ax1.set_ylabel("Energy Efficiency\n(normalized, AL=1.0\u00d7)", fontsize=9)
ax1.set_title("(a) End-to-End", fontsize=10, fontweight="bold")
ax1.set_xticks(x); lbls=ax1.set_xticklabels(cats, fontsize=9); lbls[-1].set_fontweight('bold')
ax1.set_ylim(bottom=0); ax1.grid(True, alpha=0.08, axis="y", zorder=0)

# ── Right: VGG-11 component breakdown ──
_, nl_groups = run_cnn("vgg", "nl_dpe", 1024, 128, 111.5)
_, al_groups = run_cnn("vgg", "azure_lily", 512, 128, 146.3)

all_keys = ["DPE Projections", "CLB (activation)", "CLB (reduction)", "Memory", "Other"]
nl_vals = [nl_groups.get(k, 0) for k in all_keys]
al_vals = [al_groups.get(k, 0) for k in all_keys]
nl_pcts = [v / sum(nl_vals) * 100 for v in nl_vals]
al_pcts = [v / sum(al_vals) * 100 for v in al_vals]

comp_colors = ["#059669", "#3B82F6", "#8B5CF6", "#F59E0B", "#94A3B8"]

for idx, (vals, label) in enumerate([(al_pcts, "Azure-Lily"), (nl_pcts, "NL-DPE Proposed")]):
    left = 0
    for j, (v, k) in enumerate(zip(vals, all_keys)):
        if v < 0.3: continue
        ax2.barh(idx, v, 0.6, left=left, color=comp_colors[j],
                edgecolor="white", linewidth=0.5, label=k if idx==0 else None)
        if v > 5:
            ax2.text(left + v/2, idx, f"{v:.0f}%", ha='center', va='center', fontsize=7, fontweight='bold')
        left += v

ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Azure-Lily", "NL-DPE"], fontsize=9)
ax2.set_xlabel("Energy Breakdown (%)", fontsize=9)
ax2.set_title("(b) VGG-11 Energy Breakdown", fontsize=10, fontweight="bold")
ax2.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, -0.15), ncol=3)
ax2.set_xlim(0, 105)

handles = [Patch(facecolor=colors_bar[a], label=a) for a in archs]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.35, 1.02), frameon=False)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
