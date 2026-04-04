#!/usr/bin/env python3
"""Block-level DPE comparison: ACAM vs ADC energy advantage.

Figure 1: FC + Activation per-token energy (bar chart)
  - Proposed-1, Proposed-2, Azure-Lily side by side
  - Stacked: DPE block | FPGA activation | Memory
  - Annotate: 44× cheaper, free activation

Figure 2: ACAM vs ADC per-pass energy breakdown (bar chart)
  - Crossbar+DAC vs ADC/ACAM energy per VMM pass
  - Shows ADC is the bottleneck

Output: paper/figures/benchmarks/bert_block_comparison.pdf
"""
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT / "nl_dpe"))
sys.path.insert(0, str(ROOT / "azurelily"))
sys.path.insert(0, str(ROOT / "azurelily" / "IMC"))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from style_constants import apply_style, ARCH_COLORS, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT
from area_power import dpe_specs
apply_style()

OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "block_comparison.pdf"

# ── Run FC(128,128) + activation for each architecture ───────────────────
from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
import nn

ARCHS = [
    ("Proposed-1", "nl_dpe.json", 1024, 128),
    ("Proposed-2", "nl_dpe.json", 1024, 256),
    ("Azure-Lily", "azure_lily.json", 512, 128),
]

fc_results = {}
for name, cfg_file, R, C in ARCHS:
    cfg = Config(str(ROOT / "azurelily/IMC/configs" / cfg_file))
    cfg.rows = R; cfg.cols = C; cfg.freq = 135
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    layer = nn.Layer(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0,
                     name="fc_test", type="linear", has_act=True,
                     num_computes=1, num_inputs=1, debug=False, energy_stats=False)
    layer.set_input(1, 1, 128)
    sched.run_layer(layer)
    bd = dict(stats.energy_breakdown)
    dpe_e = sum(v for k, v in bd.items() if "imc" in k)
    act_e = bd.get("fpga_activation", 0)
    mem_e = bd.get("sram_read", 0) + bd.get("sram_write", 0)
    total_e = sum(bd.values())
    fc_results[name] = {"dpe": dpe_e, "act": act_e, "mem": mem_e, "total": total_e}

# ── Per-pass energy breakdown ────────────────────────────────────────────
pass_results = {}
for name, cfg_file, R, C in ARCHS:
    s = dpe_specs(R, C)
    # ACAM/ADC energy — crossbar computed per-arch
    if "nl_dpe" in cfg_file:
        # ACAM fires once: e_digital × C cols × 1
        acam_adc_e = s["e_digital_pj"] * C * 1
        label = "ACAM"
        crossbar_e = s["e_analogue_pj"] * 8  # from area_power model
    else:
        # Azure-Lily paper: total 20 mW, ADC 14.4 mW, other 5.6 mW @ 1.2 GHz
        crossbar_e = (5.6 / 1.2) * 8   # 37.33 pJ (crossbar+DAC+other, from paper)
        acam_adc_e = 1.5 * C * 8       # 1536 pJ (ADC: 1.8mW/ADC @ 1.2GHz)
        label = "ADC"
    total_pass = crossbar_e + acam_adc_e
    pass_results[name] = {
        "crossbar": crossbar_e,
        "acam_adc": acam_adc_e,
        "total": total_pass,
        "label": label,
        "area_mm2": s["area_mm2"],
        "tile_cells": s["tile_cells"],
    }

# ── Plot: single row — energy bars + 3 area-scaled pie charts ────────────
fig = plt.figure(figsize=(5.5, 2.0))
# ── Panel (a): Broken Y-axis stacked bar chart ──────────────────────────
arch_names = ["Proposed-1", "Proposed-2", "Azure-Lily"]
x = np.arange(len(arch_names))
bar_w = 0.5

ax_bot = fig.add_axes([0.10, 0.18, 0.38, 0.38])
ax_top = fig.add_axes([0.10, 0.58, 0.38, 0.28])

for ax in [ax_bot, ax_top]:
    for i, name in enumerate(arch_names):
        r = pass_results[name]
        acam_color = '#FFB3BA' if r["label"] == "ADC" else '#B5EAD7'
        ax.bar(x[i], r["crossbar"], bar_w, color='#7EC8E3',
               edgecolor='white', linewidth=0.5)
        ax.bar(x[i], r["acam_adc"], bar_w, bottom=r["crossbar"],
               color=acam_color, edgecolor='white', linewidth=0.5)

# Bottom: 0-200 pJ (Proposed bars visible)
ax_bot.set_ylim(0, 200)
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(arch_names, fontsize=7)
ax_bot.set_ylabel('')
ax_bot.set_yticks([0, 50, 100, 150])
ax_bot.grid(True, alpha=0.08, axis='y')

# Top: 1400-1700 pJ (Azure-Lily bar top)
ax_top.set_ylim(1400, 1700)
ax_top.set_xticks(x)
ax_top.set_xticklabels([])
ax_top.grid(True, alpha=0.08, axis='y')
ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_top.tick_params(bottom=False)

# Break marks
d = 0.015
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=0.8)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1-d, 1+d), **kwargs)
ax_bot.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

# Annotations
for i, name in enumerate(arch_names):
    r = pass_results[name]
    ratio_str = f'{r["total"]:.0f} pJ'
    if name == "Azure-Lily":
        ax_top.text(x[i], r["total"] + 10, ratio_str,
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    else:
        ax_bot.text(x[i], r["total"] + 5, ratio_str,
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')

ax_top.set_title('(a) Block Energy Breakdown', fontsize=8, fontweight='bold')
ax1 = ax_bot  # keep reference

# ── Right side: 3 area-scaled pie charts ─────────────────────────────────
# Azure-Lily area from their paper: 2320000 MWTA = 78564 µm²
# (area_power.py underestimates Azure-Lily; use published value)
MWTA_UM2 = 0.033864
AL_AREA_MWTA = 2320000  # from azure_lily arch XML
AL_AREA_MM2 = AL_AREA_MWTA * MWTA_UM2 / 1e6  # 0.078564 mm²

areas_mm2 = []
for name, cfg_name, R, C in ARCHS:
    if 'azure_lily' in cfg_name:
        areas_mm2.append(AL_AREA_MM2)
    else:
        areas_mm2.append(dpe_specs(R, C)["area_mm2"])
max_area = max(areas_mm2)

# Pie positions: spread across right 65% of figure
pie_x_centers = [0.58, 0.72, 0.86]  # figure x coords
pie_y_center = 0.50
max_pie_size = 0.30

# Title for pie section
fig.text(0.72, 0.90, '(b) IMC Area Breakdown', ha='center', fontsize=8, fontweight='bold')

for i, (name, cfg_name, R, C) in enumerate(ARCHS):
    s = dpe_specs(R, C)
    bd = s["area_breakdown"]

    if 'azure_lily' in cfg_name:
        # Use published area; scale breakdown proportionally
        total = AL_AREA_MM2
        model_total = s["area_mm2"]
        scale_factor = total / model_total
        crossbar = bd["crossbar_mm2"] * scale_factor
        acam_adc = bd["acam_mm2"] * scale_factor
        other = (total - crossbar - acam_adc)
    else:
        crossbar = bd["crossbar_mm2"]
        acam_adc = bd["acam_mm2"]
        other = sum(v for k, v in bd.items() if k not in ("crossbar_mm2", "acam_mm2"))
        total = s["area_mm2"]

    conv_label = 'ACAM' if 'nl_dpe' in cfg_name else 'ADC'
    conv_color = '#B5EAD7' if 'nl_dpe' in cfg_name else '#FFB3BA'

    # Scale pie size so visual area ∝ actual area
    scale = (total / max_area) ** 0.5
    pie_size = max_pie_size * scale

    cx = pie_x_centers[i]
    cy = pie_y_center
    # Aspect ratio correction (figure is wider than tall)
    aspect = 5.5 / 2.0
    pie_w = pie_size
    pie_h = pie_size * aspect * 0.5

    pie_ax = fig.add_axes([cx - pie_w/2, cy - pie_h/2, pie_w, pie_h])

    sizes = [crossbar, acam_adc, other]
    colors = ['#7EC8E3', conv_color, '#CBD5E1']

    wedges, _ = pie_ax.pie(sizes, colors=colors, startangle=90,
                            wedgeprops=dict(edgecolor='white', linewidth=1.0))

    # Percentage labels — all segments, bold black
    pcts = [v / total * 100 for v in sizes]
    seg_labels = ['Xbar', conv_label, '']  # "Other" just shows number
    for j, (w, pct, lbl) in enumerate(zip(wedges, pcts, seg_labels)):
        ang = (w.theta2 + w.theta1) / 2
        if pct > 8:
            # Inside the wedge
            r_t = 0.55
            x_t = r_t * np.cos(np.radians(ang))
            y_t = r_t * np.sin(np.radians(ang))
            txt = f'{lbl}\n{pct:.0f}%' if lbl else f'{pct:.0f}%'
            pie_ax.text(x_t, y_t, txt, ha='center', va='center',
                        fontsize=5.5, fontweight='bold', color='black')
        else:
            # Outside the wedge for small segments
            r_t = 1.2
            x_t = r_t * np.cos(np.radians(ang))
            y_t = r_t * np.sin(np.radians(ang))
            txt = f'{pct:.0f}%'
            pie_ax.text(x_t, y_t, txt, ha='center', va='center',
                        fontsize=5, fontweight='bold', color='black')

    # Label below pie
    pie_ax.text(0, -1.3, f'{name}\n{total:.3f} mm\u00b2',
                ha='center', va='top', fontsize=6.5, fontweight='bold')

# Shared legend at bottom right
from matplotlib.patches import Patch as MPatch
pie_legend = [
    MPatch(facecolor='#7EC8E3', edgecolor='white', label='Crossbar'),
    MPatch(facecolor='#B5EAD7', edgecolor='white', label='ACAM'),
    MPatch(facecolor='#FFB3BA', edgecolor='white', label='ADC'),
    MPatch(facecolor='#CBD5E1', edgecolor='white', label='Other'),
]
fig.legend(handles=pie_legend, fontsize=6, loc='lower right',
           ncol=4, framealpha=0.9, bbox_to_anchor=(0.95, 0.02),
           columnspacing=0.6, handletextpad=0.3)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# Print summary
print(f"\nFC + Activation per token:")
for name in arch_names:
    r = fc_results[name]
    print(f"  {name}: DPE={r['dpe']:.1f}, Act={r['act']:.1f}, Mem={r['mem']:.1f}, Total={r['total']:.1f} pJ")
print(f"  Proposed-1 is {fc_results["Azure-Lily"]["total"]/fc_results["Proposed-1"]["total"]:.0f}× cheaper than Azure-Lily")

print(f"\nPer VMM pass:")
for name in arch_names:
    r = pass_results[name]
    print(f"  {name}: Crossbar={r['crossbar']:.1f}, {r['label']}={r['acam_adc']:.1f}, Total={r['total']:.1f} pJ, Area={r['area_mm2']*1e6:.0f} µm²")
