#!/usr/bin/env python3
"""3-panel figure with 3 different color palettes for comparison.

Generates 3 separate PDFs to compare side by side.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from style_constants import apply_style, BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT
apply_style()

CSV_PATH = SCRIPT_DIR / "bert_energy_sweep.csv"

# ── Load data ────────────────────────────────────────────────────────────
data = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        data[key] = {k: float(v) for k, v in row.items() if k not in ("arch", "seq_len")}

SEQ_LENS = [256, 512, 1024, 2048]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

# ── Derived metrics ──────────────────────────────────────────────────────
pct = {}
for arch in ("Proposed", "Azure-Lily"):
    for cat in ("dimm", "projffn", "other"):
        vals = []
        for N in SEQ_LENS:
            d = data[(arch, N)]
            total = d["total_pj"]
            if cat == "dimm":   vals.append(d["dimm_pj"] / total * 100)
            elif cat == "projffn": vals.append((d["proj_pj"] + d["ffn_pj"]) / total * 100)
            else:               vals.append(d["other_pj"] / total * 100)
        pct[(arch, cat)] = vals

dimm_dpe_pct = {}
for arch in ("Proposed", "Azure-Lily"):
    dimm_dpe_pct[arch] = []
    for N in SEQ_LENS:
        d = data[(arch, N)]
        dtotal = d["dimm_dpe_pj"] + d["dimm_fabric_pj"]
        dimm_dpe_pct[arch].append(d["dimm_dpe_pj"] / dtotal * 100 if dtotal > 0 else 0)

energy_ratio = [data[("Azure-Lily", N)]["total_pj"] / data[("Proposed", N)]["total_pj"] for N in SEQ_LENS]
dimm_ratio = [data[("Azure-Lily", N)]["dimm_pj"] / data[("Proposed", N)]["dimm_pj"] for N in SEQ_LENS]

# ── Color palettes ───────────────────────────────────────────────────────
PALETTES = {
    "ocean": {
        # Deep ocean → warm sand contrast
        "dimm": "#264653",       # dark teal
        "projffn": "#E76F51",    # burnt sienna
        "other": "#E0E1DD",      # warm gray
        "dpe": "#2A9D8F",        # teal
        "fabric": "#E9C46A",     # golden sand
        "line_overall": "#264653",
        "line_dimm": "#E76F51",
        "text": "black",
    },
    "nordic": {
        # Cool Scandinavian palette
        "dimm": "#5E81AC",       # steel blue
        "projffn": "#BF616A",    # muted rose
        "other": "#D8DEE9",      # frost gray
        "dpe": "#A3BE8C",        # sage green
        "fabric": "#EBCB8B",     # warm gold
        "line_overall": "#5E81AC",
        "line_dimm": "#BF616A",
        "text": "black",
    },
    "dusk": {
        # Warm sunset / dusk palette
        "dimm": "#4A4E69",       # muted slate
        "projffn": "#C9ADA7",    # dusty rose
        "other": "#F2E9E4",      # cream
        "dpe": "#9A8C98",        # mauve
        "fabric": "#22223B",     # dark indigo
        "line_overall": "#4A4E69",
        "line_dimm": "#C9ADA7",
        "text": "white",
    },
}


def make_figure(palette_name, pal):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.6),
                                         gridspec_kw={'width_ratios': [1, 1, 0.9]})
    fig.subplots_adjust(wspace=0.45, top=0.78, bottom=0.18)

    bar_w = 0.35
    arch_hatch = {"Proposed": None, "Azure-Lily": "///"}
    cat_colors = [pal["dimm"], pal["projffn"], pal["other"]]
    cat_labels = ["DIMM (QK\u1d40+S\u00d7V)", "Proj+FFN", "Other"]
    cat_keys = ["dimm", "projffn", "other"]
    txt_color = pal["text"]

    # Left panel
    for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
        offset = (ai - 0.5) * bar_w
        hatch = arch_hatch[arch]
        for i in range(n_groups):
            bottom = 0
            for ci, ck in enumerate(cat_keys):
                val = pct[(arch, ck)][i]
                ax1.bar(x[i] + offset, val, bar_w, bottom=bottom,
                        color=cat_colors[ci], edgecolor='white', linewidth=0.5,
                        hatch=hatch, zorder=3)
                if val > 10:
                    ax1.text(x[i] + offset, bottom + val / 2, f'{val:.0f}%',
                             ha='center', va='center', fontsize=4.5,
                             fontweight='bold', color=txt_color, zorder=5)
                bottom += val

    ax1.set_xticks(x); ax1.set_xticklabels([str(N) for N in SEQ_LENS])
    ax1.set_xlabel('Sequence Length (N)'); ax1.set_ylabel('Energy Composition (%)')
    ax1.set_ylim(0, 105); ax1.grid(True, alpha=0.08, axis='y', zorder=0)

    # Middle panel
    for ai, arch in enumerate(["Proposed", "Azure-Lily"]):
        offset = (ai - 0.5) * bar_w
        hatch = arch_hatch[arch]
        dpe = dimm_dpe_pct[arch]
        fab = [100 - d for d in dpe]
        for i in range(n_groups):
            ax2.bar(x[i] + offset, dpe[i], bar_w, color=pal["dpe"],
                    edgecolor='white', linewidth=0.5, hatch=hatch, zorder=3)
            ax2.bar(x[i] + offset, fab[i], bar_w, bottom=dpe[i], color=pal["fabric"],
                    edgecolor='white', linewidth=0.5, hatch=hatch, zorder=3)
            if dpe[i] > 10:
                ax2.text(x[i] + offset, dpe[i] / 2, f'{dpe[i]:.0f}%',
                         ha='center', va='center', fontsize=4.5,
                         fontweight='bold', color=txt_color, zorder=5)
            if fab[i] > 10:
                ax2.text(x[i] + offset, dpe[i] + fab[i] / 2, f'{fab[i]:.0f}%',
                         ha='center', va='center', fontsize=4.5,
                         fontweight='bold', color=txt_color, zorder=5)

    ax2.set_xticks(x); ax2.set_xticklabels([str(N) for N in SEQ_LENS])
    ax2.set_xlabel('Sequence Length (N)'); ax2.set_ylabel('DIMM Energy Composition (%)')
    ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.08, axis='y', zorder=0)

    # Shared legend
    a_handles = [
        mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, label='K=2 (Proposed)'),
        mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, hatch='///', label='Azure-Lily'),
    ]
    c_handles = [mpatches.Patch(facecolor=cat_colors[ci], edgecolor='white', label=cat_labels[ci]) for ci in range(3)]
    d_handles = [
        mpatches.Patch(facecolor=pal["dpe"], edgecolor='white', label='DPE (crossbar)'),
        mpatches.Patch(facecolor=pal["fabric"], edgecolor='white', label='FPGA fabric'),
    ]
    bbox_mid = ((ax1.get_position().x0 + ax2.get_position().x1) / 2, 0.97)
    fig.legend(handles=a_handles + c_handles + d_handles, loc='upper center', ncol=4,
               fontsize=5.5, frameon=True, framealpha=0.9, bbox_to_anchor=bbox_mid,
               columnspacing=0.8, handletextpad=0.3)

    # Right panel
    ax3.plot(x, energy_ratio, color=pal["line_overall"], linewidth=2,
             marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.8,
             label='Overall (AL/K=2)', zorder=4)
    ax3.plot(x, dimm_ratio, color=pal["line_dimm"], linewidth=2, linestyle='--',
             marker='D', markersize=4, markeredgecolor='white', markeredgewidth=0.8,
             label='DIMM only (AL/K=2)', zorder=4)
    for i, (er, dr) in enumerate(zip(energy_ratio, dimm_ratio)):
        ax3.annotate(f'{er:.2f}\u00d7', xy=(x[i], er), xytext=(0, 8),
                     textcoords='offset points', ha='center', va='bottom',
                     fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                     color=pal["line_overall"])
        y_off = -10 if i > 0 else 8
        va = 'top' if y_off < 0 else 'bottom'
        ax3.annotate(f'{dr:.2f}\u00d7', xy=(x[i], dr), xytext=(0, y_off),
                     textcoords='offset points', ha='center', va=va,
                     fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT,
                     color=pal["line_dimm"])

    ax3.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS, alpha=BASELINE_ALPHA)
    ax3.set_xticks(x); ax3.set_xticklabels([str(N) for N in SEQ_LENS])
    ax3.set_xlabel('Sequence Length (N)'); ax3.set_ylabel('Energy Ratio (AL / Proposed)')
    ax3.set_ylim(0.9, 1.85); ax3.legend(fontsize=5.5, loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.1)

    out = ROOT / "paper" / "figures" / "benchmarks" / f"bert_energy_3panel_{palette_name}.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


for name, pal in PALETTES.items():
    make_figure(name, pal)
