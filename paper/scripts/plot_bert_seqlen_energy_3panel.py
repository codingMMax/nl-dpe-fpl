#!/usr/bin/env python3
"""3-panel figure: BERT-Tiny energy convergence story (seq_len sweep).

Left:   % breakdown bars (DIMM vs Proj+FFN vs Other) — both archs, across N
Middle: DIMM-internal split (DPE analog vs Digital DSP+CLB) — both archs, across N
Right:  Line plot: overall energy ratio + DIMM-only ratio vs N

Input: benchmarks/results/bert_tiny_seqlen_fixed_fmax.csv
Output: paper/figures/benchmarks/bert_seqlen_energy_3panel.pdf
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
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from style_constants import (apply_style, ARCH_COLORS, BREAKDOWN_COLORS,
                              BASELINE_COLOR, BASELINE_LS, BASELINE_ALPHA,
                              ANNOT_FONTSIZE, ANNOT_FONTWEIGHT)
apply_style()

CSV_PATH = ROOT / "benchmarks" / "results" / "bert_tiny_seqlen_fixed_fmax.csv"
OUT_PATH = ROOT / "paper" / "figures" / "benchmarks" / "bert_seqlen_energy_3panel.pdf"

# ── Load data ────────────────────────────────────────────────────────────
raw = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["arch"], int(row["seq_len"]))
        raw[key] = {k: float(v) for k, v in row.items()
                    if k not in ("arch", "seq_len") and v.replace('.','',1).replace('-','',1).isdigit()}

SEQ_LENS = [128, 256, 512, 1024, 2048]
n_groups = len(SEQ_LENS)
x = np.arange(n_groups)

def _get(arch, N, key):
    return raw[(arch, N)][key]

# ── Derived: per-operation % ─────────────────────────────────────────────
ALL_ARCHS = ("proposed", "al_like", "azurelily")
pct = {}
for arch in ALL_ARCHS:
    for cat in ("dimm", "proj_ffn", "other"):
        vals = []
        for N in SEQ_LENS:
            total = _get(arch, N, "energy_pj")
            if cat == "dimm":
                vals.append(_get(arch, N, "energy_dimm_pj") / total * 100)
            elif cat == "proj_ffn":
                vals.append(_get(arch, N, "energy_proj_ffn_pj") / total * 100)
            else:
                vals.append(_get(arch, N, "energy_other_pj") / total * 100)
        pct[(arch, cat)] = vals

# DIMM-internal: DPE (analog) vs Digital (DSP+CLB)
dimm_dpe_pct = {}
for arch in ALL_ARCHS:
    dimm_dpe_pct[arch] = []
    for N in SEQ_LENS:
        dimm_total = _get(arch, N, "energy_dimm_pj")
        dpe = _get(arch, N, "energy_dpe_pj")
        dimm_dpe = min(dpe, dimm_total)
        dimm_dpe_pct[arch].append(dimm_dpe / dimm_total * 100 if dimm_total > 0 else 0)

# Energy ratios: Azure-Lily / Proposed
energy_ratio_p1 = [_get("azurelily", N, "energy_pj") / _get("proposed", N, "energy_pj")
                   for N in SEQ_LENS]
energy_ratio_p2 = [_get("azurelily", N, "energy_pj") / _get("al_like", N, "energy_pj")
                   for N in SEQ_LENS]
dimm_ratio_p1 = [_get("azurelily", N, "energy_dimm_pj") / _get("proposed", N, "energy_dimm_pj")
                 for N in SEQ_LENS]
dimm_ratio_p2 = [_get("azurelily", N, "energy_dimm_pj") / _get("al_like", N, "energy_dimm_pj")
                 for N in SEQ_LENS]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.6),
                                     gridspec_kw={'width_ratios': [1, 1, 1]})
fig.subplots_adjust(wspace=0.50, top=0.78, bottom=0.18, left=0.06, right=0.98)

# ── Panel left: DIMM vs Proj+FFN vs Other stacked bars ──────────────────
bar_w = 0.25
arch_list = ["proposed", "al_like", "azurelily"]
arch_hatch = {"proposed": None, "al_like": "...", "azurelily": "///"}
arch_display = {"proposed": "P1", "al_like": "P2", "azurelily": "AL"}
cat_colors = ["#7EC8E3", "#FFB3BA", "#D5D5D5"]  # light blue, pink, light gray
cat_labels = ["DIMM", "Proj+FFN", "Other"]
cat_keys = ["dimm", "proj_ffn", "other"]

for ai, arch in enumerate(arch_list):
    offset = (ai - 1) * bar_w
    hatch = arch_hatch[arch]
    for i in range(n_groups):
        bottom = 0
        for ci, ck in enumerate(cat_keys):
            val = pct[(arch, ck)][i]
            ax1.bar(x[i] + offset, val, bar_w * 0.9, bottom=bottom,
                    color=cat_colors[ci], edgecolor='white', linewidth=0.5,
                    hatch=hatch, zorder=3)
            # Only annotate Proj+FFN for N=128,256,512 (first 3)
            if ck == "proj_ffn" and val > 2 and i < 3:
                ax1.text(x[i] + offset, bottom + val / 2, f'{val:.0f}%',
                         ha='center', va='center', fontsize=3.5,
                         fontweight='bold', color='black', zorder=5)
            bottom += val

ax1.set_xticks(x)
ax1.set_xticklabels([str(N) for N in SEQ_LENS])
ax1.set_xlabel('Sequence Length (N)')
ax1.set_ylabel('BERT-Tiny Energy Breakdown (%)')
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.08, axis='y', zorder=0)

# Left panel legend
a_handles = [
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, label='Proposed-1'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, hatch='...', label='Proposed-2'),
    mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, hatch='///', label='Azure-Lily'),
]
c_handles = [mpatches.Patch(facecolor=cat_colors[ci], edgecolor='white',
             label=cat_labels[ci]) for ci in range(3)]
left_handles = a_handles + c_handles
LEGEND_Y = 0.93
bbox_left = (ax1.get_position().x0 + ax1.get_position().width / 2, LEGEND_Y)
fig.legend(handles=left_handles, loc='upper center', ncol=3, fontsize=4.5,
           frameon=True, framealpha=0.9, bbox_to_anchor=bbox_left,
           columnspacing=0.5, handletextpad=0.3)

# ── Panel middle: 3-category stacked bars, normalized to min per S ───────
# Crossbar vs Conversion (ACAM/ADC) vs Digital (DSP+CLB+BRAM)
# Highlights the ADC→ACAM replacement benefit
CAT3_GROUPS = [
    ("Crossbar",       ["energy_crossbar_pj"]),
    ("Conversion (ACAM/ADC)", ["energy_adc_acam_pj"]),
    ("Digital Fabric (DSP+CLB)", ["energy_clb_pj", "energy_dsp_pj", "energy_bram_pj"]),
]
CAT3_LABELS = [g[0] for g in CAT3_GROUPS]
CAT3_COLORS = ["#2B9E8F", "#B5EAD7", "#FFDAC1"]  # dark teal, mint, peach

for ai, arch in enumerate(arch_list):
    offset = (ai - 1) * bar_w
    hatch = arch_hatch[arch]
    for i, N in enumerate(SEQ_LENS):
        cat_vals = [sum(_get(arch, N, k) for k in keys) for _, keys in CAT3_GROUPS]
        total = sum(cat_vals)
        all_totals = [sum(sum(_get(a, N, k) for k in keys)
                         for _, keys in CAT3_GROUPS) for a in arch_list]
        min_total = min(all_totals)
        scale = total / min_total if min_total > 0 else 1.0
        bottom = 0
        for ci, cv in enumerate(cat_vals):
            h = (cv / total) * scale if total > 0 else 0
            ax2.bar(x[i] + offset, h, bar_w * 0.9, bottom=bottom,
                    color=CAT3_COLORS[ci], edgecolor='white', linewidth=0.3,
                    hatch=hatch, zorder=3)
            bottom += h
    if arch == "azurelily":
        for i, N in enumerate(SEQ_LENS):
            all_totals = [sum(sum(_get(a, N, k) for k in keys)
                             for _, keys in CAT3_GROUPS) for a in arch_list]
            min_total = min(all_totals)
            al_total = sum(sum(_get(arch, N, k) for k in keys) for _, keys in CAT3_GROUPS)
            ratio = al_total / min_total if min_total > 0 else 1.0
            ax2.text(x[i] + (2 - 1) * bar_w, ratio + 0.03, f'{ratio:.1f}\u00d7',
                     ha='center', va='bottom', fontsize=4, fontweight='bold',
                     color='#333', zorder=5)

ax2.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax2.set_xticks(x)
ax2.set_xticklabels([str(N) for N in SEQ_LENS])
ax2.set_xlabel('Sequence Length (N)')
ax2.set_ylabel('Normalized Energy (min = 1.0)')
ax2.set_ylim(0, None)
ax2.grid(True, alpha=0.08, axis='y', zorder=0)

# Middle panel legend
cat3_handles = [mpatches.Patch(facecolor=CAT3_COLORS[ci], edgecolor='white',
                label=CAT3_LABELS[ci]) for ci in range(3)]
mid_handles = a_handles + cat3_handles
bbox_mid = (ax2.get_position().x0 + ax2.get_position().width / 2, LEGEND_Y)
fig.legend(handles=mid_handles, loc='upper center', ncol=3, fontsize=4,
           frameon=True, framealpha=0.9, bbox_to_anchor=bbox_mid,
           columnspacing=0.3, handletextpad=0.2)

# ── Panel right: Normalized Inf/J/mm² (Azure-Lily = 1.0) ────────────────
# Metric: (throughput_per_j / area) normalized to Azure-Lily per S.
# >1 means better area-energy efficiency than Azure-Lily.
import matplotlib.lines as mlines

ARCH_SCATTER = {
    "proposed":  {"color": "#2B9E8F", "marker": "o", "label": "Proposed-1"},
    "al_like":   {"color": "#E8853D", "marker": "s", "label": "Proposed-2"},
    "azurelily": {"color": "#9B59B6", "marker": "^", "label": "Azure-Lily"},
}

# Compute Azure-Lily baseline per S
al_eff = []
for N in SEQ_LENS:
    tj = _get("azurelily", N, "throughput_per_j")
    a = _get("azurelily", N, "used_area_mm2")
    al_eff.append(tj / a)

for arch in arch_list:
    tput_j = [_get(arch, N, "throughput_per_j") for N in SEQ_LENS]
    areas = [_get(arch, N, "used_area_mm2") for N in SEQ_LENS]
    eff = [tj / a for tj, a in zip(tput_j, areas)]
    norm = [e / al for e, al in zip(eff, al_eff)]  # normalized to AL = 1.0
    s = ARCH_SCATTER[arch]
    ax3.plot(x, norm, color=s["color"], linewidth=1.8,
             marker=s["marker"], markersize=5,
             markeredgecolor='white', markeredgewidth=0.6,
             label=s["label"], zorder=4)
    # Annotate values
    for i, N in enumerate(SEQ_LENS):
        if arch != "azurelily":
            ax3.annotate(f'{norm[i]:.2f}', (x[i], norm[i]),
                         textcoords="offset points",
                         xytext=(0, 7) if arch == "proposed" else (0, -12),
                         fontsize=4.5, color=s["color"], fontweight='bold',
                         ha='center')

ax3.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1, linestyle=BASELINE_LS,
            alpha=BASELINE_ALPHA)
ax3.set_xticks(x)
ax3.set_xticklabels([str(N) for N in SEQ_LENS])
ax3.set_xlabel('Sequence Length (N)')
ax3.set_ylabel('Normalized Inf/J/mm²\n(Azure-Lily = 1.0)')
ax3.grid(True, alpha=0.15)
ax3.legend(fontsize=5, loc='upper right', frameon=True, framealpha=0.9)

fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")

# Print summary
print(f"\n{'N':>5} {'AL/P1':>7} {'AL/P2':>7} {'DIMM/P1':>8} {'DIMM/P2':>8} {'P1 DIMM%':>9} {'P2 DIMM%':>9} {'AL DIMM%':>9}")
print("-" * 65)
for i, N in enumerate(SEQ_LENS):
    print(f"{N:>5} {energy_ratio_p1[i]:>6.2f}\u00d7 {energy_ratio_p2[i]:>6.2f}\u00d7 "
          f"{dimm_ratio_p1[i]:>7.2f}\u00d7 {dimm_ratio_p2[i]:>7.2f}\u00d7 "
          f"{pct[('proposed','dimm')][i]:>8.1f}% {pct[('al_like','dimm')][i]:>8.1f}% "
          f"{pct[('azurelily','dimm')][i]:>8.1f}%")
