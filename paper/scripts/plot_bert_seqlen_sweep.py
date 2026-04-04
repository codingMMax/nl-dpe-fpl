#!/usr/bin/env python3
"""BERT-Tiny Seq_Len Sweep Plots — paper narrative figures.

Story: NL-DPE's analog ACAM keeps DIMM energy efficient as S grows,
while Azure-Lily's digital DSP DIMM energy explodes.

Input: benchmarks/results/bert_tiny_seqlen_variable_fmax.csv
Output:
  bert_seqlen_analysis.pdf   — (a) Energy/inf ratio, (b) DPE vs Digital energy share
  bert_seqlen_efficiency.pdf — (a) Norm. Tput/mm², (b) Norm. Inf/J
  bert_seqlen_latency.pdf    — (a) DIMM latency %, (b) Speedup breakdown
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
from style_constants import (apply_style, ARCH_COLORS, ARCH_MARKERS,
                              ARCH_LINESTYLES, BASELINE_COLOR, BASELINE_LS,
                              BASELINE_ALPHA, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT,
                              BREAKDOWN_COLORS)
apply_style()

CSV_PATH_FIXED = ROOT_DIR / "benchmarks" / "results" / "bert_tiny_seqlen_fixed_fmax.csv"
CSV_PATH_VAR = ROOT_DIR / "benchmarks" / "results" / "bert_tiny_seqlen_variable_fmax.csv"
FIG_DIR = SCRIPT_DIR.parent / "figures" / "benchmarks"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY = {"proposed": "Proposed-1", "al_like": "Proposed-2", "azurelily": "Azure-Lily"}
ARCHS = ["proposed", "al_like", "azurelily"]

# ── Load CSVs ────────────────────────────────────────────────────────────
def _load_csv(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            arch = row["arch"]
            S = int(row["seq_len"])
            d = {}
            for k, v in row.items():
                if k in ("arch", "seq_len"):
                    continue
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
            out[(arch, S)] = d
    return out

data = _load_csv(CSV_PATH_FIXED)       # fixed fmax — for analysis, efficiency, resilience
data_var = _load_csv(CSV_PATH_VAR)      # variable fmax — for latency plot

SEQ_LENS = [s for s in sorted(set(r[1] for r in data.keys())) if s <= 2048]

def _get(arch, S, key):
    return data[(arch, S)][key]

def _setup_xaxis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels([str(n) for n in SEQ_LENS], fontsize=7)
    ax.minorticks_off()
    ax.set_xlim(SEQ_LENS[0] * 0.7, SEQ_LENS[-1] * 1.3)
    ax.set_xlabel("Sequence Length (N)")


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: bert_seqlen_analysis
#   (a) Energy per inference ratio (Proposed / Azure-Lily) — shows resilience
#   (b) DPE (analog) vs Digital energy share — shows WHY proposed is better
# ══════════════════════════════════════════════════════════════════════════
def plot_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    fig.subplots_adjust(wspace=0.32, top=0.88)

    ax1.text(-0.14, 1.08, '(a)', transform=ax1.transAxes, fontsize=10,
             fontweight='bold', va='top')
    ax2.text(-0.12, 1.08, '(b)', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', va='top')

    # ── Panel (a): Energy per inference normalized to Azure-Lily ──
    # Below 1.0 = Proposed is more energy efficient
    # Azure-Lily = constant 1.0 baseline (dashed line only, no data points)
    for arch in ["proposed", "al_like"]:
        dn = DISPLAY[arch]
        ratios = [_get(arch, S, "energy_pj") / _get("azurelily", S, "energy_pj")
                  for S in SEQ_LENS]
        ax1.plot(SEQ_LENS, ratios, color=ARCH_COLORS[dn], linewidth=2,
                 linestyle=ARCH_LINESTYLES[dn], marker=ARCH_MARKERS[dn],
                 markersize=5, markeredgecolor="white", markeredgewidth=0.8,
                 label=dn)
        # Annotate last point only, shifted left to avoid overlap
        ax1.annotate(f'{ratios[-1]:.2f}×',
                     xy=(SEQ_LENS[-1], ratios[-1]),
                     xytext=(-40, 5 if arch == "proposed" else -10),
                     textcoords='offset points', fontsize=7,
                     color=ARCH_COLORS[dn], fontweight='bold')

    ax1.axhline(y=1.0, color=ARCH_COLORS["Azure-Lily"], linestyle='--',
                alpha=0.7, linewidth=1.5)
    _setup_xaxis(ax1)
    ax1.set_ylabel("Energy / Inference\n(normalized to Azure-Lily)")
    ax1.set_ylim(0.3, 1.2)
    ax1.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax1.set_title("Energy Efficiency vs Sequence Length", fontsize=8)
    ax1.grid(True, alpha=0.1)

    # ── Panel (b): Paired stacked bars — normalized height shows energy ratio ──
    # Each seq_len has 2 bars side by side: Proposed (left) and Azure-Lily (right)
    # Heights are normalized so the LOWER energy arch = 1.0, the other is taller
    # Stacking: DPE | DSP | CLB | MEM
    x = np.arange(len(SEQ_LENS))
    width = 0.35

    comp_colors = {'DPE': '#6366F1', 'DSP': '#F59E0B', 'CLB': '#CBD5E1', 'MEM': '#06B6D4'}

    for i, (arch, offset) in enumerate([("proposed", -width/2), ("azurelily", width/2)]):
        dn = DISPLAY[arch]
        for j, S in enumerate(SEQ_LENS):
            # Normalize to the lower-energy arch at this S
            p_total = _get("proposed", S, "energy_pj")
            a_total = _get("azurelily", S, "energy_pj")
            norm_base = min(p_total, a_total)
            my_total = _get(arch, S, "energy_pj")
            scale = my_total / norm_base  # >=1.0, the cheaper arch = 1.0

            dpe = _get(arch, S, "energy_dpe_pj") / my_total * scale
            dsp = _get(arch, S, "energy_dsp_pj") / my_total * scale
            clb = _get(arch, S, "energy_clb_pj") / my_total * scale
            mem = _get(arch, S, "energy_mem_pj") / my_total * scale

            bottom = 0
            for val, comp in [(dpe, 'DPE'), (dsp, 'DSP'), (clb, 'CLB'), (mem, 'MEM')]:
                if val < 0.001:
                    bottom += val
                    continue
                lbl = comp if (i == 0 and j == 0) else None
                ax2.bar(x[j] + offset, val, width * 0.9, bottom=bottom,
                        color=comp_colors[comp], edgecolor='white', linewidth=0.3,
                        label=lbl)
                # Annotate % inside bar if tall enough
                if val > 0.08 * scale:
                    pct = val / scale * 100
                    ax2.text(x[j] + offset, bottom + val / 2, f'{pct:.0f}%',
                             ha='center', va='center', fontsize=5,
                             fontweight='bold', color='white' if comp != 'CLB' else '#333')
                bottom += val

    # Arch labels on top
    for j in range(len(SEQ_LENS)):
        p_total = _get("proposed", SEQ_LENS[j], "energy_pj")
        a_total = _get("azurelily", SEQ_LENS[j], "energy_pj")
        norm_base = min(p_total, a_total)
        p_h = p_total / norm_base
        a_h = a_total / norm_base
        ax2.text(x[j] - width/2, p_h + 0.03, 'P1', ha='center', fontsize=5,
                 color=ARCH_COLORS["Proposed-1"], fontweight='bold')
        ax2.text(x[j] + width/2, a_h + 0.03, 'AL', ha='center', fontsize=5,
                 color=ARCH_COLORS["Azure-Lily"], fontweight='bold')

    ax2.axhline(y=1.0, color='#888', linewidth=0.8, linestyle=':', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in SEQ_LENS], fontsize=7)
    ax2.set_xlabel("Sequence Length (N)")
    ax2.set_ylabel("Normalized Energy\n(shorter bar = 1.0\u00d7)")
    ax2.legend(fontsize=6, loc='upper left', framealpha=0.9, ncol=2)
    ax2.set_title("(b) Energy Breakdown (norm. to lower)", fontsize=8)
    ax2.grid(True, alpha=0.08, axis='y')

    out = FIG_DIR / "bert_seqlen_analysis.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: bert_seqlen_efficiency
#   (a) Normalized Inference/s/mm², (b) Normalized Inference/s/J
#   Both normalized to Azure-Lily = 1.0× at each seq_len
# ══════════════════════════════════════════════════════════════════════════
def plot_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.subplots_adjust(wspace=0.25)

    metrics = [
        (ax1, "throughput_per_mm2", "Normalized Inference/s/mm\u00b2"),
        (ax2, "throughput_per_j", "Normalized Inference/s/J"),
    ]

    for ax, key, ylabel in metrics:
        _proposed_ys = None
        _al_like_ys = None

        for arch in ARCHS:
            dn = DISPLAY[arch]
            ys = [_get(arch, S, key) / _get("azurelily", S, key) for S in SEQ_LENS]
            ax.plot(SEQ_LENS, ys, color=ARCH_COLORS[dn], linewidth=1.5,
                    linestyle=ARCH_LINESTYLES[dn],
                    marker=ARCH_MARKERS[dn], markersize=4,
                    markeredgecolor="white", markeredgewidth=0.8, label=dn)
            if arch == "proposed":
                _proposed_ys = ys
            elif arch == "al_like":
                _al_like_ys = ys

        # Asymptotic lines — only on right panel (Inf/J), skip left panel (Tput/mm²)
        if ax == ax2:
            for arch_ys in [_proposed_ys, _al_like_ys]:
                if arch_ys:
                    asymp = arch_ys[-1]
                    ax.axhline(y=asymp, color="black", linewidth=0.8,
                               linestyle=":", alpha=0.4)
            # Manual annotation positions to avoid overlap
            if _proposed_ys:
                ax.text(1024, 1.4, f"{_proposed_ys[-1]:.2f}\u00d7",
                        ha="center", va="center", fontsize=9, color="black", fontstyle="italic")
            if _al_like_ys:
                ax.text(1024, 1.7, f"{_al_like_ys[-1]:.2f}\u00d7",
                        ha="center", va="center", fontsize=9, color="black", fontstyle="italic")

        ax.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1.2,
                   linestyle=BASELINE_LS, alpha=0.6)
        _setup_xaxis(ax)
        ax.set_ylabel(ylabel)
        all_vals = [_get(a, S, key) / _get("azurelily", S, key) for a in ARCHS for S in SEQ_LENS]
        ax.set_ylim(bottom=min(all_vals) * 0.85, top=max(all_vals) * 1.15)
        ax.grid(True, alpha=0.1)

    # Shared legend
    handles = [Patch(facecolor=ARCH_COLORS[DISPLAY[a]], label=DISPLAY[a]) for a in ARCHS]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.01), frameon=False)

    out = FIG_DIR / "bert_seqlen_efficiency.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


def plot_efficiency_varfmax():
    """Efficiency with reported (VTR) frequency + raw Fmax panel."""

    def _getv(arch, S, key):
        return data_var[(arch, S)][key]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7.16, 2.6),
                                         gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.subplots_adjust(wspace=0.35, top=0.78, bottom=0.18, left=0.06, right=0.98)

    # ── Left panel: Tput/mm² (reported Fmax) ──
    for arch in ["proposed", "al_like"]:
        dn = DISPLAY[arch]
        color = ARCH_COLORS[dn]
        ys = [_getv(arch, S, "throughput_per_mm2") / _getv("azurelily", S, "throughput_per_mm2")
              for S in SEQ_LENS]
        ax1.plot(SEQ_LENS, ys, color=color, linewidth=2,
                 linestyle=ARCH_LINESTYLES[dn],
                 marker=ARCH_MARKERS[dn], markersize=5,
                 markeredgecolor="white", markeredgewidth=0.8)

    # Azure-Lily flat line at 1.0×
    ax1.plot(SEQ_LENS, [1.0] * len(SEQ_LENS), color=ARCH_COLORS["Azure-Lily"], linewidth=2,
             linestyle=ARCH_LINESTYLES["Azure-Lily"],
             marker=ARCH_MARKERS["Azure-Lily"], markersize=5,
             markeredgecolor="white", markeredgewidth=0.8)
    _setup_xaxis(ax1)
    ax1.tick_params(axis='both', labelsize=8.5)
    ax1.set_ylabel('Inference/s/mm\u00b2', fontsize=11, fontweight='normal')
    ax1.set_xlabel('Sequence Length (N)', fontsize=9.5)
    all_vals = [_getv(a, S, "throughput_per_mm2") / _getv("azurelily", S, "throughput_per_mm2")
                for a in ["proposed", "al_like"] for S in SEQ_LENS]
    ax1.set_ylim(bottom=min(min(all_vals), 0.7) * 0.9, top=max(all_vals) * 1.15)
    ax1.grid(True, alpha=0.1)

    # ── Right panel: Inf/J (reported Fmax) ──
    for arch in ["proposed", "al_like"]:
        dn = DISPLAY[arch]
        color = ARCH_COLORS[dn]
        ys = [_getv(arch, S, "throughput_per_j") / _getv("azurelily", S, "throughput_per_j")
              for S in SEQ_LENS]
        ax2.plot(SEQ_LENS, ys, color=color, linewidth=2,
                 linestyle=ARCH_LINESTYLES[dn],
                 marker=ARCH_MARKERS[dn], markersize=5,
                 markeredgecolor="white", markeredgewidth=0.8)

    # Azure-Lily flat line at 1.0×
    ax2.plot(SEQ_LENS, [1.0] * len(SEQ_LENS), color=ARCH_COLORS["Azure-Lily"], linewidth=2,
             linestyle=ARCH_LINESTYLES["Azure-Lily"],
             marker=ARCH_MARKERS["Azure-Lily"], markersize=5,
             markeredgecolor="white", markeredgewidth=0.8)
    _setup_xaxis(ax2)
    ax2.tick_params(axis='both', labelsize=8.5)
    ax2.set_ylabel('Inference/J', fontsize=11, fontweight='normal')
    ax2.set_xlabel('Sequence Length (N)', fontsize=9.5)
    all_vals = [_getv(a, S, "throughput_per_j") / _getv("azurelily", S, "throughput_per_j")
                for a in ["proposed", "al_like"] for S in SEQ_LENS]
    ax2.set_ylim(bottom=min(min(all_vals), 0.8) * 0.9, top=max(all_vals) * 1.15)
    ax2.grid(True, alpha=0.1)

    # Asymptotic lines at 1.4× and 1.7×
    ax2.axhline(y=1.4, color='#555', linewidth=1.0, linestyle='--', alpha=0.5)
    ax2.text(SEQ_LENS[1], 1.4 + 0.02, '1.4\u00d7', ha='center', va='bottom',
             fontsize=8, color='#555', fontweight='bold')
    ax2.axhline(y=1.7, color='#555', linewidth=1.0, linestyle='--', alpha=0.5)
    ax2.text(SEQ_LENS[2], 1.7 + 0.02, '1.7\u00d7', ha='center', va='bottom',
             fontsize=8, color='#555', fontweight='bold')

    # ── First panel (ax0): Raw Fmax (MHz) ──
    for arch in ["proposed", "al_like", "azurelily"]:
        dn = DISPLAY[arch]
        color = ARCH_COLORS[dn]
        fmax_vals = [_getv(arch, S, "fmax_vtr") for S in SEQ_LENS]
        ax0.plot(SEQ_LENS, fmax_vals, color=color, linewidth=2,
                 linestyle=ARCH_LINESTYLES[dn],
                 marker=ARCH_MARKERS[dn], markersize=5,
                 markeredgecolor="white", markeredgewidth=0.8)

    _setup_xaxis(ax0)
    ax0.tick_params(axis='both', labelsize=8.5)
    ax0.set_ylabel('Fmax (MHz)', fontsize=11, fontweight='normal')
    ax0.set_xlabel('Sequence Length (N)', fontsize=9.5)
    ax0.set_ylim(0, 160)
    ax0.set_yticks([0, 40, 80, 120, 160])
    ax0.grid(True, alpha=0.1)

    # Shared legend
    arch_h = [Patch(facecolor=ARCH_COLORS[DISPLAY[a]], label=DISPLAY[a])
              for a in ["proposed", "al_like", "azurelily"]]
    fig.legend(handles=arch_h, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.93), frameon=False, columnspacing=0.8)

    out = FIG_DIR / "bert_seqlen_efficiency_varfmax.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: bert_seqlen_latency
#   (a) DIMM latency proportion (%), (b) Speedup breakdown over Azure-Lily
# ══════════════════════════════════════════════════════════════════════════
def plot_latency():
    """Left: stacked bar latency (DIMM + non-DIMM). Right: speedup with ideal vs reported."""
    import matplotlib.lines as mlines

    def _getv(arch, S, key):
        return data_var[(arch, S)][key]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    fig.subplots_adjust(wspace=0.30, top=0.82, bottom=0.18)

    # ── Panel (a): Stacked bar — DIMM + non-DIMM latency ──
    # Same style as energy 3-panel left: hatching per arch
    arch_list = ["proposed", "al_like", "azurelily"]
    arch_hatch = {"proposed": None, "al_like": "...", "azurelily": "///"}
    n_groups = len(SEQ_LENS)
    x = np.arange(n_groups)
    bar_w = 0.25

    DIMM_COLOR = '#5ec28b'
    NONDIMM_COLOR = '#7b8bff'

    for ai, arch in enumerate(arch_list):
        offset = (ai - 1) * bar_w
        hatch = arch_hatch[arch]
        for i in range(n_groups):
            total_lat = _getv(arch, SEQ_LENS[i], "latency_dimm_ns") + _getv(arch, SEQ_LENS[i], "latency_non_dimm_ns")
            dimm_pct = _getv(arch, SEQ_LENS[i], "latency_dimm_ns") / total_lat * 100
            nondimm_pct = 100 - dimm_pct
            ax1.bar(x[i] + offset, dimm_pct, bar_w * 0.9, color=DIMM_COLOR,
                    edgecolor='white', linewidth=0.5, hatch=hatch, zorder=3)
            ax1.bar(x[i] + offset, nondimm_pct, bar_w * 0.9, bottom=dimm_pct,
                    color=NONDIMM_COLOR, edgecolor='white', linewidth=0.5,
                    hatch=hatch, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(N) for N in SEQ_LENS])
    ax1.set_xlabel('Sequence Length (N)', fontsize=11)
    ax1.set_ylabel('Latency Breakdown (%)', fontsize=13, fontweight='normal')
    ax1.tick_params(axis='both', labelsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.08, axis='y')

    # Legend: arch hatching + DIMM/non-DIMM colors
    import matplotlib.patches as mpatches
    a_handles = [
        mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, label='Proposed-1'),
        mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, hatch='...', label='Proposed-2'),
        mpatches.Patch(facecolor='#999', edgecolor='black', linewidth=0.5, hatch='///', label='Azure-Lily'),
    ]
    c_handles = [
        mpatches.Patch(facecolor=DIMM_COLOR, edgecolor='white', label='DIMM'),
        mpatches.Patch(facecolor=NONDIMM_COLOR, edgecolor='white', label='Non-DIMM'),
    ]
    bbox_left = (ax1.get_position().x0 + ax1.get_position().width / 2, 1.0)
    fig.legend(handles=a_handles + c_handles, loc='upper center', fontsize=8,
               frameon=True, framealpha=0.9, bbox_to_anchor=bbox_left,
               ncol=3, columnspacing=0.5, handletextpad=0.3)

    # ── Panel (b): DIMM + non-DIMM speedup (reported + ideal) for P1 and P2 ──
    COLOR_P1 = '#5ec28b'
    COLOR_P2 = '#7b8bff'

    for arch, color in [("proposed", COLOR_P1), ("al_like", COLOR_P2)]:
        # Reported Fmax only
        dimm_sp = [_getv("azurelily", S, "latency_dimm_ns") / _getv(arch, S, "latency_dimm_ns")
                   for S in SEQ_LENS]
        nondimm_sp = [_getv("azurelily", S, "latency_non_dimm_ns") / _getv(arch, S, "latency_non_dimm_ns")
                      for S in SEQ_LENS]

        # DIMM speedup
        ax2.plot(SEQ_LENS, dimm_sp, color=color, linewidth=2, linestyle='-',
                 marker='s', markersize=4, markeredgecolor='white', markeredgewidth=0.8, zorder=4)

        # Non-DIMM speedup
        ax2.plot(SEQ_LENS, nondimm_sp, color=color, linewidth=1.5, linestyle='-',
                 marker='^', markersize=3, markeredgecolor='white', markeredgewidth=0.8,
                 alpha=0.7, zorder=4)

    _setup_xaxis(ax2)
    ax2.set_xlabel('Sequence Length (N)', fontsize=11)
    ax2.set_ylabel('Speedup', fontsize=13, fontweight='normal', labelpad=6)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, alpha=0.1)

    # Legend: arch colors + DIMM/nonDIMM markers
    arch_h = [
        Patch(facecolor=COLOR_P1, label='Proposed-1'),
        Patch(facecolor=COLOR_P2, label='Proposed-2'),
    ]
    metric_h = [
        mlines.Line2D([], [], color='black', linewidth=2, marker='s', markersize=3, linestyle='-', label='DIMM'),
        mlines.Line2D([], [], color='black', linewidth=1.5, marker='^', markersize=3, linestyle='-', alpha=0.7, label='Non-DIMM'),
    ]
    bbox_right = (ax2.get_position().x0 + ax2.get_position().width / 2, 0.95)
    fig.legend(handles=arch_h + metric_h, loc='upper center', fontsize=8,
               frameon=True, framealpha=0.9, bbox_to_anchor=(ax2.get_position().x0 + ax2.get_position().width / 2, 1.0),
               ncol=2, columnspacing=0.5, handletextpad=0.3)

    out = FIG_DIR / "bert_seqlen_latency.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: bert_seqlen_resilience
#   (a) Advantage ratio (Proposed / Azure-Lily) for Tput/mm² and Inf/J
#   (b) Per-step throughput retention (Tput(S) / Tput(S/2))
# ══════════════════════════════════════════════════════════════════════════
def plot_resilience():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.subplots_adjust(wspace=0.30, top=0.82, bottom=0.18)

    # ── Panel (a): Advantage ratio vs S ──
    # Proposed/Azure-Lily at each S — shows how long the advantage lasts
    for arch, key, color, ls, mk, label in [
        ("proposed", "throughput_per_mm2", ARCH_COLORS["Proposed-1"], "-", "o", "P1 Tput/mm\u00b2"),
        ("proposed", "throughput_per_j",   ARCH_COLORS["Proposed-1"], "--", "s", "P1 Inf/J"),
        ("al_like",  "throughput_per_mm2", ARCH_COLORS["Proposed-2"], "-", "^", "P2 Tput/mm\u00b2"),
        ("al_like",  "throughput_per_j",   ARCH_COLORS["Proposed-2"], "--", "D", "P2 Inf/J"),
    ]:
        ratios = [_get(arch, S, key) / _get("azurelily", S, key) for S in SEQ_LENS]
        ax1.plot(SEQ_LENS, ratios, color=color, linewidth=1.5,
                 linestyle=ls, marker=mk, markersize=4,
                 markeredgecolor='white', markeredgewidth=0.5, label=label)

    ax1.axhline(y=1.0, color=BASELINE_COLOR, linewidth=1.2,
                linestyle=BASELINE_LS, alpha=0.7)
    ax1.fill_between(SEQ_LENS, 1.0, max(
        max(_get("proposed", S, "throughput_per_mm2") / _get("azurelily", S, "throughput_per_mm2") for S in SEQ_LENS),
        max(_get("proposed", S, "throughput_per_j") / _get("azurelily", S, "throughput_per_j") for S in SEQ_LENS)
    ) * 1.1, alpha=0.04, color='green')
    ax1.text(SEQ_LENS[1], 1.05, 'Proposed wins', fontsize=6, color='green', alpha=0.5, fontstyle='italic')

    _setup_xaxis(ax1)
    ax1.set_ylabel("Advantage over Azure-Lily (\u00d7)")
    ax1.set_title("(a) Efficiency Advantage vs Seq Length", fontsize=8)
    ax1.legend(fontsize=5.5, loc='upper right', framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.1)

    # ── Panel (b): Per-step retention Tput(S) / Tput(S_prev) ──
    # Flatter = more resilient to O(S²)
    S_pairs = list(zip(SEQ_LENS[:-1], SEQ_LENS[1:]))
    x_labels = [f"{a}\u2192{b}" for a, b in S_pairs]
    x_pos = range(len(S_pairs))

    for arch in ARCHS:
        dn = DISPLAY[arch]
        retention_mm2 = [_get(arch, b, "throughput_per_mm2") / _get(arch, a, "throughput_per_mm2")
                         for a, b in S_pairs]
        ax2.plot(list(x_pos), retention_mm2, color=ARCH_COLORS[dn], linewidth=1.5,
                 linestyle=ARCH_LINESTYLES[dn], marker=ARCH_MARKERS[dn],
                 markersize=4, markeredgecolor='white', markeredgewidth=0.5,
                 label=dn)

    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(x_labels, fontsize=6, rotation=30, ha='right')
    ax2.set_xlabel("Sequence Length Step")
    ax2.set_ylabel("Throughput/mm\u00b2 Retention\n(ratio to previous S)")
    ax2.set_title("(b) Per-Step Throughput Retention", fontsize=8)
    ax2.set_ylim(0, 0.5)
    ax2.axhline(y=0.25, color='#888', linewidth=0.8, linestyle=':', alpha=0.5)
    ax2.text(len(S_pairs)-1, 0.26, 'O(S\u00b2) theoretical: 0.25\u00d7',
             ha='right', fontsize=6, color='#888', fontstyle='italic')
    ax2.legend(fontsize=6, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.1)

    out = FIG_DIR / "bert_seqlen_resilience.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading: {CSV_PATH_FIXED} (analysis/efficiency/resilience)")
    print(f"Reading: {CSV_PATH_VAR} (latency)")
    print(f"Seq lens: {SEQ_LENS}\n")
    plot_analysis()
    plot_efficiency()
    plot_efficiency_varfmax()
    plot_latency()
    plot_resilience()
    print("\nDone.")
