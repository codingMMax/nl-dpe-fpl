#!/usr/bin/env python3
"""BERT-Tiny Analysis: (a) DPE energy contribution vs seq_len, (b) Energy breakdown.

Input: IMC simulator (azurelily/)
Output: paper/figures/benchmarks/bert_analysis.pdf
"""
import math
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
sys.path.insert(0, str(ROOT_DIR / "azurelily"))
sys.path.insert(0, str(ROOT_DIR / "azurelily" / "IMC"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_constants import (apply_style_sc, ARCH_COLORS, ARCH_MARKERS,
                              ARCH_LINESTYLES, BASELINE_COLOR, BASELINE_LS,
                              BASELINE_ALPHA, ANNOT_FONTSIZE_SC, ANNOT_FONTWEIGHT_SC,
                              BREAKDOWN_COLORS)
apply_style_sc()

from imc_core.config import Config
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from peripherals.memory import MemoryModel
from scheduler_stats.stats import Stats
from scheduler_stats.scheduler import Scheduler
from models.attention import attention_model

OUT_PATH = SCRIPT_DIR.parent / "figures" / "benchmarks" / "bert_analysis.pdf"


def _functional_dpes(rows, cols):
    """Non-DIMM DPEs: Q/K/V/O + FFN1 + FFN2, × 2 blocks."""
    D_MODEL, D_FF = 128, 512
    per_block = (4 * math.ceil(D_MODEL/rows) * math.ceil(D_MODEL/cols)
                 + math.ceil(D_MODEL/rows) * math.ceil(D_FF/cols)
                 + math.ceil(D_FF/rows) * math.ceil(D_MODEL/cols))
    return per_block * 2


def _make_sim(cfile, R, C, fmax, total_dsp=None, total_clb=None, total_mem=None,
              total_dimm_dpes=None):
    cfg = Config(str(ROOT_DIR / "azurelily" / "IMC" / "configs" / f"{cfile}.json"))
    cfg.rows = R; cfg.cols = C; cfg.freq = fmax
    if total_dsp is not None: cfg.total_dsp = total_dsp
    if total_clb is not None: cfg.total_clb = total_clb
    if total_mem is not None: cfg.total_mem = total_mem
    if total_dimm_dpes is not None: cfg.total_dimm_dpes = total_dimm_dpes
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats); fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    return cfg, stats, sched


def run_attention(cfile, R, C, N, d_head=64):
    """Run single attention head, return (total_energy, dpe_energy)."""
    cfg = Config(str(ROOT_DIR / "azurelily" / "IMC" / "configs" / f"{cfile}.json"))
    cfg.rows = R; cfg.cols = C; cfg.freq = 150
    stats = Stats(); mem = MemoryModel(cfg, stats)
    imc = IMCCore(cfg, mem, stats)
    fpga = FPGAFabric(cfg, mem, stats, imc_core=imc)
    sched = Scheduler(cfg, stats, imc, fpga)
    layers, _ = attention_model(1, 1, N, d_head, False, False)
    for layer in layers:
        sched.run_layer(layer)
    e_bd = dict(stats.energy_breakdown)
    total_e = sum(e_bd.values())
    e_dpe = sum(v for k, v in e_bd.items() if "imc" in k)
    return total_e, e_dpe


def run_bert_total(cfile, R, C, fmax, N):
    """Run full BERT-Tiny, return total energy."""
    _, stats, sched = _make_sim(cfile, R, C, fmax)
    from models.bert_tiny import bert_tiny_model
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
    return sum(stats.energy_breakdown.values())


# VTR ground truth: AVAILABLE resources per arch
VTR_AVAILABLE = {
    "nl_dpe_1024x128": {"DSPs": 4,   "CLBs": 453,  "BRAMs": 172, "DPEs": 274},
    "nl_dpe_1024x256": {"DSPs": 4,   "CLBs": 441,  "BRAMs": 172, "DPEs": 78},
    "azure_lily":      {"DSPs": 326, "CLBs": 188,  "BRAMs": 16,  "DPEs": 18},
}


def run_bert_breakdown(cfile, R, C, fmax, N):
    """Run BERT-Tiny tracking energy per layer category.

    Categories: Proj (Q/K/V/O), FFN, DIMM QK^T, Softmax, DIMM S×V, Other (LN+Res+Embed)
    """
    from models.bert_tiny import bert_tiny_model
    md, _ = bert_tiny_model(1, 1, N, 128, False, False)

    # Look up available resources
    key = f"{cfile}_{R}x{C}" if cfile == "nl_dpe" else cfile
    avail = VTR_AVAILABLE.get(key, {})

    groups = defaultdict(float)

    def run_group(layers, group_name, sched_obj):
        """Run layers and accumulate energy into group_name."""
        e_before = sum(sched_obj.stats.energy_breakdown.values())
        for layer in layers:
            sched_obj.run_layer(layer)
        e_after = sum(sched_obj.stats.energy_breakdown.values())
        groups[group_name] += (e_after - e_before)

    # Single simulator instance — accumulate energy deltas per group
    dpe_used = avail.get("DPEs", 0)
    dimm_dpes = max(0, dpe_used - _functional_dpes(R, C))
    _, stats, sched = _make_sim(cfile, R, C, fmax,
                                 total_dsp=avail.get("DSPs"),
                                 total_clb=avail.get("CLBs"),
                                 total_mem=avail.get("BRAMs"),
                                 total_dimm_dpes=dimm_dpes)

    # Embedding + LayerNorm → Other
    run_group(md["embedding"], "Other\n(LN+Res)", sched)

    for block in md["blocks"]:
        # Q/K/V projections → Proj
        run_group(block["qkv_proj"], "Proj\n(Q/K/V/O)", sched)

        # Per-head attention: mac_qk, softmax_exp, softmax_norm, mac_sv
        for hi in range(block["num_heads"]):
            attn_layers = block["head_attention"]
            # attn_layers order: [mac_qk, softmax_exp, softmax_norm, mac_sv]
            run_group([attn_layers[0]], "DIMM\nQK\u1d40", sched)
            run_group([attn_layers[1], attn_layers[2]], "Softmax", sched)
            run_group([attn_layers[3]], "DIMM\nS\u00d7V", sched)

        # O projection → Proj, residual+LN → Other
        # post_attn: [O_proj, attn_residual, attn_ln]
        run_group([block["post_attn"][0]], "Proj\n(Q/K/V/O)", sched)
        run_group(block["post_attn"][1:], "Other\n(LN+Res)", sched)

        # FFN: [ffn1, ffn2, ffn_residual, ffn_ln]
        run_group(block["ffn"][:2], "FFN", sched)
        run_group(block["ffn"][2:], "Other\n(LN+Res)", sched)

    return dict(groups)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    fig.subplots_adjust(wspace=0.15, top=0.88)

    # Panel labels
    ax1.text(-0.12, 1.08, '(a)', transform=ax1.transAxes, fontsize=10,
             fontweight='bold', va='top')
    ax2.text(-0.10, 1.08, '(b)', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', va='top')

    # ── Left: DPE energy contribution vs seq_len ──
    seq_lens = [1024, 2048, 4096, 6144, 8192]
    attn_configs = [
        ("Proposed", "nl_dpe", 1024, 128),
        ("Azure-Lily", "azure_lily", 512, 128),
    ]
    attn_colors = {"Proposed": ARCH_COLORS["Proposed"],
                   "Azure-Lily": ARCH_COLORS["Azure-Lily"]}
    attn_markers = {"Proposed": "o", "Azure-Lily": "D"}
    attn_styles = {"Proposed": "-", "Azure-Lily": "--"}

    for cname, cfile, R, C in attn_configs:
        dpe_pcts = []
        for N in seq_lens:
            total_e, e_dpe = run_attention(cfile, R, C, N)
            dpe_pcts.append(e_dpe / total_e * 100 if total_e > 0 else 0)
        ax1.plot(seq_lens, dpe_pcts, color=attn_colors[cname], linewidth=2,
                 linestyle=attn_styles[cname], marker=attn_markers[cname],
                 markersize=4, markeredgecolor="white", markeredgewidth=0.8,
                 label=cname)

    ax1.axhline(y=50, color="#888", linewidth=0.8, linestyle=":", alpha=0.5)
    ax1.set_xlabel("Sequence Length (N)")
    ax1.set_ylabel("DPE Energy Contribution (%)")
    ax1.set_xticks(seq_lens)
    ax1.set_xticklabels([str(n) for n in seq_lens])
    ax1.set_xlim(seq_lens[0] * 0.9, seq_lens[-1] * 1.05)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.1)

    # ── Right: Stacked % breakdown — shows DIMM dominance ──
    nl_groups = run_bert_breakdown("nl_dpe", 1024, 128, 135.7, 128)
    al_groups = run_bert_breakdown("azure_lily", 512, 128, 45.3, 128)

    # Merge into 3 high-level categories for clarity:
    #   DIMM (QK^T + S×V) — the O(N²) bottleneck
    #   Proj+FFN — where NL-DPE wins 43×
    #   Other (Softmax + LN + Residual)
    cat_keys = {
        "DIMM\n(QK\u1d40 + S\u00d7V)": ["DIMM\nQK\u1d40", "DIMM\nS\u00d7V"],
        "Proj + FFN": ["Proj\n(Q/K/V/O)", "FFN"],
        "Other": ["Softmax", "Other\n(LN+Res)"],
    }
    layer_keys = ["Proj\n(Q/K/V/O)", "FFN", "DIMM\nQK\u1d40", "Softmax",
                   "DIMM\nS\u00d7V", "Other\n(LN+Res)"]

    nl_merged = {}
    al_merged = {}
    for group, keys in cat_keys.items():
        nl_merged[group] = sum(nl_groups.get(k, 0) for k in keys)
        al_merged[group] = sum(al_groups.get(k, 0) for k in keys)

    nl_total = sum(nl_merged.values())
    al_total = sum(al_merged.values())

    # Print detailed table
    print(f"\n{'Category':<25} {'NL-DPE (M pJ)':>14} {'%':>6} {'AL (M pJ)':>14} {'%':>6}")
    print("-" * 70)
    for cat, nv, av in zip(cat_keys.keys(),
                            [nl_merged[k] for k in cat_keys],
                            [al_merged[k] for k in cat_keys]):
        cat_clean = cat.replace('\n', ' ')
        print(f"{cat_clean:<25} {nv/1e6:>14.3f} {nv/nl_total*100:>5.1f}% {av/1e6:>14.3f} {av/al_total*100:>5.1f}%")
    print(f"{'TOTAL':<25} {nl_total/1e6:>14.3f} {'':>6} {al_total/1e6:>14.3f}")

    # Stacked horizontal bars normalized to NL-DPE total energy
    # NL-DPE = 100, Azure-Lily segments scaled by real energy ratio
    categories_merged = list(cat_keys.keys())
    bar_colors = [BREAKDOWN_COLORS["DIMM"], BREAKDOWN_COLORS["Proj_FFN"],
                  BREAKDOWN_COLORS["Other"]]
    archs = ["Azure-Lily ", "Proposed "]

    # Normalize all segments to NL-DPE total = 100
    arch_vals = [
        [al_merged[k] / nl_total * 100 for k in categories_merged],  # AL wider
        [nl_merged[k] / nl_total * 100 for k in categories_merged],  # NL = 100
    ]

    y_pos = np.arange(len(archs))
    bar_h = 0.45

    # Each architecture's own percentage breakdown (for annotation)
    arch_own_pcts = [
        [al_merged[k] / al_total * 100 for k in categories_merged],
        [nl_merged[k] / nl_total * 100 for k in categories_merged],
    ]

    for idx in range(len(archs)):
        left = 0
        for j, (cat, val) in enumerate(zip(categories_merged, arch_vals[idx])):
            ax2.barh(y_pos[idx], val, bar_h, left=left, color=bar_colors[j],
                     edgecolor="white", linewidth=0.8,
                     label=cat if idx == 0 else None, zorder=3)
            own_pct = arch_own_pcts[idx][j]
            if val > 8:
                ax2.text(left + val / 2, y_pos[idx], f"{own_pct:.0f}%",
                         ha='center', va='center', fontsize=ANNOT_FONTSIZE_SC,
                         fontweight='bold', color='white', zorder=5)
            left += val

    ax2.set_yticks([])
    # Place arch labels on top of each bar
    for idx, arch in enumerate(archs):
        total_width = sum(arch_vals[idx])
        ax2.text(total_width / 2, y_pos[idx] + bar_h / 2 + 0.06, arch.strip(),
                 ha='center', va='bottom', fontsize=8, fontweight='bold',
                 color='black', zorder=6)
    ax2.set_xlabel("Energy normalized to proposed")
    ax2.set_xlim(0, 190)
    ax2.set_ylim(-0.5, 2.2)
    ax2.legend(fontsize=7, loc="upper left", frameon=False, ncol=3,
               columnspacing=0.8, handletextpad=0.4)
    ax2.grid(True, alpha=0.08, axis="x", zorder=0)

    fig.savefig(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
