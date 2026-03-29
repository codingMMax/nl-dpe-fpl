#!/usr/bin/env python3
"""Generate Round 1 DSE plots for paper.

Plot 1: Config ranking bar chart (EDAP = Energy × Delay × Area, lower is better)
Plot 2: Config × Workload heatmap (normalized EDAP, annotated with V)

Outputs:
    round1_ranking.pdf
    round1_heatmap.pdf
"""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
DATA_DIR = RESULTS_DIR.parent / "data"
OUTPUT_DIR = RESULTS_DIR.parent / "figures"
CSV_PATH = DATA_DIR / "round1_results.csv"

import sys
# paper/scripts/ -> paper/ -> nl-dpe-fpl/ -> nl_dpe/
_nl_dpe_dir = str(Path(__file__).resolve().parent.parent.parent / "nl_dpe")
sys.path.insert(0, _nl_dpe_dir)
from area_power import dpe_specs

from style_constants import apply_style
apply_style()

COLOR_EDAP = "#2563EB"
COLOR_TOP = "#059669"
COLOR_AL = "#DC2626"

AL_AREA_MWTA = 2_320_000

workloads_ordered = ["fc_64_64", "fc_128_128", "fc_512_128", "fc_256_512",
                     "fc_512_512", "fc_2048_256"]
workload_labels = {
    "fc_64_64": "FC 64\u00d764", "fc_128_128": "FC 128\u00d7128",
    "fc_512_128": "FC 512\u00d7128", "fc_256_512": "FC 256\u00d7512",
    "fc_512_512": "FC 512\u00d7512", "fc_2048_256": "FC 2048\u00d7256",
}


def area_ratio_to_al(cfg_str):
    R, C = map(int, cfg_str.split("x"))
    return dpe_specs(R, C)["area_tag_mwta"] / AL_AREA_MWTA


def load_and_rank():
    """Load CSV, compute EDAP, return ranked configs."""
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["energy_pj"] = float(r["energy_pj"])
        r["latency_ns"] = float(r["latency_ns"])
        r["fpga_area_mm2"] = float(r["fpga_area_mm2"])
        r["fmax_mhz"] = float(r["fmax_mhz"])

    workloads = sorted(set(r["workload"] for r in rows))
    configs = sorted(set(r["config"] for r in rows))
    idx = {(r["config"], r["workload"]): r for r in rows}

    # EDAP per (config, workload)
    edap = {}
    best_edap = {}
    for wl in workloads:
        for c in configs:
            if (c, wl) not in idx:
                continue
            r = idx[(c, wl)]
            e, d, a = r["energy_pj"], r["latency_ns"], r["fpga_area_mm2"]
            edap[(c, wl)] = e * d * a if (e > 0 and d > 0 and a > 0) else float("inf")
        wl_vals = [edap[(c, wl)] for c in configs
                   if (c, wl) in edap and edap[(c, wl)] < float("inf")]
        best_edap[wl] = min(wl_vals) if wl_vals else 1.0

    # Normalized geomean
    scores = {}
    for cfg in configs:
        norm_vals = []
        per_wl = {}
        for wl in workloads:
            if (cfg, wl) not in edap:
                continue
            e = edap[(cfg, wl)]
            n = best_edap[wl] / e if e > 0 and e < float("inf") else 0
            norm_vals.append(n)
            per_wl[wl] = {"norm_edap": n, "edap": e}
        if norm_vals and all(v > 0 for v in norm_vals):
            gm = math.exp(sum(math.log(v) for v in norm_vals) / len(norm_vals))
            scores[cfg] = {"gm": gm, "per_wl": per_wl}

    ranked = sorted(scores.items(), key=lambda x: -x[1]["gm"])
    return rows, ranked, scores, idx


def plot_ranking(ranked, top_k=5):
    """Bar chart: EDAP geomean ranking."""
    configs_ranked = [cfg for cfg, _ in ranked]
    gm_vals = [sc["gm"] for _, sc in ranked]
    n = len(configs_ranked)
    top_configs = configs_ranked[:top_k]

    top5 = configs_ranked[:5]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    y_pos = np.arange(n)

    # Reversed so #1 at top
    colors = [COLOR_TOP if configs_ranked[n - 1 - i] in top5
              else COLOR_EDAP for i in range(n)]
    bars = ax.barh(y_pos, gm_vals[::-1], 0.55, color=colors, alpha=0.85)

    # Compute area overhead and power for all configs
    cfg_overhead = {}
    cfg_power = {}
    cfg_area = {}
    for cfg in configs_ranked:
        R, C = map(int, cfg.split("x"))
        s = dpe_specs(R, C)
        cfg_overhead[cfg] = s['routing_um2'] / s['tile_total_um2'] * 100  # routing as % of total tile
        cfg_power[cfg] = s['power_total_mw']
        cfg_area[cfg] = s['tile_total_um2']
    max_overhead = max(cfg_overhead.values())
    max_power = max(cfg_power.values())
    min_area = min(cfg_area.values())

    # Labels + annotations
    for i in range(n):
        cfg = configs_ranked[n - 1 - i]
        rank = n - i
        is_top5 = cfg in top5
        weight = "bold" if is_top5 else "normal"
        marker = " *" if is_top5 else ""
        label = f"#{rank}  {cfg}{marker}"
        ax.text(-0.02, i, label, ha="right", va="center", fontsize=7,
                fontweight=weight, transform=ax.get_yaxis_transform())

        annot_x = gm_vals[n - 1 - i] + 0.02

        # Annotate worst peripheral overhead (smallest configs)
        if cfg_area[cfg] == min_area:
            ax.text(annot_x, i,
                    f"{cfg_overhead[cfg]:.0f}% DPE logic area accounts for block peripheral",
                    ha="left", va="center", fontsize=6,
                    color=COLOR_AL)

        # Annotate highest power (largest config) + AL comparison if applicable
        # elif cfg_power[cfg] == max_power:
        #     # annot = f"DPE power: {cfg_power[cfg]:.0f} mW"
        #     annot = ""
        #     if cfg in ("1024x256", "512x256"):
        #         ratio = area_ratio_to_al(cfg)
        #         annot += f"{ratio*100:.0f}% of Azure-Lily DPE area"
        #     ax.text(annot_x, i, annot,
        #             ha="left", va="center", fontsize=6,
        #             color=COLOR_AL)

        # AL comparison for remaining AL-like configs
        elif cfg in ("1024x256", "512x256"):
            ratio = area_ratio_to_al(cfg)
            ax.text(annot_x, i, f"AL-Like:{ratio*100:.0f}% of Azure-Lily DPE area",
                    ha="left", va="center", fontsize=6,
                    fontweight="normal", color=COLOR_AL)

    ax.set_yticks([])
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Normalized Geomean EDAP Score (best = 1.0)")
    # ax.set_title("",
    #              fontweight="bold")
    ax.axvline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

    out = RESULTS_DIR / "round1_ranking.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_heatmap(rows, scores, idx):
    """Config × Workload heatmap of normalized EDAP."""
    configs_ranked = [cfg for cfg, _ in sorted(scores.items(), key=lambda x: -x[1]["gm"])]
    workloads = [wl for wl in workloads_ordered if wl in set(r["workload"] for r in rows)]

    n_cfg = len(configs_ranked)
    n_wl = len(workloads)

    mat = np.full((n_cfg, n_wl), np.nan)
    v_mat = np.full((n_cfg, n_wl), 0, dtype=int)

    for i, cfg in enumerate(configs_ranked):
        for j, wl in enumerate(workloads):
            if cfg in scores and wl in scores[cfg]["per_wl"]:
                mat[i, j] = scores[cfg]["per_wl"][wl]["norm_edap"]
            if (cfg, wl) in idx:
                v_mat[i, j] = int(idx[(cfg, wl)].get("V", 0))

    fig, ax = plt.subplots(figsize=(8, max(4, n_cfg * 0.45)))
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.4, vmax=1.0)
    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

    # Cell annotations
    for i in range(n_cfg):
        for j in range(n_wl):
            val = mat[i, j]
            v = v_mat[i, j]
            if np.isnan(val):
                continue
            tc = "white" if val < 0.2 else "black"
            acam = "\u2713" if v == 1 else ""
            ax.text(j, i, f"{val:.2f}\n{acam}", ha="center", va="center",
                    fontsize=7.5, color=tc, fontweight="bold" if v == 1 else "normal")

    ax.set_xticks(range(n_wl))
    ax.set_xticklabels([workload_labels.get(wl, wl) for wl in workloads],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_cfg))
    ax.set_yticklabels([f"#{i+1} {cfg}" for i, cfg in enumerate(configs_ranked)],
                       fontsize=8)
    ax.set_title("Round 1: Normalized EDAP per Workload\n"
                 "(\u2713 = ACAM-eligible, V=1)",
                 fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Normalized EDAP (best = 1.0)", fontsize=9)

    out = RESULTS_DIR / "round1_heatmap.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def load_combined_and_rank():
    """Load combined FC+attention CSV, compute separate rankings."""
    combined_csv = RESULTS_DIR.parent.parent / "round1_results_combined.csv"
    if not combined_csv.exists():
        return None, None

    with open(combined_csv) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        for k in ['energy_pj', 'latency_ns', 'fpga_area_mm2', 'fmax_mhz']:
            if k in r:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    r[k] = 0.0

    def compute_ranking_for(wl_filter):
        filtered = [r for r in rows if wl_filter(r['workload'])]
        workloads = sorted(set(r['workload'] for r in filtered))
        configs = sorted(set(r['config'] for r in filtered))

        edap = {}
        best_edap = {}
        for wl in workloads:
            for cfg in configs:
                matches = [r for r in filtered if r['config'] == cfg and r['workload'] == wl]
                if matches:
                    r = matches[0]
                    e, d, a = r['energy_pj'], r['latency_ns'], r['fpga_area_mm2']
                    edap[(cfg, wl)] = e * d * a if (e > 0 and d > 0 and a > 0) else float('inf')
            wl_vals = [edap[(c, wl)] for c in configs
                       if (c, wl) in edap and edap[(c, wl)] < float('inf')]
            best_edap[wl] = min(wl_vals) if wl_vals else 1.0

        rankings = []
        for cfg in configs:
            norm_vals = []
            for wl in workloads:
                if (cfg, wl) in edap and edap[(cfg, wl)] < float('inf') and best_edap[wl] > 0:
                    norm_vals.append(best_edap[wl] / edap[(cfg, wl)])
            if norm_vals and all(v > 0 for v in norm_vals):
                gm = math.exp(sum(math.log(v) for v in norm_vals) / len(norm_vals))
                rankings.append((cfg, gm))

        rankings.sort(key=lambda x: -x[1])
        return rankings

    fc_ranking = compute_ranking_for(lambda w: w.startswith('fc'))
    attn_ranking = compute_ranking_for(lambda w: w.startswith('attention'))
    return fc_ranking, attn_ranking


def plot_fc_vs_attention_ranking(fc_ranking, attn_ranking):
    """Two subplots: FC ranking and Attention ranking, with top-3 annotated."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.30, left=0.12, right=0.97)

    AL_MATCHED = {"512x128", "1024x256", "512x256"}

    for ax, ranking, title in [
        (ax1, fc_ranking, "(a) FC Workloads (6 workloads)"),
        (ax2, attn_ranking, "(b) Attention Workloads (3 workloads)"),
    ]:
        n = len(ranking)
        cfgs = [r[0] for r in ranking][::-1]   # reversed: #1 at top
        vals = [r[1] for r in ranking][::-1]
        top3 = [r[0] for r in ranking[:3]]

        y_pos = np.arange(n)

        colors = []
        for cfg in cfgs:
            if cfg in top3:
                colors.append(COLOR_TOP)
            else:
                colors.append(COLOR_EDAP)

        bars = ax.barh(y_pos, vals, 0.6, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.5)

        # Y-axis: config labels as tick labels
        ylabels = []
        for i, cfg in enumerate(cfgs):
            rank = n - i
            ylabels.append(f"#{rank}  {cfg}")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ylabels, fontsize=8.5)
        for i, label in enumerate(ax.get_yticklabels()):
            cfg = cfgs[i]
            if cfg in top3:
                label.set_fontweight("bold")
                label.set_color(COLOR_TOP)

        # Area ratio annotation for top-3 and AL-matched
        for i, (cfg, val) in enumerate(zip(cfgs, vals)):
            is_top3 = cfg in top3
            is_al = cfg in AL_MATCHED
            if is_top3 or is_al:
                ratio = area_ratio_to_al(cfg)
                annot = f"{ratio*100:.0f}% area of Azure-Lily"
                ax.text(val + 0.012, i, annot, ha="left", va="center",
                        fontsize=7, color=COLOR_AL, fontweight="bold")

        ax.set_xlim(0, max(vals) * 1.45)
        ax.set_xlabel("EDAP Geomean Score", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.1, axis="x")

    out = RESULTS_DIR / "round1_ranking_fc_vs_attention.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def main():
    rows, ranked, scores, idx = load_and_rank()

    print("Round 1 EDAP Ranking (FC only):")
    for rank, (cfg, sc) in enumerate(ranked, 1):
        print(f"  #{rank} {cfg}: GM_EDAP={sc['gm']:.4f}")

    plot_ranking(ranked)
    plot_heatmap(rows, scores, idx)

    # Generate FC vs Attention ranking plot if combined CSV exists
    fc_ranking, attn_ranking = load_combined_and_rank()
    if fc_ranking and attn_ranking:
        print("\nFC vs Attention Rankings:")
        print("  FC top-3:", [c for c, _ in fc_ranking[:3]])
        print("  Attn top-3:", [c for c, _ in attn_ranking[:3]])
        plot_fc_vs_attention_ranking(fc_ranking, attn_ranking)


if __name__ == "__main__":
    main()
