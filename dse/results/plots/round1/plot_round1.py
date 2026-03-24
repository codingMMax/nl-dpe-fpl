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
CSV_PATH = RESULTS_DIR / "round1_results.csv"

import sys
_nl_dpe_dir = str(Path(__file__).resolve().parents[4] / "nl_dpe")
sys.path.insert(0, _nl_dpe_dir)
from area_power import dpe_specs

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

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

    top3 = configs_ranked[:3]

    fig, ax = plt.subplots(figsize=(6, max(4, n * 0.4)))
    y_pos = np.arange(n)

    # Reversed so #1 at top
    colors = [COLOR_TOP if configs_ranked[n - 1 - i] in top3
              else COLOR_EDAP for i in range(n)]
    bars = ax.barh(y_pos, gm_vals[::-1], 0.6, color=colors, alpha=0.85)

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
        is_top3 = cfg in top3
        weight = "bold" if is_top3 else "normal"
        marker = "  *" if is_top3 else ""

        label = f"#{rank}  {cfg}{marker}"
        ax.text(-0.02, i, label, ha="right", va="center", fontsize=9,
                fontweight=weight, transform=ax.get_yaxis_transform())

        annot_x = gm_vals[n - 1 - i] + 0.02

        # Annotate worst peripheral overhead (smallest configs)
        if cfg_area[cfg] == min_area:
            ax.text(annot_x, i,
                    f"FPGA integration overhead: {cfg_overhead[cfg]:.0f}% block area",
                    ha="left", va="center", fontsize=7.5,
                    color=COLOR_AL)

        # Annotate highest power (largest config) + AL comparison if applicable
        elif cfg_power[cfg] == max_power:
            annot = f"DPE power: {cfg_power[cfg]:.0f} mW"
            if cfg in ("1024x256", "512x256"):
                ratio = area_ratio_to_al(cfg)
                annot += f",  {ratio*100:.0f}% of Azure-Lily DPE area"
            ax.text(annot_x, i, annot,
                    ha="left", va="center", fontsize=7.5,
                    color=COLOR_AL)

        # AL comparison for remaining AL-like configs
        elif cfg in ("1024x256", "512x256"):
            ratio = area_ratio_to_al(cfg)
            ax.text(annot_x, i, f"{ratio*100:.0f}% of Azure-Lily DPE area",
                    ha="left", va="center", fontsize=7.5,
                    fontweight="normal", color=COLOR_AL)

    ax.set_yticks([])
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Normalized Geomean EDAP Score (best = 1.0)")
    ax.set_title("DPE Config Ranking by Energy-Delay-Area Product (EDAP)",
                 fontweight="bold")
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


def main():
    rows, ranked, scores, idx = load_and_rank()

    print("Round 1 EDAP Ranking:")
    for rank, (cfg, sc) in enumerate(ranked, 1):
        print(f"  #{rank} {cfg}: GM_EDAP={sc['gm']:.4f}")

    plot_ranking(ranked)
    plot_heatmap(rows, scores, idx)


if __name__ == "__main__":
    main()
