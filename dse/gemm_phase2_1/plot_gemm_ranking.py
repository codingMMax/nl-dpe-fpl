#!/usr/bin/env python3
"""Phase 2.1 GEMM DSE ranking plot — same style as round1_ranking.pdf.

Reads dse/gemm_phase2_1/results/gemm_dse_smoke.csv and produces:
  - gemm_ranking.pdf    : horizontal bar chart, configs ranked by
                          SPEC-style normalized geomean EDAP across the
                          4 GEMM workloads (best = 1.0).
  - gemm_heatmap_edap.pdf : config × workload heatmap of normalized
                            EDAP. ACAM-eligible cells (V=1) marked.

EDAP = energy_pj × latency_ns × fpga_area_mm2.
Normalization: per-workload best EDAP → 1.0; others = best/this.
Ranking: geomean of the 4 per-workload normalized EDAPs per config.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "gemm_dse_smoke.csv"

sys.path.insert(0, str(REPO_ROOT / "nl_dpe"))
from area_power import dpe_specs  # noqa: E402

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

# Azure-Lily reference DPE area (same as Round-1). Used for "X% of AL DPE area" annotations.
AL_AREA_MWTA = 2_320_000

# Display order + labels — matches the 4 smoke-test workloads from
# paper/methodology/gemm_rc_tradeoff.md §7.
WORKLOADS_ORDERED = ["bert_qkvo", "bert_ffn2", "swin_mlp", "resnet9_conv"]
WORKLOAD_LABELS = {
    "bert_qkvo":     "BERT Q/K/V/O\n(128×128→128)",
    "bert_ffn2":     "BERT FFN2\n(128×512→128)",
    "swin_mlp":      "Swin MLP\n(49×384→1536)",
    "resnet9_conv":  "ResNet-9 conv\n(256×2304→256)",
}


def area_ratio_to_al(cfg_str: str) -> float:
    R, C = map(int, cfg_str.split("x"))
    return dpe_specs(R, C)["area_tag_mwta"] / AL_AREA_MWTA


def load_and_rank():
    """Load CSV, compute per-workload EDAP, return ranked configs."""
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["energy_pj"] = float(r["energy_pj"])
        r["latency_ns"] = float(r["latency_ns"])
        r["fpga_area_mm2"] = float(r["fpga_area_mm2"])
        r["fmax_mhz"] = float(r["fmax_mhz"])

    configs = sorted(set(r["config"] for r in rows),
                     key=lambda c: (int(c.split("x")[0]), int(c.split("x")[1])))
    workloads = [wl for wl in WORKLOADS_ORDERED
                 if wl in set(r["workload"] for r in rows)]

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
            if e > 0 and d > 0 and a > 0:
                edap[(c, wl)] = e * d * a
            else:
                edap[(c, wl)] = float("inf")
        wl_vals = [edap[(c, wl)] for c in configs
                   if (c, wl) in edap and edap[(c, wl)] < float("inf")]
        best_edap[wl] = min(wl_vals) if wl_vals else 1.0

    # Normalized (best = 1.0) and geomean across workloads
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
    return rows, ranked, scores, idx, workloads


def plot_ranking(ranked, top_k: int = 5) -> None:
    configs_ranked = [cfg for cfg, _ in ranked]
    gm_vals = [sc["gm"] for _, sc in ranked]
    n = len(configs_ranked)
    top3 = configs_ranked[:3]

    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.4)))
    y_pos = np.arange(n)

    colors = [COLOR_TOP if configs_ranked[n - 1 - i] in top3
              else COLOR_EDAP for i in range(n)]
    ax.barh(y_pos, gm_vals[::-1], 0.6, color=colors, alpha=0.85)

    cfg_overhead = {}
    cfg_power = {}
    cfg_area = {}
    for cfg in configs_ranked:
        R, C = map(int, cfg.split("x"))
        s = dpe_specs(R, C)
        cfg_overhead[cfg] = s["routing_um2"] / s["tile_total_um2"] * 100
        cfg_power[cfg] = s["power_total_mw"]
        cfg_area[cfg] = s["tile_total_um2"]
    max_power = max(cfg_power.values())
    min_area = min(cfg_area.values())

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
        if cfg_area[cfg] == min_area:
            ax.text(annot_x, i,
                    f"FPGA integration overhead: {cfg_overhead[cfg]:.0f}% block area",
                    ha="left", va="center", fontsize=7.5, color=COLOR_AL)
        elif cfg_power[cfg] == max_power:
            annot = f"DPE power: {cfg_power[cfg]:.0f} mW"
            if cfg in ("1024x256", "512x256"):
                ratio = area_ratio_to_al(cfg)
                annot += f",  {ratio*100:.0f}% of Azure-Lily DPE area"
            ax.text(annot_x, i, annot,
                    ha="left", va="center", fontsize=7.5, color=COLOR_AL)
        elif cfg in ("1024x256", "512x256"):
            ratio = area_ratio_to_al(cfg)
            ax.text(annot_x, i, f"{ratio*100:.0f}% of Azure-Lily DPE area",
                    ha="left", va="center", fontsize=7.5,
                    fontweight="normal", color=COLOR_AL)

    ax.set_yticks([])
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Normalized Geomean EDAP Score (best = 1.0)")
    ax.set_title("Phase 2.1 GEMM DSE: DPE Config Ranking by EDAP\n"
                 "(Energy × Delay × Area, lower-is-better; best = 1.0)",
                 fontweight="bold")
    ax.axvline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

    out = RESULTS_DIR / "gemm_ranking.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_heatmap(scores, idx, workloads) -> None:
    configs_ranked = [cfg for cfg, _ in sorted(scores.items(), key=lambda x: -x[1]["gm"])]
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

    fig, ax = plt.subplots(figsize=(8, max(4, n_cfg * 0.4)))
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.4, vmax=1.0)
    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

    for i in range(n_cfg):
        for j in range(n_wl):
            val = mat[i, j]
            v = v_mat[i, j]
            if np.isnan(val):
                continue
            tc = "white" if val < 0.2 else "black"
            acam = "\u2713" if v == 1 else ""
            ax.text(j, i, f"{val:.2f}\n{acam}", ha="center", va="center",
                    fontsize=7.5, color=tc,
                    fontweight="bold" if v == 1 else "normal")

    ax.set_xticks(range(n_wl))
    ax.set_xticklabels([WORKLOAD_LABELS.get(wl, wl) for wl in workloads],
                       rotation=0, ha="center", fontsize=8)
    ax.set_yticks(range(n_cfg))
    ax.set_yticklabels([f"#{i+1} {cfg}" for i, cfg in enumerate(configs_ranked)],
                       fontsize=8)
    ax.set_title("Phase 2.1 GEMM DSE: Normalized EDAP per Workload\n"
                 "(\u2713 = ACAM-eligible, V = 1)", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Normalized EDAP (best = 1.0)", fontsize=9)

    out = RESULTS_DIR / "gemm_heatmap_edap.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def main():
    rows, ranked, scores, idx, workloads = load_and_rank()
    print(f"Loaded {len(rows)} rows across {len(workloads)} workloads "
          f"and {len(scores)} configs.")
    print("Ranking (top to bottom):")
    for rank, (cfg, sc) in enumerate(ranked, 1):
        print(f"  #{rank:2d}  {cfg:10s}  gm={sc['gm']:.3f}")

    plot_ranking(ranked)
    plot_heatmap(scores, idx, workloads)


if __name__ == "__main__":
    main()
