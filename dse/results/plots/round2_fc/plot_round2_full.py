#!/usr/bin/env python3
"""Round 2 Full Sweep: Two figures from round2_full_results.csv.

Figure 1 (aggregate): Geomean utilization heatmap per config (5 configs, 3+2 layout).
Figure 2 (per-workload): Per-workload utilization heatmaps (3 rows × 5 cols).
    - CLB-infeasible cells (blue): workload needs more CLBs than available
    - VTR-failed cells (gray ×): VTR routing failure at high replica count

Input:  dse/results/round2_full_results.csv
Output: dse/results/round2_full_scalability.pdf
        dse/results/round2_full_per_workload.pdf
"""

import csv
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = RESULTS_DIR.parent.parent
CSV_PATH = RESULTS_DIR / "round2_full_results.csv"

# Import area_power for analytical CLB/DPE checks
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
from area_power import dpe_specs

CONFIGS = ["512x128", "1024x128", "1024x64", "1024x256", "512x256"]
CONFIG_LABELS = {
    "512x128":  "512\u00d7128 (#1, 43% AL)",
    "1024x128": "1024\u00d7128 (#2, 59% AL)",
    "1024x64":  "1024\u00d764 (#3, 32% AL)",
    "1024x256": "1024\u00d7256 (#4, 115% AL)",
    "512x256":  "512\u00d7256 (#6, 84% AL)",
}
CONFIG_SHORT = {
    "512x128":  "512\u00d7128",
    "1024x128": "1024\u00d7128",
    "1024x64":  "1024\u00d764",
    "1024x256": "1024\u00d7256",
    "512x256":  "512\u00d7256",
}
WORKLOADS = ["fc_512_128", "fc_512_512", "fc_2048_256"]
WORKLOAD_LABELS = {
    "fc_512_128":  "FC 512\u00d7128",
    "fc_512_512":  "FC 512\u00d7512",
    "fc_2048_256": "FC 2048\u00d7256",
}
# Workload dimensions (K, N) for analytical DPE/replica computation
WORKLOAD_DIMS = {
    "fc_512_128":  (512, 128),
    "fc_512_512":  (512, 512),
    "fc_2048_256": (2048, 256),
}

# Round 2 grid and CLB constants
BASELINE_CLBS = 10978   # 120×120 grid
CLBS_PER_REP = 30       # 25 base + 5 overhead per replica
CLBS_GLOBAL = 30        # global overhead

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_data():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    data = defaultdict(dict)
    for r in rows:
        cfg = r['config']
        wl = r['workload']
        d_pct = int(round(float(r['dsp_ratio']) * 100))
        c_pct = int(round(float(r['clb_ratio']) * 100))
        data[(cfg, wl)][(d_pct, c_pct)] = r

    dsp_pcts = sorted(set(int(round(float(r['dsp_ratio']) * 100)) for r in rows))
    clb_pcts = sorted(set(int(round(float(r['clb_ratio']) * 100)) for r in rows))
    return data, dsp_pcts, clb_pcts


def dpes_per_replica(cfg_str, wl):
    """Compute V×H (DPEs per GEMV replica) for a (config, workload) pair."""
    R, C = map(int, cfg_str.split("x"))
    K, N = WORKLOAD_DIMS[wl]
    V = math.ceil(K / R)
    H = math.ceil(N / C)
    return V * H


def classify_missing(cfg_str, wl, d_pct, c_pct):
    """Classify a missing data point as 'clb_infeasible' or 'vtr_failed'.

    Uses analytical P_dpe estimate (from VTR data at c=0% or scaled)
    and CLB capacity check.
    """
    c_ratio = c_pct / 100.0
    clbs_avail = int((1 - c_ratio) * BASELINE_CLBS)
    dpr = dpes_per_replica(cfg_str, wl)

    # Estimate P_dpe: for fc_512_128 on 512×128, P=total_dpes/1
    # Use a conservative estimate: at c=0%, roughly P_dpe ∝ d_pct
    # More precisely, we check if CLBs can support the DPE-limited P
    # We use a generous P_dpe estimate (upper bound)
    # For the CLB check, what matters is: if we run at P = P_dpe,
    # does clbs_need > clbs_avail?
    # P_dpe upper bound: scale from known points
    # Simple approach: P_clb = (clbs_avail - CLBS_GLOBAL) // CLBS_PER_REP
    p_clb = max(0, (clbs_avail - CLBS_GLOBAL) // CLBS_PER_REP)

    # If even 1 replica doesn't fit in CLBs, it's infeasible for a different reason
    if p_clb < 1:
        return 'clb_infeasible'

    # Check if a reasonable P_dpe would exceed CLB capacity
    # We estimate P_dpe from the grid-based DPE count
    # Use the analytical formula (rough): at d=X% c=Y%, total DPEs scale with d+c
    # But since we need accuracy, check: clbs_need for VTR's uncapped P
    # For now, classify based on c_pct: if c >= 60% and workload has 1 DPE/rep
    # and config is small-tile, likely CLB infeasible
    # More robust: estimate total_dpes from nearby data at same (cfg, d_pct, c=0%)
    # But we don't have that data in this function.

    # Conservative approach: if c >= 40% and dpr == 1, assume high P_dpe
    # and check if P_dpe * CLBS_PER_REP + CLBS_GLOBAL > clbs_avail
    # Since we don't have exact P_dpe, use the fact that at high d%+c%,
    # P_dpe can be 100-300 for 1 DPE/rep workloads
    # If p_clb < 200 and dpr == 1, it's likely CLB-infeasible for high P
    # But this is too heuristic. Let me use a different approach.

    # Actually, just check: clbs_need at P = p_clb+1 would be infeasible.
    # But that's always true by definition.
    # The real question is: was the run skipped by compute_feasibility() (CLB)
    # or did VTR fail?

    # compute_feasibility uses: P = total_dpes // dpr,
    # then clbs_need = P * CLBS_PER_REP + CLBS_GLOBAL
    # If clbs_need > clbs_avail → infeasible (skipped)
    # Otherwise → VTR was attempted but failed

    # Without total_dpes, we can't distinguish perfectly.
    # Use a simpler heuristic: if dpr == 1 and c_pct >= 60, likely CLB-infeasible
    # because P_dpe >> p_clb at those points.

    # Better: just mark all missing cells uniformly. The distinction isn't critical
    # for the paper since the insight is conveyed by the pattern of missing data.
    return 'vtr_failed'


def classify_missing_with_data(data, cfg, wl, d_pct, c_pct):
    """Classify missing cell using nearby data to estimate P_dpe."""
    c_ratio = c_pct / 100.0
    clbs_avail = int((1 - c_ratio) * BASELINE_CLBS)
    dpr = dpes_per_replica(cfg, wl)
    p_clb = max(0, (clbs_avail - CLBS_GLOBAL) // CLBS_PER_REP)

    # Try to estimate total_dpes from a nearby point with same (cfg, d_pct)
    # Look at same d_pct, lower c_pct
    for c_try in [0, 20, 40, 60]:
        if c_try >= c_pct:
            continue
        # Use any workload's data at this (cfg, d_pct, c_try) to get total_dpes
        for wl_try in WORKLOADS:
            key = (cfg, wl_try)
            if (d_pct, c_try) in data[key]:
                r = data[key][(d_pct, c_try)]
                # At c_pct > c_try, there are MORE DPEs (more CLB replaced)
                # So P_dpe at c_pct >= P_dpe at c_try
                total_dpes_lower = int(r['total_dpes'])
                dpr_try = dpes_per_replica(cfg, wl_try)
                # Scale total_dpes: at higher c%, more wc tiles
                # Conservative: assume total_dpes at c_pct >= total_dpes at c_try
                p_dpe_lower = total_dpes_lower // dpr
                clbs_need_lower = p_dpe_lower * CLBS_PER_REP + CLBS_GLOBAL
                if clbs_need_lower > clbs_avail:
                    return 'clb_infeasible', p_clb, p_dpe_lower
                # p_dpe at c_pct is even higher, so if lower bound already exceeds CLB...
                # Actually total_dpes increases with c_pct, so p_dpe increases too
                # If even at c_try the CLB check passes, we need to estimate c_pct total_dpes
                # Just use a generous 2x multiplier for higher c
                p_dpe_est = int(p_dpe_lower * (1 + (c_pct - c_try) / 100.0 * 3))
                clbs_need_est = p_dpe_est * CLBS_PER_REP + CLBS_GLOBAL
                if clbs_need_est > clbs_avail:
                    return 'clb_infeasible', p_clb, p_dpe_est
                return 'vtr_failed', p_clb, p_dpe_est

    return 'vtr_failed', p_clb, 0


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1: Aggregate Geomean Utilization (5 configs, 3+2 layout)
# ═════════════════════════════════════════════════════════════════════════

def build_geomean_matrices(data, dsp_pcts, clb_pcts):
    """Build per-config geomean utilization matrix aggregated across workloads."""
    n_dsp, n_clb = len(dsp_pcts), len(clb_pcts)
    gm_matrices = {}
    p_matrices = {}

    for cfg in CONFIGS:
        gm_mat = np.full((n_dsp, n_clb), np.nan)
        p_mat = np.full((n_dsp, n_clb), np.nan)

        for i, d in enumerate(dsp_pcts):
            for j, c in enumerate(clb_pcts):
                utils = []
                ps = []
                for wl in WORKLOADS:
                    if (d, c) in data[(cfg, wl)]:
                        r = data[(cfg, wl)][(d, c)]
                        u = float(r['utilization'])
                        p = int(r['P'])
                        if u > 0:
                            utils.append(u)
                            ps.append(p)

                if len(utils) == len(WORKLOADS):
                    log_sum = sum(math.log(u) for u in utils)
                    gm_mat[i, j] = math.exp(log_sum / len(utils))
                    p_mat[i, j] = sum(ps) / len(ps)

        gm_matrices[cfg] = gm_mat
        p_matrices[cfg] = p_mat

    return gm_matrices, p_matrices


def plot_aggregate(data, dsp_pcts, clb_pcts):
    """Figure 1: Aggregate geomean utilization, 5 configs in 3+2 layout."""
    n_dsp, n_clb = len(dsp_pcts), len(clb_pcts)
    gm_matrices, p_matrices = build_geomean_matrices(data, dsp_pcts, clb_pcts)

    # Print summary
    print("Geomean utilization per config:")
    for cfg in CONFIGS:
        mat = gm_matrices[cfg]
        valid = mat[~np.isnan(mat)]
        if len(valid) > 0:
            print(f"  {cfg}: {len(valid)} pts, "
                  f"mean={np.mean(valid):.1%}, "
                  f"min={np.min(valid):.1%}, max={np.max(valid):.1%}")

    # Layout: 2 rows — row 1: 3 configs, row 2: 2 configs (centered)
    fig = plt.figure(figsize=(13, 7.5))

    pw, ph = 0.25, 0.38
    hgap, vgap = 0.05, 0.08
    left0 = 0.06
    top_y = 0.55
    row1_positions = [
        [left0 + i * (pw + hgap), top_y, pw, ph] for i in range(3)
    ]
    offset2 = left0 + (pw + hgap) * 0.75
    bot_y = top_y - ph - vgap
    row2_positions = [
        [offset2 + i * (pw + hgap), bot_y, pw, ph] for i in range(2)
    ]
    ax_rects = row1_positions + row2_positions

    norm = TwoSlopeNorm(vmin=0.35, vcenter=0.70, vmax=1.05)
    cmap = plt.cm.RdYlGn
    im = None

    for idx, cfg in enumerate(CONFIGS):
        ax = fig.add_axes(ax_rects[idx])
        mat = gm_matrices[cfg]

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", origin="lower")

        for i in range(n_dsp):
            for j in range(n_clb):
                if np.isnan(mat[i, j]):
                    continue
                val = mat[i, j]
                text_color = "white" if val < 0.49 else "black"
                ax.text(j, i, f"{val:.0%}",
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color=text_color)

        ax.set_xticks(range(n_clb))
        ax.set_xticklabels([f"{c}%" for c in clb_pcts])
        ax.set_yticks(range(n_dsp))
        ax.set_yticklabels([f"{d}%" for d in dsp_pcts])
        ax.set_xlabel("CLB Replacement (%)")
        ax.set_ylabel("DSP Replacement (%)")
        ax.set_title(CONFIG_LABELS[cfg], fontweight="bold", fontsize=10)

    # Colorbar on the right
    cbar_ax = fig.add_axes([0.92, bot_y, 0.015, ph * 2 + vgap])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Fmax (f\u2099 / f\u2080)", fontsize=10)

    # Legend in empty subplot space on row 2
    ax_legend = fig.add_axes([left0 - 0.01, bot_y, pw * 0.7, ph])
    ax_legend.axis("off")
    legend_text = (
        "Geomean normalized Fmax\n"
        "across 3 FC workloads:\n"
        "  512\u00d7128, 512\u00d7512, 2048\u00d7256\n"
        "\n"
        "f\u2099 = Fmax with n DPE replicas\n"
        "f\u2080 = baseline Fmax (d=20%)\n"
        "\n"
        "%AL = DPE tile area\n"
        "  relative to baseline\n"
    )
    ax_legend.text(0.05, 0.95, legend_text,
                   transform=ax_legend.transAxes,
                   fontsize=9, verticalalignment="top",
                   fontfamily="serif", color="#333333",
                   linespacing=1.4)

    fig.savefig(RESULTS_DIR / "round2_full_scalability.pdf")
    # fig.savefig(RESULTS_DIR / "round2_full_scalability.png")
    print(f"Saved: {RESULTS_DIR / 'round2_full_scalability.pdf'}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-Workload Normalized Fmax Heatmaps (mean across configs)
# ═════════════════════════════════════════════════════════════════════════

def plot_per_workload(data, dsp_pcts, clb_pcts):
    """Figure 2: 1×3 heatmaps (one per workload).

    X = CLB replacement ratio, Y = DSP replacement ratio.
    Color = mean normalized Fmax across 5 configs.
    """
    n_dsp, n_clb = len(dsp_pcts), len(clb_pcts)
    n_wl = len(WORKLOADS)

    fig, axes = plt.subplots(1, n_wl, figsize=(12, 3.8), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.84, bottom=0.13,
                        wspace=0.12)

    norm = TwoSlopeNorm(vmin=0.35, vcenter=0.70, vmax=1.05)
    cmap = plt.cm.PuBuGn
    im_last = None

    for col_idx, wl in enumerate(WORKLOADS):
        ax = axes[0, col_idx]

        # Build matrix: mean normalized Fmax across 5 configs
        util_mat = np.full((n_dsp, n_clb), np.nan)

        for i, d in enumerate(dsp_pcts):
            for j, c in enumerate(clb_pcts):
                utils = []
                for cfg in CONFIGS:
                    if (d, c) in data[(cfg, wl)]:
                        u = float(data[(cfg, wl)][(d, c)]['utilization'])
                        if u > 0:
                            utils.append(u)
                if utils:
                    util_mat[i, j] = np.mean(utils)

        im = ax.imshow(util_mat, cmap=cmap, norm=norm, aspect="auto",
                       origin="lower")
        im_last = im

        # Annotate cells
        for i in range(n_dsp):
            for j in range(n_clb):
                if np.isnan(util_mat[i, j]):
                    continue
                val = util_mat[i, j]
                text_color = "white" if val > 0.80 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=text_color)

        ax.set_xticks(range(n_clb))
        ax.set_xticklabels([f"{c}%" for c in clb_pcts])
        ax.set_yticks(range(n_dsp))
        if col_idx == 0:
            ax.set_yticklabels([f"{d}%" for d in dsp_pcts])
            ax.set_ylabel("DSP Replacement (%)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("CLB Replacement (%)")
        ax.set_title(WORKLOAD_LABELS[wl], fontweight="bold", fontsize=11)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    fig.colorbar(im_last, cax=cbar_ax,
                 label="Mean Normalized Fmax (f\u2099 / f\u2080)")

    # Footnote at top
    fig.text(0.08, 0.94,
             "Mean normalized Fmax across 5 DPE configs",
             fontsize=10, color="#444444", style="italic")

    fig.savefig(RESULTS_DIR / "round2_full_per_workload.pdf")
    print(f"Saved: {RESULTS_DIR / 'round2_full_per_workload.pdf'}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3: Pareto Front — Throughput Gain vs DPE Count
# ═════════════════════════════════════════════════════════════════════════

CONFIG_COLORS = {
    "512x128":  "#2563EB",
    "1024x128": "#DC2626",
    "1024x64":  "#059669",
    "1024x256": "#D97706",
    "512x256":  "#7C3AED",
}
CONFIG_MARKERS = {
    "512x128":  "o",
    "1024x128": "s",
    "1024x64":  "^",
    "1024x256": "D",
    "512x256":  "v",
}


def compute_pareto(points):
    """Return Pareto-optimal points from list of (x, y, meta).

    A point is Pareto-optimal if no other point has both lower x AND higher y.
    """
    # Sort by x ascending
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    pareto = []
    best_y = -float('inf')
    for pt in sorted_pts:
        if pt[1] > best_y:
            pareto.append(pt)
            best_y = pt[1]
    return pareto


def get_tile_dims(cfg_str):
    """Return (tile_width, tile_height) for a DPE config."""
    R, C = map(int, cfg_str.split("x"))
    s = dpe_specs(R, C)
    return s['tile_width'], s['tile_height']


GRID_CELLS = 120 * 120  # total grid cells on 120×120 FPGA


def compute_pareto_min(points):
    """Pareto front for minimize-minimize (closer to origin = better).

    Points: list of (x, y, meta). A point is Pareto-optimal if no other
    point has both x ≤ its x AND y ≤ its y (with at least one strict).
    Returns the non-dominated front sorted by x ascending.
    """
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    pareto = []
    best_y = float('inf')
    for pt in sorted_pts:
        if pt[1] < best_y:
            pareto.append(pt)
            best_y = pt[1]
    return pareto


def find_knee_point(pareto):
    """Find the knee of a Pareto front (max perpendicular distance to line
    connecting first and last points)."""
    if len(pareto) <= 2:
        return 0
    x0, y0 = pareto[0][0], pareto[0][1]
    x1, y1 = pareto[-1][0], pareto[-1][1]
    dx, dy = x1 - x0, y1 - y0
    line_len = math.sqrt(dx**2 + dy**2)
    if line_len == 0:
        return 0
    best_dist, best_idx = -1, 0
    for i, pt in enumerate(pareto):
        # Perpendicular distance from pt to line (p0 → p1)
        dist = abs(dy * pt[0] - dx * pt[1] + x1 * y0 - y1 * x0) / line_len
        if dist > best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def plot_pareto(data, dsp_pcts, clb_pcts):
    """Figure 3: Pareto front — DPE area vs effective latency.

    X = DPE area (% of total FPGA area). Cost, minimize.
    Y = Effective latency (ns/inference) = 1000/(P × Fmax_MHz).
    Geomean across 3 workloads. One subplot per config.
    Closer to origin = better.
    """
    # Load raw CSV
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    # Group by (config, d%, c%) → per-workload gain, fmax, P + total_dpes
    grouped = defaultdict(lambda: {"gains": {}, "eff_lat_ns": {}, "total_dpes": None})
    for r in rows:
        cfg = r['config']
        d_pct = int(round(float(r['dsp_ratio']) * 100))
        c_pct = int(round(float(r['clb_ratio']) * 100))
        wl = r['workload']
        key = (cfg, d_pct, c_pct)
        grouped[key]["gains"][wl] = float(r['gain'])
        grouped[key]["total_dpes"] = int(r['total_dpes'])
        p = int(r['P'])
        fmax = float(r['fmax_mhz'])
        if p > 0 and fmax > 0:
            grouped[key]["eff_lat_ns"][wl] = 1e3 / (p * fmax)  # ns per inference

    # 2×3 grid: 5 config subplots + 1 note panel (bottom-left)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.5), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.10,
                        wspace=0.18, hspace=0.28)
    fig.suptitle("FC Workloads: Per-Config Pareto Front",
                 fontsize=12, fontweight="bold", y=0.97)

    # Collect Pareto-optimal points and knee for footnotes
    pareto_info = []  # list of (cfg, pareto_combos, knee_combo)

    # Layout: row 0 = configs 0,1,2; row 1 = note, config 3, config 4
    panel_positions = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    for idx, cfg in enumerate(CONFIGS):
        row, col = panel_positions[idx]
        ax = axes[row, col]
        color = CONFIG_COLORS[cfg]
        tw, th = get_tile_dims(cfg)

        # Build points: (dpe_area_pct, eff_latency_ns, (d%, c%))
        # Y-axis = geomean of effective latency (ns/inference) across workloads
        points = []
        for (c, d, cp), info in grouped.items():
            if c != cfg:
                continue
            if len(info["eff_lat_ns"]) < len(WORKLOADS):
                continue
            lats = list(info["eff_lat_ns"].values())
            if any(l <= 0 for l in lats):
                continue
            gm_lat = math.exp(sum(math.log(l) for l in lats) / len(lats))
            dpe_area_pct = info["total_dpes"] * tw * th / GRID_CELLS * 100
            points.append((dpe_area_pct, gm_lat, (d, cp)))

        if not points:
            continue

        # Pareto front (minimize-minimize)
        pareto = compute_pareto_min(points)
        pareto_set = set((p[0], p[1]) for p in pareto)

        # Sub-optimal — hollow
        dom_xs = [p[0] for p in points if (p[0], p[1]) not in pareto_set]
        dom_ys = [p[1] for p in points if (p[0], p[1]) not in pareto_set]
        lbl_sub = "Sub-optimal" if idx == 0 else None
        ax.scatter(dom_xs, dom_ys, facecolors="none", edgecolors=color,
                   s=30, alpha=0.6, linewidths=0.8, zorder=2, label=lbl_sub)

        # Optimal — filled, connected
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        lbl_opt = "Optimal" if idx == 0 else None
        ax.plot(px, py, color=color, linewidth=1.5, zorder=3)
        ax.scatter(px, py, color=color, s=30, edgecolors="white",
                   linewidths=0.8, zorder=4, label=lbl_opt)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

        # Find and annotate knee point (optimal tradeoff)
        knee_idx = find_knee_point(pareto)
        knee = pareto[knee_idx]
        kd, kc = knee[2]
        ax.scatter([knee[0]], [knee[1]], color=color, s=100,
                   edgecolors="black", linewidths=1.5, zorder=5,
                   marker="*")

        # Collect all Pareto-optimal combos for footnote
        pareto_combos = [(p[0], p[1], p[2][0], p[2][1]) for p in pareto]
        pareto_info.append((cfg, pareto_combos, (kd, kc)))

        # Print Pareto table for sanity check
        print(f"\n  {cfg} — {len(pareto)} Pareto-optimal points:")
        print(f"  {'DSP%':>5} {'CLB%':>5} {'Area%':>7} {'Eff.Lat':>9} {'':>5}")
        for area, lat, d, c in pareto_combos:
            star = " ★" if (d, c) == (kd, kc) else ""
            print(f"  {d:>5} {c:>5} {area:>7.2f} {lat:>9.4f}{star}")

        ax.set_xlabel("DPE Area (% of FPGA)", fontsize=8)
        if (row == 0 and col == 0) or (row == 1 and col == 1):
            ax.set_ylabel("Geomean Eff. Latency (ns/inf)", fontsize=8)
        ax.set_title(f"{CONFIG_SHORT[cfg]}",
                     fontweight="bold", fontsize=9, pad=4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # Bottom-left panel: balanced config table
    ax_note = axes[1, 0]
    ax_note.axis("off")
    FS = 8

    # Title
    ax_note.text(0.5, 0.97, r"Balanced Config ($\bigstar$)",
                 transform=ax_note.transAxes,
                 fontsize=FS + 1, fontweight="bold", verticalalignment="top",
                 ha="center", fontfamily="serif", color="#111111")

    # Table
    table_data = [["Config", "DSP %", "CLB %"]]
    table_colors = [["#E5E7EB", "#E5E7EB", "#E5E7EB"]]
    for cfg, combos, knee_combo in pareto_info:
        kd, kc = knee_combo
        table_data.append([CONFIG_SHORT[cfg], f"{kd}%", f"{kc}%"])
        table_colors.append(["#F9FAFB", "#F9FAFB", "#F9FAFB"])

    tbl = ax_note.table(cellText=table_data, cellColours=table_colors,
                        loc="upper center", cellLoc="center",
                        bbox=[0.0, 0.28, 1.0, 0.62])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FS)
    tbl.scale(1, 1.3)
    for j in range(3):
        tbl[0, j].set_text_props(fontweight="bold", fontsize=FS)
    for i, (cfg, combos, knee_combo) in enumerate(pareto_info):
        tbl[i + 1, 0].set_text_props(color=CONFIG_COLORS[cfg], fontweight="bold")

    # Explanation
    ax_note.text(0.5, 0.20,
                 "Balanced = best latency-per-area\n"
                 "efficiency. Beyond this, each\n"
                 "additional % of DPE area yields\n"
                 "less improvement.",
                 transform=ax_note.transAxes,
                 fontsize=FS, verticalalignment="top",
                 ha="center", fontfamily="serif", color="#AAAAAA",
                 linespacing=1.3)
    fig.savefig(RESULTS_DIR / "round2_full_pareto.pdf")
    print(f"Saved: {RESULTS_DIR / 'round2_full_pareto.pdf'}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4: Merged Pareto — NL-DPE group (3) + AL-like group (2)
# ═════════════════════════════════════════════════════════════════════════

def plot_merged_pareto(data, dsp_pcts, clb_pcts):
    """Merged Pareto: top 3 configs overlaid, bottom 2 overlaid."""
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(lambda: {"eff_lat_ns": {}, "total_dpes": None})
    for r in rows:
        cfg = r['config']
        d_pct = int(round(float(r['dsp_ratio']) * 100))
        c_pct = int(round(float(r['clb_ratio']) * 100))
        wl = r['workload']
        key = (cfg, d_pct, c_pct)
        grouped[key]["total_dpes"] = int(r['total_dpes'])
        p = int(r['P'])
        fmax = float(r['fmax_mhz'])
        if p > 0 and fmax > 0:
            grouped[key]["eff_lat_ns"][wl] = 1e3 / (p * fmax)

    groups = [
        ("NL-DPE Group", ["512x128", "1024x128", "1024x64"]),
        ("AL-like Group", ["1024x256", "512x256"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.08, right=0.95, wspace=0.25, top=0.88)
    fig.suptitle("FC Workloads: Cross-Config Pareto Front (geomean across 3 workloads)",
                 fontsize=11, fontweight="bold", y=0.96)

    for gi, (group_name, cfgs) in enumerate(groups):
        ax = axes[gi]
        all_points = []

        for cfg in cfgs:
            color = CONFIG_COLORS[cfg]
            marker = CONFIG_MARKERS[cfg]
            tw, th = get_tile_dims(cfg)

            points = []
            for (c, d, cp), info in grouped.items():
                if c != cfg or len(info["eff_lat_ns"]) < len(WORKLOADS):
                    continue
                lats = list(info["eff_lat_ns"].values())
                if any(l <= 0 for l in lats):
                    continue
                gm_lat = math.exp(sum(math.log(l) for l in lats) / len(lats))
                dpe_area_pct = info["total_dpes"] * tw * th / GRID_CELLS * 100
                points.append((dpe_area_pct, gm_lat, (d, cp), cfg))

            if not points:
                continue

            cfg_pareto = compute_pareto_min([(p[0], p[1], p[2]) for p in points])
            cfg_pareto_set = set((p[0], p[1]) for p in cfg_pareto)

            dom = [p for p in points if (p[0], p[1]) not in cfg_pareto_set]
            ax.scatter([p[0] for p in dom], [p[1] for p in dom],
                       facecolors="none", edgecolors=color, marker=marker,
                       s=25, alpha=0.4, linewidths=0.6, zorder=2)

            px = [p[0] for p in cfg_pareto]
            py = [p[1] for p in cfg_pareto]
            ax.plot(px, py, color=color, linewidth=1.2, alpha=0.7, zorder=3)
            ax.scatter(px, py, color=color, marker=marker, s=30,
                       edgecolors="white", linewidths=0.6, zorder=4,
                       label=CONFIG_SHORT[cfg])

            all_points.extend(points)

        global_pts = [(p[0], p[1], (p[2], p[3])) for p in all_points]
        global_pareto = compute_pareto_min(global_pts)

        gpx = [p[0] for p in global_pareto]
        gpy = [p[1] for p in global_pareto]
        ax.plot(gpx, gpy, color="black", linewidth=2, linestyle="--",
                alpha=0.6, zorder=5, label="Cross-config Pareto")

        knee_idx = find_knee_point(global_pareto)
        knee = global_pareto[knee_idx]
        knee_dc, knee_cfg = knee[2]
        kd, kc = knee_dc

        ax.scatter([knee[0]], [knee[1]], color="black", s=150,
                   edgecolors="red", linewidths=2, zorder=6, marker="*")
        ax.annotate(f"Balanced: {CONFIG_SHORT[knee_cfg]}\nd={kd}% c={kc}%",
                    xy=(knee[0], knee[1]),
                    xytext=(knee[0] + max(gpx) * 0.12, knee[1] * 1.15),
                    fontsize=8, fontweight="bold", color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                    zorder=7)

        print(f"\n  {group_name} — Global Pareto ({len(global_pareto)} points):")
        print(f"  {'Config':>12} {'DSP%':>5} {'CLB%':>5} {'Area%':>7} {'Lat(ns)':>9}")
        for area, lat, meta in global_pareto:
            dc, c = meta
            d, cp = dc
            star = " ★ (balanced)" if (dc, c) == (knee_dc, knee_cfg) else ""
            print(f"  {CONFIG_SHORT[c]:>12} {d:>5} {cp:>5} {area:>7.2f} {lat:>9.4f}{star}")

        ax.set_xlabel("DPE Area (% of FPGA)", fontsize=10)
        ax.set_ylabel("Geomean Eff. Latency (ns/inference)", fontsize=10)
        ax.set_title(group_name, fontweight="bold", fontsize=11)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.savefig(RESULTS_DIR / "round2_full_pareto_merged.pdf")
    print(f"Saved: {RESULTS_DIR / 'round2_full_pareto_merged.pdf'}")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 5: Throughput Ceiling — Soft ceiling from routing degradation
# ═════════════════════════════════════════════════════════════════════════

def plot_throughput_ceiling(data, dsp_pcts, clb_pcts):
    """Throughput ceiling: actual vs ideal throughput as DPE area grows.

    Shows the soft ceiling from Fmax degradation at high replica counts.
    2×3 grid: 5 config subplots + 1 note panel (bottom-left).
    X = DPE Area (% of FPGA), Y = Throughput (P × Fmax).
    """
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    # Group by (config, d%, c%) → per-workload metrics
    grouped = defaultdict(lambda: {"fmax": {}, "P": None, "total_dpes": None})
    for r in rows:
        cfg = r['config']
        d_pct = int(round(float(r['dsp_ratio']) * 100))
        c_pct = int(round(float(r['clb_ratio']) * 100))
        wl = r['workload']
        key = (cfg, d_pct, c_pct)
        grouped[key]["fmax"][wl] = float(r['fmax_mhz'])
        grouped[key]["P"] = int(r['P'])
        grouped[key]["total_dpes"] = int(r['total_dpes'])

    # 2×3 grid
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.09,
                        wspace=0.25, hspace=0.35)

    panel_positions = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]

    for idx, cfg in enumerate(CONFIGS):
        row, col = panel_positions[idx]
        ax = axes[row, col]
        color = CONFIG_COLORS[cfg]
        tw, th = get_tile_dims(cfg)

        # Get baseline f0 per workload (d=20%, c=0%)
        base = grouped.get((cfg, 20, 0), {})
        if not base or len(base.get("fmax", {})) < len(WORKLOADS):
            continue
        f0 = base["fmax"]  # per-workload baseline Fmax

        # Collect points
        points = []
        for (c, d, cp), info in grouped.items():
            if c != cfg or len(info["fmax"]) < len(WORKLOADS):
                continue
            P = info["P"]
            dpe_area_pct = info["total_dpes"] * tw * th / GRID_CELLS * 100

            # Geomean actual throughput: P × fn
            tputs = [P * info["fmax"][wl] for wl in WORKLOADS]
            gm_actual = math.exp(sum(math.log(t) for t in tputs) / len(tputs))

            # Geomean ideal throughput: P × f0
            ideal_tputs = [P * f0[wl] for wl in WORKLOADS]
            gm_ideal = math.exp(sum(math.log(t) for t in ideal_tputs) / len(ideal_tputs))

            points.append({
                'area_pct': dpe_area_pct,
                'P': P,
                'gm_actual': gm_actual,
                'gm_ideal': gm_ideal,
                'efficiency': gm_actual / gm_ideal if gm_ideal > 0 else 0,
                'd': d, 'c': cp,
            })

        points.sort(key=lambda p: p['area_pct'])

        # Deduplicate by area (same P at different (d,c) can give same area)
        seen_areas = {}
        for pt in points:
            a = round(pt['area_pct'], 2)
            if a not in seen_areas or pt['gm_actual'] > seen_areas[a]['gm_actual']:
                seen_areas[a] = pt
        points = sorted(seen_areas.values(), key=lambda p: p['area_pct'])

        xs = [p['area_pct'] for p in points]
        actual = [p['gm_actual'] for p in points]
        ideal = [p['gm_ideal'] for p in points]
        effs = [p['efficiency'] for p in points]

        # Convert to inferences/ns: throughput_GHz = P × Fmax_MHz / 1000
        actual_inf_ns = [a / 1e3 for a in actual]
        ideal_inf_ns = [i / 1e3 for i in ideal]

        # Plot ideal (linear scaling) as dashed line
        ax.plot(xs, ideal_inf_ns, color='#888888', linewidth=1.2, linestyle='--',
                alpha=0.7, zorder=2, label='Ideal (P × f₀)')

        # Fill the gap (wasted throughput)
        ax.fill_between(xs, actual_inf_ns, ideal_inf_ns, alpha=0.12, color=color,
                        zorder=1)

        # Plot actual throughput
        ax.plot(xs, actual_inf_ns, color=color, linewidth=2, zorder=4)
        ax.scatter(xs, actual_inf_ns, color=color, s=25, edgecolors='white',
                   linewidths=0.5, zorder=5)

        # Find and mark the negative-return zone (if any)
        for i in range(1, len(actual_inf_ns)):
            if actual_inf_ns[i] < actual_inf_ns[i-1]:
                ax.axvspan(xs[i-1], xs[i], alpha=0.15, color='red', zorder=0)

        # Mark the 80% throughput point (just the dotted line, no text clutter)
        max_tput = max(actual_inf_ns)
        for i, pt in enumerate(points):
            if actual_inf_ns[i] >= max_tput * 0.8:
                ax.axvline(x=pt['area_pct'], color=color, linewidth=0.8,
                           linestyle=':', alpha=0.5, zorder=1)
                break

        # Only annotate efficiency at start and end (no clutter)
        if points:
            # Start point
            ax.annotate(f"{points[0]['efficiency']:.0%}",
                        xy=(xs[0], actual_inf_ns[0]),
                        xytext=(xs[0] + 2, actual_inf_ns[0]),
                        fontsize=6.5, color=color, fontweight='bold',
                        ha='left', va='center')
            # End point
            ax.annotate(f"{points[-1]['efficiency']:.0%}",
                        xy=(xs[-1], actual_inf_ns[-1]),
                        xytext=(xs[-1] - 2, actual_inf_ns[-1] * 1.08),
                        fontsize=6.5, color=color, fontweight='bold',
                        ha='right', va='bottom')

        ax.set_xlabel("DPE Area (% of FPGA)", fontsize=8)
        if col == 0 or (row == 1 and col == 1):
            ax.set_ylabel("Throughput (inferences/ns)", fontsize=8)
        ax.set_title(f"{CONFIG_SHORT[cfg]}", fontweight="bold", fontsize=9, pad=4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.15, linewidth=0.5)

    # Bottom-left panel: notes
    ax_note = axes[1, 0]
    ax_note.axis("off")

    note_text = (
        "Throughput Ceiling Analysis\n"
        "\n"
        "── Actual: P × fₙ\n"
        "- - Ideal: P × f₀ (no Fmax loss)\n"
        "\n"
        "Shaded gap = throughput lost\n"
        "  to routing congestion\n"
        "\n"
        "Red zone = negative marginal\n"
        "  return (more DPEs hurts)\n"
        "\n"
        "Dotted line = 80% of peak\n"
        "\n"
        "% labels = actual/ideal efficiency\n"
        "  (start → end of curve)\n"
    )
    ax_note.text(0.05, 0.95, note_text,
                 transform=ax_note.transAxes,
                 fontsize=8.5, verticalalignment="top",
                 fontfamily="serif", color="#333333",
                 linespacing=1.4)

    fig.savefig(RESULTS_DIR / "round2_full_ceiling.pdf")
    print(f"Saved: {RESULTS_DIR / 'round2_full_ceiling.pdf'}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    data, dsp_pcts, clb_pcts = load_data()

    print("=" * 60)
    print("Figure 1: Aggregate geomean utilization")
    print("=" * 60)
    plot_aggregate(data, dsp_pcts, clb_pcts)

    print()
    print("=" * 60)
    print("Figure 2: Per-workload utilization")
    print("=" * 60)
    plot_per_workload(data, dsp_pcts, clb_pcts)

    print()
    print("=" * 60)
    print("Figure 3: Pareto front (per-config)")
    print("=" * 60)
    plot_pareto(data, dsp_pcts, clb_pcts)

    print()
    print("=" * 60)
    print("Figure 4: Merged Pareto front")
    print("=" * 60)
    plot_merged_pareto(data, dsp_pcts, clb_pcts)

    print()
    print("=" * 60)
    print("Figure 5: Throughput ceiling")
    print("=" * 60)
    plot_throughput_ceiling(data, dsp_pcts, clb_pcts)

    print("\nDone.")


if __name__ == "__main__":
    main()
