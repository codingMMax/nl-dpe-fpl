#!/usr/bin/env python3
"""Round 2 Attention Sweep: Scalability heatmap + Pareto front.

Input:  dse/results/round2_attention_results.csv
Output: dse/results/round2_attention_scalability.pdf
        dse/results/round2_attention_pareto.pdf
"""

import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from collections import defaultdict

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

RESULTS_DIR = Path(__file__).resolve().parent
DATA_DIR = RESULTS_DIR.parent / "data"
OUTPUT_DIR = RESULTS_DIR.parent / "figures"
CSV_PATH = RESULTS_DIR / "round2_attention_results.csv"

CONFIGS = ["512x128", "1024x128", "1024x64", "1024x256", "512x256"]
CONFIG_LABELS = {
    "512x128":  "512×128",
    "1024x128": "1024×128",
    "1024x64":  "1024×64",
    "1024x256": "1024×256",
    "512x256":  "512×256",
}

DSP_RATIOS = [0.2, 0.4, 0.6, 0.8]  # d=100% excluded (no DSPs for softmax)
CLB_RATIOS = [0.0, 0.2, 0.4, 0.6]

CLB_TILE_UM2 = 2239


def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ['dsp_ratio', 'clb_ratio', 'P', 'fmax_mhz', 'gain',
                       'utilization', 'energy_pj', 'total_dpes',
                       'fmax_baseline_mhz', 'energy_baseline_pj', 'energy_ratio']:
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except (ValueError, TypeError):
                        r[k] = 0.0
            rows.append(r)
    return rows


def plot_scalability(rows):
    """Heatmap: utilization (fn/f0) for each config, 3+2 layout matching FC style."""
    n_dsp = len(DSP_RATIOS)
    n_clb = len(CLB_RATIOS)
    dsp_pcts = [int(d * 100) for d in DSP_RATIOS]
    clb_pcts = [int(c * 100) for c in CLB_RATIOS]

    # Build per-config matrices
    matrices = {}
    p_matrices = {}
    for cfg in CONFIGS:
        mat = np.full((n_dsp, n_clb), np.nan)
        p_mat = np.full((n_dsp, n_clb), 0, dtype=int)
        cfg_rows = [r for r in rows if r['config'] == cfg]
        for r in cfg_rows:
            d_idx = c_idx = None
            for i, d in enumerate(DSP_RATIOS):
                if abs(r['dsp_ratio'] - d) < 0.01:
                    d_idx = i
            for j, c in enumerate(CLB_RATIOS):
                if abs(r['clb_ratio'] - c) < 0.01:
                    c_idx = j
            if d_idx is not None and c_idx is not None:
                mat[d_idx, c_idx] = r['utilization']
                p_mat[d_idx, c_idx] = int(r['P'])
        matrices[cfg] = mat
        p_matrices[cfg] = p_mat

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

    norm = TwoSlopeNorm(vmin=0.70, vcenter=0.90, vmax=1.10)
    cmap = plt.cm.RdYlGn
    im = None

    for idx, cfg in enumerate(CONFIGS):
        ax = fig.add_axes(ax_rects[idx])
        mat = matrices[cfg]
        p_mat = p_matrices[cfg]

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", origin="lower")

        for i in range(n_dsp):
            for j in range(n_clb):
                if np.isnan(mat[i, j]):
                    continue
                val = mat[i, j]
                p = p_mat[i, j]
                text_color = "white" if val < 0.75 else "black"
                ax.text(j, i, f"P={p}\n{val:.0%}",
                        ha="center", va="center", fontsize=7.5,
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

    # Legend in empty space (bottom-left)
    ax_legend = fig.add_axes([left0 - 0.01, bot_y, pw * 0.7, ph])
    ax_legend.axis("off")
    legend_text = (
        "Attention head utilization\n"
        "(N=128, d=128)\n"
        "\n"
        "DPEs/rep = (3V+4)\u00d7H\n"
        "P capped at 7 (BRAM)\n"
        "\n"
        "f\u2099 = Fmax with P replicas\n"
        "f\u2080 = baseline Fmax (d=20%)\n"
        "\n"
        "d=100% excluded\n"
        "(no DSPs for softmax)\n"
    )
    ax_legend.text(0.05, 0.95, legend_text,
                   transform=ax_legend.transAxes,
                   fontsize=9, verticalalignment="top",
                   fontfamily="serif", color="#333333",
                   linespacing=1.4)

    out = RESULTS_DIR / "round2_attention_scalability.pdf"
    fig.savefig(out, )
    print(f"  Saved {out}")
    plt.close(fig)


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
CONFIG_SHORT = {
    "512x128": "512\u00d7128", "1024x128": "1024\u00d7128", "1024x64": "1024\u00d764",
    "1024x256": "1024\u00d7256", "512x256": "512\u00d7256",
}

GRID_CELLS = 120 * 120


def get_tile_dims(cfg_str):
    R, C = map(int, cfg_str.split("x"))
    from area_power import dpe_specs
    s = dpe_specs(R, C)
    return s['tile_width'], s['tile_height']


def compute_pareto_min(points):
    """Pareto front for minimize-minimize (closer to origin = better)."""
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    pareto = []
    best_y = float('inf')
    for pt in sorted_pts:
        if pt[1] < best_y:
            pareto.append(pt)
            best_y = pt[1]
    return pareto


def find_knee_point(pareto):
    """Find the knee of a Pareto front (max perpendicular distance)."""
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
        dist = abs(dy * pt[0] - dx * pt[1] + x1 * y0 - y1 * x0) / line_len
        if dist > best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def plot_pareto(rows):
    """Pareto front — DPE area vs effective latency (ns/inference).

    2×3 grid: 5 config subplots + 1 note panel (bottom-left).
    Same layout as FC Pareto in plot_round2_full.py.
    """
    # Group by (config, d%, c%) → eff_latency + total_dpes
    grouped = defaultdict(lambda: {"eff_lat_ns": None, "total_dpes": None, "P": None})
    for r in rows:
        cfg = r['config']
        d_pct = int(round(r['dsp_ratio'] * 100))
        c_pct = int(round(r['clb_ratio'] * 100))
        key = (cfg, d_pct, c_pct)
        p = int(r['P'])
        fmax = r['fmax_mhz']
        if p > 0 and fmax > 0:
            grouped[key]["eff_lat_ns"] = 1e3 / (p * fmax)
        grouped[key]["total_dpes"] = int(r['total_dpes'])
        grouped[key]["P"] = p

    # 2×3 grid
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.5), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.10,
                        wspace=0.18, hspace=0.28)
    fig.suptitle("Attention Head: Per-Config Pareto Front",
                 fontsize=12, fontweight="bold", y=0.97)

    pareto_info = []

    # Layout: row 0 = configs 0,1,2; row 1 = note, config 3, config 4
    panel_positions = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    for idx, cfg in enumerate(CONFIGS):
        row, col = panel_positions[idx]
        ax = axes[row, col]
        color = CONFIG_COLORS[cfg]
        tw, th = get_tile_dims(cfg)

        # Build points: (dpe_area_pct, eff_latency_ns, (d%, c%))
        points = []
        for (c, d, cp), info in grouped.items():
            if c != cfg:
                continue
            if info["eff_lat_ns"] is None or info["eff_lat_ns"] <= 0:
                continue
            dpe_area_pct = info["total_dpes"] * tw * th / GRID_CELLS * 100
            points.append((dpe_area_pct, info["eff_lat_ns"], (d, cp)))

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

        # Find and annotate knee point
        knee_idx = find_knee_point(pareto)
        knee = pareto[knee_idx]
        kd, kc = knee[2]
        ax.scatter([knee[0]], [knee[1]], color=color, s=100,
                   edgecolors="black", linewidths=1.5, zorder=5,
                   marker="*")

        pareto_combos = [(p[0], p[1], p[2][0], p[2][1]) for p in pareto]
        pareto_info.append((cfg, pareto_combos, (kd, kc)))

        # Print Pareto table
        print(f"\n  {cfg} — {len(pareto)} Pareto-optimal points:")
        print(f"  {'DSP%':>5} {'CLB%':>5} {'Area%':>7} {'Lat(ns)':>9}")
        for area, lat, d, c in pareto_combos:
            star = " ★" if (d, c) == (kd, kc) else ""
            print(f"  {d:>5} {c:>5} {area:>7.2f} {lat:>9.4f}{star}")

        ax.set_xlabel("DPE Area (% of FPGA)", fontsize=8)
        if (row == 0 and col == 0) or (row == 1 and col == 1):
            ax.set_ylabel("Effective Latency (ns/inference)", fontsize=8)
        ax.set_title(f"{CONFIG_SHORT[cfg]}", fontweight="bold", fontsize=9, pad=4)
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

    out = RESULTS_DIR / "round2_attention_pareto.pdf"
    fig.savefig(out, )
    print(f"  Saved {out}")
    plt.close(fig)


def plot_merged_pareto(rows):
    """Merged Pareto: top 3 configs in one subplot, bottom 2 in another.

    Overlays all configs in each group on the same axes to find the
    cross-config optimal point.
    """
    grouped = defaultdict(lambda: {"eff_lat_ns": None, "total_dpes": None, "P": None})
    for r in rows:
        cfg = r['config']
        d_pct = int(round(r['dsp_ratio'] * 100))
        c_pct = int(round(r['clb_ratio'] * 100))
        key = (cfg, d_pct, c_pct)
        p = int(r['P'])
        fmax = r['fmax_mhz']
        if p > 0 and fmax > 0:
            grouped[key]["eff_lat_ns"] = 1e3 / (p * fmax)
        grouped[key]["total_dpes"] = int(r['total_dpes'])
        grouped[key]["P"] = p

    groups = [
        ("NL-DPE Group", ["512x128", "1024x128", "1024x64"]),
        ("AL-like Group", ["1024x256", "512x256"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.08, right=0.95, wspace=0.25, top=0.88)
    fig.suptitle("Attention Head: Cross-Config Pareto Front (N=128, d=128)",
                 fontsize=11, fontweight="bold", y=0.96)

    for gi, (group_name, cfgs) in enumerate(groups):
        ax = axes[gi]
        all_points = []  # collect across configs for global Pareto

        for cfg in cfgs:
            color = CONFIG_COLORS[cfg]
            tw, th = get_tile_dims(cfg)

            points = []
            for (c, d, cp), info in grouped.items():
                if c != cfg or info["eff_lat_ns"] is None or info["eff_lat_ns"] <= 0:
                    continue
                dpe_area_pct = info["total_dpes"] * tw * th / GRID_CELLS * 100
                points.append((dpe_area_pct, info["eff_lat_ns"], (d, cp), cfg))

            if not points:
                continue

            # Per-config Pareto
            cfg_pareto = compute_pareto_min([(p[0], p[1], p[2]) for p in points])
            cfg_pareto_set = set((p[0], p[1]) for p in cfg_pareto)

            # Dominated — hollow
            marker = CONFIG_MARKERS[cfg]
            dom = [p for p in points if (p[0], p[1]) not in cfg_pareto_set]
            ax.scatter([p[0] for p in dom], [p[1] for p in dom],
                       facecolors="none", edgecolors=color, marker=marker,
                       s=25, alpha=0.4, linewidths=0.6, zorder=2)

            # Per-config Pareto — filled, connected
            px = [p[0] for p in cfg_pareto]
            py = [p[1] for p in cfg_pareto]
            ax.plot(px, py, color=color, linewidth=1.2, alpha=0.7, zorder=3)
            ax.scatter(px, py, color=color, marker=marker, s=30,
                       edgecolors="white", linewidths=0.6, zorder=4,
                       label=CONFIG_SHORT[cfg])

            all_points.extend(points)

        # Global Pareto across all configs in this group
        global_pts = [(p[0], p[1], (p[2], p[3])) for p in all_points]
        global_pareto = compute_pareto_min(global_pts)

        # Draw global Pareto front
        gpx = [p[0] for p in global_pareto]
        gpy = [p[1] for p in global_pareto]
        ax.plot(gpx, gpy, color="black", linewidth=2, linestyle="--",
                alpha=0.6, zorder=5, label="Cross-config Pareto")

        # Find and annotate knee
        knee_idx = find_knee_point(global_pareto)
        knee = global_pareto[knee_idx]
        knee_meta = knee[2]  # ((d%, c%), cfg)
        knee_dc, knee_cfg = knee_meta
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
            star = " ★" if (dc, c) == (knee_dc, knee_cfg) else ""
            print(f"  {CONFIG_SHORT[c]:>12} {d:>5} {cp:>5} {area:>7.2f} {lat:>9.4f}{star}")

        ax.set_xlabel("DPE Area (% of FPGA)", fontsize=10)
        ax.set_ylabel("Effective Latency (ns/inference)", fontsize=10)
        ax.set_title(group_name, fontweight="bold", fontsize=11)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    out = RESULTS_DIR / "round2_attention_pareto_merged.pdf"
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close(fig)


def plot_throughput_ceiling(rows):
    """Throughput ceiling plot: DPE capacity vs BRAM ceiling.

    Shows how BRAM limits actual throughput well below what DPEs could deliver.
    For a representative config (512×128), plots:
    - DPE throughput ceiling: P_dpe × f0 (if DPE were the only constraint)
    - BRAM throughput ceiling: P_bram × f0 (horizontal line at P=7)
    - Actual throughput: P × fn (measured)
    """
    from area_power import dpe_specs

    # Use 512×128 as representative config
    rep_cfg = "512x128"
    cfg_rows = [r for r in rows if r['config'] == rep_cfg]
    if not cfg_rows:
        print("  WARNING: no data for representative config 512x128")
        return

    tw, th = get_tile_dims(rep_cfg)
    R, C = 512, 128
    dpes_per_rep = (3 * math.ceil(128 / R) + 4) * math.ceil(128 / C)  # 7

    # Baseline f0
    baseline_rows = [r for r in cfg_rows
                     if abs(r['dsp_ratio'] - 0.2) < 0.01
                     and abs(r['clb_ratio'] - 0.0) < 0.01]
    if not baseline_rows:
        print("  WARNING: no baseline (d=20%, c=0%) for 512x128")
        return
    f0 = baseline_rows[0]['fmax_mhz']
    p_bram = 7  # empirically calibrated

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Collect all points sorted by available DPE count
    points = []
    for r in cfg_rows:
        d_pct = int(round(r['dsp_ratio'] * 100))
        c_pct = int(round(r['clb_ratio'] * 100))
        wc = int(r.get('wc_count', 0))  # actual wc from VTR
        # Estimate available DPEs from total_dpes (= P × dpes_per_rep)
        total_dpes_avail = int(r['total_dpes'])  # this is what was placed
        p = int(r['P'])
        fn = r['fmax_mhz']
        p_dpe_max = total_dpes_avail // dpes_per_rep if dpes_per_rep > 0 else 0
        # But we need the available DPEs, not just what was placed
        # Recalculate from (d%, c%)
        import sys
        sys.path.insert(0, 'nl_dpe')
        from gen_arch_xml import (BASELINE_DSP_STARTX, BASELINE_DSP_REPEATX,
                                  BASELINE_BRAM_STARTX, BASELINE_BRAM_REPEATX,
                                  _baseline_col_positions)

        def _count_wc(dr, cr):
            interior_w = 120 - 2; interior_h = 120 - 2
            tiles_per_col = interior_h // th
            dsp_pos = _baseline_col_positions(BASELINE_DSP_STARTX, BASELINE_DSP_REPEATX, 120)
            bram_pos = _baseline_col_positions(BASELINE_BRAM_STARTX, BASELINE_BRAM_REPEATX, 120)
            n_rm = min(len(dsp_pos), max(0, round(dr * len(dsp_pos))))
            dsp_rm = sorted(dsp_pos[-n_rm:]) if n_rm > 0 else []
            dsp_kept = sorted(set(dsp_pos) - set(dsp_rm))
            wc_clb_rx = None
            if cr > 0:
                n = max(1, round(cr * interior_w / tw))
                wc_clb_rx = max(tw + 1, interior_w // n)
            occ = set(bram_pos) | set(dsp_kept)
            wc_t = 0
            for pos in dsp_rm:
                if all(1 <= pos+dx <= interior_w and (pos+dx) not in occ for dx in range(tw)):
                    wc_t += tiles_per_col
                    for dx in range(tw): occ.add(pos+dx)
            if wc_clb_rx:
                x = BASELINE_DSP_STARTX
                while x <= interior_w:
                    if all(1 <= x+dx <= interior_w and (x+dx) not in occ for dx in range(tw)):
                        wc_t += tiles_per_col
                        for dx in range(tw): occ.add(x+dx)
                    x += wc_clb_rx
            return wc_t

        avail_dpes = _count_wc(r['dsp_ratio'], r['clb_ratio'])
        p_dpe = avail_dpes // dpes_per_rep

        points.append({
            'avail_dpes': avail_dpes,
            'p_dpe': p_dpe,
            'p_actual': p,
            'fn': fn,
            'tput_actual': p * fn,
            'tput_dpe_ceil': p_dpe * f0,
            'tput_bram_ceil': p_bram * f0,
            'd_pct': d_pct, 'c_pct': c_pct,
        })

    # Sort by available DPEs
    points.sort(key=lambda p: p['avail_dpes'])

    xs = [p['avail_dpes'] for p in points]
    tput_actual = [p['tput_actual'] for p in points]
    tput_dpe = [p['tput_dpe_ceil'] for p in points]
    tput_bram = [p['tput_bram_ceil'] for p in points]

    # Plot
    ax.plot(xs, tput_dpe, 'b--', linewidth=1.5, alpha=0.7,
            label=f'DPE ceiling (P_dpe × f₀)', zorder=2)
    ax.axhline(y=p_bram * f0, color='r', linewidth=2, linestyle='-',
               alpha=0.8, label=f'BRAM ceiling (P={p_bram} × f₀={f0:.0f})',
               zorder=3)
    ax.scatter(xs, tput_actual, c='#0891B2', s=40, edgecolors='black',
               linewidth=0.5, zorder=4, label='Actual throughput (P × fn)')

    # Shade the gap between DPE ceiling and actual (wasted DPE potential)
    ax.fill_between(xs, tput_actual, tput_dpe, alpha=0.1, color='blue',
                    label='Wasted DPE capacity (BRAM bottleneck)')

    # Annotate the BRAM wall
    ax.annotate('BRAM limits P to 7\n(64 BRAMs/replica)',
                xy=(max(xs) * 0.7, p_bram * f0),
                xytext=(max(xs) * 0.5, p_bram * f0 * 1.35),
                fontsize=9, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('Available DPE Tiles', fontsize=11)
    ax.set_ylabel('Throughput (P × Fmax, MHz·replicas)', fontsize=11)
    ax.set_title(f'Attention Head Throughput Ceiling ({rep_cfg})\n'
                 f'BRAM bottleneck caps parallelism at P={p_bram}',
                 fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

    out = RESULTS_DIR / "round2_attention_ceiling.pdf"
    fig.savefig(out, )
    print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(RESULTS_DIR.parent.parent / "nl_dpe"))

    print("Loading attention DSE results...")
    rows = load_csv()
    print(f"  {len(rows)} rows loaded")

    print("Generating scalability heatmap...")
    plot_scalability(rows)

    print("Generating per-config Pareto front...")
    plot_pareto(rows)

    print("Generating merged Pareto front...")
    plot_merged_pareto(rows)

    print("Generating throughput ceiling plot...")
    plot_throughput_ceiling(rows)

    print("Done.")
