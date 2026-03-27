#!/usr/bin/env python3
"""FlexScore Pareto Front — Broken X-axis layout.

X = Non-DL Performance Degradation (1 - FlexScore), lower = better
Y = Geomean Effective Latency (ns/inf), lower = better
Color = DPE Area %
Bottom-left = ideal.

Pareto: minimize both. 1×2 layout: NL-DPE Group | AL-like Group.
Broken X-axis: 0-10% linear (detailed) // 15-85% linear (tail).

Output: round2_flexscore_pareto.pdf
"""
import csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MultipleLocator
from collections import defaultdict
from pathlib import Path
import numpy as np

from style_constants import apply_style, ANNOT_FONTSIZE, ANNOT_FONTWEIGHT
apply_style()

RESULTS_DIR = Path(__file__).resolve().parent
DL_CSV = RESULTS_DIR / "flexscore_dl_gemv_results.csv"
SUMMARY_CSV = RESULTS_DIR / "flexscore_summary.csv"

CONFIG_LABELS = {
    "512x128": "512\u00d7128", "1024x128": "1024\u00d7128", "1024x64": "1024\u00d764",
    "1024x256": "1024\u00d7256", "512x256": "512\u00d7256",
}
CONFIG_TG = {
    "512x128": "tw3", "1024x128": "tw3", "1024x64": "tw3",
    "1024x256": "tw7", "512x256": "tw4",
}
CONFIG_COLORS = {
    "512x128": "#2563EB", "1024x128": "#DC2626", "1024x64": "#059669",
    "1024x256": "#D97706", "512x256": "#7C3AED",
}
CONFIG_MARKERS = {
    "512x128": "o", "1024x128": "s", "1024x64": "^",
    "1024x256": "D", "512x256": "v",
}
DL_WORKLOADS = ["fc_512_128", "fc_512_512", "fc_2048_256"]
GROUPS = [
    ("", ["512x128", "1024x128", "1024x64"]),
    ("", ["1024x256", "512x256"]),
]


def compute_pareto(points):
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    pareto, best_y = [], float("inf")
    for pt in sorted_pts:
        if pt[1] < best_y:
            pareto.append(pt)
            best_y = pt[1]
    return pareto


def find_knee_point(pareto):
    if len(pareto) <= 2: return 0
    x0, y0 = pareto[0][0], pareto[0][1]
    x1, y1 = pareto[-1][0], pareto[-1][1]
    dx, dy = x1 - x0, y1 - y0
    ll = math.sqrt(dx**2 + dy**2)
    if ll == 0: return 0
    best_d, best_i = -1, 0
    for i, pt in enumerate(pareto):
        d = abs(dy*pt[0] - dx*pt[1] + x1*y0 - y1*x0) / ll
        if d > best_d: best_d, best_i = d, i
    return best_i


def draw_break_marks(ax_left, ax_right, d=0.015):
    """Draw diagonal break marks on the right edge of ax_left and left edge of ax_right."""
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, linewidth=0.8)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)


def plot_broken_group(fig, gs_slot, group_name, cfgs, cfg_points, cmap_area, norm_area,
                      show_ylabel=True, break_at=10, right_max=85, gi=0):
    """Create a broken-axis subplot pair for one group."""
    import matplotlib.gridspec as gridspec
    inner_gs = gs_slot.subgridspec(1, 2, width_ratios=[2.5, 1], wspace=0.06)
    ax_left = fig.add_subplot(inner_gs[0])
    ax_right = fig.add_subplot(inner_gs[1])

    all_pts = []
    for cfg in cfgs:
        marker = CONFIG_MARKERS[cfg]
        label = CONFIG_LABELS[cfg]
        cfg_color = CONFIG_COLORS[cfg]
        pts = cfg_points.get(cfg, [])
        for p in pts:
            x_pct = p[0] * 100
            for ax in [ax_left, ax_right]:
                ax.scatter(x_pct, p[1], c=[cmap_area(norm_area(p[2]))],
                           marker=marker, s=20, alpha=0.4,
                           edgecolors=cfg_color, linewidths=0.5, zorder=2)
        ax_left.scatter([], [], c=cfg_color, marker=marker, s=35,
                        edgecolors=cfg_color, linewidths=1.0, label=label)
        all_pts.extend(pts)

    if not all_pts:
        return

    pareto_raw = compute_pareto([(p[0], p[1], p) for p in all_pts])
    pareto_pts = [p[2] for p in pareto_raw]

    # Pareto points — slightly larger + bolder edge than background
    for p in pareto_pts:
        m = CONFIG_MARKERS[p[4]]
        cfg_color = CONFIG_COLORS[p[4]]
        x_pct = p[0] * 100
        for ax in [ax_left, ax_right]:
            ax.scatter(x_pct, p[1], c=[cmap_area(norm_area(p[2]))],
                       marker=m, s=40, edgecolors=cfg_color, linewidths=1.2, zorder=5)

    # Pareto line on both axes
    gpx = [p[0] * 100 for p in pareto_pts]
    gpy = [p[1] for p in pareto_pts]
    for ax in [ax_left, ax_right]:
        ax.plot(gpx, gpy, color="black", linewidth=2.0, linestyle="-",
                alpha=0.7, zorder=4)
    ax_left.plot([], [], color="black", linewidth=2.0, linestyle="-",
                 alpha=0.7, label="Pareto front")

    # Knee point — subtle highlight + annotation box
    knee_idx = find_knee_point([(p[0], p[1]) for p in pareto_pts])
    knee = pareto_pts[knee_idx]
    kcfg = CONFIG_LABELS[knee[4]]
    kx = knee[0] * 100
    for ax in [ax_left, ax_right]:
        ax.scatter([kx], [knee[1]], s=120, facecolors="none",
                   edgecolors="red", linewidths=1.5, zorder=6)
    # Group 1: center-right; Group 2: below pareto front
    if gi == 0:
        tx, ty = 0.92, 0.50
    else:
        tx, ty = 0.92, 0.15
    ax_left.annotate(f"Recommended: {kcfg}\nFPGA Area Cost = {knee[2]:.0f}%",
                     xy=(kx, knee[1]),
                     xytext=(tx, ty), textcoords="axes fraction",
                     fontsize=ANNOT_FONTSIZE, fontweight=ANNOT_FONTWEIGHT, color="#B91C1C",
                     ha="right", va="center",
                     arrowprops=dict(arrowstyle="-|>", color="#B91C1C", lw=1.0),
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2",
                               edgecolor="#B91C1C", alpha=0.95), zorder=8)

    print(f"\n  {group_name} -- Pareto ({len(pareto_pts)} pts):")
    print(f"  {'Config':>12} {'Budget':>7} {'Loss%':>6} {'Lat':>8} {'Area%':>6} {'FS':>6}")
    for p in pareto_pts:
        b = p[3]
        fs = p[5]
        star = " *" if b == knee[3] else ""
        print(f"  {CONFIG_LABELS[p[4]]:>12} {b:>6}% {p[0]*100:>5.1f}% "
              f"{p[1]:>8.2f} {p[2]:>6.1f} {fs:>6.2f}{star}")

    # --- Axis formatting ---
    ax_left.set_xlim(-1.5, break_at + 2)
    ax_left.set_xticks([0, 5])
    ax_left.set_xticklabels(["0%", "5%"])

    ax_right.set_xlim(break_at + 5, right_max)
    ax_right.set_xticks([30, 55, 80])
    ax_right.set_xticklabels(["30%", "55%", "80%"])

    all_y = [p[1] for p in all_pts]
    y_max = max(all_y) * 1.15
    ax_left.set_ylim(0, y_max)
    ax_right.set_ylim(0, y_max)

    ax_right.set_yticklabels([])
    ax_right.tick_params(left=False)
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    draw_break_marks(ax_left, ax_right)

    if show_ylabel:
        ax_left.set_ylabel("Geomean Latency (ns/inf)")
    ax_left.set_title(group_name, fontweight="bold")
    ax_left.legend(fontsize=7, loc="upper right")
    ax_left.grid(True, alpha=0.15)
    ax_right.grid(True, alpha=0.15)

    fig.text((ax_left.get_position().x0 + ax_right.get_position().x1) / 2,
             0.02, "Non-DL Perf. Degradation (1 \u2212 FlexScore)",
             ha="center")


def main():
    fs_lookup = {}
    with open(SUMMARY_CSV) as f:
        for r in csv.DictReader(f):
            fs_lookup[(r["tg"], int(r["budget_pct"]))] = float(r["flexscore"])

    by_point = defaultdict(dict)
    area_map = {}
    with open(DL_CSV) as f:
        for r in csv.DictReader(f):
            if r["status"].startswith("OK"):
                key = (r["config"], int(r["budget_pct"]))
                by_point[key][r["workload"]] = float(r["eff_latency_ns"])
                area_map[key] = float(r["dpe_area_pct"])

    cfg_points = defaultdict(list)
    for key, wl_lats in by_point.items():
        cfg, budget = key
        tg = CONFIG_TG.get(cfg)
        if not tg or (tg, budget) not in fs_lookup: continue
        vals = [wl_lats[wl] for wl in DL_WORKLOADS if wl in wl_lats]
        valid = [v for v in vals if 0 < v < 1e5]
        if len(valid) < len(DL_WORKLOADS): continue
        gm_lat = math.exp(sum(math.log(v) for v in valid) / len(valid))
        fs = fs_lookup[(tg, budget)]
        cfg_points[cfg].append((1.0 - fs, gm_lat, area_map.get(key, 0), budget, cfg, fs))

    all_areas = [p[2] for pts in cfg_points.values() for p in pts]
    cmap_area = cm.YlOrRd
    norm_area = mcolors.Normalize(vmin=0, vmax=max(all_areas)*1.1 if all_areas else 1)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(7.16, 2.65))
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.18,
                                  left=0.08, right=0.87, top=0.95, bottom=0.15)
    # fig.suptitle("Pareto-Front: Latency vs Flexibility Loss",
    #              fontweight="bold", y=0.97)

    for gi, (group_name, cfgs) in enumerate(GROUPS):
        plot_broken_group(fig, outer_gs[gi], group_name, cfgs, cfg_points,
                          cmap_area, norm_area, show_ylabel=(gi == 0),
                          break_at=10, right_max=85, gi=gi)

    sm = cm.ScalarMappable(cmap=cmap_area, norm=norm_area)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.89, 0.15, 0.015, 0.62])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DPE Area (% of FPGA)")
    cbar.ax.tick_params(labelsize=7)

    out = RESULTS_DIR / "round2_flexscore_pareto.pdf"
    fig.savefig(out)
    print(f"\n  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
