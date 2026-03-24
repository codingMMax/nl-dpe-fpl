#!/usr/bin/env python3
"""FlexScore Pareto Front (v3 — proportional budget).

X = Non-DL Performance Degradation (1 - FlexScore), lower = better
Y = Geomean Effective Latency (ns/inf), lower = better
Color = DPE Area %
Bottom-left = ideal.

Pareto: minimize both. 1×2 layout: NL-DPE Group | AL-like Group.
"""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path(__file__).resolve().parent
DL_CSV = RESULTS_DIR / "flexscore_dl_gemv_results.csv"
SUMMARY_CSV = RESULTS_DIR / "flexscore_summary.csv"

CONFIGS = ["512x128", "1024x128", "1024x64", "1024x256", "512x256"]
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
    ("NL-DPE Group", ["512x128", "1024x128", "1024x64"]),
    ("AL-like Group", ["1024x256", "512x256"]),
]


def compute_pareto(points):
    """Pareto: minimize X (FlexScore loss), minimize Y (latency)."""
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    pareto = []
    best_y = float("inf")
    for pt in sorted_pts:
        if pt[1] < best_y:
            pareto.append(pt)
            best_y = pt[1]
    return pareto


def find_knee_point(pareto):
    if len(pareto) <= 2:
        return 0
    x0, y0 = pareto[0][0], pareto[0][1]
    x1, y1 = pareto[-1][0], pareto[-1][1]
    dx, dy = x1 - x0, y1 - y0
    ll = math.sqrt(dx**2 + dy**2)
    if ll == 0:
        return 0
    best_d, best_i = -1, 0
    for i, pt in enumerate(pareto):
        d = abs(dy*pt[0] - dx*pt[1] + x1*y0 - y1*x0) / ll
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def main():
    # Load FlexScore summary: (tg, budget) -> flexscore
    fs_lookup = {}
    with open(SUMMARY_CSV) as f:
        for r in csv.DictReader(f):
            fs_lookup[(r["tg"], int(r["budget_pct"]))] = float(r["flexscore"])

    # Load DL results: (config, budget, workload) -> eff_latency
    by_point = defaultdict(dict)
    area_map = {}
    with open(DL_CSV) as f:
        for r in csv.DictReader(f):
            if r["status"].startswith("OK"):
                key = (r["config"], int(r["budget_pct"]))
                by_point[key][r["workload"]] = float(r["eff_latency_ns"])
                area_map[key] = float(r["dpe_area_pct"])

    # Points: (fs_loss, latency, area%, budget, cfg, flexscore)
    cfg_points = defaultdict(list)
    for key, wl_lats in by_point.items():
        cfg, budget = key
        tg = CONFIG_TG.get(cfg)
        if not tg or (tg, budget) not in fs_lookup:
            continue
        vals = [wl_lats[wl] for wl in DL_WORKLOADS if wl in wl_lats]
        valid = [v for v in vals if 0 < v < 1e5]
        if len(valid) < len(DL_WORKLOADS):
            continue
        gm_lat = math.exp(sum(math.log(v) for v in valid) / len(valid))
        fs = fs_lookup[(tg, budget)]
        fs_loss = 1.0 - fs
        area = area_map.get(key, 0)
        cfg_points[cfg].append((fs_loss, gm_lat, area, budget, cfg, fs))

    all_areas = [p[2] for pts in cfg_points.values() for p in pts]
    cmap_area = cm.YlOrRd
    norm_area = mcolors.Normalize(vmin=0, vmax=max(all_areas)*1.1 if all_areas else 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.22, top=0.88)
    fig.suptitle("DPE Area Recommendation: Latency vs Flexibility Loss (60\u00d760 grid)\n"
                 "Color = DPE Area % (lighter = less, darker = more)",
                 fontsize=11, fontweight="bold", y=0.97)

    for gi, (group_name, cfgs) in enumerate(GROUPS):
        ax = axes[gi]
        all_pts = []

        for cfg in cfgs:
            marker = CONFIG_MARKERS[cfg]
            label = CONFIG_LABELS[cfg]
            cfg_color = CONFIG_COLORS[cfg]
            pts = cfg_points.get(cfg, [])
            for p in pts:
                ax.scatter(p[0], p[1], c=[cmap_area(norm_area(p[2]))],
                           marker=marker, s=50, alpha=0.45,
                           edgecolors=cfg_color, linewidths=0.6, zorder=2)
            ax.scatter([], [], c=cfg_color, marker=marker, s=40,
                       edgecolors=cfg_color, linewidths=1.0, label=label)
            all_pts.extend(pts)

        if not all_pts:
            continue

        pareto_raw = compute_pareto([(p[0], p[1], p) for p in all_pts])
        pareto_pts = [p[2] for p in pareto_raw]

        for p in pareto_pts:
            m = CONFIG_MARKERS[p[4]]
            ax.scatter(p[0], p[1], c=[cmap_area(norm_area(p[2]))],
                       marker=m, s=120, edgecolors="black", linewidths=1.8, zorder=5)

        gpx = [p[0] for p in pareto_pts]
        gpy = [p[1] for p in pareto_pts]
        ax.plot(gpx, gpy, color="black", linewidth=2.5, linestyle="-",
                alpha=0.7, zorder=4, label="Pareto front")

        knee_idx = find_knee_point([(p[0], p[1]) for p in pareto_pts])
        knee = pareto_pts[knee_idx]
        kb = knee[3]
        kcfg = CONFIG_LABELS[knee[4]]
        ax.scatter([knee[0]], [knee[1]], color="gold", s=350,
                   edgecolors="red", linewidths=3, zorder=7, marker="*")
        ax.annotate(f"Balanced: {kcfg}\nBudget={kb}%\n"
                    f"FPGA Area={knee[2]:.0f}%",
                    xy=(knee[0], knee[1]),
                    xytext=(0.95, 0.55),
                    textcoords="axes fraction",
                    fontsize=11, fontweight="bold", color="#B91C1C",
                    ha="right", va="center",
                    arrowprops=dict(arrowstyle="-|>", color="#B91C1C", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2",
                              edgecolor="#B91C1C", alpha=0.95), zorder=8)

        print(f"\n  {group_name} -- Pareto ({len(pareto_pts)} pts):")
        print(f"  {'Config':>12} {'Budget':>7} {'Loss%':>6} {'Lat':>8} {'Area%':>6} {'FS':>6}")
        for p in pareto_pts:
            b = p[3]
            fs = p[5]
            star = " *" if b == kb else ""
            print(f"  {CONFIG_LABELS[p[4]]:>12} {b:>6}% {p[0]*100:>5.1f}% "
                  f"{p[1]:>8.2f} {p[2]:>6.1f} {fs:>6.2f}{star}")

        ax.set_xlabel("Non-DL Performance Degradation (1 \u2212 FlexScore)", fontsize=10)
        if gi == 0:
            ax.set_ylabel("Geomean Eff. Latency (ns/inf, lower \u2192 better)", fontsize=10)
        ax.set_title(group_name, fontweight="bold", fontsize=10)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.15)
        ax.set_xlim(left=-0.03)
        ax.set_ylim(bottom=0)
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    sm = cm.ScalarMappable(cmap=cmap_area, norm=norm_area)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("DPE Area (% of FPGA)", fontsize=9)

    out = RESULTS_DIR / "round2_flexscore_pareto.pdf"
    fig.savefig(out)
    print(f"\n  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
