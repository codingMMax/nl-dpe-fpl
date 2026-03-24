#!/usr/bin/env python3
"""Generate a dummy Pareto front PDF for DSE methodology figure drawing.

Realistic synthetic data matching expected FlexScore DSE v3 results.
Two groups: NL-DPE (3 configs) and AL-like (2 configs).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path(__file__).resolve().parent

CONFIG_LABELS = {
    "512x128": "512\u00d7128", "1024x128": "1024\u00d7128", "1024x64": "1024\u00d764",
    "1024x256": "1024\u00d7256", "512x256": "512\u00d7256",
}
CONFIG_COLORS = {
    "512x128": "#2563EB", "1024x128": "#DC2626", "1024x64": "#059669",
    "1024x256": "#D97706", "512x256": "#7C3AED",
}
CONFIG_MARKERS = {
    "512x128": "o", "1024x128": "s", "1024x64": "^",
    "1024x256": "D", "512x256": "v",
}

GROUPS = [
    ("NL-DPE Group", ["512x128", "1024x128", "1024x64"]),
    ("AL-like Group", ["1024x256", "512x256"]),
]

# Synthetic data: (fs_loss, latency, area%, budget%, config)
# fs_loss = 1 - FlexScore. Lower = better on both axes.
np.random.seed(42)

SYNTHETIC = {
    # NL-DPE group: tw3 configs share FlexScore, differ in latency
    "512x128": [
        # (fs_loss, lat, area%, budget)
        (0.01, 1.8, 5, 10),
        (0.02, 0.55, 9, 20),
        (0.25, 0.35, 23, 30),
        (0.75, 0.28, 33, 40),
        (0.75, 0.24, 37, 50),
    ],
    "1024x128": [
        (0.01, 2.2, 5, 10),
        (0.02, 0.65, 9, 20),
        (0.25, 0.30, 23, 30),
        (0.75, 0.25, 32, 40),
        (0.75, 0.22, 37, 50),
    ],
    "1024x64": [
        (0.01, 3.5, 5, 10),
        (0.02, 1.1, 9, 20),
        (0.25, 0.45, 23, 30),
        (0.75, 0.35, 32, 40),
        (0.75, 0.30, 36, 50),
    ],
    # AL-like group: different tile groups, different FlexScore curves
    "1024x256": [
        (0.02, 1.5, 21, 40),
        (0.02, 1.2, 21, 50),
    ],
    "512x256": [
        (0.01, 0.66, 16, 20),
        (0.25, 0.87, 11, 30),
        (0.75, 0.35, 27, 40),
        (0.75, 0.32, 27, 50),
    ],
}


def compute_pareto(points):
    """Minimize both X and Y."""
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    pareto = []
    best_y = float("inf")
    for pt in sorted_pts:
        if pt[1] < best_y:
            pareto.append(pt)
            best_y = pt[1]
    return pareto


def main():
    all_areas = [p[2] for pts in SYNTHETIC.values() for p in pts]
    cmap_area = cm.YlOrRd
    norm_area = mcolors.Normalize(vmin=0, vmax=max(all_areas) * 1.1)

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
            pts = SYNTHETIC.get(cfg, [])

            # Dominated points (semi-transparent)
            for p in pts:
                ax.scatter(p[0], p[1], c=[cmap_area(norm_area(p[2]))],
                           marker=marker, s=50, alpha=0.45,
                           edgecolors=cfg_color, linewidths=0.6, zorder=2)
            # Legend entry
            ax.scatter([], [], c=cfg_color, marker=marker, s=40,
                       edgecolors=cfg_color, linewidths=1.0, label=label)
            all_pts.extend(pts)

        if not all_pts:
            continue

        # Pareto front
        pareto_raw = compute_pareto([(p[0], p[1], p) for p in all_pts])
        pareto_pts = [p[2] for p in pareto_raw]

        # Pareto points - bold
        for p in pareto_pts:
            cfg = p[4] if len(p) > 4 else "512x128"
            # Find which config this point belongs to
            for c, pts_list in SYNTHETIC.items():
                if p in pts_list:
                    cfg = c
                    break
            m = CONFIG_MARKERS.get(cfg, "o")
            ax.scatter(p[0], p[1], c=[cmap_area(norm_area(p[2]))],
                       marker=m, s=120, edgecolors="black", linewidths=1.8, zorder=5)

        # Pareto front line
        gpx = [p[0] for p in pareto_pts]
        gpy = [p[1] for p in pareto_pts]
        ax.plot(gpx, gpy, color="black", linewidth=2.5, linestyle="-",
                alpha=0.7, zorder=4, label="Pareto front")

        # Knee point (pick the 2nd Pareto point as balanced)
        knee_idx = min(1, len(pareto_pts) - 1)
        knee = pareto_pts[knee_idx]
        # Find config for knee
        knee_cfg = "512x128"
        for c, pts_list in SYNTHETIC.items():
            if knee in pts_list:
                knee_cfg = c
                break
        kb = knee[3]
        kcfg = CONFIG_LABELS[knee_cfg]

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

    out = RESULTS_DIR / "dummy_pareto_front.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
