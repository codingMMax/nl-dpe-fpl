#!/usr/bin/env python3
"""Round 2 FlexScore: FC vs Attention Pareto fronts side by side.

Input:  flexscore_dl_results.csv (FC), flexscore_dl_attention_results.csv (Attention),
        flexscore_summary.csv (non-DL FlexScore)
Output: round2_flexscore_pareto_fc_vs_attention.pdf
"""
import csv, math, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "nl_dpe"))
from area_power import dpe_specs

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"serif","font.size":9,"axes.labelsize":10,"axes.titlesize":11,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.05})

RESULTS_DIR = Path(__file__).resolve().parent

CONFIG_COLORS = {
    "512x128": "#2563EB", "1024x128": "#DC2626", "1024x64": "#059669",
    "1024x256": "#D97706", "512x256": "#7C3AED",
    "128x64": "#0891B2", "512x64": "#BE185D", "128x128": "#4338CA",
}
CONFIG_MARKERS = {
    "512x128": "o", "1024x128": "s", "1024x64": "^",
    "1024x256": "D", "512x256": "v",
    "128x64": "P", "512x64": "X", "128x128": "p",
}


def get_tg(cfg):
    R, C = map(int, cfg.split('x'))
    return f"tw{dpe_specs(R, C)['tile_width']}"


def compute_pareto_min(points):
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    pareto, best_y = [], float('inf')
    for pt in sorted_pts:
        if pt[1] < best_y:
            pareto.append(pt); best_y = pt[1]
    return pareto


def find_knee(pareto):
    if len(pareto) <= 2:
        return 0
    x0, y0, x1, y1 = pareto[0][0], pareto[0][1], pareto[-1][0], pareto[-1][1]
    dx, dy = x1 - x0, y1 - y0
    ll = math.sqrt(dx**2 + dy**2)
    if ll == 0:
        return 0
    return max(range(len(pareto)),
               key=lambda i: abs(dy*pareto[i][0] - dx*pareto[i][1] + x1*y0 - y1*x0) / ll)


def main():
    dl_fc = list(csv.DictReader(open(RESULTS_DIR / "flexscore_dl_results.csv")))
    dl_attn = list(csv.DictReader(open(RESULTS_DIR / "flexscore_dl_attention_results.csv")))
    fs = {(r['tg'], int(r['budget_pct'])): float(r['flexscore'])
          for r in csv.DictReader(open(RESULTS_DIR / "flexscore_summary.csv"))}

    # FC grouped
    fc_configs = ["512x128", "1024x128", "1024x64", "1024x256", "512x256"]
    fc_wls = ["fc_512_128", "fc_512_512", "fc_2048_256"]
    fc_grouped = defaultdict(lambda: {"lats": {}, "area": None})
    for r in dl_fc:
        if r['status'] not in ['OK', 'OK (cached)']:
            continue
        key = (r['config'], int(round(float(r['dpe_area_pct']) / 5) * 5))
        fc_grouped[key]["lats"][r['workload']] = float(r['eff_latency_ns'])
        fc_grouped[key]["area"] = float(r['dpe_area_pct'])

    # Attention grouped
    attn_configs = ["128x64", "512x64", "128x128"]
    attn_wls = ["attention_128_64", "attention_256_64", "attention_128_128"]
    attn_grouped = defaultdict(lambda: {"lats": {}, "area": None})
    for r in dl_attn:
        if r['status'] != 'OK':
            continue
        key = (r['config'], int(r['budget_pct']))
        attn_grouped[key]["lats"][r['workload']] = float(r['eff_latency_ns'])
        attn_grouped[key]["area"] = float(r['dpe_area_pct'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.30)

    for ax, title, cfg_list, grouped, wl_list, wl_label in [
        (ax1, "(a) FC Workloads", fc_configs, fc_grouped, fc_wls, "Geomean of 3 FC workloads"),
        (ax2, "(b) Attention Workloads", attn_configs, attn_grouped, attn_wls, "Geomean of 3 attention workloads"),
    ]:
        all_points = []
        for cfg in cfg_list:
            color = CONFIG_COLORS.get(cfg, "#888")
            marker = CONFIG_MARKERS.get(cfg, "o")
            tg = get_tg(cfg)
            points = []
            for (c, b), info in grouped.items():
                if c != cfg or len(info["lats"]) < len(wl_list):
                    continue
                lats = [info["lats"][w] for w in wl_list if w in info["lats"]]
                if not lats or any(l <= 0 or l == float('inf') for l in lats):
                    continue
                gm_lat = math.exp(sum(math.log(l) for l in lats) / len(lats))
                flexscore = fs.get((tg, b))
                if flexscore is None:
                    continue
                points.append((1.0 - flexscore, gm_lat, (b, cfg)))
                all_points.append((1.0 - flexscore, gm_lat, (b, cfg)))
            if not points:
                continue
            cfg_pareto = compute_pareto_min(points)
            cfg_set = set((p[0], p[1]) for p in cfg_pareto)
            dom = [p for p in points if (p[0], p[1]) not in cfg_set]
            ax.scatter([p[0] for p in dom], [p[1] for p in dom],
                       facecolors="none", edgecolors=color, marker=marker,
                       s=25, alpha=0.4, linewidths=0.6, zorder=2)
            ax.plot([p[0] for p in cfg_pareto], [p[1] for p in cfg_pareto],
                    color=color, linewidth=1.2, alpha=0.7, zorder=3)
            ax.scatter([p[0] for p in cfg_pareto], [p[1] for p in cfg_pareto],
                       color=color, marker=marker, s=35, edgecolors="white",
                       linewidths=0.6, zorder=4, label=cfg.replace("x", "\u00d7"))

        if all_points:
            gp = compute_pareto_min(all_points)
            ax.plot([p[0] for p in gp], [p[1] for p in gp],
                    color="black", linewidth=2, linestyle="--", alpha=0.5, zorder=5,
                    label="Cross-config Pareto")
            ki = find_knee(gp)
            knee = gp[ki]
            kb, kcfg = knee[2]
            ax.scatter([knee[0]], [knee[1]], color="black", s=150,
                       edgecolors="red", linewidths=2, zorder=6, marker="*")
            ax.annotate(f"Balanced: {kcfg.replace('x',chr(215))}\nbudget={kb}%",
                        xy=(knee[0], knee[1]),
                        xytext=(knee[0]+0.05, knee[1]*0.85),
                        fontsize=7.5, fontweight="bold", color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=1))

        ax.set_xlabel("Non-DL Degradation (1 \u2212 FlexScore)\nlower = more flexible", fontsize=9)
        ax.set_ylabel("Amortized Time (ns/inf)\nlower = faster", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(left=-0.02)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.1)
        ax.text(0.98, 0.02, wl_label, transform=ax.transAxes, fontsize=7,
                ha='right', va='bottom', color='#888', fontstyle='italic')

    fig.suptitle("Round 2 FlexScore DSE: FC vs Attention Pareto Fronts (60\u00d760 grid)",
                 fontsize=11, fontweight="bold", y=1.02)

    out = RESULTS_DIR / "round2_flexscore_pareto_fc_vs_attention.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
