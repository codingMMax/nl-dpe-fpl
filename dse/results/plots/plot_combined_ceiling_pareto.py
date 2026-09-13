#!/usr/bin/env python3
"""Combined Throughput Ceiling + Pareto Front.

One figure per workload type (FC, Attention), each with 2 subplots:
  Left:  NL-DPE Group (512×128, 1024×128, 1024×64)
  Right: AL-like Group (1024×256, 512×256)

X = DPE Area (% of FPGA)
Y = Throughput (inferences/ns)

Each config is a colored curve. The cross-config upper envelope is the
Pareto front (bold dashed black). Gap between actual and ideal = soft ceiling.
"""

import csv
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = RESULTS_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR / "nl_dpe"))
from area_power import dpe_specs

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

CONFIGS = ["512x128", "1024x128", "1024x64", "1024x256", "512x256"]
CONFIG_SHORT = {
    "512x128": "512×128", "1024x128": "1024×128", "1024x64": "1024×64",
    "1024x256": "1024×256", "512x256": "512×256",
}
CONFIG_COLORS = {
    "512x128":  "#2563EB",
    "1024x128": "#DC2626",
    "1024x64":  "#059669",
    "1024x256": "#D97706",
    "512x256":  "#7C3AED",
}
CONFIG_MARKERS = {
    "512x128": "o", "1024x128": "s", "1024x64": "^",
    "1024x256": "D", "512x256": "v",
}

GRID_CELLS = 120 * 120
FC_WORKLOADS = ["fc_512_128", "fc_512_512", "fc_2048_256"]

GROUPS = [
    ("NL-DPE Group", ["512x128", "1024x128", "1024x64"]),
    ("AL-like Group", ["1024x256", "512x256"]),
]


def get_tile_dims(cfg_str):
    R, C = map(int, cfg_str.split("x"))
    s = dpe_specs(R, C)
    return s['tile_width'], s['tile_height']


def compute_upper_envelope(all_points):
    """Upper envelope: for each x, keep the highest y. Then extract the
    monotonically increasing front (Pareto-optimal for max throughput, min area)."""
    if not all_points:
        return []
    # Sort by x
    sorted_pts = sorted(all_points, key=lambda p: p[0])
    # Build envelope: sweep left-to-right, keep running max of y
    # But Pareto for maximize-y minimize-x: sweep right-to-left
    # A point is on the Pareto front if no other has lower x AND higher y
    # Equivalently: sweep by x ascending, keep if y > best_y_at_lower_x
    # Wait — we want max throughput. The "Pareto" is: for a given area budget,
    # what's the best throughput achievable? = upper envelope.
    # Group by approximate x, take max y per group
    from collections import OrderedDict
    buckets = OrderedDict()
    for x, y, meta in sorted_pts:
        xr = round(x, 1)
        if xr not in buckets or y > buckets[xr][1]:
            buckets[xr] = (x, y, meta)
    envelope_pts = list(buckets.values())
    # Now extract the monotonically increasing subsequence (Pareto front)
    pareto = []
    best_y = -float('inf')
    for pt in envelope_pts:
        if pt[1] > best_y:
            pareto.append(pt)
            best_y = pt[1]
    return pareto


def find_knee_point(pareto):
    """Knee = max perpendicular distance from line connecting endpoints."""
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


# ═════════════════════════════════════════════════════════════════════════
# FC Combined Ceiling/Pareto
# ═════════════════════════════════════════════════════════════════════════

def plot_fc_combined():
    csv_path = RESULTS_DIR / "round2_fc" / "round2_full_results.csv"
    rows = list(csv.DictReader(open(csv_path)))

    # Group by (config, d%, c%) → per-workload fmax, P, total_dpes
    grouped = defaultdict(lambda: {"fmax": {}, "P": None, "total_dpes": None})
    for r in rows:
        cfg = r['config']
        d = int(round(float(r['dsp_ratio']) * 100))
        c = int(round(float(r['clb_ratio']) * 100))
        wl = r['workload']
        key = (cfg, d, c)
        grouped[key]["fmax"][wl] = float(r['fmax_mhz'])
        grouped[key]["P"] = int(r['P'])
        grouped[key]["total_dpes"] = int(r['total_dpes'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.07, right=0.97, wspace=0.22, top=0.90, bottom=0.12)
    fig.suptitle("FC Workloads: Throughput Ceiling & Cross-Config Pareto",
                 fontsize=12, fontweight="bold", y=0.97)

    for gi, (group_name, cfgs) in enumerate(GROUPS):
        ax = axes[gi]
        all_envelope_pts = []  # for cross-config Pareto

        for cfg in cfgs:
            color = CONFIG_COLORS[cfg]
            marker = CONFIG_MARKERS[cfg]
            tw, th = get_tile_dims(cfg)

            # Baseline f0 (d=20%, c=0%)
            base = grouped.get((cfg, 20, 0))
            if not base or len(base["fmax"]) < len(FC_WORKLOADS):
                continue
            f0 = base["fmax"]

            # Collect points
            points = []
            for (c, d, cp), info in grouped.items():
                if c != cfg or len(info["fmax"]) < len(FC_WORKLOADS):
                    continue
                P = info["P"]
                area_pct = info["total_dpes"] * tw * th / GRID_CELLS * 100

                # Geomean actual throughput (inferences/ns)
                tputs = [P * info["fmax"][wl] / 1e3 for wl in FC_WORKLOADS]
                gm_actual = math.exp(sum(math.log(t) for t in tputs) / len(tputs))

                # Geomean ideal throughput
                ideal_tputs = [P * f0[wl] / 1e3 for wl in FC_WORKLOADS]
                gm_ideal = math.exp(sum(math.log(t) for t in ideal_tputs) / len(ideal_tputs))

                points.append({
                    'area': area_pct, 'actual': gm_actual, 'ideal': gm_ideal,
                    'P': P, 'd': d, 'c': cp,
                    'eff': gm_actual / gm_ideal if gm_ideal > 0 else 0,
                })

            points.sort(key=lambda p: p['area'])

            # Deduplicate by area (keep best actual)
            seen = {}
            for pt in points:
                a = round(pt['area'], 1)
                if a not in seen or pt['actual'] > seen[a]['actual']:
                    seen[a] = pt
            points = sorted(seen.values(), key=lambda p: p['area'])

            if not points:
                continue

            xs = [p['area'] for p in points]
            actuals = [p['actual'] for p in points]
            ideals = [p['ideal'] for p in points]

            # Ideal line (gray dashed, only draw once per config — they overlap)
            ax.plot(xs, ideals, color='#CCCCCC', linewidth=0.8, linestyle='--',
                    alpha=0.5, zorder=1)

            # Actual throughput curve
            ax.plot(xs, actuals, color=color, linewidth=1.8, alpha=0.85, zorder=3)
            ax.scatter(xs, actuals, color=color, marker=marker, s=20,
                       edgecolors='white', linewidths=0.4, zorder=4,
                       label=CONFIG_SHORT[cfg])

            # Shade gap between actual and ideal
            ax.fill_between(xs, actuals, ideals, alpha=0.06, color=color, zorder=0)

            # Collect for cross-config envelope
            for pt in points:
                all_envelope_pts.append((pt['area'], pt['actual'], (cfg, pt['d'], pt['c'])))

        # Cross-config Pareto (upper envelope)
        pareto = compute_upper_envelope(all_envelope_pts)
        if pareto:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.plot(px, py, color='black', linewidth=2.5, linestyle='--',
                    alpha=0.5, zorder=5, label='Cross-config Pareto')

            # Knee point
            knee_idx = find_knee_point(pareto)
            knee = pareto[knee_idx]
            knee_cfg, kd, kc = knee[2]
            ax.scatter([knee[0]], [knee[1]], color='black', s=180,
                       edgecolors='red', linewidths=2, zorder=7, marker='*')
            # Annotate knee
            ax.annotate(
                f"{CONFIG_SHORT[knee_cfg]}\nd={kd}% c={kc}%",
                xy=(knee[0], knee[1]),
                xytext=(knee[0] + max(px) * 0.08, knee[1] * 0.82),
                fontsize=7.5, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                zorder=8,
            )

            # Print Pareto table
            print(f"\n  FC {group_name} — Cross-config Pareto ({len(pareto)} pts):")
            print(f"  {'Config':>12} {'d%':>4} {'c%':>4} {'Area%':>7} {'Tput':>8}")
            for area, tput, (cfg, d, c) in pareto:
                star = " ★" if (cfg, d, c) == (knee_cfg, kd, kc) else ""
                print(f"  {CONFIG_SHORT[cfg]:>12} {d:>4} {c:>4} {area:>7.1f} {tput:>8.3f}{star}")

        ax.set_xlabel("DPE Area (% of FPGA)")
        ax.set_ylabel("Throughput (inferences/ns)")
        ax.set_title(group_name, fontweight='bold')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.15, linewidth=0.5)

    out = RESULTS_DIR / "fc_ceiling_pareto.pdf"
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Attention Combined Ceiling/Pareto
# ═════════════════════════════════════════════════════════════════════════

def plot_attention_combined():
    csv_path = RESULTS_DIR / "round2_attention" / "round2_attention_results.csv"
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ['dsp_ratio', 'clb_ratio', 'P', 'fmax_mhz', 'total_dpes',
                       'fmax_baseline_mhz']:
                if k in r:
                    try: r[k] = float(r[k])
                    except: r[k] = 0.0
            rows.append(r)

    # Group by (config, d%, c%)
    grouped = defaultdict(lambda: {"fmax": 0, "P": 0, "total_dpes": 0})
    for r in rows:
        cfg = r['config']
        d = int(round(r['dsp_ratio'] * 100))
        c = int(round(r['clb_ratio'] * 100))
        key = (cfg, d, c)
        grouped[key]["fmax"] = r['fmax_mhz']
        grouped[key]["P"] = int(r['P'])
        grouped[key]["total_dpes"] = int(r['total_dpes'])
        grouped[key]["fmax_baseline"] = r['fmax_baseline_mhz']

    # BRAM cap
    P_BRAM = 7

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=0.07, right=0.97, wspace=0.22, top=0.90, bottom=0.12)
    fig.suptitle("Attention Head: Throughput Ceiling & Cross-Config Pareto",
                 fontsize=12, fontweight="bold", y=0.97)

    for gi, (group_name, cfgs) in enumerate(GROUPS):
        ax = axes[gi]
        all_envelope_pts = []

        for cfg in cfgs:
            color = CONFIG_COLORS[cfg]
            marker = CONFIG_MARKERS[cfg]
            tw, th = get_tile_dims(cfg)

            # Baseline (d=20%, c=0%)
            base = grouped.get((cfg, 20, 0))
            if not base or base["fmax"] <= 0:
                continue
            f0 = base["fmax"]

            points = []
            for (c, d, cp), info in grouped.items():
                if c != cfg or info["fmax"] <= 0:
                    continue
                P = info["P"]
                area_pct = info["total_dpes"] * tw * th / GRID_CELLS * 100

                actual = P * info["fmax"] / 1e3  # inferences/ns
                ideal = P * f0 / 1e3

                points.append({
                    'area': area_pct, 'actual': actual, 'ideal': ideal,
                    'P': P, 'd': d, 'c': cp,
                    'eff': actual / ideal if ideal > 0 else 0,
                })

            points.sort(key=lambda p: p['area'])

            # Deduplicate
            seen = {}
            for pt in points:
                a = round(pt['area'], 1)
                if a not in seen or pt['actual'] > seen[a]['actual']:
                    seen[a] = pt
            points = sorted(seen.values(), key=lambda p: p['area'])

            if not points:
                continue

            xs = [p['area'] for p in points]
            actuals = [p['actual'] for p in points]
            ideals = [p['ideal'] for p in points]

            # Ideal line
            ax.plot(xs, ideals, color='#CCCCCC', linewidth=0.8, linestyle='--',
                    alpha=0.5, zorder=1)

            # Actual
            ax.plot(xs, actuals, color=color, linewidth=1.8, alpha=0.85, zorder=3)
            ax.scatter(xs, actuals, color=color, marker=marker, s=20,
                       edgecolors='white', linewidths=0.4, zorder=4,
                       label=CONFIG_SHORT[cfg])

            ax.fill_between(xs, actuals, ideals, alpha=0.06, color=color, zorder=0)

            for pt in points:
                all_envelope_pts.append((pt['area'], pt['actual'], (cfg, pt['d'], pt['c'])))

        # BRAM ceiling line
        # Find max f0 across configs in this group for reference
        max_bram_tput = 0
        for cfg in cfgs:
            base = grouped.get((cfg, 20, 0))
            if base and base["fmax"] > 0:
                bram_tput = P_BRAM * base["fmax"] / 1e3
                max_bram_tput = max(max_bram_tput, bram_tput)

        if max_bram_tput > 0:
            ax.axhline(y=max_bram_tput, color='red', linewidth=1.5, linestyle='-',
                       alpha=0.6, zorder=2)
            ax.annotate(f'BRAM ceiling (P={P_BRAM})',
                        xy=(ax.get_xlim()[1] * 0.5 if ax.get_xlim()[1] > 0 else 20,
                            max_bram_tput),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=7, color='red', fontstyle='italic',
                        ha='center')

        # Cross-config Pareto
        pareto = compute_upper_envelope(all_envelope_pts)
        if pareto:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.plot(px, py, color='black', linewidth=2.5, linestyle='--',
                    alpha=0.5, zorder=5, label='Cross-config Pareto')

            knee_idx = find_knee_point(pareto)
            knee = pareto[knee_idx]
            knee_cfg, kd, kc = knee[2]
            ax.scatter([knee[0]], [knee[1]], color='black', s=180,
                       edgecolors='red', linewidths=2, zorder=7, marker='*')
            ax.annotate(
                f"{CONFIG_SHORT[knee_cfg]}\nd={kd}% c={kc}%",
                xy=(knee[0], knee[1]),
                xytext=(knee[0] + max(px) * 0.08, knee[1] * 0.78),
                fontsize=7.5, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                zorder=8,
            )

            print(f"\n  Attention {group_name} — Cross-config Pareto ({len(pareto)} pts):")
            print(f"  {'Config':>12} {'d%':>4} {'c%':>4} {'Area%':>7} {'Tput':>8}")
            for area, tput, (cfg, d, c) in pareto:
                star = " ★" if (cfg, d, c) == (knee_cfg, kd, kc) else ""
                print(f"  {CONFIG_SHORT[cfg]:>12} {d:>4} {c:>4} {area:>7.1f} {tput:>8.3f}{star}")

        ax.set_xlabel("DPE Area (% of FPGA)")
        ax.set_ylabel("Throughput (inferences/ns)")
        ax.set_title(group_name, fontweight='bold')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.15, linewidth=0.5)

    out = RESULTS_DIR / "attention_ceiling_pareto.pdf"
    fig.savefig(out)
    print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating FC combined ceiling/Pareto...")
    plot_fc_combined()

    print("\nGenerating Attention combined ceiling/Pareto...")
    plot_attention_combined()

    print("\nDone.")
