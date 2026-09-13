#!/usr/bin/env python3
"""DSE Methodology Overview — Top-Conference Style (ISCA/MICRO/ISFPGA).

Two-row flow: Round 1 (crossbar selection) → Round 2 (area budget optimization).
Each row: Design Space → Evaluation → Result.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

RESULTS_DIR = Path(__file__).resolve().parent

# Muted professional palette
PAL = {
    "dpe":       "#059669",  # green - DPE related
    "dpe_light": "#D1FAE5",
    "fpga":      "#2563EB",  # blue - FPGA related
    "fpga_light":"#DBEAFE",
    "nondl":     "#B91C1C",  # muted red - non-DL
    "nondl_light":"#FEE2E2",
    "result":    "#374151",  # dark gray - results
    "result_light":"#F3F4F6",
    "arrow":     "#6B7280",  # medium gray
    "border":    "#D1D5DB",  # light gray
    "text":      "#111827",  # near-black
    "subtext":   "#6B7280",  # gray
    "bg_r1":     "#F0F4FF",  # very light blue
    "bg_r2":     "#F0FDF4",  # very light green
    "highlight": "#059669",  # green highlight
    "white":     "#FFFFFF",
}


def rbox(ax, x, y, w, h, fc="#FFF", ec=None, radius=0.04):
    """Rounded rectangle patch."""
    ec = ec or PAL["border"]
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=3)
    ax.add_patch(p)
    return p


def label(ax, x, y, text, fontsize=8, color=None, bold=False, ha="center",
          va="center", family="serif"):
    c = color or PAL["text"]
    fw = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fontsize, color=c, fontweight=fw,
            ha=ha, va=va, fontfamily=family, zorder=5, linespacing=1.25)


def arrow_h(ax, x0, y, x1, text=None, color=None):
    """Horizontal arrow with optional label."""
    c = color or PAL["arrow"]
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2), zorder=2)
    if text:
        ax.text((x0+x1)/2, y+0.12, text, fontsize=6, color=c,
                ha="center", va="bottom")


def arrow_v(ax, x, y0, y1, text=None, color=None):
    """Vertical arrow with optional label."""
    c = color or PAL["arrow"]
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2), zorder=2)
    if text:
        ax.text(x+0.15, (y0+y1)/2, text, fontsize=6, color=c,
                ha="left", va="center", rotation=90)


def main():
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    # ════════════════════════════════════════════════════════════════
    # ROUND 1 (top row)
    # ════════════════════════════════════════════════════════════════
    r1_y = 2.25
    r1_h = 1.6

    # Background
    rbox(ax, 0.05, r1_y, 7.1, r1_h, fc=PAL["bg_r1"], ec=PAL["border"], radius=0.06)
    label(ax, 0.45, r1_y+r1_h-0.12, "Round 1: Which crossbar size?",
          fontsize=9, bold=True, color=PAL["fpga"], ha="left", va="top")

    # ── Design Space: Config Matrix ──
    mx, my = 0.2, r1_y + 0.15
    mw, mh = 2.0, 1.15
    rbox(ax, mx, my, mw, mh, fc=PAL["white"], ec=PAL["border"])
    label(ax, mx+mw/2, my+mh-0.1, "Design Space", fontsize=7.5, bold=True,
          color=PAL["subtext"])

    # 4x3 config grid
    rows_label = ["128", "256", "512", "1024"]
    cols_label = ["64", "128", "256"]
    top3 = {(2,1), (3,1), (3,0)}  # (row, col) for 512x128, 1024x128, 1024x64

    gx0, gy0 = mx + 0.25, my + 0.1
    cw, ch = 0.42, 0.18
    gap = 0.03

    # Column headers
    for j, cl in enumerate(cols_label):
        label(ax, gx0 + j*(cw+gap) + cw/2, gy0 + len(rows_label)*(ch+gap) + 0.02,
              f"C={cl}", fontsize=5.5, color=PAL["subtext"])

    # Row labels + cells
    for i, rl in enumerate(rows_label):
        ry = gy0 + (len(rows_label)-1-i)*(ch+gap)
        label(ax, gx0 - 0.12, ry + ch/2, f"R={rl}", fontsize=5.5,
              color=PAL["subtext"], ha="right")
        for j, cl in enumerate(cols_label):
            cx = gx0 + j*(cw+gap)
            is_top = (i, j) in top3
            fc = PAL["dpe_light"] if is_top else "#F9FAFB"
            ec = PAL["dpe"] if is_top else "#E5E7EB"
            lw_cell = 1.2 if is_top else 0.5
            cell = FancyBboxPatch((cx, ry), cw, ch, boxstyle="round,pad=0.02",
                                   facecolor=fc, edgecolor=ec, linewidth=lw_cell,
                                   zorder=4)
            ax.add_patch(cell)
            tc = PAL["dpe"] if is_top else PAL["subtext"]
            fw = "bold" if is_top else "normal"
            ax.text(cx+cw/2, ry+ch/2, f"{rl}x{cl}",
                    fontsize=4.5, color=tc, fontweight=fw,
                    ha="center", va="center", zorder=5)

    label(ax, mx+mw/2, my+0.02, "12 configs (4 rows x 3 cols)",
          fontsize=5.5, color=PAL["subtext"])

    # ── Arrow →  ──
    arrow_h(ax, mx+mw+0.05, r1_y+r1_h/2, mx+mw+0.35)

    # ── Evaluation ──
    ex, ey = 2.55, r1_y + 0.15
    ew, eh = 1.8, 1.15
    rbox(ax, ex, ey, ew, eh, fc=PAL["white"], ec=PAL["border"])
    label(ax, ex+ew/2, ey+eh-0.1, "Evaluation", fontsize=7.5, bold=True,
          color=PAL["subtext"])

    # VTR icon
    rbox(ax, ex+0.2, ey+0.52, ew-0.4, 0.3, fc=PAL["fpga_light"], ec=PAL["fpga"])
    label(ax, ex+ew/2, ey+0.67, "VTR Place & Route", fontsize=7, bold=True,
          color=PAL["fpga"])

    label(ax, ex+ew/2, ey+0.38, "Auto-layout grid\n(sized per design)",
          fontsize=6, color=PAL["subtext"])
    label(ax, ex+ew/2, ey+0.1, "12 configs x 6 FC workloads\n= 72 VTR runs",
          fontsize=6, color=PAL["subtext"])

    # ── Arrow → ──
    arrow_h(ax, ex+ew+0.05, r1_y+r1_h/2, ex+ew+0.35)

    # ── Result: Ranking ──
    rx, ry_ = 4.7, r1_y + 0.15
    rw, rh = 2.3, 1.15
    rbox(ax, rx, ry_, rw, rh, fc=PAL["white"], ec=PAL["border"])
    label(ax, rx+rw/2, ry_+rh-0.1, "Result: EDAP Ranking", fontsize=7.5,
          bold=True, color=PAL["subtext"])

    # Simplified bar chart
    bar_x = rx + 0.15
    bar_y = ry_ + 0.28
    bar_h_max = 0.55
    configs = ["512\u00d7128", "1024\u00d7128", "1024\u00d764", "512\u00d7256", "1024\u00d7256"]
    scores = [1.0, 0.92, 0.85, 0.72, 0.65]
    bar_w = 0.34
    bar_gap = 0.05
    for i, (cfg, sc) in enumerate(zip(configs, scores)):
        bx = bar_x + i*(bar_w + bar_gap)
        bh = sc * bar_h_max
        fc = PAL["dpe"] if i < 3 else "#D1D5DB"
        ax.add_patch(plt.Rectangle((bx, bar_y), bar_w, bh,
                     facecolor=fc, edgecolor="none", alpha=0.8, zorder=4))
        ax.text(bx+bar_w/2, bar_y-0.02, cfg,
                fontsize=4, color=PAL["text"], ha="center", va="top",
                rotation=50, zorder=5)
        if i == 0:
            ax.text(bx+bar_w/2, bar_y+bh+0.02, "#1",
                    fontsize=5.5, color=PAL["dpe"], ha="center", fontweight="bold",
                    zorder=5)

    # ════════════════════════════════════════════════════════════════
    # ARROW: Round 1 → Round 2
    # ════════════════════════════════════════════════════════════════
    arrow_v(ax, 1.2, r1_y, r1_y-0.2, color=PAL["dpe"])
    label(ax, 1.55, r1_y-0.1, "Top-5", fontsize=6, color=PAL["dpe"], ha="left")

    # ════════════════════════════════════════════════════════════════
    # ROUND 2 (bottom row)
    # ════════════════════════════════════════════════════════════════
    r2_y = 0.05
    r2_h = 1.95

    # Background
    rbox(ax, 0.05, r2_y, 7.1, r2_h, fc=PAL["bg_r2"], ec=PAL["border"], radius=0.06)
    label(ax, 0.45, r2_y+r2_h-0.05, "Round 2: How much FPGA area for DPE?",
          fontsize=9, bold=True, color=PAL["dpe"], ha="left", va="top")

    # ── Design Space: Budget Slider ──
    dx, dy = 0.2, r2_y + 0.1
    dw, dh = 1.6, 1.55
    rbox(ax, dx, dy, dw, dh, fc=PAL["white"], ec=PAL["border"])
    label(ax, dx+dw/2, dy+dh-0.1, "Design Space", fontsize=7.5, bold=True,
          color=PAL["subtext"])

    # Budget dots (horizontal)
    budgets = [0, 10, 20, 30, 35, 40, 50]
    dot_y = dy + 1.0
    dot_x0 = dx + 0.15
    dot_span = dw - 0.3
    for i, b in enumerate(budgets):
        bx = dot_x0 + i * dot_span / (len(budgets)-1)
        intensity = b / 50.0
        fc = plt.cm.Greens(0.3 + intensity * 0.6)
        ax.scatter(bx, dot_y, s=25, c=[fc], edgecolors=PAL["dpe"],
                   linewidths=0.6, zorder=5)
        ax.text(bx, dot_y-0.1, f"{b}%", fontsize=4.5, ha="center",
                color=PAL["subtext"], zorder=5)
    # Line connecting dots
    ax.plot([dot_x0, dot_x0 + dot_span], [dot_y, dot_y],
            color=PAL["border"], linewidth=1, zorder=4)

    label(ax, dx+dw/2, dot_y-0.22, "Area Budget %", fontsize=6.5, bold=True,
          color=PAL["dpe"])

    # Resource icons
    res_y = dy + 0.28
    label(ax, dx+dw/2, res_y+0.2, "Proportional reduction:", fontsize=5.5,
          color=PAL["subtext"])
    for i, (rname, rc) in enumerate([("CLB", PAL["fpga_light"]),
                                      ("DSP", "#FDE68A"),
                                      ("BRAM", PAL["dpe_light"])]):
        bx = dx + 0.25 + i * 0.5
        ax.add_patch(plt.Rectangle((bx, res_y), 0.12, 0.1,
                     facecolor=rc, edgecolor=PAL["border"], linewidth=0.4, zorder=4))
        ax.text(bx+0.16, res_y+0.05, rname, fontsize=5, color=PAL["subtext"],
                va="center", zorder=5)

    label(ax, dx+dw/2, dy+0.05, "60x60 fixed grid, 3 seeds/pt",
          fontsize=5.5, color=PAL["subtext"])

    # ── Arrow → ──
    arrow_h(ax, dx+dw+0.05, r2_y+r2_h/2, dx+dw+0.25)

    # ── Dual Evaluation ──
    ex2 = 2.1
    ew2 = 2.8
    eh_path = 0.58

    # DL path (top)
    dl_y = r2_y + 0.92
    rbox(ax, ex2, dl_y, ew2, eh_path, fc=PAL["fpga_light"], ec=PAL["fpga"])

    label(ax, ex2+0.15, dl_y+eh_path-0.08, "DL Performance",
          fontsize=7, bold=True, color=PAL["fpga"], ha="left", va="top")

    label(ax, ex2+ew2/2, dl_y+eh_path/2-0.05,
          "Bare GEMV replicas: Lat = 1000/(P x Fmax)\n"
          "3 FC workloads, geomean latency",
          fontsize=6, color=PAL["text"])

    # Non-DL path (bottom)
    ndl_y = r2_y + 0.15
    rbox(ax, ex2, ndl_y, ew2, eh_path, fc=PAL["nondl_light"], ec=PAL["nondl"])

    label(ax, ex2+0.15, ndl_y+eh_path-0.08, "Non-DL Performance (FlexScore)",
          fontsize=7, bold=True, color=PAL["nondl"], ha="left", va="top")

    label(ax, ex2+ew2/2, ndl_y+eh_path/2-0.05,
          "FS = mean(Fmax_i(budget) / Fmax_i(baseline))\n"
          "4 VTR benchmarks: bgm, LU8PEEng, stereo1, arm",
          fontsize=6, color=PAL["text"])

    # Merge bracket / arrows to result
    arrow_h(ax, ex2+ew2+0.05, dl_y+eh_path/2, ex2+ew2+0.35, color=PAL["fpga"])
    arrow_h(ax, ex2+ew2+0.05, ndl_y+eh_path/2, ex2+ew2+0.35, color=PAL["nondl"])

    # ── Result: Pareto Front ──
    px = 5.25
    pw, ph = 1.8, 1.55
    py = r2_y + 0.1
    rbox(ax, px, py, pw, ph, fc=PAL["white"], ec=PAL["border"])

    # Mini Pareto axes (positioned inside the Result box)
    ax_p = fig.add_axes([0.76, 0.055, 0.18, 0.30])

    # Synthetic L-curve data
    fs_loss = np.array([0.01, 0.02, 0.03, 0.25, 0.75, 0.75])
    lat = np.array([1.1, 0.55, 0.47, 0.35, 0.28, 0.47])
    area = np.array([5, 9, 23, 23, 33, 37])

    # Scatter
    sc = ax_p.scatter(fs_loss, lat, c=area, cmap="YlOrRd", s=20, alpha=0.6,
                      edgecolors="#666", linewidths=0.4, zorder=3,
                      vmin=0, vmax=40)

    # Pareto front line
    pareto_x = [0.01, 0.02, 0.25, 0.75]
    pareto_y = [1.1, 0.55, 0.35, 0.28]
    ax_p.plot(pareto_x, pareto_y, "k-", linewidth=1.2, alpha=0.5, zorder=4)
    for x, y in zip(pareto_x, pareto_y):
        ax_p.scatter([x], [y], c="white", s=22, edgecolors="black",
                     linewidths=0.8, zorder=5)

    # Knee point
    ax_p.scatter([0.02], [0.55], color="gold", s=70, edgecolors=PAL["dpe"],
                 linewidths=1.2, zorder=7, marker="*")
    ax_p.annotate("Balanced", xy=(0.02, 0.55), xytext=(0.3, 0.85),
                  fontsize=5.5, fontweight="bold", color=PAL["dpe"],
                  arrowprops=dict(arrowstyle="->", color=PAL["dpe"], lw=0.8),
                  bbox=dict(boxstyle="round,pad=0.15", facecolor=PAL["dpe_light"],
                            edgecolor=PAL["dpe"], linewidth=0.5), zorder=8)

    ax_p.set_xlabel("1-FlexScore", fontsize=5.5, labelpad=1)
    ax_p.set_ylabel("Latency (ns/inf)", fontsize=5.5, labelpad=1)
    ax_p.tick_params(labelsize=4.5, length=1.5, pad=1)
    ax_p.set_xlim(-0.05, 0.85)
    ax_p.set_ylim(0, 1.3)
    ax_p.grid(True, alpha=0.1, linewidth=0.3)
    ax_p.spines['top'].set_visible(False)
    ax_p.spines['right'].set_visible(False)
    from matplotlib.ticker import FuncFormatter
    ax_p.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    label(ax, px+pw/2, py+ph-0.1, "Result: Pareto Front",
          fontsize=7.5, bold=True, color=PAL["subtext"])
    # VTR run count - place above the Pareto title
    label(ax, px+pw/2, py+ph-0.22, "~400 VTR runs",
          fontsize=5, color=PAL["subtext"])

    # ── Save ──
    out = RESULTS_DIR / "dse_methodology_overview.pdf"
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
