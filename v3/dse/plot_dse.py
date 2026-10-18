#!/usr/bin/env python3
"""plot_dse.py — the DSE figures from dse/rank.py's JSON (matplotlib: run it with the miniconda Python, the system
Python has no matplotlib):

  /mnt/vault0/jiajunh5/miniconda3/bin/python v3/dse/plot_dse.py

Ported from v2/experiment/dse/plot_dse.py and v2/work/dse_v1/combined/plot_pareto_family.py (layout unchanged; the
inputs and workload notes are v3's). Two views, one folder each under dse/results/:
ADP view (the current one, user 2026-10-16: latency vs area per family, area-delay product across the families),
from adp/latency_area.json, written to adp/:
  dse_la.{png,pdf}                              three panels side by side (for a paper)
  dse_la_fc.png, dse_la_attention.png           latency score vs area score per family, with the Pareto front
  dse_adp_cross.png                             FC ADP score (x) vs attention ADP score (y), circles of equal distance
EDAP view (the earlier one: energy x delay x area product), from edap/ranking_*.json, written to edap/:
  dse_{fc,attention}.{png,pdf}   per family, four panels:
      (a) crossbar-size map: R x C grid of the config score G (geometric mean of best/EDAP over the family's workloads)
      (b) per-workload score best/EDAP, configs in rank order, plus the G column
      (c) G per config in rank order (the raw data: Fmax = the mean of the 3 VTR seeds; no resampling, user 2026-10-16)
      (d) the three factors, each = value / best value on the workload, geometric mean over workloads (log scale)
  dse_pareto.{png,pdf}           per family: x = latency, y = energy x area (each / the best on the workload, geomean
      over the workloads). Since EDAP = D x (E x A), the lines x y = const are iso-EDAP lines (slope -1 in log-log),
      ordered as the ranking; the 2-cost front and the 3-cost (E, D, A) front are marked.
  dse_cross.{png,pdf}            FC x attention cross score: the point (1 - G_FC, 1 - G_ATT), closer to the origin is better
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

V3 = Path(__file__).resolve().parents[1]
RESULTS = V3 / "dse" / "results"
EDAP = RESULTS / "edap"            # the earlier view: energy x delay x area product ranking
ADP = RESULTS / "adp"              # the current view: latency vs area per family, area-delay product across them
OKABE = {"E": "#0072B2", "D": "#E69F00", "A": "#009E73"}
RS, CS = (128, 256, 512, 1024), (64, 128, 256)
NOTES = {
    "fc": ("FC — crossbar ranking by EDAP",
           "GEMV M × K × N = 1 × 512 × 128, 1 × 2048 × 256, 1 × 256 × 512, 1 × 512 × 512 (the v2 FC snapshot, rebuilt "
           "by the v3 model)"),
    "attention": ("Attention head — crossbar ranking by EDAP",
                  "one head, (n, d_model, d): ViT-B/16 (256, 768, 64), DeiT-T (256, 192, 64), BERT-base (128, 768, 64), "
                  "BERT-Tiny (128, 128, 64),\nJetFormer-150P (32, 64, 32), JetFormer-JetClass (128, 128, 16). The 16 "
                  "builds of the d = 64 shapes at C = 64 are priced from VPR's memory count (5–8 blocks above the "
                  "simulator's prediction)."),
}


def wl_label(fam: str, w: str) -> str:
    return w.replace("x", " × ") if fam == "fc" else w


def annotate(ax, im, data, fmt, fs=9):
    lo, hi = im.get_clim()
    for (i, j), v in np.ndenumerate(data):
        if np.isnan(v):
            continue
        dark = (v - lo) / (hi - lo + 1e-12) < 0.55
        ax.text(j, i, fmt(i, j, v), ha="center", va="center", fontsize=fs, color="white" if dark else "black")


def figure(res: dict, name: str, title: str, wl_note: str) -> None:
    fam, cfgs, wls = res["family"], res["configs"], res["workloads"]
    by = {c["config"]: c for c in cfgs}
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1, 1.15], wspace=0.32, hspace=0.32)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.995)
    n = len(wls)
    fig.text(0.06, 0.965, f"Workloads ({n}): {wl_note}\n"
             f"EDAP = energy × latency × area.  Per workload: score = best EDAP ÷ EDAP (1 = best).  "
             f"G = geomean of the {n} scores.", fontsize=10, color="#333", ha="left", va="top", linespacing=1.5)
    cmap = plt.get_cmap("viridis")

    ax = fig.add_subplot(gs[0, 0])                                     # (a) crossbar-size map
    grid = np.array([[by[f"{r}x{c}"]["G"] for c in CS] for r in RS])
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    annotate(ax, im, grid, lambda i, j, v: f"{v:.3f}\n#{by[f'{RS[i]}x{CS[j]}']['rank']}", fs=10)
    bi, bj = np.unravel_index(np.argmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="#d62728", lw=3))
    ax.set_xticks(range(len(CS)), [str(c) for c in CS])
    ax.set_yticks(range(len(RS)), [str(r) for r in RS])
    ax.set_xlabel("columns C")
    ax.set_ylabel("rows R")
    ax.set_title(f"(a) G = geomean over {n} workloads  (#rank)", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[0, 1])                                     # (b) per-workload scores
    order = [c["config"] for c in cfgs]
    cols = wls + ["G"]
    data = np.array([[by[c]["scores"][w] for w in wls] + [by[c]["G"]] for c in order])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    annotate(ax, im, data, lambda i, j, v: f"{v:.2f}", fs=8.5)
    ax.axvline(len(wls) - 0.5, color="white", lw=3)
    ax.set_xticks(range(len(cols)), [wl_label(fam, w) for w in wls] + ["G"], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(order)), [f"#{i + 1}  {c}" for i, c in enumerate(order)], fontsize=9)
    ax.set_title("(b) per-workload score, no averaging;  last column = G", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, 0])                                     # (c) G
    y = np.arange(len(order))[::-1]
    g = np.array([by[c]["G"] for c in order])
    ax.barh(y, g, color="#4c72b0", height=0.65)
    for yy, c in zip(y, order):
        ax.text(by[c]["G"] + 0.015, yy, f"{by[c]['G']:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y, [f"#{i + 1}  {c}" for i, c in enumerate(order)], fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("G   (Fmax = the mean of the 3 VTR seeds)")
    ax.set_title(f"(c) ranking by G (geomean over {n} workloads)", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, 1])                                     # (d) factors
    hgt = 0.26
    for k, (f, lab) in enumerate((("E", "energy"), ("D", "latency"), ("A", "area"))):
        v = np.array([by[c][f"factor_{f}"] for c in order])
        ax.barh(y + (1 - k) * hgt, v, height=hgt, color=OKABE[f], label=lab)
    ax.set_xscale("log")
    ticks = [t for t in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5) if t <= ax.get_xlim()[1] * 1.02]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvline(1, color="black", lw=0.8)
    ax.set_yticks(y, [f"#{i + 1}  {c}" for i, c in enumerate(order)], fontsize=9)
    ax.set_xlabel("÷ best per workload, geomean (1 = best)")
    ax.set_title(f"(d) E, D, A factors, geomean over {n} workloads", fontsize=11, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=9, frameon=False, ncol=3)

    EDAP.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(EDAP / f"dse_{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ---- Pareto (from plot_pareto_family.py) --------------------------------------------------------------------------
def load(res: dict) -> dict:
    return {c["config"]: dict(E=c["factor_E"], D=c["factor_D"], A=c["factor_A"], G=c["G"], rank=c["rank"])
            for c in res["configs"]}


def front(pts, keys):
    return {c for c, p in pts.items()
            if not any(all(q[k] <= p[k] for k in keys) and any(q[k] < p[k] for k in keys) for q in pts.values())}


def panel(ax, d, title):
    x = {c: v["D"] for c, v in d.items()}
    y = {c: v["E"] * v["A"] for c, v in d.items()}
    f2 = front({c: dict(x=x[c], y=y[c]) for c in d}, ("x", "y"))
    f3 = front(d, ("E", "D", "A"))
    top = min(d, key=lambda c: d[c]["rank"])
    k0 = x[top] * y[top]
    lo, hi = 0.9, 6.0
    isos = []
    for m in (1, 1.25, 1.5, 2, 3):
        xs = [lo, hi]
        ax.plot(xs, [k0 * m / v for v in xs], color="#2ca02c" if m == 1 else "#999999", lw=1.0 if m == 1 else .7,
                ls="-" if m == 1 else ":", zorder=0)
        isos.append((m, k0 * m))
    fr = sorted(f2, key=lambda c: x[c])
    ax.plot([x[c] for c in fr], [y[c] for c in fr], color="#d62728", lw=1.5, zorder=1)
    for c in d:
        on2, on3 = c in f2, c in f3
        ax.scatter(x[c], y[c], s=75 if on2 else 38, marker="o" if on3 else "s",
                   color="#d62728" if on2 else ("#ff9896" if on3 else "#8c8c8c"),
                   edgecolor="black" if c == top else "none", linewidth=1.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    t = [1, 1.25, 1.5, 2, 3, 4]
    ax.set_xticks(t, [f"{v:g}×" for v in t])
    yt = [1, 1.5, 2, 3, 5, 8, 12]
    ax.set_yticks(yt, [f"{v:g}×" for v in yt])
    ax.minorticks_off()
    ax.set_xlim(.92, max(2.6, max(x.values()) * 1.25))
    ax.set_ylim(.85, max(y.values()) * 1.35)
    ax.set_xlabel("latency ÷ best")
    ax.set_ylabel("energy × area ÷ best")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=.25)
    ax._lab = [(c, x[c], y[c], c in f2) for c in d]
    ax._iso = isos
    return f2, f3, top


def place_labels(fig, ax):
    """Point labels with a greedy no-overlap choice of offset; iso-EDAP labels where each line leaves the plot."""
    fig.canvas.draw()
    tr = ax.transData
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    taken = [tuple(tr.transform((px, py))) + (0,) for _, px, py, _ in ax._lab]
    boxes = []
    dpi = fig.dpi / 72

    def hit(b):
        bx0, by0, bx1, by1 = b
        if any(not (bx1 < c0 or bx0 > c2 or by1 < c1 or by0 > c3) for c0, c1, c2, c3 in boxes):
            return True
        return any(bx0 - 4 < px < bx1 + 4 and by0 - 4 < py < by1 + 4 for px, py, _ in taken)
    for c, px, py, bold in sorted(ax._lab, key=lambda t: -t[3]):
        X, Y = tr.transform((px, py))
        w, h = (6.1 if bold else 5.6) * len(c) * dpi, 10 * dpi
        for dx, dy in ((6, 4), (6, -13), (-6 - w / dpi, 4), (-6 - w / dpi, -13), (6, 13), (-w / dpi / 2, 9),
                       (-w / dpi / 2, -20)):
            b = (X + dx * dpi, Y + dy * dpi, X + dx * dpi + w, Y + dy * dpi + h)
            if not hit(b):
                break
        boxes.append(b)
        ax.annotate(c, (px, py), textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                    fontweight="bold" if bold else None)
    for m, k in ax._iso:
        col = "#2ca02c" if m == 1 else "#777777"
        xb = k / (y0 * 1.03)
        if x0 < xb < x1 * .95:
            ax.text(xb, y0 * 1.03, f"{m:g}×", fontsize=8, ha="left", va="bottom", color=col)
            continue
        yr = k / (x1 / 1.01)
        if y0 < yr < y1:
            ax.text(x1 / 1.01, yr, f"{m:g}×", fontsize=8, ha="right", va="bottom", color=col)


def pareto(fams: list) -> list:
    fig = plt.figure(figsize=(14, 6.4), constrained_layout=True)
    axs = [fig.add_subplot(1, 2, k + 1) for k in range(len(fams))]
    out = []
    for ax, (ttl, d) in zip(axs, fams):
        out.append((ttl, *panel(ax, d, ttl)))
    for ax in axs:
        place_labels(fig, ax)
    hs = [Line2D([], [], color="#d62728", lw=1.5, marker="o", ms=7, label="Pareto front"),
          Line2D([], [], color="#ff9896", lw=0, marker="o", ms=6, label="3-cost front only"),
          Line2D([], [], color="#8c8c8c", lw=0, marker="s", ms=6, label="dominated"),
          Line2D([], [], color="white", lw=0, marker="o", ms=8, mec="black", mew=1.8, label="EDAP #1"),
          Line2D([], [], color="#2ca02c", lw=1.0, label="equal EDAP to #1"),
          Line2D([], [], color="#999999", lw=.7, ls=":", label="1.25× … 3× its EDAP")]
    axs[0].legend(handles=hs, loc="upper right", fontsize=8, framealpha=.9)
    fig.suptitle("Crossbar configs per family: latency vs energy × area (each ÷ the best on the workload, geomean)",
                 fontsize=13)
    EDAP.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(EDAP / f"dse_pareto.{ext}", dpi=150)
    plt.close(fig)
    return out


def cross(cr: dict, n_fc: int, n_att: int) -> tuple:
    """The FC x attention cross score (user 2026-10-16; the plane of v2/work/dse_v1/combined/plot_pareto.py, turned to
    1 - EDAP score): x = 1 - G_FC, y = 1 - G_ATT per config, so the origin is the ideal (best on every workload of
    both families) and a config is better the closer it lies (d, dse/rank.py's ranking_cross.json). Drawn: arcs of
    equal distance (the closest config's solid) and the Pareto front; the raw data only (no resampling)."""
    cs = {c["config"]: c for c in cr["configs"]}
    pick = cr["configs"][0]
    front = [c for c in cs if cs[c]["front"]]
    lim = 0.92
    fig, ax = plt.subplots(figsize=(7.4, 7.2), constrained_layout=True)
    th = np.linspace(0, np.pi / 2, 200)
    for r in (0.2, 0.4, 0.6, 0.8):                                     # equal-distance arcs
        ax.plot(r * np.cos(th), r * np.sin(th), color="#9a9a9a", lw=.7, ls=":", zorder=0)
        ax.text(r * np.cos(np.radians(78)), r * np.sin(np.radians(78)), f"d = {r:g}", fontsize=8, color="#777777",
                rotation=-12, ha="center", va="bottom")
    r0 = pick["d"]
    ax.plot(r0 * np.cos(th), r0 * np.sin(th), color="#2ca02c", lw=1.3, zorder=1)
    ax.fill_between(r0 * np.cos(th), 0, r0 * np.sin(th), color="#2ca02c", alpha=.07, zorder=0)
    ax.scatter([0], [0], marker="*", s=200, color="black", zorder=4, clip_on=False)
    ax.annotate("ideal: best on every workload\nof both families", (0, 0), textcoords="offset points", xytext=(10, 8),
                fontsize=8.5)
    fr = sorted(front, key=lambda c: cs[c]["x"])
    ax.plot([cs[c]["x"] for c in fr], [cs[c]["y"] for c in fr], color="#d62728", lw=1.4, zorder=1)
    for c, p in cs.items():
        on = c in front
        ax.scatter(p["x"], p["y"], s=70 if on else 40, color="#d62728" if on else "#7f7f7f", zorder=3,
                   edgecolor="black" if c == pick["config"] else "none", linewidth=1.6)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"FC:  1 − EDAP score   (1 − G_FC, geomean over the {n_fc} FC workloads)")
    ax.set_ylabel(f"attention:  1 − EDAP score   (1 − G_ATT, geomean over the {n_att} attention workloads)")
    ax.set_title("FC × attention cross score, one dot per crossbar config (R × C)\n"
                 "closer to the origin is better; d = distance from the origin\n"
                 f"closest: {pick['config']}, d = {r0:.3f}", fontsize=10.5)
    ax.grid(alpha=.3)
    hs = [Line2D([], [], color="#d62728", lw=1.4, marker="o", ms=7, label="Pareto front"),
          Line2D([], [], color="#7f7f7f", lw=0, marker="o", ms=6, label="dominated"),
          Line2D([], [], color="white", lw=0, marker="o", ms=8, mec="black", mew=1.6, label="closest to the origin"),
          Line2D([], [], color="#2ca02c", lw=1.3, label=f"d = {r0:.3f}: nothing closer"),
          Line2D([], [], color="#9a9a9a", lw=.7, ls=":", label="equal distance d")]
    ax.legend(handles=hs, loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=8.5, framealpha=.92)
    ax._lab = [(c, p["x"], p["y"], c in front) for c, p in cs.items()]
    ax._iso = []
    place_labels(fig, ax)
    EDAP.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(EDAP / f"dse_cross.{ext}", dpi=160)
    plt.close(fig)
    return front, pick


def la_panel(ax, rows: list, title: str) -> None:
    """One latency vs area panel: each config at (area score, latency score), the Pareto front in red."""
    front = sorted((c for c in rows if c["front"]), key=lambda c: c["area_score"])
    ax.plot([c["area_score"] for c in front], [c["latency_score"] for c in front], color="#d62728", lw=1.4, zorder=1)
    for c in rows:
        ax.scatter(c["area_score"], c["latency_score"], s=70 if c["front"] else 40,
                   color="#d62728" if c["front"] else "#7f7f7f", zorder=3)
    ax.set_xlim(.3, 1.05)
    ax.set_ylim(.3, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("area ÷ largest area  (geomean over the workloads)")
    ax.set_ylabel("latency ÷ highest latency  (geomean over the workloads)")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=.3)
    ax.annotate("better", xy=(.33, .33), xytext=(.43, .43), fontsize=9, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2), ha="left", va="bottom")
    ax._lab = [(c["config"], c["area_score"], c["latency_score"], c["front"]) for c in rows]
    ax._iso = []


def adp_panel(ax, rows: list, n_fc: int, n_att: int, title: str) -> None:
    """The cross-family panel (user, 2026-10-16): each config at (FC ADP score, attention ADP score), the front red."""
    front = sorted((c for c in rows if c["front"]), key=lambda c: c["adp_fc"])
    ax.plot([c["adp_fc"] for c in front], [c["adp_att"] for c in front], color="#d62728", lw=1.4, zorder=1)
    for c in rows:
        ax.scatter(c["adp_fc"], c["adp_att"], s=70 if c["front"] else 40, color="#d62728" if c["front"] else "#7f7f7f",
                   zorder=3)
    th = np.linspace(0, np.pi / 2, 200)                                # circles of equal distance from the origin
    for r in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot(r * np.cos(th), r * np.sin(th), color="#9a9a9a", lw=.7, ls=":", zorder=0)
        ax.text(r * np.cos(np.radians(84)), r * np.sin(np.radians(84)) + .006, f"d = {r:g}", fontsize=8,
                color="#777777", ha="center", va="bottom")
    best = min(rows, key=lambda c: c["d"])
    r0 = best["d"]
    ax.plot(r0 * np.cos(th), r0 * np.sin(th), color="#2ca02c", lw=1.3, zorder=1)
    ax.fill_between(r0 * np.cos(th), 0, r0 * np.sin(th), color="#2ca02c", alpha=.07, zorder=0)
    ax.scatter([0], [0], marker="*", s=160, color="black", zorder=4, clip_on=False)
    ax.annotate("origin", (0, 0), textcoords="offset points", xytext=(8, 6), fontsize=8.5)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel(f"FC: area-delay product score  (geomean over the {n_fc} FC workloads)")
    ax.set_ylabel(f"attention: area-delay product score  (geomean over the {n_att} workloads)")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=.3)
    ax._lab = [(c["config"], c["adp_fc"], c["adp_att"], c["front"]) for c in rows]
    ax._iso = []
    ax._extra = [Line2D([], [], color="#2ca02c", lw=1.3, label=f"d = {r0:.3f}, the closest ({best['config']})"),
                 Line2D([], [], color="#9a9a9a", lw=.7, ls=":", label="equal distance d from the origin")]


def latency_area(la: dict) -> dict:
    """Latency vs area (user, 2026-10-16), from dse/rank.py's latency_area.json: per family, each config's area score
    (area / the largest area) against its latency score (latency / the highest latency), geometric means over the
    workloads; across the families, x = the FC area-delay product score, y = the attention one (ADP / the largest
    ADP per workload, geometric mean). Lower-left is better everywhere. Writes dse_la.{png,pdf} (three panels) and
    dse_la_fc.png, dse_la_attention.png, dse_adp_cross.png (one panel each, for the results page)."""
    nfc, nat = len(la["fc"]["workloads"]), len(la["attention"]["workloads"])
    t_fc, t_at = f"(a) FC: latency vs area  ({nfc} workloads)", f"(b) attention head: latency vs area  ({nat} workloads)"
    t_x = "(c) across the families: area-delay product, FC on x, attention on y;\ncircles = equal distance d from the origin"
    hs = [Line2D([], [], color="#d62728", lw=1.4, marker="o", ms=7, label="Pareto front"),
          Line2D([], [], color="#7f7f7f", lw=0, marker="o", ms=6, label="dominated")]
    draw = [("fc", lambda ax: la_panel(ax, la["fc"]["configs"], t_fc), "dse_la_fc", "upper right"),
            ("attention", lambda ax: la_panel(ax, la["attention"]["configs"], t_at), "dse_la_attention", "lower right"),
            ("cross", lambda ax: adp_panel(ax, la["adp"]["cross"]["configs"], nfc, nat, t_x), "dse_adp_cross",
             "lower right")]                             # legends where no point lies
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.9), constrained_layout=True)
    for ax, (_, fn, _, _) in zip(axs, draw):
        fn(ax)
    for ax in axs:
        place_labels(fig, ax)
    axs[0].legend(handles=hs, loc="upper right", fontsize=9, framealpha=.9)
    axs[2].legend(handles=hs + axs[2]._extra, loc="lower right", fontsize=8.5, framealpha=.92)
    fig.suptitle("Crossbar configs (R × C): each score = the config's value ÷ the worst config's value on the workload, "
                 "so lower is better", fontsize=12.5)
    ADP.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ADP / f"dse_la.{ext}", dpi=150)
    plt.close(fig)
    for _, fn, stem, loc in draw:
        fig, ax = plt.subplots(figsize=(6.8, 7.0), constrained_layout=True)
        fn(ax)
        place_labels(fig, ax)
        ax.legend(handles=hs + getattr(ax, "_extra", []), loc=loc, fontsize=8.5, framealpha=.92)
        fig.savefig(ADP / f"{stem}.png", dpi=160)
        plt.close(fig)
    return dict(fc=[c["config"] for c in la["fc"]["configs"] if c["front"]],
                attention=[c["config"] for c in la["attention"]["configs"] if c["front"]],
                adp_cross=[c["config"] for c in la["adp"]["cross"]["configs"] if c["front"]])


def main() -> None:
    fronts = latency_area(json.loads((ADP / "latency_area.json").read_text()))
    print(f"wrote {ADP / 'dse_la.png'} and .pdf, dse_la_fc.png, dse_la_attention.png, dse_adp_cross.png | fronts {fronts}")
    res = {name: json.loads((EDAP / f"ranking_{name}.json").read_text()) for name in ("fc", "attention")}
    cr = json.loads((EDAP / "ranking_cross.json").read_text())
    front, pick = cross(cr, len(res["fc"]["workloads"]), len(res["attention"]["workloads"]))
    print(f"wrote {EDAP / 'dse_cross.png'} and .pdf | front {sorted(front)} | closest {pick['config']} d {pick['d']:.3f}")
    for name in ("fc", "attention"):
        figure(res[name], name, *NOTES[name])
        print(f"wrote {EDAP / f'dse_{name}.png'} and .pdf")
    out = pareto([(f"(a) FC  ({len(res['fc']['workloads'])} workloads)", load(res["fc"])),
                  (f"(b) attention head  ({len(res['attention']['workloads'])} workloads)", load(res["attention"]))])
    print(f"wrote {EDAP / 'dse_pareto.png'} and .pdf")
    for ttl, f2, f3, top in out:
        print(ttl, "| front", sorted(f2), "| 3-cost front", sorted(f3), "| #1", top)


if __name__ == "__main__":
    main()
