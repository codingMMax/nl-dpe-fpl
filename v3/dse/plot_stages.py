#!/usr/bin/env python3
"""plot_stages.py — the figures of the stage analysis (dse/stage_analysis.py's results/stages/stages.json):

  stages_gantt_<case>.png   the pipelined time diagram of one case-study head at the two extreme arrays (128x64 and
                            512x256): one lane per stage (projections, tiled QK, online softmax, tiled PV, output
                            rescale), its busy intervals as bars, the run boundaries
  stages_sweep.png          the array sweep on the DSE grid, one row per head: cycles, energy, area and cycles × area
                            (relative to the smallest array) against R, one curve per C; lines = the closed forms
                            T_head, E_head, A_head, markers = the simulator (area: VTR's counts)
  stages_explain_<case>.png the time diagram of the smallest head at the smallest array with every step labelled by
                            what it computes and arrows for what it waits on (the stage page's "step by step" figure)
  stages_simple_<case>.png  the same reduced to one box per block per stage, one head at the two DIAGRAMS arrays
                            (128x64 and 512x256) stacked on a shared time axis (the brief's "head in time"; many
                            blocks: the runs labelled instead)
  stages_costs.png          the cost of every stage on the 12 arrays (grouped by C, R rising), one row per head:
                            stacked bars of the simulator's critical-path cycles by stage, the energy by stage and
                            the area by stage with the soft logic and the memory blocks (the brief, stages_brief.py)
Run with the matplotlib Python (run_all.sh's $MPL): "$MPL" v3/dse/plot_stages.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

V3 = Path(__file__).resolve().parents[1]
STAGES_DIR = V3 / "dse" / "results" / "stages"
LANE_COLOR = {"gemm": "#0072B2", "qk": "#56B4E9", "sm": "#E69F00", "pv": "#009E73", "acc": "#CC79A7"}
C_COLOR = {64: "#0072B2", 128: "#E69F00", 256: "#009E73"}
# what each stage computes (matplotlib mathtext), on the lane labels of the time diagrams
STAGE_MATH = {"projections": r"$q_t, k_t, v_t = x_t W_Q,\ x_t W_K,\ x_t W_V$",
              "tiled QK": r"$S_b = Q\,K_b^{\top}$",
              "online softmax": r"$m, l$ running; $P_b = e^{S_b - m}$",
              "tiled PV": r"$O_b = P_b\,V_b$",
              "output rescale": r"$o \leftarrow \alpha\,o + O_b$; $O = o / l$"}
CONVERT_MATH = {"tiled QK": r"$\log q_t, \log k_t$", "tiled PV": r"$\log v_t$"}


def stage_of(lane: str) -> str:
    """The stage a lane name stands for ("scores (tiled QK)" is the tiled QK stage)."""
    return "tiled QK" if lane.startswith("scores") else lane


LANE_DISPLAY = {"projections": "projections", "tiled QK": r"$QK^{\top}$", "online softmax": "online softmax", "tiled PV": "PV",
                "output rescale": "output rescale"}


def lane_labels(lanes: list) -> list:
    """The y-axis names of the lanes (the equations are in the boxes and the captions)."""
    return [LANE_DISPLAY[stage_of(name)] for name in reversed(lanes)]
PANELS = (("T", "cycles", 1.0), ("E", "energy per inference (nJ)", 1e3), ("A", "area (mm²)", 1.0),
          ("xa", "cycles × area, relative to the smallest array", 1.0))


def tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def laneColor(lane: dict) -> str:
    return LANE_COLOR[lane["stage"]]


def gantt(case: dict, out: Path) -> None:
    cfgs = list(case["gantt"])
    fig, axs = plt.subplots(len(cfgs), 1, figsize=(16, 1.6 + 2.0 * len(cfgs)), constrained_layout=True)
    for ax, cfg in zip(np.atleast_1d(axs), cfgs):
        g = case["gantt"][cfg]
        lanes = g["lanes"]
        for i, lane in enumerate(lanes):
            y = len(lanes) - 1 - i
            ax.broken_barh([(a, max(b - a, 1)) for a, b in lane["intervals"]], (y - 0.34, 0.68), color=laneColor(lane),
                           edgecolor="none", alpha=0.9)
        ax.set_yticks(range(len(lanes)), [lane["name"] for lane in reversed(lanes)], fontsize=9.5)
        ax.set_xlim(0, g["T"] * 1.01)
        ax.set_xlabel("cycle")
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        c = case["configs"][cfg]
        ax.set_title(f"{case['name']} (n {case['n']}, d_model {case['dm']}, d {case['d']}) at {cfg.replace('x', '×')}: "
                     f"T = {g['T']:,} cycles, S = {c['S']} queries per run, {c['runs']} run(s) × {c['blocks']} key blocks",
                     fontsize=10.5)
        for t in g.get("run_ends", [])[:-1]:                        # the run boundaries: the softmax's run ends
            ax.axvline(t, color="#888888", lw=0.8, ls="--")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def sweep(cases: list, out: Path) -> None:
    fig, axs = plt.subplots(len(cases), len(PANELS), figsize=(22, 4.9 * len(cases)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for row, case in enumerate(cases):
        curves = sorted(case["grid"].items(), key=lambda kv: int(kv[0]))
        Rs = curves[0][1]["R"]
        head = f"{case['name']} (n {case['n']}, d_model {case['dm']}, d {case['d']})"
        for col, (key, ylabel, div) in enumerate(PANELS):
            ax = axs[row, col]
            for C, g in curves:
                color = C_COLOR[int(C)]
                ax.plot(g["R"], np.array(g[key + "_model"]) / div, color=color, lw=1.8, label=f"C = {C}", zorder=3)
                ax.plot(g["R"], np.array(g[key]) / div, color=color, marker="o", ms=7, mfc="white", mew=1.7, lw=0, zorder=4)
            ax.set_xscale("log", base=2)
            ax.set_xticks(Rs, [str(r) for r in Rs])
            ax.set_xlabel("R (rows)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.grid(color="#e5e5e5", lw=0.6)
            ax.set_title(f"{head}: {ylabel.split(',')[0]}", fontsize=10.5)
            if key == "xa":
                ax.axhline(1, color="#888888", lw=0.8, ls="--")
            if col == 0:
                handles, labels = ax.get_legend_handles_labels()
                handles += [Line2D([], [], color="#444444", lw=1.8),
                            Line2D([], [], color="#444444", marker="o", ms=7, mfc="white", mew=1.7, lw=0)]
                labels += ["line: closed form", "marker: simulator"]
                ax.legend(handles, labels, fontsize=8.5, loc="best")
    fig.suptitle("The DSE grid, one curve per C against R: lines = the closed forms T_head, E_head, A_head; "
                 "markers = the simulator (area: VTR's counts)", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def explain(case: dict, cfg: str, out: Path) -> None:
    """The time diagram of one head at one array, every step labelled by what it computes, arrows for what it waits
    on. Meant for a head with a few blocks (the smallest case-study head), where every box is readable."""
    g, c = case["gantt"][cfg], case["configs"][cfg]
    st, T, n, S, blocks = g["steps"], g["T"], case["n"], c["S"], c["blocks"]
    bkv = n // blocks
    lanes = ["projections", "tiled QK", "online softmax", "tiled PV", "output rescale"]
    y = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}
    fig, ax = plt.subplots(figsize=(21, 7.4), constrained_layout=True)

    def box(lane, a, b, text, color, dy=0.0, h=0.6, fs=8.3):
        ax.broken_barh([(a, max(b - a, 1))], (y[lane] - h / 2 + dy, h), color=color, edgecolor="white", lw=0.8, zorder=3)
        if text:
            ax.text((a + b) / 2, y[lane] + dy, text, ha="center", va="center", fontsize=fs, color="white", zorder=4)

    def ticks(lane, steps, color, dy=0.0, h=0.6):
        ax.broken_barh([(a, max(b - a, 1)) for a, b, _ in steps], (y[lane] - h / 2 + dy, h), color=color, edgecolor="none", zorder=2)

    def arrow(x0, lane0, x1, lane1, text, dx=0, color="#222222", ls="-"):
        ax.annotate("", xy=(x1, y[lane1] + 0.3), xytext=(x0, y[lane0] - 0.3),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.1, ls=ls, shrinkA=0, shrinkB=0), zorder=5)
        ax.text((x0 + x1) / 2 + dx, (y[lane0] + y[lane1]) / 2, text, ha="left", va="center", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)

    proj = st["gq0.rows"]
    Tp = proj[1][0] - proj[0][0]
    streamEnd = max(b for _, b, _ in proj)
    keysReady = next(b for _, b, u in st["gk0.rows"] if u == bkv - 1)
    box("projections", proj[0][0], streamEnd, f"project tokens 0 .. {n - 1}: Q, K and V, one token every {Tp} cycles",
        LANE_COLOR["gemm"], fs=9)
    ax.plot([keysReady], [y["projections"] + 0.4], marker="v", color="#222222", ms=7, zorder=5)
    ax.text(keysReady, y["projections"] + 0.56, f"keys of block 0 ready (cycle {keysReady}), then wait", ha="center", va="bottom", fontsize=8)
    # tiled QK: the conversions of each arriving token, then the score jobs
    ticks("tiled QK", st["qk.b_in"], "#a9d8f3")
    ax.text((st["qk.b_in"][0][0] + st["qk.b_in"][-1][1]) / 2, y["tiled QK"] + 0.42,
            f"{CONVERT_MATH['tiled QK']} per token as it arrives ({st['qk.b_in'][0][1] - st['qk.b_in'][0][0]} cycles each): keeps pace with the stream",
            ha="center", va="bottom", fontsize=8, color="#1b6ca8")
    for a, b, u in st["qk.jobs"]:
        box("tiled QK", a, b, f"$S_{u} = Q\\,K_{u}^{{\\top}}$\nqueries 0..{S - 1} × keys {u * bkv}..{u * bkv + bkv - 1}", LANE_COLOR["qk"])
    # online softmax: the tile reads above, the EXP + statistics below, the readout
    for a, b, u in st["sm.tile"]:
        box("online softmax", a, b, f"read tile {u}\n5 scores per cycle", LANE_COLOR["sm"], dy=0.17, h=0.32, fs=7.8)
    for a, b, u in st["sm.blk"]:
        box("online softmax", a, b, f"$P_{u} = e^{{S_{u} - m}}$, $m$, $l$", "#c98a00", dy=-0.17, h=0.32, fs=7.8)
    for a, b, u in st["sm.run"]:
        box("online softmax", a, b, "$\\log l$\nread-out", "#8a5f00", dy=-0.17, h=0.32, fs=7.2)
    # tiled PV: the V conversions during the stream, then the jobs
    ticks("tiled PV", st["pv.b_in"], "#a8e6cf")
    ax.text((st["pv.b_in"][0][0] + st["pv.b_in"][-1][1]) / 2, y["tiled PV"] + 0.42, f"{CONVERT_MATH['tiled PV']} per token as it arrives",
            ha="center", va="bottom", fontsize=8, color="#007a56")
    for a, b, u in st["pv.jobs"]:
        box("tiled PV", a, b, f"$O_{u} = P_{u}\\,V_{u}$\nV rows {u * bkv}..{u * bkv + bkv - 1}", LANE_COLOR["pv"])
    # output rescale: the blocks (two in flight: alternate rows), the flush
    for a, b, u in st["acc.blk"]:
        box("output rescale", a, b, f"$o \\leftarrow \\alpha\\,o + O_{u}$", LANE_COLOR["acc"], dy=0.17 if u % 2 == 0 else -0.17, h=0.32, fs=7.8)
    for a, b, u in st["acc.run"]:
        box("output rescale", a, b, "$O = o / l$", "#8c4a73", dy=-0.17, h=0.32, fs=7.8)
    # what waits on what
    job0 = st["qk.jobs"][0][0]
    arrow(streamEnd, "projections", job0, "tiled QK", f"the first score job needs the last token's query: starts at {job0}", dx=30)
    arrow(job0, "tiled QK", st["sm.tile"][0][0], "online softmax", "scores stream into the tile", dx=40)
    arrow(st["sm.blk"][0][0], "online softmax", st["pv.jobs"][0][0], "tiled PV", "probabilities", dx=40)
    arrow(st["pv.jobs"][0][0], "tiled PV", st["acc.blk"][0][0], "output rescale", "partial outputs", dx=40)
    ax.axvline(streamEnd, color="#555555", lw=1, ls="--", zorder=1)
    ax.text(streamEnd - 8, len(lanes) - 0.45, "stream ends", ha="right", va="center", fontsize=8.5, color="#555555")
    ax.axvspan(streamEnd, T, color="#f3f3f3", zorder=0)
    ax.text((streamEnd + T) / 2, len(lanes) - 0.45,
            f"after the stream: block 0, then block 1, through four stages in turn, then the tails ({T - streamEnd} cycles)",
            ha="center", va="center", fontsize=8.5, color="#444444")
    ax.set_yticks(range(len(lanes)), lane_labels(lanes), fontsize=9.5)
    ax.set_ylim(-0.95, len(lanes) - 0.15)
    ax.set_xlim(0, T * 1.01)
    ax.set_xlabel("cycle")
    ax.grid(axis="x", color="#e5e5e5", lw=0.6, zorder=0)
    ax.set_title(f"{case['name']} (n {n}, d_model {case['dm']}, d {case['d']}) at {cfg.replace('x', '×')}: what each stage "
                 f"computes, and what it waits for. S = {S} queries per run, {c['runs']} run, {blocks} key blocks of {bkv}; T = {T:,} cycles",
                 fontsize=10.5)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_simple(ax, case: dict, cfg: str, xmax: int) -> None:
    """One head at one array on `ax`: one box per block per stage, the stream, the wait for the run's queries, the
    blocks through the four stages, the tails. Blocks are labelled when there are a few, and the later of two
    overlapping blocks is drawn lighter; with many blocks, one colour per lane and the runs labelled instead.
    `xmax` is the shared time axis (the inch widths of the labels follow it)."""
    d, c = case["diagrams"][cfg], case["configs"][cfg]
    st, T, n, S, blocks, runs = d["steps"], d["T"], case["n"], c["S"], c["blocks"], c["runs"]
    bkv = n // blocks
    few = runs * blocks <= 4
    lanes = ["projections", "scores (tiled QK)", "online softmax", "tiled PV", "output rescale"]
    y = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}

    def box(lane, a, b, text, color, fs=9.5, alpha=1.0, tx=None):
        ax.broken_barh([(a, max(b - a, 1))], (y[lane] - 0.3, 0.6), color=color, lw=0, alpha=alpha, zorder=3)
        if text:
            ax.text((a + b) / 2 if tx is None else tx, y[lane], text, ha="center", va="center", fontsize=fs, color="white", zorder=4)

    def blocks_of(lane, items, color, math):
        """Consecutive blocks, which may overlap in time: each label on its free part: the block's equation when that
        part is wide enough, else just its number."""
        items = sorted(items)
        for i, (a, b, u) in enumerate(items):
            nxt = items[i + 1][0] if i + 1 < len(items) else b
            prv = items[i - 1][1] if i else a
            lo, hi = (a, min(nxt, b)) if i % 2 == 0 else (max(prv, a), b)
            wide = (hi - lo) * 18 / xmax >= 1.3                          # inches of free box
            label = "" if not few else (f"block {u}\n" + math(u) if wide else f"block {u}")
            box(lane, a, b, label, color, fs=9, alpha=0.72 if (few and i % 2) else 1.0, tx=(lo + hi) / 2)

    proj = st["gq0.rows"]
    streamEnd = max(b for _, b, _ in proj)
    Tp = int(round(c["T_model"]["T_projection_token"]))                # the token period: the load or the drain, whichever is longer
    inches = lambda a, b: (b - a) * 18 / xmax                         # noqa: E731  the width of a span on the figure
    wide = inches(proj[0][0], streamEnd)
    box("projections", proj[0][0], streamEnd,
        f"project all {n} tokens (Q, K and V), one token every {Tp} cycles" if wide >= 6 else
        f"{n} tokens × {Tp} cycles" if wide >= 1.6 else "", LANE_COLOR["gemm"])
    job0 = st["qk.jobs"][0][0]
    for lane, key, color in (("scores (tiled QK)", "qk.b_in", "#a9d8f3"), ("tiled PV", "pv.b_in", "#a8e6cf")):
        a, b = st[key][0][0], st[key][-1][1]
        ax.broken_barh([(a, b - a)], (y[lane] - 0.12, 0.24), color=color, zorder=2)
        conv = CONVERT_MATH[stage_of(lane)]
        free = (a, b) if few else (a, min(b, job0))
        if inches(*free) >= 1.2:                                      # room for the label; the caption says it otherwise
            ax.text(sum(free) / 2, y[lane], f"{conv} per token, in step with the stream" if few else conv,
                    ha="center", va="center", fontsize=8.5, color="#333333", zorder=4)
    blocks_of("scores (tiled QK)", st["qk.jobs"], LANE_COLOR["qk"], lambda u: f"$S_{u} = Q\\,K_{u}^{{\\top}}$")
    blks = {u: b for _, b, u in st["sm.blk"]}
    blocks_of("online softmax", [(a, blks[u], u) for a, _, u in st["sm.tile"]], LANE_COLOR["sm"],
              lambda u: f"$P_{u} = e^{{S_{u} - m}}$, $m$, $l$")
    for a, b, u in st["sm.run"]:
        box("online softmax", a, b, "$\\log l$" if few else "", "#8a5f00", fs=8.5)
    blocks_of("tiled PV", st["pv.jobs"], LANE_COLOR["pv"], lambda u: f"$O_{u} = P_{u}\\,V_{u}$")
    blocks_of("output rescale", st["acc.blk"], LANE_COLOR["acc"], lambda u: f"$o \\leftarrow \\alpha\\,o + O_{u}$")
    accEnd = {}                                                       # each run's last block step end
    for a, b, u in st["acc.blk"]:
        accEnd[u // blocks] = max(accEnd.get(u // blocks, 0), b)
    for a, b, u in st["acc.run"]:                                     # the normalization overlaps the last block's waves:
        lo = max(a, accEnd.get(u, a))                                 # its label goes on the free part
        label = "$O = o/l$" if few and (b - lo) * 18 / xmax >= 0.55 else ""
        box("output rescale", a, b, label, "#8c4a73", fs=8.5, tx=(lo + b) / 2)
    # the first score block waits for its run's S queries: the last token when S = n, else the S-th token
    gate = streamEnd if runs == 1 else next(b for _, b, u in proj if u == S - 1)
    ax.annotate("", xy=(job0, y["scores (tiled QK)"] + 0.3), xytext=(gate, y["projections"] - 0.3),
                arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2, shrinkA=0, shrinkB=0), zorder=5)
    ax.text(job0 + xmax * 0.01, (y["projections"] + y["scores (tiled QK)"]) / 2,
            "the first score block needs the last token's query" if runs == 1 else
            f"the first score block needs its run's {S} queries: the first {S} tokens",
            ha="left", va="center", fontsize=9, zorder=6)
    ax.axvline(streamEnd, color="#555555", lw=1, ls="--", zorder=1)
    ax.axvspan(streamEnd, T, color="#f3f3f3", zorder=0)
    top = len(lanes) + 0.05
    ax.text(streamEnd / 2, top, f"the stream: {n} tokens × {Tp} cycles", ha="center", va="center", fontsize=9.5, color="#444444")
    fits = lambda text, a, b, pt: len(text) * pt * 0.0095 <= inches(a, b)   # noqa: E731  a rough text width in inches
    if runs == 1:
        long = f"after the stream: each block through the four stages in turn, then the tails ({T - streamEnd:,} cycles)"
        short = f"after the stream: {T - streamEnd:,} cycles"
        ax.text((streamEnd + T) / 2, top, long if fits(long, streamEnd, T, 9.5) else short, ha="center", va="center",
                fontsize=9.5, color="#444444")
    else:                                                             # the runs, bracketed above the lanes
        brk = len(lanes) - 0.35
        starts = [st["qk.jobs"][r * blocks][0] for r in range(runs)]
        for r, (a, b) in enumerate(zip(starts, d["run_ends"])):
            ax.plot([a, a, b, b], [brk - 0.08, brk, brk, brk - 0.08], color="#444444", lw=1.1, zorder=5)
            note = " (overlaps the stream)" if a < streamEnd else ""
            label = next(t for t in (f"run {r}: {blocks} blocks{note}", f"run {r}{note}", f"run {r}") if fits(t, a, b, 8.5))
            ax.text((a + b) / 2, brk + 0.03, label, ha="center", va="bottom", fontsize=8.5, color="#444444", zorder=6)
    ax.set_yticks(range(len(lanes)), lane_labels(lanes), fontsize=9.5)
    ax.set_ylim(-0.6, len(lanes) + 0.3)
    ax.set_xlim(0, xmax * 1.01)
    ax.set_xlabel("cycle")
    ax.grid(axis="x", color="#e5e5e5", lw=0.6, zorder=0)
    ax.set_title(f"{case['name']} (n {n}, d {case['d']}) at {cfg.replace('x', '×')}: {runs} run{'s' if runs > 1 else ''} of "
                 f"S = {S} queries, each over the {n} keys in {blocks} blocks of {bkv}, one block after another; T = {T:,} cycles",
                 fontsize=11)


def simple(case: dict, cfgs: list, out: Path) -> None:
    """stages_simple_<case>.png: one head at the DIAGRAMS arrays, one subplot each, stacked on a shared time axis so
    the scaling with the array is visible (the brief's "head in time" figures)."""
    fig, axs = plt.subplots(len(cfgs), 1, figsize=(18, 5.6 * len(cfgs)), constrained_layout=True, sharex=True)
    xmax = max(case["diagrams"][cfg]["T"] for cfg in cfgs)
    for ax, cfg in zip(np.atleast_1d(axs), cfgs):
        draw_simple(ax, case, cfg, xmax)
    fig.savefig(out, dpi=150)
    plt.close(fig)


EXTRA_COLOR = {"other": "#555555", "clb": "#9a9a9a", "memory": "#d4d4d4"}
EXTRA_NAME = {"other": "control", "clb": "soft logic (CLBs)", "memory": "memory blocks"}


def costs(res: dict, out: Path) -> None:
    cases, stages, names = res["cases"], res["stages"], res["stage_name"]
    fig, axs = plt.subplots(len(cases), 3, figsize=(22, 5.4 * len(cases)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for row, case in enumerate(cases):
        cfgs = sorted(case["configs"], key=lambda c: (int(c.split("x")[1]), int(c.split("x")[0])))
        nC = len({c.split("x")[1] for c in cfgs})
        x = np.arange(len(cfgs))
        head = f"{case['name']} (n {case['n']}, d_model {case['dm']}, d {case['d']})"
        cfg = lambda c: case["configs"][c]                                         # noqa: E731
        cyc = {s: np.array([cfg(c)["stages"][s]["path_cycles"] for c in cfgs], dtype=float) for s in stages}
        cyc["other"] = np.array([cfg(c)["T"] for c in cfgs], dtype=float) - sum(cyc[s] for s in stages)
        en = {s: np.array([cfg(c)["stages"][s]["energy_pj"] for c in cfgs]) / 1e3 for s in stages}
        ar = {s: np.array([cfg(c)["A_model"][s] for c in cfgs]) for s in stages}
        ar["clb"] = np.array([cfg(c)["A_model"]["clb"] for c in cfgs])
        ar["memory"] = np.array([cfg(c)["A_model"]["memory"] for c in cfgs])
        panels = ((cyc, "cycles on the critical path", "cycles by stage"), (en, "energy per inference (nJ)", "energy by stage"),
                  (ar, "area (mm²)", "area by stage"))
        for col, (data, ylabel, title) in enumerate(panels):
            ax = axs[row, col]
            bottom = np.zeros(len(cfgs))
            for key, vals in data.items():
                ax.bar(x, vals, bottom=bottom, width=0.72, color=LANE_COLOR.get(key, EXTRA_COLOR.get(key)),
                       label=names.get(key, EXTRA_NAME.get(key)), edgecolor="white", lw=0.4)
                bottom += vals
            for g in range(1, nC):                                                 # the C groups
                ax.axvline(g * len(cfgs) / nC - 0.5, color="#bbbbbb", lw=0.8, ls="--")
            ax.set_ylim(0, 1.22 * bottom.max())                                   # room for the legend
            ax.set_xticks(x, [c.replace("x", "×") for c in cfgs], rotation=45, ha="right", fontsize=8.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{head}: {title}", fontsize=10.5)
            ax.grid(axis="y", color="#e5e5e5", lw=0.6)
            ax.set_axisbelow(True)
            if col != 1:
                ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("The cost of every stage on the 12 arrays (grouped by C, R rising within a group): the simulator's "
                 "critical path by stage, the energy by stage, the area by stage", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    res = json.loads((STAGES_DIR / "stages.json").read_text())
    for case in res["cases"]:
        out = STAGES_DIR / f"stages_gantt_{tag(case['name'])}.png"
        gantt(case, out)
        print(f"wrote {out}")
    small = res["cases"][0]                                           # the smallest head, at the smallest array
    out = STAGES_DIR / f"stages_explain_{tag(small['name'])}.png"
    explain(small, res["extremes"][0], out)
    print(f"wrote {out}")
    for case in res["cases"]:                                        # both heads, the two DIAGRAMS arrays stacked
        out = STAGES_DIR / f"stages_simple_{tag(case['name'])}.png"
        simple(case, res["diagrams"], out)
        print(f"wrote {out}")
    for fn, name in ((sweep, "stages_sweep.png"), (costs, "stages_costs.png")):
        out = STAGES_DIR / name
        fn(res["cases"] if fn is sweep else res, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
