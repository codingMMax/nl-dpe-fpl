#!/usr/bin/env python3
"""pipelines.py — the page "Crossbar pipelines" (for sharing): every operation of one attention head as crossbar
passes, each pass in its three stages, drawn and animated from the simulator's own pass events.

A crossbar pass (simulator/core/crossbar.py, the pass grammar) has three stages: the load (the input rows through the
40-bit port, ceil(8 rows / 40) cycles), the compute (the 8 bit-slices, then the shift-accumulate and the output stage:
10 cycles) and the read-out (every column through the port, ceil(8 C / 40) cycles). The next pass loads from the cycle
after the last bit-slice, so a pass's read-out runs under the next pass's load and compute, and the slower side sets
the pace: max(load + 8, read-out + 1) cycles per pass when the passes run back to back.

For the two case-study heads (space.py CASE_STUDY) at the DIAGRAMS arrays (stage_analysis.py: 128x64, 512x256) the
simulator is run (timing only) with every crossbar's passes recorded: the projection tiles, the LOG converters of the
DIMM ports, the QK and PV EXP crossbars, the softmax's EXP and carry crossbars, the accumulator's EXP and LOG groups.
The page has, per head:
  per operation   an animated board of the crossbar pipeline (input port, input buffer, array with its 8 bit-slices,
                  output buffer, output port; which pass sits where, cycle by cycle) for both arrays on one clock, and
                  the pipeline diagram under it (one row per pass, load / compute / read-out) with a cursor
  one key block   the full head's pipeline around one key block: every core crossbar group as two strips (input side,
                  output side), the block's passes in colour and its neighbours faint, with a cursor and a status line
  the core start  the whole head (JetFormer) or a window at the start of the core (ViT): when attention begins
  the table       the stage lengths
The board's state rule is implemented twice, in the page's script and in this file (`unitState`); the self-test drives
the page in headless Chrome and compares every board text with this file's. Output: dse/results/stages/pipelines.json,
dse/results/stages/pipelines.html (self-contained). Usage: python3 v3/dse/pipelines.py [--self-test] [--shot]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
import simulator.core.trace as trace  # noqa: E402
from simulator.core.cycleMath import ceilDiv  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402
from dse.dataflow import CHECK_DIR, FONTS, chrome  # noqa: E402
from dse.results_page import CSS as PAGE_CSS  # noqa: E402
from dse.space import CASE_STUDY, SHAPES  # noqa: E402
from dse.stage_analysis import DIAGRAMS  # noqa: E402
from dse.stages_page import cx  # noqa: E402

STAGES_DIR = V3 / "dse" / "results" / "stages"
OUT_HTML = STAGES_DIR / "pipelines.html"
OUT_JSON = STAGES_DIR / "pipelines.json"
TIME_PAGE = "https://claude.ai/artifact/7ujYBp7ScQyLaofuaNK8xc"
BUF, P, EPS = 40, 8, 5
KEYS = ("load_start", "load_end", "compute_start", "msb_fire", "acam_done", "drain_start", "drain_end")
SHORT = dict(zip(KEYS, ("ls", "le", "cs", "mf", "ad", "ds", "de")))
EV = ("ls", "le", "cs", "mf", "ad", "ds", "de")
OPS = ("proj", "log", "qk", "smExp", "smCexp", "smClog", "pv", "accExp", "accLog")
WINDOW = 3000          # the ViT core-start windows: cycles shown at both arrays (one scale)
LEAD = 340             # a core-start window starts this many cycles before QK^T's first pass


# -- the simulator side ----------------------------------------------------------------------------------------------
def capture(xbar, sink: list) -> None:
    """Record every pass of a crossbar (group): its events, as ints, in call order."""
    run0 = xbar.run

    def wrapped(x, tReady):
        p = run0(x, tReady)
        sink.append({SHORT[k]: int(p.events[k]) for k in KEYS})
        return p
    xbar.run = wrapped


def collect(n: int, dm: int, d: int, R: int, C: int) -> dict:
    """One head at one array: every crossbar pass per operation (cycles from the head's start) with its labels, the
    counts and the extents."""
    trace.enable(False)
    hp = headParams(n, dm, d, R, C)
    if hp.P_PROJ != 1 or hp.N_EXP != 2:
        raise ValueError("the page draws the fixed mapping: P_PROJ = 1, N_EXP = 2")
    g, parts = attentionHead(hp, withValues=False)
    rec = {k: [] for k in ("gq", "gk", "gv", "lq", "lk", "lv", "qk", "pv", "sm0", "sm1", "cexp", "clog")}
    for kind in "qkv":
        capture(parts["gemm"][f"g{kind}0"].tiles, rec[f"g{kind}"])
    capture(parts["qk"].converterA[0], rec["lq"])
    capture(parts["qk"].converterB[0], rec["lk"])
    capture(parts["pv"].converterB[0], rec["lv"])
    capture(parts["qk"].farm, rec["qk"])
    capture(parts["pv"].farm, rec["pv"])
    capture(parts["sm"].exp[0], rec["sm0"])
    capture(parts["sm"].exp[1], rec["sm1"])
    capture(parts["sm"].carryExp, rec["cexp"])
    capture(parts["sm"].carryLog, rec["clog"])
    run(g)
    ctrl, acc, sm = parts["ctrl"], parts["acc"], parts["sm"]
    t0 = int(ctrl.tStart)
    rel = lambda e: {k: v - t0 for k, v in e.items()}           # noqa: E731
    # lock-step: the three projection GEMMs, the three converters, the two softmax EXP crossbars run the same passes
    lockstep = dict(proj=rec["gq"] == rec["gk"] == rec["gv"], log=rec["lq"] == rec["lk"] == rec["lv"],
                    smExp=rec["sm0"] == rec["sm1"])
    B, runs, S, I = hp.B, hp.RB, hp.S, hp.I
    ops = {}
    ops["proj"] = [dict(rel(e), tok=i) for i, e in enumerate(rec["gq"])]
    ops["log"] = [dict(rel(e), tok=i) for i, e in enumerate(rec["lq"])]
    for key, tile in (("qk", parts["qk"]), ("pv", parts["pv"])):
        TW = tile.d.T_W
        ops[key] = [dict(rel(e), run=(i // TW) // B, blk=(i // TW) % B, wave=i % TW) for i, e in enumerate(rec[key])]
    perBlock = len(rec["sm0"]) // (runs * B)
    ops["smExp"] = [dict(rel(e), run=(i // perBlock) // B, blk=(i // perBlock) % B, round=i % perBlock)
                    for i, e in enumerate(rec["sm0"])]
    ops["smCexp"] = [dict(rel(e), run=i // (B - 1), blk=1 + i % (B - 1)) for i, e in enumerate(rec["cexp"])]
    ops["smClog"] = [dict(rel(e), run=i // B, blk=i % B) for i, e in enumerate(rec["clog"])]
    jobs = sorted(((k, rel({SHORT[x]: int(e[x]) for x in KEYS})) for k, e in acc.jobs.items()), key=lambda kv: kv[1]["ls"])
    ops["accExp"] = [dict(e, run=k[0], blk=k[2], wave=k[3], kind="normalize" if k[1] == "epilogue" else "rescale")
                     for k, e in jobs if k[1] != "log"]
    ops["accLog"] = [dict(e, run=k[0], blk=k[2], wave=k[3]) for k, e in jobs if k[1] == "log"]
    # extents (rows loaded per pass, from the crossbar objects) and the crossbar counts of each operation
    rows = dict(proj=parts["gemm"]["gq0"].tiles.rowsUsed, log=parts["qk"].converterA[0].rowsUsed,
                qk=parts["qk"].farm.rowsUsed, smExp=sm.exp[0].rowsUsed, smCexp=sm.carryExp.rowsUsed,
                smClog=sm.carryLog.rowsUsed, pv=parts["pv"].farm.rowsUsed, accExp=acc.expGroup.rowsUsed,
                accLog=acc.logGroup.rowsUsed)
    if not (parts["qk"].converterB[0].rowsUsed == parts["pv"].converterB[0].rowsUsed == rows["log"]):
        raise ValueError("the q, k, v converters load different extents")
    count = dict(proj=3 * hp.V, log=3, qk=hp.N_E, smExp=hp.N_EXP, smCexp=1, smClog=1, pv=hp.N_EP, accExp=hp.N_OC,
                 accLog=hp.N_OC)
    stage = {k: dict(rows=rows[k], L=ceilDiv(8 * rows[k], BUF), D=ceilDiv(8 * C, BUF)) for k in OPS}
    for s in stage.values():
        s["own"] = max(s["L"] + P, s["D"] + 1)
        s["bound"] = "load" if s["L"] + P >= s["D"] + 1 else "read"
    return dict(n=n, dm=dm, d=d, R=R, C=C, I=I, S=S, runs=runs, blocks=B, V=hp.V, N_E=hp.N_E, N_EP=hp.N_EP,
                N_EXP=hp.N_EXP, N_OC=hp.N_OC, T=int(ctrl.done) - t0, TW_QK=parts["qk"].d.T_W,
                TW_PV=parts["pv"].d.T_W, W_ACC=acc.d.W, lockstep=lockstep, count=count, stage=stage, ops=ops,
                projEnd=max(p["de"] for p in ops["proj"]), qkStart=min(p["ls"] for p in ops["qk"]),
                # the run's last query through its converter: run 0 can start no earlier (its S queries)
                firstRunReady=ops["log"][S - 1]["de"])


def collectAll() -> dict:
    byName = {s[0]: s for s in SHAPES}
    heads = []
    for name in CASE_STUDY:
        _, n, dm, d = byName[name]
        heads.append(dict(name=name, n=n, dm=dm, d=d,
                          arrays={f"{R}x{C}": collect(n, dm, d, R, C) for R, C in DIAGRAMS}))
    return dict(arrays=[f"{R}x{C}" for R, C in DIAGRAMS], heads=heads)


# -- what each chart shows -------------------------------------------------------------------------------------------
CHART_OPS = (("proj", "Projection", ("proj",)), ("log", "LOG conversion", ("log",)),
             ("dimm", "DIMM: QKᵀ (PV is the same passes)", ("qk",)),
             ("softmax", "Online softmax", ("smExp", "smCexp", "smClog")),
             ("rescale", "Output rescale", ("accExp", "accLog")))


def rowLabel(u: str, p: dict) -> str:
    """A pass's name, the same on the per-operation chart and on the boards."""
    if u in ("proj", "log"):
        return f"token {p['tok'] + 1}"
    b = f"block {p['blk'] + 1}"
    return {"qk": f"{b} wave {p.get('wave', 0) + 1}", "pv": f"{b} wave {p.get('wave', 0) + 1}",
            "smExp": f"{b} round {p.get('round', 0) + 1}", "smCexp": f"{b} carry EXP", "smClog": f"{b} carry LOG",
            "accExp": (f"normalize wave {p.get('wave', 0) + 1}" if p.get("kind") == "normalize"
                       else f"{b} EXP wave {p.get('wave', 0) + 1}"),
            "accLog": f"{b} LOG wave {p.get('wave', 0) + 1}"}[u]


def pick(op: str, a: dict) -> list:
    """(row label, pass, crossbar group) of the per-operation chart: a few consecutive passes in the steady state."""
    ps = a["ops"]
    if op in ("proj", "log"):
        rows = [(p, op) for p in ps[op][8:12]]
    elif op == "dimm":                                # QK^T: the end of key block 1 and the start of key block 2
        b0 = [p for p in ps["qk"] if p["run"] == 0 and p["blk"] == 0]
        b1 = [p for p in ps["qk"] if p["run"] == 0 and p["blk"] == 1]
        rows = [(p, "qk") for p in b0[-2:] + b1[:2]]
    elif op == "softmax":                             # key block 2: its EXP rounds, then the two carry passes
        rows = [(p, u) for u in ("smExp", "smCexp", "smClog") for p in ps[u] if p["run"] == 0 and p["blk"] == 1]
    elif op == "rescale":                             # key block 2: its waves (the first 3 of more than 4), both groups
        k = a["W_ACC"] if a["W_ACC"] <= 4 else 3
        rows = sorted([(p, "accExp") for p in ps["accExp"]
                       if p["run"] == 0 and p["blk"] == 1 and p["kind"] == "rescale" and p["wave"] < k]
                      + [(p, "accLog") for p in ps["accLog"] if p["run"] == 0 and p["blk"] == 1 and p["wave"] < k],
                      key=lambda r: r[0]["ls"])
    else:
        raise ValueError(op)
    return [(rowLabel(u, p), p, u) for p, u in rows]


def rowsSpan(rows: list) -> int:
    return max(r[1]["de"] for r in rows) + 1 - rows[0][1]["ls"]


def segments(p: dict) -> list:
    """The stages of one pass as (stage, start, end), end exclusive: the load, a wait for the array, the 8 bit-slices
    and the shift-accumulate, a wait for the output port, the output stage, the read-out."""
    out = [("load", p["ls"], p["le"] + 1)]
    if p["cs"] > p["le"] + 1:
        out.append(("wait", p["le"] + 1, p["cs"]))
    if p["ad"] == p["mf"] + 2:
        out.append(("compute", p["cs"], p["ad"] + 1))
    else:
        out += [("compute", p["cs"], p["mf"] + 2), ("wait", p["mf"] + 2, p["ad"]), ("compute", p["ad"], p["ad"] + 1)]
    out.append(("read", p["ds"], p["de"] + 1))
    return out


def fmt(v: int) -> str:
    return f"{v:,}"


def describe(op: str, a: dict) -> str:
    """The caption bullet of one array in one operation's chart: the counts, the stage lengths, the pace."""
    st, c = a["stage"], a["count"]
    cfg = cx(f"{a['R']}x{a['C']}")
    if op == "proj":
        s = st["proj"]
        per = a["V"]
        where = (f"{c['proj']} crossbars, {per} per weight matrix, so each loads {s['rows']} of the token's {a['dm']} "
                 f"numbers" if per > 1 else
                 f"{c['proj']} crossbars, one per weight matrix, so all {s['rows']} of the token's numbers go through "
                 f"one port")
        pace = (f"The read-out hides under the next token's load and compute: one token every {s['own']} cycles."
                if s["bound"] == "load" else
                f"The read-out is the longest stage and the next token waits for it: one token every {s['own']} cycles.")
        return f"<b>{cfg}.</b> {where}. Load {s['L']}, compute 10, read out {s['D']} (all {a['C']} columns). {pace}"
    if op == "log":
        s = st["log"]
        gap = a["ops"]["log"][9]["ls"] - a["ops"]["log"][8]["ls"]
        alone = (f"on its own it could start one every {s['own']} cycles, so it waits for the projection"
                 if gap > s["own"] else "exactly the pace it can sustain on its own")
        return (f"<b>{cfg}.</b> 3 converters (q, k, v), {s['rows']} numbers each. Load {s['L']}, compute 10, read out "
                f"{s['D']}. A new row arrives every {gap} cycles, with the next token: {alone}.")
    if op == "dimm":
        s = st["qk"]
        rows = pick("dimm", a)
        pairs = list(zip(rows, rows[1:]))
        inner = [q[1]["ls"] - p[1]["ls"] for p, q in pairs if q[1]["blk"] == p[1]["blk"]]
        cross = [q[1]["ls"] - p[1]["ls"] for p, q in pairs if q[1]["blk"] != p[1]["blk"]]
        hidden = "the read-out is hidden" if s["bound"] == "load" else "the read-out sets the pace"
        waves = (f"A key block is {a['TW_QK']} waves of all {c['qk']} EXP crossbars in lock-step"
                 if a["TW_QK"] > 1 else f"A key block is one wave of all {c['qk']} EXP crossbars")
        within = f" Within a block the waves run back to back, one every {inner[0]} cycles: {hidden}." if inner else ""
        between = (f" A new block starts only on idle crossbars, so the last read-out of a block is not hidden: the "
                   f"next block's first wave starts {cross[0]} cycles after the last one, against {s['own']} back to "
                   f"back." if cross else "")
        return (f"<b>{cfg}.</b> {waves}, {s['rows']} numbers each. Load {s['L']}, compute 10, read out {s['D']}."
                f"{within}{between}")
    if op == "softmax":
        s, k = st["smExp"], st["smClog"]
        rounds = sum(1 for p in a["ops"]["smExp"] if p["run"] == 0 and p["blk"] == 1)
        pace = f", back to back every {s['own']} cycles" if rounds > 1 else ""
        plural = "s" if rounds > 1 else ""
        return (f"<b>{cfg}.</b> The block's scores go through the 2 EXP crossbars in {rounds} round{plural}, "
                f"{s['rows']} numbers each: load {s['L']}, compute 10, read out {s['D']}{pace}. Then the carry passes "
                f"update one running sum per query row ({k['rows']} numbers): load {k['L']}, compute 10, read out "
                f"{k['D']}, the LOG pass after the EXP pass.")
    if op == "rescale":
        s, W = st["accExp"], a["W_ACC"]
        shown = " (the chart shows the first three of each)" if W > 4 else ""
        return (f"<b>{cfg}.</b> {c['accExp']} EXP and {c['accLog']} LOG crossbars, {W} wave{'s' if W > 1 else ''} "
                f"per block{shown}, {s['rows']} numbers each: load {s['L']}, compute 10, read out {s['D']}. The EXP "
                f"passes rescale the running output once the softmax's α is ready. Each LOG pass then adds the "
                f"block's PV output and takes the log again, after its EXP pass and after PV's output has arrived.")
    raise ValueError(op)


CAPTION = {
    "proj": "One pass per token. A bigger crossbar has the same 40-bit port, so a token that needs fewer crossbars "
            "loads through fewer ports at once, and a wider crossbar reads out more columns.",
    "log": "The LOG converters put q, k and v into the log domain. Each converts one token's row per pass.",
    "dimm": "QKᵀ in the log domain: an add of two logs, then EXP on identity crossbars, then an adder tree. PV runs "
            "the same passes on its own EXP crossbars.",
    "softmax": "The softmax's EXP crossbars turn the scores into probabilities; its carry crossbars keep each query "
               "row's running sum in the log domain.",
    "rescale": "After each key block the running output is rescaled (EXP) and converted back to the log domain (LOG).",
}
# a crossbar group on the board: its name, what goes in, what comes out
UNIT = {"proj": ("projection crossbars", "token x", "q, k, v rows"),
        "log": ("LOG converters, q, k, v", "q, k, v rows", "log q, log k, log v"),
        "qk": ("EXP crossbars", "log q + log k", "adder tree: scores"),
        "smExp": ("EXP crossbars", "s − m", "P, row sums"),
        "smCexp": ("carry EXP crossbar", "log α + log l", "α · l"),
        "smClog": ("carry LOG crossbar", "α · l + row sum", "log l"),
        "pv": ("EXP crossbars", "log P + log v", "adder tree: O"),
        "accExp": ("EXP crossbars", "log α + lo", "α · o"),
        "accLog": ("LOG crossbars", "α · o + PV out", "lo = log o")}


# -- the board's state rule (the page's script implements the same rule; the self-test compares the two) -------------
def unitState(passes: list, rows: int, C: int, t: int) -> tuple:
    """What one crossbar group holds at cycle t: the input buffer (loading, loaded and waiting, read by the
    bit-slices), the array (bit b of 8, finishing, the result held while the output port is busy), the output buffer
    (reading out). passes: [ls, le, cs, mf, ad, ds, de, label]."""
    inp = arr = out = None
    for ls, le, cs, mf, ad, ds, de, label in passes:
        if ls <= t <= mf:
            inp = ("loading", label, min(rows, EPS * (t - ls + 1))) if t <= le else \
                (("waiting", label, 0) if t < cs else ("read", label, 0))
        if cs <= t <= ad:
            arr = ("bit", label, t - cs + 1) if t <= mf else \
                (("finishing", label, 0) if t in (mf + 1, ad) else ("held", label, 0))
        if ds <= t <= de:
            out = ("reading", label, min(C, EPS * (t - ds + 1)))
    return inp, arr, out


def stateTexts(passes: list, rows: int, C: int, t: int) -> tuple:
    """The board's four texts of one group at cycle t: the input buffer, the pass in the array, the array's state,
    the output buffer."""
    inp, arr, out = unitState(passes, rows, C, t)
    tin = "" if not inp else (f"{inp[1]} · loading {inp[2]} / {rows}" if inp[0] == "loading" else
                              f"{inp[1]} · waits for the array" if inp[0] == "waiting" else
                              f"{inp[1]} · bits read by the array")
    tarr = "" if not arr else (f"bit {arr[2]} of 8" if arr[0] == "bit" else
                               "finishing" if arr[0] == "finishing" else "held: output busy")
    tout = "" if not out else f"{out[1]} · read out {out[2]} / {C}"
    return tin, (arr[1] if arr else ""), tarr, tout


def laneStatus(name: str, passes: list, t: int) -> str:
    """One lane of the key-block chart at cycle t: the passes on its input side and on its output side.
    passes: [ls, le, cs, mf, ad, ds, de, label, mine]."""
    parts = []
    for ls, le, cs, mf, ad, ds, de, label, _ in passes:
        if ls <= t <= mf:
            parts.append(f"{label} {'loading' if t <= le else 'waiting' if t < cs else 'computing'}")
        if ds <= t <= de:
            parts.append(f"{label} reading out")
    return f"{name}: {', '.join(parts)}" if parts else ""


# -- the data the page's script animates -----------------------------------------------------------------------------
BOARD_IN_W, BOARD_OUT_W = 380, 262          # the board's input and output buffer slots (px)


def opSpec(head: dict, op: str, units: tuple) -> dict:
    """One operation's boards: per array its crossbar groups, each with every pass that overlaps the clock's span
    (cycles after the array's first charted load: the boards show what the crossbars really hold, the chart follows
    a few of the passes), the clock span, the starting cycle, the px per number of both buffers."""
    arrays = []
    rowsMax = max(head["arrays"][c]["stage"][u]["rows"] for c in head["arrays"] for u in units)
    cMax = max(head["arrays"][c]["C"] for c in head["arrays"])
    scale = min(BOARD_IN_W / rowsMax, BOARD_OUT_W / cMax)
    picks = {cfg: pick(op, a) for cfg, a in head["arrays"].items()}
    span = max(rowsSpan(r) for r in picks.values())
    first = picks[next(iter(picks))]                  # start just after the first array's second pass starts loading
    start = first[1][1]["ls"] - first[0][1]["ls"] + 3 if len(first) > 1 else 3
    for cfg, a in head["arrays"].items():
        t0 = picks[cfg][0][1]["ls"]
        groups = []
        for u in units:
            name, inLabel, outLabel = UNIT[u]
            ps = [[p[k] - t0 for k in EV] + [rowLabel(u, p)] for p in a["ops"][u]
                  if p["de"] >= t0 and p["ls"] <= t0 + span]
            groups.append(dict(key=u, name=name, count=a["count"][u], rows=a["stage"][u]["rows"], C=a["C"],
                               L=a["stage"][u]["L"], D=a["stage"][u]["D"], inLabel=inLabel, outLabel=outLabel,
                               passes=ps))
        arrays.append(dict(cfg=cfg, R=a["R"], C=a["C"], t0=t0, units=groups))
    return dict(span=span, start=start, scale=scale, arrays=arrays)


JOURNEY_LANES = (("qk", "QKᵀ EXP"), ("smExp", "softmax EXP"), ("smCexp", "softmax carry EXP"),
                 ("smClog", "softmax carry LOG"), ("pv", "PV EXP"), ("accExp", "rescale EXP"),
                 ("accLog", "rescale LOG"))


def chosenBlock(a: dict) -> tuple:
    """(run, block) the key-block chart follows: the middle key block of the last run (the steady state, away from
    the run's start and end; JetFormer's two blocks: the second)."""
    return a["runs"] - 1, a["blocks"] // 2


def passLabel(key: str, p: dict) -> str:
    b = f"block {p['blk'] + 1}"
    if key in ("qk", "pv"):
        return f"{b} wave {p['wave'] + 1}"
    if key == "smExp":
        return f"{b} round {p['round'] + 1}"
    if key == "accExp" and p["kind"] == "normalize":
        return f"normalize wave {p['wave'] + 1}"
    if key in ("accExp", "accLog"):
        return f"{b} wave {p['wave'] + 1}"
    return b


def journey(a: dict) -> dict:
    """The key-block chart of one array: the window around the chosen block's passes, and per core lane every pass
    in it [ls..de relative to the window, label, mine]."""
    run_, blk = chosenBlock(a)
    mine = lambda p: p.get("run") == run_ and p.get("blk") == blk     # noqa: E731
    own = [p for k, _ in JOURNEY_LANES for p in a["ops"][k] if mine(p)]
    lo, hi = min(p["ls"] for p in own), max(p["de"] for p in own)
    margin = max(30, (hi - lo) // 12)
    w0, w1 = max(0, lo - margin), min(a["T"], hi + margin)
    lanes = []
    for key, name in JOURNEY_LANES:
        ps = [[p[k] - w0 for k in EV] + [passLabel(key, p), bool(mine(p))]
              for p in a["ops"][key] if p["de"] >= w0 and p["ls"] <= w1]
        lanes.append(dict(key=key, name=name, count=a["count"][key], passes=ps))
    first = min(p["ls"] for p in a["ops"]["smExp"] if mine(p))
    return dict(cfg=f"{a['R']}x{a['C']}", w0=w0, w1=w1, run=run_, blk=blk, lanes=lanes, start=first - w0 + 5)


def pageData(res: dict) -> dict:
    heads = []
    for head in res["heads"]:
        ops = {op: opSpec(head, op, units) for op, _, units in CHART_OPS}
        js = [journey(a) for a in head["arrays"].values()]
        heads.append(dict(name=head["name"], ops=ops,
                          journey=dict(span=max(j["w1"] - j["w0"] for j in js), start=js[0]["start"], arrays=js)))
    return dict(heads=heads)


# -- drawing ---------------------------------------------------------------------------------------------------------
W, LEFT, RIGHT = 1400, 176, 24
PLOT = W - LEFT - RIGHT
STAGE_CLASS = {"load": "s-load", "compute": "s-compute", "read": "s-read", "wait": "s-wait"}
STAGE_WORD = {"load": "load", "compute": "compute", "read": "read out", "wait": "waiting"}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def niceStep(span: float, target: int = 10) -> int:
    raw = span / target
    for m in (1, 2, 2.5, 5, 10):
        for e in range(0, 7):
            step = m * 10 ** e
            if step >= raw:
                return int(step)
    return int(raw)


def axis(y: float, x0: float, scale: float, t0: int, span: float, top: float, label: str, absolute: bool) -> list:
    """Ticks and hairline gridlines from `top` down to the axis at y."""
    out = []
    step = niceStep(span)
    first = (t0 // step + (1 if t0 % step else 0)) * step if absolute else 0
    v = first
    while v <= t0 + span if absolute else v <= span:
        x = x0 + ((v - t0) if absolute else v) * scale
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax" x="{x:.1f}" y="{y + 15:.1f}" text-anchor="middle">{fmt(v)}</text>')
        v += step
    out.append(f'<text class="ax" x="{x0 + span * scale:.1f}" y="{y + 31:.1f}" text-anchor="end">{esc(label)}</text>')
    return out


def cursor(top: float, bottom: float) -> str:
    return f'<line class="cursor" x1="{LEFT}" y1="{top:.1f}" x2="{LEFT}" y2="{bottom:.1f}"/>'


def opChart(head: dict, op: str) -> str:
    """One operation at both arrays, one row per pass, on one cycle scale (cycles after the first load), with a
    cursor per panel that the page's script moves."""
    cfgs = list(head["arrays"])
    picks = {cfg: pick(op, head["arrays"][cfg]) for cfg in cfgs}
    span = max(rowsSpan(r) for r in picks.values())
    scale = PLOT / span
    ROW, BAR, HEAD, GAP = 21, 13, 40, 14
    y, body, panels = 8, [], []
    for cfg in cfgs:
        a, rows = head["arrays"][cfg], picks[cfg]
        st = a["stage"][rows[0][2]]
        t0 = rows[0][1]["ls"]
        body.append(f'<text class="hd" x="{LEFT}" y="{y + 15}">{esc(cx(cfg))}</text>')
        body.append(f'<text class="mut" x="{LEFT + 92}" y="{y + 15}">load {st["L"]} · compute 10 · read out {st["D"]} '
                    f'cycles per pass</text>')
        y += HEAD - 14
        top = y
        for label, p, _ in rows:
            body.append(f'<text class="rl" x="{LEFT - 10}" y="{y + BAR - 2}" text-anchor="end">{esc(label)}</text>')
            body.append(f'<g><title>{esc(label)}: {", ".join(f"{STAGE_WORD[s]} {e - b}" for s, b, e in segments(p))} '
                        f'cycles (head cycles {fmt(p["ls"])}–{fmt(p["de"])})</title>')
            for s, b, e in segments(p):
                x = LEFT + (b - t0) * scale
                body.append(f'<rect class="{STAGE_CLASS[s]}" data-solid="bar" x="{x:.2f}" y="{y}" '
                            f'width="{max(0.8, (e - b) * scale):.2f}" height="{BAR}"/>')
            body.append("</g>")
            y += ROW
        panels.append((top, y))
        y += GAP
    yAxis = y - GAP + 4
    grid = axis(yAxis, LEFT, scale, 0, span, panels[0][0] - 4, "cycles after the first pass starts loading", False)
    cursors = [cursor(top - 4, bottom - 4) for top, bottom in panels]
    H = yAxis + 40
    return (f'<svg class="pipe" viewBox="0 0 {W} {H}" data-left="{LEFT}" data-scale="{scale:.6f}" role="img" '
            f'aria-label="{esc(op)} passes at both arrays">' + "".join(grid) + "".join(body) + "".join(cursors)
            + "</svg>")


def strips(p: dict, w0: int, w1: int, y: int, scale: float, STRIP: int, extra: str = "") -> list:
    """One pass as the input strip (load, wait, compute) and the output strip (read-out), clipped to [w0, w1]."""
    out = []
    parts = [("load", p["ls"], p["le"] + 1, 0)]
    if p["cs"] > p["le"] + 1:
        parts.append(("wait", p["le"] + 1, p["cs"], 0))
    parts += [("compute", p["cs"], p["mf"] + 2, 0), ("read", p["ds"], p["de"] + 1, 1)]
    for s, b, e, strip in parts:
        b, e = max(b, w0), min(e, w1 + 1)
        if e <= b:
            continue
        out.append(f'<rect class="{STAGE_CLASS[s]}{extra}" data-solid="bar" x="{LEFT + (b - w0) * scale:.2f}" '
                   f'y="{y + strip * (STRIP + 2)}" width="{max(0.6, (e - b) * scale):.2f}" height="{STRIP}"/>')
    return out


def journeyChart(head: dict) -> str:
    """The full head around one key block: every core crossbar group as two strips, the block's passes in colour and
    the other blocks' faint, both arrays on one scale (cycles from each window's start), with cursors."""
    js = [journey(a) for a in head["arrays"].values()]
    span = max(j["w1"] - j["w0"] for j in js)
    scale = PLOT / span
    STRIP, LANE = 7, 26
    y, body, panels = 8, [], []
    for j, a in zip(js, head["arrays"].values()):
        run_, blk = j["run"], j["blk"]
        which = f"key block {blk + 1}" + (f" of run {run_ + 1}" if a["runs"] > 1 else "")
        body.append(f'<text class="hd" x="{LEFT}" y="{y + 15}">{esc(cx(j["cfg"]))}</text>')
        body.append(f'<text class="mut" x="{LEFT + 92}" y="{y + 15}">{esc(which)} in colour, its neighbours faint · '
                    f'head cycles {fmt(j["w0"])}–{fmt(j["w1"])}</text>')
        y += 30
        top = y
        for lane, (key, _) in zip(j["lanes"], JOURNEY_LANES):
            body.append(f'<text class="rl" x="{LEFT - 10}" y="{y + STRIP + 4}" text-anchor="end">{esc(lane["name"])} '
                        f'×{lane["count"]}</text>')
            for p in a["ops"][key]:
                if p["de"] < j["w0"] or p["ls"] > j["w1"]:
                    continue
                mine = p.get("run") == run_ and p.get("blk") == blk
                body.extend(strips(p, j["w0"], j["w1"], y, scale, STRIP, "" if mine else " faint"))
            y += LANE
        bottom = y - LANE + 2 * STRIP + 2
        panels.append((top, bottom))
        body.extend(axis(bottom + 6, LEFT, scale, 0, span, top - 4, "cycles from the window's start", False))
        y = bottom + 56
    H = y - 10
    cursors = [cursor(top - 4, bottom + 2) for top, bottom in panels]
    return (f'<svg class="pipe journeysvg" viewBox="0 0 {W} {H}" data-left="{LEFT}" data-scale="{scale:.6f}" '
            f'role="img" aria-label="one key block through the head at both arrays">' + "".join(body)
            + "".join(cursors) + "</svg>")


LANES = (("proj", "Projection"), ("log", "LOG conversion"), ("qk", "QKᵀ EXP"), ("smExp", "softmax EXP"),
         ("smCexp", "softmax carry EXP"), ("smClog", "softmax carry LOG"), ("pv", "PV EXP"),
         ("accExp", "rescale EXP"), ("accLog", "rescale LOG"))


def window(head: dict, a: dict) -> tuple:
    """The cycles the core-start chart shows at one array: the whole head (JetFormer, short) or WINDOW cycles from
    LEAD before QK^T's first pass."""
    if head["name"].startswith("JetFormer"):
        return 0, a["T"]
    w0 = max(0, (a["qkStart"] - LEAD) // 100 * 100)
    return w0, min(a["T"], w0 + WINDOW)


def headChart(head: dict) -> str:
    """Every crossbar group of the head as two strips (the input side: load, compute; the output side: read-out)."""
    cfgs = list(head["arrays"])
    wins = {cfg: window(head, head["arrays"][cfg]) for cfg in cfgs}
    span = max(w1 - w0 for w0, w1 in wins.values())
    scale = PLOT / span
    STRIP, LANE, HEAD = 7, 26, 58
    y, body = 8, []
    for cfg in cfgs:
        a = head["arrays"][cfg]
        w0, w1 = wins[cfg]
        whole = (w0, w1) == (0, a["T"])
        body.append(f'<text class="hd" x="{LEFT}" y="{y + 15}">{esc(cx(cfg))}</text>')
        what = (f"the whole head, {fmt(a['T'])} cycles" if whole else
                f"cycles {fmt(w0)}–{fmt(w1)} of {fmt(a['T'])}: the start of the core")
        body.append(f'<text class="mut" x="{LEFT + 92}" y="{y + 15}">{esc(what)}</text>')
        if not whole:                                 # where the window sits in the head
            bx, bw = LEFT + PLOT - 260, 260
            body.append(f'<rect class="track" x="{bx}" y="{y + 7}" width="{bw}" height="8"/>')
            body.append(f'<rect class="track-on" x="{bx + bw * w0 / a["T"]:.1f}" y="{y + 7}" '
                        f'width="{max(2.0, bw * (w1 - w0) / a["T"]):.1f}" height="8"/>')
            body.append(f'<text class="mut" x="{bx - 8}" y="{y + 15}" text-anchor="end">in the head</text>')
        y += HEAD - 22
        markTop = y
        # the markers: QK^T's first pass, and the projection's last read-out when in view; the earlier label sits left
        # of its line, the later one right of its line
        marks = [(a["qkStart"], "QKᵀ starts")]
        if w0 <= a["projEnd"] <= w1:
            marks.append((a["projEnd"], "last token projected"))
        marks.sort()
        labelRow = y + 12
        y += 20
        top = y
        for key, name in LANES:
            body.append(f'<text class="rl" x="{LEFT - 10}" y="{y + STRIP + 4}" text-anchor="end">{esc(name)} '
                        f'×{a["count"][key]}</text>')
            for p in a["ops"][key]:
                if p["de"] < w0 or p["ls"] > w1:
                    continue
                body.extend(strips(p, w0, w1, y, scale, STRIP))
            y += LANE
        bottom = y - LANE + 2 * STRIP + 2
        shown = [(t, text) for t, text in marks if w0 <= t <= w1]
        for i, (t, text) in enumerate(shown):
            x = LEFT + (t - w0) * scale
            body.append(f'<line class="marker" x1="{x:.1f}" y1="{markTop + 16:.1f}" x2="{x:.1f}" y2="{bottom:.1f}"/>')
            left = len(shown) == 2 and i == 0 or x + 6 + 7.2 * len(text) > LEFT + PLOT
            body.append(f'<text class="mk" x="{x - 6 if left else x + 6:.1f}" y="{labelRow:.1f}" '
                        f'text-anchor="{"end" if left else "start"}">{esc(text)}</text>')
        body.extend(axis(bottom + 6, LEFT, scale, w0, span, top - 4, "cycle of the head", True))
        y = bottom + 58
    H = y - 10
    return (f'<svg class="pipe" viewBox="0 0 {W} {H}" role="img" aria-label="every crossbar of {esc(head["name"])} '
            f'in time at both arrays">' + "".join(body) + "</svg>")


def stageTable(head: dict) -> str:
    """The stage lengths per pass: the table twin of the charts."""
    kinds = (("proj", "Projection"), ("log", "LOG conversion"), ("qk", "QKᵀ and PV EXP"), ("smExp", "softmax EXP"),
             ("smClog", "softmax carry EXP, LOG"), ("accExp", "rescale EXP, LOG"))
    rows = []
    for key, name in kinds:
        for cfg, a in head["arrays"].items():
            s = a["stage"][key]
            count = dict(qk=a["count"]["qk"] + a["count"]["pv"], smClog=2,
                         accExp=a["count"]["accExp"] + a["count"]["accLog"]).get(key, a["count"][key])
            side = "load + compute" if s["bound"] == "load" else "read-out"
            hidden = "read-out" if s["bound"] == "load" else "load and compute"
            rows.append(f"<tr><td>{esc(name)}</td><td>{esc(cx(cfg))}</td><td>{count}</td><td>{s['rows']}</td>"
                        f"<td>{s['L']}</td><td>10</td><td>{s['D']}</td><td>{s['own']}</td><td>{side}</td>"
                        f"<td>{hidden}</td></tr>")
    return ('<div class="tablewrap"><table class="stages"><thead><tr><th>operation</th><th>array</th>'
            '<th>crossbars</th><th>numbers in</th><th>load</th><th>compute</th><th>read out</th>'
            '<th>back to back, a pass every</th><th>set by</th><th>hidden</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def headCaption(head: dict) -> str:
    a0, a1 = (head["arrays"][c] for c in head["arrays"])
    c0, c1 = (cx(c) for c in head["arrays"])

    def run_(a, cfg):
        if a["runs"] == 1:
            return (f"On {cfg} one run holds all {a['n']} queries, so the first QKᵀ pass needs the last token: the "
                    f"projection ends at cycle {fmt(a['projEnd'])} and QKᵀ starts at {fmt(a['qkStart'])}. Nothing "
                    f"overlaps the projection.")
        return (f"On {cfg} a run holds {a['S']} queries, so the head is {a['runs']} runs. The first run needs only "
                f"tokens 1 to {a['S']}: QKᵀ starts at cycle {fmt(a['qkStart'])} while the projection runs on to cycle "
                f"{fmt(a['projEnd'])}. That is the overlap.")
    return (f"<p>Each crossbar group is two strips: the top strip is its input side (load, then compute), the bottom "
            f"strip its output side (read-out). Where both strips of a group are busy at once, a read-out is hidden "
            f"under the next pass's load.</p><ul class=\"notes\"><li>{run_(a0, c0)}</li><li>{run_(a1, c1)}</li>"
            f"<li>The core can start only when a run's queries are ready, because each QKᵀ pass works on every query "
            f"of the run at once. A run holds at most min(R, C) queries: the softmax updates one running max and sum "
            f"per query row in a single crossbar pass.</li></ul>")


def busyLanes(a: dict, run_: int, blk: int, t0: int, t1: int) -> list:
    """The core lanes other than QK^T in which block (run_, blk) has a pass during cycles t0..t1."""
    return [name for key, name in JOURNEY_LANES[1:]
            if any(p.get("run") == run_ and p.get("blk") == blk and p["ls"] <= t1 and p["de"] >= t0
                   for p in a["ops"][key])]


def andList(items: list, last: str = " and ") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + last + items[-1]


def qkSpan(a: dict, run_: int, blk: int) -> tuple:
    ps = [p for p in a["ops"]["qk"] if p["run"] == run_ and p["blk"] == blk]
    return min(p["ls"] for p in ps), max(p["de"] for p in ps)


def journeyCaption(head: dict) -> str:
    items = []
    for cfg, a in head["arrays"].items():
        run_, blk = chosenBlock(a)
        mine = [p for k, _ in JOURNEY_LANES for p in a["ops"][k] if p.get("run") == run_ and p.get("blk") == blk]
        span = max(p["de"] for p in mine) + 1 - min(p["ls"] for p in mine)
        q0, q1 = qkSpan(a, run_, blk)
        phrases = [f"block {bb + 1} is in {andList(busyLanes(a, run_, bb, q0, q1))}"
                   for bb in range(blk - 1, max(-1, blk - 4), -1) if busyLanes(a, run_, bb, q0, q1)]
        still = f" While block {blk + 1} is in QKᵀ, {andList(phrases, ', and ')}." if phrases else ""
        if blk >= 1 and not busyLanes(a, run_, blk - 1, q0, q1) and qkSpan(a, run_, blk - 1)[1] < q0:
            still += f" Block {blk} has finished QKᵀ and waits for the softmax."
        starts = sorted({p["ls"] for p in a["ops"]["qk"] if p["run"] == run_ and p["wave"] == 0})
        period = round((starts[-1] - starts[0]) / (len(starts) - 1)) if len(starts) > 1 else None
        every = (f" A new key block enters QKᵀ about every {period} cycles, so about {span / period:.0f} blocks are "
                 f"in flight at once." if period and a["blocks"] >= 4 else "")
        items.append(f"<li><b>{esc(cx(cfg))}.</b> Block {blk + 1} takes {fmt(span)} cycles from its first QKᵀ load "
                     f"to its last rescale read-out.{still}{every}</li>")
    return ('<p>The softmax of a block can start only after all of the block\'s QKᵀ passes, because it needs the '
            'block\'s maximum score. PV then follows the softmax pass by pass, and the rescale waits for both. While '
            'one block is in the softmax, a later block is already in QKᵀ: that is the latency hidden between '
            f'operations.</p><ul class="notes">{"".join(items)}</ul>')


PAGE_CSS_EXTRA = """
[hidden] { display: none !important; }
:root { --load: #2a78d6; --compute: #eb6834; --read: #1baf7a; --wait: #cfd8d7; --track: #e3eaea; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --load: #3987e5; --compute: #d95926; --read: #199e70; --wait: #34423f; --track: #22302f; } }
:root[data-theme="dark"] { --load: #3987e5; --compute: #d95926; --read: #199e70; --wait: #34423f; --track: #22302f; }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px 6px; overflow-x: auto; }
.pipe, .board { display: block; width: 100%; min-width: 1000px; height: auto; }
.pipe text, .board text { font-family: var(--body); font-size: 13px; fill: var(--fg); }
.pipe .mut, .board .mut { fill: var(--muted); }
.pipe .hd, .board .hd { font-family: var(--display); font-weight: 600; font-size: 16px; }
.pipe .rl { font-size: 12.5px; }
.pipe .ax { font-family: var(--mono); font-size: 11.5px; fill: var(--muted); }
.pipe .mk { font-size: 12.5px; font-weight: 600; }
.pipe .grid { stroke: var(--rule); stroke-width: 1; }
.pipe .marker { stroke: var(--fg); stroke-width: 1.2; }
.pipe .cursor { stroke: var(--fg); stroke-width: 1.5; }
.pipe .track { fill: var(--track); }
.pipe .track-on { fill: var(--muted); }
.s-load { fill: var(--load); } .s-compute { fill: var(--compute); } .s-read { fill: var(--read); } .s-wait { fill: var(--wait); }
.faint { opacity: 0.22; }
.board .bt { font-weight: 600; font-size: 13.5px; }
.board .bs { font-size: 11.5px; fill: var(--muted); }
.board .bl { font-size: 13px; }
.board .bst { font-family: var(--mono); font-size: 12px; }
.board .bp { font-family: var(--mono); font-size: 12px; font-weight: 500; }
.board .buf { fill: none; stroke: var(--muted); stroke-width: 1; }
.board .port { stroke: var(--muted); stroke-width: 3; }
.board .link { stroke: var(--rule); stroke-width: 2; }
.board .dot { fill: var(--fg); }
.board .arr { fill: var(--bg); stroke: var(--muted); stroke-width: 1; }
.board .arr-on { fill: color-mix(in srgb, var(--compute) 22%, var(--surface)); stroke: var(--compute); }
.board .arr-held { fill: var(--wait); stroke: var(--muted); }
.board .tick { fill: var(--wait); }
.board .tick-on { fill: var(--compute); }
.board .fill-load { fill: var(--load); }
.board .fill-loaded { fill: color-mix(in srgb, var(--load) 45%, var(--surface)); }
.board .fill-wait { fill: var(--wait); }
.board .fill-read { fill: var(--read); }
.key { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; align-items: center; }
.key span { display: inline-flex; align-items: center; gap: 7px; }
.key i { display: inline-block; width: 22px; height: 11px; border-radius: 2px; }
.rules { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 14px; }
.rule { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; display: grid; gap: 4px; min-width: 0; }
.rule b { font-family: var(--display); font-size: 16px; font-weight: 600; }
.rule .n { font-family: var(--mono); font-size: 13px; color: var(--muted); }
.toggle { display: inline-flex; flex-wrap: wrap; gap: 0; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; width: fit-content; max-width: 100%; }
.toggle button { font: 500 14px var(--body); color: var(--fg); background: var(--surface); border: 0; padding: 7px 14px; cursor: pointer; }
.toggle button + button { border-left: 1px solid var(--rule); }
.toggle button[aria-pressed="true"] { background: var(--accent-soft); color: var(--fg); box-shadow: inset 0 -2px 0 var(--accent); }
.toggle button:focus-visible, .ctl button:focus-visible, .ctl select:focus-visible, .ctl input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.headview { display: grid; gap: 34px; }
table.stages th:nth-child(n+3), table.stages td:nth-child(n+3) { text-align: right; }
table.stages th:nth-child(n+9), table.stages td:nth-child(n+9) { text-align: left; }
.opfig { display: grid; gap: 8px; }
.opfig ul { margin: 0; padding-left: 20px; display: grid; gap: 4px; font-size: 14px; }
.boards { display: grid; gap: 8px; }
.ctl { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; font-size: 14px; }
.ctl button { font: 500 14px var(--body); color: var(--fg); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 5px 14px; cursor: pointer; min-width: 5.5em; }
.ctl button:hover { border-color: var(--accent); }
.ctl input[type=range] { flex: 1 1 260px; accent-color: var(--accent); min-width: 0; }
.ctl select { font: inherit; color: var(--fg); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 3px 6px; }
.ctl .clock { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 16ch; }
.status { font-family: var(--mono); font-size: 13px; min-height: 2.6em; overflow-wrap: anywhere; }
.foot { font-size: 14px; }
.sticky { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); padding-block: 8px; display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); }
@media (prefers-reduced-motion: reduce) { .board .dot { display: none; } }
"""

# the page's script: the head toggle, the boards, the controls, the cursors, the key-block status lines. unitState
# and the texts are this file's unitState / stateTexts / laneStatus rule, line for line.
SCRIPT = r"""
(function () {
  const DATA = JSON.parse(document.getElementById("pipeData").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const X = {inEnd: 694, arr: 724, arrW: 150, outBar: 906};   // the input buffer ends at inEnd, next to the array
  const ROW = 106, TOP = 34, LBL = -16;
  const fmt = n => Math.round(n).toLocaleString("en-US");
  const el = (tag, attrs, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    parent.appendChild(e);
    return e;
  };
  // the state rule: what a crossbar group holds at cycle t (pipelines.py unitState)
  const unitState = (u, t) => {
    let inp = null, arr = null, out = null;
    for (const [ls, le, cs, mf, ad, ds, de, label] of u.passes) {
      if (ls <= t && t <= mf) inp = t <= le ? {k: "loading", label, n: Math.min(u.rows, 5 * (t - ls + 1))}
                                            : (t < cs ? {k: "waiting", label} : {k: "read", label});
      if (cs <= t && t <= ad) arr = t <= mf ? {k: "bit", label, b: t - cs + 1}
                                            : ((t === mf + 1 || t === ad) ? {k: "finishing", label} : {k: "held", label});
      if (ds <= t && t <= de) out = {k: "reading", label, n: Math.min(u.C, 5 * (t - ds + 1))};
    }
    return {inp, arr, out};
  };
  const inText = (s, u) => !s ? "" : s.k === "loading" ? `${s.label} · loading ${s.n} / ${u.rows}`
                          : s.k === "waiting" ? `${s.label} · waits for the array` : `${s.label} · bits read by the array`;
  const arrText = s => !s ? "" : s.k === "bit" ? `bit ${s.b} of 8` : s.k === "finishing" ? "finishing" : "held: output busy";
  const outText = (s, u) => !s ? "" : `${s.label} · read out ${s.n} / ${u.C}`;
  const laneStatus = (lane, t) => {
    const parts = [];
    for (const [ls, le, cs, mf, ad, ds, de, label] of lane.passes) {
      if (ls <= t && t <= mf) parts.push(`${label} ${t <= le ? "loading" : t < cs ? "waiting" : "computing"}`);
      if (ds <= t && t <= de) parts.push(`${label} reading out`);
    }
    return parts.length ? `${lane.name}: ${parts.join(", ")}` : "";
  };
  const dots = (g, x0, x1, cy) => [0, 1, 2, 3].map(() => el("circle", {class: "dot", r: 3, cx: x0, cy, opacity: 0}, g));
  const moveDots = (ds, x0, x1, on, tf) => ds.forEach((d, i) => {
    d.setAttribute("opacity", on ? 1 : 0);
    if (on) d.setAttribute("cx", (x0 + (((tf * 0.5) + i / 4) % 1) * (x1 - x0)).toFixed(1));
  });

  // one board: a heading and one row per crossbar group; returns update(t, tf)
  function buildBoard(svg, spec, scale) {
    el("text", {class: "hd", x: 20, y: 22}, svg, `${spec.R} × ${spec.C}`);
    const rows = spec.units.map((u, i) => {
      const g = el("g", {class: "unit"}, svg);
      const y = TOP + i * ROW, cy = y + 52;
      const title = el("text", {class: "bt", x: 20, y: y + 16}, g, `${u.name} ×${u.count}`);
      el("tspan", {class: "mut"}, title, `  · ${u.rows} numbers in, ${u.C} columns out`);
      const inW = u.rows * scale, outW = u.C * scale, inX = X.inEnd - inW;
      const q1 = inX - 10, q0 = q1 - 80;                          // the input port, just before the buffer
      el("text", {class: "bl", x: q0 - 10, y: cy + 4, "text-anchor": "end"}, g, u.inLabel);
      el("line", {class: "port", x1: q0, y1: cy, x2: q1, y2: cy, "data-solid": "port"}, g);
      el("text", {class: "bs", x: (q0 + q1) / 2, y: cy + 22, "text-anchor": "middle"}, g, "5 per cycle");
      el("text", {class: "bs", x: X.inEnd, y: cy + LBL, "text-anchor": "end"}, g, "input buffer");
      el("rect", {class: "buf", x: inX.toFixed(1), y: cy - 10, width: inW.toFixed(1), height: 20, "data-solid": "box"}, g);
      const inFill = el("rect", {class: "fill-load", x: inX.toFixed(1), y: cy - 10, width: 0, height: 20}, g);
      el("line", {class: "link", x1: X.inEnd + 4, y1: cy, x2: X.arr - 4, y2: cy, "data-solid": "link"}, g);
      el("text", {class: "bs", x: X.arr + X.arrW / 2, y: cy - 24, "text-anchor": "middle"}, g, `crossbar ${spec.R} × ${spec.C}`);
      const box = el("rect", {class: "arr", x: X.arr, y: cy - 18, width: X.arrW, height: 40, rx: 3, "data-solid": "box"}, g);
      const lab = el("text", {class: "bp st-lab", x: X.arr + X.arrW / 2, y: cy - 1, "text-anchor": "middle", "data-inside": "box"}, g, "");
      const tw = (X.arrW - 20) / 8;
      const ticks = [...Array(8).keys()].map(k => el("rect", {class: "tick", x: (X.arr + 10 + k * tw).toFixed(1), y: cy + 8, width: (tw - 3).toFixed(1), height: 8}, g));
      el("line", {class: "link", x1: X.arr + X.arrW + 4, y1: cy, x2: X.outBar - 4, y2: cy, "data-solid": "link"}, g);
      el("text", {class: "bs", x: X.outBar, y: cy + LBL}, g, "output buffer");
      el("rect", {class: "buf", x: X.outBar, y: cy - 10, width: outW.toFixed(1), height: 20, "data-solid": "box"}, g);
      const outFill = el("rect", {class: "fill-read", x: X.outBar, y: cy - 10, width: 0, height: 20}, g);
      const p0 = X.outBar + outW + 10, p1 = p0 + 80;
      el("line", {class: "port", x1: p0, y1: cy, x2: p1, y2: cy, "data-solid": "port"}, g);
      el("text", {class: "bs", x: (p0 + p1) / 2, y: cy + 22, "text-anchor": "middle"}, g, "5 per cycle");
      el("text", {class: "bl", x: p1 + 10, y: cy + 4}, g, u.outLabel);
      const tin = el("text", {class: "bst st-in", x: X.inEnd, y: cy + 40, "text-anchor": "end"}, g, "");
      const tarr = el("text", {class: "bst st-arr", x: X.arr + X.arrW / 2, y: cy + 40, "text-anchor": "middle"}, g, "");
      const tout = el("text", {class: "bst st-out", x: X.outBar, y: cy + 40}, g, "");
      return {u, inW, outW, inFill, box, lab, ticks, outFill, tin, tarr, tout, q0, q1,
              inDots: dots(g, q0, q1, cy), outDots: dots(g, p0, p1, cy), p0, p1};
    });
    svg.setAttribute("viewBox", `0 0 1500 ${TOP + spec.units.length * ROW + 4}`);
    return (t, tf) => {
      for (const r of rows) {
        const s = unitState(r.u, t);
        const frac = !s.inp ? 0 : s.inp.k === "loading" ? s.inp.n / r.u.rows : 1;
        r.inFill.setAttribute("width", (r.inW * frac).toFixed(1));
        r.inFill.setAttribute("class", !s.inp || s.inp.k === "loading" ? "fill-load" : s.inp.k === "waiting" ? "fill-wait" : "fill-loaded");
        r.tin.textContent = inText(s.inp, r.u);
        moveDots(r.inDots, r.q0, r.q1, !!s.inp && s.inp.k === "loading", tf);
        r.box.setAttribute("class", "arr" + (!s.arr ? "" : s.arr.k === "held" ? " arr-held" : " arr-on"));
        r.lab.textContent = s.arr ? s.arr.label : "";
        r.ticks.forEach((k, i) => k.setAttribute("class", "tick" + (s.arr && (s.arr.k !== "bit" || i < s.arr.b) ? " tick-on" : "")));
        r.tarr.textContent = arrText(s.arr);
        const left = s.out ? 1 - s.out.n / r.u.C : 0;
        r.outFill.setAttribute("x", (X.outBar + r.outW * (1 - left)).toFixed(1));
        r.outFill.setAttribute("width", (r.outW * left).toFixed(1));
        r.tout.textContent = outText(s.out, r.u);
        moveDots(r.outDots, r.p0, r.p1, !!s.out, tf);
      }
    };
  }

  // one animated section: controls, the things to update at t, a span
  function section(sec, span, start, update) {
    const btn = sec.querySelector("button.play"), slider = sec.querySelector("input[type=range]");
    const clock = sec.querySelector(".clock"), speed = sec.querySelector("select");
    slider.max = String(span);
    let t = start, playing = false, last = null;
    const draw = () => {
      const ti = Math.min(span, Math.floor(t));
      update(ti, t);
      slider.value = String(t);
      clock.textContent = `cycle ${fmt(ti)} of ${fmt(span)}`;
    };
    const frame = (now) => {
      if (!playing) return;
      if (last !== null) t += (now - last) / 1000 * (span / 16) * parseFloat(speed.value);
      last = now;
      if (t >= span) { t = span; playing = false; btn.textContent = "Play"; }
      draw();
      if (playing) requestAnimationFrame(frame);
    };
    btn.addEventListener("click", () => {
      playing = !playing;
      btn.textContent = playing ? "Pause" : "Play";
      if (playing) { if (t >= span) t = 0; last = null; requestAnimationFrame(frame); }
    });
    slider.addEventListener("input", () => { t = parseFloat(slider.value); draw(); });
    sec.pause = () => { playing = false; btn.textContent = "Play"; };
    draw();
  }

  document.querySelectorAll(".opsec").forEach(sec => {
    const spec = DATA.heads[+sec.dataset.head].ops[sec.dataset.op];
    const boards = [...sec.querySelectorAll("svg.board")].map((svg, i) => buildBoard(svg, spec.arrays[i], spec.scale));
    const chart = sec.querySelector("svg.pipe");
    const cursors = [...chart.querySelectorAll(".cursor")];
    const left = parseFloat(chart.dataset.left), scale = parseFloat(chart.dataset.scale);
    section(sec, spec.span, spec.start, (ti, tf) => {
      boards.forEach(b => b(ti, tf));
      const x = (left + tf * scale).toFixed(1);
      cursors.forEach(c => { c.setAttribute("x1", x); c.setAttribute("x2", x); });
    });
  });
  document.querySelectorAll(".journey").forEach(sec => {
    const spec = DATA.heads[+sec.dataset.head].journey;
    const chart = sec.querySelector("svg.pipe");
    const cursors = [...chart.querySelectorAll(".cursor")];
    const left = parseFloat(chart.dataset.left), scale = parseFloat(chart.dataset.scale);
    const lines = [...sec.querySelectorAll(".status")];
    section(sec, spec.span, spec.start, (ti, tf) => {
      const x = (left + tf * scale).toFixed(1);
      cursors.forEach(c => { c.setAttribute("x1", x); c.setAttribute("x2", x); });
      spec.arrays.forEach((a, i) => {
        const parts = a.lanes.map(l => laneStatus(l, ti)).filter(Boolean);
        lines[i].textContent = `${a.cfg.replace("x", " × ")}, head cycle ${fmt(a.w0 + ti)}: ` + (parts.length ? parts.join(" · ") : "idle");
      });
    });
  });

  const views = document.querySelectorAll(".headview"), buttons = document.querySelectorAll(".toggle button");
  const show = (h) => {
    views.forEach(v => { v.hidden = v.dataset.head !== h; });
    buttons.forEach(b => b.setAttribute("aria-pressed", b.dataset.head === h ? "true" : "false"));
    document.querySelectorAll(".opsec, .journey").forEach(s => { if (s.dataset.head !== h && s.pause) s.pause(); });
  };
  buttons.forEach(b => b.addEventListener("click", () => {
    show(b.dataset.head);
    try { history.replaceState(null, "", "#" + b.dataset.token); } catch (e) {}
  }));
  const fromHash = [...buttons].find(b => "#" + b.dataset.token === location.hash);
  show(fromHash ? fromHash.dataset.head : "0");
})();
"""

KEY = ('<div class="key" aria-label="Stages"><span><i style="background: var(--load)"></i>load</span>'
       '<span><i style="background: var(--compute)"></i>compute</span><span><i style="background: var(--read)"></i>'
       'read out</span><span><i style="background: var(--wait)"></i>waiting</span></div>')


def controls(prefix: str) -> str:
    return (f'<div class="ctl"><button type="button" class="play" id="{prefix}-play">Play</button>'
            f'<input type="range" id="{prefix}-t" min="0" max="1" step="0.25" value="0" aria-label="cycle">'
            f'<span class="clock" id="{prefix}-clock"></span><label for="{prefix}-speed">speed '
            f'<select id="{prefix}-speed"><option value="0.25">¼×</option><option value="0.5">½×</option>'
            f'<option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option>'
            f'</select></label></div>')


def headView(i: int, head: dict) -> str:
    figs = []
    for op, title, _ in CHART_OPS:
        bullets = "".join(f"<li>{describe(op, a)}</li>" for a in head["arrays"].values())
        boards = "".join(f'<div class="chart"><svg class="board" data-cfg="{c}" role="img" '
                         f'aria-label="{esc(title)} crossbar pipeline at {esc(cx(c))}"></svg></div>'
                         for c in head["arrays"])
        figs.append(f'<figure class="opfig opsec" id="op-{i}-{op}" data-head="{i}" data-op="{op}"><h3>{esc(title)}</h3>'
                    f'<p class="muted">{esc(CAPTION[op])}</p>{controls(f"op-{i}-{op}")}'
                    f'<div class="boards">{boards}</div><div class="chart">{opChart(head, op)}</div>'
                    f'<ul>{bullets}</ul></figure>')
    status = "".join(f'<p class="status" id="journey-{i}-{c}"></p>' for c in head["arrays"])
    return (f'<div class="headview" data-head="{i}"{" hidden" if i else ""}>'
            f'<section><h2>Each operation, pass by pass</h2>'
            f'<p>Each board is one crossbar group at work, cycle by cycle: the input port and buffer (load), the array '
            f'with its 8 bit-slices (compute), the output buffer and port (read out). The label in each part says '
            f'which pass is there, so you can see one pass reading out while the next one loads. Both arrays run on '
            f'one clock, and the diagram under the boards marks the same cycle. The boards play the passes of the '
            f'diagram, taken from the steady part of the head.</p>'
            f'{"".join(figs)}</section>'
            f'<section class="journey" id="journey-{i}" data-head="{i}"><h2>The full head, one key block</h2>'
            f'<p>Every crossbar group of the attention core, with one key block\'s passes in colour and the other '
            f'blocks faint. Each group is two strips: the top one is its input side (load, then compute), the bottom '
            f'one its output side (read-out). The whole head with all of its crossbars animated is on the page '
            f'<a href="{TIME_PAGE}">Attention head in time</a>.</p>'
            f'{controls(f"journey-{i}")}<div class="chart">{journeyChart(head)}</div>{status}'
            f'{journeyCaption(head)}</section>'
            f'<section><h2>When the core starts</h2>'
            f'<div class="chart">{headChart(head)}</div>{headCaption(head)}</section>'
            f'<section><h2>Stage lengths</h2><p>Cycles per pass. Back to back, a crossbar starts a pass every '
            f'max(load + 8, read out + 1) cycles; the other side is hidden.</p>{stageTable(head)}</section>'
            f'</div>')


def page(res: dict) -> str:
    heads = res["heads"]
    cfgs = " and ".join(cx(c) for c in res["arrays"])
    toggle = "".join(f'<button type="button" id="head{i}" data-head="{i}" data-token="{t}" aria-pressed='
                     f'"{"true" if i == 0 else "false"}">{esc(h["name"])} · n {h["n"]}, d<sub>model</sub> {h["dm"]}</button>'
                     for i, (h, t) in enumerate(zip(heads, ("jetformer", "vit"))))
    data = json.dumps(pageData(res), separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return f"""<title>Crossbar pipelines</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Every operation of one attention head on NL-DPE crossbars as passes of load, compute and read-out, animated from the simulator's own pass events.">
<meta name="color-scheme" content="light dark">
{FONTS}
<style>{PAGE_CSS}
{PAGE_CSS_EXTRA}
h3 {{ font-family: var(--display); font-weight: 600; font-size: 18px; margin: 0; }}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">NL-DPE attention · crossbar pipelines · 2026-10-21</p>
  <h1>Inside each operation: load, compute, read out</h1>
  <p class="lede">Every operation of the attention head runs as crossbar passes, and every pass has the same three stages. The boards and diagrams are the simulator's own passes for {esc(heads[0]["name"])} and {esc(heads[1]["name"])} on {cfgs}.</p>
  <div class="rules">
    <div class="rule"><b>Load</b><span>The input enters through the crossbar's 40-bit port, 5 numbers per cycle.</span><span class="n">numbers in ÷ 5 cycles</span></div>
    <div class="rule"><b>Compute</b><span>One cycle per bit of the 8-bit input, plus 2 to finish. The same on every array.</span><span class="n">10 cycles</span></div>
    <div class="rule"><b>Read out</b><span>The output leaves through a 40-bit port, 5 numbers per cycle, over all C columns.</span><span class="n">C ÷ 5 cycles</span></div>
  </div>
  <ul class="notes">
    <li><b>How the stages overlap.</b> The next pass starts loading as soon as the current pass has done its 8 bit cycles, so the read-out of one pass runs while the next pass loads and computes.</li>
    <li><b>Which stage sets the pace.</b> The longer side: load + 8, or read-out + 1. The shorter side is hidden. A wait (grey) is a pass held up by the other side or by its input.</li>
    <li><b>What changes with the array.</b> Load time follows the numbers a crossbar takes in, read-out time its column count C. Compute never changes.</li>
  </ul>
</header>
<div class="sticky"><div class="toggle" role="group" aria-label="Head">{toggle}</div>{KEY}</div>
{"".join(headView(i, h) for i, h in enumerate(heads))}
<p class="muted foot">Data: the simulator (<code>v3/simulator</code>, timing pinned to the RTL by the gates) with every crossbar pass recorded, <code>v3/dse/results/stages/pipelines.json</code>; page <code>v3/dse/pipelines.py</code>. The read-out walks every column of the array, the model's rule since 2026-10-18.</p>
</div>
<script type="application/json" id="pipeData">{data}</script>
<script>{SCRIPT}</script>
"""


# -- checks ----------------------------------------------------------------------------------------------------------
def grammarIssues(a: dict) -> list:
    """The pass grammar on every recorded pass: stage lengths, the order of the events, and between consecutive passes
    of one crossbar group the three hand-overs (input buffer, accumulator, output port)."""
    out = []
    for key, ps in a["ops"].items():
        s = a["stage"][key]
        for i, p in enumerate(ps):
            if p["le"] - p["ls"] + 1 != s["L"]:
                out.append(f"{key}[{i}] load {p['le'] - p['ls'] + 1} != {s['L']}")
            if p["mf"] - p["cs"] != P - 1:
                out.append(f"{key}[{i}] slices {p['mf'] - p['cs'] + 1} != {P}")
            if not (p["cs"] >= p["le"] + 1 and p["ad"] >= p["mf"] + 2 and p["ds"] == p["ad"] + 1):
                out.append(f"{key}[{i}] event order")
            if p["de"] - p["ds"] + 1 != s["D"]:
                out.append(f"{key}[{i}] read-out {p['de'] - p['ds'] + 1} != {s['D']}")
        for i, (p, q) in enumerate(zip(ps, ps[1:])):
            if not (q["ls"] >= p["mf"] + 1 and q["cs"] >= p["ad"] + 1 and q["ad"] >= p["de"] + 1):
                out.append(f"{key}[{i}->{i + 1}] hand-over")
    return out


HARNESS = r"""
(async function () {
  try { await document.fonts.ready; } catch (e) {}
  const rep = {boards: [], journey: [], issues: []};
  document.querySelectorAll(".headview").forEach(v => { v.hidden = false; });
  const over = (a, c) => Math.min(a.x + a.width, c.x + c.width) - Math.max(a.x, c.x) > 0.5 &&
                         Math.min(a.y + a.height, c.y + c.height) - Math.max(a.y, c.y) > 0.5;
  const inside = (a, c) => a.x >= c.x - 1 && a.y >= c.y - 1 && a.x + a.width <= c.x + c.width + 1 && a.y + a.height <= c.y + c.height + 1;
  const layout = (svg, where) => {
    const vb = svg.viewBox.baseVal;
    const texts = [...svg.querySelectorAll("text")].filter(t => t.textContent.trim()).map(t => ({t, b: t.getBBox()}));
    const solids = [...svg.querySelectorAll("[data-solid]")].map(s => {
      const b = s.getBBox();
      return {s, b: s.tagName === "line" ? {x: b.x - 1.5, y: b.y - 1.5, width: b.width + 3, height: b.height + 3} : b};
    });
    for (const {t, b} of texts)
      if (b.x < vb.x - 0.5 || b.y < vb.y - 0.5 || b.x + b.width > vb.x + vb.width + 0.5 || b.y + b.height > vb.y + vb.height + 0.5)
        rep.issues.push({kind: "outside the drawing", where, text: t.textContent});
    for (let i = 0; i < texts.length; i++)
      for (let j = i + 1; j < texts.length; j++)
        if (over(texts[i].b, texts[j].b)) rep.issues.push({kind: "label on label", where, text: texts[i].t.textContent + " | " + texts[j].t.textContent});
    for (const {t, b} of texts) for (const {s, b: c} of solids)
      if (over(b, c) && !(t.getAttribute("data-inside") === "box" && inside(b, c))) {
        rep.issues.push({kind: "label on " + s.getAttribute("data-solid"), where, text: t.textContent}); break; }
  };
  document.querySelectorAll("svg.pipe").forEach((svg, i) => layout(svg, "chart " + i));
  for (const sec of document.querySelectorAll(".opsec")) {
    const slider = sec.querySelector("input[type=range]"), span = Math.floor(parseFloat(slider.max));
    const step = Math.max(1, Math.floor(span / 150));
    for (let t = 0; t <= span; t += step) {
      slider.value = String(t); slider.dispatchEvent(new Event("input"));
      sec.querySelectorAll("svg.board").forEach((svg, bi) => {
        svg.querySelectorAll("g.unit").forEach((g, ui) => rep.boards.push([+sec.dataset.head, sec.dataset.op, bi, ui, t,
          g.querySelector(".st-in").textContent, g.querySelector(".st-lab").textContent,
          g.querySelector(".st-arr").textContent, g.querySelector(".st-out").textContent]));
        if ((t / step) % 5 === 0) layout(svg, `${sec.id} board ${bi} cycle ${t}`);
      });
    }
  }
  for (const sec of document.querySelectorAll(".journey")) {
    const slider = sec.querySelector("input[type=range]"), span = Math.floor(parseFloat(slider.max));
    const step = Math.max(1, Math.floor(span / 200));
    for (let t = 0; t <= span; t += step) {
      slider.value = String(t); slider.dispatchEvent(new Event("input"));
      sec.querySelectorAll(".status").forEach((p, ai) => rep.journey.push([+sec.dataset.head, ai, t, p.textContent]));
    }
  }
  const pre = document.createElement("pre"); pre.id = "report";
  pre.textContent = JSON.stringify({font: document.fonts.check('13px "IBM Plex Sans"'), ...rep});
  document.body.appendChild(pre);
})();
"""


def browserReport(text: str, name: str) -> dict:
    """Run HARNESS in headless Chrome on a page text; returns its report."""
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    probe = CHECK_DIR / name
    probe.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><body>{text}<script>{HARNESS}</script>'
                     f'</body></html>', encoding="utf-8")
    r = subprocess.run([chrome(), "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={CHECK_DIR / 'profile'}", "--window-size=1600,1100",
                        "--virtual-time-budget=120000", "--dump-dom", probe.as_uri()],
                       capture_output=True, text=True, timeout=600)
    m = re.search(r'<pre id="report">(.*?)</pre>', r.stdout, re.S)
    if not m:
        raise RuntimeError(f"no report from Chrome (exit {r.returncode}): {r.stderr[-400:]}")
    return json.loads(html.unescape(m.group(1)))


def mismatches(rep: dict, data: dict) -> list:
    """The page's board and status texts against this file's state rule."""
    out = []
    for h, op, bi, ui, t, *texts in rep["boards"]:
        u = data["heads"][h]["ops"][op]["arrays"][bi]["units"][ui]
        want = list(stateTexts(u["passes"], u["rows"], u["C"], t))
        if texts != want:
            out.append(f"head {h} {op} board {bi} group {ui} cycle {t}: page {texts} rule {want}")
    for h, ai, t, text in rep["journey"]:
        a = data["heads"][h]["journey"]["arrays"][ai]
        parts = [s for s in (laneStatus(l["name"], l["passes"], t) for l in a["lanes"]) if s]
        want = f"{cx(a['cfg'])}, head cycle {fmt(a['w0'] + t)}: " + (" · ".join(parts) if parts else "idle")
        if text != want:
            out.append(f"head {h} journey {ai} cycle {t}: page {text!r} rule {want!r}")
    return out


def selfTest() -> int:
    res = collectAll()
    fails, checks = [], 0

    def check(ok: bool, what: str):
        nonlocal checks
        checks += 1
        if not ok:
            fails.append(what)
    for head in res["heads"]:
        for cfg, a in head["arrays"].items():
            tag = f"{head['name']} {cfg}"
            g = grammarIssues(a)
            check(not g, f"{tag}: pass grammar {g[:3]}")
            for k, v in a["lockstep"].items():
                check(v, f"{tag}: {k} not in lock-step")
            n, B, runs = a["n"], a["blocks"], a["runs"]
            check(len(a["ops"]["proj"]) == n and len(a["ops"]["log"]) == n, f"{tag}: one pass per token")
            check(len(a["ops"]["qk"]) == runs * B * a["TW_QK"], f"{tag}: QK waves")
            check(len(a["ops"]["pv"]) == runs * B * a["TW_PV"], f"{tag}: PV waves")
            check(len(a["ops"]["smCexp"]) == runs * (B - 1) and len(a["ops"]["smClog"]) == runs * B, f"{tag}: carry")
            check(len(a["ops"]["accLog"]) == runs * B * a["W_ACC"]
                  and len(a["ops"]["accExp"]) == runs * B * a["W_ACC"], f"{tag}: accumulator waves")
            pg = sorted({q["ls"] - p["ls"] for p, q in zip(a["ops"]["proj"][8:], a["ops"]["proj"][9:])})
            check(pg == [a["stage"]["proj"]["own"]], f"{tag}: projection pace {pg}")
            lg = sorted({q["ls"] - p["ls"] for p, q in zip(a["ops"]["log"][8:], a["ops"]["log"][9:])})
            check(lg == pg and a["stage"]["log"]["own"] <= pg[0], f"{tag}: converter pace {lg}")
            for key in ("qk", "pv"):
                ps = a["ops"][key]
                inner = {q["ls"] - p["ls"] for p, q in zip(ps, ps[1:]) if (q["run"], q["blk"]) == (p["run"], p["blk"])}
                check(not inner or inner == {a["stage"][key]["own"]}, f"{tag}: {key} waves {inner}")
            check(a["qkStart"] > a["firstRunReady"] - 40, f"{tag}: QK before the run's queries")
            if runs == 1:
                check(a["qkStart"] > a["projEnd"] - 40 and a["firstRunReady"] >= a["projEnd"], f"{tag}: one-run start")
            else:
                check(a["qkStart"] < a["projEnd"] - 1000, f"{tag}: overlap")
            for op, _, _ in CHART_OPS:
                rows = pick(op, a)
                check(len(rows) >= 2, f"{tag}: {op} rows")
                check(all(r[1]["ls"] >= 0 and r[1]["de"] < a["T"] for r in rows), f"{tag}: {op} inside the head")
            # the key-block chart: the softmax starts only after the block's last QK^T read-out (the block maximum)
            run_, blk = chosenBlock(a)
            mine = lambda p: p.get("run") == run_ and p.get("blk") == blk     # noqa: E731
            qkEnd = max(p["de"] for p in a["ops"]["qk"] if mine(p))
            smStart = min(p["ls"] for p in a["ops"]["smExp"] if mine(p))
            check(smStart > qkEnd, f"{tag}: softmax before the block's QK^T ends ({smStart} <= {qkEnd})")
            # the caption's claim: while the block is in QK^T, an earlier block runs in a later operation
            check(any(busyLanes(a, run_, bb, *qkSpan(a, run_, blk)) for bb in range(max(0, blk - 3), blk)),
                  f"{tag}: no overlap between blocks")
    # the projection's pace per token, max(ceil(8 min(d_m, R) / 40) + 8, ceil(8 C / 40) + 1), and the runs n / min(n, I):
    # JetFormer (d_m 64) 21 / 53, one run; ViT (d_m 768) 34 / 111, runs 256 / 64 = 4 at 128x64 and 1 at 512x256
    want = {("JetFormer-150P", "128x64"): (21, 1), ("JetFormer-150P", "512x256"): (53, 1),
            ("ViT-B/16", "128x64"): (34, 4), ("ViT-B/16", "512x256"): (111, 1)}
    for head in res["heads"]:
        for cfg, a in head["arrays"].items():
            pace, runs = want[(head["name"], cfg)]
            check((a["stage"]["proj"]["own"], a["runs"]) == (pace, runs), f"{head['name']} {cfg}: pace/runs")
    # the state rule on hand-made passes: a load-bound pair and a read-out-bound pair (ls le cs mf ad ds de)
    loadBound = [[0, 12, 13, 20, 22, 23, 35, "A"], [21, 33, 34, 41, 43, 44, 56, "B"]]
    check(stateTexts(loadBound, 64, 64, 0) == ("A · loading 5 / 64", "", "", ""), "rule: first load cycle")
    check(stateTexts(loadBound, 64, 64, 12) == ("A · loading 64 / 64", "", "", ""), "rule: last load cycle caps")
    check(stateTexts(loadBound, 64, 64, 13) == ("A · bits read by the array", "A", "bit 1 of 8", ""), "rule: bit 1")
    check(stateTexts(loadBound, 64, 64, 21) == ("B · loading 5 / 64", "A", "finishing", ""), "rule: overlap start")
    check(stateTexts(loadBound, 64, 64, 30) == ("B · loading 50 / 64", "", "", "A · read out 40 / 64"),
          "rule: load under read-out")
    readBound = [[0, 12, 13, 20, 22, 23, 74, "A"], [21, 33, 75, 82, 75 + 300, 376, 427, "B"]]
    check(stateTexts(readBound, 64, 256, 50)[0] == "B · waits for the array", "rule: input waits")
    check(stateTexts(readBound, 64, 256, 200)[2] == "held: output busy", "rule: result held")
    check(stateTexts(loadBound, 64, 64, 100) == ("", "", "", ""), "rule: idle")
    check(laneStatus("x", [p + [True] for p in loadBound], 30) == "x: A reading out, B loading", "rule: lane status")
    # planted faults: the grammar check must catch a 1-cycle shift, a short read-out, a broken hand-over
    a = res["heads"][0]["arrays"][res["arrays"][0]]
    for mutate, what in ((lambda p: p.__setitem__("le", p["le"] + 1), "load +1"),
                         (lambda p: p.__setitem__("de", p["de"] - 1), "read-out -1"),
                         (lambda p: p.__setitem__("ls", p["ls"] - 30), "hand-over")):
        b = json.loads(json.dumps(a))
        mutate(b["ops"]["qk"][1])
        check(bool(grammarIssues(b)), f"planted fault not caught: {what}")
    # the page in headless Chrome: every board text and status line equals this file's rule at every sampled cycle;
    # no label on a label or on a mark, nothing outside a drawing; and each check catches its planted fault
    if chrome():
        text, data = page(res), pageData(res)
        rep = browserReport(text, "pipelines_check.html")
        check(rep["font"], "the page font did not load")
        check(len(rep["boards"]) > 1000 and len(rep["journey"]) > 400, f"too few samples {len(rep['boards'])}")
        mm = mismatches(rep, data)
        check(not mm, f"page vs rule: {mm[:3]}")
        check(not rep["issues"], f"layout: {rep['issues'][:4]}")
        bad = text.replace("5 * (t - ls + 1)", "5 * (t - ls)", 1)
        check(bad != text and mismatches(browserReport(bad, "pipelines_planted_rule.html"), data),
              "planted rule fault not caught")
        bad = text.replace("LBL = -16", "LBL = 2", 1).replace(f'x="{LEFT - 10}" y=', f'x="{LEFT + 30}" y=', 1)
        planted = browserReport(bad, "pipelines_planted_layout.html")["issues"]
        check(any("label on bar" in i["kind"] for i in planted), "planted chart overlap not caught")
        check(any("label on box" in i["kind"] for i in planted), "planted board overlap not caught")
    else:
        print("note: no Chrome; the page checks were skipped")
    print(f"pipelines self-test: {checks - len(fails)}/{checks} checks pass")
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


def shots(res: dict, height: int = 7600, extra: str = "") -> list:
    """Screenshots of the page at desktop width, one per head (the head picked by the page's #token), for a look;
    `extra` runs after the page's script (e.g. to move the clocks)."""
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for token in ("jetformer", "vit"):
        probe = CHECK_DIR / f"pipelines_{token}.html"
        probe.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><body>{page(res)}'
                         f'<script>{extra}</script></body></html>', encoding="utf-8")
        png = CHECK_DIR / f"pipelines_{token}.png"
        subprocess.run([chrome(), "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={CHECK_DIR / 'profile'}", f"--window-size=1600,{height}",
                        "--virtual-time-budget=20000", f"--screenshot={png}", probe.as_uri() + f"#{token}"],
                       capture_output=True, text=True, timeout=300)
        out.append(png)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--shot", action="store_true", help="screenshots of both heads into v3/work/tmp/flowcheck/")
    args = ap.parse_args()
    if args.self_test:
        return selfTest()
    res = collectAll()
    OUT_JSON.write_text(json.dumps(res, separators=(",", ":")))
    text = page(res)
    OUT_HTML.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_HTML} ({len(text):,} bytes)")
    if args.shot:
        for p in shots(res):
            print("shot", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
