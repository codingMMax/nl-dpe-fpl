#!/usr/bin/env python3
"""acc_cases.py — case writer of `softmax_online_acc`: stimulus -> model run -> GATE 1 against the oracle -> the
testbench's files (docs/simulator.md §4.8; ip/softmax/acc_spec.md §8).

The module's test graph: three sources (o_part tiles on P_CH channels, log a units, log l units) -> `softmax_online_acc`
-> one sink (E9: one plane window of N_OC I O8 values per epilogue wave, push). A source sends each unit at its availability
time, at its pace, and only with a credit in hand. GATE 1 compares, per run, every stage the model computed and every
O8 window it sent with `attention_ref.ocarry_stage_values` and `attention_ref.epilogue`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "oracle"))
import attention_ref as aref  # noqa: E402  (the oracle: used here only, never by the model)

from simulator.core.engine import portWords, run  # noqa: E402
from simulator.core.environment import Sink, Source  # noqa: E402
from simulator.core.graph import Graph  # noqa: E402
from simulator.core.records import planeWords  # noqa: E402
from simulator.models.softmaxAccumulator import SoftmaxAccParams, SoftmaxOnlineAcc  # noqa: E402

STAGES = ("t", "e", "o_new", "oq", "so", "lo")
EPILOGUE = ("tE", "eE", "O8")


@dataclass
class Case:
    """One module case: parameters (the completed RTL parameter dict), the inputs of every run, and the arrival
    schedule."""
    name: str
    params: dict
    runs: list                      # per run: dict(o_part [NBLK, S, D], log_a [NBLK, S] (row 0 unused), log_l [S])
    pAvailableAt: object = 0        # per tile, or one time: the tile's earliest first word
    pPeriod: int | None = None      # per channel window spacing (the PV DIMM's wave period)
    laAvailableAt: object = 0       # per alpha unit, or one time
    laPeriod: int | None = None     # spacing of the alpha windows (one per unit)
    llAvailableAt: object = 0       # per run, or one time
    meta: dict = field(default_factory=dict)


def completeParams(params: dict) -> dict:
    """The RTL parameter dict with the module's defaults filled in."""
    return asdict(SoftmaxAccParams(**params))


def build(case: Case, withValues: bool = True) -> tuple:
    """(graph, accumulator, sink) of a case."""
    p = SoftmaxAccParams(**case.params)
    runs = len(case.runs)
    acc = SoftmaxOnlineAcc("acc", p, runs=runs, withValues=withValues)
    d, ports, B, S, I = acc.d, acc.ports(), p.NBLK, p.S, acc.d.I

    def pWindow(u, w):              # tile u = r B + b, window w: elements w I .. of o_part[b], row-major
        r, b = divmod(u, B)
        return np.asarray(case.runs[r]["o_part"][b], dtype=np.int64).reshape(-1)[w * I:(w + 1) * I]

    def laWindow(u, w):             # unit u = r (B - 1) + b - 1: one window of the S rows' log a
        r, b1 = divmod(u, B - 1)
        return np.asarray(case.runs[r]["log_a"][b1 + 1], dtype=np.int64)

    def llWindow(u, w):
        return np.asarray(case.runs[u]["log_l"], dtype=np.int64)
    sources = [Source("p_src", replace(ports["p"], direction="out"), runs * B, d.P_OC, availableAt=case.pAvailableAt,
                      period=case.pPeriod, payload=pWindow if withValues else None, withValues=withValues),
               Source("la_src", replace(ports["la"], direction="out"), runs * (B - 1), 1, availableAt=case.laAvailableAt,
                      period=case.laPeriod, payload=laWindow if withValues else None, withValues=withValues),
               Source("ll_src", replace(ports["ll"], direction="out"), runs, 1, availableAt=case.llAvailableAt,
                      payload=llWindow if withValues else None, withValues=withValues)]
    sink = Sink("o_snk", replace(ports["o"], direction="in"), runs, d.W, withValues=withValues)
    g = Graph(case.name, sources + [acc, sink], [("p_src.p", "acc.p"), ("la_src.la", "acc.la"), ("ll_src.ll", "acc.ll"),
                                                 ("acc.o", "o_snk.o")])
    return g, acc, sink


def runCase(case: Case) -> dict:
    """Model run + GATE 1 (every stage and every sent O8 window, every run). Returns the report and the model."""
    g, acc, sink = build(case)
    rep = run(g)
    p, d = acc.p, acc.d
    for r, inputs in enumerate(case.runs):
        want = aref.ocarry_stage_values(np.asarray(inputs["o_part"]), np.asarray(inputs["log_a"]), p.LN, p.SO)
        epilogue = aref.epilogue(want["lo_final"], want["so_final"], np.asarray(inputs["log_l"]), p.LN, p.SO)
        got = acc.math.stages[r]
        for k in STAGES:
            if not np.array_equal(got[k], want[k]):
                raise RuntimeError(f"GATE 1: {case.name} run {r}: stage {k}: model != oracle")
        for k in EPILOGUE:
            if not np.array_equal(got[k], epilogue[k]):
                raise RuntimeError(f"GATE 1: {case.name} run {r}: epilogue {k}: model != oracle")
        windows = sorted((w for w in sink.received if w.unit == r), key=lambda w: w.window)
        if [w.window for w in windows] != list(range(d.W)):
            raise RuntimeError(f"GATE 1: {case.name} run {r}: E9 windows missing")
        if not np.array_equal(acc.math.e9Rows([w.values for w in windows]), epilogue["O8"]):
            raise RuntimeError(f"GATE 1: {case.name} run {r}: sent O8 != oracle")
        if sum(int(np.count_nonzero(w.values)) for w in windows) != int(np.count_nonzero(epilogue["O8"])):
            raise RuntimeError(f"GATE 1: {case.name} run {r}: E9 lanes outside the O8 values are not zero")
    return dict(report=rep, acc=acc, sink=sink)


def defines(case: Case) -> dict:
    p = case.params
    keys = dict(S="S", D="D", R="R", C="C", NOC="N_OC", NBLK="NBLK", SO="SO", LN="LN", PCH="P_CH", VWP="VWP")
    return {f"{k}_TB": int(p[v]) for k, v in keys.items()}


def creditPulsesDistinct(rep, name: str) -> None:
    """One pulse per cycle on a credit wire (the RTL's pulse cannot carry two credits in one cycle)."""
    times = {}
    for e in rep.events:
        if e.kind == "cr":
            times.setdefault(f"{e.instance}.{e.port}", []).append(e.time)
    for key, ts in times.items():
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise RuntimeError(f"{name}: two credits of {key} in one cycle (the RTL's pulse cannot carry them)")


def writeTbFiles(case: Case, outDir: Path) -> dict:
    """Model run + GATE 1, then the testbench's files (tb_softmax_online_acc.v) and the model's events."""
    r = runCase(case)
    rep, acc, sink = r["report"], r["acc"], r["sink"]
    p, d = acc.p, acc.d
    creditPulsesDistinct(rep, case.name)
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=True)
    I, B, VWP = d.I, p.NBLK, p.VWP
    sent = {}
    for e in rep.events:
        if e.kind == "win" and e.instance in ("p_src", "la_src", "ll_src"):
            sent.setdefault(f"{e.instance}.{e.port}", []).append((e.time, e.channel, e.unit, e.window))

    def planes(values, E, W):                                     # a window's 8 plane words, hex
        return " ".join(f"{x:x}" for x in planeWords(values, E, W, p.P))
    with open(out / "p_win.txt", "w") as f:                        # t ch unit window, then 8 plane words (2 lanes)
        for (t, ch, u, w) in sorted(sent.get("p_src.p", []), key=lambda x: (x[2], x[3])):
            rr, b = divmod(u, B)
            v = np.asarray(case.runs[rr]["o_part"][b], dtype=np.int64).reshape(-1)[w * I:(w + 1) * I]
            f.write(f"{t} {ch} {u} {w} " + planes(v, VWP, I) + "\n")
    with open(out / "la_win.txt", "w") as f:                       # t unit, then 8 plane words (9-bit log a, 2 lanes)
        for (t, ch, u, w) in sorted(sent.get("la_src.la", []), key=lambda x: x[2]):
            rr, b1 = divmod(u, B - 1)
            f.write(f"{t} {u} " + planes(case.runs[rr]["log_a"][b1 + 1], 9, p.S) + "\n")
    with open(out / "ll_win.txt", "w") as f:                       # t unit, then 8 plane words of S bits
        for (t, ch, u, w) in sorted(sent.get("ll_src.ll", []), key=lambda x: x[2]):
            f.write(f"{t} {u} " + planes(case.runs[u]["log_l"], 8, p.S) + "\n")
    with open(out / "o_win.txt", "w") as f:                        # E9: t run wave, then 8 plane words of N_OC I bits
        for w in sorted(sink.received, key=lambda w: (w.unit, w.window)):
            f.write(f"{w.time} {w.unit} {w.window} " + planes(w.values, 8, p.N_OC * I) + "\n")
    (out / "run_done.txt").write_text("".join(f"{acc.runDoneAt[k]}\n" for k in range(len(case.runs))))
    (out / "events.txt").write_text("".join(line + "\n" for line in rep.eventLines()))
    plusargs = dict(NP=len(sent.get("p_src.p", [])), NLA=len(sent.get("la_src.la", [])), NRUN=len(case.runs),
                    TMAX=rep.cycles + 2000)
    cj = dict(name=case.name, defines=defines(case), plusargs=plusargs,
              port_words=portWords(build(case, withValues=False)[0]), cycles=rep.cycles, params=case.params,
              meta=case.meta, events=len(rep.events), run_done=[acc.runDoneAt[k] for k in range(len(case.runs))])
    (out / "case.json").write_text(json.dumps(cj, indent=1) + "\n")
    return cj


# ---------------------------------------------------------------------------------------------------------------------
def randomRun(p: dict, rng, cls: str = "random") -> dict:
    """One run's inputs. o_part in the PV DIMM's range (|sum of 16 EXP bytes| <= 4080), log a in -255 .. 0 (row 0 of
    block 0 unused), log l int8."""
    B, S, D = p["NBLK"], p["S"], p["D"]
    if cls == "zeros":
        oPart = np.zeros((B, S, D), dtype=np.int64)
    elif cls == "allneg":
        oPart = -rng.integers(1, 4081, size=(B, S, D))
    elif cls == "extremes":
        oPart = rng.choice(np.array([-4080, -1, 0, 1, 4080]), size=(B, S, D))
    elif cls == "small":
        oPart = rng.integers(-40, 41, size=(B, S, D))
    else:
        oPart = rng.integers(-4080, 4081, size=(B, S, D))
    if cls == "alpha0":
        logAlpha = np.zeros((B, S), dtype=np.int64)
    elif cls == "extremes":
        logAlpha = rng.choice(np.array([-255, -129, -128, -127, 0]), size=(B, S))
    else:
        logAlpha = rng.integers(-255, 1, size=(B, S))
    logAlpha[0] = 0
    logL = rng.choice(np.array([-128, 127, 0]), size=S) if cls == "extremes" else rng.integers(-128, 128, size=S)
    return dict(o_part=oPart, log_a=logAlpha, log_l=logL)


def randomCase(name: str, params: dict, runs: int, rng, cls: str = "random", **kw) -> Case:
    p = completeParams(params)
    return Case(name=name, params=p, runs=[randomRun(p, rng, cls) for _ in range(runs)], **kw)


def _smoke() -> None:
    rng = np.random.default_rng(1)
    for (S, D, nOc, B, pch) in ((16, 16, 2, 3, 2), (32, 8, 3, 2, 4), (64, 64, 8, 2, 8), (8, 8, 1, 1, 1)):
        p = dict(S=S, D=D, R=64, C=64, N_OC=nOc, NBLK=B, SO=4, LN=4, P_CH=pch)
        c = randomCase(f"acc_smoke_{S}_{D}", p, 2, rng)
        r = runCase(c)
        print(S, D, nOc, B, pch, r["report"].cycles, [r["acc"].runDoneAt[k] for k in range(2)])


if __name__ == "__main__":
    _smoke()
