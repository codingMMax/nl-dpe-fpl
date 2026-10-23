#!/usr/bin/env python3
"""cases.py — case writer of `softmax_online_top`: stimulus -> model run -> GATE 1 against the oracle -> the
testbench's files (docs/simulator.md §4.8; ip/softmax/spec.md §8).

The module's test graph: a score source (tiles on S_CH channels) and a start source -> `softmax_online_top` -> three
sinks (log P, log a, log l), each returning its credit `creditLatency` cycles after a unit's last word. GATE 1
compares, per run, every stage the model computed and every value it sent with
`softmax_online_ref.logcarry_stage_values` on the run's [S, N] scores.
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
import softmax_online_ref as oref  # noqa: E402  (the oracle: used here only, never by the model)

from simulator.core.engine import portWords, run  # noqa: E402
from simulator.core.environment import Sink, Source  # noqa: E402
from simulator.core.graph import Graph  # noqa: E402
from simulator.core.records import planeWords  # noqa: E402
from simulator.models.softmaxOnline import SoftmaxOnline, SoftmaxOnlineParams  # noqa: E402

STAGES = ("m_blk", "m_new", "log_a", "logp", "sp", "t", "e", "lprime", "lq", "log_l", "m", "log_l_final")


@dataclass
class Case:
    """One module case: parameters (the completed RTL parameter dict), the scores of every run, and the arrival /
    credit schedule."""
    name: str
    params: dict
    scores: list                    # per run: [S, N] int8 scores
    sAvailableAt: object = 0        # per tile, or one time: the tile's earliest first word
    sPeriod: int | None = None      # per channel window spacing (the producer's pace)
    goAvailableAt: object = 0       # per run, or one time: the earliest start
    lpCreditLatency: int = 1
    laCreditLatency: int = 1
    llCreditLatency: int = 1
    meta: dict = field(default_factory=dict)


def completeParams(params: dict) -> dict:
    """The RTL parameter dict with the module's defaults filled in."""
    return asdict(SoftmaxOnlineParams(**params))


def tilesOf(case: Case, B: int, BKV: int) -> list:
    """The stream's tiles in order: run r, block b = scores[r][:, b BKV .. (b + 1) BKV]."""
    return [np.asarray(s, dtype=np.int64)[:, b * BKV:(b + 1) * BKV] for s in case.scores for b in range(B)]


def build(case: Case, withValues: bool = True, carryExtent: int | None = None) -> tuple:
    """(graph, softmax, sinks) of a case."""
    p = SoftmaxOnlineParams(**case.params)
    runs = len(case.scores)
    sm = SoftmaxOnline("sm", p, runs=runs, withValues=withValues, carryExtent=carryExtent)
    d = sm.d
    ports = sm.ports()
    tiles = tilesOf(case, d.B, p.BKV)
    I = d.I

    def tileWindow(u, w):
        return tiles[u].reshape(-1)[w * I:(w + 1) * I].astype(np.int64)
    src = Source("s_src", replace(ports["s"], direction="out"), len(tiles), d.NWB, availableAt=case.sAvailableAt,
                 period=case.sPeriod, payload=tileWindow if withValues else None, withValues=withValues)
    go = Source("go_src", replace(ports["go"], direction="out"), runs, 1, availableAt=case.goAvailableAt,
                withValues=withValues)
    lp = Sink("lp_snk", replace(ports["lp"], direction="in"), len(tiles), d.NWB, creditLatency=case.lpCreditLatency,
              withValues=withValues)
    la = Sink("la_snk", replace(ports["la"], direction="in"), runs * (d.B - 1), 1, creditLatency=case.laCreditLatency,
              withValues=withValues)
    ll = Sink("ll_snk", replace(ports["ll"], direction="in"), runs, 1, creditLatency=case.llCreditLatency,
              withValues=withValues)
    g = Graph(case.name, [src, go, sm, lp, la, ll],
              [("s_src.s", "sm.s"), ("go_src.go", "sm.go"), ("sm.lp", "lp_snk.lp"), ("sm.la", "la_snk.la"),
               ("sm.ll", "ll_snk.ll")])
    return g, sm, dict(lp=lp, la=la, ll=ll)


def runCase(case: Case) -> dict:
    """Model run + GATE 1 (every stage and every sent value, every run). Returns the report and the model."""
    g, sm, sinks = build(case)
    rep = run(g)
    p, d = sm.p, sm.d
    S, BKV, B = p.S, p.BKV, d.B
    for r, scores in enumerate(case.scores):
        want = oref.logcarry_stage_values(np.asarray(scores, dtype=np.int64), BKV)
        got = sm.math.stages[r]
        for k in STAGES:
            if not np.array_equal(np.asarray(got[k]), np.asarray(want[k])):
                raise RuntimeError(f"GATE 1: {case.name} run {r}: stage {k}: model != oracle")
        for b in range(B):
            j = r * B + b
            ws = sorted((w for w in sinks["lp"].received if w.unit == j), key=lambda w: w.window)
            logP = np.concatenate([w.values for w in ws]).reshape(S, BKV)
            if not np.array_equal(logP, want["logp"][b]):
                raise RuntimeError(f"GATE 1: {case.name} run {r} block {b}: sent log P != oracle")
            if b:
                u = r * (B - 1) + b - 1
                (wa,) = [w for w in sinks["la"].received if w.unit == u]
                if [int(x) for x in wa.values] != [int(x) for x in want["log_a"][b]]:
                    raise RuntimeError(f"GATE 1: {case.name} run {r} block {b}: sent log a != oracle")
        (wl,) = [w for w in sinks["ll"].received if w.unit == r]
        if not np.array_equal(np.asarray(wl.values), want["log_l_final"]):
            raise RuntimeError(f"GATE 1: {case.name} run {r}: sent log l != oracle")
    return dict(report=rep, sm=sm, sinks=sinks)


def defines(case: Case) -> dict:
    p = case.params
    keys = dict(S="S", BKV="BKV", N="N", R="R", C="C", NE="N_EXP", SCH="S_CH", LPC="LP_CAP", LAC="LA_CAP",
                LLC="LL_CAP")
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
    """Model run + GATE 1, then the testbench's files (tb_softmax_online_top.v) and the model's events."""
    r = runCase(case)
    rep, sm, sinks = r["report"], r["sm"], r["sinks"]
    p, d = sm.p, sm.d
    creditPulsesDistinct(rep, case.name)
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=True)
    I = d.I
    tiles = tilesOf(case, d.B, p.BKV)
    sent = {}
    for e in rep.events:
        if e.kind == "win" and e.instance in ("s_src", "go_src"):
            sent.setdefault(f"{e.instance}.{e.port}", []).append((e.time, e.channel, e.unit, e.window))

    def planes(values, E, W):                                     # a window's 8 plane words, hex
        return [f"{x:x}" for x in planeWords(values, E, W, p.P)]
    sRows = sorted(sent.get("s_src.s", []), key=lambda x: (x[2], x[3]))
    with open(out / "s_win.txt", "w") as f:
        for (t, ch, u, w) in sRows:
            f.write(f"{t} {ch} {u} {w} " + " ".join(planes(tiles[u].reshape(-1)[w * I:(w + 1) * I], 8, I)) + "\n")
    goRows = sorted(sent.get("go_src.go", []), key=lambda x: x[2])
    (out / "go.txt").write_text("".join(f"{t}\n" for (t, _, _, _) in goRows))
    with open(out / "lp_exp.txt", "w") as f:                     # per block, window-major: 8 plane words each
        for j in range(len(tiles)):
            ws = sorted((w for w in sinks["lp"].received if w.unit == j), key=lambda w: w.window)
            for w in ws:
                f.write("\n".join(planes(w.values, 8, I)) + "\n")
    with open(out / "la_exp.txt", "w") as f:                     # per unit: 8 plane words of 2 lanes (9-bit log a)
        for w in sorted(sinks["la"].received, key=lambda w: w.unit):
            f.write("\n".join(planes(w.values, 9, p.S)) + "\n")
    with open(out / "ll_exp.txt", "w") as f:                     # per run: 8 plane words of S bits
        for w in sorted(sinks["ll"].received, key=lambda w: w.unit):
            f.write("\n".join(planes(w.values, 8, p.S)) + "\n")
    (out / "events.txt").write_text("".join(line + "\n" for line in rep.eventLines()))
    plusargs = dict(NS=len(sRows), NRUN=len(case.scores), NBLK=len(tiles), NLA=len(sinks["la"].received),
                    LPCR=case.lpCreditLatency, LACR=case.laCreditLatency, LLCR=case.llCreditLatency,
                    TMAX=rep.cycles + 2000)
    cj = dict(name=case.name, defines=defines(case), **plusargs, plusargs=plusargs,
              port_words=portWords(build(case, withValues=False)[0]), cycles=rep.cycles, params=case.params,
              meta=case.meta, events=len(rep.events))
    (out / "case.json").write_text(json.dumps(cj, indent=1) + "\n")
    return cj


# ---------------------------------------------------------------------------------------------------------------------
def randomScores(S: int, N: int, rng, cls: str = "random") -> np.ndarray:
    if cls == "zeros":
        return np.zeros((S, N), dtype=np.int64)
    if cls == "extremes":
        return rng.choice(np.array([-128, -127, -1, 0, 1, 126, 127]), size=(S, N))
    if cls == "small":
        return rng.integers(-16, 17, size=(S, N))
    if cls == "rising":            # each block's max above the last: log a < 0 every block
        ramp = (np.arange(N) * 200) // N - 100
        return np.clip(rng.integers(-8, 9, size=(S, N)) + ramp[None, :], -128, 127)
    return rng.integers(-128, 128, size=(S, N))


def randomCase(name: str, params: dict, runs: int, rng, cls: str = "random", **kw) -> Case:
    p = completeParams(params)
    return Case(name=name, params=p, scores=[randomScores(p["S"], p["N"], rng, cls) for _ in range(runs)], **kw)


def _smoke() -> None:
    rng = np.random.default_rng(1)
    for (S, N, sch, ne) in ((32, 64, 1, 2), (64, 64, 2, 2), (8, 32, 1, 2), (16, 128, 2, 4)):
        p = dict(S=S, BKV=16, N=N, R=64, C=64, N_EXP=ne, S_CH=sch)
        c = randomCase(f"sm_smoke_{S}_{N}", p, 2, rng)
        r = runCase(c)
        print(S, N, sch, ne, r["report"].cycles, [s.events["done"] for s in r["report"].steps["sm"]
                                                   if s.process == "run"])


if __name__ == "__main__":
    _smoke()
