#!/usr/bin/env python3
"""cases.py — case writer of `att_head` (one attention head): stimulus -> model run -> GATE 1 against the oracle ->
the testbench's files (docs/simulator.md §4.8; top/spec.md §8).

The test graph is the head workload (simulator/models/attentionHead.py) with its environment: an E0 source per GEMM
copy, the start source, the O8 sink. GATE 1 compares the model's O8 rows with `attention_ref.head_stages` stage O8, and
the streams inside: the Q / K / V log bytes in the DIMMs' operand memories, every score tile, every run's log P, every
o_part tile.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "oracle"))
import attention_ref as aref  # noqa: E402  (the oracle: used here only, never by the model)

from simulator.core.engine import portWords, run  # noqa: E402
from simulator.models.attentionHead import AttentionHeadParams, attentionHead, headParams  # noqa: E402


@dataclass
class Case:
    """One head case: the head's parameters (headParams keywords), the inputs, the arrival schedule."""
    name: str
    shape: dict                      # headParams keywords: NSEQ, DM, D, R, C, and optionally the mapping
    X: np.ndarray
    WQ: np.ndarray
    WK: np.ndarray
    WV: np.ndarray
    xAvailableAt: object = 0         # one time or per token: the token's earliest burst
    xPeriod: int | None = None       # a copy's burst spacing
    tStart: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def params(self) -> AttentionHeadParams:
        return headParams(**self.shape)


def build(case: Case, withValues: bool = True) -> tuple:
    return attentionHead(case.params, X=case.X, WQ=case.WQ, WK=case.WK, WV=case.WV, withValues=withValues,
                         xAvailableAt=case.xAvailableAt, xPeriod=case.xPeriod, tStart=case.tStart)


def same(name: str, got, want) -> None:
    if not np.array_equal(np.asarray(got, dtype=np.int64), np.asarray(want, dtype=np.int64)):
        raise RuntimeError(f"GATE 1: {name}: model != oracle")


def runCase(case: Case) -> dict:
    """Model run + GATE 1 (top/spec.md §8). Returns the report, the parts and the oracle's stages."""
    g, parts = build(case)
    rep = run(g)
    hp = case.params
    S, B, BKV, D = hp.S, hp.B, hp.BKV, hp.D
    oracle = aref.head_stages(case.X, case.WQ, case.WK, case.WV, np.zeros((D, hp.DM), dtype=np.int8), BKV, hp.R, hp.C)
    qk, pv, sm = parts["qk"], parts["pv"], parts["sm"]
    for r in range(hp.RB):
        rows = slice(r * S, (r + 1) * S)
        same(f"{case.name} run {r}: L_Q (QK A memory)", qk.math.LA[r], oracle["LQ"][rows])
        same(f"{case.name} run {r}: log P", sm.math.stages[r]["logp"], oracle["softmax"]["logp"][:, rows, :])
    for b in range(B):
        keys = slice(b * BKV, (b + 1) * BKV)
        same(f"{case.name} block {b}: L_K^T (QK B memory)", qk.math.LB[b], oracle["LK"][keys].T)
        same(f"{case.name} block {b}: L_V (PV B memory)", pv.math.LB[b], oracle["LV"][keys])
    for j in range(hp.RB * B):
        r, b = divmod(j, B)
        rows, keys = slice(r * S, (r + 1) * S), slice(b * BKV, (b + 1) * BKV)
        same(f"{case.name} job {j}: s8", qk.math.outputs[j].reshape(S, BKV), oracle["s8"][rows, keys])
        same(f"{case.name} job {j}: o_part", pv.math.outputs[j].reshape(S, D), oracle["o_part"][b][rows])
    acc = parts["acc"]
    received = sorted(parts["sink"].received, key=lambda w: (w.unit, w.window))   # E9: per run, one window per wave
    if [(w.unit, w.window) for w in received] != [(r, w) for r in range(hp.RB) for w in range(acc.d.W)]:
        raise RuntimeError(f"GATE 1: {case.name}: E9 windows missing")
    o8 = np.concatenate([acc.math.e9Rows([w.values for w in received if w.unit == r]) for r in range(hp.RB)])
    same(f"{case.name}: O8", o8, oracle["O8"])
    return dict(report=rep, parts=parts, oracle=oracle)


def hexWord(values, bits: int = 8) -> str:
    v = 0
    for j, x in enumerate(values):
        v |= (int(x) & ((1 << bits) - 1)) << (bits * j)
    return f"{v:x}"


def words(values, nWords: int) -> list:
    v = [int(x) for x in np.asarray(values).reshape(-1)] + [0] * (5 * nWords - np.size(values))
    return [hexWord(v[5 * k:5 * k + 5]) for k in range(nWords)]


def defines(case: Case) -> dict:
    hp = case.params
    keys = dict(NSEQ="NSEQ", DM="DM", D="D", BKV="BKV", R="R", C="C", NEXP="N_EXP", NE="N_E", NEP="N_EP", NOC="N_OC",
                PP="P_PROJ", XS="X_S")
    return {f"{k}_TB": int(getattr(hp, v)) for k, v in keys.items()}


def writeTbFiles(case: Case, outDir: Path) -> dict:
    """Model run + GATE 1, then the testbench's files (tb_att_head.v) and the model's events."""
    r = runCase(case)
    rep, parts = r["report"], r["parts"]
    hp = case.params
    n, P_PROJ, V, R, LOADX, LW = hp.NSEQ, hp.P_PROJ, hp.V, hp.R, hp.LOADX, parts["acc"].d.LW
    creditTimes = {}
    for e in rep.events:
        if e.kind == "cr":
            creditTimes.setdefault(f"{e.instance}.{e.port}", []).append(e.time)
    for key, ts in creditTimes.items():              # one pulse per cycle on a credit wire
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise RuntimeError(f"{case.name}: two credits of {key} in one cycle (the RTL's pulse cannot carry them)")
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=True)
    for name, W in (("wq", case.WQ), ("wk", case.WK), ("wv", case.WV)):
        (out / f"{name}.mem").write_text("".join(f"{int(x) & 0xFF:02x}\n" for x in np.asarray(W).reshape(-1)))
    burst = {}                                       # (copy, token) -> the burst's first cycle
    for e in rep.events:
        if e.kind == "win" and e.port == "x" and e.instance[0] == "x" and e.window == 0:
            burst[(int(e.instance[1:]), e.unit)] = e.time
    padded = np.zeros((n, V * R), dtype=np.int64)
    padded[:, :hp.DM] = np.asarray(case.X, dtype=np.int64)
    with open(out / "x_win.txt", "w") as f:          # sorted by copy, then token: t copy token, then V x LOADX words
        for copy in range(P_PROJ):
            for k in range(hp.T_X):
                t = copy + k * P_PROJ
                lanes = [w for v in range(V) for w in words(padded[t, v * R:v * R + hp.ROWS_X], LOADX)]
                f.write(f"{burst[(copy, k)]} {copy} {k} " + " ".join(lanes) + "\n")
    nOc = hp.N_OC
    with open(out / "o_win.txt", "w") as f:          # E9 (values == the oracle's O8, GATE 1): t run wave, LW words
        for w in sorted(parts["sink"].received, key=lambda w: (w.unit, w.window)):
            v = np.asarray(w.values).reshape(LW, 5 * nOc)
            f.write(f"{w.time} {w.unit} {w.window} " + " ".join(hexWord(v[k]) for k in range(LW)) + "\n")
    (out / "start.txt").write_text(f"{case.tStart} {parts['ctrl'].done}\n")
    (out / "events.txt").write_text("".join(line + "\n" for line in rep.eventLines()))
    plusargs = dict(NBUR=n, TMAX=rep.cycles + 5000, NOWIN=len(parts["sink"].received))     # NOWIN: E9 windows
    cj = dict(name=case.name, defines=defines(case), plusargs=plusargs, port_words=portWords(build(case, False)[0]),
              cycles=rep.cycles, done=parts["ctrl"].done, shape=case.shape, n_xbar=hp.nXbar, meta=case.meta,
              events=len(rep.events))
    (out / "case.json").write_text(json.dumps(cj, indent=1) + "\n")
    return cj


# ---------------------------------------------------------------------------------------------------------------------
def randomCase(name: str, shape: dict, rng, cls: str = "random", **kw) -> Case:
    hp = headParams(**shape)
    n, dm, d = hp.NSEQ, hp.DM, hp.D
    lo, hi = {"random": (-128, 128), "small": (-6, 7), "zeros": (0, 1)}[cls]
    X = rng.integers(lo, hi, size=(n, dm)).astype(np.int8)
    W = [rng.integers(lo, hi, size=(dm, d)).astype(np.int8) for _ in range(3)]
    return Case(name=name, shape=dict(shape), X=X, WQ=W[0], WK=W[1], WV=W[2], **kw)


def _smoke() -> None:
    rng = np.random.default_rng(1)
    for shape in (dict(NSEQ=32, DM=64, D=16, R=64, C=64), dict(NSEQ=64, DM=32, D=32, R=64, C=64, P_PROJ=2),
                  dict(NSEQ=128, DM=128, D=64, R=64, C=64), dict(NSEQ=32, DM=32, D=16, R=64, C=64, P_PROJ=4)):
        c = randomCase("smoke", shape, rng, cls="small")
        r = runCase(c)
        rep = r["report"]
        print(shape, "cycles", rep.cycles, "events", len(rep.events), "done", r["parts"]["ctrl"].done,
              "xbars", c.params.nXbar)


if __name__ == "__main__":
    _smoke()
