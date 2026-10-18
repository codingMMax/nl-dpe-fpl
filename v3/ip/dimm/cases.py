#!/usr/bin/env python3
"""cases.py — case writer of `dimm_tile_top`: stimulus -> model run -> GATE 1 against the oracle -> the testbench's
files (docs/simulator.md §4.8; ip/dimm/spec.md §8).

The module's test graph: three sources (A rows / windows, B rows, jobs) -> `dimm_tile_top` -> a C sink. GATE 1 compares
every job's C with the oracle composition: L = attention_ref._log_i8(x) for an int8 operand (else the given log
bytes), signs attention_ref.sign_from_log (or 0), attention_ref.signed_tile, then attention_ref.narrow.
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
from simulator.models.dimmTile import DimmTile, DimmTileParams  # noqa: E402


@dataclass
class Case:
    """One module case: parameters (the completed RTL parameter dict), operands, jobs and the arrival schedule."""
    name: str
    params: dict
    A: list                         # A matrices [M, K] (int8 values or log bytes)
    B: list                         # B matrices [K, N]
    jobs: list                      # (a_slot, b_slot, release_a)
    aAvailableAt: object = 0        # per A credit unit, or one time
    bAvailableAt: object = 0
    jAvailableAt: object = 0
    aPeriod: int | None = None      # per channel window spacing (int8 ports: >= the converter period)
    bPeriod: int | None = None
    cCreditLatency: int = 1         # the sink's credit delay
    meta: dict = field(default_factory=dict)


def completeParams(params: dict) -> dict:
    """The RTL parameter dict with the module's defaults filled in."""
    return asdict(DimmTileParams(**params))


def convPeriod(L: int, C: int, BUF: int = 40, P: int = 8) -> int:
    """The converter's pass period for a row of L elements on a C-column array (spec §3: the int8 port contract): the
    load of L elements, the accumulator, and the drain, which walks every column of the array (2026-10-18; until then
    the drain stopped after the L columns in use, so the period did not depend on C)."""
    load, drain = -(-8 * L // BUF), -(-8 * C // BUF)
    return max(load + P, P + 2, drain + 1)


def aStream(case: Case) -> list:
    """The A port's windows in stream order."""
    p = case.params
    out = []
    for A in case.A:
        A = np.asarray(A, dtype=np.int64)
        if p.get("A_FRAME", 0) == 0:
            out += [A[r] for r in range(p["M"])]
        else:
            rows = p["A_ROWS"]
            out += [A[i * rows:(i + 1) * rows].reshape(-1) for i in range(p["M"] // rows)]
    return out


def bStream(case: Case) -> list:
    p = case.params
    out = []
    for B in case.B:
        B = np.asarray(B, dtype=np.int64)
        out += [B[:, n] for n in range(p["N"])] if p.get("TRANS_B", 1) else [B[k] for k in range(p["K"])]
    return out


def build(case: Case, withValues: bool = True) -> tuple:
    """(graph, dimm, sink) of a case."""
    params = DimmTileParams(**case.params)
    streamA, streamB = aStream(case), bStream(case)
    dimm = DimmTile("dimm", params, nA=len(streamA), nB=len(streamB), nJobs=len(case.jobs), withValues=withValues)
    d = dimm.d
    ports = dimm.ports()
    if len(streamA) % d.A_UNIT or len(streamB) % d.B_UNIT:
        raise ValueError(f"{case.name}: the streams must be whole credit units")
    srcA = Source("a_src", replace(ports["a"], direction="out"), len(streamA) // d.A_UNIT, d.A_UNIT,
                  availableAt=case.aAvailableAt, period=case.aPeriod,
                  payload=(lambda u, w: streamA[u * d.A_UNIT + w].astype(np.int64)) if withValues else None,
                  withValues=withValues)
    srcB = Source("b_src", replace(ports["b"], direction="out"), len(streamB) // d.B_UNIT, d.B_UNIT,
                  availableAt=case.bAvailableAt, period=case.bPeriod,
                  payload=(lambda u, w: streamB[u * d.B_UNIT + w].astype(np.int64)) if withValues else None,
                  withValues=withValues)
    srcJ = Source("j_src", replace(ports["j"], direction="out"), len(case.jobs), 1, availableAt=case.jAvailableAt,
                  payload=lambda u, w: np.array(case.jobs[u], dtype=np.int64), withValues=withValues)
    sink = Sink("c_snk", replace(ports["c"], direction="in"), len(case.jobs), d.W_K, creditLatency=case.cCreditLatency,
                withValues=withValues)
    g = Graph(case.name, [srcA, srcB, srcJ, dimm, sink],
              [("a_src.a", "dimm.a"), ("b_src.b", "dimm.b"), ("j_src.j", "dimm.j"), ("dimm.c", "c_snk.c")])
    return g, dimm, sink


def oracle(case: Case) -> list:
    """Per job, C [M, N] by the oracle composition."""
    p = case.params
    LAs = [aref._log_i8(np.asarray(A, dtype=np.int64)) if p["A_DOM"] == 0 else np.asarray(A, dtype=np.int64)
           for A in case.A]
    LBs = [aref._log_i8(np.asarray(B, dtype=np.int64)) if p["B_DOM"] == 0 else np.asarray(B, dtype=np.int64)
           for B in case.B]
    out, released = [], {}
    for slotA, slotB, releaseA in case.jobs:
        aMatrix = slotA + p["A_SLOTS"] * released.get(slotA, 0) if p["A_FRAME"] == 1 else slotA
        if releaseA:
            released[slotA] = released.get(slotA, 0) + 1
        LA, LB = LAs[aMatrix], LBs[slotB]
        signA = aref.sign_from_log(LA) if p["A_SIGNED"] else np.zeros_like(LA)
        signB = aref.sign_from_log(LB) if p["B_SIGNED"] else np.zeros_like(LB)
        acc = aref.signed_tile(LA, signA, LB, signB)[0]
        out.append(aref.narrow(acc, p["NARROW"]) if p["NARROW"] else acc)
    return out


def modelOutputs(case: Case, dimm: DimmTile, sink: Sink) -> list:
    """Per job, C [M, N] from the sink's windows (window w = rows w*NB .. of C)."""
    p = dimm.p
    out = []
    for j in range(len(case.jobs)):
        ws = sorted((w for w in sink.received if w.unit == j), key=lambda w: w.window)
        out.append(np.concatenate([w.values for w in ws]).reshape(p.M, p.N))
    return out


def runCase(case: Case) -> dict:
    """Model run + GATE 1. Returns the report and the model."""
    g, dimm, sink = build(case)
    rep = run(g)
    got, want = modelOutputs(case, dimm, sink), oracle(case)
    for j, (a, b) in enumerate(zip(got, want)):
        if not np.array_equal(a, b):
            raise RuntimeError(f"GATE 1: {case.name} job {j}: model C != oracle")
    return dict(report=rep, dimm=dimm, sink=sink, C=got)


def hexWord(values, bits: int) -> str:
    """Values (lane j at bits j*bits) as one hex word, two's complement per lane."""
    v = 0
    for j, x in enumerate(values):
        v |= (int(x) & ((1 << bits) - 1)) << (bits * j)
    return f"{v:x}"


def defines(case: Case) -> dict:
    p = case.params
    keys = dict(M="M", N="N", K="K", R="R", C="C", NE="N_E", XS="X_S", ADOM="A_DOM", ASIG="A_SIGNED", ACH="A_CH",
                ASL="A_SLOTS", AFR="A_FRAME", AROWS="A_ROWS", BDOM="B_DOM", BSIG="B_SIGNED", BCH="B_CH",
                BSL="B_SLOTS", TRB="TRANS_B", NAR="NARROW", CCAP="C_CAP")
    return {f"{k}_TB": int(p[v]) for k, v in keys.items()}


def writeTbFiles(case: Case, outDir: Path) -> dict:
    """Model run + GATE 1, then the testbench's files (tb_dimm_tile_top.v) and the model's events."""
    r = runCase(case)
    rep, dimm, sink = r["report"], r["dimm"], r["sink"]
    d = dimm.d
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=True)
    g = build(case)[0]
    # the windows each source sent, with their cycles (from the model's port events)
    sent = {}
    for e in rep.events:
        if e.kind == "win" and e.instance in ("a_src", "b_src", "j_src"):
            sent.setdefault(f"{e.instance}.{e.port}", []).append((e.time, e.channel, e.unit, e.window))
    streamA, streamB = aStream(case), bStream(case)

    def words(values, nWords):
        v = list(np.asarray(values, dtype=np.int64)) + [0] * (5 * nWords - len(values))
        return [hexWord(v[5 * k:5 * k + 5], 8) for k in range(nWords)]
    wordsA, wordsB = -(-d.A_W // 5), -(-d.LB // 5)
    aRows = sorted(sent.get("a_src.a", []), key=lambda x: (x[2], x[3]))
    with open(out / "a_win.txt", "w") as f:
        for (t, ch, u, w) in aRows:
            f.write(f"{t} {ch} {u} {w} " + " ".join(words(streamA[u * d.A_UNIT + w], wordsA)) + "\n")
    bRows = sorted(sent.get("b_src.b", []), key=lambda x: (x[2], x[3]))
    with open(out / "b_win.txt", "w") as f:
        for (t, ch, u, w) in bRows:
            f.write(f"{t} {ch} {u} {w} " + " ".join(words(streamB[u * d.B_UNIT + w], wordsB)) + "\n")
    jRows = sorted(sent.get("j_src.j", []), key=lambda x: x[2])
    with open(out / "j_win.txt", "w") as f:
        for (t, ch, u, w) in jRows:
            a, b, releaseA = case.jobs[u]
            f.write(f"{t} {a} {b} {releaseA}\n")
    with open(out / "c_exp.txt", "w") as f:
        for j in range(len(case.jobs)):
            ws = sorted((w for w in sink.received if w.unit == j), key=lambda w: w.window)
            assert [w.window for w in ws] == list(range(d.W_K))
            for w in ws:
                v = list(w.values) + [0] * (5 * d.LW - len(w.values))
                for k in range(d.LW):
                    f.write(hexWord(v[5 * k:5 * k + 5], d.EC) + "\n")
    images = dimm.math.memoryImages()
    with open(out / "mem.txt", "w") as f:
        for (s_, kind, m, addr), byts in sorted(images.items()):
            f.write(f"{0 if kind == 'A' else 1} {s_} {m} {addr} {hexWord(byts, 8)}\n")
    (out / "events.txt").write_text("".join(line + "\n" for line in rep.eventLines()))
    plusargs = dict(NA=len(aRows), NBW=len(bRows), NJ=len(case.jobs), NMEM=len(images), CRLAT=case.cCreditLatency,
                    TMAX=rep.cycles + 2000)
    cj = dict(name=case.name, defines=defines(case), **plusargs, plusargs=plusargs, port_words=portWords(g),
              cycles=rep.cycles, params=case.params, meta=case.meta, events=len(rep.events))
    (out / "case.json").write_text(json.dumps(cj, indent=1) + "\n")
    return cj


# ---------------------------------------------------------------------------------------------------------------------
# configurations of the head (spec §9) at a crossbar, and random operands
# ---------------------------------------------------------------------------------------------------------------------
def qkParams(S: int, d: int, Bkv: int, R: int, C: int, nE: int, pProj: int = 1, runs: int = 1, blocks: int = 1,
             cCap: int = 2) -> dict:
    return dict(M=S, N=Bkv, K=d, R=R, C=C, N_E=nE, A_DOM=0, A_SIGNED=1, A_CH=pProj, A_SLOTS=runs, A_FRAME=0,
                A_ROWS=1, B_DOM=0, B_SIGNED=1, B_CH=pProj, B_SLOTS=blocks, TRANS_B=1, NARROW=5, C_CAP=cCap)


def pvParams(S: int, d: int, Bkv: int, R: int, C: int, nEp: int, nExp: int = 2, pProj: int = 1, blocks: int = 1,
             cCap: int = 2) -> dict:
    I = min(R, C)
    windowElements = min(S * Bkv, I)
    return dict(M=S, N=d, K=Bkv, R=R, C=C, N_E=nEp, A_DOM=1, A_SIGNED=0, A_CH=nExp, A_SLOTS=2, A_FRAME=1,
                A_ROWS=windowElements // Bkv, B_DOM=0, B_SIGNED=1, B_CH=pProj, B_SLOTS=blocks, TRANS_B=0, NARROW=0,
                C_CAP=cCap)


def randomCase(name: str, params: dict, nA: int, nB: int, jobs: list, rng, cls: str = "random", **kw) -> Case:
    p = completeParams(params)
    M, N, K = p["M"], p["N"], p["K"]

    def matrix(shape, domain, signed):
        if cls == "zeros":
            return np.zeros(shape, dtype=np.int64)
        if domain == 1 and not signed:              # log|P| <= 0
            if cls == "extremes":
                return rng.choice(np.array([-128, 0]), size=shape)
            return rng.integers(-128, 1, size=shape)
        if cls == "extremes":
            return rng.choice(np.array([-128, -1, 0, 1, 127]), size=shape)
        return rng.integers(-128, 128, size=shape)
    A = [matrix((M, K), p["A_DOM"], p["A_SIGNED"]) for _ in range(nA)]
    B = [matrix((K, N), p["B_DOM"], p["B_SIGNED"]) for _ in range(nB)]
    return Case(name=name, params=p, A=A, B=B, jobs=jobs, **kw)


def _smoke() -> None:
    rng = np.random.default_rng(1)
    # QK-like: S 32, d 16, Bkv 16 on 64x64 (I 64): NB 4, W_K 8; N_E 32: NG 2, T_W 4, 2 slices of 8 columns
    p = qkParams(32, 16, 16, 64, 64, 32, pProj=2, runs=2, blocks=2)
    jobs = [(r, b, 0) for r in range(2) for b in range(2)]
    c = randomCase("qk_smoke", p, 2, 2, jobs, rng, aPeriod=convPeriod(16, 64), bPeriod=convPeriod(16, 64))
    r = runCase(c)
    print("qk", r["report"].cycles, [s.events["t_go"] for s in r["report"].steps["dimm"] if s.process == "jobs"])
    # PV-like (spec §9, N_EP = 2d): S 32, d 16, Bkv 16 on 64x64: NB 4, W_K 8, NG 2, E_w 8 words per window per slice;
    # the softmax sends a window per EXP lane every convPeriod(I, C) = 21 cycles
    for d_, nEp in ((16, 32), (64, 128)):
        p = pvParams(32, d_, 16, 64, 64, nEp, nExp=2, blocks=2)
        jobs = [(j % 2, j % 2, 1) for j in range(4)]
        c = randomCase(f"pv_smoke_d{d_}", p, 4, 2, jobs, rng, aPeriod=convPeriod(64, 64), bPeriod=convPeriod(d_, 64))
        r = runCase(c)
        print(f"pv d{d_}", r["report"].cycles, [s.events["t_go"] for s in r["report"].steps["dimm"] if s.process == "jobs"])


if __name__ == "__main__":
    _smoke()
