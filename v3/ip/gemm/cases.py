#!/usr/bin/env python3
"""cases.py — case writer of `gemm_top`: stimulus -> model run -> GATE 1 against the oracle -> the testbench's files
(ip/gemm/spec.md §7; docs/simulator.md §4.4).

The module's test graph is the FC workload (simulator/models/fc.py): an X row source -> `gemm_top` -> a Y sink. GATE 1
compares every row's output with the oracle (oracle/gemm_ref.compute_gemm: the low byte of X W).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "oracle"))
import gemm_ref  # noqa: E402  (the oracle: used here only, never by the model)

from simulator.core.engine import portWords, run  # noqa: E402
from simulator.core.records import planeWords  # noqa: E402
from simulator.models.fc import extent, fc, result  # noqa: E402

P = 8


@dataclass
class Case:
    """One module case: the shape, the pass extent, the operands and the rows' offer times."""
    name: str
    K: int
    N: int
    R: int
    C: int
    W: np.ndarray                   # [K, N] int8
    X: np.ndarray                   # [M, K] int8
    passExtent: str = "used"        # P29: "used" or "full" (fc.extent)
    availableAt: object = 0         # per row, or one time
    meta: dict = field(default_factory=dict)

    @property
    def M(self) -> int:
        return self.X.shape[0]


def build(case: Case, withValues: bool = True):
    return fc(case.K, case.N, case.R, case.C, M=case.M, P=P, passExtent=case.passExtent, W=case.W, X=case.X,
              withValues=withValues, availableAt=case.availableAt)


def runCase(case: Case) -> dict:
    """Model run + GATE 1. Returns the report, the graph, the model and Y."""
    g = build(case)
    rep = run(g)
    gemm, sink = g.modules[1], g.modules[2]
    Y = result(sink, case.M, case.N, gemm.d.H)
    want = gemm_ref.compute_gemm(case.W, case.X, case.R, case.C)[1]
    if not np.array_equal(Y, want):
        raise RuntimeError(f"GATE 1: {case.name}: model Y != oracle")
    return dict(report=rep, graph=g, gemm=gemm, sink=sink, Y=Y)


def tileWeights(case: Case, V: int, H: int) -> np.ndarray:
    """[V*H, R, C] int8: tile (v, h) = v * H + h holds W rows v*R .., columns h*C .. (zero past K and N)."""
    R, C = case.R, case.C
    padded = np.zeros((V * R, H * C), dtype=np.int8)
    padded[:case.K, :case.N] = case.W
    return padded.reshape(V, R, H, C).transpose(0, 2, 1, 3).reshape(V * H, R, C)


def writeTbFiles(case: Case, outDir: Path) -> dict:
    """Model run + GATE 1, then the testbench's files (tb_gemm_top.v) and the model's events:
      w.mem      the tiles' weights, one hex byte per line, tile-major (v, then h), row-major in a tile
      act.txt    per row: its burst cycle, then V x 8 hex plane words (lane-major, LSB plane first), ROWS_USED bits
      y_exp.txt  per row, per lane h: 8 hex plane words of COLS_USED bits
    """
    r = runCase(case)
    rep, g, gemm, sink = r["report"], r["graph"], r["gemm"], r["sink"]
    p, d = gemm.p, gemm.d
    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=True)
    tiles = tileWeights(case, d.V, d.H).astype(np.uint8)
    (out / "w.mem").write_text("".join(f"{b:02x}\n" for b in tiles.reshape(-1)))
    sent = {}                                                      # row -> its burst cycle (all lanes in lock-step)
    for e in rep.events:
        if e.kind == "win" and e.instance == "x":
            sent.setdefault(e.unit, set()).add(e.time)
    assert sorted(sent) == list(range(case.M)) and all(len(t) == 1 for t in sent.values()), sent
    padded = np.zeros((case.M, d.V * p.R), dtype=np.int8)
    padded[:, :case.K] = case.X
    with open(out / "act.txt", "w") as f:
        for u in range(case.M):
            words = [w for v in range(d.V) for w in planeWords(padded[u, v * p.R:v * p.R + p.ROWS_USED], 8,
                                                                p.ROWS_USED, P)]
            f.write(f"{min(sent[u])} " + " ".join(f"{w:x}" for w in words) + "\n")
    rows = {}
    for w in sink.received:
        rows.setdefault(w.unit, {})[w.window] = w.values
    with open(out / "y_exp.txt", "w") as f:
        for u in range(case.M):
            for h in range(d.H):
                f.write(" ".join(f"{x:x}" for x in planeWords(rows[u][h], 8, p.COLS_USED, P)) + "\n")
    (out / "events.txt").write_text("".join(line + "\n" for line in rep.eventLines()))
    defines = dict(K_TB=case.K, N_TB=case.N, R_TB=p.R, C_TB=p.C, RU_TB=p.ROWS_USED, CU_TB=p.COLS_USED)
    plusargs = dict(M=case.M, TMAX=rep.cycles + 2000)
    cj = dict(name=case.name, defines=defines, plusargs=plusargs, port_words=portWords(g), cycles=rep.cycles,
              events=len(rep.events), V=d.V, H=d.H, M=case.M, pass_extent=case.passExtent, meta=case.meta)
    (out / "case.json").write_text(json.dumps(cj, indent=1) + "\n")
    return cj


def operands(K: int, N: int, R: int, M: int, rng, cls: str = "random") -> tuple:
    """(W [K, N], X [M, K]) int8 of a class: random; zeros; extremes (-128, -1, 0, 1, 127); carry (every tile's row 0
    is the only active row, its weights -1 / -128 / 127 / 1, so the tile bytes are those and the tree's carries run
    through all 8 planes)."""
    if cls == "zeros":
        return np.zeros((K, N), dtype=np.int8), np.zeros((M, K), dtype=np.int8)
    if cls == "extremes":
        pick = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
        return rng.choice(pick, size=(K, N)), rng.choice(pick, size=(M, K))
    W = rng.integers(-128, 128, (K, N), dtype=np.int8)
    X = rng.integers(-128, 128, (M, K), dtype=np.int8)
    if cls == "carry":
        X = np.zeros((M, K), dtype=np.int8)
        X[:, 0:K:R] = 1
        W[0:K:R] = rng.choice(np.array([-1, -128, 127, 1], dtype=np.int8), size=W[0:K:R].shape)
    elif cls != "random":
        raise ValueError(f"unknown class {cls!r}")
    return W, X


def randomCase(name: str, K: int, N: int, R: int, C: int, M: int, rng, cls: str = "random", **kw) -> Case:
    W, X = operands(K, N, R, M, rng, cls)
    ru, cu = extent(K, N, R, C, kw.get("passExtent", "used"))
    return Case(name=name, K=K, N=N, R=R, C=C, W=W, X=X,
                meta=dict(cls=cls, V=-(-K // R), H=-(-N // C), rows_used=ru, cols_used=cu), **kw)
