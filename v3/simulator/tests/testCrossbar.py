#!/usr/bin/env python3
"""testCrossbar.py — gates of the crossbar model `Crossbar` / `CrossbarGroup` (docs/simulator.md §4.2).

  corpus   the v2 crossbar corpora (default 96 cases, P29 162; GATE-2-certified on the RTL in v2): the values equal
           the expected files (int32 y, every output byte); the cycles and every pass event equal the v2 kernel
           `nldpe_sim.run_workload` on the same stimulus with its drain over every column (cols_used = C: the model's
           rule since 2026-10-18, where the RTL still stops after COLS_USED); where the case drains every column
           anyway, the cycles also equal case.json (the RTL-pinned witness, delta 0)
  closed   back-to-back passes reproduce `nldpe_sim.cycle_model` at cols = C (total, T_fill, T_steady) over R, C, the
           load extent, M
  gated    passes with readiness gaps equal `pass_engine.schedule_pass_sequence` at cols = C
  group    CrossbarGroup(n) == n single crossbars in values; one timing state
  forms    the four output forms at their edges
  domain   the RTL guard (extent), int8 inputs, the platform's modes (Azure-Lily: identity only)
  timing   values off: identical events and counts (I4)
  mutants  a latency + 1, a wrong extent: the corpus check must fail
Run: python3 v3/simulator/tests/testCrossbar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "oracle" / "v2_kernels"))    # test references only
import nldpe_sim  # noqa: E402
import pass_engine  # noqa: E402

import simulator.core.crossbar as xb  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402
from simulator.tests import corpus  # noqa: E402

MODE_NAME = {0: "REGULAR", 1: "ACTIVATION", 2: "EXP", 3: "LOG"}
EVENTS = ("load_start", "load_end", "compute_start", "msb_fire", "shift_acc_done", "acam_done", "drain_start",
          "drain_end")


def loadCase(d: Path, cj: dict) -> tuple:
    R, C, M, ru = cj["R"], cj["C"], cj["M"], cj["rows_used"]
    W = corpus.readWords(d / "weights.mem").astype(np.uint8).view(np.int8).reshape(R, C)
    wordsPerPass = -(-ru * 8 // cj["BUF"])
    words = corpus.readWords(d / "act.mem").reshape(M, wordsPerPass)
    X = np.zeros((M, R), dtype=np.int8)
    for m in range(M):
        X[m, :ru] = corpus.wordsToBytes(words[m], ru)
    return W, X


def checkCase(d: Path, cj: dict, rowsUsed=None) -> list:
    """Problems of one corpus case (empty = pass)."""
    W, X = loadCase(d, cj)
    mode = MODE_NAME[cj["mode"]]
    ru, cu = cj["rows_used"], cj["cols_used"]
    crossbar = xb.Crossbar(cj["R"], cj["C"], cj["BUF"], cj["P"], mode=mode, rowsUsed=ru if rowsUsed is None else rowsUsed,
                           colsUsed=cu, weights=W)
    passes = [crossbar.run(X[m], 0) for m in range(cj["M"])]
    problems = []
    yExpected = np.load(d / "expected_y.npz")["y"]
    outExpected = corpus.readOutLines(d / "expected_out.mem")
    if not np.array_equal(np.stack([p.y for p in passes]), yExpected):
        problems.append("y")
    if not np.array_equal(np.stack([p.out for p in passes]), outExpected):
        problems.append("output bytes")
    if not all(np.array_equal(p.window.view(np.uint8), p.out[:cu]) for p in passes):
        problems.append("drained window")
    # the timing witness: the kernel with the drain over every column (cols_used = C; the model's rule since
    # 2026-10-18), the load at the case's rows; where the case drains every column anyway, case.json's cycles too
    ref = nldpe_sim.NldpeDpe(cj["R"], cj["C"], cj["BUF"], cj["P"], rows_used=ru, cols_used=cj["C"])
    ref.program_weights(W)
    timeline = ref.run_workload(X, cj["mode"]).timeline
    if passes[-1].events["drain_end"] + 1 != timeline[-1].drain_end + 1:
        problems.append(f"cycles {passes[-1].events['drain_end'] + 1} != {timeline[-1].drain_end + 1}")
    if cu == cj["C"] and passes[-1].events["drain_end"] + 1 != cj["used_cycles"]:
        problems.append(f"cycles {passes[-1].events['drain_end'] + 1} != case.json {cj['used_cycles']}")
    for m, (p, t) in enumerate(zip(passes, timeline)):
        bad = [e for e in EVENTS if p.events[e] != getattr(t, e)]
        if bad:
            problems.append(f"pass {m} events {bad}")
            break
    return problems


def testCorpus(names=("dpe", "dpe_p29")) -> int:
    n = 0
    for name in names:
        cs = corpus.cases(corpus.ensure(name))
        bad = [(d.name, pr) for d, cj in cs for pr in [checkCase(d, cj)] if pr]
        assert not bad, f"{name}: {len(bad)} cases fail, first {bad[:3]}"
        n += len(cs)
    return n


def testClosed() -> int:
    n = 0
    for R, C in [(8, 8), (40, 40), (64, 64), (128, 64), (256, 256), (256, 512), (1024, 64), (64, 1024), (41, 7)]:
        for rows, cols in [(None, None), (1, 1), (min(R, C), min(R, C)), (R, 1), (1, C), (max(1, R // 3), C)]:
            for M in (1, 2, 3, 7):
                crossbar = xb.Crossbar(R, C, rowsUsed=rows, colsUsed=cols, withValues=False)
                passes = [crossbar.run(None, 0) for _ in range(M)]
                cm = nldpe_sim.cycle_model(M, R, C, rows=rows, cols=C)      # the drain over every column
                assert passes[-1].events["drain_end"] + 1 == cm.total, (R, C, rows, cols, M)
                assert passes[0].events["drain_end"] + 1 == cm.t_fill
                if M > 1:
                    assert passes[1].events["drain_end"] - passes[0].events["drain_end"] == cm.t_steady
                assert passes[0].events["acam_done"] - passes[0].events["compute_start"] + 1 == cm.compute_cyc   # P10
                n += 1
    return n


def testGated(rng) -> int:
    n = 0
    for R, C in [(64, 64), (128, 64), (256, 256), (1024, 64)]:
        for _ in range(25):
            ready = sorted(int(t) for t in rng.integers(0, 600, size=int(rng.integers(1, 9))))
            for extent in ((None, None), (min(R, C), min(R, C))):
                crossbar = xb.Crossbar(R, C, rowsUsed=extent[0], colsUsed=extent[1], withValues=False)
                got = [crossbar.run(None, t).events for t in ready]
                want = pass_engine.schedule_pass_sequence(ready, R, C, 40, 8, rows=extent[0], cols=C)
                for g, w in zip(got, want):
                    assert all(g[k] == w[k] for k in w), (R, C, ready, g, w)
                n += 1
    return n


def testGroup(rng) -> None:
    n, R, C = 5, 64, 32
    W = rng.integers(-128, 128, size=(n, R, C), dtype=np.int8)
    X = rng.integers(-128, 128, size=(3, n, R), dtype=np.int8)
    group = xb.CrossbarGroup(n, R, C, mode="EXP", rowsUsed=40, colsUsed=20, weights=W)
    singles = [xb.Crossbar(R, C, mode="EXP", rowsUsed=40, colsUsed=20, weights=W[i]) for i in range(n)]
    for m in range(3):
        pg = group.run(X[m], 5 * m)
        for i, s in enumerate(singles):
            p1 = s.run(X[m, i], 5 * m)
            assert np.array_equal(pg.y[i], p1.y) and np.array_equal(pg.window[i], p1.window)
            assert pg.events == p1.events
    assert group.passes[("EXP", 40, 20)] == 3 * n
    identity = xb.CrossbarGroup(2, 16, 16)                               # identity weights
    x = rng.integers(-128, 128, size=(2, 16), dtype=np.int8)
    assert np.array_equal(identity.run(x, 0).window, x)


def testForms() -> None:
    y = np.array([0, 1, -1, 127, 128, -128, -129, 255, 2 ** 31 - 1, -(2 ** 31), 3, -3, 2 ** 20], dtype=np.int64)
    regular = xb.outputStage(y, "REGULAR")
    assert list(regular[:8]) == [0, 1, 255, 127, 128, 128, 127, 255]
    assert list(xb.outputStage(np.array([-5, 0, 5]), "ACTIVATION")) == [0, 0, 5]
    assert list(xb.outputStage(np.array([0, 3, -3]), "EXP")) == [1, 8, 2]              # 1 + y + floor(y^2/2)
    big = xb.outputStage(np.array([2 ** 20]), "EXP")                                   # 2^39 + ...: clipped to int32
    assert int(big[0]) == (2 ** 31 - 1) & 0xFF
    assert list(xb.outputStage(np.array([0, -(2 ** 31)]), "LOG")) == [255, 0]          # y - 1, clipped
    for bad in ("relu", "SOFTMAX"):
        try:
            xb.outputStage(y, bad)
        except ValueError:
            continue
        raise AssertionError(f"mode {bad} must be rejected")


def testDomain() -> None:
    def rejects(fn, what):
        try:
            fn()
        except (ValueError, OverflowError):
            return
        raise AssertionError(f"{what} must be rejected")
    rejects(lambda: xb.Crossbar(8, 8, rowsUsed=0), "rows 0")
    rejects(lambda: xb.Crossbar(8, 8, rowsUsed=9), "rows > R")
    rejects(lambda: xb.Crossbar(8, 8, colsUsed=0), "cols 0")
    rejects(lambda: xb.Crossbar(8, 8, colsUsed=9), "cols > C")
    rejects(lambda: xb.Crossbar(0, 8), "R 0")
    rejects(lambda: xb.Crossbar(8, 8, mode="SIGMOID"), "unknown mode")
    rejects(lambda: xb.Crossbar(8, 8, weights=np.zeros((8, 7), dtype=np.int8)), "weights shape")
    rejects(lambda: xb.Crossbar(8, 8, weights=np.full((8, 8), 200)), "weights outside int8")
    crossbar = xb.Crossbar(8, 8)
    rejects(lambda: crossbar.run(np.full(8, 128), 0), "x outside int8")
    rejects(lambda: crossbar.run(np.zeros(7, dtype=np.int8), 0), "x shape")
    al = Platform("azure_lily")
    assert xb.CrossbarGroup.fromPlatform(al, 1, mode="REGULAR").BUF == 16
    rejects(lambda: xb.CrossbarGroup.fromPlatform(al, 1, mode="EXP"), "EXP on an ADC output stage")
    nl = Platform("nl_dpe", rows=128, cols=64)
    g = xb.CrossbarGroup.fromPlatform(nl, 2, mode="LOG", withValues=False)
    assert (g.R, g.C, g.BUF, g.P) == (128, 64, 40, 8)
    # the Azure-Lily pass constants follow from the same grammar (simulator.md §4.2): 512 x 128, BUF 16
    a = xb.CrossbarGroup.fromPlatform(al, 1, withValues=False)
    assert (a.geometry.LOAD_CYC, a.geometry.COMPUTE_CYC, a.geometry.OUTPUT_CYC) == (256, 10, 64)


def testTimingOnly(rng) -> None:
    W = rng.integers(-128, 128, size=(64, 32), dtype=np.int8)
    a = xb.Crossbar(64, 32, mode="LOG", rowsUsed=50, weights=W)
    b = xb.Crossbar(64, 32, mode="LOG", rowsUsed=50, withValues=False)
    for t in (0, 0, 7, 300):
        pa, pb = a.run(rng.integers(-128, 128, 64, dtype=np.int8), t), b.run(None, t)
        assert pa.events == pb.events and pa.time == pb.time and pa.nextReadyAt == pb.nextReadyAt and pb.window is None
    assert a.passes == b.passes


def testMutants() -> int:
    """A latency + 1 and a wrong extent must each fail the corpus check."""
    d = corpus.ensure("dpe_p29")
    cs = corpus.cases(d)
    p29 = next((dd, cj) for dd, cj in cs if cj["rows_used"] < cj["R"])
    selected = [c for c in cs if c[1]["M"] >= 2][:6] + [p29]              # RDY_LAT acts from the second pass on
    caught = 0
    for name in ("RDY_LAT", "SHIFT_ACC_LAT", "ACAM_LAT", "DRAIN_LAT"):
        old = getattr(xb, name)
        setattr(xb, name, old + 1)
        try:
            caught += any(checkCase(dd, cj) for dd, cj in selected)
        finally:
            setattr(xb, name, old)
    caught += bool(checkCase(p29[0], p29[1], rowsUsed=p29[1]["R"]))        # full load where the extent is smaller
    assert caught == 5, f"{caught}/5 mutants caught"
    return caught


def main() -> int:
    rng = np.random.default_rng(20261014)
    print(f"  corpus   PASS  {testCorpus()} cases (values, cycles, every pass event == nldpe_sim)")
    print(f"  closed   PASS  {testClosed()} sweeps == nldpe_sim.cycle_model")
    print(f"  gated    PASS  {testGated(rng)} ready sequences == pass_engine.schedule_pass_sequence")
    testGroup(rng)
    print("  group    PASS")
    testForms()
    print("  forms    PASS")
    testDomain()
    print("  domain   PASS")
    testTimingOnly(rng)
    print("  timing   PASS")
    print(f"  mutants  PASS  {testMutants()}/5 caught")
    print("crossbar model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
