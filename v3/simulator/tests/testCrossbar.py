#!/usr/bin/env python3
"""testCrossbar.py — gates of the crossbar model `Crossbar` / `CrossbarGroup` (ip/xbar/spec.md; docs/simulator.md
§4.2).

  corpus   the v2 crossbar corpora (default 96 cases, P29 162; GATE-2-certified on the v2 RTL): the values equal the
           expected files (int32 y, every output byte, the drained window). The timing is the charter's closed form
           (below): v2's kernels time the pre-bit-plane pass and stay value witnesses only (since 2026-10-22)
  closed   the charter's closed form, written here by hand (ip/xbar/spec.md §5): back to back from cycle 0,
           load_start_m = (P + 2) m, every event at its offset, T(M) = (P + 2) M + P, over R, C, the extent, M
  gated    passes with readiness gaps: load_start_m = max(t_m, load_start_{m-1} + P + 2)
  bound    the output register never binds (acam_done = msb_fire + 1 on every pass), and the extent never changes
           the time
  group    CrossbarGroup(n) == n single crossbars in values; one timing state
  forms    the four output forms at their edges
  domain   the RTL guard (extent), int8 inputs, the platform's modes (Azure-Lily: identity only)
  timing   values off: identical events and counts (I4)
  mutants  each named latency + 1, and a wrong extent: the corpus check must fail
Run: python3 v3/simulator/tests/testCrossbar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))

import simulator.core.crossbar as xb  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402
from simulator.tests import corpus  # noqa: E402

MODE_NAME = {0: "REGULAR", 1: "ACTIVATION", 2: "EXP", 3: "LOG"}


def closedForm(loadStart: int, P: int = 8) -> dict:
    """ip/xbar/spec.md §4-§5: every event at its offset from load_start."""
    return dict(load_start=loadStart, load_end=loadStart + P - 1, compute_start=loadStart + 1,
                msb_fire=loadStart + P, acam_done=loadStart + P + 1, drain_start=loadStart + P + 2,
                drain_end=loadStart + 2 * P + 1)


def schedule(ready: list, P: int = 8) -> list:
    """load_start of passes ready at `ready`: the first at max(t, 0), then max(t_m, load_start_{m-1} + P + 2)."""
    out = []
    for t in ready:
        out.append(max(t, 0) if not out else max(t, out[-1] + P + 2))
    return out


def loadCase(d: Path, cj: dict) -> tuple:
    """The v2 corpus case's weights and inputs (its stimuli are pre-bit-plane words: they are only data here). The rows past
    the extent carry nonzero bytes: the model must read them as zero (P29), which the expected values assume."""
    R, C, M, ru = cj["R"], cj["C"], cj["M"], cj["rows_used"]
    W = corpus.readWords(d / "weights.mem").astype(np.uint8).view(np.int8).reshape(R, C)
    wordsPerPass = -(-ru * 8 // cj["BUF"])
    words = corpus.readWords(d / "act.mem").reshape(M, wordsPerPass)
    X = np.random.default_rng(R * 1000 + ru).integers(1, 128, size=(M, R)).astype(np.int8)
    for m in range(M):
        X[m, :ru] = corpus.wordsToBytes(words[m], ru)
    return W, X


def checkCase(d: Path, cj: dict, rowsUsed=None) -> list:
    """Problems of one corpus case (empty = pass)."""
    W, X = loadCase(d, cj)
    mode = MODE_NAME[cj["mode"]]
    ru, cu, P = cj["rows_used"], cj["cols_used"], cj["P"]
    crossbar = xb.Crossbar(cj["R"], cj["C"], P, mode=mode, rowsUsed=ru if rowsUsed is None else rowsUsed,
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
    for m, (p, ls) in enumerate(zip(passes, schedule([0] * cj["M"], P))):
        want = closedForm(ls, P)
        if p.events != want:
            problems.append(f"pass {m} events {p.events} != {want}")
            break
    if passes[-1].events["drain_end"] + 1 != (P + 2) * cj["M"] + P:
        problems.append(f"cycles {passes[-1].events['drain_end'] + 1} != {(P + 2) * cj['M'] + P}")
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
                for P in (8, 4, 1):
                    crossbar = xb.Crossbar(R, C, P, rowsUsed=rows, colsUsed=cols, withValues=False)
                    passes = [crossbar.run(None, 0) for _ in range(M)]
                    for m, p in enumerate(passes):
                        assert p.events == closedForm((P + 2) * m, P), (R, C, rows, cols, M, P, m, p.events)
                        assert p.time == p.events["drain_start"] and p.nextReadyAt == p.events["acam_done"] + 1
                    assert passes[-1].events["drain_end"] + 1 == (P + 2) * M + P               # T(M)
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
                for g, ls in zip(got, schedule(ready)):
                    assert g == closedForm(ls), (R, C, ready, g, ls)
                n += 1
    return n


def testBound(rng) -> int:
    """The output register never binds (acam_done = msb_fire + 1), and the extent never changes the time."""
    n = 0
    for _ in range(200):
        R, C = int(rng.choice([8, 64, 128, 512])), int(rng.choice([8, 64, 256]))
        ready = sorted(int(t) for t in rng.integers(0, 300, size=int(rng.integers(1, 12))))
        events = {}
        for rows, cols in ((None, None), (1, 1), (max(1, R // 2), max(1, C // 3))):
            crossbar = xb.Crossbar(R, C, rowsUsed=rows, colsUsed=cols, withValues=False)
            events[(rows, cols)] = [crossbar.run(None, t).events for t in ready]
            for e in events[(rows, cols)]:
                assert e["acam_done"] == e["msb_fire"] + xb.ACAM_LAT, e
            n += 1
        assert len({repr(v) for v in events.values()}) == 1, (R, C, ready)
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
    rejects(lambda: xb.Crossbar(8, 8, 0), "P 0")
    rejects(lambda: xb.Crossbar(8, 8, mode="SIGMOID"), "unknown mode")
    rejects(lambda: xb.Crossbar(8, 8, weights=np.zeros((8, 7), dtype=np.int8)), "weights shape")
    rejects(lambda: xb.Crossbar(8, 8, weights=np.full((8, 8), 200)), "weights outside int8")
    crossbar = xb.Crossbar(8, 8)
    rejects(lambda: crossbar.run(np.full(8, 128), 0), "x outside int8")
    rejects(lambda: crossbar.run(np.zeros(7, dtype=np.int8), 0), "x shape")
    al = Platform("azure_lily")
    a = xb.CrossbarGroup.fromPlatform(al, 1, mode="REGULAR", withValues=False)
    assert (a.R, a.C, a.P) == (512, 128, 8)
    rejects(lambda: xb.CrossbarGroup.fromPlatform(al, 1, mode="EXP"), "EXP on an ADC output stage")
    nl = Platform("nl_dpe", rows=128, cols=64)
    g = xb.CrossbarGroup.fromPlatform(nl, 2, mode="LOG", withValues=False)
    assert (g.R, g.C, g.P) == (128, 64, 8)
    # the pass rule does not depend on the platform's port width: P planes in, P planes out
    assert (a.geometry.LOAD_CYC, a.geometry.OUTPUT_CYC) == (g.geometry.LOAD_CYC, g.geometry.OUTPUT_CYC) == (8, 8)


def testTimingOnly(rng) -> None:
    W = rng.integers(-128, 128, size=(64, 32), dtype=np.int8)
    a = xb.Crossbar(64, 32, mode="LOG", rowsUsed=50, weights=W)
    b = xb.Crossbar(64, 32, mode="LOG", rowsUsed=50, withValues=False)
    for t in (0, 0, 7, 300):
        pa, pb = a.run(rng.integers(-128, 128, 64, dtype=np.int8), t), b.run(None, t)
        assert pa.events == pb.events and pa.time == pb.time and pa.nextReadyAt == pb.nextReadyAt and pb.window is None
    assert a.passes == b.passes


def testMutants() -> int:
    """Each named latency + 1, and a wrong extent, must fail the corpus check."""
    d = corpus.ensure("dpe_p29")
    cs = corpus.cases(d)
    p29 = next((dd, cj) for dd, cj in cs if cj["rows_used"] < cj["R"])
    selected = [c for c in cs if c[1]["M"] >= 2][:6] + [p29]              # RDY_LAT acts from the second pass on
    caught = 0
    for name in ("FIRE_LAT", "ACAM_LAT", "DRAIN_LAT", "RDY_LAT"):
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
    print(f"  corpus   PASS  {testCorpus()} cases (values == the expected files; events == the closed form)")
    print(f"  closed   PASS  {testClosed()} sweeps == the charter's closed form, T(M) = (P + 2) M + P")
    print(f"  gated    PASS  {testGated(rng)} ready sequences: load_start = max(t, previous + P + 2)")
    print(f"  bound    PASS  {testBound(rng)} runs: the output register never binds, the extent never changes time")
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
