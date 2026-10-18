#!/usr/bin/env python3
"""testGemm.py — gates of the `gemm_top` model (docs/simulator.md §4.4).

  corpus    the v2 GEMM corpora (default 16 cases, P29 144; GATE-2-certified on the RTL in v2), run through the FC
            workload (source -> gemm_top -> sink): the output bytes and the wide probe S equal the expected files;
            where the case drains every column, the run's cycles equal case.json and every row's events equal the v2
            kernel `gemm_sim` timeline; where it drains fewer (COLS_USED < C), the model drains every column since
            2026-10-18 (the RTL still stops), so the witness is `pass_engine`'s schedule at cols = C plus L_w and the
            COLS_USED output words (fullDrainTimeline)
  closed    back-to-back rows reproduce `gemm_sim.cycle_model` (both extents) over K, N, R, C, M where every column
            is drained, fullDrainTimeline elsewhere
  gated     rows offered late: each burst starts at max(offer, MSB_SA_Ready) and the events equal
            `pass_engine.schedule_pass_sequence` + L_w
  structure the model's parameters and port signals are the RTL's (ip/gemm/rtl/sim/gemm_top.v), derive() == the
            RTL localparams, checkDomain() == the RTL extent guard
  timing    values off: identical events, steps and counts (I4); random step order: identical Report
  mutants   a latency + 1, a dropped MSB_SA_Ready credit, a wrong extent: each must fail a check
Run: python3 v3/simulator/tests/testGemm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "oracle" / "v2_kernels"))    # test references only
import gemm_sim  # noqa: E402
import pass_engine  # noqa: E402

import simulator.models.gemm as gemm  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import Deadlock  # noqa: E402
from simulator.models.gemm import GemmParams, GemmTop  # noqa: E402
from simulator.tests import corpus  # noqa: E402
from simulator.models.fc import fc, result  # noqa: E402
from tools.check_twins import header, modules  # noqa: E402

EVENTS = ("load_start", "load_end", "compute_start", "msb_fire", "shift_acc_done", "acam_done", "drain_start",
          "drain_end", "array_drain_start", "array_drain_end", "done")


def loadCase(d: Path, cj: dict) -> tuple:
    M, K, N, R, C, V, H, ru = cj["M"], cj["K"], cj["N"], cj["R"], cj["C"], cj["V"], cj["H"], cj["rows_used"]
    w = corpus.readWords(d / "weights.mem").astype(np.uint8).view(np.int8).reshape(V, H, R, C)
    Wp = w.transpose(0, 2, 1, 3).reshape(V * R, H * C)
    loadCycles = -(-ru * 8 // cj["BUF"])
    words = corpus.readWords(d / "act.mem").reshape(M, V, loadCycles)
    Xp = np.zeros((M, V * R), dtype=np.int8)
    for m in range(M):
        for v in range(V):
            Xp[m, v * R:v * R + ru] = corpus.wordsToBytes(words[m, v], ru)
    return Wp[:K, :N].copy(), Xp[:, :K].copy()


def checkCase(d: Path, cj: dict, extentOverride=None) -> list:
    W, X = loadCase(d, cj)
    M, K, N, R, C, H = cj["M"], cj["K"], cj["N"], cj["R"], cj["C"], cj["H"]
    ext = cj["pass_extent"] if extentOverride is None else extentOverride
    g = fc(K, N, R, C, M=M, W=W, X=X, passExtent=ext)
    rep = run(g)
    sink, model = g.modules[2], g.modules[1]
    problems = []
    Y = result(sink, M, N, H)
    outExpected = corpus.readOutLines(d / "expected_out.mem").view(np.int8)
    if not np.array_equal(Y, outExpected[:, :N]):
        problems.append("output bytes")
    S = np.stack([s[:N] for s in model.math.probeS])                    # lane h = columns h*C ..; padding past N
    if not np.array_equal(S, np.load(d / "expected_psum.npz")["S"]):
        problems.append("S probe")
    V = cj["V"]
    ru, cu = extentOf(K, N, R, C, cj["pass_extent"])
    if cu == C:
        # the case drains every column: the kernel's timeline is the witness (GATE 2 on the RTL in v2)
        if rep.cycles != cj["used_cycles"]:
            problems.append(f"cycles {rep.cycles} != {cj['used_cycles']}")
        ref = gemm_sim.NldpeGemm(M, K, N, R, C, cj["BUF"], cj["P"], pass_extent=cj["pass_extent"])
        ref.program_weights(W)
        timeline = ref.run(X).timeline
        for m, (s, t) in enumerate(zip(rep.steps["gemm"], timeline)):
            bad = [e for e in EVENTS if s.events[e] != getattr(t, e)]
            if bad:
                problems.append(f"row {m} events {bad}")
                break
        return problems
    # the case drains fewer columns than C: since 2026-10-18 the model drains every column (the RTL still stops after
    # COLS_USED), so the witness is the crossbar pass schedule at cols = C plus the tree and serializer latencies, and
    # the output window keeps its COLS_USED bytes
    want = fullDrainTimeline(M, R, C, cj["BUF"], cj["P"], ru, cu, V)
    for m, (s, w) in enumerate(zip(rep.steps["gemm"], want)):
        bad = [e for e in w if s.events[e] != w[e]]
        if bad:
            problems.append(f"row {m} events {bad}")
            break
    if rep.cycles != want[-1]["done"]:
        problems.append(f"cycles {rep.cycles} != {want[-1]['done']}")
    return problems


def extentOf(K: int, N: int, R: int, C: int, passExtent: str) -> tuple:
    """(ROWS_USED, COLS_USED) of gemm_top at an extent, as fc.extent derives them."""
    V, H = -(-K // R), -(-N // C)
    return ((K if V == 1 else R), (N if H == 1 else C)) if passExtent == "used" else (R, C)


def fullDrainTimeline(M: int, R: int, C: int, BUF: int, P: int, ru: int, cu: int, V: int, offers=None) -> list:
    """Row events of gemm_top with the drain over every column: pass_engine's schedule (load at ru rows, drain at C
    columns), the first output word L_w later, COLS_USED bytes through the serializer, done after the last word.
    L_w = TREE_PIPE + 1 is the charter's (gemm.md §5.2), written here and not read from the model, so a latency mutant
    in the model is caught."""
    lw = (V - 1).bit_length() + 1                     # clog2(V) registered tree levels + the serializer register
    nWords = -(-cu // (BUF // 8))
    out = []
    for w in pass_engine.schedule_pass_sequence(offers if offers is not None else [0] * M, R, C, BUF, P, rows=ru,
                                                cols=C):
        t0 = w["drain_start"] + lw
        out.append(dict(w, array_drain_start=t0, array_drain_end=t0 + nWords - 1, done=t0 + nWords))
    return out


def testCorpus(names=("gemm", "gemm_p29")) -> int:
    n = 0
    for name in names:
        cs = corpus.cases(corpus.ensure(name))
        bad = [(d.name, pr) for d, cj in cs for pr in [checkCase(d, cj)] if pr]
        assert not bad, f"{name}: {len(bad)} cases fail, first {bad[:3]}"
        n += len(cs)
    return n


def testClosed() -> int:
    n = 0
    for R, C in [(64, 64), (128, 64), (256, 256), (512, 128), (1024, 256), (40, 8)]:
        for K in (1, 17, R, R + 1, 3 * R - 5):
            for N in (1, 9, C, C + 1, 2 * C + 3):
                for ext in ("full", "used"):
                    for M in (1, 2, 5):
                        rep = run(fc(K, N, R, C, M=M, passExtent=ext, withValues=False))
                        ru, cu = extentOf(K, N, R, C, ext)
                        if cu == C:                                  # every column drained: the kernel's closed form
                            cm = gemm_sim.cycle_model(M, K, N, R, C, pass_extent=ext)
                            total, steady = cm.total, cm.t_steady
                        else:                                        # fewer columns in use: the schedule at cols = C
                            want = fullDrainTimeline(M, R, C, 40, 8, ru, cu, -(-K // R))
                            total = want[-1]["done"]
                            steady = want[1]["done"] - want[0]["done"] if M > 1 else None
                        assert rep.cycles == total, (K, N, R, C, ext, M, rep.cycles, total)
                        if M > 1:
                            steps = rep.steps["gemm"]
                            assert steps[1].tDone - steps[0].tDone == steady
                        n += 1
    return n


def testGated(rng) -> int:
    n = 0
    for (K, N, R, C) in [(64, 64, 64, 64), (300, 40, 128, 64), (512, 512, 256, 256)]:
        for _ in range(20):
            M = int(rng.integers(1, 7))
            offers = sorted(int(t) for t in rng.integers(0, 500, size=M))
            rep = run(fc(K, N, R, C, M=M, passExtent="used", withValues=False, availableAt=offers))
            V, H = -(-K // R), -(-N // C)
            ru, cu = (K if V == 1 else R), (N if H == 1 else C)
            want = pass_engine.schedule_pass_sequence(offers, R, C, 40, 8, rows=ru, cols=C)   # every column drained
            lw = (gemm.clog2(V) if V > 1 else 0) + gemm.SER_LAT
            for s, w in zip(rep.steps["gemm"], want):
                assert s.events["load_start"] == w["load_start"] and s.events["drain_end"] == w["drain_end"]
                assert s.events["array_drain_start"] == w["drain_start"] + lw
            n += 1
    return n


def testStructure() -> None:
    text = (V3 / "ip" / "gemm" / "rtl" / "sim" / "gemm_top.v").read_text()
    params, ports = header(modules(text)["gemm_top"])
    assert [p for p, _ in params] == GemmTop.paramNames(), (params, GemmTop.paramNames())
    names = {p[3] for p in ports}
    model = GemmTop("g", GemmParams(K=600, N=300, R=256, C=128), rows=0, withValues=False)
    signals = {s for spec in model.ports().values() for s in spec.signals}
    # not in the port contract: clock, reset, the weight strobes, and reg_full (a busy flag one cycle behind the words;
    # out_valid is the word valid)
    assert signals <= names and names - signals == {"clk", "reset", "weight_in", "weight_en", "reg_full"}, (signals, names)
    # derive() == the RTL localparams at a set of builds (V, H, EPS, TREE_PIPE, NTILES)
    for K, R in [(1, 64), (64, 64), (65, 64), (300, 64), (1024, 128), (2048, 128), (513, 256)]:
        d = GemmTop.derive(GemmParams(K=K, N=70, R=R, C=64))
        V = (K + R - 1) // R
        tree = 0 if V <= 1 else max(1, int(np.ceil(np.log2(V))))
        assert (d.V, d.H, d.EPS, d.TREE_PIPE, d.NTILES) == (V, 2, 5, tree, 2 * V), (K, R, d)
    # checkDomain() == the RTL extent guard (gemm_top.v g_bad_extent)
    ok = [dict(K=64, N=64, R=64, C=64), dict(K=17, N=9, R=64, C=64, ROWS_USED=17, COLS_USED=9),
          dict(K=17, N=9, R=64, C=64, ROWS_USED=40, COLS_USED=64), dict(K=300, N=9, R=64, C=64, COLS_USED=9)]
    bad = [dict(K=17, N=9, R=64, C=64, ROWS_USED=16), dict(K=300, N=9, R=64, C=64, ROWS_USED=60),
           dict(K=17, N=90, R=64, C=64, COLS_USED=40), dict(K=17, N=9, R=64, C=64, COLS_USED=8),
           dict(K=17, N=9, R=64, C=64, ROWS_USED=65), dict(K=17, N=9, R=64, C=64, COLS_USED=0),
           dict(K=0, N=9, R=64, C=64)]
    for p in ok:
        GemmTop.checkDomain(GemmParams(**p))
    for p in bad:
        try:
            GemmTop.checkDomain(GemmParams(**p))
        except ValueError:
            continue
        raise AssertionError(f"checkDomain must reject {p}")


def testTimingAndOrder(rng) -> None:
    K, N, R, C, M = 300, 150, 128, 64, 4
    W = rng.integers(-128, 128, (K, N), dtype=np.int8)
    X = rng.integers(-128, 128, (M, K), dtype=np.int8)
    a = run(fc(K, N, R, C, M=M, W=W, X=X))
    b = run(fc(K, N, R, C, M=M, withValues=False))
    assert a.events == b.events and a.cycles == b.cycles
    assert [(s.tStart, s.tDone, s.events) for s in a.steps["gemm"]] == \
           [(s.tStart, s.tDone, s.events) for s in b.steps["gemm"]]
    assert a.counts["gemm"] == b.counts["gemm"]
    for seed in range(5):
        c = run(fc(K, N, R, C, M=M, W=W, X=X), order="random", seed=seed)
        assert c.events == a.events and c.cycles == a.cycles


def testMutants() -> int:
    d = corpus.ensure("gemm_p29")
    cs = corpus.cases(d)
    selected = [c for c in cs if c[1]["M"] >= 2 and c[1]["V"] == 1 and c[1]["K"] < c[1]["R"]][:4]
    caught = 0
    old = gemm.SER_LAT
    gemm.SER_LAT = old + 1
    try:
        caught += any(checkCase(dd, cj) for dd, cj in selected)                   # a latency + 1
    finally:
        gemm.SER_LAT = old
    caught += any(checkCase(dd, cj, extentOverride="full") for dd, cj in selected)   # a wrong extent
    original = GemmTop.stepRow

    def noCredit(self, row, io):                                                  # a dropped MSB_SA_Ready credit
        io.giveCredit = lambda *a, **kw: None
        return original(self, row, io)
    GemmTop.stepRow = noCredit
    try:
        run(fc(64, 64, 64, 64, M=3, withValues=False))
    except Deadlock:
        caught += 1
    finally:
        GemmTop.stepRow = original
    assert caught == 3, f"{caught}/3 mutants caught"
    return caught


def main() -> int:
    rng = np.random.default_rng(20261014)
    print(f"  corpus    PASS  {testCorpus()} cases (bytes, S probe, cycles, every row event == gemm_sim)")
    print(f"  closed    PASS  {testClosed()} points == gemm_sim.cycle_model (both extents)")
    print(f"  gated     PASS  {testGated(rng)} late-offer sequences == pass_engine + L_w")
    testStructure()
    print("  structure PASS  (parameters, port signals, derive, checkDomain == the RTL)")
    testTimingAndOrder(rng)
    print("  timing    PASS  (values off: same events, steps, counts; 5 random step orders: same Report)")
    print(f"  mutants   PASS  {testMutants()}/3 caught")
    print("gemm model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
