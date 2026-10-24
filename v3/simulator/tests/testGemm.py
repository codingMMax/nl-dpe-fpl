#!/usr/bin/env python3
"""testGemm.py — gates of the `gemm_top` model (ip/gemm/spec.md; docs/simulator.md §4.4).

  corpus    the v2 GEMM corpora (default 16 cases, P29 144; GATE-2-certified on the v2 RTL), run through the FC workload
            (source -> gemm_top -> sink): the output bytes and the wide probe S equal the expected files; every row's
            events and the run's cycles equal the spec's closed form (below); the pass counts carry the extent
  closed    the spec's closed form, written here by hand (ip/gemm/spec.md §5): row m's burst at
            max(offer_m, burst_{m-1} + P + 2); its crossbar events at the charter's offsets; its output window at
            drain_start + L_W, L_W = clog2(V) + 1 (V > 1) or 1; M rows back to back take (P + 2) M + P + L_W cycles
  gated     rows offered late: each burst at max(offer, MSB_SA_Ready), the same closed form
  structure the model's parameters and port signals are the RTL's (ip/gemm/rtl/sim/gemm_top.v), derive() == the
            spec's localparams, checkDomain() == the RTL extent guard
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

import simulator.models.gemm as gemm  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import Deadlock  # noqa: E402
from simulator.models.gemm import GemmParams, GemmTop  # noqa: E402
from simulator.tests import corpus  # noqa: E402
from simulator.models.fc import fc, result  # noqa: E402
from tools.check_twins import header, modules  # noqa: E402

P = 8


def rowEvents(burst: int, V: int, P: int = P) -> dict:
    """ip/gemm/spec.md §5, from the charter's offsets (ip/xbar/spec.md §4): written here, not read from the model."""
    lw = ((V - 1).bit_length() if V > 1 else 0) + 1               # clog2(V) registered levels + the output register
    out = burst + P + 2 + lw
    return dict(load_start=burst, load_end=burst + P - 1, compute_start=burst + 1, msb_fire=burst + P,
                acam_done=burst + P + 1, drain_start=burst + P + 2, drain_end=burst + 2 * P + 1,
                array_drain_start=out, array_drain_end=out + P - 1, done=out + P)


def bursts(offers: list, P: int = P) -> list:
    out = []
    for t in offers:
        out.append(max(t, 0) if not out else max(t, out[-1] + P + 2))
    return out


def loadCase(d: Path, cj: dict) -> tuple:
    M, K, N, R, C, V, H, ru = cj["M"], cj["K"], cj["N"], cj["R"], cj["C"], cj["V"], cj["H"], cj["rows_used"]
    w = corpus.readWords(d / "weights.mem").astype(np.uint8).view(np.int8).reshape(V, H, R, C)
    Wp = w.transpose(0, 2, 1, 3).reshape(V * R, H * C)
    loadCycles = -(-ru * 8 // cj["BUF"])                          # the v2 stimuli are pre-bit-plane words: only data here
    words = corpus.readWords(d / "act.mem").reshape(M, V, loadCycles)
    Xp = np.zeros((M, V * R), dtype=np.int8)
    for m in range(M):
        for v in range(V):
            Xp[m, v * R:v * R + ru] = corpus.wordsToBytes(words[m, v], ru)
    return Wp[:K, :N].copy(), Xp[:, :K].copy()


def extentOf(K: int, N: int, R: int, C: int, passExtent: str) -> tuple:
    """(ROWS_USED, COLS_USED) of gemm_top at an extent, as fc.extent derives them."""
    V, H = -(-K // R), -(-N // C)
    return ((K if V == 1 else R), (N if H == 1 else C)) if passExtent == "used" else (R, C)


def checkCase(d: Path, cj: dict, extentOverride=None) -> list:
    W, X = loadCase(d, cj)
    M, K, N, R, C, H, V = cj["M"], cj["K"], cj["N"], cj["R"], cj["C"], cj["H"], cj["V"]
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
    ru, cu = extentOf(K, N, R, C, cj["pass_extent"])
    if dict(rep.counts["gemm"].passes) != {("REGULAR", ru, cu): M * V * H}:
        problems.append(f"passes {dict(rep.counts['gemm'].passes)}")
    want = [rowEvents(b, V) for b in bursts([0] * M)]
    for m, (s, w) in enumerate(zip(rep.steps["gemm"], want)):
        if s.events != w:
            problems.append(f"row {m} events {s.events} != {w}")
            break
    if rep.cycles != want[-1]["done"]:
        problems.append(f"cycles {rep.cycles} != {want[-1]['done']}")
    return problems


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
                        V = -(-K // R)
                        want = [rowEvents(b, V) for b in bursts([0] * M)]
                        assert [s.events for s in rep.steps["gemm"]] == want, (K, N, R, C, ext, M)
                        lw = ((V - 1).bit_length() if V > 1 else 0) + 1
                        assert rep.cycles == (P + 2) * M + P + lw, (K, N, R, C, ext, M, rep.cycles)
                        n += 1
    return n


def testGated(rng) -> int:
    n = 0
    for (K, N, R, C) in [(64, 64, 64, 64), (300, 40, 128, 64), (512, 512, 256, 256)]:
        for _ in range(20):
            M = int(rng.integers(1, 7))
            offers = sorted(int(t) for t in rng.integers(0, 500, size=M))
            rep = run(fc(K, N, R, C, M=M, passExtent="used", withValues=False, availableAt=offers))
            V = -(-K // R)
            assert [s.events for s in rep.steps["gemm"]] == [rowEvents(b, V) for b in bursts(offers)], (K, N, offers)
            n += 1
    return n


def testStructure() -> None:
    text = (V3 / "ip" / "gemm" / "rtl" / "sim" / "gemm_top.v").read_text()
    params, ports = header(modules(text)["gemm_top"])
    rtl = [p for p, _ in params]
    assert rtl == GemmTop.paramNames(), (rtl, GemmTop.paramNames())
    names = {p[3] for p in ports}
    model = GemmTop("g", GemmParams(K=600, N=300, R=256, C=128), rows=0, withValues=False)
    signals = {s for spec in model.ports().values() for s in spec.signals}
    # not in the port contract: clock, reset, the weight strobes, and reg_full (a busy flag; out_valid is the word valid)
    assert signals <= names and names - signals == {"clk", "reset", "weight_in", "weight_en", "reg_full"}, (signals, names)
    # derive() == the spec's localparams at a set of builds (V, H, TREE_PIPE, NTILES, L_W)
    for K, R in [(1, 64), (64, 64), (65, 64), (300, 64), (1024, 128), (2048, 128), (513, 256)]:
        d = GemmTop.derive(GemmParams(K=K, N=70, R=R, C=64))
        V = (K + R - 1) // R
        tree = 0 if V <= 1 else max(1, int(np.ceil(np.log2(V))))
        assert (d.V, d.H, d.TREE_PIPE, d.NTILES, d.L_W) == (V, 2, tree, 2 * V, tree + 1), (K, R, d)
    # checkDomain() == the RTL extent guard (gemm_top.v g_bad_extent)
    ok = [dict(K=64, N=64, R=64, C=64), dict(K=17, N=9, R=64, C=64, ROWS_USED=17, COLS_USED=9),
          dict(K=17, N=9, R=64, C=64, ROWS_USED=40, COLS_USED=64), dict(K=300, N=9, R=64, C=64, COLS_USED=9)]
    bad = [dict(K=17, N=9, R=64, C=64, ROWS_USED=16), dict(K=300, N=9, R=64, C=64, ROWS_USED=60),
           dict(K=17, N=90, R=64, C=64, COLS_USED=40), dict(K=17, N=9, R=64, C=64, COLS_USED=8),
           dict(K=17, N=9, R=64, C=64, ROWS_USED=65), dict(K=17, N=9, R=64, C=64, COLS_USED=0),
           dict(K=0, N=9, R=64, C=64), dict(K=17, N=9, R=64, C=64, P=0)]
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
    caught += any(checkCase(dd, cj, extentOverride="full") for dd, cj in selected)   # a wrong extent (the counts)
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
    print(f"  corpus    PASS  {testCorpus()} cases (bytes, S probe, passes; every row event == the closed form)")
    print(f"  closed    PASS  {testClosed()} points == the closed form, (P + 2) M + P + L_W cycles")
    print(f"  gated     PASS  {testGated(rng)} late-offer sequences == the closed form")
    testStructure()
    print("  structure PASS  (parameters, port signals, derive, checkDomain == the RTL / the spec)")
    testTimingAndOrder(rng)
    print("  timing    PASS  (values off: same events, steps, counts; 5 random step orders: same Report)")
    print(f"  mutants   PASS  {testMutants()}/3 caught")
    print("gemm model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
