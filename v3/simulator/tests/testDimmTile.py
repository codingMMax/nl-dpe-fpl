#!/usr/bin/env python3
"""testDimmTile.py — gates of the `dimm_tile_top` model (docs/simulator.md §4.4; ip/dimm/spec.md §8), the bit-plane
pass (2026-10-22).

  gate1      GATE 1 on a sweep of QK and PV configurations (int8 / log operands, row / window frame, TRANS_B 0 / 1,
             1-4 slices, 1-4 channels, several matrices and jobs, released slots, NARROW 0 / 5), classes random /
             zeros / extremes, crossbars 64x64, 128x64, 256x128
  closed     a job with early operands: every wave's burst, every output window (time and channel), done and idle
             relative to t_go equal spec §5's closed form, written here by hand; back-to-back jobs decide at t_idle
             (until 2026-10-22 the witness was v2's `dimm_tile_sim.tile_cycle_model`, which times the pre-bit-plane pass)
  rules      the decision binds on each of its terms (job offer, A / B valid, idle, output credit) exactly as spec §5
  staging    transposer, window stager (E_WP cycles per window over the A memory ports), row writer: write times and
             credits by spec §5, overruns raise
  domain     the 324 DSE candidates (6 shapes x 9 configs x P_proj {1, 2, 4} x N_E {d, 2d}) are inside the module's
             domain for both instances; every head PV instance has E_WP = 4 (8 words over 2 memories), so the
             window-frame rate is 0.8; out-of-domain sets are rejected
  timing     values off: identical events and counts; random step orders: identical Report
  mutants    a latency + 1 (DEC_LAT, CR_LAT, IDLE_LAT, EMIT_LAT, DIST_LAT, DONE_LAT, SLICE_LAT, ROOT_LAT), an early
             release, a dropped credit, a value mutant: each fails a check
  memories   spec §6 by hand == the memories Yosys inferred from the bit-plane RTL (pinned from v3/dse/vtr_dse.py's
             elaboration of the 80 builds, 2026-10-23; I = 64, 128, 256) == the model's list
Run: python3 v3/simulator/tests/testDimmTile.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))

import simulator.models.dimmTile as dimmTile  # noqa: E402
from ip.dimm.cases import build, completeParams, convPeriod, pvParams, qkParams, randomCase, runCase  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import ContractError, Deadlock  # noqa: E402
from simulator.models.dimmTile import DimmTile, DimmTileParams  # noqa: E402

from dse.space import CONFIGS, SHAPES as NAMED_SHAPES  # noqa: E402  (the DSE's design space, defined once)
SHAPES = [shape[1:] for shape in NAMED_SHAPES]                       # (n, d_model, d)
P = 8                                                                # plane words per window (spec §2)


def inDomain(p: dict) -> bool:
    try:
        DimmTile.checkDomain(DimmTileParams(**p))
        return True
    except ValueError:
        return False


def jobsQk(runs: int, blocks: int) -> list:
    return [(r, b, 0) for r in range(runs) for b in range(blocks)]


def jobsPv(runs: int, blocks: int) -> list:
    return [(j % 2, j % blocks, 1) for j in range(runs * blocks)]


def powerOfTwoFloor(x: int) -> int:
    return 1 << (int(x).bit_length() - 1)


def gate1Cases(rng) -> list:
    out = []
    for (R, C) in ((64, 64), (128, 64), (256, 128)):
        I = min(R, C)
        for d in (16, 32, 64):
            if d > I:
                continue
            S = I if I <= 64 else 64
            for nE in (d, 2 * d):
                for pp in (1, 2):
                    for runs, blocks in ((1, 1), (2, 2)):
                        p = qkParams(S, d, 16, R, C, nE, pProj=pp, runs=runs, blocks=blocks)
                        if not inDomain(p):
                            continue
                        for cls in ("random", "zeros", "extremes"):
                            out.append(randomCase(f"qk_{R}x{C}_d{d}_e{nE}_p{pp}_r{runs}b{blocks}_{cls}", p, runs,
                                                  blocks, jobsQk(runs, blocks), rng, cls=cls,
                                                  aPeriod=convPeriod(), bPeriod=convPeriod()))
            p = pvParams(S, d, 16, R, C, 2 * d, nExp=2, pProj=1, blocks=2)
            for cls in ("random", "zeros", "extremes"):
                out.append(randomCase(f"pv_{R}x{C}_d{d}_{cls}", p, 4, 2, jobsPv(2, 2), rng, cls=cls,
                                      aPeriod=convPeriod(), bPeriod=convPeriod()))
    return out


def testGate1(rng) -> int:
    cases = gate1Cases(rng)
    for c in cases:
        runCase(c)                        # raises on a GATE 1 mismatch
    return len(cases)


def closedForm(p: dict) -> dict:
    """Spec §5 by hand, one job from t_go, the operands early and the farm idle:
      A_GRP = the largest power of two <= min(P, X_S'), START_LAT = A_GRP + 2;
      wave t's burst at START_LAT + 10 t (the pass period P + 2), its drain_start 10 later (P + 2);
      its output words from drain_start + OUT_LAT, OUT_LAT = 1 (the sign XOR and the slice popcount) + [n_s > 1]
      (the sum across the slices) + 7 (the drain planes) + 2 (root, output buffer); done = the last output word + 2;
      idle = the last drain plane (drain_start + 7) + 2."""
    I = min(p["R"], p["C"])
    NG = p["N_E"] // p["K"]
    T_W = p["M"] * p["N"] // I // NG
    XS = min(p.get("X_S", 16), p["N_E"])
    start = powerOfTwoFloor(min(P, XS)) + 2
    outLat = 1 + int(p["N_E"] // XS > 1) + (P - 1) + 2
    burst = [start + (P + 2) * t for t in range(T_W)]
    drain = [b + P + 2 for b in burst]
    out = [ds + outLat for ds in drain]
    return dict(burst=burst, out=out, done=out[-1] + (P - 1) + 2, idle=drain[-1] + (P - 1) + 2, NG=NG)


def testClosed(rng) -> int:
    n = 0
    for (R, C) in ((64, 64), (128, 128), (256, 256), (128, 64), (512, 256)):
        I = min(R, C)
        for (M, N, K, nE) in ((I, 16, 64, 64), (I, 16, 64, 128), (64, 64, 16, 128), (32, 32, 16, 32),
                              (I, 16, 16, 16), (I, 16, 32, 64), (I, 16, 4, 4)):      # N_E 4: A_GRP 4 < P
            p = dict(M=M, N=N, K=K, R=R, C=C, N_E=nE, A_DOM=1, A_SIGNED=1, B_DOM=1, B_SIGNED=1, TRANS_B=0,
                     NARROW=0, C_CAP=4, B_SLOTS=1, A_SLOTS=1)
            if not inDomain(p):
                continue
            c = randomCase(f"cf_{R}x{C}_{M}_{N}_{K}_{nE}", p, 1, 1, [(0, 0, 0)] * 3, rng)
            rep = run(build(c, withValues=False)[0])
            want = closedForm(p)
            jobs = jobSteps(rep)
            windows = sorted((e.unit, e.window, e.channel, e.time) for e in rep.events
                             if e.kind == "win" and e.instance == "dimm" and e.port == "c")
            for s in jobs:
                tGo = s.events["t_go"]
                assert [w["load_start"] - tGo for w in s.events["waves"]] == want["burst"], (p, s.events["waves"])
                got = [(w, ch, t - tGo) for (j, w, ch, t) in windows if j == s.unit]
                expect = [(g + want["NG"] * t, g, want["out"][t]) for t in range(len(want["out"]))
                          for g in range(want["NG"])]
                assert sorted(got) == sorted(expect), (p, got, expect)
                assert s.events["done"] - tGo == want["done"], (p, s.events["done"] - tGo, want["done"])
                assert s.events["t_idle"] - tGo == want["idle"], (p, s.events["t_idle"] - tGo, want["idle"])
            # back to back: the next job decides the cycle the slices read idle
            for a, b in zip(jobs, jobs[1:]):
                assert b.events["t_dec"] == a.events["t_idle"] and b.events["t_go"] == a.events["t_idle"] + 1
            n += 1
    return n


def jobSteps(rep) -> list:
    return [s for s in rep.steps["dimm"] if s.process == "jobs"]


def creditTimes(rep, instance: str, port: str) -> list:
    return [e.time for e in rep.events if e.kind == "cr" and e.instance == instance and e.port == port]


def testRules(rng) -> int:
    """Each decision term binds exactly as spec §5."""
    n = 0
    p = qkParams(64, 16, 16, 64, 64, 32, pProj=1, runs=2, blocks=2, cCap=1)
    base = randomCase("rules", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod())
    rep = run(build(base, withValues=False)[0])
    # (a) late job offers: t_dec = t_in + 1 when the offer binds
    late = randomCase("late_jobs", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod(),
                      jAvailableAt=[5000, 9000, 13000, 17000])
    for s in jobSteps(run(build(late, withValues=False)[0])):
        assert s.events["t_dec"] == s.events["t_in"] + 1, s.events
    n += 1
    # (b) output credit starvation (C_CAP 1, the sink credits 400 cycles after the job's last plane)
    starved = randomCase("starve", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod(),
                         cCreditLatency=400)
    g, dimm, sink = build(starved, withValues=False)
    js = jobSteps(run(g))
    for j in range(1, len(js)):
        last = max(w.time for w in sink.received if w.unit == j - 1) + P - 1
        assert js[j].events["t_dec"] == max(last + 400 + 1, js[j].events["t_in"] + 1), (j, js[j].events)
    n += 1
    # (c) the A and B matrices: a job waits for its operands' valid time
    lateOperands = randomCase("late_ops", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod(),
                              bAvailableAt=[0, 6000])
    g, dimm, sink = build(lateOperands, withValues=False)
    js = jobSteps(run(g))
    previousIdle = 0
    for s in js:                      # t_dec = the max of its terms (C_CAP 1: the previous job's output credit)
        e = s.events
        terms = [e["t_in"] + 1, dimm.aValidAt[e["a_matrix"]], dimm.bValidAt[e["b_matrix"]], previousIdle]
        assert e["t_dec"] == max(terms) or s.unit >= 1, e
        assert e["t_dec"] >= max(terms), e
        previousIdle = e["t_idle"]
    assert js[1].events["t_dec"] == dimm.bValidAt[1] and dimm.bValidAt[1] > js[0].events["t_idle"], "B binds job 1"
    n += 1
    # (d) idle: the base case is back to back on the farm after the first job
    js = jobSteps(rep)
    for a, b in zip(js, js[1:]):
        assert b.events["t_dec"] >= a.events["t_idle"]
    n += 1
    return n


def testStaging(rng) -> int:
    n = 0
    # transposer: write times and credits per group, the slot valid time (row frame, two channels)
    p = qkParams(64, 32, 16, 64, 64, 64, pProj=2, runs=1, blocks=1)
    c = randomCase("tr", p, 1, 1, [(0, 0, 0)], rng, aPeriod=convPeriod(), bPeriod=convPeriod())
    g, dimm, sink = build(c, withValues=False)
    rep = run(g)
    d = dimm.d
    for q, tEmit in enumerate(dimm.aEmitAt):
        rows = range(q * d.NB, (q + 1) * d.NB)
        complete = max(dimm.aLastWord[r] for r in rows)
        previous = dimm.aEmitAt[q - 1] + d.KS if q else 0
        assert tEmit == max(complete + dimmTile.EMIT_LAT, previous)
    assert creditTimes(rep, "dimm", "a") == [tEmit + d.KS for tEmit in dimm.aEmitAt]
    assert dimm.aValidAt[0] == dimm.aEmitAt[-1] + d.KS
    # a row's planes reach the slices from its converter's drain_start + DIST_LAT, the last P - 1 later
    for k, step in enumerate(s for s in rep.steps["dimm"] if s.process == "a_in"):
        assert step.events["a_last"] == step.events["t_stream"] + 1 + P - 1, step.events
        # the converter: its burst one cycle after the port's first plane, drain_start P + 2 after the burst
        assert step.events["t_stream"] == step.events["t_port"] + 1 + P + 2, step.events
    n += 1
    # converter overrun: int8 rows closer than the converter period P + 2 raise
    bad = randomCase("conv_over", p, 1, 1, [(0, 0, 0)], rng, aPeriod=convPeriod() - 1, bPeriod=convPeriod())
    try:
        run(build(bad, withValues=False)[0])
    except ContractError as e:
        assert "converter" in str(e)
        n += 1
    else:
        raise AssertionError("converter overrun not detected")
    # window stager: t_e(w) = max(a_last(w) + 1, t_e(w - 1) + E_WP); PV at 64x64, d 16, N_E 32: 8 words a window per
    # slice, columns 0-3 in A memory 0 and 4-7 in memory 1 -> E_WP = 4 (by hand, spec §6 layout)
    pv = pvParams(32, 16, 16, 64, 64, 32, nExp=2, pProj=2, blocks=2)
    c = randomCase("rw", pv, 4, 2, jobsPv(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod())
    g, dimm, sink = build(c, withValues=False)
    rep = run(g)
    assert (dimm.d.A_GRP, dimm.d.N_AB, dimm.d.EW, dimm.d.E_WP) == (8, 2, 8, 4), dimm.d
    for k, tEmit in enumerate(dimm.aEmitAt):
        assert tEmit == max(dimm.aLastWord[k] + 1, (dimm.aEmitAt[k - 1] + 4) if k else 0), (k, tEmit)
    n += 1
    # window stager overrun: N_E = d puts 8 words of a window in one A memory (E_WP 8); two lanes at one window per
    # 10 cycles each need 16 write cycles per 10
    bad = randomCase("win_over", pvParams(32, 16, 16, 64, 64, 16, nExp=2, blocks=2), 4, 2, jobsPv(2, 2), rng,
                     aPeriod=convPeriod(), bPeriod=convPeriod())
    assert DimmTile.derive(DimmTileParams(**bad.params)).E_WP == 8
    try:
        run(build(bad, withValues=False)[0])
    except ContractError as e:
        assert "overrun" in str(e)
        n += 1
    else:
        raise AssertionError("window staging overrun not detected")
    # row writer: one row per cycle in row order, credit one cycle after the write
    for t, tWrite in enumerate(dimm.bWriteAt):
        assert tWrite == max(dimm.bLastWord[t] + 1, (dimm.bWriteAt[t - 1] + 1) if t else 0)
    assert creditTimes(rep, "dimm", "b") == [tWrite + 1 for tWrite in dimm.bWriteAt]
    n += 1
    # row order (spec §3): two channels, row 1 late -> row 2 reaches the port first and the model raises
    late = randomCase("rw_late", pv, 4, 2, jobsPv(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod(),
                      bAvailableAt=[3000 if k == 1 else 0 for k in range(2 * 16)])
    try:
        run(build(late, withValues=False)[0])
    except ContractError as e:
        assert "row order" in str(e)
        n += 1
    else:
        raise AssertionError("out-of-order B rows not detected")
    # released slots: the PV A credit is the slot release, one cycle after done
    assert creditTimes(rep, "dimm", "a") == [s.events["done"] + dimmTile.REL_LAT for s in jobSteps(rep)]
    n += 1
    return n


def candidates() -> list:
    out = []
    for (nseq, dModel, d) in SHAPES:
        for (R, C) in CONFIGS:
            I = min(R, C)
            S = min(nseq, I)
            runs, blocks = nseq // S, nseq // 16
            for pp in (1, 2, 4):
                for nE in (d, 2 * d):
                    out.append((dict(shape=(nseq, dModel, d), R=R, C=C, P_PROJ=pp, N_E=nE),
                                qkParams(S, d, 16, R, C, nE, pProj=pp, runs=runs, blocks=blocks),
                                pvParams(S, d, 16, R, C, 2 * d, nExp=2, pProj=pp, blocks=blocks)))
    return out


def testDomain() -> tuple:
    cs = candidates()
    assert len(cs) == len(SHAPES) * len(CONFIGS) * 3 * 2 == 324, len(cs)
    worst = 0.0
    for meta, qk, pv in cs:
        for p in (qk, pv):
            DimmTile.checkDomain(DimmTileParams(**p))
        d = DimmTile.derive(DimmTileParams(**pv))
        assert (d.EW, d.E_WP, d.T_CV) == (8, 4, 10), (meta, d)      # spec §5: 8 words over the 2 A memories
        rate = pv["A_CH"] * d.E_WP / d.T_CV
        worst = max(worst, rate)
        assert rate <= 1.0, (meta, rate)
    for bad in (dict(M=16, N=8, K=12), dict(M=16, N=8, K=16, N_E=8), dict(M=4, N=8, K=8), dict(M=16, N=128, K=8),
                dict(M=64, N=16, K=16, N_E=64, X_S=2), dict(M=64, N=16, K=16, A_CH=8), dict(M=64, N=16, K=16, P=4),
                dict(M=64, N=16, K=16, P=9), dict(M=64, N=16, K=16, NARROW=13),
                dict(M=64, N=16, K=16, A_FRAME=1, A_ROWS=2, A_DOM=1),
                dict(M=64, N=16, K=16, A_FRAME=1, A_ROWS=4, A_DOM=0),
                dict(M=64, N=16, K=16, A_FRAME=1, A_ROWS=8, A_DOM=1),       # A_ROWS K = 128 > I = 64
                dict(M=64, N=16, K=128)):
        p = dict(dict(N_E=16, R=64, C=64), **bad)
        if inDomain(p):
            raise AssertionError(f"checkDomain must reject {bad}")
    assert inDomain(dict(M=64, N=16, K=16, N_E=16, R=64, C=64, A_FRAME=1, A_ROWS=4, A_DOM=1))   # A_ROWS K = I
    return len(cs), worst


def testTimingAndOrder(rng) -> None:
    for c in (randomCase("t_qk", qkParams(64, 32, 16, 64, 64, 64, pProj=2, runs=2, blocks=2), 2, 2, jobsQk(2, 2),
                         rng, aPeriod=convPeriod(), bPeriod=convPeriod(), cCreditLatency=37),
              randomCase("t_pv", pvParams(32, 16, 16, 64, 64, 32, nExp=2, pProj=2, blocks=2), 4, 2,
                         jobsPv(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod())):
        a = run(build(c, withValues=True)[0])
        b = run(build(c, withValues=False)[0])
        assert a.events == b.events and a.cycles == b.cycles and a.counts["dimm"] == b.counts["dimm"]
        for seed in range(4):
            r = run(build(c, withValues=True)[0], order="random", seed=seed)
            assert r.events == a.events and r.cycles == a.cycles


def testMutants(rng) -> tuple:
    """Each mutant must fail a check: the port events (latencies, an early release), a deadlock (a dropped credit),
    or GATE 1 (a value mutant)."""
    qk = randomCase("mut_qk", qkParams(64, 32, 16, 64, 64, 64, pProj=1, runs=2, blocks=2, cCap=1), 2, 2,
                    jobsQk(2, 2), rng, aPeriod=convPeriod(), bPeriod=convPeriod(), cCreditLatency=50)
    pv = randomCase("mut_pv", pvParams(32, 16, 16, 64, 64, 32, nExp=2, blocks=2), 4, 2, jobsPv(2, 2), rng,
                    aPeriod=convPeriod(), bPeriod=convPeriod())
    ref = {c.name: run(build(c, withValues=False)[0]).events for c in (qk, pv)}
    caught, total = 0, 0
    for name, delta in (("DEC_LAT", 1), ("CR_LAT", 1), ("IDLE_LAT", 1), ("EMIT_LAT", 1), ("DIST_LAT", 1),
                        ("DONE_LAT", 1), ("SLICE_LAT", 1), ("ROOT_LAT", 1),
                        ("REL_LAT", -1)):                                        # REL_LAT - 1: released early
        old = getattr(dimmTile, name)
        setattr(dimmTile, name, old + delta)
        try:
            caught += any(run(build(c, withValues=False)[0]).events != ref[c.name] for c in (qk, pv))
        finally:
            setattr(dimmTile, name, old)
        total += 1
    # a dropped output credit: the sink never frees a job -> job 1 (C_CAP 1) never decides
    g, dimm, sink = build(qk, withValues=False)
    original = sink.stepUnit

    def noCredit(unit, io):
        io.giveCredit = lambda *a, **kw: None
        return original(unit, io)
    sink.stepUnit = noCredit
    try:
        run(g)
    except Deadlock:
        caught += 1
    total += 1
    # a value mutant: the X_ADD farm without its second identity block (the crossbar then exponentiates log a alone)
    # -> GATE 1 fails on a 128x64 QK case
    xa = randomCase("mut_xadd", qkParams(64, 32, 16, 128, 64, 32, pProj=1, runs=1, blocks=1), 1, 1, jobsQk(1, 1),
                    rng, aPeriod=convPeriod(), bPeriod=convPeriod())
    assert DimmTile.derive(DimmTileParams(**completeParams(xa.params))).XADD == 1
    oldW = dimmTile.stackedIdentity

    def plainIdentity(nE, R, C, I):
        W = oldW(nE, R, C, I)
        W[:, I:2 * I, :] = 0
        return W
    dimmTile.stackedIdentity = plainIdentity
    try:
        runCase(xa)
    except RuntimeError as e:
        caught += "GATE 1" in str(e)
    finally:
        dimmTile.stackedIdentity = oldW
    total += 1
    # a value mutant: the operand sign rule off by one bit -> GATE 1 fails
    old = dimmTile.signOfLog
    dimmTile.signOfLog = lambda L: old(np.asarray(L) ^ 1)
    try:
        runCase(qk)
    except RuntimeError as e:
        caught += "GATE 1" in str(e)
    finally:
        dimmTile.signOfLog = old
    total += 1
    assert caught == total, f"{caught}/{total} mutants caught"
    return caught, total


# Yosys's memories (words, bits): count of u_qk / u_pv, from each build's elab/elab.json (vtr_dse.py --elab-only,
# 2026-10-23: the bit-plane RTL), builds (n, d_model, d, R, C)
YOSYS_MEMS = {
    (128, 128, 64, 128, 64): ({(64, 128): 8, (128, 32): 16}, {(16, 512): 8, (128, 8): 16}),    # BERT-Tiny, I = 64
    (128, 128, 64, 256, 128): ({(64, 64): 16, (64, 128): 8}, {(16, 512): 8, (128, 16): 16}),   # I = 128, R != C
    (128, 128, 64, 256, 256): ({(32, 128): 16, (64, 128): 8}, {(16, 512): 8, (64, 32): 16}),   # I = 256
    (32, 64, 32, 256, 256): ({(16, 128): 12}, {(8, 256): 4, (32, 64): 8}),                     # A and B both 16 x 128
    (128, 128, 16, 256, 256): ({(32, 128): 4, (64, 128): 2}, {(64, 128): 6}),                  # JetFormer-JetClass
    (256, 768, 64, 512, 128): ({(128, 64): 16, (128, 128): 8}, {(32, 512): 8, (128, 16): 16}), # ViT-B/16, R = 512
}


def specMemories(p: dict, aGrp: int) -> dict:
    """Spec §6 by hand: per slice X_S'/A_GRP A memories of A_SLOTS T_W A_GRP words x NB bytes, and K_s/B_GRP B memories
    of B_SLOTS B_GRP words x N bytes, B_GRP = min(A_GRP, K_s); every address field is at least 1 bit (T_W, A_GRP and
    B_GRP are powers of two, so a field of x entries holds max(2, x))."""
    I = min(p["R"], p["C"])
    NB, NG = I // p["N"], p["N_E"] // p["K"]
    T_W = p["M"] * p["N"] // I // NG
    XS = min(p["X_S"], p["N_E"])
    NS, KS = p["N_E"] // XS, XS // NG
    bGrp = min(aGrp, KS)
    mems = Counter()
    mems[(p["A_SLOTS"] * max(2, T_W) * max(2, aGrp), NB * 8)] += NS * (XS // aGrp)
    mems[(p["B_SLOTS"] * max(2, bGrp), p["N"] * 8)] += NS * (KS // bGrp)
    return dict(mems)


def testXadd(rng) -> int:
    """Spec §10 item 11: X_ADD = [R >= 2I]; with it the EXP passes use 2I rows and the pairing adds leave the CLB
    count; the C values are the same on both branches (GATE 1 holds both; here the two geometries' outputs are
    compared on the same operands)."""
    n = 0
    for (R, C), xadd in (((64, 64), 0), ((128, 128), 0), ((128, 64), 1), ((256, 128), 1), ((512, 256), 1), ((256, 64), 1)):
        I = min(R, C)
        p = completeParams(qkParams(64, 32, 16, R, C, 32))
        d = DimmTile.derive(DimmTileParams(**p))
        assert d.XADD == xadd, (R, C, d.XADD)
        c = randomCase(f"xadd_{R}x{C}", p, 1, 1, jobsQk(1, 1), rng, aPeriod=convPeriod(), bPeriod=convPeriod())
        g, dimm, _ = build(c, withValues=False)
        rep = run(g)
        rows = 2 * I if xadd else I
        assert rep.totalCounts.passes.get(("EXP", rows, I), 0) == p["N_E"] * d.T_W, rep.totalCounts.passes
        adds = d.W_K * (p["K"] - 1) * I + (0 if xadd else p["K"] * p["M"] * p["N"])
        assert rep.totalCounts.clbAdd == adds, (rep.totalCounts.clbAdd, adds)
        assert list(dimm.resources.crossbars) and ("EXP", rows, I) in dimm.resources.crossbars
        n += 1
    # the same operands through a square and a tall array give the same C (the adder and the row-sum branches)
    seedA, seedB = np.random.default_rng(7), np.random.default_rng(7)
    cA = randomCase("xadd_sq", qkParams(64, 32, 16, 64, 64, 32), 1, 1, jobsQk(1, 1), seedA, aPeriod=convPeriod(),
                    bPeriod=convPeriod())
    cB = randomCase("xadd_tall", qkParams(64, 32, 16, 128, 64, 32), 1, 1, jobsQk(1, 1), seedB, aPeriod=convPeriod(),
                    bPeriod=convPeriod())
    runA, runB = build(cA), build(cB)
    run(runA[0]); run(runB[0])
    oa, ob = runA[1].math.outputs, runB[1].math.outputs
    assert oa and ob and all(np.array_equal(oa[j], ob[j]) for j in oa), "the two branches differ"
    n += 1
    return n


def testMemories() -> int:
    n = 0
    for (nseq, dModel, d, R, C), want in YOSYS_MEMS.items():
        I = min(R, C)
        S = min(nseq, I)
        qk = completeParams(qkParams(S, d, 16, R, C, 2 * d, pProj=1, runs=nseq // S, blocks=nseq // 16))
        pv = completeParams(pvParams(S, d, 16, R, C, 2 * d, nExp=2, pProj=1, blocks=nseq // 16))
        for p, yosys in zip((qk, pv), want):
            spec = specMemories(p, powerOfTwoFloor(min(P, min(p["X_S"], p["N_E"]))))
            assert spec == yosys, (p, spec, yosys)
            model = DimmTile("m", DimmTileParams(**p), nA=1, nB=1, nJobs=1, withValues=False)
            assert dict(model.resources.memories) == yosys, (p, dict(model.resources.memories), yosys)
            n += 1
    return n


def main() -> int:
    rng = np.random.default_rng(20261014)
    print(f"  gate1    PASS  {testGate1(rng)} cases (model C == oracle, every job)")
    print(f"  closed   PASS  {testClosed(rng)} configurations == spec §5's closed form (bursts, output windows, "
          "done, idle)")
    print(f"  rules    PASS  {testRules(rng)} decision terms bind as spec §5")
    print(f"  staging  PASS  {testStaging(rng)} loader checks (transposer, converter, window stager, overruns, row "
          "writer, release credits)")
    n, worst = testDomain()
    print(f"  domain   PASS  {n} DSE candidates inside the domain (QK and PV); window-frame rate {worst:.2f}")
    testTimingAndOrder(rng)
    print("  timing   PASS  (values off: same events, counts; 4 random step orders: same Report)")
    caught, total = testMutants(rng)
    print(f"  mutants  PASS  {caught}/{total} caught")
    print(f"  xadd     PASS  {testXadd(rng)} checks: X_ADD = [R >= 2I], 2I-row EXP passes, no pairing adds, equal C")
    print(f"  memories PASS  {testMemories()} instances: spec §6 == Yosys (the bit-plane RTL) == the model")
    print("dimm model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
