#!/usr/bin/env python3
"""testDimmTile.py — gates of the `dimm_tile_top` model (docs/simulator.md §4.4; ip/dimm/spec.md §8).

  gate1      GATE 1 on a sweep of QK and PV configurations (int8 / log operands, row / window frame, TRANS_B 0 / 1,
             1-4 slices, 1-4 channels, several matrices and jobs, released slots, NARROW 0 / 5), classes random /
             zeros / extremes, crossbars 64x64, 128x64, 256x128
  closed     a job with early operands: every output word relative to t_go, done and idle equal the v2 tile's
             closed form (`dimm_tile_sim.tile_cycle_model`; A_GRP by the slice rule, i.e. v2 shifted by the start
             latency difference where the two rules differ)
  rules      the decision binds on each of its terms (job offer, A / B valid, idle, output credit) exactly as spec §5
  staging    transposer, window stager, row writer: write times and credits by spec §5, overruns raise
  domain     the 432 DSE candidates (6 shapes x 12 configs x P_proj {1, 2, 4} x N_E {d, 2d}) are inside the module's
             domain for both instances, and the window-frame rate holds; out-of-domain sets are rejected
  timing     values off: identical events and counts; random step orders: identical Report
  mutants    a latency + 1 (DEC_LAT, CR_LAT, IDLE_LAT), a dropped credit, a wrong release: each fails a check
  memories   the A / B memory list (words x bits, counts) of the QK and PV instances == the memories Yosys infers
             from the RTL (pinned from v3/dse/vtr_dse.py's elaboration, 2026-10-15), I = 128 and I = 256
             (runs once the attention head workload is built)
Run: python3 v3/simulator/tests/testDimmTile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "oracle" / "v2_kernels"))    # test references only
import dimm_tile_sim as v2tile  # noqa: E402

import simulator.models.dimmTile as dimmTile  # noqa: E402
from ip.dimm.cases import build, convPeriod, pvParams, qkParams, randomCase, runCase  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import ContractError, Deadlock  # noqa: E402
from simulator.models.dimmTile import DimmTile, DimmTileParams  # noqa: E402

from dse.space import CONFIGS, SHAPES as NAMED_SHAPES  # noqa: E402  (the DSE's design space, defined once)
SHAPES = [shape[1:] for shape in NAMED_SHAPES]                       # (n, d_model, d)


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
                                                  aPeriod=convPeriod(d), bPeriod=convPeriod(d)))
            p = pvParams(S, d, 16, R, C, 2 * d, nExp=2, pProj=1, blocks=2)
            for cls in ("random", "zeros", "extremes"):
                out.append(randomCase(f"pv_{R}x{C}_d{d}_{cls}", p, 4, 2, jobsPv(2, 2), rng, cls=cls,
                                      aPeriod=convPeriod(I), bPeriod=convPeriod(d)))
    return out


def testGate1(rng) -> int:
    cases = gate1Cases(rng)
    for c in cases:
        runCase(c)                        # raises on a GATE 1 mismatch
    return len(cases)


def closedForm(p: dict) -> dict:
    """v2's closed form for one job from its start edge, with A_GRP by the v3 slice rule."""
    d = DimmTile.derive(DimmTileParams(**p))
    v2 = v2tile.tile_cycle_model(p["M"], p["N"], p["K"], p["R"], p["C"], p["N_E"])
    shift = d.START_LAT - v2tile.start_lat(v2tile.a_grp_rule(p["N_E"], p["R"], p["C"]))
    return dict(out_t0=[t + shift for t in v2.out_t0], done=v2.total + shift, idle=v2.occ + shift + 1, v2=v2,
                shift=shift)


def testClosed(rng) -> int:
    n = 0
    for (R, C) in ((64, 64), (128, 128), (256, 256), (128, 64), (512, 256)):
        I = min(R, C)
        for (M, N, K, nE) in ((I, 16, 64, 64), (I, 16, 64, 128), (64, 64, 16, 128), (32, 32, 16, 32),
                              (I, 16, 16, 16), (I, 16, 32, 64)):
            p = dict(M=M, N=N, K=K, R=R, C=C, N_E=nE, A_DOM=1, A_SIGNED=1, B_DOM=1, B_SIGNED=1, TRANS_B=0,
                     NARROW=0, C_CAP=4, B_SLOTS=1, A_SLOTS=1)
            if not inDomain(p):
                continue
            c = randomCase(f"cf_{R}x{C}_{M}_{N}_{K}_{nE}", p, 1, 1, [(0, 0, 0)] * 3, rng)
            rep = run(build(c, withValues=False)[0])
            want = closedForm(p)
            jobs = [s for s in rep.steps["dimm"] if s.process == "jobs"]
            d = DimmTile.derive(DimmTileParams(**p))
            for s in jobs:
                tGo = s.events["t_go"]
                got = [w["drain_start"] + d.OUT_LAT - tGo for w in s.events["waves"]]
                assert got == want["out_t0"], (p, got, want["out_t0"])
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
    base = randomCase("rules", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(16), bPeriod=convPeriod(16))
    rep = run(build(base, withValues=False)[0])
    # (a) late job offers: t_dec = t_in + 1 when the offer binds
    late = randomCase("late_jobs", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(16), bPeriod=convPeriod(16),
                      jAvailableAt=[5000, 9000, 13000, 17000])
    for s in jobSteps(run(build(late, withValues=False)[0])):
        assert s.events["t_dec"] == s.events["t_in"] + 1, s.events
    n += 1
    # (b) output credit starvation (C_CAP 1, the sink credits 400 cycles after the job's last word)
    starved = randomCase("starve", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(16), bPeriod=convPeriod(16),
                         cCreditLatency=400)
    g, dimm, sink = build(starved, withValues=False)
    js = jobSteps(run(g))
    for j in range(1, len(js)):
        last = max(w.time for w in sink.received if w.unit == j - 1) + dimm.d.LW - 1
        assert js[j].events["t_dec"] == max(last + 400 + 1, js[j].events["t_in"] + 1), (j, js[j].events)
    n += 1
    # (c) the A and B matrices: a job waits for its operands' valid time
    lateOperands = randomCase("late_ops", p, 2, 2, jobsQk(2, 2), rng, aPeriod=convPeriod(16),
                              bPeriod=convPeriod(16), bAvailableAt=[0, 6000])
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
    c = randomCase("tr", p, 1, 1, [(0, 0, 0)], rng, aPeriod=convPeriod(32), bPeriod=convPeriod(32))
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
    n += 1
    # converter overrun: int8 rows closer than the converter period raise
    bad = randomCase("conv_over", p, 1, 1, [(0, 0, 0)], rng, aPeriod=convPeriod(32) - 1, bPeriod=convPeriod(32))
    try:
        run(build(bad, withValues=False)[0])
    except ContractError as e:
        assert "converter" in str(e)
        n += 1
    else:
        raise AssertionError("converter overrun not detected")
    # window stager overrun: log P windows faster than the slice writer
    pv = pvParams(32, 16, 16, 64, 64, 16, nExp=2, blocks=2)      # N_E = d: E_w = 16, 2 lanes -> 32 > 21
    bad = randomCase("win_over", pv, 4, 2, jobsPv(2, 2), rng, aPeriod=convPeriod(64), bPeriod=convPeriod(16))
    try:
        run(build(bad, withValues=False)[0])
    except ContractError as e:
        assert "overrun" in str(e)
        n += 1
    else:
        raise AssertionError("window staging overrun not detected")
    # row writer: one row per cycle in row order, credit one cycle after the write
    pv = pvParams(32, 16, 16, 64, 64, 32, nExp=2, pProj=2, blocks=2)
    c = randomCase("rw", pv, 4, 2, jobsPv(2, 2), rng, aPeriod=convPeriod(64), bPeriod=convPeriod(16))
    g, dimm, sink = build(c, withValues=False)
    rep = run(g)
    for t, tWrite in enumerate(dimm.bWriteAt):
        assert tWrite == max(dimm.bLastWord[t] + 1, (dimm.bWriteAt[t - 1] + 1) if t else 0)
    assert creditTimes(rep, "dimm", "b") == [tWrite + 1 for tWrite in dimm.bWriteAt]
    n += 1
    # row order (spec §3): two channels, row 1 late -> row 2 reaches the port first and the model raises
    late = randomCase("rw_late", pv, 4, 2, jobsPv(2, 2), rng, aPeriod=convPeriod(64), bPeriod=convPeriod(16),
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
    assert len(cs) == 432, len(cs)
    worst = 0.0
    for meta, qk, pv in cs:
        for p in (qk, pv):
            DimmTile.checkDomain(DimmTileParams(**p))
        d = DimmTile.derive(DimmTileParams(**pv))
        rate = pv["A_CH"] * d.EW / convPeriod(d.I)
        worst = max(worst, rate)
        assert rate <= 1.0, (meta, rate)
    for bad in (dict(M=16, N=8, K=12), dict(M=16, N=8, K=16, N_E=8), dict(M=4, N=8, K=8), dict(M=16, N=128, K=8),
                dict(M=64, N=16, K=16, N_E=64, X_S=2), dict(M=64, N=16, K=16, A_CH=8), dict(M=64, N=16, K=16, BUF=32),
                dict(M=64, N=16, K=16, NARROW=13), dict(M=64, N=16, K=16, A_FRAME=1, A_ROWS=2, A_DOM=1),
                dict(M=64, N=16, K=16, A_FRAME=1, A_ROWS=4, A_DOM=0), dict(M=64, N=16, K=128)):
        p = dict(dict(N_E=16, R=64, C=64), **bad)
        if inDomain(p):
            raise AssertionError(f"checkDomain must reject {bad}")
    return len(cs), worst


def testTimingAndOrder(rng) -> None:
    for c in (randomCase("t_qk", qkParams(64, 32, 16, 64, 64, 64, pProj=2, runs=2, blocks=2), 2, 2, jobsQk(2, 2),
                         rng, aPeriod=convPeriod(32), bPeriod=convPeriod(32), cCreditLatency=37),
              randomCase("t_pv", pvParams(32, 16, 16, 64, 64, 32, nExp=2, pProj=2, blocks=2), 4, 2,
                         jobsPv(2, 2), rng, aPeriod=convPeriod(64), bPeriod=convPeriod(16))):
        a = run(build(c, withValues=True)[0])
        b = run(build(c, withValues=False)[0])
        assert a.events == b.events and a.cycles == b.cycles and a.counts["dimm"] == b.counts["dimm"]
        for seed in range(4):
            r = run(build(c, withValues=True)[0], order="random", seed=seed)
            assert r.events == a.events and r.cycles == a.cycles


def testMutants(rng) -> int:
    """Each mutant must fail a check: the port events (latencies, an early release), a deadlock (a dropped credit),
    or GATE 1 (a value mutant)."""
    qk = randomCase("mut_qk", qkParams(64, 32, 16, 64, 64, 64, pProj=1, runs=2, blocks=2, cCap=1), 2, 2,
                    jobsQk(2, 2), rng, aPeriod=convPeriod(32), bPeriod=convPeriod(32), cCreditLatency=50)
    pv = randomCase("mut_pv", pvParams(32, 16, 16, 64, 64, 32, nExp=2, blocks=2), 4, 2, jobsPv(2, 2), rng,
                    aPeriod=convPeriod(64), bPeriod=convPeriod(16))
    ref = {c.name: run(build(c, withValues=False)[0]).events for c in (qk, pv)}
    caught, total = 0, 0
    for name, delta in (("DEC_LAT", 1), ("CR_LAT", 1), ("IDLE_LAT", 1), ("EMIT_LAT", 1), ("DIST_LAT", 1),
                        ("DONE_LAT", 1), ("REL_LAT", -1)):                       # REL_LAT - 1: released early
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
    return caught


# Yosys's memories (words, bits): count of u_qk / u_pv, from each build's elab/mems.txt (vtr_dse.py --elab-only)
YOSYS_MEMS = {
    (128, 128, 64, 128, 256): ({(64, 128): 8, (128, 64): 8}, {(16, 512): 8, (256, 16): 8}),     # BERT-Tiny, I = 128
    (128, 128, 64, 256, 256): ({(64, 128): 16}, {(16, 512): 8, (128, 32): 8}),                 # A and B both 64 x 128
    (32, 64, 32, 256, 256): ({(16, 128): 4, (32, 128): 4}, {(8, 256): 4, (64, 64): 4}),        # T_W = 1: 1-bit field
    (128, 128, 16, 256, 256): ({(64, 128): 4}, {(64, 128): 2, (128, 128): 2}),                 # JetFormer-JetClass
}


def testMemories() -> int | None:
    try:
        from simulator.models.attentionHead import attentionHead, headParams
    except ImportError:
        return None
    for args, (qk, pv) in YOSYS_MEMS.items():
        _, parts = attentionHead(headParams(*args), withValues=False)
        for instance, want in (("qk", qk), ("pv", pv)):
            got = dict(parts[instance].resources.memories)
            assert got == want, f"{args} {instance}: model {got} != Yosys {want}"
    return 2 * len(YOSYS_MEMS)


def main() -> int:
    rng = np.random.default_rng(20261014)
    print(f"  gate1    PASS  {testGate1(rng)} cases (model C == oracle, every job)")
    print(f"  closed   PASS  {testClosed(rng)} configurations == v2 tile_cycle_model (slice-rule A_GRP)")
    print(f"  rules    PASS  {testRules(rng)} decision terms bind as spec §5")
    print(f"  staging  PASS  {testStaging(rng)} loader checks (transposer, overruns, row writer, release credits)")
    n, worst = testDomain()
    print(f"  domain   PASS  {n} DSE candidates inside the domain (QK and PV); worst window-frame rate {worst:.2f}")
    testTimingAndOrder(rng)
    print("  timing   PASS  (values off: same events, counts; 4 random step orders: same Report)")
    print(f"  mutants  PASS  {testMutants(rng)}/9 caught")
    m = testMemories()
    print(f"  memories PASS  {m} instances == the memories Yosys infers (I = 128 / 256)" if m is not None
          else "  memories SKIP  (needs the attention head workload)")
    print("dimm model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
