#!/usr/bin/env python3
"""testAttentionHead.py — gates of the head model: att_ctrl + the head workload (docs/simulator.md §4.5; top/spec.md §8).

  gate1    GATE 1 on a sweep of heads: 64x64, 128x64 (R != C) and 128x128; V = 1, 2, 3; one and several runs;
           P_PROJ 1 / 2 / 4; N_E = d and 2d; classes random / small / zeros; late and slow X, a late start
  rules    every att_ctrl event recomputed from spec §5 (each copy's x_rdy and bursts, the job takes, the softmax
           starts, done, busy falling); each admission term (the GEMMs' ready, the three staging credits, the run start)
           binds somewhere
  domain   the 324 DSE builds (6 shapes x 9 configs x P_PROJ {1, 2, 4} x N_E {d, 2d}) are inside every module's guard
  timing   values off: identical events and counts; random step orders: identical Report
  mutants  every att_ctrl latency + 1, a dropped staging credit and a dropped job credit (deadlock): each fails a check
Run: python3 v3/simulator/tests/testAttentionHead.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
import simulator.models.attentionController as controller  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import ContractError, Deadlock  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402
from top.cases import Case, build, randomCase, runCase  # noqa: E402

from dse.space import CONFIGS, SHAPES as NAMED_SHAPES  # noqa: E402  (the DSE's design space, defined once)
SHAPES = [shape[1:] for shape in NAMED_SHAPES]                       # (n, d_model, d)


def gate1Cases(rng) -> list:
    out = []

    def add(name, cls="random", **kw):
        shape = {k: v for k, v in kw.items() if k not in ("xAvailableAt", "xPeriod", "tStart")}
        arrival = {k: v for k, v in kw.items() if k in ("xAvailableAt", "xPeriod", "tStart")}
        out.append(randomCase(name, shape, rng, cls=cls, **arrival))
    add("n32_d16", NSEQ=32, DM=64, D=16, R=64, C=64)                                  # the smallest head
    add("n32_d16_p4", NSEQ=32, DM=32, D=16, R=64, C=64, P_PROJ=4, cls="small")
    add("n64_d32_p2", NSEQ=64, DM=32, D=32, R=64, C=64, P_PROJ=2)
    add("n64_d16_ne1", NSEQ=64, DM=64, D=16, R=64, C=64, N_E=16)                      # N_E = d
    add("n32_v2", NSEQ=32, DM=128, D=32, R=64, C=64)                                  # V = 2
    add("n32_v3_p2", NSEQ=32, DM=192, D=16, R=64, C=64, P_PROJ=2, cls="small")       # V = 3
    add("n128_d64", NSEQ=128, DM=128, D=64, R=64, C=64)                               # BERT-Tiny's geometry, 2 runs
    add("n128_d16_p2", NSEQ=128, DM=128, D=16, R=64, C=64, P_PROJ=2)                  # 2 runs, 8 blocks
    add("n64_128x64", NSEQ=64, DM=128, D=32, R=128, C=64)                             # R != C
    add("n128_128x128", NSEQ=128, DM=128, D=64, R=128, C=128, cls="small")            # S = 128, one run
    add("late_x", NSEQ=64, DM=64, D=16, R=64, C=64, P_PROJ=2, xAvailableAt=[60 * t for t in range(64)])
    add("slow_x", NSEQ=32, DM=64, D=32, R=64, C=64, xPeriod=90)
    add("late_start", NSEQ=32, DM=64, D=16, R=64, C=64, tStart=37)
    add("zeros", NSEQ=32, DM=64, D=16, R=64, C=64, cls="zeros")
    return out


def testGate1(rng) -> int:
    cases = gate1Cases(rng)
    for c in cases:
        runCase(c)                                    # raises on a GATE 1 mismatch
    return len(cases)


def eventTables(rep) -> tuple:
    windows, credits = {}, {}
    for e in rep.events:
        key = f"{e.instance}.{e.port}"
        if e.kind == "win":
            windows.setdefault(key, {})[(e.unit, e.window)] = e.time
        else:
            credits.setdefault(key, {})[e.unit] = e.time
    return windows, credits


def checkRules(c: Case, bind: dict) -> None:
    """Every att_ctrl event of a case recomputed from spec §5; `bind` collects the admission terms that bound."""
    g, parts = build(c, withValues=False)
    rep = run(g)
    hp, ctrl = c.params, parts["ctrl"]
    windows, credits = eventTables(rep)
    P, B, RB = hp.P_PROJ, hp.B, hp.RB
    tStart = windows["st.start"][(0, 0)]
    tRun = tStart + controller.RUN_LAT
    for p in range(P):
        for k in range(hp.T_X):
            t = p + k * P
            burst = windows[f"x{p}.x"][(k, 0)]
            if k == 0:
                assert burst >= tRun
                continue
            terms = dict(run=tRun, gemm=max(credits[f"g{kind}{p}.act"][k - 1] for kind in "qkv"))
            for name, key, i in (("qa", "qk.a", t // (hp.I // hp.BKV) - 2), ("qb", "qk.b", t // hp.BKV - 2),
                                 ("pb", "pv.b", t - 4 * P)):
                if i >= 0:
                    terms[name] = credits[key][i] + controller.CR_LAT
            xReady = max(terms.values())
            assert credits[f"ctrl.x{p}"][k - 1] == xReady, (c.name, p, k, credits[f"ctrl.x{p}"][k - 1], terms)
            assert burst >= xReady, (c.name, p, k)
            for name, v in terms.items():
                if v == xReady:
                    bind[name] = bind.get(name, 0) + 1
        assert len(credits[f"ctrl.x{p}"]) == hp.T_X - 1
    for j in range(RB * B):
        for port, dimm in (("qj", "qk"), ("pj", "pv")):
            want = tRun if j == 0 else max(tRun, credits[f"{dimm}.j"][j - 1])
            assert windows[f"ctrl.{port}"][(j, 0)] == want, (c.name, port, j)
    for r in range(RB):
        want = tRun if r == 0 else max(tRun, credits["sm.go"][r - 1])
        assert windows["ctrl.go"][(r, 0)] == want, (c.name, r)
    done = credits["acc.run_done"][RB - 1] + controller.DONE_LAT
    assert ctrl.done == done and credits["ctrl.start"][0] == done + controller.BUSY_LAT
    assert credits["acc.run_done"][RB - 1] == parts["acc"].runEndAt[RB - 1] + 1   # the last run's end (its last E9 word)
    assert rep.cycles == parts["acc"].runEndAt[RB - 1] + 1 < done                 # the last E9 word + 1, before done


def rulesCases(rng) -> list:
    return [randomCase("r_base", dict(NSEQ=64, DM=64, D=16, R=64, C=64), rng),
            randomCase("r_p4", dict(NSEQ=64, DM=32, D=16, R=64, C=64, P_PROJ=4), rng),
            randomCase("r_p2_d32", dict(NSEQ=128, DM=32, D=32, R=64, C=64, P_PROJ=2), rng),
            randomCase("r_late", dict(NSEQ=64, DM=64, D=32, R=64, C=64, P_PROJ=2), rng,
                       xAvailableAt=[40 * t for t in range(64)], tStart=11),
            randomCase("r_v2", dict(NSEQ=64, DM=128, D=64, R=64, C=64), rng)]


def testRules(rng) -> dict:
    bind = {}
    for c in rulesCases(rng):
        checkRules(c, bind)
    # run binds only token 0; qb only with a slow K transposer; pb never at 4 rows per copy (its loop, X burst -> V
    # row -> conversion -> write -> pb_cr, is shorter than 4 GEMM row periods on every legal build)
    for k in ("gemm", "qa"):
        assert bind.get(k, 0) > 0, f"admission term {k} never binds ({bind})"
    assert bind.get("pb", 0) == 0, f"the PV B credit binds at 4 rows per copy ({bind})"
    # the X port's order rule (spec §3): a source that starves copy 1 breaks token order and the model raises
    c = randomCase("r_skew", dict(NSEQ=64, DM=64, D=16, R=64, C=64, P_PROJ=2), rng,
                   xAvailableAt=[2000 if t % 2 else 0 for t in range(64)])
    try:
        run(build(c, withValues=False)[0])
    except ContractError as e:
        assert "row order" in str(e), e
    else:
        raise AssertionError("X bursts out of token order not detected")
    return bind


def testDomain() -> int:
    n = 0
    for (nseq, dm, d) in SHAPES:
        for (R, C) in CONFIGS:
            for pp in (1, 2, 4):
                for ne in (d, 2 * d):
                    attentionHead(headParams(nseq, dm, d, R, C, P_PROJ=pp, N_E=ne), withValues=False)   # every guard
                    n += 1
    return n


def testTimingAndOrder(rng) -> None:
    for c in (randomCase("t_a", dict(NSEQ=64, DM=64, D=16, R=64, C=64, P_PROJ=2), rng),
              randomCase("t_b", dict(NSEQ=128, DM=64, D=32, R=64, C=64), rng, xPeriod=50)):
        a = run(build(c, withValues=True)[0])
        b = run(build(c, withValues=False)[0])
        assert a.events == b.events and a.cycles == b.cycles
        assert all(a.counts[k] == b.counts[k] for k in a.counts)
        for seed in range(3):
            r = run(build(c, withValues=True)[0], order="random", seed=seed)
            assert r.events == a.events and r.cycles == a.cycles


MUTANT_LATS = ("RUN_LAT", "CR_LAT", "DONE_LAT", "BUSY_LAT")


def testMutants(rng) -> tuple:
    cases = [randomCase("m_a", dict(NSEQ=64, DM=32, D=16, R=64, C=64, P_PROJ=4), rng),
             randomCase("m_b", dict(NSEQ=64, DM=64, D=16, R=64, C=64), rng, tStart=5)]

    def signature(c):
        g, parts = build(c, withValues=False)
        rep = run(g)
        return tuple(rep.events), parts["ctrl"].done
    ref = {c.name: signature(c) for c in cases}
    caught, total = 0, 0
    for name in MUTANT_LATS:
        old = getattr(controller, name)
        setattr(controller, name, old + 1)
        try:
            caught += any(signature(c) != ref[c.name] for c in cases)
        finally:
            setattr(controller, name, old)
        total += 1
    for drop in ("a", "j"):                            # the QK DIMM never frees its A staging / its job register
        g, parts = build(cases[0], withValues=False)
        qk = parts["qk"]
        original = qk.step

        def step(process, unit, io, original=original, drop=drop):
            real = io.giveCredit
            io.giveCredit = lambda port, *a, **kw: None if port == drop else real(port, *a, **kw)
            return original(process, unit, io)
        qk.step = step
        try:
            run(g)
        except Deadlock:
            caught += 1
        total += 1
    assert caught == total, f"{caught}/{total} mutants caught"
    return caught, total


def main() -> int:
    rng = np.random.default_rng(20261014)
    t0 = time.time()
    print(f"  gate1    PASS  {testGate1(rng)} heads (O8 and every stream inside == head_stages)  "
          f"[{time.time() - t0:.0f}s]")
    b = testRules(rng)
    print(f"  rules    PASS  every att_ctrl event recomputed from spec §5; admission terms bound: {dict(sorted(b.items()))}")
    print(f"  domain   PASS  {testDomain()} DSE builds inside every module's guard")
    testTimingAndOrder(rng)
    print("  timing   PASS  (values off: same events, counts; 3 random step orders: same Report)")
    c, t = testMutants(rng)
    print(f"  mutants  PASS  {c}/{t} caught")
    print(f"head model: ALL PASS  [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
