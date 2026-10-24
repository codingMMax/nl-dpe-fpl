#!/usr/bin/env python3
"""testSoftmaxAccumulator.py — gates of the `softmax_online_acc` model (docs/simulator.md §4.4; ip/softmax/acc_spec.md
§8), the bit-plane pass (2026-10-22).

  gate1    GATE 1 on a sweep: S 8..128, D 8..64, N_OC 1..8 (partial last waves), NBLK 1..4, P_CH 1..8, SO / LN equal and
           unequal, 1-3 runs; crossbars 64x64, 128x64, 128x128; classes random / zeros / allneg / extremes / small /
           alpha0; late and slow inputs
  closed   the jobs fed input schedules (early, staggered windows, late alpha, late log l) == spec §5's job rules
           written here by hand: every LOG and EXP job's decide, load_start, msb_fire, drain_start, drain_end; p_cr,
           la_cr, ll_cr; the run end (until 2026-10-22 the witness was v2's `ocarry_cycle_model`, which times the
           pre-bit-plane pass)
  rules    every job decision, credit, E9 window and run_done recomputed from the spec §5 formulas; each readiness
           term and each input credit binds somewhere
  domain   the DSE's accumulator instances (6 shapes x 9 configs, N_EP = 2d, N_OC by the floor rule) are inside the
           guard; out-of-domain sets are rejected
  timing   values off: identical events and counts; random step orders: identical Report
  mutants  every named latency + 1 (events or run_done change), a dropped alpha credit (deadlock), a value mutant
           (GATE 1): each fails a check
Run: python3 v3/simulator/tests/testSoftmaxAccumulator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))

import simulator.models.softmaxAccumulator as accumulator  # noqa: E402
from ip.softmax.acc_cases import Case, build, randomCase, runCase  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import Deadlock  # noqa: E402
from simulator.models.softmaxAccumulator import SoftmaxAccParams, SoftmaxOnlineAcc  # noqa: E402

from dse.space import CONFIGS, SHAPES as NAMED_SHAPES  # noqa: E402  (the DSE's design space, defined once)
SHAPES = [shape[1:] for shape in NAMED_SHAPES]                       # (n, d_model, d)
BKV = 16
P = 8                                                                # plane words per window (spec §2)
PASS = P + 2                                                         # a crossbar's pass period (ip/xbar/spec.md)


def inDomain(p: dict) -> bool:
    try:
        SoftmaxOnlineAcc.checkDomain(SoftmaxAccParams(**p))
        return True
    except ValueError:
        return False


def nOcFloor(S: int, D: int, R: int, C: int, bkv: int = BKV) -> int:
    """architecture.md §5.2, held at its pre-bit-plane form: the smallest N_OC with T(ceil(P_oc / N_OC)) <= max(ceil(S Bkv / 5),
    T(1)), T the pre-bit-plane pass."""
    I = min(R, C)
    LW = -(-I // 5)
    tFill, tSteady = LW + 10 + LW, max(LW + 8, 10, LW + 1)
    pOc = S * D // I

    def T(n: int) -> int:
        return tFill + (n - 1) * tSteady
    floor = max(-(-S * bkv // 5), T(1))
    return next(n for n in range(1, pOc + 1) if T(-(-pOc // n)) <= floor)


def gate1Cases(rng) -> list:
    out = []
    classes = ("random", "zeros", "allneg", "extremes", "small", "alpha0")
    for (R, C) in ((64, 64), (128, 64), (128, 128)):
        for S in (8, 16, 32, 64, 128):
            for D in (8, 16, 32, 64):
                for nOc, pch in ((1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8)):
                    for B in (1, 2, 4):
                        p = dict(S=S, D=D, R=R, C=C, N_OC=nOc, NBLK=B, SO=4, LN=4, P_CH=pch)
                        if not inDomain(p) or (B != 2 and (D != 16 or nOc not in (1, 3))):
                            continue
                        cls = classes[len(out) % len(classes)]
                        out.append(randomCase(f"g1_{R}x{C}_S{S}_D{D}_n{nOc}_c{pch}_B{B}_{cls}", p,
                                              1 + len(out) % 3, rng, cls=cls))
    for so, ln in ((5, 3), (2, 6), (0, 0)):                      # the epilogue scale 2^(SO - LN) both ways
        p = dict(S=16, D=16, R=64, C=64, N_OC=2, NBLK=3, SO=so, LN=ln, P_CH=2)
        out.append(randomCase(f"g1_so{so}_ln{ln}", p, 2, rng, cls="extremes"))
    # arrival patterns (values on)
    p = dict(S=32, D=16, R=64, C=64, N_OC=3, NBLK=3, SO=5, LN=5, P_CH=4)
    for kw in (dict(pAvailableAt=[500 * j for j in range(6)]), dict(pPeriod=40),
               dict(laAvailableAt=[900 * u for u in range(4)]), dict(laPeriod=7), dict(llAvailableAt=[0, 4000])):
        out.append(randomCase("g1_arr_" + "_".join(kw), p, 2, rng, **kw))
    return out


def testGate1(rng) -> int:
    cases = gate1Cases(rng)
    for c in cases:
        runCase(c)                        # raises on a GATE 1 mismatch
    return len(cases)


def handJobs(S: int, D: int, I: int, nOc: int, B: int, runs: int, tLast, tAlpha, tLogL) -> tuple:
    """Spec §5's job rules by hand. tLast(r, b, q): window q's last plane; tAlpha(r, b), tLogL(r): the units' last
    planes. A pass loading at ls fires its MSB at ls + 8, its ACAM at ls + 9, drains ls + 10 .. ls + 17, and the
    leader takes the next burst from ls + 10; a job decided at d loads at d + 17 (LOG) or d + 10 (EXP), the group's
    next job decides at d + 16 (LOG, the single corner-turns) or d + 17 (EXP, the single hold) at the earliest; an
    EXP job's last lo plane read is at d + 7, a LOG job's last o_part word read at d + 15. Returns
    ({(r, kind, b, w): events}, {credit: times}, {r: t_o8})."""
    pOc = S * D // I
    W = -(-pOc // nOc)
    jobs, credits, runEnd = {}, {}, {}
    leader = {"log": None, "exp": None}              # the group's last load_start
    gap = {"log": 16, "exp": 17}                     # the single feed buffers (spec §5 LOG_GAP, EXP_GAP)

    def place(group, key, ready, pre):
        ls = ready + pre if leader[group] is None else max(ready + pre, leader[group] + PASS,
                                                           leader[group] - pre + gap[group] + pre)
        leader[group] = ls
        jobs[key] = dict(decide=ls - pre, load_start=ls, msb_fire=ls + 8, drain_start=ls + 10, drain_end=ls + 17)
        return jobs[key]
    for r in range(runs):
        for b in range(B):
            for w in range(W):
                wave = range(w * nOc, min((w + 1) * nOc, pOc))
                if b:
                    e = place("exp", (r, "rescale", b, w),
                              max(tAlpha(r, b) + 1, jobs[(r, "log", b - 1, w)]["drain_end"] + 1), 10)
                    if w == W - 1:
                        credits[("la", r, b)] = e["decide"] + 7 + 2
                ready = max(tLast(r, b, q) for q in wave) + 1
                if b:
                    ready = max(ready, jobs[(r, "rescale", b, w)]["drain_end"] + 1)
                elif r:
                    ready = max(ready, runEnd[r - 1] + 1)
                e = place("log", (r, "log", b, w), ready, 17)
                if w == W - 1:
                    credits[("p", r, b)] = e["decide"] + 15 + 1
        for w in range(W):
            e = place("exp", (r, "epilogue", B - 1, w),
                      max(tLogL(r) + 1, jobs[(r, "log", B - 1, w)]["drain_end"] + 1), 10)
        credits[("ll", r)] = e["decide"] + 7 + 2
        runEnd[r] = e["drain_start"] + 10 + 7                         # the last E9 plane
    return jobs, credits, runEnd


def testClosed() -> int:
    n = 0
    keys = ("decide", "load_start", "msb_fire", "drain_start", "drain_end")
    schedules = {                                       # (tLast(r, b, q), tAlpha(r, b), tLogL(r))
        "early": (lambda r, b, q: 7, lambda r, b: 7, lambda r: 7),
        "staggered": (lambda r, b, q: 900 * r + 120 * b + 10 * q, lambda r, b: 900 * r + 120 * b, lambda r: 900 * r),
        "late_alpha": (lambda r, b, q: 7, lambda r, b: 2000 * r + 400 * b, lambda r: 7),
        "late_logl": (lambda r, b, q: 7, lambda r, b: 7, lambda r: 3000 * (r + 1)),
        "slow_windows": (lambda r, b, q: 1500 * r + 300 * b + 40 * q, lambda r, b: 7, lambda r: 7),
    }
    for (S, d, R, C, nOc, B, ngp) in ((16, 16, 64, 64, 2, 3, 2), (32, 8, 64, 64, 4, 4, 4), (16, 16, 64, 64, 2, 3, 4),
                                      (128, 64, 256, 256, 7, 8, 8), (64, 64, 64, 64, 8, 2, 8),
                                      (16, 16, 64, 64, 2, 1, 2), (32, 32, 64, 64, 3, 2, 4), (64, 32, 128, 64, 5, 3, 8)):
        I = min(R, C)
        for name, (tLast, tAlpha, tLogL) in schedules.items():
            acc = SoftmaxOnlineAcc("acc", SoftmaxAccParams(S=S, D=d, R=R, C=C, N_OC=nOc, NBLK=B, SO=4, LN=4, P_CH=ngp),
                                   runs=3, withValues=False)
            got = {}
            for r in range(3):
                for b in range(B):
                    c = acc.block(r * B + b, [tLast(r, b, q) for q in range(acc.d.P_OC)], tAlpha(r, b) if b else None)
                    got[("p", r, b)] = c["p_cr"]
                    if b:
                        got[("la", r, b)] = c["la_cr"]
                got[("ll", r)] = acc.epilogue(r, tLogL(r))["ll_cr"]
            jobs, credits, runEnd = handJobs(S, d, I, nOc, B, 3, tLast, tAlpha, tLogL)
            for key, ev in jobs.items():
                assert all(acc.jobs[key][k] == ev[k] for k in keys), (S, d, name, key, acc.jobs[key], ev)
            assert len(acc.jobs) == len(jobs) and got == credits, (S, d, name)
            assert all(acc.runEndAt[r] == runEnd[r] for r in range(3)), (S, d, name, acc.runEndAt, runEnd)
            n += 1
    return n


def eventTables(rep) -> dict:
    """{key: {(unit, w): (t, ch)}} windows and {key: [t by unit]} credits from the report's events."""
    windows, credits = {}, {}
    for e in rep.events:
        key = f"{e.instance}.{e.port}"
        if e.kind == "win":
            windows.setdefault(key, {})[(e.unit, e.window)] = (e.time, e.channel)
        else:
            credits.setdefault(key, {})[e.unit] = e.time
    return dict(win=windows, cr={k: [v[u] for u in sorted(v)] for k, v in credits.items()})


def checkRules(c: Case, bind: dict) -> None:
    """Every job, credit and output of a case recomputed from spec §5; `bind` collects which terms bound."""
    g, acc, sink = build(c, withValues=False)
    rep = run(g)
    p, d = acc.p, acc.d
    S, B, W, n = p.S, p.NBLK, d.W, p.N_OC
    ev = eventTables(rep)
    cr = ev["cr"]
    pw, law, llw = ev["win"]["p_src.p"], ev["win"].get("la_src.la", {}), ev["win"]["ll_src.ll"]
    for (u, w), (t, ch) in pw.items():
        assert ch == w % p.P_CH
    runs = len(c.runs)

    def bound(kind: str, terms: dict, decide: int) -> None:
        assert decide == max(terms.values()), (c.name, kind, decide, terms)
        for k, v in terms.items():
            if v == decide:
                bind[f"{kind}.{k}"] = bind.get(f"{kind}.{k}", 0) + 1
    last = {"log": None, "exp": None}                     # the group's previous pass (acam_done)
    prev = {"log": None, "exp": None}                     # the group's previous decision (the gap rule)
    runEnd = {}
    for r in range(runs):
        for b in range(B):
            j = r * B + b
            u = r * (B - 1) + b - 1
            for w in range(W):
                wave = range(w * n, min((w + 1) * n, d.P_OC))
                if b:
                    e = acc.jobs[(r, "rescale", b, w)]
                    assert w or law[(u, 0)][1] == 0                         # one E6 window per unit
                    terms = dict(la=law[(u, 0)][0] + P - 1 + accumulator.LA_LAT,
                                 l_done=acc.jobs[(r, "log", b - 1, w)]["drain_end"] + accumulator.JOB_DONE_LAT)
                    if last["exp"] is not None:                             # the pass finds the leader ready
                        terms["leader"] = last["exp"] + 1 - accumulator.PRE_EXP
                        terms["gap"] = prev["exp"] + accumulator.EXP_GAP
                    bound("rescale", terms, e["decide"])
                    assert e["load_start"] == e["decide"] + accumulator.PRE_EXP
                    last["exp"], prev["exp"] = e["acam_done"], e["decide"]
                    if w == W - 1:
                        assert cr["acc.la"][u] == e["decide"] + P - 1 + accumulator.LA_CR_LAT
                e = acc.jobs[(r, "log", b, w)]
                terms = dict(data=max(pw[(j, q)][0] + P - 1 for q in wave) + accumulator.FIFO_LAT)
                if b:
                    terms["e_done"] = acc.jobs[(r, "rescale", b, w)]["drain_end"] + accumulator.JOB_DONE_LAT
                elif r:
                    terms["restart"] = runEnd[r - 1] + accumulator.RESTART_LAT
                if last["log"] is not None:
                    terms["leader"] = last["log"] + 1 - accumulator.PRE_LOG
                    terms["gap"] = prev["log"] + accumulator.LOG_GAP
                bound("log", terms, e["decide"])
                assert e["load_start"] == e["decide"] + accumulator.PRE_LOG
                last["log"], prev["log"] = e["acam_done"], e["decide"]
                if w == W - 1:
                    assert cr["acc.p"][j] == e["decide"] + 2 * P - 1 + accumulator.P_CR_LAT
        for w in range(W):
            e = acc.jobs[(r, "epilogue", B - 1, w)]
            terms = dict(ll=llw[(r, 0)][0] + P - 1 + accumulator.LL_LAT,
                         l_done=acc.jobs[(r, "log", B - 1, w)]["drain_end"] + accumulator.JOB_DONE_LAT)
            if last["exp"] is not None:
                terms["leader"] = last["exp"] + 1 - accumulator.PRE_EXP
                terms["gap"] = prev["exp"] + accumulator.EXP_GAP
            bound("epilogue", terms, e["decide"])
            assert e["load_start"] == e["decide"] + accumulator.PRE_EXP
            last["exp"], prev["exp"] = e["acam_done"], e["decide"]
            if w == W - 1:
                assert cr["acc.ll"][r] == e["decide"] + P - 1 + accumulator.LL_CR_LAT
        runEnd[r] = e["drain_start"] + accumulator.E9_LAT + P - 1           # the last E9 plane
        assert e["drain_end"] == e["drain_start"] + P - 1
        for w in range(W):                                                  # E9: one window per epilogue wave
            t0 = acc.jobs[(r, "epilogue", B - 1, w)]["drain_start"] + accumulator.E9_LAT
            assert ev["win"]["acc.o"][(r, w)] == (t0, 0), (c.name, r, w)
        assert acc.runDoneAt[r] == runEnd[r] + accumulator.RUN_DONE_LAT
    assert len(ev["win"]["acc.o"]) == runs * W                              # no other E9 window
    # the input credits bind at the sources: a unit u >= CAP starts at its credit
    for key, src, cap in (("acc.p", pw, 2), ("acc.la", law, 2), ("acc.ll", llw, 1)):
        for (u, w), (t, _) in src.items():
            if w == 0 and u >= cap and t == cr[key][u - cap]:
                bind[f"credit.{key}"] = bind.get(f"credit.{key}", 0) + 1
    assert rep.cycles == runEnd[runs - 1] + 1, (rep.cycles, runEnd)        # the last E9 plane + 1


def rulesCases(rng) -> list:
    p = dict(S=32, D=16, R=64, C=64, N_OC=3, NBLK=3, SO=5, LN=5, P_CH=4)
    return [randomCase("r_base", p, 2, rng),
            randomCase("r_slow", p, 2, rng, pPeriod=60, pAvailableAt=[300 * j for j in range(6)]),
            randomCase("r_la", p, 2, rng, laAvailableAt=[1500 * u for u in range(4)]),
            randomCase("r_ll", p, 3, rng, llAvailableAt=[0, 6000, 9000]),
            randomCase("r_one", dict(p, NBLK=1), 3, rng),
            randomCase("r_big", dict(S=64, D=64, R=64, C=64, N_OC=8, NBLK=2, SO=6, LN=6, P_CH=8), 2, rng),
            randomCase("r_wide", dict(S=64, D=32, R=128, C=128, N_OC=5, NBLK=3, SO=6, LN=6, P_CH=8), 2, rng,
                       laPeriod=4)]


def testRules(rng) -> dict:
    bind = {}
    for c in rulesCases(rng):
        checkRules(c, bind)
    want = ("log.data", "log.e_done", "log.restart", "log.gap", "rescale.la", "rescale.l_done", "rescale.gap",
            "epilogue.ll", "epilogue.l_done", "credit.acc.p", "credit.acc.la", "credit.acc.ll")
    for k in want:
        assert bind.get(k, 0) > 0, f"rule term {k} never binds ({bind})"
    return bind


def testDomain() -> int:
    n = 0
    for (nseq, dModel, dHead) in SHAPES:
        for (R, C) in CONFIGS:
            I = min(R, C)
            S = min(nseq, I)
            p = dict(S=S, D=dHead, R=R, C=C, N_OC=nOcFloor(S, dHead, R, C), NBLK=nseq // BKV,
                     SO=nseq.bit_length() - 1, LN=nseq.bit_length() - 1, P_CH=2 * dHead // BKV)
            SoftmaxOnlineAcc.checkDomain(SoftmaxAccParams(**p))
            n += 1
    base = dict(S=16, D=16, R=64, C=64, N_OC=2, NBLK=2, SO=4, LN=4, P_CH=2)
    for bad in (dict(S=12), dict(S=4), dict(S=128), dict(D=128), dict(D=2, S=8, P_CH=1, N_OC=1), dict(N_OC=5),
                dict(N_OC=3), dict(P_CH=3), dict(P_CH=8), dict(P=4), dict(VWP=17), dict(NBLK=0), dict(SO=-1),
                dict(N_OC=0)):
        if inDomain(dict(base, **bad)):
            raise AssertionError(f"checkDomain must reject {bad}")
    return n


def testTimingAndOrder(rng) -> None:
    for c in (randomCase("t_a", dict(S=32, D=16, R=64, C=64, N_OC=3, NBLK=3, SO=5, LN=5, P_CH=4), 2, rng,
                         laAvailableAt=[400 * u for u in range(4)]),
              randomCase("t_b", dict(S=16, D=16, R=64, C=64, N_OC=2, NBLK=2, SO=4, LN=4, P_CH=2), 3, rng)):
        a = run(build(c, withValues=True)[0])
        b = run(build(c, withValues=False)[0])
        assert a.events == b.events and a.cycles == b.cycles and a.counts["acc"] == b.counts["acc"]
        for seed in range(4):
            r = run(build(c, withValues=True)[0], order="random", seed=seed)
            assert r.events == a.events and r.cycles == a.cycles


MUTANT_LATS = ("FIFO_LAT", "LA_LAT", "LL_LAT", "PRE_LOG", "PRE_EXP", "LOG_GAP", "EXP_GAP", "JOB_DONE_LAT", "P_CR_LAT",
               "LA_CR_LAT", "LL_CR_LAT", "RESTART_LAT", "E9_LAT", "RUN_DONE_LAT")


def signature(c: Case) -> tuple:
    g, acc, _ = build(c, withValues=False)
    rep = run(g)
    return tuple(rep.events), tuple(sorted(acc.runDoneAt.items()))


def testMutants(rng) -> tuple:
    p = dict(S=32, D=16, R=64, C=64, N_OC=3, NBLK=3, SO=5, LN=5, P_CH=4)
    cases = [randomCase("m_base", p, 2, rng),
             randomCase("m_late", p, 2, rng, laAvailableAt=[1500 * u for u in range(4)], llAvailableAt=[0, 9000]),
             randomCase("m_slow", p, 1, rng, pPeriod=60, pAvailableAt=[300 * j for j in range(3)])]
    ref = {c.name: signature(c) for c in cases}
    caught, total = 0, 0
    for name in MUTANT_LATS:
        old = getattr(accumulator, name)
        setattr(accumulator, name, old + 1)
        try:
            caught += any(signature(c) != ref[c.name] for c in cases)
        finally:
            setattr(accumulator, name, old)
        total += 1
    # a dropped alpha credit: alpha unit 2 is never sent -> the rescale of block 3 never decides
    g, acc, _ = build(cases[0], withValues=False)
    original = acc.stepBlock

    def noAlphaCredit(unit, io):
        real = io.giveCredit
        io.giveCredit = lambda port, *a, **kw: None if port == "la" else real(port, *a, **kw)
        return original(unit, io)
    acc.stepBlock = noAlphaCredit
    try:
        run(g)
    except Deadlock:
        caught += 1
    total += 1
    # a value mutant: the oq clamp off by one -> GATE 1 fails
    accumulator.OQ_MAX = 126
    try:
        runCase(randomCase("m_val", dict(S=16, D=16, R=64, C=64, N_OC=2, NBLK=2, SO=4, LN=4, P_CH=2), 1, rng,
                           cls="extremes"))
    except RuntimeError as e:
        caught += "GATE 1" in str(e)
    finally:
        accumulator.OQ_MAX = 127
    total += 1
    assert caught == total, f"{caught}/{total} mutants caught"
    return caught, total


def testSoAtW1() -> int:
    """so_mem is a memory of W words x I bits for W >= 2 and no memory at W = 1, where the one word is a register
    (spec §10 item 11, 2026-10-23: Parmys maps no depth-1 memory; found by the DSE sweep on JetFormer-150P at 256x256
    and 512x256). Boundaries: W = 1 at 64x64 (the new GATE-2 case acc_w1), the two DSE builds with W = 1, their
    neighbours with W = 2, and W = 8; no instance lists a depth-1 memory; lo_mem / e_mem are unchanged."""
    n = 0
    for (S, D, R, C, nOc, pch), W in (((16, 16, 64, 64, 4, 4), 1), ((16, 16, 64, 64, 2, 2), 2),
                                       ((32, 32, 256, 256, 4, 4), 1), ((32, 32, 512, 256, 4, 4), 1),
                                       ((32, 32, 256, 128, 4, 4), 2), ((64, 64, 64, 64, 8, 8), 8)):
        acc = SoftmaxOnlineAcc("acc", SoftmaxAccParams(S=S, D=D, R=R, C=C, N_OC=nOc, NBLK=3, SO=4, LN=4, P_CH=pch),
                               runs=1, withValues=False)
        I, mems, shared = min(R, C), acc.resources.memories, acc.resources.memShared
        assert acc.d.W == W, (S, D, R, C, nOc, acc.d.W, W)
        assert mems[(acc.d.MDEPTH, I)] == 2 * nOc and shared[(acc.d.MDEPTH, I)] == nOc, mems
        if W == 1:
            assert (1, I) not in mems and (1, I) not in shared, (mems, shared)
        else:
            assert mems[(W, I)] == nOc and shared[(W, I)] == nOc, (mems, shared)
        assert all(depth >= 2 for depth, _ in mems), mems
        n += 1
    return n


def main() -> int:
    rng = np.random.default_rng(20261014)
    print(f"  gate1    PASS  {testGate1(rng)} cases (every stage and every sent E9 window == oracle, every run)")
    print(f"  closed   PASS  {testClosed()} schedules == spec §5's job rules by hand (every job, p_cr, la_cr, ll_cr, "
          "run end)")
    b = testRules(rng)
    print(f"  rules    PASS  every event recomputed from spec §5; terms bound: {dict(sorted(b.items()))}")
    print(f"  domain   PASS  {testDomain()} DSE accumulator instances inside the domain; out-of-domain sets rejected")
    testTimingAndOrder(rng)
    print("  timing   PASS  (values off: same events, counts; 4 random step orders: same Report)")
    c, t = testMutants(rng)
    print(f"  mutants  PASS  {c}/{t} caught")
    print(f"  so@W=1   PASS  {testSoAtW1()} instances: no so memory at W = 1 (a register), W words otherwise; no "
          "depth-1 memory")
    print("accumulator model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
