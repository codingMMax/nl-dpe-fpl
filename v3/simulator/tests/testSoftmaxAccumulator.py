#!/usr/bin/env python3
"""testSoftmaxAccumulator.py — gates of the `softmax_online_acc` model (docs/simulator.md §4.4; ip/softmax/acc_spec.md
§8).

  gate1    GATE 1 on a sweep: S 8..128, D 8..64, N_OC 1..8 (partial last waves), NBLK 1..4, P_CH 1..8, SO / LN equal and
           unequal, 1-3 runs; crossbars 64x64, 128x64, 128x128; classes random / zeros / allneg / extremes / small /
           alpha0; late and slow inputs
  closed   the jobs fed v2's station inputs (v2 `feasible_schedule`) == v2 `ocarry_cycle_model`: every LOG and EXP
           job's decide, load_start, msb_fire, drain_start, drain_end; op_free == p_cr, la_free == la_cr; the run end
           (v2's O-feed hold off: v3 has no O reader)
  rules    every job decision, credit, E9 window and run_done recomputed from the spec §5 formulas; each readiness
           term and each input credit binds somewhere
  domain   the DSE's accumulator instances (6 shapes x 12 configs, N_EP = 2d, N_OC by the floor rule) are inside the
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
sys.path.insert(0, str(V3 / "oracle" / "v2_kernels"))    # test references only
import ocarry_sim as v2oc  # noqa: E402

import simulator.models.softmaxAccumulator as accumulator  # noqa: E402
from ip.softmax.acc_cases import Case, build, randomCase, runCase  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import Deadlock  # noqa: E402
from simulator.models.softmaxAccumulator import SoftmaxAccParams, SoftmaxOnlineAcc  # noqa: E402

from dse.space import CONFIGS, SHAPES as NAMED_SHAPES  # noqa: E402  (the DSE's design space, defined once)
SHAPES = [shape[1:] for shape in NAMED_SHAPES]                       # (n, d_model, d)
BKV = 16


def inDomain(p: dict) -> bool:
    try:
        SoftmaxOnlineAcc.checkDomain(SoftmaxAccParams(**p))
        return True
    except ValueError:
        return False


def nOcFloor(S: int, D: int, R: int, C: int, bkv: int = BKV) -> int:
    """architecture.md §5.2: the smallest N_OC with T(ceil(P_oc / N_OC)) <= max(ceil(S Bkv / 5), T(1))."""
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


def testClosed() -> int:
    n = 0
    keys = ("decide", "load_start", "msb_fire", "drain_start", "drain_end")
    for (S, d, R, C, nOc, B, ngp) in ((16, 16, 64, 64, 2, 3, 2), (32, 8, 64, 64, 4, 4, 4), (16, 16, 64, 64, 2, 3, 4),
                                      (128, 64, 256, 256, 7, 8, 8), (64, 64, 64, 64, 8, 2, 8),
                                      (16, 16, 64, 64, 2, 1, 2), (32, 32, 64, 64, 3, 2, 4)):
        g = v2oc.geometry(S, d, R, C, nOc, B, ngp=ngp)
        for lead, early, feed in ((200, 0, 0), (0, 0, 0), (-150, 0, 0), (0, 400, 0), (0, 0, 3000)):
            schedule = v2oc.feasible_schedule(g, 3, 10, op_pace=g.LOAD * 3, la_lead=lead, ll_lead=50, early=early,
                                              feed=feed)
            acc = SoftmaxOnlineAcc("acc", SoftmaxAccParams(S=S, D=d, R=R, C=C, N_OC=nOc, NBLK=B, SO=4, LN=4, P_CH=ngp),
                                   runs=3, withValues=False)
            credits = {}
            for r, ri in enumerate(schedule):
                for b in range(B):
                    credits[(r, b)] = acc.block(r * B + b, ri.op_ready[b], ri.la_go[b] if b else None)
                acc.epilogue(r, ri.ll_go)
                ri.o_fed = acc.runEndAt[r]                              # v2's O-feed hold off: no O reader in v3
            want = v2oc.ocarry_cycle_model(g, schedule)
            for r in range(3):
                for (b, w, ev) in want.log[r]:
                    got = acc.jobs[(r, "log", b, w)]
                    assert all(got[k] == ev[k] for k in keys), (S, d, r, "log", b, w, got, ev)
                for (kind, b, w, ev) in want.exp[r]:
                    got = acc.jobs[(r, kind, b, w)]
                    assert all(got[k] == ev[k] for k in keys), (S, d, r, kind, b, w, got, ev)
                for b in range(B):
                    assert credits[(r, b)]["p_cr"] == want.op_free[r][b]
                    if b:
                        assert credits[(r, b)]["la_cr"] == want.la_free[r][b]
                assert acc.runEndAt[r] == want.o8[r][-1][-1]
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
    S, B, W, LW, n = p.S, p.NBLK, d.W, d.LW, p.N_OC
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
    last = {"log": None, "exp": None}                     # the group's previous pass (msb_fire)
    runEnd = {}
    for r in range(runs):
        for b in range(B):
            j = r * B + b
            u = r * (B - 1) + b - 1
            for w in range(W):
                wave = range(w * n, min((w + 1) * n, d.P_OC))
                if b:
                    e = acc.jobs[(r, "rescale", b, w)]
                    terms = dict(la=max(law[(u, m)][0] for m in range(S)) + accumulator.LA_LAT,
                                 l_done=acc.jobs[(r, "log", b - 1, w)]["drain_end"] + accumulator.JOB_DONE_LAT)
                    if last["exp"] is not None:
                        terms["leader"] = last["exp"] + 1
                    bound("rescale", terms, e["decide"])
                    assert e["load_start"] == e["decide"] + accumulator.PRE
                    last["exp"] = e["msb_fire"]
                    if w == W - 1:
                        assert cr["acc.la"][u] == e["decide"] + LW - 1 + accumulator.LA_CR_LAT
                e = acc.jobs[(r, "log", b, w)]
                terms = dict(data=max(pw[(j, q)][0] + LW - 1 for q in wave) + accumulator.FIFO_LAT)
                if b:
                    terms["e_done"] = acc.jobs[(r, "rescale", b, w)]["drain_end"] + accumulator.JOB_DONE_LAT
                elif r:
                    terms["restart"] = runEnd[r - 1] + accumulator.RESTART_LAT
                if last["log"] is not None:
                    terms["leader"] = last["log"] + 1
                bound("log", terms, e["decide"])
                assert e["load_start"] == e["decide"] + accumulator.PRE
                last["log"] = e["msb_fire"]
                if w == W - 1:
                    assert cr["acc.p"][j] == e["decide"] + LW - 1 + accumulator.P_CR_LAT
        for w in range(W):
            e = acc.jobs[(r, "epilogue", B - 1, w)]
            terms = dict(ll=llw[(r, 0)][0] + S - 1 + accumulator.LL_LAT,
                         l_done=acc.jobs[(r, "log", B - 1, w)]["drain_end"] + accumulator.JOB_DONE_LAT)
            if last["exp"] is not None:
                terms["leader"] = last["exp"] + 1
            bound("epilogue", terms, e["decide"])
            last["exp"] = e["msb_fire"]
            if w == W - 1:
                assert cr["acc.ll"][r] == e["decide"] + LW - 1 + accumulator.LL_CR_LAT
        runEnd[r] = e["drain_end"] + accumulator.O8_LAT
        assert e["drain_end"] == e["drain_start"] + LW - 1                  # the last window's last E9 word is at t_o8
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
    assert rep.cycles == runEnd[runs - 1] + 1, (rep.cycles, runEnd)        # the last E9 word + 1


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
    want = ("log.data", "log.e_done", "log.restart", "log.leader", "rescale.la", "rescale.l_done", "epilogue.ll",
            "epilogue.l_done", "credit.acc.p", "credit.acc.la", "credit.acc.ll")
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
                dict(N_OC=3), dict(P_CH=3), dict(P_CH=8), dict(BUF=32), dict(VWP=33), dict(NBLK=0), dict(SO=-1),
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


MUTANT_LATS = ("FIFO_LAT", "LA_LAT", "LL_LAT", "PRE", "JOB_DONE_LAT", "P_CR_LAT", "LA_CR_LAT", "LL_CR_LAT", "O8_LAT",
               "RESTART_LAT", "E9_LAT", "RUN_DONE_LAT")


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


def main() -> int:
    rng = np.random.default_rng(20261014)
    print(f"  gate1    PASS  {testGate1(rng)} cases (every stage and every sent E9 window == oracle, every run)")
    print(f"  closed   PASS  {testClosed()} schedules == v2 ocarry_cycle_model (every job, op_free, la_free, run end)")
    b = testRules(rng)
    print(f"  rules    PASS  every event recomputed from spec §5; terms bound: {dict(sorted(b.items()))}")
    print(f"  domain   PASS  {testDomain()} DSE accumulator instances inside the domain; out-of-domain sets rejected")
    testTimingAndOrder(rng)
    print("  timing   PASS  (values off: same events, counts; 4 random step orders: same Report)")
    c, t = testMutants(rng)
    print(f"  mutants  PASS  {c}/{t} caught")
    print("accumulator model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
