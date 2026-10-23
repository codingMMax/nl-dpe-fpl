#!/usr/bin/env python3
"""testSoftmaxOnline.py — gates of the `softmax_online_top` model (docs/simulator.md §4.4; ip/softmax/spec.md §8), the
bit-plane pass (2026-10-22).

  gate1    GATE 1 on a sweep: S 8..128, N = S .. 8 S, BKV 8 / 16 / 32, N_EXP 1 / 2 / 4, S_CH 1 / 2 / 4, 1-3 runs,
           classes random / zeros / extremes / small / rising; crossbars 64x64, 128x64, 256x128; late tiles and
           credit holds
  closed   the core fed a stream schedule (word 0 at 2, back to back, block b >= 2 at ret(b - 2) + 1) == spec §5's
           recurrence written here by hand: every block's EXP bursts, sp_done, wait, carry passes and retire, and the
           total (until 2026-10-22 the witness was v2's `softmax_online_cycle_model`, which times the 40-bit pass)
  rules    every tile decision, plane read, corner-turn word, E5 / E6 / E7 window, carry pass, s_cr, done and start
           credit recomputed from the spec §5 formulas; each gate term (idle, arm, data, slot, lp, la) and the ll
           credit bind somewhere
  domain   the DSE's softmax instances (6 shapes x 8 configs x S_CH {1, 2}) are inside the guard; out-of-domain sets
           are rejected
  timing   values off: identical events and counts; random step orders: identical Report
  mutants  every named latency + 1 (MAX_REG through FEED_LAT and E6: m_new itself never binds the carry, as
           sp_done > a_last + MAX_LAT, asserted in rules), a dropped credit (deadlock), a value mutant (GATE 1): each
           fails a check
Run: python3 v3/simulator/tests/testSoftmaxOnline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))

import simulator.models.softmaxOnline as softmax  # noqa: E402
from ip.softmax.cases import Case, build, randomCase, runCase  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.errors import Deadlock  # noqa: E402
from simulator.models.softmaxOnline import SoftmaxOnline, SoftmaxOnlineParams  # noqa: E402

from dse.space import CONFIGS, SHAPES as NAMED_SHAPES  # noqa: E402  (the DSE's design space, defined once)
SHAPES = [shape[1:] for shape in NAMED_SHAPES]                       # (n, d_model, d)
P = 8                                                                # plane words per window (spec §2)
PASS = P + 2                                                         # a crossbar's pass period (ip/xbar/spec.md)


def inDomain(p: dict) -> bool:
    try:
        SoftmaxOnline.checkDomain(SoftmaxOnlineParams(**p))
        return True
    except ValueError:
        return False


def gate1Cases(rng) -> list:
    out = []
    for (R, C) in ((64, 64), (128, 64), (256, 128)):
        for S in (8, 16, 32, 64, 128):
            for mult in (1, 2, 8):
                for bkv in (8, 16, 32):
                    for ne, sch in ((1, 1), (2, 1), (2, 2), (4, 2), (4, 4)):
                        p = dict(S=S, BKV=bkv, N=S * mult, R=R, C=C, N_EXP=ne, S_CH=sch)
                        if not inDomain(p) or (bkv != 16 and (mult != 2 or ne != 2)):
                            continue
                        cls = ("random", "zeros", "extremes", "small", "rising")[len(out) % 5]
                        runs = 1 + len(out) % 3
                        out.append(randomCase(f"g1_{R}x{C}_S{S}_N{S * mult}_B{bkv}_e{ne}_c{sch}_{cls}", p, runs,
                                              rng, cls=cls))
    # arrival patterns and credit holds (values on)
    p = dict(S=32, BKV=16, N=128, R=64, C=64, N_EXP=2, S_CH=2)
    for kw in (dict(sAvailableAt=[400 * j for j in range(16)]), dict(sPeriod=40), dict(lpCreditLatency=500),
               dict(laCreditLatency=700), dict(llCreditLatency=3000), dict(goAvailableAt=[0, 2500])):
        out.append(randomCase("g1_arr_" + "_".join(kw), p, 2, rng, **kw))
    return out


def testGate1(rng) -> int:
    cases = gate1Cases(rng)
    for c in cases:
        runCase(c)                        # raises on a GATE 1 mismatch
    return len(cases)


def handCore(p: dict, firstWord) -> list:
    """Spec §5's core by hand for one run, the block's first stream word at firstWord(b, ret) (ret: the blocks so
    far). MAX_LAT = log2 min(I/8, BKV) + 2, FEED_LAT = MAX_LAT + P + 2; a pass drains from load_start + P + 2, its
    last plane P - 1 later, and the next pass on the crossbar loads P + 2 after this one."""
    I = min(p["R"], p["C"])
    NWB, B, nExp = p["S"] * p["BKV"] // I, p["N"] // p["BKV"], p["N_EXP"]
    maxLat = (min(I // 8, p["BKV"]).bit_length() - 1) + 2
    feedLat = maxLat + P + 2
    lane, carryExp, carryLog = {}, None, None       # the last load_start of each crossbar
    blocks, ret = [], []
    for b in range(B):
        first = firstWord(b, ret)
        aLast = first + P * NWB - 1
        loads = []
        for w in range(NWB):
            g = (b * NWB + w) % nExp
            ls = max(aLast + feedLat, lane[g] + PASS) if g in lane else aLast + feedLat
            lane[g] = ls
            loads.append(ls)
        spDone = max(ls + PASS + P - 1 for ls in loads) + 3                  # last drain plane + SP_LAT
        wait = max(spDone, aLast + maxLat, ret[b - 1] if b else 0) + 1       # + WAIT_LAT
        cexpLast = None
        if b == 0:
            logReady = wait + 2                                              # + GO_LAT
        else:
            carryExp = max(wait + 2, carryExp + PASS) if carryExp is not None else wait + 2
            cexpLast = carryExp + PASS + P - 1
            logReady = cexpLast + 1 + 2                                      # + E_LAT + GO_LAT
        carryLog = max(logReady, carryLog + PASS) if carryLog is not None else logReady
        ret.append(carryLog + PASS + P - 1)
        blocks.append(dict(first=first, a_last=aLast, loads=loads, sp_done=spDone, wait=wait, cexp_last=cexpLast,
                           retire=ret[-1], la=aLast + maxLat + 1))
    return blocks


def streamSchedule(b: int, ret: list, previous: dict) -> int:
    """The closed test's stream: block 0 at 2, then back to back, block b >= 2 not before ret(b - 2) + 1."""
    first = 2 if b == 0 else previous["last"] + 1
    if b >= 2:
        first = max(first, ret[b - 2] + 1)
    return first


def testClosed() -> int:
    n = 0
    for (R, C) in ((64, 64), (128, 128), (256, 256), (128, 64), (512, 256)):
        for S in (8, 16, 32, 64, 128, 256):
            for N in (S, 4 * S, 16):
                for bkv in (8, 16, 32):
                    for ne in (1, 2, 4):
                        p = dict(S=S, BKV=bkv, N=N, R=R, C=C, N_EXP=ne, S_CH=1)
                        if not inDomain(p) or N > 1024:
                            continue
                        state = {}

                        def firstWord(b, ret):
                            first = streamSchedule(b, ret, state)
                            state["last"] = first + P * (S * bkv // min(R, C)) - 1
                            return first
                        want = handCore(p, firstWord)
                        sm = SoftmaxOnline("sm", SoftmaxOnlineParams(**p), runs=1, withValues=False)
                        for b, wb in enumerate(want):
                            o = sm.core(b, wb["a_last"])
                            got = dict(loads=[e["load_start"] for e in o["passes"]], sp_done=o["sp_done"],
                                       wait=o["wait"], cexp_last=o["cexp_last"], retire=o["retire"])
                            assert [t for (_, _, t, _) in o["lp"]] == got["loads"]
                            assert got == {k: wb[k] for k in got}, (p, b, got, wb)
                        assert sm.retireAt[len(want) - 1] + 1 == want[-1]["retire"] + 1
                        n += 1
    return n


def eventTables(rep) -> dict:
    """{key: [(t, ch, unit, w)] windows} and {key: [(t, unit)] credits by unit} from the report's events."""
    windows, credits = {}, {}
    for e in rep.events:
        key = f"{e.instance}.{e.port}"
        if e.kind == "win":
            windows.setdefault(key, []).append((e.time, e.channel, e.unit, e.window))
        else:
            credits.setdefault(key, []).append((e.time, e.unit))
    for v in credits.values():
        v.sort(key=lambda x: x[1])
    return dict(win=windows, cr=credits)


def checkRules(c: Case, bind: dict) -> None:
    """Every event of a case recomputed from spec §5; `bind` collects which gate terms bound."""
    g, sm, sinks = build(c, withValues=False)
    rep = run(g)
    p, d = sm.p, sm.d
    ev = eventTables(rep)
    S, B = p.S, d.B
    maxLat = (min(d.I // 8, p.BKV).bit_length() - 1) + 2           # spec §5, by hand
    feedLat = maxLat + P + 2
    assert (d.MAX_LAT, d.FEED_LAT) == (maxLat, feedLat)
    cr = {k: [t for (t, _) in v] for k, v in ev["cr"].items()}
    arrival = {}
    for (t, ch, u, w) in ev["win"]["s_src.s"]:
        arrival[(u, w)] = t
        assert ch == w % p.S_CH
    go = [t for (t, _, _, _) in sorted(ev["win"]["go_src.go"], key=lambda x: x[2])]
    steps = {(s.process, s.unit): s for s in rep.steps["sm"]}
    readerFree = 0
    lpAt = {(u, w): (t, ch) for (t, ch, u, w) in ev["win"].get("sm.lp", [])}
    laAt = {u: t for (t, ch, u, w) in ev["win"].get("sm.la", [])}
    assert all(w == 0 for (_, _, _, w) in ev["win"].get("sm.la", []))      # one E6 window per block b >= 1
    llAt = {u: t for (t, ch, u, w) in ev["win"].get("sm.ll", [])}
    lastAcam = {}                                   # crossbar -> acam_done of its last pass
    for j in range(len(c.scores) * B):
        r, b = divmod(j, B)
        st = steps[("tile", j)].events
        terms = dict(idle=readerFree, arm=go[r] + softmax.ARM_LAT, data=arrival[(j, 0)] + softmax.FIFO_LAT)
        if b >= 2:
            terms["slot"] = steps[("blk", j - 2)].events["retire"] + softmax.RET_LAT
        if j >= p.LP_CAP:
            terms["lp"] = cr["lp_snk.lp"][j - p.LP_CAP] + softmax.CR_IN_LAT
        u = sm.alphaUnit(j)
        if u is not None and u >= p.LA_CAP:
            terms["la"] = cr["la_snk.la"][u - p.LA_CAP] + softmax.CR_IN_LAT
        tDecide = max(terms.values())
        assert st["t_dec"] == tDecide, (c.name, j, st["t_dec"], terms)
        for k, v in terms.items():
            if v == tDecide:
                bind[k] = bind.get(k, 0) + 1
        reads = sm.readTimes[j]
        t = tDecide + softmax.OPEN_LAT - 1
        for i in range(d.NWORDS):
            w, cc = divmod(i, P)
            t = max(t + 1, arrival[(j, w)] + cc + softmax.FIFO_LAT)
            assert reads[i] == t, (c.name, j, i)
        readerFree = int(reads[-1]) + 1
        assert cr["sm.s"][j] == reads[-1] + softmax.SCR_LAT
        stream = sm.streamTimes[j]                  # the corner-turn: window w's P words from its last plane + 2
        assert list(stream) == [int(reads[w * P + P - 1]) + softmax.CT_LAT + cc for w in range(d.NWB)
                                for cc in range(P)], (c.name, j)
        # the core: E5 windows at their pass's load_start, the pass placed by the pass rule
        be = steps[("blk", j)].events
        lastWord = int(stream[-1])
        assert be["a_last"] == lastWord
        for w, ps in enumerate(be["passes"]):
            lane = (b * d.NWB + w) % p.N_EXP
            want = max(lastWord + feedLat, lastAcam[lane] + 1 if lane in lastAcam else 0)
            assert ps["load_start"] == want and lpAt[(j, w)] == (want, lane), (c.name, j, w)
            lastAcam[lane] = ps["acam_done"]
        sumsDone = max(ps["drain_start"] for ps in be["passes"]) + P - 1 + softmax.SP_LAT
        assert be["sp_done"] == sumsDone and sumsDone > lastWord + maxLat
        previousRetire = steps[("blk", j - 1)].events["retire"] if b else 0
        assert be["wait"] == max(sumsDone, lastWord + maxLat, previousRetire) + softmax.WAIT_LAT
        if b == 0:
            logReady = be["wait"] + softmax.GO_LAT
            assert be["carry_exp"] is None
        else:
            ce = be["carry_exp"]
            assert ce["load_start"] == max(be["wait"] + softmax.GO_LAT, lastAcam["cexp"] + 1 if "cexp" in lastAcam
                                           else 0), (c.name, j)
            lastAcam["cexp"] = ce["acam_done"]
            assert be["cexp_last"] == ce["drain_start"] + P - 1
            logReady = be["cexp_last"] + softmax.E_LAT + softmax.GO_LAT
        cl = be["carry_log"]
        assert cl["load_start"] == max(logReady, lastAcam["clog"] + 1 if "clog" in lastAcam else 0), (c.name, j)
        lastAcam["clog"] = cl["acam_done"]
        assert be["retire"] == cl["drain_start"] + P - 1
        if u is not None:
            assert laAt[u] == lastWord + maxLat + softmax.LA_LAT, (c.name, j)
    for r in range(len(c.scores)):
        rs = steps[("run", r)].events
        coreDone = steps[("blk", (r + 1) * B - 1)].events["retire"] + 1
        terms = dict(core=coreDone)
        if r >= p.LL_CAP:
            terms["ll"] = cr["ll_snk.ll"][r - p.LL_CAP] + softmax.CR_IN_LAT
        tD7 = max(terms.values())
        if terms.get("ll") == tD7:
            bind["ll"] = bind.get("ll", 0) + 1
        assert rs["core_done"] == coreDone and rs["t_d7"] == tD7 and llAt[r] == tD7 + softmax.E7_LAT
        assert rs["done"] == tD7 + softmax.E7_LAT + P - 1 + softmax.DONE_LAT
        assert cr["sm.go"][r] == rs["done"] + softmax.GO_CR_LAT
        if r:
            assert go[r] >= cr["sm.go"][r - 1]


def rulesCases(rng) -> list:
    p = dict(S=32, BKV=16, N=128, R=64, C=64, N_EXP=2, S_CH=1)
    return [randomCase("r_base", p, 2, rng),
            randomCase("r_late", dict(p, S_CH=2), 2, rng, sAvailableAt=[900 * j for j in range(16)]),
            randomCase("r_lp", p, 2, rng, lpCreditLatency=600),
            randomCase("r_la", p, 2, rng, laCreditLatency=800),
            randomCase("r_ll", dict(p, N=32), 3, rng, llCreditLatency=3000),
            randomCase("r_go", p, 2, rng, goAvailableAt=[50, 9000]),
            randomCase("r_slot", dict(S=8, BKV=16, N=128, R=64, C=64, N_EXP=2, S_CH=1), 2, rng),
            randomCase("r_wide", dict(S=64, BKV=16, N=256, R=256, C=128, N_EXP=4, S_CH=2), 1, rng, sPeriod=60)]


def testRules(rng) -> dict:
    bind = {}
    for c in rulesCases(rng):
        checkRules(c, bind)
    for k in ("idle", "arm", "data", "slot", "lp", "la", "ll"):
        assert bind.get(k, 0) > 0, f"rule term {k} never binds ({bind})"
    return bind


def testDomain() -> int:
    n = 0
    for (nseq, dModel, dHead) in SHAPES:
        for (R, C) in CONFIGS:
            I = min(R, C)
            for sch in (1, 2):                   # S_CH = N_E / d, N_E in {d, 2d}
                p = dict(S=min(nseq, I), BKV=16, N=nseq, R=R, C=C, N_EXP=2, S_CH=sch)
                SoftmaxOnline.checkDomain(SoftmaxOnlineParams(**p))
                n += 1
    for bad in (dict(S=8, BKV=8, N=64, R=128, C=128), dict(S=16, BKV=16, N=64, S_CH=8),
                dict(S=8, BKV=16, N=16, R=128, C=128, N_EXP=2), dict(S=128, BKV=16, N=128, R=64, C=64),
                dict(S=16, BKV=32, N=16), dict(S=24, BKV=16, N=48), dict(S=16, BKV=16, N=64, P=4),
                dict(S=16, BKV=16, N=64, LP_CAP=0), dict(S=4, BKV=16, N=64)):
        q = dict(dict(R=64, C=64), **bad)
        if inDomain(q):
            raise AssertionError(f"checkDomain must reject {bad}")
    return n


def testTimingAndOrder(rng) -> None:
    for c in (randomCase("t_a", dict(S=32, BKV=16, N=128, R=64, C=64, N_EXP=2, S_CH=2), 2, rng, lpCreditLatency=300,
                         laCreditLatency=200),
              randomCase("t_b", dict(S=8, BKV=16, N=64, R=64, C=64, N_EXP=2, S_CH=1), 3, rng, llCreditLatency=500)):
        a = run(build(c, withValues=True)[0])
        b = run(build(c, withValues=False)[0])
        assert a.events == b.events and a.cycles == b.cycles and a.counts["sm"] == b.counts["sm"]
        for seed in range(4):
            r = run(build(c, withValues=True)[0], order="random", seed=seed)
            assert r.events == a.events and r.cycles == a.cycles


MUTANT_LATS = ("FIFO_LAT", "CR_IN_LAT", "RET_LAT", "ARM_LAT", "OPEN_LAT", "CT_LAT", "SCR_LAT", "MAX_REG", "FEED_REG",
               "SP_LAT", "WAIT_LAT", "GO_LAT", "E_LAT", "LA_LAT", "E7_LAT", "DONE_LAT", "GO_CR_LAT")


def testMutants(rng) -> tuple:
    cases = [randomCase("m_slot", dict(S=8, BKV=16, N=128, R=64, C=64, N_EXP=2, S_CH=1), 2, rng),
             randomCase("m_hold", dict(S=32, BKV=16, N=64, R=64, C=64, N_EXP=2, S_CH=2), 3, rng, lpCreditLatency=400,
                        laCreditLatency=300, llCreditLatency=1500),
             randomCase("m_slow", dict(S=32, BKV=16, N=64, R=64, C=64, N_EXP=2, S_CH=1), 1, rng, sPeriod=30)]
    ref = {c.name: run(build(c, withValues=False)[0]).events for c in cases}
    caught, total = 0, 0
    for name in MUTANT_LATS:
        old = getattr(softmax, name)
        setattr(softmax, name, old + 1)
        try:
            caught += any(run(build(c, withValues=False)[0]).events != ref[c.name] for c in cases)
        finally:
            setattr(softmax, name, old)
        total += 1
    # a dropped credit: the lp sink never frees a block -> tile LP_CAP never opens
    g, sm, sinks = build(cases[0], withValues=False)
    original = sinks["lp"].stepUnit

    def noCredit(unit, io):
        io.giveCredit = lambda *a, **kw: None
        return original(unit, io)
    sinks["lp"].stepUnit = noCredit
    try:
        run(g)
    except Deadlock:
        caught += 1
    total += 1
    # a value mutant: the log P floor off by one -> GATE 1 fails
    softmax.LP_FLOOR = -127
    try:
        runCase(randomCase("m_val", dict(S=16, BKV=16, N=64, R=64, C=64), 1, rng, cls="extremes"))
    except RuntimeError as e:
        caught += "GATE 1" in str(e)
    finally:
        softmax.LP_FLOOR = -128
    total += 1
    assert caught == total, f"{caught}/{total} mutants caught"
    return caught, total


def main() -> int:
    rng = np.random.default_rng(20261014)
    print(f"  gate1    PASS  {testGate1(rng)} cases (every stage and every sent value == oracle, every run)")
    print(f"  closed   PASS  {testClosed()} configurations == spec §5's core recurrence (by hand)")
    b = testRules(rng)
    print(f"  rules    PASS  every event recomputed from spec §5; terms bound: {dict(sorted(b.items()))}")
    print(f"  domain   PASS  {testDomain()} DSE softmax instances inside the domain; out-of-domain sets rejected")
    testTimingAndOrder(rng)
    print("  timing   PASS  (values off: same events, counts; 4 random step orders: same Report)")
    c, t = testMutants(rng)
    print(f"  mutants  PASS  {c}/{t} caught")
    print("softmax model: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
