#!/usr/bin/env python3
"""testTrace.py — gates of the traced times (simulator/core/trace.py; docs/simulator.md §3.1).

  time      Time arithmetic at its edges: an origin, + and - append and merge segments, a Time minus a Time is a
            duration, max / min / sort keep the chosen object and a tie keeps the first, a 0 step adds no segment, a
            plain int is one untraced segment; a Time equals, hashes, formats, json- and pickle-serializes as its int;
            numpy on the left keeps the value; label nesting and the role; an unknown kind is refused
  helpers   readyAt and the stages (Serial, Fifo, Repack, Memory, Buffer) keep the Time they choose
  crossbar  the pass grammar's segments: load / compute / drain of a pass, a burst gated by the previous pass's
            msb_fire, by its accumulator, by its output buffer (hand-computed on 8 x 8 and 8 x 1024)
  engine    an FC row: the breakdown equals the closed form per kind (V = 1 and V = 3); every port event's time is
            traced; a late source is an origin of that many cycles; a module that drops the chain shows as untraced
  head      BERT-Tiny at 128x64 and ViT-B/16 at 128x64 (two runs): the segments add up to T, none untraced or
            negative, the kinds known; the path opens with the first token's load; the same breakdown in 3 random
            step orders
Run: python3 v3/simulator/tests/testTrace.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
import simulator.core.crossbar as xb  # noqa: E402
from simulator.core.cycleMath import ceilDiv, readyAt  # noqa: E402
from simulator.core.engine import run, traced  # noqa: E402
from simulator.core.environment import Sink, Source  # noqa: E402
from simulator.core.graph import Graph  # noqa: E402
from simulator.core.records import PortSpec, Step  # noqa: E402
from simulator.core.stages import Buffer, Fifo, Memory, Repack, Serial  # noqa: E402
import simulator.core.trace as trace  # noqa: E402
from simulator.core.trace import KINDS, Time, breakdown, current, label, segments  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402
from simulator.models.fc import fc  # noqa: E402


def kinds(t, since=0) -> dict:
    out = {}
    for (_, k), c in breakdown(t, since).items():
        out[k] = out.get(k, 0) + c
    return out


def testTime() -> int:
    n = 0
    t = Time(5, "src")
    assert int(t) == 5 and segments(t) == [(("src", "origin"), 5)]; n += 1
    u = t + 3
    assert isinstance(u, Time) and u == 8 and segments(u) == [(("src", "origin"), 5), (("none", "logic"), 3)]; n += 1
    assert segments(u + 2) == [(("src", "origin"), 5), (("none", "logic"), 5)], "same label: one segment"; n += 1
    assert segments(u - 1)[-1] == (("none", "logic"), 2) and (u - 1) == 7; n += 1
    assert segments(u) == [(("src", "origin"), 5), (("none", "logic"), 3)], "nodes are immutable"; n += 1
    assert u - t == 3 and type(u - t) is int and 10 - u == 2 and type(10 - u) is int; n += 1
    assert segments(3 + t)[-1] == (("none", "logic"), 3), "int + Time"; n += 1
    v = t + 0
    assert v == 5 and segments(v) == segments(t), "a 0 step adds no segment"; n += 1
    assert segments(Time(2, "a") - 2)[-1][1] == -2 and segments(Time(2, "a") + 0 - 0) == [(("a", "origin"), 2)]; n += 1
    # max / min / sort keep the object; a tie keeps the first
    a, b = Time(3, "a") + 4, Time(9, "b")
    assert max(a, b) is b and min(a, b) is a and sorted([b, a])[0] is a; n += 1
    c = Time(7, "c")
    assert max(a, c) is a and max(c, a) is c, "a tie keeps the first"; n += 1
    assert readyAt(a, None, b) is b and readyAt(None) == 0 and readyAt(b, a) is b; n += 1
    # a plain int is one untraced segment
    assert segments(7) == [(("untraced", "untraced"), 7)] and breakdown(7) == {("untraced", "untraced"): 7}; n += 1
    assert breakdown(u, since=5) == {("src", "origin"): 0, ("none", "logic"): 3}; n += 1
    # an int in every other respect
    assert u == 8 and hash(u) == hash(8) and {u: 1}[8] == 1 and f"{u:4d}" == "   8" and str(u) == "8" and repr(u) == "8"
    n += 1
    assert json.dumps({"t": u}) == '{"t": 8}' and json.loads(json.dumps([u])) == [8]; n += 1
    p = pickle.loads(pickle.dumps(u))
    assert p == 8 and type(p) is int; n += 1
    assert np.int64(2) + u == 10 and u + np.int64(2) == 10 and isinstance(u + np.int64(2), Time); n += 1
    assert u * 2 == 16 and type(u * 2) is int and u // 3 == 2 and -u == -8; n += 1
    assert bool(Time(0)) is False and list(range(Time(3))) == [0, 1, 2]; n += 1
    # labels: nested, the role inherited or set
    with label("dpe_load", role="gemm"):
        assert current() == ("gemm", "dpe_load")
        w = t + 2
        with label("stream"):
            assert current() == ("gemm", "stream")
            w = w + 3
        with label("logic", role="sink"):
            w = w + 1
        w = w + 4
    assert current() == ("none", "logic") and segments(w) == [(("src", "origin"), 5), (("gemm", "dpe_load"), 2),
                                                               (("gemm", "stream"), 3), (("sink", "logic"), 1),
                                                               (("gemm", "dpe_load"), 4)]; n += 1
    try:
        with label("bogus"):
            pass
    except ValueError:
        n += 1
    else:
        raise AssertionError("an unknown kind was accepted")
    try:
        with label("stream"):
            raise KeyError("x")
    except KeyError:
        assert current() == ("none", "logic"), "the label is popped on an exception"; n += 1
    x = traced(4, "m.p")
    assert isinstance(x, Time) and segments(x) == [(("m.p", "untraced"), 4)] and traced(u, "m.p") is u; n += 1
    # the switch: off, arithmetic gives plain ints, the engine wraps nothing, and a breakdown refuses
    trace.enable(False)
    try:
        assert not trace.enabled() and type(Time(1, "a") + 1) is int and type(5 + Time(1, "a")) is int; n += 1
        assert type(Time(3, "a") - 1) is int and traced(4, "m.p") == 4 and type(traced(4, "m.p")) is int; n += 1
        for fn in (lambda: segments(Time(1, "a")), lambda: breakdown(7)):
            try:
                fn()
            except RuntimeError:
                n += 1
            else:
                raise AssertionError("a breakdown with the trace off was accepted")
    finally:
        trace.enable(True)
    assert trace.enabled() and isinstance(Time(1, "a") + 1, Time); n += 1
    return n


def testHelpers() -> int:
    n = 0
    s = Serial()
    t = s.take(Time(10, "a"), 3)
    assert t is not None and segments(t) == [(("a", "origin"), 10)] and s.freeAt == 13; n += 1
    t2 = s.take(Time(2, "b"), 1)
    assert t2 == 13 and segments(t2)[0] == (("a", "origin"), 10), "the busy resource binds: its chain"; n += 1
    f = Fifo(4)
    f.write(None, Time(5, "w"))
    _, r = f.read(Time(2, "d"))
    assert r == 6 and segments(r)[0] == (("w", "origin"), 5); n += 1
    f.write(None, Time(7, "w"))
    _, r = f.read(Time(20, "d"))
    assert r == 20 and segments(r) == [(("d", "origin"), 20)]; n += 1
    rp = Repack([5, 5, 2], 5, 1)
    times = rp.times([Time(1, "r"), Time(2, "r"), Time(3, "r")])
    assert all(isinstance(x, Time) for x in times) and [int(x) for x in times] == [2, 3, 4]; n += 1
    m = Memory(2)
    v = m.write(1, None, Time(9, "m"))
    assert v == 10 and isinstance(v, Time) and isinstance(m.read(1, Time(12, "x"))[1], Time); n += 1
    bf = Buffer(1)
    bf.release(0, Time(3, "b"))
    assert isinstance(bf.freeFrom[0], Time); n += 1
    return n


def testCrossbar() -> int:
    n = 0
    c = xb.Crossbar(8, 8, withValues=False)                                 # LOAD 2, OUTPUT 2, P 8
    p1 = c.run(None, Time(0, "t"))
    e = p1.events
    assert kinds(e["drain_end"]) == {"origin": 0, "dpe_load": 2, "dpe_compute": 9, "dpe_drain": 2}; n += 1
    assert kinds(e["drain_start"]) == {"origin": 0, "dpe_load": 2, "dpe_compute": 9, "dpe_drain": 1}; n += 1
    assert kinds(e["acam_done"]) == {"origin": 0, "dpe_load": 2, "dpe_compute": 9}; n += 1
    assert kinds(e["msb_fire"]) == {"origin": 0, "dpe_load": 2, "dpe_compute": 7}; n += 1
    assert kinds(p1.nextReadyAt) == {"origin": 0, "dpe_load": 2, "dpe_compute": 8} and p1.nextReadyAt == 10; n += 1
    assert kinds(c.readyAt()) == kinds(p1.nextReadyAt) and kinds(c.accumulatorFreeAt)["dpe_compute"] == 10; n += 1
    assert kinds(c.outputFreeAt) == {"origin": 0, "dpe_load": 2, "dpe_compute": 9, "dpe_drain": 3}; n += 1
    # a burst offered early waits for msb_fire + RDY_LAT: the chain runs through the previous pass
    p2 = c.run(None, Time(0, "t"))
    assert p2.events["load_start"] == 10 and kinds(p2.events["load_start"]) == {"origin": 0, "dpe_load": 2,
                                                                                 "dpe_compute": 8}; n += 1
    assert kinds(p2.events["drain_end"]) == {"origin": 0, "dpe_load": 4, "dpe_compute": 17, "dpe_drain": 2}; n += 1
    # a late burst: its own origin
    p3 = c.run(None, Time(100, "late"))
    assert segments(p3.events["load_start"]) == [(("late", "origin"), 100)]; n += 1
    # an output buffer still draining binds the ACAM: the chain runs through the previous pass's drain
    w = xb.Crossbar(8, 1024, withValues=False)                              # OUTPUT 205
    q1 = w.run(None, Time(0, "t"))
    q2 = w.run(None, Time(0, "t"))
    assert q2.events["acam_done"] == q1.events["drain_end"] + 1 == 217; n += 1
    assert kinds(q2.events["drain_end"]) == {"origin": 0, "dpe_load": 2, "dpe_compute": 9, "dpe_drain": 411}; n += 1
    # an accumulator still busy binds compute_start (a short load, LOAD 1 at R = 1)
    a = xb.Crossbar(1, 8, withValues=False)                                 # LOAD 1
    a.run(None, Time(0, "t"))                                               # acam_done 10, accumulator free 11
    r2 = a.run(None, Time(0, "t"))                                          # load_start 9, 9 + 1 < 11
    assert r2.events["compute_start"] == 11 and kinds(r2.events["compute_start"]) == {"origin": 0, "dpe_load": 1,
                                                                                       "dpe_compute": 10}; n += 1
    return n


def fcKinds(K, N, R, C, availableAt=0) -> tuple:
    g = fc(K, N, R, C, withValues=False, availableAt=availableAt)
    rep = run(g)
    V, H = ceilDiv(K, R), ceilDiv(N, C)
    rows, cols = (K if V == 1 else R), (N if H == 1 else C)
    want = {"origin": availableAt, "dpe_load": ceilDiv(8 * rows, 40), "dpe_compute": 9, "dpe_drain": 1,
            "logic": (max(0, (V - 1).bit_length()) if V > 1 else 0) + 1, "stream": ceilDiv(cols, 5)}
    return rep, kinds(rep.cycles), want


def testEngine() -> int:
    n = 0
    for K, N, R, C in ((512, 128, 512, 128), (300, 40, 128, 64), (64, 64, 64, 64), (2048, 256, 1024, 256)):
        rep, got, want = fcKinds(K, N, R, C)
        assert got == want, (K, N, R, C, got, want)
        assert sum(got.values()) == rep.cycles; n += 1
        assert all(isinstance(e.time, Time) for e in rep.events) and not any(
            k == "untraced" for e in rep.events for (_, k) in breakdown(e.time)), "every port event traced"; n += 1
        roles = {r for (r, _) in breakdown(rep.cycles)}
        assert roles == {"x", "gemm_top", "sink"}, roles; n += 1
    rep, got, want = fcKinds(64, 64, 64, 64, availableAt=37)
    assert got == want and got["origin"] == 37 and kinds(rep.cycles, since=37)["origin"] == 0; n += 1
    trace.enable(False)
    try:
        off = run(fc(300, 40, 128, 64, withValues=False, availableAt=37))
    finally:
        trace.enable(True)
    on = run(fc(300, 40, 128, 64, withValues=False, availableAt=37))
    assert off.cycles == on.cycles and off.eventLines() == on.eventLines() and type(off.cycles) is int; n += 1
    # a module that drops the chain: the engine wraps the int as an untraced origin, the breakdown shows it
    spec = PortSpec("d", "out", "push", 8, 5, 10, 1, "row")
    src = Source("s", spec, 2, 1, availableAt=0, withValues=False)
    snk = Sink("k", PortSpec("d", "in", "push", 8, 5, 10, 1, "row"), 2, 1, withValues=False)
    rep = run(Graph("stream", [src, snk], [("s.d", "k.d")]))
    assert kinds(rep.cycles) == {"origin": 0, "stream": 4}, kinds(rep.cycles)   # 2 windows of 2 words, back to back
    n += 1
    src2 = Source("s", spec, 1, 1, availableAt=0, withValues=False)
    snk2 = Sink("k", PortSpec("d", "in", "push", 8, 5, 10, 1, "row"), 1, 1, withValues=False)

    def emitsInt(unit, io):                                                # a step that lost the chain
        io.emit("d", 0, unit, 0, 5)
        return Step("units", unit, 5, 5, 7)
    src2.stepUnit = emitsInt
    rep = run(Graph("lossy", [src2, snk2], [("s.d", "k.d")]))
    assert breakdown(rep.cycles).get(("s.d", "untraced")) == 5, breakdown(rep.cycles); n += 1
    return n


def headCheck(nseq, dm, d, R, C, orders=3) -> tuple:
    hp = headParams(nseq, dm, d, R, C)
    g, parts = attentionHead(hp, withValues=False)
    rep = run(g)
    ctrl = parts["ctrl"]
    T = ctrl.done - ctrl.tStart
    bd = breakdown(ctrl.done, since=ctrl.tStart)
    segs = segments(ctrl.done)
    assert sum(bd.values()) == T, (sum(bd.values()), T)
    assert all(k in KINDS and k != "untraced" for (_, k) in bd), sorted({k for (_, k) in bd})
    assert all(c >= 0 for _, c in segs), [s for s in segs if s[1] < 0][:3]
    assert all(isinstance(e.time, Time) for e in rep.events)
    for seed in range(orders):
        g2, parts2 = attentionHead(hp, withValues=False)
        run(g2, order="random", seed=seed)
        assert breakdown(parts2["ctrl"].done, since=parts2["ctrl"].tStart) == bd, f"order {seed}"
    return T, bd, segs


def testHead() -> int:
    n = 0
    T, bd, segs = headCheck(128, 128, 64, 128, 64)
    assert T == 7752; n += 1
    assert segs[0] == (("x0", "origin"), 1) and segs[1] == (("gemm", "dpe_load"), 26) and \
        segs[2] == (("gemm", "dpe_compute"), 8), segs[:3]; n += 1
    byKind = {}
    for (_, kind), c in bd.items():
        byKind[kind] = byKind.get(kind, 0) + c
    assert byKind["dpe_load"] >= 128 * 26 and byKind["dpe_compute"] >= 128 * 8, byKind   # every token's projection
    n += 1
    T2, bd2, segs2 = headCheck(256, 768, 64, 128, 64, orders=1)              # two runs, V = 6
    assert T2 == 24395 and sum(bd2.values()) == T2; n += 1
    return n


def main() -> int:
    print(f"  time     PASS  {testTime()} checks (arithmetic, labels, int behavior)")
    print(f"  helpers  PASS  {testHelpers()} checks (readyAt and the stages keep the chosen Time)")
    print(f"  crossbar PASS  {testCrossbar()} checks (load / compute / drain, gated bursts)")
    print(f"  engine   PASS  {testEngine()} checks (FC closed form per kind, traced events, a late source, an int)")
    print(f"  head     PASS  {testHead()} checks (two heads: exact sums, no untraced, 3 random orders)")
    print("testTrace: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
