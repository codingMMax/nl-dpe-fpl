#!/usr/bin/env python3
"""testCore.py — the simulator core: records, stages, engine (docs/simulator.md §3, §4.3, §4.6).

Synthetic graphs with hand-computed answers: a source -> relay -> sink chain on push + credit edges (CAP 2 and 1),
whose every window and credit time is worked out here; random step orders (determinism); a graph used as a module
(composition == flattened); deadlock, discipline and order errors; the stages at their edges; the model stack loads
no oracle, v2 kernel or pricing module.
Run: python3 v3/simulator/tests/testCore.py

The chain: 5 units of one 2-word window. Source: start = max(previous end + 1, credit u-2 from the relay); relay:
send = max(arrival + 3, previous send end + 1, sink credit u-1 + 1), input credit at its send; sink: credit at last
word + 1.
    unit   source start   relay send   relay credit   sink credit
     0          0              3             3              5
     1          2              6             6              8
     2          4 (cr 0 = 3)   9             9             11
     3          6 (cr 1 = 6)  12            12             14
     4          9 (cr 2 = 9)  15            15             17      -> cycles = 16 + 1 = 17
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
from simulator.core.cycleMath import readyAt  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.environment import Sink, Source  # noqa: E402
from simulator.core.errors import ContractError, Deadlock, EdgeError  # noqa: E402
from simulator.core.graph import Graph  # noqa: E402
from simulator.core.module import Module, Need  # noqa: E402
from simulator.core.records import PortSpec, Step  # noqa: E402
from simulator.core.stages import Buffer, Delay, Fifo, Map, Memory, Repack, Serial, Tree  # noqa: E402

RELAY_CR_LAT = 1


class Relay(Module):
    """Test module: unit u leaves k cycles after it arrived, after its out-credit (+ RELAY_CR_LAT) and after the
    previous output; it returns the input credit when it sends (its last read)."""
    NAME = "relay"
    STEP_METHODS = {"units": "stepUnit"}

    def __init__(self, instance, k, capIn, capOut, units, W):
        super().__init__(instance)
        self.k, self.capIn, self.capOut, self.units, self.W = k, capIn, capOut, units, W
        self.lastEnd = -1

    def ports(self):
        return {"i": PortSpec("i", "in", "push+credit", 8, 5, self.W, 1, "u", CAP=self.capIn),
                "o": PortSpec("o", "out", "push+credit", 8, 5, self.W, 1, "u", CAP=self.capOut)}

    def processes(self):
        return {"units": self.units}

    def need(self, process, unit):
        c = self.capOut
        return Need(windows={"i": {0: 1}}, credits={"o": (unit - c) if unit >= c else None})

    def stepUnit(self, unit, io):
        (w,) = io.windows("i")
        words = self.ports()["o"].words
        credit = io.creditTime("o")
        t = max(w.time + self.k, self.lastEnd + 1, -1 if credit is None else credit + RELAY_CR_LAT)
        io.emit("o", 0, unit, 0, t, w.values)
        self.lastEnd = t + words - 1
        io.giveCredit("i", unit, t)
        return Step("units", unit, t, t, t + words)


def chain(units=5, sinkCredit=True):
    portIn = PortSpec("i", "in", "push+credit", 8, 5, 10, 1, "u", CAP=2)
    src = Source("x", replace(portIn, name="out", direction="out"), units, payload=lambda u, w: np.full(10, u, dtype=np.int8))
    relay = Relay("r", k=3, capIn=2, capOut=1, units=units, W=10)
    sink = Sink("y", PortSpec("in", "in", "push+credit", 8, 5, 10, 1, "u", CAP=1), units)
    if not sinkCredit:
        sink.stepUnit = noCreditStep(sink)
    return Graph("chain", [src, relay, sink], [("x.out", "r.i"), ("r.o", "y.in")]), src, relay, sink


def noCreditStep(sink):
    def stepUnit(unit, io):
        io.giveCredit = lambda *a, **kw: None
        return Sink.stepUnit(sink, unit, io)
    return stepUnit


def testChain() -> None:
    g, src, relay, sink = chain()
    rep = run(g)
    # hand-computed (module docstring): source starts 0, 2, 4, 6, 9; relay sends 3, 6, 9, 12, 15
    assert [s.tStart for s in rep.steps["x"]] == [0, 2, 4, 6, 9], rep.steps["x"]
    assert [s.tStart for s in rep.steps["r"]] == [3, 6, 9, 12, 15], rep.steps["r"]
    assert rep.cycles == 17
    assert [w.time for w in sink.received] == [3, 6, 9, 12, 15]
    assert all(np.array_equal(w.values, np.full(10, w.unit)) for w in sink.received)
    credits = [e for e in rep.eventLines() if " cr " in e]
    assert credits == ["3 r.i cr u0", "5 y.in cr u0", "6 r.i cr u1", "8 y.in cr u1", "9 r.i cr u2", "11 y.in cr u2",
                       "12 r.i cr u3", "14 y.in cr u3", "15 r.i cr u4", "17 y.in cr u4"], credits
    for seed in range(8):                                             # determinism
        r2 = run(chain()[0], order="random", seed=seed)
        assert r2.events == rep.events and r2.cycles == rep.cycles
        assert {k: [(s.tStart, s.tDone) for s in v] for k, v in r2.steps.items()} == \
               {k: [(s.tStart, s.tDone) for s in v] for k, v in rep.steps.items()}


def testComposition() -> None:
    portIn = PortSpec("i", "in", "push+credit", 8, 5, 10, 1, "u", CAP=2)

    def parts():
        src = Source("x", replace(portIn, name="out", direction="out"), 4)
        sink = Sink("y", PortSpec("in", "in", "push+credit", 8, 5, 10, 1, "u", CAP=1), 4)
        return src, sink
    src, sink = parts()
    sub = Graph("sub", [Relay("r", k=3, capIn=2, capOut=1, units=4, W=10)], [], ports={"i": "r.i", "o": "r.o"})
    nested = run(Graph("top", [src, sub, sink], [("x.out", "sub.i"), ("sub.o", "y.in")]))
    src, sink = parts()
    flat = run(Graph("top", [src, Relay("sub.r", k=3, capIn=2, capOut=1, units=4, W=10), sink],
                     [("x.out", "sub.r.i"), ("sub.r.o", "y.in")]))
    assert nested.events == flat.events and nested.cycles == flat.cycles
    assert sorted(nested.steps) == sorted(flat.steps) == ["sub.r", "x", "y"]


class Taker(Module):
    """Test module: takes a burst window at t, gives its ready credit at t + k, sends the values on at t + 1 (push)."""
    NAME = "taker"
    STEP_METHODS = {"units": "stepUnit"}

    def __init__(self, instance, k, units, W):
        super().__init__(instance)
        self.k, self.units, self.W = k, units, W

    def ports(self):
        return {"i": PortSpec("i", "in", "burst", 8, 5, self.W, 1, "u", CAP=1),
                "o": PortSpec("o", "out", "push", 8, 5, self.W, 1, "u")}

    def processes(self):
        return {"units": self.units}

    def need(self, process, unit):
        return Need(windows={"i": {0: 1}})

    def stepUnit(self, unit, io):
        (w,) = io.windows("i")
        io.giveCredit("i", unit, w.time + self.k)
        io.emit("o", 0, unit, 0, w.time + 1, w.values)
        return Step("units", unit, w.time, w.time, w.time + 1)


class Gate(Module):
    """Test controller: watches the source's bursts (a fan-out consumer), takes the takers' ready pulses and the
    joiner's credits on pulse ports, and gives the source's credit k at max(ready_a(k), ready_b(k), joiner credit
    k - 1 + 1)."""
    NAME = "gate"
    STEP_METHODS = {"x": "stepBurst"}

    def __init__(self, instance, units, W):
        super().__init__(instance)
        self.units, self.W = units, W

    def ports(self):
        return {"x": PortSpec("x", "in", "burst", 8, 5, self.W, 1, "u", CAP=1),
                **{n: PortSpec(n, "in", "pulse", 1, 1, 1, 1, "u") for n in ("ra", "rb", "cz")}}

    def processes(self):
        return {"x": self.units}

    def need(self, process, unit):
        last = unit == self.units - 1
        return Need(windows={"x": {0: 1}}, credits={"ra": None if last else unit, "rb": None if last else unit,
                                                     "cz": None if (last or unit == 0) else unit - 1})

    def stepBurst(self, unit, io):
        (w,) = io.windows("x")
        if unit < self.units - 1:
            cz = io.creditTime("cz")
            io.giveCredit("x", unit, readyAt(io.creditTime("ra"), io.creditTime("rb"), None if cz is None else cz + 1))
        return Step("x", unit, w.time, w.time, w.time + 1)


class Joiner(Module):
    """Test module: unit k = one window on each of its 2 channels; its credit at the later one + 2, a pulse at + 3."""
    NAME = "joiner"
    STEP_METHODS = {"units": "stepUnit"}

    def __init__(self, instance, units, W):
        super().__init__(instance)
        self.units, self.W = units, W

    def ports(self):
        return {"i": PortSpec("i", "in", "push+credit", 8, 5, self.W, 2, "u", CAP=2),
                "ev": PortSpec("ev", "out", "pulse", 1, 1, 1, 1, "u")}

    def processes(self):
        return {"units": self.units}

    def need(self, process, unit):
        return Need(windows={"i": Need.eachChannel(2)})

    def stepUnit(self, unit, io):
        ws = io.windows("i")
        if sorted(w.channel for w in ws) != [0, 1]:
            raise RuntimeError("the windows must land on the consumer's channels")
        t = max(w.time for w in ws)
        io.giveCredit("i", unit, t + 2)
        io.giveCredit("ev", unit, t + 3)
        return Step("units", unit, t, t, t + 1)


FAN_EDGES = [("s.x", "a.i"), ("s.x", "b.i"), ("s.x", "g.x"), ("a.o", "z.i", 0), ("b.o", "z.i", 1)]
FAN_CREDITS = [("a.i", "g.ra"), ("b.i", "g.rb"), ("g.x", "s.x"), ("z.i", "g.cz")]


def fan(units=4, edges=None, credits=None):
    """The fan graph (docstring of testFan); `edges` / `credits` replace the default lists."""
    W = 5
    src = Source("s", PortSpec("x", "out", "burst", 8, 5, W, 1, "u", CAP=1), units,
                 payload=lambda u, w: np.full(W, u, dtype=np.int8))
    mods = [src, Taker("a", k=3, units=units, W=W), Taker("b", k=5, units=units, W=W), Gate("g", units=units, W=W),
            Joiner("z", units=units, W=W)]
    return Graph("fan", mods, FAN_EDGES if edges is None else edges, credits=FAN_CREDITS if credits is None else credits)


def testFan() -> None:
    """Fan-out, channel placement, pulse ports and credit routes, hand-computed. The source s (burst, 1-word windows)
    feeds takers a (ready at t + 3) and b (ready at t + 5) and the gate g; a and b push to channels 0 and 1 of the
    joiner z (credit at its later window + 2), whose credits go to g. g gives s's credit k at max(ra(k), rb(k),
    cz(k - 1) + 1):
        unit   s start   a, b ready   z windows   z credit   s credit (from g)
         0        0        3, 5          1           3          5
         1        5        8, 10         6           8          10 = max(8, 10, 3 + 1)
         2       10       13, 15        11          13          15 = max(13, 15, 8 + 1)
         3       15       18, 20        16          18          (last unit: none)        -> cycles = 16 + 1 = 17
    z's pulse output ev stays open: dropped, not logged."""
    rep = run(fan())
    assert [s.tStart for s in rep.steps["s"]] == [0, 5, 10, 15], rep.steps["s"]
    credits = {}
    for e in rep.events:
        if e.kind == "cr":
            credits.setdefault(f"{e.instance}.{e.port}", []).append(e.time)
    assert credits == {"a.i": [3, 8, 13, 18], "b.i": [5, 10, 15, 20], "g.x": [5, 10, 15], "z.i": [3, 8, 13, 18]}, credits
    assert rep.cycles == 17 and not any(e.instance == "z" and e.port == "ev" for e in rep.events)
    assert sum(1 for e in rep.events if e.kind == "win" and e.instance in ("a", "b") and e.port == "o") == 8
    for seed in range(8):
        r2 = run(fan(), order="random", seed=seed)
        assert r2.events == rep.events and r2.cycles == rep.cycles


def testFanErrors() -> None:
    drop = lambda x: [c for c in FAN_CREDITS if c != x]  # noqa: E731
    # a fan-out consumer whose credits are not routed; a push producer into a credited port with no route
    expect(EdgeError, lambda: run(fan(credits=drop(("a.i", "g.ra")))), "need a credit edge")
    expect(EdgeError, lambda: run(fan(credits=drop(("z.i", "g.cz")))), "need a credit edge")
    # two givers for one taker (and none for another); a giver routed twice
    swapped = [c if c != ("b.i", "g.rb") else ("b.i", "g.ra") for c in FAN_CREDITS]
    expect(EdgeError, lambda: run(fan(credits=swapped)), "takes credits from")
    expect(EdgeError, lambda: run(fan(credits=FAN_CREDITS + [("a.i", "g.rb")])), "already gives")
    # a credit edge between ports that neither give nor take; a channel fed twice; channels outside the consumer
    expect(EdgeError, lambda: run(fan(credits=FAN_CREDITS + [("a.o", "g.cz")])), "must give credits")
    edges = [e if e != ("b.o", "z.i", 1) else ("b.o", "z.i", 0) for e in FAN_EDGES]
    expect(EdgeError, lambda: run(fan(edges=edges)), "fed twice")
    edges = [e if e != ("b.o", "z.i", 1) else ("b.o", "z.i", 2) for e in FAN_EDGES]
    expect(EdgeError, lambda: run(fan(edges=edges)), "outside")
    # a data edge into a pulse port
    expect(EdgeError, lambda: run(fan(edges=FAN_EDGES + [("a.o", "g.ra")])), "data output to a data input")
    # a credit given on a pulse input
    g = fan()
    gate = g.modules[3]
    original = gate.stepBurst

    def bad(unit, io):
        io.giveCredit("ra", unit, 99)
        return original(unit, io)
    gate.stepBurst = bad
    expect(EdgeError, lambda: run(g), "not a credited input or a pulse output")
    expect(ValueError, lambda: PortSpec("p", "in", "pulse", 1, 1, 1, 1, "u", CAP=1), "no CAP")


def expect(err, fn, text=""):
    try:
        fn()
    except err as e:
        assert text in str(e), f"{err.__name__} without {text!r}: {e}"
        return
    raise AssertionError(f"{err.__name__} expected ({text})")


def testErrors() -> None:
    # a dropped credit: the sink never frees a unit -> the relay waits for its out-credit 0
    expect(Deadlock, lambda: run(chain(sinkCredit=False)[0]), "o: credit 0")
    # a source that ignores credits
    g, src, relay, sink = chain()
    src.need = lambda process, unit: Need()
    expect(EdgeError, lambda: run(g), "without credit")
    # windows out of order on a channel
    g, src, relay, sink = chain(units=2)
    original = src.stepUnit

    def backwards(unit, io):
        emit = io.emit
        io.emit = lambda port, ch, u, w, t, values=None: emit(port, ch, u, w, 100 - 10 * unit, values)
        return original(unit, io)
    src.stepUnit = backwards
    expect(EdgeError, lambda: run(g), "breaks order")
    # a credit before the unit's last word
    g, src, relay, sink = chain(units=1)
    relayStep = relay.stepUnit

    def early(unit, io):
        giveCredit = io.giveCredit
        io.giveCredit = lambda port, u, t: giveCredit(port, u, 1)
        return relayStep(unit, io)
    relay.stepUnit = early
    expect(EdgeError, lambda: run(g), "not after its last word")
    # an unconnected port; mismatched window sizes; a push port with a CAP; a credited port without one
    g, src, relay, sink = chain()
    expect(EdgeError, lambda: run(Graph("g", [src, relay, sink], [("x.out", "r.i")])), "not connected")
    src2 = Source("x", PortSpec("out", "out", "push+credit", 8, 5, 12, 1, "u", CAP=2), 1)
    expect(EdgeError, lambda: run(Graph("g", [src2, Relay("r", k=3, capIn=2, capOut=1, units=1, W=10), sink],
                                        [("x.out", "r.i"), ("r.o", "y.in")])), "W 12 != 10")
    expect(ValueError, lambda: PortSpec("p", "out", "push", 8, 5, 10, 1, "u", CAP=1), "no CAP")
    expect(ValueError, lambda: PortSpec("p", "in", "push+credit", 8, 5, 10, 1, "u"), "CAP")
    expect(ValueError, lambda: PortSpec("p", "in", "push", 8, 5, 0, 1, "u"), ">= 1")
    expect(ValueError, lambda: PortSpec("p", "up", "push", 8, 5, 1, 1, "u"), "direction")
    # windows left unread
    g, src, relay, sink = chain(units=2)
    sink.processes = lambda: {"units": 1}
    expect(RuntimeError, lambda: run(g), "never consumed")
    assert PortSpec("p", "in", "push", 8, 5, 11, 1, "u").words == 3


def testStages() -> None:
    assert Delay(0)(7, 3) == (7, 3) and Delay(4)(None, 10) == (None, 14)
    big = [np.array([127, -128, 2 ** 40]), np.array([127, -128, -1]), np.array([1, 0, 0])]
    s, t = Tree(3)(big, 5)
    assert list(s) == [255, -256, 2 ** 40 - 1] and t == 8
    assert Tree(0)([np.array([1])], 2)[1] == 2
    expect(ValueError, lambda: Tree(1)([], 0), "at least one")
    assert Map(np.abs, 2)(np.array([-3]), 1)[1] == 3
    f = Fifo(2)
    f.write("a", 0)
    f.write("b", 1)
    expect(OverflowError, lambda: f.write("c", 2), "busy")
    assert f.read(0) == ("a", 1)                       # read at max(write + 1, decision)
    expect(OverflowError, lambda: f.write("c", 1), "busy")    # slot of "a" is free only after its read at 1
    f.write("c", 2)
    assert f.read(10) == ("b", 10) and f.read(0) == ("c", 3)
    expect(IndexError, lambda: f.read(0), "empty")
    expect(ValueError, lambda: Fifo(0), "depth")
    g = Fifo(3)
    g.write(1, 5)
    expect(ValueError, lambda: g.write(2, 5), "not after")
    m = Memory(2)
    assert m.write(1, "v", 9) == 10
    expect(ValueError, lambda: m.read(1, 9), "before it is valid")
    assert m.read(1, 10) == ("v", 11)
    expect(ValueError, lambda: m.read(0, 50), "before it is valid")
    expect(IndexError, lambda: m.write(2, "x", 0), "outside")
    # readyAt: the latest condition; None is no condition; none at all is cycle 0
    assert readyAt() == 0 and readyAt(None) == 0 and readyAt(None, 5) == 5 and readyAt(3, 7, None, 2) == 7
    assert readyAt(np.int64(4), 0) == 4
    # Serial: one unit at a time
    s = Serial()
    assert s.take(5) == 5 and s.freeAt == 6
    assert s.take(0) == 6 and s.freeAt == 7                 # ready early: waits for the previous unit
    assert s.take(10, 3) == 10 and s.freeAt == 13           # a unit of 3 cycles
    assert s.take(11, 0) == 13 and s.freeAt == 13           # a zero-length unit does not move the resource
    assert s.take(20, 0) == 20
    # Buffer: rotation over 2 entries
    b = Buffer(2, "stg")
    b.put(0, 5)
    b.put(1, 6)
    expect(ContractError, lambda: b.put(2, 7), "unit 0 has not released")
    b.release(0, 10)
    expect(ContractError, lambda: b.put(2, 9), "free from 10")      # one cycle early
    b.put(2, 10)                                                      # exactly at the free cycle
    expect(ValueError, lambda: b.release(2, 12), "out of order")
    b.release(1, 11)
    b.release(2, 12)
    expect(ContractError, lambda: b.put(3, 10), "free from 11")
    expect(ValueError, lambda: Buffer(0), "entries")
    # Repack: 3 words of 5, 5, 4 bytes -> 3 words; one word per input word
    assert Repack([5, 5, 4], 5, 2).times([0, 1, 2]) == [2, 3, 4]
    # two 9-byte windows (5 + 4 bytes each) -> 4 words: the last input word completes two of them
    r = Repack([5, 4, 5, 4], 5, 2)
    assert r.sourceWord == [0, 2, 3, 3] and r.times([0, 1, 2, 3]) == [2, 4, 5, 6]
    assert Repack([5, 4, 5, 4], 5, 2).times([0, 10, 11, 20]) == [2, 13, 22, 23]   # gaps in the reads
    assert Repack([5, 5], 5, 1).times([3, 4]) == [4, 5]                         # an exact multiple: no padding
    assert Repack([1], 5, 0).times([7]) == [7]                                  # one byte, one padded word
    expect(ValueError, lambda: Repack([], 5, 2), "sizes")
    expect(ValueError, lambda: Repack([5, 0], 5, 2), "sizes")
    expect(ValueError, lambda: Repack([5, 4], 5, 2).times([0]), "read times")


MODEL_STACK = ("simulator.core.engine", "simulator.core.environment", "simulator.core.stages",
               "simulator.core.crossbar", "simulator.models.gemm", "simulator.models.dimmTile",
               "simulator.models.softmaxOnline", "simulator.models.softmaxAccumulator",
               "simulator.models.attentionController", "simulator.models.fc", "simulator.models.attentionHead")


def testIndependence() -> list:
    """The model stack imports no oracle code, no v2 kernel and no pricing (docs/simulator.md §5)."""
    code = (f"import sys; sys.path.insert(0, {str(V3)!r})\n"
            f"import {', '.join(MODEL_STACK)}\n"
            "mods = sorted(m for m in sys.modules if m.endswith('_ref') or m.endswith('_sim') or m in "
            "('pass_engine', 'cost', 'platforms') or m.startswith('simulator.platform') or m.startswith('dse'))\n"
            "print(','.join(mods) or '-')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "-", f"the model stack loads {out}"
    return list(MODEL_STACK)


def main() -> int:
    testChain()
    print("  chain        PASS  (hand-computed windows and credits; 8 random step orders identical)")
    testComposition()
    print("  composition  PASS  (a graph run as a module == the same graph flattened)")
    testErrors()
    print("  errors       PASS  (dropped credit -> deadlock; no-credit send, order, early credit, unconnected, "
          "mismatch, unread windows)")
    testFan()
    print("  fan          PASS  (fan-out, channel placement, pulse ports, credit routes: hand-computed; 8 random "
          "orders identical)")
    testFanErrors()
    print("  fan errors   PASS  (unrouted fan-out credits, a push producer into an unrouted credited port, two givers, "
          "a double route, a wrong credit edge, a doubly fed channel, channels outside, data into a pulse port)")
    testStages()
    print("  stages       PASS  (readyAt, Delay, Tree, Map, Serial, Buffer, Repack, Fifo, Memory at their edges)")
    mods = testIndependence()
    print(f"  independence PASS  ({', '.join(mods)} load no oracle, v2 kernel or pricing module)")
    print("core: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
