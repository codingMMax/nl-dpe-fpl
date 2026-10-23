"""engine.py — the dependency-driven engine (docs/simulator.md §4.6).

The engine flattens a graph, checks its wiring against the port disciplines, then runs any ready step of any process
until every process has finished; if none is ready before that, it raises `Deadlock` with what each waiting process
needs. A step reads only finished data, so the result does not depend on the pick order (tested with random orders).

Port disciplines (architecture.md §3.2), checked on every edge:
  push+credit  unit u >= CAP may start only after credit u - CAP: its windows are not earlier than that credit
  burst        the same rule with CAP = 1: the credit is x_rdy (MSB_SA_Ready) rising, at the cycle a burst may start
  push         no credit
  pulse        a one-bit wire with one pulse per unit, no windows: it carries credits a controller takes or gives
and on every port: windows leave each channel in increasing (unit, window) order with no overlap (W1, W3); credits
come back in unit order.

Edges are the RTL's wires. A data edge carries a producer's windows to one or more consumers (fan-out), channel c to
the consumer's channel c + offset. The credits a port gives (a credited input, or a pulse output) go to the port that
takes them (a credited output, or a pulse input): by default a credited consumer credits its only producer, and a
credit edge routes them elsewhere (to a controller, as att_ctrl takes the DIMMs' staging credits). A push producer may
feed a credited consumer whose credits are routed so. A pulse output may stay open (its pulses are dropped, not
logged). A credit returned on a one-to-one edge of one discipline, where the units on both sides are the same, must
come after the last word of its unit.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from simulator.core.errors import Deadlock, EdgeError
from simulator.core.graph import Graph
from simulator.core.module import Module, Need
from simulator.core.records import CREDITED, Credit, PortEvent, PortSpec, Report, Window
from simulator.core.trace import Time, enabled, label


def traced(time, where: str):
    """The time as a Time; a plain int (a model that lost the chain) becomes an untraced origin, which the
    critical-path breakdown reports. With the trace off, the time as given."""
    if isinstance(time, Time) or not enabled():
        return time
    return Time(int(time), origin=where, kind="untraced")


def givesCredits(s: PortSpec) -> bool:
    """A credited input, or a pulse output."""
    return (s.direction == "in" and s.discipline in CREDITED) or (s.direction == "out" and s.discipline == "pulse")


def takesCredits(s: PortSpec) -> bool:
    """A credited output, or a pulse input."""
    return (s.direction == "out" and s.discipline in CREDITED) or (s.direction == "in" and s.discipline == "pulse")


class StepIO:
    """What a running step sees: its input windows (taken from the queues), the credits it waited for, and the
    functions that emit windows and credits."""

    def __init__(self, engine: "Engine", module: Module, need: Need):
        self._engine, self._module = engine, module
        key = module.instance
        self._windows = {port: {ch: [engine.queues[(key, port)][ch].pop(0) for _ in range(n)]
                                for ch, n in channels.items()} for port, channels in need.windows.items()}
        self._credits = {port: (None if i is None else engine.creditsIn[(key, port)][i])
                         for port, i in need.credits.items()}

    def windows(self, port: str) -> list:
        """The step's windows of `port`, ordered by (unit, window)."""
        return sorted((w for ws in self._windows.get(port, {}).values() for w in ws),
                      key=lambda w: (w.unit, w.window))

    def creditTime(self, port: str) -> int | None:
        c = self._credits.get(port)
        return None if c is None else c.time

    def emit(self, port: str, channel: int, unit: int, window: int, time: int, values=None) -> Window:
        keep = self._module.withValues or self._engine.spec[f"{self._module.instance}.{port}"].control
        w = Window(port, channel, unit, window, traced(time, f"{self._module.instance}.{port}"), values if keep else None)
        self._engine.deliver(self._module, w)
        return w

    def giveCredit(self, port: str, unit: int, time: int) -> Credit:
        c = Credit(port, unit, traced(time, f"{self._module.instance}.{port}"))
        self._engine.returnCredit(self._module, c)
        return c


class Engine:
    def __init__(self, graph: Graph, order: str = "fixed", seed: int = 0):
        self.leaves, dataEdges, creditEdges, self.external = graph.flatten()
        self.modules = {m.instance: m for m in self.leaves}
        if len(self.modules) != len(self.leaves):
            raise ValueError("duplicate instance names after flattening")
        self.spec = self._portSpecs()                                  # "instance.port" -> PortSpec
        self.route, feeder = self._routeData(dataEdges)                # producer -> [(consumer, offset)]
        self.externalOutputs = self._externalOutputs()                 # leaf port -> external name
        self._checkConnected(feeder)
        self.creditTaker, self.sameUnits = self._routeCredits(creditEdges, dataEdges, feeder)
        self.queues, self.creditsIn = self._initQueues()
        self.lastWindow = {}                 # "instance.port" -> {channel: (unit, window, last word)}
        self.unitEnd = {}                    # consumer "instance.port" -> {unit: its last word}
        self.creditsGiven = {}               # giver "instance.port" -> credits given so far
        self.events, self.outputs = [], {leaf: [] for leaf in self.external.values()}
        self.order, self.rng = order, random.Random(seed)
        self.steps = {m.instance: [] for m in self.leaves}
        self.progress = {}

    # -- wiring -----------------------------------------------------------------------------------------------------
    def _portSpecs(self) -> dict:
        spec = {}
        for m in self.leaves:
            for port, s in m.ports().items():
                if s.name != port:
                    raise ValueError(f"{m.instance}: port key {port} != spec name {s.name}")
                spec[f"{m.instance}.{port}"] = s
        return spec

    def _routeData(self, edges: list) -> tuple:
        """Every consumer channel is fed by exactly one producer; the shapes of both ends agree."""
        route, feeder = {}, {}                                         # (consumer, channel) -> producer
        for a, b, offset in edges:
            sa, sb = self.spec[a], self.spec[b]
            if sa.direction != "out" or sb.direction != "in" or "pulse" in (sa.discipline, sb.discipline):
                raise EdgeError(f"edge {a} -> {b}: must connect a data output to a data input")
            for k in ("E", "Q", "W", "planes"):
                if getattr(sa, k) != getattr(sb, k):
                    raise EdgeError(f"edge {a} -> {b}: {k} {getattr(sa, k)} != {getattr(sb, k)}")
            if offset < 0 or offset + sa.L > sb.L:
                raise EdgeError(f"edge {a} -> {b}: channels {offset} .. {offset + sa.L - 1} outside 0 .. {sb.L - 1}")
            for c in range(sa.L):
                if (b, offset + c) in feeder:
                    raise EdgeError(f"edge {a} -> {b}: channel {offset + c} of {b} is fed twice")
                feeder[(b, offset + c)] = a
            route.setdefault(a, []).append((b, offset))
        return route, feeder

    def _externalOutputs(self) -> dict:
        out = {}
        for name, leaf in self.external.items():
            s = self.spec[leaf]
            if s.direction != "out" or s.discipline != "push":
                raise EdgeError(f"external port {name}: only push outputs can leave a top-level run")
            out[leaf] = name
        return out

    def _checkConnected(self, feeder: dict) -> None:
        """Every data port is wired (pulse inputs are checked with the credit routes; pulse outputs may stay open)."""
        for key, s in self.spec.items():
            if s.discipline == "pulse":
                continue
            unconnected = (key not in self.route and key not in self.externalOutputs) if s.direction == "out" \
                else any((key, c) not in feeder for c in range(s.L))
            if unconnected:
                raise EdgeError(f"port {key} is not connected")

    def _routeCredits(self, creditEdges: list, dataEdges: list, feeder: dict) -> tuple:
        """The port that takes the credits each port gives: the explicit credit edges, then the default (a credited
        consumer credits its only producer, when both have the same discipline and the producer has no fan-out)."""
        taker, explicit = {}, set()
        for a, b in creditEdges:
            if not (givesCredits(self.spec[a]) and takesCredits(self.spec[b])):
                raise EdgeError(f"credit edge {a} -> {b}: {a} must give credits (a credited input or a pulse output) "
                                f"and {b} take them (a credited output or a pulse input)")
            if a in taker:
                raise EdgeError(f"credit edge {a} -> {b}: {a} already gives its credits to {taker[a]}")
            taker[a] = b
            explicit.add(a)
        sameUnits = set()                    # consumers whose units are their only producer's units
        for b, sb in self.spec.items():
            if sb.direction != "in" or sb.discipline not in CREDITED:
                continue
            producers = {feeder[(b, c)] for c in range(sb.L)}
            a = producers.pop() if len(producers) == 1 else None
            if a is not None and self.spec[a].discipline == sb.discipline:
                sameUnits.add(b)
            if b not in taker:
                if b not in sameUnits or len(self.route[a]) != 1:
                    raise EdgeError(f"port {b}: its credits need a credit edge (several producers, fan-out, or a "
                                    f"producer of another discipline)")
                taker[b] = a
        for a, b, _ in dataEdges:
            da, db = self.spec[a].discipline, self.spec[b].discipline
            if da != db and not (da == "push" and db in CREDITED and b in explicit):
                raise EdgeError(f"edge {a} -> {b}: discipline {da} != {db}")
        takers = Counter(taker.values())
        for key, s in self.spec.items():
            if takesCredits(s) and takers[key] != 1:
                raise EdgeError(f"port {key} takes credits from {takers[key]} ports (it needs exactly one)")
        return taker, sameUnits

    def _initQueues(self) -> tuple:
        queues, creditsIn = {}, {}
        for key, s in self.spec.items():
            instance, port = key.rsplit(".", 1)
            if s.direction == "in" and s.discipline != "pulse":
                queues[(instance, port)] = {ch: [] for ch in range(s.L)}
            if takesCredits(s):
                creditsIn[(instance, port)] = []
        return queues, creditsIn

    # -- delivery and the protocol checks ---------------------------------------------------------------------------
    def deliver(self, module: Module, w: Window) -> None:
        key = f"{module.instance}.{w.port}"
        s = self.spec.get(key)
        if s is None or s.direction != "out" or s.discipline == "pulse":
            raise EdgeError(f"{module.instance} emits on {w.port}, not a data output port")
        if not 0 <= w.channel < s.L:
            raise EdgeError(f"{key}: channel {w.channel} outside 0..{s.L - 1}")
        previous = self.lastWindow.setdefault(key, {}).get(w.channel)
        if previous is not None and ((w.unit, w.window) <= previous[:2] or w.time <= previous[2]):
            raise EdgeError(f"{key} ch{w.channel}: window (u{w.unit}, w{w.window}) at {w.time} breaks order or "
                            f"overlaps the previous window (u{previous[0]}, w{previous[1]}, last word {previous[2]})")
        with label("stream"):                                    # the window's words on the port
            end = w.time + s.words - 1
        self.lastWindow[key][w.channel] = (w.unit, w.window, end)
        if s.discipline in CREDITED and w.unit >= s.CAP:
            got = self.creditsIn[(module.instance, w.port)]
            i = w.unit - s.CAP
            if i >= len(got) or w.time < got[i].time:
                raise EdgeError(f"{key}: unit {w.unit} at {w.time} sent without credit {i}"
                                + ("" if i >= len(got) else f" (credit at {got[i].time})"))
        self.events.append(PortEvent(w.time, module.instance, w.port, "win", w.unit, w.channel, w.window))
        if key in self.externalOutputs:
            self.outputs[key].append(w)
            return
        for consumer, offset in self.route[key]:                       # every consumer, on its channel c + offset
            instance, port = consumer.rsplit(".", 1)
            self.queues[(instance, port)][w.channel + offset].append(w if offset == 0 else
                                                                      replace(w, channel=w.channel + offset))
            ends = self.unitEnd.setdefault(consumer, {})
            ends[w.unit] = max(ends.get(w.unit, end), end)

    def returnCredit(self, module: Module, c: Credit) -> None:
        key = f"{module.instance}.{c.port}"
        s = self.spec.get(key)
        if s is None or not givesCredits(s):
            raise EdgeError(f"{module.instance} returns a credit on {c.port}, not a credited input or a pulse output")
        n = self.creditsGiven.get(key, 0)
        if c.unit != n:
            raise EdgeError(f"{key}: credit for unit {c.unit}, expected unit {n} (credits come back in order)")
        if key in self.sameUnits:                      # the unit's words came from its producer: after the last one
            last = self.unitEnd.get(key, {}).get(c.unit)
            if last is None or c.time <= last:
                raise EdgeError(f"{key}: credit for unit {c.unit} at {c.time} is not after its last word ({last})")
        self.creditsGiven[key] = n + 1
        taker = self.creditTaker.get(key)
        if taker is None:                              # an open pulse output
            return
        self.events.append(PortEvent(c.time, module.instance, c.port, "cr", c.unit))
        instance, port = taker.rsplit(".", 1)
        self.creditsIn[(instance, port)].append(Credit(port, c.unit, c.time))

    # -- the loop ---------------------------------------------------------------------------------------------------
    def _missing(self, module: Module, need: Need) -> list:
        """What `need` still waits for (empty = ready)."""
        missing = []
        for port, channels in need.windows.items():
            queue = self.queues[(module.instance, port)]
            for ch, n in channels.items():
                if len(queue[ch]) < n:
                    missing.append(f"{port} ch{ch}: {len(queue[ch])}/{n} windows")
        for port, i in need.credits.items():
            if i is not None and len(self.creditsIn[(module.instance, port)]) <= i:
                missing.append(f"{port}: credit {i}")
        for process, n in need.after.items():
            done = self.progress[(module.instance, process)]
            if done < n:
                missing.append(f"process {process}: {done}/{n} steps")
        return missing

    def run(self) -> Report:
        self.progress = {(m.instance, p): 0 for m in self.leaves for p in m.processes()}
        total = {(m.instance, p): n for m in self.leaves for p, n in m.processes().items()}
        while True:
            todo = [key for key, done in self.progress.items() if done < total[key]]
            if not todo:
                break
            if self.order == "random":
                self.rng.shuffle(todo)
            ran = False
            for instance, process in todo:
                module, unit = self.modules[instance], self.progress[(instance, process)]
                need = module.need(process, unit)
                if self._missing(module, need):
                    continue
                with label("logic", role=module.role):          # the step's segments carry its module's role
                    st = module.step(process, unit, StepIO(self, module, need))
                if st.process != process or st.unit != unit:
                    raise RuntimeError(f"{instance}.{process}: step returned ({st.process}, {st.unit}), "
                                       f"expected ({process}, {unit})")
                self.steps[instance].append(st)
                self.progress[(instance, process)] = unit + 1
                ran = True
                if self.order == "random":
                    break
            if not ran:
                lines = [f"  {i}.{p} step {self.progress[(i, p)]}: " + "; ".join(
                    self._missing(self.modules[i], self.modules[i].need(p, self.progress[(i, p)]))) for i, p in todo]
                raise Deadlock("no process can step:\n" + "\n".join(lines))
        left = {f"{i}.{p}": sum(map(len, q.values())) for (i, p), q in self.queues.items() if any(q.values())}
        if left:
            raise RuntimeError(f"windows never consumed: {left}")
        ends = [end for channels in self.lastWindow.values() for (_, _, end) in channels.values()]
        with label("stream", role="sink"):                     # cycles = the last output word's cycle + 1
            cycles = (max(ends) + 1) if ends else 0
        return Report(cycles=cycles,
                      steps={i: sorted(s, key=lambda x: (x.process, x.unit)) for i, s in self.steps.items()},
                      events=sorted(self.events, key=lambda e: (e.time, e.text)),
                      counts={m.instance: m.counts for m in self.leaves},
                      resources={m.instance: m.resources for m in self.leaves},
                      basis={m.instance: m.BASIS for m in self.leaves},
                      roles={m.instance: m.role for m in self.leaves},
                      outputs={self.externalOutputs[k]: v for k, v in self.outputs.items()})


def run(graph: Graph, order: str = "fixed", seed: int = 0) -> Report:
    return Engine(graph, order=order, seed=seed).run()


def portWords(graph: Graph) -> dict:
    """Words per window of every data port, by its event-line name `instance.port` (to read RTL event lines)."""
    leaves = graph.flatten()[0]
    return {f"{m.instance}.{p}": s.words for m in leaves for p, s in m.ports().items() if s.discipline != "pulse"}
