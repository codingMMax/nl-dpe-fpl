"""trace.py — traced cycle counts (docs/simulator.md §3.1, "traced times").

A `Time` is an int that remembers how it was reached: a chain of (label, cycles) segments from an origin. `max`,
`min` and sorting compare values and keep the chosen object, so the winner's chain survives a decision; `t + n`
appends a segment labeled by the current context; segments of one label merge. The time of a run's last event thus
carries the critical path that set it, and the segments' cycles add up to that time exactly.

  Time(v, origin)          an origin (a source's availability, the start): one segment ((origin, "origin"), v)
  t + n, t - n             a Time with the segment (label, n) appended (n an int); a Time minus a Time is a plain int
  label(kind, role=None)   context manager: segments added inside get the label (role, kind); the engine sets the
                           role of every step (its module's job in the workload); nested: the innermost label
  segments(t)              [((role, kind), cycles)] from the origin to t; a plain int gives one "untraced" segment
  breakdown(t, since=0)    (role, kind) -> cycles, the origin's counted from `since`
  enable(on)               on (the default): every run's times carry their critical path, at about three times the
                           run time (a head: 47 -> 140 ms); off: `t + n` is a plain int and a run costs what it did
                           before the trace; segments() and breakdown() then refuse

Kinds: origin, dpe_load (a pass's input burst), dpe_compute (the bit-slices, the shift-accumulate, the ACAM, the
hand-overs), dpe_drain (the first output word and the drain words a consumer takes), stream (the words of a port, a
FIFO read one word per cycle), logic (every other named latency), untraced (a plain int where a Time was expected).
A Time compares, hashes, prints, formats and serializes (json, pickle) as its int; values never change.
"""
from __future__ import annotations

from contextlib import contextmanager

KINDS = ("origin", "dpe_load", "dpe_compute", "dpe_drain", "stream", "logic", "untraced")
_STACK = [("none", "logic")]            # the label context: (role, kind), the innermost last
_ON = [True]


def enable(on: bool = True) -> None:
    """Chains on (the default) or off (plain ints: the run time of the simulator before the trace)."""
    _ON[0] = bool(on)


def enabled() -> bool:
    return _ON[0]


class _Node:
    __slots__ = ("parent", "label", "cycles")

    def __init__(self, parent, label, cycles):
        self.parent, self.label, self.cycles = parent, label, cycles


class Time(int):
    """An int cycle count with the chain of segments that reached it."""

    def __new__(cls, value, origin: str | None = None, kind: str = "origin", node=None):
        self = int.__new__(cls, value)
        self.node = node if node is not None else _Node(None, (origin or "origin", kind), int(value))
        return self

    def _extend(self, n: int) -> "Time":
        if not _ON[0]:
            return int(self) + n
        if n == 0:
            return Time(int(self), node=self.node)
        lab = _STACK[-1]
        node = self.node
        if node.label == lab and node.parent is not None:         # the same label: one segment (never the origin's)
            node = _Node(node.parent, lab, node.cycles + n)
        else:
            node = _Node(node, lab, n)
        return Time(int(self) + n, node=node)

    def __add__(self, n):
        return self._extend(int(n))

    __radd__ = __add__

    def __sub__(self, n):
        if isinstance(n, Time):
            return int(self) - int(n)                              # a duration
        return self._extend(-int(n))

    def __rsub__(self, n):
        return int(n) - int(self)

    def __reduce__(self):                                          # pickles as its int
        return (int, (int(self),))


def current() -> tuple:
    """The label segments get now."""
    return _STACK[-1]


@contextmanager
def label(kind: str, role: str | None = None):
    if kind not in KINDS:
        raise ValueError(f"unknown trace kind {kind!r} (one of {KINDS})")
    _STACK.append((_STACK[-1][0] if role is None else role, kind))
    try:
        yield
    finally:
        _STACK.pop()


def segments(t) -> list:
    """[(label, cycles)] from the origin to t."""
    if not _ON[0]:
        raise RuntimeError("the trace is off: trace.enable() before the run")
    if not isinstance(t, Time):
        return [(("untraced", "untraced"), int(t))]
    out, node = [], t.node
    while node is not None:
        out.append((node.label, node.cycles))
        node = node.parent
    out.reverse()
    return out


def breakdown(t, since: int = 0) -> dict:
    """label -> cycles, summed over the chain; the origin segment counted from `since` (the run's start)."""
    out = {}
    for i, (lab, cycles) in enumerate(segments(t)):
        if i == 0:
            cycles -= int(since)
        out[lab] = out.get(lab, 0) + cycles
    return out
