"""records.py — the simulator's records (docs/simulator.md §3.1–§3.2).

Time is in integer cycles of the design's one clock; cycle 0 is the cycle `start` is high. A window's time is the
cycle of its first word; word c of the window is at time + c (W1, gapless). A credit's time is the cycle the
consumer's pulse is high.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from simulator.core.cycleMath import ceilDiv

DISCIPLINES = ("push+credit", "burst", "push", "pulse")   # architecture.md §3.2; pulse = a one-bit wire, one pulse
                                                          # per unit (a credit, x_rdy, run_done), no windows
CREDITED = ("push+credit", "burst")                       # data ports whose consumer gives one credit per unit


@dataclass(frozen=True)
class Window:
    """Window `window` of unit `unit` on channel `channel` of `port`, first word at cycle `time`; `values` is None in
    timing-only mode."""
    port: str
    channel: int
    unit: int
    window: int
    time: int
    values: np.ndarray | None = None


@dataclass(frozen=True)
class Credit:
    """The consumer freed the buffer of `unit` of `port` at cycle `time` (on a burst port: x_rdy rose at `time`)."""
    port: str
    unit: int
    time: int


@dataclass(frozen=True)
class PortSpec:
    """A port of architecture.md §3: E bits per element, Q slots per word, W elements per window, L channels; CAP is
    the consumer's buffer in units (push+credit), 1 on a burst port, None on a push port; `signals` are the RTL
    signals of the port."""
    name: str
    direction: str            # "in" | "out"
    discipline: str           # DISCIPLINES
    E: int
    Q: int
    W: int
    L: int
    unit: str                 # what one unit is (a token row, a tile, ...)
    CAP: int | None = None
    signals: tuple = ()
    control: bool = False     # the words are control fields (a job's slots), carried even in timing-only mode

    def __post_init__(self):
        if self.direction not in ("in", "out") or self.discipline not in DISCIPLINES:
            raise ValueError(f"port {self.name}: direction {self.direction!r}, discipline {self.discipline!r}")
        if min(self.E, self.Q, self.W, self.L) < 1:
            raise ValueError(f"port {self.name}: E, Q, W, L must be >= 1")
        if self.discipline in CREDITED and (self.CAP is None or self.CAP < 1):
            raise ValueError(f"port {self.name}: {self.discipline} needs CAP >= 1")
        if self.discipline not in CREDITED and self.CAP is not None:
            raise ValueError(f"port {self.name}: a {self.discipline} port has no CAP")

    @property
    def words(self) -> int:
        """Words per window: ceil(W / Q)."""
        return ceilDiv(self.W, self.Q)


@dataclass(frozen=True)
class PortEvent:
    """One port event in the testbench port-monitor format: `<t> <inst.port> win ch<c> u<unit> w<w>` for a window's
    first word, `<t> <inst.port> cr u<unit>` for a credit pulse."""
    time: int
    instance: str
    port: str
    kind: str                 # "win" | "cr"
    unit: int
    channel: int = 0
    window: int = 0

    @property
    def text(self) -> str:
        """The line without its time (the events of one cycle are ordered by it)."""
        key = f"{self.instance}.{self.port}"
        if self.kind == "win":
            return f"{key} win ch{self.channel} u{self.unit} w{self.window}"
        return f"{key} cr u{self.unit}"

    @property
    def line(self) -> str:
        return f"{self.time} {self.text}"


@dataclass
class Step:
    """One job of a process: the times the testbench also logs, plus the module's named events (spec names)."""
    process: str
    unit: int
    tDecide: int
    tStart: int
    tDone: int
    events: dict = field(default_factory=dict)


@dataclass
class Counts:
    """The energy ledger of one module (docs/simulator.md §4.7): crossbar passes per (mode, rows, cols), CLB adds,
    compares and selects, memory bytes read and written, FIFO words."""
    passes: Counter = field(default_factory=Counter)
    clbAdd: int = 0
    clbCompare: int = 0
    clbSelect: int = 0
    memReadBytes: int = 0
    memWriteBytes: int = 0
    fifoWords: int = 0

    def add(self, other: "Counts") -> "Counts":
        self.passes.update(other.passes)
        self.clbAdd += other.clbAdd
        self.clbCompare += other.clbCompare
        self.clbSelect += other.clbSelect
        self.memReadBytes += other.memReadBytes
        self.memWriteBytes += other.memWriteBytes
        self.fifoWords += other.fifoWords
        return self

    @property
    def nPasses(self) -> int:
        return sum(self.passes.values())


@dataclass
class Resources:
    """What one module instantiates: crossbars per (mode, rows, cols); memories as (words, bits) -> count; memShared
    as (words, bits) -> copies per group whose copies share every address and write enable (VTR packs such a group by
    its summed width, simulator/platform/area.py)."""
    crossbars: Counter = field(default_factory=Counter)
    memories: Counter = field(default_factory=Counter)
    memShared: Counter = field(default_factory=Counter)

    @property
    def nCrossbars(self) -> int:
        return sum(self.crossbars.values())


@dataclass
class Report:
    """One run of a workload: cycles (last output word + 1), the steps, port events, counts, resources, basis and
    role of every instance, and the windows that left on external ports."""
    cycles: int
    steps: dict                                   # instance -> [Step]
    events: list                                  # [PortEvent], by (time, text)
    counts: dict                                  # instance -> Counts
    resources: dict                               # instance -> Resources
    basis: dict                                   # instance -> "rtl-pinned" | "model-only"
    roles: dict                                   # instance -> role (the workload's name for the instance's job)
    outputs: dict = field(default_factory=dict)   # external output port -> [Window]

    def eventLines(self) -> list:
        return [e.line for e in self.events]

    @property
    def totalCounts(self) -> Counts:
        total = Counts()
        for c in self.counts.values():
            total.add(c)
        return total

    def countsByRole(self) -> dict:
        """role -> the summed Counts of its instances."""
        out = {}
        for instance, c in self.counts.items():
            out.setdefault(self.roles[instance], Counts()).add(c)
        return out

    @property
    def totalResources(self) -> Resources:
        total = Resources()
        for r in self.resources.values():
            total.crossbars.update(r.crossbars)
            total.memories.update(r.memories)
            total.memShared.update(r.memShared)
        return total

    @property
    def modelOnly(self) -> bool:
        return any(b != "rtl-pinned" for b in self.basis.values())
