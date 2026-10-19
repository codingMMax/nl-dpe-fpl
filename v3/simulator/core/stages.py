"""stages.py — the stages a module is built from (docs/simulator.md §4.3). Stages never connect two modules.

Each stage has one timing rule. The latency of each use is a named constant of the module that uses it (I5); a plain
fixed latency is just `t + NAME_LAT` in the module.

  readyAt(*t)  a decision: the first cycle at which every condition holds (simulator/core/cycleMath.py)
  Serial       one unit at a time: a unit ready at t, busy n cycles, starts at max(t, the end of the previous unit)
  Buffer       entries used in rotation: a unit may enter only after the unit `entries` before it has left (else an
               overrun error: the producer broke its contract)
  Repack       a byte stream re-cut into words: the time of each output word from the read times of the input words
  Delay        values unchanged, time + k
  Tree         exact integer sum, time + levels
  Map          a function per element, time + k
  Fifo         a word is read at max(its write time + 1, the reader's decision)
  Memory       a slot is valid at its last write + 1; a read returns one cycle after its address
"""
from __future__ import annotations

import numpy as np

from simulator.core.cycleMath import ceilDiv
from simulator.core.errors import ContractError


class Serial:
    """A resource that serves one unit at a time (a word writer, a reader taking one word per cycle, a packer's
    output). A unit ready at t that keeps the resource busy n cycles starts at max(t, the end of the previous unit)."""

    def __init__(self):
        self.freeAt = 0                # the first cycle the next unit can start

    def take(self, tReady: int, n: int = 1) -> int:
        """The start cycle of a unit ready at tReady that needs n cycles."""
        t = max(tReady, self.freeAt)                   # the chosen Time keeps its chain
        self.freeAt = t + int(n)
        return t


class Buffer:
    """`entries` entries used in rotation (a double buffer, a row register per channel, slots). Unit u uses the entry
    of unit u - entries, so it may enter only from the cycle that unit released its entry. Units are released in
    order. An early unit is an overrun of the producer's contract."""

    def __init__(self, entries: int, name: str = "buffer"):
        if entries < 1:
            raise ValueError(f"{name}: entries must be >= 1, got {entries!r}")
        self.entries, self.name = int(entries), name
        self.freeFrom = []             # unit -> the first cycle its entry can take the next unit

    def put(self, unit: int, t: int) -> None:
        """Unit `unit` arrives at cycle t."""
        previous = unit - self.entries
        if previous < 0:
            return
        if previous >= len(self.freeFrom):
            raise ContractError(f"{self.name} overrun: unit {unit} at {t}, unit {previous} has not released its entry")
        if t < self.freeFrom[previous]:
            raise ContractError(f"{self.name} overrun: unit {unit} at {t}, its entry is free from {self.freeFrom[previous]}")

    def release(self, unit: int, t: int) -> None:
        """Unit `unit`'s entry can take a new unit from cycle t."""
        if unit != len(self.freeFrom):
            raise ValueError(f"{self.name}: unit {unit} released out of order (expected unit {len(self.freeFrom)})")
        self.freeFrom.append(t)


class Repack:
    """A byte stream re-cut into words of `width` bytes, the last word padded. Input word i carries sizes[i] bytes.
    Output word k is complete with the input word that holds its last byte; it leaves `latency` cycles after that
    input word is read, one output word per cycle (a second word completed by the same read leaves one cycle later)."""

    def __init__(self, sizes, width: int, latency: int):
        sizes = [int(s) for s in sizes]
        if not sizes or min(sizes) < 1 or width < 1:
            raise ValueError("Repack: sizes must be >= 1 bytes and width >= 1")
        ends = np.cumsum(sizes)                                   # the byte after each input word
        total = int(ends[-1])
        lastByte = np.minimum(np.arange(ceilDiv(total, width)) * width + width - 1, total - 1)
        self.sourceWord = [int(i) for i in np.searchsorted(ends, lastByte, side="right")]   # input word of each output
        self.nInputs, self.latency = len(sizes), int(latency)
        self.output = Serial()

    def times(self, readTimes) -> list:
        """The output words' cycles, from the input words' read cycles."""
        if len(readTimes) != self.nInputs:
            raise ValueError(f"Repack: {len(readTimes)} read times for {self.nInputs} input words")
        return [self.output.take(readTimes[i] + self.latency) for i in self.sourceWord]


class Delay:
    """Values unchanged, time + k (k registers)."""

    def __init__(self, k: int):
        self.k = int(k)

    def __call__(self, values, t: int):
        return values, t + self.k


class Tree:
    """Exact integer sum of the inputs, time + levels (one register per level)."""

    def __init__(self, levels: int):
        self.levels = int(levels)

    def __call__(self, inputs, t: int):
        if inputs is None:
            return None, t + self.levels
        if len(inputs) == 0:
            raise ValueError("Tree needs at least one input")
        total = np.zeros(np.shape(inputs[0]), dtype=np.int64)
        for x in inputs:
            total += np.asarray(x, dtype=np.int64)
        return total, t + self.levels


class Map:
    """`fn` per element (narrowing, sign, clamp, compaction), time + k."""

    def __init__(self, fn, k: int):
        self.fn, self.k = fn, int(k)

    def __call__(self, values, t: int):
        return (None if values is None else self.fn(values)), t + self.k


class Fifo:
    """Values unchanged; a word is read at max(its write time + 1, the reader's decision). Holds `depth` entries;
    writing into a full FIFO, or reading an empty one, is an error."""

    def __init__(self, depth: int):
        if depth < 1:
            raise ValueError(f"FIFO depth must be >= 1, got {depth!r}")
        self.depth = int(depth)
        self.queue = []                # (values, write time)
        self.readTimes = []            # read times of the entries already read, to free slots in time order

    def write(self, values, t: int) -> None:
        # an entry read at time r frees its slot for a write at r + 1 or later
        busy = len(self.queue) + sum(1 for r in self.readTimes if r >= t)
        if busy >= self.depth:
            raise OverflowError(f"FIFO write at {t}: {self.depth} entries busy")
        if self.queue and t <= self.queue[-1][1]:
            raise ValueError(f"FIFO write at {t} not after the previous write at {self.queue[-1][1]}")
        self.queue.append((values, t))

    def read(self, tDecide: int):
        if not self.queue:
            raise IndexError("FIFO read while empty")
        values, tWrite = self.queue.pop(0)
        tRead = max(tWrite + 1, tDecide)
        self.readTimes.append(tRead)
        return values, tRead


class Memory:
    """Slots of words. A slot is valid at its last write + 1; a read returns its words one cycle after its address."""

    def __init__(self, slots: int):
        if slots < 1:
            raise ValueError(f"Memory needs >= 1 slot, got {slots!r}")
        self.slots = int(slots)
        self.data = [None] * self.slots
        self.validAt = [None] * self.slots

    def write(self, slot: int, values, tLastWrite: int) -> int:
        if not 0 <= slot < self.slots:
            raise IndexError(f"slot {slot} outside 0..{self.slots - 1}")
        self.data[slot], self.validAt[slot] = values, tLastWrite + 1
        return self.validAt[slot]

    def read(self, slot: int, tAddress: int):
        if not 0 <= slot < self.slots:
            raise IndexError(f"slot {slot} outside 0..{self.slots - 1}")
        if self.validAt[slot] is None or tAddress < self.validAt[slot]:
            raise ValueError(f"read of slot {slot} at {tAddress} before it is valid ({self.validAt[slot]})")
        return self.data[slot], tAddress + 1
