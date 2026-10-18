"""environment.py — the environment of a workload: a source and a sink (docs/simulator.md §4.4, last row).

Windows are numbered across the stream: window i (= unit u, window w of the unit, i = u * windowsPerUnit + w) goes
on channel i mod L (architecture.md §3.1). One unit may span several channels, and one channel carries the units of
the stream in order.

Source: sends units on one output port. A window starts at the latest of:
- the unit's availability time;
- its channel being free: one cycle after the channel's previous window ends, and at least `period` cycles after
  that window's start (the producer's row period; default: back to back);
- for unit u >= CAP, the credit u - CAP, plus SRC_CR_LAT = 0 (on a burst port that credit is x_rdy rising, and the
  burst starts in that cycle).
The windows of one unit that start on different channels in the same cycle stay in lock-step (the GEMM's lanes).

Sink: takes every word of its input port, unit by unit; on a credited port it pulses the credit `creditLatency`
cycles after the unit's last word (default SINK_CR_LAT = 1).
"""
from __future__ import annotations

import numpy as np

from simulator.core.module import Module, Need
from simulator.core.records import PortSpec, Step

SRC_CR_LAT = 0     # source: a credit (or x_rdy) at t lets the next unit start at t
SINK_CR_LAT = 1    # sink: the credit pulses the cycle after the unit's last word


def windowsPerChannel(unit: int, n: int, L: int) -> dict:
    """{channel: windows of `unit` on it} when the unit's n windows are dealt across the stream (window i on i mod L)."""
    out = {}
    for i in range(unit * n, (unit + 1) * n):
        out[i % L] = out.get(i % L, 0) + 1
    return out


class Source(Module):
    NAME = "source"
    BASIS = "rtl-pinned"          # the environment: the testbench drives exactly this
    STEP_METHODS = {"units": "stepUnit"}

    def __init__(self, instance: str, spec: PortSpec, units: int, windowsPerUnit: int = 1, availableAt=0,
                 period: int | None = None, payload=None, withValues: bool = True):
        super().__init__(instance, withValues=withValues)
        if spec.direction != "out":
            raise ValueError(f"{instance}: a source needs an output PortSpec")
        self.spec, self.units, self.windowsPerUnit = spec, int(units), int(windowsPerUnit)
        self.availableAt = availableAt            # one time, or a time per unit
        self.period = period                      # a channel's minimum spacing between window starts
        self.payload = payload                    # payload(unit, window) -> the window's values, or None
        self.channelEnd = [-1] * spec.L           # last word cycle per channel
        self.channelStart = [None] * spec.L       # last window start per channel

    def ports(self) -> dict:
        return {self.spec.name: self.spec}

    def processes(self) -> dict:
        return {"units": self.units}

    def _availableAt(self, unit: int) -> int:
        t = self.availableAt
        return int(t if isinstance(t, (int, np.integer)) else t[unit])

    def need(self, process: str, unit: int) -> Need:
        s = self.spec
        credited = s.discipline != "push" and unit >= s.CAP
        return Need(credits={s.name: (unit - s.CAP) if credited else None})

    def stepUnit(self, unit: int, io) -> Step:
        s = self.spec
        tCredit = io.creditTime(s.name)
        tUnit = max(self._availableAt(unit), -1 if tCredit is None else tCredit + SRC_CR_LAT)
        starts = []
        for w in range(self.windowsPerUnit):
            channel = (unit * self.windowsPerUnit + w) % s.L
            t = max(tUnit, self.channelEnd[channel] + 1)
            if self.period is not None and self.channelStart[channel] is not None:
                t = max(t, self.channelStart[channel] + self.period)
            carried = self.payload is not None and (self.withValues or s.control)
            values = self.payload(unit, w) if carried else None
            io.emit(s.name, channel, unit, w, t, values)
            self.channelStart[channel], self.channelEnd[channel] = t, t + s.words - 1
            starts.append(t)
        return Step("units", unit, tUnit, min(starts), max(starts) + s.words)


class Sink(Module):
    NAME = "sink"
    BASIS = "rtl-pinned"
    STEP_METHODS = {"units": "stepUnit"}

    def __init__(self, instance: str, spec: PortSpec, units: int, windowsPerUnit: int = 1,
                 creditLatency: int = SINK_CR_LAT, withValues: bool = True):
        super().__init__(instance, withValues=withValues)
        if spec.direction != "in":
            raise ValueError(f"{instance}: a sink needs an input PortSpec")
        self.spec, self.units, self.windowsPerUnit = spec, int(units), int(windowsPerUnit)
        self.creditLatency = int(creditLatency)
        self.received = []                        # every window, in arrival order

    def ports(self) -> dict:
        return {self.spec.name: self.spec}

    def processes(self) -> dict:
        return {"units": self.units}

    def need(self, process: str, unit: int) -> Need:
        return Need(windows={self.spec.name: windowsPerChannel(unit, self.windowsPerUnit, self.spec.L)})

    def stepUnit(self, unit: int, io) -> Step:
        s = self.spec
        windows = io.windows(s.name)
        if any(w.unit != unit for w in windows):
            raise RuntimeError(f"{self.instance}: unit {unit} expected, got units {sorted({w.unit for w in windows})}")
        self.received += windows
        tFirst = min(w.time for w in windows)
        tLast = max(w.time for w in windows) + s.words - 1
        if s.discipline != "push":
            io.giveCredit(s.name, unit, tLast + self.creditLatency)
        return Step("units", unit, tFirst, tFirst, tLast + 1)
