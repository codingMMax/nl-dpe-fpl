"""crossbar.py — the crossbar primitive `dpe` (ip/xbar/spec.md, the row-parallel interface) as `Crossbar` and
`CrossbarGroup` (docs/simulator.md §4.2).

One call, `run(x, tReady)`, computes a pass's values (int8 MAC to int32, then the output stage's form), its timing on
the crossbar's own state (the pass rule below), and adds the pass to the counts. `CrossbarGroup(n)` is n crossbars in
lock-step from one leader (one timing state, the n crossbars' values in one numpy operation): a GEMM's tiles, a DIMM
wave, an accumulator group.

Pass rule (ip/xbar/spec.md §4; user 2026-10-21/22): a burst of P bit planes, LSB first, one per cycle; each plane fires
the cycle after it arrives; the ACAM fires the cycle after the MSB plane; P drain planes follow; the next burst may
start the cycle after the ACAM (one accumulator, one output register):
    load_start    = max(tReady, acam_prev + RDY_LAT)              (the first pass: max(tReady, 0))
    load_end      = load_start + P - 1
    compute_start = load_start + FIRE_LAT
    msb_fire      = compute_start + P - 1
    acam_done     = max(msb_fire + ACAM_LAT, out_free)            (out_free = drain_end_prev + 1: never binds)
    drain_start   = acam_done + DRAIN_LAT
    drain_end     = acam_done + P
So a pass alone takes 2P + 2 = 18 cycles and back-to-back passes start every P + 2 = 10, on every array and at any
extent. The extent (ROWS_USED, COLS_USED) sets the values and the energy only. Until 2026-10-22 the crossbar had a
40-bit port each way (LOAD = ceil(8 ROWS_USED / 40), OUTPUT = ceil(8 C / 40) words).

Values (the certified integer math; the oracle stays independent):
    y        = x @ W exactly (int32), input rows >= ROWS_USED read as zero (P29)
    REGULAR    trunc8(y)                ACTIVATION  trunc8(relu(y))
    EXP        trunc8(1 + y + floor(y^2 / 2))   (wide intermediate)          LOG  trunc8(y - 1)
    trunc8(z) = the low byte of z clipped to int32
The drained window holds the first COLS_USED output bytes (the columns past it drain as zero).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from simulator.core.trace import label

NAME = "dpe"
# RTL parameter -> model name (ip/xbar/spec.md §2: KERNEL_WIDTH = R, NUM_COLS = C, PRECISION = P)
PARAMS = {"KERNEL_WIDTH": "R", "NUM_COLS": "C", "PRECISION": "P", "ROWS_USED": "rowsUsed", "COLS_USED": "colsUsed"}
MODES = {"REGULAR": 0, "ACTIVATION": 1, "EXP": 2, "LOG": 3}       # nl_dpe_control codes
PLATFORM_MODES = {"identity": "REGULAR", "relu": "ACTIVATION", "exp": "EXP", "log": "LOG"}
NL_DPE_MODES = ("REGULAR", "ACTIVATION", "EXP", "LOG")

# named latencies of the pass rule (I5)
FIRE_LAT = 1        # a plane fires the cycle after it arrives
ACAM_LAT = 1        # the output stage fires the cycle after the MSB plane (no add cycle)
DRAIN_LAT = 1       # the first drain plane leaves the cycle after the output stage fires
RDY_LAT = 1         # MSB_SA_Ready rises the cycle after the output stage fires: the next burst may start


@dataclass(frozen=True)
class Geometry:
    """One crossbar at its pass extent."""
    R: int
    C: int
    P: int
    ROWS_USED: int
    COLS_USED: int
    LOAD_CYC: int       # planes per burst, P
    OUTPUT_CYC: int     # drain planes, P
    WR_CYC: int         # weight-programming strobes, R C (setup, untimed)


def geometry(R: int, C: int, P: int = 8, rowsUsed: int | None = None, colsUsed: int | None = None) -> Geometry:
    """The RTL's elaboration guard is 1 <= ROWS_USED <= R and 1 <= COLS_USED <= C."""
    rowsUsed = R if rowsUsed is None else rowsUsed
    colsUsed = C if colsUsed is None else colsUsed
    if not (1 <= rowsUsed <= R and 1 <= colsUsed <= C):
        raise ValueError(f"pass extent rows={rowsUsed}, cols={colsUsed} outside 1..{R} / 1..{C}")
    if P < 1:
        raise ValueError(f"P must be >= 1, got {P}")
    return Geometry(R, C, P, rowsUsed, colsUsed, LOAD_CYC=P, OUTPUT_CYC=P, WR_CYC=R * C)


def trunc8(z) -> np.ndarray:
    clipped = np.clip(np.asarray(z, dtype=np.int64), -(2 ** 31), 2 ** 31 - 1)
    return (clipped & 0xFF).astype(np.uint8)


def outputStage(y: np.ndarray, mode: str) -> np.ndarray:
    """The output stage's form of the mode, as unsigned bytes."""
    y64 = np.asarray(y, dtype=np.int64)
    if mode == "REGULAR":
        return trunc8(y64)
    if mode == "ACTIVATION":
        return trunc8(np.maximum(y64, 0))
    if mode == "EXP":
        return trunc8(1 + y64 + (y64 * y64) // 2)
    if mode == "LOG":
        return trunc8(y64 - 1)
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class Pass:
    """One pass of a group: the drained windows' values and the pass events (absolute cycles)."""
    y: np.ndarray | None        # int32 [n, C]: the crossbar output before the output stage
    out: np.ndarray | None      # uint8 [n, C]: the output stage's bytes for every column
    window: np.ndarray | None   # int8 [n, COLS_USED]: the drained window (the first COLS_USED bytes)
    time: int                   # drain_start: the window's first plane
    events: dict                # load_start .. drain_end (spec names)
    nextReadyAt: int            # the cycle the next burst may start (MSB_SA_Ready rises)


class CrossbarGroup:
    """n crossbars of R x C in lock-step: one timing state, one mode, one extent; weights [n, R, C] (identity if
    None)."""

    def __init__(self, count: int, R: int, C: int, P: int = 8, *, mode: str = "REGULAR", rowsUsed: int | None = None,
                 colsUsed: int | None = None, weights=None, modes=NL_DPE_MODES, withValues: bool = True):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode not in modes:
            raise ValueError(f"mode {mode} is not available on this platform's output stage (modes {tuple(modes)})")
        self.count = int(count)
        self.geometry = geometry(R, C, P, rowsUsed, colsUsed)
        self.R, self.C, self.P = R, C, P
        self.rowsUsed, self.colsUsed = self.geometry.ROWS_USED, self.geometry.COLS_USED
        self.mode, self.withValues = mode, withValues
        if withValues:
            self.weights64 = self._weights(weights)
        self.acamPrev, self.outputFreeAt = None, 0
        self.nextReady = 0                                        # acam_prev + RDY_LAT (a traced Time)
        self.passes = Counter()                                   # (mode, rows, cols) -> passes since last cleared

    @classmethod
    def fromPlatform(cls, platform, count: int, *, R: int | None = None, C: int | None = None, **kw) -> "CrossbarGroup":
        """P and the output stage's modes from a platform (simulator/platform/platform.py `Platform`)."""
        modes = tuple(PLATFORM_MODES[m] for m in platform.modes)
        return cls(count, platform.rows if R is None else R, platform.cols if C is None else C,
                   platform.inputBitSlices, modes=modes, **kw)

    def _weights(self, weights) -> np.ndarray:
        n, R, C = self.count, self.R, self.C
        if weights is None:
            return np.broadcast_to(np.eye(R, C, dtype=np.int8), (n, R, C)).astype(np.int64)
        W = np.asarray(weights)
        if W.shape == (R, C) and n == 1:
            W = W[None]
        if W.shape != (n, R, C):
            raise ValueError(f"weights must be [{n}, {R}, {C}], got {W.shape}")
        if W.dtype != np.int8 and (W.min() < -128 or W.max() > 127):
            raise ValueError("weights must be int8")
        return W.astype(np.int64)

    def readyAt(self) -> int:
        """The first cycle the next burst may start."""
        return self.nextReady

    def run(self, x, tReady: int) -> Pass:
        """One lock-step pass: x [n, R] int8 (one input per crossbar; [R] when n == 1), ready at `tReady`."""
        P = self.P
        loadStart = max(tReady, self.readyAt())                   # a traced Time keeps its chain (trace.py)
        with label("dpe_load"):                                   # plane 0 arrives
            computeStart = loadStart + FIRE_LAT
        with label("dpe_compute"):                                # the P planes fire, then the output stage
            msbFire = computeStart + P - 1
            acamDone = max(msbFire + ACAM_LAT, self.outputFreeAt)
            nextReady = acamDone + RDY_LAT
        with label("dpe_drain"):
            drainStart = acamDone + DRAIN_LAT
            drainEnd = acamDone + P
            outputFree = drainEnd + 1
        self.acamPrev, self.outputFreeAt, self.nextReady = acamDone, outputFree, nextReady
        self.passes[(self.mode, self.rowsUsed, self.colsUsed)] += self.count
        events = dict(load_start=loadStart, load_end=loadStart + P - 1, compute_start=computeStart, msb_fire=msbFire,
                      acam_done=acamDone, drain_start=drainStart, drain_end=drainEnd)
        y = out = window = None
        if self.withValues:
            y, out, window = self._values(x)
        return Pass(y=y, out=out, window=window, time=drainStart, events=events, nextReadyAt=nextReady)

    def _values(self, x) -> tuple:
        xa = np.asarray(x)
        if xa.shape == (self.R,) and self.count == 1:
            xa = xa[None]
        if xa.shape != (self.count, self.R):
            raise ValueError(f"x must be [{self.count}, {self.R}], got {xa.shape}")
        if xa.min(initial=0) < -128 or xa.max(initial=0) > 127:
            raise ValueError("x must be int8")
        x64 = xa.astype(np.int64)
        x64[:, self.rowsUsed:] = 0                                 # P29: rows past the extent read as zero
        y64 = np.einsum("nr,nrc->nc", x64, self.weights64)
        if np.abs(y64).max(initial=0) > 2 ** 31 - 1:
            raise OverflowError("crossbar output exceeds int32")
        y = y64.astype(np.int32)
        out = outputStage(y, self.mode)
        return y, out, out[:, :self.colsUsed].view(np.int8)


class Crossbar(CrossbarGroup):
    """One crossbar: weights [R, C], x [R]; values come back as [C] / [COLS_USED]."""

    def __init__(self, R: int, C: int, P: int = 8, **kw):
        super().__init__(1, R, C, P, **kw)

    def run(self, x, tReady: int) -> Pass:
        p = super().run(x, tReady)
        if p.y is not None:
            p.y, p.out, p.window = p.y[0], p.out[0], p.window[0]
        return p
