"""crossbar.py — the crossbar primitive `dpe` (ip/xbar/rtl/sim/dpe_nldpe.v) as `Crossbar` and `CrossbarGroup`
(docs/simulator.md §4.2).

One call, `run(x, tReady)`, computes a pass's values (int8 MAC to int32, then the output stage's form), its timing on
the crossbar's own state (the pass grammar below), and adds the pass to the counts. `CrossbarGroup(n)` is n crossbars
in lock-step from one leader (one timing state, the n crossbars' values in one numpy operation): a GEMM's tiles, a DIMM
wave, an accumulator group.

Pass grammar (spec dpe_nldpe.md §4–§5; LOAD at the P29 extent ROWS_USED; the drain walks every column, user 2026-10-18,
where the RTL still stops after COLS_USED: the model is not RTL-pinned for passes with COLS_USED < C):
    load_start    = max(tReady, msb_fire_prev + RDY_LAT)           (the first pass: max(tReady, 0))
    compute_start = max(load_start + LOAD_CYC, acc_free)
    msb_fire      = compute_start + P - 1
    acam_done     = max(msb_fire + SHIFT_ACC_LAT + ACAM_LAT, out_free)
    drain_start   = acam_done + DRAIN_LAT
    drain_end     = acam_done + OUTPUT_CYC
    state:  msb_fire_prev = msb_fire;  acc_free = acam_done + 1;  out_free = drain_end + 1
LOAD_CYC = ceil(8 ROWS_USED / BUF), OUTPUT_CYC = ceil(8 C / BUF) (every column; until 2026-10-18 ceil(8 COLS_USED / BUF));
COMPUTE_CYC = P + 2 emerges (acam_done - compute_start + 1). BUF, P and the output stage's modes come from the
platform: on NL-DPE BUF = 40, P = 8, ACAM.

Values (the certified integer math; the oracle stays independent):
    y        = x @ W exactly (int32), input rows >= ROWS_USED read as zero (P29)
    REGULAR    trunc8(y)                ACTIVATION  trunc8(relu(y))
    EXP        trunc8(1 + y + floor(y^2 / 2))   (wide intermediate)          LOG  trunc8(y - 1)
    trunc8(z) = the low byte of z clipped to int32
The drained window holds the first COLS_USED output bytes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from simulator.core.cycleMath import ceilDiv
from simulator.core.trace import label

NAME = "dpe"
# RTL parameter -> model name (dpe_nldpe.v: KERNEL_WIDTH = R, NUM_COLS = C, DPE_BUF_WIDTH = BUF, PRECISION = P)
PARAMS = {"KERNEL_WIDTH": "R", "NUM_COLS": "C", "DPE_BUF_WIDTH": "BUF", "PRECISION": "P",
          "ROWS_USED": "rowsUsed", "COLS_USED": "colsUsed"}
MODES = {"REGULAR": 0, "ACTIVATION": 1, "EXP": 2, "LOG": 3}       # nl_dpe_control codes
PLATFORM_MODES = {"identity": "REGULAR", "relu": "ACTIVATION", "exp": "EXP", "log": "LOG"}
NL_DPE_MODES = ("REGULAR", "ACTIVATION", "EXP", "LOG")

# named latencies of the pass grammar (I5)
RDY_LAT = 1         # MSB_SA_Ready rises the cycle after the MSB slice fires: the next burst may start (P1/A9)
SHIFT_ACC_LAT = 1   # the MSB slice is shifted and accumulated into y the cycle after it fires
ACAM_LAT = 1        # the output stage fires the cycle after (bounded by the output buffer, P11)
DRAIN_LAT = 1       # the first output word leaves the cycle after the output stage fires


@dataclass(frozen=True)
class Geometry:
    """The RTL's localparams of one crossbar at its pass extent (dpe_nldpe.v)."""
    R: int
    C: int
    BUF: int
    P: int
    ROWS_USED: int
    COLS_USED: int
    LOAD_CYC: int
    COMPUTE_CYC: int
    OUTPUT_CYC: int
    WR_CYC: int
    EPS: int


def geometry(R: int, C: int, BUF: int = 40, P: int = 8, rowsUsed: int | None = None,
             colsUsed: int | None = None) -> Geometry:
    """The localparams; the RTL's elaboration guard is 1 <= ROWS_USED <= R and 1 <= COLS_USED <= C. ROWS_USED sets the
    load; COLS_USED is the valid prefix of the drained window, while the drain itself walks every column (user,
    2026-10-18)."""
    rowsUsed = R if rowsUsed is None else rowsUsed
    colsUsed = C if colsUsed is None else colsUsed
    if not (1 <= rowsUsed <= R and 1 <= colsUsed <= C):
        raise ValueError(f"pass extent rows={rowsUsed}, cols={colsUsed} outside 1..{R} / 1..{C}")
    return Geometry(R, C, BUF, P, rowsUsed, colsUsed, LOAD_CYC=ceilDiv(rowsUsed * 8, BUF), COMPUTE_CYC=P + 2,
                    OUTPUT_CYC=ceilDiv(C * 8, BUF), WR_CYC=R * C, EPS=BUF // 8)     # the drain walks every column


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
    time: int                   # drain_start: the window's first word
    events: dict                # load_start .. drain_end (spec names)
    nextReadyAt: int            # the cycle the next burst may start (MSB_SA_Ready rises)


class CrossbarGroup:
    """n crossbars of R x C in lock-step: one timing state, one mode, one extent; weights [n, R, C] (identity if
    None)."""

    def __init__(self, count: int, R: int, C: int, BUF: int = 40, P: int = 8, *, mode: str = "REGULAR",
                 rowsUsed: int | None = None, colsUsed: int | None = None, weights=None, modes=NL_DPE_MODES,
                 withValues: bool = True):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode not in modes:
            raise ValueError(f"mode {mode} is not available on this platform's output stage (modes {tuple(modes)})")
        self.count = int(count)
        self.geometry = geometry(R, C, BUF, P, rowsUsed, colsUsed)
        self.R, self.C, self.BUF, self.P = R, C, BUF, P
        self.rowsUsed, self.colsUsed = self.geometry.ROWS_USED, self.geometry.COLS_USED
        self.mode, self.withValues = mode, withValues
        if withValues:
            self.weights64 = self._weights(weights)
        self.msbFirePrev, self.accumulatorFreeAt, self.outputFreeAt = None, 0, 0
        self.nextReady = 0                                        # msb_fire_prev + RDY_LAT (a traced Time)
        self.passes = Counter()                                   # (mode, rows, cols) -> passes since last cleared

    @classmethod
    def fromPlatform(cls, platform, count: int, *, R: int | None = None, C: int | None = None, **kw) -> "CrossbarGroup":
        """BUF, P and the output stage's modes from a platform (simulator/platform/platform.py `Platform`)."""
        modes = tuple(PLATFORM_MODES[m] for m in platform.modes)
        return cls(count, platform.rows if R is None else R, platform.cols if C is None else C,
                   platform.portWidthBits, platform.inputBitSlices, modes=modes, **kw)

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
        g = self.geometry
        loadStart = max(tReady, self.readyAt())                   # a traced Time keeps its chain (trace.py)
        with label("dpe_load"):
            computeStart = max(loadStart + g.LOAD_CYC, self.accumulatorFreeAt)
        with label("dpe_compute"):
            msbFire = computeStart + self.P - 1
            shiftAccDone = msbFire + SHIFT_ACC_LAT
            acamDone = max(shiftAccDone + ACAM_LAT, self.outputFreeAt)
            nextReady, accumulatorFree = msbFire + RDY_LAT, acamDone + 1
        with label("dpe_drain"):
            drainStart = acamDone + DRAIN_LAT
            drainEnd = acamDone + g.OUTPUT_CYC
            outputFree = drainEnd + 1
        self.msbFirePrev, self.accumulatorFreeAt, self.outputFreeAt = msbFire, accumulatorFree, outputFree
        self.nextReady = nextReady
        self.passes[(self.mode, self.rowsUsed, self.colsUsed)] += self.count
        events = dict(load_start=loadStart, load_end=loadStart + g.LOAD_CYC - 1, compute_start=computeStart,
                      msb_fire=msbFire, shift_acc_done=shiftAccDone, acam_done=acamDone, drain_start=drainStart,
                      drain_end=drainEnd)
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
        x64[:, self.rowsUsed:] = 0                                 # P29: rows not loaded read as zero
        y64 = np.einsum("nr,nrc->nc", x64, self.weights64)
        if np.abs(y64).max(initial=0) > 2 ** 31 - 1:
            raise OverflowError("crossbar output exceeds int32")
        y = y64.astype(np.int32)
        out = outputStage(y, self.mode)
        return y, out, out[:, :self.colsUsed].view(np.int8)


class Crossbar(CrossbarGroup):
    """One crossbar: weights [R, C], x [R]; values come back as [C] / [COLS_USED]."""

    def __init__(self, R: int, C: int, BUF: int = 40, P: int = 8, **kw):
        super().__init__(1, R, C, BUF, P, **kw)

    def run(self, x, tReady: int) -> Pass:
        p = super().run(x, tReady)
        if p.y is not None:
            p.y, p.out, p.window = p.y[0], p.out[0], p.window[0]
        return p
