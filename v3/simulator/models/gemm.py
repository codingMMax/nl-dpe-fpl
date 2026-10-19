"""gemm.py — behavior model of `gemm_top` (ip/gemm/rtl/sim/gemm_top.v; charter v2/sim/spec/gemm.md v0.3;
docs/architecture.md §4.3, docs/simulator.md §4.4).

Y = trunc8(X W), REGULAR: V = ceil(K/R) row tiles x H = ceil(N/C) column tiles of `dpe`, in lock-step (one
`CrossbarGroup` of V*H), a byte tree over the V tiles of a column (TREE_PIPE registered levels, exact sum of the
sign-extended tile bytes), then the lane serializer (one register, the low byte of the sum). No output stage after the
reduction.

Process `rows`, one step per input row:
  in   `act` (burst on ready, CAP 1): the row's V lane windows, lane v = rows v*R .. v*R + ROWS_USED - 1, all lanes in
       the same cycles; the burst starts in a cycle MSB_SA_Ready is high
  pass the V*H tiles, tile (v, h) on lane v
  out  `out` (push): H lane windows of COLS_USED bytes, lane h = columns h*C ..; the first word at drain_start + L_W,
       L_W = TREE_PIPE + SER_LAT; dpe_done one cycle after the last word
  credit on `act`: MSB_SA_Ready rises at msb_fire + 1 (the next burst may start)
Counts per row: V*H passes (REGULAR, ROWS_USED, COLS_USED); N*(V-1) tree adds; the N output bytes written.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.core.cycleMath import ceilDiv, clog2
from simulator.core.errors import ContractError
from simulator.core.module import Module, Need, checkRules
from simulator.core.records import PortSpec, Step
from simulator.core.stages import Delay, Tree
from simulator.core.trace import label
from simulator.core.crossbar import NL_DPE_MODES, CrossbarGroup

SER_LAT = 1        # the lane serializer register (charter §5.2: L_w = TREE_PIPE + 1)


@dataclass
class GemmParams:
    """The RTL parameters of gemm_top (RTL names). ROWS_USED / COLS_USED: the P29 pass extent (default: the array)."""
    K: int = 128
    N: int = 128
    R: int = 256
    C: int = 256
    BUF: int = 40
    P: int = 8
    ROWS_USED: int | None = None
    COLS_USED: int | None = None

    def __post_init__(self):
        if self.ROWS_USED is None:
            self.ROWS_USED = self.R
        if self.COLS_USED is None:
            self.COLS_USED = self.C


@dataclass(frozen=True)
class GemmDerived:
    """The RTL's localparams."""
    V: int             # row tiles, ceil(K / R)
    H: int             # column tiles, ceil(N / C)
    EPS: int           # elements per port word, BUF / 8
    TREE_PIPE: int     # registered levels of the byte tree
    NTILES: int        # V * H
    L_W: int           # drain_start -> the first output word: TREE_PIPE + SER_LAT


class GemmMath:
    """The values of gemm_top: the tiles' weights, a row's lane inputs, the tiles' bytes for the tree and the output
    bytes. `probeS` keeps every row's wide reduced sum [H * COLS_USED] (the testbench's S_col probe)."""

    def __init__(self, p: GemmParams, d: GemmDerived, weights):
        if weights is None:
            raise ValueError("gemm_top: weights [K, N] are needed with values on")
        W = np.asarray(weights)
        if W.shape != (p.K, p.N):
            raise ValueError(f"gemm_top: weights must be [{p.K}, {p.N}], got {W.shape}")
        padded = np.zeros((d.V * p.R, d.H * p.C), dtype=np.int8)
        padded[:p.K, :p.N] = W
        # tile (v, h) = v * H + h
        self.tileWeights = padded.reshape(d.V, p.R, d.H, p.C).transpose(0, 2, 1, 3).reshape(d.NTILES, p.R, p.C)
        self.p, self.d = p, d
        self.probeS = []

    def laneInput(self, lanes: list) -> np.ndarray:
        """[NTILES, R] int8: lane v feeds tiles (v, 0 .. H-1) in its first ROWS_USED rows."""
        p, d = self.p, self.d
        x = np.zeros((d.NTILES, p.R), dtype=np.int8)
        for w in lanes:
            x[w.window * d.H:(w.window + 1) * d.H, :p.ROWS_USED] = w.values
        return x

    def tileBytes(self, window: np.ndarray) -> list:
        """The V tile byte planes [H, COLS_USED], sign-extended, for the tree."""
        return list(window.astype(np.int64).reshape(self.d.V, self.d.H, self.p.COLS_USED))

    def outputBytes(self, S: np.ndarray, h: int) -> np.ndarray:
        """Lane h's output: the low byte of the reduced sum."""
        return (S[h] & 0xFF).astype(np.uint8).view(np.int8)


class GemmTop(Module):
    NAME = "gemm_top"
    PARAMS_CLASS = GemmParams
    BASIS = "rtl-pinned"       # gemm_top RTL GATE 2 (v2, unchanged); this model reproduces gemm_sim (tests)
    STEP_METHODS = {"rows": "stepRow"}

    @staticmethod
    def derive(p: GemmParams) -> GemmDerived:
        V, H = ceilDiv(p.K, p.R), ceilDiv(p.N, p.C)
        treePipe = clog2(V) if V > 1 else 0
        return GemmDerived(V=V, H=H, EPS=p.BUF // 8, TREE_PIPE=treePipe, NTILES=V * H, L_W=treePipe + SER_LAT)

    @staticmethod
    def checkDomain(p: GemmParams) -> None:
        """The RTL's elaboration guard, including the P29 extent guard (gemm_top.v g_bad_extent)."""
        V, H = ceilDiv(p.K, p.R), ceilDiv(p.N, p.C)
        checkRules("gemm_top", [
            (min(p.K, p.N, p.R, p.C, p.BUF, p.P) >= 1, "K, N, R, C, BUF, P >= 1"),
            (1 <= p.ROWS_USED <= p.R and 1 <= p.COLS_USED <= p.C, "1 <= ROWS_USED <= R, 1 <= COLS_USED <= C"),
            (p.ROWS_USED == p.R if V > 1 else p.ROWS_USED >= p.K, "ROWS_USED = R when V > 1, >= K when V = 1"),
            (p.COLS_USED == p.C if H > 1 else p.COLS_USED >= p.N, "COLS_USED = C when H > 1, >= N when H = 1")])

    def __init__(self, instance: str, params: GemmParams, *, rows: int, weights=None, withValues: bool = True,
                 modes=NL_DPE_MODES, role: str | None = None):
        super().__init__(instance, params, withValues=withValues, role=role)
        p, d = self.p, self.d
        self.nRows = int(rows)
        self.math = GemmMath(p, d, weights) if withValues else None
        # the blocks: the V*H tiles in lock-step, the byte tree, the lane serializer
        self.tiles = CrossbarGroup(d.NTILES, p.R, p.C, p.BUF, p.P, mode="REGULAR", rowsUsed=p.ROWS_USED,
                                   colsUsed=p.COLS_USED, weights=None if self.math is None else self.math.tileWeights,
                                   modes=modes, withValues=withValues)
        self.tree, self.serializer = Tree(d.TREE_PIPE), Delay(SER_LAT)
        self.resources.crossbars[("REGULAR", p.ROWS_USED, p.COLS_USED)] = d.NTILES

    def ports(self) -> dict:
        p, d = self.p, self.d
        return {"act": PortSpec("act", "in", "burst", 8, d.EPS, p.ROWS_USED, d.V, "row", CAP=1,
                                signals=("act_in", "act_en", "MSB_SA_Ready")),
                "out": PortSpec("out", "out", "push", 8, d.EPS, p.COLS_USED, d.H, "row",
                                signals=("data_out", "out_valid", "dpe_done"))}       # reg_full is a busy flag

    def processes(self) -> dict:
        return {"rows": self.nRows}

    def need(self, process: str, unit: int) -> Need:
        return Need(windows={"act": Need.eachChannel(self.d.V)})

    def stepRow(self, row: int, io) -> Step:
        p, d = self.p, self.d
        lanes = io.windows("act")
        tBurst = lanes[0].time
        if any(w.time != tBurst for w in lanes):                      # the port contract: lanes in lock-step
            raise ContractError(f"{self.instance}: row {row} lanes not in lock-step: {[w.time for w in lanes]}")
        if tBurst < self.tiles.readyAt():                             # burst on ready: MSB_SA_Ready must be high
            raise ContractError(f"{self.instance}: row {row} burst at {tBurst} before MSB_SA_Ready "
                                f"({self.tiles.readyAt()})")
        x = self.math.laneInput(lanes) if self.math else None
        xbarPass = self.tiles.run(x, tBurst)
        io.giveCredit("act", row, xbarPass.nextReadyAt)
        S, tTree = self.tree(self.math.tileBytes(xbarPass.window) if self.math else None, xbarPass.time)
        _, tOut = self.serializer(None, tTree)
        if self.math:
            self.math.probeS.append(S.reshape(-1).copy())
        for h in range(d.H):
            io.emit("out", h, row, h, tOut, self.math.outputBytes(S, h) if self.math else None)
        nWords = ceilDiv(p.COLS_USED, d.EPS)
        with label("stream"):                                        # the serializer's words on the port
            lastWord = tOut + nWords - 1
            done = lastWord + 1
        self.counts.passes.update(self.tiles.passes)
        self.tiles.passes.clear()
        self.counts.clbAdd += p.N * (d.V - 1)
        self.counts.memWriteBytes += p.N
        events = dict(xbarPass.events, array_drain_start=tOut, array_drain_end=lastWord, done=done)
        return Step("rows", row, tBurst, tBurst, done, events)
