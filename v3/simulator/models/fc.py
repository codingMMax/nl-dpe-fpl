"""fc.py — the FC workload: an X row source, one `gemm_top`, a sink (docs/simulator.md §4.5).

`fc(...)` builds the graph; `result(...)` reads Y back from the sink. Pricing a run into a DSE row is
simulator/platform/rows.py.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from simulator.core.environment import Sink, Source
from simulator.core.graph import Graph
from simulator.core.crossbar import NL_DPE_MODES, PLATFORM_MODES
from simulator.models.gemm import GemmParams, GemmTop


def extent(K: int, N: int, R: int, C: int, passExtent: str) -> tuple:
    """(ROWS_USED, COLS_USED): "full" = the whole array; "used" (P29) = K rows when V == 1, N columns when H == 1."""
    V, H = -(-K // R), -(-N // C)
    if passExtent == "full":
        return R, C
    if passExtent == "used":
        return (K if V == 1 else R), (N if H == 1 else C)
    raise ValueError(f"passExtent must be 'full' or 'used', got {passExtent!r}")


def fc(K: int, N: int, R: int, C: int, *, M: int = 1, platform=None, BUF: int = 40, P: int = 8,
       passExtent: str = "used", W=None, X=None, withValues: bool = True, availableAt=0) -> Graph:
    """X [M, K] int8 through W [K, N] int8 on R x C crossbars. With a platform, BUF, P and the output stage's modes are
    the platform's."""
    modes = NL_DPE_MODES
    if platform is not None:
        BUF, P = platform.portWidthBits, platform.inputBitSlices
        modes = tuple(PLATFORM_MODES[m] for m in platform.modes)
    rowsUsed, colsUsed = extent(K, N, R, C, passExtent)
    gemm = GemmTop("gemm", GemmParams(K=K, N=N, R=R, C=C, BUF=BUF, P=P, ROWS_USED=rowsUsed, COLS_USED=colsUsed),
                   rows=M, weights=W, withValues=withValues, modes=modes)
    V, H = gemm.d.V, gemm.d.H
    payload = None
    if withValues:
        if X is None:
            raise ValueError("fc: X [M, K] is needed with values on")
        X = np.asarray(X)
        if X.shape != (M, K):
            raise ValueError(f"fc: X must be [{M}, {K}], got {X.shape}")
        padded = np.zeros((M, V * R), dtype=np.int8)
        padded[:, :K] = X

        def payload(row: int, lane: int) -> np.ndarray:            # lane w of row u: rows w*R .. w*R + ROWS_USED - 1
            return padded[row, lane * R:lane * R + rowsUsed].copy()

    ports = gemm.ports()
    source = Source("x", replace(ports["act"], direction="out"), M, V, availableAt=availableAt, payload=payload,
                    withValues=withValues)
    sink = Sink("y", replace(ports["out"], direction="in"), M, H, withValues=withValues)
    return Graph(f"fc_K{K}_N{N}_R{R}_C{C}", [source, gemm, sink], [("x.act", "gemm.act"), ("gemm.out", "y.out")])


def result(sink: Sink, M: int, N: int, H: int) -> np.ndarray:
    """Y [M, N] int8 from the sink's windows (lane h = columns h*C ..)."""
    rows = {}
    for w in sink.received:
        rows.setdefault(w.unit, {})[w.window] = w.values
    return np.stack([np.concatenate([rows[u][h] for h in range(H)])[:N] for u in range(M)])
