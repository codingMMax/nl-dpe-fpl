"""attentionHead.py — one attention head as the graph of `att_head` (top/spec.md §4, §6; docs/simulator.md §4.5).

`headParams(...)` derives the head's values and every module's parameters (spec §2) into one `AttentionHeadParams`.
`attentionHead(params, ...)` builds the netlist: one E0 source per GEMM copy, the start source, 3 P_PROJ GEMMs, the QK
DIMM, the online softmax, the PV DIMM, the accumulator, att_ctrl, the O8 sink. Every edge and every credit route is a
wire of att_head. Instance roles (for the breakdowns): gemm (the projection copies), qk, sm, pv, acc, ctrl.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from simulator.core.cycleMath import ceilDiv, clog2, isPowerOfTwo
from simulator.core.environment import Sink, Source
from simulator.core.graph import Graph
from simulator.core.module import checkRules
from simulator.core.records import PortSpec
import simulator.models.attentionController as controller
from simulator.models.attentionController import AttCtrl, AttCtrlParams
from simulator.models.dimmTile import DimmTile, DimmTileParams
from simulator.models.gemm import GemmParams, GemmTop
from simulator.models.softmaxAccumulator import SoftmaxAccParams, SoftmaxOnlineAcc
from simulator.models.softmaxOnline import SoftmaxOnline, SoftmaxOnlineParams

BKV_DEFAULT = 16


NOC_BUF, NOC_P = 40, 8    # the floor rule's pre-bit-plane pass (held 2026-10-22: the mapping counts do not change)


def nOcFloor(S: int, D: int, R: int, C: int, BKV: int) -> int:
    """architecture.md §5.2, held at its pre-bit-plane form: the smallest N_OC with T(ceil(P_oc / N_OC)) <=
    max(ceil(S BKV / 5), T(1)), T the pre-bit-plane pass of I x I (LW = ceil(8 I / 40) words each way)."""
    I = min(R, C)
    LW = ceilDiv(I * 8, NOC_BUF)
    tFill, tSteady = LW + NOC_P + 2 + LW, max(LW + NOC_P, NOC_P + 2, LW + 1)
    pOc = S * D // I
    floor = max(ceilDiv(S * BKV, 5), tFill)
    return next(n for n in range(1, pOc + 1) if tFill + (ceilDiv(pOc, n) - 1) * tSteady <= floor)


@dataclass
class AttentionHeadParams:
    """The head's parameters (top/spec.md §2): the workload (NSEQ tokens, DM model width, D head width), the crossbar
    (R, C, P), the mapping (BKV, N_EXP, N_E, N_EP, N_OC, P_PROJ, X_S), the derived sizes, and every module's
    parameters."""
    NSEQ: int
    DM: int
    D: int
    BKV: int
    R: int
    C: int
    P: int
    N_EXP: int
    N_E: int
    N_EP: int
    N_OC: int
    P_PROJ: int
    X_S: int
    I: int            # min(R, C)
    S: int            # rows per run, min(NSEQ, I)
    RB: int           # runs (row blocks), NSEQ / S
    B: int            # key blocks, NSEQ / BKV
    V: int            # row tiles of a projection, ceil(DM / R)
    ROWS_X: int       # rows a projection lane loads: DM when V = 1, else R
    LOADX: int        # plane words per X burst, P
    LN: int           # clog2 NSEQ
    NG_QK: int        # QK output groups, N_E / D
    NG_PV: int        # PV output groups, N_EP / BKV
    T_X: int          # tokens per GEMM copy
    nXbar: int        # crossbars of the head
    gemm: GemmParams
    qk: DimmTileParams
    pv: DimmTileParams
    sm: SoftmaxOnlineParams
    acc: SoftmaxAccParams
    ctrl: AttCtrlParams


def headParams(NSEQ: int, DM: int, D: int, R: int, C: int, *, BKV: int = BKV_DEFAULT, N_E: int | None = None,
               N_EP: int | None = None, N_EXP: int = 2, N_OC: int | None = None, P_PROJ: int = 1, X_S: int = 16,
               P: int = 8) -> AttentionHeadParams:
    """The head's parameters (top/spec.md §2); N_E defaults to d (2026-10-23; 2d before), N_EP to 2d, N_OC to the
    floor rule (experiment.md §4.2)."""
    I = min(R, C)
    S = min(NSEQ, I)
    N_E = D if N_E is None else N_E
    N_EP = 2 * D if N_EP is None else N_EP
    N_OC = nOcFloor(S, D, R, C, BKV) if N_OC is None else N_OC
    V = ceilDiv(DM, R)
    rowsX = DM if V == 1 else R
    RB, B = NSEQ // S, NSEQ // BKV
    LN = clog2(NSEQ)
    checkRules("att_head", [
        (isPowerOfTwo(NSEQ) and isPowerOfTwo(P_PROJ) and P_PROJ <= NSEQ, "NSEQ, P_PROJ powers of two, P_PROJ <= NSEQ"),
        (D <= C, "D <= C (H = 1 in every GEMM)"),
        (rowsX >= D, "ROWS_X >= D (a copy's row period covers the DIMM converter's)")])
    windowElements = min(S * BKV, I)
    return AttentionHeadParams(
        NSEQ=NSEQ, DM=DM, D=D, BKV=BKV, R=R, C=C, P=P, N_EXP=N_EXP, N_E=N_E, N_EP=N_EP, N_OC=N_OC,
        P_PROJ=P_PROJ, X_S=X_S, I=I, S=S, RB=RB, B=B, V=V, ROWS_X=rowsX, LOADX=P, LN=LN,
        NG_QK=N_E // D, NG_PV=N_EP // BKV, T_X=NSEQ // P_PROJ,
        nXbar=3 * P_PROJ * V + 3 * P_PROJ + N_E + N_EP + (N_EXP + 2) + 2 * N_OC,
        gemm=GemmParams(K=DM, N=D, R=R, C=C, P=P, ROWS_USED=rowsX, COLS_USED=D),
        qk=DimmTileParams(M=S, N=BKV, K=D, R=R, C=C, P=P, N_E=N_E, X_S=X_S, A_DOM=0, A_SIGNED=1, A_CH=P_PROJ,
                          A_SLOTS=RB, A_FRAME=0, A_ROWS=1, B_DOM=0, B_SIGNED=1, B_CH=P_PROJ, B_SLOTS=B, TRANS_B=1,
                          NARROW=5, C_CAP=2),
        pv=DimmTileParams(M=S, N=D, K=BKV, R=R, C=C, P=P, N_E=N_EP, X_S=X_S, A_DOM=1, A_SIGNED=0, A_CH=N_EXP,
                          A_SLOTS=2, A_FRAME=1, A_ROWS=windowElements // BKV, B_DOM=0, B_SIGNED=1, B_CH=P_PROJ,
                          B_SLOTS=B, TRANS_B=0, NARROW=0, C_CAP=2),
        sm=SoftmaxOnlineParams(S=S, BKV=BKV, N=NSEQ, R=R, C=C, P=P, N_EXP=N_EXP, S_CH=N_E // D, LP_CAP=2,
                               LA_CAP=2, LL_CAP=1),
        acc=SoftmaxAccParams(S=S, D=D, R=R, C=C, P=P, N_OC=N_OC, NBLK=B, SO=LN, LN=LN, P_CH=N_EP // BKV,
                             VWP=9 + clog2(BKV)),                        # VWP: the PV DIMM's output width
        ctrl=AttCtrlParams(NSEQ=NSEQ, S=S, BKV=BKV, I=I, P_PROJ=P_PROJ, LOADX=P))


def attentionHead(params: AttentionHeadParams, *, X=None, WQ=None, WK=None, WV=None, withValues: bool = True,
                  xAvailableAt=0, xPeriod: int | None = None, tStart: int = 0) -> tuple:
    """(graph, parts) of one head. X [n, d_m] int8 and W* [d_m, d] int8 with values on. xAvailableAt: one time or per
    token (global index t), each copy's first burst no earlier than the run start; xPeriod: a copy's burst spacing."""
    hp = params
    n, P_PROJ, V, R = hp.NSEQ, hp.P_PROJ, hp.V, hp.R
    if withValues and any(a is None for a in (X, WQ, WK, WV)):
        raise ValueError("attentionHead: X, WQ, WK, WV are needed with values on")
    gemms = {f"g{kind}{c}": GemmTop(f"g{kind}{c}", hp.gemm, rows=hp.T_X, weights=W, withValues=withValues, role="gemm")
             for c in range(P_PROJ) for kind, W in (("q", WQ), ("k", WK), ("v", WV))}
    qk = DimmTile("qk", hp.qk, nA=n, nB=n, nJobs=hp.RB * hp.B, withValues=withValues, role="qk")
    pv = DimmTile("pv", hp.pv, nA=hp.RB * hp.B * (hp.S // hp.pv.A_ROWS), nB=n, nJobs=hp.RB * hp.B,
                  withValues=withValues, role="pv")
    sm = SoftmaxOnline("sm", hp.sm, runs=hp.RB, withValues=withValues, role="sm")
    acc = SoftmaxOnlineAcc("acc", hp.acc, runs=hp.RB, withValues=withValues, role="acc")
    xSpec = gemms["gq0"].ports()["act"]
    ctrl = AttCtrl("ctrl", hp.ctrl, xSpec=xSpec, withValues=withValues, role="ctrl")
    padded = None
    if withValues:
        padded = np.zeros((n, V * R), dtype=np.int8)
        padded[:, :hp.DM] = np.asarray(X, dtype=np.int8)
    sources = []
    for c in range(P_PROJ):
        tokens = list(range(c, n, P_PROJ))
        available = [xAvailableAt if isinstance(xAvailableAt, (int, np.integer)) else xAvailableAt[t] for t in tokens]
        available[0] = max(available[0], tStart + controller.RUN_LAT)   # the copy's first x_rdy (spec §5)

        def payload(u, w, tokens=tokens):                             # copy c's token u, lane w: rows w R .. + ROWS_X - 1
            return padded[tokens[u], w * R:w * R + hp.ROWS_X].copy()
        sources.append(Source(f"x{c}", replace(xSpec, name="x", direction="out"), hp.T_X, V, availableAt=available,
                              period=xPeriod, payload=payload if withValues else None, withValues=withValues))
    start = Source("st", PortSpec("start", "out", "burst", 1, 1, 1, 1, "head", CAP=1), 1, 1, availableAt=tStart,
                   withValues=withValues)
    sink = Sink("o", replace(acc.ports()["o"], direction="in"), hp.RB, acc.d.W, withValues=withValues)   # E9: O8 windows
    edges, credits = [], []
    for c in range(P_PROJ):
        edges += [(f"x{c}.x", f"g{kind}{c}.act") for kind in "qkv"] + [(f"x{c}.x", f"ctrl.x{c}")]
        edges += [(f"gq{c}.out", "qk.a", c), (f"gk{c}.out", "qk.b", c), (f"gv{c}.out", "pv.b", c)]
        credits += [(f"g{kind}{c}.act", f"ctrl.r{kind}{c}") for kind in "qkv"] + [(f"ctrl.x{c}", f"x{c}.x")]
    edges += [("ctrl.qj", "qk.j"), ("ctrl.pj", "pv.j"), ("qk.c", "sm.s"), ("ctrl.go", "sm.go"), ("sm.lp", "pv.a"),
              ("sm.la", "acc.la"), ("sm.ll", "acc.ll"), ("pv.c", "acc.p"), ("acc.o", "o.o"), ("st.start", "ctrl.start")]
    credits += [("qk.a", "ctrl.qa"), ("qk.b", "ctrl.qb"), ("pv.b", "ctrl.pb"), ("acc.run_done", "ctrl.rd")]
    modules = sources + [start, *gemms.values(), qk, sm, pv, acc, ctrl, sink]
    g = Graph(f"head_n{n}_dm{hp.DM}_d{hp.D}_{R}x{hp.C}_p{P_PROJ}", modules, edges, credits=credits)
    return g, dict(gemm=gemms, qk=qk, sm=sm, pv=pv, acc=acc, ctrl=ctrl, sink=sink, sources=sources, start=start)
