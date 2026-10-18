"""dimmTile.py — behavior model of `dimm_tile_top` (the tiled DIMM), contract `ip/dimm/spec.md`.

One execution gives the values, the cycle of every port event and the counts (docs/simulator.md §1). The model is the
blocks of spec §4, each a stage (simulator/core/stages.py) or a crossbar group (simulator/core/crossbar.py):
  converterA, converterB   the LOG converters of the int8 ports, one crossbar per channel
  aWriter, bWriter         the loaders' writers (Serial): the transposer, the window stager, the shared row writer
  aStaging, bStaging       the staging entries (Buffer): the transposer's two groups, the stager's two windows per
                           channel, the row writer's two registers per channel
  aSlot                    the window-frame A slots (Buffer): a matrix may arrive only once the job using its slot
                           released it
  farm                     the N_E EXP crossbars in lock-step (CrossbarGroup). The slices run in lock-step by
                           construction (the same words reach every slice in the same cycle), so one timing state
                           serves them all; they appear in the resources and the memory images
Processes (one step per unit):
  a_in   an A input window      b_in   a B row      take   a job offer      jobs   a job
The values (the operands' log bytes, a wave's farm input, the signed sums, the narrowing, the memory images) are
`DimmTileMath`; a timing-only model has none.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.core.cycleMath import ceilDiv, clog2, isPowerOfTwo, readyAt
from simulator.core.errors import ContractError
from simulator.core.module import Module, Need, checkRules
from simulator.core.records import PortSpec, Step
from simulator.core.stages import Buffer, Serial
from simulator.core.crossbar import NL_DPE_MODES, Crossbar, CrossbarGroup, trunc8

# named latencies (spec §5)
CONV_IN_LAT = 1     # a row's first port word -> its converter burst
DIST_LAT = 1        # a stream word -> every slice
EMIT_LAT = 1        # a staged unit complete at the slices -> its first write
CR_LAT = 1          # a unit's last write -> its staging credit
DEC_LAT = 1         # the registered job decision
DRAIN_REG = 1       # the registered signed drain value
DONE_LAT = 2        # the last output word -> done
IDLE_LAT = 2        # the last drain word -> the slices read idle
REL_LAT = 1         # done -> a released slot reads invalid, its credit pulses
EPS = 5             # int8 elements per 40-bit port word


def signOfLog(L: np.ndarray) -> np.ndarray:
    """s(L) = L[7] ^ &L[6:0]: the operand's sign from its log byte."""
    b = np.asarray(L, dtype=np.int64) & 0xFF
    return ((b >> 7) & 1) ^ ((b & 0x7F) == 0x7F).astype(np.int64)


def addressBits(x: int) -> int:
    """The RTL declares every address field at least 1 bit wide (LTW, LGW, LBGW)."""
    return max(1, clog2(x))


@dataclass(kw_only=True)
class DimmTileParams:
    """The RTL parameters of dimm_tile_top (RTL names and defaults). The A port: int8 rows (A_DOM 0, row frame) or
    log windows (A_DOM 1, window frame); the B port: columns of B^T (TRANS_B 1) or rows (TRANS_B 0)."""
    M: int
    N: int
    K: int
    R: int = 64
    C: int = 64
    BUF: int = 40
    P: int = 8
    N_E: int
    X_S: int = 16
    A_DOM: int = 0
    A_SIGNED: int = 1
    A_CH: int = 1
    A_SLOTS: int = 1
    A_FRAME: int = 0
    A_ROWS: int = 1
    B_DOM: int = 0
    B_SIGNED: int = 1
    B_CH: int = 1
    B_SLOTS: int = 1
    TRANS_B: int = 1
    NARROW: int = 0
    C_CAP: int = 2


@dataclass(frozen=True)
class DimmTileDerived:
    """The RTL's localparams."""
    I: int            # min(R, C): the window size
    NB: int           # rows of C per window, I / N
    W_K: int          # windows per job, M N / I
    NG: int           # groups (crossbars per window position), N_E / K
    T_W: int          # waves per job, W_K / NG
    LW: int           # words per window, ceil(8 I / BUF)
    XS: int           # crossbars per slice
    NS: int           # slices
    KS: int           # k values per slice, XS / NG
    A_GRP: int        # rows of the A transposer group (a power of two <= min(LW, XS))
    B_GRP: int        # min(A_GRP, KS)
    LB: int           # the B row length: K (TRANS_B) or N
    A_W: int          # elements of an A window, A_ROWS K
    EW: int           # window-frame writer words per window
    VW: int           # the output value width, 9 + clog2 K
    EC: int           # the C port's element width: 8 when narrowed, else VW
    TREE_LEVELS: int  # clog2 K
    START_LAT: int    # job go -> the first farm burst, A_GRP + 2
    OUT_LAT: int      # drain start -> the group tree's output, DRAIN_REG + TREE_LEVELS
    A_WPM: int        # A input windows per matrix
    B_WPM: int        # B rows per matrix
    CAP_A: int        # the A port's credit units in flight
    A_UNIT: int       # A windows per credit unit
    CAP_B: int        # the B port's credit units in flight
    B_UNIT: int       # B rows per credit unit
    SAW: int          # the job port's A slot field width
    SBW: int          # the job port's B slot field width


@dataclass
class JobOffer:
    offeredAt: int
    slotA: int
    slotB: int
    releaseA: int


class DimmTileMath:
    """The values of dimm_tile_top: the operands' log bytes (LA [M, K] per A matrix, LB [K, N] per B matrix), a wave's
    farm input trunc8(LA + LB) over every slice's window, the signed EXP bytes summed per group (the trees), the
    narrowing, and the memory images of the loaders' writes (spec §6)."""

    def __init__(self, p: DimmTileParams, d: DimmTileDerived):
        self.p, self.d = p, d
        self.LA, self.LB = {}, {}                  # matrix -> log bytes
        self.outputs = {}                          # job -> [W_K, I] output values (narrowed)
        crossbar = np.arange(p.N_E)                # crossbar f = g + NG k works on window g + NG t of value k
        self.groupOf, self.kOf = crossbar % d.NG, crossbar // d.NG
        self.jobOut = None

    # -- the operands -----------------------------------------------------------------------------------------------
    def asLog(self, values) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    def converterInput(self, values) -> np.ndarray:
        """An int8 port row as the converter's input: the row, then zero rows (P29 extent)."""
        v = np.asarray(values, dtype=np.int64)
        if v.min(initial=0) < -128 or v.max(initial=0) > 127:
            raise ValueError(f"an int8 port carries {v.min()}..{v.max()}")
        x = np.zeros(self.p.R, dtype=np.int8)
        x[:v.size] = v
        return x

    def storeARow(self, matrix: int, row: int, logBytes: np.ndarray) -> None:
        self.LA.setdefault(matrix, np.zeros((self.p.M, self.p.K), dtype=np.int64))[row] = logBytes

    def storeAWindow(self, matrix: int, firstRow: int, logBytes: np.ndarray) -> None:
        p = self.p
        LA = self.LA.setdefault(matrix, np.zeros((p.M, p.K), dtype=np.int64))
        LA[firstRow:firstRow + p.A_ROWS] = logBytes.reshape(p.A_ROWS, p.K)

    def storeBColumn(self, matrix: int, n: int, logBytes: np.ndarray) -> None:
        self.LB.setdefault(matrix, np.zeros((self.p.K, self.p.N), dtype=np.int64))[:, n] = logBytes

    def storeBRow(self, matrix: int, k: int, logBytes: np.ndarray) -> None:
        self.LB.setdefault(matrix, np.zeros((self.p.K, self.p.N), dtype=np.int64))[k] = logBytes

    # -- a job's waves ----------------------------------------------------------------------------------------------
    def startJob(self) -> None:
        self.jobOut = np.zeros((self.d.W_K, self.d.I), dtype=np.int64)

    def _waveElements(self, wave: int) -> tuple:
        """(m, n, k) of every element [N_E, I] of the wave: crossbar f works on window groupOf[f] + NG wave."""
        p, d = self.p, self.d
        windows = self.groupOf + d.NG * wave
        e = windows[:, None] * d.I + np.arange(d.I)[None, :]
        return e // p.N, e % p.N, np.broadcast_to(self.kOf[:, None], e.shape)

    def waveInput(self, aMatrix: int, bMatrix: int, wave: int) -> np.ndarray:
        """The farm's input [N_E, R]: trunc8(log a + log b) in the first I rows."""
        m, n, k = self._waveElements(wave)
        X = np.zeros((self.p.N_E, self.p.R), dtype=np.int8)
        X[:, :self.d.I] = trunc8(self.LA[aMatrix][m, k] + self.LB[bMatrix][k, n]).view(np.int8)
        return X

    def waveOutput(self, aMatrix: int, bMatrix: int, wave: int, expBytes: np.ndarray) -> np.ndarray:
        """The group trees' sums [NG, I] of the wave's signed EXP bytes; kept as rows g + NG wave of the job output."""
        p, d = self.p, self.d
        m, n, k = self._waveElements(wave)
        e = expBytes[:, :d.I].astype(np.int64)                        # unsigned EXP bytes
        signA = signOfLog(self.LA[aMatrix][m, k]) if p.A_SIGNED else np.zeros(e.shape, dtype=np.int64)
        signB = signOfLog(self.LB[bMatrix][k, n]) if p.B_SIGNED else np.zeros(e.shape, dtype=np.int64)
        signed = np.where((signA ^ signB).astype(bool), -e, e)
        sums = np.stack([signed[self.groupOf == g].sum(axis=0) for g in range(d.NG)])
        self.jobOut[d.NG * wave + np.arange(d.NG)] = sums
        return sums

    def narrowed(self, v: np.ndarray) -> np.ndarray:
        """sat8(v >> NARROW) when narrowed, else the value."""
        return np.clip(v >> self.p.NARROW, -128, 127) if self.p.NARROW else v.copy()

    def finishJob(self, job: int) -> None:
        self.outputs[job] = self.narrowed(self.jobOut)

    # -- memory images (gate D1) ------------------------------------------------------------------------------------
    def memoryImages(self) -> dict:
        """Per slice, the A and B memory words the loaders wrote: {(slice, "A", mem, addr): bytes,
        (slice, "B", mem, addr): bytes} (spec §6)."""
        p, d = self.p, self.d
        NG, KS, AG, BG, NB = d.NG, d.KS, d.A_GRP, d.B_GRP, d.NB
        images = {}
        for i, LA in self.LA.items():
            slot = i % p.A_SLOTS
            for k in range(p.K):
                s, kl = divmod(k, KS)
                for w in range(d.W_K):
                    fl = (w % NG) + NG * kl
                    addr = (slot * d.T_W + w // NG) * AG + fl % AG
                    images[(s, "A", fl // AG, addr)] = tuple(int(x) & 0xFF for x in LA[w * NB:(w + 1) * NB, k])
        for i, LB in self.LB.items():
            slot = i % p.B_SLOTS
            for k in range(p.K):
                s, kl = divmod(k, KS)
                images[(s, "B", kl // BG, slot * BG + kl % BG)] = tuple(int(x) & 0xFF for x in LB[k])
        return images


class DimmTile(Module):
    NAME = "dimm_tile_top"
    PARAMS_CLASS = DimmTileParams
    BASIS = "rtl-pinned"           # GATE 2 2026-10-14: 16/16 corpus + 5/5 bring-up on 64x64, every event equal
    STEP_METHODS = {"a_in": "stepLoadA", "b_in": "stepLoadB", "take": "stepTakeJob", "jobs": "stepRunJob"}

    @staticmethod
    def derive(p: DimmTileParams) -> DimmTileDerived:
        M, N, K, N_E = p.M, p.N, p.K, p.N_E
        I = min(p.R, p.C)
        NB = I // N
        W_K = (M * N) // I
        NG = N_E // K
        T_W = W_K // NG if NG else 0
        LW = ceilDiv(I * 8, p.BUF)
        XS = min(p.X_S, N_E)
        NS = N_E // XS
        KS = XS // NG if NG else 0
        cap = max(1, min(LW, XS))
        A_GRP = 1 << (cap.bit_length() - 1)
        LB = K if p.TRANS_B else N
        VW = 9 + clog2(K)
        return DimmTileDerived(
            I=I, NB=NB, W_K=W_K, NG=NG, T_W=T_W, LW=LW, XS=XS, NS=NS, KS=KS, A_GRP=A_GRP, B_GRP=min(A_GRP, KS),
            LB=LB, A_W=p.A_ROWS * K, EW=(p.A_ROWS // NB) * KS if NB else 0, VW=VW, EC=8 if p.NARROW > 0 else VW,
            TREE_LEVELS=clog2(K), START_LAT=A_GRP + 2, OUT_LAT=DRAIN_REG + clog2(K),
            A_WPM=M // p.A_ROWS, B_WPM=N if p.TRANS_B else K,
            CAP_A=2 if p.A_FRAME == 0 else p.A_SLOTS, A_UNIT=NB if p.A_FRAME == 0 else M // p.A_ROWS,
            CAP_B=2 if p.TRANS_B else 4 * p.B_CH, B_UNIT=N if p.TRANS_B else 1,
            SAW=addressBits(p.A_SLOTS), SBW=addressBits(p.B_SLOTS))

    @staticmethod
    def checkDomain(p: DimmTileParams) -> None:
        M, N, K, N_E = p.M, p.N, p.K, p.N_E
        I = min(p.R, p.C)
        flags = ("A_DOM", "A_SIGNED", "A_FRAME", "B_DOM", "B_SIGNED", "TRANS_B")
        rules = [
            (all(isPowerOfTwo(x) for x in (M, N, K, N_E, p.X_S, I)), "M, N, K, N_E, X_S, I powers of two"),
            (p.BUF == 40, "BUF = 40"),
            (I >= 8 and N <= I and M * N >= I, "I >= 8, N <= I, M*N >= I"),
            (2 <= K <= N_E, "2 <= K <= N_E"),
            (min(p.A_CH, p.B_CH, p.A_SLOTS, p.B_SLOTS, p.C_CAP) >= 1, "A_CH, B_CH, A_SLOTS, B_SLOTS, C_CAP >= 1"),
            (all(getattr(p, f) in (0, 1) for f in flags), "A_DOM, A_SIGNED, A_FRAME, B_DOM, B_SIGNED, TRANS_B in 0, 1"),
            (0 <= p.NARROW < 9 + clog2(K), "0 <= NARROW < VW"),
        ]
        if all(holds for holds, _ in rules[:4]):                      # the rest needs the basic shape
            NG, W_K, XS = N_E // K, (M * N) // I, min(p.X_S, N_E)
            LB, NB = (K if p.TRANS_B else N), I // N
            rules += [
                (NG <= W_K and NG <= XS, "N_E/K <= W_K and N_E/K <= min(X_S, N_E)"),
                (K <= I and LB <= I, "K <= I and the B row length <= I"),
                (p.A_FRAME != 0 or (p.A_ROWS == 1 and p.A_CH <= NB and isPowerOfTwo(p.A_CH)),
                 "row frame: A_ROWS = 1, A_CH a power of two <= NB"),
                (p.A_FRAME != 1 or (p.A_ROWS >= 1 and p.A_ROWS % NB == 0 and M % p.A_ROWS == 0 and p.A_DOM == 1
                                    and p.A_ROWS * K <= 5 * 2 ** 16),
                 "window frame: A_ROWS a multiple of NB, M a multiple of A_ROWS, A_DOM = 1"),
            ]
        checkRules("dimm_tile_top", rules)

    # ------------------------------------------------------------------------------------------------------------
    def __init__(self, instance: str, params: DimmTileParams, *, nA: int, nB: int, nJobs: int,
                 withValues: bool = True, modes=NL_DPE_MODES, role: str | None = None):
        super().__init__(instance, params, withValues=withValues, role=role)
        p, d = self.p, self.d
        self.nA, self.nB, self.nJobs = nA, nB, nJobs
        if p.A_FRAME == 0 and nA > p.A_SLOTS * d.A_WPM:
            raise ValueError(f"{instance}: row-frame A port: more matrices than A_SLOTS (slots never wrap)")
        if nB > p.B_SLOTS * d.B_WPM:
            raise ValueError(f"{instance}: B port: more matrices than B_SLOTS (slots never wrap)")
        self.math = DimmTileMath(p, d) if withValues else None
        # the blocks (spec §4)
        R, C, I = p.R, p.C, d.I
        self.converterA = [Crossbar(R, C, p.BUF, p.P, mode="LOG", rowsUsed=p.K, colsUsed=p.K, modes=modes,
                                    withValues=withValues) for _ in range(p.A_CH)] if p.A_DOM == 0 else []
        self.converterB = [Crossbar(R, C, p.BUF, p.P, mode="LOG", rowsUsed=d.LB, colsUsed=d.LB, modes=modes,
                                    withValues=withValues) for _ in range(p.B_CH)] if p.B_DOM == 0 else []
        self.aWriter, self.bWriter = Serial(), Serial()
        self.aStaging = Buffer(2 if p.A_FRAME == 0 else 2 * p.A_CH, f"{instance}: A staging")
        self.bStaging = Buffer(2 if p.TRANS_B else 2 * p.B_CH,
                               f"{instance}: B staging" if p.TRANS_B else f"{instance}: B row register")
        self.aSlot = Buffer(p.A_SLOTS, f"{instance}: A slot")
        self.farm = CrossbarGroup(p.N_E, R, C, p.BUF, p.P, mode="EXP", rowsUsed=I, colsUsed=I, modes=modes,
                                  withValues=withValues)
        # what the steps record
        self.aLastWord, self.bLastWord = {}, {}    # input window -> its last word at the slices
        self.aEmitAt = []                          # A staged unit (group / window) -> its first write
        self.bWriteAt = []                         # B row (TRANS_B = 0) -> its write
        self.bPortTime = None                      # the last B row's first word at the port (TRANS_B = 0)
        self.aValidAt, self.bValidAt = {}, {}      # matrix -> the cycle it is valid in every slice
        self.taken = []                            # job -> JobOffer
        self.jobsGoAt = []                         # job -> t_go
        self.slicesIdleFrom = 0                    # the slices read idle from this cycle
        self.released = {}                         # A slot -> its releases so far
        # resources: the crossbars, and the A and B memories as the RTL declares them (the two kinds are added, as
        # they can have the same shape: QK at I = 256)
        self.resources.crossbars[("EXP", I, I)] = p.N_E
        if p.A_DOM == 0:
            self.resources.crossbars[("LOG", p.K, p.K)] += p.A_CH
        if p.B_DOM == 0:
            self.resources.crossbars[("LOG", d.LB, d.LB)] += p.B_CH
        aMemories, bMemories = d.XS // d.A_GRP, d.KS // d.B_GRP
        self.resources.memories[(p.A_SLOTS << (addressBits(d.T_W) + addressBits(d.A_GRP)), d.NB * 8)] += d.NS * aMemories
        self.resources.memories[(p.B_SLOTS << addressBits(d.B_GRP), p.N * 8)] += d.NS * bMemories

    # -- declaration --------------------------------------------------------------------------------------------
    def ports(self) -> dict:
        p, d = self.p, self.d
        return {
            "a": PortSpec("a", "in", "push+credit", 8, EPS, d.A_W, p.A_CH, "row group" if p.A_FRAME == 0 else "matrix",
                          CAP=d.CAP_A, signals=("a_d", "a_v", "a_cr")),
            "b": PortSpec("b", "in", "push+credit", 8, EPS, d.LB, p.B_CH, "matrix" if p.TRANS_B else "row",
                          CAP=d.CAP_B, signals=("b_d", "b_v", "b_cr")),
            "j": PortSpec("j", "in", "burst", d.SAW + d.SBW + 1, 1, 1, 1, "job", CAP=1,
                          signals=("j_a", "j_b", "j_ra", "j_v", "j_rdy"), control=True),
            "c": PortSpec("c", "out", "push+credit", d.EC, EPS, d.I, d.NG, "job output", CAP=p.C_CAP,
                          signals=("c_d", "c_v", "c_cr")),
        }

    def processes(self) -> dict:
        return {"a_in": self.nA, "b_in": self.nB, "take": self.nJobs, "jobs": self.nJobs}

    def jobMatrices(self, job: int) -> tuple:
        """Job's A and B matrices: a released A slot holds its oldest matrix not yet released."""
        offer = self.taken[job]
        aMatrix = offer.slotA + self.p.A_SLOTS * self.released.get(offer.slotA, 0) if self.p.A_FRAME == 1 \
            else offer.slotA
        return aMatrix, offer.slotB

    def need(self, process: str, unit: int) -> Need:
        p, d = self.p, self.d
        if process == "a_in":
            return Need(windows={"a": {unit % p.A_CH: 1}})
        if process == "b_in":
            return Need(windows={"b": {unit % p.B_CH: 1}})
        if process == "take":
            return Need(windows={"j": {0: 1}})
        if unit >= len(self.taken):
            return Need(after={"take": unit + 1})
        aMatrix, bMatrix = self.jobMatrices(unit)
        return Need(credits={"c": (unit - p.C_CAP) if unit >= p.C_CAP else None},
                    after={"a_in": (aMatrix + 1) * d.A_WPM, "b_in": (bMatrix + 1) * d.B_WPM})

    # -- the input side ------------------------------------------------------------------------------------------
    def convert(self, converters: list, channel: int, w) -> tuple:
        """(log bytes, stream time) of an input window: through the channel's converter (an int8 port), or as given
        (a log port). The stream time is the first word the converter drains (or the port word)."""
        if not converters:
            return (self.math.asLog(w.values) if self.math else None), w.time
        x = self.math.converterInput(w.values) if self.math else None
        xbarPass = converters[channel].run(x, w.time + CONV_IN_LAT)
        if xbarPass.events["load_start"] != w.time + CONV_IN_LAT:    # the port contract: a row per converter period
            raise ContractError(f"{self.instance}: converter {channel} overrun: a row at {w.time} needs its burst at "
                                f"{w.time + CONV_IN_LAT}, the crossbar is ready at {xbarPass.events['load_start']}")
        return (None if xbarPass.window is None else xbarPass.window.astype(np.int64)), xbarPass.time

    def stepLoadA(self, k: int, io) -> Step:
        p, d = self.p, self.d
        (w,) = io.windows("a")
        logBytes, tStream = self.convert(self.converterA, k % p.A_CH, w)
        if self.converterA:
            self.counts.passes[("LOG", p.K, p.K)] += 1
        first = tStream + DIST_LAT                                     # the window's first and last word at the slices
        last = first + ceilDiv(d.A_W, EPS) - 1
        self.aLastWord[k] = last
        if p.A_FRAME == 0:
            self._stageRowFrame(k, first, logBytes, io)
        else:
            self._stageWindowFrame(k, first, last, logBytes, io)
        return Step("a_in", k, w.time, tStream, last + 1, dict(t_port=w.time, t_stream=tStream, a_last=last))

    def _stageRowFrame(self, k: int, first: int, logBytes, io) -> None:
        """Row frame: row k joins the transposer's group of NB rows; a complete group is written as KS column words."""
        p, d = self.p, self.d
        matrix, row = divmod(k, p.M)
        if self.math:
            self.math.storeARow(matrix, row, logBytes)
        group = k // d.NB
        self.aStaging.put(group, first)
        if k % d.NB == d.NB - 1:
            complete = readyAt(*(self.aLastWord[x] for x in range(group * d.NB, k + 1)))
            tEmit = self.aWriter.take(complete + EMIT_LAT, d.KS)
            self.aEmitAt.append(tEmit)
            self.aStaging.release(group, tEmit + d.KS)
            io.giveCredit("a", group, tEmit + d.KS - 1 + CR_LAT)
            self.counts.memWriteBytes += p.K * d.NB
            if group % d.W_K == d.W_K - 1:
                self.aValidAt[matrix] = self.aWriter.freeAt            # the matrix's last write + 1

    def _stageWindowFrame(self, k: int, first: int, last: int, logBytes, io) -> None:
        """Window frame: window k of its matrix; the stager writes its EW words; the matrix holds an A slot."""
        p, d = self.p, self.d
        matrix, local = divmod(k, d.A_WPM)
        if self.math:
            self.math.storeAWindow(matrix, local * p.A_ROWS, logBytes)
        self.aSlot.put(matrix, first)
        self.aStaging.put(k, first)
        tEmit = self.aWriter.take(last + EMIT_LAT, d.EW)
        self.aEmitAt.append(tEmit)
        self.aStaging.release(k, tEmit + d.EW)
        self.counts.memWriteBytes += p.A_ROWS * p.K
        if local == d.A_WPM - 1:
            self.aValidAt[matrix] = self.aWriter.freeAt

    def stepLoadB(self, k: int, io) -> Step:
        p, d = self.p, self.d
        (w,) = io.windows("b")
        logBytes, tStream = self.convert(self.converterB, k % p.B_CH, w)
        if self.converterB:
            self.counts.passes[("LOG", d.LB, d.LB)] += 1
        first = tStream + DIST_LAT
        last = first + ceilDiv(d.LB, EPS) - 1
        self.bLastWord[k] = last
        if p.TRANS_B:
            self._stageTransposed(k, first, logBytes, io)
        else:
            self._stageRow(k, w.time, first, last, logBytes, io)
        return Step("b_in", k, w.time, tStream, last + 1, dict(t_port=w.time, t_stream=tStream, b_last=last))

    def _stageTransposed(self, k: int, first: int, logBytes, io) -> None:
        """TRANS_B = 1: row k of B^T is column n of its matrix; the transposer stages the matrix's N rows, then writes
        KS words."""
        p, d = self.p, self.d
        matrix, n = divmod(k, p.N)
        if self.math:
            self.math.storeBColumn(matrix, n, logBytes)
        self.bStaging.put(matrix, first)
        if n == p.N - 1:
            complete = readyAt(*(self.bLastWord[x] for x in range(matrix * p.N, k + 1)))
            tEmit = self.bWriter.take(complete + EMIT_LAT, d.KS)
            self.bStaging.release(matrix, tEmit + d.KS)
            io.giveCredit("b", matrix, tEmit + d.KS - 1 + CR_LAT)
            self.bValidAt[matrix] = self.bWriter.freeAt
            self.counts.memWriteBytes += p.K * p.N

    def _stageRow(self, k: int, tPort: int, first: int, last: int, logBytes, io) -> None:
        """TRANS_B = 0: row kk of its matrix, one B word written by the shared row writer, one row per cycle in row
        order. Port contract (spec §3): rows reach the port in row order, else later rows wait in their registers."""
        p = self.p
        if k and tPort < self.bPortTime:
            raise ContractError(f"{self.instance}: B row {k} at {tPort}, before row {k - 1} (at {self.bPortTime}); "
                                "a TRANS_B = 0 port takes its rows in row order")
        self.bPortTime = tPort
        matrix, kk = divmod(k, p.K)
        if self.math:
            self.math.storeBRow(matrix, kk, logBytes)
        self.bStaging.put(k, first)
        tWrite = self.bWriter.take(last + EMIT_LAT)
        self.bWriteAt.append(tWrite)
        self.bStaging.release(k, tWrite)                               # the register is read (written out) at tWrite
        io.giveCredit("b", k, tWrite + CR_LAT)
        self.counts.memWriteBytes += p.N
        if kk == p.K - 1:
            self.bValidAt[matrix] = self.bWriter.freeAt

    # -- jobs -----------------------------------------------------------------------------------------------------
    def stepTakeJob(self, k: int, io) -> Step:
        (w,) = io.windows("j")
        if self.jobsGoAt and w.time < self.jobsGoAt[-1]:
            raise ContractError(f"{self.instance}: job {k} offered at {w.time} while the job register is full")
        slotA, slotB, releaseA = (int(x) for x in w.values)
        if not (0 <= slotA < self.p.A_SLOTS and 0 <= slotB < self.p.B_SLOTS) or releaseA not in (0, 1):
            raise ContractError(f"{self.instance}: job {k} fields ({slotA}, {slotB}, {releaseA}) outside the slots")
        if releaseA and self.p.A_FRAME == 0:
            raise ContractError(f"{self.instance}: job {k} releases a row-frame A slot")
        self.taken.append(JobOffer(w.time, slotA, slotB, releaseA))
        return Step("take", k, w.time, w.time, w.time + 1)

    def stepRunJob(self, job: int, io) -> Step:
        p, d = self.p, self.d
        offer = self.taken[job]
        aMatrix, bMatrix = self.jobMatrices(job)
        credit = io.creditTime("c")
        # the registered decision (spec §5 Jobs): after the job register, both matrices valid, the slices idle, an
        # output credit
        tDecide = readyAt(offer.offeredAt + 1, self.aValidAt[aMatrix], self.bValidAt[bMatrix], self.slicesIdleFrom,
                          None if credit is None else credit + 1)
        tGo = tDecide + DEC_LAT
        self.jobsGoAt.append(tGo)
        io.giveCredit("j", job, tGo)
        if self.math:
            self.math.startJob()
        waves = []
        for wave in range(d.T_W):                   # one lock-step farm pass per wave
            feed = self.math.waveInput(aMatrix, bMatrix, wave) if self.math else None
            xbarPass = self.farm.run(feed, tGo + d.START_LAT)
            waves.append(xbarPass.events)
            sums = self.math.waveOutput(aMatrix, bMatrix, wave, xbarPass.out) if self.math else None
            tOut = xbarPass.events["drain_start"] + d.OUT_LAT         # the registered drain value + the tree levels
            for g in range(d.NG):
                io.emit("c", g, job, g + d.NG * wave, tOut, None if sums is None else self.math.narrowed(sums[g]))
        last = waves[-1]
        done = last["drain_start"] + d.OUT_LAT + d.LW - 1 + DONE_LAT
        self.slicesIdleFrom = last["drain_end"] + IDLE_LAT
        if self.math:
            self.math.finishJob(job)
        if offer.releaseA:                                             # release the A slot: its credit
            self.released[offer.slotA] = self.released.get(offer.slotA, 0) + 1
            self.aSlot.release(aMatrix, done + REL_LAT)
            io.giveCredit("a", aMatrix, done + REL_LAT)
        # counts (spec §7)
        self.counts.passes.update(self.farm.passes)
        self.farm.passes.clear()
        self.counts.clbAdd += p.K * p.M * p.N                          # the pairing adds: log a + log b per product
        self.counts.clbAdd += d.W_K * (p.K - 1) * d.I                  # the trees: K - 1 adds per output element
        if p.NARROW:
            self.counts.clbCompare += p.M * p.N                        # the narrowing sat8(C >> NARROW) per element
        self.counts.memReadBytes += p.K * p.M + p.K * p.N
        return Step("jobs", job, tDecide, tGo, done,
                    dict(t_in=offer.offeredAt, t_dec=tDecide, t_go=tGo, a_matrix=aMatrix, b_matrix=bMatrix,
                         done=done, t_idle=self.slicesIdleFrom, waves=[dict(e) for e in waves]))
