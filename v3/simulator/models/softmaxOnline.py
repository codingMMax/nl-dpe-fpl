"""softmaxOnline.py — behavior model of `softmax_online_top` (the online softmax), contract `ip/softmax/spec.md`.

One execution gives the values, the cycle of every port event and the counts (docs/simulator.md §1). The model is the
blocks of spec §4, each a stage (simulator/core/stages.py) or a crossbar (simulator/core/crossbar.py):
  reader              the tile reader (Serial): one FIFO word per cycle, a word readable FIFO_LAT after it arrived
  packer              window words -> the core's stream words (Repack)
  exp                 the N_EXP EXP crossbars (I x I); carryExp, carryLog: the carry crossbars (S x S)
The core's timing is the v2 recurrence (v2 `softmax_online_sim.softmax_online_cycle_model`, RTL-pinned) driven by the
stream word times instead of a fixed start; its values go through the crossbars (`SoftmaxMath`).
Processes (one step per unit):
  arm    a run's start pulse -> the arm time
  tile   an input tile: the gate, the reads, the packer, s_cr
  blk    a block: the EXP passes (the lp windows), the row sums, the carry passes, the la rows, the retire
  run    a run's end: the log l read-out (the ll window), done, the start port's credit (busy falling)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.core.cycleMath import ceilDiv, clog2, isPowerOfTwo, readyAt
from simulator.core.errors import ContractError
from simulator.core.module import Module, Need, checkRules
from simulator.core.records import PortSpec, Step
from simulator.core.stages import Repack, Serial
from simulator.core.crossbar import NL_DPE_MODES, Crossbar

# named latencies (spec §5)
FIFO_LAT = 1        # an input word valid at t can be read from t + 1
CR_IN_LAT = 1       # a credit pulse at t counts for the gates from t + 1
RET_LAT = 1         # a block retired at t counts for the store-slot gate from t + 1
ARM_LAT = 2         # start taken at t: the run is armed from t + 2
OPEN_LAT = 1        # the reader opens a tile one cycle after its decision
PACK_LAT = 2        # a read at r -> its completed stream word at the core at r + 2
SCR_LAT = 1         # s_cr one cycle after a tile's last read
MAX_LAT, FEED_LAT, SP_LAT, WAIT_LAT, GO_LAT = 4, 9, 2, 1, 4     # the v2 core
LA_EV_LAT = 5       # the stream word completing a row -> the row's log a on la_v
E7_LAT = 3          # the log l read-out decided at t -> its first word at t + 3
DONE_LAT = 1        # the last log l word -> done
GO_CR_LAT = 1       # done -> busy falls (the start port's credit)
EPS = 5             # int8 elements per 40-bit port word
LP_FLOOR = -128     # log P = max(s - m_new, LP_FLOOR) (spec §1)


@dataclass(kw_only=True)
class SoftmaxOnlineParams:
    """The RTL parameters of softmax_online_top (RTL names and defaults); N defaults to S."""
    S: int
    BKV: int
    N: int | None = None
    R: int = 64
    C: int = 64
    BUF: int = 40
    P: int = 8
    N_EXP: int = 2
    S_CH: int = 1
    LP_CAP: int = 2
    LA_CAP: int = 2
    LL_CAP: int = 1

    def __post_init__(self):
        if self.N is None:
            self.N = self.S


@dataclass(frozen=True)
class SoftmaxOnlineDerived:
    """The RTL's localparams."""
    I: int            # min(R, C): the window size
    B: int            # key blocks per run, N / BKV
    LN: int           # clog2 N: the log l scale
    BLKEL: int        # elements of a block, S BKV
    NWB: int          # windows per block, BLKEL / I
    PW: int           # windows per run, B NWB
    LW: int           # words per window, ceil(8 I / BUF)
    L_LAST: int       # elements of a window's last word
    STREAM: int       # stream words per block, ceil(BLKEL / EPS)
    T_W: int          # tile windows per channel, NWB / S_CH
    DF: int           # the input FIFO's depth (a power of two >= 2 T_W LW)
    WL: int           # words of the carry walk, ceil(S / EPS)
    NB: int           # banks of the log P store, max(2, N_EXP)
    WPW: int          # words per window (LW)
    DW: int           # bank depth in windows
    BDEPTH: int       # bank depth in words, DW LW
    NWORDS: int       # words per tile, NWB LW
    RK: tuple         # the stream word that completes row m, per row


class SoftmaxMath:
    """The values of softmax_online_top: a run's score tiles, the block maxima and the carry (m, log l), the log P
    scores the EXP crossbars take, the row sums, the carry crossbars' inputs and the log l read-out. `stages` keeps
    every stage per run (GATE 1: softmax_online_ref.logcarry_stage_values)."""

    def __init__(self, p: SoftmaxOnlineParams, d: SoftmaxOnlineDerived):
        self.p, self.d = p, d
        self.scores = {}                           # block -> [S, BKV] int8 scores
        self.runningMax, self.logL = {}, {}        # run -> m, log l after the last block so far
        self.stages = {}                           # run -> {stage: [B, S] / [B, S, BKV]}
        self.block = None                          # the block in flight: its intermediate values

    def storeTile(self, block: int, windows: list) -> None:
        flat = np.concatenate([np.asarray(w.values, dtype=np.int64) for w in windows])
        if flat.min(initial=0) < -128 or flat.max(initial=0) > 127:
            raise ValueError(f"tile {block} carries scores outside int8")
        self.scores[block] = flat.reshape(self.p.S, self.p.BKV)

    def startBlock(self, block: int, run: int, b: int) -> None:
        """The block max, the new running max, log alpha and the log P scores (spec §1)."""
        s = self.scores[block]
        mBlock = s.max(axis=1)
        mNew = mBlock if b == 0 else np.maximum(self.runningMax[run], mBlock)
        logAlpha = np.zeros(self.p.S, dtype=np.int64) if b == 0 else self.runningMax[run] - mNew
        logP = np.maximum(s - mNew[:, None], LP_FLOOR)
        self.block = dict(m_blk=mBlock, m_new=mNew, log_a=logAlpha, logp=logP, flat=logP.reshape(-1),
                          pBytes=np.zeros(self.d.BLKEL, dtype=np.int64))

    def expInput(self, w: int) -> np.ndarray:
        I = self.d.I
        X = np.zeros(self.p.R, dtype=np.int8)
        X[:I] = self.block["flat"][w * I:(w + 1) * I].astype(np.int8)
        return X

    def logPWindow(self, w: int) -> np.ndarray:
        I = self.d.I
        return self.block["flat"][w * I:(w + 1) * I].copy()

    def collectExp(self, w: int, out: np.ndarray) -> None:
        I = self.d.I
        self.block["pBytes"][w * I:(w + 1) * I] = out[:I].astype(np.int64)

    def carryExpInput(self, run: int) -> np.ndarray:
        """t = clamp8(log a + log l): the carry EXP input (b >= 1)."""
        t = np.clip(self.block["log_a"] + self.logL[run], -128, 127)
        self.block["t"] = t
        X = np.zeros(self.p.R, dtype=np.int8)
        X[:self.p.S] = t.astype(np.int8)
        return X

    def collectCarryExp(self, out: np.ndarray) -> None:
        self.block["e"] = out[:self.p.S].astype(np.int64)

    def carryLogInput(self, b: int) -> np.ndarray:
        """lq = min(l' >> LN, 127) with l' = (e << LN) + row sums (b >= 1) or the row sums (b = 0): the LOG input."""
        p, d, blk = self.p, self.d, self.block
        sp = blk["pBytes"].reshape(p.S, p.BKV).sum(axis=1)
        if b == 0:
            blk["t"], blk["e"] = np.zeros(p.S, dtype=np.int64), np.zeros(p.S, dtype=np.int64)
            lprime = sp
        else:
            lprime = (blk["e"] << d.LN) + sp
        lq = np.minimum(lprime >> d.LN, 127)
        blk.update(sp=sp, lprime=lprime, lq=lq)
        X = np.zeros(p.R, dtype=np.int8)
        X[:p.S] = lq.astype(np.int8)
        return X

    def finishBlock(self, run: int, window: np.ndarray) -> None:
        blk = self.block
        logL = window[:self.p.S].astype(np.int64)
        self.runningMax[run], self.logL[run] = blk["m_new"], logL
        stages = self.stages.setdefault(run, {})
        for key in ("m_blk", "m_new", "log_a", "logp", "sp", "t", "e", "lprime", "lq"):
            stages.setdefault(key, []).append(blk[key])
        stages.setdefault("log_l", []).append(logL)

    def blockLogAlpha(self) -> np.ndarray:
        return self.block["log_a"]

    def finishRun(self, run: int) -> None:
        stages = self.stages[run]
        for key in list(stages):
            stages[key] = np.stack(stages[key])
        stages["m"], stages["log_l_final"] = self.runningMax[run], self.logL[run]


class SoftmaxOnline(Module):
    NAME = "softmax_online_top"
    PARAMS_CLASS = SoftmaxOnlineParams
    BASIS = "rtl-pinned"           # GATE 2 2026-10-14: 18/18 corpus + 5/5 bring-up on 64x64, every event equal
    STEP_METHODS = {"arm": "stepArm", "tile": "stepTile", "blk": "stepBlock", "run": "stepRunEnd"}

    @staticmethod
    def derive(p: SoftmaxOnlineParams) -> SoftmaxOnlineDerived:
        S, BKV, N = p.S, p.BKV, p.N
        I = min(p.R, p.C)
        BLKEL = S * BKV
        NWB = BLKEL // I
        B = N // BKV
        LW = ceilDiv(I * 8, p.BUF)
        T_W = NWB // p.S_CH if p.S_CH else 0
        NB = max(2, p.N_EXP)
        DW = (2 * NWB) // NB if 2 * NWB >= NB else 1
        return SoftmaxOnlineDerived(
            I=I, B=B, LN=clog2(N), BLKEL=BLKEL, NWB=NWB, PW=B * NWB, LW=LW, L_LAST=I - EPS * (LW - 1),
            STREAM=ceilDiv(BLKEL, EPS), T_W=T_W, DF=1 << clog2(2 * T_W * LW), WL=ceilDiv(S, EPS), NB=NB, WPW=LW,
            DW=DW, BDEPTH=DW * LW, NWORDS=NWB * LW, RK=tuple(((m + 1) * BKV - 1) // EPS for m in range(S)))

    @staticmethod
    def checkDomain(p: SoftmaxOnlineParams) -> None:
        S, BKV, N = p.S, p.BKV, p.N
        I = min(p.R, p.C)
        rules = [
            (min(p.N_EXP, p.S_CH, p.LP_CAP, p.LA_CAP, p.LL_CAP) >= 1, "N_EXP, S_CH, LP_CAP, LA_CAP, LL_CAP >= 1"),
            (all(isPowerOfTwo(x) for x in (S, BKV, N, I, p.N_EXP, p.S_CH)), "S, BKV, N, I, N_EXP, S_CH powers of two"),
            (8 <= S <= I, "8 <= S <= I"),
            (8 <= BKV <= min(N, I), "8 <= BKV <= min(N, I)"),
            (p.BUF == 40, "BUF = 40"),
        ]
        if all(holds for holds, _ in rules):
            if S * BKV < I:
                rules.append((False, "S*BKV >= I (whole windows)"))
            else:
                NWB, B = S * BKV // I, N // BKV
                rules += [(p.S_CH <= NWB, "S_CH <= NWB"), ((B * NWB) % p.N_EXP == 0, "B*NWB a multiple of N_EXP")]
        checkRules("softmax_online_top", rules)

    # ------------------------------------------------------------------------------------------------------------
    def __init__(self, instance: str, params: SoftmaxOnlineParams, *, runs: int, withValues: bool = True,
                 modes=NL_DPE_MODES, carryExtent: int | None = None, role: str | None = None):
        super().__init__(instance, params, withValues=withValues, role=role)
        p, d = self.p, self.d
        self.runs = int(runs)
        self.math = SoftmaxMath(p, d) if withValues else None
        # the blocks (spec §4)
        R, C, I, S = p.R, p.C, d.I, p.S
        self.reader = Serial()
        self.packer = Repack(([EPS] * (d.LW - 1) + [d.L_LAST]) * d.NWB, EPS, PACK_LAT)
        self.carryExtent = S if carryExtent is None else carryExtent     # the carry passes' extent (spec §1: S)
        self.exp = [Crossbar(R, C, p.BUF, p.P, mode="EXP", rowsUsed=I, colsUsed=I, modes=modes, withValues=withValues)
                    for _ in range(p.N_EXP)]
        self.carryExp = Crossbar(R, C, p.BUF, p.P, mode="EXP", rowsUsed=self.carryExtent, colsUsed=self.carryExtent,
                                 modes=modes, withValues=withValues)
        self.carryLog = Crossbar(R, C, p.BUF, p.P, mode="LOG", rowsUsed=self.carryExtent, colsUsed=self.carryExtent,
                                 modes=modes, withValues=withValues)
        # what the steps record
        self.startAt, self.armedAt = {}, {}        # run -> the start pulse, the arm time
        self.readTimes, self.streamTimes = {}, {}  # tile -> its reads, its stream words at the core
        self.lastStreamWord = {}                   # tile -> its last stream word
        self.retireAt = {}                         # block -> retire cycle (the LOG drain word with row S - 1)
        self.coreDoneAt, self.doneAt = {}, {}      # run -> the core's done, done
        self.resources.crossbars[("EXP", I, I)] += p.N_EXP
        self.resources.crossbars[("EXP", self.carryExtent, self.carryExtent)] += 1
        self.resources.crossbars[("LOG", self.carryExtent, self.carryExtent)] += 1
        self.resources.memories[(d.DF, 40)] = p.S_CH
        self.resources.memories[(d.BDEPTH, 8)] = d.NB * EPS

    # -- declaration --------------------------------------------------------------------------------------------
    def ports(self) -> dict:
        p, d = self.p, self.d
        return {
            "s": PortSpec("s", "in", "push+credit", 8, EPS, d.I, p.S_CH, "tile", CAP=2, signals=("s_d", "s_v", "s_cr")),
            "go": PortSpec("go", "in", "burst", 1, 1, 1, 1, "run", CAP=1, signals=("start", "busy")),
            "lp": PortSpec("lp", "out", "push+credit", 8, EPS, d.I, p.N_EXP, "block", CAP=p.LP_CAP,
                           signals=("lp_d", "lp_v", "lp_cr")),
            "la": PortSpec("la", "out", "push+credit", 9, 1, 1, 1, "block b >= 1", CAP=p.LA_CAP,
                           signals=("la_d", "la_v", "la_cr")),
            "ll": PortSpec("ll", "out", "push+credit", 8, 1, p.S, 1, "run", CAP=p.LL_CAP,
                           signals=("ll_d", "ll_v", "ll_cr")),
        }

    def processes(self) -> dict:
        n = self.runs * self.d.B
        return {"arm": self.runs, "tile": n, "blk": n, "run": self.runs}

    def alphaUnit(self, block: int) -> int | None:
        """The la unit of a block (blocks b >= 1 of each run), or None for a run's first block."""
        run, b = divmod(block, self.d.B)
        return None if b == 0 else run * (self.d.B - 1) + b - 1

    def need(self, process: str, unit: int) -> Need:
        p, d = self.p, self.d
        if process == "arm":
            return Need(windows={"go": {0: 1}})
        if process == "tile":
            run, b = divmod(unit, d.B)
            u = self.alphaUnit(unit)
            return Need(windows={"s": Need.eachChannel(p.S_CH, d.T_W)},
                        credits={"lp": (unit - p.LP_CAP) if unit >= p.LP_CAP else None,
                                 "la": (u - p.LA_CAP) if (u is not None and u >= p.LA_CAP) else None},
                        after={"arm": run + 1, "blk": (unit - 1) if b >= 2 else 0})
        if process == "blk":
            return Need(after={"tile": unit + 1})
        return Need(credits={"ll": (unit - p.LL_CAP) if unit >= p.LL_CAP else None},
                    after={"blk": (unit + 1) * d.B})

    # -- steps --------------------------------------------------------------------------------------------------
    def stepArm(self, run: int, io) -> Step:
        (w,) = io.windows("go")
        if run and w.time < self.doneAt[run - 1] + GO_CR_LAT:
            raise ContractError(f"{self.instance}: run {run} started at {w.time} while busy (until {self.doneAt[run - 1]})")
        self.startAt[run], self.armedAt[run] = w.time, w.time + ARM_LAT
        return Step("arm", run, w.time, w.time + ARM_LAT, w.time + ARM_LAT)

    def stepTile(self, tile: int, io) -> Step:
        p, d = self.p, self.d
        run, b = divmod(tile, d.B)
        windows = io.windows("s")
        arrivals = [w.time for w in windows]
        lp, la = io.creditTime("lp"), io.creditTime("la")
        # the tile gate (spec §5): the reader idle, the run armed, the first word in FIFO 0, the store slot (block
        # b - 2 retired), a log P credit, an alpha credit
        tDecide = readyAt(self.reader.freeAt, self.armedAt[run], arrivals[0] + FIFO_LAT,
                          self.retireAt[tile - 2] + RET_LAT if b >= 2 else None,
                          None if lp is None else lp + CR_IN_LAT, None if la is None else la + CR_IN_LAT)
        # the reads: from the cycle after the decision, one word per cycle once the word is in its FIFO
        reads = []
        for i in range(d.NWORDS):
            w, c = divmod(i, d.LW)
            reads.append(self.reader.take(readyAt(arrivals[w] + c + FIFO_LAT, tDecide + OPEN_LAT if i == 0 else None)))
        self.readTimes[tile] = reads
        io.giveCredit("s", tile, reads[-1] + SCR_LAT)
        stream = self.packer.times(reads)                              # the core's stream words
        self.streamTimes[tile], self.lastStreamWord[tile] = stream, stream[-1]
        if self.math:
            self.math.storeTile(tile, windows)
        self.counts.fifoWords += d.NWORDS
        fifoBytes = d.NWORDS * ceilDiv(p.BUF, 8)                       # FIFO traffic: each word written once, read once
        self.counts.memWriteBytes += fifoBytes
        self.counts.memReadBytes += fifoBytes
        return Step("tile", tile, tDecide, int(reads[0]), int(reads[-1]) + 1,
                    dict(t_dec=tDecide, r0=int(reads[0]), r_last=int(reads[-1]), a_first=int(stream[0]),
                         a_last=int(stream[-1])))

    def core(self, block: int, lastStreamWord: int) -> dict:
        """The v2 core for one block of the stream (b = block mod B), whose last stream word reaches the core at
        `lastStreamWord`: the EXP passes, the row sums, the carry passes, the retire. Returns the events and the lp
        windows. Shared by the block step and the closed-form test."""
        p, d, math = self.p, self.d, self.math
        run, b = divmod(block, d.B)
        if math:
            math.startBlock(block, run, b)
        passes, logPWindows = [], []
        for w in range(d.NWB):
            lane = (b * d.NWB + w) % p.N_EXP
            xbarPass = self.exp[lane].run(math.expInput(w) if math else None, lastStreamWord + FEED_LAT)
            passes.append(xbarPass.events)
            logPWindows.append((lane, w, xbarPass.events["load_start"], math.logPWindow(w) if math else None))
            if math:
                math.collectExp(w, xbarPass.out)
        # the row sums are complete with the block's last drain word (+ SP_LAT); the carry waits for them, the row
        # maxima and the previous block's retire
        sumsDone = readyAt(*(e["drain_start"] + d.LW - 1 for e in passes)) + SP_LAT
        wait = readyAt(sumsDone, lastStreamWord + MAX_LAT, self.retireAt[block - 1] if b else None) + WAIT_LAT
        walk = d.WL                          # the walk's words; the carry drain word with row S - 1 is word WL - 1
        carryExpLast = None
        if b == 0:
            logReady = wait + walk + GO_LAT
        else:
            xbarPass = self.carryExp.run(math.carryExpInput(run) if math else None, wait + GO_LAT)
            carryExpLast = xbarPass.events["drain_start"] + walk - 1
            logReady = carryExpLast + GO_LAT
            if math:
                math.collectCarryExp(xbarPass.out)
        xbarPass = self.carryLog.run(math.carryLogInput(b) if math else None, logReady)
        retire = xbarPass.events["drain_start"] + walk - 1
        self.retireAt[block] = retire
        if math:
            math.finishBlock(run, xbarPass.window)
        return dict(lp=logPWindows, a_last=lastStreamWord, sp_done=sumsDone, wait=wait, cexp_last=carryExpLast,
                    retire=retire, passes=passes)

    def stepBlock(self, block: int, io) -> Step:
        p, d = self.p, self.d
        lastWord = self.lastStreamWord[block]
        o = self.core(block, lastWord)
        for lane, w, t, values in o["lp"]:
            io.emit("lp", lane, block, w, t, values)
        unit = self.alphaUnit(block)
        if unit is not None:
            logAlpha = self.math.blockLogAlpha() if self.math else None
            for m in range(p.S):
                io.emit("la", 0, unit, m, int(self.streamTimes[block][d.RK[m]]) + LA_EV_LAT,
                        None if logAlpha is None else np.array([logAlpha[m]], dtype=np.int64))
        for crossbar in self.exp + [self.carryExp, self.carryLog]:
            self.counts.passes.update(crossbar.passes)
            crossbar.passes.clear()
        elements = p.S * p.BKV
        self.counts.clbCompare += elements + p.S
        self.counts.clbAdd += 2 * elements + 2 * p.S
        self.counts.clbSelect += elements + 2 * p.S
        self.counts.memWriteBytes += elements
        self.counts.memReadBytes += elements
        return Step("blk", block, lastWord, lastWord + FEED_LAT, o["retire"] + 1,
                    {key: o[key] for key in ("a_last", "sp_done", "wait", "cexp_last", "retire", "passes")})

    def stepRunEnd(self, run: int, io) -> Step:
        p, d = self.p, self.d
        coreDone = self.retireAt[(run + 1) * d.B - 1] + 1               # the core's done
        credit = io.creditTime("ll")
        tD7 = readyAt(coreDone, None if credit is None else credit + CR_IN_LAT)   # the log l read-out: an ll credit
        tE7 = tD7 + E7_LAT
        io.emit("ll", 0, run, 0, tE7, self.math.logL[run].copy() if self.math else None)
        done = tE7 + p.S - 1 + DONE_LAT
        self.coreDoneAt[run], self.doneAt[run] = coreDone, done
        io.giveCredit("go", run, done + GO_CR_LAT)
        if self.math:
            self.math.finishRun(run)
        return Step("run", run, coreDone, tD7, done + 1, dict(core_done=coreDone, t_d7=tD7, t_e7=tE7, done=done))
