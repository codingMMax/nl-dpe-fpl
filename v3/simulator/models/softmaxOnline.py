"""softmaxOnline.py — behavior model of `softmax_online_top` (the online softmax), contract `ip/softmax/spec.md` (the
bit-plane pass, 2026-10-22: plane words on every port, a double corner-turn instead of the packer, one EXP burst per
E5 window, E6 and E7 as plane windows).

One execution gives the values, the cycle of every port event and the counts (docs/simulator.md §1). The model is the
blocks of spec §4, each a stage (simulator/core/stages.py) or a crossbar (simulator/core/crossbar.py):
  reader              the tile reader (Serial): one FIFO plane per cycle, a plane readable FIFO_LAT after it arrived;
                      a window's P core words (Q_W bytes each) leave the corner-turn from CT_LAT after its last plane
  exp                 the N_EXP EXP crossbars (I x I); carryExp, carryLog: the carry crossbars (S x S)
The core's timing is v2's recurrence on the bit-plane pass (spec §5), driven by the stream word times; its values go
through the crossbars (`SoftmaxMath`).
Processes (one step per unit):
  arm    a run's start pulse -> the arm time
  tile   an input tile: the gate, the reads, the corner-turn, s_cr
  blk    a block: the EXP passes (the lp windows), the row sums, the carry passes, the la window, the retire
  run    a run's end: the log l read-out (the ll window), done, the start port's credit (busy falling)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.core.cycleMath import clog2, isPowerOfTwo, readyAt
from simulator.core.errors import ContractError
from simulator.core.module import Module, Need, checkRules
from simulator.core.records import PortSpec, Step, planePort
from simulator.core.stages import Serial
from simulator.core.trace import label
from simulator.core.crossbar import NL_DPE_MODES, Crossbar

# named latencies (spec §5)
FIFO_LAT = 1        # an input plane valid at t can be read from t + 1
CR_IN_LAT = 1       # a credit pulse at t counts for the gates from t + 1
RET_LAT = 1         # a block retired at t counts for the store-slot gate from t + 1
ARM_LAT = 2         # start taken at t: the run is armed from t + 2
OPEN_LAT = 1        # the reader opens a tile one cycle after its decision
CT_LAT = 2          # a window's last plane read at r -> its first core word at r + 2 (registered read, corner-turn)
SCR_LAT = 1         # s_cr one cycle after a tile's last read
MAX_REG = 2         # MAX_LAT = log2 min(Q_W, BKV) + MAX_REG: the input register, the running max with m
FEED_REG = 2        # FEED_LAT = MAX_LAT + P + FEED_REG: the registered read, the registered s - m_new and clamp
SP_LAT = 3          # the last EXP drain plane -> the row sums (count register, weighted add, sum register)
WAIT_LAT = 1        # the registered carry decision
GO_LAT = 2          # a carry input from the tables (registered), then its plane register
E_LAT = 1           # the carry-EXP's last drain plane -> e whole
LA_LAT = 1          # m_new -> the E6 window's first plane (the registered m - m_new)
E7_LAT = 2          # the log l read-out decided at t -> its first plane at t + 2
DONE_LAT = 1        # the last log l plane -> done
GO_CR_LAT = 1       # done -> busy falls (the start port's credit)
LP_FLOOR = -128     # log P = max(s - m_new, LP_FLOOR) (spec §1)


@dataclass(kw_only=True)
class SoftmaxOnlineParams:
    """The RTL parameters of softmax_online_top (RTL names and defaults); N defaults to S."""
    S: int
    BKV: int
    N: int | None = None
    R: int = 64
    C: int = 64
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
    Q_W: int          # bytes per core stream word, I / 8 (a window is P words)
    STREAM: int       # core stream words per block, P NWB
    T_W: int          # tile windows per channel, NWB / S_CH
    DF: int           # the input FIFO's depth in plane words (a power of two >= 2 T_W P)
    NB: int           # banks of the log P store, max(2, N_EXP)
    DW: int           # bank depth in windows
    BDEPTH: int       # bank depth in words of Q_W bytes, DW P
    NWORDS: int       # plane words per tile, NWB P
    MAX_LAT: int      # the block's last core word -> every m_new, log2 min(Q_W, BKV) + MAX_REG
    FEED_LAT: int     # the block's last core word -> its EXP windows ready, MAX_LAT + P + FEED_REG


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
    BASIS = "model-only"           # the bit-plane pass (2026-10-22); the 40-bit version was GATE 2-pinned 2026-10-14
    STEP_METHODS = {"arm": "stepArm", "tile": "stepTile", "blk": "stepBlock", "run": "stepRunEnd"}

    @staticmethod
    def derive(p: SoftmaxOnlineParams) -> SoftmaxOnlineDerived:
        S, BKV, N = p.S, p.BKV, p.N
        I = min(p.R, p.C)
        BLKEL = S * BKV
        NWB = BLKEL // I
        B = N // BKV
        T_W = NWB // p.S_CH if p.S_CH else 0
        NB = max(2, p.N_EXP)
        DW = (2 * NWB) // NB if 2 * NWB >= NB else 1
        Q_W = I // 8
        maxLat = clog2(min(Q_W, BKV)) + MAX_REG
        return SoftmaxOnlineDerived(
            I=I, B=B, LN=clog2(N), BLKEL=BLKEL, NWB=NWB, PW=B * NWB, Q_W=Q_W, STREAM=p.P * NWB, T_W=T_W,
            DF=1 << clog2(2 * T_W * p.P), NB=NB, DW=DW, BDEPTH=DW * p.P, NWORDS=NWB * p.P, MAX_LAT=maxLat,
            FEED_LAT=maxLat + p.P + FEED_REG)

    @staticmethod
    def checkDomain(p: SoftmaxOnlineParams) -> None:
        S, BKV, N = p.S, p.BKV, p.N
        I = min(p.R, p.C)
        rules = [
            (min(p.N_EXP, p.S_CH, p.LP_CAP, p.LA_CAP, p.LL_CAP) >= 1, "N_EXP, S_CH, LP_CAP, LA_CAP, LL_CAP >= 1"),
            (all(isPowerOfTwo(x) for x in (S, BKV, N, I, p.N_EXP, p.S_CH)), "S, BKV, N, I, N_EXP, S_CH powers of two"),
            (8 <= S <= I, "8 <= S <= I"),
            (8 <= BKV <= min(N, I), "8 <= BKV <= min(N, I)"),
            (p.P == 8, "P = 8"),
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
        self.carryExtent = S if carryExtent is None else carryExtent     # the carry passes' extent (spec §1: S)
        self.exp = [Crossbar(R, C, p.P, mode="EXP", rowsUsed=I, colsUsed=I, modes=modes, withValues=withValues)
                    for _ in range(p.N_EXP)]
        self.carryExp = Crossbar(R, C, p.P, mode="EXP", rowsUsed=self.carryExtent, colsUsed=self.carryExtent,
                                 modes=modes, withValues=withValues)
        self.carryLog = Crossbar(R, C, p.P, mode="LOG", rowsUsed=self.carryExtent, colsUsed=self.carryExtent,
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
        self.resources.memories[(d.DF, I)] = p.S_CH
        self.resources.memories[(d.BDEPTH, I)] += d.NB                  # += : the same shape as a FIFO can occur

    # -- declaration --------------------------------------------------------------------------------------------
    def ports(self) -> dict:
        p, d = self.p, self.d
        return {
            "s": planePort("s", "in", "push+credit", 8, d.I, p.S_CH, "tile", p.P, CAP=2,
                           signals=("s_d", "s_v", "s_cr")),
            "go": PortSpec("go", "in", "burst", 1, 1, 1, 1, "run", CAP=1, signals=("start", "busy")),
            "lp": planePort("lp", "out", "push+credit", 8, d.I, p.N_EXP, "block", p.P, CAP=p.LP_CAP,
                            signals=("lp_d", "lp_v", "lp_cr")),
            "la": planePort("la", "out", "push+credit", 9, p.S, 1, "block b >= 1", p.P, CAP=p.LA_CAP,
                            signals=("la_d", "la_v", "la_cr")),
            "ll": planePort("ll", "out", "push+credit", 8, p.S, 1, "run", p.P, CAP=p.LL_CAP,
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
        # the reads: from the cycle after the decision, one plane per cycle once the plane is in its FIFO
        reads = []
        tOpen = tDecide + OPEN_LAT
        with label("stream"):                   # the score planes: plane c of window w in its FIFO, read one per cycle
            for i in range(d.NWORDS):
                w, c = divmod(i, p.P)
                reads.append(self.reader.take(readyAt(arrivals[w] + c + FIFO_LAT, tOpen if i == 0 else None)))
        # the corner-turn: window w's P core words from CT_LAT after its last plane's read, one per cycle
        stream = []
        for w in range(d.NWB):
            tWord = reads[w * p.P + p.P - 1] + CT_LAT
            with label("stream"):
                stream += [tWord + c for c in range(p.P)]
        self.readTimes[tile] = reads
        io.giveCredit("s", tile, reads[-1] + SCR_LAT)
        self.streamTimes[tile], self.lastStreamWord[tile] = stream, stream[-1]
        if self.math:
            self.math.storeTile(tile, windows)
        self.counts.fifoWords += d.NWORDS
        fifoBytes = d.NWORDS * d.Q_W                                   # FIFO traffic: each plane written once, read once
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
            xbarPass = self.exp[lane].run(math.expInput(w) if math else None, lastStreamWord + d.FEED_LAT)
            passes.append(xbarPass.events)
            logPWindows.append((lane, w, xbarPass.events["load_start"], math.logPWindow(w) if math else None))
            if math:
                math.collectExp(w, xbarPass.out)
        # the row sums are complete with the block's last drain plane (+ SP_LAT); the carry waits for them, the row
        # maxima and the previous block's retire
        with label("dpe_drain"):                                     # each pass's last drain plane
            lastDrain = [e["drain_start"] + p.P - 1 for e in passes]
        sumsDone = readyAt(*lastDrain) + SP_LAT
        wait = readyAt(sumsDone, lastStreamWord + d.MAX_LAT, self.retireAt[block - 1] if b else None) + WAIT_LAT
        carryExpLast = carryExpEvents = None
        if b == 0:
            logReady = wait + GO_LAT
        else:
            xbarPass = self.carryExp.run(math.carryExpInput(run) if math else None, wait + GO_LAT)
            carryExpEvents = xbarPass.events
            with label("dpe_drain"):                                 # the carry-EXP's drain planes
                carryExpLast = xbarPass.events["drain_start"] + p.P - 1
            logReady = carryExpLast + E_LAT + GO_LAT
            if math:
                math.collectCarryExp(xbarPass.out)
        xbarPass = self.carryLog.run(math.carryLogInput(b) if math else None, logReady)
        with label("dpe_drain"):                                     # the carry-LOG's drain planes into the log l table
            retire = xbarPass.events["drain_start"] + p.P - 1
        self.retireAt[block] = retire
        if math:
            math.finishBlock(run, xbarPass.window)
        return dict(lp=logPWindows, a_last=lastStreamWord, sp_done=sumsDone, wait=wait, cexp_last=carryExpLast,
                    retire=retire, passes=passes, carry_exp=carryExpEvents, carry_log=xbarPass.events)

    def stepBlock(self, block: int, io) -> Step:
        p, d = self.p, self.d
        lastWord = self.lastStreamWord[block]
        o = self.core(block, lastWord)
        for lane, w, t, values in o["lp"]:
            io.emit("lp", lane, block, w, t, values)
        unit = self.alphaUnit(block)
        if unit is not None:                                           # the block's log a: one window of S rows
            io.emit("la", 0, unit, 0, lastWord + d.MAX_LAT + LA_LAT,
                    self.math.blockLogAlpha().copy() if self.math else None)
        for crossbar in self.exp + [self.carryExp, self.carryLog]:
            self.counts.passes.update(crossbar.passes)
            crossbar.passes.clear()
        elements = p.S * p.BKV
        self.counts.clbCompare += elements + p.S
        self.counts.clbAdd += 2 * elements + 2 * p.S
        self.counts.clbSelect += elements + 2 * p.S
        self.counts.memWriteBytes += elements
        self.counts.memReadBytes += elements
        return Step("blk", block, lastWord, lastWord + d.FEED_LAT, o["retire"] + 1,
                    {key: o[key] for key in ("a_last", "sp_done", "wait", "cexp_last", "retire", "passes", "carry_exp",
                                             "carry_log")})

    def stepRunEnd(self, run: int, io) -> Step:
        p, d = self.p, self.d
        coreDone = self.retireAt[(run + 1) * d.B - 1] + 1               # the core's done
        credit = io.creditTime("ll")
        tD7 = readyAt(coreDone, None if credit is None else credit + CR_IN_LAT)   # the log l read-out: an ll credit
        tE7 = tD7 + E7_LAT
        io.emit("ll", 0, run, 0, tE7, self.math.logL[run].copy() if self.math else None)
        with label("stream"):                                        # the ll window's planes
            lastLl = tE7 + p.P - 1
        done = lastLl + DONE_LAT
        self.coreDoneAt[run], self.doneAt[run] = coreDone, done
        io.giveCredit("go", run, done + GO_CR_LAT)
        if self.math:
            self.math.finishRun(run)
        return Step("run", run, coreDone, tD7, done + 1, dict(core_done=coreDone, t_d7=tD7, t_e7=tE7, done=done))
