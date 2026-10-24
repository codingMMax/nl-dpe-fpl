"""softmaxAccumulator.py — behavior model of `softmax_online_acc` (the online softmax's accumulator), contract
`ip/softmax/acc_spec.md` (the bit-plane pass, 2026-10-22: the LOG feed corner-turns, the EXP feeds add bit-serially;
lo_mem and e_mem hold plane words; E9 is one plane window per epilogue wave; the trims of 2026-10-23: E8 carries value
words read straight into the LOG arithmetic, and single feed buffers space a group's jobs by LOG_GAP / EXP_GAP).

One execution gives the values, the cycle of every port event and the counts (docs/simulator.md §1). The model is the
blocks of spec §4, two crossbar groups (simulator/core/crossbar.py):
  logGroup, expGroup   the LOG and EXP groups: N_OC identity crossbars each, in lock-step from crossbar 0 (I x I)
A job is decided at the first cycle it is ready from which its pass, PRE_LOG / PRE_EXP cycles later, finds the
group's leader ready, and at least LOG_GAP / EXP_GAP after the group's previous decision (spec §5): the feed is
prepared while the previous pass runs. The job rules are the v2 station's (v2 `ocarry_sim.OcStation`) with stream
inputs.
Processes (one step per unit):
  blk    a block: the rescale jobs (b >= 1) and the LOG jobs, wave by wave; p_cr, la_cr
  run    a run's end: the epilogue jobs and their O8 windows on E9, ll_cr, run_done
There is no O reader: the epilogue's O8 words leave on E9 as they drain, N_OC words per cycle (spec §10 item 7).
The values (the o_part windows, the carry lo / so / e, the epilogue, the E9 words) are `AccumulatorMath`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.core.cycleMath import ceilDiv, clog2, isPowerOfTwo, readyAt
from simulator.core.module import Module, Need, checkRules
from simulator.core.records import PortSpec, Step, planePort, valuePort
from simulator.core.trace import label
from simulator.core.crossbar import NL_DPE_MODES, CrossbarGroup

# named latencies (spec §5)
FIFO_LAT = 1        # an E8 word valid at t is counted in its FIFO from t + 1
LA_LAT = 1          # an E6 unit's last plane at t: the unit is present from t + 1
LL_LAT = 1          # an E7 unit's last plane at t: the unit is present from t + 1
PRE_LOG = 17        # a LOG job decided at d: its burst at d + 2P + 1 (8 e / so plane reads, the registered read, the
#                     8 groups of I/8 elements with their o_part words, read at d + 8 .., into corner-turn B)
LOG_GAP = 16        # a group's next LOG job decides at >= d + 2P (corner-turn A shifted until d + 16, B until d + 24)
PRE_EXP = 10        # an EXP job decided at d: its burst at d + P + 2 (8 lo plane reads, the registered read, the clamp
#                     flags register)
EXP_GAP = 17        # a group's next EXP job decides at >= d + 2P + 1 (the single hold bursts until d + 17)
JOB_DONE_LAT = 1    # a job's last drain plane at t counts for the other group's readiness from t + 1
P_CR_LAT = 1        # p_cr one cycle after the last o_part word read of a block's last LOG job (d + 2P - 1)
LA_CR_LAT = 2       # la_cr two cycles after the last lo plane read of the unit's last rescale job
LL_CR_LAT = 2       # ll_cr two cycles after the last lo plane read of the run's last epilogue job
RESTART_LAT = 1     # the next run's LOG jobs may decide from t_o8 + 1
E9_LAT = 10         # an epilogue drain plane at drain_start + b is on E9 at drain_start + E9_LAT + b (P + 2: the
#                     8-plane hold, the saturation flags, the output register)
RUN_DONE_LAT = 1    # run_done one cycle after the run's end (t_o8)
OQ_MAX = 127        # oq = min(|o'| >> SO, OQ_MAX) (spec §1)


@dataclass(kw_only=True)
class SoftmaxAccParams:
    """The RTL parameters of softmax_online_acc (RTL names and defaults)."""
    S: int
    D: int
    R: int = 64
    C: int = 64
    P: int = 8
    N_OC: int
    NBLK: int
    SO: int
    LN: int
    P_CH: int
    VWP: int = 13


@dataclass(frozen=True)
class SoftmaxAccDerived:
    """The RTL's localparams."""
    I: int            # min(R, C): the window size
    P_OC: int         # windows per run, S D / I
    RPW: int          # rows per window, I / D
    W: int            # waves per block, ceil(P_OC / N_OC)
    LWC: int          # crossbars active in the last wave
    LANES: int        # bytes per E8 element (its value-word slot), ceil(VWP / 8)
    T_W: int          # windows per input channel, P_OC / P_CH
    FD: int           # the o_part FIFO's depth in words (a power of two >= 2 T_W P)
    NJ: int           # LOG jobs per run, NBLK W
    SHL: int          # the epilogue's left shift, max(SO - LN, 0)
    SHR: int          # the epilogue's right shift, max(LN - SO, 0)
    MDEPTH: int       # lo_mem / e_mem depth in plane words, W P (so_mem: W words, one plane per window; W = 1: a register)


class AccumulatorMath:
    """The values of softmax_online_acc: the o_part windows and log alpha rows of every block, the log l of every
    run, the carried window state (lo, so, e), the epilogue's O8 and the E9 words. `stages` keeps every stage per
    run (GATE 1: attention_ref.ocarry_stage_values and attention_ref.epilogue)."""

    def __init__(self, p: SoftmaxAccParams, d: SoftmaxAccDerived):
        self.p, self.d = p, d
        element = np.arange(d.P_OC)[:, None] * d.I + np.arange(d.I)[None, :]   # element e = m D + c of window p
        self.row, self.col = element // p.D, element % p.D
        self.oPart, self.logAlpha, self.logL = {}, {}, {}  # block -> [P_OC, I]; block -> [S]; run -> [S]
        self.state = {}                                    # run -> window state lo, so, e, o8 [P_OC, I]
        self.stages = {}                                   # run -> {stage: [NBLK, S, D] or [S, D]}
        self.record = None                                 # the block in flight: t, e, o_new, oq, so, lo
        self.epilogueValues = None                         # the run's epilogue in flight: tE, eE, o8

    # -- the inputs -------------------------------------------------------------------------------------------------
    def storeBlock(self, block: int, windows: list, alphaWindow) -> None:
        oPart = np.stack([np.asarray(w.values, dtype=np.int64) for w in windows])
        limit = 1 << (self.p.VWP - 1)
        if oPart.min(initial=0) < -limit or oPart.max(initial=0) >= limit:
            raise ValueError(f"tile {block} carries o_part outside {self.p.VWP} bits")
        self.oPart[block] = oPart
        if alphaWindow is not None:
            logAlpha = np.asarray(alphaWindow.values, dtype=np.int64)
            if logAlpha.min() < -256 or logAlpha.max() > 255:
                raise ValueError(f"the alpha unit of block {block} is outside 9 bits")
            self.logAlpha[block] = logAlpha

    def storeLogL(self, run: int, values) -> None:
        logL = np.asarray(values, dtype=np.int64)
        if logL.min() < -128 or logL.max() > 127:
            raise ValueError(f"log l of run {run} outside int8")
        self.logL[run] = logL

    def feed(self, values: dict) -> np.ndarray:
        """{crossbar i: I feed values} -> the group's input [N_OC, R] (inactive crossbars get zeros)."""
        X = np.zeros((self.p.N_OC, self.p.R), dtype=np.int8)
        for i, v in values.items():
            X[i, :self.d.I] = np.asarray(v, dtype=np.int64).astype(np.int8)
        return X

    # -- a block's jobs ---------------------------------------------------------------------------------------------
    def startBlock(self, run: int) -> None:
        shape = (self.d.P_OC, self.d.I)
        self.state.setdefault(run, {k: np.zeros(shape, dtype=np.int64) for k in ("lo", "so", "e", "o8")})
        self.record = {k: np.zeros(shape, dtype=np.int64) for k in ("t", "e", "o_new", "oq", "so", "lo")}

    def rescaleInput(self, run: int, block: int, wave: range) -> np.ndarray:
        """t = clamp8(log a + lo): the rescale EXP input of the wave's windows."""
        st, logAlpha = self.state[run], self.logAlpha[block]
        t = {i: np.clip(logAlpha[self.row[q]] + st["lo"][q], -128, 127) for i, q in enumerate(wave)}
        for i, q in enumerate(wave):
            self.record["t"][q] = t[i]
        return self.feed(t)

    def collectRescale(self, run: int, wave: range, out: np.ndarray) -> None:
        st = self.state[run]
        for i, q in enumerate(wave):
            st["e"][q] = out[i, :self.d.I].astype(np.int64)             # unsigned EXP byte
            self.record["e"][q] = st["e"][q]

    def logInput(self, run: int, block: int, b: int, wave: range) -> np.ndarray:
        """o' = (b ? (so ? -e : e) << SO : 0) + o_part; oq = min(|o'| >> SO, OQ_MAX): the LOG input."""
        st, oPart, SO = self.state[run], self.oPart[block], self.p.SO
        oq = {}
        for i, q in enumerate(wave):
            oNew = np.asarray(oPart[q], dtype=np.int64)
            if b:
                oNew = (np.where(st["so"][q].astype(bool), -st["e"][q], st["e"][q]) << SO) + oNew
            oq[i] = np.minimum(np.abs(oNew) >> SO, OQ_MAX)
            self.record["o_new"][q], self.record["oq"][q], self.record["so"][q] = oNew, oq[i], (oNew < 0)
        return self.feed(oq)

    def collectLog(self, run: int, wave: range, window: np.ndarray) -> None:
        st = self.state[run]
        for i, q in enumerate(wave):
            st["lo"][q] = window[i].astype(np.int64)                     # signed LOG byte
            st["so"][q] = self.record["so"][q]
            self.record["lo"][q] = st["lo"][q]

    def finishBlock(self, run: int) -> None:
        stages = self.stages.setdefault(run, {})
        for key, v in self.record.items():
            stages.setdefault(key, []).append(self.matrix(v))

    # -- the epilogue -----------------------------------------------------------------------------------------------
    def startEpilogue(self, run: int) -> None:
        shape = (self.d.P_OC, self.d.I)
        self.epilogueValues = {k: np.zeros(shape, dtype=np.int64) for k in ("tE", "eE", "o8")}

    def epilogueInput(self, run: int, wave: range) -> np.ndarray:
        """tE = clamp8(lo - log l): the epilogue EXP input."""
        st, logL = self.state[run], self.logL[run]
        t = {i: np.clip(st["lo"][q] - logL[self.row[q]], -128, 127) for i, q in enumerate(wave)}
        for i, q in enumerate(wave):
            self.epilogueValues["tE"][q] = t[i]
        return self.feed(t)

    def collectEpilogue(self, run: int, wave: range, out: np.ndarray) -> None:
        st, d, ep = self.state[run], self.d, self.epilogueValues
        for i, q in enumerate(wave):
            e = out[i, :d.I].astype(np.int64)                            # unsigned EXP byte
            scaled = (e << d.SHL) >> d.SHR
            ep["eE"][q] = scaled
            ep["o8"][q] = np.where(st["so"][q].astype(bool), np.maximum(-scaled, -128), np.minimum(scaled, 127))

    def finishEpilogue(self, run: int) -> None:
        ep = self.epilogueValues
        self.state[run]["o8"] = ep["o8"]
        stages = self.stages[run]
        for key in list(stages):
            stages[key] = np.stack(stages[key])
        stages["tE"], stages["eE"], stages["O8"] = self.matrix(ep["tE"]), self.matrix(ep["eE"]), self.matrix(ep["o8"])
        stages["lo_final"], stages["so_final"] = stages["lo"][-1], stages["so"][-1]

    def e9Values(self, run: int, wave: range) -> np.ndarray:
        """One E9 window's values [N_OC I]: elements i I .. i I + I - 1 are crossbar i's window (zeros on crossbars
        without a window)."""
        p, d = self.p, self.d
        v = np.zeros((p.N_OC, d.I), dtype=np.int64)
        for i, q in enumerate(wave):
            v[i] = self.state[run]["o8"][q]
        return v.reshape(-1)

    def e9Rows(self, waves: list) -> np.ndarray:
        """One run's E9 window values (wave order) -> its O8 rows [S, D] (the O projection's view of E9)."""
        p, d = self.p, self.d
        a = np.stack([np.asarray(v, dtype=np.int64).reshape(p.N_OC, d.I) for v in waves])
        windows = [a[q // p.N_OC, q % p.N_OC] for q in range(d.P_OC)]
        return np.concatenate(windows).reshape(p.S, p.D)

    def matrix(self, windowValues: np.ndarray) -> np.ndarray:
        """Window values [P_OC, I] -> the S x D matrix (element e = m D + c)."""
        out = np.zeros((self.p.S, self.p.D), dtype=np.int64)
        out[self.row, self.col] = windowValues
        return out


class SoftmaxOnlineAcc(Module):
    NAME = "softmax_online_acc"
    PARAMS_CLASS = SoftmaxAccParams
    BASIS = "rtl-pinned"           # the bit-plane RTL: GATE 2 at 64x64, 19/19 + 5/5 (2026-10-23)
    STEP_METHODS = {"blk": "stepBlock", "run": "stepRunEnd"}

    @staticmethod
    def derive(p: SoftmaxAccParams) -> SoftmaxAccDerived:
        I = min(p.R, p.C)
        P_OC = p.S * p.D // I
        W = ceilDiv(P_OC, p.N_OC)
        T_W = P_OC // p.P_CH
        return SoftmaxAccDerived(I=I, P_OC=P_OC, RPW=I // p.D, W=W, LWC=P_OC - (W - 1) * p.N_OC,
                                 LANES=ceilDiv(p.VWP, 8), T_W=T_W, FD=1 << clog2(2 * T_W * p.P), NJ=p.NBLK * W,
                                 SHL=max(p.SO - p.LN, 0), SHR=max(p.LN - p.SO, 0), MDEPTH=W * p.P)

    @staticmethod
    def checkDomain(p: SoftmaxAccParams) -> None:
        I = min(p.R, p.C)
        rules = [
            (min(p.S, p.D, p.N_OC, p.NBLK, p.P_CH, p.VWP) >= 1 and min(p.SO, p.LN) >= 0,
             "S, D, N_OC, NBLK, P_CH, VWP >= 1 and SO, LN >= 0"),
            (all(isPowerOfTwo(x) for x in (p.S, p.D, I, p.P_CH)), "S, D, I, P_CH powers of two"),
            (8 <= p.S <= I, "8 <= S <= I"),
            (p.D <= I and (p.S * p.D) % I == 0, "D <= I and I | S*D"),
            (p.P == 8, "P = 8"),
            (p.VWP <= 16, "VWP <= 16 (at most two lanes)"),
        ]
        if all(holds for holds, _ in rules):
            P_OC = p.S * p.D // I
            rules += [(p.N_OC <= P_OC, "N_OC <= P_OC"), (p.N_OC <= p.P_CH, "N_OC <= P_CH"),
                      (P_OC % p.P_CH == 0, "P_CH | P_OC")]
        checkRules("softmax_online_acc", rules)

    # ------------------------------------------------------------------------------------------------------------
    def __init__(self, instance: str, params: SoftmaxAccParams, *, runs: int, withValues: bool = True,
                 modes=NL_DPE_MODES, role: str | None = None):
        super().__init__(instance, params, withValues=withValues, role=role)
        p, d = self.p, self.d
        self.runs = int(runs)
        self.math = AccumulatorMath(p, d) if withValues else None
        # the blocks (spec §4)
        n, I = p.N_OC, d.I
        self.logGroup = CrossbarGroup(n, p.R, p.C, p.P, mode="LOG", rowsUsed=I, colsUsed=I, modes=modes,
                                      withValues=withValues)
        self.expGroup = CrossbarGroup(n, p.R, p.C, p.P, mode="EXP", rowsUsed=I, colsUsed=I, modes=modes,
                                      withValues=withValues)
        # what the steps record
        self.jobs = {}                             # (run, kind, b, w) -> job events; kind log | rescale | epilogue
        self.lastDecide = {}                       # group mode -> its previous job's decision (the gap rule)
        self.runEndAt, self.runDoneAt = {}, {}     # run -> t_o8, run_done
        self.resources.crossbars[("LOG", I, I)] += n
        self.resources.crossbars[("EXP", I, I)] += n
        self.resources.memories[(d.FD, d.LANES * I)] += p.P_CH
        self.resources.memories[(d.MDEPTH, I)] += 2 * n          # lo_mem, e_mem (plane words)
        if d.W > 1:                                              # so_mem (one plane per window); at W = 1 the one
            self.resources.memories[(d.W, I)] += n               # word is a register, not a memory (spec §10 item 11)
        # the N_OC crossbars' copies of each slot memory share every address and write enable (lock-step groups)
        self.resources.memShared[(d.MDEPTH, I)] = n
        if d.W > 1:
            self.resources.memShared[(d.W, I)] = n

    # -- declaration --------------------------------------------------------------------------------------------
    def ports(self) -> dict:
        p, d = self.p, self.d
        return {
            "p": valuePort("p", "in", "push+credit", p.VWP, d.I, p.P_CH, "tile", p.P, CAP=2,
                           signals=("p_d", "p_v", "p_cr")),
            "la": planePort("la", "in", "push+credit", 9, p.S, 1, "block b >= 1", p.P, CAP=2,
                            signals=("la_d", "la_v", "la_cr")),
            "ll": planePort("ll", "in", "push+credit", 8, p.S, 1, "run", p.P, CAP=1, signals=("ll_d", "ll_v", "ll_cr")),
            "o": planePort("o", "out", "push", 8, p.N_OC * d.I, 1, "run", p.P,
                           signals=("o_d", "o_v")),                  # one window per epilogue wave (e9Rows)
            "run_done": PortSpec("run_done", "out", "pulse", 1, 1, 1, 1, "run", signals=("run_done",)),
        }

    def processes(self) -> dict:
        return {"blk": self.runs * self.p.NBLK, "run": self.runs}

    def need(self, process: str, unit: int) -> Need:
        p, d = self.p, self.d
        if process == "blk":
            run, b = divmod(unit, p.NBLK)
            windows = {"p": Need.eachChannel(p.P_CH, d.T_W)}
            if b:
                windows["la"] = {0: 1}
            return Need(windows=windows, after={"run": run})          # the previous runs' ends: restart
        return Need(windows={"ll": {0: 1}}, after={"blk": (unit + 1) * p.NBLK})

    # -- the jobs -----------------------------------------------------------------------------------------------
    def wave(self, w: int) -> range:
        n = self.p.N_OC
        return range(w * n, min((w + 1) * n, self.d.P_OC))

    def _job(self, group: CrossbarGroup, key: tuple, ready: int, feed, active: int, pre: int, gap: int) -> tuple:
        """Decide a job: the first cycle it is ready from which its pass, `pre` cycles later, finds the leader ready,
        and at least `gap` after the group's previous decision (decide = max(ready, leader ready - pre, previous +
        gap)); the pass loads at decide + pre."""
        prev = self.lastDecide.get(group.mode)
        loadStart = readyAt(ready + pre, group.readyAt(), None if prev is None else prev + gap + pre)
        decide = loadStart - pre
        self.lastDecide[group.mode] = decide
        xbarPass = group.run(feed, loadStart)
        group.passes.clear()
        I = self.d.I
        self.counts.passes[(group.mode, I, I)] += active                 # the active crossbars of the wave
        events = dict(xbarPass.events, ready=ready, decide=decide)
        self.jobs[key] = events
        return events, xbarPass

    def lastRead(self, decide):
        """The last of a job's P planes, read one per cycle from its decision."""
        with label("stream"):
            return decide + self.p.P - 1

    def lastWordRead(self, decide):
        """The last of a LOG job's P o_part words, read one per cycle from decide + P (spec §5 PRE_LOG)."""
        with label("stream"):
            return decide + 2 * self.p.P - 1

    def block(self, j: int, tLast: list, tAlpha: int | None = None) -> dict:
        """Block j's jobs (spec §5), wave by wave: rescale (b, w) for b >= 1, then LOG (b, w). tLast[q] = the last
        word of o_part window q; tAlpha = the alpha unit's last word (b >= 1). Returns the credit times."""
        p, d, math = self.p, self.d, self.math
        run, b = divmod(j, p.NBLK)
        if math:
            math.startBlock(run)
        out = {}
        for w in range(d.W):
            wave = self.wave(w)
            if b:
                # rescale (b, w): t = clamp8(log a + lo) -> e
                ready = readyAt(tAlpha + LA_LAT, self.jobs[(run, "log", b - 1, w)]["drain_end"] + JOB_DONE_LAT)
                events, xbarPass = self._job(self.expGroup, (run, "rescale", b, w), ready,
                                             math.rescaleInput(run, j, wave) if math else None, len(wave), PRE_EXP,
                                             EXP_GAP)
                if math:
                    math.collectRescale(run, wave, xbarPass.out)
                if w == d.W - 1:
                    out["la_cr"] = self.lastRead(events["decide"]) + LA_CR_LAT
            # LOG (b, w): o' = (b ? (so ? -e : e) << SO : 0) + o_part -> oq, so -> lo
            ready = readyAt(max(tLast[q] for q in wave) + FIFO_LAT,
                            self.jobs[(run, "rescale", b, w)]["drain_end"] + JOB_DONE_LAT if b else None,
                            self.runEndAt[run - 1] + RESTART_LAT if (b == 0 and run) else None)
            events, xbarPass = self._job(self.logGroup, (run, "log", b, w), ready,
                                         math.logInput(run, j, b, wave) if math else None, len(wave), PRE_LOG,
                                         LOG_GAP)
            if math:
                math.collectLog(run, wave, xbarPass.window)
            if w == d.W - 1:
                out["p_cr"] = self.lastWordRead(events["decide"]) + P_CR_LAT
        if math:
            math.finishBlock(run)
        # counts (spec §7)
        elements, memoryBytes = p.S * p.D, d.P_OC * d.I                # a window set: P plane words of I / 8 bytes
        self.counts.clbAdd += elements * (2 if b else 1)
        self.counts.clbSelect += elements * (2 if b else 1)
        self.counts.memWriteBytes += memoryBytes * (2 if b else 1)
        self.counts.memReadBytes += memoryBytes * (2 if b else 0)
        self.counts.fifoWords += d.P_OC * p.P
        fifoBytes = d.P_OC * p.P * d.LANES * d.I // 8                   # FIFO traffic: each word written once, read once
        self.counts.memWriteBytes += fifoBytes
        self.counts.memReadBytes += fifoBytes
        return out

    def epilogue(self, run: int, tLogL: int) -> dict:
        """The run's epilogue jobs (spec §5): tE = clamp8(lo - log l) -> e_E -> O8, the words to E9. tLogL = the log l
        unit's last word. Returns the ll credit and t_o8."""
        p, d, math = self.p, self.d, self.math
        B = p.NBLK
        if math:
            math.startEpilogue(run)
        out = {}
        for w in range(d.W):
            wave = self.wave(w)
            ready = readyAt(tLogL + LL_LAT, self.jobs[(run, "log", B - 1, w)]["drain_end"] + JOB_DONE_LAT)
            events, xbarPass = self._job(self.expGroup, (run, "epilogue", B - 1, w), ready,
                                         math.epilogueInput(run, wave) if math else None, len(wave), PRE_EXP,
                                         EXP_GAP)
            if math:
                math.collectEpilogue(run, wave, xbarPass.out)
            if w == d.W - 1:
                out["ll_cr"] = self.lastRead(events["decide"]) + LL_CR_LAT
        tFirst = events["drain_start"] + E9_LAT                          # the last wave's first E9 plane
        with label("stream"):                                          # its planes: the run's last O8 plane
            self.runEndAt[run] = out["t_o8"] = tFirst + p.P - 1
        self.runDoneAt[run] = self.runEndAt[run] + RUN_DONE_LAT
        if math:
            math.finishEpilogue(run)
        elements, memoryBytes = p.S * p.D, d.P_OC * d.I
        self.counts.clbAdd += elements
        self.counts.clbSelect += 2 * elements
        self.counts.memReadBytes += memoryBytes                          # lo_mem; the O8 planes go to E9, not memory
        return out

    def e9Windows(self, run: int) -> list:
        """Run's E9 windows (spec §3): window w = epilogue wave w, its first plane E9_LAT after the wave's drain start.
        Returns [(w, t, values)]."""
        out = []
        for w in range(self.d.W):
            t = self.jobs[(run, "epilogue", self.p.NBLK - 1, w)]["drain_start"] + E9_LAT
            out.append((w, t, self.math.e9Values(run, self.wave(w)) if self.math else None))
        return out

    # -- steps --------------------------------------------------------------------------------------------------
    def stepBlock(self, j: int, io) -> Step:
        p, d = self.p, self.d
        run, b = divmod(j, p.NBLK)
        windows = io.windows("p")
        tAlpha, alphaUnit, alphaWindow = None, None, None
        if b:
            (alphaWindow,) = io.windows("la")
            alphaUnit = run * (p.NBLK - 1) + b - 1
            with label("stream"):                                    # the alpha window's last plane
                tAlpha = alphaWindow.time + p.P - 1
        if self.math:
            self.math.storeBlock(j, windows, alphaWindow)
        with label("stream"):                                        # each o_part window's last plane
            tLast = [w.time + p.P - 1 for w in windows]
        credits = self.block(j, tLast, tAlpha)
        io.giveCredit("p", j, credits["p_cr"])
        if b:
            io.giveCredit("la", alphaUnit, credits["la_cr"])
        kinds = ("rescale", "log") if b else ("log",)
        jobs = [self.jobs[(run, kind, b, w)] for w in range(d.W) for kind in kinds]
        return Step("blk", j, min(e["decide"] for e in jobs), min(e["load_start"] for e in jobs),
                    max(e["drain_end"] for e in jobs) + 1, dict(p_cr=credits["p_cr"], la_cr=credits.get("la_cr")))

    def stepRunEnd(self, run: int, io) -> Step:
        p = self.p
        (w,) = io.windows("ll")
        if self.math:
            self.math.storeLogL(run, w.values)
        with label("stream"):                                        # the log l window's last plane
            tLogL = w.time + p.P - 1
        o = self.epilogue(run, tLogL)
        io.giveCredit("ll", run, o["ll_cr"])
        windows = self.e9Windows(run)
        for (wave, t, values) in windows:
            io.emit("o", 0, run, wave, t, values)
        io.giveCredit("run_done", run, self.runDoneAt[run])               # the pulse (open in the module's own graph)
        return Step("run", run, self.jobs[(run, "epilogue", p.NBLK - 1, 0)]["decide"], windows[0][1],
                    self.runDoneAt[run],
                    dict(t_o8=o["t_o8"], ll_cr=o["ll_cr"], first_o=windows[0][1], run_done=self.runDoneAt[run]))
