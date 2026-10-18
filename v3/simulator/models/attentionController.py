"""attentionController.py — behavior model of `att_ctrl` (the head's tiling controller), contract `top/spec.md`.

att_ctrl holds counters only (spec §5, §6). Processes (one step per unit):
  start   the head's start window: t_s; the head runs from t_s + RUN_LAT
  x<c>    copy c's token k: its burst must start at >= x_rdy(c, k); the source's credit k at x_rdy(c, k + 1), the
          latest of the copy's three GEMM ready pulses k, the staging pulses the copy's next token needs (+ CR_LAT),
          and the run start (no credit after the copy's last token)
  qj, pj  QK / PV job j: offered from the run start, taken at the job port's credit j - 1 (the DIMM's t_go); the
          job's fields (the A and B slots, the release flag) are control data and travel in timing-only mode too
  go      softmax run r: from the run start, at the softmax's credit r - 1 (busy falling)
  done    the accumulator's run_done pulse R_b - 1: done, then the start port's credit (busy falling)
The ports x<c> watch copy c's X bus (they take its shape, `xSpec`); the ready and staging pulses arrive on pulse ports.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from simulator.core.cycleMath import clog2, isPowerOfTwo, readyAt
from simulator.core.errors import ContractError
from simulator.core.module import Module, Need, checkRules
from simulator.core.records import PortSpec, Step

# named latencies (spec §5)
RUN_LAT = 1         # start taken at t_s: the head runs from t_s + 1
CR_LAT = 1          # a staging credit pulse at t counts for admission from t + 1
DONE_LAT = 1        # done one cycle after the last run's run_done
BUSY_LAT = 1        # busy falls one cycle after done
PB_CAP = 4          # PV B credits (edge E3) per GEMM copy, in rows (spec §5)


@dataclass(kw_only=True)
class AttCtrlParams:
    """The RTL parameters of att_ctrl (RTL names)."""
    NSEQ: int
    S: int
    BKV: int
    I: int
    P_PROJ: int
    LOADX: int


@dataclass(frozen=True)
class AttCtrlDerived:
    """The RTL's localparams."""
    RB: int           # runs (row blocks), NSEQ / S
    NBLK: int         # key blocks, NSEQ / BKV
    NJ: int           # jobs, RB NBLK
    NB_A: int         # QK A groups per credit, I / BKV
    T_X: int          # tokens per GEMM copy, NSEQ / P_PROJ
    QSA: int          # the QK job port's A slot field width
    QSB: int          # the QK job port's B slot field width
    PSA: int          # the PV job port's A slot field width
    PSB: int          # the PV job port's B slot field width


def pulsePort(name: str, unit: str) -> PortSpec:
    return PortSpec(name, "in", "pulse", 1, 1, 1, 1, unit, signals=(name,))


class AttCtrl(Module):
    NAME = "att_ctrl"
    PARAMS_CLASS = AttCtrlParams
    BASIS = "rtl-pinned"           # 2026-10-15: bring-up 5/5 + gate H 6/6 on 64x64 (every event, cycles, values)

    @staticmethod
    def derive(p: AttCtrlParams) -> AttCtrlDerived:
        RB, NBLK = p.NSEQ // p.S, p.NSEQ // p.BKV
        return AttCtrlDerived(RB=RB, NBLK=NBLK, NJ=RB * NBLK, NB_A=p.I // p.BKV, T_X=p.NSEQ // p.P_PROJ,
                              QSA=max(1, clog2(RB)), QSB=max(1, clog2(NBLK)), PSA=1, PSB=max(1, clog2(NBLK)))

    @staticmethod
    def checkDomain(p: AttCtrlParams) -> None:
        checkRules("att_ctrl", [
            (min(p.NSEQ, p.S, p.BKV, p.I, p.P_PROJ, p.LOADX) >= 1, "NSEQ, S, BKV, I, P_PROJ, LOADX >= 1"),
            (all(isPowerOfTwo(x) for x in (p.NSEQ, p.S, p.BKV, p.I, p.P_PROJ)), "NSEQ, S, BKV, I, P_PROJ powers of two"),
            (p.S <= p.NSEQ and p.BKV <= p.NSEQ and p.P_PROJ <= p.NSEQ and p.BKV <= p.I,
             "S, BKV, P_PROJ <= NSEQ and BKV <= I")])

    def __init__(self, instance: str, params: AttCtrlParams, *, xSpec: PortSpec, withValues: bool = True,
                 role: str | None = None):
        super().__init__(instance, params, withValues=withValues, role=role)
        if xSpec.words != self.p.LOADX:
            raise ValueError(f"att_ctrl: the X bus carries {xSpec.words} words per burst, LOADX = {self.p.LOADX}")
        self.xSpec = xSpec
        self.tStart = self.tRun = None
        self.xReadyAt = {}                     # (copy, token k) -> x_rdy rising (k >= 1)
        self.done = None

    # -- declaration --------------------------------------------------------------------------------------------
    def ports(self) -> dict:
        p, d = self.p, self.d
        ports = {"start": PortSpec("start", "in", "burst", 1, 1, 1, 1, "head", CAP=1, signals=("start", "busy")),
                 "qj": PortSpec("qj", "out", "burst", d.QSA + d.QSB + 1, 1, 1, 1, "job", CAP=1,
                                signals=("qj_v", "qj_a", "qj_b", "qj_ra", "qj_rdy"), control=True),
                 "pj": PortSpec("pj", "out", "burst", d.PSA + d.PSB + 1, 1, 1, 1, "job", CAP=1,
                                signals=("pj_v", "pj_a", "pj_b", "pj_ra", "pj_rdy"), control=True),
                 "go": PortSpec("go", "out", "burst", 1, 1, 1, 1, "run", CAP=1, signals=("sm_start", "sm_busy")),
                 "qa": pulsePort("qa", "QK A group"), "qb": pulsePort("qb", "QK B matrix"),
                 "pb": pulsePort("pb", "PV B row"), "rd": pulsePort("rd", "run")}
        for c in range(p.P_PROJ):
            ports[f"x{c}"] = replace(self.xSpec, name=f"x{c}", direction="in", signals=("x_v", "x_rdy"))
            for g in ("q", "k", "v"):
                ports[f"r{g}{c}"] = pulsePort(f"r{g}{c}", "row")
        return ports

    def processes(self) -> dict:
        p, d = self.p, self.d
        return {"start": 1, **{f"x{c}": d.T_X for c in range(p.P_PROJ)}, "qj": d.NJ, "pj": d.NJ, "go": d.RB,
                "done": 1}

    def need(self, process: str, unit: int) -> Need:
        p, d = self.p, self.d
        if process == "start":
            return Need(windows={"start": {0: 1}})
        after = {"start": 1}
        if process.startswith("x"):
            copy = int(process[1:])
            nextToken = copy + (unit + 1) * p.P_PROJ                 # the copy's next token
            credits = {}
            if nextToken < p.NSEQ:
                credits = {f"rq{copy}": unit, f"rk{copy}": unit, f"rv{copy}": unit}
                for port, i in (("qa", nextToken // d.NB_A - 2), ("qb", nextToken // p.BKV - 2),
                                ("pb", nextToken - PB_CAP * p.P_PROJ)):
                    credits[port] = i if i >= 0 else None
            return Need(windows={process: Need.eachChannel(self.xSpec.L)}, credits=credits, after=after)
        if process in ("qj", "pj", "go"):
            return Need(credits={process: (unit - 1) if unit else None}, after=after)
        return Need(credits={"rd": d.RB - 1}, after=after)

    # -- steps --------------------------------------------------------------------------------------------------
    def step(self, process: str, unit: int, io) -> Step:
        if process == "start":
            return self.stepStart(io)
        if process.startswith("x"):
            return self.stepToken(int(process[1:]), unit, io)
        if process in ("qj", "pj"):
            return self.stepJob(process, unit, io)
        if process == "go":
            return self.stepGo(unit, io)
        return self.stepDone(io)

    def stepStart(self, io) -> Step:
        (w,) = io.windows("start")
        self.tStart, self.tRun = w.time, w.time + RUN_LAT
        return Step("start", 0, w.time, self.tRun, self.tRun)

    def stepToken(self, copy: int, k: int, io) -> Step:
        """Copy c's token k: its burst, and the credit (x_rdy) of the copy's next token."""
        p = self.p
        process = f"x{copy}"
        tBurst = io.windows(process)[0].time
        want = self.xReadyAt.get((copy, k), self.tRun)
        if tBurst < want:
            raise ContractError(f"{self.instance}: copy {copy} token {k} bursts at {tBurst}, before x_rdy ({want})")
        events = dict(burst=tBurst, x_rdy=want)
        if copy + (k + 1) * p.P_PROJ < p.NSEQ:
            staging = [io.creditTime(x) for x in ("qa", "qb", "pb")]
            nextReady = readyAt(self.tRun, io.creditTime(f"rq{copy}"), io.creditTime(f"rk{copy}"),
                                io.creditTime(f"rv{copy}"), *(None if t is None else t + CR_LAT for t in staging))
            self.xReadyAt[(copy, k + 1)] = events["x_rdy_next"] = nextReady
            io.giveCredit(process, k, nextReady)
        return Step(process, k, want, tBurst, tBurst + p.LOADX, events)

    def stepJob(self, port: str, k: int, io) -> Step:
        """QK / PV job k: offered at the run start or the job port's previous credit; the fields (A slot, B slot,
        release A) are the tiling's."""
        d = self.d
        t = readyAt(self.tRun, io.creditTime(port))
        fields = (k // d.NBLK, k % d.NBLK, 0) if port == "qj" else (k % 2, k % d.NBLK, 1)
        io.emit(port, 0, k, 0, t, np.array(fields, dtype=np.int64))
        return Step(port, k, t, t, t + 1, dict(fields=fields))

    def stepGo(self, run: int, io) -> Step:
        t = readyAt(self.tRun, io.creditTime("go"))
        io.emit("go", 0, run, 0, t)
        return Step("go", run, t, t, t + 1)

    def stepDone(self, io) -> Step:
        tRunDone = io.creditTime("rd")
        self.done = tRunDone + DONE_LAT
        io.giveCredit("start", 0, self.done + BUSY_LAT)
        return Step("done", 0, tRunDone, self.done, self.done + 1, dict(run_done=tRunDone, done=self.done))
