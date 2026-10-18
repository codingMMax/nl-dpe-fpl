#!/usr/bin/env python3
"""ocarry_sim.py — the attention head's o-carry station `att_ocarry_top` (F3; implementation_plan.md
"ATT-2c as designed (2026-10-08)"): values through the certified primitive, RTL-pinned cycle model.

Value contract: `attention_ref.ocarry_stage_values` (per block) + `attention_ref.epilogue` (once per run).
Structure (= the RTL):
  * an EXP group and a LOG group of n_oc identity crossbars each (P29 extent I x I), each lock-step from its
    crossbar 0; element e = r*d + c (token-major: the untransposed PV tile's accumulator, user 2026-10-08), window
    p = e div I (P_oc = S*d/I windows of I/d query rows), word q (5 elements, LW = ceil(I/5) words), wave
    w = p div n_oc, crossbar p mod n_oc; log alpha is 9-bit (m_old - m_new >= -255, the online softmax's la_val);
  * EXP jobs per run: rescale (b, w) for b = 1..NBLK-1, then the epilogue (w); LOG jobs: (b, w) for b = 0..NBLK-1;
    each group runs its jobs in order;
  * rescale feed  t = clamp8(log_a[b][r] + lo[r, c]); drain e -> e_mem (slot w)
    LOG feed      o' = (b ? (+-e) << SO : 0) + o_part; oq = min(|o'| >> SO, 127) (feed), so <- (o' < 0)
                  drain lo -> lo_mem (slot w)
    epilogue feed tE = clamp8(lo - log_l[r]); drain O8 = sat8(+-(e scaled by 2^(SO-LN))) -> output stream.
o_part input (rework M1, 2026-10-08): the window-major PV tile's output stream — window p arrives in wave p div NGP
(NGP = n_E' / Bkv groups), one word per cycle, into FIFO p mod NGP; a LOG job reads its n_oc windows from distinct
FIFOs (n_oc <= NGP). `op_ready[b][p]` = the cycle window p's last word of block b is pushed.
Readiness (one job of a group may start when its leader's MSB_SA_Ready is high and the job is ready):
    LOG(b, w):    every window of the job complete in its FIFO (op_ready + FIFO_LAT); b >= 1: EXP(b, w) drained;
                  b = 0 of a later run: the station restarted after the previous run's last O8 word (+ X5_LAT)
    EXP(b, w):    log_a of block b written (la_go seen); LOG(b-1, w) drained; the previous run's O8 read (o_fed)
    EPI(w):       log_l written (ll_go seen); LOG(NBLK-1, w) drained; the previous run's O8 read (o_fed)
O buffer (rework M2, user 2026-10-09): the epilogue writes O8 into e_mem (the rescale memory); the head's O feed reads
it through the station's read port and pulses o_fed, so every EXP job of the next run (they write e_mem) waits for it.
Cycle grammar per group (the certified primitive, `pass_engine.schedule_pass_sequence`, plus the station's
prefetch): decision d = max(job ready, previous msb_fire + 1); first feed strobe d + PRE; then the primitive.
Every number here is a structural latency of the RTL (named constants below), pinned by strict GATE 2.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "oracle"))
import nldpe_sim as prim  # noqa: E402
import nldpe_ref as nref  # noqa: E402
import attention_ref as aref  # noqa: E402
from pass_engine import schedule_pass_sequence  # noqa: E402

BUF, P, EPS = 40, 8, 5
PRE = 2           # decision -> first feed strobe (memory read, then the registered feed word)
GO_LAT = 1        # an input pulse (la_go / ll_go) is visible to the engines one cycle later
FIFO_LAT = 1      # a window's last o_part word pushed -> its FIFO count lets the LOG job start
X5_LAT = 1        # the run's last O8 word -> the station's next run may decide (state restarts)
VWP = 13          # o_part lane width of the station cases (Bkv 16: |sum of 16 EXP bytes| <= 4,080)
DONE_LAT = 1      # a drained window is visible to the other group's readiness one cycle after its last word
OP_FREE_LAT = 1   # op_free pulses the cycle after the block's last o_part read request
LA_FREE_LAT = 2   # la_free pulses two cycles after the slot's last read request (its last feed word computed)
O8_LAT = 2        # epilogue drain word -> O8 output word (sign read, then the registered output)
O_FED_LAT = 1     # o_fed (the previous run's O8 read from e_mem) -> this run's EXP jobs may decide


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class OcGeometry:
    S: int
    d: int
    R: int
    C: int
    n_oc: int
    n_blk: int
    I: int
    P_oc: int          # windows
    W: int             # waves = ceil(P_oc / n_oc)
    LW: int            # words per window
    LOAD: int          # feed strobes per window (P29: I rows)
    OUT: int           # drain words per window (P29: I columns)
    NGP: int = 1       # PV tile groups = o_part FIFOs (window p in FIFO p mod NGP, wave p div NGP)
    TWP: int = 1       # PV waves per block (P_oc / NGP)


def geometry(S: int, d: int, R: int, C: int, n_oc: int, n_blk: int, pass_extent: str = "used",
             scale: float = 1.0, ngp: int | None = None) -> OcGeometry:
    """`pass_extent` "used" (default) = the RTL (every pass loads / drains I rows / columns); "full" = the certified
    full load (R in, C out), for full-extent comparisons only. `scale` stretches the crossbar load / drain times
    (stress tests of the head recurrence only)."""
    I = min(R, C)
    if not (_pow2(S) and _pow2(d) and _pow2(I) and n_oc >= 1 and n_blk >= 1):
        raise ValueError(f"S={S}, d={d}, I={I} must be powers of two; n_oc, n_blk >= 1")
    if not (8 <= S <= I) or (S * d) % I:
        raise ValueError(f"need 8 <= S <= I and I | S*d (S={S}, d={d}, I={I}); the RTL also needs d <= I")
    P_oc = S * d // I
    if n_oc > P_oc:
        raise ValueError(f"n_oc={n_oc} > P_oc={P_oc} windows")
    if ngp is None:                                      # smallest power of two >= n_oc
        ngp = 1 << (n_oc - 1).bit_length()
    if not (_pow2(ngp) and n_oc <= ngp and P_oc % ngp == 0):
        raise ValueError(f"NGP={ngp}: a power of two, >= n_oc={n_oc}, dividing P_oc={P_oc}")
    if pass_extent not in ("used", "full"):
        raise ValueError(f"pass_extent must be 'used' or 'full', got {pass_extent!r}")
    rows = I if pass_extent == "used" else None
    cm = prim.cycle_model(1, R, C, P, BUF, rows=rows, cols=rows)
    sc = lambda x: int(-(-x * scale // 1)) if scale != 1.0 else int(x)  # noqa: E731
    return OcGeometry(S=S, d=d, R=R, C=C, n_oc=n_oc, n_blk=n_blk, I=I, P_oc=P_oc, W=_cdiv(P_oc, n_oc),
                      LW=_cdiv(I, EPS), LOAD=sc(cm.load_cyc), OUT=sc(cm.output_cyc), NGP=ngp, TWP=P_oc // ngp)


def stream_ready(g: OcGeometry, start: int, pace: int) -> list:
    """op_ready of one block whose PV stream starts at `start` (wave 0's first word) with waves `pace` apart."""
    assert pace >= g.LW, "a wave's words are gapless and waves do not overlap"
    return [start + (p // g.NGP) * pace + g.LW - 1 for p in range(g.P_oc)]


def window_rc(g: OcGeometry, p: int) -> tuple:
    """(r, c) of the I elements of window p (e = p*I + k, token-major e = r*d + c)."""
    e = p * g.I + np.arange(g.I)
    return e // g.d, e % g.d


# ---------------------------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------------------------
class NldpeOcarry:
    """Values through the certified primitive: one identity pass per window (EXP / LOG mode)."""

    def __init__(self, S: int, d: int, R: int, C: int, n_oc: int, n_blk: int,
                 SO: int, LN: int, ngp: int | None = None) -> None:
        self.g = geometry(S, d, R, C, n_oc, n_blk, ngp=ngp)
        self.SO, self.LN = SO, LN
        I = self.g.I
        self.xbar = prim.NldpeDpe(R, C, BUF, P, rows_used=I, cols_used=I)
        self.xbar.program_weights(np.eye(R, C, dtype=np.int8))

    def _passes(self, feed: np.ndarray, mode: int) -> np.ndarray:
        """feed int [S, d] (int8 range) -> crossbar output bytes [S, d] (signed int8 as int64)."""
        g = self.g
        X = np.zeros((g.P_oc, g.R), dtype=np.int8)
        for p in range(g.P_oc):
            r, c = window_rc(g, p)
            X[p, :g.I] = feed[r, c].astype(np.int8)
        out = self.xbar.run_workload(X, mode).out_stream[:, :g.I].astype(np.int64)
        out = ((out & 0xFF) ^ 0x80) - 0x80                                    # the output byte as int8
        res = np.zeros((g.S, g.d), dtype=np.int64)
        for p in range(g.P_oc):
            r, c = window_rc(g, p)
            res[r, c] = out[p]
        return res

    def run(self, o_part: np.ndarray, log_a: np.ndarray, log_l: np.ndarray) -> dict:
        """o_part int [B, S, d]; log_a int [B, S] (row 0 unused); log_l int [S] -> per block + epilogue."""
        g, SO = self.g, self.SO
        op = np.asarray(o_part, dtype=np.int64)
        la = np.asarray(log_a, dtype=np.int64)
        B = op.shape[0]
        assert B == g.n_blk and op.shape[1:] == (g.S, g.d) and la.shape == (B, g.S)
        keys = ("t", "e", "o_new", "oq", "so", "lo")
        st = {k: np.zeros(op.shape, dtype=np.int64) for k in keys}
        lo = so = None
        n_exp = n_log = 0
        for b in range(B):
            if b == 0:
                t = np.zeros((g.S, g.d), dtype=np.int64)
                e = np.zeros((g.S, g.d), dtype=np.int64)
                o_new = op[0].copy()
            else:
                t = np.clip(la[b][:, None] + lo, -128, 127)
                e = self._passes(t, nref.MODE_EXP) & 0xFF                     # unsigned EXP byte
                n_exp += g.P_oc
                o_new = (np.where(so.astype(bool), -e, e) << SO) + op[b]
            so = (o_new < 0).astype(np.int64)
            oq = np.minimum(np.abs(o_new) >> SO, 127)
            lo = self._passes(oq, nref.MODE_LOG)                              # signed LOG byte
            n_log += g.P_oc
            for k, v in zip(keys, (t, e, o_new, oq, so, lo)):
                st[k][b] = v
        tE = np.clip(lo - np.asarray(log_l, dtype=np.int64)[:, None], -128, 127)
        eE = aref._scale_pow2(self._passes(tE, nref.MODE_EXP) & 0xFF, SO - self.LN)
        n_exp += g.P_oc
        O8 = aref._sat8_signed(so, eE)
        return {**st, "lo_final": lo, "so_final": so, "tE": tE, "eE": eE, "O8": O8,
                "exp_passes": n_exp, "log_passes": n_log}

    def gate1(self, o_part, log_a, log_l) -> dict:
        r = self.run(o_part, log_a, log_l)
        ref = aref.ocarry_stage_values(o_part, log_a, self.LN, self.SO)
        for k in ("t", "e", "o_new", "oq", "so", "lo"):
            if not np.array_equal(r[k], ref[k]):
                raise RuntimeError(f"GATE 1: o-carry stage {k} != attention_ref.ocarry_stage_values")
        ep = aref.epilogue(r["lo_final"], r["so_final"], log_l, self.LN, self.SO)
        for k in ("tE", "eE", "O8"):
            if not np.array_equal(r[k], ep[k]):
                raise RuntimeError(f"GATE 1: epilogue {k} != attention_ref.epilogue")
        return r


# ---------------------------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------------------------
class _Group:
    """One lock-step crossbar group: the certified pass grammar with the station's PRE-cycle prefetch."""

    def __init__(self, g: OcGeometry) -> None:
        self.g = g
        self.ready_from = 0          # leader MSB_SA_Ready high (previous msb_fire + 1)
        self.acc_free = 0
        self.out_free = 0

    def place(self, job_ready: int) -> dict:
        g = self.g
        d = max(job_ready, self.ready_from)
        load_start = d + PRE
        compute_start = max(load_start + g.LOAD, self.acc_free)
        msb_fire = compute_start + P - 1
        acam_done = max(msb_fire + 2, self.out_free)
        drain_start = acam_done + 1
        drain_end = acam_done + g.OUT
        self.ready_from = msb_fire + 1
        self.acc_free = acam_done + 1
        self.out_free = drain_end + 1
        return dict(decide=d, load_start=load_start, msb_fire=msb_fire, drain_start=drain_start,
                    drain_end=drain_end)


@dataclass
class OcRunInputs:
    la_go: list            # cycle of la_go for blocks 1..NBLK-1 (index b; entry 0 unused)
    op_ready: list         # per block 0..NBLK-1: per window p, the cycle its last o_part word is pushed
    ll_go: int
    op_start: list = field(default_factory=list)   # (TB schedules) per block: wave 0's first word
    o_fed: int | None = None   # the O feed's o_fed pulse after this run (its O8 read from e_mem); None: not yet


@dataclass
class OcCycles:
    exp: list = field(default_factory=list)       # per run: list of (kind, b, w, events)
    log: list = field(default_factory=list)       # per run: list of (b, w, events)
    op_free: list = field(default_factory=list)   # per run: [NBLK]
    la_free: list = field(default_factory=list)   # per run: [NBLK] (entry 0 = None)
    o8: list = field(default_factory=list)        # per run: [W][LW] output cycle of word (w, q)
    run_done: list = field(default_factory=list)  # per run


class OcStation:
    """The station's event model, driven one block at a time (then the epilogue) — the head recurrence calls it
    incrementally; `ocarry_cycle_model` is the batch form. `x5`: a later run's block 0 waits for the previous
    epilogue (the station holds o state per run) and a run's EXP jobs wait for the previous run's o_fed (`fed`: its
    O8 read from e_mem, M2); False = unlimited buffers, for the head's lower bound."""

    def __init__(self, g: OcGeometry, x5: bool = True) -> None:
        self.g, self.x5 = g, x5
        self.ex, self.lg = _Group(g), _Group(g)
        self.prev_o8_last = None
        self.o_ok = 0                # e_mem may be written from this cycle (None: O8 of the last run not yet read)
        self._new_run()

    def fed(self, t: int) -> None:
        """The O feed's o_fed pulse at cycle t: the last run's O8 was read; the next run's EXP jobs may start."""
        self.o_ok = t + O_FED_LAT

    def _exp_ok(self) -> int:
        if not self.x5:
            return 0
        if self.o_ok is None:
            raise RuntimeError("EXP job before the previous run's o_fed (call fed() after the O feed)")
        return self.o_ok

    def _new_run(self) -> None:
        g = self.g
        self.b = 0
        self.logs, self.exps = {}, []
        self.op_free, self.la_free = [None] * g.n_blk, [None] * g.n_blk

    def block(self, la_go: int, op_ready: list) -> dict:
        """The next block of the run: its rescale EXP waves (b >= 1) and LOG waves. op_ready[p]: the cycle window
        p's last o_part word is pushed into its FIFO."""
        g, b = self.g, self.b
        if b >= g.n_blk:
            raise RuntimeError("all blocks of the run issued; call epilogue()")
        for w in range(g.W):
            if b >= 1:
                rdy = max(la_go + GO_LAT, self.logs[(b - 1, w)]["drain_end"] + DONE_LAT, self._exp_ok())
                ev = self.ex.place(rdy)
                self.exps.append(("rescale", b, w, ev))
                if w == g.W - 1:
                    self.la_free[b] = ev["decide"] + g.LW - 1 + LA_FREE_LAT
            rdy = max(op_ready[p] for p in range(w * g.n_oc, min((w + 1) * g.n_oc, g.P_oc))) + FIFO_LAT
            if b >= 1:
                rdy = max(rdy, self.exps[-1][3]["drain_end"] + DONE_LAT)
            elif self.x5 and self.prev_o8_last is not None:
                rdy = max(rdy, self.prev_o8_last + X5_LAT)
            ev = self.lg.place(rdy)
            self.logs[(b, w)] = ev
            if w == g.W - 1:
                self.op_free[b] = ev["decide"] + g.LW - 1 + OP_FREE_LAT
        self.b += 1
        return dict(op_free=self.op_free[b], la_free=self.la_free[b],
                    log_end=self.logs[(b, g.W - 1)]["drain_end"],
                    exp_end=self.exps[-1][3]["drain_end"] if b >= 1 else None)

    def epilogue(self, ll_go: int) -> dict:
        """The run's epilogue EXP waves; returns the run's events and starts the next run."""
        g = self.g
        if self.b != g.n_blk:
            raise RuntimeError("epilogue before the run's last block")
        o8 = []
        for w in range(g.W):
            ev = self.ex.place(max(ll_go + GO_LAT, self.logs[(g.n_blk - 1, w)]["drain_end"] + DONE_LAT,
                                   self._exp_ok()))
            self.exps.append(("epilogue", g.n_blk - 1, w, ev))
            o8.append([ev["drain_start"] + q + O8_LAT for q in range(g.LW)])
        self.prev_o8_last = o8[-1][-1]
        self.o_ok = None
        res = dict(exp=self.exps, log=[(b, w, self.logs[(b, w)]) for b in range(g.n_blk) for w in range(g.W)],
                   op_free=self.op_free, la_free=self.la_free, o8=o8, run_done=o8[-1][-1] + 1)
        self._new_run()
        return res


def ocarry_cycle_model(g: OcGeometry, runs: list) -> OcCycles:
    """Event times of the station for back-to-back runs (`OcRunInputs` each, absolute cycles)."""
    st = OcStation(g)
    out = OcCycles()
    for ri in runs:
        assert len(ri.op_ready) == g.n_blk and len(ri.la_go) == g.n_blk
        for b in range(g.n_blk):
            st.block(ri.la_go[b], ri.op_ready[b])
        res = st.epilogue(ri.ll_go)
        if ri.o_fed is not None:
            st.fed(ri.o_fed)
        out.exp.append(res["exp"])
        out.log.append(res["log"])
        out.op_free.append(res["op_free"])
        out.la_free.append(res["la_free"])
        out.o8.append(res["o8"])
        out.run_done.append(res["run_done"])
    return out


def or_reads(g: OcGeometry) -> int:
    """Words the O feed reads back per run through the read port (TB: every {w, q} once, one per cycle)."""
    return g.W * g.LW


def feasible_schedule(g: OcGeometry, n_runs: int, t0: int, op_pace: int, la_lead: int, ll_lead: int,
                      pv_buf: int = 2, wave_pace: int | None = None, early: int = 0, feed: int = 0) -> list:
    """Input schedules for `n_runs` back-to-back runs that respect the buffer-reuse rules the head guarantees: block
    b's o_part stream starts after op_free(b - pv_buf) (the FIFOs hold two tiles), la_go(b) after la_free(b-2) (two
    log_a slots), and a run's la / ll writes after the previous run's last use (its run_done). op_start(b) ~
    t0 + b*op_pace, waves `wave_pace` apart (default the crossbar pass period); la_go(b) is `la_lead` cycles before
    op_start(b) (negative: after); ll_go `ll_lead` cycles before the last op_start. `early` > 0 starts the next run's
    o_part stream that many cycles before the previous run_done (the station's run restart then binds). The TB
    writes a log_a / log_l set in the S cycles before its pulse, so a slot is rewritten from pulse - S on. The TB's O
    feed reads the run's O8 back (`or_reads` words, one per cycle) from run_done on, `feed` idle cycles later pulses
    o_fed (o_fed = run_done + or_reads + 1 + feed); a long feed makes the next run's EXP hold bind."""
    wp = wave_pace if wave_pace is not None else max(g.LW, g.LOAD + P)
    runs: list = []
    t, prev_done, s_end = t0, -10 ** 9, -10 ** 9          # s_end: the previous block's stream end (one PV tile)
    for _ in range(n_runs):
        op_start, la_go = [0] * g.n_blk, [0] * g.n_blk
        ri = OcRunInputs(la_go=la_go, op_ready=[[0] * g.P_oc for _ in range(g.n_blk)], ll_go=0, op_start=op_start)
        for b in range(g.n_blk):
            want = max(t + b * op_pace, s_end + 1)
            j = len(runs) * g.n_blk + b                        # the FIFOs hold two tiles across runs too (X3)
            if j >= pv_buf:
                cm = ocarry_cycle_model(g, runs + [_partial(ri, b)])
                free = [x for rr in cm.op_free for x in rr]
                want = max(want, free[j - pv_buf] + 1)
            op_start[b] = want
            ri.op_ready[b] = stream_ready(g, want, wp)
            s_end = ri.op_ready[b][-1]
            if b >= 1:
                la = max(want - la_lead, t + g.S, prev_done + 1 + g.S)
                if b >= 2:
                    la = max(la, la_go[b - 1] + g.S)                   # one write port: sets in turn
                if b >= 3:
                    cm = ocarry_cycle_model(g, runs + [_partial(ri, b)])
                    la = max(la, cm.la_free[-1][b - 2] + 1 + g.S)
                la_go[b] = la
        ri.ll_go = max(op_start[-1] - ll_lead, t + g.S, prev_done + 1 + g.S)
        cm = ocarry_cycle_model(g, runs + [ri])
        ri.o_fed = cm.run_done[-1] + or_reads(g) + 1 + feed
        runs.append(ri)
        prev_done = cm.run_done[-1]
        t = prev_done + 3 - early
    return runs


def _partial(ri: OcRunInputs, b: int) -> OcRunInputs:
    """Inputs with blocks >= b pushed far out (only events before block b are read)."""
    far = 10 ** 9
    return OcRunInputs(la_go=[x if i < b else far for i, x in enumerate(ri.la_go)],
                       op_ready=[x if i < b else [far] * len(x) for i, x in enumerate(ri.op_ready)], ll_go=far)


# ---------------------------------------------------------------------------------------------
# Case files for tb_att_ocarry_top.v
# ---------------------------------------------------------------------------------------------
def _hex_word(vals, width: int) -> str:
    v = 0
    for i, x in enumerate(vals):
        v |= (int(x) & ((1 << width) - 1)) << (width * i)
    return f"{v:x}"


def dump_case(case_dir: Path, oc: NldpeOcarry, data: list, sched: list, cls: str = "") -> dict:
    """data: per run dict(o_part [B,S,d], log_a [B,S], log_l [S]); sched: per run OcRunInputs (cycles relative
    to the TB's origin = the first cycle after weight programming). GATE 1 per run, then:
      op_<r>_<b>.mem  per (p, q): one o_part word = 5 int32 (lane j at bits 32j), P_oc*LW lines, p-major; the TB
                      streams it as the PV tile does: wave t word q carries window gi + NGP*t in group gi
      la_<r>_<b>.mem  S lines, one int8 per row (b >= 1)        ll_<r>.mem  S lines (log l)
      lo_<r>_<b>.mem  expected lo bytes per (p, q) word (5 x 8 bit), P_oc*LW lines
      o8_<r>.mem      expected O8 per (p, q) word (5 x 8 bit), P_oc*LW lines
      sched.txt       per run: NBLK op_start (wave 0's first word), NBLK la_go (0 = none), ll_go, o_fed (the TB
                      reads the run's O8 back through the read port in the or_reads cycles before o_fed - 1)
      case.json       params + expected event cycles"""
    g = oc.g
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    cm = ocarry_cycle_model(g, sched)

    def per_word(arr: np.ndarray, p: int, q: int) -> list:
        r, c = window_rc(g, p)
        k = 5 * q + np.arange(EPS)
        return [int(arr[r[x], c[x]]) if x < g.I else 0 for x in k]

    for ri, dd in enumerate(data):
        res = oc.gate1(dd["o_part"], dd["log_a"], dd["log_l"])
        for b in range(g.n_blk):
            (case_dir / f"op_{ri}_{b}.mem").write_text("".join(
                _hex_word(per_word(np.asarray(dd["o_part"][b]), p, q), 32) + "\n"
                for p in range(g.P_oc) for q in range(g.LW)))
            (case_dir / f"lo_{ri}_{b}.mem").write_text("".join(
                _hex_word(per_word(res["lo"][b], p, q), 8) + "\n" for p in range(g.P_oc) for q in range(g.LW)))
            if b >= 1:
                (case_dir / f"la_{ri}_{b}.mem").write_text(
                    "".join(f"{int(v) & 0x1FF:03x}\n" for v in np.asarray(dd["log_a"][b])))
        (case_dir / f"ll_{ri}.mem").write_text("".join(f"{int(v) & 0xFF:02x}\n" for v in np.asarray(dd["log_l"])))
        (case_dir / f"o8_{ri}.mem").write_text("".join(
            _hex_word(per_word(res["O8"], p, q), 8) + "\n" for p in range(g.P_oc) for q in range(g.LW)))
    (case_dir / "sched.txt").write_text("".join(
        " ".join(str(x) for x in (s.op_start + s.la_go + [s.ll_go, s.o_fed])) + "\n" for s in sched))
    case = dict(S=g.S, d=g.d, R=g.R, C=g.C, N_OC=g.n_oc, NBLK=g.n_blk, SO=oc.SO, LN=oc.LN, NRUN=len(data),
                I=g.I, P_oc=g.P_oc, W=g.W, LW=g.LW, NGP=g.NGP, TWP=g.TWP, VWP=VWP, cls=cls,
                WAVE_PACE=(sched[0].op_ready[0][g.NGP] - sched[0].op_ready[0][0]) if g.TWP > 1 else g.LW,
                exp_op_free=cm.op_free, exp_la_free=[[x if x is not None else -1 for x in r] for r in cm.la_free],
                exp_o8_first=[[w[0] for w in r] for r in cm.o8], exp_o8_last=[[w[-1] for w in r] for r in cm.o8],
                exp_run_done=cm.run_done, OR_READS=or_reads(g), o_fed=[s.o_fed for s in sched],
                exp_log_drain_end=[[ev["drain_end"] for (_, _, ev) in r] for r in cm.log])
    (case_dir / "case.json").write_text(json.dumps(case, indent=1) + "\n")
    return case


# ---------------------------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/simulator/kernels/ocarry_sim.py
# ---------------------------------------------------------------------------------------------
def _random_inputs(g: OcGeometry, rng, cls: str = "random") -> dict:
    B, S, d = g.n_blk, g.S, g.d
    if cls == "zeros":
        op = np.zeros((B, S, d), dtype=np.int64)
    elif cls == "allneg":
        op = -rng.integers(1, 4096, size=(B, S, d))
    elif cls == "extremes":
        op = rng.choice(np.array([-4080, -1, 0, 1, 4080]), size=(B, S, d))
    else:
        op = rng.integers(-4080, 4081, size=(B, S, d))
    if cls == "alpha0":
        la = np.zeros((B, S), dtype=np.int64)
    elif cls == "extremes":
        la = rng.choice(np.array([-255, -129, -128, -127, 0]), size=(B, S))
    else:
        la = rng.integers(-255, 1, size=(B, S))
    la[0] = 0
    ll = rng.integers(-128, 128, size=S)
    return dict(o_part=op, log_a=la, log_l=ll)


def _self_test() -> None:
    rng = np.random.default_rng(2026_10_08)
    n = 0
    # values: GATE 1 across geometries / classes (including B = 1, partial last wave, n_oc = P_oc)
    for S, d, R, C, n_oc, B in ((16, 8, 64, 64, 1, 3), (16, 16, 64, 64, 3, 2), (32, 8, 64, 64, 4, 4),
                                (8, 8, 32, 32, 2, 1), (16, 16, 64, 32, 8, 2), (64, 32, 256, 256, 3, 2)):
        for cls in ("random", "zeros", "allneg", "extremes", "alpha0"):
            oc = NldpeOcarry(S, d, R, C, n_oc, B, SO=4, LN=4)
            r = oc.gate1(**_random_inputs(oc.g, rng, cls))
            assert r["exp_passes"] == B * oc.g.P_oc and r["log_passes"] == B * oc.g.P_oc
            n += 1
    # the group grammar with PRE = 0 and job readiness = the certified readiness gate == pass_engine
    g = geometry(16, 16, 64, 64, 2, 3)
    global PRE
    saved, PRE = PRE, 0
    try:
        grp, ready = _Group(g), [0, 5, 300, 301, 302, 900]
        mine = [grp.place(x) for x in ready]
        ref = schedule_pass_sequence(ready, 64, 64, BUF, P, rows=g.I, cols=g.I)
        for a, b in zip(mine, ref):
            assert (a["load_start"], a["msb_fire"], a["drain_start"], a["drain_end"]) == \
                   (b["load_start"], b["msb_fire"], b["drain_start"], b["drain_end"]), (a, b)
            n += 1
    finally:
        PRE = saved
    # cycle-model invariants: in-order groups, dependencies, buffer-reuse schedule is legal
    for S, d, R, C, n_oc, B, ngp in ((16, 16, 64, 64, 2, 3, 2), (32, 8, 64, 64, 4, 4, 4), (128, 64, 256, 256, 7, 8, 8),
                                     (16, 16, 64, 64, 2, 3, 4)):
        g = geometry(S, d, R, C, n_oc, B, ngp=ngp)
        for lead, early in ((200, 0), (0, 0), (-150, 0), (0, 400)):
            sched = feasible_schedule(g, 2, 10, op_pace=g.LOAD * 3, la_lead=lead, ll_lead=50, early=early)
            cm = ocarry_cycle_model(g, sched)
            for r in range(2):
                logs = {(b, w): ev for (b, w, ev) in cm.log[r]}
                for kind, b, w, ev in cm.exp[r]:
                    dep = logs[(b - 1, w)] if kind == "rescale" else logs[(B - 1, w)]
                    assert ev["decide"] > dep["drain_end"]
                    if kind == "rescale":
                        assert logs[(b, w)]["decide"] > ev["drain_end"]
                for b in range(B):                     # a LOG job starts only after its windows are complete
                    for (bb, w, ev) in cm.log[r]:
                        if bb == b:
                            assert ev["decide"] > max(sched[r].op_ready[b][p] for p in
                                                      range(w * n_oc, min((w + 1) * n_oc, g.P_oc)))
                free = [x for rr in cm.op_free for x in rr]
                for b in range(B):
                    j = r * B + b
                    if j >= 2:
                        assert sched[r].op_start[b] > free[j - 2]
                    if b >= 3:
                        assert sched[r].la_go[b] - g.S > cm.la_free[r][b - 2]
                    if b >= 2:
                        assert sched[r].la_go[b] >= sched[r].la_go[b - 1] + g.S
                for seq in (cm.exp[r], cm.log[r]):
                    evs = [x[-1] for x in seq]
                    assert all(a["msb_fire"] < b["decide"] for a, b in zip(evs, evs[1:]))
                if r:                                  # the station restarts after the previous run's last O8 word
                    first = min(ev["decide"] for (b, w, ev) in cm.log[r] if b == 0)
                    assert first >= cm.o8[r - 1][-1][-1] + X5_LAT
                    if early:
                        assert first == cm.o8[r - 1][-1][-1] + X5_LAT, "early stream: the restart must bind"
                n += 1
    # M2: a run's EXP jobs (they write e_mem, the O buffer) wait for the previous run's o_fed; a long O feed binds
    # (the first EXP decision is o_fed + O_FED_LAT exactly), a short one leaves the other edges binding; NBLK = 1: the
    # epilogue is the run's first EXP job
    for S, d, R, C, n_oc, B, ngp in ((16, 16, 64, 64, 2, 3, 2), (16, 16, 64, 64, 2, 1, 2), (32, 8, 64, 64, 4, 2, 4)):
        g = geometry(S, d, R, C, n_oc, B, ngp=ngp)
        for feed, early in ((0, 0), (3000, 0), (3000, 400)):
            sched = feasible_schedule(g, 3, 10, op_pace=g.LOAD * 3, la_lead=0, ll_lead=50, early=early, feed=feed)
            cm = ocarry_cycle_model(g, sched)
            for r in range(3):
                assert sched[r].o_fed == cm.run_done[r] + or_reads(g) + 1 + feed
                if r:
                    first_e = min(ev["decide"] for (_, _, _, ev) in cm.exp[r])
                    assert first_e >= sched[r - 1].o_fed + O_FED_LAT
                    if feed:
                        assert first_e == sched[r - 1].o_fed + O_FED_LAT, "a long O feed must bind the EXP hold"
                n += 1
    try:                                               # the next run's EXP job before the O feed released the buffer
        g = geometry(16, 16, 64, 64, 2, 2)
        sched = feasible_schedule(g, 2, 10, op_pace=g.LOAD * 3, la_lead=0, ll_lead=50)
        sched[0].o_fed = None
        ocarry_cycle_model(g, sched)
    except RuntimeError:
        n += 1
    else:
        raise AssertionError("EXP job without the previous run's o_fed not rejected")
    # monotonicity: later inputs never make an event earlier
    g = geometry(32, 8, 64, 64, 4, 4)
    base = feasible_schedule(g, 1, 10, op_pace=100, la_lead=0, ll_lead=0)
    late = [OcRunInputs(la_go=[x + 37 for x in base[0].la_go], op_ready=[[x + 37 for x in r] for r in base[0].op_ready],
                        ll_go=base[0].ll_go + 37)]
    a, b = ocarry_cycle_model(g, base), ocarry_cycle_model(g, late)
    assert b.run_done[0] >= a.run_done[0] and all(y >= x for x, y in zip(a.op_free[0], b.op_free[0]))
    n += 1
    # guards
    for bad in ((12, 8, 64, 64, 1, 1), (16, 8, 64, 64, 9, 1), (4, 8, 64, 64, 1, 1), (16, 2, 64, 64, 1, 1),
                (16, 16, 64, 64, 3, 1, "used", 1.0, 2), (16, 16, 64, 64, 2, 1, "used", 1.0, 3),
                (16, 16, 64, 64, 2, 1, "used", 1.0, 8)):      # NGP < n_oc, not a power of two, not dividing P_oc
        try:
            geometry(*bad)
        except ValueError:
            n += 1
        else:
            raise AssertionError(f"guard missed {bad}")
    print(f"ocarry_sim: ALL PASS ({n} checks: values == attention_ref o-carry + epilogue through the certified "
          f"EXP / LOG passes, group grammar == pass_engine, FIFO readiness, run restart, schedule legality, "
          f"monotonicity, guards)")


if __name__ == "__main__":
    _self_test()
