#!/usr/bin/env python3
"""attention_sim.py — NL-DPE attention head: composition simulator.

Part 1 (ATT-1c-i, value-free): the cycle recurrence of the charter `v2/sim/spec/attention.md` §4–§4.4,
the operation ledger, the crossbar count and the candidate grid. Values (ATT-1c-ii: `NldpeAttention`,
`dump_case`, GATE 1 against `v2/sim/oracle/attention_ref.py`) come in part 2.

Everything here composes certified models only — no cycle is re-derived:
  * crossbar passes: `pass_engine.{crossbar_total, schedule_pass_sequence}` on the certified primitive timing,
    with the P29 pass extent (`dpe_nldpe.md` v2.0.3; default "used": conversion passes load/drain I = min(R, C),
    a V=1 projection loads d_model rows and drains d_head columns);
  * projections and the O-projection: `nldpe_sim.cycle_model` (V = 1) / `gemm_sim.cycle_model` (V > 1);
  * the online softmax: the certified recurrence of `softmax_online_sim.softmax_online_cycle_model`, re-run here
    with one extra term per block (score availability, F8 / D9) and the LA hold-off (edge X4) — self-test (a)
    gates it to the certified model exactly when both are zero;
  * the tile E stage: T(ceil(P_mm / n_E)) + the certified registered-drain +1 (self-test (b) against
    `dimm_sim.dimm_cycle_model` with DimmPassPlan(0, 0, E)).
The only constants this module adds are the charter's named constants (§4.3): G, TREE_PIPE_T, OC_ADD, RD_LAT,
LA_LAT and the buffer depths; overlaps OVL_k are 0 (conservative). All of them are assumptions until the
attention RTL (ATT-2) measures them, which is why every result carries `cycles_basis`.

Charter symbols: I = min(R, C); S = min(n_seq, I) query rows per run; R_b = n_seq / S runs; B = n_seq / B_kv key
blocks; P_mm = ceil(S B_kv d / I) windows per matmul tile; P_sm = ceil(S B_kv / I) softmax EXP windows per block;
P_oc = ceil(S d / I) output-correction windows per block; W_st = ceil(S B_kv / 5) score words per block.

Run:  python3 v2/sim/simulator/kernels/attention_sim.py            # self-test (ALL PASS)
      python3 v2/sim/simulator/kernels/attention_sim.py --table    # 60-point reference-candidate table
"""
from __future__ import annotations

import argparse
import math
import random
import sys

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_sim as prim  # noqa: E402
import gemm_sim  # noqa: E402
import dimm_sim  # noqa: E402
import dimm_tile_sim  # noqa: E402
import ocarry_sim  # noqa: E402
import proj_sim  # noqa: E402
from pass_engine import crossbar_total, identity_extent, schedule_pass_sequence  # noqa: E402
from softmax_online_sim import (EPS, FEED_LAT, GO_LAT, MAX_LAT, SP_LAT, WAIT_LAT,  # noqa: E402
                                check_domain as softmax_check_domain, softmax_online_cycle_model)

BUF, P = 40, 8
BKV = 16
# Named constants (charter §4.3) — assumptions until ATT-2 measures them.
G = 1            # registered hand-off at a station boundary
OC_ADD = 1       # o-carry signed add + compaction, registered
# the head's tile A memory grouping is derived inside the tile (RTL localparam A_GRP = `dimm_tile_sim.a_grp_rule`,
# memory rework item 1, user 2026-10-09; was a fixed 8)
# attention_top sequencer edges (RTL-pinned 2026-10-08; times = the cycle an event's pulse is high, from `start`)
SEQ_LAT = 2        # an event -> a registered event counter -> the sequencer's registered decision
X_SEQ = 2          # tile restart: start_ready (registered idle) one edge after the occupancy + the registered start
SM_START_LAT = 2   # start -> the softmax's first start pulse
SCORE_ACC_LAT = 3  # score stream decision -> its first word accepted when the word waits in its FIFO (registered go,
#                    FIFO read, packer); `score_stream` derives every word's cycle, this is its all-ready special case
PT_LAT = 1         # last lp word of the block's last lane window + its BKV column writes (own channel) -> p_done
LA_GO_LAT = 1      # last log(alpha) row of a block -> la_go (registered)
LL_LAT = 3         # log l copy: rd_row register + the 2-cycle rd_ll read + the ll_go register
OFEED_LAT = 3      # O buffer go -> first O8 burst: LOADO + 1 row reads, the row register, the burst decision
OFEED_ROW = 2      # att_ofeed row period = LOADO + OFEED_ROW (LOADO + 1 row reads + 1; its O FIFO never holds it)
CAT_LAT = 3        # a token's last row word in its O FIFO -> att_ocat offers the burst (count, pop step 0, word 0)
LA_LAT = MAX_LAT + 1  # the block's last log(alpha) row event, after its last score word (RTL-measured 2026-10-08:
#                       rows leave as their block max completes during the stream; the last one MAX_LAT + 1 later)
SHAPES = [(64, 32, 96), (128, 64, 128), (256, 64, 192), (512, 64, 128), (1024, 64, 128)]
CONFIGS = [(R, C) for R in (128, 256, 512, 1024) for C in (64, 128, 256)]
CYCLES_BASIS = ("composition of certified models (P29 pass extent '{ext}'); named constants G=1, OC_ADD=1, "
                "LA_LAT=MAX_LAT+S, RD_LAT=S+2, overlaps 0 — assumptions until ATT-2; tiles = dimm_tile_sim "
                "(RTL-pinned: TREE_PIPE_T = tree levels + 2)")


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def pow2_up(x: float) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1.0, x))))


def check_domain(n_seq: int, d: int, dm: int, R: int, C: int, bkv: int = BKV) -> None:
    """The kernel's elaboration guards (charter §3.4), union of the bricks' guards."""
    I = min(R, C)
    if not (pow2(n_seq) and pow2(d) and pow2(bkv) and pow2(I)):
        raise ValueError("n_seq, d_head, B_kv and I = min(R, C) must be powers of two")
    S = min(n_seq, I)
    if not 8 <= S <= I:
        raise ValueError(f"need 8 <= S <= I (S={S}, I={I})")
    if not 8 <= bkv <= min(n_seq, I):
        raise ValueError(f"need 8 <= B_kv <= min(n_seq, I) (B_kv={bkv}, n_seq={n_seq}, I={I})")
    # general projection tiling (user 2026-10-08): d_head > C -> H = ceil(d/C) column tiles, d_head > R ->
    # V_O = ceil(d/R) O-projection row tiles, d_head > I -> conversion windows span tokens; no upper bound here
    if dm < 1:
        raise ValueError("d_model must be >= 1")


# =====================================================================================================
# Online softmax with per-block availability (certified recurrence + two terms)
# =====================================================================================================
class SoftmaxRun:
    """One run of the certified online-softmax recurrence, fed block by block.

    Identical to `softmax_online_cycle_model` except that block b's first score word is additionally held to
    `max(avail, hold)` (absolute cycles): the score availability from the QK tile (F8) and the LA hold-off X4.
    Times are absolute; the run's `start` edge is `t0`."""

    def __init__(self, S: int, bkv: int, R: int, C: int, n_exp: int, N: int, t0: int,
                 pass_extent: str = "used") -> None:
        softmax_check_domain(S, bkv, R, C, BUF, n_exp, N)
        self.S, self.bkv, self.R, self.C, self.n_exp, self.N, self.t0 = S, bkv, R, C, n_exp, N, t0
        self.rows, self.cols = identity_extent(R, C, pass_extent)
        self.B, blkel = N // bkv, S * bkv
        self.STREAM = cdiv(blkel, EPS)
        nwel = min(blkel, min(R, C))
        self.NWB = blkel // nwel
        self.WPW, self.WL = cdiv(nwel, EPS), cdiv(S, EPS)
        self.load_cyc = int(prim.cycle_model(1, R, C, P, BUF, rows=self.rows, cols=self.cols).load_cyc)
        self.ready = [[] for _ in range(n_exp)]
        self.cexp_ready, self.clog_ready = [], []
        self.a_last, self.ret, self.wait, self.blocks = [], [], [], []

    def _seq(self, ready: list) -> dict:
        return schedule_pass_sequence(ready, self.R, self.C, BUF, P, rows=self.rows, cols=self.cols)[-1]

    def admit(self) -> int:
        """The cycle (absolute) the next block's first score word may be accepted: score_ready rises (the previous
        block streamed, the block two before retired)."""
        b = len(self.a_last)
        first = 2 if b == 0 else self.a_last[b - 1] + 1
        if b >= 2:
            first = max(first, self.ret[b - 2] + 1)
        return first + self.t0

    def block(self, avail: int | None = None, hold: int | None = None,
              offers: list | None = None, last_min: int | None = None) -> dict:
        """Advance one block; returns its events (absolute cycles).

        `offers` (optional, PRE-1): per-word earliest offer edges of this block's STREAM words (absolute):
        the producer drives `score_valid` for word w only from offers[w] on, so word w is accepted at
        max(accept(w-1) + 1, offers[w]) — the kernel keeps `score_ready` high inside a block. Without
        offers the block streams contiguously (the certified model)."""
        b = len(self.a_last)
        assert b < self.B, "all blocks of this run were already fed"
        first = 2 if b == 0 else self.a_last[b - 1] + 1          # relative to t0, as in the certified model
        if b >= 2:
            first = max(first, self.ret[b - 2] + 1)
        if avail is not None:
            first = max(first, avail - self.t0)
        if hold is not None:
            first = max(first, hold - self.t0)
        if offers is None:
            last = first + self.STREAM - 1
            if last_min is not None:                              # the producer's own pace bounds the last word
                last = max(last, last_min - self.t0)
            self.a_last.append(last)
        else:
            assert len(offers) == self.STREAM, "one offer edge per score word of the block"
            first = max(first, offers[0] - self.t0)
            acc = first
            for w in range(1, self.STREAM):
                acc = max(acc + 1, offers[w] - self.t0)
            self.a_last.append(acc)
        sp_done, lp_first, lp_last, lp_win_last = 0, None, 0, 0
        for w in range(self.NWB):
            g = b * self.NWB + w
            c = g % self.n_exp
            self.ready[c].append(self.a_last[b] + FEED_LAT)
            ev = self._seq(self.ready[c])
            sp_done = max(sp_done, ev["drain_start"] + (self.WPW - 1) + SP_LAT)
            lp_first = ev["load_start"] if lp_first is None else min(lp_first, ev["load_start"])
            lp_last = max(lp_last, ev["load_start"] + self.load_cyc - 1)
            lp_win_last = max(lp_win_last, ev["load_start"] + self.WPW - 1)   # the window's own words (R > C: fewer)
        cond = max(sp_done, self.a_last[b] + MAX_LAT, self.ret[b - 1] if b else 0)
        self.wait.append(cond + WAIT_LAT)
        if b == 0:
            self.clog_ready.append(self.wait[b] + self.WL + GO_LAT)
            x = None
        else:
            self.cexp_ready.append(self.wait[b] + GO_LAT)
            evx = self._seq(self.cexp_ready)
            x = evx["drain_start"] + (self.WL - 1)
            self.clog_ready.append(x + GO_LAT)
        evl = self._seq(self.clog_ready)
        self.ret.append(evl["drain_start"] + (self.WL - 1))
        t0 = self.t0
        ev = dict(first_word=first + t0, last_word=self.a_last[b] + t0, sp_full=sp_done + t0,
                  carry_start=self.wait[b] + t0, cexp_last=None if x is None else x + t0, retire=self.ret[b] + t0,
                  lp_first=lp_first + t0, lp_last=lp_last + t0, lp_win_last=lp_win_last + t0,
                  la_done=self.a_last[b] + LA_LAT + t0)
        self.blocks.append(ev)
        return ev

    def done(self) -> int:
        """`done` edge of the run (absolute) — the certified `total` shifted by t0."""
        assert len(self.a_last) == self.B, "run not finished"
        return self.ret[-1] + 1 + self.t0


def softmax_availability_model(S: int, bkv: int, R: int, C: int, n_exp: int, N: int,
                               avail: list | None = None, hold: list | None = None,
                               pass_extent: str = "used", offers: list | None = None) -> dict:
    """Stand-alone run (t0 = 0) with optional per-block availability / hold-off and per-word offer edges
    (`offers`: flat, one per score word in stream order, B * STREAM), all relative to the start edge."""
    run = SoftmaxRun(S, bkv, R, C, n_exp, N, 0, pass_extent)
    st = run.STREAM
    for b in range(N // bkv):
        run.block(None if avail is None else avail[b], None if hold is None else hold[b],
                  None if offers is None else offers[b * st:(b + 1) * st])
    return dict(total=run.done(), blocks=run.blocks)


# =====================================================================================================
# Projections
# =====================================================================================================
def projection_schedule(R: int, C: int, n_seq: int, d: int, dm: int, p_proj: int = 1,
                        pass_extent: str = "used", scale_steady: float = 1.0) -> dict:
    """t_p(t): cycle token t's log bytes (Q, K, V alike) reach the operand buffers (charter §4).

    V = 1: the LOG-mode projection crossbar, P_PROJ copies round-robin: t_p = T_fill + floor(t/P) T_steady + G.
    V > 1 (user 2026-10-08; RTL `att_proj_top` V > 1, pinned by `proj_sim`): `gemm_sim` (REGULAR); each copy's lane
    words stream into its own LOG converter (identity, one token per pass, extent d x d; ceil(d / I) chunks when
    d > I, model only): n_pool = P_PROJ converters per operand, dedicated to the projection.
    H = ceil(d/C) column tiles run in lock-step (the full tiles pace the array), for V = 1 and V > 1 alike."""
    V = cdiv(dm, R)
    I = min(R, C)
    if V == 1:
        rows, cols = ((min(R, dm), min(C, d)) if pass_extent == "used" else (None, None))
        cm = prim.cycle_model(1, R, C, P, BUF, rows=rows, cols=cols)
        tf, ts = int(cm.t_fill), int(cm.t_steady)
        ts_s = int(math.ceil(ts * scale_steady))
        tp = [tf + (t // p_proj) * ts_s + G for t in range(n_seq)]
        return dict(V=1, n_pool=0, t_fill=tf, t_steady=ts_s, t_p=tp, busy=cdiv(n_seq, p_proj) * ts_s)
    tg = [int(gemm_sim.cycle_model(t // p_proj + 1, dm, d, R, C, BUF, P, pass_extent=pass_extent).total)
          for t in range(n_seq)]
    if scale_steady != 1.0:
        base = tg[0]
        tg = [base + int(math.ceil((x - base) * scale_steady)) for x in tg]
    # the converter of copy k: token t's lane words (out_valid from tg - OUT_g, reg_full one cycle later) are its
    # load; ceil(d / I) chunk passes per token (one at d <= I, every DSE row)
    used = pass_extent == "used"
    out_g = cdiv((min(C, d) if used and cdiv(d, C) == 1 else C) * P, BUF)
    nch, wch = cdiv(d, I), min(d, I)
    rows_c, cols_c = (wch, wch) if used else (R, C)
    ready = [[] for _ in range(p_proj)]
    tp = []
    for t in range(n_seq):
        k = t % p_proj
        ready[k] += [tg[t] - out_g + 1] * nch
        ev = schedule_pass_sequence(ready[k], R, C, BUF, P, rows=rows_c, cols=cols_c)[-1]
        tp.append(ev["drain_end"] + 1 + G)
    return dict(V=V, n_pool=p_proj, t_fill=tg[0], t_steady=(tg[-1] - tg[0]) // max(1, (n_seq - 1) // p_proj),
                t_p=tp, busy=tg[-1])


# =====================================================================================================
# Sizing rules (charter §4.1 / §5.4)
# =====================================================================================================
def geometry(R: int, C: int, n_seq: int, d: int, bkv: int = BKV) -> dict:
    I = min(R, C)
    S = min(n_seq, I)
    return dict(I=I, S=S, Rb=n_seq // S, B=n_seq // bkv, P_mm=cdiv(S * bkv * d, I), P_sm=cdiv(S * bkv, I),
                P_oc=cdiv(S * d, I), W_st=cdiv(S * bkv, EPS))


def conv_T(p: int, R: int, C: int, pass_extent: str) -> int:
    """Conversion-pass T(p) on one crossbar under the pass extent."""
    rows, cols = identity_extent(R, C, pass_extent)
    return crossbar_total(max(1, p), R, C, P, BUF, rows=rows, cols=cols)


def select_n_exp(R: int, C: int, n_seq: int, d: int, bkv: int = BKV, pass_extent: str = "used") -> int:
    """§4.1 reference candidate: the smallest power of two whose E stage reaches the stream / fill floor."""
    g = geometry(R, C, n_seq, d, bkv)
    floor = max(g["W_st"], conv_T(1, R, C, pass_extent))
    n = 1
    while conv_T(cdiv(g["P_sm"], n), R, C, pass_extent) > floor and n < g["P_sm"]:
        n *= 2
    return n


def farm_sizes(R: int, C: int, n_seq: int, d: int, n_exp: int, n_E: int | None = None, bkv: int = BKV,
               pass_extent: str = "used") -> dict:
    """QK farm n_E, PV farm n_Ep (= n_E), and the o-carry's dedicated EXP group n_x and LOG group n_oc (§5.4; F3
    2026-10-08, user: the EXP windows run on their own group, not the PV farm), all from the stream / fill floor F."""
    g = geometry(R, C, n_seq, d, bkv)
    T = lambda p: conv_T(p, R, C, pass_extent)  # noqa: E731
    F = max(g["W_st"], T(1))
    n_E = d * n_exp if n_E is None else int(n_E)
    n_oc = next(k for k in range(1, 1 << 16) if T(cdiv(g["P_oc"], k)) <= F or k >= g["P_oc"])
    n_x = n_oc                                           # o-carry EXP group: the same P_oc passes and floor
    return dict(n_E=n_E, n_Ep=n_E, n_x=n_x, n_oc=n_oc, F=F)


# =====================================================================================================
# The head recurrence (charter §4 data edges + §4.2 resource edges X1–X9)
# =====================================================================================================
@dataclass
class AttentionCycles:
    total: int
    total_lo: int
    total_hi: int
    ii_cycles: int
    binding: str
    proj_bound_fraction: float
    stages: dict                    # busy cycles per station
    S: int
    Rb: int
    B: int
    V: int
    H: int                          # projection column tiles ceil(d_head / C)
    V_O: int                        # O-projection row tiles ceil(n_heads * d_head / R)
    H_O: int
    n_exp: int
    n_E: int
    n_Ep: int
    n_x: int                        # o-carry EXP crossbars (dedicated group, F3)
    n_oc: int
    n_pool: int
    p_proj: int
    acc_buf: int
    pass_extent: str
    cycles_basis: str
    la_slot_late: int = 0           # log(alpha) sets that would overwrite a station slot in use (must be 0)
    n_heads: int = 1                # heads of the layer (M5): dedicated projections, one engine, one O-projection
    passes: list = field(default_factory=list)      # first x_go of each head's X pass
    events: list = field(default_factory=list)      # per block (r, b) event dict (the `total` schedule)
    runs: list = field(default_factory=list)        # per run event dict


def pv_a_nw(d: int, bkv: int = BKV) -> int:
    """A words per write of the (untransposed) PV tile's bandwidth-matched A port for log|P|: one column k of a softmax
    window (I / Bkv rows) per write = (I / Bkv) / NB words, NB = I / d."""
    return max(1, d // bkv)


def tile_a_grp(n_E: int, R: int, C: int, a_nw: int = 1, a_ch: int = 1) -> int:
    """A tile's A memory grouping — derived by the tile itself (`dimm_tile_sim.a_grp_rule`, the RTL localparam);
    reported here for the ledger and the memory inventory."""
    return dimm_tile_sim.a_grp_rule(n_E, R, C, a_nw, a_ch)


def _score_pace(nbytes: int, I: int) -> int:
    """att_score: cycles from the stream decision to the block's last word accepted when the softmax takes every
    offered word. Reads issue one per cycle from decision + 1 (window words of 5 bytes, the window's last one
    I - 5 (ceil(I/5) - 1) bytes); read m's bytes can leave from decision + m + 2; one 5-byte word per cycle."""
    lw = cdiv(I, EPS)
    deficit = EPS * lw - I
    nread = cdiv(nbytes, I) * lw
    have, m, last = 0, 0, 0
    for k in range(cdiv(nbytes, EPS)):
        need = min(EPS * (k + 1), nbytes)
        while have < need:
            m += 1
            have += EPS - (deficit if m % lw == 0 else 0)
        last = max(last + 1, m + 2) if k else m + 2
    assert m <= nread
    return last


def score_stream(go_min: int, avail: list, W_K: int, LW: int, EV: int, BLK: int, ready_from: int = 0) -> dict:
    """att_score for one block (cycle-exact port of the RTL, rework M1): the QK tile's narrowed words wait in
    FIFOs; `avail[w][c]` = the cycle word c of window w was pushed (readable from the next cycle). go = the first
    cycle >= go_min with window 0's first word in FIFO 0; reads (one per cycle, in window order) wait for their word
    and for room in the 20-byte packer; the softmax raises score_ready at `ready_from` (its admission of the
    block: the previous block streamed, the block two before retired) and keeps it high inside the block. Returns
    go, the acceptance cycle of every stream word and the rel pulse cycle."""
    G = max(go_min, avail[0][0] + 1)
    w = c = 0
    act, inflight, fl = True, False, False
    pc = nbytes = nv_q = 0
    offers, rel = [], None
    nwords = cdiv(BLK, EPS)
    t = G + 1
    while len(offers) < nwords:
        pop = (pc >= EPS or (fl and pc != 0)) and t >= ready_from
        if pop:
            offers.append(t)
        pc_pop = (pc - EPS if pc >= EPS else 0) if pop else pc
        can_rd = act and avail[w][c] + 1 <= t and pc + (EPS if inflight else 0) <= 15
        if inflight:
            nbytes += nv_q
            if nbytes == BLK:
                fl, nbytes = (pc_pop + nv_q != 0), 0
            pc = pc_pop + nv_q
        else:
            pc = pc_pop
            if fl and pc_pop == 0:
                fl = False
        if can_rd:
            nv_q = max(0, min(EPS, EV - EPS * c))
            if w == W_K - 1 and c == LW - 1:
                act, rel = False, t + 1
            elif c == LW - 1:
                w, c = w + 1, 0
            else:
                c += 1
        inflight = can_rd
        t += 1
        assert t < G + 10 ** 7, "score stream does not finish"
    return dict(go=G, offers=offers, rel=rel)


def _operand_ready(R, C, n_seq, d, dm, bkv, p_proj, S, Rb, B, tp) -> tuple:
    """One projection pass with x_go at cycles 1 .. n_seq: the cycles the sequencer may start on Q of run r / K of
    block b / V of block b (+ SEQ_LAT), and the pass's stations-done cycle (the last of the three `done` pulses)."""
    if cdiv(d, C) == 1 and p_proj == 1 and d <= min(R, C):              # V = 1 and V > 1 (the station RTL)
        xg = list(range(1, n_seq + 1))
        out, dn = [], 0
        for kind, per in (("q", None), ("k", None), ("vr", None)):
            g = proj_sim.geometry(kind, n_seq, dm, d, R, C, bkv)
            pm = proj_sim.proj_cycle_model(g, xg)
            w = pm["writes"]
            dn = max(dn, pm["done"])
            nw = len(w) // (n_seq // g.G)                             # writes per group (Q / K) or per token (VR)
            out.append([w[(k + 1) * nw - 1] + SEQ_LAT for k in range(n_seq // g.G)])
        gq = proj_sim.geometry("q", n_seq, dm, d, R, C, bkv).G
        q_rdy = [out[0][(r + 1) * S // gq - 1] for r in range(Rb)]
        k_rdy = out[1]
        v_rdy = [out[2][(b + 1) * bkv - 1] for b in range(B)]
        return q_rdy, k_rdy, v_rdy, dn
    return ([tp[(r + 1) * S - 1] + SEQ_LAT for r in range(Rb)], [tp[(b + 1) * bkv - 1] + SEQ_LAT for b in range(B)],
            [tp[(b + 1) * bkv - 1] + SEQ_LAT for b in range(B)], tp[-1] + 1)


def _head(R, C, n_seq, d, dm, n_exp, n_E, n_Ep, n_x, n_oc, bkv, p_proj, acc_buf, la_buf, pass_extent,
          resources, tree_pipe_t, scale, pv_buf: int = 2, sm_extent: str | None = None,
          n_heads: int = 1) -> dict:
    """The layer schedule (M5): n_heads heads take the engine in turn (head order, global runs h * Rb + r); head h's
    projections run X pass h; one O-projection over the concatenated head outputs. n_heads = 1: the head."""
    g = geometry(R, C, n_seq, d, bkv)
    I, S, Rb, B = g["I"], g["S"], g["Rb"], g["B"]
    Pmm, Poc, Wst = g["P_mm"], g["P_oc"], g["W_st"]
    sc = lambda k, v: int(math.ceil(v * scale.get(k, 1.0)))  # noqa: E731
    T = lambda p: conv_T(p, R, C, pass_extent)  # noqa: E731
    # tiles = the window-major tile DIMM (dimm_tile_sim, rework M1, RTL-pinned): QK (M=S, N=B_kv, K=d), PV untransposed
    # (M=S, N=d, K=B_kv; o_part / O8 token-major, V = B rows, log|P| = A column segments); occupancy
    # occ = START_LAT + T(T_W) (start -> start_ready, edge X1); window w's words leave at out_t0[w div NG] + c into the
    # consumer's FIFOs (QK: att_score, PV: the o-carry station); done = start + total
    tq = dimm_tile_sim.tile_cycle_model(S, bkv, d, R, C, n_E, pass_extent)
    # d > I (general tiling, model only — the head RTL covers d <= I, every DSE row): ceil(d / I) column chunks of
    # N = I run back to back on the PV farm per block
    hp = cdiv(d, I)
    tv = dimm_tile_sim.tile_cycle_model(S, min(d, I), bkv, R, C, n_Ep, pass_extent,
                                        a_nw=pv_a_nw(min(d, I), bkv), a_ch=n_exp)
    tp_qk = (tq.total - tq.occ - 1) if tree_pipe_t is None else tree_pipe_t
    tp_pv = (tv.total - tv.occ - 1) if tree_pipe_t is None else tree_pipe_t
    NGQ, NGP, LWT = tq.geometry.NG, tv.geometry.NG, cdiv(I, EPS)

    def win_out(tm, w: int) -> int:
        """Cycle offset (from the tile's start) of window w's first output word: its wave (window-major RTL domain);
        model only (farm < K, the candidate grid): its last partial wave + the accumulate step."""
        gg = tm.geometry
        if gg.acc_waves == 1:
            return tm.out_t0[w // gg.NG]
        return tm.out_t0[(w + 1) * gg.acc_waves - 1] + dimm_tile_sim.RMW_LAT
    P_qk, P_pv = tq.passes, hp * tv.passes
    t_qk_occ = sc("qk", tq.occ)
    t_pv_occ = [sc("pv", hp * tv.occ) for b in range(B)]
    t_oc_log = sc("oc", T(cdiv(Poc, n_oc)))
    t_oc_exp = sc("oc", T(cdiv(Poc, n_x)))
    # o-carry station (F3, RTL-pinned): `ocarry_sim.OcStation` — dedicated EXP and LOG groups of n_x = n_oc
    # crossbars, window-pipelined; o_part arrives in the station's FIFOs as the PV tile emits each window (rework M1:
    # LOG wave w waits for its windows), la_go = log(alpha) written, ll_go = log(l) read out; X3 = op_free (the PV
    # tile two blocks ahead), X5 = the station's own run restart after the previous run's last O8 word
    if n_x != n_oc:
        raise ValueError("o-carry station: the EXP and LOG groups have the same size")
    ngp_st = NGP if tv.geometry.rtl else max(NGP, pow2_up(n_oc))   # model-only farms: FIFO grouping is notional
    while ngp_st > 1 and Poc % ngp_st:
        ngp_st //= 2
    st = ocarry_sim.OcStation(ocarry_sim.geometry(S, d, R, C, n_oc, B, pass_extent, scale.get("oc", 1.0),
                                                  ngp=max(ngp_st, pow2_up(n_oc))), x5=resources)
    RD_LAT = S + 2
    proj = projection_schedule(R, C, n_seq, d, dm, p_proj, pass_extent, scale.get("proj", 1.0))
    tp = proj["t_p"]
    # the layer's O-projection (M5): one gemm over the concatenated head outputs, K_O = n_heads * d rows
    KO = n_heads * d
    V_O = cdiv(KO, R)
    t_O = sc("oproj", int(gemm_sim.cycle_model(cdiv(S, p_proj), KO, dm, R, C, BUF, P, pass_extent=pass_extent).total))

    # operand readiness (a registered event counter + the sequencer's registered decision: SEQ_LAT): Q of run r,
    # K and V of block b. V = 1, H = 1, P_PROJ = 1 (the head RTL): the projection stations' RTL-pinned writers
    # (`proj_sim`, x_go one per cycle from cycle 1); otherwise the projection schedule's token times. Pass h (head h)
    # is the same pass shifted to its first x_go X_h (the stations are idle until then).
    q_rdy0, k_rdy0, v_rdy0, pdone0 = _operand_ready(R, C, n_seq, d, dm, bkv, p_proj, S, Rb, B, tp)
    EVQ, BLK = min(S * bkv, I), S * bkv                       # valid elements per QK window, bytes per block
    LOADO = cdiv(d, EPS)                                      # O8 words per token (att_ofeed bursts)
    RP = LOADO + OFEED_ROW                                    # att_ofeed row period (its FIFO consumer never holds)
    used_o = pass_extent == "used"
    rows_o = KO if (used_o and V_O == 1) else R
    cols_o = dm if (used_o and cdiv(dm, C) == 1) else C
    LOADK = cdiv(rows_o * P, BUF)                             # O-projection input words per token
    # y of a token: its O-projection pass's drain end + the gemm's tree / serializer latency (certified gemm model)
    k_y = int(gemm_sim.cycle_model(1, KO, dm, R, C, BUF, P, pass_extent=pass_extent).total) \
        - schedule_pass_sequence([0], R, C, BUF, P, rows=rows_o, cols=cols_o)[0]["drain_end"]

    ev, runs = [], []
    a_last, rel, pv_done_at, op_free, la_free = {}, {}, {}, {}, {}
    la_slot_late = 0                           # log(alpha) sets that would overwrite a slot still in use
    qk_start_prev = pv_start_prev = None
    pv_occ_prev = 0
    run_done_prev = fed_prev = ll_go_prev = None
    passes, pv_last = [], {}
    row_last = [[0] * n_seq for _ in range(n_heads)]          # cycle a head's token row is in its O FIFO
    for h in range(n_heads):
        # X pass h: after pass h-1's last x_go and its stations' done (registered latch + decision); from the third
        # pass on also after the engine finished head h-2 (its operand region): its last PV tile done (X11)
        if h == 0:
            xh = 1
        else:
            xh = max(passes[h - 1] + n_seq + 1, passes[h - 1] - 1 + pdone0 + 2)
            if resources and h >= 2:
                xh = max(xh, pv_last[h - 2] + 2)
        passes.append(xh)
        q_rdy, k_rdy, v_rdy = ([x + xh - 1 for x in a] for a in (q_rdy0, k_rdy0, v_rdy0))
        for rh in range(Rb):
            r = h * Rb + rh                                    # global run (head order)
            # the softmax's run r starts once the station finished run r-1 (conservative run boundary)
            t0 = SM_START_LAT if r == 0 else (max(ll_go_prev + 1, run_done_prev + SEQ_LAT) if resources
                                              else ll_go_prev + 1)
            sm = SoftmaxRun(S, bkv, R, C, n_exp, n_seq, t0, sm_extent or pass_extent)
            for b in range(B):
                j = r * B + b
                # --- QK tile: Q of the run + K of the block written, X1 farm, X2 accumulator half
                s_qk = max(q_rdy[rh], k_rdy[b])
                if qk_start_prev is not None:
                    s_qk = max(s_qk, qk_start_prev + t_qk_occ + X_SEQ)                       # X1 (one QK farm)
                if resources and j - acc_buf >= 0:
                    s_qk = max(s_qk, rel[j - acc_buf] + SEQ_LAT)                              # X2 accumulator halves
                qk_done = s_qk + t_qk_occ + 1 + tp_qk
                # --- score stream (att_score): window 0's first word in its FIFO, the previous block drained, X4 /
                # X10, its run started; reads wait for their words (the tile's output stream)
                go_min = t0
                if j >= 1:
                    go_min = max(go_min, a_last[j - 1] + 1)
                if resources and j - la_buf >= 0:
                    go_min = max(go_min, pv_done_at[j - la_buf] + 1)                          # X4 P selections
                if resources and b >= 3:
                    go_min = max(go_min, la_free[j - 2] + 1)                                  # X10 log(alpha) slots
                avail = [[s_qk + win_out(tq, w) + c for c in range(LWT)] for w in range(tq.geometry.W_K)]
                ss = score_stream(go_min, avail, tq.geometry.W_K, LWT, EVQ, BLK, ready_from=sm.admit())
                go = ss["go"]
                sev = sm.block(offers=ss["offers"])
                if sev["first_word"] != ss["offers"][0] or sev["last_word"] != ss["offers"][-1]:
                    raise RuntimeError(f"block {j}: att_score port and softmax acceptance disagree")
                a_last[j] = sev["last_word"]
                rel[j] = ss["rel"]                                                            # the tile's words read
                # --- log|P| corner turn (att_pturn): each EXP lane writes its windows on its own PV-tile A channel,
                # BKV columns per window; the block's last window emitted
                p_done = sev["lp_win_last"] + bkv + PT_LAT
                # --- PV tile: P of the block + V of the block, X9 farm, X3 o_part half
                s_pv = max(p_done + SEQ_LAT, v_rdy[b])
                if pv_start_prev is not None:
                    s_pv = max(s_pv, pv_start_prev + pv_occ_prev + X_SEQ)                     # X9 (one PV farm)
                if resources and j - pv_buf >= 0:
                    s_pv = max(s_pv, op_free[j - pv_buf] + SEQ_LAT)                           # X3 o_part halves
                pv_done = s_pv + t_pv_occ[b] + 1 + tp_pv
                pv_done_at[j] = pv_done
                # --- o-carry station (F3, `ocarry_sim.OcStation`): window p's last o_part word enters its FIFO
                if hp == 1:
                    op_ready = [s_pv + win_out(tv, p) + LWT - 1 for p in range(tv.geometry.W_K)]
                else:                               # d > I (model only): every window after the last column chunk
                    op_ready = [s_pv + (hp - 1) * tv.occ + win_out(tv, tv.geometry.W_K - 1) + LWT - 1] * Poc
                la_go = sev["la_done"] + LA_GO_LAT if b else 0
                if b >= 3 and sev["first_word"] <= la_free[j - 2]:
                    la_slot_late += 1                                                         # 2 log(alpha) slots
                ob = st.block(la_go=la_go, op_ready=op_ready)
                op_free[j] = ob["op_free"]
                la_free[j] = ob["la_free"]
                ev.append(dict(r=r, b=b, h=h, qk_start=s_qk, qk_done=qk_done, go=go, sm_first=sev["first_word"],
                               sm_last=sev["last_word"], lp_first=sev["lp_first"], lp_last=sev["lp_last"],
                               la_done=sev["la_done"], la_go=la_go, p_done=p_done, pv_start=s_pv, pv_done=pv_done,
                               e=ob["exp_end"], ell=ob["log_end"]))
                qk_start_prev, pv_start_prev, pv_occ_prev = s_qk, s_pv, t_pv_occ[b]
            pv_last[h] = pv_done
            sm_done = sm.done()
            # log l copy (rd_row -> rd_ll -> the station's ll register): after the softmax's run and run r-1's o-carry
            # (its epilogue read log l); the O buffer is the station's e_mem (M2): the station holds the run's EXP
            # jobs until the previous run's O feed released it (`OcStation.fed`, edge X7)
            x = sm_done + 1
            if resources and r > 0:
                x = max(x, run_done_prev + 1)
            ll_go = x + S + LL_LAT
            epi = st.epilogue(ll_go=ll_go)["run_done"]                                    # epilogue EXP waves -> O8
            # att_ofeed (go = the run's last O8 word): row t of the run is read (LOADO + 1 words) and burst into
            # head h's O FIFO; rows every RP cycles, the first burst OFEED_LAT + LOADO after go
            o_start = epi + LOADO + OFEED_LAT                                             # the run's first O8 word
            for t in range(S):
                row_last[h][rh * S + t] = o_start + t * RP + LOADO - 1
            fed = row_last[h][rh * S + S - 1] + 1                                         # the O buffer is free
            st.fed(fed)                                          # e_mem: the next run's EXP jobs may write it
            runs.append(dict(r=r, h=h, t0=t0, sm_done=sm_done, ll_go=ll_go, epi=epi, fed=fed, oproj_start=o_start))
            run_done_prev, fed_prev, ll_go_prev = epi, fed, ll_go
    # the O-projection (att_ocat): token u's burst is offered CAT_LAT after its last head's row is in the O FIFO
    # (row visible, pop step 0, form word 0); the gemm takes it at its pass's load start (certified pass grammar)
    offer = [max(row_last[h][u] for h in range(n_heads)) + CAT_LAT for u in range(n_seq)]
    oev = schedule_pass_sequence(offer, R, C, BUF, P, rows=rows_o, cols=cols_o)
    for u in range(1, n_seq):                          # att_ocat's own turnaround never binds (the gemm's is longer)
        if oev[u]["load_start"] < oev[u - 1]["load_start"] + LOADK + CAT_LAT - 1:
            raise RuntimeError(f"O-projection token {u}: att_ocat turnaround binds (model incomplete)")
    y_last = [e["drain_end"] + k_y for e in oev]
    for x in runs:
        if x["h"] == n_heads - 1:
            rh = x["r"] - (n_heads - 1) * Rb
            x["oproj_done"] = y_last[rh * S + S - 1]
    total = y_last[-1] + 1                                                                # done: after the last y word
    stages = dict(projection=proj["busy"], qk_farm=n_heads * Rb * B * t_qk_occ,
                  softmax=sum(x["sm_done"] - x["t0"] for x in runs), pv_farm=n_heads * Rb * sum(t_pv_occ),
                  ocarry=n_heads * Rb * B * t_oc_log, oproj=oev[-1]["drain_end"] - oev[0]["load_start"])
    return dict(total=total, events=ev, runs=runs, stages=stages, proj=proj, t_O=t_O, passes=passes,
                t_qk_occ=t_qk_occ, t_pv_occ=t_pv_occ, t_oc_log=t_oc_log, t_oc_exp=t_oc_exp,
                la_slot_late=la_slot_late, y_last=y_last)


def attention_cycle_model(R: int, C: int, n_seq: int, d: int, dm: int, n_exp: int | None = None,
                          n_E: int | None = None, bkv: int = BKV, p_proj: int = 1, acc_buf: int = 2,
                          la_buf: int = 2, pass_extent: str = "used", tree_pipe_t: int | None = None,
                          _scale: dict | None = None, pv_buf: int = 2, n_Ep: int | None = None,
                          n_oc: int | None = None, sm_extent: str | None = None,
                          n_heads: int = 1) -> AttentionCycles:
    """Cold latency of the blocked FlashAttention layer (charter §4.4) with its bracket: n_heads heads (M5; 1 = the
    head) with dedicated projections taking one engine in turn, one O-projection over the concatenated outputs.

    total    : the kernel with `acc_buf` QK-accumulator buffers (default 2, D12) and all resource edges (overlaps 0)
    total_lo : unlimited buffers — no X2 / X3 / X4 holds, no o-state (X5 run boundary) or O-buffer (X7: the next run's
               EXP jobs wait for the O feed, M2) hold; the
               single stations (QK and PV farms X1 / X9, the softmax unit X6, the O-projection) stay serial
    total_hi : the single-buffered QK accumulator (acc_buf = 1)
    pv_buf   : PV accumulator halves (2, 2026-10-08: the o-carry station reads o_part during its LOG feeds, so one half
               would serialize the PV tile behind the station's last LOG wave; the halves cost depth, not memory blocks)
    n_Ep, n_oc : override the PV farm and the o-carry group sizes (an RTL configuration; default derived)
    sm_extent : the online softmax's pass extent (default `pass_extent`); 'full' = the certified softmax RTL (P29 not
               yet in it: at R != C it loads R rows and drains C columns per pass)"""
    check_domain(n_seq, d, dm, R, C, bkv)
    if pass_extent not in ("full", "used"):
        raise ValueError("pass_extent must be 'full' or 'used'")
    if p_proj < 1 or acc_buf < 1 or la_buf < 1:
        raise ValueError("p_proj, acc_buf and la_buf must be >= 1")
    g = geometry(R, C, n_seq, d, bkv)
    n_exp = select_n_exp(R, C, n_seq, d, bkv, pass_extent) if n_exp is None else int(n_exp)
    if not (pow2(n_exp) and n_exp <= max(1, g["P_sm"])):
        raise ValueError(f"n_exp must be a power of two <= P_sm = {g['P_sm']} (got {n_exp})")
    fs = farm_sizes(R, C, n_seq, d, n_exp, n_E, bkv, pass_extent)
    if n_Ep is not None:                                 # an RTL configuration (default: n_E' = n_E)
        fs["n_Ep"] = int(n_Ep)
    if n_oc is not None:                                 # an RTL configuration (default: the floor pick, n_x = n_oc)
        fs["n_oc"] = fs["n_x"] = int(n_oc)
    if fs["n_E"] < 1:
        raise ValueError("n_E must be >= 1")
    scale = _scale or {}
    args = (R, C, n_seq, d, dm, n_exp, fs["n_E"], fs["n_Ep"], fs["n_x"], fs["n_oc"], bkv, p_proj)
    if n_heads < 1:
        raise ValueError("n_heads must be >= 1")
    ag = dict(sm_extent=sm_extent, n_heads=n_heads)
    main = _head(*args, acc_buf, la_buf, pass_extent, True, tree_pipe_t, scale, pv_buf, **ag)
    lo = _head(*args, acc_buf, la_buf, pass_extent, False, tree_pipe_t, scale, pv_buf, **ag)
    hi = _head(*args, 1, la_buf, pass_extent, True, tree_pipe_t, scale, pv_buf, **ag) if acc_buf != 1 else main
    st = main["stages"]
    binding = max(st, key=st.get)
    proj_frac = min(1.0, (main["proj"]["t_p"][-1] + main["t_O"]) / main["total"])
    return AttentionCycles(
        total=main["total"], total_lo=lo["total"], total_hi=hi["total"], ii_cycles=max(st.values()),
        binding=binding, proj_bound_fraction=proj_frac, stages=st, S=g["S"], Rb=g["Rb"], B=g["B"],
        V=main["proj"]["V"], H=cdiv(d, C), V_O=cdiv(n_heads * d, R), H_O=cdiv(dm, C), n_exp=n_exp, n_E=fs["n_E"], n_Ep=fs["n_Ep"], n_x=fs["n_x"],
        n_oc=fs["n_oc"], n_pool=main["proj"]["n_pool"], p_proj=p_proj, acc_buf=acc_buf, pass_extent=pass_extent,
        cycles_basis=CYCLES_BASIS.format(ext=pass_extent), la_slot_late=main["la_slot_late"],
        events=main["events"], runs=main["runs"], n_heads=n_heads, passes=main["passes"])


# =====================================================================================================
# Crossbar count, ledger, candidate grid (charter §6, §7)
# =====================================================================================================
def attention_dpe_count(cm: AttentionCycles, d: int, dm: int, R: int, C: int) -> dict:
    """Weight-bearing (dedicated per projection, head and layer) vs shareable engine crossbars (the layer: n_heads
    heads' projections + one O-projection over K = n_heads * d)."""
    V = cdiv(dm, R)
    H = cdiv(d, C)
    # V > 1: the LOG converters (one per operand and projection copy) belong to the head's projections (dedicated)
    weight = cm.n_heads * (3 * V * H * cm.p_proj + 3 * cm.n_pool) + cm.V_O * cm.H_O * cm.p_proj
    engine = cm.n_E + cm.n_Ep + cm.n_x + (cm.n_exp + 2) + cm.n_oc
    return dict(dpe_weight=weight, dpe_engine=engine, dpe_count=weight + engine)


def attention_ledger(cm: AttentionCycles, n_seq: int, d: int, dm: int, R: int, C: int, bkv: int = BKV) -> dict:
    """Operation counts per layer (charter §6; n_heads heads, one O-projection over K = n_heads * d),
    value-independent; priced later with `cost.py`."""
    g = geometry(R, C, n_seq, d, bkv)
    V, H = cdiv(dm, R), cdiv(d, C)
    Rb, B, S = g["Rb"], g["B"], g["S"]
    nh = cm.n_heads
    sm = softmax_online_cycle_model(S, bkv, R, C, BUF, P, cm.n_exp, n_seq, pass_extent=cm.pass_extent)
    return dict(
        projection_passes=nh * 3 * V * H * n_seq,
        projection_tree_adds=nh * 3 * n_seq * d * (V - 1),
        pool_passes=nh * 3 * n_seq * cdiv(d, g["I"]) if V > 1 else 0,                # one per token (chunk)
        qk_passes=nh * Rb * B * dimm_tile_sim.geometry(S, bkv, d, R, C, cm.n_E).PT,  # per-k windows (= P_mm on
        pv_passes=nh * Rb * B * dimm_tile_sim.geometry(d, S, bkv, R, C, cm.n_Ep).PT, # every DSE shape)
        tile_log_adds=nh * 2 * Rb * B * S * bkv * d, tile_acc_adds=nh * 2 * Rb * B * S * bkv * d,
        narrow_compares=nh * Rb * B * S * bkv,
        softmax_exp_passes=nh * Rb * sum(sm.exp_passes), softmax_carry_exp_passes=nh * Rb * sm.carry_exp_passes,
        softmax_carry_log_passes=nh * Rb * sm.carry_log_passes,
        ocarry_exp_passes=nh * Rb * ((B - 1) + 1) * g["P_oc"], ocarry_log_passes=nh * Rb * B * g["P_oc"],
        ocarry_adds=nh * Rb * (B - 1) * 2 * S * d,
        oproj_passes=n_seq * cdiv(dm, C) * cdiv(nh * d, R),
        oproj_tree_adds=n_seq * dm * (cdiv(nh * d, R) - 1))


def candidate_grid(R: int, C: int, n_seq: int, d: int, dm: int, bkv: int = BKV, pass_extent: str = "used",
                   scales=(1, 2, 4, 8), p_projs=(1, 2, 4)) -> list:
    """§4.1 / D2b: n_exp (powers of two <= P_sm) x farm scale {1, 1/2, 1/4, 1/8} x P_PROJ {1, 2, 4}."""
    g = geometry(R, C, n_seq, d, bkv)
    ref = select_n_exp(R, C, n_seq, d, bkv, pass_extent)
    out, n_exp = [], 1
    while n_exp <= max(1, g["P_sm"]):
        for sdiv in scales:
            n_E = max(1, (d * n_exp) // sdiv)
            for pp in p_projs:
                cm = attention_cycle_model(R, C, n_seq, d, dm, n_exp, n_E, bkv, pp, pass_extent=pass_extent)
                row = dict(R=R, C=C, n_seq=n_seq, d=d, dm=dm, n_exp=n_exp, farm_scale=f"1/{sdiv}", n_E=cm.n_E,
                           p_proj=pp, total=cm.total, total_lo=cm.total_lo, total_hi=cm.total_hi,
                           ii_cycles=cm.ii_cycles, binding=cm.binding,
                           pick="reference" if (n_exp == ref and sdiv == 1 and pp == 1) else "")
                row.update(attention_dpe_count(cm, d, dm, R, C))
                out.append(row)
        n_exp *= 2
    return out


# =====================================================================================================
# Values through the certified kernels (ATT-1c-ii, GATE 1 vs v2/sim/oracle/attention_ref.py)
# =====================================================================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "oracle"))
import attention_ref as aref  # noqa: E402
import nldpe_ref as _nref  # noqa: E402
from softmax_online_sim import NldpeSoftmaxOnline  # noqa: E402


class NldpeAttention:
    """One head's values through the certified bricks (projections, tile DIMMs, online softmax).

    Stages computed here: projection log bytes (V = 1: the primitive in LOG mode per column tile, P29 rows =
    d_model; V > 1: `NldpeGemm` REGULAR + one identity LOG pass per token, extent d x d — the station's converter),
    the QK tiles (`NldpeDimmTile`, window-major, M = S, N = B_kv, K = d), the D1 narrowing, the online softmax per run
    (`NldpeSoftmaxOnline`), the untransposed PV tiles (M = S, N = d, K = B_kv; A = log|P| unsigned, B = V signed;
    d > I as ceil(d / I) column chunks, model only), the o-carry (contract accepted 2026-10-08, `review_R2.md`: SO = log2 n_seq; rescale EXP and
    compaction LOG through identity passes of the primitive), the epilogue (one EXP set) and the O-projection
    (`NldpeGemm`, P29 extent "used"). `gate1` certifies every stage against `attention_ref.head_stages`."""

    def __init__(self, R: int, C: int, n_seq: int, d: int, dm: int, n_exp: int | None = None,
                 n_E: int | None = None, bkv: int = BKV, score_shift: int = aref.SCORE_SHIFT,
                 n_Ep: int | None = None) -> None:
        check_domain(n_seq, d, dm, R, C, bkv)
        self.R, self.C, self.n_seq, self.d, self.dm, self.bkv = R, C, n_seq, d, dm, bkv
        g = geometry(R, C, n_seq, d, bkv)
        self.S, self.Rb, self.B = g["S"], g["Rb"], g["B"]
        self.n_exp = select_n_exp(R, C, n_seq, d, bkv) if n_exp is None else n_exp
        self.n_E = d * self.n_exp if n_E is None else n_E
        self.score_shift = score_shift
        self.n_Ep = self.n_E if n_Ep is None else n_Ep
        I = min(R, C)
        self.qk = dimm_tile_sim.NldpeDimmTile(self.S, bkv, d, R, C, self.n_E, True, True)
        self.pv = dimm_tile_sim.NldpeDimmTile(self.S, min(d, I), bkv, R, C, self.n_Ep, False, True)
        self.sm = NldpeSoftmaxOnline(self.S, R=R, C=C, n_exp=self.n_exp, Bkv=bkv, N=n_seq)
        self.LN = n_seq.bit_length() - 1
        self.SO = self.LN                                   # o-carry compaction shift (R2, decision A)
        I = min(R, C)
        self._eye = {}
        for mode in (_nref.MODE_EXP, _nref.MODE_LOG):       # o-carry identity crossbars (P29 extent I x I)
            e = prim.NldpeDpe(R, C, BUF, P, rows_used=I, cols_used=I)
            e.program_weights(np.eye(R, C, dtype=np.int8))
            self._eye[mode] = e

    def _conv(self, x: np.ndarray, mode: int) -> np.ndarray:
        """Identity conversion of an int8 array through crossbar passes of capacity I (F5/F6)."""
        flat = np.asarray(x, dtype=np.int64).reshape(-1).astype(np.int8)
        out, _ = dimm_sim.identity_pass(flat, mode, self.R, self.C, BUF, P, dpe=self._eye[mode])
        return out.reshape(np.shape(x))

    def project(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Log bytes [n_seq, d] of X @ W as the stations produce them."""
        X = np.asarray(X, dtype=np.int8)
        W = np.asarray(W, dtype=np.int8)
        R, C, dm, d = self.R, self.C, self.dm, self.d
        if cdiv(dm, R) == 1:
            L = np.zeros((X.shape[0], d), dtype=np.int64)
            Xp = np.zeros((X.shape[0], R), dtype=np.int8)
            Xp[:, :dm] = X
            for h in range(cdiv(d, C)):
                cols = min(C, d - h * C)
                Wt = np.zeros((R, C), dtype=np.int8)
                Wt[:dm, :cols] = W[:, h * C:h * C + cols]
                dpe = prim.NldpeDpe(R, C, BUF, P, rows_used=dm, cols_used=cols)
                dpe.program_weights(Wt)
                res = dpe.run_workload(Xp, _nref.MODE_LOG)
                L[:, h * C:h * C + cols] = res.out_stream[:, :cols].view(np.int8)
            return L
        gm = gemm_sim.NldpeGemm(X.shape[0], dm, d, R, C, BUF, P, pass_extent="used")
        gm.program_weights(W)
        x8 = np.asarray(gm.run(X).out8, dtype=np.int8)
        conv = prim.NldpeDpe(R, C, BUF, P, rows_used=min(d, R), cols_used=min(d, C))   # the LOG converter
        conv.program_weights(np.eye(R, C, dtype=np.int8))
        L = np.zeros((X.shape[0], d), dtype=np.int64)
        for h in range(cdiv(d, min(R, C))):                  # one token per pass (d > I: chunks, model only)
            w = min(min(R, C), d - h * min(R, C))
            Xc = np.zeros((X.shape[0], R), dtype=np.int8)
            Xc[:, :w] = x8[:, h * min(R, C):h * min(R, C) + w]
            L[:, h * min(R, C):h * min(R, C) + w] = conv.run_workload(Xc, _nref.MODE_LOG).out_stream[:, :w].view(np.int8)
        return L

    def values(self, X, WQ, WK, WV, WO=None) -> dict:
        n, d, S, B, bkv = self.n_seq, self.d, self.S, self.B, self.bkv
        LN, SO = self.LN, self.SO
        LQ, LK, LV = (self.project(X, W) for W in (WQ, WK, WV))
        acc = np.zeros((n, n), dtype=np.int64)
        s8 = np.zeros((n, n), dtype=np.int64)
        logp = np.zeros((B, n, bkv), dtype=np.int64)
        log_a = np.zeros((B, n), dtype=np.int64)
        log_l = np.zeros(n, dtype=np.int64)
        o_part = np.zeros((B, n, d), dtype=np.int64)
        oq = np.zeros((B, n, d), dtype=np.int64)
        lo_f = np.zeros((n, d), dtype=np.int64)
        so_f = np.zeros((n, d), dtype=np.int64)
        O8 = np.zeros((n, d), dtype=np.int64)
        for r in range(self.Rb):
            rows = slice(r * S, (r + 1) * S)
            for b in range(B):
                keys = slice(b * bkv, (b + 1) * bkv)
                acc[rows, keys] = self.qk.run(LQ[rows], LK[keys].T)["acc"]
            s8[rows] = aref.narrow(acc[rows], self.score_shift)
            sv = self.sm.values(s8[rows])
            logp[:, rows], log_a[:, rows], log_l[rows] = sv["logp"], sv["log_a"], sv["log_l_final"]
            lo = so = None
            for b in range(B):
                keys = slice(b * bkv, (b + 1) * bkv)
                I = min(self.R, self.C)
                for h in range(cdiv(d, I)):                 # untransposed PV tile (d > I: column chunks)
                    cols = slice(h * I, min(d, (h + 1) * I))
                    o_part[b, rows, cols] = self.pv.run(sv["logp"][b], LV[keys, cols])["acc"]
                # o-carry: rescale (EXP set) for b >= 1, signed add, compaction, LOG set
                if b == 0:
                    o_new = o_part[0, rows].copy()
                else:
                    t = np.clip(sv["log_a"][b][:, None] + lo, -128, 127)
                    e = self._conv(t, _nref.MODE_EXP).view(np.uint8).astype(np.int64)
                    o_new = (np.where(so.astype(bool), -e, e) << SO) + o_part[b, rows]
                so = (o_new < 0).astype(np.int64)
                oq[b, rows] = np.minimum(np.abs(o_new) >> SO, 127)
                lo = self._conv(oq[b, rows], _nref.MODE_LOG).astype(np.int64)
            # epilogue: one EXP set, scale 2^(SO - LN), saturate to int8
            tE = np.clip(lo - sv["log_l_final"][:, None], -128, 127)
            eE = self._conv(tE, _nref.MODE_EXP).view(np.uint8).astype(np.int64)
            eE = eE << (SO - LN) if SO >= LN else eE >> (LN - SO)
            O8[rows] = np.where(so.astype(bool), np.maximum(-eE, -128), np.minimum(eE, 127))
            lo_f[rows], so_f[rows] = lo, so
        out = {"LQ": LQ, "LK": LK, "LV": LV, "acc": acc, "s8": s8, "logp": logp, "log_a": log_a,
               "log_l_final": log_l, "o_part": o_part, "oq": oq, "lo_final": lo_f, "so_final": so_f, "O8": O8}
        if WO is not None:                                  # O-projection (P29 extent "used")
            go = gemm_sim.NldpeGemm(n, d, self.dm, self.R, self.C, BUF, P, pass_extent="used")
            go.program_weights(np.asarray(WO, dtype=np.int8))
            out["y8"] = go.run(O8.astype(np.int8)).out8.astype(np.int64)
        return out

    def gate1(self, X, WQ, WK, WV, WO) -> dict:
        """GATE 1: kernel values == attention_ref.head_stages on every stage (raises on mismatch)."""
        v = self.values(X, WQ, WK, WV, WO)
        o = aref.head_stages(X, WQ, WK, WV, WO, self.bkv, self.R, self.C, self.score_shift)
        want = {"LQ": o["LQ"], "LK": o["LK"], "LV": o["LV"], "acc": o["acc"], "s8": o["s8"],
                "logp": o["softmax"]["logp"], "log_a": o["softmax"]["log_a"],
                "log_l_final": o["softmax"]["log_l_final"], "o_part": o["o_part"],
                "oq": o["ocarry"]["oq"], "lo_final": o["ocarry"]["lo_final"], "so_final": o["ocarry"]["so_final"],
                "O8": o["O8"], "y8": o["y8"]}
        for k, w in want.items():
            if not np.array_equal(np.asarray(v[k]), np.asarray(w)):
                raise AssertionError(f"GATE 1 mismatch on {k} ({self.R}x{self.C}, n={self.n_seq}, d={self.d}, "
                                     f"dm={self.dm}, Bkv={self.bkv})")
        return v


def layer_gate1(R: int, C: int, X, WQ, WK, WV, WO, n_exp: int | None = None, bkv: int = BKV) -> dict:
    """GATE 1 of the layer (M5): every head through `NldpeAttention` (its own projections, the engine's kernels) ==
    `attention_ref.layer_stages` head by head, and the O-projection over the concatenated O8 (`NldpeGemm`,
    K = H*d, P29 'used') == its y8."""
    WQ, WK, WV = (np.asarray(w, dtype=np.int8) for w in (WQ, WK, WV))
    H, dm, d = WQ.shape
    n = np.shape(X)[0]
    o = aref.layer_stages(X, WQ, WK, WV, WO, bkv, R, C)
    O8 = []
    for h in range(H):
        at = NldpeAttention(R, C, n, d, dm, n_exp=n_exp, bkv=bkv)
        v = at.values(X, WQ[h], WK[h], WV[h])
        oh = o["heads"][h]
        for k, w in (("s8", oh["s8"]), ("O8", oh["O8"]), ("LQ", oh["LQ"]), ("o_part", oh["o_part"])):
            if not np.array_equal(np.asarray(v[k]), np.asarray(w)):
                raise AssertionError(f"layer GATE 1 mismatch: head {h} {k} ({R}x{C}, n={n}, d={d}, H={H})")
        O8.append(v["O8"])
    cat = np.concatenate(O8, axis=1)
    go = gemm_sim.NldpeGemm(n, H * d, dm, R, C, BUF, P, pass_extent="used")
    go.program_weights(np.asarray(WO, dtype=np.int8))
    y8 = go.run(cat.astype(np.int8)).out8.astype(np.int64)
    if not np.array_equal(y8, o["y8"]):
        raise AssertionError(f"layer GATE 1 mismatch: y8 ({R}x{C}, n={n}, d={d}, H={H})")
    return dict(O8cat=cat, y8=y8)


def _values_self_test() -> None:
    """GATE 1 of the kernel composition on micro geometries: V = 1 / V = 2, H = 2, R_b > 1, R != C."""
    rng = np.random.default_rng(2026_10_08)
    n = 0
    for (R, C, ns, d, dm, bkv, ne) in ((64, 64, 16, 8, 16, 8, 1), (64, 64, 32, 16, 48, 8, 2),
                                       (64, 64, 32, 16, 96, 16, 1), (32, 32, 64, 16, 32, 8, 2),
                                       (64, 32, 32, 64, 32, 8, 1), (128, 64, 64, 32, 160, 16, 2)):
        X = rng.integers(-128, 128, size=(ns, dm))
        Ws = [rng.integers(-128, 128, size=(dm, d)) for _ in range(3)]
        WO = rng.integers(-128, 128, size=(d, dm))
        NldpeAttention(R, C, ns, d, dm, n_exp=ne, bkv=bkv).gate1(X, *Ws, WO)
        n += 1
    nl = 0
    for (R, C, ns, d, dm, bkv, H) in ((64, 64, 32, 16, 32, 8, 2), (64, 64, 32, 16, 48, 8, 3),
                                      (64, 64, 32, 32, 96, 8, 3), (32, 32, 64, 16, 32, 8, 2)):
        X = rng.integers(-128, 128, size=(ns, dm))
        WQ, WK, WV = (rng.integers(-128, 128, size=(H, dm, d)) for _ in range(3))
        WO = rng.integers(-128, 128, size=(H * d, dm))
        layer_gate1(R, C, X, WQ, WK, WV, WO, bkv=bkv)
        nl += 1
    print(f"attention_sim values: ALL PASS ({n} micro geometries + {nl} layers (H = 2, 3; O-projection V_O = 2): "
          "projections, QK tiles, narrowing, online "
          "softmax, PV tiles, o-carry, epilogue, O-projection == attention_ref, through the certified kernels)")


# =====================================================================================================
# Self-test
# =====================================================================================================
def _self_test() -> None:
    fails: list = []
    # att_score port (rework M1): with every word already in its FIFO and the softmax ready, the stream equals the
    # previously pinned contiguous form (first word go + SCORE_ACC_LAT, last word go + _score_pace); words that
    # arrive later or a softmax that admits later only delay it, never reorder it
    for S_, bkv_, I_ in ((32, 8, 64), (128, 16, 256), (64, 16, 64), (16, 8, 256), (128, 16, 128)):
        blk, lw = S_ * bkv_, cdiv(I_, EPS)
        wk = max(1, blk // I_)
        early = score_stream(100, [[0] * lw for _ in range(wk)], wk, lw, min(blk, I_), blk)
        if (early["go"], early["offers"][0], early["offers"][-1]) != (100, 100 + SCORE_ACC_LAT, 100 + _score_pace(blk, I_)):
            fails.append(f"score_stream all-ready {S_}x{bkv_} @ I={I_}: {early['offers'][0]}, {early['offers'][-1]}")
        late = score_stream(100, [[300 + 3 * (w * lw + c) for c in range(lw)] for w in range(wk)], wk, lw,
                            min(blk, I_), blk, ready_from=150)
        if not all(b2 > a2 for a2, b2 in zip(late["offers"], late["offers"][1:])) or late["go"] != 301 or \
                any(x < y for x, y in zip(late["offers"], early["offers"])):
            fails.append(f"score_stream late words {S_}x{bkv_} @ I={I_}")

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # (a) availability 0 => the certified online-softmax recurrence, exactly (both extents)
    na = 0
    for R, C in [(256, 256), (128, 64), (512, 128), (1024, 256), (64, 64)]:
        I = min(R, C)
        for S in [s for s in (8, 16, 64, 128, 256) if s <= I]:
            for N in (S, 4 * S):
                for bkv in [k for k in (8, 16, 32) if k <= min(N, I)]:
                    for ne in [n for n in (1, 2, 4) if n <= max(1, cdiv(S * bkv, I))]:
                        for ext in ("full", "used"):
                            cert = softmax_online_cycle_model(S, bkv, R, C, BUF, P, ne, N, pass_extent=ext)
                            ours = softmax_availability_model(S, bkv, R, C, ne, N, pass_extent=ext)
                            same = ours["total"] == cert.total and all(
                                all(o[k] == c[k] for k in ("first_word", "last_word", "sp_full", "carry_start",
                                                           "cexp_last", "retire"))
                                for o, c in zip(ours["blocks"], cert.blocks))
                            check(same, f"(a) {R}x{C} S={S} N={N} Bkv={bkv} n_exp={ne} {ext}")
                            na += 1
    # availability later than the stream only delays (monotone), never reorders
    base = softmax_availability_model(64, 16, 256, 256, 1, 256)
    late = softmax_availability_model(64, 16, 256, 256, 1, 256, avail=[500 * (b + 1) for b in range(16)])
    check(late["total"] > base["total"], "(a) late availability must delay the run")
    check(all(x["first_word"] >= 500 * (i + 1) for i, x in enumerate(late["blocks"])), "(a) availability honoured")
    # per-word offers: all-early offers == the certified stream; first-word-only offers == `avail`;
    # intra-block bubbles only move the block's last word (monotone)
    for R, C, S, N, bkv, ne in ((256, 256, 64, 256, 16, 1), (128, 64, 64, 128, 16, 2), (512, 128, 128, 512, 16, 4)):
        st = cdiv(S * bkv, EPS)
        Bn = N // bkv
        base = softmax_availability_model(S, bkv, R, C, ne, N, pass_extent="full")
        early = softmax_availability_model(S, bkv, R, C, ne, N, pass_extent="full", offers=[0] * (Bn * st))
        check(early["total"] == base["total"] and early["blocks"] == base["blocks"], f"(a) offers early {R}x{C}")
        av = [300 * b for b in range(Bn)]
        offs = [x for b in range(Bn) for x in [av[b]] + [0] * (st - 1)]
        a1 = softmax_availability_model(S, bkv, R, C, ne, N, avail=av, pass_extent="full")
        a2 = softmax_availability_model(S, bkv, R, C, ne, N, pass_extent="full", offers=offs)
        check(a1["total"] == a2["total"] and a1["blocks"] == a2["blocks"], f"(a) offers == avail {R}x{C}")
        bub = [x + 3 * (w // 7) for x, w in zip(offs, range(Bn * st))]
        a3 = softmax_availability_model(S, bkv, R, C, ne, N, pass_extent="full", offers=bub)
        check(a3["total"] >= a1["total"] and all(y["last_word"] >= x["last_word"]
                                                 for x, y in zip(a1["blocks"], a3["blocks"])), f"(a) bubbles {R}x{C}")

    # (b) the tile E stage == dimm_cycle_model's farm term with a pool-free plan
    for R, C in [(256, 256), (128, 64), (1024, 64)]:
        for ext in ("full", "used"):
            for n_E in (1, 8, 128):
                g = geometry(R, C, 128, 64)
                plan = dimm_sim.DimmPassPlan(passes_A=0, passes_B=0, passes_E=g["P_mm"])
                dm_ = dimm_sim.dimm_cycle_model(g["S"], BKV, 64, R, C, BUF, P, n_E=n_E, plan=plan, pass_extent=ext)
                check(dm_.T_E == conv_T(cdiv(g["P_mm"], n_E), R, C, ext), f"(b) {R}x{C} n_E={n_E} {ext}")

    # (c)-(j) over all 60 points x both extents, the head (n_heads = 1) and the layer (n_heads = dm / d, M5)
    for ext in ("used", "full"):
        for R, C in CONFIGS:
            for (n, d, dm) in SHAPES:
              for nh in (1, dm // d):
                tag = f"{R}x{C} {n}/{d}/{dm} H{nh} {ext}"
                cm = attention_cycle_model(R, C, n, d, dm, pass_extent=ext, n_heads=nh)
                # bracket and lower bounds
                check(cm.total_lo <= cm.total <= cm.total_hi, f"(c) bracket {tag}: {cm.total_lo} {cm.total} {cm.total_hi}")
                check(cm.total >= cm.ii_cycles, f"(c) total < busiest station {tag}")
                check(cm.la_slot_late == 0, f"(i) o-carry log(alpha) slot overwritten in use {tag}: {cm.la_slot_late}")
                if cm.V == 1 and ext == "used":                # the VR writer keeps up (proj_sim guard); log|P| A port fits
                    try:
                        proj_sim.geometry("vr", n, dm, d, R, C, BKV)
                        a_nw = pv_a_nw(min(d, min(R, C)))
                        dimm_tile_sim.check_domain(cm.S, min(d, min(R, C)), BKV, R, C, cm.n_Ep, a_nw, cm.n_exp)
                    except ValueError as e:
                        fails.append(f"(i) PV operand path {tag}: {e}")
                # schedule invariants (resource edges honoured)
                evs = cm.events
                for j in range(1, len(evs)):
                    check(evs[j]["qk_start"] >= evs[j - 1]["qk_start"], f"(i) QK order {tag}")
                    check(evs[j]["pv_start"] >= evs[j - 1]["pv_start"], f"(i) PV order {tag}")
                    check(evs[j]["ell"] >= evs[j - 1]["ell"] or evs[j]["b"] == 0, f"(i) o-carry order {tag}")
                for j in range(cm.acc_buf, len(evs)):
                    check(evs[j]["qk_start"] >= evs[j - cm.acc_buf]["sm_last"], f"(i) X2 accumulator {tag}")
                for j in range(2, len(evs)):                                   # la_buf = 2
                    check(evs[j]["lp_first"] >= evs[j - 2]["pv_done"] + G, f"(i) X4 LA half overwritten {tag}")
                for rr in cm.runs[1:]:
                    prev = cm.runs[rr["r"] - 1]
                    check(rr["t0"] >= prev["sm_done"] + cm.S + 2 + 1, f"(i) X6 re-arm {tag}")
                    # X7 (M2): the run's first EXP job (rescale of block 1, or the epilogue at B = 1) writes the O
                    # buffer (e_mem) only after the previous run's O feed released it
                    first_e = evs[rr["r"] * cm.B + 1]["e"] if cm.B > 1 else rr["epi"]
                    check(first_e > prev["fed"] + 1, f"(i) X7 O buffer (e_mem) written before it was fed {tag}")
                # layer (M5): X passes in head order, each after the previous pass; X11: pass h >= 2 after the
                # engine's last PV tile of head h - 2 (its operand region); y only for the last head's runs
                for h in range(1, cm.n_heads):
                    check(cm.passes[h] >= cm.passes[h - 1] + n + 1, f"(i) X pass order {tag}")
                    if h >= 2:
                        last = max(e["pv_done"] for e in evs if e["h"] == h - 2)
                        check(cm.passes[h] >= last + 2, f"(i) X11 operand region reuse {tag}")
                check(sum("oproj_done" in x for x in cm.runs) == cm.Rb, f"(i) O-projection runs {tag}")
                # counts
                cnt = attention_dpe_count(cm, d, dm, R, C)
                check(cnt["dpe_count"] == cnt["dpe_weight"] + cnt["dpe_engine"], f"(g) count split {tag}")
                led = attention_ledger(cm, n, d, dm, R, C)
                sm = softmax_online_cycle_model(cm.S, BKV, R, C, BUF, P, cm.n_exp, n, pass_extent=ext)
                check(led["softmax_exp_passes"] == nh * cm.Rb * sum(sm.exp_passes), f"(g) softmax passes {tag}")
                check(led["projection_passes"] == nh * 3 * cm.V * n, f"(g) projection passes {tag}")
                check(led["oproj_passes"] == n * cm.H_O * cdiv(nh * d, R), f"(g) O-projection passes {tag}")
                check(led["ocarry_exp_passes"] == led["ocarry_log_passes"], f"(g) o-carry set counts {tag}")
                # (e) the reference candidate meets the knee rule and its half does not
                g = geometry(R, C, n, d)
                F = max(g["W_st"], conv_T(1, R, C, ext))
                ne = cm.n_exp
                check(conv_T(cdiv(g["P_sm"], ne), R, C, ext) <= F or ne >= g["P_sm"], f"(e) knee {tag}")
                if ne > 1:
                    check(conv_T(cdiv(g["P_sm"], ne // 2), R, C, ext) > F, f"(e) half knee {tag}")
    # (d) more resources never slow the head; (j) projection copies
    for R, C in [(256, 256), (1024, 64), (128, 128)]:
        for (n, d, dm) in [(128, 64, 128), (512, 64, 128), (256, 64, 192)]:
            t = [attention_cycle_model(R, C, n, d, dm, n_E=f).total for f in (16, 32, 64, 128, 256)]
            check(all(b_ <= a for a, b_ in zip(t, t[1:])), f"(d) n_E monotone {R}x{C} {n}: {t}")
            tpp = [attention_cycle_model(R, C, n, d, dm, p_proj=p).total for p in (1, 2, 4)]
            check(all(b_ <= a for a, b_ in zip(tpp, tpp[1:])), f"(j) P_PROJ monotone {R}x{C} {n}: {tpp}")
            ps = [projection_schedule(R, C, n, d, dm, p)["t_p"][-1] for p in (1, 2)]
            if cdiv(dm, R) == 1:
                cmv = prim.cycle_model(1, R, C, P, BUF, rows=min(R, dm), cols=min(C, d))
                check(ps[0] - int(cmv.t_fill) - G == (n - 1) * int(cmv.t_steady), "(j) P=1 stream")
                check(ps[1] - int(cmv.t_fill) - G == ((n - 1) // 2) * int(cmv.t_steady), "(j) P=2 stream")
    # (i) monotone in every station time: scaling any station up never shortens the head
    rng = random.Random(7)
    keys = ("qk", "pv", "oc", "oproj", "proj")
    for _ in range(60):
        R, C = rng.choice([(256, 256), (128, 64), (512, 128), (1024, 256)])
        n, d, dm = rng.choice([(128, 64, 128), (512, 64, 128), (64, 32, 96)])
        lo = {k: rng.uniform(0.5, 1.5) for k in keys}
        hi = {k: v * rng.uniform(1.0, 1.5) for k, v in lo.items()}
        a = attention_cycle_model(R, C, n, d, dm, _scale=lo).total
        b_ = attention_cycle_model(R, C, n, d, dm, _scale=hi).total
        check(b_ >= a, f"(i) monotone in station times {R}x{C} {n}: {a} -> {b_}")
    # (h) domain guards
    for bad in [dict(n_seq=100), dict(d=48), dict(bkv=4), dict(R=96, C=96), dict(n_seq=4), dict(bkv=32, n_seq=16)]:
        kw = dict(R=256, C=256, n_seq=128, d=64, dm=128, bkv=16)
        kw.update(bad)
        try:
            attention_cycle_model(kw["R"], kw["C"], kw["n_seq"], kw["d"], kw["dm"], bkv=kw["bkv"])
        except ValueError:
            continue
        fails.append(f"(h) guard missed {bad}")
    # general tiling: d_head > C is accepted now (H = 2)
    check(attention_cycle_model(256, 64, 128, 128, 128).H == 2, "(h) d_head > C must be accepted")
    for bad in [dict(n_exp=3), dict(p_proj=0), dict(acc_buf=0), dict(pass_extent="half")]:
        try:
            attention_cycle_model(256, 256, 128, 64, 128, **bad)
        except ValueError:
            continue
        fails.append(f"(h) guard missed {bad}")
    # general projection tiling: d_head > C (H > 1), d_head > R (V_O > 1), d_head > I (windows span tokens)
    for (R, C, n, d, dm) in [(256, 64, 128, 128, 128), (128, 64, 128, 128, 192), (64, 64, 128, 128, 128),
                             (64, 32, 64, 64, 128), (256, 128, 256, 256, 256)]:
        tag = f"tiling {R}x{C} n={n} d={d} dm={dm}"
        cm = attention_cycle_model(R, C, n, d, dm)
        V, H, V_O, H_O = cdiv(dm, R), cdiv(d, C), cdiv(d, R), cdiv(dm, C)
        check((cm.V, cm.H, cm.V_O, cm.H_O) == (V, H, V_O, H_O), f"{tag}: tiling fields")
        cnt = attention_dpe_count(cm, d, dm, R, C)
        check(cnt["dpe_weight"] == 3 * V * H + V_O * H_O + (3 if V > 1 else 0),
              f"{tag}: weight crossbars {cnt['dpe_weight']}")
        check(cm.total_lo <= cm.total <= cm.total_hi and cm.total >= cm.ii_cycles, f"{tag}: bracket")
        if V > 1:
            check(cm.n_pool == 1, f"{tag}: one LOG converter per operand")
        led = attention_ledger(cm, n, d, dm, R, C)
        check(led["projection_passes"] == 3 * V * H * n and led["oproj_passes"] == n * H_O * V_O, f"{tag}: ledger")
        check(led["oproj_tree_adds"] == n * dm * (V_O - 1), f"{tag}: O-projection tree adds")
    # V > 1: one converter per operand and projection copy; the head model's operand readiness == the station RTL
    # (proj_sim) at the ViT rows, and the projection schedule's token times == the station's drains + G
    for (R, C) in [(128, 64), (128, 128), (128, 256)]:
        check(projection_schedule(R, C, 256, 64, 192, 2)["n_pool"] == 2, f"converters {R}x{C}")
        g_st = proj_sim.geometry("vr", 256, 192, 64, R, C)
        dr = [e["drain_end"] for e in proj_sim.proj_cycle_model(g_st, [0] * 1 + list(range(1, 256)))["tokens"]]
        tp1 = projection_schedule(R, C, 256, 64, 192, 1)["t_p"]
        off = {a - b for a, b in zip(tp1, dr)}
        check(len(off) == 1, f"V > 1 schedule == station drains + const {R}x{C}: offsets {sorted(off)[:4]}")
    # candidate grid: one reference row, every row counted
    grid = candidate_grid(256, 256, 128, 64, 128)
    check(sum(r["pick"] == "reference" for r in grid) == 1, "grid: exactly one reference row")
    check(all(r["dpe_count"] == r["dpe_weight"] + r["dpe_engine"] for r in grid), "grid: counts")
    for msg in fails[:40]:
        print("FAIL:", msg)
    if fails:
        raise SystemExit(f"attention_sim self-test: {len(fails)} FAIL")
    print(f"attention_sim self-test: ALL PASS ({na} certified-softmax equivalence runs; 60 points x 2 extents x "
          f"(head, layer H = d_model / d_head); "
          f"bracket, resource-edge invariants, monotonicity, counts, knee, guards)")


def _table() -> None:
    """The 60 DSE rows as layers (n_heads = d_model / d_head, M5)."""
    print(f"{'cfg':>9} {'shape':>13} H | {'total':>8} {'lo':>8} {'hi':>8} {'ii':>8} {'binding':>10} {'proj%':>5} | "
          f"{'n_exp':>5} {'n_E':>5} {'n_oc':>4} {'xbars':>5} ({'w':>2}+{'eng':>4})")
    for R, C in CONFIGS:
        for (n, d, dm) in SHAPES:
            cm = attention_cycle_model(R, C, n, d, dm, n_heads=dm // d)
            c = attention_dpe_count(cm, d, dm, R, C)
            print(f"{R:>4}x{C:<4} {n:>4}/{d}/{dm:<4} {cm.n_heads} | {cm.total:8d} {cm.total_lo:8d} {cm.total_hi:8d} {cm.ii_cycles:8d} "
                  f"{cm.binding:>10} {100 * cm.proj_bound_fraction:5.0f} | {cm.n_exp:5d} {cm.n_E:5d} {cm.n_oc:4d} "
                  f"{c['dpe_count']:5d} ({c['dpe_weight']:2d}+{c['dpe_engine']:4d})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", action="store_true", help="print the 60-point reference-candidate table (layers)")
    ap.add_argument("--values", action="store_true", help="GATE 1 of the value composition (micro geometries)")
    a = ap.parse_args()
    if a.table:
        _table()
    elif a.values:
        _values_self_test()
    else:
        _self_test()
