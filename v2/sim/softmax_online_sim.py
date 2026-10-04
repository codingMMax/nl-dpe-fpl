#!/usr/bin/env python3
"""softmax_online_sim.py — NL online-softmax fused behavior + timing simulator.

Companion to `v2/oracle/softmax_online_ref.py` (values) and
`v2/spec/softmax_online.md` (contract).

Machine model (mirrors `softmax_sim`: values and cycles come from ONE execution)
-------------------------------------------------------------------------------
Scores stream in **key blocks** of `Bkv` (B = S / Bkv blocks, block-major).

  A  block max    per-block row max (the intended block-local reference)
  B  EXP          the block-major flat `S^2` exp_input converted by `n_exp`
                  identity crossbars over **packed stride-I windows**; the
                  per-block partials `Sp_b[r]` accumulate from the drains
  C  combine      deferred-alpha `L = sum_b factor_b * Sp_b`, `lq = min(L>>log2 S,127)`
  D  LOG          the `S` `lq` values converted by `n_log` identity crossbars
  E  emit         `out = clamp(s - m - ls)`; the `S^2` output is NOT counted

The **same `packed_windows` list** drives both the value conversion (through the
certified `dpe` primitive) and the event schedule, and the schedule's pass counts
are the ones the conversion actually issued. This is the property the user
required: the cycle model cannot drift from the behavior.

`done` = results READY; the `S^2` serialisation is reported separately (here 0).

Run:  python3 v2/sim/softmax_online_sim.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import nldpe_ref as nref  # noqa: E402
import nldpe_sim as prim  # noqa: E402
import softmax_online_ref as oref  # noqa: E402
from softmax_sim import _convert_packed_stream  # noqa: E402
from pass_engine import crossbar_total, packed_windows, \
    schedule_pass_sequence  # noqa: E402


# ---------------------------------------------------------------------------
# Check model (non-contract): phase envelope for the blocked machine.
# ---------------------------------------------------------------------------
@dataclass
class SoftmaxOnlinePassPlan:
    """Pass counts the schedule actually issues (schedule-injected)."""

    blocks: int
    passes_exp: int
    passes_log: int
    passes_factor: int = 0


@dataclass
class SoftmaxOnlineCycleModel:
    """Phase-envelope lower bound (check only, NOT the timing contract)."""

    S: int
    R: int
    C: int
    BUF: int
    P: int
    I: int
    Bkv: int
    blocks: int
    n_exp: int
    n_log: int
    stream_bytes: int
    clb_pipe: int
    passes_exp: int
    passes_log: int
    passes_factor: int
    xbar_passes_exp: int
    xbar_passes_log: int
    xbar_passes_factor: int
    T_stream: int
    T_exp: int
    T_factor: int
    T_combine: int
    T_log: int
    output_cycles: int
    total: int


def softmax_online_cycle_model(S, R=256, C=256, BUF=40, P=8, Bkv=16,
                               n_exp=1, n_log=1, clb_pipe=1,
                               stream_bytes=None, *,
                               plan: SoftmaxOnlinePassPlan,
                               ) -> SoftmaxOnlineCycleModel:
    I = min(R, C)
    if stream_bytes is None:
        stream_bytes = max(BUF // 8, 1)
    xbar_exp = -(-plan.passes_exp // n_exp) if plan.passes_exp else 0
    xbar_fac = -(-plan.passes_factor // n_exp) if plan.passes_factor else 0
    xbar_log = -(-plan.passes_log // n_log) if plan.passes_log else 0
    T_stream = -(-(S * S) // stream_bytes)
    T_exp = crossbar_total(xbar_exp, R, C, P, BUF)
    T_factor = crossbar_total(xbar_fac, R, C, P, BUF)
    T_combine = int(np.ceil(np.log2(max(plan.blocks, 1)))) + clb_pipe
    T_log = crossbar_total(xbar_log, R, C, P, BUF)
    total = T_stream + T_combine + T_log     # lower-bound envelope
    return SoftmaxOnlineCycleModel(
        S=S, R=R, C=C, BUF=BUF, P=P, I=I, Bkv=Bkv, blocks=plan.blocks,
        n_exp=n_exp, n_log=n_log, stream_bytes=stream_bytes, clb_pipe=clb_pipe,
        passes_exp=plan.passes_exp, passes_log=plan.passes_log,
        passes_factor=plan.passes_factor,
        xbar_passes_exp=xbar_exp, xbar_passes_log=xbar_log,
        xbar_passes_factor=xbar_fac,
        T_stream=T_stream, T_exp=T_exp, T_factor=T_factor,
        T_combine=T_combine, T_log=T_log,
        output_cycles=S * S, total=total)


@dataclass
class SoftmaxOnlineStages:
    """Staged diagnostics (mirror the datapath)."""

    blk_max: np.ndarray       # int32 [B, S]
    blk_Sp: np.ndarray        # int64 [B, S]
    factor: np.ndarray        # int64 [B, S]
    L: np.ndarray             # int64 [S]
    lq: np.ndarray            # int32 [S]
    ls: np.ndarray            # int32 [S]
    softmax_out: np.ndarray   # int8 [S, S]


@dataclass
class SoftmaxOnlineResult:
    softmax_out: np.ndarray
    ready_cycles: int             # `done`: last result value READY
    serialize_cycles: int         # 0 (output not counted)
    passes_exp: int
    passes_log: int
    passes_factor: int
    cycle_model: SoftmaxOnlineCycleModel
    stages: SoftmaxOnlineStages | None = None
    timeline: list = field(default_factory=list)

    @property
    def used_cycles(self) -> int:
        return self.ready_cycles

    @property
    def compute_cycles(self) -> int:
        return self.ready_cycles

    @property
    def emit_start(self) -> int:
        return self.ready_cycles

    @property
    def drain_cycles(self) -> int:
        return self.serialize_cycles


class NldpeSoftmaxOnline:
    """Online (blocked, deferred-alpha) softmax — values + bundled timing."""

    def __init__(self, S, R=256, C=256, BUF=40, P=8, n_exp=1, n_log=1,
                 Bkv=16, clb_pipe=1, n_fac=None) -> None:
        assert S > 0 and (S & (S - 1)) == 0, "S must be a power of two"
        assert Bkv >= 1 and S % Bkv == 0, "Bkv must divide S"
        self.S, self.R, self.C, self.BUF, self.P = S, R, C, BUF, P
        self.I = min(R, C)
        self.Bkv = Bkv
        self.B = S // Bkv
        self.n_exp, self.n_log = n_exp, n_log
        self.n_fac = n_exp if n_fac is None else n_fac   # dedicated factor bank
        self.clb_pipe = clb_pipe
        self.stream_bytes = max(BUF // 8, 1)
        self.clb_tree_latency = int(np.ceil(np.log2(S))) + clb_pipe
        self.clb_clamp_latency = 1 + clb_pipe
        self.combine_latency = int(np.ceil(np.log2(self.B))) + clb_pipe
        self.prologue = 6
        # Certified identity crossbars (EXP bank, dedicated factor bank, LOG bank).
        eye = np.eye(R, C, dtype=np.int8)

        def _eye():
            d = prim.NldpeDpe(R, C, BUF, P)
            d.program_weights(eye)
            return d
        self._exp_crossbars = [_eye() for _ in range(n_exp)]
        self._fac_crossbars = [_eye() for _ in range(self.n_fac)]
        self._log_crossbars = [_eye() for _ in range(n_log)]

    # -- values + timing from a single execution ---------------------------
    def run(self, scores: np.ndarray,
            collect_stages: bool = False) -> SoftmaxOnlineResult:
        """Run the blocked softmax: values (certified crossbars) + cycles.

        The EXP / factor / LOG conversions run through the certified `dpe` and
        use the **same `packed_windows` list** as the event schedule, so the
        cycle model is a by-product of the executed schedule.
        """
        scores = np.asarray(scores, dtype=np.int8)
        assert scores.shape == (self.S, self.S), \
            f"scores must be [{self.S}, {self.S}]"
        S, Bkv, B, I = self.S, self.Bkv, self.B, self.I
        BLKEL = S * Bkv
        log2_S = S.bit_length() - 1
        s32 = scores.astype(np.int32)

        # ---- block-local row max (intended difference from the conventional)
        blk_max = np.empty((B, S), dtype=np.int32)
        for b in range(B):
            blk_max[b] = s32[:, b * Bkv:(b + 1) * Bkv].max(axis=1)
        m = blk_max.max(axis=0)                       # global row max

        # ---- EXP: flat block-major exp_input through the EXP crossbars ----
        e = np.arange(S * S)
        b_of = e // BLKEL
        r_of = (e % BLKEL) // Bkv
        c_of = e % Bkv
        exp_input = np.maximum(
            s32[r_of, b_of * Bkv + c_of] - blk_max[b_of, r_of], -128
        ).astype(np.int8)
        eb_flat, exp_passes = _convert_packed_stream(
            exp_input, nref.MODE_EXP, self.R, self.C, self.BUF, self.P,
            self._exp_crossbars)
        blk_Sp = (eb_flat.view(np.uint8).astype(np.int64)
                  .reshape(B, S, Bkv).sum(axis=2))

        # ---- factor ACAM_EXP(m_b - m) through the dedicated factor bank ----
        fac_in = np.clip(blk_max.reshape(-1) - np.tile(m, B), -128, 127
                         ).astype(np.int8)
        factor_flat, fac_passes = _convert_packed_stream(
            fac_in, nref.MODE_EXP, self.R, self.C, self.BUF, self.P,
            self._fac_crossbars)
        factor = factor_flat.view(np.uint8).astype(np.int64).reshape(B, S)

        # ---- deferred-alpha combine ----
        L = (factor * blk_Sp).sum(axis=0)
        lq = np.minimum(L >> log2_S, 127).astype(np.int32)

        # ---- LOG ACAM_LOG(lq) through the LOG crossbars ----
        ls_flat, log_passes = _convert_packed_stream(
            lq.astype(np.int8), nref.MODE_LOG, self.R, self.C, self.BUF,
            self.P, self._log_crossbars)
        ls = ls_flat.astype(np.int32)

        out8 = np.clip(s32 - m[:, None] - ls[:, None],
                       -128, 127).astype(np.int8)

        passes_exp = int(np.sum(exp_passes))
        passes_factor = int(np.sum(fac_passes))
        passes_log = int(np.sum(log_passes))

        # ---- timing: same windowing (`packed_windows`) as the conversions ---
        t = self._schedule(passes_exp, passes_factor, passes_log)
        plan = SoftmaxOnlinePassPlan(blocks=B, passes_exp=passes_exp,
                                     passes_log=passes_log,
                                     passes_factor=passes_factor)
        stages = (SoftmaxOnlineStages(
            blk_max=blk_max, blk_Sp=blk_Sp, factor=factor, L=L, lq=lq, ls=ls,
            softmax_out=out8) if collect_stages else None)
        return SoftmaxOnlineResult(
            softmax_out=out8, ready_cycles=t["ready_cycles"],
            serialize_cycles=0, passes_exp=passes_exp,
            passes_log=passes_log, passes_factor=passes_factor,
            cycle_model=self.shadow_cycles(plan), stages=stages,
            timeline=t["timeline"])

    # -- timing from the same windowing ------------------------------------
    def _schedule(self, passes_exp, passes_factor, passes_log) -> dict:
        """Event schedule over the same `packed_windows` as the values.

        block readiness (max fold) -> EXP windows -> factor windows -> combine
        -> LOG windows -> results ready.
        """
        S, I, Bkv, B = self.S, self.I, self.Bkv, self.B
        BLKEL = S * Bkv
        stream_len = -(-BLKEL // self.stream_bytes)
        block_ready = np.array([(b + 1) * stream_len + self.clb_tree_latency
                                for b in range(B)], dtype=np.int64)
        last_block_ready = int(block_ready.max()) if B else 0

        # EXP windows: same list the EXP conversion used
        n_win, wlo, wlen, per_xbar = packed_windows(S * S, I, self.n_exp)
        exp_ready = np.zeros(n_win, dtype=np.int64)
        for j in range(n_win):
            lo, hi = int(wlo[j]), int(wlo[j] + wlen[j])
            exp_ready[j] = int(
                block_ready[lo // BLKEL:(hi - 1) // BLKEL + 1].max())
        exp_events = [
            schedule_pass_sequence([int(exp_ready[j]) for j in per_xbar[c]],
                                   self.R, self.C, self.BUF, self.P)
            for c in range(self.n_exp)]
        block_drain_end = np.zeros(B, dtype=np.int64)
        for c in range(self.n_exp):
            for j, ev in zip(per_xbar[c], exp_events[c]):
                lo, hi = int(wlo[j]), int(wlo[j] + wlen[j])
                for b in range(lo // BLKEL, (hi - 1) // BLKEL + 1):
                    block_drain_end[b] = max(block_drain_end[b],
                                             int(ev["drain_end"]))
        exp_done = int(block_drain_end.max()) if B else 0

        # factor windows: gated by `m_ready` (m_q commits the cycle after
        # all blocks are max-folded), then the last drain word.
        m_ready = last_block_ready + 1
        n_fw, _, _, fac_per = packed_windows(B * S, I, self.n_fac)
        fac_last = []
        for c in range(self.n_fac):
            ev = schedule_pass_sequence([m_ready] * len(fac_per[c]),
                                        self.R, self.C, self.BUF, self.P)
            fac_last.append(int(ev[-1]["drain_end"]) if ev else m_ready)
        fac_done = max(fac_last) if fac_last else m_ready

        # combine_start: both passes' last drain completes, then the combine.
        combine_start = max(exp_done, fac_done)
        combine_done = combine_start + self.combine_latency

        # LOG windows: same list the LOG conversion used
        n_lw, _, _, log_per = packed_windows(S, I, self.n_log)
        log_events = [schedule_pass_sequence([combine_done] * len(log_per[c]),
                                             self.R, self.C, self.BUF, self.P)
                      for c in range(self.n_log)]
        log_done = (max(int(ev[-1]["drain_end"]) for ev in log_events if ev)
                    if n_lw else combine_done)

        # RTL structural prologue (stream alignment + feed/log drain end)
        ready = log_done + self.clb_clamp_latency + self.prologue
        return {"ready_cycles": ready, "load_cycles": B * stream_len,
                "timeline": [("stream", None, S * S, 0, last_block_ready),
                             ("exp", None, int(n_win),
                              int(exp_ready.min()) if n_win else 0, exp_done),
                             ("factor", None, int(n_fw), last_block_ready,
                              fac_done),
                             ("combine", None, B, combine_start, combine_done),
                             ("log", None, int(n_lw), combine_done, log_done)]}

    def shadow_cycles(self, plan: SoftmaxOnlinePassPlan
                      ) -> SoftmaxOnlineCycleModel:
        return softmax_online_cycle_model(
            self.S, self.R, self.C, self.BUF, self.P, self.Bkv,
            self.n_exp, self.n_log, self.clb_pipe, self.stream_bytes, plan=plan)

    # -- stimulus dump (GATE 1 + GATE 2 expected files) --------------------
    def dump_case(self, case_dir: Path, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.int8)
        assert scores.shape == (self.S, self.S)
        result = self.run(scores, collect_stages=True)
        st = result.stages
        expected = oref.softmax_online_model(scores.astype(np.int32), self.Bkv)

        # GATE 1: certify vs the ACAM model oracle (values) + pass counts.
        for key, got in (("blk_max", st.blk_max), ("blk_Sp", st.blk_Sp),
                         ("factor", st.factor), ("L", st.L), ("lq", st.lq),
                         ("ls", st.ls), ("out8", st.softmax_out)):
            want = expected[key]
            if got.shape != want.shape or not np.array_equal(got, want):
                raise RuntimeError(f"GATE 1: sim/oracle '{key}' mismatch")
        # independent check: the schedule's EXP/LOG counts must equal the
        # packed-window formula (the oracle's; the schedule is normative).
        pc = oref.softmax_online_pass_counts(
            self.S, self.R, self.C, self.Bkv, self.n_exp, self.n_log)
        if (result.passes_exp, result.passes_log) != \
                (pc["passes_exp"], pc["passes_log"]):
            raise RuntimeError(
                f"GATE 1: schedule passes {(result.passes_exp, result.passes_log)}"
                f" != packed formula {(pc['passes_exp'], pc['passes_log'])}")

        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        stream_len = -(-(self.S * self.Bkv) // self.stream_bytes)
        t_load = self.B * stream_len
        parts = []
        for b in range(self.B):
            blk = scores[:, b * self.Bkv:(b + 1) * self.Bkv].reshape(-1)
            pad = stream_len * self.stream_bytes - blk.size
            parts.append(np.concatenate([blk, np.zeros(pad, dtype=np.int8)]))
        block_major = np.concatenate(parts).astype(np.int8)
        with open(case_dir / "scores.mem", "w") as f:
            for word in nref.pack_act_stream(block_major):
                f.write(f"{word:010x}\n")

        def w8(name, vals):
            with open(case_dir / name, "w") as f:
                for v in np.asarray(vals).reshape(-1):
                    f.write(f"{int(v) & 0xFF:02x}\n")

        def w32(name, vals):
            with open(case_dir / name, "w") as f:
                for v in np.asarray(vals).reshape(-1):
                    f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

        w8("expected_blkmax.mem", st.blk_max)
        w8("expected_lq.mem", st.lq)
        w8("expected_ls.mem", st.ls)
        w8("expected_out.mem", st.softmax_out)
        w32("expected_blksp.mem", st.blk_Sp)
        w32("expected_factor.mem", st.factor)
        w32("expected_L.mem", st.L)
        case = {
            "S": self.S, "R": self.R, "C": self.C, "BUF": self.BUF,
            "P": self.P, "Bkv": self.Bkv, "blocks": self.B,
            "n_exp": self.n_exp, "n_log": self.n_log,
            "stream_bytes": self.stream_bytes, "clb_pipe": self.clb_pipe,
            "passes_exp": int(result.passes_exp),
            "passes_log": int(result.passes_log),
            "passes_factor": int(result.passes_factor),
            "ready_cycles": int(result.ready_cycles),
            "serialize_cycles": 0, "drain_cycles": 0,
            "load_cycles": int(t_load),
            "e2e_cycles": int(result.ready_cycles),
            "compute_cycles": int(result.ready_cycles),
            "used_cycles": int(result.ready_cycles),
            "load_words": int(len(nref.pack_act_stream(block_major))),
            "weight_cycles": int(self.R * self.C),
        }
        with open(case_dir / "case.json", "w") as f:
            json.dump(case, f, indent=2)
            f.write("\n")


# (used_cycles, compute_cycles) per (S, Bkv, n_exp, n_log); schedule-determined.
_FROZEN_MEASURED_PINS = {
    (32, 16, 1, 1): (525, 525),
    (128, 16, 1, 1): (4436, 4436),
    (128, 16, 4, 1): (3586, 3586),
    (128, 128, 1, 1): (7300, 7300),
}


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/softmax_online_sim.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(7)

    # ── bit-exact vs oracle model; schedule pass counts == packed formula ──
    for S in (32, 128):
        s = rng.integers(-64, 64, size=(S, S), dtype=np.int8)
        for Bkv, n_exp in ((S, 1), (16, 1), (16, 4), (4, 2)):
            if S % Bkv:
                continue
            unit = NldpeSoftmaxOnline(S=S, Bkv=Bkv, n_exp=n_exp, n_log=1)
            res = unit.run(s, collect_stages=True)
            exp = oref.softmax_online_model(s.astype(np.int32), Bkv)
            for key, got in (("blk_max", res.stages.blk_max),
                             ("blk_Sp", res.stages.blk_Sp),
                             ("factor", res.stages.factor), ("L", res.stages.L),
                             ("lq", res.stages.lq), ("ls", res.stages.ls),
                             ("out8", res.softmax_out)):
                assert np.array_equal(got, exp[key]), (S, Bkv, n_exp, key)
            pc = oref.softmax_online_pass_counts(S, 256, 256, Bkv, n_exp, 1)
            assert (res.passes_exp, res.passes_log) == \
                (pc["passes_exp"], pc["passes_log"]), (S, Bkv, n_exp)
            assert res.serialize_cycles == 0   # output not counted
            pin = _FROZEN_MEASURED_PINS.get((S, Bkv, n_exp, 1))
            if pin is not None:
                assert (res.ready_cycles, res.compute_cycles) == pin, \
                    (S, Bkv, n_exp, res.ready_cycles, pin)
            print(f"  [sim] S={S} Bkv={Bkv} n_exp={n_exp}: bit-exact vs oracle; "
                  f"passes e/f/l={res.passes_exp}/{res.passes_factor}/"
                  f"{res.passes_log}; ready={res.ready_cycles}: OK")

    # ── corridor + monotonicity ────────────────────────────────────────────
    s = rng.integers(-64, 64, size=(128, 128), dtype=np.int8)
    r1 = NldpeSoftmaxOnline(S=128, Bkv=16, n_exp=1).run(s)
    r4 = NldpeSoftmaxOnline(S=128, Bkv=16, n_exp=4).run(s)
    assert r4.ready_cycles <= r1.ready_cycles
    assert r1.ready_cycles >= r1.cycle_model.total, \
        (r1.ready_cycles, r1.cycle_model.total)
    print(f"  [corridor] n_exp=1 -> {r1.ready_cycles} >= envelope "
          f"{r1.cycle_model.total}; n_exp=4 -> {r4.ready_cycles}: OK")

    # ── Bkv=S reduces to the full-row machine bit-exactly ─────────────────
    from softmax_sim import NldpeSoftmax
    full = NldpeSoftmax(S=128, n_exp=1, n_log=1).run(s)
    one = NldpeSoftmaxOnline(S=128, Bkv=128, n_exp=1).run(s)
    assert np.array_equal(one.softmax_out, full.softmax_out), "Bkv=S != full-row"
    outs = {Bkv: NldpeSoftmaxOnline(S=128, Bkv=Bkv).run(s).softmax_out
            for Bkv in (1, 4, 16, 128)}
    ndiff = {b: int((o != outs[128]).sum()) for b, o in outs.items()}
    print(f"  [Bkv] S=128: Bkv=S == full-row bit-exact; differing elements vs "
          f"Bkv=S: {ndiff}: OK")

    print("softmax_online_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
