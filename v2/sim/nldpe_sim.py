#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the sibling oracle module importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import nldpe_ref as ref  # noqa: E402

# Mode codes — mirror nldpe_ref (keep the spec's one source of truth).
MODE_REGULAR, MODE_ACTIVATION, MODE_EXP, MODE_LOG = 0, 1, 2, 3


@dataclass
class CycleModel:
    """§5.3 closed-form cycle contract (analytical shadow of the schedule)."""

    load_cyc: int       # LOAD_CYC   = ceil(R*8/BUF)
    compute_cyc: int    # COMPUTE_CYC = P + 2
    output_cyc: int     # OUTPUT_CYC  = ceil(C*8/BUF)
    # WR_CYC     = R*C   (one int8 word/cycle, P23; one-time)
    wr_cyc: int
    t_fill: int         # LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
    t_steady: int       # max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC + 1)
    total: int          # T_fill + (M-1) * T_steady


@dataclass
class PassTimeline:
    """Stage events for one pass (absolute cycles; see module docstring)."""

    m: int
    load_start: int        # first ACT burst cycle
    load_end: int          # inclusive
    compute_start: int     # first crossbar fire
    msb_fire: int          # last crossbar fire (slice P-1)
    shift_acc_done: int    # MSB slice accumulated into y
    acam_done: int         # ACAM fires: out8 committed to output buffer
    drain_start: int       # first output byte on the port
    drain_end: int         # inclusive


@dataclass
class SimResult:
    """Everything the cross-check harness consumes (Stage 1.5)."""

    # uint8 [M, C]  — drained output stream payload, one byte per column
    out_stream: np.ndarray
    # int32 [M, C]  — pre-ACAM crossbar output (T3, hierarchical compare P26)
    y: np.ndarray
    used_cycles: int               # measured Total(M): last drain_end + 1
    weight_cycles: int             # WR_CYC, reported separately (§5.3)
    timeline: list[PassTimeline] = field(default_factory=list)


class NldpeDpe:

    def __init__(self, R: int = 256, C: int = 512, BUF: int = 40, P: int = 8) -> None:
        # §1 parameter table. Cross-check config is 256x256; reference 256x512.
        self.R, self.C, self.BUF, self.P = R, C, BUF, P
        # §3 storage: bit-sliced input banks, int8 output buffer, int32
        # accumulators. Weights undefined until program_weights (A10/A12).
        self.banks = np.zeros((P, R), dtype=np.uint8)
        self.out_buf = np.zeros(C, dtype=np.int8)
        self.acc = np.zeros(C, dtype=np.int32)
        self.W = None
        self._programmed = False

    # -- weight programming (§4.2) -----------------------------------------
    def program_weights(self, W: np.ndarray) -> int:
        W = np.asarray(W)
        assert W.shape == (self.R, self.C), f"W must be [{self.R}, {self.C}]"
        self.W = W.astype(np.int8, copy=True)
        self._programmed = True
        return self.R * self.C

    # -- datapath (values) --------------------------------------------------
    def _corner_turn(self, x_m: np.ndarray) -> None:
        ub = np.ascontiguousarray(x_m).view(np.uint8)          # (R,) bytes
        bits = np.arange(self.P, dtype=np.uint8)[:, None]      # (P, 1)
        self.banks[:] = (ub[None, :] >> bits) & 1

    def _trunc8(self, z: np.ndarray) -> np.ndarray:
        t = np.asarray(z, dtype=np.int64)
        t = np.clip(t, -(2**31), 2**31 - 1)
        return (t & 0xFF).astype(np.uint8).view(np.int8)

    def acam_fire(self, y: np.ndarray, mode: int) -> np.ndarray:
        """
            mode 0 REGULAR    : f(v) = v
            mode 1 ACTIVATION : f(v) = relu(v) = v if v > 0 else 0
            mode 2 EXP        : f(v) = 1 + v + floor(v²/2)   (wide intermediate)
            mode 3 LOG        : f(v) = v - 1
        """
        if mode == MODE_REGULAR:
            return self._trunc8(y)
        if mode == MODE_ACTIVATION:
            return self._trunc8(np.maximum(y, 0))
        if mode == MODE_EXP:
            y64 = y.astype(np.int64)
            return self._trunc8(1 + y64 + (y64 * y64) // 2)
        if mode == MODE_LOG:
            return self._trunc8(y - 1)
        if mode not in (MODE_REGULAR, MODE_ACTIVATION, MODE_EXP, MODE_LOG):
            raise ValueError(f"invalid ACAM mode: {mode}")

    def _fire_pass(self, mode: int) -> tuple[np.ndarray, np.ndarray]:

        assert self._programmed and self.W is not None, f"current DPE is not programmed has none Weights"
        Wi = self.W.astype(np.int32)
        y = np.zeros(self.C, dtype=np.int32)
        for b in range(self.P):
            s_b = self.banks[b].astype(np.int32) @ Wi      # exact [C]
            if b < self.P - 1:
                y += s_b << b
            else:
                y -= s_b << b                              # P2
        return y, self.acam_fire(y, mode)

    # -- timing engine (cycles) ---------------------------------------------

    def run_workload(self, X: np.ndarray, mode: int) -> SimResult:
        """
        Scheduling per pass m (see module docstring for the event grammar):
          load_start     = msb_fire_{m-1} + 1          (P1/A9), first pass = 0
          compute_start  = max(load_start + LOAD_CYC, acc_free_cycle)
          msb_fire       = compute_start + P - 1
          shift_acc_done = msb_fire + 1
          acam_done     = max(shift_acc_done + 1, out_free_cycle)  (P11)
          drain          = [acam_done + 1, acam_done + OUTPUT_CYC]

        `mode` is the workload's ACAM mode (P27: programmed with the weights,
        constant across all passes — one mode per call, never per pass).

        Accumulator gate (§5.2, v2.0.1): the accumulator is single and is
        freed by its ACAM write, not by MSB shift&acc — `acc_free_cycle =
        acam_done + 1`. (The §5.3 totals are unchanged; per-pass event times
        for output-bound configs are optimistic vs a single-acc hardware.)
        """
        assert self._programmed, "weights must be programmed before workload started"
        M = X.shape[0]
        assert mode in (MODE_REGULAR, MODE_ACTIVATION, MODE_EXP, MODE_LOG), \
            f"invalid ACAM mode {mode}"
        load_cycles = -(-self.R * 8 // self.BUF)      # §4.3: ceil(R*8/BUF)
        output_cycles = -(-self.C * 8 // self.BUF)    # §4.5: ceil(C*8/BUF)

        Y = np.empty((M, self.C), dtype=np.int32)
        OUT = np.empty((M, self.C), dtype=np.uint8)
        timeline: list[PassTimeline] = []

        load_start, acc_free_cycle, out_free_cycle = 0, 0, 0
        for m in range(M):
            # buffer full + accumulator free
            compute_start = max(load_start + load_cycles, acc_free_cycle)
            msb_fire = compute_start + self.P - 1   # last crossbar fire
            shift_acc_done = msb_fire + 1           # MSB shift&acc completes
            acam_done = max(shift_acc_done + 1, out_free_cycle)   # P11
            drain_start = acam_done + 1
            drain_end = acam_done + output_cycles  # inclusive

            self._corner_turn(X[m])
            y, acam_out = self._fire_pass(mode)
            Y[m] = y
            OUT[m] = acam_out.view(np.uint8)

            timeline.append(PassTimeline(
                m=m,
                load_start=load_start,
                load_end=load_start + load_cycles - 1,
                compute_start=compute_start,
                msb_fire=msb_fire,
                shift_acc_done=shift_acc_done,
                acam_done=acam_done,
                drain_start=drain_start,
                drain_end=drain_end,
            ))

            load_start = msb_fire + 1               # P1/A9: MSB_SA_Ready rises
            acc_free_cycle = acam_done + 1          # §5.2 v2.0.1: freed by ACAM
            out_free_cycle = drain_end + 1

        return SimResult(
            out_stream=OUT,
            y=Y,
            used_cycles=timeline[-1].drain_end + 1,
            weight_cycles=self.R * self.C,          # WR_CYC, one-time (§5.3)
            timeline=timeline,
        )

    # -- stimulus dump (§9, consumed by the Stage 1.5 harness) --------------
    def dump_case(self, case_dir: Path, X: np.ndarray, mode: int) -> None:
        """Write stimulus + expected files for the RTL cross-check.

        `mode` is the workload's ACAM mode, programmed with the weights and
        constant across all passes (P27).

        Runs `run_workload(X, mode)` and writes (format contract):
          weights.mem    — one int8 weight word per cycle on data_in[7:0],
                           row-major row-outer (P23); 10-hex-digit word/line
          act.mem        — all M bursts concatenated (10-hex-digit word/line)
          expected_y.npz — int32 [M, C]  (T3 crossbar output, hierarchical P26)
          expected_out.mem  — one line per pass: C column-order bytes,
                           2 hex digits/byte (§4.5)
          case.json      — {R, C, BUF, P, M, mode, used_cycles, weight_cycles}

        The harness falls back to in-memory compare if files are absent.
        """
        res = self.run_workload(X, mode)
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        with open(case_dir / "weights.mem", "w") as f:
            for w in ref.pack_weight_stream(self.W):
                f.write(f"{w:010x}\n")

        with open(case_dir / "act.mem", "w") as f:
            for m in range(X.shape[0]):
                for w in ref.pack_act_stream(X[m]):
                    f.write(f"{w:010x}\n")

        np.savez(case_dir / "expected_y.npz", y=res.y)

        with open(case_dir / "expected_out.mem", "w") as f:
            for m in range(res.out_stream.shape[0]):
                f.write("".join(f"{b:02x}" for b in res.out_stream[m]) + "\n")

        case = {
            "R": self.R,
            "C": self.C,
            "BUF": self.BUF,
            "P": self.P,
            "M": int(X.shape[0]),
            "mode": int(mode),
            "used_cycles": int(res.used_cycles),
            "weight_cycles": int(res.weight_cycles),
        }
        with open(case_dir / "case.json", "w") as f:
            json.dump(case, f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Analytical shadow (§5.3) — the closed form the measured schedule must
# reproduce for the canonical schedule.
# ---------------------------------------------------------------------------
def cycle_model(M: int, R: int, C: int, P: int = 8, BUF: int = 40) -> CycleModel:
    """§5.3 closed-form cycle contract.

      LOAD_CYC    = ceil(R*8/BUF)
      COMPUTE_CYC = P + 2
      OUTPUT_CYC  = ceil(C*8/BUF)
      WR_CYC      = R*C
      T_fill      = LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
      T_steady    = max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC + 1)
      total       = T_fill + (M-1) * T_steady
    """
    LOAD_CYCLE = np.ceil(R * P / BUF)
    COMPUTE_CYCLE = P + 2
    OUTPUT_CYCLE = np.ceil(C * P / BUF)
    WR_CYCLE = R * C
    T_fill = LOAD_CYCLE + COMPUTE_CYCLE + OUTPUT_CYCLE
    T_steady = max(LOAD_CYCLE + P, COMPUTE_CYCLE, OUTPUT_CYCLE + 1)
    total = T_fill + (M - 1) * T_steady

    return CycleModel(
        load_cyc=LOAD_CYCLE,
        compute_cyc=COMPUTE_CYCLE,
        output_cyc=OUTPUT_CYCLE,
        wr_cyc=WR_CYCLE,
        t_fill=T_fill,
        t_steady=T_steady,
        total=total
    )


# ---------------------------------------------------------------------------
# Self-test — run:  python3 nldpe_sim.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(1)

    # §5.3 table values (frozen) — cycle_model must reproduce them exactly.
    cm = cycle_model(M=1, R=256, C=256)
    assert (cm.load_cyc, cm.compute_cyc, cm.output_cyc) == (52, 10, 52)
    assert (cm.t_fill, cm.t_steady) == (114, 60)
    cm512 = cycle_model(M=1, R=256, C=512)
    assert (cm512.t_fill, cm512.t_steady) == (165, 104)

    dpe = NldpeDpe(R=256, C=256)
    W = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
    wc = dpe.program_weights(W)
    assert wc == 65536, "WR_CYC mismatch"

    for M in (1, 2, 4, 8):
        X = rng.integers(-128, 128, size=(M, 256), dtype=np.int8)
        res = dpe.run_workload(X, MODE_REGULAR)

        # Values ≡ reference (§9): full int32 y and the byte stream (P26).
        assert res.y.dtype == np.int32
        assert (res.y == ref.compute_y(W, X)).all(), f"F2 mismatch M={M}"
        assert (res.out_stream == ref.acam_transform(res.y, MODE_REGULAR).view(np.uint8)).all()

        # Cycles ≡ closed form (§5.3): the measured schedule IS the theorem.
        assert res.used_cycles == cycle_model(M, 256, 256).total, (
            f"measured {res.used_cycles} != formula {cycle_model(M, 256, 256).total}"
        )
        # COMPUTE_CYC must emerge from the schedule, not be asserted (§10 P10).
        t = res.timeline[0]
        assert (t.acam_done - t.compute_start + 1) == 10

    # -- Weight-static projection (VMM) at an attention-like shape ----------
    # Stationary W with M activation rows IS the projection workload: one
    # crossbar fire per row (M passes). Same dual check as above, now at a
    # realistic shape (d_model=128 -> M=128) and with a measured T_steady step.
    M_proj = 128
    Xp = rng.integers(-128, 128, size=(M_proj, 256), dtype=np.int8)
    res_p = dpe.run_workload(Xp, MODE_REGULAR)
    assert (res_p.y == ref.compute_y(W, Xp)).all(), "projection F2 mismatch"
    assert (res_p.out_stream
            == ref.acam_transform(res_p.y, MODE_REGULAR).view(np.uint8)).all(), \
        "projection byte stream mismatch"
    cm_p = cycle_model(M_proj, 256, 256)
    assert res_p.used_cycles == cm_p.total, (
        f"projection cycles {res_p.used_cycles} != formula {cm_p.total}"
    )
    # Measured steady-state step: one extra pass costs exactly T_steady.
    step = (dpe.run_workload(Xp[:2], MODE_REGULAR).used_cycles
            - dpe.run_workload(Xp[:1], MODE_REGULAR).used_cycles)
    assert step == cm_p.t_steady, (
        f"projection step {step} != T_steady {cm_p.t_steady}"
    )

    # Identity weights (§8 I7): out[c] == x[c] in REGULAR mode.
    dpe_i = NldpeDpe(R=256, C=256)
    dpe_i.program_weights(np.eye(256, dtype=np.int8))
    X = rng.integers(-128, 128, size=(4, 256), dtype=np.int8)
    res = dpe_i.run_workload(X, MODE_REGULAR)
    assert (res.y == X.astype(np.int32)).all(), "identity y failed"
    assert (res.out_stream == X.view(np.uint8)).all(), "identity stream failed"

    # -- Mode axis (P27: one constant mode per workload) --------------------
    # Same weights + activations, each mode as its own workload: y is
    # mode-independent, the byte stream uses the programmed mode, and the
    # cycle count is identical for all four modes.
    dpe_m = NldpeDpe(R=256, C=256)
    dpe_m.program_weights(W)
    Xm = rng.integers(-128, 128, size=(2, 256), dtype=np.int8)
    y_ref = ref.compute_y(W, Xm)
    cm_m = cycle_model(2, 256, 256)
    for mode in (MODE_REGULAR, MODE_ACTIVATION, MODE_EXP, MODE_LOG):
        res_m = dpe_m.run_workload(Xm, mode)
        assert (res_m.y == y_ref).all(), f"mode {mode}: F2 mismatch"
        exp = ref.acam_transform(y_ref, mode).view(np.uint8)
        assert (res_m.out_stream == exp).all(), f"mode {mode}: stream mismatch"
        assert res_m.used_cycles == cm_m.total, f"mode {mode}: cycle mismatch"

    # -- Output-bound config (C=512): §5.3 totals survive the single-acc ACAM
    #    gate (v2.0.1). T(4) = 165 + 3*104 = 477.
    dpe_512 = NldpeDpe(R=256, C=512)
    dpe_512.program_weights(np.eye(256, 512, dtype=np.int8))
    X512 = rng.integers(-128, 128, size=(4, 256), dtype=np.int8)
    res_512 = dpe_512.run_workload(X512, MODE_REGULAR)
    assert (res_512.y[:, :256] == X512.astype(np.int32)).all(), "512 identity y failed"
    assert (res_512.y[:, 256:] == 0).all(), "512 identity padding failed"
    assert res_512.used_cycles == cycle_model(4, 256, 512).total == 477

    print("nldpe_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
