#!/usr/bin/env python3
"""nldpe_sim.py — OOP golden model for the NL-DPE primitive (v2 clean-room).

Verification chain (two gates, trust flows downward only; the spec decides
all disagreements — `v2/sim/spec/dpe_nldpe.md`: spec > oracle > legacy):

    spec v2.0.1                 human charter; the only arbiter
       │  transcription
       ▼
    v2/sim/oracle/nldpe_ref.py      exact integer reference; no quantization policy
       │  GATE 1 (per case): bit-exact int32 y + output bytes + §5.3 cycles,
       │  enforced inside `dump_case` BEFORE any expected file is written —
       │  a case that disagrees with the oracle is never dumped
       ▼
    this file                   golden model; owns the quantize/trunc8
       │                        boundary; dumps expected bits for the TB
       │  GATE 2: bit-exact values (RTL vs dumped expected_y/expected_out,
       │  dual compare P26) + cycles vs §5.3 with one implementation
       │  constant Δ_impl (RTL is never required to match sim cycles exactly)
       ▼
    v2/sim/rtl/dpe_nldpe.v          DUT; integer-only datapath (no fp in RTL)

Later stages (softmax, DIMM) keep this policy: transcendentals/high precision
stay in the oracle, the sim carries the quantization boundary, and the RTL
implements only the exact integer/fixed-point forms.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the sibling oracle module importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "oracle"))
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
    rows_loaded: int = 0    # rows streamed per pass (P29 pass extent; R = certified full load)
    cols_drained: int = 0   # columns drained per pass (P29; C = certified full drain)


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

    def __init__(self, R: int = 256, C: int = 512, BUF: int = 40, P: int = 8,
                 rows_used: int | None = None, cols_used: int | None = None) -> None:
        # §1 parameter table. Cross-check config is 256x256; reference 256x512.
        self.R, self.C, self.BUF, self.P = R, C, BUF, P
        # P29 pass extent (spec v2.0.3): rows streamed in / columns drained per pass.
        # Default = the full array (A4); mirrors the RTL parameters ROWS_USED / COLS_USED.
        self.rows = R if rows_used is None else int(rows_used)
        self.cols = C if cols_used is None else int(cols_used)
        if not (1 <= self.rows <= R and 1 <= self.cols <= C):
            raise ValueError(f"pass extent rows={self.rows}, cols={self.cols} outside 1..{R} / 1..{C}")
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

    def _extent_input(self, x_m: np.ndarray) -> np.ndarray:
        """P29: rows >= rows_used are not loaded and read as zero (F5 zero padding)."""
        x = np.array(x_m, dtype=np.int8, copy=True)
        x[self.rows:] = 0
        return x

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
        load_cycles = -(-self.rows * 8 // self.BUF)   # §4.3: ceil(rows*8/BUF) (P29; rows = R by default)
        output_cycles = -(-self.cols * 8 // self.BUF) # §4.5: ceil(cols*8/BUF) (P29; cols = C by default)

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

            self._corner_turn(self._extent_input(X[m]))
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

        GATE 1 (verification chain): a case is dumped only after the sim
        agrees with the NumPy oracle bit-exactly on this exact stimulus —
        int32 `y`, the ACAM output bytes, and the §5.3 cycle total. A
        mismatch raises before any file is written (spec §9: the oracle is
        the numerical ground truth; the spec arbitrates disagreements).
        """
        res = self.run_workload(X, mode)
        X_eff = np.array(X, dtype=np.int8, copy=True)
        X_eff[:, self.rows:] = 0                      # P29: rows not loaded read as zero

        # -- GATE 1: per-case oracle certification -------------------------
        y_ref = ref.compute_y(self.W, X_eff)
        if res.y.shape != y_ref.shape or res.y.dtype != y_ref.dtype:
            raise RuntimeError(
                f"GATE 1: sim/oracle shape or dtype mismatch (sim "
                f"{res.y.shape}/{res.y.dtype}, oracle "
                f"{y_ref.shape}/{y_ref.dtype}) — case not dumped")
        if not (res.y == y_ref).all():
            raise RuntimeError(
                "GATE 1: sim/oracle F2 (int32 y) mismatch — case not dumped")
        out_ref = ref.acam_transform(y_ref, mode).view(np.uint8)
        if not (res.out_stream == out_ref).all():
            raise RuntimeError(
                "GATE 1: sim/oracle ACAM byte-stream mismatch — case not dumped")
        cm = cycle_model(X.shape[0], self.R, self.C, self.P, self.BUF,
                         rows=self.rows, cols=self.cols)
        if res.used_cycles != cm.total:
            raise RuntimeError(
                f"GATE 1: sim schedule {res.used_cycles} != §5.3 total "
                f"{int(cm.total)} — case not dumped")

        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        with open(case_dir / "weights.mem", "w") as f:
            for w in ref.pack_weight_stream(self.W):
                f.write(f"{w:010x}\n")

        with open(case_dir / "act.mem", "w") as f:
            for m in range(X.shape[0]):
                for w in ref.pack_act_stream(X_eff[m][:self.rows]):   # P29: rows in use only
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
            "rows_used": int(self.rows),
            "cols_used": int(self.cols),
        }
        with open(case_dir / "case.json", "w") as f:
            json.dump(case, f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Analytical shadow (§5.3) — the closed form the measured schedule must
# reproduce for the canonical schedule.
# ---------------------------------------------------------------------------
def cycle_model(M: int, R: int, C: int, P: int = 8, BUF: int = 40, *,
                rows: int | None = None, cols: int | None = None) -> CycleModel:
    """§5.3 closed-form cycle contract.

      LOAD_CYC    = ceil(rows*8/BUF)      rows = R by default (certified full load)
      COMPUTE_CYC = P + 2
      OUTPUT_CYC  = ceil(cols*8/BUF)      cols = C by default (certified full drain)
      WR_CYC      = R*C                   (the whole array is programmed either way)
      T_fill      = LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
      T_steady    = max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC + 1)
      total       = T_fill + (M-1) * T_steady

    `rows` / `cols` (keyword-only) are the P29 pass extent: the rows streamed in and the
    columns drained per pass when only part of the R x C array is in use (1 <= rows <= R,
    1 <= cols <= C). P29 (dpe_nldpe.md v2.0.3) is implemented in the primitive RTL
    (parameters ROWS_USED / COLS_USED) and in `NldpeDpe(rows_used=, cols_used=)`; primitive
    GATE 2 holds in both extents (2026-10-08).
    """
    rows = R if rows is None else int(rows)
    cols = C if cols is None else int(cols)
    if not (1 <= rows <= R and 1 <= cols <= C):
        raise ValueError(f"pass extent rows={rows}, cols={cols} outside 1..R={R}, 1..C={C}")
    LOAD_CYCLE = np.ceil(rows * P / BUF)
    COMPUTE_CYCLE = P + 2
    OUTPUT_CYCLE = np.ceil(cols * P / BUF)
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
        total=total,
        rows_loaded=rows,
        cols_drained=cols,
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

    _test_pass_extent()
    _test_behaviour_extent(rng)
    print("nldpe_sim self-test: ALL PASS")


def _test_behaviour_extent(rng) -> None:
    """P29 behaviour model: measured cycles == closed form; values == oracle on the zero-padded input;
    weights in unused rows (random, non-zero) never matter; default extent unchanged."""
    for R, C, rows, cols in [(8, 8, 5, 3), (40, 40, 1, 1), (40, 40, 17, 23), (128, 64, 64, 64),
                             (256, 256, 128, 64), (256, 512, 256, 129), (64, 256, 64, 64)]:
        W = rng.integers(-128, 128, size=(R, C), dtype=np.int8)
        for M in (1, 3):
            X = rng.integers(-128, 128, size=(M, R), dtype=np.int8)
            d = NldpeDpe(R=R, C=C, rows_used=rows, cols_used=cols)
            d.program_weights(W)
            for mode in (MODE_REGULAR, MODE_EXP, MODE_LOG):
                res = d.run_workload(X, mode)
                Xe = X.copy(); Xe[:, rows:] = 0
                y_ref = ref.compute_y(W, Xe)
                assert (res.y == y_ref).all(), f"{R}x{C} r{rows} c{cols}: y != oracle on padded input"
                assert res.used_cycles == cycle_model(M, R, C, rows=rows, cols=cols).total
                # garbage in the unused input rows must not change anything
                Xg = X.copy(); Xg[:, rows:] = rng.integers(-128, 128, size=(M, R - rows), dtype=np.int8)
                assert (d.run_workload(Xg, mode).y == res.y).all(), "unused rows leaked into y"
            full = NldpeDpe(R=R, C=C); full.program_weights(W)
            assert full.run_workload(X, MODE_REGULAR).used_cycles == cycle_model(M, R, C).total
    for bad in ({"rows_used": 0}, {"rows_used": 9}, {"cols_used": 0}, {"cols_used": 9}):
        try:
            NldpeDpe(R=8, C=8, **bad)
        except ValueError:
            continue
        raise AssertionError(f"extent {bad} must be rejected")


def _test_pass_extent() -> None:
    """P29 pass extent: default == certified full extent; partial extents time like a smaller array."""
    shapes = [(8, 8), (40, 40), (128, 64), (256, 256), (256, 512), (1024, 64), (64, 1024)]
    for R, C in shapes:
        for M in (1, 2, 7):
            full = cycle_model(M, R, C)
            # default and explicit full extent are the certified contract, unchanged
            assert full == cycle_model(M, R, C, rows=R, cols=C)
            assert (full.rows_loaded, full.cols_drained) == (R, C)
            # identity pass on I x I: same timing as an I x I array, same R*C programming
            I = min(R, C)
            used = cycle_model(M, R, C, rows=I, cols=I)
            small = cycle_model(M, I, I)
            assert (used.t_fill, used.t_steady, used.total) == (small.t_fill, small.t_steady, small.total)
            assert used.wr_cyc == R * C and used.total <= full.total
            if R == C:
                assert used.total == full.total
    # monotone in the extent; a one-row pass still takes one load word
    prev = None
    for rows in (1, 5, 6, 64, 1024):
        t = cycle_model(3, 1024, 64, rows=rows).total
        assert prev is None or t >= prev, "load time must grow with rows"
        prev = t
    assert cycle_model(1, 1024, 64, rows=1, cols=1).load_cyc == 1
    # worked values used by engine_sizing.md: 1024x64 identity pass 213 -> 21 cycles
    assert cycle_model(1, 1024, 64).t_steady == 213
    assert cycle_model(1, 1024, 64, rows=64, cols=64).t_steady == 21
    assert cycle_model(1, 1024, 64, rows=128, cols=64).t_steady == 34   # 128x64 projection matrix
    for bad in ({"rows": 0}, {"rows": 1025}, {"cols": 0}, {"cols": 65}):
        try:
            cycle_model(1, 1024, 64, **bad)
        except ValueError:
            continue
        raise AssertionError(f"extent {bad} must be rejected")


if __name__ == "__main__":
    _self_test()
