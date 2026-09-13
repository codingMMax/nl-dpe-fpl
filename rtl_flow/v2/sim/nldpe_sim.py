#!/usr/bin/env python3
"""nldpe_sim.py — OOP behavior simulator for the NL-DPE primitive (v2 clean-room).

Role
----
Transaction-level (stage-granularity) behavior simulator transcribed from the
FROZEN spec `v2/spec/dpe_nldpe.md` (v1.0). Two engines:

  * Datapath (values) — follows the spec's actual dataflow:
    corner-turn (§4.3) -> bit-serial fires LSB->MSB with shift-acc and
    MSB subtract (§6 F2) -> ACAM mode transform (§6 F3) -> output bytes.
  * Timing engine (cycles) — an event scheduler over stage durations,
    enforcing the §5.2 independence rules as scheduling constraints. Cycle
    counts are MEASURED from the simulated schedule, then checked against
    the §5.3 closed form (`cycle_model`) — the formula becomes a verified
    theorem of the scheduling logic, not an axiom.

This is neither a bare value counter nor a cycle-stepped model: state and
scheduling live at stage/pass granularity (your "analytical behavior
simulator"). It is also the seed of the Stage-5 workload simulator — later
stages compose `NldpeDpe` instances (V×H arrays, wrapper pacing) on top of
the same class.

Verification quadrilateral (§9):
  sim values      ≡ oracle/nldpe_ref.py          (must, bit-exact)
  sim values      ≡ legacy oracle MAC numerics   (harness, Stage 1.5)
  sim used_cycles ≡ cycle_model(...) closed form (must, canonical schedule)
  v2 RTL          ≡ sim                          (Stage 1.4/1.5)

Conventions
-----------
* Cycle 0 = first cycle of the workload's first ACT burst (weight
  programming is reported separately as `weight_cycles`).
* Stage events per pass m (PassTimeline), all in absolute cycles:

      burst_m        : [L_m, L_m + LOAD_CYC)          ACT burst (§4.3)
      compute_start  : max(L_m + LOAD_CYC, acc_free)  buffer full + acc free (§5.2)
      msb_fire       : compute_start + P - 1          last crossbar fire
      acc_done       : msb_fire + 1                   MSB shift&acc completes
      acam           : max(acc_done + 1, out_free)    P11: strictly after
                                                        previous drain
      out            : [acam + 1, acam + OUTPUT_CYC]  drain (§4.5)
      acc_free       : acc_done + 1                   y captured into ACAM input
      out_free       : out_end + 1
      L_{m+1}        : msb_fire_m + 1                 P1/A9 refill permit
                                                       (MSB_SA_Ready rises)

* COMPUTE_CYC = acam - compute_start + 1 = P + 2 (must emerge — §5.1/§10 P10).
"""

from __future__ import annotations

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
    output_cyc: int     # OUTPUT_CYC = ceil(C*8/BUF)
    wr_cyc: int         # WR_CYC     = ceil(R*C*8/BUF)   (one-time)
    t_fill: int         # LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
    t_steady: int       # max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC + 1)
    total: int          # T_fill + (M-1) * T_steady


@dataclass
class PassTimeline:
    """Stage events for one pass (absolute cycles; see module docstring)."""

    m: int
    load_start: int
    load_end: int        # inclusive
    compute_start: int
    msb_fire: int
    acc_done: int
    acam: int
    out_start: int
    out_end: int         # inclusive


@dataclass
class SimResult:
    """Everything the cross-check harness consumes (Stage 1.5)."""

    out_bytes: np.ndarray          # uint8 [M, C]  — per-pass output, column order
    y_int: np.ndarray              # int32 [M, C]  — pre-ACAM accumulators
    used_cycles: int               # measured Total(M): last out_end + 1
    weight_cycles: int             # WR_CYC, reported separately (§5.3)
    timeline: list[PassTimeline] = field(default_factory=list)


class NldpeDpe:
    """One NL-DPE hard block (§1): crossbar + ACAM + input/output buffers.

    Storage model (§3): weight storage R×C int8; single bit-sliced input
    buffer (P banks × R bits); single output buffer C×8; C×32 accumulators.
    """

    def __init__(self, R: int = 256, C: int = 512, BUF: int = 40, P: int = 8) -> None:
        # §1 parameter table. Cross-check config is 256x256; reference 256x512.
        self.R, self.C, self.BUF, self.P = R, C, BUF, P
        # TODO(you): weight storage (§3), input banks (P × R bit slices),
        # output buffer (C × 8), pass bookkeeping. Weights undefined until
        # program_weights (A10/A12).
        raise NotImplementedError

    # -- weight programming (§4.2) -----------------------------------------
    def program_weights(self, W: np.ndarray) -> int:
        """Store int8 [R, C] weights; return WR_CYC = ceil(R*C*8/BUF).

        TODO(you): validate shape/dtype, store a copy. The byte-stream
        packing itself is `ref.pack_weight_stream` (stimulus side); the sim
        keeps the array form.
        """
        raise NotImplementedError

    # -- datapath (values) --------------------------------------------------
    def _corner_turn(self, x_m: np.ndarray) -> None:
        """§4.3 — park one activation vector into the input buffer.

        bank[b][j] = bit b of x_m[j], b = 0..P-1 (the fixed corner-turn).

        TODO(you): implement with (x.view(np.uint8) >> b) & 1 per bank.
        """
        raise NotImplementedError

    def _fire_pass(self, mode: int) -> tuple[np.ndarray, np.ndarray]:
        """§6 F2/F3 — bit-serial compute + ACAM for the parked vector.

        y = 0 (int32 [C])
        for b in 0..P-1:            # one fire per cycle, LSB -> MSB
            s_b = banks[b] @ W      # int32 [C]: sum_r bank_b[r] * W[r, c]
            y = (y << 1) + s_b      if b < P-1
            y = (y << 1) - s_b      if b == P-1   # MSB subtract (P2)
        out8 = acam_transform(y, mode)             # §6 F3, int8 [C]

        Returns (y_int, out8). TODO(you): implement. Keep the loop shape —
        it must mirror the hardware's per-slice datapath, not one matmul.
        """
        raise NotImplementedError

    # -- timing engine (cycles) ---------------------------------------------
    def run_workload(self, X: np.ndarray, mode: int) -> SimResult:
        """Execute M passes under the §5.2 independence rules.

        Scheduling per pass m (see module docstring for the event grammar):
          L_m            = msb_fire_{m-1} + 1          (P1/A9),  L_0 = 0
          compute_start  = max(L_m + LOAD_CYC, acc_free)
          msb_fire       = compute_start + P - 1
          acc_done       = msb_fire + 1
          acam           = max(acc_done + 1, out_free) (P11)
          out            = [acam + 1, acam + OUTPUT_CYC - 1]
        Values come from _corner_turn + _fire_pass in the same order.

        TODO(you): implement the loop, fill PassTimeline per pass, return
        SimResult(used_cycles = timeline[-1].out_end + 1).
        """
        raise NotImplementedError

    # -- stimulus dump (§9, consumed by the Stage 1.5 harness) --------------
    def dump_case(self, case_dir: Path, X: np.ndarray, mode: int) -> None:
        """Write stimulus + expected files for the RTL cross-check.

        Files (format contract):
          weights.mem    — ref.pack_weight_stream, one 10-hex-digit word/line
          act.mem        — all M bursts concatenated (same word format)
          expected_yint.npz — int32 [M, C]
          expected_out.mem  — uint8 [M, C] column-order bytes, 2 hex digits/byte
          case.json      — {R, C, BUF, P, M, mode, used_cycles, weight_cycles}

        TODO(you, optional now): implement; the harness falls back to
        in-memory compare if absent.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Analytical shadow (§5.3) — the closed form the measured schedule must
# reproduce for the canonical schedule.
# ---------------------------------------------------------------------------
def cycle_model(M: int, R: int, C: int, P: int = 8, BUF: int = 40) -> CycleModel:
    """§5.3 closed-form cycle contract.

    TODO(you): implement the four derived quantities exactly as the spec
    table (no other constants allowed):
      LOAD_CYC    = ceil(R*8/BUF)
      COMPUTE_CYC = P + 2
      OUTPUT_CYC  = ceil(C*8/BUF)
      WR_CYC      = ceil(R*C*8/BUF)
      T_fill      = LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
      T_steady    = max(LOAD_CYC + P, COMPUTE_CYC, OUTPUT_CYC + 1)
      total       = T_fill + (M-1) * T_steady
    """
    raise NotImplementedError


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
    assert wc == 13108, "WR_CYC mismatch"

    for M in (1, 2, 4, 8):
        X = rng.integers(-128, 128, size=(M, 256), dtype=np.int8)
        res = dpe.run_workload(X, MODE_REGULAR)

        # Values ≡ reference (§9).
        assert (res.y_int == ref.compute_y_int(W, X)).all(), f"F1 mismatch M={M}"
        assert (res.out_bytes == ref.acam_transform(res.y_int, MODE_REGULAR)).all()

        # Cycles ≡ closed form (§5.3): the measured schedule IS the theorem.
        assert res.used_cycles == cycle_model(M, 256, 256).total, (
            f"measured {res.used_cycles} != formula {cycle_model(M, 256, 256).total}"
        )
        # COMPUTE_CYC must emerge from the schedule, not be asserted (§10 P10).
        t = res.timeline[0]
        assert (t.acam - t.compute_start + 1) == 10

    # Identity weights (§8 I7): out[c] == x[c] in REGULAR mode.
    dpe_i = NldpeDpe(R=256, C=256)
    dpe_i.program_weights(np.eye(256, dtype=np.int8))
    X = rng.integers(-128, 128, size=(4, 256), dtype=np.int8)
    res = dpe_i.run_workload(X, MODE_REGULAR)
    assert (res.out_bytes == (X & 0xFF).astype(np.uint8)).all(), "identity failed"

    print("nldpe_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
