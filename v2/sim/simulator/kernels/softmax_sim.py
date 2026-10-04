#!/usr/bin/env python3
"""softmax_sim.py — NL safe-softmax fused behavior + timing simulator.

Machine model (DIMM-style crossbars, fused compute/cycle simulation):
  Certified (R, C) identity crossbars; capacity I = min(R, C) elements per
  identity pass; port width BUF (EPS = BUF/8 bytes/cycle).

    A  row max (CLB tree, latency = log2(S) + clb_pipe)
    B  EXP: the exp_input matrix read row-major (S^2 elements) converted by
       n_exp crossbars over packed stride-I windows (window j -> crossbar
       j % n_exp; final window zero-padded, padding discarded)
    Cs unsigned output sum -> log_input = min(sum >> log2 S, 127) (CLB tree,
       latency = log2(S) + clb_pipe)
    C  LOG: the S row sums converted by n_log crossbars over packed stride-I
       windows
    D  softmax_out = clamp(scores - row_max - log_output) (CLB, latency =
       1 + clb_pipe)

  Cycles are a by-product of execution — the simulator schedules unit events
  (primitive pass timelines for the crossbars, structural latencies for the
  CLB stages) and resolves the row-level dependency chain:

      row max -> EXP pass -> row-sum lq -> LOG pass -> clamp -> output drain

  Buffering is unbounded (full in-flight overlap): every unit starts a new
  item as soon as its inputs are ready and the unit is free. The row-max fold
  streams rows at `clb_width` bytes/cycle (row r completes `log2(S) + 1`
  cycles after its last byte enters), so EXP window readiness is staggered by
  row order; the row-sum `lq` keeps the tree latency after the row's last
  output drains. There are no residual cycle constants; the crossbar event
  offsets are the certified primitive model (`nldpe_sim.NldpeDpe.run_workload`
  semantics).

Value contract : `v2/oracle/softmax_ref.py` (bit-exact; imported as ref).
Check model    : `SoftmaxCycleModel` / `softmax_cycle_model()` — the phase
  envelope kept as a non-contract lower-bound check (it prices CLB at 0 and
  ignores per-window event detail); the self-test asserts
  `compute_cycles >= cycle_model.total`.

Run:  python3 v2/sim/softmax_sim.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make sibling modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "oracle"))
import nldpe_sim as prim  # noqa: E402
import nldpe_ref as nref  # noqa: E402
import softmax_ref as ref  # noqa: E402
from dimm_sim import identity_pass  # noqa: E402
from pass_engine import (crossbar_total, packed_windows,  # noqa: E402
                         schedule_pass_sequence)


# ---------------------------------------------------------------------------
# Check model (non-contract): phase envelope, CLB priced 0.
# ---------------------------------------------------------------------------
@dataclass
class SoftmaxPassPlan:
    """Pass counts the schedule actually issues (schedule-injected)."""

    passes_exp: int     # packed EXP windows = ceil(S^2 / I)
    passes_log: int     # packed LOG windows = ceil(S / I)


@dataclass
class SoftmaxCycleModel:
    """Phase-envelope lower bound (check only, NOT the timing contract).

    `total` excludes the output drain; the fused simulator's measured
    `compute_cycles` must be >= this bound.
    """

    S: int
    R: int
    C: int
    BUF: int
    P: int
    I: int                  # min(R, C): elements per identity pass
    n_exp: int              # EXP crossbars (windows round-robin)
    n_log: int              # LOG crossbars (windows round-robin)
    overlap: bool           # True: max(T_exp, T_start + T_log); False: sum
    passes_exp: int
    passes_log: int
    xbar_passes_exp: int    # ceil(passes_exp / n_exp)
    xbar_passes_log: int    # ceil(passes_log / n_log)
    T_exp: int              # T(xbar_passes_exp)
    T_log: int              # T(xbar_passes_log)
    T_start: int            # completion of EXP windows feeding LOG window 0
    output_cycles: int      # S^2: final drain, one value/cycle
    total: int              # phase envelope (excludes output_cycles)


def softmax_cycle_model(S: int, R: int = 256, C: int = 256, BUF: int = 40,
                        P: int = 8, n_exp: int = 1, n_log: int = 1,
                        overlap: bool = True, *,
                        plan: SoftmaxPassPlan) -> SoftmaxCycleModel:
    """Phase envelope for the packed-window structure (check model)."""
    assert plan.passes_exp >= 0 and plan.passes_log >= 0
    assert n_exp >= 1 and n_log >= 1
    I = min(R, C)
    xbar_passes_exp = -(-plan.passes_exp // n_exp) if plan.passes_exp else 0
    xbar_passes_log = -(-plan.passes_log // n_log) if plan.passes_log else 0
    T_exp = crossbar_total(xbar_passes_exp, R, C, P, BUF)
    T_log = crossbar_total(xbar_passes_log, R, C, P, BUF)

    first_log_rows = min(I, S)
    windows_for_first_log = (-(-(first_log_rows * S) // I)
                             if first_log_rows else 0)
    if windows_for_first_log:
        last_window = windows_for_first_log - 1
        T_start = crossbar_total(last_window // n_exp + 1, R, C, P, BUF)
    else:
        T_start = 0

    output_cycles = S * S
    total = (max(T_exp, T_start + T_log) if overlap else T_exp + T_log)
    return SoftmaxCycleModel(
        S=S, R=R, C=C, BUF=BUF, P=P, I=I, n_exp=n_exp, n_log=n_log,
        overlap=overlap, passes_exp=plan.passes_exp,
        passes_log=plan.passes_log, xbar_passes_exp=xbar_passes_exp,
        xbar_passes_log=xbar_passes_log, T_exp=T_exp, T_log=T_log,
        T_start=T_start, output_cycles=output_cycles, total=total,
    )


# ---------------------------------------------------------------------------
# Packed-window value conversion (packing + timing live in `pass_engine`).
# ---------------------------------------------------------------------------
def _convert_packed_stream(flat: np.ndarray, mode: int, R: int, C: int,
                           BUF: int, P: int, crossbars: list,
                           return_y: bool = False) -> tuple:
    """Convert a flat stream through identity crossbars, packed windows.

    Values only (timing lives in the event scheduler). Returns
    (values, passes_per_crossbar) — plus the pre-ACAM wide output `y` when
    `return_y=True` — with values in original element order.
    """
    flat = np.asarray(flat, dtype=np.int8).reshape(-1)
    I = min(R, C)
    _, window_lo, window_len, per_crossbar = packed_windows(
        flat.size, I, len(crossbars))

    values = np.empty(flat.size, dtype=np.int8)
    y_values = np.empty(flat.size, dtype=np.int32) if return_y else None
    passes_per_crossbar = []
    for c, window_ids in enumerate(per_crossbar):
        if not window_ids:
            passes_per_crossbar.append(0)
            continue
        chunk = np.concatenate([flat[window_lo[j]:window_lo[j] + window_len[j]]
                                for j in window_ids])
        if return_y:
            converted, pass_count, y = identity_pass(
                chunk, mode, R, C, BUF, P, dpe=crossbars[c], return_y=True)
        else:
            converted, pass_count = identity_pass(
                chunk, mode, R, C, BUF, P, dpe=crossbars[c])
        passes_per_crossbar.append(pass_count)
        offset = 0
        for j in window_ids:
            length = int(window_len[j])
            values[window_lo[j]:window_lo[j] + length] = \
                converted[offset:offset + length]
            if return_y:
                y_values[window_lo[j]:window_lo[j] + length] = \
                    y[offset:offset + length]
            offset += length
    if return_y:
        return values, passes_per_crossbar, y_values
    return values, passes_per_crossbar


@dataclass
class SoftmaxStages:
    """Staged diagnostics for one softmax run (`collect_stages=True`).

    Stage order mirrors the datapath: row max -> subtract/clamp -> EXP crossbar
    y (int32, pre-ACAM) -> ACAM output -> output sum / log input -> LOG ACAM
    output -> final softmax output.
    """

    row_max: np.ndarray          # int8  [S]     row maximum
    exp_input: np.ndarray        # int8  [S, S]  max(scores - row_max, -128)
    exp_crossbar_y: np.ndarray   # int32 [S, S]  EXP crossbar y (pre-ACAM; == exp_input)
    exp_acam_output: np.ndarray  # int8  [S, S]  ACAM_EXP output
    output_sum: np.ndarray       # int64 [S]     unsigned exp-output row sums
    log_input: np.ndarray        # int8  [S]     min(output_sum >> log2 S, 127)
    log_output: np.ndarray       # int8  [S]     ACAM_LOG(log_input)
    softmax_out: np.ndarray      # int8  [S, S]  log-domain output


@dataclass
class SoftmaxResult:
    """Everything the self-test / later cross-checks consume."""

    softmax_out: np.ndarray         # int8 [S, S] log-domain output
    used_cycles: int                # measured makespan (incl. output drain)
    compute_cycles: int             # last output value ready (before drain)
    drain_cycles: int               # S^2: output drain length (one value/cycle)
    passes_exp: int                 # issued EXP window passes (all crossbars)
    passes_log: int                 # issued LOG window passes
    cycle_model: SoftmaxCycleModel  # phase-envelope check (lower bound)
    stages: SoftmaxStages | None = None
    timeline: list = field(default_factory=list)  # (unit, index, n, start, end)


class NldpeSoftmax:
    """NL safe-softmax on NL-DPE primitives (packed-window fused simulator).

    Structure is frozen by the P0 contract; values are checked per stage
    against `softmax_ref.softmax_stage_values` (formula path) by the self-test,
    and timing is measured from the unit event schedule.
    """

    def __init__(self, S: int, R: int = 256, C: int = 256, BUF: int = 40,
                 P: int = 8, n_exp: int = 1, n_log: int = 1,
                 overlap: bool = True, clb_pipe: int = 1,
                 clb_width: int = 32) -> None:
        assert S > 0 and (S & (S - 1)) == 0, "S must be a power of two"
        assert n_exp >= 1 and n_log >= 1
        assert clb_width > 0, "clb_width must be positive"
        self.S, self.R, self.C, self.BUF, self.P = S, R, C, BUF, P
        self.I = min(R, C)
        self.n_exp, self.n_log = n_exp, n_log
        self.overlap = overlap
        self.clb_pipe = clb_pipe
        self.clb_width = clb_width
        log2_S = S.bit_length() - 1
        self.clb_tree_latency = log2_S + clb_pipe     # max / sum trees
        self.clb_clamp_latency = 1 + clb_pipe         # subtract + clamp

        # Resources: n_exp EXP + n_log LOG identity crossbars (certified `dpe`).
        identity_eye = np.eye(R, C, dtype=np.int8)
        self._exp_crossbars = []
        for _ in range(n_exp):
            crossbar = prim.NldpeDpe(R, C, BUF, P)
            crossbar.program_weights(identity_eye)
            self._exp_crossbars.append(crossbar)
        self._log_crossbars = []
        for _ in range(n_log):
            crossbar = prim.NldpeDpe(R, C, BUF, P)
            crossbar.program_weights(identity_eye)
            self._log_crossbars.append(crossbar)

    def shadow_cycles(self, plan: SoftmaxPassPlan) -> SoftmaxCycleModel:
        """Phase envelope for this instance's geometry (check model)."""
        return softmax_cycle_model(
            self.S, R=self.R, C=self.C, BUF=self.BUF, P=self.P,
            n_exp=self.n_exp, n_log=self.n_log, overlap=self.overlap,
            plan=plan)

    def _run_machine(self, exp_events: list, log_events: list,
                     lq_ready: np.ndarray) -> dict:
        """Resolve the dependency DAG over unit events; measure cycles.

        exp_events[c] / log_events[c]: per-window event dicts (one per window
        of that crossbar); lq_ready[r]: row-sum readiness from the EXP events.
        Returns the measured cycles and the derived readiness / drain timeline.
        """
        S = self.S
        EPS = self.BUF // 8
        _, log_lo, log_len, per_xbar_log = packed_windows(S, self.I,
                                                           self.n_log)

        # out_ready[r]: clamp output ready after the LOG window holding r
        # drains (word index within the pass) + L_clamp.
        out_ready = np.zeros(S, dtype=np.int64)
        for crossbar in range(self.n_log):
            for position, window in enumerate(per_xbar_log[crossbar]):
                event = log_events[crossbar][position]
                rows_lo = int(log_lo[window])
                rows_hi = rows_lo + int(log_len[window])
                for r in range(rows_lo, rows_hi):
                    drain_cycle = (event["drain_start"]
                                   + (r - rows_lo) // EPS)
                    out_ready[r] = (drain_cycle + 1
                                    + self.clb_clamp_latency)

        # Output drain: row-major, one value per cycle, gated per row.
        next_free = 0
        drain_start_cycle = 0
        for r in range(S):
            if next_free < out_ready[r]:
                next_free = int(out_ready[r])
            if r == 0:
                drain_start_cycle = next_free
            next_free += S
        used_cycles = next_free

        timeline = []
        for crossbar in range(self.n_exp):
            events = exp_events[crossbar]
            if events:
                timeline.append(("exp", crossbar, len(events),
                                 events[0]["load_start"],
                                 events[-1]["drain_end"]))
        for crossbar in range(self.n_log):
            events = log_events[crossbar]
            if events:
                timeline.append(("log", crossbar, len(events),
                                 events[0]["load_start"],
                                 events[-1]["drain_end"]))
        timeline.append(("out", None, S, int(out_ready.min()),
                         int(out_ready.max())))
        timeline.append(("drain", None, S * S, drain_start_cycle,
                         used_cycles - 1))
        return {"used_cycles": used_cycles,
                "compute_cycles": int(out_ready.max()),
                "drain_start_cycle": drain_start_cycle,
                "timeline": timeline}

    def exp_window_ready(self) -> np.ndarray:
        """Readiness cycle of each packed EXP window (streaming row-max fold).

        Row r completes `clb_tree_latency` cycles after its last byte passes
        the tree (at `clb_width` bytes/cycle); a window is ready when all the
        rows it covers are ready.
        """
        S, I = self.S, self.I
        n_windows, window_lo, window_len, _ = packed_windows(
            S * S, I, self.n_exp)
        row_max_ready = np.array(
            [self.clb_tree_latency + -(-(r + 1) * S // self.clb_width)
             for r in range(S)], dtype=np.int64)
        ready = np.zeros(n_windows, dtype=np.int64)
        for j in range(n_windows):
            lo = int(window_lo[j])
            hi = lo + int(window_len[j])
            rows_lo, rows_hi = lo // S, (hi - 1) // S
            ready[j] = int(row_max_ready[rows_lo:rows_hi + 1].max())
        return ready

    def run(self, scores: np.ndarray,
            collect_stages: bool = False) -> SoftmaxResult:
        """Run safe softmax on int8 scores [S, S] -> SoftmaxResult.

        Values must equal `softmax_ref.softmax_pass_view(scores)`; the staged
        dumps (when collected) must equal
        `softmax_ref.softmax_stage_values(scores)` bit-exactly. Cycles are
        measured from the fused unit schedule (`used_cycles` includes the
        output drain; `compute_cycles` is the last value ready).
        """
        scores = np.asarray(scores, dtype=np.int8)
        assert scores.shape == (self.S, self.S), \
            f"scores must be [{self.S}, {self.S}]"
        S, I = self.S, self.I
        mean_shift = S.bit_length() - 1
        scores_int16 = scores.astype(np.int16)

        # A — row max (CLB, structural latency L_max).
        row_max = scores.max(axis=1)
        exp_input = np.maximum(scores_int16 - row_max[:, None].astype(np.int16),
                               -128).astype(np.int8)

        # B — packed EXP conversion (values; timing via the event scheduler).
        exp_acam_flat, exp_passes, exp_y_flat = _convert_packed_stream(
            exp_input.reshape(-1), prim.MODE_EXP, self.R, self.C, self.BUF,
            self.P, self._exp_crossbars, return_y=True)
        exp_acam_output = exp_acam_flat.reshape(S, S)
        exp_crossbar_y = exp_y_flat.reshape(S, S)

        # Cs — unsigned output sum -> log input (CLB, structural latency L_sum).
        output_sum = exp_acam_output.view(np.uint8).astype(np.int64).sum(axis=1)
        log_input = np.minimum(output_sum >> mean_shift, 127).astype(np.int8)

        # C — packed LOG conversion (values; timing via the event scheduler).
        log_output, log_passes = _convert_packed_stream(
            log_input, prim.MODE_LOG, self.R, self.C, self.BUF, self.P,
            self._log_crossbars)

        # D — log-domain output (CLB, structural latency L_clamp).
        softmax_out = np.clip(
            scores_int16 - row_max[:, None].astype(np.int16)
            - log_output[:, None].astype(np.int16),
            -128, 127).astype(np.int8)

        # Event schedule: the row-max fold streams rows at clb_width bytes/
        # cycle, so row r completes max_latency cycles after its last byte
        # passes the tree; an EXP window is ready when all its rows are.
        # LOG windows are ready when their lq inputs are ready.
        n_windows_exp, _, _, per_xbar_exp = packed_windows(
            S * S, I, self.n_exp)
        n_windows_log, log_lo, log_len, per_xbar_log = packed_windows(
            S, I, self.n_log)
        exp_ready = self.exp_window_ready()
        exp_events = [
            schedule_pass_sequence([int(exp_ready[j]) for j in per_xbar_exp[c]],
                                   self.R, self.C, self.BUF, self.P)
            for c in range(self.n_exp)]

        lq_ready = self._lq_ready_from_events(exp_events)
        log_ready = np.zeros(n_windows_log, dtype=np.int64)
        for w in range(n_windows_log):
            rows_lo = int(log_lo[w])
            rows_hi = rows_lo + int(log_len[w])
            log_ready[w] = (int(lq_ready[rows_lo:rows_hi].max())
                            if rows_hi > rows_lo else 0)
        log_events = [
            schedule_pass_sequence([int(log_ready[w]) for w in per_xbar_log[c]],
                               self.R, self.C, self.BUF, self.P)
            for c in range(self.n_log)]

        machine = self._run_machine(exp_events, log_events, lq_ready)
        plan = SoftmaxPassPlan(passes_exp=int(sum(exp_passes)),
                               passes_log=int(sum(log_passes)))
        cycle_model = self.shadow_cycles(plan)
        stages = (SoftmaxStages(row_max=row_max, exp_input=exp_input,
                                exp_crossbar_y=exp_crossbar_y,
                                exp_acam_output=exp_acam_output,
                                output_sum=output_sum, log_input=log_input,
                                log_output=log_output, softmax_out=softmax_out)
                  if collect_stages else None)
        return SoftmaxResult(softmax_out=softmax_out,
                             used_cycles=machine["used_cycles"],
                             compute_cycles=machine["compute_cycles"],
                             drain_cycles=S * S,
                             passes_exp=plan.passes_exp,
                             passes_log=plan.passes_log,
                             cycle_model=cycle_model,
                             stages=stages,
                             timeline=machine["timeline"])

    # -- stimulus dump (GATE 1 + GATE 2 expected files) ---------------------
    def dump_case(self, case_dir: Path, scores: np.ndarray) -> None:
        """Write stimulus + expected files for the softmax RTL cross-check.

        GATE 1: certify this exact case against `softmax_ref` (all eight
        stages) and the packed pass counts BEFORE any file is written; an
        uncertified case is never dumped.

        File contract (consumed by `v2/tb/tb_softmax_top.v`, expanded by
        `v2/smoke/run_softmax_rtl.py`):
          scores.mem            — S*S int8 row-major, packed BUF-wide (10-hex)
          expected_rowmax.mem   — S int8 (2-hex/line)
          expected_expin.mem    — S*S int8  (crossbar feed input)
          expected_expout.mem   — S*S int8  (ACAM_EXP output)
          expected_sum.mem      — S int32 (8-hex/line)
          expected_lq.mem       — S int8
          expected_logout.mem   — S int8   (ACAM_LOG output)
          expected_out.mem      — S*S int8 (softmax_out stream order)
          case.json             — params + pass counts + measured cycles
        """
        scores = np.asarray(scores, dtype=np.int8)
        assert scores.shape == (self.S, self.S), \
            f"scores must be [{self.S}, {self.S}]"
        result = self.run(scores, collect_stages=True)
        stages = result.stages
        expected = ref.softmax_stage_values(scores)

        # -- GATE 1: per-case oracle certification -------------------------
        for key in _STAGE_KEYS:
            got, want = getattr(stages, key), expected[key]
            if got.shape != want.shape or not np.array_equal(got, want):
                raise RuntimeError(
                    f"GATE 1: sim/oracle stage '{key}' mismatch "
                    f"— case not dumped")
        expected_passes = ref.packed_pass_counts(self.S, self.R, self.C)
        issued_passes = (result.passes_exp, result.passes_log)
        if issued_passes != expected_passes:
            raise RuntimeError(
                f"GATE 1: issued passes {issued_passes} != schedule formula "
                f"{expected_passes} — case not dumped")

        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        # scores.mem — packed BUF-wide stream words
        with open(case_dir / "scores.mem", "w") as f:
            for word in nref.pack_act_stream(scores.reshape(-1)):
                f.write(f"{word:010x}\n")

        def write_int8(name: str, values: np.ndarray) -> None:
            with open(case_dir / name, "w") as f:
                for v in np.asarray(values).reshape(-1):
                    f.write(f"{int(v) & 0xFF:02x}\n")

        def write_int32(name: str, values: np.ndarray) -> None:
            with open(case_dir / name, "w") as f:
                for v in np.asarray(values).reshape(-1):
                    f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

        write_int8("expected_rowmax.mem", stages.row_max)
        write_int8("expected_expin.mem", stages.exp_input)
        write_int8("expected_expout.mem", stages.exp_acam_output)
        write_int32("expected_sum.mem", stages.output_sum)
        write_int8("expected_lq.mem", stages.log_input)
        write_int8("expected_logout.mem", stages.log_output)
        write_int8("expected_out.mem", stages.softmax_out)

        case = {
            "S": self.S, "R": self.R, "C": self.C, "BUF": self.BUF,
            "P": self.P, "n_exp": self.n_exp, "n_log": self.n_log,
            "clb_width": self.clb_width, "clb_pipe": self.clb_pipe,
            "passes_exp": int(result.passes_exp),
            "passes_log": int(result.passes_log),
            "used_cycles": int(result.compute_cycles),   # done = results ready
            "compute_cycles": int(result.compute_cycles),
            "drain_cycles": 0,                           # output not counted
            "load_cycles": int(len(nref.pack_act_stream(scores.reshape(-1)))),
            "serialize_cycles": 0,                       # output not counted
            "e2e_cycles": int(len(nref.pack_act_stream(scores.reshape(-1)))
                              + result.compute_cycles),
            "load_words": int(len(nref.pack_act_stream(scores.reshape(-1)))),
            "weight_cycles": int(self.R * self.C),
        }
        with open(case_dir / "case.json", "w") as f:
            json.dump(case, f, indent=2)
            f.write("\n")

    def _lq_ready_from_events(self, exp_events: list) -> np.ndarray:
        """Row-sum readiness from the EXP event schedule (shared by run)."""
        S, I = self.S, self.I
        EPS = self.BUF // 8
        _, exp_lo, _, per_xbar_exp = packed_windows(S * S, I, self.n_exp)
        lq_ready = np.zeros(S, dtype=np.int64)
        for r in range(S):
            element = (r + 1) * S - 1
            window = int(element // I)
            crossbar = window % self.n_exp
            position = per_xbar_exp[crossbar].index(window)
            drain_cycle = (exp_events[crossbar][position]["drain_start"]
                           + (element - int(exp_lo[window])) // EPS)
            lq_ready[r] = drain_cycle + 1 + self.clb_tree_latency
        return lq_ready


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/softmax_sim.py
# Gates: per-stage values bit-exact vs the staged oracle, pass counts vs the
# schedule formula, count-invariance, measured-timing sanity (monotonicity,
# drain, analytical lower bound), edge classes, dual witnesses.
# ---------------------------------------------------------------------------
_STAGE_KEYS = ("row_max", "exp_input", "exp_crossbar_y", "exp_acam_output",
               "output_sum", "log_input", "log_output", "softmax_out")

# Phase-0 characterization freeze (refactor regression contract): measured
# (used_cycles, compute_cycles) per config. These are schedule-determined —
# independent of the score values — so any input of the right shape matches.
_FROZEN_MEASURED_PINS = {
    (128, 1, 1): (20367, 4008),
    (128, 4, 2): (17511, 1152),
    (256, 1, 1): (81041, 15556),
    (256, 2, 2): (73369, 7884),
}


def _check_stages(result: SoftmaxResult, expected_stages: dict,
                  label: str) -> None:
    stages = result.stages
    assert stages is not None, f"{label}: stages not collected"
    for key in _STAGE_KEYS:
        got, want = getattr(stages, key), expected_stages[key]
        if got.shape != want.shape or not np.array_equal(got, want):
            raise AssertionError(f"{label}: stage '{key}' mismatch")


def _self_test() -> None:
    # ── Check-model arithmetic (schedule-injected plans) ───────────────────
    plan = SoftmaxPassPlan(*ref.packed_pass_counts(128, 256, 256))
    cm = softmax_cycle_model(128, 256, 256, 40, 8, n_exp=4, n_log=1,
                             plan=plan)
    assert (cm.passes_exp, cm.passes_log) == (64, 1)
    assert (cm.xbar_passes_exp, cm.T_exp) == (16, 1014)   # T(16)=114+15*60
    assert (cm.xbar_passes_log, cm.T_log) == (1, 114)     # T(1) = T_fill
    assert cm.T_start == cm.T_exp                         # S <= I: serial tail
    assert cm.total == 1128                               # 1014 + 114
    assert cm.output_cycles == 128 * 128
    print(f"  [check model] S<=I: T_exp={cm.T_exp} T_log={cm.T_log} "
          f"T_start=T_exp total={cm.total}: OK")

    plan512 = SoftmaxPassPlan(*ref.packed_pass_counts(512, 256, 256))
    ovl = softmax_cycle_model(512, 256, 256, 40, 8, n_exp=8, n_log=1,
                              overlap=True, plan=plan512)
    seq = softmax_cycle_model(512, 256, 256, 40, 8, n_exp=8, n_log=1,
                              overlap=False, plan=plan512)
    assert (ovl.passes_exp, ovl.T_exp) == (1024, 7734)    # T(128)
    assert (ovl.T_start, ovl.T_log) == (3894, 174)        # T(64), T(2)
    assert ovl.total == 7734 and seq.total == 7908
    print(f"  [check model] S>I: overlap={ovl.total} < serial={seq.total}: OK")

    rng = np.random.default_rng(6)

    # ── Stage gates + count-invariance (same scores per S) ─────────────────
    for S, count_configs in ((128, [(1, 1), (2, 1), (4, 2), (3, 3)]),
                             (256, [(1, 1), (2, 2)])):
        scores = rng.integers(-128, 128, size=(S, S), dtype=np.int8)
        expected_stages = ref.softmax_stage_values(scores)
        expected_passes = ref.packed_pass_counts(S, 256, 256)
        outputs = []
        for n_exp, n_log in count_configs:
            softmax_unit = NldpeSoftmax(S=S, n_exp=n_exp, n_log=n_log)
            result = softmax_unit.run(scores, collect_stages=True)
            label = f"S={S} n_exp={n_exp} n_log={n_log}"
            _check_stages(result, expected_stages, label)
            assert np.array_equal(result.softmax_out,
                                  ref.softmax_pass_view(scores)), label
            assert (result.passes_exp, result.passes_log) == expected_passes, \
                (label, result.passes_exp, result.passes_log)
            assert result.used_cycles >= result.compute_cycles, label
            assert result.used_cycles >= result.drain_cycles, label
            frozen_pin = _FROZEN_MEASURED_PINS.get((S, n_exp, n_log))
            if frozen_pin is not None:
                assert (result.used_cycles, result.compute_cycles) \
                    == frozen_pin, \
                    (label, result.used_cycles, result.compute_cycles,
                     frozen_pin)
            outputs.append(result.softmax_out)
            print(f"  [stages] {label}: row_max exp_input exp_crossbar_y "
                  "exp_acam_output output_sum log_input log_output softmax_out OK")
        for other in outputs[1:]:
            assert np.array_equal(other, outputs[0]), \
                f"S={S}: crossbar-count invariance violated"
        print(f"  [invariance] S={S}: softmax_out/exp_acam_output invariant "
              f"over n_exp x n_log = {count_configs}: OK")
    print(f"  [freeze] Phase-0 measured pins matched for "
          f"{len(_FROZEN_MEASURED_PINS)} configs: OK")

    # ── Measured-timing sanity (fused schedule) ────────────────────────────
    # The check model is a corridor: the precise schedule cannot finish the
    # last output before the EXP phase (lower bound) and cannot exceed the
    # phase-separated envelope + CLB latencies (upper bound).
    scores = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)
    plan_128 = SoftmaxPassPlan(*ref.packed_pass_counts(128, 256, 256))
    serial_1 = softmax_cycle_model(128, 256, 256, 40, 8, n_exp=1, n_log=1,
                                   overlap=False, plan=plan_128)
    serial_4 = softmax_cycle_model(128, 256, 256, 40, 8, n_exp=4, n_log=1,
                                   overlap=False, plan=plan_128)
    unit_1 = NldpeSoftmax(S=128, n_exp=1, n_log=1)
    unit_4 = NldpeSoftmax(S=128, n_exp=4, n_log=1)
    run_1 = unit_1.run(scores)
    run_4 = unit_4.run(scores)
    assert run_4.used_cycles < run_1.used_cycles, \
        (run_4.used_cycles, run_1.used_cycles)
    clb_slack = 2 * unit_4.clb_tree_latency + unit_4.clb_clamp_latency
    prologue_1 = int(unit_1.exp_window_ready()[:unit_1.n_exp].max())
    prologue_4 = int(unit_4.exp_window_ready()[:unit_4.n_exp].max())
    assert run_1.compute_cycles >= run_1.cycle_model.T_exp \
        + unit_1.clb_tree_latency
    assert run_1.compute_cycles <= serial_1.total + prologue_1 + clb_slack
    assert run_4.compute_cycles >= run_4.cycle_model.T_exp \
        + unit_4.clb_tree_latency
    assert run_4.compute_cycles <= serial_4.total + prologue_4 + clb_slack
    assert run_4.used_cycles - run_4.compute_cycles < run_4.drain_cycles, \
        "drain must overlap compute, not serialize after it"
    # n_log cannot bind for S <= I (single LOG window): timing is identical.
    run_4b = NldpeSoftmax(S=128, n_exp=4, n_log=2).run(scores)
    assert run_4b.used_cycles == run_4.used_cycles, \
        (run_4b.used_cycles, run_4.used_cycles)
    print(f"  [measured] S=128: n_exp=1 -> {run_1.used_cycles} cyc "
          f"(compute {run_1.compute_cycles}), n_exp=4 -> "
          f"{run_4.used_cycles} cyc (compute {run_4.compute_cycles}); "
          f"corridors [T_exp+L_max, serial+prologue+slack] = "
          f"[{run_4.cycle_model.T_exp + unit_4.clb_tree_latency}, "
          f"{serial_4.total + prologue_4 + clb_slack}]: OK")

    # ── PLOG > 1: n_log splits the LOG windows, values invariant ───────────
    small_scores = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)
    small_outs = {}
    small_tlogs = {}
    for n_log in (1, 2):
        unit = NldpeSoftmax(S=128, R=64, C=64, n_exp=2, n_log=n_log)
        res = unit.run(small_scores, collect_stages=True)
        _check_stages(res, ref.softmax_stage_values(small_scores),
                      f"S=128 R=C=64 n_log={n_log}")
        assert (res.passes_exp, res.passes_log) == (256, 2), (n_log, res.passes_log)
        assert res.cycle_model.xbar_passes_log == -(-2 // n_log)
        small_outs[n_log] = res.softmax_out
        small_tlogs[n_log] = res.cycle_model.T_log
    assert np.array_equal(small_outs[1], small_outs[2]), "n_log changed values"
    assert small_tlogs[1] == crossbar_total(2, 64, 64, 8, 40)
    assert small_tlogs[2] == crossbar_total(1, 64, 64, 8, 40)
    assert small_tlogs[2] < small_tlogs[1]
    print(f"  [n_log split] R=C=64 S=128 PLOG=2: values invariant, "
          f"T_log {small_tlogs[1]} -> {small_tlogs[2]}: OK")

    # ── Edge classes (routing exercised: n_exp=2, n_log=2) ─────────────────
    tie_row = np.full(128, 5, dtype=np.int8)
    single_max = np.zeros((128, 128), dtype=np.int8)
    single_max[:, 7] = 127
    edges = {
        "extremes": np.array([[-128, 127] * 64, [127, -128] * 64] * 64,
                             dtype=np.int8),
        "uniform": np.zeros((128, 128), dtype=np.int8),
        "ties": np.tile(tie_row, (128, 1)),
        "single-max": single_max,
    }
    for name, edge_scores in edges.items():
        softmax_unit = NldpeSoftmax(S=128, n_exp=2, n_log=2)
        result = softmax_unit.run(edge_scores, collect_stages=True)
        _check_stages(result, ref.softmax_stage_values(edge_scores), name)
        print(f"  [edges] {name}: stages OK")

    # ── Dual witnesses: staged/pass/dual elementwise views agree ───────────
    witness_scores = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)
    result = NldpeSoftmax(S=128).run(witness_scores)
    assert np.array_equal(result.softmax_out,
                          ref.softmax_pass_view(witness_scores)), "pass view"
    assert np.array_equal(result.softmax_out,
                          ref.softmax_elementwise_view(witness_scores)), \
        "elementwise dual view"
    print("  [witnesses] softmax_out == softmax_pass_view == "
          "softmax_elementwise_view: OK")

    print("softmax_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
