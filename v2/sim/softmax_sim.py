#!/usr/bin/env python3
"""softmax_sim.py — NL safe-softmax behavior simulator (lockstep row pipeline).

Machine model (structural contract; mirrors the intended `softmax_top.v`):
  lanes = 16 lockstep lanes; lane k owns rows {k, k+lanes, ...}
  (rows_per_lane = S / lanes).
  Per row step:
    A  row max (CLB tree)
    B  exp_input = clamp(scores - row_max, -128) chunked into n_exp
       consecutive pieces of elements_per_exp_crossbar = ceil(S/n_exp); each
       piece is one identity pass through that lane's EXP crossbar
       (eye weights, MODE_EXP)
    Cs unsigned output sum of the pieces -> log_input = min(sum >> log2 S, 127)
    C  the lane sums are split into n_log consecutive groups of
       lane_sums_per_log_group = ceil(lanes/n_log); each group is one identity
       pass through a shared LOG crossbar (eye weights, MODE_LOG)
    D  softmax_out = clamp(scores - row_max - log_output, -128, 127) int8

`n_exp` (per lane) and `n_log` (shared) are crossbar counts: they change
where work runs, never values (each element crosses exactly one crossbar).

Value contract : `v2/oracle/softmax_ref.py` (bit-exact; imported as ref).
Cycle contract : `SoftmaxCycleModel` / `softmax_cycle_model()` — locked
                 streaming anchors 290 (S=128, n_exp=1), 514 (S=256, n_exp=2),
                 956 (S=256, n_exp=1). Timing (pass latencies, fill/steady,
                 measured == shadow) is a separate todo; `run()` reports the
                 structural pass counts it issues and leaves `used_cycles`
                 None until that todo lands.

Run:  python3 v2/sim/softmax_sim.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make sibling modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import nldpe_sim as prim  # noqa: E402
import softmax_ref as ref  # noqa: E402
from dimm_sim import identity_pass  # noqa: E402

BUF = 40    # DPE port width (5 bytes/cycle)
P = 8       # activation bit precision


@dataclass
class SoftmaxCycleModel:
    """Locked NL streaming cycle form (spec §6 / SOFTMAX_STUDY §4).

    Anchors: S=128 -> 290, S=256 (n_exp=2) -> 514, S=256 (n_exp=1) -> 956.
    """

    lanes: int              # W = 16
    rows_per_lane: int      # RPL = S / lanes
    words_per_row: int      # lane-wide CLB stage width (WPR = S / 16)
    n_exp: int              # exp crossbars per lane
    n_log: int              # shared log crossbars
    exp_load_cycles: int    # ceil((S/n_exp)/5) — exp port rate per row (LCYC)
    fill: int               # (words_per_row+4) + (exp_load_cycles+10
                            #  + exp_load_cycles+2) + 20 + words_per_row+4
    steady: int             # max(words_per_row, exp_load_cycles)
    total: int              # fill + (rows_per_lane - 1) * steady


def softmax_cycle_model(S: int, n_exp: int = 1, lanes: int = 16,
                        C: int | None = None, n_log: int = 1) -> SoftmaxCycleModel:
    """NL locked streaming closed form (spec §6 / SOFTMAX_STUDY §4)."""
    assert S % lanes == 0, "S must divide evenly across lanes"
    if C is not None and -(-S // n_exp) > C:
        raise ValueError(
            f"exp crossbar width ceil(S/n_exp)={-(-S // n_exp)} exceeds C={C}")
    rows_per_lane = S // lanes
    words_per_row = S // lanes
    elements_per_exp_crossbar = -(-S // n_exp)
    exp_load_cycles = -(-elements_per_exp_crossbar // 5)
    steady = max(words_per_row, exp_load_cycles)
    fill = ((words_per_row + 4)
            + (exp_load_cycles + 10 + exp_load_cycles + 2)
            + 20 + words_per_row + 4)
    return SoftmaxCycleModel(
        lanes=lanes, rows_per_lane=rows_per_lane,
        words_per_row=words_per_row, n_exp=n_exp, n_log=n_log,
        exp_load_cycles=exp_load_cycles, fill=fill, steady=steady,
        total=fill + (rows_per_lane - 1) * steady,
    )


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
    used_cycles: int | None         # timing todo; None until then
    fill: int | None                # timing todo
    steady: int | None              # timing todo
    passes_exp: int                 # issued EXP identity passes (structural)
    passes_log: int                 # issued LOG identity passes (structural)
    stages: SoftmaxStages | None = None
    timeline: list = field(default_factory=list)  # stage events (timing todo)


class NldpeSoftmax:
    """NL safe-softmax on NL-DPE primitives (lockstep 16-lane row pipeline).

    Structure is frozen by the P0 contract; values are checked per stage
    against `softmax_ref.softmax_stage_values` (formula path) by the self-test.
    """

    def __init__(self, S: int, C: int = 128, lanes: int = 16,
                 n_exp: int | None = None, n_log: int = 1) -> None:
        assert S & (S - 1) == 0, "S must be a power of two"
        assert S % lanes == 0, "S must divide evenly across lanes"
        assert n_log >= 1, "n_log must be >= 1"
        self.S, self.C, self.lanes = S, C, lanes
        self.n_exp = n_exp if n_exp is not None else -(-S // C)
        self.elements_per_exp_crossbar = -(-S // self.n_exp)
        assert self.elements_per_exp_crossbar <= C, \
            f"exp crossbar width {self.elements_per_exp_crossbar} exceeds C={C}"
        self.n_log = n_log
        self.lane_sums_per_log_group = -(-lanes // n_log)

        # Resources: n_exp EXP crossbars per lane + n_log shared LOG crossbars,
        # all identity-programmed instances of the certified `dpe` primitive.
        exp_identity_eye = np.eye(self.elements_per_exp_crossbar,
                                  self.elements_per_exp_crossbar, dtype=np.int8)
        self._exp_crossbars = []
        for _ in range(lanes):
            lane_crossbars = []
            for _ in range(self.n_exp):
                crossbar = prim.NldpeDpe(self.elements_per_exp_crossbar,
                                         self.elements_per_exp_crossbar,
                                         BUF, P)
                crossbar.program_weights(exp_identity_eye)
                lane_crossbars.append(crossbar)
            self._exp_crossbars.append(lane_crossbars)
        log_identity_eye = np.eye(lanes, lanes, dtype=np.int8)
        self._log_crossbars = []
        for _ in range(n_log):
            crossbar = prim.NldpeDpe(lanes, lanes, BUF, P)
            crossbar.program_weights(log_identity_eye)
            self._log_crossbars.append(crossbar)

    def shadow_cycles(self) -> SoftmaxCycleModel:
        """Cycle contract for this instance's geometry (scaffold-owned)."""
        return softmax_cycle_model(self.S, self.n_exp, self.lanes,
                                   n_log=self.n_log)

    def run(self, scores: np.ndarray,
            collect_stages: bool = False) -> SoftmaxResult:
        """Run safe softmax on int8 scores [S, S] -> SoftmaxResult.

        Values must equal `softmax_ref.softmax_pass_view(scores)`; the staged
        dumps (when collected) must equal
        `softmax_ref.softmax_stage_values(scores)` bit-exactly. Cycles are
        owned by the timing todo.
        """
        scores = np.asarray(scores, dtype=np.int8)
        assert scores.shape == (self.S, self.S), \
            f"scores must be [{self.S}, {self.S}]"
        S = self.S
        lanes = self.lanes
        rows_per_lane = S // lanes
        elements_per_exp_crossbar = self.elements_per_exp_crossbar
        lane_sums_per_log_group = self.lane_sums_per_log_group
        mean_shift = S.bit_length() - 1

        # Lane mapping: row r -> lane r % lanes at step r // lanes.
        scores_by_lane = scores.reshape(rows_per_lane, lanes, S) \
            .transpose(1, 0, 2)                      # [lanes, rows_per_lane, S]
        softmax_out = np.empty_like(scores)
        if collect_stages:
            stage_row_max = np.empty(S, dtype=np.int8)
            stage_exp_input = np.empty((S, S), dtype=np.int8)
            stage_exp_crossbar_y = np.empty((S, S), dtype=np.int32)
            stage_exp_acam_output = np.empty((S, S), dtype=np.int8)
            stage_output_sum = np.empty(S, dtype=np.int64)
            stage_log_input = np.empty(S, dtype=np.int8)
            stage_log_output = np.empty(S, dtype=np.int8)
            stage_softmax_out = np.empty((S, S), dtype=np.int8)

        passes_exp = 0
        passes_log = 0
        for step in range(rows_per_lane):
            rows = scores_by_lane[:, step, :]            # [lanes, S] int8
            global_rows = np.arange(step * lanes, (step + 1) * lanes)
            rows_int16 = rows.astype(np.int16)

            # A — row max.
            row_max = rows.max(axis=1)                   # int8 [lanes]

            # B — subtract/clamp, then one identity EXP pass per chunk.
            exp_input = np.maximum(rows_int16 - row_max[:, None].astype(np.int16),
                                   -128).astype(np.int8)
            exp_acam_output = np.empty((lanes, S), dtype=np.int8)
            exp_crossbar_y = np.empty((lanes, S), dtype=np.int32)
            for chunk_index in range(self.n_exp):
                chunk_lo = chunk_index * elements_per_exp_crossbar
                chunk_hi = min(chunk_lo + elements_per_exp_crossbar, S)
                if chunk_lo >= chunk_hi:
                    continue
                for lane in range(lanes):
                    converted, pass_count, crossbar_y = identity_pass(
                        exp_input[lane, chunk_lo:chunk_hi], prim.MODE_EXP,
                        elements_per_exp_crossbar, elements_per_exp_crossbar,
                        BUF, P, dpe=self._exp_crossbars[lane][chunk_index],
                        return_y=True)
                    exp_acam_output[lane, chunk_lo:chunk_hi] = converted
                    exp_crossbar_y[lane, chunk_lo:chunk_hi] = crossbar_y
                    passes_exp += pass_count

            # Cs — unsigned output sum -> log input.
            output_sum = exp_acam_output.view(np.uint8).astype(np.int64).sum(axis=1)
            log_input = np.minimum(output_sum >> mean_shift, 127).astype(np.int8)

            # C — shared LOG passes over groups of lane sums.
            log_output = np.empty(lanes, dtype=np.int8)
            for group_index in range(self.n_log):
                group_lo = group_index * lane_sums_per_log_group
                group_hi = min(group_lo + lane_sums_per_log_group, lanes)
                if group_lo >= group_hi:
                    continue
                converted, pass_count = identity_pass(
                    log_input[group_lo:group_hi], prim.MODE_LOG, lanes, lanes,
                    BUF, P, dpe=self._log_crossbars[group_index])
                log_output[group_lo:group_hi] = converted
                passes_log += pass_count

            # D — log-domain output.
            step_output = np.clip(
                rows_int16 - row_max[:, None].astype(np.int16)
                - log_output[:, None].astype(np.int16),
                -128, 127).astype(np.int8)
            softmax_out[global_rows] = step_output
            if collect_stages:
                stage_row_max[global_rows] = row_max
                stage_exp_input[global_rows] = exp_input
                stage_exp_crossbar_y[global_rows] = exp_crossbar_y
                stage_exp_acam_output[global_rows] = exp_acam_output
                stage_output_sum[global_rows] = output_sum
                stage_log_input[global_rows] = log_input
                stage_log_output[global_rows] = log_output
                stage_softmax_out[global_rows] = step_output

        stages = (SoftmaxStages(row_max=stage_row_max, exp_input=stage_exp_input,
                                exp_crossbar_y=stage_exp_crossbar_y,
                                exp_acam_output=stage_exp_acam_output,
                                output_sum=stage_output_sum,
                                log_input=stage_log_input,
                                log_output=stage_log_output,
                                softmax_out=stage_softmax_out)
                  if collect_stages else None)
        return SoftmaxResult(softmax_out=softmax_out, used_cycles=None,
                             fill=None, steady=None, passes_exp=passes_exp,
                             passes_log=passes_log, stages=stages)


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/softmax_sim.py
# Gates: per-stage values bit-exact vs the staged oracle, count-invariance,
# edge classes, dual witnesses. Cycles are owned by the timing todo.
# ---------------------------------------------------------------------------
_STAGE_KEYS = ("row_max", "exp_input", "exp_crossbar_y", "exp_acam_output",
               "output_sum", "log_input", "log_output", "softmax_out")


def _check_stages(result: SoftmaxResult, expected_stages: dict,
                  label: str) -> None:
    stages = result.stages
    assert stages is not None, f"{label}: stages not collected"
    for key in _STAGE_KEYS:
        got, want = getattr(stages, key), expected_stages[key]
        if got.shape != want.shape or not np.array_equal(got, want):
            raise AssertionError(f"{label}: stage '{key}' mismatch")


def _self_test() -> None:
    # ── Locked streaming anchors (timing todo owns the measured gate) ──────
    assert softmax_cycle_model(128, n_exp=1).total == 290
    assert softmax_cycle_model(256, n_exp=2).total == 514
    assert softmax_cycle_model(256, n_exp=1).total == 956
    print("  [cycle model] locked anchors 290/514/956: OK")

    rng = np.random.default_rng(6)

    # ── Stage gates + count-invariance (same scores per S) ─────────────────
    for S, count_configs in ((128, [(1, 1), (2, 1), (1, 2), (4, 2), (3, 3)]),
                             (256, [(1, 1), (2, 1), (2, 2)])):
        scores = rng.integers(-128, 128, size=(S, S), dtype=np.int8)
        expected_stages = ref.softmax_stage_values(scores)
        outputs = []
        for n_exp, n_log in count_configs:
            softmax_unit = NldpeSoftmax(S=S, C=S, n_exp=n_exp, n_log=n_log)
            result = softmax_unit.run(scores, collect_stages=True)
            label = f"S={S} n_exp={n_exp} n_log={n_log}"
            _check_stages(result, expected_stages, label)
            assert np.array_equal(result.softmax_out,
                                  ref.softmax_pass_view(scores)), label
            assert result.passes_exp == S * n_exp, (label, result.passes_exp)
            assert result.passes_log == (S // 16) * n_log, \
                (label, result.passes_log)
            outputs.append(result.softmax_out)
            print(f"  [stages] {label}: row_max exp_input exp_crossbar_y "
                  "exp_acam_output output_sum log_input log_output softmax_out OK")
        for other in outputs[1:]:
            assert np.array_equal(other, outputs[0]), \
                f"S={S}: crossbar-count invariance violated"
        print(f"  [invariance] S={S}: softmax_out/exp_acam_output invariant "
              f"over n_exp x n_log = {count_configs}: OK")

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
    for name, scores in edges.items():
        softmax_unit = NldpeSoftmax(S=128, C=128, n_exp=2, n_log=2)
        result = softmax_unit.run(scores, collect_stages=True)
        _check_stages(result, ref.softmax_stage_values(scores), name)
        print(f"  [edges] {name}: stages OK")

    # ── Dual witnesses: staged/pass/dual elementwise views agree ───────────
    scores = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)
    result = NldpeSoftmax(S=128, C=128).run(scores)
    assert np.array_equal(result.softmax_out, ref.softmax_pass_view(scores)), \
        "pass view"
    assert np.array_equal(result.softmax_out,
                          ref.softmax_elementwise_view(scores)), \
        "elementwise dual view"
    print("  [witnesses] softmax_out == softmax_pass_view == "
          "softmax_elementwise_view: OK")

    print("softmax_sim self-test: ALL PASS "
          "(values + structure; timing pending)")


if __name__ == "__main__":
    _self_test()
