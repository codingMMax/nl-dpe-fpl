#!/usr/bin/env python3
"""softmax_ref.py — NumPy value reference for the NL safe-softmax operator.

Role
----
Plain math, no state, no time. The behavior simulator (`v2/sim/softmax_sim.py`)
must agree with this file bit-exactly on values; the cycle contract lives in
that simulator (`softmax_cycle_model`), not here.

NL log-domain safe softmax (per row, integer; S must be a power of two)
-----------------------------------------------------------------------
    row_max            = max_j scores[j]                    (study symbol: m)
    exp_input          = max(scores[j] - row_max, -128)     (d) log-domain headroom
    exp_acam_output    = ACAM_EXP(exp_input)                (eb) MODE_EXP (P24)
    output_sum         = sum_j exp_acam_output              (s) unsigned output sum
    log_input          = min(output_sum >> log2(S), 127)    (lq) mean, capped
    log_output         = ACAM_LOG(log_input)                (ls) MODE_LOG (P24)
    softmax_out[j]     = clamp(scores[j] - row_max - log_output, -128, 127)

Transcribed from the locked NL contract of `softmax_study/SOFTMAX_STUDY.md`
(§1 mapping, §4 formulas) and `softmax_study/run_softmax_smoke.py::oracle_nl`,
re-expressed with the v2 ACAM forms (`v2/spec/dpe_nldpe.md` §6 F3 / P16 / P24).
For `exp_input` in [-128, 0] the v2 EXP form and the study's
`(1 + v + v^2//2) & 0xFF` are bit-identical (result in [0, 8065], no int32
clamp reachable).

Notes
-----
* Operator pass layer (§6 F5-F8): ACAM never runs standalone — the EXP of a
  row slice and the LOG of the lane sums are identity-crossbar passes with
  capacity I = min(R,C); pass counts are schedule-owned. `softmax_pass_view`
  is the pass view; `softmax_elementwise_view` is the dual view and the
  self-test asserts bit-equality (the packing/geometry invariance).
* Output is LOG-domain (log p_i approximation), consumed directly by the
  downstream S*V DIMM input path (`mac_sv`); it is not comparable with a
  linear-domain (Azure-Lily) softmax.
* The model is the contract: ACAM exp/log are declared approximations, so
  accuracy vs float softmax is out of scope (cf. spec A13).
* `log_input` uses an arithmetic right shift by log2(S) (mean) and caps at 127
  because the shared log DPE consumes an int8-domain value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the sibling oracle module importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_ref as ref  # noqa: E402


def softmax_elementwise_view(scores: np.ndarray) -> np.ndarray:
    """Dual (elementwise, geometry-free) view of NL safe-softmax — int8 [S, S].

    Kept as the independent witness for the pass view; do not consume it from
    the behavior sim (use `softmax_pass_view`).
    """
    scores = np.asarray(scores, dtype=np.int32)
    assert scores.ndim == 2 and scores.shape[0] == scores.shape[1], \
        "scores must be [S, S]"
    S = scores.shape[1]
    assert S & (S - 1) == 0, "S must be a power of two"
    mean_shift = S.bit_length() - 1

    softmax_out = np.zeros_like(scores, dtype=np.int8)
    for r in range(scores.shape[0]):
        score_row = scores[r]
        row_max = int(score_row.max())
        exp_input = np.maximum(score_row - row_max, -128)
        exp_acam_output = ref.acam_transform(exp_input, ref.MODE_EXP) \
            .view(np.uint8).astype(np.int64)
        output_sum = int(exp_acam_output.sum())
        log_input = min(output_sum >> mean_shift, 127)
        log_output = int(ref.acam_transform(
            np.array([log_input], dtype=np.int32), ref.MODE_LOG)[0])
        softmax_out[r] = np.clip(score_row - row_max - log_output,
                                 -128, 127).astype(np.int8)
    return softmax_out


def softmax_pass_view(scores: np.ndarray, R: int = 256, C: int = 256
                      ) -> np.ndarray:
    """NL safe-softmax over an S x S score matrix (identity-pass view). int8.

    scores : int8 (or integer) [S, S], S a power of two. Log-domain output.
    R, C   : crossbar geometry for the EXP/LOG identity passes (F5/F6).
    """
    scores = np.asarray(scores, dtype=np.int32)
    assert scores.ndim == 2 and scores.shape[0] == scores.shape[1], \
        "scores must be [S, S]"
    S = scores.shape[1]
    assert S & (S - 1) == 0, "S must be a power of two"
    mean_shift = S.bit_length() - 1

    softmax_out = np.zeros_like(scores, dtype=np.int8)
    for r in range(scores.shape[0]):
        score_row = scores[r]
        row_max = int(score_row.max())
        exp_input = np.maximum(score_row - row_max, -128).astype(np.int8)
        # ACAM EXP is a crossbar pass (identity weights), capacity I = min(R,C).
        exp_acam_output, _exp_passes = ref.convert_stream(exp_input, ref.MODE_EXP,
                                                    R, C)
        output_sum = int(exp_acam_output.view(np.uint8).astype(np.int64).sum())
        log_input = min(output_sum >> mean_shift, 127)
        # ACAM LOG is a crossbar pass on the int8 lane-sum value.
        log_output, _log_passes = ref.convert_stream(
            np.array([log_input], dtype=np.int8), ref.MODE_LOG, R, C)
        softmax_out[r] = np.clip(score_row - row_max - int(log_output[0]),
                                 -128, 127).astype(np.int8)
    return softmax_out


def softmax_stage_values(scores: np.ndarray) -> dict:
    """Per-stage integer values of NL safe-softmax — staged oracle.

    Formula path (independent of the primitive behavior model used by
    `v2/sim/softmax_sim.py`): every stage is computed in closed form from the
    operator definition, so the simulator's primitive-derived stage values can
    be compared bit-exactly.

    Returns dict of stages (row order, no geometry):
      row_max            int8  [S]     row maximum
      exp_input          int8  [S, S]  max(scores - row_max, -128)
      exp_crossbar_y     int32 [S, S]  eye-identity crossbar y (pre-ACAM; == exp_input)
      exp_acam_output    int8  [S, S]  ACAM_EXP output
      output_sum         int64 [S]     unsigned exp-output row sum
      log_input          int8  [S]     min(output_sum >> log2 S, 127)
      log_output         int8  [S]     ACAM_LOG(log_input) (trunc8(log_input - 1))
      softmax_out        int8  [S, S]  clamp(scores - row_max - log_output, -128, 127)
    """
    scores = np.asarray(scores, dtype=np.int32)
    assert scores.ndim == 2 and scores.shape[0] == scores.shape[1], \
        "scores must be [S, S]"
    S = scores.shape[1]
    assert S & (S - 1) == 0, "S must be a power of two"
    mean_shift = S.bit_length() - 1

    row_max = scores.max(axis=1).astype(np.int8)
    exp_input = np.maximum(scores - row_max[:, None], -128).astype(np.int8)
    exp_crossbar_y = exp_input.astype(np.int32)      # eye: crossbar y == input
    exp_acam_output = ref.acam_transform(exp_input, ref.MODE_EXP)
    output_sum = exp_acam_output.view(np.uint8).astype(np.int64).sum(axis=1)
    log_input = np.minimum(output_sum >> mean_shift, 127).astype(np.int8)
    log_output = ref.acam_transform(log_input.astype(np.int32),
                                    ref.MODE_LOG).astype(np.int8)
    softmax_out = np.clip(scores - row_max[:, None] - log_output[:, None],
                          -128, 127).astype(np.int8)
    return {"row_max": row_max, "exp_input": exp_input,
            "exp_crossbar_y": exp_crossbar_y,
            "exp_acam_output": exp_acam_output,
            "output_sum": output_sum, "log_input": log_input,
            "log_output": log_output, "softmax_out": softmax_out}


def packed_pass_counts(S: int, R: int = 256, C: int = 256) -> tuple[int, int]:
    """Schedule pass counts for the packed-window softmax structure.

    EXP stream: the S x S exp_input matrix read row-major (S^2 elements)
                -> ceil(S^2 / I) identity passes;
    LOG stream: the S row sums (lq), read in row order
                -> ceil(S / I) identity passes.
    I = min(R, C) is the identity-pass capacity (F5/F6); windows are packed
    stride-I with the final window zero-padded and its padding discarded.
    """
    I = min(R, C)
    return -(-(S * S) // I), -(-S // I)


# ---------------------------------------------------------------------------
# Self-test — run:  python3 softmax_ref.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # ── Hand-computed S=2 rows ─────────────────────────────────────────────
    # row [0,0]: m=0, d=[0,0], eb=[1,1], s=2, lq=2>>1=1, ls=0 -> [0,0]
    # row [1,3]: m=3, d=[-2,0], eb=[1,1], s=2, lq=1, ls=0 -> [-2,0]
    softmax_out = softmax_pass_view(np.array([[0, 0], [1, 3]], dtype=np.int8))
    assert softmax_out.dtype == np.int8 and softmax_out.shape == (2, 2)
    assert np.array_equal(softmax_out,
                          np.array([[0, 0], [-2, 0]], dtype=np.int8)), \
        softmax_out

    # row [10,14]: m=14, d=[-4,0], eb=[5,1], s=6, lq=3, ls=2 -> [-6,-2]
    softmax_out = softmax_pass_view(np.array([[10, 14], [10, 14]],
                                             dtype=np.int8))
    assert np.array_equal(softmax_out,
                          np.array([[-6, -2], [-6, -2]], dtype=np.int8)), \
        softmax_out

    # ── Signed extremes: d clamped at -128, output clamped to int8 ─────────
    # d=-128 -> eb = 8065 & 0xFF = 129 (unsigned); s=130 -> lq=65, ls=64;
    # out = clamp([-128-127-64, 127-127-64]) = [-128, -64]
    softmax_out = softmax_pass_view(np.array([[-128, 127], [-128, 127]],
                                             dtype=np.int8))
    assert np.array_equal(
        softmax_out, np.array([[-128, -64], [-128, -64]], dtype=np.int8)), \
        softmax_out

    # ── All-zero S=128: eb=1, s=128, lq=128>>7=1, ls=0 -> all zeros ────────
    zeros = softmax_pass_view(np.zeros((128, 128), dtype=np.int8))
    assert zeros.shape == (128, 128) and (zeros == 0).all()

    # ── Independent structural reference on a random S=128 matrix ──────────
    rng = np.random.default_rng(5)
    scores = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)

    def study_formula_reference(a: np.ndarray) -> np.ndarray:
        o = np.zeros_like(a, dtype=np.int8)
        shift = a.shape[1].bit_length() - 1
        for r in range(a.shape[0]):
            score_row = [int(v) for v in a[r]]
            row_max = max(score_row)
            output_sum = 0
            for v in score_row:
                exp_input = max(v - row_max, -128)
                output_sum += (1 + exp_input + (exp_input * exp_input) // 2) & 0xFF
            log_output = min(output_sum >> shift, 127) - 1
            for j, v in enumerate(score_row):
                o[r, j] = max(-128, min(127, v - row_max - log_output))
        return o

    assert np.array_equal(softmax_pass_view(scores),
                          study_formula_reference(scores)), \
        "structural mismatch"

    # ── Pass view == elementwise dual view (F5-F7 geometry invariance) ─────
    assert np.array_equal(softmax_pass_view(scores),
                          softmax_elementwise_view(scores)), \
        "pass view != elementwise view"
    # L > I: R=C=8 forces 16 EXP passes per row (S=128); values unchanged.
    assert np.array_equal(softmax_pass_view(scores, R=8, C=8),
                          softmax_elementwise_view(scores)), \
        "geometry changed values"

    # ── Staged oracle: every stage, hand-computed (S=2) ────────────────────
    # row [0,0]:  m=0, d=[0,0],  y=[0,0],   eb=[1,1],   s=2, lq=1, ls=0
    # row [1,3]:  m=3, d=[-2,0], y=[-2,0],  eb=[1,1],   s=2, lq=1, ls=0
    stages = softmax_stage_values(np.array([[0, 0], [1, 3]], dtype=np.int8))
    for key, want in (
        ("row_max", [0, 3]),
        ("exp_input", [[0, 0], [-2, 0]]),
        ("exp_crossbar_y", [[0, 0], [-2, 0]]),
        ("exp_acam_output", [[1, 1], [1, 1]]),
        ("output_sum", [2, 2]),
        ("log_input", [1, 1]),
        ("log_output", [0, 0]),
        ("softmax_out", [[0, 0], [-2, 0]]),
    ):
        got = stages[key]
        want_arr = np.array(want, dtype=got.dtype)
        assert got.shape == want_arr.shape, (key, got.shape, want_arr.shape)
        assert np.array_equal(got, want_arr), (key, got, want_arr)

    # ── Staged oracle consistency with the pass/dual views (random S=128) ──
    stages = softmax_stage_values(scores)
    assert (stages["exp_crossbar_y"] == stages["exp_input"].astype(np.int32)) \
        .all(), "eye y != exp_input"
    assert np.array_equal(
        stages["exp_acam_output"],
        ref.acam_transform(stages["exp_input"], ref.MODE_EXP)), \
        "exp_acam_output != acam_transform(exp_input, EXP)"
    assert np.array_equal(stages["output_sum"],
                          stages["exp_acam_output"].view(np.uint8)
                          .astype(np.int64).sum(axis=1)), \
        "output_sum mismatch"
    assert np.array_equal(stages["softmax_out"], softmax_pass_view(scores)), \
        "staged softmax_out != pass view"
    assert np.array_equal(stages["softmax_out"],
                          softmax_elementwise_view(scores)), \
        "staged softmax_out != elementwise view"

    # ── Packed-window pass counts (schedule formula, structure-level) ──────
    assert packed_pass_counts(128, 256, 256) == (64, 1)
    assert packed_pass_counts(256, 256, 256) == (256, 1)
    assert packed_pass_counts(512, 256, 256) == (1024, 2)
    assert packed_pass_counts(128, 8, 8) == (2048, 16)

    print("softmax_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
