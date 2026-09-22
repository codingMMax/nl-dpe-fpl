#!/usr/bin/env python3
"""softmax_ref.py — NumPy value reference for the NL safe-softmax operator.

Role
----
Plain math, no state, no time. The behavior simulator (`v2/sim/softmax_sim.py`)
must agree with this file bit-exactly on values; the cycle contract lives in
that simulator (`softmax_cycle_model`), not here.

NL log-domain safe softmax (per row, integer; S must be a power of two)
-----------------------------------------------------------------------
    m     = max_j x[j]
    d     = max(x[j] - m, -128)                  lower clamp (log-domain headroom)
    eb    = ACAM_EXP(d)                          MODE_EXP (P24), int8 bytes
    s     = sum_j eb                             unsigned byte sum (CLB tree)
    lq    = min(s >> log2(S), 127)               mean-quantized lane sum
    ls    = ACAM_LOG(lq)                         MODE_LOG (P24): trunc8(lq - 1)
    out[j] = clamp(x[j] - m - ls, -128, 127)     int8, log domain

Transcribed from the locked NL contract of `softmax_study/SOFTMAX_STUDY.md`
(§1 mapping, §4 formulas) and `softmax_study/run_softmax_smoke.py::oracle_nl`,
re-expressed with the v2 ACAM forms (`v2/spec/dpe_nldpe.md` §6 F3 / P16 / P24).
For d in [-128, 0] the v2 EXP form and the study's `(1 + v + v^2//2) & 0xFF`
are bit-identical (f(d) in [0, 8065], no int32 clamp reachable).

Notes
-----
* Operator pass layer (§6 F5-F8): ACAM never runs standalone — the EXP of a
  row slice and the LOG of the lane sums are identity-crossbar passes with
  capacity I = min(R,C); pass counts are schedule-owned. `safe_softmax_nl`
  is the pass view; `elementwise_softmax_nl` is the dual view and the
  self-test asserts bit-equality (the packing/geometry invariance).
* Output is LOG-domain (log p_i approximation), consumed directly by the
  downstream S*V DIMM input path (`mac_sv`); it is not comparable with a
  linear-domain (Azure-Lily) softmax.
* The model is the contract: ACAM exp/log are declared approximations, so
  accuracy vs float softmax is out of scope (cf. spec A13).
* `lq` uses an arithmetic right shift by log2(S) (mean) and caps at 127
  because the shared log DPE consumes an int8-domain value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the sibling oracle module importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_ref as ref  # noqa: E402


def elementwise_softmax_nl(scores: np.ndarray) -> np.ndarray:
    """Dual (elementwise, geometry-free) view of NL safe-softmax — int8 [S, S].

    Kept as the independent witness for the pass view; do not consume it from
    the behavior sim (use `safe_softmax_nl`).
    """
    x = np.asarray(scores, dtype=np.int32)
    assert x.ndim == 2 and x.shape[0] == x.shape[1], "scores must be [S, S]"
    S = x.shape[1]
    assert S & (S - 1) == 0, "S must be a power of two"
    log_shift = S.bit_length() - 1

    out = np.zeros_like(x, dtype=np.int8)
    for r in range(x.shape[0]):
        row = x[r]
        m = int(row.max())
        d = np.maximum(row - m, -128)
        eb = ref.acam_transform(d, ref.MODE_EXP).view(np.uint8).astype(np.int64)
        s = int(eb.sum())
        lq = min(s >> log_shift, 127)
        ls = int(ref.acam_transform(np.array([lq], dtype=np.int32),
                                    ref.MODE_LOG)[0])
        out[r] = np.clip(row - m - ls, -128, 127).astype(np.int8)
    return out


def safe_softmax_nl(scores: np.ndarray, R: int = 256, C: int = 256
                    ) -> np.ndarray:
    """NL safe-softmax over an S x S score matrix (pass view). int8 [S, S].

    scores : int8 (or integer) [S, S], S a power of two. Log-domain output.
    R, C   : crossbar geometry for the EXP/LOG identity passes (F5/F6).
    """
    x = np.asarray(scores, dtype=np.int32)
    assert x.ndim == 2 and x.shape[0] == x.shape[1], "scores must be [S, S]"
    S = x.shape[1]
    assert S & (S - 1) == 0, "S must be a power of two"
    log_shift = S.bit_length() - 1

    out = np.zeros_like(x, dtype=np.int8)
    for r in range(x.shape[0]):
        row = x[r]
        m = int(row.max())
        d = np.maximum(row - m, -128).astype(np.int8)
        # ACAM EXP is a crossbar pass (identity weights), capacity I = min(R,C).
        eb, _p_exp = ref.convert_stream(d, ref.MODE_EXP, R, C)
        s = int(eb.view(np.uint8).astype(np.int64).sum())
        lq = min(s >> log_shift, 127)
        # ACAM LOG is a crossbar pass on the int8 lane-sum value.
        ls, _p_log = ref.convert_stream(np.array([lq], dtype=np.int8),
                                        ref.MODE_LOG, R, C)
        out[r] = np.clip(row - m - int(ls[0]), -128, 127).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Self-test — run:  python3 softmax_ref.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # ── Hand-computed S=2 rows ─────────────────────────────────────────────
    # row [0,0]: m=0, d=[0,0], eb=[1,1], s=2, lq=2>>1=1, ls=0 -> [0,0]
    # row [1,3]: m=3, d=[-2,0], eb=[1,1], s=2, lq=1, ls=0 -> [-2,0]
    out = safe_softmax_nl(np.array([[0, 0], [1, 3]], dtype=np.int8))
    assert out.dtype == np.int8 and out.shape == (2, 2)
    assert np.array_equal(out, np.array([[0, 0], [-2, 0]], dtype=np.int8)), out

    # row [10,14]: m=14, d=[-4,0], eb=[5,1], s=6, lq=3, ls=2 -> [-6,-2]
    out = safe_softmax_nl(np.array([[10, 14], [10, 14]], dtype=np.int8))
    assert np.array_equal(out, np.array([[-6, -2], [-6, -2]], dtype=np.int8)), out

    # ── Signed extremes: d clamped at -128, output clamped to int8 ─────────
    # d=-128 -> eb = 8065 & 0xFF = 129 (unsigned); s=130 -> lq=65, ls=64;
    # out = clamp([-128-127-64, 127-127-64]) = [-128, -64]
    out = safe_softmax_nl(np.array([[-128, 127], [-128, 127]], dtype=np.int8))
    assert np.array_equal(out,
                          np.array([[-128, -64], [-128, -64]], dtype=np.int8)), out

    # ── All-zero S=128: eb=1, s=128, lq=128>>7=1, ls=0 -> all zeros ────────
    zeros = safe_softmax_nl(np.zeros((128, 128), dtype=np.int8))
    assert zeros.shape == (128, 128) and (zeros == 0).all()

    # ── Independent structural reference on a random S=128 matrix ──────────
    rng = np.random.default_rng(5)
    x = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)

    def nl_ref(a: np.ndarray) -> np.ndarray:
        o = np.zeros_like(a, dtype=np.int8)
        shift = a.shape[1].bit_length() - 1
        for r in range(a.shape[0]):
            row = [int(v) for v in a[r]]
            m = max(row)
            s = 0
            for v in row:
                d = max(v - m, -128)
                s += (1 + d + (d * d) // 2) & 0xFF
            ls = min(s >> shift, 127) - 1
            for j, v in enumerate(row):
                o[r, j] = max(-128, min(127, v - m - ls))
        return o

    assert np.array_equal(safe_softmax_nl(x), nl_ref(x)), "structural mismatch"

    # ── Pass view == elementwise dual view (F5-F7 geometry invariance) ─────
    assert np.array_equal(safe_softmax_nl(x), elementwise_softmax_nl(x)), \
        "pass view != elementwise view"
    # L > I: R=C=8 forces 16 EXP passes per row (S=128); values unchanged.
    assert np.array_equal(safe_softmax_nl(x, R=8, C=8),
                          elementwise_softmax_nl(x)), "geometry changed values"

    print("softmax_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
