#!/usr/bin/env python3
"""gemm_ref.py — NumPy composition reference for the GEMM array (v2 clean-room).

Role
----
Exact value reference for the Stage-2 GEMM array, transcribed from the FROZEN
charter `v2/spec/gemm.md` (**v0.3 FROZEN 2026-09-17**). The array is a
composition of already-certified NL-DPE primitives, so this file reuses
`nldpe_ref` per tile and models only the composition:

    tile (v,h)   : primitive F1/F2 exact integer MAC   -> y_vh    int32 [M,C]
    tile ACAM    : primitive F3 REGULAR then `trunc8`  -> out8_vh int8  [M,C]
    reduce (G5)  : S[m,n] = sum_v sign_extend(out8_vh) -> S       int32 [M,N]
    serializer   : out8 = trunc8(S)  (low byte)        -> out8    int8  [M,N]

v0.3: there is **no ACAM after the reduction** — every tile runs REGULAR and
the lane serializer takes the low byte of the reduced sum. Because
`sign_extend(trunc8(y)) == y (mod 256)`, this is *exactly* the low byte of the
true partial sum: `out8 = trunc8(sum_v y_v)` for every case (exactness
theorem, charter §6). Nonlinear forms (ACTIVATION/EXP/LOG) are out of scope
for this array.

What this models (and what it does not)
---------------------------------------
* Models: exact integer arithmetic of the composed datapath, the int8 tile
  quantization, zero padding, and the column mapping `n = h*C + c`
  (A5, §1.1).
* The self-test also asserts the **end-to-end single-matmul identity**
  `out8 == trunc8(X @ W)` (one matmul, no tiling knowledge) as an extra
  witness of the padding/slicing/reduction logic (charter §6 exactness).
* Does NOT model: timing/scheduling (that is `v2/sim/gemm_sim.py`), analog
  noise, upstream quantization, or any nonlinear output form.

Conventions
-----------
* W : np.int8 [K, N];  X : np.int8 [M, K].
* S : np.int32 [M, N] (|S| <= 128*V, int32-exact for supported V); out8 : int8.
* Every step is exact integer (no floats, no rounding).

Charter sections implemented here:
  §1.1 W/X mapping; §6 F1-F4 + exactness theorem; G4/G5/G6; A5 padding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_ref as ref  # noqa: E402

MODE_REGULAR = 0


def derive_vh(K: int, N: int, R: int, C: int) -> tuple[int, int]:
    """V = ceil(K/R) K-tiles, H = ceil(N/C) N-tiles (charter §1)."""
    return -(-K // R), -(-N // C)


def compute_gemm(W: np.ndarray, X: np.ndarray, R: int, C: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Charter §6 F1-F4 — array values, int8 in / int8 out (v0.3).

    Returns (S, out8):
      S    : np.int32 [M, N]  — wide partial after the byte-tree reduction
      out8 : np.int8  [M, N]  — array output (low byte of S; no ACAM after
                                reduction)
    """
    W = np.asarray(W, dtype=np.int8)
    X = np.asarray(X, dtype=np.int8)
    assert W.ndim == 2 and X.ndim == 2 and X.shape[1] == W.shape[0], \
        "need W [K,N] and X [M,K]"
    K, N = W.shape
    M = X.shape[0]
    V, H = derive_vh(K, N, R, C)

    # A5: zero padding to R/C multiples (rows/cols beyond K/N are zero).
    Wp = np.zeros((V * R, H * C), dtype=np.int8)
    Wp[:K, :N] = W
    Xp = np.zeros((M, V * R), dtype=np.int8)
    Xp[:, :K] = X

    # G4: tiles are REGULAR; G5: reduce the formal int8 tile outputs.
    S = np.zeros((M, H * C), dtype=np.int64)
    for v in range(V):
        Xv = Xp[:, v * R:(v + 1) * R]
        for h in range(H):
            Wt = Wp[v * R:(v + 1) * R, h * C:(h + 1) * C]
            y_vh = ref.compute_y(Wt, Xv)                    # int32 [M,C] (F1)
            out8_vh = ref.acam_transform(y_vh, MODE_REGULAR)  # int8 [M,C] (F2)
            S[:, h * C:(h + 1) * C] += out8_vh.astype(np.int64)

    assert S.min() >= -(2 ** 31) and S.max() <= 2 ** 31 - 1, "S exceeds int32 (B8)"
    S = S.astype(np.int32)[:, :N]
    out8 = ref.trunc8(S)                                    # serializer (F4)
    return S, out8


# ---------------------------------------------------------------------------
# Self-test — run:  python3 gemm_ref.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # ── Independent structural composition (explicit per-tile loops) ────────
    def structural(W: np.ndarray, X: np.ndarray, R: int, C: int
                   ) -> tuple[np.ndarray, np.ndarray]:
        K, N = W.shape
        M = X.shape[0]
        V = -(-K // R)
        H = -(-N // C)
        Wp = np.zeros((V * R, H * C), dtype=np.int8)
        Wp[:K, :N] = W
        Xp = np.zeros((M, V * R), dtype=np.int8)
        Xp[:, :K] = X
        S = np.zeros((M, H * C), dtype=np.int64)
        for v in range(V):
            for h in range(H):
                y = ref.compute_y(Wp[v * R:(v + 1) * R, h * C:(h + 1) * C],
                                  Xp[:, v * R:(v + 1) * R])
                o = ref.acam_transform(y, MODE_REGULAR)
                for m in range(M):
                    for c in range(C):
                        S[m, h * C + c] += int(o[m, c])
        S32 = S.astype(np.int32)[:, :N]
        return S32, ref.trunc8(S32)

    def assert_end_to_end(W, X, S, out8):
        """v0.3 exactness, end-to-end: ONE matmul, no tiling knowledge at all.

        `wide = X @ W` (int64) must agree with the composition in the low
        byte: S ≡ wide (mod 256) and out8 == trunc8(wide). This is an
        independent witness of the padding/slicing/reduction logic — it never
        sees R, C, V or H. (Identity holds while |sum y_v| < 2^31, the
        supported domain; beyond that trunc8's clamp vs the array's
        unclamped S could differ, charter B8/P25.)
        """
        wide = X.astype(np.int64) @ W.astype(np.int64)
        assert ((S.astype(np.int64) - wide) % 256 == 0).all(), \
            "end-to-end: S not congruent to X@W (mod 256)"
        assert np.array_equal(out8, ref.trunc8(wide)), \
            "end-to-end: out8 != trunc8(X@W)"

    # ── 1. Padded geometry (V=2, H=2), vectorized == structural ─────────────
    W = rng.integers(-128, 128, size=(5, 4), dtype=np.int8)   # K=5, N=4
    X = rng.integers(-128, 128, size=(2, 5), dtype=np.int8)   # M=2
    S, out = compute_gemm(W, X, 4, 3)                          # R=4, C=3
    Ss, os_ = structural(W, X, 4, 3)
    assert S.shape == (2, 4) and out.shape == (2, 4)
    assert np.array_equal(S, Ss), "V=2/H=2 structural mismatch"
    assert np.array_equal(out, os_), "V=2/H=2 output mismatch"
    assert_end_to_end(W, X, S, out)

    # ── 2. I6: V=1,H=1 reduces to the primitive REGULAR exactly ─────────────
    W1 = rng.integers(-128, 128, size=(8, 6), dtype=np.int8)
    X1 = rng.integers(-128, 128, size=(3, 8), dtype=np.int8)
    tile = ref.acam_transform(ref.compute_y(W1, X1), MODE_REGULAR)
    S1, out1 = compute_gemm(W1, X1, 8, 6)
    assert np.array_equal(S1, tile.astype(np.int64)), "I6 S mismatch"
    assert np.array_equal(out1, tile), "I6 out8 mismatch"
    assert_end_to_end(W1, X1, S1, out1)

    # ── 3. I6 with H>1: each column block equals its tile's REGULAR output ──
    W2 = rng.integers(-128, 128, size=(6, 8), dtype=np.int8)
    X2 = rng.integers(-128, 128, size=(2, 6), dtype=np.int8)
    S2, out2 = compute_gemm(W2, X2, 6, 4)                      # V=1, H=2
    for h in range(2):
        t = ref.acam_transform(ref.compute_y(W2[:, h * 4:(h + 1) * 4], X2),
                               MODE_REGULAR)
        assert np.array_equal(out2[:, h * 4:(h + 1) * 4], t), f"H>1 block {h}"
    assert_end_to_end(W2, X2, S2, out2)

    # ── 4. I7 exactness theorem: out8 == trunc8(wide sum), S ≡ sum (mod 256) ─
    W3 = rng.integers(-128, 128, size=(9, 5), dtype=np.int8)   # K=9, N=5
    X3 = rng.integers(-128, 128, size=(2, 9), dtype=np.int8)
    R, Cc = 4, 5                                                # V=3, H=1
    S3, out3 = compute_gemm(W3, X3, R, Cc)
    Wp = np.zeros((3 * R, Cc), dtype=np.int8)
    Wp[:9, :5] = W3
    Xp = np.zeros((2, 3 * R), dtype=np.int8)
    Xp[:, :9] = X3
    ysum = np.zeros((2, 5), dtype=np.int64)
    for v in range(3):
        ysum += ref.compute_y(Wp[v * R:(v + 1) * R], Xp[:, v * R:(v + 1) * R])
    assert ((S3.astype(np.int64) - ysum) % 256 == 0).all(), "I7 mod-256 failed"
    assert np.array_equal(out3, ref.trunc8(ysum)), "I7 trunc8(wide sum) failed"
    assert_end_to_end(W3, X3, S3, out3)

    # ── 5. Byte-tree exactness on a sign-crossing case ──────────────────────
    # tile0 y = 127+1 = 128 -> out8 = -128 ; tile1 y = -2 -> out8 = -2
    # S = -130 ; wide sum = +126. Both trunc8 to 126 -> the byte tree is exact.
    Wc = np.array([[127], [1], [-2], [0]], dtype=np.int8)      # K=4, N=1, R=2
    Xc = np.array([[1, 1, 1, 0]], dtype=np.int8)
    Sc, outc = compute_gemm(Wc, Xc, 2, 1)
    assert Sc[0, 0] == -130 and outc[0, 0] == 126, "exactness theorem failed"
    assert outc[0, 0] == ref.trunc8(np.array([126], dtype=np.int64))[0]
    assert_end_to_end(Wc, Xc, Sc, outc)

    # ── 6. Determinism across random geometries (vectorized == structural) ──
    for _ in range(20):
        R = int(rng.integers(1, 9))
        C = int(rng.integers(1, 9))
        K = int(rng.integers(1, 17))
        N = int(rng.integers(1, 17))
        M = int(rng.integers(1, 4))
        W = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
        X = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
        S, out = compute_gemm(W, X, R, C)
        Ss, os_ = structural(W, X, R, C)
        assert np.array_equal(S, Ss) and np.array_equal(out, os_), \
            f"random geometry mismatch R={R} C={C} K={K} N={N} M={M}"
        assert_end_to_end(W, X, S, out)

    print("gemm_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
