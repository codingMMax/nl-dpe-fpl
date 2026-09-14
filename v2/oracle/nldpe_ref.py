#!/usr/bin/env python3
"""nldpe_ref.py — NumPy value reference for the NL-DPE primitive (v2 clean-room).

Role
----
Independent numerical reference, transcribed from the spec
`v2/spec/dpe_nldpe.md` (**v2.0 integer dataflow, 2026-09-14**). Plain math, no
state, no time. The OOP behavior simulator (`v2/sim/nldpe_sim.py`) must agree
with this file bit-exactly on values; the legacy oracle (`rtl_flow/smoke/
oracles/nldpe_mac_oracle.py`) is a structural reference only.

Integer arithmetic is EXACT: there is no rounding and no summation-order
contract (P22). The simulator must still mirror the hardware's per-slice
structure, but any correct integer evaluation is bit-identical.

Conventions
-----------
* W : np.int8 [R, C] — weight matrix (spec §2 T4), stationary.
* X : np.int8 [M, R] — M activation vectors, one per pass (§2 T1).
* y : np.int32 [M, C] — exact crossbar output, §6 F2.
* Modes are integers 0..3 matching nl_dpe_control[1:0] (§4.1):
  0=REGULAR, 1=ACTIVATION, 2=EXP, 3=LOG (integer forms, P24).
* ACAM output: np.int8 in column order (§2 T2). The 40-bit stream exposes the
  same 8 bits as bytes; unsigned representation appears only at serialization
  (packers, .mem files, TB byte compares) — flattening a pass's row (c-order)
  IS the §4.5 stream order.
* Overflow policy (A17/P25): |y| <= R * 2^14, so int32 is exact for
  R <= 131072; EXP uses a wider intermediate.

Weights are NOT streamed as data (P23): programming is a one-time WEIGHT-strobe
load handled at TB/system level; this reference takes W directly as an array.

Spec sections implemented here (transcribe, don't improvise):
  §6 F1/F2 — compute_y (exact integer MAC)
  §6 F3   — trunc8, acam_transform (integer mode forms)
  §4.2    — pack_weight_stream (stimulus side; WEIGHT strobe, P23)
  §4.3    — pack_act_stream
"""

from __future__ import annotations

import numpy as np

# Mode codes (§4.1 / §6 F3) — keep in one place.
MODE_REGULAR = 0     # nl_dpe_control = 2'b00
MODE_ACTIVATION = 1  # nl_dpe_control = 2'b01
MODE_EXP = 2         # nl_dpe_control = 2'b10  (integer EXP, P24)
MODE_LOG = 3         # nl_dpe_control = 2'b11  (integer LOG, P24)


def compute_y(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """§6 F1/F2 — exact integer MAC.

    W : int8 [R, C]; X : int8 [M, R]. Returns y : int32 [M, C] with

        y[m, c] = Σ_r W[r, c] · x[m, r]        (exact, int64 intermediate)

    F2's bit-serial partial-shift structure
        y = Σ_{b=0..P-2} 2^b·s_b  −  2^(P-1)·s_{P-1}
    is mathematically identical to this direct sum because integer arithmetic
    is exact (P22): no rounding, no order dependence.
    """
    W64 = np.asarray(W, dtype=np.int64)
    X64 = np.asarray(X, dtype=np.int64)
    assert W64.ndim == 2, "W must be [R, C]"
    assert X64.ndim == 2 and X64.shape[1] == W64.shape[0], "X must be [M, R]"
    return (X64 @ W64).astype(np.int32)


def trunc8(z: np.ndarray) -> np.ndarray:
    """§6 F3 — ACAM output rule (P16): clamp to int32, keep the low byte.

    z : integer array. Truncation toward zero is the identity for integers;
    the value is clamped to [-2^31, 2^31-1] and the low byte is reinterpreted
    as signed int8 (two's-complement wrap — no saturation at 8 bits).
    """
    t = np.asarray(z, dtype=np.int64)
    t = np.clip(t, -(2**31), 2**31 - 1)
    return (t & 0xFF).astype(np.uint8).view(np.int8)


def acam_transform(y: np.ndarray, mode: int) -> np.ndarray:
    """§6 F3 — ACAM stage: int32 in, int8 out (T2), 1 cycle, parallel.

    All modes: integer functional form first, then `trunc8` (P16/P24).

        mode 0 REGULAR    : f(v) = v
        mode 1 ACTIVATION : f(v) = relu(v) = v if v > 0 else 0
        mode 2 EXP        : f(v) = 1 + v + floor(v²/2)   (exact wide intermediate)
        mode 3 LOG        : f(v) = v - 1

    y : int32 array [.., C]. Returns int8 array, same shape. Raises ValueError
    for mode not in {0, 1, 2, 3}.

    EXP note: v² >= 0, so floor(v²/2) = truncation; only `v mod 512` affects
    the output byte after trunc8.
    """
    y = np.asarray(y)
    if mode == MODE_REGULAR:
        return trunc8(y)
    if mode == MODE_ACTIVATION:
        return trunc8(np.maximum(y, 0))
    if mode == MODE_EXP:
        y64 = y.astype(np.int64)
        return trunc8(1 + y64 + (y64 * y64) // 2)
    if mode == MODE_LOG:
        return trunc8(y - 1)
    if mode not in (MODE_REGULAR, MODE_ACTIVATION, MODE_EXP, MODE_LOG):
        raise ValueError(f"invalid ACAM mode: {mode}")


def pack_act_stream(x_m: np.ndarray) -> list[int]:
    """§4.3 — one pass's activation byte stream packed into 40-bit words.

    Byte #j (j = 0 .. R-1) carries x_m[j] (activation for crossbar row j).
    5 bytes per word: word bit `8i+7 : 8i` = byte (5t + i), i = 0..4.
    LOAD_CYC = ceil(R*8/40);  R=256 -> 52 words.
    Serialization boundary: int8 values are masked to unsigned bytes here
    (& 0xFF) — the only place signedness is dropped.

    Returns a list of Python ints (0 .. 2**40-1), one per word, in stream
    order. The .mem hex file (Stage 1.5) is one 10-hex-digit line per word.

    Weights are programmed by strobe — see `pack_weight_stream` (P23).
    """
    b = np.ascontiguousarray(x_m).view(np.uint8)      # (R,)
    n = -(-b.size // 5)                               # number of words
    buf = np.concatenate([b, np.zeros(n*5 - b.size, dtype=np.uint8)]).reshape(n, 5)
    shifts = np.array([0, 8, 16, 24, 32], dtype=np.uint64)     # (5,)
    words = (buf.astype(np.uint64) << shifts).sum(axis=1)      # (n,)
    return [int(w) for w in words]


def pack_weight_stream(W: np.ndarray) -> list[int]:
    """§4.2 (P23) — int8 weights packed into 40-bit WEIGHT-strobe words.

    One int8 weight per strobe cycle on `data_in[7:0]` (upper bits zero),
    row-major row-outer: word #k (k = 0 .. R*C-1) carries W[k // C, k % C] as
    an unsigned byte. WR_CYC = R*C, one-time, excluded from per-pass formulas.

    Returns a list of Python ints (0 .. 255). The .mem hex file (Stage 1.5) is
    one 10-hex-digit line per word.
    """
    Wb = np.asarray(W, dtype=np.int8)
    assert Wb.ndim == 2, "W must be [R, C]"
    return [int(b) for b in np.ascontiguousarray(Wb).reshape(-1).view(np.uint8)]


# ---------------------------------------------------------------------------
# Self-test — run:  python3 nldpe_ref.py
# All asserts must pass before the simulator work starts.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    R, C, M = 8, 16, 3  # small shapes; math is size-independent

    # ── Independent structural F2 reference (per-slice partial-shift, P22) ──
    def f2_ref(W: np.ndarray, X: np.ndarray) -> np.ndarray:
        rw, cw = W.shape
        out = np.zeros((X.shape[0], cw), dtype=np.int64)
        for m in range(X.shape[0]):
            for c in range(cw):
                y = 0
                for b in range(8):
                    s = 0
                    for r in range(rw):
                        if (int(X[m, r]) >> b) & 1:
                            s += int(W[r, c])
                    y += s << b if b < 7 else -(s << b)   # MSB subtract (P2)
                out[m, c] = y
        return out.astype(np.int32)

    # ── int8 weights: both signs, a zero row, signed extremes ──
    W = rng.integers(-128, 128, size=(R, C), dtype=np.int8)
    W[0, :] = 0
    W[1, 0] = -128
    W[1, 1] = 127
    X = rng.integers(-128, 128, size=(M, R), dtype=np.int8)

    # I7 identity check: y = x exactly.
    I = np.eye(R, dtype=np.int8)
    y_id = compute_y(I, X)
    assert y_id.shape == (M, R) and y_id.dtype == np.int32, f"shape:{y_id.shape}, type:{y_id.dtype}"
    assert (y_id == X.astype(np.int32)).all(), "identity-weight MAC failed"

    # F2 bit-exact vs the explicit structural reference.
    y = compute_y(W, X)
    assert y.dtype == np.int32
    assert np.array_equal(y, f2_ref(W, X)), "F2 structural mismatch"

    # ── F3 trunc8: low byte, no 8-bit saturation; int32 clamp boundary ──
    z = np.array([130, -130, 3, -3], dtype=np.int64)
    assert trunc8(z).dtype == np.int8
    assert (trunc8(z) == np.array([-126, 126, 3, -3], dtype=np.int8)).all()
    # clamp: 2^31 -> 2^31-1 -> 0xFF -> -1 ; -2^31 -> 0x00 -> 0
    assert trunc8(np.array([2**31], dtype=np.int64))[0] == -1
    assert trunc8(np.array([-(2**31)], dtype=np.int64))[0] == 0

    # ── F3 modes (integer form first, trunc8 last; int8 out) ──
    t = np.array([[130, -130, 3, -3]], dtype=np.int64)
    assert (acam_transform(t, MODE_REGULAR)
            == np.array([[-126, 126, 3, -3]], dtype=np.int8)).all()

    t2 = np.array([[-3, 0, 2, 0]], dtype=np.int64)
    assert (acam_transform(t2, MODE_ACTIVATION)
            == np.array([[0, 0, 2, 0]], dtype=np.int8)).all()

    # EXP: 1 + v + v^2/2 ; v=2048 -> 2099201 -> low byte 1 ; v=0,-1 -> 1,0
    te = np.array([[0, 1, -1, 2048]], dtype=np.int64)
    assert (acam_transform(te, MODE_EXP)
            == np.array([[1, 2, 0, 1]], dtype=np.int8)).all()
    # EXP clamp path: huge v -> clamp to int32 max -> low byte 0xFF -> -1
    big = np.array([1 << 24], dtype=np.int64)
    assert acam_transform(big, MODE_EXP)[0] == -1

    # LOG: v - 1
    tl = np.array([[1, 3, -1, 2]], dtype=np.int64)
    assert (acam_transform(tl, MODE_LOG)
            == np.array([[0, 2, -2, 1]], dtype=np.int8)).all()

    # Invalid mode codes are a programming error.
    for bad in (-1, 4):
        try:
            acam_transform(t, bad)
            raise AssertionError(f"mode {bad} should raise ValueError")
        except ValueError:
            pass

    # ── §4.3 activation packer: word count + byte order ──
    words_x = pack_act_stream(X[0])
    assert len(words_x) == -(-R * 8 // 40), "LOAD_CYC mismatch"
    x_bytes = [(words_x[t] >> (8 * i)) & 0xFF for t in range(len(words_x)) for i in range(5)]
    assert bytes(x_bytes[:R]) == X[0].tobytes(), "act byte order wrong"

    # ── §4.2 weight packer: count, row-major order, byte round-trip ──
    words_w = pack_weight_stream(W)
    assert len(words_w) == R * C, "WR_CYC mismatch"
    assert all(0 <= w <= 255 for w in words_w), "weight word must be a byte"
    assert words_w[0] == int(W[0, 0].view(np.uint8)), "first word order wrong"
    assert words_w[-1] == int(W[R - 1, C - 1].view(np.uint8)), "last word order wrong"
    wt_rt = np.array(words_w, dtype=np.uint8).view(np.int8).reshape(R, C)
    assert np.array_equal(wt_rt, W), "weight byte round-trip failed"

    print("nldpe_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
