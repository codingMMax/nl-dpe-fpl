#!/usr/bin/env python3
"""nldpe_ref.py — NumPy value reference for the NL-DPE primitive (v2 clean-room).

Role
----
Independent numerical reference, transcribed from the spec
`v2/spec/dpe_nldpe.md` (**v1.1 amended 2026-09-12**). Plain math, no state,
no time. The OOP behavior simulator (`v2/sim/nldpe_sim.py`) must agree with
this file bit-exactly on values; the legacy oracle (`rtl_flow/smoke/oracles/
nldpe_mac_oracle.py`) is a structural reference only — it encodes the
superseded int8-weight semantics and is NOT a numerical witness under v1.1.

This file is deliberately dumb: if the simulator and this reference ever
disagree, the spec (§ references below) decides who is wrong.

Conventions
-----------
* W : np.float32 [R, C] — weight matrix (spec §2 T4), stationary.
* X : np.int8 [M, R] — M activation vectors, one per pass (§2 T1).
* y : np.float32 [M, C] — crossbar output, §6 F2 structural sequence.
* Modes are integers 0..3 matching nl_dpe_control[1:0] (§4.1):
  0=REGULAR, 1=ACTIVATION, 2=EXP, 3=LOG (all defined, P18).
* ACAM output: np.int8 in column order (§2 T2). The 40-bit stream exposes the
  same 8 bits as bytes; unsigned representation appears only at serialization
  (packers, .mem files, TB byte compares) — flattening a pass's row (c-order)
  IS the §4.5 stream order.
* fp32 discipline (A17): every operation is IEEE-754 binary32, RNE, no FMA,
  gradual underflow; accumulate with np.float32, never Python floats.

Weights are NOT streamed (P17): programming is a one-time WEIGHT-strobe load
handled at TB/system level; this reference takes W directly as an array.

Spec sections implemented here (transcribe, don't improvise):
  §6 F1/F2 — compute_y
  §6 F3   — trunc8, acam_transform
  §4.3    — pack_act_stream
"""

from __future__ import annotations

import numpy as np

# Mode codes (§4.1 / §6 F3) — keep in one place.
MODE_REGULAR = 0     # nl_dpe_control = 2'b00
MODE_ACTIVATION = 1  # nl_dpe_control = 2'b01
MODE_EXP = 2         # nl_dpe_control = 2'b10  (EXP_FN, §6 F3 / P18)
MODE_LOG = 3         # nl_dpe_control = 2'b11  (LOG_FN, §6 F3 / P18)


def compute_y(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """§6 F1/F2 — structural fp32 MAC (normative sequence).

    For each pass m and column c, F2 fixes the exact operation order:

        s := +0.0
        for r = 0 .. R-1:  s := fl32( s + ( bit_b(X[m,r]) ? W[r,c] : +0.0 ) )
        p_b[c] := s                                   (b = 0..P-1, P = 8)

        y := +0.0
        for b = 0 .. P-2:  y := fl32( y + 2**b * p_b[c] )
        y := fl32( y - 2**(P-1) * p_{P-1}[c] )        (2's-complement MSB, P2)

    The slice partial enters at its own significance (exact power-of-two
    scale); the accumulator does not shift (P20).

    Returns y : np.float32 [M, C]. Rounding: IEEE-754 binary32 RNE, no FMA
    (A17/P19). Bit b of an int8 activation is taken from its two's-complement
    representation (b = 7 is the sign bit).

    TODO(you): implement. Explicit loops with np.float32 accumulators are
    fine (small shapes in the self-test). Do NOT use `@`, np.sum, np.dot, or
    Python floats — their summation order is not F2 and will differ by >=1 ulp.
    """
    # return X.astype(np.int8) @ W.astype(np.float32) 
    rows, cols = W.shape
    out = np.zeros((X.shape[0], cols))
    zero = np.float32(0.0)
    WIDTH = 8
    for m in range(X.shape[0]):
        for c in range(cols):
            parts = []
            # compute output at each column
            for bit in range(WIDTH):
                slice = X[m,c] >> bit & 1
                s = zero
                for r in range(rows):
                    tmp = W[r, c] if slice else zero
                    s = np.float32(s + tmp)
                parts.append(s) # bit-sliced partial sum at current column
        # reduce across bit-sliced partial sums
        acc = zero
        for bit in range(WIDTH):
            scale = 2 ** bit
            acc = np.float32(parts[bit] * scale + acc)
        out[m,c] = acc
    
    return out              

                

def trunc8(z: np.ndarray) -> np.ndarray:
    """§6 F3 — the ACAM output rule (P16).

        t    := truncate_toward_zero(z), clamped to [-2^31, 2^31-1]
        out8 := t[7:0]        (two's-complement low byte, wrap — no saturation)

    z : array of fp32. Returns int8 array, same shape (§2 T2: the ACAM output
    IS int8; the low byte is reinterpreted as signed). This is NOT a
    clamp/requantizer: only the low byte of the (int32-clamped) truncated
    value survives.

    TODO(you): implement. Hints: np.trunc for toward-zero; the clamp only
    matters for |z| >= 2^31 — do it in float64, because the fp32 literal
    2^31-1 rounds up to 2^31; finish with (& 0xFF) then a *bit
    reinterpretation* to int8 (.astype(np.uint8).view(np.int8)), not a value
    cast.
    """
    t = np.trunc(z)
    t = t.astype(np.float64)
    t = np.clip(t, -(2**31), 2**31-1)
    t = t.astype(np.int32)
    return (t & 0xff).astype(np.uint8).view(np.int8)



def acam_transform(y: np.ndarray, mode: int) -> np.ndarray:
    """§6 F3 — ACAM stage: fp32 in, int8 out (spec §2 T2), 1 cycle, parallel.

    All modes: evaluate the functional form in fp32 (normative order), then
    apply trunc8 (functional form FIRST, truncation LAST — P16).

        mode 0 REGULAR    : f(v) = v
        mode 1 ACTIVATION : f(v) = relu(v) = v if v > 0 else +0.0
        mode 2 EXP        : f(v) = fl32(1 + fl32(v + fl32(0.5*fl32(v*v))))
        mode 3 LOG        : f(v) = fl32(v - 1)

    y : fp32 array [.., C]. Returns int8 array, same shape (§2 T2). Raises
    ValueError for mode not in {0, 1, 2, 3}.

    TODO(you): implement all four modes with explicit np.float32 operation
    order (the EXP evaluation order is normative). Then trunc8.
    """
    if mode == MODE_REGULAR:
        return y.astype(np.int8)
    if mode == MODE_ACTIVATION:
        return np.maximum(0.0, y.astype(np.int8))
    if mode == MODE_EXP:
        return trunc8(np.exp(y))
    if mode == MODE_LOG:
        return trunc8(np.log(y))
        

def pack_act_stream(x_m: np.ndarray) -> list[int]:
    """§4.3 — one pass's activation byte stream packed into 40-bit words.

    Byte #j (j = 0 .. R-1) carries x_m[j] (activation for crossbar row j).
    5 bytes per word: word bit `8i+7 : 8i` = byte (5t + i), i = 0..4.
    LOAD_CYC = ceil(R*8/40);  R=256 -> 52 words.
    Serialization boundary: int8 values are masked to unsigned bytes here
    (& 0xFF) — the only place signedness is dropped.

    Returns a list of Python ints (0 .. 2**40-1), one per word, in stream
    order. The .mem hex file (Stage 1.5) is one 10-hex-digit line per word.

    TODO(you): implement (unchanged from v1.0; the weight packer is gone —
    weights are programmed by strobe, P17).
    """
    b = np.ascontiguousarray(x_m).view(np.uint8)      # (R,)
    n = -(-b.size // 5)                               # number of words
    buf = np.concatenate([b, np.zeros(n*5 - b.size, dtype=np.uint8)]).reshape(n, 5)
    shifts = np.array([0, 8, 16, 24, 32], dtype=np.uint64)     # (5,)
    words = (buf.astype(np.uint64) << shifts).sum(axis=1)      # (n,)
    return [int(w) for w in words]
                
              
    
# ---------------------------------------------------------------------------
# Self-test — run:  python3 nldpe_ref.py
# All asserts must pass before the simulator work starts.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    R, C, M = 8, 16, 3  # small shapes; math is size-independent

    # ── Independent F2 reference (explicit order, deliberately not vectorized) ──
    def f2_ref(W: np.ndarray, X: np.ndarray) -> np.ndarray:
        RW, CW = W.shape
        out = np.empty((X.shape[0], CW), dtype=np.float32)
        zero = np.float32(0.0)
        scales = [np.float32(2.0 ** b) for b in range(8)]   # exact powers of two
        for m in range(X.shape[0]):
            for c in range(CW):
                parts = []
                for b in range(8):
                    s = zero
                    for r in range(RW):
                        inc = W[r, c] if ((int(X[m, r]) >> b) & 1) else zero
                        s = np.float32(s + inc)
                    parts.append(s)
                acc = zero
                for b in range(7):
                    acc = np.float32(acc + parts[b] * scales[b])
                out[m, c] = np.float32(acc - parts[7] * scales[7])
        return out

    # ── fp32 weights: mixed exponents, both signs, a zero row, no NaN/Inf ──
    W = (rng.standard_normal((R, C)) * (2.0 ** rng.integers(-8, 9, size=(R, C)))).astype(np.float32)
    W[0, :] = np.float32(0.0)
    X = rng.integers(-128, 128, size=(M, R), dtype=np.int8)
    words_x = pack_act_stream(X[0])

    # F1/F2 identity check: y = x exactly (§8 I7).
    I = np.eye(R, dtype=np.float32)
    y_id = compute_y(I, X)
    assert y_id.shape == (M, R) and y_id.dtype == np.float32
    assert (y_id == X.astype(np.float32)).all(), "identity-weight MAC failed"

    # F2 bit-exact vs the explicit reference above.
    y = compute_y(W, X)
    assert y.dtype == np.float32
    assert np.array_equal(y, f2_ref(W, X)), "F2 sequence mismatch vs reference"

    # ── F3 trunc8: toward zero, then low byte (wrap), reinterpreted as int8 ──
    z = np.array([1.9, -1.9, 130.1, -130.1, 3.0, -3.0], dtype=np.float32)
    assert trunc8(z).dtype == np.int8
    assert (trunc8(z) == np.array([1, -1, -126, 126, 3, -3], dtype=np.int8)).all()
    # int32 clamp boundary: 2^31 -> clamp to 2^31-1 -> 0xFF -> -1 ; -2^31 -> 0x00 -> 0
    assert trunc8(np.array([2**31], dtype=np.float32))[0] == -1
    assert trunc8(np.array([-(2**31)], dtype=np.float32))[0] == 0

    # ── F3 modes (functional form first, trunc8 last; int8 out) ──
    t = np.array([[1.9, -1.9, 130.1, -130.1]], dtype=np.float32)
    assert (acam_transform(t, MODE_REGULAR)
            == np.array([[1, -1, -126, 126]], dtype=np.int8)).all()

    t2 = np.array([[-3.2, 0.0, 2.7, -0.0]], dtype=np.float32)
    assert (acam_transform(t2, MODE_ACTIVATION)
            == np.array([[0, 0, 2, 0]], dtype=np.int8)).all()

    # EXP: 1 + v + v^2/2 ; v=2048 -> 2099201 -> low byte 1
    te = np.array([[0.0, 1.0, -1.0, 2048.0]], dtype=np.float32)
    assert (acam_transform(te, MODE_EXP)
            == np.array([[1, 2, 0, 1]], dtype=np.int8)).all()

    # LOG: v - 1 ; -1.5 truncates toward zero to -1
    tl = np.array([[1.0, 3.75, -0.5, 1.5]], dtype=np.float32)
    assert (acam_transform(tl, MODE_LOG)
            == np.array([[0, 2, -1, 0]], dtype=np.int8)).all()

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

    print("nldpe_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
