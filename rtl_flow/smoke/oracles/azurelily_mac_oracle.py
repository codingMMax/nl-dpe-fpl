#!/usr/bin/env python3
"""
Bit-exact numpy oracle for the FAITHFUL Azure-Lily DPE primitive (Task #92).

Independent of the RTL and of the simulator's cycle formula.

  * Functional oracle: computes the expected MAC from first principles
    (signed int8 multiply, int32 accumulator, 2's complement). Returns
    one int32 per output column. NO ACAM transform -- Azure-Lily has
    no analog-CAM nonlinear stage.

  * Cycle-count oracle: predicts compute / total cycles from the
    DECLARED physical model (PRECISION + (PIPELINE_DEPTH - 1) +
    ACAM_CYCLES). Azure-Lily has PIPELINE_DEPTH=3 (Crossbar + ADC +
    ShiftAdd) and ACAM_CYCLES=0 -- giving the same CCYC=10 as NL
    faithful (which has PD=2, AC=1) by structural symmetry, but with a
    different physical decomposition. Does NOT read RTL or sim.

The script also serves as a test-vector generator. Run

  python azurelily_mac_oracle.py --gen-test-vectors

to write the .mem files consumed by tb_dpe_azurelily_faithful.v
($readmemh) into oracles/test_vectors_al/ (note the _al suffix --
separate from NL's test_vectors/ to avoid collision).

Bit-serial signed multiply convention (matches RTL):
    For unsigned bit b in 0..PRECISION-2: ADD (crossbar_sum << b)
    For MSB bit PRECISION-1:               SUBTRACT (crossbar_sum << b)
    where crossbar_sum is sum_r (input_bit_r[b] ? W[r,c] : 0)

The AL faithful primitive routes crossbar_sum through an extra ADC
register stage before the shift-add accumulator, but functionally the
operation is identical (transparent register, no quantisation loss).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# --------------------- Functional oracle ---------------------


def signed_int8_mac(weights_2d, inputs_1d):
    """Per-column signed int8 x int8 -> int32 dot product.

    Args:
        weights_2d: numpy array, shape (R, C), dtype int8 (or castable).
        inputs_1d:  numpy array, shape (R,),   dtype int8 (or castable).

    Returns:
        numpy array shape (C,), dtype int32. Element c =
        sum_r (weights[r, c] * inputs[r]) with proper signed 2's
        complement arithmetic.
    """
    w = np.asarray(weights_2d, dtype=np.int8).astype(np.int32)
    x = np.asarray(inputs_1d, dtype=np.int8).astype(np.int32)
    return w.T @ x


def bit_serial_signed_mac(weights_2d, inputs_1d, precision=8):
    """Reference bit-serial implementation that mirrors the RTL.

    For each output column c:
        mac = 0
        for b in 0 .. precision-2:
            crossbar_sum_b = sum_r ( (inputs[r] >> b) & 1 ) * weights[r, c]
            mac += crossbar_sum_b << b
        # MSB bit (signed 2's complement)
        b = precision - 1
        crossbar_sum_b = sum_r ( (inputs[r] >> b) & 1 ) * weights[r, c]
        mac -= crossbar_sum_b << b

    Equivalent to ``signed_int8_mac`` for ``precision == 8`` provided
    the inputs sit in [-128, 127]. Kept here as a reference / debug
    aid; the RTL must match BOTH ``signed_int8_mac`` and this expansion.

    The AL faithful primitive routes the bit-serial xbar sum through
    an extra ADC register stage (vs NL's direct accumulation), but
    functionally this is a transparent register pass-through -- the
    bit-serial signed MAC math is identical to NL.
    """
    w = np.asarray(weights_2d, dtype=np.int8).astype(np.int32)
    x_signed = np.asarray(inputs_1d, dtype=np.int8).astype(np.int32)
    # interpret as 2's-complement bit pattern (so bit (P-1) is the sign)
    x_bits = x_signed.astype(np.int64) & ((1 << precision) - 1)
    R, C = w.shape
    out = np.zeros(C, dtype=np.int64)
    for b in range(precision):
        bit_slice = ((x_bits >> b) & 1).astype(np.int32)  # (R,)
        # crossbar_sum[c] = sum_r bit_slice[r] * weights[r, c]
        crossbar_sum = (bit_slice.astype(np.int32) @ w).astype(np.int64)  # (C,)
        if b == precision - 1:
            out -= crossbar_sum << b
        else:
            out += crossbar_sum << b
    return out.astype(np.int32)


def truncate_to_int8_byte(x_int32):
    """Take the low 8 bits of int32 -> 0..255 (byte view).

    The RTL's OUTPUT drain writes ``mac_acc[c][7:0]`` which is the
    low byte of the int32 value, viewed as an unsigned byte by the TB
    capture buffer.

    (Compared to NL faithful: there the OUTPUT drains acam_out[c][7:0];
    here AL drains mac_acc[c][7:0] directly since there is no ACAM
    intermediate buffer.)
    """
    arr = np.asarray(x_int32, dtype=np.int64)
    return (arr & 0xFF).astype(np.uint8)


def expected_outputs(weights_2d, inputs_1d, precision=8):
    """End-to-end functional oracle: weights+input -> output bytes.

    Returns (mac_int32, byte_uint8) both shape (C,).

    (Compared to NL faithful: no acam tuple element -- AL has no ACAM
    stage, so the post-MAC value is the output value directly.)
    """
    mac = signed_int8_mac(weights_2d, inputs_1d)
    # Sanity: bit-serial reference matches the direct multiply.
    bs = bit_serial_signed_mac(weights_2d, inputs_1d, precision=precision)
    if not np.array_equal(mac, bs):
        raise RuntimeError(
            "bit-serial reference disagrees with direct multiply; oracle bug"
        )
    byte = truncate_to_int8_byte(mac)
    return mac, byte


# --------------------- Cycle oracle ---------------------


def expected_load_cycles(R, BUF):
    eps = BUF // 8
    return (R + eps - 1) // eps


def expected_output_cycles(C, BUF):
    eps = BUF // 8
    return (C + eps - 1) // eps


def expected_compute_cycles(precision=8, pipeline_depth=3, acam_cycles=0):
    """CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES.

    For Azure-Lily PD=3, ACAM=0 -> CCYC = P + 2 (INT8 = 10).
    Same total as NL faithful (PD=2, AC=1 -> P + 2 = 10) by structural
    symmetry: AL trades the ACAM read-out stage for an extra pipeline
    sample/quantize stage (ADC). Different physics, same cycle count.

    This is the DECLARED architectural value -- the TB MEASURES the
    actual gap from the RTL and compares against this.
    """
    return precision + (pipeline_depth - 1) + acam_cycles


def expected_total_cycles(
    R, C, BUF, M=1, precision=8, pipeline_depth=3, acam_cycles=0,
):
    """Total observed cycles between first LOAD strobe and last OUTPUT byte.

    Unified analytical formula (Task #99 — double-buffered LOAD substrate):

        T_fill   = LCYC + CCYC + OCYC           ← architectural minimum
        T_steady = max(LCYC, CCYC, OCYC)
        T(M)     = T_fill + (M - 1) * T_steady

    Double-buffered LOAD (Task #99): each DPE owns TWO input-buffer
    substrates A/B; LOAD pass-(k+1) writes to substrate B while COMPUTE
    pass-k reads substrate A. The LOAD-gate (`load_safe` register) of
    the single-substrate Option A1 design is removed, dropping the
    +PRECISION term from T_steady. The double-buffer area is already
    accounted for in the architectural area budget (out of this scope).

    For typical AL configs (LCYC=256, PRECISION=8, CCYC=10, OCYC=64),
    T_steady drops from 264 (=LCYC+P) to 256 (=LCYC). LCYC still
    dominates (LCYC >> CCYC, OCYC for AL), so the saving is exactly
    PRECISION per pass.
    """
    lcyc = expected_load_cycles(R, BUF)
    ocyc = expected_output_cycles(C, BUF)
    ccyc = expected_compute_cycles(precision, pipeline_depth, acam_cycles)
    t_fill = lcyc + ccyc + ocyc
    # Task #99: double-buffered LOAD — no LOAD-gate, no +PRECISION term.
    t_steady = max(lcyc, ccyc, ocyc)
    return t_fill + (M - 1) * t_steady


# --------------------- Test-vector generation ---------------------


def write_mem_int8_array(path, values, width=8):
    """Write a 1-D int array as one hex-byte-per-line .mem file.

    The TB ``$readmemh``s these into ``reg [7:0] arr[0:N-1]``. Negative
    int8s are written as 2's complement (so e.g. -1 -> ff).
    """
    arr = np.asarray(values, dtype=np.int64)
    mask = (1 << width) - 1
    with open(path, "w") as fh:
        for v in arr.flat:
            fh.write(f"{int(v) & mask:0{width//4}x}\n")


def write_mem_int32_array(path, values):
    """Write a 1-D int array as one hex-int32-per-line .mem file.

    The TB ``$readmemh``s these into ``reg [31:0] arr[0:N-1]``.
    """
    arr = np.asarray(values, dtype=np.int64)
    mask = (1 << 32) - 1
    with open(path, "w") as fh:
        for v in arr.flat:
            fh.write(f"{int(v) & mask:08x}\n")


def make_t1_identity(R, C):
    """T1: identity weights, all-ones input. Expected[c] = 1 for c < min(R, C)."""
    w = np.zeros((R, C), dtype=np.int8)
    for i in range(min(R, C)):
        w[i, i] = 1
    x = np.ones(R, dtype=np.int8)
    return w, x


def make_t2_random(R, C, seed):
    """T2: random int8 weights and inputs (full range to exercise signed mul)."""
    rng = np.random.default_rng(seed)
    # Use a moderate range to keep mac in int32 without overflow concerns
    # (R=512 * 31 * 31 ~ 5e5 < 2^31).
    w = rng.integers(-32, 31, size=(R, C), dtype=np.int8)
    x = rng.integers(-32, 31, size=(R,), dtype=np.int8)
    return w, x


def make_t5_signed_inputs(R, C, seed):
    """T5: identity-like weights, random INT8 inputs including negatives.

    (Same setup as NL faithful's T7; renumbered to T5 here because AL
    has no ACAM modes, so T5/T6 exp/log tests are not applicable.)
    """
    rng = np.random.default_rng(seed)
    # weights: small diagonal so MAC = inputs[c] for c < min(R,C)
    w = np.zeros((R, C), dtype=np.int8)
    for i in range(min(R, C)):
        w[i, i] = 1
    # inputs span -128..127 (full signed int8 range)
    x = rng.integers(-128, 127, size=(R,), dtype=np.int8)
    return w, x


def write_test_vectors(out_dir, R=512, C=128, precision=8):
    """Generate .mem files for all 5 TB tests.

    Each test writes:
        t{n}_weights.mem  : R*C bytes (row-major: weights[r][c] -> r*C+c)
        t{n}_inputs.mem   : R bytes
        t{n}_mac.mem      : C int32 values (final MAC, AL has no ACAM)
        t{n}_expected.mem : C bytes (low byte of MAC, truncated)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def emit(idx, weights, inputs):
        mac, byte = expected_outputs(weights, inputs, precision=precision)
        # row-major flatten: weights[r][c] at index r*C+c
        wflat = weights.reshape(-1)
        write_mem_int8_array(out / f"t{idx}_weights.mem", wflat)
        write_mem_int8_array(out / f"t{idx}_inputs.mem", inputs)
        write_mem_int32_array(out / f"t{idx}_mac.mem", mac)
        write_mem_int8_array(out / f"t{idx}_expected.mem", byte)
        # Header tag file (text) for debug visibility
        with open(out / f"t{idx}_meta.txt", "w") as fh:
            fh.write(f"R={R} C={C} precision={precision} arch=Azure-Lily\n")
            fh.write(f"weights.shape={weights.shape}\n")
            fh.write(f"inputs.shape={inputs.shape}\n")
            fh.write(f"mac[0:8]={mac[:8].tolist()}\n")
            fh.write(f"byte[0:8]={byte[:8].tolist()}\n")

    # T1: identity weights, all-ones input -> mac=1 for c<min(R,C)
    w, x = make_t1_identity(R, C)
    emit(1, w, x)

    # T2: random int8 weights+inputs (deterministic seed)
    w, x = make_t2_random(R, C, seed=20260516)
    emit(2, w, x)

    # T3: identity-like to make cycle measurement easy. Reuse T1's vectors.
    w, x = make_t1_identity(R, C)
    emit(3, w, x)

    # T4: M-sweep -- per-pass identity weights, distinct constant inputs.
    # The TB constructs M passes' inputs procedurally; here we just
    # emit a single-pass identity reference so the TB can verify pass-0.
    w, x = make_t1_identity(R, C)
    emit(4, w, x)

    # T5: random signed inputs (including negatives)
    # (NL faithful's T7 equivalent; AL has no T5/T6 ACAM modes.)
    w, x = make_t5_signed_inputs(R, C, seed=20260517)
    emit(5, w, x)

    # Cycle-oracle summary (analytical sim cycle count per test, Task #99).
    with open(out / "cycle_oracle.txt", "w") as fh:
        fh.write("# Azure-Lily faithful primitive cycle oracle (Task #99 — double-buffered LOAD)\n")
        fh.write("# CCYC      = PRECISION + (PIPELINE_DEPTH-1) + ACAM_CYCLES\n")
        fh.write(f"#           = {precision} + 2 + 0 = {precision + 2}\n")
        fh.write("# T_fill    = LCYC + CCYC + OCYC\n")
        fh.write("# T_steady  = max(LCYC, CCYC, OCYC)\n")
        fh.write("# Pass-(k+1) LOAD writes to substrate B while pass-k\n")
        fh.write("# COMPUTE reads substrate A — no LOAD-gate, no +PRECISION.\n")
        for m in (1, 2, 4, 8):
            tot = expected_total_cycles(R, C, BUF=16, M=m,
                                         precision=precision,
                                         pipeline_depth=3,
                                         acam_cycles=0)
            ccyc = expected_compute_cycles(precision, 3, 0)
            lcyc = expected_load_cycles(R, 16)
            ocyc = expected_output_cycles(C, 16)
            t_steady = max(lcyc, ccyc, ocyc)
            fh.write(
                f"M={m}: LCYC={lcyc} CCYC={ccyc} OCYC={ocyc} "
                f"T_fill={lcyc + ccyc + ocyc} "
                f"T_steady={t_steady} "
                f"T(M)={tot}\n"
            )

    # Expected-cycle text file (matches TB sim_exp; RTL delta is reported
    # by the harness as info — see CYCLE_ACCOUNTING.md for current value).
    with open(out / "expected_cycles_azurelily.txt", "w") as fh:
        fh.write(f"# Azure-Lily faithful primitive, Task #99 (double-buffered LOAD), R={R} C={C} BUF=16 INT{precision}\n")
        lcyc = expected_load_cycles(R, 16)
        ccyc = expected_compute_cycles(precision, 3, 0)
        ocyc = expected_output_cycles(C, 16)
        t_fill = lcyc + ccyc + ocyc
        t_steady = max(lcyc, ccyc, ocyc)
        fh.write(f"LCYC={lcyc} OCYC={ocyc} CCYC={ccyc}\n")
        fh.write(f"T_fill={t_fill}  (= LCYC + CCYC + OCYC)\n")
        fh.write(f"T_steady={t_steady}  (= max(LCYC, CCYC, OCYC))\n")
        for m in (1, 2, 4, 8):
            tot = t_fill + (m - 1) * t_steady
            fh.write(f"M={m}: T(M) = T_fill + ({m}-1)*T_steady = {tot}\n")

    print(f"[azurelily_mac_oracle] wrote test vectors to {out}")


# --------------------- CLI ---------------------


def main(argv=None):
    p = argparse.ArgumentParser(description="Faithful Azure-Lily DPE oracle / test vector generator")
    p.add_argument("--gen-test-vectors", action="store_true",
                   help="Generate .mem files for tb_dpe_azurelily_faithful.v")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for .mem files "
                        "(default: oracles/test_vectors_al/)")
    p.add_argument("--R", type=int, default=512, help="DPE rows (default 512)")
    p.add_argument("--C", type=int, default=128, help="DPE cols (default 128)")
    p.add_argument("--precision", type=int, default=8, help="bit precision (default 8)")
    p.add_argument("--show-cycle-oracle", action="store_true",
                   help="Print the declared cycle count for INT8 Azure-Lily M-sweep")
    args = p.parse_args(argv)

    if args.show_cycle_oracle:
        for m in (1, 2, 4, 8):
            tot = expected_total_cycles(args.R, args.C, BUF=16, M=m,
                                         precision=args.precision,
                                         pipeline_depth=3, acam_cycles=0)
            ccyc = expected_compute_cycles(args.precision, 3, 0)
            lcyc = expected_load_cycles(args.R, 16)
            ocyc = expected_output_cycles(args.C, 16)
            t_steady = max(lcyc, ccyc, ocyc)
            print(f"M={m}: LCYC={lcyc} CCYC={ccyc} OCYC={ocyc} "
                  f"T_fill={lcyc + ccyc + ocyc} T_steady={t_steady} "
                  f"(=max(LCYC,CCYC,OCYC)) "
                  f"T(M)={tot}")
        return 0

    if args.gen_test_vectors:
        here = Path(__file__).resolve().parent
        out_dir = Path(args.out_dir) if args.out_dir else here / "test_vectors_al"
        write_test_vectors(out_dir, R=args.R, C=args.C, precision=args.precision)
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
