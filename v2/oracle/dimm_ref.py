#!/usr/bin/env python3
"""dimm_ref.py — NumPy value reference for the NL-DPE DIMM operator.

Role
----
Plain math, no state, no time. The behavior simulator (`v2/sim/dimm_sim.py`)
must agree with this file bit-exactly on values; the cycle contract lives in
that simulator (`dimm_cycle_model`), not here.

Operator (NL-DPE, idealized integer semantics)
----------------------------------------------
    C[m,n] = sum_k E[m,k,n]
    E      = ACAM_EXP(U)                      identity crossbar, MODE_EXP
    U      = logA[m,k] + logB[k,n]            CLB integer add (log domain)
    logA   = ACAM_LOG(A)                      identity crossbar, MODE_LOG
    logB   = ACAM_LOG(B)

with the v2 integer ACAM forms (`v2/spec/dpe_nldpe.md` §6 F3 / P16 / P24):

    log(v) = trunc8(v - 1)                    (MODE_LOG)
    exp(u) = trunc8(1 + u + floor(u^2/2))     (MODE_EXP)
    trunc8(z): clamp to int32, keep the low byte as signed int8

Transcribed from the DIMM workload definition (`rtl_flow/docs/
FIDELITY_METHODOLOGY.md` §5) and the mapping (`paper/methodology/
attention_dimm_mapping.md` §3-§4), expressed in the v2 clean-room contract.

Notes
-----
* A, B are int8 dynamic inputs (T1/A16); the identity crossbar is exact
  (§8 I7), so the log/exp conversions see the int32 value of the element.
* K-identity packing and W-lane scheduling are implementation parallelism:
  they change cycle counts only, never values. Padded lanes carry zeros and
  contribute nothing to the reduction.
* Idealized integer model: log/exp are the ACAM's declared approximations, so
  the result is not numerically meaningful attention. Do not compare against a
  float matmul — the model itself is the contract (cf. spec A13).
* Final reduce is an exact int32 sum of unsigned int8 exp bytes (no trunc8 at
  the output). K * 255 < 2^31 holds for any K <= 2^23.

Functions
---------
  log_domain(x)      -> int8, ACAM LOG (identity path)
  exp_domain(u)      -> int8, ACAM EXP (identity path)
  dimm_stages(A, B)  -> (logA int8 [M,K], logB int8 [K,N], C int32 [M,N])
  dimm_matmul(A, B)  -> int32 [M,N] reduced output (linear domain)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the sibling oracle module importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_ref as ref  # noqa: E402


def log_domain(x: np.ndarray) -> np.ndarray:
    """Identity crossbar + ACAM LOG: int8 in -> int8 out (P24).

    y = trunc8(x - 1), where x is read as its int32 value.
    """
    return ref.acam_transform(np.asarray(x, dtype=np.int32), ref.MODE_LOG)


def exp_domain(u: np.ndarray) -> np.ndarray:
    """Identity crossbar + ACAM EXP: integer in -> int8 out (P24).

    e = trunc8(1 + u + floor(u*u/2)) (exact wide intermediate, int32 clamp).
    """
    return ref.acam_transform(np.asarray(u), ref.MODE_EXP)


def _reduce(logA: np.ndarray, logB: np.ndarray) -> np.ndarray:
    """Exact int32 sum over k of exp(logA[m,k] + logB[k,n])."""
    out = np.zeros((logA.shape[0], logB.shape[1]), dtype=np.int64)
    for k in range(logA.shape[1]):
        u = (logA[:, k].astype(np.int64)[:, None]
             + logB[k].astype(np.int64)[None, :])
        out += exp_domain(u).view(np.uint8).astype(np.int64)
    return out.astype(np.int32)


def dimm_stages(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logA, logB, C) for intermediate cross-checks.

    logA : int8 [M, K]; logB : int8 [K, N]; C : int32 [M, N].
    """
    A8 = np.asarray(A, dtype=np.int8)
    B8 = np.asarray(B, dtype=np.int8)
    assert A8.ndim == 2 and B8.ndim == 2, "A, B must be 2-D"
    assert A8.shape[1] == B8.shape[0], "inner dimension mismatch"
    logA = log_domain(A8)
    logB = log_domain(B8)
    return logA, logB, _reduce(logA, logB)


def dimm_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact idealized DIMM: C[m,n] = sum_k exp(logA[m,k] + logB[k,n]).

    A : int8 [M, K]; B : int8 [K, N]. Returns int32 [M, N], the exact sum of
    the unsigned int8 EXP bytes over k (see module docstring).
    """
    return dimm_stages(A, B)[2]


# ---------------------------------------------------------------------------
# Self-test — run:  python3 dimm_ref.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # ── Hand-computed 2x2 ──────────────────────────────────────────────────
    # logA = A-1 = [[1,2],[3,4]]; logB = B-1 = [[0,1],[2,3]];
    # C[m,n] = sum_k exp(logA[m,k] + logB[k,n]), exp(u) = 1+u+floor(u^2/2):
    #   f(1)=2 f(2)=5 f(3)=8 f(4)=13 f(5)=18 f(6)=25 f(7)=32
    # C[0,0] = f(1+0)+f(2+2) = 2+13 = 15    C[0,1] = f(1+1)+f(2+3) = 5+18 = 23
    # C[1,0] = f(3+0)+f(4+2) = 8+25 = 33    C[1,1] = f(3+1)+f(4+3) = 13+32 = 45
    A = np.array([[2, 3], [4, 5]], dtype=np.int8)
    B = np.array([[1, 2], [3, 4]], dtype=np.int8)
    C = dimm_matmul(A, B)
    assert C.dtype == np.int32 and C.shape == (2, 2), f"{C.dtype} {C.shape}"
    assert np.array_equal(C, np.array([[15, 23], [33, 45]], dtype=np.int32)), C

    # ── All-ones: log(1)=0, exp(0)=1, C = K exactly ────────────────────────
    ones = dimm_matmul(np.ones((3, 5), dtype=np.int8),
                       np.ones((5, 4), dtype=np.int8))
    assert (ones == 5).all(), ones

    # ── Independent elementwise structural reference (signed extremes) ─────
    rng = np.random.default_rng(3)
    A = rng.integers(-128, 128, size=(4, 3), dtype=np.int8)
    B = rng.integers(-128, 128, size=(3, 5), dtype=np.int8)
    A[0, 0], A[1, 1] = -128, 127
    B[2, 2] = -128

    def f_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
        for m in range(a.shape[0]):
            for n in range(b.shape[1]):
                total = 0
                for k in range(a.shape[1]):
                    la = int((int(a[m, k]) - 1) & 0xFF)      # log low byte
                    lb = int((int(b[k, n]) - 1) & 0xFF)
                    la -= 256 if la >= 128 else 0            # signed byte
                    lb -= 256 if lb >= 128 else 0
                    u = la + lb
                    e = (1 + u + (u * u) // 2) & 0xFF        # exp low byte
                    total += e                               # unsigned sum
                out[m, n] = total
        return out

    y = dimm_matmul(A, B)
    assert np.array_equal(y.astype(np.int64), f_ref(A, B)), "structural mismatch"

    # ── log/exp domain helpers agree with the staged reference ─────────────
    logA, logB, C2 = dimm_stages(A, B)
    assert logA.dtype == np.int8 and logB.dtype == np.int8
    assert np.array_equal(C2, y)
    assert (logA == log_domain(A)).all() and (logB == log_domain(B)).all()

    print("dimm_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
