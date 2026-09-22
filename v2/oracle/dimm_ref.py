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
* Operator pass layer (§6 F5-F8): every conversion is a crossbar pass with
  the ACAM as its output stage, capacity I = min(R,C) elements; the feed is
  int8 (F7 invariance) and the schedule owns the pass count (ideal packed =
  ceil(W/I); unpacked = Σ ceil(len_i/I)). The elementwise functions here are
  the *dual view*; the self-test asserts the pass view agrees bit-exactly.
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

    Elementwise *dual view* (no pass geometry). The normative pass view is
    `log_domain_pass` (§6 F5/F6); the self-test asserts they agree.
    """
    return ref.acam_transform(np.asarray(x, dtype=np.int32), ref.MODE_LOG)


def log_domain_pass(x: np.ndarray, R: int = 256, C: int = 256
                    ) -> tuple[np.ndarray, int]:
    """§6 F5/F6 pass view of ACAM LOG: ceil(L/I) identity passes.

    Returns (int8 values, passes); schedule owns the pass count.
    """
    shape = np.shape(x)
    v, p = ref.convert_stream(np.asarray(x, dtype=np.int8).reshape(-1),
                              ref.MODE_LOG, R, C)
    return v.reshape(shape), p


def exp_domain(u: np.ndarray) -> np.ndarray:
    """Identity crossbar + ACAM EXP: integer in -> int8 out (P24).

    Elementwise *dual view* over the wide argument `u`. The normative pass
    view feeds `trunc8(u)` (F7) — `exp_domain_pass`; values agree by the F7
    invariance inside the un-clamped EXP range.
    """
    return ref.acam_transform(np.asarray(u), ref.MODE_EXP)


def exp_domain_pass(u: np.ndarray, R: int = 256, C: int = 256
                    ) -> tuple[np.ndarray, int]:
    """§6 F5/F7 pass view of ACAM EXP: `trunc8` feed + ceil(L/I) passes.

    Returns (int8 values, passes).
    """
    shape = np.shape(u)
    fed = ref.trunc8(np.asarray(u)).reshape(-1)          # F7: int8 crossbar feed
    v, p = ref.convert_stream(fed, ref.MODE_EXP, R, C)
    return v.reshape(shape), p


# ---------------------------------------------------------------------------
# Staged views (harness diagnostics): crossbar y -> ACAM bytes per pass.
# ---------------------------------------------------------------------------
def log_domain_stages(x: np.ndarray, R: int = 256, C: int = 256
                      ) -> tuple[np.ndarray, np.ndarray, int]:
    """Staged ACAM LOG conversion: (y int32, bytes int8, passes).

    Identity weights (F5): the crossbar output is the fed int8 value
    sign-extended; `bytes = trunc8(y - 1)` (P24). Same windowing/padding as
    `nldpe_sim.identity_pass`.
    """
    shape = np.shape(x)
    flat = np.asarray(x, dtype=np.int8).reshape(-1)
    if flat.size == 0:
        return (np.zeros(shape, dtype=np.int32),
                np.zeros(shape, dtype=np.int8), 0)
    I = min(R, C)
    n_passes = -(-flat.size // I)
    y_out = np.empty(n_passes * I, dtype=np.int32)
    b_out = np.empty(n_passes * I, dtype=np.int8)
    for p in range(n_passes):
        chunk = flat[p * I:(p + 1) * I]
        y = chunk.astype(np.int32)
        y_out[p * I:p * I + chunk.size] = y
        b_out[p * I:p * I + chunk.size] = ref.acam_transform(y, ref.MODE_LOG)
    return (y_out[:flat.size].reshape(shape),
            b_out[:flat.size].reshape(shape), n_passes)


def exp_stages(u: np.ndarray, R: int = 256, C: int = 256
               ) -> tuple[np.ndarray, np.ndarray, int]:
    """Staged ACAM EXP conversion: (y int32, bytes int8, passes).

    F7: the wide CLB argument `u` is truncated to int8 before the crossbar;
    `y` is that feed sign-extended; `bytes = trunc8(1 + y + y²/2)` (P24).
    """
    shape = np.shape(u)
    fed = ref.trunc8(np.asarray(u)).reshape(-1)
    if fed.size == 0:
        return (np.zeros(shape, dtype=np.int32),
                np.zeros(shape, dtype=np.int8), 0)
    I = min(R, C)
    n_passes = -(-fed.size // I)
    y_out = np.empty(n_passes * I, dtype=np.int32)
    b_out = np.empty(n_passes * I, dtype=np.int8)
    for p in range(n_passes):
        chunk = fed[p * I:(p + 1) * I]
        y = chunk.astype(np.int32)
        y_out[p * I:p * I + chunk.size] = y
        b_out[p * I:p * I + chunk.size] = ref.acam_transform(y, ref.MODE_EXP)
    return (y_out[:fed.size].reshape(shape),
            b_out[:fed.size].reshape(shape), n_passes)


def dimm_stages_full(A: np.ndarray, B: np.ndarray, R: int = 256, C: int = 256
                     ) -> dict:
    """All DIMM stages, pass-structured (harness diagnostics).

    Keys: logA_y/logA/logB_y/logB [M,K]/[K,N]; exp_u int64, exp_y int32,
    exp_bytes int8 [M,N,K]; acc int64 [M,N]; C int32 [M,N]; passes tuple.
    """
    A8 = np.asarray(A, dtype=np.int8)
    B8 = np.asarray(B, dtype=np.int8)
    M, K = A8.shape
    N = B8.shape[1]
    logA_y, logA, passes_A = log_domain_stages(A8, R, C)
    logB_y, logB, passes_B = log_domain_stages(B8, R, C)
    exp_u = np.empty((M, N, K), dtype=np.int64)
    exp_y = np.empty((M, N, K), dtype=np.int32)
    exp_b = np.empty((M, N, K), dtype=np.int8)
    passes_E = 0
    for k in range(K):
        u = (logA[:, k].astype(np.int64)[:, None]
             + logB[k, :].astype(np.int64)[None, :])
        y, b, p = exp_stages(u, R, C)
        exp_u[:, :, k], exp_y[:, :, k], exp_b[:, :, k] = u, y, b
        passes_E += p
    acc = exp_b.view(np.uint8).astype(np.int64).sum(axis=2)
    return dict(logA_y=logA_y, logA=logA, logB_y=logB_y, logB=logB,
                exp_u=exp_u, exp_y=exp_y, exp_bytes=exp_b, acc=acc,
                C=acc.astype(np.int32), passes=(passes_A, passes_B, passes_E))


def _reduce(logA: np.ndarray, logB: np.ndarray, R: int = 256, C: int = 256
            ) -> tuple[np.ndarray, int]:
    """Exact int32 sum over k of exp(logA[m,k] + logB[k,n]) via passes.

    Returns (C int32 [M,N], total exp passes over all k).
    """
    out = np.zeros((logA.shape[0], logB.shape[1]), dtype=np.int64)
    passes_E = 0
    for k in range(logA.shape[1]):
        u = (logA[:, k].astype(np.int64)[:, None]
             + logB[k].astype(np.int64)[None, :])
        eb, p = exp_domain_pass(u, R, C)
        out += eb.view(np.uint8).astype(np.int64)
        passes_E += p
    return out.astype(np.int32), passes_E


def dimm_stages(A: np.ndarray, B: np.ndarray, R: int = 256, C: int = 256
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logA, logB, C) for intermediate cross-checks.

    logA : int8 [M, K]; logB : int8 [K, N]; C : int32 [M, N].
    """
    A8 = np.asarray(A, dtype=np.int8)
    B8 = np.asarray(B, dtype=np.int8)
    assert A8.ndim == 2 and B8.ndim == 2, "A, B must be 2-D"
    assert A8.shape[1] == B8.shape[0], "inner dimension mismatch"
    logA, _ = log_domain_pass(A8, R, C)
    logB, _ = log_domain_pass(B8, R, C)
    return logA, logB, _reduce(logA, logB, R, C)[0]


def dimm_matmul(A: np.ndarray, B: np.ndarray, R: int = 256, C: int = 256
                ) -> np.ndarray:
    """Exact idealized DIMM: C[m,n] = sum_k exp(logA[m,k] + logB[k,n]).

    A : int8 [M, K]; B : int8 [K, N]. Returns int32 [M, N], the exact sum of
    the unsigned int8 EXP bytes over k (see module docstring).
    """
    return dimm_stages(A, B, R, C)[2]


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

    # ── §6 F5-F7 pass view == elementwise dual view ────────────────────────
    pv, pa = log_domain_pass(A)
    assert np.array_equal(pv, log_domain(A)) and pa == -(-A.size // 256)
    u = (logA[:, 0].astype(np.int64)[:, None]
         + logB[0].astype(np.int64)[None, :])
    pv_e, pe = exp_domain_pass(u)
    assert np.array_equal(pv_e, exp_domain(u)), "F7 invariance broke"
    assert pe == -(-u.size // 256)

    # L > I geometry: R=C=8 chunks the same streams; values are geometry-free.
    logA8, logB8, C8 = dimm_stages(A, B, R=8, C=8)
    assert np.array_equal(logA8, logA) and np.array_equal(logB8, logB)
    assert np.array_equal(C8, y)
    assert dimm_matmul(A, B, R=8, C=8).tolist() == y.tolist()

    # ── Staged views agree with the elementwise dual and the final oracle ──
    st = dimm_stages_full(A, B)
    assert st["exp_u"].shape == (4, 5, 3), st["exp_u"].shape
    assert np.array_equal(st["logA_y"], A.astype(np.int32))
    assert np.array_equal(st["logB_y"], B.astype(np.int32))
    assert np.array_equal(st["logA"], log_domain(A))
    assert np.array_equal(st["logB"], log_domain(B))
    assert np.array_equal(st["exp_bytes"], exp_domain(st["exp_u"]))
    assert np.array_equal(st["acc"], y.astype(np.int64))
    assert np.array_equal(st["C"], y)
    assert st["passes"][2] == sum(
        -(-u.size // 256) for u in [st["exp_u"][:, :, k] for k in range(3)])

    print("dimm_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
