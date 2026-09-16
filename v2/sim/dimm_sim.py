#!/usr/bin/env python3
"""dimm_sim.py — NL-DPE DIMM behavior simulator (SCAFFOLD — behavior TODO(you)).

Operator: C[m,n] = sum_k E[m,k,n]; E = ACAM_EXP(LA[m,k] + LB[k,n]);
LA/LB = ACAM_LOG(A/B) through identity crossbars.

Parallelism — pool/farm model (`v2/spec/pool_farm_model.md`):

    logA pool (n_A) ──▶ LA[M,K] int8 ──┐
                                       ├─▶ exp farm (n_E) ──▶ acc[M,N] int32
    logB pool (n_B) ──▶ LB[K,N] int8 ──┘     + CLB add         (exact sum)

    reuse theorem : W_E : W_A : W_B = M·N : M : N
    balance law   : n_A = ceil(n_E/N), n_B = ceil(n_E/M)   (K cancels)
    timing        : per-pool passes ceil(work/I), ceil(passes/n) per crossbar,
                    T = primitive T(p) (`nldpe_sim.cycle_model`, §5.3);
                    total = max(...) if overlapped, sum(...) if phase-separated

Value contract : `v2/oracle/dimm_ref.py` (bit-exact; imported as ref)
Cycle contract : `DimmCycleModel` / `dimm_cycle_model()` in this file

Your behavior implementation must make the gated self-test pass: values equal
the oracle, measured cycles equal the shadow.

Run:  python3 v2/sim/dimm_sim.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the sibling primitive sim + oracle modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import nldpe_sim as prim  # noqa: E402
import dimm_ref as ref  # noqa: E402


@dataclass
class DimmCycleModel:
    """Pool/farm cycle shadow (spec §3-§4): work → passes → per-crossbar
    passes → primitive T, combined by the phase policy."""

    I: int                  # elements per identity pass = min(R, C)
    work_A: int             # M*K
    work_B: int             # K*N
    work_E: int             # M*N*K
    passes_A: int           # ceil(work_A / I)
    passes_B: int
    passes_E: int
    n_A: int                # crossbars (design parameters)
    n_B: int
    n_E: int
    xbar_passes_A: int      # ceil(passes_A / n_A)
    xbar_passes_B: int
    xbar_passes_E: int
    T_A: int                # primitive T(xbar_passes) — 0 if no work
    T_B: int
    T_E: int
    overlap: bool           # True: max(T_*); False: sum(T_*) [phase-separated]
    total: int


def _T(passes: int, R: int, C: int, P: int, BUF: int) -> int:
    """Primitive §5.3 T(p) for a single crossbar; 0 passes -> 0 cycles."""
    if passes <= 0:
        return 0
    return int(prim.cycle_model(passes, R, C, P, BUF).total)


def balanced_counts(n_E: int, M: int, N: int) -> tuple[int, int]:
    """PF3: log-pool crossbar counts that equalize pool/farm finish times."""
    return max(1, -(-n_E // N)), max(1, -(-n_E // M))


def dimm_cycle_model(M: int, N: int, K: int, R: int = 256, C: int = 256,
                     BUF: int = 40, P: int = 8, n_A: int = 1, n_B: int = 1,
                     n_E: int = 16, overlap: bool = True) -> DimmCycleModel:
    """Declared cycle shadow for one DIMM: A[M,K] @ B[K,N] (spec §3)."""
    I = min(R, C)
    work_A, work_B, work_E = M * K, K * N, M * N * K

    def passes(work: int) -> int:
        return -(-work // I) if work > 0 else 0

    passes_A, passes_B, passes_E = passes(work_A), passes(work_B), passes(work_E)
    xp_A = -(-passes_A // n_A) if passes_A else 0
    xp_B = -(-passes_B // n_B) if passes_B else 0
    xp_E = -(-passes_E // n_E) if passes_E else 0
    T_A, T_B, T_E = (_T(xp_A, R, C, P, BUF), _T(xp_B, R, C, P, BUF),
                     _T(xp_E, R, C, P, BUF))
    total = max(T_A, T_B, T_E) if overlap else T_A + T_B + T_E
    return DimmCycleModel(
        I=I, work_A=work_A, work_B=work_B, work_E=work_E,
        passes_A=passes_A, passes_B=passes_B, passes_E=passes_E,
        n_A=n_A, n_B=n_B, n_E=n_E,
        xbar_passes_A=xp_A, xbar_passes_B=xp_B, xbar_passes_E=xp_E,
        T_A=T_A, T_B=T_B, T_E=T_E, overlap=overlap, total=total,
    )


def identity_convert(x: np.ndarray, mode: int, R: int = 256, C: int = 256,
                     BUF: int = 40, P: int = 8) -> tuple[np.ndarray, int]:
    """Identity crossbar + ACAM conversion of a flat int8 stream (PF9).

    Pass budget: I = min(R, C) elements per pass, ceil(len(x)/I) passes, each
    padded to a full R-byte burst; output sliced back to len(x).

    Returns (converted int8 [len(x)], passes). Programs a fresh identity
    primitive per call — cache/reuse at the call site once the behavior lands.
    """
    flat = np.asarray(x, dtype=np.int8).reshape(-1)
    if flat.size == 0:
        return flat.copy(), 0
    dpe = prim.NldpeDpe(R, C, BUF, P)
    dpe.program_weights(np.eye(R, C, dtype=np.int8))
    I = min(R, C)
    n_passes = -(-flat.size // I)
    X = np.zeros((n_passes, R), dtype=np.int8)
    for p in range(n_passes):
        chunk = flat[p * I:(p + 1) * I]
        X[p, :chunk.size] = chunk
    res = dpe.run_workload(X, mode)
    out = res.out_stream[:, :I].reshape(-1)[:flat.size].reshape(-1)
    return out.view(np.int8), n_passes


@dataclass
class DimmResult:
    """Everything the self-test / later cross-checks consume."""

    C: np.ndarray                       # int32 [M, N] exact reduced output
    used_cycles: int                    # measured; must equal cycle_model.total
    cycle_model: DimmCycleModel         # the shadow this run is compared to
    logA: np.ndarray | None = None      # int8 [M, K] (debug / intermediate)
    logB: np.ndarray | None = None      # int8 [K, N] (debug / intermediate)
    timeline: list = field(default_factory=list)  # stage events (free-form)


class NldpeDimm:
    """DIMM operator on NL-DPE primitives (pool/farm configuration)."""

    def __init__(self, R: int = 256, C: int = 256, BUF: int = 40, P: int = 8,
                 n_A: int = 1, n_B: int = 1, n_E: int = 16,
                 overlap: bool = True) -> None:
        self.R, self.C, self.BUF, self.P = R, C, BUF, P
        self.n_A, self.n_B, self.n_E = n_A, n_B, n_E
        self.overlap = overlap
        # TODO(you): instantiate the pool/farm resources once:
        #   - n_A / n_B identity-programmed primitive DPEs (MODE_LOG)
        #   - n_E identity-programmed primitive DPEs (MODE_EXP)
        #   - LA[M,K] / LB[K,N] buffers + acc[M,N] int32 accumulators
        # `identity_convert()` above owns the I=min(R,C) pass budget/padding.

    def shadow_cycles(self, M: int, N: int, K: int) -> DimmCycleModel:
        """Cycle contract for this instance's geometry (scaffold-owned)."""
        return dimm_cycle_model(M, N, K, R=self.R, C=self.C, BUF=self.BUF,
                                P=self.P, n_A=self.n_A, n_B=self.n_B,
                                n_E=self.n_E, overlap=self.overlap)

    def run_matmul(self, A: np.ndarray, B: np.ndarray) -> DimmResult:
        """Run one DIMM: A int8 [M,K] @ B int8 [K,N] -> DimmResult.

        Values must equal `dimm_ref.dimm_matmul(A, B)`; used_cycles must equal
        `self.shadow_cycles(M, N, K).total`.
        """
        A8 = np.asarray(A, dtype=np.int8)
        B8 = np.asarray(B, dtype=np.int8)
        assert A8.ndim == 2 and B8.ndim == 2, "A, B must be 2-D"
        assert A8.shape[1] == B8.shape[0], "inner dimension mismatch"

        # ------------------------------------------------------------------
        # TODO(you) 1 — producers (log pools, spec §1)
        #   LA = identity_convert(A.flatten(), MODE_LOG); LB likewise.
        #   Schedule them so T_A/T_B match the shadow (n_A/n_B crossbars).
        # ------------------------------------------------------------------
        # TODO(you) 2 — buffers
        #   Park LA[M,K] and LB[K,N] (int8) so the farm can stream them.
        # ------------------------------------------------------------------
        # TODO(you) 3 — exp farm (spec §1 inner loop)
        #   For each (m,n,k): u = LA[m,k] + LB[k,n] (CLB integer add);
        #   e = ACAM_EXP(u) on the identity n_E crossbars (MODE_EXP).
        #   Re-read each LA/LB value the required N/M times from the buffers
        #   (no per-element log reconversion).
        # ------------------------------------------------------------------
        # TODO(you) 4 — reduction
        #   acc[m,n] = exact int32 sum over k of the unsigned exp bytes.
        # ------------------------------------------------------------------
        # TODO(you) 5 — pool/farm schedule
        #   Issue ceil(passes/n) crossbar passes per pool; overlap them if
        #   self.overlap (buffers must be filled ahead), else run phases
        #   sequentially (A, B, then farm).
        # ------------------------------------------------------------------
        # TODO(you) 6 — cycle engine
        #   Measure the schedule and return used_cycles = shadow.total.
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "TODO(you): implement DIMM behavior blocks 1-6 in run_matmul()"
        )


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/dimm_sim.py
# Gated: reports shadow-only until the behavior TODO blocks are implemented,
# so it can never print a false ALL PASS.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # ── Pool/farm anchors (spec §4.2: M=N=128, K=64, R=C=256, I=256) ───────
    cm = dimm_cycle_model(128, 128, 64, R=256, C=256, n_A=1, n_B=1, n_E=128)
    assert (cm.work_A, cm.work_B, cm.work_E) == (8192, 8192, 1_048_576), cm
    assert (cm.passes_A, cm.passes_B, cm.passes_E) == (32, 32, 4096), cm
    assert (cm.xbar_passes_A, cm.xbar_passes_B, cm.xbar_passes_E) == (32, 32, 32)
    assert cm.T_A == cm.T_B == cm.T_E, (cm.T_A, cm.T_B, cm.T_E)
    assert cm.total == cm.T_E == 1974, cm.total   # T(32) = 114 + 31*60

    # ── Balance law and K-cancellation (PF3) ───────────────────────────────
    assert balanced_counts(128, 128, 128) == (1, 1)
    for K in (1, 64):
        c = dimm_cycle_model(128, 128, K, n_A=1, n_B=1, n_E=128)
        assert c.T_A == c.T_B == c.T_E, (K, c.T_A, c.T_B, c.T_E)

    # ── Phase policy (overlap = max, sequential = sum) ─────────────────────
    ovl = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128, overlap=True)
    seq = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128, overlap=False)
    assert ovl.total == max(ovl.T_A, ovl.T_B, ovl.T_E)
    assert seq.total == seq.T_A + seq.T_B + seq.T_E

    # ── PF9: identity pass budget (I=256 -> 300 elements = 2 padded passes) ─
    rng = np.random.default_rng(7)
    x = rng.integers(-128, 128, size=300, dtype=np.int8)
    conv, passes = identity_convert(x, prim.MODE_LOG, R=256, C=256)
    assert passes == 2, passes
    assert np.array_equal(conv, ref.log_domain(x)), "identity_convert != oracle"

    A = rng.integers(-128, 128, size=(8, 6), dtype=np.int8)
    B = rng.integers(-128, 128, size=(6, 10), dtype=np.int8)
    expected = ref.dimm_matmul(A, B)

    dpe = NldpeDimm(R=256, C=256)
    try:
        res = dpe.run_matmul(A, B)
    except NotImplementedError as err:
        print(f"dimm_sim self-test: shadow PASS; behavior TODO — {err}")
        return

    # Values ≡ oracle (dual: final reduce here; logA/logB when provided).
    assert res.C.dtype == np.int32 and res.C.shape == (8, 10), res.C.shape
    assert np.array_equal(res.C, expected), "DIMM values != dimm_ref"
    if res.logA is not None:
        assert np.array_equal(res.logA, ref.log_domain(A)), "logA != oracle"
    if res.logB is not None:
        assert np.array_equal(res.logB, ref.log_domain(B)), "logB != oracle"

    # Cycles ≡ shadow.
    shadow = dpe.shadow_cycles(8, 10, 6)
    assert res.used_cycles == shadow.total, (
        f"measured {res.used_cycles} != shadow {shadow.total}"
    )

    print("dimm_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
