#!/usr/bin/env python3
"""dimm_sim.py — NL-DPE DIMM behavior simulator (pool/farm model).

Operator: C[m,n] = sum_k E[m,k,n]; E = ACAM_EXP(LA[m,k] + LB[k,n]);
LA/LB = ACAM_LOG(A/B) through identity crossbars.

Parallelism — pool/farm model (`v2/spec/dimm.md`):

    logA pool (n_A) ──▶ LA[M,K] int8 ──┐
                                       ├─▶ exp farm (n_E) ──▶ acc[M,N] int32
    logB pool (n_B) ──▶ LB[K,N] int8 ──┘     + CLB add         (exact sum)

    reuse theorem : W_E : W_A : W_B = M·N : M : N
    balance law   : n_A = ceil(n_E/N), n_B = ceil(n_E/M)   (K cancels)
    timing        : pass counts are **injected by the schedule** (DimmPassPlan:
                    ideal packed or unpacked, §6 F5-F8); per-crossbar passes
                    ceil(passes/n), T = primitive T(p) (`nldpe_sim.cycle_model`
                    §5.3); total = max(...) if overlapped, sum(...) if
                    phase-separated. `ideal_pass_plan()` is the packed-count
                    convenience, not an assumption of the model.

Value contract : `v2/oracle/dimm_ref.py` (bit-exact; imported as ref)
Cycle contract : `DimmCycleModel` / `dimm_cycle_model()` in this file

The self-test gates values (≡ oracle) and cycles (measured ≡ shadow) and the
balance contract (derive-by-default counts + residual). `run_matmul(...,
collect_stages=True)` returns per-stage diagnostics (`DimmStages`) for staged
intermediate verification: crossbar y (int32) → ACAM bytes (int8) → CLB add /
exact reduction (int64) → final C (int32).

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
import nldpe_ref as nref  # noqa: E402
import dimm_ref as ref  # noqa: E402


@dataclass
class DimmPassPlan:
    """Schedule-owned pass counts (§6 F5-F8 / PF9).

    The schedule decides how the workload maps onto crossbar passes — ideal
    packed (`ceil(W/I)`, one full-capacity pass per I elements) or unpacked
    (`Σ ceil(len_i/I)`, e.g. one short vector per pass). The cycle model
    consumes the counts; it never guesses them.
    """

    passes_A: int           # logA identity-conversion passes
    passes_B: int           # logB identity-conversion passes
    passes_E: int           # exp identity-conversion passes (farm, all k)


def ideal_pass_plan(M: int, N: int, K: int, R: int = 256,
                    C: int = 256) -> DimmPassPlan:
    """Packed-count convenience: `ceil(work/I)` per pool, `I = min(R, C)`.

    This is the *ideal* (minimum) pass count; a real schedule injects the
    counts it actually issues (unpacked counts are ≥ these and must be
    reported, not silently idealized).
    """
    I = min(R, C)
    return DimmPassPlan(-(-(M * K) // I), -(-(K * N) // I),
                        -(-(M * N * K) // I))


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
    balanced: bool          # balance_residual == 0
    balance_residual: int   # max(T_A,T_B,T_E) - min(T_A,T_B,T_E)


def _T(passes: int, R: int, C: int, P: int, BUF: int) -> int:
    """Primitive §5.3 T(p) for a single crossbar; 0 passes -> 0 cycles."""
    if passes <= 0:
        return 0
    return int(prim.cycle_model(passes, R, C, P, BUF).total)


def balanced_counts(n_E: int, M: int, N: int) -> tuple[int, int]:
    """PF3 continuous-law reference (form-symmetric in M↔N).

    `(max(1, ceil(n_E/N)), max(1, ceil(n_E/M)))` — the closed-form optimum of
    the continuous rate model. Not used by the production derivation (which is
    exact, see `dimm_cycle_model`); kept as the human-facing design law and as
    a ±1 sanity bound in the self-test.
    """
    return max(1, -(-n_E // N)), max(1, -(-n_E // M))


def dimm_cycle_model(M: int, N: int, K: int, R: int = 256, C: int = 256,
                     BUF: int = 40, P: int = 8,
                     n_A: int | None = None, n_B: int | None = None,
                     n_E: int = 16, overlap: bool = True,
                     *, plan: DimmPassPlan) -> DimmCycleModel:
    """Declared cycle shadow for one DIMM: A[M,K] @ B[K,N] (spec §3-§4).

    `plan` is **required** and schedule-owned (DimmPassPlan): the schedule
    injects the pass counts it actually issues. Use `ideal_pass_plan()` only
    as the packed-count convenience for sizing.

    `n_A`/`n_B` are both `None` (default) → **derive-by-default**: with the
    farm's per-crossbar passes fixed at `ceil(P_E/n_E)`, `T` is increasing in
    passes, so the exact optimum is the smallest machine count that fits the
    log pool inside the farm's latency:

        n_log = max(1, ceil(P_log / ceil(P_E / n_E)))

    This is lexicographically optimal for `(max T, residual, machines)`.
    Explicit `n_A`/`n_B` (both or neither) are design overrides and are
    reported with `balance_residual`/`balanced`. `balanced_counts` (PF3) is
    the continuous closed-form reference, within ±1 of the exact counts.
    """
    assert (n_A is None) == (n_B is None), \
        "derive both log-pool counts (None) or override both"
    assert plan.passes_A >= 0 and plan.passes_B >= 0 and plan.passes_E >= 0
    assert n_E >= 1

    I = min(R, C)
    logA_work, logB_work, exp_work = M * K, K * N, M * N * K
    logA_passes, logB_passes = plan.passes_A, plan.passes_B
    exp_passes = plan.passes_E

    # Farm first: with n_E fixed its per-crossbar passes set the latency
    # floor T(exp_per_xbar) that the log pools must not exceed.
    exp_per_xbar = -(-exp_passes // n_E) if exp_passes else 0
    t_exp = _T(exp_per_xbar, R, C, P, BUF)

    if n_A is None:                      # derive-by-default (dimm.md §4)
        n_A = max(1, -(-logA_passes // exp_per_xbar)) if exp_per_xbar else 1
        n_B = max(1, -(-logB_passes // exp_per_xbar)) if exp_per_xbar else 1

    logA_per_xbar = -(-logA_passes // n_A) if logA_passes else 0
    logB_per_xbar = -(-logB_passes // n_B) if logB_passes else 0
    t_logA = _T(logA_per_xbar, R, C, P, BUF)
    t_logB = _T(logB_per_xbar, R, C, P, BUF)

    residual = max(t_logA, t_logB, t_exp) - min(t_logA, t_logB, t_exp)
    total = (max(t_logA, t_logB, t_exp) if overlap
             else t_logA + t_logB + t_exp)
    return DimmCycleModel(
        I=I, work_A=logA_work, work_B=logB_work, work_E=exp_work,
        passes_A=logA_passes, passes_B=logB_passes, passes_E=exp_passes,
        n_A=n_A, n_B=n_B, n_E=n_E,
        xbar_passes_A=logA_per_xbar, xbar_passes_B=logB_per_xbar,
        xbar_passes_E=exp_per_xbar,
        T_A=t_logA, T_B=t_logB, T_E=t_exp, overlap=overlap, total=total,
        balanced=residual == 0, balance_residual=residual,
    )


def identity_pass(x: np.ndarray, mode: int, R: int = 256, C: int = 256,
                  BUF: int = 40, P: int = 8,
                  dpe: prim.NldpeDpe | None = None,
                  return_y: bool = False) -> tuple:
    """§6 F5/F6 identity conversion of a flat int8 stream (PF9).

    Pass budget: I = min(R, C) elements per pass, ceil(len(x)/I) passes, each
    zero-padded to a full R-byte burst; padding outputs are discarded.

    Returns (converted int8 [len(x)], passes); with `return_y=True` also the
    wide crossbar output (int32 [len(x)], pre-ACAM) for staged diagnostics.

    `dpe` may be a pre-programmed identity DPE to reuse across calls (the
    pool/farm behavior caches one per mode in `NldpeDpe.__init__`); when
    `None`, a fresh identity primitive is constructed and programmed — the
    reference path used by the self-tests.
    """
    flat = np.asarray(x, dtype=np.int8).reshape(-1)
    if flat.size == 0:
        empty = flat.copy()
        return (empty, 0, empty.astype(np.int32)) if return_y else (empty, 0)
    if dpe is None:
        dpe = prim.NldpeDpe(R, C, BUF, P)
        dpe.program_weights(np.eye(R, C, dtype=np.int8))
    I = min(R, C)
    n_passes = -(-flat.size // I)
    X = np.zeros((n_passes, R), dtype=np.int8)
    for p in range(n_passes):
        chunk = flat[p * I:(p + 1) * I]
        X[p, :chunk.size] = chunk
    res = dpe.run_workload(X, mode)
    out = res.out_stream[:, :I].reshape(-1)[:flat.size]
    if not return_y:
        return out.view(np.int8), n_passes
    y = res.y[:, :I].reshape(-1)[:flat.size].astype(np.int32)
    return out.view(np.int8), n_passes, y


@dataclass
class DimmStages:
    """Staged diagnostics for one DIMM run (`collect_stages=True`).

    Stage order mirrors the datapath: crossbar y (int32, pre-ACAM) → ACAM
    bytes (int8) → CLB log-add / exact reduction (int64) → final (int32).
    """

    logA_y: np.ndarray      # int32 [M, K] producer crossbar output
    logA: np.ndarray        # int8  [M, K] producer ACAM LOG output
    logB_y: np.ndarray      # int32 [K, N]
    logB: np.ndarray        # int8  [K, N]
    exp_u: np.ndarray       # int64 [M, N, K] CLB log-domain add
    exp_y: np.ndarray       # int32 [M, N, K] farm crossbar output (= trunc8(u))
    exp_bytes: np.ndarray   # int8  [M, N, K] farm ACAM EXP output
    acc: np.ndarray         # int64 [M, N] exact reduction (pre-cast)


@dataclass
class DimmResult:
    """Everything the self-test / later cross-checks consume."""

    C: np.ndarray                       # int32 [M, N] exact reduced output
    used_cycles: int                    # measured; must equal cycle_model.total
    cycle_model: DimmCycleModel         # the shadow this run is compared to
    logA: np.ndarray | None = None      # int8 [M, K] (debug / intermediate)
    logB: np.ndarray | None = None      # int8 [K, N] (debug / intermediate)
    timeline: list = field(default_factory=list)  # stage events (free-form)
    stages: DimmStages | None = None    # staged diagnostics (collect_stages)


class NldpeDimm:
    """DIMM operator on NL-DPE primitives (pool/farm configuration)."""

    def __init__(self, R: int = 256, C: int = 256, BUF: int = 40, P: int = 8,
                 n_A: int | None = None, n_B: int | None = None,
                 n_E: int = 16, overlap: bool = True) -> None:
        self.R, self.C, self.BUF, self.P = R, C, BUF, P
        # n_A/n_B: None = derive-by-default (dimm.md §4); explicit = override.
        self.n_A, self.n_B, self.n_E = n_A, n_B, n_E
        self.overlap = overlap
        # Pool resources, instantiated once (weights stationary, A5). The
        # behavior converts each unique element once; n_A/n_B/n_E are the
        # *virtual* crossbar counts the cycle model uses for timing.
        eye = np.eye(R, C, dtype=np.int8)
        self._dpe_log = prim.NldpeDpe(R, C, BUF, P)
        self._dpe_log.program_weights(eye)
        self._dpe_exp = prim.NldpeDpe(R, C, BUF, P)
        self._dpe_exp.program_weights(eye)

    def shadow_cycles(self, M: int, N: int, K: int,
                      plan: DimmPassPlan) -> DimmCycleModel:
        """Cycle contract for this instance's geometry (schedule injects plan)."""
        return dimm_cycle_model(M, N, K, R=self.R, C=self.C, BUF=self.BUF,
                                P=self.P, n_A=self.n_A, n_B=self.n_B,
                                n_E=self.n_E, overlap=self.overlap, plan=plan)

    def run_matmul(self, A: np.ndarray, B: np.ndarray,
                   collect_stages: bool = False) -> DimmResult:
        """Run one DIMM: A int8 [M,K] @ B int8 [K,N] -> DimmResult.

        Values must equal `dimm_ref.dimm_matmul(A, B)`; `used_cycles` is
        measured from the schedule this run issues (issued pass plan +
        crossbar counts + phase policy) and must equal the independently
        computed `shadow_cycles(M, N, K, plan).total`.

        `collect_stages=True` additionally returns per-stage diagnostics
        (`DimmResult.stages`) for staged intermediate verification; the
        default keeps the hot path free of the [M,N,K] arrays.
        """
        A8 = np.asarray(A, dtype=np.int8)
        B8 = np.asarray(B, dtype=np.int8)
        assert A8.ndim == 2 and B8.ndim == 2, "A, B must be 2-D"
        assert A8.shape[1] == B8.shape[0], "inner dimension mismatch"
        M, K = A8.shape
        N = B8.shape[1]
        pool_geom = dict(R=self.R, C=self.C, BUF=self.BUF, P=self.P)

        # 1 — producers (log pools): each unique operand converted once.
        prodA = identity_pass(A8, prim.MODE_LOG, **pool_geom,
                              dpe=self._dpe_log, return_y=collect_stages)
        prodB = identity_pass(B8, prim.MODE_LOG, **pool_geom,
                              dpe=self._dpe_log, return_y=collect_stages)
        logA_flat, logA_passes = prodA[0], prodA[1]
        logB_flat, logB_passes = prodB[0], prodB[1]

        # 2 — buffers: LA [M,K], LB [K,N] parked for the farm's re-reads.
        logA = logA_flat.reshape(M, K)
        logB = logB_flat.reshape(K, N)

        # 3 — exp farm: per k, exact CLB add -> int8 feed (F7) -> identity
        #     EXP passes; accumulate the unsigned bytes (work M*N*K).
        acc = np.zeros((M, N), dtype=np.int64)
        exp_passes = 0
        if collect_stages:
            exp_u = np.empty((M, N, K), dtype=np.int64)
            exp_y = np.empty((M, N, K), dtype=np.int32)
            exp_b = np.empty((M, N, K), dtype=np.int8)
        for k in range(K):
            log_sum = (logA[:, k].astype(np.int64)[:, None]
                       + logB[k, :].astype(np.int64)[None, :])
            conv = identity_pass(nref.trunc8(log_sum), prim.MODE_EXP,
                                 **pool_geom, dpe=self._dpe_exp,
                                 return_y=collect_stages)
            exp_bytes, k_passes = conv[0], conv[1]
            acc += exp_bytes.view(np.uint8).reshape(M, N).astype(np.int64)
            exp_passes += k_passes
            if collect_stages:
                exp_u[:, :, k] = log_sum
                exp_y[:, :, k] = conv[2].reshape(M, N)
                exp_b[:, :, k] = exp_bytes.reshape(M, N)

        # 4 — reduction: exact int32 sum of the unsigned exp bytes.
        assert int(np.abs(acc).max()) <= 2**31 - 1, "C exceeds int32 (B8)"
        C = acc.astype(np.int32)
        stages = (DimmStages(logA_y=prodA[2].reshape(M, K), logA=logA,
                             logB_y=prodB[2].reshape(K, N), logB=logB,
                             exp_u=exp_u, exp_y=exp_y, exp_bytes=exp_b,
                             acc=acc)
                  if collect_stages else None)

        # 5 — pool/farm schedule: plan from the passes actually issued;
        #     distribute over the instance's crossbar counts (derived when
        #     None); T(p) per pool.
        plan = DimmPassPlan(logA_passes, logB_passes, exp_passes)
        shadow = self.shadow_cycles(M, N, K, plan)
        t_logA = _T(-(-logA_passes // shadow.n_A) if logA_passes else 0,
                    self.R, self.C, self.P, self.BUF)
        t_logB = _T(-(-logB_passes // shadow.n_B) if logB_passes else 0,
                    self.R, self.C, self.P, self.BUF)
        t_exp = _T(-(-exp_passes // shadow.n_E) if exp_passes else 0,
                   self.R, self.C, self.P, self.BUF)

        # 6 — cycle engine: measured from the issued schedule; the shadow is
        #     the independent comparison (`measured == shadow.total`).
        measured = (max(t_logA, t_logB, t_exp) if self.overlap
                    else t_logA + t_logB + t_exp)
        timeline = [("logA", logA_passes, shadow.n_A, t_logA),
                    ("logB", logB_passes, shadow.n_B, t_logB),
                    ("exp", exp_passes, shadow.n_E, t_exp)]
        return DimmResult(C=C, used_cycles=measured, cycle_model=shadow,
                          logA=logA, logB=logB, timeline=timeline,
                          stages=stages)


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/dimm_sim.py
# Gates: values ≡ oracle, cycles measured ≡ shadow, derive/balance contract.
# ---------------------------------------------------------------------------
def _T256(passes: int) -> int:
    """Reference T(p) for the 256×256 geometry (fill 114, steady 60)."""
    return _T(passes, 256, 256, 8, 40)


def _test_constants() -> None:
    """Spec §6 worked example: work → passes → per-crossbar passes → T."""
    plan = ideal_pass_plan(128, 128, 64, R=256, C=256)
    cm = dimm_cycle_model(128, 128, 64, R=256, C=256,
                          n_A=1, n_B=1, n_E=128, plan=plan)
    assert (cm.work_A, cm.work_B, cm.work_E) == (8192, 8192, 1_048_576)
    assert (cm.passes_A, cm.passes_B, cm.passes_E) == (32, 32, 4096)
    assert (cm.xbar_passes_A, cm.xbar_passes_B, cm.xbar_passes_E) == (32, 32, 32)
    t32 = _T256(32)
    assert t32 == 1974                      # 114 + 31*60
    assert (cm.T_A, cm.T_B, cm.T_E) == (t32, t32, t32)
    assert cm.total == t32
    print("  [reference values] 128x128 K=64 worked example: OK")


def _test_balance_reference() -> None:
    """PF3 continuous reference and K-cancellation (ratio is K-independent)."""
    assert balanced_counts(128, 128, 128) == (1, 1)
    for K in (1, 64):
        c = dimm_cycle_model(128, 128, K, n_A=1, n_B=1, n_E=128,
                             plan=ideal_pass_plan(128, 128, K))
        assert c.T_A == c.T_B == c.T_E, (K, c.T_A, c.T_B, c.T_E)
    print("  [balance reference] PF3 + K-cancellation: OK")


def _test_phase_policy() -> None:
    """overlap=True → max(T_i); overlap=False → sum(T_i)."""
    plan = ideal_pass_plan(128, 128, 64, R=256, C=256)
    ovl = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128,
                           overlap=True, plan=plan)
    seq = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128,
                           overlap=False, plan=plan)
    assert ovl.total == max(ovl.T_A, ovl.T_B, ovl.T_E)
    assert seq.total == seq.T_A + seq.T_B + seq.T_E
    print("  [phase policy] max vs sum: OK")


def _test_plan_and_packing() -> None:
    """Schedule-injected counts: packing changes counts, never values (F8)."""
    vec = np.arange(-32, 32, dtype=np.int8)              # 64 elements
    one, one_pass = identity_pass(vec, prim.MODE_LOG, R=256, C=256)
    packed, packed_pass = identity_pass(np.tile(vec, 4), prim.MODE_LOG,
                                        R=256, C=256)
    assert one_pass == 1                                 # 64 ≤ I = 256
    assert 4 * one_pass == 4                             # unpacked: 4 passes
    assert packed_pass == 1                              # 4×64 = I fills a pass
    assert np.array_equal(packed, np.tile(one, 4)), "packing changed values"

    plan = ideal_pass_plan(128, 128, 64, R=256, C=256)
    cm = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128, plan=plan)
    bare = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128,
                            plan=DimmPassPlan(64, 64, 8192))
    assert bare.passes_E != cm.passes_E, "plan must be consumed, not guessed"
    print("  [plan/packing] injected counts consumed; values invariant: OK")


def _test_derive_contract() -> None:
    """Derive-by-default (dimm.md §4): exact counts, residual always reported."""
    def derive(M: int, N: int, K: int, n_E: int):
        return dimm_cycle_model(M, N, K, n_E=n_E,
                                plan=ideal_pass_plan(M, N, K),
                                n_A=None, n_B=None)

    t16, t32, t64 = _T256(16), _T256(32), _T256(64)

    # Reference: the farm sets the pace and both logs fit exactly.
    ref = derive(128, 128, 64, 128)
    assert (ref.n_A, ref.n_B) == (1, 1)
    assert (ref.T_A, ref.T_B, ref.T_E) == (t32, t32, t32)
    assert ref.balanced and ref.balance_residual == 0

    # Floor: n_E < N makes equality impossible (max(1,·) binds) — reported.
    floor = derive(128, 128, 64, 64)
    assert (floor.T_A, floor.T_B, floor.T_E) == (t32, t32, t64)
    assert not floor.balanced and floor.balance_residual == t64 - t32

    # Scaling: more farm crossbars shorten the whole pipeline (PF4).
    scale = derive(128, 128, 64, 256)
    assert (scale.n_A, scale.n_B) == (2, 2)
    assert scale.balanced and scale.total == t16
    assert scale.total < ref.total < floor.total

    # Counts are K-independent (PF3).
    for K in (1, 64):
        c = derive(128, 128, K, 128)
        assert (c.n_A, c.n_B) == (1, 1)

    # Rectangular (S×V-like, M≠N): counts are asymmetric by the law.
    rect = derive(64, 128, 64, 128)
    assert (rect.n_A, rect.n_B) == (1, 2)
    assert (rect.T_A, rect.T_B, rect.T_E) == (t16, t16, t16)
    assert rect.balanced and rect.balance_residual == 0
    assert balanced_counts(128, 64, 128) == (1, 2)       # continuous reference

    # Forcing symmetric counts on a rectangular shape is strictly worse.
    sym = dimm_cycle_model(64, 128, 64, n_E=128, n_A=1, n_B=1,
                           plan=ideal_pass_plan(64, 128, 64))
    assert sym.total > rect.total and not sym.balanced
    assert sym.balance_residual > 0

    # An explicit override is reported, and loses to the derived counts.
    ov = dimm_cycle_model(128, 128, 64, n_E=256, n_A=1, n_B=1,
                          plan=ideal_pass_plan(128, 128, 64))
    assert ov.total > scale.total and not ov.balanced
    assert ov.balance_residual > 0
    print("  [derive contract] exact counts + residual reporting: OK")


def _test_identity_pass() -> None:
    """F5/F6 pass budget, oracle values, cached ≡ fresh DPE."""
    rng = np.random.default_rng(7)
    x = rng.integers(-128, 128, size=300, dtype=np.int8)
    conv, passes = identity_pass(x, prim.MODE_LOG, R=256, C=256)
    assert passes == 2                                   # ceil(300 / I=256)
    assert np.array_equal(conv, ref.log_domain(x)), "identity_pass != oracle"

    dm = NldpeDimm(R=256, C=256)
    fresh, fresh_passes = identity_pass(x, prim.MODE_LOG, R=256, C=256)
    cached, cached_passes = identity_pass(x, prim.MODE_LOG, R=256, C=256,
                                          dpe=dm._dpe_log)
    assert (fresh_passes, cached_passes) == (passes, passes)
    assert np.array_equal(fresh, cached), "cached != fresh DPE"
    print("  [identity pass] budget + oracle + cached/fresh: OK")


def _check_stage(label: str, got: np.ndarray, expected: np.ndarray) -> None:
    """Bit-exact stage comparison with first-mismatch coordinates."""
    assert got.shape == expected.shape, (
        f"{label}: shape {got.shape} != {expected.shape}")
    assert got.dtype == expected.dtype, (
        f"{label}: dtype {got.dtype} != {expected.dtype}")
    bad = np.argwhere(got != expected)
    assert len(bad) == 0, (
        f"{label}: {len(bad)} mismatch(es), first at {tuple(bad[0])}: "
        f"got {got[tuple(bad[0])]} exp {expected[tuple(bad[0])]}")


def _test_behavior() -> None:
    """Staged end-to-end over a shape matrix.

    Per shape and phase policy: intermediates must align first (crossbar y
    int32 -> ACAM bytes int8 -> CLB add -> exact reduction int64), then the
    final int32 output and the cycle contract.
    """
    shapes = [
        #  M   N   K   R    C   n_E
        (8, 10, 6, 256, 256, 16),     # baseline; farm unpacked 6 vs ideal 2
        (64, 128, 64, 256, 256, 128),  # rectangular S×V-like -> (1,2)
        (128, 64, 64, 256, 256, 128),  # mirrored -> (2,1)
        (4, 5, 3, 8, 8, 4),           # small crossbar: multi-pass everywhere
        (16, 16, 4, 64, 32, 8),       # R ≠ C (I = 32)
        (32, 32, 8, 128, 64, 16),     # R ≠ C, larger
    ]
    rng = np.random.default_rng(11)
    for (M, N, K, R, C, n_E) in shapes:
        A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
        B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
        exp = ref.dimm_stages_full(A, B, R=R, C=C)
        for overlap in (True, False):
            dpe = NldpeDimm(R=R, C=C, n_E=n_E, overlap=overlap)
            res = dpe.run_matmul(A, B, collect_stages=True)
            st = res.stages

            # Stage 1 — crossbar outputs (int32, pre-ACAM).
            _check_stage("logA crossbar y (int32)", st.logA_y, exp["logA_y"])
            _check_stage("logB crossbar y (int32)", st.logB_y, exp["logB_y"])
            _check_stage("exp crossbar y (int32)", st.exp_y, exp["exp_y"])

            # Stage 2 — ACAM outputs (int8).
            _check_stage("logA ACAM bytes (int8)", st.logA, exp["logA"])
            _check_stage("logB ACAM bytes (int8)", st.logB, exp["logB"])
            _check_stage("exp ACAM bytes (int8)", st.exp_bytes, exp["exp_bytes"])

            # Stage 3 — CLB add and exact reduction (int64).
            _check_stage("CLB log-add u (int64)", st.exp_u, exp["exp_u"])
            _check_stage("reduced acc (int64)", st.acc, exp["acc"])

            # Stage 4 — final output and cycles.
            _check_stage("final C (int32)", res.C, exp["C"])
            assert res.used_cycles == res.cycle_model.total, (
                f"cycles measured {res.used_cycles} != shadow "
                f"{res.cycle_model.total}")

            print(f"  [behavior] M={M} N={N} K={K} {R}x{C} ovl={int(overlap)}: "
                  f"stages OK, cycles OK ({res.used_cycles})")


def _self_test() -> None:
    print("dimm_sim self-test:")
    _test_constants()
    _test_balance_reference()
    _test_phase_policy()
    _test_plan_and_packing()
    _test_derive_contract()
    _test_identity_pass()
    _test_behavior()
    print("dimm_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
