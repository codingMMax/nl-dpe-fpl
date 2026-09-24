#!/usr/bin/env python3
"""dimm_sim.py — NL-DPE DIMM behavior simulator (pool/farm model).

Operator: C[m,n] = sum_k E[m,k,n]; E = ACAM_EXP(LA[m,k] + LB[k,n]);
LA/LB = ACAM_LOG(A/B) through identity crossbars.

Parallelism — pool/farm model:

    logA pool (n_A) ──▶ LA[M,K] int8 ──┐
                                       ├─▶ exp farm (n_E) ──▶ acc[M,N] int32
    logB pool (n_B) ──▶ LB[K,N] int8 ──┘     + CLB add         (exact sum)

    work split  : A converted once (M·K elements), B once (K·N); the
                  (m,n) plane re-converted for every k (M·N·K element-
                  stages; k is sequential)
    reuse       : farm windows : logA windows : logB windows = M·N : M : N
                  (one window = I = min(R,C) elements of a flat stream)
    balance law : n_A = max(1, ceil(n_E/N)), n_B = max(1, ceil(n_E/M))
                  (K cancels in the ratio — continuous reference only; the
                  exact rule is n = max(1, ceil(P_pool / ceil(P_E/n_E)),
                  used by `dimm_cycle_model` when n_A/n_B are None)
    timing      : pass counts are injected by the schedule (DimmPassPlan:
                  packed ceil(W/I) or unpacked Σ ceil(len_i/I) passes);
                  per-crossbar passes = ceil(passes/n);
                  T(p) = T_fill + (p-1)·T_steady with primitive constants
                  T_fill/T_steady (see `_xbar_total`);
                  total = max(T_A, T_B, T_start + T_E) if overlapped,
                          T_A + T_B + T_E if phase-separated.
                  `ideal_pass_plan()` is the packed-count convenience, not
                  an assumption of the model.

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

import json
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
    """Schedule-owned pass counts.

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
    """Pool/farm cycle shadow: work → passes → per-crossbar passes →
    primitive T(p), combined by the phase policy."""

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
    W_A: int                # first-slice windows = ceil(M / I)
    W_B: int                # first-slice windows = ceil(N / I)
    T_start: int            # producer->farm fill: max_p[T_fill + (ceil(W_p/n_p)-1)*T_steady]
    T_add: int              # stage1 add — 0: combinational CLB add in the feed path
    T_reduce: int           # stage3 reduce — 0: acc RMW fused into the exp drain
    serialize_cycles: int   # final output stage M*N — reported separately, not in `total`
    overlap: bool           # True: max(T_A,T_B,T_start+T_E); False: sum
    total: int
    balanced: bool          # balance_residual == 0
    balance_residual: int   # max(T_A,T_B,T_E) - min(T_A,T_B,T_E)


def _xbar_total(passes: int, R: int, C: int, P: int, BUF: int) -> int:
    """Primitive-layer T(p) for one crossbar (delegates to `nldpe_sim`).

    Formulas (the constants this call evaluates):

        LOAD_CYC    = ceil(R·P / BUF)          P = precision in bits
        COMPUTE_CYC = P + 2                    P fires + MSB-acc drain + ACAM
        OUTPUT_CYC  = ceil(C·P / BUF)
        T_fill      = LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
        T_steady    = max(LOAD_CYC + P,        input-buffer bound
                          COMPUTE_CYC,          compute bound
                          OUTPUT_CYC + 1)       output-buffer bound
        T(p)        = T_fill + (p − 1)·T_steady

    Layer contract — notation is never mixed across layers:

      primitive layer (above): LOAD/COMPUTE/OUTPUT, T_fill, T_steady, T(p)
      DIMM layer (`dimm_cycle_model`):
          T_A / T_B / T_E — phase totals over all passes of a pool/farm:
                            T(ceil(P_pool / n_pool))
          T_start         — producer->farm fill; the farm phase's start cycle

    Symbol collisions with the DIMM layer, disambiguated here:
      * `passes` maps to `prim.cycle_model`'s `M` argument (= pass count on
        one crossbar) — NOT the DIMM matrix `M` (A rows);
      * `P` is the activation bit precision (8) — NOT a pass count; pass
        counts are named `passes_*` / `P_x`.

    0 passes -> 0 cycles.
    """
    if passes <= 0:
        return 0
    return int(prim.cycle_model(passes, R, C, P, BUF).total)


def balanced_counts(n_E: int, M: int, N: int) -> tuple[int, int]:
    """Continuous-law reference (form-symmetric in M↔N).

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
    """Declared cycle shadow for one DIMM: A[M,K] @ B[K,N].

    Inputs and assumptions:
      * geometry M,N,K,R,C,BUF,P (BUF-wide ports, P-bit activations);
      * crossbar counts n_E (required) and n_A/n_B (or None → derive);
      * `plan` (keyword-only, required): the pass counts the schedule
        actually issues — packed ceil(W/I) or unpacked Σ ceil(len_i/I).
        The model consumes these counts; it never guesses them.
        `ideal_pass_plan()` is only the packed-count convenience for sizing.
      * overlap=True: concurrent phases, total = max(T_A, T_B, T_start+T_E);
        overlap=False: sequential phases, total = T_A + T_B + T_E.

    Derive-by-default (n_A/n_B are both None): with n_E fixed, the farm's
    per-crossbar passes p_E = ceil(P_E / n_E) fix the time floor T(p_E);
    T(p) is increasing in p, so the smallest pool count that fits inside
    that floor is

        n_log = max(1, ceil(P_log / p_E)),   log ∈ {A, B}

    lexicographically optimal for `(max T, residual, machines)` (any smaller
    n makes T_pool > T_E and the pools become the bottleneck; among feasible
    n, the smallest gives the largest T_pool ≤ T_E, i.e. residual closest to
    0). Explicit `n_A`/`n_B` (both or neither) are design overrides and are
    reported with `balance_residual`/`balanced`. `balanced_counts` is the
    continuous closed-form reference, within ±1 of the exact counts.

    T_start is the producer→farm fill: the first W_A / W_B windows (the
    k=0 slices) must drain into LA/LB before block 0 can read them:

        W_A = ceil(M / I),  W_B = ceil(N / I),  I = min(R, C)
        T_start = max over {A-pool, B-pool} of
                  T_fill + (ceil(W / n) − 1)·T_steady

    (T_fill/T_steady are primitive-layer constants read from
    `nldpe_sim.cycle_model`, see `_xbar_total`). This is a data dependency,
    not implementation overhead: for M,N <= I it reduces to T_fill.

    Stage ledger — every stage's cycle cost is an explicit term of `total`:

        stage0 (LogA ‖ LogB)   T_A, T_B            crossbar T(p), concurrent
        stage1 (pairwise add)  T_add = 0           combinational CLB add + trunc8
                                                   in the farm feed path (no cycle)
        stage2 (exp)           T_start + T_E       producer prologue + crossbar T(p)
        stage3 (reduce)        T_reduce = 0        incremental acc RMW fused into
                                                   the exp drain (no tree, no cycle)
        output (final C)       serialize_cycles    M*N, reported separately

    The zero-cost stages are explicit, not omitted: stage1 is combinational,
    and stage3's accumulator RMW executes in the exp drain cycles (its
    port-bank sum is combinational), so neither adds cycles. The final-output
    serializer is outside `total` by contract
    (GATE-2 expected = total + serialize_cycles).
    """
    assert (n_A is None) == (n_B is None), \
        "derive both log-pool counts (None) or override both"
    assert plan.passes_A >= 0 and plan.passes_B >= 0 and plan.passes_E >= 0
    assert n_E >= 1

    I = min(R, C)
    tot_logA_work, tot_logB_work, tot_exp_work = M * K, K * N, M * N * K
    tot_logA_passes, tot_logB_passes = plan.passes_A, plan.passes_B
    tot_exp_passes = plan.passes_E

    # Exp side first: with n_E fixed its per-crossbar passes set the latency
    # floor T(exp_passes_per_xbar) that the log stages must not exceed.
    exp_passes_per_xbar = (-(-tot_exp_passes // n_E)
                           if tot_exp_passes else 0)
    t_exp = _xbar_total(exp_passes_per_xbar, R, C, P, BUF)

    if n_A is None:          # derive: smallest n with ceil(P/n) <= p_E
        n_A = (max(1, -(-tot_logA_passes // exp_passes_per_xbar))
               if exp_passes_per_xbar else 1)
        n_B = (max(1, -(-tot_logB_passes // exp_passes_per_xbar))
               if exp_passes_per_xbar else 1)

    logA_passes_per_xbar = (-(-tot_logA_passes // n_A)
                            if tot_logA_passes else 0)
    logB_passes_per_xbar = (-(-tot_logB_passes // n_B)
                            if tot_logB_passes else 0)
    t_logA = _xbar_total(logA_passes_per_xbar, R, C, P, BUF)
    t_logB = _xbar_total(logB_passes_per_xbar, R, C, P, BUF)

    # producer->farm fill: T_start = max over log stages of
    #   T_fill + (ceil(W/n) - 1)*T_steady,   W = ceil(dim/I) first-slice
    # windows (they must drain before farm block 0 can read LA/LB).
    # T_fill/T_steady are primitive-layer constants, read directly from the
    # shared primitive cycle model (not re-derived as T(1), T(2)−T(1)).
    W_A = -(-M // I)
    W_B = -(-N // I)
    primitive_geometry = prim.cycle_model(1, R, C, P, BUF)
    t_fill, t_steady = primitive_geometry.t_fill, primitive_geometry.t_steady
    t_start = max(t_fill + (-(-W_A // n_A) - 1) * t_steady,
                  t_fill + (-(-W_B // n_B) - 1) * t_steady)

    balance_residual = max(t_logA, t_logB, t_exp) - min(t_logA, t_logB,
                                                        t_exp)

    # Stage ledger — every stage's cost is an explicit term; stage1/stage3 are
    # zero because they consume no cycles, not because they are ignored:
    #   stage1 add: combinational CLB add + trunc8 in the farm feed path;
    #   stage3 reduce: acc RMW executes in the exp drain cycles (incremental,
    #                  no adder tree; the port-bank sum is combinational).
    # The final-output serializer is outside `total`, added by the harness gate.
    t_add = 0
    t_reduce = 0
    t_final_output = M * N

    t_total = (max(t_logA, t_logB, t_add, t_start + t_exp, t_reduce)
               if overlap
               else t_logA + t_add + t_logB + t_exp + t_reduce)
    return DimmCycleModel(
        I=I, work_A=tot_logA_work, work_B=tot_logB_work,
        work_E=tot_exp_work,
        passes_A=tot_logA_passes, passes_B=tot_logB_passes,
        passes_E=tot_exp_passes,
        n_A=n_A, n_B=n_B, n_E=n_E,
        xbar_passes_A=logA_passes_per_xbar,
        xbar_passes_B=logB_passes_per_xbar,
        xbar_passes_E=exp_passes_per_xbar,
        T_A=t_logA, T_B=t_logB, T_E=t_exp,
        W_A=W_A, W_B=W_B, T_start=t_start,
        T_add=t_add, T_reduce=t_reduce, serialize_cycles=t_final_output,
        overlap=overlap, total=t_total,
        balanced=balance_residual == 0,
        balance_residual=balance_residual,
    )


def identity_pass(x: np.ndarray, mode: int, R: int = 256, C: int = 256,
                  BUF: int = 40, P: int = 8,
                  dpe: prim.NldpeDpe | None = None,
                  return_y: bool = False) -> tuple:
    """Identity conversion of a flat int8 stream through a crossbar eye
    (mode = LOG or EXP).

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
    used_cycles: int                    # = cycle_model.total (single cycle source)
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
        # n_A/n_B: None = derive n = max(1, ceil(P_pool / ceil(P_E/n_E)));
        #          explicit = override (both or neither).
        self.n_A, self.n_B, self.n_E = n_A, n_B, n_E
        self.overlap = overlap
        # Pool resources, instantiated once (weights are the identity eye,
        # programmed once and reused every pass). The behavior converts each
        # unique element once; n_A/n_B/n_E are the *virtual* crossbar counts
        # the cycle model uses for timing.
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

        Values must equal `dimm_ref.dimm_matmul(A, B)`; `used_cycles` is the
        shadow stage-ledger total for the plan this run issues (single source
        of cycle truth — the independent witness is the RTL under GATE 2,
        which is gated against this value + serialize_cycles).

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
        xbar_geom = dict(R=self.R, C=self.C, BUF=self.BUF, P=self.P)

        # 1 — producers (log pools): each unique operand converted once.
        prodA = identity_pass(A8, prim.MODE_LOG, **xbar_geom,
                              dpe=self._dpe_log, return_y=collect_stages)
        prodB = identity_pass(B8, prim.MODE_LOG, **xbar_geom,
                              dpe=self._dpe_log, return_y=collect_stages)
        logA_flat, tot_logA_passes = prodA[0], prodA[1]
        logB_flat, tot_logB_passes = prodB[0], prodB[1]

        # 2 — buffers: LA [M,K], LB [K,N] parked for the farm's re-reads.
        logA = logA_flat.reshape(M, K)
        logB = logB_flat.reshape(K, N)

        # 3 — exp farm: per k, exact CLB add -> trunc8 to int8 feed (the
        #     int8 feed is mode-invariant for LOG/EXP/REGULAR modes) ->
        #     identity EXP passes; accumulate the unsigned bytes (work M*N*K).
        acc = np.zeros((M, N), dtype=np.int64)
        tot_exp_passes = 0
        if collect_stages:
            exp_u = np.empty((M, N, K), dtype=np.int64)
            exp_y = np.empty((M, N, K), dtype=np.int32)
            exp_b = np.empty((M, N, K), dtype=np.int8)
        for k in range(K):
            log_sum = (logA[:, k].astype(np.int64)[:, None]
                       + logB[k, :].astype(np.int64)[None, :])
            conv = identity_pass(nref.trunc8(log_sum), prim.MODE_EXP,
                                 **xbar_geom, dpe=self._dpe_exp,
                                 return_y=collect_stages)
            exp_bytes, k_passes = conv[0], conv[1]
            acc += exp_bytes.view(np.uint8).reshape(M, N).astype(np.int64)
            tot_exp_passes += k_passes
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

        # 5 — schedule: plan from the passes actually issued; the shadow
        #     model is the single source of cycle truth (stage ledger).
        plan = DimmPassPlan(tot_logA_passes, tot_logB_passes,
                            tot_exp_passes)
        shadow = self.shadow_cycles(M, N, K, plan)

        # 6 — cycle engine: used_cycles is the shadow's stage-ledger total;
        #     GATE 2 gates this value (+ serialize_cycles) against the RTL.
        measured = shadow.total
        timeline = [("logA", shadow.passes_A, shadow.n_A, shadow.T_A),
                    ("logB", shadow.passes_B, shadow.n_B, shadow.T_B),
                    ("fill", 0, 0, shadow.T_start),
                    ("exp", shadow.passes_E, shadow.n_E, shadow.T_E)]
        return DimmResult(C=C, used_cycles=measured, cycle_model=shadow,
                          logA=logA, logB=logB, timeline=timeline,
                          stages=stages)

    # -- stimulus dump (GATE 1 + GATE 2 expected files) ----------------------
    def dump_case(self, case_dir: Path, A: np.ndarray, B: np.ndarray) -> None:
        """Write stimulus + expected files for the DIMM RTL cross-check.

        GATE 1: certify this exact case against `dimm_ref.dimm_stages_full`
        (logA, logB, exp_u, exp_bytes, acc, C), the issued pass counts and the
        schedule cycle total BEFORE any file is written; an uncertified case
        is never dumped.

        File contract (consumed by `v2/tb/tb_dimm_top.v`, expanded by
        `v2/smoke/run_dimm_rtl.py`):
          a.mem            — A row-major int8 bytes packed BUF-wide (10-hex/line)
          b.mem            — B row-major int8 bytes packed BUF-wide
          expected_la.npz  — int8 [K, M]  transposed (index k*M+m, probe order)
          expected_lb.npz  — int8 [K, N]  (index k*N+n)
          expected_acc.npz — int32 [M, N] (index m*N+n; valid at `done`)
          expected_c.mem   — M*N int32 words, 8-hex/line, row-major
          case.json        — params + issued plan + cycle model + setup cycles

        RTL runtime note: the model total is the overlapped pool/farm time
        (`max(T_A,T_B,T_E)`); the RTL additionally serializes the reduced
        accumulator to the C stream at one int32 word/cycle, reported as
        `serialize_cycles = M*N` so the harness can gate Δ_impl = measured −
        (total + serialize_cycles) without touching the model.
        """
        A8 = np.asarray(A, dtype=np.int8)
        B8 = np.asarray(B, dtype=np.int8)
        assert A8.ndim == 2 and B8.ndim == 2, "A, B must be 2-D"
        assert A8.shape[1] == B8.shape[0], "inner dimension mismatch"
        M, K = A8.shape
        N = B8.shape[1]

        res = self.run_matmul(A8, B8, collect_stages=True)
        st = res.stages
        exp = ref.dimm_stages_full(A8, B8, R=self.R, C=self.C)
        cm = res.cycle_model

        # -- GATE 1: per-case oracle certification -------------------------
        checks = (
            ("logA (int8)", st.logA, exp["logA"]),
            ("logB (int8)", st.logB, exp["logB"]),
            ("exp_u (int64)", st.exp_u, exp["exp_u"]),
            ("exp bytes (int8)", st.exp_bytes, exp["exp_bytes"]),
            ("acc (int64)", st.acc, exp["acc"]),
            ("C (int32)", res.C, exp["C"]),
        )
        for label, got, want in checks:
            if got.shape != want.shape or not np.array_equal(got, want):
                raise RuntimeError(
                    f"GATE 1: sim/oracle {label} mismatch — case not dumped")
        if tuple(exp["passes"]) != (cm.passes_A, cm.passes_B, cm.passes_E):
            raise RuntimeError(
                f"GATE 1: sim/oracle pass counts mismatch "
                f"{tuple(exp['passes'])} != {(cm.passes_A, cm.passes_B, cm.passes_E)}"
                f" — case not dumped")
        if res.used_cycles != cm.total:
            raise RuntimeError(
                f"GATE 1: sim schedule {res.used_cycles} != model total "
                f"{cm.total} — case not dumped")

        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        # a.mem / b.mem — row-major byte streams packed BUF-wide
        # (EPS = BUF/8 bytes per hex word)
        with open(case_dir / "a.mem", "w") as f:
            for w in nref.pack_act_stream(A8.reshape(-1)):
                f.write(f"{w:010x}\n")
        with open(case_dir / "b.mem", "w") as f:
            for w in nref.pack_act_stream(B8.reshape(-1)):
                f.write(f"{w:010x}\n")

        # expected_la is stored in the RTL probe layout: [K, M] transposed
        np.savez(case_dir / "expected_la.npz", la=st.logA.T.astype(np.int8))
        np.savez(case_dir / "expected_lb.npz", lb=st.logB.astype(np.int8))
        np.savez(case_dir / "expected_acc.npz",
                 acc=st.acc.astype(np.int32))

        with open(case_dir / "expected_c.mem", "w") as f:
            for v in res.C.reshape(-1):
                f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

        case = {
            "M": M, "N": N, "K": K, "R": self.R, "C": self.C,
            "BUF": self.BUF, "P": self.P,
            "n_A": int(cm.n_A), "n_B": int(cm.n_B), "n_E": int(cm.n_E),
            "passes_A": int(cm.passes_A), "passes_B": int(cm.passes_B),
            "passes_E": int(cm.passes_E),
            "xbar_passes_A": int(cm.xbar_passes_A),
            "xbar_passes_B": int(cm.xbar_passes_B),
            "xbar_passes_E": int(cm.xbar_passes_E),
            "T_A": int(cm.T_A), "T_B": int(cm.T_B), "T_E": int(cm.T_E),
            "W_A": int(cm.W_A), "W_B": int(cm.W_B),
            "T_start": int(cm.T_start),
            "balanced": bool(cm.balanced),
            "balance_residual": int(cm.balance_residual),
            "used_cycles": int(res.used_cycles),
            "weight_cycles": int(self.R * self.C),
            "serialize_cycles": int(cm.serialize_cycles),
            "load_words_a": int(len(nref.pack_act_stream(A8.reshape(-1)))),
            "load_words_b": int(len(nref.pack_act_stream(B8.reshape(-1)))),
        }
        with open(case_dir / "case.json", "w") as f:
            json.dump(case, f, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/dimm_sim.py
# Gates: values ≡ oracle, cycles measured ≡ shadow, derive/balance contract.
# ---------------------------------------------------------------------------
def _T256(passes: int) -> int:
    """Reference T(p) for the 256×256 geometry (fill 114, steady 60)."""
    return _xbar_total(passes, 256, 256, 8, 40)


def _test_constants() -> None:
    """Worked reference M=N=128, K=64 @ 256×256: work → passes → T → fill."""
    plan = ideal_pass_plan(128, 128, 64, R=256, C=256)
    cm = dimm_cycle_model(128, 128, 64, R=256, C=256,
                          n_A=1, n_B=1, n_E=128, plan=plan)
    assert (cm.work_A, cm.work_B, cm.work_E) == (8192, 8192, 1_048_576)
    assert (cm.passes_A, cm.passes_B, cm.passes_E) == (32, 32, 4096)
    assert (cm.xbar_passes_A, cm.xbar_passes_B, cm.xbar_passes_E) == (32, 32, 32)
    t32 = _T256(32)
    assert t32 == 1974                      # 114 + 31*60
    assert (cm.T_A, cm.T_B, cm.T_E) == (t32, t32, t32)
    assert (cm.W_A, cm.W_B, cm.T_start) == (1, 1, 114)   # T_start = T_fill (W=1)
    assert (cm.T_add, cm.T_reduce, cm.serialize_cycles) == (0, 0, 16384)
    assert cm.total == max(cm.T_A, cm.T_B, cm.T_add,
                           cm.T_start + cm.T_E, cm.T_reduce)
    assert cm.total == t32 + 114            # 1974 + 114 = 2088
    print("  [reference values] 128x128 K=64 worked example (+ fill): OK")


def _test_balance_reference() -> None:
    """Continuous balance law reference and K-cancellation (ratio is
    K-independent: P_A and P_E both scale with K, so P_A/P_E does not)."""
    assert balanced_counts(128, 128, 128) == (1, 1)
    for K in (1, 64):
        c = dimm_cycle_model(128, 128, K, n_A=1, n_B=1, n_E=128,
                             plan=ideal_pass_plan(128, 128, K))
        assert c.T_A == c.T_B == c.T_E, (K, c.T_A, c.T_B, c.T_E)
        assert c.total == c.T_start + c.T_E, (K, c.total, c.T_start, c.T_E)
    print("  [balance reference] balance law + K-cancellation: OK")


def _test_phase_policy() -> None:
    """overlap=True → max(T_i); overlap=False → sum(T_i)."""
    plan = ideal_pass_plan(128, 128, 64, R=256, C=256)
    ovl = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128,
                           overlap=True, plan=plan)
    seq = dimm_cycle_model(128, 128, 64, n_A=1, n_B=1, n_E=128,
                           overlap=False, plan=plan)
    assert ovl.total == max(ovl.T_A, ovl.T_B, ovl.T_start + ovl.T_E)
    assert seq.total == seq.T_A + seq.T_B + seq.T_E
    print("  [phase policy] max(T_A,T_B,T_start+T_E) vs sum: OK")


def _test_stage_ledger() -> None:
    """Stage terms are explicit: add (combinational) and reduce (fused into
    the exp drain) cost 0; the final-output serializer is M*N, outside."""
    plan = ideal_pass_plan(8, 10, 6, R=256, C=256)
    for overlap in (True, False):
        cm = dimm_cycle_model(8, 10, 6, R=256, C=256, overlap=overlap,
                              plan=plan)
        assert cm.T_add == 0 and cm.T_reduce == 0
        assert cm.serialize_cycles == 8 * 10
        if overlap:
            assert cm.total == max(cm.T_A, cm.T_B, cm.T_add,
                                   cm.T_start + cm.T_E, cm.T_reduce)
        else:
            assert cm.total == (cm.T_A + cm.T_add + cm.T_B + cm.T_E
                                + cm.T_reduce)
    print("  [stage ledger] explicit stage terms (add/reduce 0, "
          "serialize M*N): OK")


def _test_plan_and_packing() -> None:
    """Schedule-injected counts: packing changes counts, never values."""
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
    """Derive-by-default n = max(1, ceil(P_pool / ceil(P_E/n_E))):
    exact counts, residual always reported."""
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
    assert ref.T_start == 114 and ref.total == t32 + 114

    # Floor: n_E < N makes equality impossible (max(1,·) binds) — reported.
    floor = derive(128, 128, 64, 64)
    assert (floor.T_A, floor.T_B, floor.T_E) == (t32, t32, t64)
    assert not floor.balanced and floor.balance_residual == t64 - t32
    assert floor.total == 114 + t64

    # Scaling: more farm crossbars shorten the whole pipeline.
    scale = derive(128, 128, 64, 256)
    assert (scale.n_A, scale.n_B) == (2, 2)
    assert scale.balanced and scale.total == t16 + 114
    assert scale.total < ref.total < floor.total

    # Counts are K-independent (P_A and P_E both scale with K).
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


def _test_fill_windows() -> None:
    """T_start = T_fill + (ceil(W/n) − 1)·T_steady, W = ceil(len/I).

    Also covers W > 1 (M or N beyond the pass capacity) and the `ceil(W/n)`
    round division when the pool has multiple crossbars.
    """
    primitive_geometry = prim.cycle_model(1, 8, 8, 8, 40)
    t_fill, t_steady = (primitive_geometry.t_fill,
                        primitive_geometry.t_steady)
    assert (t_fill, t_steady) == (14, 10)
    # M = 20 > I = 8 -> W_A = 3 windows for column 0; N = 3 -> W_B = 1
    plan = ideal_pass_plan(20, 3, 1, R=8, C=8)
    c1 = dimm_cycle_model(20, 3, 1, R=8, C=8,
                          n_A=1, n_B=1, n_E=1, plan=plan)
    assert (c1.W_A, c1.W_B) == (3, 1)
    assert c1.T_start == t_fill + 2 * t_steady       # ceil(3/1) rounds
    c2 = dimm_cycle_model(20, 3, 1, R=8, C=8,
                          n_A=2, n_B=1, n_E=1, plan=plan)
    assert c2.T_start == t_fill + 1 * t_steady       # ceil(3/2) rounds
    # mirrored: the N side binds
    c3 = dimm_cycle_model(3, 20, 1, R=8, C=8, n_A=1, n_B=1, n_E=1,
                          plan=ideal_pass_plan(3, 20, 1, R=8, C=8))
    assert (c3.W_A, c3.W_B) == (1, 3)
    assert c3.T_start == t_fill + 2 * t_steady
    print("  [fill windows] T_start = T_fill + (ceil(W/n)-1)*T_steady: OK")


def _test_identity_pass() -> None:
    """Pass budget ceil(len/I), oracle values, cached ≡ fresh DPE."""
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
    _test_stage_ledger()
    _test_plan_and_packing()
    _test_derive_contract()
    _test_fill_windows()
    _test_identity_pass()
    _test_behavior()
    print("dimm_sim self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
