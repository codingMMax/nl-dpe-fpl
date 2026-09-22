# Spec — NL-DPE DIMM operator (v2 clean-room, Stage 4)

**Status**: v0.1 working spec, 2026-09-20. DIMM-only split of
[`pool_farm_model.md`](pool_farm_model.md) (whose DIMM sections this spec
supersedes); the softmax row-pipeline model there (§6) is **deferred** to its
own spec and is out of scope here.

**Contracts above this doc**:

- Values: `v2/oracle/dimm_ref.py` — DIMM stages `logA`, `logB`, `C` with the
  pass-structured view normative and the elementwise view kept as the dual
  witness (self-test asserts bit-equality).
- Operator passes: `dpe_nldpe.md` §6 **F5–F8** (v2.0.2) — ACAM is the
  crossbar's output stage (no standalone unit), identity conversion capacity
  `I = min(R, C)`, stride-`I` schedule, normative padding discard, int8 feed
  invariance, packing a schedule property.
- Primitive timing: `dpe_nldpe.md` §5.3 —
  `T(p) = T_fill + (p−1)·T_steady` per crossbar (p back-to-back passes).

---

## §1 Operator & value contract

```
C[m,n] = Σ_k E[m,k,n]
E      = ACAM_EXP(U)                 identity pass, MODE_EXP
U      = logA[m,k] + logB[k,n]       CLB integer add (log domain)
logA   = ACAM_LOG(A)                 identity pass, MODE_LOG
logB   = ACAM_LOG(B)
```

- `A : int8 [M,K]`, `B : int8 [K,N]`, `C : int32 [M,N]` (exact sum of the
  unsigned int8 EXP bytes; `K·255 < 2^31` for any `K ≤ 2^23`).
- Every conversion is an operator pass (F5): crossbar first, ACAM last.
  `logA`/`logB` feed the pass as int8; the farm's `U` is truncated to int8
  before the EXP pass (F7 invariance — value-neutral inside the un-clamped
  EXP range `|U| ≤ 65535`).
- The value contract is the oracle: bit-exact equality with
  `dimm_ref.dimm_stages()` is the only value criterion.

## §2 Datapath & work accounting (PF1, PF2)

```
logA pool (n_A) ──▶ LA[M,K] int8 ──┐
                                    ├─▶ exp farm (n_E) ──▶ acc[M,N] int32
logB pool (n_B) ──▶ LB[K,N] int8 ──┘     (+ CLB add, exact)
```

- Producers convert each unique operand **once** into buffers; the farm is
  the only stage that touches every `(m,n,k)`.
- Reuse theorem (PF2): `W_E : W_A : W_B = M·N : M : N`, with
  `W_A = M·K`, `W_B = K·N`, `W_E = M·N·K`.

## §3 Pass accounting — schedule-injected counts

- The schedule declares the passes it issues as a `DimmPassPlan`
  (`dimm_sim.py`): `passes_A`, `passes_B`, `passes_E`. **The cycle model
  consumes the plan; it never guesses it.**
- Ideal (packed) count: `P = ceil(W / I)` per pool. Unpacked count:
  `P = Σ_i ceil(len_i / I)` per independent vector (≥ ideal). Packing changes
  counts only; values are invariant (F8).
- `ideal_pass_plan(M, N, K, R, C)` is a convenience for sizing, not an
  assumption of the model.

## §4 Balance law & derive-by-default

**Objective (normative)**: choose integer crossbar counts `n_i ≥ 1` to
minimize the worst stage finish time

```
minimize  max_i T( ceil(P_i / n_i) )        i ∈ {A, B, E}
```

`T(·)` is §5.3; because `T` has a fill term, equalizing *pass counts* (not
just continuous work) is what equalizes finish times exactly.

**Design law (continuous optimum, PF3/PF4)**

```
same geometry :  n_A = max(1, ceil(n_E / N))     n_B = max(1, ceil(n_E / M))
general       :  n_A = max(1, ceil(n_E · (I_E·L_A) / (N · I_A·L_E)))
                 n_B = max(1, ceil(n_E · (I_E·L_B) / (M · I_B·L_E)))
```

with `I_x = min(R_x, C_x)` and `L_x = T_steady` of pool `x`'s geometry. `K`
cancels in the ratio (PF3); throughput scales with `n_E` until the `max(1,·)`
floor binds (PF4).

**Derive-by-default (exact)**: with `n_E` fixed, the farm's per-crossbar passes
are `p_E = ceil(P_E / n_E)` and `T(·)` is increasing in passes, so the exact
optimum (identical pool geometry) is

```
n_log = max(1, ceil(P_log / p_E))          log ∈ {A, B}
```

lexicographically optimal for `(max T, residual, machines)`. The continuous
design law above is the closed-form reference (PF3), within ±1 of the exact
counts. `dimm_cycle_model(..., n_A=None, n_B=None)` derives; explicit
`n_A`/`n_B` (**both or neither**) are overrides (design experiments) and must
be reported together with the residual.

**Residual reporting (always)**: the result exposes `balanced: bool` and
`balance_residual = max(T_A,T_B,T_E) − min(T_A,T_B,T_E)`. Exact equality is
not always attainable (integrality, fill overhead, `max(1,·)` floor); the
residual must be minimized, reported, and never assumed zero.

## §5 Cycle contract

```
T_pool = T( ceil(P_pool / n_pool) )          primitive §5.3, per crossbar
total  = max(T_A, T_B, T_E)                  overlapped  (default, PF5)
       = T_A + T_B + T_E                     phase-separated (precompute)
```

Packing, lane count and buffer layout never change `T(·)`; they change
`P_pool` only (F8). Buffer/accumulator bandwidth is assumed sufficient unless
declared otherwise (PF8).

## §6 Worked example (reference values)

`M = N = 128, K = 64, R = C = 256` → `I = 256`, `T(p) = 114 + (p−1)·60`:

| quantity | logA | logB | exp |
|---|---:|---:|---:|
| work | 8,192 | 8,192 | 1,048,576 |
| passes `ceil(W/I)` | 32 | 32 | 4,096 |
| balanced `n` (n_E=128) | 1 | 1 | 128 |
| per-crossbar passes | 32 | 32 | 32 |
| `T` | 1,974 | 1,974 | 1,974 |

`total = 1,974` cycles, `balanced = True`, `balance_residual = 0`.
Floor example: `n_E = 64` → `n_A = n_B = 1`, `T = (1974, 1974, 3894)`,
residual `1920` (reported, not hidden).

## §7 Scope & decisions

| # | Decision |
|---|---|
| D1 | Values: pass-structured `dimm_ref` normative; elementwise view = dual witness |
| D2 | Pass counts: schedule-injected (`DimmPassPlan`); ideal formula is a convenience |
| D3 | Balance: derive-by-default from `n_E` (law + discrete objective); overrides must report residual |
| D4 | Timing: primitive `T(p)`; overlapped = `max`, phase-separated = `sum` |
| D5 | **Softmax is out of scope** here (its row-pipeline model stays in `pool_farm_model.md` §6 until split off) |
| D6 | Multi-output "different matrices side-by-side" trick: **out of scope** (not needed by DIMM/softmax conversion) |
| D7 | Values are invariant to pools/lanes/packing (PF7); oracles are the value contract |

Carried design decisions PF1–PF9 from `pool_farm_model.md` (PF2 reuse theorem,
PF3 balance/K-cancellation, PF4 scaling floor, PF5 phase policy, PF8 buffer
bandwidth advisory, PF9 identity pass budget).
