# Spec — NL-DPE DIMM operator (v2 clean-room, Stage 4)

**Status**: v0.2 working spec, 2026-09-22. DIMM-only split of
[`pool_farm_model.md`](pool_farm_model.md) (whose DIMM sections this spec
supersedes); the softmax row-pipeline model there (§6) is **deferred** to its
own spec and is out of scope here.

Revision history:
- **v0.2 (2026-09-22)**: normative **schedule mapping** (§2.1: rank-1 outer
  product over `k`, A column-major / B row-major, operands converted once into
  LA/LB, farm re-reads) + **block prefix gate** (§3) + the **producer→farm
  fill** `T_start` in the cycle contract (§5). The Stage-4 RTL
  (`v2/rtl/dimm_top.v`) is gated to this contract exactly (window spans equal
  `T(p)`, serializer = `M·N` words, no residual constants).
- **v0.1 (2026-09-20)**: DIMM-only split; pass contract references
  `dpe_nldpe.md` §6 F5–F8 (v2.0.2).

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

### §2.1 Schedule mapping (normative)

The realized schedule is the rank-1 **outer product** over `k`:

```
for k = 0 .. K-1:    C += A[:,k] ⊗ B[k,:]        (accumulated in k order)
```

- **Production order**: A is produced **column-major** (k-major, m-minor) and
  B **row-major** (k-major, n-minor), so both streams are k-major and block
  `k`'s operand data is the stream prefix of length `(k+1)·M` (A) /
  `(k+1)·N` (B).
- **Convert once, re-read**: producers write `LA[M,K]` / `LB[K,N]` (int8, log
  domain) exactly once per unique element; the farm re-reads them via the
  contiguous per-block ranges `[k·M,(k+1)·M)` and `[k·N,(k+1)·N)`. Conversion
  work is `M·K` / `K·N` — buffering is what makes the PF2 reuse ratio true
  (without it the log pools would re-convert `N` resp. `M` times).
- **Block `k`** contributes the partial sum `term_k(m,n)` to every output
  element; the reduction is incremental (accumulator RMW in `k` order), so
  the final `C` exists only after block `K−1`.
- **Window** = one identity pass = `I = min(R,C)` elements of a flat element
  stream (A stream `M·K`, B stream `K·N`, farm plane `M·N` per block). A
  window is the atomic unit of work: one primitive pass (`T_fill` for the
  first, `T_steady` back-to-back). Windows may straddle k-slices — packing is
  a schedule property (F8/P28).
- **Parallelism**: `n_A`/`n_B` crossbars over windows of the operand streams;
  `n_E` crossbars over windows of the `(m,n)` plane of the current block; `k`
  is sequential. The parallelism unit is the window, never a matrix
  row/column.

## §3 Pass accounting — schedule-injected counts

- The schedule declares the passes it issues as a `DimmPassPlan`
  (`dimm_sim.py`): `passes_A`, `passes_B`, `passes_E`. **The cycle model
  consumes the plan; it never guesses it.**
- Ideal (packed) count: `P = ceil(W / I)` per pool. Unpacked count:
  `P = Σ_i ceil(len_i / I)` per independent vector (≥ ideal). Packing changes
  counts only; values are invariant (F8).
- `ideal_pass_plan(M, N, K, R, C)` is a convenience for sizing, not an
  assumption of the model.
- **Block availability (prefix gate)**: after `q` producer windows the
  converted stream prefix is `q·I` elements, so block `k` becomes available
  when
  ```
  a_wins >= ceil((k+1)·M / I_A)   and   b_wins >= ceil((k+1)·N / I_B)
  ```
  (completed-window counters; prefix exactness holds because the producer
  crossbars share geometry and start together). The gate uses
  `ceil((k+1)·M/I)`, not `(k+1)·ceil(M/I)`, because windows straddle slices.
- **First-slice window counts**: `W_A = ceil(M / I_A)`, `W_B = ceil(N / I_B)`
  are the windows covering the k=0 slices (column 0 / row 0) — they set the
  farm's fill latency `T_start` (§5).

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
T_pool  = T( ceil(P_pool / n_pool) )         primitive §5.3, per crossbar
T_start = max over pools p of                producer→farm fill (true data
            T_fill_p + (ceil(W_p / n_p) - 1)·T_steady_p     dependency)
          W_A = ceil(M / I_A),  W_B = ceil(N / I_B)
total   = max(T_A, T_B, T_start + T_E)       overlapped  (default, PF5)
        = T_A + T_B + T_E                    phase-separated
```

`T_start` is the earliest cycle at which the farm's block 0 can read its
operands: the first `W_A`/`W_B` producer windows must have **drained** (§2.1),
each pass costing `T_fill` (first) or `T_steady` (back-to-back), spread over
`n_p` crossbars (`ceil(W_p/n_p)` rounds). It is a data dependency, not
implementation overhead: a farm that starts earlier would read incomplete
LA/LB. For `M,N ≤ I` (the usual case) `W_A = W_B = 1` and
`T_start = T_fill_pool`.

Packing, lane count and buffer layout never change `T(·)`; they change
`P_pool` only (F8). Buffer/accumulator bandwidth is assumed sufficient unless
declared otherwise (PF8).

**Output stage (reported separately)**: after the last farm window the
reducer serializes `M·N` int32 C words, one per cycle (`serialize_cycles` in
the case dump). It is not part of `total`; the runtime gate is
`measured = total + serialize_cycles`.

## §6 Worked example (reference values)

`M = N = 128, K = 64, R = C = 256` → `I = 256`, `T(p) = 114 + (p−1)·60`:

| quantity | logA | logB | exp |
|---|---:|---:|---:|
| work | 8,192 | 8,192 | 1,048,576 |
| passes `ceil(W/I)` | 32 | 32 | 4,096 |
| balanced `n` (n_E=128) | 1 | 1 | 128 |
| per-crossbar passes | 32 | 32 | 32 |
| `T` | 1,974 | 1,974 | 1,974 |

Fill: `W_A = ceil(128/256) = 1`, `W_B = 1` →
`T_start = max(114 + 0·60, 114 + 0·60) = 114`.
`total = max(1974, 1974, 114 + 1974) = 2,088` cycles (was 1,974 under v0.1's
fill-free contract), `balanced = True`, `balance_residual = 0`.

Floor example: `n_E = 64` → `n_A = n_B = 1`, `T = (1974, 1974, 3894)`,
`T_start = 114`, `total = 114 + 3894 = 4,008`; residual `1920` (reported, not
hidden).

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
| D8 | Schedule mapping (normative, §2.1): rank-1 outer product over `k`; A column-major / B row-major; operands converted once into LA/LB, farm re-reads (PF2) |
| D9 | Cycle contract includes the producer→farm fill `T_start` (true data dependency): `total = max(T_A, T_B, T_start + T_E)`; the serializer is reported separately (`M·N` words) |
| D10 | Window = one identity pass of `I = min(R,C)` elements; block availability is the prefix gate `ceil((k+1)·M/I)` / `ceil((k+1)·N/I)` (§3) |

Carried design decisions PF1–PF9 from `pool_farm_model.md` (PF2 reuse theorem,
PF3 balance/K-cancellation, PF4 scaling floor, PF5 phase policy, PF8 buffer
bandwidth advisory, PF9 identity pass budget).
