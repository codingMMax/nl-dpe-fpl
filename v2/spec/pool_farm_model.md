# Pool/Farm Model — NL-DPE DIMM & softmax operator parallelism

> **Superseded for DIMM**: the DIMM content of this document (§§1–5, 7–9) is
> now normative in [`v2/spec/dimm.md`](dimm.md) (2026-09-20), which owns the
> value/pass/pacing/balance/cycle contracts. Only **§6 (softmax
> row-pipeline)** is retained here until the softmax spec is split off.

**Status**: working spec, 2026-09-15. Applies to the v2 operator layer
(`v2/sim/dimm_sim.py`, `v2/sim/softmax_sim.py`).

**Contracts above this doc**:

- Values: `v2/oracle/dimm_ref.py` and `v2/oracle/softmax_ref.py` (bit-exact).
- Primitive timing: `v2/spec/dpe_nldpe.md` §5.3 —
  `T(p) = T_fill + (p−1)·T_steady` per crossbar pass.
- Primitive geometry: `I = min(R, C)` elements convertible per identity pass,
  `P = 8` bit-slices, `BUF = 40` bit port (5 bytes/cycle).

This doc replaces the "dedicated datapath per output element" mental picture
with a producer/consumer (pool/farm) model that explains crossbar counts, the
exp/log work asymmetry, and the throughput-balancing rule.

---

## 1. Datapath

For `C[M,N] = A[M,K] @ B[K,N]` in the log domain:

```
 logA pool (n_A)   ──▶  LA[M,K] int8  ──┐
                                        ├─▶  exp farm (n_E)  ──▶  acc[M,N] int32
 logB pool (n_B)   ──▶  LB[K,N] int8  ──┘     + CLB add            (exact sum)
```

Per output element `(m,n)` the logical inner loop over `k` is:

```
u = LA[m,k] + LB[k,n]        (CLB integer add, log domain)
e = ACAM_EXP(u)              (identity crossbar, MODE_EXP)
acc[m,n] += e                (reduction is AFTER exp — exp is nonlinear)
```

- **Producers** convert each unique operand value exactly once into the
  buffers: `logA = trunc8(A − 1)`, `logB = trunc8(B − 1)` (identity pass + ACAM
  LOG, `dpe_nldpe.md` P24).
- **Buffers** decouple production from consumption (LA, LB are `M·K` and `K·N`
  bytes; e.g. 8 KB each at 128×64).
- **Farm** is the only stage that touches every triple `(m,n,k)`; it reads the
  buffers, adds, exps, accumulates.
- **Accumulators** are `M·N` int32 partial sums.

The per-output datapath remains the logical picture; the *physical* resources
(pools, buffers) are shared. There is no per-element log conversion.

## 2. Work accounting & the reuse theorem

| work | expression | reuse factor |
|---|---|---|
| `W_A` logA conversions (unique) | `M·K` | read `N` times each |
| `W_B` logB conversions (unique) | `K·N` | read `M` times each |
| `W_E` exp conversions | `M·N·K` | **1** (nothing reuses an exp) |

The farm issues `M·N·K` reads of each buffer, but only `M·K` / `K·N` unique
values exist; the pool creates each once, the farm re-reads it from the buffer.

**PF2 (reuse theorem)** — the conversion work ratio is

```
W_E : W_A : W_B  =  M·N : M : N
```

*Correction to earlier docs*: FIDELITY_METHODOLOGY §5 states the exp phase
dominates "by factor K". The general factor is `M·N/(M+N)`; for square `M=N`
that is `M/2`. It equals `K` only when `K = M·N/(M+N)` (e.g. `M=N=128, K=64`,
where the shorthand happens to be exact).

## 3. Rate matching (derivation)

Per-cycle rates contain no matrix dimensions — they are hardware only:

```
r_A = n_A · I / L_pass        LA elements/cycle     (L_pass = T_steady)
r_B = n_B · I / L_pass        LB elements/cycle
r_E = n_E · I / L_pass        exp ops/cycle
```

`M, K, N` enter through total work; times are `T_i = W_i / r_i`, so:

```
T_A = M·K   · L / (n_A·I)
T_B = K·N   · L / (n_B·I)
T_E = M·N·K · L / (n_E·I)
```

Equalizing finish times (`T_A = T_E`, then `T_B = T_E`):

```
n_A = n_E / N            n_B = n_E / M          (same geometry for all pools)
```

`M`, `K`, `L`, `I` all cancel. General form when pools have different
geometries:

```
n_A = n_E · (1/N) · (I_E · L_A) / (I_A · L_E)
```

**PF3 (K-cancellation)** — the balance ratio is independent of `K`; `K` only
scales absolute time. **PF4 (design law)** — provision
`n_A = ceil(n_E/N)`, `n_B = ceil(n_E/M)`, minimum 1 each; likewise,
throughput scales linearly with `n_E` until a floor (SRAM bandwidth, CLB, or
integer rounding) binds.

Phase policy:

```
overlapped (streaming, buffered):   total = max(T_A, T_B, T_E) + fill
phase-separated (precompute logs):  total = T_A + T_B + T_E
```

Equalizing finish times minimizes total time under either policy; only the
`ceil`/min-1 effects differ.

## 4. Worked examples

### 4.1 Toy — where reuse comes from

`M=2, N=3, K=2`:

```
unique logs: a00 a01 a10 a11 (4 = M·K)   b00..b12 (6 = K·N)
exps:        12 = M·N·K
reuse:       log(a00) used by all 3 outputs of row 0        → N = 3
             log(b00) used by both outputs of column 0     → M = 2
             exp(a00+b00) used once                        → 1
```

On-demand (no reuse): `12 + 12 + 12 = 36` conversions.
Pooled: `4 + 6 + 12 = 22` conversions. The gap grows with `M` and `N`.

### 4.2 Full — `M=N=128, K=64, R=C=256` (`I=256`, `L=60`)

| quantity | logA | logB | exp |
|---|---:|---:|---:|
| work (elements) | 8,192 | 8,192 | 1,048,576 |
| passes `ceil(work/I)` | 32 | 32 | 4,096 |
| balanced crossbars `n` | 1 | 1 | 128 |
| crossbar passes `ceil(passes/n)` | 32 | 32 | 32 |
| `T` = primitive `T(32)` | 1,974 | 1,974 | 1,974 |

`T(32) = T_fill + 31·T_steady = 114 + 31·60 = 1,974` cycles. Overlapped total
= 1,974 cycles with **130 crossbars** — versus `3·M·N = 49,152` under the
no-reuse dedicated-datapath picture.

### 4.3 The FIDELITY §7 lane mapping as a special case

`W=16` lanes, `C=128` (`I=128`), sequential phases (`4 + 64 + 512 = 580`
passes per lane):

- primitive `T(580)` at `R=256, C=128` (`T_fill=88`, `T_steady=60`) ≈ **34,828
  cycles**.
- pool/farm with `n_E=16` (one exp crossbar per lane), balanced logs:
  `max(T(512), T(64), T(64))` = `T(512)` ≈ **30,748 cycles**.
- pool/farm balanced at `n_E=128`: `T(64)` ≈ **3,868 cycles** (~9×).

The lane mapping is valid, but throughput-unbalanced: the exp farm is the only
stage worth scaling, and its serialized log phase is avoidable by overlap.

## 5. Design-space notes

- **Square crossbars are best for pools/farm**: `I = min(R,C)`, but the pass
  loads `R` bytes regardless. `R=C=256` moves 4.27 elem/cycle; `R=256, C=128`
  moves 2.13 elem/cycle.
- **Integer floor**: `ceil` and `max(1, ·)` mean small `n_E` over-provisions
  the log pools; they then idle. Pools can be shared across DIMM instances.
- **Buffer bandwidth is a possible cap**: the farm pulls 2 operands per exp
  (2·r_E reads/cycle) plus accumulator read-modify-write (~2·r_E accesses).
  BRAM ports can bind before crossbars do.
- **Failure mode to avoid**: scaling only the log pools (or only `n_A`) leaves
  the farm as the cap; scaling only the farm starves it if buffers cannot be
  filled ahead (needs double-buffering).
- **Geometry changes the ratio**: use the general `n_A` formula when pools and
  farm use different `(I, L)`.

## 6. Softmax — streaming row-pipeline model (distinct machine)

Softmax is **not** modeled with the pass-gated pool/farm timing. It is a
row-pipeline stage machine with dedicated **streaming** converters (measured,
locked in `softmax_study/SOFTMAX_STUDY.md` §1/§4):

```
row pipeline per lane (16 lanes, rows {k, k+16, ...}, RPL = S/16):
  A: max tree (WPR)
  B: (x−max) clamped, ACAM EXP via n_exp DPEs, 5 elem/cycle/port (LCYC)
  Cs: unsigned byte sum + shared log DPE (20)
  D: clamp(x−max−ls) → log-domain int8 (WPR)
  steady = max(WPR, LCYC);  fill = (WPR+4)+(LCYC+10+LCYC+2)+20+WPR+4
  total  = fill + (RPL−1)·steady
```

**PF6 (rationale)** — softmax's converters are fed continuously row-by-row
(rows overlap), so the binding rate is the 5 elem/cycle port, not the
pass-gated `LOAD+P` interval: `steady = 26` cycles/row for `S=128` streaming,
versus 34 under the pass-gated primitive (≈30% slower). This model is locked
to measured RTL and its anchors: `290` (`S=128, n_exp=1`), `514`
(`S=256, n_exp=2`), `956` (`S=256, n_exp=1`).

Producer/consumer reading of the same structure (explains why the log stage
never binds):

| work | expression |
|---|---|
| exp | `S²` (one per element) |
| log | `S` (one per row sum) |

exp : log = `S : 1`, so `n_log/n_exp ≈ 1/S`. The study's 1 shared log DPE per
16 lanes is over-provisioned by `S/(16·n_exp)` (8× at `S=128, n_exp=1`) — which
is exactly why the Cs/log occupancy of 20 cycles never binds.

## 7. Composition note (attention)

Softmax emits **log-domain** values, and the S×V operator's inner loop needs
`log(attn) + log(V)`. Therefore the S×V DIMM's attenuation-side logB pool is
not needed when fed by softmax (its A-side producer *is* softmax's output);
only the V-side log pool runs. This is the log-domain fusion already assumed by
the attention mapping.

## 8. Decisions & assumptions

| # | Decision |
|---|---|
| PF1 | Physical model = producers (log pools) → buffers → exp farm → accumulators; per-output datapath is logical only |
| PF2 | Reuse theorem: `W_E : W_A : W_B = M·N : M : N`; FIDELITY's "factor K" corrected to `M·N/(M+N)` |
| PF3 | Balance ratio `n_A = ceil(n_E/N)`, `n_B = ceil(n_E/M)`; independent of `K` |
| PF4 | Throughput scales with `n_E`; log pools scale `1/N`, `1/M`, minimum 1 |
| PF5 | DIMM timing = pass-gated primitive `T(passes)`; overlap policy configurable (`max` vs `sum`) |
| PF6 | Softmax timing = streaming row-pipeline model (study-locked), **not** pass-gated |
| PF7 | Values are invariant to pools/lanes/packing; oracles are the value contract |
| PF8 | Buffer/accumulator bandwidth assumed sufficient unless declared otherwise (advisory check) |
| PF9 | Identity pass budget: `I = min(R,C)`, `ceil(elements/I)` passes, zero-pad to `R`; primitive unchanged, wrapper owns padding. Pass counts are **schedule-injected** (`DimmPassPlan` in `dimm_sim.py`); the normative pass layer is `dpe_nldpe.md` §6 F5–F8 (capacity, stride, padding discard, int8 feed invariance, packing) |

## 9. TODO

- **User behavior**: fill `NldpeDimm.run_matmul()` and `NldpeSoftmax.run()`
  TODO blocks; self-tests are gated until then.
- Later: reconcile softmax's streaming steady rate with the pass-gated
  primitive if/when the primitive gains a streaming mode.
- Later: attention composition (QK^T → softmax → S×V) reusing §7.
