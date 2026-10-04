# Softmax — NL-DPE blocked (online) operator (deferred-α carry)

**Status**: v0.3, 2026-09-30. **Oracle + behavior model implemented** (ACAM mapping);
RTL deferred. Normative for the *online* (key-block streamed) softmax and for its
embedding in flash-attention (`flash_attention.md`).

**Supersedes**: `softmax.md` §6 **S6** ("online / FlashAttention softmax is a
different operator, out of scope"). Online is now in scope.

**v0.3 correction (important)**: the online operator uses the **ACAM** exp/log forms
— the **same op mapping as `softmax_ref` / `softmax_sim`** (no `np.exp`/`np.log`/
float division). It is **bit-identical to `softmax_ref` at `Bkv ∈ {1, S}`** and a
distinct, offset-skewed approximation for intermediate `Bkv` (see §4). The v0.2
"exact dense softmax via full-precision/wide exp" framing is **withdrawn** (that path
survives only as an `exact` reference witness).

**Contracts above this doc**:

- Values: `v2/oracle/softmax_online_ref.py` — `softmax_online_model` (the **ACAM
  contract**), `softmax_online_exact` / `regular_exact` (reference witnesses),
  `softmax_online_global`, `softmax_online_pass_counts`.
- Behavior/timing: `v2/sim/softmax_online_sim.py` — `NldpeSoftmaxOnline`,
  `softmax_online_cycle_model`.
- Shipped operator: `v2/oracle/softmax_ref.py` (`NldpeSoftmax` in
  `v2/sim/softmax_sim.py`); the `Bkv = S` case.
- Primitive timing: `v2/spec/dpe_nldpe.md` §5.3 — `T(p)`.
- Operator pass layer: `v2/spec/dpe_nldpe.md` §6 F5–F8 — capacity `I = min(R, C)`,
  stride-`I` windows, padding discard, packing a schedule property.
- ACAM forms: `exp(y) = trunc8(1 + y + ⌊y²/2⌋)`, `log(y) = trunc8(y − 1)`.
- Composition: `v2/spec/flash_attention.md`.

Revision history:

- **v0.3.1 (2026-10-03)**: RTL cycle alignment — `N_FAC` parallel factor bank
  (round-robin windows, `gen_fac`; `n_fac` = sim sweep knob), sum-unit
  **direct per-row commit** (`DIRECT_COMMIT=1`, online only; conventional
  60/60 unchanged — inertness proof), combine gate reduced to
  `all_exp_done && all_fac_done`, LOG starts at `combine_done` (d1 stage
  dropped). **GATE 2 64/64, `Δ_impl = 0`**; sim + pins untouched.
- **v0.3 (2026-09-30)**: contract switched to the **ACAM** mapping (same as
  `softmax_ref`); oracle/sim rewritten to use only ACAM exp/log; extremes
  (`Bkv ∈ {1, S}`) shown bit-identical to `softmax_ref`; intermediate-`Bkv`
  divergence recorded.
- **v0.2 (2026-09-30)**: (withdrawn) exact/full-precision rescale framing.
- **v0.1 (2026-09-30)**: first charter (withdrawn contract).

---

## §1 Contract — the ACAM streamed softmax (G1′)

Per row `r` (S a power of two; rows independent); keys streamed in `B = S/Bkv`
blocks. **Every transcendental is the ACAM form — the same op mapping as
`softmax_ref` / `softmax_sim`** (no `np.exp`, no `np.log`, no float division):

```
per block b : m_b    = max_{j in block b} s[r,j]
              eb_j   = ACAM_EXP(max(s[r,j] - m_b, -128))       (unsigned byte)
              Sp_b   = sum_j eb_j                              (int32)
m_r          = max_b m_b                                       (global row max)
factor_b     = ACAM_EXP(max(m_b - m_r, -128))                  (unsigned)
L_r          = sum_b factor_b * Sp_b                           (deferred-alpha, int32)
lq_r         = min(L_r >> log2 S, 127)                         (the /S, as softmax_ref)
ls_r         = ACAM_LOG(lq_r) = trunc8(lq_r - 1)
out[r,j]     = clamp(s[r,j] - m_r - ls_r, -128, 127)
```

- `ACAM_EXP(y) = trunc8(1 + y + ⌊y²/2⌋)`, `ACAM_LOG(y) = trunc8(y − 1)`; `log(l/S)`
  is realized exactly as in `softmax_ref` (`L >> log2 S` then `ACAM_LOG`).
- **Extremes are exact.** At `Bkv = 1` and `Bkv = S` the result is **bit-identical to
  `softmax_ref`**: one block ⇒ `factor = ACAM_EXP(0) = 1`, `L = Sp_0 = output_sum`;
  single-key blocks ⇒ `Sp_b = 1`, `factor_b = ACAM_EXP(s − m)`, and the sum equals
  the global-max sum. (`softmax_ref` is the `Bkv = S` case.)
- **Intermediate `Bkv` differ** from `softmax_ref` (see §4).

## §2 Machine model — key-block streaming, deferred α

- Stream blocks of `Bkv` keys (`B = S/Bkv`, block-major; block = `[S, Bkv]`).
- Per block: the per-row block max `m_b`, then ACAM_EXP weights, then `Sp_b`.
- Carry `(m, L)`; merge `L = Σ_b ACAM_EXP(m_b − m) · Sp_b` (deferred α, one end
  merge); normalizer via `lq = L >> log2 S` and `ACAM_LOG`.
- No per-block rescale of a running accumulator.
- Storage: one key-block buffer + `O(B)` carry, versus the shipped `S²` buffer.

## §3 Value paths

`v2/oracle/softmax_online_ref.py`:

| path | definition | role |
|---|---|---|
| `exact` | float64 block-local max + rescale, true `exp` | reference witness (`== textbook softmax`); **not** the hardware contract |
| `model` | **ACAM** streamed realization (§1 op mapping) | the contract; `NldpeSoftmaxOnline` reproduces it **bit-exactly** |
| `global` | shipped `softmax_ref` (global max, ACAM) | the `Bkv = S` special case |

Measured (S=128, `Bkv=16`, logits `[−16,16]`): `model` vs `softmax_ref` within-row
`rel ≈ 20`, offset ≈ 119, clamp 48%; at `Bkv ∈ {1, S}` the two are **bit-equal**.

## §4 Shift-invariance limit (why intermediate `Bkv` differ)

`ACAM_EXP` is a polynomial magnitude table, not `exp`, so the deferred-α factor
`ACAM_EXP(m_b − m)` **cannot** undo a block-local max: `ACAM_EXP(s − m_b)` scaled by
it is not `ACAM_EXP(s − m)`. Hence the ACAM online operator equals `softmax_ref` only
at `Bkv ∈ {1, S}`. `out_j = s_j − m − (row constant)` still holds, so the
**distribution is preserved except where the offset drives values onto the ±128
rails** (measured 39–99% clamped at `Bkv=16`, versus 1–2% for `softmax_ref`). A
true-exponential rescale (the `exact` reference) would be shift-invariant but is not
the ACAM hardware — recorded as the open point.

## §5 Pass counts

```
EXP   : per key block, `S*Bkv` elements packed stride-I
        -> ceil(S*Bkv / I) windows / block;  passes_exp = B * that
LOG   : the S per-row boundary normalizers -> ceil(S / I)
combine: one log-sum-exp merge over B terms per row
```

Pass counts and issue order depend on `Bkv`; with the ACAM mapping the **values also
depend on `Bkv`** (§4) — only `Bkv ∈ {1, S}` reproduce `softmax_ref`. Counts are
schedule-injected (`SoftmaxOnlinePassPlan`).

## §6 Cycle contract

- Primitive `T(p)` per crossbar; the blocked machine streams block-major at
  `clb_width` bytes/cycle, converts each block (`schedule_pass_sequence` semantics),
  then combines and logs.
- **Measured:** `NldpeSoftmaxOnline` (`compute_cycles` = last normalizer ready =
  **whole result computed**; the final result is read combinationally and is
  **not drained/serialized**). Example (S=128, `Bkv=16`): `n_exp=1` → 4436 cyc,
  `n_exp=4` → 3586; the full-row `NldpeSoftmax` computes in 4008. **The online
  win is storage (`O(S·B)` vs `S²`), not throughput** — the measured comparison
  is the contract.

## §7 RTL mapping (`v2/rtl/softmax_online_top.v`)

## §7 RTL mapping (`v2/rtl/softmax_online_top.v`)

- **Parametric** (mirrors `softmax_top`): `S`, `R`, `C`, `BUF`, `P`, `BKV`,
  `N_EXP`, `N_LOG`, `N_FAC` (factor bank width, i.e. parallel factor
  crossbars; 1 ⇒ one serialized feed); derives `B=S/BKV`, `I=min(R,C)`, `SQ`,
  `BLKEL=S*BKV`, `STREAM=ceil(BLKEL/EPS)`, `NWB=ceil(BLKEL/I)`.
- **Shared primitives**: `softmax_online_top` reuses the *unified*
  `softmax_wprog` / `softmax_exp_feed` / `softmax_sum_unit` from `softmax_top.v`
  (compiled together). `softmax_exp_feed` is parameterized by `ROW_STRIDE`
  (0⇒`S`; online = `BKV`), `NELEM` (0⇒`S*S`), and a `win_span` input;
  `softmax_sum_unit` by `RS`/`NROWS`/`SPAN`/`OUT_LQ` (online: raw `Sp`) plus
  `DIRECT_COMMIT`.
- **Factor bank**: `N_FAC` parallel `softmax_exp_feed` instances (`gen_fac`,
  mirrors the `gen_exp` bank): windows dealt round-robin
  (`window j -> crossbar j mod N_FAC`, matching `packed_windows`), each feed
  serializing its share back-to-back, all gated by `m_ready`. `N_FAC=1`
  collapses to a single serialized feed (the sim's default is `n_fac = n_exp`).
- **Sum unit commit**: with `DIRECT_COMMIT=1` (online only; the conventional
  top keeps `=0` and its certified path), a row's `Sp` total
  `Σ_p (bank_p[r] + bank_add_p[r])` is written to `sp_q` at the cycle its
  element count completes — out-of-order, no row-order pipe; the per-crossbar
  `dc_*` write bus commits up to `N_EXP` rows/cycle (windows are disjoint, so
  no arbitration). Value contract unchanged: `sp_q` holds the same totals.
- **Combine gate**: `combine_fire = all_exp_done && all_fac_done` (one-shot) —
  i.e. the cycle after the **later** of the two passes' last drain words. With
  the direct commit, every `sp_q`/`factor_q` write completes at that posedge,
  so the combine reads committed tables one cycle later with margin (no
  bypass). `sp_last_q`/`fac_last_q` remain as observability regs, not gates.
- **Online-only modules**: `online_blk_loader` (BUF-wide block-major stream →
  block-contiguous store, per-block de-pad, `blk_loaded`/`store_avail`),
  `online_max_unit` (streaming fold → per-block `blk_max`/`blk_ready`),
  `online_combine` (combinational: `L = Σ_b factor_b·Sp_b`,
  `lq = min(L>>log2 S,127)`), and `online_out_unit` (combinational final result
  `clamp(score − m − ls)`, read via `out_addr`/`data_out`, no drain).
- **Score stream**: `BUF`-wide words, 1/cycle, block-major (row-major within a
  block), each block zero-padded to `STREAM` words; loaded during the run
  (`stream_bytes = BUF/8`).
- Frozen probes: `blk_max_q[B*S]`, `sp_q[B*S]`, `factor_q[B*S]`, `L_q[S]`,
  `lq_q[S]`, `ls_q[S]`, `m_q[S]`, and the `out_addr`/`data_out` result read.
- Structural timings: block-max latency `LOG2S+1`; combine `ceil(log2 B)+1`;
  log `2`; plus a fixed structural prologue (`prologue = 6`) in the sim's
  `_schedule` so the modelled schedule matches the DUT (GATE 2 Δ = 0).

## §8 Decisions

| # | decision |
|---|---|
| O1 | Contract = **ACAM streamed softmax (G1′)**: block-local max + `ACAM_EXP` weights + deferred-α `ACAM_EXP(m_b−m)` + `>>log2S` / `ACAM_LOG` — the **same mapping as `softmax_ref`** |
| O2 | Key-block streaming; one-block buffer + `O(B)` carry; no `S²` buffer |
| O3 | **Deferred α**: per-block ACAM partials + one end merge; no per-block rescale |
| O4 | Sim is **bit-exact to the `model` (ACAM) path**; `model == softmax_ref` at `Bkv ∈ {1, S}` |
| O5 | `Bkv` is a schedule parameter (`Bkv \| S`); with the ACAM the **values depend on `Bkv`** (not invariant) |
| O6 | `done` = whole result computed (last normalizer ready); the final result is read combinationally (`out_addr`/`data_out`), no `S²` drain |
| O7 | **All exp/log are ACAM** (no `np.exp`/`np.log`/float division); the ACAM is not shift-invariant, so intermediate `Bkv` are a distinct approximation |
| O8 | The `exact`/true-exp path is kept as a **reference witness only**, not the contract |

## §9 Verification status (2026-10-03)

- `python3 v2/oracle/softmax_online_ref.py` — **ALL PASS**: `exact ≡ textbook`,
  `model` ACAM ranges + determinism, **`model(Bkv=S)` ≡ `softmax_ref` bit-exactly**,
  budgets vs `softmax_ref`/exact, pass counts, extremes.
- `python3 v2/sim/softmax_online_sim.py` — **ALL PASS**: bit-exact vs the ACAM
  `model` oracle across S ∈ {32,128}, `Bkv ∈ {4,16,32,128}`, `n_exp ∈ {1,2,4}`;
  **`Bkv=S` == full-row `NldpeSoftmax` bit-exact**; corridor; cycles-vs-full-row.
- **GATE 2** (`v2/rtl/softmax_online_top.v` + `v2/tb/tb_softmax_online_top.v`
  + `v2/smoke/run_softmax_online_rtl.py`) — **64/64 PASS, `Δ_impl = 0`
  (2026-10-03)**, after the `N_FAC` parallel factor bank + direct-commit sum +
  `all_exp_done && all_fac_done` gate + one-stage-earlier LOG start: all probes
  (`blk_max`, `blk_Sp`, `factor`, `L`, `lq`, `ls`, `m`) and the final result
  (combinational `out_addr`/`data_out`) **bit-exact**, `measured == compute_cycles`
  on every case (frozen pins 525/4436/3586/7300 hit exactly).
  Corpus: S ∈ {16,32,64}, `BKV ∈ {8,16,32}`, `n_exp ∈ {1,4}`, classes
  {random, extremes, uniform, ties}; `N_FAC = n_exp` (the sim default); cases
  generated/certified by `v2/smoke/gen_softmax_online_cases.py` (GATE 1).
- **Shared-sum-unit inertness (2026-10-03)**: the conventional softmax
  (`v2/rtl/softmax_top.v`, which compiles the same `softmax_sum_unit`) re-ran
  **60/60 PASS, `Δ_impl = 0`** after the `DIRECT_COMMIT` edit — the Stage-3
  certified corpus is unaffected.

## §10 Comparison — conventional vs online softmax (2026-09-30)

`python3 v2/smoke/compare_softmax.py [--csv out.csv]` — both machines on identical
scores/geometry (R=C=256). Both now use **ACAM** exp/log; the online adds the
block-local max + deferred-α factor. Gate per row: `offset` = per-row additive
constant (arbitrary in log-domain softmax); `rel` = within-row error; `L1` = max
per-row L1 of the reconstructed distribution vs exact `p`; `clamp%` = outputs at
the ±128 rails. (Online rows use `Bkv=16`.)

### A. Values (deviation from the exact dense softmax)

`p̂` = distribution reconstructed from the int8 log output (`row softmax of exp(out8)`).

| S | class | score range | conv offset | conv rel | conv L1(p̂,p) | conv clamp% | online offset | online rel | online L1 | online clamp% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | small | [−16,16] | 111 | 11 | 0.0000 | 1.29% | 119 | 20 | 1.6732 | 48.3% |
| 128 | full | [−128,127] | 35 | 131 | 1.8891 | 95.4% | 35 | 131 | 1.8898 | 98.6% |
| 256 | small | [−16,16] | 105 | 6 | 0.0000 | 0.29% | 117 | 18 | 1.6306 | 56.6% |
| 256 | full | [−128,127] | 25 | 131 | 1.9354 | 96.0% | 25 | 132 | 1.9354 | 99.0% |

- **Conventional is distribution-exact on a realistic logit range** (`L1 = 0.0000`):
  its large arbitrary offset is harmless and the `rel` (6–11) sits on
  negligible-probability tail elements. Full-range it saturates.
- **Online (ACAM, `Bkv=16`) is offset-skewed**: the deferred-α factor
  `ACAM_EXP(m_b−m)` inflates `L`, capping `lq` and pushing the row down, so **48–99%**
  of outputs clamp and `L1 ≈ 1.6–1.9`. It is **not** the regular softmax for
  intermediate `Bkv`; it is **bit-equal to `softmax_ref` only at `Bkv ∈ {1, S}`**.
- **Attribution**: with the ACAM, only the *row offset* changes (the within-row
  structure `s_j − s_k` is exact), so the divergence is entirely the offset shifting
  values onto the int8 rails.

### B. Cycles (identical scores/geometry; `conv` = full-row machine; **no-drain
### convention 2026-10-03**: the output stage is not timed — `e2e = load +
### compute` for `conv`, `e2e = compute` for `online` (stream overlapped))

| S | Bkv | n_exp | n_log | conv load | conv comp | conv e2e | online comp (=e2e) | Δe2e | online load (in-stream) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 1 | 1 | 3277 | 4008 | 7285 | 4436 | −2849 | 3280 |
| 128 | 16 | 2 | 1 | 3277 | 2096 | 5373 | 3706 | −1667 | 3280 |
| 128 | 16 | 4 | 2 | 3277 | 1152 | 4429 | 3586 | −843 | 3280 |
| 128 | 32 | 1 | 1 | 3277 | 4008 | 7285 | 4845 | −2440 | 3280 |
| 128 | 32 | 4 | 2 | 3277 | 1152 | 4429 | 3705 | −724 | 3280 |
| 128 | 128 | 1 | 1 | 3277 | 4008 | 7285 | 7300 | +15 | 3277 |
| 128 | 128 | 4 | 2 | 3277 | 1152 | 4429 | 4420 | −9 | 3277 |
| 256 | 16 | 1 | 1 | 13108 | 15556 | 28664 | 16368 | −12296 | 13120 |
| 256 | 16 | 2 | 1 | 13108 | 7884 | 20992 | 13789 | −7203 | 13120 |
| 256 | 16 | 4 | 2 | 13108 | 4060 | 17168 | 13549 | −3619 | 13120 |
| 256 | 32 | 1 | 1 | 13108 | 15556 | 28664 | 17186 | −11478 | 13112 |
| 256 | 32 | 4 | 2 | 13108 | 4060 | 17168 | 13779 | −3389 | 13112 |
| 256 | 256 | 1 | 1 | 13108 | 15556 | 28664 | 28652 | −12 | 13108 |
| 256 | 256 | 4 | 2 | 13108 | 4060 | 17168 | 17132 | −36 | 13108 |

- `n_log > 1` is an idle axis at `R=C=256` (`S ≤ I` ⇒ `passes_log = 1`; S4).
- **Flipped vs the 2026-09-30 table** (that table counted the removed output
  drain on the conv side: `used = comp + S²·drain`): with the output stage
  untimed, the online machine is **faster end-to-end** at intermediate `Bkv`
  (streaming overlaps the block-max/EXP/factor chain), converging to **parity
  at `Bkv = S`** (the full-row degenerate case, +12..15 prologue cycles).
  Compute-only, the conventional machine is faster (its EXP chain is shorter;
  the online pays block-max + factor + deferred-α combine). Cycles are the
  certified sim numbers (`Δ_impl = 0` on the GATE-2 corpus; S=128/256 online
  rows are the same frozen sim, pins 4436/3586/7300).
- The comparator's `conv_used` column still reports the legacy drain-inclusive
  makespan (`result.used_cycles`); the contract columns are `comp` / `e2e`.

### C. Pass counts + score-side storage

| S | Bkv | blocks | conv passes e/l | online passes e/l | conv store (B) | online store (B) | ratio |
|---|---:|---:|---|---|---:|---:|---:|
| 128 | 16 | 8 | 64/1 | 64/1 | 16768 | 7168 | 2.34× |
| 128 | 32 | 4 | 64/1 | 64/1 | 16768 | 6656 | 2.52× |
| 128 | 128 | 1 | 64/1 | 64/1 | 16768 | 17024 | 0.98× |
| 256 | 16 | 16 | 256/1 | 256/1 | 66304 | 24576 | 2.70× |
| 256 | 32 | 8 | 256/1 | 256/1 | 66304 | 18432 | 3.60× |
| 256 | 256 | 1 | 256/1 | 256/1 | 66304 | 66816 | 0.99× |

- Pass counts are identical here because `Bkv | I` (no per-block padding); the
  online counts can exceed the packed total when `Bkv ∤ I`.
- **The online win is storage**: 2.3–3.6× less at `Bkv ∈ {16,32}`, and it vanishes
  at `Bkv = S` (0.98×) — the buffered special case.

### Verdict

With **ACAM exp/log** (same mapping as `softmax_ref`), the online operator is
**bit-identical to `softmax_ref` only at `Bkv ∈ {1, S}`**; for intermediate `Bkv`
the deferred-α factor cannot reconstruct the global-max normalizer (the ACAM is not
shift-invariant), so it is a distinct, offset-skewed approximation that clamps
48–99% of outputs. With the output stage untimed (no-drain convention), the
online machine is **faster end-to-end** at intermediate `Bkv` (streaming overlap)
and ~parity at `Bkv = S`; compute-only the conventional machine is faster. The
material structural advantage remains **storage** (2.3–3.6× less at
`Bkv ∈ {16,32}`). The RTL reproduces the sim **bit-exactly with `Δ_impl = 0`** (§9).

Gates in `compare_softmax.py`: `conv ≡ softmax_ref`, `online ≡ model` (bit-equal);
oracle self-test: `model(Bkv=S) ≡ softmax_ref` bit-equal + budgets; GATE 2:
`softmax_online_top` values bit-exact + `measured == compute_cycles`.
