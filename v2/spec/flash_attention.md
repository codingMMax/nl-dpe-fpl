# FlashAttention on NL-DPE — log-carry form

**Status**: v0.1, 2026-09-30. **Spec-only** (oracle / sim / RTL deferred).
Binds the verified **log-carry flash-attention recurrence** to the NL-DPE
integer/ACAM contract and composes it from certified bricks. The recurrence is
carried by `software` in `v2/oracle/flash-attention-log-domain.ipynb` (float
golden: identical to textbook attention to ≈4e-15 over 500 randomized cases,
arbitrary block splits).

**Contracts above this doc**:

- Primitive values/timing: `v2/spec/dpe_nldpe.md` (§5.3, §6 F3/F5–F8).
- Softmax stage: `v2/spec/softmax_online.md` (blocked, deferred-α).
- Matmul bricks: `v2/spec/dimm.md` (DIMM, upstream), `v2/spec/gemm.md` (GEMM).
- Float golden witness: `v2/oracle/flash-attention-log-domain.ipynb` — a
  **witness, not a contract** (it is float; the integer contract below is normative).

Revision history:

- **v0.1 (2026-09-30)**: first charter. Log-carry dataflow; NL integer binding;
  composition from DIMM + `softmax_online`; DIMM sign-path impact; cycle/energy
  contract vs full-row; FA-level verification plan.

---

## §1 The log-carry recurrence (reference dataflow)

The carried state is **entirely log-domain**: `(m, log l, log|o|, sign(o))`.
Every `×`/`÷` is a log-domain `+`/`−` plus exactly one ACAM `exp`/`log`; there are
**no multipliers and no dividers**. The moves:

- `log α = m_old − m_new` — α is *never* exponentiated on its own; it is carried as
  `log α` (a peripheral subtract).
- `log|P| = s − m_new` — free (a peripheral subtract); it feeds **both** the
  denominator (`l`) chain and the P·V chain.
- denominator `log l′ = log(exp(log α + log l) + Σp)`.
- output `o′ = exp(log α + log|o|)·sign(o) + Σ_b exp(log|P| + log|V_b|)·sign(V_b)`,
  then `log|o′| = log|o′|`, `sign(o′) = sign(o′)`.
- epilogue `O = exp(log|o| − log l)·sign(o)` (division → subtract + exp).

### Per-op ledger (the spec seed)

| node | op | engine | inputs (shape) | output (shape) |
|---|---|---|---|---|
| PA1 | `+` | periph | `log\|Q\| [Bq,d]` · `log\|Kb\| [Bkv,d]` | `log\|qk\| [Bq,Bkv,d]` |
| PE1 | `exp` | ACAM | `log\|qk\| [Bq,Bkv,d]` | `\|q·k\| [Bq,Bkv,d]` |
| PX1 | `⊕` | periph | `sign(Q) [Bq,d]` · `sign(K) [Bkv,d]` | `sign(qk) [Bq,Bkv,d]` |
| PS1 | `Σ_d` | crossbar | `\|q·k\|` · `sign(qk)` | `s [Bq,Bkv]` |
| RM | `max` | periph | `s [Bq,Bkv]` | `m_blk [Bq]` |
| NM | `max` | periph | `m [Bq]` · `m_blk [Bq]` | `m_new [Bq]` |
| LS | `−` | periph | `m [Bq]` · `m_new [Bq]` | `log α [Bq]` |
| LP | `−` | periph | `s [Bq,Bkv]` · `m_new [Bq]` | `log\|P\| [Bq,Bkv]` |
| EP | `exp` | ACAM | `log\|P\| [Bq,Bkv]` | `p [Bq,Bkv]` |
| PS2 | `Σ` | crossbar | `p [Bq,Bkv]` | `Σp [Bq]` |
| LA | `+` | periph | `log α` · `log l [Bq]` | `log α + log l [Bq]` |
| EA | `exp` | ACAM | `log α + log l [Bq]` | `α·l [Bq]` |
| LB | `+` | periph | `α·l` · `Σp [Bq]` | `l′ [Bq]` |
| LG | `log` | ACAM | `l′ [Bq]` | `log l′ [Bq]` |
| PV1 | `+` | periph | `log\|P\| [Bq,Bkv]` · `log\|Vb\| [Bkv,d]` | `log\|P·V\| [Bq,Bkv,d]` |
| PV2 | `exp` | ACAM | `log\|P·V\| [Bq,Bkv,d]` | `\|P·V\| [Bq,Bkv,d]` |
| PV3 | `⊕` | periph | `sign(V) [Bkv,d]` (sign P ≡ +) | `sign(PV) [Bq,Bkv,d]` |
| PV4 | `Σ_Bkv` | crossbar | `\|P·V\|` · `sign(PV)` | `o_part [Bq,d]` |
| OA | `+` | periph | `log α` · `log\|o\| [Bq,d]` | `log α + log\|o\| [Bq,d]` |
| OE | `exp` | ACAM | `log α + log\|o\| [Bq,d]` | `α·o [Bq,d]` |
| OB | `+` | periph | `α·o × sign(o)` · `o_part` | `o′ [Bq,d]` |
| OG | `log` | ACAM | `o′ [Bq,d]` | `log\|o′\| [Bq,d]` |
| SN | `sgn` | periph | `o′ [Bq,d]` | `sign(o′) [Bq,d]` |
| ES | `−` | periph | `log\|o\|` · `log l [Bq]` | `log\|O\| [Bq,d]` |
| EE | `exp` | ACAM | `log\|O\| [Bq,d]` | `O [Bq,d]` |

Carries between KV blocks: `m` (linear), `log l`, `log|o|`, `sign(o)`.

## §2 NL integer binding

Each engine maps to an existing NL element — **no new primitive is required**:

| engine | NL binding |
|---|---|
| `exp` (PE1/EP/EA/PV2/OE/EE) | `ACAM_EXP`: `trunc8(1 + y + ⌊y²/2⌋)`, exponent input clamped per §2.1 |
| `log` (LG/OG) | `ACAM_LOG`: `trunc8(y − 1)` — **range caveat §2.2** |
| `+ / −` | peripheral integer add/sub |
| `max` (RM/NM) | peripheral signed max |
| `⊕ / sgn` | peripheral sign XOR / sign bit |
| `Σ_d`, `Σ_Bkv`, `Σ` | crossbar exact int reduce (the certified DIMM reduce) |

### §2.1 EXP range / clamp

The FA exponent is `s − m_new ≤ 0`, so `ACAM_EXP` sees `y ≤ 0`; the current
operator additionally clamps `d = max(s − m, −128)` (`softmax.md` §1). The same
clamp applies to PE1/PV1 inputs before the ACAM.

### §2.2 EXP/LOG range caveat (measured)

`ACAM_LOG(y) = trunc8(y − 1)` is only meaningful for `y` near 1, so the notebook's
`log l′` (over a wide `l′`) is **not** directly feedable to the ACAM; the carried
normalizer must use the two representations of `softmax_online.md` (compact
`lq`/`ls` or wide int32).

Likewise `ACAM_EXP(y) = 1 + y + ⌊y²/2⌋` is a polynomial magnitude table, not `exp`.
Per the online-softmax decision (`softmax_online.md` **v0.3**), the FA uses the **same
ACAM mapping**: the elementwise `exp`/`log` nodes (PE1/EP/PV2/OE/EE, LG/OG) are the
ACAM forms, matching `softmax_ref`. The online folded state is **bit-identical to
`softmax_ref` only at the block extremes** (`Bkv ∈ {1, S}`); intermediate blocks are
offset-skewed (measured; `softmax_online.md` §4/§10). The FA inherits this mapping;
the numerical budget of the full log-carry chain vs the exact flash recurrence is a
verification-phase item.

## §3 Composition

```
Q,K,V (int8, signed)
  → producers: (log|Q|, sign Q), (log|K|, sign K), (log|V|, sign V)      [DIMM producers]
  → QK^T   : DIMM brick + sign extension   (PA1,PX1,PS1 via U=logA+logB, ACAM_EXP, signed reduce)
  → softmax: softmax_online.md             (blocked, deferred-α)         [block brick]
  → P·V    : DIMM brick + sign extension   (PV1,PX1,PV4 via logA=log|P|, logB=log|V|, signed reduce)
  → epilogue: O = exp(log|o| − log l)·sign(o)                            [peripheral + ACAM]
```

- The softmax brick returns the **unnormalized** `log|P| = s − m` (equivalently the
  surrogate `eb`) for the P·V activation side, and the running `l`; the final `o/l`
  is the FA epilogue (`softmax_online.md` §1/§2), not a separate normalizer stage.
- Certified-brick **reuse map**: QK^T and P·V are the DIMM operator (`dimm.md`) —
  same `U = logA + logB → ACAM_EXP → exact reduce` structure — extended with signs
  (§4). The softmax is `softmax_online.md`.
- Producer→farm fill `T_start` follows `dimm.md` §5, with the softmax's per-block
  completion as the producer reference.

## §4 DIMM sign-path impact (does DIMM need to change?)

The certified DIMM contract (`dimm.md` §1) is **unsigned int8 magnitudes** with an
exact int32 sum of unsigned EXP bytes. FA needs **signed** products:

- QK^T: `sign(qk) = sign(Q) ⊕ sign(K)` (PX1).
- P·V : `sign(p·v) = sign(V)` since `p ≥ 0` (PV1/PV3).

**Change required**: add `signA`/`signB` operand bits to the DIMM producers, form
the product sign by XOR, and reduce over signed values (magnitude-split exact
int32: `Σ|pos| − Σ|neg|`). Value contract delta: a signed oracle
(`dimm_ref` + sign). Interface delta: operand sign bits + signed reduce output.
Cost delta: a small control/logic addition, **not** a new crossbar pass.

**Alternatives**: (a) keep unsigned magnitudes and carry an explicit sign pass;
(b) defer per-block contributions and combine at the end (`O(B)·M·N` storage);
(c) accept signed int8 operands directly in the crossbar feed.

**Verdict**: **DIMM change is required for FA.** It is **not** required for the
online softmax alone (the DIMM there only consumes the softmax's int8 output).

## §5 Cycle / energy contract

- Per block: QK^T (DIMM `T(p)`, with `T_start`), softmax EXP passes, P·V (DIMM
  `T(p)`), plus the combine and epilogue. Sub-`K` block sizes are the innermost
  loop; `Σ_d` and `Σ_Bkv` are the crossbar reductions.
- **Comparison contract vs full-row** (deferred to RTL): same shapes / corpus;
  report Δ cycles and Δ energy. As the notebook states, flash bounds the
  **working set** (`O(B+d)` vs `O(S)`) and does **not** cut attention compute:
  `≈ 2·S·d + S` exps either way. The claim to test is the footprint / fused
  streaming win, not a compute reduction.

## §6 Verification plan (later phases)

- **Golden witness** (not contract): the float log-carry notebook vs textbook
  attention.
- **Integer oracle** `flash_attention_ref.py`: the same dataflow with
  `ACAM_EXP/ACAM_LOG`, both representations (`softmax_online.md` §3), signed DIMM
  reduce; must reproduce `softmax_ref` on the softmax stage and a textbook
  attention witness on `O` under the integer contract.
- **GATE 1/2** reuse per brick (DIMM signed variant gets its own GATE 1).
- **Sign-path certification**: signed-DIMM values ≡ the signed elementwise oracle.

## §7 Decisions

| # | decision |
|---|---|
| F1 | Reference dataflow = the notebook's **log-carry** recurrence (no multipliers/dividers) |
| F2 | Elementwise `exp`/`log` use the **ACAM** forms (same mapping as `softmax_ref`), per `softmax_online.md` v0.3; add/xor/reduce are peripheral |
| F3 | **Unnormalized** `log\|P\| = s − m` feeds P·V; final `o/l` is the epilogue |
| F4 | QK^T and P·V **reuse the certified DIMM brick** with a sign extension |
| F5 | Per-block max/rescale follows `softmax_online.md` (**deferred α**, ACAM) |
| F6 | Softmax stage is **bit-exact to `softmax_ref` at `Bkv ∈ {1, S}`**; intermediate `Bkv` are a reported approximation |
| F7 | The float notebook is a **golden witness, not a contract** |
| F8 | `Bkv` and `Bq` are schedule parameters (values invariant, F8 of the primitive) |

## §8 Risks & decision gates

- **Integer `log` range** (§2.2) may force the wide carry (Rep-2) for `log l`/`log|o|`.
- **Sign-path cost**: signed reduce may add logic/energy; gate = signed-DIMM oracle
  certification + a cost estimate.
- **Deferred-α exactness under the surrogate**: gate = certify Rep-1 ≡ Rep-2 on the
  corpus (`softmax_online.md` §3); deviation is the reported approximation budget.
- **`Bkv` sensitivity**: block size trades carry state against combine cost; sweep
  in the oracle phase.
