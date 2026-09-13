# Spec v1.1 (AMENDED) — NL-DPE primitive (`dpe_nldpe`, v2 clean-room)

**Status**: **v1.1 amended 2026-09-12** (from v1.0 FROZEN 2026-08-29) —
decision points P1–P19 closed. v1.1 changes the numeric contract to the fp32
dataflow: fp32 stationary weights, fp32 crossbar output, structural fp32 MAC
(§6 F2), and functional-then-truncate ACAM (§6 F3) — see P14–P19 (§10).
Amendments require a new revision (v1.x) + decision-log entry + oracle
re-transcription.
**Precedence**: this spec > v2 RTL ≡ Python oracle > legacy RTL (witness).
**Normative rule**: every choice the oracle makes must exist here first; if
oracle and spec disagree, the spec is wrong until fixed.

Revision history:
- **v1.1 (2026-09-12)**: fp32 dataflow amendment — T3/T4/T5 rewritten;
  weight programming via WEIGHT strobe (one fp32 word/cycle) replaces the
  v1.0 weight byte stream (P17, supersedes P8); ACAM = functional form then
  `trunc8` (P16, supersedes P4/P5); EXP_FN/LOG_FN defined now (P18, closes
  P6); fp32 arithmetic corner-case policy fixed (P19); MAC = structural fp32
  sequence (P15).
  - **same-day correction**: F2 shift semantics fixed to partial-shift
    (`y ± 2^b·p_b`) — the accumulator does **not** shift (P20); the
    v1.0/v1.1 `2·y ± p` wording was incompatible with identity invariant I7.
- **v1.0 (2026-08-29)**: initial frozen charter (P1–P13).

Sources: project-lead session input 2026-08-29 (pipeline description, output
buffer + per-column ACAM, precision contract, port-sharing and compute-term
closures); project-lead decisions 2026-09-12 (fp32 dataflow, trunc8, weight
strobe, EXP/LOG forms, fp32 corner-case policy). Legacy generated RTL
consulted only for conformance questions, never as a source of requirements.

---

## §1 Block definition

One NL-DPE = **crossbar + ACAM**, with three internal storage structures and
**two physically separate external memory interfaces** (P12: one input-side,
one output-side; LOAD and OUTPUT never contend for a port).

```
 ext.mem(in) ──40b/cyc──▶ INPUT BUFFER ──1 bit-slice/cyc──▶ CROSSBAR ──▶ SHIFT&ACC
                           (R × 8 b, sliced)   (R bits)     (R×C fp32 W)   │ ▲ pipelined (fp32)
                                                                         ▼ │
                           OUTPUT BUFFER ◀── ACAM ×C units (1 cyc, parallel) ┘
                           (C × 8 b) ──40b/cyc──▶ ext.mem(out)
```

| Parameter | Meaning | Default / example |
|---|---|---|
| `R` | crossbar rows = input-vector length | 256 |
| `C` | crossbar cols = output width | 512 (reference); 256 (legacy cross-check) |
| `BUF` | external port width, both sides | 40 → 5 bytes/cycle |
| `P` | input precision (bit-slices per vector) | 8 |

**Out of scope**: analog noise, device nonlinearity, multi-DPE arrays, pass
scheduling (wrapper owns scheduling; this block enforces only its own
readiness rules), and all **external quantizers** — any upstream precision
(int16/int32/fp32) is quantized to int8 *before* the activation stream is
presented (A16); the block itself contains no input-quantizer. The block
computes the **structural fp32 behavior** of §6 F2 — ground truth assumes
noise is already taken care of elsewhere.

## §2 Data types & precision contract

- **T1 (strict)**: crossbar input activations are **int8** (2's complement).
- **T2 (strict)**: ACAM output is **int8** per column (output buffer + stream).
- **T3**: crossbar output / ACAM input is **fp32** (IEEE-754 binary32),
  produced by the structural sequence of §6 F2. The sequence and its rounding
  are normative; no wider intermediate is exposed.
- **T4**: weights are **fp32** (IEEE-754 binary32) values, programmed once per
  workload through the WEIGHT strobe (§4.2) and stationary thereafter (A5).
- **T5 (idealization)**: the analog MAC is modeled as the **structural fp32
  arithmetic of §6 F2** with an ideal (noiseless) read-out; the mathematical
  intent is `y[c] = Σ_r W[r,c]·x[r]`, and F2 defines the exact fp32 rounding
  order that is normative for verification.

## §3 Storage organization

| Structure | Size | Organization |
|---|---|---|
| Weight storage | `R·C × 32` bits | fp32 (binary32) word per (r, c); programmed once per workload (P17) |
| Input buffer | `R × P` bits (**single** instance — P1) | **bit-sliced**: bank `b` holds bit b of every row; byte-major writes via corner-turn (§4.3) |
| Output buffer | `C × 8` bits (**single** instance) | one int8 per column; written wholesale by the ACAM stage (1 cycle), drained 5 bytes/cycle |
| Accumulators | `C × 32` bits | fp32 (T3); shift&acc pipeline stage |

**P1 (closed)**: input buffer is **single** with in-place refresh. A new
input burst may begin refilling only **after the MSB fire of the in-flight
pass has completed** (all P banks consumed). Earlier refill would corrupt
un-fired slices (a byte-major write touches all banks at once).
**P12 (closed)**: external interface = **two physically separate memory
buffers/ports** (in / out). LOAD and OUTPUT never contend.

## §4 External interfaces — exact VTR port surface (normative)

Port names/widths are **identical to the VTR blackbox contract**
(`vtr/dpe_blackbox.v`, arch XML `<model name="dpe">`). No new external
signals exist in v2. v2 assigns semantics as follows:

| Port | Dir | v2 semantics |
|---|---|---|
| `clk`, `reset` | in | clock; synchronous active-high reset (A10) |
| `data_in[39:0]` | in | ACT payload (5 bytes/cycle, byte i in bits `8i+7:8i`) **or** one fp32 weight word (WEIGHT strobes: `[31:0]` = IEEE-754 binary32, `[39:32]` ignored) |
| `w_buf_en` | in | **ACT burst strobe**: present + accept one ACT word |
| `load_input_reg` | in | **WEIGHT strobe**: present + accept one fp32 weight word (`data_in[31:0]`) |
| `nl_dpe_control[1:0]` | in | **ACAM mode**: 00=REGULAR, 01=ACTIVATION, 10=EXP, 11=LOG (§6) |
| `shift_add_control` | in | reserved, tied 0, documented |
| `shift_add_bypass` | in | reserved, tied 0, documented |
| `load_output_reg` | in | reserved, tied 0, documented |
| `MSB_SA_Ready` | out | **input-refill permit**: high when the input buffer may be refilled (after reset; during a burst; from the cycle after MSB fire of an in-flight compute). Low while a full vector awaits/undergoes compute. |
| `data_out[39:0]` | out | result stream (5 bytes/cycle, §4.5) |
| `dpe_done` | out | 1-cycle pulse after the last output byte of a pass |
| `reg_full` | out | output-busy: high from ACAM write until drain completes (back-pressure observable) |
| `shift_add_done` | out | reserved observability: 1-cycle pulse at MSB shift&acc completion |
| `shift_add_bypass_ctrl` | out | reserved, driven 0 |

Verification-only state (fp32 `y` per column, §6 F2) is **not a port** — the
TB reads it hierarchically (revised D8; port surface stays pristine).

### 4.2 Weight programming (WEIGHT strobes) — one-time per workload

- One fp32 weight word per strobe cycle; `WR_CYC = R·C` (one-time, **excluded**
  from the per-pass formulas of §5.3). 256×512 → **131 072**; 256×256 → 65 536.
- **Order (closed, P17): row-major, row-outer** — word #k carries
  `W[k/C][k mod C]` (all C columns of row 0 first, then row 1, …).
- Weights persist across arbitrarily many passes (I6); re-programming only
  between workloads. Programming is **not** part of the ACT/output streaming
  contract: it completes before the first pass.

### 4.3 Activation input stream (ACT strobes) — per pass

- `LOAD_CYC = ⌈R·8/40⌉` (R=256 → **52**).
- Byte #j carries `x[j]` (activation for crossbar row j); 5 bytes/cycle.
- **Corner-turn (internal, fixed wiring)**: `bank[b][j] ← bit b of x[j]`,
  b = 0..P−1.

### 4.4 Readiness contract

`MSB_SA_Ready` is the sole throttle (A1): the wrapper may present ACT strobes
only while it is high; violations are discarded-and-flagged (TB assertion),
never silently corrupted. The block accepts a burst only in full (R bytes).

### 4.5 Output stream — per pass

- `OUTPUT_CYC = ⌈C·8/40⌉`. C=256 → **52**; C=512 → **103**.
- Byte #k carries the ACAM int8 result of column k (column order), gapless,
  starting the cycle after the pass's ACAM write; `dpe_done` after the last
  byte. `data_out` is idle (zero) outside drains.

## §5 Pipeline behavior & cycle contract

### 5.1 Stage latencies (derived, no constants)

| Stage | Latency | Value |
|---|---|---|
| WEIGHT programming | `WR_CYC = R·C` | one-time, before the first pass; excluded from `T_fill`/`T_steady` |
| LOAD (ACT burst) | `LOAD_CYC` | ⌈R·8/40⌉ |
| CROSSBAR fires | P | 8 (one bit-slice per cycle, LSB→MSB); per-slice partial = fp32 row sum (§6 F2) |
| SHIFT&ACC | pipelined | fp32 partial-shift (§6 F2): each slice partial enters at `2^b·p_b` (exact scale); MSB accumulate completes 1 cycle after MSB fire |
| ACAM | 1 | C units parallel, once per pass, after MSB shift&acc; functional form → trunc8 (§6 F3) |
| OUTPUT drain | `OUTPUT_CYC` | ⌈C·8/40⌉ |

`COMPUTE_CYC := P + 2 = 10` (fires + MSB-acc drain + ACAM) — closed (a), 2026-08-29.

### 5.2 Independence rules (closed: P1 single buffer, P11 strict output ordering)

| Transition | May begin when |
|---|---|
| ACT burst pass k+1 | cycle after pass k's **MSB fire** — `MSB_SA_Ready` rises (§4.4) |
| CROSSBAR pass k+1 | input buffer holds a complete new vector **and** accumulator free (freed by pass k's MSB shift&acc) |
| ACAM pass k+1 | pass k+1's MSB shift&acc done **and** pass k's output drain fully complete (strictly after last `out_valid` cycle — P11) |
| OUTPUT drain pass k | cycle after pass k's ACAM write |

LOAD and OUTPUT run concurrently by default (P12: separate ports).

### 5.3 Cycle formulas (normative)

```
WR_CYC      = R·C                              (fp32 weight words, one-time,
                                                excluded from per-pass)
T_fill      = LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC
T_steady    = max( LOAD_CYC + P ,               ← input-buffer bound (P1: refill starts P cycles into compute)
                   COMPUTE_CYC ,                ← compute bound
                   OUTPUT_CYC + 1 )             ← output-buffer bound (P11: strict ordering)
Total(M)    = T_fill + (M−1) · T_steady          (architectural)
measured(M) = Total(M) + Δ_impl                  (Δ_impl: one implementation-
                                                 declared constant, invariant
                                                 across M ∈ {1,2,4,8}, modes)
```

| Config | LOAD_CYC | COMPUTE_CYC | OUTPUT_CYC | T_fill | T_steady |
|---|---|---|---|---|---|
| 256×256 (legacy cross-check) | 52 | 10 | 52 | **114** | **60** |
| 256×512 (reference) | 52 | 10 | 103 | **165** | **104** |

Weight programming (`WR_CYC`, one-time) is reported separately and does not
enter `T_fill`/`T_steady`.

### 5.4 Worked timeline (256×256, M = 2)

```
cycles:  0         51 52    59 60   61  62        113 111 112  121 122     173
         [ACT burst 0]  [fires]  ▲ [ACAM] [OUT 0]    [ACT burst 1] [fires][ACAM][OUT 1]
                                 MSB                                            ▲ done
                                 fire @59 → MSB_SA_Ready @60
```

- ACT burst 1 starts at 60 (MSB fire at 59) — overlaps ACAM 0 and OUT 0 (P12).
- COMPUTE 1 starts at 112 (burst complete at 111; accumulator free since 60).
- ACAM 1 at 121 > OUT 0 last cycle 113 ✓ (P11 needs no extra wait here).
- **Total(2) = 174 = T_fill + T_steady**. (Legacy, double-buffered: 168 — a
  documented *differing witness*, §10 P1.)

## §6 Functional semantics

Notation: `fl32(·)` = IEEE-754 binary32 operation with round-to-nearest-even
(RNE), no fused multiply-add (FMA), gradual underflow (no FTZ); `2^b·v`
denotes exact power-of-two scaling (exponent adjustment; A17/P19).

- **F1 (mathematical intent)**: `y[c] = Σ_{r=0}^{R-1} W[r,c] · x[r]` over
  fp32 weights and int8 activations.
- **F2 (normative fp32 sequence)**: fires LSB→MSB, one slice per cycle.
  Per slice `b = 0..P−1`, per column `c`, the crossbar partial is the fp32
  running sum in ascending row order:

  ```
  s := +0.0
  for r = 0 .. R-1:  s := fl32( s + ( bit_b(x[r]) ? W[r,c] : +0.0 ) )
  p_b[c] := s
  ```

  Shift&acc adds each slice partial at its own significance — the *partial*
  is scaled by `2^b` (exact), the accumulator does **not** shift (P20);
  signed two's-complement MSB subtract (P2):

  ```
  y := +0.0
  for b = 0 .. P-2:  y := fl32( y + 2^b · p_b[c] )
  y := fl32( y − 2^(P-1) · p_{P-1}[c] )
  ```

  `y` is T3's crossbar output. COMPUTE_CYC must emerge structurally (§5.1),
  not from a hold-counter. F1 and F2 agree when every summation is exact;
  F2 is normative for verification.
- **F3 ACAM modes** (`nl_dpe_control`, sampled at compute start, stable until
  `dpe_done`; all modes: 1 cycle, C units parallel, int8 out). The output rule
  is the same for all modes — **functional form first, then `trunc8`**
  (P16: truncate toward zero, saturate to int32, keep the low byte):

  ```
  trunc8(z): t := trunc_toward_zero(z), clamped to [-2^31, 2^31-1]
             out8 := t[7:0]                     (two's-complement low byte)
  ```

| code | name | semantics |
|---|---|---|
| 00 | REGULAR | `out8 = trunc8(y)` (supersedes P4) |
| 01 | ACTIVATION | `out8 = trunc8(relu(y))`, relu(y) = y if y > 0 else +0.0 (supersedes P5) |
| 10 | EXP | `out8 = trunc8(EXP_FN(y))` with `EXP_FN(v) = fl32(1 + fl32(v + fl32(0.5·fl32(v·v))))`; evaluation order is normative (P18) |
| 11 | LOG | `out8 = trunc8(LOG_FN(y))` with `LOG_FN(v) = fl32(v − 1)` (P18) |

## §7 Assumption register (closed 2026-08-29 unless noted)

| # | Assumption |
|---|---|
| A1 | No back-pressure except `MSB_SA_Ready` (input side) and `reg_full` (output observability); output sink always ready |
| A2 | One DPE instance per module; arrays are Stage 2 composition |
| A3 | Output stream gapless, column order; pass k's drain completes before pass k+1's ACAM (P11) |
| A4 | Every pass streams exactly R bytes; no partial passes |
| A5 | Weights fixed for the whole workload; re-program only between workloads |
| A6 | int8 = signed 2's complement throughout |
| A7 | `dpe_done` = 1-cycle pulse after last output byte emitted |
| A8 | Compute auto-starts when a complete burst has landed; wrapper paces via `MSB_SA_Ready` |
| A9 | Input buffer single-instance; refill permitted from the cycle after MSB fire (P1) |
| A10 | Reset: synchronous, active-high `reset`; clears readiness, FSM, counters; weights/substrates undefined until programmed |
| A11 | ACAM mode sampled at compute start, stable until `dpe_done` |
| A12 | Weights are fp32 values programmed via the WEIGHT strobe (§4.2) and stationary for the workload |
| A13 | Dequant / scale folding / requantize-to-int8 belong to the mapping layer (Stage 5) |
| A14 | Noise / device nonlinearity excluded — structural-fp32 idealization (T5) |
| A15 | External interface = two physically separate memory buffers/ports (P12); LOAD and OUTPUT never contend |
| A16 | Activations arrive int8; quantization of any higher upstream precision (int16/int32/fp32) to int8 is external and not modeled |
| A17 | fp32 arithmetic = IEEE-754 binary32, RNE, no FMA, gradual underflow; NaN/Inf are outside the supported stimulus space (their appearance is a stimulus-construction error, asserted against) |
| A18 | ACAM is the only in-block quantizer: functional form → trunc8 (§6 F3); no other rounding or clamping exists in the block |

## §8 Invariants (independently checkable)

- **I1 Functional**: ACAM int8 outputs and the hierarchical fp32 `y` (§6 F2)
  bit-exact vs oracle for all §9 stimulus classes, all modes {00, 01, 10, 11}.
- **I2 Cycles**: `measured(M) = T_fill + (M−1)·T_steady + Δ_impl`, Δ_impl
  invariant across M ∈ {1,2,4,8}, modes {00,01,10,11}; `WR_CYC = R·C` one-time,
  reported separately.
- **I3 Readiness honesty**: `MSB_SA_Ready` per §4.4; no ACT byte accepted while
  low; bursts accepted only in full.
- **I4 Mode isolation**: mode change affects the next pass only.
- **I5 Stream integrity**: pass-k outputs unaffected by pass-(k+1) ACT arrival
  timing within the readiness contract.
- **I6 Stationarity**: weights persist across arbitrarily many passes.
- **I7 Layout conformance**: ACT/output byte k lands exactly per §4.3/4.5
  (identity-weights check: `y = x` exactly — int8 values are fp32-exact under
  F2; `out[c] = trunc8(x[c]) = x[c]`).
- **I8 Port-surface conformance**: module port list identical to
  `vtr/dpe_blackbox.v`; no new external signals.

## §9 Verification contract

- **Oracle**: NumPy, written from this spec only. Implements the F2 fp32
  sequence and F3 `trunc8` from first principles (no RTL mimicry). Generates
  stimulus files (fp32 weights per §4.2, int8 activations per §4.3), expected
  hierarchical fp32 `y`, expected output bytes, expected cycle counts (§5.3).
- **fp32 sub-primitive check**: the RTL fp32 add/mul cores are verified
  bit-exact against NumPy float32 vectors *before* DPE-level checks (option A,
  2026-09-12).
- **Stimulus classes**: identity W (I7); random fp32 W over a range of
  exponents (both signs, zeros; no NaN/Inf per A17); x ~ uniform int8; signed
  extremes (−128/+127, all-zero rows); M ∈ {1,2,4,8}; modes {00, 01, 10, 11}.
- **Three-way agreement**: v2 RTL ≡ oracle (bit-exact: hierarchical fp32 `y`
  and output bytes; cycles vs §5.3 + Δ_impl) on identical stimulus files;
  legacy RTL runs the same files as a third witness. Legacy divergence
  expected at T_steady (60 vs 52), the weight interface (int8 backdoor), ACAM
  semantics (low-byte/ReLU on int32), and all fp32 values; reported, not gated.
- **Clean-room rule**: legacy RTL read-only; never consulted while writing v2
  RTL.

## §10 Decision log

| # | Decision | Closure |
|---|---|---|
| P1 | Input buffering | **Single** buffer, refill gated on MSB fire; legacy (double) = differing witness. Collaborator-confirmed single 2026-09-12 |
| P2 | MSB slice | **Subtract** (2's-complement MSB) |
| P3 | Internal precision | fp32 accumulator — superseded by P14 |
| P4 | REGULAR mode | Low byte mod-256 — **superseded by P16** |
| P5 | ACTIVATION mode | ReLU on int32, then low byte — **superseded by P16** |
| P6 | EXP/LOG | Deferred to DIMM stage — **closed by P18** |
| P7 | Output path | 40-bit stream, C bytes column-order, gapless, after ACAM |
| P8 | Weight interface | Byte stream via `load_input_reg` — **superseded by P17** |
| P9 | Geometry | R, C independent; reference 256×512; cross-check 256×256 |
| P10 | Compute term | **COMPUTE_CYC = 10** (8 fires + MSB-acc drain + ACAM) — closed (a), 2026-08-29 |
| P11 | Output overlap | Forbidden (strict after last `out_valid`) → OUTPUT_CYC+1 bound |
| P12 | External ports | **Separate in/out ports, two physically separate memory buffers** — closed (b) |
| P13 | Port surface | Exact VTR blackbox names; semantics per §4 mapping; no new signals |
| P14 | Weight & internal precision | **fp32** weights per cell; crossbar output / accumulator fp32 (T3/T4, 2026-09-12) |
| P15 | MAC semantics | Structural fp32 sequence F2: row-ascending slice sums; LSB→MSB partial-shift (each `p_b` enters at `2^b`); RNE; no FMA |
| P16 | ACAM output | Functional form first, then `trunc8` (truncate toward zero, saturate int32, low byte); supersedes P4/P5 |
| P17 | Weight interface | WEIGHT strobe (`load_input_reg`), one fp32 word/cycle on `data_in[31:0]`, row-major row-outer; one-time `WR_CYC = R·C`, excluded from per-pass; supersedes P8 |
| P18 | EXP_FN / LOG_FN | Defined now (closes P6): `exp(v) = 1 + v + v²/2`, `log(v) = v − 1`; fp32 evaluation order normative in F3, then `trunc8` |
| P19 | fp32 corner cases | IEEE-754 binary32 RNE; no FMA; gradual underflow; no NaN/Inf in stimulus (A17) |
| P20 | F2 shift semantics | Slice partials enter at their own significance (`y ± 2^b·p_b`, LSB→MSB); the accumulator does not shift. Corrects the v1.0/v1.1 `2·y ± p` wording; restores identity (I7) and matches legacy witness arithmetic |
| D1 | Ground truth | This spec + oracle; legacy and v2 both implementers |
| D8 | Wide output view | Hierarchical TB read of the fp32 `y` (F2); **no port** |
