# Spec v2.0 (CLEAN REWRITE) — NL-DPE primitive (`dpe_nldpe`, v2 clean-room)

**Status**: **v2.0 integer dataflow, 2026-09-14** — clean rewrite superseding the
v1.1 fp32 amendment (2026-09-12). Numeric contract: **int8 stationary weights,
int8 activations, exact integer MAC (int32 accumulator), integer ACAM mode
forms with the `trunc8` low-byte output rule.** Advisor consultation
2026-09-14: network accuracy/quantization is handled by the mapping layer, so
the primitive models an *idealized integer MAC*; fp32 weight/crossbar modeling
(v1.1) is retired.
**Precedence**: this spec > v2 RTL ≡ Python oracle > legacy RTL (witness).
**Normative rule**: every choice the oracle makes must exist here first; if
oracle and spec disagree, the spec is wrong until fixed.

Revision history:
- **v2.0 (2026-09-14)**: clean integer rewrite. T3/T4/T5 and F2/F3 become exact
  integer; WEIGHT strobe payload is int8 on `data_in[7:0]` (P23); EXP/LOG are
  integer forms (P24); accumulator width guaranteed (P25); dual compare
  contract (P26). Retired: P14, P15, P17, P18, P19, P20. Retained: P1–P13, P16.
- **v1.1 (2026-09-12)**: fp32 dataflow amendment (retired by v2.0).
- **v1.0 (2026-08-29)**: initial frozen charter (P1–P13).

Sources: project-lead session input 2026-08-29 (pipeline description, output
buffer + per-column ACAM, port-sharing and compute-term closures); project-lead
decisions 2026-09-12 (retired fp32 amendment); advisor consultation +
project-lead decisions 2026-09-14 (integer dataflow, dual comparison).

---

## §1 Block definition

One NL-DPE = **crossbar + ACAM**, with three internal storage structures and
**two physically separate external memory interfaces** (P12: one input-side,
one output-side; LOAD and OUTPUT never contend for a port).

```
 ext.mem(in) ──40b/cyc──▶ INPUT BUFFER ──1 bit-slice/cyc──▶ CROSSBAR ──▶ SHIFT&ACC
                           (R × 8 b, sliced)   (R bits)     (R×C int8 W)    │
                                                                          ▼
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
readiness rules), and all **external quantizers** — any upstream precision is
quantized to int8 *before* the activation stream is presented (A16), and
weights are likewise already quantized to int8 (T4/A16). The block computes the
**exact integer behavior of §6 F2** — accuracy and quantization are taken care
of elsewhere.

## §2 Data types & precision contract

- **T1 (strict)**: crossbar input activations are **int8** (2's complement).
- **T2 (strict)**: ACAM output is **int8** per column (output buffer + stream).
- **T3**: the crossbar/accumulator domain is **int32** (2's complement). The
  exact MAC of §6 F2 never overflows it: `|y| ≤ R·2^14`, so int32 is guaranteed
  for `R ≤ 131072` (P25). Only the low byte is exposed (F3).
- **T4**: weights are **int8** (2's complement) values, programmed once per
  workload through the WEIGHT strobe (§4.2) and stationary thereafter (A5).
- **T5 (idealization)**: the analog MAC is modeled as the **exact integer
  arithmetic of §6 F2** with an ideal (noiseless) read-out; the mathematical
  intent is `y[c] = Σ_r W[r,c]·x[r]`, and integer arithmetic is exact — no
  rounding, no ordering constraints, no precision corner cases.

## §3 Storage organization

| Structure | Size | Organization |
|---|---|---|
| Weight storage | `R·C × 8` bits | int8 word per (r, c); programmed once per workload (P23) |
| Input buffer | `R × P` bits (**single** instance — P1) | **bit-sliced**: bank `b` holds bit b of every row; byte-major writes via corner-turn (§4.3) |
| Output buffer | `C × 8` bits (**single** instance) | one int8 per column; written wholesale by the ACAM stage (1 cycle), drained 5 bytes/cycle |
| Accumulators | `C × 32` bits | int32 (T3); partial-shift accumulate stage |

**P1 (closed)**: input buffer is **single** with in-place refresh. A new input
burst may begin refilling only **after the MSB fire of the in-flight pass has
completed** (all P banks consumed). Earlier refill would corrupt un-fired
slices (a byte-major write touches all banks at once).
**P12 (closed)**: external interface = **two physically separate memory
buffers/ports** (in / out). LOAD and OUTPUT never contend.

## §4 External interfaces — exact VTR port surface (normative)

Port names/widths are **identical to the VTR blackbox contract**
(`vtr/dpe_blackbox.v`, arch XML `<model name="dpe">`). No new external signals
exist in v2. v2 assigns semantics as follows:

| Port | Dir | v2 semantics |
|---|---|---|
| `clk`, `reset` | in | clock; synchronous active-high reset (A10) |
| `data_in[39:0]` | in | ACT payload (5 bytes/cycle, byte i in bits `8i+7:8i`) **or** one int8 weight word (WEIGHT strobes: `[7:0]` = weight, `[39:8]` ignored) |
| `w_buf_en` | in | **ACT burst strobe**: present + accept one ACT word |
| `load_input_reg` | in | **WEIGHT strobe**: present + accept one int8 weight word (`data_in[7:0]`) |
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

Verification-only state (int32 `y` per column, §6 F2) is **not a port** — the
TB reads it hierarchically (revised D8; port surface stays pristine). The TB
compares **both** the full int32 `y` and the drained 8-bit stream (P26).

### 4.2 Weight programming (WEIGHT strobes) — one-time per workload

- One int8 weight word per strobe cycle on `data_in[7:0]`; `WR_CYC = R·C`
  (one-time, **excluded** from the per-pass formulas of §5.3). 256×512 → 131 072;
  256×256 → 65 536.
- **Order (closed, P23): row-major, row-outer** — word #k carries
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
| CROSSBAR fires | P | 8 (one bit-slice per cycle, LSB→MSB); per-slice partial = integer row sum (§6 F2) |
| SHIFT&ACC | pipelined | integer partial-shift (exact; §6 F2): each slice partial enters at `2^b·s_b`; MSB accumulate completes 1 cycle after MSB fire |
| ACAM | 1 | C units parallel, once per pass, after MSB shift&acc; integer form → trunc8 (§6 F3) |
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
WR_CYC      = R·C                              (int8 weight words, one-time,
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

Notation: integer arithmetic is exact; `⌊·⌋` = floor (equal to truncation
toward zero for the non-negative squares used in EXP).

- **F1 (mathematical intent)**: `y[c] = Σ_{r=0}^{R-1} W[r,c] · x[r]` over int8
  weights and int8 activations; `y` is an exact integer (T5).
- **F2 (normative sequence)**: fires LSB→MSB, one slice per cycle. Per slice
  `b = 0..P−1`, per column `c`, the crossbar partial is the integer sum over
  the selected rows:

  ```
  s_b[c] := Σ_{r} ( bit_b(x[r]) ? W[r,c] : 0 )      (exact; order immaterial)
  ```

  Shift&acc adds each slice partial at its own significance (P20: the
  *partial* is scaled by `2^b`, the accumulator does not shift); signed
  two's-complement MSB subtract (P2):

  ```
  y := Σ_{b=0..P-2} 2^b · s_b[c]  −  2^(P-1) · s_{P-1}[c]
  ```

  `y` is T3's crossbar output (int32). Because all arithmetic is exact, any
  summation order yields the same result — the sequence above is the hardware
  structure, not a rounding contract. COMPUTE_CYC must emerge structurally
  (§5.1), not from a hold-counter.
- **F3 ACAM modes** (`nl_dpe_control`, sampled at compute start, stable until
  `dpe_done`; all modes: 1 cycle, C units parallel, int8 out). The output rule
  is the same for all modes — **integer functional form first, then `trunc8`**
  (P16: truncate toward zero, saturate to int32, keep the low byte):

  ```
  trunc8(z): t := trunc_toward_zero(z), clamped to [-2^31, 2^31-1]
             out8 := t[7:0]                     (two's-complement low byte)
  ```

  | code | name | semantics |
  |---|---|---|
  | 00 | REGULAR | `out8 = trunc8(y)` |
  | 01 | ACTIVATION | `out8 = trunc8(relu(y))`, relu(y) = y if y > 0 else 0 |
  | 10 | EXP | `out8 = trunc8(1 + y + ⌊y²/2⌋)`, evaluated exactly (64-bit or wider); only `y mod 512` affects the output byte |
  | 11 | LOG | `out8 = trunc8(y − 1)` |

## §7 Assumption register (v2.0)

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
| A12 | Weights are int8 values programmed via the WEIGHT strobe (§4.2) and stationary for the workload |
| A13 | Dequant / scale folding / requantize-to-int8 belong to the mapping layer (Stage 5) |
| A14 | Noise / device nonlinearity excluded — exact-integer idealization (T5) |
| A15 | External interface = two physically separate memory buffers/ports (P12); LOAD and OUTPUT never contend |
| A16 | Activations and weights arrive already quantized to int8; quantization of any higher upstream precision is external and not modeled |
| A17 | Integer arithmetic is exact; no overflow occurs for `R ≤ 131072` (T3/P25); EXP uses a wider intermediate as allowed by F3 |
| A18 | ACAM is the only in-block quantizer: integer functional form → trunc8 (§6 F3); no other rounding or clamping exists in the block |

## §8 Invariants (independently checkable)

- **I1 Functional**: ACAM int8 outputs **and** the hierarchical int32 `y`
  (§6 F2) bit-exact vs oracle for all §9 stimulus classes, all modes {00,01,10,11}.
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
  (identity-weights check: `y = x` exactly; `out[c] = trunc8(x[c]) = x[c]`).
- **I8 Port-surface conformance**: module port list identical to
  `vtr/dpe_blackbox.v`; no new external signals.

## §9 Verification contract

- **Oracle**: NumPy, written from this spec only. Implements the exact integer
  MAC and the integer ACAM forms from first principles (no RTL mimicry).
  Generates stimulus files (int8 weights per §4.2, int8 activations per §4.3),
  expected hierarchical int32 `y`, expected output bytes, expected cycle counts
  (§5.3).
- **Dual compare (P26)**: the TB compares **both** (a) the full int32 `y`
  hierarchically (D8; verification-only state) and (b) the drained 8-bit
  per-column stream. The byte stream alone can alias errors that change `y` by
  a multiple of 256; the full-width compare closes that blind spot.
- **Stimulus classes**: identity W (I7); random int8 W (both signs, zeros,
  signed extremes); x ~ uniform int8 plus signed extremes (−128/+127, all-zero
  rows); M ∈ {1,2,4,8}; modes {00, 01, 10, 11}.
- **Three-way agreement**: v2 RTL ≡ oracle (bit-exact: hierarchical int32 `y`
  and output bytes; cycles vs §5.3 + Δ_impl) on identical stimulus files;
  legacy RTL runs the same files as a third witness. Legacy shares the integer
  arithmetic, so value comparisons are now expected to match; remaining known
  divergences (schedule/buffer T_steady 60 vs 52, weight interface, ACAM mode
  details) are reported, not gated.
- **Clean-room rule**: legacy RTL read-only; never consulted while writing v2
  RTL.

## §10 Decision log

| # | Decision | Closure |
|---|---|---|
| P1 | Input buffering | **Single** buffer, refill gated on MSB fire; legacy (double) = differing witness. Collaborator-confirmed single 2026-09-12 |
| P2 | MSB slice | **Subtract** (2's-complement MSB) |
| P4 | REGULAR mode | Low byte mod-256 — **superseded by P16** |
| P5 | ACTIVATION mode | ReLU then low byte — **superseded by P16** |
| P6 | EXP/LOG | Deferred to DIMM stage — closed by P24 |
| P7 | Output path | 40-bit stream, C bytes column-order, gapless, after ACAM |
| P8 | Weight interface | Byte stream via `load_input_reg` — **superseded by P23** |
| P9 | Geometry | R, C independent; reference 256×512; cross-check 256×256 |
| P10 | Compute term | **COMPUTE_CYC = 10** (8 fires + MSB-acc drain + ACAM) — closed (a), 2026-08-29 |
| P11 | Output overlap | Forbidden (strict after last `out_valid`) → OUTPUT_CYC+1 bound |
| P12 | External ports | **Separate in/out ports, two physically separate memory buffers** — closed (b) |
| P13 | Port surface | Exact VTR blackbox names; semantics per §4 mapping; no new signals |
| P16 | ACAM output | Integer functional form first, then `trunc8` (truncate toward zero, saturate int32, low byte) |
| P21 | Numeric domain | **int8** weights and activations; **int32** accumulator; exact integer MAC (T3/T4/T5) — replaces P14/P15 |
| P22 | MAC semantics | Exact integer partial-shift: `y = Σ 2^b·s_b − 2^(P-1)·s_{P-1}`; no rounding or ordering contract — replaces P15/P20 |
| P23 | Weight interface | WEIGHT strobe (`load_input_reg`), one int8 word/cycle on `data_in[7:0]`, row-major row-outer; one-time `WR_CYC = R·C` — replaces P17 |
| P24 | EXP/LOG forms | Integer forms (closes P6): `exp(y) = 1 + y + ⌊y²/2⌋` (exact, wide intermediate), `log(y) = y − 1`, then `trunc8` — replaces P18 |
| P25 | Accumulator width | int32 suffices exactly for `R ≤ 131072` (`|y| ≤ R·2^14`); EXP may use a wider intermediate — replaces P19 |
| P26 | Comparison contract | TB compares full hierarchical int32 `y` **and** the 8-bit stream (dual compare) |
| D1 | Ground truth | This spec + oracle; legacy and v2 both implementers |
| D8 | Wide output view | Hierarchical TB read of the int32 `y` (F2); **no port** |

**Retired by v2.0** (recorded for history): P3 (fp32 accumulator), P14 (fp32
weights/crossbar), P15 (structural fp32 sequence), P17 (fp32 weight strobe),
P18 (fp32 EXP/LOG), P19 (fp32 corner cases), P20 (fp32 partial-shift wording —
the *structural* rule survives as P22).
