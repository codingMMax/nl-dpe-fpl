# Spec v1.0 (FROZEN) — NL-DPE primitive (`dpe_nldpe`, v2 clean-room)

**Status**: **v1.0 FROZEN 2026-08-29** — decision points P1–P13 closed, final
read complete. Amendments require a new revision (v1.x) + decision-log entry
+ oracle re-transcription.
**Precedence**: this spec > v2 RTL ≡ Python oracle > legacy RTL (witness).
**Normative rule**: every choice the oracle makes must exist here first; if
oracle and spec disagree, the spec is wrong until fixed.

Sources: project-lead session input 2026-08-29 (pipeline description, output
buffer + per-column ACAM, precision contract, port-sharing and compute-term
closures). Legacy generated RTL consulted only for conformance questions,
never as a source of requirements.

---

## §1 Block definition

One NL-DPE = **crossbar + ACAM**, with three internal storage structures and
**two physically separate external memory interfaces** (P12: one input-side,
one output-side; LOAD and OUTPUT never contend for a port).

```
 ext.mem(in) ──40b/cyc──▶ INPUT BUFFER ──1 bit-slice/cyc──▶ CROSSBAR ──▶ SHIFT&ACC
                          (R × 8 b, sliced)   (R bits)      (R×C int8)    │ ▲ pipelined
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

**Out of scope**: analog noise, ADC quantization error, device nonlinearity,
weight/output requantization policies, multi-DPE arrays, pass scheduling
(wrapper owns scheduling; this block enforces only its own readiness rules).
The block computes the **ideal integer behavior** — ground truth assumes
quantization and noise are already taken care of elsewhere.

## §2 Data types & precision contract

- **T1 (strict)**: crossbar input activations are **int8** (2's complement).
- **T2 (strict)**: ACAM output is **int8** per column (output buffer + stream).
- **T3 (internal, don't-care)**: crossbar output / ACAM input precision is not
  constrained; modeled as the exact integer sum (int32). Any implementation
  width is acceptable if the ACAM int8 result is unchanged.
- **T4**: weights are **clean int8** at the RTL level; float32→int8 weight
  quantization happens at programming time, outside this RTL flow.
- **T5 (idealization)**: the analog MAC is modeled as **exact signed integer
  arithmetic** with an ideal (infinite-precision, noiseless) read-out:
  `y_int[c] = Σ_r W[r,c]·x[r]` in int32, no saturation, no rounding.

## §3 Storage organization

| Structure | Size | Organization |
|---|---|---|
| Weight storage | `R·C` bytes | int8 cell per (r, c); programmed once per workload |
| Input buffer | `R × P` bits (**single** instance — P1) | **bit-sliced**: bank `b` holds bit b of every row; byte-major writes via corner-turn (§4.3) |
| Output buffer | `C × 8` bits (**single** instance) | one int8 per column; written wholesale by the ACAM stage (1 cycle), drained 5 bytes/cycle |
| Accumulators | `C × 32` bits | internal (T3); shift&acc pipeline stage |

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
| `clk`, `reset` | in | clock; synchronous active-high reset (A11) |
| `data_in[39:0]` | in | payload for both WEIGHT and ACT streams (5 bytes/cycle, byte i in bits `8i+7:8i`) |
| `w_buf_en` | in | **ACT burst strobe**: present + accept one ACT word |
| `load_input_reg` | in | **WEIGHT strobe**: present + accept one weight word (repurposed; legacy faithful left it unwired) |
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

Verification-only state (`y_int` per column) is **not a port** — the TB reads
it hierarchically (revised D8; port surface stays pristine).

### 4.2 Weight stream (WEIGHT strobes) — one-time per workload

- `WR_CYC = ⌈R·C·8/40⌉`. 256×512 → **26 215**; 256×256 → 13 108.
- **Byte order (closed, P8): row-major, row-outer** — byte #k carries
  `W[k/C][k mod C]` (all C columns of row 0 first, then row 1, …).
- Weights persist across arbitrarily many passes (I6); re-programming only
  between workloads.

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
| LOAD (ACT burst) | `LOAD_CYC` | ⌈R·8/40⌉ |
| CROSSBAR fires | P | 8 (one bit-slice per cycle, LSB→MSB) |
| SHIFT&ACC | pipelined | MSB accumulate completes 1 cycle after MSB fire |
| ACAM | 1 | C units parallel, once per pass, after MSB shift&acc |
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
WR_CYC      = ⌈R·C·8/40⌉                        (one-time, excluded per-pass)
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

- **F1 (normative)**: `y_int[c] = Σ_{r=0}^{R-1} W[r,c] · x[r]` — exact signed
  2's-complement int32 (T5).
- **F2 (structural requirement)**: fires LSB→MSB, one slice per cycle;
  shift&acc implements `y ← (y << 1) + s_b` for b < P−1 and
  `y ← (y << 1) − s_{P-1}` for the MSB slice. Final y must equal F1
  bit-exactly, and COMPUTE_CYC must emerge structurally (§5.1), not from a
  hold-counter.
- **F3 ACAM modes** (`nl_dpe_control`, sampled at compute start, stable until
  `dpe_done`; all modes: 1 cycle, C units parallel, int8 out):

| code | name | semantics |
|---|---|---|
| 00 | REGULAR | rescale to int8, nothing additional: `out8[c] = y_int[c][7:0]` (low byte, mod-256) — closed P4 |
| 01 | ACTIVATION | ReLU on int32, then low byte: `y_int[c] < 0 → 8'h00`, else `y_int[c][7:0]` — closed P5 |
| 10 | EXP | `EXP_FN(y_int)` → int8 — **named parameter**, exact fixed-point definition deferred to DIMM stage (closed P6) |
| 11 | LOG | `LOG_FN(y_int)` → int8 — **named parameter**, same deferral |

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
| A12 | RTL consumes post-quantization int8 weights (float32→int8 conversion external) |
| A13 | Dequant / scale folding / requantize-to-int8 belong to the mapping layer (Stage 5) |
| A14 | Noise / ADC error / nonlinearity excluded — ideal-integer model (T5) |
| A15 | External interface = two physically separate memory buffers/ports (P12); LOAD and OUTPUT never contend |

## §8 Invariants (independently checkable)

- **I1 Functional**: ACAM int8 outputs (and hierarchical `y_int`) bit-exact vs
  oracle for all §9 stimulus classes, modes {00, 01} (10, 11 once EXP_FN/LOG_FN
  defined).
- **I2 Cycles**: `measured(M) = T_fill + (M−1)·T_steady + Δ_impl`, Δ_impl
  invariant across M ∈ {1,2,4,8}, modes {00,01}.
- **I3 Readiness honesty**: `MSB_SA_Ready` per §4.4; no ACT byte accepted while
  low; bursts accepted only in full.
- **I4 Mode isolation**: mode change affects the next pass only.
- **I5 Stream integrity**: pass-k outputs unaffected by pass-(k+1) ACT arrival
  timing within the readiness contract.
- **I6 Stationarity**: weights persist across arbitrarily many passes.
- **I7 Layout conformance**: stream byte k lands exactly per §4.2/4.3/4.5
  (identity-weights check: out[c] = x[c] per F1).
- **I8 Port-surface conformance**: module port list identical to
  `vtr/dpe_blackbox.v`; no new external signals.

## §9 Verification contract

- **Oracle**: NumPy, written from this spec only. Generates stimulus `.mem`
  files (weights, activations — byte streams per §4), expected `y_int`,
  expected output bytes, expected cycle counts (§5.3). Computes F1 from first
  principles; does not mimic RTL structure.
- **Stimulus classes**: identity W (I7); random W, x ~ uniform int8; signed
  extremes (±128, all-zero rows); ReLU-negative-heavy; M ∈ {1,2,4,8}; modes
  {00, 01}.
- **Three-way agreement**: v2 RTL ≡ oracle (bit-exact values; cycles vs §5.3 +
  Δ_impl) on identical stimulus files; legacy RTL runs the same files as a
  third witness. Legacy divergence expected at T_steady (60 vs 52) and ACAM
  modes 01–11; reported, not gated.
- **Clean-room rule**: legacy RTL read-only; never consulted while writing v2
  RTL.

## §10 Decision log

| # | Decision | Closure (2026-08-29) |
|---|---|---|
| P1 | Input buffering | **Single** buffer, refill gated on MSB fire; legacy (double) = differing witness |
| P2 | MSB slice | **Subtract** (2's-complement MSB) |
| P3 | Internal precision | Don't-care; modeled exact int32 |
| P4 | REGULAR mode | Low byte mod-256 |
| P5 | ACTIVATION mode | ReLU on int32, then low byte |
| P6 | EXP/LOG | Interface chartered; `EXP_FN`/`LOG_FN` parameters, defined at DIMM stage |
| P7 | Output path | 40-bit stream, C bytes column-order, gapless, after ACAM |
| P8 | Weight interface | `load_input_reg` strobe, same 40-bit port, row-major (row-outer), WR_CYC one-time |
| P9 | Geometry | R, C independent; reference 256×512; cross-check 256×256 |
| P10 | Compute term | **COMPUTE_CYC = 10** (8 fires + MSB-acc drain + ACAM) — closed (a) |
| P11 | Output overlap | Forbidden (strict after last `out_valid`) → OUTPUT_CYC+1 bound |
| P12 | External ports | **Separate in/out ports, two physically separate memory buffers** — closed (b) |
| P13 | Port surface | Exact VTR blackbox names; semantics per §4 mapping; no new signals |
| D1 | Ground truth | This spec + oracle; legacy and v2 both implementers |
| D8 | Wide output view | Hierarchical TB read of `y_int`; **no port** |
