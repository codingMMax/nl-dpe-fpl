# Spec v0.3 (FROZEN) — GEMM array (`gemm_top`, v2 clean-room)

**Status**: **v0.3 FROZEN 2026-09-17** — Stage 2 charter for the VMM/GEMM
projection array built from verified NL-DPE primitives. Frozen by project
lead 2026-09-17; amendments require a new revision + decision-log entry +
oracle/sim/RTL re-transcription (primitive precedent).
Revision history:
- **v0.3 (2026-09-17)**: **no ACAM after reduction.** Every tile runs
  REGULAR (its ACAM is the identity form → `trunc8(y_v)`); the array output
  is the low byte of the reduced sum. Removed: the output-stage form (F4 of
  v0.2), the `nl_dpe_control` port, and the mode-placement rule (G4 of
  v0.2). Consequence: `out8 = trunc8(Σ_v y_v)` **exactly** (mod-256 algebra),
  so the v0.2 "reduce int8 partials" caveat disappears. Nonlinearity
  (ReLU/EXP/LOG) is out of scope for this array.
- **v0.2 (2026-09-17)**: array instantiation explicit; W/X→tile mapping
  (§1.1); reduction pipeline formalized as latency-vs-period; O1–O3
  dispositions.
- **v0.1 (2026-09-17)**: initial draft.

**Precedence**: this spec > `gemm_ref` ≡ `gemm_sim` > legacy `fc_top`
(witness). Below this charter sits [`dpe_nldpe.md`](dpe_nldpe.md) (v2.0.1),
whose per-pass schedule and port semantics are **normative and unchanged** —
the array composes that primitive, it does not modify it.
**Normative rule**: every choice the oracle/sim make must exist here first;
if they disagree, this spec is wrong until fixed.

Sources: project-lead decisions 2026-09-17 (reduce the ACAM int8 outputs;
**tiles always REGULAR; no ACAM after reduction**; incremental 1A→1D,
parameterized; clean-room `gemm_sim` + `gemm_top`); legacy `fc_top.v` and
`FC_RTL_PLAN.md` consulted only as witness/context, never as requirements.

---

## §1 Block definition

`gemm_top` = **V×H array of NL-DPE primitives + byte-reduction tree +
lane serializer**:

```
 Y[M,N] = X[M,K] · W[K,N]

                 v = 0            v = 1                 v = V-1
              ┌──────────┐     ┌──────────┐          ┌──────────┐
 X rows ──▶   │  dpe     │     │  dpe     │   ...    │  dpe     │   H columns
 (M passes)   │ (R×C W)  │     │ (R×C W)  │          │ (R×C W)  │   per v
              └────┬─────┘     └────┬─────┘          └────┬─────┘
                   │ int8 ACAM out8_v (REGULAR: trunc8(y_v))
                   └───────────────┴─────────┬───────────┘
                                             ▼
                                  byte-reduction tree: S = Σ_v
                                  sign_extend(out8_v)  (int32)
                                             ▼
                                  lane serializer: low byte of S
                                  (NO ACAM after reduction)
                                             ▼
                                  Y lanes  (per pass: H lanes × C bytes,
                                  lane h = columns h·C + c)
```

| Parameter | Meaning | Example |
|---|---|---|
| `M` | rows / passes (streamed) | 1, 8, 128 |
| `K` | reduction length | 128, 400, 2048 |
| `N` | output width | 120, 256, 512 |
| `R`, `C`, `BUF`, `P` | DPE geometry (primitive §1) | 256, 256, 40, 8 |
| `V` | `⌈K/R⌉` K-tiles (derived) | 1, 2, 8 |
| `H` | `⌈N/C⌉` N-tiles (derived) | 1, 2 |

**Instantiation (normative)**: the RTL module `gemm_top` **instantiates
`V·H` instances of the primitive module `dpe`** — no MAC, crossbar or ACAM
logic is re-implemented at the array level. The golden sim instantiates
`V·H` `NldpeDpe` objects (`v2/sim/nldpe_sim.py`); the oracle calls
`nldpe_ref.compute_y` / `acam_transform` per tile. New logic at this level
is exactly: weight routing, activation fan-out, the reduce tree, the lane
serializer, and control/readiness.

**Out of scope**: all nonlinear output forms (ACTIVATION/EXP/LOG) at this
array — they live downstream (mapping / later stages); attention
projections/composition (Stage 4); log-domain DIMM (`pool_farm_model.md`);
softmax (Stage 3); VTR wrapping (Stage 5); weight/activation quantization
policy (streams arrive int8).

**Fixed array parameters (this revision)**: all tiles REGULAR; one workload
per programming cycle; no partial-N drain truncation (§4.4); no
back-pressure beyond readiness.

### 1.1 W/X mapping onto the array (plain description)

`W[K,N]` is cut into `V×H` tiles of `R×C`; `X` rows are cut into `V` slices
of `R`:

```
             h=0          h=1                X row m (K bytes)
           ┌──────────┬──────────┐          ┌────────┬────────┐
  v=0      │ W[0:R,   │ W[0:R,   │          │  Xs_0  │  Xs_1  │   Xs_v = X[m, vR:(v+1)R]
           │   0:C]   │   C:2C]  │          │   R    │   R    │   (zero-pad past K)
           ├──────────┼──────────┤          └────────┴────────┘
  v=1      │ W[R:2R,  │ W[R:2R,  │               │        │
           │   0:C]   │   C:2C]  │               └───┬────┘
           └──────────┴──────────┘                   │ slice v broadcast to all H
                                │                    ▼
                                │     DPE(v,h) computes the partial outputs for
                                │     columns h·C .. h·C+C-1 over its own R rows
                                ▼     of K (its weight tile)
                     V partials per output column → reduce tree → lane serializer
                     → lane h drains columns h·C .. h·C+C-1
```

Rules:

- tile `(v,h)` holds `W[v·R + r][h·C + c]`, zero-padded when `v·R + r ≥ K`
  or `h·C + c ≥ N` (A5);
- activation slice `v` goes to **all H** DPEs of row `v`; the DPEs of a row
  differ only in their weight tiles (different output columns);
- output column `n = h·C + c` receives the V partials
  `out8_(0,h)[c] … out8_(V-1,h)[c]` — these are the reduce tree's leaves
  for lane `h`;
- programming order is the tile walk of §4.1: `(v,h)` v-major, then
  row-major row-outer inside the tile.

**Worked example** — `M=1, K=400, N=512, R=C=256` → `V=2, H=2`:

- tiles `(0,0)=W[0:256, 0:256]`, `(0,1)=W[0:256, 256:512]`,
  `(1,0)=W[256:400, 0:256]` (rows 400–511 zero-padded),
  `(1,1)=W[256:400, 256:512]`;
- slice 0 = `X[0, 0:256]` → both `(0,*)`; slice 1 = `X[0, 256:400]` + 112
  zeros → both `(1,*)`;
- column 100 = reduce of partials from `(0,0)` and `(1,0)` → lane 0;
  column 300 = partials from `(0,1)` and `(1,1)` → lane 1;
- weight stream sends tile words in the order `(0,0), (0,1), (1,0), (1,1)`.

## §2 Data types

- **A1 (strict)**: activation bytes, weight bytes and stream bytes are int8
  (2's complement), per primitive T1/T2/T4.
- **A2**: per-tile crossbar output is the primitive's exact int32 `y`
  (T3); per-tile formal output is the ACAM identity form
  `out8_v = trunc8(y_v)` (primitive F3 REGULAR, P16).
- **A3**: reduction partial `S[m,n] = Σ_v sign_extend(out8_v[m, n mod C])`
  is **int32**; `|S| ≤ 128·V` (int32 exact for any V this design supports).
- **A4 (strict)**: array output `out8[m,n] = trunc8(S[m,n])` — the lane
  serializer takes the low byte; same `trunc8` rule as the primitive (P16).
  There is **no ACAM unit after the reduction** (v0.3).
- **A5**: padding cells (rows/cols beyond K/N) use **zero** weights and
  zero activations; padding outputs are drained but are not part of `Y`
  (host ignores them, §4.4).

## §3 Storage & composition

| Structure | Size | Notes |
|---|---|---|
| Weight storage | `V·H·R·C` int8 total | one primitive instance per (v,h); each holds its `R×C` tile |
| Activation buffers | per primitive (single, P1) | per (v,h) instance; V-lane bus feeds all H copies of a v-row identically |
| Reduction tree | `V` leaves per output column | sign-extend the tiles' int8 outputs → int32, pairwise sum, `TREE_PIPE` registered stages |
| Lane serializer | `H × C` bytes per pass | takes the low byte of `S` and drains §4.4; no transform |
| Output lanes | H independent | per-lane drain of C bytes, column order |

Memory-hierarchy notes: no new storage semantics are defined here; the
array is a *composition*, so P1/A9 (single input buffer per DPE), P11
(strict output ordering), and P27 (mode latched with the weights) apply
inside each instance unchanged. The tiles' workload form is always REGULAR
(v0.3), so P27 has no observable array-level effect.

## §4 External interfaces (normative for this revision)

Port surface is **streaming**, primitive-style; VTR/BRAM wrapping is
Stage 5.

| Port | Dir | Semantics |
|---|---|---|
| `clk`, `reset` | in | synchronous, active-high reset (primitive A10) |
| `act_in[V·BUF-1:0]` | in | **V lanes**; lane `v` carries tile-`v` ACT bytes in the primitive's §4.3 packing (5 bytes/cycle, byte i in bits `8i+7:8i`), zero-padded once `K` is exhausted |
| `act_en` | in | ACT strobe: all V lanes present one word; accepted only while `MSB_SA_Ready` high |
| `weight_in[7:0]` | in | one int8 weight per strobe |
| `weight_en` | in | WEIGHT strobe (programming only; no mode latch in v0.3) |
| `MSB_SA_Ready` | out | array refill permit = **AND** of all V·H primitive `MSB_SA_Ready` outputs |
| `data_out[H·BUF-1:0]` | out | result lanes, 5 bytes/cycle per lane; lane `h` carries tile-column block `h` (§4.4) |
| `dpe_done` | out | 1-cycle pulse after the last byte of the last lane of a pass |
| `reg_full` | out | output busy: high from lane-serializer write until all lanes' drains complete |
| `dpe_done_v[H-1:0]`, `msb_ready_v[V-1:0]` | out | observability-only per-tile pulses (probes; may be omitted in VTR revision) |
| reserved | — | `shift_add_*`, `load_output_reg` style signals are **not** part of this module; no reserved ports |

### 4.1 Weight programming (one-time per workload)

- Order (closed, G2): **v-major, then h, then row-major row-outer within the
  tile**. Word `k` (`k = 0..V·H·R·C−1`) carries
  `W_tile(v,h)[r][c]` with
  `v = k div (H·R·C)`, `h = (k div (R·C)) mod H`,
  `r = (k div C) mod R`, `c = k mod C`;
  `W_tile(v,h)[r][c] = W[v·R + r][h·C + c]` (zero when out of range, A5).
- `WR_CYC = V·H·R·C`, one-time, **excluded** from per-pass formulas.

### 4.2 Activation input stream (per pass)

The tiling itself is §1.1; this section defines only how the already-tiled
slices are delivered.

- Host presents `LCYC = ⌈R·8/BUF⌉` words **on all V lanes in parallel**
  (one strobe per cycle writes all tiles of the array).
- Lane `v` word `w` carries bytes `X[m][v·R + 5w .. v·R + 5w+4]`, padded
  with zeros beyond `K`; the h-dimension receives the same lane data
  (broadcast).
- All V·H instances see identical per-lane strobes; `act_en` must be low
  whenever `MSB_SA_Ready` is low (array contract, §5.3).

### 4.3 Compute

Per primitive (§5 of `dpe_nldpe.md`), tile mode **REGULAR** (v0.3):
V×H instances fire in lockstep (identical geometry and identical cadence).

### 4.4 Output lanes (per pass)

- Each h-tile owns one **independent output lane**; lanes drain in parallel
  (H×5 bytes/cycle total). Lane `h` carries the C bytes of columns
  `h·C + c`, column order, gapless; bytes beyond `N` are padding (A5).
- The lane serializer takes the **low byte** of `S` (`trunc8`, A4); no
  transform follows the reduction.
- `OUTPUT_CYC_prim = ⌈C·8/BUF⌉` per lane; `dpe_done` pulses after the last
  byte of the last lane; lanes are idle (zero) outside drains.
- The reduce tree consumes each tile's `drain_valid` words **as they
  stream** (no `dpe_done`-gated block reduction); a lane's block `m+1` is
  written only after block `m` has fully drained (array-level P11,
  guaranteed by `T_steady ≥ OUTPUT_CYC_prim + 1`).

## §5 Pipeline behavior & cycle contract

### 5.1 Per-tile stages (inherited, normative)

| Stage | Latency |
|---|---|
| LOAD (per lane, all tiles parallel) | `LOAD_CYC = ⌈R·8/BUF⌉` |
| CROSSBAR fires | `P` (one slice/cycle, LSB→MSB) |
| SHIFT&ACC | MSB accumulate completes 1 cycle after MSB fire |
| ACAM (REGULAR) | 1 (primitive F3/T2) |
| OUTPUT drain (per primitive) | `OUTPUT_CYC_prim = ⌈C·8/BUF⌉` |

`COMPUTE_CYC = P + 2` (primitive P10), structurally emergent.

### 5.2 Array additions (latency vs period)

- **Reduction pipeline**: `TREE_PIPE = ⌈log₂ V⌉` registered pairwise folds
  (V leaves, one per K-tile). Its **period is one five-byte word per lane
  per cycle**, matched to the primitive drain rate, so it never binds
  `T_steady`; it is consumed **streaming** from each tile's `drain_valid`
  output (no `dpe_done`-gated block reduction).
- **Lane serializer**: registered low-byte extraction, 1 cycle, applied once
  per pass (A4); period one word/lane/cycle.
- **Fill latency** `L_w = TREE_PIPE + 1` (tree + lane-serializer register).
  It depends on `V` only — **not on `C`** — and is paid **once**: it shifts
  `T_fill`, and constant terms cancel in `measured(M2) − measured(M1)`.
- **Block-serial counter-model (excluded)**: a reducer that must finish a
  pass's reduction before accepting the next pass would have period
  `≈ OCYC + L_w`, adding `L_w` to every step. The T_steady calibration gate
  detects this; this charter requires the pipelined form.

### 5.3 Cycle formulas (normative)

```
WR_CYC       = V·H·R·C                       (one-time, excluded)
T_fill_tile  = LOAD_CYC + COMPUTE_CYC + OUTPUT_CYC_prim
T_steady     = max( LOAD_CYC + P ,           ← primitive input-buffer bound (P1)
                    COMPUTE_CYC ,            ← compute bound
                    OUTPUT_CYC_prim + 1 ,    ← primitive output-buffer bound (P11)
                    RED_PERIOD )             ← reduction/serializer period
RED_PERIOD   = 1                             (pipelined word stream; never binds,
                                              stated for completeness)
L_w          = TREE_PIPE + 1                 (fill latency; V-only, C-independent)
T_fill_array = T_fill_tile + L_w
Total(M)     = T_fill_array + (M−1)·T_steady        (architectural)
measured(M)  = Total(M) + Δ_impl                    (one implementation-declared
                                                     constant, invariant across
                                                     M and geometries)
```

At `R=C=256, BUF=40, P=8`: `LOAD=52`, `COMPUTE=10`, `OUTPUT_prim=52`,
`T_fill_tile=114`, `T_steady=60`. At `C=512`: `OUTPUT_prim=103`,
`T_fill_tile=165`, `T_steady=104`. (Legacy `fc_top` double-buffered
(`T_steady=52`, `T_fill=120+…`) is a **differing witness**, reported only —
primitive P1/A9 governs this charter.)

**Worked timeline (256×256, V=2 → L_w=2, M=2):**

```
cycle:   61  62                     113 114       121 122                173 174
         ACAM ├──── tile drain 52 ───┤ done        ACAM ├── tile drain ───┤ done
              │+L_w                                     │+L_w
              ▼                                         ▼
              ├──── lane out 64..115 ┤ ~116            ├── lane out 124..175 ┤ ~176

array completion interval = 176 − 116 = 60 = T_steady;  L_w is paid once (T_fill)
```

**Per-pass cadence claim (to be verified structurally):** because all tiles
are lockstep and the reduction/serializer pipeline is word-pipelined, the
array emits one pass per `T_steady` cycles; the steady step is identical to
the primitive's, and `L_w` is absorbed in the fill.

## §6 Functional semantics

- **F1 (tile)**: instance `(v,h)` computes, for pass `m`,
  `y_vh[c] = Σ_{r} W_tile(v,h)[r][c] · x_v[m][r]` exactly as primitive F2/F1,
  where `x_v[m][r] = X[m][v·R + r]` (0 beyond K).
- **F2 (tile ACAM)**: `out8_vh = trunc8(y_vh)` — the primitive F3 **REGULAR**
  form. All tiles are REGULAR (v0.3).
- **F3 (reduction)**: per pass `m` and column `n = h·C + c`:
  `S[m,n] = Σ_{v=0}^{V-1} sign_extend(out8_vh[c])` — wide int32 sum.
- **F4 (serializer)**: `out8[m,n] = trunc8(S[m,n])` — the low byte; **no
  ACAM after reduction** (v0.3).
- **Exactness theorem (v0.3)**: because `sign_extend(trunc8(y_v)) ≡ y_v
  (mod 256)` for every tile, the array output is
  `out8[m,n] = trunc8( Σ_v y_v[m,n] )` **exactly** — the byte-domain
  reduction is the true low byte of the partial sum, for every case (no
  "int8-partial" approximation; the nonlinear-form caveat of v0.2 is gone).
  The oracle asserts this against the wide sum.

## §7 Assumptions

| # | Assumption |
|---|---|
| B1 | `M ≥ 1` passes are streamed; the host may refill only while `MSB_SA_Ready` is high |
| B2 | One workload per programming cycle; weights stationary (primitive A5/A12) |
| B3 | All tiles are REGULAR and run homogeneously (same geometry, same pass index) |
| B4 | Padding is zeros (A5); the host ignores padded outputs (`n ≥ N`, `v·R + r ≥ K`) |
| B5 | Reduction operates on the primitive's formal int8 output, never on internals (no hierarchical reads outside the TB) |
| B6 | The host keeps `act_en` low while any tile's ready is low; violations are flagged (TB), not silently corrupted |
| B7 | Output sink always ready; `reg_full` is observability only (primitive A1) |
| B8 | Integer arithmetic is exact; `S` is int32-exact for all supported V |

## §8 Invariants (independently checkable)

- **I1 Functional**: per-pass wide partial `S` and the array output bytes
  bit-exact vs `gemm_ref` for all §9 cases.
- **I2 Cycles**: `measured(M) = Total(M) + Δ_impl`, `Δ_impl` invariant
  across `M ∈ {1,2,4,8}` and geometries; `WR_CYC` reported separately;
  T_steady steps match `(M2−M1)·T_steady` exactly.
- **I3 Readiness**: array `MSB_SA_Ready` = AND of tile readies; no strobe
  accepted while low; bursts accepted only in full.
- **I4 Ordering**: per-pass output bytes land per §4.4 (lane order, column
  order, gapless); pass-k outputs unaffected by pass-(k+1) arrival timing.
- **I5 Stationarity**: weights persist across passes.
- **I6 V=1 identity**: with `V=1,H=1`, the array is **value-identical** to a
  single primitive instance in REGULAR mode (bit-exact `S` and output
  bytes); cycles follow the array formula, offset only by the declared
  serializer register constant captured in `Δ_impl`.
- **I7 Byte-tree exactness**: `S ≡ Σ_v y_v (mod 256)` and
  `out8 = trunc8(Σ_v y_v)` — for every case (the v0.3 theorem, §6).
- **I8 Composition only**: no new numerics beyond §§3–6; the primitive is
  reused unchanged.

## §9 Verification contract

- **Oracle** `v2/oracle/gemm_ref.py`: exact composition from this spec
  (per-tile `nldpe_ref` calls, byte-tree reduce, low-byte serializer);
  written from the charter, no RTL/sim mimicry; self-test includes I6/I7 and
  padding/ordering cases.
- **Sim** `v2/sim/gemm_sim.py`: golden model instantiating `NldpeDpe`
  instances (tiles REGULAR); schedule copied from primitive
  `SimResult.timeline` (no re-derived cycle math); `dump_case` analogue
  certifies `sim ≡ oracle` (wide `S`, output bytes, cycle formula)
  **before** writing expected files (GATE 1).
- **RTL** `v2/rtl/gemm_top.v`: hand-written from this charter,
  **instantiating `V·H` primitive `dpe` modules**; TB
  `v2/tb/tb_gemm_top.v` + harness `v2/smoke/run_gemm_rtl.py` gate
  RTL ≡ certified expected bits (GATE 2) with the primitive's dual-compare
  analogue: wide partial `S` (probe) + output lanes.
- **Stimulus classes**: 1A (`V=1,H=1`) exact-GEMM cases; 1B (`V>1,H=1`)
  K-reduction; 1C (`V=1,H>1`) N-concat; 1D (`V>1,H>1`); M∈{1,2,4,8};
  identity/random/extremes; padding corners (`K mod R ≠ 0`, `N mod C ≠ 0`).
  (No mode axis in v0.3.)
- **Witness**: legacy `fc_top`/oracle on identical workloads where
  interfaces permit (byte-tree compatibility expected); cadence differences
  (52 vs 60) reported, not gated.

## §10 Decision log

| # | Decision | Closure |
|---|---|---|
| G1 | Composition | Streaming V×H array of unmodified `dpe` primitives; reduction across V only; H is concatenation |
| G2 | Weight stream | Single int8 weight bus; v-major, then h, then row-major row-outer; `WR_CYC = V·H·R·C` |
| G3 | Activation stream | V-lane parallel bus (one word per lane per strobe); h-broadcast; zero-pad beyond K |
| G4 | Tile mode | **All tiles REGULAR; no mode port; no mode-placement rule.** Nonlinear output forms are out of scope for this array (v0.3) |
| G5 | Reduction | Byte tree: sign-extend the primitives' int8 ACAM outputs, wide int32 sum `S`, `TREE_PIPE = ⌈log₂V⌉` |
| G6 | Output | Lane serializer takes the low byte of `S` (**no ACAM after reduction**); H independent BUF-wide lanes, `OUTPUT_CYC_prim` per lane, full C bytes incl. padding (v0.3) |
| G7 | Cycles | `T_steady = max(LOAD+P, COMPUTE, OUTPUT_prim+1, RED_PERIOD)` with `RED_PERIOD = 1`; `L_w = TREE_PIPE+1` is **fill latency** (V-only, C-independent), counted once in `T_fill`; one invariant `Δ_impl` |
| G8 | Comparison | Dual compare at the array boundary: wide `S` (probe) + output bytes |
| G9 | Scope | Incremental 1A→1D; all shapes parameterized in oracle/sim/RTL |

## §11 Sign-off items (disposition)

| # | Item | Disposition (project lead, 2026-09-17) |
|---|---|---|
| O1 | Output bus width (G6): H parallel lanes vs a single serialized stream | **H parallel lanes confirmed** — one per h-tile, per-lane `OUTPUT_CYC_prim` |
| O2 | Output-stage forms on `S` | **Removed in v0.3** — no ACAM after reduction; nonlinearity downstream |
| O3 | Padding drain (A5/§4.4) | **Full `H·C` bytes confirmed**; host ignores padded columns |

**Frozen 2026-09-17 (v0.3).** Stage 2 proceeds: `gemm_ref` (oracle) →
`gemm_sim` + `gemm_top` → GATE 2 harness, incremental 1A→1D.
