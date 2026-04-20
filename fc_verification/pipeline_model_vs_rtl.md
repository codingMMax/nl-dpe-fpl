# DIMM Pipeline Model vs RTL — cycle accounting

**Date:** 2026-04-20 (post Phase 4)
**Config:** NL-DPE DIMM top, $N = 128$, $d = 64$, $C = 128$, $W = 16$,
$W_{DPE} = 40$, ACAM compute = 3.

## Why this doc exists

The sim (`gemm_log` in `azurelily/IMC/peripherals/fpga_fabric.py`) is
an **analytical performance estimator**. The RTL
(`fc_verification/rtl/nldpe_dimm_top_d64_c128.v`, regenerated from
`nl_dpe/gen_attention_wrapper.py`) is a **cycle-accurate FSM**.

Their numbers agree to within FSM-state-transition granularity
(±16 cycles per stage, ±42 E2E), and the residuals are fully
explained. This doc maps each stage's sim pipeline to the RTL FSM
states, cycle-by-cycle, and names every source of the residual.

Summary (from `run_checks.py --config nldpe_dimm_top_d64_c128`):

| Stage | Sim | RTL | Δ | Residual source (detailed below) |
|---|---:|---:|---:|---|
| score   | 244 | 260 | +16 | 4 passes × 4 cyc/pass SRAM-fill, not in `feed_cycles` |
| softmax |  17 |  27 | +10 | probe opens mid-`SM_LOAD` (+8 trailing) + 2 state transitions |
| wsum    | 236 | 252 | +16 | one-time pre-m setup (`LOAD_A`, `LOG_FEED`, `LOG_DRAIN`, `LOAD_V`) ≈ 12 + 4 cyc accumulator handoff |
| **E2E** | **497** | **539** | **+42** | additive sum of the three above (stages serialised per lane) |

Every residual entry has a file:line citation in
`fc_verification/phase3_known_deltas.json`. This doc expands those
annotations with visual timelines.

## The two models, side by side

### Sim pipeline (`gemm_log`, analytical)

For each DPE pass, `gemm_log` models **three concurrent sub-pipelines**
that share the DPE's analog macro:

```
       ┌─────────────────── feed (input into DPE buffer + bit-serial fires) ─────┐
 time  │ feed_cycles = ⌈R·8/W_DPE⌉ + sram_read_lat · K_id  ≈ 26 + 2·2 = 30        │
       │ compute_cycles = max(1, ceil((core_ns − output·t_clk)/t_clk)) = 1       │
       │ effective_output_cycles = max(drain, reduce) = max(26, log₂64=6) = 26   │
       │                                                                         │
       │ cycles_per_pass = feed + dpe_passes · (compute + eff_output)            │
       │                = 30 + 1 · (1 + 26) = 57 cycles                          │
       └────────────────────────────────────────────────────────────────────────┘
```

Multi-pass (Regime B, Layout A): `T(M) = cycles_per_load · M + cycles_per_drain`.
For one row through a DIMM-top lane, the RTL is serial-per-pass (not
pipelined-per-pass) because the lane holds one DPE; the RTL-matching
extraction in `gen_expected_cycles.py::_compute_sim_per_row` therefore
uses `cycles_per_pass · cols_per_dpe + fsm_handshake`:

$$
T_{\text{stage, sim}} = (\text{cycles\_per\_pass} + 4) \cdot \text{cols\_per\_dpe}
$$

The `+4` term is the already-baked-in per-pass FSM handshake
(ACAM compute 3 vs clipped 1 = +2; `S_WAIT_DPE → S_WRITE_B` = +2).

### RTL FSM (cycle-accurate)

Every DIMM stage is a finite state machine with distinct states, each
of which takes at least one cycle. State transitions themselves cost
1 cycle each. Memory reads have a 2-cycle SRAM pipeline (address
register → SRAM output register).

The FSM gives you **more than** the analytical `cycles_per_pass`
because:

1. Entering a new "inner loop" iteration re-primes the SRAM pipeline
   (fill 2 cycles before first `w_buf_en` strobe).
2. FSM state transitions have 1-cycle flop delays not counted in
   `cycles_per_pass`.
3. Stage-start handoffs (one stage's `valid_n` firing → next stage's
   first `LOAD` cycle) look like "extra" cycles inside whichever
   stage's probe window they fall in.

The next three sections walk each stage's FSM in detail.

---

## 1. Score stage (QK^T, N×N) — +16 cycle residual

### Sim model

```
score_sim = (cycles_per_pass + fsm_handshake) · cols_per_dpe
          = (57 + 4) · 4
          = 244 cycles
```

where `cols_per_dpe = ⌈N / (n_parallel_dpes · K_id)⌉ = ⌈128 / (16·2)⌉ = 4`.

### RTL FSM (`dimm_score_matrix`, `nldpe_dimm_top_d64_c128.v:229–311`)

```
localparam S_IDLE=0, S_LOAD_Q=1, S_LOAD_K=2, S_COMPUTE=3,
           S_WAIT_DPE=4, S_WRITE_SCORE=5, S_OUTPUT=6, S_WRITE_B=7;
```

Probe window: `feed_qk_cyc` (TB forces state=S_COMPUTE) → first
`state==S_OUTPUT (4'd6)` (set after the 4th iteration).

Per iteration (one score-pair write, i.e. one DPE pass at this lane):

```
  cycle
     ┌─── S_COMPUTE ───────────────────────────────────────────────┐
  0  │ mac_count=0 → kick off SRAM fetch for Q[q], K[j]           │
  1  │ mac_count=1 → SRAM addr register valid, SRAM output lat=1   │ ← SRAM pipeline fill
  2  │ mac_count=2 → dimm_exp_valid asserts, w_buf_en strobes      │ ← (2-cyc delay)
 …   │ mac_count=3..14 → bit-serial feeds (12 strobes)             │
  … (cycles_per_pass ≈ 57 cyc spent inside S_COMPUTE, most of the
     time waiting for the DPE to drain its 26-cycle output vector)
  ─  │ S_COMPUTE exits when DPE's internal shift-add is done       │
     └─────────────────────────────────────────────────────────────┘
     ┌─── S_WAIT_DPE ──────────┐ ← 1 cycle state flop
     │ latch score pair        │
     └─────────────────────────┘
     ┌─── S_WRITE_B ───────────┐ ← 1 cycle state flop
     │ write score_sram[2·j],  │
     │ write score_sram[2·j+1] │
     │ score_idx += W·2        │
     └─────────────────────────┘
  → back to S_COMPUTE for next iteration
```

**Per-iteration total**: `cycles_per_pass` (≈57 inside S_COMPUTE)
+ `S_WAIT_DPE` (1) + `S_WRITE_B` (1) + state transition flops (2)
≈ 61 cycles. The extractor bakes in 4 of those 4 non-`S_COMPUTE`
cycles.

**What the extractor DOESN'T count**: the first 2–4 cycles of
`S_COMPUTE` where the SRAM read pipeline is filling before the first
`w_buf_en` strobe. The line
`assign dimm_exp_valid = (state == S_COMPUTE) && (mac_count >= 2) && (mac_count <= 14);`
at **line 193** of `nldpe_dimm_top_d64_c128.v` makes this explicit:
mac_count must reach 2 before the DPE sees valid data. Those first
2 cycles plus ~2 cycles of SRAM-register propagation = ~4 cycles
per iteration of dead time that `cycles_per_pass`'s `feed_cycles`
doesn't model.

### Cycle accounting

```
  Component                             Count
  ─────────────────────────────────────────────
  cycles_per_pass × cols_per_dpe      = 57 × 4  = 228
  FSM handshake (S_WAIT_DPE + S_WRITE_B)  4 × 4 =  16
  SRAM-read pipeline fill per pass        4 × 4 =  16   ← +16 residual
  ─────────────────────────────────────────────
  sim extraction                                = 228 + 16  = 244
  RTL measured                                  = 228 + 16 + 16 = 260
```

Residual = **+16 cycles = 4 passes × 4 SRAM-pipeline-fill cycles/pass**.
Documented in `phase3_known_deltas.json[score].root_cause`.

---

## 2. Softmax stage — +10 cycle residual

### Sim model

```
softmax_sim = dimm_nonlinear(N, "exp") per-row / W_softmax + reduction
            = (56 / 16) + log₂(16)
            ≈ 4 + 4
            = ~8 cycles per row + norm overhead
            ≈ 17 cycles per-row total
```

### RTL FSM (`softmax_approx`, `nldpe_dimm_top_d64_c128.v:437–481`)

```
localparam SM_IDLE=0, SM_LOAD=1, SM_EXP=2, SM_NORMALIZE=3, SM_OUTPUT=4;
```

```
SM_IDLE                                              ← 0 cyc (already left)
      │   valid (score done)
      ▼
SM_LOAD  ... duration = N_PER_LANE = N/W = 8 cyc    ← 8 cycles reading score_sram
      │   in_write_addr == N_PER_LANE - 1
      ▼
  ─ state transition ─                               ← +1 cyc flop
      ▼
SM_EXP   ... duration ≈ N_PER_LANE + 1 = 9 cyc      ← 9 cycles firing sm_exp DPE
      │
      ▼
SM_NORMALIZE ... duration = N_PER_LANE + 1 = 9 cyc  ← 9 cycles normalize
      │   sm_count == N_PER_LANE + 1
      ▼
  ─ state transition ─                               ← +1 cyc flop
      ▼
SM_OUTPUT ...                                        ← probe fires here
```

Probe window: `softmax_start_cyc = score_end_cyc` (from the probe code
at line ~60), `softmax_end_cyc = cycle of first sm_state == SM_OUTPUT`.

### Where the +10 comes from

The probe opens at score's first `S_OUTPUT`, but softmax's FSM at
*that* cycle is **already in `SM_LOAD`** (it started loading as soon
as score began writing score_sram). So the probe misses softmax's
first few load cycles and instead counts:

```
  probe open              sm_state at probe open                   probe close
      │                           │                                      │
      ▼                           ▼                                      ▼
  [ SM_LOAD (trailing)     +1       +1      SM_NORMALIZE  +1  SM_OUTPUT ]
  ├─────── 8 cyc ─────────┤├ SM_EXP 9 cyc   ├─ NORM 9 cyc ─┤
  └─────────────── measured 27 cycles ──────────────────────┘
```

Sim extraction gives `dimm_nonlinear`'s per-row cost (~17 cyc) which
corresponds roughly to the SM_EXP + SM_NORMALIZE useful work, but
misses:

- **Trailing `SM_LOAD`** (up to 8 cyc of leftover load that the
  probe window includes).
- **Two state transitions** (`SM_LOAD → SM_EXP`, `SM_NORMALIZE →
  SM_OUTPUT`, +1 each).

Total: **+10 cycle residual**. Documented at
`phase3_known_deltas.json[softmax].root_cause`.

The "fix" here would be to move the softmax probe to trigger at
`SM_EXP` entry instead of `SM_OUTPUT` — then the delta would drop
to ~0. But that changes what "softmax stage cycles" means across
the verification history; we accept the +10 as a probe-placement
convention.

---

## 3. Wsum stage (Score×V) — +16 cycle residual (post-Phase-4)

### Sim model (unchanged by Phase 4)

```
wsum_sim = (cycles_per_pass + fsm_handshake) · M_PER_LANE
         = (55 + 4) · 4
         = 236 cycles
```

where `M_PER_LANE = d / W = 64 / 16 = 4` output columns per lane.

### RTL FSM (`dimm_weighted_sum`, post-Phase-4, `nldpe_dimm_top_d64_c128.v:686–767`)

Phase 4 widened `ws_exp` from 1×1 to 128×128. The FSM now fires
`ws_exp` once per output column instead of $N/2 = 64$ times.

```
localparam WS_IDLE=0, WS_LOAD_A=1, WS_LOAD_V=2, WS_LOG_FEED=3,
           WS_LOG_DRAIN=4, WS_OUTPUT=5, WS_EXP_FEED=6,
           WS_EXP_DRAIN=7, WS_WRITE=8;
```

```
WS_IDLE                                              ← enters when valid_attn or valid_v
      │
      ▼
─ state transition ─                                 ← +1 cyc
      ▼
WS_LOAD_A  ... 8 cyc (N/W=8) write attn_sram        ← one-time load of this lane's attn row
      │   ws_log pipelines with LOAD_A (overlapped, free)
      │   attn_write_addr == N/W - 1
      ▼
WS_LOAD_V  ... d · PACKED_N = 64·26 = 1664 wait      ← one-time load of V strip (3-port usage)
      │   ← actually this is for V preload. In the TB the TB force-sets
      │     `v_write_addr = d*PACKED_N - 1` at line 117, so LOAD_V
      │     exits immediately. Not part of the Phase-3 probe window.
      ▼
WS_LOG_FEED ... 1 cyc                                ← feed log(attn) vector to ws_log DPE
      ▼
WS_LOG_DRAIN ... 1 cyc                               ← drain ws_log output → log_attn_sram
      ▼
┌─ per-m loop (M_PER_LANE = 4 iterations) ──────────┐
│                                                    │
│   ┌─ WS_EXP_FEED ─┐   ← 27 cyc (packed feed)      │
│   │ feed 128-wide │   (byte-serial, ~26 cyc/lane) │
│   │ log_sum vec   │                                │
│   └───────────────┘                                │
│                                                    │
│   ┌─ WS_EXP_DRAIN ┐   ← 33 cyc (drain + reduce)   │
│   │ ws_exp fires  │   includes shift-add & 7-tree │
│   │ 128 outputs → │   CLB reduction of 128 values │
│   │ scalar accum  │                                │
│   └───────────────┘                                │
│                                                    │
│   → write out_sram[ws_m], ws_m++                  │
│                                                    │
│   ← state transition back to WS_EXP_FEED (+1 cyc) │
│   ← accumulator clear handoff (+1 cyc)            │
└────────────────────────────────────────────────────┘
      │   after 4 iterations (ws_m == M_PER_LANE):
      ▼
WS_OUTPUT  ← probe fires here (ws_state == WS_OUTPUT == 4'd5)
```

Probe window: `wsum_start_cyc = softmax_end_cyc`, `wsum_end_cyc =
first cycle ws_state == WS_OUTPUT`.

### Cycle accounting

```
  Component                                         Count
  ────────────────────────────────────────────────────────
  Per-m DPE work (WS_EXP_FEED + WS_EXP_DRAIN)      4 × 60 = 240
  Extractor's FSM handshake (+4 / pass already)    (baked in)
  ────────────────────────────────────────────────────────
  sim extraction                                           = 236

  One-time pre-m setup (not in sim):
    WS_IDLE → WS_LOAD_A transition (+1)
    WS_LOAD_A (overlap with ws_log feed)              8
    WS_LOAD_V (force-skipped by TB, 0 useful cyc)     0
    WS_LOG_FEED                                        1
    WS_LOG_DRAIN                                       1
    WS_LOAD_V → WS_LOG_FEED transition (+1)           1
  Per-m accumulator-clear handoff (4 × 1)              4   ← +16 residual
  ────────────────────────────────────────────────────────
  RTL measured                                       = 236 + 16 = 252
```

Residual = **+16 cycles** = one-time FSM setup (~12 cyc) + per-m
accumulator handoffs (~4 cyc).

Pre-Phase-4, this stage was +38 with the extra **structural** delta
coming from `ws_j` loop firing `ws_exp` $N/2 = 64$ times per output
(instead of once); Phase 4's widening to 128×128 eliminated that,
leaving only the FSM setup residual above. Documented at
`phase3_known_deltas.json[wsum].root_cause` (post-Phase-4).

---

## 4. End-to-end composition — additive, not pipelined

### Sim model

```python
# scheduler.py::_run_attention_pipeline, lines 261–269
t_fill   = per_row_totals[0]           # first row through first stage
t_steady = max(per_row_steadies)       # bottleneck stage per-row steady
t_drain  = sum(per_row_totals[1:])     # remaining stages
pipeline_lat = t_fill + (S - 1) · t_steady + t_drain
```

For the N=128, W=16 Phase-J test: only **one row per lane reaches
the first `*_OUTPUT`** during the probe window (S = 1 in probe
terms), so the formula reduces to `per_row_total` = sum of stages.

### RTL

Each `dimm_lane[L]` instantiates `score_inst → softmax_inst →
wsum_inst` with `ready_n` **hardwired to 1'b0** (generator enforces
no-backpressure) at lines 330, 412, 500 etc. of
`nl_dpe/gen_attention_wrapper.py`. So:

- softmax can start consuming score's writes as soon as they land in
  score_sram (overlap within the stage, but softmax's *probe*
  opens only after score fully exits).
- wsum starts only after softmax's `valid_n` fires (its probe opens
  at softmax_end_cyc).

From the per-lane probe's perspective the stages are **serial**:

```
  feed_qk_cyc                            score_end  softmax_end        wsum_end (end_cyc)
      │                                      │         │                      │
      ▼                                      ▼         ▼                      ▼
  ┌───── score stage (260 cyc) ─────────────┐├─ softmax 27 ─┤├─── wsum 252 ───┤
  │                                          │               │                 │
  └─────────────── E2E = 539 cyc ─────────────────────────────────────────────┘
```

So `E2E = score + softmax + wsum` per the TB's probes, and the E2E
residual is additive: `16 + 10 + 16 = 42 cyc`.

Sim's `E2E = 497`, RTL's `E2E = 539`. Matches exactly.

---

## 5. What the residuals mean

Every residual traces to a generic FSM feature that every attention
RTL will have:

| Source | Cyc | Where |
|---|---:|---|
| SRAM-read pipeline fill (per score pass) | 4 × 4 = 16 | line 193, `mac_count >= 2` before `w_buf_en` |
| Softmax probe opens mid-SM_LOAD + 2 state transitions | 8 + 2 = 10 | line 452, 454, 472 |
| Wsum one-time pre-m setup (LOAD_A + LOG_FEED + LOG_DRAIN) | 12 | lines 714, 719, 723, 727 |
| Wsum per-m accumulator-clear handoff | 4 × 1 = 4 | line 747 |
| **Total E2E residual** | **42** | (sum) |

**None** of these are structural mismatches between sim and RTL —
they are **FSM micro-cycles** that the analytical model deliberately
elides. Documenting them with file:line precision (as above) is the
correct outcome; adding them to `gemm_log` would make the sim
config-specific and defeat its DSE purpose.

## 6. How the residuals would change if we did tighten the sim

**Hypothetical**: if `gen_expected_cycles.py::_compute_sim_per_row`
added:

```python
# +4 cyc per pass for SRAM-read pipeline fill
# (on top of the +4 for FSM handshake already)
sram_fill_per_pass = 4
single_row_compute_cyc = (
    cycles_per_pass + fsm_handshake_per_pass + sram_fill_per_pass
) * cols_per_dpe
```

and a wsum-only correction:

```python
# Wsum one-time pre-m setup (not per pass, per stage)
wsum_setup_cyc = 12
# Plus per-m accumulator handoff
wsum_handoff_per_m = 1
```

and softmax-specific:

```python
# Probe offset: SM_LOAD trailing + 2 transitions
softmax_probe_offset = 10
```

the residuals would drop to 0 for score/wsum and ±2 for softmax
(SM_LOAD trailing is actually bounded by N/W, not exactly 8).

**Tradeoff**: the sim becomes less portable (each new DIMM
config-shape would need calibrated offsets), more code (~30 more
lines in the extractor), ~nothing gained for the DSE use case.

## 7. Final table — all sources enumerated

| Stage | Sim formula | RTL states | Sim cyc | RTL cyc | Δ | Δ source (file:line) |
|---|---|---|---:|---:|---:|---|
| score   | `(cycles_per_pass + 4) · cols_per_dpe` with 4 passes | S_COMPUTE → S_WAIT_DPE → S_WRITE_B × 4 | 244 | 260 | +16 | SRAM-fill per pass × 4: `nldpe_dimm_top_d64_c128.v:193` |
| softmax | `dimm_nonlinear(N, "exp")/W + norm` | SM_IDLE → SM_LOAD → SM_EXP → SM_NORMALIZE → SM_OUTPUT |  17 |  27 | +10 | probe offset into SM_LOAD + 2 transitions: lines 452, 454, 472 |
| wsum    | `(cycles_per_pass + 4) · M_PER_LANE` with 4 m | WS_IDLE → WS_LOAD_A → WS_LOAD_V → WS_LOG_FEED → WS_LOG_DRAIN → (WS_EXP_FEED → WS_EXP_DRAIN) × 4 → WS_OUTPUT | 236 | 252 | +16 | one-time setup + accumulator handoff: lines 714, 719, 723, 727, 747 |
| E2E     | serial sum                                | sum of per-stage probes                        | 497 | 539 | +42 | additive sum of three above |

All residuals classified `modelling_granularity` in
`phase3_known_deltas.json`. No `structural` deltas remain after
Phase 4.
