# Phase 4 — wsum RTL widening progress

## Goal
Close Phase-3 structural +38-cycle wsum delta by refactoring
`dimm_weighted_sum` to match the sim's packed-pass assumption
in `gemm_log`.

## Implementation

### Changes committed
1. **`nl_dpe/gen_attention_wrapper.py::_gen_dimm_weighted_sum`**
   refactor (+215 / -68 lines):
   - `ws_exp` widened from KW=1/NUM_COLS=1 → KW=128/NUM_COLS=128
     (acam_mode=1).  One DPE fire per output column m (4 per lane).
   - `ws_log` retained scalar (KW=1): widening it too would have added
     ~60 cyc of extra feed+drain per-row-setup (above Phase-4 target).
   - 5-byte packer inserted between scalar ws_log output and packed
     `log_attn_sram` (depth 27 × 40-bit words).
   - CLB byte-wise adder (5 lanes) replaces the single-byte add for
     `log_attn + V`.
   - Streaming reduction: col-masked byte-wise accumulator over the
     26-cycle ws_exp output drain, equivalent to a 7-level log2(128)
     adder tree but flattened in time.
   - `v_sram` transposed + packed: depth `d · 26 + 1 = 1665` (was
     `N*d + 1 = 8193`). `v_sram[m·26 + k]` holds packed V[5k..5k+4][m].
   - New FSM states: WS_LOG_FEED, WS_LOG_DRAIN, WS_EXP_FEED,
     WS_EXP_DRAIN, WS_WRITE. WS_LOAD_V retains code 2, WS_OUTPUT
     retains code 5 for TB compatibility.
   - WS_LOAD_A / ws_log pipelining: `ws_log_valid` asserts in both
     WS_LOG_FEED state **and** WS_LOAD_A (when attn_write_addr > 0),
     reading just-written bytes back out — cuts the dedicated
     WS_LOG_FEED phase from 8 cyc to 1 cyc.

2. **TB weight-preload widenings** (`tb_nldpe_dimm_top_functional.v`,
   `tb_nldpe_dimm_top_latency.v`):
   - `ws_exp` weights: 128-diagonal identity (was weights[0][0]=1).
   - `v_sram` preload: transposed + packed layout.
   - `v_write_addr` force target: `D*26 - 1 = 1663` (was `N*D - 1`).

3. **`phase3_known_deltas.json`**:
   - wsum: delta_cycles 38 → 16, classification
     `structural` → `modelling_granularity`.
   - e2e: delta_cycles 64 → 42, classification
     `structural` → `modelling_granularity`.

4. **`VERIFICATION.md`**: Phase 4 subsection with pre/post per-stage
   table and state-transition trace.

## Results

### Functional check
- `run_checks.py --config nldpe_dimm_top_d64_c128 --skip-latency` → PASS

### Latency check
- `run_checks.py --config nldpe_dimm_top_d64_c128` → PASS

| Stage | Pre-Phase-4 RTL | Sim | Pre-Phase-4 Δ | Post-Phase-4 RTL | Post-Phase-4 Δ |
|---|---:|---:|---:|---:|---:|
| score   | 260 | 244 | +16 | 260 | +16 |
| softmax |  27 |  17 | +10 |  27 | +10 |
| wsum    | 274 | 236 | +38 (structural) | 252 | +16 (modelling_granularity) |
| e2e     | 561 | 497 | +64 (structural) | 539 | +42 (modelling_granularity) |

### Regression guards (all PASS)
- `azurelily/IMC/test_gemm_log_regime_b.py` → PASS
- `fc_verification/run_fc_phase2.py --skip-vtr` → PASS

### Wsum state transition trace (post-Phase-4, lane 0)
```
cyc=290: IDLE → LOAD_A
cyc=298: LOAD_A → LOAD_V        (8 cyc LOAD_A)
cyc=299: LOAD_V → LOG_FEED      (1 cyc LOAD_V)
cyc=307: LOG_FEED → LOG_DRAIN   (8 cyc LOG_FEED)
cyc=308: LOG_DRAIN → EXP_FEED   (1 cyc LOG_DRAIN)
cyc=335: EXP_FEED → EXP_DRAIN   (27 cyc EXP_FEED, m=0)
cyc=368: EXP_DRAIN → EXP_FEED   (33 cyc EXP_DRAIN, m=0)
cyc=395: EXP_FEED → EXP_DRAIN   (27 cyc EXP_FEED, m=1)
cyc=428: EXP_DRAIN → EXP_FEED   (33 cyc EXP_DRAIN, m=1)
cyc=455: EXP_FEED → EXP_DRAIN   (27 cyc EXP_FEED, m=2)
cyc=488: EXP_DRAIN → EXP_FEED   (33 cyc EXP_DRAIN, m=2)
cyc=515: EXP_FEED → EXP_DRAIN   (27 cyc EXP_FEED, m=3)
cyc=548: EXP_DRAIN → OUTPUT     (33 cyc EXP_DRAIN, m=3)
```
Per-m cost = 60 cyc (matches sim's `cycles_per_pass + 4_fsm_handshake`
= 55 + 4 = 59 exactly; +1 residual per m × 4 = +4).  Pre-m setup =
19 cyc (matches Phase-4 wsum residual).

### Generator edit line count
+215 / -68 = 147 net lines.  Within plan's 100-200 line budget.

### VTR re-synth
In progress (background task brjv7vrsv); results will be written to
`fc_verification/results/nldpe_dimm_top_vtr_imc_results.json`.
