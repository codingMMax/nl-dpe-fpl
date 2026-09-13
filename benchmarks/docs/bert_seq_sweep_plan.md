# BERT-Tiny Seq_Len Sweep v2 — Issue Checklist & Plan

## Issues Identified

### Issue 1: SRAM Depth Packing (5× inflation)

All SRAMs use `DATA_WIDTH = 40` but store one int8 (8-bit) per entry.
Correct depth = `num_int8_elements × 8 / 40 = num_elements / 5`.

**Files to fix:**
- [ ] `gen_bert_tiny_wrapper.py:606` — Azure-Lily Q/K/V buffers: `S×d` → `S×d/5`
- [ ] `gen_bert_tiny_wrapper.py:983` — NL-DPE base_dimm_depth: `S×d` → varies per SRAM
- [ ] `gen_bert_tiny_wrapper.py:246` — clb_softmax sm_buf: `N` → `N/5`
- [ ] `gen_attention_wrapper.py:148,156` — K SRAM: `N*d` → `N*d/5`
- [ ] `gen_attention_wrapper.py:559` — V SRAM: `N*d` → `N*d/5`

### Issue 2: Wrong Buffer Sizes (NL-DPE DIMM modules)

All 10 SRAMs inside DIMM modules use `DEPTH = base_dimm_depth = S×d`,
but most only store S elements (row buffers) or d elements (vector buffers).

**Correct sizes (before packing):**

| SRAM | Module | What it stores | Elements |
|------|--------|----------------|----------|
| q_sram | dimm_score_matrix | One query vector | **d** |
| k_sram | dimm_score_matrix | All key vectors | **S×d** |
| score_sram | dimm_score_matrix | One score row | **S** |
| in_sram | softmax_approx | One score row | **S** |
| exp_sram | softmax_approx | One exp row | **S** |
| out_sram | softmax_approx | One normalized row | **S** |
| attn_sram | dimm_weighted_sum | One attn row | **S** |
| v_sram | dimm_weighted_sum | All V vectors | **S×d** |
| log_attn_sram | dimm_weighted_sum | One log(attn) row | **S** |
| out_sram | dimm_weighted_sum | One output row | **d** |

**Files to fix:**
- [ ] `gen_attention_wrapper.py` — pass separate depth params per SRAM instead of one shared DEPTH
- [ ] `gen_bert_tiny_wrapper.py:983` — compute per-SRAM depths instead of single base_dimm_depth

### Issue 3: dsp_mac int_sop_4 and K Parameter

The dsp_mac processes 5 int8 pairs per cycle via `int_sop_4`.
K (accumulation cycles) should be `inner_dim / 5`, not `inner_dim`.

**Files to fix:**
- [ ] `gen_bert_tiny_wrapper.py:634` — QK^T K: `D_HEAD` → `ceil(D_HEAD/5)` = 13
- [ ] `gen_bert_tiny_wrapper.py:661` — Score×V K: `seq_len` → `ceil(seq_len/5)`

### Issue 4: Simulator Latency — Row-Level Streaming Pipeline

The simulator models softmax as full-matrix I/O (`_run_softmax_exp` reads S×S,
`_run_softmax_norm` reads S×S). The RTL streams row-by-row.

- **Energy**: no change needed (same total bytes S×S regardless of streaming)
- **Latency**: TODO — implement row-level pipelining in simulator
  (QK^T row i overlaps with softmax row i-1, etc.)

**Files to fix:**
- [ ] `scheduler.py:_run_softmax_exp` — model as S iterations of S-element row I/O
- [ ] `scheduler.py:_run_softmax_norm` — same
- [ ] Consider overlap: QK^T row production can pipeline with softmax consumption

### Issue 5: Per-Category Energy Breakdown

The paper needs energy reported in 5 categories: **Crossbar, ADC/ACAM, CLB, DSP, BRAM**.

Current simulator keys → target categories:

| Target | Simulator keys |
|--------|---------------|
| **Crossbar** | `imc_vmm` + crossbar portion of `imc_dimm_exp`/`imc_dimm_log` |
| **ADC/ACAM** | `imc_conversion` + `imc_digital_post` + ACAM portion of `imc_dimm_exp`/`imc_dimm_log` |
| **CLB** | `clb_reduction` + `clb_exp` + `clb_norm_*` + `clb_add` + `clb_layernorm` + `clb_embed_add` + `fpga_activation` + `clb_compare` + `clb_subtract` |
| **DSP** | `dsp_gemm` + `dsp_add` + `mul` |
| **BRAM** | `sram_read` + `sram_write` |

**Problem**: `dimm_nonlinear()` in `imc_core.py:86-89` lumps crossbar + ACAM into
one `imc_dimm_exp` or `imc_dimm_log` bucket. For projections the split already exists
(`imc_vmm` vs `imc_digital_post`), but for DIMM it's merged.

**Config values** (from json):
- NL-DPE: `e_analoge_pj=3.89` (crossbar+DAC), `e_conv_pj=0`, `e_digital_pj=0.171` (ACAM/col)
- Azure-Lily: `e_analoge_pj=0`, `e_conv_pj=2.33` (ADC), `e_digital_pj=0`

**Fix**: Split `dimm_nonlinear()` to record 3 sub-components separately:
- `imc_dimm_{op}_vmm` → crossbar energy
- `imc_dimm_{op}_conv` → ADC energy (0 for NL-DPE)
- `imc_dimm_{op}_digital` → ACAM energy (0 for Azure-Lily)

Then aggregate into 5 categories in the plotting/CSV export scripts.

**Files to fix:**
- [ ] `imc_core.py:dimm_nonlinear()` — split into 3 sub-breakdown keys
- [ ] `stats.py` — add new breakdown keys
- [ ] `benchmarks/run_seqlen_imc.py` — aggregate into 5 categories for CSV output
- [ ] Plot scripts — use 5-category breakdown

### Issue 6: Azure-Lily DPE 16-bit Data Width

Azure-Lily DPE has 16-bit data_in/data_out (arch XML: `num_pins="16"`).
NL-DPE (Proposed/AL-Like) DPE has 40-bit (arch XML: `num_pins="40"`).

This means DPE-connected SRAMs have different packing:
- NL-DPE: 40-bit SRAM → 5 int8/entry → depth = N/5
- Azure-Lily: 16-bit SRAM → 2 int8/entry → depth = N/2

CLB and DSP always access full 40-bit BRAM width (both archs).

**Affected SRAMs**: only projection/FFN SRAMs inside `conv_layer_single_dpe`.
Azure-Lily Q/K/V intermediate buffers are DSP-connected (40-bit, PACK=5).

**RTL fix**: Azure-Lily `conv_layer_single_dpe` uses `DATA_WIDTH=16`.
**Simulator fix**: DPE memory bandwidth = 2 bytes/cycle for Azure-Lily.

- [ ] `gen_bert_tiny_wrapper.py` — pass DATA_WIDTH=16 for Azure-Lily projections
- [ ] `azure_lily.json` or scheduler — DPE bram_width=16 for latency

### Issue 7: Azure-Lily DSP Retention (parmys merging)

`dsp_mac` uses `*` operator which parmys merges aggressively.
Retention drops from 108% → 53% as S grows.

**Previous attempt**: `int_sop_4` hard block (should be un-mergeable like `dpe`).
- [ ] Verify `int_sop_4` dsp_mac with VTR — check DSP retention ≈ 100%
- [ ] If retention OK: regenerate all Azure-Lily RTL

## Buffering Strategy: Row-by-Row Streaming

```
                    ┌─── repeated S times ───┐
Proj_Q ──► [Q: d]──┤                        │
Proj_K ──► [K: S×d]──► QK^T row ──► [S] ──► Softmax ──► [S] ──► Score×V row
Proj_V ──► [V: S×d]─────────────────────────────────────────┘      │
                    └────────────────────────────────────── [d] ◄──┘
```

- **O(S×d)**: K and V (must store all tokens, re-read S times)
- **O(S)**: score row, softmax row, attn row (one row live at a time)
- **O(d)**: Q (one token), output (one row)

The score matrix is S×S total data, but produced/consumed row-by-row.
Buffer cost is O(S), not O(S²). Total traffic is still O(S²).

## Correct SRAM Depths (with packing)

`DEPTH = num_int8_elements × 8 / 40`

### NL-DPE (per head)

| SRAM | Elements | DEPTH | Current |
|------|----------|-------|---------|
| q_sram | d = 64 | **13** | S×d |
| k_sram | S×d | **S×d/5** | S×d |
| score_sram | S | **S/5** | S×d |
| in_sram | S | **S/5** | S×d |
| exp_sram | S | **S/5** | S×d |
| out_sram (softmax) | S | **S/5** | S×d |
| attn_sram | S | **S/5** | S×d |
| v_sram | S×d | **S×d/5** | S×d |
| log_attn_sram | S | **S/5** | S×d |
| out_sram (wsum) | d = 64 | **13** | S×d |

### Azure-Lily (per head)

| SRAM | Elements | DEPTH | Current |
|------|----------|-------|---------|
| q_buf | S×d | **S×d/5** | S×d |
| k_buf | S×d | **S×d/5** | S×d |
| v_buf | S×d | **S×d/5** | S×d |
| sm_buf | S | **S/5** | S |

## Execution Plan

1. [ ] Fix RTL generator: SRAM depths (Issues 1 + 2)
2. [ ] Fix RTL generator: dsp_mac K parameter (Issue 3)
3. [ ] Test int_sop_4 dsp_mac with VTR for one Azure-Lily design (Issue 6)
4. [ ] Regenerate all 18 RTL files (3 archs × 6 seq_lens)
5. [ ] Run VTR sweep (54 runs: 18 × 3 seeds)
6. [ ] Verify resource counts (DPE, DSP, BRAM) against expected
7. [ ] Split dimm_nonlinear() energy into crossbar/ACAM sub-keys (Issue 5)
8. [ ] Add 5-category energy aggregation to IMC runner + CSV (Issue 5)
9. [ ] Update simulator: row-level streaming latency (Issue 4)
10. [ ] Run IMC simulator with updated model
11. [ ] Regenerate all paper figures with 5-category energy breakdown
