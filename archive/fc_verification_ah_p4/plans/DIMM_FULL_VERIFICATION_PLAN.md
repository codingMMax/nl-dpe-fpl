# DIMM Full Verification Plan — W=16, RTL-accurate simulator alignment

## Context

The user is executing a strict V-model workflow for DIMM verification:

```
1. Fix mapping strategy                (paper/methodology/attention_dimm_mapping.md)
2. Write RTL                           (implement mapping faithfully)
3. Verify RTL                          (functional + latency vs simulator)
4. Align IMC simulator                 (match exact RTL pipeline structure)
5. Invoke VTR flow                     (using verified RTL)
6. Verify VTR metrics                  (Fmax, resource count sanity)
7. Run IMC with VTR-measured metrics   (real Fmax → real energy/latency)
8. Generate comparison                 (block-level plot with real data)
```

Current work done (per-stage verification at W=1) is a **starting point**, not the
deliverable. We need to go through all 8 steps for the **full DIMM** (DIMM1 +
softmax + DIMM2) at **W=16** parallelism.

Scope is **DIMM only** — not the full attention head. The DIMM top module must
be composable: DIMM top + FC top wired together = full attention.

## Frozen decisions

1. **Parallelism**: W=16 for both architectures (lane-matched). Applied to all 4 DIMM
   stages for NL-DPE (score_exp, softmax_exp, wsum_log, wsum_exp). For Azure-Lily,
   applied to DIMM1 (16 dsp_mac for mac_qk) and DIMM2 (16 dsp_mac for mac_sv).
2. **Softmax parallelism**: **W=16** (not shared). A shared softmax would bottleneck
   DIMM1 by ~15×. 16 softmax lanes, each processing one attention row at a time.
3. **Azure-Lily `dsp_mac`**: drop the CLB-multiply helper (5th element). Use only
   `int_sop_4` → pure 4 MAC/cycle. Matches simulator's DSP_WIDTH=4 model exactly.
4. **Dimensions**: N=128, d=64 (BERT-Tiny) for **both** RTL and simulator. No
   scaled-down tests. Functional tests use structured inputs (one-hot, identity)
   so expected outputs are hand-computable even at N=128.
5. **VTR strictness**:
   - **DPE count**: must match exactly (hard block, explicit instantiation)
   - **DSP count**: ±small drift OK (VTR packing variation)
   - **Fmax, CLB, BRAM**: accept whatever VTR reports
6. **RTL ≡ simulator**: no "simulator models more parallelism than RTL" shortcuts.
   Both must have identical structural pipeline and W=16 parallelism.
7. **Doc/code alignment**: fix stale comments in `gen_bert_tiny_wrapper.py` and
   bring `attention_dimm_mapping.md` consistent with the actual architecture
   used in comparison.

## Total DPE/DSP count per head (target)

| | NL-DPE (Proposed) | Azure-Lily |
|---|---|---|
| score_exp (DIMM1) | 16 DPEs | 16 `dsp_mac` |
| softmax_exp | 16 DPEs | (part of `clb_softmax` — 16 lanes) |
| wsum_log (DIMM2 step 1) | 16 DPEs | N/A |
| wsum_exp (DIMM2 step 3) | 16 DPEs | 16 `dsp_mac` |
| **Total compute primitives** | **64 DPEs** | **32 DSPs** + shared CLB softmax |

## Phase H — Full DIMM top RTL (W=16)

### H.1 NL-DPE full DIMM top

File: `fc_verification/rtl/nldpe_dimm_top_d64_c128.v`

Interface:
```verilog
module nldpe_dimm_top #(
    parameter N = 128, D = 64, DATA_WIDTH = 40
)(
    input  wire clk, rst,
    input  wire valid_q, valid_k, valid_v,
    input  wire [DW-1:0] data_in_q, data_in_k, data_in_v,
    output wire [DW-1:0] data_out,            // attention output (N×d packed)
    output wire ready_q, ready_k, ready_v,
    output wire valid_n
);
```

Internal structure:
- Q/K/V input SRAMs (from upstream FC, pre-loaded for standalone verification)
- 16 × `score_exp` lanes: CLB byte-wise add (Q+K) → DPE(I|exp, K_qkt=2) → CLB reduce
  → accumulator → score BRAM (128×128)
- 16 × `softmax_exp` lanes: DPE(I|exp) on score row → shared CLB sum → CLB recip →
  CLB mul → attn-weight BRAM (128×128)
- 16 × `wsum_log` lanes: DPE(I|log) on attn weights → log-domain BRAM (128×128)
- 16 × `wsum_exp` lanes: CLB byte-wise add (log_attn + log_V) → DPE(I|exp, K_sv=1) →
  CLB reduce → output BRAM (128×64)

Generator: `nl_dpe/gen_dimm_nldpe_top.py` (new) — parameterized by N, d, C, W.

### H.2 Azure-Lily full DIMM top

File: `fc_verification/rtl/azurelily_dimm_top_d64_c128.v`

Interface: identical to NL-DPE top.

Internal structure:
- Q/K/V input SRAMs
- 16 × `dsp_mac` (K=ceil(d/epw)=13): one dsp_mac per attention-row lane, computes
  QK^T dot product using int_sop_4 only (**4 MAC/cycle, no CLB helper**)
- Score BRAM (128×128)
- 16 × `clb_softmax` lanes: one per attention row. Existing `clb_softmax` module
  (already patched for SRAM[0] and last-output bugs) stamped out 16 times.
- Attn-weight BRAM (128×128)
- 16 × `dsp_mac` (K=ceil(N/epw)=26): one per output column lane, computes S×V
- Output BRAM (128×64)

Generator: `nl_dpe/gen_dimm_azurelily_top.py` (new).

**Modified `dsp_mac`**: remove `p4 = data_a[39:32] * data_b[39:32]` CLB multiply.
Use only the `int_sop_4` result. This makes `dsp_mac` a pure 4-wide DSP MAC. The
5th int8 in the 40-bit bus is ignored (or forced to zero) — this is an
acknowledged minor loss of bus density, in exchange for RTL ≡ simulator.

## Phase I — Full DIMM RTL verification

### I.1 Functional tests (N=128, d=64)

Structured test patterns so expected outputs are hand-computable:

**NL-DPE top functional:**
- Q = one-hot at index 0: [1, 0, ..., 0]
- K matrix: K[i][j] = δ(i, j) for i, j < d (identity-extended)
- V matrix: V[i][j] = i (attention-weighted output = identity function on row indices)
- Expected output[m][k] for each output row m, channel k: computable analytically.

For verification we check:
- Score matrix values (intermediate BRAM dump) match expected QK^T scores
- Softmax output values match expected normalized exp
- Final output values match expected S×V

**Azure-Lily top functional**: same inputs, same expected outputs (within int8 arithmetic
tolerance). Both architectures compute the same function.

Files:
- `tb_nldpe_dimm_top_functional.v`
- `tb_azurelily_dimm_top_functional.v`

### I.2 Latency tests (N=128, d=64)

Measure end-to-end cycle count (first input valid → last output valid).
Compare against simulator's attention model prediction (with `total_dimm_dpes=16`
or `total_dsp=16`). Iterate until delta is within documented FSM-overhead tolerance
(typically ≤20 cycles per head-level iteration).

Files:
- `tb_nldpe_dimm_top_latency.v`
- `tb_azurelily_dimm_top_latency.v`

### I.3 Iteration loop

If latency doesn't match:
1. Identify which stage has the mismatch (intermediate BRAM timestamp probes)
2. Debug: either RTL pipeline-overlap bug or simulator-model bug
3. Fix the mismatch on whichever side is wrong per first principles
4. Re-run

If functional fails:
1. Per-stage dump: check each intermediate BRAM
2. Isolate the broken stage's RTL
3. Fix and re-run

## Phase J — Simulator alignment to full DIMM

**J.1**: Run `azurelily/IMC/test.py --model attention` with N=128, d=64 and:
- nl_dpe.json: total_dimm_dpes=16
- azure_lily.json: total_dsp=16 (after dsp_mac becomes pure 4-wide)

Verify simulator's end-to-end cycle prediction matches RTL Phase I.2 measurement.

**J.2**: If mismatch, debug:
- Is the simulator's `_run_attention_pipeline` correctly handling the W=16 case?
- Is the per-row timing (`row_timing` dict returned by gemm_log/gemm_dsp)
  correctly reflecting W=16?
- Are inter-stage buffer latencies modeled correctly?

**J.3**: Document the aligned end-to-end cycle count.

## Phase K — VTR for full DIMM

### K.1 NL-DPE VTR orchestrator

File: `fc_verification/gen_nldpe_dimm_top_vtr.py`
- Runs VTR 3 seeds on `nldpe_dimm_top_d64_c128.v`
- Parses: Fmax avg, CLB, BRAM, DPE counts, DSP count

### K.2 Azure-Lily VTR orchestrator

File: `fc_verification/gen_azurelily_dimm_top_vtr.py`
- Runs VTR 3 seeds on `azurelily_dimm_top_d64_c128.v`
- Parses: Fmax avg, CLB, BRAM, DSP count

## Phase L — VTR metric verification

| Metric | NL-DPE expected | Azure-Lily expected | Strictness |
|---|---|---|---|
| DPE count | 64 | 0 | Exact |
| DSP count | 0 | 32 ± small | ±packing drift |
| Fmax | — | — | Accept |
| CLB | — | — | Accept |
| BRAM | — | — | Accept |

If DPE count is off → RTL has an instantiation bug, fix and re-run.
If DSP count is way off → sanity-check VTR arch XML supports `int_sop_4`.
Fmax/CLB/BRAM are informational.

## Phase M — IMC with VTR metrics

- Patch nl_dpe.json with VTR-measured NL-DPE Fmax
- Patch azure_lily.json with VTR-measured Azure-Lily Fmax
- Re-run IMC for each architecture
- Regenerate block-level comparison plot

## Phase N — Docs & comments cleanup

### N.1 Update `paper/methodology/attention_dimm_mapping.md`

Existing claims:
- "Azure-Lily has 261 DPEs but cannot use them for DIMM (no log-domain support).
  Its DPE columns replaced all DSP columns (0 DSPs remaining)"
- "Azure-Lily (DSP) | N/A | N/A | N/A | **0** (0 DSPs) | 7% idle"

These describe the **extreme / paper-canonical** Azure-Lily. Our DIMM-only
comparison uses a lane-matched **W=16 DSP lane Azure-Lily** for fair comparison.
Add a clarifying section:

> ### DIMM-only comparison setup (verification)
> The "0 DSPs" Azure-Lily in §6 is the canonical paper architecture with all
> DSP columns replaced by DPE columns. For apples-to-apples DIMM verification
> against NL-DPE's W=16 mapping, we instantiate a DIMM-only Azure-Lily variant
> with W=16 `dsp_mac` lanes (each = one `int_sop_4` block, 4 MAC/cycle).
> This matches the lane count of NL-DPE's W=16 DPE configuration at the
> "one compute primitive per lane" level, preserving the ~25× per-lane MAC
> throughput difference that is the paper's central claim.

### N.2 Fix `gen_bert_tiny_wrapper.py` comments

Line 441: `// Projections/FFN: DPE, DIMM: CLB MAC (0 DSPs)` is wrong — the actual
implementation uses `dsp_mac`. Correct to:

```
// Projections/FFN: DPE, DIMM: DSP MAC (int_sop_4, W=ceil(S/C) lanes)
```

Also the `_gen_dsp_mac_module` header comment mentions "5th element via CLB" —
update when we drop the CLB helper in Phase H.

### N.3 VERIFICATION.md

Add sections for:
- Full DIMM top-level verification (Phase H-I results)
- VTR-backed block-level comparison (Phase K-M)
- Mapping-to-RTL consistency note

### N.4 Alignment logs

- `fc_verification/results/nldpe_dimm_top_alignment_log.txt`: full end-to-end
  NL-DPE DIMM alignment (functional + latency + VTR)
- `fc_verification/results/azurelily_dimm_top_alignment_log.txt`: same for AL

## Files to create (summary)

**Generators (nl_dpe/):**
- `gen_dimm_nldpe_top.py` — W=16 full DIMM top generator for NL-DPE
- `gen_dimm_azurelily_top.py` — W=16 full DIMM top generator for Azure-Lily

**RTL (fc_verification/rtl/):**
- `nldpe_dimm_top_d64_c128.v`
- `azurelily_dimm_top_d64_c128.v`
- Modified `dsp_mac` (embedded in AL RTL) — no CLB helper

**Testbenches (fc_verification/):**
- `tb_nldpe_dimm_top_functional.v`
- `tb_nldpe_dimm_top_latency.v`
- `tb_azurelily_dimm_top_functional.v`
- `tb_azurelily_dimm_top_latency.v`

**VTR orchestrators (fc_verification/):**
- `gen_nldpe_dimm_top_vtr.py`
- `gen_azurelily_dimm_top_vtr.py`

**Results (fc_verification/results/):**
- `nldpe_dimm_top_alignment_log.txt`
- `azurelily_dimm_top_alignment_log.txt`
- Updated `dimm_architecture_comparison.pdf`
- Updated `dimm_vtr_imc_results.json` and `azurelily_dimm_vtr_imc_results.json`

**Docs to edit:**
- `paper/methodology/attention_dimm_mapping.md`: add DIMM-only comparison clarification
- `nl_dpe/gen_bert_tiny_wrapper.py`: fix misleading comment
- `fc_verification/VERIFICATION.md`: append full DIMM verification section

## Commit milestones

1. Phase H.1 done: NL-DPE full DIMM top generates + syntax-clean
2. Phase H.2 done: Azure-Lily full DIMM top (with dsp_mac fix) generates + syntax-clean
3. Phase I.1 done: functional tests pass both architectures
4. Phase I.2 done: latency aligned within FSM tolerance
5. Phase J done: simulator ↔ RTL full DIMM alignment verified
6. Phase K done: VTR runs complete for both (3 seeds each)
7. Phase L done: DPE/DSP counts verified
8. Phase M done: final block-level comparison with VTR-backed Fmax
9. Phase N done: all docs updated, comments fixed

## Open questions (before executing Phase H)

None remaining — all four of user's directions answered. Softmax parallelism
decided (W=16 per doc). Plan frozen.

## Risk register

| Risk | Mitigation |
|---|---|
| W=16 RTL has lane-mixing bug (not all 16 lanes independent) | Debug via per-lane BRAM dump in functional test |
| Latency mismatch between sim and RTL on full DIMM | Add per-stage timestamp probes to pinpoint offending stage |
| VTR routing congestion at 64-DPE config | Accept lower Fmax; if too low to be useful, reduce W and rerun |
| `int_sop_4` behavioral stub incomplete for large N | Extend stub if needed; currently tested only at small K |
| Functional test with structured inputs doesn't catch subtle bugs | Add a random-input test once structured test passes |

## Progress tracking

Use TaskList with one task per phase. Update as we complete each milestone.
