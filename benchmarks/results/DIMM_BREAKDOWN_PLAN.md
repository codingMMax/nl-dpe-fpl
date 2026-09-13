# DIMM Energy Breakdown Plan — BERT-Tiny Paper Figure

## Objective

Show that NL-DPE's ACAM-enhanced DIMM mapping enables efficient attention mechanisms
despite the DPE being limited to weight-persistent VMM. This is the paper's central claim.

## Proposed Figures

### Figure 1: BERT-Tiny End-to-End Energy Breakdown (%)

**3 stacked bar charts side-by-side**: NL-DPE Proposed | NL-DPE AL-Matched | Azure-Lily

**Energy categories (from IMC breakdown keys):**

| Category | NL-DPE keys | Azure-Lily keys | Color |
|----------|------------|----------------|-------|
| DPE Projection (VMM) | imc_vmm + imc_digital_post | imc_conversion | Green |
| DPE DIMM (exp/log) | imc_dimm_exp + imc_dimm_log | — (not applicable) | Dark Green |
| FPGA CLB (DIMM add/reduce) | clb_add + clb_reduction | — | Blue |
| FPGA DSP (DIMM MAC) | — | dsp_gemm | Orange |
| CLB Softmax | clb_norm_sum + mul | clb_exp + clb_norm_sum + clb_norm_inv + mul | Light Blue |
| CLB LayerNorm | clb_layernorm | clb_layernorm | Gray |
| CLB Activation | — (ACAM absorbs) | fpga_activation | Red |
| Memory (SRAM) | sram_read + sram_write | sram_read + sram_write | Purple |
| CLB Other (embed, residual) | clb_embed_add | clb_embed_add + clb_add | Light Gray |

**Key insight to highlight:**
- NL-DPE: DPE DIMM (exp/log) is a significant fraction but cheaper than Azure-Lily's DSP DIMM
- Azure-Lily: DSP DIMM (dsp_gemm) dominates — this is the cost NL-DPE avoids
- NL-DPE: zero CLB activation (ACAM absorbs GELU for free)
- Azure-Lily: pays CLB activation for GELU

### Figure 2: DIMM Proportion — Attention vs Non-Attention Energy

**Two-part horizontal bars per architecture:**

```
NL-DPE:     [=== Attention DIMM ===|=== Projections + FFN + LN + Embed ===]
Azure-Lily: [======= Attention DIMM =======|=== Projections + FFN + LN ===]
```

Shows what fraction of total energy goes to attention DIMM ops (QK^T, Softmax, Score×V).

**For NL-DPE**: DIMM = imc_dimm_exp + imc_dimm_log + clb_add (for DIMM) + clb_reduction (for DIMM)
**For Azure-Lily**: DIMM = dsp_gemm (QK^T + Score×V) + clb_exp + clb_norm (softmax)

### Figure 3: Per-Stage Latency Breakdown

**Gantt-style timeline for one BERT-Tiny inference:**

```
NL-DPE:     [Embed][LN][Q|K|V proj][DIMM QK^T][Softmax][DIMM SV][O proj][Res][LN][FFN1][FFN2][Res][LN] × 2 blocks
Azure-Lily: [Embed][LN][Q|K|V proj][DSP QK^T ][Softmax][DSP SV ][O proj][Res][LN][FFN1][FFN2][Res][LN] × 2 blocks
```

Shows latency per stage. NL-DPE DIMM stages should be faster than Azure-Lily DSP stages
(parallel DPE passes vs sequential DSP MACs).

## Data Source

Run IMC with `--debug` flag to get per-layer energy/latency breakdown:
```bash
python benchmarks/run_imc_with_vtr_freq.py --model bert_tiny --imc_config nl_dpe --rows 1024 --cols 128 --fmax 139.4
python benchmarks/run_imc_with_vtr_freq.py --model bert_tiny --imc_config azure_lily --rows 512 --cols 128 --fmax 124.9
```

## Key Numbers to Extract

For each architecture:
1. Total energy per inference
2. Attention DIMM energy (QK^T + Softmax + Score×V)
3. Projection energy (Q/K/V/O/FFN1/FFN2)
4. LayerNorm + Residual + Embedding energy
5. DIMM % of total
6. Projection % of total
7. DPE energy vs FPGA fabric energy vs Memory energy

## Selling Point Narrative

"Despite the ReRAM crossbar being limited to weight-persistent VMM, the NL-DPE's
programmable ACAM enables efficient attention mechanism by:
1. Converting projections to log domain (ACAM=log) → enables log-domain DIMM
2. Replacing N² DSP MACs with N² CLB additions + DPE exp passes → 2.4× faster
3. Absorbing GELU activation for free (ACAM=GELU) → zero CLB activation overhead
4. Result: 1.2× lower energy and 2.4× faster than Azure-Lily for BERT-Tiny"
