# Paper Figures — Include List & Narrative

## Confirmed Figures

| # | Figure | File |
|---|--------|------|
| 1 | Block Comparison ✓ | `benchmarks/block_comparison.pdf` |
| 2 | Energy 3-Panel ✓ | `benchmarks/bert_seqlen_energy_3panel.pdf` |
| 3 | Efficiency (VarFmax) ✓ | `benchmarks/bert_seqlen_efficiency_varfmax.pdf` |
| 4 | Latency ✓ | `benchmarks/bert_seqlen_latency.pdf` |

## Per-Figure Narrative

### Figure 1: block_comparison.pdf — *Why NL-DPE wins at the block level*

**Panels**: (a) Block energy breakdown: ACAM vs ADC per VMM pass. (b) IMC area breakdown: area-scaled pie charts.

**Story**: The ACAM replaces the power-hungry ADC for output conversion. While the crossbar computation costs similar energy across all architectures, the ADC dominates Azure-Lily's block energy at over 99%. ACAM achieves the same function at **42× lower energy**. The area pie charts show this isn't a size trick — both conversion stages occupy similar silicon fractions. The block-level advantage is the foundation for all system-level gains.

**Key takeaway**: ACAM provides 42× block-level energy advantage over ADC at comparable area.

---

### Figure 2: bert_seqlen_energy_3panel.pdf — *How it plays out at system level*

**Panels**: (a) BERT-Tiny energy breakdown (DIMM vs Proj+FFN vs Other). (b) DIMM energy split: IMC block vs FPGA fabric. (c) Normalized energy consumption: P1/P2 vs Azure-Lily.

**Story**: DIMM dominates BERT-Tiny energy at all sequence lengths (>96%). The middle panel reveals the key difference: Proposed keeps roughly half its DIMM energy in the efficient IMC block, while Azure-Lily pushes nearly all DIMM energy through digital FPGA fabric. The right panel shows the net effect — Proposed-1 and Proposed-2 consistently use less energy than Azure-Lily, converging to **1.4× and 1.7×** savings at longer sequences. The DIMM-only energy ratio actually increases with sequence length (the ACAM advantage compounds), but the overall ratio converges because DIMM already dominates total energy.

**Key takeaway**: Block-level advantage translates to 1.4–1.7× system energy savings on BERT-Tiny.

---

### Figure 3: bert_seqlen_efficiency_varfmax.pdf — *The practical tradeoff*

**Panels**: (a) Normalized Inf/s/mm² — reported vs ideal frequency. (b) Normalized Inf/s/J — same treatment.

**Story**: The gap between dashed (ideal) and solid (reported) lines shows the implementation cost. At short sequences both architectures achieve their potential. At longer sequences, NL-DPE's larger tile (3×7 = 21 cells vs DSP's 1×4 = 4 cells) causes routing congestion that degrades frequency — narrowing the throughput/mm² advantage. Energy efficiency (Inf/J) is less affected since it doesn't depend on frequency. This points to future optimization opportunity in tile design and FPGA routing for heterogeneous blocks.

**Key takeaway**: Routing congestion at longer sequences is the main practical limitation. Energy advantage persists; area efficiency is the bottleneck.

---

### Figure 4: bert_seqlen_latency.pdf — *Where the time goes*

**Panels**: (a) Latency breakdown %: DIMM vs Non-DIMM. (b) DIMM and Non-DIMM speedup: reported vs ideal frequency for P1 and P2.

**Story**: DIMM latency grows to dominate all architectures at longer sequences, reaching nearly 100% at S=2048. The speedup decomposition shows Proposed achieves higher DIMM speedup than non-DIMM speedup, confirming the ACAM benefit is strongest in the attention-heavy operations. The shaded gap between ideal and reported frequency quantifies the implementation cost — at short sequences the gap is negligible, but at longer sequences routing degradation narrows the speedup for both DIMM and non-DIMM components.

**Key takeaway**: DIMM speedup confirms the analog nonlinearity advantage for attention workloads. Frequency degradation at scale is a shared challenge.

---

## One-Sentence Summary per Figure (for abstract/intro)

1. ACAM provides 42× block-level energy advantage over ADC at comparable area
2. This translates to 1.4–1.7× system energy savings on BERT-Tiny across sequence lengths
3. Routing congestion at longer sequences limits area efficiency but energy advantage persists
4. DIMM speedup confirms the analog nonlinearity advantage for attention workloads

## Data Sources

| CSV | Used by |
|-----|---------|
| `benchmarks/results/bert_tiny_seqlen_fixed_fmax.csv` | energy_3panel, efficiency (ideal lines) |
| `benchmarks/results/bert_tiny_seqlen_variable_fmax.csv` | efficiency (reported lines), latency |
| `benchmarks/results/bert_tiny_seqlen_vtr_summary.csv` | VTR resource counts |

## Plot Scripts

| Script | Generates |
|--------|-----------|
| `paper/scripts/plot_bert_block_comparison.py` | block_comparison.pdf |
| `paper/scripts/plot_bert_seqlen_sweep.py` | efficiency_varfmax, latency (+ analysis, efficiency, resilience) |
| `paper/scripts/plot_bert_seqlen_energy_3panel.py` | energy_3panel |

## Other Plots (not confirmed for paper)

| Figure | File | Notes |
|--------|------|-------|
| Energy Analysis | `benchmarks/bert_seqlen_analysis.pdf` | Overlaps with energy_3panel |
| Efficiency (Fixed) | `benchmarks/bert_seqlen_efficiency.pdf` | Superseded by varfmax version |
| Resilience | `benchmarks/bert_seqlen_resilience.pdf` | Retention rates similar across archs |
