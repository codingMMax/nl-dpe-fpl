# FPGA Grid Sizes — VTR Auto-Layout Results

Last updated: 2026-03-24

## Per-Design Grid Sizes

| Design | Config | Grid | Cells | DPEs | CLBs | BRAMs |
|--------|--------|------|-------|------|------|-------|
| resnet9_proposed | NL-DPE 1024×128 | 58×58 | 3,364 | 19 | 61 | 28 |
| resnet9_al_matched | NL-DPE 1024×256 | 46×46 | 2,116 | 12 | 63 | 28 |
| resnet_1_azurelily | Azure-Lily 512×128 | 61×61 | 3,721 | 35 | 185 | 16 |
| vgg11_proposed | NL-DPE 1024×128 | 123×123 | 15,129 | 85 | 99 | 30 |
| vgg11_al_matched | NL-DPE 1024×256 | 83×83 | 6,889 | 44 | 101 | 30 |
| vgg11_1_azurelily | Azure-Lily 512×128 | 120×120 | 14,400 | 148 | 234 | 18 |

## Largest Benchmark

**VGG-11 on NL-DPE 1024×128** requires **123×123** grid (15,129 cells).

## Fixed FPGA Size for Fair Comparison

For end-to-end comparison (ResNet-9 + VGG-11 + BERT-Tiny), the fixed FPGA must fit
the largest benchmark with ~70-80% resource utilization.

- **Minimum**: 123×123 (fits VGG-11 proposed exactly)
- **With 20% headroom**: 147×147
- **BERT-Tiny TBD**: will need grid size from VTR auto-layout once RTL is generated

## FINAL: Fixed FPGA Size = 150×150

Largest auto-layout design: Azure-Lily VGG-11 at 142×142. With headroom → **150×150**.

All 9 benchmarks (3 models × 3 architectures) run on same 150×150 fixed grid.
- NL-DPE Proposed (1024×128): d=80% DSP replacement
- NL-DPE AL-Matched (1024×256): d=30% DSP replacement
- Azure-Lily (512×128): d=89% DSP + c=10% CLB replacement

Results saved to: `benchmarks/results/imc_benchmark_results.csv`

## Usage

When implementing the fixed-FPGA comparison:
1. Use this grid size for `gen_arch_xml.py --mode fixed_*`
2. All three architectures (NL-DPE proposed, NL-DPE AL-matched, Azure-Lily) use the SAME grid
3. Fmax from VTR on the fixed grid feeds into IMC simulator via `run_imc_with_vtr_freq.py`
