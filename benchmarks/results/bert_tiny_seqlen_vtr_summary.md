# BERT-Tiny Seq_Len Sweep v2 — VTR Resource Summary

## Proposed (NL-DPE 1024x128, DPE tile 3x7)

| seq_len | DPE | DSP | BRAM | CLB | Grid | Area (mm2) | Fmax (MHz) | Seeds |
|--------:|----:|----:|-----:|----:|-----:|-----------:|-----------:|------:|
| 128 | 40 | 10 | 150 | 659 | 68x68 | 10.4 | 132.5 | 3/3 |
| 256 | 56 | 10 | 158 | 685 | 82x82 | 15.1 | 131.6 | 3/3 |
| 512 | 88 | 10 | 174 | 695 | 93x93 | 19.4 | 132.7 | 3/3 |
| 1024 | 152 | 10 | 206 | 708 | 127x127 | 36.1 | 136.1 | 3/3 |
| 2048 | 280 | 10 | 270 | 728 | 172x172 | 66.2 | 133.9 | 3/3 |
| 4096 | 535 | 10 | 388 | 692 | 235x235 | 123.6 | 134.1 | 3/3 |

## AL-Like (NL-DPE 1024x256, DPE tile 5x8)

| seq_len | DPE | DSP | BRAM | CLB | Grid | Area (mm2) | Fmax (MHz) | Seeds |
|--------:|----:|----:|-----:|----:|-----:|-----------:|-----------:|------:|
| 128 | 36 | 10 | 150 | 655 | 110x110 | 27.1 | 130.9 | 3/3 |
| 256 | 36 | 10 | 158 | 676 | 110x110 | 27.1 | 135.1 | 3/3 |
| 512 | 52 | 10 | 174 | 694 | 146x146 | 47.7 | 128.9 | 3/3 |
| 1024 | 84 | 10 | 206 | 713 | 194x194 | 84.3 | 131.8 | 3/3 |
| 2048 | 148 | 10 | 270 | 732 | 222x222 | 110.3 | 136.6 | 3/3 |
| 4096 | 275 | 10 | 388 | 689 | 320x320 | 229.3 | 133.2 | 3/3 |

## Azure-Lily (512x128, DPE tile 6x5)

| seq_len | DPE | DSP | BRAM | CLB | Grid | Area (mm2) | Fmax (MHz) | Seeds |
|--------:|----:|----:|-----:|----:|-----:|-----------:|-----------:|------:|
| 128 | 18 | 30 | 54 | 380 | 48x48 | 5.2 | 131.4 | 3/3 |
| 256 | 18 | 38 | 66 | 403 | 54x54 | 6.5 | 137.3 | 3/3 |
| 512 | 18 | 54 | 90 | 455 | 64x64 | 9.2 | 128.8 | 3/3 |
| 1024 | 18 | 86 | 138 | 533 | 80x80 | 14.3 | 135.6 | 3/3 |
| 2048 | 18 | 150 | 234 | 694 | 102x102 | 23.3 | 136.1 | 3/3 |
| 4096 | 18 | 278 | 426 | 989 | 142x142 | 45.1 | 134.4 | 3/3 |

## Key Observations

- **Fmax is flat (~130-137 MHz)** across all architectures and seq_lens. Critical path
  is the LayerNorm DSP multiply -> adder carry chain, shared by all designs. This isolates
  the comparison to area and energy.
- **DPE** scales with S for NL-DPE (DIMM parallelism), fixed at 18 for Azure-Lily.
- **DSP** fixed at 10 for NL-DPE (LayerNorm only), scales with S for Azure-Lily (DIMM dsp_mac).
- **BRAM** scales with S for all architectures (K/V buffers + row buffers).
  Packed int8: depth = N_elements x 8 / 40.
- **Area**: NL-DPE 2-3x larger than Azure-Lily (DPE tile 21 cells vs DSP tile 4 cells).

## Critical Path

All designs share the same critical path:
```
Register -> DSP multiply (2.14ns) -> CLB adder carry chain (~3.5ns) -> Register
```
This is the LayerNorm variance/normalize accumulation path.
DIMM (the throughput bottleneck at O(S^2)) is NOT on the critical path.
