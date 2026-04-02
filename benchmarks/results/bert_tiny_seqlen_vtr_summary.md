# BERT-Tiny Seq_Len Sweep — VTR Resource Summary

## Proposed (NL-DPE 1024×128, DPE tile 3×7)

| seq_len | DPE | DSP | BRAM | CLB | Grid | Area (mm²) | Fmax (MHz) | Seeds |
|--------:|----:|----:|-----:|----:|-----:|-----------:|-----------:|------:|
| 512 | 88 ✓ | 4 | 473 | 551 | 122×122 | 33.3 | 117.8 | 3/3 |
| 1024 | 152 ✓ | 4 | 908 | 722 | 168×168 | 63.2 | 95.6 | 3/3 |
| 2048 | 280 ✓ | 4 | 1612 | 1007 | 228×228 | 116.4 | 79.7 | 3/3 |
| 4096 | 536 ✓ | 4 | 3148 | 1602 | 318×318 | 226.4 | 60.0 | 3/3 |
| 6144 | 792 ✓ | 4 | 14444 | 5886 | 676×676 | 1023.2 | 40.9 | 1/3 |
| 8192 | 1048 ✓ | 4 | 6220 | 2865 | 448×448 | 449.4 | 43.1 | 3/3 |

## AL-Like (NL-DPE 1024×256, DPE tile 5×8)

| seq_len | DPE | DSP | BRAM | CLB | Grid | Area (mm²) | Fmax (MHz) | Seeds |
|--------:|----:|----:|-----:|----:|-----:|-----------:|-----------:|------:|
| 512 | 52 ✓ | 4 | 470 | 544 | 146×146 | 47.7 | 119.3 | 3/3 |
| 1024 | 84 ✓ | 4 | 844 | 713 | 194×194 | 84.3 | 96.9 | 3/3 |
| 2048 | 148 ✓ | 4 | 2127 | 1006 | 260×260 | 151.4 | 79.8 | 3/3 |
| 4096 | 276 ✓ | 4 | 3148 | 1573 | 320×320 | 229.3 | 59.0 | 3/3 |
| 6144 | 404 ✓ | 4 | 14444 | 5841 | 676×676 | 1023.2 | 41.5 | 2/3 |
| 8192 | 532 ✓ | 4 | 6220 | 2828 | 448×448 | 449.4 | 43.9 | 2/3 |

## Azure-Lily (512×128, DPE tile 6×5)

| seq_len | DPE | DSP | BRAM | CLB | Grid | Area (mm²) | Fmax (MHz) | Seeds |
|--------:|----:|----:|-----:|----:|-----:|-----------:|-----------:|------:|
| 512 | 18 ✓ | 326 | 400 | 275 | 150×150 | 50.4 | 131.0 | 3/3 |
| 1024 | 18 ✓ | 326 | 784 | 317 | 160×160 | 57.3 | 125.8 | 3/3 |
| 2048 | 18 ✓ | 326 | 1556 | 413 | 226×226 | 114.4 | 126.0 | 3/3 |
| 4096 | 18 ✓ | 326 | 3100 | 573 | 312×312 | 217.9 | 85.0 | 3/3 |
| 6144 | 18 ✓ | 326 | 3116 | 588 | 314×314 | 220.8 | 89.3 | 3/3 |
| 8192 | 18 ✓ | 326 | 6188 | 987 | 444×444 | 441.4 | 57.9 | 3/3 |

## Notes

- **All DPE counts match expected values** across all 18 designs.
- **s6144 anomaly**: Both NL-DPE architectures produce a 676×676 grid with 14444 BRAMs — larger than s8192 (448×448, 6220 BRAMs). Multiple seeds failed. May need investigation.
- **Azure-Lily** resources are constant except BRAM (scales with seq_len due to Q/K/V intermediate buffers) and CLB (minor growth).
- **Fmax** decreases with seq_len for all architectures. Azure-Lily maintains higher Fmax (DSP critical path) vs NL-DPE (SRAM address MUX critical path).
- **4 failed seeds**: proposed s6144 (2/3 failed), al_like s6144 (1/3 failed), al_like s8192 (1/3 failed).
