# Attention: crossbar configs side by side

Per head, the geometric mean over the six heads (`dse/config_table.py`; definitions in `docs/experiment.md` §9.3).

## 1. The chain, relative to 128×64

Every column is relative to 128×64 (the best ADP of the table unless `--ref` names another), and the columns multiply
exactly: latency = cycles ÷ Fmax, ADP = area × latency, EDAP = energy × ADP. Lower is better, except Fmax.

| config | area | cycles | Fmax | latency = cycles ÷ Fmax | ADP = area × latency | energy | EDAP = energy × ADP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×64 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 | **1.00** |
| 256×64 | 1.02 | 1.13 | 1.00 | 1.13 | 1.15 | 1.04 | 1.20 |
| 512×64 | 1.05 | 1.29 | 1.00 | 1.30 | 1.37 | 1.14 | 1.56 |
| 1024×64 | 1.11 | 1.42 | 0.99 | 1.43 | 1.58 | 1.34 | 2.13 |
| 128×128 | 1.32 | 1.10 | 0.95 | 1.15 | 1.52 | 1.03 | 1.57 |
| 256×128 | 1.34 | 1.20 | 0.97 | 1.24 | 1.65 | 1.08 | 1.78 |
| 512×128 | 1.39 | 1.37 | 0.92 | 1.49 | 2.07 | 1.18 | 2.44 |
| 1024×128 | 1.50 | 1.51 | 0.96 | 1.58 | 2.36 | 1.39 | 3.27 |
| 128×256 | 1.51 | 1.50 | 0.94 | 1.60 | 2.40 | 1.60 | 3.84 |
| 256×256 | 1.94 | 1.52 | 0.88 | 1.73 | 3.35 | 1.14 | 3.80 |
| 512×256 | 2.05 | 1.72 | 0.87 | 1.98 | 4.05 | 1.25 | 5.05 |
| 1024×256 | 2.22 | 1.88 | 0.84 | 2.24 | 4.97 | 1.47 | 7.31 |

## 2. What is inside

Area: VPR counts (crossbar tiles, CLBs, BRAM blocks at 2 cells). Energy: the priced ledger. Cycles: the head's critical
path by kind (the chain of waits that set the head's done; its segments add up to T exactly): DPE load = a pass's
input burst, DPE compute = the bit-slices, the shift-accumulate, the ACAM and the hand-overs, DPE drain = the first
output word and the drain words a consumer takes, stream = a port's words or a FIFO read one word per cycle (the 40-bit
paths between modules), logic = every other named latency. Fmax: VTR, the mean of seeds {11, 13, 17}. The parts are
geometric means too, so they come close to their total but do not add up to it exactly.

| config | crossbar mm² | CLB mm² | BRAM mm² | area mm² | energy nJ | DPE load cyc | DPE compute cyc | DPE drain cyc | stream cyc | logic cyc | cycles T | Fmax MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128×64 | 2.3 | 21.6 | 1.2 | 25.1 | 679 | 4,293 | 1,872 | 183 | 1,582 | 350 | 8,410 | 130 |
| 256×64 | 2.8 | 21.6 | 1.1 | 25.5 | 710 | 5,322 | 1,872 | 183 | 1,582 | 349 | 9,492 | 130 |
| 512×64 | 3.7 | 21.6 | 1.1 | 26.4 | 778 | 6,460 | 1,872 | 183 | 1,582 | 349 | 10,871 | 130 |
| 1024×64 | 5.0 | 21.6 | 1.1 | 27.8 | 913 | 7,295 | 1,872 | 183 | 1,582 | 349 | 11,963 | 129 |
| 128×128 | 5.1 | 26.9 | 1.1 | 33.2 | 702 | 3,739 | 1,204 | 360 | 2,229 | 341 | 9,234 | 124 |
| 256×128 | 5.5 | 26.9 | 1.1 | 33.6 | 731 | 4,585 | 1,216 | 337 | 2,028 | 310 | 10,062 | 126 |
| 512×128 | 6.8 | 26.9 | 1.1 | 34.9 | 801 | 5,551 | 1,216 | 337 | 2,028 | 310 | 11,548 | 120 |
| 1024×128 | 9.5 | 26.9 | 1.1 | 37.6 | 942 | 6,263 | 1,216 | 337 | 2,028 | 309 | 12,727 | 125 |
| 128×256 | 9.7 | 27.0 | 1.1 | 37.8 | 1,085 | 456 | 202 | 9,630 | 1,606 | 341 | 12,616 | 122 |
| 256×256 | 11.0 | 36.4 | 1.2 | 48.7 | 772 | 3,176 | 540 | 2,511 | 2,376 | 270 | 12,747 | 114 |
| 512×256 | 13.7 | 36.5 | 1.2 | 51.4 | 847 | 3,804 | 540 | 2,511 | 2,376 | 270 | 14,464 | 113 |
| 1024×256 | 18.1 | 36.4 | 1.2 | 55.9 | 998 | 4,273 | 540 | 2,511 | 2,376 | 269 | 15,846 | 110 |

## 3. The critical path as shares of T

By kind, then by module (the module whose wait the cycle is: the projections gemm, the QK and PV DIMMs, the online
softmax sm, the accumulator acc, the controller); the mean over the six heads of each share. A cycle that two inputs
set together is charged to the first the model names (a tie).

| config | T | DPE load | DPE compute | DPE drain | stream | logic | gemm | qk | sm | pv | acc | ctrl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128×64 | 8,410 | 51 % | 23 % | 2 % | 20 % | 4 % | 49 % | 3 % | 33 % | 11 % | 4 % | 0 % |
| 256×64 | 9,492 | 56 % | 20 % | 2 % | 17 % | 4 % | 55 % | 3 % | 29 % | 10 % | 4 % | 0 % |
| 512×64 | 10,871 | 60 % | 19 % | 2 % | 15 % | 4 % | 59 % | 3 % | 26 % | 8 % | 3 % | 0 % |
| 1024×64 | 11,963 | 62 % | 18 % | 2 % | 15 % | 3 % | 61 % | 3 % | 25 % | 8 % | 3 % | 0 % |
| 128×128 | 9,234 | 44 % | 14 % | 12 % | 26 % | 4 % | 39 % | 2 % | 37 % | 15 % | 7 % | 0 % |
| 256×128 | 10,062 | 50 % | 13 % | 12 % | 22 % | 3 % | 49 % | 2 % | 31 % | 12 % | 7 % | 0 % |
| 512×128 | 11,548 | 54 % | 11 % | 12 % | 19 % | 3 % | 54 % | 2 % | 28 % | 11 % | 6 % | 0 % |
| 1024×128 | 12,727 | 56 % | 11 % | 12 % | 18 % | 3 % | 56 % | 2 % | 26 % | 10 % | 6 % | 0 % |
| 128×256 | 12,616 | 4 % | 2 % | 77 % | 15 % | 3 % | 46 % | 6 % | 26 % | 16 % | 6 % | 0 % |
| 256×256 | 12,747 | 32 % | 5 % | 41 % | 20 % | 2 % | 56 % | 1 % | 26 % | 11 % | 6 % | 0 % |
| 512×256 | 14,464 | 36 % | 4 % | 40 % | 18 % | 2 % | 60 % | 1 % | 23 % | 9 % | 6 % | 0 % |
| 1024×256 | 15,846 | 38 % | 4 % | 40 % | 16 % | 2 % | 63 % | 1 % | 21 % | 9 % | 6 % | 0 % |

**Per head at 128×64** (shares of T: by kind, then by module)

| head | T | DPE load | DPE compute | DPE drain | stream | logic | gemm | qk | sm | pv | acc | ctrl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | 24,395 | 47 % | 21 % | 2 % | 26 % | 5 % | 36 % | 0 % | 43 % | 20 % | 1 % | 0 % |
| DeiT-T | 24,393 | 47 % | 21 % | 2 % | 26 % | 5 % | 36 % | 0 % | 43 % | 20 % | 1 % | 0 % |
| BERT-base | 7,755 | 55 % | 21 % | 2 % | 18 % | 3 % | 56 % | 1 % | 30 % | 9 % | 3 % | 0 % |
| BERT-Tiny | 7,752 | 56 % | 21 % | 2 % | 18 % | 3 % | 56 % | 1 % | 30 % | 9 % | 3 % | 0 % |
| JetFormer-150P | 1,279 | 47 % | 31 % | 5 % | 11 % | 6 % | 53 % | 14 % | 22 % | 0 % | 11 % | 0 % |
| JetFormer-JetClass | 7,732 | 56 % | 21 % | 2 % | 18 % | 3 % | 56 % | 1 % | 30 % | 9 % | 3 % | 0 % |

**Per head at 128×128** (shares of T: by kind, then by module)

| head | T | DPE load | DPE compute | DPE drain | stream | logic | gemm | qk | sm | pv | acc | ctrl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | 22,204 | 43 % | 13 % | 3 % | 37 % | 4 % | 20 % | 0 % | 54 % | 23 % | 2 % | 0 % |
| DeiT-T | 22,202 | 43 % | 13 % | 3 % | 37 % | 4 % | 20 % | 0 % | 54 % | 23 % | 2 % | 0 % |
| BERT-base | 9,408 | 54 % | 17 % | 2 % | 24 % | 3 % | 46 % | 1 % | 35 % | 12 % | 6 % | 0 % |
| BERT-Tiny | 9,405 | 54 % | 17 % | 2 % | 24 % | 3 % | 46 % | 1 % | 35 % | 12 % | 6 % | 0 % |
| JetFormer-150P | 1,532 | 15 % | 6 % | 62 % | 10 % | 7 % | 56 % | 6 % | 10 % | 7 % | 21 % | 0 % |
| JetFormer-JetClass | 9,277 | 54 % | 17 % | 2 % | 25 % | 3 % | 47 % | 1 % | 35 % | 12 % | 5 % | 0 % |

## 4. FC: the critical path by kind

One pass per row (the load of min(R, K) rows, the slices, the first drain word, the tree and the serializer, the
output words on the port); the mean over the four shapes of each share.

| config | T (geomean) | DPE load | DPE compute | DPE drain | stream | logic |
|---|---:|---:|---:|---:|---:|---:|
| 128×64 | 52 | 50 % | 17 % | 2 % | 25 % | 6 % |
| 256×64 | 77 | 67 % | 12 % | 1 % | 17 % | 3 % |
| 512×64 | 112 | 78 % | 8 % | 1 % | 12 % | 1 % |
| 1024×64 | 130 | 80 % | 7 % | 1 % | 11 % | 1 % |
| 128×128 | 65 | 40 % | 14 % | 2 % | 40 % | 5 % |
| 256×128 | 90 | 58 % | 10 % | 1 % | 29 % | 2 % |
| 512×128 | 125 | 70 % | 7 % | 1 % | 21 % | 1 % |
| 1024×128 | 143 | 72 % | 7 % | 1 % | 19 % | 1 % |
| 128×256 | 84 | 31 % | 11 % | 1 % | 53 % | 4 % |
| 256×256 | 109 | 48 % | 8 % | 1 % | 41 % | 2 % |
| 512×256 | 146 | 61 % | 6 % | 1 % | 32 % | 1 % |
| 1024×256 | 164 | 64 % | 6 % | 1 % | 29 % | 1 % |
