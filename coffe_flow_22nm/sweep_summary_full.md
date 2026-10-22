# dpe local-input-crossbar — row-parallel sweep (COFFE `-hi`, TensorSlices-calibrated constants)

Row-parallel interface (one data pin per crossbar row): `num_gen_inputs = R + 7` (7 fixed controls), `num_gen_outputs = C + 5` (status; inert — COFFE never reads it, so one measured run per R covers every C). Load/drain = `ceil(8R/R)`/`ceil(8C/C)` = **8/8 cycles** (int8 bit-serial). Runs: `work/dpe_full_R{64,128,256,512}_C64` (`run_dpe.sh <R>`). Logic area from `nl_dpe/area_power.py`. Calibration anchor: `work/tensorslice` reproduces the published 2,771 µm² to +0.0117%. Configs are restricted to `C <= R` (DSE constraint): **9 valid rows**.

| R×C | n_in | n_out | cycles | mux | bits | bare µm² | SRAM µm² | **total µm²** | per-pin | logic µm² | **xbar/logic** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64×64 | 71 | 69 | 8/8 | 36:1 | 12 | 105.8 | 115.4 | **221.2** | 3.12 | 11300 | **1.96%** |
| 128×64 | 135 | 69 | 8/8 | 72:1 | 17 | 368.9 | 310.9 | **679.8** | 5.04 | 12198 | **5.57%** |
| 128×128 | 135 | 133 | 8/8 | 72:1 | 17 | 368.9 | 310.9 | **679.8** | 5.04 | 24042 | **2.83%** |
| 256×64 | 263 | 69 | 8/8 | 132:1 | 23 | 1260.7 | 819.4 | **2080.1** | 7.91 | 13994 | **14.86%** |
| 256×128 | 263 | 133 | 8/8 | 132:1 | 23 | 1260.7 | 819.4 | **2080.1** | 7.91 | 27280 | **7.62%** |
| 256×256 | 263 | 261 | 8/8 | 132:1 | 23 | 1260.7 | 819.4 | **2080.1** | 7.91 | 53850 | **3.86%** |
| 512×64 | 519 | 69 | 8/8 | 272:1 | 33 | 4949.7 | 2320.0 | **7269.6** | 14.01 | 17586 | **41.34%** |
| 512×128 | 519 | 133 | 8/8 | 272:1 | 33 | 4949.7 | 2320.0 | **7269.6** | 14.01 | 33755 | **21.54%** |
| 512×256 | 519 | 261 | 8/8 | 272:1 | 33 | 4949.7 | 2320.0 | **7269.6** | 14.01 | 66093 | **11.00%** |

All 4 measured runs converged to identical area-counted widths (L1/L2 pass 1/1, restorer 1, inv_1 2/1); unit costs: pass 0.032713 µm², inv_1 pair 0.082976 µm²; bare = `n·[(impl+L2+1)·A_p + A_inv1]` MWTA exactly (asserted per run).

Note: block names are lowercase (`dpe_full_r<R>`) because HSPICE folds netlist/measure names to lowercase in `.lis` output while COFFE looks them up with the configured case.

Comparison context: the real fixed 40-bit-port block (`work/dpe_R128_C64`, 47 pins) is 112.15 µm² = 0.92% of 128×64 logic; the proportional-bandwidth family is in `sweep_summary.csv`/`sweep_summary.md` (0.23–12.58%).
