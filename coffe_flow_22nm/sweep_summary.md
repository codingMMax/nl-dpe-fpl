# dpe local-input-crossbar — proportional-bandwidth sweep (COFFE `-hi`, TensorSlices-calibrated constants)

Proportional-bandwidth family: `B_in = 40·R/128` (+7 controls), `B_out = 40·C/64` (+5 status; inert), load/drain = 26/13 cycles. Runs: `work/dpe_R{128,256,512,1024}_C64` (`run_dpe.sh <R> bw`). Logic area from `nl_dpe/area_power.py`. Calibration anchor: `work/tensorslice` reproduces the published 2,771 µm² to +0.0117%. Configs are restricted to `C <= R` (DSE constraint): **11 valid rows**.

| R×C | n_in | n_out | cycles | mux | bits | bare µm² | SRAM µm² | **total µm²** | per-pin | logic µm² | **xbar/logic** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128×64 | 47 | 45 | 26/13 | 24:1 | 10 | 48.5 | 63.7 | **112.2** | 2.39 | 12198 | **0.92%** |
| 128×128 | 47 | 85 | 26/13 | 24:1 | 10 | 48.5 | 63.7 | **112.2** | 2.39 | 24042 | **0.47%** |
| 256×64 | 87 | 45 | 26/13 | 48:1 | 14 | 163.7 | 165.0 | **328.7** | 3.78 | 13994 | **2.35%** |
| 256×128 | 87 | 85 | 26/13 | 48:1 | 14 | 163.7 | 165.0 | **328.7** | 3.78 | 27280 | **1.21%** |
| 256×256 | 87 | 165 | 26/13 | 48:1 | 14 | 163.7 | 165.0 | **328.7** | 3.78 | 53850 | **0.61%** |
| 512×64 | 167 | 45 | 26/13 | 90:1 | 19 | 560.2 | 429.8 | **990.0** | 5.93 | 17586 | **5.63%** |
| 512×128 | 167 | 85 | 26/13 | 90:1 | 19 | 560.2 | 429.8 | **990.0** | 5.93 | 33755 | **2.93%** |
| 512×256 | 167 | 165 | 26/13 | 90:1 | 19 | 560.2 | 429.8 | **990.0** | 5.93 | 66093 | **1.50%** |
| 1024×64 | 327 | 45 | 26/13 | 168:1 | 26 | 1963.3 | 1151.6 | **3114.9** | 9.53 | 24770 | **12.58%** |
| 1024×128 | 327 | 85 | 26/13 | 168:1 | 26 | 1963.3 | 1151.6 | **3114.9** | 9.53 | 46706 | **6.67%** |
| 1024×256 | 327 | 165 | 26/13 | 168:1 | 26 | 1963.3 | 1151.6 | **3114.9** | 9.53 | 90578 | **3.44%** |

All 4 measured runs converged to identical area-counted widths (L1/L2 pass 1/1, restorer 1, inv_1 2/1); unit costs: pass 0.032713 µm², inv_1 pair 0.082976 µm²; bare = `n·[(impl+L2+1)·A_p + A_inv1]` MWTA exactly (asserted per run).

Note: block names are lowercase (`dpe_r<R>`) because HSPICE folds netlist/measure names to lowercase in `.lis` output while COFFE looks them up with the configured case.
