# Phase 2.1 — GEMM DSE smoke test

Mirrors `gemv_dse.py::run_round1` (Round-1 FC DSE) but for four GEMM
workloads with a real batch / GEMM row dimension `M > 1`, swept across
the 12-point `(R, C)` crossbar grid.

## Workloads (smoke-test subset)

| Name           | (M, K, N)         | Source                                   |
|----------------|-------------------|------------------------------------------|
| `bert_qkvo`    | (128, 128, 128)   | BERT-Tiny Q / K / V / O projection       |
| `bert_ffn2`    | (128, 512, 128)   | BERT-Tiny FFN2 down-projection           |
| `swin_mlp`     | (49,  384, 1536)  | Swin-Tiny stage-3 MLP up-projection      |
| `resnet9_conv` | (256, 2304, 256)  | ResNet-9 mid conv (im2col)               |

## (R, C) grid — 12 configs

Rows R ∈ {128, 256, 512, 1024}, Cols C ∈ {64, 128, 256} → 12 crossbar
geometries, 4 × 12 = 48 VTR runs (single seed = 42, mirroring Round-1).

## CSV columns

The 48-row output CSV (`results/gemm_dse_smoke.csv`) carries the same
columns as Round-1's `round1_results.csv`, plus:

| Column         | Meaning                                                      |
|----------------|--------------------------------------------------------------|
| `M`            | GEMM row / batch dimension                                   |
| `L_A_cyc`      | Layout-A per-pass load cycles (`ceil(R·8/W) + 8 + 3`)       |
| `O_cyc`        | Drain cycles (`ceil(C·8/W)`)                                 |
| `steady_B_cyc` | Steady interval `max(L_A, O)`                                |
| `T_pred_cyc`   | Analytical Regime-B prediction `L_A + steady·(M_eff−1) + O`  |
| `T_pred_ns`    | `T_pred_cyc × 1000 / fmax_mhz`                               |
| `regime`       | `feed-bound` / `drain-bound` / `balanced`                    |

Sim-reported `latency_ns` is compared against `T_pred_ns` at CSV-write
time; a `|sim - pred|/pred > 2.0` mismatch is logged as a warning (no
auto-fail) per plan.

## Re-running

```bash
# Full 48-config sweep (jobs capped at 12 per plan)
python3 dse/gemm_phase2_1/run_gemm_dse.py --smoke

# Single config sanity check (~6 s for V>1, <1 s for V=1)
python3 dse/gemm_phase2_1/run_gemm_dse.py --smoke \
    --workloads bert_qkvo --configs 256x128 --jobs 1

# Resume after interruption — reuses existing imc_result.json
python3 dse/gemm_phase2_1/run_gemm_dse.py --smoke --skip-existing

# Dry-run (prints the 48-job table, no VTR)
python3 dse/gemm_phase2_1/run_gemm_dse.py --smoke --dry-run

# Plots (after CSV exists)
python3 dse/gemm_phase2_1/plot_gemm_pareto.py
python3 dse/gemm_phase2_1/plot_gemm_heatmap.py
```

## Implementation notes

- `run_gemm_dse.py` reuses Round-1's helpers directly
  (`gemv_dse.patch_imc_config`, the IMC output regexes, and the VTR
  flow wrapper `run_single`).  Only the job-list builder, the CSV
  schema, and the IMC runner needed Phase-2.1-specific changes.
- The IMC sim was extended with a new `--batch M` flag (submodule
  commit `690d7fe`).  `fc_model` sets the layer's `num_inputs` to `M`
  so `_run_linear` picks it up and the Regime-B gemm_log path sees the
  correct batch factor.
- **VTR-compatibility strip**: `gen_fc_wrapper`'s V>1 branch emits
  `dpe #(.KERNEL_WIDTH(X), .NUM_COLS(Y)) dpe_c… (…)` for TB alignment
  (commit `2678040`).  Parmys rejects those parameters because the
  arch XML blackbox `dpe` model declares no such ports.  The driver
  therefore writes a sibling `*_acam_dw40_vtr.v` with the decorator
  stripped and feeds *that* file to VTR.  The TB-facing original is
  untouched.

## Expected sweet spots (48-row sweep)

| Workload      | Sweet-spot (R, C) | Sim latency (ns) | Regime |
|---------------|-------------------|------------------|--------|
| `bert_qkvo`   | 128x64 / 256x64 / 512x64 (tied) | 8094 | F |
| `bert_ffn2`   | 128x64            | 10765            | F |
| `swin_mlp`    | 128x128           | 6804             | F |
| `resnet9_conv`| 128x64            | 29068            | F |

Note that ResNet-9 uses M=256, Swin-MLP uses M=49, BERT uses M=128.
All four sweet spots are **feed-bound** regime, reflecting the
W_DPE=40 bus that makes `L_A = ceil(R·8/40) + 11` grow linearly in R
while `O = ceil(C·8/40)` stays tiny at C=64.
