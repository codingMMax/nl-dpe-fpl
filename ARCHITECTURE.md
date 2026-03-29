# ARCHITECTURE.md — NL-DPE FPGA Project

## System Overview
```
FC workload (K, N)
       │
       ▼
gen_gemv_wrappers.py ──────────► fc_{K}_{N}_{R}x{C}.v   (Verilog RTL)
gen_arch_xml.py ───────────────► nl_dpe_{R}x{C}_auto.xml (VTR arch XML)
       │
       ▼
    VTR flow (run_vtr.py)
       │
       ├─► Fmax (MHz)
       └─► Grid size (W × H CLB tiles)
               │
               ▼
        patch_imc_config()          ← area_power.py::dpe_specs(R, C)
               │
               ▼
    IMC simulator (test.py --model fc)
               │
               ├─► energy (pJ)
               └─► latency (ns)
                       │
                       ▼
               gemv_dse.py (metrics)
               ├─► throughput = 1e9/lat [inf/s]
               ├─► area = W×H×2239/1e6 [mm²]
               ├─► tput/mm² , tput/J
               └─► normalized geomean ranking
```

## Key Modules

### `gemv_dse.py` — DSE Orchestrator
- Entry point for all DSE runs
- `--round 1`: 9 configs × 5 workloads, auto_layout, normalized geomean ranking
- `--round 2`: top-3 configs × fixed layout × CLB/DSP/BRAM sweeps (not yet implemented)
- `--skip-existing`: resume interrupted runs
- `--dry-run`: preview jobs without executing
- Outputs: `dse/results/round{N}_results.csv`, `top3_configs.json`

### `nl_dpe/gen_arch_xml.py` — VTR XML Generator
- **auto**: patches `<tile name="wc">` W/H/area from `dpe_specs(R,C)`; keeps `<auto_layout>`
- **fixed_clb_replace**: `<fixed_layout>` with DPE columns replacing CLBs at configurable ratio
- **fixed_dsp_bram**: replaces DPE tiles with equivalent DSPs+BRAMs for comparison
- Template: `nl_dpe/nl_dpe_22nm_auto.xml`

### `nl_dpe/gen_gemv_wrappers.py` — RTL Generator
- **GEMV mode**: parameter substitution into `gemv_1_channel.v`
- **FC mode** (`--fc` flag): generates full `fc_top` + `fc_layer` with V×H DPE array
  - V = ceil(K/R), H = ceil(N/C)
  - V>1: column adder tree (`_gen_adder_tree`) + `activation_lut` per column
  - V=1,H=1: reuses single-DPE path (no extra logic)
  - All paths absolute; each DPE gets unique `MSB_SA_Ready_c{col}_r{row}` wire

### `nl_dpe/area_power.py` — Physical Model
- `dpe_specs(rows, cols)` → `{tile_w, tile_h, area_tag, power_mw, ...}`
- Area formula: routing-aware, SB+CB overhead included
- Power: ACAM (43.89 mW), digital reduction, SRAM
- Used to patch IMC simulator config and VTR arch XML

### `azurelily/IMC/test.py` + `azurelily/models/fc.py` — IMC Simulator
- Runs analytical energy/latency model for a given workload + config
- `fc_model`: V=1 → ACAM absorbs activation (fpga.activation() not called); V>1 → CLB activation energy
- Config patched at runtime by `gemv_dse.py::patch_imc_config()`
- Call: `python test.py --model fc --rows R --cols C --K K --N N`

## DSE Directory Structure
```
dse/
├── configs/arch/          ← nl_dpe_{R}x{C}_auto.xml (9 files)
├── rtl/                   ← fc_{K}_{N}_{R}x{C}.v (45 files)
├── round1/<config>/<wl>/  ← VTR outputs per run (vpr_stdout.log, ...)
└── results/
    ├── round1_results.csv ← main data table
    ├── top3_configs.json  ← top-3 configs for Round 2
    └── sanity_check_run.log
```

## NL-DPE Crossbar Configs (Round 1)
| Config | R | C | V for K=512 | ACAM-eligible (K≤R) |
|--------|---|---|-------------|----------------------|
| 128×64 | 128 | 64 | 4 | only K≤128 |
| 128×128 | 128 | 128 | 4 | only K≤128 |
| 128×256 | 128 | 256 | 4 | only K≤128 |
| 256×64 | 256 | 64 | 2 | K≤256 |
| 256×128 | 256 | 128 | 2 | K≤256 |
| **256×256** | 256 | 256 | 2 | K≤256 |
| 512×64 | 512 | 64 | 1 | all K≤512 ✓ |
| 512×128 | 512 | 128 | 1 | all K≤512 ✓ |
| **512×256** | 512 | 256 | 1 | all K≤512 ✓ |

## FC Workloads (Round 1)
| Workload | K | N | Notes |
|----------|---|---|-------|
| fc_64_64 | 64 | 64 | tiny; all configs V=1 |
| fc_128_128 | 128 | 128 | small |
| fc_256_256 | 256 | 256 | medium |
| fc_512_128 | 512 | 128 | deep, narrow |
| fc_512_512 | 512 | 512 | deep, wide; stresses all configs |

## Azure-Lily Baseline
- Submodule at `azurelily/`
- IMC simulator at `azurelily/IMC/`
- Calibrated ground truth: `e_conv = 2.33` (from calibration study)
- NL-DPE energy ratio vs Azure-Lily: 0.548× on LeNet (known)
