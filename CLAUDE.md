# CLAUDE.md — NL-DPE FPGA Research Project

## Project in One Line
Crossbar-size DSE for NL-DPE (IMC FPGA hard block) to find optimal (rows, cols) for FC+activation and attention workloads, targeting a research paper comparing NL-DPE vs Azure-Lily.

## Session Start Protocol
1. Read SESSION_STATE.md → understand where we are
2. Read TASKS.md → pick the top unblocked task
3. Run `git status` to see uncommitted changes
4. If DSE is running or was interrupted: check `dse/results/` for CSVs and `dse/round1/` for partial VTR outputs

## Key Paths
| Path | Role |
|------|------|
| `gemv_dse.py` | Round 1 DSE orchestrator (main entry point) |
| `nl_dpe/gen_arch_xml.py` | VTR arch XML generator (auto / fixed_clb_replace / fixed_dsp_bram) |
| `nl_dpe/gen_gemv_wrappers.py` | Verilog RTL generator (GEMV and FC modes) |
| `nl_dpe/gen_attention_wrapper.py` | (planned) Parameterized attention head RTL generator |
| `nl_dpe/attention_head_1_channel.v` | Hand-written attention head RTL (N=128, d=128, 3 DPEs) |
| `nl_dpe/area_power.py` | DPE physical specs: `dpe_specs(rows, cols)` → tile W/H/area/power |
| `nl_dpe/run_vtr.py` | VTR flow runner (called by gemv_dse.py) |
| `azurelily/IMC/test.py` | IMC energy/latency simulator (supports `fc` and `attention` models) |
| `azurelily/models/attention.py` | Attention energy model (linear_Q/K/V + mac_qk + softmax + mac_sv) |
| `dse_experiment_plan.md` | Full methodology spec (authoritative) |
| `paper_outline.md` | Paper structure and narrative |
| `dse/results/` | CSVs, JSONs, plots, analysis (DSE outputs) |
| `dse/round1/<config>/<workload>/` | Per-run VTR outputs |

## Architecture Constants (do not hardcode elsewhere)
- CLB_tile_um2 = 2239 µm²  (from routing-aware formula: SB=688, CB=303)
- DPE configs: R ∈ {128, 256, 512}, C ∈ {64, 128, 256} → 9 configs
- FC workloads: fc_64_64, fc_128_128, fc_512_128, fc_256_512, fc_512_512, fc_2048_256
- Attention workload: N=128 seq_length, d=128 head_dim (3 DPEs for Q/K/V + CLB DIMM/softmax)
- Tiling: V = ceil(K/R), H = ceil(N/C); ACAM-eligible iff V == 1
- Area = grid_W × grid_H × 2239 / 1e6 [mm²]
- Throughput = 1e9 / latency_ns [inferences/s]
- Ranking: SPEC-style normalized geomean across workloads (per-workload best = 1.0)

## Coding Rules
- All paths passed to VTR subprocesses must be **absolute** (VTR changes CWD to its scripts dir)
- `MSB_SA_Ready` is an **OUTPUT** of the `dpe` hard block — each instance needs its own wire
- Adder tree internal wire names must be prefixed with `col{col}` to avoid multi-driver collisions when H > 1
- IMC simulator config is patched at runtime by `gemv_dse.py::patch_imc_config()` — do not hardcode Fmax
- VTR arch XML: only patch `<tile name="wc" ...>` for auto mode; do not touch `<auto_layout>`

## Workflow Pattern
**Plan → Implement → Sanity-check → Run → Verify results → Proceed**

- For any DSE run: always do a 1–3 point dry run first, verify CSV output, then full sweep
- Sanity check output: `dse/results/sanity_check_run.log`
- Full DSE: `python gemv_dse.py --round 1` (54 runs with 6 workloads, use `--jobs 12` to limit CPU)
- Resume interrupted: `python gemv_dse.py --round 1 --skip-existing`

## Context Economy Rules
- Do NOT re-read paper_outline.md or dse_experiment_plan.md unless working on paper narrative or methodology changes
- Do NOT re-run VTR on already-completed configs (check `dse/round1/`)
- SESSION_STATE.md is ground truth for project status — update it after every milestone
- TASKS.md is ground truth for what to do next — update after completing each task
