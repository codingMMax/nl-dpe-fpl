# CLAUDE.md — NL-DPE FPGA Research Project

## Project in One Line
NL-DPE FPGA hard block research: crossbar-size DSE (complete) + BERT-Tiny end-to-end workload (in progress) for a paper comparing NL-DPE vs Azure-Lily.

## Session Start Protocol
1. Read SESSION_STATE.md → understand where we are
2. Read this file's "Active TODO Tracks" section below → current in-flight work
3. Run `git status` to see uncommitted changes
4. (If touching DSE) check `dse/results/` for CSVs and `dse/round1/` for partial VTR outputs

Note: `TASKS.md` is the **prior-submission** sprint log (historical only as of
2026-04-18). Do not treat its open `[ ]` items as active without cross-checking
the live tracks below.

## Key Paths
| Path | Role |
|------|------|
| `gemv_dse.py` | DSE orchestrator: Round 1 (`--round 1`), Round 2 prototype (`--round2-proto`), Round 2 full (`--round2-full`) |
| `nl_dpe/gen_arch_xml.py` | VTR arch XML generator (auto / fixed_clb_replace / fixed_dsp_bram / fixed_dsp_clb_replace) |
| `nl_dpe/gen_gemv_wrappers.py` | Verilog RTL generator (GEMV and FC modes) |
| `nl_dpe/gen_gemm_wrapper.py` | P-replica GEMM wrapper generator (Round 2, imports from gen_gemv_wrappers) |
| `nl_dpe/gen_attention_wrapper.py` | Parameterized attention head RTL generator |
| `nl_dpe/attention_head_1_channel.v` | Hand-written attention head RTL (N=128, d=128, 3 DPEs) |
| `nl_dpe/area_power.py` | DPE physical specs: `dpe_specs(rows, cols)` → tile W/H/area/power |
| `nl_dpe/run_vtr.py` | VTR flow runner (called by gemv_dse.py) |
| `azurelily/IMC/test.py` | IMC energy/latency simulator (supports `fc`, `attention`, `bert_tiny` models) |
| `azurelily/models/attention.py` | Attention energy model (linear_Q/K/V + mac_qk + softmax + mac_sv) |
| `azurelily/models/bert_tiny.py` | BERT-Tiny model (2L/2H/128d/512ff, embedding + LayerNorm + multi-head attention + FFN) |
| `azurelily/nn/layernorm_layer.py` | LayerNorm layer class for scheduler dispatch |
| `azurelily/nn/embedding_layer.py` | Embedding layer class for scheduler dispatch |
| `dse_experiment_plan.md` | Full methodology spec (authoritative) |
| `paper_outline.md` | Paper structure and narrative |
| `paper/methodology/attention_dimm_mapping.md` | Attention → DPE/DSP mapping, K-identity, W=16 lane spec |
| `paper/methodology/dpe_pipeline_model.md` | Per-pass DPE dataflow model (Layout A vs B, transpose, multi-pass pipelining) — §8.1 has open TODOs |
| `fc_verification/VERIFICATION.md` | RTL↔sim verification story, Phase H-N results |
| `dse/results/` | CSVs, JSONs, plots, analysis (DSE outputs) |
| `dse/round1/<config>/<workload>/` | Per-run VTR outputs (Round 1) |
| `dse/round2_proto/` | Per-run VTR outputs (Round 2 prototype, fc_2048_256 only) |
| `dse/round2_full/` | Per-run VTR outputs (Round 2 full sweep, all workloads) |

## Architecture Constants (do not hardcode elsewhere)
- CLB_tile_um2 = 2239 µm²  (from routing-aware formula: SB=688, CB=303)
- DPE configs: R ∈ {128, 256, 512, 1024}, C ∈ {64, 128, 256} → 12 configs (Round 1); top-5 for Round 2
- Round 2 configs: 512×128, 1024×128, 1024×64, 1024×256, 512×256
- Round 2 FC workloads: fc_512_128, fc_512_512, fc_2048_256
- Round 2 grid: 120×120 fixed (fits all 5 configs including 1024×256 tile 7×9)
- Attention workload: N=128 seq_length, d=128 head_dim (3 DPEs for Q/K/V + CLB DIMM/softmax)
- Tiling: V = ceil(K/R), H = ceil(N/C); ACAM-eligible iff V == 1
- Area = grid_W × grid_H × 2239 / 1e6 [mm²]
- Throughput = 1e9 / latency_ns [inferences/s]
- Ranking: SPEC-style normalized geomean across workloads (per-workload best = 1.0)
- BRAM: height=2, startx=2, repeatx=16. 120×120 grid has 472 BRAMs. Per replica: 4 BRAMs (K≤1024), 6 BRAMs (K>1024)
- BERT-Tiny: 2 layers, 2 heads, d_model=128, d_head=64, d_ff=512, vocab=30522, max_pos=512, default seq_len=1024
- BERT-Tiny DPE-independent modules: LayerNorm, Residual Add, Embedding Add (all pure CLB)
- BERT-Tiny DPE-dependent modules: Q/K/V/O projections, FFN1/FFN2, Attention DIMM (QK^T, Score×V)

## Coding Rules
- All paths passed to VTR subprocesses must be **absolute** (VTR changes CWD to its scripts dir)
- `MSB_SA_Ready` is an **OUTPUT** of the `dpe` hard block — each instance needs its own wire
- Adder tree internal wire names must be prefixed with `col{col}` to avoid multi-driver collisions when H > 1
- IMC simulator config is patched at runtime by `gemv_dse.py::patch_imc_config()` — do not hardcode Fmax
- VTR arch XML: only patch `<tile name="wc" ...>` for auto mode; do not touch `<auto_layout>`
- Round 2 DPE counting must be **grid-based** (count_available_wc), NOT area-based — DPE tiles (width=3, height=8 for 512×128) are placed by VTR column priority, not packed by area
- Round 2 replica count P = min(P_dpe, P_clb, P_bram) — **three resource limits**, not just DPE+CLB
- Round 2 throughput: T = n × Fmax (n = GEMV replicas), utilization = fn/f0 (sole metric)
- Multi-seed: Round 2 runs 3 seeds per VTR point (seeds 1,2,3), averages Fmax for stability

## Workflow Pattern
**Plan → Implement → Sanity-check → Run → Verify results → Proceed**

- For any DSE run: always do a 1–3 point dry run first, verify CSV output, then full sweep
- Sanity check output: `dse/results/sanity_check_run.log`
- Round 1: `python gemv_dse.py --round 1` (54 runs, 9 configs × 6 workloads)
- Round 2 prototype: `python gemv_dse.py --round2-proto` (25 runs, fc_2048_256 only)
- Round 2 full: `python gemv_dse.py --round2-full` (300 points × 3 seeds = 900 VTR runs, 5 configs × 3 workloads × 20 (d,c) points)
- Resume interrupted: add `--skip-existing`
- Use `--jobs 12` to limit CPU

## Context Economy Rules
- Do NOT re-read paper_outline.md or dse_experiment_plan.md unless working on paper narrative or methodology changes
- Do NOT re-run VTR on already-completed configs (check `dse/round1/`)
- SESSION_STATE.md is ground truth for project status — update it after every milestone
- TASKS.md is the **prior-submission** sprint log (historical). Current work
  is tracked in the "Active TODO Tracks" section of this file, the relevant
  methodology docs (`paper/methodology/*.md` §TODOs), and user memory
  (`project_multipass_dpe_todos.md` for the P4 track).

## Active TODO Tracks
- **P4 — multi-pass pipelined DPE model** (opened 2026-04-18):
  `gemm_log` sum-over-passes → `L + max(L,O)·(M−1) + O`. Sequence:
  T30 (sim) → T31 (transpose block) → T32 (FC+GEMM RTL re-verify) → T33 (DIMM re-align).
  Full details: `paper/methodology/dpe_pipeline_model.md` §8.1; task breakdown in `TASKS.md` §P4.
