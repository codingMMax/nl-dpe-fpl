# CLAUDE.md — NL-DPE FPGA Research Project

## Project in One Line
NL-DPE FPGA hard block research: crossbar-size DSE (complete) + RTL/sim alignment for paper-target workloads (in progress) for a paper comparing NL-DPE vs Azure-Lily.

## Session Start Protocol
1. Read this file's "Active TODO Track" section below → current in-flight work
2. Read `fc_verification/FIDELITY_METHODOLOGY.md` if working on RTL behavior models or RTL/sim alignment (it is the canonical methodology anchor)
3. Run `git status` to see uncommitted changes
4. (If touching DSE) check `dse/results/` for CSVs and `dse/round1/` for partial VTR outputs

**Important historical context (2026-05-01)**: previous CLAUDE.md described
"AH track" (attention head verification) and "P4 track" (multi-pass DIMM
pipelined model) as active and "all phases closed". Both lived in agent
worktrees that were **never merged to main**. They have been removed from
this file. The actual current state on main is documented in the "Active
TODO Track" section below. `SESSION_STATE.md` is also stale for the same
reason — pending update.

`TASKS.md` is the **prior-submission** sprint log (historical only as of
2026-04-18). Do not treat its open `[ ]` items as active without cross-checking
the live track below.

## Key Paths

### RTL behavior models + verification (live, on main)
| Path | Role |
|------|------|
| `fc_verification/FIDELITY_METHODOLOGY.md` | **Canonical** RTL/sim alignment methodology (§3 DPE arch, §4 single-buffered drain-load overlap pipeline, §5 workload classes VMM/DIMM, §7 tiling) |
| `fc_verification/FC_RTL_PLAN.md` | FC/GEMM RTL build-out plan (Stage 1A→1D) |
| `fc_verification/rtl/dpe_nldpe.v` | NL-DPE behavior model (Model Y FSM, precision-agnostic, ACAM modes) |
| `fc_verification/rtl/dpe_azurelily.v` | Azure-Lily DPE behavior model (Model Y FSM, no ACAM) |
| `fc_verification/rtl/dsp_mac.v` | Azure-Lily DSP-MAC behavior model (int_sop_4 hard block, DSP_WIDTH=4) |
| `fc_verification/rtl/fc_top.v` | Parameterized FC/GEMM top (V/H ≥ 1; Stage 1A V=1 H=1 validated) |
| `fc_verification/tb_dpe_vmm.v`, `tb_dpe_acam.v`, `tb_dsp_mac.v`, `tb_fc.v` | Primitive + FC smoke TBs |
| `fc_verification/Makefile` | CLI build harness (R/C/BUF/PRECISION/PIPELINE_DEPTH/K/DSP_WIDTH knobs) |
| `fc_verification/run_dpe_smoke.py` | Primitive smoke sweep harness (36 cases) |
| `fc_verification/run_fc_smoke.py` | FC smoke sweep harness (Stage 1A subset) |
| `nl_dpe/gen_dpe_stub.py` | DPE behavior model generator (writes `fc_verification/rtl/dpe_*.v` from per-arch JSON) |
| `nl_dpe/gen_dsp_mac.py` | DSP-MAC behavior model generator |

### Simulator (live, on main)
| Path | Role |
|------|------|
| `azurelily/IMC/test.py` | IMC energy/latency simulator (supports `fc`, `attention`, `bert_tiny` models) |
| `azurelily/IMC/imc_core/imc_core.py` | Canonical `run_gemm` encoding (passes_per_dpe + pipeline cycle formula) |
| `azurelily/models/attention.py` | Attention energy model (linear_Q/K/V + mac_qk + softmax + mac_sv) |
| `azurelily/models/bert_tiny.py` | BERT-Tiny model (2L/2H/128d/512ff, embedding + LayerNorm + multi-head attention + FFN) |
| `azurelily/nn/layernorm_layer.py` | LayerNorm layer class for scheduler dispatch |
| `azurelily/nn/embedding_layer.py` | Embedding layer class for scheduler dispatch |

### DSE infrastructure (complete, on main)
| Path | Role |
|------|------|
| `gemv_dse.py` | DSE orchestrator: Round 1 (`--round 1`), Round 2 prototype (`--round2-proto`), Round 2 full (`--round2-full`) |
| `nl_dpe/gen_arch_xml.py` | VTR arch XML generator (auto / fixed_clb_replace / fixed_dsp_bram / fixed_dsp_clb_replace) |
| `nl_dpe/area_power.py` | DPE physical specs: `dpe_specs(rows, cols)` → tile W/H/area/power |
| `nl_dpe/run_vtr.py` | VTR flow runner (called by gemv_dse.py) |
| `dse/results/` | CSVs, JSONs, plots, analysis (DSE outputs) |
| `dse/round1/<config>/<workload>/` | Per-run VTR outputs (Round 1) |
| `dse/round2_proto/`, `dse/round2_full/` | Per-run VTR outputs (Round 2) |

### Methodology docs
| Path | Role |
|------|------|
| `dse_experiment_plan.md` | Full DSE methodology spec |
| `paper_outline.md` | Paper structure and narrative |
| `paper/methodology/attention_dimm_mapping.md` | Attention → DPE/DSP mapping, K-identity, W=16 lane spec |
| `paper/methodology/dpe_pipeline_model.md` | Per-pass DPE dataflow model (Layout A vs B, transpose, multi-pass pipelining) — design-space reference |

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
- TASKS.md is the **prior-submission** sprint log (historical). Current work is tracked in:
  1. The "Active TODO Track" section of this file
  2. The task tools (TaskList) — task IDs #67+ describe in-flight RTL/sim work
  3. `fc_verification/FIDELITY_METHODOLOGY.md` for canonical methodology
- `SESSION_STATE.md` is currently stale (describes worktree-only AH/P4 work that didn't merge); pending update.

## Active TODO Track

**RTL build-out on main, FIDELITY_METHODOLOGY-aligned** (opened 2026-05-01)

The project is rebuilding the RTL/sim alignment cascade on main, anchored to
`fc_verification/FIDELITY_METHODOLOGY.md`. The previous AH-track and P4-track
work documented in older CLAUDE.md versions lived in agent worktrees that
were never merged; they are reference material only.

### Status (as of 2026-05-01)

**Done — committed**:
- DPE behavior model primitives — `dpe_nldpe.v`, `dpe_azurelily.v`, `dsp_mac.v`. Module name `dpe` matches VTR arch XML `<model name="dpe">` contract. Model Y FSM: precision-agnostic, controller-driven compute hold (commits `202cdf1`, `6a7fdd6`, `d60493d`).
- Generators — `nl_dpe/gen_dpe_stub.py`, `gen_dsp_mac.py` emit primitives from per-arch JSON.
- Primitive smoke TBs — `tb_dpe_vmm.v`, `tb_dpe_acam.v`, `tb_dsp_mac.v`. 36-case smoke sweep: ALL PASS (commit `d60493d`).
- Build harness — `Makefile` with CLI knobs for R/C/BUF/PRECISION/PIPELINE_DEPTH/K/DSP_WIDTH (commit `618f8c1`).
- Methodology anchor — `FIDELITY_METHODOLOGY.md` (commit `511126e`).
- Simulator alignment — Azure-Lily principle fixes (F1.5 always-CLB activation, F2/F3/F4 §4 refactor, F5 W=16 DIMM lanes); `Config.patch` runtime override (commits `0f41295`, `8933ae9`, `bd7ef8c`, `ec7ccd5`).

**Done — uncommitted (working tree, post Tasks #82/#83/#84)**:
- Plan doc — `fc_verification/FC_RTL_PLAN.md`.
- Walkthrough docs — `fc_verification/DPE_PRIMITIVE_WALKTHROUGH.md` (primitive FSM), `fc_verification/FC_GEMM_WALKTHROUGH.md` (workload-level fc_top + TB + driver).
- TB pattern fix — `tb_dsp_mac.v` (all-ones × all-ones to avoid int8 sign-wrap aliasing).
- **Task #82 — DPE primitive overlap refactor**: `dpe_nldpe.v`, `dpe_azurelily.v`, `dsp_mac.v` + generators implement single-buffered drain-load overlap with 3 parallel sub-FSMs (LOAD/COMPUTE/OUTPUT) and a depth-4 pass-tagged ring buffer. Single-pass M=1 byte-identical to legacy serial FSM. Primitive smoke 52/52 PASS (36 originals + 16 M-sweep).
- **Stage 1A FC top** — `fc_top.v`, `tb_fc.v`, `run_fc_smoke.py`, Makefile additions. Hierarchical-force weights/inputs; FSM controller; 7 fc_smoke cases.
- **Stages 1B + 1C extension** — V>1 K-tile reduction (CLB adder tree), V=1 H>1 N-tile concatenation (output mux). 6 additional fc_smoke cases.
- **Task #83 — Path A architectural fix (sim + methodology + driver)**: simulator's `imc_core.run_gemm` uses Path A formula `passes_per_dpe = M` (V·H weight-stationary, all DPEs fire in parallel per row). FIDELITY_METHODOLOGY.md §5 rewritten. `+1` CLB cycle gate: `(V > 1) OR (ACTIVATION_MODE AND not HAS_ACAM)`. activation_mode threaded through `run_gemm`. azurelily/IMC/test.py 8/8 PASS.
- **Task #84 — Path A architectural fix (RTL + TB)**: `fc_top.v` v_round/v_round_out redundancy removed; each DPE fires M times (one per output row), not M·V. tb_fc.v cycle expectation matches Path A. Functional pattern (all-ones × all-ones identity weights) verified including ReLU truncation for lenet_fc1_NL.
- **Verifier**: `run_dpe_smoke.py` 52/52 PASS; `run_fc_smoke.py` 13/13 PASS, all 0% fidelity (Stages 1A+1B+1C). Stage 1A regression byte-identical to baseline (114, 114, 270, 478, 331, 330, 1098).

### In flight: Stage 1D — general V·H

After commit of Tasks #82/#83/#84, dispatch Stage 1D agent: combined K-tile reduction (V>1) and N-tile concatenation (H>1). Same `fc_top.v` module, parameter elaboration only. Workloads: vgg_fc3 (V=16 H=4), resnet_fc (V=2 H=4), bert_qkv_batched (M=128 V=1 H=1). Stage 1D may need `vgg_fc2` (V=16 H=16) abbreviated due to iverilog elaboration cost on 256 DPE instances.

After Stage 1D lands: Task #73 (DIMM RTL) and Task #74 (attention head RTL) per the forward plan.

### Forward plan (after overlap refactor lands)

| Task | Scope | Notes |
|---|---|---|
| Stage 1B | `fc_top.v` validates V>1 H=1 K-tile reduction; CLB adder tree + activation LUT cycle | Workloads: lenet_fc1 (M=1 K=400 N=120), vgg_fc1_v synthetic |
| Stage 1C | Validates V=1 H>1 N-tile concatenation (output mux) | Workloads: bert_ffn1 (M=1 K=128 N=512), synthetic_h2 |
| Stage 1D | Validates general V×H | Workloads: vgg_fc2/3, resnet_fc, bert_qkv (M=128 batched-attention input) |
| DIMM RTL (Task #73) | Behavioral DIMM module instantiated against new overlap-aware primitives | Uses `paper/methodology/attention_dimm_mapping.md` |
| Attention head RTL (Task #74) | Composed attention head, end-to-end RTL/sim alignment at N=128 d=64 C=128 W=16 | Replaces worktree-only AH-track work |
| BERT-Tiny end-to-end | Multi-head + LayerNorm + residual + embedding, full inference | Per `azurelily/models/bert_tiny.py` |

### Authoritative docs
- Methodology: `fc_verification/FIDELITY_METHODOLOGY.md` (§3 DPE arch, §4 pipeline, §5 workload classes, §7 tiling)
- Plan: `fc_verification/FC_RTL_PLAN.md` (Stage 1A→1D)
- Pipeline model context: `paper/methodology/dpe_pipeline_model.md` (design-space reference; some sections describe retired Layout B / transpose block / Regime C — design-space reference only, not implemented)
- Attention mapping: `paper/methodology/attention_dimm_mapping.md`
