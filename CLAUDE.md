# CLAUDE.md — NL-DPE FPGA Research Project

## Project in One Line
NL-DPE FPGA hard block research: crossbar-size DSE (complete) + RTL/sim fidelity alignment (live) + safe-softmax study (complete), for a paper comparing NL-DPE vs Azure-Lily.

## Direction (2026-08-27 repo reorg)
- The Azure-Lily simulator was **de-submoduled** to `archive/azurelily_simulator/` — reference only. We will build our **own simulator and RTL flow** in this repo; exact direction is TBD and this file is the place it gets pinned down as it takes shape.
- DSE rounds are **complete** (Round 1, Round 2 FC/attention/FC+softmax/flexscore). Results live as CSVs/plots under `dse/results/`; raw VTR run dirs were deleted to save space — do not expect them to exist.
- Legacy tracks (worktree-era AH/P4, `transformer/`, TACO experiments, `block_comp_apr_11/`) are archived under `archive/` — **reference only, not active**. Do not treat anything under `archive/` as live work.

## Session Start Protocol
1. Read "Active TODO Track" below → current in-flight work
2. If working on RTL/sim alignment: read `fc_verification/FIDELITY_METHODOLOGY.md` (canonical methodology anchor)
3. Run `git status`
4. (If touching DSE-era scripts) results are in `dse/results/`; simulator code is `archive/azurelily_simulator/`

## Repo Map
| Path | Role |
|------|------|
| `fc_verification/` | **LIVE** — RTL behavior models, TBs, smoke harnesses, methodology docs |
| `softmax_study/` | **COMPLETE** — safe-softmax RTL + VTR + energy study (AL vs NL-DPE) |
| `nl_dpe/` | DPE physical specs (`area_power.py`), VTR arch XML + stub generators, VTR runner |
| `dse/` | DSE results (CSVs, JSONs, plots) + Round-1 VTR outputs (`dse/round1/`) |
| `benchmarks/` | BERT-Tiny / CNN benchmark infra (DSE-era, still runs) |
| `paper/` | Paper methodology, figures, scripts, writing materials |
| `archive/` | **Reference only**: `azurelily_simulator/` (de-submoduled), `transformer/`, `azurelily_TACO_experiments/`, `block_comp_apr_11/`, AH/P4 worktree material, legacy VTR outputs |
| `gemv_dse.py` (root) | DSE orchestrator (Round 1 / Round 2, DSE-era) |
| `flexscore_dse.py` (root) | Flexscore budget sweep (DSE-era) |

## Key Paths (live RTL/sim alignment)
| Path | Role |
|------|------|
| `fc_verification/FIDELITY_METHODOLOGY.md` | **Canonical** RTL/sim alignment methodology (§3 DPE arch, §4 single-buffered drain-load overlap pipeline, §5 workload classes VMM/DIMM, §7 tiling) |
| `fc_verification/FC_RTL_PLAN.md` | FC/GEMM RTL build-out plan (Stage 1A→1D) |
| `fc_verification/CYCLE_ACCOUNTING.md` | Unified cycle formula: `T(M) = T_fill + (M−1)·T_steady`, `T_fill = L+C+O` |
| `fc_verification/rtl/dpe_nldpe.v` | NL-DPE behavior model (Model Y FSM, precision-agnostic, ACAM modes) |
| `fc_verification/rtl/dpe_azurelily.v` | Azure-Lily DPE behavior model (Model Y FSM, no ACAM) |
| `fc_verification/rtl/dpe_nldpe_faithful.v` | Faithful NL-DPE primitive (double-buffered slice-major substrate; CCYC emerges structurally) |
| `fc_verification/rtl/dpe_azurelily_faithful.v` | Faithful AL primitive (MAC→ADC→ShiftAdd) |
| `fc_verification/rtl/dsp_mac.v` | Azure-Lily DSP-MAC behavior model (int_sop_4 hard block, DSP_WIDTH=4) |
| `fc_verification/rtl/fc_top.v` | Parameterized FC/GEMM top (V×H DPE array, Path A weight-stationary) |
| `fc_verification/rtl/fc_top_synth.v` | VTR-targeted `fc_top` clone (uncommitted WIP; binds `dpe_blackbox.v`) |
| `fc_verification/rtl/dpe_blackbox.v` | VTR blackbox port contract (`<model name="dpe">`) |
| `fc_verification/tb_*.v` | Primitive + FC smoke TBs |
| `fc_verification/Makefile` | CLI build harness (R/C/BUF/PRECISION/PIPELINE_DEPTH/K/DSP_WIDTH knobs) |
| `fc_verification/run_dpe_smoke.py` | Primitive smoke sweep (52 cases) |
| `fc_verification/run_fc_smoke.py` | FC smoke sweep (13 cases, `--stage 1A/1B/1C/1D`) |
| `fc_verification/run_vtr_smoke.py` | VTR smoke (3 cases, NL only; uncommitted WIP) |
| `nl_dpe/gen_dpe_stub.py` | DPE behavior model generator (per-arch JSON → `fc_verification/rtl/dpe_*.v`) |
| `nl_dpe/gen_dsp_mac.py` | DSP-MAC behavior model generator |
| `softmax_study/rtl/softmax_{al,nldpe}.v` | Safe-softmax RTL (AL CLB/DSP; NL 16·N_EXP+1 DPEs, log-domain) |

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
- DPE cycle axioms: NL LOAD=52 / COMPUTE=10 / OUTPUT=52 → T_fill=114, T_steady=52; AL 256 / 10 / 64 → 330 / 256

## Coding Rules
- All paths passed to VTR subprocesses must be **absolute** (VTR changes CWD to its scripts dir)
- `MSB_SA_Ready` is an **OUTPUT** of the `dpe` hard block — each instance needs its own wire
- Adder tree internal wire names must be prefixed with `col{col}` to avoid multi-driver collisions when H > 1
- VTR arch XML: only patch `<tile name="wc" ...>` for auto mode; do not touch `<auto_layout>`
- DSE-era scripts import the IMC simulator from `archive/azurelily_simulator/` (post-reorg path) — simulator config is patched at runtime by `gemv_dse.py::patch_imc_config()`, do not hardcode Fmax
- Round 2 DPE counting must be **grid-based** (count_available_wc), NOT area-based — DPE tiles (width=3, height=8 for 512×128) are placed by VTR column priority, not packed by area
- Round 2 replica count P = min(P_dpe, P_clb, P_bram) — **three resource limits**, not just DPE+CLB
- Round 2 throughput: T = n × Fmax (n = GEMV replicas), utilization = fn/f0 (sole metric)
- Multi-seed: Round 2 runs 3 seeds per VTR point (seeds 1,2,3), averages Fmax for stability

## Workflow Pattern
**Plan → Implement → Sanity-check → Run → Verify results → Proceed**

- RTL/sim verification commands (run from `fc_verification/`):
  - `python3 run_dpe_smoke.py` — primitive smoke (52 cases)
  - `python3 run_fc_smoke.py` — FC smoke (13 cases)
  - `python3 run_vtr_smoke.py` — VTR smoke (needs VTR_ROOT)
- For any DSE-era run: 1–3 point dry run first, verify CSV output, then full sweep; resume with `--skip-existing`; `--jobs 12` to limit CPU

## Context Economy Rules
- Do NOT re-read `paper_outline.md` or `dse/results/plots/round1/round1_analysis.md` unless working on paper narrative
- Do NOT re-run VTR on already-completed configs (check `dse/round1/`)
- Current work is tracked in:
  1. The "Active TODO Track" section below
  2. `fc_verification/FIDELITY_METHODOLOGY.md` for canonical methodology
- Everything under `archive/` is reference material: read on demand, never edit as live work

## Active TODO Track

**RTL/sim alignment on main, FIDELITY_METHODOLOGY-aligned** (opened 2026-05-01)

### Status (as of 2026-08-27)

**Done — committed**:
- DPE behavior model primitives — `dpe_nldpe.v`, `dpe_azurelily.v`, `dsp_mac.v`; Model Y FSM; module name `dpe` matches VTR arch XML `<model name="dpe">` contract.
- Generators — `nl_dpe/gen_dpe_stub.py`, `gen_dsp_mac.py`, faithful generators.
- Build harness — `Makefile` with CLI knobs.
- Methodology anchor — `FIDELITY_METHODOLOGY.md`.
- Simulator alignment fixes (F1.5, F2/F3/F4, F5 W=16 DIMM lanes; `Config.patch`) — code now in `archive/azurelily_simulator/`.
- Tasks #82 (primitive overlap refactor), #83 (Path A sim fix), #84 (Path A RTL fix), #98/#99 (cycle-formula unification + double-buffered LOAD) — sim side in archived simulator, RTL side in `fc_top.v`/TB expectations.
- Stages 1A (V=1 H=1), 1B (V>1 H=1), 1C (V=1 H>1) validated.
- `softmax_study/` — COMPLETE: AL vs NL safe-softmax, 8/8 smoke bit-exact, 24 VTR runs, energy model. (committed Aug 2026)

**Validation state (regression guards)**:
- `run_dpe_smoke.py` — 52/52 PASS
- `run_fc_smoke.py` — 13/13 PASS, fidelity reported not gated (Stages 1A+1B+1C)
- `run_vtr_smoke.py` — 3/3 OK (NL only): bert_qkv_proj, lenet_fc1, bert_ffn1

**Uncommitted WIP in working tree**:
- `fc_verification/rtl/fc_top_synth.v` + `fc_verification/run_vtr_smoke.py` (Task #89, VTR smoke)
- 2026-08-27 reorg: `archive/` created, azurelily de-submoduled, stale docs deleted, path fixes

### In flight: Stage 1D — general V·H

Combined K-tile reduction (V>1) and N-tile concatenation (H>1). Same `fc_top.v` module, parameter elaboration only. Workloads: vgg_fc3 (V=16 H=4), resnet_fc (V=2 H=4), bert_qkv_batched (M=128 V=1 H=1). Stage 1D may need `vgg_fc2` (V=16 H=16) abbreviated due to iverilog elaboration cost on 256 DPE instances.

### Forward plan

| Task | Scope | Notes |
|---|---|---|
| Stage 1D | Validates general V×H | Workloads: vgg_fc2/3, resnet_fc, bert_qkv (M=128 batched-attention input) |
| DIMM RTL (Task #73) | Behavioral DIMM module against overlap-aware primitives | Uses `paper/methodology/attention_dimm_mapping.md` |
| Attention head RTL (Task #74) | Composed attention head, end-to-end RTL/sim alignment at N=128 d=64 C=128 W=16 | Replaces worktree-only AH-track work |
| BERT-Tiny end-to-end | Multi-head + LayerNorm + residual + embedding, full inference | Per archived `bert_tiny.py` model |
| **New in-repo simulator** | Replaces archived azurelily simulator; spec TBD | Pin direction here when decided |

### Authoritative docs
- Methodology: `fc_verification/FIDELITY_METHODOLOGY.md` (§3 DPE arch, §4 pipeline, §5 workload classes, §7 tiling)
- Cycle accounting: `fc_verification/CYCLE_ACCOUNTING.md`
- Plan: `fc_verification/FC_RTL_PLAN.md` (Stage 1A→1D)
- Pipeline model context: `paper/methodology/dpe_pipeline_model.md` (design-space reference; some sections describe retired Layout B / transpose block / Regime C — reference only, not implemented)
- Attention mapping: `paper/methodology/attention_dimm_mapping.md`
- Softmax study: `softmax_study/SOFTMAX_STUDY.md`
