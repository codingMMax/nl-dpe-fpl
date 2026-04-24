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
| `nl_dpe/gen_attention_wrapper.py` | Parameterized attention head RTL generator (DIMM stages) |
| `nl_dpe/gen_azurelily_fc_wrapper.py` | Azure-Lily FC wrapper generator (dsp_mac, AH track T1) |
| `fc_verification/rtl/{nldpe,azurelily}_attn_head_d64_c128.v` | Composed attention-head top RTL (FC_QKV + DIMM + FC_O), AH track T2 |
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

- **AH — Attention Head RTL verification + latency alignment**
  (opened 2026-04-24, pre-implementation): **Stage 1 (T1) ready to
  start.** Ultimate goal: end-to-end RTL↔sim cycle-accurate alignment
  for a composed attention head (FC_Q/K/V → DIMM top → FC_O), both
  architectures, all residuals classified `modelling_granularity`.

  **Scope anchor (frozen):** single config point N=128, d=64, C=128,
  W=16, W_DPE=40, K_id=2 — inherits verified DIMM-top surface.
  Configs: `nldpe_attn_head_d64_c128`, `azurelily_attn_head_d64_c128`.

  **Prerequisites:** NL-DPE FC ✓ (12/12 Phase 2), Azure-Lily FC ✗
  (T1 closes this gap). NL-DPE DIMM top ✓ (P3/4, +42 E2E m.g.),
  Azure-Lily DIMM top ✓ (P5/6A, +2 E2E m.g.).

  **Stages:**
  - **T1 — Azure-Lily FC Phase-2 harness** (prerequisite for T4 only;
    independent of T2, can run in parallel or either order). New
    `tb_azurelily_fc_{512_128,2048_256}.v`, extend
    `phase2_known_deltas.json` and `run_fc_phase2.py`. Gate: unified
    Phase-2 report shows 12/12 PASS for both archs.
  - **T2 — Attention-head top generator + RTL.** New
    `nl_dpe/gen_{nldpe,azurelily}_attn_head_top.py` (mirrors
    `gen_dimm_*_top.py`), emits RTLs in `fc_verification/rtl/`.
    Wire Q/K/V FC → DIMM top → O FC with shared valid/ready_n
    interface, `ready_n=1'b0` hardwired. Retire the stale hand-written
    `nl_dpe/attention_head_1_channel.v`. Gate: iverilog clean +
    resource sanity (NL-DPE ≈ 96 DPE, AL ≈ 40 DSP per head).
  - **T3 — Functional TBs.** Scaled-identity Q/K/V projections +
    identity V-matrix + one-hot input → hand-computable output. Gate:
    cross-arch equivalence within int8 tolerance (first cross-arch
    functional equivalence check in the project).
  - **T4 — Latency TBs + sim extractor + known-deltas.** Per-stage
    timestamps on 5 stages (FC_Q, DIMM_score, DIMM_softmax, DIMM_wsum,
    FC_O) + 2 handoff boundaries. Extend `gen_expected_cycles.py` to
    wrap `_run_attention_pipeline`. New `phase7_known_deltas.json`.
    Residual budget: NL-DPE E2E ≤ +50 cyc (≈ 42 DIMM + 2×~4 handoff),
    AL E2E ≤ +10 cyc (≈ 2 DIMM + 2×~4 handoff), all
    `modelling_granularity`. Gate: both `run_checks.py --config
    *_attn_head_d64_c128` exit 0.
  - **T5 — VTR + regression gate extension.** 3-seed VTR per arch,
    strict DPE=96 / DSP≈40 counts, append two new lines to
    `VERIFICATION.md §Phase 5` gate list. Gate: all 7 regression
    commands exit 0.

  **Dependency DAG:** T1 ∥ T2 are independent (no shared files, no
  artifact dependency — T2 can emit the head-top RTL without T1's
  verification having passed). T3 needs T2 only (functional test uses
  identity projections). **T4 needs both T1 and T2** — the real join
  point, where AL FC known-deltas from T1 are needed to attribute
  head-level residuals cleanly. T5 needs T4. Soft guidance: T1 PASS
  also improves T3 diagnostic clarity (isolates any functional blame
  to DIMM or handoff rather than FC).

  **Explicitly deferred (out of this plan):**
  - Multi-N attention head — re-emit DIMM + head at N ∈ {256, 512,
    1024} and re-run Phase 3/5 per N before composing. Separate
    track after Stage 5 closes; prerequisite for paper-wide seq_len
    scaling story.
  - Multi-head + BERT-block composition (2 heads, LayerNorm,
    residual, embedding). Partly covered by monolithic
    `benchmarks/rtl/bert_tiny_*.v` (VTR-only, no cycle alignment).
  - d ≠ 64 regime (d=128 → K_id=1, d=32 → K_id=4) — FSM paths
    untested.

  **Authoritative plan:**
  `fc_verification/plans/ATTENTION_HEAD_VERIFICATION_PLAN.md`
  **Upstream plan (closed):**
  `fc_verification/plans/DIMM_FULL_VERIFICATION_PLAN.md`
  **Session-recovery memory:** `project_attention_head_todos.md`

- **P4 — multi-pass pipelined DPE model** (opened 2026-04-18, scope
  reduced 2026-04-19 to Layout A + Regime B only): **ALL PHASES CLOSED
  2026-04-20.** Layout A + Regime B committed path; sim and RTL
  aligned with annotated FSM-granularity residuals. No structural
  deltas remain.

  **Final DIMM-top alignment** (NL-DPE, N=128 d=64 W=16):
  score 260/244 Δ+16 · softmax 27/17 Δ+10 · wsum 252/236 Δ+16 ·
  E2E 539/497 Δ+42 — all classified `modelling_granularity` with
  file:line citations in `fc_verification/phase3_known_deltas.json`.

  **Phases (all closed):**
  - **Phase 1 — sim Regime B swap (Layout A):** ✅ `c15797f` / `92bbb00`.
    `gemm_log` emits `T(M) = L_A · M + O`.
  - **Phase 2 — FC RTL re-verify + func + latency + VTR:** ✅
    `1678443`, `86e539b`. 12/12 FC configs pass; +4 compute /
    +1 valid_n annotated. Block-level comparison figures regenerable
    from `block_comp_apr_11/results/block_comparison_results.csv`
    via `plot_block_comparison.py`.
  - **Phase 2.1 — GEMM DSE smoke:** ✅ `81b2517`, `7431af0`. 48-point
    DSE on 4 real-benchmark GEMM shapes; winner **512×128** (matches
    Round-1). PDFs in `dse/gemm_phase2_1/results/`.
  - **Phase 3 — DIMM RTL re-verify under Regime-B sim:** ✅ `145a85e`.
    TB NBA race fixed; stage extraction updated; all residuals
    annotated as modelling_granularity.
  - **Phase 4 — wsum RTL widening (1×1 → 128×128):** ✅ `844b4a8`.
    Closed the last structural delta. Fmax +14% (90.1 → 102.9 MHz),
    CLB −10%, BRAM −50% on DIMM top.
  - **Docs — apple-to-apple pipeline comparison:** ✅ `3cceca7`.
    `fc_verification/DIMM_pipeline_model_vs_rtl.md` shows sim and RTL
    in a shared 5-phase notation (L / F / D / S / W) with per-cycle
    delta attribution.

  **Follow-ups (non-blocking):**
  - Phase 2.1 full sweep — extend 4 workloads to 6 (BERT FFN1,
    VGG-16 block-4 conv) for paper-wide GEMM DSE coverage.
  - Softmax probe-placement tidy — cosmetic, +10 Δ is probe-convention.
  - Azure-Lily DIMM functional parser regex — pre-existing, orthogonal.

  **Out of scope — retired / archived** (design-space reference only
  in the model doc §§3.2, 4, 5.4, 5.7): Layout B as an active
  alternative (archived), the transpose block (retired, was TODO 2.1),
  and Regime C / double-buffering (retired, was TODO 3 — archived as
  reference only).

  - **Model & assumptions:** `paper/methodology/dpe_pipeline_model.md`
    §§1–7 (analog IMC primer, Layout A vs B design-space references,
    multi-pass timing, Layout A 512×128 walkthrough in §5.3.1, committed
    layout choice in §5.7)
  - **What is implemented today:** same doc §8 (RTL = Layout A +
    Regime B; sim = Regime A pre-Phase 1, Regime B post-Phase 1)
  - **Phase definitions with detail:** same doc §8.1
  - **Mapping-doc scope of changes:** `paper/methodology/attention_dimm_mapping.md` §10
  - **Verification baseline to beat:** `fc_verification/VERIFICATION.md`
    Phase I.2 (score 260 / softmax 27 / wsum 274 / E2E 561 cyc,
    Layout A, Regime A sim — post-Phase 1 sim moves to Regime B)
  - **Session-recoverable sequencing:** memory `project_multipass_dpe_todos.md`
