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
  (opened 2026-04-24): **T2v2/T3v2/T4v2 closed (commit `e118b11`,
  2026-04-24).** Streaming FC refactor + sim oracle + arch-tagged
  `phase7_known_deltas.json` + `run_checks.py` AH dispatcher all
  landed; both `run_checks.py --config *_attn_head_d64_c128` exit 0.
  T5v2 (VTR 3-seed) pending user sign-off. T6 (BERT-Tiny generator
  refinement) remains future follow-up.

  **Scope anchor (frozen):** single config point N=128, d=64, C=128,
  W=16, W_DPE=40, K_id=2 — inherits verified DIMM-top surface.
  Configs: `nldpe_attn_head_d64_c128`, `azurelily_attn_head_d64_c128`.

  **Final per-stage alignment** (RTL ↔ sim, all residuals classified
  in `phase7_known_deltas.json`):

  | Stage | NL-DPE RTL/sim/Δ/class | AL RTL/sim/Δ/class |
  |---|---|---|
  | linear_qkv  | 3,350 / 2,424 / +926 / m.g. | 4,529 / 4,000 / +529 / m.g. |
  | mac_qk      | 1,948 / 5,690 / −3,742 / structural | 1,683 / 6,661 / −4,978 / structural |
  | softmax_exp | 8 / 650 / −642 / structural | 2,289 / 938 (exp+norm fold) / +1,351 / m.g. |
  | softmax_norm| 10 / 376 / −366 / structural | 0 / — / 0 (folded) / m.g. |
  | mac_sv      | 251 / 5,339 / −5,088 / structural | 32 / 6,091 / −6,059 / structural |
  | **E2E**     | **6,692 / 14,480 / −7,788 / structural** | **8,597 / 17,689 / −9,092 / structural** |

  Negative residuals are classified `structural` because the sim's
  `gemm_log`/`gemm_dsp` analytical bodies are conservative single-lane
  lower-bounds while the RTL realises W=16 hardware-lane parallelism.
  Positive residuals are `modelling_granularity` (FSM transitions,
  streaming fill/drain edges, AL CLB-serial softmax bottleneck).

  **Resource counts (per head):**
  - NL-DPE: 6 DPE (3 arms × 2 ping-pong) + 64 DIMM = **70 DPE**
  - Azure-Lily: 192 dsp_mac (3 arms × 64 parallel-output) + 32 DIMM
    + 16 clb_softmax = **224 dsp_mac + 16 softmax**

  **Stages:**
  - **T1 ✓ closed (commit `9e6a913`)** — AL FC Phase-2 harness, 14/14
    unified gate (NL-DPE 12/12 + AL 2/2).
  - **T2 v0 closed (commit `2f5956e`); T2v2 ✓ closed (commit
    `e118b11`)** — streaming FC composition matching sim's
    `attention_model`. New `nl_dpe/gen_nldpe_attn_head_top.py`
    (instantiates `fc_top_qkv_streaming.v` ping-pong DPE) + new
    `nl_dpe/gen_azurelily_attn_head_top.py` (N parallel dsp_macs
    parallel-output streaming, 64 dsp_macs per arm). `gen_gemv_wrappers.py`
    extended with additive `streaming=True` mode (default off,
    preserves T1 / Phase-2 single-inference behaviour). Both head
    RTLs drop O projection per sim's attention_model definition.
  - **T3v2 ✓ closed (commit `e118b11`, partial in `2e0c559`)** —
    Combined functional+latency TBs `tb_{nldpe,azurelily}_attn_head_v2.v`
    drive N=128 tokens with identity Q/K/V weights. AL sim
    `total_softmax_lanes` config + 3 AL RTL composition bugs fixed
    en route (commit `2e0c559`). Functional Overall=PASS for both archs.
  - **T4v2 ✓ closed (commit `e118b11`)** — Sim oracle invokes
    `attention_model` at full N-token scope, captured into
    `expected_cycles.json`. Per-stage probes (linear_qkv, mac_qk,
    softmax_exp/norm, mac_sv, e2e) emit timestamped boundaries.
    `phase7_known_deltas.json` arch-tagged with file:line root-cause
    citations for every residual. `run_checks.py` extended with AH
    dispatcher (`check_ah_attn_head`) including AL softmax_exp+norm
    fold semantics. Gate: both `run_checks.py --config
    *_attn_head_d64_c128` exit 0.
  - **T5v2 — VTR + regression (PENDING user sign-off, separate
    stage).** Strict counts target: NL-DPE DPE=70, AL
    DSP≈224. Will append to `VERIFICATION.md §Phase 7` gate list
    once VTR 3-seed numbers land.
  - **T6 — BERT-Tiny generator refinement (post-T5 follow-up).** The
    verified head becomes the canonical reference for diffing against
    `gen_bert_tiny_wrapper.py`. Surfaces likely bugs (W=1 vs paper-spec
    W=16, etc.) in the BERT-Tiny benchmark RTL family. Out of scope
    for the AH track gate.

  **Gate command list (all 5 commands exit 0 post-`e118b11`):**
  ```
  python3 azurelily/IMC/test_gemm_log_regime_b.py                          # Phase 1
  python3 fc_verification/run_fc_phase2.py --arch both --skip-vtr           # T1 14/14 PASS
  python3 fc_verification/run_checks.py --config nldpe_dimm_top_d64_c128    # P3+P4
  python3 fc_verification/run_checks.py --config azurelily_dimm_top_d64_c128  # P5+P6A
  python3 fc_verification/run_checks.py --config nldpe_attn_head_d64_c128   # P7 NL-DPE
  python3 fc_verification/run_checks.py --config azurelily_attn_head_d64_c128  # P7 AL
  ```

  **Dependency DAG:** T1 ∥ T2 are independent; T3 needs T2 only;
  T4 needs both T1 and T2 (residual attribution join point);
  T5 needs T4.

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
