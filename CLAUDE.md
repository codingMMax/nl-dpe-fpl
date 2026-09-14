# CLAUDE.md — NL-DPE FPGA Research Project

## Project in One Line
NL-DPE FPGA hard block research: crossbar-size DSE (complete) + RTL/sim fidelity alignment (live) + safe-softmax study (complete), for a paper comparing NL-DPE vs Azure-Lily.

## Direction (2026-08-29, pinned)
- **Legacy RTL work lives in `rtl_flow/`** — primitives, fc_top, generators (`rtl_flow/gen/`), per-arch specs (`rtl_flow/specs/`), TBs, smoke harnesses, methodology docs, VTR synth path. Entry point: `rtl_flow/README.md`. The active clean-room `v2/` tree lives at the **repo root** `v2/` (moved 2026-09-13) — work there for v2, in `rtl_flow/` for legacy.
- **Bottom-up primitive-first re-verification is the active plan** (user directive 2026-08-29): current verification is self-consistent but not user-validated (FC functional check is a one-byte pattern; cycle formula from unremembered sessions). Ladder: **primitives → fc_top → softmax → projections+DIMM → (then) mapping+simulator**. Each rung = behavioral charter (user-approved) + independent NumPy oracle + RTL matching both. Ground truth = charter + oracles; faithful RTL enforces them.
- **Clean-room v2 flow (user directive 2026-08-29)**: legacy generated RTL is FROZEN as reference (still green: 52/52, 13/13 — do not edit). New hand-written flow lives in `v2/` at the repo root (spec/ rtl/ tb/ oracle/ sim/ smoke/). Process per primitive: charter → NumPy oracle (before RTL) → starter skeleton → **user hand-writes RTL** (LLM reviews, never edits) → cross-check v2 vs legacy vs oracle on identical stimulus. Legacy is read-only reference; copying from it mid-flight is forbidden (independent-witness property).
- Charter home: `rtl_flow/SPEC.md` — first live spec `v2/spec/dpe_nldpe.md` **v1.1 amended 2026-09-12** (fp32 weights/crossbar + structural fp32 MAC + trunc8 ACAM; P1–P20 closed; supersedes v1.0 FROZEN 2026-08-29). Ladder position: Stage 1.2 (v2 NumPy oracle, v1.1 re-transcription) → skeleton → user hand-write → cross-check.
- The old azurelily simulator stays archived — the new minimal simulator (Stage 5, deferred) will consume the Stage 1–4 charters verbatim.

## Prior direction (2026-08-27 repo reorg)
- The Azure-Lily simulator was **de-submoduled** to `archive/azurelily_simulator/` — reference only. We will build our **own simulator and RTL flow** in this repo; exact direction is TBD and this file is the place it gets pinned down as it takes shape.
- DSE rounds are **complete** (Round 1, Round 2 FC/attention/FC+softmax/flexscore). Results live as CSVs/plots under `dse/results/`; raw VTR run dirs were deleted to save space — do not expect them to exist.
- Legacy tracks (worktree-era AH/P4, `transformer/`, TACO experiments, `block_comp_apr_11/`) are archived under `archive/` — **reference only, not active**. Do not treat anything under `archive/` as live work.

## Session Start Protocol
1. Read "Active TODO Track" below → current in-flight work
2. If working on RTL/sim alignment: read `rtl_flow/docs/FIDELITY_METHODOLOGY.md` (canonical methodology anchor)
3. Run `git status`
4. (If touching DSE-era scripts) results are in `dse/results/`; simulator code is `archive/azurelily_simulator/`

## Repo Map
| Path | Role |
|------|------|
| `rtl_flow/` | **LIVE — all RTL work**: primitives, fc_top, generators, specs, TBs, smoke, docs, VTR path. Entry: `rtl_flow/README.md` |
| `softmax_study/` | **COMPLETE** — safe-softmax RTL + VTR + energy study (AL vs NL-DPE) |
| `nl_dpe/` | DSE-era VTR arch XML / workload-wrapper generators, VTR runner, `area_power.py` (RTL model generators moved to `rtl_flow/gen/`) |
| `dse/` | DSE results (CSVs, JSONs, plots) + Round-1 VTR outputs (`dse/round1/`) |
| `benchmarks/` | BERT-Tiny / CNN benchmark infra (DSE-era, still runs) |
| `paper/` | Paper methodology, figures, scripts, writing materials |
| `archive/` | **Reference only**: `azurelily_simulator/` (de-submoduled), `transformer/`, `azurelily_TACO_experiments/`, `block_comp_apr_11/`, AH/P4 worktree material, legacy VTR outputs |
| `gemv_dse.py` (root) | DSE orchestrator (Round 1 / Round 2, DSE-era) |
| `flexscore_dse.py` (root) | Flexscore budget sweep (DSE-era) |

## Key Paths (live RTL/sim alignment)
| Path | Role |
|------|------|
| `rtl_flow/FIDELITY_METHODOLOGY.md` | **Canonical** RTL/sim alignment methodology (§3 DPE arch, §4 single-buffered drain-load overlap pipeline, §5 workload classes VMM/DIMM, §7 tiling) — in `rtl_flow/docs/` |
| `rtl_flow/SPEC.md` | **Behavioral charter** (Stage 1.1 in progress; open decisions D1–D5) |
| `rtl_flow/docs/FC_RTL_PLAN.md` | FC/GEMM RTL build-out plan (Stage 1A→1D) |
| `rtl_flow/docs/CYCLE_ACCOUNTING.md` | Unified cycle formula: `T(M) = T_fill + (M−1)·T_steady`, `T_fill = L+C+O` |
| `rtl_flow/rtl/dpe_nldpe.v` | NL-DPE behavior model (Model Y FSM, precision-agnostic, ACAM modes) |
| `rtl_flow/rtl/dpe_azurelily.v` | Azure-Lily DPE behavior model (Model Y FSM, no ACAM) |
| `rtl_flow/rtl/dpe_nldpe_faithful.v` | Faithful NL-DPE primitive (double-buffered slice-major substrate; CCYC emerges structurally) |
| `rtl_flow/rtl/dpe_azurelily_faithful.v` | Faithful AL primitive (MAC→ADC→ShiftAdd) |
| `rtl_flow/rtl/dsp_mac.v` | Azure-Lily DSP-MAC behavior model (int_sop_4 hard block, DSP_WIDTH=4) |
| `rtl_flow/rtl/fc_top.v` | Parameterized FC/GEMM top (V×H DPE array, Path A weight-stationary) |
| `rtl_flow/vtr/fc_top_synth.v` | VTR-targeted `fc_top` clone (Task #89 WIP; binds `dpe_blackbox.v`) |
| `rtl_flow/vtr/dpe_blackbox.v` | VTR blackbox port contract (`<model name="dpe">`) |
| `rtl_flow/tb/*.v` | Primitive + FC smoke TBs |
| `rtl_flow/Makefile` | CLI build harness (R/C/BUF/PRECISION/PIPELINE_DEPTH/K/DSP_WIDTH knobs) |
| `rtl_flow/smoke/run_dpe_smoke.py` | Primitive smoke sweep (52 cases) |
| `rtl_flow/smoke/run_fc_smoke.py` | FC smoke sweep (13 cases, `--stage 1A/1B/1C/1D`) |
| `rtl_flow/vtr/run_vtr_smoke.py` | VTR smoke (3 cases, NL only) |
| `rtl_flow/smoke/oracles/` | Independent mac oracles + test vectors (re-verification Stage 1.3 extends these) |
| `rtl_flow/specs/{nl_dpe,azure_lily}.json` | Per-arch hard-block specs (generator inputs) |
| `rtl_flow/gen/gen_dpe_stub.py` | DPE behavior model generator (specs JSON → `rtl_flow/rtl/dpe_*.v`) |
| `rtl_flow/gen/gen_dsp_mac.py` | DSP-MAC behavior model generator |
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

- RTL/sim verification commands (run from `rtl_flow/`):
  - `python3 smoke/run_dpe_smoke.py` — primitive smoke (52 cases)
  - `python3 smoke/run_fc_smoke.py` — FC smoke (13 cases; ~13 min, iverilog ~50 s/case)
  - `python3 vtr/run_vtr_smoke.py` — VTR smoke (needs VTR_ROOT)
- For any DSE-era run: 1–3 point dry run first, verify CSV output, then full sweep; resume with `--skip-existing`; `--jobs 12` to limit CPU

## Context Economy Rules
- Do NOT re-read `paper_outline.md` or `dse/results/plots/round1/round1_analysis.md` unless working on paper narrative
- Do NOT re-run VTR on already-completed configs (check `dse/round1/`)
- Current work is tracked in:
  1. The "Active TODO Track" section below
  2. `rtl_flow/docs/FIDELITY_METHODOLOGY.md` for canonical methodology
- Everything under `archive/` is reference material: read on demand, never edit as live work

## Active TODO Track

**RTL/sim alignment on main, FIDELITY_METHODOLOGY-aligned** (opened 2026-05-01)

### Status (as of 2026-09-13)

**Done — committed**:
- Reorg `bd229f1` (Aug 27 work): azurelily de-submoduled, legacy archived, stale docs purged, CLAUDE.md/README rewritten.
- rtl_flow migration (Aug 29): all RTL assets consolidated under `rtl_flow/` (specs/ gen/ rtl/ tb/ smoke/ docs/ vtr/); paths fixed; generators read `rtl_flow/specs/*.json`.
- DPE behavior model primitives, Model Y FSM, module name `dpe` matching VTR arch XML contract.
- Stages 1A (V=1 H=1), 1B (V>1 H=1), 1C (V=1 H>1) validated.
- `softmax_study/` — COMPLETE (committed Aug 2026).
- **v2 clean-room Stage 1.2 — DONE** (2026-09-13): spec v1.1; `v2/oracle/nldpe_ref.py` + `v2/sim/nldpe_sim.py` self-tests green; independent review passed (values bit-exact vs oracle across all 4 modes/geometries, measured cycles ≡ §5.3 closed form, P1/P10/P11 timeline invariants); `pack_weight_stream` (P17) and `dump_case` file contract added and round-trip verified. v2 tree moved to repo root `v2/`.

**Validation state (regression guards, re-run green after migration)**:
- `rtl_flow/smoke/run_dpe_smoke.py` — 52/52 PASS
- `rtl_flow/smoke/run_fc_smoke.py` — 13/13 PASS, fidelity reported not gated (Stages 1A+1B+1C)
- `rtl_flow/vtr/run_vtr_smoke.py` — 3/3 OK (NL only): bert_qkv_proj, lenet_fc1, bert_ffn1
- `python3 v2/oracle/nldpe_ref.py` — ALL PASS; `python3 v2/sim/nldpe_sim.py` — ALL PASS

**Verification caveat (drives the active plan)**: functional truth for FC is a
one-byte pattern (`tb_fc.v` `expected_byte_fn`); cycle truth is the Task #98
formula authored in unremembered sessions. Bottom-up re-verification is in
progress — see "Direction (2026-08-29)" above.

### In flight: Stage 1.3/1.4 — stimulus contract + hand-written v2 RTL

- Stage 1.3a DONE: `v2/smoke/gen_cases.py` emits §9 stimulus classes for both
  geometries (256×256, 256×512); `dump_case` round-trip verified on both.
- Stage 1.4a DONE: `v2/rtl/dpe_nldpe.v` port-only skeleton — exact I8 surface,
  parameterized R/C/P/BUF, TODO blocks 1–8; elaborates under iverilog.
- Next (user): fp32 add/mul cores (bit-exact vs NumPy, §9 option A), then the
  datapath from spec v1.1 only. Then Stage 1.5 TB/harness + legacy witness.

### Forward plan

| Rung | Scope | Gate |
|---|---|---|
| Stage 1 — primitives | v2 clean-room: spec v1.1 + NumPy oracle + sim (done); hand-written RTL + three-way cross-check | Cross-check green (v2 ≡ oracle; legacy as witness) |
| Stage 2 — fc_top (VMM/projection) | VMM charter; replace one-byte check with full GEMM oracle; re-verify 13 cases | Your sign-off |
| Stage 3 — softmax | Port oracles + RTL into rtl_flow; pin log-domain output contract | Your sign-off |
| Stage 4 — projections + DIMM | Q/K/V composition on trusted fc_top; DIMM charter from `paper/methodology/attention_dimm_mapping.md`, oracle → RTL → smoke | Your sign-off |
| Stage 5 — mapping + simulator | Spec module from Stage 1–4 charters; new minimal sim consuming it; BERT-Tiny end-to-end; VTR closure | Deferred until ladder trusted |

### Authoritative docs
- Methodology: `rtl_flow/docs/FIDELITY_METHODOLOGY.md` (§3 DPE arch, §4 pipeline, §5 workload classes, §7 tiling)
- Cycle accounting: `rtl_flow/docs/CYCLE_ACCOUNTING.md`
- Plan: `rtl_flow/docs/FC_RTL_PLAN.md` (Stage 1A→1D)
- Pipeline model context: `paper/methodology/dpe_pipeline_model.md` (design-space reference; some sections describe retired Layout B / transpose block / Regime C — reference only, not implemented)
- Attention mapping: `paper/methodology/attention_dimm_mapping.md`
- Softmax study: `softmax_study/SOFTMAX_STUDY.md`
