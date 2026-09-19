# CLAUDE.md — NL-DPE FPGA Research Project

## Project in One Line
NL-DPE FPGA hard block research: crossbar-size DSE (complete) + RTL/sim fidelity alignment (live) + safe-softmax study (complete), for a paper comparing NL-DPE vs Azure-Lily.

## Direction (2026-08-29, pinned)
- **Legacy RTL work lives in `rtl_flow/`** — primitives, fc_top, generators (`rtl_flow/gen/`), per-arch specs (`rtl_flow/specs/`), TBs, smoke harnesses, methodology docs, VTR synth path. Entry point: `rtl_flow/README.md`. The active clean-room `v2/` tree lives at the **repo root** `v2/` (moved 2026-09-13) — work there for v2, in `rtl_flow/` for legacy.
- **Bottom-up primitive-first re-verification is the active plan** (user directive 2026-08-29): current verification is self-consistent but not user-validated (FC functional check is a one-byte pattern; cycle formula from unremembered sessions). Ladder: **primitives → fc_top → softmax → projections+DIMM → (then) mapping+simulator**. Each rung = behavioral charter (user-approved) + independent NumPy oracle + RTL matching both. Ground truth = charter + oracles; faithful RTL enforces them.
- **Clean-room v2 flow (user directive 2026-08-29)**: legacy generated RTL is FROZEN as reference (still green: 52/52, 13/13 — do not edit). New hand-written flow lives in `v2/` at the repo root (spec/ rtl/ tb/ oracle/ sim/ smoke/). Process per primitive: charter → NumPy oracle (before RTL) → starter skeleton → **user hand-writes RTL** (LLM reviews, never edits) → cross-check v2 vs legacy vs oracle on identical stimulus. Legacy is read-only reference; copying from it mid-flight is forbidden (independent-witness property).
- Charter home: `rtl_flow/SPEC.md` — first live spec `v2/spec/dpe_nldpe.md` **v2.0 clean integer rewrite 2026-09-14** (int8 weights/activations, exact integer MAC, integer ACAM forms + trunc8 low byte; P1–P13 + P16 + P21–P26 live; supersedes the v1.1 fp32 amendment per advisor consultation). Ladder position: Stage 1.2 (v2 oracle + sim, integer) done → user hand-write → cross-check.
- The old azurelily simulator stays archived — the new minimal simulator (Stage 5, deferred) will consume the Stage 1–4 charters verbatim.

## Prior direction (2026-08-27 repo reorg)
- The Azure-Lily simulator was **de-submoduled** to `archive/azurelily_simulator/` — reference only. We will build our **own simulator and RTL flow** in this repo; exact direction is TBD and this file is the place it gets pinned down as it takes shape.
- DSE rounds are **complete** (Round 1, Round 2 FC/attention/FC+softmax/flexscore). Results live as CSVs/plots under `dse/results/`; raw VTR run dirs were deleted to save space — do not expect them to exist.
- Legacy tracks (worktree-era AH/P4, `transformer/`, TACO experiments, `block_comp_apr_11/`) are archived under `archive/` — **reference only, not active**. Do not treat anything under `archive/` as live work.

## Session Start Protocol
1. Read "Active TODO Track" below → current in-flight work
2. If working on RTL/sim alignment: read the v2 charter `v2/spec/dpe_nldpe.md` (v2.0.1, integer) for v2 work; `rtl_flow/docs/FIDELITY_METHODOLOGY.md` is legacy cadence reference (double-buffered T_steady=52, superseded for v2 by P1/A9)
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
| `rtl_flow/FIDELITY_METHODOLOGY.md` | Legacy RTL/sim methodology (§3 DPE arch, §5 workload classes, §7 tiling) — its §3.2/§4 double-buffered cadence (T_steady=52) is **superseded for v2 by `v2/spec/dpe_nldpe.md` P1/A9** (single buffer: 60/104) |
| `rtl_flow/SPEC.md` | **Charter home pointer** → first live spec `v2/spec/dpe_nldpe.md` **v2.0.1** (integer rewrite 2026-09-14/15; P1–P27, D1/D8) |
| `v2/spec/dpe_nldpe.md` | **First live v2 charter** (v2.0.1): int8 weights/activations, exact integer MAC, integer ACAM + trunc8; P1–P27; dual-compare contract P26 |
| `v2/spec/gemm.md` | **Stage-2 charter (v0.3 FROZEN 2026-09-17)**: GEMM array = V×H `dpe` instances + byte-tree reduce + lane serializer; tiles REGULAR, **no ACAM after reduction**, `out8 = trunc8(Σ y_v)` exactly; G1–G9 |
| `v2/oracle/nldpe_ref.py` | v2 NumPy oracle — numerical ground truth (GATE 1 reference; exact integer, no quantization policy) |
| `v2/sim/nldpe_sim.py` | v2 golden model — owns quantize/trunc8; `dump_case` certifies each case vs oracle (GATE 1) before writing expected bits |
| `v2/smoke/run_dpe_rtl.py` + `v2/tb/tb_dpe_nldpe.v` | v2 RTL cross-check harness (GATE 2: dual compare + Δ_impl/T_steady gates) |
| `v2/rtl/dpe_nldpe.v` | Hand-written v2 integer RTL — all 7 blocks; structural control channels, Δ_impl = 0 (Stage 1.5 complete) |
| `v2/smoke/test_dpe_primitive.py` | DPE-primitive case verifier — no args: corpus table (`sim_cyc/rtl_cyc/delta_cyc`, `y32_match`, `out8_match`, verdict); `--list` index (`*` = latest run); `<case>` full view |
| `v2/oracle/gemm_ref.py` + `v2/sim/gemm_sim.py` | Stage-2 GEMM oracle (exact composition) + golden model (V×H `NldpeDpe`, byte tree, low-byte serializer; GATE 1 in `dump_case`) |
| `v2/smoke/stimuli/` | Corpus (gitignored, accumulates across runs) + `manifest.txt` scoping the harness to the latest generation |
| `v2/smoke/logs/` | Test logs (gitignored): `<case>.log` per-case TB stdout + timestamped `rtl_smoke_*.log` run summaries + `latest.log` |
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
- v2 clean-room commands (run from repo root):
  - `python3 v2/smoke/run_dpe_rtl.py [--quick]` — GATE 2 cross-check; writes observed dumps under `stimuli/<case>/observed/` and logs under `v2/smoke/logs/`
  - `python3 v2/smoke/test_dpe_primitive.py` (no args: corpus table) · `--list` · `<case> [--full]` — inspect vectors/expected/observed and verify values/cycles
  - `python3 v2/smoke/legacy_witness.py` — independent legacy witness on identical stimulus
- For any DSE-era run: 1–3 point dry run first, verify CSV output, then full sweep; resume with `--skip-existing`; `--jobs 12` to limit CPU

## Context Economy Rules
- Do NOT re-read `paper_outline.md` or `dse/results/plots/round1/round1_analysis.md` unless working on paper narrative
- Do NOT re-run VTR on already-completed configs (check `dse/round1/`)
- Current work is tracked in:
  1. The "Active TODO Track" section below
  2. `rtl_flow/docs/FIDELITY_METHODOLOGY.md` for canonical methodology
- Everything under `archive/` is reference material: read on demand, never edit as live work

## Active TODO Track

**RTL/sim alignment on main — v2 clean-room flow** (opened 2026-05-01)

### Status (as of 2026-09-17)

**Done — committed**:
- Reorg `bd229f1` (Aug 27 work): azurelily de-submoduled, legacy archived, stale docs purged, CLAUDE.md/README rewritten.
- rtl_flow migration (Aug 29): all RTL assets consolidated under `rtl_flow/` (specs/ gen/ rtl/ tb/ smoke/ docs/ vtr/); paths fixed; generators read `rtl_flow/specs/*.json`.
- DPE behavior model primitives, Model Y FSM, module name `dpe` matching VTR arch XML contract.
- Stages 1A (V=1 H=1), 1B (V>1 H=1), 1C (V=1 H>1) validated.
- `softmax_study/` — COMPLETE (committed Aug 2026).
- **v2 clean-room Stage 1.2 — DONE** (2026-09-13, re-transcribed 2026-09-14): spec **v2.0 clean integer rewrite** (int8 weights/activations, exact integer MAC, integer ACAM forms + trunc8 low byte; fp32 amendment retired per advisor consultation); `v2/oracle/nldpe_ref.py` + `v2/sim/nldpe_sim.py` self-tests green; independent review passed (dual compare: full int32 `y` + byte stream, all 4 modes/geometries, measured cycles ≡ §5.3, P1/P10/P11 timeline invariants); `pack_weight_stream` (P23) and `dump_case` file contract round-trip verified. v2 tree at repo root `v2/`.

**Validation state (regression guards, re-run green after migration)**:
- `rtl_flow/smoke/run_dpe_smoke.py` — 52/52 PASS
- `rtl_flow/smoke/run_fc_smoke.py` — 13/13 PASS, fidelity reported not gated (Stages 1A+1B+1C)
- `rtl_flow/vtr/run_vtr_smoke.py` — 3/3 OK (NL only): bert_qkv_proj, lenet_fc1, bert_ffn1
- `python3 v2/oracle/nldpe_ref.py` — ALL PASS; `python3 v2/sim/nldpe_sim.py` — ALL PASS
- `python3 v2/smoke/gen_cases.py` — 48/48 default cases pass GATE 1 (sim ≡ oracle, run 2026-09-16)
- `python3 v2/smoke/run_dpe_rtl.py` — **GATE 2 green (2026-09-17)**: 48/48 default
  (256×256/256×512, M∈{1,2}, modes 0–3, identity/random/extremes) + 16/16 M-sweep
  (M∈{1,2,4,8}) + 48/48 micro-geometry (8×8, 40×40); Δ_impl = 0 on every geometry
- `python3 v2/smoke/legacy_witness.py` — independent witness on identical stimulus
  (mode 0, 256×256 M=1): legacy oracle MAC + bytes OK, frozen legacy RTL PASS
  (cycles reported: +2 NBA handoff, double-buffer split 52 vs 60)
- `python3 rtl_flow/smoke/run_dpe_smoke.py` — 52/52 PASS (frozen legacy, unchanged)

**Verification caveat (drives the active plan)**: functional truth for FC is a
one-byte pattern (`tb_fc.v` `expected_byte_fn`); cycle truth is the Task #98
formula authored in unremembered sessions. Bottom-up re-verification is in
progress — see "Direction (2026-08-29)" above.

### Stage 1.5 — COMPLETE (2026-09-17): hand-written v2 RTL + cross-check harness

- Stage 1.3a DONE: `v2/smoke/gen_cases.py` emits §9 stimulus classes for both
  geometries (256×256, 256×512); modes are a per-workload case axis (P27);
  `dump_case` round-trip verified on both.
- Stage 1.4a DONE: `v2/rtl/dpe_nldpe.v` skeleton; **interface freeze 2026-09-15**:
  ports identical to legacy (`rtl_flow/vtr/dpe_blackbox.v`, arch XML), outputs
  `reg`, widths `[DPE_BUF_WIDTH-1:0]`; parameter superset
  (`KERNEL_WIDTH, NUM_COLS, DPE_BUF_WIDTH, PRECISION, PIPELINE_DEPTH,
  ACAM_CYCLES, COMPUTE_CYCLES, ACAM_MODE`); machine-gated by
  `v2/smoke/check_interface.py`.
- Stage 1.5 harness READY (2026-09-15, validated against a /tmp stub):
  `v2/tb/tb_dpe_nldpe.v` + `v2/smoke/run_dpe_rtl.py` — dual compare (hierarchical
  int32 `y` + drained stream, P26), readiness I3, structural spans (P+1 / P+2),
  cycle formula + Δ_impl constancy + **T_steady calibration gate** per geometry
  (measured(M2)−measured(M1) == (M2−M1)·T_steady). Mode via `+MODE=` plusarg,
  held from the weight strobes (P27). TB probe contract (pinned in the RTL
  header): `state`, `acc`, `acam_fire`, `drain_valid`.
- Spec **v2.0.1** (2026-09-15): §5.2 accumulator freed by ACAM write (single-acc
  gate; §5.3 totals unchanged, sim updated); §6 F3 EXP clamp caveat (no mod-512
  shortcut); **P27** ACAM mode is workload configuration latched by the WEIGHT
  strobes (`mode_q`), never changed per pass. Sim + oracle self-tests green.
- **Verification chain pinned + GATE 1 wired (2026-09-16)**: trust flows
  downward only, spec arbitrates: `spec → oracle →(GATE 1: per case, inside
  dump_case — sim ≡ oracle bit-exact on int32 y, output bytes, §5.3 cycles;
  uncertified cases are never written) → sim →(GATE 2: RTL ≡ dumped expected
  bits, dual compare P26; cycles = §5.3 + invariant Δ_impl) → RTL`. The RTL is
  integer-only (no fp); all numerical modeling stays in oracle/sim — the same
  policy extends to later stages (softmax, DIMM).
- **Stage 1.5 COMPLETE (2026-09-17)**: `v2/rtl/dpe_nldpe.v` now implements all
  seven blocks as structure/event-driven control channels (load/ready, compute
  fires, ACAM gate, drain) — no cycle counters; COMPUTE_CYC = P+2 emerges
  structurally (P10). GATE 2: int32 `y` + drained byte stream bit-exact vs the
  GATE-1-certified sim; `measured = T_fill + (M−1)·T_steady` with **Δ_impl = 0**
  on every geometry (8×8, 40×40, 256×256, 256×512), M ∈ {1,2,4,8}, all four
  modes; T_steady steps 10/16/60/104 exact (incl. the compute-bound 8×8 corner).
  Legacy witness (`v2/smoke/legacy_witness.py`): legacy oracle MAC/bytes and the
  frozen legacy RTL agree with v2 on identical stimulus (mode 0; cadence is a
  documented differing witness, 52 vs 60).
- The harness `run_dpe_rtl.py` now uses the portable tempdir default (the
  previous hardcoded `/tmp/opencode` was not writable on all nodes).
- Next: Stage 2 — GEMM array (`v2/spec/gemm.md` v0.3 **frozen**): `gemm_ref`
  oracle **done**; `gemm_sim` (V×H `NldpeDpe` + byte tree + low-byte serializer)
  in progress → `gemm_top` (hand-written) → GATE 2 harness, incremental 1A→1D.

### Forward plan

| Rung | Scope | Gate |
|---|---|---|
| Stage 1 — primitives | v2 clean-room: spec v2.0.1 + NumPy oracle + sim; hand-written integer RTL; legacy witness | ✅ **DONE** — RTL ≡ sim ≡ oracle per case, Δ_impl = 0 (GATE 1+2); legacy witness PASS |
| Stage 2 — GEMM array (VMM/projection) | `v2/spec/gemm.md` v0.3 frozen (REGULAR tiles, byte tree, no post-reduction ACAM); `gemm_ref` done → `gemm_sim` → `gemm_top` (V×H `dpe`) → GATE 2; incremental 1A→1D | Your sign-off |
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
