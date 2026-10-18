# v3 — plan

Status: 2026-10-14. F0 docs are written; F1 (copy), F2 (simulator core, crossbar and GEMM models, FC workload) and F3
(the DIMM, the online softmax and its accumulator, each with its model, both RTL twins, GATE 2 and a VTR smoke) are
done; F4 is next. `docs/simulator.md` is reviewed; `docs/experiment.md`, `docs/architecture.md` and this plan wait for
your review.

## Why

v2 grew three overlapping tracks (`v2/experiment/dse/`, `v2/experiment/dse_v2/`, `v2/simnext/`), each with its own
ledger, copies and notes. Its simulator and its RTL described the same blocks in different shapes. Its attention layer
needed glue logic and a central sequencer around the blocks.

v3 starts clean with three goals:
1. **One simulator that is the RTL's twin.** There is one behavior model per RTL module, with the module's name,
   parameters and ports. Values, cycles and counts come out of the same execution (`docs/simulator.md`).
2. **General, reusable RTL modules.** GEMM, tiled DIMM, and online softmax with its accumulator. Each is gated alone
   (values, events, a VTR smoke) before any composition, and nothing sits between them as glue
   (`docs/architecture.md`).
3. **Design docs coupled to the experiment.** The experiment, the architecture, the simulator and each module's spec
   live in `docs/` and `ip/*/spec.md`, and every gate result points back to them.

v2 was the frozen reference: nothing in v2 was edited by v3 work, and v2/ left the tree on 2026-10-17 (git tag
`v2-final`). v3 reused v2 files by copy with provenance, never by import.

## Rules

- **R1 one module, one folder:** `ip/<module>/{spec.md, rtl/sim/, rtl/vtr/, tb/, cases.py}` (the model: `simulator/models/`
  since the 2026-10-17 refactor). A structure
  check gates every module: the model's name, parameters, derived values, guards and ports must equal the RTL's.
- **R2 composition, not re-implementation:** the head model is a graph of module models, and the RTL top is the same
  graph of RTL modules. Modules connect only through their ports. There is no glue logic, and no module logic in a top
  or in a workload.
- **R3 copy with provenance:** reused v2 files are copied with a sha256 lock, with only path lines edited. A gate fails
  when a source or a copy drifts (the sources are read from the git tag `v2-final`).
- **R4 gate ladder per module:** spec → oracle → model → RTL → module GATE 2 on 64×64 (small corpus) → VTR smoke of
  the module alone (crossbar count == expected, no logic optimized away) → composition.
  - The model is written before its RTL.
  - A full iverilog launch needs your approval.
  - RTL verification runs on 64×64 only, plus one R ≠ C case for the softmax extent if you approve it.
  - VTR runs from the VTR twins, on your go.
- **R5 one ledger:** `LEDGER.md` holds status, log and needs-you. Run files and logs go in `v3/work/` (gitignored).
- **R6 docs first:** a parameter, a workload or a metric enters the code only after it is in `docs/`.
- **R7 one behavior model per module:** values, cycles and counts come from one execution. The oracle stays
  independent as GATE 1's witness. Closed forms are tests only (`docs/simulator.md` §1).
- **R8 weights are preloaded, never strobed** (user, 2026-10-14). A testbench loads the crossbar weights at zero
  simulated time. It reads the case's weight file with `$readmemh` and, after every reset, writes each `dpe`'s
  `weights` array and its mode `mode_q` through the hierarchy, as `tb_softmax_online_top.v` already does.
  - Identity crossbars get the identity in the same zero-time way.
  - No testbench spends R·C cycles per crossbar on `weight_in` / `weight_en` strobes.
  - No strobe case either (2026-10-14, user): the strobe path exists only in the sim twin (the VTR twin has bare
    `dpe` blocks), and a wrong preload fails the value checks.
  - Weight programming is untimed setup (WR_CYC is outside every cycle contract), so no measured number changes.

## Layout

```
v3/
  README.md  PLAN.md  LEDGER.md
  docs/        experiment.md     the DSE: question, design space, workloads, mapping space, metrics, procedure
               architecture.md   one attention head from general modules: ports, credits, micro-architecture, gates
               simulator.md      the behavior models, the engine, workloads, gates
               mapping_tiling.md parameter measurements on an earlier layer model (reference only)
  ip/          xbar/     the dpe primitive
               gemm/     gemm_top
               dimm/     the tiled DIMM (dimm_tile_top)
               softmax/  the online softmax (softmax_online_top) and its accumulator (softmax_online_acc)
               lib/      the 1W1R FIFO
               (each module: rtl/sim/ and rtl/vtr/ for the twins, tb/ for the testbench, cases.py / gen_cases.py
               = the case writer and corpus generator, runner; the behavior model is in simulator/models/)
  top/         att_ctrl (tiling controller) and att_head (wiring), sim and VTR twins
  oracle/      what the simulator is checked against: the numpy oracles (copies) and v2_kernels/ (the ten v2
               kernels: test references and case writers, never models)
  simulator/   the behavior-model package: core/ (engine, crossbar), models/ (module models, workloads), platform/,
               tests/
               (README.md = the map and the rules of the code)
  vtr/         everything that runs VTR: arch/ (the 12 arch XMLs), clock.sdc, run_vtr.py, smoke.py, vtr_dse.py
               (the DSE driver), dse_watch.py
  dse/         space.py (shapes, configs), att_rows.py, rank.py, breakdown.py, plot_dse.py, results_page.py;
               fc_snapshot/ (v2's FC rows); results/ (tracked: vtr_att_72.csv, att_priced_72.csv, breakdown.json, and
               adp/ = latency vs area + area-delay product, edap/ = the earlier energy-delay-area ranking)
  tools/       copies.py + copies.lock (copy lock), check_twins.py (twin check), lint_rtl.py, check_head_params.py,
               f1_check.py / f2_check.py (the gates), rtl_gate2.py (the GATE-2 runner), probe_site/ (isolation probe)
  work/        runs and logs (gitignored): vtr/{dse,smoke}, corpus, equivalence, dse (logs), tmp, per-module RTL runs
```

## Reuse from v2

| take | from | as |
|---|---|---|
| crossbar primitive (certified) | `v2/sim/rtl/dpe_nldpe.v`, `v2/experiment/rtl/dpe_blackbox.v`, `v2/sim/oracle/nldpe_ref.py`, `v2/sim/simulator/kernels/{nldpe_sim,pass_engine}.py`, `v2/sim/smoke/run_dpe_rtl.py` + TB | `ip/xbar/`. RTL unchanged; the model merges the integer math and the pass grammar into `Xbar`. |
| GEMM (certified) | `gemm_top.v` twins, `gemm_sim.py`, `gemm_ref.py`, case generator / runner / TB | `ip/gemm/`. RTL unchanged; one behavior model. |
| tiled DIMM core | `v2/{sim,experiment}/rtl/dimm_tile_top.v` (memories, window engine, farm, trees), `dimm_tile_sim.py`, its generators, runner, TB | `ip/dimm/`. Core kept; new operand ports, converters, loaders, slices, slots, job queue, narrowing (`architecture.md` §4.4). |
| online softmax | `softmax_online_top.v` twins, `softmax_online_sim.py`, `softmax_online_ref.py`, generators, runner, TB | `ip/softmax/`. Datapath kept; new input stage, credits, output framing, extent (`architecture.md` §4.5). |
| accumulator | `att_ocarry_top.v` + `att_oc_engine` twins, `ocarry_sim.py` | `ip/softmax/softmax_online_acc`. Datapath kept; stream inputs, $\alpha$ ring, credits, O reader (`architecture.md` §4.6). |
| FIFO | `att_fifo.v` | `ip/lib/` |
| oracles | `v2/sim/oracle/{nldpe_ref,gemm_ref,softmax_online_ref,attention_ref}.py` and their imports | `oracle/`. The head's contract is `attention_ref.head_stages`, stage `O8`. |
| FC DSE results | `v2/experiment/dse/results/{vtr_results.csv,dse_v2.csv,ranking_fc.json}` (FC rows) | `dse/fc_snapshot/`, no re-run (user 2026-10-13). Valid while the FC shapes, `gemm_top`, the prices and the area formula are unchanged. |
| platform and prices | `v2/sim/simulator/{configs/,platforms.py,cost.py}`; the energy counting rules of `v2/sim/spec/simulator.md` §4 | `simulator/platform/` (v3's own code since 2026-10-17; the two configs stay verbatim copies; only NL-DPE is used, decision 12). |
| VTR flow | `v2/experiment/arch/nl_dpe_*_auto.xml`, `v2/experiment/vtr/{run_vtr.py,clock.sdc,clean_vtr_intermediates.sh}`, `v2/experiment/dse/run_vtr_dse.py` | `vtr/` (arch XMLs in `vtr/arch/`) |
| gates | `v2/experiment/rtl/check_split.py`, `check_kernel.py` (no-runtime-products lint), `v2/simnext/tools/copies.py`, `v2/simnext/cocheck/structure.py` | `tools/` |
| ideas only | `v2/simnext/spec/operator_contract.md` (records, timing rules), `v2/experiment/dse/plan.md` §2–§3.1 (protocol, workloads, references) | written into `docs/` |

The v2 cycle-model functions (`cycle_model`, `tile_cycle_model`, `softmax_online_cycle_model`, `ocarry_cycle_model`)
are copied only as tests: each behavior model must reproduce them wherever its datapath is unchanged.

**As built in F1 (2026-10-14)** (`tools/copies.py` lists every copy and its path lines):
- Each module's RTL twins, testbench, case generator and runner are in `ip/<module>/`. The oracles are in `oracle/`.
  The kernels are in `oracle/v2_kernels/` (`sim/v2_kernels/` until 2026-10-17), keeping their v2 depth, so kernels
  and oracles needed no edit.
- **All ten v2 kernels are copied,** because the copied case writers import them: `gen_softmax_online_cases` imports
  `attention_sim` (→ `proj_sim`), and `softmax_online_sim` imports `softmax_sim` (→ `dimm_sim`). `attention_sim`,
  `proj_sim`, `dimm_sim` and `softmax_sim` are import dependencies only; no v3 code uses them as models.
- **The DPE runner's interface pre-check** is `tools/check_twins.py`, not v2's `check_interface.py`, which reads the
  legacy `rtl_flow/` files. The check that matters stays: `dpe` ports == `dpe_blackbox.v` == the arch XMLs.
- **Derived, not copied:**
  - `tools/copies.py` is modeled on `v2/simnext/tools/copies.py`;
  - `tools/check_twins.py` comes from `check_split.py` and the static check of `vtr_tops.py`, set to the
    per-module layout.
- **Later, when first used:**
  - `vtr_tops.py`, `run_vtr_dse.py` and `clean_vtr_intermediates.sh`, at the first VTR run (phase A). They are tied
    to the v2 kernel set and the earlier DSE rows, so they need adapting.
  - The products lint of `check_kernel.py`, at the first RTL edit (phase A).
  - `cocheck/structure.py`, once models exist (F2, F3).
- **Not copied:** the v2 verifier tables (`test_*.py`).

**Not taken:**
- the projection station `att_proj_top`: the projections use `gemm_top`;
- the old layer top `attention_top`; its layer model `attention_sim` is copied only as the import dependency above;
- the glue modules `att_proj_wr`, `att_score`, `att_pturn`, `att_ofeed` and `att_ocat`. They are not copied as
  modules; their logic only seeds the new sub-blocks that `architecture.md` names (the softmax input stage, the
  window-frame loader, the O reader);
- the `dse_v2` runtime-$n$ edits;
- every v2 ledger and progress note.

## Phases

Each module's model comes before its RTL (R4). The stops:
- your review after F0;
- your approval for every full iverilog launch;
- your go for every VTR run.

| phase | work | gate | simulator / RTL phases |
|---|---|---|---|
| F0 docs | `experiment.md`, `architecture.md`, `simulator.md` | complete against their open items | your review |
| F1 copy | the reuse table into the layout, with the provenance lock | every copied self-test passes unchanged, and so do the crossbar and GEMM corpora and the DIMM / softmax / accumulator corpora on their unchanged cores | — |
| F2 simulator core | records, graph, engine, stages, `Xbar`; the GEMM model and the FC workload | the crossbar's v2 corpus (values, pass events, P29); the 48 FC rows (cycles, energy) | S0, S1 |
| F3 modules | the DIMM, then the online softmax and its accumulator; per module the model first, then the RTL | GATE 1; the closed forms; module GATE 2 on 64×64 (D1, D2, S, A); a VTR smoke per module | S2, S3 / A, B, C |
| F4 head | `att_ctrl`, `att_head`, the head workload, the $P_{proj}$ copies | the bring-up and gate H on 64×64 (values == `head_stages` stage `O8`, events == the model); the R ≠ C softmax case (your approval); a VTR smoke of the head | S4 / D |
| F5 DSE | calibration VTR (18 runs), the 432-candidate sweep, the picks, VTR on the picks (216 runs), pricing, ranking | `experiment.md` §4.4–§6 | S5 |
| F6 later | full-model modules (FFN activation, LayerNorm, residual add) and their workloads | GATE 1 per module; model-only until their RTL exists | S6 |

S0–S6 are the phases of `simulator.md` §8; A–D are the RTL phases of `architecture.md` §8.

## Decisions (user, 2026-10-14)

1. **Folder:** `v3/` at the repo root.
2. **Sequence length:** build-time $n$.
3. **The attention row:** one head of textbook attention, i.e. the Q / K / V projections plus the attention core. No
   output projection, and no `NH` parameter.
4. **The v2 tracks** `dse_v2` step 7 and `simnext` are frozen (they left the tree with v2/, 2026-10-17).
5. **The mapping search** is adopted, including the $P_{proj}$ RTL work (`experiment.md` §4).
6. **General modules only:** GEMM, tiled DIMM, and online softmax with its accumulator.
   - `att_ctrl` is the only logic specific to attention.
   - There are no glue modules, and every module-to-module edge is push + credit (`architecture.md` §3).
7. **Projections:** `gemm_top`, unchanged. The LOG conversion sits inside the DIMM's operand ports.
8. **The o-carry** becomes the online softmax IP's second module, `softmax_online_acc`.
9. **Extent rule:** every pass loads and drains only the rows and columns in use, at most $I = \min(R, C)$ for a
   conversion, the softmax included.
10. **DIMM slices:** the farm is cut into slices of $X_S = 16$ crossbars, fixed. The DSE searches $N_E$.
11. **Simulator:** one behavior model per module. Values, cycles and counts come from one execution; the oracle stays
    independent; closed forms are tests only.
12. **Platform:** the platform layer stays. NL-DPE is the only platform implemented; the layer stays open for
    Azure-Lily later, but nothing for it is built now (`simulator.md` §4.7).

Open items: `experiment.md` §7 and `architecture.md` §10, collected in `LEDGER.md` "Needs you".
