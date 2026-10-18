# v3 simulator (refactored, 2026-10-17 →)

The behavior-model simulator of the v3 attention head and FC workloads, rewritten for readability from the tree
committed as git `816807b` (the restore point: the models were in `ip/*/model.py` and `top/model.py`, the core in
`sim/`). Same semantics: one model per RTL module, timed dataflow, values + cycles + counts from one execution
(`docs/simulator.md`). The rewrite was proven equal to the old simulator on the DSE's own inputs (below); the old tree
is gone.

## Layout

```text
simulator/
  core/      cycleMath (ceilDiv, clog2, isPowerOfTwo, readyAt) · errors · records (Window, Credit, PortSpec,
             PortEvent, Step, Counts, Resources, Report) · module (Module, Need, checkRules) · graph · engine ·
             stages (Serial, Buffer, Repack, Delay, Tree, Map, Fifo, Memory) · environment (Source, Sink) ·
             crossbar (Geometry, Crossbar, CrossbarGroup, outputStage: the `dpe` primitive and the pass grammar)
  models/    the module models: gemm (gemm_top) · dimmTile (dimm_tile_top) · softmaxOnline (softmax_online_top) ·
             softmaxAccumulator (softmax_online_acc) · attentionController (att_ctrl);
             the workloads that wire them: fc · attentionHead
  platform/  platform (Platform; nl_dpe.json, azure_lily.json) · prices (energy = counts x prices) ·
             area (crossbars, memory blocks, VTR) · rows (a run priced into a DSE row) ·
             breakdown (latency by phase, energy by role, area by component)
  tests/     testCore, testCrossbar, testGemm, testFcRows, testDimmTile, testSoftmaxOnline,
             testSoftmaxAccumulator, testAttentionHead · checkEquivalence · corpus (the v2 corpora helpers)
```

The RTL-facing files stay beside the RTL: `ip/<module>/{spec.md, rtl/, tb/, cases.py, gen_cases.py}` and
`top/{cases,gen_cases}.py`. A case writer imports its model from this package. `oracle/v2_kernels/` (the ten v2
kernels) is the closed-form witness the tests compare against, next to the numpy oracles; it is never a model.

## Rules of the code

- **Names.** Python attributes, methods, functions, locals and file names are camelCase; classes are CapWords.
  Three things keep their hardware names: RTL parameters and localparams (`N_E`, `T_W`, `LOAD_CYC`, the fields of a
  module's `Params` / `Derived` dataclasses), the named latencies of the specs (`DEC_LAT`, `SER_LAT`), and the spec's
  event and process names used as dictionary keys (`load_start`, `t_go`, `"a_in"`). PortSpec keeps the letters of
  `architecture.md` §3 (`E, Q, W, L, CAP`). The DSE row columns (`e_pass_pj`, `T_cycles`) are the results' data
  format, not Python names.
- **A module** = a `Params` dataclass (RTL names, RTL defaults) + a `Derived` dataclass from `derive(p)` +
  `checkDomain(p)` (the RTL's elaboration guard as a table of rules) + `ports()` + `processes()` + `need()` + one
  method per process, listed in `STEP_METHODS`. Values live in a separate `*Math` class per module (`None` in
  timing-only mode), so a step reads as timing + emit.
- **Checks kept** (about a third of the old ones): the engine's wiring checks at graph build and its protocol checks
  at delivery (order, overlap, a unit before its credit, a credit before the last word), the RTL domain guards, and
  the port contracts of the specs (`ContractError`: a converter overrun, rows out of order, a full register).
  **Dropped:** type and shape guards on parameters, and checks that repeat what the engine already guarantees.
- **Independence.** The model stack (`core`, `models`) imports no oracle, no v2 kernel and nothing from `platform/`
  (tested); `platform/` imports the models, never the other way round.

## Use it

The pattern is always: build a graph, `run` it, read the `Report`, price it. From the repository root
(`import sys; sys.path.insert(0, "v3")`); `withValues=False` is timing only (cycles, events, counts; ~0.03 s per head).

```python
from simulator.core.engine import run
from simulator.platform import area, prices
from simulator.platform.platform import Platform
from simulator.models.fc import fc, result
from simulator.models.attentionHead import attentionHead, headParams

# an FC layer: M rows of X[K] through W[K, N] on R x C crossbars
p = Platform("nl_dpe", rows=256, cols=128)                          # prices and geometry (platform/nl_dpe.json)
rep = run(fc(512, 128, 256, 128, M=4, platform=p, withValues=False))
rep.cycles, rep.totalResources.nCrossbars                           # 270 cycles, 2 crossbars
prices.energy(rep.totalCounts, p)["e_total_pj"]                     # 336.99 pJ (e_pass_pj, e_adds_pj, ... apart)

# one attention head: n = 128 tokens, d_model = 128, d = 64, on 128 x 128 crossbars
hp = headParams(128, 128, 64, 128, 128)                             # the DSE's fixed mapping; keywords change it
g, parts = attentionHead(hp, withValues=False)                      # parts: gemm, qk, sm, pv, acc, ctrl, ...
rep = run(g)
T = parts["ctrl"].done - parts["ctrl"].tStart                       # 9405 cycles
rep.totalResources.nCrossbars                                       # 278
area.predictedMemBlocks(rep.resources.values(), 128, 128)           # 376 memory blocks
headParams(128, 128, 64, 128, 128, P_PROJ=2)                        # a what-if: 2 projection copies -> 7229 cycles
```

Report: `cycles`, `steps[instance]` (every step's times and named events), `events` (the port events in the testbench's
format: `rep.eventLines()`), `counts[instance]` and `totalCounts` (passes, CLB adds, memory bytes), `resources[instance]`
(crossbars, memories), `roles[instance]` (gemm / qk / sm / pv / acc / ctrl, for `platform/breakdown.py`).
With values on (`withValues=True`, weights and inputs given), the same run also computes the outputs
(`fc.result(sink, ...)`, the head's O8 windows); `top/cases.py` `runCase` runs a head that way and compares every stream
with the oracle (GATE 1). Area and Fmax come from VTR (`dse/results/vtr_att_72.csv`); `area.withVtr(cycles, R, C, row)`
turns cycles + a VTR row into area (mm²) and latency (ns).

**How the DSE calls it** (`platform/rows.py` is the one entry for rows):

| script | call | uses |
|---|---|---|
| `dse/att_rows.py` | `attentionRow(shape, n, d_model, d, R, C, vtr)` per build, 72 times | `headParams` → `attentionHead` → `run` → cycles = `ctrl.done - ctrl.tStart`; `prices.energy(totalCounts)`; crossbars and memory blocks; with the VTR row: area, latency, EDAP → `dse/results/att_priced_72.csv` |
| `dse/rank.py` | `fcRow(R, C, (M, K, N), platform, vtr)` for the 48 FC rows; reads the priced attention CSV | `fc` → `run` → the same pricing; the FC snapshot's VTR table |
| `vtr/vtr_dse.py` | `Build`: `headParams` + `attentionHead` per build | the expected crossbars (`wc`) and memory list for the elaboration check and the VPR check; `area.predictedMemBlocks` |
| `dse/breakdown.py` | `attentionHead` + `run`, `analysis.breakdown.projectionEnd` | where the projection stream ends in the events |
| `tools/check_head_params.py` | `headParams` + `attentionHead` | the model's module parameters == the RTL's, for the 72 builds |
| `ip/*/cases.py`, `top/cases.py` | module graphs with `Source` / `Sink`, `run`, GATE 1 | the testbench files and `events.txt` that GATE 2 compares with the RTL |

## Tests

```text
python3 v3/tools/f2_check.py                              the eight tests below under the isolation probe (25 s)
python3 v3/simulator/tests/test<Name>.py                  one of them
python3 v3/simulator/tests/checkEquivalence.py            the refactored simulator against the frozen old outputs
```

`tests/test*.py`: the v2 corpora and closed forms, GATE 1 against the oracles, the spec rules, timing-only, random step
orders, mutants. `checkEquivalence.py` reproduces the outputs frozen from `816807b` in `work/equivalence/ref/`
(`manifest.json` holds the sha256 of every file and the commit): every FC and attention DSE build's Report (cycles,
every event line, every step, counts, resources), the DSE rows, and the 8 corpora the case writers produce (75 cases,
byte-identical files).

## History: the refactor's gates (all green, 2026-10-17)

| step | content | gate |
|---|---|---|
| 0 | restore point git `816807b`; reference frozen (666 files: 120 Reports, 8 corpora with 75 cases, 2 row tables) | `work/equivalence/ref/manifest.json` |
| 1 | core, crossbar, platform, prices, area | testCore; testCrossbar (258 v2 corpus cases, 216 closed, 200 gated, 5/5 mutants) |
| 2 | gemm, fc workload, rows.fcRow | testGemm (160 corpus, 900 closed, 3/3 mutants); testFcRows 48/48 (max float difference 2.2e-16); 48 FC Reports and rows equal |
| 3 | dimmTile, its case writer and generator | testDimmTile (243 GATE 1, 30 closed, 9/9 mutants, 8 Yosys memory lists); corpus 16/16 files equal; random differential 150 cases: 276 runs equal, 24 rejected by both, 0 differ |
| 4 | softmaxOnline, softmaxAccumulator, their case writers and generators | testSoftmaxOnline (210 GATE 1, 466 closed, 17/17); testSoftmaxAccumulator (290 GATE 1, 35 closed, 14/14); 6 corpora (48 cases) equal; differential 300 cases: 600 runs equal, 0 differ |
| 5 | attentionController, attentionHead, rows.attentionRow, analysis.breakdown, the head case writer and generator | testAttentionHead (14 heads, 432 builds in domain, 6/6); 72 attention Reports and rows equal; head corpora 11 cases equal; differential 40 heads: 80 runs equal, 0 differ |
| 6 | the switch of the tools, the DSE scripts and the VTR scripts to the package | copies 48/48; check_head_params 72/72; `att_priced_72.csv` regenerated byte-identical; `breakdown.json` identical; vtr_dse self-test; F2 ALL PASS |

The random differential (`differentialRandom.py`, 640 random cases through both simulators, values on and off) and the
freezing script needed the old tree; both were removed with it.
