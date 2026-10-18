# v3 — NL-DPE crossbar-size DSE: FC and one attention head

Status 2026-10-17: the DSE is done (VTR of the 72 attention builds, the latency vs area fronts, the cross-family
ranking, the FC vs attention analysis: `docs/experiment.md` §8–§9). The simulator is the `simulator/` package, proven
equal to the pre-refactor one. `LEDGER.md` has the status, the decisions, what waits for you, and the log.

## Where things are (one question per folder)

| folder | what it holds |
|---|---|
| `docs/` | `experiment.md` (question, design space, procedure, results), `architecture.md` (the head), `simulator.md` (the models), `mapping_tiling.md` (reference); `architecture_page/` (sources of the architecture page) |
| `ip/` `top/` | what we build: RTL per module (`xbar gemm dimm softmax lib`) and the head (`top` = `att_ctrl` + `att_head`): `spec.md`, `rtl/{sim,vtr}`, `tb/`, `cases.py`, `gen_cases.py` |
| `oracle/` | what the simulator is checked against: the numpy oracles and `v2_kernels/` (never imported by the models) |
| `simulator/` | the behavior models (`core` engine + crossbar, `models` module models + workloads, `platform` prices, area, DSE rows, `tests`); `README.md` = layout, rules, how to call it |
| `vtr/` | the generic VTR flow: `arch/` (12 arch XMLs), `clock.sdc`, `run_vtr.py`, `smoke.py` (one module's or the head's VTR smoke) |
| `dse/` | `README.md` (the pipeline, step by step, and `run_all.sh`), `space.py` (the 6 shapes x 12 configs, defined once), `fc_rows.py` and `attn_rows.py` (the priced tables), `data.py` (their loader), `vtr_dse.py` (the VTR run of the 72 builds and its CSV; `dse_watch.py` watches a run) and the scripts that turn results into tables and figures; `results/` (tracked): per family the VTR table and the priced table (`fc_vtr_results.csv`, `fc_priced_results.csv`, `attn_vtr_results.csv`, `attn_priced_results.csv`) and `breakdown.json`, then one folder per view: `adp/` (the current one: latency vs area, area-delay product) and `edap/` (the earlier energy-delay-area ranking), each with its JSON, figures and page |
| `tools/` | static checks and gates: copy lock, RTL twins, lint, head params, the F1 / F2 gates, the GATE-2 runner |
| `work/` | scratch, gitignored: `vtr/{dse,smoke}` (the VTR runs, 276 GB), `corpus`, `equivalence`, `dse` (logs), `tmp`, per-module RTL runs |

Design docs: `PLAN.md` (rules R1–R8, layout, reuse from v2), `LEDGER.md`, `docs/`. v3 copied files from v2 with a
sha256 lock (`tools/copies.py`) and never imported it. v2/ itself left the tree on 2026-10-17: its last state is the git
tag `v2-final` (`git checkout v2-final -- v2` restores it), and the copy lock reads its sources from that tag. Run files
go in `work/`.

## Commands (from the repository root; `PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/v3/work/tmp`)

| to do | command |
|---|---|
| simulator tests (8 tests, 25 s) | `python3 v3/tools/f2_check.py` (or one: `python3 v3/simulator/tests/testGemm.py`) |
| refactor equivalence | `python3 v3/simulator/tests/checkEquivalence.py` |
| static gates | `python3 v3/tools/copies.py` · `check_twins.py` · `lint_rtl.py` · `check_head_params.py` |
| F1 gate (copies, twins, self-tests, corpora) | `python3 v3/tools/f1_check.py --steps probe,lock,twins,selftests,regen`; its `rtl` step simulates (needs approval) |
| a module's corpus, then GATE 2 | `python3 v3/ip/<module>/gen_cases.py` → `python3 v3/ip/<module>/tb/run_*_rtl.py --stimuli DIR` (iverilog: needs approval) |
| VTR of the 72 builds | `python3 v3/dse/vtr_dse.py` (hours; `--elab-only`, `--build SHAPE@RxC`, `--aggregate` = rebuild `results/attn_vtr_results.csv` from the run folders, `--self-test`); `python3 v3/dse/dse_watch.py --pid PID` |
| the DSE results, end to end | `bash v3/dse/run_all.sh`: priced rows → rankings → breakdown → figures → pages; each step and what it writes in `v3/dse/README.md` |
| architecture page | `python3 v3/docs/architecture_page/page_data.py v3/work/arch_page/builds.json` → `build.py` |
