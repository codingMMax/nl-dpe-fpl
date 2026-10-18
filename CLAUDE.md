# CLAUDE.md — NL-DPE FPGA Research Project

## Project in one line
NL-DPE FPGA hard-block research for a paper comparing NL-DPE with Azure-Lily. The live experiment (`v3/`): which crossbar
size (R × C) makes the best NL-DPE FPGA for FC layers and for one head of textbook attention. The DSE ran on
2026-10-15/16; results and analysis are in `v3/docs/experiment.md` §8–§9.

## Direction (pinned)
- **`v3/` is the live flow** (user, 2026-10-12: a fresh start). Resume at `v3/LEDGER.md` (status, decisions, needs-you,
  log); the folder map and the commands are in `v3/README.md`; the rules R1–R8 are in `v3/PLAN.md`. Current status lives
  only in the ledger, not in this file.
- **`v2/` was removed on 2026-10-17** (user: "now we are fully committed to v3"; commit `afb1caf`). Its last state, with
  the files that were never committed, is the git tag `v2-final`: `git show v2-final:<path>` reads one file, and
  `git checkout v2-final -- v2` restores the folder (the user runs that). v3 copied files from v2 with a sha256 lock
  (`v3/tools/copies.py` reads the sources from the tag) and never imported it.
  - The tag holds v2's charters under `v2/sim/spec/`: `dpe_nldpe.md` (crossbar), `gemm.md`, `dimm.md`,
    `softmax_online.md`, `simulator.md`, `attention.md`; and its history log `v2/docs/history.md`.
  - `v3/ip/xbar` and `v3/ip/gemm` are unchanged v2 modules, so their charters are v2's (`dpe_nldpe.md`, `gemm.md`).
- **Legacy, reference only; never edit as live work:** `rtl_flow/` (the v1 flow, frozen; `rtl_flow/CLAUDE.md` loads
  there), `archive/`, `softmax_study/` (complete), and the DSE-era `dse/`, `nl_dpe/`, `benchmarks/`, `gemv_dse.py`,
  `flexscore_dse.py` (Round 1/2: results as CSVs and plots; the raw VTR run folders were deleted).
- This file's text from before the v3 rewrite (it described v2 as live) is in git: `git show d44e71d:CLAUDE.md`.
- **Local only:** GitHub's `main` starts v3 as one commit (`df0c4e3`, on `86dc189`). The step-by-step history, the tag
  `v2-final` and the commits cited here and in `v3/` (`816807b`, `afb1caf`, `d44e71d`, ...) are on the local branch
  `att-kernel` in `/mnt/vault0/jiajunh5/nl-dpe-fpl`; they are not pushed (they carry 1.7 GB of generated test corpora).
  `tools/copies.py` needs that tag. Work on `main`.

## Rules (user decisions; do not re-raise)
- **Fix VTR problems by editing the RTL only** (hardline, 2026-10-05, repeated 2026-10-17). No Parmys / Yosys / ABC / VPR
  source or internals analysis (no gdb, ABC replays, netlist forensics, Yosys elaboration probes). The flow is a black
  box, read through pass/fail, Fmax, critical path and resources.
- **A full iverilog launch needs the user's approval.** RTL verification is small: 64×64 crossbars. A VTR run, or any
  long run, starts only on the user's explicit go; long runs go to the background and are not polled.
- **Gate ladder per module** (PLAN R4): spec → oracle → model → RTL → GATE 2 on 64×64 → a VTR smoke of the module alone
  (crossbar count == expected) → composition. The model is written before its RTL. The oracle stays independent, and the
  RTL is integer-only.
- **Twins** (R1): `ip/<module>/rtl/sim` and `rtl/vtr` are two hand-maintained files, with no `ifdef` single source and
  no generated copy. Shared edits go in both; `tools/check_twins.py` gates them. **Weights are preloaded, never
  strobed** (R8).
- **The synthesized netlist is the product** (2026-10-05): RTL must be VTR-realizable and of VTR quality (Fmax, area,
  Parmys/ABC viability); where it is not, fix the RTL, not the model. No runtime products (`a*b`, `idx % N`,
  `x[runtime*CONST +: W]`) in datapaths: they become DSPs and set Fmax (`tools/lint_rtl.py` gates it). Gate every VTR
  result on `wc` == the expected crossbar count.
- **A copied file** (listed in `tools/copies.py`) is edited only on the user's go; the edit takes it out of the table and
  the LEDGER log records the sha256 it started from.
- **Run files go in `v3/work/`** (gitignored), never on the root disk `/` (it once filled up and stopped every job):
  `export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/v3/work/tmp`. Paths passed to VTR subprocesses are absolute (VTR changes
  its working directory). Kill a process only by an explicit PID. `MSB_SA_Ready` is an output of the `dpe` block: each
  instance needs its own wire.
- **The agent cannot** `rm`, `mv`, `git rm`, `git checkout --` or `git push` (the user's permission rules), and never
  force-pushes. To delete or move: copy first, then write a script with a dry run and `--go` for the user to run (the
  v2 removal, commit `afb1caf`, went that way). The user pushes with `! git push ...`.
- **Simulator code** (`v3/simulator/README.md`): camelCase names, the RTL's parameter names kept (`N_E`, `T_W`), one
  `Params` / `Derived` pair per module, general and simple, no over-engineering; a check only where an RTL guard or a port
  contract needs one. The user takes over its development.
- **Docs first** (R6): a parameter, workload or metric enters code only after it is in `v3/docs/`. One ledger (R5). The
  attention row is a *head*: say "head", not "layer".
- Settled decisions: `v3/LEDGER.md` (Decisions) and `memory.md`. Recurring traps, such as Parmys memory inference:
  `memory.md` (Pitfalls).

## Session start
1. Read `memory.md` (background; the repo wins on a conflict), then `v3/LEDGER.md` (Status, Needs you).
2. `git status`, `git log --oneline -5`.
3. Commands: `v3/README.md`. Module facts: `v3/ip/<module>/spec.md`, `v3/top/spec.md`. The models:
   `v3/simulator/README.md`.

## Repo map
| Path | Role |
|------|------|
| `v3/` | **LIVE.** `docs/` (experiment, architecture, simulator, mapping_tiling; `architecture_page/`) · `ip/{xbar,gemm,dimm,softmax,lib}` (RTL per module: `rtl/{sim,vtr}`, `tb/`, `spec.md`) · `top/` (`att_ctrl`, `att_head`) · `oracle/` (numpy oracles; `v2_kernels/` = test references, never models) · `simulator/` (the behavior-model package) · `vtr/` (the generic flow: 12 arch XMLs, `clock.sdc`, `run_vtr.py`, `smoke.py`) · `dse/` (`README.md` + `run_all.sh`, `space.py`, `vtr_dse.py` = the VTR run of the builds and its CSV, the collecting and plotting scripts, `results/{adp,edap}`) · `tools/` (gates) · `work/` (runs and logs, gitignored) |
| `memory.md` | durable repo facts, settled decisions, pitfalls |
| `rtl_flow/` | legacy v1 flow, frozen reference |
| `softmax_study/` | safe-softmax RTL + VTR + energy study (complete) |
| `dse/`, `nl_dpe/`, `benchmarks/`, `gemv_dse.py`, `flexscore_dse.py`, other root scripts | DSE-era (Round 1/2) generators, runners, results |
| `paper/`, `paper_outline.md` | paper methodology, figures, scripts |
| `v1-v2/` | the v1-v2 comparison deliverable (`README.md`, `cycles.csv`, `logs/`) |
| `archive/` | reference only |

## The v3 experiment in numbers
- 12 configs: $R \in \{128, 256, 512, 1024\} \times C \in \{64, 128, 256\}$, $I = \min(R, C)$; `BUF` = 40 bits (a crossbar
  loads 5 B per cycle), `P` = 8.
- FC: four GEMMs $(K, N) \in \{(512,128), (2048,256), (256,512), (512,512)\}$, $V = \lceil K/R \rceil$,
  $H = \lceil N/C \rceil$, batch 1, used pass extent (P29). Its 48 rows use v2's VTR run (`v3/dse/results/fc_vtr_results.csv`, not
  re-run), priced by the v3 model into `fc_priced_results.csv`; v2's own rows are the regression reference in
  `v3/simulator/tests/ref/`.
- Attention: one head, six shapes (`v3/dse/space.py`), 72 builds. One fixed mapping: $P_{proj} = 1$, $N_E = N_{EP} = 2d$,
  $N_{EXP} = 2$, $N_{OC}$ by the floor rule (2–8), $B_{kv} = 16$, $X_S = 16$, $B_q = S$.
- Cycles: the simulator's compute cycles, weights preloaded, $T = t_{done} - t_{start}$ (pinned to the RTL by the gates).
  Energy: frozen per-operation prices (`simulator/platform/nl_dpe.json`). Area:
  $A = (\mathrm{clb} + \sum_{wc} W H + 2\,\mathrm{mem}) \times 66{,}117$ MWTA from VPR's counts (the routing-aware CLB tile
  is 2239 µm²; 1 MWTA = 0.033864 µm²). Fmax: VTR, the mean of seeds {11, 13, 17}, channel width 300, `vtr/clock.sdc`
  (I/O-excluding).
- Results (2026-10-16): the FC front is 512×128 → 256×128 → 256×64 → 128×64 (a real area/latency trade-off); attention's
  front is 128×64 alone; across the families the front is 256×128, 256×64, 128×64, where 128×64 and 256×64 tie and 256×64
  is the balanced choice. The large configs are never competitive (`v3/docs/experiment.md` §8–§9).

## Commands (from the repo root; the full table is in `v3/README.md`)
```
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/v3/work/tmp
python3 v3/tools/f2_check.py                        # the 8 simulator tests, ~25 s
python3 v3/simulator/tests/checkEquivalence.py      # the simulator against the frozen pre-refactor outputs
python3 v3/tools/copies.py; python3 v3/tools/check_twins.py; python3 v3/tools/lint_rtl.py; python3 v3/tools/check_head_params.py
python3 v3/dse/vtr_dse.py --self-test               # the VTR driver with fake tools; the real run takes hours: user's go only
bash v3/dse/run_all.sh                              # priced rows -> rankings -> breakdown -> figures -> pages (12 s; v3/dse/README.md)
```

## Context economy
- Do not re-read `paper_outline.md` unless working on the paper narrative.
- Do not re-run VTR on finished builds: results are in `v3/dse/results/attn_vtr_results.csv` and `v3/work/vtr/dse/<build>_<hash12>/`
  (the hash is the circuit's sha256, so an RTL edit gives every build a new folder).
- `archive/`, `rtl_flow/` and the `v2-final` tag are reference: read on demand.

## Authoritative docs
- `v3/docs/experiment.md` (question, design space, metrics, procedure, results), `v3/docs/architecture.md` (the head from
  general modules), `v3/docs/simulator.md` (the behavior models), `v3/ip/*/spec.md`, `v3/top/spec.md`.
- Legacy: `rtl_flow/docs/FIDELITY_METHODOLOGY.md`, `rtl_flow/docs/CYCLE_ACCOUNTING.md`, `rtl_flow/docs/FC_RTL_PLAN.md`,
  `paper/methodology/dpe_pipeline_model.md`, `paper/methodology/attention_dimm_mapping.md`,
  `softmax_study/SOFTMAX_STUDY.md`.
