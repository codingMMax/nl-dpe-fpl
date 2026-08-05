# memory.md — NL-DPE FPGA

Durable cross-session facts. Read at session start as background, not as ground
truth: if the repo contradicts an entry, trust the repo and correct the entry.

## Commands

- Unit regressions: `cd azurelily/IMC && python3 test_units.py` (no pytest in this env)
- Softmax operator isolation: `cd azurelily && python3 ../benchmarks/run_softmax_only.py`
  (flags: `--no-rescale` to reproduce the stale-constant behaviour)
- Both must be run with cwd inside `azurelily/` — `nn/` imports `nn.constant` by bare name.
- Seq-len sweep: `python3 benchmarks/run_seqlen_imc.py` — **requires**
  `benchmarks/vtr_runs_seqlen/`, which is absent as of 2026-08-05.

## Repo layout (three nested repos, easy to get wrong)

- `-FPL2026-NL-DPE-FPGA` — paper, `codingMMax/-FPL2026-NL-DPE-FPGA`.
  `nl-dpe-fpl/` is **gitignored** by it, not a submodule.
- `nl-dpe-fpl` — code, `codingMMax/nl-dpe-fpl`.
- `azurelily` — simulator, **submodule**, remote `leizhaocs/azurelily` (same
  author, push access confirmed). A local-only commit here makes the recorded
  submodule SHA unresolvable for anyone else — always push the branch.

## Invariants

- Dynamic energy is per-operation: **reported energy must never depend on Fmax.**
- A latency is built only from cycle counts and clock periods: **latency must
  never depend on an energy constant.**
- Clock period in ns is `1000 / f_MHz`. `cycles_to_ns()` is the only sanctioned
  conversion — do not hand-roll it.
- `_clb_reduction_energy_latency()` returns **(energy_pj, latency_ns)**.
- Softmax over attention scores runs on the S×S matrix: `Softmax_*_Layer(d=S, N=S)`.
- Per-op energy constants in `configs/*.json` describe a **256×256** array. Any
  other geometry must rescale via `dpe_specs(R, C)` — both `gemv_dse.patch_imc_config`
  and `benchmarks/run_seqlen_imc.run_bert_sim` now do.

## Settled decisions

- 2026-08-05 — Fused-softmax CLB subtract is batched over available CLBs
  (`ceil(cols/min(total_clb, cols))` cycles), mirroring how `norm_fpga` batches
  the DSP multiply. Chosen over both `cols/freq` (the original units error) and
  unbatched `cycles_to_ns(cols, freq)` (which would make the fused path lose to
  the DSP baseline). End-to-end impact measured at <0.03%, so the choice is not
  load-bearing.
- 2026-08-05 — `exp_fpga` models `floor(total_clb/8)` parallel exp units and
  charges energy per evaluation. Full N-way parallelism was rejected: it needs
  N×8 CLBs (8192 at N=1024) against ~533 reported by VTR. Impact on the baseline
  is ~+1.3% end-to-end, so this choice is likewise not load-bearing.
- 2026-08-05 — CNN benchmarks deliberately not re-measured after the fixes.
  ResNet-9/VGG-11 go through `run_gemm`, not `gemm_log`, so they get none of the
  gemm_log correction but all of the rescaling penalty (P-1 ~+4%, P-2 ~+18%
  energy). Expected to make CNN ratios modestly worse. Do not re-raise as an
  oversight; re-measure when the extension work starts.

## Pitfalls

- 2026-08-05 — The submodule was checked out 4 commits behind the recorded
  pointer (872f968 vs d834aa3), which is why `Scheduler.run_attention_head`
  appeared "missing". Run `git submodule update` before diagnosing anything.
- 2026-08-05 — `_run_softmax_norm` adds memory I/O that dominates the compute
  term, so a wrong (even negative) compute latency does not show up as a
  negative total. Test the composition, not the sign of the total.
- 2026-08-05 — The published CSVs were produced at d834aa3, which carries all
  six defects. `benchmarks/results/pre-fix/` holds them as a diff baseline.
  Fixed-vs-published at N=1024: AL/P-1 1.46 → 2.08, AL/P-2 1.69 → 2.15; latency
  and throughput unchanged to three significant figures.
