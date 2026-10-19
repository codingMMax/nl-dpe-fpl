# v3 ledger

The single ledger of v3 (PLAN R5): status, decisions, needs-you, log. Run files and logs go in `v3/work/`
(gitignored).

## Status

- **Phase:** F1, F2 and F3 are done (F3 ran one module at a time).
  - Module 1, the tiled DIMM (`dimm_tile_top`), is **closed** (2026-10-14): spec, model, both RTL twins, GATE 2 at
    64×64 (16/16, plus 5/5 bring-up), VTR smoke at 128×128.
  - Module 2, the online softmax (`softmax_online_top`), is **validated** (2026-10-14): spec, model, both RTL twins,
    GATE 2 at 64×64 (18/18, plus 5/5 bring-up), VTR smoke at 128×128. Open: one $R \ne C$ case (Needs you 3).
  - Module 3, the accumulator (`softmax_online_acc`, phase C), is **validated** (2026-10-14): spec, model, both RTL
    twins, GATE 2 at 64×64 (19/19, plus 5/5 bring-up), VTR smoke at 128×128 (12/12 crossbars, 156.3 MHz after the
    Fmax fix). 2026-10-15: its own $O_8$ memory (the O read-out leaves the cycles; Needs you 17).
  - F3 is complete: the DIMM, the online softmax and its accumulator.
  - **F4, the head** (`att_ctrl` + `att_head`, renamed from `att_layer` 2026-10-15) is **closed** (2026-10-15):
    spec `top/spec.md`, the engine's netlist features, the controller model and the head workload (ALL PASS), both
    RTL twins, bring-up 5/5 and gate H 6/6 at 64×64 (every event, the cycles and the values equal the model's), the
    softmax R ≠ C case 1/1, VTR smoke at 128×128 (278/278 crossbars, 116.5 MHz); `att_ctrl` rtl-pinned. PV B credit
    4 rows per copy with the X order rule. No further functional RTL simulation (you).
  - F5 is done (2026-10-16): VTR of the 72 builds, pricing, the latency vs area fronts per family and the
    area-delay product cross score (`experiment.md` §8), and the analysis of why attention and FC differ (§9).
  - **Now:** the simulator refactor (2026-10-17, `v3/simulator/`), the folder reorganization and the removal of v2/
    (the same day, `afb1caf`) are done and committed. Every gate is green on the tree without v2/ (log; `README.md`
    has the map and the commands). `main` is pushed (Needs you 23). Next: the sensitivity-study steps (Needs
    you 19: S-b, S-c).
  - `experiment.md`, `architecture.md` and `PLAN.md` still **wait for your review** (`simulator.md` is done).
  - Git: work on `main` (pushed 2026-10-17 as `df0c4e3`, one v3-only commit on `86dc189`). The step-by-step history
    is the local branch `att-kernel` (not pushed: it carries 1.7 GB of generated test corpora); the restore point
    `816807b` and the tag `v2-final` = `db76d1b` (v2 as it was on disk) are on it, local only.
- **Docs:**
  - `PLAN.md`: rules R1–R8, layout, reuse from v2 (with the F1 as-built notes), phases F0–F6, decisions;
  - `docs/experiment.md`: question, design space, workloads, mapping space, metrics, procedure, shape sources;
  - `docs/architecture.md`: one attention head from general modules (computation stages, ports and credits, module
    micro-architecture, gates, phases A–E);
  - `docs/simulator.md`: one behavior model per module (engine, workloads, gates, phases S0–S6; S0–S1 as built);
  - `docs/mapping_tiling.md`: parameter measurements on an earlier layer model, reference only.
- **Gates:** `python3 v3/tools/copies.py` (copy lock), `python3 v3/tools/check_twins.py` (RTL twins),
  `python3 v3/tools/lint_rtl.py` (no runtime products), `python3 v3/tools/f1_check.py` (F1),
  `python3 v3/tools/f2_check.py` (F2 = the eight simulator tests `v3/simulator/tests/test*.py`);
  `python3 v3/simulator/tests/checkEquivalence.py` (the refactored simulator against the frozen
  pre-refactor outputs); per module `ip/<module>/tb/run_*_rtl.py` (GATE 2 through `tools/rtl_gate2.py`: values,
  every event, total cycles); runs and logs in `v3/work/`.
- **R8 (weight preload) in the copied testbenches:** `tb_softmax_online_top.v` already preloads, and the DIMM's
  rewritten `tb_dimm_tile_top.v` preloads (F3). The other three still strobe and change when their module is next
  worked on; each edit takes the file out of `tools/copies.py`'s table (its v2 sha256 stays in `tools/copies.lock`):
  - `ip/softmax/tb/tb_att_ocarry_top.v`: phase C (`g_x[*].u_lg` LOG, `g_x[*].u_ex` EXP);
  - `ip/xbar/tb/tb_dpe_nldpe.v` (`dut`, mode from `+MODE`) and `ip/gemm/tb/tb_gemm_top.v`
    (`gen_v[*].gen_h[*].u_dpe`, REGULAR): before their next RTL run.
- **Next:** F5 on your go: the model's cycles and energy for the 72 builds, VTR on them (Needs you 16), pricing and
  rankings; before it, the head Fmax work (15) and the O read-out checks (17). Any full iverilog launch waits for your
  approval.

## Decisions

| date | decision | by |
|---|---|---|
| 2026-10-12 | v3 is a fresh start. v2 is the frozen reference; v3 copies from it with a sha256 lock and never imports it. | user |
| 2026-10-13 | Six attention shapes, 2 per application including 2 jet tagging; $n$ taken to a power of two (`experiment.md` §3.2, App. A). | user |
| 2026-10-13 | The earlier attention DSE is not a valid experiment and is not re-run. The FC results (48 VTR + 48 priced rows) are reused as the v3 FC snapshot. | user |
| 2026-10-14 | Folder `v3/`; build-time $n$; `dse_v2` step 7 and `simnext` frozen. | user |
| 2026-10-14 | Mapping search adopted: $P_{proj}$ {1, 2, 4} × $N_E$ $\{d, 2d\}$, including the $P_{proj}$ RTL work. | user |
| 2026-10-14 | The attention row is one head of textbook attention: the Q / K / V projections plus the attention core. No output projection; no `NH` parameter. | user |
| 2026-10-14 | General modules only: GEMM, tiled DIMM, and online softmax with its accumulator. `att_ctrl` is the only attention-specific logic; there are no glue modules; every module-to-module edge is push + credit. | user |
| 2026-10-14 | The projections use `gemm_top`, unchanged. The LOG conversion sits in the DIMM's operand ports. | user (recommendation accepted) |
| 2026-10-14 | The o-carry becomes the online softmax IP's second module, `softmax_online_acc`. | user (recommendation accepted) |
| 2026-10-14 | Extent rule: every pass loads and drains only the rows and columns in use, at most $I = \min(R, C)$ for a conversion, the softmax included. | user |
| 2026-10-14 | The DIMM farm is cut into slices of $X_S = 16$ crossbars, fixed. The DSE searches $N_E$. | user |
| 2026-10-14 | Simulator: one behavior model per module. Values, cycles and counts come from one execution; the oracle stays independent; closed forms are tests only. | user |
| 2026-10-14 | `simulator.md` reviewed and accepted, with its speed targets. The platform layer stays: NL-DPE is the only platform implemented, and the layer stays open for Azure-Lily later (nothing built for it now). | user |
| 2026-10-14 | Testbenches preload the crossbar weights at zero simulated time (`$readmemh`, then a hierarchical write of each `dpe`'s `weights` and `mode_q` after every reset); no cycle-by-cycle weight strobes, in any case (`PLAN.md` R8). | user |
| 2026-10-14 | Simulator code is written in a general way and kept as simple as possible, without over-engineering: module models are wirings of shared stages (`sim/stages.py`). You review it and take control of its future development. | user |
| 2026-10-14 | The DIMM closes on GATE 2 (64×64) and the VTR smoke (128×128). The RTL gates have no mutants, no credit-never case, no weight-strobe case and no runs at the 432 candidates' parameter sets; the model covers those builds (`architecture.md` §7). | user (recommendation accepted) |
| 2026-10-15 | The PV B credit (edge E3) is 4 rows per GEMM copy (was 2), with an order rule on X: the copies' bursts start in token order. The PV DIMM keeps its 2 row registers per copy; no DIMM RTL change (4 registers per copy was the alternative). The model raises on a violation. | user |
| 2026-10-15 | "Layer" renamed to "head" in v3 (`att_head`, gate H, `head_params`, ...): the row is one attention head. | user |
| 2026-10-15 | No further functional RTL simulation for the head: the bring-up and gate H (64×64) cover the other heads; the DSE shapes and configs rely on the model (pinned by these gates) and on the VTR runs. | user |
| 2026-10-15 | F4 closes on the bring-up (5) plus gate H (6 heads: late and slow X, a late start, $N_E = d$, $V = 3$, $d = 64$), the softmax R ≠ C case (128×64) and the VTR smoke; the 7 corpus cases that repeated the bring-up are dropped. Then stop and discuss $N_{EP}$ and the tiling / mapping parameters before F5. | user (recommendation accepted) |
| 2026-10-15 | **One fixed mapping for every attention shape, no search:** $P_{proj} = 4$, $N_E = d$, $N_{EP} = 2d$, $N_{EXP} = 2$, $N_{OC}$ by the floor rule, $B_{kv} = 16$, $X_S = 16$, $B_q = S$ (`experiment.md` §4.2). Rule: every stage keeps up with its producer and consumer, with as few resources as that needs. No candidate sweep, pick rule or calibration VTR; the DSE is the 72 builds; no new functional runs for them. | user |
| 2026-10-15 | **Cycles = compute cycles:** from `start` to the last run's last $O_8$ written into the accumulator's `e_mem`; the final O read-out is not counted (earlier runs' read-outs stay inside: they delay the next run). Superseded the same day by the next row. | user |
| 2026-10-15 | **The fixed mapping uses $P_{proj} = 1$ and $N_E = 2d$**, the configuration the head smoke synthesized (was $P_{proj} = 4$, $N_E = d$); the rest is unchanged ($N_{EP} = 2d$, $N_{EXP} = 2$, $N_{OC}$ by the floor rule, $B_{kv} = 16$, $X_S = 16$, $B_q = S$). These are the `head_params` defaults. | user |
| 2026-10-15 | The Fmax work ends at fix 1 (no more VTR smokes or seed checks before the DSE). The final head RTL (fixed DIMM, no O reader) is the DSE's RTL; its smoke gave 116.4 MHz on seed 11 (129.6 on the circuit before the O-reader removal, same DIMM path class), and the DSE's 3-seed mean measures it. | user |
| 2026-10-15 | An RTL change is validated by one 64×64 head run (correctness) and the 128×128 head VTR smoke (synthesis). No module-alone smokes or module corpora. | user |
| 2026-10-15 | `gemm_top`'s reduction tree is held as one wide register in both twins (stage $s$, slot $k$ at bits $32(s \cdot NSLOT + k)$), not a 2-D array: with $V > 1$ Yosys collected the array into one memory with a port per access, in the 21 DSE builds with $d_m > R$ (found by the VTR driver's elaboration check). Validated by the 64×64 head run with $V = 3$ (`H_v3_p2`). | user |
| 2026-10-16 | **Across the families: the area-delay product.** Each config's ADP score per family (area × latency ÷ the largest on each workload, geometric mean over the family's workloads); x = FC, y = attention; the Pareto front of that plane. Replaces the geometric mean of the separate area and latency scores (the two families' fronts hold different configs). | user |
| 2026-10-16 | **Latency vs area only, replacing EDAP:** per workload, area ÷ the largest area and latency ÷ the highest latency (lower is better); geometric mean over a family's workloads; across the families, the geometric mean of the two family scores; the result is the Pareto front per family and across the families. | user |
| 2026-10-16 | **The raw data only, no seed resampling:** the rankings, the cross score and the figures use each build's Fmax as measured (the mean of its 3 seeds); no 2,000-draw resampling, no seed bands or rank-1 shares. | user |
| 2026-10-16 | **Cross-family ranking:** each config is the point $(1 - G_{FC}, 1 - G_{ATT})$ (1 − EDAP score per family); the origin is the ideal and the config closest to it (distance $d$) is the best. Replaces v2's deferred combined ranking (its max-min pick would be 256×128). | user |
| 2026-10-15 | **No O read-out in the cycles, in the simulator and in the RTL:** the read-out is set by the 40-bit port, not by the crossbar size or the mapping. Implementation (you, the same day, after the first version's extra memory): **no O reader**; the head outputs the final $O_8$ values, `o_v` (1 bit) + `o_d` ($N_{OC}$ × 40 bits), as the normalize jobs drain them, and the O projection reads them (`acc_spec.md` §10 item 7). `run_done` pulses at the run's end, so the head's `done` = the last run's $t_{o8} + 2$, and $T = t_{done} - t_{start}$ (`experiment.md` §5). | user |
| 2026-10-17 | **v2/ is untracked and removed from the tree** ("now we are fully committed to v3"). Its last state is the tag `v2-final`; the copy lock (`tools/copies.py`) reads its sources from the tag; the F1 gate no longer runs v2. | user |
| 2026-10-17 | **`CLAUDE.md` is rewritten for v3** (112 lines; the v2-era text is commit `d44e71d`). | user |
| 2026-10-17 | **One ranking notation for both views:** per workload, score = value ÷ the worst config's value (1 = the worst, lower is better); $G_X$ = their geometric mean over a family's workloads, $X$ = A, D, E, ADP, EDAP. The EDAP view changes from best ÷ EDAP (higher is better) to this; its ranks and every Pareto front stay; its cross-family distance is now from the origin of $(G_{EDAP}^{FC}, G_{EDAP}^{ATT})$ (`experiment.md` §5). | user |
| 2026-10-17 | **The pass price: the whole array fires every time, only the rows in use are read, outputs full.** The $R \times C$ cells, the ACAM on every column, the output buffer and the output drain are charged in full; the input-buffer bytes and the input port words follow `ROWS_USED` (P29). The earlier full-array price on both sides overcharged every conversion pass on $R > C$ arrays and every GEMM pass with $K < R$ (`simulator/platform/prices.py`). | user |
| 2026-10-17 | **Area and energy splits per config** (the (b) and (c) panels): attention = the geometric mean of each share over its 6 shapes; FC = the share of the total summed over its 4 shapes (FC builds without a memory block or a CLB add would make a geometric mean 0). Energy parts: DPE passes, CLB adds + compares, the passes' block-RAM port traffic, memory bytes. | user (recommendation accepted) |
| 2026-10-18 | **`dse/results/` holds both families, one VTR table and one priced table each:** `fc_vtr_results.csv` (the 48 FC rows of v2's VTR table, verbatim), `fc_priced_results.csv` (the v3 model on them, `dse/fc_rows.py`), `attn_vtr_results.csv` (`dse/vtr_dse.py`), `attn_priced_results.csv` (`dse/attn_rows.py`). `fc_snapshot/` is merged away; v2's own rows and ranking are the regression reference in `simulator/tests/ref/`. `rank.py` and `breakdown.py` read the two priced tables and never run the simulator. The FC shapes are `space.py FC_SHAPES`. | user |
| 2026-10-18 | **The simulator drains all C columns on every pass (your own edit of `core/crossbar.py`: `OUTPUT_CYC` from C, `colsUsed` removed), the RTL keeps the early-stopping P29 drain.** Energy and timing now both charge the full output side. The model is therefore not RTL-pinned for passes with COLS_USED < C (3–8 % of the attention passes at C = 128 and the larger C = 256 arrays, all of them at 128×256). Earlier the same day you had kept the P29 drain; this supersedes it. | user |

## Needs you

1. **Review the F0 docs:** `experiment.md`, `architecture.md` and `PLAN.md` (`simulator.md` is done).
2. (resolved 2026-10-15: $N_{EP} = 2d$ in the fixed mapping; the measurements are in the log.)
3. (done 2026-10-15: the softmax R ≠ C case `sm_rc_128x64`, 128×64 crossbars, 1/1 PASS: 1,462 events, 4,421 cycles.)
4. **Cleanup:** run `bash v3/tools/cleanup_f3.sh`. It removes the obsolete copied scripts and o-carry files of F3, the
   five pre-rename files (`att_layer.v` ×2, `tb_att_layer.v`, `run_layer_rtl.py`, `twin_baseline/att_layer.diff`)
   and the old corpus `work/corpus/layer_64x64`.
5. (done 2026-10-14: the softmax GATE 2 corpus, 18/18.)
6. (dropped 2026-10-15: the calibration VTR; there is no pick rule.)
7. (resolved 2026-10-15: VTR on the 72 builds of the fixed mapping.)
8. (resolved 2026-10-15: the final O read-out is outside the cycles; measured 3–22 % of the full cycles, median 11 %.
   Refined the same day: no O read-out at all, item 17.)
9. **Optional:** the FC shapes (Round-1 vs the six models' FFNs).
10. (done 2026-10-14: the accumulator's GATE 2 corpus, 19/19.)
11. (done 2026-10-14: the accumulator's Fmax fixed first, 103.3 → 156.3 MHz.)
12. (done 2026-10-15: gate H 6/6 and the bring-up 5/5 on the final RTL.)
13. (done 2026-10-15: the PV B credit is 4 rows per copy with the X order rule; Decisions. Measurements in the log.)
14. (done 2026-10-15: "layer" renamed to "head"; the old files are in the cleanup script, item 4.)
15. **Head Fmax (2026-10-15, your request): fix 1 gave 129.6 MHz** (from 116.5; log), the QK DIMM's own limit. The final RTL (no O reader) gave 116.4 MHz on seed 11 with the same DIMM path class. Closed: no more VTR runs (you). the VTR smoke gave 116.5 MHz, below every module alone
    (132–156 MHz). Fix 1 (read addresses from slice registers, no cycle change) is in both DIMM twins; the QK DIMM
    alone gave 130.6 MHz, and the head smoke is running. That head smoke predates the $O_8$ memory (17), so the final
    head numbers need one more smoke on the final RTL. The head bring-up 5/5 has since run on the fix-1 DIMM (item 17),
    every event equal; no DIMM corpus run is needed (you). A next limit is expected near the DIMM's own 132 MHz (the
    engine's take-now path).
16. (done 2026-10-16: VTR, pricing, figures; results `experiment.md` §8.) **VTR of the 72 builds (F5), the launch plan in `experiment.md` §6** (updated 2026-10-15, your request: 36
    workers, plus the figures stage): 36 workers, memory-guarded (per-process reserves in a memory budget, syntheses
    60 s apart, at most 12 VPRs); about 300–400 CPU-hours, 10–14 h wall [estimate]; then pricing and the v2-style
    figures (geomean ranking, Pareto front). The driver `vtr/vtr_dse.py` is written and its test ran (log); with the
    tree fixed (item 18) all 72 builds pass the elaboration check. **Launched 2026-10-15 20:30, done 2026-10-16 15:31** (log):
    all 216 seeds; priced and ranked, figures done 2026-10-16 (log).
18. (done 2026-10-15, your go: "fix the tree, run the 64x64 head check": the tree is one wide register in both
    twins; the 64×64 head check with $V = 3$ PASS, every event; the elaboration check 72 / 72. Log.) The finding was:
    **The projections' reduction tree with $V > 1$ (2026-10-15, found by the driver test's elaboration check):** in
    the 21 builds with $d_m > R$ (ViT-B/16 and BERT-base at $R \le 512$, DeiT-T at $R = 128$), Yosys infers
    `gemm_top`'s `tree` array (the registered pairwise adder stages) as a memory with many ports in each of the three
    GEMMs: 24 × 32 bits with 5 read / 12 write ports at $V = 2$, 64 × 32 with 37 / 64 at $V = 3$, 128 × 32 with
    149 / 128 at $V = 6$. The simulator has no memory there; everything else in those builds' memory lists is equal.
    By v2's recorded rule Parmys does not abort on it (more than one port on both sides), but builds it from RAM
    blocks (v2's FC rows with $V > 1$ each show 1 memory block, those with $V = 1$ none). Proposal: in both
    `gemm_top` twins, hold the tree as one wide register vector (each stage's slots as 32-bit slices) instead of a 2-D
    array, so it stays registers; only the $V > 1$ branch changes, but any edit changes every build's circuit hash, so
    it goes in before the launch. `gemm_top` is the locked v2 copy (your 2026-10-14 decision: unchanged), so this
    needs your go; validation by your rule: one 64×64 head run with $V > 1$ (gate H's $V = 3$ head) and the VTR of a
    $V > 1$ build (the DSE's own runs, elaboration first). The FC snapshot's $V > 1$ rows carry that 1 block.
17. **O read-out out of the cycles (2026-10-15, your request): done** as no O reader; E9 = `o_v` + `o_d`
    ($N_{OC}$ × 40 bits) straight from the normalize drain (`acc_spec.md` §10 item 7). No RTL simulation after this
    edit (you); the head VTR smoke on it is in the log. `tb_att_head.v` now takes the E9 windows (2026-10-15, for the
    tree's head check: `o_win.txt`, `+NOWIN`; the first RTL run of E9 passed, log); `tb_softmax_online_acc.v` still
    checks the old row format and must take them before its next run. The architecture page's accumulator tab and its 16 links into
    `softmax_online_acc.v` are stale (rebuilt on request).

20. (done 2026-10-17, by you: the old simulator tree removed with `tools/retire_old_tree.sh`, the refactor committed as
    `c73c4d8`.)
21. (done 2026-10-17, by you: the cleanup script ran and the new layout is committed, `02b3cd9` and `1043cad`.)
22. (done 2026-10-17: the old folders and `reorganize_cleanup.sh` are gone, committed.)
24. (done 2026-10-18: the simulator drains all C columns, your edit, with `colsUsed` restored as the window's valid
    width; the test references follow, your "fix it and update the test references"; the RTL stays. Log of
    2026-10-18.) **The drain rule** Trial: one line in `core/crossbar.py`
    (`OUTPUT_CYC` from C instead of COLS_USED; reverted), the tables in `work/tmp/fulldrain/`. Energy and pass counts
    unchanged on every row (the price already charges the full output side). FC cycles unchanged on all 48 rows (one
    pass per row; the serializer finishes with the useful words). Attention: 22 of 72 builds change, all at C ≥ 128:
    none at C = 64; 4 of 24 at C = 128 (max +13.8 %, JetFormer-150P, a 13-cycle load against a 26-cycle drain); 18 of
    24 at C = 256 (median +26 %; 128×256 +26 to +83 %, the other C = 256 configs up to +62 %). Rankings: both latency vs
    area fronts unchanged; the ADP tie flips to 256×64 first (d 0.478 against 128×64 0.483), its front unchanged; EDAP:
    512×64 and 128×128 swap places 3 and 4 in attention, FC unchanged. To adopt: (1) `dpe_nldpe.v` (sim twin): the drain
    counter from NUM_COLS, `drain_valid` / `data_out` for the first ceil(8 COLS_USED / 40) words only, `dpe_done` after
    the full drain; the VTR twin is a black box, so the 72 VTR builds stay valid; (2) the DIMM converter contract
    (`ip/dimm/spec.md` CONV_IN_LAT row, `dimmTile.py`): a converter of extent K on C columns is paced by the full drain,
    so the port's minimum row spacing grows; the standalone generator overruns today under the rule ("a row at 48 needs
    its burst at 49, the crossbar is ready at 51"), the 72 heads do not (the GEMM row period grows alongside); (3) the
    xbar and gemm case writers still import the v2 kernels, which encode the early-stopping drain: move them onto the v3
    models, or compare pass events against the kernel run at cols_used = C; (4) GATE 2 on iverilog: the crossbar P29
    corpus (162), the GEMM P29 corpus (144), the DIMM 64×64 corpus, the softmax R ≠ C case, the 64×64 head bring-up and
    gate H; (5) re-freeze the equivalence reference (56 of 72 attention reports, 22 rows differ by design), re-price,
    pages, `experiment.md` §8–§9.3. About a session. Order on a go: model + specs, tests, RTL, runs, re-price.

19. **The simulator for sensitivity studies** (your idea, 2026-10-16; 2026-10-17: the refactor delivered S-a as
    `simulator/platform/breakdown.py` + `Report.countsByRole()` and S-d as `simulator/README.md`; S-b and S-c
    remain, on your go). Today the model gives cycles, energy per
    module and term, crossbars and memory blocks for any parameter set, but: (a) CLB count and Fmax exist only from
    VTR, so an area or latency what-if needs hours of VTR; (b) the assumptions of `experiment.md` §9.6 are hard-wired
    (the full-array pass price in `sim/platform/prices.py`, one projection copy, the 40-bit ports); (c) breakdowns are
    not first-class (the projection share in §9 came from parsing the events); (d) each study is a one-off script.
    Proposal, each step on your go and kept simple (one model per module, no new framework):
    - S-a: one breakdown function on the Report: latency by phase (projection stream, core, tail), energy by module
      and term, area by component; `att_rows.py` and `breakdown.py` use it.
    - S-b: the assumptions as named switches with today's values as defaults (pass energy: full array or used
      extent; port width; projection copies; the mapping knobs), plus one sweep script that varies one switch and
      writes a table.
    - S-c: a CLB and Fmax estimate fitted on the 72 VTR builds, its error against VTR reported, for exploration only
      (VTR stays the ground truth).
    - S-d: a readability pass: one short page per module model (what it models, its parameters, its gates) and the
      map of the simulator's layout in `simulator.md`.

23. (done 2026-10-17: choice (b) with the generated files dropped; `main` = `df0c4e3`, pushed by you; log.)
    **Push to `main` (2026-10-17, your request).** The remote `main` is still `86dc189` (4 Oct); no push has
    arrived. A plain push of `att-kernel` sends 128 commits, 8461 objects, 1.78 GB raw (a 545 MB pack, measured 2026-10-17; the
    largest file is 23 MB, so GitHub accepts it). The size is one commit: `e93ce0e`, the first of the 4 commits that
    local `main` has beyond `origin/main`, added 1.7 GB of generated test corpora under `v2/sim/smoke/` because
    `.gitignore` still named the old paths. They are out of the tree now (v2/ is removed) but stay in the history, so
    every clone would download them. Choices: (a) push `att-kernel:main` as it is (full history, ~545 MB, in the remote
    for good); (b) one v3-only commit on top of `origin/main` (210 files, 7.3 MB raw) on a local branch `main-v3`, the
    commits stay local; (c) rewrite the history without the corpora (changes the SHAs the docs cite: `816807b`,
    `c73c4d8`, `d515d8c`). Uncommitted in your tree: the deletions of `sweep.csv`, `sweep.log` and
    `softmax-regular-vs-online.log` (HEAD still has them; a snapshot built from HEAD would too), the untracked
    `arch.info` (a VPR leftover, 143 KB) and `rtl_flow/CLAUDE.md`.

## Log

- 2026-10-12: v3 created: `README.md`, `PLAN.md` draft, `docs/mapping_tiling.md`, `.gitignore`.
- 2026-10-13: six-shape set recorded (`v2/experiment/dse/plan.md` §3.1, now moved to `docs/experiment.md` App. A).
- 2026-10-14: F0 docs written (first drafts).
  - Correction to an earlier note: `gemm_top` has had the P29 extent since 2026-10-08 (commit d515d8c: default corpus
    16/16, P29 corpus 144/144), so no `gemm_top` P29 work is pending.
  - Finding: the PV farm could not be halved inside the then-current tile RTL (its multi-channel write guard,
    $N_{EP}/B_{kv}$ a multiple of $N_{EXP}\,d/B_{kv}$). The 2026-10-12 farm measurement had halved both farms in the
    model only.
- 2026-10-14: architecture redesigned with you (`docs/architecture.md`, rewritten as new work).
  - Changes:
    - the head without an output projection;
    - general modules with row-major ports and push + credit edges;
    - projections on `gemm_top`, with the LOG conversion in the DIMM;
    - the o-carry as `softmax_online_acc`;
    - the extent rule;
    - 16-crossbar DIMM slices.
  - Slices [derived, RTL estimate]:
    - the worst fan-out drops from 16,384 to 2,688 sinks per control bit;
    - the worst per-slice operand write rate is 0.76 words per cycle, so $P_{proj} = 4$ is no longer throttled;
    - with one write port per slice, the PV farm halving guard above no longer applies: $N_{EP} = d$ fits at
      $I \ge 128$.
  - $B_{kv} = 16$ re-derived as the only value legal on every candidate: $B_{kv} \le d$, and the DIMM guard
    $NG \le W_K$.
- 2026-10-14: `docs/experiment.md` updated to the new architecture:
  - the row unit and the parameter list;
  - the crossbar formula $P_{proj}(3V + 3)$ + engine;
  - the reasons for $B_{kv}$ and $N_{EP}$;
  - the procedure.

  `docs/simulator.md` written. `PLAN.md` and `README.md` updated: rules R1–R7, layout, reuse, phases F0–F6,
  decisions.
- 2026-10-14: `simulator.md` reviewed by you. Recorded:
  - the platform decision: `simulator.md` §4.2 (the pass formula in `BUF` and `P`, the output stage taken from the
    platform), §4.7 and §10; `PLAN.md` decision 12 and its reuse row;
  - the speed targets, accepted with the document;
  - two stale lines removed (`PLAN.md`'s last line, `simulator.md` §10 item 3).
- 2026-10-14: **F1 done** (copy with provenance; the check agreed with you: no full iverilog launch).
  - 70 files copied into the layout: 59 byte for byte, 11 with path lines only (`tools/copies.py`, sha256 lock
    `tools/copies.lock`). All ten v2 kernels are in `sim/v2_kernels/` because the copied case writers import them
    (`PLAN.md`, as-built notes).
  - Run `f1_run1` (`v3/work/f1/f1_run1/summary.txt`):
    - copy lock 70/70; twin gate (`tools/check_twins.py`) ALL PASS, the twin diffs equal v2's recorded baselines;
    - copied self-tests 19/19 PASS;
    - 7/7 default corpora regenerated by the v3 copies byte-identical to the v2 generators' (1,196 GATE-1 cases);
    - RTL, one 64×64 case per harness, 6/6 PASS (dpe, gemm, tiled DIMM, online softmax, o-carry, FIFO).
  - The isolation probe raised 7 false flags (it read `+VDIR=` arguments as repo-root paths). Fixed, with
    regression cases in its self-check. F1 stands on this run (no re-run) as you moved on to F2.
- 2026-10-14: **F2 done** (simulator S0 + S1, Python only). Run `f2_run1` (`v3/work/f2/f2_run1/summary.txt`), ALL PASS:
  - core: a hand-computed source → relay → sink chain on push + credit edges; 8 random step orders identical;
    composition == flattened; dropped credit → deadlock; send without credit, order, early credit, unconnected and
    mismatched ports rejected; stages at their edges; the model stack loads no oracle, v2 kernel or pricing module;
  - `Xbar`: the v2 crossbar corpora, 258 cases (96 default + 162 P29): values, cycles and every pass event equal;
    216 closed-form sweeps == `nldpe_sim.cycle_model`; 200 readiness sequences == `pass_engine`; mutants 5/5;
  - `gemm_top`: the v2 GEMM corpora, 160 cases (16 + 144 P29): bytes, the S probe, cycles and every row event equal;
    900 points == `gemm_sim.cycle_model` (both extents); structure == the RTL (parameters, port signals, derived
    values, extent guard); timing-only and random order identical; mutants 3/3;
  - the 48 FC rows of the v2 snapshot: every column equal (15 integer exact, 8 float within 2.2e-16 relative);
  - isolation: 0 accesses outside `v3/`.
  - Finding: v2's closed form uses $\lceil rows \cdot P / BUF \rceil$ load cycles, the RTL
    $\lceil 8\,rows / BUF \rceil$; equal at $P = 8$ (`simulator.md` §8, as built).
- 2026-10-14: rule R8 (you): testbenches preload the weights, never strobe them (`PLAN.md` R8, `architecture.md` §7,
  `simulator.md` §4.8). Checked against the copied testbenches: every one measures from after programming (dpe and
  GEMM from the first activation word, the tile from each `start`, the o-carry from its origin `tc = 0`), so the
  preload changes no measured number. No testbench is edited and nothing is run yet; the four that still strobe are
  listed under Status.
- 2026-10-14: **F3, module 1: the tiled DIMM** (`dimm_tile_top`). You: one module at a time, RTL simulation on
  64×64, VTR smoke on 128×128.
  - Spec `ip/dimm/spec.md`: ports, named latencies, decision rules, §10 refinements (`architecture.md` §4.4 updated
    to match).
  - Model `ip/dimm/model.py` (`DimmTile`), case writer `ip/dimm/cases.py`, tests `ip/dimm/test_model.py` ALL PASS
    (`v3/work/f3/logs/dimm_model.log`):
    - GATE 1 on 243 cases;
    - 30 configurations equal v2's `tile_cycle_model` (idle = v2 `occ` + 1);
    - the 432 DSE candidates are inside the domain (worst window-frame write rate 0.76);
    - timing-only and random order identical; mutants 9/9.
  - RTL: sim twin `ip/dimm/rtl/sim/dimm_tile_top.v`, VTR twin `ip/dimm/rtl/vtr/dimm_tile_top.v`.
    - Twin gate ALL PASS (baseline `tools/twin_baseline/dimm_tile.diff`: 11 hunks, weight path only).
    - `tools/lint_rtl.py`: no runtime products in the DIMM.
  - Testbench `ip/dimm/tb/tb_dimm_tile_top.v` (R8 preload) and runner `ip/dimm/tb/run_dimm_rtl.py`: bring-up 5/5 PASS
    on 64×64, every port event equal to the model's (`v3/work/f3/bringup/`).
  - Corpus: `ip/dimm/gen_cases.py` wrote 16 GATE-1-certified cases (10 QK + 6 PV) to `v3/work/corpus/dimm_64x64`. The
    RTL run is not launched; it waits for you.
  - Forked from v2: the sim and VTR RTL, the testbench and the twin baseline. They are out of the copy lock; their v2
    sha256 start points are in `tools/copies.lock`. v2's two DIMM case scripts are obsolete (`tools/cleanup_f3.sh`,
    user-run). Copy gate 64/64 OK + 6 forked.
  - VTR smoke (`v3/vtr/smoke.py`): 128×128, seed 11, route channel width 300, `clock.sdc`. The point is BERT-Tiny:
    $n = 128$, $d = 64$, $N_E = N_{EP} = 128$, $P_{proj} = 1$.

    | point | `dpe` at elab / expected | `wc` placed / expected | Fmax (MHz) | CLB | memory | DSP | wall |
    |---|---|---|---|---|---|---|---|
    | QK | 130 / 130 | 130 / 130 | 132.3 | 8,991 | 88 | 0 | 66 min (ABC 59) |
    | PV | 129 / 129 | 129 / 129 | 140.6 | 5,090 | 216 | 0 | 10 min |

    - Nothing is optimized away. All 16 inferred memories per point have legal shapes.
    - QK critical path, 7.56 ns: a slice leader's `MSB_SA_Ready` → `leader_ready` → the controller's feed decision
      (v2 `dimm_tile_ctrl`) → a register. It has 3 LUT levels and 6.95 ns of routing. The 100 worst paths lie in
      7.56–6.62 ns and start at many registers.
    - PV critical path, 7.11 ns: one register → two adders in series → 5 LUT levels → a register. 84 of the 100
      worst paths start at that register; it is not yet mapped back to its RTL signal.
    - Runs: `v3/work/vtr/dimm_tile_top_{qk,pv}_bert_tiny_128/` (`result.json`, `s11/`).
  - GATE 2 on the corpus (your approval): **16/16 PASS** in 194 s, 4 jobs (`v3/work/f3/gate2_run1/`, summary
    `v3/work/f3/logs/gate2_run1.out`).
    - Every case: the testbench reports 0 errors (values, memory images, slices in lock step, converter overrun,
      protocol), and every event line equals the model's (68–270 events per case).
    - `DimmTile.BASIS` = "rtl-pinned".
- 2026-10-14: **DIMM closed** (you accepted the recommendation).
  - Removed from the RTL gates (`architecture.md` §7–§9, `PLAN.md` R8): the RTL mutants, the credit-never case, the
    runs at the 432 candidates' parameter sets, and the weight-strobe case. D1 stays as the memory-image check inside
    every D2 case. Reasons in `architecture.md` §7.
  - The strobe case was my addition when R8 was written down, not your words; the R8 decision row is corrected.
  - The §9 fan-out risk did not show up: QK's path #1 runs from a slice's crossbar 0 to that slice's own controller
    (`dimm_tile_ctrl` sits inside `dimm_slice`), not across the farm.
  - Not done (optional): a few-second check that the RTL runner reports a changed event line; the runner's event
    comparison has not yet been seen failing on the RTL.
- 2026-10-14: **F3, module 2: the online softmax** (`softmax_online_top`; your go).
  - Spec `ip/softmax/spec.md`: ports, named latencies, the tile gate, the packer, the run control; §10 refinements of
    `architecture.md` §4.5 (E6 one row per window; carry passes $S \times S$; the gates in the input reader with one
    idle cycle between tiles; `done` after the last E7 word; `rd_m` dropped). `architecture.md` §1, §3.3, §4.5 updated.
  - Model `ip/softmax/model.py` (`SoftmaxOnline`; processes arm, tile, blk, run), case writer `ip/softmax/cases.py`,
    tests `ip/softmax/test_model.py` ALL PASS (`v3/work/f3/logs/softmax_model.log`):
    - GATE 1 on 210 cases (every stage and every sent value == `softmax_online_ref`, every run);
    - the core fed v2's stream schedule == v2 `softmax_online_cycle_model` (used extent) on 466 configurations;
    - every event recomputed from spec §5, each gate term binding somewhere;
    - the 144 DSE softmax instances inside the domain; values off and random order identical; mutants 17/17.
  - RTL, both twins: the input stage (FIFOs, reader, packer), the tile gate and credit counters, the output framing,
    the E7 read-out, P29 extents. The 7 runtime-indexed products of v2 rewritten as shifts and one-hot muxes;
    `tools/lint_rtl.py` now lints the softmax twin strictly: 0 products. Twin gate ALL PASS, its self-test ALL PASS
    (baseline `tools/twin_baseline/softmax_online.diff` re-recorded: only the sim twin's `dpe` line changed).
  - Testbench `ip/softmax/tb/tb_softmax_online_top.v` (R8 preload; replays the model's sources, reactive sinks,
    EV lines) and runner `ip/softmax/tb/run_softmax_rtl.py`.
  - Bring-up on 64×64 (`v3/work/f3/softmax/bringup`, `bringup_rtl2`): **5/5 PASS**, every event equal to the model.
    It found one bug of mine: the row-group base `(p_ra >> RL) << (RL + 3)` was computed at the row index's width and
    overflowed to 0 (every event still matched; the log P values of row group 1 on did not). Fixed in both twins.
  - VTR smoke (`v3/vtr/smoke.py --module softmax --point bert_tiny_128`): 128×128, seed 11, BERT-Tiny ($S = N = 128$,
    `S_CH` = 2, $N_{EXP} = 2$):

    | `dpe` at elab / expected | `wc` placed / expected | Fmax (MHz) | CLB | memory | DSP | wall |
    |---|---|---|---|---|---|---|
    | 4 / 4 | 4 / 4 | 140.3 | 1,365 | 14 | 0 | 6 min |

    - 12 inferred memories, all 1R1W: 2 input FIFOs (512 × 40), 10 store lanes (416 × 8).
    - Path #1, 7.13 ns: one register → 5 LUT levels → a subtractor → 2 LUT levels → a register; all 100 worst paths
      start at that register (one cone, not mapped to its RTL signal). v2's twin measured 121–142 MHz.
    - A first launch ran on the twin before the bug fix and was stopped (explicit PIDs).
  - Corpus: `ip/softmax/gen_cases.py` wrote 18 GATE-1-certified cases to `v3/work/corpus/softmax_64x64`.
  - GATE 2 on the corpus (your approval): **18/18 PASS** in 15 s, 4 jobs (`v3/work/f3/softmax/gate2_run1/`, summary
    `v3/work/f3/logs/softmax_gate2_run1.out`). Every case: 0 testbench errors (values on E5 / E6 / E7, no gap inside a
    window, word counts, one done per run, credits granted before every tile and start), and every event line equals
    the model's (55–1,574 events per case). `SoftmaxOnline.BASIS` = "rtl-pinned".
- 2026-10-14: **Simulator refactor** (your request: the models were too loose; general, simple code).
  - Your question: were the cycle counts checked against the RTL? Yes, through the event comparison (each event is
    a cycle stamp, and gapless windows fix every word), but the totals were never shown. From the GATE 2 logs: 34/34
    cases with equal total cycles (`v3/work/f3/logs/cycles_model_vs_rtl.txt`).
  - Cause of the looseness: the GEMM model was built from the stages, but the DIMM and softmax models hand-coded their
    timing (about a dozen inline `max` rules each) and used none.
  - `sim/stages.py` gained `ready` (a decision), `Serial` (one unit at a time), `Buffer` (entries in rotation, overrun
    check) and `Repack` (a byte stream re-cut into words), with edge tests in `sim/tests/test_core.py`.
  - `ip/dimm/model.py`: converters, writers (`Serial`), staging buffers and slots (`Buffer`), the farm; the job
    decision by `ready`. `ip/softmax/model.py`: the reader (`Serial`), the packer (`Repack`), the gates by `ready`.
  - GATE 2 runner: one shared `tools/rtl_gate2.py` (self-test ALL PASS); a case passes on the testbench's PASS, equal
    events and equal total cycles. `case.json` now carries the plusargs and every port's words per window. The two
    module runners are thin wrappers.
  - Checks (logs `v3/work/f3/refactor/`):
    - both model suites ALL PASS (DIMM mutants 9/9, softmax 17/17); core, crossbar, GEMM and FC tests ALL PASS;
    - the DIMM corpus (16), the softmax corpus (18) and the softmax bring-up (5) regenerated: every file byte for
      byte equal to the sets the RTL ran on, except `case.json` gaining the two keys;
    - the stored RTL logs re-checked against them (`--recheck`, no simulation): 16/16, 18/18, 5/5 with equal events
      and equal cycles.
  - The DIMM bring-up sets (`v3/work/f3/bringup/`) came from an unsaved inline script and keep the old `case.json`;
    the corpus covers their configurations.

  - Forked from v2 (out of the copy lock; start points in `tools/copies.lock`): both RTL twins, the testbench, the twin
    baseline; v2's two softmax harness scripts are obsolete (`tools/cleanup_f3.sh`). Copy gate 58/58 OK + 12 forked.
- 2026-10-14: **F3, module 3: the accumulator** (`softmax_online_acc`; your go).
  - Spec `ip/softmax/acc_spec.md`: ports, named latencies, the job rules, the O reader. Its §10 refinements of
    `architecture.md` §4.6:
    1. the $\alpha$ ring is in arrival order across runs and is not cleared at a run's end;
    2. the O reader starts after the run's last $O_8$ word, one row per $\lceil d/5 \rceil + 1$ cycles;
    3. the table credits keep v2's timing;
    4. `run_done` comes one cycle after the run's last row;
    5. one FIFO write enable per channel.

    `architecture.md` §4.6 and `simulator.md` §4.4 and the as-built note were updated to match.
  - Model `ip/softmax/acc_model.py` (`SoftmaxOnlineAcc`): the LOG and EXP groups (`XbarGroup`), the O reader (`Serial`),
    decisions by `ready`; processes blk and run. Case writer `acc_cases.py`, corpus generator `acc_gen_cases.py`, tests
    `test_acc_model.py` ALL PASS (`v3/work/f3/acc/logs/acc_model.log`):
    - GATE 1 on 290 cases (every stage == `attention_ref.ocarry_stage_values` / `epilogue`; every sent $O_8$ row);
    - fed v2's station inputs, every job event equals v2 `ocarry_cycle_model` on 35 schedules (decide, load_start,
      msb_fire, drains; `op_free` == `p_cr`, `la_free` == `la_cr`; the run end);
    - every event recomputed from spec §5, every readiness term and each input credit binding;
    - the 72 DSE instances inside the domain ($N_{EP} = 2d$, $N_{OC}$ by the floor rule);
    - values off and random order identical; mutants 16/16.
  - RTL, both twins `ip/softmax/rtl/{sim,vtr}/softmax_online_acc.v`, from v2's `att_ocarry_top.v`. The groups,
    engines, slot memories and FIFOs are unchanged. New: the E6 / E7 stream inputs into the $\alpha$ ring and the $\log l$
    table, the credits, the O reader, `run_done`. Twin gate ALL PASS (baseline `tools/twin_baseline/softmax_online_acc.diff`:
    7 hunks, weight path only, as v2's); lint 0 products.
  - Testbench `ip/softmax/tb/tb_softmax_online_acc.v` (R8 preload; replays the sources; checks the $O_8$ words,
    `run_done` cycles, credits before units, the probes) and runner `ip/softmax/tb/run_acc_rtl.py`.
  - Bring-up on 64×64 (`v3/work/f3/acc/bringup_rtl1`): **5/5 PASS** on the first run, every event and the total cycles
    equal to the model.
  - VTR smoke (`v3/vtr/smoke.py --module acc --point bert_tiny_128`; 128×128, seed 11; S 128, D 64, NBLK 8, SO = LN = 7,
    `P_CH` 8, $N_{OC}$ = 6):

    | `dpe` at elab / expected | `wc` placed / expected | Fmax (MHz) | CLB | memory | DSP | wall |
    |---|---|---|---|---|---|---|
    | 12 / 12 | 12 / 12 | 103.3 | 805 | 58 | 0 | 95 s |

    - 26 inferred memories, legal shapes: 8 FIFOs (512 × 65, 1R1W), 6 × lo_mem and e_mem (352 × 40, 1R1W), 6 × so_mem
      (352 × 5, 2R1W).
    - Path #1, 9.71 ns: register → NOT → 3-bit adder → 5 LUT levels → NOT → adder → NOT → subtractor → LUT → register.
      88 of the 100 worst paths start at that register, 12 at a second one (slack −9.68 … −9.15 ns).
    - Inferred from the path's shape, not mapped by name: the LOG engine's decision in one cycle, from the FIFO
      selection `(gf + P_CH − lbase) mod P_CH` and its compares, through `fifo_ok` and `decide`, into the same-cycle pop
      `(gf + P_CH − base_rd) mod P_CH < nact_rd`, to a FIFO pointer. This is v2 logic, unchanged here.
  - Corpus: `acc_gen_cases.py` wrote 19 GATE-1-certified cases to `v3/work/corpus/acc_64x64`.
  - **Fmax fix** (you: fix the Fmax first, then GATE 2). RTL only, both twins, no cycle changed (spec §10 item 6):
    - the FIFO sets of the LOG jobs became registered one-hot vectors (`nb1`, `rb1`, `cb1`, `need_l`, `need_r`) rotated
      by $N_{OC}$ per wave. They replace the base registers, the two `(gf + P_CH − base) mod P_CH` subtractions with
      their compares, and the compute stage's `(cbase + gi) mod P_CH` priority select (now an AND-OR select);
    - the LOG feed's adders are $\max(9 + SO, VWP) + 1$ = 17 bits at the smoke point (v2: 34).

    Twin gate ALL PASS with the baseline unchanged; lint 0; bring-up 5/5 again with every event and cycle equal
    (`v3/work/f3/acc/bringup_rtl2`). VTR smoke, same point and seed (run 1 kept in `v3/work/f3/acc/vtr_run1`):

    | run | `wc` | Fmax (MHz) | CLB | memory | DSP | wall |
    |---|---|---|---|---|---|---|
    | 1 (as written) | 12 / 12 | 103.3 | 805 | 58 | 0 | 95 s |
    | 2 (one-hot FIFO sets) | 12 / 12 | **156.3** | 754 | 58 | 0 | 77 s |

    Run 2's worst paths (`v3/work/f3/acc/vtr_run2`), 6.42 … 6.12 ns: 52 of 100 start at one register (adder → 6 LUT
    levels → adder → LUT → register), 48 at another (5 LUT levels → a slot memory's read address). Not mapped by name.
    The accumulator is now above the DIMM (132.3 / 140.6) and the softmax (140.3).
  - GATE 2 on the corpus, on the fixed RTL (your approval): **19/19 PASS** in 31 s, 8 jobs
    (`v3/work/f3/acc/gate2_run1/`, summary `v3/work/f3/acc/logs/gate2_run1.out`). Every case: 0 testbench errors
    ($O_8$ words, `run_done` at the model's cycle, no gap inside a row, credits granted before every unit, the FIFO /
    e_mem / reader probes, word and credit counts), every event line equal to the model's (69–2,082 events per case),
    total cycles equal. `SoftmaxOnlineAcc.BASIS` = "rtl-pinned"; the model suite re-run ALL PASS
    (`v3/work/f3/acc/logs/acc_model_final.log`).
- 2026-10-14: **F4, the layer** (`att_ctrl` + `att_layer`; your go). Logs in `v3/work/f4/logs/`.
  - Spec `top/spec.md`: the netlist (every wire), `att_ctrl`'s rules and latencies (`RUN_LAT`, `CR_LAT`, `DONE_LAT`,
    `BUSY_LAT`, 1 each), the model. Its §10 refinements of `architecture.md` §4.7 (updated): `x_rdy` waits for the
    copy's burst to end (a `dpe` keeps `MSB_SA_Ready` high through its load); `att_ctrl` reads `x_v`; jobs and runs
    offered from the start; one start per reset.
  - **Engine** (`sim/core/engine.py`; `simulator.md` §3.2, §4.5): data edges with fan-out and channel placement, a
    `pulse` port kind, credit edges (a credit can go to a controller instead of the producer). Existing graphs run
    unchanged: core tests ALL PASS (new: a hand-computed fan graph and 9 error cases), and the DIMM, softmax and
    accumulator corpora regenerate byte for byte. The accumulator's `run_done` became a pulse port (open in its own
    graph, so its events are unchanged).
  - Model: `top/model.py` (`AttCtrl`), `sim/workloads/attention_head.py` (`layer_params`, `attention_head`), case
    writer `top/cases.py`, tests `top/test_model.py` ALL PASS:
    - GATE 1 on 14 layers: O8 and every stream inside (L_Q, L_K^T, L_V in the DIMMs' memories, every score tile,
      every run's log P, every o_part tile) == `head_stages`; non-trivial values (BERT-Tiny's shape: 7,787 of 8,192
      O8 values nonzero, 89 distinct), 0.36 s per such layer;
    - every `att_ctrl` event recomputed from spec §5; admission terms bound: GEMM ready 222, PV B staging 248, QK A
      staging 56 (QK B never: its 2 matrices of $B_{kv}$ rows always have room);
    - the 432 DSE builds inside every module's guard; values off and random order identical; mutants 6/6.
  - RTL: `top/rtl/{sim,vtr}/att_layer.v` (wiring only; twin baseline: the 6 weight tie-offs) and `att_ctrl.v`
    (counters only; the same file in both twins). Twin gate ALL PASS (8 pairs; the whole layer elaborates in both
    twins); lint 0 products.
  - **Found at the first bring-up:** `gemm_top`'s `reg_full` is a busy flag one cycle behind the words, not the word
    valid (`architecture.md` §4.3 had it wrong); the word valid is the internal `out_valid` the v2 testbench read by
    name and the GEMM model was pinned to. All 5 cases failed on it (every DIMM operand shifted by one word). Fix:
    `out_valid` made a `gemm_top` output port (both twins; no logic changed; the GEMM's own testbench still passes,
    delta 0); `att_layer` takes it. `gemm_top.v` moved to FORKED in `tools/copies.py`.
  - Testbench `top/tb/tb_att_layer.v` (R8 preload of every crossbar; general window and credit monitors log every
    edge of the netlist by the stream rule) and runner `top/tb/run_layer_rtl.py`. Bring-up on 64×64
    (`v3/work/f4/bringup_rtl2`): **5/5 PASS**, every event of every edge (431–3,121 per case), the total cycles, the
    O8 values and `done` equal to the model.
  - Model-only finding (Needs you 13): the PV B staging (2 rows per copy) throttles the X stream on short GEMM
    periods (`v3/work/f4/exp/pb_cap.py`, `v3/work/f4/logs/exp_pb_cap.log`).
  - Gate-L corpus: `top/gen_cases.py --preset corpus` wrote 13 GATE-1-certified layers to
    `v3/work/corpus/layer_64x64`.
  - Forked from v2: both RTL twins, the testbench, the twin baseline. Obsolete (`tools/cleanup_f3.sh`): the two
    `att_ocarry_top.v` copies, `tb_att_ocarry_top.v`, `gen_ocarry_cases.py`, `run_ocarry_rtl.py`,
    `twin_baseline/att_ocarry.diff`; the `att_ocarry` twin pair is replaced by `softmax_online_acc`. Copy gate 52/52 OK +
    22 forked.
- 2026-10-15 (model only, answering the user's questions on Needs you 12–13): the PV B credit loop measured per token
  (`v3/work/f4/exp/pb_loop.py`): loop = GEMM part + PV B part = 25 + 40 at BERT-Tiny 64×64, 37 + 40 at 128×128, 24 +
  28 at JetFormer-150P 128×64; the credit binds when the loop exceeds 2 GEMM row periods. `pb_regs.py` (10 layers)
  and `pb_dse_regs.py` (432 DSE builds): 4 credits per copy == no credit, at most one row per channel in the row
  registers. `pb_dse.py`: 176/432 builds faster with 4 credits, max 8.9 %, none slower.
- 2026-10-15 F4 close (your go):
  - VTR smoke of the head (`v3/vtr/smoke.py`, then `--module layer`, BERT-Tiny at 128×128, P_PROJ 1, seed 11;
    `v3/work/vtr/att_layer_bert_tiny_128/`, log `v3/work/f4/logs/vtr_smoke_layer.out`): PASS. Elaboration 278 / 278
    `dpe`, 70 memories, 0 Parmys-illegal shapes; VPR wc 278 (expected 278), **116.547 MHz**, CLB 15,837, memory 376,
    DSP 0, io 88, wirelength 3.94 M; 12,306.6 s (ABC most of it). Path #1: 8.57 ns = clock-to-Q + 2 LUTs (0.36 ns)
    + 8.16 ns of wire, a register to the read address of a 64 × 128 dual-port memory (the QK DIMM's B memory, the
    only 64-deep one). The run used the 2-credit `att_ctrl`; the 4-credit change is one compare constant.
  - PV B credit 4 rows per copy (`att_ctrl.v` both twins, `top/model.py` `PB_CAP`, `top/test_model.py`, the DIMM
    model's `CAP_B`), then the order rule (the DIMM model's B port raises on rows out of row order; tests in
    `ip/dimm/test_model.py` staging and `top/test_model.py` rules; spec text in `top/spec.md` §3/§5/§10,
    `ip/dimm/spec.md` §3/§5/§10, `architecture.md` E0/E3/§5.2). The order is needed: P_PROJ 2 with copy 1 starved
    overran the row registers at 4 credits (`v3/work/f4/exp/pb_skew.py`); the user chose the rule over 4 registers.
    Gates: head model ALL PASS, DIMM model ALL PASS, DIMM corpus regenerates byte-identically, all 432 DSE builds
    run clean.
  - Rename "layer" → "head": new `top/rtl/{sim,vtr}/att_head.v` (VTR twin from `v3/work/f4/make_top_vtr.py`; equal
    to the old twin apart from the name), `top/tb/tb_att_head.v`, `top/tb/run_head_rtl.py`,
    `tools/twin_baseline/att_head.diff` (equal to the old baseline apart from the name), `head_params`, smoke
    `--module head`, gate H; docs where "layer" meant the head. Gates: check_twins ALL PASS (+ self-test), lint 0
    products, copies 50/50 OK + 24 forked, head / DIMM / core models ALL PASS.
  - Softmax R ≠ C (`ip/softmax/gen_cases.py --preset rc`; the softmax corpus regenerates byte-identically):
    `sm_rc_128x64` 1/1 PASS (1,462 events, 4,421 cycles, values; `v3/work/f4/softmax_rc_run1`).
  - Bring-up again on the final RTL (4 credits, new names; `v3/work/f4/bringup_h`, `v3/work/f4/bringup_rtl3`):
    **5/5 PASS**, every event (428–3,121 per case), the total cycles (933–7,049) and the values equal the model's.
  - Gate H (`v3/work/corpus/head_64x64`, `v3/work/f4/gateH_run1`, log `v3/work/f4/logs/gateH_run1.out`): **6/6 PASS**:
    every event (431–4,017 per case), the total cycles (1,100–8,109) and the O8 values equal the model's; X later
    than `x_rdy`, slow X, a late start, $N_E = d$, $V = 3$, $d = 64$ (`H_n128_d64`: 285 crossbars, 41 min). With the
    bring-up 5/5 and the softmax R ≠ C 1/1, **F4 is closed**; `AttCtrl.BASIS` = rtl-pinned; head / core models ALL
    PASS after the change.
  - The mapping discussion (your request, model only): $N_{EP} = d$ against the guards and in cycles
    (`v3/work/f4/exp/nep_{sweep,cap}.py`, logs `nep_*.out`; Needs you 2); $N_E$ and $P_{proj}$ ratios at 4 credits
    ($N_E = d$ ×1.005–1.17 cycles; $P_{proj}$ 2 ×0.59–0.92, 4 ×0.39–0.91); the VTR stage times of the head smoke
    (Needs you 16).
- 2026-10-15 Fmax fix 1 (your request "fix the frequency", after the architecture page): the head smoke's 100
  reported paths were one class, the QK DIMM's B memory read address (8.47–8.57 ns, 0.4 ns of logic): `go_q` (the
  job start) crossed the chip to the 8 slices and selected the first read address through `take_start` muxes.
  `dimm_tile_top.v` (both twins): each slice loads its A / B slot from the job register `ja_q` / `jb_q` while idle
  (`ga_q` / `gb_q` removed; `ja_q` / `jb_q` reset), the wave count `ns_q` and the fill index `fi_q` are zero while
  idle, so `ra` and `a_ra` come from slice registers only. Every read address in a job's cycles is unchanged (no
  timing-rule change; `ip/dimm/spec.md` §10 item 7). Twin gate ALL PASS, lint 0; a 4-case DIMM check
  (`v3/work/f4/fmax_check_rtl1`: two slices and channels with slot changes, a starved job, PV over 2 runs, 8 slices)
  4/4 PASS (every event, cycles, values, memory images). Running: VTR smokes of the head
  (`v3/work/vtr/att_head_bert_tiny_128`) and of the QK DIMM alone (its F3 summary saved in
  `v3/work/vtr/dimm_tile_top_qk_bert_tiny_128_f3`; F3 path #1 was a different class: a crossbar's `MSB_SA_Ready`
  → 3 LUTs → an engine register, 7.58 ns, 6.9 ns of it wire). Afterwards (your approval): the DIMM GATE 2 corpus and
  the head bring-up on the changed RTL.
  - QK DIMM VTR smoke on the fix-1 RTL (`v3/work/vtr/dimm_tile_top_qk_bert_tiny_128`, 4,638 s): 130/130 `dpe`,
    **130.643 MHz** (F3: 132.342), CLB 9,004 (8,991), memory 88: no change, as expected (the address path was not
    the DIMM's own limit). Its top 100 paths form a wall of classes at 6.7–7.7 ns: the leader's `MSB_SA_Ready` →
    3 LUTs → the other crossbars' `w_buf_en` (7.68 ns, 12 paths; the take-now feed strobe), a comparator register
    path (7.49), the leader's ready → engine registers (7.30), and counter → adders → 4 LUTs → register paths
    (7.13 and below, ~80 paths). Beyond ~130 MHz needs work on several of these classes inside the slice.
- 2026-10-15 mapping (your decisions after reading the architecture page): the latency terms on the 72 rows
  (`v3/work/f4/exp/latency_terms.py`, log `latency_terms.out`): front = $S T_p / P_{proj}$ + 80–140, core = $R_b B$
  blocks at the score stream $\lceil S B_{kv}/5 \rceil$ (median period 1.02× the stage estimate), tail = the last
  accumulator jobs and the O read-out. One at a time around the chosen point (`nexp_ne_72.out`): $P_{proj}$ 1 / 2
  +41–48 % / +11–16 %; $N_E = d$ ×1.011–1.169 (72/72 inside); $N_{EXP} = 4$ −6 % median but outside on 20 rows;
  $B_{kv}$ 8 / 32 0 to +8 %; $X_S$ 8 / 32 ±1.5 %. Compute cycles at the chosen mapping (`compute_cycles_72.out`): the
  final O read-out is 3.0–21.7 % of the full cycles (median 11.0 %); e.g. BERT-Tiny 128×128 8,684 → 6,889 cycles,
  232 crossbars. `experiment.md` §4–§7 and `architecture.md` §1 / §10 rewritten to the fixed mapping.
- 2026-10-15 **O read-out out of the cycles** (your request: the read-out is set by the 40-bit port, not by the
  crossbar size or the mapping; `acc_spec.md` §10 item 7):
  - RTL, both accumulator twins: the epilogue's $O_8$ goes to its own memory `o8_mem` per crossbar
    ($W \cdot 2^{\lceil \log_2 LW \rceil}$ × 40 bits), and the O reader reads it on its own port while the next run
    computes. `e_mem` keeps only the rescale's $e$ (its read-address mux and write-data mux are gone). The read-out
    hold `o_ok` now gates only the next run's epilogue. `run_done` pulses one cycle after the run's end (`o8_last`),
    so the head's `done` = the last run's $t_{o8} + 2$. Probe `emem_err` → `omem_err` (an $O_8$ write while the
    reader reads). `att_ctrl` and `att_head` are unchanged.
  - Model: `acc_model.py` (the rescale no longer waits for $t_{free}$; `run_done` at $t_{o8} + 1$; memories
    $3 N_{OC}$). `test_acc_model.py`: rules (`epilogue.free` binds 4×, `rescale.free` gone); the closed form vs v2
    with v2's O-feed hold off (77 of 105 runs compared, 14 schedules stop at an epilogue held by the read-out, a rule
    v2 does not have); a mutant case where the hold binds. `top/test_model.py`: `done` before the read-out. Specs:
    `acc_spec.md` §2–§7, §10; `architecture.md` §4.6, §4.7, §9, §10; `experiment.md` §5; `top/spec.md` §5.
  - Gates: twins ALL PASS, lint 0, copies 50/50 + 24 forked; accumulator model ALL PASS (GATE 1 290, mutants
    16/16); head model ALL PASS (432 builds inside the guards, mutants 6/6).
  - Accumulator VTR smoke (128×128, seed 11; the F3 result kept in `v3/work/vtr/softmax_online_acc_bert_tiny_128_f3`):
    12/12 crossbars, **152.4 MHz** (F3 156.3), CLB 726 (754), memory 70 (58: the 6 `o8_mem` take 12 blocks), DSP 0,
    88 s. Path #1 6.55 ns: LUT levels into a slot memory's read address, the F3 class.
  - RTL at 64×64 (small): the accumulator bring-up 5/5 + `acc_noc1_b1` 1/1 (the reader overlaps the next run in 5
    cases; the epilogue hold binds in 2, on 8 and 4 jobs; `v3/work/f4/omem/acc_rtl1`, `acc_rtl1_hold`); the head
    bring-up 5/5 (`v3/work/f4/omem/head_rtl1`, 272 s). Every event, the cycles, `run_done` / `done` and every $O_8$
    value equal the model's. The head runs also carry the fix-1 DIMM.
  - The 72 builds (model; `v3/work/f4/exp/compute_cycles.py`, `logs/cc72_{before,after}.tsv`, `omem_hold72.log`):
    the 42 builds with $R_b = 1$ are unchanged ($T = t_{o8} + 2$, one cycle more than the first rule's $t_{o8} + 1$);
    the 30 with $R_b \ge 2$ end 0–3.0 % earlier (median 1.6 %), because the earlier read-outs no longer hold the
    next run. The epilogue hold never binds in the 72 builds (smallest margin 1,743 cycles). E.g. BERT-Tiny 128×128
    $T$ = 6,890 (8,684 with the final read-out); ViT-B/16 128×128 21,745 → 21,101.
- 2026-10-15 **No O reader** (you, after asking why the `o8_mem` version used more memory: "remove the O reader and
  just expose the final accumulated results as output of the top module, the O projection will consume it"):
  - RTL, both twins: `softmax_online_acc.v` loses the O reader, `o8_mem` and the read-out wait (`o_ok`); E9 is
    `o_v` (1 bit, high while the normalize jobs drain) and `o_d` ($N_{OC}$ × 40 bits; word $i$ = crossbar $i$'s
    $O_8$ word, registered two cycles after the drain; zeros on crossbars without a window in the last wave);
    `run_done` one cycle after the run's last $O_8$ word. `att_head.v`: `o_d` widens to $N_{OC}$ × 40 bits.
    `att_ctrl` unchanged. Twins ALL PASS (both elaborate), lint 0.
  - Model: `acc_model.py` (E9 = one window per normalize wave, `E9_LAT` = 2, `e9_rows` decodes it to the $O_8$
    rows; no reader; memories $2 N_{OC}$; no $O_8$ memory write, no reader reads), the case writers (GATE 1 on the
    decoded E9; `o_win.txt`), the head workload's sink. Accumulator model ALL PASS (GATE 1 290, closed form 35
    schedules == v2 on every run, mutants 14/14); head model ALL PASS. The 72 builds: $T$ equal on 72/72 to the
    `o8_mem` version (`logs/cc72_noreader.tsv`), e.g. BERT-Tiny 128×128 $T$ = 6,890.
  - No RTL simulation (you). Head VTR smoke on this RTL (fix-1 DIMM included), BERT-Tiny 128×128, P_PROJ 1, seed 11,
    `clock.sdc` (clock only, no I/O delays: I/O paths are not timed), in `v3/work/vtr/att_head_bert_tiny_128_o8out`
    (`smoke.py --tag`, beside the fix-1 smoke still running). Elaboration: 278 `dpe` (expected 278), the same 70
    inferred memories (749,376 bits) as the fix-1 circuit with the old accumulator; `o_d` is 240 output pins.
- 2026-10-15 Fmax fix 1, head smoke result (`v3/work/vtr/att_head_bert_tiny_128`, BERT-Tiny 128×128, P_PROJ 1,
  seed 11, `clock.sdc` read by VPR; this circuit predates the O-reader removal): 278/278 `dpe`, **129.625 MHz**
  (F4: 116.547, +11.2 %), CLB 15,891 (15,837), memory 376 (376), DSP 0, io 88, 4.0 h. Path #1 7.74 ns: register →
  3 LUTs → a subtractor → register, about 88 % wire. The top 100 paths (7.74–6.84 ns) are mixed: 51 register →
  register, 32 into a crossbar's `w_buf_en` (the DIMM's take-now feed strobe, its own limit alone at 7.68 ns), 17
  others. The head now sits at the QK DIMM's own limit (130.6 MHz alone); going higher means the multi-class slice
  work (Needs you 15).
- 2026-10-15 head VTR smoke on the final RTL (no O reader, `o_d` 240 bits; fix-1 DIMM), BERT-Tiny 128×128, P_PROJ 1,
  seed 11, `clock.sdc` read by VPR (`v3/work/vtr/att_head_bert_tiny_128_o8out`, 4.0 h): 278/278 `dpe`, CLB 15,804
  (15,891 with the O reader), memory 376 (376), DSP 0, io 288 (88: `o_d` 40 → 240 bits), **116.387 MHz** (129.625 on
  the fix-1 circuit with the O reader). Resources as expected. Path #1 8.62 ns is DIMM logic, not the accumulator: a
  crossbar's `MSB_SA_Ready` → 5 LUTs → a subtractor → register, about 87 % wire; the top 100 paths (8.62–7.17 ns): 45
  register → register through an adder, 33 `MSB_SA_Ready` → register, 17 into `w_buf_en`. The logic on the path did not
  change, so the gap is placement: one seed cannot tell seed noise from an effect of the 200 extra output pins
  (wirelength-driven placement still pulls logic toward them; timing excludes I/O). Proposed (your go): VPR only, seeds
  13 and 17, on both synthesized netlists (about 30 min, 4 runs in parallel).
- 2026-10-15 **The DSE's model rows** at the fixed mapping $P_{proj} = 1$, $N_E = N_{EP} = 2d$ (your decision):
  - `tools/check_head_params.py` (new, static): for all 72 builds it evaluates `top/rtl/vtr/att_head.v` (localparams,
    the 8 instances, each module's header defaults) and compares with the simulator's module models: every parameter
    equal, crossbars RTL == `head_params` == the module models' resources. ALL PASS; a negative control (two corrupted
    model parameters) gives 144 mismatches.
  - Energy counts fixed to v2's attention ledger: the DIMM's pairing adds (one per product, `ip/dimm/model.py`) and
    narrowing compares (one per QK output) were missing; FIFO words now count as memory bytes, written once and read
    once (softmax: 5 bytes per word; accumulator: 9). Selects stay unpriced, as in v2. Specs: `ip/dimm/spec.md` §7,
    `ip/softmax/spec.md` §7, `acc_spec.md` §7, `simulator.md` §4.7, `experiment.md` §5. Gates: DIMM, softmax,
    accumulator and head models ALL PASS; F2 (the 48 FC rows) ALL PASS.
  - `dse/att_rows.py` (new) -> `work/dse/att_model_72.csv`: T, energy per term and per module, setup energy,
    crossbars, crossbar area, the model's memory bits; the smoke point also its VTR area and latency. BERT-Tiny 128×128:
    T 9,405 cycles, 933.6 nJ (setup 45.1 nJ apart), 278 crossbars (= VTR), area 43.92 mm² (CLB 15,804, memory 376,
    crossbars from VTR), 80.8 µs at 116.4 MHz (seed 11). Over the 72 rows the energy is 61.0 % crossbar passes, 9.8 %
    their port traffic, 27.9 % adds, 0.7 % compares, 0.5 % memory; by module QK 46.3 %, PV 45.3 %, accumulator 5.2 %,
    GEMMs 1.9 %, softmax 1.3 %. The GEMMs still count their output bytes as written (as in FC), which double-counts
    3 n d bytes in the head (BERT-Tiny: 0.24 nJ, below 0.03 %).
  - CLB, memory blocks and Fmax of the other 71 builds need VTR (Needs you 16).
- 2026-10-15 **Memory blocks: traced and predicted** (your question: is the memory count valid, can the simulator
  predict it):
  - VPR's packed netlist of the head smoke (`att_head_bert_tiny_128_o8out`): all 376 blocks in the arch's 20 Kb block
    in its 1024 × 20 dual-port mode, holding exactly the 70 memories Yosys inferred (nothing added, nothing lost). Each
    memory takes ⌈bits / 20⌉ blocks (1, 2, 4, 7 or 26), which sums to 380; the accumulator's six 5-bit sign memories
    (one per crossbar, same addresses and write enable) share 2 blocks instead of 6: 376. The DIMM's banks in a slice
    share addresses but not write enables, so they do not pack.
  - The simulator's memory inventory equals the RTL's 70 memories at that point (every shape and count). New:
    `Resources.mem_shared` (the accumulator's lock-step groups), `area.mem_modes / mem_blocks / predicted_mem_blocks`
    (the dual-port modes read from the arch XML), and `dse/att_rows.py` columns `mem_blocks_model`,
    `area_mem_model_mwta`, `mem_max_words`, `mem_check`. Prediction at the smoke point: 376 = VPR. The 72 builds:
    73–392 blocks; the deepest memory is 1024 words, inside the 1024 × 20 mode. Gates ALL PASS.
  - For the DSE: per build, Yosys's memory list must equal the simulator's before the long run, and VPR's memory and
    crossbar counts must equal the prediction after it. CLB is the only area term the simulator cannot predict.
- 2026-10-15 **F5 VTR driver `vtr/vtr_dse.py` and its test** (you: "proceed, report to me after the driver test
  finished"):
  - The driver: the 72 builds at the fixed mapping; circuit = the smoke's (`smoke.circuit` gained a `params`
    argument, `smoke.elab` a runner hook; the smoke's own circuit is unchanged, byte-identical to the final head
    smoke's file); folder `work/vtr/dse/<build>_<hash12>/`; elaboration check, one synthesis (`-end abc`), VPR per seed
    (the full flow's VPR command line under `time -v`), checks per seed, `work/dse/vtr_att_72.csv`, run log
    `work/dse/vtr_dse.log`; memory gate (per-process reserves in a budget), `prlimit` 48 GiB, no timeout; resume by
    step. `--self-test` (fake Yosys, flow and VPR) ALL PASS; 6 of 6 injected bugs caught. `dse/att_rows.py --vtr`
    reads the CSV and adds EDAP (pJ × ns × mm²).
  - Elaboration of the 72 (36 at once, 3.5 min, Yosys peaks 0.41–0.84 GiB): crossbars equal on all 72; 37 passed at
    first. The memory lists found two bugs in the DIMM model's memory list (`ip/dimm/model.py`): the A and B bank
    memories of one shape were counted once (an assignment, QK at $I = 256$), and depths were not rounded to the
    RTL's address fields (at least 1 bit per field; $T_W = 1$). Fixed; regression test `test_memories` in
    `ip/dimm/test_model.py` (values from Yosys; the old formula fails 4 of its 8 instances); DIMM gate ALL PASS,
    `check_head_params` ALL PASS. The 18 builds at $I = 256$ now predict 14–56 more memory blocks (range 75–412;
    `work/dse/att_model_72.csv` regenerated; cycles, energy and crossbars unchanged). The driver now caches Yosys's
    result and recomputes the check every run: 51 / 72 pass with no Yosys rerun. The other 21 are the $V > 1$ builds,
    three extra memories each, `gemm_top`'s `tree` (Needs you 18).
  - JetFormer-150P 128×128 end to end: synthesis 3,218 s (Parmys 64 s at 1.52 GiB, ABC 52 min at 0.5 GiB); VPR
    seeds 11 / 13 / 17 in parallel, 432–437 s each at 4.26 GiB: 132.3 / 136.5 / 147.2 MHz, mean 138.7 MHz; CLB 8,038;
    memory 151 = predicted 151; crossbars 146 = 146; I/O 208; DSP 0. Priced from the CSV: 1,346 cycles, 34.86 nJ,
    9.71 µs, 22.27 mm², EDAP 7.53e9.
  - The same command again: no tool started, every step skipped, the CSV line byte-identical.
  - Time: ABC grows about with the square of the netlist size (43 runs of v2 and v3; v3's heads take about twice
    v2's attention at equal size); this build took 52 min against 61 min scaled from the head smoke. The full run's
    critical path is the largest build's ABC: about 11–16 h if the $I = 256$ netlists are about 1.6× the smoke's, as
    in v2 [estimate]; the whole run about 12–17 h wall (was 10–14 h) and about 350 CPU-hours.
- 2026-10-15 **The $V > 1$ reduction tree fixed** (you: "fix the tree, run the 64x64 head check"):
  - RTL, both `gemm_top` twins identically: `tree` is one wide register (stage $s$, slot $k$ at bits
    $32(s \cdot NSLOT + k)$, stage 0 unused so no index goes negative), each index a function of the loop counters only
    (the per-iteration `g` variable removed); the read-out `red` reads the last stage's slices. Only the $V > 1$ branch
    and the read-out changed. Gates: `check_twins` ALL PASS (the twin diff unchanged), `lint_rtl` 0 runtime products
    (the 2 reported in `gemm_top` are v2's serializer `s_idx * EPS`, unchanged), `copies` 50 / 50 (gemm_top a fork
    since F4).
  - Testbench `tb_att_head.v` takes the E9 windows (`o_win.txt`, `o_d` $N_{OC}$ × 40 bits, the window monitor with
    LW words and W waves per run, `+NOWIN` from the case writer); `top/cases.py` writes `NOWIN`.
  - The 64×64 head check, gate H's $V = 3$ head `H_v3_p2` (n 32, $d_m$ 192, d 16, P_PROJ 2; the corpus's stimulus,
    regenerated in `work/f5/head_v3`, GATE 1 passed): **PASS**, 466 / 466 events equal the model's, 937 cycles = the
    model's, testbench errors 0 (52 O words = 4 windows × 13 with the oracle's values, done at the model's cycle);
    23 s (`work/f5/head_v3_rtl`). It is also the first RTL run of the E9 output since the O-reader removal.
  - The elaboration check of the 72 builds on the fixed RTL (new circuit hashes, new folders): **72 / 72 OK**, no
    `tree` memory left, the 21 $V > 1$ builds' memory lists equal the simulator's; 3 min 15 s, Yosys peaks
    0.38–0.84 GiB. The folders of the earlier circuit (the first elaboration and the JetFormer test, whose result
    138.7 MHz stays in its folder) are no longer used; the JetFormer build runs again in the launch.
- 2026-10-15 20:30 **F5 VTR launch** (you: "launch the full DSE. and spawn the background watcher, report every 90 mins"):
  `vtr/vtr_dse.py` on all 72 builds, seeds 11 / 13 / 17, 36 workers, the memory gate (budget 99 GiB), 48 GiB per
  process, no timeout. Driver pid 4190865 (its own session: it survives this Claude session; `work/dse/vtr_dse.pid`),
  stdout `work/dse/vtr_dse_full.out`, every event `work/dse/vtr_dse.log`, results `work/dse/vtr_att_72.csv`.
  Watcher `vtr/dse_watch.py` (pid 4191428; self-test ALL PASS): a status report 2 min after its start, then every
  90 min, and a final one with every build when the driver exits, appended to `work/dse/vtr_dse_status.log`.
  The 72 elaborations were reused (72 / 72 OK); syntheses start largest first, 60 s apart. Estimate: 12–17 h wall.
  To stop: kill the driver by its pid, then its tools (their pids are in the run log). To resume: the same command.
- 2026-10-16 10:30 **DSE status** (13 h 50 min in): 21 / 72 builds complete, 39 syntheses, 96 / 216 VPR seeds; no
  crash, crossbars equal on every seed. Synthesis under the 36-way load: 3.3–5.3 h at $I = 64$, 7.2–8.5 h at
  $I = 128$, 11.5–13.5 h at $I = 256$. Memory blocks equal the prediction on every seed at $I = 128$ (58) and $I = 256$
  (17); at $I = 64$ VPR places 5–7 more than predicted (397–399 vs 392, the same on every seed of a build), so those
  21 seeds are "not counted" by the strict check (their VPR numbers are in the folders and the CSV). The memory gate
  (12 GiB per VPR) holds the last 32 syntheses behind the VPR queue for about an hour. **You: no fix needed (neither
  the prediction nor the gate); wait for the run.**
- 2026-10-16 15:31 **F5 VTR done** (19 h 00 min after the launch): 72 / 72 syntheses, 216 / 216 VPR seeds, 0 tool
  failures, 0 driver errors; crossbars equal on every seed. Memory blocks equal the prediction on 56 builds; on the
  16 builds of the four $d = 64$ shapes at $C = 64$ VPR places 5–8 more (397–400 vs 392, the same on every seed), so
  the strict check counts none of their seeds (the driver's exit 1); their VPR numbers are in the CSV and folders.
  Fmax, mean of 3 seeds: 93.1–147.6 MHz, median 122.3; seed spread median 12.2 MHz, max 35.3. Syntheses: median
  3 h 56 min, max 13 h 28 min (the $I = 256$ builds 11.5–13.5 h under the 36-way load); VPR: median 20 min, max
  50 min. Run folders 296 GB (0.96 TB free). Results `work/dse/vtr_att_72.csv`; the final report (every build) at the
  end of `work/dse/vtr_dse_status.log`. Next, on your go: pricing (`dse/att_rows.py --vtr`; whether the 16 builds are
  priced from VPR's own memory count), then the figures (geomean ranking, Pareto front).
- 2026-10-16 **Pricing and figures** (you: "go with pricing and figures"; the 16 builds priced from VPR's count):
  - `vtr/vtr_dse.py`: a seed counts when VPR ended OK, Fmax > 0 and the crossbars are equal; the memory blocks vs the
    prediction go to a new column `mem_check` (self-test ALL PASS); `--aggregate` rebuilt the CSV from the folders
    (no tool): 72 / 72 builds counted, 3 seeds each; `mem_check` equal on 56, "VPR 397–400 vs predicted 392" on the 16
    builds of the $d = 64$ shapes at $C = 64$; the 56 others unchanged.
  - Pricing: `dse/att_rows.py --vtr work/dse/vtr_att_72.csv` -> `work/dse/att_priced_72.csv` (each seed's Fmax kept):
    latency 8.8–626 µs, area 10.9–76.6 mm², energy 34 nJ–8.3 µJ, EDAP 5.5e9–2.8e14 pJ·ns·mm².
  - Ranking: `dse/rank.py` (ported from v2's `rank_dse.py`, forked in `tools/copies.py`; self-test 21 checks ALL
    PASS): FC from the 48 snapshot rows rebuilt by the v3 model, each seed's Fmax re-read from v2's VPR folders —
    reproduces v2's FC ranking (largest relative difference 3.7e-16). **Attention: 128×64 is the best on all six
    workloads** (G 1.000 [0.994, 1.000], first in 100 % of 2,000 seed resamples; factors E ×1.03, D ×1.05, A ×1.00),
    then 256×64 0.753, 128×128 0.689, 256×128 0.574, ... 1024×256 0.150. **FC: 512×128** (G 0.829, first in 100 %),
    then 256×128 0.622, 512×256 0.619.
  - Figures (`dse/plot_dse.py`, miniconda Python; ported from v2's `plot_dse.py` and `plot_pareto_family.py`):
    `work/dse/figures/dse_{fc,attention}.{png,pdf}` (four panels) and `dse_pareto.{png,pdf}` (FC: front 128×64,
    256×64, 256×128, 512×128; attention: 128×64 alone on the 2-cost front, 128×128 also on the 3-cost front).
- 2026-10-16 Results page (you, for viewing in the Claude app): https://claude.ai/artifact/KiVh2rqfLT4ifXSiCMxbeR
  (private) — both ranking tables, the three figures with a zoom toggle; source `work/dse/figures/dse_results.html`
  (generated from the ranking JSONs), images published beside it.
- 2026-10-16 **FC × attention cross score** (you: the Pareto front with x = the FC EDAP score, y = the attention EDAP
  score; v2's metric, `v2/work/dse_v1/combined/plot_pareto.py`, recalled): `dse/plot_dse.py` -> 
  `work/dse/figures/dse_cross.{png,pdf}` (forked in `tools/copies.py`): one dot per config at (G_FC, G_ATT), the
  Pareto front (no config better in both) **128×64, 256×64, 256×128, 512×128**, v2's max-min pick **256×128** (both
  scores ≥ 0.574, the only config in that square), seed bands (5–95 %) as whiskers. v2's other rules on the same
  scores: geo √(G_FC × G_ATT) and the pooled 10-workload geomean pick 256×64 (0.626 / 0.650); the worst single
  workload picks 256×128 (0.436); best pair 128×64 (attention) + 512×128 (FC). Which rule is the headline: yours (the
  combined ranking is still deferred in `experiment.md` §5). Added to the results page (version 2).
- 2026-10-16 **Cross score in 1 − EDAP form** (you: "Let's do 1-EDAP. The point closer to the origin is the better"):
  `dse/rank.py` adds `cross_ranking` (self-test 24 checks ALL PASS; the family rankings byte-identical) ->
  `work/dse/ranking_cross.json`: d = distance of (1 − G_FC, 1 − G_ATT) from the origin, its seed band, and how often
  each config is the closest (draw i of the FC resamples with draw i of the attention ones). **Closest: 256×64,
  d = 0.539 (0.521–0.564), the closest in 99 % of the 2,000 draws**; then 256×128 0.569 (1 %), 512×128 0.641, 512×64
  0.704, 128×64 0.714; the front unchanged (128×64, 256×64, 256×128, 512×128). Figure `figures/dse_cross.{png,pdf}`
  redrawn (equal axes, distance arcs); results page version 3 with the distance table.
- 2026-10-16 **No resampling** (you: "dont use the 2000 points. just the raw data"): `dse/rank.py` rewritten without
  the seed draws (self-test 21 checks ALL PASS; every G, rank, factor and per-workload score identical to before; FC
  still equals v2); `dse/plot_dse.py` panel (c) = G only, the cross figure without whiskers; new `dse/results_page.py`
  regenerates the page from the three JSONs (results page version 4, no resampling text). Results unchanged:
  attention #1 128×64 (G 1.000), FC #1 512×128 (0.829), cross: 256×64 closest (d 0.539), then 256×128 (0.569).
- 2026-10-16 **Latency vs area** (you: "just latency vs. area ... each config has an area / latency score = current
  value / max value, so the lower the better"; my choices, stated: per-workload scores averaged by the geometric mean,
  the families combined by the geometric mean of their scores). `dse/rank.py` `la_scores` / `la_cross` (self-test 25
  checks ALL PASS) -> `work/dse/latency_area.json`; `dse/plot_dse.py` -> `figures/dse_la.{png,pdf}` (three panels)
  and `dse_la_{fc,attention,cross}.png`; `dse/results_page.py` -> results page version 5 (this view only).
  Fronts (area score / latency score): **FC** 512×128 (0.353 / 0.710), 256×128 (0.467 / 0.516), 256×64 (0.511 /
  0.489), 128×64 (0.740 / 0.354); **attention** 128×64 alone (0.449 / 0.522: smallest area and lowest latency);
  **across** 512×64 (0.460 / 0.703), 256×64 (0.483 / 0.537), 128×64 (0.577 / 0.430). 256×128 is not on the cross
  front: 256×64 is lower on both (0.483 / 0.537 against 0.529 / 0.571).
- 2026-10-16 **Cross-family on the area-delay product** (you: the equal-weight area / latency scores across
  families are wrong, since the families' fronts differ; use the ADP score, x = FC, y = attention): `dse/rank.py`
  `adp_scores` / `adp_cross` replace `la_cross` (self-test 26 checks ALL PASS); `latency_area.json` key `adp`.
  **Front: 256×128 (FC 0.404, attention 0.384), 256×64 (0.419, 0.272), 128×64 (0.439, 0.238)**; the FC ADP scores on
  the front lie within 9 % of each other, the attention ones within 38 %. Figure `figures/dse_adp_cross.png` (and
  panel (c) of `dse_la.{png,pdf}`); results page version 6.
- 2026-10-16 **Energy column and distance circles** (you): `dse/rank.py` adds the energy score (energy ÷ the highest
  energy per workload, geometric mean; beside the fronts, not part of them) and, on the ADP cross plane, the distance
  d = sqrt(FC ADP² + attention ADP²) with the rows sorted by d (self-test 26 checks ALL PASS). Figure
  `dse_adp_cross.png` / panel (c) of `dse_la.{png,pdf}`: axes from 0, circles of equal d, the solid one through the
  closest. **d: 128×64 0.499, 256×64 0.500 (0.13 % apart, a tie), 256×128 0.558**, then 512×64 0.635, 512×128 0.639.
  Energy scores: FC 128×64 0.848 / 256×64 0.489 / 256×128 0.424 (512×128 the lowest, 0.306); attention 0.470 /
  0.544 / 0.505. Results page version 7. Correction to my chat note: 128×64 uses 2.8× 512×128's FC energy (3.6× the
  best of each workload, not of 512×128).
- 2026-10-16 **Final results and the FC vs attention analysis** (you: update `experiment.md` and the LEDGER with
  the final results, and explain why attention differs from FC). `experiment.md` §8 (results) and §9 (analysis);
  `dse/breakdown.py` -> `work/dse/breakdown.json`. Findings:
  - Architecture: FC is a weight-stationary GEMV (1–64 tiles, a larger array holds more weights per tile);
    attention multiplies activations, so its crossbars are EXP / LOG converters on $I \times I$ windows, their count
    fixed by $d$ (78 / 146 / 278–297 for $d$ = 16 / 32 / 64) and nearly the same at every array size.
  - Area: FC 75–96 % crossbar tiles (8× the grid area for 32× the cells, so it falls with size); attention 65–86 %
    CLB (10.4k / 12.9k / 17.5k CLBs at $C$ = 64 / 128 / 256) plus tiles that grow with $R \times C$.
  - Latency: FC 52–172 cycles at 358–451 MHz; attention 12k–22k cycles at 110–131 MHz. Measured in the simulator:
    the attention projections end at $n(\lceil \min(d_m, R)/5 \rceil + 8)$ cycles (within 7 cycles on all 72),
    a median 50 % of the cycles (36–86 %), growing with $R$ only while $d_m > R$; the core is about $n^2/5$
    (the score port), independent of the array.
  - Energy: a pass costs the full array (14.8 to 103.4 pJ); FC passes fall 30 → 2 with size, so its energy falls
    with area; attention's pass count depends on $I$ only (62k / 31k / 16k), so at a fixed $I$ energy rises with $R$.
  - Assumptions that drive it (§9.6): the fixed mapping, the full-array pass price, one projection copy, the 40-bit
    ports, FC as one GEMV. Next: Needs you 19 (the simulator for sensitivity studies).
- 2026-10-17: **the simulator refactor** (your request: human-readable, camelCase, OOP, modular, same results; your
  decisions: one package `v3/simulator/`, about a third of the checks kept — the engine's wiring and protocol checks,
  the RTL domain guards, the port contracts — the RTL parameter and localparam names kept, the platform shrunk to
  NL-DPE's needs with functional equivalence as the bar, the agent implementing; "git tracking is good" → the
  pre-refactor tree committed as `816807b`).
  - Step 0: the reference frozen from the old tree (`simulator/tests/equivalence/freezeReference.py`, 11 s): the 72
    attention + 48 FC DSE builds' timing-only Reports (cycles, every event line, every step, counts, resources),
    the 8 corpora the old case writers produce (75 cases), the DSE rows; sha256 manifest keyed to the commit.
  - Steps 1–5: `core/` (cycleMath, errors, records with structured `PortEvent`s, module with `Params` / `Derived`
    dataclasses and `STEP_METHODS`, graph, engine, stages, environment), `crossbar/`, `modules/` (gemm, dimmTile,
    softmaxOnline, softmaxAccumulator, attentionController, each with a `*Math` class holding the values),
    `workloads/` (fc, attentionHead with `AttentionHeadParams`), `platform/` (platform, prices, area: 6 price
    functions, no Azure-Lily code, no v2 kernel import), `analysis/` (rows, breakdown). Job ports carry their
    fields as control data (`PortSpec.control`), which removed the old `set_job_fields` side channel. One real
    finding on the way: the head workload must read the controller's `RUN_LAT` live (a by-value import broke the
    mutant test, not the results).
  - Gates, all green: the ported module tests (crossbar 258 corpus cases, gemm 160, DIMM 243 GATE 1 + 8 Yosys
    memory lists, softmax 210, accumulator 290, head 14; every closed form, every mutant); `checkEquivalence.py`:
    120 Reports, 120 rows, 75 corpus cases all equal; `differentialRandom.py`: 640 random cases through both
    simulators, values on and off, 0 differences; after the switch `check_head_params` 72/72, the copy lock 48/48,
    `att_priced_72.csv` byte-identical, `breakdown.json` identical, `vtr_dse --self-test`, F2 ALL PASS (25 s).
  - Step 6: `tools/{check_head_params,f1_check,f2_check,copies}.py`, `dse/{att_rows,fc_rows,breakdown}.py`,
    `vtr/{smoke,vtr_dse}.py` and the case writers / generators import the package; `docs/simulator.md` §4.8 and
    `PLAN.md` R1 point at it. The old tree is unused and waits for your cleanup (Needs you 20).
- 2026-10-17: **the folder reorganization** (your request: vtr, dse results, driver scripts and the simulator more
  organized and compact; your "yes. go" with the defaults: results tracked, `work/` cleanup in the script, the 276 GB
  of VTR builds left). One question per folder: `ip/` `top/` (what we build), `oracle/` (what the simulator is
  checked against), `simulator/`, `vtr/` (everything that runs VTR), `dse/` (scripts, `inputs/`, `results/`),
  `tools/` (gates), `work/` (scratch). 11 top-level folders became 9.
  - Moves (copy, then fix the paths; the old places wait for the script): `sim/v2_kernels` → `oracle/v2_kernels`
    (`sim/` gone), `arch/` → `vtr/arch/`, `dse/v2_snapshot` → `dse/inputs/fc_snapshot`, `work/dse` results →
    `dse/results/` (now tracked), `simulator/tests/equivalence/checkEquivalence.py` → `simulator/tests/`,
    `work/f4/arch_page` + `exp/page_data.py` → `docs/architecture_page/` (`page_data.py` ported to the package and
    checked against the old one run on `816807b`: the 720 in-domain builds identical, the 144 out-of-domain differ
    only in the error class name, `RuntimeError` → `ContractError`; `build.py` now writes to `work/arch_page/`),
    smokes → `work/vtr/smoke/`. `tools/copies.py` follows (kernels, arch XMLs, FC snapshot: 25 copies re-made,
    48/48 OK, the lock is keyed by the v2 source so it did not change).
  - One real fix: `dse/rank.py` built two score dicts from a `set`, so the key order of `latency_area.json` changed
    from run to run (a tracked file would churn); they keep the points' order now, and the JSON is identical across
    hash seeds.
  - Checks: copy lock 48/48, twins, lint (9 twins, 0 products), check_head_params 72/72, F2 ALL PASS, checkEquivalence
    ALL EQUAL, rank self-test, `vtr_dse` and `dse_watch` self-tests, `vtr_dse.py --aggregate` byte-identical to
    `dse/results/vtr_att_72.csv`, the 72 build folder names unchanged (the 276 GB of runs stay valid), every
    regenerated result byte-identical to the original (PDFs equal after their date fields; the results page
    differs only in its data paths; one JSON in key order). F1 (probe, lock, twins, selftests 17/17 from
    `oracle/v2_kernels/`, regen of 6 corpora byte-identical to v2, isolation 0 accesses outside v3/) ALL PASS; its
    `rtl` step was not run (it simulates; needs your approval). The first F1 run failed on `dimm_tile` only: its v3
    generator was removed in F3 by `cleanup_f3.sh`, so that entry had been dead since then, not because of the
    reorganization; it is out of `tools/f1_check.py` now (the DIMM corpora are `ip/dimm/gen_cases.py` + GATE 2).
    README.md rewritten as the map plus a command table; PLAN.md, simulator.md, experiment.md,
    simulator/README.md follow.
- 2026-10-17 (later): your cleanup script applied. `simulator/README.md` got a "Use it" section (the build → `run` →
  `Report` → price pattern, a what-if, and a table of which DSE script calls what; every number in its snippet was
  asserted by running it). `energyByRole` existed twice (`rows.py` totals, `breakdown.py` per term); only the
  `breakdown.py` one is left, and `att_priced_72.csv` regenerates byte-identical.
- 2026-10-17 (later still): **`simulator/` flattened** (your remark: too many folders). 7 sub-folders plus `configs/` became
  4: `core/` (+ `crossbar.py`), `models/` (the module models + the two workloads, `fc` and `attentionHead`),
  `platform/` (+ `rows.py`, `breakdown.py` from `analysis/`, and the two JSON configs from `configs/`), `tests/`.
  Imports: `simulator.crossbar.crossbar` → `simulator.core.crossbar`, `simulator.modules.X` and `simulator.workloads.X`
  → `simulator.models.X`, `simulator.analysis.X` → `simulator.platform.X`. Same method as before (copy, rewrite the
  imports, verify); nothing but file locations changed: F2 ALL PASS, checkEquivalence ALL EQUAL, check_head_params 72/72,
  copy lock 48/48, `att_priced_72.csv`, the rankings, `breakdown.json`, 9 figures and the results page regenerate
  byte-identical, `builds.json` (864 builds) identical, no module loads an old path. The old folders wait for your
  one-line `git rm` (Needs you 22).
- 2026-10-17 (DSE, your "what about DSE?"): `dse/` had 5 scripts and 2 data folders, so no restructure; two untidy things
  fixed. (1) `dse/inputs/fc_snapshot/` (a folder with one child) is `dse/fc_snapshot/`. (2) The six shapes and twelve
  configs were copy-pasted (5 and 14 places, 4 simulator tests among them); `dse/space.py` defines them once and
  `att_rows`, `breakdown`, `vtr_dse`, `check_head_params`, `page_data` and the four tests import it (the unported
  `docs/architecture_page/exp/` scripts keep their own). Checks: F2, checkEquivalence, check_head_params, vtr_dse and
  rank self-tests, copy lock 48/48; `att_priced_72.csv`, the rankings, `latency_area.json`, `breakdown.json` and
  `builds.json` regenerate byte-identical; the 72 VTR build folder names are unchanged.
- 2026-10-17 (DSE views; your answer: keep the EDAP view, but classify the results into EDAP and ADP; snake_case stays):
  `dse/results/` keeps the data both views read (`vtr_att_72.csv`, `att_priced_72.csv`, `breakdown.json`) and has one flat
  folder per view: `adp/` (the current view: `latency_area.json`, `dse_la.{png,pdf}`, `dse_la_fc.png`,
  `dse_la_attention.png`, `dse_adp_cross.png`, `dse_results.html`) and `edap/` (the earlier energy x delay x area ranking:
  `ranking_{fc,attention,cross}.json`, `dse_{fc,attention,pareto,cross}.{png,pdf}`, `dse_results_edap.html`). `figures/`
  is gone. `rank.py`, `plot_dse.py` and `results_page.py` write there (the page script writes both pages now; the EDAP
  one used to be an unused function). Every JSON and PNG is byte-identical to the old file, the PDFs are equal after their
  date fields, the ADP page equals the old page after its data paths; the EDAP page builds. File names are unchanged, so
  the published page's PNG names stay valid.
- 2026-10-17 (v2 removed; your words: "untrack and remove the v2/ as now we are fully committed to v3"): the restore point
  is commit `db76d1b`, the tag `v2-final`: v2 as it was on disk, with its 55 files that existed only in the working
  tree (16 edited, 16 new such as `v2/docs/history.md`, `v2/sim/simulator/KERNELS.md` and the attention plots, and
  `v2/work/dse_v1`, whose two plot scripts are sources in `copies.lock`). Not saved: 81 regenerated logs and 38 GB of
  git-ignored scratch (`v2/work` 20 GB, `v2/experiment/vtr/out` 18 GB, stimulus corpora, byte-code). Only two v3 tools
  read v2: `tools/copies.py` now reads its sources from the tag (48/48 with no file under `v2/` opened; a missing tag
  gives 0/48 and `origin/main` 2/48) and `tools/f1_check.py` (the probe's dirty command targets `softmax_study/`;
  `regen` runs the v3 generators only; `369aa21`). You ran `v3/work/tmp/remove_v2.sh --go`: `git rm -r --cached v2`,
  commit `afb1caf` (7,980 files), `rm -rf v2` (39 GB; 944 GB free). Gates on the tree without v2/, all passing: copy
  lock 48/48 and its self-test; check_twins; lint_rtl (9 VTR twins, 0 runtime products); check_head_params (72 builds);
  F2 (24 s); checkEquivalence (all equal); the vtr_dse self-test; the 72 priced rows regenerated byte-identical to
  `att_priced_72.csv`; F1 (probe, lock, twins, 17 self-tests, isolation with 0 accesses outside v3/; 470 s). The root
  `.gitignore` lost its v2 rules; the README, PLAN and `memory.md` say v2 is removed (`d0545ca`). `CLAUDE.md` (your "do
  needs update") is rewritten for v3 (`3bb2cc5`); the old text is `d44e71d`. Dropped from it, and in `d44e71d`: the
  Round 1/2 constants (the 120×120 grid, BRAM counts, replica formulas), the BERT-Tiny constants, the DPE cycle axioms
  and the v2 key-path tables.
- 2026-10-17 (the push; your "drop the first four and junk, build main-v3", then "done"): `main-v3` = `df0c4e3`, one
  commit on `86dc189` with att-kernel's files minus 61 generated files (13.9 MB: 33 auto-generated benchmark RTL files,
  the softmax study's 18 smoke work files, 5 DSE-era run logs, two root logs, `sweep.csv`, the empty `.gitmodules`, a
  `.DS_Store`). Kept after a closer look: the softmax study's 24 per-run VPR logs and `vtr_softmax.log` (its `.gitignore`
  keeps them as evidence), `v1-v2/logs/` (the deliverable's evidence) and the two Azure-Lily CNN sources in
  `benchmarks/rtl`. The push: one fast-forward commit, a 2.7 MB pack; 760 files, 22.7 MB, no `v2/`. You ran
  `drop_generated.sh --go` (`c5c1bf0` on att-kernel, the same 61 files, so both branches hold the same files) and
  `git push origin main-v3:main`. Local `main` now points at `df0c4e3` and is the working branch (its 4 old commits are
  inside att-kernel). The tag `v2-final` and the history stay local, so `tools/copies.py` runs only in this clone.
- 2026-10-17 (one notation; your words: "lets do 1 is the worst, the smaller the better. so parato front remains the same,
  only the ranking needs regenerating but the actual rank does not change"): `dse/rank.py` has one normalizer (value ÷ the
  largest on the workload, geometric mean) for EDAP, its E / D / A factors, latency vs area and ADP; the result files name
  every family figure $G_X$ (`G_EDAP`, `G_E`, `G_D`, `G_A`, `G_ADP`; the cross rows `G_FC`, `G_ATT`, `d`); `plot_dse.py`
  and `results_page.py` use the same names (the EDAP heatmaps use the reversed colormap, so the best cells stay bright).
  Checks against the previous files: the per-family EDAP rank orders are identical; old $G$ × new $G_{EDAP}$ is one
  constant per family (spread at most 7e-16), and so are the new $G_E$, $G_D$, $G_A$ against the old factors; the
  latency vs area and ADP numbers are bit-identical; every Pareto front is unchanged (FC: 128×64, 256×64, 256×128,
  512×128, plus 128×128 on the 3-cost front; attention: 128×64, plus 128×128 on the 3-cost front). Only the EDAP
  cross-family order moved, as expected (the distance depends on the normalizer): 256×64 (d 0.313) and 256×128 (0.330)
  stay first and second, 512×64 and 512×128 swap places 3 and 4, 128×64 stays 5th; its front is unchanged.
  `rank.py --self-test` ALL PASS (26 checks, rewritten for the new direction). `experiment.md` §5 has a notation row,
  §6 and §8 use the names. The published results page is not republished.
- 2026-10-17 (the pass price; your words: "we always fire full crossbar, every time, it is just no reading all row inputs
  when only some rows are used", then "outputs full, fix it and re-price"): `prices.py` prices each ledger key (mode, rows
  used, columns used) on its own: cells, ACAM, output buffer and drain in full, the input-buffer bytes and input port words
  by the rows used; the dead `activeColumns` argument (an ADC term NL-DPE has no component for) is gone. Effect on the 72
  attention builds (nothing but energy changed): e_total −6 % at 128×64, −15 % at 256×64, −27 % at 512×64, −41 % at
  1024×64, −14 % at 512×128, −12 to −15 % at 1024×256, under −1 % at 128×128; an attention pass now costs 13.9 pJ at
  128×64 (14.8 full), 23.0 at 1024×64 (37.3), 91.7 at 1024×256 (103.4). FC: 12 of 48 rows (K < R with V = 1) are
  7.4–39.0 % cheaper. Rankings: the latency vs area view and the ADP cross are bit-identical (no energy in them); EDAP
  per family keeps #1 and #2 in both families (FC 512×128 then 512×256 now, 256×128 third; attention 128×64, 256×64);
  1024×64 rises 9 → 6 in attention and 10 → 9 in FC; EDAP across the families: 256×64 (d 0.340), 256×128 (0.366),
  512×64 (0.369, now on the front), 512×128, 1024×64, 128×64 (6th). Tests: `testFcRows.py` rewritten (the full extent
  reproduces v2's energy exactly; the used extent equals v2 minus the unread rows' input cost, computed from the raw price
  table; 48/48, max relative difference 2.2e-16); `checkEquivalence.py` leaves the 8 re-priced columns to it (the
  reports, counts and the other columns stay equal). Gates: F2 ALL PASS, equivalence ALL EQUAL, head params 72/72, rank
  self-test 26. `dse/breakdown.py` now writes `area_split` and `energy_split` per config (and prints both tables), the
  pass count from the rows' new `passes` column, the mean price paid per pass beside the full-array price; the per-family
  figures have three panels (Pareto front, area split, energy split). Energy splits: attention DPE passes 54–70 %, CLB
  25–40 %, port traffic 4–5 %, memory under 1 %; FC DPE 69–88 %, CLB 2–23 %, port traffic 5–17 %. `dse/` got
  `README.md` (the pipeline, step by step), `run_all.sh` (one command, about a minute) and `data.py` (the two input
  tables, loaded once for `rank.py` and `breakdown.py`); `experiment.md` §5, §8 ($G_E$ columns), §9.2, §9.4, §9.6 and
  `simulator.md` §4.7 state the rule and the new numbers; the page is republished.
- 2026-10-17 (your "these scripts together with the driver should be all placed inside dse/"): `vtr_dse.py` (the VTR
  run of the 72 builds; `--aggregate` rebuilds `results/vtr_att_72.csv` from the run folders) and `dse_watch.py` moved
  from `vtr/` to `dse/`, path lines only (the driver puts `v3/vtr` on its path for `run_vtr.py` and `smoke.py`);
  verified from the new place: both self-tests pass and `--aggregate` reproduces the committed CSV byte for byte.
  `vtr/` keeps the generic flow (`arch/`, `clock.sdc`, `run_vtr.py`, `smoke.py`). `dse/README.md` has the driver as
  step 0 (not part of `run_all.sh`, which never runs VTR). The originals leave with your `git rm`
  (`v3/work/tmp/move_vtr_dse.sh`, which also commits the move as renames).
- 2026-10-18 (your "use the vtr_results.csv and re-price the FC results. and merge the fc_snapshot/ and results/ ...
  naming"): `dse/fc_rows.py` (new, the mirror of the attention row writer) prices the 48 FC rows of `space.py FC_SHAPES`
  × `CONFIGS` with the v3 model on `results/fc_vtr_results.csv` and writes `results/fc_priced_results.csv` (the same
  e_*_pj columns as the attention table: the memory term is `e_mem_pj`, v2's `e_ser_pj`; plus `area_total_mm2`, `edap`);
  `att_rows.py` becomes `attn_rows.py` → `attn_priced_results.csv` (byte-identical to `att_priced_72.csv`);
  `vtr_att_72.csv` becomes `attn_vtr_results.csv` (`vtr_dse.py` writes it; `--aggregate` reproduces it byte for byte).
  `data.py` loads the two priced tables (no simulator run), so `rank.py`'s and `breakdown.py`'s JSONs come out
  byte-identical to the committed ones. v2's `dse_v2.csv`, `vtr_results.csv`, `ranking_fc.json` move to
  `simulator/tests/ref/` as the locked regression reference (copy lock 48/48, 29 forked, `fc_vtr_results.csv` recorded
  as the 48 gemm rows of v2's table); `testFcRows.py` and `checkEquivalence.py` read them there (ALL PASS / rows ok).
  README, PLAN, `dse/README.md` (step 1 is now two writers), `experiment.md` §3.1 and §6, `simulator.md`, CLAUDE.md
  and memory.md name the new files. The old paths (`fc_snapshot/`, the two old CSVs, `att_rows.py`, and the earlier
  `vtr/vtr_dse.py`, `vtr/dse_watch.py`) leave with your `git rm` (`v3/work/tmp/finish_results_layout.sh`, one commit
  with the renames).
- 2026-10-18 (your "so all results are now priced with fixed energy modeling?"): checked, not assumed.
  `dse/check_energy.py` (new, 4 s, part of `run_all.sh`) restates the pass price from the raw price table and prices the
  simulator's ledger of every row with it: the 48 FC rows equal `fc_priced_results.csv` to 1e-13 pJ, the 72 attention rows
  equal `attn_priced_results.csv` within the stored 0.01 pJ, the pass counts and the five role columns agree, and the
  pass price holds at its boundaries (84 cases: 1, 2, 5, 6, R/2, R-1, R rows on all 12 configs; 0 and R+1 rejected; an
  empty ledger costs nothing). The rule moved 12 of 48 FC rows (7-39 % cheaper) and all 72 attention builds (to 41 %).
  Downstream: the JSONs, figures and both pages regenerate byte-identically from the tables (`run_all.sh`); HEAD
  (`bbe3360`) holds the re-priced attention table and everything derived; the FC priced table and the renames are staged
  for `finish_results_layout.sh`. Sized, deliberate rules: outputs full (your decision) vs by the columns used would move
  attention energy 0-4 % (128x256 3.3-4.2 %, 256x256 up to 2.1 %, the others under 1.1 %), because the projection GEMMs
  drain N = d < C columns too (my earlier "only 128x256" was wrong); the zero-padded rows of a partial last row tile
  are read and charged (9 builds, 0.02-0.22 % of their energy).
- 2026-10-18 (the drain rule; your "ensure the latency modeling is correct: only load the used rows, but always drain the
  full output column"): trial and measurements in Needs you 24; the model change is reverted until your go. Under the
  rule F2 fails crossbar (126 P29 cases: `drain_end`, cycles), gemm (96) and dimm (the converter overrun above); softmax,
  accumulator and head pass; the equivalence gate differs on 4 FC reports (`drain_end` only), 56 attention reports and
  22 attention rows. Today's short drains are 0.3 % of the attention passes at C = 64, 3–8 % at C = 128 and the larger
  C = 256 arrays, 100 % at 128×256, and in FC only the 1×512×128 shape on C = 256 (11–14 % of its passes).
- 2026-10-18 (the drain rule, your decision): the trial is reverted (`core/crossbar.py` identical to HEAD, checked), the RTL
  untouched; the timing keeps the early-stopping P29 drain and the energy the full output side (Decisions). Nothing in the
  results changes.
- 2026-10-18 (your edit, then "rerun the run_all.sh ... re-publish"): `core/crossbar.py` drains `ceil(8 C / BUF)` words on every
  pass and no longer takes `colsUsed`; `gemm.py`, `dimmTile.py`, `softmaxOnline.py`, `softmaxAccumulator.py` call it without the
  argument. My re-run of `run_all.sh` reproduces your tables, JSONs and page byte for byte, so the page is republished from them;
  `check_energy.py` run on them: ALL PASS (its step is not in your `run_all.sh`: line 11 is not a command any more). Against the committed results: energy identical on all 72 builds; cycles up on 22 of 72
  (none at C = 64; up to +13.8 % at C = 128; +26 to +83 % at 128×256, up to +62 % at the other C = 256); FC unchanged on all 48
  rows; both latency vs area fronts unchanged; ADP across the families now 256×64 (d 0.478), 128×64 (0.483), 256×128 (0.523).
  Open: with `colsUsed` gone the drained window is C bytes wide while the consumers still slice COLS_USED / K / I bytes, so the
  values path breaks (F2: gemm `cannot reshape (64) into (1,1,9)`, dimm `(64,) into (16,)`, head; testCrossbar still passes
  `colsUsed=`); the timing-only DSE path is unaffected. The fix is to keep `colsUsed` as the window's valid width with the drain
  at C (the form of the 2026-10-18 trial); then the cycle references of the crossbar, GEMM and DIMM tests and the frozen
  equivalence reports need the new drain (Needs you 24, steps 2–5 without the RTL).
- 2026-10-18 (your "yes, fix it and update the test references"): `core/crossbar.py` keeps `colsUsed` as the drained
  window's valid width and sizes the drain from C (`OUTPUT_CYC = ceil(8 C / BUF)`); the four models pass their extents
  again, so the values path works. Test references: `testCrossbar` takes the v2 kernel run at `cols_used = C` as its
  timing witness (values from the expected files; case.json's cycles too where a case drains every column);
  `testGemm` the same where every column is drained, else `pass_engine`'s schedule at cols = C plus L_w (written as the
  charter's TREE_PIPE + 1, not read from the model: a first version read `gemm.SER_LAT` and missed the latency mutant)
  and the COLS_USED output words; the DIMM's converter period becomes `convPeriod(L, C)` = max(load + P, P + 2, the
  drain of C columns + 1) in `ip/dimm/cases.py`, `gen_cases.py`, the tests and `spec.md` (the standalone generator no
  longer overruns); `freezeReference.py` (new) froze the model into `work/equivalence/ref_2026-10-18/`, now
  `checkEquivalence.py`'s default (the 816807b reference stays as the refactor's proof: against it 4 FC reports differ
  in `drain_end` only, 56 attention reports and 22 attention rows). F2 ALL PASS (crossbar 258 cases, 216 sweeps, 5/5
  mutants; GEMM 160, 900, 3/3; DIMM, softmax, accumulator, head), equivalence ALL EQUAL on the new reference,
  `check_energy` ALL PASS. The projection-stream formula gains the drain term, $T_p = \max(\lceil \min(d_m, R)/5
  \rceil + 8, \lceil 8C/40 \rceil + 1)$: within 28 cycles of the simulator on all 72 builds (4,852 without it); the
  projections' median share is 56 % (was 50 %); attention cycles 12k–23k per config (was 12k–22k). `experiment.md`
  §4.4, §9.3, `simulator.md` §4.2, `architecture.md` §1, the DIMM and softmax specs and the page's latency note say so;
  the page is republished. The RTL twins are untouched: `dpe_nldpe.v` still drains COLS_USED words, so the model is
  not RTL-pinned for passes with COLS_USED < C (0.3 % of the attention passes at C = 64, 3–8 % at C = 128 and the
  larger C = 256 arrays, all of 128×256).
- 2026-10-18: the results layout is committed (`ffe0ef2`, your run of `finish_results_layout.sh`; the first run stopped
  at `git rm` of `att_rows.py`, whose staged edit `attn_rows.py` carries, so the script now uses `-f` for that file
  only). Today's work follows as the next commit; gates on the tree before it: F2 ALL PASS, equivalence ALL EQUAL
  (`ref_2026-10-18`, local in `work/`; `freezeReference.py` re-freezes it), check_energy ALL PASS, copies 48/48,
  head params 72/72, rank self-test, `run_all.sh` reproduces every output. `run_all.sh` keeps your commented-out
  `check_energy.py` line.
- 2026-10-18 (your "crossbar area, CLB area, bram area, energy and load cycles, compute cycles, drain cycles (0 if
  hide), Fmax (% to the lowest), geomean across 6 heads for 128x64, 128x128, 256x128, 256x64"): new
  `dse/config_table.py` (definitions first in `experiment.md` §9.3) -> `results/attn_config_table.md`; rows grouped
  by C with R inside; the drain column is the arithmetic mean with the count of heads where it is not hidden (a
  geomean of a column that is 0 on most heads is 0). `--self-test` ALL PASS (38 checks: the drain boundary, four
  hand-computed heads, the three cycle counts == n T_p on 24 builds, the area parts == area_total_mwta on every row).
- 2026-10-18 (your "add 2 more columns: ADP and EDAP ... for 12 configs", then "build both tables into
  config_table.py"): `config_table.py` writes two tables over the 12 configs: (1) the chain relative to the best ADP
  (128×64), whose columns multiply exactly (latency = cycles ÷ Fmax, ADP = area × latency, EDAP = energy × ADP),
  and (2) what is inside (the area parts and total, the energy, the cycles as load / compute / drain not hidden /
  rest and total, Fmax). `--self-test` ALL PASS (149 checks; the chain against rank.py's G ratios at 1e-6 relative,
  since the priced table rounds latency_ns to 0.001 ns; the four cycle parts == the head's cycles on all 72 builds).
  Reading: the rest of the head is the same at every R for C = 64 (4,244 cycles), so rows only add load and
  crossbar area; columns add CLB area (I wide), the rest (longer query runs) and, at C = 256, the drain.
- 2026-10-18 (your "drop the share column, build the critical-path trace"): the share column is gone from
  `config_table.py`. **Traced times** (`simulator/core/trace.py`, `simulator.md` §3.1): a `Time` is an int that
  carries the chain of (label, cycles) segments that reached it; `max` keeps the winner's chain, `+ n` appends a
  segment labeled by the context (the engine: the module's role per step; the crossbar: dpe_load / dpe_compute /
  dpe_drain; the models: the port words and FIFO reads as `stream`, the drain words a consumer takes as
  `dpe_drain`; the rest `logic`). The run's last event therefore holds its critical path, and the segments add up
  to T exactly. Values unchanged (a Time compares, prints and serializes as its int): F2 ALL PASS (9 tests, the new
  `testTrace.py`: 63 checks), equivalence ALL EQUAL, `run_all.sh` byte-identical, `check_energy` ALL PASS.
  `dse/critical_path.py` → `results/critical_path.{json,md}` (self-test 265 checks on the 120 builds: the chain ==
  T, none untraced or negative, the FC closed form per kind, T == the priced tables). **Reading (attention, mean
  share of T over the 6 heads):** DPE compute is 2–23 % everywhere; at C = 64 the path is DPE load 51–62 % +
  stream 15–20 % (the score words into the softmax, one per cycle); at C = 256 the projections are drain-bound
  (128×256: DPE drain 77 %, the loads hidden under the previous token's drain); by module the projections hold
  39–63 % and the softmax 21–37 %. FC: load 50–80 %, the serializer's words 11–53 %, compute 6–17 %. A tie keeps
  the first input the model names. **Cost, measured A/B against HEAD on the same machine:** the chains make every
  run about 3× longer (72 heads 3.4 → 10.1 s; the gate's corpus part 14.6 → 41.9 s; the generators alone
  +10–60 %), so `trace.enable(False)` turns them off for a run that needs no breakdown (then `t + n` is a plain
  int and `breakdown` refuses); on by default, used by `critical_path.py` and `testTrace.py`. The first gate run
  (751 s equivalence, 68 s F2) had external load on top. Scratch of the A/B: `work/tmp/ab_*`, `time_builds.py`.
- 2026-10-18 (your "merge it into dse/config_table.py"): `critical_path.py` is folded into `config_table.py`, which
  now runs every build (timing only, ~15 s) and writes `attn_config_table.md` (table 2's cycle columns are the
  critical path by kind, replacing the projection-formula split; table 3 the shares by kind and by module; per-head
  tables at 128×64 and 128×128; the FC rows by kind) plus `critical_path.json` (every build). The origin's cycle
  (the start) counts as logic / ctrl, so the shares sum to 1. `--self-test` 301 checks on the 120 builds. The two
  leftovers are untracked and yours to remove: `bash v3/work/tmp/drop_critical_path.sh --go` (`critical_path.py`,
  `results/critical_path.md`). Docs: `experiment.md` §9.3 (one block), `simulator.md` §3.1, both READMEs.
