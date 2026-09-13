# Legacy RTL Verification — Status Brief

**Prepared for collaborator meeting, 2026-08-29.** Scope: what the legacy
(pre-v2) RTL flow verified, how strong that verification actually is, and the
concrete errors found — historical and current. Legacy = generated RTL under
`rtl_flow/rtl|tb|smoke/` (frozen) + archived simulator
(`archive/azurelily_simulator/`).

---

## 1. Verification inventory & progress

| Area | Coverage | Status | Cycle-check result |
|---|---|---|---|
| **DPE primitives** (NL faithful, AL faithful, DSP-MAC) | 52 cases: identity / random / signed-extreme inputs, ACAM modes, M-sweeps {1,2,4,8}, precision {4,8,16} | **52/52 PASS** (re-verified green this week) | uniform **+2** delta (NBA handoff), decomposed in `CYCLE_ACCOUNTING.md` §3 |
| **Weight-stationary projection / FC** (`fc_top.v` + `tb_fc`) | 13 cases: Stage 1A (V=1,H=1) ×7, 1B (V>1 K-reduction) ×3, 1C (H>1 N-concat) ×3; both archs, ReLU on/off | **13/13 PASS** | deltas fully decomposed: +6 wrap, +7 AL-activation, +8 tree+clb — `delta = nba+tree+clb+wrap` identity holds |
| **VTR synth path** (`fc_top_synth.v` + `dpe` blackbox) | 3 workloads (bert_qkv, lenet_fc1, bert_ffn1), NL only | **3/3 OK** (Task #89, WIP) | Fmax + resources from VPR, 3-seed capable |
| **Safe-softmax study** (separate flow, `softmax_study/`) | AL + NL RTL, S ∈ {128,256}, bit-exact oracles | **8/8 PASS**, bit-exact + cycle-exact | 18-run 3-seed VTR sweep + supply-matched AL rebuild |
| **Stage 1D (general V×H)** | vgg_fc3, resnet_fc, bert_qkv_batched | not started in harness | — |
| **DIMM / attention head RTL** | — | **never built** (spec exists: `attention_dimm_mapping.md`) | — |

## 2. What the PASSes actually mean (verification quality caveats)

1. **FC functional truth is a one-byte pattern** — `tb_fc.v:expected_byte_fn`
   checks a single repeated byte, not an independent GEMM reference. A real
   datapath bug in fc_top could pass. Full random-matrix oracle never existed.
2. **Cycle truth is an analytical formula** (Task #98/#99,
   `T(M) = T_fill + (M−1)·T_steady`) authored in prior sessions —
   self-consistent across all 65 cases, but never independently grounded.
   Superseded by the frozen v2 spec (`v2/spec/dpe_nldpe.md` v1.0).
3. **Weights enter the faithful primitive via TB backdoor** — the module has
   no weight-programming port (zero-initialized in `initial`).
4. **Only two geometries ever exercised**: NL 256×256, AL 512×128.

## 3. Errors found & fixed (historical, documented)

| Error | Fix | Impact |
|---|---|---|
| Simulator "six defects" baked into published CSVs: F1.5 (activation energy charged without CLB), F2/F3/F4 (gemm_dsp §4 refactor), F5 (DIMM lanes W=16), energy-constant rescaling in seq-len sweep | all fixed; pre-fix CSVs kept as diff baseline (`benchmarks/results/pre-fix/`) | fixed-vs-published energy: AL/P-1 1.46 → **2.08**, AL/P-2 1.69 → **2.15**; latency unchanged |
| Fused-softmax CLB subtract **units error** (`cols/freq`) | batched over available CLBs, `ceil(cols/min(total_clb, cols))` | <0.03% end-to-end (not load-bearing) |
| `exp_fpga` assumed N-way parallel exp | rejected — needs N×8 = 8192 CLBs vs ~533 VTR-reported; modeled `floor(total_clb/8)` | ~+1.3% end-to-end |
| Attention-head gate Bug 1: head FSM missing outer N×N loop (single-row attention only) | outer loop added, both archs | correctness of full attention |
| Attention-head gate Bug 2: `defparam sm_exp/ws_log` wrong crossbar geometry | pinned to KERNEL_WIDTH=1, NUM_COLS=1 | correctness of softmax DPE instances |
| Single-substrate LOAD-gate design charged +PRECISION per pass (WAR hazard) | redesigned to double-buffered substrate, gate eliminated (Task #93→#99) | steady period −8 cycles/pass (NL 60→52) |
| Round-2 DSE counted DPEs by area | grid-based `count_available_wc` | replica counts correct |
| Round-2 replica limit ignored BRAM | three-resource limit min(P_dpe, P_clb, P_bram) | replica counts correct |

## 4. Errors found this week (new — not yet in any doc)

1. **Latent output-buffer overwrite hazard** in `dpe_nldpe_faithful.v`: no
   gate exists between the compute channel and the output drain
   (`acam_out` written whenever compute k+1 finishes, regardless of
   `output_busy`). Safe only when the schedule keeps ACAM k+1 after drain k —
   true at 256×256, **false at 256×512** (ACAM k+1 fires ~50 cycles before
   drain k ends → pass-k output bytes corrupted mid-stream). Never surfaced:
   only C=256/128 ever tested. The v2 spec's P11 (strict write-after-drain)
   removes the hazard class.
2. **Interface semantic gaps**: `dpe_done` is a level (not pulse), reset is
   asynchronous, four VTR ports dead (`_unused_aux`), ACAM mode is
   compile-time only — all redefined in v2 spec §4/§7.
3. **Verification-chain weakness** (drives current work): weak functional
   oracle + formula-from-memory ⇒ bottom-up re-verification launched
   (spec v1.0 frozen → user-built oracle/simulator → clean-room RTL →
   three-way cross-check with legacy as witness).

## 5. Open architectural assumption for collaborators — CLOSED 2026-09-12

**Input buffer double-buffering.** Legacy assumes two ping-pong substrates
(refill any time → `T_steady = max(LOAD, COMPUTE, OUTPUT)` = 52 @256×256).
The v2 spec assumes a single buffer, refill gated on MSB fire
(`T_steady = max(LOAD+P, COMPUTE, OUTPUT+1)` = 60 @256×256, 104 @256×512).

**Resolution (2026-09-12): collaborators confirm single buffer** — it is *not*
double-buffered. v2 spec v1.0 stands as frozen: no P1 flip, no oracle
re-transcription, no ~13% headline bump. Legacy's 52/103 is a confirmed
differing witness (reported, not gated, spec §9). Item 4.1's output-buffer
hazard is covered by spec P11 (strict output ordering) — no double output
buffer required.
