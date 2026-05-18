# FC / GEMM RTL Plan (Stages 1A → 1D)

**Status**: planning, pre-implementation. Builds on the verified DPE behavior
model (commits `202cdf1`, `d60493d`, `6a7fdd6`).

**Anchor**: `fc_verification/FIDELITY_METHODOLOGY.md` §5 (VMM workload class),
§7 (VMM tiling), §3 (DPE architecture). Plus `paper/methodology/gemm_rc_tradeoff.md`
for the analytical formulas. Plus `azurelily/IMC/imc_core/imc_core.py:run_gemm`
for the simulator's encoding.

---

## §1 Principle: FC == GEMM at the cycle level

`Y = X @ W [+ b] [→ activation]` where:
- `X` is `[M × K]` (batch × in_features for FC; M × K for GEMM)
- `W` is `[K × N]` (weight matrix)
- `Y` is `[M × N]`

Same M·K·N MAC count, same cycle count, same DPE pass count. The only
addition for FC is a post-matmul activation LUT (+1 CLB cycle per
`run_gemm` when V>1 or `has_acam=False`).

**Decision**: ONE top module `fc_top` parameterized by:
- `M`, `K`, `N` — workload shape
- `R`, `C`, `BUF` — DPE geometry (per-arch)
- `ACTIVATION` — 0 (raw GEMM) or 1 (FC with CLB activation LUT)

Same module covers FC and GEMM. Cycle formula identical except the
activation cycle.

---

## §2 Documented mapping (recap)

### Per-DPE pass cycles (§4 single-buffered drain-load overlap)

```
LOAD       = ceil(R × 8 / BUF)   # input vector streaming
COMPUTE    = N_b + pipeline_depth - 1   # bit-serial; controlled by TB/controller
                                         # (DPE itself is precision-agnostic, Model Y)
OUTPUT     = ceil(C × 8 / BUF)
T_steady   = max(LOAD, COMPUTE, OUTPUT)
TREE_PIPE  = ⌈log₂(V)⌉ for V > 1, else 0   # Phase 2 pipelined CLB tree

# Phase 1 baseline (DPE primitive only, no wrapper):
T_fill_phase1 = LOAD + COMPUTE + OUTPUT + 2   # +2 FSM handoffs

# Phase 2 with synthesizable fc_top wrapper (Task #88):
T_fill_phase2 = T_fill_phase1 + 4 + TREE_PIPE
              = LOAD + COMPUTE + OUTPUT + 6 + TREE_PIPE
```

For NL-DPE INT8 (R=C=256, BUF=40): T_fill_phase2 = 120 (V=1) / 121 (V=2), T_steady=52.
For AL INT8 (R=512, C=128, BUF=16): T_fill_phase2 = 336 (V=1) / 337 (V=2), T_steady=256.

### Tiling (§5 + §7 + run_gemm)

```
V = ceil(K / R)        # K-axis vertical tiles per output column
H = ceil(N / C)        # N-axis horizontal tiles per output row
n_parallel_dpes = V × H    # one DPE per (k-tile, n-tile)
passes_per_dpe  = M × V × ceil(H / n_parallel_dpes) ≈ M × V
                  # H tiles fire in parallel; each DPE does M × V passes
```

### Total cycles (matches simulator's run_gemm post-Phase-2)

```
total_cycles = T_fill_phase2 + (passes_per_dpe - 1) × T_steady
             + (1 if CLB_NEEDED else 0)
CLB_NEEDED   = (V > 1) OR (activation_mode AND not has_acam)
T_fill_phase2 = L + C + O + 6 + TREE_PIPE
TREE_PIPE    = ⌈log₂(V)⌉ for V > 1, else 0
```

Path A claim: passes_per_dpe = M (each DPE fires once per output row,
NOT M·V). See FC_GEMM_WALKTHROUGH §1.

For ACTIVATION=0 (raw GEMM): omit the +1 unless V > 1.
For ACTIVATION=1 (FC): apply the +1 per the rule above.

### Storage layout

- **Input X**: stored in BRAM, M × K bytes. Single read port (or per-DPE replicated).
- **Weights W**: distributed across V × H DPE instances, each holding R × C tile.
- **Output Y**: M × N bytes, written to BRAM.
- **CLB adder tree**: V-wide per output column, summing K-tile partial sums (only when V > 1).
- **CLB activation LUT**: per output column, applied after CLB tree (when ACTIVATION=1 + V>1, or always for AL).

### Dataflow per row m

```
for v in 0..V-1:                         # K-tile passes (sequential per DPE)
    for h in 0..H-1:                     # H DPEs fire in parallel
        DPE(v, h).fire()                 # one VMM pass: input X[m][v*R:(v+1)*R]
                                         # weights W[v*R:(v+1)*R, h*C:(h+1)*C]
                                         # output: C-element partial sum
    # All H DPE outputs available; concatenate into a row-of-N partial sum

# After V K-tile passes:
for n in 0..N-1:
    Y_partial[m][n] = sum_{v=0..V-1} (DPE(v, h_for_n).output[n mod C])  # CLB tree
    if ACTIVATION:
        Y[m][n] = activation_lut(Y_partial[m][n])
    else:
        Y[m][n] = Y_partial[m][n]
```

For consecutive m: §4 pipeline — DPE(v,h)'s output drain for row m overlaps
with DPE(v,h)'s input load for row m+1.

---

## §3 Stage 1A — V=1, H=1 single-DPE (simplest)

**Scope**: single DPE instance, no K-tile reduction, no N-tile mux, no CLB
adder tree. Optionally CLB activation LUT (if ACTIVATION=1 and AL).

**RTL module**: `fc_verification/rtl/fc_top.v` (or `gemm_fc_top.v`)

Internal structure:
- Input SRAM holding X (M × K bytes)
- One DPE instance (uses verified `dpe_nldpe.v` or `dpe_azurelily.v`)
- Optional CLB activation LUT (param-gated)
- Output SRAM holding Y (M × N bytes)
- FSM controller orchestrating M sequential DPE fires with §4 pipelining

Top-level handshake (workload-level, NOT DPE-level): standard valid/ready or
similar; design choice. Likely:
```
input  wire        clk, reset, valid_in
output wire        ready_in
input  wire [BUF-1:0] data_in    // streaming X bytes
output wire [BUF-1:0] data_out   // streaming Y bytes
output wire        valid_out
```

Or simpler: hierarchical-force X and W from TB, no streaming interface.

**Recommended for Stage 1A**: hierarchical-force X + W (TB sets `dut.input_sram`
and `dut.dpe_inst.weights`); top emits Y to a register/SRAM that TB reads.
Keeps the TB pattern consistent with the primitive smoke TBs.

**Test shapes (Stage 1A)**:
- `M=1, K=R, N=C` — V=1, H=1, single-fire (= one full DPE pass)
- `M=4, K=R, N=C` — V=1, H=1, 4 sequential fires (validates §4 pipelining)
- `M=8, K=256, N=256` — batched single-DPE GEMM
- Both NL-DPE (R=C=256) and AL (R=512, C=128) with appropriate K, N

**Expected cycles** (post-Task #88 Phase 2: T_fill = L+C+O+6+TREE_PIPE):
- M=1, K=R=256, N=C=256, V=1, ACTIVATION=0: T_fill = 52+10+52+6+0 = 120
- M=4: T_fill + 3×T_steady = 120 + 3×52 = 276
- For ACTIVATION=1 with NL-DPE V=1: same as ACTIVATION=0 (ACAM-fused, no +1)
- For ACTIVATION=1 with AL V=1: +1 cycle (no_acam → CLB always)

**TB validation**:
1. Functional: weights pre-loaded (identity or numpy-generated); inputs pre-loaded;
   compare `Y_observed[m][n]` against numpy `(X @ W)` for all m, n.
2. Latency: T_first_fire to T_last_output, compare to expected formula.
3. Activation: when ACTIVATION=1, optional ReLU (or whatever LUT we model);
   numpy oracle applies the same activation.

---

## §4 Stage 1B — V>1, H=1 K-tile reduction

**Scope**: V parallel DPEs along K axis. CLB adder tree per output column.
+1 CLB activation cycle per `run_gemm` rule.

**RTL additions vs Stage 1A**:
- Instantiate V copies of DPE: `dpe inst_0 ... inst_{V-1}`
- Each DPE_v holds W[v*R:(v+1)*R, :] tile
- Per-DPE input: X[m][v*R:(v+1)*R] (different K-slice per DPE)
- All V DPEs fire in parallel for output row m
- CLB adder tree (depth ⌈log2(V)⌉) sums V partial sums per output column
- Activation LUT applied AFTER the adder tree

**FSM**: V DPEs share a synchronized fire clock; tree reduction happens in
parallel with the OUTPUT drain stage (per the simulator's `effective_output_cycles
= max(output, reduce)` model — usually output dominates so reduction is hidden).

**Test shapes (Stage 1B)**:
- `lenet_fc1`: M=1, K=400, N=120 → V=2 at R=256
- `bert_ffn2`: M=1, K=512, N=128 → V=2 at R=256, H=1 at C=256... wait, N=128 < C=256, so H=1 ✓
- `resnet_fc`: M=1, K=512, N=1000 → V=2 at R=256, H=4 at C=256 (this is Stage 1D, not 1B)
  - For 1B-only test: pick K=512, N=256 (V=2, H=1) — synthetic shape

Restrict 1B test catalog to V>1, H=1:
- M=1, K=2R, N=C (synthetic, V=2 H=1)
- M=4, K=2R, N=C (synthetic, V=2 H=1)
- `lenet_fc1` derivative: M=1, K=400, N=120 (V=2, H=1 at R=C=256)

**Expected cycles** (post-Task #88 Phase 2: T_fill_phase2 includes TREE_PIPE):
- M=1, K=2R, N=C, V=2, TREE_PIPE=1, ACTIVATION=0: T_fill + 1 (V>1 CLB) = (52+10+52+6+1) + 1 = 122
- M=1, K=2R, N=C, V=2, ACTIVATION=1: same 122 (V>1 already drives CLB tree)
- Note: for M=1 V=2 H=1, there is no T_steady term (only one row).

---

## §5 Stage 1C — V=1, H>1 N-tile concatenation

**Scope**: H parallel DPEs along N axis. Output concatenation, no CLB tree
(since V=1, no K-reduction needed).

**RTL additions vs Stage 1A**:
- Instantiate H copies of DPE
- Each DPE_h holds W[:, h*C:(h+1)*C] tile
- Input X is broadcast to all H DPEs (same K elements)
- All H DPEs fire in parallel
- Output concatenated: Y[m][h*C:(h+1)*C] = DPE_h.output

**Test shapes (Stage 1C)**:
- M=1, K=R, N=2C (synthetic, V=1 H=2)
- M=4, K=R, N=2C
- `bert_ffn1` derivative: M=1, K=128, N=512 → V=1 (K=128 < R=256), H=2 at C=256

**Expected cycles** (post-Task #88 Phase 2): same as Stage 1A (H tiles
fire in parallel, no extra cycles beyond single-DPE since H DPEs are
independent; TREE_PIPE=0 for V=1).
- M=1, K=R, N=2C, V=1, ACTIVATION=0: T_fill_phase2 = 120
- M=1, K=R, N=2C, V=1, ACTIVATION=1 + has_acam: T_fill_phase2 = 120 (V=1 ACAM-fused, no +1)
- ACTIVATION=1 + AL (no_acam): +1 cycle

---

## §6 Stage 1D — General V×H

**Scope**: V × H DPE array, CLB adder tree, optional activation, output mux.

Combines Stage 1B (V>1) and Stage 1C (H>1) machinery.

**Test shapes (Stage 1D, real CNN workloads)**:
- `vgg_fc2`: M=1, K=4096, N=4096 → V=16, H=16 at R=C=256 (huge but tractable)
- `vgg_fc3`: M=1, K=4096, N=1000 → V=16, H=4
- `alexnet_fc1`: M=1, K=9216, N=4096 → V=36, H=16 (very large; may be slow in iverilog)
- `resnet_fc`: M=1, K=512, N=1000 → V=2, H=4
- `bert_qkv_proj`: M=128, K=128, N=128 → V=1 H=1 but M=128 (batched attention input)

For very large shapes (V>10), consider abbreviating in the harness — substitute
with smaller synthetic shapes that exercise the same V/H structure.

---

## §7 Workload catalog (driver seed)

```python
WORKLOADS = [
    # (label, M, K, N, activation, notes)
    # Trivial / single-DPE (V=1 H=1)
    ("bert_qkv_proj",   1,   128,  128, "relu",   "trivial single-DPE"),
    ("gemm_trivial",    1,   256,  256, "none",   "GEMM, no activation"),
    ("gemm_batched",    8,   256,  256, "none",   "batched GEMM"),
    # K-tile (V>1 H=1)
    ("lenet_fc1",       1,   400,  120, "relu",   "V=2 K-tile"),
    ("vgg_fc1_v",       1,   512,  256, "relu",   "synthetic V=2 H=1"),
    # N-tile (V=1 H>1)
    ("bert_ffn1",       1,   128,  512, "relu",   "V=1 H=2 (at R=C=256)"),
    ("synthetic_h2",    1,   256,  512, "relu",   "synthetic V=1 H=2"),
    # General (V>1 H>1)
    ("vgg_fc2",         1,  4096, 4096, "relu",   "V=16 H=16 (large)"),
    ("vgg_fc3",         1,  4096, 1000, "relu",   "V=16 H=4"),
    ("resnet_fc",       1,   512, 1000, "relu",   "V=2 H=4"),
    # Stress / extreme
    # ("alexnet_fc1",   1,  9216, 4096, "relu",   "V=36 H=16, may be slow"),
]

ARCHS = [
    # (arch_tag, R, C, BUF, has_acam)
    ("nldpe", 256, 256, 40, True),
    ("al",    512, 128, 16, False),
]

PRECISIONS = [4, 8, 16]   # default 8
```

User-extensible: `--shape M,K,N` CLI override adds a custom workload.

---

## §8 Python driver (`fc_verification/run_fc_smoke.py`)

Mirrors `run_dpe_smoke.py` pattern:

```python
# CLI:
#   python3 fc_verification/run_fc_smoke.py
#   python3 fc_verification/run_fc_smoke.py --workload bert_ffn1
#   python3 fc_verification/run_fc_smoke.py --shape 1,128,128 --activation relu
#   python3 fc_verification/run_fc_smoke.py --arch nldpe --precision 4
#   python3 fc_verification/run_fc_smoke.py --stage 1A          # only V=1 H=1 cases

# For each (workload, arch, precision):
#   1. Compute V = ceil(K/R), H = ceil(N/C)
#   2. Compute expected_cycles per the §2 formula
#   3. Compute expected_output Y = numpy(X @ W) [+ activation]
#   4. Generate iverilog command:
#      iverilog -DARCH_<arch> -DM_TB=<M> -DK_TB=<K> -DN_TB=<N> ...
#               -DR_TB=<R> -DC_TB=<C> -DBUF_TB=<BUF>
#               -DPRECISION_TB=<P> -DPIPELINE_DEPTH_TB=3
#               -DACTIVATION_TB=<0|1>
#               -o /tmp/tb_fc_<label>
#               fc_verification/rtl/dpe_<arch>.v
#               fc_verification/rtl/fc_top.v
#               fc_verification/tb_fc.v
#   5. Run vvp, parse:
#      - functional check (PASS/FAIL based on byte-level Y match)
#      - cycle count (T_first_fire to T_last_output)
#   6. Compute fidelity = (RTL_cyc - Sim_cyc) / Sim_cyc  (sim from run_gemm)
#   7. Report per-case: PASS/FAIL, expected vs observed, fidelity %

# Output: fc_verification/results/fc_smoke.log
```

CLI knobs:
- `--workload <label>`: filter to specific workload(s)
- `--shape M,K,N`: custom workload shape
- `--activation none|relu`: override activation
- `--arch nldpe|al|both`: filter
- `--precision N`: override precision (default 8)
- `--stage 1A|1B|1C|1D`: filter to stage's V/H constraints
- `--quick`: smaller subset
- `--keep`: keep tmp binaries

---

## §9 Makefile additions

`fc_verification/Makefile` gets new targets:

```makefile
# FC workload tests
M ?=
K ?=
N ?=
ACTIVATION ?= 0   # 0 = raw GEMM, 1 = FC with activation LUT

tb-fc-nldpe:
    @echo "[make] tb-fc-nldpe M=$(M) K=$(K) N=$(N) ACTIVATION=$(ACTIVATION)"
    iverilog $(IV_FLAGS) -DARCH_NLDPE \
       -DM_TB=$(M) -DK_TB=$(K) -DN_TB=$(N) \
       -DR_TB=$(NLDPE_R) -DC_TB=$(NLDPE_C) -DBUF_TB=$(NLDPE_BUF) \
       -DPRECISION_TB=$(PRECISION) -DPIPELINE_DEPTH_TB=$(PIPELINE_DEPTH) \
       -DACTIVATION_TB=$(ACTIVATION) \
       -o $(BIN)/tb_fc_nldpe \
       $(RTL)/dpe_nldpe.v $(RTL)/fc_top.v $(TB_DIR)/tb_fc.v
    vvp $(BIN)/tb_fc_nldpe

tb-fc-al:    # ditto for AL

fc-smoke:
    python3 $(REPO)/fc_verification/run_fc_smoke.py
```

Usage:
```bash
make tb-fc-nldpe M=1 K=128 N=128                        # bert_qkv_proj on NL-DPE
make tb-fc-nldpe M=4 K=256 N=256 ACTIVATION=1          # batched GEMM with activation
make tb-fc-al    M=1 K=400 N=120                        # lenet_fc1 on AL
make fc-smoke                                            # full FC sweep
```

---

## §10 Stage execution order

```
Stage 1A: V=1 H=1 single-DPE
  - fc_top.v (V=1 H=1 specialization OR parameterized for V/H ≥ 1)
  - tb_fc.v (functional + latency)
  - run_fc_smoke.py (workload catalog, drives multiple shapes)
  - Makefile targets
  - Test: bert_qkv_proj, gemm_trivial, gemm_batched + custom shapes
  - Verify: numpy oracle match + cycle count match
  - STOP for review

Stage 1B: V>1 H=1 K-tile reduction
  - Add CLB adder tree + activation LUT
  - Test: lenet_fc1, vgg_fc1_v + synthetic V=2 cases
  - Verify

Stage 1C: V=1 H>1 N-tile concatenation
  - Add output mux for H tiles
  - Test: bert_ffn1, synthetic_h2

Stage 1D: General V×H
  - Combine 1B+1C
  - Test: vgg_fc2, vgg_fc3, resnet_fc

After 1D lands, Phase 2 ENDS for FC. Move to Phase 3 (DIMM, attention head).
```

Each stage:
- Single dispatch to agent (Opus, max effort)
- Agent verifies all sanity tests PASS before reporting
- Main session sanity-checks before commit (per "no commit without explicit OK")

---

## §11 Open design choices for the dispatched agent

The dispatched agent will need to make these design calls. Documented here so
the choice is explicit:

1. **fc_top instantiation pattern**: parameterized for V/H ≥ 1 from the start
   (Stage 1A is just the V=1 H=1 special case), OR V=1 H=1-specific module
   for Stage 1A then generalize? **Recommendation: parameterized from start.**
   Then Stage 1B/1C/1D only add validation (no module surgery).

2. **Weight loading**: hierarchical force from TB (`dut.dpe_inst[v][h].weights[r][c]
   = ...`)? Or runtime weight-load via a port?
   **Recommendation: hierarchical force** — matches the primitive smoke pattern.

3. **Input streaming**: hierarchical force `dut.input_sram[m][k] = ...`, OR
   actual data-streaming through valid/ready handshake?
   **Recommendation: hierarchical force input + output reading** for Stage 1A.
   Streaming handshake is a Phase 2 enhancement (or Stage 1D).

4. **CLB activation LUT**: ReLU? Identity? Configurable?
   **Recommendation: ReLU** (`max(0, x)`) for ACTIVATION=1; identity for
   ACTIVATION=0.

5. **Output bit-width**: int8 (truncate post-MAC)? int32 (full precision)?
   **Recommendation: int8 output** — matches DPE primitive's data_out interface
   and what real CNN inference does.

---

## §12 What survives if we stop here

This plan doc, plus:
- The verified DPE primitive (`dpe_nldpe.v`, `dpe_azurelily.v`)
- The smoke harness (`run_dpe_smoke.py`, `Makefile`)
- The simulator (post-revert + principle-aligned)

Stage 1A → 1D builds on top of this. Each stage is a self-contained agent
dispatch + main-session sanity check + commit (when user OKs).
