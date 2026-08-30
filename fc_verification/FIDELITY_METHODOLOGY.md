# Simulator Fidelity Methodology

**Status:** Principle-locked; implemented in sim + RTL (Path A, single-buffered drain-load overlap, §5 workload classes — Tasks #82–#84, #99). Stage 1A–1C validated, Stage 1D in flight.
**Authored:** 2026-04-30
**Anchor for:** simulator + RTL re-org under "analytical sim, RTL ground truth, measured fidelity" framing

---

## What this document is, and isn't

This is the **principle anchor** for the simulator + RTL re-org. It pins
down the methodology decisions before any code changes. It is *not* an
implementation plan; per-stage implementation plans get written separately,
*against* this doc, after each STOP gate.

If a future change conflicts with anything here, this doc is updated
first; code follows.

---

## §1 Principle

```
SIMULATOR  =  pure analytical workload performance model
              (work-volume × lane-parallelism + DPE-axiom only)
              No FSM overhead. No handshake fudge. No drain calibration.

RTL        =  ground truth for cycle count
              (all real overhead from controllers, FSMs, SRAMs)

Fidelity   =  (RTL_cyc − Sim_cyc) / Sim_cyc
              Reported per stage, per workload, per architecture.
              No pre-committed threshold; the number is a measurement.
```

The simulator is **not** expected to predict overhead. The simulator is
a fast-but-optimistic predictor; RTL discloses overhead reality; the gap
is the simulator's measured error rate. We report this honestly.

The previous "RTL ≈ sim within Δ" framing — with classified residuals,
modelling-granularity vs structural deltas, tolerance bands, and
overhead-shaped sim constants — is **superseded**. That work calibrated
the simulator to RTL post-hoc, which made alignment partly tautological.

---

## §2 Two simulators, one workload spec

Workloads are defined architecture-agnostically (e.g., "GEMM
M=128 K=64 N=64", "Attention head N=128 d=64"). Each architecture has
its own simulator that consumes the same workload spec and emits cycles
using its own architecture parameters.

**NL-DPE simulator:**
- W concurrent DPE+tree lanes (W = `total_softmax_lanes`, e.g., 16)
- DPE primitive includes ACAM with log/exp mode

**Azure-Lily simulator:**
- DSP_WIDTH concurrent dsp_macs (DSP_WIDTH = 4 int8 pairs/cycle)
- DPE primitive is pure VMM (ADC mode only, no log/exp)

Both predictions take the form:

```
total_cycles = (number of lane-passes) × cycles_per_pass
where:
  number_of_lane_passes = ceil(total_DPE_work / (C × n_parallel_lanes))
  cycles_per_pass        = LOAD + COMPUTE + OUTPUT  (DPE-axiom)
```

`total_DPE_work` is workload-derived. `n_parallel_lanes` is
workload-allocated (see §7 Tiling). `cycles_per_pass` is per-arch
config-derived (see §3). **No FSM overhead, no calibration constants,
no handshake fudge anywhere in the simulator.**

---

## §3 DPE architecture: config-derived, single source of truth

A DPE = analog R × C crossbar + ACAM conversion stage + private CLB
reduction tree of width K (the workload's inner dim).

**ACAM has three modes** (architecture-level, programmable per-pass):
- **ADC mode**: digitize the analog VMM result (no nonlinearity).
  Used by VMM workloads.
- **Activation mode**: apply piecewise activation (ReLU, etc.).
- **Log/exp mode**: crossbar configured as C × C identity. Per pass,
  ACAM applies `exp(input[c])` or `log(input[c])` for c in 0..C-1.
  Used by DIMM workloads.

**Per-pass cost (energy + cycles) is full crossbar + full tree** —
the DPE pays for all C columns and full tree depth regardless of how
many slots are useful for the workload, and regardless of which ACAM
mode is active. This is intentional: it keeps the simulator honest
about what the hardware fundamentally costs.

**Per-architecture config JSON owns DPE primitive parameters:**

```json
// nl_dpe.json (illustrative; actual keys finalized at implementation)
{
  "dpe_buf_width":         40,
  "kernel_width":         128,
  "num_cols":             128,
  "compute_cycles":         3,
  "has_acam":            true,
  "total_softmax_lanes":   16
}

// azurelily.json
{
  "dpe_buf_width":         16,
  "kernel_width":         512,
  "num_cols":             128,
  "compute_cycles":        44,
  "has_acam":           false,
  "total_softmax_lanes":   16
}
```

**Single source of truth:** both the simulator and the DPE generator
consume the same JSON. The DPE primitive cycle accounting
(`LOAD_CYCLES = ceil(KW × 8 / BUF)`, `OUTPUT_CYCLES = ceil(C × 8 / BUF)`,
`COMPUTE_CYCLES = config_value`) is derived from config at runtime / RTL
emission time. **No hand-pinned constants in either side. No
`sram_read_latency` magic numbers.**

**DPE generator** (`nl_dpe/gen_dpe_stub.py`, to be written):

```
Inputs:   per-arch config JSON
Outputs:  fc_verification/rtl/dpe_stub_<arch>.v   (one per arch)

Both arch outputs share:
  - Same handshake protocol (w_buf_en, reg_full, nl_dpe_control,
    dpe_done, MSB_SA_Ready)
  - Same FSM (S_IDLE → S_LOAD → S_WAIT_EXEC → S_COMPUTE → S_OUTPUT
    → S_DRAIN)
  - Same behavioral 1-clock VMM at fire time
  - Same parameterized LOAD_CYCLES, OUTPUT_CYCLES derivations

Differ only in:
  - ACAM_MODE branch (log/exp computation) present iff config.has_acam == true
  - COMPUTE_CYCLES, BUF_WIDTH baked from config (no defparam)
```

The behavioral VMM stays: full-precision MAC at fire time,
non-synthesizable but functionally correct, no bit-serial decomposition.
The cycle-emulation FSM stays: load_strobes / compute_cycles /
output_cycles hold-counters. The math/timing separation already exists
in today's `dpe_stub.v` and is preserved.

### §3.1 Per-arch CCYC decomposition (Task #86)

The DPE primitive's `COMPUTE_CYCLES` parameter is the per-pass
bit-serial pipeline latency. Two architectures, two distinct internal
pipelines:

**NL-DPE** has a 2-stage internal bit-serial pipeline plus 1 cycle for
ACAM read-out at the end of each pass. **ACAM always fires** regardless
of activation mode — it is the read-out path; activation mode just
selects the LUT contents. So:

```
NL-DPE bit-serial pipeline (2 stages + 1 ACAM read-out)

Cycle:    0    1    2    3    4    5    6    7    8    9
bit b0:  MAC──Acc
bit b1:       MAC──Acc
bit b2:            MAC──Acc
bit b3:                 MAC──Acc
bit b4:                      MAC──Acc
bit b5:                           MAC──Acc
bit b6:                                MAC──Acc
bit b7:                                     MAC──Acc
                                                  └─ all slices accumulated end of cycle 8
                                                 ACAM fires cycle 9
                                                 (always, regardless of activation mode)

CCYC_NL = PRECISION + (PIPELINE_DEPTH_NL - 1) + ACAM_CYCLES
        = 8 + (2 - 1) + 1
        = 10
```

**Azure-Lily** has a 3-stage internal bit-serial pipeline (no ACAM):

```
Azure-Lily bit-serial pipeline (3 stages, no ACAM)

Cycle:    0    1    2    3    4    5    6    7    8    9
bit b0:  MAC──ADC──SA
bit b1:       MAC──ADC──SA
bit b2:            MAC──ADC──SA
bit b3:                 MAC──ADC──SA
bit b4:                      MAC──ADC──SA
bit b5:                           MAC──ADC──SA
bit b6:                                MAC──ADC──SA
bit b7:                                     MAC──ADC──SA
                                                       └─ last bit drains end of cycle 9

CCYC_AL = PRECISION + (PIPELINE_DEPTH_AL - 1) + ACAM_CYCLES
        = 8 + (3 - 1) + 0
        = 10
```

**General formula** (used by `nl_dpe/gen_dpe_stub.py`, all TBs, and
the smoke drivers):

```
CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
```

**Per-arch config keys** (under `capabilities` in the per-arch JSON):

| arch       | pipeline_depth | acam_cycles | stage description                       |
|------------|----------------|-------------|-----------------------------------------|
| NL-DPE     | 2              | 1           | (crossbar MAC, analog Acc) + ACAM       |
| Azure-Lily | 3              | 0           | (crossbar MAC, ADC, shift-add); no ACAM |

**Note (ACAM always fires for NL-DPE):** even when ACAM_MODE=0
(identity passthrough) the read-out path consumes 1 cycle. Activation
mode selects the LUT contents (identity, ReLU, exp, log), not whether
ACAM fires.

**Note (structural symmetry, not coincidence):** both arches give
`CCYC = PRECISION + 2` at every precision under current parameters.
This is structural: `(D_NL - 1) + ACAM_NL = (D_AL - 1) + ACAM_AL = 2`.
If either side's pipeline depth or ACAM latency changes, the symmetry
breaks immediately.

| Precision | NL CCYC                  | AL CCYC                | Match? |
|-----------|--------------------------|------------------------|--------|
| INT4      | 4 + (2-1) + 1 = 6        | 4 + (3-1) + 0 = 6      | yes    |
| INT8      | 8 + (2-1) + 1 = 10       | 8 + (3-1) + 0 = 10     | yes    |
| INT16     | 16 + (2-1) + 1 = 18      | 16 + (3-1) + 0 = 18    | yes    |

**Backward compatibility:** if the per-arch JSON omits the
`pipeline_depth` / `acam_cycles` keys, the generator falls back to
the legacy single-knob defaults `(3, 0)` — exactly the pre-Task-#86
behaviour where `CCYC = PRECISION + 3 - 1 = PRECISION + 2` at PD=3,
AC=0.

See `fc_verification/DPE_PRIMITIVE_WALKTHROUGH.md` §10a for a
deeper walkthrough of the two-layer model (combinational functional
VMM fire + cycle-accurate timing burn) and how the per-arch
decomposition flows from JSON → generator → emitted Verilog → TBs
→ Python drivers.

### §3.2 Input substrate layout: double-buffered slice-major (Task #99)

The DPE input substrate is organized **bit-slice-major** and
**double-buffered** — two physically separate substrates A and B that
ping-pong via a `load_phase` selector:

```
input_buf_slice_a[0..PRECISION-1][0..R-1]
input_buf_slice_b[0..PRECISION-1][0..R-1]
load_phase  : selector (1 bit) — toggles each pass; LOAD writes
              substrate indicated by load_phase, COMPUTE reads the
              other substrate
```

Each substrate is PRECISION banks of R bit cells; bank `b` holds
bit-position `b` of every row for the pass currently parked there. The
crossbar reads one bank per cycle of the substrate the COMPUTE
sub-FSM is currently tagged to (pass-tagged ring discipline).

**The LOAD interface remains byte-major.** BRAM stores byte-major data
(the natural format for host data and inter-layer activations); the
wrapper streams 5 bytes/cycle for NL (BUF=40) or 2 bytes/cycle for AL
(BUF=16). The DPE applies a **corner-turn** (fixed bit permutation) to
distribute the incoming bytes across the PRECISION bit-slice banks of
whichever substrate `load_phase` currently selects. Per LOAD cycle,
EPS bytes × 8 bits are routed as:

```verilog
// substrate := (load_phase == 0) ? slice_a : slice_b
for (j = 0; j < ELEMS_PER_STROBE; j = j + 1)            // EPS = BUF/8
  for (i_bit = 0; i_bit < PRECISION; i_bit = i_bit + 1)
    substrate[i_bit][load_cycle*EPS + j] <= data_in[j*8 + i_bit];
```

This is pure wiring + one shared address counter + one ping-pong
selector; no decode logic. The corner-turn LOAD pattern is preserved
exactly as in the single-substrate design — only the destination
substrate now alternates.

**Two input substrates, single mac/output substrates.** The input
substrate is doubled (A and B) for ping-pong; `mac_acc[0..C-1]` and
`acam_out[0..C-1]` remain single substrates (a pass-tagged ring of
depth 4 carries pass IDs through COMPUTE → OUTPUT, but the storage
itself is one bank each). Total DPE storage thus has **two input
banks + one mac bank + one output bank**. Area cost vs the
single-substrate Task #93 design: **+1× the input buffer**
(an extra PRECISION × R flops per DPE — e.g., 8 × 256 = 2048 flops for
NL R=256; 8 × 512 = 4096 flops for AL R=512). This area increment is
already accounted for in the arch budget.

**No LOAD-gate, no `load_safe` register, no WAR hazard.** Because
pass-(k+1) LOAD writes substrate B while pass-k COMPUTE reads
substrate A (and vice versa on the next pass), the two read/write
windows touch disjoint physical cells. The Task #93 LOAD-gate
(`load_safe`) and its `+PRECISION` cycle cost are retired. The honest
cycle cost of the double-buffered design is **zero extra cycles per
pass** — `T_steady` reduces to `max(LCYC, CCYC, OCYC)` (see §4).

This is the double-buffered slice-major / corner-turn design adopted
in Task #99 (superseding the Task #93 / Option A1 single-substrate +
LOAD-gate design). The `dpe_*_faithful.v` modules implement it; the
legacy `dpe_*.v` modules retain the 4-slot ring buffer pending the
faithful migration of `fc_top.v`.

### §3.3 Data transpose strategy: corner-turn LOAD into a double-buffered substrate

The DPE input substrate is **slice-major** (bit-stratified), but BRAM
storage in the rest of the system is **byte-major** (a host writes byte-
major activations, every layer's output is byte-major, every layer's
input is byte-major). At some boundary the byte→slice transpose has to
happen. The Task #99 design pairs:

**On-the-fly transpose (corner-turn LOAD):**
- LOAD applies a **corner-turn** (fixed bit permutation, zero pipeline
  depth, no extra storage) that distributes each cycle's 40 input bits
  across all 8 slice banks — 5 bits per slice at positions `5k..5k+4`
  of the substrate currently selected by `load_phase`.
- BRAM stays byte-major (universal across the whole system).
- Hardware cost in DPE outside the substrates themselves: zero beyond
  wiring.
- Hardware cost outside DPE: zero.

**Double-buffered substrate (substrates A + B, `load_phase` selector):**
- Two physically separate input substrates. Pass-(k+1) LOAD writes
  whichever substrate `load_phase` selects; pass-k COMPUTE reads the
  other one.
- Eliminates the write-after-read hazard that was unavoidable in the
  single-substrate corner-turn design (every LOAD cycle touches all
  PRECISION slices simultaneously, so pass-(k+1) writes would corrupt
  any slices pass-k COMPUTE still has to read).
- Storage cost in DPE: **2× input substrate** (one extra PRECISION × R
  flops per DPE — e.g., 2048 flops for NL, 4096 for AL). No extra
  buffer outside the DPE; no dual-port read-old/write-new semantics
  required.
- Pipeline cost: **zero extra cycles per pass.** `T_steady = max(LCYC,
  CCYC, OCYC)` — no `+PRECISION` term, no `+1` NBA-safety term, no
  inter-pass stall.

### Why the double-buffer pairing wins

| Resource | Single-substrate + LOAD-gate (retired, Task #93) | Double-buffer + corner-turn (Task #99) |
|---|---|---|
| Input substrate storage | PRECISION × R bit cells | 2 × (PRECISION × R) bit cells |
| Other DPE storage | 0 extra | 0 extra |
| BRAM format constraint | byte-major (universal) | byte-major (universal) |
| Inter-layer data flow | direct | direct |
| Substrate port arity | single-port | single-port (two substrates) |
| Per-pass cycle cost | `+PRECISION` (LOAD-gate) | 0 (overlapped on disjoint substrate) |
| Net cycles saved vs Task #93 | — | `PRECISION × (M-1)` per workload |

At our typical R values, the area trade is `2 × PRECISION × R` extra
flops per DPE — modest at the DPE level (a few KB) — in exchange for
eliminating the `+PRECISION × (M-1)` cycle cost that Task #93's
LOAD-gate paid on every multi-pass workload. The save scales linearly
with M; for M=8 it's 56 cycles, for M=128 batched attention it's 1016
cycles per DPE. With the area cost already in the arch budget, the
silicon-faithful choice for this system is to pay the area and keep
the steady-state cadence unblocked.

The methodology consequence: the pipeline formula simplifies to
`T_steady = max(LCYC, CCYC, OCYC)`. A future system that would prefer
the smaller input buffer at the cost of the LOAD-gate cycles can
revert to a single-substrate Task #93 design, and the formula updates
back to `T_steady = max(LCYC + PRECISION, CCYC, OCYC)`. Both formulas
are silicon-faithful; they describe different silicon choices.

---

## §4 Pipeline model

**Double-buffered slice-major LOAD with overlapped COMPUTE (Task #99,
superseding Task #93 / Option A1).**

Per pass:
- LOAD    (L cycles): BRAM → corner-turn → `input_buf_slice_a` or
                      `input_buf_slice_b` (selected by `load_phase`)
- COMPUTE (C_cyc cycles): bit-fire (PRECISION) + tail
                          (PIPELINE_DEPTH−1) + ACAM_CYCLES, reading
                          from the substrate not currently being
                          written
- OUTPUT  (O cycles): `acam_out` (NL) or `mac_acc` (AL) → BRAM

Storage: **two input substrates + one mac substrate + one output
substrate**, the doubled input pair matching the ping-pong discipline
needed to overlap LOAD and COMPUTE without WAR conflicts. Multi-pass
overlap is unconstrained: pass-(k+1) LOAD runs concurrently with
pass-k COMPUTE on a physically separate substrate, so the steady-state
cadence reduces to the natural pipeline-overlap maximum of LOAD,
COMPUTE, OUTPUT.

**Task #99 — double-buffered LOAD.** Pass-(k+1) writes substrate B
while pass-k COMPUTE reads substrate A; the substrates are physically
separate registers, so the write-after-read hazard that Task #93's
single substrate had to gate against (with `load_safe` and a
`+PRECISION` cost in `T_steady`) does not exist in Task #99. No
`load_safe` register, no wrapper-side inter-pass stall in `fc_top.v`,
no `+PRECISION` term in the formula.

**Steady-state interval = max(L, C_cyc, O).**

For typical configs:

| Arch | L   | C_cyc | O  | T_steady = max(L, C, O) |
|---|---|---|---|---|
| NL   | 52  | 10    | 52 | max(52, 10, 52) = **52**  |
| AL   | 256 | 10    | 64 | max(256, 10, 64) = **256**|

**Unified analytical formula at every layer** (primitive and workload,
Task #98 unification + Task #99 LOAD model):

```
T_fill        = LCYC + CCYC + OCYC                       (architectural minimum)
T_steady      = max(LCYC, CCYC, OCYC)                    (Task #99 double-buffer)
T_total(M)    = T_fill + (M − 1) × T_steady
```

**No additive constants in the formula.** The `+2` NBA sub-FSM handoff
(primitive), `TREE_PIPE` (CLB adder tree pipeline for V > 1),
`CLB_NEEDED` (activation LUT cycle), and `+6` wrapper structural
overhead all exist in real RTL but are reported as per-stage deltas
in `CYCLE_ACCOUNTING.md`, not embedded in the sim formula.

**The simulator emits this exact formula.** No fudge factors, no
calibration constants — per §1's principle.

**The RTL emits this exact formula.** Double-buffered slice-major
storage with `load_phase` ping-pong. Per-cycle behaviour is observable
via TB probes (`dut.load_phase`, `dut.compute_busy`, `dut.bit_idx_s0`).

**Fidelity:** primitive-level cycle counts match the formula exactly
(0% delta in both NL and AL faithful primitives, M ∈ {1, 2, 4, 8}) up
to the +2 NBA handoff that surfaces once in T_fill. Workload-level
cycle counts diverge by the synthesizable wrapper's structural
register overhead (paid once in T_fill, not per pass). Per-stage
breakdown in §4.2. All 12 primitive + 13 workload smoke cases PASS
under post-Task-#99 cadence; deltas decompose cleanly into
`wrap + tree + clb` with `nba = 0` at the workload layer.

For per-testcase cycle traces, see `CYCLE_ACCOUNTING.md`.
For the workload-level (FC/GEMM) formula extension, see §4.1.

### §4.1 Workload-level cycle model (FC/GEMM)

**The workload sim uses the same formula as the primitive sim**
(Task #98 unification + Task #99 double-buffered LOAD):

```
T_fill        = LCYC + CCYC + OCYC                       (architectural minimum)
T_steady      = max(LCYC, CCYC, OCYC)                    (Task #99 double-buffer)
T_wrkld(M)    = T_fill + (M − 1) × T_steady
```

The workload's architectural extras — CLB adder tree pipeline depth
(`TREE_PIPE = ⌈log₂(V)⌉`), activation LUT cycle (`CLB_NEEDED = (V > 1) OR
(activation_mode AND NOT has_acam)`), and the synthesizable wrapper's
six structural registers — are real RTL cycles **but they are reported
as per-stage deltas, not embedded in the sim formula**. See
`CYCLE_ACCOUNTING.md §6` for the per-workload delta decomposition
(`nba + tree + clb + wrap`).

**TREE_PIPE is architectural, not implementation overhead:** a
pipelined balanced adder tree of fanin V has depth ⌈log₂(V)⌉ by basic
combinational-logic theory. For V=1 (single tile) the tree is just a
pass-through, TREE_PIPE = 0. For V=2, one stage. Etc.

**CLB_NEEDED is architectural:** one CLB-stage cycle between DPE OUTPUT
and the final write to output_sram, present iff there is a CLB-side
transformation (V-fold for V>1, or activation LUT for AL+act). The
ACAM-fused activation in NL-DPE (V=1 case) eliminates that CLB cycle.

### §4.2 RTL cycles vs sim formula — per-stage delta breakdown

Under the unified formula (Task #98) with Task #99's double-buffered
LOAD, `T_fill = LCYC + CCYC + OCYC` and `T_steady = max(LCYC, CCYC,
OCYC)` are the **architectural minima** emitted by the sim at every
layer. The RTL pays additional cycles for specific named structural
reasons, each documented per-stage in `CYCLE_ACCOUNTING.md` and
summarised here.

### Primitive layer

| Workload | sim_exp | rtl_obs | delta | source |
|---|---|---|---|---|
| NL faithful (any test, any M) | LCYC + CCYC + OCYC + (M−1)·T_steady | sim_exp + 2 | **+2** | 2 NBA sub-FSM handoffs (LOAD→COMPUTE, COMPUTE→OUTPUT) |
| AL faithful (any test, any M) | same | sim_exp + 2 | **+2** | same |

All 12 primitive testcases (NL T1–T7 + AL T1–T5) show uniform `+2`
delta paid once in T_fill. T_steady is bit-exact between sim and RTL.

### Workload layer

The workload delta decomposes into four per-stage contributors:

```
   delta_total = nba + tree + clb + wrap
```

| Contributor | Value | Source |
|---|---|---|
| `nba` | 0 | Primitive's NBA handoffs absorbed by wrapper's BRAM-read pipeline + registered handshake |
| `tree` | ⌈log₂(V)⌉ | CLB adder tree pipeline depth (V > 1) |
| `clb` | 1 if (V > 1) OR (act AND NOT has_acam), else 0 | Activation LUT cycle (post-tree) |
| `wrap` | 6 | Six structural registers in `fc_top.v` (BRAM-read pipe, registered DPE handshake, sign-extend latch, BRAM-write tap, in-BRAM register, done-detect latch) |

Post-Task-#99 deltas observed across the 13-case workload smoke:

| Configuration | delta | nba | tree | clb | wrap |
|---|---|---|---|---|---|
| V=1, no act, NL+has_acam (e.g., `gemm_trivial_NL`) | **+6** | 0 | 0 | 0 | 6 |
| V=1, act, NL+has_acam (e.g., `bert_qkv_proj_NL`) | **+6** | 0 | 0 | 0 | 6 |
| V=1, act, AL+NO has_acam (e.g., `bert_qkv_proj_AL`) | **+7** | 0 | 0 | 1 | 6 |
| V>1, any act (e.g., `lenet_fc1_NL`, `gemm_v2_AL`) | **+8** | 0 | 1 | 1 | 6 |
| V=1, H>1, any act (e.g., `bert_ffn1_NL`) | **+6** | 0 | 0 | 0 | 6 |
| M>1, V=1, H=1, NL+has_acam (e.g., `gemm_batch8_NL`) | **+6** | 0 | 0 | 0 | 6 |
| M>1, V=1, H=1, AL+NO has_acam (e.g., `gemm_batched_AL`) | **+6** | 0 | 0 | 0 | 6 |

The delta is **constant in M** (`wrap + tree + clb` are all paid
once in T_fill; T_steady is bit-exact between sim and RTL because
the double-buffered LOAD removes the inter-pass stall).
| V=1, any act, H>1 (e.g., `bert_ffn1_NL`) | **+6** | 0 | 0 | 0 | 6 |

T_steady is bit-exact between sim and RTL — the delta is always paid
**once in T_fill**, never multiplied by M.

For per-testcase cycle log of all 12 primitive + 13 workload cases
with full delta decomposition, see `CYCLE_ACCOUNTING.md §4 + §6`.

**Verifier (post Task #93 / #94 / #97 / #98 / #99)**:
- 12/12 faithful primitive cases PASS (functional only); cycle delta = +2 uniform
- 13/13 fc_smoke cases PASS (functional only); cycle delta = `nba + tree + clb + wrap`
- 52/52 lazy `dpe_smoke` cases PASS (lazy primitive unchanged, retained for compile-link sanity)
- 8/8 `azurelily/IMC/test.py` sanity tests PASS with the unified sim formula

**Per-arch CCYC derivation:** `C_cyc` above is the per-arch
`COMPUTE_CYCLES`, derived from `§3.1`'s decomposition
`CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES`. For NL-DPE
(`PD=2, AC=1`) and Azure-Lily (`PD=3, AC=0`), both yield `CCYC = P + 2`
at every precision — structural symmetry, not coincidence.

Precision sweep — ideal sim T_fill (NL-DPE R=256 C=256 BUF=40, M=1,
V=1), under Task #99's double-buffered LOAD (`T_steady = max(LCYC,
CCYC, OCYC)`):

| Precision | CCYC | T_fill_ideal (= L + CCYC + O) | T_steady |
|-----------|------|--------------------------------|----------|
| INT4      | 6    | 52 + 6 + 52 = 110              | 52       |
| INT8      | 10   | 52 + 10 + 52 = 114             | 52       |
| INT16     | 18   | 52 + 18 + 52 = 122             | 52       |

The RTL pays +2 over each (T_fill_rtl primitive = 112/116/124); the
synthesizable fc_top.v wrapper adds another +4 (T_fill_wrapper =
116/120/128). Both gaps surface as fidelity.

Note: this supersedes the "Regime A vs Regime B" terminology used in
`paper/methodology/dpe_pipeline_model.md` (which was framed around an
older sim that lacked the drain-load overlap). Those labels are not
used here.

**Historical note (pre-Task #87 / pre-Task #88)**: earlier iterations
of the methodology baked the +2 (Task #87) and +4 (Task #88) into the
analytical model so that sim and RTL agreed by construction. Task #90
rolled both constants back: baking implementation-specific overhead
into the simulator is circular validation. Sim now emits the ideal
cycle count per §1 principle, RTL discloses real overhead, fidelity
is the honest measurement of the gap.

**Historical note (Task #93 — Option A1 single-substrate refactor,
superseded by Task #99)**: prior to Task #93, the faithful primitives
used a 4-slot byte-major ring buffer (a simulation convenience). Task
#93 rewrote the primitives to use single-substrate slice-major storage
with corner-turn LOAD and a `load_safe` gate, exposing the silicon
constraint as `+PRECISION` per additional pass in `T_steady = max(LCYC
+ PRECISION, CCYC, OCYC)`.

**Task #99 — double-buffered LOAD**: the LOAD-gate cost from Task #93
was paid because every corner-turn LOAD cycle touched all PRECISION
slices simultaneously, so pass-(k+1) writes could not overlap any of
pass-k COMPUTE's slice reads on the same substrate. Task #99 keeps the
corner-turn LOAD but adds a second physical input substrate
(`input_buf_slice_b`) and a `load_phase` selector that ping-pongs the
two substrates per pass. pass-(k+1) LOAD writes substrate B while
pass-k COMPUTE reads substrate A; substrates are physically separate,
so there is no WAR hazard and no LOAD-gate is required. `load_safe`
and the wrapper-side PRECISION-cycle inter-pass stall in `fc_top.v`
are both retired. The cycle cost reduces to `T_steady = max(LCYC,
CCYC, OCYC)`; the area cost is one extra PRECISION × R bit cells per
DPE (already in the arch budget). See §3.2 for the storage layout and
§3.3 for the on-the-fly transpose vs double-buffer trade.

---

## §5 Workload classes

The DPE+ACAM hardware supports two workload classes. They differ in
ACAM mode, crossbar contents, and the workload→pass mapping.

### VMM workload — weight-persistent matmul (Stage 1 GEMM, Path A)

Crossbar holds an R × C **weight matrix W**. ACAM in **ADC mode**.
Per pass: input vector of R elements × W → output vector of C elements.

For matmul Y[M × N] = X[M × K] @ W[K × N], with crossbar R × C, **Path A
weight-stationary array**:
- V = `ceil(K / R)`   (K-axis tiles)
- H = `ceil(N / C)`   (N-axis tiles)
- **V × H DPE primitives instantiated in parallel**, each holding one
  weight tile `W[v·R:(v+1)·R, h·C:(h+1)·C]` *permanently* (weight-stationary).
- For each output row m, **all V·H DPEs fire once in lockstep**.
  - DPE_(v, h) input: `X[m, v·R:(v+1)·R]` (per-v K-slice; broadcast over h)
  - DPE_(v, h) output: `partial[v, h] = sum_k X[m, k] · W[k, h·C:(h+1)·C]`
                       for `k ∈ [v·R, (v+1)·R)`
  - CLB tree across v: `Y[m, h·C:(h+1)·C] = sum_v partial[v, h]`
  - Output mux across h: stitches the H tile-columns into Y[m, 0:N]
- Per-DPE firing count = M (one fire per output row, **NOT M·V**).

**Total VMM passes per lane (Path A):**

```
n_parallel_lanes = V * H
passes_per_lane  = M × ceil(V·H / n_parallel_lanes) = M
total_cycles     = T_wrkld(M) + (1 if CLB_NEEDED else 0)
                 = T_fill_wrkld + (M − 1) × T_steady + (1 if CLB_NEEDED else 0)
                                                       per §4.1 workload formula
T_fill_wrkld     = L + C_cyc + O + TREE_PIPE          (ideal — no FSM/wrapper)
TREE_PIPE        = ⌈log₂(V)⌉ for V > 1, else 0        (architectural tree depth)
CLB_NEEDED       = (V > 1) OR (activation_mode AND !has_acam)
```

RTL pays +2 (FSM) + +4 (synthesizable wrapper) cycles on T_fill that
the simulator does not model. Surfaces as fidelity. See §4.2.

Path A claims V·H weight-stationary silicon yields T(M) latency, faster
than Path B (K-time-multiplexed) which would be T(M·V) on H DPEs. We
choose Path A because (i) AH-track precedent counts DPEs as V·H tiles;
(ii) DSE infrastructure assumes V·H silicon; (iii) real analog crossbars
are weight-stationary by physics; (iv) Path A's T(M) is faster per fixed
silicon. See `fc_verification/FC_GEMM_WALKTHROUGH.md` §13 for the full
discussion.

**+1 CLB-stage cycle** (`CLB_NEEDED`) gating rule — one CLB-side stage
between DPE OUTPUT and final output_sram write:

| (V, ACTIVATION_MODE, HAS_ACAM)              | CLB_NEEDED | Why |
|---|---|---|
| V > 1, any ACT, any HAS_ACAM                 | TRUE        | CLB tree must combine V partial sums |
| V = 1, ACT = 1, HAS_ACAM = 0  (AL+ReLU)      | TRUE        | CLB ReLU LUT (AL has no ACAM) |
| V = 1, ACT = 0, HAS_ACAM = 0  (AL no act)    | **FALSE**   | No CLB transformation needed |
| V = 1, ACT = 1, HAS_ACAM = 1  (NL+ReLU)      | FALSE       | ACAM-fused activation (note 1) |
| V = 1, ACT = 0, HAS_ACAM = 1  (NL no act)    | FALSE       | No CLB stage |

Note 1: NL-DPE behavior model writes raw VMM bytes for the V=1 ReLU case;
this is a methodology approximation. See `FC_GEMM_WALKTHROUGH.md` §6.

`imc_core.run_gemm` accepts `activation_mode` (default False) so callers
that don't pass it (e.g. plain GEMM benchmarks) won't pay an unnecessary
+1 cycle on AL.

### DIMM workload — log-domain matmul (Stage 2 Attention's mac_qk / mac_sv)

Crossbar configured as **C × C identity**. ACAM in **log/exp mode**.

Algorithm for matmul A[M × K] × B[K × N] (e.g., Q × K^T → score):
1. **Phase 1a (log A)**: convert A → log_A. Cost: M × K log ops on DPE.
2. **Phase 1b (log B)**: convert B → log_B. Cost: K × N log ops on DPE.
3. **Phase 2 (CLB add)**: for each (m, n, k), `log_A[m][k] + log_B[k][n]`. Off-DPE; CLB hardware.
4. **Phase 3 (exp + sum)**: for each (m, n, k), `exp(log_A + log_B)`, then sum_k. Cost: M × N × K exp ops on DPE; reduction in private tree.

**Total DPE work** = (M × K) + (K × N) + (M × N × K).
The exp phase dominates (factor of K over the log conversions).

---

## §6 Workload definitions

### Stage 1 — GEMM at multiple shapes (VMM workload)

Single matmul: `C[M × N] = A[M × K] @ B[K × N]`. Work = M × K × N MACs.

Shape sweep (specific shapes finalized at Stage 1 implementation, but
the spec is "general matmul, multiple shapes for scaling probes").

This stage covers what was previously called "projection" (single FC
arm), "FC", and the linear projection portions of attention. They are
all the same primitive at different shapes.

### Stage 2 — Attention head (DIMM workload)

Composition: `linear_Q + linear_K + linear_V → mac_qk → softmax → mac_sv`.
Linear_O is dropped per current sim convention (`azurelily/models/
attention.py:96-102` does not include it).

Concrete shape: N=128, d_model=128, d_head=64, W=16 softmax lanes.

The linear projections (Q/K/V) use **VMM workload**; mac_qk and mac_sv
use **DIMM workload**.

### Out of scope for this re-org

- BERT-Tiny composition (multi-head + FFN + LayerNorm + residual + embedding)
- Multi-N scaling (N=256, 512, 1024)
- d ≠ 64 regime (d=128, d=32)
- All non-attention BERT-Tiny modules

---

## §7 Tiling model

### VMM workload tiling

Path A weight-stationary V × H array: V·H DPEs in parallel, each holding
one weight tile permanently. Per output row m, all V·H DPEs fire once in
lockstep. Per-DPE firing count = M (one fire per output row). See §5 for
the pass count formula and the CLB_NEEDED gate.

Each DPE has private SRAM for its weight tile (W matrix slice) and its
output column slice. Inputs are per-v K-slices (broadcast across h);
weights are unique per (v, h). Reduction across V (K-axis tiles) happens
in a CLB adder tree; concatenation across H (N-axis tiles) is an output
mux. Both fold into the single CLB_NEEDED cycle when applicable.

### DIMM workload tiling — W-lane row-parallel with shared B + broadcast

For matmul A[M × K] × B[K × N] with W parallel DPE+tree lanes:
- Each lane owns `ceil(M / W)` rows of output (M-axis row tiling).
- A is **row-tiled** across lanes (each lane has its M/W rows of A in
  private SRAM).
- B is **shared** — single SRAM holds full B, content broadcast to all
  W lanes via fanout bus during phase 3.
- Each lane has its own private CLB reduction tree of width K.

**Why shared B (not replicated):** at W=16, replicating B would cost
~16 × (K × N + N × d) ≈ ~256KB BRAM for attention head, ≈ 37% of the
120 × 120 grid's 472 BRAMs. Sharing B with broadcast reduces this to
single-copy storage (~8KB per matrix) plus per-lane scratch — total
~9% BRAM utilization. Same cycle count (broadcast bus delivers same
data to all lanes simultaneously; W lanes work in lockstep on (n, k)
iteration but on their own m-rows).

**Per-lane DPE pass count:**

```
phase_1a = ceil((M/W × K) / C)              # log A subset, parallel across lanes
phase_1b = ceil((K × N) / C)                # log B once globally (shared)
phase_3  = ceil((M/W × N × K) / C)          # exp+sum, parallel across lanes
                                              (private tree handles reduction)

passes_per_lane = phase_1a + phase_1b + phase_3
total_cycles    = T(passes_per_lane)         per §4 pipeline model
```

For attention head N=128, d=K=64, W=16, C=128:
- phase_1a = ceil(8 × 64 / 128)   = 4 passes
- phase_1b = ceil(64 × 128 / 128) = 64 passes
- phase_3  = ceil(8 × 128 × 64 / 128) = 512 passes
- passes_per_lane = 4 + 64 + 512 = 580 passes

**Memory model (DIMM):**
- Lane-private SRAM: A subset (M/W × K), score buffer (M/W × N),
  softmax intermediate (M/W × N), output buffer.
- Shared SRAM: full B (K × N) + broadcast bus delivering one element
  per cycle to all W lanes.

**RTL implementation considerations** (forward-looking, for Stage 2 RTL):
- Synchronized (n, k) iteration FSM across W lanes (SIMD-style).
- Single shared B SRAM with W-way fanout (physical wire fanout, no
  switching network needed since all lanes consume identical data each
  cycle).
- Per-lane FSM differs only in m-axis indexing into private SRAMs.

### Why W-lane row-parallel for DIMM

1. Matches the row-parallel softmax structure already pinned (each
   softmax lane owns the same M/W rows as the upstream mac_qk lane).
2. Single allocation knob (`total_softmax_lanes`) controls W across
   mac_qk, softmax, and mac_sv — unified attention head structure.
3. Reduction tree per lane is naturally bounded by K (workload's inner
   dim), independent of M, N.
4. Scales cleanly to larger N (shared B SRAM scales as N × d, not as
   W × N × d).

---

## §8 RTL expected scope

Each top module contains:
- DPE primitive instance(s) — generated by DPE generator (§3)
- Handshake interconnect
- FSM (load → fire → compute → output → drain, double-buffered LOAD
  with overlapped COMPUTE per §4)
- Necessary storage (lane-private SRAMs + shared SRAMs per §7)

Each top module **does not** contain:
- Cosmetic alignment knobs (`BACK_TO_BACK_MODE`, `WIDE_ADDR_MODE`,
  `SCORE_BACK_TO_BACK_MODE`, `WSUM_BACK_TO_BACK_MODE`, etc.)
- Drain states added to match sim cycle counts
- Per-Q-row K/V amortization unless architecturally correct
- Parameterized fudge factors

**Rule:** if a piece of RTL exists today only because it was added
during calibration, it gets deleted in the rewrite.

---

## §9 Fidelity metric

```
fidelity = (RTL_cyc − Sim_cyc) / Sim_cyc

Positive = simulator is optimistic (under-predicts cycles)
Negative = simulator is pessimistic (over-predicts cycles)
```

Reported:
- Per architecture (NL-DPE, Azure-Lily) — separately
- Per workload shape
- Per stage (only where stages are independently observable in RTL)
- E2E (always)

**No pre-committed acceptable threshold.** We observe → diagnose → decide:
- If gap is plausible overhead (FSM glue, drain, broadcast sync, SRAM
  access): accept, document.
- If gap is large (> 30%, say): inspect — is the simulator's work
  model wrong? Is the RTL doing extra work it shouldn't?

The fidelity number is a *measurement*, not a target. The paper section
that reports it is honest about what it represents.

---

## §10 What dies, what survives

### Deprecated (delete during re-org)

| Path | Status |
|---|---|
| `fc_verification/phase{2,3,5,7}_known_deltas.json` | DELETED 102a52b |
| `fc_verification/expected_cycles.json` | DELETED 102a52b |
| `fc_verification/per_stage_targets.json` | DELETED 102a52b |
| `fc_verification/known_count_deltas.json` | DELETED 102a52b |
| `fc_verification/expected_counters.json` | DELETED 102a52b |
| `fc_verification/functional_whitelist.json` | DELETED 102a52b |
| `fc_verification/run_checks.py` | DELETED 102a52b |
| `fc_verification/run_fc_phase2.py` | DELETED 102a52b |
| `fc_verification/tb_*.v` | DELETED 102a52b |
| `fc_verification/rtl/*` (alignment-era) | DELETED 102a52b |
| `nl_dpe/gen_*` (alignment-era) | DELETED 102a52b |
| `block_comp_apr_11/rtl/setup{0..5}/fc_*.v` | DELETE / archive (deferred) |
| `fc_verification/DIMM_pipeline_model_vs_rtl.md` | ARCHIVED 102a52b |
| `fc_verification/VERIFICATION.md` | ARCHIVED 102a52b |
| Submodule (`azurelily/`) post-`c15797f` history | REVERTED ec7ccd5 |

### Concepts deprecated

- "K_id" / "K-identity" / "single-identity" / "dual-identity" — replaced
  by §3's "ACAM modes" framing and §7's tiling. Per-pass cost is full
  crossbar regardless of how many output slots are useful; the
  simulator counts work in total exp/log ops divided by C.
- "Regime A" / "Regime B" labels from `paper/methodology/dpe_pipeline_model.md`
  — replaced by §4's single pipeline model (drain-load overlap).
- `sram_read_latency` magic-number constant — removed entirely.
  No magic-number physics in the simulator.

### Survives (keep, possibly with edits)

| Path | Status |
|---|---|
| `azurelily/IMC/` workload definitions (`attention.py`, `bert_tiny.py`) | KEEP — pure-work models, principle-aligned |
| `azurelily/IMC/imc_core/` | KEEP — architecture configs |
| `azurelily/IMC/scheduler_stats/scheduler.py` | KEEP after §11 strip pass |
| Per-arch JSON configs (`nl_dpe.json`, `azurelily.json`) | KEEP, become single source of truth (§3) |
| `gemv_dse.py` (DSE driver) | KEEP — DSE flow continues, decoupled from cycle alignment |
| `nl_dpe/run_vtr.py`, `nl_dpe/area_power.py`, `nl_dpe/gen_arch_xml.py` | KEEP — VTR + area/power infrastructure unchanged |

---

## §11 Per-stage verification workflow

```
1. Workload spec confirmed (shape, parameters)
2. Simulator: predicts cycles for the workload
   (analytical: §2 formula + §4 pipeline model + §7 tiling)
3. RTL: top module designed
   (handshake + FSM + DPE primitive, minimal scope per §8)
4. RTL generator: emits the top module from architecture config
5. TB: combined functional + cycle measurement on generated RTL
   (one TB per config; functional check uses real weights/inputs and
    compares against numpy oracle; cycle measurement parses TB output)
6. Fidelity computed: (RTL_cyc − Sim_cyc) / Sim_cyc, reported
7. Inspection:
   - Is sim's work model correct per workload definition?
   - Is RTL doing only expected work per §8 scope?
   - Does the gap match plausible overhead structure?
8. Decide: ship the fidelity number, fix the work model, or both.
9. STOP gate — proceed to next stage only after explicit confirmation.
```

Every STOP is an explicit gate. Do not proceed without confirmation.

---

## §12 Submodule revert plan (Option B) — EXECUTED

**Target:** parent commit `c15797f^` (predecessor of "Phase 1 (sim):
azurelily submodule → Regime B gemm_log").

**Executed at:** parent commit `ec7ccd5` (2026-04-30). Submodule reset
to `8cae3ea` and cherry-picked: `690d7fe` (Phase 2.1 --batch M),
`65750d8` (IMC parallelism: gemm_log intra-row + gemm_dsp
parallel-output + AL softmax W=16), `993aec5` (AH softmax parallelism
fix).

**Surfaced findings during simulator review** (§11 inspection of the
post-revert simulator):

1. `total_softmax_lanes` config field referenced but missing from JSON.
2. `sram_read_latency` config field referenced but missing from JSON
   (and identified as magic-number guess — to be deleted entirely).
3. K_id / "single-identity" terminology persisting in code comments
   (resolved by §3 + §7 reformulation).
4. Softmax `cols_per_lane = cols // W` double-counts parallelism
   (correction: row-parallel only per §7).
5. `imc_core.run_gemm` latency missing `k_tile` multiplier — real bug.
6. Regime B-shaped formula leaked into `gemm_log` via cherry-pick of
   `65750d8`. Per §4, the drain-load overlap *is* the right model — so
   the pipelined formula stays, but the K_id factor in `n_passes`
   computation is dropped.

These are addressed in the §13 code work below.

---

## §13 Execution flow

The methodology is executed in this order:

1. **Cleanup.** Delete deprecated artifacts and archive historical
   docs. ✅ Done at commit `102a52b`.

2. **Simulator review.** Submodule revert + post-revert simulator
   inspection. ✅ Done at commit `ec7ccd5` + this doc's §12.

3. **Simulator code work** (next). Apply the surfaced findings to the
   reverted simulator: drop K_id, drop `sram_read_latency`, fix
   `run_gemm` k_tile bug, cleanup softmax, add `total_softmax_lanes`
   to configs. Code plan to be discussed and locked before changes.

4. **Behavioral DPE generator.** Author `nl_dpe/gen_dpe_stub.py`
   per §3. Generator emits `dpe_stub_nldpe.v` and
   `dpe_stub_azurelily.v` from per-arch config JSON.

5. **DPE functional verification.** A simple TB drives each
   generated DPE with weights and inputs, checks the VMM result
   matches a numpy oracle. No top module yet.

6. **Stage 1 (GEMM).** Per §6 + §11: top module design, generator,
   TB, fidelity measurement, inspection.

7. **Stage 2 (Attention head).** Same pattern, composing the
   verified GEMM primitive into the attention pipeline.

Each step finishes before the next starts. Inspection at every
boundary is part of the methodology, not an exception to it.

---

## Glossary

- **Work-volume:** number of MACs (or higher-level ops) the workload
  fundamentally requires. Independent of architecture.
- **Lane parallelism:** number of MACs an architecture computes per
  cycle in steady state. Architecture-specific.
- **DPE-axiom:** per-DPE-fire latency =
  `LOAD_CYCLES + COMPUTE_CYCLES + OUTPUT_CYCLES`. Derived from
  per-arch config; physics-bound.
- **Pipeline model:** §4 — double-buffered slice-major LOAD with
  overlapped COMPUTE (Task #99). Both sim and RTL implement this.
  Cycle delta = FSM/control overhead.
- **VMM workload:** weight-persistent matmul; ACAM in ADC mode;
  crossbar holds W matrix.
- **DIMM workload:** log-domain matmul; ACAM in log/exp mode;
  crossbar configured as C × C identity.
- **W lanes:** parallel DPE+tree units (config field
  `total_softmax_lanes`). Each lane owns M/W rows of DIMM output and
  the same M/W rows of softmax output downstream.
- **Shared B + broadcast:** §7 tiling pattern for DIMM. B held in
  single shared SRAM; broadcast bus fans out to all W lanes
  simultaneously (lockstep (n, k) iteration; lanes differ only on
  m-axis).
- **Fidelity:** relative under-prediction of the simulator vs RTL.
  `(RTL − Sim) / Sim`, reported per stage / workload / arch.
- **Overhead:** cycles spent in FSM transitions, handshake gaps,
  drain states, broadcast sync, etc. **Not modeled in simulator;**
  captured by RTL; reflected in fidelity number.
- **STOP gate:** explicit confirmation point before proceeding to
  the next sequence step.
