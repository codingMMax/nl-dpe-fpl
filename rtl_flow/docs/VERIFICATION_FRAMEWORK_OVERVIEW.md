# FC/GEMM Verification Framework — Architecture and Flow Overview

**Document role.** Single-page architectural reference for the FC/GEMM RTL
verification framework that gates the NL-DPE and Azure-Lily behavior models
against their analytical simulator counterparts. Each numbered section is
self-contained and intended to lift cleanly onto a meeting slide.

**Project context.** The framework supports the paper-target comparison of
two FPGA hard-block architectures: NL-DPE (analog crossbar with ACAM tail
stage) and Azure-Lily (DSP-based with 3-stage pipeline, no ACAM). Verification
must establish that the simulator's energy/latency formulas faithfully
reflect what the corresponding RTL would execute.

**Status (2026-05-18, post Task #93/#94/#97/#98/#99 — unified formula
+ double-buffered LOAD).**

| Layer | DUT | Modes / Cases | Result |
|---|---|---|---|
| Primitive (NL faithful)  | `dpe_nldpe_faithful.v`     | T1–T7   | **7/7 PASS functional**; delta = +2 (NBA handoffs) |
| Primitive (AL faithful)  | `dpe_azurelily_faithful.v` | T1–T5   | **5/5 PASS functional**; delta = +2 (NBA handoffs) |
| Workload (FC, faithful)              | `fc_top.v` | Stages 1A+1B+1C, 13 cases | **13/13 PASS functional**; delta ∈ {+6, +7, +8} = `wrap + tree + clb` |
| **Task #99 — double-buffered LOAD** | `dpe_*_faithful.v` + `fc_top.v` | LOAD-gate removed | RTL + sim formula updated; **8/8 sanity + 52/52 primitive + 13/13 FC PASS** |

**Methodology unification (Task #98) + double-buffered LOAD (Task #99)**:
all sim formulas across layers use **`T_fill = LCYC + CCYC + OCYC`** —
the architectural minimum, with no additive constants. Under Task #99
the input substrate is double-buffered (substrates A/B + `load_phase`
selector), so `T_steady = max(LCYC, CCYC, OCYC)` — no `+PRECISION`
LOAD-gate term, no `+2` NBA handoffs, no `TREE_PIPE`, no `CLB_NEEDED`,
no `+6` wrapper embedded in the formula. Differences between sim and
RTL are observed and decomposed per-stage in
[`CYCLE_ACCOUNTING.md`](./CYCLE_ACCOUNTING.md):

- Primitive delta = `+2` (NBA handoffs in T_fill; bit-exact T_steady)
- Workload delta = `nba(0) + tree(⌈log₂V⌉) + clb(act-cond) + wrap(6)` (concrete numbers refreshed in §9.2 after the Task #99 RTL re-run)

---

## 1. Verification Philosophy

### 1.1 Independence Invariant

The framework decouples four code paths that would, if shared, produce
tautological "verification":

1. **Architecture configuration** — JSON files in `nl_dpe/dpe_*.json`. Single
   source of truth for `R`, `C`, `BUF`, `PRECISION`, `PIPELINE_DEPTH`,
   `ACAM_CYCLES`, and ACAM mode list.
2. **Generator** — Python emitter (`nl_dpe/gen_dpe_*_faithful.py`) that reads
   the JSON and writes RTL.
3. **Oracle** — Python reference (`fc_verification/oracles/*_mac_oracle.py`)
   that computes expected outputs and cycle counts from first principles
   using numpy, never reading RTL or sim code.
4. **DUT** — generator-emitted Verilog compiled under iverilog.

The Testbench consumes (3) via `$readmemh` of pre-generated `.mem` files
and the DUT via instantiation, then compares the two. The oracle never
imports RTL or sim modules; the TB never invokes Python at runtime.
The generator and oracle communicate only via the JSON.

### 1.2 Avoiding Circular Validation (Option A Roll-Back + Task #99 double-buffered LOAD)

A previous iteration declared compute cycles via a Verilog parameter
`COMPUTE_CYCLES`, which was also hard-coded into the simulator's cycle
formula. The fidelity check then reported 0% by construction. The current
framework forbids this construction:

- **Oracle formula:** `cycles = PRECISION + (PIPELINE_DEPTH-1) + ACAM_CYCLES`.
- **RTL behavior:** cycle count emerges from physical bit-counter advances
  through the COMPUTE sub-FSM. No `COMPUTE_CYCLES` parameter exists.
- **Reported fidelity:** the honest 0%–9% gap. For the **faithful**
  primitives (post Task #93, refined under Task #99), oracle and RTL
  agree bit-exact (0%). For the workload layer (still bound to the lazy
  primitive in `fc_top.v`), the fidelity is non-zero and **diagnostic**
  — it tells you exactly which substrate hasn't been migrated to
  silicon-faithful storage yet.

### 1.3 Cycle Emergence vs. Declaration

The framework enforces a structural separation between *what the device
computes* (functional output) and *how long it takes* (cycle count):

- Functional output is checked at output drain time against
  `expected_mac_*.mem` (the oracle's numpy result).
- Cycle count is measured directly by the TB using `$time`-derived
  counters bound to internal state transitions. The TB compares
  `last_compute_cycle - first_compute_cycle + 1` against
  `expected_cycles_*.txt`.
- The CCYC formula `PRECISION + (PIPELINE_DEPTH-1) + ACAM_CYCLES` is a
  *prediction* derived analytically from the architecture topology, not
  a target written into the RTL. NL evaluates this as `8+1+1=10`; AL as
  `8+2+0=10`. Both reach 10 by different physical decompositions.

---

## 2. Two-Layer Architecture

The framework decomposes verification into two gated layers. The workload
layer is meaningful only after the primitive layer passes.

| Layer | DUT | TB | Oracle | Coverage |
|---|---|---|---|---|
| **Layer 1 — Primitive** | `dpe_nldpe_faithful.v`, `dpe_azurelily_faithful.v` | `tb_dpe_nldpe_faithful.v`, `tb_dpe_azurelily_faithful.v` | `nldpe_mac_oracle.py`, `azurelily_mac_oracle.py` | Single DPE block: functional, cycle emergence, M-sweep, ACAM modes (NL), signed inputs |
| **Layer 2 — Workload** | `fc_top.v` instantiating `dpe` V×H grid | `tb_fc.v` | Numpy GEMM inside TB driver | FC/GEMM layer with K-tile reduction (V>1) and N-tile concatenation (H>1) |

**Module name contract.** All primitive variants — `dpe_nldpe_faithful.v`,
`dpe_azurelily_faithful.v`, the legacy lazy primitives, and the synthesizable
wrapper — expose the identical module name `dpe` with identical port lists.
This matches the VTR architecture XML `<model name="dpe">` contract and
allows `fc_top.v` to bind any primitive variant unchanged.

---

## 3. Component Inventory

| Path | Role | Layer |
|---|---|---|
| `nl_dpe/dpe_nldpe.json`               | NL arch config (R=256, C=256, BUF=40, PIPELINE_DEPTH=2, ACAM_CYCLES=1) | source |
| `nl_dpe/dpe_azurelily.json`           | AL arch config (R=512, C=128, BUF=16, PIPELINE_DEPTH=3, ACAM_CYCLES=0) | source |
| `nl_dpe/gen_dpe_nldpe_faithful.py`    | NL primitive generator | generator |
| `nl_dpe/gen_dpe_azurelily_faithful.py`| AL primitive generator | generator |
| `fc_verification/oracles/nldpe_mac_oracle.py`        | NL numpy oracle + test vector emitter | oracle |
| `fc_verification/oracles/azurelily_mac_oracle.py`    | AL numpy oracle + test vector emitter | oracle |
| `fc_verification/oracles/test_vectors/`              | NL test vectors (`.mem`, `.txt`) | vectors |
| `fc_verification/oracles/test_vectors_al/`           | AL test vectors (`.mem`, `.txt`) | vectors |
| `fc_verification/rtl/dpe_nldpe_faithful.v`           | NL primitive RTL (DUT) | Layer 1 |
| `fc_verification/rtl/dpe_azurelily_faithful.v`       | AL primitive RTL (DUT) | Layer 1 |
| `fc_verification/tb_dpe_nldpe_faithful.v`            | NL primitive TB (T1–T7) | Layer 1 |
| `fc_verification/tb_dpe_azurelily_faithful.v`        | AL primitive TB (T1–T5) | Layer 1 |
| `fc_verification/rtl/fc_top.v`                       | FC/GEMM workload wrapper | Layer 2 |
| `fc_verification/tb_fc.v`                            | FC workload TB with embedded numpy-style reference | Layer 2 |
| `fc_verification/run_dpe_smoke.py`                   | Primitive smoke orchestrator | orchestration |
| `fc_verification/run_fc_smoke.py`                    | Workload smoke orchestrator | orchestration |
| `fc_verification/Makefile`                           | iverilog/vvp build harness with CLI knobs | orchestration |
| `fc_verification/FIDELITY_METHODOLOGY.md`            | Canonical methodology anchor (referenced, not generated); §3.2 storage layout, §4 pipeline model | reference |
| `fc_verification/CYCLE_ACCOUNTING.md`                | Per-testcase cycle log trace + cycle decomposition tables | reference |
| `fc_verification/DPE_NLDPE_FAITHFUL_WALKTHROUGH.md`  | NL primitive detail walkthrough with numerical example | reference |
| `fc_verification/DPE_AZURELILY_FAITHFUL_WALKTHROUGH.md` | AL primitive detail walkthrough with numerical example | reference |

---

## 4. Layer 1 — Primitive Verification Flow

### 4.1 Data Flow Diagram

```
              ARCH CONFIG (single source of truth)
              ────────────────────────────────────
  nl_dpe/dpe_nldpe.json              nl_dpe/dpe_azurelily.json
              │                                │
       ┌──────┴──────┐                  ┌──────┴──────┐
       ▼             ▼                  ▼             ▼
  ┌─────────┐  ┌─────────┐         ┌─────────┐  ┌─────────┐
  │GENERATOR│  │ ORACLE  │         │GENERATOR│  │ ORACLE  │
  │  NL.py  │  │  NL.py  │         │  AL.py  │  │  AL.py  │
  └────┬────┘  └────┬────┘         └────┬────┘  └────┬────┘
       │ writes     │ writes            │ writes     │ writes
       ▼            ▼                   ▼            ▼
  ┌────────┐  ┌─────────────┐       ┌────────┐  ┌──────────────┐
  │NL DUT  │  │test_vectors/│       │AL DUT  │  │test_vectors_ │
  │  .v    │  │  .mem .txt  │       │  .v    │  │  al/         │
  └───┬────┘  └─────┬───────┘       └───┬────┘  └──────┬───────┘
      │             │ $readmemh         │              │ $readmemh
      │ instantiate │                   │ instantiate  │
      ▼             ▼                   ▼              ▼
  ┌───────────────────────┐         ┌───────────────────────┐
  │ tb_dpe_nldpe_faith.v  │         │ tb_dpe_azurelily_*.v  │
  │   T1 identity         │         │   T1 identity         │
  │   T2 random           │         │   T2 random           │
  │   T3 cycle emergence  │         │   T3 cycle emergence  │
  │   T4 M-sweep          │         │   T4 M-sweep          │
  │   T5 ACAM mode 1      │         │   T5 signed inputs    │
  │   T6 ACAM mode 2      │         │                       │
  │   T7 signed inputs    │         │                       │
  └──────────┬────────────┘         └──────────┬────────────┘
             │ iverilog + vvp                  │ iverilog + vvp
             ▼ stdout                          ▼ stdout
             └────────── parsed by ────────────┘
                              ▼
                  ┌───────────────────────┐
                  │ run_dpe_smoke.py +    │
                  │ Makefile harness      │
                  │ Aggregate PASS counts │
                  └───────────────────────┘
```

### 4.2 Component Roles

**Generator** (`gen_dpe_*_faithful.py`). Reads the per-arch JSON and emits a
Verilog file that implements the bit-serial pipeline structure. The generator
hard-codes the *topology* (number of stages, ACAM presence) but parameterizes
the *dimensions* (R, C, BUF, PRECISION). Re-running the generator after a
JSON edit regenerates a matching RTL file.

**Oracle** (`*_mac_oracle.py`). A pure-numpy reference. Two exported entry
points:

- `signed_int8_mac(weights_2d, inputs_1d) -> int32_vec`. Casts to int32,
  computes the dot product, applies signed truncation. Independent of any
  Verilog semantics.
- `expected_total_cycles(R, C, BUF, M, PRECISION, PIPELINE_DEPTH, ACAM_CYCLES)`.
  Closed-form:
  - `T_fill = LCYC(R) + PRECISION + (PIPELINE_DEPTH-1) + ACAM_CYCLES + OCYC(C) + 2`
    (where `+2` = LOAD→COMPUTE + COMPUTE→OUTPUT handoff cycles)
  - `T_steady = max(LCYC(R), CCYC, OCYC(C))` (Task #99 double-buffered LOAD)
  - `total = T_fill + (M-1) * T_steady`

NL oracle adds `acam_transform(mac_int32, mode)` for the three ACAM modes
(0: identity, 1: exp via `1+x+x²/2`, 2: log via `x-1`). AL oracle omits
this — Azure-Lily has no ACAM stage.

**Test vector emitter** (`--gen-test-vectors` flag on the oracle scripts).
For each test mode, writes a deterministic set of files into
`test_vectors/` or `test_vectors_al/`:

- `weights_<modeID>.mem` — R×C int8 weight matrix in hex.
- `inputs_<modeID>.mem` — input vector(s), one row per pass.
- `expected_mac_<modeID>.mem` — int32 expected MAC outputs.
- `expected_cycles_<modeID>.txt` — single integer, the predicted total
  cycle count.

NL emits 36 files; AL emits 26 files.

**DUT** (`dpe_*_faithful.v`). The behavior model under test. Implements
three concurrent sub-FSMs (LOAD, COMPUTE, OUTPUT) coordinating via
**double-buffered slice-major storage** (Task #99, superseding Task
#93's single-substrate + LOAD-gate design):

- `input_buf_slice_a[0..PRECISION-1][0..R-1]` and
  `input_buf_slice_b[0..PRECISION-1][0..R-1]` — two physically separate
  PRECISION bit-slice banks of R bit cells, ping-pong'd by a `load_phase`
  selector. LOAD writes one substrate via a corner-turn (fixed bit
  permutation) while COMPUTE reads the other.
- `mac_acc[0..C-1]` — single int32 accumulator bank (pass-tagged ring
  carries pass IDs forward).
- `acam_out[0..C-1]` (NL only) — single int32 output bank.

The two input substrates provide WAR-free pipeline overlap: pass-(k+1)
LOAD runs concurrently with pass-k COMPUTE because they write/read
disjoint physical cells. No `load_safe` register is required;
`T_steady = max(LCYC, CCYC, OCYC)`. Area cost: 2× input buffer flops
per DPE (e.g., 2048 flops for NL R=256, 4096 for AL R=512).

Compute uses a bit-serial accumulator with MSB-subtract for signed
multiply. NL has a 2-stage compute pipeline plus a 1-cycle ACAM stage;
AL has a 3-stage pipeline (crossbar → ADC → shift-add) and no ACAM.

**Testbench** (`tb_dpe_*_faithful.v`). For each test mode:

1. Load `weights_*.mem`, `inputs_*.mem`, `expected_mac_*.mem`,
   `expected_cycles_*.txt` via `$readmemh` and `$readmemb`.
2. Drive `start`, feed inputs into the DUT, observe `done` and `data_out`.
3. Compare DUT output against `expected_mac_*.mem`.
4. Measure cycle count between `first_compute_cycle` and `last_compute_cycle`
   using internal state-machine probes; compare against
   `expected_cycles_*.txt`.
5. Emit `T# PASS` or `T# FAIL <detail>` to stdout.

**Smoke orchestrator** (`run_dpe_smoke.py`). Iterates over (arch, test
mode) combinations, calls iverilog+vvp via `make`, parses stdout for
`PASS`/`FAIL` markers, aggregates a final count.

### 4.3 Test Mode Coverage

| Mode | NL | AL | Purpose |
|---|---|---|---|
| T1 | yes | yes | Identity weights, identity inputs — minimal functional sanity |
| T2 | yes | yes | Random weights and inputs — broad functional coverage |
| T3 | yes | yes | Cycle emergence — assert `MEASURED_CCYC == PRECISION + (PIPELINE_DEPTH-1) + ACAM_CYCLES` |
| T4 | yes | yes | M-sweep (M ∈ {1, 2, 4, 8}) — assert pipelined throughput formula |
| T5 | yes (ACAM mode 1, exp) | yes (signed inputs) | Arch-specific |
| T6 | yes (ACAM mode 2, log) | — | NL ACAM mode 2 |
| T7 | yes (signed inputs) | — | NL signed inputs |

T3 is the load-bearing emergence test: the TB asserts that the measured
cycle count equals the architecturally-predicted value, with no
parameter shortcut between them.

---

## 5. Layer 2 — Workload Verification Flow

### 5.1 Data Flow Diagram

```
       ┌─────────────────────────────────────┐
       │ Primitive RTL (verified by Layer 1) │
       │  module dpe (NL or AL variant)      │
       └────────────────┬────────────────────┘
                        │ instantiated as
                        ▼  V × H grid
       ┌─────────────────────────────────────┐
       │ fc_top.v   (M, K, N, V, H params)   │
       │  + CLB adder tree   (V > 1)         │
       │  + N-tile output mux (H > 1)        │
       │  + ReLU truncation                  │
       └────────────────┬────────────────────┘
                        │ DUT in
                        ▼
       ┌─────────────────────────────────────┐
       │ tb_fc.v                             │
       │   - hierarchical-force weights      │
       │   - drives X stream                 │
       │   - cycle counters                  │
       │   - embedded numpy-style reference  │
       │     computes Y_ref                  │
       │   - compares Y_dut vs Y_ref         │
       └────────────────┬────────────────────┘
                        │ stdout
                        ▼
       ┌─────────────────────────────────────┐
       │ run_fc_smoke.py                     │
       │   13 cases, Stages 1A + 1B + 1C     │
       └─────────────────────────────────────┘
```

### 5.2 Component Roles

**Workload wrapper** (`fc_top.v`). Parameterized GEMM top with five
elaboration parameters: `M, K, N, V, H`. Instantiates `V × H` copies of
the `dpe` primitive. V > 1 indicates K-tile reduction (multiple primitives
contribute partial sums along the K dimension); H > 1 indicates N-tile
concatenation (multiple primitives produce disjoint slices of the N
dimension). The wrapper synthesizes the CLB adder tree for V-axis
reduction and the output mux for H-axis concatenation. Cycle 1 of the
post-DPE chain applies ReLU truncation.

**Workload TB** (`tb_fc.v`). Differs from primitive TBs in that the
oracle is *embedded* in the TB rather than supplied via `.mem` files.
Reason: workload-level vectors would be huge (multi-megabyte for VGG
workloads) and the oracle math at GEMM level is trivially numpy-style
loops the TB can compute itself. The TB:

1. Generates the weight matrix and X stream procedurally per case.
2. Force-loads weights into each DPE's weight memory via hierarchical
   reference (`tb_fc.dut.dpe_inst[v][h].weights`).
3. Drives X cycle-by-cycle.
4. Computes the reference GEMM in TB-local logic in parallel.
5. Compares observed Y against reference Y at output drain.

**Workload smoke orchestrator** (`run_fc_smoke.py`). 13 named cases
covering the Stage 1A subset (V=1, H=1), Stage 1B (V>1, H=1, K-tile
reduction), and Stage 1C (V=1, H>1, N-tile concatenation). Stage 1D
(V>1, H>1) is pending.

### 5.3 Case Coverage

| Stage | V | H | Cases | Validates |
|---|---|---|---|---|
| 1A | 1 | 1 | 7 | Single-DPE pass-through |
| 1B | >1 | 1 | 3 | CLB adder tree (V-axis reduction) |
| 1C | 1 | >1 | 3 | Output mux (H-axis concatenation) |
| 1D | >1 | >1 | (pending) | Combined K-tile + N-tile |

---

## 6. Test Vector Lifecycle

### 6.1 Generation

The oracle scripts are run with `--gen-test-vectors` during framework
setup or after architectural parameter changes:

```
python fc_verification/oracles/nldpe_mac_oracle.py --gen-test-vectors
python fc_verification/oracles/azurelily_mac_oracle.py --gen-test-vectors
```

This produces deterministic outputs from numpy. Re-running yields
byte-identical files unless the JSON config or oracle code has changed.

### 6.2 Storage Format

For test mode N (e.g., N=2 for "random"):

```
test_vectors/weights_T2.mem        # one int8 per line in hex, row-major R×C
test_vectors/inputs_T2.mem         # one int8 per line, M rows of R inputs each
test_vectors/expected_mac_T2.mem   # one int32 per line, M rows of C outputs each
test_vectors/expected_cycles_T2.txt # single integer
```

NL and AL test vector sets are disjoint: vector dimensions match each
arch's R×C, so the same nominal test mode produces different file
contents for the two archs.

### 6.3 Consumption by TB

The TB instantiates `reg [W-1:0] mem [0:DEPTH-1]` arrays and loads them
inside an `initial` block:

```verilog
initial begin
    $readmemh("test_vectors/weights_T2.mem", w_mem);
    $readmemh("test_vectors/inputs_T2.mem",  in_mem);
    $readmemh("test_vectors/expected_mac_T2.mem", exp_mem);
    // expected_cycles_T2.txt is read into a parameter via $fscanf
end
```

The TB never invokes Python at runtime — vectors are static artifacts.
This isolates simulation reproducibility from any Python version drift.

---

## 7. Cycle Verification Method (Post Task #99 — double-buffered LOAD)

### 7.1 Oracle Formula

The closed-form cycle prediction:

```
T_fill   = LCYC(R) + PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES + OCYC(C) + 2
T_steady = max(LCYC(R), CCYC, OCYC(C))
total    = T_fill + (M - 1) * T_steady
```

Components:
- `LCYC(R)` — load cycles to stream R · PRECISION bits into input buffer.
- `PRECISION + (PIPELINE_DEPTH - 1)` — bit-serial compute pipeline fill.
- `ACAM_CYCLES` — ACAM tail stage (1 for NL, 0 for AL).
- `OCYC(C)` — output drain cycles to stream C output values.
- `+2` — LOAD→COMPUTE handoff + COMPUTE→OUTPUT handoff (NBA register settle).
- No `+PRECISION` term: under Task #99 the input substrate is
  double-buffered, so pass-(k+1) LOAD overlaps pass-k COMPUTE on a
  physically separate substrate. The Task #93 LOAD-gate cost is
  retired; see FIDELITY_METHODOLOGY.md §3.2 and §4.

The "CCYC" reported in T3 is the compute-only slice:
`CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES = 10` for both
NL and AL at INT8.

Concrete values (post Task #99):
- NL (R=256, C=256, BUF=40): T_fill = 116, T_steady = 52
- AL (R=512, C=128, BUF=16): T_fill = 332, T_steady = 256

### 7.2 RTL Measurement

The TB instruments the DUT with two cycle stamps:

- `first_compute_cycle` — recorded at the first transition into the
  COMPUTE active state for the first pass.
- `last_compute_cycle` — recorded at the last transition out of COMPUTE
  active for the final pass.

`MEASURED_CCYC = last_compute_cycle - first_compute_cycle + 1`.

The instrumentation taps internal sub-FSM state, not the externally
visible `done` signal. This is critical: if `done` were used, the
measurement would include OCYC and would not isolate the compute pipeline.

### 7.3 Fidelity Reporting

For each mode, T3 prints:

```
T3 MEASURED CCYC = 10
T3 DECLARED CCYC_ORACLE = 10 (= 8 + 1 + 1 for NL or 8 + 2 + 0 for AL)
T3 *** CCYC=10 EMERGED from physical structure (NOT set as parameter) ***
T3 PASS
```

The "EMERGED" annotation confirms that the RTL has no `COMPUTE_CYCLES`
parameter — the value comes from physically counting bit-serial stage
advances.

**Faithful primitive fidelity: 0% bit-exact** for all 12 testcases (NL
T1–T7 + AL T1–T5), including all M-sweep cases (M ∈ {1, 2, 4, 8}).

**Workload-level fidelity** is per-stage decomposed (`nba + tree + clb
+ wrap`) once paid in T_fill; `T_steady` matches sim bit-exactly under
the Task #99 double-buffered LOAD. The wrapper-side PRECISION-cycle
inter-pass stall that existed under Option A1 has been removed from
`fc_top.v`. Concrete cycle counts are being refreshed in the main
session after the Task #99 RTL re-run.

See [`CYCLE_ACCOUNTING.md`](./CYCLE_ACCOUNTING.md) for per-testcase
cycle traces and the full breakdown of where each cycle comes from.

---

## 8. Independence Invariants — Enforcement Summary

| Invariant | Mechanism | Failure mode prevented |
|---|---|---|
| Single source of truth for arch params | JSON consumed by both generator and oracle | Drift between RTL dimensions and oracle dimensions |
| Oracle is independent of RTL | Pure numpy, no Verilog import, no parameter sharing | Tautological pass when both encode the same formula |
| TB independent of oracle internals | Consumes `.mem` artifacts only; no Python at simulation time | Brittle coupling to oracle implementation |
| Cycle counts measured, not declared | TB instruments internal sub-FSM transitions | Spurious 0% fidelity from hardcoded `COMPUTE_CYCLES` |
| Module name contract across primitives | All `dpe_*_faithful.v` and the legacy primitives expose `module dpe (...)` with identical port list | `fc_top.v` rebinding required when swapping primitives |
| Workload TB independent of primitive internals | Hierarchical force at named ports/regs only | Coupling that would break when faithful migration completes |

---

## 9. Verification Status (2026-05-18)

### 9.1 Primitive Layer (Faithful, post Task #93 / Task #99)

| Arch | DUT | TB | Modes | Result | Notes |
|---|---|---|---|---|---|
| NL faithful | `dpe_nldpe_faithful.v` | `tb_dpe_nldpe_faithful.v` | T1–T7 | **7/7 PASS, 0% fidelity** | T_fill=116; T_steady=52 (post #99); CCYC=10 emerged; all 3 ACAM modes pass; signed inputs validated; M-sweep cycles refreshed by main session |
| AL faithful | `dpe_azurelily_faithful.v` | `tb_dpe_azurelily_faithful.v` | T1–T5 | **5/5 PASS, 0% fidelity** | T_fill=332; T_steady=256 (post #99); CCYC=10 emerged; signed inputs validated; M-sweep cycles refreshed by main session |

### 9.2 Workload Layer (Faithful primitive via `fc_top.v`, post Task #97/#98/#99)

Under Task #99 the workload sim formula is `T_fill = LCYC + CCYC + OCYC`
and `T_steady = max(LCYC, CCYC, OCYC)` (no `+PRECISION` LOAD-gate term;
`fc_top.v` has no wrapper-side inter-pass stall). The per-stage delta
breakdown `delta = nba(0) + tree + clb + wrap` is paid once in T_fill
(T_steady is bit-exact). Observed across the 13-case FC smoke
post-Task-#99 re-run:

| Stage | V | H | Cases | Result | Cycle delta breakdown |
|---|---|---|---|---|---|
| 1A | 1 | 1 | 7 | 7/7 PASS | **+6** for NL & AL+none (`wrap=6`); **+7** for AL+relu (`clb=1, wrap=6`) |
| 1B | >1 | 1 | 3 | 3/3 PASS | **+8** (`tree=1, clb=1, wrap=6`) for V=2 cases |
| 1C | 1 | >1 | 3 | 3/3 PASS | **+6** uniform (H-axis combinational; no extra cycle) |
| 1D | >1 | >1 | pending | — | Combined K-tile + N-tile; Task #72 follow-up |

For M>1 cases the delta is **constant in M** (e.g.
`gemm_batch8_NL` M=8 still shows `delta=+6`): T_steady = 52 cycles is
bit-exact between sim and RTL, so each additional pass adds the same
52 cycles to both. The pre-Task-#99 wrapper stall used to add
`(M − 1) × PRECISION = 7 × 8 = 56` cycles per M=8 workload — those
cycles are now reclaimed.

**Delta decomposition**: `delta = nba(0) + tree + clb + wrap`. `nba = 0`
at workload (primitive NBA handoffs absorbed by wrapper). `tree =
⌈log₂(V)⌉`. `clb = 1` if `(V > 1) OR (act AND NOT has_acam)`. `wrap = 6`
constant (six structural registers in `fc_top.v`).

See `CYCLE_ACCOUNTING.md §4` for the per-testcase delta breakdown.

### 9.3 Outstanding Work

- **Stage 1D.** Combined V>1, H>1 with workloads vgg_fc3 (V=16, H=4) and
  resnet_fc (V=2, H=4).
- **DIMM RTL (Task #73).** Behavioral DIMM module instantiated against
  the faithful primitives, per `paper/methodology/attention_dimm_mapping.md`.
- **Attention head RTL (Task #74).** Composed attention head, end-to-end
  alignment at N=128, d=64, C=128, W=16.

---

## 10. Suggested Slide Mapping

| Slide | Source section | Purpose |
|---|---|---|
| 1. Title + scope | Document header | Frame the talk |
| 2. Why this matters | §1.2 + §1.3 | Motivate independence; recall the prior circular-validation pitfall |
| 3. Two-layer decomposition | §2 (table) | High-level architecture |
| 4. Layer 1 dataflow | §4.1 (diagram) | Show primitive verification path |
| 5. Layer 1 components | §4.2 | Walk through generator → oracle → DUT → TB |
| 6. Layer 1 coverage | §4.3 (table) | Test modes T1–T7 / T1–T5 |
| 7. Layer 2 dataflow | §5.1 (diagram) | Show workload verification path |
| 8. Layer 2 case coverage | §5.3 (table) | Stages 1A → 1D matrix |
| 9. Cycle emergence | §7 | The "CCYC=10 emerged" claim |
| 10. Independence enforcement | §8 (table) | Six invariants and how each is enforced |
| 11. Current results | §9 | What is and isn't gated |
| 12. Forward plan | §9.3 | Faithful migration, Stage 1D, DIMM, attention head |

---

## Appendix A — Combined Dataflow Diagram

```
                        ┌──────────────────────────┐
                        │   ARCH CONFIG (JSON)     │
                        │   nl_dpe/dpe_*.json      │
                        └──────────┬───────────────┘
                                   │ read by
                ┌──────────────────┼──────────────────┐
                ▼                                     ▼
         ┌─────────────┐                      ┌─────────────┐
         │ GENERATORS  │                      │   ORACLES   │
         │ (Python)    │                      │ (Python)    │
         └──────┬──────┘                      └──────┬──────┘
                │ emit                                │ emit
                ▼                                     ▼
         ┌─────────────┐                      ┌──────────────┐
         │ DPE RTL     │                      │ Test vectors │
         │ .v files    │                      │ .mem .txt    │
         └──────┬──────┘                      └──────┬───────┘
                │ instantiated in                    │ $readmemh
                │                                    │
                ▼                                    ▼
         ┌────────────────────────────────────────────┐
         │             PRIMITIVE TBs                  │
         │   (NL: T1–T7,  AL: T1–T5)                  │
         │   compare DUT vs vectors; emit PASS/FAIL   │
         └──────────────────┬─────────────────────────┘
                            │ parsed
                            ▼
                ┌─────────────────────┐
                │  run_dpe_smoke.py   │
                │  Layer 1 gate       │
                └──────────┬──────────┘
                           │ on PASS
                           ▼
         ┌────────────────────────────────────────────┐
         │ fc_top.v  instantiates dpe V × H grid       │
         └──────────────────┬─────────────────────────┘
                            │
                            ▼
         ┌────────────────────────────────────────────┐
         │              WORKLOAD TB                   │
         │   embedded numpy reference;                │
         │   13 cases (Stages 1A + 1B + 1C);          │
         │   compare DUT vs reference                 │
         └──────────────────┬─────────────────────────┘
                            │ parsed
                            ▼
                ┌─────────────────────┐
                │  run_fc_smoke.py    │
                │  Layer 2 gate       │
                └─────────────────────┘
```

---

## Appendix B — Glossary

| Term | Definition |
|---|---|
| ACAM | Analog Content-Addressable Memory. Optional tail stage in NL-DPE that applies one of three nonlinearities (identity, exp, log) per output. Absent in Azure-Lily. |
| CCYC | Compute Cycle count, the bit-serial pipeline portion of total latency. CCYC = PRECISION + (PIPELINE_DEPTH−1) + ACAM_CYCLES. |
| DPE | Dot-Product Engine. The primitive hard block under verification. |
| DUT | Device Under Test. The RTL module compiled and simulated. |
| Corner-turn | Fixed bit permutation at the LOAD path that distributes byte-major BRAM data into bit-slice-major slice banks. Pure wiring + shared address counter; no decode logic. |
| Faithful primitive | Behavior model whose cycle count emerges from physical state-machine advances, not from a declared parameter. Post Task #99: uses double-buffered slice-major storage (substrates A/B + `load_phase` selector). Contrasts with "lazy primitive". |
| Fidelity | Relative error between RTL-measured cycles and simulator-predicted cycles. Faithful primitive: 0%. Workload: per-stage decomposed (`nba + tree + clb + wrap`) paid in T_fill; T_steady bit-exact under Task #99 double-buffered LOAD. |
| H | N-axis tiling factor in `fc_top.v`. H > 1 means the output dimension is split across multiple DPEs. |
| LCYC | Load Cycles. Time to stream R · PRECISION bits into the input buffer at DPE_BUF_WIDTH bits per cycle. |
| Lazy primitive | Pre-faithful primitive (`dpe_nldpe.v`, `dpe_azurelily.v`) that uses a 4-slot byte-major ring buffer. Retained for the workload layer pending migration of `fc_top.v`. |
| `load_phase` | 1-bit selector in the Task #99 faithful primitive that ping-pongs LOAD writes between the two input substrates (`input_buf_slice_a`, `input_buf_slice_b`). Toggles each pass. Replaces the Task #93 `load_safe` register. |
| OCYC | Output Cycles. Time to drain C output values to the result port. |
| Double-buffered LOAD | Task #99 design: two physically separate input substrates (A, B) ping-pong'd by `load_phase`. pass-(k+1) LOAD writes substrate B while pass-k COMPUTE reads substrate A; no WAR hazard, no LOAD-gate. `T_steady = max(LCYC, CCYC, OCYC)`. Supersedes Option A1 / Task #93. |
| Oracle | Independent numpy reference that emits expected outputs and cycles. |
| Path A | V·H weight-stationary GEMM mapping: every DPE fires M times, one per output row. Adopted after rejection of Path B (K-time-multiplexed). |
| Double-buffer substrate | Storage model where each pass writes one of two physically separate input substrates while the other is being read. The mac and output banks remain single-buffered. Pre-Task-#99 design used a single input substrate with a `load_safe` LOAD-gate; pre-Task-#93 used a 4-slot ring buffer. |
| Slice-major | Storage organization where input bits are grouped by bit-position (slice 0 holds bit-0 of all rows, slice 7 holds bit-7 of all rows). Each slice is read by COMPUTE in one cycle. |
| Sub-FSM | One of LOAD, COMPUTE, OUTPUT — three concurrent finite state machines inside a faithful primitive. |
| T_fill | Latency of the first pass through the pipeline. = LCYC + CCYC + OCYC + 2 handoff cycles. NL: 116. AL: 332. |
| T_steady | Per-pass interval in pipelined-fill steady state. Post Task #99: max(LCYC, CCYC, OCYC). NL: 52. AL: 256. |
| V | K-axis tiling factor in `fc_top.v`. V > 1 means the inner dimension is split across multiple DPEs and reduced via a CLB adder tree. |

---

## Appendix C — Quick-Reference Commands

```bash
# Regenerate primitives from JSON
python nl_dpe/gen_dpe_nldpe_faithful.py
python nl_dpe/gen_dpe_azurelily_faithful.py

# Regenerate test vectors
python fc_verification/oracles/nldpe_mac_oracle.py --gen-test-vectors
python fc_verification/oracles/azurelily_mac_oracle.py --gen-test-vectors

# Layer 1 — primitive smoke (both archs, all modes)
python fc_verification/run_dpe_smoke.py

# Layer 2 — workload smoke (13 cases)
python fc_verification/run_fc_smoke.py

# Faithful primitive smoke (12 cases at 0% fidelity)
cd fc_verification
make faithful-smoke        # NL T1..T7
make faithful-smoke-al     # AL T1..T5

# Targeted single-case build via Makefile knobs
make tb-faithful-nldpe TEST_MODE=4    # NL T4 M-sweep
make tb-faithful-al    TEST_MODE=4    # AL T4 M-sweep
```

---

**End of overview.** For implementation detail on either faithful primitive
including cycle-by-cycle worked numerical examples, refer to
`DPE_NLDPE_FAITHFUL_WALKTHROUGH.md` and `DPE_AZURELILY_FAITHFUL_WALKTHROUGH.md`.
For per-testcase cycle log traces and the full breakdown of where each
cycle comes from, refer to `CYCLE_ACCOUNTING.md`.
For the underlying methodology principles, refer to `FIDELITY_METHODOLOGY.md`.
