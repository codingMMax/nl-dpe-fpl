# Spec — v2 backbone simulator: behavior, cycles, and energy (Stage 5)

**Status**: v0.2 DRAFT for review, 2026-10-03. Nothing in this document has
been implemented yet. It is written to be reviewed section by section; the
implementation will be planned and driven separately. Revision v0.2 (after
review of v0.1): §3 revamped into the three-part platform abstraction
(fabric / compute tile / operator realizations), §5.3 adds the one-cycle-law
decision, §13 expanded into the implementation guide, SIM15–SIM16 added.

**As-built note (2026-10-03).** Phases 0–1 are implemented under
`v2/sim/simulator/` (file layout pinned in §13.0). The config format and the
energy model were redesigned during implementation (one shared crossbar +
input/output buffers on every platform, one output stage — ACAM or ADC —
per-unit costs scaled by geometry, `nn/` energy granularity plus an
`area_power.py` crossbar term, no DAC); the as-built truth is
`v2/sim/simulator/{configs/,platforms.py,cost.py}` and their docstrings.
§3.2–§3.5, §4.2 and §10 still describe the v0.2 format and numbers — rewrite
pending.

**What this document is.** The charter for one uniform simulator that runs
any v2 workload and reports three things together — the computed values, the
cycle count, and the energy — and that builds larger workloads (up to
BERT-Tiny end to end) out of the already-certified kernel simulators. Every
rule below carries its reason next to it. No new jargon is introduced: every
term is either an established project term (crossbar, pass, window, ACAM,
GEMM, DIMM, block RAM, CLB, DSP) or is spelled out in full at first use.

**Contracts above this doc** (this simulator consumes them and adds nothing
to their value or timing truth):

- `v2/spec/dpe_nldpe.md` — the crossbar primitive: values (§6), cycle
  formulas (§5.3), the operator pass layer (F5–F8, P28).
- `v2/spec/gemm.md` — the GEMM array.
- `v2/spec/dimm.md` — the DIMM operator (pool/farm, schedule-injected pass
  counts, the balance law, the producer-to-farm fill).
- `v2/spec/softmax.md` — the row softmax machine.
- `v2/spec/softmax_online.md` — the blocked online softmax machine.
- `v2/spec/flash_attention.md` — the flash-attention composition and its
  Δ-cycles / Δ-energy comparison contract (§5).
- `paper/methodology/attention_dimm_mapping.md` — resource allocation for
  attention (crossbar counts, identity packing, lane parallelism). This
  document predates the v2 charters: where it disagrees with them (its
  softmax normalizes with a reciprocal and multiplies; its projections emit
  log-domain outputs), the v2 charters above win. It stays the authority for
  resource counts and packing.

---

## §1 What this simulator is for

Every run of the backbone simulator must answer three questions at once:

1. **What values does the workload compute?** (behavior)
2. **How many cycles does it take?** (latency)
3. **How much energy does it consume?** (energy)

and, for composed workloads, answer them **per operator / per layer** as well
as end to end. This is what the later main experiments need: end-to-end
throughput, end-to-end energy, and a per-layer / per-operator breakdown, for
both platforms (NL-DPE and Azure-Lily), from one uniform model.

The kernel simulators that already exist answer questions 1 and 2 per kernel
(each certified against its oracle and its RTL). Nothing in the repo answers
question 3. The backbone simulator is the layer that:

- wraps the five certified kernel simulators (GEMM array, DIMM, row softmax,
  online softmax, and the crossbar primitive) without changing them,
- adds energy accounting on top of the schedules those simulators already
  produce,
- composes workloads into larger workloads (linear layer → attention head →
  encoder block → BERT-Tiny) with explicit dependency rules,
- reports values (optional), cycles, energy, and throughput in one record.

**Position in the ladder.** This is Stage 5 of the forward plan ("a new
minimal simulator consuming the Stage 1–4 charters; BERT-Tiny end to end"),
extended with the energy axis, which the ladder never had. It is also the
machinery that the flash-attention charter needs for its promised
Δ-cycles / Δ-energy comparison (`flash_attention.md` §5).

---

## §2 Position: what the backbone may and may not change

The verification chain is unchanged:

```
spec → oracle → GATE 1 (per case, inside dump_case) → sim → GATE 2 (RTL) → RTL
```

The backbone simulator:

- **consumes** the certified kernel simulators through thin adapters. It
  never edits them, never re-implements their value paths, and never
  re-derives their cycle math.
- **introduces no new value truth.** Values come from the kernel simulators
  (which are gated against their oracles). Composed value runs just route
  real arrays from one kernel to the next.
- **introduces no new cycle truth.** Cycle counts come from each kernel's
  own contract (§5 lists the three contract styles). Composition only
  places kernels in time relative to each other.
- **introduces exactly one new axis: energy.** Energy is analytical. It is
  never gated against the RTL (the RTL is integer-only; there is no energy
  truth in it to gate against). Its verification is pins and witnesses (§9).

**One principle that holds for both cycles and energy:** costs are
**value-independent**. Every cycle count and every energy count is derived
from the workload's shapes and from the pass counts the schedule actually
issues — never from the data values. (This is already the discipline of the
cycle models: the softmax frozen pins are "schedule-determined, independent
of the score values". Energy inherits it.)

---

## §3 The platform and its configuration files

A **platform** is the hardware being modeled. Two platforms are in scope
from day one — **NL-DPE** and **Azure-Lily** — because the paper's
comparison needs both from the same backbone. A platform file states three
things: the shared fabric (§3.3), the compute tile as an explicit pass
pipeline with per-stage costs and charge scopes (§3.4), and the operator
realizations — which engine runs which operator (§3.5). The files live in
v2, in v2's own format:

```
v2/sim/simulator/configs/nl_dpe.json
v2/sim/simulator/configs/azure_lily.json
```

Frozen copies rather than reading the archive at run time, because: the
archive is reference-only by repo policy and v2 must stay self-contained;
the archived file **format** may change later and v2 must not follow it
(the physical facts — the constant values — stay the same, and those are
what v2 freezes); and every constant carries provenance in this charter so
every number is traceable to its archived source and its derivation.

### §3.1 What the two platforms actually are (the architecture insight)

The two architectures differ in exactly one hardware aspect — the output
stage of the compute tile:

- the **crossbar behavior is exactly the same** in both: bit-serial
  multiply-accumulate, the same pass structure, the same cycle law;
- **NL-DPE's tile ends in an ACAM** (the analog nonlinear output stage —
  identity, rectified-linear, exp, log — applied to the accumulated
  column value, once per pass over all columns);
- **Azure-Lily's tile ends in an ADC** (conversion per input bit slice per
  active column; every nonlinear function must instead run on the FPGA
  fabric as table lookups and multiplies);
- **both output stages cost the same cycles** (both legacy RTLs measured
  `COMPUTE_CYC = P + 2 = 10`): the difference is energy and capability,
  not time.

The v2 frozen cycle-count law (`dpe_nldpe.md` §5.3) is the law **all
tiles follow**, parameterized by each tile's geometry and port width
(§5.3 of this charter records the Azure-Lily numbers under that law).

### §3.2 Why a new file format (what was wrong with the archived one)

The archived configuration files stay untouched as provenance; the
physical facts are frozen into the new format. Six specific problems with
the archived format, each fixed here:

| archived weirdness | fix in the v2 format |
|---|---|
| a `scale_with_geometry` flag silently multiplies stored numbers by total columns at load, so the same field means a different thing per file | every energy field states its own unit and charge scope (`per_fire`, `per_pass`, `per_column`, `whole_array`, `active_columns`) |
| behavior as capability flags (`analog_nonlinear`, `digital_accum`, `acam_cycles`, `log_softmax_fusion`) that a hidden factor table translates into operation counts | an explicit pass pipeline: each stage object carries its own counts and costs; the model walks the list, nothing is reconstructed |
| the pass pipeline (load → 8 bit-slice fires → accumulate → output stage → drain) never appears in the file | `pass_pipeline` is the spine of the tile section |
| two clocks mixed silently (`freq_MHz` core vs `freq` fabric) plus nanosecond constants the v2 cycle laws supersede | one `clock_mhz` per platform for time conversion; energy constants are per operation (§3.6) |
| device budgets (`total_*`) mixed in with per-operation constants | a separate optional `resource_budgets` section, informational only (§6.5) |
| a workload-mapping fact (`log_softmax_fusion`) sits in the platform file | mapping facts live in `operator_realizations` (§3.5), where they belong |

### §3.3 The shared fabric (identical in both platform files)

```jsonc
"fabric": {
  "clock_mhz": 300,
  "block_ram": {"port_width_bits": 40, "energy_pj_per_access": 0.0495},
  "clb": {"energy_pj_per_op_unit": 0.660,
          "add_pj": 0.08498, "compare_pj": 0.26439,
          "activation_unit_pj": 0.45,
          "lookup_table_read_pj": 2.64, "lookup_activity_factor": 1.0},
  "dsp": {"multiply_accumulate_pj": 1.2}
}
```

The loader asserts the two files' fabric sections are equal; a fabric
constant may never differ between platforms by file edit — only by a
recorded decision in this charter.

| constant | value | provenance |
|---|---:|---|
| block RAM, per access of 5 bytes (width 40) | 0.0495 pJ | archived files → `fpga_specs.bram_pj_per_access` (same value in both) |
| CLB operation unit | 0.660 pJ | `fpga_specs.clb_pj_per_mac` |
| add | 0.08498 pJ | measured reference `84.98358e-6` nJ (archived `nn/constant.py`); also the basis of the archived core's add coefficient |
| compare | 0.26439 pJ | measured reference `793.1801e-6` nJ / 3 (same source); basis of the archived core's compare coefficient |
| activation unit | 0.45 pJ | `fpga_specs.act_energy_pj_per_op` (measured reference 0.4532 pJ; 0.45 is the frozen value) |
| lookup-table read | 2.64 pJ × activity 1.0 | a 256×8-bit table = 2048 bits / 64 bits per LUT6 = 32 LUTs = 4 CLB units × 0.660 pJ; the softmax study's adopted form |
| DSP multiply-accumulate | 1.2 pJ | `fpga_specs.dsp_pj_per_mac` |

The archived device budgets (`total_io`, `total_clb`, `total_dsp`,
`total_mem`, `total_imc`, `total_softmax_lanes`) move to an optional
`resource_budgets` section — informational only; the loader reads them,
the cost model never does (SIM8, §6.5).

### §3.4 The compute tile (the part that differs, as an explicit pipeline)

NL-DPE:

```jsonc
"compute_tile": {
  "crossbar": {"rows": 256, "columns": 256},
  "input_bit_slices": 8,
  "port_width_bits": 40,
  "pass_pipeline": [
    {"stage": "analog_activation", "fires": "per_bit_slice",
     "energy_pj_per_fire": 3.89, "charge_scope": "whole_array"},
    {"stage": "conversion", "energy_pj": 0},
    {"stage": "output_stage", "kind": "acam", "fires": "per_pass",
     "energy_pj_per_column": 0.171445313, "charge_scope": "all_columns",
     "modes": ["identity", "relu", "exp", "log"]}
  ],
  "accumulator": {"kind": "in_tile", "fabric_charge_per_pass": 0}
}
```

Azure-Lily:

```jsonc
"compute_tile": {
  "crossbar": {"rows": 512, "columns": 128},
  "input_bit_slices": 8,
  "port_width_bits": 16,
  "pass_pipeline": [
    {"stage": "analog_activation", "energy_pj": 0},
    {"stage": "conversion", "kind": "adc",
     "conversions": "per_bit_slice_per_column",
     "energy_pj_per_column": 2.33, "charge_scope": "active_columns"},
    {"stage": "output_stage", "kind": "none"}
  ],
  "accumulator": {"kind": "fabric_shift_add", "fabric_charge_per_pass": 0}
}
```

The energy model **walks the pipeline and multiplies** — no flags, no
load-time scaling, no hidden factor tables. Reading either tile section
says what a pass does and what each stage costs, in the file itself.

Per-constant provenance for the tile section:

- **NL-DPE, 3.89 pJ per analog activation fire** (per input bit slice,
  whole array): derived in `nl_dpe/area_power.py` from analog power
  3.8906 mW (crossbar 1.31 + driving circuits 2.44 + input buffer 0.1406)
  at the 1 GHz derivation clock.
- **NL-DPE, 0.171445313 pJ per column per ACAM pass**: derived from
  digital power (ACAM 43.52 + output buffer 0.1406 + exclusive-or tree
  0.235 = 43.8956 mW) / 256 columns / 1 GHz. The derivation reproduces
  the frozen value to within 0.013% (the frozen value implies
  43.8900 mW); **the frozen file value is the truth**, the derivation is
  provenance, and the pin test checks the derivation only within 0.02%.
- **NL-DPE, conversion 0**: the ACAM absorbs the conversion.
- **Azure-Lily, 2.33 pJ per input bit per active column**: the archived
  file stores 2.33 with the `scale_with_geometry` flag; the v2 format
  states the per-column meaning directly.
- **Azure-Lily, analog activation 0 and output stage none**: archived
  values; the tile has neither an analog-activation charge nor an
  in-tile output stage.
- **Azure-Lily's `fabric_charge_per_pass: 0`** for the shift-add
  accumulator: the archived files charge it zero — OPEN, SIM12.

### §3.5 Operator realizations (the comparison as data)

NL-DPE:

```jsonc
"operator_realizations": {
  "linear_matmul":     {"engine": "gemm_array"},
  "log_domain_matmul": {"engine": "dimm"},
  "exp":  {"engine": "tile_pass", "mode": "exp"},
  "log":  {"engine": "tile_pass", "mode": "log"},
  "softmax_normalize": {"engine": "log_domain"}
}
```

Azure-Lily:

```jsonc
"operator_realizations": {
  "linear_matmul":     {"engine": "gemm_array"},
  "log_domain_matmul": {"engine": "unavailable"},
  "exp":  {"engine": "fabric", "lookup_table_reads": 1},
  "log":  {"engine": "fabric", "lookup_table_reads": 1},
  "softmax_normalize": {"engine": "fabric", "lookup_table_reads": 1,
                        "dsp_multiplies": 1},
  "attention_score_matmul": {"engine": "dsp_mac_lanes"}
}
```

This table is where the architectural difference lives as data:

- NL-DPE realizes `exp` and `log` as tile passes (the ACAM modes);
  Azure-Lily must realize them on the fabric (table lookups, multiplies)
  — the softmax energy story in one row pair;
- Azure-Lily **cannot** run the log-domain DIMM — `unavailable` is
  stated in the file, not implied by an absent flag; its attention score
  path is the DSP-lane variant of the attention mapping document §8;
- the workload library maps each operator through this table, so the same
  workload graph runs on both platforms with different engine assignments,
  and the report's per-node rows are directly comparable.

### §3.6 Clocks (one decision, stated plainly)

The archived files carry two frequencies: a 1000 MHz "core" clock (the one
the energy constants were derived at) and a 300 MHz fabric clock. The v2
cycle counts are counts of system clock cycles of the RTL, which in the VTR
flow runs at the fabric clock. The backbone therefore:

- treats **cycles as the primary unit** everywhere;
- declares **one `system_clock_mhz` per platform** (default 300) used only
  to convert cycle counts to nanoseconds;
- keeps the energy constants **frozen per operation** — they are not
  re-derived when the system clock changes. The derivation through
  power ÷ clock is provenance, not a live dependency.

The alternative (re-derive energy at the operating clock) would scale all
analog energies by the same factor on both platforms — changing absolute
numbers but no comparison — and would raise a physical question (the analog
search pulse has its own duration, 2 ns in the archived physics sheet,
independent of the digital clock). That question is recorded as OPEN in §12.

---

## §4 Energy accounting

### §4.1 The counting principle

Energy is counted **per operation actually performed** — per crossbar pass,
per fabric add, per block-RAM access — never per unit time, and never from
idealized operation counts. The pass counts are the ones the schedule
**actually issues** (the same schedule-injected rule as the cycle models,
P28): if a schedule issues unpacked passes, the unpacked count is what is
charged. Idealized packed counts exist only as a sizing convenience.

### §4.2 One crossbar pass, both platforms

A **pass** is one full input burst through one crossbar: the schedule packs
up to `min(R, C)` elements into it, zero-padded to a full row of `R` input
bytes; the pass fires the analog array once per input bit slice (8 slices
per byte-wide element stream).

**NL-DPE, per pass (256 rows × 256 columns):**

```
analog activation      8 bit slices × e_analoge_pj        = 8 × 3.89         = 31.12 pJ
conversion             e_conv_pj per input bit per
                       active column                      = 0                = 0
output stage (ACAM)    1 fire × e_digital_pj × C          = 0.171445313×256  = 43.89 pJ
crossbar part total                                                           75.01 pJ
port traffic           ceil(R·8/40) input accesses
                       + ceil(C·8/40) output accesses
                       = 52 + 52 accesses × 0.0495 pJ                         =  5.15 pJ
per-pass total (with traffic)                                                  80.16 pJ
```

At 128 columns the crossbar part is `8×3.89 + 0.171445313×128 = 53.07 pJ`.
Both values (53.07 and 75.01 pJ) are the ones already printed by the softmax
study — the cross-check that the frozen constants and the per-pass formula
reproduce published numbers.

**Whole-array policy (frozen convention, stated with its reason):** the
output stage fires **all C columns** and the analog activation fires the
**whole array**, regardless of how many elements the pass carries. This is
how the archived core charges the output stage ("all columns fire regardless
of utilization"), and it is the physical reason identity packing helps (the
attention mapping document §1–2: a pass costs the same whether it carries
64 elements or 256, so packing more elements per pass divides the per-element
cost).

**Azure-Lily, per pass:**

```
analog activation      0
conversion             8 bit slices × 2.33 pJ × active columns
                       (only the columns that produce outputs are charged)
digital post          0 (archived value; the per-bit shift-add runs on the
                       fabric — OPEN whether to charge it, SIM12)
port traffic           ceil(R·8/40) + ceil(C·8/40) accesses × 0.0495 pJ
```

At full 128 active columns the conversion part is `8 × 2.33 × 128 = 2385.28 pJ`
per pass.

### §4.3 Fabric operations

| operation | energy | used by |
|---|---:|---|
| add | 0.08498 pJ | reductions, residual adds, the DIMM farm feed, layer normalization |
| compare | 0.26439 pJ | row-max folds, max pooling |
| activation-unit operation | 0.45 pJ | activation units |
| DSP multiply-accumulate | 1.2 pJ | multiply paths (none in the v2 log-domain softmax; used where a composite genuinely multiplies) |
| block-RAM access | 0.0495 pJ per 5 bytes | all input/output/parked-buffer traffic |
| lookup-table ROM read | 2.64 pJ × activity factor (1.0) | table lookups (exponent ROM on Azure-Lily, reciprocal and square-root tables) |

### §4.4 Block-RAM traffic, and why per-access

One access moves `bram_width/8 = 5` bytes. The backbone charges
`ceil(bytes / 5) × 0.0495 pJ`. This is the archived core's convention
(`IMC/peripherals/memory.py`). Two other granularities exist in the repo and
are recorded here so the choice is trackable:

- the archived event-model stack charges 0.00495 pJ per byte (half the
  per-access rate);
- the softmax study charges 0.0495 pJ per element (5× the per-access rate).

**Decision (SIM4): the per-access form is normative.** Consequence: the
backbone's softmax energy will differ from the softmax study's published
table; the study is kept as a **differing witness**, not overthrown — its
relative conclusions (per-element comparisons at one convention) stand, and
the charter records the delta.

Per-pass port traffic is already part of §4.2 (52 + 52 accesses at 256×256 —
the same counts as `LOAD_CYC` and `OUTPUT_CYC`, because one access happens
per port cycle). Beyond pass traffic, the schedules that re-read parked
buffers (the DIMM farm re-reads of the converted operands) and the final
output serialization are charged as their own accesses (§10 shows the full
trail).

### §4.5 Weight programming (reported separately)

Programming one crossbar writes `R×C` weight bytes: `ceil(R·C/5)` block-RAM
accesses, reported as **setup energy** per crossbar and **excluded** from
per-inference energy — mirroring `weight_cycles`, which every kernel already
reports separately. The analog write pulses themselves are charged zero
(the archived model does not charge them; recorded as an assumption, §11).

### §4.6 What is charged zero (explicit list, so nothing is silently free)

- NL-DPE conversion (the ACAM absorbs it).
- Azure-Lily analog activation (archived value zero).
- Azure-Lily digital post (archived value zero — OPEN, SIM12).
- Crossbar weight write pulses (§4.5).
- Static / leakage energy (dynamic only, as in the archive and the study).
- Data-dependent effects of any kind (costs are value-independent, §2).
- Clock gating is never modeled: a partially filled pass costs the same as
  a full one (the whole-array policy, §4.2).

---

## §5 Cycles and time

### §5.1 Cycles are the primary unit

Every node reports cycles. Nanoseconds are derived:
`ns = cycles × 1000 / system_clock_mhz` (one cycle at 300 MHz = 3.33 ns).
Throughput is derived the same way.

### §5.2 The three cycle-contract styles are preserved, not flattened

The kernel simulators have three different relationships between the
measured schedule and the closed-form formula. The backbone preserves each
and **tags every node with its style** in the report:

| style | kernels | meaning |
|---|---|---|
| measured = formula | crossbar primitive, GEMM array | the measured event schedule must equal the closed form exactly (Δ_impl = 0 gate) |
| model total is the truth | DIMM | the shadow model's stage-ledger total is the declared cycle truth; the serializer (M·N words) is reported separately and added by the harness gate |
| measured schedule is the truth | row softmax, online softmax | the measured event schedule is the contract; the analytic envelope is only a lower-bound corridor check |

Reason: these postures were earned per kernel through GATE 2 against the
RTL; a uniform backbone must not quietly change what any kernel's number
means. The report says, per node, where the number came from.

### §5.3 One cycle law, every tile (SIM16)

The v2 frozen cycle law (`dpe_nldpe.md` §5.3) is the law **all tiles**
follow — parameterized by each tile's geometry and port width:

| tile | LOAD = ceil(R·8/port) | COMPUTE = P+2 | OUTPUT = ceil(C·8/port) | T_fill | T_steady |
|---|---:|---:|---:|---:|---:|
| NL-DPE 256×256, port 40 | 52 | 10 | 52 | 114 | 60 |
| Azure-Lily 512×128, port 16 | 256 | 10 | 64 | 330 | 264 |

Every row uses the same formulas, with
`T_steady = max(LOAD+P, COMPUTE, OUTPUT+1)` — the single-buffer law
(P1/A9). The legacy Azure-Lily axiom (`T_steady = 256`) used the
double-buffered law `max(LOAD, COMPUTE, OUTPUT)` — the same
v2-vs-legacy buffering difference already documented for NL-DPE at
Stage 1 (60 vs 52, the legacy witness's "double-buffer split"). The
legacy axioms are recorded as a **differing witness**; giving Azure-Lily
a buffering discipline NL-DPE does not have would make the comparison
unfair. The fill number (330) is identical under both laws.

### §5.4 Composition of time, and why there is no central event loop

The backbone composes time by **placing each kernel's own schedule in time
relative to its producers** — offset arithmetic on already-certified event
lists — not by a central event loop. The reasons, recorded as SIM7:

- the five kernel timings are already event schedules, produced by
  certified grammars; a central loop would have to re-express them as actors
  and re-verify them;
- the only event loop in the repo (the archived `main.py`) is built on the
  v1 conventions v2 dropped (nanosecond grid, hardcoded Azure-Lily geometry);
- fine-grained overlap is still available through readiness schedules (§6.3),
  because the kernel schedulers already accept per-pass readiness gates;
- repeated inputs and throughput come from each kernel's own
  `T = T_fill + (M−1)·T_steady` law lifted one level (§6.4), not from
  unrolling a loop.

If a genuinely dynamic arbitration study ever appears, an event loop can be
built **on top of** the same records, with the backbone report as its
cross-check. Nothing here forecloses that.

---

## §6 How workloads compose

### §6.1 Nodes and adapters

A **node** is one workload instance. It declares: its input shapes, its
resource counts (how many crossbars of each kind, fabric unit counts), and
its kernel adapter. The **adapter** wraps one certified kernel simulator and
translates its outputs (values, cycle contract, event schedule) into the
uniform record (§7). Adapters add the counted work (block-RAM traffic,
fabric adds and compares, setup energy) per this charter's tables. The
kernel simulators themselves are not edited.

### §6.2 Sequential composition (data dependency)

A consumer starts when its input is ready. Default readiness is
**whole-output**: the consumer starts at the producer's last output cycle
(plus any declared handoff cost). The composed latency is the finish of the
last node; the composed energy is the sum. Inside a node, phases may overlap
exactly as the kernel's own contract says (the DIMM's
`max(T_A, T_B, T_start + T_E)` stays what it is; composition never narrows
a kernel's internal overlap).

### §6.3 Readiness schedules (fine-grained overlap, when it matters)

A producer may expose **part-level readiness**: "row `r` of my output is
ready at cycle `t(r)`". Consumers that accept part-level readiness — the
kernel schedulers already take per-pass readiness gates — start before the
producer finishes. Both modes are legal; the report records which was used.
Example: row softmax over an S×S score matrix can begin on row 0 while the
score producer is still emitting later rows. Reason this is safe: the
readiness gate is the same mechanism the certified grammars already use
(`schedule_pass_sequence` takes a readiness cycle per pass), so no new
timing machinery is introduced.

### §6.4 Repeated inputs and end-to-end throughput

For `n` back-to-back inputs (tokens, matrices) through a composed chain:

- **first-input latency** = the chain's fill time (each node's first fill,
  along the dependency path);
- **steady interval** = the maximum per-input steady cost over the nodes on
  the critical path (each node already exposes `T_steady` per input by its
  own `T = T_fill + (M−1)·T_steady` law);
- **throughput** = `system_clock / steady interval`.

This is the standard pipeline result and it follows from each kernel's own
law lifted one level — no loop, no new formula. All three numbers are
reported.

### §6.5 Parallel composition and resources

Nodes declared **parallel** run concurrently on provisioned resources; the
combined time is the max of the individual finish times. Resources are
**provisioned, not contested**: each node declares how many crossbars of
each kind it uses, and the DIMM balance law (`dimm.md` §4) is the sizing
tool. The platform files carry resource budgets (total CLB, DSP, memory,
IMC tiles) as **informational** fields; checking that a composed workload
fits the budget is a later mapping phase (§14), not this charter's promise.

---

## §7 The record and the report

### §7.1 The record

One run produces:

- **scheduled events**: kind, node, start cycle, end cycle (from the kernel
  adapters' translated schedules);
- **counted operations**: kind, node, count (block-RAM accesses, fabric
  adds / compares / activation / DSP / lookups, crossbar passes, setup).

Energy = sum over counted operations × the §4 costs. Latency = the finish of
the last scheduled event (per §5.2's posture rules). Every entry is tagged
with its node, so per-layer / per-operator breakdown is a group-by, not a
separate model.

### §7.2 The report

Every run reports:

- **per node**: name, kind, shapes, pass counts issued, crossbar counts,
  cycles (with posture tag; serialization reported separately where the
  kernel's contract says so), energy by component;
- **per component, summed**: analog activation, conversion, output stage,
  fabric adds, fabric compares, activation units, DSP, block RAM, lookups,
  setup (weight programming);
- **end to end**: total cycles, total energy, first-input latency, steady
  interval, throughput;
- **units**: cycles and pJ primary; nanoseconds and nJ derived at the
  platform clock.

JSON shape (illustrative, fields stable at implementation):

```json
{
  "platform": "nl_dpe",
  "workload": "bert_tiny",
  "sequence_length": 128,
  "clock_mhz": 300,
  "cycles":  {"total": 0, "by_node": {}, "serialize_by_node": {}},
  "energy_pj": {"total": 0, "by_component": {}, "by_node": {},
                "setup_weight_programming": 0},
  "throughput": {"first_input_cycles": 0, "steady_interval_cycles": 0,
                 "inputs_per_second": 0}
}
```

A printed table (one row per node, component columns) is the human-facing
form — the same shape as the DSE and softmax-study tables.

---

## §8 The workload library

### §8.1 Atomic workloads (adapters over the certified kernel simulators)

| workload | certified simulator | notes |
|---|---|---|
| GEMM array (projections, feed-forward) | `v2/sim/simulator/kernels/gemm_sim.py` | Path A weight-stationary; per-pass energies + `M·N·(V−1)` reduction-tree adds + port traffic |
| DIMM (log-domain matmul: attention scores, attention × V) | `v2/sim/simulator/kernels/dimm_sim.py` | pool/farm passes + `M·N·K` fabric adds in the feed + parked-buffer re-reads + serialization |
| row softmax | `v2/sim/simulator/kernels/softmax_sim.py` | EXP passes + LOG passes + row-max compares + row sums + clamps |
| online softmax (blocked) | `v2/sim/simulator/kernels/softmax_online_sim.py` | block stream + EXP passes + factor passes + LOG passes + combine adds |
| crossbar primitive (standalone) | `v2/sim/simulator/kernels/nldpe_sim.py` | mainly for tests; inside the kernels its passes are counted by the kernels' adapters |

### §8.2 Fabric workloads (no crossbar)

| workload | operations charged |
|---|---|
| elementwise add (residual) | one add per element + block-RAM read/write traffic |
| activation, fused in the crossbar output stage (the primitive's rectified-linear mode) | no charge beyond the pass: the fold is part of the ACAM output stage, already charged per pass (§4.2) |
| activation, fabric activation unit | 0.45 pJ per element, for activations outside the crossbar path |
| activation (GELU) | **OPEN (SIM10)**: fabric lookup-table reads, or the rectified-linear mode as an approximation, or both reported — the certified ACAM modes do not include GELU |
| layer normalization (per token) | per the archived fabric model: mean reduction + division, `n` subtracts + `n` squares + reduction + division, reciprocal-square-root lookup, `4n` normalize operations, plus per-token `n`-byte block-RAM reads and writes |
| embedding lookup (per token) | 3 reads + 1 write of model-dimension bytes + fabric adds (archived fabric model) |
| lookup-table reads | 2.64 pJ × activity factor per read |

### §8.3 Composite workloads

Built from §8.1 + §8.2, per the v2 charters (the attention mapping document
supplies resource counts and packing, not the operator choices). Engines
are assigned through the platform's `operator_realizations` (§3.5): the
lists below are the NL-DPE realizations; the Azure-Lily run of the same
composites routes through its own table (softmax on fabric lookups and
DSP multiplies; attention scores via the DSP-lane variant).

- **linear layer** = GEMM + optional activation + optional bias adds.
- **attention head** (head dimension 64) = score matmul (`query · key
  transposed`, S×64 @ 64×S, on the DIMM) → row softmax or online softmax →
  weighted-sum matmul (`scores · value`, S×S @ S×64, on the DIMM).
- **multi-head attention** (2 heads) = two attention heads + concat (no
  compute charged) + output projection (GEMM, S×128 @ 128×128).
- **encoder block** = multi-head attention + residual add + layer
  normalization + feed-forward (inner GEMM S×128 @ 128×512 + activation +
  outer GEMM S×512 @ 512×128) + residual add + layer normalization.
- **BERT-Tiny** = embedding lookup + 2 encoder blocks + final layer
  normalization. Constants (repo-pinned): 2 layers, 2 heads, model dimension
  128, head dimension 64, feed-forward dimension 512, vocabulary 30522;
  sequence length is a workload parameter (reference values 128 and 1024).
  **OPEN (SIM11)**: whether the headline number includes the embedding
  lookup (recommendation: encoder blocks as the headline, embedding
  reported alongside).
- **flash-attention comparison** = the composition of
  `flash_attention.md` §3 (blocked score softmax + blocked weighted sum +
  combine + epilogue) vs the full-row attention composite, same shapes and
  corpus, reporting **Δ cycles and Δ energy** — the comparison that charter
  §5 promises and currently has no machinery for.

**Blocked-on notes (costs are not blocked).** The score and weighted-sum
matmuls need the **signed DIMM** extension (`flash_attention.md` §4 — the
certified DIMM value contract is unsigned magnitudes; attention products
are signed). That extension is pending on the value path. The **cost
adapters are not blocked**: pass counts and operation counts are identical
with or without signs (costs are value-independent, §2), so the attention
composites can land cost-first and gain the signed value path when it
certifies.

---

## §9 Verification policy

### §9.1 Pins (self-tests, frozen in this charter at implementation time)

- **derivation pins**: the power-model derivation reproduces
  `e_analoge_pj` = 3.89 pJ exactly (to rounding) and `e_digital_pj` =
  0.171445313 pJ within 0.02% at 256×256 and the 1 GHz derivation clock
  (§3.4 records the 0.013% drift);
- **per-pass pins**: 75.01 pJ (256 columns) and 53.07 pJ (128 columns) —
  the softmax study's printed values, reproduced from the frozen constants;
- **worked-example pins**: the §10 DIMM row, exactly as tabulated;
- **composition arithmetic self-tests**: sequential chaining equals
  hand-shifted offsets; parallel equals the max; repeated inputs give
  steady interval = max per-node steady; report totals equal the sum of
  components and of nodes;
- **BERT-Tiny reference pin**: one reference configuration's totals frozen
  at implementation, so later changes are detected, not silently absorbed.

### §9.2 Witnesses (independent, where they exist)

- the **archived in-memory computing core** (`archive/azurelily_simulator/
  IMC/imc_core.py`) on shared shapes (GEMM runs, identity-pass streams) —
  a **differing witness**: its cadence is the older double-buffered one and
  its block-RAM model is per-access like ours, but its GEMM cycle law and
  some op groupings differ; differences are documented, not reconciled
  away;
- the **softmax study** as a differing witness for the block-RAM
  granularity decision (§4.4);
- the **existing kernel gates are untouched**: GATE 1 (oracle per case) and
  GATE 2 (RTL) keep certifying what they already certify.

### §9.3 What energy verification can never be, and the mitigation

Energy is analytical; there is no RTL energy truth to gate against. The
mitigation is exactly the discipline above: every constant carries its
provenance (§3), every formula is pinned to a worked example with published
cross-checks (§4.2, §10), and independent witnesses run where they exist
(§9.2). Energy numbers are never presented as measured hardware truth.

---

## §10 Worked examples with reference numbers

### §10.1 One crossbar pass, both platforms

| columns | analog 8×3.89 | output stage e_digital×C | crossbar total | port accesses (in + out) | port traffic | per-pass total |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 31.12 pJ | 43.89 pJ | **75.01 pJ** | 52 + 52 = 104 | 5.15 pJ | **80.16 pJ** |
| 128 | 31.12 pJ | 21.95 pJ | **53.07 pJ** | 52 + 26 = 78 | 3.86 pJ | **56.93 pJ** |

(Input is always a full row burst of `R` bytes — 52 accesses at `R` = 256;
output is `C` bytes — 26 accesses at `C` = 128.)

Azure-Lily, per pass (512×128, port 16 bits = 2 bytes per access, all 128
columns active): analog 0; conversion `8 × 2.33 × 128 = 2385.28 pJ`;
output stage none; port traffic `(256 + 64) accesses × 0.0495 pJ =
15.84 pJ`; **per-pass total ≈ 2401.1 pJ**. One line, the whole story: the
ADC dominates Azure-Lily's pass (≈ 2.4 nJ) where the ACAM gives NL-DPE
75.01 pJ — and the cycles are the same law (§5.3).

### §10.2 The DIMM worked example, extended with energy

The `dimm.md` §6 example: `M = N = 128, K = 64`, crossbar 256×256, so
`I = 256`; balanced counts `n_A = n_B = 1, n_E = 128`; issued passes 32
(logA) + 32 (logB) + 4096 (exp) = **4160 passes**; cycles `total = 2088`,
serializer `16384` words → 18,472 cycles → **61.6 µs at 300 MHz**.

Energy, with the full trail so every term is checkable:

| term | count | unit cost | energy |
|---|---:|---:|---:|
| logA pool passes | 32 | 75.01 pJ | 2,400.3 pJ |
| logB pool passes | 32 | 75.01 pJ | 2,400.3 pJ |
| exp farm passes | 4,096 | 75.01 pJ | 307,241.0 pJ |
| farm feed adds (`M·N·K`) | 1,048,576 | 0.08498 pJ | 89,108.0 pJ |
| pass input traffic | 4,160 passes × 52 accesses | 0.0495 pJ | 10,707.8 pJ |
| pass output traffic | 4,160 passes × 52 accesses | 0.0495 pJ | 10,707.8 pJ |
| parked-buffer re-reads (LA+LB, `K·(M+N)` bytes) | 16,384 bytes → 3,277 accesses | 0.0495 pJ | 162.2 pJ |
| final serialization (`M·N` int32 words) | 65,536 bytes → 13,108 accesses | 0.0495 pJ | 648.8 pJ |
| **total** | | | **423,376.3 pJ ≈ 423.4 nJ** |

Per output element: 423,376.3 pJ / 16,384 = **25.84 pJ**. (Check lines:
4,160 × 75.01 = 312,041.6 pJ crossbar part; 1,048,576 × 0.08498 =
89,108.0 pJ adds; block-RAM accesses 3,328 + 3,328 + 212,992 + 212,992 +
3,277 + 13,108 = 449,025 × 0.0495 = 22,226.7 pJ; sum = 423,376.3 pJ. The
table rows are rounded to one decimal; the pin asserts the unrounded sum.)

---

## §11 Assumptions

1. Buffer and accumulator bandwidth is sufficient (inherited, PF8). The
   DIMM accumulator drain traffic **is** charged as block-RAM accesses (the
   per-pass output traffic rows in §10.2); if the accumulator is later
   modeled as registers, that row becomes zero — a recorded change, not a
   silent one.
2. The final output stream is int32 words (the DIMM contract); other
   kernels' output traffic is byte-wide as their ports define.
3. Dynamic energy only; no static or leakage energy (§4.6).
4. Lookup-table activity factor 1.0 (upper bound, reported with every
   lookup-inclusive number).
5. Weight programming is setup energy, excluded from per-inference totals
   (§4.5).
6. Costs are value-independent (§2); value runs are optional behavior
   checks, never cost inputs.
7. Resource budgets in the platform files are informational until the
   mapping phase (§6.5).

---

## §12 Decisions

| # | decision | reason |
|---|---|---|
| SIM1 | Scope: one simulator reports values + cycles + energy, per operator and end to end, through BERT-Tiny and the flash-attention comparison | the main experiments need all three, broken down, on both platforms |
| SIM2 | The backbone consumes the five certified kernel simulators via adapters and adds no value or cycle truth | the chain `spec → oracle → GATE 1 → sim → GATE 2 → RTL` is already earned; re-implementing any of it would dilute it |
| SIM3 | Energy constants are frozen copies in `v2/sim/simulator/configs/`, in v2's own three-part format (shared fabric / compute tile as an explicit pass pipeline / operator realizations), with per-constant provenance | archive is reference-only; the archived format may change without moving v2; the physical facts stay frozen |
| SIM4 | Energy conventions follow the archived in-memory computing core: per-operation counting, whole-array activation and output stage, per-access block RAM, reference-measurement fabric ops | the user's chosen normative source; the softmax study's per-element block-RAM form becomes a differing witness (§4.4) |
| SIM5 | Cycles primary; one `system_clock_mhz` per platform converts to time; energy constants frozen per operation, not re-derived at the system clock | two clocks exist in the archived files; conflating them would change absolute numbers without changing any comparison (§3.6) |
| SIM6 | The three cycle-contract styles are preserved and tagged per node | the postures were earned through GATE 2; a uniform report must not change what a number means (§5.2) |
| SIM7 | Time composes by placing certified schedules in time (dependency chaining with readiness schedules), not by a central event loop | the certified grammars stay the engines; nothing is re-verified; fine-grained overlap and throughput still come out (§5.4, §6) |
| SIM8 | Parallel composition assumes provisioned resources; node-declared crossbar counts; the DIMM balance law sizes them | matches the existing pool/farm design machinery; contention, where it exists, is an explicit serial rule in the schedule, never emergent behavior |
| SIM9 | Weight programming energy is setup energy, reported separately | mirrors `weight_cycles`; per-inference energy must not carry one-time costs |
| SIM10 | **OPEN**: FFN activation form — fabric lookup GELU vs the certified rectified-linear mode vs both reported | BERT-Tiny uses GELU; the certified ACAM modes do not include it (§8.2) |
| SIM11 | **OPEN**: whether the BERT-Tiny headline includes the embedding lookup (recommendation: encoder blocks as headline) | embedding is a lookup, not compute; the DSE-era attention workload was attention-only |
| SIM12 | **OPEN**: whether Azure-Lily's per-bit shift-add accumulation is charged as fabric adds (archived value: zero) | charging it raises Azure-Lily's pass energy; the archived files chose zero |
| SIM13 | K-identity packing (attention mapping document §2) is expressed as the schedule's packing property — pass counts are injected, packed and unpacked both reportable, values invariant | this is exactly the operator pass layer F8/P28; packing changes counts, never values |
| SIM14 | Energy verification = pins + witnesses, never RTL gates; provenance recorded per constant | there is no RTL energy truth (§9.3) |
| SIM15 | The per-operator engine assignment (`operator_realizations`) lives in each platform file | the platform file becomes the single statement of the architectural difference; `unavailable` is stated in the file, not implied; the workload library stays one code path routing through the table (§3.5) |
| SIM16 | One cycle law for every tile: the v2 frozen law parameterized by tile geometry and port width; Azure-Lily gets T_fill = 330 / T_steady = 264; the legacy axioms (330 / 256, double-buffered) are a differing witness — the same single-vs-double-buffer effect documented for NL-DPE (60 vs 52) | the crossbar behavior is identical and both output stages cost the same cycles; mixing buffering disciplines between platforms would make the comparison unfair (§5.3) |

Reconciliation items recorded (not new decisions): `pass_engine.py` becomes
the single home of the per-crossbar `T(p)` helper (the DIMM's private copy
delegates to it); the report layer uses one field name for "cycles" per node
with a posture tag, while the kernel simulators' own field names stay
untouched; the online-softmax charter's per-block pass wording is reconciled
to the implemented global packing (the implemented form is the contract).

---

## §13 Implementation guide (per phase: files, interfaces, tests, done-gates)

Written to be driven phase by phase. Every module is self-tested the
house way: `python3 <module>.py` ends `ALL PASS`. No phase starts before
the previous phase's done-gate is green. Nothing in `v2/oracle/`,
`v2/sim/simulator/kernels/{nldpe,gemm,dimm,softmax,softmax_online}_sim.py`, `v2/rtl/`,
`v2/tb/`, or `v2/smoke/` is ever edited (SIM2) — new files only.

### §13.0 File layout (pinned 2026-10-03)

The backbone simulator is self-contained in one folder. The certified
kernel simulators live in its `kernels/` subfolder (moved from `v2/sim/` on
2026-10-04; the only edit was the one-line oracle path in five of them —
values and timing untouched) and are otherwise never edited (SIM2); the
oracles stay in `v2/oracle/` as the independent ground truth.

```
v2/sim/simulator/
  configs/nl_dpe.json, configs/azure_lily.json   platform configs            phase 0  (built)
  platforms.py      Platform(config) + `--imc` CLI                           phase 0  (built)
  cost.py           per-operation prices + the per-tile cycle law            phase 1  (built)
  kernels/          certified kernel simulators (read-only, SIM2): nldpe_sim.py
                    pass_engine.py gemm_sim.py dimm_sim.py softmax_sim.py
                    softmax_online_sim.py                                    certified
  backbone.py       record types + workload interface; composition may
                    split into compose.py (phase-3 decision)                 phase 2-3
  workloads/        package (__init__.py); one adapter per kernel: gemm.py dimm.py softmax.py
                    softmax_online.py primitive.py                           phase 2
                    fabric_ops.py linear.py attention.py encoder.py
                    bert_tiny.py flash_attention_compare.py                  phase 4
  main.py           runner, archive convention: --imc <config> --model <workload>
                    [--seq_length N] [--json out.json]                       phase 4
  test_simulator.py aggregate runner of every self-test above                phase 5
```

Imports: modules in `v2/sim/simulator/` put their own folder and
`kernels/` on `sys.path`; files in `workloads/` put `parents[1]` (the
simulator folder) and `parents[1] / "kernels"`. The kernels reach the
oracles at `parents[3] / "oracle"` (`v2/oracle/`); `v2/smoke` generators
reach the kernels at `v2/sim/simulator/kernels/`. No file
may shadow a standard-library or certified module name (hence
`platforms.py`, not `platform.py`: numpy imports the standard `platform`).

### Phase 0 — platform files and loader

*As built (2026-10-03): `v2/sim/simulator/configs/*.json` +
`v2/sim/simulator/platforms.py` (`Platform(config, rows, cols, buffer_size)`);
the v0.2 plan below is kept for the record.*

Files:
- `v2/sim/simulator/configs/nl_dpe.json`, `.../azure_lily.json` —
  the three-part schema of §3 (fabric / compute_tile /
  operator_realizations), values exactly as tabulated in §3;
- `v2/sim/simulator/platforms.py` — loader + validation + derivation check.

Public interface:
- `load_platform(name: str) -> Platform` — reads
  `v2/sim/simulator/configs/<name>.json`;
- `Platform.fabric`, `Platform.compute_tile`,
  `Platform.operator_realizations`, `Platform.resource_budgets`;
- `Platform.derived_energy_constants()` — re-derives the tile energy
  constants from the power model (the `nl_dpe/area_power.py` chain) for
  the pin check;
- validation: schema check; **fabric equality across both files**;
  realization tables contain the §3.5 entries.

Tests (`python3 v2/sim/simulator/platforms.py`):
- derivation pins: 3.89 pJ exact to rounding; 0.171445313 pJ within 0.02%
  (the 0.013% drift documented in §3.4);
- both files parse; fabric sections byte-identical.

Done when: self-test ends `ALL PASS`, and the provenance tables in §3
match the archived files field by field.

### Phase 1 — cost functions

File: `v2/sim/simulator/cost.py` (*as built 2026-10-03: functions take a
`Platform`; see its docstring*).

Public interface (all pure functions of the platform + counts):
- `tile_pass_energy(tile, active_columns=None) -> dict` — walks the
  `pass_pipeline` stage list; per-stage breakdown + total;
  `active_columns=None` means all columns (the whole-array policy);
- `fabric_add(n)`, `fabric_compare(n)`, `fabric_activation(n)`,
  `dsp_mac(n)`, `lookup_table_read(n)` — pJ per count;
- `bram_accesses(n_bytes) -> int`, `bram_energy(n_bytes) -> float` —
  `ceil(bytes / (port_width/8))` accesses;
- `tile_cycle_law(tile, passes) -> CycleModel` — the one law of §5.3 for
  any tile, delegating to the certified `nldpe_sim.cycle_model` with the
  tile's parameters.

Tests: per-pass pins 75.01 / 53.07 pJ (NL at 256 / 128 columns); the
Azure-Lily per-pass pin ≈ 2401.1 pJ (§10.1); the §10.2 worked-example row
recomputes exactly; cycle-law pins (114/60 and 330/264).

Done when: self-test ends `ALL PASS` and every §10 number recomputes.

### Phase 2 — the record and the kernel adapters

Files:
- `v2/sim/simulator/backbone.py` — the record types and the workload interface;
- `v2/sim/simulator/workloads/` — one adapter file per kernel.

Record types:
- `CountedOperation(kind, node, count)` — energy-only entries;
- `ScheduledEvent(kind, node, start_cycle, end_cycle)` — time entries;
- `NodeRecord(name, kind, shapes, pass_counts, crossbar_counts,
  cycle_style, cycles, serialize_cycles, energy_by_component,
  operations, events)`;
- `Report(platform, nodes, totals, throughput)` + `write_json()` +
  `print_table()`.

Workload interface (every adapter implements it):
- `run(inputs, collect_values=False) -> NodeRecord` — wraps the kernel
  simulator call; never re-implements it.

Adapter content (what each translates):
- GEMM: per-tile pass counts → `tile_pass_energy`; `M·N·(V−1)`
  reduction adds; input/output port traffic; setup energy from
  `program_weights`;
- DIMM: pool/farm pass counts → `tile_pass_energy`; `M·N·K` feed adds;
  parked-buffer re-reads; `M·N` serialization; the shadow total is the
  cycle truth (style: model-total);
- row softmax / online softmax: EXP/LOG(/factor) pass counts; row-max
  compares; row-sum adds; clamps; block-stream traffic (online); cycle
  style: measured-schedule;
- crossbar primitive (standalone): pass energy + traffic; style:
  measured = formula.

Tests: each adapter's energy total equals an independent recomputation on
reference shapes (a recompute function inside the self-test); each
adapter's cycles equal the kernel's own contract value; a value run
through one adapter equals the kernel simulator's direct output.

Done when: all five adapters pass on reference shapes and no certified
file was touched (only new files under `v2/sim/simulator/`).

### Phase 3 — composition and the report

File: `v2/sim/simulator/backbone.py` (composition part) — or a `v2/sim/simulator/compose.py`
split if backbone.py grows; the split is a phase-3 decision, recorded
either way.

Public interface:
- `chain(*nodes)` — sequential composition; consumer starts at producer
  readiness (whole-output default; part-level when the consumer declares
  it and the producer exposes a readiness schedule, §6.3);
- `parallel(*nodes)` — combined time = max of finishes; resources
  provisioned per node (SIM8);
- `repeated(node_or_chain, n_inputs)` — first-input latency, steady
  interval = max per-input steady on the critical path, throughput =
  clock / steady interval (§6.4);
- `Report.from_chain(...)` + the JSON writer (§7.2 schema) + the table
  printer.

Data flow: adapters emit event lists relative to their own start;
composition shifts offsets along readiness edges and concatenates; the
report is a group-by over the merged record.

Tests: a hand-built two-node chain (offsets computed by hand and
compared); parallel = max; `repeated` steady interval = max steady;
report totals = sum of components = sum of nodes; JSON validates; a
two-node readiness test (consumer starts on part-level readiness,
offsets match a hand computation).

Done when: all composition self-tests pass.

### Phase 4 — the workload library

Files under `v2/sim/simulator/workloads/`:
- `fabric_ops.py` — elementwise add, activation (both forms of §8.2),
  layer normalization, embedding lookup, lookup reads;
- `linear.py`; `attention.py` (score matmul + softmax + weighted sum,
  both softmax variants, engines routed through `operator_realizations`);
  `encoder.py`; `bert_tiny.py`;
- `flash_attention_compare.py` — the Δ-cycles / Δ-energy runner
  (`flash_attention.md` §5): same shapes and corpus, blocked vs full-row.

Runner: `v2/sim/simulator/main.py --imc {nl_dpe,azure_lily}
--model {linear,attention,encoder,bert_tiny,flash_compare}
[--seq_length N] [--collect-values] [--json out.json]` — prints the
per-node table and the end-to-end numbers.

Tests: a BERT-Tiny reference run (NL-DPE, sequence length 128) produces
the per-layer table; its totals are frozen as the reference pin (§9.1);
the Azure-Lily run of the same workload completes through its realization
table (attention via the DSP-lane variant); the flash-attention
comparison prints Δ cycles and Δ energy.

Done when: the reference pins are green on both platforms and the
comparison table prints.

### Phase 5 — witnesses and frozen pins

Files: `v2/sim/simulator/test_simulator.py` (aggregating) + this charter updated
with the frozen pin values.

Content: the archived-core differing witness on shared shapes (GEMM
runs, identity-pass streams) with a documented-delta table; the softmax
study as a differing witness for the block-RAM granularity (§4.4); all
pins written into §9/§10.

Done when: witnesses run, deltas documented in this charter, and the
full chain is green:
`python3 v2/sim/simulator/test_simulator.py` (runs `platforms.py`,
`cost.py`, `backbone.py` and every `workloads/` self-test).

## §14 Out of scope / deferred

- streaming refinements beyond the fill + steady law (token-level overlap
  inside a layer);
- resource-feasibility mapping (fitting composed workloads into the
  platform resource budgets) — the budgets stay informational;
- area accounting (existing conventions live in the DSE and the softmax
  study);
- static / leakage energy;
- CNN workloads (convolution, pooling) — add only if the paper needs them;
- VTR linkage for the clock (the declared clock stays a platform field; a
  VTR-derived frequency can populate it later without touching the model).
