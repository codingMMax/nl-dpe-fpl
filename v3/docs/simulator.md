# v3 simulator: one behavior model per module

Status: 2026-10-14, reviewed by you (outcome in §10). Nothing here is implemented. Tags:
- **[derived]**: from the formulas here;
- **[estimate]**: not backed by a run.

The hardware is in `architecture.md`; the experiment that uses the simulator is in `experiment.md`.

## 1 Overview

**What the simulator is for.**
1. **The RTL gates.** For every module and for the head, it produces each test case's expected values and expected
   port events. The case writer runs the model, checks its values against the oracle (GATE 1), and writes the files
   the testbench compares against (GATE 2).
2. **The DSE.** It produces cycles, energy and resources for the 432 attention candidates and the 48 FC rows, from the
   same models.
3. **Later, full models.** Heads, layers, FFN and LayerNorm are built by composing the same modules.

**Central rule** (user, 2026-10-14): **one behavior model per RTL module.** A model executes what its module does:
crossbar passes, CLB stages, memories, start decisions. Each step:
- computes the values it produces;
- computes the cycle at which they appear;
- adds its operations to the counts.

Values, cycle counts and energy all come out of the same execution. There is no separate cycle model.

**Two things stay separate on purpose:**
- **The oracle** (`oracle/`): the math only, in numpy, with no hardware structure. It is the independent witness of
  GATE 1 and never shares code with a model.
- **Closed forms**, such as $T(p) = T_{fill} + (p - 1)T_{steady}$, the DIMM tile total or the softmax stream floor.
  They are tests that the behavior must reproduce under back-to-back arrivals, and they explain results. They are
  never a second model.

**Requirements.**

| item | requirement |
|---|---|
| exactness | values bit-exact to the oracle; every port event equal to the RTL testbench's ($\Delta = 0$) on the 64×64 corpora |
| granularity | the module boundary of `architecture.md`: the same module names, parameters, ports and window events |
| speed | one attention head in about a second; the 432-candidate sweep in minutes; a full model in minutes [estimate] |
| extensibility | new modules and new workloads without changes to existing modules; a second crossbar type (Azure-Lily) later, through the platform layer (§4.7) |
| reuse | the certified integer crossbar math, the pass grammar, the platform files and the prices, copied with sha256 provenance |

**Design principle: timed dataflow.** Time travels with the data. Every window carries its values and the cycle of
its first word. A module's step reads the times of its inputs and computes the times of its outputs by the hardware's
rules: the pass grammar, register stages, start decisions. Nothing steps cycle by cycle, and there is no global clock
and no event queue.

Two alternatives are rejected:
- **A cycle-by-cycle simulator.** It is an RTL written a second time: slow, and a second RTL to maintain.
- **A closed-form cycle model beside a value model.** That is two code paths that drift apart, and closed totals
  cannot be connected port to port. A layer model built from them has to re-implement its modules.

## 2 Trust chain and co-design

```text
 spec (architecture.md, ip/<module>/spec.md)
   -> oracle (numpy math)            -- GATE 1: model values == oracle, every stage, every case
   -> behavior model (this simulator) -- GATE 2: RTL values and every port event == model, every case
   -> RTL (sim twin; VTR twin checked against it by the structure gate)
```

Trust flows downward only, and the spec arbitrates. The model is the RTL's twin at the module boundary: it has the same
names, parameters, ports and events. That makes it both the RTL's reference and the DSE's engine.

**Co-design loop for a new module** (for example a LayerNorm):
1. Its spec fixes the port contract (the format of `architecture.md` §3).
2. Its oracle states the math.
3. Its model is written, with basis model-only. The DSE can already evaluate it.
4. Its RTL is written to the same ports.
5. GATE 2 pins the model's named latencies; the basis becomes rtl-pinned.

**For a changed module:** model and RTL change together. A latency changed on only one side fails GATE 2. A timing
that changes on purpose updates the module's closed-form test in the same step.

## 3 Interfaces

### 3.1 Time convention

- Time is counted in integer cycles of the design's clock (one clock domain). VTR's Fmax converts cycles to ns.
- Cycle 0 is the cycle `start` is high, as in the testbenches.
- A word's time is the cycle it is valid at its producer's output port.
- A window's time is the time of its first word. Word $c$ of the window is at time $+\,c$ (W1: windows are gapless).
- A credit's time is the cycle the consumer's pulse is high.
- Every latency between a producer's output and a consumer's decision is a named constant of the module that has it
  (for example a registered hand-off). Nothing is implicit.

### 3.2 Records

| record | fields | meaning |
|---|---|---|
| `Window` | `port`, `ch`, `unit`, `w`, `t`, `vals` | window `w` of unit `unit` on channel `ch`, first word at cycle `t`; `vals` = numpy array of its $W$ elements (`None` in timing-only mode) |
| `Credit` | `port`, `unit`, `t` | the consumer freed the buffer of `unit` at cycle `t` |
| `PortSpec` | `name`, `dir`, `discipline`, `E`, `Q`, `W`, `L`, `unit`, `CAP` | a port of `architecture.md` §3.1–§3.3; `discipline` ∈ {push + credit, burst on ready, push, pulse}; a pulse port is a one-bit wire with one pulse per unit (a credit, `x_rdy`, `run_done`) and carries no windows |
| `Step` | `process`, `unit`, `t_decide`, `t_start`, `t_done`, named events | one job: the times the testbench also logs |
| `Counts` | passes per (mode, rows, cols); CLB adds, compares, selects; memory reads and writes in bytes; FIFO words | the energy ledger, per module |
| `Resources` | crossbars per mode and extent; memories (words × bits, count) | per module |
| `Report` | cycles (last output word + 1); `Step` records; port events; `Counts`; `Resources`; energy; basis per module; values (optional) | one run of a workload |

The port events of a `Report` are written in the same line format as the testbenches' port monitors, so GATE 2 is a
plain comparison.

### 3.3 The module contract

**Declaration:**
1. **Name and parameters.**
   - `NAME` = the RTL module name; parameters keep their RTL names.
   - `derive(params)` returns the RTL's localparams, with the same formulas.
   - `guard(params)` rejects the same parameter sets as the RTL's elaboration guards.
2. **Ports.** One `PortSpec` per RTL port.
3. **Processes.** A process is an in-order job sequence of the module; its steps are the jobs. For example, the DIMM
   has an A-loader process (one step per input row or window), a B-loader process, and a tile-job process. A step
   declares what it needs:
   - windows of an input port, by count;
   - credits, by unit index;
   - progress of another process of the same module.
4. **Carried state**, and the rule that resets it.
5. **Basis**: rtl-pinned or model-only.
6. **Resources.**

**Behavior.** `step(process)` executes one job:
1. read the inputs (values and times);
2. compute the outputs (values and times) by the module's hardware rules;
3. emit the windows and the credits;
4. record the step's events;
5. add the counts.

**Invariants** (each one is tested, §7):
- **I1 pure:** a step depends only on its inputs, the credits it read and the module's state.
- **I2 in order:** a process steps its units in order, and a port's windows leave in order.
- **I3 causal:** an output's time is at least the time of every input it depends on plus the named latency. A credit
  is returned after the unit's last read.
- **I4 value-independent timing:** turning values off changes no time and no count.
- **I5 one source:** every latency the RTL has is a named constant in its module's model and spec, and nowhere else
  in the simulator.

## 4 Architecture

### 4.1 Structure

```text
 workloads:  attention_head(shape, config, mapping) -> Graph;  fc(K, N, config) -> Graph;  (later: encoder, model)
                 |
 Graph:      modules + edges (output port -> input port, CAP) + external ports; a Graph is itself a module
                 |
 modules:    gemm_top, dimm_tile_top, softmax_online_top, softmax_online_acc, att_ctrl, source, sink
             (simulator/models/, simulator/core/environment.py)
                 |  built from
 primitives: Crossbar, CrossbarGroup (values + pass timing, simulator/core/crossbar.py)   stages: Delay, Tree, Map, Fifo, Memory
                 |
 engine:     runs ready steps until every process is done  ->  Report (cycles, events, values, counts, resources, basis)
                 |
 platform:   configs, prices (energy), area model (crossbars exact + CLB / memory fitted from VTR)  ->  dse/
```

### 4.2 The crossbar primitive (`Xbar`, `XbarGroup`)

`Xbar` models one `dpe`. Its configuration: `R, C, BUF, P`, the weights (or identity), the output-stage mode and the
extent (rows, columns; `architecture.md` §1). On NL-DPE the output stage is the ACAM, with the modes REGULAR, EXP and
LOG; a mode the platform's output stage does not have is rejected. One call, `pass_(x, t_ready)`, does all of this:
- **Values:** an int8 MAC to int32, then the ACAM form of the mode ($\mathrm{trunc8}$, $\mathrm{EXP}$ or
  $\mathrm{LOG}$). This is the certified integer math (`nldpe_ref`).
- **Timing:** the certified pass grammar (`pass_engine.schedule_pass_sequence`), one pass at a time, on the
  crossbar's own state:

  ```text
  load_start    = max(t_ready, msb_fire_prev + 1)
  compute_start = max(load_start + LOAD, acc_free)          LOAD   = ceil(8 rows / BUF)
  msb_fire      = compute_start + P - 1
  acam_done     = max(msb_fire + 2, out_free)
  drain_start   = acam_done + 1                             OUTPUT = ceil(8 cols / BUF)
  drain_end     = acam_done + OUTPUT
  state:  msb_fire_prev = msb_fire;  acc_free = acam_done + 1;  out_free = drain_end + 1
  ```

  `BUF` (the port width in bits) and `P` come from the platform. On NL-DPE, `BUF` = 40, i.e. 5 bytes per cycle, and
  `P` = 8.

- **Result:** the output window (values, time `drain_start`) and the pass events, plus one pass added to the counts.

`XbarGroup(n)` models $n$ crossbars run in lock-step from one leader, as the RTL does: a DIMM wave, an accumulator
group, a GEMM's tiles. It has one timing state and the values of all $n$ crossbars computed in one numpy operation.
The DIMM's slices are lock-step by design (`architecture.md` §4.4), so the model runs one timing state for the whole
farm. The slices appear only in the resources.

### 4.3 Stages (inside modules only)

| stage | values | time |
|---|---|---|
| `ready(*t)` | — | a decision: the first cycle at which every condition holds (the latest time; `None` = no condition) |
| `Delay(k)` | unchanged | $+k$ |
| `Tree(levels)` | exact integer sum of the inputs | $+$ levels (registered) |
| `Map(fn, k)` | `fn` per element: narrowing, sign, clamp, compaction | $+k$ |
| `Serial()` | — | one unit at a time: a unit ready at $t$, busy $n$ cycles, starts at $\max(t$, the end of the previous unit$)$ (a word writer, a reader, a packer's output) |
| `Buffer(entries)` | — | entries in rotation: unit $u$ may enter only after unit $u -$ `entries` released its entry; an early unit is a producer overrun (an error) |
| `Repack(sizes, width, lat)` | — | a byte stream re-cut into words: output word $k$ leaves `lat` after the read of the input word holding its last byte, one per cycle |
| `Fifo(depth)` | unchanged | a word is read at $\max$(its write time + 1, the reader's decision) |
| `Memory(slots)` | the words of each slot | a slot is valid at its last write + 1; a read returns one cycle after its address |

The $k$ of each use is a named constant of its module (I5); a plain fixed latency is just $t +$ `NAME_LAT` in the
module. Stages never connect two modules: modules meet only at their ports (`architecture.md` §1). A module model is
the wiring of its spec's blocks: each block is one stage or crossbar object, created once, and each step calls them in
order (for example `simulator/models/dimmTile.py`: converters, writers, staging buffers, slots, the farm).

### 4.4 Module models

| module | processes (one step = ) | what a step computes | built from | must reproduce |
|---|---|---|---|---|
| GEMM `gemm_top` | rows (one row) | the $V \times H$ tiles as an `XbarGroup` (REGULAR); byte tree; $\mathrm{trunc8}$; output words timed by the pass grammar plus the tree and serializer stages | v2 `gemm_sim` (values), the `gemm_top` structure | `gemm_sim.cycle_model`; the 48 FC snapshot rows |
| tiled DIMM `dimm_tile_top` | A loader (one input row or window), B loader (one input row), tile (one job) | loaders: LOG converters (`Xbar`, LOG), distribution, staging, slot writes. Tile: the decision; the start latency; per wave one `XbarGroup` pass of the farm on $\mathrm{EXP}(\mathrm{trunc8}(L_A + L_B))$; signs; the tree; narrowing; the output windows | v2 `dimm_tile_sim` (values and its RTL-pinned rules) | `dimm_tile_sim.tile_cycle_model` for back-to-back jobs |
| online softmax `softmax_online_top` | input stage (one tile), block (one key block), run end (one run) | FIFOs and packer; store; row max; per EXP lane the feed and its passes; row sums; the carry passes; $\log\alpha$ events; the $\log l$ copy | v2 `softmax_online_sim` | `softmax_online_cycle_model` with every score available at start |
| accumulator `softmax_online_acc` | block (one tile: its rescale and LOG jobs, wave by wave), run end (one run: the epilogue jobs, the O reader) | the LOG and EXP groups (an `XbarGroup` each) under the job rules; the O reader (`Serial`); the $O_8$ rows | v2 `ocarry_sim` | `ocarry_cycle_model` fed v2's station inputs |
| controller `att_ctrl` | admission (one token), job lists (one job), runs (one run) | the rules of `architecture.md` §4.7 | new | — |
| source, sink | X source (one burst on `x_rdy`), $O_8$ sink (takes every word) | the head's environment | new | — |

Each model starts from the certified v2 kernel named in the table: its values, and its RTL-pinned timing rules.
"Must reproduce" means that, under the conditions the v2 closed form was pinned for, the behavior model gives the same
cycles. Where the datapath is unchanged, this carries the v2 RTL co-verification over to the new model.

### 4.5 Graphs and workloads

- `Graph(name, modules, edges, ports, credits)`: the edges are the RTL's wires.
  - A **data edge** connects an output port to one or more input ports (fan-out). Written `(a, b, ch)`, it puts the
    producer's channel $c$ on the consumer's channel $c + ch$, so several producers can feed one multi-channel port
    (the GEMM copies into a DIMM port). Every consumer channel has exactly one producer.
  - A **credit edge** `(giver, taker)` routes the credits a port gives (a credited input, or a pulse output) to the
    port that takes them (a credited output, or a pulse input). By default a credited consumer credits its only
    producer, so a one-to-one edge needs no credit edge. A credit edge is needed when the RTL wires the credit to a
    controller (the DIMMs' staging credits go to `att_ctrl`), or when the producer has several consumers.
  - A push producer may feed a credited consumer whose credits are routed to a controller: the controller then holds
    the credit loop (`att_ctrl` admits a token only when the staging has room).
  - A pulse output may stay open: its pulses are dropped and not logged (the accumulator's `run_done` in its own test
    graph).
  - The graph's external ports make it usable as a module.
- **Flattening:** the engine runs the flattened graph, with hierarchical names such as `head.qk.a_loader`. Composition
  therefore costs nothing at run time.
- **Workload builders** (`simulator/models/fc.py`, `attentionHead.py`):
  - `attention_head(shape, config, mapping)`: the graph of `architecture.md` §4.1, with the same instances and
    connections as `att_head` (checked by the structure gate);
  - `fc(K, N, config)`: one GEMM with a row source and a sink;
  - later, `encoder_layer` (heads + FFN + LayerNorm + residual add) and `model` (layers in sequence or pipelined),
    each a graph of graphs.
- **Schedules are controllers.** How heads share one engine, or how layers pipeline, is a controller module like
  `att_ctrl`, never code inside a module.

### 4.6 The engine

1. Flatten the graph. Create one window queue per edge channel and one credit queue per edge. Give each producer its
   `CAP`.
2. Repeat: pick any process whose next step is ready, and run that step. A step is ready when:
   - its input windows are queued;
   - its credits are queued;
   - the processes it depends on have progressed far enough.
3. Stop when every process has finished its units. If no process is ready before that, raise a deadlock error that
   lists what each waiting process needs.
4. Build the `Report`.

**Determinism:** the result does not depend on which ready process runs first. A step reads only finished data (I1),
so any order gives the same values, times and counts. A test runs every workload with random pick orders.
`architecture.md` §5.3 guarantees that some process is always ready.

### 4.7 Platform: energy, area, Fmax

- **Platform** (`simulator/platform/`; user, 2026-10-14). A platform is one crossbar type. Its config gives the crossbar's
  `BUF`, `P` and output stage, the prices and the crossbar area. NL-DPE is the only platform implemented. The layer
  stays open for a second type (Azure-Lily) later:
  - no model hard-codes an NL-DPE value: `Xbar` reads `BUF`, `P` and its output stage from the platform, and prices
    and area come from the platform files;
  - the DIMM, the online softmax and its accumulator are built on EXP and LOG passes, so they are NL-DPE modules. An
    Azure-Lily head would be a different graph, with its own modules for those steps;
  - nothing for Azure-Lily is built now: no module models, workloads or tests. Its config comes along with the
    platform copy only because the copied self-tests load both configs.
- **Energy** = $\sum$ counts × prices, from the platform files and `cost.py`:
  - crossbar passes: the whole array fires (the cells, the ACAM on every column, the output buffer and the drain);
    only the rows in use are read, so the input-buffer bytes and the input port words follow `ROWS_USED` (P29;
    user 2026-10-17, before that every pass was priced as the full array on both sides);
  - CLB adds and compares;
  - memory bytes per access;
  - FIFO traffic.

  Weight programming is reported separately as setup, never in the per-inference energy. The counting rules are those
  of `v2/sim/spec/simulator.md` §4. FIFO traffic is counted as memory bytes (each word written once and read once),
  and the DIMM counts one pairing add per product and one compare per narrowed output, as v2's attention ledger did
  (2026-10-15). CLB selects are counted but not priced, as in v2.
- **Area:**
  - crossbars are exact: count × the arch XML footprint;
  - memory blocks are predicted (`area.predicted_mem_blocks`, 2026-10-15): every module lists its memories (words ×
    bits), each takes its cheapest dual-port mode of the arch's 20 Kb block (1024 × 20 or 2048 × 10: ⌈bits / 20⌉
    blocks up to 1024 words), and copies that share every address and write enable (`Resources.mem_shared`: the
    accumulator's per-crossbar slot memories) pack by their summed width. At the head smoke point the inventory equals
    the RTL's 70 inferred memories and the prediction equals VPR's packed count (376);
  - CLB comes from VTR per row.
- **Fmax** comes from VTR per row; latency in ns = cycles × 1000 / Fmax.

### 4.8 Layout

```text
v3/
  simulator/             the simulator, one package (its README.md is the map and the rules of the code):
    core/                  cycleMath, errors, records, module, graph, engine, stages, environment (Source, Sink),
                           crossbar (Geometry, Crossbar, CrossbarGroup, outputStage: the dpe primitive)
    models/                gemm, dimmTile, softmaxOnline, softmaxAccumulator, attentionController (the module
                           models); fc, attentionHead (the workloads that wire them)
    platform/              platform (+ nl_dpe.json, azure_lily.json), prices, area; rows (a run priced into a DSE
                           row), breakdown (latency by phase, energy by role, area by component)
    tests/                 the module tests; checkEquivalence.py (the gate against the frozen pre-refactor outputs)
  ip/<module>/           spec.md, rtl/, tb/, cases.py (the case writer: stimulus -> model run -> GATE 1 against the
                         oracle -> the testbench's expected files, with the weights as $readmemh files the testbench
                         preloads; PLAN.md R8), gen_cases.py; the model is imported from simulator/models/
  oracle/                the numpy oracles (copies) and v2_kernels/: the ten v2 kernels, closed-form witnesses for
                         the tests, never models (what the simulator is checked against)
  vtr/                   the VTR flow: arch/, clock.sdc, runner, DSE driver, watcher
  dse/                   scripts (rows, ranking, breakdown, figures, page), results/ (the fc_ / attn_ VTR and priced tables, adp/, edap/) (calls simulator/)
  tools/                 structure gate, copy lock, the F1 / F2 gates
```

(Until 2026-10-17 the models lived in `ip/<module>/model.py` and the core in `sim/`; the refactor moved them into
the `simulator/` package without changing their behavior: `simulator/README.md`, "Equivalence gate".)

## 5 Policies

| policy | rule |
|---|---|
| domain | A model's guard is its RTL's guard. It runs outside that guard only with `model_only=True`, and every result then carries basis model-only. |
| named latencies | Every RTL latency is a named constant in its module's model and spec (I5). GATE 2 fails when one side changes alone. |
| pinned against | Each model records the sha256 of the RTL sources its last GATE 2 passed on. A changed RTL marks the model stale until the gate passes again. |
| basis | A `Report` carries the basis of each module. A workload result is model-only if any module or any arrival pattern in it is. |
| timing-only | `values=False` runs the same steps without the arithmetic; times and counts stay identical (I4). |
| independence | The oracle imports no model code and the model imports no oracle code. Only the case writer uses both. |
| no glue | Modules connect only through ports. A transform a connection needs belongs to a module. |

## 6 Scaling

- **Work grows with steps, not with cycles.** A step is one token row, one wave of a lock-step group, one loader group
  or one key block.
- **Example**, ViT-B/16 at 128×64, reference candidate [estimate]: about 5,000 steps, each a few numpy operations,
  i.e. well under a second:
  - 512 QK waves and 512 PV waves;
  - 256 tokens × 3 GEMMs, plus their converters;
  - about 1,000 softmax passes;
  - about 1,000 accumulator waves.
- **The 432-candidate sweep** takes minutes [estimate]. A 12-layer, 12-head model is about 150 heads, i.e. minutes
  [estimate]. Sweeps can run timing-only.
- **Memory:** events are kept as arrays (time, unit, window, channel). Values are kept only when asked for.

## 7 Validation plan

| gate or test | checks |
|---|---|
| structure | every model against both RTL twins: name, parameters, derived values at a set of builds, guards, ports. The head workload against `att_head`'s instances and connections. |
| GATE 1 (in each case writer) | the model's values at every stage == the oracle's, bit-exact |
| GATE 2 (the RTL gates of `architecture.md` §7) | the RTL's values and every port event == the model's ($\Delta = 0$) |
| counts | the testbench's crossbar pass counters per mode == the model's counts |
| closed forms | back-to-back runs reproduce `nldpe_sim.cycle_model`, `gemm_sim.cycle_model`, `tile_cycle_model`, `softmax_online_cycle_model` and `ocarry_cycle_model` wherever the datapath is unchanged |
| FC regression | the 48 FC snapshot rows: cycles and energy |
| determinism | random pick orders give identical Reports |
| timing-only | values off: identical events and counts |
| composition | a graph run as a module == the same graph flattened |
| mutants | a latency +1, a dropped credit, a wrong extent: each must fail at least one check above |

## 8 Implementation phases

Each module's model exists before its RTL phase, so the RTL is written against the model's events.

| phase | work | accepted when | with the RTL (`architecture.md` §8) |
|---|---|---|---|
| S0 | records, graph, engine, stages; `Xbar`, `XbarGroup` | the crossbar's v2 corpus (values and every pass event, including the P29 corpus); `nldpe_sim.cycle_model` reproduced | — |
| S1 | GEMM model, FC workload | the GEMM closed form; the 48 FC rows | — (the GEMM is unchanged) |
| S2 | DIMM model: ports, converters, loaders, slices, slots, jobs, narrowing | GATE 1; `tile_cycle_model` reproduced for back-to-back jobs | before phase A: its events are phase A's GATE 2 reference |
| S3 | online softmax and accumulator models | GATE 1; their closed forms | before phases B and C |
| S4 | `att_ctrl`, source, sink, the head workload | the head's values == `head_stages`, stage `O8`; determinism; composition | before phase D (gate L) |
| S5 | energy, area, Report, DSE sweep | the FC rows' energy; a sweep of the 432 candidates | — |
| S6 (later) | new modules (FFN activation, LayerNorm, residual add); encoder and model workloads | GATE 1 per module; basis model-only until their RTL exists | — |

**As built, the refactor (2026-10-17):** the notes below name the files as they were built (`ip/<module>/model.py`,
`sim/core`, `sim/stages.py`, `sim/workloads`, `sim/platform`). On 2026-10-17 the simulator was rewritten for
readability into the `simulator/` package (§4.8; `simulator/README.md`: the layout, the naming rules, the checks
kept) with the same behavior: every frozen Report of the 72 attention and 48 FC DSE builds (cycles, every event,
every step, counts, resources), every DSE row and every case file of the 8 corpora is reproduced exactly, and random
cases through both simulators give equal Reports (`simulator/tests/checkEquivalence.py` against the outputs frozen in
`work/equivalence/ref/`). The old files were removed on 2026-10-17 (git `816807b` is the restore point).

**As built, S0 and S1 (2026-10-14; gate `tools/f2_check.py`):**
- **Code:**
  - `sim/core/{records,engine,env}.py`, `sim/stages.py`;
  - `ip/xbar/model.py` (`Xbar`, `XbarGroup`), `ip/gemm/model.py` (`gemm_top`);
  - `sim/workloads/fc.py`; pricing in `sim/platform/{prices,area}.py` and `dse/fc_rows.py`, outside the model stack.
- **Burst on ready** is a credit with `CAP` = 1. Its time is the cycle `MSB_SA_Ready` rises ($t_{msb} + 1$), and the
  burst starts in that cycle.
- **The engine checks every edge:**
  - no unit is sent before its credit;
  - windows on a channel go in order, without overlap;
  - credits come back in order, each after the last word of its unit.
- **Named latencies:**
  - the pass grammar's `RDY_LAT`, `SHIFT_ACC_LAT`, `ACAM_LAT` and `DRAIN_LAT`, 1 cycle each;
  - the GEMM serializer's `SER_LAT` = 1, so $L_w$ = `TREE_PIPE` + 1.
- **Port event lines** have the form `<t> <inst.port> win ch<c> u<unit> w<w>` and `<t> <inst.port> cr u<unit>`. The
  v3 testbench port monitors will print the same lines (F3).
- **Finding:** the v2 closed form `nldpe_sim.cycle_model` loads in $\lceil rows \cdot P / BUF \rceil$ cycles, while the
  RTL and the v2 behavior model use $\lceil 8\,rows / BUF \rceil$. The two agree at $P = 8$, the only value used so far.
  The v3 model follows the RTL.

**As built, S2 and S3 so far (2026-10-14; F3):**
- **Models:** `ip/dimm/model.py` (`dimm_tile_top`) and `ip/softmax/model.py` (`softmax_online_top`), each with its
  case writer `cases.py`, corpus generator `gen_cases.py` and tests `test_model.py`. Both are built from the stages
  above (`Serial`, `Buffer`, `Repack`, `ready`) and the crossbar objects; refactored onto them on 2026-10-14, with every
  case of both corpora regenerated byte for byte.
- **Accumulator** (`softmax_online_acc`): `ip/softmax/acc_model.py` with `acc_cases.py`, `acc_gen_cases.py` and
  `test_acc_model.py`. Blocks: the LOG and EXP groups (`XbarGroup`), the O reader (`Serial`); job decisions by `ready`.
  `Fifo` and `Memory` are still unused: the o_part FIFOs and the slot memories add no timing rule of their own (a LOG
  job waits for its whole windows, and the engines read each slot after its producer drained it).

**As built, S4 (2026-10-14; F4):**
- **Engine:** data edges with fan-out and channel placement, pulse ports, and credit edges (§4.5). Existing graphs
  run unchanged: the core tests pass, and the DIMM, softmax and accumulator corpora regenerate byte for byte. New
  tests (`sim/tests/test_core.py`): a hand-computed fan graph (a source into two takers and a gate, the takers into
  one two-channel joiner, every credit routed), and its errors.
- **Controller and head:** `top/model.py` (`AttCtrl`), `sim/workloads/attention_head.py` (`head_params`,
  `attention_head`: the netlist of `att_head`), the head's case writer `top/cases.py` and tests `top/test_model.py`.
  The accumulator's `run_done` became a pulse port.
- **GATE 2** runs through one shared runner, `tools/rtl_gate2.py` (each module's `tb/run_*_rtl.py` names its
  testbench and RTL). A case passes when the testbench reports PASS, every event equals the model's, and the total
  cycles (last word + 1) are equal; `--recheck` re-checks existing RTL logs against regenerated model results without
  a simulation. Each case's `case.json` carries its plusargs and every port's words per window.

## 9 Risks and decision gates

| risk | gate | response |
|---|---|---|
| a behavior model misses an RTL timing rule (take-now acceptance, the drain queue, registered decisions) | the closed-form tests, then GATE 2 | port the rule from the v2 RTL-pinned kernel; never fit a constant to a measurement without naming its mechanism |
| speed for full models | S4: one ViT-B/16 head in ≤ 2 s; S6: a full model in ≤ 10 min [targets, accepted 2026-10-14] | more vectorization; timing-only mode |
| the engine depends on the pick order | the determinism test | find the step that reads unfinished data |
| timing-only mode drifts from the full run | the timing-only test | — |
| model-only modules inside a full-model result | the Report's basis | flag the row |

## 10 Review outcome

You reviewed this document on 2026-10-14:
1. **Platform:** the platform layer stays. NL-DPE is the only platform implemented, and the layer stays open for
   Azure-Lily later; nothing for it is built now (§4.7).
2. **Speed targets:** the §9 targets are accepted with the document.
