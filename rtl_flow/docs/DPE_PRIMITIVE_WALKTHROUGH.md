# DPE Primitive Overlap — Implementation Walkthrough

**Companion doc for verifying Task #82 (DPE primitive overlap refactor).**

**Workload-level walkthrough**: see `FC_GEMM_WALKTHROUGH.md` for `fc_top.v` (V·H weight-stationary array, CLB tree, output mux), `tb_fc.v`, and `run_fc_smoke.py`. The DPE primitive described below is the *lane-level hardware*; a `fc_top` instance places V·H of these primitives in parallel via `generate for (gv) for (gh)`.

Code reference: `fc_verification/rtl/dpe_nldpe.v`, `dpe_azurelily.v`, `dsp_mac.v` (faithful NBA-FSM behavior model). The Azure-Lily DPE and DSP-MAC primitives mirror the same FSM structure with arch-specific localparams; everything below uses NL-DPE numbers (R=C=256, BUF=40, INT8 → LCYC=52, CCYC=10, OCYC=52, T_steady=52) for concreteness.

**Cycle formulae** (Option A, post Task #90 methodology roll-back):

```
# Sim (ideal analytical model — §4 architectural):
T_fill_ideal = LOAD + COMPUTE + OUTPUT     (no implementation overhead)
T_steady     = max(LOAD, COMPUTE, OUTPUT)
T_sim(M)     = T_fill_ideal + (M-1) × T_steady

# RTL (faithful FSM with NBA register propagation):
T_fill_rtl   = LOAD + COMPUTE + OUTPUT + 2 (+2 = FSM register-propagation overhead)
T_rtl(M)     = T_fill_rtl + (M-1) × T_steady

# Fidelity (reported, not gated):
fidelity = (T_rtl - T_sim) / T_sim = 2 / T_sim(M)
```

For NL-DPE INT8 R=C=256 BUF=40, M=1: T_sim = 114, T_rtl = 116 → fidelity = +1.75%.
For M=8: T_sim = 478, T_rtl = 480 → fidelity = +0.42% (the +2 amortises across passes).

**The +2 in T_fill_rtl is the RTL's faithful FSM handoff cost:** 1 cycle for
`LOAD→COMPUTE` (q_load_tail NBA-propagation) plus 1 cycle for
`COMPUTE→OUTPUT` (q_compute_head NBA-propagation). The pre-Task-#87
model used blocking same-cycle pulses to collapse these into 0 cycles,
which was implausible for real silicon FSMs. **The +2 is real silicon
overhead of OUR particular RTL design — NOT part of the methodology
formula in FIDELITY_METHODOLOGY.md §4.** A different RTL implementation
(combinational handoffs, 2-sub-FSM, etc.) would yield a different
overhead. The simulator emits the ideal `L + C + O`; the RTL pays +2;
fidelity is the honest measurement of that gap.

Anchored to `fc_verification/FIDELITY_METHODOLOGY.md` §4 (architectural
pipeline) + §4.2 (RTL implementation overhead surfaced as fidelity).

---

## 1. Module interface — what changed, what didn't

`dpe_nldpe.v:47-69`

```verilog
module dpe #(
    parameter KERNEL_WIDTH   = 256,
    parameter NUM_COLS       = 256,
    parameter DPE_BUF_WIDTH  = 40,
    parameter COMPUTE_CYCLES = 10,    // NEW (default 10 = INT8 + 3-stage pipeline - 1)
    parameter ACAM_MODE      = 0
)(
    input  wire clk, reset, ...        // unchanged port list
    input  wire [1:0] nl_dpe_control,
    ...
);
```

- **Port list is byte-identical to legacy** — VTR arch XML `<model name="dpe">` contract preserved. VTR sees the same hard block.
- **Only addition**: `COMPUTE_CYCLES` parameter (compile-time, not in port list).

---

## 2. Internal storage — single buffer, single ring queue

`dpe_nldpe.v:87-93`

```verilog
reg signed  [7:0] weights      [0:KERNEL_WIDTH-1][0:NUM_COLS-1];   // R×C weight tile (TB hierarchical-forces)
reg signed  [7:0] input_buffer [0:KERNEL_WIDTH-1];                  // ONE input buffer (single-buffered)
reg signed [31:0] vmm_queue    [0:QDEPTH-1][0:NUM_COLS-1];          // depth-4 ring buffer of completed accumulators
```

Three storage elements:

| Storage | Role | Lifetime |
|---|---|---|
| `weights` | R×C weight tile | One-time, hierarchical-forced by TB |
| `input_buffer` | Single physical buffer | Overwritten each pass |
| `vmm_queue` | Ring buffer of completed VMM results | Holds completed outputs between COMPUTE and OUTPUT phases |

**The "single-buffered" claim** (per FIDELITY_METHODOLOGY §4) refers to `input_buffer` being **one** physical array. The trick that makes single-buffer overlap correct: VMM math fires on the *last strobe* of each pass with **blocking assignments**, committing the result to `vmm_queue` *before* the next pass's first strobe overwrites `input_buffer`.

`vmm_queue` is **NOT a second input buffer** — it holds outputs awaiting OUTPUT drain. Depth=4 covers max 3 passes in flight (LOAD just finished + COMPUTE running + OUTPUT running) with one slot of headroom.

---

## 3. Three parallel sub-FSMs

All three live in **one** `always @(posedge clk or posedge reset)` block, but they're functionally independent — each has its own busy flag, counter, and queue index.

### 3a. LOAD sub-FSM (Task #87 Phase 1)

```verilog
if (w_buf_en) begin
    if (load_cycle_idx == LOAD_CYCLES - 1) begin
        // ── Last strobe of pass: BLOCKING fire ──
        for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)
            if (load_count + b < KERNEL_WIDTH)
                input_buffer[load_count + b] = data_in[b*8 +: 8];   // BLOCKING (=)
        // Fire VMM into vmm_queue[q_load_tail]
        for (c = 0; c < NUM_COLS; c = c + 1) begin
            vmm_queue[q_load_tail][c] = 0;                          // BLOCKING
            for (r = 0; r < KERNEL_WIDTH; r = r + 1)
                vmm_queue[q_load_tail][c] = vmm_queue[q_load_tail][c]
                                          + input_buffer[r] * weights[r][c];
        end
        // ACAM modes 1/2 transform the result here
        load_count      <= 0;
        load_cycle_idx <= 0;
        q_load_tail     <= (q_load_tail + 1) % QDEPTH;              // NBA: COMPUTE reads next cycle
    end else begin
        // ── Non-final strobe: NBA write into input_buffer ──
        for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)
            if (load_count + b < KERNEL_WIDTH)
                input_buffer[load_count + b] <= data_in[b*8 +: 8];  // NBA (<=)
        load_count      <= load_count + ELEMS_PER_STROBE;
        load_cycle_idx <= load_cycle_idx + 1;
    end
end
```

**Key correctness invariant — why blocking on the last strobe?**

The VMM math reads `input_buffer[r]` and `weights[r][c]` to compute `vmm_queue[q_load_tail][c]`. If the last strobe's bytes were NBA-written, the math would read *stale* values (the previous strobe's bytes). So:
1. The last strobe writes `input_buffer` BLOCKING.
2. The VMM math reads the freshly-loaded bytes BLOCKING.
3. The result is committed to `vmm_queue` BLOCKING.

All in one cycle. Once `vmm_queue[q_load_tail]` holds the result, `input_buffer` is *logically free* — the next pass's first strobe (NBA-written next cycle) overwrites byte 0 without losing anything.

`q_load_tail` is **NBA-advanced**, so COMPUTE's idle branch reads
`n_pending_compute > 0` one cycle later. That's the LOAD→COMPUTE
handoff cost (Task #87 Phase 1: 1 cycle of register propagation).

**This is the single-buffer correctness argument in two lines of Verilog**: "fire VMM with blocking; NBA-advance q_load_tail."

### 3b. COMPUTE sub-FSM (Task #87 Phase 1)

```verilog
if (compute_busy) begin
    if (compute_cycle + 1 >= COMPUTE_CYCLES) begin
        compute_busy     <= 0;
        compute_cycle    <= 0;
        q_compute_head   <= (q_compute_head + 1) % QDEPTH;          // NBA: OUTPUT reads next cycle
        MSB_SA_Ready     <= 1;
        shift_add_done   <= 1;
    end else begin
        compute_cycle <= compute_cycle + 1;
    end
end else begin
    // Wake one cycle after LOAD's last strobe (q_load_tail's NBA
    // must commit before n_pending_compute > 0 is observable).
    if (n_pending_compute > 0) begin
        compute_busy     <= 1;
        compute_cycle    <= 0;
        MSB_SA_Ready     <= 0;
        shift_add_done   <= 0;
    end
end
```

- COMPUTE is busy for `COMPUTE_CYCLES` cycles per pass.
- **No overlap across passes** — that's the one phase that's serialized per pass, modeling the bit-serial pipeline drain.
- When COMPUTE finishes, NBA-advances `q_compute_head`. OUTPUT's idle branch reads `n_pending_output > 0` one cycle later (COMPUTE→OUTPUT handoff: 1 cycle).
- When idle, polls **only** `n_pending_compute > 0`. Pre-Task-#87 also checked `fire_now` (a blocking same-cycle pulse) which bypassed the +1 handoff cost; that pulse has been removed.

### 3c. OUTPUT sub-FSM (Task #87 Phase 1)

```verilog
if (output_busy) begin
    data_out <= 0;
    for (b = 0; b < ELEMS_PER_STROBE; b = b + 1)
        if (output_col_idx * ELEMS_PER_STROBE + b < NUM_COLS)
            data_out[b*8 +: 8] <= vmm_queue[q_output_head][output_col_idx * ELEMS_PER_STROBE + b][7:0];
    if (output_col_idx + 1 >= OUTPUT_CYCLES) begin
        q_output_head  <= (q_output_head + 1) % QDEPTH;
        output_col_idx <= 0;
        // Chain to next pass when COMPUTE has already advanced past
        // the current OUTPUT pass.  Saves the next-pass COMPUTE→OUTPUT
        // handoff cycle by keeping output_busy=1.
        if (n_pending_output > 1)
            output_busy <= 1;
        else
            output_busy <= 0;
    end else
        output_col_idx <= output_col_idx + 1;
end else begin
    // Wake one cycle after COMPUTE done.
    if (n_pending_output > 0) begin
        output_busy    <= 1;
        output_col_idx <= 0;
    end
end
```

- OUTPUT drives `data_out` for `OUTPUT_CYCLES` cycles per pass.
- Each cycle reads the next column slice from `vmm_queue[q_output_head]`.
- After draining all OUTPUT_CYCLES strobes, advances `q_output_head` and **chains directly to the next pass's OUTPUT** when `n_pending_output > 1` (i.e., COMPUTE has already advanced past the current OUTPUT pass). Pre-Task-#87 also chained on `compute_done_now` (blocking same-cycle pulse from COMPUTE); that pulse has been removed — but the chain condition `n_pending_output > 1` already covers all post-Task-#87 timing because pass m+1's COMPUTE done happens 1 cycle BEFORE pass m's OUTPUT last strobe, so `q_compute_head` has been NBA-advanced in time.

---

## 4. NBA register propagation — the faithful FSM handoff (RTL-side overhead)

This is the subtle part. The behavior model implements a faithful FSM
in which **each stage handoff costs exactly 1 cycle of register
propagation** — matching real silicon, where state transitions cannot
happen in 0 cycles.

**Two cycle models live side-by-side (Option A, post Task #90):**

- **Sim emits `T_fill_ideal = L + C + O = 114`** (architectural §4 formula —
  no implementation overhead).
- **RTL exhibits `T_fill_rtl = L + C + O + 2 = 116`** (faithful FSM with NBA
  register propagation).

The +2 cycle gap surfaces as +1.75% fidelity at M=1, decreasing with M
(steady-state cadence amortises the +2). For NL-DPE M=8: 480 RTL vs
478 sim → +0.42% fidelity.

For NL-DPE M=1 (LCYC=52, CCYC=10, OCYC=52), the RTL cycle layout is:

```
LOAD          gap COMPUTE       gap OUTPUT
[cycle 0..51] [52]  [53..62]     [63]  [64..115]   T_rtl(1) = 116
              ^                  ^
       LOAD→COMPUTE         COMPUTE→OUTPUT
       handoff (1 cycle)    handoff (1 cycle)
```

The mechanism: sub-FSM coupling is via **NBA queue indices**, not blocking
pulses. Pre-Task-#87 used combinational temporary regs (`fire_now`,
`compute_done_now`) to communicate *within the same posedge tick*; that
collapsed both handoffs into 0 cycles and yielded `T = L + C + O =
114` exactly — implausibly tight for real silicon. The faithful model
removes those pulses; consumers read the producer's NBA'd state one
cycle later, paying +2 in T_fill_rtl.

**Why surface this as fidelity rather than bake into the sim**: the +2
is implementation-specific to OUR RTL design choice (NBA queue
propagation). A different RTL (combinational handoffs, single-cycle
state register, 2-sub-FSM merge) would yield a different number.
Baking +2 into the simulator would be circular validation by
construction. Per FIDELITY_METHODOLOGY §1 / §4.2: sim emits the
architectural model; RTL discloses the implementation cost; fidelity
is the honest measurement.

| Cycle | LOAD does | COMPUTE does | OUTPUT does |
|---|---|---|---|
| 51 | last strobe → fire VMM blocking, NBA `q_load_tail<=1` | `compute_busy=0` entering; `n_pending_compute=0` (q_load_tail NBA pending); stays idle | — |
| 52 | (idle) | `q_load_tail=1` (NBA'd); `n_pending_compute=1`; NBA `compute_busy<=1`, `compute_cycle<=0` | — |
| 53 | (idle) | `compute_busy=1`, `compute_cycle=0`; NBA `compute_cycle<=1` | — |
| 53..62 | (idle) | counts cycles 0..9 (10 cycles total) | — |
| 62 | (idle) | `compute_cycle=9` → done; NBA `compute_busy<=0`, `q_compute_head<=1` | `n_pending_output=0` (q_compute_head NBA pending); stays idle |
| 63 | (idle) | (idle) | `q_compute_head=1` (NBA'd); `n_pending_output=1`; NBA `output_busy<=1` |
| 64 | (idle) | (idle) | `output_busy=1`; NBA byte 0 from `vmm_queue[q_output_head]` |
| 64..115 | (idle) | (idle) | drains OCYC=52 bytes |

**Why the +2**: each handoff (LOAD→COMPUTE at cycle 52, COMPUTE→OUTPUT
at cycle 63) consumes 1 cycle of NBA-propagation. The previous model's
blocking pulses bypassed this by communicating combinationally in the
same tick — efficient for simulation but unfaithful to real silicon.

T_steady = max(L,C,O) is **unchanged** because subsequent passes
amortise their handoff costs into the existing pipeline depth via the
OUTPUT chain mechanism (`n_pending_output > 1` keeps `output_busy=1`
across pass boundaries when COMPUTE has already advanced its head past
the current OUTPUT pass).

---

## 5. Cycle trace for M=2 — proving the overlap (post-Task #87 Phase 1)

NL-DPE INT8 R=C=256 BUF=40: LCYC=52, CCYC=10, OCYC=52, T_steady=52, T_fill=116.

```
cycle:  0..51  52..103  53..62   104..114  64..115  116..167
        ─────  ───────  ──────   ────────  ───────  ────────
LOAD:   pass0  pass1                                          ← back-to-back, NO gap
COMPUTE:                pass0    pass1
OUTPUT:                                    pass0    pass1     ← chains across passes
                                                              (LCYC=OCYC=52, chain fires
                                                               when n_pending_output > 1)
```

Pass 0: LOAD 0..51, +1 handoff cycle 52, COMPUTE 53..62, +1 handoff cycle 63, OUTPUT 64..115.
Pass 1: LOAD must start *immediately* after pass 0's LOAD ends → cycles 52..103. +1 handoff at 104, COMPUTE 105..114, OUTPUT 116..167 (chained from pass 0).

**Total** T(2) = T_done_last - T_first_load + 1 = **168** = T_fill + 1·T_steady = 116 + 52. ✓

**Chain across passes**: at cycle 115 (pass 0 OUTPUT last strobe), the
chain check `n_pending_output > 1` reads `q_compute_head=2`
(NBA'd at cycle 114 from pass 1 COMPUTE done) and `q_output_head=0`,
yielding `n_pending_output=2 > 1` → chain fires, `output_busy` stays 1,
pass 1 OUTPUT begins byte 0 at cycle 116 with **no extra handoff** for
pass 1's COMPUTE→OUTPUT (the chain absorbs it). This is how subsequent
passes amortise into T_steady=52 cadence rather than each paying +2.

**Note for NL-DPE specifically**: LCYC = OCYC = 52, so OUTPUT chains continuously — `dpe_done` stays high from cycle 64 through cycle 167 with no gap.

**For Azure-Lily** (LCYC=256, OCYC=64), there's a gap between successive passes' OUTPUT phases:
- Pass 0 OUTPUT: cycles 268..331
- Pass 1 OUTPUT: cycles 524..587
- Gap of 192 cycles where `dpe_done = 0`

For AL the chain condition `n_pending_output > 1` fails (the gap is so
wide that pass 1 COMPUTE hasn't completed by pass 0 OUTPUT's last
strobe), so pass 1's COMPUTE→OUTPUT handoff again costs 1 cycle — but
that handoff happens *during* the LCYC>OCYC gap and is absorbed into
T_steady=256, so total per-pass cost remains T_steady. Net: the +2
overhead is paid ONCE in T_fill regardless of arch.

The TB's continuous-capture always-block (gated on `dpe_done`) naturally pauses through the gap.

---

## 6. TB walkthrough — `tb_dpe_vmm_msweep.v`

The MS-sweep TB has three jobs:

### 6a. Setup (lines 250-262)

```verilog
// Identity weights (R x C, identity on min(R,C))
for (i = 0; i < R; i = i + 1)
    for (k = 0; k < C; k = k + 1)
        dut.weights[i][k] = 8'h00;
for (i = 0; i < R; i = i + 1)
    if (i < C)
        dut.weights[i][i] = 8'h01;

nl_dpe_control = 2'b11;
```

### 6b. Drive M·LCYC strobes back-to-back (lines 265-277)

```verilog
for (pass_idx = 0; pass_idx < M_PASSES; pass_idx = pass_idx + 1) begin
    for (load_cycle_idx = 0; load_cycle_idx < LCYC; load_cycle_idx = load_cycle_idx + 1) begin
        data_in_full = 40'h0;
        for (b = 0; b < EPS; b = b + 1)
            if (load_cycle_idx * EPS + b < R)
                data_in_full[b*8 +: 8] = (pass_idx + 1) & 8'hFF;     // per-pass marker
        w_buf_en = 1'b1;
        @(posedge clk); #1;
        if (pass_idx == 0 && load_cycle_idx == 0) T_first_load = cycle_count;
    end
end
w_buf_en = 1'b0;
nl_dpe_control = 2'b00;
```

The test pattern is **per-pass constant**: pass 0 input bytes are `0x01`, pass 1 input bytes are `0x02`, etc. Each pass's expected output column is `(pass+1) & 0xFF` for cols < min(R,C).

**Why this pattern catches mis-routing**: every pass produces a *different* output value. If `vmm_queue`'s ring buffer is mis-indexed (e.g., q_output_head mis-advances), an earlier pass's value would appear at the wrong index and the per-pass byte compare would catch it.

### 6c. Continuous capture always-block (lines 202-214)

```verilog
always @(posedge clk) begin
    if (dpe_done && cap_strobe_idx < M_PASSES * OCYC) begin
        for (b = 0; b < EPS; b = b + 1)
            captured[cap_strobe_idx * EPS + b] = data_out_full[b*8 +: 8];
        cap_strobe_idx = cap_strobe_idx + 1;
        T_done_last = cycle_count;
    end
end
```

This is the **only** way to capture cleanly across the LCYC>OCYC gap (Azure-Lily case):
- `dpe_done` pulses when OUTPUT is busy.
- Capture pauses naturally during gaps.
- `cap_strobe_idx` resumes correctly when `dpe_done` re-asserts for the next pass.

### 6d. Verification

- **Functional**: per-pass per-byte compare `captured[m·OCYC·EPS + col]` vs `expected[m·OCYC·EPS + col]`.
- **Cycle**: `T_done_last - T_first_load + 1 == T_FILL + (M-1)·T_STEADY`.

---

## 7. Items to specifically eyeball during code review

| Concern | Where to check | Why it matters |
|---|---|---|
| Single-buffer correctness | `dpe_nldpe.v:166-189` | The blocking `=` writes on the last strobe MUST commit the VMM result to `vmm_queue` before the next NBA-write to `input_buffer` lands. **If you spot a `<=` in the VMM-fire block, that's a bug.** |
| NBA queue propagation | `dpe_nldpe.v` COMPUTE idle branch + OUTPUT idle branch | After Task #87 Phase 1, sub-FSM coordination is via NBA-advanced queue indices (`q_load_tail`, `q_compute_head`, `q_output_head`), NOT blocking same-cycle pulses. Each handoff cost 1 cycle of register propagation — faithful to real silicon. The chain condition is `n_pending_output > 1` only. |
| Queue ring overflow | `dpe_nldpe.v:84` | `QDEPTH=4` covers max 3 in flight. Could `q_load_tail` ever lap `q_output_head`? Only if 4 passes pile up — would need OUTPUT stalled while LOAD/COMPUTE keep firing, which can't happen in this controller. |
| Reset semantics | `dpe_nldpe.v:143-159` | Async reset, all sub-FSM state cleared synchronously. `MSB_SA_Ready=1`, `shift_add_done=1` matches legacy idle convention. |
| TB capture across gap | `tb_dpe_vmm_msweep.v:202-214` | For AL where LCYC>OCYC, dpe_done pulses with idle gaps. The capture's `if (dpe_done && cap_strobe_idx < ...)` MUST NOT increment during the gap; the `dpe_done` gate handles that. Empirically OK (AL M=4/M=8 PASS). |
| Backward compat for legacy single-pass TBs | `dpe_nldpe.v:123-125` | The `state` wire (combinational priority decode) lets old TBs (`tb_dpe_vmm.v`, `tb_dpe_acam.v`) still probe `dut.state == 3'd4` and see the right OUTPUT-phase boundary. Empirically: 36/36 single-pass cases still PASS. |

---

## 8. Design choices that may warrant discussion

1. **`COMPUTE_CYCLES` is now a module parameter** — was previously implicit (controller-driven via `nl_dpe_control` deassert in Model Y). New FSM needs an internal counter. Default 10 (INT8 + 3-stage pipeline - 1). Override at instantiation for different precision regimes.

2. **VMM math fires combinationally on the last LOAD strobe** — behavioral simplification. Real silicon spreads the math across PRECISION bit-slice ticks; behaviorally we collapse into one tick because sim's cycle accounting only cares about LOAD/COMPUTE/OUTPUT phase boundaries.

3. **ACAM modes 1 and 2 transform the VMM result in the same combinational fire** — `1 + x + x²/2` for mode 1 (exp), `x - 1` for mode 2 (log). Match `paper/methodology/dpe_pipeline_model.md` §3 ACAM specs.

4. **Queue depth 4 with 3-in-flight max** — gives 1 slot of headroom. Could be reduced to 3 if you want strict back-pressure. Currently no back-pressure is implemented (controller is expected to never over-issue, which is true for fc_top + sim's run_gemm tiling).

---

## 9. Verified behavior — empirical bottom line

After Task #82 commits:

| Test | Cases | Result |
|---|---|---|
| Single-pass smoke (legacy) | 36 | All PASS, **byte-identical cycle counts** to pre-refactor |
| M-sweep primitive (NL-DPE) | M ∈ {1,2,4,8}, R/C variants | All PASS, cycles = T_fill + (M-1)·T_steady exactly |
| M-sweep primitive (Azure-Lily) | M ∈ {1,2,4,8} | All PASS, cycles = T_fill + (M-1)·T_steady exactly |
| M-sweep DSP-MAC | K=64, M ∈ {1,2,4,8} | All PASS, cycles = T_fill + (M-1)·T_steady |
| Stage 1A FC smoke | 7 cases (V=1, H=1, M up to 8) | All PASS, **all 0% fidelity** vs sim's run_gemm |

Specifically, Stage 1A's M>1 cases that previously had +70-92% fidelity gap now match the simulator exactly:

| Case | Pre-refactor | Post-refactor | Sim |
|---|---|---|---|
| gemm_batched_NL M=4 | 459 | 270 | 270 |
| gemm_batch8_NL M=8 | 919 | 478 | 478 |
| gemm_batched_AL M=4 | 1323 | 1098 | 1098 |

---

## 10. Sub-FSM coordination — where compute happens and how latency is aligned

This section answers three questions that come up when explaining the model:

1. **How do the three sub-FSMs work together?** (§10b coordination, §10c queue indices)
2. **Where does the actual compute happen?** (§10a two-layer model)
3. **How is latency aligned with the simulator and methodology?** (§10e–§10f)

It ends with a cycle-by-cycle trace for M=2 (§10d) so the answer is concrete.

### 10a. Two layers — functional fire vs. timing burn

The DPE primitive separates *what* (the output values) from *when* (the cycle delay):

| Layer | Where it happens | Cycle cost |
|---|---|---|
| **Functional VMM math** — actually computes the output values | Combinational fire on the **last LOAD strobe** (LOAD sub-FSM, `dpe_nldpe.v:174-189`) | **0 cycles** — collapses into one posedge |
| **Bit-serial pipeline timing** — models the physical compute delay | COMPUTE sub-FSM counter (`dpe_nldpe.v:206-224`) | **per-arch CCYC** (see decomposition below) |

#### Per-arch bit-serial pipeline (Task #86)

The two architectures have **distinct** internal pipelines, even though they happen to give the same numerical CCYC at every precision under current parameters:

```
NL-DPE bit-serial pipeline (2 stages + ACAM)

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
        = 10 cycles  (INT8)
```

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
        = 10 cycles  (INT8)
```

**Note (ACAM always fires for NL-DPE):** even when ACAM_MODE=0 (identity passthrough), the read-out path consumes 1 cycle. Activation mode selects the LUT contents (identity, ReLU, exp, log), not whether ACAM fires.

#### Precision sweep — both arches give CCYC = PRECISION + 2

Under the current parameters (`PIPELINE_DEPTH_NL=2, ACAM_CYCLES_NL=1, PIPELINE_DEPTH_AL=3, ACAM_CYCLES_AL=0`), both arches produce `CCYC = PRECISION + 2`:

| Precision | NL CCYC                  | AL CCYC                | Match? |
|-----------|--------------------------|------------------------|--------|
| INT4      | 4 + (2-1) + 1 = 6        | 4 + (3-1) + 0 = 6      | yes    |
| INT8      | 8 + (2-1) + 1 = 10       | 8 + (3-1) + 0 = 10     | yes    |
| INT16     | 16 + (2-1) + 1 = 18      | 16 + (3-1) + 0 = 18    | yes    |

The match is **structural, not coincidental**: `(D_NL - 1) + ACAM_NL = (D_AL - 1) + ACAM_AL = 2`. If either side's pipeline depth or ACAM latency changes, the symmetry breaks.

Concrete INT4 sketch (4 bit-slices, both arches):

```
NL-DPE INT4 (PD=2, AC=1)                 Azure-Lily INT4 (PD=3, AC=0)
Cycle:    0    1    2    3    4    5      Cycle:    0    1    2    3    4    5
bit b0:  MAC──Acc                        bit b0:  MAC──ADC──SA
bit b1:       MAC──Acc                   bit b1:       MAC──ADC──SA
bit b2:            MAC──Acc              bit b2:            MAC──ADC──SA
bit b3:                 MAC──Acc         bit b3:                 MAC──ADC──SA
                              └ ACAM fires                                  └─ drain
CCYC_NL = 4 + 1 + 1 = 6 cycles          CCYC_AL = 4 + 2 + 0 = 6 cycles
```

Same numerical answer, distinct physical pipelines. The per-arch decomposition is encoded in the JSON config (`capabilities.pipeline_depth`, `capabilities.acam_cycles`) and flows through the generator → emitted Verilog header → TBs → Python drivers (see `fc_verification/FIDELITY_METHODOLOGY.md` §3.1).

#### Code layer — functional VMM math

The "compute" lives in TWO places, doing different jobs:

```verilog
// dpe_nldpe.v:175-181 — INSIDE the LOAD sub-FSM, on the last strobe
for (c = 0; c < NUM_COLS; c = c + 1) begin
    vmm_queue[q_load_tail][c] = 0;                              // BLOCKING (=)
    for (r = 0; r < KERNEL_WIDTH; r = r + 1)
        vmm_queue[q_load_tail][c] = vmm_queue[q_load_tail][c]
                                  + input_buffer[r] * weights[r][c];
end
```

This R×C multiply-accumulate is the **functional** compute. It runs in a single Verilog tick (one posedge) using blocking assignments — the result lands in `vmm_queue` before the next strobe.

Real silicon would spread this same math across CCYC cycles via a bit-serial pipeline (PRECISION bits per byte × per-arch pipeline stages + ACAM read-out for NL-DPE). Our behavioral model collapses that to one tick — but the COMPUTE sub-FSM still **burns** CCYC cycles afterwards to preserve the cycle accounting:

```verilog
// dpe_nldpe.v COMPUTE sub-FSM (post Task #87 Phase 1) — no math, just cycles
if (compute_busy) begin
    if (compute_cycle + 1 >= COMPUTE_CYCLES) begin
        compute_busy     <= 0;
        compute_cycle    <= 0;
        q_compute_head   <= (q_compute_head + 1) % QDEPTH;  // NBA: OUTPUT reads next cycle
        ...
    end else
        compute_cycle <= compute_cycle + 1;
end else begin
    if (n_pending_compute > 0) begin                         // NBA wake (1-cycle handoff from LOAD)
        compute_busy <= 1;
        compute_cycle <= 0;
    end
end
```

**Why the split**: external observers (the TB, downstream logic) only see (a) when `w_buf_en` strobes are accepted, (b) when `data_out` emerges. The math happening "instantly" inside the model is invisible. What matters is the elapsed cycles between LOAD-end and OUTPUT-start — and the COMPUTE counter ensures that's exactly CCYC.

This is the standard pattern for cycle-accurate behavioral models of analog hard blocks: **combinational functional model + cycle-accurate timing wrapper**. Two layers, both necessary, doing different jobs.

**FSM-level invariant:** the COMPUTE sub-FSM burns `COMPUTE_CYCLES` cycles regardless of how the per-arch decomposition derives it. The per-arch decomposition only changes the *default value* of `COMPUTE_CYCLES` in the emitted Verilog header; the FSM logic is unchanged.

### 10b. Sub-FSM coordination — NBA register propagation (RTL implementation overhead)

The three sub-FSMs are coordinated by **NBA-advanced queue indices**,
NOT by blocking same-cycle pulses. The producer advances its queue index
via NBA (`q_load_tail <= q_load_tail + 1`); the consumer reads the
updated `n_pending_*` count one cycle later via the combinational wire.

```
q_load_tail     ← LOAD NBAs ++ on its last strobe (after VMM math)
q_compute_head  ← COMPUTE NBAs ++ when its CCYC counter expires
q_output_head   ← OUTPUT NBAs ++ when its OCYC drain completes
```

Each handoff costs **exactly 1 cycle of register propagation** in the
RTL, matching real silicon. There are no blocking combinational
temporaries.

| Producer | Action | Consumer | Cycle to react |
|---|---|---|---|
| LOAD last strobe at cycle T | NBA `q_load_tail <= +1` | COMPUTE idle branch | reads `n_pending_compute>0` at T+1; NBA `compute_busy<=1`; busy at T+2 |
| COMPUTE done at cycle U | NBA `q_compute_head <= +1` | OUTPUT idle branch | reads `n_pending_output>0` at U+1; NBA `output_busy<=1`; busy at U+2 |

**Why NBA, not blocking pulses?** Real silicon FSM transitions cost 1
cycle per state register update — the FSM state cannot change in 0
cycles. The pre-Task-#87 model used blocking pulses (`fire_now`,
`compute_done_now`) to communicate within the same always-block tick,
which collapsed each handoff into 0 cycles. That was efficient for
simulation but unfaithful to real hardware.

With NBA propagation, **the RTL** observes `T_fill_rtl = LCYC + CCYC + OCYC + 2`.
For NL-DPE M=1: T_fill_rtl = 52 + 10 + 52 + 2 = **116**. **The sim**
emits the ideal `T_fill_ideal = 52 + 10 + 52 = 114` (Option A, post
Task #90). The +2 gap shows up as +1.75% fidelity.

**T_steady unchanged in both sim and RTL**: subsequent passes do not
pay an additional +2 per pass. The OUTPUT chain condition
(`n_pending_output > 1`) absorbs the per-pass COMPUTE→OUTPUT handoff
into the existing pipeline depth, yielding T_steady = max(LCYC, CCYC,
OCYC) regardless of M.

**Methodology framing (post Task #90)**: the +2 is **NOT in the §4 sim
formula** — it's a real silicon overhead of our particular RTL design
choice. The simulator emits the architectural cycle model
(`L + C + O`); the RTL discloses the implementation cost via faithful
NBA propagation; **fidelity is the measurement.** A different RTL
implementation (combinational handoffs, single-cycle state register
merge) would yield a different gap. The current observed gap of +2
is the empirical cost of our specific NBA-propagating FSM.

### 10c. Queue indices — pass-tag tracking across overlap

Three ring-buffer indices track which pass each sub-FSM "owns" right now:

| Index | Owned by | NBA-advances when | Consumer reads at |
|---|---|---|---|
| `q_load_tail` | LOAD | Last strobe of a pass (after VMM math fires) | COMPUTE's idle branch, 1 cycle later |
| `q_compute_head` | COMPUTE | CCYC counter expires | OUTPUT's idle branch, 1 cycle later; OUTPUT's chain check, 1 cycle later |
| `q_output_head` | OUTPUT | OCYC strobes drained for that pass | — |

Pass m's VMM result lives in `vmm_queue[q_load_tail at fire-time]`. COMPUTE later "consumes" it (no actual read of vmm_queue — the slot is just *marked busy* for CCYC cycles via q_compute_head). OUTPUT then drains the slot via `data_out <= vmm_queue[q_output_head][col]`.

The NBA-then-read-one-cycle-later pattern is what gives the FSM its
faithful per-handoff cost. Task #87 Phase 1 removed the blocking
combinational pulses (`fire_now`, `compute_done_now`) that previously
bypassed this cost.

**Invariant**: at any moment, `q_load_tail ≥ q_compute_head ≥ q_output_head` (modulo QDEPTH=4). Max passes in flight = 3 (LOAD just done, COMPUTE running, OUTPUT running). A 4th simultaneous pass would lap the queue — but the controller (sim's run_gemm cadence) never issues that many in flight.

### 10d. Cycle-by-cycle trace — M=2 NL-DPE INT8 R=C=256 BUF=40 (RTL behavior)

Setup: LCYC=52, CCYC=10, OCYC=52. T_fill_rtl=116, T_steady=52. Two passes back-to-back.

(Sim ideal T_fill_ideal=114, so RTL T(2)=168 vs sim T_sim(2)=166 →
fidelity = +1.20%.)

| Cycle | Event | LOAD | COMPUTE | OUTPUT | (q_l, q_c, q_o) |
|---:|---|---|---|---|:-:|
| 0 | first w_buf_en strobe pass 0 | start | idle | idle | (0,0,0) |
| 1..50 | LOAD pass 0 strobes 1..50 | strobing | idle | idle | (0,0,0) |
| **51** | **last LOAD strobe pass 0**: VMM math fires combinationally → `vmm_queue[0]` filled (BLOCKING); `q_load_tail` NBA'd 0→1 | done; first strobe of pass 1 NBA'd for cycle 52 | n_pending_compute=0 (q_load_tail NBA pending); stays idle | idle | (1,0,0)→ |
| **52** | first w_buf_en strobe pass 1 (back-to-back, no gap); **LOAD→COMPUTE handoff cycle** | strobing pass 1 | q_load_tail=1 (NBA'd); n_pending_compute=1; NBA `compute_busy<=1` | idle | (1,0,0) |
| **53** | LOAD pass 1 strobe 1; COMPUTE pass 0 first counting cycle | strobing pass 1 | compute_busy=1; compute_cycle=0→1 NBA | idle | (1,0,0) |
| 54..61 | LOAD pass 1 strobes 2..9; COMPUTE pass 0 counts 2..8 | strobing pass 1 | counting | idle | (1,0,0) |
| **62** | LOAD pass 1 strobe 10; COMPUTE pass 0 cycle 9 of 10: done; `q_compute_head` NBA'd 0→1 | strobing pass 1 | done | n_pending_output=0 (q_compute_head NBA pending); stays idle | (1,1,0)→ |
| **63** | LOAD pass 1 strobe 11; **COMPUTE→OUTPUT handoff cycle** | strobing pass 1 | idle | q_compute_head=1 (NBA'd); n_pending_output=1; NBA `output_busy<=1` | (1,1,0) |
| **64** | OUTPUT pass 0 first byte: `data_out` NBA'd from `vmm_queue[0][0..EPS-1]`; LOAD pass 1 strobe 12 | strobing pass 1 | idle | drain 0/52 | (1,1,0) |
| 65..103 | LOAD pass 1 strobes 13..51; OUTPUT pass 0 strobes 1..39 (continuous overlap) | strobing pass 1 | idle | drain 1..39 | (1,1,0) |
| **103** | **last LOAD strobe pass 1**: VMM math fires → `vmm_queue[1]`; `q_load_tail` NBA'd 1→2 | done | n_pending_compute=0 (NBA pending) | drain 39/52 | (2,1,0)→ |
| **104** | **LOAD→COMPUTE handoff cycle for pass 1** | idle | q_load_tail=2 (NBA'd); n_pending_compute=1; NBA `compute_busy<=1` | drain 40/52 | (2,1,0) |
| 105..113 | COMPUTE pass 1 counts 0..8 (9 cycles); OUTPUT pass 0 strobes 41..49 | idle | counting | draining | (2,1,0) |
| **114** | COMPUTE pass 1 cycle 9 of 10: done; `q_compute_head` NBA'd 1→2 | idle | done | OUTPUT pass 0 strobe 50/52 | (2,2,0)→ |
| **115** | OUTPUT pass 0 LAST strobe (col 51); chain check: `n_pending_output > 1` reads (q_compute_head=2, q_output_head=0) → 2>1=TRUE → chain: `output_busy` stays 1; `q_output_head` NBA'd 0→1, `output_col_idx` NBA'd to 0 | idle | idle | drain pass 0 last; queue advances to pass 1 | (2,2,1)→ |
| **116** | OUTPUT pass 1 first byte: `data_out` NBA'd from `vmm_queue[1][0..EPS-1]` | idle | idle | drain 0/52 (pass 1) | (2,2,1) |
| 117..166 | OUTPUT pass 1 strobes 1..50 | idle | idle | draining | (2,2,1) |
| **167** | OUTPUT pass 1 LAST strobe; `output_busy` NBA'd to 0; `q_output_head` 1→2 | idle | idle | drain 51/52 | (2,2,2)→ |
| 168 | DPE quiesced (`dpe_done`=0 next posedge) | idle | idle | idle | (2,2,2) |

**Total**: T_done_last − T_first_load + 1 = 167 − 0 + 1 = **168 cycles** ✓ matches `T_fill + (M-1)·T_steady = 116 + 52 = 168`.

Key observations:
- LOAD pass 0 (cycles 0-51) → LOAD pass 1 (cycles 52-103) — **back-to-back, no idle gap**.
- COMPUTE pass 0 (53-62) is shifted 1 cycle later vs pre-Task-#87 (52-61) — LOAD→COMPUTE handoff at cycle 52.
- OUTPUT pass 0 (64-115) is shifted 2 cycles later vs pre-Task-#87 (62-113) — both handoffs paid.
- OUTPUT pass 1 starts at 116 — chained from pass 0's last strobe at cycle 115 via `n_pending_output > 1` (no additional +1 cycle for pass 1's COMPUTE→OUTPUT — the chain absorbs it).
- COMPUTE pass 0 (53-62) and COMPUTE pass 1 (105-114) are **disjoint** — compute does not overlap across passes (per §4 spec).
- Pass 0→1 cadence at OUTPUT is exactly T_steady = 52: pass 0 byte 0 NBA at cycle 64, pass 1 byte 0 NBA at cycle 116; delta = 52. ✓

### 10e. Latency alignment — sim ideal vs RTL faithful (Option A, post Task #90)

The simulator (`azurelily/IMC/imc_core/imc_core.py::_pipeline_total_cycles`) emits the **ideal** cycle count:
```python
T_fill_ideal = L + C + O          # no implementation overhead
T_steady     = max(L, C, O)
T_sim(M)     = T_fill_ideal + (M - 1) * T_steady
```

For NL-DPE INT8 R=C=256 BUF=40 M=2: `T_sim(2) = (52+10+52) + 1×52 = 166`.

The RTL behavior model exhibits the **faithful** cycle count (+2 FSM
register propagation):
```
T_fill_rtl = L + C + O + 2          # NBA register propagation cost
T_rtl(M)   = T_fill_rtl + (M - 1) * T_steady
```

For NL-DPE INT8 R=C=256 BUF=40 M=2: `T_rtl(2) = (52+10+52+2) + 1×52 = 168`. Matches RTL trace above ✓.

**Fidelity (reported, not gated)**: `(168 − 166) / 166 = +1.20%`.

The §4 methodology claim:
> *Drain of pass k can overlap with load of pass k+1. Compute is sandwiched between load and output and cannot overlap across passes. Steady-state interval = max(L, C, O). The §4 formula models the architecture; the RTL pays the implementation cost.*

Verified against the trace:
- ✓ LOAD pass 1 (52-103) overlaps with OUTPUT pass 0 (64-115) — drain-load overlap.
- ✓ COMPUTE passes are disjoint in time (53-62 vs 105-114) — compute does not overlap across passes.
- ✓ Steady-state interval = T_steady = 52 cycles. Pass m+1 OUTPUT byte 0 starts T_steady cycles after pass m OUTPUT byte 0 (64 → 116 = +52 ✓).
- ✓ T_fill_rtl = 52 + 10 + 52 + 2 = 116 cycles for pass 0 alone — LOAD 0..51, +1 LOAD→COMPUTE handoff at 52, COMPUTE 53..62, +1 COMPUTE→OUTPUT handoff at 63, OUTPUT 64..115. The last byte NBA at cycle 115 → captured at cycle 116 → T_done_last=115 → total = 116. ✓
- ✓ T_fill_ideal = 52 + 10 + 52 = 114 cycles — what the simulator predicts (no FSM handoffs modeled). The RTL pays +2; the simulator does not; +1.75% fidelity at M=1.

### 10f. Why the two-layer model is correct

The behavioral primitive faithfully implements the §4 architectural claim:
- **One physical input buffer** (`input_buffer[KERNEL_WIDTH]`) — overwritten each pass; freed for pass m+1 the cycle after pass m's VMM math fires (math result is parked in vmm_queue).
- **One physical compute unit** (the analog crossbar) — represented by COMPUTE sub-FSM, *no overlap across passes* (one COMPUTE busy at a time).
- **One physical output stage** (the ADC/shift-add accumulator + drain) — represented by OUTPUT sub-FSM. The `vmm_queue` ring buffer is a **modeling artifact** for pass-tagging — physical silicon would have one accumulator that gets read out incrementally; behaviorally we park completed accumulators in queue slots so OUTPUT can drain pass m while COMPUTE handles pass m+1.

The combinational VMM fire is a **simulation acceleration trick**, not a hardware claim:
- Real silicon's bit-serial multiply takes CCYC cycles to traverse the bit-serial pipeline.
- Behavioral model collapses to one tick (faster simulation), but the COMPUTE counter "burns" exactly CCYC cycles afterward.
- External observers (TB, downstream FSM) see `data_out` emerge at the correct cycle — they cannot distinguish this from real silicon.

For the advisor: this is the standard cycle-accurate behavioral modeling pattern for analog hard blocks — combinational functional layer + cycle-accurate timing layer, separated cleanly so each can be reasoned about and verified independently. The two-layer separation maps directly to lines 175-189 (functional fire) and lines 207-216 (timing burn) in `dpe_nldpe.v`.

The cycle-level contract `T(M) = T_fill + (M-1)·T_steady` is satisfied **exactly**, byte-identical to the simulator's analytical prediction, for all (M, R, C, BUF, PRECISION, PIPELINE_DEPTH) combinations swept by `run_dpe_smoke.py` (52/52 PASS).

---

## 11. Paired TB ↔ DUT cycle-by-cycle walkthrough (M=7 NL-DPE)

This section pairs `tb_dpe_vmm_msweep.v` Verilog code with DUT internal state, the ring buffer contents, and the TB's capture buffer — cycle by cycle — for a concrete M=7 NL-DPE INT8 R=C=256 BUF=40 run.

**Why M=7 (odd)**: with QDEPTH=4 and M=7, the ring wraps once and the run terminates mid-ring (slot 2, not slot 3). This exercises:
- Initial ring fill (slots 0→1→2→3 for passes 0..3)
- Ring wrap (slots 0→1→2 reused for passes 4..6)
- Mid-ring termination (last pass at slot 2; slot 3 untouched after pass 3)

### 11a. Test target

```
Workload:   M_PASSES = 7
Arch:       NL-DPE INT8
Geometry:   R = 256, C = 256, BUF = 40 → EPS = 5 bytes/strobe
Derived:    LCYC = 52, CCYC = 10, OCYC = 52
            T_fill_rtl    = 52 + 10 + 52 + 2 = 116    (RTL faithful FSM)
            T_fill_ideal  = 52 + 10 + 52     = 114    (sim ideal, Option A)
            T_steady = max(52, 10, 52) = 52
Expected:   T_rtl = T_fill_rtl + (M-1)·T_steady = 116 + 6·52 = 428 cycles
            T_sim = T_fill_ideal + (M-1)·T_steady = 114 + 6·52 = 426 cycles
            Fidelity = (428 - 426) / 426 = +0.47%
```

### 11b. Test pattern (the "hardware" data going in and out)

**Weights**: identity matrix on min(R,C). TB hierarchical-forces these at setup:
```verilog
// tb_dpe_vmm_msweep.v:254-260
for (i = 0; i < R; i = i + 1)
    for (k = 0; k < C; k = k + 1)
        dut.weights[i][k] = 8'h00;
for (i = 0; i < R; i = i + 1)
    if (i < C)
        dut.weights[i][i] = 8'h01;
```

After setup: `dut.weights[c][c] = 0x01` for c ∈ [0, 256); all other entries = 0x00.

**Inputs per pass** (per-pass distinguishable marker — this is the load-bearing trick):
```verilog
// tb_dpe_vmm_msweep.v:267-270
for (b = 0; b < EPS; b = b + 1)
    if (load_strobe_idx * EPS + b < R)
        data_in_full[b*8 +: 8] = (pass_idx + 1) & 8'hFF;
```

For each pass `m`, every input byte = `(m+1) & 0xFF`:

| Pass m | Input bytes | Expected MAC[c] | Expected data_out byte |
|---|---|---|---|
| 0 | all `0x01` | sum_r 1·weights[r][c] = 1 (identity) | `0x01` |
| 1 | all `0x02` | 2 | `0x02` |
| 2 | all `0x03` | 3 | `0x03` |
| 3 | all `0x04` | 4 | `0x04` |
| 4 | all `0x05` | 5 | `0x05` |
| 5 | all `0x06` | 6 | `0x06` |
| 6 | all `0x07` | 7 | `0x07` |

**Why this pattern catches ring buffer bugs**: each pass's output is *different*. If `q_output_head` mis-advances by even one slot, pass 1's `0x02` bytes appear where pass 0's `0x01` should be, and the per-pass byte compare catches it.

### 11c. Cycle-by-cycle table

Cycles are **relative** to the first LOAD strobe (cycle 0). The TB's absolute `cycle_count` register has a setup offset (~3-4 cycles of reset + weight-load); subtract the offset to get the values below. All NBA/blocking semantics are explicit so you can match against the Verilog.

For compactness only **transition cycles** are shown; phases between transitions just continue counting strobes.

**Cycle shift note**: the RTL trace below shows the faithful NBA-FSM
behavior (total = 428 cycles). The sim's ideal model predicts 426
cycles (no FSM handoffs modeled). The +2 difference surfaces as
+0.47% fidelity. The trace below is the RTL cycle-by-cycle behavior;
the sim does not produce a cycle-by-cycle trace.

Cycle shifts in RTL behavior:
- LOAD cycles **unchanged** (TB drives strobes back-to-back, cadence T_steady=52).
- COMPUTE cycles shifted **+1** (LOAD→COMPUTE handoff — RTL implementation overhead).
- OUTPUT cycles shifted **+2** (LOAD→COMPUTE + COMPUTE→OUTPUT handoffs — RTL implementation overhead).
- TB capture cycles shifted **+2** (follow OUTPUT).

These shifts are RTL-side only; the ideal sim model predicts a
non-shifted layout. The simulator does not emit a cycle-by-cycle trace
— it emits the aggregate ideal cycle count 426.

| Cycle | TB code (tb_dpe_vmm_msweep.v) | DUT input | DUT state transitions | `(q_l, q_c, q_o)` | Captured |
|---:|---|---|---|:-:|---|
| 0 | drive `data_in = {5{0x01}}`, `w_buf_en=1`; first iteration `(pass_idx=0, load_strobe_idx=0)`; @posedge; #1; `T_first_load = cycle_count` | strobe 0 of pass 0 | LOAD: NBA `input_buffer[0..4]<=0x01`, `load_cycle_idx<=1` | (0,0,0) | 0 captured |
| 1..50 | loop continues: strobe i of pass 0 | strobes 1..50 of pass 0 | LOAD NBA writes byte 5..254 | (0,0,0) | 0 captured |
| **51** | `load_strobe_idx==51==LCYC-1` last strobe pass 0; same TB drive | strobe 51 of pass 0 (last) | LOAD: **BLOCKING** `input_buffer[255]=0x01`; **BLOCKING** VMM math runs: `vmm_queue[0][c] = sum_r input_buffer[r]·weights[r][c] = 1` for c<256; NBA `q_load_tail<=1`. COMPUTE idle reads n_pending_compute=0 (q_load_tail NBA pending) — stays idle this cycle. | (0→1, 0, 0) | 0 captured |
| **52** | pass_idx=1 starts: drive `data_in={5{0x02}}` (LOAD→COMPUTE handoff cycle) | strobe 0 of pass 1 | LOAD: NBA `input_buffer[0..4]<=0x02`. COMPUTE idle: q_load_tail=1 (NBA'd); n_pending_compute=1>0; NBA `compute_busy<=1, compute_cycle<=0` | (1, 0, 0) | 0 |
| **53** | drive strobe 1 of pass 1 | strobe 1 of pass 1 | LOAD writes; COMPUTE pass 0 first counting cycle: compute_busy=1, compute_cycle=0→1 NBA | (1, 0, 0) | 0 |
| 54..61 | loop strobes pass 1 | strobes 2..9 of pass 1 | LOAD writes; COMPUTE counts 1..8 | (1, 0, 0) | 0 |
| **62** | drive strobe 10 of pass 1 | strobe 10 of pass 1 | LOAD: NBA writes. COMPUTE: `compute_cycle+1==10==CCYC` → NBA `compute_busy<=0, q_compute_head<=1, MSB_SA_Ready<=1`. OUTPUT idle reads n_pending_output=0 (q_compute_head NBA pending) — stays idle. dpe_done reads OLD `output_busy=0` → NBA `dpe_done<=0` | (1, 0→1, 0) | 0 |
| **63** | drive strobe 11 of pass 1 (COMPUTE→OUTPUT handoff cycle) | strobe 11 of pass 1 | LOAD: NBA writes. OUTPUT idle: q_compute_head=1 (NBA'd); n_pending_output=1>0; NBA `output_busy<=1, output_col_idx<=0`. dpe_done reads OLD output_busy=0 → NBA dpe_done<=0 | (1, 1, 0) | 0 |
| **64** | drive strobe 12 of pass 1 | strobe 12 of pass 1 | LOAD NBA writes. OUTPUT: `output_busy=1`; NBA `data_out<={vmm_queue[0][4],...,vmm_queue[0][0]}={5{0x01}}`; NBA `output_col_idx<=1`. dpe_done reads `output_busy=1` → NBA `dpe_done<=1` | (1, 1, 0) | 0 (TB's always-block reads dpe_done=0 from cycle 63's NBA) |
| **65** | drive strobe 13 of pass 1 | strobe 13 of pass 1 | LOAD; OUTPUT NBA `data_out<=vmm_queue[0][5..9]`. **TB-side**: TB capture always-block samples `dpe_done=1` (NBA'd at cycle 64) — captures `data_out` (NBA'd at cycle 64 = `{5{0x01}}`); `cap_strobe_idx 0→1`; `T_done_last=cycle_count` | (1, 1, 0) | `captured[0..4]={5{0x01}}` ✓ pass 0 byte 0 |
| 66..103 | loop continues pass 1 LOAD strobes | strobes 14..51 of pass 1 | LOAD: NBA writes; pass 1 fires at cycle **103** → `vmm_queue[1][c]=2`; NBA `q_load_tail<=2`. TB-side captures strobes 1..38 of pass 0 OUTPUT | (1→2, 1, 0) at cycle 103 | `captured` accumulates pass 0 bytes |
| **104** | pass_idx=2 first strobe: drive `data_in={5{0x03}}` (LOAD→COMPUTE pass 1 handoff) | strobe 0 of pass 2 | LOAD: NBA `input_buffer[0..4]<=0x03`. COMPUTE idle: q_load_tail=2 (NBA'd); n_pending_compute=1>0; NBA `compute_busy<=1, compute_cycle<=0` | (2, 1, 0) | accumulating |
| 105..113 | loop strobes pass 2 | strobes 1..9 of pass 2 | LOAD writes; COMPUTE pass 1 counts 0..8 | (2, 1, 0) | pass 0 OUTPUT continues |
| **114** | drive strobe 10 of pass 2 | strobe 10 of pass 2 | COMPUTE pass 1: `compute_cycle+1==10` → done; NBA `q_compute_head<=2, compute_busy<=0`. OUTPUT pass 0 NBA byte 50 | (2, 1→2, 0) | captured ~49 bytes pass 0 |
| **115** | drive strobe 11 of pass 2 | strobe 11 of pass 2 | OUTPUT pass 0: `output_col_idx==51==OCYC-1` → LAST strobe pass 0; NBA `q_output_head<=1, output_col_idx<=0`; chain check `n_pending_output>1` reads (q_compute_head=2, q_output_head=0) → 2>1=TRUE → **chain**: `output_busy` stays 1 | (2, 2, 0→1) | captured ~50 bytes pass 0 |
| **116** | drive strobe 12 of pass 2 | strobe 12 of pass 2 | OUTPUT: now draining `vmm_queue[q_output_head=1]`; NBA `data_out<=vmm_queue[1][0..4]={5{0x02}}` | (2, 2, 1) | captured ~51 bytes pass 0 (last) |
| **117** | drive strobe 13 of pass 2 | strobe 13 of pass 2 | OUTPUT NBA `data_out<=vmm_queue[1][5..9]`. **TB capture**: reads NBA'd `data_out={5{0x02}}` from cycle 116 — captures into `captured[OCYC*EPS..]` slot for pass 1 byte 0 ✓ | (2, 2, 1) | `captured[260..264]={5{0x02}}` ✓ pass 1 byte 0 |
| 118..155 | loop pass 2 LOAD continues | pass 2 strobes 14..51 | pass 2 fires at cycle **155** → vmm_queue[2][c]=3; q_load_tail<=3 | (2→3, 2, 1) at 155 | accumulating pass 1 captures |
| **156** | pass_idx=3 first strobe: `data_in={5{0x04}}` (LOAD→COMPUTE pass 2 handoff) | strobe 0 of pass 3 | LOAD writes input_buffer. COMPUTE idle wakes for pass 2: NBA `compute_busy<=1` | (3, 2, 1) | |
| **166** | strobe 10 of pass 3 | strobe 10 of pass 3 | COMPUTE pass 2 done at cycle 166 (started cycle 156); NBA q_compute_head<=3 | (3, 2→3, 1) | |
| **167** | strobe 11 of pass 3 | strobe 11 of pass 3 | OUTPUT pass 1 LAST strobe (col 51); chain check 3>1=TRUE → chain to pass 2 OUTPUT; q_output_head<=2 | (3, 3, 1→2) | pass 1 fully captured |
| **207** | last LOAD strobe of pass 3 | strobe 51 of pass 3 | LOAD fires pass 3 → vmm_queue[3][c]=4; **q_load_tail wraps NBA `3→0`** | (3→0, 3, 2) | accumulating pass 2 captures |
| **208** | pass_idx=4 first strobe: `data_in={5{0x05}}` (LOAD→COMPUTE pass 3 handoff) | strobe 0 of pass 4 | LOAD NBA `input_buffer[0..4]<=0x05`. COMPUTE idle wakes for pass 3: NBA compute_busy<=1 | (0, 3, 2) | |
| **218** | strobe 10 of pass 4 | strobe 10 of pass 4 | COMPUTE pass 3 done; NBA q_compute_head<=0 (3→0 wrap) | (0, 3→0, 2) | |
| **219** | strobe 11 of pass 4 | strobe 11 of pass 4 | OUTPUT pass 2 LAST strobe; chain to pass 3 OUTPUT (n_pending_output=2: q_c=0 mod 4 minus q_o=2 = QDEPTH+0-2=2); q_output_head<=3 | (0, 0, 2→3) | pass 2 fully captured |
| 220..258 | pass 4 LOAD continues | strobes 12..50 of pass 4 | LOAD writes input_buffer overwriting pass 3 (safe — vmm_queue[3] preserved); OUTPUT drains pass 3 from vmm_queue[3] | (0, 0, 3) | accumulating pass 3 |
| **259** | **last LOAD strobe of pass 4** | strobe 51 of pass 4 | LOAD fires pass 4 → **vmm_queue[0][c]=5** (REUSED slot 0!); q_load_tail NBA `0→1`. Slot 0 was freed by pass 0 OUTPUT ending at cycle 115 (144 cycles ago — plenty of margin) | (0→1, 0, 3) | |
| **260** | pass_idx=5 first strobe: `data_in={5{0x06}}` (LOAD→COMPUTE pass 4 handoff) | strobe 0 of pass 5 | LOAD writes. COMPUTE idle wakes for pass 4 (vmm_queue[0] = slot for compute); NBA compute_busy<=1 | (1, 0, 3) | |
| **270** | strobe 10 of pass 5 | strobe 10 of pass 5 | COMPUTE pass 4 done; NBA q_compute_head<=1 (0→1) | (1, 0→1, 3) | |
| **271** | strobe 11 of pass 5 | strobe 11 of pass 5 | OUTPUT pass 3 LAST; chain to pass 4 OUTPUT; q_output_head NBA `3→0` (wraps) | (1, 1, 3→0) | pass 3 fully captured |
| **311** | last LOAD strobe of pass 5 | strobe 51 of pass 5 | LOAD fires pass 5 → vmm_queue[1][c]=6 (slot 1 reused); q_load_tail NBA `1→2` | (1→2, 1, 0) | |
| **312** | pass_idx=6 first strobe: `data_in={5{0x07}}` (LOAD→COMPUTE pass 5 handoff) | strobe 0 of pass 6 | LOAD writes. COMPUTE idle wakes for pass 5 | (2, 1, 0) | |
| **322** | strobe 10 of pass 6 | strobe 10 of pass 6 | COMPUTE pass 5 done; NBA q_compute_head<=2 | (2, 1→2, 0) | |
| **323** | strobe 11 of pass 6 | strobe 11 of pass 6 | OUTPUT pass 4 LAST; chain to pass 5 OUTPUT; q_output_head NBA `0→1` | (2, 2, 0→1) | pass 4 fully captured |
| **363** | **last LOAD strobe of pass 6 (LAST pass for M=7)** | strobe 51 of pass 6 | LOAD fires pass 6 → vmm_queue[2][c]=7 (slot 2 reused); q_load_tail NBA `2→3`. TB exits inner loop. **TB drops `w_buf_en=0`, `nl_dpe_control=2'b00`** | (2→3, 2, 1) | |
| **364** | (LOAD→COMPUTE pass 6 handoff cycle); w_buf_en=0 | (no input strobes) | COMPUTE idle wakes for pass 6: NBA compute_busy<=1 | (3, 2, 1) | OUTPUT pass 5 draining |
| 365..373 | drain wait | (idle) | COMPUTE pass 6 counts cycles 0..8 | (3, 2, 1) | OUTPUT pass 5 draining |
| **374** | wait_drain loop polling | (idle) | COMPUTE pass 6 done; NBA q_compute_head<=3 | (3, 2→3, 1) | |
| **375** | wait_drain | (idle) | OUTPUT pass 5 LAST strobe (col 51); chain check n_pending_output=2 → chain to pass 6 OUTPUT; q_output_head<=2 | (3, 3, 1→2) | pass 5 fully captured |
| 376..426 | drain wait continues | (idle) | OUTPUT pass 6 strobes 0..50 (TB captures cycles 377..427) | (3, 3, 2) | accumulating pass 6 |
| **427** | wait_drain reaches cap_strobe_idx == M·OCYC = 364 | (idle) | OUTPUT pass 6 LAST strobe (col 51); output_busy NBA<=0 (no chain — n_pending_output=1); q_output_head NBA `2→3` | (3, 3, 2→3) | pass 6 byte 51 captured at cycle 428 |
| **428** | exit wait_drain; final compare loop | (idle) | dpe_done NBA<=0 (output_busy=0). `T_done_last = 427` (last capture posedge) | (3, 3, 3) | All M·OCYC·EPS = 7·52·5 = 1820 bytes captured ✓ |

**Verification at exit** (`tb_dpe_vmm_msweep.v:303-315`):
```verilog
for (mi = 0; mi < M_PASSES; mi = mi + 1)
    for (i = 0; i < OCYC * EPS; i = i + 1)
        if (captured[mi * OCYC * EPS + i] !== expected[mi * OCYC * EPS + i])
            error_count = error_count + 1;
```

For each pass `m`, expected bytes are `(m+1) & 0xFF` for cols < min(R,C)=256, 0 otherwise. The per-pass distinguishability is what catches ring buffer mis-routing.

**Cycle check** (`tb_dpe_vmm_msweep.v:320-321`):
```verilog
(T_done_last - T_first_load + 1) == EXPECTED_CYCLES
```

`T_first_load = 0` (cycle of first LOAD strobe). `T_done_last = 427` (cycle of last data_out NBA capture). Total = 428 = T_fill_rtl + 6·T_steady = 116 + 6·52 ✓.

Sim's ideal prediction: T_sim = T_fill_ideal + 6·T_steady = 114 + 6·52 = 426. Fidelity = (428-426)/426 = +0.47%.

### 11d. Ring buffer slot reuse — visual at peak wrap (cycle 259)

```
At cycle 259 (pass 4 LOAD fires; slot 0 reused):

vmm_queue ring (depth 4):
  ┌─────────┬─────────┬─────────┬─────────┐
  │ slot 0  │ slot 1  │ slot 2  │ slot 3  │
  │ pass 4  │ pass 1  │ pass 2  │ pass 3  │
  │  =0x05  │  =0x02  │  =0x03  │  =0x04  │
  │ JUST    │ drained │ drained │ being   │
  │ FIRED   │  (old)  │  (old)  │ drained │
  └─────────┴─────────┴─────────┴─────────┘
        ↑                            ↑
   q_load_tail                  q_output_head
   advanced 0→1                 currently 3
                              (about to advance 3→0
                               at cycle 271 when
                               pass 3 OUTPUT ends)

Slot 0 was last drained at cycle 115 (pass 0 OUTPUT ended).
Now reused at cycle 259 (pass 4 LOAD fires).
144 cycles of dead time — plenty of margin; no possibility of corruption.

q_compute_head = 0: pass 3's COMPUTE is in progress
   (cycles 208..218; advanced 3→0 at cycle 218 when done — note
    the +1 LOAD→COMPUTE handoff means COMPUTE pass m's done-cycle
    is 11 cycles past pass m's LOAD last strobe, not 10).

At cycle 259 q_compute_head = 0 (advanced after pass 3 compute done
at cycle 218). Pass 4 fires at 259 → slot 0 holds pass 4's accumulator
→ COMPUTE wakes cycle 260 (LOAD→COMPUTE handoff) and busy at 261
starting pass 4.
```

The ring's compactness: with M=7 and depth=4, slot 0 holds:
- Pass 0's accumulator: cycles 51..115 (64 cycles)
- (free): cycles 116..258 (143 cycles)
- Pass 4's accumulator: cycles 259..323 (64 cycles)
- (free, never used again)

Each slot is reused at most twice (passes m and m+4) in this M=7 run.

### 11e. NBA-timing aside — why the TB sees `dpe_done` one cycle late

The DPE NBA's `dpe_done <= 1` based on the **prior cycle's** value of `output_busy`. With Task #87 Phase 1 NBA register propagation:

```
Cycle U:    COMPUTE done; NBA: q_compute_head++. OUTPUT idle reads
            n_pending_output=0 (NBA pending); stays idle.
Cycle U+1:  q_compute_head committed. n_pending_output=1>0; NBA:
            output_busy<=1. dpe_done uses OLD output_busy=0 → dpe_done<=0.
Cycle U+2:  output_busy=1; NBA: data_out<=vmm_queue[...]; output_col_idx<=1.
            dpe_done updates using NEW output_busy=1 → dpe_done<=1.
Cycle U+3:  data_out = NBA'd value from U+2 (first OUTPUT byte);
            dpe_done = 1 (NBA'd from U+2).
            TB capture always-block samples (dpe_done=1) → captures
            (data_out = first OUTPUT byte). cap_strobe_idx 0→1.
```

So the TB captures the first OUTPUT byte at cycle U+3, where U is the cycle of COMPUTE done (the cycle NBA q_compute_head++).

For pass 0: COMPUTE done at cycle 62 → first capture at cycle 65. For pass 6 LAST byte: OUTPUT'd at cycle 427 → captured at cycle 428 (the cycle of `T_done_last + 1`).

**Why this gives the right formula**: both `T_first_load` and `T_done_last` are sampled at cycles offset by the same NBA delay (one cycle after the DUT's internal NBA happens). The **difference** `T_done_last - T_first_load + 1` cancels out the offset and yields exactly `T_fill + (M-1)·T_steady` = `LCYC + CCYC + OCYC + 2 + (M-1)·T_steady`. The cycle math works out cleanly only because the TB's sampling convention is consistent at both endpoints.

If you tried to instrument with absolute cycle numbers (e.g., "TB asserts strobe at cycle X, DUT emits byte at cycle Y"), the +1/+2 NBA offsets would clutter the analysis. Using `T_done_last - T_first_load + 1` is the right *relative* measurement.

### 11f. Bug-catching corner cases

Each design choice in the ring buffer is load-bearing — here's what fails if you break each one:

| Hypothetical bug | Concrete failure in this M=7 trace | Caught by |
|---|---|---|
| VMM math uses NBA writes on last strobe (instead of blocking) | At cycle 51, `input_buffer[255]<=0x01` is NBA — math reads OLD value (stale pass-from-previous-run's data, or 0x00 on fresh reset). `vmm_queue[0][c]` gets wrong value. Pass 0 expected `0x01`, actual `0x00`. | Per-pass byte compare: `captured[0..4] != {5{0x01}}` → 5+ errors per pass × 7 passes |
| `q_load_tail` advance is BLOCKING (not NBA) | At cycle 51, q_load_tail becomes 1 immediately within the always-block. COMPUTE's idle branch in the same tick sees n_pending_compute=1 and NBAs compute_busy<=1 — bypassing the +1 LOAD→COMPUTE handoff (collapses back to pre-Task-#87 behavior). T(M=7) becomes 426 not 428. | Cycle compare: 426 != 428 → FAIL |
| Chain condition uses `n_pending_output > 0` (instead of `> 1`) | Pass m+1's OUTPUT would re-trigger immediately the cycle pass m OUTPUT ends, even when COMPUTE pass m+1 hasn't completed. For AL (LCYC>OCYC) this causes pass m+1's bytes to be NBA'd from a stale `vmm_queue[q_output_head]`. | Per-pass byte compare: pass m+1's bytes corrupted |
| Missing chain (drop the `if n_pending_output > 1` entirely) | Pass 0→1 transition inserts +1 idle cycle at the OUTPUT boundary. T(M=7) becomes 428 + 6 extra (one per inter-pass transition) = 434. | Cycle compare FAIL |
| `q_output_head` advances out of order (e.g., 0→2 skipping 1) | At cycle 115, slot 1 is "skipped" — pass 1's bytes never drain. cap_strobe_idx never reaches M·OCYC → `wait_drain` guard expires → TB reports "only captured X/364 strobes" error. | Drain-wait guard: explicit error message |
| QDEPTH = 2 (too small) | At cycle 207 (pass 3 fires), q_load_tail wraps from 1→0 prematurely; collides with pass 1 still being drained (q_output_head=1). vmm_queue[0]'s pass 0 bytes (still being drained) get overwritten with pass 3's bytes. Pass 0 trailing bytes corrupt. | Per-pass byte compare: pass 0's last ~10 bytes are 0x04 instead of 0x01 → errors |

The M=7 test pattern with per-pass distinguishable markers + the cycle-count check is **sufficient** to detect every plausible ring buffer or sub-FSM coordination bug we've identified. The combination of *per-pass byte compare* (functional correctness) and *cycle-formula check* (timing correctness) provides the verification surface.

### 11g. Summary

This walkthrough is the load-bearing piece of evidence that the DPE primitive's behavioral model:
1. Correctly models the §4 single-buffered drain-load overlap pipeline.
2. Achieves the cycle-accurate RTL contract `T_rtl(M) = T_fill_rtl + (M-1)·T_steady` where `T_fill_rtl = L + C + O + 2` (faithful NBA-FSM).
3. Maintains per-pass functional correctness across ring wraps.
4. Survives odd-M termination without leaking state from a "phantom slot 3" (pass 3's slot 3 is never overwritten in M=7, but neither is it drained beyond pass 3 — it's correctly idle at end).

In the full smoke matrix (`run_dpe_smoke.py`), this scenario is exercised at M ∈ {1, 2, 4, 8} for NL-DPE, Azure-Lily, and DSP-MAC primitives — covering both "no wrap" (M ≤ 4) and "wrap" (M=8) cases. M=7 is not in the standard sweep but the analysis here demonstrates that odd-M with ring wrap is correctly handled by the FSM design.

**Fidelity (Option A, post Task #90)**: the simulator (`imc_core._pipeline_total_cycles`) emits the ideal `T_sim(M) = T_fill_ideal + (M-1)·T_steady` where `T_fill_ideal = L + C + O` (NO +2). The +2 cycle gap RTL-vs-sim shows up as the honest fidelity measurement:

| M | T_sim | T_rtl | Fidelity |
|---|---|---|---|
| 1 | 114 | 116 | +1.75% |
| 2 | 166 | 168 | +1.20% |
| 4 | 270 | 272 | +0.74% |
| 7 | 426 | 428 | +0.47% |
| 8 | 478 | 480 | +0.42% |

The fidelity decreases with M because the +2 amortises across M passes:
fidelity = 2 / T_sim(M) → 0 as M grows.

**Workload-level cross-reference**: the synthesizable `fc_top.v` wrapper adds **another +4 cycles** on top of the primitive's +2, for a total RTL overhead of +6 over the ideal sim model at the workload level. See `FC_GEMM_WALKTHROUGH.md §1 / §8` and `FIDELITY_METHODOLOGY.md §4.2`. The DPE primitive itself is unchanged by the workload-level wrapper — all 52/52 dpe_smoke cases pass with identical RTL cycle counts.
