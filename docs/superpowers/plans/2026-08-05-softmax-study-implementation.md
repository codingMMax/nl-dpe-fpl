# Safe-Softmax RTL + VTR Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two safe-softmax RTL blocks (AL pure-FPGA, NL DPE-based), verify them functionally and cycle-exactly with iverilog against Python oracles, run the 18-point VTR sweep, and produce the resource/Fmax/cycles/energy table.

**Architecture:** Per spec `docs/superpowers/specs/2026-08-05-softmax-rtl-vtr-study-design.md`. W=16 lockstep lanes driven by one controller; 4 overlapped stage FSMs (A max, B exp+sum, Cs scalar, D normalize) with per-stage row pointers and interlocks; single-port BRAM copies per reader (AL: 2 score copies + 2 parity exp banks; NL: 3 score copies, no exp storage); NL binds the existing `dpe` behavior model (sim) / blackbox (VTR).

**Tech Stack:** Verilog-2005, iverilog 12, VTR (`run_vtr_flow.py`), Python 3 + numpy.

**Execution location:** everything under `softmax_study/` at repo root.

---

## Locked facts (verified against source, do not re-derive)

1. **DPE driving** (`fc_verification/rtl/dpe_nldpe.v:184-289`): only `w_buf_en` + `data_in` drive it. Assert `w_buf_en` with 5 bytes/cycle for `LOAD_CYCLES = ceil(KW/5)` consecutive cycles; VMM+ACAM fire on the last strobe (blocking). COMPUTE wakes +1 cycle, runs `COMPUTE_CYCLES`; OUTPUT wakes +1 cycle, drains `OUTPUT_CYCLES` words on `data_out` with `dpe_done` high exactly when a word is valid. Contiguous passes chain back-to-back. All other input ports tied 0; per-instance `MSB_SA_Ready`/`reg_full`/`shift_add_*` wires dangle.
2. **ACAM semantics** (`dpe_nldpe.v:203-209`): mode 1: `v' = 1 + v + (v*v)/2` on the 32b signed VMM accumulator; mode 2: `v' = v − 1`. Drained byte = `v'[7:0]`.
3. **Weights**: no port — TB hierarchical `force`-equivalent procedural assign into `u.weights[r][c]` (pattern proven in `fc_verification/run_fc_smoke.py` TBs).
4. **VTR instantiation**: bare `dpe u (...)` with NO `#(...)` (blackbox `fc_verification/rtl/dpe_blackbox.v` has no parameter list; pattern `fc_verification/rtl/fc_top_synth.v:219`). Sim uses parameterized instantiation. Switch via `` `ifdef SYNTHESIS ``.
5. **mac_int_9x9**: ports `clk, reset, a[8:0], b[8:0], out[17:0]`, registered output (arch XML `<model>` line 487; usage `transformer/softmax.v:45-51`). Bare instantiation binds to arch model in VTR (proven by the 720-run fc_softmax DSE); sim needs a behavioral def (we provide `tb/sim_models.v`, excluded from VTR circuit).
6. **BRAM inference** (Parmys-proven style, `fc_top_synth.v`): `reg [W-1:0] mem[0:D-1];` + single `always @(posedge clk)` with `if (wen) mem[waddr] <= wdata; rdata <= mem[raddr];`. Single-port only — every memory in this design has one reader; concurrent readers get their own copy.
7. **Arch XMLs**: `benchmarks/arch/proposed_auto.xml` (P1), `al_like_auto.xml` (P2), `azure_lily_auto.xml` (AL). All define `dpe`, `mac_int_9x9`, `memory` models. DPE tile is `wc`.
8. **VTR flow**: `$VTR_ROOT/vtr_flow/scripts/run_vtr_flow.py <circuit> <arch> -temp_dir <dir> --route_chan_width 300 --seed N`; parse `vpr_stdout.log` (patterns in `fc_verification/run_vtr_smoke.py:103-170`).

## Fixed-point conventions (oracle and RTL must match bit-exactly)

| Item | Definition |
|---|---|
| scores `x` | int8 signed, TB-generated (numpy, seeded) |
| `max` | int8, exact row max |
| AL exp index | `u = max − x` ∈ [0,255], exact in 8b unsigned |
| AL exp ROM | `E8[u] = min(255, int(255*exp(-u/16.0) + 0.5))` |
| AL sum | Σ E8, unsigned, 16b (≤ 65280 at S=256) |
| AL recip index | `idx = sum >> R_SHIFT`, `R_SHIFT = $clog2(S)` |
| AL recip ROM | `R8[0]=255; R8[i]=min(255, (2*256+i)//(2*i))` (round-half-up 256/i) |
| AL output | `p = min(255, (E8[u]*R8[idx]) >> R_SHIFT)` via mac_int_9x9 (9b zero-padded inputs) |
| NL DPE input | `d = max(x − max, −128)` int8 (saturating clamp of [−255,0]) |
| NL exp byte | `(1 + d + (d*d)//2) & 0xFF`, summed as unsigned |
| NL log input | `lq = min(sum >> LOG_SHIFT, 127)`, `LOG_SHIFT = $clog2(S)` |
| NL log byte | `ls = lq − 1` (int8; lq ≥ 0 so ls ∈ [−1,126]) |
| NL output | `clamp(x − max − ls, −128, 127)` — true 9b diff, not the clamped d |

## File structure

```
softmax_study/
  rtl/softmax_luts.vh        # exp + recip ROM functions (generated once, checked in)
  rtl/softmax_al.v           # AL block
  rtl/softmax_nldpe.v        # NL block
  tb/sim_models.v            # behavioral mac_int_9x9 for iverilog
  tb/tb_softmax_al.v         # loads scores, runs, checks outputs + cycles
  tb/tb_softmax_nldpe.v      # same + forces identity weights into DPEs
  gen_luts.py                # emits rtl/softmax_luts.vh (run once)
  run_softmax_smoke.py       # oracle + iverilog sweep + cycle-formula check
  run_vtr_softmax.py         # wrapper gen + 18-run VTR sweep + parse
  softmax_energy.py          # op-count energy + final table assembly
  results/                   # smoke logs, VTR scratch, CSV/JSON
  SOFTMAX_STUDY.md           # methodology + result tables
```

Shared top-level interface (both blocks):

```verilog
module softmax_al #(parameter S = 128) (          // softmax_nldpe adds: C, N_EXP
    input  wire        clk, reset, start,
    output reg         done,
    input  wire        in_wen,
    input  wire [3:0]  in_lane,
    input  wire [13:0] in_addr,      // element index r*S+j within lane, r = local row
    input  wire [7:0]  in_data,
    input  wire [3:0]  out_lane,
    input  wire [13:0] out_addr,
    output wire [7:0]  out_rdata
);
```

Load is pre-`start` and unmeasured; `done` rises when the last D write retires.
Cycle count = posedge count from `start` high to `done` high.

---

### Task 1: Scaffold, LUT generator, sim models

**Files:**
- Create: `softmax_study/gen_luts.py`, `softmax_study/rtl/softmax_luts.vh`, `softmax_study/tb/sim_models.v`, `softmax_study/results/.gitkeep`

- [ ] **Step 1: Write `gen_luts.py`**

```python
#!/usr/bin/env python3
"""Emit rtl/softmax_luts.vh: exp and reciprocal ROM functions (AL block).
Conventions locked in docs/superpowers/plans/2026-08-05-softmax-study-implementation.md."""
import math
from pathlib import Path

def e8(u):  return min(255, int(255 * math.exp(-u / 16.0) + 0.5))
def r8(i):  return 255 if i == 0 else min(255, (2 * 256 + i) // (2 * i))

lines = ["// AUTO-GENERATED by gen_luts.py -- do not edit by hand.",
         "function [7:0] exp_lut; input [7:0] u; begin case (u)"]
lines += [f"  8'd{u}: exp_lut = 8'd{e8(u)};" for u in range(256)]
lines += ["  endcase end endfunction", "",
          "function [7:0] recip_lut; input [7:0] i; begin case (i)"]
lines += [f"  8'd{i}: recip_lut = 8'd{r8(i)};" for i in range(256)]
lines += ["  endcase end endfunction", ""]
Path(__file__).parent.joinpath("rtl/softmax_luts.vh").write_text("\n".join(lines))
print("wrote rtl/softmax_luts.vh")
```

- [ ] **Step 2: Run it** — `python3 softmax_study/gen_luts.py`; verify `.vh` has 2 functions × 256 entries.

- [ ] **Step 3: Write `tb/sim_models.v`**

```verilog
// Behavioral mac_int_9x9 for iverilog only. VTR binds the arch <model> instead.
module mac_int_9x9 (input reset, input [8:0] a, input [8:0] b,
                    output reg [17:0] out, input clk);
    always @(posedge clk) out <= reset ? 18'd0 : a * b;
endmodule
```

- [ ] **Step 4: Commit** — `git add softmax_study && git commit -m "softmax_study: scaffold + LUT ROMs + sim models"`

---

### Task 2: Oracle + smoke driver skeleton (test-first)

**Files:**
- Create: `softmax_study/run_softmax_smoke.py`

- [ ] **Step 1: Write the oracle functions and driver.** The oracle implements the fixed-point table above, emits per-case hex files (`scores.hex`, `expected.hex`) into `results/smoke_<case>/`, builds the iverilog command, parses TB output (`CYCLES=<n>`, `MISMATCH ...`, `TB_PASS`/`TB_FAIL`), and compares measured cycles against a `predict_cycles(arch, S, C, n)` function. Cycle constants start as the spec §3/§4 predictions and get locked to RTL in Tasks 3/4.

```python
# Core oracle (AL) — bit-exact per conventions table:
def oracle_al(scores, S):            # scores: (rows, S) int8 numpy
    R_SHIFT = S.bit_length() - 1     # log2(S)
    out = np.zeros_like(scores, dtype=np.uint8)
    for r in range(scores.shape[0]):
        row = scores[r].astype(int); m = row.max()
        e = np.array([E8[m - x] for x in row])        # E8 from gen_luts formulas
        s = int(e.sum()); idx = min(255, s >> R_SHIFT)
        rec = R8[idx]
        out[r] = np.minimum(255, (e * rec) >> R_SHIFT)
    return out

# Core oracle (NL):
def oracle_nl(scores, S):
    LOG_SHIFT = S.bit_length() - 1
    out = np.zeros_like(scores, dtype=np.int8)
    for r in range(scores.shape[0]):
        row = scores[r].astype(int); m = row.max()
        d = np.maximum(row - m, -128)
        eb = np.array([(1 + v + (v * v) // 2) & 0xFF for v in d])
        s = int(eb.sum()); lq = min(s >> LOG_SHIFT, 127); ls = lq - 1
        out[r] = np.clip(row - m - ls, -128, 127)
    return out
```

Cases: `al_s128`, `al_s256`, `nl_p1_s128` (C=128,n=1), `nl_p2_s128` (C=256→E=128,n=1; sim-identical to p1 but kept as a case for bookkeeping), `nl_p1_s256` (n=2,E=128), `nl_p2_s256` (n=1,E=256). Scores: `np.random.RandomState(7).randint(-128,128,(S,S))`.

- [ ] **Step 2: Run driver with no RTL present** — expect clean per-case FAIL (compile error reported, not a crash).

- [ ] **Step 3: Commit** — `git commit -m "softmax_study: bit-exact oracles + smoke driver (failing: no RTL yet)"`

---

### Task 3: `softmax_al.v` + TB → smoke green

**Files:**
- Create: `softmax_study/rtl/softmax_al.v`, `softmax_study/tb/tb_softmax_al.v`

Structure (single controller, 16 generate-lane datapaths):

- Memories per lane, all single-port, style per Locked fact 6:
  - `score_a` copy: 128b × (RPL·S/16) — reader: stage A (and load writes)
  - `score_b` copy: same geometry — reader: stage B
  - `exp_bank0/1`: 128b × (S/16) — writer B (row parity), reader D (other parity)
  - `out_mem`: 128b × (RPL·S/16) — writer D, reader: external port
- Stage FSMs with row pointers `a_row, b_row, c_row, d_row` and interlocks:
  `B starts row r when a_done > r; Cs when b_done > r; D when c_done > r; B starts row r when d_done > r-2` (parity-bank safety).
- Stage A: word-per-cycle read, 16-input max tree pipelined 16→4 (reg) →1 (reg); `max_hist[r]` regfile write at drain.
- Stage B: word read → 16× `u = max_hist[b_row] − x` → 16× `exp_lut(u)` (reg) → pack write `exp_bank[b_row%2]`; adder tree 16→4 (reg) →1 (reg) → 32b acc (reg).
- Stage Cs: 1 cycle: `rec_hist[r] <= recip_lut(min(255, sum >> R_SHIFT))`.
- Stage D: word read from `exp_bank[d_row%2]` → 16× mac_int_9x9 (`{1'b0,e}`,`{1'b0,rec_hist[d_row]}`) → `p = min(255, out >> R_SHIFT)` → pack → `out_mem`.
- `done` when `d_done == RPL`.

- [ ] **Step 1: Write the TB** — `tb_softmax_al.v`: `$readmemh` scores → byte-serial load all lanes; pulse `start`; count cycles to `done`; read back all outputs via port; compare against `expected.hex`; print `CYCLES=%0d` and `TB_PASS`/`TB_FAIL` + first 10 mismatches. Plusargs/defines: `-DS_TB=<S>`.
- [ ] **Step 2: Run smoke, watch it fail** (module missing → compile fail listed as FAIL).
- [ ] **Step 3: Write `softmax_al.v`** per structure above (ROMs via `` `include "softmax_luts.vh" `` inside the module).
- [ ] **Step 4: Iterate `al_s128` to functional PASS.** Then `al_s256`.
- [ ] **Step 5: Lock the cycle formula.** Read measured CYCLES for both S; set `predict_cycles('al', S)` = exact closed form `FILL_AL + (RPL−1)·max(stage occupancies)` with the measured fill constant; document each +k in a comment; re-run — both cases must report `cycles: exact`.
- [ ] **Step 6: Commit** — `git commit -m "softmax_study: AL safe-softmax RTL, smoke 2/2 functional+cycle-exact"`

---

### Task 4: `softmax_nldpe.v` + TB → smoke green

**Files:**
- Create: `softmax_study/rtl/softmax_nldpe.v`, `softmax_study/tb/tb_softmax_nldpe.v`

Parameters: `S`, `C`, `N_EXP` (n), derived `E = S/N_EXP`, `LCYC = (E+4)/5`, `OCYC = LCYC`, `CCYC = 10`, `LOG_LCYC = 4`.

- Memories per lane: `score_a` (128b), `score_p` (40·n b words — DPE feed geometry), `score_d` (128b), `out_mem` (128b). No exp memory.
- DPE instantiation (per Locked fact 4):

```verilog
generate for (gk = 0; gk < 16; gk = gk + 1) begin : lane
  for (ge = 0; ge < N_EXP; ge = ge + 1) begin : edpe
`ifdef SYNTHESIS
    dpe u_exp (.clk(clk), .reset(reset), .data_in(exp_din[gk][ge]),
               .nl_dpe_control(2'b00), .shift_add_control(1'b0),
               .w_buf_en(exp_wen), .shift_add_bypass(1'b0),
               .load_output_reg(1'b0), .load_input_reg(1'b0),
               .MSB_SA_Ready(nc0[gk][ge]), .data_out(exp_dout[gk][ge]),
               .dpe_done(exp_done[gk][ge]), .reg_full(nc1[gk][ge]),
               .shift_add_done(nc2[gk][ge]), .shift_add_bypass_ctrl(nc3[gk][ge]));
`else
    dpe #(.KERNEL_WIDTH(E), .NUM_COLS(E), .DPE_BUF_WIDTH(40),
          .COMPUTE_CYCLES(10), .ACAM_MODE(1)) u_exp (/* same ports */);
`endif
  end
end endgenerate
// + one shared log DPE: KERNEL_WIDTH=16, NUM_COLS=16, ACAM_MODE=2
```

- Stage B per row: `LCYC` strobe cycles: read `score_p` word (5n elems) → 5n subtract-clamp (`d = max(x − max_hist, −128)`) → n×40b to the n DPE `data_in`, `exp_wen` high. Drain: on `exp_done[k][e]`, consume n×40b/cycle → 5n bytes → adder tree (5n-input, 2 reg stages) → acc. Count `OCYC` drain words → row sum done.
- Stage Cs per row-group: all 16 lane sums ready (lockstep) → `lq[k] = min(sum[k] >> LOG_SHIFT, 127)` → 4 strobes into log DPE (`{lq[4],lq[3],lq[2],lq[1],lq[0]}` then 5–9, 10–14, `{32'b0,lq[15]}`) → drain 4 words → `ls_hist[r][k] = byte_k`.
- Stage D per row: read `score_d` 16-wide → `out = clamp(x − max_hist[d_row] − ls_hist[d_row], −128, 127)` (9b/10b signed arithmetic) → pack → `out_mem`.
- TB identity-weight forcing (initial block, after elaboration):

```verilog
integer i;
initial begin
  #1;
  for (i = 0; i < 16; i = i + 1) force_identity(i);  // task with hierarchical refs:
  // dut.lane[k].edpe[e].u_exp.weights[r][r] = 8'sd1;  for r < E
  // dut.u_log.weights[r][r] = 8'sd1;                  for r < 16
end
```

(iverilog: hierarchical procedural assignment into arrays inside generate scopes — same mechanism `run_fc_smoke.py` TBs use. If scoped array assign fights iverilog, fall back to a `defparam`-free approach: give `dpe` an optional `IDENTITY_INIT` sim-only parameter? NO — do not modify `dpe_nldpe.v`. The proven fallback is a TB `initial` with explicit generate-scope paths, which run_fc_smoke already exercises.)

- [ ] **Step 1: Write the TB** (mirror of AL TB + weight forcing; `-DS_TB -DC_TB -DNEXP_TB`).
- [ ] **Step 2: Run smoke `nl_p1_s128`, watch it fail.**
- [ ] **Step 3: Write `softmax_nldpe.v`.**
- [ ] **Step 4: Iterate to functional PASS in order:** `nl_p1_s128` → `nl_p2_s128` → `nl_p2_s256` (n=1, E=256) → `nl_p1_s256` (n=2 — exercises multi-DPE lanes).
- [ ] **Step 5: Lock cycle formula** exactly as Task 3 Step 5 (`predict_cycles('nl', S, C, n)`), constants documented per +k.
- [ ] **Step 6: Commit** — `git commit -m "softmax_study: NL safe-softmax RTL (17/33-DPE), smoke 4/4 functional+cycle-exact"`

---

### Task 5: VTR wrapper generator + single-point sanity

**Files:**
- Create: `softmax_study/run_vtr_softmax.py`

- [ ] **Step 1: Write the runner.** Reuse the skeleton of `fc_verification/run_vtr_smoke.py` (metric regexes, resource parser, table writer — copy the functions, they are ~100 lines). Point definitions:

```python
POINTS = [  # (row, arch_xml, S, C, n_exp, rtl)
  ("AL",  "azure_lily_auto.xml", 128, None, None, "softmax_al.v"),
  ("AL",  "azure_lily_auto.xml", 256, None, None, "softmax_al.v"),
  ("P1",  "proposed_auto.xml",   128, 128, 1, "softmax_nldpe.v"),
  ("P1",  "proposed_auto.xml",   256, 128, 2, "softmax_nldpe.v"),
  ("P2",  "al_like_auto.xml",    128, 256, 1, "softmax_nldpe.v"),
  ("P2",  "al_like_auto.xml",    256, 256, 1, "softmax_nldpe.v"),
]
SEEDS = [1, 2, 3]
```

Circuit file assembly per point (order matters — top module LAST):
`"`define SYNTHESIS\n"` + `softmax_luts.vh`? — NO: the `.vh` is `` `include ``d by the RTL; pass `-I softmax_study/rtl` equivalent by inlining the include before concatenation (read the .v, textually substitute the include line with the .vh contents — 3 lines of python) + (NL only) `dpe_blackbox.v` + the block RTL + a generated top wrapper instantiating the block with that point's parameters, all ports to top-level pins.

- [ ] **Step 2: Run ONE point** (`AL, S=128, seed 1`) end-to-end. Check: `dsp_top` count ≈ ceil(256 macs/4-per-tile), `memory` > 0 (buffers inferred, ROMs NOT in memory), Fmax parsed. Fix inference issues here (this is the risk gate from spec §11).
- [ ] **Step 3: Run ONE NL point** (`P1, S=128, seed 1`). Check `wc = 17`, `dsp_top = 0`.
- [ ] **Step 4: Commit** — `git commit -m "softmax_study: VTR runner + 2-point sanity (AL/P1 S=128)"`

---

### Task 6: Full 18-run sweep

- [ ] **Step 1: Run** `python3 softmax_study/run_vtr_softmax.py --all` (6 points × 3 seeds; sequential or --jobs 3). Expect minutes-to-tens-of-minutes per point at S=256.
- [ ] **Step 2: Verify** `results/vtr_softmax.json`: 18 OK rows, per-point seed-averaged Fmax, resource counts stable across seeds (they must be identical — placement changes, netlist doesn't).
- [ ] **Step 3: Commit results** — `git add softmax_study/results/vtr_softmax.json results/vtr_softmax.log && git commit -m "softmax_study: 18-run VTR sweep results"`

---

### Task 7: Energy calculator + study document

**Files:**
- Create: `softmax_study/softmax_energy.py`, `softmax_study/SOFTMAX_STUDY.md`
- Modify: `docs/superpowers/specs/2026-08-05-softmax-rtl-vtr-study-design.md` (as-built deltas)

- [ ] **Step 1: Write `softmax_energy.py`** — op counts × constants per spec §6 (constants read from the two arch JSONs at runtime; CLB add/compare from the locked `ref_*` values, stated inline with provenance comments). Inputs: measured cycles (from `results/smoke.log`), Fmax (from `results/vtr_softmax.json`). Output: `results/softmax_table.csv` + markdown table with columns `Arch | R×C | S | CLB | DSP | wc | BRAM | Fmax | Cycles | Latency µs | rows/s | pJ total | pJ/element`.
- [ ] **Step 2: Run it; sanity-check** P2 pJ/element ≈ 0.586 at S=128 and ≈ 0.293 at S=256 on the DPE component (spec §5 prediction), AL DSP energy = S²·1.2 pJ.
- [ ] **Step 3: Write `SOFTMAX_STUDY.md`** — methodology summary (link spec), conventions table, as-built deltas (NL has no exp storage — pass 3 recomputes x−max; NL B-stage plumbing is 5n-wide at the port, per spec's port-honest rule; score-copy counts), result tables, and the two predictions checked (P1-vs-P2 latency, energy crossover).
- [ ] **Step 4: Amend spec** — §4 datapath: pass 3 reads score (not exp bram), exp values are sum-only; §9: add `gen_luts.py`, `sim_models.v`. Mark spec **as-built**.
- [ ] **Step 5: Final commit** — `git commit -m "softmax_study: energy model + study doc + as-built spec"`

---

## Self-review notes

- Spec coverage: §2 shared spec → Tasks 3/4 structure; §3 AL → Task 3; §4 NL (17/33 DPEs, shared log, port-honest cycles) → Task 4; §5 predictions → Task 7 Step 2/3; §6 energy → Task 7; §7 VTR (18 runs, 3 seeds, chan width fixed 300) → Tasks 5/6; §8 verification (functional bit-exact vs model semantics + cycle-exact formula) → Tasks 2/3/4; §9 files → all tasks; §10 table → Task 7; §11 risks → Task 5 Step 2 (inference gate), Task 4 (occupancy reporting via TB `CYCLES` + formula lock).
- NL S=128 P1 vs P2 are sim-identical (E=128, n=1) — both kept as smoke cases, and both get separate VTR rows (different arch XML). Stated in Task 2.
- Type consistency: `predict_cycles(arch, S, C, n)` used by Tasks 2/3/4; TB prints `CYCLES=` consumed by smoke driver and `softmax_energy.py`.
