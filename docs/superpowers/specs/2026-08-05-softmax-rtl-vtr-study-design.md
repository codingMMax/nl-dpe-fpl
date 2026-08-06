# Safe-Softmax RTL + VTR Study — Design

**Date**: 2026-08-05
**Status**: design approved, pending implementation plan
**Scope**: standalone study. Independent of `azurelily/IMC/*`. No simulator
changes, no fidelity gate against `scheduler.py`. Cycles, throughput, and
energy are counted explicitly here.
**Location**: everything new lives in `softmax_study/` at the repo root — not
inside `fc_verification/`. The only reach-outs are read-only references to
`fc_verification/rtl/dpe_nldpe.v` (behavior model, for iverilog) and
`fc_verification/rtl/dpe_blackbox.v` (VTR blackbox). Those are not copied: the
DPE port contract and FSM semantics must stay single-sourced.

---

## 1. Purpose

Measure and compare the cost of one softmax stage across three architecture
points, with real synthesis numbers rather than assumed ones:

| Row | Crossbar R×C | VTR arch XML | Softmax uses DPE? |
|---|---|---|---|
| Proposed-1 | 1024×128 | `benchmarks/arch/proposed_auto.xml` (wc 3w×7h) | yes |
| Proposed-2 | 1024×256 | `benchmarks/arch/al_like_auto.xml` (wc 5w×8h) | yes |
| Azure-Lily | 512×128 | `benchmarks/arch/azure_lily_auto.xml` (wc 6w×5h) | no (`wc`=0) |

Row identities from `paper/scripts/plot_bert_block_comparison.py:46-48`.

Deliverable: a table of CLB / DSP / DPE(`wc`) / BRAM / Fmax per row per
sequence length, plus explicitly counted cycles, derived throughput, and
analytically computed energy.

---

## 2. Shared workload spec

Every row obeys the same assumption. Only the exp/log mechanism differs.

| Item | Value |
|---|---|
| Workload | safe softmax over an S×S score matrix (S rows, S elements/row) |
| S sweep | 128, 256 |
| Lanes | W = 16 row-parallel (`total_softmax_lanes`, both arch JSONs) |
| Rows per lane | `ceil(S/W)` → 8, 16 |
| Tiling | none — full row buffered, classic 3-pass safe softmax |
| Element type | int8 scores; exp values int8 (`LUT_OUT_W` param, default 8) |
| Sum accumulator | 32-bit |
| CLB datapath width | 16 elements/cycle, both architectures |
| DPE port width | 40 bit = 5 int8 elements/cycle (`dpe_buf_width`) |
| Row scheduling | **pipelined** — row *r+1*'s max/exp overlaps row *r*'s log/normalize |
| Lane scheduling | 16 lanes in lockstep, one row each per row-group |
| Buffering | score and exp buffers **double-buffered** (2 × S per lane each) |

Both architectures use the same pipelined discipline, so the comparison stays
symmetric. Block cycles:

```
T_block = T_fill + (rows_per_lane − 1) × T_steady
T_fill  = A + B + Cs + D           # one row through every stage
T_steady= max(A, B, Cs, D)         # slowest stage sets the rate
```

where A = max pass, B = exp pass, Cs = scalar op (log fire / reciprocal),
D = normalize pass. Double buffering is what allows the overlap: row *r+1*
writes one score bank while row *r* still reads the other, and likewise for exp.

Lockstep is required by the shared log DPE — all 16 sums must be ready before it
fires. It moves only those 16 sums through the port, so its occupancy is 10
cycles against an exp stage of 26–52 (§4): comfortable headroom, never the
limiter. The TB reports its occupancy so this stays a measurement.

**Output format**: NL emits log-domain values (`log p_i`), AL emits linear
probabilities (`p_i`). This is the `log_softmax_fusion` credit — the downstream
`mac_sv` DIMM consumes log-domain directly, so a real pipeline never undoes the
exp. Every table and figure must say so in the caption; the two blocks do not
produce identical outputs.

### Safe softmax, 3 passes per row

`softmax(x_i) = exp(x_i − max_j x_j) / Σ_j exp(x_j − max_j x_j)`

| Pass | Work | Azure-Lily | NL-DPE |
|---|---|---|---|
| 1 | `max` over S | CLB comparator tree, 16-wide | CLB comparator tree, 16-wide |
| 2 | `x − max`, exp, accumulate Σ | CLB subtract → 16 exp LUT ROMs → CLB adder tree | CLB subtract → **DPE(I\|exp)** → CLB adder tree |
| 3 | scalar op + combine | recip LUT ROM → 16 DSP multiplies | **DPE(I\|log)** on batched sums → CLB subtract (log domain) |

Pass 1 and pass 2 both read the score buffer; pass 2 writes the exp buffer;
pass 3 reads it. Four memory touches per element, per the standard 3-pass
formulation (Milakov & Gimelshein, arXiv:1805.02867).

CLB stages run at the same 16-wide chunk on both architectures — identical
fabric, identical width — so the only architectural difference is the
mechanism inside pass 2 and pass 3.

---

## 3. `softmax_al.v` — pure FPGA

Parameters: `S`, `W=16`, `E_CLB=16`, `LUT_OUT_W=8`.

Per lane:
- score buffer (S deep) and exp buffer (S deep), inferred BRAM
- 16-wide comparator tree + running max register (pass 1)
- 16 subtractors (pass 2 and 3 share them)
- 16 exp LUT ROMs, 256 × `LUT_OUT_W`, written as combinational `case` so they
  map to CLB LUTs rather than being inferred as BRAM
- 16-input CLB adder tree + 32-bit accumulator
- 1 reciprocal LUT ROM, 256 × 16b, same `case` treatment
- 16 `mac_int_9x9` multipliers (pass 3)

No `dpe` instance, so VTR reports `wc = 0` even though the AL arch XML defines
the tile.

Index convention: exp LUT is addressed by `max − x ∈ [0, 255]`, returning
`exp(−(max−x))` in fixed point. Safe softmax guarantees a non-positive
exponent, so no saturation branch is needed.

### Datapath

```
LANE k   (k = 0..15)   owns rows m = k, k+16, k+32, ...   (S/16 rows)

   score_bram   [bank A | bank B]   double-buffered, S deep, int8
        |
        |  16 elem/cyc
        +-------------------------------+------------------------+
        |                               |                        |
   PASS 1 (max)                    PASS 2 (exp, sum)             |
        v                               v                        |
  +------------------+          +------------------+             |
  | 16-wide compare  |          | 16x subtract     |<--[max reg]<-+
  |      tree        |--> [max] |    x - max       |
  +------------------+          +--------+---------+
   occ S/16, +4 drain                    | 16 elem/cyc
                                         v
                              +------------------------+
                              | 16 x exp LUT ROM       |  256 x LUT_OUT_W
                              | comb. case -> CLB LUTs |  ~8 CLB each
                              +-----------+------------+
                                          | 16 elem/cyc
                        +-----------------+-----------------+
                        v                                   v
              exp_bram [bank A | bank B]          +------------------+
                        |                         | 16-in adder tree |
                        |                         +--------+---------+
                        |                                  v
                        |                           [ sum acc 32b ]
                        |                                  |
                        |                                  v
                        |                        +------------------+
                        |                        |  recip LUT ROM   | 256 x 16b
                        |                        +--------+---------+
   PASS 3 (normalize)   |                                  | 1/sum
                        |  16 elem/cyc                     v
                        +--------------------->+-----------------------+
                                               | 16 x mac_int_9x9      |
                                               |   p = e * (1/sum)     |
                                               +-----------+-----------+
                                                           v
                                                       out_bram
                                                    (linear probs)
```

### Cycles

Each stage has a *latency* (its contribution to fill) and an *occupancy* (how
long it holds the pipeline per row, which sets the rate). The trees are
pipelined — they accept 16 new elements every cycle — so their drain counts
in latency only.

```
             latency                    occupancy
A  (max)     ceil(S/16) + 4             ceil(S/16)
B  (exp)     ceil(S/16) + 4             ceil(S/16)
Cs (recip)   1                          1
D  (norm)    ceil(S/16)                 ceil(S/16)

T_block = sum(latency) + (rows_per_lane − 1) × max(occupancy)
```

| S | fill | steady | rows/lane | block cycles |
|---|---|---|---|---|
| 128 | 33 | 8 | 8 | 89 |
| 256 | 57 | 16 | 16 | 297 |

---

## 4. `softmax_nldpe.v` — DPE for exp and log

Parameters: `S`, `W=16`, `E_CLB=16`, `C`, `R`, `BUF=40`, `CCYC=10`.
Elaborated twice: (R=1024, C=128) and (R=1024, C=256).

Per lane: score buffer, exp buffer, 16-wide comparator tree, 16 subtractors,
16-input CLB adder tree, 32-bit accumulator, 16-wide log-domain subtractor.

DPE instances — `16·n + 1`, where **n = `ceil(S/C)` exp DPEs per lane**:
- **exp DPEs**, n per lane:
  `dpe #(.KERNEL_WIDTH(E), .NUM_COLS(E), .ACAM_MODE(1))` with `E = S/n`, all n
  firing concurrently so a row completes in one pass each.
- **1 shared log DPE**:
  `dpe #(.KERNEL_WIDTH(16), .NUM_COLS(16), .ACAM_MODE(2))`. The 16 lanes each
  finish a row and hand up one sum; one pass converts all 16, `ceil(S/16)`
  passes for the whole matrix.

| S | Arch | n | E | exp DPEs | log DPEs | total |
|---|---|---|---|---|---|---|
| 128 | P1 (C=128) | 1 | 128 | 16 | 1 | 17 |
| 128 | P2 (C=256) | 1 | 128 | 16 | 1 | 17 |
| 256 | P1 (C=128) | 2 | 128 | 32 | 1 | 33 |
| 256 | P2 (C=256) | 1 | 256 | 16 | 1 | 17 |

`n = ceil(S/C)` is the free-split point: every pass stays full (E = C, or E = S
when the row is shorter than the crossbar), so splitting adds parallelism
without adding passes, and pass count is what drives energy. Splitting further
would under-fill the crossbar — which still fires completely — and cost energy
1:1 for cycles.

Per-lane log DPEs would waste C−1 of the ACAM's C output slots on a scalar; one
shared, batched instance is the correct consequence of the ACAM being as wide as
the crossbar. `ACAM_MODE` is elaboration-time, so exp and log cannot share an
instance.

Output is log-domain (`log_softmax_fusion` semantics): pass 3 subtracts
`log(Σ)` rather than multiplying by `1/Σ`, so no DSP is needed. The `log` exists
precisely to turn the division into a subtraction — NL-DPE has no divider and no
elementwise multiplier, but the ACAM computes `log` natively on the one scalar
that needs it.

### Datapath — one lane

```
LANE k    n = ceil(S/C) exp DPEs per lane,  E = S/n elements per DPE per row

   score_bram   [bank A | bank B]   double-buffered, S deep, int8
        |
        +-------------------------------+
        |  16 elem/cyc                  |  5n elem/cyc
   PASS 1 (max)                    PASS 2 (exp, sum)
        v                               v
  +------------------+          +------------------+
  | 16-wide compare  |          | 16x subtract     |<--[max reg]
  |      tree        |--> [max] |    x - max       |
  +------------------+          +--------+---------+
   occ S/16, +4 drain                    |
                          5 elem/cyc per DPE (40-bit port)
                    +------------------+-+- - - - - +
                    v                  v            v
            +---------------+  +---------------+   (n instances)
            | dpe ACAM_MODE |  | dpe ACAM_MODE |
            |     = 1 (exp) |  |     = 1 (exp) |   E x E identity
            | exp DPE 0     |  | exp DPE n-1   |   L = O = ceil(E/5)
            +-------+-------+  +-------+-------+   CCYC = 10
                    |  5 elem/cyc      |           occ = max(L,10,O)
                    +--------+---------+
                             |
              +--------------+--------------+
              v                             v
     exp_bram [bank A | bank B]    +------------------+
              |                    | 16-in adder tree |
              |                    +--------+---------+
              |                             v
              |                      [ sum acc 32b ] ---> to shared log DPE
              |                                                    |
   PASS 3 (normalize)                                              |
              |  16 elem/cyc                     log_sum[k] <------+
              v                                       |
        +-------------------+                         |
        | 16x subtract      |<------------------------+
        |   e - log(sum)    |
        +---------+---------+
                  v
              out_bram        (log-domain: log p_i)
```

### Datapath — shared log DPE

```
   lane0.sum  --+
   lane1.sum  --+    16 sums = 16 bytes
      ...       +--> +---------------------+
   lane15.sum --+    | dpe ACAM_MODE = 2   |  16 x 16 identity, ACAM log
                     | shared, 1 instance  |  L = ceil(16*8/40) = 4
                     +----------+----------+  CCYC = 10, O = 4
                                |             latency 20, occupancy 10
              +-----------------+-----------------+
              v                 v                 v
        log_sum[0]         log_sum[1]  ...   log_sum[15]
          -> LANE 0          -> LANE 1         -> LANE 15
```

One pass converts all 16 lane sums. Occupancy 10 cycles against an exp stage of
26–52 — never the limiter.

### Pipeline

```
  row-group g    [A max][B exp          ][Cs log][D norm]
  row-group g+1         [A max][B exp          ][Cs log][D norm]
  row-group g+2                [A max][B exp          ][Cs log][D norm]
                        |<-- steady = max(A,B,Cs,D) = B (exp) -->|
```

Double buffering is what allows the overlap: row-group *g+1* fills one score
bank while *g* still reads the other, same for exp.

### Port bandwidth governs pass 2

The ACAM produces C values simultaneously, but they drain through the 40-bit
port at 5 elements/cycle. **Port time scales with the elements actually moved,
not with C** — feeding 128 values into a 256-column crossbar costs 26 cycles,
not 52. The full crossbar still fires, so energy is charged on C (§6) while time
is charged on E.

```
E = elements moved per pass = S / n
L = O = ceil(E·8/40)         # 26 cycles (E=128), 52 (E=256), 4 (E=16)
C_cyc = 10                   # PRECISION 8 + (PIPELINE_DEPTH 2 − 1) + ACAM_CYCLES 1
latency   = L + C_cyc + O + 2
occupancy = max(L, C_cyc, O)
```

This diverges deliberately from `imc_core.dimm_nonlinear`, which charges
`load_dim = cfg.cols` on every pass. That is the conservative choice inside the
simulator; this study models the port honestly. Do not cross-compare cycle
counts between the two without accounting for it.

### Cycles

```
             latency                    occupancy
A  (max)     ceil(S/16) + 4             ceil(S/16)
B  (exp)     L + 10 + O + 2             max(L, 10, O)
Cs (log)     4 + 10 + 4 + 2 = 20        10
D  (norm)    ceil(S/16)                 ceil(S/16)

T_block = sum(latency) + (rows_per_lane − 1) × max(occupancy)
```

The log fire is one per row-group and each lane contributes one row per
row-group, so it appears once per row in the fill term. Its occupancy is 10
cycles — it moves only 16 sums through the port — so it is never the limiter.

| S | Arch | A | B | Cs | D | fill | steady | rows/lane | block cycles |
|---|---|---|---|---|---|---|---|---|---|
| 128 | P1 | 8 | 26 | 10 | 8 | 104 | 26 | 8 | 286 |
| 128 | P2 | 8 | 26 | 10 | 8 | 104 | 26 | 8 | 286 |
| 256 | P1 (n=2) | 16 | 26 | 10 | 16 | 120 | 26 | 16 | 510 |
| 256 | P2 | 16 | 52 | 10 | 16 | 172 | 52 | 16 | 952 |

Columns A–D are occupancies; `fill` sums the latencies. The exp stage sets the
rate in every case.

---

## 5. Predictions to check against the measurements

These fall directly out of the mapping and should be validated, not assumed.

**Throughput is port-bound, not column-bound.** A lane moves `5·n`
elements/cycle through its exp DPEs regardless of C; AL moves 16. The cycle
ratio is therefore `16/(5n)` — crossbar width does not appear. Predicted total
block cycles (16 lanes concurrent, pipelined rows):

| S | Azure-Lily | Proposed-1 | Proposed-2 | NL/AL |
|---|---|---|---|---|
| 128 | 89 | 286 (n=1) | 286 (n=1) | 3.2× |
| 256 | 297 | 510 (n=2) | 952 (n=1) | 1.7× / 3.2× |

At S=128 the two block sizes are **cycle-identical** — both move 128 elements
per pass through the same port. At S=256 Proposed-1 pulls ahead only because
`ceil(S/C) = 2` earns it a free second exp DPE per lane, while Proposed-2's
single 256-element pass stays port-bound at 52 cycles. The 3.2× against AL is
exactly the 16-vs-5 element-rate ratio; if the measurement disagrees, the
mapping is wrong somewhere.

Whether AL also wins in *time* depends on Fmax, which is what the VTR runs
settle: the DPE is a hard block, while AL's critical path runs through LUT
ROMs, a 16-input adder tree, and DSP chains.

**Energy per element favors the wider crossbar.** Per-pass DPE energy:

```
E_pass = k_vmm·e_analoge + k_conv·e_conv + k_digital·e_digital·C
       = 8×3.89 + 8×0 + 1×0.171445313×C
P1: 31.12 + 21.94 = 53.07 pJ/pass → 53.07/128 = 0.415 pJ/element
P2: 31.12 + 43.89 = 75.01 pJ/pass → 75.01/256 = 0.293 pJ/element
```

The VMM term is C-independent, so it amortizes better at C=256 — **but only if
the crossbar is full**. It fires completely whether or not every column carries
a useful element, so a half-filled pass pays full price:

| S | P1 elements/pass | P1 pJ/element | P2 elements/pass | P2 pJ/element |
|---|---|---|---|---|
| 128 | 128 of 128 | 53.07/128 = **0.415** | 128 of 256 | 75.01/128 = **0.586** |
| 256 | 128 of 128 | 106.14/256 = **0.415** | 256 of 256 | 75.01/256 = **0.293** |

So the two-point sweep straddles the crossover: at S=128 Proposed-1 wins both
cycles (tied) and energy (1.41×); at S=256 Proposed-2 wins energy by 1.42× and
loses cycles by 1.87×. Proposed-2's block size only pays off once the sequence
is long enough to fill it.

**Azure-Lily should be latency-competitive and resource-expensive.** Its 16
exp LUTs per lane process 16 elements/cycle against the DPE path's 5, but cost
16 ROMs × 16 lanes of CLB plus 256 DSP MACs.

---

## 6. Analytical energy model

Constants (from `azurelily/IMC/configs/*.json` and `imc_core/config.py`,
read directly — no simulator import):

| Constant | Value | Source |
|---|---|---|
| `e_analoge_pj` | 3.89 | `nl_dpe.json` |
| `e_conv_pj` | 0 (NL) | `nl_dpe.json`. AL's 2.33 is unused here — AL's softmax fires no DPE pass |
| `e_digital_pj` | 0.171445313 per column | `nl_dpe.json` |
| CLB add | 0.08498 pJ/op | `ref_sum_pj = 84.98358e-6 × 1e3` |
| CLB compare | 0.26439 pJ/op | `ref_compare_pj = (793.1801e-6/3) × 1e3` |
| CLB generic | 0.660 pJ/mac | `clb_pj_per_mac` |
| DSP MAC | 1.2 pJ | `dsp_pj_per_mac` |
| BRAM access | 0.0495 pJ | `bram_pj_per_access` |

Op counts per S×S softmax:

| Op | Count | Charged to |
|---|---|---|
| max compares | `S² − S` | CLB compare |
| subtract (pass 2) | `S²` | CLB add |
| sum adds | `S² − S` | CLB add |
| exp LUT lookups | `S²` | AL only: `n_clb_per_rom × clb_pj_per_mac × α` |
| recip lookups | `S` | AL only, same form |
| DSP multiplies | `S²` | AL only: DSP MAC |
| log-domain subtract | `S²` | NL only: CLB add |
| exp DPE passes | `S × ceil(S/C)` | NL only: `E_pass` |
| log DPE passes | `ceil(S/16)` | NL only: `E_pass` |
| BRAM accesses | `≈ 6S²` element-granularity | both |

`α` is the LUT activity factor, default 1.0 (upper bound), exposed as a knob
and reported alongside results so the assumption is visible.

Latency `= measured_cycles / Fmax_VTR`. Throughput `= 1 / latency` softmax
matrices/s, also reported as rows/s and elements/s. Energy reported as total
pJ per S×S softmax and pJ/element.

---

## 7. VTR plan

- Runner modeled on `fc_verification/run_vtr_smoke.py`: generate a per-point
  top wrapper, concatenate `dpe_blackbox.v` + the softmax RTL + wrapper into
  one circuit file, invoke `run_vtr_flow.py`, parse `vpr_stdout.log`.
- 3 archs × 2 sequence lengths × 3 seeds (1, 2, 3) = **18 runs**. Fmax averaged
  over seeds, per the `MULTI_SEEDS` convention in `gemv_dse.py:79`.
- `--route_chan_width` fixed across all points so Fmax is comparable.
- Metrics parsed: `clb`, `dsp_top`, `wc`, `memory`, `io`, Fmax, wirelength,
  plus grid dimensions for an optional derived area column.
- All paths absolute (VTR changes CWD).

---

## 8. Verification

No fidelity gate against the simulator — this study is self-contained. Two
checks per (arch, S):

1. **Functional**: TB drives a known score matrix and checks the output.
   The NL check is against the behavior model's own ACAM semantics
   (`ACAM_MODE=1` is `1 + x + x²/2`, `ACAM_MODE=2` is `x − 1`, per
   `dpe_nldpe.v:203-207`), not against true `exp`/`log`. The AL check is
   against the LUT contents the generator emitted.
2. **Cycle count**: TB measures start→done and the runner compares it to the
   §3/§4 analytical formula. Any mismatch means the formula (used for the
   throughput column) is wrong and must be fixed before the table is trusted.

---

## 9. Files

All new files live under `softmax_study/`:

| Path | Role |
|---|---|
| `softmax_study/rtl/softmax_al.v` | AL block, parameterized by S |
| `softmax_study/rtl/softmax_nldpe.v` | NL block, parameterized by (S, C, n) |
| `softmax_study/tb/tb_softmax_al.v` | functional + cycle TB |
| `softmax_study/tb/tb_softmax_nldpe.v` | functional + cycle TB |
| `softmax_study/run_softmax_smoke.py` | iverilog sweep, cycle-formula check |
| `softmax_study/run_vtr_softmax.py` | 18-point VTR sweep → resources + Fmax |
| `softmax_study/softmax_energy.py` | analytical energy/throughput calculator |
| `softmax_study/results/` | VTR scratch dirs, logs, result JSON/CSV |
| `softmax_study/SOFTMAX_STUDY.md` | methodology + result tables |

Read-only dependencies, referenced by absolute path, never copied:

| Path | Used by |
|---|---|
| `fc_verification/rtl/dpe_nldpe.v` | `tb_softmax_nldpe.v` (behavior model for iverilog) |
| `fc_verification/rtl/dpe_blackbox.v` | `run_vtr_softmax.py` (VTR blackbox) |
| `benchmarks/arch/{proposed,al_like,azure_lily}_auto.xml` | `run_vtr_softmax.py` |
| `azurelily/IMC/configs/{nl_dpe,azure_lily}.json` | `softmax_energy.py` (constants only) |

---

## 10. Deliverable table

| Arch | R×C | S | CLB | DSP | DPE (wc) | BRAM | Fmax (MHz) | Cycles | Latency (µs) | Throughput | Energy (pJ) | pJ/element |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

6 rows (3 archs × 2 sequence lengths), each Fmax averaged over 3 seeds. Caption
must state that NL outputs are log-domain and AL outputs are linear.

---

## 11. Risks and open items

- **BRAM inference**: with double buffering each lane holds 4 banks (2 score,
  2 exp) of S entries. At S=256 that is 16 lanes × 4 × 256 entries. If Parmys
  flattens them into registers instead of inferring memory, CLB counts explode.
  Mitigation: check `memory` counts at S=128 before running S=256.
- **LUT ROMs escaping to BRAM**: the AL exp/recip ROMs are meant to be CLB
  logic. Verify the `memory` count matches buffer count only.
- **Log DPE occupancy**: 10 cycles per row-group against an exp stage of 26–52,
  so it has ≥2.6× headroom and is not the limiter. The TB should still report
  its occupancy so the claim is measured rather than assumed.
- **Port-time convention**: charging port cycles on elements moved (not on C)
  is what makes P1 and P2 cycle-identical at S=128 and what reduces the log
  pass to 20 cycles. It is the physically honest reading, but it diverges from
  `imc_core.dimm_nonlinear`. If a reviewer or a later study assumes the
  simulator's convention, the numbers will not line up — state it explicitly in
  `SOFTMAX_STUDY.md`.
- **Port width is the real lever**: the entire NL/AL cycle gap is
  `16/(5n)`, set by `dpe_buf_width = 40`. Note in the discussion that a wider
  DPE port — not a wider crossbar — is what would close it; not measured here,
  since no DSE area model covers that block.
- **VTR runtime**: these designs are far larger than the Task #89 smoke points
  (~3 s). Run S=128 end-to-end before launching the full sweep.
- **`α` activity factor** for LUT energy is an assumption, not a measurement.
  Reported explicitly rather than buried.
- **P2 slower than P1** is a prediction, not a result. If it holds, it is a
  finding about DIMM-mode port bandwidth, and worth stating in the paper as
  such.
