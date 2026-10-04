# V1 → V2 (NIFA → NIFA++) comparison — simulated cycles

**Scope**: DPE primitive, GEMM, softmax. DIMM parked (v1 has no runnable DIMM RTL; see
repo history). **Basis**: single DPE tile geometry **256×256**; all values are
**simulated cycles**, NIFA = v1 (`rtl_flow/`, NL behavior/faithful RTL), NIFA++ = v2
(`v2/`, clean-room RTL). Every number traces to a file in `logs/` (see *Evidence*).

**Method**

- v1 cycles are **rerun** on the frozen v1 RTL (`rtl_flow/`), not recomputed from formulas.
- v2 cycles are read from GATE-1-certified `case.json` (sim ≡ oracle), with GATE-2
  RTL runs used to confirm `Δ_impl = 0` where the case was freshly generated.
- One repair was required to rerun v1 softmax: `softmax_study/run_softmax_smoke.py`
  pointed at a deleted model path (`fc_verification/rtl/dpe_nldpe.v`); it now points at
  `rtl_flow/rtl/dpe_nldpe.v`. No RTL was edited.
- **Softmax uses a compute-only basis**: NIFA++'s trailing `S²` output drain is excluded
  (it is reported separately); the score input load is excluded on both sides.
- `*` is not used below — all values are final.

Legend: **NIFA** = v1, **NIFA++** = v2. Cycles only; no area/energy in this table.

---

## 1. DPE primitive — tile 256×256, REGULAR mode

| Shape / cycles | NIFA (256×256) | NIFA++ (256×256) |
|---|---|---|
| M = 1 | 116 | 114 |
| M = 2 | 168 | 174 |
| M = 4 | 272 | 294 |

**Stage-level cycle breakdown (one pass, tile 256×256)**

Both designs use the same three per-pass stages with identical rates:

| Stage / term | NIFA | NIFA++ | difference |
|---|---|---|---|
| LOAD — fill R=256 activations @ 5 bytes/clk | 52 | 52 | 0 |
| COMPUTE — P = 8 slices + accumulator write + ACAM | 10 | 10 | 0 |
| OUTPUT — drain C=256 results @ 5 bytes/clk | 52 | 52 | 0 |
| FSM handoff registers (fill only) | +2 | 0 | −2 |
| **T_fill** | **116** | **114** | **−2** |
| steady — input channel | 52 (`max(L,C,O)`) | 60 (`LOAD+P`) | +8 |
| steady — output channel | 52 | 53 (`OUTPUT+1`) | +1 |
| **T_steady** | **52** | **60** | **+8** |

**Cycle difference formula.** With `T(M) = T_fill + (M−1)·T_steady`:

```
D(M) = T_NIFA++(M) − T_NIFA(M)
     = (114 − 116) + (60 − 52)·(M−1)
     = −2 + 8·(M−1)
```

The `−2` is a one-time fill advantage (NIFA++ pays no FSM handoff); the `+8` repeats
per extra M (NIFA++'s single in-place input buffer forces `LOAD+P = 60`). Crossover at
`M = 1.25`, i.e. NIFA++ is faster only at M=1.

**Major changes from V1 to V2**

*Assumption changes*
- Steady-state law: NIFA `T_steady = max(L,C,O)` = **52** → NIFA++ `max(L+P, C, O+1)` = **60** @256×256.
- Cause: NIFA++ has a single in-place input buffer, so the next pass must pipeline the `P = 8`
  slice fires; output pays one extra drain beat (`O+1`).
- Fill: NIFA RTL pays `+2` NBA handoffs (`T_fill = 116`) → NIFA++ `T_fill = 114`; v2 moves
  cost from fill into steady state.
- Net: NIFA++ is 2 cycles faster at M=1 but slower for M≥2 (M=4: 272→294, +22).
- NIFA was internally inconsistent: the lazy smoke used `T_steady = 52`, the faithful TB
  used the retired Option-A1 `max(L+C+P,…) = 60`.

*Implementation changes*
- NIFA: procedural LOAD/COMPUTE/OUTPUT FSM + 4-slot result ring; faithful variant
  double-buffers input bit-slices. Weights enter via TB backdoor (no weight port).
- NIFA: ACAM `mac²` term can silently wrap int32; latent output-buffer overwrite hazard
  (no compute↔drain gate; would corrupt at 256×512).
- NIFA++: structural/event-driven control channels (no cycle counters); `COMPUTE_CYC = P+2`
  emerges structurally; real weight ports; output-buffer hazard gated.
- NIFA++: int32 `y` + drained byte stream dual-compared against an exact integer oracle
  (GATE 1) and the RTL (GATE 2), `Δ_impl = 0` over 97 cases.

*Evidence*: `logs/v1_dpe_primitive_msweep.log`; `v2/smoke/stimuli/*_256x256_M{1,2,4}_m0/case.json`.

---

## 2. GEMM — tile 256×256 (baseline M=2, N=512; M sweep below)

| Shape / cycles | NIFA (256×256) | NIFA++ (256×256) |
|---|---|---|
| K = 512 (V=2, H=2) | 174 | 176 |
| K = 1024 (V=4, H=2) | 175 | 177 |

**Stage-level cycle breakdown (M=2, N=512, tile 256×256)**

| Term | NIFA | NIFA++ |
|---|---|---|
| tile LOAD / COMPUTE / OUTPUT | 52 / 10 / 52 | 52 / 10 / 52 |
| reduction tree `⌈log₂V⌉` | +1 (V=2) / +2 (V=4) | +1 / +2 (inside `L_w`) |
| serializer register (+1) | +1 | +1 |
| wrapper registers | +6 | 0 |
| **fill** | **122** (V=2) / **123** (V=4) | **116** (V=2) / **117** (V=4) |
| **steady** | **52** | **60** |

**M sweep (cycles; tile 256×256, class `random`, v2 `Δ_impl = 0`)**

| Shape | M | NIFA | NIFA++ | D = NIFA++ − NIFA |
|---|---|---|---|---|
| K=512, N=512 (V=2, H=2) | 128 | 6726 | 7736 | **+1010** |
| K=512, N=512 (V=2, H=2) | 256 | 13382 | 15416 | **+2034** |
| K=512, N=512 (V=2, H=2) | 512 | 26694 | 30776 | **+4082** |
| K=1024, N=1024 (V=4, H=4) | 128 | 6727 | 7737 | **+1010** |
| K=1024, N=1024 (V=4, H=4) | 256 | 13383 | 15417 | **+2034** |
| K=1024, N=1024 (V=4, H=4) | 512 | 26695 | 30777 | **+4082** |

**Cycle difference formula.** With `T(M) = fill + (M−1)·steady`:

```
D(M) = T_NIFA++(M) − T_NIFA(M)
     = (114 + L_w) − (114 + tree + clb + 6)  +  (60 − 52)·(M−1)
     =            −6                         +        8·(M−1)
```

- `tree` cancels (both pay it), so `D` is **shape-independent** for V>1 (hence the same
  D for both shapes here); only the absolute totals shift by +1 (V=4 vs V=2).
- The gap grows linearly with M: NIFA++ wins only at M=1 (`D = −6`), then loses —
  +2 (M=2), +50 (M=8), +82 (M=12), and the same line predicts **+8178** at M=1024
  (= BERT-Tiny `seq_len`, not measured).
- Note: the v1 GEMM fill base is 114, not 116 — `run_fc_smoke.py` sets `DELTA_NBA = 0`
  because the primitive's +2 registered handoff is absorbed by the wrapper's BRAM-read
  pipeline; `fc_top` instantiates the same `dpe` primitive.

**Major changes from V1 to V2**

*Assumption changes*
- Fill/wrapper: NIFA `T_fill = L+C+O` **+ wrapper delta** (`+6` registers `+TREE_PIPE`) →
  NIFA++ `T_fill = L+C+O+(TREE_PIPE+1)` (no `+6`). NIFA observed delta: `+8` (V=2), `+9` (V=4).
- Steady state is inherited from the primitive: NIFA 52 → NIFA++ 60.
- NIFA had **no 1D case (V>1, H>1)** and never exercised V=4; both shapes here are new v1
  ground (the v1 TB derives V, H generically, so it still runs).
- Reduction semantics: NIFA++ `out8 = trunc8(Σ_v y_v)` exactly, no ACAM after reduction
  (v0.3 theorem); NIFA applies its per-arch ACAM config and emits the low byte of int32.
- Net: near-parity — NIFA++ is `+2` cycles in both rows; v1's `+6` wrapper is offset by
  v2's `+8` steady-state term.

*Implementation changes*
- NIFA: Path-A weight-stationary `V×H` array; weights forced via TB backdoor; 6 wrapper
  registers; per-case `delta` decomposition is **reported, not gated**.
- NIFA functional truth = **one-byte pattern** (`tb_fc.v`: all-ones X/W → `K & 0xFF`); a real
  datapath error can pass undetected.
- NIFA++: weight-demux walker + combinational tile workload, registered byte-reduction tree
  (`TREE_PIPE`), lane serializer.
- NIFA++: GATE-1 oracle + independent NumPy witness (`(X@W) & 0xFF`); GATE-2 gates wide `S_col`
  + lane bytes bit-exact and `Δ_impl = 0`.

*Evidence*: `logs/v1_gemm_K512_N512.log`, `logs/v1_gemm_K1024_N512.log`;
`logs/v2_gemm_512x512_M2_gate2.log`, `v2/smoke/gemm_stimuli/*_256x256_{512,1024}x512_M2/case.json`.
The `512x512_M2` v2 case was generated for this comparison (was absent from the corpus).
M sweep: `logs/v1_gemm_K512_N512_M{128,256,512}.log`,
`logs/v1_gemm_K1024_N1024_M{128,256,512}.log`,
`logs/v2_gemm_{K512_N512,K1024_N1024}_M128-512_gate2.log`
(harness ran with `-DMAXM_TB=512`; v1 `vvp` timeout raised 180 → 7200 s for M≥512).

---

## 3. Softmax — NL, crossbar 256, S = 128 / 256 (compute cycles)

| Shape / cycles | NIFA (256×256) | NIFA++ (256×256) |
|---|---|---|
| S = 128, n_exp = 1, n_log = 1 | 290 | 4008 |
| S = 256, n_exp = 1, n_log = 1 | 956 | 15556 |

Reported value = **cycles(compute)** for both columns.

**Basis / cycle semantics**
- Cycles are measured as RTL `start` → `done` (`t_done − t_start`); the TB computes no
  formula — the number is defined by when the RTL asserts `done`.
- NIFA `done` = normalize stage complete (`d_done_c == RPL`); its outputs sit in `out_m`
  memory and are read back out-of-band → **compute only, no drain stage**.
- NIFA++ as-built asserts `done` at the last of the `S²` drained output words, so its RTL
  `measured` = compute + drain. This table instead quotes `case.json.compute_cycles`
  (last output value ready); the drain (`S²` = 16384 / 65536) is excluded and reported
  separately.
- Input (score load) is **excluded on both sides** (scores preloaded before `start`).
- Resource basis differs and is documented, not normalized: NIFA = **17 units** (16 exp
  DPEs + 1 shared log DPE); NIFA++ = `n_exp + n_log` = **2 units** at the quoted `n_exp=1`
  rows.

**Hardware mapping and datapaths**

A *lane* is one complete copy of the per-row pipeline. NIFA has 16 lanes; lane k owns rows
`k, k+16, k+32, …` (S/16 rows per lane). NIFA++ has no lanes: the S×S matrix is flattened
row-major and split into windows of `I = min(R,C) = 256` elements.

NIFA (16 lanes; 16 exp grids E×E with E = S; 1 shared 16×16 log grid):

```
 S x S score matrix
 row 0,16,32,... -> lane 0     ...     row 15,31,... -> lane 15

   lane 0                    lane 1                   ...  lane 15
 +-----------------+      +-----------------+           +-----------------+
 | score memory    |      | score memory    |           | score memory    |
 +--------+--------+      +--------+--------+           +--------+--------+
          |                        |                             |
   (A) 16 bytes/clk          (A) 16 bytes/clk             (A) 16 bytes/clk
   16-input max tree         16-input max tree            16-input max tree
          |                        |                             |
   (B) subtract max          (B) subtract max             (B) subtract max
       clamp to -128             clamp to -128                clamp to -128
          |                        |                             |
   exp grid E x E            exp grid E x E               exp grid E x E
   (E = S, 5 bytes/clk)      (5 bytes/clk)                (5 bytes/clk)
          |                        |                             |
   adder tree on output      adder tree on output         adder tree on output
   -> row sum                -> row sum                   -> row sum
          |                        |                             |
      lq_a[0]                  lq_a[1]                       lq_a[15]
          \________________________|_______________________________/
                                   |
                   ONE shared log grid 16 x 16   (ACAM = LOG)
                   one use = load 4 + compute 10 + output 4 = 20 clocks
                                   |
                   log_sum[0..15]  sent back to every lane
                                   |
   (D) per lane: out = clamp(score - max - log_sum)  -> out memory
```

NIFA++ (1 exp grid 256×256 + 1 log grid 256×256; no lanes):

```
 S x S score matrix
        |
   (A) streaming row-max fold, 32 bytes/clk  ---> row_max[0..S-1]
        |
   exp_input = max(score - row_max, -128)     (fused in the feed path)
        |
   flatten row-major -> S*S element stream
        |
   split into windows of I = 256 elements (window j = elements 256j .. 256j+255)
        |
 +================ softmax_exp_feed: one grid 256x256, ACAM = EXP ==============+
 |  one window = one pass:  load 256 (52) + compute (10) + output 256 (52)      |
 |  passes = ceil(S*S / 256) = 64 (S=128)  or  256 (S=256)                      |
 +==============================================================================+
        |
   exp outputs -> per-row adder tree -> sum[0..S-1]
        |
   lq[r] = min(sum[r] >> log2(S), 127)
        |
   S lq values -> windows of 256 -> exactly 1 window (because S <= 256)
        |
 +================ softmax_log_unit: one grid 256x256, ACAM = LOG ==============+
 |  one pass:  load 256 (52) + compute (10) + output 256 (52) = 114 clocks      |
 +==============================================================================+
        |
   (D) out = clamp(score - row_max - log)   (combinational, 2-cycle latency)
        |
   1 byte/clk output stream, S*S values
```

**Stage-level cycle comparison (compute only; output/drain excluded on both sides)**

NIFA's parallelism is 16 lanes, so it replicates the per-row hardware 16×
(16 max trees, 16 exp grids, 16 adder trees, 16 normalize units); the log unit
is shared (1). NIFA++ has one of each.

| Stage | Work | NIFA hardware | NIFA rate | NIFA cycles (128 / 256) | NIFA++ hardware | NIFA++ rate | NIFA++ cycles | NIFA / NIFA++ |
|---|---|---|---|---|---|---|---|---|
| A — row max | S² scores | 16 trees (one/lane) | 16 × 16 B/clk = **256 B/clk** | 64 / 256 | 1 tree | **32 B/clk** | 512 / 2048 | 0.13 → **NIFA 8×** |
| B — exp | S² elements | 16 grids (one/lane) | 16 × 5 B/clk = **80 B/clk** | 208 / 832 | 1 grid, 256 elem/pass | **4.27 elem/clk** | 3894 / 15414 | 0.053 → **NIFA 18.8×** |
| C1 — row sum | S² exp bytes | 16 adder trees (one/lane), fused to exp drain | 16× adder | 0 extra (hidden) | 1 adder tree, fused to exp drain | 1× adder | 0 extra (hidden) | 16× HW, both hidden |
| C2 — log | S sums | 1 **shared** 16×16 grid, 16 sums/use, 20 clk/use | 0.8 sums/clk | 160 / 320 busy → **0 on critical path** | 1 pass of 256×256 grid | 256 elems/114 clk | **114 (serial, on critical path)** | NIFA++ faster in raw cycles (1.4× / 2.8×), but hidden vs serial |
| D — normalize | S² elements | 16 trees (one/lane) | 256 B/clk | 64 / 256 (hidden) | combinational clamp | follows feed | ~2 clk latency (hidden) | not comparable |
| ~~Output~~ | ~~S²~~ | **excluded** | — | **0** | **excluded** | — | **0** | — |
| **Compute total** | | | | **290 / 956** | | | **4008 / 15556** | **13.8× / 16.3×** |

Why the ratios are not simply 16×:

- Row max = 8×: NIFA++'s one tree is already 32 B/clk; NIFA's per-lane tree is
  16 B/clk × 16 lanes = 256 B/clk.
- Exp = 18.8×: NIFA's per-grid rate is 5 B/clk; NIFA++'s effective per-grid rate
  is 4.27 elem/clk (one 256-element pass every 60 clocks, plus the 114-clock
  first pass).
- Row sum is hidden in both (fused to the exp drain); the difference is hardware
  count (16 trees vs 1).
- Log: NIFA's shared 16-wide grid is hidden (20 clk/use < exp's 26/52 per row);
  NIFA++'s single 256-wide pass is serial (+114).
- Normalize is hidden behind exp in both.

Totals (compute only): NIFA 290 / 956 (set by exp plus pipeline fill);
NIFA++ 4008 / 15556 (exp passes 3894 / 15414 + the serial log pass 114).

**Major changes from V1 to V2**

*Assumption / measurement-basis changes*
- Latency boundary differs: NIFA ends at compute (output memory-mapped); NIFA++ as-built
  ends at the last drained word, i.e. it includes an `S²` output drain. Compared here on a
  common **compute-only** basis.
- Score input load is out of scope for both.
- Resource quoting differs: NIFA is a 17-unit machine, NIFA++ rows are 2-unit (`n_exp=1`);
  raw cycles are therefore not resource-normalized.
- `n_exp` semantics differ: NIFA = exp DPEs per lane (fixed by S/C); NIFA++ = a free
  crossbar count (1 / 2 / 4 in the corpus).
- Domain: NIFA-NL is log-domain but NIFA-AL is linear (not mutually comparable); NIFA++ is
  log-domain only.

*Implementation changes*
- NIFA: W = 16 lockstep lanes, lane k owns rows `{k + 16i}`; per row A(max, CLB) →
  B(exp, `N_EXP` DPE(I|exp)) → Cs(log, one shared DPE) → D(norm, CLB); outputs stored to
  `out_m`.
- NIFA++: one 7-module block (`wprog` / `max_unit` / `exp_feed` / `sum_unit` / `log_unit` /
  `out_unit` / `top`); `exp_input` fused into the feed path; streaming row-max fold
  (`CLB_WIDTH = 32`); sum/lq combine pipeline; `out_unit` forms the final
  `clamp(score − row_max − log)` and merges all `n_log` drains; 1 value/cycle row-major
  output stream.
- The v1 softmax smoke was **not reproducible as-is** (dead model path); repaired with a
  one-line path fix for this rerun. NIFA value truth is the behavior-model ACAM
  approximation (no independent oracle); NIFA++ is GATE-1 + GATE-2 certified (67/67, Δ=0).

*Evidence*: `logs/v1_softmax_nl.log` (`softmax_study/results/smoke.json`);
`v2/smoke/softmax_stimuli/random_S{128,256}_nX{1,2}_nL1/case.json` and frozen pins in
`v2/sim/simulator/kernels/softmax_sim.py`.

---

## Appendix — v2 test matrix (sizes actually exercised)

| Block | Corpus | Sizes / axes | Status |
|---|---|---|---|
| DPE primitive | 97 cases | R×C ∈ {256×256, 256×512} (+8×8); M ∈ {1,2,4,8}; modes 0–3 (REGULAR/ACTIVATION/EXP/LOG); classes identity/random/extremes | 97/97 PASS, Δ=0 |
| GEMM | 259 cases (251 + 8 generated) | R×C ∈ {8×8, 40×40, 128×64, 256×256, 256×512, 512×128, 1024×64}; K ∈ {9,17,128,400,512,1024,2048,2050}; N ∈ {8,11,64,120,128,256,512,1024}; M ∈ {1,2,4,6,8,10,12} + {128,256,512} (comparison sweep); stages 1A–1D | 251/251 PASS (corpus), Δ=0; comparison cases (M2 + M-sweep) GATE-2 PASS |
| Softmax | 67 cases | S ∈ {128,256}; classes ×5; n_exp ∈ {1,2,4}; n_log ∈ {1,2}; R=C=256. X2: R=C ∈ {64,128}, n_exp=2, n_log ∈ {1,2,4} | 67/67 PASS, Δ=0 |
| DIMM | 70 cases | 7 shapes (M×N×K) × n_E ∈ {1,2,4,8,16} × {random,extremes} | 70/70 PASS, Δ=0 (parked for this report) |

Frozen pins (softmax, `used_cycles`/`compute_cycles`, `(S, n_exp, n_log)`):
`(128,1,1) 20367/4008`, `(128,4,2) 17511/1152`, `(256,1,1) 81041/15556`,
`(256,2,2) 73369/7884`. Golden corpus hashes in `v2/smoke/golden_manifest.txt`.

---

## Follow-up (not in this comparison)

- **Softmax RTL `done`/drain boundary.** Decouple the RTL `done` from the `S²` output
  drain: assert it at last-value-ready, keep the output stream for the oracle capture, and
  report `drain_cycles` separately (DIMM `serialize_cycles`-style). Requires a
  `v2/spec/softmax.md` update and re-certification of the 67-case corpus (frozen pins
  change). Until then this comparison uses `compute_cycles`.
