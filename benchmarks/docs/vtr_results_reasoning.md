# Why the VTR Results Make Sense

## 1. What determines the FPGA grid size?

VTR's `auto_layout` sizes the grid to fit the **most-demanded resource**. The grid
has a fixed ratio of DPE : DSP : BRAM : CLB tiles baked into the architecture XML.
VTR grows the grid until every resource type has enough tiles.

The **bottleneck resource** — the one that reaches ~90-100% utilization — determines
the grid size. All other resources will be under-utilized (the grid has "leftover"
tiles of those types).

## 2. NL-DPE designs are DPE-bounded

For Proposed (1024x128) and AL-Like (1024x256), the DPE utilization is 85-99%
across all sequence lengths. This means **DPE tiles are the bottleneck resource**
that forces the grid to grow.

**Why?** The DIMM attention stages (QK^T, softmax, Score x V) each require
`h_dimm = ceil(S / C)` DPE instances per stage. With 4 stages x 2 heads x 2 blocks,
the total DIMM DPEs = `16 x ceil(S/C)`. As S grows, more DPEs are needed:

| S    | Proposed DPE (C=128) | AL-Like DPE (C=256) |
|------|---------------------|---------------------|
| 128  | 18 + 16x1 = 34      | 14 + 16x1 = 30      |
| 512  | 18 + 16x4 = 82      | 14 + 16x2 = 46      |
| 1024 | 18 + 16x8 = 146     | 14 + 16x4 = 78      |
| 4096 | 18 + 16x32 = 530    | 14 + 16x16 = 270    |

Each DPE tile occupies 3x7 = 21 grid cells (Proposed) or 5x8 = 40 grid cells
(AL-Like). These are large tiles. The grid grows primarily to fit them.

**Evidence**: DPE utilization is 85-99%, while CLB is 1-23% and BRAM is 22-91%.
The grid is sized for DPEs; CLB and BRAM are side effects.

## 3. Azure-Lily designs are DSP-bounded

Azure-Lily's DPE count is fixed at 18 (only projections + FFN, no DIMM DPEs).
Instead, the DIMM stages use DSP multiply-accumulate units. The DSP count scales
with S using the same formula: `16 x ceil(S/C)` DSP instances, plus 10 for LayerNorm.

| S    | Azure-Lily DSP     |
|------|--------------------|
| 128  | 10 + 16x1 = 26     |
| 512  | 10 + 16x4 = 74     |
| 1024 | 10 + 16x8 = 138    |
| 4096 | 10 + 16x32 = 522   |

VTR reports ~70% of the expected DSP count (parmys merges some identical `*`
operators). The DSP tile is small (1x4 = 4 grid cells), so Azure-Lily grids are
much smaller than NL-DPE for the same S.

**Evidence**: DSP utilization is 90-100%, while DPE is 7-67%. The grid is sized
for DSPs; DPEs have excess capacity.

## 4. Both architectures use the same parallelism formula

This is the key design point. Both architectures implement the same DIMM
parallelism:

```
DIMM_compute_units = 4 stages x ceil(S/128) x 2 heads x 2 blocks = 16 x ceil(S/128)
```

The only difference is **what** each compute unit is:
- NL-DPE: one DPE tile (3x7 = 21 cells, analog crossbar + ACAM)
- Azure-Lily: one DSP tile (1x4 = 4 cells, digital multiply-accumulate)

The DPE tile is 5.25x larger per unit. This is why NL-DPE grids are 2-3x larger.

## 5. BRAM scales linearly with S

The BRAM usage comes from intermediate buffers between pipeline stages:

- **K buffer** (all key vectors): depth = S x d_head / 5 (packed int8 in 40-bit words)
- **V buffer** (all value vectors): same as K
- **Row buffers** (score, softmax, attention rows): depth = S / 5 each

The K/V buffers dominate and scale as O(S). There are 4 copies (2 heads x 2 blocks),
each placed as a top-level SRAM with a unique depth to prevent parmys deduplication.

The row buffers are O(S) each but small (one row of S scores, not the full S x S
matrix). The RTL processes the attention matrix **row-by-row** (streaming), so only
one row needs buffering at a time. The full S x S matrix is never stored.

**Evidence**: BRAM grows roughly linearly: 150 -> 158 -> 174 -> 206 -> 270 -> 388
for Proposed across S = 128 to 4096 (roughly 2.6x for 32x increase in S, sub-linear
because fixed-size SRAMs like projections (512 deep) dominate at small S).

## 6. Fmax is flat at ~133 MHz because of LayerNorm

**All 18 designs share the same critical path:**

```
Register -> DSP multiply (2.14 ns) -> CLB adder carry chain (~3.5 ns) -> Register
Total: ~7.5 ns -> ~133 MHz
```

This path is inside the **LayerNorm module**, which computes:

```
y_i = (x_i - mean) * rsqrt(variance)
```

The variance step does: `accumulator += diff * diff` (DSP multiply -> CLB add).
The normalize step does: `output = diff * rsqrt_val` (DSP multiply -> CLB add).

Both steps have a wide DSP product (27x27 = 54-bit) feeding a CLB carry-chain
adder in a single clock cycle. This DSP-to-CLB path is the longest combinational
path in every design.

**Why is this the same for all architectures?**

LayerNorm is the ONLY module that uses DSP multiplies in all three architectures:

| Module              | Proposed (NL-DPE)     | Azure-Lily           |
|---------------------|-----------------------|----------------------|
| Projections / FFN   | DPE crossbar (analog) | DPE crossbar (analog)|
| DIMM (QK^T, S x V) | DPE + ACAM (analog)   | DSP MAC (digital)    |
| Softmax             | DPE + ACAM (analog)   | CLB LUT              |
| **LayerNorm**       | **DSP multiply**      | **DSP multiply**     |
| Residual / Embed    | CLB add               | CLB add              |

The DPE crossbar and ACAM are hard blocks — VTR treats them as black boxes with
fixed internal delay (2.14 ns). They don't create cross-fabric routing bottlenecks.

The DSP MAC in Azure-Lily's DIMM also doesn't create the critical path because
its accumulation is internal to the DSP block.

Only LayerNorm routes a DSP multiply output through CLB fabric for accumulation,
creating the long DSP -> adder -> register path.

**Result**: All designs have the same ~133 MHz Fmax because they all use the same
LayerNorm module with the same DSP multiply -> CLB adder critical path.

## 7. Why this makes the comparison fair

Since Fmax is identical (~133 MHz) and determined by a **shared module** (LayerNorm),
the performance differences between NL-DPE and Azure-Lily come entirely from:

1. **Area**: How many grid cells each architecture needs (DPE-bounded vs DSP-bounded)
2. **Energy**: How much energy each DIMM compute unit consumes per operation
3. **Latency**: How fast the DIMM pipeline processes (DPE bandwidth + parallelism)

The Fmax does NOT favor either architecture. It's a level playing field:
- Same clock speed for all stages
- The throughput bottleneck is DIMM (O(S^2)), not LayerNorm
- DIMM runs at the same clock but with different compute units (DPE vs DSP)

This isolates the comparison to the architectural question the paper asks:
**Is analog IMC (DPE + ACAM) more efficient than digital (DSP) for attention DIMM?**

Answer from the data:
- **Energy**: Yes, 1.7x less energy (analog crossbar + ACAM vs digital DSP)
- **Latency**: Yes, 1.5-9x faster (wider DPE bus + more parallelism + pipeline overlap)
- **Area**: No, 2-4x larger (DPE tile is 5.25x bigger than DSP tile)

The tradeoff: NL-DPE wins energy and latency, loses area.
