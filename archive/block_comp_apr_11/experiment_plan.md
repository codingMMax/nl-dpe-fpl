# DPE Block Comparison — Isolation Experiment Plan

## Objective

Isolate the individual contributions of three architectural differences between
Azure-Lily and NL-DPE by changing **one variable at a time**:

1. **Interface width**: BRAM-to-DPE data bus (16-bit vs 40-bit)
2. **Conversion type**: ADC (digital) vs ACAM (analog nonlinear)
3. **Crossbar size**: 512x128 vs 1024x128

## Experiment Setups

| Setup | Crossbar | Interface | Conversion | Description |
|-------|----------|-----------|------------|-------------|
| **0** | 512x128 | 16-bit | ADC | Azure-Lily baseline |
| **1** | 512x128 | 40-bit | ADC | Wider bus only |
| **2** | 512x128 | 40-bit | ACAM | + Analog nonlinear |
| **3** | 1024x128 | 16-bit | ADC | Larger crossbar, narrow bus |
| **4** | 1024x128 | 40-bit | ADC | Larger crossbar + wider bus |
| **5** | 1024x128 | 40-bit | ACAM | Full NL-DPE (Proposed-1) |

## Pairwise Comparisons

| Comparison | Variable Isolated | What it measures |
|-----------|-------------------|-----------------|
| **Setup 1 vs 0** | Interface: 16 -> 40 bit | BRAM-to-DPE bandwidth. 40-bit loads input vector 2.5x faster per VMM. Affects latency, not energy. |
| **Setup 2 vs 1** | Conversion: ADC -> ACAM | Analog nonlinear benefit. ACAM eliminates CLB activation LUT (V=1 case). Reduces CLB usage and energy. |
| **Setup 3 vs 0** | Crossbar: 512 -> 1024 rows | Larger crossbar effect with narrow bus. More compute per pass (fewer vertical tiles for large K) but 2x more data to fill input buffer through 16-bit bus. May be a net penalty. |
| **Setup 4 vs 0** | Crossbar + Interface | Combined effect of larger crossbar with wider bus. Tests whether 40-bit bus compensates for larger buffer fill. |
| **Setup 5 vs 0** | All three combined | Full NL-DPE advantage over Azure-Lily baseline. |

### Attribution Chain

```
Setup 0 ──[+bus width]──> Setup 1 ──[+ACAM]──> Setup 2
                                                  │
Azure-Lily baseline                          512x128 NL-DPE
                                                  
Setup 0 ──[+crossbar]──> Setup 3 ──[+bus width]──> Setup 4 ──[+ACAM]──> Setup 5
                                                                           │
                                                                    1024x128 NL-DPE
                                                                     (Proposed-1)
```

## Workloads

| Workload | K | N | Purpose |
|----------|---|---|---------|
| **fc_512_128** | 512 | 128 | Small FC. V=1 for both crossbar sizes. Tests single-DPE VMM — no tiling, no reduction. Isolates DPE-level differences. |
| **fc_2048_256** | 2048 | 256 | Large FC. Requires vertical tiling + CLB adder tree reduction. Tests multi-DPE scaling. |

### Tiling per Setup

| Setup | Crossbar | fc_512_128 (V x H) | fc_2048_256 (V x H) |
|-------|----------|--------------------|--------------------|
| 0, 1, 2 | 512x128 | 1 x 1 (1 DPE) | 4 x 2 (8 DPEs) |
| 3, 4, 5 | 1024x128 | 1 x 1 (1 DPE) | 2 x 2 (4 DPEs) |

For fc_2048_256: the 1024-row crossbar needs only 4 DPEs (V=2) vs 8 DPEs (V=4)
for 512-row. Fewer tiles = less reduction overhead, but each tile is physically larger.

## RTL Differences

| Feature | ADC setups (0,1,3,4) | ACAM setups (2,5) |
|---------|---------------------|-------------------|
| Activation (V=1) | CLB `activation_lut` after DPE | None (ACAM handles it inside DPE) |
| Activation (V>1) | CLB `activation_lut` after adder tree | CLB `activation_lut` after adder tree |
| DPE DATA_WIDTH | `dpe_data_width` (16 or 40) | `dpe_data_width` (40) |

## VTR Runs

| Setup | Arch XML | Unique RTL? | VTR needed? |
|-------|----------|-------------|-------------|
| 0 | azure_lily_auto.xml | Yes (adc, dw16) | Yes |
| 1 | azure_lily_auto.xml | Yes (adc, dw40) | Yes |
| 2 | azure_lily_auto.xml | Yes (acam, dw40) | Yes |
| 3 | proposed_auto.xml | Yes (adc, dw16) | Yes |
| 4 | proposed_auto.xml | Yes (adc, dw40) | Yes |
| 5 | proposed_auto.xml | Yes (acam, dw40) | Yes |

- Setups 0-2 use `azure_lily_auto.xml` (DPE tile sized for 512x128)
- Setups 3-5 use `proposed_auto.xml` (DPE tile sized for 1024x128)
- 6 setups x 2 workloads x 3 seeds = **36 VTR runs**

## Evaluation Metrics

### Per-Setup Metrics (from VTR)

| Metric | Source | What it shows |
|--------|--------|--------------|
| DPE count | VTR netlist | Number of DPE hard blocks used |
| CLB count | VTR netlist | Logic usage (includes activation LUT for ADC) |
| BRAM count | VTR netlist | Memory blocks for SRAM buffers |
| Fmax (MHz) | VTR timing | Achievable clock frequency |
| Grid size | VTR auto_layout | FPGA area (grid_w x grid_h x CLB_tile_um2) |

### Per-Setup Metrics (from Simulator)

| Metric | Source | What it shows |
|--------|--------|--------------|
| DPE input buffer fill (ns) | `_dpe_buf_fill_row(K)` | BRAM→DPE transfer time per VMM pass. Depends on dpe_buf_width and min(K,R). |
| Core VMM latency (ns) | `_core_bit_pipeline_row_latency()` | DPE compute per VMM: 8 bit-slices + ACAM/ADC + output serialization. |
| Output serialization (ns) | `ceil(C×8/dpe_bw) × t_clk` | DPE→downstream transfer. Part of core row latency. |
| Reduction latency (ns) | `clb_reduction_energy_latency()` | Registered adder tree for V>1 (log2(V) pipeline stages). |
| Write-back (ns) | `_dpe_buf_fill_row(min(N,C))` | DPE→BRAM output transfer through dpe_buf_width. |
| First output (ns) | `gemm_pipeline_profile()["fill"]` | Latency to first result: read + core + reduction + write. |
| Steady-state (ns) | `gemm_pipeline_profile()["steady"]` | Bottleneck stage per row (max of read/core/reduc/write). |
| Total inference latency (ns) | `run_gemm()` | End-to-end: fill + (M-1) × steady, pipelined across M rows. |
| Energy per inference (pJ) | IMC energy model | Crossbar + ADC/ACAM + CLB reduction + BRAM breakdown. |

### Block-Level IO Latency Breakdown (at 300 MHz reference)

The streaming pipeline per VMM pass:
```
BRAM →[dpe_buf_width]→ DPE input buffer → 8 bit-slices → ADC/ACAM
  → DPE output serialize (C cols × int8 through dpe_buf_width)
  → CLB reduction adder (registered, V>1 only)
  → CLB activation LUT (registered, ADC + V=1 only)
  → BRAM write (through dpe_buf_width)
```

**Per-component latency (ns) at 300 MHz:**

| Component | Setup 0 (16b) | Setup 1 (40b) | Setup 2 (ACAM/40b) | Notes |
|-----------|:---:|:---:|:---:|-------|
| VMM (8 bit-slices) | 8.0 | 8.0 | 8.0 | Same analog crossbar |
| Conversion (ADC/ACAM) | 139.5 | 139.5 | 8.0 | ACAM: 17× faster |
| Digital post-processing | 0.0 | 0.0 | 1.0 | ACAM threshold compare |
| Output serialize (C=128) | 213.3 | 86.7 | 86.7 | ceil(128×8/bw) cycles |
| **Core row total** | **353.9** | **227.2** | **96.7** | Excludes buffer fill |

**Per-workload pipeline profile (ns) at 300 MHz:**

```
fc_512_128 (K=512, N=128)
                  Buf Fill   Core    Reduc   Write   1st Out  Steady  Bottleneck
Setup 0 (16b)      853.3    353.9     0.0    213.3   1420.5   853.3   buf_fill
Setup 1 (40b)      343.3    227.2     0.0     86.7    657.2   343.3   buf_fill
Setup 2 (ACAM)     343.3     96.7     0.0     86.7    526.7   343.3   buf_fill
Setup 3 (1024/16)  853.3    353.9     0.0    213.3   1420.5   853.3   buf_fill (K<R: same as 0)
Setup 4 (1024/40)  343.3    227.2     0.0     86.7    657.2   343.3   buf_fill
Setup 5 (1024/ACAM)343.3     96.7     0.0     86.7    526.7   343.3   buf_fill

fc_2048_256 (K=2048, N=256)
                  Buf Fill   Core    Reduc   Write   1st Out  Steady  Bottleneck
Setup 0 (V=4)      853.3    353.9     6.7    213.3   1427.2   853.3   buf_fill
Setup 1 (V=4)      343.3    227.2     6.7     86.7    663.9   343.3   buf_fill
Setup 2 (V=4)      343.3     96.7     6.7     86.7    533.3   343.3   buf_fill
Setup 3 (V=2)     1706.7    353.9     3.3    213.3   2277.2  1706.7   buf_fill (2x rows → 2x fill!)
Setup 4 (V=2)      683.3    227.2     3.3     86.7   1000.5   683.3   buf_fill
Setup 5 (V=2)      683.3     96.7     3.3     86.7    870.0   683.3   buf_fill
```

**Key observations:**
1. **Buffer fill is always the bottleneck** — it determines steady-state throughput
2. **Bus width (16→40 bit)** gives 2.5× speedup on buffer fill (853→343 ns)
3. **ACAM vs ADC** saves 130.5 ns/pass in core latency but doesn't change the bottleneck
4. **Larger crossbar (1024 vs 512) with narrow bus is WORSE** for fc_2048_256: V halves (4→2 DPEs) but each DPE loads 2× more data through the same 16-bit bus (1706.7 vs 853.3 ns)
5. For fc_512_128, K=512≤R for both crossbar sizes, so Setups 0/3 and 1/4 are pairwise identical
6. The bus width advantage + larger crossbar (Setup 4/5) is the sweet spot: fewer DPEs AND faster fill

### Comparison Metrics (Pairwise)

| Metric | Formula | What it isolates |
|--------|---------|-----------------|
| Latency ratio | Setup_X / Setup_0 | Speedup vs baseline |
| Energy ratio | Setup_X / Setup_0 | Energy savings vs baseline |
| CLB delta | CLB_X - CLB_0 | Extra CLB from activation LUT (ADC vs ACAM) |
| DPE delta | DPE_X - DPE_0 | Tiling difference (crossbar size effect) |
| Fmax delta | Fmax_X - Fmax_0 | Critical path impact |
| IO latency breakdown | per-stage ns | Which pipeline stage dominates |

### Target Result Table

```
                      fc_512_128 (1 DPE)                            fc_2048_256 (multi-DPE)
Setup  Config          DPE  CLB  Fmax  Lat(ns) Energy(pJ)  IO_dom   DPE  CLB  Fmax  Lat(ns)  Energy(pJ)  IO_dom
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  0    512x128/adc/16    1   ?    ?     ?       ?          buf_fill   8    ?    ?     ?        ?          buf_fill
  1    512x128/adc/40    1   ?    ?     ?       ?          buf_fill   8    ?    ?     ?        ?          buf_fill
  2    512x128/acam/40   1   ?    ?     ?       ?          buf_fill   8    ?    ?     ?        ?          buf_fill
  3    1024x128/adc/16   1   ?    ?     ?       ?          buf_fill   4    ?    ?     ?        ?          buf_fill
  4    1024x128/adc/40   1   ?    ?     ?       ?          buf_fill   4    ?    ?     ?        ?          buf_fill
  5    1024x128/acam/40  1   ?    ?     ?       ?          buf_fill   4    ?    ?     ?        ?          buf_fill

Pairwise deltas (vs Setup 0):
  1 vs 0:  bus width     ->  Lat: ~0.46x (fc_512), ~0.47x (fc_2048)   CLB: +?
  2 vs 1:  ACAM          ->  Lat: ~0.80x (within core, not bottleneck)  CLB: -? (no act LUT for V=1)
  3 vs 0:  crossbar size ->  Lat: 1.0x (fc_512), ~1.60x WORSE (fc_2048, narrow bus penalty)
  4 vs 0:  xbar + bus    ->  Lat: ~0.46x (fc_512), ~0.70x (fc_2048)
  5 vs 0:  all three     ->  Lat: ~0.37x (fc_512), ~0.61x (fc_2048)
```

**Note:** Latency ratios above are from simulator at fixed 300 MHz reference clock.
Actual VTR Fmax may differ per setup, which will shift the ratios. The final
result table will use VTR-reported Fmax per setup.

## Modeling Assumptions

### Streaming Pipeline Model (RTL ↔ Simulator aligned)
Both RTL and simulator model the same streaming data path:

```
Stage 1: BRAM → DPE input buffer  (through dpe_buf_width port)
Stage 2: 8 bit-serial VMM slices  (internal to DPE)
Stage 3: ADC/ACAM conversion      (internal to DPE)
Stage 4: Output serialization     (C cols × int8 through dpe_buf_width)
Stage 5: CLB reduction adder      (registered, V>1 only, streaming)
Stage 6: CLB activation LUT       (registered, ADC + V=1 only, streaming)
Stage 7: BRAM write               (through dpe_buf_width port)
```

No intermediate BRAM between stages 4–7. Reduction and activation are pipelined
with the output stream (1 registered cycle each, overlapped with serialize).

### Parallel DPE Loading
All V×H DPEs load their input data from BRAM **in parallel**. Each DPE row has
its own SRAM instance. The wall-clock buffer fill time = one DPE's fill time,
regardless of V or H.

RTL: for V>1, each vertical DPE row has its own SRAM + controller_scalable(N_DPE_V=1).
Simulator: `_dpe_buf_fill_row(K)` loads `min(K, R)` values per DPE.

### Buffer Fill is One-Time Per VMM
Data path: BRAM →[dpe_buf_width]→ DPE Internal Buffer →[internal]→ Crossbar.
The full input vector (up to R int8 values) is loaded ONCE through the
dpe_buf_width interface. Then 8 bit-slices fire internally from the buffer —
no further BRAM access. Zero-skip: when K < R, only K values are loaded.

### DPE Read and Write Use dpe_buf_width
Both BRAM→DPE (read) and DPE→BRAM (write) go through the dpe_buf_width
interface (16-bit for setups 0/3, 40-bit for setups 1/2/4/5). General CLB/DSP
access uses bram_width=40 for all setups.

### Tiling Latency
With parallel DPEs, the tiling (V>1) adds latency only through the **adder tree
reduction**: log2(V) registered stages per output column. The buffer fill and
crossbar compute are parallel across all V DPEs — no sequential penalty.

### Activation Latency
For V=1 ADC setups, the CLB activation LUT is a single registered pipeline
stage after the DPE output. Its latency (1 cycle) is absorbed into the
streaming pipeline — it doesn't add to the critical path since
t_output_serialize >> 1 cycle. Energy is still counted.

For ACAM setups (V=1), the DPE handles activation internally (analog
nonlinear). No CLB activation is needed.

### Fmax Sensitivity
All latency numbers above assume a 300 MHz reference clock. VTR may report
different Fmax per setup (the critical path depends on the generated RTL
complexity and routing). The final metrics use per-setup VTR Fmax — this
captures the **physical cost** of wider buses and larger crossbars on
achievable clock frequency.

## Experiment Results

### VTR + IMC Results Summary

All values from `block_comparison_results.csv`. VTR: 3 seeds per point, Fmax averaged.
Read = SRAM→DPE transfer via `_dpe_buf_fill_row(K)` using `dpe_buf_width`.

```
                                fc_512_128 (V=1, 1 DPE)                      fc_2048_256 (multi-DPE)
Setup  Config          Fmax   Lat(ns)  read   core_row  Energy(pJ)          Fmax   Lat(ns)  read     core_row  Energy(pJ)  V×H  DPEs CLB
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  0    512/adc/16     311.1  1375.0   823.0   346.3     2593.9              306.4  1400.4    835.6   349.4     20460.5     4×2   8   38
  1    512/adc/40     281.6   690.9   365.7   232.8     2609.0              307.3   651.5    335.2   225.1     20457.1     4×2   8   40
  2    512/acam/40    355.5   446.0   289.8    83.1      132.0              307.3   521.0    335.2    94.6      1365.8     4×2   8   40
  3    1024/adc/16    310.2  1378.5   825.3   346.8     2738.8              328.3  2092.8   1559.3   335.4     10795.0     2×2   4   22
  4    1024/adc/40    281.6   690.9   365.7   232.8     2768.0              338.0   903.9    606.5   217.4     10763.9     2×2   4   20
  5    1024/acam/40   348.6   454.6   295.4    84.6      171.8              338.0   773.3    606.5    86.9       842.2     2×2   4   20
```

### DPE Energy Breakdown (pJ)

```
                          fc_512_128                         fc_2048_256
Setup  Config         Crossbar    ADC     ACAM   DPE total   Crossbar    ADC       ACAM   DPE total
──────────────────────────────────────────────────────────────────────────────────────────────────────
  0    512/adc/16       144.0   2385.9     0.0    2529.9      1169.8   19087.4      0.0    20257.1
  1    512/adc/40       159.1   2385.9     0.0    2545.0      1166.4   19087.4      0.0    20253.8
  2    512/acam/40       63.3      0.0    62.3     125.6       585.5       0.0    576.9     1162.5
  3    1024/adc/16      288.9   2385.9     0.0    2674.8      1091.5    9543.7      0.0    10635.2
  4    1024/adc/40      318.1   2385.9     0.0    2704.1      1060.4    9543.7      0.0    10604.1
  5    1024/acam/40     101.0      0.0    64.4     165.4       416.8       0.0    265.6      682.4
```

---

### Latency Analysis: Pipeline Model

The total inference latency follows the pipelined streaming model:

```
latency = first_output + (M − 1) × steady
```

where M = number of output rows (M=1 for GEMV inference), so **latency = first_output**.

`first_output` is the sum of four pipeline stages, each running at the
VTR-reported Fmax (t_clk = 1/Fmax):

```
first_output = read + core_row + reduction + write

read       = ceil(min(K, R) × 8 / dpe_buf_width) × t_clk    ← SRAM→DPE via dpe_buf_width
core_row   = (8 bit-slice pipeline) + (ADC or ACAM) + ceil(C×8 / dpe_buf_width) × t_clk
reduction  = log2(V) × t_clk                                 ← 0 when V=1
write      = ceil(min(N, C) × 8 / dpe_buf_width) × t_clk    ← DPE→BRAM via dpe_buf_width
```

Each DPE has its own dedicated SRAM. All V×H DPEs load in parallel from their
own SRAMs. The read stage models the SRAM→DPE transfer through the
`dpe_buf_width` port. The BRAM→SRAM fill latency is accounted for in the
write stage of the previous layer (data flows in through the same port width).

The `dpe_buf_width` affects **all three IO stages**: read, output serialization
(inside core_row), and write. Only the bit-slice pipeline and ADC/ACAM
conversion are independent of bus width.

Example for fc_512_128 (min(K,R)=512 for 512-row crossbar):
- 16-bit: read = ceil(512×8/16) = 256 accesses
- 40-bit: read = ceil(512×8/40) = 103 accesses (2.49× fewer)

The steady-state bottleneck (for M > 1):

```
steady = max(read, core_row, reduction, write)
```

In all setups, **read is the bottleneck** — it is always the largest stage.

---

### Comparison 1: Setup 1 vs 0 — Bus Width (16-bit → 40-bit)

**Isolated variable**: dpe_buf_width (16 → 40). Same crossbar (512×128), same ADC.

#### fc_512_128 (V=1, 1 DPE)

| Stage | Setup 0 (16-bit, 311 MHz) | Setup 1 (40-bit, 282 MHz) | Change |
|-------|:---:|:---:|--------|
| read | 823.0 ns (256 acc × 3.21 ns) | 365.7 ns (103 acc × 3.55 ns) | **−457.3 ns** (2.25×) |
| core_row | 346.3 ns | 232.8 ns | **−113.5 ns** (output serialize: 64→26 cyc) |
| reduction | 0.0 | 0.0 | — |
| write | 205.8 ns | 92.3 ns | **−113.5 ns** (write: 64→26 cyc) |
| **latency** | **1375.0 ns** | **690.9 ns** | **−684.1 ns (0.50×)** |
| **bottleneck** | **read** (823.0 ns) | **read** (365.7 ns) | read dominates both |

With 16-bit dpe_buf_width, the DPE can only receive 2 bytes/cycle, so loading
512 int8 values requires ceil(512×8/16)=256 cycles. With 40-bit, it drops to
ceil(512×8/40)=103 cycles — a 2.49× reduction in access count. The 40-bit bus
also shrinks core_row (output serialization: 64→26 cycles) and write (64→26).

Setup 1 has lower Fmax (282 vs 311 MHz) which makes each cycle 10% slower,
partially offsetting the access count reduction. Net: **50% latency reduction**.

#### fc_2048_256 (V=4, 8 DPEs)

| Stage | Setup 0 (16-bit, 306 MHz) | Setup 1 (40-bit, 307 MHz) | Change |
|-------|:---:|:---:|--------|
| read | 835.6 ns (256 acc × 3.26 ns) | 335.2 ns (103 acc × 3.25 ns) | **−500.4 ns** (2.49×) |
| core_row | 349.4 ns | 225.1 ns | **−124.3 ns** |
| reduction | 6.5 ns | 6.5 ns | — |
| write | 208.9 ns | 84.6 ns | **−124.3 ns** |
| **latency** | **1400.4 ns** | **651.5 ns** | **−748.9 ns (0.47×)** |
| **bottleneck** | **read** (835.6 ns) | **read** (335.2 ns) | read dominates both |

Each DPE loads min(2048,512)=512 values from its dedicated SRAM through the
dpe_buf_width port. With 16-bit: 256 accesses. With 40-bit: 103 accesses.
Same 2.49× ratio as fc_512_128. **53% latency reduction**.

#### Energy

Bus width does not affect compute energy. Setup 0: 2593.9 pJ, Setup 1: 2609.0 pJ
(~identical). The small difference comes from different Fmax scaling the
crossbar energy (power/freq).

---

### Comparison 2: Setup 2 vs 1 — Conversion (ADC → ACAM)

**Isolated variable**: conversion type. Same crossbar (512×128), same 40-bit bus.

#### fc_512_128 (V=1, 1 DPE)

| Stage | Setup 1 (ADC, 282 MHz) | Setup 2 (ACAM, 355 MHz) | Change |
|-------|:---:|:---:|--------|
| read | 365.7 ns (103 acc × 3.55 ns) | 289.8 ns (103 acc × 2.81 ns) | **−75.9 ns** |
| core_row | 232.8 ns | 83.1 ns | **−149.7 ns** |
| write | 92.3 ns | 73.1 ns | −19.2 ns |
| **latency** | **690.9 ns** | **446.0 ns** | **−244.9 ns (0.65×)** |
| **Fmax** | 281.6 MHz | 355.5 MHz | **+73.9 MHz (+26%)** |
| **DPE energy** | 2545.0 pJ | 125.6 pJ | **20× reduction** |

Two effects compound:

1. **Fmax improvement (+26%)**: ACAM V=1 RTL has no CLB activation_lut on the
   critical path (ACAM handles activation inside the DPE). VTR achieves 355 vs
   282 MHz. This shrinks ALL stages proportionally — read drops from 365.7 to
   289.8 ns despite having the same data volume and bus width.

2. **Core row shortening**: ADC conversion takes 139.5 ns (at core_freq), ACAM
   takes only 8+1=9 ns. Combined with shorter output serialization cycles
   (from higher Fmax), core_row drops from 232.8 to 83.1 ns.

Decomposing the 244.9 ns improvement:
- Fmax effect (all stages scale by 282/355 = 0.79): saves ~145 ns
- ACAM conversion effect (core_row only): saves ~100 ns

#### fc_2048_256 (V=4, 8 DPEs)

| Stage | Setup 1 (ADC, 307 MHz) | Setup 2 (ACAM, 307 MHz) | Change |
|-------|:---:|:---:|--------|
| read | 335.2 ns | 335.2 ns | identical |
| core_row | 225.1 ns | 94.6 ns | **−130.5 ns** |
| write | 84.6 ns | 84.6 ns | identical |
| **latency** | **651.5 ns** | **521.0 ns** | **−130.5 ns (0.80×)** |
| **DPE energy** | 20253.8 pJ | 1162.5 pJ | **17× reduction** |

Both setups have identical Fmax (307 MHz) because the multi-DPE critical path
is the CLB adder tree, not the activation_lut. With identical Fmax and bus width,
read and write are the same. The 130.5 ns improvement comes entirely from
ACAM's shorter core_row (no ADC conversion overhead). **20% latency reduction**.

Energy: 8 passes × ADC = 19087 pJ vs 8 passes × ACAM = 577 pJ. **17× reduction**.

---

### Comparison 3: Setup 3 vs 0 — Crossbar Size (512 → 1024 rows)

**Isolated variable**: crossbar rows (512 → 1024). Same ADC, same 16-bit bus.

#### fc_512_128 (V=1, 1 DPE)

| Metric | Setup 0 (512, 311 MHz) | Setup 3 (1024, 310 MHz) | Change |
|--------|:---:|:---:|--------|
| min(K,R) | min(512,512) = 512 | min(512,1024) = 512 | **same** |
| read | 823.0 ns (256 acc) | 825.3 ns (256 acc) | ~same |
| core_row | 346.3 ns | 346.8 ns | ~same |
| latency | 1375.0 ns | 1378.5 ns | **~same (1.003×)** |
| DPE energy | 2529.9 pJ | 2674.8 pJ | +144.9 pJ (+5.7%) |

K=512 fits entirely in both crossbar sizes. Both load 512 values through
the same 16-bit DPE port → identical read (256 access cycles). Slightly more
crossbar energy (288.9 vs 144.0 pJ) from the larger array.

#### fc_2048_256 (V=4→2, 8→4 DPEs)

| Stage | Setup 0 (V=4, 306 MHz) | Setup 3 (V=2, 328 MHz) | Change |
|-------|:---:|:---:|--------|
| read | 835.6 ns (256 acc × 3.26 ns) | 1559.3 ns (512 acc × 3.05 ns) | **+723.7 ns (1.87×)** |
| core_row | 349.4 ns | 335.4 ns | −14.0 ns |
| reduction | 6.5 ns (log2(4)=2) | 3.0 ns (log2(2)=1) | −3.5 ns |
| write | 208.9 ns | 194.9 ns | −14.0 ns |
| **latency** | **1400.4 ns** | **2092.8 ns** | **+692.4 ns (1.49× worse)** |
| CLB | 38 | 22 | **−16 CLBs** |
| DPE energy | 20257.1 pJ | 10635.2 pJ | **0.53× (half the ADC passes)** |

The larger crossbar halves V (4→2), halving ADC energy (19087→9544 pJ) and
CLB usage. But each DPE now loads min(2048,1024)=1024 values through the
16-bit port: ceil(1024×8/16)=512 accesses vs Setup 0's ceil(512×8/16)=256.
The read stage nearly doubles (836→1559 ns).

**Net: 49% worse latency** — the 16-bit DPE port cannot keep up with the larger
crossbar's data appetite. Energy still improves 0.53× from fewer passes.

This is the key insight: **a larger crossbar with a narrow bus is a latency
anti-pattern**. The energy benefit (fewer passes) is real, but the latency
penalty from the longer per-DPE fill far outweighs it.

---

### Comparison 4: Setup 4 vs 0 — Crossbar + Bus Width

**Combined effect**: 1024-row crossbar + 40-bit bus (vs 512/16-bit baseline).

#### fc_512_128

| Stage | Setup 0 (512/16b, 311 MHz) | Setup 4 (1024/40b, 282 MHz) | Change |
|-------|:---:|:---:|--------|
| read | 823.0 ns (256 acc × 3.21 ns) | 365.7 ns (103 acc × 3.55 ns) | **−457.3 ns** |
| core_row | 346.3 ns | 232.8 ns | **−113.5 ns** |
| write | 205.8 ns | 92.3 ns | **−113.5 ns** |
| **latency** | **1375.0 ns** | **690.9 ns** | **−684.1 ns (0.50×)** |

K=512 fits in both crossbars → crossbar size has no effect. Same as Setup 1 vs 0:
the entire improvement comes from bus width.

#### fc_2048_256

| Stage | Setup 0 (V=4, 16b, 306 MHz) | Setup 4 (V=2, 40b, 338 MHz) | Change |
|-------|:---:|:---:|--------|
| read | 835.6 ns (256 acc × 3.26 ns) | 606.5 ns (205 acc × 2.96 ns) | **−229.1 ns** |
| core_row | 349.4 ns | 217.4 ns | **−132.0 ns** |
| reduction | 6.5 ns | 3.0 ns | −3.5 ns |
| write | 208.9 ns | 76.9 ns | **−132.0 ns** |
| **latency** | **1400.4 ns** | **903.9 ns** | **−496.5 ns (0.65×)** |
| DPE energy | 20257.1 pJ | 10604.1 pJ | **0.52×** |

Each DPE loads min(2048,1024)=1024 values. With 40-bit: ceil(1024×8/40)=205
accesses at 338 MHz. With 16-bit (Setup 0's DPEs load 512 values):
ceil(512×8/16)=256 accesses at 306 MHz. The 40-bit bus + Fmax improvement
give 229 ns savings on read alone. Combined with core_row+write: **35%
latency reduction + 48% energy reduction**.

Contrast with Comparison 3 (1024/16-bit, 1.49× worse): the 40-bit bus
completely reverses the crossbar size penalty.

---

### Comparison 5: Setup 5 vs 0 — All Three (NL-DPE vs Azure-Lily)

**Full NL-DPE**: 1024×128, 40-bit bus, ACAM vs Azure-Lily baseline.

#### fc_512_128

| Stage | Setup 0 (Azure-Lily) | Setup 5 (NL-DPE) | Change |
|-------|:---:|:---:|--------|
| read | 823.0 ns (256 acc @ 311 MHz) | 295.4 ns (103 acc @ 349 MHz) | **−527.6 ns** |
| core_row | 346.3 ns | 84.6 ns | **−261.7 ns** |
| write | 205.8 ns | 74.6 ns | **−131.2 ns** |
| **latency** | **1375.0 ns** | **454.6 ns** | **0.33× (3.0× faster)** |
| DPE energy | 2529.9 pJ | 165.4 pJ | **0.065× (15× cheaper)** |

All three factors contribute:
- **40-bit bus** → read: 256→103 acc, output serialize: 64→26 cyc, write: 64→26 cyc
- **ACAM** → no ADC conversion in core_row, plus higher Fmax (349 vs 311 MHz)
- **Fmax (+12%)** → all stages scale down proportionally

#### fc_2048_256

| Stage | Setup 0 (Azure-Lily) | Setup 5 (NL-DPE) | Change |
|-------|:---:|:---:|--------|
| read | 835.6 ns (256 acc × 3.26 ns) | 606.5 ns (205 acc × 2.96 ns) | **−229.1 ns** |
| core_row | 349.4 ns | 86.9 ns | **−262.5 ns** |
| reduction | 6.5 ns | 3.0 ns | −3.5 ns |
| write | 208.9 ns | 76.9 ns | **−132.0 ns** |
| **latency** | **1400.4 ns** | **773.3 ns** | **0.55× (1.81× faster)** |
| DPE energy | 20257.1 pJ | 682.4 pJ | **0.034× (30× cheaper)** |

All three factors contribute to the 627 ns savings:
- **40-bit bus** → read: 256→205 acc, output serialize: 64→26 cyc, write: 64→26 cyc
- **1024 crossbar** → each DPE loads min(2048,1024)=1024 vs 512, but 40-bit bus
  keeps this manageable (205 acc vs Setup 3's 512 acc at 16-bit)
- **ACAM + Fmax** → shorter core_row (87 vs 349 ns) + faster clock (338 vs 306 MHz)

---

### Summary of Contributions

**Latency** (fc_512_128 / fc_2048_256):

| Factor | Mechanism | fc_512 | fc_2048 |
|--------|-----------|:---:|:---:|
| Bus width (16→40) | 2.5× fewer SRAM→DPE accesses on all IO stages | **0.50×** | **0.47×** |
| ACAM (vs ADC) | Shorter core_row + Fmax boost (V=1 only) | 0.65× | 0.80× |
| Crossbar 1024 (16b) | More data per DPE through narrow bus | 1.00× | **1.49× worse** |
| Crossbar 1024 (40b) | Fewer passes, manageable read increase | — | 0.65× (vs 0) |
| All three (5 vs 0) | Combined | **0.33×** | **0.55×** |

The `dpe_buf_width` controls all three IO stages (read, output serialize, write).
For fc_512_128, the 40-bit bus gives a clean 2× speedup. For fc_2048_256, the
larger crossbar loads 1024 values per DPE (vs 512), so the 16-bit bus penalty
is even more severe: Setup 3 (1024/16-bit) is **49% slower** than the baseline
Setup 0 (512/16-bit). The 40-bit bus reverses this entirely.

**Energy** (fc_512_128 / fc_2048_256):

| Factor | Mechanism | fc_512 | fc_2048 |
|--------|-----------|:---:|:---:|
| Bus width (16→40) | Does not affect compute energy | 1.00× | 1.00× |
| ACAM | Eliminates ADC (94% of DPE energy) | **0.05×** | **0.07×** |
| Crossbar 1024 | Half VMM passes = half ADC cost | 1.06× | **0.53×** |
| All three (5 vs 0) | Combined | **0.07×** | **0.04×** |

**The two dominant levers target different bottlenecks**:
- **Bus width** improves latency by reducing SRAM→DPE transfer cycles on all
  three IO stages (read, output serialize, write). This is the dominant latency
  factor for both workloads (0.50× for fc_512, 0.47× for fc_2048).
- **ACAM** improves energy by eliminating ADC conversion (94% of DPE energy).
  It also provides a latency bonus through higher Fmax (simpler RTL routes
  better), but this is an indirect effect specific to V=1 designs.
- **Larger crossbar** helps energy through fewer VMM passes, but **hurts latency
  with narrow bus** because each DPE loads more data. Only beneficial when
  paired with the wider bus.

## File Structure

```
block_comp_apr_11/
├── experiment_plan.md          <- this file
├── rtl/
│   ├── setup0/                 <- 512x128, adc, dw16 (Azure-Lily baseline)
│   │   ├── fc_512_128_512x128_adc_dw16.v
│   │   └── fc_2048_256_512x128_adc_dw16.v
│   ├── setup1/                 <- 512x128, adc, dw40
│   ├── setup2/                 <- 512x128, acam, dw40
│   ├── setup3/                 <- 1024x128, adc, dw16
│   ├── setup4/                 <- 1024x128, adc, dw40
│   └── setup5/                 <- 1024x128, acam, dw40
├── vtr_runs/                   <- VTR output (to be generated)
└── results/                    <- CSVs and plots (to be generated)
```
