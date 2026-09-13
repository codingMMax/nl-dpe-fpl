# IMC_new vs Event‑Driven Simulator — Operator‑Level Study (Operator‑Only Model)

This report compares operator‑level energy/latency breakdowns.
Event‑driven latency breakdown is a proxy (non‑overlapped analytic estimate).
Event‑driven read/write latency uses per‑event SRAM model (default).

Inter-layer pipeline/parallelism assumptions (IMC_new):
- Layer order is the model order; no global reordering across different layer names.
- Layer timing profile is (first_output_ns, steady_ns, events).
- Conv/maxpool dependencies are converted to required upstream token indices R_i(n) from receptive-field geometry.
- Streaming timeline uses token recurrence: T_i(n)=max(T_i(n-1)+steady_i, T_{i-1}(R_i(n))+first_output_i).
- Effective layer latency on timeline includes dependency-wait (backpressure_add); critical path accumulates max-overlap only.
- Attention special case: linear_Q/linear_K/linear_V are grouped as one parallel block via max latency.
Intra-layer stage assumptions (IMC_new):
- Conv/linear and maxpool use first_output + (events-1)*steady pipelines.
- IMC core stage pipeline uses max(vmm, adc, digital, reduction) per output when cfg.pipelinable is true.
- Activation is practical (finite throughput): act_units parallel ops, act_cycles_per_op per op, configurable in JSON.
- Memory model assumes enough BRAM banks to sustain configured effective width (bram_width * mem_bw_utilization).
Assumptions (Event‑Driven reference):
- Energy units are nJ; constants are from nn/constant.py.
- Runtime latency is event-driven with explicit readiness/resource checks; breakdown here is analytic proxy.
- Reads/writes use ELEMS_PER_MV and NUM_* resources to compute event counts.
- Activation/reduction/maxpool are explicit per-event stages, then globally overlapped by scheduler.
Expected behavior:
- Memory‑bound layers (small M) show latency ratios driven by effective port width/bandwidth.
- Compute‑bound layers (large M) converge as steady‑state compute dominates fill cost.
Effective mem access: event‑driven uses ELEMS_PER_MV=2 and BYTES_PER_ELEM=1, bytes/access=2 with SRAM_LAT=3.3ns → BW=0.606 bytes/ns; IMC_new uses BRAM width=40b with mem_bw_utilization=0.400 → effective 2.000 bytes/access (raw 5), cycle=3.333ns → BW=0.600 bytes/ns.
Configured activation knobs: act_units=16, act_cycles_per_op=1.

Execution orchestration (IMC_new):
- Scheduler maps conv/linear to IMC core GEMM, and maxpool/activation/softmax to FPGA peripherals.
- MemoryModel provides BRAM read/write latency and energy for all mapped operators.
- Stats tracks per-layer energy and raw latency, while PipelineProfiler tracks overlap-aware critical path.
- End-to-end latency is the critical path from PipelineProfiler; raw sum is reported separately for sanity checks.

## Cross‑Layer Pipeline Strategy Guide

High-level design: each layer is reduced to a timing tuple (first_output_ns, steady_ns, events, required_upstream_outputs). The scheduler then computes a critical-path overlap timeline.

Control parameters and how they are computed:
- `first_output_ns`: time from layer start to first produced output token.
- `steady_ns`: per-token interval after pipeline fill.
- `events`: number of produced output tokens for the layer.
- `required_upstream_outputs`: conservative dependency count before downstream launch; for conv/maxpool = max(1, kernel_size - padding) * input_width * num_inputs.
- Token readiness recurrence (streaming layers):
  T_i(1) = T_{i-1}(R_i(1)) + first_output_i, T_i(n) = max(T_i(n-1) + steady_i, T_{i-1}(R_i(n)) + first_output_i).
- `R_i(n)` is upstream token index needed for downstream token `n` from window geometry.
- `backpressure_add` in tables is the accumulated dependency-wait beyond local steady cadence.

### ResNet timestamp walkthrough (conv1 → conv2 → pool1 → conv3)

| Time | Event | Formula (ns) | Timestamp (ns) |
| --- | --- | --- | --- |
| t0 | launch conv1 | 0 | 0.00 |
| t1 | conv1 first output ready | 0.00 + 292.85 | 292.85 |
| t2 | conv1 drained | 0.00 + 143021.81 | 143021.81 |
| t3 | launch conv2 | T_prev(R= 64) | 4897.01 |
| t4 | conv2 first output ready | 4897.01 + 1189.52 | 6086.53 |
| t5 | conv2 drained | 4897.01 + 860509.52 | 865406.53 |
| t6 | launch pool1 | T_prev(R= 64) | 33806.53 |
| t7 | pool1 first output ready | 33806.53 + 940.00 | 34746.53 |
| t8 | pool1 drained | 33806.53 + 832540.00 | 866346.53 |
| t9 | launch conv3 | T_prev(R= 32) | 90186.53 |
| t10 | conv3 first output ready | 90186.53 + 2029.52 | 92216.05 |
| t11 | conv3 drained | 90186.53 + 806749.52 | 896936.05 |

Per-layer pipeline terms (same units and definitions as scheduler):
| Layer | Type | Events | Required Upstream Outputs | First Output (ns) | Steady (ns) | Base Total (ns) | Backpressure Add (ns) | Effective Total (ns) | Critical Contribution (ns) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1 | conv2d | 1024 | 64 | 292.85 | 139.52 | 143021.81 | 0.00 | 143021.81 | 143021.81 |
| conv2 | conv2d | 1024 | 64 | 1189.52 | 840.00 | 860509.52 | 0.00 | 860509.52 | 722384.72 |
| pool1 | maxpool | 256 | 64 | 940.00 | 746.67 | 191340.00 | 641200.00 | 832540.00 | 940.00 |
| conv3 | conv2d | 256 | 32 | 2029.52 | 1680.00 | 430429.52 | 376320.00 | 806749.52 | 30589.52 |

Interpretation:
- Larger `required_upstream_outputs` delays downstream launch beyond previous first output.
- `Backpressure Add` appears when upstream steady-state is slower than downstream demand.
- Critical-path contribution can be smaller than layer effective total when overlap is high.

## CONV2D (layers: conv1)

## LINEAR (layers: full1)

## MAXPOOL (layers: pool1)

### Maxpool Latency Audit — pool1
Counts and scaling assumptions:
| Metric | Event‑Driven | IMC_new |
| --- | --- | --- |
| output_positions (count) | 256 | 256 |
| num_events (events) | 256 | n/a |
| window_elements (count) | 4 | 4 |

Latency contributions (ns):
| Metric | Event‑Driven (proxy) | IMC_new |
| --- | --- | --- |
| read_latency | 13516.80 | 13653.33 |
| write_latency | 13.20 | 13.33 |
| compare_latency | 6.60 | 6.67 |
| total_latency | 13536.60 | 13673.33 |
Note: read/write gaps often reflect different effective port widths (event‑driven uses ELEMS_PER_MV; IMC_new uses BRAM width).

## ResNet Layer‑Wise Latency and Energy Breakdown

Event‑Driven totals are from per‑layer event simulation. IMC_new totals are per‑layer IMC simulation.
Breakdown buckets use the same proxy model as elsewhere in this report.
Event‑Driven per-layer total latency uses proxy model for runtime efficiency.

### ResNet Layer‑Wise Latency Breakdown (ns)

| Layer | Type | Event‑Driven Total (ns) | Event‑Mem (ns) | Event‑Core (ns) | Event‑Periph (ns) | IMC_new Total (ns) | IMC‑Mem (ns) | IMC‑Core (ns) | IMC‑Periph (ns) | Ratio (new/old) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1 | conv2d | 145549.80 | 151.80 | 145305.60 | 92.40 | 143021.81 | 140.00 | 142868.48 | 13.33 | 0.98 |
| conv2 | conv2d | 852093.00 | 851743.20 | 165.00 | 184.80 | 860509.52 | 860346.67 | 139.52 | 23.33 | 1.01 |
| pool1 | maxpool | 189466.20 | 189420.00 | 0.00 | 46.20 | 191340.00 | 191333.33 | 0.00 | 6.67 | 1.01 |
| conv3 | conv2d | 426336.90 | 425964.00 | 188.10 | 184.80 | 430429.52 | 430266.67 | 139.52 | 23.33 | 1.01 |
| conv4 | conv2d | 426336.90 | 425964.00 | 188.10 | 184.80 | 430429.52 | 430266.67 | 139.52 | 23.33 | 1.01 |
| res1 | residual | 47496.90 | 47493.60 | 0.00 | 3.30 | 143376.67 | 95573.33 | 0.00 | 853.33 | 3.02 |
| conv5 | conv2d | 426752.70 | 426148.80 | 234.30 | 369.60 | 430639.52 | 430453.33 | 139.52 | 46.67 | 1.01 |
| pool2 | maxpool | 95079.60 | 94987.20 | 0.00 | 92.40 | 95953.33 | 95946.67 | 0.00 | 6.67 | 1.01 |
| conv6 | conv2d | 213909.30 | 213259.20 | 280.50 | 369.60 | 215599.52 | 215413.33 | 139.52 | 46.67 | 1.01 |
| pool3 | maxpool | 24116.40 | 24024.00 | 0.00 | 92.40 | 24273.33 | 24266.67 | 0.00 | 6.67 | 1.01 |
| conv7 | conv2d | 54242.10 | 53592.00 | 280.50 | 369.60 | 54319.52 | 54133.33 | 139.52 | 46.67 | 1.00 |
| conv8 | conv2d | 54242.10 | 53592.00 | 280.50 | 369.60 | 54319.52 | 54133.33 | 139.52 | 46.67 | 1.00 |
| res2 | residual | 6286.50 | 6283.20 | 0.00 | 3.30 | 17923.33 | 11946.67 | 0.00 | 53.33 | 2.85 |
| pool4 | maxpool | 6468.00 | 6283.20 | 0.00 | 184.80 | 6360.00 | 6346.67 | 0.00 | 13.33 | 0.98 |
| full1 | linear | 528.00 | 386.10 | 141.90 | 0.00 | 529.52 | 390.00 | 139.52 | 0.00 | 1.00 |

### ResNet Layer‑Wise Energy Breakdown (pJ)

| Layer | Type | Event‑Driven Total (pJ) | Event‑Mem (pJ) | Event‑Core (pJ) | Event‑Periph (pJ) | IMC_new Total (pJ) | IMC‑Mem (pJ) | IMC‑Core (pJ) | IMC‑Periph (pJ) | Ratio (new/old) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv1 | conv2d | 1096838.50 | 1955.41 | 1068892.16 | 25990.93 | 1095117.69 | 420.73 | 1068892.16 | 25804.80 | 1.00 |
| conv2 | conv2d | 4350143.65 | 12846.56 | 4285315.24 | 51981.85 | 4330310.17 | 3122.41 | 4275568.64 | 51619.12 | 1.00 |
| pool1 | maxpool | 24870.96 | 2128.90 | 0.00 | 22742.06 | 23451.72 | 709.66 | 0.00 | 22742.06 | 0.94 |
| conv3 | conv2d | 2163700.26 | 5610.53 | 2145094.27 | 12995.46 | 2152134.56 | 1419.29 | 2137784.32 | 12930.95 | 0.99 |
| conv4 | conv2d | 2163700.26 | 5610.53 | 2145094.27 | 12995.46 | 2152134.56 | 1419.29 | 2137784.32 | 12930.95 | 0.99 |
| res1 | residual | 590.73 | 568.97 | 21.76 | 0.00 | 2862.45 | 425.80 | 0.00 | 2436.65 | 4.85 |
| conv5 | conv2d | 4322499.62 | 6320.16 | 4290188.54 | 25990.93 | 4913795.43 | 1561.21 | 4886364.16 | 25870.07 | 1.14 |
| pool2 | maxpool | 12435.48 | 1064.45 | 0.00 | 11371.03 | 11725.87 | 354.84 | 0.00 | 11371.03 | 0.94 |
| conv6 | conv2d | 2155515.80 | 2705.47 | 2146312.59 | 6497.73 | 2450495.23 | 709.66 | 2443182.08 | 6603.49 | 1.14 |
| pool3 | maxpool | 3108.87 | 266.11 | 0.00 | 2842.76 | 2931.49 | 88.73 | 0.00 | 2842.76 | 0.94 |
| conv7 | conv2d | 538832.38 | 629.80 | 536578.15 | 1624.43 | 612738.04 | 177.43 | 610795.52 | 1765.09 | 1.14 |
| conv8 | conv2d | 538832.38 | 629.80 | 536578.15 | 1624.43 | 612738.04 | 177.43 | 610795.52 | 1765.09 | 1.14 |
| res2 | residual | 72.40 | 71.04 | 1.36 | 0.00 | 357.82 | 53.24 | 0.00 | 304.58 | 4.94 |
| pool4 | maxpool | 944.91 | 56.55 | 0.00 | 888.36 | 907.22 | 18.86 | 0.00 | 888.36 | 0.96 |
| full1 | linear | 190.98 | 4.58 | 186.40 | 0.00 | 187.56 | 1.16 | 186.40 | 0.00 | 0.98 |

### Residual Energy Numerical Example (Why mismatch is large)

Using `res1` (`output_h=16`, `output_w=16`, `out_channels=112`, `num_inputs=1`):

- \(M = 16 \times 16 \times 1 = 256\)
- \(N = 112\)
- CLB add energy/op \(= 0.08498358\) pJ

Event-driven residual compute (current event abstraction):

- \(E_{compute,event} = M \times 0.08498358 = 256 \times 0.08498358 = 21.76\) pJ

IMC residual compute (element-wise add abstraction):

- \(E_{compute,IMC} = M \times N \times 0.08498358 = 256 \times 112 \times 0.08498358 = 2436.65\) pJ

Compute-only ratio:

- \(E_{compute,IMC} / E_{compute,event} = 2436.65 / 21.76 \approx 112\times (=N)\)

Same pattern on `res2` (`M=16`, `N=224`):

- Event compute: \(16 \times 0.08498358 = 1.36\) pJ
- IMC compute: \(16 \times 224 \times 0.08498358 = 304.58\) pJ
- Ratio: \(304.58/1.36 \approx 224\times (=N)\)

Conclusion: the large residual gap is primarily from compute scaling choice: event-driven scales residual sum with \(M\), while IMC element-wise abstraction scales with \(M \times N\).

### ResNet Sanity Check (Layer‑Sum vs End‑to‑End)
| Metric | Event‑Driven (ns / pJ) | IMC_new (ns / pJ) |
| --- | --- | --- |
| Layer-wise isolated sum | 2968904.40 | 3099024.64 |
| End-to-end total | 992541.00 | 927525.57 |
| Overlap factor (isolated/e2e) | 2.99 | 3.34 |
| Layer-wise energy sum (pJ) | 17372277.17 | 18361887.85 |
| End-to-end energy (pJ) | 17372277.17 | 18361887.85 |
| Energy ratio (layer-sum/e2e) | 1.00 | 1.00 |

# End‑to‑End Comparison

## ResNet

### Totals (energy=pJ, latency=ns)
| Metric | IMC_new | Event‑Driven | Ratio (new/old) |
| --- | --- | --- | --- |
| Total Energy (pJ) | 18361887.85 | 17372277.17 | 1.06 |
| Total Latency (ns) | 927525.57 | 992541.00 | 0.93 |
| Event‑Driven breakdown vs Event‑Driven total (ns) | 2968904.40 | 992541.00 | 2.99 |

### Energy Breakdown (percent of total, unit=pJ)
| Bucket | IMC_new | Event‑Driven | Delta |
| --- | --- | --- | --- |
| memory | 0.06% | 0.23% | -0.17% |
| core | 98.96% | 98.75% | 0.22% |
| peripheral | 0.98% | 1.02% | -0.04% |

### Latency Breakdown (percent of total, proxy for event‑driven)
| Bucket | IMC_new | Event‑Driven (proxy) | Delta |
| --- | --- | --- | --- |
| memory | 95.23% | 94.96% | 0.27% |
| core | 4.73% | 4.95% | -0.23% |
| peripheral | 0.04% | 0.09% | -0.05% |
