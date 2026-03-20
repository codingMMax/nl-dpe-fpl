

|                          | Azurelily            | NL-DPE(internal)        | NL-DPE(paper)              |
| :----------------------- | :------------------- | :---------------------- | :------------------------- |
| **ResNet, LeNet**        | Arch, RTL, Simulator | Arch', RTL', Simulator' | Arch’’, RTL’’, Simulator’’ |
| **Attention, BERT-Tiny** | X, X, X              | X, X,X                  | Arch’’, RTL’’, Simulator’’ |

Arch: Azurelily architecture file

Notes:
- What we want: baseline Azurelily architecture for VTR that matches the shipped RTL wrappers and 16-bit PHIT interface.
- Current: `azurelily_TACO_experiments/azure_lily_22nm_with_dpe_auto.xml` and `azurelily_TACO_experiments/azure_lily_22nm_with_dpe_550x550.xml` define the `.subckt dpe` block with `data_in/out` width 16 and are the default in `azurelily_TACO_experiments/makefile`.
- How to execute: run VTR with `ARCH=.../azure_lily_22nm_with_dpe_auto.xml` (makefile) or `python run_vtr.py <design.v> .../azure_lily_22nm_with_dpe_auto.xml`.

* [Optional] Modified to match the simulator PHIT size that is 16-bit interface on the DPE block

RTL: Azurelily Verilog file

* [Optional] Modified to match the simulator PHIT size that is 16-bit interface on the DPE block

Simulator: Azurelily Event-driven simulator

Notes:
- What we want: simulator PHIT size matches DPE data width (16-bit) so layer scheduling and energy models align with RTL/VTR.
- Current: `PHIT_SIZE = 16` in `azurelily/nn/constant.py` and defaults in `azurelily/main.py` are 16 unless overridden by CLI.
- How to execute: keep `--phit_size 16` in `azurelily/main.py` runs; change only if we also change the DPE interface width and XML wiring.

Arch': NL-DPE architecture file:
- [x] Replace DPE block in VTR with NL-DPE
  - What we want: VTR uses the NL-DPE architecture XML (`nl_dpe_rtl/nl_dpe_22nm_auto.xml` or `nl_dpe_rtl/nl_dpe_22nm_550x550.xml`) so `.subckt dpe` maps to NL-DPE tiles.
  - Current: NL-DPE XMLs exist in `nl_dpe_rtl/`, but the default flow in `azurelily_TACO_experiments/makefile` still points to Azurelily XML unless overridden.
  - How to execute: run `make -C azurelily_TACO_experiments run ARCH=../nl_dpe_rtl/nl_dpe_22nm_auto.xml DESIGN=<design.v>` or `python azurelily_TACO_experiments/run_vtr.py <design.v> nl_dpe_rtl/nl_dpe_22nm_auto.xml`.
- [x] Signal interface is the same (16-bit) as Azurelily
  - What we want: DPE `data_in[15:0]` and `data_out[15:0]` match simulator PHIT size (16 bits) and wrapper wiring.
  - Current: NL-DPE XMLs already define `data_in/out` width 16 and `wc_slice.in[15:0]` mapping; some wrappers still drive 8-bit into `dpe` (e.g., `conv_layer_single_dpe.v`), while others pad to 16 (e.g., `dpe_one_layer.v`).
  - How to execute: ensure the RTL `dpe` module uses 16-bit ports and update wrappers to pack/pad to 16 bits consistently, then re-run Yosys/VTR for a small design.
- [x] Area of the NL-DPE block
  - What we want: NL-DPE tile area reflects the intended NL-DPE macro (not Azurelily).
  - Current: `nl_dpe_rtl/nl_dpe_22nm_auto.xml` sets `tile name="wc"` area to `1595429` (verify against NL-DPE spec).
  - How to execute: update the `tile name="wc"` area in both NL-DPE XMLs and re-run VTR to confirm area reporting changes.
- [ ] Delay of the NL-DPE block *(deferred: using Azurelily delay constants for now — resolve later)*
  - What we want: timing model (`delay_constant`, `T_setup`, `T_clock_to_Q`) matches NL-DPE characterization.
  - Current: NL-DPE XMLs still use the Azurelily-like delays (e.g., `2.14e-9` constants).
  - How to execute: replace delay values under `pb_type name="nl_dpe"` in both NL-DPE XMLs, then run VTR to validate timing reports.

RTL': Azurelily RTL implementation modified with the following:

- [ ] Instantiate NL-DPE (with 16-bit interface) instead of Azurelily DPE.  
- What we want: all design wrappers instantiate a real NL-DPE RTL module `dpe` that matches the XML port contract (`data_in/out[15:0]`, control/status pins).
- Current: there is no NL-DPE RTL in `nl_dpe_rtl/` yet, and some wrappers still assume 8-bit inputs (e.g., `conv_layer_single_dpe.v`).
- How to execute: implement `nl_dpe_rtl/dpe.v` with the exact port list from the NL-DPE XML, add it to the VTR/Yosys file list, and update wrappers to pad/pack inputs to 16 bits.
- [ ] Activation happens within NL-DPE block  
- What we want: activation logic is contained in the NL-DPE block so activation cost/timing is attributed to DPE.
- Current: activation is either external or not modeled in the current wrappers (e.g., separate `tanh_LUT_comb.v` exists in `azurelily_TACO_experiments/`).
- How to execute: move activation logic into `dpe.v` or make `dpe.v` conditionally apply activation based on control bits, then update simulators to match.
- [ ] Reduction, maxpool, vector-add logics (the actual implementation and parallelism) are the same as Azurelily RTL
- What we want: non-DPE compute blocks retain Azurelily behavior/parallelism for apples-to-apples comparison.
- Current: these are implemented in the model RTL (e.g., `max_pooling_N_CHANNELS_1` and `adder_dpe_*` modules in `azurelily_TACO_experiments/lenet_1_channel.v`, `resnet_1_channel.v`, `vgg11_1_channel.v`).
- How to execute: keep these modules unchanged; only swap the `dpe` module and its wiring, then re-synthesize and compare resource/timing outputs.

Simulator': Analytical simulator. 

* For non-DPE operations, use the same resource usage for each from the Azurelily RTL implementation (instead of estimating resource usage on our own in the simulator)
  - What we want: analytical IMC uses Azurelily RTL-derived resource counts (adders/maxpools/activations) to avoid mismatched assumptions.
  - Current: IMC defaults are set by constants and CLI flags (`azurelily/IMC/test.py`, `azurelily/IMC/report_utils.py`) and may not reflect RTL module counts.
  - How to execute: extract resource counts from Azurelily RTL, then pass them via `--num_adds`, `--num_maxpools`, `--num_acts` or bake them into the NL-DPE config JSON before running `azurelily/IMC/test.py`.

Arch’’: NL-DPE architecture file:

- [ ] Replace DPE block in VTR with NL-DPE  
- What we want: VTR uses NL-DPE XML with 40-bit data interface (paper spec).
- Current: 40-bit XML variant is not present; only 16-bit NL-DPE XMLs exist in `nl_dpe_rtl/`.
- How to execute: clone the NL-DPE XMLs and widen `data_in/out` and `wc_slice` wiring to 40 bits, then point VTR to the new XML.
- [ ] Signal interface is 40-bit  
- What we want: `data_in/out[39:0]` for NL-DPE paper path.
- Current: all existing XMLs and RTL wrappers are 16-bit or 8-bit.
- How to execute: update XML port widths and RTL wrappers to pack/unpack 40-bit PHITs, then update simulator PHIT size (`--phit_size 40`).
- [ ] Area of the NL-DPE block with 40-bit interface  
- What we want: area reflects the 40-bit NL-DPE macro.
- Current: area values reflect 16-bit estimates.
- How to execute: update `tile name="wc"` area in the 40-bit XML and confirm VTR area reports.
- [ ] Delay of the NL-DPE block with 40-bit interface
- What we want: timing reflects 40-bit NL-DPE characterization.
- Current: delay constants are 16-bit/placeholder values.
- How to execute: update delay constants and clock-to-Q/setup entries under `pb_type name="nl_dpe"` in the 40-bit XML.

RTL'’: Azurelily RTL implementation modified with the following:

- [ ] Instantiate NL-DPE (with 40 bit interface) instead of Azurelily DPE.  
- What we want: RTL uses a 40-bit NL-DPE `dpe` module and matching wiring.
- Current: RTL assumes 8-bit or 16-bit DPE inputs.
- How to execute: implement a 40-bit `dpe` and update wrappers to pack/unpack 40-bit PHITs.
- [ ] Activation happens within NL-DPE block  
- What we want: activation logic is inside NL-DPE for the 40-bit path.
- Current: activation is external or not integrated in DPE.
- How to execute: add activation to the 40-bit DPE RTL and verify functional simulation.
- [ ] [May change later] Reduction, maxpool, vector-add logics (the actual implementation and parallelism) are the same as Azurelily RTL
- What we want: keep non-DPE logic aligned with Azurelily for comparability.
- Current: non-DPE logic is still Azurelily RTL in the wrappers.
- How to execute: do not modify maxpool/adder/reduction modules; only change DPE width and wiring.

Simulator'’: Analytical simulator. 

* [May change later] For non-DPE operations, use the same resource usage for each from the Azurelily RTL implementation (instead of estimating resource usage on our own in the simulator)
  - What we want: 40-bit simulator path uses RTL-based resource counts for non-DPE ops.
  - Current: IMC defaults are 16-bit oriented unless overridden.
  - How to execute: run IMC with `--phit_size 40` and RTL-derived resource counts, and update NL-DPE config JSON if needed.

---

# NL-DPE vs Azure-Lily: VTR FPGA Comparison (16-bit interface)

## Architecture Parameters

| Parameter | NL-DPE | Azure-Lily |
|-----------|--------|------------|
| Crossbar size | 256 × 256 | 512 × 128 |
| Data width | 16-bit | 16-bit |
| Activation | ACAM (in-DPE) | LUT-based (FPGA fabric) |
| Tiling formula | k_tile = ⌈K/256⌉, n_tile = ⌈N/256⌉ | k_tile = ⌈K/512⌉, n_tile = ⌈N/128⌉ |

---

## LeNet

### Workload Mapping

| Layer | K | N | NL-DPE (256×256) | | Azure-Lily (512×128) | |
|-------|---|---|------|------|---------|------|
| | | | Tiling | DPEs | Tiling | DPEs |
| conv1 | 25 | 6 | 1×1 | 1 (single) | 1×1 | 1 (single) |
| conv2 | 150 | 16 | 1×1 | 1 (single) | 1×1 | 1 (single) |
| full1 | 400 | 120 | **2×1** | **2 (V2_H1)** | 1×1 | 1 (single) |
| full2 | 120 | 84 | 1×1 | 1 (single) | 1×1 | 1 (single) |
| full3 | 84 | 10 | 1×1 | 1 (single) | 1×1 | 1 (single) |
| **Total** | | | | **6** | | **5** |

> NL-DPE needs 1 extra DPE at full1 because K=400 > 256 rows (needs 2 vertical tiles).
> Azure-Lily fits K=400 in 512 rows with a single tile.

### Resource Utilization (VTR)

| Resource | NL-DPE | Azure-Lily | Delta |
|----------|--------|------------|-------|
| DPEs (wc) | 6 | 5 | +1 (+20%) |
| CLBs | 53 | 28 | +25 (+89%) |
| BRAMs | 8 | 13 | -5 (-38%) |
| IOs | 37 | 37 | 0 |

> NL-DPE has more CLBs due to V2_H1 adder tree in fabric.
> NL-DPE has fewer BRAMs because activation layers (removed) used SRAM buffers.

### Critical Path & Frequency

| Metric | NL-DPE | Azure-Lily |
|--------|--------|------------|
| Critical path delay | 3.613 ns | 3.900 ns |
| Fmax | **276.8 MHz** | **256.4 MHz** |
| Critical path type | Latch → Latch (FSM) | Latch → Latch (FSM) |
| Logic depth | 4 LUTs | 4 LUTs |
| Routing delay (total) | 2.87 ns (79%) | 3.18 ns (82%) |

> NL-DPE is ~8% faster. Both bottlenecked by controller FSM logic, not DPE routing.
> NL-DPE benefits from fewer controller states (no activation layer FSMs).

---

## ResNet

### Workload Mapping

| Layer | K | N | NL-DPE (256×256) | | Azure-Lily (512×128) | |
|-------|---|---|------|------|---------|------|
| | | | Tiling | DPEs | Tiling | DPEs |
| conv1 | 9 | 56 | 1×1 | 1 (single) | 1×1 | 1 (single) |
| conv2 | 504 | 112 | **2×1** | **2 (V2_H1)** | 1×1 | 1 (single) |
| conv3 | 1008 | 112 | **4×1** | **4 (V4_H1)** | 2×1 | 2 (V2_H1) |
| conv4 | 1008 | 112 | **4×1** | **4 (V4_H1)** | 2×1 | 2 (V2_H1) |
| conv5 | 1008 | 224 | **4×1** | **4 (V4_H1)** | 2×2 | 4 (V2_H2) |
| conv6 | 2016 | 224 | **8×1** | **8 (V8_H1)** | 4×2 | 8 (V4_H2) |
| conv7 | 2016 | 224 | **8×1** | **8 (V8_H1)** | 4×2 | 8 (V4_H2) |
| conv8 | 2016 | 224 | **8×1** | **8 (V8_H1)** | 4×2 | 8 (V4_H2) |
| conv9 | 224 | 10 | 1×1 | 1 (single) | 1×1 | 1 (single) |
| **Total** | | | | **40** | | **35** |

> NL-DPE needs 5 more DPEs because 256 rows requires more vertical tiles than 512 rows.
> NL-DPE uses vertical-only stacking (H=1); Azure-Lily uses both vertical + horizontal (H=2).
> Azure-Lily needs horizontal tiles at conv5-8 because N=224 > 128 cols.

### Resource Utilization (VTR)

| Resource | NL-DPE | Azure-Lily | Delta |
|----------|--------|------------|-------|
| DPEs (wc) | 40 | 35 | +5 (+14%) |
| CLBs | 281 | 185 | +96 (+52%) |
| BRAMs | 16 | 16 | 0 |
| IOs | 37 | 37 | 0 |

> NL-DPE CLB overhead comes from larger adder trees: V8_H1 needs 7 adders (3-level tree)
> vs Azure-Lily V4_H2 which uses 3 adders per horizontal column + output mux.

### Critical Path & Frequency

| Metric | NL-DPE | Azure-Lily |
|--------|--------|------------|
| Critical path delay | 5.647 ns | 4.648 ns |
| Fmax | **177.1 MHz** | **215.1 MHz** |
| Critical path type | DPE → DPE (reg_full → data_in) | DPE → DPE (reg_full → data_in) |
| Logic depth | 3 LUTs | 2 LUTs |
| Routing delay (total) | 5.11 ns (90%) | 4.26 ns (91%) |

> NL-DPE is ~18% slower. Both bottlenecked by DPE-to-DPE routing through xbar/controller.
> NL-DPE has 1 extra LUT level due to 8-output xbar decode complexity.
> Longer routes caused by 8 DPEs per conv_layer spread across larger physical area.

---

## Summary

| | LeNet | | ResNet | |
|---|---|---|---|---|
| Metric | NL-DPE | Azure-Lily | NL-DPE | Azure-Lily |
| DPEs | 6 | 5 | 40 | 35 |
| CLBs | 53 | 28 | 281 | 185 |
| BRAMs | 8 | 13 | 16 | 16 |
| Fmax (MHz) | **276.8** | 256.4 | 177.1 | **215.1** |
| Critical path | FSM logic | FSM logic | DPE routing | DPE routing |

**Key takeaways:**
- LeNet: NL-DPE slightly more DPEs (+1) but achieves higher frequency due to simpler pipeline (no activation layers)
- ResNet: NL-DPE needs more DPEs (+5) and more CLBs (+52%) due to deeper vertical stacking (V8_H1); frequency drops 18% due to routing-dominated critical path across 8-DPE groups
- Both architectures are routing-dominated (>79% of critical path is interconnect delay)
- The frequency gap in ResNet is a physical layout issue (more DPEs → larger area → longer wires), not a logic depth issue
