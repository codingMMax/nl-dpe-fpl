# Onboarding Notes for NL-DPE Framework
_Last updated: 2026-03-13 (America/Chicago)_

## 1. Scope and Current Goal
This workspace combines Python simulators and RTL/VTR experiments for processor architecture evaluation.

The stated near-term goal is:
- implement the required RTL for NL-DPE in `nl_dpe_rtl` (note: folder name is `nl_dpe_rtl` in this workspace).

## 2. Repository Map (What exists now)

### Root
- `azurelily/` (git submodule): Python simulators + models.
- `azurelily_TACO_experiments/`: Verilog workloads + VTR/Yosys scripts + generated VTR outputs.
- `nl_dpe_rtl/`: NL-DPE architecture XML files only (no RTL `.v` yet).
- `design.sdc`: placeholder clock constraint (`create_clock -period 0 *`).

### `azurelily/` (Python side)
- Event-driven reference simulator:
  - `main.py`
  - `nn/*` (layer/event simulation for conv/linear/maxpool/residual/attention ops)
  - `models/*` (LeNet/ResNet/VGG/single-layer/attention graphs)
- Analytical IMC simulator (multi-architecture):
  - `IMC/test.py` (runner)
  - `IMC/simulator.py` (top-level composition)
  - `IMC/imc_core/*` (core model + config loading)
  - `IMC/peripherals/*` (memory/fabric models)
  - `IMC/scheduler_stats/*` (mapping, stats, critical-path profiling)
  - `IMC/configs/{azure_lily,nl_dpe,sram_cha}.json`
- Calibration:
  - `calibration_study/equivalence_study.py`
  - `calibration_study/assumptions_for_slides.md`

### `azurelily_TACO_experiments/` (RTL + CAD flow)
- Handwritten/assembled Verilog designs:
  - `conv_layer_single_dpe.v`
  - `conv_layer_stacked_dpes_conv3.v`
  - `dpe_one_layer*.v`
  - `lenet_1_channel.v`, `resnet_1_channel*.v`, `vgg11_1_channel.v`
  - `xbar_module.v`, `tanh_LUT_comb.v`
- Architecture files:
  - `azure_lily_22nm_with_dpe_auto.xml`
  - `azure_lily_22nm_with_dpe_550x550.xml`
- Flow scripts:
  - `makefile`
  - `dpe_synthesis_script.ys`
  - `yosys_test_opt.ys`
- Generated run dirs (build artifacts):
  - `*_550_550/` (contains `.blif/.net/.route/.rpt/.out/...`).

### `nl_dpe_rtl/` (target area)
- `nl_dpe_22nm_auto.xml`
- `nl_dpe_22nm_550x550.xml`
- No RTL implementation files currently.

## 3. Python Simulator Split (Important for agent context)

## 3.1 Event-driven reference (`azurelily/main.py`)
- Simulates per-event scheduling inside each layer object (`update()` loops).
- Energy unit: nJ (`nn/constant.py`).
- Useful as behavior/timing reference for Azure-Lily-specific flow.

## 3.2 Analytical IMC simulator (`azurelily/IMC`)
- Supports multiple IMC architectures via JSON config:
  - Azure-Lily
  - NL-DPE
  - SRAM-CHA
- Tracks:
  - layer energy totals
  - component energy breakdown
  - raw latency sum
  - overlap-aware critical-path latency
  - resource totals/peaks/layer usage
- Energy unit: pJ.

## 3.3 Attention path specifics
- Attention model uses layer names `linear_Q`, `linear_K`, `linear_V`, `mac_qk`, `softmax_exp`, `softmax_norm`, `mac_sv`.
- IMC critical-path profiler merges Q/K/V as a parallel group (`attn_qkv_parallel`).

## 4. Naming Status (Post-Update)
- `nl_dpe_rtl/` naming is now consistent with project intent.
- `azurelily/IMC/DEVELOPER_ONBOARDING.md` is now aligned to `IMC/*` (not `IMC_new/*`).
- Remaining legacy labels still exist in a few scripts (mostly text/variable naming, not core functionality):
  - `azurelily/calibration_study/equivalence_study.py` keeps `IMC_NEW_DIR` and report labels like `IMC_new`.
  - `azurelily/IMC/tools.py` still has description/default output path containing `IMC_new`.

Do not create a parallel `IMC_new/` tree unless explicitly requested.

## 5. Working Baseline Commands (verified in this workspace)
Run from repo root:

```bash
python3 azurelily/IMC/test.py --model lenet --imc_file azurelily/IMC/configs/azure_lily.json
python3 azurelily/main.py --model lenet
python3 azurelily/calibration_study/equivalence_study.py --imc_file azurelily/IMC/configs/azure_lily.json --out /tmp/equivalence_report_test.md
```

Observed on 2026-03-13:
- IMC LeNet run succeeded, with matching energy totals by layer and breakdown.
- Event-driven LeNet run succeeded.
- Equivalence study command succeeded and emitted report.

## 6. NL-DPE RTL Interface Contract (Most Critical)
The architecture XML in `nl_dpe_rtl` defines a custom model `.subckt dpe` and `wc` tile integration. This is the contract that RTL must satisfy for VTR flows.

### Required DPE ports
From `nl_dpe_rtl/nl_dpe_22nm_auto.xml` model/pb_type definitions:
- Inputs:
  - `data_in[15:0]`
  - `nl_dpe_control[1:0]`
  - `shift_add_control`
  - `w_buf_en`
  - `shift_add_bypass`
  - `load_output_reg`
  - `load_input_reg`
  - `reset`
  - `clk`
- Outputs:
  - `data_out[15:0]`
  - `MSB_SA_Ready`
  - `dpe_done`
  - `reg_full`
  - `shift_add_done`
  - `shift_add_bypass_ctrl`

### Mapping through `wc` tile
- `wc_slice.in[15:0] -> nl_dpe.data_in`
- control bits mapped from `wc_slice.in[22:16]`
- outputs mapped back to `wc_slice.out[20:0]`.

Any RTL implementation for NL-DPE must preserve this exact visible interface, widths, and signal names if it is intended to work with existing XML + `.subckt dpe` usage.

## 7. Relation Between `azurelily_TACO_experiments` and `nl_dpe_rtl`
- The XMLs in `nl_dpe_rtl` are close variants of the Azure-Lily/TACO XMLs.
- Key observed delta: `wc` tile area changed (`2320000` -> `1595429`) in NL-DPE XMLs.
- `auto` vs `550x550` pair difference is primarily layout mode:
  - `*_auto.xml` enables `<auto_layout>`
  - `*_550x550.xml` comments out auto layout and uses a fixed `dense_dpe` layout.

## 8. Practical VTR Flow Notes
`azurelily_TACO_experiments/makefile` expects a VTR installation and Python venv:
- default `VTR_ROOT` from `.env` (currently `/home/gajjar/VTR/vtr-verilog-to-routing`).
- default `DESIGN` and `ARCH` are set in the makefile and can be overridden via CLI.

Typical run pattern:
```bash
make -C azurelily_TACO_experiments run \
  DESIGN=conv_layer_single_dpe.v \
  ARCH=../nl_dpe_rtl/nl_dpe_22nm_auto.xml \
  OUTPUT_DIR=$(pwd)/azurelily_TACO_experiments/conv_layer_single_dpe_test
```

## 9. Immediate Development Roadmap for `nl_dpe_rtl`

### Phase 1: Make the RTL exist and compile
- Add NL-DPE RTL module(s) in `nl_dpe_rtl/` with top module `dpe` and exact port contract above.
- Keep a minimal synthesizable implementation first (correct interface + deterministic behavior + clocks/resets).
- Add a small self-contained testbench in `nl_dpe_rtl` for signal sanity.

### Phase 2: Connect to existing experiment wrappers
- Reuse wrapper/controller structures in `azurelily_TACO_experiments` (`conv_layer_single_dpe.v`, `dpe_one_layer.v`) to drive the new `dpe`.
- Ensure no port mismatch under Yosys/Parmys elaboration.

### Phase 3: CAD closure loop
- Route at least one small benchmark (`conv_layer_single_dpe` or `dpe_one_layer-small*`) with NL-DPE architecture XML.
- Capture resource/timing summary from output reports.
- Compare trend (not exact cycle matching) against IMC NL-DPE config outputs.

### Phase 4: Iterate architecture/perf fidelity
- Refine RTL micro-architecture for better alignment to intended NL-DPE behavior.
- Update XML timing/area assumptions only when RTL-level evidence justifies it.

## 10. Guardrails for Future Coding Agents
- Do not break `.subckt dpe` naming used by architecture model and Verilog wrappers.
- Keep control/status signals available even if early RTL uses simplified internals.
- Respect simulator unit differences:
  - `nn`: nJ
  - `IMC`: pJ
- Prefer adding new files in `nl_dpe_rtl` over mutating generated artifacts in `*_550_550/`.
- Treat `azurelily` as a submodule: coordinate edits carefully.

## 11. Suggested First Task for the Next Agent
“Implement a minimal synthesizable `dpe.v` in `nl_dpe_rtl` with the exact XML interface, then run one VTR flow (`conv_layer_single_dpe.v` + `nl_dpe_22nm_auto.xml`) and report parse/synth/pack/place/route pass-fail plus key resource/timing lines.”

## 12. High-Signal Files to Open First
- `azurelily/IMC/DEVELOPER_ONBOARDING.md`
- `azurelily/IMC/test.py`
- `azurelily/IMC/simulator.py`
- `azurelily/IMC/scheduler_stats/scheduler.py`
- `azurelily/IMC/scheduler_stats/pipeline_profiler.py`
- `azurelily/main.py`
- `azurelily/nn/layer.py`
- `azurelily/models/attention.py`
- `azurelily_TACO_experiments/conv_layer_single_dpe.v`
- `azurelily_TACO_experiments/dpe_one_layer.v`
- `azurelily_TACO_experiments/makefile`
- `nl_dpe_rtl/nl_dpe_22nm_auto.xml`
- `nl_dpe_rtl/nl_dpe_22nm_550x550.xml`
