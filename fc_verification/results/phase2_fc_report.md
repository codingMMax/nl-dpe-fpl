# Phase 2 FC RTL verification report
**Gate:** PASS

- tb_dpe_vmm (DPE stub VMM correctness): PASS — tb_dpe_vmm: PASS

## 12-config matrix (NL-DPE)

| # | Setup | Workload | Func | Route | Feed Δ | Comp Δ | Out Δ | Red+Act Δ | Sim | RTL | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | s0/adc | fc_512_128 | P | clb_lut=sim | 0 | 4 | 0 | - | 364 | 368 | PASS |
| 1 | s1/adc | fc_512_128 | P | clb_lut=sim | 0 | 4 | 0 | - | 173 | 177 | PASS |
| 2 | s2/acam | fc_512_128 | P | acam_absorbed=sim | 0 | 4 | 0 | - | 132 | 136 | PASS |
| 3 | s3/adc | fc_512_128 | P | clb_lut=sim | 0 | 4 | 0 | - | 364 | 368 | PASS |
| 4 | s4/adc | fc_512_128 | P | clb_lut=sim | 0 | 4 | 0 | - | 173 | 177 | PASS |
| 5 | s5/acam | fc_512_128 | P | acam_absorbed=sim | 0 | 4 | 0 | - | 132 | 136 | PASS |
| 0 | s0/adc | fc_2048_256 | P | clb_lut=sim | 0 | 4 | 0 | 1 | 364 | 368 | PASS |
| 1 | s1/adc | fc_2048_256 | P | clb_lut=sim | 0 | 4 | 0 | 1 | 173 | 177 | PASS |
| 2 | s2/acam | fc_2048_256 | P | clb_lut=sim | 0 | 4 | 0 | 1 | 132 | 136 | PASS |
| 3 | s3/adc | fc_2048_256 | P | clb_lut=sim | 0 | 4 | 0 | 1 | 620 | 624 | PASS |
| 4 | s4/adc | fc_2048_256 | P | clb_lut=sim | 0 | 4 | 0 | 1 | 275 | 279 | PASS |
| 5 | s5/acam | fc_2048_256 | P | clb_lut=sim | 0 | 4 | 0 | 1 | 234 | 238 | PASS |

## Per-stage annotations

### setup0/fc_512_128 (adc, dw16, V1H1)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup1/fc_512_128 (adc, dw40, V1H1)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup2/fc_512_128 (acam, dw40, V1H1)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- Activation routing RTL=acam_absorbed / sim=acam_absorbed → MATCH

### setup3/fc_512_128 (adc, dw16, V1H1)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup4/fc_512_128 (adc, dw40, V1H1)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup5/fc_512_128 (acam, dw40, V1H1)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- Activation routing RTL=acam_absorbed / sim=acam_absorbed → MATCH

### setup0/fc_2048_256 (adc, dw16, V4H2)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- `reduction_plus_activation`: Δ=1 cyc — annotated: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup1/fc_2048_256 (adc, dw40, V4H2)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- `reduction_plus_activation`: Δ=1 cyc — annotated: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup2/fc_2048_256 (acam, dw40, V4H2)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- `reduction_plus_activation`: Δ=1 cyc — annotated: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup3/fc_2048_256 (adc, dw16, V2H2)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- `reduction_plus_activation`: Δ=1 cyc — annotated: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup4/fc_2048_256 (adc, dw40, V2H2)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- `reduction_plus_activation`: Δ=1 cyc — annotated: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

### setup5/fc_2048_256 (acam, dw40, V2H2)
- `feed`: Δ=0 cyc — exact
- `compute`: Δ=4 cyc — annotated: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pi
- `output`: Δ=0 cyc — exact
- `reduction_plus_activation`: Δ=1 cyc — annotated: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v
- Activation routing RTL=clb_lut / sim=clb_lut → MATCH

## Azure-Lily FC (T1 of AH track)

Single-DSP serialised FC (dsp_mac, pure 4-wide int_sop_4, Phase 6A canonical). Per-output cost = PACKED_K + 3 cycles (2 SRAM-prime + PACKED_K dsp.valid + 1 latch).

| Workload | K | N | PACKED_K | per_output Δ | first_out Δ | aggregate Δ | drain Δ | Verdict |
|---|---|---|---|---|---|---|---|---|
| fc_512_128 | 512 | 128 | 128 | 0 | 1 | 1 | 0 | PASS |
| fc_2048_256 | 2048 | 256 | 512 | 0 | 1 | 1 | 0 | PASS |

### Per-stage annotations (AL FC)

#### al/fc_512_128
- `compute_first_out`: Δ=1 cyc — annotated: Same probe-convention root cause as compute_aggregate (force-state at cycle 0, real start at cycle 1). T_first_out - T_c
- `compute_aggregate`: Δ=1 cyc — annotated: Force-state probe convention in tb_azurelily_fc.v: T_compute_start is captured at cycle 0 (the simulation moment when st
- `output_drain`: Δ=0 cyc — exact
- `per_output_steady`: Δ=0 cyc — exact
- functional: PASS (dsp_out=1 expect=1 parallel, top_valid_n=26, expected_drain=26 packed-5)

#### al/fc_2048_256
- `compute_first_out`: Δ=1 cyc — annotated: Same probe-convention root cause as compute_aggregate (force-state at cycle 0, real start at cycle 1). T_first_out - T_c
- `compute_aggregate`: Δ=1 cyc — annotated: Force-state probe convention in tb_azurelily_fc.v: T_compute_start is captured at cycle 0 (the simulation moment when st
- `output_drain`: Δ=0 cyc — exact
- `per_output_steady`: Δ=0 cyc — exact
- functional: PASS (dsp_out=1 expect=1 parallel, top_valid_n=52, expected_drain=52 packed-5)

## Known deltas (phase2_known_deltas.json)
- [nl_dpe] stage=`compute`, Δ=4, applies_to=all
  Root: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pipeline flops (nl_dpe/gemv_1_channel.v lines 488-510) + 1 cycle S_WAIT_EXEC check
- [nl_dpe] stage=`reduction_plus_activation`, Δ=1, applies_to=all
  Root: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v controller_scalable always-block driving valid_n). The reduction adder tree and
- [azurelily] stage=`compute_aggregate`, Δ=1, applies_to=all
  Root: Force-state probe convention in tb_azurelily_fc.v: T_compute_start is captured at cycle 0 (the simulation moment when state was forced to S_COMPUTE via blocking assign in the initial block) but the FS
- [azurelily] stage=`compute_first_out`, Δ=1, applies_to=all
  Root: Same probe-convention root cause as compute_aggregate (force-state at cycle 0, real start at cycle 1). T_first_out - T_compute_start + 1 = sim PACKED_K+3 + 1 = PACKED_K + 4. Per-output steady-state (b
