# Phase 2 FC RTL verification report
**Gate:** PASS

- tb_dpe_vmm (DPE stub VMM correctness): PASS — tb_dpe_vmm: PASS

## 12-config matrix

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

## Known deltas (phase2_known_deltas.json)
- stage=`compute`, Δ=4, applies_to=all
  Root: conv_controller FSM handshake: reg_full -> dpe_exec_signal -> nl_dpe_control=2'b11 -> DPE fires in S_WAIT_EXEC. Three pipeline flops (nl_dpe/gemv_1_channel.v lines 488-510) + 1 cycle S_WAIT_EXEC check
- stage=`reduction_plus_activation`, Δ=1, applies_to=all
  Root: controller_scalable valid_n handshake: dpe_done -> (register) -> valid_n in controller_scalable (nl_dpe/gemv_1_channel.v controller_scalable always-block driving valid_n). The reduction adder tree and
