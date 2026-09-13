# Phase 2 Sub-agent A — FC RTL structural alignment (iterative)

Start: 2026-04-19 15:25 MST
Budget: 80 iterations or 2 h wall-clock.

## Seed discrepancy list (from the plan §"Known discrepancies")

1. **+4 cyc on compute stage across all 12 configs.** Root: FSM handshake in
   `nl_dpe/gemv_1_channel.v::conv_controller` lines 488-510. Registered
   `reg_full → dpe_exec_signal → nl_dpe_control=2'b11 → DPE fires`. Three
   pipeline flops + one extra cycle for the DPE stub's S_WAIT_EXEC check.
   Nature: modelling granularity. Expected outcome: add to
   `phase2_known_deltas.json`, no structural fix.
2. **Reduction stage +1 cyc RTL vs log2(V) sim.** Root (V=4 case):
   sim `math.ceil(log2(max(2,K=2048))) = 11` cycles, but the RTL
   implementation pipelines reduction with output (streaming accumulate).
   Nature: structural. Expected outcome: align sim to RTL (reduce is
   pipelined, effectively `max(output, reduc)` per the RTL's streaming
   accumulate).
3. **`sram_read_lat = 2` in sim.** Scripts include this in `feed_cycles`;
   RTL has 2 cycles of SRAM pipeline (addr reg + SRAM read reg).
   Appears in the measurement as part of `read` → actually accounted for.
4. **Activation placement**: per policy table:
   - Setup 0/1/3/4 (ADC): CLB LUT always (V=1 and V>1).
   - Setup 2/5 (ACAM) V=1: ACAM inside DPE, no LUT.
   - Setup 2/5 (ACAM) V>1: CLB LUT after reduction tree.
   Audit shows RTL gen already implements this exactly (`needs_clb_activation
   = (conversion == 'adc') or (v > 1)`). Confirm RTL behavior matches.
5. **Back-to-back inference pipelining (Regime B streaming).** FC is
   single-shot here; M=1 so no inter-pass compose needed. Out of scope for
   the FC latency TB (TB measures a single inference).

## Baseline measurement (from run_alignment.sh, pre-fix)

```
config              RTL_DPE_pipeline  RTL_full  sim_predicted  |delta|
setup0/fc_512_128   368               368       364            +4 (compute)
setup1/fc_512_128   177               177       173            +4 (compute)
setup2/fc_512_128   136               136       132            +4 (compute)
setup3/fc_512_128   368               368       364            +4 (compute)
setup4/fc_512_128   177               177       173            +4 (compute)
setup5/fc_512_128   136               136       132            +4 (compute)

setup0/fc_2048_256  368   full=305    366       +2 DPE, -61 full (reduction)
setup1/fc_2048_256  177   full=152    175       +2 DPE, -23 full
setup2/fc_2048_256  136   full=111    134       +2 DPE, -23 full
setup3/fc_2048_256  624   full=561    621       +3 DPE, -60 full (log2(2)=1)
setup4/fc_2048_256  279   full=254    276       +3 DPE, -22 full
setup5/fc_2048_256  238   full=213    235       +3 DPE, -22 full
```

Full-pipeline measurement (`T_validn - T_wbuf_first`) under-counts vs sim for
V>1 because `valid_n` fires 1 cycle after the first `dpe_done` when
reduction is streamed. The stage-wise `DPE_pipeline` (which sums
feed+compute+output_serialize) over-counts by `+4` due to the compute
handshake.

For this harness we measure per-stage (feed, compute, output, reduction,
activation) and compare each against the sim oracle separately, with the
+4 delta annotated.

## Log

Iteration format: `[iter N] config=X stage=Y action=Z result=pass/fail (D cyc)`

[iter 1] config=tb_dpe_vmm stage=functional action="override ACAM_MODE=0 (was defaulting to ACAM_MODE=1 exp approx, contaminating pure VMM expected values)" result=pass

