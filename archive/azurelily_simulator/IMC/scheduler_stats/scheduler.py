from __future__ import annotations
import math

from nn import MAC_QK_Layer, MAC_SV_Layer, Softmax_Exp_Layer, Softmax_Norm_Layer, Linear_Layer, LayerNorm_Layer, Embedding_Layer
from nn.layer import Layer
from peripherals.fpga_fabric import FPGAFabric
from scheduler_stats.common import log
from imc_core.imc_core import IMCCore 
class Scheduler:
    def __init__(self, cfg, stats, imc_core: IMCCore, fpga: FPGAFabric):
        self.cfg = cfg
        self.stats = stats
        self.imc_core = imc_core
        self.fpga = fpga
        self._global_bram_counted = False

    def _merge_pipelined_activation_latency(self, base_latency_ns, M, K, N, act_row_ns):
        profile = self.imc_core.gemm_pipeline_profile(K, N)
        base_steady = profile["steady"]
        extra_steady = max(0.0, act_row_ns - base_steady) * max(0, M - 1)
        # Activation starts when first output row is produced and adds one drain row.
        return base_latency_ns + act_row_ns + extra_steady

    def _required_upstream_outputs_before_launch(self, layer: Layer):
        if layer.type in ("conv2d", "maxpool"):
            # Conservative receptive-field dependency estimate:
            # downstream layer waits until enough upstream output positions are produced
            # to populate the first valid window rows.
            required_rows = max(1, layer.kernel_size - layer.padding)
            outputs_per_row = max(1, layer.input_width * layer.num_inputs)
            return max(1, required_rows * outputs_per_row)
        return 1

    def run_attention_head(self, layers, S):
        """Run attention head layers as a streaming pipeline.

        Args:
            layers: list of [mac_qk, softmax_exp, softmax_norm, mac_sv]
            S: sequence length (number of streaming rows)

        Returns: (pipeline_latency_ns, total_energy_pj)
        """
        return self._run_attention_pipeline(layers, S)

    def run_layer(self, layer: Layer):
        log(f"Running Layer {layer.name} on {self.cfg.imc} Simulator")
        # Top-level global SRAM counted once on first layer
        if not self._global_bram_counted:
            self._global_bram_counted = True
            self.stats.record_resource("memory_blocks", 1)
        self.stats.begin_layer(layer.name)
        try:
            timing = {"first_output_ns": 0.0}
            if layer.type == "conv2d":
                lat, energy, timing = self._run_conv2d(layer)
            elif layer.type == "linear":
                lat, energy, timing = self._run_linear(layer)
            elif layer.type == "maxpool":
                lat, energy, timing = self.fpga.maxpool(layer, return_profile=True)
            elif layer.type == "residual":
                lat, energy, timing = self._run_residual(layer)
            elif layer.type == "mac_qk":
                lat, energy, _ = self._run_dimm(layer)
                timing = {"first_output_ns": lat, "steady": lat, "events": 1}
            elif layer.type == "mac_sv":
                lat, energy, _ = self._run_dimm(layer)
                timing = {"first_output_ns": lat, "steady": lat, "events": 1}
            elif layer.type == "softmax_exp":
                lat, energy = self._run_softmax_exp(layer)
                timing = {"first_output_ns": lat, "steady": lat, "events": 1}
            elif layer.type == "softmax_norm":
                lat, energy = self._run_softmax_norm(layer)
                timing = {"first_output_ns": lat, "steady": lat, "events": 1}
            elif layer.type == "layernorm":
                lat, energy = self._run_layernorm(layer)
                timing = {"first_output_ns": lat, "steady": lat, "events": 1}
            elif layer.type == "embedding":
                lat, energy = self._run_embedding(layer)
                timing = {"first_output_ns": lat, "steady": lat, "events": 1}
            else:
                raise ValueError(f"Unsupported layer type {layer.type}")

            timing["required_upstream_outputs"] = self._required_upstream_outputs_before_launch(layer)
            self.stats.record_energy(layer.name, energy)
            self.stats.record_latency(layer, lat, timing=timing)
            return lat, energy
        finally:
            self.stats.end_layer()

    def _run_conv2d(self, layer: Layer):
        assert layer.type == "conv2d", f"_run_conv2d() supports conv2d layer only"
        K = layer.kernel_size * layer.kernel_size * layer.in_channels
        M = layer.output_height * layer.output_width * layer.num_inputs
        N = layer.out_channels
        log(f"=== Running conv2d in {M, K, N} GEMM form ===")
        ks = layer.kernel_size
        c_in = layer.in_channels
        # Path A CLB-cycle gate uses activation_mode=layer.has_act.
        lat_ns, energy_pj = self.imc_core.run_gemm(
            M, K, N, kernel_size=ks, in_channels=c_in,
            activation_mode=bool(getattr(layer, 'has_act', False)),
        )
        gemm_profile = self.imc_core.gemm_pipeline_profile(K, N, kernel_size=ks, in_channels=c_in)
        timing = {
            "first_output_ns": gemm_profile["first_output_ns"],
            "steady": gemm_profile["steady"],
            "events": M,
        }
        # Conv wrapper has 1 BRAM for input buffer
        self.stats.record_resource("memory_blocks", 1, peak=True)
        log(f"=== conv2d latency {lat_ns:.2f} ns, energy {energy_pj:.2f}pj ===\n")

        if layer.has_act:
            if self.imc_core.analoge_nonlinear_check(M, K, N):
                return lat_ns, energy_pj, timing
            # Activation is streaming: pipelined with DPE output serialization.
            # No separate BRAM round-trip. Only record energy.
            _, act_energy_pj = self.fpga.activation(M, N, return_profile=False)
            energy_pj += act_energy_pj
            return lat_ns, energy_pj, timing
        return lat_ns, energy_pj, timing

    def _run_linear(self, layer: Layer | Linear_Layer):
        assert layer.type == "linear", f"_run_linear() supports linear layer only"
        K = layer.in_channels
        N = layer.out_channels
        M = layer.num_inputs

        # Baseline FPGA: no DPE, route all linear layers through DSP GEMM.
        # Parallel-output FC has one dsp_mac per output channel (= N),
        # so n_parallel_outputs=N captures the per-column parallelism
        # rather than the global DSP-pool batching default.
        if self.cfg.imc == "Baseline":
            log(f"=== Running linear layer in ({M}, {K}, {N}) GEMM form on DSP ===")
            lat_ns, energy_pj, _row = self.fpga.gemm_dsp(
                M, K, N, n_parallel_outputs=N,
            )
            timing = {"first_output_ns": lat_ns, "steady": lat_ns / M, "events": M}
            self.stats.record_resource("memory_blocks", 1, peak=True)
            if not isinstance(layer, Linear_Layer) and layer.has_act:
                act_lat_ns, act_energy_pj, act_profile = self.fpga.activation(
                    M, N, return_profile=True)
                lat_ns += act_lat_ns
                energy_pj += act_energy_pj
                self.stats.record_resource("memory_blocks", 1, peak=True)
            log(f"=== linear layer: latency {lat_ns:.2f}ns, energy {energy_pj:.2f}pj ===\n")
            return lat_ns, energy_pj, timing

        ks = getattr(layer, 'kernel_size', 1)
        c_in = layer.in_channels
        # Path A CLB-cycle gate uses activation_mode=layer.has_act
        # (Linear_Layer pure-linear has no has_act attribute → defaults False).
        lat_ns, energy_pj = self.imc_core.run_gemm(
            M, K, N, kernel_size=ks, in_channels=c_in,
            activation_mode=bool(getattr(layer, 'has_act', False)),
        )
        gemm_profile = self.imc_core.gemm_pipeline_profile(K, N, kernel_size=ks, in_channels=c_in)
        timing = {
            "first_output_ns": gemm_profile["first_output_ns"],
            "steady": gemm_profile["steady"],
            "events": M,
        }
        # Linear wrapper has 1 BRAM for input buffer
        self.stats.record_resource("memory_blocks", 1, peak=True)
        if isinstance(layer, Linear_Layer):
            return lat_ns, energy_pj, timing

        if layer.has_act:
            if self.imc_core.analoge_nonlinear_check(M, K, N):
                return lat_ns, energy_pj, timing
            # Activation is streaming: pipelined with DPE output serialization.
            # No separate BRAM round-trip. Only record energy; latency is absorbed
            # into the DPE output serialize stage (t_output in core pipeline).
            _, act_energy_pj = self.fpga.activation(M, N, return_profile=False)
            energy_pj += act_energy_pj
            return lat_ns, energy_pj, timing
        return lat_ns, energy_pj, timing

    def _run_dimm(self, layer: MAC_QK_Layer | MAC_SV_Layer):
        if isinstance(layer, MAC_SV_Layer):
            M = layer.N; K = layer.N; N = layer.d
        else:
            M = layer.N; K = layer.d; N = layer.N

        # DIMM hardware has W parallel-row lanes (one dsp_mac / DPE per
        # output row of the score / weighted-sum stream) — pass N as the
        # effective DSP-axis parallelism so gemm_dsp models the actual
        # dataflow rather than serialising along the output axis.
        # This matches the parallel-row DIMM RTL (W=16 lanes for the
        # current 128×128 attention head; the scheduler hands gemm_dsp the
        # logical N which is the row count along the parallel axis).
        n_parallel_outputs_dimm = N

        row_timing = None
        if self.cfg.imc == "Baseline":
            log(f"=== Running {layer.name} layer in ({M}, {K}, {N}) GEMM form on DSP ===")
            latency, energy, row_timing = self.fpga.gemm_dsp(
                M, K, N, n_parallel_outputs=n_parallel_outputs_dimm,
            )
        elif self.cfg.imc == "Azure-Lily":
            # §7: AL DIMM uses W DSP-MAC lanes (paper/attention_dimm_mapping.md
            # §6 confirms 16 dsp_mac per DIMM stage, DSP_WIDTH=4). Use the
            # same `total_softmax_lanes` allocation knob as NL-DPE for a
            # unified W across architectures, rather than overclaiming
            # n_parallel_outputs=N.
            n_parallel_outputs_al_dimm = max(
                1, getattr(self.cfg, 'total_softmax_lanes', 16)
            )
            if self.cfg.total_dsp > 0:
                log(f"=== Running {layer.name} layer in ({M}, {K}, {N}) GEMM form on FPGA DSP "
                    f"(W={n_parallel_outputs_al_dimm} lanes) ===")
                latency, energy, row_timing = self.fpga.gemm_dsp(
                    M, K, N, n_parallel_outputs=n_parallel_outputs_al_dimm,
                )
            else:
                log(f"=== Running {layer.name} layer in ({M}, {K}, {N}) GEMM form on CLB MAC (0 DSPs) ===")
                latency, energy = self.fpga.gemm_clb(M, K, N)
        elif self.cfg.imc == "NL-DPE":
            # §7: single allocation knob for W lanes across mac_qk / softmax /
            # mac_sv (row-parallel, Pattern β shared B + broadcast).
            n_parallel_dpes = max(1, getattr(self.cfg, 'total_softmax_lanes', 1))
            log(f"=== Running {layer.name} layer in ({M}, {K}, {N}) GEMM-log on {n_parallel_dpes} DIMM DPEs ===")
            latency, energy, row_timing = self.fpga.gemm_log(M, K, N, n_parallel_dpes=n_parallel_dpes)
            if isinstance(layer, MAC_SV_Layer) and self.cfg.analoge_nonlinear_support:
                if self.cfg.log_softmax_fusion:
                    log(f"\t[log-softmax fusion] Skipping DPE(I|log) on attn weights")
                else:
                    t_log, e_log = self.imc_core.dimm_nonlinear(K, op="log", record_breakdown=False)
                    t_log_total = t_log * M
                    e_log_total = e_log * M
                    num_p = math.ceil(K / self.cfg.cols)
                    self.stats.record_energy_breakdown("imc_dimm_log_vmm", M * num_p * self.imc_core.energy_per_vmm())
                    self.stats.record_energy_breakdown("imc_dimm_log_conv", M * num_p * self.imc_core.energy_per_conversion())
                    self.stats.record_energy_breakdown("imc_dimm_log_digital", M * num_p * self.imc_core.energy_per_digital_post())
                    latency += t_log_total
                    energy += e_log_total
                    if row_timing:
                        row_timing["per_row_compute"] += t_log / 1  # amortized per row
                        row_timing["per_row_total"] += t_log
                        row_timing["per_row_steady"] = max(
                            row_timing["per_row_read"], row_timing["per_row_compute"], row_timing["per_row_write"])
        elif self.cfg.imc == "SRAM-CHA":
            latency, energy = self.imc_core.run_gemm(M, K, N)
        return latency, energy, row_timing

    def _run_attention_pipeline(self, layers, S):
        """Model the streaming attention pipeline with inter-stage buffer overlap.

        RTL streams row-by-row:
          QK^T row → [score_buf] → Softmax row → [attn_buf] → Score×V row

        Pipeline: fill + (S-1) × max(stage_row_latencies) + drain
        Energy: unchanged (same total work).
        """
        # Run each stage to get energy + per-row timing
        stage_results = []
        for layer in layers:
            if layer.type in ("mac_qk", "mac_sv"):
                lat, energy, row_timing = self._run_dimm(layer)
            elif layer.type == "softmax_exp":
                lat, energy = self._run_softmax_exp(layer)
                rows = layer.N
                # Per-row: compute + buffer I/O
                t_row_compute = lat / rows  # amortized
                t_buf_rw = self.fpga.memory.latency(rows) + self.fpga.memory.latency(rows, read=False)
                row_timing = {
                    "per_row_steady": t_row_compute + t_buf_rw / rows,
                    "per_row_total": t_row_compute + t_buf_rw / rows,
                    "rows": rows,
                }
            elif layer.type == "softmax_norm":
                lat, energy = self._run_softmax_norm(layer)
                rows = layer.N
                t_row_compute = lat / rows
                t_buf_rw = self.fpga.memory.latency(rows) + self.fpga.memory.latency(rows, read=False)
                row_timing = {
                    "per_row_steady": t_row_compute + t_buf_rw / rows,
                    "per_row_total": t_row_compute + t_buf_rw / rows,
                    "rows": rows,
                }
            else:
                continue

            # Record energy (already done inside each _run method)
            self.stats.record_energy(layer.name, energy)

            stage_results.append({
                "name": layer.name,
                "energy": energy,
                "total_lat": lat,
                "row_timing": row_timing,
            })

        if not stage_results:
            return 0.0, 0.0

        total_energy = sum(s["energy"] for s in stage_results)

        # Pipeline latency: fill + (S-1) × steady + drain
        per_row_steadies = [s["row_timing"]["per_row_steady"] for s in stage_results if s["row_timing"]]
        per_row_totals = [s["row_timing"]["per_row_total"] for s in stage_results if s["row_timing"]]

        if per_row_steadies:
            t_steady = max(per_row_steadies)  # bottleneck stage per row
            t_fill = per_row_totals[0]  # first row through first stage
            t_drain = sum(per_row_totals[1:])  # last row through remaining stages
            pipeline_lat = t_fill + max(0, S - 1) * t_steady + t_drain

            sequential_lat = sum(s["total_lat"] for s in stage_results)
            speedup = sequential_lat / pipeline_lat if pipeline_lat > 0 else 1.0

            log(f"\t=== Attention Pipeline (S={S}) ===")
            for s in stage_results:
                rt = s["row_timing"]
                log(f"\t  {s['name']}: steady={rt['per_row_steady']:.1f} ns/row")
            log(f"\t  Bottleneck: {t_steady:.1f} ns/row")
            log(f"\t  Pipeline: fill={t_fill:.1f} + {S-1}×{t_steady:.1f} + drain={t_drain:.1f} "
                f"= {pipeline_lat:.1f} ns")
            log(f"\t  Sequential: {sequential_lat:.1f} ns → {speedup:.2f}× speedup")
        else:
            pipeline_lat = sum(s["total_lat"] for s in stage_results)

        # Record pipeline latency in stats (replaces per-layer sequential latency)
        for s in stage_results:
            self.stats.latency_raw[s["name"]] = self.stats.latency_raw.get(s["name"], 0)
            # Distribute pipeline latency proportionally across stages
            stage_fraction = s["total_lat"] / max(1, sum(sr["total_lat"] for sr in stage_results))
            allocated_lat = pipeline_lat * stage_fraction
            self.stats.latency_breakdown[s["name"]] = self.stats.latency_breakdown.get(s["name"], 0) + allocated_lat
            self.stats.latency_raw[s["name"]] += s["total_lat"]  # raw keeps sequential

        return pipeline_lat, total_energy

    def _run_softmax_exp(self, layer: Softmax_Exp_Layer):
        rows = layer.N
        cols = layer.d
        log(f"=== Running softmax-Exp on FPGA for ({rows}x{cols}) Matrix ===")

        # W_softmax: parallel softmax hardware lanes (§7 row-parallel).
        # Each lane owns rows_per_lane = ceil(rows / W) rows of output;
        # within a row, each lane reads FULL cols (no within-row col split).
        W_softmax = max(1, self.cfg.total_softmax_lanes)
        rows_per_lane = math.ceil(rows / W_softmax)

        # Memory: row-by-row streaming. Each lane reads one full row of
        # `cols` elements, computes exp, writes one full row.
        total_elements = rows * cols
        t_read_row_ns = self.fpga.memory.latency(cols)
        t_write_row_ns = self.fpga.memory.latency(cols, read=False)
        t_read_ns = t_read_row_ns * rows_per_lane
        t_write_ns = t_write_row_ns * rows_per_lane
        e_read_pj = self.fpga.memory.energy(total_elements)
        e_write_pj = self.fpga.memory.energy(total_elements, read=False)
        log(f"\tMemory (row streaming): {rows} rows × {cols} elements/row, "
            f"W_softmax={W_softmax}, rows_per_lane={rows_per_lane}")
        log(f"\t\tread latency {t_read_ns:.2f} ns, energy {e_read_pj:.2f} pJ")
        log(f"\t\twrite latency {t_write_ns:.2f} ns, energy {e_write_pj:.2f} pJ")

        if self.cfg.analoge_nonlinear_support:
            # NL-DPE: exp via DPE(I|exp). Each lane owns rows_per_lane rows;
            # per-row latency is one full DPE pass for `cols` elements.
            latency_per_row, energy_per_row = self.imc_core.dimm_nonlinear(
                cols, op="exp", record_breakdown=False
            )
            compute_latency = latency_per_row * rows_per_lane
            compute_energy = energy_per_row * rows  # every row fires a DPE
            # Split into 3 sub-keys
            num_p = math.ceil(cols / self.cfg.cols)
            self.stats.record_energy_breakdown("imc_dimm_exp_vmm", rows * num_p * self.imc_core.energy_per_vmm())
            self.stats.record_energy_breakdown("imc_dimm_exp_conv", rows * num_p * self.imc_core.energy_per_conversion())
            self.stats.record_energy_breakdown("imc_dimm_exp_digital", rows * num_p * self.imc_core.energy_per_digital_post())
        else:
            latency, energy = self.fpga.exp_fpga(cols)
            # AL: W_softmax clb_exp LUT instances, each running rows_per_lane rows.
            compute_latency, compute_energy = latency * rows_per_lane, energy * rows
            self.stats.record_energy_breakdown("clb_exp", compute_energy)

        total_latency = t_read_ns + compute_latency + t_write_ns
        total_energy = e_read_pj + compute_energy + e_write_pj
        log(f"\tTotal Exp-Function latency {total_latency:.2f} Total Exp-Function energy {total_energy:.2f}")
        return total_latency, total_energy

    def _run_softmax_norm(self, layer: Softmax_Norm_Layer):
        rows = layer.N
        cols = layer.d
        log(f"=== Running softmax-Norm on FPGA for ({rows}x{cols}) Matrix ===")

        # W_softmax: parallel softmax hardware lanes (§7 row-parallel).
        # Each lane owns rows_per_lane = ceil(rows / W) rows of output;
        # within a row, each lane reads FULL cols (no within-row col split).
        W_softmax = max(1, self.cfg.total_softmax_lanes)
        rows_per_lane = math.ceil(rows / W_softmax)

        # Memory: row-by-row streaming. Each lane reads/writes full `cols`
        # per row. Per-lane latency = memory.latency(cols) × rows_per_lane.
        total_elements = rows * cols
        t_mem_read_row = self.fpga.memory.latency(cols)
        t_mem_write_row = self.fpga.memory.latency(cols, read=False)
        t_mem_read_ns = t_mem_read_row * rows_per_lane
        t_mem_write_ns = t_mem_write_row * rows_per_lane
        e_mem_read_pj = self.fpga.memory.energy(total_elements)
        e_mem_write_pj = self.fpga.memory.energy(total_elements, read=False)
        log(f"\tMemory: read {total_elements} exp values, write {total_elements} normalized weights "
            f"(W_softmax={W_softmax}, rows_per_lane={rows_per_lane})")
        log(f"\t\tread latency {t_mem_read_ns:.2f} ns, energy {e_mem_read_pj:.2f} pJ")
        log(f"\t\twrite latency {t_mem_write_ns:.2f} ns, energy {e_mem_write_pj:.2f} pJ")

        if self.cfg.analoge_nonlinear_support and self.cfg.log_softmax_fusion:
            # Log-softmax fusion: output log_softmax(x_i) = x_i - log(Σ exp(x_j))
            # instead of the standard softmax(x_i) = exp(x_i) / Σ exp(x_j).
            #
            # Steps per row:
            #   1. CLB sum: Σ exp(x_j)  (adder tree, from softmax_exp output)
            #   2. DPE(I|log): log(sum)  (one DPE pass)
            #   3. CLB subtract: x_i - log(sum)  (element-wise, replaces DSP multiply)
            #
            # The output is in log-domain, so mac_sv (Score×V DIMM) can skip
            # its DPE(I|log) conversion — the exp→log cancel out.
            latency, energy, e_sum_pj, e_inv_pj, e_mul_pj = self.fpga.norm_fpga(cols)
            # Step 2: DPE(I|log) on scalar sum
            t_log_ns, e_log_pj = self.imc_core.dimm_nonlinear(
                1, op="log", record_breakdown=False
            )
            # Step 3: CLB subtract (cols elements per row) — much cheaper than DSP mul
            e_sub_pj = cols * self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add
            adjusted_energy = (e_sum_pj + e_log_pj + e_sub_pj)
            # Per §7: W lanes work in parallel along the rows axis. Each lane
            # processes its rows_per_lane rows serially with full per-row work.
            total_latency = latency * rows_per_lane
            total_energy = adjusted_energy * rows
            self.stats.record_energy_breakdown("clb_norm_sum", e_sum_pj * rows)
            # Split DPE(I|log) scalar into 3 sub-keys
            num_p_log = math.ceil(1 / self.cfg.cols)  # 1 element → 1 pass
            self.stats.record_energy_breakdown("imc_dimm_log_vmm", rows * num_p_log * self.imc_core.energy_per_vmm())
            self.stats.record_energy_breakdown("imc_dimm_log_conv", rows * num_p_log * self.imc_core.energy_per_conversion())
            self.stats.record_energy_breakdown("imc_dimm_log_digital", rows * num_p_log * self.imc_core.energy_per_digital_post())
            self.stats.record_energy_breakdown("clb_subtract", e_sub_pj * rows)
            log(f"\t[log-softmax fusion] sum={e_sum_pj:.2f} + log={e_log_pj:.2f} + "
                f"sub={e_sub_pj:.2f} pJ/row, {rows} rows")
        elif self.cfg.analoge_nonlinear_support:
            # NL-DPE without fusion: sum + DPE(I|log) + DSP multiply
            latency, energy, e_sum_pj, e_inv_pj, e_mul_pj = self.fpga.norm_fpga(cols)
            t_log_ns, e_log_pj = self.imc_core.dimm_nonlinear(
                1, op="log", record_breakdown=False
            )
            e_inv_replaced = e_log_pj
            adjusted_energy = (e_sum_pj + e_inv_replaced + e_mul_pj)
            # §7: W lanes parallel along rows axis.
            total_latency = latency * rows_per_lane
            total_energy = adjusted_energy * rows
            self.stats.record_energy_breakdown("clb_norm_sum", e_sum_pj * rows)
            # Split DPE(I|log) scalar into 3 sub-keys
            num_p_log = math.ceil(1 / self.cfg.cols)
            self.stats.record_energy_breakdown("imc_dimm_log_vmm", rows * num_p_log * self.imc_core.energy_per_vmm())
            self.stats.record_energy_breakdown("imc_dimm_log_conv", rows * num_p_log * self.imc_core.energy_per_conversion())
            self.stats.record_energy_breakdown("imc_dimm_log_digital", rows * num_p_log * self.imc_core.energy_per_digital_post())
            self.stats.record_energy_breakdweown("mul", e_mul_pj * rows)
        else:
            latency, energy, e_sum_pj, e_inv_pj, e_mul_pj = self.fpga.norm_fpga(cols)
            # AL: W lanes process rows_per_lane rows in parallel.
            total_latency, total_energy = latency * rows_per_lane, energy * rows
            self.stats.record_energy_breakdown("clb_norm_sum", e_sum_pj * rows)
            self.stats.record_energy_breakdown("clb_norm_inv", e_inv_pj * rows)
            self.stats.record_energy_breakdown("mul", e_mul_pj * rows)
        # Add memory I/O costs (read exp buffer + write normalized output)
        total_latency += t_mem_read_ns + t_mem_write_ns
        total_energy += e_mem_read_pj + e_mem_write_pj
        log(f"\tTotal normalization latency {total_latency:.2f} Total normalization energy {total_energy:.2f}")
        return total_latency, total_energy

    def _run_layernorm(self, layer: LayerNorm_Layer):
        log(f"=== Running LayerNorm({layer.normalized_shape}) x {layer.seq_len} tokens ===")
        latency, energy = self.fpga.layernorm(layer.normalized_shape, layer.seq_len)
        log(f"=== LayerNorm: latency {latency:.2f} ns, energy {energy:.2f} pJ ===\n")
        return latency, energy

    def _run_embedding(self, layer: Embedding_Layer):
        log(f"=== Running Embedding: {layer.seq_len} tokens x {layer.d_model}d ===")
        latency, energy = self.fpga.embedding_lookup(layer.seq_len, layer.d_model)
        log(f"=== Embedding: latency {latency:.2f} ns, energy {energy:.2f} pJ ===\n")
        return latency, energy

    def _run_residual(self, layer: Layer):
        assert layer.type == "residual", f"_run_residual() supports residual layer only"
        output_positions = layer.output_height * layer.output_width * layer.num_inputs
        out_channels = layer.out_channels
        log(
            f"=== Running residual layer in ({output_positions}, {out_channels}) "
            f"element-wise add form ==="
        )
        lat_ns, energy_pj, profile = self.fpga.residual_add(
            output_positions=output_positions,
            out_channels=out_channels,
            num_computes=layer.num_computes,
            return_profile=True,
        )
        log(f"=== Residual layer: latency {lat_ns:.2f}ns, energy {energy_pj:.2f}pj ===\n")
        return lat_ns, energy_pj, profile
