from __future__ import annotations

import math

from scheduler_stats.common import DSP_WDITH, cycles_to_ns, log


class FPGAFabric:
    def __init__(self, cfg, memory, stats, imc_core=None):
        self.cfg = cfg
        self.memory = memory
        self.stats = stats
        self.imc_core = imc_core

    def _memory_latency_model(self, bytes_):
        bytes_per_access = max(
            1e-9, math.floor(self.cfg.bram_width / 8) * getattr(self.cfg, "mem_bw_utilization", 1.0)
        )
        num_access = math.ceil(bytes_ / bytes_per_access)
        if self.cfg.bram_mode == self.cfg.SP:
            num_cycle = num_access
        elif self.cfg.bram_mode == self.cfg.TDP:
            num_cycle = math.ceil(num_access / 2)
        else:
            num_cycle = num_access
        return cycles_to_ns(num_cycle, self.cfg.freq)

    def vector_add(self, num_vectors, vec_length, record_breakdown=True, return_compute=False):
        """Element-wise add of num_vectors vectors, each of length vec_length.

        Used by gemm_log for attention QK^T accumulation.
        Returns (latency_ns, energy_pj) or (latency_ns, energy_pj, compute_energy_pj)
        if return_compute=True.
        """
        total_ops = (num_vectors - 1) * vec_length  # pairwise adds
        read_bytes = num_vectors * vec_length
        write_bytes = vec_length

        t_read_ns = self.memory.latency(read_bytes)
        e_read_pj = self.memory.energy(read_bytes)

        per_op_pj = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add
        compute_energy_pj = total_ops * per_op_pj
        parallel_ops = max(1, self.cfg.total_clb)
        compute_cycles = math.ceil(total_ops / parallel_ops)
        t_compute_ns = cycles_to_ns(compute_cycles, self.cfg.freq)

        if record_breakdown:
            self.stats.record_energy_breakdown("clb_add", compute_energy_pj)
            self.stats.record_energy_breakdown("sram_read", e_read_pj)

        t_write_ns = self.memory.latency(write_bytes, read=False)
        e_write_pj = self.memory.energy(write_bytes, read=False)
        if record_breakdown:
            self.stats.record_energy_breakdown("sram_write", e_write_pj)

        t_total = t_read_ns + t_compute_ns + t_write_ns
        e_total = e_read_pj + compute_energy_pj + e_write_pj

        if return_compute:
            return t_total, e_total, compute_energy_pj
        return t_total, e_total

    def residual_add(self, output_positions, out_channels, num_computes=1, engine="clb", return_profile=False):
        """
        Element-wise residual add on two tensors of shape [output_positions, out_channels].
        """
        assert output_positions > 0 and out_channels > 0, (
            f"Illegal residual size ({output_positions}, {out_channels})"
        )
        num_computes = max(1, int(num_computes))

        # y = a + b reads two inputs and writes one output.
        read_bytes = 2 * output_positions * out_channels
        write_bytes = output_positions * out_channels
        total_ops = output_positions * out_channels
        num_steps = math.ceil(output_positions / num_computes)

        step_positions = num_computes
        read_step_bytes = 2 * step_positions * out_channels
        write_step_bytes = step_positions * out_channels
        step_ops = step_positions * out_channels

        log(f"\tReading 2 x ({output_positions} x {out_channels}) bytes from BRAM\t")
        t_read_ns = self.memory.latency(read_bytes)
        e_read_pj = self.memory.energy(read_bytes)
        log(f"\t\tread latency {t_read_ns:.2f} ns, read energy {e_read_pj:.2f} pj")

        if engine == "clb":
            per_op_pj = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add
            compute_energy_pj = total_ops * per_op_pj
            parallel_ops = max(1, self.cfg.total_clb)
            compute_cycles = math.ceil(total_ops / parallel_ops)
            add_step_cycles = math.ceil(step_ops / parallel_ops)
            self.stats.record_energy_breakdown("clb_add", compute_energy_pj)
            self.stats.record_resource("clb_used", min(parallel_ops, step_ops), peak=True)
            # Residual layer has 2 BRAMs (one per input branch)
            self.stats.record_resource("memory_blocks", 2, peak=True)
            log(
                f"\tResidual add via CLB op model: ops {total_ops}, "
                f"energy/op {per_op_pj:.6f} pj"
            )
        else:
            per_op_pj = self.cfg.e_dsp_pj_per_mac * 0.4
            compute_energy_pj = total_ops * per_op_pj
            parallel_ops = max(1, self.cfg.total_dsp)
            compute_cycles = math.ceil(total_ops / parallel_ops)
            add_step_cycles = math.ceil(step_ops / parallel_ops)
            self.stats.record_energy_breakdown("dsp_add", compute_energy_pj)
            self.stats.record_resource("dsp_used", min(parallel_ops, step_ops), peak=True)
            # Residual layer has 2 BRAMs (one per input branch)
            self.stats.record_resource("memory_blocks", 2, peak=True)
            log(
                f"\tResidual add via DSP op model: ops {total_ops}, "
                f"energy/op {per_op_pj:.6f} pj"
            )

        t_compute_ns = cycles_to_ns(compute_cycles, self.cfg.freq)
        log(f"\t\tResidual add latency {t_compute_ns:.2f} ns, energy {compute_energy_pj:.2f} pj")

        t_write_ns = self.memory.latency(write_bytes, read=False)
        e_write_pj = self.memory.energy(write_bytes, read=False)
        log(f"\tWriting {output_positions} x {out_channels} bytes to BRAM")
        log(f"\t\twrite latency {t_write_ns:.2f} ns, write energy {e_write_pj:.2f} pj")

        read_step_ns = self._memory_latency_model(read_step_bytes)
        add_step_ns = cycles_to_ns(add_step_cycles, self.cfg.freq)
        write_step_ns = self._memory_latency_model(write_step_bytes)
        first_output_ns = read_step_ns + add_step_ns + write_step_ns
        steady_ns = max(read_step_ns, add_step_ns, write_step_ns)
        total_latency = first_output_ns + max(0, num_steps - 1) * steady_ns
        total_energy = e_read_pj + compute_energy_pj + e_write_pj
        log(f"\tTotal residual latency {total_latency:.2f} ns, total residual energy {total_energy:.2f} pj")
        if return_profile:
            return total_latency, total_energy, {
                "first_output_ns": first_output_ns,
                "steady": steady_ns,
                "events": num_steps,
                "read_step_ns": read_step_ns,
                "add_step_ns": add_step_ns,
                "write_step_ns": write_step_ns,
            }
        return total_latency, total_energy

    def activation(self, M, N, return_profile=False):
        log(f"\tRunning Activation using FPGA logic resources ===")
        luts_per_act = 32
        luts_per_clb = 8
        clb_per_act = max(1, math.ceil(luts_per_act / luts_per_clb))
        max_units_by_clb = max(1, self.cfg.total_clb // clb_per_act)
        act_units = max(1, min(self.cfg.act_units, max_units_by_clb))

        act_ops = M * N
        act_row_ops = N
        act_cycles = math.ceil(act_ops / act_units) * self.cfg.act_cycles_per_op
        act_row_cycles = math.ceil(act_row_ops / act_units) * self.cfg.act_cycles_per_op

        act_latency = cycles_to_ns(act_cycles, self.cfg.freq)
        act_row_latency = cycles_to_ns(act_row_cycles, self.cfg.freq)
        act_energy = act_ops * self.cfg.act_energy_pj_per_op
        self.stats.record_energy_breakdown("fpga_activation", act_energy)
        self.stats.record_resource("act_units", act_units, peak=True)
        self.stats.record_resource("clb_used", act_units * clb_per_act, peak=True)
        log(f"\t\tactivation latnency {act_latency:.2f} ns, activtion energy {act_energy:.2f} pj")
        log(f"\t\tactivation row latency {act_row_latency:.2f} ns @ units {act_units}\n")
        if return_profile:
            return act_latency, act_energy, {"row": act_row_latency, "parallelism": act_units}
        return act_latency, act_energy

    def gemm_dsp(self, M, K, N, n_parallel_outputs=None):
        """DSP-based GEMM (§4 single-buffered drain-load overlap).

        AL crossbar has no ACAM, so DIMM matmul falls back to the DSP-MAC
        array. Each lane = one ``int_sop_4`` hard block (DSP_WIDTH=4 int8
        MACs/cycle), holding a private accumulator. W lanes work in
        lockstep on (m, n) iteration, producing W output elements per
        outer pass.

        Per-pass DPE-axiom (§4):
          LOAD    L = ceil(K × 8 / dpe_buf_width)
                    feed the K-element accumulation chain (one input
                    vector and matching weight column slice) per output
                    element, through the dpe_buf_width port.
          COMPUTE C = ceil(K / DSP_WIDTH)
                    K MACs at DSP_WIDTH per cycle on the int_sop_4 hard
                    block.
          OUTPUT  O = max(1, ceil(precision_bits / dpe_buf_width))
                    drain one int8 accumulator per lane back to BRAM.

        Each pass produces n_lanes output elements. With M·N total
        outputs:
            passes_per_lane = ceil(M × N / n_lanes)
            total_cycles    = T_fill + (passes_per_lane − 1) × T_steady
                            = (L + C + O) + (passes − 1) × max(L, C, O)

        Memory I/O is folded into per-pass L/O — there is **no** outer
        ``t_read + t_gemm + t_write`` serialization. Energy still tracks
        all SRAM reads/writes; only the latency is captured by L+O per
        pass per §4.

        Reference: paper/methodology/attention_dimm_mapping.md §6
        ("Azure-Lily W=16 DSP variant: 16 dsp_mac per DIMM matmul stage,
        each = one int_sop_4 hard block = 4 MAC/cycle"). Used by AL
        DIMM (W=16 lanes) and Baseline FC (n_parallel_outputs=N).

        Args:
            M, K, N: matrix dims (A=M×K · B=K×N → C=M×N).
            n_parallel_outputs: number of parallel DSP-MAC lanes (W).
                Defaults to ``self.cfg.total_dsp``. AL DIMM passes
                ``total_softmax_lanes`` (= W = 16), Baseline FC passes
                ``N`` for parallel-output FCs.
        """
        assert M > 0 and K > 0 and N > 0, f"Illegal GEMM/GEMV sizes: ({M,K,N})"

        DSP_WIDTH = DSP_WDITH  # 4 — int_sop_4 hard block, int8 pairs/cycle
        n_lanes = (
            self.cfg.total_dsp if n_parallel_outputs is None
            else max(1, int(n_parallel_outputs))
        )
        n_lanes = max(1, int(n_lanes))

        # ── Per-pass DPE-axiom (§4) ──
        dpe_bw = getattr(self.cfg, 'dpe_buf_width', self.cfg.bram_width)
        precision_bits = getattr(self.cfg, 'precision_bits', 8)
        L = math.ceil(K * precision_bits / dpe_bw)        # feed K elements per output
        C_cyc = max(1, math.ceil(K / DSP_WIDTH))           # K MACs at DSP_WIDTH/cycle
        O = max(1, math.ceil(precision_bits / dpe_bw))     # drain 1 result per lane

        # ── Total work + passes per lane ──
        total_outputs = M * N
        passes_per_lane = max(1, math.ceil(total_outputs / n_lanes))
        if self.imc_core is not None:
            # DSP-MAC primitive (dsp_mac.v) is NOT the single-substrate
            # faithful DPE; it stays on the ideal §4 formula. Pass
            # ``precision=0`` to suppress the Option A1 LOAD-gate term
            # in _pipeline_total_cycles_explicit.
            total_cycles = self.imc_core._pipeline_total_cycles_explicit(
                passes_per_lane, L, C_cyc, O, precision=0
            )
        else:
            # Ideal T_fill = L + C + O. DSP-MAC primitive has no single-
            # substrate slice-major store -- no LOAD-gate constraint
            # applies, so T_steady = max(L, C, O).
            t_fill = L + C_cyc + O
            t_steady = max(L, C_cyc, O)
            total_cycles = t_fill + max(0, passes_per_lane - 1) * t_steady

        t_clk = 1e3 / self.cfg.freq
        t_compute_ns = cycles_to_ns(total_cycles, self.cfg.freq)

        # ── Memory energy (latency folded into per-pass L/O per §4) ──
        # SRAM reads/writes still happen — energy is recorded honestly,
        # but the latency is already captured by LOAD/OUTPUT cycles.
        total_read_bytes = M * K + K * N
        write_bytes = M * N
        e_read_pj = self.memory.energy(total_read_bytes)
        e_write_pj = self.memory.energy(write_bytes, read=False)
        self.stats.record_energy_breakdown("sram_write", e_write_pj)
        log(f"\tReading A({M}×{K}) + B({K}×{N}) = {total_read_bytes} bytes from BRAM "
            f"(folded into per-pass LOAD)")
        log(f"\t\tread energy {e_read_pj:.2f} pj")
        log(f"\tWriting {write_bytes} bytes to BRAM (folded into per-pass OUTPUT)")
        log(f"\t\twrite energy {e_write_pj:.2f} pj")

        # ── DSP MAC energy ──
        # Total MACs = M × K × N (full work-volume, not n_lanes-batched).
        total_mac = M * K * N
        e_gemm_pj = total_mac * self.cfg.e_dsp_pj_per_mac
        self.stats.record_energy_breakdown("dsp_gemm", e_gemm_pj)
        self.stats.record_resource("dsp_used", n_lanes, peak=True)

        log(f"\tgemm_dsp({M},{K},{N}) §4: {n_lanes} lanes (DSP_WIDTH={DSP_WIDTH}), "
            f"per-pass L={L} C={C_cyc} O={O}, passes/lane={passes_per_lane}, "
            f"T_fill={L+C_cyc+O} (=L+C+O), T_steady={max(L,C_cyc,O)} "
            f"(DSP-MAC: no Option A1 LOAD-gate)")
        log(f"\t\tcompute cycles {total_cycles}, latency {t_compute_ns:.2f} ns")
        log(f"\t\tGEMM energy {e_gemm_pj:.2f} pj")

        # ── Total ──
        t_total_ns = t_compute_ns   # memory I/O latency folded into per-pass L/O
        e_total_pj = e_read_pj + e_gemm_pj + e_write_pj
        log(f"\tTotal GEMM latency {t_total_ns:.2f} ns, total GEMM energy {e_total_pj:.2f} pj")

        # ── Per-row timing for streaming pipeline composition ──
        # Each row of output (N elements) is produced over ceil(N / n_lanes)
        # passes per lane (averaging M rows across the full work). Streaming
        # interval per row = passes_per_row × T_steady; first-row fill includes
        # T_fill once.
        if M > 0 and passes_per_lane > 0:
            passes_per_row = max(1, math.ceil(N / n_lanes))
            t_steady_pass = max(L, C_cyc, O) * t_clk
            t_compute_row_ns = passes_per_row * t_steady_pass
            t_read_row_ns = L * t_clk          # one LOAD per row's fill
            t_write_row_ns = O * t_clk         # one OUTPUT per row's drain
            t_row_steady = max(t_read_row_ns, t_compute_row_ns, t_write_row_ns)
            row_timing = {
                # Option A (Task #90): ideal per-row fill = L + C + O (no FSM
                # handoff overhead modeled; surfaces as fidelity).
                "per_row_total": (L + C_cyc + O) * t_clk + max(0, passes_per_row - 1) * t_steady_pass,
                "per_row_steady": t_row_steady,
                "per_row_read": t_read_row_ns,
                "per_row_compute": t_compute_row_ns,
                "per_row_write": t_write_row_ns,
                "first_output_ns": (L + C_cyc + O) * t_clk,
                "rows": M,
            }
            log(f"\tPer-row: passes/row={passes_per_row}, "
                f"compute={t_compute_row_ns:.1f} read={t_read_row_ns:.1f} "
                f"write={t_write_row_ns:.1f} → steady={t_row_steady:.1f} ns/row")
        else:
            row_timing = None

        return t_total_ns, e_total_pj, row_timing

    def gemm_clb(self, M, K, N):
        """CLB-based GEMM for architectures with 0 DSPs (Azure-Lily on DPE-heavy FPGA).

        Uses CLB LUTs for multiply-accumulate. Same structure as gemm_dsp but
        uses clb_pj_per_mac energy and estimates CLB MAC parallelism from total_clb.
        """
        assert M > 0 and K > 0 and N > 0, f"Illegal GEMM sizes: ({M, K, N})"
        CLB_PER_MAC = 12  # ~12 CLBs per multiply-accumulate unit
        n_mac_units = max(1, self.cfg.total_clb // CLB_PER_MAC)

        log(f"\tCLB-based GEMM({M}, {K}, {N}) with {n_mac_units} MAC units "
            f"({self.cfg.total_clb} CLBs / {CLB_PER_MAC})")

        # Memory read
        t_read_ns = self.memory.latency(M * K)
        e_read_pj = self.memory.energy(M * K)

        # Compute: M*N output elements, K MACs each
        total_macs = M * N * K
        e_gemm_pj = total_macs * self.cfg.e_clb_pj_per_mac
        self.stats.record_energy_breakdown("clb_multiply", e_gemm_pj)

        # Latency: ceil(M*N / n_mac_units) batches, K cycles per batch
        n_batches = math.ceil(M * N / n_mac_units)
        total_cycles = n_batches * K
        t_gemm_ns = cycles_to_ns(total_cycles, self.cfg.freq)

        # Memory write
        t_write_ns = self.memory.latency(M * N, read=False)
        e_write_pj = self.memory.energy(M * N)
        self.stats.record_energy_breakdown("sram_write", e_write_pj)

        t_total_ns = t_read_ns + t_gemm_ns + t_write_ns
        e_total_pj = e_read_pj + e_gemm_pj + e_write_pj
        log(f"\tCLB GEMM: {total_macs} MACs, {n_batches} batches, "
            f"latency {t_total_ns:.2f} ns, energy {e_total_pj:.2f} pj")
        return t_total_ns, e_total_pj

    def gemm_log(self, M, K, N, n_parallel_dpes=None):
        """Log-domain GEMM for NL-DPE DIMM (§5 DIMM workload, §4 pipeline,
        §7 W-lane row-parallel tiling — Pattern β shared B + broadcast).

        Algorithm per §5 + §7: A[M×K] × B[K×N] via log/exp, three phases:

          Phase 1a (log A) — per-lane parallel
              Each lane converts its ceil(M/W) rows of A to log domain.
              passes_per_lane_1a = ceil((ceil(M/W) × K) / C)

          Phase 1b (log B) — shared / broadcast
              One lane (or dedicated DPE) converts full B to log domain.
              Result broadcast to all W lanes via Pattern-β shared bus.
              passes_1b = ceil((K × N) / C)
              All lanes wait for this before phase 3 starts.

          Phase 2 (CLB add)  — off-DPE, energy-only
              M × N × K element-wise adds (log_A + log_B) on CLB hardware.

          Phase 3 (exp + sum) — per-lane parallel
              Each lane computes its share of (exp(log_A + log_B), summed
              over K) via DPE(I|exp) + private K-wide tree.
              passes_per_lane_3 = ceil((ceil(M/W) × N × K) / C)

        Per-lane critical path:
              passes_per_lane = phase_1a + phase_1b + phase_3   (sequential)
        Latency uses the §4 single-buffered drain-load overlap pipeline
        applied to passes_per_lane total. (Pipeline applied across the
        concatenated phase sequence; each pass is full-crossbar per §3.)

        Energy (DPE work, all phases):
              total_DPE_passes = W × phase_1a + phase_1b + W × phase_3
                              = (M × K)/C    + (K × N)/C  + (M × N × K)/C
              (since phase_1a-per-lane × W aggregates to M×K/C, etc.)

        Per-pass cost is full crossbar (C columns), regardless of how
        many slots are useful (Methodology §3).

        Args:
            M: output rows
            K: inner dimension (dot product length)
            N: output columns
            n_parallel_dpes: DPE+tree lane count (W). One lane owns
                ceil(M/W) rows of output (§7 row-parallel). If None,
                defaults to ``cfg.total_softmax_lanes`` (single allocation
                knob shared with mac_qk / softmax / mac_sv per §7).
        """
        assert self.cfg.analoge_nonlinear_support, (
            "We can only run DIMM on FPGA in LOG domain when IMC can perform non-linear operations"
        )
        if n_parallel_dpes is None:
            n_parallel_dpes = max(1, getattr(self.cfg, 'total_softmax_lanes', 1))
        n_lanes = max(1, int(n_parallel_dpes))
        C = self.cfg.cols

        total_output_elements = M * N
        rows_per_lane = math.ceil(M / n_lanes)

        # ── Memory I/O: each parallel DPE has its own BRAM ──
        read_bytes = M * K + K * N
        write_bytes = M * N
        bytes_per_port_read = math.ceil(read_bytes / n_lanes)
        bytes_per_port_write = math.ceil(write_bytes / n_lanes)
        t_read_ns = self.memory.latency(bytes_per_port_read)
        e_read_pj = self.memory.energy(read_bytes, record_breakdown=False)
        t_write_ns = self.memory.latency(bytes_per_port_write, read=False)
        e_write_pj = self.memory.energy(write_bytes, read=False, record_breakdown=False)
        self.stats.record_energy_breakdown("sram_read", e_read_pj)
        self.stats.record_energy_breakdown("sram_write", e_write_pj)
        log(f"\tMemory: read {read_bytes} bytes, write {write_bytes} bytes")
        log(f"\t\t read latency {t_read_ns:.2f} ns, energy {e_read_pj:.2f} pJ")
        log(f"\t\t write latency {t_write_ns:.2f} ns, energy {e_write_pj:.2f} pJ")

        # ── CLB add (Phase 2): M × N × K element-wise adds (off-DPE) ──
        per_add_pj = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add
        e_vec_add_pj = total_output_elements * K * per_add_pj
        self.stats.record_energy_breakdown("clb_add", e_vec_add_pj)

        # ── §7 three-phase DPE pass counts ──
        phase_1a_passes_per_lane = math.ceil(rows_per_lane * K / C)        # log A
        phase_1b_passes          = math.ceil(K * N / C)                    # log B (shared)
        phase_3_passes_per_lane  = math.ceil(rows_per_lane * N * K / C)    # exp+sum
        passes_per_lane = (phase_1a_passes_per_lane
                           + phase_1b_passes
                           + phase_3_passes_per_lane)

        # Total DPE passes across all lanes (energy basis, all phases):
        total_phase_1a_passes = n_lanes * phase_1a_passes_per_lane
        total_phase_1b_passes = phase_1b_passes  # shared, single-lane
        total_phase_3_passes  = n_lanes * phase_3_passes_per_lane

        # Per-pass DPE energy (full crossbar — methodology §3)
        if self.imc_core is not None and self.cfg.analoge_nonlinear_support:
            e_vmm_per_pass     = self.imc_core.energy_per_vmm()
            e_conv_per_pass    = self.imc_core.energy_per_conversion()
            e_digital_per_pass = self.imc_core.energy_per_digital_post()
            e_per_pass_pj      = e_vmm_per_pass + e_conv_per_pass + e_digital_per_pass

            # Phase 1a + Phase 1b → log conversions
            log_passes = total_phase_1a_passes + total_phase_1b_passes
            self.stats.record_energy_breakdown("imc_dimm_log_vmm",     log_passes * e_vmm_per_pass)
            self.stats.record_energy_breakdown("imc_dimm_log_conv",    log_passes * e_conv_per_pass)
            self.stats.record_energy_breakdown("imc_dimm_log_digital", log_passes * e_digital_per_pass)

            # Phase 3 → exp ops
            self.stats.record_energy_breakdown("imc_dimm_exp_vmm",     total_phase_3_passes * e_vmm_per_pass)
            self.stats.record_energy_breakdown("imc_dimm_exp_conv",    total_phase_3_passes * e_conv_per_pass)
            self.stats.record_energy_breakdown("imc_dimm_exp_digital", total_phase_3_passes * e_digital_per_pass)

            e_tot_log_pj = log_passes * e_per_pass_pj
            e_tot_exp_pj = total_phase_3_passes * e_per_pass_pj
            e_tot_dpe_pj = e_tot_log_pj + e_tot_exp_pj

            log(f"\tDPE phases: 1a={phase_1a_passes_per_lane} pass/lane×{n_lanes}lanes, "
                f"1b={phase_1b_passes} (shared), "
                f"3={phase_3_passes_per_lane} pass/lane×{n_lanes}lanes "
                f"(M={M},K={K},N={N}, C={C}, full-crossbar/pass per §3)")
        else:
            # Fallback for archs without analog non-linear support
            t_exp_per_elem_ns, e_exp_per_elem_pj = self.exp_fpga(K)
            e_tot_exp_pj = total_output_elements * e_exp_per_elem_pj
            e_tot_log_pj = 0.0
            e_tot_dpe_pj = e_tot_exp_pj
            self.stats.record_energy_breakdown("clb_exp", e_tot_exp_pj)

        # CLB reduction (per output element, K-wide adder tree)
        t_reduc_per_elem_ns, e_reduc_per_elem_pj = self._clb_reduction_energy_latency(K)
        e_tot_reduc_pj = total_output_elements * e_reduc_per_elem_pj
        self.stats.record_energy_breakdown("clb_reduction", e_tot_reduc_pj)

        # ── Latency: §4 single-buffered drain-load overlap pipeline ──
        # Per-lane critical path = phase_1a + phase_1b + phase_3 passes.
        # The §4 pipeline applies across the concatenated stream of passes
        # since every pass uses the same DPE-axiom (L, C_cyc, O).
        if self.imc_core:
            total_compute_cycles = self.imc_core._pipeline_total_cycles(passes_per_lane, workload="dimm")
            feed_cycles, compute_cycles, output_cycles = self.imc_core._pipeline_pass_cycles(workload="dimm")
            t_compute_ns = cycles_to_ns(total_compute_cycles, self.cfg.freq)
        else:
            feed_cycles = compute_cycles = output_cycles = 0
            total_compute_cycles = 0
            t_compute_ns = 0.0

        log(f"\tgemm_log({M},{K},{N}): {n_lanes} DPE lanes, "
            f"per-lane passes = 1a({phase_1a_passes_per_lane}) + 1b({phase_1b_passes}) + 3({phase_3_passes_per_lane}) = {passes_per_lane}, "
            f"per-pass L={feed_cycles} C={compute_cycles} O={output_cycles} "
            f"(§4 single-buffered drain-load overlap)")
        log(f"\t\t CLB add: {e_vec_add_pj:.2f} pJ, DPE log: {e_tot_log_pj:.2f} pJ, "
            f"DPE exp: {e_tot_exp_pj:.2f} pJ, CLB reduce: {e_tot_reduc_pj:.2f} pJ")
        log(f"\t\t compute latency: {t_compute_ns:.2f} ns")

        t_total_ns = t_read_ns + t_compute_ns + t_write_ns
        e_total_pj = e_read_pj + e_vec_add_pj + e_tot_dpe_pj + e_tot_reduc_pj + e_write_pj
        log(f"\tTotal GEMM-log: latency {t_total_ns:.2f} ns, energy {e_total_pj:.2f} pJ")

        # Per-row timing for streaming pipeline (used by attention pipeline
        # composition). passes_per_row = passes_per_lane / max(1, M) — average
        # per-output-row cost in the drain-load overlap regime.
        if M > 0 and self.imc_core is not None:
            passes_per_row = max(1, math.ceil(passes_per_lane / max(1, M)))
            cycles_per_row = self.imc_core._pipeline_total_cycles(passes_per_row, workload="dimm")
            t_compute_row_ns = cycles_to_ns(cycles_per_row, self.cfg.freq)
        else:
            t_compute_row_ns = t_compute_ns / max(1, M)
        t_read_row_ns = self.memory.latency(math.ceil((K + N) / n_lanes))
        t_write_row_ns = self.memory.latency(math.ceil(N / n_lanes), read=False)
        t_row_total = t_read_row_ns + t_compute_row_ns + t_write_row_ns
        t_row_steady = max(t_read_row_ns, t_compute_row_ns, t_write_row_ns)
        row_timing = {
            "per_row_total": t_row_total,
            "per_row_steady": t_row_steady,
            "per_row_read": t_read_row_ns,
            "per_row_compute": t_compute_row_ns,
            "per_row_write": t_write_row_ns,
            "rows": M,
        }
        log(f"\tPer-row: read={t_read_row_ns:.1f} compute={t_compute_row_ns:.1f} "
            f"write={t_write_row_ns:.1f} → steady={t_row_steady:.1f} ns/row")

        return t_total_ns, e_total_pj, row_timing

    def exp_fpga(self, vec_length):
        assert vec_length > 0, f"Illegal softmax vector length {vec_length}"
        INPUT_WIDTH = 8
        LUT_PER_CLB = 8
        OUTPUT_WIDTH = 16
        LUT_WIDTH = 6
        total_store_values = 2 ** INPUT_WIDTH
        total_store_bits = OUTPUT_WIDTH * total_store_values
        total_luts = math.ceil(total_store_bits / (2 ** LUT_WIDTH))
        total_clbs = math.ceil(total_luts / LUT_PER_CLB)
        assert total_clbs < self.cfg.total_clb, (
            f"No enough CLBs on FPGA, required {total_clbs} CLBs, {self.cfg.total_clb} CLBs available"
        )
        self.stats.record_resource("clb_used", total_clbs, peak=True)
        latency_ns = cycles_to_ns(1, self.cfg.freq)
        energy_pj = total_clbs * self.cfg.e_clb_pj_per_mac * 0.4
        return latency_ns, energy_pj

    def norm_fpga(self, vec_length):
        assert vec_length > 0, f"Illegal softmax vector length {vec_length}"
        e_sum_pj, t_sum_ns = self._clb_reduction_energy_latency(vec_length)

        INV_INPUT_WIDTH = 8
        INV_OUTPUT_WIDTH = 16
        LUT_WIDTH = 6
        LUT_PER_CLB = 8
        total_store_values = 2 ** INV_INPUT_WIDTH
        total_store_bits = INV_OUTPUT_WIDTH * total_store_values
        total_luts = math.ceil(total_store_bits / (2 ** LUT_WIDTH))
        inv_clbs = math.ceil(total_luts / LUT_PER_CLB)
        assert inv_clbs < self.cfg.total_clb, (
            f"No enough CLBs on FPGA, required {inv_clbs} CLBs, {self.cfg.total_clb} CLBs available"
        )
        self.stats.record_resource("clb_used", max(inv_clbs, max(0, vec_length - 1)), peak=True)
        t_inv_ns = cycles_to_ns(1, self.cfg.freq)
        e_inv_pj = inv_clbs * self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_compare

        if self.cfg.total_dsp > 0:
            parallelism = min(self.cfg.total_dsp, vec_length)
            mul_cycles = math.ceil(vec_length / parallelism)
            t_mul_ns = cycles_to_ns(mul_cycles, self.cfg.freq)
            e_mul_pj = vec_length * self.cfg.e_dsp_pj_per_mac
            self.stats.record_resource("dsp_used", parallelism, peak=True)
        else:
            t_mul_ns = cycles_to_ns(vec_length, self.cfg.freq)
            e_mul_pj = vec_length * self.cfg.e_clb_pj_per_mac

        latency_ns = t_sum_ns + t_inv_ns + t_mul_ns
        energy_pj = e_sum_pj + e_inv_pj + e_mul_pj
        return latency_ns, energy_pj, e_sum_pj, e_inv_pj, e_mul_pj

    def maxpool(self, layer, return_profile=False):
        window_elements = layer.kernel_size * layer.kernel_size
        compare_tree_levels = math.ceil(math.log2(window_elements))
        output_positions = layer.output_height * layer.output_width * layer.num_inputs
        
        reads_per_output = window_elements * layer.in_channels
        writes_per_output = layer.out_channels
        
        total_reads = reads_per_output * output_positions
        total_writes = writes_per_output * output_positions
        total_outputs = output_positions * layer.out_channels

        log(f"=== Running maxpool layer ===")
        log(f"\tReading {total_reads} bytes from BRAM")
        t_read_ns = self.memory.latency(reads_per_output, read=True)
        e_read_pj = self.memory.energy(total_reads, read=True)
        log(f"\t\tread latency {t_read_ns:.2f} ns, read energy {e_read_pj:.2f} pj")

        log(f"\tComparing {window_elements} elements across {layer.out_channels} channels")
        compare_parallel = max(1, min(layer.out_channels, self.cfg.total_clb))
        compare_batches = math.ceil(layer.out_channels / compare_parallel)
        t_compare_ns = cycles_to_ns(compare_tree_levels * compare_batches, self.cfg.freq)
        compare_ops = (2 ** compare_tree_levels - 1) * total_outputs
        e_compare_pj = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_compare * compare_ops
        self.stats.record_energy_breakdown("clb_compare", e_compare_pj)
        self.stats.record_resource("clb_used", compare_parallel, peak=True)
        # Pool layer has 1 BRAM for window buffer
        self.stats.record_resource("memory_blocks", 1, peak=True)
        log(f"\t\tcompare latency {t_compare_ns:.2f} ns, compare energy {e_compare_pj:.2f} pj")

        log(f"\tWriting {total_writes} bytes to BRAM")
        t_write_ns = self.memory.latency(writes_per_output, read=False)
        e_write_pj = self.memory.energy(total_writes, read=False)
        log(f"\t\twrite latency {t_write_ns:.2f} ns, write energy {e_write_pj:.2f} pj")

        t_fill = t_read_ns + t_compare_ns + t_write_ns
        t_steady = max(t_read_ns, t_compare_ns, t_write_ns)
        latency = t_fill + max(0, output_positions - 1) * t_steady
        log(f"\tPipelined latency: {latency:.2f} ns (steady-state {t_steady:.2f} ns)")
        energy = e_read_pj + e_compare_pj + e_write_pj
        log(f"=== maxpool layer: latency {latency:.2f}ns, energy {energy:.2f}pj ===\n")
        if return_profile:
            return latency, energy, {
                "first_output_ns": t_fill,
                "fill": t_fill,
                "steady": t_steady,
                "events": output_positions,
            }
        return latency, energy

    def layernorm(self, normalized_shape, seq_len):
        """LayerNorm over seq_len tokens, each of dimension normalized_shape.

        Per-token operations (all CLB):
          1. Mean: reduction tree (sum of normalized_shape elements) + divide
          2. Variance: subtract mean (n ops) + square (n ops) + reduction tree + divide
          3. rsqrt LUT: same CLB cost as inverse in norm_fpga (8-bit in, 16-bit out)
          4. Normalize: subtract mean (n), multiply rsqrt (n), scale gamma (n), shift beta (n)

        Returns (total_latency_ns, total_energy_pj).
        """
        n = normalized_shape

        # --- Mean: reduction(n) ---
        e_mean_reduc, t_mean_reduc = self._clb_reduction_energy_latency(n)
        # divide by n is a shift/multiply — 1 CLB op
        e_mean_div = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add
        t_mean_div = cycles_to_ns(1, self.cfg.freq)

        # --- Variance: n subtracts + n squares + reduction(n) + divide ---
        per_op_add = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add
        per_op_mul = self.cfg.e_clb_pj_per_mac  # multiply ~ 1 MAC

        e_var_sub = n * per_op_add          # subtract mean from each element
        e_var_sq = n * per_op_mul           # square each element
        e_var_reduc, t_var_reduc = self._clb_reduction_energy_latency(n)
        e_var_div = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add

        parallel_ops = max(1, self.cfg.total_clb)
        t_var_eltwise = cycles_to_ns(math.ceil(2 * n / parallel_ops), self.cfg.freq)

        # --- rsqrt LUT: same cost model as inverse in norm_fpga ---
        INV_INPUT_WIDTH = 8
        INV_OUTPUT_WIDTH = 16
        LUT_WIDTH = 6
        LUT_PER_CLB = 8
        total_store_values = 2 ** INV_INPUT_WIDTH
        total_store_bits = INV_OUTPUT_WIDTH * total_store_values
        total_luts = math.ceil(total_store_bits / (2 ** LUT_WIDTH))
        rsqrt_clbs = math.ceil(total_luts / LUT_PER_CLB)
        t_rsqrt = cycles_to_ns(1, self.cfg.freq)
        e_rsqrt = rsqrt_clbs * self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_compare

        # --- Normalize: subtract(n) + mul_rsqrt(n) + mul_gamma(n) + add_beta(n) = 4n ops ---
        e_norm_ops = n * (2 * per_op_add + 2 * per_op_mul)  # 2 adds + 2 muls per element
        t_norm_ops = cycles_to_ns(math.ceil(4 * n / parallel_ops), self.cfg.freq)

        # --- Memory: read input + write output per token ---
        read_bytes = n   # read one token
        write_bytes = n  # write normalized token
        t_read = self.memory.latency(read_bytes)
        e_read = self.memory.energy(read_bytes)
        t_write = self.memory.latency(write_bytes, read=False)
        e_write = self.memory.energy(write_bytes, read=False)

        # Per-token totals
        per_token_energy = (e_mean_reduc + e_mean_div +
                           e_var_sub + e_var_sq + e_var_reduc + e_var_div +
                           e_rsqrt +
                           e_norm_ops +
                           e_read + e_write)
        per_token_latency = (t_mean_reduc + t_mean_div +
                            t_var_eltwise + t_var_reduc +
                            t_rsqrt +
                            t_norm_ops +
                            t_read + t_write)

        total_energy = seq_len * per_token_energy
        total_latency = seq_len * per_token_latency

        # Record breakdown — CLB compute portion only (memory already recorded by memory model)
        clb_energy = total_energy - seq_len * (e_read + e_write)
        self.stats.record_energy_breakdown("clb_layernorm", clb_energy)
        self.stats.record_resource("clb_used", min(parallel_ops, 4 * n), peak=True)

        log(f"\tLayerNorm({n}) x {seq_len} tokens: "
            f"latency {total_latency:.2f} ns, energy {total_energy:.2f} pJ")
        return total_latency, total_energy

    def embedding_lookup(self, seq_len, d_model):
        """BERT-style embedding: token + position + segment lookups + add.

        - Read: 3 embedding vectors per token from SRAM
        - Add: 2 element-wise adds per element (token+pos, then +seg)
        - Write: result to SRAM

        Returns (total_latency_ns, total_energy_pj).
        """
        # Memory: 3 reads + 1 write per token, each d_model bytes
        read_bytes = 3 * seq_len * d_model
        write_bytes = seq_len * d_model

        t_read = self.memory.latency(read_bytes)
        e_read = self.memory.energy(read_bytes)
        t_write = self.memory.latency(write_bytes, read=False)
        e_write = self.memory.energy(write_bytes, read=False)

        # CLB: 2 adds per element (token+pos, then +seg)
        total_add_ops = 2 * seq_len * d_model
        per_op_pj = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add
        e_add = total_add_ops * per_op_pj
        parallel_ops = max(1, self.cfg.total_clb)
        t_add = cycles_to_ns(math.ceil(total_add_ops / parallel_ops), self.cfg.freq)

        self.stats.record_energy_breakdown("clb_embed_add", e_add)
        self.stats.record_resource("clb_used", min(parallel_ops, total_add_ops), peak=True)

        total_latency = t_read + t_add + t_write
        total_energy = e_read + e_add + e_write

        log(f"\tEmbedding lookup: {seq_len} tokens x {d_model}d, "
            f"latency {total_latency:.2f} ns, energy {total_energy:.2f} pJ")
        return total_latency, total_energy

    def _clb_reduction_energy_latency(self, reductions):
        if reductions <= 1:
            return 0.0, 0.0
        levels = math.ceil(math.log2(reductions))
        total_clbs = reductions - 1
        energy = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add * total_clbs
        latency = cycles_to_ns(levels, self.cfg.freq)
        return energy, latency
