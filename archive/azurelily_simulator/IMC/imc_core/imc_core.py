from __future__ import annotations

import math

from scheduler_stats.common import log


class IMCCore:
    def __init__(self, cfg, memory, stats):
        self.cfg = cfg
        self.memory = memory
        self.stats = stats

    def analoge_nonlinear_check(self, M, K, N):
        k_tile = math.ceil(K / self.cfg.rows)
        if self.cfg.imc == "NL-DPE" and k_tile > 1:
            return False
        return self.cfg.analoge_nonlinear_support

    def _get_arch_factors(self):
        return self.cfg._get_arch_factors()

    def latency_per_vmm(self):
        k_slicing, _, _ = self._get_arch_factors()
        return k_slicing * self.cfg.t_analoge_ns

    def energy_per_vmm(self):
        k_slicing, _, _ = self._get_arch_factors()
        return k_slicing * self.cfg.e_analoge_pj

    def latency_per_conversion(self):
        _, k_adc, _ = self._get_arch_factors()
        return k_adc * self.cfg.t_conv_ns

    def energy_per_conversion(self, active_cols=None):
        _, k_adc, _ = self._get_arch_factors()
        if active_cols is None:
            return k_adc * self.cfg.e_conv_pj
        active_cols = max(1, min(active_cols, self.cfg.cols))
        util = active_cols / max(1, self.cfg.cols)
        return k_adc * self.cfg.e_conv_pj * util

    def latency_per_digital_post(self):
        _, _, k_accum = self._get_arch_factors()
        return k_accum * self.cfg.t_digital_ns

    def energy_per_digital_post(self, active_cols=None):
        """
        Energy for the digital post-processing stage.

        For NL-DPE: e_digital_pj is per-column ACAM energy (scale_with_geometry=false).
        The ACAM is part of the static analog macro — all cols fire regardless of
        utilization, so energy = k_accum * e_digital_pj * cols (always full crossbar).

        For Azure-Lily: e_digital_pj=0, so this doesn't matter.
        """
        _, _, k_accum = self._get_arch_factors()
        if not self.cfg.scale_with_geometry:
            return k_accum * self.cfg.e_digital_pj * self.cfg.cols
        return k_accum * self.cfg.e_digital_pj

    def _pipeline_pass_cycles(self, workload="vmm"):
        """Per-pass DPE-axiom (LOAD, COMPUTE, OUTPUT) cycle counts (§3, §4).

        LOAD   = ceil(load_dim × 8 / dpe_buf_width)
                   load_dim = R for VMM (input vector of R elements)
                   load_dim = C for DIMM (C × C identity, input is C elems)
        COMPUTE= core bit-serial pipeline minus output stage
        OUTPUT = ceil(C × 8 / dpe_buf_width)        (DPE → SRAM)

        Per §3, all three are derived purely from per-arch config — no
        hand-pinned constants. workload selects which crossbar dimension
        drives LOAD.
        """
        t_clk = 1e3 / self.cfg.freq  # ns per cycle
        dpe_bw = getattr(self.cfg, 'dpe_buf_width', self.cfg.bram_width)
        load_dim = self.cfg.rows if workload == "vmm" else self.cfg.cols
        feed_cycles = math.ceil(load_dim * 8 / dpe_bw)
        output_cycles = math.ceil(self.cfg.cols * 8 / dpe_bw)
        core_ns = self._core_bit_pipeline_row_latency()
        compute_cycles = max(1, math.ceil((core_ns - output_cycles * t_clk) / t_clk))
        return feed_cycles, compute_cycles, output_cycles

    def _pipeline_total_cycles(self, num_passes, workload="vmm"):
        """§4 double-buffered drain-load overlap pipeline (Task #99).

            T(M)           = T_fill_ideal + (M − 1) × T_steady
            T_fill_ideal   = L + C + O
            T_steady       = max(L, C, O)

        Task #99 (double-buffered LOAD): each DPE owns TWO input-buffer
        substrates A/B; pass-(k+1) LOAD writes to substrate B while
        pass-k COMPUTE reads substrate A. The single-substrate LOAD-gate
        of Option A1 is removed, dropping +PRECISION from T_steady. The
        double-buffer flop cost is small (PRECISION × R extra bits per
        DPE; for R=256/P=8 that's +2048 flops) and is already accounted
        for in the architectural area budget.

        For typical configs:
          NL  (R=256, BUF=40, P=8): L=52, max(52,10,52) → T_steady=52.
          AL  (R=512, BUF=16, P=8): L=256, max(256,10,64) → T_steady=256.
        """
        feed, compute, output = self._pipeline_pass_cycles(workload=workload)
        t_fill = feed + compute + output
        # Task #99: double-buffered LOAD — pass-(k+1) writes to the free
        # substrate while pass-k COMPUTE reads the other. No LOAD-gate.
        t_steady = max(feed, compute, output)
        return t_fill + max(0, num_passes - 1) * t_steady

    def _pipeline_total_cycles_explicit(self, num_passes, feed, compute, output,
                                        precision=None):
        """§4 double-buffered drain-load overlap with explicit per-pass L/C/O.

        Task #99 formula:
            T(M)           = T_fill_ideal + (M − 1) × T_steady
            T_fill_ideal   = L + C + O
            T_steady       = max(L, C, O)

        Used by primitives whose per-pass cycle counts depend on per-call
        parameters (e.g. AL DSP-MAC's K-tile and DSP_WIDTH) that the
        static `_pipeline_pass_cycles` helper cannot know.

        Args:
            precision: kept for API back-compat; ignored under Task #99
                (no LOAD-gate, no +PRECISION term).
        """
        t_fill = feed + compute + output
        t_steady = max(feed, compute, output)
        return t_fill + max(0, num_passes - 1) * t_steady

    def dimm_nonlinear(self, vec_length, op="exp", record_breakdown=True):
        """Energy and latency for DPE(I|op): identity crossbar + ACAM (§3, §4).

        The crossbar stores C × C identity weights; ACAM is configured
        for `op` (exp or log). Per pass: full crossbar fire — C columns,
        ACAM on all C, full-precision tree reduction. Per-pass cost
        is full crossbar + full tree regardless of how many slots are
        useful. (Methodology §3.)

        For vec_length > C, multiple passes are needed:
            num_passes = ceil(vec_length / C)
        Latency uses the §4 single-buffered drain-load overlap pipeline
        T(num_passes) = (L + C_cyc + O) + (num_passes − 1) × max(L, C_cyc, O).

        Args:
            vec_length: number of elements to process (e.g., d for exp over
                        a d-dimensional vector)
            op: "exp" or "log" (same hardware, different ACAM programming)
            record_breakdown: if True, record energy in stats breakdown

        Returns: (latency_ns, energy_pj)
        """
        assert vec_length > 0, f"dimm_nonlinear: vec_length must be > 0, got {vec_length}"
        num_passes = math.ceil(vec_length / self.cfg.cols)

        # Per-pass energy = one full DPE fire (VMM + conversion + ACAM on all C cols)
        e_vmm_per_pass = self.energy_per_vmm()
        e_conv_per_pass = self.energy_per_conversion()
        e_digital_per_pass = self.energy_per_digital_post()
        e_per_pass = e_vmm_per_pass + e_conv_per_pass + e_digital_per_pass

        # §4 pipeline: T(M) = T_fill + (M−1) × T_steady, derived from DPE-axiom.
        t_clk = 1e3 / self.cfg.freq  # ns per cycle
        feed_cycles, compute_cycles, output_cycles = self._pipeline_pass_cycles(workload="dimm")
        total_cycles = self._pipeline_total_cycles(num_passes, workload="dimm")
        total_latency = total_cycles * t_clk
        total_energy = num_passes * e_per_pass

        if record_breakdown:
            # Split into 3 sub-keys for 5-category energy breakdown:
            # vmm → Crossbar, conv → ADC, digital → ACAM
            self.stats.record_energy_breakdown(f"imc_dimm_{op}_vmm", num_passes * e_vmm_per_pass)
            self.stats.record_energy_breakdown(f"imc_dimm_{op}_conv", num_passes * e_conv_per_pass)
            self.stats.record_energy_breakdown(f"imc_dimm_{op}_digital", num_passes * e_digital_per_pass)

        log(f"\tDPE(I|{op}) on {vec_length} elements ({num_passes} passes, "
            f"L={feed_cycles} C={compute_cycles} O={output_cycles}, "
            f"§4 double-buffered drain-load overlap pipeline T_steady=max(L,C,O)): "
            f"latency {total_latency:.2f} ns, energy {total_energy:.2f} pJ")
        return total_latency, total_energy

    def clb_reduction_energy_latency(self, reductions, active_cols=1):
        if reductions <= 1 or active_cols <= 0:
            return 0.0, 0.0

        levels = math.ceil(math.log2(reductions))
        clbs_per_col = reductions - 1
        total_add_ops = clbs_per_col * active_cols
        energy = self.cfg.e_clb_pj_per_mac * self.cfg.clb_coeff_add * total_add_ops

        total_clb = max(1, int(getattr(self.cfg, "total_clb", 1)))
        parallel_cols = max(1, total_clb // clbs_per_col)
        col_batches = math.ceil(active_cols / parallel_cols)
        latency = col_batches * levels * (1e9 / (self.cfg.freq * 1e6))
        return energy, latency

    def _dpe_buf_fill_row(self, K_bytes):
        """BRAM → DPE input buffer transfer time per DPE (ns).

        Each DPE loads min(K, R) int8 values from BRAM through the
        dpe_buf_width-bit interface (one-time, before 8 bit-slices fire).
        When K < R, zero-skip: only K values loaded (unused rows zero-padded
        internally). When K > R, each DPE loads at most R values (vertical
        tiling: V DPEs load in parallel, each handling its R-row portion).

        NL-DPE  (R=1024, dpe_buf_width=40, K=128): min(128,1024)=128 → ceil(128*8/40) = 26 acc
        Azure-Lily (R=512, dpe_buf_width=16, K=128): min(128,512)=128 → ceil(128*8/16) = 64 acc
        NL-DPE  (R=1024, dpe_buf_width=40, K=2048): min(2048,1024)=1024 → ceil(1024*8/40) = 205 acc
        """
        dpe_bw = getattr(self.cfg, 'dpe_buf_width', self.cfg.bram_width)
        elems_per_dpe = min(K_bytes, self.cfg.rows)  # each DPE loads at most R values
        n_access = math.ceil(elems_per_dpe * 8 / dpe_bw)
        return n_access * (1e3 / self.cfg.freq)  # ns (freq in MHz)

    def _core_bit_pipeline_row_latency(self):
        """
        Row latency for core compute path, modeled as a bit-serial pipeline.

        Full DPE pipeline per VMM pass:
          [buf_fill] → [8 bit-slices: VMM + Accum/ADC] → [ACAM] → [output serialize]

        The buf_fill is handled by _dpe_buf_fill_row() (separate).
        The output serialization (C columns × int8 through dpe_buf_width)
        is included here as a sequential stage after the VMM compute.

        Downstream stages (reduction, activation, BRAM write) are pipelined
        with the output stream — no BRAM round-trip between them.
        """
        k_vmm, k_adc, k_digital = self._get_arch_factors()
        t_vmm = self.latency_per_vmm()
        t_adc = self.latency_per_conversion()
        t_digital = self.latency_per_digital_post()

        # DPE output serialization: C columns × int8 through dpe_buf_width port
        dpe_bw = getattr(self.cfg, 'dpe_buf_width', self.cfg.bram_width)
        t_output = math.ceil(self.cfg.cols * 8 / dpe_bw) * (1e3 / self.cfg.freq)

        # NL-DPE: 2 pipelinable stages (DAC→Crossbar, Analog Accum) + ACAM + output
        if self.cfg.analoge_accum and self.cfg.input_bs and not self.cfg.digital_accum:
            bits = self.cfg.n_in  # 8
            t_vmm_bit = t_vmm / k_vmm
            t_conv_bit = t_adc / k_adc
            fill = t_vmm_bit + t_conv_bit
            steady = max(t_vmm_bit, t_conv_bit)
            t_acam = t_digital
            return fill + max(0, bits - 1) * steady + t_acam + t_output

        # Azure-Lily: digital accumulation, 4-stage bit-serial pipeline + output
        if k_vmm == k_adc and k_vmm > 1 and k_digital in (0, k_vmm):
            bits = k_vmm
            t_vmm_bit = t_vmm / k_vmm if k_vmm > 0 else 0.0
            t_adc_bit = t_adc / k_adc if k_adc > 0 else 0.0
            t_digital_bit = (t_digital / k_digital) if k_digital > 0 else 0.0
            fill = t_vmm_bit + t_adc_bit + t_digital_bit
            steady = max(t_vmm_bit, t_adc_bit, t_digital_bit)
            return fill + max(0, bits - 1) * steady + t_output

        # Default fallback
        return t_vmm + t_adc + t_digital + t_output

    def _bram_latency_row(self, bytes_):
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
        return num_cycle * (1e9 / (self.cfg.freq * 1e6))

    def _streaming_read_latency_row(self, kernel_size, in_channels):
        """
        Per-position read latency matching NN simulator's access pattern:
        reads k*k spatial positions, each reading ceil(C_in / elems_per_mv)
        accesses from SRAM.  This matches the RTL's serial streaming interface.
        """
        elems_per_mv = max(1, math.floor(self.cfg.bram_width / 8))
        reads_per_pos = math.ceil(in_channels / elems_per_mv)
        num_access = reads_per_pos * kernel_size * kernel_size
        cycle_ns = 1e9 / (self.cfg.freq * 1e6)
        if self.cfg.bram_mode == self.cfg.SP:
            num_cycle = num_access
        elif self.cfg.bram_mode == self.cfg.TDP:
            num_cycle = math.ceil(num_access / 2)
        else:
            num_cycle = num_access
        return num_cycle * cycle_ns

    def gemm_pipeline_profile(self, K, N, kernel_size=None, in_channels=None,
                              use_dpe_buf=True):
        """Pipeline profile for one GEMM. If use_dpe_buf=True (default for DPE
        operations), the read uses dpe_buf_width instead of bram_width."""
        k_tile = math.ceil(K / self.cfg.rows)

        # Read latency: SRAM → DPE input buffer through dpe_buf_width.
        # Each DPE has its own SRAM; all DPEs load in parallel.
        # The BRAM→SRAM fill is accounted for in the write stage of the
        # previous layer, so we only model the SRAM→DPE transfer here.
        if use_dpe_buf:
            t_read_row = self._dpe_buf_fill_row(K)
        elif kernel_size is not None and in_channels is not None:
            t_read_row = self._streaming_read_latency_row(kernel_size, in_channels)
        else:
            t_read_row = self._bram_latency_row(K)

        if use_dpe_buf:
            # Each DPE outputs min(N, C) values through dpe_buf_width interface
            t_write_row = self._dpe_buf_fill_row(min(N, self.cfg.cols))
        else:
            t_write_row = self._bram_latency_row(N)
        t_core_row = self._core_bit_pipeline_row_latency()
        t_reduc_row = self.clb_reduction_energy_latency(k_tile, active_cols=N)[1]

        if self.cfg.pipelinable:
            t_fill = t_read_row + t_core_row + t_reduc_row + t_write_row
            t_steady = max(t_read_row, t_core_row, t_reduc_row, t_write_row)
            t_compute_row = max(t_core_row, t_reduc_row)
        else:
            t_compute_row = t_core_row + t_reduc_row
            t_fill = t_read_row + t_compute_row + t_write_row
            t_steady = max(t_read_row, t_compute_row, t_write_row)

        return {
            "read_row": t_read_row,
            "core_row": t_core_row,
            "reduction_row": t_reduc_row,
            "compute_row": t_compute_row,
            "write_row": t_write_row,
            "first_output_ns": t_fill,
            "fill": t_fill,
            "steady": t_steady,
        }

    def run_gemm(self, M, K, N, kernel_size=None, in_channels=None,
                 activation_mode=False):
        """Run a VMM GEMM workload on the IMC fabric (Path A weight-stationary).

        Path A model (committed for FC/GEMM, see fc_verification/FC_GEMM_WALKTHROUGH.md
        and FIDELITY_METHODOLOGY.md §5):
            V = ceil(K / R)        # K-axis tiles
            H = ceil(N / C)        # N-axis tiles
            n_parallel_dpes = V * H
            passes_per_dpe  = M * ceil(V*H / n_parallel_dpes) = M
        i.e. each of V·H DPEs is weight-stationary, holds one tile of W
        permanently, and fires exactly once per output row m. Total parallel
        firings per row = V·H.

        Unified cycle formula (Task #99 — double-buffered LOAD):
            T(M) = T_fill + (M - 1) * T_steady
            T_fill   = LCYC + CCYC + OCYC                ← architectural minimum
            T_steady = max(LCYC, CCYC, OCYC)
            total_cycles = T(M)                          ← single clean formula

        Task #99: the input substrate is now double-buffered (A/B
        ping-pong), so pass-(k+1) LOAD overlaps pass-k COMPUTE without
        a LOAD-gate. The +PRECISION term from Option A1 is removed.

        Removed from the sim's analytical output (Task #98):
          - TREE_PIPE   = ⌈log₂(V)⌉ for V > 1 — adder tree pipeline depth.
                          A real architectural cost of the synthesizable RTL,
                          but reported as a DELTA, not absorbed into sim_exp.
          - CLB_NEEDED  = 1 if (V > 1) OR (activation_mode AND not has_acam) —
                          CLB activation LUT cycle. Likewise a real RTL cost
                          reported as a delta.
          - +2 NBA handoffs (LOAD→COMPUTE wake, COMPUTE→OUTPUT wake) —
                          primitive-FSM cost, reported as a delta.
          - +4..+6 wrapper plumbing (BRAM read pipeline, registered DPE
                          handshake, sign-extend latch, BRAM-write tap,
                          BRAM internal write commit, last-strobe drive) —
                          fc_top wrapper cost, reported as a delta.

        All five categories are real silicon costs paid by the RTL; they
        surface as the rtl_obs − sim_exp delta, broken down per-stage in
        the workload smoke harness (see run_fc_smoke.py).

        Args:
            M, K, N: matmul dimensions Y[M×N] = X[M×K] @ W[K×N].
            kernel_size, in_channels: optional layer geometry for streaming
                read modeling (used by conv2d).
            activation_mode: whether the workload requests a non-identity
                activation (e.g. ReLU). Still threaded through for energy
                accounting and downstream consumers; the cycle-count gate
                is now reported as a delta rather than baked into the model.
        """
        assert M > 0 and K > 0 and N > 0, f"Illegal GEMM/GEMV sizes: ({M,K,N})"
        log(f"=== Running linear layer in {M, K, N} GEMM form on IMC===")

        k_tile = math.ceil(K / self.cfg.rows)
        n_tile = math.ceil(N / self.cfg.cols)
        total_imc = k_tile * n_tile
        # Each DPE tile has its own BRAM — parallel memory access
        n_parallel_ports = max(1, total_imc)

        # Use streaming read pattern when layer geometry is known
        total_read_bytes = M * K
        if kernel_size is not None and in_channels is not None:
            per_row_read = self._streaming_read_latency_row(kernel_size, in_channels)
            t_read_ns = per_row_read * M / n_parallel_ports
            e_read_pj = self.memory.energy(total_read_bytes)
        else:
            bytes_per_port = math.ceil(total_read_bytes / n_parallel_ports)
            t_read_ns = self.memory.latency(bytes_per_port)
            e_read_pj = self.memory.energy(total_read_bytes)
        log(f"\tReading {M} x {K} bytes from BRAM ({n_parallel_ports} parallel ports)\t")
        log(f"\t\tread latency {t_read_ns:.2f} ns, read energy {e_read_pj:.2f} pj")
        self.stats.record_resource("imc_tiles", total_imc, peak=True)

        reduction_per_vert_line = k_tile
        e_reduc_pj, t_reduc_ns = self.clb_reduction_energy_latency(
            reduction_per_vert_line, active_cols=N
        )
        self.stats.record_energy_breakdown("clb_reduction", e_reduc_pj)
        clbs_per_col = max(0, reduction_per_vert_line - 1)
        if clbs_per_col > 0:
            total_clb = max(1, int(getattr(self.cfg, "total_clb", 1)))
            parallel_cols = max(1, total_clb // clbs_per_col)
            cols_in_parallel = min(N, parallel_cols)
            reduction_clbs_peak = clbs_per_col * cols_in_parallel
            self.stats.record_resource("clb_used", reduction_clbs_peak, peak=True)

        log(f"\tPerforming {reduction_per_vert_line} reduction per col on total {N} cols")
        log(f"\t\treduction latency {t_reduc_ns:.2f} ns, reduction energy {e_reduc_pj:.2f} pj")

        t_vmm_row = self.latency_per_vmm()
        t_vmm_ns = M * t_vmm_row
        e_vmm_pj = M * total_imc * self.energy_per_vmm()
        self.stats.record_energy_breakdown("imc_vmm", e_vmm_pj)
        log(f"\tRunning  {M} VMM of {self.cfg.rows, self.cfg.cols} on IMC")
        log(f"\t\tVMM on IMC: latency {t_vmm_ns:.2f}ns, energy {e_vmm_pj:.2f}pj")

        t_conv_row = self.latency_per_conversion()
        t_conv_ns = M * t_conv_row
        e_conv_pj = M * total_imc * self.energy_per_conversion(active_cols=min(self.cfg.cols, N))
        self.stats.record_energy_breakdown("imc_conversion", e_conv_pj)
        log(f"\tConversion: latency {t_conv_ns:.2f}ns, energy {e_conv_pj:.2f}pj")

        t_digital_row = self.latency_per_digital_post()
        t_digital_post_ns = M * t_digital_row
        e_digital_post_pj = M * total_imc * self.energy_per_digital_post()
        self.stats.record_energy_breakdown("imc_digital_post", e_digital_post_pj)
        log(f"\tDigital Post Process: latency {t_digital_post_ns:.2f}ns, energy {e_digital_post_pj:.2f}pj")

        write_bytes_per_port = math.ceil(M * N / n_parallel_ports)
        t_write_ns = self.memory.latency(write_bytes_per_port, read=False)
        e_write_pj = self.memory.energy(M * N, read=False)
        log(f"\tWriting {M} x {N} bytes to BRAM ({n_parallel_ports} parallel ports)")
        log(f"\t\twrite latency {t_write_ns:.2f} ns, write energy {e_write_pj:.2f} pj")

        # §5 VMM workload tiling + §4 pipeline model — Path A weight-stationary.
        # V = ceil(K/R), H = ceil(N/C). n_parallel_dpes = V*H DPEs in silicon,
        # each weight-stationary. Per output row m, ALL V·H DPEs fire once in
        # parallel. Per-DPE firing count = M (one fire per output row).
        # passes_per_dpe = M * ceil(V*H / n_parallel_dpes) = M (always, since
        # n_parallel_dpes >= V*H by construction). See FC_GEMM_WALKTHROUGH §1.
        n_parallel_dpes = max(1, k_tile * n_tile)
        passes_per_dpe = M * math.ceil(k_tile * n_tile / n_parallel_dpes)
        # Unified cycle formula (Task #99 — double-buffered LOAD):
        #   T_fill   = LCYC + CCYC + OCYC
        #   T_steady = max(LCYC, CCYC, OCYC)
        #   T(M)     = T_fill + (M − 1) · T_steady
        # Double-buffered LOAD: pass-(k+1) writes to the free substrate
        # while pass-k COMPUTE reads the other. No LOAD-gate, no +PRECISION.
        #
        # Removed from the analytical output (Task #98):
        #   - TREE_PIPE   = ⌈log₂(V)⌉ for V > 1 — adder tree depth.
        #   - CLB_NEEDED  = 1 if (V > 1) OR (act AND !has_acam) — CLB cycle.
        #   - +2 NBA handoffs in the primitive.
        #   - +4..+6 wrapper plumbing in fc_top.v.
        # All are real silicon costs paid by the RTL; they surface as the
        # rtl_obs − sim_exp delta and are reported per-stage by the workload
        # smoke harness (run_fc_smoke.py). The +TREE_PIPE and CLB_NEEDED
        # bookkeeping is computed for downstream logging only.
        if k_tile > 1:
            tree_pipe = 0
            val = k_tile - 1
            while val > 0:
                tree_pipe += 1
                val >>= 1
        else:
            tree_pipe = 0
        total_cycles = self._pipeline_total_cycles(passes_per_dpe)
        # CLB_NEEDED gate (architectural, but now reported as a delta —
        # not added to the analytical output):
        #   CLB_NEEDED = (V > 1) OR (activation_mode AND !has_acam)
        # Rationale (info-only):
        #   • V > 1: CLB adder tree must sum V partial sums.
        #   • V == 1, has_acam (NL-DPE), any activation: ACAM-fused.
        #   • V == 1, !has_acam (AL), activation_mode=True: CLB ReLU LUT.
        #   • V == 1, !has_acam (AL), activation_mode=False: no CLB stage.
        has_acam = getattr(self.cfg, 'analoge_nonlinear_support', True)
        clb_needed = (k_tile > 1) or (bool(activation_mode) and not has_acam)
        t_clk = 1e3 / self.cfg.freq
        latency = total_cycles * t_clk
        log(f"\tVMM Path A: V={k_tile}, H={n_tile}, n_parallel_dpes={n_parallel_dpes}, "
            f"passes_per_dpe = M×ceil(V·H/n_parallel_dpes) = {M}×{math.ceil(k_tile * n_tile / n_parallel_dpes)} = {passes_per_dpe}, "
            f"TREE_PIPE = ⌈log₂(V)⌉ = {tree_pipe} (reported as delta, not added to sim_exp)")
        log(f"\t§4 unified pipeline T({passes_per_dpe}) = T_fill + (M−1)·T_steady = "
            f"{total_cycles} cycles → {latency:.2f} ns "
            f"(CLB_NEEDED={clb_needed}, has_acam={has_acam}, "
            f"activation_mode={bool(activation_mode)} — info-only)")
        energy = e_vmm_pj + e_reduc_pj + e_read_pj + e_conv_pj + e_digital_post_pj + e_write_pj
        log(f"=== linear layer: latency {latency:.2f}ns, energy {energy:.2f}pj ===\n")
        return latency, energy
