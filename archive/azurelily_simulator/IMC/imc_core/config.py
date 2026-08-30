import json
import sys
import math

class Config:
    """
    Unified IMC Simulator Configuration.

    Features:
    - Latency/Energy Estimation (3-Stage Model)
    - Architecture Capabilities (Linear vs. Non-Linear)
    - Geometric Scaling & Area Estimation
    """

    def __init__(self, json_file_path=None, data_dict=None):
        # Accept exactly one of: json_file_path (existing API) or data_dict (new API).
        if (json_file_path is None) == (data_dict is None):
            raise ValueError(
                "Config requires exactly one of `json_file_path` or `data_dict`."
            )

        if json_file_path is not None:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        else:
            data = data_dict

        self._parse_dict(data)
        self._compute_derived()

    @classmethod
    def from_dict(cls, d):
        """Construct Config from a dict (e.g., DSE-generated config).

        Bypasses JSON file loading entirely. The dict structure must match
        the JSON schema (same nested keys: 'geometry', 'capabilities',
        'params', 'fpga_specs', etc.).

        Example:
            cfg = Config.from_dict({
                "core_name": "NL-DPE",
                "geometry": {"array_rows": 512, ...},
                ...
            })
        """
        return cls(data_dict=d)

    def patch(self, **overrides):
        """Override any public field and recompute derived attrs.

        Useful for DSE sweeps or applying VTR-reported Fmax/resources without
        writing a temp JSON file.

        Example:
            cfg = Config("nl_dpe.json")
            cfg.patch(rows=512, cols=256, core_freq_MHz=1500)
            # core_cycle_ns, t_conv_ns, etc. are recomputed automatically.

        Args:
            **overrides: any attribute name -> new value. Unknown attrs raise
                AttributeError.
        """
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise AttributeError(f"Config has no attribute '{k}'")
            setattr(self, k, v)
        self._compute_derived()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    def _parse_dict(self, data):
        """Set all public fields from a config dict (JSON-schema shaped)."""
        # --- Metadata ---
        self.imc = data.get("core_name", "Unknown")
        self.tech_node = data.get("tech_node_nm", 22)
        # --- Geometry & Capabilities ---
        self.geometry = data.get("geometry", {})
        self.rows = self.geometry.get("array_rows", 1)
        self.cols = self.geometry.get("array_cols", 1)
        self.capabilities = data.get("capabilities", {})
        self.analoge_nonlinear_support = self.capabilities.get("analog_nonlinear", False)
        self.log_softmax_fusion = self.capabilities.get("log_softmax_fusion", False)
        self.area_params = data.get("area_params", {})

        # --- Simulation Inputs ---
        self.n_in = data.get("precision_bits", 8)
        self.input_bs = self.capabilities['input_bit_slicing']
        self.analoge_accum = self.capabilities['analog_accum']
        self.digital_accum = self.capabilities['digital_accum']
        # Inter-position pipeline: read/compute/reduce/write stages overlap
        # across output positions.  Both Azure-Lily (digital accum, bit-serial
        # pipeline) and NL-DPE (analog accum, 3-stage pipeline + ACAM) support
        # this — the datapath can start reading the next position while the
        # current position's output is being reduced/written.
        self.pipelinable = True

        # --- Unified Timing (ns) & Energy (pJ) ---
        self.core_freq_MHz = data["params"].get("freq_MHz", 1000)

        params = data.get("params", {})
        self.cols_per_adc = params.get("cols_per_adc", 1)
        self.scale_with_geometry = params.get("scale_with_geometry", False)

        # Base (per-cycle / per-cell) values from JSON. Derived attrs
        # (t_*_ns, e_*_pj after geometry scaling) are computed in
        # _compute_derived() so patch() can refresh them.
        self.t_analoge_base = params.get("t_analoge", 0.0)
        self.t_conv_base = params.get("t_conv", 0.0)
        self.t_digital_base = params.get("t_digital", 0.0)

        self.e_analoge_pj_base = params.get("e_analoge_pj", 0.0)
        self.e_conv_pj_base = params.get("e_conv_pj", 0.0)
        self.e_digital_pj_base = params.get("e_digital_pj", 0.0)

        # --- External buffer interface ---
        # 2 for true dual-port
        # 1 for simple dual-port
        # 0 for single-port
        self.SP = 0
        self.TDP = 1
        # self.SDP = 2
        self.bram_mode = data['fpga_specs'].get("bram_mode", 0)

        self.e_bram_pj_per_access = data['fpga_specs'].get("bram_pj_per_access", 10)
        self.e_dsp_pj_per_mac = data['fpga_specs'].get("dsp_pj_per_mac", 10)
        self.e_clb_pj_per_mac = data['fpga_specs'].get("clb_pj_per_mac", 5)

        # Event-driven reference op energies converted from nJ to pJ.
        ref_compare_pj = (793.1801e-6 / 3.0) * 1e3
        ref_sum_pj = 84.98358e-6 * 1e3
        ref_act_pj = 453.2458e-6 * 1e3

        eps = 1e-12
        self.clb_coeff_compare = data['fpga_specs'].get("clb_coeff_compare")
        if self.clb_coeff_compare is None:
            self.clb_coeff_compare = ref_compare_pj / max(eps, self.e_clb_pj_per_mac)

        self.clb_coeff_add = data['fpga_specs'].get("clb_coeff_add")
        if self.clb_coeff_add is None:
            self.clb_coeff_add = ref_sum_pj / max(eps, self.e_clb_pj_per_mac)

        self.clb_coeff_act = data['fpga_specs'].get("clb_coeff_act")
        explicit_act_pj = data['fpga_specs'].get("act_energy_pj_per_op")
        if self.clb_coeff_act is None:
            if explicit_act_pj is None:
                self.clb_coeff_act = ref_act_pj / max(eps, self.e_clb_pj_per_mac)
            else:
                self.clb_coeff_act = explicit_act_pj / max(eps, self.e_clb_pj_per_mac)

        self.act_units = max(1, int(data['fpga_specs'].get("act_units", 16)))
        self.act_cycles_per_op = max(1, int(data['fpga_specs'].get("act_cycles_per_op", 1)))

        self.bram_width = data['fpga_specs'].get("bram_width", 40) # 40-bit wide BRAM port (CLB/DSP access)
        self.dpe_buf_width = data['fpga_specs'].get("dpe_buf_width", self.bram_width)  # BRAM→DPE input buffer width
        self.mem_bw_utilization = data['fpga_specs'].get("mem_bw_utilization", 1.0)
        if self.mem_bw_utilization <= 0:
            self.mem_bw_utilization = 1.0
        if self.mem_bw_utilization > 1.0:
            self.mem_bw_utilization = 1.0
        self.freq = data['fpga_specs'].get("freq", 200) # 200MHz fpga frequency
        self.total_dsp = data['fpga_specs'].get("total_dsp", 0)
        self.total_clb = data['fpga_specs'].get("total_clb", 0)
        self.total_io = data['fpga_specs'].get("total_io", 0)
        self.total_mem = data['fpga_specs'].get("total_mem", 0)
        # §3 / §7: softmax row-parallel lanes (W). One DPE+tree lane per row
        # tile group; same W applies to mac_qk, softmax, and mac_sv.
        # Single allocation knob for the entire DIMM workload (Pattern β).
        self.total_softmax_lanes = data['fpga_specs'].get("total_softmax_lanes", 1)

    def _compute_derived(self):
        """Recompute derived attributes from base fields.

        Called at the end of __init__ and after every patch(). All attrs
        whose values depend on `core_freq_MHz`, `cols`, `cols_per_adc`,
        `scale_with_geometry`, `e_clb_pj_per_mac`, or `clb_coeff_act` are
        derived here.
        """
        self.core_cycle_ns = self.core_freq_MHz / 1000

        self.t_analoge_ns = self.t_analoge_base * self.core_cycle_ns
        self.t_conv_ns = self.t_conv_base * self.core_cycle_ns
        self.t_digital_ns = self.t_digital_base * self.core_cycle_ns

        self.e_analoge_pj = self.e_analoge_pj_base
        self.e_conv_pj = self.e_conv_pj_base
        self.e_digital_pj = self.e_digital_pj_base

        if self.scale_with_geometry:
            self.t_conv_ns = self.t_conv_base * self.core_cycle_ns * self.cols_per_adc
            self.e_analoge_pj = self.e_analoge_pj_base * self.cols
            self.e_conv_pj = self.e_conv_pj_base * self.cols
            self.e_digital_pj = self.e_digital_pj_base * self.cols

        # BRAM port-width derived bandwidth (depends on bram_width and TDP flag).
        self.t_bram_bw_byte_cycle = (
            self.bram_width // 4 if self.bram_width == self.TDP else self.bram_width // 8
        )

        # CLB activation energy (depends on e_clb_pj_per_mac and clb_coeff_act).
        self.act_energy_pj_per_op = self.e_clb_pj_per_mac * self.clb_coeff_act

    # --- 1. Geometry Accessor ---
    def get_geometry(self):
        """Returns physical dimensions of the IMC array."""
        return {
            "rows": self.geometry.get("array_rows", 0),
            "cols": self.geometry.get("array_cols", 0),
            "acam_rows": self.geometry.get("acam_rows", 0), # 0 if not supported
            "weight_bits_per_cell": self.geometry.get("weight_bits_per_cell", 1)
        }

    # --- 2. Capabilities Accessor ---
    def get_capabilities(self):
        """Returns dictionary of supported operations (Linear/Non-Linear)."""
        return self.capabilities

    # --- Previous Latency/Energy Methods ---
    def _get_arch_factors(self):
        """
        Returns (k_vmm, k_conv, k_digital) — repeat counts for each stage.

        Azure-Lily (digital_accum=True, analoge_accum=False):
          t_analoge = DAC→Crossbar       (×8, per bit)
          t_conv    = ADC                 (×8, per bit)
          t_digital = Shift & Add         (×8, per bit)

        NL-DPE (analoge_accum=True, digital_accum=False):
          t_analoge = DAC→Crossbar        (×8, per bit)
          t_conv    = Analog Accumulation  (×8, per bit)
          t_digital = ACAM→Output         (×1, fires once)
        """
        k_vmm = self.n_in
        k_conv = self.n_in
        k_digital = 0

        if self.input_bs:
            k_vmm = self.n_in

        if self.analoge_accum and not self.digital_accum:
            # NL-DPE: analog accum runs per bit (pipelined with VMM),
            # ACAM fires once after all bits are accumulated
            k_conv = self.n_in
            k_digital = 1
        elif self.analoge_accum:
            # Hybrid: analog accum reduces ADC conversion to 1
            k_conv = 1
        if self.digital_accum:
            k_digital = self.n_in

        return k_vmm, k_conv, k_digital

    def __str__(self):
        """Returns a formatted string representation of the configuration."""
        geometry = self.get_geometry()
        # latency = self.get_latency_per_VMM()
        # energy = self.get_energy_per_VMM()

        return (
            f"=== {self.imc} ===\n"
            f"Tech Node: {self.tech_node}nm\n"
            f"Precision: {self.n_in}-bit\n"
            f"Array: {geometry['rows']}x{geometry['cols']} "
            f"({geometry['weight_bits_per_cell']}-bit cells)\n"
            f"Core Frequency {self.core_freq_MHz} MHz\n"
            f"FPGA Frequency: {self.freq} MHz\n"
            # f"Latency per VMM: {latency:.2f} ns\n"
            # f"Energy per VMM: {energy:.2f} pJ"
        )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python config.py <json_file_path>")
        sys.exit(1)

    test_file = sys.argv[1]
    azure_lily = Config(test_file)
    print(azure_lily.__str__())
