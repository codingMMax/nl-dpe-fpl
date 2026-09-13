from __future__ import annotations

from scheduler_stats.pipeline_profiler import PipelineProfiler


class Stats:
    def __init__(self):
        self.energy_stats = {}
        self.latency_stats = {}
        self.energy_breakdown = {
            "sram_read": 0.0,
            "sram_write": 0.0,
            "imc_vmm": 0.0,
            "imc_conversion": 0.0,
            "imc_digital_post": 0.0,
            "clb_reduction": 0.0,
            "fpga_activation": 0.0,
            "dsp_gemm": 0.0,
            "dsp_add": 0.0,
            "clb_exp": 0.0,
            "clb_norm_sum": 0.0,
            "clb_norm_inv": 0.0,
            "mul": 0.0,
            "clb_compare": 0.0,
            "clb_add": 0.0,
            "clb_layernorm": 0.0,
            "clb_embed_add": 0.0,
            "imc_dimm_exp_vmm": 0.0,
            "imc_dimm_exp_conv": 0.0,
            "imc_dimm_exp_digital": 0.0,
            "imc_dimm_log_vmm": 0.0,
            "imc_dimm_log_conv": 0.0,
            "imc_dimm_log_digital": 0.0,
        }   
        self.latency_breakdown = {
            "sram_read": 0.0,
            "sram_write": 0.0,
        }
        self.resource_layer = {}
        self.resource_total = {
            "imc_tiles": 0.0,
            "dsp_used": 0.0,
            "clb_used": 0.0,
            "act_units": 0.0,
            "memory_blocks": 0.0,
            "bram_read_cnt_access": 0.0,
            "bram_write_cnt_access": 0.0,
        }
        self.resource_peak = {
            "imc_tiles": 0,
            "dsp_used": 0,
            "clb_used": 0,
            "act_units": 0,
            "memory_blocks": 0,
        }
        self._active_layer = None
        self.latency_raw = {}
        self.pipeline_profiler = PipelineProfiler()

    def _stat_add(self, stats, key, value):
        stats[key] = stats.get(key, 0) + value

    def record_energy(self, key, value):
        self._stat_add(self.energy_stats, key, value)

    def record_energy_breakdown(self, key, value):
        self._stat_add(self.energy_breakdown, key, value)

    def record_latency_breakdown(self, key, value):
        self._stat_add(self.latency_breakdown, key, value)

    def begin_layer(self, layer_name):
        self._active_layer = layer_name
        if layer_name not in self.resource_layer:
            self.resource_layer[layer_name] = {}

    def end_layer(self):
        self._active_layer = None

    def record_resource(self, key, value, layer_name=None, peak=False):
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if value == 0:
            return

        target_layer = layer_name if layer_name is not None else self._active_layer
        if target_layer is not None:
            layer_stats = self.resource_layer.setdefault(target_layer, {})
            self._stat_add(layer_stats, key, value)

        self._stat_add(self.resource_total, key, value)
        if peak:
            self.resource_peak[key] = max(self.resource_peak.get(key, 0.0), value)

    def record_latency(self, layer, value, timing=None):
        self._stat_add(self.latency_raw, layer.name, value)
        contributions = self.pipeline_profiler.record(layer, value, timing=timing)
        for key, contribution in contributions.items():
            self._stat_add(self.latency_stats, key, contribution)

    def finalize_latency(self):
        contributions = self.pipeline_profiler.finalize()
        for key, contribution in contributions.items():
            self._stat_add(self.latency_stats, key, contribution)

    def total_energy(self):
        return sum(self.energy_stats.values())

    def total_latency(self):
        return sum(self.latency_stats.values())
