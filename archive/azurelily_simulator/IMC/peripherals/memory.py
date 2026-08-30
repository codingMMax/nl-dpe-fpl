from __future__ import annotations

import math

from scheduler_stats.common import cycles_to_ns


class MemoryModel:
    def __init__(self, cfg, stats):
        self.cfg = cfg
        self.stats = stats

    def _calc_energy(self, bytes):
        bytes_per_access = math.floor(self.cfg.bram_width / 8)
        num_access = math.ceil(bytes / bytes_per_access)
        return self.cfg.e_bram_pj_per_access * num_access

    def energy(self, bytes, read=True, record_breakdown=True):
        energy = self._calc_energy(bytes)
        if record_breakdown:
            key = "sram_read" if read else "sram_write"
            self.stats.record_energy_breakdown(key, energy)
        return energy

    def latency(self, bytes, read=True):
        bytes_per_access = max(
            1e-9, math.floor(self.cfg.bram_width / 8) * getattr(self.cfg, "mem_bw_utilization", 1.0)
        )
        num_access = math.ceil(bytes / bytes_per_access)
        if self.cfg.bram_mode == self.cfg.SP:
            num_cycle = num_access
        elif self.cfg.bram_mode == self.cfg.TDP:
            num_cycle = math.ceil(num_access / 2)
        else:
            num_cycle = num_access
        lat = cycles_to_ns(num_cycle, self.cfg.freq)
        key = "sram_read" if read else "sram_write"
        self.stats.record_latency_breakdown(key, lat)
        res_key = "bram_read_cnt_access" if read else "bram_write_cnt_access"
        self.stats.record_resource(res_key, num_access)
        return lat
