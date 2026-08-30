from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imc_core.config import Config
from scheduler_stats.stats import Stats
from peripherals.memory import MemoryModel
from imc_core.imc_core import IMCCore
from peripherals.fpga_fabric import FPGAFabric
from scheduler_stats.scheduler import Scheduler


class IMC:
    def __init__(self, json_file_path: str, core_count: int = 1, cfg: Config = None) -> None:
        self.cfg = None
        self.core_count = core_count
        assert json_file_path is not None or cfg is not None, f"Json file path and core config cannot both be none"
        if cfg is None:
            self.cfg = Config(json_file_path)
        else:
            self.cfg = cfg

        self.stats = Stats()
        self.memory = MemoryModel(self.cfg, self.stats)
        self.imc_core = IMCCore(self.cfg, self.memory, self.stats)
        self.fpga = FPGAFabric(self.cfg, self.memory, self.stats, imc_core=self.imc_core)
        self.scheduler = Scheduler(self.cfg, self.stats, self.imc_core, self.fpga)

    @property
    def energy_stats(self):
        return self.stats.energy_stats

    @property
    def energy_breakdown(self):
        return self.stats.energy_breakdown

    @property
    def latency_stats(self):
        return self.stats.latency_stats

    @property
    def latency_raw(self):
        return self.stats.latency_raw

    @property
    def resource_layer(self):
        return self.stats.resource_layer

    @property
    def resource_total(self):
        return self.stats.resource_total

    @property
    def resource_peak(self):
        return self.stats.resource_peak

    def run_layer(self, layer):
        return self.scheduler.run_layer(layer)

    def finalize_latency_stats(self):
        self.stats.finalize_latency()

    def total_energy(self):
        return self.stats.total_energy()

    def total_latency(self):
        return self.stats.total_latency()
