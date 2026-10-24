from __future__ import annotations

import numpy as np
from pathlib import Path
import json
import math

# nl_dpe.json, azure_lily.json sit beside this file
CONFIG_DIR = Path(__file__).resolve().parent
MWTA_UM2 = 0.033864
COMPONENTS = {  # name: (area unit, area key, energy key, energy charged per unit)
    "crossbar":      ("cell",   "area_um2_per_cell",  "energy_pj_per_cell",       "bit_slice"),
    "input_buffer":  ("entry",  "area_um2_per_entry", "energy_pj_per_byte",       "input_byte"),
    "output_buffer": ("entry",  "area_um2_per_entry", "energy_pj_per_byte",       "output_byte"),
    "acam":          ("column", "area_um2_per_col",   "energy_pj_per_col",        "pass"),
    "adc":           ("adc",    "area_um2_per_adc",   "energy_pj_per_conversion", "bit_slice"),
}


FORMS = {"identity": lambda y: y,              
         "relu": lambda y: np.maximum(y, 0),
         "exp": lambda y: 1 + y + y * y // 2,
         "log": lambda y: y - 1}

def resolveConfig(source) -> Path:
    for candidate in (Path(source), CONFIG_DIR / Path(source), CONFIG_DIR / f"{source}.json"):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"no platform config {str(source)!r} (looked in {CONFIG_DIR})")


def parseTile(key, cnt, src):
    return src[key] if cnt is None else cnt


class Core:
    def _units(self, name: str) -> int:
        unit = COMPONENTS[name][0]
        if unit == "cell":
            return self.rows * self.cols
        if unit == "column":
            return self.cols
        if unit == "entry":
            return self.buf_size
        return math.ceil(self.cols / self.components["adc"]["cols_per_adc"])

    def __init__(self, config="nl_dpe", rows: int | None = None, cols: int | None = None,
                 buf_size: int | None = None):
        if isinstance(config, dict):
            self.path, data = Path("<dict>"), config
        else:
            self.path = resolveConfig(config)
            data = json.loads(self.path.read_text(encoding="utf-8"))

        self.fabric = data["fabric"]
        tile = data["compute_tile"]
        self.name = data["platform_name"]

        self.rows = parseTile("rows", rows, tile)
        self.cols = parseTile("cols", cols, tile)
        self.buf_size = parseTile("buffer_size", buf_size, tile)
        self.input_bit_slices = tile["input_bit_slices"]
        self.port_width = tile["port_width_bits"]
        self.output_stage = "acam" if "acam" in tile else "adc"
        self.modes = tuple(tile[self.output_stage]["modes"])
        self.components = {name: tile[name] for name in COMPONENTS if name in tile}  # acam or adc, not both
        
        self.units = {name: self._units(name) for name in self.components}
        
        self.area_um2 = {name: self.units[name] * component[COMPONENTS[name][1]]
                         for name, component in self.components.items()}
        self.total_area_um2 = sum(self.area_um2.values())
        self.component_energy_pj = {  # per fire (crossbar: the whole array; acam: all columns), byte, or conversion
            name: component[COMPONENTS[name][2]] *
            (self.units[name] if name in ("crossbar", "acam") else self.cols if name == "adc" else 1)
            for name, component in self.components.items()}
        self.weights = None     # int64 [rows, cols], set by program()
        self.mode = None

    @property
    def clock_mhz(self) -> float:
        return self.fabric["clock_mhz"]

    @property
    def total_area_mm2(self) -> float:
        return self.total_area_um2 / 1e6

    @property
    def total_area_mwta(self) -> float:
        return self.total_area_um2 / MWTA_UM2

    def __repr__(self) -> str:
        return f"Core({self.name!r}, {self.rows}x{self.cols}, {self.output_stage}, modes={list(self.modes)})"
    @staticmethod
    def _trunc8(z):
        return (np.clip(z, -2**31, 2**31 -1) & 0xFF).astype(np.uint8).view(np.int8)
    
    def program(self, weights = None, mode = "identity"):
        if mode not in self.modes or mode not in FORMS:
            raise ValueError(f"mode {mode!r} not available on {self.name} (modes {self.modes})")
        if weights is None:
            self.weights = np.eye(self.rows, self.cols, dtype=np.int32)
        else:
            w = np.asarray(weights)
            if w.ndim != 2 or not (1 <= w.shape[0] <= self.rows and 1 <= w.shape[1] <= self.cols):
                raise ValueError(f"weights {w.shape} do not fit a {self.rows} x {self.cols} array")
            if w.min() < -128 or w.max() > 127:
                raise ValueError("weights must be int8 values (-128..127)")
            self.weights = np.zeros((self.rows, self.cols), dtype=np.int32)
            self.weights[:w.shape[0], :w.shape[1]] = w
        self.mode = mode
        
    def run(self, x):
        assert self.weights is not None, f"Weights must be programmed before running behavior simulation"
        assert x <= self.rows, f"Input X vector size cannot larger than rows count."
        bits = x.view(np.uint8)
        P = self.input_bit_slices
        acc, prev, t = np.zeros(self.cols, dtype=np.int64), None, 0
        for b in range(P):
            s = (bits >> b & 1).astype(np.int32) @ self.weights[:len(x)]
            if prev is not None: 
                acc += prev << (b - 1)
            prev = s
            t = t + 1
        # last accumulation
        acc -= prev << (P - 1)
        t = t + 1
        out = self._trunc8(FORMS[self.mode](acc))
        t = t + 1
        return out, t
        
            

class CorePass:
    def __init__(self, core: Core, used_rows: int | None = None):
        assert core is not None, f"CorePass require valid core to instantiate"
        self.core = core
        if used_rows is None:
            self.used_rows = core.rows
        else:
            self.used_rows = used_rows

    @property
    def pass_cycles(self) -> dict[str: float]:
        """
        return the cycles take for one pass
        """
        load_cyles = math.ceil(self.used_rows * self.core.input_bit_slices
                                    / self.core.port_width)
        drain_cycles = math.ceil(self.core.cols * self.core.input_bit_slices /
                                      self.core.port_width)
        fire_cycles = math.ceil(self.core.input_bit_slices)

        cycles = {
            "load": load_cyles,
            "fire": fire_cycles,
            "drain": drain_cycles,
            "total": max(load_cyles + fire_cycles, fire_cycles + 2, drain_cycles + 1)
        }

        return cycles

    @property
    def pass_energy_pj(self) -> dict[str: float]:
        """
        return energy spent per pass
        """
        INPUT_ACCESS = 3
        OUTPUT_ACCESS = 2
        events = {"bit_slice": self.core.input_bit_slices,          # crossbar cells, ADC bank
                  "input_byte": INPUT_ACCESS * self.used_rows,
                  "output_byte": OUTPUT_ACCESS * self.core.cols,
                  "pass": 1}                                        # ACAM
        energy = {name: events[COMPONENTS[name][3]] * pj for name,
                  pj in self.core.component_energy_pj.items()}
        energy["total"] = sum(energy.values())
        return energy
    
    def run(self, x):
        assert len(x) == self.used_rows, f"input X vector must match the used rows count"
        res, _ = self.core.run(x)
        cycles = self.pass_cycles['total']
        energy = self.pass_energy_pj['total']
        return res, cycles, energy
    
    
if __name__ == "__main__":
    cfg = "azure_lily"
    nl_dpe = Core(cfg)
    nl_dpe_pass = CorePass(nl_dpe)
    # print(nl_dpe.components.items())
    # print(nl_dpe_pass.passEnergyPj)
    # print(nl_dpe_pass.pass_cycles)
    # print(nl_dpe.total_area_um2)
    # print(nl_dpe.components[name] for name in COMPONENTS)

    rng = np.random.default_rng()
    weights = rng.integers(low=-128, high=128, size=(nl_dpe.rows, nl_dpe.cols), dtype=np.int8)
    x = rng.integers(low=-128, high=128, size = nl_dpe.rows, dtype=np.int8)
    
    nl_dpe.program(weights)
    print(nl_dpe.run(x))    