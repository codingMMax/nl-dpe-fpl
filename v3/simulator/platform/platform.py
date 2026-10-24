"""platform.py — `Platform`: the crossbar type the models run on (docs/simulator.md §4.7).

A config (<name>.json beside this file, or any path) gives the shared `fabric` (block RAM and CLB prices) and the
`compute_tile`: the crossbar geometry, the input bit slices, the port (R bits in and C bits out per cycle, P planes per
pass, priced per bit: `port`; or a fixed width, `port_width_bits`, as azure_lily.json), the buffers, and the components with
their area and energy per unit:

  component      area unit                        energy charged per pass (prices.py)
  crossbar       cell (rows x cols)               input_bit_slices fires x cells x e_cell
  input_buffer   entry (buffer_size)              3 x rows bytes x e_byte
  output_buffer  entry (buffer_size)              2 x cols bytes x e_byte
  acam           column                           1 fire x cols x e_col     (all columns: the whole-array price)
  adc            ADC (ceil(cols / cols_per_adc))  input_bit_slices x active columns x e_conversion

The output stage (acam or adc) lists the modes its crossbars can run. NL-DPE (`nl_dpe.json`) is the platform in use;
`azure_lily.json` is kept as data so the layer stays open for a second crossbar type (user decision 2026-10-14).
The prices follow the v2 charter (v2/sim/spec/simulator.md §4) and are analytical, never measured hardware truth.

Named `platform.py` inside the package; a script's own folder is never this one, so numpy's import of the stdlib
`platform` module is safe.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent          # nl_dpe.json, azure_lily.json sit beside this file
MWTA_UM2 = 0.033864  # nl_dpe/area_power.py:27
COMPONENTS = {  # name: (area unit, area key, energy key, energy charged per)
    "crossbar":      ("cell",   "area_um2_per_cell",  "energy_pj_per_cell",       "bit_slice"),
    "input_buffer":  ("entry",  "area_um2_per_entry", "energy_pj_per_byte",       "input_byte"),
    "output_buffer": ("entry",  "area_um2_per_entry", "energy_pj_per_byte",       "output_byte"),
    "acam":          ("column", "area_um2_per_col",   "energy_pj_per_col",        "pass"),
    "adc":           ("adc",    "area_um2_per_adc",   "energy_pj_per_conversion", "conversion"),
}
BUFFER_ACCESSES = {"input_byte": 3, "output_byte": 2}   # Azure-Lily nn/: ext-buffer write + read, internal write;
#                                                          output: write + read


def resolveConfig(source) -> Path:
    for candidate in (Path(source), CONFIG_DIR / Path(source), CONFIG_DIR / f"{source}.json"):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"no platform config {str(source)!r} (looked in {CONFIG_DIR})")


class Platform:
    def __init__(self, config="nl_dpe", rows: int | None = None, cols: int | None = None,
                 bufferSize: int | None = None):
        if isinstance(config, dict):
            self.path, data = Path("<dict>"), config
        else:
            self.path = resolveConfig(config)
            data = json.loads(self.path.read_text(encoding="utf-8"))
        tile = data["compute_tile"]
        self.name = data["platform_name"]
        self.fabric = data["fabric"]
        self.rows = tile["rows"] if rows is None else rows
        self.cols = tile["cols"] if cols is None else cols
        self.bufferSize = tile["buffer_size"] if bufferSize is None else bufferSize
        self.inputBitSlices = tile["input_bit_slices"]
        port = tile.get("port")                           # the row-parallel interface (2026-10-23, user)
        if port is not None:
            self.portFixedWidthBits = None
            self.portInBits = self.rows if port["in_bits"] == "rows" else int(port["in_bits"])
            self.portOutBits = self.cols if port["out_bits"] == "cols" else int(port["out_bits"])
            self.portWordsPerPass = (self.inputBitSlices if port["planes_per_pass"] == "input_bit_slices"
                                     else int(port["planes_per_pass"]))
            self.portEnergyPjPerBit = port["energy_pj_per_bit"]
        else:                                             # a fixed-width port (azure_lily.json, kept as data)
            self.portFixedWidthBits = tile["port_width_bits"]
            self.portInBits = self.portOutBits = self.portFixedWidthBits
            self.portWordsPerPass = None
            self.portEnergyPjPerBit = self.fabric["block_ram"]["energy_pj_per_access"] / self.portFixedWidthBits
        self.components = {name: tile[name] for name in COMPONENTS if name in tile}
        self.outputStage = "acam" if "acam" in tile else "adc"
        self.modes = tuple(tile[self.outputStage]["modes"])
        self.units = {name: self._units(name) for name in self.components}
        self.areaUm2 = {name: self.units[name] * component[COMPONENTS[name][1]]
                        for name, component in self.components.items()}
        self.totalAreaUm2 = sum(self.areaUm2.values())
        self.eventEnergyPj = {  # per fire (crossbar: the whole array; acam: all columns), byte, or conversion
            name: component[COMPONENTS[name][2]] * (self.units[name] if name in ("crossbar", "acam") else 1)
            for name, component in self.components.items()}

    def _units(self, name: str) -> int:
        unit = COMPONENTS[name][0]
        if unit == "cell":
            return self.rows * self.cols
        if unit == "column":
            return self.cols
        if unit == "entry":
            return self.bufferSize
        return math.ceil(self.cols / self.components["adc"]["cols_per_adc"])

    @property
    def clockMhz(self):
        return self.fabric["clock_mhz"]

    @property
    def totalAreaMm2(self) -> float:
        return self.totalAreaUm2 / 1e6

    @property
    def totalAreaMwta(self) -> float:
        return self.totalAreaUm2 / MWTA_UM2

    def __repr__(self) -> str:
        return f"Platform({self.name!r}, {self.rows}x{self.cols}, {self.outputStage}, modes={list(self.modes)})"
