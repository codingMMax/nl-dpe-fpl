#!/usr/bin/env python3
"""platforms.py — `Platform(config)`: load a v2 platform JSON and evaluate its compute tile.

A config (any path, or a bare name in `v2/sim/simulator/configs/`; the archive's
`Config(json_file_path)` convention) has `platform_name`, the shared `fabric`,
the `compute_tile`, and informational `resource_budgets` (SIM8).

Every platform's tile is the SAME crossbar + input/output buffers (identical
per-unit costs in every config, checked on `load_all`) plus ONE output stage:

  component      area unit                  energy charged per pass (cost.py)
  crossbar       cell (rows x cols)         input_bit_slices fires x cells x e_cell
  input_buffer   entry (buffer_size)        3 x rows bytes x e_byte   (nn/: ext-buffer write + read,
                                                                       internal-buffer write)
  output_buffer  entry (buffer_size)        2 x cols bytes x e_byte   (nn/: write + read)
  acam           column                     1 fire x cols x e_col     (all columns, whole-array §4.2)
  adc            ADC (ceil(cols / cols_per_adc))  input_bit_slices x active columns x e_conversion

Buffer and ADC energy follow the Azure-Lily event simulator (`archive/azurelily_simulator/nn/`,
the ground truth: `constant.py` BUFFER_*_ENERGY 4.95e-6 nJ/byte, ADC_ENERGY 2.33 pJ per bit per
column; `linear_layer.py:115,131-134,158`), which has no crossbar term; the crossbar fire is
`nl_dpe/area_power.py`'s 1.31 mW x 1 ns per 256x256 fire, charged on both platforms (decision
2026-10-03). Shared area follows `area_power.py` (crossbar 0.011534 mm² and each buffer
0.0003542 mm² at 256 units, no DAC). ACAM (+ its XOR encoder) area/energy are
`area_power.py`'s; ADC area is the archive's 275 um²/column x 16 columns per ADC. Azure-Lily's shift-add is charged zero (OPEN SIM12).
Operator realizations follow from the output stage's `modes`. Configs are trusted as written.

Named `platforms.py`, not `platform.py`: a script's own folder sits at `sys.path[0]`, and numpy
imports the stdlib `platform` module.

Run:  python3 v2/sim/simulator/platforms.py                       self-test
      python3 v2/sim/simulator/platforms.py --imc nl_dpe [--rows R] [--cols C] [--buffer-size B]
      python3 v2/sim/simulator/platforms.py --imc a.json b.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MWTA_UM2 = 0.033864  # nl_dpe/area_power.py:27
SHARED = ("crossbar", "input_buffer", "output_buffer")
COMPONENTS = {  # name: (area unit, area key, energy key, energy charged per)
    "crossbar":      ("cell",   "area_um2_per_cell",  "energy_pj_per_cell",       "bit_slice"),
    "input_buffer":  ("entry",  "area_um2_per_entry", "energy_pj_per_byte",       "input_byte"),
    "output_buffer": ("entry",  "area_um2_per_entry", "energy_pj_per_byte",       "output_byte"),
    "acam":          ("column", "area_um2_per_col",   "energy_pj_per_col",        "pass"),
    "adc":           ("adc",    "area_um2_per_adc",   "energy_pj_per_conversion", "conversion"),
}
BUFFER_ACCESSES = {"input_byte": 3, "output_byte": 2}  # nn/linear_layer.py:115,131-132 / :134,158
FABRIC_FALLBACK = {  # charter §3.5 (Azure-Lily column); relu -> activation unit (charter §4.3)
    "exp": {"engine": "fabric", "lookup_table_reads": 1},
    "log": {"engine": "fabric", "lookup_table_reads": 1},
    "relu": {"engine": "fabric", "activation_ops": 1},
    "softmax_normalize": {"engine": "fabric", "lookup_table_reads": 1, "dsp_multiplies": 1},
}


def realize_operators(modes) -> dict:
    tile = set(modes)
    log_domain = {"exp", "log"} <= tile
    ops = {"linear_matmul": {"engine": "gemm_array"}}
    for op in ("exp", "log", "relu"):
        ops[op] = {"engine": "tile_pass", "mode": op} if op in tile else dict(FABRIC_FALLBACK[op])
    ops["log_domain_matmul"] = {"engine": "dimm" if log_domain else "unavailable"}
    ops["attention_score_matmul"] = {"engine": "dimm" if log_domain else "dsp_mac_lanes"}
    ops["softmax_normalize"] = ({"engine": "log_domain"} if log_domain
                                else dict(FABRIC_FALLBACK["softmax_normalize"]))
    return ops


def resolve_config(source) -> Path:
    for cand in (Path(source), CONFIG_DIR / Path(source), CONFIG_DIR / f"{source}.json"):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"no platform config {str(source)!r} (looked in {CONFIG_DIR})")


class Platform:
    def __init__(self, config, rows: int | None = None, cols: int | None = None,
                 buffer_size: int | None = None):
        if isinstance(config, dict):
            self.path, data = Path("<dict>"), config
        else:
            self.path = resolve_config(config)
            data = json.loads(self.path.read_text(encoding="utf-8"))
        tile = data["compute_tile"]
        self.raw = data
        self.name = data["platform_name"]
        self.fabric = data["fabric"]
        self.resource_budgets = data["resource_budgets"]
        self.rows = tile["rows"] if rows is None else rows
        self.cols = tile["cols"] if cols is None else cols
        self.buffer_size = tile["buffer_size"] if buffer_size is None else buffer_size
        self.input_bit_slices = tile["input_bit_slices"]
        self.port_width_bits = tile["port_width_bits"]
        self.components = {name: tile[name] for name in COMPONENTS if name in tile}
        self.output_stage = "acam" if "acam" in tile else "adc"
        self.modes = tuple(tile[self.output_stage]["modes"])
        self.operator_realizations = realize_operators(self.modes)

        self.units = {name: self._units(name) for name in self.components}
        self.area_um2 = {name: self.units[name] * comp[COMPONENTS[name][1]]
                         for name, comp in self.components.items()}
        self.total_area_um2 = sum(self.area_um2.values())
        self.event_energy_pj = {  # per fire (crossbar: whole array; acam: all columns), byte, conversion
            name: comp[COMPONENTS[name][2]] * (self.units[name] if name in ("crossbar", "acam") else 1)
            for name, comp in self.components.items()}

    def _units(self, name: str) -> int:
        unit = COMPONENTS[name][0]
        if unit == "cell":
            return self.rows * self.cols
        if unit == "column":
            return self.cols
        if unit == "entry":
            return self.buffer_size
        return math.ceil(self.cols / self.components["adc"]["cols_per_adc"])

    @property
    def system_clock_mhz(self):
        return self.fabric["clock_mhz"]

    @property
    def total_area_mm2(self) -> float:
        return self.total_area_um2 / 1e6

    @property
    def total_area_mwta(self) -> float:
        return self.total_area_um2 / MWTA_UM2

    def __repr__(self) -> str:
        return f"Platform({self.name!r}, {self.rows}x{self.cols}, {self.output_stage}, modes={list(self.modes)})"


def _check_shared(platforms: list) -> None:
    for key in ("fabric",) + SHARED:
        values = {json.dumps(p.fabric if key == "fabric" else p.components[key], sort_keys=True)
                  for p in platforms}
        if len(values) != 1:
            raise ValueError(f"{key} differs across {[p.name for p in platforms]} (must be identical)")


def load_all() -> dict[str, Platform]:
    platforms = [Platform(path) for path in sorted(CONFIG_DIR.glob("*.json"))]
    _check_shared(platforms)
    return {p.name: p for p in platforms}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def summary(p: Platform) -> str:
    import cost  # noqa: E402  (summary only; cost imports the certified cycle law)

    e = cost.tile_pass_energy(p)
    traffic = cost.pass_traffic_energy(p)
    cm = cost.tile_cycle_law(p, 1)
    unit_label = {"crossbar": "pJ / fire", "input_buffer": "pJ / byte", "output_buffer": "pJ / byte",
                  "acam": "pJ / fire", "adc": "pJ / conv."}
    lines = [
        f"=== {p.name}  ({p.path}) ===",
        f"tile          {p.rows} x {p.cols} crossbar, {p.input_bit_slices} input bit slices, "
        f"port {p.port_width_bits} b, buffers {p.buffer_size} entries; "
        f"output stage {p.output_stage} {list(p.modes)}",
        f"{'component':14s}{'units':>8s}{'area um2':>12s}{'energy':>12s}  {'':11s}charged per pass",
    ]
    for name in p.components:
        lines.append(f"{name:14s}{p.units[name]:>8d}{p.area_um2[name]:>12.1f}"
                     f"{p.event_energy_pj[name]:>12.5f}  {unit_label[name]:11s}{e[name]:.4f} pJ")
    lines += [
        f"area          {p.total_area_um2:.1f} um2 = {p.total_area_mm2:.6f} mm2 = {p.total_area_mwta:.0f} MWTA",
        f"per pass      tile {e['total']:.4f} pJ + port traffic {traffic:.4f} = {e['total'] + traffic:.4f} pJ",
        f"power         {cost.tile_steady_power_mw(p):.4f} mW (tile, back-to-back passes at "
        f"T_steady, {p.system_clock_mhz} MHz)",
        f"cycle law     LOAD {int(cm.load_cyc)}  COMPUTE {int(cm.compute_cyc)}  OUTPUT {int(cm.output_cyc)}"
        f"  T_fill {int(cm.t_fill)}  T_steady {int(cm.t_steady)}",
        "operators     " + "; ".join(f"{op} {r['engine']}" + (f"({r['mode']})" if "mode" in r else "")
                                     for op, r in p.operator_realizations.items()),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate v2 platform config(s): area, energy, cycle law. "
                                             "No arguments: self-test.")
    ap.add_argument("--imc", nargs="+", metavar="CONFIG",
                    help="platform JSON path(s), or bare names in v2/sim/simulator/configs/")
    ap.add_argument("--rows", type=int, help="override crossbar rows")
    ap.add_argument("--cols", type=int, help="override crossbar columns")
    ap.add_argument("--buffer-size", type=int, help="override input/output buffer entries")
    args = ap.parse_args(argv)
    if not args.imc:
        _self_test()
        return 0
    try:
        platforms = [Platform(src, args.rows, args.cols, args.buffer_size) for src in args.imc]
    except FileNotFoundError as exc:
        ap.error(str(exc))
    print("\n\n".join(summary(p) for p in platforms))
    return 0


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/simulator/platforms.py
# ---------------------------------------------------------------------------
_DSE_REFERENCE = (  # dse/results/config_tile_reference.csv (area_power.py output): R, C, MWTA, um2
    (128, 64, 360204, 12197), (128, 128, 709950, 24041), (128, 256, 1419900, 48083),
    (256, 64, 413238, 13993), (256, 128, 805559, 27279), (256, 256, 1590199, 53850),
    (512, 64, 519307, 17585), (512, 128, 996776, 33754), (512, 256, 1951715, 66092),
    (1024, 64, 731444, 24769), (1024, 128, 1379212, 46705), (1024, 256, 2674748, 90577),
)
_AL_ARCH_XML_MWTA = 2_320_000  # nl_dpe/area_power.py:206 (Azure-Lily published arch XML, 6x5 tile)
_CHARTER_REALIZATIONS = {  # charter §3.5, verbatim
    "nl_dpe": {"linear_matmul": {"engine": "gemm_array"},
               "log_domain_matmul": {"engine": "dimm"},
               "exp": {"engine": "tile_pass", "mode": "exp"},
               "log": {"engine": "tile_pass", "mode": "log"},
               "softmax_normalize": {"engine": "log_domain"}},
    "azure_lily": {"linear_matmul": {"engine": "gemm_array"},
                   "log_domain_matmul": {"engine": "unavailable"},
                   "exp": {"engine": "fabric", "lookup_table_reads": 1},
                   "log": {"engine": "fabric", "lookup_table_reads": 1},
                   "softmax_normalize": {"engine": "fabric", "lookup_table_reads": 1, "dsp_multiplies": 1},
                   "attention_score_matmul": {"engine": "dsp_mac_lanes"}},
}


def _self_test() -> None:
    import copy

    # -- 1. configs load; parameters come from the files; shared parts equal --
    platforms = load_all()
    nl, al = platforms["nl_dpe"], platforms["azure_lily"]
    for p in platforms.values():
        t = p.raw["compute_tile"]
        assert (p.rows, p.cols, p.buffer_size, p.input_bit_slices, p.port_width_bits) == \
            (t["rows"], t["cols"], t["buffer_size"], t["input_bit_slices"], t["port_width_bits"])
    assert list(nl.components) == ["crossbar", "input_buffer", "output_buffer", "acam"]
    assert list(al.components) == ["crossbar", "input_buffer", "output_buffer", "adc"]
    for name in SHARED:
        assert nl.components[name] == al.components[name], name
    drifted = copy.deepcopy(al)
    drifted.components["input_buffer"] = {**al.components["input_buffer"], "energy_pj_per_byte": 0.005}
    try:
        _check_shared([nl, drifted])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a non-identical shared component")

    # -- 2. NL area ≡ area_power.py minus its DAC term (DSE reference table) --
    for r, c, mwta, um2 in _DSE_REFERENCE:
        p = Platform("nl_dpe", rows=r, cols=c, buffer_size=max(r, c))
        dac_um2 = 78.2 * c / 256  # area_power.py:72-73 (no DAC here)
        assert 0 <= p.total_area_um2 + dac_um2 - um2 < 1, (r, c, p.total_area_um2 + dac_um2, um2)
        assert 0 <= p.total_area_mwta + dac_um2 / MWTA_UM2 - mwta < 1, (r, c)
    print(f"NL area + area_power.py's DAC term reproduces dse/results/config_tile_reference.csv: "
          f"{len(_DSE_REFERENCE)}/{len(_DSE_REFERENCE)} geometries")

    # -- 3. event energies: nn/ granularity on the shared parts --------------
    for p in (nl, al):
        assert abs(p.event_energy_pj["crossbar"] - 1.31 * p.rows * p.cols / 65536) <= 1e-12
        assert p.event_energy_pj["input_buffer"] == p.event_energy_pj["output_buffer"] == 4.95e-6 * 1e3
    assert abs(nl.event_energy_pj["acam"] - (43.52 + 0.235)) <= 1e-12
    assert al.event_energy_pj["adc"] == 0.00233 * 1e3

    # -- 4. area scaling: each component by its own unit count ---------------
    base = Platform("nl_dpe")
    for kw, doubles in (({"rows": 512}, {"crossbar"}),
                        ({"cols": 512}, {"crossbar", "acam"}),
                        ({"buffer_size": 512}, {"input_buffer", "output_buffer"})):
        q = Platform("nl_dpe", **kw)
        for name in base.components:
            k = 2 if name in doubles else 1
            assert q.area_um2[name] == k * base.area_um2[name], (kw, name)
            assert q.event_energy_pj[name] == (k if name in ("crossbar", "acam") else 1) * base.event_energy_pj[name]
    for cols, n_adc in ((1, 1), (15, 1), (16, 1), (17, 2), (128, 8), (129, 9)):
        q = Platform("azure_lily", cols=cols)
        assert q.units["adc"] == n_adc and q.area_um2["adc"] == n_adc * 4400.0, (cols, q.units)
    assert abs(al.total_area_um2 - (512 * 128 * 11534 / 65536 + 2 * 512 * 354.2 / 256 + 8 * 4400.0)) <= 1e-6
    assert round(al.total_area_um2, 1) == 48150.8
    xml_um2 = _AL_ARCH_XML_MWTA * MWTA_UM2
    print(f"AL area {al.total_area_um2:.1f} um2 vs published arch-XML tile {xml_um2:.1f} um2 "
          f"(x{xml_um2 / al.total_area_um2:.3f}; differing witness, not gated)")
    as_dict = copy.deepcopy(nl.raw)
    as_dict["compute_tile"].update(rows=512, cols=128, buffer_size=512)
    from_dict, from_cli = Platform(as_dict), Platform("nl_dpe", rows=512, cols=128, buffer_size=512)
    assert (from_dict.area_um2, from_dict.event_energy_pj) == (from_cli.area_um2, from_cli.event_energy_pj)

    # -- 5. operator realizations follow from the output stage modes ---------
    for p in (nl, al):
        r = p.operator_realizations
        assert {op: r[op] for op in _CHARTER_REALIZATIONS[p.name]} == _CHARTER_REALIZATIONS[p.name]
    assert nl.operator_realizations["relu"] == {"engine": "tile_pass", "mode": "relu"}
    assert nl.operator_realizations["attention_score_matmul"] == {"engine": "dimm"}
    assert realize_operators(["identity"]) == al.operator_realizations
    r = realize_operators(["identity", "exp"])
    assert r["exp"]["engine"] == "tile_pass" and r["log"] == FABRIC_FALLBACK["log"]
    assert r["log_domain_matmul"] == {"engine": "unavailable"}

    # -- 6. loader edges + CLI smoke -----------------------------------------
    assert Platform(nl.path).event_energy_pj == Platform("nl_dpe.json").event_energy_pj == nl.event_energy_pj
    try:
        Platform("no_such_platform")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("resolved a missing config")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["--imc", "nl_dpe", "azure_lily", "--cols", "128"]) == 0
    text = out.getvalue()
    assert "256 x 128 crossbar" in text and "512 x 128 crossbar" in text, text
    assert "T_steady 60" in text and "T_steady 264" in text, text
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            main(["--imc", "no_such_platform"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("CLI accepted a missing config")

    print("platform self-test: ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
