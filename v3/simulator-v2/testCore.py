#!/usr/bin/env python3
"""testCore.py — core.py's Core against the platform layer it was copied from (v3/simulator/platform/platform.py).

Regression (2026-10-20): `python core.py` stopped in Core.__init__: json.load on a string, then the JSON key
buffer_size read as buf_size, then units, areas and energies built over all five COMPONENTS (also the output stage the
tile does not have) and read from COMPONENTS' tuples instead of the tile.
Checks, for both platforms at their own size and at 1 x 1, 128 x 64, 64 x 128, 1000 x 17 (C not a multiple of the 16
columns per ADC) and 1024 x 256: the components (acam or adc, never both), units, areas, per-event energies, modes and
clock equal v3's Platform exactly (but the ADC entry: one bit-slice of the whole bank, C conversions, where v3 keeps one
conversion; user 2026-10-20), and the units, areas and per-event energies equal the rules written out here; a
buffer-size override; one pass's energy (CorePass.pass_energy_pj) equals v3's prices.passEnergyPj for rows used from 1
to R, and three passes by hand; program(): identity on square, wide and tall arrays, full and edge tiles zero-padded
with int8 extremes, a stored copy, reprogramming, the modes each platform has, and the inputs it must reject; the
name, a path and a dict load the same core; an unknown config raises FileNotFoundError;
`python core.py` exits 0. Core checks no geometry (neither does Platform), so zero or negative sizes are not tested.
Run: python3 v3/simulator-v2/testCore.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                                     # no __pycache__ beside core.py
HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path[:0] = [str(HERE), str(V3)]
import core  # noqa: E402
from simulator.platform import prices  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402

CONFIGS = {"nl_dpe": HERE / "nl_dpe.json", "azure_lily": HERE / "azure_lily.json"}   # the copies beside core.py
GEOMETRIES = (None, (1, 1), (128, 64), (64, 128), (1000, 17), (1024, 256))   # None: the JSON's own R x C
AREA_KEY = {"crossbar": "area_um2_per_cell", "input_buffer": "area_um2_per_entry",
            "output_buffer": "area_um2_per_entry", "acam": "area_um2_per_col", "adc": "area_um2_per_adc"}


def testScript() -> None:
    """The failure as reported: `python core.py` from its own folder."""
    r = subprocess.run([sys.executable, "core.py"], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr.strip().splitlines()[-1:]


def testLoads() -> None:
    """The name, a path and a dict give the same core; an unknown name raises."""
    byName = core.Core("nl_dpe")
    byPath = core.Core(str(CONFIGS["nl_dpe"]))
    byDict = core.Core(json.loads(CONFIGS["nl_dpe"].read_text(encoding="utf-8")))
    for c in (byPath, byDict):
        assert (c.rows, c.cols, c.buf_size, c.units, c.area_um2, c.component_energy_pj) == \
            (byName.rows, byName.cols, byName.buf_size, byName.units, byName.area_um2, byName.component_energy_pj)
    assert byName.path == CONFIGS["nl_dpe"].resolve() and byDict.path == Path("<dict>")
    try:
        core.Core("no_such_platform")
    except FileNotFoundError:
        return
    raise AssertionError("an unknown config loaded")


def testAgainstPlatform() -> int:
    """Every config x geometry: the numbers of v3's Platform, compared exactly (the same expressions)."""
    n = 0
    for name, path in CONFIGS.items():
        for g in GEOMETRIES:
            R, C = g or (None, None)
            c, p = core.Core(str(path), rows=R, cols=C), Platform(str(path), rows=R, cols=C)
            assert (c.rows, c.cols, c.buf_size) == (p.rows, p.cols, p.bufferSize), (name, g)
            assert c.components == p.components and c.units == p.units, (name, g)
            assert c.area_um2 == p.areaUm2 and c.total_area_um2 == p.totalAreaUm2, (name, g)
            assert (c.total_area_mm2, c.total_area_mwta) == (p.totalAreaMm2, p.totalAreaMwta), (name, g)
            event = dict(p.eventEnergyPj)
            if "adc" in event:
                event["adc"] *= p.cols                      # v3 prices one conversion, Core one bit-slice of the bank
            assert c.component_energy_pj == event, (name, g)
            assert (c.output_stage, c.modes, c.clock_mhz) == (p.outputStage, p.modes, p.clockMhz), (name, g)
            n += 1
    return n


def testRules() -> int:
    """Units, areas and per-event energies written out: the crossbar fires all R x C cells per bit-slice, the ADC bank
    converts all C columns per bit-slice (C conversions, not ceil(C / 16) x 16: 17 at C = 17), the ACAM fires all C
    columns per pass; a buffer byte is one event; one ADC per 16 columns, rounded up."""
    n = 0
    for name, path in CONFIGS.items():
        tile = json.loads(path.read_text(encoding="utf-8"))["compute_tile"]
        stage = "acam" if name == "nl_dpe" else "adc"
        for g in GEOMETRIES:
            R, C = g or (tile["rows"], tile["cols"])
            c = core.Core(str(path), rows=R, cols=C)
            nStage = C if stage == "acam" else -(-C // tile["adc"]["cols_per_adc"])
            units = {"crossbar": R * C, "input_buffer": tile["buffer_size"], "output_buffer": tile["buffer_size"],
                     stage: nStage}
            assert c.output_stage == stage and c.units == units, (name, g, c.units)
            for k, u in units.items():
                assert math.isclose(c.area_um2[k], u * tile[k][AREA_KEY[k]], rel_tol=1e-12), (name, g, k)
            assert math.isclose(c.total_area_um2, sum(c.area_um2.values()), rel_tol=1e-12), (name, g)
            e = c.component_energy_pj
            assert math.isclose(e["crossbar"], R * C * tile["crossbar"]["energy_pj_per_cell"], rel_tol=1e-12)
            assert e["input_buffer"] == tile["input_buffer"]["energy_pj_per_byte"]
            assert e["output_buffer"] == tile["output_buffer"]["energy_pj_per_byte"]
            if stage == "acam":
                assert math.isclose(e["acam"], C * tile["acam"]["energy_pj_per_col"], rel_tol=1e-12), (name, g)
            else:
                assert math.isclose(e["adc"], C * tile["adc"]["energy_pj_per_conversion"], rel_tol=1e-12), (name, g)
            n += 1
        one = core.Core(str(path), buf_size=1)                         # the override: one entry per buffer
        assert one.units["input_buffer"] == one.units["output_buffer"] == 1, name
        assert one.area_um2["input_buffer"] == tile["input_buffer"]["area_um2_per_entry"], name
    return n


def testPassEnergy() -> int:
    """One pass equals v3's prices.passEnergyPj (the same counting rule) on both platforms, every geometry and rows used
    from 1 to R; three passes by hand: NL-DPE 256 x 256 with 256 rows, Azure-Lily 512 x 128 with 512 rows, and NL-DPE
    1024 x 64 with 64 rows (a conversion pass on a tall array: the input side follows the rows, the rest is the whole
    array); the total is the sum of the parts."""
    n = 0
    for name, path in CONFIGS.items():
        for g in GEOMETRIES:
            R, C = g or (None, None)
            c, p = core.Core(str(path), rows=R, cols=C), Platform(str(path), rows=R, cols=C)
            for rows in sorted(r for r in {1, 5, 6, 64, c.rows // 2, c.rows} if 1 <= r <= c.rows):
                mine, ref = core.CorePass(c, rows).pass_energy_pj, prices.passEnergyPj(p, rows)
                assert mine.keys() == ref.keys(), (name, g, rows, mine.keys())
                assert all(math.isclose(mine[k], ref[k], rel_tol=1e-12) for k in ref), (name, g, rows)
                n += 1
    for name, R, C, rows, total in (("nl_dpe", 256, 256, 256, 60.571), ("azure_lily", 512, 128, 512, 2405.2704),
                                    ("nl_dpe", 1024, 64, 64, 23.00275)):
        e = core.CorePass(core.Core(str(CONFIGS[name]), rows=R, cols=C), rows).pass_energy_pj
        assert math.isclose(e["total"], total, rel_tol=1e-12), (name, R, C, e["total"])
        assert math.isclose(e["total"], sum(v for k, v in e.items() if k != "total"), rel_tol=1e-12), (name, R, C)
    return n


def expectError(fn, what: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"accepted {what}")


def testProgram() -> int:
    """program(): the weights land in the array exactly, zero elsewhere; None is the identity over min(R, C); the
    caller's array is copied; a second program() replaces the first; each platform takes its own modes only; wrong
    shapes, empty tiles, non-integers and values outside int8 raise ValueError."""
    rng = np.random.default_rng(7)
    n = 0
    for R, C in ((128, 64), (64, 128), (256, 256), (1, 1), (1000, 17)):
        c = core.Core(str(CONFIGS["nl_dpe"]), rows=R, cols=C)
        assert c.weights is None and c.mode is None, (R, C)
        c.program()
        assert c.mode == "identity" and c.weights.shape == (R, C), (R, C)
        assert np.array_equal(c.weights, np.eye(R, C, dtype=np.int64)), (R, C)
        assert c.weights.sum() == min(R, C), (R, C)
        for k, m in ((R, C), (1, 1), (max(1, R // 2), max(1, C // 3)), (R, 1), (1, C)):
            W = rng.integers(-128, 128, size=(k, m), dtype=np.int8)
            W[0, 0], W[-1, -1] = -128, 127
            c.program(W, "exp")
            want = np.zeros((R, C), dtype=np.int64)
            want[:k, :m] = W
            assert c.mode == "exp" and c.weights.dtype == np.int64 and np.array_equal(c.weights, want), (R, C, k, m)
            n += 1
    c = core.Core(str(CONFIGS["nl_dpe"]), rows=8, cols=8)
    W = np.full((8, 8), 5, dtype=np.int8)
    c.program(W)
    W[0, 0] = -1                                                       # the caller's array changes afterwards
    assert c.weights[0, 0] == 5, "program() kept a view of the caller's array"
    c.program(np.ones((2, 3), dtype=np.int64), "log")                  # reprogramming: the old 5s are gone
    assert c.weights.sum() == 6 and c.mode == "log"
    for mode in ("identity", "relu", "exp", "log"):
        c.program(mode=mode)
        assert c.mode == mode
    a = core.Core(str(CONFIGS["azure_lily"]), rows=8, cols=8)
    a.program()
    assert a.mode == "identity"
    for mode in ("relu", "exp", "log", "tanh", "IDENTITY", None):
        expectError(lambda: a.program(mode=mode), f"mode {mode!r} on the ADC core")
    expectError(lambda: c.program(mode="tanh"), "an unknown mode")
    bad = {"a vector": np.zeros(8, dtype=np.int8), "a 3-D array": np.zeros((2, 2, 2), dtype=np.int8),
           "R + 1 rows": np.zeros((9, 8), dtype=np.int8), "C + 1 columns": np.zeros((8, 9), dtype=np.int8),
           "no rows": np.zeros((0, 8), dtype=np.int8), "no columns": np.zeros((8, 0), dtype=np.int8),
           "floats": np.full((2, 2), 0.5), "booleans": np.ones((2, 2), dtype=bool),
           "128": np.array([[0, 128]]), "-129": np.array([[-129, 0]]), "a scalar": np.int8(3)}
    for what, w in bad.items():
        expectError(lambda: c.program(w), what)
    assert c.mode == "log", "a rejected program() changed the mode"
    return n


def main() -> int:
    testScript()
    print("  script    PASS  (python core.py exits 0)")
    testLoads()
    print("  loads     PASS  (the name, a path and a dict give the same core; an unknown name raises)")
    n = testAgainstPlatform()
    print(f"  platform  PASS  ({n} config x geometry pairs equal v3's Platform exactly)")
    n = testRules()
    print(f"  rules     PASS  ({n} pairs: units, areas, per-event energies by hand; ADCs rounded up at C = 1 and 17; "
          "buffer override)")
    n = testPassEnergy()
    print(f"  pass      PASS  ({n} passes equal v3's prices.passEnergyPj; 60.571, 2405.2704, 23.00275 pJ by hand)")
    n = testProgram()
    print(f"  program   PASS  ({n} tiles exact on 5 arrays; identity, copy, reprogram, modes per platform, 11 bad inputs)")
    print("core: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
