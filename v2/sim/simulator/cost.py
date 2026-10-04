#!/usr/bin/env python3
"""cost.py — energy + cycle cost functions over a `Platform` (backbone phase 1).

Charter `v2/spec/simulator.md` §4–§5. One pass of a tile charges each component
(events per pass) x (its event energy at the platform's geometry), at the
Azure-Lily event simulator's granularity (`archive/azurelily_simulator/nn/`):

  bit_slice     crossbar        input_bit_slices fires (whole array; area_power.py energy — nn/ has none)
  input_byte    input_buffer    3 x rows bytes   (nn/: ext-buffer write + read, internal-buffer write)
  output_byte   output_buffer   2 x cols bytes   (nn/: write + read)
  pass          acam            1 fire (all columns: whole-array policy, §4.2)
  conversion    adc             input_bit_slices x active columns

plus fabric operations (block RAM per access of port_width/8 bytes — charter SIM4, where nn/
charges its SRAM per byte — CLB adds / compares / activations, DSP MACs, lookup-table reads),
the one v2 cycle law for any tile (SIM16, delegated verbatim to the certified
`nldpe_sim.cycle_model`), and the tile's steady-state power (pass energy / T_steady).

What it is not: no schedule and no pass counting — every count is injected by the caller
(charter §4.1 / P28); no values; no composition. Energies are analytical, never measured
hardware truth (charter §9.3).

Run:  python3 v2/sim/simulator/cost.py
"""

from __future__ import annotations

import math
import numbers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_sim  # noqa: E402  (read-only, certified)
from platforms import BUFFER_ACCESSES, COMPONENTS  # noqa: E402


def _count(where: str, n) -> int:
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0:
        raise ValueError(f"{where}: expected a non-negative integer count, got {n!r}")
    return int(n)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# Crossbar pass (charter §4.2)
# ---------------------------------------------------------------------------
def tile_pass_energy(p, active_columns: int | None = None) -> dict:
    n_active = p.cols if active_columns is None else _count("active_columns", active_columns)
    if n_active > p.cols:
        raise ValueError(f"active_columns {n_active} > cols {p.cols}")
    events = {"bit_slice": p.input_bit_slices,
              "input_byte": BUFFER_ACCESSES["input_byte"] * p.rows,
              "output_byte": BUFFER_ACCESSES["output_byte"] * p.cols,
              "pass": 1,
              "conversion": p.input_bit_slices * n_active}
    out = {name: events[COMPONENTS[name][3]] * p.event_energy_pj[name] for name in p.components}
    out["total"] = sum(out.values())
    return out


def pass_port_accesses(p) -> tuple[int, int]:
    return _ceil_div(p.rows * 8, p.port_width_bits), _ceil_div(p.cols * 8, p.port_width_bits)


def pass_traffic_energy(p) -> float:
    return sum(pass_port_accesses(p)) * p.fabric["block_ram"]["energy_pj_per_access"]


def weight_setup_energy(p, n_crossbars: int) -> float:
    # charter §4.5: ceil(R·C/5) accesses per crossbar, reported as setup energy
    return _count("n_crossbars", n_crossbars) * bram_energy(p.fabric, p.rows * p.cols)


def tile_cycle_law(p, passes: int) -> nldpe_sim.CycleModel:
    if isinstance(passes, bool) or not isinstance(passes, numbers.Integral) or passes < 1:
        raise ValueError(f"passes must be an integer >= 1, got {passes!r}")
    return nldpe_sim.cycle_model(int(passes), p.rows, p.cols, p.input_bit_slices, p.port_width_bits)


def tile_steady_power_mw(p) -> float:
    # pJ per pass / (T_steady cycles x 1000/f_MHz ns) = mW
    return tile_pass_energy(p)["total"] * p.system_clock_mhz / (1000 * tile_cycle_law(p, 1).t_steady)


# ---------------------------------------------------------------------------
# Fabric operations (charter §4.3–§4.4)
# ---------------------------------------------------------------------------
def bram_accesses(fabric: dict, n_bytes: int) -> int:
    return _ceil_div(_count("n_bytes", n_bytes) * 8, fabric["block_ram"]["port_width_bits"])


def bram_energy(fabric: dict, n_bytes: int) -> float:
    return bram_accesses(fabric, n_bytes) * fabric["block_ram"]["energy_pj_per_access"]


def fabric_add(fabric: dict, n: int) -> float:
    return _count("n", n) * fabric["clb"]["add_pj"]


def fabric_compare(fabric: dict, n: int) -> float:
    return _count("n", n) * fabric["clb"]["compare_pj"]


def fabric_activation(fabric: dict, n: int) -> float:
    return _count("n", n) * fabric["clb"]["activation_unit_pj"]


def dsp_mac(fabric: dict, n: int) -> float:
    return _count("n", n) * fabric["dsp"]["multiply_accumulate_pj"]


def lookup_table_read(fabric: dict, n: int) -> float:
    clb = fabric["clb"]
    return _count("n", n) * clb["lookup_table_read_pj"] * clb["lookup_activity_factor"]


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/simulator/cost.py
# ---------------------------------------------------------------------------
_NN_BUFFER_PJ_PER_BYTE = 4.95e-6 * 1e3  # nn/constant.py BUFFER_*_ENERGY (nJ/byte -> pJ)
_NN_ADC_PJ = 0.00233 * 1e3               # nn/constant.py ADC_ENERGY (per input bit per column)


def _nn_linear_tile_pj(K: int, N: int) -> float:
    # nn/linear_layer.py per input vector, buffers + ADC (SRAM excluded): read() :115,
    # compute() :131-134, write() :158; BIT_WIDTH = 8, one DPE vertically
    b = _NN_BUFFER_PJ_PER_BYTE
    return b * K + (b * K + b * K + _NN_ADC_PJ * N * 8 + b * N) + b * N


def _pin(label: str, computed: float, frozen: float, tol: float, ndigits: int = 2) -> None:
    got = round(computed, ndigits)
    print(f"  {label:34s} {computed:>18.6f}  pin {frozen}")
    assert abs(got - frozen) <= tol, f"{label}: {computed} (rounded {got}) != pin {frozen} ± {tol}"


def _expect_reject(label: str, fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"accepted invalid input: {label}")


def _self_test() -> None:
    import platforms as plat  # noqa: E402

    nl, al = plat.Platform("nl_dpe"), plat.Platform("azure_lily")
    nl128 = plat.Platform("nl_dpe", cols=128)
    fabric = nl.fabric

    # -- A. per pass, NL tile -------------------------------------------------
    print("A. NL pass (256x256, port 40, buffers 256)")
    e = tile_pass_energy(nl)
    assert set(e) == set(nl.components) | {"total"}
    _pin("crossbar (8 fires)", e["crossbar"], 10.48, 0.005)
    _pin("input buffer (3 x 256 bytes)", e["input_buffer"], 3.80, 0.005)
    _pin("output buffer (2 x 256 bytes)", e["output_buffer"], 2.53, 0.005)
    _pin("acam (1 fire, all columns)", e["acam"], 43.755, 0.0005, 3)
    _pin("tile total", e["total"], 60.57, 0.005)
    assert pass_port_accesses(nl) == (52, 52)
    _pin("pass traffic", pass_traffic_energy(nl), 5.15, 0.005)
    _pin("per-pass total", e["total"] + pass_traffic_energy(nl), 65.72, 0.005)
    _pin("steady power (mW)", tile_steady_power_mw(nl), 0.3029, 0.00005, 4)
    for n in (0, 1, 128, 256):
        assert tile_pass_energy(nl, n) == e
    _pin("256x128 tile total", tile_pass_energy(nl128)["total"], 32.19, 0.005)
    assert pass_port_accesses(nl128) == (52, 26)
    print(f"  witness: charter 75.01 (archive: crossbar + DAC + buffers lumped as 3.89/fire) vs "
          f"{e['total']:.4f} (nn/ buffers + area_power.py crossbar/ACAM, no DAC)")

    # -- B. per pass, Azure-Lily tile ≡ nn/ ground truth ---------------------
    print("B. AL pass (512x128, port 16, buffers 512)")
    ea = tile_pass_energy(al)
    _pin("crossbar (8 fires)", ea["crossbar"], 10.48, 0.005)
    _pin("input buffer (3 x 512 bytes)", ea["input_buffer"], 7.60, 0.005)
    _pin("output buffer (2 x 128 bytes)", ea["output_buffer"], 1.27, 0.005)
    # charter §4.2/§10.1 and brief §4 print 2385.28: arithmetic slip; 8·2.33·128 = 2385.92
    _pin("adc (8 x 128 conversions)", ea["adc"], 2385.92, 0.005)
    _pin("tile total", ea["total"], 2405.27, 0.005)
    assert abs(ea["total"] - ea["crossbar"] - _nn_linear_tile_pj(512, 128)) <= 1e-9, "AL tile != nn/ + crossbar"
    for rows, cols in ((512, 128), (512, 64), (512, 16), (512, 1), (256, 128), (1, 1)):
        q = plat.Platform("azure_lily", rows=rows, cols=cols)
        t = tile_pass_energy(q)
        assert abs(t["total"] - t["crossbar"] - _nn_linear_tile_pj(rows, cols)) <= 1e-9 * cols, (rows, cols)
    print(f"  witness: nn/ (no crossbar term) {_nn_linear_tile_pj(512, 128):.4f} + crossbar "
          f"{ea['crossbar']:.4f} = {ea['total']:.4f} ({ea['crossbar'] / ea['total'] * 100:.2f}% crossbar)")
    partial = tile_pass_energy(al, 64)["total"] - tile_pass_energy(al, 64)["crossbar"]
    print(f"  witness: partial pass (64 of 128 columns) {partial:.4f} vs nn/ layer N=64 "
          f"{_nn_linear_tile_pj(512, 64):.4f} (output buffer charged as a full {al.cols}-byte burst, §4.2)")
    assert pass_port_accesses(al) == (256, 64)
    _pin("pass traffic", pass_traffic_energy(al), 15.84, 0.005)
    _pin("per-pass total", ea["total"] + pass_traffic_energy(al), 2421.11, 0.005)
    _pin("steady power (mW)", tile_steady_power_mw(al), 2.7333, 0.00005, 4)
    print(f"  witness: nn/ SRAM {_NN_BUFFER_PJ_PER_BYTE * (512 + 128):.3f} pJ/B-rate vs block RAM "
          f"{pass_traffic_energy(al):.3f} pJ per-access (charter SIM4)")
    for n in (0, 1, 64, 128):
        assert tile_pass_energy(al, n)["adc"] == 8 * n * 2.33

    # -- C. DIMM worked example (charter §10.2; pass counts injected) --------
    print("C. DIMM worked example (M=N=128, K=64; 4160 passes injected)")
    M = N = 128
    K = 64
    passes = {"logA": 32, "logB": 32, "exp": 4096}
    n_pass = sum(passes.values())
    assert n_pass == 4160
    port_in, port_out = pass_port_accesses(nl)
    per_access = fabric["block_ram"]["energy_pj_per_access"]
    crossbar = n_pass * e["total"]
    adds = fabric_add(fabric, M * N * K)
    in_acc, out_acc = n_pass * port_in, n_pass * port_out
    parked = bram_accesses(fabric, K * (M + N))
    ser = bram_accesses(fabric, M * N * 4)
    accesses = in_acc + out_acc + parked + ser
    bram = accesses * per_access
    total = crossbar + adds + bram
    assert (in_acc, out_acc, parked, ser, accesses) == (216320, 216320, 3277, 13108, 449025)
    _pin("tile passes", crossbar, 251975.36, 0.005)
    _pin("farm feed adds", adds, 89107.99, 0.005)
    _pin("block RAM", bram, 22226.74, 0.005)
    _pin("total", total, 363310.09, 0.005)
    _pin("row: logA pool passes", passes["logA"] * e["total"], 1938.27, 0.005)
    _pin("row: exp farm passes", passes["exp"] * e["total"], 248098.82, 0.005)
    _pin("per output element", total / (M * N), 22.17, 0.005)
    print(f"  witness: charter 423,376.3 (archive convention) vs {total:,.1f}")

    # -- D. cycle-law delegation (charter §5.3, SIM16) -----------------------
    print("D. cycle law")
    for name, p, want in (("NL 256x256", nl, (52, 10, 52, 114, 60)),
                          ("AL 512x128", al, (256, 10, 64, 330, 264)),
                          ("NL 256x128", nl128, (52, 10, 26, 88, 60))):
        cm = tile_cycle_law(p, 1)
        got = (cm.load_cyc, cm.compute_cyc, cm.output_cyc, cm.t_fill, cm.t_steady)
        print(f"  {name:10s} LOAD/COMPUTE/OUTPUT/T_fill/T_steady = {tuple(int(v) for v in got)}")
        assert got == want, f"{name}: {got} != {want}"
        assert pass_port_accesses(p) == (cm.load_cyc, cm.output_cyc)
        for n in (1, 2, 7, 4160):
            assert tile_cycle_law(p, n).total == cm.t_fill + (n - 1) * cm.t_steady
    assert tile_cycle_law(nl, nldpe_sim.np.int64(3)).total == 114 + 2 * 60

    # -- E. dual compute: standalone formulas vs the platform walk ------------
    b, xb = _NN_BUFFER_PJ_PER_BYTE, 8 * 1.31  # crossbar: 8 fires of a 65536-cell array
    dual = (
        ("NL tile", e["total"], xb + 3 * 256 * b + 2 * 256 * b + (43.52 + 0.235)),
        ("NL 256x128 tile", tile_pass_energy(nl128)["total"],
         xb / 2 + 3 * 256 * b + 2 * 128 * b + (43.52 + 0.235) / 2),
        ("AL tile", ea["total"], xb + 3 * 512 * b + 2 * 128 * b + 8 * 128 * 2.33),
        ("NL traffic", pass_traffic_energy(nl), (math.ceil(256 * 8 / 40) * 2) * 0.0495),
        ("AL traffic", pass_traffic_energy(al), (math.ceil(512 * 8 / 16) + math.ceil(128 * 8 / 16)) * 0.0495),
        ("DIMM total", total, 4160 * (xb + 5 * 256 * b + 43.52 + 0.235) + 128 * 128 * 64 * 0.08498
         + (4160 * 52 * 2 + math.ceil(64 * 256 / 5) + math.ceil(128 * 128 * 4 / 5)) * 0.0495),
    )
    for label, walk, standalone in dual:
        assert abs(walk - standalone) <= 1e-9 * max(1.0, abs(standalone)), f"dual {label}: {walk} != {standalone}"

    # -- adversarial: counts, boundaries, guards -----------------------------
    for n_bytes, want_acc in ((0, 0), (1, 1), (4, 1), (5, 1), (6, 2), (10, 2), (11, 3)):
        assert bram_accesses(fabric, n_bytes) == want_acc, (n_bytes, bram_accesses(fabric, n_bytes))
    unit = {fabric_add: 0.08498, fabric_compare: 0.26439, fabric_activation: 0.45,
            dsp_mac: 1.2, lookup_table_read: 2.64}
    for fn, rate in unit.items():
        assert fn(fabric, 0) == 0.0 and fn(fabric, 1) == rate and fn(fabric, 2 ** 40) == 2 ** 40 * rate
        for bad in (-1, 2.5, True, None):
            _expect_reject(f"{fn.__name__}({bad!r})", lambda fn=fn, bad=bad: fn(fabric, bad))
    half = {**fabric, "clb": {**fabric["clb"], "lookup_activity_factor": 0.5}}
    assert lookup_table_read(half, 10) == 10 * 2.64 * 0.5
    for bad in (129, -1, 2.5, True):
        _expect_reject(f"AL active_columns={bad!r}", lambda bad=bad: tile_pass_energy(al, bad))
    for bad in (0, -1, 2.5, True, None):
        _expect_reject(f"tile_cycle_law passes={bad!r}", lambda bad=bad: tile_cycle_law(nl, bad))
    setup1 = weight_setup_energy(nl, 1)
    assert setup1 == 13108 * 0.0495 and weight_setup_energy(nl, 0) == 0.0
    assert weight_setup_energy(nl, 2) == 2 * setup1 != bram_energy(fabric, 2 * 256 * 256)
    assert weight_setup_energy(al, 1) == 13108 * 0.0495

    print("cost self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
