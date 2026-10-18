#!/usr/bin/env python3
"""run_dimm_rtl.py — GATE 2 of `dimm_tile_top` (ip/dimm/spec.md §8) through the shared runner tools/rtl_gate2.py.
The testbench also reports values, the memory images, the slices in lock step and the converters' overrun probes.

Usage: python3 v3/ip/dimm/tb/run_dimm_rtl.py --stimuli DIR [--only SUBSTR] [--jobs N] [--work DIR] [--recheck]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[2]
sys.path.insert(0, str(V3))
from tools.rtl_gate2 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(tb=HERE / "tb_dimm_tile_top.v",
                  rtl=[V3 / "ip/xbar/rtl/sim/dpe_nldpe.v", V3 / "ip/dimm/rtl/sim/dimm_tile_top.v"],
                  top="tb_dimm_tile_top", pass_mark="[tb_dimm] PASS", name="run_dimm_rtl"))
