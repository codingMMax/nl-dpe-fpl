#!/usr/bin/env python3
"""run_softmax_rtl.py — GATE 2 of `softmax_online_top` (ip/softmax/spec.md §8) through the shared runner
tools/rtl_gate2.py. The testbench also reports the values on E5 / E6 / E7, gaps inside windows, the word counts and
one done per run.

Usage: python3 v3/ip/softmax/tb/run_softmax_rtl.py --stimuli DIR [--only SUBSTR] [--jobs N] [--work DIR] [--recheck]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[2]
sys.path.insert(0, str(V3))
from tools.rtl_gate2 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(tb=HERE / "tb_softmax_online_top.v",
                  rtl=[V3 / "ip/xbar/rtl/sim/dpe_nldpe.v", V3 / "ip/softmax/rtl/sim/softmax_online_top.v"],
                  top="tb_softmax_online_top", pass_mark="[tb_smol] PASS", name="run_softmax_rtl"))
