#!/usr/bin/env python3
"""run_layer_rtl.py — gate L of `att_layer` (top/spec.md §8) through the shared runner tools/rtl_gate2.py. The
testbench also reports the O8 values, gaps inside windows, x_rdy at every burst start, done at the model's cycle and
the word counts.

Usage: python3 v3/top/tb/run_layer_rtl.py --stimuli DIR [--only SUBSTR] [--jobs N] [--work DIR] [--recheck]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1]
sys.path.insert(0, str(V3))
from tools.rtl_gate2 import main  # noqa: E402

RTL = [V3 / "ip/xbar/rtl/sim/dpe_nldpe.v", V3 / "ip/lib/rtl/sim/att_fifo.v", V3 / "ip/gemm/rtl/sim/gemm_top.v",
       V3 / "ip/dimm/rtl/sim/dimm_tile_top.v", V3 / "ip/softmax/rtl/sim/softmax_online_top.v",
       V3 / "ip/softmax/rtl/sim/softmax_online_acc.v", V3 / "top/rtl/sim/att_ctrl.v", V3 / "top/rtl/sim/att_layer.v"]

if __name__ == "__main__":
    sys.exit(main(tb=HERE / "tb_att_layer.v", rtl=RTL, top="tb_att_layer", pass_mark="[tb_layer] PASS",
                  name="run_layer_rtl"))
