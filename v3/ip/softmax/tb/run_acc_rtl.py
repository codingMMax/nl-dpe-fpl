#!/usr/bin/env python3
"""run_acc_rtl.py — GATE 2 of `softmax_online_acc` (ip/softmax/acc_spec.md §8) through the shared runner
tools/rtl_gate2.py. The testbench also reports the O8 values, gaps inside rows, run_done against the model's cycle,
the word and credit counts, and the FIFO / e_mem / O reader probes.

Usage: python3 v3/ip/softmax/tb/run_acc_rtl.py --stimuli DIR [--only SUBSTR] [--jobs N] [--work DIR] [--recheck]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[2]
sys.path.insert(0, str(V3))
from tools.rtl_gate2 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(tb=HERE / "tb_softmax_online_acc.v",
                  rtl=[V3 / "ip/xbar/rtl/sim/dpe_nldpe.v", V3 / "ip/lib/rtl/sim/att_fifo.v",
                       V3 / "ip/softmax/rtl/sim/softmax_online_acc.v"],
                  top="tb_softmax_online_acc", pass_mark="[tb_acc] PASS", name="run_acc_rtl"))
