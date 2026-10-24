#!/usr/bin/env python3
"""run_gemm_gate2.py — GATE 2 of `gemm_top` (ip/gemm/spec.md §7) through the shared runner tools/rtl_gate2.py.
The testbench also checks the values, the output windows without gaps, dpe_done and every tile's err flag.

Usage: python3 v3/ip/gemm/tb/run_gemm_gate2.py --stimuli DIR [--only SUBSTR] [--jobs N] [--work DIR] [--recheck]
       (the corpus: python3 v3/ip/gemm/gen_cases.py, default v3/work/corpus/bitplane/gemm_64x64)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[2]
sys.path.insert(0, str(V3))
from tools.rtl_gate2 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(tb=HERE / "tb_gemm_top.v",
                  rtl=[V3 / "ip/xbar/rtl/sim/dpe_nldpe.v", V3 / "ip/gemm/rtl/sim/gemm_top.v"],
                  top="tb_gemm_top", pass_mark="[tb_gemm] PASS", name="run_gemm_gate2"))
