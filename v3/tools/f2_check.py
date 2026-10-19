#!/usr/bin/env python3
"""f2_check.py — the F2 gate: the simulator's tests (simulator/tests/, docs/simulator.md §7). Python only, no RTL run.

  core         testCore.py               records, stages, engine: hand-computed chain, determinism, composition,
                                         errors; the model stack loads no oracle, v2 kernel or pricing module
  crossbar     testCrossbar.py           the crossbar vs the v2 crossbar corpora, nldpe_sim.cycle_model, pass_engine
  gemm         testGemm.py               gemm_top vs the v2 GEMM corpora, gemm_sim.cycle_model, the RTL structure
  fc           testFcRows.py             the 48 FC rows of the v2 snapshot: cycles, energy, area, Fmax, latency
  dimm         testDimmTile.py           GATE 1, the v2 tile closed form, the spec rules, the Yosys memories
  softmax      testSoftmaxOnline.py      GATE 1, softmax_online_cycle_model, the spec rules
  accumulator  testSoftmaxAccumulator.py GATE 1, ocarry_cycle_model, the spec rules
  head         testAttentionHead.py      GATE 1 on heads, the att_ctrl rules, the 432 DSE builds' domain
Each runs under the isolation probe (tools/probe_site): no access to the repository outside v3/.
Outputs: v3/work/f2/<tag>/{logs,probe}/, summary.txt.
Usage: python3 v3/tools/f2_check.py [--tag T]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tools.f1_check import PY, V3, Ctx, isolation  # noqa: E402

TESTS = (("core", "simulator/tests/testCore.py"), ("trace", "simulator/tests/testTrace.py"),
         ("crossbar", "simulator/tests/testCrossbar.py"),
         ("gemm", "simulator/tests/testGemm.py"), ("fc", "simulator/tests/testFcRows.py"),
         ("dimm", "simulator/tests/testDimmTile.py"), ("softmax", "simulator/tests/testSoftmaxOnline.py"),
         ("accumulator", "simulator/tests/testSoftmaxAccumulator.py"), ("head", "simulator/tests/testAttentionHead.py"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    c = Ctx(a.tag, 1, phase="f2")
    c.say(f"F2 check {a.tag}")
    t0 = time.time()
    for name, script in TESTS:
        rc, dt, log, last, text = c.run(name, [PY, V3 / script], timeout=3600)
        c.say(f"[{name}] {'PASS' if rc == 0 else 'FAIL'} {dt:6.1f}s  [{c.rel(log)}]")
        for ln in text.splitlines()[1:]:
            if ln.startswith("  "):
                c.say("  " + ln.strip())
        if rc != 0:
            c.fails.append(name)
            c.say("  " + last)
    c.say("[isolation]")
    isolation(c)
    c.say(f"F2 check: {'ALL PASS' if not c.fails else 'FAIL: ' + ', '.join(c.fails)} ({time.time() - t0:.0f}s)")
    (c.root / "summary.txt").write_text("\n".join(c.lines) + "\n")
    return 0 if not c.fails else 1


if __name__ == "__main__":
    sys.exit(main())
