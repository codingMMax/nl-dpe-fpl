#!/usr/bin/env python3
"""f2_check.py — the F2 gate: the simulator's tests (simulator/tests/, docs/simulator.md §7). Python only, no RTL run.

  core         testCore.py               records, stages, engine: hand-computed chain, determinism, composition,
                                         errors; the model stack loads no oracle, v2 kernel or pricing module
  crossbar     testCrossbar.py           the crossbar's values vs the v2 crossbar corpora, the pass rule's closed form
                                         (ip/xbar/spec.md, the bit-plane pass of 2026-10-22)
  gemm         testGemm.py               gemm_top's values vs the v2 GEMM corpora, ip/gemm/spec.md's closed form, the RTL
                                         structure
  fc           testFcRows.py             the 48 FC rows of the v2 snapshot: energy, area and Fmax equal to v2; cycles and
                                         latency by the bit-plane closed form
  area         testArea.py               the crossbar tile: the row-parallel rule at its edges, the 9 tiles, the legacy table
  dimm         testDimmTile.py           GATE 1, spec §5's closed form, the spec rules, the memories (spec §6 vs Yosys)
  softmax      testSoftmaxOnline.py      GATE 1, spec §5's core recurrence, the spec rules
  accumulator  testSoftmaxAccumulator.py GATE 1, spec §5's job rules, the spec rules
  head         testAttentionHead.py      GATE 1 on heads, the att_ctrl rules, the 288 DSE builds' domain
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
         ("area", "simulator/tests/testArea.py"),
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
