"""Model-only, the 432 DSE builds: with 4 PV B credits per GEMM copy,
(a) the layer's cycles equal those with no PV B credit at all (the credit never binds), and
(b) the PV DIMM's row registers hold at most one row per channel at any time (the RTL has two per channel)."""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.workloads.attention_head import attention_head, head_params

SHAPES = [(256, 768, 64), (256, 192, 64), (128, 768, 64), (128, 128, 64), (32, 64, 32), (128, 128, 16)]
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]


class Rec:
    def __init__(self):
        self.put_t, self.rel_t = {}, {}

    def put(self, u, t):
        self.put_t[u] = t

    def release(self, u, t):
        self.rel_t[u] = t


def go(lp, K):
    g, parts = attention_head(lp, values=False)
    ctrl, pv = parts["ctrl"], parts["pv"]
    rec = Rec()
    pv.b_stage = rec
    orig = ctrl.need

    def need(proc, k):
        nd = orig(proc, k)
        if proc.startswith("x") and "pb" in nd.credits and nd.credits["pb"] is not None:
            i = int(proc[1:]) + (k + 1) * lp["P_PROJ"] - K * lp["P_PROJ"]
            nd.credits["pb"] = i if i >= 0 else None
        return nd
    ctrl.need = need
    cyc = run(g).cycles
    ch = lp["pv"]["B_CH"]
    worst = 0
    for c in range(ch):
        rows = [t for t in sorted(rec.put_t) if t % ch == c]
        for j, t in enumerate(rows):
            worst = max(worst, sum(1 for u in rows[:j + 1] if rec.rel_t[u] > rec.put_t[t]))
    return cyc, worst


n_eq = n_one = total = 0
worst_all = 0
for (n, dm, d) in SHAPES:
    for R, C in CONFIGS:
        for pp in (1, 2, 4):
            for ne in (d, 2 * d):
                lp = head_params(n, dm, d, R, C, P_PROJ=pp, N_E=ne)
                (c4, w4), (cn, wn) = go(lp, 4), go(lp, 10 ** 6)
                total += 1
                n_eq += c4 == cn
                n_one += max(w4, wn) <= 1
                worst_all = max(worst_all, w4, wn)
                if c4 != cn or max(w4, wn) > 1:
                    print(f"  n {n} dm {dm} d {d} {R}x{C} P {pp} N_E {ne}: K=4 {c4} (regs {w4}), none {cn} (regs {wn})")
print(f"{total} builds: cycles(4 credits) == cycles(no credit) on {n_eq}; at most 1 row per channel in the row "
      f"registers on {n_one} (worst {worst_all})")
