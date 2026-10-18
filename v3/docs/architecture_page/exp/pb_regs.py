"""Model-only: how many V rows the PV DIMM's row registers hold at once, per channel, when att_ctrl gives the PV B port
K credits per GEMM copy (K = 2 is the RTL; K = 1000 = no PV B credit at all).

A row occupies its register from its first drained word at the slices (a0) to its write (tw): the DIMM model's
`b_stage` put / release. Rows on channel c = t mod B_CH. Also: the converter rate check (it raises on an overrun).
"""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.workloads.attention_head import attention_head, head_params


class Rec:
    """A b_stage that never raises and records each row's [put, release) span."""
    def __init__(self):
        self.put_t, self.rel_t = {}, {}

    def put(self, u, t):
        self.put_t[u] = t

    def release(self, u, t):
        self.rel_t[u] = t


def occupancy(lp, K):
    g, parts = attention_head(lp, values=False)
    ctrl, pv = parts["ctrl"], parts["pv"]
    rec = Rec()
    pv.b_stage = rec
    orig = ctrl.need

    def need(proc, k):
        nd = orig(proc, k)
        if proc.startswith("x") and "pb" in nd.credits and nd.credits["pb"] is not None:
            t1 = int(proc[1:]) + (k + 1) * lp["P_PROJ"]
            i = t1 - K * lp["P_PROJ"]
            nd.credits["pb"] = i if i >= 0 else None
        return nd
    ctrl.need = need
    rep = run(g)
    ch = lp["pv"]["B_CH"]
    worst, gap = 0, None
    for c in range(ch):
        rows = [t for t in sorted(rec.put_t) if t % ch == c]
        for j, t in enumerate(rows):
            live = sum(1 for u in rows[:j + 1] if rec.rel_t[u] > rec.put_t[t])
            worst = max(worst, live)
            if j:
                g_ = rec.put_t[t] - rec.rel_t[rows[j - 1]]
                gap = g_ if gap is None else min(gap, g_)
    span = max(rec.rel_t[u] - rec.put_t[u] for u in rec.put_t)
    return rep.cycles, worst, gap, span


rows = ((128, 128, 64, 64, 64, 1), (128, 128, 64, 128, 128, 1), (32, 64, 32, 128, 64, 1), (128, 128, 16, 128, 128, 4),
        (128, 64, 16, 64, 64, 1), (256, 768, 64, 256, 256, 1), (128, 128, 16, 64, 64, 2), (64, 32, 16, 64, 64, 4),
        (32, 192, 16, 64, 64, 2), (64, 64, 32, 64, 64, 2))
for (n, dm, d, R, C, pp) in rows:
    lp = head_params(n, dm, d, R, C, P_PROJ=pp)
    res = []
    for K in (2, 4, 1000):
        try:
            cyc, worst, gap, span = occupancy(lp, K)
            res.append(f"K={K}: {cyc} cycles, max rows/channel in the registers {worst}, min gap {gap}, span {span}")
        except Exception as e:                      # the converter's rate check, or any other model check
            res.append(f"K={K}: RAISES {type(e).__name__}: {e}")
    print(f"n {n:3d} dm {dm:3d} d {d:2d} {R}x{C} P {pp}: " + "; ".join(res))
