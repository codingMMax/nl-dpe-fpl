"""Model-only experiment: the layer's cycles when the PV B port (TRANS_B = 0) holds K rows per copy instead of 2."""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
import top.model as cm
from sim.core.engine import run, Need
from sim.stages import Buffer
from sim.workloads.attention_head import attention_head, head_params

def cycles(lp, K):
    g, parts = attention_head(lp, values=False)
    ctrl, pv = parts["ctrl"], parts["pv"]
    pv.b_stage = Buffer(K * lp["pv"]["B_CH"], "pv: B row register")
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
    xs = sorted(int(e.split()[0]) for e in rep.events if " x0.x win " in e and " w0" in e)
    period = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else 0
    return rep.cycles, period

for (n, dm, d, R, C, pp) in ((128, 128, 64, 64, 64, 1), (128, 128, 64, 128, 128, 1), (256, 768, 64, 256, 256, 1),
                             (256, 192, 64, 256, 128, 2), (32, 64, 32, 128, 64, 1), (128, 128, 16, 128, 128, 4)):
    lp = head_params(n, dm, d, R, C, P_PROJ=pp)
    res = {K: cycles(lp, K) for K in (2, 4, 8, 64)}
    base = res[2][0]
    print(f"n {n:3d} dm {dm:3d} d {d:2d} {R}x{C} P_proj {pp}: GEMM row period T_p ~ {lp['LOADX'] + 8}; "
          + "; ".join(f"K={K}: {c} cycles ({100 * (c - base) / base:+.1f}%), X period {p:.1f}" for K, (c, p) in res.items()))
