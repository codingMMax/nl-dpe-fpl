"""Model-only, the 432 DSE builds: the fewest PV B credits per GEMM copy that are as fast as no PV B credit."""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.workloads.attention_head import attention_head, head_params

SHAPES = [(256, 768, 64), (256, 192, 64), (128, 768, 64), (128, 128, 64), (32, 64, 32), (128, 128, 16)]
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]


def cycles(lp, K):
    g, parts = attention_head(lp, values=False)
    ctrl = parts["ctrl"]
    orig = ctrl.need

    def need(proc, k):
        nd = orig(proc, k)
        if proc.startswith("x") and "pb" in nd.credits and nd.credits["pb"] is not None:
            i = int(proc[1:]) + (k + 1) * lp["P_PROJ"] - K * lp["P_PROJ"]
            nd.credits["pb"] = i if i >= 0 else None
        return nd
    ctrl.need = need
    return run(g).cycles


need = {}
for (n, dm, d) in SHAPES:
    for R, C in CONFIGS:
        for pp in (1, 2, 4):
            for ne in (d, 2 * d):
                lp = head_params(n, dm, d, R, C, P_PROJ=pp, N_E=ne)
                free = cycles(lp, 10 ** 6)
                k = next(K for K in (1, 2, 3, 4, 5, 6, 8) if cycles(lp, K) == free)
                need[k] = need.get(k, 0) + 1
                if k >= 4:
                    print(f"  needs {k}: n {n} dm {dm} d {d} {R}x{C} P {pp} N_E {ne}")
print("credits per copy needed to match no credit -> builds:", dict(sorted(need.items())))
