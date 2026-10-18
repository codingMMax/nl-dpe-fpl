"""Model-only: where the X stream waits, per layer, with the PV B port holding K rows per GEMM copy (K = 2 is the RTL).

Per token t (copy c = t mod P_PROJ): X burst start x(t), the V GEMM's first output word v(t), the PV B credit pb(t).
  loop    = pb(t) + CR_LAT - x(t): from a burst start until its V row's credit counts at att_ctrl
  T_p     = the GEMM copy's own row period (MSB_SA_Ready rising after burst k, minus the burst start)
  X period (per copy, middle half of the tokens); "pb-bound" = tokens whose burst starts at pb(t - K P_PROJ) + 1
"""
import statistics as st
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.stages import Buffer
from sim.workloads.attention_head import attention_head, head_params


def build(lp, K):
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
    return g


def analyse(lp, K):
    rep = run(build(lp, K))
    P = lp["P_PROJ"]
    x, v, pb, act = {}, {}, {}, {}
    for e in rep.events:
        f = e.split()
        t, who, kind = int(f[0]), f[1], f[2]
        if who.endswith(".x") and who.startswith("x") and kind == "win" and f[5] == "w0":
            c, k = int(who[1:-2]), int(f[4][1:])
            x[c + k * P] = t
        elif who.startswith("gv") and who.endswith(".out") and kind == "win" and f[5] == "w0":
            c, k = int(who[2:-4]), int(f[4][1:])
            v[c + k * P] = t
        elif who.startswith("gv") and who.endswith(".act") and kind == "cr":
            c, k = int(who[2:-4]), int(f[3][1:])
            act[c + k * P] = t
        elif who == "pv.b" and kind == "cr":
            pb[int(f[3][1:])] = t
    n = lp["NSEQ"]
    mid = range(n // 4, 3 * n // 4)
    loop = st.median(pb[t] + 1 - x[t] for t in mid)
    gl = st.median(v[t] - x[t] for t in mid)
    tp = st.median(act[t] - x[t] for t in mid if t in act)
    per = st.median(x[t] - x[t - P] for t in mid if t >= P)
    bound = sum(1 for t in range(K * P, n) if x[t] == pb[t - K * P] + 1)
    return rep.cycles, loop, gl, tp, per, bound


rows = ((128, 128, 64, 64, 64, 1, "BERT-Tiny"), (128, 128, 64, 128, 128, 1, "BERT-Tiny"),
        (32, 64, 32, 128, 64, 1, "JetFormer-150P"), (128, 128, 16, 128, 128, 4, "JetFormer-JetClass"),
        (128, 64, 16, 64, 64, 1, "bring-up bu_runs"), (256, 768, 64, 256, 256, 1, "d_m 768"))
for (n, dm, d, R, C, pp, tag) in rows:
    lp = head_params(n, dm, d, R, C, P_PROJ=pp)
    out = {K: analyse(lp, K) for K in (2, 4)}
    c2, loop, gl, tp, per2, b2 = out[2]
    c4, _, _, _, per4, b4 = out[4]
    print(f"{tag:18s} n {n:3d} dm {dm:3d} d {d:2d} {R}x{C} P {pp}: T_p {tp}, loop {loop} (GEMM {gl} + PV B "
          f"{loop - gl}); X period/copy K=2 {per2} (pb-bound {b2}/{n}), K=4 {per4} (pb-bound {b4}/{n}); "
          f"cycles {c2} -> {c4} ({100 * (c4 - c2) / c2:+.1f}%)")
