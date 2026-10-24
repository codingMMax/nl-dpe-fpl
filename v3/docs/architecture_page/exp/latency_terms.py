"""Model-only: where a head's cycles go, and what each tunable moves.

Part 1, the 72 DSE rows at N_E = N_EP = 2d, P_PROJ 1 and 4:
  front  = the first QK output window (the run's Q and K block 0 projected, plus one QK job)
  period = the median gap between successive QK jobs' first output windows (the core's block period)
  tail   = cycles after the last PV output window (the accumulator's last jobs, normalize, O read-out)
  floor  = the score stream ceil(S * B_kv / 5) cycles per block (one pre-bit-plane port into the online softmax)
  stage estimates per block (pass grammar: fill 2 LW + 10, steady LW + 8, LW = ceil(I / 5)):
     QK = fill + (T_W - 1) steady + A_GRP + 2 + log2 d + 4, T_W = (S B_kv / I) / (N_E / d)
     PV = the same with T_W = (S d / I) / (N_EP / B_kv), log2 B_kv
     EXP = ceil(windows / N_EXP) steady, windows = S B_kv / min(S B_kv, I)
     ACC = fill + (ceil(P_oc / N_OC) - 1) steady, P_oc = S d / I
     KV  = B_kv * T_p / P_PROJ (a key block's projection), T_p = LOADX + 8
Part 2, one parameter at a time around P_PROJ 4, N_E = N_EP = 2d, N_EXP 2, B_kv 16, X_S 16, at three rows.
"""
import statistics as st
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.workloads.attention_head import attention_head, head_params

SHAPES = [("ViT-B/16", 256, 768, 64), ("DeiT-T", 256, 192, 64), ("BERT-base", 128, 768, 64),
          ("BERT-Tiny", 128, 128, 64), ("JetFormer-150P", 32, 64, 32), ("JetFormer-JetClass", 128, 128, 16)]
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]


def clog2(x):
    return max(0, (x - 1).bit_length())


def stages(lp, parts):
    I, S, d, bkv = lp["I"], lp["S"], lp["D"], lp["BKV"]
    LW = -(-I // 5)
    fill, steady = 2 * LW + 10, LW + 8
    qk, pv = parts["qk"].d, parts["pv"].d
    t_qk = fill + (qk["T_W"] - 1) * steady + qk["A_GRP"] + 2 + clog2(d) + 4
    t_pv = fill + (pv["T_W"] - 1) * steady + pv["A_GRP"] + 2 + clog2(bkv) + 4
    wins = S * bkv // min(S * bkv, I)
    t_exp = -(-wins // lp["N_EXP"]) * steady
    t_acc = fill + (parts["acc"].d["W"] - 1) * steady
    t_kv = bkv * (lp["LOADX"] + 8) / lp["P_PROJ"]
    return dict(floor=-(-S * bkv // 5), QK=t_qk, PV=t_pv, EXP=t_exp, ACC=t_acc, KV=t_kv)


def measure(lp):
    g, parts = attention_head(lp, values=False)
    rep = run(g)
    first = {}
    pvlast = 0
    for e in rep.events:
        f = e.split()
        t, who = int(f[0]), f[1]
        if who == "qk.c" and f[2] == "win" and f[5] == "w0":
            first.setdefault(int(f[4][1:]), t)
        if who == "pv.c" and f[2] == "win":
            pvlast = max(pvlast, t)
    js = sorted(first)
    gaps = [first[j] - first[j - 1] for j in js[1:]]
    return rep.cycles, first[js[0]], (st.median(gaps) if gaps else 0), rep.cycles - pvlast, stages(lp, parts)


print("Part 1: 72 rows, N_E = N_EP = 2d")
for pp in (1, 4):
    rows = []
    for (name, n, dm, d) in SHAPES:
        for R, C in CONFIGS:
            lp = head_params(n, dm, d, R, C, P_PROJ=pp)
            cyc, front, period, tail, s = measure(lp)
            core = cyc - front - tail
            binding = max(("QK", "PV", "EXP", "ACC"), key=lambda k: s[k])
            rows.append((name, R, C, cyc, front, period, tail, core, s, binding))
    fr = [100 * r[4] / r[3] for r in rows]
    co = [100 * r[7] / r[3] for r in rows]
    ta = [100 * r[6] / r[3] for r in rows]
    per = [r[5] / r[8]["floor"] for r in rows if r[5]]
    est = [r[5] / max(r[8]["floor"], r[8]["QK"], r[8]["PV"], r[8]["EXP"], r[8]["ACC"]) for r in rows if r[5]]
    print(f" P_PROJ {pp}: front {min(fr):.0f}-{max(fr):.0f} % of the cycles (median {st.median(fr):.0f}), "
          f"core {min(co):.0f}-{max(co):.0f} % (median {st.median(co):.0f}), tail {min(ta):.0f}-{max(ta):.0f} %")
    print(f"   block period / score-stream floor: {min(per):.2f}-{max(per):.2f} (median {st.median(per):.2f}); "
          f"period / max(stage estimates): {min(est):.2f}-{max(est):.2f} (median {st.median(est):.2f})")
    for r in rows:
        if (r[0], r[1], r[2]) in (("BERT-Tiny", 128, 128), ("ViT-B/16", 1024, 256), ("JetFormer-150P", 128, 64), ("ViT-B/16", 256, 256)):
            s = r[8]
            print(f"   {r[0]:15s} {r[1]:4d}x{r[2]:<3d} cycles {r[3]:6d} = front {r[4]:5d} + core {r[7]:6d} + tail {r[6]:4d}; "
                  f"period {r[5]:.0f} vs floor {s['floor']}, QK {s['QK']}, PV {s['PV']}, EXP {s['EXP']}, ACC {s['ACC']}, "
                  f"KV {s['KV']:.0f}")

print("Part 2: one parameter at a time (base P_PROJ 4, N_E = N_EP = 2d, N_EXP 2, B_kv 16, X_S 16)")
BASE = dict(P_PROJ=4, N_EXP=2, BKV=16, X_S=16)
for (name, n, dm, d, R, C) in (("BERT-Tiny", 128, 128, 64, 128, 128), ("ViT-B/16", 256, 768, 64, 256, 256),
                               ("JetFormer-150P", 32, 64, 32, 128, 64)):
    base_lp = head_params(n, dm, d, R, C, **BASE)
    c0 = measure(base_lp)[0]
    x0 = base_lp["n_xbar"]
    print(f" {name} at {R}x{C}: base {c0} cycles, {x0} crossbars")
    for key, vals in (("P_PROJ", (1, 2, 4)), ("N_E", (d, 2 * d, 4 * d)), ("N_EXP", (1, 2, 4)), ("BKV", (8, 16, 32)),
                      ("X_S", (8, 16, 32))):
        cells = []
        for v in vals:
            kw = dict(BASE)
            kw[key] = v
            try:
                lp = head_params(n, dm, d, R, C, **kw)
                c = measure(lp)[0]
                cells.append(f"{v}: {100 * (c - c0) / c0:+.1f} % cycles, {lp['n_xbar'] - x0:+d} xbar")
            except Exception as e:
                cells.append(f"{v}: outside ({str(e)[:60]})")
        print(f"   {key:6s} " + " | ".join(cells))
