"""Model-only: N_EP = d with N_OC capped at the accumulator's input channels (P_CH = N_EP / B_kv; guard N_OC <= P_CH),
against the reference N_EP = 2d (N_OC by the floor rule), on the 432 DSE builds."""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.workloads.attention_head import attention_head, head_params, n_oc_floor

SHAPES = {(256, 768, 64): "ViT-B/16", (256, 192, 64): "DeiT-T", (128, 768, 64): "BERT-base",
          (128, 128, 64): "BERT-Tiny", (32, 64, 32): "JetFormer-150P", (128, 128, 16): "JetFormer-JetClass"}
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]
rows, bad = [], {}
for (n, dm, d), name in SHAPES.items():
    for R, C in CONFIGS:
        for pp in (1, 2, 4):
            for ne in (d, 2 * d):
                ref = head_params(n, dm, d, R, C, P_PROJ=pp, N_E=ne)
                c2 = run(attention_head(ref, values=False)[0]).cycles
                noc = min(ref["N_OC"], d // ref["BKV"])
                try:
                    lp = head_params(n, dm, d, R, C, P_PROJ=pp, N_E=ne, N_EP=d, N_OC=noc)
                    c1 = run(attention_head(lp, values=False)[0]).cycles
                    rows.append((name, R, C, pp, ne, c2, ref["n_xbar"], c1, lp["n_xbar"], ref["N_OC"], noc))
                except Exception as e:
                    bad.setdefault((min(R, C), str(e)[:100]), []).append((name, R, C, pp, ne))
for (I, msg), lst in sorted(bad.items()):
    print(f"outside at I = {I}: {len(lst)} builds: {msg}")
print(f"N_EP = d (N_OC capped) inside on {len(rows)} of 432 builds")
for name in SHAPES.values():
    s = [r for r in rows if r[0] == name]
    if not s:
        print(f"  {name:18s}: none inside")
        continue
    cr = [r[7] / r[5] for r in s]
    xr = [r[8] / r[6] for r in s]
    print(f"  {name:18s}: {len(s):2d} builds, N_OC {s[0][9]} -> {s[0][10]}, cycles x{min(cr):.3f}..x{max(cr):.3f},"
          f" crossbars {s[0][6] - s[0][8]} fewer (x{min(xr):.3f}..x{max(xr):.3f}), cycles x crossbars "
          f"x{min(a * b for a, b in zip(cr, xr)):.3f}..x{max(a * b for a, b in zip(cr, xr)):.3f}")
