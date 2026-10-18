"""Model-only: N_EP = d against the reference N_EP = 2d on the 432 DSE builds (6 shapes x 12 configs x P_PROJ x N_E).
Per build: inside the modules' domain or not (and why), the head's cycles, the crossbar count."""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.workloads.attention_head import attention_head, head_params

SHAPES = {(256, 768, 64): "ViT-B/16", (256, 192, 64): "DeiT-T", (128, 768, 64): "BERT-base",
          (128, 128, 64): "BERT-Tiny", (32, 64, 32): "JetFormer-150P", (128, 128, 16): "JetFormer-JetClass"}
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]


def one(n, dm, d, R, C, pp, ne, nep):
    try:
        lp = head_params(n, dm, d, R, C, P_PROJ=pp, N_E=ne, N_EP=nep)
        return run(attention_head(lp, values=False)[0]).cycles, lp["n_xbar"], None
    except Exception as e:                       # a module guard, or a model rate check
        return None, None, f"{type(e).__name__}: {str(e)[:110]}"


rows, bad = [], {}
for (n, dm, d), name in SHAPES.items():
    for R, C in CONFIGS:
        for pp in (1, 2, 4):
            for ne in (d, 2 * d):
                c2, x2, e2 = one(n, dm, d, R, C, pp, ne, 2 * d)
                c1, x1, e1 = one(n, dm, d, R, C, pp, ne, d)
                assert e2 is None, (name, R, C, pp, ne, e2)
                if e1:
                    bad.setdefault((min(R, C), e1.split(":")[0]), []).append((name, R, C, pp, ne, e1))
                rows.append((name, R, C, pp, ne, c2, x2, c1, x1))
print(f"{len(rows)} builds at N_EP = 2d, all inside the domain")
for (I, kind), lst in sorted(bad.items()):
    print(f"N_EP = d outside the domain at I = {I}: {len(lst)} builds ({kind}), e.g. {lst[0][0]} {lst[0][1]}x{lst[0][2]}"
          f" P {lst[0][3]} N_E {lst[0][4]}: {lst[0][5]}")
ok = [r for r in rows if r[7] is not None]
print(f"N_EP = d inside the domain on {len(ok)} builds")
for I in (64, 128, 256):
    s = [r for r in ok if min(r[1], r[2]) == I]
    if not s:
        continue
    ratio = [r[7] / r[5] for r in s]
    saved = sorted({r[6] - r[8] for r in s})
    print(f"  I = {I:3d}: {len(s):3d} builds, cycles x{min(ratio):.3f} .. x{max(ratio):.3f} (mean x{sum(ratio)/len(ratio):.3f}),"
          f" crossbars saved {saved}")
for name in SHAPES.values():
    s = [r for r in ok if r[0] == name]
    if s:
        ratio = [r[7] / r[5] for r in s]
        xr = [r[8] / r[6] for r in s]
        print(f"  {name:18s}: {len(s):2d} builds, cycles x{min(ratio):.3f} .. x{max(ratio):.3f}, crossbars x{min(xr):.3f} .. "
              f"x{max(xr):.3f}, cycles x crossbars x{min(a * b for a, b in zip(ratio, xr)):.3f} .. "
              f"x{max(a * b for a, b in zip(ratio, xr)):.3f}")
