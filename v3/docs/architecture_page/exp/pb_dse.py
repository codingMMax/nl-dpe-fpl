"""Model-only: the layer's cycles at the 432 DSE builds (top/test_model.py SHAPES x CONFIGS x P_PROJ x N_E) with the
PV B port credited 2 rows per GEMM copy (the RTL) and 4 rows per copy."""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "work" / "f4" / "exp"))
from sim.core.engine import run
from sim.workloads.attention_head import head_params
from pb_loop import build

SHAPES = {(256, 768, 64): "ViT-B/16", (256, 192, 64): "DeiT-T", (128, 768, 64): "BERT-base",
          (128, 128, 64): "BERT-Tiny", (32, 64, 32): "JetFormer-150P", (128, 128, 16): "JetFormer-JetClass"}
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]

rows = []
for (n, dm, d), name in SHAPES.items():
    for R, C in CONFIGS:
        for pp in (1, 2, 4):
            for ne in (d, 2 * d):
                lp = head_params(n, dm, d, R, C, P_PROJ=pp, N_E=ne)
                c2, c4 = run(build(lp, 2)).cycles, run(build(lp, 4)).cycles
                rows.append((name, R, C, pp, ne, c2, c4))
                print(f"{name:18s} {R:4d}x{C:<3d} P {pp} N_E {ne:3d}: {c2:6d} -> {c4:6d} ({100 * (c4 - c2) / c2:+.1f}%)")
hit = [r for r in rows if r[6] < r[5]]
print(f"\n{len(rows)} builds; faster with 4 rows: {len(hit)}; slower: {sum(1 for r in rows if r[6] > r[5])}")
for name in SHAPES.values():
    s = [100 * (r[6] - r[5]) / r[5] for r in rows if r[0] == name]
    h = [x for x in s if x < 0]
    print(f"  {name:18s}: {len(h):2d}/{len(s)} builds faster, saving max {-min(s):.1f}%, mean over those "
          f"{(-sum(h) / len(h)) if h else 0:.1f}%")
