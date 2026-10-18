"""Model-only: the 72 DSE builds at the fixed mapping (P_PROJ 4, N_E = d, N_EP = 2d, N_EXP 2, N_OC floor rule,
B_kv 16, X_S 16). Per build: R_b, the last run's t_o8 (its last O8 word in memory), the head's done, the report's
cycles (the last O8 row out), crossbars. Usage: python3 compute_cycles.py OUT.tsv
"""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
from sim.core.engine import run
from sim.workloads.attention_head import attention_head, head_params

SHAPES = [("ViT-B/16", 256, 768, 64), ("DeiT-T", 256, 192, 64), ("BERT-base", 128, 768, 64),
          ("BERT-Tiny", 128, 128, 64), ("JetFormer-150P", 32, 64, 32), ("JetFormer-JetClass", 128, 128, 16)]
CONFIGS = [(r, c) for r in (128, 256, 512, 1024) for c in (64, 128, 256)]

rows = ["shape\tR\tC\tRB\tt_o8_last\tdone\tcycles_all\tn_xbar"]
for (name, n, dm, d) in SHAPES:
    for R, C in CONFIGS:
        lp = head_params(n, dm, d, R, C, P_PROJ=4, N_E=d, N_EP=2 * d)
        g, parts = attention_head(lp, values=False)
        rep = run(g)
        acc = parts["acc"]
        rows.append(f"{name}\t{R}\t{C}\t{lp['RB']}\t{acc.t_o8[lp['RB'] - 1]}\t{parts['ctrl'].done}\t{rep.cycles}\t"
                    f"{lp['n_xbar']}")
Path(sys.argv[1]).write_text("\n".join(rows) + "\n")
print(f"{len(rows) - 1} builds -> {sys.argv[1]}")
