"""Export the head model's derived values, resources and cycles for every DSE build, for the architecture page's
explorer: 6 shapes x 12 configs x P_PROJ {1, 2, 4} x N_E {d, 2d} x N_EP {2d, d (N_OC capped at P_CH)}.
Model only (4 PV B credits per copy, the current RTL).
Usage: python3 v3/docs/architecture_page/page_data.py v3/work/arch_page/builds.json
The JSON keys keep the RTL names (and n_xbar), because page.js reads them."""
import json
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V3))
from dse.space import CONFIGS, SHAPES  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402

HEAD_KEYS = ("I", "S", "RB", "B", "V", "ROWS_X", "LOADX", "LN", "NG_QK", "NG_PV", "N_OC", "N_EXP", "BKV")
KEEP = {"qk": ("NB", "W_K", "NG", "T_W", "LW", "XS", "NS", "KS", "A_GRP", "B_GRP", "START_LAT", "TREE_LEVELS"),
        "pv": ("NB", "W_K", "NG", "T_W", "LW", "XS", "NS", "KS", "A_GRP", "B_GRP", "START_LAT", "TREE_LEVELS", "EW"),
        "sm": ("NWB", "NB", "T_W", "DF", "STREAM"),
        "acc": ("P_OC", "RPW", "W", "FD")}


def memBits(module):
    return sum(words * bits * n for (words, bits), n in module.resources.memories.items())


rows = []
for (name, n, dm, d) in SHAPES:
    for R, C in CONFIGS:
        for pp in (1, 2, 4):
            for ne in (d, 2 * d):
                for nep in (2 * d, d):
                    rec = dict(shape=name, n=n, dm=dm, d=d, R=R, C=C, P_PROJ=pp, N_E=ne, N_EP=nep)
                    try:
                        if nep == 2 * d:
                            hp = headParams(n, dm, d, R, C, P_PROJ=pp, N_E=ne)
                        else:
                            ref = headParams(n, dm, d, R, C, P_PROJ=pp, N_E=ne)
                            hp = headParams(n, dm, d, R, C, P_PROJ=pp, N_E=ne, N_EP=nep,
                                            N_OC=min(ref.N_OC, nep // ref.BKV))
                        g, parts = attentionHead(hp, withValues=False)
                        rec.update({k: getattr(hp, k) for k in HEAD_KEYS})
                        rec["n_xbar"] = hp.nXbar
                        for k, keys in KEEP.items():
                            rec[k] = {x: getattr(parts[k].d, x) for x in keys}
                            rec[k]["mem_bits"] = memBits(parts[k])
                            rec[k]["xbar"] = parts[k].resources.nCrossbars
                        rec["gemm_xbar"] = sum(m.resources.nCrossbars for m in parts["gemm"].values())
                        rec["cycles"] = run(g).cycles
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"[:160]
                    rows.append(rec)
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, separators=(",", ":")))
ok = sum(1 for r in rows if "cycles" in r)
print(f"{len(rows)} builds, {ok} inside the domain, {len(rows) - ok} outside -> {out} ({out.stat().st_size} bytes)")
