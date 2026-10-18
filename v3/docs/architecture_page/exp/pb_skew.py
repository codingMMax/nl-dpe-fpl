"""Model-only: P_PROJ = 2 with an X source that starves copy 1 (odd tokens arrive at cycle 2000, even tokens at 0).
The PV DIMM writes V rows in row order, so copy 0's rows wait in its two row registers until row 1 is written."""
import sys
from pathlib import Path
V3 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V3))
import numpy as np
import top.model as cm
from sim.core.engine import run
from top.cases import build, random_case

rng = np.random.default_rng(1)
c = random_case("skew", dict(NSEQ=64, DM=64, D=16, R=64, C=64, P_PROJ=2), rng,
                x_avail=[0 if t % 2 == 0 else 2000 for t in range(64)])
for cap in (2, 4):
    cm.PB_CAP = cap
    try:
        rep = run(build(c, values=False)[0])
        print(f"PB_CAP {cap}: OK, {rep.cycles} cycles")
    except Exception as e:
        print(f"PB_CAP {cap}: {type(e).__name__}: {e}")
