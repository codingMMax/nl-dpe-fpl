#!/usr/bin/env python3
"""freezeReference.py — freeze the simulator's outputs as the reference of checkEquivalence.py.

The first reference (work/equivalence/ref/, git 816807b) was frozen from the old simulator to prove the refactor equal;
it is kept. A rule change makes the model differ from it by design: the drain of 2026-10-18 (every pass drains all C
columns; the RTL still stops after COLS_USED). This script freezes the current model into a new folder in the same
format, so the gate goes on catching unintended changes:

  reports/fc_<R>x<C>_<M>_<K>_<N>.json.gz    the 48 FC builds of the DSE, timing-only (checkEquivalence.newReportDict)
  reports/att_<shape>_<R>x<C>.json.gz      the 72 attention builds (+ head_params, ctrl_t_s, ctrl_done)
  rows/fc_rows.csv, rows/att_rows.csv      the DSE rows (simulator/platform/rows.py; the FC rows with v2's VTR table,
                                           the attention rows without VTR)
  corpus/<name>/                           the corpora of checkEquivalence.GENERATORS
  manifest.json                            commit (+dirty), python, numpy, the sha256 of every file

Usage: python3 v3/simulator/tests/freezeReference.py OUT_DIR [--jobs 8]     (OUT_DIR must not exist)
       then point checkEquivalence.py --ref at it (its default names the current reference)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform as pyplatform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[2]
REPO = V3.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "simulator" / "tests"))
import checkEquivalence as ce  # noqa: E402
from dse.space import CONFIGS, FC_PASS_EXTENT, FC_SHAPES, SHAPES  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402
from simulator.models.fc import fc  # noqa: E402
from simulator.platform import area  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402
from simulator.platform.rows import attentionRow, fcRow  # noqa: E402


def dump(path: Path, obj: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def tag(shape: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", shape)


def writeRows(path: Path, rows: list) -> None:
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; freeze into a new folder")
    t0 = time.time()
    for d in ("reports", "rows", "corpus"):
        (out / d).mkdir(parents=True)
    logDir = out.parent / "logs"
    logDir.mkdir(exist_ok=True)
    # reports
    n = 0
    for R, C in CONFIGS:
        for M, K, N in FC_SHAPES:
            g = fc(K, N, R, C, M=M, platform=Platform("nl_dpe", rows=R, cols=C), passExtent=FC_PASS_EXTENT,
                   withValues=False)
            dump(out / "reports" / f"fc_{R}x{C}_{M}_{K}_{N}.json.gz",
                 ce.newReportDict(run(g), dict(pass_extent=FC_PASS_EXTENT)))
            n += 1
    for name, nseq, dm, d in SHAPES:
        for R, C in CONFIGS:
            hp = headParams(nseq, dm, d, R, C)
            g, parts = attentionHead(hp, withValues=False)
            rep = run(g)
            ctrl = parts["ctrl"]
            dump(out / "reports" / f"att_{tag(name)}_{R}x{C}.json.gz",
                 ce.newReportDict(rep, dict(head_params=dict(NSEQ=nseq, DM=dm, D=d, R=R, C=C), ctrl_t_s=ctrl.tStart,
                                            ctrl_done=ctrl.done)))
            n += 1
    # rows
    vtr = area.vtrTable(ce.REF / "vtr_results.csv")
    writeRows(out / "rows" / "fc_rows.csv",
              [fcRow(R, C, shape, Platform("nl_dpe", rows=R, cols=C), vtr, passExtent=FC_PASS_EXTENT)
               for R, C in CONFIGS for shape in FC_SHAPES])
    writeRows(out / "rows" / "att_rows.csv",
              [attentionRow(name, nseq, dm, d, R, C) for name, nseq, dm, d in SHAPES for R, C in CONFIGS])
    # corpora, by the same generators the gate runs
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(lambda name: ce.regenerate(name, out, logDir), list(ce.GENERATORS)))
    failed = [name for name, rc in results if rc]
    if failed:
        raise SystemExit(f"generators failed: {failed} (logs in {logDir})")
    # manifest
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(out))] = hashlib.sha256(p.read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "v3"], cwd=REPO, capture_output=True,
                           text=True).stdout.strip() != ""
    manifest = dict(commit=commit + ("+dirty" if dirty else ""), python=pyplatform.python_version(),
                    numpy=np.__version__, reports=n, seconds=round(time.time() - t0, 1),
                    generated=time.strftime("%Y-%m-%d %H:%M"), files=files, n_files=len(files))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"freezeReference: {n} reports, 2 row tables, {len(ce.GENERATORS)} corpora, {len(files)} files -> {out} "
          f"({manifest['seconds']} s, git {manifest['commit'][:12]}{'+dirty' if dirty else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
