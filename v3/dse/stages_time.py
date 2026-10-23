#!/usr/bin/env python3
"""stages_time.py — the page "Attention head in time" (for sharing): the log-domain mapping of one attention head on
NL-DPE crossbars, animated from the simulator's crossbar passes (dse/dataflow.py `map_section`): three bands (the
log domain, the crossbars, the linear domain), one column per operation, every edge labelled with the data it
carries, a block's packets moving along the edges at the real hand-over times, each operation lit while its passes
run, the two arrays 128x64 and 512x256 on one clock, and the time diagram with a cursor under each board. Nothing
else: the analysis is the brief (stages_brief.py). Reads dse/results/stages/stages.json; writes
dse/results/stages/time.html (self-contained, no files beside it).
Usage: python3 v3/dse/stages_time.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse.dataflow import CSS as FLOW_CSS, map_section  # noqa: E402
from dse.results_page import CSS, SCRIPT  # noqa: E402
from dse.stages_page import cx  # noqa: E402

STAGES_DIR = V3 / "dse" / "results" / "stages"


def main() -> int:
    res = json.loads((STAGES_DIR / "stages.json").read_text())
    cfgs = " and ".join(cx(c) for c in res["diagrams"])
    heads = " and ".join(f"{c['name']} (n {c['n']}, d {c['d']})" for c in res["cases"])
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="claude:viewport-fit-sides" content="cover">
<title>Attention head in time</title>
<meta name="description" content="The log-domain mapping of one attention head on NL-DPE crossbars, animated in time from the simulator's crossbar passes: adds of logs for the products, identity crossbars for EXP and LOG, adder trees for the sums.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}
{FLOW_CSS}
.notes li {{ margin: 0; }}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">NL-DPE attention · the log-domain mapping in time · 2026-10-20</p>
  <h1>One attention head, in the log domain, in time</h1>
  <p class="lede">Attention on NL-DPE crossbars without a single multiply in the core. The crossbars hold identity weights and only convert: EXP down into the linear domain, LOG up out of it. Every product is an add of two logs in soft logic, every sum is an adder tree in the linear domain, and the divide at the end is a subtract. The animation plays the simulator's own crossbar passes for {heads}, each at {cfgs}, on one clock.</p>
  <ul class="notes">
    <li><b>Reading the board.</b> Three bands: the log domain on top, the crossbars in the middle, the linear domain below. Each column is one operation; the label on an edge is the data that crosses it.</li>
    <li><b>What moves.</b> One key block at a time: a dot in the block's colour travels along the edges, and an operation lights while its passes run. The projections feed q, k, v on the left; O leaves on the right at the end of each run.</li>
    <li><b>Two places need no crossbar.</b> A score s is already a log, so log P = s − m is a subtract; and o is kept as a log between blocks, so the final O = o / l is a subtract followed by one EXP.</li>
    <li><b>The strip</b> under each board is the time diagram with a cursor: the projections, QKᵀ, softmax, PV and the rescale, block by block.</li>
  </ul>
</header>
{map_section(res)}
<div class="notes muted">
  <p>Notation: q, k, v the projected rows; S<sub>b</sub> the scores of the run's S queries against key block b; m and l the running max and sum of each query row; P<sub>b</sub> = e<sup>S<sub>b</sub> − m</sup>; α = e<sup>m<sub>old</sub> − m<sub>new</sub></sup>; O<sub>b</sub> = P<sub>b</sub> V<sub>b</sub>; o the accumulated output and lo its log; O = o / l. Data <code>v3/dse/results/stages/stages.json</code> (<code>dse/stage_analysis.py</code>), board <code>dse/dataflow.py</code>, page <code>dse/stages_time.py</code>.</p>
</div>
</div>
<script>{SCRIPT}</script>
"""
    out = STAGES_DIR / "time.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page)} bytes); self-contained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
