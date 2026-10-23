#!/usr/bin/env python3
"""stages_brief.py — the one-page brief "Why an attention head wants the small crossbar", for discussion with readers
who do not need the model (the stage page): the head as stages with their work and their crossbar counts, what one
crossbar delivers and costs, the cost of every stage on the 12 arrays (the simulator's critical path, energy and area:
results/stages/stages_costs.png), why the mapping knobs cannot reverse the ranking (results/mapping_optimum.json), and
the contrast with FC (results/fc_priced_results.csv). Every number is read from those files.
Reads dse/results/stages/stages.json; writes dse/results/stages/brief.html (published with stages_costs.png).
Usage: python3 v3/dse/stages_brief.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse.results_page import CSS, SCRIPT  # noqa: E402
from dse.space import CONFIGS  # noqa: E402
from dse.dataflow import CSS as FLOW_CSS, section as flow_section  # noqa: E402
from dse.stages_page import cx, energy, figure, tag  # noqa: E402

STAGES_DIR = V3 / "dse" / "results" / "stages"
RESULTS = V3 / "dse" / "results"
SMALL = f"{min(R for R, _ in CONFIGS)}x{min(C for _, C in CONFIGS)}"
LARGE = f"{max(R for R, _ in CONFIGS)}x{max(C for _, C in CONFIGS)}"
FC_CASE = ("2048", "256")                      # the FC GEMM of the contrast: K = 2048, N = 256
BUF, P, K_READER = 40, 8, 5


def tau(I: int) -> int:
    return -(-8 * I // BUF) + P


def rate(I: int) -> float:
    return I / tau(I)


def fmt(x: float) -> str:
    return f"{x:,.0f}"


def change(a: float, b: float) -> str:
    """b against a: a factor when it is at least 1.95x, else a percentage."""
    r = b / a
    return f"{r:.1f}×" if r >= 1.95 else f"{100 * (r - 1):+.0f} %"


def td(x: str, cls: str = "") -> str:
    return f'<td class="{cls}">{x}</td>' if cls else f"<td>{x}</td>"


def stage_table(res: dict) -> str:
    heads = {c["name"]: c for c in res["cases"]}
    proj = "; ".join(f"{h['name']} {h['configs'][SMALL]['stages']['gemm']['crossbars']} at R = {SMALL.split('x')[0]}, "
                     f"{h['configs'][LARGE]['stages']['gemm']['crossbars']} at R = {LARGE.split('x')[0]}" for h in heads.values())
    d = {h["name"]: h["d"] for h in heads.values()}
    twoD = " / ".join(f"{2 * v}" for v in d.values())
    noc = " / ".join(f"{h['configs'][SMALL]['stages']['acc']['crossbars']}" for h in heads.values())
    rows = [
        ("Projections Q, K, V", "3 · n · d<sub>model</sub> · d multiply-adds",
         "A weight-stationary matrix–vector product. A token's d<sub>model</sub> inputs stream in through the port; d outputs come out. "
         "A projection needs ⌈d<sub>model</sub> / R⌉ tiles.",
         f"3 · ⌈d<sub>model</sub> / R⌉ ({proj})", "Capacity: enough tiles to hold the three matrices."),
        ("Tiled QK", "n² · d products",
         "Identity-weight passes that convert I values each. The products and the sums are soft-logic adds in the log domain.",
         f"N<sub>E</sub> = 2d ({twoD})", "Bandwidth: keep up with the score stream, 5 scores per cycle with d conversions each."),
        ("Online softmax", "n² scores: running max, exp, sum",
         "A reader takes 5 scores per cycle from one port. EXP passes convert I values each.",
         "N<sub>EXP</sub> + 2 = 4", "Bandwidth: keep up with the reader."),
        ("Tiled PV", "n² · d products", "As tiled QK.", f"N<sub>EP</sub> = 2d ({twoD})", "Bandwidth: as tiled QK."),
        ("Output rescale", "n · d outputs, rescaled once per key block", "Rescale and LOG passes.",
         f"2 · N<sub>OC</sub> ({noc})", "Bandwidth: keep up with the block period."),
    ]
    body = "\n".join("<tr>" + "".join(td(x) for x in r) + "</tr>" for r in rows)
    return ('<div class="tablewrap"><table class="ranks"><thead><tr><th>stage</th><th>the work, fixed by the head</th>'
            '<th>what the crossbars do</th><th>crossbars in our mapping</th><th>what sets the count</th></tr></thead>'
            f"<tbody>\n{body}\n</tbody></table></div>")


def crossbar_table() -> str:
    """The pass period of one crossbar, I by I: the load through the port, the bit-slices, the period, the rate."""
    Is = sorted({min(R, C) for R, C in CONFIGS} | {max(R for R, _ in CONFIGS)})
    rows = "\n".join(f"<tr><td>{I}</td><td>⌈8 × {I} / {BUF}⌉ = {-(-8 * I // BUF)}</td><td>{P}</td><td>{tau(I)}</td>"
                     f"<td>{I}</td><td>{I} ÷ {tau(I)} = {rate(I):.1f}</td><td>{tau(I) / I:.3f}</td></tr>" for I in Is)
    return ('<div class="tablewrap"><table class="ranks"><thead><tr><th>I = min(R, C)</th><th>load: ⌈8I / 40⌉ cycles</th>'
            '<th>compute: the bit-slices</th><th>pass period τ(I)</th><th>values per pass</th><th>values per cycle</th>'
            f'<th>cycles per value = 1/5 + 8/I</th></tr></thead><tbody>\n{rows}\n</tbody></table></div>')


def cost_table(case: dict, stages: tuple, names: dict) -> str:
    s, l = case["configs"][SMALL], case["configs"][LARGE]
    Is, Il = min(map(int, SMALL.split("x"))), min(map(int, LARGE.split("x")))
    Cs, Cl = int(SMALL.split("x")[1]), int(LARGE.split("x")[1])
    xs, xl = s["stages"]["gemm"]["crossbars"], l["stages"]["gemm"]["crossbars"]
    gain = {
        "gemm": (f"Fewer tiles: {xs} crossbars become {xl}. Fewer tiles means fewer ports, so the token stream is slower, not faster."
                 if xl < xs else
                 f"None. The same {xs} tiles: no row beyond d<sub>model</sub> = {case['dm']} is ever loaded. The stream is slower "
                 f"only because every pass drains {Cl} columns instead of {Cs}."),
        "qk": (f"Passes {fmt(s['stages']['qk']['passes'])} become {fmt(l['stages']['qk']['passes'])}, but each takes "
               f"{tau(Il)} cycles instead of {tau(Is)}: {rate(Il) / rate(Is):.1f}× per crossbar, and the reader paces the stage anyway."),
        "sm": "None. The reader is soft logic and takes 5 scores per cycle on any array.",
        "pv": "As tiled QK.",
        "acc": (f"Fewer runs: {s['runs']} become {l['runs']}." if l["runs"] < s["runs"] else
                "None. One run at both arrays." if s["runs"] == 1 else f"None. {s['runs']} runs at both arrays."),
    }
    rows = []
    for st in stages:
        a, b = s["stages"][st], l["stages"][st]
        rows.append("<tr>" + td(names[st]) + td(f"{fmt(a['path_cycles'])} → {fmt(b['path_cycles'])}")
                    + td(f"{energy(a['energy_pj'])} → {energy(b['energy_pj'])}")
                    + td(f"{s['A_model'][st]:.2f} → {l['A_model'][st]:.2f}") + td(gain[st]) + "</tr>")
    fixedA = lambda c: c["A_model"]["clb"] + c["A_model"]["memory"]                     # noqa: E731
    rows.append("<tr>" + td("Soft logic and memory") + td("—") + td("—")
                + td(f"{fixedA(s):.2f} → {fixedA(l):.2f}") + td("None. The soft logic around each crossbar is sized by its window I.") + "</tr>")
    rows.append('<tr class="top">' + td("Whole head") + td(f"{fmt(s['T'])} → {fmt(l['T'])} ({change(s['T'], l['T'])})")
                + td(f"{energy(s['E'])} → {energy(l['E'])} ({change(s['E'], l['E'])})")
                + td(f"{s['A']:.1f} → {l['A']:.1f} ({change(s['A'], l['A'])})") + td("") + "</tr>")
    return ('<div class="tablewrap"><table class="ranks"><thead><tr><th>stage</th><th>cycles on the critical path</th>'
            '<th>energy</th><th>area (mm²)</th><th>what the larger array gains</th></tr></thead><tbody>\n'
            + "\n".join(rows) + "\n</tbody></table></div>")


NOTATION = ("Notation: x<sub>t</sub> a token and q<sub>t</sub>, k<sub>t</sub>, v<sub>t</sub> its projections; S<sub>b</sub> the scores of the run's S queries against key block b; m and l the running max and sum of each query row; P<sub>b</sub> = e<sup>S<sub>b</sub> − m</sup>; α = e<sup>m<sub>old</sub> − m<sub>new</sub></sup> the rescale when the max moves; O<sub>b</sub> = P<sub>b</sub> V<sub>b</sub> the block's partial output; o the accumulated output; O = o / l the head's output. The row conversions are the LOG passes that take the projections into the log domain, where the products are adds.")


def diagram_facts(case: dict, cfg: str) -> dict:
    d, c = case["diagrams"][cfg], case["configs"][cfg]
    st = d["steps"]
    proj = st["gq0.rows"]
    return dict(Tp=int(round(c["T_model"]["T_projection_token"])), end=max(b for _, b, _ in proj), job0=st["qk.jobs"][0][0], T=d["T"],
                S=c["S"], runs=c["runs"], blocks=c["blocks"], tiles=c["stages"]["gemm"]["crossbars"], ends=d["run_ends"],
                readout=st["sm.run"][-1], flush=st["acc.run"][-1], conv=st["qk.b_in"])


def pair_notes(case: dict, cfgs: list) -> str:
    """The bullets under one head's pair of diagrams: how the head flows at the small array, then what the large
    array changes; every number from the simulator's steps."""
    n, (small, large) = case["n"], cfgs
    fa, fb = diagram_facts(case, small), diagram_facts(case, large)
    bkv = n // fa["blocks"]
    li = []
    if fa["runs"] == 1:
        li += [f"<li><b>The stream.</b> The projections push all {n} tokens through their ports, one token every {fa['Tp']} cycles. "
               f"The other stages work on blocks of {bkv} keys. The light bars are the LOG conversion of each token's rows into the "
               f"log domain as it arrives, one short pass per row; they never fall behind.</li>",
               f"<li><b>The wait.</b> A score block needs all S queries of its run. Here S = n = {n}, one run, so the first score "
               f"block needs the last token and starts at cycle {fa['job0']:,}, when the stream ends at {fa['end']:,}. The keys were "
               f"ready long before.</li>",
               f"<li><b>The chain.</b> Each block then passes through scores, softmax, PV and rescale in turn, each stage starting "
               f"when the one before delivers. With {fa['blocks']} blocks the pipeline never fills: the {fa['T'] - fa['end']:,} cycles "
               f"after the stream are one pipeline latency plus the run's tails, the log l read-out and the normalization. Doubling "
               f"any stage's crossbars would not shorten them.</li>",
               f"<li><b>Overlaps.</b> Where a block starts before the previous one ends, the later block is drawn lighter: the "
               f"softmax reads the next tile while it finishes the previous block, and the rescale runs two blocks on two crossbar groups.</li>"]
    else:
        later = (fa["ends"][1] - fa["ends"][0]) / fa["blocks"]
        li += [f"<li><b>The first run overlaps the stream.</b> S = {fa['S']} &lt; n = {n}, so the first run needs only the first "
               f"{fa['S']} tokens and starts at cycle {fa['job0']:,}, while the stream runs to {fa['end']:,}. Its blocks come at the "
               f"stream's pace, a {bkv}-key block every {bkv} × {fa['Tp']} = {bkv * fa['Tp']} cycles, so every stage works in bursts "
               f"and waits for keys.</li>",
               f"<li><b>The later runs come at the core's pace.</b> After the stream, a block every {later:.0f} cycles: the hand-over "
               f"between the softmax and PV holds two blocks, so block k + 2 cannot start before PV has consumed block k. Blocks come "
               f"in pairs, and every stage idles between pairs.</li>",
               f"<li><b>Every run ends with its tails</b>, the log l read-out and the normalization: the small dark boxes.</li>"]
    reason = ("fewer tiles hold the matrices, so fewer ports feed the stream" if fb["tiles"] < fa["tiles"] else
              "the same tiles, but every pass now drains all the wider array's columns")
    runsNote = (f" With S = {fb['S']} instead of {fa['S']} there {'are' if fb['runs'] > 1 else 'is'} {fb['runs']} "
                f"run{'s' if fb['runs'] > 1 else ''} instead of {fa['runs']}, and the time after "
                f"the stream is {fb['T'] - fb['end']:,} cycles against {fa['T'] - fa['end']:,}." if fb["runs"] != fa["runs"] else
                f" The time after the stream is {fb['T'] - fb['end']:,} cycles against {fa['T'] - fa['end']:,}: the core's passes are "
                f"longer on the wider array, but the core does not get faster.")
    li.append(f"<li><b>From {cx(small)} to {cx(large)}.</b> A token takes {fb['Tp']} cycles instead of {fa['Tp']}: {reason}. The "
              f"stream ends at {fb['end']:,} instead of {fa['end']:,}, and the first score block starts at {fb['job0']:,} instead of "
              f"{fa['job0']:,}. The head takes {fb['T']:,} cycles instead of {fa['T']:,}, {change(fa['T'], fb['T'])}.{runsNote}</li>")
    return '  <ul class="notes">\n    ' + "\n    ".join(li) + "\n  </ul>\n"


def time_section(res: dict) -> str:
    """Section 2: each head at the two DIAGRAMS arrays, stacked on a shared time axis, one box per block per stage."""
    cfgs = res["diagrams"]
    pair = " and ".join(cx(c) for c in cfgs)
    out = []
    for i, case in enumerate(res["cases"]):
        title = f"The time diagrams: {case['name']}"
        caption = (f"{case['name']} at {pair}, from the simulator, on one time axis. A run takes S queries over all {case['n']} keys in "
                   f"blocks of 16, one block after another. One box per block per stage; the light bars are the per-token row "
                   f"conversions." + (f" {NOTATION}" if i == 0 else ""))
        out.append(figure(f"time{i}", title, f"stages_simple_{tag(case['name'])}.png",
                          f"{case['name']} at {pair}: one box per block per stage, the stream, the wait for the run's queries, the blocks through the four stages and the tails, the two arrays on one time axis",
                          caption, pair_notes(case, cfgs)))
    return "".join(out)


def mapping_paragraph(res: dict) -> str:
    opt = json.loads((RESULTS / "mapping_optimum.json").read_text())
    names = [c["name"] for c in res["cases"]]
    ratios = {name: [b["best"]["adp"]["adp"] / b["fixed"]["adp"] for b in opt["builds"] if b["shape"] == name] for name in names}
    allR = [b["best"]["adp"]["adp"] / b["fixed"]["adp"] for b in opt["builds"]]
    fixed = sorted(opt["rankings"]["fixed"], key=lambda c: opt["rankings"]["fixed"][c]["rank_adp"])
    best = sorted(opt["rankings"]["adp"], key=lambda c: opt["rankings"]["adp"][c]["rank_adp"])
    moved = [i + 1 for i, (a, b) in enumerate(zip(fixed, best)) if a != b]
    swap = ("the ranking does not move" if not moved else
            f"only positions {' and '.join(map(str, moved))} swap ({' and '.join(cx(fixed[i - 1]) for i in moved)})")
    perHead = "; ".join(f"{name} {min(r):.2f} to {max(r):.2f}" for name, r in ratios.items())
    ran = set(opt["rankings"]["fixed"])
    stale = ("" if ran == {f"{R}x{C}" for R, C in CONFIGS} else
             f" (This check ran on {len(ran)} configs with the crossbar tiles of that day; it is not yet re-run on the "
             f"{len(CONFIGS)} configs and the row-parallel tiles of 2026-10-22.)")
    return (f"We checked this with the per-build optimum over the knobs (<code>mapping_sensitivity.py --optimize</code>). "
            f"Giving every build its own best P<sub>proj</sub>, N<sub>E</sub>, N<sub>EP</sub>, N<sub>EXP</sub> and B<sub>kv</sub> "
            f"cuts its ADP to {min(allR):.2f} to {max(allR):.2f} of the fixed mapping ({perHead}), and across the six heads {swap}. "
            f"{cx(best[0])} stays first." + stale)


def choices_paragraph(res: dict, small: str) -> str:
    """The two design choices that are not knobs. Their cycle cost is the 2026-10-19 estimate by the equations (worked on
    the 12 arrays of that day); the claim that removing it leaves cycles x area above twice the small array's is checked
    here on the grid's 256-column arrays."""
    cost = {"ViT-B/16": 0.10, "JetFormer-150P": 0.40}
    worst = {c["name"]: min(c["grid"]["256"]["xa"]) * (1 - cost[c["name"]]) for c in res["cases"]}
    assert all(v > 2 for v in worst.values()), worst
    return ("Two choices in the design are not knobs: a run holds S = min(n, I) queries, and every pass drains all C "
            "columns. Both cost the large arrays something: by the equations (worked on the 12 arrays of 2026-10-19), about "
            "10 % of the cycles at ViT-B/16 and up to 40 % at JetFormer-150P on the 256-column arrays. Removing both would "
            f"still leave the product of cycles and area at more than twice that of {small}.")


def fc_paragraph() -> str:
    rows = {r["config"]: r for r in csv.DictReader((RESULTS / "fc_priced_results.csv").open())
            if (r["K"], r["N"]) == FC_CASE}
    s, l = rows[SMALL], rows[LARGE]
    K, N = FC_CASE
    return (f"An FC layer's weight matrix is large: {K} × {N} is {int(K) * int(N):,} cells, which is {s['dpe_count']} tiles of "
            f"{cx(SMALL)} or {l['dpe_count']} tiles of {cx(LARGE)}. Its crossbar count is set by capacity, so a bigger array means "
            f"fewer crossbars and less soft logic around them. The area falls from {float(s['area_total_mm2']):.2f} to "
            f"{float(l['area_total_mm2']):.2f} mm² while the cycles rise from {s['T_cycles']} to {l['T_cycles']}. That is a real "
            f"trade-off, and the FC front spans several arrays. Attention's matrices are small (d<sub>model</sub> × d, and identity "
            f"blocks of d × d), and its counts are set by bandwidth, so a bigger array only adds cells.")


def main() -> int:
    res = json.loads((STAGES_DIR / "stages.json").read_text())
    stages, names = tuple(res["stages"]), res["stage_name"]
    cases = res["cases"]
    small, large = cx(SMALL), cx(LARGE)
    heads = ", ".join(f"{c['name']} (n {c['n']}, d<sub>model</sub> {c['dm']}, d {c['d']})" for c in cases)
    Is, Il = min(map(int, SMALL.split("x"))), min(map(int, LARGE.split("x")))
    cells = lambda cfg: int(cfg.split("x")[0]) * int(cfg.split("x")[1])                 # noqa: E731
    costTables = "\n".join(f'<div class="family"><h2>{c["name"]}: {small} → {large}</h2>\n{cost_table(c, stages, names)}</div>'
                           for c in cases)
    # the facts the lede and the bullets quote
    faster = [(c["name"], cfg) for c in cases for cfg in c["configs"] if c["configs"][cfg]["T"] < c["configs"][SMALL]["T"]]
    exceptions = "; ".join(f"only {name}'s {cx(cfg)} has fewer cycles, by {100 * (1 - c['configs'][cfg]['T'] / c['configs'][SMALL]['T']):.0f} %, "
                           f"at {100 * (c['configs'][cfg]['A'] / c['configs'][SMALL]['A'] - 1):.0f} % more area"
                           for name, cfg in faster for c in cases if c["name"] == name)
    ledeClaim = (f"the least area, the least energy and the lowest cycles × area of any array ({exceptions})" if faster
                 else "fewer cycles, less energy and less area than any larger array")
    def by(c, stage, key): return c["configs"][LARGE]["stages"][stage][key] - c["configs"][SMALL]["stages"][stage][key]  # noqa: E704
    extra = "; ".join(f"{c['name']} {fmt(by(c, 'gemm', 'path_cycles'))} of {fmt(c['configs'][LARGE]['T'] - c['configs'][SMALL]['T'])}"
                      for c in cases)
    farmE = lambda c, cfg: sum(c["configs"][cfg]["stages"][st]["energy_pj"] for st in ("qk", "pv"))        # noqa: E731
    farms = "; ".join(f"{c['name']} {change(farmE(c, SMALL), farmE(c, LARGE))}, "
                      f"{100 * farmE(c, LARGE) / c['configs'][LARGE]['E']:.0f} % of the head's energy" for c in cases)
    clbShare = "; ".join(f"{c['name']} {100 * (c['configs'][LARGE]['A_model']['clb'] + c['configs'][LARGE]['A_model']['memory']) / c['configs'][LARGE]['A']:.0f} %"
                         for c in cases)
    bandwidthPrice = (cells(LARGE) / rate(Il)) / (cells(SMALL) / rate(Is))
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="claude:viewport-fit-sides" content="cover">
<title>Attention wants small crossbars</title>
<meta name="description" content="Why one attention head on NL-DPE costs least on the smallest crossbar: the cost of every stage against the array, and why the mapping cannot reverse it.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}
.family table.ranks td:first-child {{ text-align: left; }}
table.ranks td:nth-child(n+2) {{ white-space: normal; text-align: left; min-width: 11ch; }}
table.ranks th:nth-child(n+2) {{ text-align: left; }}
.rule {{ font-size: 17px; border-left: 3px solid var(--accent); padding: 6px 14px; }}
{FLOW_CSS}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">v3 DSE · attention on NL-DPE · for discussion · 2026-10-19</p>
  <h1>Why an attention head wants the small crossbar</h1>
  <p class="lede">One head of attention costs least on the smallest array of the DSE, {small}: {ledeClaim}. This page shows the cost of each stage, how it changes with the array, and why no mapping choice reverses it. The numbers are the cycle-accurate simulator's and VTR's, for the smallest and the largest head of the DSE: {heads}.</p>
</header>

<section>
  <h2>1. The head in five stages</h2>
  <p>Every stage works in passes of one crossbar. The work of a stage is fixed by the head; the mapping decides how many crossbars share it.</p>
{stage_table(res)}
  <p class="rule">The projections' crossbar count is set by capacity: a bigger tile holds more of the matrix, so fewer tiles are needed. Every other stage's count is set by bandwidth: the number of ports needed to keep pace with the score stream. A bigger crossbar has the same 40-bit port, so those counts do not change with the array.</p>
</section>

{flow_section(res)}
{time_section(res)}
<section>
  <h2>3. What one crossbar delivers, and what it costs</h2>
  <p>A pass loads I = min(R, C) rows of 8-bit values through the 40-bit port, 5 bytes per cycle, then runs the 8 bit-slices, one bit plane of the input per cycle. The C output columns drain at ⌈8C / 40⌉ + 1 cycles; the drain overlaps the next load and binds only when C is wider than the rows loaded, which happens to the projections of a small head, not to these passes. A pass converts I values, one per row loaded, because the stored weights are identity blocks. So the pass period is τ(I) = ⌈8I / 40⌉ + 8, and the time per value is 1/5 + 8/I cycles: the 1/5 is the port and never changes, the 8/I is the pipeline spread over the rows of one pass.</p>
{crossbar_table()}
  <p>The two constants are the hard block's, not the mapping's: a 40-bit port and 8 bit-slices in every array. The simulator's pass period is gated against the RTL on the 64 × 64 cases. A pass fires all R × C cells whatever it uses, and a crossbar's area is its R × C tile plus soft logic sized by its window I.</p>
  <p class="rule">From {small} to {large} a crossbar fires {cells(LARGE) // cells(SMALL)}× the cells and converts {rate(Il) / rate(Is):.1f}× the values per cycle. Bandwidth is bought per crossbar, and a crossbar costs R × C cells. So a unit of bandwidth costs about {bandwidthPrice:.0f}× more cells on the large array.</p>
</section>

{figure("costs", "4. The cost of every stage, on every array", "stages_costs.png",
        "Stacked bars of the cycles on the critical path, the energy and the area by stage for the 12 arrays and both heads",
        "The simulator's critical path assigns every cycle of the head's time to the stage that was waited on, so the five parts add up to the total. Energy and area are per stage as well; the soft logic and the memory blocks are shown separately in the area panel. Arrays are grouped by C, with R rising inside a group.")}

<section>
  <h2>5. Reading the bars: {small} against {large}</h2>
  <div class="families">
{costTables}
  </div>
  <ul class="notes">
    <li><b>Cycles.</b> The projections account for most of the extra cycles ({extra}). A big tile holds more of the matrix, so there are fewer tiles and fewer ports for the token stream; where the tiles do not change, the wider drain does it. The online softmax reader paces everything else at 5 scores per cycle on any array, so tiled QK and tiled PV do not get faster even though each of their crossbars does.</li>
    <li><b>Energy.</b> Tiled QK and tiled PV fire {cells(LARGE) // cells(SMALL)}× the cells per pass for {rate(Il) / rate(Is):.1f}× the values per cycle. Their energy rises ({farms}). The projections are a few percent of the energy at any array.</li>
    <li><b>Area.</b> The bandwidth-set stages keep their crossbars and the tiles grow {cells(LARGE) // cells(SMALL)}×. The soft logic and the memory blocks are most of the area ({clbShare} at {large}), and the soft logic grows with the window I.</li>
    <li><b>So every stage's cost goes up. The only gain, {rate(Il) / rate(Is):.1f}× per crossbar in tiled QK and tiled PV, is one no stage can use.</b></li>
  </ul>
</section>

<section>
  <h2>6. Why sweeping the mapping does not change the answer</h2>
  <p>Every mapping knob adds crossbars to a stage: P<sub>proj</sub> copies the projection tiles, N<sub>E</sub>, N<sub>EP</sub>, N<sub>EXP</sub> and N<sub>OC</sub> widen the other stages. A crossbar brings one port and costs one R × C tile, so the cycles × area of a stage can only get worse with the array at any knob setting. The large array can buy back the projections' speed with copies, but it pays {cells(LARGE) // cells(SMALL)}× more cells per copy than the small one.</p>
  <p>{mapping_paragraph(res)}</p>
  <p>{choices_paragraph(res, small)}</p>
</section>

<section>
  <h2>7. Why the same hardware likes big arrays for FC</h2>
  <p>{fc_paragraph()}</p>
  <p class="rule">Attention needs ports, not cells. The small array has the most ports per cell, so it wins by structure, not by tuning.</p>
</section>

<div class="notes muted">
  <p>Data <code>v3/dse/results/stages/stages.json</code> (<code>dse/stage_analysis.py</code>), <code>mapping_optimum.json</code> (<code>dse/mapping_sensitivity.py --optimize</code>), <code>fc_priced_results.csv</code>; figure <code>dse/plot_stages.py</code>; page <code>dse/stages_brief.py</code>. The full model with every equation is the stage-analysis page; the write-up is <code>v3/docs/attention_stages.md</code>.</p>
</div>
</div>
<script>{SCRIPT}</script>
"""
    out = STAGES_DIR / "brief.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page)} bytes); publish with stages_costs.png beside it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
