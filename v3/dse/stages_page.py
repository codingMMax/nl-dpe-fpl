#!/usr/bin/env python3
"""stages_page.py — the stage-analysis page (docs/attention_stages.md §6): for the two case-study heads, the
Pre-bit-plane analysis, frozen 2026-10-23: it models the previous design (the 5-byte port, BUF = 40) at the
N_E = 2d mapping and describes the 2026-10-16 run; not part of run_all.sh. Port it after stage 4.
pipelined time diagram, the per-stage tables, the analytic cycle and energy models with their errors against the
simulator, and the sweep over the array size. Reads dse/results/stages/stages.json and the PNGs of dse/plot_stages.py;
writes dse/results/stages/stages.html (published with the three PNGs beside it). The look is the results page's
(results_page.py's CSS and script); the equations render with KaTeX 0.16.11: its script from cdnjs, its stylesheet
inlined from dse/katex.min.css (the CDN's file with the woff2 sources only, relative), its 20 woff2 fonts fetched into
v3/work/katex/fonts and published beside the page as fonts/<name>.
Usage: python3 v3/dse/stages_page.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse.results_page import CSS, SCRIPT  # noqa: E402

STAGES_DIR = V3 / "dse" / "results" / "stages"
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11"
# the KaTeX stylesheet inlined (the artifact viewer admits no stylesheet from a CDN), its fonts published beside the
# page as fonts/*.woff2 (fetched from the CDN into v3/work/katex/fonts by fetchFonts; the viewer admits the artifact's own
# files only)
KATEX_CSS = (Path(__file__).resolve().parent / "katex.min.css").read_text()
FONTS_DIR = V3 / "work" / "katex" / "fonts"


def fetchFonts() -> list:
    """The woff2 fonts the stylesheet names, fetched once into v3/work/katex/fonts; returns their paths."""
    import re
    import urllib.request
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name in sorted(set(re.findall(r"fonts/(KaTeX_[A-Za-z0-9-]+\.woff2)", KATEX_CSS))):
        f = FONTS_DIR / name
        if not f.exists() or f.stat().st_size == 0:
            urllib.request.urlretrieve(f"{KATEX}/fonts/{name}", f)
        out.append(f)
    return out


def cx(c: str) -> str:
    return c.replace("x", " × ")


def png_size(name: str) -> tuple:
    return struct.unpack(">II", (STAGES_DIR / name).read_bytes()[16:24])


def figure(fid: str, title: str, img: str, alt: str, caption: str, extra: str = "") -> str:
    w, h = png_size(img)
    return (f'<section aria-labelledby="h_{fid}">\n  <div class="fighead"><h2 id="h_{fid}">{title}</h2>'
            f'<button class="zoom" id="z_{fid}" type="button" aria-controls="m_{fid}" aria-pressed="false">Zoom in'
            f'</button></div>\n  <figure><div class="mat" id="m_{fid}"><img src="{img}" alt="{alt}" width="{w}" '
            f'height="{h}"></div>\n  <figcaption class="muted">{caption}</figcaption></figure>\n{extra}</section>\n')


def tag(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def energy(pj: float) -> str:
    return f"{pj / 1e6:.2f} µJ" if pj >= 1e6 else f"{pj / 1e3:.1f} nJ"


def sweep_notes(res: dict) -> str:
    """The bullets under the sweep figure; every number is read from the grid (stage_analysis.py `grid`, whose
    findings the self-test asserts)."""
    cases = res["cases"]
    grids = [{int(C): g for C, g in case["grid"].items()} for case in cases]
    Cs = sorted(grids[0])
    C0, C1 = Cs[0], Cs[-1]
    Rs = sorted({R for C in Cs for R in grids[0][C]["R"]})        # each C's curve holds its own R (C <= R)
    R0, R1 = Rs[0], Rs[-1]
    base, top = f"{R0} × {C0}", f"{R1} × {C1}"
    at = lambda g, C, R, key: g[C][key][g[C]["R"].index(R)]       # noqa: E731  one point of a curve
    li = []
    for case, g in zip(cases, grids):
        name, dm = case["name"], case["dm"]
        if dm <= R0:
            atC = ", ".join(f"{g[C]['T'][-1]:,}{' cycles' if C == C0 else ''} at C = {C}" for C in Cs)
            li.append(f"<li><b>Cycles, {name}.</b> d<sub>model</sub> is {dm}, so a token never loads more than {dm} rows. "
                      f"Time does not change with R. At every R ≥ C it is {atC}. It grows with C, because every pass "
                      f"drains all C columns.</li>")
        else:
            fastest = {min((C for C in Cs if R in g[C]["R"]), key=lambda C: at(g, C, R, "T")) for R in Rs}
            left = {C: case["configs"][f"{R1}x{C}"]["T_model"] for C in Cs}
            after = {C: max(1, t["N_run"] - 1) * t["N_block"] * t["T_block"] for C, t in left.items()}
            runs = {C: t["N_run"] for C, t in left.items()}
            fast = (f"C = {next(iter(fastest))} is the fastest at every R." if len(fastest) == 1
                    else "The fastest C changes with R.")
            li.append(f"<li><b>Cycles, {name}.</b> d<sub>model</sub> is {dm}, so a token loads all R rows at every R of "
                      f"the grid. Time grows with R: {g[C0]['T'][0]:,} to {g[C0]['T'][-1]:,} cycles at C = {C0}. {fast} "
                      f"After the projection stream ends, the core still has to finish the later runs, or the whole run "
                      f"when there is only one. That leftover is "
                      + ", ".join(f"{after[C] / 1e3:.1f}k{' cycles' if C == C0 else ''} at C = {C} ({runs[C]} run"
                                  f"{'s' if runs[C] > 1 else ''}, S = {left[C]['S']})" for C in Cs)
                      + f". It is smallest at S = {left[min(Cs, key=lambda C: after[C])]['S']}.</li>")
    floor = ", ".join(f"{case['name']} {energy(case['e_clb'])} of {energy(case['configs'][f'{R0}x{C0}']['E'])}"
                      for case in cases)
    rise = ", ".join(f"{case['name']} {energy(g[C0]['E'][0])} to {energy(g[C0]['E'][-1])}" for case, g in zip(cases, grids))
    wide = [C for C in Cs if min(grids[0][C]["R"]) < C]          # a curve with an R < C point (none since 2026-10-22)
    wideNote = ""
    if wide:
        Cw = wide[0]
        cmp = " and ".join(f"{energy(g[Cw]['E'][0])} against {energy(g[Cw]['E'][1])} at {case['name']}"
                           for case, g in zip(cases, grids))
        wideNote = (f" With R &lt; C ({R0} × {Cw}) the passes double, and each still drains {Cw} columns: it costs more "
                    f"than {g[Cw]['R'][1]} × {Cw} at both heads ({cmp}).")
    li.append(f"<li><b>Energy.</b> The CLB adds, compares and memory are the same at every array: {floor} at {base}. "
              f"The rest is the passes. A pass fires all R × C cells and drains all C columns for I converted values. "
              f"So energy rises with R at every C once R ≥ C: {rise} at C = {C0}.{wideNote}</li>")
    areas = ", ".join(f"{case['name']} {g[C0]['A'][0]:.1f} to {g[C1]['A'][-1]:.1f} mm²" for case, g in zip(cases, grids))
    li.append(f"<li><b>Area.</b> The crossbar tiles scale with R × C, the CLB with I. Area rises with R and with C at both "
              f"heads: {areas} from {base} to {top}.</li>")
    tops = " and ".join(f"{g[C1]['xa'][-1]:.1f}× ({case['name']})" for case, g in zip(cases, grids))
    runnerUp = []
    for case, g in zip(cases, grids):
        for C in (C for C in Cs[1:] if R0 in g[C]["R"]):
            if at(g, C, R0, "T") < g[C0]["T"][0]:
                runnerUp.append(f"{case['name']}'s {R0} × {C} is {100 * (1 - at(g, C, R0, 'T') / g[C0]['T'][0]):.0f} % "
                                f"faster but {100 * (at(g, C, R0, 'A') / g[C0]['A'][0] - 1):.0f} % larger, so its product "
                                f"is {at(g, C, R0, 'xa'):.2f}×")
    li.append(f"<li><b>Cycles × area.</b> It rises with R and with C at every point. {base}, the smallest array of the "
              f"grid, is the best at both heads; {top} costs {tops}. " + ("; ".join(runnerUp) + ". " if runnerUp else "")
              + "The DSE's ADP adds Fmax on top; its attention front is 128 × 64 alone.</li>")
    li.append("<li><b>Why.</b> Attention uses an array as a port-fed converter. A projection pass loads min(d<sub>model</sub>, R) "
              "rows. A DIMM pass loads I = min(R, C) rows. Every pass fires all R × C cells and drains all C columns. Rows "
              "and columns beyond those cost area, energy and drain time, and save no time.</li>")
    return '  <ul class="notes">\n    ' + "\n    ".join(li) + "\n  </ul>\n"


NOTATION = ("Notation: x<sub>t</sub> a token and q<sub>t</sub>, k<sub>t</sub>, v<sub>t</sub> its projections; S<sub>b</sub> the scores of the run's S queries against key block b; m and l the running max and sum of each query row; P<sub>b</sub> = e<sup>S<sub>b</sub> − m</sup>; α = e<sup>m<sub>old</sub> − m<sub>new</sub></sup> the rescale when the max moves; O<sub>b</sub> = P<sub>b</sub> V<sub>b</sub> the block's partial output; o the accumulated output; O = o / l the head's output. The row conversions are the LOG passes that take the projections into the log domain, where the products are adds.")


def explain_notes(case: dict, cfg: str) -> str:
    """The bullets under the step-by-step diagram, every number from the simulator's steps."""
    g, c = case["gantt"][cfg], case["configs"][cfg]
    st, T, n, S, blocks = g["steps"], g["T"], case["n"], c["S"], c["blocks"]
    bkv = n // blocks
    proj = st["gq0.rows"]
    Tp, streamEnd = proj[1][0] - proj[0][0], max(b for _, b, _ in proj)
    keysReady = next(b for _, b, u in st["gk0.rows"] if u == bkv - 1)
    conv = st["qk.b_in"]
    job0, tile0, exp0, pv0, acc0 = (st[k][0] for k in ("qk.jobs", "sm.tile", "sm.blk", "pv.jobs", "acc.blk"))
    job1 = st["qk.jobs"][1] if len(st["qk.jobs"]) > 1 else None
    readout, flush = st["sm.run"][-1], st["acc.run"][-1]
    span = lambda s: f"{s[0]:,}–{s[1]:,}"                                                   # noqa: E731
    li = [
        f"<li><b>The stream.</b> The projections deliver one token every {Tp} cycles and end at cycle {streamEnd:,}. Tiled QK "
        f"converts each token's Q and K rows as it arrives, {conv[0][1] - conv[0][0]} cycles each, and is done "
        f"{conv[-1][1] - streamEnd} cycles after the last token. It never falls behind.</li>",
        f"<li><b>The wait.</b> A score job needs all S queries of its run. Here S = n = {n}, one run, so the first job needs the "
        f"last token's query. It starts at cycle {job0[0]:,}, {job0[0] - streamEnd} cycles after the stream ends. The keys of "
        f"block 0 were ready at cycle {keysReady} and waited.</li>",
        f"<li><b>The chain.</b> Then block 0 passes through four stages in turn, each starting when the one before delivers: "
        f"score job {span(job0)}, tile read {span(tile0)}, EXP {span(exp0)}, PV job {span(pv0)}, rescale {span(acc0)}. "
        + (f"Block 1 follows {job1[0] - job0[0]} cycles behind. " if job1 else "")
        + f"With {blocks} blocks this is one pass through the pipeline, not a steady state.</li>",
        f"<li><b>The tails.</b> The run ends with the log l read-out, the softmax sending each query's final normalizer to the "
        f"accumulator, one row per cycle ({readout[1] - readout[0]} cycles), and the normalization O = o / l, EXP passes over "
        f"the accumulator's groups that emit the head's output ({flush[1] - flush[0]} cycles).</li>",
        f"<li><b>Why the rescale blocks overlap and the PV jobs do not.</b> The accumulator has two crossbar groups, an EXP "
        f"group for the rescale and a LOG group for the o-carry, and its waits are per wave: block 1's rescale waves start "
        f"on the EXP group while block 0's LOG waves are still on the LOG group. A PV job runs all its waves on the one PV "
        f"farm in lock-step, so the next job starts only when the farm is free.</li>",
        f"<li><b>So the {T - streamEnd:,} cycles after the stream are one pipeline latency plus the tails.</b> More crossbars in "
        f"any stage would not shorten them: each stage is waiting for the one before, not working. A head with S &lt; n, such "
        f"as ViT-B/16 at 128 × 64 (S = 64), starts its first run after S tokens and overlaps the rest of the stream.</li>",
    ]
    return '  <ul class="notes">\n    ' + "\n    ".join(li) + "\n  </ul>\n"


def head_label(c: dict) -> str:
    return f"{c['name']} (n {c['n']}, d<sub>model</sub> {c['dm']}, d {c['d']})"


def stage_tables(res: dict) -> str:
    """Per head: energy and cycles by stage at the two extreme arrays, the layout of the chat tables."""
    out = []
    names = res["stage_name"]
    for case in res["cases"]:
        cfgs = res["extremes"]
        head = "".join(f"<th>{cx(c)}</th>" for c in cfgs)
        eRows, tRows, xRows = "", "", ""
        for s in res["stages"]:
            e = "".join(f"<td>{case['configs'][c]['stages'][s]['energy_pj'] / 1e3:,.0f} nJ "
                        f"({100 * case['configs'][c]['stages'][s]['energy_pj'] / case['configs'][c]['E']:.0f} %)</td>" for c in cfgs)
            t = "".join(f"<td>{case['configs'][c]['stages'][s]['path_cycles']:,} "
                        f"({100 * case['configs'][c]['stages'][s]['path_cycles'] / case['configs'][c]['T']:.0f} %)</td>" for c in cfgs)
            x = "".join(f"<td>{case['configs'][c]['stages'][s]['crossbars']} / {case['configs'][c]['stages'][s]['passes']:,}</td>" for c in cfgs)
            eRows += f"<tr><td>{names[s]}</td>{e}</tr>"
            tRows += f"<tr><td>{names[s]}</td>{t}</tr>"
            xRows += f"<tr><td>{names[s]}</td>{x}</tr>"
        eRows += "<tr class=\"top\"><td>head</td>" + "".join(f"<td>{case['configs'][c]['E'] / 1e3:,.0f} nJ</td>" for c in cfgs) + "</tr>"
        tRows += "<tr class=\"top\"><td>head</td>" + "".join(f"<td>{case['configs'][c]['T']:,}</td>" for c in cfgs) + "</tr>"
        xRows += "<tr class=\"top\"><td>head</td>" + "".join(f"<td>{case['configs'][c]['nXbar']} / {sum(case['configs'][c]['stages'][s]['passes'] for s in res['stages']):,}</td>" for c in cfgs) + "</tr>"
        out.append(f"""  <div class="family">
    <h2>{head_label(case)}</h2>
    <p class="muted">Crossbars / passes per stage</p>
    <div class="tablewrap"><table class="ranks"><thead><tr><th>stage</th>{head}</tr></thead><tbody>{xRows}</tbody></table></div>
    <p class="muted">Energy per stage (share of the head)</p>
    <div class="tablewrap"><table class="ranks"><thead><tr><th>stage</th>{head}</tr></thead><tbody>{eRows}</tbody></table></div>
    <p class="muted">Cycles on the critical path per stage (share of T)</p>
    <div class="tablewrap"><table class="ranks"><thead><tr><th>stage</th>{head}</tr></thead><tbody>{tRows}</tbody></table></div>
  </div>""")
    return '<div class="families">\n' + "\n".join(out) + "\n</div>"


def laws_tables(res: dict) -> str:
    """Per head the three laws' terms at the two extreme arrays: time (the stream, the last run's scores, the floor),
    energy (per stage, the fired cells per value), area (CLBs, crossbar tiles, memory)."""
    out = []
    names = res["stage_name"]
    for case in res["cases"]:
        cfgs = res["extremes"]
        head = "".join(f"<th>{cx(c)}</th>" for c in cfgs)
        L = {c: case["configs"][c]["laws"] for c in cfgs}
        T = {c: case["configs"][c]["T"] for c in cfgs}
        rows = [("S = min(n, I) queries per run", lambda c: f"{L[c]['S']}"),
                ("T<sub>p</sub>: cycles per token through a projection", lambda c: f"{L[c]['T_p']}"),
                ("the stream, n · T<sub>p</sub>", lambda c: f"{L[c]['stream']:,}"),
                ("the last run's scores, n · S / K (K = 5)", lambda c: f"{L[c]['last_run']:,}"),
                ("stream-bound: n · T<sub>p</sub> + n · S / K", lambda c: f"{L[c]['stream'] + L[c]['last_run']:,}"),
                ("core-bound: S · T<sub>p</sub> + n² / K", lambda c: f"{L[c]['S'] * L[c]['T_p'] + L[c]['core_floor']:,}"),
                ("the law: the larger of the two", lambda c: f"{max(L[c]['stream'] + L[c]['last_run'], L[c]['S'] * L[c]['T_p'] + L[c]['core_floor']):,}"),
                ("the simulator's T (the rest: the block pipeline's overheads, the run tails)", lambda c: f"<b>{T[c]:,}</b>")]
        tRows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{f(c)}</td>" for c in cfgs) + "</tr>" for k, f in rows)
        eRows = "".join(f"<tr><td>{names[s]}</td>" + "".join(f"<td>{L[c]['E_stage_pj'][s] / 1e3:,.0f}</td>" for c in cfgs) + "</tr>"
                        for s in res["stages"])
        eRows += "<tr><td>fixed (CLB adds, compares, memories)</td>" + "".join(f"<td>{L[c]['E_fixed_pj'] / 1e3:,.0f}</td>" for c in cfgs) + "</tr>"
        eRows += "<tr class=\"top\"><td>head (nJ)</td>" + "".join(f"<td>{case['configs'][c]['E_model']['E'] / 1e3:,.0f}</td>" for c in cfgs) + "</tr>"
        eRows += "<tr><td>price of a pass, p(R, C, I) (pJ)</td>" + "".join(f"<td>{L[c]['price_pj']:.1f}</td>" for c in cfgs) + "</tr>"
        eRows += "<tr><td>fired cells per converted value, RC / I</td>" + "".join(f"<td>{L[c]['fired_per_value']:,.0f}</td>" for c in cfgs) + "</tr>"
        aRows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{L[c]['area_mm2'][key]:.1f}</td>" for c in cfgs) + "</tr>"
                        for k, key in (("CLBs (VTR; grows with I)", "clb"), ("crossbar tiles (N<sub>xbar</sub> × tile)", "xbar"), ("memory blocks (VTR)", "mem")))
        aRows += "<tr class=\"top\"><td>head (mm²)</td>" + "".join(f"<td>{sum(L[c]['area_mm2'].values()):.1f}</td>" for c in cfgs) + "</tr>"
        aRows += "<tr><td>crossbars / cells per tile</td>" + "".join(f"<td>{L[c]['nXbar']} / {L[c]['tile_cells']}</td>" for c in cfgs) + "</tr>"
        out.append(f"""  <div class="family">
    <h2>{head_label(case)}</h2>
    <p class="muted">Latency (cycles)</p>
    <div class="tablewrap"><table class="ranks"><thead><tr><th>term</th>{head}</tr></thead><tbody>{tRows}</tbody></table></div>
    <p class="muted">Energy (nJ, the closed form)</p>
    <div class="tablewrap"><table class="ranks"><thead><tr><th>term</th>{head}</tr></thead><tbody>{eRows}</tbody></table></div>
    <p class="muted">Area (mm²)</p>
    <div class="tablewrap"><table class="ranks"><thead><tr><th>term</th>{head}</tr></thead><tbody>{aRows}</tbody></table></div>
  </div>""")
    return '<div class="families">\n' + "\n".join(out) + "\n</div>"


def model_table(res: dict) -> str:
    out = []
    for case in res["cases"]:
        rows = ""
        for cfg, c in case["configs"].items():
            tm, em, am = c["T_model"], c["E_model"], c["A_model"]
            rows += (f"<tr><td>{cx(cfg)}</td><td>{c['T']:,}</td><td>{tm['T_head']:,.0f}</td><td>{100 * c['T_err']:+.0f} %</td>"
                     f"<td>{tm['T_projection_stream']:,.0f}</td><td>{tm['T_block']:,.0f}</td><td>{tm['bound']}</td>"
                     f"<td>{c['E'] / 1e3:,.0f}</td><td>{em['E'] / 1e3:,.0f}</td><td>{100 * c['E_err']:+.1f} %</td>"
                     f"<td>{c['A']:.1f}</td><td>{am['A']:.1f}</td></tr>")
        out.append(f"""  <div class="family">
    <h2>{head_label(case)}</h2>
    <div class="tablewrap"><table class="ranks"><thead><tr><th>R × C</th><th>T simulator</th><th>T<sub>head</sub></th><th>error</th>
      <th>T<sub>projection_stream</sub></th><th>T<sub>block</sub></th><th>bound</th><th>E simulator (nJ)</th><th>E<sub>head</sub></th><th>error</th><th>A VTR (mm²)</th><th>A<sub>head</sub></th></tr></thead>
      <tbody>{rows}</tbody></table></div>
  </div>""")
    return '<div class="families">\n' + "\n".join(out) + "\n</div>"


def main() -> int:
    res = json.loads((STAGES_DIR / "stages.json").read_text())
    k = res["constants"]
    cases = res["cases"]
    small, large = cases[0], cases[1]
    lo, hi = res["extremes"]                                  # the smallest and the largest array
    tiles = {cfg: c["laws"]["tile_cells"] for cfg, c in small["configs"].items()}
    pace = {cfg: large["configs"][cfg]["laws"]["T_p"] for cfg in (lo, hi)}       # the large head's cycles per token
    runs = {cfg: large["configs"][cfg]["runs"] for cfg in (lo, hi)}
    assert runs[hi] == 1 < runs[lo], runs                                         # the text: "runs -> one, no overlap"
    farms = [100 * (c["configs"][cfg]["stages"]["qk"]["energy_pj"] + c["configs"][cfg]["stages"]["pv"]["energy_pj"])
             / c["configs"][cfg]["E"] for c in cases for cfg in (lo, hi)]
    errT = ", ".join(f"{c['name']} within {100 * c['errors']['T_max_abs_rel']:.0f} %" for c in cases)
    errE = ", ".join(f"{100 * c['errors']['E_max_abs_rel']:.1f} %" for c in cases)
    gantts = "".join(figure(f"g{i}", f"Time diagram: {c['name']}", f"stages_gantt_{tag(c['name'])}.png",
                            f"Busy intervals of the five stages of the {c['name']} head at {lo} and at {hi}",
                            f"One lane per stage (its busy time; the bars merge the stage's units); dashed lines = run boundaries. Top: {cx(lo)}, bottom: {cx(hi)}.")
                     for i, c in enumerate(cases))
    page = f"""<title>Attention Stage Analysis</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"></script>
<style>{KATEX_CSS}</style>
<style>{CSS}
.eq {{ margin: 10px 0 14px; overflow-x: auto; }}
.katex-display {{ margin: 0.4em 0; }}
.family table.ranks td:first-child {{ text-align: left; }}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">v3 DSE · stage analysis · 2026-10-19 · 8 configs, crossbar tiles 2026-10-22</p>
  <h1>Where an attention head's time and energy go</h1>
  <p class="lede">Two heads, the smallest and the largest of the DSE: {head_label(small)} and {head_label(large)}; two arrays, {cx(lo)} and {cx(hi)}; the DSE's fixed mapping. Numbers from the simulator; the closed forms below are checked against it on all {len(tiles)} arrays ({errT} on cycles, {errE} on energy).</p>
</header>

{gantts}
{figure("ex", f"Reading the time diagram, step by step: {small['name']} at {cx(res['extremes'][0])}",
        f"stages_explain_{tag(small['name'])}.png",
        "The time diagram of the smallest head at the smallest array with every step labelled by what it computes and arrows for what it waits on",
        "Every box says what the stage computes. Arrows say what a stage waits for. The light ticks are the per-token conversions during the stream. "
        + NOTATION,
        explain_notes(small, res["extremes"][0]))}
<section aria-labelledby="h_read">
  <h2 id="h_read">Highlights</h2>
  <div class="notes">
    <p><b>The projections stream the tokens; tiled QK, the online softmax, tiled PV and the output rescale form one pipeline behind them</b>, working in runs of S = min(n, I) queries and blocks of 16 keys. A run's QK can start once its S queries are projected, so short runs (small I) start early and overlap the stream; a run of S = n waits for every token.</p>
    <p><b>A bigger array stretches the stream and lengthens the runs</b> ({large['name']}: {pace[lo]} → {pace[hi]} cycles per token; {runs[lo]} runs → one, no overlap). The pipeline's work is the same on every array.</p>
  </div>
</section>

<section aria-labelledby="h_tab">
  <h2 id="h_tab">The stages, by the numbers</h2>
  <p class="muted">The farms (QK and PV) are {min(farms):.0f}–{max(farms):.0f} % of the energy; the projections and the softmax are the critical path.</p>
{stage_tables(res)}
</section>

<section aria-labelledby="h_eq">
  <h2 id="h_eq">The model</h2>
  <div class="notes">
    <p><b>1. Symbols and the two primitives</b></p>
    <ul>
      <li>\\(n\\) tokens, \\(d_m\\) model width, \\(d\\) head width. The array is \\(R \\times C\\) and \\(I = \\min(R, C)\\).</li>
      <li>A run is \\(S = \\min(n, I)\\) queries. There are \\(N_{{run}} = n / S\\) runs and \\(N_{{block}} = n / B_{{kv}}\\) key blocks, with \\(B_{{kv}} = 16\\).</li>
      <li>A projection has \\(V = \\lceil d_m / R \\rceil\\) tiles.</li>
      <li>A crossbar has one port. The port moves \\(w = 5\\) bytes per cycle. A pass runs \\(P = 8\\) bit-slices.</li>
    </ul>
    <div class="eq">\\[ \\tau(r) = \\max\\!\\Big(\\Big\\lceil \\tfrac{{r}}{{w}} \\Big\\rceil + P,\\ \\Big\\lceil \\tfrac{{C}}{{w}} \\Big\\rceil + 1\\Big), \\qquad e_{{pass}}(r) = 0.17\\,C + 0.00016\\,RC + 0.015\\,r + 0.010\\,C\\ \\text{{pJ}}, \\qquad e_{{port}}(r) = 0.0495\\,\\Big(\\Big\\lceil \\tfrac{{8r}}{{40}} \\Big\\rceil + \\Big\\lceil \\tfrac{{8C}}{{40}} \\Big\\rceil\\Big)\\ \\text{{pJ}} \\]</div>
    <ul>
      <li>\\(\\tau(r)\\) is the time of one pass that loads \\(r\\) rows. The rows come in through the port, then the bit-slices run, then the \\(C\\) columns drain. The drain overlaps the next load unless it is longer.</li>
      <li>\\(e_{{pass}}(r)\\) is the energy of one pass. The whole array fires every time: the ACAM on every column, every cell. Only the input buffer follows the rows loaded.</li>
      <li>\\(e_{{port}}(r)\\) is the energy of the pass's port words.</li>
      <li>A conversion pass handles \\(I\\) values. A score costs \\(d\\) conversions.</li>
    </ul>

    <p><b>2. Cycles per stage</b> (one key block of \\(S \\times B_{{kv}}\\) scores, on the stage's own crossbars)</p>
    <div class="eq">\\[ T_{{projection\\_token}} = \\tau(\\min(d_m, R)), \\qquad T_{{projection\\_stream}} = \\frac{{n \\cdot T_{{projection\\_token}}}}{{P_{{proj}}}} \\]</div>
    <ul>
      <li>A projection pass loads one token's \\(\\min(d_m, R)\\) inputs. The \\(V\\) tiles load in parallel.</li>
      <li>The three projections run in lock step. Each copy takes one token per \\(T_{{projection\\_token}}\\) cycles.</li>
    </ul>
    <div class="eq">\\[ T_{{QK}} = \\frac{{S B_{{kv}} d}}{{I N_E}}\\,\\tau(I), \\qquad T_{{PV}} = \\frac{{S d B_{{kv}}}}{{I N_{{EP}}}}\\,\\tau(I) \\]</div>
    <ul>
      <li>A block of QK needs \\(S B_{{kv}} d\\) conversions. Each pass converts \\(I\\) values. The \\(N_E\\) crossbars share the passes.</li>
      <li>PV is the same with \\(N_{{EP}}\\) crossbars.</li>
      <li>The adds and the trees are CLB logic. They keep up with the passes and add no cycles.</li>
    </ul>
    <div class="eq">\\[ T_{{reader}} = \\Big\\lceil \\tfrac{{S B_{{kv}}}}{{K_{{reader}}}} \\Big\\rceil, \\qquad T_{{EXP}} = \\frac{{S B_{{kv}}}}{{I N_{{EXP}}}}\\,\\tau(I), \\qquad T_{{softmax\\_latency}} = T_{{reader}} + T_{{EXP}} + 2\\,\\tau(S) + \\Big\\lceil \\tfrac{{S}}{{w}} \\Big\\rceil, \\qquad T_{{softmax}} = \\max\\!\\Big(T_{{reader}},\\ \\tfrac{{T_{{softmax\\_latency}}}}{{2}}\\Big) \\]</div>
    <ul>
      <li>The scores of a block enter the online softmax through one reader. The reader takes \\(K_{{reader}}\\) scores per cycle.</li>
      <li>The EXP passes convert the block's scores on \\(N_{{EXP}}\\) crossbars.</li>
      <li>The two carry passes update the running max and sum. They load \\(S\\) rows. The walk writes \\(S\\) values at \\(w\\) per cycle.</li>
      <li>Blocks pipeline two deep in the softmax. So its period is the larger of the reader and half the block latency.</li>
    </ul>
    <div class="eq">\\[ W_{{rescale}} = \\Big\\lceil \\frac{{S d}}{{I N_{{OC}}}} \\Big\\rceil, \\qquad T_{{output\\_rescale}} = \\frac{{T_{{EXP}} + T_{{PV}} + W_{{rescale}}\\,\\tau(I) + 2\\,\\tau(I)}}{{2}} \\]</div>
    <ul>
      <li>The output rescale gives every output element a rescale pass and a LOG pass per block. \\(N_{{OC}}\\) crossbars take a wave of \\(I\\) elements each, so a kind needs \\(W_{{rescale}}\\) waves.</li>
      <li>A block's rescale needs the block's row maxima from the softmax. Its LOG passes need the block's PV output. So one block goes through the EXP passes, the PV job, and the rescale waves in a chain.</li>
      <li>The softmax hands the row maxima over through a ring that holds two blocks. The ring frees a slot when the rescale has read it. So the chain may take two block periods, and the period is half the chain.</li>
    </ul>

    <p><b>3. Latency</b></p>
    <div class="eq">\\[ T_{{block}} = \\max\\big(T_{{QK}},\\ T_{{softmax}},\\ T_{{EXP}},\\ T_{{PV}},\\ T_{{output\\_rescale}}\\big) \\]</div>
    <ul>
      <li>The four core stages work on different blocks at the same time. Each has its own crossbars.</li>
      <li>So the pipeline moves one block per \\(T_{{block}}\\), the slowest stage.</li>
    </ul>
    <div class="eq">\\[ T_{{stream\\_bound}} = T_{{projection\\_stream}} + \\max(1, N_{{run}} - 1)\\, N_{{block}}\\, T_{{block}}, \\qquad T_{{core\\_bound}} = \\frac{{S \\cdot T_{{projection\\_token}}}}{{P_{{proj}}}} + N_{{run}} N_{{block}}\\, T_{{block}} \\]</div>
    <ul>
      <li>A run can start when its \\(S\\) queries are projected. Its blocks also need their keys, which the stream delivers in order.</li>
      <li>If the stream is slower than the core, the first run follows the stream block by block. The other runs come after the stream, one after another. With one run, all of it comes after the stream.</li>
      <li>If the core is slower, the stream waits for it. Then the first run starts after \\(S\\) tokens and every block takes \\(T_{{block}}\\).</li>
      <li>The larger of the two bounds holds.</li>
    </ul>
    <div class="eq">\\[ T_{{fill}} = \\lceil \\log_2 V \\rceil + 4 + 24, \\qquad T_{{first\\_block}} = N_{{run}}\\,(T_{{softmax\\_latency}} - T_{{block}}), \\qquad T_{{readout}} = (N_{{run}} - 1)(S + 4), \\qquad T_{{tail}} = T_{{PV}} + W_{{rescale}}\\,\\tau(I) \\]</div>
    <ul>
      <li>\\(T_{{fill}}\\): the projection output path and the first QK job.</li>
      <li>\\(T_{{first\\_block}}\\): a run's first block pays the full softmax latency. The others follow at the period.</li>
      <li>\\(T_{{readout}}\\): a run ends by reading out its \\(S\\) log-sum values. The next run starts after that.</li>
      <li>\\(T_{{tail}}\\): the last block still needs its PV job and the epilogue after the softmax is done.</li>
    </ul>
    <div class="eq">\\[ T_{{head}} = \\max\\big(T_{{stream\\_bound}},\\ T_{{core\\_bound}}\\big) + T_{{fill}} + T_{{first\\_block}} + T_{{readout}} + T_{{tail}} \\]</div>

    <p><b>4. Energy per stage</b> (pJ; \\(e_{{add}} = 0.085\\), \\(e_{{compare}} = 0.264\\), \\(e_{{memory}} = 0.0495\\) per 5 bytes)</p>
    <div class="eq">\\[ E_{{projection}} = 3Vn\\,[e_{{pass}}(\\min(d_m, R)) + e_{{port}}] + 3nd(V - 1)\\,e_{{add}} \\]</div>
    <div class="eq">\\[ E_{{QK}} = \\Big(\\tfrac{{n^2 d}}{{I}} + 2n\\Big)[e_{{pass}}(I) + e_{{port}}] + (2n^2 d - n^2)\\,e_{{add}} + n^2\\,e_{{compare}} \\]</div>
    <div class="eq">\\[ E_{{softmax}} = \\Big(\\tfrac{{n^2}}{{I}} + N_{{run}}(2N_{{block}} - 1)\\Big)[e_{{pass}}(I) + e_{{port}}] + \\tfrac{{17}}{{8}} n^2\\,e_{{add}} + \\tfrac{{17}}{{16}} n^2\\,e_{{compare}} \\]</div>
    <div class="eq">\\[ E_{{PV}} = \\Big(\\tfrac{{n^2 d}}{{I}} + n\\Big)[e_{{pass}}(I) + e_{{port}}] + \\Big(2n^2 d - \\tfrac{{n^2 d}}{{B_{{kv}}}}\\Big)e_{{add}} \\]</div>
    <div class="eq">\\[ E_{{output\\_rescale}} = \\tfrac{{2 n^2 d}}{{I B_{{kv}}}}\\,[e_{{pass}}(I) + e_{{port}}] + \\tfrac{{n^2 d}}{{8}}\\,e_{{add}} \\]</div>
    <div class="eq">\\[ E_{{memory}} \\approx \\tfrac{{e_{{memory}}}}{{5}}\\Big(3nd + \\tfrac{{2n^2 d}}{{B_{{kv}}}} + \\tfrac{{2n^2 d}}{{S}} + 4n^2 + \\tfrac{{7 n^2 d}}{{B_{{kv}}}}\\Big), \\qquad E_{{head}} = E_{{projection}} + E_{{QK}} + E_{{softmax}} + E_{{PV}} + E_{{output\\_rescale}} + E_{{memory}} \\]</div>
    <ul>
      <li>Each stage's energy is its passes times the pass price, plus its CLB work.</li>
      <li>The passes are the work divided by \\(I\\). The price grows with \\(C\\) and with \\(R C\\). So the pass energy grows as fired cells per converted value, \\(RC / I\\).</li>
      <li>The CLB adds and compares do not depend on the array. QK and PV each add about \\(2 n^2 d\\) times.</li>
      <li>\\(E_{{memory}}\\) is the block-RAM traffic of the stages. It is about 1 % of the head.</li>
    </ul>

    <p><b>5. Area per stage</b> (\\(A_{{tile}}(R, C)\\) = the tile's grid cells × 66,117 MWTA × 0.033864 µm²; the row-parallel tile of <code>experiment.md</code> §5.1: {tiles[lo]} cells at {cx(lo)}, {tiles[hi]} at {cx(hi)})</p>
    <div class="eq">\\[ A_{{projection}} = 3 P_{{proj}} V\\, A_{{tile}}, \\quad A_{{QK}} = (N_E + 2P_{{proj}})\\, A_{{tile}}, \\quad A_{{softmax}} = (N_{{EXP}} + 2)\\, A_{{tile}}, \\quad A_{{PV}} = (N_{{EP}} + P_{{proj}})\\, A_{{tile}}, \\quad A_{{output\\_rescale}} = 2 N_{{OC}}\\, A_{{tile}} \\]</div>
    <div class="eq">\\[ A_{{head}} = A_{{projection}} + A_{{QK}} + A_{{softmax}} + A_{{PV}} + A_{{output\\_rescale}} + A_{{CLB}}(I) + A_{{memory}} \\]</div>
    <ul>
      <li>Each stage's area is its crossbar count times one tile. The counts come from the mapping, not from the array.</li>
      <li>The tile grows with the array. So the crossbar area grows with the array for the same work.</li>
      <li>\\(A_{{CLB}}(I)\\) is the soft logic. It is head-level, from VTR. It grows with \\(I\\): 10.4k, 12.9k and 17.5k CLBs at \\(I\\) = 64, 128 and 256.</li>
      <li>\\(A_{{memory}}\\) is the block RAM, also from VTR.</li>
    </ul>

    <p><b>6. Our mapping</b></p>
    <ul>
      <li>\\(P_{{proj}} = 1\\), \\(N_E = N_{{EP}} = 2d\\), \\(N_{{EXP}} = 2\\), \\(N_{{OC}}\\) by the floor rule (2 to 8), \\(B_{{kv}} = 16\\), \\(S = \\min(n, I)\\), \\(K_{{reader}} = 5\\).</li>
      <li>With these, \\(T_{{QK}} = T_{{PV}} = T_{{EXP}}\\) and \\(T_{{reader}} = S B_{{kv}} / 5\\). The output rescale's chain is the largest term on every array. So \\(T_{{block}} = T_{{output\\_rescale}}\\).</li>
      <li>The array enters the core only through \\(S\\) and \\(\\tau(I)\\). It enters the stream through \\(\\min(d_m, R)\\). It enters the energy through \\(e_{{pass}} / I\\). It enters the area through \\(A_{{tile}}\\) and \\(A_{{CLB}}(I)\\).</li>
    </ul>
  </div>
{model_table(res)}
</section>

{figure("sw", "Sweeping the array: cycles, energy, area and their product", "stages_sweep.png",
        f"Cycles, energy, area and cycles × area of both heads against R, one curve per C, as lines of the closed forms with the simulator's {len(tiles)} configs as markers",
        "The DSE grid, one curve per C against R. Lines: the closed forms T<sub>head</sub>, E<sub>head</sub>, A<sub>head</sub>. Markers: the simulator (area: VTR's counts). The last panel multiplies cycles by area and divides by the product at 128 × 64.",
        sweep_notes(res))}
<div class="notes muted">
  <p>Data <code>v3/dse/results/stages/stages.json</code> (<code>dse/stage_analysis.py</code>), figures <code>dse/plot_stages.py</code>, page <code>dse/stages_page.py</code>, write-up <code>v3/docs/attention_stages.md</code>. BUF {k['BUF']}, P {k['P']}, B<sub>kv</sub> {k['BKV']}.</p>
</div>
</div>
<script>{SCRIPT}</script>
<script>
document.addEventListener("DOMContentLoaded", function () {{
  if (window.renderMathInElement) renderMathInElement(document.body, {{delimiters: [{{left: "\\\\[", right: "\\\\]", display: true}}, {{left: "\\\\(", right: "\\\\)", display: false}}], throwOnError: false}});
}});
</script>
"""
    out = STAGES_DIR / "stages.html"
    out.write_text(page)
    fonts = fetchFonts()
    print(f"wrote {out} ({len(page)} bytes); publish with the 3 PNGs and the {len(fonts)} fonts of {FONTS_DIR} as fonts/<name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
