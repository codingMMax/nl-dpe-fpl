#!/usr/bin/env python3
"""results_page.py — the DSE results pages, one per view:
  adp/dse_results.html        latency vs area (user, 2026-10-16) per family and across the families, from dse/rank.py's
                              adp/latency_area.json and dse/plot_dse.py's dse_la_{fc,attention}.png, dse_adp_cross.png
                              and dse_critical_path.png (published beside the page); EDAP against ADP per family, from
                              edap/ranking_*.json; the critical path from dse/config_table.py's critical_path.json
  edap/dse_results_edap.html  the earlier EDAP page (not published), from edap/ranking_*.json and its figures
The raw data only: Fmax = the mean of the 3 VTR seeds.
Usage: python3 v3/dse/results_page.py      (then publish the ADP page with its three PNGs)
"""
from __future__ import annotations

import json
import math
import statistics as st
import struct
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse import data  # noqa: E402
from dse.space import CONFIGS  # noqa: E402
from simulator.platform import area  # noqa: E402

RESULTS = V3 / "dse" / "results"
EDAP = RESULTS / "edap"            # the earlier view: energy x delay x area product ranking
ADP = RESULTS / "adp"              # the current view: latency vs area per family, area-delay product across them


def cx(c: str) -> str:
    return c.replace("x", " × ")


def counts() -> dict:
    """The pages' counts and ranges, from the priced tables (dse/data.py)."""
    at, fc = data.attentionRows(), data.fcRows()
    diff = [r for r in at if r["mem_check"] == "DIFFERS"]
    assert all(r["d"] == "64" and r["C"] == "64" for r in diff), "the memory note names the d = 64 shapes at C = 64"
    over = [int(r["mem_blocks"]) - int(r["mem_blocks_model"]) for r in diff]
    return dict(nAt=len(at), nFc=len(fc), nCfg=len(CONFIGS), nShapes=len({r["shape"] for r in at}),
                nFcShapes=len({data.fcWorkload(r) for r in fc}), nMem=len(diff), overLo=min(over), overHi=max(over),
                vprLo=min(int(r["mem_blocks"]) for r in diff), vprHi=max(int(r["mem_blocks"]) for r in diff),
                model="/".join(sorted({r["mem_blocks_model"] for r in diff})),
                fmaxLo=min(float(r["fmax_mhz"]) for r in at), fmaxHi=max(float(r["fmax_mhz"]) for r in at))


TILES = ("Crossbar tiles by the row-parallel rule (2026-10-22; experiment.md §5.1): R data pins in and C out, the "
         "height from the pins (60 per tile row), the width from the logic plus COFFE's local input crossbar, every "
         "cell one CLB tile with its routing.")


def png_size(name: str) -> tuple:
    """(width, height) of a figure, found in the ADP or the EDAP folder (the file names differ between the two)."""
    path = next(d / name for d in (ADP, EDAP) if (d / name).exists())
    return struct.unpack(">II", path.read_bytes()[16:24])


def family_table(res: dict, tid: str) -> str:
    rows = "".join(f'<tr{" class=\"top\"" if c["rank"] == 1 else ""}><td>{c["rank"]}</td><td>{cx(c["config"])}</td>'
                   f'<td>{c["G_EDAP"]:.3f}</td><td>{c["G_E"]:.2f}</td><td>{c["G_D"]:.2f}</td>'
                   f'<td>{c["G_A"]:.2f}</td></tr>' for c in res["configs"])
    return (f'<div class="tablewrap"><table id="{tid}"><thead><tr><th>#</th><th>R × C</th><th>G<sub>EDAP</sub></th>'
            f'<th title="energy ÷ the worst">G<sub>E</sub></th><th title="latency ÷ the worst">G<sub>D</sub></th>'
            f'<th title="area ÷ the worst">G<sub>A</sub></th></tr></thead><tbody>{rows}</tbody></table></div>')


def cross_table(cr: dict) -> str:
    rows = "".join(f'<tr{" class=\"top\"" if c["rank"] == 1 else ""}><td>{c["rank"]}</td><td>{cx(c["config"])}</td>'
                   f'<td>{c["G_FC"]:.3f}</td><td>{c["G_ATT"]:.3f}</td>'
                   f'<td>{c["d"]:.3f}</td><td>{"yes" if c["front"] else ""}</td></tr>' for c in cr["configs"])
    return ('<div class="tablewrap"><table id="t_cross"><thead><tr><th>#</th><th>R × C</th><th>FC G<sub>EDAP</sub></th>'
            '<th>attention G<sub>EDAP</sub></th><th>d</th><th>front</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>')


def figure(fid: str, title: str, img: str, alt: str, caption: str) -> str:
    w, h = png_size(img)
    return (f'<section aria-labelledby="h_{fid}">\n  <div class="fighead"><h2 id="h_{fid}">{title}</h2>'
            f'<button class="zoom" id="z_{fid}" type="button" aria-controls="m_{fid}" aria-pressed="false">Zoom in'
            f'</button></div>\n  <figure><div class="mat" id="m_{fid}"><img src="{img}" alt="{alt}" width="{w}" '
            f'height="{h}"></div>\n  <figcaption class="muted">{caption}</figcaption></figure>\n')


CSS = """
/* Layout: one reading column for text; figures run full width on white plot-paper mats; the two family summaries sit side by side and stack on phones */
:root {
  --bg: #f3f6f6; --surface: #ffffff; --fg: #1b2525; --muted: #586a69; --rule: #d3dcdb;
  --accent: #0a6a65; --accent-soft: #dfefed; --paper: #ffffff;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", "Helvetica Neue", sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1717; --surface: #162121; --fg: #e2eaea; --muted: #9db0ae; --rule: #2a3a39;
  --accent: #5ec7be; --accent-soft: #1b3533; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f1717; --surface: #162121; --fg: #e2eaea; --muted: #9db0ae; --rule: #2a3a39;
  --accent: #5ec7be; --accent-soft: #1b3533; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.55; }
.wrap { max-width: none; margin: 0; display: grid; gap: 40px; padding-block: 28px 56px;
  padding-left: max(24px, env(safe-area-inset-left, 0px)); padding-right: max(24px, env(safe-area-inset-right, 0px)); }
header { display: grid; gap: 10px; }
.eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0; }
h1 { font-family: var(--display); font-weight: 600; font-size: clamp(28px, 4.2vw, 40px); line-height: 1.1; margin: 0; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 600; font-size: 22px; margin: 0; text-wrap: balance; }
p { margin: 0; }
.lede { max-width: none; }
.muted { color: var(--muted); }
.families { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 20px; }
.family { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 18px 14px; display: grid; gap: 12px; min-width: 0; }
.winner { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }
.winner .cfg { font-family: var(--mono); font-weight: 500; font-size: 30px; color: var(--accent); font-variant-numeric: tabular-nums; }
.winner .g { font-family: var(--mono); font-size: 15px; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 13px; font-variant-numeric: tabular-nums; }
th { font-weight: 500; color: var(--muted); text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:nth-child(-n+2), td:nth-child(-n+2) { text-align: left; }
table.ranks th:nth-child(n+2), table.ranks td:nth-child(n+2) { text-align: right; }
tr.top td { background: var(--accent-soft); color: var(--fg); font-weight: 500; }
.legend { font-size: 13px; }
section { display: grid; gap: 14px; min-width: 0; }
.fighead { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 8px 16px; }
figure { margin: 0; display: grid; gap: 10px; min-width: 0; }
.mat { background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: 10px; overflow-x: auto; }
.mat img { display: block; width: auto; max-width: 100%; height: auto; margin-inline: auto; }
.mat.full img { max-width: none; }
figcaption { font-size: 14px; }
button.zoom { font: 500 13px var(--body); color: var(--fg); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 5px 12px; cursor: pointer; }
button.zoom:hover { border-color: var(--accent); }
button.zoom:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.notes { display: grid; gap: 6px; font-size: 14px; }
code { font-family: var(--mono); font-size: 13px; overflow-wrap: anywhere; }
"""

SCRIPT = """
document.querySelectorAll("button.zoom").forEach(function (b) {
  b.addEventListener("click", function () {
    var mat = document.getElementById(b.getAttribute("aria-controls"));
    var on = mat.classList.toggle("full");
    b.setAttribute("aria-pressed", on ? "true" : "false");
    b.textContent = on ? "Fit to width" : "Zoom in";
  });
});
"""


def la_table(rows: list, tid: str) -> str:
    body = "".join(f'<tr{" class=\"top\"" if c["front"] else ""}><td>{cx(c["config"])}</td><td>{c["G_A"]:.3f}</td>'
                   f'<td>{c["G_D"]:.3f}</td><td>{c["G_E"]:.3f}</td>'
                   f'<td>{"yes" if c["front"] else ""}</td></tr>' for c in rows)
    return (f'<div class="tablewrap"><table id="{tid}"><thead><tr><th>R × C</th><th>G<sub>A</sub> area</th>'
            f'<th>G<sub>D</sub> latency</th><th>G<sub>E</sub> energy</th><th>front</th></tr></thead><tbody>{body}'
            f'</tbody></table></div>')


def la_section(fid: str, title: str, img: str, alt: str, caption: str, rows: list) -> str:
    return figure(fid, title, img, alt, caption) + "  " + la_table(rows, f"t_{fid}") + "\n</section>\n"


def front_text(rows: list, key: str = "G_A") -> str:
    fr = [c for c in rows if c["front"]]
    return ", ".join(cx(c["config"]) for c in sorted(fr, key=lambda c: c[key]))


def adp_table(rows: list) -> str:
    body = "".join(f'<tr{" class=\"top\"" if c["front"] else ""}><td>{c["rank"]}</td><td>{cx(c["config"])}</td>'
                   f'<td>{c["G_FC"]:.3f}</td><td>{c["G_ATT"]:.3f}</td><td>{c["d"]:.3f}</td>'
                   f'<td>{c["G_E_FC"]:.3f}</td><td>{c["G_E_ATT"]:.3f}</td>'
                   f'<td>{"yes" if c["front"] else ""}</td></tr>' for c in rows)
    return ('<div class="tablewrap"><table id="t_cross"><thead><tr><th>#</th><th>R × C</th><th>FC G<sub>ADP</sub></th>'
            '<th>attention G<sub>ADP</sub></th><th>d</th><th>FC G<sub>E</sub></th><th>attention G<sub>E</sub></th>'
            f'<th>front</th></tr></thead><tbody>{body}</tbody></table></div>')


def compare_table(fam: str, la: dict, tid: str) -> str:
    """EDAP against ADP for one family (user, 2026-10-17: R x C, the ADP rank, the EDAP rank, nothing else), rows in
    ADP order."""
    ed = {c["config"]: c for c in json.loads((EDAP / f"ranking_{fam}.json").read_text())["configs"]}
    g = la["adp"][fam]["G_ADP"]
    if set(g) != set(ed):
        raise ValueError(f"{fam}: the EDAP and ADP results cover different configs")
    adp_order = sorted(g, key=lambda c: (g[c], c))
    body = "".join(f'<tr><td>{cx(c)}</td><td>{k + 1}</td><td>{ed[c]["rank"]}</td></tr>' for k, c in enumerate(adp_order))
    return (f'<div class="tablewrap"><table class="ranks" id="{tid}"><thead><tr><th>R × C</th><th>ADP rank</th>'
            f'<th>EDAP rank</th></tr></thead><tbody>{body}</tbody></table></div>')


def edap_vs_adp_section(la: dict) -> str:
    """The two rankings side by side per family (user, 2026-10-17)."""
    cards = [f'  <div class="family">\n    <h2>{title}</h2>\n    {compare_table(fam, la, tid)}\n  </div>'
             for fam, title, tid in (("fc", "FC", "t_cmp_fc"), ("attention", "Attention head", "t_cmp_att"))]
    cards_html = "\n".join(cards)
    return f"""<section aria-labelledby="h_cmp">
  <h2 id="h_cmp">EDAP against ADP, per family</h2>
  <p class="lede">Each config's rank by ADP and by EDAP (1 = best), rows in ADP order. EDAP is energy × ADP, so energy alone moves a config between the two.</p>
  <div class="families">
{cards_html}
  </div>
</section>
"""


CP_KINDS = (("dpe_load", "DPE load"), ("dpe_compute", "DPE compute"), ("dpe_drain", "DPE drain"), ("stream", "stream"),
            ("logic", "logic"))
CP_ROLES = (("gemm", "gemm"), ("qk", "qk"), ("sm", "sm"), ("pv", "pv"), ("acc", "acc"), ("ctrl", "ctrl"))
CP_ORDER = [f"{r}x{c}" for r, c in sorted(CONFIGS, key=lambda rc: (rc[1], rc[0]))]   # by C, then R


def cp_summary() -> dict:
    return json.loads((RESULTS / "critical_path.json").read_text())["summary"]


def cp_range(cp: dict, kind: str, configs: list) -> str:
    v = [100 * cp[c]["shares"][kind] for c in configs]
    return f"{min(v):.0f}–{max(v):.0f} %"


def cp_table(fam: dict, tid: str, keys: tuple) -> str:
    head = "".join(f"<th>{name}</th>" for _, name in keys)
    body = "".join(f"<tr><td>{cx(c)}</td><td>{fam[c]['T']:,.0f}</td>"
                   + "".join(f"<td>{100 * fam[c]['shares'][k]:.0f} %</td>" for k, _ in keys) + "</tr>"
                   for c in CP_ORDER if c in fam)
    return (f'<div class="tablewrap"><table id="{tid}" class="ranks"><thead><tr><th>R × C</th><th>cycles T</th>{head}'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def critical_path_section() -> str:
    """Where the cycles go on the critical path (experiment.md §9.3), from dse/config_table.py's critical_path.json."""
    cp = cp_summary()
    at, fc = cp["attention"], cp["fc"]
    c64 = [c for c in CP_ORDER if c.endswith("x64")]
    c256 = [c for c in CP_ORDER if c.endswith("x256")]
    drain256 = -(-8 * 256 // 40) + 1                  # a projection pass's drain of all 256 columns
    load192 = -(-8 * 192 // 40) + 8                   # the load of the largest d_model <= 192 (DeiT-T)
    assert all(at[c]["shares"]["dpe_drain"] > at[c]["shares"]["dpe_load"] for c in c256), "C = 256 is not drain-led"
    return figure("cp", "Critical path: where the cycles go", "dse_critical_path.png",
                  "Stacked bars per crossbar config: the attention head's cycles on the critical path by kind of wait and by module, and the FC rows' by kind",
                  "Each build is run in the simulator, whose times carry their history: every decision keeps the input that bound it and every added latency is a labeled segment, so the head's done carries the one chain of waits that set it, from the start to the last output word, and the chain's cycles add up to the latency exactly. (a) and (c): the chain by kind of wait, DPE load = a pass's input burst into a crossbar, DPE compute = the bit-slices, the shift-accumulate, the ACAM and the hand-overs, DPE drain = the first output word and the drain words a consumer takes, stream = a port's words or a FIFO read one word per cycle (the 40-bit paths between modules), logic = every other named latency. (b): the same chain by the module whose wait it is. Shares of the cycles, the mean over the family's workloads; front configs in bold.") + f"""  <div class="families">
  <div class="family">
    <h2>Attention head: by kind, then by module</h2>
    {cp_table(at, "t_cp_att", CP_KINDS + CP_ROLES)}
  </div>
  <div class="family">
    <h2>FC: by kind</h2>
    {cp_table(fc, "t_cp_fc", CP_KINDS)}
  </div>
  </div>
  <div class="notes">
    <p><b>Reading.</b> DPE compute is {cp_range(at, 'dpe_compute', CP_ORDER)} of an attention head's cycles and never separates the configs. At C = 64 the path is the DPE load ({cp_range(at, 'dpe_load', c64)}) plus the score words streaming into the online softmax ({cp_range(at, 'stream', c64)}): more rows add load and nothing else, so down each C block only the load grows. At C = 256 the drain leads: a projection pass drains all 256 columns in {drain256} cycles, longer than the load of the four shapes with d<sub>model</sub> ≤ 192 (at most {load192}), so the drain holds {cp_range(at, 'dpe_drain', c256)} of the path and the load {cp_range(at, 'dpe_load', c256)}; the two d<sub>model</sub> = 768 shapes stay load-bound. By module, the projections hold {cp_range(at, 'gemm', CP_ORDER)} of the path and the softmax {cp_range(at, 'sm', CP_ORDER)}. FC is one pass per row: the load {cp_range(fc, 'dpe_load', CP_ORDER)}, the output words {cp_range(fc, 'stream', CP_ORDER)}, compute {cp_range(fc, 'dpe_compute', CP_ORDER)}.</p>
    <p class="muted">A cycle that two inputs set together is charged to the first the model names (a tie). Where a stream feeds a crossbar directly (a GEMM's output into a DIMM converter) the chain records the crossbar's load, not the producer's words; the X bus's words are the projection crossbar's load. Per build: <code>v3/dse/results/critical_path.json</code>; the config tables: <code>v3/dse/results/attn_config_table.md</code>.</p>
  </div>
</section>
"""


def analysis_section() -> str:
    """Why attention and FC differ (experiment.md §9), from dse/breakdown.py's breakdown.json."""
    bd = json.loads((RESULTS / "breakdown.json").read_text())
    cp = cp_summary()["attention"]
    pick = CP_ORDER                                   # every config (8 since 2026-10-22)
    rows = "".join(
        f'<tr><td>{cx(c)}</td><td>{bd["fc"][c]["crossbars"]:.0f}</td><td>{100 * bd["fc"][c]["area_split"]["dpe"]:.0f} %</td>'
        f'<td>{bd["fc"][c]["cycles"]:.0f}</td><td>{bd["fc"][c]["fmax_mhz"]:.0f}</td>'
        f'<td>{bd["attention"][c]["crossbars"]:.0f}</td><td>{100 * bd["attention"][c]["area_split"]["clb"]:.0f} %</td>'
        f'<td>{bd["attention"][c]["cycles"]:,.0f}</td><td>{bd["attention"][c]["fmax_mhz"]:.0f}</td></tr>' for c in pick)
    at, fcb = bd["attention"], bd["fc"]
    la = {f: {c["config"]: c for c in json.loads((ADP / "latency_area.json").read_text())[f]["configs"]}
          for f in ("fc", "attention")}
    small, large = CP_ORDER[0], CP_ORDER[-1]
    rc = lambda c: tuple(int(x) for x in c.split("x"))  # noqa: E731
    used = lambda c: area.blockAreaMwta(*rc(c)) / (math.prod(area.footprint(*rc(c))) * area.MWTA_PER_CELL)  # noqa: E731
    sparse = min(at, key=lambda c: at[c]["used_cells"])
    rise = [c for c in CP_ORDER if rc(c)[0] == 512 and la["fc"][c]["G_A"] > la["fc"][f"256x{rc(c)[1]}"]["G_A"]]
    assert rise == ["512x64", "512x128"] and la["fc"]["512x256"]["G_A"] < la["fc"]["256x256"]["G_A"], rise  # the text
    clbAt = lambda C: st.mean(at[c]["clb_count"] for c in CP_ORDER if rc(c)[1] == C) / 1000  # noqa: E731
    rng = lambda fam, k, f=1: (min(fam[c][k] for c in fam) * f, max(fam[c][k] for c in fam) * f)  # noqa: E731
    share = lambda fam, k: tuple(100 * v for v in rng({c: fam[c]["area_split"] for c in fam}, k))  # noqa: E731
    paid = lambda c: at[c]["pass_price_pj"]           # noqa: E731  the mean price paid per pass on the attention builds
    kp = lambda c: f"{at[c]['passes'] / 1000:.0f}k"   # noqa: E731
    return f"""<section aria-labelledby="h_why">
  <h2 id="h_why">Why attention and FC differ</h2>
  <p class="lede">FC rewards large crossbars: a larger array holds more weights per tile, so a layer needs fewer tiles and less area, at the cost of longer passes. Attention rewards small crossbars: its crossbar count is set by the mapping, not by the array size, and each crossbar works on an I × I window (I = min(R, C)). A larger array adds area, energy per pass and soft logic, but no useful work.</p>
  <div class="tablewrap"><table id="t_why"><thead><tr><th>R × C</th><th>FC crossbars</th><th>FC tile area</th><th>FC cycles</th><th>FC MHz</th><th>attention crossbars</th><th>attention CLB area</th><th>attention cycles</th><th>attention MHz</th></tr></thead><tbody>{rows}</tbody></table></div>
  <p class="legend muted">Crossbars, cycles and MHz: means over each family's workloads. Area shares from the VPR counts, summarized as in the figures above: attention the geometric mean of each share over its shapes, FC the share of the area summed over its shapes.</p>
  <div class="notes">
    <p><b>Architecture.</b> FC is a weight-stationary GEMV: the crossbars store the weights, 1 to 64 tiles per build. Attention multiplies activations by activations, so its crossbars act as EXP and LOG converters with identity weights on one I-wide window per pass. Their number follows the head dimension (78, 146 or 278–297 crossbars for d = 16, 32, 64) and hardly changes with the array size; at {cx(sparse)} only {100 * at[sparse]['used_cells']:.1f} % of a converter's cells are in use.</p>
    <p><b>Area.</b> FC area is crossbar tiles ({share(fcb, 'dpe')[0]:.0f}–{share(fcb, 'dpe')[1]:.0f} %), and a larger tile stores a weight in less area ({fcb[large]['tile_cells'] / fcb[small]['tile_cells']:.0f}× the grid area for {math.prod(rc(large)) / math.prod(rc(small)):.0f}× the cells), so FC area falls from {cx(small)} (G<sub>A</sub> {la['fc'][small]['G_A']:.2f}) to {cx(large)} ({la['fc'][large]['G_A']:.2f}). Not at every step: at C = 64 and 128, R = 512 takes more area than R = 256 (G<sub>A</sub> {la['fc']['256x64']['G_A']:.2f} → {la['fc']['512x64']['G_A']:.2f} and {la['fc']['256x128']['G_A']:.2f} → {la['fc']['512x128']['G_A']:.2f}): its 519 input pins make the tile 9 cells tall, its local input crossbar adds {100 * area.localCrossbarUm2(512) / area.logicAreaUm2(512, 64):.0f} % and {100 * area.localCrossbarUm2(512) / area.logicAreaUm2(512, 128):.0f} % to the logic, and the area then needs just over one or two columns, so the last column is mostly empty ({100 * used('512x64'):.0f} % and {100 * used('512x128'):.0f} % of the tile used; the tile rule, experiment.md §5.1). Attention area is soft logic ({share(at, 'clb')[0]:.0f}–{share(at, 'clb')[1]:.0f} % CLB: about {clbAt(64):.1f}k CLBs at C = 64, {clbAt(128):.1f}k at 128, {clbAt(256):.1f}k at 256), plus crossbar tiles ({share(at, 'dpe')[0]:.0f}–{share(at, 'dpe')[1]:.0f} %) that grow with the array.</p>
    <p><b>Latency.</b> FC is one pass per tile, {rng(fcb, 'cycles')[0]:.0f}–{rng(fcb, 'cycles')[1]:.0f} cycles at {rng(fcb, 'fmax_mhz')[0]:.0f}–{rng(fcb, 'fmax_mhz')[1]:.0f} MHz. Attention takes {rng(at, 'cycles', 1e-3)[0]:.0f}k–{rng(at, 'cycles', 1e-3)[1]:.0f}k cycles at {rng(at, 'fmax_mhz')[0]:.0f}–{rng(at, 'fmax_mhz')[1]:.0f} MHz. Its projections stream every token through one GEMM copy and end at n × max(⌈min(d<sub>model</sub>, R)/5⌉ + 8, ⌈8C/40⌉ + 1) cycles (the X load, or the drain of all C columns), which grows with R only while d<sub>model</sub> &gt; R. The rest is bound by the score stream into the softmax (one 40-bit port, about n²/5 cycles), whatever the array size. Where the cycles go, exactly, is the critical-path section below: DPE compute is {cp_range(cp, 'dpe_compute', CP_ORDER)} of them everywhere.</p>
    <p><b>Energy.</b> A crossbar pass fires the whole array (the cells, the ACAM on every column, the output buffer) and reads only the rows in use, so an attention pass costs {paid(small):.1f} pJ at {cx(small)} and {paid(large):.1f} pJ at {cx(large)} on average (the full array would be {at[small]['pass_price_full_pj']:.1f} and {at[large]['pass_price_full_pj']:.1f}). FC needs fewer passes as the array grows (30 to 2), so its energy falls with size. Attention's pass count depends on I only (about {kp('128x64')}, {kp('128x128')}, {kp('256x256')} at I = 64, 128, 256), and the price still grows with R through the cells and the ACAM, so at a fixed I its energy rises with R.</p>
    <p><b>Assumptions behind it.</b> The fixed attention mapping (the crossbar count does not follow the array size), the pass price (the whole array fires; only the rows in use are read), one projection copy, the 40-bit ports in the cycles (the row-parallel tile of 2026-10-22 is in the area only), and FC as a single GEMV. Changing any of them could move the attention ranking.</p>
  </div>
</section>
"""


def main() -> int:
    la = json.loads((ADP / "latency_area.json").read_text())
    nfc, nat = len(la["fc"]["workloads"]), len(la["attention"]["workloads"])
    k = counts()
    page = f"""<title>NL-DPE Crossbar DSE</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">v3 DSE · VTR seeds 11, 13, 17 · 2026-10-16 · crossbar tiles 2026-10-22</p>
  <h1>NL-DPE Crossbar DSE</h1>
  <p class="lede">Which crossbar size, rows R × columns C, gives the lowest latency and area? Attention: {k['nAt']} builds of one head ({k['nShapes']} shapes × {k['nCfg']} configs), each placed and routed with three VTR seeds. FC: {k['nFc']} rows of the v2 snapshot ({k['nFcShapes']} shapes × {k['nCfg']} configs). {TILES} One notation throughout: on each workload, a config's value is divided by the worst config's value, so the worst config scores 1 and lower is better; G is the geometric mean of those scores over a family's workloads, G<sub>A</sub> for area, G<sub>D</sub> for latency, G<sub>E</sub> for energy and G<sub>ADP</sub> for area × latency. Per family the plot is G<sub>A</sub> against G<sub>D</sub>; across the families, each config's G<sub>ADP</sub> in FC is x and in attention y. The Pareto front is the set of configs that no other config beats on both axes. G<sub>E</sub> is shown beside the fronts and is not part of them.</p>
</header>

{edap_vs_adp_section(la)}
{la_section("fc", "FC", "dse_la_fc.png",
            f"FC: G_D against G_A for {k['nCfg']} crossbar configs with the Pareto front, then each config's area split and energy split as stacked bars",
            f"{nfc} workloads, M × K × N: 1 × 512 × 128, 1 × 2048 × 256, 1 × 256 × 512, 1 × 512 × 512. (a) Pareto front, from the smallest area to the lowest latency: {front_text(la['fc']['configs'])}. (b) Each config's area split into DPE tiles, CLBs and memory blocks (the VPR counts). (c) Its energy split into the DPE passes, the CLB adds, the passes' block-RAM port traffic and the memory bytes (the simulator's ledger). Both as the share of the total summed over the {nfc} shapes: several FC builds have no memory block and no CLB add, so a geometric mean of those shares would be 0. Front configs in bold.",
            la["fc"]["configs"])}
{la_section("att", "Attention head", "dse_la_attention.png",
            f"Attention: G_D against G_A for {k['nCfg']} crossbar configs with the Pareto front, then each config's area split and energy split as stacked bars",
            f"{nat} workloads (n, d<sub>model</sub>, d): ViT-B/16 (256, 768, 64), DeiT-T (256, 192, 64), BERT-base (128, 768, 64), BERT-Tiny (128, 128, 64), JetFormer-150P (32, 64, 32), JetFormer-JetClass (128, 128, 16). (a) Pareto front: {front_text(la['attention']['configs'])}, the smallest area and the lowest latency at once. (b) Each config's area split into DPE tiles, CLBs and memory blocks (the VPR counts). (c) Its energy split into the DPE passes, the CLB adds and compares, the passes' block-RAM port traffic and the memory bytes (the simulator's ledger). Both as the geometric mean of each share over the {nat} shapes. Front config in bold.",
            la["attention"]["configs"])}
{analysis_section()}
{critical_path_section()}
{figure("cross", "Across the families: area-delay product", "dse_adp_cross.png",
        f"Attention G_ADP against the FC G_ADP for {k['nCfg']} crossbar configs, with the Pareto front",
        f"x = G<sub>ADP</sub> in FC, y = G<sub>ADP</sub> in attention; each is area × latency ÷ the largest on each workload, geometric mean over the family's workloads. Pareto front, from the best in FC to the best in attention: {front_text(la['adp']['cross']['configs'], 'G_FC')}. The circles mark equal distance d from the origin; the table is sorted by d. {cx(la['adp']['cross']['configs'][0]['config'])} (d = {la['adp']['cross']['configs'][0]['d']:.3f}) and {cx(la['adp']['cross']['configs'][1]['config'])} (d = {la['adp']['cross']['configs'][1]['d']:.3f}) are {100 * (la['adp']['cross']['configs'][1]['d'] / la['adp']['cross']['configs'][0]['d'] - 1):.2f} % apart: a tie.")}  {adp_table(la["adp"]["cross"]["configs"])}
</section>
<div class="notes muted">
  <p>Latency = cycles ÷ Fmax, with Fmax the mean of the three VTR seeds; area counts CLBs, crossbar tiles and memory blocks from VPR. The {k['nMem']} attention builds of the d = 64 shapes at C = 64 use VPR's own memory count, {k['overLo']}–{k['overHi']} blocks above the simulator's prediction.</p>
  <p>The three panels side by side (for a paper): <code>v3/dse/results/adp/dse_la.pdf</code>. Scores: <code>v3/dse/results/adp/latency_area.json</code>. The earlier EDAP rankings stay in <code>v3/dse/results/edap/ranking_*.json</code>.</p>
</div>
</div>
<script>{SCRIPT}</script>
"""
    out = ADP / "dse_results.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page)} bytes)")
    return 0


def main_edap() -> int:
    """The earlier EDAP page (superseded 2026-10-16 by the latency vs area page, main)."""
    fc = json.loads((EDAP / "ranking_fc.json").read_text())
    at = json.loads((EDAP / "ranking_attention.json").read_text())
    cr = json.loads((EDAP / "ranking_cross.json").read_text())
    a1, f1, c1 = at["configs"][0], fc["configs"][0], cr["configs"][0]
    c2 = cr["configs"][1]
    front = ", ".join(cx(c["config"]) for c in sorted((c for c in cr["configs"] if c["front"]), key=lambda c: c["G_FC"]))
    best_all = all(v == a1["config"] for v in at["best_per_workload"].values())
    k = counts()
    page = f"""<title>NL-DPE Crossbar DSE</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">v3 DSE · VTR seeds 11, 13, 17 · 2026-10-16 · crossbar tiles 2026-10-22</p>
  <h1>NL-DPE Crossbar DSE</h1>
  <p class="lede">Which crossbar size, rows R × columns C, gives the lowest EDAP (energy × latency × area)? Attention: {k['nAt']} builds of one head ({k['nShapes']} shapes × {k['nCfg']} configs), each placed and routed with three VTR seeds. FC: {k['nFc']} rows of the v2 snapshot ({k['nFcShapes']} shapes × {k['nCfg']} configs). {TILES} The notation of the latency vs area page: on each workload, a config's score is its EDAP ÷ the worst config's EDAP, so the worst scores 1 and lower is better; G<sub>EDAP</sub> is the geometric mean of its scores, and G<sub>E</sub>, G<sub>D</sub>, G<sub>A</sub> the same for energy, latency and area. Fmax is the mean of the three seeds.</p>
</header>

<div class="families">
  <div class="family">
    <h2>Attention head</h2>
    <div class="winner"><span class="cfg">{cx(a1["config"])}</span><span class="g">G<sub>EDAP</sub> {a1["G_EDAP"]:.3f}</span></div>
    <p class="muted">{"Best on all " + str(len(at["workloads"])) + " workloads. " if best_all else ""}Workloads (n, d<sub>model</sub>, d): ViT-B/16 (256, 768, 64), DeiT-T (256, 192, 64), BERT-base (128, 768, 64), BERT-Tiny (128, 128, 64), JetFormer-150P (32, 64, 32), JetFormer-JetClass (128, 128, 16).</p>
    {family_table(at, "t_att")}
  </div>
  <div class="family">
    <h2>FC</h2>
    <div class="winner"><span class="cfg">{cx(f1["config"])}</span><span class="g">G<sub>EDAP</sub> {f1["G_EDAP"]:.3f}</span></div>
    <p class="muted">v2's ranking used its own tiles (the legacy rule); the row-parallel tiles change it. Workloads M × K × N: 1 × 512 × 128, 1 × 2048 × 256, 1 × 256 × 512, 1 × 512 × 512.</p>
    {family_table(fc, "t_fc")}
  </div>
</div>
<p class="legend muted">G<sub>E</sub>, G<sub>D</sub>, G<sub>A</sub>: energy, latency and area, each ÷ the worst on the workload, geometric mean over the workloads (1.00 = the worst, lower is better).</p>

{figure("cross", "EDAP across the families", "dse_cross.png",
        f"G_EDAP in FC on x, G_EDAP in attention on y, one dot per crossbar config; closer to the origin is better and {cx(c1['config'])} is the closest",
        f"x = G<sub>EDAP</sub> in FC, y = G<sub>EDAP</sub> in attention, one dot per crossbar config; the closer a point is to the origin, the better, and d is that distance. Closest: {cx(c1['config'])}, d = {c1['d']:.3f}; next {cx(c2['config'])}, d = {c2['d']:.3f}. Pareto front: {front}.")}  {cross_table(cr)}
</section>

{figure("att", "Attention head ranking", "dse_attention.png",
        f"Attention head ranking: G_EDAP map over R and C, per-workload scores, G_EDAP per config, and G_E, G_D, G_A for {k['nCfg']} crossbar configs",
        "(a) G<sub>EDAP</sub> on the R × C grid. (b) Each workload's EDAP score, configs in rank order. (c) G<sub>EDAP</sub> per config. (d) G<sub>E</sub>, G<sub>D</sub> and G<sub>A</sub>.")}</section>

{figure("fc", "FC ranking", "dse_fc.png",
        f"FC ranking: G_EDAP map over R and C, per-workload scores, G_EDAP per config, and G_E, G_D, G_A for {k['nCfg']} crossbar configs",
        "The same four panels for the FC family.")}</section>

{figure("par", "Pareto fronts per family", "dse_pareto.png",
        "Pareto fronts: G_D against G_E times G_A, for FC and the attention head",
        f"x = G<sub>D</sub>, y = G<sub>E</sub> × G<sub>A</sub> (each value ÷ the worst on the workload, geometric means over the workloads). The diagonal lines are equal-EDAP lines, so their order is the ranking. Attention: {cx(a1['config'])} beats every other config on both axes.")}</section>

<div class="notes muted">
  <p>The {k['nMem']} attention builds of the d = 64 shapes at C = 64 are priced from VPR's own memory count, which is {k['overLo']}–{k['overHi']} blocks above the simulator's prediction ({k['vprLo']}–{k['vprHi']} against {k['model']}). Every other count matches the prediction.</p>
  <p>Fmax is the mean of the three seeds ({k['fmaxLo']:.0f}–{k['fmaxHi']:.0f} MHz over the {k['nAt']} builds). Latency = cycles ÷ Fmax; area counts CLBs, crossbar tiles and memory blocks from VPR.</p>
  <p>Data: <code>v3/dse/results/attn_priced_results.csv</code>, <code>v3/dse/results/fc_priced_results.csv</code>, <code>v3/dse/results/edap/ranking_fc.json</code>, <code>v3/dse/results/edap/ranking_attention.json</code>, <code>v3/dse/results/edap/ranking_cross.json</code>.</p>
</div>
</div>
<script>{SCRIPT}</script>
"""
    out = EDAP / "dse_results_edap.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    main()
    main_edap()
