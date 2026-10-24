#!/usr/bin/env python3
"""compare_page.py — the DSE as measured beside its projection to the bit-plane design (user, 2026-10-23: "create an
HTML comparison figure including the FC/Attention EDAP, ADP ranking, pareto front. use the exact same format/layout as
NL-DPE Crossbar DSE HTML. but do an aligned comparsion in the same HTML").

Every section of the two DSE pages appears twice, aligned (dse/results_page.py's CSS, script and tables):
  Old        dse/results/: the priced tables of the pre-bit-plane RTL's cycles (the VTR run of 2026-10-16, the crossbar
             tiles of 2026-10-22) and the rankings and figures made from them
  Projected  dse/results/projection/: dse/projection.py's tables (the bit-plane simulator; each build's Fmax and CLB
             held at the old VTR values, its crossbars and memory blocks predicted) and the same scripts' rankings and
             figures (DSE_RESULTS=v3/dse/results/projection: rank.py, breakdown.py, plot_dse.py)
Sections: the assumptions, the calibration at the smoke point, what the projection changes, what moved, EDAP against
ADP per family (the ranks and their shift), latency vs area per family, ADP across the families, the EDAP rankings
per family, the Pareto fronts, EDAP across the families. Left out: the critical path and "why attention and FC
differ" (both analyze the pre-bit-plane cycles). STALE since the DIMM revision of 2026-10-23 (the projection carries
N_E = 2d and the adders in logic); stage 5's results supersede it. The calibration reads the bit-plane VTR smoke from
dse/results/attn_vtr_bitplane.csv once it has a result.
Output: dse/results/compare/dse_compare.html and the figures it shows (old_*.png, proj_*.png, copied beside it).
Usage: python3 v3/dse/compare_page.py      (then publish the page with the 14 PNGs)
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import statistics as st
import struct
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse import results_page as rp  # noqa: E402
from dse.space import CONFIGS, FC_SHAPES, SHAPES  # noqa: E402

OLD = V3 / "dse" / "results"
NEW = OLD / "projection"
OUT = OLD / "compare"
BITPLANE_CSV = (OLD / "attn_vtr_bitplane.csv", OLD / "fc_vtr_bitplane.csv")    # dse/vtr_dse.py, the bit-plane builds
SMOKE = ("BERT-Tiny", "256x128")                                                   # the bit-plane VTR smoke point
TAGS = {"old": "Old · pre-bit-plane cycles", "proj": "Projected · bit-plane"}
ORDER = [f"{r}x{c}" for r, c in sorted(CONFIGS, key=lambda rc: (rc[1], rc[0]))]   # by C, then R
FIGURES = (("adp", "dse_la_fc.png"), ("adp", "dse_la_attention.png"), ("adp", "dse_adp_cross.png"),
           ("edap", "dse_fc.png"), ("edap", "dse_attention.png"), ("edap", "dse_pareto.png"), ("edap", "dse_cross.png"))

CSS = rp.CSS + """
/* The comparison: Old and Projected side by side where a figure is near square, one above the other where it is wide */
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr)); gap: 20px; align-items: start; }
.stack { display: grid; gap: 18px; }
.side { display: grid; gap: 8px; min-width: 0; }
.tag { font-family: var(--mono); font-size: 12px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0; }
.tag.proj { color: var(--accent); }
.cap { font-size: 14px; }
.assume { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 16px; margin: 0; font-size: 14px; }
.assume dt { font-family: var(--mono); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); padding-top: 2px; }
.assume dd { margin: 0; min-width: 0; }
table.moves td { white-space: normal; text-align: left; vertical-align: top; }
.families .family { align-content: start; }
table.moves th { text-align: left; }
@media (max-width: 520px) { .assume { grid-template-columns: minmax(0, 1fr); } }
"""


def rows_of(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load(d: Path) -> dict:
    """One side: its priced rows and its rankings."""
    return dict(at=rows_of(d / "attn_priced_results.csv"), fc=rows_of(d / "fc_priced_results.csv"),
                la=json.loads((d / "adp" / "latency_area.json").read_text()),
                ed={f: json.loads((d / "edap" / f"ranking_{f}.json").read_text()) for f in ("fc", "attention", "cross")})


def gm(xs) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def adp_ranks(la: dict, fam: str) -> dict:
    """{config: rank by G_ADP} (ties by name, as results_page.compare_table)."""
    g = la["adp"][fam]["G_ADP"]
    return {c: k + 1 for k, c in enumerate(sorted(g, key=lambda c: (g[c], c)))}


def edap_ranks(ed: dict, fam: str) -> dict:
    return {c["config"]: c["rank"] for c in ed[fam]["configs"]}


def shift(old: int, new: int) -> str:
    """A rank's move from old to projected: ↑ k = k places up (better), ↓ k = down, – = the same."""
    return "–" if old == new else (f"↑{old - new}" if new < old else f"↓{new - old}")


def shift_table(fam: str, S: dict, tid: str) -> str:
    """Per config the ADP and the EDAP rank, old and projected, with the shift; rows in the projected ADP order."""
    ao, an = adp_ranks(S["old"]["la"], fam), adp_ranks(S["proj"]["la"], fam)
    eo, en = edap_ranks(S["old"]["ed"], fam), edap_ranks(S["proj"]["ed"], fam)
    if not set(ao) == set(an) == set(eo) == set(en):
        raise ValueError(f"{fam}: the two sides rank different configs")
    body = "".join(f"<tr><td>{rp.cx(c)}</td><td>{ao[c]}</td><td>{an[c]}</td><td>{shift(ao[c], an[c])}</td>"
                   f"<td>{eo[c]}</td><td>{en[c]}</td><td>{shift(eo[c], en[c])}</td></tr>" for c in sorted(an, key=an.get))
    return (f'<div class="tablewrap"><table class="ranks" id="{tid}"><thead><tr><th>R × C</th><th>ADP old</th>'
            f'<th>ADP proj.</th><th>shift</th><th>EDAP old</th><th>EDAP proj.</th><th>shift</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def shot(fid: str, side: str, name: str, alt: str) -> str:
    """One side's figure (copied into OUT as <side>_<name>): its tag, the zoom button, the image."""
    w, h = struct.unpack(">II", (OUT / f"{side}_{name}").read_bytes()[16:24])
    return (f'    <figure><div class="fighead"><p class="tag {side}">{TAGS[side]}</p><button class="zoom" '
            f'id="z_{fid}_{side}" type="button" aria-controls="m_{fid}_{side}" aria-pressed="false">Zoom in</button>'
            f'</div>\n      <div class="mat" id="m_{fid}_{side}"><img src="{side}_{name}" alt="{alt} ({TAGS[side]})" '
            f'width="{w}" height="{h}"></div></figure>')


def sides(f) -> str:
    """Two blocks side by side, f(side) for Old and for Projected, each under its tag."""
    return ('  <div class="pair">\n' + "\n".join(f'    <div class="side"><p class="tag {s}">{TAGS[s]}</p>{f(s)}</div>'
                                                 for s in TAGS) + "\n  </div>\n")


def fig_section(fid: str, title: str, name: str, alt: str, caption: str, layout: str, tables=None) -> str:
    """A figure of both sides (layout "pair": side by side; "stack": one above the other), the caption, and the two
    sides' tables side by side (tables(side) -> html)."""
    figs = "\n".join(shot(fid, s, name, alt) for s in TAGS)
    return (f'<section aria-labelledby="h_{fid}">\n  <h2 id="h_{fid}">{title}</h2>\n  <div class="{layout}">\n{figs}\n'
            f'  </div>\n  <p class="muted cap">{caption}</p>\n{sides(tables) if tables else ""}</section>\n')


def front(rows: list, key: str) -> str:
    return rp.front_text(rows, key) if any(c["front"] for c in rows) else "none"


def first(cfgs: list, key: str) -> str:
    c = cfgs[0]
    return f"{rp.cx(c['config'])} ({key} {c[key]:.3f})"


def moves_table(rows: list) -> str:
    keep = lambda t: t.replace(" × ", "&nbsp;×&nbsp;").replace("(d ", "(d&nbsp;")  # noqa: E731  (a config on one line)
    body = "".join(f"<tr><td>{v}</td><td>{keep(o)}</td><td>{keep(p)}</td></tr>" for v, o, p in rows)
    return (f'<div class="tablewrap"><table class="moves"><thead><tr><th>view</th><th>old</th><th>projected</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def changes(S: dict) -> tuple:
    """Per config, what the projection changes (attention: per build ratios and means; FC: means and sums), and the
    largest energy change on any build; with the checks behind the page's text."""
    at = {s: {(r["shape"], r["config"]): r for r in S[s]["at"]} for s in TAGS}
    fc = {s: {(r["config"], r["M"], r["K"], r["N"]): r for r in S[s]["fc"]} for s in TAGS}
    if set(at["old"]) != set(at["proj"]) or set(fc["old"]) != set(fc["proj"]):
        raise ValueError("the two sides price different builds")
    num = lambda r, k: float(r[k])  # noqa: E731
    for fam in (at, fc):
        assert all(num(fam["proj"][k], "T_cycles") < num(fam["old"][k], "T_cycles") for k in fam["old"]), "cycles"
        assert all(fam["proj"][k]["fmax_mhz"] == fam["old"][k]["fmax_mhz"] and
                   fam["proj"][k]["clb_count"] == fam["old"][k]["clb_count"] for k in fam["old"]), "Fmax, CLB held"
    assert all(int(at["proj"][k]["mem_blocks"]) > int(at["old"][k]["mem_blocks"]) for k in at["old"]), "memory rises"
    assert all(r["mem_blocks"] == "0" for r in S["proj"]["fc"]), "the bit-plane GEMM has no memory"
    de = max(abs(num(fam["proj"][k], "e_total_pj") / num(fam["old"][k], "e_total_pj") - 1) for fam in (at, fc)
             for k in fam["old"])
    out = {}
    for c in ORDER:
        ka, kf = [k for k in at["old"] if k[1] == c], [k for k in fc["old"] if k[0] == c]
        ratio = lambda fam, ks, key: gm(num(fam["proj"][k], key) / num(fam["old"][k], key) for k in ks)  # noqa: E731
        mean = lambda fam, ks, s, key: st.mean(num(fam[s][k], key) for k in ks)  # noqa: E731
        out[c] = dict(at_T=ratio(at, ka, "T_cycles"), at_mem=(mean(at, ka, "old", "mem_blocks"),
                                                              mean(at, ka, "proj", "mem_blocks")),
                      at_A=ratio(at, ka, "area_total_mm2"), fc_T=(mean(fc, kf, "old", "T_cycles"),
                                                                 mean(fc, kf, "proj", "T_cycles")),
                      fc_mem=(sum(num(fc["old"][k], "mem_blocks") for k in kf), 0),
                      fc_A=ratio(fc, kf, "area_total_mwta"))
    return out, de, (min(int(r["T_cycles"]) for r in S["old"]["fc"]), max(int(r["T_cycles"]) for r in S["old"]["fc"]),
                     min(int(r["T_cycles"]) for r in S["proj"]["fc"]), max(int(r["T_cycles"]) for r in S["proj"]["fc"]))


def changes_section(ch: dict, de: float, fcT: tuple) -> str:
    rows = "".join(f'<tr><td>{rp.cx(c)}</td><td>{v["at_T"]:.2f}</td><td>{v["at_mem"][0]:.0f} → {v["at_mem"][1]:.0f}</td>'
                   f'<td>{v["at_A"]:.3f}</td><td>{v["fc_T"][0]:.0f} → {v["fc_T"][1]:.0f}</td>'
                   f'<td>{v["fc_mem"][0]:.0f} → {v["fc_mem"][1]}</td><td>{v["fc_A"]:.3f}</td></tr>'
                   for c, v in ch.items())
    lo = min(v["at_T"] for v in ch.values())
    hi = max(v["at_T"] for v in ch.values())
    return f"""<section aria-labelledby="h_chg">
  <h2 id="h_chg">What the projection changes</h2>
  <p class="lede">Latency changes by the cycles alone, since Fmax is held. The attention head's cycles fall to {lo:.2f}–{hi:.2f} × of the old, the more the wider the array; FC rows take {fcT[2]}–{fcT[3]} cycles instead of {fcT[0]}–{fcT[1]}. The attention head's predicted memory blocks rise on every build, so its area grows a little. Energy moves by at most {100 * de:.2f} % on any build.</p>
  <div class="tablewrap"><table class="ranks" id="t_chg"><thead><tr><th>R × C</th><th>attention cycles ×</th><th>attention memory blocks</th><th>attention area ×</th><th>FC cycles</th><th>FC memory blocks</th><th>FC area ×</th></tr></thead><tbody>{rows}</tbody></table></div>
  <p class="legend muted">Projected ÷ old. Attention: cycles × and area × are geometric means of the per-build ratios over the {len(SHAPES)} shapes, memory blocks the mean per build. FC: cycles the mean over the {len(FC_SHAPES)} shapes, memory blocks summed over them (v2's old GEMM placed 0 or 1 block per row), area × the geometric mean of the ratios.</p>
</section>
"""


def calibration(S: dict) -> str:
    """The smoke point: the old VPR counts, the projected (held or predicted) values and the smoke's, when it has run."""
    shape, cfg = SMOKE
    old = next(r for r in S["old"]["at"] if (r["shape"], r["config"]) == SMOKE)
    new = next(r for r in S["proj"]["at"] if (r["shape"], r["config"]) == SMOKE)
    sm = next(r for r in rows_of(BITPLANE_CSV[0]) if (r["shape"], r["config"]) == SMOKE)
    done = bool(sm["clb"])
    meas = lambda k: f"{int(sm[k]):,}" if done else "running"  # noqa: E731
    fmax = (f"{float(sm['fmax_mhz'] or sm['fmax_s11']):.1f}" if done and (sm["fmax_mhz"] or sm["fmax_s11"])
            else ("no timing" if done else "running"))
    rows = (("crossbars", f"{int(old['wc_vtr'])}", f"{int(new['wc_vtr'])} (simulator)", meas("wc")),
            ("memory blocks", f"{int(old['mem_blocks'])}", f"{int(new['mem_blocks'])} (predicted)", meas("memory")),
            ("CLBs", f"{int(old['clb_count']):,}", f"{int(new['clb_count']):,} (held)", meas("clb")),
            ("Fmax (MHz)", f"{float(old['fmax_mhz']):.1f}", f"{float(new['fmax_mhz']):.1f} (held)", fmax))
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)
    seeds = sm["seeds"] or "11"
    note = (f"Measured by VPR on the bit-plane RTL, seed {seeds}." if done else
            "The smoke runs VTR with seed 11 on the bit-plane head (started 2026-10-23, 10–20 h [estimate]); its "
            "column fills in when it ends.")
    return f"""  <div class="family">
    <h2>Calibration at the smoke point</h2>
    <p class="muted">{shape} at {rp.cx(cfg)}: the build the bit-plane VTR smoke places and routes. Old Fmax is the mean of seeds 11, 13, 17 (seed 11 alone: {float(old['fmax_s11']):.1f} MHz).</p>
    <div class="tablewrap"><table class="ranks" id="t_cal"><thead><tr><th>quantity</th><th>old (VPR)</th><th>projected</th><th>smoke (VPR)</th></tr></thead><tbody>{body}</tbody></table></div>
    <p class="legend muted">{note}</p>
  </div>"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for side, d in (("old", OLD), ("proj", NEW)):
        for sub, name in FIGURES:
            shutil.copyfile(d / sub / name, OUT / f"{side}_{name}")
    S = {"old": load(OLD), "proj": load(NEW)}
    la = {s: S[s]["la"] for s in TAGS}
    ed = {s: S[s]["ed"] for s in TAGS}
    for s in TAGS:                                     # the same design space on both sides
        assert len(S[s]["at"]) == len(SHAPES) * len(CONFIGS) and len(S[s]["fc"]) == len(FC_SHAPES) * len(CONFIGS)
        assert sorted(la[s]["fc"]["workloads"]) == sorted(la["old"]["fc"]["workloads"])
        assert sorted(la[s]["attention"]["workloads"]) == sorted(la["old"]["attention"]["workloads"])
    elab = [r for p in BITPLANE_CSV for r in rows_of(p)]
    assert len(elab) == len(S["old"]["at"]) + len(S["old"]["fc"]) and all(r["elab"] == "OK" for r in elab), \
        "every bit-plane build elaborates with the simulator's crossbars and memories"
    diff = [r for r in S["old"]["at"] if r["mem_check"] == "DIFFERS"]
    assert all(r["d"] == "64" and r["C"] == "64" for r in diff), "the memory note names the d = 64 shapes at C = 64"
    over = [int(r["mem_blocks"]) - int(r["mem_blocks_model"]) for r in diff]
    ch, de, fcT = changes(S)
    nfc, nat = len(la["old"]["fc"]["workloads"]), len(la["old"]["attention"]["workloads"])
    cross = {s: la[s]["adp"]["cross"]["configs"] for s in TAGS}
    ecross = {s: ed[s]["cross"]["configs"] for s in TAGS}

    def mv(fam):
        return [("front, latency vs area", front(la["old"][fam]["configs"], "G_A"), front(la["proj"][fam]["configs"], "G_A")),
                ("#1 by ADP", *(rp.cx(min(la[s]["adp"][fam]["G_ADP"].items(), key=lambda kv: (kv[1], kv[0]))[0])
                                for s in TAGS)),
                ("#1 by EDAP", *(rp.cx(ed[s][fam]["configs"][0]["config"]) for s in TAGS))]
    moves_cross = [("front, ADP plane", front(cross["old"], "G_FC"), front(cross["proj"], "G_FC")),
                   ("closest, ADP plane", first(cross["old"], "d"), first(cross["proj"], "d")),
                   ("front, EDAP plane", front(ecross["old"], "G_FC"), front(ecross["proj"], "G_FC")),
                   ("closest, EDAP plane", first(ecross["old"], "d"), first(ecross["proj"], "d"))]
    fc_wl = "1 × 512 × 128, 1 × 2048 × 256, 1 × 256 × 512, 1 × 512 × 512"
    at_wl = ("ViT-B/16 (256, 768, 64), DeiT-T (256, 192, 64), BERT-base (128, 768, 64), BERT-Tiny (128, 128, 64), "
             "JetFormer-150P (32, 64, 32), JetFormer-JetClass (128, 128, 16)")
    fr = lambda fam, key="G_A": "; ".join(f"{w} {front(la[s][fam]['configs'] if fam != 'cross' else cross[s], key)}"  # noqa: E731
                                          for s, w in (("old", "old"), ("proj", "projected")))
    page = f"""<title>NL-DPE DSE Projection</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">v3 DSE · old vs projected · 2026-10-23 · before the bit-plane VTR run</p>
  <h1>NL-DPE DSE Projection</h1>
  <p class="lede">The crossbar DSE as measured, beside a projection of the same {len(S['old']['at'])} attention builds and {len(S['old']['fc'])} FC rows to the bit-plane design. Old is the pre-bit-plane RTL: its simulator cycles, the VTR run of 2026-10-16 and the crossbar tiles of 2026-10-22. Projected keeps each build's Fmax and CLB count from that VTR run, takes its crossbars and memory blocks from the simulator, and its cycles and energy from the bit-plane simulator. Every projected number is an estimate: no bit-plane netlist has been placed and routed yet. The notation is the DSE page's: on each workload a config's value is divided by the worst config's value on the same side, G is the geometric mean of those scores over a family's workloads, and lower is better. G values are relative to their own side, so across the two sides compare ranks and fronts.</p>
</header>

<div class="families">
  <div class="family">
    <h2>Assumptions</h2>
    <dl class="assume">
      <dt>held</dt><dd>Fmax (the mean of VTR seeds 11, 13, 17) and the CLB count of each build, from its previous (2026-10-16) VTR run. FC: v2's VTR run of its old GEMM.</dd>
      <dt>predicted</dt><dd>Crossbars: the simulator's count, which Yosys matched on all {len(elab)} bit-plane builds (elaboration, 2026-10-23). Memory blocks: the simulator's memories packed into blocks (area.predictedMemBlocks); Yosys's memory lists matched too, and VPR checks the block count at the smoke. FC: no memory block, the bit-plane GEMM has none.</dd>
      <dt>simulated</dt><dd>Cycles (weights preloaded, done − start) and energy (the frozen per-operation prices), from the bit-plane simulator.</dd>
      <dt>area</dt><dd>One rule on both sides: CLBs, crossbar tiles (row-parallel, 2026-10-22) and 2 cells per memory block.</dd>
      <dt>left out</dt><dd>The critical path and why attention and FC differ: both analyze the pre-bit-plane cycles.</dd>
    </dl>
  </div>
{calibration(S)}
</div>

{changes_section(ch, de, fcT)}
<section aria-labelledby="h_mv">
  <h2 id="h_mv">What moved</h2>
  <div class="families">
  <div class="family">
    <h2>FC</h2>
    {moves_table(mv("fc"))}
  </div>
  <div class="family">
    <h2>Attention head</h2>
    {moves_table(mv("attention"))}
  </div>
  <div class="family">
    <h2>Across the families</h2>
    {moves_table(moves_cross)}
  </div>
  </div>
  <p class="legend muted">Fronts in order from the smallest area (per family) or the best FC score (across the families); d = the distance from the origin.</p>
</section>

<section aria-labelledby="h_cmp">
  <h2 id="h_cmp">EDAP against ADP, per family</h2>
  <p class="lede">Each config's rank by ADP and by EDAP (1 = best), old and projected, rows in the projected ADP order. A shift of ↑2 means two places better than before.</p>
  <div class="families">
  <div class="family">
    <h2>FC</h2>
    {shift_table("fc", S, "t_cmp_fc")}
  </div>
  <div class="family">
    <h2>Attention head</h2>
    {shift_table("attention", S, "t_cmp_att")}
  </div>
  </div>
</section>

{fig_section("fc", "FC: latency vs area", "dse_la_fc.png",
             f"FC: G_D against G_A for {len(CONFIGS)} crossbar configs with the Pareto front, then each config's area split and energy split as stacked bars",
             f"{nfc} workloads, M × K × N: {fc_wl}. (a) G<sub>D</sub> against G<sub>A</sub> and the Pareto front. (b) Each config's area split into DPE tiles, CLBs and memory blocks. (c) Its energy split into the DPE passes, the CLB adds, the passes' block-RAM port traffic and the memory bytes. Both as the share of the total summed over the {nfc} shapes. Front, from the smallest area to the lowest latency: {fr('fc')}. Front configs in bold.",
             "stack", lambda s: rp.la_table(la[s]["fc"]["configs"], f"t_la_fc_{s}"))}
{fig_section("att", "Attention head: latency vs area", "dse_la_attention.png",
             f"Attention: G_D against G_A for {len(CONFIGS)} crossbar configs with the Pareto front, then each config's area split and energy split as stacked bars",
             f"{nat} workloads (n, d<sub>model</sub>, d): {at_wl}. (a) G<sub>D</sub> against G<sub>A</sub> and the Pareto front. (b) Each config's area split into DPE tiles, CLBs and memory blocks. (c) Its energy split into the DPE passes, the CLB adds and compares, the passes' block-RAM port traffic and the memory bytes. Both as the geometric mean of each share over the {nat} shapes. Front: {fr('attention')}. Front configs in bold.",
             "stack", lambda s: rp.la_table(la[s]["attention"]["configs"], f"t_la_att_{s}"))}
{fig_section("cross", "Across the families: area-delay product", "dse_adp_cross.png",
             f"Attention G_ADP against the FC G_ADP for {len(CONFIGS)} crossbar configs, with the Pareto front",
             f"x = G<sub>ADP</sub> in FC, y = G<sub>ADP</sub> in attention; each is area × latency ÷ the largest on each workload, geometric mean over the family's workloads. The circles mark equal distance d from the origin; the tables are sorted by d. Front, from the best in FC to the best in attention: {fr('cross', 'G_FC')}.",
             "pair", lambda s: rp.adp_table(cross[s], f"t_cross_{s}"))}
{fig_section("efc", "FC: ranking by EDAP", "dse_fc.png",
             f"FC ranking: G_EDAP map over R and C, per-workload scores, G_EDAP per config, and G_E, G_D, G_A for {len(CONFIGS)} crossbar configs",
             "(a) G<sub>EDAP</sub> on the R × C grid. (b) Each workload's EDAP score, configs in rank order. (c) G<sub>EDAP</sub> per config. (d) G<sub>E</sub>, G<sub>D</sub> and G<sub>A</sub>.",
             "pair", lambda s: rp.family_table(ed[s]["fc"], f"t_efc_{s}"))}
{fig_section("eatt", "Attention head: ranking by EDAP", "dse_attention.png",
             f"Attention head ranking: G_EDAP map over R and C, per-workload scores, G_EDAP per config, and G_E, G_D, G_A for {len(CONFIGS)} crossbar configs",
             "The same four panels for the attention head.",
             "pair", lambda s: rp.family_table(ed[s]["attention"], f"t_eatt_{s}"))}
{fig_section("par", "Pareto fronts per family", "dse_pareto.png",
             "Pareto fronts: G_D against G_E times G_A, for FC and the attention head",
             "x = G<sub>D</sub>, y = G<sub>E</sub> × G<sub>A</sub> (each value ÷ the worst on the workload, geometric means over the workloads). The diagonal lines are equal-EDAP lines, so their order is the ranking.",
             "stack")}
{fig_section("ecross", "EDAP across the families", "dse_cross.png",
             f"G_EDAP in FC on x, G_EDAP in attention on y, one dot per crossbar config",
             f"x = G<sub>EDAP</sub> in FC, y = G<sub>EDAP</sub> in attention, one dot per crossbar config; the closer to the origin, the better. Closest: old {first(ecross['old'], 'd')}, projected {first(ecross['proj'], 'd')}.",
             "pair", lambda s: rp.cross_table(ed[s]["cross"], f"t_ecross_{s}"))}
<div class="notes muted">
  <p>FC holds the Fmax and CLB counts of v2's old GEMM, which had an adder-tree array and an output serializer; the DSE run re-synthesizes the bit-plane GEMM. Old memory: the {len(diff)} attention builds of the d = 64 shapes at C = 64 use VPR's count, {min(over)}–{max(over)} blocks above the simulator's prediction; every other old build matches it.</p>
  <p>Projected rows: <code>v3/dse/results/projection/attn_priced_results.csv</code>, <code>v3/dse/results/projection/fc_priced_results.csv</code> (<code>v3/dse/projection.py</code>; its self-test reproduces today's priced rows from their inputs). Rankings and figures: <code>v3/dse/results/projection/edap/</code> and <code>adp/</code>, the old ones in <code>v3/dse/results/</code>.</p>
</div>
</div>
<script>{rp.SCRIPT}</script>
"""
    out = OUT / "dse_compare.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page)} bytes) and {2 * len(FIGURES)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
