#!/usr/bin/env python3
"""dataflow.py — the animated dataflow board of the brief (stages_brief.py): one attention head on NL-DPE crossbars,
Pre-bit-plane analysis, frozen 2026-10-23: it models the previous design (the 5-byte port, BUF = 40) at the
N_E = 2d mapping and describes the 2026-10-16 run; not part of run_all.sh. Port it after stage 4.
animated from the simulator's crossbar passes (dse/results/stages/stages.json, the DIAGRAMS arrays, `events`).

One board per array, the boards on one clock (play, speed, a scrubber), three panels and a time strip:
  projections   one token at a time: x_t is broadcast to the W_Q, W_K and W_V tiles, drawn to one scale with their
                weights inside the R x C array; each tile's input bar fills with the used rows during the load through
                its pre-bit-plane port, the weights light up for the 8 bit-slices, the output bar under the tile sweeps all
                C columns during the drain (the used ones coloured), and q_t, k_t, v_t leave for their stores
  stores        Q, K and V filling row by row (kept as logs, so the products are adds); the run's queries and the key
                and value block in use are outlined
  attention     one key block at a time, as matrices of their true shapes: S_b = Q K_b^T filling wave by wave, the
                softmax reading it at 5 scores per cycle and its exp passes, O_b = P_b V_b filling wave by wave, o
                folding the blocks (o <- alpha o + O_b), and O written at each run's end (O = o / l)
Every motion is a pass or a wave of the log; no FIFOs, credits, jobs or controller.

`boards(res)` is the compact data, `section(res)` the HTML. Self-test: python3 v3/dse/dataflow.py: the data, the
script under node with a stub DOM, and, with Google Chrome, a layout check in headless Chrome (every label inside its
board, no label on a label or on a drawn shape, at many times per head) and a planted fault it must catch.
`--shot HEAD FRACTION` writes a screenshot to v3/work/tmp/flowcheck/.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
from dse.stage_analysis import mapping  # noqa: E402

STAGES_DIR = V3 / "dse" / "results" / "stages"
CHECK_DIR = V3 / "work" / "tmp" / "flowcheck"
STEPS = {"convQ": "qk.a_in", "convK": "qk.b_in", "convV": "pv.b_in", "qk": "qk.jobs", "tile": "sm.tile", "exp": "sm.blk",
         "logl": "sm.run", "pv": "pv.jobs", "resc": "acc.blk", "norm": "acc.run"}
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
         'family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">')


def board(case: dict, cfg: str) -> dict:
    d, c = case["diagrams"][cfg], case["configs"][cfg]
    R, C = (int(x) for x in cfg.split("x"))
    m = mapping(case["n"], case["dm"], case["d"], R, C)
    e = d["events"]
    out = dict(cfg=cfg, R=R, C=C, I=min(R, C), S=c["S"], runs=c["runs"], blocks=c["blocks"], V=m["V"], P_PROJ=m["P_PROJ"],
               rowsPerTile=[min(R, case["dm"] - v * R) for v in range(m["V"])], N_E=m["N_E"], N_EP=m["N_EP"],
               N_EXP=m["N_EXP"], N_OC=m["N_OC"], crossbars={k: c["stages"][k]["crossbars"] for k in c["stages"]},
               T=d["T"], streamEnd=max(p[7] for p in e["proj"]),
               proj=e["proj"], qkW=e["qk"], pvW=e["pv"], smP=e["sm"], smR=e["smRetire"], smCarry=e["smCarry"],
               accW=e["acc"], accE=e["accE"], accL=e["accL"], epi=e["epi"])
    for key, src in STEPS.items():
        out[key] = [[int(a), int(b), int(u)] for a, b, u in d["steps"][src]]
    return out


def boards(res: dict) -> dict:
    cfgs = res["diagrams"]
    return dict(cfgs=cfgs, heads=[dict(name=case["name"], n=case["n"], dm=case["dm"], d=case["d"],
                                       bkv=case["n"] // case["configs"][cfgs[0]]["blocks"],
                                       arrays={cfg: board(case, cfg) for cfg in cfgs}) for case in res["cases"]])


CSS = """
.flow { display: grid; gap: 12px; }
.flowctl { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; font-size: 14px; }
.flowctl label { display: inline-flex; gap: 6px; align-items: center; }
.flowctl select { font: inherit; color: var(--fg); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 3px 6px; }
.flowctl input[type=range] { flex: 1 1 260px; accent-color: var(--accent); }
.flowclock { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 14ch; }
.flowboards { display: grid; grid-template-columns: 1fr; gap: 14px; }
.flowsvg { width: 100%; height: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; display: block; }
.ft { font-family: var(--body); font-size: 12px; fill: var(--fg); }
.fth { font-size: 15px; font-weight: 600; }
.ftp { font-size: 13px; font-weight: 600; fill: var(--muted); }
.ftl { font-size: 12.5px; font-weight: 600; }
.fts { font-size: 11.5px; fill: var(--muted); }
.ftm { font-family: var(--mono); font-size: 12px; }
.frame { fill: none; stroke: var(--rule); stroke-width: 1; }
.xbar { fill: var(--bg); stroke: var(--muted); stroke-width: 1; }
.wts { fill: #9ecae1; }
.wts.on { fill: #0072B2; }
.mat { fill: var(--bg); stroke: var(--muted); stroke-width: 1; }
.arrow { stroke: var(--rule); stroke-width: 2; fill: none; }
.opnode { fill: var(--bg); stroke: var(--muted); stroke-width: 1; }
.xnode { fill: var(--bg); stroke: var(--fg); stroke-width: 1.4; }
.fto { font-size: 11.5px; }
.ftn { font-size: 12.5px; }
.fte { font-size: 11.5px; fill: var(--muted); font-style: italic; }
.arrowhead { fill: var(--rule); }
.cursor { stroke: var(--fg); stroke-width: 1.2; }
"""

SCRIPT = r"""
(function () {
  const DATA = JSON.parse(document.getElementById("flowData").textContent);
  const COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#B8860B", "#7F7F7F"];
  const NS = "http://www.w3.org/2000/svg";
  const W = 1500, H = 800, SC = 5;              // SC: rows or columns of a crossbar per drawing unit (all tiles to one scale)
  const TOP = 98;                               // the top of the drawn arrays
  const GX0 = 104, GX1 = W - 16, GY0 = 676, LANE = 19;
  const LANES = ["projections", "QK^{⊤}", "softmax", "PV", "rescale"];
  const LANE_COLOR = ["#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"];
  // the attention panel's slots, [x0, width]: a matrix and its labels stay inside the slot
  const SLOT = {qk: [824, 140], sm: [978, 150], pv: [1142, 128], acc: [1284, 128], out: [1426, 58]};
  const fmt = n => Math.floor(n).toLocaleString("en-US");
  const el = (tag, attrs, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    parent.appendChild(e);
    return e;
  };
  // a label with real subscripts and superscripts: "S_{b} = Q K_{b}^{T}"
  const rich = (parent, attrs, spec) => {
    const t = el("text", attrs, parent);
    const re = /([_^])\{([^}]*)\}/g;
    let last = 0, m;
    while ((m = re.exec(spec)) !== null) {
      if (m.index > last) el("tspan", {}, t, spec.slice(last, m.index));
      el("tspan", {"baseline-shift": m[1] === "_" ? "sub" : "super", "font-size": "78%"}, t, m[2]);
      last = re.lastIndex;
    }
    if (last < spec.length) el("tspan", {}, t, spec.slice(last));
    return t;
  };
  const within = (lo, t, hi) => lo <= t && t <= hi;
  const active = (steps, t) => { for (const s of steps) if (s[0] <= t && t < s[1]) return s; return null; };
  const lastActive = (steps, t) => { let r = null; for (const s of steps) if (s[0] <= t && t < s[1]) r = s; return r; };
  const doneCount = (steps, t) => { let n = 0; for (const s of steps) if (s[1] <= t) n++; return n; };
  const wavesDone = (ws, t) => { let n = 0; for (const w of ws) if (w[3] <= t) n++; return n; };
  const waveProgress = (ws, t) => { let p = 0; for (const w of ws) p += w[3] <= t ? 1 : (w[0] <= t ? (t - w[0]) / Math.max(1, w[3] - w[0]) : 0); return p; };
  // the phase of the wave under way at t: the load through the port, the 8 bit-slices, the drain; null when none
  const phaseOf = (ws, t) => { for (const w of ws) { if (t < w[0]) break; if (t <= w[3]) return t <= w[1] ? "load" : (t < w[2] ? "slices" : "drain"); } return null; };
  const PHASE = {load: "adds of logs", slices: "EXP passes", drain: "adder trees"};
  const color = b => COLORS[b % COLORS.length];
  const frac = (a, b, t) => Math.max(0, Math.min(1, (t - a + 1) / Math.max(1, b - a + 1)));

  function arrow(svg, xa, ya, xb, yb) {
    const line = el("line", {x1: xa, y1: ya, x2: xb - 7, y2: yb, class: "arrow", "data-solid": "arrow"}, svg);
    const head = el("polygon", {points: `${xb - 8},${yb - 5} ${xb},${yb} ${xb - 8},${yb + 5}`, class: "arrowhead"}, svg);
    const dot = el("circle", {cx: xa, cy: ya, r: 4.5, visibility: "hidden"}, svg);
    return {line, head, dot, xa, ya, xb: xb - 7, yb};
  }
  function lightArrow(ar, p, c) {                 // p in [0, 1]: the packet's position; null: idle
    if (p === null) { ar.line.style.stroke = ""; ar.head.style.fill = ""; ar.dot.setAttribute("visibility", "hidden"); return; }
    ar.line.style.stroke = c; ar.head.style.fill = c;
    ar.dot.setAttribute("visibility", "visible"); ar.dot.setAttribute("fill", c);
    ar.dot.setAttribute("cx", ar.xa + (ar.xb - ar.xa) * p); ar.dot.setAttribute("cy", ar.ya + (ar.yb - ar.ya) * p);
  }
  function matrix(svg, x, y, w, h) {             // a matrix: its frame, a progress fill, an overlay fill
    const base = el("rect", {x, y, width: w, height: 0, fill: "#d9d9d9"}, svg);
    const fill = el("rect", {x, y, width: w, height: 0}, svg);
    el("rect", {x, y, width: w, height: h, class: "mat", "fill-opacity": 0, "data-solid": "matrix"}, svg);
    return {x, y, w, h, base, fill};
  }
  function setFill(r, y, h) { r.setAttribute("y", y); r.setAttribute("height", Math.max(0, h)); }

  function build(container, head, a, tmax) {
    container.innerHTML = "";
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, class: "flowsvg", role: "img",
                           "aria-label": `${head.name} at ${a.R} by ${a.C}: the dataflow of one head`}, container);
    const g = {};
    const d = head.d, n = head.n;
    el("text", {x: 16, y: 26, class: "ft fth"}, svg, `${head.name} at ${a.R} × ${a.C}`);
    g.clock = el("text", {x: W - 16, y: 26, class: "ft ftm", "text-anchor": "end"}, svg, "");

    // ---- the projections, one token at a time ----------------------------------------------------------------
    el("text", {x: 16, y: 56, class: "ft ftp"}, svg, "projections, one token at a time");
    // x_t, then one W_Q tile enlarged (its proportions true inside it); the W_K and W_V tiles do the same in lock-step
    const XX = 202, BX = 278, BY = TOP + 4, BW = 140, BH = 160;
    rich(svg, {x: XX - 2, y: 82, class: "ft ftl"}, "x_{t}");
    el("rect", {x: XX, y: BY, width: 12, height: BH, class: "xbar", "data-solid": "vector"}, svg);
    g.xFill = el("rect", {x: XX, y: BY, width: 12, height: 0, fill: "#0072B2"}, svg);
    el("text", {x: XX - 2, y: BY + BH + 16, class: "ft fts"}, svg, `${head.dm} values`);
    g.toTile = arrow(svg, XX + 18, BY + BH / 2, BX - 16, BY + BH / 2);
    rich(svg, {x: BX, y: 82, class: "ft ftl"}, "one W_{Q} tile, enlarged");
    const usedRows = a.rowsPerTile[0], uh = BH * usedRows / a.R, uw = BW * d / a.C;
    el("rect", {x: BX - 12, y: BY, width: 8, height: BH, class: "frame", "data-solid": "bar"}, svg);
    g.inFill = el("rect", {x: BX - 12, y: BY, width: 8, height: 0, fill: "#0072B2"}, svg);
    el("rect", {x: BX, y: BY, width: BW, height: BH, class: "xbar", "data-solid": "tile"}, svg);
    g.bigW = el("rect", {x: BX, y: BY, width: uw, height: uh, class: "wts", "data-solid": "weights"}, svg);
    el("rect", {x: BX, y: BY + BH + 4, width: BW, height: 9, class: "frame", "data-solid": "bar"}, svg);
    g.outUsed = el("rect", {x: BX, y: BY + BH + 4, width: 0, height: 9, fill: "#0072B2"}, svg);
    g.outRest = el("rect", {x: BX + uw, y: BY + BH + 4, width: 0, height: 9, fill: "#b0b0b0"}, svg);
    g.bits = [];
    for (let k = 0; k < 8; k++) g.bits.push(el("rect", {x: BX + k * 17.5, y: BY + BH + 20, width: 13, height: 8, class: "frame", "data-solid": "bar"}, svg));
    // the projected rows q_t, k_t, v_t
    const OX = BX + BW + 34;
    g.qkv = ["q", "k", "v"].map((lab, i) => {
      const y = BY + 30 + i * 46;
      rich(svg, {x: OX, y: y - 10, class: "ft fts"}, `${lab}_{t}`);
      el("rect", {x: OX, y, width: 34, height: 10, class: "frame", "data-solid": "vector"}, svg);
      return el("rect", {x: OX, y, width: 0, height: 10, fill: "#0072B2"}, svg);
    });
    g.toQkv = arrow(svg, BX + BW + 6, BY + BH / 2, OX - 4, BY + BH / 2);
    // every tile of the three projections, to one scale, lit in lock-step
    const MX = 16, MS = 16, mw = a.C / MS, mh = a.R / MS;
    el("text", {x: MX, y: 82, class: "ft ftl"}, svg, "all tiles, to one scale");
    g.mini = [];
    ["Q", "K", "V"].forEach((lab, i) => {
      const gx = MX + i * 58;
      rich(svg, {x: gx, y: TOP + 2, class: "ft fts"}, `W_{${lab}}`);
      for (let v = 0; v < a.V; v++) {
        const y = TOP + 12 + v * (mh + 3), used = a.rowsPerTile[v] / MS;
        el("rect", {x: gx, y, width: mw, height: mh, class: "xbar", "data-solid": "tile"}, svg);
        g.mini.push(el("rect", {x: gx, y, width: d / MS, height: used, class: "wts", "data-solid": "weights"}, svg));
      }
    });
    const miniBottom = TOP + 12 + a.V * (mh + 3);
    el("text", {x: MX, y: miniBottom + 16, class: "ft fts"}, svg, `${3 * a.V} tiles of ${a.R} × ${a.C}`);
    el("text", {x: MX, y: miniBottom + 32, class: "ft fts"}, svg, "one pre-bit-plane port each");
    // the lines under the projections
    const L1 = BY + BH + 50;
    el("text", {x: 16, y: L1, class: "ft fts"}, svg, `a tile holds ${usedRows} × ${d} weights in its ${a.R} × ${a.C} array`);
    el("text", {x: 16, y: L1 + 16, class: "ft fts"}, svg, `a pass: ${usedRows} rows in (${a.proj[0][1] - a.proj[0][0] + 1} cycles), ` +
       `8 bit-slices, all ${a.C} columns out (${a.proj[0][5] - a.proj[0][4] + 1} cycles)`);
    g.inText = el("text", {x: 16, y: L1 + 36, class: "ft"}, svg, "");
    g.outText = el("text", {x: 16, y: L1 + 54, class: "ft"}, svg, "");
    // the projected rows leave for their stores together (the three projections run in lock-step)
    g.toStores = arrow(svg, OX + 40, BY + BH / 2, 598, BY + BH / 2);

    // ---- the stores ------------------------------------------------------------------------------------------
    el("text", {x: 604, y: 56, class: "ft ftp"}, svg, "stores");
    const SX = {Q: 606, K: 660, V: 714}, SH = 200;
    g.stores = {};
    for (const k of ["Q", "K", "V"]) {
      el("text", {x: SX[k] + 16, y: 82, class: "ft ftl", "text-anchor": "middle"}, svg, k);
      g.stores[k] = {fill: el("rect", {x: SX[k], y: TOP, width: 32, height: 0, fill: k === "V" ? "#009E73" : "#56B4E9"}, svg)};
      el("rect", {x: SX[k], y: TOP, width: 32, height: SH, class: "mat", "fill-opacity": 0, "data-solid": "store"}, svg);
    }
    for (let b = 1; b < a.blocks; b++)
      for (const k of ["K", "V"]) el("line", {x1: SX[k], x2: SX[k] + 32, y1: TOP + SH * b / a.blocks, y2: TOP + SH * b / a.blocks, stroke: "#9a9a9a", "stroke-width": 0.6}, svg);
    for (let r = 1; r < a.runs; r++) el("line", {x1: SX.Q, x2: SX.Q + 32, y1: TOP + SH * r / a.runs, y2: TOP + SH * r / a.runs, stroke: "#555", "stroke-width": 1}, svg);
    g.qRun = el("rect", {x: SX.Q - 3, y: TOP, width: 38, height: SH / a.runs, fill: "none", stroke: "#222", "stroke-width": 2, visibility: "hidden"}, svg);
    g.kBlk = el("rect", {x: SX.K - 3, y: TOP, width: 38, height: SH / a.blocks, fill: "none", stroke: "#222", "stroke-width": 2, visibility: "hidden"}, svg);
    g.vBlk = el("rect", {x: SX.V - 3, y: TOP, width: 38, height: SH / a.blocks, fill: "none", stroke: "#222", "stroke-width": 2, visibility: "hidden"}, svg);
    const SL = TOP + SH + 24;
    el("text", {x: 604, y: SL, class: "ft fts"}, svg, `${n} rows of ${d} values`);
    el("text", {x: 604, y: SL + 16, class: "ft fts"}, svg, `K, V: ${a.blocks} blocks of ${head.bkv} rows`);
    el("text", {x: 604, y: SL + 32, class: "ft fts"}, svg, "stored as logs, so the");
    el("text", {x: 604, y: SL + 48, class: "ft fts"}, svg, "products become adds");
    g.storeText = el("text", {x: 604, y: SL + 66, class: "ft"}, svg, "");

    // ---- attention, one key block at a time ------------------------------------------------------------------
    el("text", {x: SLOT.qk[0], y: 56, class: "ft ftp"}, svg, "attention, one key block at a time");
    const ms = Math.min(200 / a.S, 128 / d), Hm = a.S * ms, my = TOP;   // this board's matrix scale
    g.S = matrix(svg, SLOT.qk[0], my, 16 * ms, Hm);
    g.P = matrix(svg, SLOT.sm[0], my, 16 * ms, Hm);
    g.ml = el("rect", {x: SLOT.sm[0] + 16 * ms + 6, y: my, width: 8, height: Hm, class: "mat", "data-solid": "matrix"}, svg);
    g.O = matrix(svg, SLOT.pv[0], my, d * ms, Hm);
    g.o = matrix(svg, SLOT.acc[0], my, d * ms, Hm);
    g.Out = matrix(svg, SLOT.out[0], my, 30, SH);
    rich(svg, {x: SLOT.qk[0], y: 82, class: "ft ftl"}, "S_{b} = Q K_{b}^{⊤}");
    rich(svg, {x: SLOT.sm[0], y: 82, class: "ft ftl"}, "P_{b} = exp(S_{b} − m)");
    rich(svg, {x: SLOT.pv[0], y: 82, class: "ft ftl"}, "O_{b} = P_{b} V_{b}");
    rich(svg, {x: SLOT.acc[0], y: 82, class: "ft ftl"}, "o ← α o + O_{b}");
    el("text", {x: SLOT.out[0], y: 82, class: "ft ftl"}, svg, "O");
    const AL = TOP + SH + 24, lines = (slot, texts) => texts.map((t, i) => el("text", {x: SLOT[slot][0], y: AL + 16 * i, class: "ft fts"}, svg, t));
    lines("qk", [`${a.S} × ${head.bkv} scores per block`, `${a.N_E} crossbars, ${a.qkW[0].length} wave${a.qkW[0].length > 1 ? "s" : ""}`, `a pass: ${a.I} products`]);
    lines("sm", ["reads 5 scores per cycle", `${a.N_EXP} crossbars for exp`, `${a.smP[0].length} exp passes per block`]);
    lines("pv", [`${a.S} × ${d} per block`, `${a.N_EP} crossbars, ${a.pvW[0].length} wave${a.pvW[0].length > 1 ? "s" : ""}`, `a pass: ${a.I} products`]);
    lines("acc", [`${a.S} × ${d}, all blocks`, `${2 * a.N_OC} crossbars`, `${a.accW[a.blocks > 1 ? 1 : 0].length} waves per block`]);
    lines("out", [`${n} rows`, "O = o / l"]);
    g.qkText = el("text", {x: SLOT.qk[0], y: AL + 54, class: "ft"}, svg, "");
    g.smText = el("text", {x: SLOT.sm[0], y: AL + 54, class: "ft"}, svg, "");
    g.smText2 = el("text", {x: SLOT.sm[0], y: AL + 72, class: "ft"}, svg, "");
    g.pvText = el("text", {x: SLOT.pv[0], y: AL + 54, class: "ft"}, svg, "");
    g.accText = el("text", {x: SLOT.acc[0], y: AL + 54, class: "ft"}, svg, "");
    g.accText2 = el("text", {x: SLOT.acc[0], y: AL + 72, class: "ft"}, svg, "");
    g.outText2 = el("text", {x: SLOT.out[0], y: AL + 54, class: "ft"}, svg, "");
    const midY = my + Math.min(Hm, SH) / 2;
    g.arrows = {inQK: arrow(svg, 754, midY, SLOT.qk[0] - 4, midY),
                qkSm: arrow(svg, SLOT.qk[0] + 16 * ms + 8, midY, SLOT.sm[0] - 4, midY),
                smPv: arrow(svg, SLOT.sm[0] + 16 * ms + 22, midY, SLOT.pv[0] - 4, midY),
                pvAcc: arrow(svg, SLOT.pv[0] + d * ms + 8, midY, SLOT.acc[0] - 4, midY),
                accOut: arrow(svg, SLOT.acc[0] + d * ms + 8, midY, SLOT.out[0] - 4, midY)};

    // ---- how a block is computed: the log-domain mapping ----------------------------------------------------
    const CY = 402, RL = CY + 48, RX = CY + 106, RN = CY + 164;      // the log band, the crossbar band, the linear band
    el("text", {x: 16, y: CY + 6, class: "ft ftp"}, svg, "how a block is computed: the log-domain mapping");
    const bandLabel = (y, l1, l2) => { el("text", {x: 16, y: y - 4, class: "ft ftl"}, svg, l1); el("text", {x: 16, y: y + 12, class: "ft fts"}, svg, l2); };
    bandLabel(RL, "log domain", "soft logic: + and \u2212 of logs");
    bandLabel(RX, "crossbars", "identity weights: EXP \u2193, LOG \u2191");
    bandLabel(RN, "linear domain", "soft logic: adder trees, max");
    for (const y of [RX - 29, RX + 29]) el("line", {x1: 186, x2: W - 16, y1: y, y2: y, stroke: "#c8c8c8", "stroke-width": 0.8, "stroke-dasharray": "4 4"}, svg);
    const ops = {};
    const node = (x, y, w, text) => {                                  // a soft-logic operation
      const r = el("rect", {x: x - w / 2, y: y - 14, width: w, height: 28, rx: 6, class: "opnode", "data-solid": "node"}, svg);
      rich(svg, {x, y: y + 4, class: "ft fto", "text-anchor": "middle", "data-inside": "node"}, text);
      return r;
    };
    const cross = (x, text, count) => {                                // a crossbar group converting between the domains
      const r = el("rect", {x: x - 34, y: RX - 13, width: 68, height: 26, rx: 3, class: "xnode", "data-solid": "node"}, svg);
      el("text", {x, y: RX + 4, class: "ft fto", "text-anchor": "middle", "data-inside": "node"}, svg, text);
      if (count) el("text", {x: x + 40, y: RX + 4, class: "ft fts"}, svg, count);
      return r;
    };
    const link = (pts) => {                                            // a polyline with a head on its last segment
      el("polyline", {points: pts.map(q => q.join(",")).join(" "), class: "arrow", "data-solid": "arrow"}, svg);
      const [x1, y1] = pts[pts.length - 2], [x2, y2] = pts[pts.length - 1];
      const dx = Math.sign(x2 - x1), dy = Math.sign(y2 - y1);
      const head = dx ? `${x2},${y2} ${x2 - 7 * dx},${y2 - 4} ${x2 - 7 * dx},${y2 + 4}` : `${x2},${y2} ${x2 - 4},${y2 - 7 * dy} ${x2 + 4},${y2 - 7 * dy}`;
      el("polygon", {points: head, class: "arrowhead"}, svg);
    };
    const up = x => { link([[x, RN - 14], [x, RX + 14]]); link([[x, RX - 13], [x, RL + 15]]); };
    const dn = x => { link([[x, RL + 14], [x, RX - 14]]); link([[x, RX + 13], [x, RN - 15]]); };
    // the operands: int8 rows through the LOG converters
    ops.opIn = node(250, RN, 96, "q_{t}, k_{t}, v_{t}"); ops.opLog = cross(250, "LOG \u2191", "1 per port");
    ops.opOut = node(250, RL, 118, "log q, log k, log v"); up(250);
    // QK: adds of logs, EXP, adder trees with the signs
    ops.qkAdd = node(420, RL, 112, "log q + log k"); ops.qkExp = cross(420, "EXP \u2193", `\u00d7${a.N_E}`);
    ops.qkSum = node(420, RN, 124, "\u03a3 over d, \u00b1 \u2192 s"); dn(420);
    link([[309, RL], [363, RL]]);
    // the softmax: the max and the subtract need no crossbar, a score is already a log
    ops.smMax = node(600, RN, 116, "max over 16 \u2192 m"); link([[482, RN], [541, RN]]);
    ops.smSub = node(600, RL, 118, "log P = s \u2212 m");
    el("line", {x1: 600, x2: 600, y1: RN - 14, y2: RL + 15, class: "arrow", "stroke-dasharray": "3 3", "data-solid": "arrow"}, svg);
    el("text", {x: 608, y: RX + 4, class: "ft fts"}, svg, "no crossbar");
    ops.smExp = cross(730, "EXP \u2193", `\u00d7${a.N_EXP}`); link([[659, RL], [730, RL], [730, RX - 14]]);
    ops.smSum = node(730, RN, 128, "\u03a3 over 16 \u2192 l"); link([[730, RX + 13], [730, RN - 15]]);
    el("text", {x: 730, y: RN + 28, class: "ft fts", "text-anchor": "middle"}, svg, "l \u2190 \u03b1 l + \u03a3 P");
    ops.smLog = cross(830, "LOG \u2191", "carry"); link([[794, RN], [830, RN], [830, RX + 14]]);
    ops.smLl = node(830, RL, 62, "log l"); link([[830, RX - 13], [830, RL + 15]]);
    // PV: log P from the softmax, log v from the store
    ops.pvAdd = node(960, RL, 118, "log P + log v"); ops.pvExp = cross(960, "EXP \u2193", `\u00d7${a.N_EP}`);
    ops.pvSum = node(960, RN, 134, "\u03a3 over 16 keys \u2192 O_{b}"); dn(960);
    link([[600, RL - 14], [600, RL - 24], [960, RL - 24], [960, RL - 15]]);
    el("text", {x: 780, y: RL - 28, class: "ft fts", "text-anchor": "middle"}, svg, "log P");
    // the rescale: alpha from the max change, o kept as a log between blocks
    ops.rsAdd = node(1090, RL, 102, "log \u03b1 + lo"); ops.rsExp = cross(1090, "EXP \u2193", `\u00d7${a.N_OC}`);
    ops.rsSum = node(1090, RN, 112, "\u00b1 \u03b1 o + O_{b}"); dn(1090); link([[1027, RN], [1033, RN]]);
    ops.rsLog = cross(1200, "LOG \u2191", `\u00d7${a.N_OC}`); link([[1146, RN], [1200, RN], [1200, RX + 14]]);
    ops.rsLo = node(1200, RL, 92, "lo = log o"); link([[1200, RX - 13], [1200, RL + 15]]);
    el("text", {x: 1200, y: RN + 28, class: "ft fts", "text-anchor": "middle"}, svg, "o stays a log between blocks");
    // the normalization at the run's end: the divide is a subtract
    ops.nmSub = node(1340, RL, 102, "lo \u2212 log l"); ops.nmExp = cross(1340, "EXP \u2193", "per run");
    ops.nmOut = node(1340, RN, 62, "O"); dn(1340); link([[1246, RL], [1288, RL]]);
    link([[830, RL - 14], [830, RL - 34], [1340, RL - 34], [1340, RL - 15]]);
    el("text", {x: 1085, y: RL - 38, class: "ft fts", "text-anchor": "middle"}, svg, "log l");
    // the counts of one block, from its shapes
    const S = a.S, bkv = head.bkv, I = a.I, wq = a.qkW[0].length, wp = a.pvW[0].length;
    el("text", {x: 16, y: RN + 52, class: "ft fts"}, svg,
       `one block: QK\u1D40 ${fmt(S * bkv * d)} adds of logs, ${fmt(S * bkv * d)} EXP (${wq} wave${wq > 1 ? "s" : ""} \u00d7 ${a.N_E} crossbars, ${I} per pass), ` +
       `${fmt(S * bkv * (d - 1))} tree adds \u2192 ${fmt(S * bkv)} scores \u00b7 softmax: ${fmt(S * bkv)} max, ${fmt(S * bkv)} subtracts, ` +
       `${fmt(S * bkv)} EXP (${a.smP[0].length} passes), ${fmt(S * (bkv - 1))} adds \u2192 l, then ${S} EXP + ${S} LOG for the carry`);
    el("text", {x: 16, y: RN + 68, class: "ft fts"}, svg,
       `PV: ${fmt(S * bkv * d)} adds of logs, ${fmt(S * bkv * d)} EXP (${wp} wave${wp > 1 ? "s" : ""} \u00d7 ${a.N_EP}), ${fmt(S * d * (bkv - 1))} tree adds \u2192 O\u1D47 (${S} \u00d7 ${d}) \u00b7 ` +
       `rescale: ${fmt(S * d)} adds of logs, ${fmt(S * d)} EXP, ${fmt(S * d)} adds, ${fmt(S * d)} LOG \u00b7 per run: ${fmt(S * d)} subtracts, ${fmt(S * d)} EXP \u2192 O`);
    g.ops = ops;

    // ---- the status line and the time diagram with a cursor -------------------------------------------------
    g.status = el("text", {x: 16, y: GY0 - 14, class: "ft fts"}, svg, "");
    const sx = t => GX0 + (GX1 - GX0) * t / tmax;
    const lanes = [[[a.proj[0][0], a.streamEnd]], a.qk.map(s => [s[0], s[1]]),
                   a.tile.map((s, i) => [s[0], a.exp[i] ? a.exp[i][1] : s[1]]).concat(a.logl.map(s => [s[0], s[1]])),
                   a.pv.map(s => [s[0], s[1]]), a.resc.map(s => [s[0], s[1]]).concat(a.norm.map(s => [s[0], s[1]]))];
    lanes.forEach((iv, i) => {
      const y = GY0 + i * LANE;
      rich(svg, {x: GX0 - 10, y: y + 14, class: "ft fts", "text-anchor": "end"}, LANES[i]);
      for (const [s, e] of iv) el("rect", {x: sx(s), y: y + 2, width: Math.max(1, sx(e) - sx(s)), height: LANE - 4, fill: LANE_COLOR[i], "data-solid": "lane"}, svg);
    });
    el("text", {x: GX0, y: GY0 + 5 * LANE + 16, class: "ft fts"}, svg, "0");
    el("text", {x: GX1, y: GY0 + 5 * LANE + 16, class: "ft fts", "text-anchor": "end"}, svg, `${fmt(tmax)} cycles`);
    g.cursor = el("line", {x1: GX0, x2: GX0, y1: GY0 - 2, y2: GY0 + 5 * LANE, class: "cursor"}, svg);
    g.sx = sx; g.Hm = Hm; g.SH = SH; g.BW = BW; g.BH = BH; g.uw = uw; g.uh = uh;
    return g;
  }

  function update(g, head, a, t) {
    const n = head.n, d = head.d, parts = [];
    g.clock.textContent = t >= a.T ? `done at cycle ${fmt(a.T)}` : `cycle ${fmt(t)} of ${fmt(a.T)}`;
    // the projections: the token loading (load to the last bit-slice), the token draining, the token leaving
    let ld = -1, dr = -1, ot = -1;
    a.proj.forEach((p, i) => {
      if (within(p[0], t, p[3])) ld = i;
      if (within(p[4], t, p[5])) dr = i;
      if (within(p[6], t, p[7])) ot = i;
    });
    const loading = ld >= 0 && t <= a.proj[ld][1], computing = ld >= 0 && t >= a.proj[ld][2];
    const lf = ld >= 0 ? frac(a.proj[ld][0], a.proj[ld][1], t) : 0;
    setFill(g.inFill, TOP + 4, ld >= 0 ? g.uh * lf : 0);
    setFill(g.xFill, TOP + 4, ld >= 0 ? g.BH * lf : 0);
    g.bigW.classList.toggle("on", computing);
    g.mini.forEach(w => w.classList.toggle("on", computing));
    const slice = computing ? t - a.proj[ld][2] : -1;
    g.bits.forEach((b, k) => { b.style.fill = k <= slice ? "#0072B2" : ""; });
    const q = dr >= 0 ? frac(a.proj[dr][4], a.proj[dr][5], t) * g.BW : 0;
    g.outUsed.setAttribute("width", Math.min(q, g.uw));
    g.outRest.setAttribute("width", Math.max(0, q - g.uw));
    g.qkv.forEach(r => r.setAttribute("width", ot >= 0 ? 34 * frac(a.proj[ot][6], a.proj[ot][7], t) : 0));
    lightArrow(g.toTile, loading ? lf : null, "#0072B2");
    lightArrow(g.toQkv, ot >= 0 ? frac(a.proj[ot][6], a.proj[ot][7], t) : null, "#0072B2");
    g.inText.textContent = ld < 0 ? "" : computing
      ? `token ${ld}: bit-slice ${slice + 1} of 8 in every tile`
      : `token ${ld}: loading its ${a.rowsPerTile[0]} rows into every tile`;
    g.outText.textContent = dr < 0 ? "" : a.C > d
      ? `token ${dr}: draining all ${a.C} columns of every tile, only ${d} hold its q, k, v`
      : `token ${dr}: draining the ${a.C} columns of every tile`;
    if (ld >= 0) parts.push(`projecting token ${ld}`);
    lightArrow(g.toStores, ot >= 0 ? frac(a.proj[ot][6], a.proj[ot][7], t) : null, "#0072B2");
    // the stores
    const nQ = doneCount(a.convQ, t), nK = doneCount(a.convK, t), nV = doneCount(a.convV, t);
    setFill(g.stores.Q.fill, TOP, g.SH * nQ / n);
    setFill(g.stores.K.fill, TOP, g.SH * nK / n);
    setFill(g.stores.V.fill, TOP, g.SH * nV / n);
    const qk = active(a.qk, t), pv = active(a.pv, t);
    if (qk) {
      const r = Math.floor(qk[2] / a.blocks), b = qk[2] % a.blocks;
      g.qRun.setAttribute("visibility", "visible"); g.qRun.setAttribute("y", TOP + g.SH * r / a.runs);
      g.kBlk.setAttribute("visibility", "visible"); g.kBlk.setAttribute("y", TOP + g.SH * b / a.blocks);
      g.kBlk.setAttribute("stroke", color(b)); g.qRun.setAttribute("stroke", color(b));
    } else { g.qRun.setAttribute("visibility", "hidden"); g.kBlk.setAttribute("visibility", "hidden"); }
    if (pv) {
      const b = pv[2] % a.blocks;
      g.vBlk.setAttribute("visibility", "visible"); g.vBlk.setAttribute("y", TOP + g.SH * b / a.blocks); g.vBlk.setAttribute("stroke", color(b));
    } else g.vBlk.setAttribute("visibility", "hidden");
    g.storeText.textContent = t < a.streamEnd ? `${nQ} of ${n} rows in` : "all rows in";
    // S_b = Q K_b^T: the job's waves fill the score tile
    if (qk) {
      const j = qk[2], b = j % a.blocks, ws = a.qkW[j], k = wavesDone(ws, t);
      g.S.fill.setAttribute("fill", color(b)); setFill(g.S.fill, g.S.y, g.Hm * waveProgress(ws, t) / ws.length);
      g.qkText.textContent = `block ${b}: wave ${Math.min(k + 1, ws.length)} of ${ws.length}`;
      const p = (t - qk[0]) / Math.max(1, (qk[1] - qk[0]) * 0.3);
      lightArrow(g.arrows.inQK, p < 1 ? p : null, color(b));
      parts.push(`scores of block ${b}`);
    } else { setFill(g.S.fill, g.S.y, 0); g.qkText.textContent = "idle"; lightArrow(g.arrows.inQK, null); }
    // the softmax: the reader sweeps the score tile; the exp passes and the row statistics m, l
    const tile = active(a.tile, t);
    if (tile) {
      const b = tile[2] % a.blocks, p = frac(tile[0], tile[1], t);
      g.P.fill.setAttribute("fill", color(b)); setFill(g.P.fill, g.P.y, g.Hm * p);
      g.smText.textContent = `block ${b}: reading`;
      lightArrow(g.arrows.qkSm, p, color(b));
      parts.push(`softmax reading block ${b}`);
    } else { setFill(g.P.fill, g.P.y, 0); g.smText.textContent = "idle"; lightArrow(g.arrows.qkSm, null); }
    const ex = lastActive(a.exp, t);
    if (ex) {
      const j = ex[2], ps = a.smP[j], k = wavesDone(ps, t);
      g.smText2.textContent = `block ${j % a.blocks}: exp ${Math.min(k + 1, ps.length)} of ${ps.length}`;
      g.ml.style.fill = within(a.smR[j] - 2, t, a.smR[j] + 24) ? color(j % a.blocks) : "";
    } else { g.smText2.textContent = ""; g.ml.style.fill = ""; }
    // O_b = P_b V_b: the job's waves fill the block's output
    if (pv) {
      const j = pv[2], b = j % a.blocks, ws = a.pvW[j], k = wavesDone(ws, t);
      g.O.fill.setAttribute("fill", color(b)); setFill(g.O.fill, g.O.y, g.Hm * waveProgress(ws, t) / ws.length);
      g.pvText.textContent = `block ${b}: wave ${Math.min(k + 1, ws.length)} of ${ws.length}`;
      const p = (t - pv[0]) / Math.max(1, (pv[1] - pv[0]) * 0.3);
      lightArrow(g.arrows.smPv, p < 1 ? p : null, color(b));
      parts.push(`PV of block ${b}`);
    } else { setFill(g.O.fill, g.O.y, 0); g.pvText.textContent = "idle"; lightArrow(g.arrows.smPv, null); }
    // o <- alpha o + O_b: the blocks folded in this run, the block in progress wave by wave
    const rs = lastActive(a.resc, t), nm = active(a.norm, t);
    const runNow = rs ? Math.floor(rs[2] / a.blocks) : (nm ? nm[2] : Math.min(a.runs - 1, doneCount(a.norm, t)));
    let folded = 0;
    for (let b = 0; b < a.blocks; b++) { const ws = a.accW[runNow * a.blocks + b]; if (ws && ws.length && ws[ws.length - 1][3] <= t) folded++; }
    g.o.base.setAttribute("fill", "#cfcfcf"); setFill(g.o.base, g.o.y, g.Hm * (nm || doneCount(a.norm, t) > runNow ? 1 : folded / a.blocks));
    if (rs) {
      const j = rs[2], b = j % a.blocks, ws = a.accW[j], k = wavesDone(ws, t);
      g.o.fill.setAttribute("fill", color(b)); setFill(g.o.fill, g.o.y, g.Hm * waveProgress(ws, t) / ws.length);
      g.accText.textContent = `block ${b}: wave ${Math.min(k + 1, ws.length)} of ${ws.length}`;
      const p = (t - rs[0]) / Math.max(1, (rs[1] - rs[0]) * 0.3);
      lightArrow(g.arrows.pvAcc, p < 1 ? p : null, color(b));
      parts.push(`folding block ${b}`);
    } else { setFill(g.o.fill, g.o.y, 0); g.accText.textContent = nm ? "" : "idle"; lightArrow(g.arrows.pvAcc, null); }
    g.accText2.textContent = `folded ${folded} of ${a.blocks}`;
    // O = o / l at a run's end
    const runsOut = doneCount(a.norm, t);
    let outRows = runsOut * a.S;
    if (nm) {
      const ws = a.epi[nm[2]];
      outRows += a.S * waveProgress(ws, t) / ws.length;
      g.outText2.textContent = `run ${nm[2]}`;
      lightArrow(g.arrows.accOut, frac(nm[0], nm[1], t), "#7A3E66");
      parts.push(`O = o / l for run ${nm[2]}`);
    } else { g.outText2.textContent = runsOut ? `${runsOut} of ${a.runs}` : ""; lightArrow(g.arrows.accOut, null); }
    g.Out.fill.setAttribute("fill", "#7A3E66"); setFill(g.Out.fill, g.Out.y, g.SH * outRows / n);
    // the log-domain chain: each operation lights in the phase of the pass that performs it
    const ops = g.ops;
    const lit = (r, on, c) => { r.style.fill = on ? c : ""; r.style.fillOpacity = on ? 0.3 : ""; r.style.stroke = on ? c : ""; r.style.strokeWidth = on ? 2 : ""; };
    const conv = active(a.convQ, t) || active(a.convK, t) || active(a.convV, t);
    for (const k of ["opIn", "opLog", "opOut"]) lit(ops[k], !!conv, "#0072B2");
    const qph = qk ? phaseOf(a.qkW[qk[2]], t) : null, qc = qk ? color(qk[2] % a.blocks) : "";
    lit(ops.qkAdd, qph === "load", qc); lit(ops.qkExp, qph === "slices", qc); lit(ops.qkSum, qph === "drain", qc);
    if (qph) parts[parts.findIndex(x => x.startsWith("scores of"))] += `: ${PHASE[qph]}`;
    const tc = tile ? color(tile[2] % a.blocks) : "";
    const eph = ex ? phaseOf(a.smP[ex[2]], t) : null, ec = ex ? color(ex[2] % a.blocks) : "";
    lit(ops.smMax, !!tile, tc); lit(ops.smSub, !!tile || eph === "load", tile ? tc : ec);
    lit(ops.smExp, eph === "slices", ec); lit(ops.smSum, eph === "drain", ec);
    const carry = ex ? within(a.smCarry[ex[2]][0], t, a.smCarry[ex[2]][1]) : false;
    lit(ops.smLog, carry, ec); lit(ops.smLl, carry, ec);
    if (eph) parts.push(`softmax block ${ex[2] % a.blocks}: ${eph === "load" ? "s \u2212 m" : eph === "slices" ? "EXP passes" : "row sums"}`);
    if (carry) parts.push(`carry: m, log l of block ${ex[2] % a.blocks}`);
    const pph = pv ? phaseOf(a.pvW[pv[2]], t) : null, pc = pv ? color(pv[2] % a.blocks) : "";
    lit(ops.pvAdd, pph === "load", pc); lit(ops.pvExp, pph === "slices", pc); lit(ops.pvSum, pph === "drain", pc);
    if (pph) parts[parts.findIndex(x => x.startsWith("PV of"))] += `: ${PHASE[pph]}`;
    const rE = rs ? phaseOf(a.accE[rs[2]], t) : null, rL = rs ? phaseOf(a.accL[rs[2]], t) : null, rc = rs ? color(rs[2] % a.blocks) : "";
    lit(ops.rsAdd, rE === "load", rc); lit(ops.rsExp, rE === "slices", rc); lit(ops.rsSum, rE === "drain" || rL === "load", rc);
    lit(ops.rsLog, rL === "slices", rc); lit(ops.rsLo, rL === "drain", rc);
    if (rE || rL) parts[parts.findIndex(x => x.startsWith("folding"))] += `: ${rE ? PHASE[rE] : rL === "slices" ? "LOG passes" : rL === "load" ? "adds" : "lo stored"}`;
    const nph = nm ? phaseOf(a.epi[nm[2]], t) : null;
    lit(ops.nmSub, nph === "load", "#7A3E66"); lit(ops.nmExp, nph === "slices", "#7A3E66"); lit(ops.nmOut, nph === "drain", "#7A3E66");
    g.status.textContent = t >= a.T ? "done" : (parts.length ? parts.join(" · ") : "waiting");
    g.cursor.setAttribute("x1", g.sx(Math.min(t, a.T))); g.cursor.setAttribute("x2", g.sx(Math.min(t, a.T)));
  }

  const headSel = document.getElementById("flowHead"), playBtn = document.getElementById("flowPlay");
  const speedSel = document.getElementById("flowSpeed"), slider = document.getElementById("flowT");
  const clock = document.getElementById("flowClock");
  const containers = [document.getElementById("flowA"), document.getElementById("flowB")];
  let head, arrays, boards = [], tmax = 1, t = 0, playing = false, last = null;

  function mount() {
    head = DATA.heads[parseInt(headSel.value, 10)];
    arrays = DATA.cfgs.map(c => head.arrays[c]);
    tmax = Math.max(...arrays.map(a => a.T));
    boards = arrays.map((a, i) => build(containers[i], head, a, tmax));
    slider.max = tmax; t = 0; playing = false; last = null; playBtn.textContent = "Play";
    const want = tmax / 25;                                  // about 25 seconds for the longer board
    let best = speedSel.options[0];
    for (const o of speedSel.options) if (Math.abs(Math.log(parseFloat(o.value) / want)) < Math.abs(Math.log(parseFloat(best.value) / want))) best = o;
    speedSel.value = best.value;
    render();
  }
  function render() {
    boards.forEach((g, i) => update(g, head, arrays[i], t));
    slider.value = t;
    clock.textContent = `cycle ${fmt(t)}`;
  }
  function frame(ts) {
    if (!playing) return;
    if (last !== null) t = Math.min(tmax, t + (ts - last) / 1000 * parseFloat(speedSel.value));
    last = ts;
    render();
    if (t >= tmax) { playing = false; playBtn.textContent = "Replay"; return; }
    requestAnimationFrame(frame);
  }
  playBtn.addEventListener("click", () => {
    if (playing) { playing = false; playBtn.textContent = "Play"; return; }
    if (t >= tmax) t = 0;
    playing = true; last = null; playBtn.textContent = "Pause";
    requestAnimationFrame(frame);
  });
  slider.addEventListener("input", () => { t = parseFloat(slider.value); render(); });
  headSel.addEventListener("change", mount);
  mount();
})();
"""


MAP_SCRIPT = r"""
(function () {
  const DATA = JSON.parse(document.getElementById("flowData").textContent);
  const COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#B8860B", "#7F7F7F"];
  const NS = "http://www.w3.org/2000/svg";
  const W = 1600, H = 520;
  const RL = 120, RX = 200, RN = 280;                        // the log band, the crossbar band, the linear band
  const GX0 = 104, GX1 = W - 16, GY0 = 400, LANE = 19;
  const LANES = ["projections", "QK^{⊤}", "softmax", "PV", "rescale"];
  const LANE_COLOR = ["#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"];
  const fmt = n => Math.floor(n).toLocaleString("en-US");
  const el = (tag, attrs, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    parent.appendChild(e);
    return e;
  };
  const rich = (parent, attrs, spec) => {
    const t = el("text", attrs, parent);
    const re = /([_^])\{([^}]*)\}/g;
    let last = 0, m;
    while ((m = re.exec(spec)) !== null) {
      if (m.index > last) el("tspan", {}, t, spec.slice(last, m.index));
      el("tspan", {"baseline-shift": m[1] === "_" ? "sub" : "super", "font-size": "78%"}, t, m[2]);
      last = re.lastIndex;
    }
    if (last < spec.length) el("tspan", {}, t, spec.slice(last));
    return t;
  };
  const within = (lo, t, hi) => lo <= t && t <= hi;
  const active = (steps, t) => { for (const s of steps) if (s[0] <= t && t < s[1]) return s; return null; };
  const lastActive = (steps, t) => { let r = null; for (const s of steps) if (s[0] <= t && t < s[1]) r = s; return r; };
  const doneCount = (steps, t) => { let n = 0; for (const s of steps) if (s[1] <= t) n++; return n; };
  const color = b => COLORS[b % COLORS.length];
  const frac = (a, b, t) => Math.max(0, Math.min(1, (t - a + 1) / Math.max(1, b - a + 1)));
  // the wave under way at t and how far into its phase: {phase, p}; the phases are the load through the port, the
  // 8 bit-slices and the drain; null when no wave runs
  const phaseOf = (ws, t) => {
    for (const w of ws) {
      if (t < w[0]) break;
      if (t <= w[3]) return t <= w[1] ? {phase: "load", p: frac(w[0], w[1], t)} : (t < w[2] ? {phase: "slices", p: frac(w[1] + 1, w[2] - 1, t)} : {phase: "drain", p: frac(w[2], w[3], t)});
    }
    return null;
  };

  function build(container, head, a, tmax) {
    container.innerHTML = "";
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, class: "flowsvg", role: "img",
                           "aria-label": `${head.name} at ${a.R} by ${a.C}: the log-domain mapping of one head in time`}, container);
    const g = {ops: {}, dots: {}, paths: {}};
    const ops = g.ops;
    el("text", {x: 16, y: 26, class: "ft fth"}, svg, `${head.name} at ${a.R} × ${a.C}`);
    g.clock = el("text", {x: W - 16, y: 26, class: "ft ftm", "text-anchor": "end"}, svg, "");
    const bandLabel = (y, l1, l2) => { el("text", {x: 16, y: y - 4, class: "ft ftl"}, svg, l1); el("text", {x: 16, y: y + 12, class: "ft fts"}, svg, l2); };
    bandLabel(RL, "log domain", "soft logic: + and − of logs");
    bandLabel(RX, "crossbars", "identity weights: EXP ↓, LOG ↑");
    bandLabel(RN, "linear domain", "soft logic: adder trees, max");
    for (const y of [RX - 33, RX + 33]) el("line", {x1: 156, x2: W - 16, y1: y, y2: y, stroke: "#c8c8c8", "stroke-width": 0.8, "stroke-dasharray": "4 4"}, svg);
    const node = (x, y, w, text) => {
      const r = el("rect", {x: x - w / 2, y: y - 15, width: w, height: 30, rx: 7, class: "opnode", "data-solid": "node"}, svg);
      rich(svg, {x, y: y + 4.5, class: "ft ftn", "text-anchor": "middle", "data-inside": "node"}, text);
      return r;
    };
    const cross = (x, text, count, below) => {               // a crossbar group converting between the domains, its count beside or below
      const r = el("rect", {x: x - 36, y: RX - 14, width: 72, height: 28, rx: 3, class: "xnode", "data-solid": "node"}, svg);
      el("text", {x, y: RX + 4.5, class: "ft ftn", "text-anchor": "middle", "data-inside": "node"}, svg, text);
      if (count && below) el("text", {x, y: RX + 27, class: "ft fts", "text-anchor": "middle"}, svg, count);
      else if (count) el("text", {x: x + 41, y: RX + 4.5, class: "ft fts"}, svg, count);
      return r;
    };
    const link = (pts, dashed) => {
      el("polyline", {points: pts.map(q => q.join(",")).join(" "), class: "arrow", "data-solid": "arrow", ...(dashed ? {"stroke-dasharray": "3 3"} : {})}, svg);
      const [x1, y1] = pts[pts.length - 2], [x2, y2] = pts[pts.length - 1];
      const dx = Math.sign(x2 - x1), dy = Math.sign(y2 - y1);
      const head = dx ? `${x2},${y2} ${x2 - 7 * dx},${y2 - 4} ${x2 - 7 * dx},${y2 + 4}` : `${x2},${y2} ${x2 - 4},${y2 - 7 * dy} ${x2 + 4},${y2 - 7 * dy}`;
      el("polygon", {points: head, class: "arrowhead"}, svg);
      return pts;
    };
    const lab = (x, y, text, anchor) => rich(svg, {x, y, class: "ft fte", "text-anchor": anchor || "start"}, text);
    const P = g.paths;
    // edge labels sit above a horizontal edge's row (clear of the node boxes) or right of a vertical edge
    const above = (x, row, text, sub) => lab(x, row - (sub ? 25 : 20), text, "middle");
    const beside = (x, rowA, rowB, text) => lab(x + 8, (rowA + rowB) / 2 + 5, text);
    const pp = a.P_PROJ, xb = a.crossbars;
    // the RTL modules: a frame around each module's columns, its instances and crossbars in the lane's colour
    const group = (x0, x1, name, detail, c) => {             // the frame, with the module's name and counts at its foot
      el("rect", {x: x0, y: RL - 24, width: x1 - x0, height: RN + 78 - (RL - 24), rx: 10, fill: "none", stroke: c, "stroke-width": 1.3, "stroke-dasharray": "6 4", opacity: 0.85}, svg);
      rich(svg, {x: x1 - 10, y: RN + 56, class: "ft ftl", fill: c, "text-anchor": "end"}, name);
      el("text", {x: x1 - 10, y: RN + 71, class: "ft fts", "text-anchor": "end"}, svg, detail);
    };
    group(156, 304, "Projection", `${3 * pp} GEMM${3 * pp > 1 ? "s" : ""}, ${xb.gemm} crossbars`, "#0072B2");
    group(314, 604, "QK^{⊤}", `1 DIMM tile, ${a.N_E} EXP + ${2 * pp} LOG`, "#56B4E9");
    group(614, 964, "online softmax", `1 instance, ${a.N_EXP} EXP + 2 carry`, "#E69F00");
    group(974, 1194, "PV", `1 DIMM tile, ${a.N_EP} EXP + ${pp} LOG`, "#009E73");
    group(1204, 1576, "rescale and normalize", `1 accumulator, ${a.N_OC} EXP + ${a.N_OC} LOG`, "#CC79A7");
    // the projections: a plain matrix-vector product on weight-stationary tiles; q, k go to the QK tile, v to the PV tile
    ops.proj = node(230, RN, 118, "x · W_{Q}, W_{K}, W_{V}");
    g.projFill = el("rect", {x: 171, y: RN + 20, width: 0, height: 5, fill: "#0072B2"}, svg);
    el("rect", {x: 171, y: RN + 20, width: 118, height: 5, class: "frame", "data-solid": "bar"}, svg);
    el("text", {x: 230, y: RN + 40, class: "ft fts", "text-anchor": "middle"}, svg, "rows projected");
    P.opIn = link([[289, RN], [372, RN], [372, RX + 15]]); above(330, RN, "q, k");
    ops.opLog = cross(372, "LOG ↑", `×${2 * pp}`); ops.opOut = node(372, RL, 96, "log q, log k");
    P.opUp = link([[372, RX - 14], [372, RL + 16]]);
    P.v = link([[300, RN], [300, RN + 90], [1010, RN + 90], [1010, RX + 15]]); lab(294, RN + 36, "v", "end");
    // QK^T: adds of logs, EXP, adder trees with the signs
    ops.qkAdd = node(530, RL, 124, "log q + log k"); ops.qkExp = cross(530, "EXP ↓", `×${a.N_E}`); ops.qkSum = node(530, RN, 132, "Σ over d, ± → s");
    P.qkIn = link([[420, RL], [467, RL]]); above(443, RL, "log q, log k");
    P.qkDown = link([[530, RL + 16], [530, RX - 15]]); beside(530, RL, RX, "log(q·k)");
    P.qkOut = link([[530, RX + 15], [530, RN - 16]]); beside(530, RX, RN, "q·k");
    // the softmax: a score is already a log, so the max and the subtract need no crossbar
    ops.smMax = node(690, RN, 124, "max over 16 → m"); P.s = link([[596, RN], [627, RN]]); above(611, RN, "s");
    ops.smSub = node(690, RL, 136, "s − m = log P"); P.smUp = link([[690, RN - 16], [690, RL + 16]], true);
    el("text", {x: 698, y: RX + 4.5, class: "ft fts"}, svg, "no crossbar");
    ops.smExp = cross(820, "EXP ↓", `×${a.N_EXP}`); P.smIn = link([[758, RL], [820, RL], [820, RX - 15]]); above(789, RL, "log P");
    ops.smSum = node(820, RN, 120, "Σ over 16 → l"); P.smOut = link([[820, RX + 15], [820, RN - 16]]); beside(820, RX, RN, "P");
    el("text", {x: 820, y: RN + 30, class: "ft fts", "text-anchor": "middle"}, svg, "l ← α l + Σ P");
    ops.smLog = cross(920, "LOG ↑"); P.carry = link([[880, RN], [920, RN], [920, RX + 15]]); above(900, RN, "l");
    ops.smLl = node(920, RL, 60, "log l"); P.carryUp = link([[920, RX - 14], [920, RL + 16]]);
    // PV: log P from the softmax, v through the tile's own LOG converter
    ops.pvLog = cross(1010, "LOG ↑", `×${pp}`); ops.pvV = node(1010, RL, 60, "log v"); P.pvUp = link([[1010, RX - 14], [1010, RL + 16]]);
    ops.pvAdd = node(1120, RL, 128, "log P + log v"); ops.pvExp = cross(1120, "EXP ↓", `×${a.N_EP}`); ops.pvSum = node(1120, RN, 140, "Σ over 16 keys → O_{b}");
    P.pvIn = link([[1040, RL], [1055, RL]]);
    P.pvDown = link([[1120, RL + 16], [1120, RX - 15]]); beside(1120, RL, RX, "log(P·v)");
    P.pvOut = link([[1120, RX + 15], [1120, RN - 16]]); beside(1120, RX, RN, "P·v");
    P.logP = link([[690, RL - 15], [690, RL - 34], [1120, RL - 34], [1120, RL - 16]]); lab(905, RL - 39, "log P", "middle");
    // the rescale: alpha from the change of the max; o is kept as a log between the blocks
    ops.rsAdd = node(1270, RL, 112, "log α + lo"); ops.rsExp = cross(1270, "EXP ↓", `×${a.N_OC}`); ops.rsSum = node(1270, RN, 122, "± α o + O_{b}");
    P.Ob = link([[1190, RN], [1208, RN]]);
    P.rsDown = link([[1270, RL + 16], [1270, RX - 15]]); beside(1270, RL, RX, "log(α·o)");
    P.rsOut = link([[1270, RX + 15], [1270, RN - 16]]); beside(1270, RX, RN, "α·o");
    ops.rsLog = cross(1380, "LOG ↑", `×${a.N_OC}`); P.o = link([[1331, RN], [1380, RN], [1380, RX + 15]]); above(1355, RN, "o");
    ops.rsLo = node(1380, RL, 92, "lo = log o"); P.oUp = link([[1380, RX - 14], [1380, RL + 16]]);
    el("text", {x: 1380, y: RN + 30, class: "ft fts", "text-anchor": "middle"}, svg, "o stays a log between blocks");
    P.alpha = link([[690, RL - 34], [690, RL - 56], [1270, RL - 56], [1270, RL - 16]]); lab(1000, RL - 61, "log α", "middle");
    // the normalization at a run's end: the divide is a subtract, on the accumulator's EXP group
    ops.nmSub = node(1510, RL, 104, "lo − log l"); ops.nmExp = cross(1510, "EXP ↓", "shared"); ops.nmOut = node(1510, RN, 70, "O");
    P.lo = link([[1426, RL], [1457, RL]]); above(1441, RL, "lo");
    P.nmDown = link([[1510, RL + 16], [1510, RX - 15]]); beside(1510, RL, RX, "log O");
    P.nmOut = link([[1510, RX + 15], [1510, RN - 16]]); beside(1510, RX, RN, "O");
    P.logL = link([[920, RL - 15], [920, RL - 78], [1510, RL - 78], [1510, RL - 16]]); lab(1215, RL - 83, "log l", "middle");
    // the packets
    for (const k of ["op", "v", "qk", "sm", "pv", "rs", "nm"]) g.dots[k] = el("circle", {cx: 0, cy: 0, r: 5.5, visibility: "hidden"}, svg);
    // the status line and the time diagram with a cursor
    g.status = el("text", {x: 16, y: GY0 - 14, class: "ft fts"}, svg, "");
    const sx = t => GX0 + (GX1 - GX0) * t / tmax;
    const lanes = [[[a.proj[0][0], a.streamEnd]], a.qk.map(s => [s[0], s[1]]),
                   a.tile.map((s, i) => [s[0], a.exp[i] ? a.exp[i][1] : s[1]]).concat(a.logl.map(s => [s[0], s[1]])),
                   a.pv.map(s => [s[0], s[1]]), a.resc.map(s => [s[0], s[1]]).concat(a.norm.map(s => [s[0], s[1]]))];
    lanes.forEach((iv, i) => {
      const y = GY0 + i * LANE;
      rich(svg, {x: GX0 - 10, y: y + 14, class: "ft fts", "text-anchor": "end"}, LANES[i]);
      for (const [s, e] of iv) el("rect", {x: sx(s), y: y + 2, width: Math.max(1, sx(e) - sx(s)), height: LANE - 4, fill: LANE_COLOR[i], "data-solid": "lane"}, svg);
    });
    el("text", {x: GX0, y: GY0 + 5 * LANE + 16, class: "ft fts"}, svg, "0");
    el("text", {x: GX1, y: GY0 + 5 * LANE + 16, class: "ft fts", "text-anchor": "end"}, svg, `${fmt(tmax)} cycles`);
    g.cursor = el("line", {x1: GX0, x2: GX0, y1: GY0 - 2, y2: GY0 + 5 * LANE, class: "cursor"}, svg);
    g.sx = sx;
    return g;
  }

  const lit = (r, on, c) => { r.style.fill = on ? c : ""; r.style.fillOpacity = on ? 0.3 : ""; r.style.stroke = on ? c : ""; r.style.strokeWidth = on ? 2 : ""; };
  function place(dot, pts, p, c) {                            // the packet at fraction p along a polyline
    if (p === null || pts === null) { dot.setAttribute("visibility", "hidden"); return; }
    let total = 0; const seg = [];
    for (let i = 1; i < pts.length; i++) { const L = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(L); total += L; }
    let d = Math.max(0, Math.min(1, p)) * total, x = pts[0][0], y = pts[0][1];
    for (let i = 1; i < pts.length; i++) {
      if (d <= seg[i - 1] || i === pts.length - 1) { const f = seg[i - 1] ? Math.min(1, d / seg[i - 1]) : 1; x = pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f; y = pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f; break; }
      d -= seg[i - 1];
    }
    dot.setAttribute("visibility", "visible"); dot.setAttribute("fill", c); dot.setAttribute("cx", x); dot.setAttribute("cy", y);
  }
  const where = (a, j) => a.runs > 1 ? `run ${Math.floor(j / a.blocks)}, block ${j % a.blocks}` : `block ${j % a.blocks}`;

  function update(g, head, a, t) {
    const ops = g.ops, P = g.paths, D = g.dots, parts = [];
    g.clock.textContent = t >= a.T ? `done at cycle ${fmt(a.T)}` : `cycle ${fmt(t)} of ${fmt(a.T)}`;
    // the projections and the converters
    g.projFill.setAttribute("width", 128 * doneCount(a.proj, t) / head.n);
    const projOn = active(a.proj, t) !== null;
    lit(ops.proj, projOn, "#0072B2");
    const conv = active(a.convQ, t) || active(a.convK, t), convV = active(a.convV, t);
    lit(ops.opLog, !!conv, "#0072B2"); lit(ops.opOut, !!conv, "#0072B2");
    place(D.op, conv ? P.opIn.concat(P.opUp) : null, conv ? frac(conv[0], conv[1], t) : null, "#0072B2");
    lit(ops.pvLog, !!convV, "#009E73"); lit(ops.pvV, !!convV, "#009E73");
    place(D.v, convV ? P.v.concat(P.pvUp) : null, convV ? frac(convV[0], convV[1], t) : null, "#009E73");
    if (projOn) parts.push("projecting");
    // QK^T
    const qk = active(a.qk, t), tile = active(a.tile, t);
    const qph = qk ? phaseOf(a.qkW[qk[2]], t) : null, qc = qk ? color(qk[2] % a.blocks) : "";
    lit(ops.qkAdd, !!qph && qph.phase === "load", qc); lit(ops.qkExp, !!qph && qph.phase === "slices", qc); lit(ops.qkSum, !!qph && qph.phase === "drain", qc);
    if (qph) place(D.qk, qph.phase === "load" ? P.qkIn : qph.phase === "slices" ? P.qkDown : P.qkOut, qph.p, qc);
    else if (tile && t < tile[0] + 0.3 * (tile[1] - tile[0])) place(D.qk, P.s, (t - tile[0]) / (0.3 * (tile[1] - tile[0])), color(tile[2] % a.blocks));
    else place(D.qk, null, null);
    if (qk) parts.push(`QKᵀ ${where(a, qk[2])}`);
    // the softmax
    const tc = tile ? color(tile[2] % a.blocks) : "", ex = lastActive(a.exp, t);
    const eph = ex ? phaseOf(a.smP[ex[2]], t) : null, ec = ex ? color(ex[2] % a.blocks) : "";
    lit(ops.smMax, !!tile, tc); lit(ops.smSub, !!tile || (!!eph && eph.phase === "load"), tile ? tc : ec);
    lit(ops.smExp, !!eph && eph.phase === "slices", ec); lit(ops.smSum, !!eph && eph.phase === "drain", ec);
    const carry = ex ? within(a.smCarry[ex[2]][0], t, a.smCarry[ex[2]][1]) : false;
    lit(ops.smLog, carry, ec); lit(ops.smLl, carry, ec);
    if (eph) place(D.sm, eph.phase === "load" ? P.smIn : eph.phase === "slices" ? null : P.smOut, eph.phase === "slices" ? null : eph.p, ec);
    else if (carry) place(D.sm, P.carry.concat(P.carryUp), frac(a.smCarry[ex[2]][0], a.smCarry[ex[2]][1], t), ec);
    else if (tile) place(D.sm, P.smUp, frac(tile[0], tile[1], t), tc);
    else place(D.sm, null, null);
    if (tile) parts.push(`softmax ${where(a, tile[2])}`); else if (ex) parts.push(`softmax ${where(a, ex[2])}`);
    // PV
    const pv = active(a.pv, t);
    const pph = pv ? phaseOf(a.pvW[pv[2]], t) : null, pc = pv ? color(pv[2] % a.blocks) : "";
    lit(ops.pvAdd, !!pph && pph.phase === "load", pc); lit(ops.pvExp, !!pph && pph.phase === "slices", pc); lit(ops.pvSum, !!pph && pph.phase === "drain", pc);
    const pvLead = pv && t < pv[0] + 0.25 * (pv[1] - pv[0]);
    if (pvLead) place(D.pv, P.logP, (t - pv[0]) / (0.25 * (pv[1] - pv[0])), pc);
    else if (pph) place(D.pv, pph.phase === "load" ? P.pvIn : pph.phase === "slices" ? P.pvDown : P.pvOut, pph.p, pc);
    else place(D.pv, null, null);
    if (pv) parts.push(`PV ${where(a, pv[2])}`);
    // the rescale
    const rs = lastActive(a.resc, t), nm = active(a.norm, t);
    const rE = rs ? phaseOf(a.accE[rs[2]], t) : null, rL = rs ? phaseOf(a.accL[rs[2]], t) : null, rc = rs ? color(rs[2] % a.blocks) : "";
    lit(ops.rsAdd, !!rE && rE.phase === "load", rc); lit(ops.rsExp, !!rE && rE.phase === "slices", rc);
    lit(ops.rsSum, (!!rE && rE.phase === "drain") || (!!rL && rL.phase === "load"), rc);
    lit(ops.rsLog, !!rL && rL.phase === "slices", rc); lit(ops.rsLo, !!rL && rL.phase === "drain", rc);
    if (rE) place(D.rs, rE.phase === "load" ? P.alpha : rE.phase === "slices" ? P.rsDown : P.rsOut, rE.p, rc);
    else if (rL) place(D.rs, rL.phase === "load" ? P.Ob.concat([[1270, RN]]).concat(P.o) : rL.phase === "slices" ? null : P.oUp, rL.phase === "slices" ? null : rL.p, rc);
    else place(D.rs, null, null);
    if (rs) parts.push(`rescale ${where(a, rs[2])}`);
    // the normalization
    const nph = nm ? phaseOf(a.epi[nm[2]], t) : null;
    lit(ops.nmSub, !!nph && nph.phase === "load", "#7A3E66"); lit(ops.nmExp, !!nph && nph.phase === "slices", "#7A3E66"); lit(ops.nmOut, !!nph && nph.phase === "drain", "#7A3E66");
    if (nph) place(D.nm, nph.phase === "load" ? P.logL : nph.phase === "slices" ? P.nmDown : P.nmOut, nph.p, "#7A3E66");
    else place(D.nm, null, null);
    if (nm) parts.push(a.runs > 1 ? `O = o / l, run ${nm[2]}` : "O = o / l");
    g.status.textContent = t >= a.T ? "done" : (parts.length ? parts.join(" · ") : "waiting");
    g.cursor.setAttribute("x1", g.sx(Math.min(t, a.T))); g.cursor.setAttribute("x2", g.sx(Math.min(t, a.T)));
  }

  const headSel = document.getElementById("flowHead"), playBtn = document.getElementById("flowPlay");
  const speedSel = document.getElementById("flowSpeed"), slider = document.getElementById("flowT");
  const clock = document.getElementById("flowClock");
  const containers = [document.getElementById("flowA"), document.getElementById("flowB")];
  let head, arrays, boards = [], tmax = 1, t = 0, playing = false, last = null;
  function mount() {
    head = DATA.heads[parseInt(headSel.value, 10)];
    arrays = DATA.cfgs.map(c => head.arrays[c]);
    tmax = Math.max(...arrays.map(a => a.T));
    boards = arrays.map((a, i) => build(containers[i], head, a, tmax));
    slider.max = tmax; t = 0; playing = false; last = null; playBtn.textContent = "Play";
    const want = tmax / 25;
    let best = speedSel.options[0];
    for (const o of speedSel.options) if (Math.abs(Math.log(parseFloat(o.value) / want)) < Math.abs(Math.log(parseFloat(best.value) / want))) best = o;
    speedSel.value = best.value;
    render();
  }
  function render() { boards.forEach((g, i) => update(g, head, arrays[i], t)); slider.value = t; clock.textContent = `cycle ${fmt(t)}`; }
  function frame(ts) {
    if (!playing) return;
    if (last !== null) t = Math.min(tmax, t + (ts - last) / 1000 * parseFloat(speedSel.value));
    last = ts;
    render();
    if (t >= tmax) { playing = false; playBtn.textContent = "Replay"; return; }
    requestAnimationFrame(frame);
  }
  playBtn.addEventListener("click", () => {
    if (playing) { playing = false; playBtn.textContent = "Play"; return; }
    if (t >= tmax) t = 0;
    playing = true; last = null; playBtn.textContent = "Pause";
    requestAnimationFrame(frame);
  });
  slider.addEventListener("input", () => { t = parseFloat(slider.value); render(); });
  headSel.addEventListener("change", mount);
  mount();
})();
"""


def controls(options: str) -> str:
    return f"""    <div class="flowctl">
      <label>head <select id="flowHead">{options}</select></label>
      <button class="zoom" id="flowPlay" type="button">Play</button>
      <label>speed <select id="flowSpeed">
        <option value="10">10 cycles per second</option><option value="40">40 cycles per second</option>
        <option value="150">150 cycles per second</option><option value="600">600 cycles per second</option>
        <option value="2400">2,400 cycles per second</option>
      </select></label>
      <input type="range" id="flowT" min="0" max="1" step="1" value="0" aria-label="cycle">
      <span class="flowclock" id="flowClock">cycle 0</span>
    </div>"""


def map_section(res: dict, script: str | None = None) -> str:
    """The mapping-first animation (the time page): the log-domain chain with labelled edges and moving block
    packets, both arrays on one clock, the time diagram with a cursor under each board."""
    data = boards(res)
    options = "".join(f'<option value="{i}">{h["name"]}</option>' for i, h in enumerate(data["heads"]))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f"""<section aria-labelledby="h_map">
  <h2 id="h_map">The log-domain mapping, in time</h2>
  <div class="flow">
{controls(options)}
    <div class="flowboards"><div id="flowA"></div><div id="flowB"></div></div>
  </div>
  <script type="application/json" id="flowData">{payload}</script>
  <script>{script or MAP_SCRIPT}</script>
</section>
"""


def section(res: dict, script: str | None = None) -> str:
    """The section's HTML; `script` replaces the board script (the self-test plants a fault through it)."""
    data = boards(res)
    options = "".join(f'<option value="{i}">{h["name"]}</option>' for i, h in enumerate(data["heads"]))
    cfgs = " and ".join(c.replace("x", " × ") for c in data["cfgs"])
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f"""<section aria-labelledby="h_flow">
  <h2 id="h_flow">2. The head in time</h2>
  <p>One head, animated from the simulator's crossbar passes. <b>Projections:</b> each token x<sub>t</sub> goes to every W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub> tile at once. The enlarged tile shows one pass: the bar on its left fills as the token's rows load through the tile's pre-bit-plane port, the weights light up for the 8 bit-slices, and the bar underneath sweeps all the columns as the result drains out, the useful ones in blue. All the tiles, drawn beside it to one scale, do the same in lock-step. q<sub>t</sub>, k<sub>t</sub> and v<sub>t</sub> then go to the stores. <b>Attention:</b> each key block of 16 runs through four steps, drawn as matrices of their true shapes, each board to its own scale: the scores S<sub>b</sub>, the softmax reading them at 5 per cycle, the block's output O<sub>b</sub>, and the running output o. O is written at the end of each run. <b>How a block is computed:</b> the bottom panel is the log-domain mapping. Every product is an add of two logs in soft logic; the identity crossbars only convert, EXP down into the linear domain and LOG up out of it; the sums are adder trees in the linear domain. A score is already a log, so the softmax needs no conversion before its EXP, and the divide at the end is a subtract. Each operation lights in the phase of the pass that performs it: the adds during the load, the conversion during the 8 bit-slices, the trees during the drain. The two arrays, {cfgs}, run on one clock.</p>
  <div class="flow">
    <div class="flowctl">
      <label>head <select id="flowHead">{options}</select></label>
      <button class="zoom" id="flowPlay" type="button">Play</button>
      <label>speed <select id="flowSpeed">
        <option value="10">10 cycles per second</option><option value="40">40 cycles per second</option>
        <option value="150">150 cycles per second</option><option value="600">600 cycles per second</option>
        <option value="2400">2,400 cycles per second</option>
      </select></label>
      <input type="range" id="flowT" min="0" max="1" step="1" value="0" aria-label="cycle">
      <span class="flowclock" id="flowClock">cycle 0</span>
    </div>
    <div class="flowboards"><div id="flowA"></div><div id="flowB"></div></div>
  </div>
  <script type="application/json" id="flowData">{payload}</script>
  <script>{script or SCRIPT}</script>
</section>
"""


HARNESS = r"""
const payload = __DATA__;
const handlers = {};
function mkEl(tag) {
  return {tag, attrs: {}, children: [], style: {}, classList: {toggle() {}}, textContent: "",
          setAttribute(k, v) { this.attrs[k] = v; }, appendChild(c) { this.children.push(c); }};
}
const elems = {};
function mkDom(id) {
  const e = mkEl("div"); e.id = id; e.value = "0"; e.max = "1"; e.innerHTML = "";
  e.options = [{value: "10"}, {value: "40"}, {value: "150"}, {value: "600"}, {value: "2400"}];
  e.addEventListener = (ev, fn) => { handlers[id + ":" + ev] = fn; };
  if (id === "flowData") e.textContent = JSON.stringify(payload);
  return e;
}
const document = {getElementById: id => elems[id] || (elems[id] = mkDom(id)), createElementNS: (ns, tag) => mkEl(tag)};
const requestAnimationFrame = fn => 0;
__SCRIPT__
function walk(e, f) { f(e); for (const c of e.children || []) walk(c, f); }
for (const hv of ["0", "1"]) {
  elems.flowHead.value = hv; handlers["flowHead:change"]();
  const tmax = parseFloat(elems.flowT.max);
  if (!(tmax > 1)) throw new Error("slider max not set: " + elems.flowT.max);
  for (let k = 0; k <= 200; k++) {
    elems.flowT.value = String(Math.floor(tmax * k / 200)); handlers["flowT:input"]();
    if (!/cycle/.test(elems.flowClock.textContent)) throw new Error("clock not rendered");
    for (const id of ["flowA", "flowB"]) walk(elems[id], e => {
      for (const key of ["x", "y", "width", "height", "cx", "cy", "x1", "x2", "y1", "y2"])
        if (key in e.attrs && !Number.isFinite(Number(e.attrs[key]))) throw new Error(`${e.tag}.${key} = ${e.attrs[key]} at ${k}/200`);
      if ("height" in e.attrs && Number(e.attrs.height) < 0) throw new Error(`${e.tag} negative height`);
    });
  }
  handlers["flowPlay:click"](); handlers["flowPlay:click"]();
}
console.log("runtime ok");
"""

# the layout check run in headless Chrome: every label inside its board, no label on another label or on a drawn shape
LAYOUT = r"""
(async function () {
  try { await document.fonts.ready; } catch (e) {}
  const issues = [], seen = new Set();
  const box = e => { const b = e.getBBox(); return e.tagName === "line" ? {x: b.x - 1.5, y: b.y - 1.5, width: b.width + 3, height: b.height + 3} : b; };
  const boxes = e => {                                         // a polyline: one thin box per segment
    if (e.tagName !== "polyline") return [box(e)];
    const pts = e.getAttribute("points").trim().split(/\s+/).map(q => q.split(",").map(Number)), out = [];
    for (let i = 1; i < pts.length; i++) {
      const [x1, y1] = pts[i - 1], [x2, y2] = pts[i];
      out.push({x: Math.min(x1, x2) - 1.5, y: Math.min(y1, y2) - 1.5, width: Math.abs(x2 - x1) + 3, height: Math.abs(y2 - y1) + 3});
    }
    return out;
  };
  const over = (a, c) => Math.min(a.x + a.width, c.x + c.width) - Math.max(a.x, c.x) > 1 &&
                         Math.min(a.y + a.height, c.y + c.height) - Math.max(a.y, c.y) > 1;
  const inside = (a, c) => a.x >= c.x - 1 && a.y >= c.y - 1 && a.x + a.width <= c.x + c.width + 1 && a.y + a.height <= c.y + c.height + 1;
  const push = (o) => { const k = JSON.stringify([o.kind, o.board.split(" at ")[0], o.text]); if (!seen.has(k)) { seen.add(k); issues.push(o); } };
  const R = b => [b.x, b.y, b.width, b.height].map(v => Math.round(v * 10) / 10);
  const hs = document.getElementById("flowHead"), sl = document.getElementById("flowT");
  for (let h = 0; h < hs.options.length; h++) {
    hs.value = String(h); hs.dispatchEvent(new Event("change"));
    const tmax = parseFloat(sl.max);
    for (let k = 0; k <= 60; k++) {
      sl.value = String(Math.floor(tmax * k / 60)); sl.dispatchEvent(new Event("input"));
      document.querySelectorAll("svg.flowsvg").forEach((svg, bi) => {
        const vb = svg.viewBox.baseVal;
        const texts = [...svg.querySelectorAll("text")].filter(t => t.textContent.trim()).map(t => ({t, b: box(t)}));
        const solids = [...svg.querySelectorAll("[data-solid]")].flatMap(s => boxes(s).map(b => ({s, b})));
        const where = `head ${h} board ${bi} at ${k}/60`;
        for (const {t, b} of texts)
          if (b.x < vb.x - 0.5 || b.y < vb.y - 0.5 || b.x + b.width > vb.x + vb.width + 0.5 || b.y + b.height > vb.y + vb.height + 0.5)
            push({kind: "outside the board", board: where, text: t.textContent});
        for (let i = 0; i < texts.length; i++)
          for (let j = i + 1; j < texts.length; j++)
            if (over(texts[i].b, texts[j].b)) push({kind: "label on label", board: where, text: texts[i].t.textContent + " | " + texts[j].t.textContent,
                                                    detail: `${R(texts[i].b)} vs ${R(texts[j].b)}`});
        for (const {t, b} of texts) for (const {s, b: c} of solids)
          if (over(b, c) && !(t.getAttribute("data-inside") === "node" && s.getAttribute("data-solid") === "node" && inside(b, c)))
            push({kind: "label on " + s.getAttribute("data-solid"), board: where, text: t.textContent,
                  detail: `label ${R(b)} vs ${s.tagName} ${R(c)} ${s.getAttribute("points") || ""}`});   // a node's own label may sit inside it
        for (const {s, b: c} of solids)                      // shapes inside the board
          if (c.x < vb.x - 0.5 || c.x + c.width > vb.x + vb.width + 0.5 || c.y + c.height > vb.y + vb.height + 0.5)
            push({kind: s.getAttribute("data-solid") + " outside the board", board: where, text: ""});
      });
    }
  }
  const pre = document.createElement("pre"); pre.id = "report";
  pre.textContent = JSON.stringify({font: document.fonts.check('12px "IBM Plex Sans"'), issues});
  document.body.appendChild(pre);
})();
"""


def chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
        path = shutil.which(name)
        if path:
            return path
    return None


def check_page(res: dict, extra: str, script: str | None = None, which=None) -> str:
    from dse.results_page import CSS as PAGE_CSS
    make = which or section
    return (f'<!doctype html><html lang="en"><meta charset="utf-8">{FONTS}<style>{PAGE_CSS}{CSS}</style>'
            f'<body><div class="wrap">{make(res, script)}</div><script>{extra}</script></body></html>')


def run_chrome(page: Path, extra_args: list, timeout: int = 300) -> subprocess.CompletedProcess:
    exe = chrome()
    profile = CHECK_DIR / "profile"
    return subprocess.run([exe, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                           f"--user-data-dir={profile}", "--window-size=1600,1100", "--virtual-time-budget=60000",
                           *extra_args, page.as_uri()], capture_output=True, text=True, timeout=timeout)


def layout_check(res: dict, script: str | None = None, which=None) -> dict:
    """Run LAYOUT in headless Chrome on a board (`which`: section or map_section; `script` replaces its script);
    returns {"font": bool, "issues": [...]}."""
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    page = CHECK_DIR / "layout.html"
    page.write_text(check_page(res, LAYOUT, script, which), encoding="utf-8")
    r = run_chrome(page, ["--dump-dom"])
    m = re.search(r'<pre id="report">(.*?)</pre>', r.stdout, re.S)
    if not m:
        raise RuntimeError(f"no layout report from Chrome (exit {r.returncode}): {r.stderr[-400:]}")
    return json.loads(html.unescape(m.group(1)))


def screenshot(res: dict, head: int, frac: float, which=None) -> Path:
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    page = CHECK_DIR / f"shot_{head}.html"
    page.write_text(check_page(res, f"""
const hs = document.getElementById("flowHead"); hs.value = "{head}"; hs.dispatchEvent(new Event("change"));
const s = document.getElementById("flowT"); s.value = String(Math.floor(parseFloat(s.max) * {frac})); s.dispatchEvent(new Event("input"));
""", None, which), encoding="utf-8")
    out = CHECK_DIR / f"shot_{'map_' if which is map_section else ''}{head}_{int(round(frac * 1000)):03d}.png"
    run_chrome(page, ["--window-size=1600,1500", f"--screenshot={out}"])
    return out


def selfTest() -> int:
    res = json.loads((STAGES_DIR / "stages.json").read_text())
    data = boards(res)
    n = 0
    for h in data["heads"]:
        for cfg, a in h["arrays"].items():
            for case in res["cases"]:
                if case["name"] == h["name"]:
                    assert case["diagrams"][cfg]["events"]["projSame"], (h["name"], cfg)   # K, V tiles in lock-step with Q
            n += 1
            assert len(a["proj"]) == h["n"] and all(len(p) == 8 for p in a["proj"]); n += 1
            assert all(p[0] <= p[1] < p[2] <= p[3] < p[4] <= p[5] and p[4] <= p[6] <= p[7] for p in a["proj"]), (h["name"], cfg)
            n += 1                                                    # load, then compute, then drain; the row leaves during the drain
            assert all(q[0] > p[3] for p, q in zip(a["proj"], a["proj"][1:])); n += 1   # the next load after the last bit-slice
            assert a["proj"][0][1] - a["proj"][0][0] + 1 == -(-8 * a["rowsPerTile"][0] // 40); n += 1   # 5 values per cycle in
            assert a["proj"][0][5] - a["proj"][0][4] + 1 == -(-8 * a["C"] // 40); n += 1          # all C columns out
            jobs = a["runs"] * a["blocks"]
            assert len(a["qkW"]) == len(a["pvW"]) == len(a["smP"]) == len(a["accW"]) == len(a["accE"]) == len(a["accL"]) == jobs; n += 1
            assert all(len(a["accE"][j]) == (0 if j % a["blocks"] == 0 else len(a["accL"][j])) for j in range(jobs)); n += 1  # block 0: no rescale
            assert all(c[0] <= c[1] for c in a["smCarry"]) and len(a["smCarry"]) == jobs; n += 1
            assert len(a["qk"]) == len(a["tile"]) == len(a["exp"]) == len(a["pv"]) == len(a["resc"]) == jobs; n += 1
            assert len(a["epi"]) == len(a["norm"]) == len(a["logl"]) == a["runs"]; n += 1
            for steps, waves in ((a["qk"], a["qkW"]), (a["pv"], a["pvW"])):
                assert all(s[0] <= w[0][0] and w[-1][3] <= s[1] for s, w in zip(steps, waves)); n += 1   # waves inside their job
                assert all(x[0] <= x[1] < x[2] <= x[3] for w in waves for x in w); n += 1                 # load, slices, drain
            assert all(len(w) == len(a["qkW"][0]) for w in a["qkW"]); n += 1
            assert len(a["convQ"]) == len(a["convK"]) == len(a["convV"]) == h["n"]; n += 1
            assert a["qk"][0][0] >= (a["streamEnd"] if a["runs"] == 1 else a["proj"][a["S"] - 1][7]); n += 1
            assert sum(a["rowsPerTile"]) == h["dm"]; n += 1
            pp, xb = a["P_PROJ"], a["crossbars"]                          # the module labels of the mapping board
            assert xb["gemm"] == 3 * pp * a["V"] and xb["qk"] == a["N_E"] + 2 * pp and xb["sm"] == a["N_EXP"] + 2; n += 1
            assert xb["pv"] == a["N_EP"] + pp and xb["acc"] == 2 * a["N_OC"]; n += 1
    assert "flowData" in section(res) and "flowData" in map_section(res); n += 1
    assert "<\\/" not in SCRIPT and "<\\/" not in MAP_SCRIPT; n += 1
    plants = {"board": ('rich(svg, {x: SLOT.pv[0], y: 82, class: "ft ftl"}, "O_{b} = P_{b} V_{b}");', "SLOT.pv[0]", "SLOT.sm[0] + 20"),
              "map": ('ops.smSub = node(690, RL, 136, "s \u2212 m = log P");', "690, RL, 136", "550, RL, 136")}
    node = shutil.which("node")
    for name, script, which in (("board", SCRIPT, section), ("map", MAP_SCRIPT, map_section)):
        if node:                                                      # the script under a stub DOM, both heads, 201 times each
            harness = HARNESS.replace("__DATA__", json.dumps(data)).replace("__SCRIPT__", script)
            CHECK_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, dir=str(CHECK_DIR), encoding="utf-8") as f:
                f.write(harness)
            r = subprocess.run([node, f.name], capture_output=True, text=True, timeout=120)
            assert r.returncode == 0 and "runtime ok" in r.stdout, (name, r.stderr or r.stdout); n += 1
        else:
            print("  node not found: the script's runtime check is skipped")
        if chrome():
            rep = layout_check(res, None, which)
            for issue in rep["issues"]:
                print(f"  layout ({name}): {issue['kind']}: {issue['text']!r} ({issue['board']}) {issue.get('detail', '')}")
            assert not rep["issues"], f"{name}: {len(rep['issues'])} layout issues"
            n += 1
            print(f"  {name}: layout check in headless Chrome: no issues (IBM Plex Sans loaded: {rep['font']})")
            planted, old, new = plants[name]
            assert script.count(planted) == 1, (name, planted)
            caught = layout_check(res, script.replace(planted, planted.replace(old, new)), which)["issues"]
            assert any(i["kind"] == "label on label" for i in caught), f"{name}: the layout check missed a planted overlap"
            n += 1
            print(f"  {name}: planted overlap caught ({len(caught)} reports)")
        else:
            print("  Chrome not found: the layout check is skipped")
    print(f"dataflow self-test: ALL PASS ({n} checks)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shot", nargs=2, metavar=("HEAD", "FRACTION"), help="screenshot head HEAD at FRACTION of the time")
    ap.add_argument("--map", action="store_true", help="with --shot: the mapping board (the time page) instead of the full board")
    a = ap.parse_args()
    if a.shot:
        res = json.loads((STAGES_DIR / "stages.json").read_text())
        print(screenshot(res, int(a.shot[0]), float(a.shot[1]), map_section if a.map else None))
        return 0
    return selfTest()


if __name__ == "__main__":
    sys.exit(main())
