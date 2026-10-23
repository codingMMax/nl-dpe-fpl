#!/usr/bin/env python3
"""stage_analysis.py — the mapping-aware stage analysis of the two case-study heads (docs/attention_stages.md §6):
the pipelined time diagram, the per-stage tables, the analytic end-to-end cycle and energy models, and their sweep
over the array size.

For each case-study head (space.py CASE_STUDY: the smallest and the largest attention shape) and each of the 8
configs the simulator is run (timing only, the trace on for the critical path) and the script records:
  stages   per stage (projections, QK, softmax, PV, accumulator): crossbars, passes, energy, cycles on the critical path
  gantt    the busy intervals of each of the five stages (the union of its processes' steps), for the time diagram
  laws     the three scaling laws' terms per config (docs/attention_stages.md §5): the projection stream n T_p, the
           last run's scores n S / 5, the core floor n² / 5, the tail; the energy per stage in closed form; the area
           as the fixed build's CLBs and memory blocks plus the crossbar tiles
  model    the analytic cycles T_model and energy E_model of docs/attention_stages.md §6, with their parts
  grid     the array sweep on the DSE grid: per C the curve over R of the simulator's T, E, A and the closed forms'
           T_head, E_head, A_head, and the product cycles × area relative to the smallest array
  diagrams the per-process steps at the DIAGRAMS arrays (128x64 and 512x256), for the brief's "head in time" pair
The analytic models are checked against the simulator on the 8 configs (--self-test prints the largest errors).
Output: dse/results/stages/stages.json (dse/plot_stages.py draws it, dse/stages_page.py writes the page).
Usage: python3 v3/dse/stage_analysis.py [--self-test]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
import simulator.core.trace as trace  # noqa: E402
from dse import data  # noqa: E402
from dse.space import CASE_STUDY, CONFIGS, SHAPES  # noqa: E402
from simulator.core.cycleMath import ceilDiv  # noqa: E402
from simulator.core.engine import run  # noqa: E402
from simulator.core.trace import breakdown  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams, nOcFloor  # noqa: E402
from simulator.platform import area, prices  # noqa: E402
from simulator.platform.platform import Platform  # noqa: E402

OUT_DIR = data.RESULTS / "stages"
OUT_JSON = OUT_DIR / "stages.json"
STAGES = ("gemm", "qk", "sm", "pv", "acc")
STAGE_NAME = {"gemm": "projections", "qk": "tiled QK", "sm": "online softmax", "pv": "tiled PV", "acc": "output rescale"}
EXTREMES = ((128, 64), (512, 256))
DIAGRAMS = ((128, 64), (512, 256))        # the brief's "head in time" pair: the smallest and the largest array; their steps are stored
BUF, P, BKV, EPS = 40, 8, 16, 5
# the pass price (simulator/platform/nl_dpe.json through prices.py): the cells per bit-slice per cell, the ACAM per
# column, the input buffer per row loaded (3 accesses), the output buffer per column (2 accesses); pJ
CELL_PJ, ACAM_PJ_PER_COL, BUF_PJ = 0.16375 / 8192, 10.93875 / 64, 0.00495
BRAM_PJ = 0.0495
# the fixed latencies the cycle model carries (the modules' named constants, simulator/models/*)
L_PROJ = 4          # a token's outputs: the tree (log2 V, counted apart) + the serializer + the converter's input register
FILL_QK = 24        # the first QK job: the staging, the registered decision, the farm's start latency, the tree
TAIL_RUN = 4        # a run's end: the softmax's done (after its S log l words) -> busy falls -> go -> armed


def tau(I: int, C: int) -> int:
    """A pass that loads I rows on a C-column array."""
    return max(ceilDiv(8 * I, BUF) + P, ceilDiv(8 * C, BUF) + 1)


def passPrice(R: int, C: int, rows: int) -> float:
    """pJ of one pass that loads `rows` rows: the cells, the ACAM, the buffers (prices.passEnergyPj in closed form)."""
    return P * CELL_PJ * R * C + ACAM_PJ_PER_COL * C + 3 * BUF_PJ * rows + 2 * BUF_PJ * C


def passTraffic(C: int, rows: int) -> float:
    """pJ of one pass's port words (block RAM accesses)."""
    return (ceilDiv(8 * rows, BUF) + ceilDiv(8 * C, BUF)) * BRAM_PJ


def mapping(n: int, dm: int, d: int, R: int, C: int) -> dict:
    """The fixed mapping's sizes and the derived counts of a head (headParams, without building it)."""
    I = min(R, C)
    S = min(n, I)
    return dict(I=I, S=S, RB=n // S, B=n // BKV, V=ceilDiv(dm, R), rowsX=min(dm, R), N_E=2 * d, N_EP=2 * d, N_EXP=2,
                N_OC=nOcFloor(S, d, R, C, BKV), P_PROJ=1)


# -- the analytic model: the named equations of docs/attention_stages.md §5 (the stage page) ------------------------
K_READER = EPS      # the softmax's score reader: one 40-bit word, 5 scores per cycle (softmax_online_instage)
E_ADD, E_COMPARE, E_MEMORY = 0.08498, 0.26439, BRAM_PJ      # pJ per CLB add, per compare, per 5-byte memory access


def modelCycles(n: int, dm: int, d: int, R: int, C: int) -> dict:
    """T_head and every named term (the stage page, section "Cycles per stage" and "Latency")."""
    m = mapping(n, dm, d, R, C)
    I, S, N_run, N_block, V = m["I"], m["S"], m["RB"], m["B"], m["V"]
    tauI, tauS = tau(I, C), tau(S, C)
    T_projection_token = max(ceilDiv(8 * m["rowsX"], BUF) + P, ceilDiv(8 * C, BUF) + 1)
    T_projection_stream = n * T_projection_token / m["P_PROJ"]
    T_QK = S * BKV * d * tauI / (I * m["N_E"])
    T_PV = S * d * BKV * tauI / (I * m["N_EP"])
    T_EXP = S * BKV * tauI / (I * m["N_EXP"])
    T_reader = ceilDiv(S * BKV, K_READER)
    T_softmax_latency = T_reader + T_EXP + 2 * tauS + ceilDiv(S, EPS)
    T_softmax = max(T_reader, T_softmax_latency / 2)
    W_rescale = ceilDiv(S * d, I * m["N_OC"])
    # the output rescale's loop: a block's EXP passes, its PV job, its rescale waves and the two pass latencies, through
    # the alpha ring that holds two blocks (architecture.md 5.2, E6): the loop may take two block periods
    T_output_rescale = (T_EXP + T_PV + W_rescale * tauI + 2 * tauI) / 2
    T_block = max(T_QK, T_softmax, T_EXP, T_PV, T_output_rescale)
    # the stream slower than the core: the first run is pinned to the stream, the other runs follow it; one run: all
    # of it follows the stream
    T_stream_bound = T_projection_stream + max(1, N_run - 1) * N_block * T_block
    T_core_bound = S * T_projection_token / m["P_PROJ"] + N_run * N_block * T_block
    T_fill = (V - 1).bit_length() + L_PROJ + FILL_QK
    T_first_block = N_run * max(0.0, T_softmax_latency - T_block)
    T_readout = (N_run - 1) * (S + TAIL_RUN)
    T_tail = T_PV + W_rescale * tauI
    T_head = max(T_stream_bound, T_core_bound) + T_fill + T_first_block + T_readout + T_tail
    return dict(T=T_head, T_head=T_head, T_projection_token=T_projection_token, T_projection_stream=T_projection_stream,
                T_QK=T_QK, T_reader=T_reader, T_EXP=T_EXP, T_softmax_latency=T_softmax_latency, T_softmax=T_softmax,
                T_PV=T_PV, W_rescale=W_rescale, T_output_rescale=T_output_rescale, T_block=T_block,
                T_stream_bound=T_stream_bound, T_core_bound=T_core_bound, T_fill=T_fill, T_first_block=T_first_block,
                T_readout=T_readout, T_tail=T_tail, tau_I=tauI, tau_S=tauS, S=S, I=I, N_run=N_run, N_block=N_block,
                V=V, bound="core" if T_core_bound >= T_stream_bound else "stream",
                # the keys the sweep figure reads
                stream=T_projection_stream, core=N_run * N_block * T_block,
                tail=T_fill + T_first_block + T_readout + T_tail)


def modelEnergy(n: int, dm: int, d: int, R: int, C: int, eClb: float = 0.0) -> dict:
    """E_head and every stage's energy (the stage page, section "Energy per stage"), in pJ; eClb unused (kept for
    the call sites): the CLB terms are closed forms here."""
    m = mapping(n, dm, d, R, C)
    I, S, N_run, N_block, V = m["I"], m["S"], m["RB"], m["B"], m["V"]
    e = lambda r: passPrice(R, C, r) + passTraffic(C, r)          # noqa: E731  one pass loading r rows, with its port words
    passes = dict(gemm=3 * V * n, qk=n * n * d // I + 2 * n, sm=n * n // I + N_run * (2 * N_block - 1),
                  pv=n * n * d // I + n, acc=2 * n * n * d // (I * BKV))
    out = dict(
        gemm=3 * V * n * e(m["rowsX"]) + 3 * n * d * (V - 1) * E_ADD,
        qk=(n * n * d // I) * e(I) + 2 * n * e(d) + (2 * n * n * d - n * n) * E_ADD + n * n * E_COMPARE,
        sm=(n * n // I) * e(I) + N_run * (2 * N_block - 1) * e(S) + (2 * n * n + n * n // 8) * E_ADD
           + (n * n + n * n // 16) * E_COMPARE,
        pv=(n * n * d // I) * e(I) + n * e(d) + (2 * n * n * d - n * n * d // BKV) * E_ADD,
        acc=(2 * n * n * d // (I * BKV)) * e(I) + (n * n * d // 8) * E_ADD)
    memBytes = 3 * n * d + 2 * n * n * d / BKV + 2 * n * n * d / S + 4 * n * n + 7 * n * n * d / BKV
    out["memory"] = E_MEMORY * memBytes / 5
    out["clb"] = out["memory"]                                      # the name the laws table used
    out["E"] = sum(out[k] for k in ("gemm", "qk", "sm", "pv", "acc", "memory"))
    out["passes"] = passes
    return out


def modelArea(n: int, dm: int, d: int, R: int, C: int, clb: int, mem: int) -> dict:
    """A_head and every stage's crossbar area (the stage page, section "Area per stage"), in mm²; the CLBs and the
    memory blocks are the fixed build's VTR counts."""
    m = mapping(n, dm, d, R, C)
    w, h = area.footprint(R, C)
    tile = w * h * area.MWTA_PER_CELL * data.MM2_PER_MWTA
    xb = dict(gemm=3 * m["P_PROJ"] * m["V"], qk=m["N_E"] + 2 * m["P_PROJ"], sm=m["N_EXP"] + 2,
              pv=m["N_EP"] + m["P_PROJ"], acc=2 * m["N_OC"])
    out = {k: v * tile for k, v in xb.items()}
    out["clb"] = clb * area.MWTA_PER_CELL * data.MM2_PER_MWTA
    out["memory"] = mem * area.MEM_TILE_CELLS * area.MWTA_PER_CELL * data.MM2_PER_MWTA
    out["A"] = sum(out[k] for k in ("gemm", "qk", "sm", "pv", "acc", "clb", "memory"))
    out["crossbars"] = xb
    out["tile_cells"] = w * h
    return out


# -- the simulator side -----------------------------------------------------------------------------------------------
def simulate(n: int, dm: int, d: int, R: int, C: int) -> dict:
    """One head: T, E, the per-stage table, the Gantt lanes, the fixed CLB energy."""
    trace.enable(True)
    hp = headParams(n, dm, d, R, C)
    g, parts = attentionHead(hp, withValues=False)
    rep = run(g)
    ctrl = parts["ctrl"]
    T = ctrl.done - ctrl.tStart
    path = {}
    for (role, _), c in breakdown(ctrl.done, since=ctrl.tStart).items():
        path[role] = path.get(role, 0) + c
    p = Platform("nl_dpe", rows=R, cols=C)
    xb = dict(gemm=3 * hp.P_PROJ * hp.V, qk=hp.N_E + 2 * hp.P_PROJ, sm=hp.N_EXP + 2, pv=hp.N_EP + hp.P_PROJ, acc=2 * hp.N_OC)
    stages = {s: dict(crossbars=xb[s], passes=0, energy_pj=0.0, path_cycles=path.get(s, 0)) for s in STAGES}
    eClb = 0.0
    for inst, c in rep.counts.items():
        role = rep.roles[inst]
        e = prices.energy(c, p)
        if role in stages:
            stages[role]["passes"] += sum(c.passes.values())
            stages[role]["energy_pj"] += e["e_total_pj"]
        eClb += e["e_adds_pj"] + e["e_cmp_pj"] + e["e_mem_pj"]
    E = sum(s["energy_pj"] for s in stages.values())
    # the accumulator's run step starts at its first output word; its normalization passes (O = o / l) load and
    # compute before that, so the drawn step starts at the first normalization pass instead (else the lane shows
    # the passes' load and bit-slices as idle: 25 cycles at I = 64, 64 at I = 256)
    epiStart = {}
    for (run_, kind, _b, _w), ev in parts["acc"].jobs.items():
        if kind == "epilogue":
            epiStart[run_] = min(epiStart.get(run_, ev["load_start"]), ev["load_start"])
    start = lambda inst, s: min(s.tStart, epiStart[s.unit]) if inst == "acc" and s.process == "run" else s.tStart  # noqa: E731
    lanes = [("gemm", ("gq0", "gk0", "gv0"), ("rows",)), ("qk", ("qk",), ("a_in", "b_in", "jobs")),
             ("sm", ("sm",), ("tile", "blk", "run")), ("pv", ("pv",), ("a_in", "jobs")), ("acc", ("acc",), ("blk", "run"))]
    gantt = []
    for stage, insts, procs in lanes:                                # one lane per stage: its processes' steps merged
        iv = sorted((int(start(inst, s) - ctrl.tStart), int(s.tDone - ctrl.tStart))
                    for inst in insts if inst in rep.steps for s in rep.steps[inst] if s.process in procs)
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        gantt.append(dict(stage=stage, name=STAGE_NAME[stage], intervals=merged))
    runEnds = sorted(int(s.tDone - ctrl.tStart) for s in rep.steps["sm"] if s.process == "run")
    steps = {}                                                        # every step of the stages, for the annotated diagram
    for inst in ("gq0", "gk0", "gv0", "qk", "sm", "pv", "acc"):
        for s in rep.steps.get(inst, []):
            steps.setdefault(f"{inst}.{s.process}", []).append((int(start(inst, s) - ctrl.tStart), int(s.tDone - ctrl.tStart),
                                                                 int(s.unit)))
    for v in steps.values():
        v.sort()
    return dict(T=int(T), E=E, stages=stages, gantt=gantt, run_ends=runEnds, steps=steps, events=passEvents(rep, parts),
                e_clb=eClb, nXbar=hp.nXbar, S=hp.S, runs=hp.RB, blocks=hp.B)


PROJ_EVENTS = ("load_start", "load_end", "compute_start", "msb_fire", "drain_start", "drain_end", "array_drain_start",
               "array_drain_end")


PHASES = ("load_start", "load_end", "drain_start", "drain_end")   # a pass: load, the 8 bit-slices, the drain


def passEvents(rep, parts) -> dict:
    """The crossbar passes behind the steps, relative to the head's start (the dataflow board of the brief). A pass
    is its PHASES: the load through the port, then the bit-slices (load_end + 1 .. drain_start - 1), then the drain.
      proj   per token, the projection tile's pass: PROJ_EVENTS (the K and V tiles run in lock-step with Q's)
      qk/pv  per job (block), its waves: the PHASES of each lock-step pass of the crossbars
      sm     per block, its EXP passes (PHASES), its carry window [wait, retire] (the carry EXP and LOG passes that
             fold the block's sums into (m, log l)) and the retire
      acc    per block, its rescale waves (the EXP group: EXP(log alpha + lo)) and its LOG waves (LOG(o')), and
             both together; epi per run, its normalization waves (EXP(lo - log l))"""
    t0 = parts["ctrl"].tStart
    rel = lambda v: int(v - t0)                                       # noqa: E731
    ph = lambda e: [rel(e[k]) for k in PHASES]                        # noqa: E731
    proj = {inst: [[rel(s.events[k]) for k in PROJ_EVENTS] for s in sorted(rep.steps[inst], key=lambda s: s.unit)]
            for inst in ("gq0", "gk0", "gv0")}
    waves = lambda inst: [[ph(w) for w in s.events["waves"]]          # noqa: E731
                          for s in sorted((x for x in rep.steps[inst] if x.process == "jobs"), key=lambda x: x.unit)]
    blk = sorted((x for x in rep.steps["sm"] if x.process == "blk"), key=lambda x: x.unit)
    acc = parts["acc"]
    nblk = acc.p.NBLK
    accE, accL, epi = [[] for _ in range(len(blk))], [[] for _ in range(len(blk))], [[] for _ in range(acc.runs)]
    for (run_, kind, b, w), e in acc.jobs.items():
        (epi[run_] if kind == "epilogue" else accE[run_ * nblk + b] if kind == "rescale" else accL[run_ * nblk + b]).append(ph(e))
    return dict(proj=proj["gq0"], projSame=proj["gk0"] == proj["gq0"] and proj["gv0"] == proj["gq0"],
                qk=waves("qk"), pv=waves("pv"),
                sm=[[ph(q) for q in x.events["passes"]] for x in blk],
                smCarry=[[rel(x.events["wait"]), rel(x.events["retire"])] for x in blk],
                smRetire=[rel(x.events["retire"]) for x in blk],
                accE=[sorted(v) for v in accE], accL=[sorted(v) for v in accL],
                acc=[sorted(e + l) for e, l in zip(accE, accL)], epi=[sorted(v) for v in epi])


def laws(n: int, dm: int, d: int, R: int, C: int, sim: dict, em: dict) -> dict:
    """The three scaling laws' terms of one build (docs/attention_stages.md §5)."""
    m = mapping(n, dm, d, R, C)
    I, S = m["I"], m["S"]
    Tp = max(ceilDiv(8 * m["rowsX"], BUF) + P, ceilDiv(8 * C, BUF) + 1)
    rows = {(r["shape"], r["config"]): r for r in data.attentionRows()}
    r = rows[(next(s[0] for s in SHAPES if s[1:] == (n, dm, d)), f"{R}x{C}")]
    w, h = area.footprint(R, C)
    cells = dict(clb=int(r["clb_count"]), xbar=sim["nXbar"] * w * h, mem=area.MEM_TILE_CELLS * int(r["mem_blocks"]))
    return dict(T_p=Tp, stream=n * Tp, last_run=n * S // EPS, core_floor=n * n // EPS, S=S, I=I,
                fired_per_value=R * C / I, price_pj=passPrice(R, C, I),
                E_stage_pj={s: em[s] for s in STAGES}, E_fixed_pj=em["memory"],
                area_cells=cells, area_mm2={k: v * area.MWTA_PER_CELL * data.MM2_PER_MWTA for k, v in cells.items()},
                tile_cells=w * h, nXbar=sim["nXbar"])


def analyze(name: str, n: int, dm: int, d: int) -> dict:
    out = dict(name=name, n=n, dm=dm, d=d, configs={}, gantt={}, diagrams={}, grid=None, errors={})
    eClb = None
    for R, C in CONFIGS:
        sim = simulate(n, dm, d, R, C)
        if eClb is None:
            eClb = sim["e_clb"]
        tm = modelCycles(n, dm, d, R, C)
        em = modelEnergy(n, dm, d, R, C)
        cfg = f"{R}x{C}"
        row = next(x for x in data.attentionRows() if x["shape"] == name and x["config"] == cfg)
        am = modelArea(n, dm, d, R, C, int(row["clb_count"]), int(row["mem_blocks"]))
        out["configs"][cfg] = dict(T=sim["T"], E=sim["E"], A=float(row["area_total_mm2"]), stages=sim["stages"],
                                   nXbar=sim["nXbar"], S=sim["S"], runs=sim["runs"], blocks=sim["blocks"],
                                   e_clb=sim["e_clb"], T_model=tm, E_model=em, A_model=am,
                                   T_err=tm["T"] / sim["T"] - 1, E_err=em["E"] / sim["E"] - 1,
                                   A_err=am["A"] / float(row["area_total_mm2"]) - 1)
        out["configs"][cfg]["laws"] = laws(n, dm, d, R, C, sim, em)
        if (R, C) in EXTREMES:
            out["gantt"][cfg] = dict(T=sim["T"], lanes=sim["gantt"], run_ends=sim["run_ends"], steps=sim["steps"])
        if (R, C) in DIAGRAMS:
            out["diagrams"][cfg] = dict(T=sim["T"], run_ends=sim["run_ends"], steps=sim["steps"], events=sim["events"])
    out["e_clb"] = eClb
    out["grid"] = grid(out["configs"])
    out["errors"] = dict(T_max_abs_rel=max(abs(c["T_err"]) for c in out["configs"].values()),
                         E_max_abs_rel=max(abs(c["E_err"]) for c in out["configs"].values()))
    return out


def grid(configs: dict) -> dict:
    """The array sweep on the DSE grid: per C (the JSON key, a string) the curve over R of the simulator's T, E, A and
    the closed forms' T_head, E_head, A_head, and cycles × area relative to the smallest array (the simulator's and
    the closed forms' products each to their own)."""
    base = configs[f"{min(R for R, _ in CONFIGS)}x{min(C for _, C in CONFIGS)}"]
    out = {}
    for C in sorted({C for _, C in CONFIGS}):
        Rs = sorted(R for R, c in CONFIGS if c == C)
        cs = [configs[f"{R}x{C}"] for R in Rs]
        out[str(C)] = dict(R=Rs, I=[min(R, C) for R in Rs], bound=[c["T_model"]["bound"] for c in cs],
                           T=[c["T"] for c in cs], T_model=[c["T_model"]["T"] for c in cs],
                           E=[c["E"] for c in cs], E_model=[c["E_model"]["E"] for c in cs],
                           A=[c["A"] for c in cs], A_model=[c["A_model"]["A"] for c in cs],
                           xa=[c["T"] * c["A"] / (base["T"] * base["A"]) for c in cs],
                           xa_model=[c["T_model"]["T"] * c["A_model"]["A"]
                                     / (base["T_model"]["T"] * base["A_model"]["A"]) for c in cs])
    return out


def cases() -> list:
    byName = {s[0]: s for s in SHAPES}
    return [byName[c] for c in CASE_STUDY]


def r_row(r: dict, cfg: str) -> dict:
    return next(x for x in data.attentionRows() if x["shape"] == r["name"] and x["config"] == cfg)


def selfTest() -> int:
    n = 0
    # the closed-form pass price equals prices.passEnergyPj on every config, at I rows and at every rows value used
    for R, C in CONFIGS:
        p = Platform("nl_dpe", rows=R, cols=C)
        for rows in sorted({min(R, C), 16, 32, 64, min(R, 768), R}):
            assert abs(passPrice(R, C, rows) - prices.passEnergyPj(p, rows)["total"]) < 1e-9, (R, C, rows); n += 1
            assert abs(passTraffic(C, rows) - prices.passTrafficEnergyPj(p, rows)) < 1e-9; n += 1
    res = [analyze(*c) for c in cases()]
    for r in res:
        # the model's passes per stage equal the simulator's; the CLB energy is config-independent (< 2 %)
        for cfg, c in r["configs"].items():
            for s in STAGES:
                assert c["E_model"]["passes"][s] == c["stages"][s]["passes"], (r["name"], cfg, s); n += 1
            assert abs(c["e_clb"] / r["e_clb"] - 1) < 0.02, (r["name"], cfg, c["e_clb"], r["e_clb"]); n += 1
            assert all(0 <= a <= b <= c["T"] + 1 for lane in r.get("gantt", {}).get(cfg, {}).get("lanes", [])
                       for a, b in lane["intervals"]); n += 1
            st = (r["gantt"].get(cfg) or r["diagrams"].get(cfg) or {}).get("steps")
            if st:                                                    # the pipeline order and the query gate of the steps
                first = lambda k: st[k][0][0]                         # noqa: E731
                assert first("qk.jobs") < first("sm.tile") < first("sm.blk") < first("pv.jobs") < first("acc.blk"); n += 1
                assert all(0 <= a <= b <= c["T"] for v in st.values() for a, b, _ in v); n += 1
                if c["runs"] == 1:                                    # one run: the first score job needs the last query
                    assert first("qk.jobs") >= max(b for _, b, _ in st["gq0.rows"]), (r["name"], cfg); n += 1
                ev = (r["diagrams"].get(cfg) or {}).get("events")
                if ev:                                                # every pass inside its drawn step (no undrawn work)
                    for kind, step, passes in (("normalize", "acc.run", ev["epi"]), ("rescale", "acc.blk", ev["acc"]),
                                               ("QK", "qk.jobs", ev["qk"]), ("PV", "pv.jobs", ev["pv"])):
                        for (a, b, u), ps in zip(st[step], passes):
                            assert a <= ps[0][0] and ps[-1][-1] <= b + 1, (r["name"], cfg, kind, u, (a, b), ps[0], ps[-1]); n += 1
                            assert all(w[0] <= w[1] < w[2] <= w[3] for w in ps), (r["name"], cfg, kind, u)   # load, slices, drain
            lw = c["laws"]
            assert abs(sum(lw["E_stage_pj"].values()) + lw["E_fixed_pj"] - c["E_model"]["E"]) < 1e-6; n += 1
            assert abs(sum(lw["area_mm2"].values()) - float(r_row(r, cfg)["area_total_mm2"])) < 1e-3, (r["name"], cfg)
            assert lw["stream"] == c["T_model"]["stream"] and lw["last_run"] <= lw["core_floor"]; n += 1
        print(f"  {r['name']:16s} T_head vs simulator: max |error| {100 * r['errors']['T_max_abs_rel']:.1f} %  "
              f"({', '.join(f'{cfg} {100 * c[chr(84)+chr(95)+chr(101)+chr(114)+chr(114)]:+.0f}' for cfg, c in r['configs'].items())}) | "
              f"E_model: max |error| {100 * r['errors']['E_max_abs_rel']:.2f} %")
        assert r["errors"]["E_max_abs_rel"] < 0.03, r["name"]; n += 1
        assert r["errors"]["T_max_abs_rel"] < 0.30, r["name"]; n += 1
        assert max(abs(c["A_err"]) for c in r["configs"].values()) < 1e-4, r["name"]; n += 1   # area_total_mm2 is rounded
        n += gridChecks(r)
    n += gridMutants()
    print(f"stage_analysis self-test: ALL PASS ({n} checks)")
    return 0


def rising(xs: list) -> bool:
    return all(a < b for a, b in zip(xs, xs[1:]))


def gridChecks(r: dict) -> int:
    """The grid holds every config once, and the sweep's findings hold in the data (the stage page's bullets):
    area and cycles × area rise with R at every C; energy rises with R once R >= C; cycles are flat in R when no row
    beyond d_model is ever loaded (d_model <= the smallest R, at R >= C so that I = C) and rise with R otherwise."""
    g, n = r["grid"], 0
    assert sorted(int(C) for C in g) == sorted({C for _, C in CONFIGS}); n += 1
    assert sorted((R, int(C)) for C, gc in g.items() for R in gc["R"]) == sorted(CONFIGS); n += 1
    for C, gc in g.items():
        C = int(C)
        assert gc["R"] == sorted(gc["R"]) and gc["I"] == [min(R, C) for R in gc["R"]]; n += 1
        for key in ("T", "E", "A", "xa"):
            assert len(gc[key]) == len(gc[key + "_model"]) == len(gc["R"]); n += 1
        for key in ("A", "A_model", "xa", "xa_model"):
            assert rising(gc[key]), (r["name"], C, key); n += 1
        atI = [i for i, R in enumerate(gc["R"]) if R >= C]                     # the points with I = C
        for key in ("E", "E_model"):
            assert rising([gc[key][i] for i in atI]), (r["name"], C, key); n += 1
        flat = r["dm"] <= min(gc["R"][i] for i in atI)
        for key in ("T", "T_model"):
            ts = [gc[key][i] for i in atI]
            assert (len(set(ts)) == 1) if flat else rising(ts), (r["name"], C, key, ts); n += 1
    smallest = str(min(C for _, C in CONFIGS))
    assert g[smallest]["xa"][0] == 1.0 and g[smallest]["xa_model"][0] == 1.0; n += 1
    return n


def gridMutants() -> int:
    """gridChecks rejects a grid whose curves are mis-ordered or whose product is off by one point."""
    n = 0
    fake = {}
    for i, (R, C) in enumerate(CONFIGS):
        fake[f"{R}x{C}"] = dict(T=1000 + i, E=1.0 + R + 0.5 * C, A=1.0 + R * C / 1e3,
                                T_model=dict(T=1000.0 + i, bound="core"), E_model=dict(E=1.0 + R + 0.5 * C),
                                A_model=dict(A=1.0 + R * C / 1e3))
    r = dict(name="fake", dm=2048, grid=grid(fake))
    assert gridChecks(r) > 0; n += 1
    last = len(grid(fake)[str(max(C for _, C in CONFIGS))]["R"]) - 1           # the largest C's last point, R >= C
    for key, where in (("A", 1), ("xa", -1), ("T", last), ("E", last)):       # T and E: two points with R >= C
        g = grid(fake)
        gc = g[str(max(C for _, C in CONFIGS))]
        gc[key][where] = gc[key][where - 1] if where > 0 else gc[key][where - 1] * 0.5
        try:
            gridChecks(dict(name="mutant", dm=2048, grid=g))
        except AssertionError:
            n += 1
        else:
            raise AssertionError(f"grid mutant not caught: {key}[{where}]")
    g = grid(fake)
    g["64"]["T"] = [g["64"]["T"][0]] * len(g["64"]["T"])                       # flat where the head says rising
    try:
        gridChecks(dict(name="mutant", dm=2048, grid=g))
    except AssertionError:
        n += 1
    else:
        raise AssertionError("grid mutant not caught: flat T")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return selfTest()
    res = [analyze(*c) for c in cases()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(stages=STAGES, stage_name=STAGE_NAME, extremes=[f"{R}x{C}" for R, C in EXTREMES],
                                        diagrams=[f"{R}x{C}" for R, C in DIAGRAMS],
                                        constants=dict(BUF=BUF, P=P, BKV=BKV, L_PROJ=L_PROJ, FILL_QK=FILL_QK,
                                                       TAIL_RUN=TAIL_RUN), cases=res), indent=1) + "\n")
    for r in res:
        print(f"{r['name']}: T_model max |error| {100 * r['errors']['T_max_abs_rel']:.1f} %, E_model {100 * r['errors']['E_max_abs_rel']:.2f} %")
    print(f"-> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
