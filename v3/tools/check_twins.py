#!/usr/bin/env python3
"""check_twins.py — consistency gate for the hand-split simulation / VTR RTL twins of v3 (PLAN R1, R4).

Derived from v2/experiment/rtl/check_split.py (the parsing helpers and the five pair checks are kept as they were) and
from the static check of v2/experiment/vtr/vtr_tops.py, set to v3's layout: each module keeps its twins in
ip/<module>/rtl/sim/ (behavior `dpe` model, weight programming, probes) and ip/<module>/rtl/vtr/ (bare `dpe` hard
blocks, never simulated). Checks:

  dpe       the `dpe` ports of ip/xbar/rtl/sim/dpe_nldpe.v == ip/xbar/rtl/vtr/dpe_blackbox.v, widths by the same
            expressions (R-bit data_in, C-bit data_out: the bit-plane interface, 2026-10-23) == the
            `<model name="dpe">` port set of every arch XML in v3/vtr/arch/; the blackbox has no body; and
            vtr/gen_arch.py --check (the staged XMLs are its output)
  widths    every `dpe` connects exactly R and C bits: the head's two twins elaborate at an R != C build
            (BERT-Tiny at 256x128; the VTR twin with a copy of the blackbox whose defaults are that R and C), and
            iverilog must report no port padding or pruning. Runs with the full gate and with --only att_head
  identical shared fabric files with no `dpe` must be byte-identical in both twins (att_fifo.v)
  per pair  1 static   the VTR twin passes the static check (no `dpe` model, no parameterized `dpe`, no weight
                       programming, the expected number of bare `dpe` instance sites); the simulation twin must fail
                       it (negative control)
            2 modules  VTR modules == simulation modules minus the declared sim-only ones
            3 top      the top's parameters (names + defaults) are identical; its ports are identical minus the
                       declared sim-only ports
            4 diff     the normalized twin difference (comments, blank lines and whitespace removed; positions
                       ignored) equals the recorded baseline tools/twin_baseline/<pair>.diff; after an intended
                       twin-only edit, re-record with --update and review the diff
            5 elab     iverilog elaborates the simulation twin with dpe_nldpe.v and the VTR twin with dpe_blackbox.v

The copied DPE runner (ip/xbar/tb/run_dpe_rtl.py) runs this gate as its interface pre-check (exit code 0 required).

Usage:
  python3 v3/tools/check_twins.py               # gate every pair
  python3 v3/tools/check_twins.py --only gemm   # one or more pairs (comma list)
  python3 v3/tools/check_twins.py --update      # re-record the twin-diff baselines
  python3 v3/tools/check_twins.py --self-test   # mutations of a passing pair must be caught
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
IP = V3 / "ip"
BASE = HERE / "twin_baseline"
ARCH_DIR = V3 / "vtr" / "arch"                # the row-parallel XMLs (one folder since 2026-10-23)
WIDTH_BUILD = (128, 128, 64, 256, 128)          # (NSEQ, DM, D, R, C) of the width gate: BERT-Tiny at 256x128

# pair -> (module folder, file, top module, sim-only modules, sim-only top ports, (dependency folder, file)...)
PAIRS = {
    "gemm": ("gemm", "gemm_top.v", "gemm_top", (), ("weight_in", "weight_en"), ()),
    "dimm_tile": ("dimm", "dimm_tile_top.v", "dimm_tile_top", (), ("weight_in", "weight_en"), ()),
    "softmax_online": ("softmax", "softmax_online_top.v", "softmax_online_top", (), (), ()),
    "softmax_online_acc": ("softmax", "softmax_online_acc.v", "softmax_online_acc", (), ("weight_in", "weight_en"),
                           (("lib", "att_fifo.v"),)),
    "att_head": ("top", "att_head.v", "att_head", (), (),
                  (("lib", "att_fifo.v"), ("gemm", "gemm_top.v"), ("dimm", "dimm_tile_top.v"),
                   ("softmax", "softmax_online_top.v"), ("softmax", "softmax_online_acc.v"), ("top", "att_ctrl.v"))),
}
# shared fabric files with no `dpe`: the simulation and VTR copies must be byte-identical
IDENTICAL = (("lib", "att_fifo.v"), ("top", "att_ctrl.v"))
# bare `dpe` instance sites per VTR file (vtr_tops.BARE_DPE_INSTANCES)
BARE_DPE_INSTANCES = {"gemm_top.v": 1, "softmax_online_top.v": 1, "dimm_tile_top.v": 2, "softmax_online_acc.v": 2,
                      "att_head.v": 0}
DPE_SIM = IP / "xbar" / "rtl" / "sim" / "dpe_nldpe.v"
DPE_VTR = IP / "xbar" / "rtl" / "vtr" / "dpe_blackbox.v"


def twin(folder: str, fname: str, side: str) -> Path:
    """A module folder's twin; `top` is the head's folder v3/top/."""
    return (V3 / "top" if folder == "top" else IP / folder) / "rtl" / side / fname


# ---------------------------------------------------------------------------------------------------------------------
# parsing helpers (comment-stripped source text), as in check_split.py
# ---------------------------------------------------------------------------------------------------------------------
def strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def normalized_lines(s: str) -> list:
    out = []
    for ln in strip_comments(s).splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            out.append(ln)
    return out


def modules(s: str) -> dict:
    code = strip_comments(s)
    return {m.group(1): m.group(0) for m in
            re.finditer(r"^\s*module\s+(\w+)\b.*?^\s*endmodule\b", code, re.M | re.S)}


def _paren_group(s: str, i: int) -> tuple:
    """s[i] == '(' -> (inner text, index after the matching ')')."""
    assert s[i] == "("
    depth = 0
    for j in range(i, len(s)):
        depth += {"(": 1, ")": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    raise ValueError("unbalanced parentheses")


def _split_top(s: str) -> list:
    """Split on commas outside (), [] and {}."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def header(mod_text: str) -> tuple:
    """(parameters [(name, default)], ports [(dir, kind, width, name)]) of a module."""
    m = re.match(r"\s*module\s+\w+\s*", mod_text)
    i = m.end()
    params_txt = ""
    if mod_text[i] == "#":
        i = mod_text.index("(", i)
        params_txt, i = _paren_group(mod_text, i)
    i = mod_text.index("(", i)
    ports_txt, _ = _paren_group(mod_text, i)
    params = []
    for p in _split_top(params_txt):
        pm = re.match(r"\s*parameter\s+(?:\[[^\]]*\]\s*)?(\w+)\s*=\s*(.*?)\s*$", p, re.S)
        if pm:
            params.append((pm.group(1), re.sub(r"\s+", "", pm.group(2))))
    ports = []
    for p in _split_top(ports_txt):
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        pm = re.match(r"(input|output|inout)\s+(wire|reg)?\s*(\[[^\]]*\])?\s*(\w+)$", p)
        if pm is None:
            raise ValueError(f"cannot parse port {p!r}")
        d, kind, w, name = pm.groups()
        ports.append((d, kind or "wire", re.sub(r"\s+", "", w or ""), name))
    return params, ports


def hunks(sim: str, vtr: str) -> list:
    """Position-free normalized difference: [(removed lines, added lines)]."""
    a, b = normalized_lines(sim), normalized_lines(vtr)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((tuple(a[i1:i2]), tuple(b[j1:j2])))
    return out


def render(h: list) -> str:
    parts = []
    for rem, add in h:
        parts.append("@@\n" + "".join(f"- {x}\n" for x in rem) + "".join(f"+ {x}\n" for x in add))
    return "".join(parts)


def iverilog_ok(files: list, top: str) -> tuple:
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["iverilog", "-g2005", "-o", str(Path(td) / "a.out"), "-s", top,
                            *map(str, files)], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def static_check(src: str, fname: str) -> list:
    """Problems that make a text unfit as a VTR twin (empty list = clean); vtr_tops.static_check on a text."""
    code = re.sub(r"//[^\n]*", "", src)
    problems = []
    if re.search(r"^\s*module\s+dpe\b", code, re.M):
        problems.append("defines module dpe (behavior model)")
    if re.search(r"\bdpe\s*#", code):
        problems.append("parameterized dpe instance (dpe #(...))")
    if re.search(r"\w+_wprog\b", code):
        problems.append("weight-programming module/instance present")
    for tok in ("dpe_weight_data", "dpe_load_input_reg", "prog_start", "prog_ready", "weight_in", "weight_en"):
        if re.search(rf"\b{tok}\b", code):
            problems.append(f"weight-programming signal {tok}")
    for m in re.finditer(r"\.load_input_reg\(([^)]*)\)", code):
        if m.group(1).strip() != "1'b0":
            problems.append(f"load_input_reg driven by {m.group(1).strip()}")
    n_bare = len(re.findall(r"^\s*dpe\s+\w+\s*\(", code, re.M))
    want = BARE_DPE_INSTANCES.get(fname)
    if want is not None and n_bare != want:
        problems.append(f"{n_bare} bare dpe instance sites, expected {want}")
    return problems


# ---------------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------------
def check_pair(pair: str, sim: str, vtr: str, baseline: str | None, elab: bool = True) -> list:
    folder, fname, top, sim_only_mods, sim_only_ports, deps = PAIRS[pair]
    probs = []
    # 1 static (+ negative control on the simulation twin)
    probs += [f"static: {x}" for x in static_check(vtr, fname)]
    if not static_check(sim, fname):
        probs.append("static: the simulation twin passes the static check (negative control)")
    # 2 modules
    ms, mv = modules(sim), modules(vtr)
    want = set(ms) - set(sim_only_mods)
    if set(mv) != want:
        probs.append(f"modules: VTR has {sorted(set(mv) - want)} extra, misses {sorted(want - set(mv))}")
    missing_sim_only = [m for m in sim_only_mods if m not in ms]
    if missing_sim_only:
        probs.append(f"modules: declared sim-only {missing_sim_only} not in the simulation twin")
    # 3 top header
    if top in ms and top in mv:
        ps, pts = header(ms[top])
        pv, ptv = header(mv[top])
        if ps != pv:
            probs.append(f"top params differ: sim {ps} vs vtr {pv}")
        expect = [p for p in pts if p[3] not in sim_only_ports]
        if [p[::2] + (p[3],) for p in expect] != [p[::2] + (p[3],) for p in ptv]:
            probs.append("top ports differ (sim minus sim-only ports vs vtr): "
                         f"{[p[3] for p in expect]} vs {[p[3] for p in ptv]}")
    else:
        probs.append(f"top module {top} missing")
    # 4 diff vs baseline
    got = render(hunks(sim, vtr))
    if baseline is None:
        probs.append("diff: no baseline recorded (run --update and review it)")
    elif got != baseline:
        g, b = got.splitlines(), baseline.splitlines()
        delta = [x for x in difflib.unified_diff(b, g, "baseline", "current", n=0, lineterm="")][2:12]
        probs.append("diff: twin difference changed vs baseline:\n      " + "\n      ".join(delta))
    # 5 elaboration
    if elab:
        with tempfile.TemporaryDirectory() as td:
            s_p, v_p = Path(td) / "sim.v", Path(td) / "vtr.v"
            s_p.write_text(sim)
            v_p.write_text(vtr)
            ok, msg = iverilog_ok([s_p, *(twin(f, n, "sim") for f, n in deps), DPE_SIM], top)
            if not ok:
                probs.append(f"elab sim: {msg[:300]}")
            ok, msg = iverilog_ok([v_p, *(twin(f, n, "vtr") for f, n in deps), DPE_VTR], top)
            if not ok:
                probs.append(f"elab vtr: {msg[:300]}")
    return probs


def dpe_ports(text: str) -> list:
    _, ports = header(modules(text)["dpe"])
    return [(d, w, n) for d, _, w, n in ports]


def check_dpe(sim_text: str | None = None, bb_text: str | None = None) -> list:
    sim_text = DPE_SIM.read_text() if sim_text is None else sim_text
    bb_text = DPE_VTR.read_text() if bb_text is None else bb_text
    probs = []
    s, v = dpe_ports(sim_text), dpe_ports(bb_text)
    if s != v:
        probs.append(f"dpe ports: sim {s} != blackbox {v}")
    archs = sorted(ARCH_DIR.glob("nl_dpe_*_auto.xml"))
    if not archs:
        probs.append(f"no arch XML under {ARCH_DIR}")
    for arch in archs:
        root = ET.parse(arch).getroot()
        model = next(m for m in root.iter("model") if m.get("name") == "dpe")
        xml = {("input", p.get("name")) for p in model.find("input_ports")} | \
              {("output", p.get("name")) for p in model.find("output_ports")}
        if {(d, n) for d, _, n in v} != xml:
            probs.append(f"dpe ports vs {arch.name} <model name=\"dpe\">: {sorted(xml ^ {(d, n) for d, _, n in v})}")
    if re.search(r"\balways\b|\bassign\b", strip_comments(bb_text)):
        probs.append("dpe_blackbox.v has a body")
    sp, bp = (dict((n, d) for n, d in header(modules(t)["dpe"])[0]) for t in (sim_text, bb_text))
    for k in ("KERNEL_WIDTH", "NUM_COLS"):
        if k not in bp or sp.get(k) != bp[k]:
            probs.append(f"dpe parameter {k}: sim {sp.get(k)} vs blackbox {bp.get(k)}")
    return probs


def head_overrides(build: tuple = WIDTH_BUILD) -> list:
    """att_head's top parameters at one build (simulator/models/attentionHead.py headParams), as iverilog -P flags."""
    sys.path.insert(0, str(V3))
    from simulator.models.attentionHead import headParams
    from tools.check_head_params import TOP_KEYS
    hp = headParams(*build)
    return [f"-Patt_head.{k}={int(getattr(hp, k))}" for k in TOP_KEYS]


def check_widths(build: tuple = WIDTH_BUILD) -> list:
    """Elaborate both head twins at an R != C build; any port padding or pruning (a width mismatch on any instance,
    the bare dpe ones included) is a problem."""
    folder, fname, top, _, _, deps = PAIRS["att_head"]
    R, C = build[3], build[4]
    flags = head_overrides(build)
    probs = []
    with tempfile.TemporaryDirectory() as td:
        bb = Path(td) / "dpe_blackbox.v"
        bb.write_text(DPE_VTR.read_text().replace("parameter KERNEL_WIDTH = 256", f"parameter KERNEL_WIDTH = {R}")
                      .replace("parameter NUM_COLS     = 256", f"parameter NUM_COLS     = {C}"))
        for side, dpe in (("sim", DPE_SIM), ("vtr", bb)):
            files = [twin(folder, fname, side), *(twin(f, n, side) for f, n in deps), dpe]
            r = subprocess.run(["iverilog", "-g2005", "-o", str(Path(td) / f"{side}.out"), "-s", top, *flags,
                                *map(str, files)], capture_output=True, text=True)
            text = r.stdout + r.stderr
            bad = [ln for ln in text.splitlines() if re.search(r"[Pp]adding|[Pp]runing", ln)]
            if r.returncode != 0:
                probs.append(f"widths {side}: elaboration failed: {text.strip()[:300]}")
            elif bad:
                probs.append(f"widths {side}: {len(bad)} port width mismatches at {R}x{C}, first: {bad[0][:200]}")
    return probs


def check_arch() -> list:
    r = subprocess.run([sys.executable, str(V3 / "vtr" / "gen_arch.py"), "--check"], capture_output=True, text=True)
    return [] if r.returncode == 0 else [f"gen_arch.py --check: {(r.stdout + r.stderr).strip()[-300:]}"]


def run(pairs: list, update: bool) -> int:
    fails = 0
    probs = check_dpe() + check_arch()
    print(f"  dpe             {'OK' if not probs else 'FAIL'}  (sim == blackbox == {len(list(ARCH_DIR.glob('*.xml')))} "
          f"row-parallel arch XMLs; gen_arch --check)" + "".join(f"\n      {p}" for p in probs))
    fails += bool(probs)
    if "att_head" in pairs:
        probs = check_widths()
        print(f"  widths          {'OK' if not probs else 'FAIL'}  (the head's twins at {WIDTH_BUILD[3]}x{WIDTH_BUILD[4]}: "
              f"no port padding or pruning)" + "".join(f"\n      {p}" for p in probs))
        fails += bool(probs)
    BASE.mkdir(exist_ok=True)
    for folder, f in IDENTICAL:
        same = twin(folder, f, "sim").read_bytes() == twin(folder, f, "vtr").read_bytes()
        print(f"  {f:15s} {'OK' if same else 'FAIL'}  (identical in both twins)"
              + ("" if same else "\n      the simulation and VTR copies differ"))
        fails += not same
    for k in pairs:
        folder, fname = PAIRS[k][:2]
        sim, vtr = twin(folder, fname, "sim").read_text(), twin(folder, fname, "vtr").read_text()
        bl = BASE / f"{k}.diff"
        if update:
            bl.write_text(render(hunks(sim, vtr)))
        baseline = bl.read_text() if bl.exists() else None
        probs = check_pair(k, sim, vtr, baseline)
        h = hunks(sim, vtr)
        print(f"  {k:15s} {'OK' if not probs else 'FAIL'}  ({len(h)} twin hunks, "
              f"-{sum(len(r) for r, _ in h)}/+{sum(len(a) for _, a in h)} lines)"
              + (" [baseline re-recorded]" if update else "")
              + "".join(f"\n      {p}" for p in probs))
        fails += bool(probs)
    print("check_twins: ALL PASS" if not fails else f"check_twins: {fails} FAILURE(S)")
    return 0 if not fails else 1


def self_test() -> int:
    """Mutations of a passing pair must each be caught; a comment-only edit must not be."""
    k = "softmax_online"
    folder, fname = PAIRS[k][:2]
    sim, vtr = twin(folder, fname, "sim").read_text(), twin(folder, fname, "vtr").read_text()
    base = render(hunks(sim, vtr))
    assert not check_pair(k, sim, vtr, base, elab=False), "clean pair flagged"
    shared = "localparam LRW    = $clog2(RWIN);"
    clk = "    input  wire               clk,\n"
    j = vtr.index(clk, vtr.index("module softmax_online_top #("))     # the top's first port

    def top_port(name: str) -> str:
        return vtr[:j] + clk + f"    input  wire               {name},\n" + vtr[j + len(clk):]

    assert sim.count(shared) == 1 and vtr.count(shared) == 1, "anchor: shared localparam"
    assert vtr.count("dpe u_dpe (") == 1, "anchor: the dpe instance"
    cases = {
        "shared edit in the sim twin only": (sim.replace(shared, "localparam LRW    = $clog2(RWIN + 1);"), vtr),
        "shared edit in the VTR twin only": (sim, vtr.replace(shared, "localparam LRW    = $clog2(RWIN + 1);")),
        "top port added to the VTR twin": (sim, top_port("extra")),
        "parameterized dpe in the VTR twin": (sim, vtr.replace("dpe u_dpe (", "dpe #(.KERNEL_WIDTH(R)) u_dpe (")),
        "module dropped from the VTR twin": (sim, re.sub(r"module softmax_online_ohmux\b.*?endmodule\n", "", vtr,
                                                         count=1, flags=re.S)),
        "weight port in the VTR twin": (sim, top_port("weight_en")),
    }
    caught = 0
    for name, (s, v) in cases.items():
        assert s != sim or v != vtr, f"mutation '{name}' did not apply"
        probs = check_pair(k, s, v, base, elab=False)
        print(f"  mutation: {name:38s} {'caught' if probs else 'MISSED'}"
              + (f"  ({probs[0].splitlines()[0][:70]})" if probs else ""))
        caught += bool(probs)
    probs = check_pair(k, sim.replace("// ====", "// ----", 1), vtr, base, elab=False)
    print(f"  comment-only edit                             {'ignored' if not probs else 'FLAGGED'}")
    # the dpe check: a clean pass, then a port direction flip, a dropped port and a body must be caught
    bb = DPE_VTR.read_text()
    ready = "    output wire                    MSB_SA_Ready,\n"
    assert bb.count(ready) == 1 and bb.count("endmodule") == 1, "anchors: dpe blackbox"
    assert not check_dpe(), "clean dpe flagged"
    dpe_cases = {
        "dpe blackbox port direction": bb.replace(ready, ready.replace("output", "input ")),
        "dpe blackbox port dropped": bb.replace(ready, ""),
        "dpe blackbox body": bb.replace("endmodule", "    assign data_out = 0;\nendmodule"),
        "dpe blackbox width": bb.replace("[NUM_COLS-1:0]", "[NUM_COLS:0]"),
    }
    for name, text in dpe_cases.items():
        hit = bool(check_dpe(bb_text=text))
        print(f"  mutation: {name:38s} {'caught' if hit else 'MISSED'}")
        caught += hit
    ok = caught == len(cases) + len(dpe_cases) and not probs
    print("check_twins self-test: ALL PASS" if ok else "check_twins self-test: FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma list of " + ",".join(PAIRS))
    ap.add_argument("--update", action="store_true", help="re-record the twin-diff baselines")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    pairs = list(PAIRS) if a.only is None else a.only.split(",")
    return run(pairs, a.update)


if __name__ == "__main__":
    sys.exit(main())
