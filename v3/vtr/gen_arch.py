#!/usr/bin/env python3
"""gen_arch.py — the row-parallel arch XMLs (docs/experiment.md §2.1, §5.1; user, 2026-10-22) in vtr/arch/.

The template is vtr/arch/nl_dpe_128x64_auto.xml, itself a row-parallel XML: the others differ from it only in the wc
tile and the wc pb_type (v2's XMLs, the original template, left the tree on 2026-10-23, user: one vtr/arch/). One file
per DSE config (dse/space.py CONFIGS; 64x64 is one since 2026-10-23) and per extra size
(SMOKE, none today), nl_dpe_<R>x<C>_auto.xml, changes only:
  the wc tile      height H and width W (simulator/platform/area.py footprint); area = the block's logic + local
                   crossbar in MWTA (area.blockAreaMwta); the input split (user, 2026-10-23): I = the R data pins,
                   IC = the 7 control pins, both equivalent="none" (the data pins were swappable, equivalent="full",
                   until 2026-10-23 evening: VPR's post-route pin fix-up aborts on them, post_routing_pb_pin_fixup.cpp:559
                   '1 == cnt'; the area still pays for the local input crossbar, user); O = C + 5; the clock's
                   fc_override 0, as the CLB's
  the wc pb_type   I, IC and O (both inputs equivalent="none"), wc_slice's in (R + 7) and out; the wc-level
                   directs wc.I -> wc_slice.in[R-1:0], wc.IC -> wc_slice.in[R+6:R]
  nl_dpe           data_in R pins, data_out C pins; the directs: the data at in[R-1:0] / out[C-1:0], the 7 controls at
                   in[R..R+6], the 5 status bits at out[C..C+4] (the template's order)
Everything else stays byte-identical: pinlocations spread, the CLB's Fc 0.15 / 0.10, switchblock_locations all, the
layout, the memory, the delays.
  --check   every staged XML: parses, its footprint, area and pins equal the rule, its directs cover the ports, and its
            line diff from the template lies inside the wc tile and the wc pb_type, only on those lines; then planted
            faults (a wrong height, a wrong pin count, a shifted control bit, a stray edit, a missing file) must fail
Usage: python3 v3/vtr/gen_arch.py [--check]
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path.insert(0, str(V3))
from dse.space import CONFIGS  # noqa: E402
from simulator.platform import area  # noqa: E402

OUT = HERE / "arch"
TEMPLATE = OUT / "nl_dpe_128x64_auto.xml"
SMOKE: list = []                    # extra sizes outside the design space (64x64 was one until it joined, 2026-10-23)
STAGED = CONFIGS + SMOKE
CONTROL, STATUS = area.TILE["control_pins"], area.TILE["status_pins"]
FC_LINE = '\t\t<fc in_type="frac" in_val="0.15" out_type="frac" out_val="0.10" />\n'
FC_BLOCK = ('\t\t<fc in_type="frac" in_val="0.15" out_type="frac" out_val="0.10">\n'
            '\t\t  <!-- clock pins do not connect to local routing (as the CLB tile) -->\n'
            '\t\t  <fc_override fc_type="frac" fc_val="0" port_name="clk"/>\n'
            '\t\t</fc>\n')


def sub(text: str, pattern: str, repl, count: int) -> str:
    """re.sub that must match exactly `count` times."""
    new, n = re.subn(pattern, repl, text)
    if n != count:
        raise ValueError(f"{pattern!r}: {n} matches in the template, expected {count}")
    return new


def blocks(text: str) -> tuple:
    """The line ranges [a, b) of the wc tile and of the wc pb_type in a file's lines."""
    lines = text.splitlines(keepends=True)
    t0 = next(i for i, x in enumerate(lines) if re.match(r'\s*<tile name="wc" height=', x))
    t1 = next(i for i in range(t0, len(lines)) if "</tile>" in lines[i]) + 1
    p0 = next(i for i, x in enumerate(lines) if '<pb_type name="wc">' in x)
    depth, p1 = 0, None
    for i in range(p0, len(lines)):
        depth += lines[i].count("<pb_type") - lines[i].count("</pb_type>")
        if depth == 0:
            p1 = i + 1
            break
    return (t0, t1), (p0, p1)


def staged_text(R: int, C: int, template: str) -> str:
    """The XML of R x C from the template, a row-parallel XML: the wc tile's size and area, the data pin counts and
    the directs' bit indices follow R and C; everything else is the template's."""
    w, h = area.footprint(R, C)
    a = round(area.blockAreaMwta(R, C))
    r0 = int(re.search(r'<input name="data_in" num_pins="(\d+)"', template).group(1))     # the template's R
    c0 = int(re.search(r'<output name="data_out" num_pins="(\d+)"', template).group(1))   # and its C
    t = sub(template, r'<tile name="wc" height="\d+" width="\d+" area="\d+">',
            f'<tile name="wc" height="{h}" width="{w}" area="{a}">', 1)
    t = sub(t, f'<input name="I" num_pins="{r0}" equivalent="none"/>',
            f'<input name="I" num_pins="{R}" equivalent="none"/>', 2)
    t = sub(t, f'<output name="O" num_pins="{c0 + STATUS}"/>', f'<output name="O" num_pins="{C + STATUS}"/>', 2)
    t = sub(t, f'<input name="in" num_pins="{r0 + CONTROL}"/>', f'<input name="in" num_pins="{R + CONTROL}"/>', 1)
    t = sub(t, f'<output name="out" num_pins="{c0 + STATUS}"/>', f'<output name="out" num_pins="{C + STATUS}"/>', 1)
    t = sub(t, f'<input name="data_in" num_pins="{r0}"', f'<input name="data_in" num_pins="{R}"', 1)
    t = sub(t, f'<output name="data_out" num_pins="{c0}"', f'<output name="data_out" num_pins="{C}"', 1)

    def shift(m, port: str, old: int, new: int) -> str:
        hi, lo = int(m.group(1)), m.group(2)
        if lo is not None and (hi, int(lo)) == (old - 1, 0):        # the data bits
            return f"wc_slice.{port}[{new - 1}:0]"
        if hi < old:
            raise ValueError(f"wc_slice.{port}[{m.group(0)}] cuts the data bits")
        return f"wc_slice.{port}[{hi - old + new}" + (f":{int(lo) - old + new}]" if lo is not None else "]")
    # the directs: the data range, the control bits (the 2-bit nl_dpe_control, 5 single bits) and the two wc-level
    # directs on wc_slice.in; the data range and the 5 status bits on wc_slice.out
    t = sub(t, r"wc_slice\.in\[(\d+)(?::(\d+))?\]", lambda m: shift(m, "in", r0, R), 1 + 6 + 2)
    return sub(t, r"wc_slice\.out\[(\d+)(?::(\d+))?\]", lambda m: shift(m, "out", c0, C), 1 + 5)


def ranges(root, port: str) -> list:
    """Every bit of wc_slice.<port> that a direct inside wc_slice connects to nl_dpe (in: the direct's input end; out:
    its output end), in order."""
    bits = []
    for d in root.iter("direct"):
        end = d.get("input") if port == "in" else d.get("output")
        other = d.get("output") if port == "in" else d.get("input")
        m = re.fullmatch(rf"wc_slice\.{port}\[(\d+)(?::(\d+))?\]", end or "")
        if m and (other or "").startswith("nl_dpe."):
            hi, lo = int(m.group(1)), int(m.group(2) if m.group(2) is not None else m.group(1))
            bits += list(range(lo, hi + 1))
    return sorted(bits)


def check_text(R: int, C: int, text: str, template: str) -> list:
    """The problems of one staged XML (empty: it is right)."""
    probs = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        return [f"not XML: {e}"]
    w, h = area.footprint(R, C)
    tile = next(x for x in root.iter("tile") if x.get("name") == "wc")
    if (int(tile.get("width")), int(tile.get("height"))) != (w, h):
        probs.append(f"tile {tile.get('width')} x {tile.get('height')} (W x H), the rule gives {w} x {h}")
    if int(tile.get("area")) != round(area.blockAreaMwta(R, C)):
        probs.append(f"tile area {tile.get('area')} MWTA != {round(area.blockAreaMwta(R, C))}")
    sub_tile = tile.find("sub_tile")
    pins = {x.get("name"): int(x.get("num_pins")) for x in sub_tile if x.tag in ("input", "output", "clock")}
    want = {"I": R, "IC": CONTROL, "O": C + STATUS, "reset": 1, "clk": 1}
    if pins != want:
        probs.append(f"tile pins {pins} != {want}")
    eq = {x.get("name"): x.get("equivalent") for x in sub_tile if x.tag == "input"}
    if (eq.get("I"), eq.get("IC")) != ("none", "none"):
        probs.append(f"tile pin equivalence I {eq.get('I')}, IC {eq.get('IC')} (no swappable pins: VPR's post-route "
                     "pin fix-up aborts on them, 2026-10-23)")
    fc = sub_tile.find("fc")
    over = {o.get("port_name"): o.get("fc_val") for o in fc.findall("fc_override")}
    if (fc.get("in_val"), fc.get("out_val"), over) != ("0.15", "0.10", {"clk": "0"}):
        probs.append(f"Fc {fc.get('in_val')} / {fc.get('out_val')}, overrides {over}")
    if sub_tile.find("pinlocations").get("pattern") != "spread" or \
            sub_tile.find("switchblock_locations").get("pattern") != "all":
        probs.append("pin locations or switch blocks changed")
    pbs = {pb.get("name"): pb for pb in root.iter("pb_type")}
    io = lambda pb: {x.get("name"): int(x.get("num_pins")) for x in pb if x.tag in ("input", "output")}  # noqa: E731
    if {k: v for k, v in io(pbs["wc"]).items() if k in ("I", "IC", "O")} != {"I": R, "IC": CONTROL, "O": C + STATUS}:
        probs.append(f"pb_type wc {io(pbs['wc'])}")
    peq = {x.get("name"): x.get("equivalent") for x in pbs["wc"] if x.tag == "input"}
    if (peq.get("I"), peq.get("IC")) != ("none", "none"):
        probs.append(f"pb_type wc pin equivalence I {peq.get('I')}, IC {peq.get('IC')} (no swappable pins)")
    top = {d.get("name"): (d.get("input"), d.get("output")) for d in root.iter("direct")
           if d.get("name") in ("wc_data_in", "wc_ctrl_in")}
    if top != {"wc_data_in": ("wc.I", f"wc_slice.in[{R - 1}:0]"),
               "wc_ctrl_in": ("wc.IC", f"wc_slice.in[{R + CONTROL - 1}:{R}]")}:
        probs.append(f"the wc-level input directs {top}")
    if {k: v for k, v in io(pbs["wc_slice"]).items() if k in ("in", "out")} != {"in": R + CONTROL, "out": C + STATUS}:
        probs.append(f"pb_type wc_slice {io(pbs['wc_slice'])}")
    dpe = io(pbs["nl_dpe"])
    if (dpe["data_in"], dpe["data_out"]) != (R, C):
        probs.append(f"nl_dpe data_in {dpe['data_in']}, data_out {dpe['data_out']}")
    if ranges(root, "in") != list(range(R + CONTROL)) or ranges(root, "out") != list(range(C + STATUS)):
        probs.append("the directs do not cover wc_slice.in / out exactly once, in order")
    ctl = {d.get("output").split(".")[1]: d.get("input") for d in root.iter("direct")
           if (d.get("input") or "").startswith("wc_slice.in[")}
    if ctl.get("data_in") != f"wc_slice.in[{R - 1}:0]" or ctl.get("load_input_reg") != f"wc_slice.in[{R + 6}]":
        probs.append(f"data or controls on the wrong bits: {ctl}")
    # the diff from the template: inside the wc tile and the wc pb_type only, on the lines the rule names
    a, b = template.splitlines(keepends=True), text.splitlines(keepends=True)
    (ta0, ta1), (pa0, pa1) = blocks(template)
    allowed = re.compile(r'\s*(<tile name="wc" height=|<input name="(I|IC|in|data_in)" num_pins=|'
                         r'<output name="(O|out|data_out)" num_pins=|'
                         r'<direct name="(dpe_(in2in|out2out)_|wc_(data|ctrl)_in")|<fc |</fc>|'
                         r'<fc_override fc_type="frac" fc_val="0" port_name="clk"/>|<!-- clock pins do not connect)')
    for op, i0, i1, j0, j1 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        inside = all(ta0 <= i < ta1 or pa0 <= i < pa1 for i in range(i0, max(i1, i0 + 1)))
        lines = a[i0:i1] + b[j0:j1]
        if not inside or not all(allowed.match(x) for x in lines):
            probs.append(f"a stray edit at template line {i0 + 1}: {(a[i0:i1] or b[j0:j1])[0].strip()[:90]}")
    return probs


def write() -> int:
    template = TEMPLATE.read_text(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    for R, C in STAGED:
        path = OUT / f"nl_dpe_{R}x{C}_auto.xml"
        path.write_text(staged_text(R, C, template), encoding="utf-8")
        w, h = area.footprint(R, C)
        print(f"  {path.name}: tile {h} x {w} (H x W) = {w * h} cells, I {R + CONTROL}, O {C + STATUS}")
    print(f"gen_arch: {len(STAGED)} XMLs -> {OUT}")
    return 0


def check_set(texts: dict, template: str) -> list:
    """The problems of a staged set {(R, C): its text, or None when the file is missing} (empty: all right)."""
    probs = []
    for (R, C), text in texts.items():
        name = f"nl_dpe_{R}x{C}_auto.xml"
        if text is None:
            probs.append(f"{name}: missing")
            continue
        probs += [f"{name}: {x}" for x in check_text(R, C, text, template)]
        if text != staged_text(R, C, template):
            probs.append(f"{name}: differs from what the generator writes now")
    return probs


def check() -> int:
    template = TEMPLATE.read_text(encoding="utf-8")
    texts = {}
    for R, C in STAGED:
        path = OUT / f"nl_dpe_{R}x{C}_auto.xml"
        texts[(R, C)] = path.read_text(encoding="utf-8") if path.exists() else None
    fails = check_set(texts, template)
    names = {f"nl_dpe_{R}x{C}_auto.xml" for R, C in STAGED}
    stray = sorted(x.name for x in OUT.glob("*.xml") if x.name not in names)
    if stray:
        fails.append(f"files outside the design space and the smoke points: {stray}")
    print(f"  staged   {'PASS' if not fails else 'FAIL'}  {sum(v is not None for v in texts.values())} of "
          f"{len(STAGED)} XMLs: footprint, area, pins, Fc, directs, the diff from the template")
    # planted faults: each must give at least one problem
    good = staged_text(256, 128, template)
    w, h = area.footprint(256, 128)
    faults = {
        "a wrong height": good.replace(f'height="{h}" width="{w}"', f'height="{h + 1}" width="{w}"'),
        "a wrong pin count": good.replace('<input name="I" num_pins="256" equivalent="none"/>',
                                          '<input name="I" num_pins="255" equivalent="none"/>', 1),
        "the data pins swappable": good.replace('<input name="I" num_pins="256" equivalent="none"/>',
                                                '<input name="I" num_pins="256" equivalent="full"/>', 1),
        "the control pins swappable": good.replace(f'<input name="IC" num_pins="{CONTROL}" equivalent="none"/>',
                                                   f'<input name="IC" num_pins="{CONTROL}" equivalent="full"/>', 1),
        "the controls on the data pins": good.replace('output="wc_slice.in[262:256]"/>', 'output="wc_slice.in[6:0]"/>'),
        "a shifted control bit": good.replace(f'input="wc_slice.in[{256 + 6}]"', f'input="wc_slice.in[{256 + 7}]"'),
        "the old pre-bit-plane data": good.replace('<input name="data_in" num_pins="256"', '<input name="data_in" num_pins="40"'),
        "a stray edit (a CLB pin)": good.replace('<input name="I1" num_pins="15" equivalent="full"/>',
                                                 '<input name="I1" num_pins="16" equivalent="full"/>'),
        "a stray edit (the clock's Fc)": good.replace('fc_val="0" port_name="clk"/>\n\t\t</fc>',
                                                      'fc_val="0.1" port_name="clk"/>\n\t\t</fc>'),
        "a stray edit (a delay)": good.replace('max="2.14e-9"', 'max="2.15e-9"', 1),
    }
    caught = []
    for name, text in faults.items():
        if text == good:
            fails.append(f"planted fault {name!r} did not change the text")
        elif check_text(256, 128, text, template):
            caught.append(name)
        else:
            fails.append(f"planted fault {name!r} passed the check")
    if any("missing" in x for x in check_set({**texts, (256, 128): None}, template)):
        caught.append("a missing file")
    else:
        fails.append("planted fault 'a missing file' passed the check")
    print(f"  faults   {'PASS' if len(caught) == len(faults) + 1 else 'FAIL'}  {len(caught)} of {len(faults) + 1} "
          f"planted faults caught: {', '.join(caught)}")
    for f in fails:
        print("    " + f)
    print(f"gen_arch --check: {'ALL PASS' if not fails else f'{len(fails)} failures'}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    return check() if ap.parse_args().check else write()


if __name__ == "__main__":
    sys.exit(main())
