#!/usr/bin/env python3
"""gen_arch.py — the row-parallel arch XMLs (docs/experiment.md §2.1, §5.1; user, 2026-10-22), staged in vtr/arch_rowpar/.

The template is the live vtr/arch/nl_dpe_128x64_auto.xml (v2's; the 12 live XMLs differ only in the wc tile's height,
width and area). One file per DSE config (dse/space.py CONFIGS), nl_dpe_<R>x<C>_auto.xml, changes only:
  the wc tile      height H and width W (simulator/platform/area.py footprint); area = the block's logic + local
                   crossbar in MWTA (area.blockAreaMwta); I = R + 7, O = C + 5; the clock's fc_override 0, as the CLB's
  the wc pb_type   I and O, and wc_slice's in and out
  nl_dpe           data_in R pins, data_out C pins; the directs: the data at in[R-1:0] / out[C-1:0], the 7 controls at
                   in[R..R+6], the 5 status bits at out[C..C+4] (the template's order)
Everything else stays byte-identical: pinlocations spread, the CLB's Fc 0.15 / 0.10, switchblock_locations all, the
layout, the memory, the delays. VTR keeps the live XMLs until the RTL has the row-parallel ports.
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

TEMPLATE = HERE / "arch" / "nl_dpe_128x64_auto.xml"
OUT = HERE / "arch_rowpar"
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
    w, h = area.footprint(R, C)
    a = round(area.blockAreaMwta(R, C))
    data_in = int(re.search(r'<input name="data_in" num_pins="(\d+)"', template).group(1))
    data_out = int(re.search(r'<output name="data_out" num_pins="(\d+)"', template).group(1))
    pin_in, pin_out = data_in + CONTROL, data_out + STATUS
    t = sub(template, r'<tile name="wc" height="\d+" width="\d+" area="\d+">',
            f'<tile name="wc" height="{h}" width="{w}" area="{a}">', 1)
    t = sub(t, f'<input name="I" num_pins="{pin_in}"/>', f'<input name="I" num_pins="{R + CONTROL}"/>', 2)
    t = sub(t, f'<output name="O" num_pins="{pin_out}"/>', f'<output name="O" num_pins="{C + STATUS}"/>', 2)
    t = sub(t, f'<input name="in" num_pins="{pin_in}"/>', f'<input name="in" num_pins="{R + CONTROL}"/>', 1)
    t = sub(t, f'<output name="out" num_pins="{pin_out}"/>', f'<output name="out" num_pins="{C + STATUS}"/>', 1)
    t = sub(t, f'<input name="data_in" num_pins="{data_in}"', f'<input name="data_in" num_pins="{R}"', 1)
    t = sub(t, f'<output name="data_out" num_pins="{data_out}"', f'<output name="data_out" num_pins="{C}"', 1)

    def shift(m, port: str, old: int, new: int) -> str:
        hi, lo = int(m.group(1)), m.group(2)
        if lo is not None and (hi, int(lo)) == (old - 1, 0):        # the data bits
            return f"wc_slice.{port}[{new - 1}:0]"
        if hi < old:
            raise ValueError(f"wc_slice.{port}[{m.group(0)}] cuts the data bits")
        return f"wc_slice.{port}[{hi - old + new}" + (f":{int(lo) - old + new}]" if lo is not None else "]")
    t = sub(t, r"wc_slice\.in\[(\d+)(?::(\d+))?\]", lambda m: shift(m, "in", data_in, R), 1 + 6)
    t = sub(t, r"wc_slice\.out\[(\d+)(?::(\d+))?\]", lambda m: shift(m, "out", data_out, C), 1 + 5)
    return sub(t, re.escape(FC_LINE), FC_BLOCK.replace("\\", "\\\\"), 1)


def ranges(root, port: str) -> list:
    """Every bit of wc_slice.<port> that a direct inside nl_dpe's parent connects, in order."""
    bits = []
    for d in root.iter("direct"):
        for end in (d.get("input"), d.get("output")):
            m = re.fullmatch(rf"wc_slice\.{port}\[(\d+)(?::(\d+))?\]", end or "")
            if m:
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
    want = {"I": R + CONTROL, "O": C + STATUS, "reset": 1, "clk": 1}
    if pins != want:
        probs.append(f"tile pins {pins} != {want}")
    fc = sub_tile.find("fc")
    over = {o.get("port_name"): o.get("fc_val") for o in fc.findall("fc_override")}
    if (fc.get("in_val"), fc.get("out_val"), over) != ("0.15", "0.10", {"clk": "0"}):
        probs.append(f"Fc {fc.get('in_val')} / {fc.get('out_val')}, overrides {over}")
    if sub_tile.find("pinlocations").get("pattern") != "spread" or \
            sub_tile.find("switchblock_locations").get("pattern") != "all":
        probs.append("pin locations or switch blocks changed")
    pbs = {pb.get("name"): pb for pb in root.iter("pb_type")}
    io = lambda pb: {x.get("name"): int(x.get("num_pins")) for x in pb if x.tag in ("input", "output")}  # noqa: E731
    if {k: v for k, v in io(pbs["wc"]).items() if k in ("I", "O")} != {"I": R + CONTROL, "O": C + STATUS}:
        probs.append(f"pb_type wc {io(pbs['wc'])}")
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
    allowed = re.compile(r'\s*(<tile name="wc" height=|<input name="(I|in|data_in)" num_pins=|<output name="(O|out|data_out)"'
                         r' num_pins=|<direct name="dpe_(in2in|out2out)_|<fc |</fc>|<fc_override fc_type="frac" fc_val="0" '
                         r'port_name="clk"/>|<!-- clock pins do not connect)')
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
    for R, C in CONFIGS:
        path = OUT / f"nl_dpe_{R}x{C}_auto.xml"
        path.write_text(staged_text(R, C, template), encoding="utf-8")
        w, h = area.footprint(R, C)
        print(f"  {path.name}: tile {h} x {w} (H x W) = {w * h} cells, I {R + CONTROL}, O {C + STATUS}")
    print(f"gen_arch: {len(CONFIGS)} XMLs -> {OUT}")
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
    for R, C in CONFIGS:
        path = OUT / f"nl_dpe_{R}x{C}_auto.xml"
        texts[(R, C)] = path.read_text(encoding="utf-8") if path.exists() else None
    fails = check_set(texts, template)
    names = {f"nl_dpe_{R}x{C}_auto.xml" for R, C in CONFIGS}
    stray = sorted(x.name for x in OUT.glob("*.xml") if x.name not in names)
    if stray:
        fails.append(f"files outside the design space: {stray}")
    print(f"  staged   {'PASS' if not fails else 'FAIL'}  {sum(v is not None for v in texts.values())} of "
          f"{len(CONFIGS)} XMLs: footprint, area, pins, Fc, directs, the diff from the template")
    # planted faults: each must give at least one problem
    good = staged_text(256, 128, template)
    w, h = area.footprint(256, 128)
    faults = {
        "a wrong height": good.replace(f'height="{h}" width="{w}"', f'height="{h + 1}" width="{w}"'),
        "a wrong pin count": good.replace(f'<input name="I" num_pins="{256 + CONTROL}"/>',
                                          f'<input name="I" num_pins="{256 + CONTROL - 1}"/>', 1),
        "a shifted control bit": good.replace(f'input="wc_slice.in[{256 + 6}]"', f'input="wc_slice.in[{256 + 7}]"'),
        "the old 40-bit data": good.replace('<input name="data_in" num_pins="256"', '<input name="data_in" num_pins="40"'),
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
