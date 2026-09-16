#!/usr/bin/env python3
"""check_interface.py — machine gate for the frozen DPE primitive interface.

The v2 clean-room primitive (`v2/rtl/dpe_nldpe.v`) may implement a completely
different datapath, but its *interface* must stay exactly the legacy one:

  * ports  : `rtl_flow/vtr/dpe_blackbox.v` (ordered names/directions/widths)
             and `rtl_flow/rtl/dpe_nldpe{,_faithful}.v` (same + `reg` kind)
  * params : legacy superset — every parameter name from the legacy behavior
             model and the legacy faithful model must be declared, with the
             legacy defaults
  * XML    : `nl_dpe/nl_dpe_22nm_auto.xml` `<model name="dpe">` port set

Run:  python3 v2/smoke/check_interface.py
Exit code 0 = interface frozen and conformant; non-zero = drift.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2_RTL = REPO / "v2" / "rtl" / "dpe_nldpe.v"
BLACKBOX = REPO / "rtl_flow" / "vtr" / "dpe_blackbox.v"
LEGACY_MODELS = [
    REPO / "rtl_flow" / "rtl" / "dpe_nldpe.v",
    REPO / "rtl_flow" / "rtl" / "dpe_nldpe_faithful.v",
]
XML = REPO / "nl_dpe" / "nl_dpe_22nm_auto.xml"

# Legacy defaults the v2 superset must reproduce (NL-DPE).
EXPECTED_PARAM_DEFAULTS = {
    "KERNEL_WIDTH": 256,
    "NUM_COLS": 256,
    "DPE_BUF_WIDTH": 40,
    "PRECISION": 8,
    "PIPELINE_DEPTH": 2,
    "ACAM_CYCLES": 1,
    "COMPUTE_CYCLES": 10,
    "ACAM_MODE": 0,
}


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Return the parenthesized group starting at `text[start] == '('`."""
    assert text[start] == "("
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced parentheses")


def _split_top(text: str) -> list[str]:
    """Split on top-level commas (brackets/parens/braces aware)."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def _eval_int(expr: str, params: dict[str, int]) -> int | str:
    """Evaluate a width/default expression against known parameter values."""
    expr = expr.strip()
    if expr in params:
        return params[expr]
    sub = expr
    for name, val in params.items():
        sub = re.sub(rf"\b{re.escape(name)}\b", str(val), sub)
    if not re.fullmatch(r"[0-9+\-*/() ]+", sub):
        return expr
    try:
        return int(eval(sub, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return expr


def parse_module(path: Path) -> dict:
    text = strip_comments(path.read_text())
    m = re.search(r"\bmodule\s+(\w+)", text)
    if not m:
        raise ValueError(f"{path}: no module declaration")
    name, i = m.group(1), m.end()

    params: dict[str, object] = {}
    j = i
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "#":
        k = j + 1
        while k < len(text) and text[k].isspace():
            k += 1
        ptext, i = _balanced(text, k)
        for item in _split_top(ptext):
            pm = re.search(r"parameter\s+(?:\w+\s+)?(\w+)\s*=\s*(.+?)\s*$",
                           item.strip(), flags=re.S)
            if pm:
                params[pm.group(1)] = _eval_int(pm.group(2), params)

    while i < len(text) and text[i].isspace():
        i += 1
    ptext, _ = _balanced(text, i)

    ports = []
    for item in _split_top(ptext):
        item = " ".join(item.split())
        if not item:
            continue
        dm = re.search(r"\b(input|output|inout)\b", item)
        if not dm:
            continue
        km = re.search(r"\b(wire|reg)\b", item)
        wm = re.search(r"\[([^\]]+)\]", item)
        nm = re.search(r"(\w+)\s*$", item)
        if wm:
            sides = wm.group(1).split(":")
            width = tuple(_eval_int(s, params) for s in sides)
        else:
            width = (0, 0)
        ports.append({
            "name": nm.group(1),
            "dir": dm.group(1),
            "kind": km.group(1) if km else "wire",
            "width": width,
            "width_raw": wm.group(1) if wm else None,
        })
    return {"name": name, "params": params, "ports": ports}


def fmt_ports(ports: list[dict]) -> str:
    out = []
    for p in ports:
        w = p["width_raw"] if p["width_raw"] else ""
        out.append(f"{p['dir']} {p['kind']} [{w}] {p['name']} {tuple(p['width'])}")
    return "\n".join(out)


def main() -> int:
    errors: list[str] = []
    v2 = parse_module(V2_RTL)
    bb = parse_module(BLACKBOX)

    print(f"v2 primitive : {V2_RTL.relative_to(REPO)}")
    print(f"blackbox     : {BLACKBOX.relative_to(REPO)}")

    if v2["name"] != "dpe":
        errors.append(f"v2 module must be named `dpe`, got `{v2['name']}`")

    # -- ports: ordered parity vs blackbox (name, dir, width) ----------------
    if len(v2["ports"]) != len(bb["ports"]):
        errors.append(
            f"port count differs: v2={len(v2['ports'])} blackbox={len(bb['ports'])}")
    for a, b in zip(v2["ports"], bb["ports"]):
        if (a["name"], a["dir"], a["width"]) != (b["name"], b["dir"], b["width"]):
            errors.append(
                f"port mismatch vs blackbox: v2 `{a['name']} {a['dir']} "
                f"{a['width']}` != blackbox `{b['name']} {b['dir']} {b['width']}`")

    # -- ports: full parity (kind included) vs the legacy RTL models ---------
    legacy = [(p, parse_module(p)) for p in LEGACY_MODELS]
    for path, mod in legacy:
        if len(v2["ports"]) != len(mod["ports"]):
            errors.append(f"{path.name}: port count differs")
        for a, b in zip(v2["ports"], mod["ports"]):
            if (a["name"], a["dir"], a["kind"], a["width"]) != \
               (b["name"], b["dir"], b["kind"], b["width"]):
                errors.append(
                    f"port mismatch vs {path.name}: v2 `{a['name']} {a['dir']} "
                    f"{a['kind']} {a['width']}` != legacy `{b['name']} "
                    f"{b['dir']} {b['kind']} {b['width']}`")

    # -- parameters: superset + legacy defaults ------------------------------
    v2p = v2["params"]
    for path, mod in legacy:
        for pname in mod["params"]:
            if pname not in v2p:
                errors.append(f"{path.name}: parameter `{pname}` missing from v2")
    for pname, default in EXPECTED_PARAM_DEFAULTS.items():
        if pname not in v2p:
            errors.append(f"parameter `{pname}` missing from v2")
        elif v2p[pname] != default:
            errors.append(
                f"parameter `{pname}` default {v2p[pname]} != legacy {default}")

    # -- VTR arch XML <model name="dpe"> port set ----------------------------
    if XML.exists():
        model = ET.parse(XML).getroot().find(".//model[@name='dpe']")
        if model is None:
            errors.append(f"{XML.name}: <model name=\"dpe\"> not found")
        else:
            xml_ports = set()
            for ip in model.find("input_ports"):
                xml_ports.add((ip.get("name"), "input"))
            for op in model.find("output_ports"):
                xml_ports.add((op.get("name"), "output"))
            v2_ports = {(p["name"], p["dir"]) for p in v2["ports"]}
            if xml_ports != v2_ports:
                errors.append(
                    f"port set differs vs {XML.name}: "
                    f"missing={sorted(xml_ports - v2_ports)} "
                    f"extra={sorted(v2_ports - xml_ports)}")
    else:
        errors.append(f"arch XML not found: {XML}")

    if errors:
        print("\nFAIL — interface drift:")
        for e in errors:
            print(f"  - {e}")
        print("\nv2 port surface:")
        print(fmt_ports(v2["ports"]))
        return 1

    print(f"ports  : {len(v2['ports'])} — ordered parity with blackbox + "
          f"{len(legacy)} legacy models, reg kinds, widths OK")
    print(f"params : superset OK ({', '.join(v2p)})")
    print(f"xml    : port set matches {XML.name}")
    print("\ncheck_interface: PASS — interface frozen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
