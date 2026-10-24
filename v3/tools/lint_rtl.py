#!/usr/bin/env python3
"""lint_rtl.py — the no-runtime-products lint over every VTR twin of v3 (the H1 rule of v2; user 2026-10-10).

Every `*`, `/`, `%` whose operand is a runtime signal (not a parameter / localparam / genvar / loop index / function)
fails: synthesis maps such index arithmetic onto DSP multipliers, which set Fmax. A power-of-two literal operand (a
shift or mask) passes, and so does a line carrying the waiver `pow2: NAME[, NAME]` naming parameters that the module's
domain guard forces to powers of two. The parser helpers are those of v2/experiment/rtl/check_kernel.py (verbatim).
Modules of the unchanged certified files (CERTIFIED) are reported, not failed.

Usage: python3 v3/tools/lint_rtl.py [--self-test]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
CERTIFIED = ()     # unchanged v2 files: products reported, not failed (none since the bit-plane GEMM, 2026-10-23)
MOD_RE = re.compile(r"^\s*module\s+(\w+)")
END_RE = re.compile(r"^\s*endmodule\b")


def modules(text: str) -> list:
    """[(name, block text)] in file order; a block runs from its `module` line through its `endmodule` line."""
    out, cur, name = [], None, None
    for line in text.splitlines(keepends=True):
        if cur is None:
            m = MOD_RE.match(line)
            if m:
                cur, name = [line], m.group(1)
        else:
            cur.append(line)
            if END_RE.match(line):
                out.append((name, "".join(cur)))
                cur = None
    if cur is not None:
        raise ValueError(f"module {name} has no endmodule")
    return out


KEYWORDS = set("""module endmodule input output inout wire reg integer signed parameter localparam genvar generate
endgenerate begin end if else for always assign posedge negedge or and not case endcase default function endfunction
initial while repeat""".split())
WAIVER_RE = re.compile(r"pow2:\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)")


def _strip(text: str) -> str:
    """Comments and attributes blanked (same length, newlines kept, so offsets map to lines)."""
    keep = lambda m: re.sub(r"[^\n]", " ", m.group(0))  # noqa: E731
    text = re.sub(r"/\*.*?\*/", keep, text, flags=re.S)
    text = re.sub(r"(?<!@)\(\*(?!\)).*?\*\)", keep, text, flags=re.S)   # attributes, not @(*)
    text = re.sub(r"//[^\n]*", keep, text)
    return re.sub(r'"[^"\n]*"', keep, text)


def _split0(s: str) -> list:
    """Split at depth-0 commas (outside (), [], {})."""
    out, d, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        if ch == "," and d == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _constants(code: str) -> set:
    """Names that elaborate to constants: parameters, localparams, genvars, loop indices, functions."""
    names = set()
    for m in re.finditer(r"\b(?:parameter|localparam)\b((?:\s+(?:integer|signed))?\s*(?:\[[^\]]*\])?)([^;]*?)(?=;|\)\s*\(|$)",
                         code, re.S):
        for part in _split0(m.group(2)):
            n = re.match(r"\s*(?:parameter\s+|localparam\s+)?(?:\[[^\]]*\]\s*)?([A-Za-z_]\w*)\s*=", part)
            if n:
                names.add(n.group(1))
    for m in re.finditer(r"\bgenvar\b([^;]*);", code):
        names.update(x.strip() for x in m.group(1).split(",") if x.strip())
    names.update(re.findall(r"\bfor\s*\(\s*([A-Za-z_]\w*)\s*=", code))
    names.update(re.findall(r"\bfunction\b(?:\s+(?:integer|signed))?\s*(?:\[[^\]]*\]\s*)?([A-Za-z_]\w*)", code))
    return names


def _match(code: str, j: int, step: int) -> int:
    """Index of the bracket matching code[j] (step -1: j at a closer, scan left; +1: j at an opener, scan right)."""
    pairs = {")": "(", "]": "[", "(": ")", "[": "]"}
    me, other, d = code[j], pairs[code[j]], 0
    while 0 <= j < len(code):
        d += (code[j] == me) - (code[j] == other)
        if d == 0:
            return j
        j += step
    return j - step


def _word(c: str) -> bool:
    return c.isalnum() or c in "_$'"


def _left(code: str, i: int) -> str:
    """The operand ending just before the operator at i: name[..][..], literal, (group) or f(args)."""
    j = i - 1
    while j >= 0 and code[j].isspace():
        j -= 1
    end = j + 1
    while j >= 0:
        if code[j] == ")":
            j = _match(code, j, -1) - 1
            while j >= 0 and _word(code[j]):
                j -= 1
            break
        if code[j] == "]":
            j = _match(code, j, -1) - 1
            continue
        if _word(code[j]):
            while j >= 0 and _word(code[j]):
                j -= 1
        break
    return code[j + 1:end]


def _right(code: str, i: int) -> str:
    """The operand starting just after the operator at i."""
    n, j = len(code), i + 1
    while j < n and code[j].isspace():
        j += 1
    start = j
    if j < n and code[j] in "-~!":
        j += 1
    if j < n and code[j] == "(":
        return code[start:_match(code, j, +1) + 1]
    while j < n and _word(code[j]):
        j += 1
    if j < n and code[j] == "(":
        return code[start:_match(code, j, +1) + 1]
    while j < n and code[j] == "[":
        j = _match(code, j, +1) + 1
    return code[start:j]


def _pow2_literal(op: str) -> bool:
    m = re.fullmatch(r"\s*(?:\d*'[sS]?[dD])?\s*(\d+)\s*", op)
    return bool(m) and int(m.group(1)) > 0 and (int(m.group(1)) & (int(m.group(1)) - 1)) == 0


def runtime_products(block: str) -> list:
    """[(line in block, operator, left, right)] for every `*` `/` `%` with a runtime operand (see the header)."""
    code = _strip(block)
    consts = _constants(code)
    raw_lines = block.split("\n")
    found = []
    for m in re.finditer(r"(?<![*/(])([*/%])(?![*/)])", code):
        i = m.start()
        left, right = _left(code, i), _right(code, i)
        ids = lambda s: [t for t in re.findall(r"(?<![\w$'])([A-Za-z_]\w*)\b(?!\s*\()", s)  # noqa: E731
                         if t not in KEYWORDS]
        runtime = [t for t in ids(left) + ids(right) if t not in consts]
        if not runtime:
            continue
        if m.group(1) in "*/%" and (_pow2_literal(left) or _pow2_literal(right)):
            continue
        line = code.count("\n", 0, i)
        w = WAIVER_RE.search(raw_lines[line])
        if w:
            named = {x.strip() for x in w.group(1).split(",")}
            const_ops = [t for t in ids(left) + ids(right) if t in consts]
            if const_ops and set(const_ops) <= named:
                continue
        found.append((line + 1, m.group(1), left.strip(), right.strip()))
    return found




def lint(files=None) -> tuple:
    files = files or sorted([p for p in (V3 / "ip").glob("*/rtl/vtr/*.v") if p.name != "dpe_blackbox.v"]
                            + list((V3 / "top" / "rtl" / "vtr").glob("*.v")))
    fails, reported = [], []
    for f in files:
        for name, block in modules(f.read_text()):
            for line, op, l, r in runtime_products(block):
                entry = f"{f.relative_to(V3)}:{name}+{line}: {l} {op} {r}"
                (reported if f.name in CERTIFIED else fails).append(entry)
    return files, fails, reported


def self_test() -> int:
    good = "module m #(parameter A = 4) (input [7:0] x, output [9:0] y);\n  assign y = x * 4 + (A * 3) + x[2:0] * 8;\nendmodule\n"
    bad = "module m (input [7:0] x, input [7:0] z, output [15:0] y);\n  assign y = x * z;\nendmodule\n"
    loop = "module m (input [7:0] x, output reg [7:0] y);\n  integer i;\n  always @(*) for (i = 0; i < 4; i = i + 1) y = x + i * 3;\nendmodule\n"
    assert not runtime_products(modules(good)[0][1]), "constants and pow2 literals pass"
    assert runtime_products(modules(bad)[0][1]), "a runtime product fails"
    assert not runtime_products(modules(loop)[0][1]), "a loop index passes"
    star = ("module m #(parameter W = 4) (input [7:0] x, input [7:0] z, output reg [7:0] y, output [15:0] p);\n"
            "  always @(*) y = x;\n  localparam Q = W + 1;\n  assign p = x * z;\n  always @(*) y = z;\nendmodule\n")
    assert runtime_products(modules(star)[0][1]), "code between two always @(*) blocks is linted"
    attr = "module m (input [7:0] x, output [7:0] y);\n  (* keep = 1 *) wire [7:0] k = x;\n  assign y = k * 2;\nendmodule\n"
    assert not runtime_products(modules(attr)[0][1]), "an attribute is skipped"
    print("lint_rtl self-test: ALL PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    files, fails, reported = lint()
    for e in fails:
        print(f"  FAIL {e}")
    print(f"lint_rtl: {len(files)} VTR twins, {len(fails)} runtime products"
          + (f" ({len(reported)} in the certified unchanged files, reported)" if reported else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
