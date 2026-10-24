"""Build nl_dpe_head.html (the architecture page) from template2.html, params_block.html, page.js, docs/architecture.md
Frozen 2026-10-23: the page documents the pre-bit-plane RTL of 2026-10-15 (its line references are of that RTL) and the
N_E = 2d mapping; it is not regenerated for the bit-plane head.
and builds.json.

Usage, from the repository root:
  python3 v3/docs/architecture_page/page_data.py v3/work/arch_page/builds.json     # the model's numbers, 864 builds
  python3 v3/docs/architecture_page/build.py                                       # -> v3/work/arch_page/nl_dpe_head.html
The page is published as an artifact; the generated files (builds.json, nl_dpe_head.html, main_script.js) stay in
v3/work/arch_page/. The line references in the page (ip/dimm/rtl/sim/dimm_tile_top.v:N, ...) are remapped by MAP, GMAP
and HMAP below; re-check them after any RTL edit.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent            # the sources: template2.html, params_block.html, page.js
V3 = HERE.parents[1]
OUT = V3 / "work" / "arch_page"                    # the generated files
MAP = {497: 498, 543: 540, 548: 545, 552: 549, 565: 562, 585: 582, 598: 595, 618: 615, 658: 655, 706: 703,
       723: 720, 755: 752, 885: 882, 928: 925, 980: 977, 1031: 1028}      # dimm_tile_top.v after Fmax fix 1
MAP[122] = 123
GMAP = {198: 199, 206: 207, 254: 255, 321: 322, 370: 371}      # gemm_top.v: the section's title line
HMAP = {69: 70}                                                   # att_head.v: the g_cp loop


def remap(s):
    s = re.sub(r"(ip/dimm/rtl/sim/dimm_tile_top\.v:)(\d+)", lambda m: m.group(1) + str(MAP.get(int(m.group(2)), int(m.group(2)))), s)
    s = re.sub(r"(ip/gemm/rtl/sim/gemm_top\.v:)(\d+)", lambda m: m.group(1) + str(GMAP.get(int(m.group(2)), int(m.group(2)))), s)
    return re.sub(r"(top/rtl/sim/att_head\.v:)(\d+)", lambda m: m.group(1) + str(HMAP.get(int(m.group(2)), int(m.group(2)))), s)


t = remap((HERE / "template2.html").read_text())
params = remap((HERE / "params_block.html").read_text())
js = remap((HERE / "page.js").read_text())
md = (V3 / "docs" / "architecture.md").read_text()
data = (OUT / "builds.json").read_text()
for name, s in (("md", md), ("data", data), ("js", js)):
    assert "</script" not in s, name
for ph in ("{{PARAMS}}", "{{MD}}", "{{DATA}}", "{{JS}}"):
    assert t.count(ph) == 1, ph
out = t.replace("{{PARAMS}}", params).replace("{{MD}}", md).replace("{{DATA}}", data).replace("{{JS}}", js)
(OUT / "nl_dpe_head.html").write_text(out)
(OUT / "main_script.js").write_text(js)
print("written", len(out), "bytes ->", OUT / "nl_dpe_head.html")
