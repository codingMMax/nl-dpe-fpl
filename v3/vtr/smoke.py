#!/usr/bin/env python3
"""smoke.py — VTR smoke of one v3 module (PLAN R4; docs/architecture.md §7 gate V): synthesis keeps every block
(the placed `wc` count == the expected crossbar count, nothing optimized away), plus Fmax, CLB, memory and the
critical path.

The circuit is the module's VTR twin followed by a parameter wrapper as the only uninstantiated module (as
v2/experiment/vtr/vtr_tops.py builds it): ports up to WIDE bits are pins; a wider input loads through a shift
register from a 32-bit pin and a wider output is XOR-folded into a 32-bit register, so every output stays observed.
Steps:
  elab   VTR's Parmys front end (synthesis.tcl order) through `opt -full`: the chosen top, the `dpe` count, and every
         inferred memory's port shape (Parmys's RAM mapper aborts on shapes other than 1R1W / 1R2W / 2R1W / 2R2W)
  flow   run_vtr_flow.py (Parmys, ABC, VPR) on the module's arch XML, route channel width 300, clock.sdc, one seed

Usage:
  python3 v3/vtr/smoke.py --module dimm --point qk_bert_tiny_128x64 [--step elab|flow|all] [--seed 11] [--tag T]
Points: bert_tiny_64 (64x64, the adder branch), bert_tiny_128x64 (the smoke point since 2026-10-23), bert_tiny_128;
the DIMM's are qk_/pv_ + the size; the GEMM's fc_<K>x<N>_<R>x<C>.
Runs and logs: v3/work/vtr/<top>/ (with --tag: <top>_<T>/, a run beside another); the result line in result.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(HERE))
from tools.check_twins import header, modules  # noqa: E402
from run_vtr import build_vtr_command, find_vpr_log, parse_metrics, parse_resources  # noqa: E402

VTR_ROOT = Path(os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
YOSYS = VTR_ROOT / "build" / "bin" / "yosys"
WORK = V3 / "work" / "vtr" / "smoke"
SDC = HERE / "clock.sdc"
ROUTE_CHAN_WIDTH = 300
WIDE = 2048            # ports up to 2048 bits are pins: no DSE build gets wrapper logic (FC K = 2048, C = 256)
ARCH_DIR = HERE / "arch"            # the row-parallel XMLs (vtr/gen_arch.py; one folder since 2026-10-23)


def _gemm_points() -> dict:
    from simulator.models.fc import extent
    out = {}
    for K, N, R, C in ((2048, 256, 256, 128), (512, 128, 128, 64)):
        ru, cu = extent(K, N, R, C, "used")
        out[f"fc_{K}x{N}_{R}x{C}"] = dict(K=K, N=N, R=R, C=C, P=8, ROWS_USED=ru, COLS_USED=cu)
    return out


def _heads() -> dict:
    """BERT-Tiny (n 128, d_m 128, d 64) at the fixed mapping (headParams: N_E = d, N_EP = 2d, N_OC by the floor rule,
    P_PROJ 1) on the three small arrays: 64x64 (the functional-validation size: R = C, the adder branch), 128x64 (the
    smoke point since 2026-10-23: the smallest array with the add in the crossbar's rows, a DSE config) and 128x128
    (the smoke size until 2026-10-22)."""
    from simulator.models.attentionHead import headParams
    return {"64": headParams(128, 128, 64, 64, 64, P_PROJ=1), "128x64": headParams(128, 128, 64, 128, 64, P_PROJ=1),
            "128": headParams(128, 128, 64, 128, 128, P_PROJ=1)}


def _dimm_points() -> dict:
    """The head's two DIMM instances (ip/dimm/cases.py qkParams / pvParams) at every size of _heads()."""
    from ip.dimm.cases import pvParams, qkParams
    out = {}
    for k, h in _heads().items():
        out[f"qk_bert_tiny_{k}"] = qkParams(h.S, h.D, h.BKV, h.R, h.C, h.N_E, pProj=h.P_PROJ, runs=h.RB, blocks=h.B)
        out[f"pv_bert_tiny_{k}"] = pvParams(h.S, h.D, h.BKV, h.R, h.C, h.N_EP, nExp=h.N_EXP, pProj=h.P_PROJ, blocks=h.B)
    return out


def _softmax_points() -> dict:
    """The head's online softmax at every size of _heads(): S_CH = the QK DIMM's output channels, N_E / d."""
    return {f"bert_tiny_{k}": dict(S=h.S, BKV=h.BKV, N=h.NSEQ, R=h.R, C=h.C, N_EXP=h.N_EXP, S_CH=h.NG_QK, LP_CAP=2,
                                   LA_CAP=2, LL_CAP=1) for k, h in _heads().items()}


def _acc_points() -> dict:
    """The head's accumulator at every size of _heads(): P_CH = the PV DIMM's output channels, N_EP / B_kv."""
    return {f"bert_tiny_{k}": dict(S=h.S, D=h.D, R=h.R, C=h.C, N_OC=h.N_OC, NBLK=h.B, SO=h.LN, LN=h.LN, P_CH=h.NG_PV,
                                   VWP=h.acc.VWP) for k, h in _heads().items()}


HEAD_KEYS = ("NSEQ", "DM", "D", "BKV", "R", "C", "P", "N_EXP", "N_E", "N_EP", "N_OC", "P_PROJ", "X_S")   # BUF left the
#   model 2026-10-22 and the RTL 2026-10-23 (the bit-plane pass)


def head_point(hp) -> dict:
    """att_head's top parameters from the head's parameters (simulator/models/attentionHead.py headParams), plus the
    expected crossbar count."""
    return dict({k: getattr(hp, k) for k in HEAD_KEYS}, n_xbar=hp.nXbar)


def _head_points() -> dict:
    """att_head at every size of _heads()."""
    return {f"bert_tiny_{k}": head_point(h) for k, h in _heads().items()}


_VTR_MODS = [V3 / "ip/lib/rtl/vtr/att_fifo.v", V3 / "ip/gemm/rtl/vtr/gemm_top.v", V3 / "ip/dimm/rtl/vtr/dimm_tile_top.v",
             V3 / "ip/softmax/rtl/vtr/softmax_online_top.v", V3 / "ip/softmax/rtl/vtr/softmax_online_acc.v",
             V3 / "top/rtl/vtr/att_ctrl.v"]

MODULES = {
    "gemm": dict(file=V3 / "ip/gemm/rtl/vtr/gemm_top.v", top="gemm_top", points=_gemm_points,
                 expected=lambda p: (-(-p["K"] // p["R"])) * (-(-p["N"] // p["C"]))),
    "dimm": dict(file=V3 / "ip/dimm/rtl/vtr/dimm_tile_top.v", top="dimm_tile_top", points=_dimm_points,
                 expected=lambda p: p["N_E"] + (p["A_CH"] if p["A_DOM"] == 0 else 0)
                 + (p["B_CH"] if p["B_DOM"] == 0 else 0)),
    "softmax": dict(file=V3 / "ip/softmax/rtl/vtr/softmax_online_top.v", top="softmax_online_top",
                    points=_softmax_points, expected=lambda p: p["N_EXP"] + 2),
    "acc": dict(file=V3 / "ip/softmax/rtl/vtr/softmax_online_acc.v", top="softmax_online_acc",
                points=_acc_points, expected=lambda p: 2 * p["N_OC"], deps=[V3 / "ip/lib/rtl/vtr/att_fifo.v"]),
    "head": dict(file=V3 / "top/rtl/vtr/att_head.v", top="att_head", points=_head_points,
                 expected=lambda p: p["n_xbar"], deps=_VTR_MODS),
}


def _clog2(x) -> int:
    return max(0, (int(x) - 1).bit_length())


def vexpr(s: str) -> str:
    """A Verilog constant expression as Python: `c ? a : b`, $clog2, integer division."""
    s = s.strip()
    if s.startswith("("):
        depth = 0
        for i, ch in enumerate(s):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth == 0:
                if i == len(s) - 1:
                    return "(" + vexpr(s[1:-1]) + ")"
                break
    depth, q = 0, None
    for i, ch in enumerate(s):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "?" and depth == 0 and q is None:
            q = i
        elif ch == ":" and depth == 0 and q is not None:
            return f"(({vexpr(s[q + 1:i])}) if ({vexpr(s[:q])}) else ({vexpr(s[i + 1:])}))"
    return s.replace("$clog2", "clog2").replace("//", "/").replace("/", "//")


def veval(s: str, env: dict) -> int:
    return int(eval(vexpr(s), {"clog2": _clog2}, dict(env)))


def eval_width(expr: str, params: dict) -> int:
    """A port width expression like `[A_CH*BUF-1:0]` evaluated with the module's parameters."""
    if not expr:
        return 1
    m = re.fullmatch(r"\[(.*):(.*)\]", expr)
    return veval(m.group(1), params) - veval(m.group(2), params) + 1


def circuit(module: str, point: str, params: dict | None = None) -> tuple:
    """(top, circuit text, expected crossbars, parameters) of a module at a point of its table, or at `params` (the
    point then only names the top, e.g. the DSE builds of v3/dse/vtr_dse.py)."""
    spec = MODULES[module]
    p = spec["points"]()[point] if params is None else params
    src = "".join(f.read_text() + "\n" for f in spec.get("deps", [])) + spec["file"].read_text()   # the twin + its files
    params, ports = header(modules(src)[spec["top"]])
    # every parameter's value (defaults evaluated in order, overridden by the point)
    env = {}
    for name, default in params:
        env[name] = p[name] if name in p else veval(default, env)
    top = f"{spec['top']}_{point}"
    pins, body, conn = [], [], []
    for d, kind, w, n in ports:
        width = eval_width(w, env)
        if width > WIDE:
            if d == "input":
                pins.append(("input", n + "_n", 32))
                body += [f"    reg [{width - 1}:0] {n}_r;",
                         f"    always @(posedge clk) {n}_r <= {{{n}_r[{width - 33}:0], {n}_n}};"]
                conn.append(f".{n}({n}_r)")
            else:
                pins.append(("output", n + "_n", 32))
                fold = " ^ ".join(f"{n}_w[{32 * k + 31}:{32 * k}]" for k in range(width // 32))
                body += [f"    wire [{width - 1}:0] {n}_w;", f"    reg  [31:0] {n}_f;",
                         f"    always @(posedge clk) {n}_f <= {fold};", f"    assign {n}_n = {n}_f;"]
                conn.append(f".{n}({n}_w)")
        else:
            pins.append((d, n, width))
            conn.append(f".{n}({n})")
    decl = ",\n".join(f"    {d:<6} wire {'' if w == 1 else f'[{w - 1}:0] '}{n}" for d, n, w in pins)
    pmap = ", ".join(f".{k}({v})" for k, v in p.items() if k in dict(params))
    wrapper = (f"// VTR top for {module} {point} — generated by v3/vtr/smoke.py\n"
               f"module {top} (\n{decl}\n);\n" + "".join(x + "\n" for x in body)
               + f"    {spec['top']} #({pmap}) u_mod (\n        " + ",\n        ".join(conn) + "\n    );\nendmodule\n")
    return top, src + "\n" + wrapper, spec["expected"](p), p


def arch_for(p: dict) -> Path:
    """The config's row-parallel arch XML; its crossbar block must have R data_in and C data_out pins."""
    path = ARCH_DIR / f"nl_dpe_{p['R']}x{p['C']}_auto.xml"
    text = path.read_text(encoding="utf-8")
    pins = (re.search(r'<input name="data_in" num_pins="(\d+)"', text),
            re.search(r'<output name="data_out" num_pins="(\d+)"', text))
    if not all(pins) or (int(pins[0].group(1)), int(pins[1].group(1))) != (p["R"], p["C"]):
        raise ValueError(f"{path.name}: the dpe block's data pins are not {p['R']} in, {p['C']} out")
    return path


def elab(path: Path, arch: Path, work: Path, run=None) -> dict:
    """run(cmd, cwd) -> exit code: how Yosys is started (default: plainly; the DSE driver caps its memory)."""
    script = work / "elab.ys"
    top_f, dpe_f, mem_f = work / "top.txt", work / "dpe.txt", work / "mems.txt"
    script.write_text("\n".join([
        "plugin -i parmys", f"parmys_arch -a {arch}", "read_verilog -nomem2reg +/parmys/vtr_primitives.v",
        "setattr -mod -set keep_hierarchy 1 single_port_ram", "setattr -mod -set keep_hierarchy 1 dual_port_ram",
        f"read_verilog -sv -nolatches {path}", "scc -select", "select -assert-none %", "select -clear",
        "hierarchy -check -auto-top -purge_lib", f"tee -q -o {top_f} select -list A:top",
        "opt_expr", "opt_clean", "check", "opt -nodffe -nosdff", "proc -norom", "fsm", "opt", "wreduce", "peepopt",
        "opt_clean", "share", "opt -full", "memory -nomap", "flatten", "opt -full",
        f"tee -q -o {dpe_f} select -count t:dpe", f"tee -q -o {mem_f} dump t:$mem_v2", ""]))
    t0 = time.time()
    cmd = [str(YOSYS), "-q", "-l", str(work / "elab.log"), "-s", str(script)]
    rc = (run(cmd, work) if run is not None else
          subprocess.run(cmd, cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode)
    res = dict(rc=rc, wall_s=round(time.time() - t0, 1))
    if rc == 0:
        tops = [ln.strip() for ln in top_f.read_text().splitlines() if ln.strip() and "/" not in ln]
        res["top"] = tops[0] if len(tops) == 1 else tops
        m = re.search(r"(\d+)", dpe_f.read_text())
        res["dpe"] = int(m.group(1)) if m else None
        mems = []
        for m in re.finditer(r"cell \$mem_v2 (\S+)\n(.*?)\n\s*end", mem_f.read_text(), re.S):
            body = m.group(2)
            g = lambda k: int(re.search(rf"parameter \\{k} (\d+)", body).group(1))  # noqa: E731
            mems.append(dict(name=m.group(1), size=g("SIZE"), width=g("WIDTH"), rd=g("RD_PORTS"), wr=g("WR_PORTS")))
        res["mems"] = mems
    return res


def flow(path: Path, arch: Path, work: Path, seed: int) -> dict:
    vtr_dir = work / f"s{seed}"
    vtr_dir.mkdir(parents=True, exist_ok=True)
    py = VTR_ROOT / ".venv" / "bin" / "python"
    cmd = build_vtr_command(VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py", py if py.is_file() else None,
                            path, arch, ROUTE_CHAN_WIDTH, SDC, vtr_dir, seed=seed)
    t0 = time.time()
    with open(work / f"flow_s{seed}.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=vtr_dir, stdout=fh, stderr=subprocess.STDOUT).returncode
    res = dict(rc=rc, wall_s=round(time.time() - t0, 1), dir=str(vtr_dir))
    log = find_vpr_log(vtr_dir)
    if log.exists():
        try:
            res["wirelength"], res["fmax_mhz"] = parse_metrics(log)
        except (ValueError, FileNotFoundError) as e:
            res["metric_error"] = str(e)
        res["resources"] = parse_resources(log)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", required=True, choices=sorted(MODULES))
    ap.add_argument("--point", required=True)
    ap.add_argument("--step", default="all", choices=("elab", "flow", "all"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="", help="work folder suffix (a run beside another of the same top)")
    a = ap.parse_args()
    top, text, expected, p = circuit(a.module, a.point)
    work = WORK / (f"{top}_{a.tag}" if a.tag else top)
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{top}.v"
    path.write_text(text)
    arch = arch_for(p)
    out = dict(module=a.module, point=a.point, top=top, expected_dpe=expected, params=p, arch=str(arch))
    ok = True
    if a.step in ("elab", "all"):
        e = elab(path, arch, work)
        out["elab"] = e
        bad_mem = [m for m in e.get("mems", []) if not (m["wr"] == 0 or (m["rd"], m["wr"]) in
                                                        ((1, 1), (1, 2), (2, 1), (2, 2)))]
        ok = e["rc"] == 0 and e.get("top") == top and e.get("dpe") == expected and not bad_mem
        print(f"elab {'OK' if ok else 'FAIL'}: rc {e['rc']}, top {e.get('top')}, dpe {e.get('dpe')} (expected {expected}), "
              f"{len(e.get('mems', []))} memories, Parmys-illegal shapes {len(bad_mem)}, {e['wall_s']}s")
    if ok and a.step in ("flow", "all"):
        f = flow(path, arch, work, a.seed)
        out["flow"] = f
        wc = f.get("resources", {}).get("wc")
        ok = f["rc"] == 0 and wc == expected
        print(f"flow {'OK' if ok else 'FAIL'}: rc {f['rc']}, wc {wc} (expected {expected}), Fmax {f.get('fmax_mhz')} MHz, "
              f"clb {f.get('resources', {}).get('clb')}, memory {f.get('resources', {}).get('memory')}, "
              f"{f['wall_s']}s")
    (work / "result.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
