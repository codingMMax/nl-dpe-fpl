#!/usr/bin/env python3
"""check_head_params.py — the simulator builds the same head as the RTL, for every DSE build.

For each of the 54 builds (docs/experiment.md §4.3) at the fixed mapping, `headParams` gives the head's top
parameters. This gate evaluates the VTR twin `top/rtl/vtr/att_head.v` with them, statically (no simulator, no tool):
its localparams, then every module instance's parameters, each completed with the module's own header defaults from
its VTR twin. It checks:
  params    every parameter of every instance == the parameter of the matching module model in the simulator's head
            graph (simulator/models/attentionHead.py), after the model's own defaults
  names     every model parameter is an RTL parameter of that module (same names, PLAN R1)
  crossbars the RTL count (GEMMs V each, DIMM farms + LOG converters, softmax N_EXP + 2, accumulator 2 N_OC) == the
            simulator's headParams nXbar == the sum of the module models' crossbar resources
Usage: python3 v3/tools/check_head_params.py [--p-proj 1] [--n-e-mult 2]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3 / "vtr"))
from tools.check_twins import _paren_group, _split_top, header, modules, strip_comments  # noqa: E402
from smoke import veval  # noqa: E402
from dse.space import CONFIGS, SHAPES  # noqa: E402
from simulator.models.attentionHead import attentionHead, headParams  # noqa: E402

TOP_KEYS = ("NSEQ", "DM", "D", "BKV", "R", "C", "P", "N_EXP", "N_E", "N_EP", "N_OC", "P_PROJ", "X_S")   # BUF left the
#   model 2026-10-22 and the RTL 2026-10-23 (the bit-plane pass)
FILES = {"gemm_top": "ip/gemm/rtl/vtr/gemm_top.v", "dimm_tile_top": "ip/dimm/rtl/vtr/dimm_tile_top.v",
         "softmax_online_top": "ip/softmax/rtl/vtr/softmax_online_top.v",
         "softmax_online_acc": "ip/softmax/rtl/vtr/softmax_online_acc.v", "att_ctrl": "top/rtl/vtr/att_ctrl.v"}
INSTANCE_MODEL = {"u_gq": "gq0", "u_gk": "gk0", "u_gv": "gv0", "u_qk": "qk", "u_sm": "sm", "u_pv": "pv", "u_acc": "acc",
                  "u_ctrl": "ctrl"}


def module_header(name: str) -> list:
    src = (V3 / FILES[name]).read_text()
    return header(modules(src)[name])[0]


def head_text() -> str:
    return modules((V3 / "top/rtl/vtr/att_head.v").read_text())["att_head"]


def localparams(body: str) -> list:
    return [(m.group(1), m.group(2)) for m in re.finditer(r"\blocalparam\s+(\w+)\s*=\s*([^;]+);", body)]


def instances(body: str) -> list:
    """[(module, instance, {param: expression})] of every parameterized instance in att_head."""
    out = []
    for m in re.finditer(r"\b(" + "|".join(FILES) + r")\s*#\s*\(", body):
        txt, i = _paren_group(body, m.end() - 1)
        inst = re.match(r"\s*(\w+)", body[i:]).group(1)
        ps = {}
        for item in _split_top(txt):
            pm = re.fullmatch(r"\s*\.(\w+)\s*\((.*)\)\s*", item, re.S)
            ps[pm.group(1)] = pm.group(2)
        out.append((m.group(1), inst, ps))
    return out


def rtl_params(module: str, given: dict, env: dict) -> dict:
    """The instance's full parameter set: the given expressions (in att_head's scope), then the header defaults."""
    full = {k: veval(v, env) for k, v in given.items()}
    for name, default in module_header(module):
        if name not in full:
            full[name] = veval(default, full)
    return full


def rtl_crossbars(insts: dict) -> int:
    g, qk, pv, sm, acc = (insts[k] for k in ("u_gq", "u_qk", "u_pv", "u_sm", "u_acc"))
    gemm = 3 * insts["u_ctrl"]["P_PROJ"] * (-(-g["K"] // g["R"])) * (-(-g["N"] // g["C"]))
    dimm = sum(x["N_E"] + (x["A_CH"] if x["A_DOM"] == 0 else 0) + (x["B_CH"] if x["B_DOM"] == 0 else 0) for x in (qk, pv))
    return gemm + dimm + (sm["N_EXP"] + 2) + 2 * acc["N_OC"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p-proj", type=int, default=1)
    ap.add_argument("--n-e-mult", type=int, default=None, help="N_E = this x d (default: headParams's d)")
    ap.add_argument("--n-ep-mult", type=int, default=None, help="N_EP = this x d (default: headParams's 2d)")
    a = ap.parse_args()
    body = strip_comments(head_text())
    top_defaults, lps, insts = header(head_text())[0], localparams(body), instances(body)
    bad, n = [], 0
    for (name, nseq, dm, d) in SHAPES:
        for R, C in CONFIGS:
            hp = headParams(nseq, dm, d, R, C, P_PROJ=a.p_proj, N_E=None if a.n_e_mult is None else a.n_e_mult * d,
                            N_EP=None if a.n_ep_mult is None else a.n_ep_mult * d)
            env = {k: getattr(hp, k) for k in TOP_KEYS}
            for k, default in top_defaults:                      # derived top parameters (V)
                if k not in env:
                    env[k] = veval(default, env)
            for k, expr in lps:
                env[k] = veval(expr, env)
            _, parts = attentionHead(hp, withValues=False)
            mods = dict(parts["gemm"], qk=parts["qk"], sm=parts["sm"], pv=parts["pv"], acc=parts["acc"],
                        ctrl=parts["ctrl"])
            rtl = {}
            for (module, inst, given) in insts:
                full = rtl_params(module, given, env)
                rtl[inst] = full
                model = mods[INSTANCE_MODEL[inst]]
                for k in model.paramNames():
                    if k not in full:
                        bad.append(f"{name} {R}x{C} {inst}: model parameter {k} is not an RTL parameter of {module}")
                    elif int(getattr(model.p, k)) != full[k]:
                        bad.append(f"{name} {R}x{C} {inst}.{k}: RTL {full[k]} != model {getattr(model.p, k)}")
            x_rtl = rtl_crossbars(rtl)
            x_mod = sum(sum(m.resources.crossbars.values()) for m in mods.values())
            if not x_rtl == hp.nXbar == x_mod:
                bad.append(f"{name} {R}x{C}: crossbars RTL {x_rtl}, headParams {hp.nXbar}, module models {x_mod}")
            n += 1
    for b in bad[:20]:
        print("  MISMATCH", b)
    print(f"check_head_params: {n} builds (P_PROJ {a.p_proj}, N_E {a.n_e_mult or 'd'}, N_EP {a.n_ep_mult or '2d'}), "
          f"{len(insts)} instances each: "
          + ("ALL PASS" if not bad else f"{len(bad)} mismatches"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
