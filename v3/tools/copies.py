#!/usr/bin/env python3
"""copies.py — provenance lock for the files v3 copies from v2 (PLAN R3, phase F1).

Modeled on v2/simnext/tools/copies.py. Every copied file is listed in COPIES with its source and its path-line
substitutions. A substitution (old, new) replaces whole lines: `old` is one or more complete lines of the source and
must occur there exactly once; it is replaced by `new` plus the mark NOTE. Only lines that set a path are substituted
(sys.path entries, REPO / file / directory constants, default output and stimulus directories). Nothing else may
differ. Per entry the gate checks:
  1 copy    the copy == the source with its substitutions, byte for byte
  2 source  the source has the sha256 recorded in copies.lock
The sources are read from the git tag v2-final: v2/ was removed from the tree on 2026-10-17 (`git checkout v2-final -- v2`
restores it), so the gate needs no v2 folder.
When v3 starts editing a copied file (for example phase A edits dimm_tile_top.v), its entry leaves COPIES in the same
step, and the log in LEDGER.md records the v2 sha256 it started from.

Usage:
  python3 v3/tools/copies.py              # gate
  python3 v3/tools/copies.py --make       # write missing copies and their lock entries (never overwrites a
                                          # differing copy or a recorded hash)
  python3 v3/tools/copies.py --list       # the table: copy <- source, number of substituted blocks
  python3 v3/tools/copies.py --self-test  # a changed copy, a changed source, a stale or ambiguous substitution
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
REPO = V3.parent
LOCK = HERE / "copies.lock"
NOTE = "  # v3 path (tools/copies.py)"
SOURCE_TAG = "v2-final"          # the last commit with v2/ in the tree; the sources are read from it

# ---------------------------------------------------------------------------------------------------------------------
# path-line substitutions
# ---------------------------------------------------------------------------------------------------------------------
_R3 = ("REPO = Path(__file__).resolve().parents[3]", "REPO = Path(__file__).resolve().parents[4]")
_GK = ('sys.path.insert(0, str(REPO / "v2" / "sim" / "simulator" / "kernels"))',
       'sys.path.insert(0, str(REPO / "v3" / "oracle" / "v2_kernels"))')
_GO = ('sys.path.insert(0, str(REPO / "v2" / "sim" / "oracle"))',
       'sys.path.insert(0, str(REPO / "v3" / "oracle"))')


def _v3(*parts: str) -> str:
    return "REPO / " + " / ".join(f'"{p}"' for p in ("v3",) + parts)


def _out(old_corpus: str, module: str, corpus: str, flag: str = "--out") -> tuple:
    return (f'    ap.add_argument("{flag}", default=str(REPO / "v2" / "sim" / "smoke" / "{old_corpus}"))',
            f'    ap.add_argument("{flag}", default=str({_v3("work", module, corpus)}))')


_XB_SIM = _v3("ip", "xbar", "rtl", "sim", "dpe_nldpe.v")
GEN_DPE = (_R3, _GK, _GO, _out("stimuli", "xbar", "dpe_stimuli"))
RUN_DPE = (
    _R3,
    ('SMOKE = REPO / "v2" / "sim" / "smoke"', f'SMOKE = {_v3("ip", "xbar", "tb")}'),
    ('STIMULI = SMOKE / "stimuli"', f'STIMULI = {_v3("work", "xbar", "dpe_stimuli")}'),
    ('LOGS = SMOKE / "logs"', f'LOGS = {_v3("work", "xbar", "logs")}'),
    ('TB = REPO / "v2" / "sim" / "tb" / "tb_dpe_nldpe.v"', f'TB = {_v3("ip", "xbar", "tb", "tb_dpe_nldpe.v")}'),
    ('DEFAULT_RTL = REPO / "v2" / "sim" / "rtl" / "dpe_nldpe.v"', f"DEFAULT_RTL = {_XB_SIM}"),
    # the interface pre-check: v3's own gate (dpe == dpe_blackbox == arch XML, plus the twins) replaces the v2 check
    # against the legacy rtl_flow files, which v3 does not copy
    ('CHECK_IF = SMOKE / "check_interface.py"', f'CHECK_IF = {_v3("tools", "check_twins.py")}'),
    ('    lock_path = SMOKE / ".dpe_rtl.lock"', f'    lock_path = {_v3("work", "xbar", ".dpe_rtl.lock")}'),
)
GEN_GEMM = (_R3, _GK, _GO, _out("gemm_stimuli", "gemm", "gemm_stimuli"))
RUN_GEMM = (
    _R3,
    ('SMOKE = REPO / "v2" / "sim" / "smoke"', f'SMOKE = {_v3("ip", "gemm", "tb")}'),
    ('STIMULI = SMOKE / "gemm_stimuli"', f'STIMULI = {_v3("work", "gemm", "gemm_stimuli")}'),
    ('STIMULI_P29 = SMOKE / "gemm_stimuli_p29"     # P29 pass-extent corpus (--p29)',
     f'STIMULI_P29 = {_v3("work", "gemm", "gemm_stimuli_p29")}     # P29 pass-extent corpus (--p29)'),
    ('LOGS = SMOKE / "logs"', f'LOGS = {_v3("work", "gemm", "logs")}'),
    ('TB = REPO / "v2" / "sim" / "tb" / "tb_gemm_top.v"', f'TB = {_v3("ip", "gemm", "tb", "tb_gemm_top.v")}'),
    ('DEFAULT_RTL = REPO / "v2" / "sim" / "rtl" / "gemm_top.v"',
     f'DEFAULT_RTL = {_v3("ip", "gemm", "rtl", "sim", "gemm_top.v")}'),
    ('PRIM_RTL = REPO / "v2" / "sim" / "rtl" / "dpe_nldpe.v"', f"PRIM_RTL = {_XB_SIM}"),
    ('    lock_path = SMOKE / ".gemm_rtl.lock"', f'    lock_path = {_v3("work", "gemm", ".gemm_rtl.lock")}'),
)
GEN_TILE = (_R3, _GK, _GO, _out("dimm_tile_stimuli", "dimm", "dimm_tile_stimuli"))
RUN_TILE = (
    _R3,
    ('RTL = [REPO / "v2" / "sim" / "rtl" / f for f in ("dpe_nldpe.v", "dimm_tile_top.v")]',
     f'RTL = [{_XB_SIM}, {_v3("ip", "dimm", "rtl", "sim", "dimm_tile_top.v")}]'),
    ('TB = REPO / "v2" / "sim" / "tb" / "tb_dimm_tile_top.v"', f'TB = {_v3("ip", "dimm", "tb", "tb_dimm_tile_top.v")}'),
    ('LOGS = REPO / "v2" / "sim" / "smoke" / "logs"', f'LOGS = {_v3("work", "dimm", "logs")}'),
    _out("dimm_tile_stimuli", "dimm", "dimm_tile_stimuli", "--stimuli"),
)
GEN_SMOL = (_R3, _GK, _GO, _out("softmax_online_stimuli", "softmax", "softmax_online_stimuli"))
RUN_SMOL = (
    _R3,
    ('RTL_FILES = [REPO / "v2" / "sim" / "rtl" / "dpe_nldpe.v",\n'
     '             REPO / "v2" / "sim" / "rtl" / "softmax_online_top.v"]',
     f'RTL_FILES = [{_XB_SIM},\n'
     f'             {_v3("ip", "softmax", "rtl", "sim", "softmax_online_top.v")}]'),
    ('TB_FILE = REPO / "v2" / "sim" / "tb" / "tb_softmax_online_top.v"',
     f'TB_FILE = {_v3("ip", "softmax", "tb", "tb_softmax_online_top.v")}'),
    ('LOGS = REPO / "v2" / "sim" / "smoke" / "logs"', f'LOGS = {_v3("work", "softmax", "logs")}'),
    ('LOCK = REPO / "v2" / "sim" / "smoke" / ".softmax_online_rtl.lock"',
     f'LOCK = {_v3("work", "softmax", ".softmax_online_rtl.lock")}'),
    _out("softmax_online_stimuli", "softmax", "softmax_online_stimuli", "--stimuli"),
)
# ---------------------------------------------------------------------------------------------------------------------
# the table: (copy under v3/, source under the repo, substitutions)
# ---------------------------------------------------------------------------------------------------------------------
ORACLES = ("nldpe_ref.py", "gemm_ref.py", "dimm_ref.py", "softmax_ref.py", "softmax_online_ref.py",
           "attention_ref.py", "flash-attention-log-domain.ipynb")
# the whole v2 kernel set: the copied case writers import it (gen_softmax_online_cases -> attention_sim -> proj_sim,
# softmax_online_sim -> softmax_sim -> dimm_sim). v3 code uses these only as test references (closed forms) and case
# writers, never as models; they sit in oracle/v2_kernels/ with the oracles (what the simulator is checked against).
# Kernels and oracles keep their v2 relative depth (kernels two levels below the folder that holds oracle/), so they
# need no edits.
KERNELS = ("nldpe_sim.py", "pass_engine.py", "gemm_sim.py", "dimm_sim.py", "softmax_sim.py", "softmax_online_sim.py",
           "dimm_tile_sim.py", "ocarry_sim.py", "proj_sim.py", "attention_sim.py")
S, X = "v2/sim", "v2/experiment"
COPIES = [
    *[(f"oracle/{f}", f"{S}/oracle/{f}", ()) for f in ORACLES],
    *[(f"oracle/v2_kernels/{f}", f"{S}/simulator/kernels/{f}", ()) for f in KERNELS],
    # the platform code (platforms.py, cost.py) is no longer a copy: simulator/platform/ is v3's own (2026-10-17, the
    # refactor's decision 4); azure_lily.json stays a verbatim copy; nl_dpe.json forked 2026-10-23 (the row-parallel
    # port, FORKED below)
    ("simulator/platform/azure_lily.json", f"{S}/simulator/configs/azure_lily.json", ()),  # data only
    # crossbar primitive
    # the crossbar's two files: forked 2026-10-23 for the bit-plane interface (FORKED below)
    # the crossbar's pre-bit-plane test harness: retired 2026-10-23 (FORKED below; tools/cleanup_bitplane.sh removes it)
    # GEMM
    # gemm_top.v: v3-owned since F4 (2026-10-14: out_valid made a port; see FORKED below)
    # tb_gemm_top.v and the twin baseline gemm.diff: forked 2026-10-23 for the bit-plane interface (FORKED below)
    # the GEMM's v2 harness scripts: retired 2026-10-23 (FORKED below; the GATE-2 harness replaces them)
    # tiled DIMM: v3-owned since F3 / phase A (2026-10-14; see FORKED below)
    # online softmax: v3-owned since F3 / phase B (2026-10-14; see FORKED below)
    # its accumulator: v3-owned since F3 / phase C (2026-10-14; softmax_online_acc, see FORKED below)
    # FIFO
    ("ip/lib/rtl/sim/att_fifo.v", f"{S}/rtl/att_fifo.v", ()),
    ("ip/lib/rtl/vtr/att_fifo.v", f"{X}/rtl/att_fifo.v", ()),
    ("ip/lib/tb/tb_att_fifo.v", f"{S}/tb/tb_att_fifo.v", ()),
    # VTR flow (the generic parts; vtr_tops.py, run_vtr_dse.py and the cleanup script come with the first VTR run)
    # v2's 12 arch XMLs (vtr/arch/nl_dpe_<R>x<C>_auto.xml) left the tree on 2026-10-23 (user: one vtr/arch/ with the
    # row-parallel XMLs of vtr/gen_arch.py); their lines were here until then
    ("vtr/run_vtr.py", f"{X}/vtr/run_vtr.py", ()),
    ("vtr/clock.sdc", f"{X}/vtr/clock.sdc", ()),
    # twin-diff baselines recorded in v2: the last one (gemm.diff) forked 2026-10-23 (FORKED below)
    # v2's FC results (user 2026-10-13: the FC rows are reused, not re-run; the attention rows in these files are the
    # earlier attention DSE, which is not a valid experiment): the regression reference of simulator/tests/testFcRows.py
    # (in dse/fc_snapshot/ until 2026-10-17; the current FC results are dse/results/fc_*_results.csv)
    *[(f"simulator/tests/ref/{f}", f"{X}/dse/results/{f}", ()) for f in ("vtr_results.csv", "dse_v2.csv", "ranking_fc.json")],
]

# copies v3 has since edited or replaced: (path under v3/, the v2 source it started from). They are v3 files now; the
# lock keeps the source's sha256 as the record of the starting point (LEDGER.md log).
FORKED = [
    ("simulator/platform/nl_dpe.json", f"{S}/simulator/configs/nl_dpe.json"),       # the port priced per bit (2026-10-23)
    ("ip/dimm/rtl/sim/dimm_tile_top.v", f"{S}/rtl/dimm_tile_top.v"),
    ("ip/dimm/rtl/vtr/dimm_tile_top.v", f"{X}/rtl/dimm_tile_top.v"),
    ("ip/dimm/tb/tb_dimm_tile_top.v", f"{S}/tb/tb_dimm_tile_top.v"),
    ("tools/twin_baseline/dimm_tile.diff", f"{X}/rtl/split_baseline/dimm_tile.diff"),
    ("ip/dimm/tb/gen_dimm_tile_cases.py", f"{S}/smoke/gen_dimm_tile_cases.py"),     # obsolete (v2 interface)
    ("ip/dimm/tb/run_dimm_tile_rtl.py", f"{S}/smoke/run_dimm_tile_rtl.py"),         # obsolete (v2 interface)
    ("ip/softmax/rtl/sim/softmax_online_top.v", f"{S}/rtl/softmax_online_top.v"),
    ("ip/softmax/rtl/vtr/softmax_online_top.v", f"{X}/rtl/softmax_online_top.v"),
    ("ip/softmax/tb/tb_softmax_online_top.v", f"{S}/tb/tb_softmax_online_top.v"),
    ("tools/twin_baseline/softmax_online.diff", f"{X}/rtl/split_baseline/softmax_online.diff"),
    ("ip/softmax/tb/gen_softmax_online_cases.py", f"{S}/smoke/gen_softmax_online_cases.py"),   # obsolete (v2 interface)
    ("ip/softmax/tb/run_softmax_online_rtl.py", f"{S}/smoke/run_softmax_online_rtl.py"),       # obsolete (v2 interface)
    ("ip/softmax/rtl/sim/softmax_online_acc.v", f"{S}/rtl/att_ocarry_top.v"),
    ("ip/softmax/rtl/vtr/softmax_online_acc.v", f"{X}/rtl/att_ocarry_top.v"),
    ("ip/softmax/tb/tb_softmax_online_acc.v", f"{S}/tb/tb_att_ocarry_top.v"),
    ("tools/twin_baseline/softmax_online_acc.diff", f"{X}/rtl/split_baseline/att_ocarry.diff"),
    ("ip/softmax/rtl/sim/att_ocarry_top.v", f"{S}/rtl/att_ocarry_top.v"),           # obsolete (softmax_online_acc.v)
    ("ip/softmax/rtl/vtr/att_ocarry_top.v", f"{X}/rtl/att_ocarry_top.v"),           # obsolete (softmax_online_acc.v)
    ("ip/softmax/tb/tb_att_ocarry_top.v", f"{S}/tb/tb_att_ocarry_top.v"),           # obsolete (tb_softmax_online_acc.v)
    ("ip/softmax/tb/gen_ocarry_cases.py", f"{S}/smoke/gen_ocarry_cases.py"),        # obsolete (v2 interface)
    ("ip/softmax/tb/run_ocarry_rtl.py", f"{S}/smoke/run_ocarry_rtl.py"),            # obsolete (v2 interface)
    ("tools/twin_baseline/att_ocarry.diff", f"{X}/rtl/split_baseline/att_ocarry.diff"),   # obsolete (the pair is gone)
    ("ip/gemm/rtl/sim/gemm_top.v", f"{S}/rtl/gemm_top.v"),            # out_valid made a port; 2026-10-15 the
                                                                       # reduction tree held as one wide register
    ("ip/gemm/rtl/vtr/gemm_top.v", f"{X}/rtl/gemm_top.v"),
    # the DSE figures stage (2026-10-16): ported, v3 inputs; the metric, resampling and layout unchanged
    ("dse/rank.py", f"{X}/dse/rank_dse.py"),
    ("dse/plot_dse.py", f"{X}/dse/plot_dse.py"),
    ("dse/plot_dse.py", "v2/work/dse_v1/combined/plot_pareto_family.py"),   # its Pareto part
    ("dse/plot_dse.py", "v2/work/dse_v1/combined/plot_pareto.py"),          # the FC x attention cross score
    # the FC VTR table of the results (2026-10-17): the 48 family == gemm rows of v2's vtr_results.csv, verbatim lines
    ("dse/results/fc_vtr_results.csv", f"{X}/dse/results/vtr_results.csv"),
    # the bit-plane interface (2026-10-23, user's go): R-bit data_in, C-bit data_out, 10-cycle pass (ip/xbar/spec.md)
    ("ip/xbar/rtl/sim/dpe_nldpe.v", f"{S}/rtl/dpe_nldpe.v"),
    ("ip/xbar/rtl/vtr/dpe_blackbox.v", f"{X}/rtl/dpe_blackbox.v"),
    ("ip/gemm/tb/tb_gemm_top.v", f"{S}/tb/tb_gemm_top.v"),            # the GATE-2 testbench (preload, plane words)
    ("tools/twin_baseline/gemm.diff", f"{X}/rtl/split_baseline/gemm.diff"),   # the bit-plane twins' difference
    # retired 2026-10-23 with the bit-plane pass (tools/cleanup_bitplane.sh removes them; the lock keeps the record)
    ("ip/xbar/tb/tb_dpe_nldpe.v", f"{S}/tb/tb_dpe_nldpe.v"),
    ("ip/xbar/tb/gen_cases.py", f"{S}/smoke/gen_cases.py"),
    ("ip/xbar/tb/run_dpe_rtl.py", f"{S}/smoke/run_dpe_rtl.py"),
    ("ip/gemm/tb/gen_gemm_cases.py", f"{S}/smoke/gen_gemm_cases.py"),
    ("ip/gemm/tb/run_gemm_rtl.py", f"{S}/smoke/run_gemm_rtl.py"),
]

# ---------------------------------------------------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------------------------------------------------
def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _block_re(old: str) -> re.Pattern:
    return re.compile(r"(?m)^" + re.escape(old) + r"$")


def transform(src: str, subs: tuple) -> str:
    """The source with each whole-line block `old` (exactly one occurrence) replaced by `new` + NOTE."""
    out = src
    for old, new in subs:
        k, k_out = len(_block_re(old).findall(src)), len(_block_re(old).findall(out))
        if k != 1 or k_out != 1:
            raise ValueError(f"substitution source block found {k} times: {old[:70]!r}")
        out = _block_re(old).sub(lambda _m, new=new: new + NOTE, out, count=1)
    return out


def check_entry(copy_text: str, src_text: str, subs: tuple) -> str | None:
    try:
        want = transform(src_text, subs)
    except ValueError as e:
        return str(e)
    if copy_text != want:
        a, b = copy_text.splitlines(), want.splitlines()
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return f"copy differs from the declared transform at line {k + 1}"
    return None


def source_bytes(rel: str) -> bytes | None:
    """The source file as committed at SOURCE_TAG; None if the tag has no such file."""
    r = subprocess.run(["git", "cat-file", "blob", f"{SOURCE_TAG}:{rel}"], cwd=REPO, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _read(p: Path) -> str:
    return p.read_bytes().decode()          # no newline translation: byte-for-byte


def gate() -> int:
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    dsts = [d for d, _, _ in COPIES]
    bad = 0 if len(set(dsts)) == len(dsts) else 1
    if bad:
        print("  FAIL table: a copy path is listed twice")
    for dst, src, subs in COPIES:
        d = V3 / dst
        sb = source_bytes(src)
        errs = []
        if sb is None or not d.exists():
            errs.append("missing " + (f"source at tag {SOURCE_TAG}" if sb is None else "copy"))
        else:
            if lock.get(src) != sha(sb):
                errs.append("source sha256 differs from copies.lock")
            e = check_entry(_read(d), sb.decode(), subs)
            if e:
                errs.append(e)
        if errs:
            bad += 1
            print(f"  FAIL {dst} <- {src}: {'; '.join(errs)}")
    for dst, src in FORKED:
        e = None if lock.get(src) else "no recorded sha256"
        if e:
            bad += 1
            print(f"  FAIL forked {dst} <- {src}: {e}")
    n_sub = sum(1 for _, _, s in COPIES if s)
    print(f"copies: {len(COPIES) - bad}/{len(COPIES)} OK ({len(COPIES) - n_sub} verbatim, {n_sub} with path lines); "
          f"{len(FORKED)} forked into v3 (start points recorded)" + ("" if not bad else f", {bad} FAIL"))
    return 1 if bad else 0


def make() -> int:
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    wrote = 0
    for dst, src, subs in COPIES:
        d = V3 / dst
        sb = source_bytes(src)
        if sb is None:
            raise SystemExit(f"{src}: not in tag {SOURCE_TAG}; not copied")
        want = transform(sb.decode(), subs)
        if d.exists() and _read(d) != want:
            raise SystemExit(f"{dst} exists and differs from the declared transform; not overwritten")
        if src in lock and lock[src] != sha(sb):
            raise SystemExit(f"{src}: recorded sha256 differs; not updated")
        if not d.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(want.encode())
            wrote += 1
        lock[src] = sha(sb)
    LOCK.write_text(json.dumps(dict(sorted(lock.items())), indent=1) + "\n")
    print(f"copies --make: {len(COPIES)} entries, {wrote} written, lock {LOCK.relative_to(REPO)}")
    return 0


def listing() -> int:
    for dst, src, subs in COPIES:
        print(f"  {dst:44s} <- {src}" + (f"  [{len(subs)} path blocks]" if subs else ""))
    print(f"{len(COPIES)} entries")
    return 0


def self_test() -> int:
    src = "a = 1\nREPO = X\nb = REPO / 'v2'\nb = REPO / 'v2'\n"
    sub = (("REPO = X", "REPO = Y"),)
    want = "a = 1\nREPO = Y" + NOTE + "\nb = REPO / 'v2'\nb = REPO / 'v2'\n"
    assert transform(src, sub) == want
    assert check_entry(want, src, sub) is None, "the declared transform must pass"
    assert check_entry(want.replace("a = 1", "a = 2"), src, sub), "a changed copy must fail"
    assert check_entry(want, src + "c = 3\n", sub), "a changed source must fail"
    assert check_entry(want, src.replace("REPO = X", "REPO = Z"), sub), "a stale substitution must fail"
    assert check_entry(want, src, (("b = REPO / 'v2'", "b = 0"),)), "an ambiguous substitution must fail"
    assert check_entry(want, src, (("REPO =", "R ="),)), "a partial-line substitution must fail"
    multi = "x\nL = [a,\n     b]\ny\n"
    assert transform(multi, (("L = [a,\n     b]", "L = [c,\n     d]"),)) == "x\nL = [c,\n     d]" + NOTE + "\ny\n"
    # the real table: every substitution applies once to its source
    n = 0
    for _dst, s, subs in COPIES:
        if subs:
            transform(source_bytes(s).decode(), subs)
            n += 1
    print(f"copies --self-test: ALL PASS (changed copy, changed source, stale / ambiguous / partial substitution "
          f"caught; the {n} substituted entries apply)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--make", action="store_true")
    g.add_argument("--list", action="store_true")
    g.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    return make() if a.make else listing() if a.list else self_test() if a.self_test else gate()


if __name__ == "__main__":
    sys.exit(main())
