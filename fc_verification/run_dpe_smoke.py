#!/usr/bin/env python3
"""DPE / DSP-MAC behavior model smoke test sweep.

Exercises the three primitive behavior models (dpe_stub_nldpe.v,
dpe_stub_azurelily.v, dsp_mac.v) across multiple geometries via the
`+define+` CLI parameter mechanism on tb_dpe_vmm.v / tb_dpe_acam.v /
tb_dsp_mac.v.

Verifies for each (arch, R, C, BUF, PRECISION, PIPELINE_DEPTH) combo:
  1. functional correctness (TB asserts byte-level match)
  2. cycle count matches T_fill = LOAD + COMPUTE + OUTPUT (§4 pipeline)

DPE COMPUTE_CYCLES is derived from the precision-driven bit-serial
pipeline model (FIDELITY_METHODOLOGY.md §3):

    CCYC = PRECISION_BITS + PIPELINE_DEPTH - 1

This is arch-agnostic — both NL-DPE and AL DPE share the same
fire -> VMM -> accumulate compute structure.

Writes a clean summary to stdout AND fc_verification/results/dpe_smoke.log.

Usage:
    python3 fc_verification/run_dpe_smoke.py                     # default: PREC=8 + sweep {4,8,16}
    python3 fc_verification/run_dpe_smoke.py --precision 4       # sweep PREC=4 only
    python3 fc_verification/run_dpe_smoke.py --quick             # smaller matrix
    python3 fc_verification/run_dpe_smoke.py --keep              # keep tmp binaries
"""
from __future__ import annotations
import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTL = REPO / "fc_verification" / "rtl"
TB_DIR = REPO / "fc_verification"
RESULTS = REPO / "fc_verification" / "results"

# Precision-driven compute pipeline: see FIDELITY_METHODOLOGY.md §3.
DEFAULT_PRECISION = 8
DEFAULT_PIPELINE_DEPTH = 3


@dataclass
class Case:
    arch: str             # "NL-DPE" | "AL" | "NL-DPE_ACAM" | "AL_DSP_MAC"
    label: str            # short tag for log
    defines: dict         # +define+ key=value pairs passed to iverilog
    expected_cycles: int  # T_fill computed analytically
    src_v: list           # list of .v paths to compile
    tb: str               # which TB ("dpe_vmm", "dpe_acam", "dsp_mac")
    pass_re: re.Pattern = field(default=re.compile(r"PASS"))
    fail_re: re.Pattern = field(default=re.compile(r"(FAIL|MISMATCH|ERROR)"))


def _compute_cycles(precision: int, pipeline_depth: int) -> int:
    """Precision-driven bit-serial pipeline cycle count.

    CCYC = PRECISION + PIPELINE_DEPTH - 1.
    Default INT8 + 3-stage = 10 cycles.
    """
    return precision + pipeline_depth - 1


def _t_fill_dpe(R: int, C: int, BUF: int, precision: int,
                pipeline_depth: int) -> int:
    eps = BUF // 8
    lstr = math.ceil(R / eps)
    ocyc = math.ceil(C / eps)
    ccyc = _compute_cycles(precision, pipeline_depth)
    return lstr + ccyc + ocyc


def _t_fill_dsp_mac(K: int, BUF: int, DSP_WIDTH: int, PREC: int = 8) -> int:
    lstr = math.ceil(K * PREC / BUF)
    ccyc = max(1, math.ceil(K / DSP_WIDTH))
    ocyc_raw = math.ceil(PREC / BUF)
    ocyc = max(1, ocyc_raw)
    return lstr + ccyc + ocyc


def _add_dpe_cases(cases, precision, pipeline_depth, quick):
    """Add NL-DPE / AL-DPE / ACAM cases for a given precision."""
    tag = f"P{precision}D{pipeline_depth}"
    common_defs = {
        "PRECISION_TB": precision,
        "PIPELINE_DEPTH_TB": pipeline_depth,
    }

    # ── NL-DPE DPE VMM (BUF=40, R >= C) ──
    nldpe_shapes = [(256, 256), (512, 256), (1024, 256),
                    (256, 128), (512, 128), (1024, 128)]
    if not quick:
        nldpe_shapes += [(256, 64), (1024, 1024), (2048, 256)]
    for (R, C) in nldpe_shapes:
        defs = dict(common_defs)
        defs.update({"ARCH_NLDPE": "1", "R_TB": R, "C_TB": C, "BUF_TB": 40})
        cases.append(Case(
            arch="NL-DPE",
            label=f"VMM_NLDPE_R{R}_C{C}_{tag}",
            defines=defs,
            expected_cycles=_t_fill_dpe(R, C, 40, precision, pipeline_depth),
            src_v=[str(RTL / "dpe_stub_nldpe.v"), str(TB_DIR / "tb_dpe_vmm.v")],
            tb="dpe_vmm",
        ))

    # ── AL DPE VMM (BUF=16, R >= C) ──
    al_shapes = [(512, 128), (1024, 128), (256, 128), (512, 64)]
    if not quick:
        al_shapes += [(256, 64), (1024, 64), (2048, 128)]
    for (R, C) in al_shapes:
        defs = dict(common_defs)
        defs.update({"ARCH_AL": "1", "R_TB": R, "C_TB": C, "BUF_TB": 16})
        cases.append(Case(
            arch="AL",
            label=f"VMM_AL_R{R}_C{C}_{tag}",
            defines=defs,
            expected_cycles=_t_fill_dpe(R, C, 16, precision, pipeline_depth),
            src_v=[str(RTL / "dpe_stub_azurelily.v"), str(TB_DIR / "tb_dpe_vmm.v")],
            tb="dpe_vmm",
        ))

    # ── NL-DPE ACAM exp (BUF=40, R >= C) ──
    acam_shapes = [(256, 256), (512, 256), (1024, 256), (256, 128), (1024, 128)]
    if not quick:
        acam_shapes += [(256, 64), (2048, 256)]
    for (R, C) in acam_shapes:
        defs = dict(common_defs)
        defs.update({"R_TB": R, "C_TB": C, "BUF_TB": 40})
        cases.append(Case(
            arch="NL-DPE_ACAM",
            label=f"ACAM_NLDPE_R{R}_C{C}_{tag}",
            defines=defs,
            expected_cycles=_t_fill_dpe(R, C, 40, precision, pipeline_depth),
            src_v=[str(RTL / "dpe_stub_nldpe.v"), str(TB_DIR / "tb_dpe_acam.v")],
            tb="dpe_acam",
        ))


def build_cases(quick: bool, precision_override: int | None) -> list[Case]:
    cases = []

    if precision_override is None:
        # Default sweep: full geometry matrix at PREC=8 (the canonical
        # INT8 model), plus a precision axis check at one (R,C) shape
        # for {4, 16} to verify CCYC scales linearly with precision.
        _add_dpe_cases(cases, DEFAULT_PRECISION, DEFAULT_PIPELINE_DEPTH, quick)

        # Precision-axis sanity (one shape per arch, P ∈ {4, 16}).
        for prec in (4, 16):
            tag = f"P{prec}D{DEFAULT_PIPELINE_DEPTH}"
            # NL-DPE @ R=256 C=256 BUF=40
            cases.append(Case(
                arch="NL-DPE",
                label=f"VMM_NLDPE_R256_C256_{tag}",
                defines={"ARCH_NLDPE": "1", "R_TB": 256, "C_TB": 256,
                         "BUF_TB": 40, "PRECISION_TB": prec,
                         "PIPELINE_DEPTH_TB": DEFAULT_PIPELINE_DEPTH},
                expected_cycles=_t_fill_dpe(256, 256, 40, prec,
                                            DEFAULT_PIPELINE_DEPTH),
                src_v=[str(RTL / "dpe_stub_nldpe.v"),
                       str(TB_DIR / "tb_dpe_vmm.v")],
                tb="dpe_vmm",
            ))
            # AL @ R=512 C=128 BUF=16
            cases.append(Case(
                arch="AL",
                label=f"VMM_AL_R512_C128_{tag}",
                defines={"ARCH_AL": "1", "R_TB": 512, "C_TB": 128,
                         "BUF_TB": 16, "PRECISION_TB": prec,
                         "PIPELINE_DEPTH_TB": DEFAULT_PIPELINE_DEPTH},
                expected_cycles=_t_fill_dpe(512, 128, 16, prec,
                                            DEFAULT_PIPELINE_DEPTH),
                src_v=[str(RTL / "dpe_stub_azurelily.v"),
                       str(TB_DIR / "tb_dpe_vmm.v")],
                tb="dpe_vmm",
            ))
            # NL-DPE ACAM @ R=256 C=256 BUF=40
            cases.append(Case(
                arch="NL-DPE_ACAM",
                label=f"ACAM_NLDPE_R256_C256_{tag}",
                defines={"R_TB": 256, "C_TB": 256, "BUF_TB": 40,
                         "PRECISION_TB": prec,
                         "PIPELINE_DEPTH_TB": DEFAULT_PIPELINE_DEPTH},
                expected_cycles=_t_fill_dpe(256, 256, 40, prec,
                                            DEFAULT_PIPELINE_DEPTH),
                src_v=[str(RTL / "dpe_stub_nldpe.v"),
                       str(TB_DIR / "tb_dpe_acam.v")],
                tb="dpe_acam",
            ))
    else:
        # Sweep this precision across the full DPE geometry matrix.
        _add_dpe_cases(cases, precision_override, DEFAULT_PIPELINE_DEPTH, quick)

    # ── AL DSP-MAC (BUF=16, DSP_WIDTH=4, K varies) ──
    # DSP-MAC compute model is a separate TODO (Task #79); leave as-is.
    dsp_K = [16, 32, 64, 128]
    if not quick:
        dsp_K += [256, 512, 8]
    for K in dsp_K:
        cases.append(Case(
            arch="AL_DSP_MAC",
            label=f"DSPMAC_K{K}",
            defines={"K_TB": K, "BUF_TB": 16, "DSP_WIDTH_TB": 4},
            expected_cycles=_t_fill_dsp_mac(K, 16, 4),
            src_v=[str(RTL / "dsp_mac.v"), str(TB_DIR / "tb_dsp_mac.v")],
            tb="dsp_mac",
        ))

    return cases


def run_case(c: Case, tmpdir: Path) -> tuple[bool, str, int | None]:
    """Compile + simulate one case. Returns (passed, summary_line, observed_cycles or None)."""
    bin_path = tmpdir / f"tb_{c.label}"
    cmd_iv = ["iverilog", "-o", str(bin_path)]
    for k, v in c.defines.items():
        cmd_iv.extend(["-D", f"{k}={v}"])
    cmd_iv.extend(c.src_v)
    proc = subprocess.run(cmd_iv, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"COMPILE_FAIL:\n{(proc.stdout + proc.stderr)[-400:]}", None

    proc = subprocess.run(["vvp", str(bin_path)], capture_output=True, text=True, timeout=30)
    out = proc.stdout + proc.stderr

    # Parse total_cycles and PASS/FAIL
    m_cycles = re.search(r"total_cycles=(\d+)", out)
    observed_cycles = int(m_cycles.group(1)) if m_cycles else None

    pass_hit = bool(c.pass_re.search(out))
    fail_hit = bool(c.fail_re.search(out))
    cycle_match = (observed_cycles == c.expected_cycles) if observed_cycles is not None else False

    passed = pass_hit and not fail_hit and cycle_match
    summary = (
        f"  observed={observed_cycles}, expected={c.expected_cycles}, "
        f"PASS={pass_hit}, FAIL={fail_hit}, cycle_match={cycle_match}"
    )
    return passed, summary, observed_cycles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller test matrix")
    ap.add_argument("--keep", action="store_true", help="don't delete tmp build dir")
    ap.add_argument("--precision", type=int, default=None,
                    help="If set, sweep only this PRECISION across the full "
                         "DPE matrix (skips the default {4,8,16} mini-axis). "
                         "Default: full sweep at PREC=8 + axis check at "
                         "{4,16} for one shape.")
    args = ap.parse_args()

    cases = build_cases(args.quick, args.precision)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "dpe_smoke.log"

    tmpdir = Path(tempfile.mkdtemp(prefix="dpe_smoke_"))
    print(f"=== DPE behavior-model smoke sweep — {len(cases)} cases ===")
    if args.precision is None:
        print(f"Precision: default sweep (PREC=8 full + PREC ∈ {{4,16}} axis)")
    else:
        print(f"Precision: PREC={args.precision} (full DPE matrix)")
    print(f"tmpdir: {tmpdir}")
    print(f"log:    {log_path}\n")

    log_lines = [
        f"DPE / DSP-MAC behavior model smoke sweep",
        f"Mode: {'quick' if args.quick else 'full'}",
        f"Precision override: {args.precision}",
        f"Cases: {len(cases)}",
        "",
    ]

    n_pass = n_fail = 0
    for c in cases:
        passed, summary, obs = run_case(c, tmpdir)
        status = "PASS" if passed else "FAIL"
        line = f"[{status}] {c.label:<40} expected_cycles={c.expected_cycles:>6}  observed={obs}"
        print(line)
        if not passed:
            print(summary)
        log_lines.append(line)
        log_lines.append(summary)
        log_lines.append("")
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    log_lines.append(f"\nSUMMARY: {n_pass}/{len(cases)} PASS, {n_fail} FAIL")
    log_path.write_text("\n".join(log_lines) + "\n")

    print(f"\n=== Summary: {n_pass}/{len(cases)} PASS, {n_fail} FAIL ===")
    print(f"Log:        {log_path}")

    if not args.keep:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
