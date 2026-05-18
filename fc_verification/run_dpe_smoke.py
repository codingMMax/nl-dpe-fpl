#!/usr/bin/env python3
"""DPE / DSP-MAC behavior model smoke test sweep.

Exercises the three primitive behavior models (dpe_nldpe.v,
dpe_azurelily.v, dsp_mac.v) across multiple geometries via the
`+define+` CLI parameter mechanism on tb_dpe_vmm.v / tb_dpe_acam.v /
tb_dsp_mac.v.

Both DPE arch files declare `module dpe` (matching the VTR arch XML
<model name="dpe"> blackbox port contract); only one of the two files
is compiled per case.

Cycle formulae (Option A, post Task #90 methodology roll-back):

  SIM cycles (ideal analytical, NO implementation overhead):
    T_fill_ideal = LOAD + COMPUTE + OUTPUT      (no +2)
    T_steady     = max(LOAD, COMPUTE, OUTPUT)
    SIM(M)       = T_fill_ideal + (M - 1) * T_steady

  RTL cycles (faithful behavior model — REAL silicon overhead):
    T_fill_rtl   = LOAD + COMPUTE + OUTPUT + 2  (FSM register-propagation
                   handoffs: LOAD→COMPUTE and COMPUTE→OUTPUT NBA cost)
    EXPECTED_RTL_CYCLES(M) = T_fill_rtl + (M - 1) * T_steady

  Fidelity = (observed - SIM_CYCLES) / SIM_CYCLES → reported, NOT gated.

Verifies for each (arch, R, C, BUF, PRECISION, PIPELINE_DEPTH) combo:
  1. functional correctness (TB asserts byte-level match)
  2. observed RTL cycles == EXPECTED_RTL_CYCLES (faithful FSM behavior)
  PASS requires both. Fidelity is reported as the honest measurement of
  the +2 implementation overhead that the RTL pays but the sim does not
  model.

The DPE module is precision-agnostic (Model Y). PRECISION_TB and
PIPELINE_DEPTH_TB +defines control the TB controller's hold duration
on nl_dpe_control = 2'b11; the DPE's S_COMPUTE waits for ctrl deassert.

    CCYC = PRECISION_BITS + PIPELINE_DEPTH - 1 + ACAM_CYCLES

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

# Per-arch CCYC decomposition (Task #86):
#   CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
#
# NL-DPE     : 2-stage pipeline (crossbar MAC -> analog accumulator) +
#              1-cycle ACAM read-out (always fires; activation mode just
#              selects the LUT contents, not whether ACAM fires).
#              PIPELINE_DEPTH=2, ACAM_CYCLES=1 → CCYC = P + 2.
# Azure-Lily : 3-stage pipeline (crossbar MAC -> ADC -> shift-add);
#              no ACAM stage.
#              PIPELINE_DEPTH=3, ACAM_CYCLES=0 → CCYC = P + 2.
#
# Both arches give CCYC = P + 2 — structural symmetry, not coincidence.
ARCH_PIPELINE = {
    "nldpe": dict(pipeline_depth=2, acam_cycles=1),
    "al":    dict(pipeline_depth=3, acam_cycles=0),
}


@dataclass
class Case:
    arch: str             # "NL-DPE" | "AL" | "NL-DPE_ACAM" | "AL_DSP_MAC"
    label: str            # short tag for log
    defines: dict         # +define+ key=value pairs passed to iverilog
    expected_cycles: int  # RTL cycle target (T_fill_rtl, with +2 implementation overhead)
    sim_cycles: int       # Ideal sim cycles (T_fill_ideal, NO +2 overhead) — Option A
    src_v: list           # list of .v paths to compile
    tb: str               # which TB ("dpe_vmm", "dpe_acam", "dsp_mac")
    pass_re: re.Pattern = field(default=re.compile(r"PASS"))
    fail_re: re.Pattern = field(default=re.compile(r"(FAIL|MISMATCH|ERROR)"))


def _compute_cycles(precision: int, pipeline_depth: int,
                    acam_cycles: int = 0) -> int:
    """Per-arch bit-serial pipeline cycle count (Task #86).

    CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES.

    NL-DPE     : pipeline_depth=2, acam_cycles=1 → P + 2.
    Azure-Lily : pipeline_depth=3, acam_cycles=0 → P + 2.

    Both give the same numerical CCYC under current parameters; the
    per-arch decomposition is principled (different stage counts +
    different ACAM presence) rather than a single-knob fudge.
    """
    return precision + (pipeline_depth - 1) + acam_cycles


# RTL implementation overhead — the faithful DPE behavior model pays +2
# cycles of FSM register-propagation overhead (LOAD→COMPUTE and
# COMPUTE→OUTPUT NBA handoffs). This is a real-silicon cost of OUR
# particular RTL design (NBA-propagated queue indices), NOT part of the
# §4 ideal analytical model. The sim emits the ideal cycle count
# (T_fill = L + C + O); the RTL pays +2; fidelity is the measurement.
_T_FILL_RTL_OVERHEAD = 2


def _t_fill_dpe_rtl(R: int, C: int, BUF: int, precision: int,
                    pipeline_depth: int, acam_cycles: int = 0) -> int:
    """RTL cycle target: T_fill_rtl = L + C + O + 2 (faithful FSM behavior)."""
    eps = BUF // 8
    lstr = math.ceil(R / eps)
    ocyc = math.ceil(C / eps)
    ccyc = _compute_cycles(precision, pipeline_depth, acam_cycles)
    return lstr + ccyc + ocyc + _T_FILL_RTL_OVERHEAD


def _t_fill_dpe_sim(R: int, C: int, BUF: int, precision: int,
                    pipeline_depth: int, acam_cycles: int = 0) -> int:
    """SIM cycle target (Option A ideal): T_fill_ideal = L + C + O. No +2."""
    eps = BUF // 8
    lstr = math.ceil(R / eps)
    ocyc = math.ceil(C / eps)
    ccyc = _compute_cycles(precision, pipeline_depth, acam_cycles)
    return lstr + ccyc + ocyc


def _t_fill_dsp_mac_rtl(K: int, BUF: int, DSP_WIDTH: int, PREC: int = 8) -> int:
    lstr = math.ceil(K * PREC / BUF)
    ccyc = max(1, math.ceil(K / DSP_WIDTH))
    ocyc_raw = math.ceil(PREC / BUF)
    ocyc = max(1, ocyc_raw)
    return lstr + ccyc + ocyc + _T_FILL_RTL_OVERHEAD


def _t_fill_dsp_mac_sim(K: int, BUF: int, DSP_WIDTH: int, PREC: int = 8) -> int:
    lstr = math.ceil(K * PREC / BUF)
    ccyc = max(1, math.ceil(K / DSP_WIDTH))
    ocyc_raw = math.ceil(PREC / BUF)
    ocyc = max(1, ocyc_raw)
    return lstr + ccyc + ocyc


def _t_msweep_dpe_rtl(R: int, C: int, BUF: int, precision: int,
                      pipeline_depth: int, M: int,
                      acam_cycles: int = 0) -> int:
    """RTL T(M) = T_fill_rtl + (M-1) * T_steady (with +2 FSM overhead)."""
    eps = BUF // 8
    lstr = math.ceil(R / eps)
    ocyc = math.ceil(C / eps)
    ccyc = _compute_cycles(precision, pipeline_depth, acam_cycles)
    t_fill = lstr + ccyc + ocyc + _T_FILL_RTL_OVERHEAD
    t_steady = max(lstr, ccyc, ocyc)
    return t_fill + (M - 1) * t_steady


def _t_msweep_dpe_sim(R: int, C: int, BUF: int, precision: int,
                      pipeline_depth: int, M: int,
                      acam_cycles: int = 0) -> int:
    """Sim T(M) = T_fill_ideal + (M-1) * T_steady (Option A — no +2)."""
    eps = BUF // 8
    lstr = math.ceil(R / eps)
    ocyc = math.ceil(C / eps)
    ccyc = _compute_cycles(precision, pipeline_depth, acam_cycles)
    t_fill = lstr + ccyc + ocyc
    t_steady = max(lstr, ccyc, ocyc)
    return t_fill + (M - 1) * t_steady


def _t_msweep_dsp_rtl(K: int, BUF: int, DSP_WIDTH: int, M: int,
                      PREC: int = 8) -> int:
    lstr = math.ceil(K * PREC / BUF)
    ccyc = max(1, math.ceil(K / DSP_WIDTH))
    ocyc_raw = math.ceil(PREC / BUF)
    ocyc = max(1, ocyc_raw)
    t_fill = lstr + ccyc + ocyc + _T_FILL_RTL_OVERHEAD
    t_steady = max(lstr, ccyc, ocyc)
    return t_fill + (M - 1) * t_steady


def _t_msweep_dsp_sim(K: int, BUF: int, DSP_WIDTH: int, M: int,
                      PREC: int = 8) -> int:
    lstr = math.ceil(K * PREC / BUF)
    ccyc = max(1, math.ceil(K / DSP_WIDTH))
    ocyc_raw = math.ceil(PREC / BUF)
    ocyc = max(1, ocyc_raw)
    t_fill = lstr + ccyc + ocyc
    t_steady = max(lstr, ccyc, ocyc)
    return t_fill + (M - 1) * t_steady


def _add_dpe_cases(cases, precision, pipeline_depth, quick):
    """Add NL-DPE / AL-DPE / ACAM cases for a given precision.

    Per-arch CCYC decomposition (Task #86): each arch passes its own
    (PIPELINE_DEPTH_TB, ACAM_CYCLES_TB) so the TB derives
        CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
    per arch. `pipeline_depth` is the legacy single-knob default; if
    the caller passes the legacy value (3), the per-arch table below
    overrides it for principled per-arch decomposition.
    """
    tag = f"P{precision}D{pipeline_depth}"
    nldpe_pd = ARCH_PIPELINE["nldpe"]["pipeline_depth"]
    nldpe_ac = ARCH_PIPELINE["nldpe"]["acam_cycles"]
    al_pd = ARCH_PIPELINE["al"]["pipeline_depth"]
    al_ac = ARCH_PIPELINE["al"]["acam_cycles"]

    # ── NL-DPE DPE VMM (BUF=40, R >= C) ──
    nldpe_shapes = [(256, 256), (512, 256), (1024, 256),
                    (256, 128), (512, 128), (1024, 128)]
    if not quick:
        nldpe_shapes += [(256, 64), (1024, 1024), (2048, 256)]
    for (R, C) in nldpe_shapes:
        defs = {
            "PRECISION_TB": precision,
            "PIPELINE_DEPTH_TB": nldpe_pd,
            "ACAM_CYCLES_TB": nldpe_ac,
            "ARCH_NLDPE": "1", "R_TB": R, "C_TB": C, "BUF_TB": 40,
        }
        cases.append(Case(
            arch="NL-DPE",
            label=f"VMM_NLDPE_R{R}_C{C}_{tag}",
            defines=defs,
            expected_cycles=_t_fill_dpe_rtl(R, C, 40, precision, nldpe_pd, nldpe_ac),
            sim_cycles=_t_fill_dpe_sim(R, C, 40, precision, nldpe_pd, nldpe_ac),
            src_v=[str(RTL / "dpe_nldpe.v"), str(TB_DIR / "tb_dpe_vmm.v")],
            tb="dpe_vmm",
        ))

    # ── AL DPE VMM (BUF=16, R >= C) ──
    al_shapes = [(512, 128), (1024, 128), (256, 128), (512, 64)]
    if not quick:
        al_shapes += [(256, 64), (1024, 64), (2048, 128)]
    for (R, C) in al_shapes:
        defs = {
            "PRECISION_TB": precision,
            "PIPELINE_DEPTH_TB": al_pd,
            "ACAM_CYCLES_TB": al_ac,
            "ARCH_AL": "1", "R_TB": R, "C_TB": C, "BUF_TB": 16,
        }
        cases.append(Case(
            arch="AL",
            label=f"VMM_AL_R{R}_C{C}_{tag}",
            defines=defs,
            expected_cycles=_t_fill_dpe_rtl(R, C, 16, precision, al_pd, al_ac),
            sim_cycles=_t_fill_dpe_sim(R, C, 16, precision, al_pd, al_ac),
            src_v=[str(RTL / "dpe_azurelily.v"), str(TB_DIR / "tb_dpe_vmm.v")],
            tb="dpe_vmm",
        ))

    # ── NL-DPE ACAM exp (BUF=40, R >= C) ──
    acam_shapes = [(256, 256), (512, 256), (1024, 256), (256, 128), (1024, 128)]
    if not quick:
        acam_shapes += [(256, 64), (2048, 256)]
    for (R, C) in acam_shapes:
        defs = {
            "PRECISION_TB": precision,
            "PIPELINE_DEPTH_TB": nldpe_pd,
            "ACAM_CYCLES_TB": nldpe_ac,
            "R_TB": R, "C_TB": C, "BUF_TB": 40,
        }
        cases.append(Case(
            arch="NL-DPE_ACAM",
            label=f"ACAM_NLDPE_R{R}_C{C}_{tag}",
            defines=defs,
            expected_cycles=_t_fill_dpe_rtl(R, C, 40, precision, nldpe_pd, nldpe_ac),
            sim_cycles=_t_fill_dpe_sim(R, C, 40, precision, nldpe_pd, nldpe_ac),
            src_v=[str(RTL / "dpe_nldpe.v"), str(TB_DIR / "tb_dpe_acam.v")],
            tb="dpe_acam",
        ))


def build_cases(quick: bool, precision_override: int | None) -> list[Case]:
    cases = []

    if precision_override is None:
        # Default sweep: full geometry matrix at PREC=8 (the canonical
        # INT8 model), plus a precision axis check at one (R,C) shape
        # for {4, 16} to verify CCYC scales linearly with precision.
        _add_dpe_cases(cases, DEFAULT_PRECISION, DEFAULT_PIPELINE_DEPTH, quick)

        # Precision-axis sanity (one shape per arch, P ∈ {4, 16}). Uses
        # per-arch (PIPELINE_DEPTH, ACAM_CYCLES) decomposition from
        # ARCH_PIPELINE (Task #86).
        nldpe_pd = ARCH_PIPELINE["nldpe"]["pipeline_depth"]
        nldpe_ac = ARCH_PIPELINE["nldpe"]["acam_cycles"]
        al_pd = ARCH_PIPELINE["al"]["pipeline_depth"]
        al_ac = ARCH_PIPELINE["al"]["acam_cycles"]
        for prec in (4, 16):
            tag = f"P{prec}D{DEFAULT_PIPELINE_DEPTH}"
            # NL-DPE @ R=256 C=256 BUF=40
            cases.append(Case(
                arch="NL-DPE",
                label=f"VMM_NLDPE_R256_C256_{tag}",
                defines={"ARCH_NLDPE": "1", "R_TB": 256, "C_TB": 256,
                         "BUF_TB": 40, "PRECISION_TB": prec,
                         "PIPELINE_DEPTH_TB": nldpe_pd,
                         "ACAM_CYCLES_TB": nldpe_ac},
                expected_cycles=_t_fill_dpe_rtl(256, 256, 40, prec,
                                                nldpe_pd, nldpe_ac),
                sim_cycles=_t_fill_dpe_sim(256, 256, 40, prec,
                                           nldpe_pd, nldpe_ac),
                src_v=[str(RTL / "dpe_nldpe.v"),
                       str(TB_DIR / "tb_dpe_vmm.v")],
                tb="dpe_vmm",
            ))
            # AL @ R=512 C=128 BUF=16
            cases.append(Case(
                arch="AL",
                label=f"VMM_AL_R512_C128_{tag}",
                defines={"ARCH_AL": "1", "R_TB": 512, "C_TB": 128,
                         "BUF_TB": 16, "PRECISION_TB": prec,
                         "PIPELINE_DEPTH_TB": al_pd,
                         "ACAM_CYCLES_TB": al_ac},
                expected_cycles=_t_fill_dpe_rtl(512, 128, 16, prec,
                                                al_pd, al_ac),
                sim_cycles=_t_fill_dpe_sim(512, 128, 16, prec,
                                           al_pd, al_ac),
                src_v=[str(RTL / "dpe_azurelily.v"),
                       str(TB_DIR / "tb_dpe_vmm.v")],
                tb="dpe_vmm",
            ))
            # NL-DPE ACAM @ R=256 C=256 BUF=40
            cases.append(Case(
                arch="NL-DPE_ACAM",
                label=f"ACAM_NLDPE_R256_C256_{tag}",
                defines={"R_TB": 256, "C_TB": 256, "BUF_TB": 40,
                         "PRECISION_TB": prec,
                         "PIPELINE_DEPTH_TB": nldpe_pd,
                         "ACAM_CYCLES_TB": nldpe_ac},
                expected_cycles=_t_fill_dpe_rtl(256, 256, 40, prec,
                                                nldpe_pd, nldpe_ac),
                sim_cycles=_t_fill_dpe_sim(256, 256, 40, prec,
                                           nldpe_pd, nldpe_ac),
                src_v=[str(RTL / "dpe_nldpe.v"),
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
            expected_cycles=_t_fill_dsp_mac_rtl(K, 16, 4),
            sim_cycles=_t_fill_dsp_mac_sim(K, 16, 4),
            src_v=[str(RTL / "dsp_mac.v"), str(TB_DIR / "tb_dsp_mac.v")],
            tb="dsp_mac",
        ))

    # ── M-sweep cases: drain-load overlap pipeline (FIDELITY_METHODOLOGY §4) ──
    # Verifies T(M) = T_fill + (M-1)*T_steady where T_steady = max(L, C, O).
    # Per-arch (PIPELINE_DEPTH, ACAM_CYCLES) decomposition from ARCH_PIPELINE.
    M_SWEEP = [1, 2, 4, 8]
    nldpe_pd = ARCH_PIPELINE["nldpe"]["pipeline_depth"]
    nldpe_ac = ARCH_PIPELINE["nldpe"]["acam_cycles"]
    al_pd = ARCH_PIPELINE["al"]["pipeline_depth"]
    al_ac = ARCH_PIPELINE["al"]["acam_cycles"]

    # NL-DPE M-sweep at canonical R=C=256 BUF=40 INT8.
    for M in M_SWEEP:
        cases.append(Case(
            arch="NL-DPE_MSW",
            label=f"MSW_NLDPE_R256_C256_M{M}",
            defines={"ARCH_NLDPE": "1",
                     "R_TB": 256, "C_TB": 256, "BUF_TB": 40,
                     "PRECISION_TB": 8,
                     "PIPELINE_DEPTH_TB": nldpe_pd,
                     "ACAM_CYCLES_TB": nldpe_ac,
                     "M_TB": M},
            expected_cycles=_t_msweep_dpe_rtl(256, 256, 40, 8, nldpe_pd, M, nldpe_ac),
            sim_cycles=_t_msweep_dpe_sim(256, 256, 40, 8, nldpe_pd, M, nldpe_ac),
            src_v=[str(RTL / "dpe_nldpe.v"),
                   str(TB_DIR / "tb_dpe_vmm_msweep.v")],
            tb="dpe_msweep",
            pass_re=re.compile(r"\[tb_dpe_vmm_msweep\]\s*PASS"),
            fail_re=re.compile(r"\[tb_dpe_vmm_msweep\]\s*FAIL|MISMATCH|ERROR"),
        ))

    # NL-DPE M-sweep at smaller R=128 C=128 to vary geometry.
    for M in M_SWEEP:
        cases.append(Case(
            arch="NL-DPE_MSW",
            label=f"MSW_NLDPE_R256_C128_M{M}",
            defines={"ARCH_NLDPE": "1",
                     "R_TB": 256, "C_TB": 128, "BUF_TB": 40,
                     "PRECISION_TB": 8,
                     "PIPELINE_DEPTH_TB": nldpe_pd,
                     "ACAM_CYCLES_TB": nldpe_ac,
                     "M_TB": M},
            expected_cycles=_t_msweep_dpe_rtl(256, 128, 40, 8, nldpe_pd, M, nldpe_ac),
            sim_cycles=_t_msweep_dpe_sim(256, 128, 40, 8, nldpe_pd, M, nldpe_ac),
            src_v=[str(RTL / "dpe_nldpe.v"),
                   str(TB_DIR / "tb_dpe_vmm_msweep.v")],
            tb="dpe_msweep",
            pass_re=re.compile(r"\[tb_dpe_vmm_msweep\]\s*PASS"),
            fail_re=re.compile(r"\[tb_dpe_vmm_msweep\]\s*FAIL|MISMATCH|ERROR"),
        ))

    # AL M-sweep at canonical R=512 C=128 BUF=16 INT8.
    for M in M_SWEEP:
        cases.append(Case(
            arch="AL_MSW",
            label=f"MSW_AL_R512_C128_M{M}",
            defines={"ARCH_AL": "1",
                     "R_TB": 512, "C_TB": 128, "BUF_TB": 16,
                     "PRECISION_TB": 8,
                     "PIPELINE_DEPTH_TB": al_pd,
                     "ACAM_CYCLES_TB": al_ac,
                     "M_TB": M},
            expected_cycles=_t_msweep_dpe_rtl(512, 128, 16, 8, al_pd, M, al_ac),
            sim_cycles=_t_msweep_dpe_sim(512, 128, 16, 8, al_pd, M, al_ac),
            src_v=[str(RTL / "dpe_azurelily.v"),
                   str(TB_DIR / "tb_dpe_vmm_msweep.v")],
            tb="dpe_msweep",
            pass_re=re.compile(r"\[tb_dpe_vmm_msweep\]\s*PASS"),
            fail_re=re.compile(r"\[tb_dpe_vmm_msweep\]\s*FAIL|MISMATCH|ERROR"),
        ))

    # DSP-MAC M-sweep (default K=64).
    for M in M_SWEEP:
        cases.append(Case(
            arch="AL_DSP_MAC_MSW",
            label=f"MSW_DSPMAC_K64_M{M}",
            defines={"K_TB": 64, "BUF_TB": 16, "DSP_WIDTH_TB": 4,
                     "M_TB": M},
            expected_cycles=_t_msweep_dsp_rtl(64, 16, 4, M),
            sim_cycles=_t_msweep_dsp_sim(64, 16, 4, M),
            src_v=[str(RTL / "dsp_mac.v"),
                   str(TB_DIR / "tb_dsp_mac_msweep.v")],
            tb="dsp_mac_msweep",
            pass_re=re.compile(r"\[tb_dsp_mac_msweep\]\s*PASS"),
            fail_re=re.compile(r"\[tb_dsp_mac_msweep\]\s*FAIL|MISMATCH|ERROR"),
        ))

    return cases


def run_case(c: Case, tmpdir: Path) -> tuple[bool, str, int | None, float | None]:
    """Compile + simulate one case.

    Returns (passed, summary_line, observed_cycles, fidelity_pct).
    PASS = functional + observed RTL cycles match expected_cycles (RTL contract).
    Fidelity = (observed - sim_cycles) / sim_cycles — REPORTED only, NOT gating.
    """
    bin_path = tmpdir / f"tb_{c.label}"
    cmd_iv = ["iverilog", "-o", str(bin_path)]
    for k, v in c.defines.items():
        cmd_iv.extend(["-D", f"{k}={v}"])
    cmd_iv.extend(c.src_v)
    proc = subprocess.run(cmd_iv, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"COMPILE_FAIL:\n{(proc.stdout + proc.stderr)[-400:]}", None, None

    proc = subprocess.run(["vvp", str(bin_path)], capture_output=True, text=True, timeout=30)
    out = proc.stdout + proc.stderr

    # Parse total_cycles and PASS/FAIL
    m_cycles = re.search(r"total_cycles=(\d+)", out)
    observed_cycles = int(m_cycles.group(1)) if m_cycles else None

    pass_hit = bool(c.pass_re.search(out))
    fail_hit = bool(c.fail_re.search(out))
    cycle_match = (observed_cycles == c.expected_cycles) if observed_cycles is not None else False

    passed = pass_hit and not fail_hit and cycle_match
    # Fidelity (reported, not gated): (RTL - Sim) / Sim — the honest measure
    # of the +2 FSM implementation overhead in our particular DPE RTL.
    if observed_cycles is not None and c.sim_cycles > 0:
        fidelity_pct = 100.0 * (observed_cycles - c.sim_cycles) / c.sim_cycles
    else:
        fidelity_pct = None
    summary = (
        f"  observed={observed_cycles}, rtl_exp={c.expected_cycles}, sim={c.sim_cycles}, "
        f"PASS={pass_hit}, FAIL={fail_hit}, cycle_match={cycle_match}"
    )
    return passed, summary, observed_cycles, fidelity_pct


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
        passed, summary, obs, fid_pct = run_case(c, tmpdir)
        status = "PASS" if passed else "FAIL"
        fid_str = f"{fid_pct:+.2f}%" if fid_pct is not None else "n/a"
        line = (f"[{status}] {c.label:<40} "
                f"rtl_exp={c.expected_cycles:>6} sim={c.sim_cycles:>6} "
                f"observed={obs}  fidelity={fid_str}")
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
