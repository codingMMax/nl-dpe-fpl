#!/usr/bin/env python3
"""FC / GEMM smoke harness for Stages 1A + 1B + 1C.

Drives the fc_top + tb_fc + per-arch FAITHFUL dpe primitive across a
catalogue of (M, K, N, activation, arch) cases. Stages:

  1A — V=1, H=1 single-DPE (single tile, no K reduction, no N concat).
  1B — V>1, H=1 K-tile reduction (V parallel DPEs + pipelined CLB tree).
  1C — V=1, H>1 N-tile concatenation (H parallel DPEs + per-h output BRAM).

Cycle formulae (Task #98 — unified analytical formula, single source
of truth, NO compensation constants):

  SIM cycles (architectural minimum; matches
  azurelily/IMC/imc_core/imc_core.py:run_gemm):

    T_fill_sim   = LCYC + CCYC + OCYC
    T_steady_sim = max(LCYC, CCYC, OCYC)
    SIM_CYCLES   = T_fill_sim + (M - 1) * T_steady_sim

  Task #99 — double-buffered LOAD substrate: the faithful primitive
  owns two substrates A/B; pass-(k+1) LOAD writes substrate B while
  pass-k COMPUTE reads substrate A. There is no LOAD-gate and no
  +PRECISION term in T_steady. The wrapper streams strobes back-to-back
  across passes (no predictive PRECISION-cycle stall).

  RTL cycles (synthesizable fc_top.v + FAITHFUL primitive) DIFFER from
  SIM_CYCLES by real silicon costs:

    delta = rtl_obs - sim_exp  =  nba + tree + clb + wrap

  Per-stage delta decomposition (Task #98):

    nba   — primitive NBA handoff (LOAD→COMPUTE wake + COMPUTE→OUTPUT
            wake). In the standalone faithful primitive TB this is +2.
            In the fc_top workload it appears absorbed by the
            BRAM-read pipeline and is empirically 0 (workload delta
            does NOT decompose into 2+4; it is constant 6 wrapper).
    tree  — ⌈log₂(V)⌉ CLB adder tree pipeline depth (V > 1 only).
    clb   — +1 CLB cycle if (V > 1) OR (act AND NOT has_acam).
    wrap  — constant wrapper plumbing overhead. Empirically 6 cycles:
              +1 BRAM-read pipeline on LOAD
              +1 dpe_done_r register (DPE → wrapper handshake)
              +1 stage-0 sign-extend latch (CLB tree entry)
              +1 BRAM-write tap NBA
              +1 BRAM internal write commit
              +1 last-strobe drive-cycle
            Paid ONCE per workload (in T_fill), independent of M.

  delta_total = nba + tree + clb + wrap     (target identity)

Per-case workflow:
  1. Compute V = ceil(K/R), H = ceil(N/C).
  2. Compute LCYC, OCYC, CCYC, T_fill_sim, T_steady_sim.
  3. Compute SIM_CYCLES = T_fill_sim + (M-1) * T_steady_sim.
  4. Compile via iverilog with the right `+define+` knobs.
  5. Run vvp; parse PASS/FAIL + observed total_cycles.
  6. Compute delta = observed - SIM_CYCLES and decompose into
     nba/tree/clb/wrap per the formulae above.
  7. PASS criteria: FUNCTIONAL ONLY (per Task #97/#98: cycle delta is
     reported but does not gate PASS/FAIL).

Output: stdout summary AND fc_verification/results/fc_smoke.log.
Exits nonzero if any case fails functional check.

Usage:
    python3 fc_verification/run_fc_smoke.py                       # all stages
    python3 fc_verification/run_fc_smoke.py --stage 1A             # V=1 H=1 only
    python3 fc_verification/run_fc_smoke.py --stage 1B             # V>1 H=1 only
    python3 fc_verification/run_fc_smoke.py --stage 1C             # V=1 H>1 only
    python3 fc_verification/run_fc_smoke.py --workload bert_qkv_proj_NL
    python3 fc_verification/run_fc_smoke.py --shape 4,256,256 --activation none --arch nldpe
    python3 fc_verification/run_fc_smoke.py --arch al
    python3 fc_verification/run_fc_smoke.py --quick
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

DEFAULT_PRECISION = 8
DEFAULT_PIPELINE_DEPTH = 3

# Per-arch DPE geometry (must match the verified primitives) and per-arch
# CCYC decomposition (Task #86):
#   CCYC = PRECISION + (pipeline_depth - 1) + acam_cycles
#
# NL-DPE     : pipeline_depth=2 (MAC, Acc) + acam_cycles=1 (read-out)
# Azure-Lily : pipeline_depth=3 (MAC, ADC, SA) + acam_cycles=0 (no ACAM)
#
# Both arches give CCYC = P + 2 at every precision under current params —
# structural symmetry, not coincidence: (D_NL-1)+ACAM_NL = (D_AL-1)+ACAM_AL = 2.
ARCH_PARAMS = {
    "nldpe": dict(R=256, C=256, BUF=40, has_acam=1,
                  pipeline_depth=2, acam_cycles=1,
                  # Task #97: bind faithful primitive (silicon-faithful
                  # single-substrate slice with LOAD-gated cadence).
                  rtl_file=str(RTL / "dpe_nldpe_faithful.v"),
                  arch_define="ARCH_NLDPE",
                  fc_top_define="FC_TOP_ARCH_NLDPE"),
    "al":    dict(R=512, C=128, BUF=16, has_acam=0,
                  pipeline_depth=3, acam_cycles=0,
                  rtl_file=str(RTL / "dpe_azurelily_faithful.v"),
                  arch_define="ARCH_AL",
                  fc_top_define=None),
}

# Stage 1A workload catalogue (V=1, H=1).
STAGE_1A_WORKLOADS = [
    # (label,            M, K,   N,   activation, notes)
    ("bert_qkv_proj_NL", 1, 128, 128, "relu",     "trivial single-DPE NL"),
    ("gemm_trivial_NL",  1, 256, 256, "none",     "GEMM no activation NL (V=1 H=1 exact)"),
    ("gemm_batched_NL",  4, 256, 256, "none",     "batched GEMM NL"),
    ("gemm_batch8_NL",   8, 256, 256, "none",     "M=8 batched GEMM NL"),
    ("bert_qkv_proj_AL", 1, 128, 128, "relu",     "trivial single-DPE AL"),
    ("gemm_trivial_AL",  1, 512, 128, "none",     "AL V=1 H=1"),
    ("gemm_batched_AL",  4, 512, 128, "none",     "AL batched"),
]

# Stage 1B workload catalogue (V>1, H=1).
# K > R triggers V>1; N <= C keeps H=1.
STAGE_1B_WORKLOADS = [
    # (label,           M, K,    N,   activation, notes)
    ("lenet_fc1_NL",    1,  400, 120, "relu",     "V=2 H=1 NL-DPE (R=C=256 → V=2 H=1)"),
    ("gemm_v2_synth_NL",1,  512, 256, "none",     "V=2 H=1 synthetic NL"),
    ("gemm_v2_AL",      1, 1024, 128, "none",     "V=2 H=1 AL (R=512 → V=2; C=128 → H=1)"),
]

# Stage 1C workload catalogue (V=1, H>1).
# K <= R keeps V=1; N > C triggers H>1.
STAGE_1C_WORKLOADS = [
    # (label,            M, K,   N,   activation, notes)
    ("bert_ffn1_NL",     1, 128, 512, "relu",     "V=1 H=2 NL-DPE (K<R, N=2C)"),
    ("synthetic_h2_NL",  1, 256, 512, "none",     "V=1 H=2 NL-DPE"),
    ("synthetic_h2_AL",  1, 256, 256, "none",     "V=1 H=2 AL (K<R=512, C=128 → H=2)"),
]

# All workloads stitched together (Stage 1A first; Stage 1B; Stage 1C).
ALL_WORKLOADS = STAGE_1A_WORKLOADS + STAGE_1B_WORKLOADS + STAGE_1C_WORKLOADS


@dataclass
class Case:
    label: str
    arch: str
    M: int
    K: int
    N: int
    activation: str          # "none" | "relu"
    R: int
    C: int
    BUF: int
    PRECISION: int
    PIPELINE_DEPTH: int      # per-arch (NL=2, AL=3)
    ACAM_CYCLES: int         # per-arch (NL=1, AL=0)
    has_acam: int
    # Computed:
    V: int = 0
    H: int = 0
    LCYC: int = 0
    OCYC: int = 0
    CCYC: int = 0
    T_FILL_SIM: int = 0          # SIM: LCYC + CCYC + OCYC  (Task #98 unified)
    T_STEADY_SIM: int = 0        # Task #99 cadence: max(L, C, O)
    SIM_CYCLES: int = 0          # SIM total = T_fill_sim + (M-1)*T_steady_sim
    # Per-stage delta predictions (reported as info; sum to total delta):
    DELTA_NBA: int = 0           # NBA handoffs (primitive). 0 in fc_top wrapper
                                 # (absorbed by BRAM-read pipeline).
    DELTA_TREE: int = 0          # ⌈log₂(V)⌉ for V > 1 else 0
    DELTA_CLB: int = 0           # +1 if (V > 1) OR (act AND NOT has_acam)
    DELTA_WRAP: int = 0          # +6 constant wrapper plumbing
    DELTA_TOTAL: int = 0         # sum of the above; target = rtl_obs - sim_exp


def fill_case_metrics(c: Case) -> Case:
    c.V = math.ceil(c.K / c.R)
    c.H = math.ceil(c.N / c.C)
    eps = c.BUF // 8
    c.LCYC = math.ceil(c.R / eps)
    c.OCYC = math.ceil(c.C / eps)
    # Per-arch CCYC decomposition (Task #86):
    #   CCYC = PRECISION + (PIPELINE_DEPTH - 1) + ACAM_CYCLES
    c.CCYC = c.PRECISION + (c.PIPELINE_DEPTH - 1) + c.ACAM_CYCLES
    # SIM T_fill — Task #98 unified architectural minimum (no compensation
    # constants):
    c.T_FILL_SIM = c.LCYC + c.CCYC + c.OCYC
    # T_steady: Task #99 double-buffered LOAD removes the LOAD-gate, so
    # T_steady = max(LCYC, CCYC, OCYC) (no +PRECISION). Both SIM and RTL
    # follow this cadence — the wrapper streams strobes back-to-back
    # across passes; the primitive's substrate-tag flow handles WAR.
    c.T_STEADY_SIM = max(c.LCYC, c.CCYC, c.OCYC)
    # SIM_CYCLES — Task #98 unified (matches imc_core.run_gemm):
    #   sim = T_fill_sim + (M - 1) * T_steady_sim
    c.SIM_CYCLES = c.T_FILL_SIM + (c.M - 1) * c.T_STEADY_SIM
    # ── Per-stage delta predictions (target identity:
    #     delta_total = nba + tree + clb + wrap == rtl_obs - sim_exp) ──
    # NBA handoff cost: the primitive pays +2 internally, but in the
    # synthesizable fc_top.v workload it is absorbed by the BRAM-read
    # pipeline and registered DPE handshake. Empirically the workload
    # delta is constant 6 wrapper across all (V, H, act) combinations,
    # so we model NBA = 0 at the workload layer. The primitive faithful
    # smoke TB reports its own +2 delta separately.
    c.DELTA_NBA = 0
    # TREE_PIPE: ⌈log₂(V)⌉ for V > 1, else 0 (CLB adder tree depth).
    if c.V > 1:
        tree_pipe = 0
        val = c.V - 1
        while val > 0:
            tree_pipe += 1
            val >>= 1
    else:
        tree_pipe = 0
    c.DELTA_TREE = tree_pipe
    # CLB_NEEDED gate: +1 if (V > 1) OR (act AND NOT has_acam).
    act_mode = 1 if c.activation == "relu" else 0
    c.DELTA_CLB = int((c.V > 1) or (act_mode == 1 and c.has_acam == 0))
    # Wrapper plumbing: constant +6 paid once in T_fill, independent of M.
    c.DELTA_WRAP = 6
    c.DELTA_TOTAL = c.DELTA_NBA + c.DELTA_TREE + c.DELTA_CLB + c.DELTA_WRAP
    return c


def build_cases(args) -> list[Case]:
    """Build the case list filtered by --workload / --arch / --shape / etc."""
    cases: list[Case] = []
    if args.shape is not None:
        # Custom shape from CLI, applied to chosen arch(es).
        m_str, k_str, n_str = args.shape.split(",")
        M, K, N = int(m_str), int(k_str), int(n_str)
        act = args.activation if args.activation else "none"
        archs = [args.arch] if args.arch in ARCH_PARAMS else list(ARCH_PARAMS.keys())
        for arch in archs:
            ap = ARCH_PARAMS[arch]
            label = f"custom_{arch}_M{M}_K{K}_N{N}_{act}"
            c = Case(
                label=label, arch=arch, M=M, K=K, N=N, activation=act,
                R=ap["R"], C=ap["C"], BUF=ap["BUF"],
                PRECISION=args.precision,
                PIPELINE_DEPTH=ap["pipeline_depth"],
                ACAM_CYCLES=ap["acam_cycles"],
                has_acam=ap["has_acam"],
            )
            cases.append(fill_case_metrics(c))
        return cases

    # Catalogue mode: enumerate ALL_WORKLOADS (1A + 1B + 1C), filtered
    # by --workload, --arch, and --stage.
    for label, M, K, N, act_default, _notes in ALL_WORKLOADS:
        # Determine arch from label suffix (_NL vs _AL).
        if label.endswith("_NL"):
            arch = "nldpe"
        elif label.endswith("_AL"):
            arch = "al"
        else:
            continue

        # Filter by --arch.
        if args.arch == "nldpe" and arch != "nldpe":
            continue
        if args.arch == "al" and arch != "al":
            continue

        # Filter by --workload.
        if args.workload is not None and args.workload != label:
            continue

        act = args.activation if args.activation else act_default
        ap = ARCH_PARAMS[arch]
        c = Case(
            label=label, arch=arch, M=M, K=K, N=N, activation=act,
            R=ap["R"], C=ap["C"], BUF=ap["BUF"],
            PRECISION=args.precision,
            PIPELINE_DEPTH=ap["pipeline_depth"],
            ACAM_CYCLES=ap["acam_cycles"],
            has_acam=ap["has_acam"],
        )
        c = fill_case_metrics(c)

        # Stage filter: 1A → V==1 H==1; 1B → V>1 H==1; 1C → V==1 H>1;
        # 1D → V>1 H>1; 'all' → no filter.
        if args.stage == "1A" and not (c.V == 1 and c.H == 1):
            continue
        if args.stage == "1B" and not (c.V > 1 and c.H == 1):
            continue
        if args.stage == "1C" and not (c.V == 1 and c.H > 1):
            continue
        if args.stage == "1D" and not (c.V > 1 and c.H > 1):
            continue

        cases.append(c)

    if args.quick:
        cases = cases[:2]

    return cases


def run_case(c: Case, tmpdir: Path, keep: bool) -> tuple[bool, str, dict]:
    """Compile + simulate one case. Returns (passed, summary, metrics)."""
    bin_path = tmpdir / f"tb_fc_{c.label}"
    ap = ARCH_PARAMS[c.arch]

    cmd_iv = ["iverilog", "-g2005", "-o", str(bin_path)]
    cmd_iv.extend(["-D", ap["arch_define"]])
    if ap["fc_top_define"]:
        cmd_iv.extend(["-D", ap["fc_top_define"]])
    cmd_iv.extend([
        "-D", f"M_TB={c.M}",
        "-D", f"K_TB={c.K}",
        "-D", f"N_TB={c.N}",
        "-D", f"R_TB={c.R}",
        "-D", f"C_TB={c.C}",
        "-D", f"BUF_TB={c.BUF}",
        "-D", f"PRECISION_TB={c.PRECISION}",
        "-D", f"PIPELINE_DEPTH_TB={c.PIPELINE_DEPTH}",
        "-D", f"ACAM_CYCLES_TB={c.ACAM_CYCLES}",
        "-D", f"ACTIVATION_TB={1 if c.activation == 'relu' else 0}",
    ])
    cmd_iv.extend([
        ap["rtl_file"],
        str(RTL / "fc_top.v"),
        str(TB_DIR / "tb_fc.v"),
    ])

    proc = subprocess.run(cmd_iv, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = f"COMPILE_FAIL:\n{(proc.stdout + proc.stderr)[-600:]}"
        return False, msg, {}

    proc = subprocess.run(
        ["vvp", str(bin_path)],
        capture_output=True, text=True, timeout=180
    )
    out = proc.stdout + proc.stderr

    # Parse total_cycles
    m = re.search(r"total_cycles=(\d+)", out)
    observed = int(m.group(1)) if m else None

    # Functional PASS / FAIL parsing (the tb_fc PASS line may be emitted
    # only on functional success; the FAIL line on any mismatch).
    pass_hit = bool(re.search(r"\[tb_fc\]\s*FUNCTIONAL_PASS\b", out))
    fail_hit = bool(re.search(r"\[tb_fc\]\s*FUNCTIONAL_FAIL\b", out))

    # PASS = functional only (per Task #97 user gate).
    passed = pass_hit and not fail_hit

    # Cycle delta = observed - SIM_CYCLES (Task #98 unified sim formula).
    # Target identity: delta = nba + tree + clb + wrap (predicted via the
    # per-stage breakdown in fill_case_metrics).
    if observed is not None and c.SIM_CYCLES > 0:
        delta = observed - c.SIM_CYCLES
    else:
        delta = None

    summary = (
        f"  observed={observed}  sim_exp={c.SIM_CYCLES}  delta={delta:+d}  "
        f"(nba={c.DELTA_NBA}, tree={c.DELTA_TREE}, "
        f"clb={c.DELTA_CLB}, wrap={c.DELTA_WRAP}, "
        f"pred_total={c.DELTA_TOTAL})  "
        f"functional={'PASS' if pass_hit else 'FAIL'}"
        if delta is not None else
        f"  observed={observed}  sim_exp={c.SIM_CYCLES}"
    )

    metrics = dict(
        label=c.label, arch=c.arch, M=c.M, K=c.K, N=c.N,
        activation=c.activation, V=c.V, H=c.H,
        LCYC=c.LCYC, CCYC=c.CCYC, OCYC=c.OCYC,
        T_FILL_SIM=c.T_FILL_SIM,
        T_STEADY_SIM=c.T_STEADY_SIM,
        SIM_CYCLES=c.SIM_CYCLES,
        DELTA_NBA=c.DELTA_NBA, DELTA_TREE=c.DELTA_TREE,
        DELTA_CLB=c.DELTA_CLB, DELTA_WRAP=c.DELTA_WRAP,
        DELTA_TOTAL=c.DELTA_TOTAL,
        observed=observed, delta=delta,
        pass_hit=pass_hit, fail_hit=fail_hit,
        passed=passed,
    )

    if not passed:
        # Tail of vvp output to help debugging
        metrics["tail"] = out.splitlines()[-12:]

    return passed, summary, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default=None,
                    help="filter to specific workload label (e.g. bert_qkv_proj_NL)")
    ap.add_argument("--shape", default=None,
                    help="custom workload shape M,K,N (overrides catalogue)")
    ap.add_argument("--activation", choices=("none", "relu"), default=None,
                    help="override per-workload activation")
    ap.add_argument("--arch", choices=("nldpe", "al", "both"), default="both",
                    help="filter by arch (default: both)")
    ap.add_argument("--precision", type=int, default=DEFAULT_PRECISION,
                    help=f"INT precision (default {DEFAULT_PRECISION})")
    ap.add_argument("--stage", choices=("1A", "1B", "1C", "1D", "all"),
                    default="all",
                    help="Stage filter: 1A=V=1 H=1, 1B=V>1 H=1, "
                         "1C=V=1 H>1, 1D=V>1 H>1; default 'all' "
                         "(union of 1A+1B+1C; no 1D cases yet)")
    ap.add_argument("--quick", action="store_true",
                    help="smaller subset (first 2 cases)")
    ap.add_argument("--keep", action="store_true",
                    help="don't delete tmp build dir")
    ap.add_argument("--jobs", type=int, default=1,
                    help="(reserved; harness is currently serial)")
    args = ap.parse_args()

    cases = build_cases(args)
    if not cases:
        print("[run_fc_smoke] no cases selected (after filters)")
        return 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "fc_smoke.log"

    tmpdir = Path(tempfile.mkdtemp(prefix="fc_smoke_"))
    print(f"=== FC smoke sweep — {len(cases)} cases ===")
    print(f"Stage:      {args.stage}")
    print(f"Precision:  {args.precision}")
    print(f"tmpdir:     {tmpdir}")
    print(f"log:        {log_path}\n")

    log_lines = [
        "FC / GEMM smoke harness",
        f"Stage:      {args.stage}",
        f"Mode:       {'quick' if args.quick else 'full'}",
        f"Precision:  {args.precision}",
        f"Cases:      {len(cases)}",
        "",
    ]

    n_pass = n_fail = 0
    fail_records = []

    # Per Task #98: single source of truth = unified architectural minimum
    # T_fill = LCYC + CCYC + OCYC. The delta column decomposes into
    # per-stage deltas (nba / tree / clb / wrap) summing to delta_total.
    # PASS column is functional-only.
    header = (
        f"{'STATUS':<8} {'LABEL':<22} {'ARCH':<6} "
        f"{'M':>3} {'K':>5} {'N':>5} {'V':>3} {'H':>3} "
        f"{'ACT':>4}  "
        f"{'LCYC':>5} {'CCYC':>4} {'OCYC':>4}  "
        f"{'Tf_sim':>7} {'Tstd_sim':>8}  "
        f"{'sim_exp':>8} {'rtl_obs':>8} {'delta':>6}  "
        f"{'nba':>4} {'tree':>5} {'clb':>4} {'wrap':>5}"
    )
    print(header)
    print("-" * len(header))
    log_lines.append(header)
    log_lines.append("-" * len(header))

    for c in cases:
        passed, summary, metrics = run_case(c, tmpdir, args.keep)
        status = "PASS" if passed else "FAIL"
        obs = metrics.get("observed")
        obs_str = str(obs) if obs is not None else "n/a"
        delta = metrics.get("delta")
        delta_str = f"{delta:+d}" if delta is not None else "n/a"
        line = (
            f"{status:<8} {c.label:<22} {c.arch:<6} "
            f"{c.M:>3} {c.K:>5} {c.N:>5} {c.V:>3} {c.H:>3} "
            f"{c.activation:>4}  "
            f"{c.LCYC:>5} {c.CCYC:>4} {c.OCYC:>4}  "
            f"{c.T_FILL_SIM:>7} {c.T_STEADY_SIM:>8}  "
            f"{c.SIM_CYCLES:>8} {obs_str:>8} {delta_str:>6}  "
            f"{c.DELTA_NBA:>4} {c.DELTA_TREE:>5} "
            f"{c.DELTA_CLB:>4} {c.DELTA_WRAP:>5}"
        )
        print(line)
        log_lines.append(line)
        if not passed:
            log_lines.append(summary)
            for tl in metrics.get("tail", []):
                log_lines.append(f"    {tl}")
            log_lines.append("")
            fail_records.append((c.label, summary))
            n_fail += 1
        else:
            n_pass += 1

    print("-" * len(header))
    summary_line = f"\nSUMMARY: {n_pass}/{len(cases)} PASS, {n_fail} FAIL"
    log_lines.append("-" * len(header))
    log_lines.append(summary_line)

    print(summary_line)
    print(f"Log: {log_path}")

    log_path.write_text("\n".join(log_lines) + "\n")

    if not args.keep:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
