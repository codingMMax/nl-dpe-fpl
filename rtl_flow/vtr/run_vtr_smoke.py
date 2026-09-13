#!/usr/bin/env python3
"""VTR smoke test for the Phase 2 synthesizable fc_top.v (Task #89).

Drives the (renamed) `fc_top_synth` wrapper through VTR's CAD flow:
  Verilog (Parmys) → ABC → VPR (place + route) → resource + Fmax report.

For each target workload:
  1. Compute V, H, LCYC, OCYC, CCYC from per-arch geometry.
  2. Generate a per-workload top wrapper that instantiates `fc_top_synth`
     with the workload's (M, K, N, ACTIVATION_MODE) and the NL-DPE arch's
     (R, C, BUF=40, PIPELINE_DEPTH=2, ACAM_CYCLES=1, HAS_ACAM=1).
  3. Concatenate fc_top_synth.v + dpe_blackbox.v + wrapper into a
     single-file circuit, written to a temp dir.
  4. Run `run_vtr_flow.py` on it against `nl_dpe_22nm_auto.xml`.
  5. Parse the resulting `vpr.out` (or `vpr_stdout.log`) for:
       - Block counts (dpe, memory, dsp_top, clb)
       - Final Fmax (MHz)
       - Wirelength
  6. Emit a per-workload row.

Usage:
    python3 rtl_flow/vtr/run_vtr_smoke.py                    # P1 only
    python3 rtl_flow/vtr/run_vtr_smoke.py --all-priority      # P1 + P2 + P3
    python3 rtl_flow/vtr/run_vtr_smoke.py --workload lenet_fc1_NL
    python3 rtl_flow/vtr/run_vtr_smoke.py --route-chan-width 200
    python3 rtl_flow/vtr/run_vtr_smoke.py --keep-temp

Outputs:
  - rtl_flow/results/vtr_smoke_{workload}/ (per-workload VTR scratch)
  - rtl_flow/results/vtr_smoke.log (aggregated summary)

This file deliberately does NOT touch:
  - rtl_flow/rtl/fc_top.v (the sim-verified Phase 2 master)
  - rtl_flow/rtl/dpe_nldpe.v / dpe_azurelily.v / dsp_mac.v
  - nl_dpe/run_vtr.py (the legacy benchmark-RTL VTR runner)
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RTL = REPO / "rtl_flow" / "rtl"
RESULTS = REPO / "rtl_flow" / "results"

FC_TOP_SYNTH = Path(__file__).resolve().parent / "fc_top_synth.v"
DPE_BLACKBOX = Path(__file__).resolve().parent / "dpe_blackbox.v"
NL_ARCH = REPO / "nl_dpe" / "nl_dpe_22nm_auto.xml"

VTR_ROOT = Path(
    os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing")
)
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
VTR_PYTHON = VTR_ROOT / ".venv" / "bin" / "python"
if not VTR_PYTHON.is_file():
    VTR_PYTHON = None

DEFAULT_ROUTE_CHAN_WIDTH = 300
DEFAULT_TIMEOUT_S = 1800  # 30 minutes per workload (per task spec)

# ── NL-DPE arch geometry (matches run_fc_smoke.py's ARCH_PARAMS["nldpe"]) ──
NL_GEOM = dict(
    R=256, C=256, BUF=40,
    PRECISION=8, PIPELINE_DEPTH=2, ACAM_CYCLES=1, HAS_ACAM=1,
)

# ── Target workloads ────────────────────────────────────────────────────
@dataclass
class Workload:
    priority: str
    label: str
    M: int
    K: int
    N: int
    activation: str  # "none" | "relu"
    expected_dpe: int
    expected_bram_lb: int  # lower bound (informational)
    expected_dsp: int
    notes: str

WORKLOADS = [
    Workload("P1", "bert_qkv_proj_NL", 1, 128, 128, "relu",
             expected_dpe=1, expected_bram_lb=2, expected_dsp=0,
             notes="V=1, H=1; 1 DPE; in/out BRAMs"),
    Workload("P2", "lenet_fc1_NL", 1, 400, 120, "relu",
             expected_dpe=2, expected_bram_lb=3, expected_dsp=0,
             notes="V=2, H=1; 2 DPEs; CLB tree fold"),
    Workload("P3", "bert_ffn1_NL", 1, 128, 512, "relu",
             expected_dpe=2, expected_bram_lb=3, expected_dsp=0,
             notes="V=1, H=2; 2 DPEs; H output banks"),
]


# ── Metric parsing (matches nl_dpe/run_vtr.py patterns) ────────────────
FMAX_FINAL_RE = re.compile(
    r"Final critical path delay .*?Fmax:\s*([\d.]+)\s*MHz"
)
FMAX_ANY_RE = re.compile(r"Fmax:\s*([\d.]+)\s*MHz")
WIRE_RE = re.compile(r"Total wirelength:\s*([0-9]+(?:\.[0-9]+)?)")
CPD_RE = re.compile(
    r"([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)\s+CPD:\s*([\d.]+)\s*ns\s*\(([\d.]+)\s*MHz\)"
)
RESOURCE_HEADER_RE = re.compile(r"^Resource usage", re.IGNORECASE)
CONTEXT_RE = re.compile(r"^\s*(Netlist|Architecture)\s*$")
COUNT_RE = re.compile(r"^\s*(\d+)\s+blocks of type:\s+(\S+)")


def find_vpr_log(run_dir: Path) -> Path:
    """Locate the VPR log (vpr_stdout.log or vpr.out)."""
    for name in ("vpr_stdout.log", "vpr.out"):
        p = run_dir / name
        if p.is_file():
            return p
        p = run_dir / "temp" / name
        if p.is_file():
            return p
    return run_dir / "vpr_stdout.log"


def parse_metrics(log: Path) -> tuple[float, float]:
    """Extract (wirelength, fmax_mhz) from a VPR log."""
    content = log.read_text(errors="replace")
    wire_matches = WIRE_RE.findall(content)
    wirelength = float(wire_matches[-1]) if wire_matches else 0.0

    cpd_matches = CPD_RE.findall(content)
    if cpd_matches:
        clk = next((m for m in cpd_matches if m[0] == m[1] == "clk"), None)
        chosen = clk if clk else cpd_matches[0]
        fmax = float(chosen[3])
    else:
        fmax_matches = FMAX_FINAL_RE.findall(content) or FMAX_ANY_RE.findall(content)
        fmax = float(fmax_matches[-1]) if fmax_matches else 0.0

    return wirelength, fmax


def parse_resources(log: Path) -> dict[str, int]:
    """Extract netlist block counts from the VPR log."""
    if not log.is_file():
        return {}
    lines = log.read_text(errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if RESOURCE_HEADER_RE.search(line):
            start = i
            break
    if start is None:
        return {}
    res: dict[str, int] = {}
    ctx = None
    for line in lines[start + 1:]:
        if line and not line[0].isspace():
            break
        m = CONTEXT_RE.match(line)
        if m:
            ctx = m.group(1).lower()
            continue
        m = COUNT_RE.match(line)
        if m and ctx == "netlist":
            res[m.group(2)] = int(m.group(1))
    return res


# ── Per-workload top wrapper generation ──────────────────────────────────
def gen_top_wrapper(wl: Workload) -> str:
    """Emit the workload-specific top-level instantiation of fc_top_synth.

    Top module name = `<label>_top`. Holds reset asserted on power-up,
    derives all internal logic from clk only. Inputs that VTR-synth needs
    to be externally driven (in_bram_wen, etc.) are exposed at the top
    level; outputs (done, out_bram_rdata) are exposed similarly.
    """
    g = NL_GEOM
    act = 1 if wl.activation == "relu" else 0
    top_name = f"{wl.label}_top"
    return f"""// Auto-generated by rtl_flow/vtr/run_vtr_smoke.py.
// Top wrapper for workload {wl.label} (M={wl.M}, K={wl.K}, N={wl.N},
// activation={wl.activation}).

`timescale 1ns / 1ps

module {top_name} (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        done,
    input  wire        in_bram_wen,
    input  wire [31:0] in_bram_waddr,
    input  wire [{g["BUF"]-1}:0] in_bram_wdata,
    input  wire [31:0] out_bram_raddr,
    output wire [{g["BUF"]-1}:0] out_bram_rdata
);
    fc_top_synth #(
        .M({wl.M}),
        .K({wl.K}),
        .N({wl.N}),
        .R({g["R"]}),
        .C({g["C"]}),
        .BUF({g["BUF"]}),
        .PRECISION({g["PRECISION"]}),
        .PIPELINE_DEPTH({g["PIPELINE_DEPTH"]}),
        .ACAM_CYCLES({g["ACAM_CYCLES"]}),
        .ACTIVATION_MODE({act}),
        .HAS_ACAM({g["HAS_ACAM"]})
    ) u_fc (
        .clk(clk),
        .reset(reset),
        .start(start),
        .done(done),
        .in_bram_wen(in_bram_wen),
        .in_bram_waddr(in_bram_waddr),
        .in_bram_wdata(in_bram_wdata),
        .out_bram_raddr(out_bram_raddr),
        .out_bram_rdata(out_bram_rdata)
    );
endmodule
"""


def build_circuit_file(wl: Workload, scratch: Path) -> Path:
    """Concatenate dpe_blackbox + fc_top_synth + per-workload top into one .v.

    The top module is placed LAST so Parmys treats it as the design top.
    """
    parts = [
        DPE_BLACKBOX.read_text(),
        FC_TOP_SYNTH.read_text(),
        gen_top_wrapper(wl),
    ]
    out = scratch / f"{wl.label}_circuit.v"
    out.write_text("\n".join(parts))
    return out


# ── VTR command construction ────────────────────────────────────────────
def build_vtr_cmd(circuit: Path, run_dir: Path,
                  route_chan_width: int) -> list[str]:
    cmd = [
        str(VTR_FLOW),
        str(circuit),
        str(NL_ARCH),
        "-temp_dir", str(run_dir),
        "--route_chan_width", str(route_chan_width),
    ]
    if VTR_PYTHON is not None:
        return [str(VTR_PYTHON), *cmd]
    return [sys.executable, *cmd]


def run_vtr(wl: Workload, scratch: Path, route_chan_width: int,
            timeout_s: int) -> dict:
    """Run VTR for one workload. Returns metrics dict + status.

    Note: VTR's run_vtr_flow.py copies the circuit_file into temp_dir,
    so circuit_file must NOT live inside temp_dir (avoids SameFileError).
    We put the circuit in scratch.parent/scratch.name + '_input', and
    point temp_dir at scratch.
    """
    run_dir = scratch
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dir = scratch.parent / (scratch.name + "_input")
    input_dir.mkdir(parents=True, exist_ok=True)
    circuit = build_circuit_file(wl, input_dir)

    cmd = build_vtr_cmd(circuit, run_dir, route_chan_width)
    print(f"[{wl.label}] running VTR (timeout {timeout_s}s)...")
    print(f"    cwd={run_dir}")
    print(f"    cmd={' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return dict(
            label=wl.label, status="TIMEOUT", elapsed_s=elapsed,
            error=f"VTR exceeded {timeout_s}s",
        )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        return dict(
            label=wl.label, status="FAILED", elapsed_s=elapsed,
            returncode=proc.returncode,
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
        )

    log = find_vpr_log(run_dir)
    if not log.is_file():
        return dict(
            label=wl.label, status="NO_LOG", elapsed_s=elapsed,
            error=f"No VPR log at {log}",
        )

    wirelength, fmax = parse_metrics(log)
    resources = parse_resources(log)
    return dict(
        label=wl.label, status="OK", elapsed_s=elapsed,
        wirelength=wirelength, fmax_mhz=fmax,
        resources=resources, log_path=str(log),
    )


# ── Report ────────────────────────────────────────────────────────────────
def fmt_table(results: list[dict]) -> str:
    """Markdown-style results table.

    Note on naming: the arch XML names the DPE tile `wc` (weight crossbar).
    So the column "DPE" reads from netlist['wc']. The arch also defines
    a `dsp_top` tile (DSP slice) and a `memory` tile (BRAM); these are
    inferred from RTL multipliers and `reg [BUF-1:0] storage[0:N-1]`
    arrays respectively.
    """
    header = (
        "| Workload | Status | DPE (wc) | BRAM (mem) | DSP | CLB | "
        "Fmax (MHz) | Wire | Wall |"
    )
    sep = (
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = [header, sep]
    for r in results:
        if r["status"] == "OK":
            res = r["resources"]
            rows.append(
                f"| {r['label']} | OK | {res.get('wc', 0)} | "
                f"{res.get('memory', 0)} | {res.get('dsp_top', 0)} | "
                f"{res.get('clb', 0)} | "
                f"{r['fmax_mhz']:.2f} | {r['wirelength']:.0f} | "
                f"{r['elapsed_s']:.1f}s |"
            )
        else:
            rows.append(
                f"| {r['label']} | {r['status']} | - | - | - | - | - | - | "
                f"{r.get('elapsed_s', 0.0):.1f}s |"
            )
    return "\n".join(rows)


def write_summary(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("VTR smoke test for fc_top_synth.v (Task #89)\n")
        f.write("=" * 70 + "\n\n")
        f.write(fmt_table(results) + "\n\n")
        for r in results:
            f.write(f"\n[{r['label']}]\n")
            for k, v in r.items():
                if k == "resources":
                    f.write(f"  resources:\n")
                    for rk, rv in v.items():
                        f.write(f"    {rk}: {rv}\n")
                elif k in ("stderr_tail", "stdout_tail"):
                    f.write(f"  {k}:\n")
                    for line in str(v).splitlines():
                        f.write(f"    {line}\n")
                else:
                    f.write(f"  {k}: {v}\n")


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--workload",
        action="append",
        default=None,
        help="filter to specific workload label (repeatable).",
    )
    p.add_argument(
        "--priority",
        choices=["P1", "P2", "P3"],
        default=None,
        help="Run only this priority tier (default: P1).",
    )
    p.add_argument(
        "--all-priority",
        action="store_true",
        help="Run P1+P2+P3 (default: P1 only).",
    )
    p.add_argument(
        "--route-chan-width",
        type=int,
        default=DEFAULT_ROUTE_CHAN_WIDTH,
        help=f"VTR route_chan_width (default {DEFAULT_ROUTE_CHAN_WIDTH}).",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-workload timeout in seconds (default {DEFAULT_TIMEOUT_S}).",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep VTR scratch dirs after each run (default: keep always).",
    )
    return p.parse_args()


def select_workloads(args: argparse.Namespace) -> list[Workload]:
    wls = list(WORKLOADS)
    if args.workload:
        wls = [w for w in wls if w.label in args.workload]
    elif args.priority:
        wls = [w for w in wls if w.priority == args.priority]
    elif not args.all_priority:
        wls = [w for w in wls if w.priority == "P1"]
    return wls


def main() -> int:
    args = parse_args()
    if not VTR_FLOW.is_file():
        print(f"ERROR: VTR flow script not found at {VTR_FLOW}", file=sys.stderr)
        return 2

    for required in (FC_TOP_SYNTH, DPE_BLACKBOX, NL_ARCH):
        if not required.is_file():
            print(f"ERROR: missing required file {required}", file=sys.stderr)
            return 2

    workloads = select_workloads(args)
    if not workloads:
        print("ERROR: no workloads selected", file=sys.stderr)
        return 2

    print(f"VTR root:        {VTR_ROOT}")
    print(f"Arch XML:        {NL_ARCH}")
    print(f"Workloads:       {', '.join(w.label for w in workloads)}")
    print(f"Route chan width: {args.route_chan_width}")
    print(f"Per-WL timeout:  {args.timeout_s}s")
    print()

    results: list[dict] = []
    for wl in workloads:
        scratch = RESULTS / f"vtr_smoke_{wl.label}"
        if scratch.exists():
            shutil.rmtree(scratch)
        input_dir = scratch.parent / (scratch.name + "_input")
        if input_dir.exists():
            shutil.rmtree(input_dir)
        r = run_vtr(wl, scratch, args.route_chan_width, args.timeout_s)
        results.append(r)
        if r["status"] == "OK":
            print(
                f"[{wl.label}] OK  fmax={r['fmax_mhz']:.2f} MHz  "
                f"wire={r['wirelength']:.0f}  "
                f"elapsed={r['elapsed_s']:.1f}s  "
                f"DPE(wc)={r['resources'].get('wc', 0)}  "
                f"BRAM(mem)={r['resources'].get('memory', 0)}  "
                f"DSP={r['resources'].get('dsp_top', 0)}  "
                f"CLB={r['resources'].get('clb', 0)}"
            )
        else:
            print(
                f"[{wl.label}] {r['status']} "
                f"elapsed={r.get('elapsed_s', 0.0):.1f}s"
            )
            if "stderr_tail" in r:
                print("  STDERR TAIL:")
                for line in r["stderr_tail"].splitlines():
                    print(f"    {line}")
            if "stdout_tail" in r:
                print("  STDOUT TAIL:")
                for line in r["stdout_tail"].splitlines():
                    print(f"    {line}")

    print()
    print("─" * 70)
    print(fmt_table(results))
    print("─" * 70)

    summary = RESULTS / "vtr_smoke.log"
    write_summary(results, summary)
    print(f"\nDetailed log written to {summary}")

    n_ok = sum(1 for r in results if r["status"] == "OK")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
