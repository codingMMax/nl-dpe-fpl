#!/usr/bin/env python3
"""Run VTR flow for one or more Verilog designs with optional multi-seed runs.

Single run (backward compatible):
    python run_vtr.py lenet_1_channel.v --arch nl_dpe_22nm_auto.xml

Multi-seed runs:
    python run_vtr.py --arch nl_dpe_22nm_auto.xml --design lenet_1_channel.v resnet_1_channel.v --runs 5
"""

import argparse
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VTR_ROOT = Path(
    os.environ.get("VTR_ROOT", "/mnt/vault0/jiajunh5/vtr-verilog-to-routing")
)
DEFAULT_ROUTE_CHAN_WIDTH = 300
DEFAULT_RUNS = 1

# ── metric parsing regexes ────────────────────────────────────────────
FMAX_FINAL_RE = re.compile(
    r"Final critical path delay .*?Fmax:\s*([\d.]+)\s*MHz"
)
FMAX_ANY_RE = re.compile(r"Fmax:\s*([\d.]+)\s*MHz")
WIRE_RE = re.compile(r"Total wirelength:\s*([0-9]+(?:\.[0-9]+)?)")
CPD_RE = re.compile(
    r"([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)\s+CPD:\s*([\d.]+)\s*ns\s*\(([\d.]+)\s*MHz\)"
)

# ── resource parsing ──────────────────────────────────────────────────
RESOURCE_HEADER_RE = re.compile(r"^Resource usage", re.IGNORECASE)
CONTEXT_RE = re.compile(r"^\s*(Netlist|Architecture)\s*$")
COUNT_RE = re.compile(r"^\s*(\d+)\s+blocks of type:\s+(\S+)")


# ── data classes ──────────────────────────────────────────────────────
@dataclass
class RunResult:
    seed: int
    wirelength: float
    fmax_mhz: float
    elapsed_s: float
    run_dir: Path
    resources: Dict[str, int] = field(default_factory=dict)


# ── metric extraction ─────────────────────────────────────────────────
def find_vpr_log(run_dir: Path) -> Path:
    """Find the VPR log in a run directory."""
    for name in ("vpr_stdout.log", "vpr.out"):
        p = run_dir / name
        if p.is_file():
            return p
    # VTR sometimes nests under temp/
    for name in ("vpr_stdout.log", "vpr.out"):
        p = run_dir / "temp" / name
        if p.is_file():
            return p
    return run_dir / "vpr_stdout.log"


def parse_metrics(log_path: Path) -> Tuple[float, float]:
    """Extract (wirelength, fmax_mhz) from a VPR log."""
    if not log_path.exists():
        raise FileNotFoundError(f"VPR log not found at {log_path}")
    content = log_path.read_text(errors="replace")

    # wirelength
    wire_matches = WIRE_RE.findall(content)
    wirelength = float(wire_matches[-1]) if wire_matches else 0.0

    # fmax – prefer intra-domain CPD, fall back to legacy Fmax lines
    cpd_matches = CPD_RE.findall(content)
    if cpd_matches:
        clk_match = next(
            (m for m in cpd_matches if m[0] == m[1] == "clk"), None
        )
        chosen = clk_match if clk_match else cpd_matches[0]
        fmax_mhz = float(chosen[3])
    else:
        fmax_final = FMAX_FINAL_RE.findall(content)
        fmax_all = fmax_final if fmax_final else FMAX_ANY_RE.findall(content)
        if not fmax_all:
            raise ValueError(f"Could not find Fmax in {log_path}")
        fmax_mhz = float(fmax_all[-1])

    return wirelength, fmax_mhz


def parse_resources(log_path: Path) -> Dict[str, int]:
    """Extract netlist block counts from a VPR log."""
    if not log_path.exists():
        return {}
    lines = log_path.read_text(errors="replace").splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if RESOURCE_HEADER_RE.search(line):
            start_idx = i
            break
    if start_idx is None:
        return {}
    resources: Dict[str, int] = {}
    ctx = None
    for line in lines[start_idx + 1 :]:
        if line and not line[0].isspace():
            break
        m = CONTEXT_RE.match(line)
        if m:
            ctx = m.group(1).lower()
            continue
        m = COUNT_RE.match(line)
        if m and ctx == "netlist":
            resources[m.group(2)] = int(m.group(1))
    return resources


# ── VTR command construction ──────────────────────────────────────────
def build_vtr_command(
    vtr_flow: Path,
    vtr_python: Optional[Path],
    design: Path,
    arch: Path,
    route_chan_width: int,
    sdc_file: Optional[Path],
    temp_dir: Path,
    seed: Optional[int] = None,
) -> List[str]:
    cmd = [
        str(vtr_flow),
        str(design),
        str(arch),
        "--route_chan_width",
        str(route_chan_width),
        "-temp_dir",
        str(temp_dir),
    ]
    if sdc_file and sdc_file.is_file():
        cmd.extend(["--sdc_file", str(sdc_file)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if vtr_python and vtr_python.is_file():
        return [str(vtr_python), *cmd]
    return [sys.executable, *cmd]


# ── single run ────────────────────────────────────────────────────────
def run_single(
    vtr_flow: Path,
    vtr_python: Optional[Path],
    design: Path,
    arch: Path,
    route_chan_width: int,
    sdc_file: Optional[Path],
    run_dir: Path,
    seed: int,
    run_index: int,
    total_runs: int,
    design_name: str,
) -> RunResult:
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_vtr_command(
        vtr_flow, vtr_python, design, arch,
        route_chan_width, sdc_file, run_dir, seed,
    )
    print(f"  [{design_name}] run {run_index + 1}/{total_runs}  seed={seed}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        # Dump last 10 lines of stderr for debugging
        tail = "\n".join(result.stderr.strip().splitlines()[-10:])
        raise RuntimeError(
            f"VTR failed for {design_name} seed={seed} "
            f"(exit {result.returncode}):\n{tail}"
        )

    log_path = find_vpr_log(run_dir)
    wirelength, fmax_mhz = parse_metrics(log_path)
    resources = parse_resources(log_path)

    print(
        f"    wirelength={wirelength:.0f}  fmax={fmax_mhz:.3f} MHz  "
        f"elapsed={elapsed:.1f}s  wc={resources.get('wc', '?')}"
    )
    return RunResult(
        seed=seed,
        wirelength=wirelength,
        fmax_mhz=fmax_mhz,
        elapsed_s=elapsed,
        run_dir=run_dir,
        resources=resources,
    )


# ── run log writer ────────────────────────────────────────────────────
def write_run_log(
    design_dir: Path,
    design_name: str,
    arch_name: str,
    results: List[RunResult],
    failures: List[Tuple[int, str]],
) -> None:
    log_path = design_dir / "run.log"
    with open(log_path, "w") as f:
        f.write(f"Design: {design_name}\n")
        f.write(f"Architecture: {arch_name}\n")
        f.write(f"Total runs: {len(results) + len(failures)}\n")
        f.write(f"Successful: {len(results)}\n")
        f.write(f"Failed: {len(failures)}\n")
        f.write("\n")

        if results:
            # Header
            f.write(
                f"{'Run':<6} {'Seed':<10} {'Wirelength':>12} "
                f"{'Fmax (MHz)':>12} {'Elapsed':>10}"
            )
            # Resource columns from first result
            res_keys = sorted(results[0].resources.keys())
            for k in res_keys:
                f.write(f"  {k:>8}")
            f.write("\n")
            f.write("-" * (52 + 10 * len(res_keys)) + "\n")

            for i, r in enumerate(results):
                f.write(
                    f"{i + 1:<6} {r.seed:<10} {r.wirelength:>12.0f} "
                    f"{r.fmax_mhz:>12.3f} {r.elapsed_s:>9.1f}s"
                )
                for k in res_keys:
                    f.write(f"  {r.resources.get(k, 0):>8}")
                f.write(f"  {r.run_dir}\n")

            # Averages
            wirelengths = [r.wirelength for r in results]
            fmaxes = [r.fmax_mhz for r in results]

            # Trim outliers if > 5 runs (matching HEART2026 convention)
            if len(results) > 5:
                wirelengths = sorted(wirelengths)[1:-2]
                fmaxes = sorted(fmaxes)[1:-2]

            avg_wire = sum(wirelengths) / len(wirelengths)
            avg_fmax = sum(fmaxes) / len(fmaxes)
            min_fmax = min(r.fmax_mhz for r in results)
            max_fmax = max(r.fmax_mhz for r in results)

            f.write(f"\nAverages over {len(wirelengths)} runs")
            if len(wirelengths) < len(results):
                f.write(f" (trimmed from {len(results)})")
            f.write(":\n")
            f.write(f"  wirelength = {avg_wire:.0f}\n")
            f.write(f"  fmax       = {avg_fmax:.3f} MHz\n")
            f.write(f"  fmax range = [{min_fmax:.3f}, {max_fmax:.3f}] MHz\n")

        for seed, err in failures:
            f.write(f"\nFAILED seed={seed}: {err}\n")

    print(f"  [{design_name}] run.log written to {log_path}")


# ── arg parsing ───────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VTR flow for Verilog designs with optional multi-seed runs."
    )
    # Support both positional (backward compat) and --design
    parser.add_argument(
        "positional_designs",
        nargs="*",
        default=[],
        help="Verilog design files (positional, backward compatible).",
    )
    parser.add_argument(
        "--design",
        nargs="+",
        default=[],
        help="Verilog design files.",
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Architecture XML file.",
    )
    parser.add_argument(
        "--route_chan_width",
        type=int,
        default=DEFAULT_ROUTE_CHAN_WIDTH,
        help=f"Route channel width (default: {DEFAULT_ROUTE_CHAN_WIDTH}).",
    )
    parser.add_argument(
        "--sdc_file",
        default="clock.sdc",
        help="SDC constraints file (default: clock.sdc).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of randomized VTR runs per design (default: 1).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Max parallel jobs (default: auto).",
    )
    parser.add_argument(
        "--vtr_root",
        default=os.environ.get("VTR_ROOT", str(DEFAULT_VTR_ROOT)),
        help="VTR root path.",
    )
    parser.add_argument(
        "--seed-min",
        type=int,
        default=1,
        help="Min VPR seed value (default: 1).",
    )
    parser.add_argument(
        "--seed-max",
        type=int,
        default=2**16 - 1,
        help="Max VPR seed value (default: 65535).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Python RNG seed for reproducible seed generation.",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()

    # Merge positional and --design args
    all_designs = args.positional_designs + args.design
    if not all_designs:
        print("Error: no design files specified", file=sys.stderr)
        return 1

    design_paths = [Path(d).resolve() for d in all_designs]
    arch_path = Path(args.arch).resolve()
    sdc_path = Path(args.sdc_file).resolve() if args.sdc_file else None

    # Validate inputs
    missing = [str(p) for p in design_paths if not p.is_file()]
    if missing:
        print("Error: design file(s) not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 1
    if not arch_path.is_file():
        print(f"Error: arch file not found: {arch_path}", file=sys.stderr)
        return 1

    # Resolve VTR paths
    vtr_root = Path(args.vtr_root).resolve()
    vtr_flow = vtr_root / "vtr_flow" / "scripts" / "run_vtr_flow.py"
    if not vtr_flow.is_file():
        print(f"Error: VTR script not found at {vtr_flow}", file=sys.stderr)
        return 1
    vtr_python = vtr_root / ".venv" / "bin" / "python"
    if not vtr_python.is_file():
        vtr_python = None

    runs = max(1, args.runs)
    rng = random.Random(args.rng_seed)

    print(f"VTR root:  {vtr_root}")
    print(f"Arch:      {arch_path.name}")
    print(f"Designs:   {', '.join(p.stem for p in design_paths)}")
    print(f"Runs/design: {runs}")
    print()

    # ── single-run mode (backward compatible) ─────────────────────
    if runs == 1:
        # No run_* subdirectories, output directly in <design_stem>/
        jobs = []
        for dp in design_paths:
            temp_dir = dp.with_suffix("")
            seed = rng.randint(args.seed_min, args.seed_max)
            jobs.append((dp, temp_dir, seed))

        if len(jobs) == 1:
            dp, temp_dir, seed = jobs[0]
            cmd = build_vtr_command(
                vtr_flow, vtr_python, dp, arch_path,
                args.route_chan_width, sdc_path, temp_dir, seed,
            )
            print("Running:", " ".join(cmd))
            return subprocess.run(cmd).returncode

        cpu_count = os.cpu_count() or 1
        max_workers = (
            args.jobs if args.jobs > 0 else min(len(jobs), cpu_count)
        )
        failures = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futs = {}
            for dp, temp_dir, seed in jobs:
                cmd = build_vtr_command(
                    vtr_flow, vtr_python, dp, arch_path,
                    args.route_chan_width, sdc_path, temp_dir, seed,
                )
                futs[executor.submit(subprocess.run, cmd)] = dp.stem
            for fut in as_completed(futs):
                name = futs[fut]
                rc = fut.result().returncode
                if rc != 0:
                    print(f"  [{name}] FAILED (exit {rc})")
                    failures += 1
                else:
                    print(f"  [{name}] OK")
        return 1 if failures else 0

    # ── multi-run mode ────────────────────────────────────────────
    # Build all (design, run_index, seed) jobs
    @dataclass
    class Job:
        design: Path
        design_name: str
        run_index: int
        seed: int
        run_dir: Path

    all_jobs: List[Job] = []
    for dp in design_paths:
        design_name = dp.stem
        design_dir = dp.parent / design_name
        design_dir.mkdir(parents=True, exist_ok=True)
        for i in range(runs):
            seed = rng.randint(args.seed_min, args.seed_max)
            run_dir = design_dir / f"run_{i + 1}_{seed}"
            all_jobs.append(Job(dp, design_name, i, seed, run_dir))

    total_jobs = len(all_jobs)
    cpu_count = os.cpu_count() or 1
    max_workers = args.jobs if args.jobs > 0 else min(total_jobs, cpu_count)

    print(
        f"Launching {total_jobs} VTR jobs "
        f"({len(design_paths)} designs × {runs} runs) "
        f"with {max_workers} workers\n"
    )

    # Results grouped by design
    results_by_design: Dict[str, List[RunResult]] = {
        dp.stem: [] for dp in design_paths
    }
    failures_by_design: Dict[str, List[Tuple[int, str]]] = {
        dp.stem: [] for dp in design_paths
    }

    def _run_job(job: Job) -> Tuple[str, Optional[RunResult], Optional[str]]:
        try:
            r = run_single(
                vtr_flow, vtr_python, job.design, arch_path,
                args.route_chan_width, sdc_path, job.run_dir,
                job.seed, job.run_index, runs, job.design_name,
            )
            return job.design_name, r, None
        except Exception as exc:
            print(
                f"  [{job.design_name}] run {job.run_index + 1}/{runs} "
                f"seed={job.seed} FAILED: {exc}",
                file=sys.stderr,
            )
            return job.design_name, None, str(exc)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = {executor.submit(_run_job, job): job for job in all_jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            design_name, result, error = fut.result()
            if result is not None:
                results_by_design[design_name].append(result)
            if error is not None:
                failures_by_design[design_name].append((job.seed, error))

    # Sort results by run index (they may complete out of order)
    for dp in design_paths:
        name = dp.stem
        results_by_design[name].sort(
            key=lambda r: int(r.run_dir.name.split("_")[1])
        )

    # Write run.log per design and print summary
    print(f"\n{'=' * 70}")
    exit_code = 0
    for dp in design_paths:
        name = dp.stem
        design_dir = dp.parent / name
        results = results_by_design[name]
        failures = failures_by_design[name]

        write_run_log(design_dir, name, arch_path.name, results, failures)

        if results:
            fmaxes = [r.fmax_mhz for r in results]
            wires = [r.wirelength for r in results]
            avg_fmax = sum(fmaxes) / len(fmaxes)
            avg_wire = sum(wires) / len(wires)
            print(
                f"  [{name}] {len(results)}/{runs} OK  "
                f"avg_fmax={avg_fmax:.1f} MHz  "
                f"avg_wire={avg_wire:.0f}  "
                f"range=[{min(fmaxes):.1f}, {max(fmaxes):.1f}]"
            )
        if failures:
            print(f"  [{name}] {len(failures)} FAILED")
            exit_code = 1

    print(f"{'=' * 70}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
