#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional


DEFAULT_VTR_ROOT = "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"
DEFAULT_ROUTE_CHAN_WIDTH = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke VTR flow for one or more Verilog designs against an architecture XML."
    )
    parser.add_argument(
        "designs",
        nargs="+",
        help="One or more Verilog design files (.v).",
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
        help="Optional SDC file to pass to VTR.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel jobs (default: min(len(designs), cpu_count)).",
    )
    parser.add_argument(
        "--vtr_root",
        default=os.environ.get("VTR_ROOT", DEFAULT_VTR_ROOT),
        help="VTR root path (default: $VTR_ROOT or built-in path).",
    )
    return parser.parse_args()


def build_command(
    vtr_run: Path,
    vtr_python: Optional[Path],
    design_path: Path,
    arch_path: Path,
    route_chan_width: int,
    sdc_file: Optional[str],
) -> List[str]:
    cmd = [
        str(vtr_run),
        str(design_path),
        str(arch_path),
        "--route_chan_width",
        str(route_chan_width),
        "-temp_dir",
        str(design_path.with_suffix("")),
    ]
    if sdc_file:
        cmd.extend(["--sdc_file", str(Path(sdc_file).resolve())])

    if vtr_python and vtr_python.is_file():
        return [str(vtr_python), *cmd]
    return [sys.executable, *cmd]


def run_one(cmd: List[str]) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    args = parse_args()

    design_paths = [Path(d).resolve() for d in args.designs]
    arch_path = Path(args.arch).resolve()
    missing = [str(p) for p in design_paths if not p.is_file()]
    if missing:
        print("Error: design file(s) not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 1
    if not arch_path.is_file():
        print(f"Error: arch file not found: {arch_path}", file=sys.stderr)
        return 1

    vtr_root = Path(args.vtr_root).resolve()
    vtr_run = vtr_root / "vtr_flow" / "scripts" / "run_vtr_flow.py"
    if not vtr_run.is_file():
        print(f"Error: VTR script not found at {vtr_run}", file=sys.stderr)
        return 1

    vtr_python = vtr_root / ".venv" / "bin" / "python"
    commands: List[List[str]] = [
        build_command(
            vtr_run,
            vtr_python if vtr_python.is_file() else None,
            design_path,
            arch_path,
            args.route_chan_width,
            args.sdc_file,
        )
        for design_path in design_paths
    ]

    if len(commands) == 1:
        return run_one(commands[0])

    cpu_count = os.cpu_count() or 1
    jobs = args.jobs if args.jobs and args.jobs > 0 else min(len(commands), cpu_count)

    failures = 0

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {executor.submit(run_one, cmd): cmd for cmd in commands}
        for future in as_completed(future_map):
            if future.result() != 0:
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
