#!/usr/bin/env python3
"""Compare VTR resource usage between NL-DPE and Azure-Lily designs."""

import argparse
from pathlib import Path
import re
import sys

RESOURCE_HEADER_RE = re.compile(r"^Resource usage", re.IGNORECASE)
CONTEXT_RE = re.compile(r"^\s*(Netlist|Architecture)\s*$")
COUNT_RE = re.compile(r"^\s*(\d+)\s+blocks of type:\s+(\S+)")

# Also extract critical path and Fmax
CPDELAY_RE = re.compile(r"Final critical path delay.*?:\s+([\d.]+)\s+ns")
FMAX_RE = re.compile(r"Final setup Fmax.*?:\s+([\d.]+)\s+MHz", re.IGNORECASE)


def parse_vpr_log(text: str):
    """Parse resource usage and timing from VPR log."""
    lines = text.splitlines()

    # Resource usage
    start_idx = None
    for i, line in enumerate(lines):
        if RESOURCE_HEADER_RE.search(line):
            start_idx = i
            break

    resources = {}
    order = []
    if start_idx is not None:
        ctx = None
        for line in lines[start_idx + 1:]:
            if line and not line[0].isspace():
                break
            m = CONTEXT_RE.match(line)
            if m:
                ctx = m.group(1).lower()
                continue
            m = COUNT_RE.match(line)
            if m and ctx:
                count = int(m.group(1))
                block_type = m.group(2)
                if block_type not in resources:
                    resources[block_type] = {"netlist": 0, "architecture": 0}
                    order.append(block_type)
                resources[block_type][ctx] = count

    # Timing
    cp_delay = None
    fmax = None
    for line in lines:
        m = CPDELAY_RE.search(line)
        if m:
            cp_delay = float(m.group(1))
        m = FMAX_RE.search(line)
        if m:
            fmax = float(m.group(1))

    return order, resources, cp_delay, fmax


def find_log(design_dir: Path) -> Path:
    """Find the VPR stdout log in a VTR output directory."""
    candidates = [
        design_dir / "vpr_stdout.log",
        design_dir / "vpr.out",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Try to find any vpr log
    logs = list(design_dir.glob("**/vpr_stdout.log"))
    if logs:
        return logs[0]
    return design_dir / "vpr_stdout.log"


def main():
    parser = argparse.ArgumentParser(
        description="Compare VTR resource usage: NL-DPE vs Azure-Lily"
    )
    parser.add_argument(
        "--nl-dpe-dir",
        default="/mnt/vault0/jiajunh5/nl-dpe-fpl/nl_dpe_rtl",
        help="NL-DPE RTL directory",
    )
    parser.add_argument(
        "--azurelily-dir",
        default="/mnt/vault0/jiajunh5/nl-dpe-fpl/azurelily_TACO_experiments",
        help="Azure-Lily RTL directory",
    )
    parser.add_argument(
        "--designs",
        nargs="+",
        default=["lenet_1_channel", "resnet_1_channel"],
        help="Design names (without .v extension)",
    )
    args = parser.parse_args()

    nl_dir = Path(args.nl_dpe_dir)
    al_dir = Path(args.azurelily_dir)

    # Expected DPE counts for sanity checking
    expected_dpes = {
        "lenet_1_channel": {"nl_dpe": 6, "azurelily": 5},
        "resnet_1_channel": {"nl_dpe": 40, "azurelily": 35},
        "resnet_1_channel_small": {"nl_dpe": 40, "azurelily": 35},
    }

    for design in args.designs:
        print(f"\n{'='*70}")
        print(f"  Design: {design}")
        print(f"{'='*70}")

        results = {}
        for label, base_dir in [("NL-DPE", nl_dir), ("Azure-Lily", al_dir)]:
            design_dir = base_dir / design
            log_path = find_log(design_dir)

            if not log_path.is_file():
                print(f"\n  [{label}] Log not found: {log_path}")
                continue

            content = log_path.read_text(errors="replace")
            order, resources, cp_delay, fmax = parse_vpr_log(content)
            results[label] = (order, resources, cp_delay, fmax)

        if len(results) < 2:
            print("  Skipping comparison (missing logs)")
            continue

        # Merge block types from both
        nl_order, nl_res, nl_cp, nl_fmax = results["NL-DPE"]
        al_order, al_res, al_cp, al_fmax = results["Azure-Lily"]

        all_types = []
        seen = set()
        for bt in nl_order + al_order:
            if bt not in seen:
                all_types.append(bt)
                seen.add(bt)

        # Resource comparison table
        print(f"\n  {'Block Type':<15} {'NL-DPE':>10} {'AzureLily':>10} {'Diff':>10}")
        print(f"  {'-'*45}")
        for bt in all_types:
            nl_count = nl_res.get(bt, {}).get("netlist", 0)
            al_count = al_res.get(bt, {}).get("netlist", 0)
            diff = nl_count - al_count
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"  {bt:<15} {nl_count:>10} {al_count:>10} {diff_str:>10}")

        # Timing comparison
        print(f"\n  {'Timing':<15} {'NL-DPE':>10} {'AzureLily':>10}")
        print(f"  {'-'*35}")
        if nl_cp is not None and al_cp is not None:
            print(f"  {'CP Delay (ns)':<15} {nl_cp:>10.3f} {al_cp:>10.3f}")
        if nl_fmax is not None and al_fmax is not None:
            print(f"  {'Fmax (MHz)':<15} {nl_fmax:>10.1f} {al_fmax:>10.1f}")

        # DPE sanity check
        if design in expected_dpes:
            exp = expected_dpes[design]
            nl_dpe_count = nl_res.get("wc", {}).get("netlist", 0)
            al_dpe_count = al_res.get("wc", {}).get("netlist", 0)
            print(f"\n  DPE Sanity Check:")
            for label, actual, expected in [
                ("NL-DPE", nl_dpe_count, exp["nl_dpe"]),
                ("Azure-Lily", al_dpe_count, exp["azurelily"]),
            ]:
                status = "PASS" if actual == expected else "FAIL"
                print(f"    {label}: {actual} DPEs (expected {expected}) [{status}]")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
