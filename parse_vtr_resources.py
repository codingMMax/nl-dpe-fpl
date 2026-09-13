#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys


RESOURCE_HEADER_RE = re.compile(r"^Resource usage", re.IGNORECASE)
CONTEXT_RE = re.compile(r"^\s*(Netlist|Architecture)\s*$")
COUNT_RE = re.compile(r"^\s*(\d+)\s+blocks of type:\s+(\S+)")


def parse_resource_usage(text: str):
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if RESOURCE_HEADER_RE.search(line):
            start_idx = i
            break
    if start_idx is None:
        return None

    order = []
    data = {}
    ctx = None
    for line in lines[start_idx + 1 :]:
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
            if block_type not in data:
                data[block_type] = {"netlist": None, "architecture": None}
                order.append(block_type)
            data[block_type][ctx] = count

    return order, data


def format_resource_usage(order, data):
    # Header
    header = f"{'Block Type':<15} {'Netlist':>10} {'Arch':>10} {'Util %':>10}"
    sep = "-" * len(header)
    out_lines = [header, sep]
    for block_type in order:
        counts = data[block_type]
        net = counts.get("netlist") or 0
        arch = counts.get("architecture") or 0
        util = f"{net / arch * 100:.2f}" if arch > 0 else "N/A"
        out_lines.append(f"{block_type:<15} {net:>10} {arch:>10} {util:>10}")
    return "\n".join(out_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract resource usage from one or more VPR vpr_stdout.log files."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Path(s) to vpr_stdout.log files.",
    )
    args = parser.parse_args()

    exit_code = 0
    for idx, log_path in enumerate(args.logs):
        path = Path(log_path)
        if not path.is_file():
            print(f"Error: log not found: {path}", file=sys.stderr)
            exit_code = 1
            continue

        content = path.read_text(errors="replace")
        parsed = parse_resource_usage(content)
        if parsed is None:
            print(f"{path}: Resource usage section not found.", file=sys.stderr)
            exit_code = 1
            continue

        if len(args.logs) > 1:
            if idx > 0:
                print()
            print(f"== {path} ==")
        print(format_resource_usage(*parsed))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
