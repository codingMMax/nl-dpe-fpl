#!/usr/bin/env python3
"""BERT-Tiny Sequence Length Sweep: RTL generation → VTR runs → result summary.

Generates BERT-Tiny RTL for 3 architectures × 6 sequence lengths,
runs VTR with auto_layout (3 seeds each), and produces a summary CSV.

Usage:
    # Full pipeline: generate RTL + run VTR + parse results
    python benchmarks/run_seqlen_sweep.py

    # Skip RTL generation (already done)
    python benchmarks/run_seqlen_sweep.py --skip-rtl

    # Skip VTR (just parse existing results)
    python benchmarks/run_seqlen_sweep.py --skip-rtl --skip-vtr

    # Sanity check: 1 arch, 1 seq_len, 1 seed
    python benchmarks/run_seqlen_sweep.py --sanity
"""
import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "nl_dpe"))

from gen_bert_tiny_wrapper import gen_bert_tiny
from run_vtr import (
    build_vtr_command, run_single, write_run_log, find_vpr_log,
    parse_metrics, parse_resources, RunResult,
    DEFAULT_VTR_ROOT, DEFAULT_ROUTE_CHAN_WIDTH,
)

# ── Constants ────────────────────────────────────────────────────────────
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]

# (arch_type for generator, rows, cols, label, arch_xml_name)
ARCH_CONFIGS = [
    ("nl_dpe",     1024, 128, "proposed",  "proposed_auto.xml"),
    ("nl_dpe",     1024, 256, "al_like",   "al_like_auto.xml"),
    ("azure_lily",  512, 128, "azurelily", "azure_lily_auto.xml"),
]

ARCH_DIR = PROJECT_DIR / "benchmarks" / "arch"
RTL_DIR = PROJECT_DIR / "benchmarks" / "rtl"
VTR_DIR = PROJECT_DIR / "benchmarks" / "vtr_runs_seqlen"
SDC_FILE = PROJECT_DIR / "benchmarks" / "clock.sdc"
RESULTS_DIR = PROJECT_DIR / "benchmarks" / "results"

CLB_TILE_UM2 = 2239

GRID_SIZE_RE = re.compile(
    r"FPGA sized to (\d+) x (\d+)", re.MULTILINE
)
RESOURCE_NETLIST_RE = re.compile(
    r"(\d+)\s+blocks of type:\s+(\w+)"
)


# ── Step 1: Generate RTL ────────────────────────────────────────────────
def generate_rtl(seq_lens=None, archs=None):
    """Generate BERT-Tiny RTL for all (arch, seq_len) combinations."""
    if seq_lens is None:
        seq_lens = SEQ_LENS
    if archs is None:
        archs = ARCH_CONFIGS

    RTL_DIR.mkdir(parents=True, exist_ok=True)
    for arch_type, rows, cols, label, _ in archs:
        for S in seq_lens:
            full_label = f"{label}_s{S}"
            print(f"Generating {full_label}...")
            gen_bert_tiny(arch_type, rows, cols, str(RTL_DIR), full_label,
                          seq_len=S, dimm_width=1)


# ── Step 2: Run VTR ─────────────────────────────────────────────────────
def run_vtr_sweep(seq_lens=None, archs=None, num_seeds=3, max_jobs=56,
                  rng_seed=42):
    """Run VTR for all (arch, seq_len) combinations with multi-seed."""
    import random
    if seq_lens is None:
        seq_lens = SEQ_LENS
    if archs is None:
        archs = ARCH_CONFIGS

    vtr_root = Path(os.environ.get("VTR_ROOT", str(DEFAULT_VTR_ROOT))).resolve()
    vtr_flow = vtr_root / "vtr_flow" / "scripts" / "run_vtr_flow.py"
    vtr_python = vtr_root / ".venv" / "bin" / "python"
    if not vtr_python.is_file():
        vtr_python = None

    if not vtr_flow.is_file():
        print(f"ERROR: VTR flow script not found at {vtr_flow}")
        sys.exit(1)
    if not SDC_FILE.is_file():
        print(f"ERROR: SDC file not found at {SDC_FILE}")
        sys.exit(1)

    rng = random.Random(rng_seed)
    VTR_DIR.mkdir(parents=True, exist_ok=True)

    # Build all jobs
    jobs = []
    for _, _, _, label, arch_xml_name in archs:
        arch_path = (ARCH_DIR / arch_xml_name).resolve()
        if not arch_path.is_file():
            print(f"ERROR: Arch XML not found: {arch_path}")
            sys.exit(1)

        for S in seq_lens:
            design_name = f"bert_tiny_{label}_s{S}"
            design_path = (RTL_DIR / f"{design_name}.v").resolve()
            if not design_path.is_file():
                print(f"ERROR: RTL not found: {design_path}")
                sys.exit(1)

            design_dir = VTR_DIR / design_name
            design_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_seeds):
                seed = rng.randint(1, 65535)
                run_dir = design_dir / f"run_{i+1}_{seed}"
                jobs.append({
                    "design_path": design_path,
                    "arch_path": arch_path,
                    "run_dir": run_dir,
                    "seed": seed,
                    "run_index": i,
                    "design_name": design_name,
                    "total_runs": num_seeds,
                })

    total = len(jobs)
    workers = min(max_jobs, total, os.cpu_count() or 1)
    print(f"\nLaunching {total} VTR jobs ({len(archs)} archs × "
          f"{len(seq_lens)} seq_lens × {num_seeds} seeds) "
          f"with {workers} workers\n")

    results = {}  # design_name -> [RunResult]
    failures = {}  # design_name -> [(seed, error)]

    def _run_job(job):
        try:
            r = run_single(
                vtr_flow, vtr_python, job["design_path"], job["arch_path"],
                DEFAULT_ROUTE_CHAN_WIDTH, SDC_FILE, job["run_dir"],
                job["seed"], job["run_index"], job["total_runs"],
                job["design_name"],
            )
            return job["design_name"], r, None
        except Exception as exc:
            print(f"  [{job['design_name']}] run {job['run_index']+1} "
                  f"seed={job['seed']} FAILED: {exc}", file=sys.stderr)
            return job["design_name"], None, str(exc)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futs = {executor.submit(_run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            name, result, error = fut.result()
            if result is not None:
                results.setdefault(name, []).append(result)
            if error is not None:
                failures.setdefault(name, []).append((futs[fut]["seed"], error))

    elapsed = time.time() - start_time
    n_ok = sum(len(v) for v in results.values())
    n_fail = sum(len(v) for v in failures.values())
    print(f"\nVTR sweep complete: {n_ok}/{total} OK, {n_fail} failed "
          f"({elapsed:.0f}s)")

    # Write run logs per design
    for _, _, _, label, _ in archs:
        for S in seq_lens:
            design_name = f"bert_tiny_{label}_s{S}"
            design_dir = VTR_DIR / design_name
            rr = results.get(design_name, [])
            ff = failures.get(design_name, [])
            if rr or ff:
                write_run_log(design_dir, design_name, label, rr, ff)

    return results, failures


# ── Step 3: Parse VTR results ───────────────────────────────────────────
def parse_vtr_summary(seq_lens=None, archs=None):
    """Parse all VTR results and produce summary rows."""
    if seq_lens is None:
        seq_lens = SEQ_LENS
    if archs is None:
        archs = ARCH_CONFIGS

    rows = []
    for _, _, _, label, _ in archs:
        for S in seq_lens:
            design_name = f"bert_tiny_{label}_s{S}"
            design_dir = VTR_DIR / design_name

            if not design_dir.is_dir():
                print(f"  SKIP {design_name}: no VTR results")
                continue

            # Find all run directories
            run_dirs = sorted(design_dir.glob("run_*_*"))
            if not run_dirs:
                print(f"  SKIP {design_name}: no run dirs")
                continue

            fmax_values = []
            grid_sizes = []
            resources_used = None
            resources_avail = None

            for rd in run_dirs:
                try:
                    log_path = find_vpr_log(rd)
                except FileNotFoundError:
                    # Try rglob
                    logs = list(rd.rglob("vpr_stdout.log"))
                    if not logs:
                        continue
                    log_path = logs[0]

                txt = log_path.read_text(errors="replace")

                # Fmax
                fmax_matches = re.findall(r'Fmax:\s*([\d.]+)\s*MHz', txt)
                if fmax_matches:
                    fmax_values.append(float(fmax_matches[-1]))

                # Grid size
                grid_match = GRID_SIZE_RE.search(txt)
                if grid_match:
                    grid_sizes.append((int(grid_match.group(1)),
                                       int(grid_match.group(2))))

                # Resources (first seed only for used/avail)
                if resources_used is None:
                    # Parse Netlist (used) and Architecture (available)
                    used = {}
                    avail = {}
                    ctx = None
                    for line in txt.splitlines():
                        if "Netlist" in line and "block" not in line:
                            ctx = "netlist"
                        elif "Architecture" in line and "block" not in line:
                            ctx = "arch"
                        elif ctx and (m := RESOURCE_NETLIST_RE.match(line.strip())):
                            count = int(m.group(1))
                            btype = m.group(2)
                            if ctx == "netlist":
                                used[btype] = count
                            elif ctx == "arch":
                                avail[btype] = count
                    if used:
                        resources_used = used
                        resources_avail = avail

            if not fmax_values:
                print(f"  SKIP {design_name}: no Fmax found")
                continue

            avg_fmax = sum(fmax_values) / len(fmax_values)

            # Use first grid size (should be same across seeds for auto_layout)
            gw, gh = grid_sizes[0] if grid_sizes else (0, 0)
            area_mm2 = gw * gh * CLB_TILE_UM2 / 1e6

            def _get(d, key, default=0):
                return d.get(key, default) if d else default

            row = {
                "arch": label,
                "seq_len": S,
                "fmax_avg": round(avg_fmax, 1),
                "fmax_s1": fmax_values[0] if len(fmax_values) > 0 else 0,
                "fmax_s2": fmax_values[1] if len(fmax_values) > 1 else 0,
                "fmax_s3": fmax_values[2] if len(fmax_values) > 2 else 0,
                "grid_w": gw,
                "grid_h": gh,
                "fpga_area_mm2": round(area_mm2, 4),
                "dpe_used": _get(resources_used, "wc"),
                "dpe_avail": _get(resources_avail, "wc"),
                "dsp_used": _get(resources_used, "dsp_top"),
                "dsp_avail": _get(resources_avail, "dsp_top"),
                "bram_used": _get(resources_used, "memory"),
                "bram_avail": _get(resources_avail, "memory"),
                "clb_used": _get(resources_used, "clb"),
                "clb_avail": _get(resources_avail, "clb"),
            }

            # Compute utilization %
            for res in ["dpe", "dsp", "bram", "clb"]:
                used_key = f"{res}_used"
                avail_key = f"{res}_avail"
                pct_key = f"{res}_pct"
                if row[avail_key] > 0:
                    row[pct_key] = round(row[used_key] / row[avail_key] * 100, 1)
                else:
                    row[pct_key] = 0.0

            rows.append(row)
            print(f"  {design_name}: Fmax={avg_fmax:.1f} MHz, "
                  f"grid={gw}x{gh}, DPEs={row['dpe_used']}/{row['dpe_avail']}")

    return rows


def write_summary(rows):
    """Write CSV and print markdown summary table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "bert_tiny_seqlen_vtr_summary.csv"

    if not rows:
        print("No results to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Print markdown table
    print(f"\n{'='*120}")
    print(f"{'Arch':>12} {'S':>5} {'Fmax':>7} {'Grid':>9} {'Area mm²':>9} "
          f"{'DPE':>10} {'DSP':>10} {'BRAM':>10} {'CLB':>12}")
    print(f"{'-'*120}")
    for r in rows:
        grid_str = f"{r['grid_w']}x{r['grid_h']}"
        dpe_str = f"{r['dpe_used']}/{r['dpe_avail']}({r['dpe_pct']:.0f}%)"
        dsp_str = f"{r['dsp_used']}/{r['dsp_avail']}({r['dsp_pct']:.0f}%)"
        bram_str = f"{r['bram_used']}/{r['bram_avail']}({r['bram_pct']:.0f}%)"
        clb_str = f"{r['clb_used']}/{r['clb_avail']}({r['clb_pct']:.0f}%)"
        print(f"{r['arch']:>12} {r['seq_len']:>5} {r['fmax_avg']:>6.1f} "
              f"{grid_str:>9} {r['fpga_area_mm2']:>8.2f} "
              f"{dpe_str:>10} {dsp_str:>10} {bram_str:>10} {clb_str:>12}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BERT-Tiny seq_len sweep: RTL gen → VTR → summary")
    parser.add_argument("--skip-rtl", action="store_true",
                        help="Skip RTL generation")
    parser.add_argument("--skip-vtr", action="store_true",
                        help="Skip VTR runs (parse existing results)")
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity check: proposed s512, 1 seed")
    parser.add_argument("--jobs", type=int, default=56,
                        help="Max parallel VTR jobs (default: 56)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="VTR seeds per design (default: 3)")
    args = parser.parse_args()

    if args.sanity:
        # Sanity check: just proposed, seq_len=512, 1 seed
        sanity_archs = [ARCH_CONFIGS[0]]  # proposed only
        sanity_lens = [512]
        print("=== SANITY CHECK: proposed s512, 1 seed ===\n")
        if not args.skip_rtl:
            generate_rtl(sanity_lens, sanity_archs)
        if not args.skip_vtr:
            run_vtr_sweep(sanity_lens, sanity_archs, num_seeds=1,
                          max_jobs=1)
        rows = parse_vtr_summary(sanity_lens, sanity_archs)
        write_summary(rows)
        return

    # Full sweep
    if not args.skip_rtl:
        print("=== Step 1: Generating RTL ===\n")
        generate_rtl()

    if not args.skip_vtr:
        print("\n=== Step 2: Running VTR ===\n")
        run_vtr_sweep(num_seeds=args.seeds, max_jobs=args.jobs)

    print("\n=== Step 3: Parsing VTR Results ===\n")
    rows = parse_vtr_summary()
    write_summary(rows)


if __name__ == "__main__":
    main()
