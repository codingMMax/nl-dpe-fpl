#!/usr/bin/env python3
"""VTR sweep for the safe-softmax study: 3 archs x 2 seq lengths x 3 seeds.

Per point:
  1. Assemble a single-file circuit:  `define SYNTHESIS
     + [dpe_blackbox.v (NL only)] + block RTL (LUT include inlined)
     + generated top wrapper (LAST -> Parmys treats it as design top).
  2. Run $VTR_ROOT/vtr_flow/scripts/run_vtr_flow.py against the point's
     arch XML with --route_chan_width 300 --seed N, 3 seeds.
  3. Parse vpr_stdout.log: netlist block counts (clb / dsp_top / wc /
     memory / io), Fmax (clk-to-clk CPD), wirelength.
  4. Seed-average Fmax; resources must be seed-invariant.

Usage:
    python3 softmax_study/run_vtr_softmax.py --point AL_s128 --seed 1   # sanity
    python3 softmax_study/run_vtr_softmax.py --all [--jobs 3]

Outputs: softmax_study/results/vtr_<label>_seed<N>/ (VTR scratch),
         softmax_study/results/vtr_softmax.json, vtr_softmax.log
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RTL = HERE / "rtl"
RESULTS = HERE / "results"
ARCH_DIR = REPO / "benchmarks" / "arch"
DPE_BLACKBOX = REPO / "fc_verification" / "rtl" / "dpe_blackbox.v"

VTR_ROOT = Path(os.environ.get("VTR_ROOT",
                               "/mnt/vault0/jiajunh5/vtr-verilog-to-routing"))
VTR_FLOW = VTR_ROOT / "vtr_flow" / "scripts" / "run_vtr_flow.py"
VTR_PYTHON = VTR_ROOT / ".venv" / "bin" / "python"
if not VTR_PYTHON.is_file():
    VTR_PYTHON = None

ROUTE_CHAN_WIDTH = 300
TIMEOUT_S = 3600
SEEDS = [1, 2, 3]

# (label, arch_xml, S, n_exp, rtl_file, E)
#   n_exp None => AL block, E = its datapath width (elements/cycle/lane)
#   AL_*  : unconstrained AL, E=16 (128-bit operand feed)
#   AL5_* : supply-matched AL, E=5 (40 bit/cycle = one NL-DPE port)
POINTS = [
    ("AL_s128",  "azure_lily_auto.xml", 128, None, "softmax_al.v",    16),
    ("AL_s256",  "azure_lily_auto.xml", 256, None, "softmax_al.v",    16),
    ("AL5_s128", "azure_lily_auto.xml", 128, None, "softmax_al.v",     5),
    ("AL5_s256", "azure_lily_auto.xml", 256, None, "softmax_al.v",     5),
    ("P1_s128",  "proposed_auto.xml",   128, 1,    "softmax_nldpe.v", None),
    ("P1_s256",  "proposed_auto.xml",   256, 2,    "softmax_nldpe.v", None),
    ("P2_s128",  "al_like_auto.xml",    128, 1,    "softmax_nldpe.v", None),
    ("P2_s256",  "al_like_auto.xml",    256, 1,    "softmax_nldpe.v", None),
]

# ── parsers (conventions of fc_verification/run_vtr_smoke.py) ───────────
WIRE_RE = re.compile(r"Total wirelength:\s*([0-9]+(?:\.[0-9]+)?)")
GRID_RE = re.compile(r"FPGA sized to (\d+) x (\d+)")
CPD_RE = re.compile(
    r"([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)\s+CPD:\s*([\d.]+)\s*ns\s*\(([\d.]+)\s*MHz\)")
FMAX_RE = re.compile(r"Fmax:\s*([\d.]+)\s*MHz")
RESOURCE_HEADER_RE = re.compile(r"^Resource usage", re.IGNORECASE)
CONTEXT_RE = re.compile(r"^\s*(Netlist|Architecture)\s*$")
COUNT_RE = re.compile(r"^\s*(\d+)\s+blocks of type:\s+(\S+)")


def find_vpr_log(run_dir: Path) -> Path:
    for name in ("vpr_stdout.log", "vpr.out"):
        for base in (run_dir, run_dir / "temp"):
            p = base / name
            if p.is_file():
                return p
    return run_dir / "vpr_stdout.log"


def parse_metrics(log: Path) -> tuple[float, float, tuple[int, int]]:
    content = log.read_text(errors="replace")
    wire = WIRE_RE.findall(content)
    wirelength = float(wire[-1]) if wire else 0.0
    cpd = CPD_RE.findall(content)
    if cpd:
        clk = next((m for m in cpd if m[0] == m[1] == "clk"), None)
        fmax = float((clk or cpd[0])[3])
    else:
        m = FMAX_RE.findall(content)
        fmax = float(m[-1]) if m else 0.0
    g = GRID_RE.findall(content)
    grid = (int(g[-1][0]), int(g[-1][1])) if g else (0, 0)
    return wirelength, fmax, grid


def parse_resources(log: Path) -> dict[str, int]:
    if not log.is_file():
        return {}
    lines = log.read_text(errors="replace").splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if RESOURCE_HEADER_RE.search(ln)), None)
    if start is None:
        return {}
    res, ctx = {}, None
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


# ── circuit assembly ────────────────────────────────────────────────────
def inline_includes(rtl_text: str) -> str:
    """Textually inline `include "softmax_luts.vh" (VTR gets one flat file)."""
    lut_text = (RTL / "softmax_luts.vh").read_text()
    out = []
    for line in rtl_text.splitlines():
        if "`include" in line and "softmax_luts.vh" in line:
            out.append(lut_text)
        else:
            out.append(line)
    return "\n".join(out)


def gen_wrapper(label: str, S: int, n_exp: int | None, E: int | None) -> str:
    """Top wrapper. AL and NL differ in port list: AL takes (word, byte
    offset) so it needs no divider for non-power-of-2 E; NL takes a flat
    element index (its words are 16 elements, a power of two)."""
    if n_exp is None:
        return f"""
// Auto-generated by run_vtr_softmax.py -- top wrapper for {label}.
module softmax_vtr_top (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        done,
    input  wire        in_wen,
    input  wire [3:0]  in_lane,
    input  wire [13:0] in_waddr,
    input  wire [3:0]  in_boff,
    input  wire [7:0]  in_data,
    input  wire [3:0]  out_lane,
    input  wire [13:0] out_waddr,
    input  wire [3:0]  out_boff,
    output wire [7:0]  out_rdata
);
    softmax_al #(.S({S}), .E({E})) u_blk (
        .clk(clk), .reset(reset), .start(start), .done(done),
        .in_wen(in_wen), .in_lane(in_lane), .in_waddr(in_waddr),
        .in_boff(in_boff), .in_data(in_data),
        .out_lane(out_lane), .out_waddr(out_waddr), .out_boff(out_boff),
        .out_rdata(out_rdata)
    );
endmodule
"""
    return f"""
// Auto-generated by run_vtr_softmax.py -- top wrapper for {label}.
module softmax_vtr_top (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        done,
    input  wire        in_wen,
    input  wire [3:0]  in_lane,
    input  wire [13:0] in_addr,
    input  wire [7:0]  in_data,
    input  wire [3:0]  out_lane,
    input  wire [13:0] out_addr,
    output wire [7:0]  out_rdata
);
    softmax_nldpe #(.S({S}), .N_EXP({n_exp})) u_blk (
        .clk(clk), .reset(reset), .start(start), .done(done),
        .in_wen(in_wen), .in_lane(in_lane), .in_addr(in_addr),
        .in_data(in_data),
        .out_lane(out_lane), .out_addr(out_addr), .out_rdata(out_rdata)
    );
endmodule
"""


def build_circuit(label: str, S: int, n_exp: int | None, rtl_file: str,
                  out_dir: Path, E: int | None = None) -> Path:
    parts = ["`define SYNTHESIS 1\n"]
    if n_exp is not None:
        parts.append(DPE_BLACKBOX.read_text())
    parts.append(inline_includes((RTL / rtl_file).read_text()))
    parts.append(gen_wrapper(label, S, n_exp, E))
    out_dir.mkdir(parents=True, exist_ok=True)
    circuit = out_dir / f"softmax_{label}.v"
    circuit.write_text("\n".join(parts))
    return circuit


# ── one VTR run ─────────────────────────────────────────────────────────
def run_one(label: str, arch_xml: str, S: int, n_exp: int | None,
            rtl_file: str, E: int | None, seed: int) -> dict:
    run_dir = RESULTS / f"vtr_{label}_seed{seed}"
    input_dir = RESULTS / f"vtr_{label}_seed{seed}_input"
    for d in (run_dir, input_dir):
        if d.exists():
            shutil.rmtree(d)
    run_dir.mkdir(parents=True)
    circuit = build_circuit(label, S, n_exp, rtl_file, input_dir, E)

    cmd = [str(VTR_FLOW), str(circuit), str(ARCH_DIR / arch_xml),
           "-temp_dir", str(run_dir),
           "--route_chan_width", str(ROUTE_CHAN_WIDTH),
           # Per-lane-addressed wide memories (AL exp banks) have addr-net
           # fanout >= the default memory:128 threshold, which removes them
           # from clustering attraction and scatters BRAM slices 2-3 per
           # block (measured: AL_s128 mem=1542 vs ~284 expected). Raising
           # the threshold restores same-address packing.
           "--pack_high_fanout_threshold", "memory:100000",
           "--seed", str(seed)]
    cmd = ([str(VTR_PYTHON)] if VTR_PYTHON else [sys.executable]) + cmd

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(run_dir), timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return dict(label=label, seed=seed, status="TIMEOUT",
                    elapsed_s=time.time() - t0)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return dict(label=label, seed=seed, status="FAILED",
                    elapsed_s=elapsed,
                    stdout_tail=(proc.stdout or "").splitlines()[-15:],
                    stderr_tail=(proc.stderr or "").splitlines()[-15:])

    log = find_vpr_log(run_dir)
    wirelength, fmax, grid = parse_metrics(log)
    resources = parse_resources(log)
    return dict(label=label, seed=seed, status="OK", elapsed_s=elapsed,
                fmax_mhz=fmax, wirelength=wirelength, resources=resources,
                grid=list(grid))


# ── sweep ───────────────────────────────────────────────────────────────
def rebuild_aggregate() -> int:
    """Re-parse every kept results/vtr_<label>_seed<n>/vpr_stdout.log and
    regenerate the aggregate JSON without re-running VTR."""
    agg = []
    for (lbl, ax, S, ne, rf, e) in POINTS:
        runs = []
        for seed in SEEDS:
            log = RESULTS / f"vtr_{lbl}_seed{seed}" / "vpr_stdout.log"
            if not log.is_file():
                continue
            wl, fmax, grid = parse_metrics(log)
            res = parse_resources(log)
            if fmax and res:
                runs.append(dict(seed=seed, fmax_mhz=fmax, wirelength=wl,
                                 resources=res, grid=list(grid)))
        if not runs:
            print(f"  {lbl}: no logs found, skipped")
            continue
        fmaxes = [r["fmax_mhz"] for r in runs]
        agg.append(dict(
            label=lbl, S=S, arch=ax, n_exp=ne, E=e, status="OK",
            seeds=[r["seed"] for r in runs], fmax_seeds=fmaxes,
            fmax_avg_mhz=sum(fmaxes) / len(fmaxes),
            wirelength_avg=sum(r["wirelength"] for r in runs) / len(runs),
            resources=runs[0]["resources"], grid=runs[0]["grid"],
        ))
        print(f"  {lbl}: {len(runs)} seeds, fmax_avg="
              f"{sum(fmaxes)/len(fmaxes):.2f} MHz")
    (RESULTS / "vtr_softmax.json").write_text(json.dumps(agg, indent=2))
    print(f"rebuilt {len(agg)} points -> {RESULTS / 'vtr_softmax.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--point", action="append",
                    help="run only these labels (e.g. AL_s128)")
    ap.add_argument("--seed", type=int, action="append",
                    help="run only these seeds (default 1,2,3)")
    ap.add_argument("--all", action="store_true", help="full sweep")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the aggregate JSON from kept VPR logs "
                         "(no VTR runs)")
    args = ap.parse_args()

    if args.rebuild:
        return rebuild_aggregate()

    if not VTR_FLOW.is_file():
        print(f"ERROR: VTR flow not found at {VTR_FLOW}", file=sys.stderr)
        return 2

    points = [p for p in POINTS if not args.point or p[0] in args.point]
    seeds = args.seed or SEEDS
    if not args.all and not args.point:
        points = points[:1]
        seeds = seeds[:1]

    jobs = [(lbl, ax, S, ne, rf, e, sd)
            for (lbl, ax, S, ne, rf, e) in points for sd in seeds]
    print(f"{len(jobs)} VTR runs, jobs={args.jobs}")

    runs: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            runs.append(r)
            if r["status"] == "OK":
                res = r["resources"]
                print(f"[{r['label']} seed{r['seed']}] OK "
                      f"fmax={r['fmax_mhz']:.2f} MHz "
                      f"clb={res.get('clb', 0)} dsp={res.get('dsp_top', 0)} "
                      f"wc={res.get('wc', 0)} mem={res.get('memory', 0)} "
                      f"({r['elapsed_s']:.0f}s)")
            else:
                print(f"[{r['label']} seed{r['seed']}] {r['status']}")
                for k in ("stdout_tail", "stderr_tail"):
                    for ln in r.get(k, []):
                        print(f"    {ln}")

    # aggregate per point
    agg = []
    for (lbl, ax, S, ne, rf, e) in points:
        mine = [r for r in runs if r["label"] == lbl and r["status"] == "OK"]
        if not mine:
            agg.append(dict(label=lbl, S=S, arch=ax, n_exp=ne, E=e,
                            status="FAILED"))
            continue
        fmaxes = [r["fmax_mhz"] for r in mine]
        agg.append(dict(
            label=lbl, S=S, arch=ax, n_exp=ne, E=e, status="OK",
            seeds=[r["seed"] for r in mine],
            fmax_seeds=fmaxes,
            fmax_avg_mhz=sum(fmaxes) / len(fmaxes),
            wirelength_avg=sum(r["wirelength"] for r in mine) / len(mine),
            resources=mine[0]["resources"],
            grid=mine[0].get("grid", [0, 0]),
        ))

    RESULTS.mkdir(exist_ok=True)
    # Merge with any existing aggregate: a partial run (--point) must not
    # clobber results for points it did not run.
    agg_path = RESULTS / "vtr_softmax.json"
    merged = {}
    if agg_path.is_file():
        try:
            for r in json.loads(agg_path.read_text()):
                merged[r["label"]] = r
        except Exception:
            pass
    for r in agg:
        merged[r["label"]] = r
    order = [p[0] for p in POINTS]
    agg_out = sorted(merged.values(),
                     key=lambda r: order.index(r["label"])
                     if r["label"] in order else 99)
    agg_path.write_text(json.dumps(agg_out, indent=2))
    with (RESULTS / "vtr_softmax.log").open("w") as f:
        f.write(json.dumps(dict(runs=runs, aggregate=agg), indent=2))
    print(f"\nAggregate -> {RESULTS / 'vtr_softmax.json'}")

    n_ok = sum(1 for a in agg if a["status"] == "OK")
    return 0 if n_ok == len(points) else 1


if __name__ == "__main__":
    sys.exit(main())
