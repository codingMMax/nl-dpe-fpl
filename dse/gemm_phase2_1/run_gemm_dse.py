#!/usr/bin/env python3
"""Phase 2.1 GEMM DSE smoke-test driver.

Mirrors ``gemv_dse.py::run_round1`` for 4 GEMM workloads × 12 (R, C)
configs.  Generates FC-wrapper RTL, runs VTR once per config with a
single seed (42 — Round-1 behaviour), invokes the IMC sim with the
VTR-reported Fmax and the new ``--batch M`` flag, then emits a 48-row
CSV plus the analytical Regime B prediction columns per row.

Usage
-----
    python3 dse/gemm_phase2_1/run_gemm_dse.py --smoke [--dry-run]
    python3 dse/gemm_phase2_1/run_gemm_dse.py --smoke \
        --workloads bert_qkvo --configs 256x128 --jobs 1

CLI
---
  --smoke              (default, no-op flag kept for interface parity)
  --workloads NAMES    comma-separated subset of GEMM_WORKLOADS
  --configs RxC,…      comma-separated (R, C) pairs (subset of DPE_CONFIGS_PHASE21)
  --jobs N             VTR parallelism (default 12, capped at 12 per plan)
  --dry-run            print the job table and exit
  --skip-existing      reuse run_dir/imc_result.json if present
  --force              re-run even if imc_result.json exists
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../nl-dpe-fpl
DSE_DIR = Path(__file__).resolve().parent             # .../dse/gemm_phase2_1

sys.path.insert(0, str(PROJECT_ROOT / "nl_dpe"))
from run_vtr import run_single, find_vpr_log, parse_metrics, parse_resources  # noqa: E402
from gen_arch_xml import gen_arch_xml  # noqa: E402
from gen_gemv_wrappers import gen_fc_wrapper  # noqa: E402
from area_power import dpe_specs  # noqa: E402

# Reuse Round-1 constants / helpers by importing from the parent driver.
sys.path.insert(0, str(PROJECT_ROOT))
import gemv_dse as _g1  # noqa: E402

# ── VTR paths (mirror Round-1) ────────────────────────────────────────────
VTR_FLOW = _g1.VTR_FLOW
VTR_PYTHON = _g1.VTR_PYTHON
AZURELILY_ROOT = _g1.AZURELILY_ROOT
IMC_TEST = _g1.IMC_TEST
BASE_IMC_CONFIG = _g1.BASE_IMC_CONFIG
CLB_TILE_UM2 = _g1.CLB_TILE_UM2
DEFAULT_ROUTE_CHAN_WIDTH = _g1.DEFAULT_ROUTE_CHAN_WIDTH
DEFAULT_SEED = _g1.DEFAULT_SEED
parse_grid_size = _g1.parse_grid_size
patch_imc_config = _g1.patch_imc_config
RE_ENERGY_LAYER = _g1.RE_ENERGY_LAYER
RE_LAT_CRIT = _g1.RE_LAT_CRIT
RE_ENERGY_GROUPED = _g1.RE_ENERGY_GROUPED
RE_BREAKDOWN_LINE = _g1.RE_BREAKDOWN_LINE

# ── GEMM smoke-test workloads ─────────────────────────────────────────────
# Tuples are (M, K, N) with M = batch / GEMM row dimension.
GEMM_WORKLOADS = {
    "bert_qkvo":     (128, 128, 128),   # BERT-Tiny Q/K/V/O projection
    "bert_ffn2":     (128, 512, 128),   # BERT-Tiny FFN2 down-projection
    "swin_mlp":      (49,  384, 1536),  # Swin-Tiny stage-3 MLP up-projection
    "resnet9_conv":  (256, 2304, 256),  # ResNet-9 mid conv (im2col)
}

# ── (R, C) grid — 12 configs ──────────────────────────────────────────────
DSE_R = (128, 256, 512, 1024)
DSE_C = (64, 128, 256)
DPE_CONFIGS_PHASE21 = [(r, c) for r in DSE_R for c in DSE_C]

# ── Analytical Regime B constants (§7 gemm_rc_tradeoff.md) ────────────────
W_DPE = 40          # DPE data bus width, matches dpe_data_width=40
N_B = 8             # per-element bit width (int8 weights / activations)
T_ACAM = 3          # ACAM bit-serial compute cycles (ADC would be 44)

# ── CSV schema ────────────────────────────────────────────────────────────
CSV_COLUMNS = list(_g1.CSV_COLUMNS) + [
    'M',
    'L_A_cyc', 'O_cyc', 'steady_B_cyc',
    'T_pred_cyc', 'T_pred_ns',
    'regime',
]

# ── Regime B analytical prediction ────────────────────────────────────────
def regime_b_prediction(M: int, K: int, N: int, R: int, C: int,
                        fmax_mhz: float) -> dict:
    """Compute the analytical Regime B per-row formula for (M, K, N) on (R, C).

    Formulas (from ``paper/methodology/gemm_rc_tradeoff.md`` §7):
        L_feed = ceil(R * 8 / W_DPE)
        L_comp = N_b + t_acam
        L_A    = L_feed + L_comp
        O      = ceil(C * 8 / W_DPE)
        steady = max(L_A, O)
        V = ceil(K / R), H = ceil(N / C)
        M_eff  = M * ceil(N / (H * C))
        T      = L_A + steady * (M_eff - 1) + O
    """
    L_feed = math.ceil(R * N_B / W_DPE)
    L_comp = N_B + T_ACAM
    L_A = L_feed + L_comp
    O = math.ceil(C * N_B / W_DPE)
    steady_B = max(L_A, O)
    V = math.ceil(K / R)
    H = math.ceil(N / C)
    # ceil(N / (H * C)) is 1 whenever H = ceil(N / C); defensive math below.
    M_eff = M * max(1, math.ceil(N / max(1, H * C)))
    T_pred_cyc = L_A + steady_B * (M_eff - 1) + O
    T_pred_ns = (T_pred_cyc * 1000.0 / fmax_mhz) if fmax_mhz else None

    if L_A > O:
        regime = "feed-bound"
    elif O > L_A:
        regime = "drain-bound"
    else:
        regime = "balanced"

    return {
        'L_A_cyc': L_A,
        'O_cyc': O,
        'steady_B_cyc': steady_B,
        'T_pred_cyc': T_pred_cyc,
        'T_pred_ns': T_pred_ns,
        'regime': regime,
        'V': V, 'H': H, 'M_eff': M_eff,
    }


# ── IMC runner with --batch ───────────────────────────────────────────────
def run_imc_fc_gemm(config_path: Path, K: int, N: int, M: int):
    """Run the IMC simulator for an FC GEMM workload with explicit batch M.

    Returns: (energy_pj, latency_ns, breakdown_dict)
    """
    cmd = [
        sys.executable, str(IMC_TEST),
        "--model", "fc",
        "--imc_file", str(config_path),
        "--seq_length", str(K),
        "--head_dim", str(N),
        "--batch", str(M),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(AZURELILY_ROOT)
    )
    out = result.stdout + result.stderr

    m = RE_ENERGY_LAYER.search(out)
    energy_pj = float(m.group(1)) if m else None

    m = RE_LAT_CRIT.search(out)
    latency_ns = float(m.group(1)) if m else None

    breakdown = {}
    for m in RE_BREAKDOWN_LINE.finditer(out):
        breakdown[m.group(1)] = float(m.group(2))

    m = RE_ENERGY_GROUPED.search(out)
    if m:
        breakdown['_dpe_grouped'] = float(m.group(1))
        breakdown['_mem_grouped'] = float(m.group(2))
        breakdown['_fpga_grouped'] = float(m.group(3))

    if result.returncode != 0 and energy_pj is None:
        print(f"    [IMC] WARNING: non-zero exit ({result.returncode}), "
              f"last output:\n{out[-500:]}", file=sys.stderr)

    return energy_pj, latency_ns, breakdown


# ── VTR DSE wrapper (mirrors Round-1 run_vtr_dse) ─────────────────────────
def run_vtr_gemm(rtl: Path, arch: Path, run_dir: Path,
                 rows: int, cols: int, K: int, N: int,
                 wl_name: str) -> dict:
    config_name = f"{rows}x{cols}"
    design_name = f"fc_{K}_{N}_{config_name}"
    result = run_single(
        vtr_flow=VTR_FLOW,
        vtr_python=VTR_PYTHON,
        design=rtl,
        arch=arch,
        route_chan_width=DEFAULT_ROUTE_CHAN_WIDTH,
        sdc_file=None,
        run_dir=run_dir,
        seed=DEFAULT_SEED,
        run_index=0,
        total_runs=1,
        design_name=design_name,
    )
    log_path = find_vpr_log(run_dir)
    try:
        grid_w, grid_h = parse_grid_size(log_path)
    except ValueError:
        grid_w, grid_h = 0, 0
    return {
        'fmax_mhz': result.fmax_mhz,
        'grid_w': grid_w,
        'grid_h': grid_h,
        'resources': result.resources,
        'run_dir': str(run_dir),
        'wirelength': result.wirelength,
        'elapsed_s': result.elapsed_s,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true',
                    help="Run the 4-workload × 12-config smoke sweep (default).")
    ap.add_argument('--workloads', type=str, default=None,
                    help="Comma-separated workload names (default: all 4).")
    ap.add_argument('--configs', type=str, default=None,
                    help="Comma-separated (R, C) pairs, e.g. '256x128,512x128'.")
    ap.add_argument('--jobs', type=int, default=12,
                    help="VTR parallelism (default 12, capped at 12).")
    ap.add_argument('--dry-run', action='store_true',
                    help="Print the job table and exit.")
    ap.add_argument('--skip-existing', action='store_true',
                    help="Skip configs whose run_dir/imc_result.json exists.")
    ap.add_argument('--force', action='store_true',
                    help="Re-run even if imc_result.json exists.")
    args = ap.parse_args()

    if args.jobs > 12:
        print(f"[WARN] --jobs {args.jobs} exceeds plan cap of 12; clamping.",
              file=sys.stderr)
        args.jobs = 12

    # ── Parse filters ────────────────────────────────────────────────────
    if args.configs:
        configs = []
        for s in args.configs.split(","):
            s = s.strip()
            r, c = s.split("x")
            configs.append((int(r), int(c)))
    else:
        configs = list(DPE_CONFIGS_PHASE21)

    if args.workloads:
        names = [w.strip() for w in args.workloads.split(",")]
        workloads = {k: v for k, v in GEMM_WORKLOADS.items() if k in names}
        missing = set(names) - set(workloads.keys())
        if missing:
            print(f"[WARN] unknown workloads: {missing}", file=sys.stderr)
    else:
        workloads = dict(GEMM_WORKLOADS)

    # ── Dirs ─────────────────────────────────────────────────────────────
    arch_dir = DSE_DIR / "configs" / "arch"
    imc_dir = DSE_DIR / "configs" / "imc"
    rtl_dir = DSE_DIR / "rtl"
    run_root = DSE_DIR / "vtr_runs"
    results_dir = DSE_DIR / "results"
    for d in [arch_dir, imc_dir, rtl_dir, run_root, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: generate arch XML + RTL ────────────────────────────────
    print("=" * 70)
    print("Phase 1: generating architecture XMLs and FC RTL")
    print("=" * 70)
    for (R, C) in configs:
        existing_arch = arch_dir / f"nl_dpe_{R}x{C}_auto.xml"
        if not existing_arch.is_file():
            # Try the legacy repo-wide arch dir first (most configs already built).
            legacy = PROJECT_ROOT / "dse" / "configs" / "arch" / f"nl_dpe_{R}x{C}_auto.xml"
            if legacy.is_file():
                existing_arch.write_bytes(legacy.read_bytes())
                print(f"  [arch] reused {legacy.name}")
            else:
                gen_arch_xml(R, C, mode="auto", output_dir=arch_dir)
                print(f"  [arch] generated {existing_arch.name}")
        for wl_name, (M, K, N) in workloads.items():
            rtl_path = rtl_dir / f"fc_{K}_{N}_{R}x{C}_acam_dw40.v"
            if not rtl_path.is_file():
                gen_fc_wrapper(K, N, R, C, output_dir=rtl_dir,
                               conversion="acam", dpe_data_width=40)
            # VTR-compatibility strip: the blackbox ``dpe`` model in the
            # arch XML does not carry KERNEL_WIDTH / NUM_COLS parameters,
            # but gen_fc_wrapper's V>1 branch emits them for TB alignment
            # (commit 2678040).  Parmys rejects those params during synth,
            # so we produce a sibling RTL with the params stripped and
            # feed that file to VTR.  The TB-facing original is left
            # untouched so block_comp_apr_11 phase-2 verification still
            # receives the full param set from the dpe_stub.
            vtr_rtl_path = rtl_dir / f"fc_{K}_{N}_{R}x{C}_acam_dw40_vtr.v"
            src_text = rtl_path.read_text()
            # Strip the "dpe #(.KERNEL_WIDTH(X), .NUM_COLS(Y))" decorator
            # so the instantiation becomes the plain "dpe inst_name (" form
            # that matches conv_layer_single_dpe's working V=1 usage.  The
            # pattern uses two explicit ``\([^)]*\)`` groups because the
            # outer ``#(...)`` contains inner parenthesised values and a
            # simple ``[^)]*`` stops at the first inner ``)``.
            stripped = re.sub(
                r"dpe\s*#\(\.KERNEL_WIDTH\([^)]*\)\s*,\s*"
                r"\.NUM_COLS\([^)]*\)\)\s+(dpe_c\d+_r\d+)",
                r"dpe \1",
                src_text,
            )
            vtr_rtl_path.write_text(stripped)

    # ── Phase 2: build job list ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: building VTR job list")
    print("=" * 70)
    jobs = []
    for (R, C) in configs:
        arch_path = arch_dir / f"nl_dpe_{R}x{C}_auto.xml"
        if not arch_path.is_file():
            print(f"  [WARN] arch XML missing: {arch_path}", file=sys.stderr)
            continue
        for wl_name, (M, K, N) in workloads.items():
            # Use the stripped (VTR-compatible) sibling when present.
            vtr_rtl_path = rtl_dir / f"fc_{K}_{N}_{R}x{C}_acam_dw40_vtr.v"
            rtl_path = vtr_rtl_path if vtr_rtl_path.is_file() else (
                rtl_dir / f"fc_{K}_{N}_{R}x{C}_acam_dw40.v"
            )
            if not rtl_path.is_file():
                print(f"  [WARN] RTL missing: {rtl_path}", file=sys.stderr)
                continue
            V = math.ceil(K / R)
            H = math.ceil(N / C)
            dpe_count = V * H
            run_dir = run_root / f"{R}x{C}" / wl_name
            acam = (V == 1)
            job = {
                'rtl': rtl_path, 'arch': arch_path, 'run_dir': run_dir,
                'R': R, 'C': C, 'wl_name': wl_name,
                'M': M, 'K': K, 'N': N,
                'V': V, 'H': H, 'dpe_count': dpe_count,
                'acam_eligible': acam,
            }
            imc_result = run_dir / "imc_result.json"
            if args.skip_existing and not args.force and imc_result.exists():
                print(f"  SKIP (existing): {R}x{C} / {wl_name}")
                job['_skipped'] = True
            jobs.append(job)

    active_jobs = [j for j in jobs if not j.get('_skipped')]
    print(f"\n  Total jobs: {len(jobs)}   to-run: {len(active_jobs)}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN — job table")
        print("=" * 70)
        hdr = (f"{'Config':<10} {'Workload':<14} {'M':>4} {'K':>5} {'N':>5} "
               f"{'V':>3} {'H':>3} {'DPEs':>5} {'acam':>5} RTL")
        print(hdr); print("-" * len(hdr))
        for j in jobs:
            acam_str = "yes" if j['acam_eligible'] else "no"
            print(f"{j['R']}x{j['C']:<6} {j['wl_name']:<14} "
                  f"{j['M']:>4} {j['K']:>5} {j['N']:>5} "
                  f"{j['V']:>3} {j['H']:>3} {j['dpe_count']:>5} {acam_str:>5} "
                  f"{j['rtl'].name}")
        return 0

    # ── Phase 3: run VTR in parallel ────────────────────────────────────
    vtr_results = {}
    print("\n" + "=" * 70)
    print(f"Phase 3: running VTR synth (workers={args.jobs})")
    print("=" * 70)

    t0 = time.time()

    def _one(job):
        try:
            r = run_vtr_gemm(
                rtl=job['rtl'], arch=job['arch'], run_dir=job['run_dir'],
                rows=job['R'], cols=job['C'],
                K=job['K'], N=job['N'], wl_name=job['wl_name'],
            )
            return job, r, None
        except Exception as e:
            return job, None, str(e)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(_one, j): j for j in active_jobs}
        for fut in as_completed(futures):
            job, result, err = fut.result()
            key = (job['R'], job['C'], job['wl_name'])
            if err:
                print(f"  FAILED: {key}: {err}", file=sys.stderr)
            else:
                vtr_results[key] = result
                print(f"  OK: {job['R']}x{job['C']} / {job['wl_name']}  "
                      f"fmax={result['fmax_mhz']:.1f} MHz  "
                      f"grid={result['grid_w']}x{result['grid_h']}  "
                      f"wc={result['resources'].get('wc', '?')}")

    print(f"\n  Completed: {len(vtr_results)}/{len(active_jobs)} "
          f"in {time.time() - t0:.1f}s")

    # ── Phase 4: run IMC sequentially ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 4: running IMC simulator")
    print("=" * 70)
    all_results = []
    for job in jobs:
        key = (job['R'], job['C'], job['wl_name'])
        run_dir = job['run_dir']
        imc_result_path = run_dir / "imc_result.json"

        # Reload path — either skipped or --force disabled
        if job.get('_skipped'):
            try:
                with open(imc_result_path) as f:
                    imc = json.load(f)
                log_path = find_vpr_log(run_dir)
                grid_w, grid_h = parse_grid_size(log_path)
                resources = parse_resources(log_path)
                vtr = {
                    'fmax_mhz': imc['fmax_mhz'],
                    'grid_w': grid_w, 'grid_h': grid_h,
                    'resources': resources, 'run_dir': str(run_dir),
                }
                energy_pj = imc['energy_pj']
                latency_ns = imc['latency_ns']
                breakdown = imc.get('breakdown', {})
                print(f"  RELOAD: {key} fmax={vtr['fmax_mhz']:.1f}")
            except Exception as e:
                print(f"  [WARN] reload failed for {key}: {e}", file=sys.stderr)
                continue
        else:
            if key not in vtr_results:
                continue
            vtr = vtr_results[key]

            # Patch IMC config at this Fmax
            config_name = f"{job['R']}x{job['C']}"
            imc_cfg_path = imc_dir / f"nl_dpe_{config_name}_{job['wl_name']}.json"
            patch_imc_config(BASE_IMC_CONFIG, job['R'], job['C'],
                             vtr['fmax_mhz'], imc_cfg_path)

            print(f"  IMC: {config_name} / {job['wl_name']} "
                  f"@ {vtr['fmax_mhz']:.1f} MHz (M={job['M']}) ... ",
                  end="", flush=True)
            energy_pj, latency_ns, breakdown = run_imc_fc_gemm(
                imc_cfg_path, job['K'], job['N'], job['M']
            )
            if energy_pj is not None and latency_ns is not None:
                print(f"E={energy_pj:.2f} pJ  L={latency_ns:.2f} ns")
            else:
                print("PARSE FAILED")
                continue

            # Save imc_result.json for --skip-existing on reruns
            imc_out = {
                'config': config_name,
                'workload': job['wl_name'],
                'M': job['M'], 'K': job['K'], 'N': job['N'],
                'fmax_mhz': vtr['fmax_mhz'],
                'energy_pj': energy_pj,
                'latency_ns': latency_ns,
                'breakdown': breakdown,
            }
            imc_result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(imc_result_path, 'w') as f:
                json.dump(imc_out, f, indent=2)

        # Derived metrics
        grid_w = vtr['grid_w']; grid_h = vtr['grid_h']
        fpga_area_mm2 = grid_w * grid_h * CLB_TILE_UM2 / 1e6
        throughput_per_s = 1e9 / latency_ns if latency_ns > 0 else 0
        throughput_per_mm2 = (throughput_per_s / fpga_area_mm2
                              if fpga_area_mm2 > 0 else 0)
        throughput_per_J = 1e12 / energy_pj if energy_pj > 0 else 0
        edap = (energy_pj * latency_ns * fpga_area_mm2
                if (energy_pj > 0 and latency_ns > 0) else float('inf'))

        resources = vtr.get('resources', {})
        pred = regime_b_prediction(
            job['M'], job['K'], job['N'], job['R'], job['C'], vtr['fmax_mhz'],
        )

        row = {
            'config': f"{job['R']}x{job['C']}",
            'rows': job['R'], 'cols': job['C'],
            'workload': job['wl_name'], 'wl_type': 'gemm',
            'K': job['K'], 'N': job['N'],
            'n_seq': 0, 'd_head': 0, 'h_dimm': 0, 'dimm_dpes': 0,
            'V': job['V'], 'H': job['H'],
            'dpe_count': job['dpe_count'],
            'acam_eligible': job['acam_eligible'],
            'fmax_mhz': vtr['fmax_mhz'],
            'grid_w': grid_w, 'grid_h': grid_h,
            'fpga_area_mm2': fpga_area_mm2,
            'latency_ns': latency_ns, 'energy_pj': energy_pj,
            'throughput_per_mm2': throughput_per_mm2,
            'throughput_per_J': throughput_per_J,
            'edap': edap,
            'e_imc_vmm': breakdown.get('imc_vmm', 0),
            'e_imc_digital': breakdown.get('imc_digital_post', 0),
            'e_clb_reduction': breakdown.get('clb_reduction', 0),
            'e_clb_activation': breakdown.get('fpga_activation', 0),
            'e_sram': (breakdown.get('sram_read', 0)
                       + breakdown.get('sram_write', 0)),
            'clb_count': resources.get('clb', 0),
            'dsp_count': resources.get('dsp_top', 0),
            'mem_count': resources.get('memory', 0),
            'wc_count': resources.get('wc', 0),
            'run_dir': str(run_dir),
            # Phase-2.1 additions
            'M': job['M'],
            'L_A_cyc': pred['L_A_cyc'],
            'O_cyc': pred['O_cyc'],
            'steady_B_cyc': pred['steady_B_cyc'],
            'T_pred_cyc': pred['T_pred_cyc'],
            'T_pred_ns': pred['T_pred_ns'],
            'regime': pred['regime'],
        }

        # Escalation check: |sim - analytical| / analytical > 2.0
        if pred['T_pred_ns']:
            err = abs(latency_ns - pred['T_pred_ns']) / pred['T_pred_ns']
            if err > 2.0:
                print(f"  [WARN] large T_pred vs sim mismatch at {key}: "
                      f"sim={latency_ns:.1f} ns  pred={pred['T_pred_ns']:.1f} ns "
                      f"(rel err {err:.2f})", file=sys.stderr)

        all_results.append(row)

    # ── Phase 5: write CSV ──────────────────────────────────────────────
    csv_path = results_dir / "gemm_dse_smoke.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        # Sort by workload, then by throughput/mm² desc for readability.
        for r in sorted(all_results,
                        key=lambda x: (x['workload'], -x['throughput_per_mm2'])):
            # Fill any missing keys (defensive)
            for k in CSV_COLUMNS:
                r.setdefault(k, '')
            writer.writerow(r)
    print(f"\n  Written: {csv_path}  ({len(all_results)} rows)")

    # ── Phase 6: per-workload summary ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Per-workload summary  (best latency_ns = sweet spot)")
    print("=" * 70)
    for wl_name in workloads:
        rows = [r for r in all_results if r['workload'] == wl_name]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda x: x['latency_ns'])
        best = rows_sorted[0]
        print(f"\n  {wl_name} (M={best['M']} K={best['K']} N={best['N']})")
        hdr = (f"    {'Config':<10} {'Fmax':>7} {'DPEs':>5} {'CLB':>5} "
               f"{'wc':>4} {'Lat ns':>10} {'T_pred ns':>10} "
               f"{'L_A':>4} {'O':>4} {'regime':<12}")
        print(hdr)
        for r in rows_sorted:
            regime_short = {'feed-bound': 'F', 'drain-bound': 'D',
                            'balanced': 'B'}.get(r['regime'], '?')
            tp = f"{r['T_pred_ns']:.1f}" if r['T_pred_ns'] else "—"
            print(f"    {r['config']:<10} {r['fmax_mhz']:>7.1f} "
                  f"{r['dpe_count']:>5} {r['clb_count']:>5} "
                  f"{r['wc_count']:>4} {r['latency_ns']:>10.1f} "
                  f"{tp:>10} "
                  f"{r['L_A_cyc']:>4} {r['O_cyc']:>4} "
                  f"{r['regime']:<12}")
        print(f"    → sweet spot: {best['config']}  "
              f"latency={best['latency_ns']:.1f} ns  ({regime_short})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
