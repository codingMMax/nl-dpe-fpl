#!/usr/bin/env python3
"""Regenerate fc_verification/expected_cycles.json from the live simulator.

Phase 1 (Regime B, Layout A): wires directly to
azurelily.IMC.peripherals.fpga_fabric.FPGAFabric and imc_core.IMCCore so
the numbers match whatever the sim currently computes. The --freeze flag
keeps the Phase-J frozen baseline as an escape hatch.

Stage → sim call mapping for the NL-DPE DIMM top (N=128, d=64):
  score   = gemm_log(M=N, K=d, N=N, n_parallel_dpes=N) compute-cycles only
  softmax = _run_softmax_exp row-rate + _run_softmax_norm row-rate (W-aware)
  wsum    = gemm_log(M=N, K=N, N=d, n_parallel_dpes=d) compute-cycles only
  e2e     = t_fill + (S-1)·t_steady + t_drain  (scheduler.py §_run_attention_pipeline)

Azure-Lily stages remain null (gemm_dsp path not in Phase 1 scope).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "fc_verification" / "expected_cycles.json"

# Phase-J frozen baseline. Source of truth for Regime A, Layout A.
FROZEN_BASELINE = {
    "source": "Phase J measured (Regime A sim, Layout A RTL, N=128 d=64 W=16)",
    "regime": "A",
    "layout": "A",
    "updated": "2026-04-19",
    "note": "Regenerate via gen_expected_cycles.py after any sim-model change.",
    "tolerance_cycles": {"per_stage": 3, "e2e": 5},
    "configs": {
        "nldpe_dimm_top_d64_c128": {
            "params": {"N": 128, "d": 64, "C": 128, "W": 16},
            "stages": {"score": 267, "softmax": 26, "wsum": 257, "e2e": 550},
        },
        "azurelily_dimm_top_d64_c128": {
            "params": {"N": 128, "d": 64, "C": 128, "W": 16, "total_dsp": 16},
            "stages": {"mac_qk": None, "e2e": None},
            "todo": "Populate from gemm_dsp() path in a future phase.",
        },
    },
}


def _reconstruct_cycles_per_pass(cfg, imc_core, K):
    """Replicate gemm_log's internal cycles_per_pass / cycles_per_load
    analytically. Used so we can report the single-row "stage" cycles
    (load + drain, i.e. Layout A cycles_per_pass·cols_per_dpe) which is
    what the RTL latency TB measures.
    """
    K_id_lat = max(1, cfg.cols // K) if (imc_core and cfg.analoge_nonlinear_support) else 1
    dpe_passes_per_elem = math.ceil(K / cfg.cols)
    t_clk_ns = 1e3 / cfg.freq
    dpe_bw = getattr(cfg, 'dpe_buf_width', cfg.bram_width)
    sram_read_lat = getattr(cfg, 'sram_read_latency', 2)
    read_cycles = math.ceil(cfg.cols * 8 / dpe_bw)
    feed_cycles = read_cycles + sram_read_lat * K_id_lat
    output_cycles = math.ceil(cfg.cols * 8 / dpe_bw)
    core_ns = imc_core._core_bit_pipeline_row_latency()
    compute_cycles = max(1, math.ceil((core_ns - output_cycles * t_clk_ns) / t_clk_ns))
    reduce_cycles = math.ceil(math.log2(max(2, K)))
    effective_output_cycles = max(output_cycles, reduce_cycles)
    cycles_per_load = feed_cycles + dpe_passes_per_elem * compute_cycles
    cycles_per_drain = dpe_passes_per_elem * effective_output_cycles
    cycles_per_pass = cycles_per_load + cycles_per_drain
    return {
        "K_id_lat": K_id_lat,
        "cycles_per_load": cycles_per_load,
        "cycles_per_drain": cycles_per_drain,
        "cycles_per_pass": cycles_per_pass,
    }


def _compute_sim_per_row(fpga, imc_core, M, K, N, n_parallel_dpes):
    """Call gemm_log and return per-row timing details.

    RTL latency TB reports single-row stage cycles = time for ONE row to
    transit the DPE lane from LOAD-start to DRAIN-end. In Regime B
    (Layout A) that is `cycles_per_pass · cols_per_dpe` (load + fires +
    drain, not the steady-state interval). We therefore report this
    "single-row" value in the expected_cycles JSON rather than
    `per_row_compute` (which under Regime B reflects the steady interval
    = cycles_per_load · cols_per_dpe).
    """
    cfg = fpga.cfg
    t_total_ns, _, row_timing = fpga.gemm_log(M, K, N, n_parallel_dpes=n_parallel_dpes)
    pass_info = _reconstruct_cycles_per_pass(cfg, imc_core, K)
    K_id_lat = pass_info["K_id_lat"]
    effective_parallel = n_parallel_dpes * K_id_lat
    cols_per_dpe = math.ceil(N / effective_parallel)
    # Single-row compute cycles (RTL-equivalent): load + drain per cols_per_dpe
    single_row_compute_cyc = (
        pass_info["cycles_per_load"] * cols_per_dpe
        + pass_info["cycles_per_drain"]
    )
    # Regime B steady-state per-row interval (Layout A): cycles_per_load · cols_per_dpe
    per_row_steady_cyc = pass_info["cycles_per_load"] * cols_per_dpe
    return {
        "single_row_compute_cyc": single_row_compute_cyc,
        "per_row_steady_cyc": per_row_steady_cyc,
        "cycles_per_pass": pass_info["cycles_per_pass"],
        "cycles_per_load": pass_info["cycles_per_load"],
        "cycles_per_drain": pass_info["cycles_per_drain"],
        "cols_per_dpe": cols_per_dpe,
        "t_total_ns": t_total_ns,
        "row_timing": row_timing,
    }


def _cycles(ns, freq_MHz):
    return int(round(ns * freq_MHz / 1e3))


def refresh_from_sim() -> dict | None:
    """Rebuild the expected-cycles dict by calling the live simulator.

    Returns None (triggering the frozen baseline fallback) only if import
    or instantiation fails — for any live-sim numerical result the dict
    is populated from the sim.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "azurelily" / "IMC"))
        from simulator import IMC  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        print(f"[expected-cycles] sim import failed ({exc}); using frozen baseline",
              file=sys.stderr)
        return None

    # Use the top-config as that's the one the RTL TB's latency is aligned to
    # (cols=128, freq=90.05 MHz, matches nldpe_dimm_top_d64_c128.v physical
    # Fmax and geometry). Falls back to the default nl_dpe.json if the
    # top-config is not present.
    top_config_path = (
        REPO_ROOT / "fc_verification" / "results"
        / "nldpe_dimm_top_imc_config.json"
    )
    default_config_path = REPO_ROOT / "azurelily" / "IMC" / "configs" / "nl_dpe.json"
    config_path = top_config_path if top_config_path.exists() else default_config_path
    try:
        imc = IMC(str(config_path))
        cfg = imc.cfg
        fpga = imc.fpga
        imc_core = imc.imc_core
        print(f"[expected-cycles] using config: {config_path.name} "
              f"(cols={cfg.cols}, freq={cfg.freq:.2f} MHz, "
              f"total_dimm_dpes={getattr(cfg, 'total_dimm_dpes', 1)})",
              file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[expected-cycles] sim init failed ({exc}); using frozen baseline",
              file=sys.stderr)
        return None

    # Config for the NL-DPE DIMM top (see expected_cycles.json nldpe_dimm_top_d64_c128).
    N = 128
    d = 64
    freq = cfg.freq  # MHz
    t_clk_ns = 1e3 / freq
    # Parallel DPEs for mac_qk / mac_sv — matches scheduler.py line 183:
    # n_parallel_dpes = max(1, cfg.total_dimm_dpes).
    n_parallel = max(1, getattr(cfg, 'total_dimm_dpes', 1))
    W_softmax = max(1, getattr(cfg, 'total_softmax_lanes',
                               getattr(cfg, 'total_dimm_dpes', 1)))

    # --- Score (QK^T): gemm_log(M=N, K=d, N=N, n_parallel_dpes=total_dimm_dpes) ---
    # Single-row compute cycles = what the RTL TB's "Score stage" probe
    # reports (time for one row to transit load + drain through the lane).
    score = _compute_sim_per_row(
        fpga, imc_core, M=N, K=d, N=N, n_parallel_dpes=n_parallel
    )
    score_cycles = score["single_row_compute_cyc"]

    # --- Wsum (Score×V): gemm_log(M=N, K=N, N=d, n_parallel_dpes=total_dimm_dpes) ---
    wsum = _compute_sim_per_row(
        fpga, imc_core, M=N, K=N, N=d, n_parallel_dpes=n_parallel
    )
    wsum_cycles = wsum["single_row_compute_cyc"]

    # --- Softmax row-rate ---
    # Replicates _run_softmax_exp + _run_softmax_norm per-row cycles.
    # cols = N for the N×N softmax (each row has N scores).
    cols = N

    # Exp per-row: dimm_nonlinear(cols, op="exp") / W_softmax + log2(W) cycles
    exp_lat_per_row_ns, _ = imc_core.dimm_nonlinear(cols, op="exp", record_breakdown=False)
    reduction_ns = math.ceil(math.log2(max(2, W_softmax))) * t_clk_ns
    exp_row_ns = exp_lat_per_row_ns / W_softmax + reduction_ns

    # Norm per-row: fpga.norm_fpga(cols) / W_softmax + reduction.
    cols_per_lane_norm = max(1, cols // W_softmax)
    norm_mem_read_row_ns = fpga.memory.latency(cols_per_lane_norm)
    norm_mem_write_row_ns = fpga.memory.latency(cols_per_lane_norm, read=False)
    if cfg.analoge_nonlinear_support and getattr(cfg, 'log_softmax_fusion', False):
        norm_lat_ns, _, _, _, _ = fpga.norm_fpga(cols)
        norm_row_ns = norm_lat_ns / W_softmax + reduction_ns
    else:
        norm_row_ns = 0.0

    # The RTL's "Softmax stage" single-row cycles reflect the exp+norm per-row
    # combined compute (the bottleneck cycle within the softmax lane). Take
    # the larger of the two compute terms so we match the scheduler's
    # per_row_steady for the softmax sub-pipeline.
    softmax_row_ns = max(exp_row_ns, norm_row_ns)
    softmax_cycles = _cycles(softmax_row_ns, freq)

    # --- End-to-end for ONE output row through 3 stages ---
    # The RTL TB measures `end_cyc - feed_qk_cyc` — time from FSM force to
    # the first row reaching WS_OUTPUT. That is a single row cascading
    # through score → softmax → wsum (serial stage composition per row).
    # e2e = score + softmax + wsum.
    e2e_cycles = score_cycles + softmax_cycles + wsum_cycles

    # --- Build the JSON payload ---
    today = _dt.date.today().isoformat()
    data = {
        "source": f"Live sim refresh (Regime B, Layout A) on {today}",
        "regime": "B",
        "layout": "A",
        "updated": today,
        "note": "Regenerate via gen_expected_cycles.py after any sim-model change.",
        "tolerance_cycles": {"per_stage": 3, "e2e": 5},
        "configs": {
            "nldpe_dimm_top_d64_c128": {
                "params": {"N": N, "d": d, "C": 128, "W": W_softmax},
                "stages": {
                    "score": score_cycles,
                    "softmax": softmax_cycles,
                    "wsum": wsum_cycles,
                    "e2e": e2e_cycles,
                },
            },
            "azurelily_dimm_top_d64_c128": {
                "params": {"N": N, "d": d, "C": 128, "W": W_softmax, "total_dsp": 16},
                "stages": {"mac_qk": None, "e2e": None},
                "todo": "Populate from gemm_dsp() path in a future phase.",
            },
        },
    }
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true",
                        help="Skip sim refresh; force write of the frozen baseline.")
    args = parser.parse_args()

    data = None if args.freeze else refresh_from_sim()
    if data is None:
        data = FROZEN_BASELINE

    OUT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[expected-cycles] wrote {OUT_PATH} (regime={data['regime']}, layout={data['layout']})")
    stages = data["configs"]["nldpe_dimm_top_d64_c128"]["stages"]
    print(f"[expected-cycles] nldpe_dimm_top stages: {stages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
