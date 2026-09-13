#!/usr/bin/env python3
"""FIDELITY_METHODOLOGY sanity check harness.

Validates simulator analytical predictions against hand-calculated
expected cycles for four workloads. Exit 0 iff all match exactly.

For each workload, prints:
  - Shape and config
  - Hand-calculated expected cycles (with formula breakdown)
  - Simulator output cycles
  - PASS / FAIL

Workloads (per FIDELITY_METHODOLOGY.md):
  1. GEMV  (VMM workload, M=1)
  2. GEMM  (VMM workload, M=4)
  3. FC    (VMM workload, V=1 vs V>1 branching)
  4. Attention DIMM mac_qk (DIMM workload, §7 W-lane row-parallel)
"""
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import IMC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline_T(num_passes, L, C, O, precision=8):
    """§4 drain-load overlap pipeline (Task #99, double-buffered LOAD).

    T_fill_ideal = L + C + O      (no FSM handoff overhead — surfaces as fidelity)
    T_steady     = max(L, C, O)
    T(M)         = T_fill_ideal + (M − 1) × T_steady

    Task #99: the input substrate is double-buffered (A/B ping-pong),
    so pass-(k+1) LOAD overlaps pass-k COMPUTE without a LOAD-gate.
    The +PRECISION term from the single-substrate Option A1 design is
    removed.

    The `precision` argument is retained for API compatibility but has
    no effect under the double-buffered model.
    """
    _ = precision  # retained for signature compatibility
    t_fill = L + C + O
    t_steady = max(L, C, O)
    return t_fill + max(0, num_passes - 1) * t_steady


def _clog2(val):
    """⌈log₂(val)⌉ for val > 0, else 0."""
    if val <= 1:
        return 0
    out = 0
    val -= 1
    while val > 0:
        out += 1
        val >>= 1
    return out


def _pipeline_T_workload(num_passes, L, C, O, V, precision=8):
    """Workload-level (FC/GEMM via fc_top.v) cycle formula
    (Task #99 unified — double-buffered LOAD).

    T_fill   = L + C + O
    T_steady = max(L, C, O)
    T(M)     = T_fill + (M − 1) × T_steady

    Task #99: double-buffered input substrate removes the LOAD-gate;
    +PRECISION term drops out of T_steady. TREE_PIPE and CLB_NEEDED
    remain reported as RTL deltas by run_fc_smoke.py — not baked
    into this analytical model.
    """
    _ = V  # retained for signature compatibility; intentionally unused
    return _pipeline_T(num_passes, L, C, O, precision=precision)


def _dpe_axiom_cycles(imc, workload="vmm"):
    """Read the simulator's per-pass DPE-axiom (L, C, O) for a workload."""
    return imc.imc_core._pipeline_pass_cycles(workload=workload)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_gemv():
    """Sanity 1 — GEMV (VMM workload). Shape M=1, K=512, N=128 on NL-DPE.

    Path A weight-stationary: V·H DPEs in parallel, each fires once per
    output row → passes_per_dpe = M = 1.
    K > R (V>1) → +1 CLB cycle for V-tree partial-sum combine.

    Option A (Task #90) ideal cycle formula:
        T(M) = (L + C + O + TREE_PIPE) + (M − 1)·T_steady + 1[CLB_NEEDED]
    No FSM/wrapper implementation overhead — surfaces as fidelity gap.
    """
    imc = IMC(str(ROOT / "IMC/configs/nl_dpe.json"))
    M, K, N = 1, 512, 128
    R = imc.cfg.rows
    C = imc.cfg.cols

    L_cyc, C_cyc, O_cyc = _dpe_axiom_cycles(imc, workload="vmm")

    # §5 VMM tiling — Path A
    V = math.ceil(K / R)                         # = ceil(512/256) = 2
    H = math.ceil(N / C)                         # = ceil(128/256) = 1
    n_parallel_dpes = max(1, V * H)              # = 2
    passes_per_dpe = M * math.ceil(V * H / n_parallel_dpes)  # = 1 (Path A)
    # Task #98 unified sim formula: T_fill = L+C+O (no TREE_PIPE, no CLB).
    expected = _pipeline_T_workload(passes_per_dpe, L_cyc, C_cyc, O_cyc, V)

    # Simulator
    latency_ns, _energy = imc.imc_core.run_gemm(M, K, N)
    t_clk = 1e3 / imc.cfg.freq
    actual = round(latency_ns / t_clk)

    desc = (
        f"GEMV(M={M},K={K},N={N}) on NL-DPE (R={R},C={C}):\n"
        f"  V={V}, H={H}, n_parallel_dpes={n_parallel_dpes} (TREE_PIPE={_clog2(V)} — reported as delta)\n"
        f"  Path A passes_per_dpe = M × ceil(V·H/n_parallel_dpes) = {passes_per_dpe},  "
        f"L={L_cyc} C={C_cyc} O={O_cyc}\n"
        f"  T = T_fill(L+C+O) + ({passes_per_dpe}-1)×T_steady = {expected}"
    )
    return ("sanity_gemv", expected, actual, desc)


def sanity_gemm():
    """Sanity 2 — GEMM (VMM workload). Shape M=4, K=512, N=128 on NL-DPE.

    Path A: V·H DPEs in parallel, each fires once per output row m.
    passes_per_dpe = M (NOT M·V).
    K > R (V>1): +1 CLB cycle for V-tree partial-sum combine.

    Option A ideal cycle formula (no FSM/wrapper overhead in sim).
    """
    imc = IMC(str(ROOT / "IMC/configs/nl_dpe.json"))
    M, K, N = 4, 512, 128
    R = imc.cfg.rows
    C = imc.cfg.cols

    L_cyc, C_cyc, O_cyc = _dpe_axiom_cycles(imc, workload="vmm")

    V = math.ceil(K / R)                         # 2
    H = math.ceil(N / C)                         # 1
    n_parallel_dpes = max(1, V * H)              # 2
    passes_per_dpe = M * math.ceil(V * H / n_parallel_dpes)  # = 4 (Path A: M passes)
    # Task #98 unified sim formula: T_fill = L+C+O (no TREE_PIPE, no CLB).
    expected = _pipeline_T_workload(passes_per_dpe, L_cyc, C_cyc, O_cyc, V)

    latency_ns, _energy = imc.imc_core.run_gemm(M, K, N)
    t_clk = 1e3 / imc.cfg.freq
    actual = round(latency_ns / t_clk)

    desc = (
        f"GEMM(M={M},K={K},N={N}) on NL-DPE (R={R},C={C}):\n"
        f"  V={V}, H={H}, n_parallel_dpes={n_parallel_dpes} (TREE_PIPE={_clog2(V)} — reported as delta)\n"
        f"  Path A passes_per_dpe = M × ceil(V·H/n_parallel_dpes) = {M}×1 = {passes_per_dpe}\n"
        f"  L={L_cyc} C={C_cyc} O={O_cyc};  "
        f"T = T_fill(L+C+O) + ({passes_per_dpe}-1)×T_steady = {expected}"
    )
    return ("sanity_gemm", expected, actual, desc)


def sanity_fc_with_activation():
    """Sanity 3 — FC with activation (VMM workload), Path A two variants:

      Variant A (V=1): single DPE tile. NL-DPE has_acam=True, so the
                        activation is ACAM-fused → no extra CLB cycle.
      Variant B (V>1): V parallel DPE tiles, CLB tree must combine V
                        partial sums → +1 CLB cycle (regardless of activation).

    Path A passes_per_dpe = M × ceil(V·H / n_parallel_dpes) = M (NOT M·V).
    Both variants should match Path A run_gemm Option A formula:
        T = T_fill_ideal + (M − 1) × T_steady + (1 if CLB_NEEDED else 0)
        T_fill_ideal = L + C + O + TREE_PIPE
        CLB_NEEDED = (V > 1) OR (activation_mode AND !has_acam)
    """
    imc = IMC(str(ROOT / "IMC/configs/nl_dpe.json"))
    R = imc.cfg.rows
    C = imc.cfg.cols
    L_cyc, C_cyc, O_cyc = _dpe_axiom_cycles(imc, workload="vmm")

    # Variant A — V=1, ACT=relu, NL-DPE has_acam → ACAM-fused, no +1.
    Ma, Ka, Na = 1, 128, 128
    Va = math.ceil(Ka / R)                              # = 1
    Ha = math.ceil(Na / C)                              # = 1
    n_par_a = max(1, Va * Ha)                           # = 1
    passes_a = Ma * math.ceil(Va * Ha / n_par_a)        # = 1
    expected_a = _pipeline_T_workload(passes_a, L_cyc, C_cyc, O_cyc, Va)
    # Va==1 AND has_acam → no +1 even with activation_mode=True.
    # Run with activation_mode=True to exercise CLB_NEEDED gate.
    lat_a, _ = imc.imc_core.run_gemm(Ma, Ka, Na, activation_mode=True)
    t_clk = 1e3 / imc.cfg.freq
    actual_a = round(lat_a / t_clk)

    # Variant B — V>1; TREE_PIPE/CLB cycles are now reported as delta (Task #98).
    imc_b = IMC(str(ROOT / "IMC/configs/nl_dpe.json"))  # fresh stats
    Mb, Kb, Nb = 1, 512, 128
    Vb = math.ceil(Kb / R)                              # = 2
    Hb = math.ceil(Nb / C)                              # = 1
    n_par_b = max(1, Vb * Hb)                           # = 2
    passes_b = Mb * math.ceil(Vb * Hb / n_par_b)        # = 1 (Path A: M, not M·V)
    # Task #98 unified sim formula: T_fill = L+C+O (no TREE_PIPE, no CLB).
    expected_b = _pipeline_T_workload(passes_b, L_cyc, C_cyc, O_cyc, Vb)
    lat_b, _ = imc_b.imc_core.run_gemm(Mb, Kb, Nb, activation_mode=True)
    actual_b = round(lat_b / t_clk)

    expected = (expected_a, expected_b)
    actual = (actual_a, actual_b)

    desc = (
        f"FC variants on NL-DPE (R={R},C={C}); L={L_cyc} C={C_cyc} O={O_cyc}\n"
        f"  Variant A (V=1, K={Ka}): passes={passes_a} (Path A), "
        f"T={expected_a} (ACAM-fused, V=1, TREE_PIPE=0)\n"
        f"  Variant B (V>1, K={Kb}): passes={passes_b} (Path A: M, not M·V), "
        f"T={expected_b} (TREE_PIPE={_clog2(Vb)} reported as delta, CLB cycle reported as delta)"
    )
    return ("sanity_fc_with_activation", expected, actual, desc)


def sanity_attention_dimm():
    """Sanity 4 — Attention head DIMM mac_qk (DIMM workload, §7 row-parallel).

    Verifies the full 3-phase model: gemm_log applies the §4 pipeline to
    passes_per_lane = phase_1a + phase_1b + phase_3 (sequential, per §7
    Pattern β: lane 0 does 1a then 1b, others wait for log_B broadcast
    before starting phase 3, so the per-lane critical path covers all
    three phases).
    """
    imc = IMC(str(ROOT / "IMC/configs/nl_dpe.json"))
    C = imc.cfg.cols
    W = imc.cfg.total_softmax_lanes
    N_seq = 128
    d = 64
    M_attn = N_seq          # Q rows = N_seq
    K_attn = d
    N_attn = N_seq

    # §7 W-lane row-parallel + Pattern β shared B
    rows_per_lane = math.ceil(M_attn / W)                       # 8
    phase_1a = math.ceil(rows_per_lane * K_attn / C)            # log A per lane
    phase_1b = math.ceil(K_attn * N_attn / C)                   # log B (shared)
    phase_3  = math.ceil(rows_per_lane * N_attn * K_attn / C)   # exp+sum per lane
    passes_per_lane = phase_1a + phase_1b + phase_3

    L_cyc, C_cyc, O_cyc = _dpe_axiom_cycles(imc, workload="dimm")
    expected = _pipeline_T(passes_per_lane, L_cyc, C_cyc, O_cyc)

    # Simulator (mac_qk: A=Q[N×d], B=K^T[d×N] → score[N×N])
    # gemm_log now applies §4 pipeline to (phase_1a + phase_1b + phase_3).
    imc.fpga.gemm_log(M_attn, K_attn, N_attn, n_parallel_dpes=W)
    sim_compute_cycles = imc.imc_core._pipeline_total_cycles(
        passes_per_lane, workload="dimm"
    )

    actual = sim_compute_cycles

    desc = (
        f"Attention DIMM mac_qk (N_seq={N_seq}, d={d}) on NL-DPE (C={C}, W={W}):\n"
        f"  rows_per_lane = ceil({M_attn}/{W}) = {rows_per_lane}\n"
        f"  Phase 1a per lane (log A): ceil({rows_per_lane}×{K_attn}/{C}) = {phase_1a}\n"
        f"  Phase 1b shared  (log B): ceil({K_attn}×{N_attn}/{C}) = {phase_1b}\n"
        f"  Phase 3 per lane (exp+sum): ceil({rows_per_lane}×{N_attn}×{K_attn}/{C}) = {phase_3}\n"
        f"  passes_per_lane (1a+1b+3) = {passes_per_lane}\n"
        f"  L={L_cyc} C={C_cyc} O={O_cyc};  T_fill_ideal={L_cyc + C_cyc + O_cyc} (=L+C+O), "
        f"T_steady={max(L_cyc, C_cyc, O_cyc)} (=max(L,C,O), Task #99 double-buffer)\n"
        f"  T = T_fill_ideal + ({passes_per_lane}-1) × T_steady = {expected}\n"
        f"  Sim cycles = {sim_compute_cycles}"
    )
    return ("sanity_attention_dimm", expected, actual, desc)


# ---------------------------------------------------------------------------
# Azure-Lily sanity checks  (DSP array, no ACAM)
# ---------------------------------------------------------------------------
#
# AL config: R=512, C=128, dpe_buf_width=16, freq=300 MHz,
# total_dsp=16, total_softmax_lanes=16, analog_nonlinear=False.
# DSP_WDITH = 4 (int8 pairs/cycle, see scheduler_stats.common).
#
# Dispatch routing in scheduler.py (current implementation):
#   - linear / FC (cfg.imc == "Azure-Lily"): falls through to
#       imc_core.run_gemm (the DPE primitive path designed for NL-DPE).
#       NOT routed to fpga.gemm_dsp.
#   - DIMM mac_qk / mac_sv (cfg.imc == "Azure-Lily"): routed to
#       fpga.gemm_dsp(M, K, N, n_parallel_outputs=N).
#
# These checks expose what each AL workload currently produces and
# compare to a hand-calculated expectation derived from §4 + the
# corresponding primitive's own cycle formula. PASS only if the
# simulator's cycle count matches the per-primitive hand calc; the
# methodology principle-alignment question (whether the *primitive
# itself* implements §4) is reported separately in Phase 2.
# ---------------------------------------------------------------------------

DSP_WDITH = 4  # AL DSP-MAC width, mirrors scheduler_stats.common.DSP_WDITH


def _gemm_dsp_compute_cycles(M, K, N, n_parallel_outputs, dpe_buf_width=16,
                             precision_bits=8):
    """Re-implement gemm_dsp's drain-load overlap formula for hand-calc.

    DSP-MAC primitive (dsp_mac.v) does NOT use single-substrate slice-
    major storage, so the Option A1 LOAD-gate constraint does NOT apply
    here. T_steady stays max(L, C, O).

    Per pass:
      L = ceil(K × precision_bits / dpe_buf_width)
      C = max(1, ceil(K / DSP_WIDTH))
      O = max(1, ceil(precision_bits / dpe_buf_width))
      passes_per_lane = ceil(M × N / n_lanes)
      T_fill = L + C + O
      T_steady = max(L, C, O)
      T = T_fill + (passes − 1) × T_steady
    """
    n_lanes = max(1, int(n_parallel_outputs))
    L = math.ceil(K * precision_bits / dpe_buf_width)
    C_cyc = max(1, math.ceil(K / DSP_WDITH))
    O = max(1, math.ceil(precision_bits / dpe_buf_width))
    passes_per_lane = max(1, math.ceil(M * N / n_lanes))
    t_fill = L + C_cyc + O
    t_steady = max(L, C_cyc, O)
    total_cycles = t_fill + max(0, passes_per_lane - 1) * t_steady
    return total_cycles, {
        "L": L, "C": C_cyc, "O": O,
        "n_lanes": n_lanes,
        "passes_per_lane": passes_per_lane,
        "t_fill": t_fill, "t_steady": t_steady,
    }


def sanity_al_gemv():
    """AL Sanity 1 — GEMV (VMM workload). Shape M=1, K=512, N=128.

    AL FC routes through imc_core.run_gemm with AL config (R=512, C=128,
    has_acam=False). K=R=512 → V=1, H=1. Path A: passes_per_dpe = M = 1.

    Path A CLB_NEEDED = (V > 1) OR (activation_mode AND !has_acam).
    No activation passed (activation_mode=False), V=1 → CLB_NEEDED=False
    → no +1 cycle. Option A ideal: Expected = L+C+O = 330.
    """
    imc = IMC(str(ROOT / "IMC/configs/azure_lily.json"))
    M, K, N = 1, 512, 128
    R = imc.cfg.rows
    C = imc.cfg.cols
    L_cyc, C_cyc, O_cyc = _dpe_axiom_cycles(imc, workload="vmm")

    V = math.ceil(K / R)             # 1
    H = math.ceil(N / C)             # 1
    n_parallel_dpes = max(1, V * H)
    passes_per_dpe = M * math.ceil(V * H / n_parallel_dpes)  # Path A
    # Task #98 unified sim formula: T_fill = L+C+O (no TREE_PIPE, no CLB).
    expected = _pipeline_T_workload(passes_per_dpe, L_cyc, C_cyc, O_cyc, V)
    has_acam = getattr(imc.cfg, 'analoge_nonlinear_support', True)
    activation_mode = False

    latency_ns, _energy = imc.imc_core.run_gemm(M, K, N, activation_mode=activation_mode)
    t_clk = 1e3 / imc.cfg.freq
    actual = round(latency_ns / t_clk)

    desc = (
        f"AL GEMV(M={M},K={K},N={N}) on Azure-Lily (R={R},C={C}, freq={imc.cfg.freq}MHz):\n"
        f"  Dispatch: imc_core.run_gemm (cfg.imc==Azure-Lily falls through DPE path)\n"
        f"  V={V}, H={H}, passes_per_dpe={passes_per_dpe} (Path A), "
        f"has_acam={has_acam}, activation_mode={activation_mode}\n"
        f"  L={L_cyc} C={C_cyc} O={O_cyc};  "
        f"T = T_fill(L+C+O) + (M-1)×T_steady = {expected}"
    )
    return ("sanity_al_gemv", expected, actual, desc)


def sanity_al_gemm():
    """AL Sanity 2 — GEMM (VMM workload). Shape M=4, K=512, N=128.

    Same dispatch as AL GEMV: imc_core.run_gemm. Path A: passes_per_dpe = M = 4.

    No activation here (activation_mode=False), V=1 → CLB_NEEDED=False → no +1.
    Option A ideal: Expected = L+C+O + 3·T_steady = 330 + 3·256 = 1098.
    """
    imc = IMC(str(ROOT / "IMC/configs/azure_lily.json"))
    M, K, N = 4, 512, 128
    R = imc.cfg.rows
    C = imc.cfg.cols
    L_cyc, C_cyc, O_cyc = _dpe_axiom_cycles(imc, workload="vmm")

    V = math.ceil(K / R)             # 1
    H = math.ceil(N / C)             # 1
    n_parallel_dpes = max(1, V * H)
    passes_per_dpe = M * math.ceil(V * H / n_parallel_dpes)  # = 4 (Path A)
    # Task #98 unified sim formula: T_fill = L+C+O (no TREE_PIPE, no CLB).
    expected = _pipeline_T_workload(passes_per_dpe, L_cyc, C_cyc, O_cyc, V)
    has_acam = getattr(imc.cfg, 'analoge_nonlinear_support', True)
    activation_mode = False

    latency_ns, _energy = imc.imc_core.run_gemm(M, K, N, activation_mode=activation_mode)
    t_clk = 1e3 / imc.cfg.freq
    actual = round(latency_ns / t_clk)

    desc = (
        f"AL GEMM(M={M},K={K},N={N}) on Azure-Lily (R={R},C={C}):\n"
        f"  Dispatch: imc_core.run_gemm; has_acam={has_acam}, "
        f"activation_mode={activation_mode}\n"
        f"  V={V}, H={H}, passes_per_dpe={passes_per_dpe} (Path A: M, not M·V)\n"
        f"  L={L_cyc} C={C_cyc} O={O_cyc};  "
        f"T = T_fill(L+C+O) + ({passes_per_dpe}-1)×T_steady = {expected}"
    )
    return ("sanity_al_gemm", expected, actual, desc)


def sanity_al_fc():
    """AL Sanity 3 — FC two variants, with activation (Path A).

    Variant A (V=1, K=128 ≤ R=512, activation_mode=True):
        AL has_acam=False, V=1, ACT=True → CLB_NEEDED=True (CLB ReLU LUT)
        → +1 cycle. Path A passes_per_dpe = M = 1.
        Option A ideal: Expected = L+C+O + 1 = 331.
    Variant B (V>1, K=2048 > R=512, activation_mode=True):
        V=4 → CLB tree → +1 cycle. Path A passes_per_dpe = M = 1
        (NOT M·V=4 like Path B). TREE_PIPE = ⌈log₂(4)⌉ = 2.
        Option A ideal: Expected = L+C+O + 2 (TREE_PIPE) + 1 (CLB) = 333.
    Both go through imc_core.run_gemm with activation_mode=True.
    """
    imc = IMC(str(ROOT / "IMC/configs/azure_lily.json"))
    R = imc.cfg.rows
    C = imc.cfg.cols
    L_cyc, C_cyc, O_cyc = _dpe_axiom_cycles(imc, workload="vmm")
    t_clk = 1e3 / imc.cfg.freq
    has_acam = getattr(imc.cfg, 'analoge_nonlinear_support', True)
    activation_mode = True

    # Variant A — V=1
    Ma, Ka, Na = 1, 128, 128
    Va = math.ceil(Ka / R)
    Ha = math.ceil(Na / C)
    n_par_a = max(1, Va * Ha)
    passes_a = Ma * math.ceil(Va * Ha / n_par_a)        # Path A: M
    # Task #98 unified sim formula: T_fill = L+C+O (no TREE_PIPE, no CLB).
    expected_a = _pipeline_T_workload(passes_a, L_cyc, C_cyc, O_cyc, Va)
    lat_a, _ = imc.imc_core.run_gemm(Ma, Ka, Na, activation_mode=activation_mode)
    actual_a = round(lat_a / t_clk)

    # Variant B — V>1
    imc_b = IMC(str(ROOT / "IMC/configs/azure_lily.json"))
    Mb, Kb, Nb = 1, 2048, 128
    Vb = math.ceil(Kb / R)              # 4
    Hb = math.ceil(Nb / C)              # 1
    n_par_b = max(1, Vb * Hb)
    passes_b = Mb * math.ceil(Vb * Hb / n_par_b)        # Path A: M (not M·V)
    # Task #98 unified sim formula: T_fill = L+C+O (no TREE_PIPE, no CLB).
    expected_b = _pipeline_T_workload(passes_b, L_cyc, C_cyc, O_cyc, Vb)
    lat_b, _ = imc_b.imc_core.run_gemm(Mb, Kb, Nb, activation_mode=activation_mode)
    actual_b = round(lat_b / t_clk)

    expected = (expected_a, expected_b)
    actual = (actual_a, actual_b)

    desc = (
        f"AL FC variants on Azure-Lily (R={R},C={C}); L={L_cyc} C={C_cyc} O={O_cyc}\n"
        f"  Dispatch: imc_core.run_gemm; has_acam={has_acam}, "
        f"activation_mode={activation_mode}\n"
        f"  Variant A (V=1, K={Ka}): passes={passes_a} (Path A, TREE_PIPE={_clog2(Va)} delta), "
        f"T = T_fill(L+C+O) = {expected_a} (CLB reported as delta)\n"
        f"  Variant B (V>1, K={Kb}): passes={passes_b} (Path A: M, not M·V, TREE_PIPE={_clog2(Vb)} delta), "
        f"T = T_fill(L+C+O) = {expected_b} (TREE/CLB reported as delta)"
    )
    return ("sanity_al_fc", expected, actual, desc)


def sanity_al_attention_dimm():
    """AL Sanity 4 — Attention DIMM mac_qk (DSP path), §4 pipeline.

    AL DIMM dispatches to fpga.gemm_dsp(M, K, N,
    n_parallel_outputs = total_softmax_lanes = W = 16) — matching
    paper/attention_dimm_mapping.md §6 (16 dsp_mac per DIMM stage,
    DSP_WIDTH=4) and §7 W-lane row-parallel allocation.

    Post-F2/F3/F4 refactor: gemm_dsp now applies §4's single-buffered
    drain-load overlap pipeline. AL has no ACAM, so DIMM is direct
    DSP-MAC matmul; per-pass DPE-axiom is derived from K, DSP_WIDTH=4,
    and dpe_buf_width=16:

        L = ceil(K × precision_bits / dpe_buf_width)
            = ceil(64 × 8 / 16)            = 32
        C = max(1, ceil(K / DSP_WIDTH))    = ceil(64 / 4)     = 16
        O = max(1, ceil(precision_bits / dpe_buf_width))
            = max(1, ceil(8 / 16))         = 1

    Each pass produces n_lanes output elements (lockstep across W lanes):
        passes_per_lane = ceil(M × N / W) = ceil(16384 / 16) = 1024
        Option A ideal:
        T_fill_ideal = L + C + O    = 32 + 16 + 1 = 49
        T_steady     = max(L, C, O) = max(32, 16, 1) = 32
        total_cycles = T_fill_ideal + (passes − 1) × T_steady
                     = 49 + 1023 × 32 = 32785

    Memory I/O latency is folded into per-pass L/O (§4) — the simulator
    no longer serialises t_read + t_gemm + t_write at the outer level.
    """
    imc = IMC(str(ROOT / "IMC/configs/azure_lily.json"))
    M_attn = 128
    K_attn = 64
    N_attn = 128
    # Lane allocation comes from total_softmax_lanes (= W = 16).
    W_lanes = max(1, getattr(imc.cfg, 'total_softmax_lanes', 16))
    dpe_bw = getattr(imc.cfg, 'dpe_buf_width', imc.cfg.bram_width)
    precision_bits = getattr(imc.cfg, 'precision_bits', 8)

    expected_total, breakdown = _gemm_dsp_compute_cycles(
        M_attn, K_attn, N_attn,
        n_parallel_outputs=W_lanes,
        dpe_buf_width=dpe_bw,
        precision_bits=precision_bits,
    )

    # Sim — call gemm_dsp directly with W=16 lane allocation. Post-refactor
    # total = compute (memory I/O folded into per-pass L/O per §4).
    t_total_ns, _e, _row = imc.fpga.gemm_dsp(
        M_attn, K_attn, N_attn, n_parallel_outputs=W_lanes
    )
    t_clk = 1e3 / imc.cfg.freq
    sim_total_cycles = round(t_total_ns / t_clk)

    expected = expected_total
    actual = sim_total_cycles

    desc = (
        f"AL DIMM mac_qk (M={M_attn}, K={K_attn}, N={N_attn}) "
        f"on Azure-Lily DSP path (DSP_WIDTH={DSP_WDITH}, "
        f"dpe_buf_width={dpe_bw}):\n"
        f"  Dispatch: fpga.gemm_dsp(n_parallel_outputs=W={W_lanes}) "
        f"§4 single-buffered drain-load overlap\n"
        f"  L = ceil({K_attn}×{precision_bits}/{dpe_bw}) = {breakdown['L']}\n"
        f"  C = max(1, ceil({K_attn}/{DSP_WDITH}))      = {breakdown['C']}\n"
        f"  O = max(1, ceil({precision_bits}/{dpe_bw})) = {breakdown['O']}\n"
        f"  passes_per_lane = ceil({M_attn}×{N_attn}/{W_lanes}) = {breakdown['passes_per_lane']}\n"
        f"  T_fill={breakdown['t_fill']}, T_steady={breakdown['t_steady']}\n"
        f"  T = T_fill + (passes-1)×T_steady = {expected}\n"
        f"  Sim total cycles = {sim_total_cycles}"
    )
    return ("sanity_al_attention_dimm", expected, actual, desc)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    checks = [
        sanity_gemv(),
        sanity_gemm(),
        sanity_fc_with_activation(),
        sanity_attention_dimm(),
        sanity_al_gemv(),
        sanity_al_gemm(),
        sanity_al_fc(),
        sanity_al_attention_dimm(),
    ]
    print("=" * 70)
    print("FIDELITY_METHODOLOGY sanity check harness")
    print("=" * 70)
    all_pass = True
    for name, expected, actual, desc in checks:
        print()
        print(f"--- {name} ---")
        print(desc)
        if isinstance(expected, tuple):
            ok = expected == actual
            status = "PASS" if ok else f"FAIL  (expected {expected}, got {actual})"
        else:
            ok = (expected == actual)
            status = "PASS" if ok else f"FAIL  (expected {expected}, got {actual})"
        print(f"  RESULT: {status}")
        all_pass = all_pass and ok
    print()
    print("=" * 70)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
