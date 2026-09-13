#!/usr/bin/env python3
r"""
acam_crossbar_snr.py
====================
ACAM Crossbar SNR and Precision Projection Tool.

Given crossbar rows (R), crossbar cols (C), and a target output
precision in bits, projects the absolute noise, signal, SNR, and
effective bit precision at the ACAM output. ACAM is fixed at 130
rows (Gray-coded → 256 output levels = 8-bit capability). Non-ideal
effects (IR drop, sneak path, correlated noise) are opt-in via CLI
flags.

================================================================
USAGE
================================================================
  | Flag             | Purpose                              | Default          |
  |------------------|--------------------------------------|------------------|
  | --R              | crossbar rows (required)              | —                |
  | --C              | crossbar cols (required)              | —                |
  | --bits           | target output precision               | 8                |
  | --ir-drop        | turn IR-drop model ON                 | OFF              |
  | --sneak          | turn sneak-path noise ON              | OFF              |
  | --correlated     | turn correlated common-mode noise ON  | OFF              |
  | --ir-alpha       | IR-drop coefficient (see derivation)   | 2.81e-7          |
  | --sneak-sigma0   | per-cell sneak σ in nA                | 0.45             |
  | --corr-delta     | per-row correlated σ in nA            | 0.10             |
  | --project        | print scaling table across R          | off              |
  | --formula        | print analytic scaling summary         | off              |

  Example invocations:
    python acam_crossbar_snr.py --R 128 --C 128 --bits 8
    python acam_crossbar_snr.py --R 1024 --C 128 --bits 8 --ir-drop --project
    python acam_crossbar_snr.py --R 512 --C 64 --bits 8 \
           --ir-drop --sneak --correlated --project --formula

================================================================
KEY ASSUMPTIONS — IDEAL vs NON-IDEAL
================================================================
  | Assumption                           | Ideal mode           | Non-ideal mode            |
  |--------------------------------------|----------------------|---------------------------|
  | Per-cell programming σ_G              | 0.4 μS (paper)       | Same                      |
  | Per-cell noise independence           | YES (CLT applies)    | Still YES for σ_ideal      |
  | Column current per cell               | g · V_read           | Reduced by IR drop         |
  | V_read across rows                    | Uniform              | Drops toward row tail      |
  | Sneak-path / OFF-cell contribution    | Ignored              | σ_off × V_read per cell    |
  | Common-mode noise (driver, thermal)   | Ignored              | δ × R per column (opt-in)  |
  | ACAM thresholds                       | Auto-scaled to I_col | Same                      |
  | Target precision                      | Capped at ACAM bits   | Same                      |

================================================================
NOISE MODEL — WHY EACH TERM SCALES DIFFERENTLY
================================================================
Every scaling law reflects the physical mechanism of the noise
source. Three distinct regimes:

  ----------------------------------------------------------------
  (A) σ_ideal(R) ∝ √R   — independent random errors
  ----------------------------------------------------------------
  Each cell's programming σ_G is independent from every other cell
  (process variation on mature ReRAM is not spatially correlated).
  Adding R independent random variables: **variances** add linearly,
  so standard deviation adds as √R. This is the central limit
  theorem and is the same reason that averaging N measurements
  reduces noise by √N.

      σ_ideal(R) = σ_G · V_read · √R

  Same structure applies to thermal noise and shot noise — any
  noise source that is independent per cell has √R scaling.

  ----------------------------------------------------------------
  (B) σ_sneak(R) ∝ √R   — independent OFF-state errors (same family)
  ----------------------------------------------------------------
  OFF-state cells each have a small residual conductance g_off with
  its own programming variance. Cumulative sneak noise is the sum
  of R independent contributions → √R, same reasoning as σ_ideal.

      σ_sneak(R) = σ_off_current · √R

  where σ_off_current is the per-cell sneak σ in nA. This term does
  NOT cause the SNR to degrade with R; it only raises the noise
  floor. It's included separately from σ_ideal so its magnitude can
  be toggled and calibrated independently.

  ----------------------------------------------------------------
  (C) IR-drop(R) ∝ R²   — deterministic geometric compounding
  ----------------------------------------------------------------
  This is NOT a random noise source — it's a systematic signal loss
  that scales with R². Derivation:

      Row-wire segment resistance per cell pitch:     r_wire   (Ω)
      Cells: R, each drawing approx V_read · g_avg
      At wire position k, current flowing from driver =
          (R - k + 1) · I_per_cell
      Cumulative voltage drop at cell k:
          ΔV(k) = r_wire · Σ_{p=1..k} (R - p + 1) · I_per_cell
                ≈ r_wire · I_per_cell · [k·R - k²/2]

      At the tail cell (k = R):
          ΔV(R) ≈ r_wire · I_per_cell · R²/2
      Average across all cells:
          ΔV_avg ≈ r_wire · I_per_cell · 3R²/8

      Fractional IR drop averaged over cells:
          f_IR_avg = ΔV_avg / V_read
                   = r_wire · g_avg · 3R²/8
                   = α · R²
      with  α = r_wire · g_avg · 3/8

  Both the *wire resistance* (∝ R) and the *wire current* (∝ R)
  grow with R, and their product appears in ΔV. That is why IR drop
  is the one effect that scales as R² rather than R or √R.

  This is a **signal bias**, not a stochastic σ. We apply it as a
  multiplicative loss on I_col_max.

  ----------------------------------------------------------------
  (D) σ_correlated(R) ∝ R   — common-mode / correlated noise (opt)
  ----------------------------------------------------------------
  If a noise source affects all rows coherently (driver noise,
  thermal gradient, supply-rail ripple), then the per-row errors
  are **correlated** rather than independent. Adding R perfectly
  correlated random variables: standard deviations add linearly,
  giving σ ∝ R.

      σ_correlated(R) = δ · R

  Off by default. Turn on via --correlated to model a worst-case
  correlated regime. δ is the per-row correlated σ contribution in
  nA (a phenomenological knob).

================================================================
HOW THE DEFAULT COEFFICIENTS WERE CHOSEN
================================================================
  σ_G = 0.4 μS, g_max = 150 μS, V_read = 0.3 V — from NL-DPE paper
  (arXiv:2511.13950) Table II and Figure 7 fits.

  Parameter ranges (for sensitivity study; override via CLI):

    Parameter | Conservative | Typical (default) | Pessimistic
    ----------|--------------|-------------------|-------------
    α (IR)    | 1e-7         | 3e-7               | 1e-6
    σ₀ (sneak)| 0.003        | 0.05               | 0.15
    δ (corr)  | 0.01         | 0.05               | 0.2

  α (IR-drop):
    Formula: α = r_wire · g_avg · 3/8
      r_wire = row-wire resistance per cell pitch (Ω/cell)
             0.005 (conservative, wide metal-1) to
             0.05  (aggressive, thin wire)
      g_avg  = 75 μS (half of g_max, average active cell)
    Default: r_wire = 0.01 Ω/cell, g_avg = 75 μS
           → α ≈ 3e-7   (f_IR ≈ 31 % at R=1024)

  σ_off_current (sneak per-cell σ):
    Formula: σ₀ ≈ σ_g_off · V_read
      σ_g_off = OFF-state conductance variance
        1T1R (transistor-isolated):  ≈ 0.001 μS → σ₀ ≈ 0.0003 nA
        2T2R or 1T1R w/ leakage:      ≈ 0.17  μS → σ₀ ≈ 0.05 nA
        0T1R (passive crossbar):      ≈ 0.5   μS → σ₀ ≈ 0.15 nA
    NL-DPE uses 2T2R ACAM cells → σ₀ ≈ 0.05 nA/cell is realistic.
    Default: 0.05 nA/cell → σ_sneak ≈ 1.6 nA at R=1024 (negligible
    vs σ_correlated).

  δ (correlated σ per row):
    Formula: δ ≈ σ_V_read · g_avg
      σ_V_read = driver voltage RMS noise
        Well-regulated LDO:     0.1 mV → δ ≈ 0.0075 nA/row
        Typical on-chip driver: 1 mV   → δ ≈ 0.075  nA/row
        Poor regulation:        10 mV  → δ ≈ 0.75   nA/row
    Default: 0.05 nA/row ≈ σ_V_read = 0.7 mV RMS
           → σ_corr ≈ 51 nA at R=1024 (≈ 0.28 × LSB)

  Note: these are derived from first principles + published IMC
  figures. For publication-quality numbers, calibrate α, σ₀, δ
  against SPICE-level simulation of the actual NL-DPE layout.

================================================================
ANALYTIC SCALING (ideal physics only)
================================================================
  I_col_max(R)  = R · g_max · V_read                    ∝  R
  σ_I(R)        = σ_G · V_read · √R                      ∝  √R
  LSB(R)        = I_col_max / 256                        ∝  R
  σ_I / LSB     = 256 · σ_G / (g_max · √R)                ∝  1/√R
  SNR(R)        = g_max · √R / σ_G                       ∝  √R
  eff_bits(R)   = log₂(SNR) = log₂(g_max/σ_G) + ½·log₂(R)
                ≈ 8.55 + ½ · log₂(R)   (capped at ACAM's 8 bits)

Take-away: in ideal physics, precision improves with R because
signal ∝ R while independent noise ∝ √R. Adding non-ideal terms
can flip this: IR drop (∝ R²) eats the signal, correlated noise
(∝ R) beats √R scaling, and sneak paths (∝ √R) just raise the
floor without changing the scaling.

Reference
---------
Zhao et al., "NL-DPE: An Analog In-memory Non-Linear Dot Product
Engine for Efficient CNN and LLM Inference", arXiv:2511.13950.
"""

import argparse
import math
from dataclasses import dataclass


# --------------------------------------------------------------------
# Device constants (from NL-DPE paper)
# --------------------------------------------------------------------
SIGMA_G     = 0.4     # μS, peak per-cell programming σ
G_MAX       = 150.0   # μS, max programmed conductance
V_READ      = 0.3     # V, bit-slice read voltage
ACAM_ROWS   = 130     # rows per ACAM unit (fixed)
ACAM_LEVELS = 256     # output levels (Gray-coded 8-bit)
ACAM_BITS   = int(math.log2(ACAM_LEVELS))  # = 8

# Default non-ideal coefficients (physically derived; see docstring §
# "HOW THE DEFAULT COEFFICIENTS WERE CHOSEN" for derivation)
DEFAULT_IR_ALPHA      = 3.0e-7    # Typical: r_wire ≈ 0.01 Ω/cell, g_avg = 75 μS
                                   # → f_IR ≈ 31 % at R=1024
DEFAULT_SNEAK_SIGMA0  = 0.05      # 2T2R RRAM cell: σ_g_off ≈ 0.01 μS · V_read
                                   # → σ_sneak ≈ 1.6 nA at R=1024 (negligible)
DEFAULT_CORR_DELTA    = 0.05      # V_read driver σ ≈ 0.7 mV RMS, g_avg = 75 μS
                                   # → σ_corr ≈ 51 nA at R=1024


# --------------------------------------------------------------------
# Report dataclass
# --------------------------------------------------------------------
@dataclass
class SNRReport:
    R: int
    C: int
    target_bits: int

    # Signal
    i_col_max_nA: float
    effective_i_col_nA: float  # after IR-drop bias

    # Noise terms (each independently)
    sigma_ideal_nA: float       # ∝ √R  (independent programming noise)
    sigma_sneak_nA: float       # ∝ √R  (independent OFF-state σ)
    sigma_corr_nA: float        # ∝ R   (correlated / common-mode)
    ir_drop_frac: float         # ∝ R²  (deterministic signal bias)
    sigma_total_nA: float       # quadrature sum of the three σ terms

    # Discretization + precision
    lsb_nA: float
    sigma_over_lsb: float
    snr_linear: float
    snr_dB: float
    effective_bits_raw: float
    effective_bits: float
    meets_target: bool


# --------------------------------------------------------------------
# Physical models — each term isolated so scaling is transparent
# --------------------------------------------------------------------
def signal_i_col_max(R: int) -> float:
    """Max column current in nA (all rows on, all cells at g_max)."""
    return R * G_MAX * V_READ


def sigma_ideal(R: int) -> float:
    """Independent programming noise — √R scaling (central limit)."""
    return SIGMA_G * V_READ * math.sqrt(R)


def sigma_sneak(R: int, sigma_off_current: float) -> float:
    """Independent OFF-state σ — √R scaling (same family as ideal)."""
    return sigma_off_current * math.sqrt(R)


def sigma_correlated(R: int, delta: float) -> float:
    """Correlated / common-mode noise — R scaling (perfectly correlated)."""
    return delta * R


def ir_drop_fraction(R: int, alpha: float) -> float:
    """Deterministic signal bias from wire IR drop — R² scaling."""
    f = alpha * R * R
    return f if f < 1.0 else 1.0


def lsb_nA(R: int) -> float:
    """ACAM bin size when thresholds are auto-scaled to I_col_max."""
    return signal_i_col_max(R) / ACAM_LEVELS


def post_reduction_effective_bits(R: int, K: int) -> float:
    """
    Effective bits at the FC layer output AFTER CLB reduction.

    For an FC layer with K total inputs tiled into V = ceil(K/R) passes,
    each pass quantizes its partial sum to ACAM_LEVELS (256). Per-pass
    quantization variance is LSB²/12 (uniform in one bin). Cumulative
    after V independent passes: σ² = V · LSB²/12. At fixed K, this
    scales as √R because LSB grows linearly with R.

      eff_bits ≈ log₂(K · max / σ_total)
              = 8 + ½·log₂(12 · K / R)

    Returns infinity if K <= R (single pass; no reduction).
    """
    if K <= R:
        # Single pass: ACAM cap is the limit, no V-averaging effect.
        # Effective bits is the ACAM cap plus the uniform-quant headroom.
        return ACAM_BITS + 0.5 * math.log2(12)
    return ACAM_BITS + 0.5 * math.log2(12.0 * K / R)


# --------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------
def compute(R: int, C: int, target_bits: int,
            ir_alpha: float = 0.0,
            sneak_sigma0: float = 0.0,
            corr_delta: float = 0.0) -> SNRReport:
    i_max = signal_i_col_max(R)
    s_ideal = sigma_ideal(R)
    s_sneak = sigma_sneak(R, sneak_sigma0)
    s_corr  = sigma_correlated(R, corr_delta)
    ir_f    = ir_drop_fraction(R, ir_alpha)

    # Quadrature sum of independent random terms
    s_total = math.sqrt(s_ideal**2 + s_sneak**2 + s_corr**2)

    eff_signal = i_max * (1.0 - ir_f)
    bin_size = lsb_nA(R)

    snr = eff_signal / s_total if s_total > 0 else float("inf")
    snr_dB = 20.0 * math.log10(snr) if snr > 0 else float("-inf")
    eff_raw = math.log2(snr) if snr > 0 else 0.0
    eff = min(eff_raw, ACAM_BITS)

    return SNRReport(
        R=R, C=C, target_bits=target_bits,
        i_col_max_nA=i_max,
        effective_i_col_nA=eff_signal,
        sigma_ideal_nA=s_ideal,
        sigma_sneak_nA=s_sneak,
        sigma_corr_nA=s_corr,
        ir_drop_frac=ir_f,
        sigma_total_nA=s_total,
        lsb_nA=bin_size,
        sigma_over_lsb=s_total / bin_size if bin_size > 0 else float("inf"),
        snr_linear=snr,
        snr_dB=snr_dB,
        effective_bits_raw=eff_raw,
        effective_bits=eff,
        meets_target=(eff >= target_bits),
    )


# --------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------
def _yes_no(cond: bool) -> str:
    return "YES" if cond else "NO"


def print_report(r: SNRReport, mode_flags: list) -> None:
    title = f"Crossbar SNR Report — R={r.R}, C={r.C}, target={r.target_bits}-bit"
    mode  = "(" + ", ".join(mode_flags) + ")" if mode_flags else "(ideal physics only)"
    bar = "=" * max(len(title), len(mode))
    print("\n" + bar)
    print(title)
    print(mode)
    print(bar)

    print(f"  Signal")
    print(f"    I_col_max                   : {r.i_col_max_nA:>12,.1f} nA")
    if r.ir_drop_frac > 0:
        print(f"    I_col effective (post-IR)   : {r.effective_i_col_nA:>12,.1f} nA")
        print(f"    IR drop (signal bias, R²)   : {r.ir_drop_frac*100:>11.2f} %")

    print(f"  Noise (by source)")
    print(f"    σ ideal (indep., √R)         : {r.sigma_ideal_nA:>12.4f} nA")
    if r.sigma_sneak_nA > 0:
        print(f"    σ sneak (indep., √R)         : {r.sigma_sneak_nA:>12.4f} nA")
    if r.sigma_corr_nA > 0:
        print(f"    σ correlated (common-mode, R): {r.sigma_corr_nA:>12.4f} nA")
    print(f"    σ total (quadrature sum)     : {r.sigma_total_nA:>12.4f} nA")

    print(f"  Discretization (ACAM auto-scaled)")
    print(f"    LSB                           : {r.lsb_nA:>12.2f} nA")
    print(f"    σ / LSB                       : {r.sigma_over_lsb:>12.4f}")

    print(f"  Precision (at ACAM output)")
    print(f"    SNR                           : {r.snr_linear:>12,.1f}  ({r.snr_dB:.2f} dB)")
    print(f"    Effective bits (raw)          : {r.effective_bits_raw:>12.2f}")
    print(f"    Effective bits (ACAM-capped)  : {r.effective_bits:>12.2f}  (ACAM cap = {ACAM_BITS})")
    print(f"    Meets {r.target_bits}-bit target         : {_yes_no(r.meets_target):>12s}")


def print_scaling_table(C: int, target_bits: int,
                         ir_alpha: float, sneak_sigma0: float,
                         corr_delta: float) -> None:
    print("\nScaling projection across R  "
          f"(C={C}, target={target_bits}-bit)")
    header = (f"{'R':>6}  {'I_col_max(nA)':>14}  {'σ_total(nA)':>12}  "
              f"{'σ/LSB':>9}  {'SNR(dB)':>9}  {'eff_bits':>9}  {'meets':>6}")
    print(header)
    print("-" * len(header))
    for R in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        r = compute(R, C, target_bits, ir_alpha, sneak_sigma0, corr_delta)
        print(f"{R:>6}  {r.i_col_max_nA:>14,.0f}  {r.sigma_total_nA:>12.3f}  "
              f"{r.sigma_over_lsb:>9.4f}  {r.snr_dB:>9.2f}  "
              f"{r.effective_bits:>9.2f}  {_yes_no(r.meets_target):>6s}")


def print_quantization_table(K: int, target_bits: int) -> None:
    """
    Post-reduction precision projection across R for a fixed FC dimension K.
    This is the QUANTIZATION-ONLY analysis (assumes clean analog).
    """
    print(f"\nPost-reduction effective bits  (FC layer, K={K} inputs)")
    print("(quantization noise only; analog assumed ideal)")
    header = (f"{'R':>6}  {'V (passes)':>11}  {'per-pass LSB':>13}  "
              f"{'cumulative σ':>13}  {'eff_bits':>9}  {'meets':>6}")
    print(header)
    print("-" * len(header))
    for R in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if R > K:
            continue
        V = math.ceil(K / R)
        lsb_per_pass = R * 1.0 / ACAM_LEVELS  # in normalized math units
        cum_sigma = math.sqrt(V) * lsb_per_pass / math.sqrt(12)
        eff = post_reduction_effective_bits(R, K)
        meets = "YES" if eff >= target_bits else "NO"
        print(f"{R:>6}  {V:>11,}  {lsb_per_pass:>13.4f}  "
              f"{cum_sigma:>13.4f}  {eff:>9.2f}  {meets:>6s}")


def print_formula_summary() -> None:
    print("\nScaling laws by noise source")
    print("-" * 70)
    print(f"  σ_G     = {SIGMA_G} μS   (per-cell programming σ)")
    print(f"  g_max   = {G_MAX} μS  (max conductance)")
    print(f"  V_read  = {V_READ} V   (bit-slice voltage)")
    print(f"  ACAM    = {ACAM_ROWS} rows → {ACAM_LEVELS} levels ({ACAM_BITS}-bit)")
    print()
    print("  Term           | Scaling | Physical reason")
    print("  ---------------|---------|--------------------------------------")
    print("  σ_ideal         | √R      | independent per-cell errors (CLT)")
    print("  σ_sneak         | √R      | independent OFF-state σ (CLT)")
    print("  σ_correlated    |  R      | common-mode / correlated noise")
    print("  IR drop         |  R²     | wire R × wire I, both ∝ R")
    print()
    print("Analytic ideal-physics scaling")
    print("-" * 70)
    print("  I_col_max(R) = R · g_max · V_read           ∝  R")
    print("  σ_I(R)       = σ_G · V_read · √R             ∝  √R")
    print("  LSB(R)       = I_col_max / 256               ∝  R")
    print("  σ_I / LSB    = 256 · σ_G / (g_max · √R)       ∝  1/√R")
    print("  SNR(R)       = g_max · √R / σ_G              ∝  √R")
    print(f"  eff_bits(R) = log₂(g_max / σ_G) + ½ log₂(R)  ≈  "
          f"{math.log2(G_MAX/SIGMA_G):.2f} + ½ log₂(R)")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="ACAM crossbar SNR and precision projection tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--R", type=int, required=True,
                   help="crossbar rows (input dimension)")
    p.add_argument("--C", type=int, required=True,
                   help="crossbar cols (output dimension)")
    p.add_argument("--bits", type=int, default=8,
                   help="target output precision in bits (default 8)")

    p.add_argument("--ir-drop", action="store_true",
                   help="include IR-drop bias (∝ R²)")
    p.add_argument("--sneak", action="store_true",
                   help="include sneak-path σ (∝ √R)")
    p.add_argument("--correlated", action="store_true",
                   help="include correlated common-mode σ (∝ R)")

    p.add_argument("--ir-alpha", type=float, default=DEFAULT_IR_ALPHA,
                   help=f"IR-drop α, f_IR = α·R² "
                        f"(default {DEFAULT_IR_ALPHA:.2e} → ~29%% at R=1024)")
    p.add_argument("--sneak-sigma0", type=float, default=DEFAULT_SNEAK_SIGMA0,
                   help=f"per-cell sneak σ in nA, σ_sneak = σ₀·√R "
                        f"(default {DEFAULT_SNEAK_SIGMA0} nA/cell)")
    p.add_argument("--corr-delta", type=float, default=DEFAULT_CORR_DELTA,
                   help=f"per-row correlated δ in nA, σ_corr = δ·R "
                        f"(default {DEFAULT_CORR_DELTA} nA/row)")

    p.add_argument("--project", action="store_true",
                   help="print scaling table across R values")
    p.add_argument("--formula", action="store_true",
                   help="print scaling formula summary")
    p.add_argument("--K", type=int, default=None,
                   help="FC layer input dimension; enables post-reduction "
                        "quantization analysis (cumulative quant σ scales as √R at fixed K)")
    args = p.parse_args()

    ir_a  = args.ir_alpha     if args.ir_drop    else 0.0
    sn_0  = args.sneak_sigma0 if args.sneak      else 0.0
    co_d  = args.corr_delta   if args.correlated else 0.0

    mode_flags = []
    if args.ir_drop:    mode_flags.append("IR drop ON")
    if args.sneak:      mode_flags.append("sneak ON")
    if args.correlated: mode_flags.append("correlated ON")

    r = compute(args.R, args.C, args.bits, ir_a, sn_0, co_d)
    print_report(r, mode_flags)

    if args.project:
        print_scaling_table(args.C, args.bits, ir_a, sn_0, co_d)

    if args.K is not None:
        print_quantization_table(args.K, args.bits)

    if args.formula:
        print_formula_summary()


if __name__ == "__main__":
    main()
