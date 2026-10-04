#!/usr/bin/env python3
"""softmax_online_ref.py — NumPy value reference for the NL-DPE online softmax.

Role
----
Plain math, no state, no time. Companion to `v2/spec/softmax_online.md`.
Value paths:

  exact    — block-local max + rescale in **full precision** (float64, true
             exponential).  A reference witness only (not the hardware
             contract): softmax is shift-invariant, so the streamed block
             recurrence reconstructs the exact dense softmax
             `p_j = exp(s_j - m) / sum_j exp(s_j - m)`; it is invariant to `Bkv`.

  model    — the **ACAM** streamed realization; the contract the behavior
             simulator reproduces bit-exactly.  Same op mapping as
             `softmax_ref`: block-local max, `ACAM_EXP` elementwise weights,
             deferred-alpha factor `ACAM_EXP(m_b - m)`, `lq = L >> log2 S` and
             `ACAM_LOG(lq)`.  No `np.exp`/`np.log`, no float division.  It is
             exactly `softmax_ref` at `Bkv = 1` and `Bkv = S` (factor 1 /
             single block) and differs for intermediate blocks (the ACAM is
             not shift-invariant, so the deferred-alpha factor does not
             reconstruct the global-max normalizer).

  global   — the shipped full-row operator (`softmax_ref.softmax_stage_values`):
             global max, int8 mean-shifted log — the `Bkv = S` special case.

Output convention (matches `softmax.md` §1): `out = s - m - ACAM_LOG(lq)`,
`out8 = clamp(out, -128, 127)`.  `exp(out) = S * p`.

Run:  python3 v2/oracle/softmax_online_ref.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sibling oracle modules importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_ref as ref  # noqa: E402
import softmax_ref as sref  # noqa: E402


def _blocks(S: int, Bkv: int) -> list[tuple[int, int]]:
    assert Bkv >= 1 and S % Bkv == 0, "Bkv must divide S"
    return [(b * Bkv, (b + 1) * Bkv) for b in range(S // Bkv)]


# ---------------------------------------------------------------------------
# Textbook dual witness (no blocking) — `exact` must match it.
# ---------------------------------------------------------------------------
def textbook_softmax(scores: np.ndarray) -> dict:
    """Direct per-row softmax in the operator's mean-shifted log convention."""
    scores = np.asarray(scores, dtype=np.int32)
    S = scores.shape[1]
    m = scores.max(axis=1)
    e = np.exp((scores - m[:, None]).astype(np.float64))
    l = e.sum(axis=1)
    p = e / l[:, None]
    log_p = scores.astype(np.float64) - m[:, None] - np.log(l)[:, None]
    out = log_p + np.log(S)                       # mean-shifted operator form
    return {"m": m, "l": l, "p": p, "log_p": log_p, "out_exact": out,
            "out8_exact": np.clip(np.round(out), -128, 127).astype(np.int8)}


# ---------------------------------------------------------------------------
# exact — the normative contract (float64, true exp, shift-invariant)
# ---------------------------------------------------------------------------
def softmax_online_exact(scores: np.ndarray, Bkv: int) -> dict:
    """Block-local max + rescale, full precision. Exact dense softmax.

    Returns dict:
      m          int32   [S]      exact row max (= max of block maxima)
      l          float64 [S]      exact normalizer sum_j exp(s_j - m)
      blk_max    int32   [B,S]    per-block row maxima
      blk_Sp     float64 [B,S]    per-block partials sum_{j in b} exp(s_j-m_b)
      log_p      float64 [S,S]    log-softmax s - m - log l
      p          float64 [S,S]    exact softmax (rows sum to 1)
      out_exact  float64 [S,S]    operator convention s - m - log(l/S)
      out8_exact int8    [S,S]    clamp(round(out_exact))
    """
    scores = np.asarray(scores, dtype=np.int32)
    assert scores.ndim == 2 and scores.shape[0] == scores.shape[1]
    S = scores.shape[1]
    assert S & (S - 1) == 0, "S must be a power of two"
    blocks = _blocks(S, Bkv)
    B = len(blocks)

    m = scores.max(axis=1).astype(np.int32)
    blk_max = np.empty((B, S), dtype=np.int32)
    blk_Sp = np.empty((B, S), dtype=np.float64)
    for b, (lo, hi) in enumerate(blocks):
        tile = scores[:, lo:hi]
        mb = tile.max(axis=1)
        blk_max[b] = mb
        blk_Sp[b] = np.exp((tile - mb[:, None]).astype(np.float64)).sum(axis=1)

    # Deferred-alpha combine: l = sum_b exp(m_b - m) * Sp_b  (exact).
    l = np.zeros(S, dtype=np.float64)
    for b in range(B):
        l += np.exp((blk_max[b] - m).astype(np.float64)) * blk_Sp[b]

    log_l = np.log(l)
    log_p = scores.astype(np.float64) - m[:, None] - log_l[:, None]
    p = np.exp(log_p)
    out_exact = log_p + np.log(S)
    out8_exact = np.clip(np.round(out_exact), -128, 127).astype(np.int8)
    return {"m": m, "l": l, "blk_max": blk_max, "blk_Sp": blk_Sp,
            "log_p": log_p, "p": p, "out_exact": out_exact,
            "out8_exact": out8_exact}


# ---------------------------------------------------------------------------
# model — the ACAM streamed realization (sim must match bit-exactly)
# ---------------------------------------------------------------------------
def softmax_online_model(scores: np.ndarray, Bkv: int) -> dict:
    """ACAM streamed realization — same op mapping as `softmax_ref`.

    Block-local max + deferred-alpha, all transcendentals via the ACAM forms
    (no `np.exp`/`np.log`, no float division):

      per block b :  m_b = max_j s_b,j
                     eb_j = ACAM_EXP(max(s_b,j - m_b, -128))        (unsigned)
                     Sp_b = sum_j eb_j
      combine     :  L = sum_b ACAM_EXP(max(m_b - m, -128)) * Sp_b  (deferred alpha)
      normalizer  :  lq = min(L >> log2 S, 127);  ls = ACAM_LOG(lq)
      output      :  out8 = clamp(s - m - ls, -128, 127)

    Returns dict:
      m        int32   [S]
      blk_max  int32   [B,S]
      blk_Sp   int64   [B,S]   sum_j unsigned(ACAM_EXP(max(s-m_b, -128)))
      factor   int64   [B,S]   unsigned(ACAM_EXP(max(m_b-m, -128)))
      L        int64   [S]     sum_b factor_b * blk_Sp_b
      lq       int32   [S]     min(L >> log2 S, 127)
      ls       int32   [S]     ACAM_LOG(lq) = trunc8(lq - 1)
      out8     int8    [S,S]   clamp(s - m - ls)
    """
    scores = np.asarray(scores, dtype=np.int32)
    assert scores.ndim == 2 and scores.shape[0] == scores.shape[1]
    S = scores.shape[1]
    assert S & (S - 1) == 0, "S must be a power of two"
    log2_S = S.bit_length() - 1
    blocks = _blocks(S, Bkv)
    B = len(blocks)

    m = scores.max(axis=1).astype(np.int32)
    blk_max = np.empty((B, S), dtype=np.int32)
    blk_Sp = np.empty((B, S), dtype=np.int64)
    for b, (lo, hi) in enumerate(blocks):
        tile = scores[:, lo:hi]
        mb = tile.max(axis=1)
        blk_max[b] = mb
        d = np.maximum(tile - mb[:, None], -128)
        eb = ref.acam_transform(d, ref.MODE_EXP).view(np.uint8).astype(np.int64)
        blk_Sp[b] = eb.sum(axis=1)

    factor = np.empty((B, S), dtype=np.int64)
    L = np.zeros(S, dtype=np.int64)
    for b in range(B):
        f = ref.acam_transform(np.clip(blk_max[b] - m, -128, 127),
                               ref.MODE_EXP).view(np.uint8).astype(np.int64)
        factor[b] = f
        L += f * blk_Sp[b]

    lq = np.minimum(L >> log2_S, 127).astype(np.int32)
    ls = ref.acam_transform(lq, ref.MODE_LOG).astype(np.int32)
    out8 = np.clip(scores - m[:, None] - ls[:, None], -128, 127).astype(np.int8)
    return {"m": m, "blk_max": blk_max, "blk_Sp": blk_Sp, "factor": factor,
            "L": L, "lq": lq, "ls": ls, "out8": out8}


# ---------------------------------------------------------------------------
# global / regular-exact references + distribution helpers
# ---------------------------------------------------------------------------
def regular_exact(scores: np.ndarray) -> dict:
    """Full-row (global-max) **exact** softmax — the value-fair 'regular' reference.

    Identical to `softmax_online_exact(scores, Bkv=S)` (one block = global max);
    same operator convention. The online/blocked form must reproduce this.
    """
    scores = np.asarray(scores, dtype=np.int32)
    ex = softmax_online_exact(scores, scores.shape[1])
    return {"m": ex["m"], "l": ex["l"], "p": ex["p"],
            "out_exact": ex["out_exact"], "out8": ex["out8_exact"]}


def reconstruct_p(out8: np.ndarray) -> np.ndarray:
    """Distribution implied by an int8 log-domain output: row softmax of exp(out8).

    `out8 = s - m - ls` (mean-shifted), so the row constant cancels and this is
    the softmax weight the output encodes.
    """
    o = np.asarray(out8, dtype=np.float64)
    e = np.exp(o - o.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def clamped_fraction(out8: np.ndarray) -> float:
    """Fraction of int8 outputs pinned at the ±128/127 rails."""
    o = np.asarray(out8)
    return float(((o <= -128) | (o >= 127)).mean())


def softmax_online_global(scores: np.ndarray) -> dict:
    """The shipped full-row operator (global max) — `softmax_ref` stages."""
    return sref.softmax_stage_values(np.asarray(scores, dtype=np.int32))


def softmax_online_pass_counts(S: int, R: int, C: int, Bkv: int,
                               n_exp: int = 1, n_log: int = 1) -> dict:
    """Packed-window pass counts for the blocked machine (**global packing**).

    EXP:    the block-major flat `S^2` packed stride-I -> ceil(S^2 / I) windows
    factor: the `B*S` `(m_b - m)` table          -> ceil(B*S / I) windows
    LOG:    the `S` `lq` values                  -> ceil(S / I) windows
    (`I = min(R, C)`; windows are packed globally, as in `softmax_sim`, not
    per-block.)
    """
    I = min(R, C)
    B = S // Bkv
    passes_exp = -(-(S * S) // I)
    passes_factor = -(-(B * S) // I) if B * S else 0
    passes_log = -(-S // I)
    return {"blocks": B, "passes_exp": passes_exp,
            "passes_factor": passes_factor, "passes_log": passes_log,
            "xbar_passes_exp": -(-passes_exp // n_exp),
            "xbar_passes_factor": -(-passes_factor // n_exp) if n_exp else 0,
            "xbar_passes_log": -(-passes_log // n_log),
            "combine_terms": B}


# ---------------------------------------------------------------------------
# Self-test — run:  python3 softmax_online_ref.py
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(11)

    # ── Hand-computed S=4, Bkv=2 ───────────────────────────────────────────
    small = np.array([[0, 0, 1, 3],
                      [-2, 5, 5, 5],
                      [10, 14, 10, 14],
                      [-128, 127, 0, 1]], dtype=np.int32)
    ex = softmax_online_exact(small, Bkv=2)
    tb = textbook_softmax(small)
    assert np.array_equal(ex["m"], tb["m"])
    assert np.allclose(ex["p"], tb["p"], atol=1e-12), "exact != textbook"
    assert np.allclose(ex["p"].sum(axis=1), 1.0, atol=1e-12)
    print("  [hand] S=4 Bkv=2: exact == textbook, rows sum to 1: OK")

    # ── exact ≡ textbook; Bkv-invariance; exact partial identity ───────────
    for S in (8, 32, 128):
        s = rng.integers(-128, 128, size=(S, S), dtype=np.int32)
        tb = textbook_softmax(s)
        ref_out8 = None
        for Bkv in (1, 2, 4, 8, S):
            if S % Bkv:
                continue
            ex = softmax_online_exact(s, Bkv)
            assert np.allclose(ex["p"], tb["p"], atol=1e-10), (S, Bkv)
            assert np.allclose(
                np.array([np.exp(ex["blk_max"][b] - ex["m"])
                          * ex["blk_Sp"][b] for b in range(S // Bkv)]).sum(0),
                ex["l"], rtol=1e-9), (S, Bkv, "partial identity")
            if Bkv == 1:
                ref_out8 = ex["out8_exact"]
            else:
                assert np.array_equal(ex["out8_exact"], ref_out8), \
                    (S, Bkv, "exact not Bkv-invariant")
        print(f"  [exact] S={S}: == textbook, Bkv-invariant, partials exact: OK")

    # ── ACAM model: ranges + Bkv=S reduces to softmax_ref bit-exactly ──────
    for S in (32, 128):
        s = rng.integers(-64, 64, size=(S, S), dtype=np.int32)
        gl = softmax_online_global(s)
        for Bkv in (1, 4, 16, S):
            if S % Bkv:
                continue
            md = softmax_online_model(s, Bkv)
            assert md["out8"].shape == (S, S) and md["out8"].dtype == np.int8
            assert (md["blk_Sp"] >= 0).all() and (md["L"] >= 0).all()
            assert (md["lq"] >= 0).all() and (md["lq"] <= 127).all()
        # One block == global max == the shipped operator's own mapping:
        # factor = ACAM_EXP(0) = 1, Sp_0 = output_sum, ls = ACAM_LOG(log_input).
        assert np.array_equal(softmax_online_model(s, S)["out8"],
                              gl["softmax_out"]), (S, "Bkv=S != softmax_ref")
    print("  [model/ACAM] ranges OK; Bkv=S reduces to softmax_ref bit-exactly: OK")

    # ── ACAM online vs the shipped operator: budget + distribution ─────────
    for S in (64, 128):
        s = rng.integers(-16, 17, size=(S, S), dtype=np.int32)
        ex = softmax_online_exact(s, Bkv=16)
        md = softmax_online_model(s, Bkv=16)
        gl = softmax_online_global(s)
        d_gl = int(np.abs(md["out8"].astype(np.int32)
                          - gl["softmax_out"].astype(np.int32)).max())
        l1_md = float(np.abs(reconstruct_p(md["out8"]) - ex["p"]).sum(1).max())
        l1_gl = float(np.abs(reconstruct_p(gl["softmax_out"])
                             - ex["p"]).sum(1).max())
        cf_md = clamped_fraction(md["out8"])
        cf_gl = clamped_fraction(gl["softmax_out"])
        print(f"  [compare] S={S}: max|online(ACAM)-shipped|={d_gl}; "
              f"L1(p̂,p) online={l1_md:.3f} shipped={l1_gl:.3f}; "
              f"clamp% online={cf_md:.3f} shipped={cf_gl:.3f}")

    # ── pass counts ────────────────────────────────────────────────────────
    pc = softmax_online_pass_counts(128, 256, 256, Bkv=16, n_exp=1, n_log=1)
    assert pc["blocks"] == 8 and pc["passes_log"] == 1
    assert pc["passes_exp"] == 8 * ((128 * 16 + 255) // 256) == 8 * 8
    pc2 = softmax_online_pass_counts(128, 256, 256, Bkv=128, n_exp=4, n_log=1)
    assert pc2["passes_exp"] == (128 * 128 + 255) // 256 == 64
    assert pc2["xbar_passes_exp"] == 16
    print(f"  [counts] Bkv=16 -> passes_exp={pc['passes_exp']}; "
          f"Bkv=128 n_exp=4 -> xbar_passes_exp={pc2['xbar_passes_exp']}: OK")

    # ── extremes / uniform / ties ──────────────────────────────────────────
    edges = {
        "extremes": np.array([[-128, 127] * 32, [127, -128] * 32] * 32,
                             dtype=np.int32),
        "uniform": np.zeros((64, 64), dtype=np.int32),
        "ties": np.tile(np.array([5] * 64, dtype=np.int32), (64, 1)),
    }
    for name, e in edges.items():
        ex = softmax_online_exact(e, Bkv=8)
        assert np.allclose(ex["p"].sum(axis=1), 1.0, atol=1e-9), name
        md = softmax_online_model(e, Bkv=8)
        assert md["out8"].shape == e.shape and md["out8"].dtype == np.int8, name
        print(f"  [edges] {name}: exact valid, ACAM model emits: OK")

    print("softmax_online_ref self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
