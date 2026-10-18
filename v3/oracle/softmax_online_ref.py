#!/usr/bin/env python3
"""softmax_online_ref.py — NumPy value reference for the online softmax (log-domain carry).

Role
----
Value contract of the online softmax built for the FlashAttention interface
(charter `v2/sim/spec/softmax_online.md`). Plain math, no state, no time; the RTL and the behavior simulator must agree with
it bit-exactly on every stage below. Cycles are not modelled here.

Shapes: `scores` is [S, N] — S query rows (processed together), N keys streamed in
B = N/Bkv blocks of Bkv keys (2026-10-06: N decoupled from S; N = S is the square case).

Contract (per row r; key blocks b = 0..B-1; P24 ACAM forms)
-----------------------------------------------------------
    m_blk   = max_j s[r, j]                                  (j in block b)
    m_new   = m_blk                       (b = 0)
            = max(m, m_blk)               (b > 0)
    log_a   = m - m_new                   (b > 0; <= 0, range [-255, 0]; 0 at b = 0)
    logp_j  = max(s[r, j] - m_new, -128)  -> the block's log|P| output (int8, to P.V)
    sp      = sum_j ACAM_EXP(logp_j)      (unsigned bytes, exact)
    l'      = sp                                            (b = 0)
            = (ACAM_EXP(t) << log2 N) + sp,
              t = clamp(log_a + log_l, -128, 127)           (b > 0; EXP byte unsigned)
    lq      = min(l' >> log2 N, 127)      (the mean over the N keys, as softmax_ref)
    log_l'  = ACAM_LOG(lq)                (signed int8)
    carry   : m <- m_new, log_l <- log_l'
    end     : (m, log_l) per row

ACAM_EXP(y) = trunc8(1 + y + floor(y^2/2)), ACAM_LOG(y) = trunc8(y - 1) (P24).
For checking only, the standalone result is rebuilt as
    out[r, j] = clamp(s[r, j] - m - log_l, -128, 127).

Properties
----------
* Bkv = N (one block) is bit-identical to the conventional softmax on N-key rows
  (`softmax_ref` for N = S; `softmax_rect` below for any N) — self-test gate.
* Bkv < N: the carry passes through ACAM_EXP(ACAM_LOG(.)), which is not the
  identity under P24, so values drift block by block — the accepted
  approximation of decision A (2026-10-05); the drift is reported, not gated.
* Rows are independent: the result of a row block does not depend on S.
* Domain: N, Bkv powers of two, 8 <= Bkv <= N; S >= 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nldpe_ref as ref  # noqa: E402
import softmax_ref as sref  # noqa: E402


def _check_domain(S: int, N: int, Bkv: int) -> None:
    if N & (N - 1) or Bkv & (Bkv - 1) or N <= 0 or Bkv <= 0:
        raise ValueError(f"N={N}, Bkv={Bkv}: both must be powers of two")
    if not 8 <= Bkv <= N:
        raise ValueError(f"Bkv={Bkv} outside 8 <= Bkv <= N={N}")
    if S < 1:
        raise ValueError(f"S={S}: need at least one row")


def _exp_u8(y: np.ndarray) -> np.ndarray:
    """ACAM_EXP as the unsigned byte the row sums consume."""
    return ref.acam_transform(np.asarray(y, dtype=np.int64), ref.MODE_EXP) \
        .view(np.uint8).astype(np.int64)


def _log_i8(y: np.ndarray) -> np.ndarray:
    """ACAM_LOG as the signed byte the carry holds."""
    return ref.acam_transform(np.asarray(y, dtype=np.int64), ref.MODE_LOG) \
        .astype(np.int64)


def softmax_rect(scores: np.ndarray) -> dict:
    """Conventional (full-row) NL softmax on [S, N] scores — `softmax_ref` with the
    mean taken over the N keys; equal to `softmax_ref.softmax_stage_values` for N = S."""
    s = np.asarray(scores, dtype=np.int64)
    N = s.shape[1]
    assert N & (N - 1) == 0, "N must be a power of two"
    LN = N.bit_length() - 1
    m = s.max(axis=1)
    d = np.maximum(s - m[:, None], -128)
    total = _exp_u8(d).sum(axis=1)
    lq = np.minimum(total >> LN, 127)
    ls = _log_i8(lq)
    return {"row_max": m, "exp_input": d, "output_sum": total, "log_input": lq,
            "log_output": ls, "softmax_out": np.clip(s - m[:, None] - ls[:, None], -128, 127)}


def logcarry_stage_values(scores: np.ndarray, Bkv: int) -> dict:
    """Per-block, per-stage values of the log-carry online softmax (vectorized).

    scores: [S, N]. Returns (b = block, r = row, c = key within the block):
      m_blk  [B, S]       int   block max
      m_new  [B, S]       int   running max after block b
      log_a  [B, S]       int   m_old - m_new (0 at b = 0), range [-255, 0]
      logp   [B, S, Bkv]  int   max(s - m_new, -128)          (log|P| output)
      p      [B, S, Bkv]  int   ACAM_EXP(logp) as unsigned byte
      sp     [B, S]       int   sum_c p
      t      [B, S]       int   clamp(log_a + log_l_prev, -128, 127) (0 at b = 0)
      e      [B, S]       int   ACAM_EXP(t) as unsigned byte (0 at b = 0)
      lprime [B, S]       int   l'
      lq     [B, S]       int   min(l' >> log2 N, 127)
      log_l  [B, S]       int   ACAM_LOG(lq) (signed)
      m      [S]  final running max;  log_l_final [S];  out [S, N] standalone check
    """
    s = np.asarray(scores, dtype=np.int64)
    assert s.ndim == 2, "scores must be [S, N]"
    S, N = s.shape
    _check_domain(S, N, Bkv)
    B = N // Bkv
    LN = N.bit_length() - 1

    blocks = s.reshape(S, B, Bkv).transpose(1, 0, 2)          # [B, S, Bkv]
    m_blk = blocks.max(axis=2)
    keys = ("m_new", "log_a", "sp", "t", "e", "lprime", "lq", "log_l")
    st = {k: np.zeros((B, S), dtype=np.int64) for k in keys}
    logp = np.zeros((B, S, Bkv), dtype=np.int64)
    p = np.zeros((B, S, Bkv), dtype=np.int64)

    m = None
    log_l = None
    for b in range(B):
        m_new = m_blk[b] if b == 0 else np.maximum(m, m_blk[b])
        log_a = np.zeros(S, dtype=np.int64) if b == 0 else m - m_new
        logp[b] = np.maximum(blocks[b] - m_new[:, None], -128)
        p[b] = _exp_u8(logp[b])
        sp = p[b].sum(axis=1)
        if b == 0:
            t = np.zeros(S, dtype=np.int64)
            e = np.zeros(S, dtype=np.int64)
            lprime = sp
        else:
            t = np.clip(log_a + log_l, -128, 127)
            e = _exp_u8(t)
            lprime = (e << LN) + sp
        lq = np.minimum(lprime >> LN, 127)
        log_l = _log_i8(lq)
        m = m_new
        for k, v in zip(keys, (m_new, log_a, sp, t, e, lprime, lq, log_l)):
            st[k][b] = v

    out = np.clip(s - m[:, None] - log_l[:, None], -128, 127)
    return {"S": S, "N": N, "Bkv": Bkv, "B": B, "m_blk": m_blk, "logp": logp, "p": p,
            **st, "m": m, "log_l_final": log_l, "out": out}


def logcarry_pass_view(scores: np.ndarray, Bkv: int, R: int = 256, C: int = 256) -> dict:
    """Dual witness: per row, scalar recurrence; every ACAM goes through an
    identity crossbar pass (`convert_stream`, capacity I = min(R, C))."""
    s = np.asarray(scores, dtype=np.int64)
    S, N = s.shape
    _check_domain(S, N, Bkv)
    B = N // Bkv
    LN = N.bit_length() - 1
    m_fin = np.zeros(S, dtype=np.int64)
    ll_fin = np.zeros(S, dtype=np.int64)
    sp_all = np.zeros((B, S), dtype=np.int64)
    ll_all = np.zeros((B, S), dtype=np.int64)
    logp_all = np.zeros((B, S, Bkv), dtype=np.int64)
    for r in range(S):
        m = ll = None
        for b in range(B):
            blk = s[r, b * Bkv:(b + 1) * Bkv]
            mb = int(blk.max())
            mn = mb if b == 0 else max(m, mb)
            la = 0 if b == 0 else m - mn
            lp = np.maximum(blk - mn, -128).astype(np.int8)
            pe, _ = ref.convert_stream(lp, ref.MODE_EXP, R, C)
            sp = int(pe.view(np.uint8).astype(np.int64).sum())
            if b == 0:
                l1 = sp
            else:
                tt = int(min(max(la + ll, -128), 127))
                ee, _ = ref.convert_stream(np.array([tt], dtype=np.int8), ref.MODE_EXP, R, C)
                l1 = (int(ee.view(np.uint8)[0]) << LN) + sp
            lq = min(l1 >> LN, 127)
            lg, _ = ref.convert_stream(np.array([lq], dtype=np.int8), ref.MODE_LOG, R, C)
            m, ll = mn, int(lg[0])
            sp_all[b, r] = sp
            ll_all[b, r] = ll
            logp_all[b, r] = lp
        m_fin[r], ll_fin[r] = m, ll
    out = np.clip(s - m_fin[:, None] - ll_fin[:, None], -128, 127)
    return {"m": m_fin, "log_l_final": ll_fin, "sp": sp_all, "log_l": ll_all,
            "logp": logp_all, "out": out}


def drift_vs_softmax_ref(scores: np.ndarray, Bkv: int) -> dict:
    """Reported (not gated): deviation of the rebuilt standalone result from the
    conventional softmax on the same rows, and the clamped fraction."""
    a = logcarry_stage_values(scores, Bkv)["out"]
    g = softmax_rect(scores)["softmax_out"]
    d = np.abs(a - g)
    return {"equal_frac": float((d == 0).mean()), "mean_abs": float(d.mean()),
            "max_abs": int(d.max()), "clamped_frac": float(((a == -128) | (a == 127)).mean())}


def _stimuli(S: int, N: int, rng: np.random.Generator) -> dict:
    return {
        "random": rng.integers(-128, 128, size=(S, N)),
        "narrow": rng.integers(-16, 17, size=(S, N)),
        "extremes": np.tile(np.array([[-128, 127], [127, -128]]), (-(-S // 2), N // 2))[:S],
        "uniform": np.zeros((S, N), dtype=np.int64),
        "ties": np.full((S, N), 5, dtype=np.int64),
        "rising": np.tile(np.arange(N) - N // 2, (S, 1)).clip(-128, 127),
    }


def _self_test() -> None:
    rng = np.random.default_rng(2026_10_05)
    n = 0
    # 1) softmax_rect == softmax_ref on square scores
    for S in (8, 32, 128):
        for name, sc in _stimuli(S, S, rng).items():
            g = sref.softmax_stage_values(sc.astype(np.int32))
            w = softmax_rect(sc)
            for k in ("row_max", "output_sum", "log_input", "log_output", "softmax_out"):
                assert np.array_equal(w[k], np.asarray(g[k], dtype=np.int64)), (S, name, k)
            n += 1
    # 2) Bkv = N is the conventional softmax, bit for bit (square and rectangular)
    for S, N in ((8, 8), (16, 16), (64, 64), (256, 256), (16, 64), (8, 256), (128, 32), (64, 1024)):
        for name, sc in _stimuli(S, N, rng).items():
            v = logcarry_stage_values(sc, N)
            g = softmax_rect(sc)
            assert np.array_equal(v["m"], g["row_max"]), (S, N, name, "m")
            assert np.array_equal(v["sp"][0], g["output_sum"]), (S, N, name, "sp")
            assert np.array_equal(v["lq"][0], g["log_input"]), (S, N, name, "lq")
            assert np.array_equal(v["log_l_final"], g["log_output"]), (S, N, name, "log_l")
            assert np.array_equal(v["logp"][0], g["exp_input"]), (S, N, name, "logp")
            assert np.array_equal(v["out"], g["softmax_out"]), (S, N, name, "out")
            n += 1
    # 3) vectorized view == scalar crossbar-pass view; invariants; row independence
    for S, N in ((8, 8), (16, 16), (32, 32), (64, 64), (16, 64), (8, 128), (32, 16), (64, 256)):
        for Bkv in [k for k in (8, 16, 32, 64) if k <= N]:
            for name, sc in _stimuli(S, N, rng).items():
                v = logcarry_stage_values(sc, Bkv)
                w = logcarry_pass_view(sc, Bkv)
                for k in ("m", "log_l_final", "out", "sp", "log_l", "logp"):
                    assert np.array_equal(v[k], w[k]), (S, N, Bkv, name, k)
                assert (v["log_a"] <= 0).all() and (v["log_a"] >= -255).all()
                assert (v["logp"] <= 0).all() and (v["logp"] >= -128).all()
                assert (v["lq"] >= 0).all() and (v["lq"] <= 127).all()
                assert np.array_equal(v["m"], np.asarray(sc).max(axis=1))
                assert (v["log_a"][0] == 0).all() and (v["t"][0] == 0).all()
                if S >= 2:                            # rows are independent of the row count
                    h = logcarry_stage_values(sc[: S // 2], Bkv)
                    assert np.array_equal(h["out"], v["out"][: S // 2]), (S, N, Bkv, name, "rows")
                n += 1
    # 4) domain guards
    for S, N, Bkv in ((16, 16, 4), (16, 16, 32), (16, 24, 8), (16, 16, 12), (0, 16, 8)):
        try:
            logcarry_stage_values(np.zeros((S, N), dtype=np.int64), Bkv)
        except (ValueError, AssertionError):
            continue
        raise AssertionError(f"domain guard missed S={S} N={N} Bkv={Bkv}")
    print(f"softmax_online_ref: ALL PASS ({n} cases: softmax_rect == softmax_ref, "
          "Bkv=N == conventional softmax (square + rectangular), vectorized == crossbar-pass "
          "view, invariants, row independence, domain guards)")
    # 5) drift report (Bkv < N), not gated
    print("  drift vs the conventional softmax (equal fraction / mean |d| / clamped fraction):")
    for S, N, Bkv in ((64, 64, 16), (128, 128, 16), (64, 512, 16), (128, 1024, 32)):
        sc = rng.integers(-128, 128, size=(S, N))
        d = drift_vs_softmax_ref(sc, Bkv)
        print(f"    S={S:3d} N={N:4d} Bkv={Bkv:2d}: equal {d['equal_frac']:.3f}  "
              f"mean|d| {d['mean_abs']:.1f}  max|d| {d['max_abs']}  clamped {d['clamped_frac']:.3f}")


if __name__ == "__main__":
    _self_test()
