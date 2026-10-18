#!/usr/bin/env python3
"""proj_sim.py — the attention head's projection station `att_proj_top` (F2, V = 1, H = 1; Q, K and V writers):
values through the certified primitive, cycle model of the RTL structure (pinned by strict GATE 2).

Value contract: `attention_ref.project_log_in_pass` (V = 1: ACAM LOG on the projection crossbar's int32 y) — equal to
`attention_ref.project(...)["L"]` (the oracle self-test pins V = 1 == V > 1).
Structure (= the RTL):
  * one crossbar in LOG mode with the weights W [dm, d] (P29 extent dm rows x d columns); token t's X row is read
    from the head's input buffer (one 40-bit word per cycle, RD_LAT 1). Words 0 and 1 of the next token are
    prefetched into registers in the read port's free slots (from cycle p: reads at p, p+1, valid at p+2, p+3), so
    the burst starts the cycle the leader's MSB_SA_Ready allows (decision d = first strobe; word c >= 2 read at
    d + c - 2) — back to back the station streams at the certified T_fill + t*T_steady;
    p(t) = max(x_go(t) + GO_LAT, end of token t-1's reads), d(t) = max(p(t) + 2, previous msb_fire + 1);
    the drain gives the token's d log bytes, 5 per cycle;
  * a transposing writer: the drained words of the G tokens of a group go to G slot banks (one per token, OUT words
    of 40 bit), two halves (group parity); after the group's last token drained, one column word per cycle:
      Q (QK tile A): G = I / Bkv query tokens, word (sel = run, k = dim, w = window of the group in the run)
      K (QK tile B): G = Bkv key tokens (one block), word (sel = block, k = dim)
    word k's bytes: dim k of the group's tokens in order (byte i = token i of the group);
  * V (PV tile A, the transposed tile M = d, N = S, K = Bkv): token t is column k = t mod Bkv of V^T in selection
    (block) t div Bkv; its d bytes are W_K = d / NB window words of NB = I / S bytes. The drained token is held in a
    register set (two sets, token parity); after its drain, ceil(W_K / A_NW) writes of up to A_NW consecutive
    window words, one per cycle (the tile's bandwidth-matched A port);
  * VR (PV tile B, the untransposed tile M = S, N = d, K = Bkv — the head's PV, 2026-10-08): token t is row
    k = t mod Bkv of V in selection t div Bkv: one d-byte B word per token, after its drain.
Readiness: token t is fed once its X row is available (x_go pulse t seen) and the leader's MSB_SA_Ready is high.
V > 1 (dm > R; user 2026-10-08: one dedicated LOG crossbar per operand, one token per pass): the projection is the
certified `gemm_top` (V = ceil(dm / R) REGULAR tiles, full R rows each, d columns; X words carry V lanes, lane v =
X[t, vR .. vR + R - 1]; out8 = trunc8(sum_v y_v)), and its lane words (reg_full = out_valid delayed one cycle, the
word registered) stream straight into a LOG crossbar with identity weights (P29 extent d x d), whose drain feeds the
writers above unchanged. The converter never waits (d <= R: its T_steady = ceil(d/5) + P <= the gemm's
ceil(R/5) + P), so its load starts L_w + 1 cycles after the gemm tile drain starts (L_w = TREE_PIPE + 1).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "oracle"))
import nldpe_sim as prim  # noqa: E402
import gemm_sim  # noqa: E402
import attention_ref as aref  # noqa: E402
from pass_engine import schedule_pass_sequence  # noqa: E402

BUF, P, EPS = 40, 8, 5
PF_LAT = 2        # prefetch start -> word 0 in its register (read, then the register)
GO_LAT = 1        # an x_go pulse is visible to the engine one cycle later
EMIT_LAT = 2      # the group's last drain word -> its first column word (slot write, then the registered read)
KINDS = ("q", "k", "v", "vr")


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class ProjGeometry:
    kind: str
    n: int             # tokens
    dm: int
    d: int
    R: int
    C: int
    S: int
    bkv: int
    I: int
    G: int             # tokens per group (column word bytes); V: 1
    NB: int            # V: bytes per window word (I / S); Q / K: G
    W_K: int           # V: window words per token (d / NB); Q / K: 1
    A_NW: int          # V: window words per write; Q / K: 1
    LOAD: int          # feed words per token (V = 1: dm rows; V > 1: R rows per lane)
    OUT: int           # drain words per token (d columns)
    T_fill: int
    T_steady: int
    VP: int = 1        # projection tiles ceil(dm / R); > 1: gemm_top + the LOG converter
    L_W: int = 0       # V > 1: gemm_top fill latency TREE_PIPE + 1
    LOADC: int = 0     # V > 1: converter feed words per token (d rows)


def geometry(kind: str, n: int, dm: int, d: int, R: int, C: int, bkv: int = 16, a_nw: int = 1) -> ProjGeometry:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    I = min(R, C)
    if not (_pow2(n) and _pow2(d) and _pow2(bkv) and _pow2(I)):
        raise ValueError("n, d, bkv, I must be powers of two")
    VP = _cdiv(dm, R)
    if d > C:
        raise ValueError(f"H = 1 only: d={d} <= C={C}")
    if VP > 1 and d > R:
        raise ValueError(f"V > 1: the LOG converter keeps up only if d={d} <= R={R}")
    if not (8 <= bkv <= min(n, I)):
        raise ValueError("need 8 <= bkv <= min(n, I)")
    S = min(n, I)
    G = I // bkv if kind == "q" else (bkv if kind == "k" else 1)
    if G < 1 or S % G or n % G:
        raise ValueError(f"group G={G} must divide S={S} and n={n}")
    NB, W_K = (I // S, d // (I // S)) if kind == "v" else ((d, 1) if kind == "vr" else (G, 1))
    if kind == "v" and (W_K < 1 or d % NB):
        raise ValueError(f"V: d={d} must be a multiple of NB = I/S = {NB}")
    if kind != "v" and a_nw != 1:
        raise ValueError("A_NW > 1 is the V writer's")
    if not 1 <= a_nw <= W_K:
        raise ValueError(f"A_NW={a_nw} must be 1 .. W_K={W_K}")
    if VP == 1:
        cm = prim.cycle_model(1, R, C, P, BUF, rows=dm, cols=d)
        t_fill, l_w, loadc = int(cm.t_fill), 0, 0
    else:                                                    # gemm tiles: full R rows (lock-step), d columns
        cm = prim.cycle_model(1, R, C, P, BUF, rows=R, cols=d)
        cc = prim.cycle_model(1, R, C, P, BUF, rows=d, cols=d)
        l_w = int(gemm_sim.cycle_model(1, dm, d, R, C, BUF, P, pass_extent="used").l_w)
        loadc = int(cc.load_cyc)
        # tile pass up to its first drain word, lane words L_w + 1 later = the converter's load, converter pass
        t_fill = int(cm.t_fill) - int(cm.output_cyc) + l_w + 1 + int(cc.t_fill)
    if int(cm.load_cyc) < 3:
        raise ValueError(f"dm={dm}: the feed needs >= 3 words (dm >= 11)")
    nw = d if kind in ("q", "k") else -(-W_K // a_nw)        # writes per group (V / VR: per token)
    if nw > G * int(cm.t_steady):
        raise ValueError(f"{kind}: {nw} writes per group exceed {G} x T_steady = {G * int(cm.t_steady)} "
                         f"(the writer would fall behind; V: raise A_NW)")
    return ProjGeometry(kind=kind, n=n, dm=dm, d=d, R=R, C=C, S=S, bkv=bkv, I=I, G=G, NB=NB, W_K=W_K, A_NW=a_nw,
                        LOAD=int(cm.load_cyc), OUT=int(cm.output_cyc), T_fill=t_fill,
                        T_steady=int(cm.t_steady), VP=VP, L_W=l_w, LOADC=loadc)


def identity_weights(g: ProjGeometry) -> np.ndarray:
    """V > 1 converter weights: identity on the d x d extent, zero elsewhere."""
    Wi = np.zeros((g.R, g.C), dtype=np.int8)
    Wi[np.arange(g.d), np.arange(g.d)] = 1
    return Wi


def word_addr(g: ProjGeometry, grp: int) -> tuple:
    """(sel, w) of group grp's column words (Q / K)."""
    if g.kind == "q":
        t0 = grp * g.G
        return t0 // g.S, (t0 % g.S) // g.G
    return grp, 0


# ---------------------------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------------------------
class NldpeProj:
    """Values through the certified kernels. V = 1: one LOG-mode pass per token (weights W, extent dm x d).
    V > 1: the certified GEMM (REGULAR, out8 = trunc8(X @ W)), then one identity LOG pass per token (extent d x d)."""

    def __init__(self, kind: str, n: int, dm: int, d: int, R: int, C: int, bkv: int = 16, a_nw: int = 1) -> None:
        self.g = geometry(kind, n, dm, d, R, C, bkv, a_nw)
        if self.g.VP == 1:
            self.xbar = prim.NldpeDpe(R, C, BUF, P, rows_used=dm, cols_used=d)
        else:
            self.gemm = gemm_sim.NldpeGemm(n, dm, d, R, C, BUF, P, pass_extent="used")
            self.xbar = prim.NldpeDpe(R, C, BUF, P, rows_used=d, cols_used=d)

    def run(self, X: np.ndarray, W: np.ndarray) -> dict:
        g = self.g
        X = np.asarray(X, dtype=np.int8)
        W = np.asarray(W, dtype=np.int8)
        assert X.shape == (g.n, g.dm) and W.shape == (g.dm, g.d)
        if g.VP == 1:
            Wp = np.zeros((g.R, g.C), dtype=np.int8)
            Wp[:g.dm, :g.d] = W
            Xp = np.zeros((g.n, g.R), dtype=np.int8)
            Xp[:, :g.dm] = X
        else:                                                # gemm out8 -> identity rows of the LOG converter
            self.gemm.program_weights(W)
            x8 = self.gemm.run(X).out8
            Wp = identity_weights(g)
            Xp = np.zeros((g.n, g.R), dtype=np.int8)
            Xp[:, :g.d] = x8
        self.xbar.program_weights(Wp)
        out = self.xbar.run_workload(Xp, aref.ref.MODE_LOG).out_stream[:, :g.d].astype(np.int64)
        L = ((out & 0xFF) ^ 0x80) - 0x80
        words = []                                             # writes: (sel, k, w, cnt, bytes)
        if g.kind in ("v", "vr"):
            for t in range(g.n):
                for w0 in range(0, g.W_K, g.A_NW):
                    cnt = min(g.A_NW, g.W_K - w0)
                    words.append((t // g.bkv, t % g.bkv, w0, cnt,
                                  [int(L[t, w0 * g.NB + i]) for i in range(cnt * g.NB)]))
        else:
            for grp in range(g.n // g.G):
                sel, w = word_addr(g, grp)
                for k in range(g.d):
                    words.append((sel, k, w, 1, [int(L[grp * g.G + i, k]) for i in range(g.G)]))
        return {"L": L, "words": words}

    def gate1(self, X, W) -> dict:
        r = self.run(X, W)
        if not np.array_equal(r["L"], aref.project_log_in_pass(X, W)):
            raise RuntimeError("GATE 1: projection log bytes != attention_ref.project_log_in_pass")
        if not np.array_equal(r["L"], aref.project(X, W, self.g.R, self.g.C)["L"]):
            raise RuntimeError("GATE 1: projection log bytes != attention_ref.project (V = 1 == GEMM path)")
        return r


# ---------------------------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------------------------
def proj_cycle_model(g: ProjGeometry, x_go: list) -> dict:
    """Event times for tokens 0..n-1 with X-row pulses x_go[t] (absolute cycles, one pulse per cycle at most)."""
    assert len(x_go) == g.n and all(b > a for a, b in zip(x_go, x_go[1:])), "x_go: one pulse per token, increasing"
    ready_from = acc_free = out_free = 0
    c_ready = c_acc = c_out = 0                               # V > 1: the LOG converter's state
    port_free = 0                                             # first cycle the X read port is free
    toks, writes = [], []
    emit_free = 0                                             # the writer emits one word per cycle
    for t in range(g.n):
        pf = max(x_go[t] + GO_LAT, port_free)                 # prefetch words 0, 1
        d0 = max(pf + PF_LAT, ready_from)
        load_start = d0
        port_free = d0 + g.LOAD - 2                           # words 2 .. LOAD-1 read at d0 .. d0 + LOAD - 3
        compute_start = max(load_start + g.LOAD, acc_free)
        msb_fire = compute_start + P - 1
        acam_done = max(msb_fire + 2, out_free)
        drain_start, drain_end = acam_done + 1, acam_done + g.OUT
        ready_from, acc_free, out_free = msb_fire + 1, acam_done + 1, drain_end + 1
        ev = dict(prefetch=pf, decide=d0, load_start=load_start, msb_fire=msb_fire, drain_start=drain_start,
                  drain_end=drain_end)
        if g.VP > 1:                                          # lane words L_w + 1 later stream into the converter
            cl = drain_start + g.L_W + 1
            if cl < c_ready:
                raise RuntimeError(f"token {t}: the LOG converter is busy at {cl} (ready {c_ready}): words dropped")
            c_msb = max(cl + g.LOADC, c_acc) + P - 1
            c_acam = max(c_msb + 2, c_out)
            ev.update(g_drain_start=drain_start, g_drain_end=drain_end, conv_load=cl, conv_msb=c_msb,
                      drain_start=c_acam + 1, drain_end=c_acam + g.OUT)
            c_ready, c_acc, c_out = c_msb + 1, c_acam + 1, c_acam + g.OUT + 1
            drain_end = ev["drain_end"]
        toks.append(ev)
        if (t + 1) % g.G == 0:                                # the group's last token (V: every token)
            e0 = max(drain_end + EMIT_LAT, emit_free)
            nw = g.d if g.kind in ("q", "k") else -(-g.W_K // g.A_NW)
            writes += [e0 + k for k in range(nw)]
            emit_free = e0 + nw
    # two buffer halves: group grp+2 starts draining into the half group grp still emits from
    nw = g.d if g.kind in ("q", "k") else -(-g.W_K // g.A_NW)
    for grp in range(2, g.n // g.G):
        first_drain = toks[grp * g.G]["drain_start"]
        last_emit = writes[(grp - 2) * nw + nw - 1]
        if first_drain <= last_emit:
            raise RuntimeError(f"buffer half reuse: group {grp} drains at {first_drain} before group {grp - 2}"
                               f" finished emitting at {last_emit}")
    return dict(tokens=toks, writes=writes, done=writes[-1] + 1)


# ---------------------------------------------------------------------------------------------
# Case files for tb_att_proj_top.v
# ---------------------------------------------------------------------------------------------
def dump_case(case_dir: Path, pj: NldpeProj, X: np.ndarray, W: np.ndarray, x_go: list, cls: str = "") -> dict:
    """GATE 1, then: x.mem (token t's X row as LOAD words of VP lanes x 40 bits, t-major; lane v = X[t, vR ..]),
    w.mem (weight strobes: V = 1 R*C row-major; V > 1 the gemm_top order v-major, row-major, then the converter's
    identity R*C), xgo.txt (x_go cycle per token), words.mem (the column words in write order: sel k w data),
    case.json."""
    g = pj.g
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    r = pj.gate1(X, W)
    cm = proj_cycle_model(g, x_go)
    X = np.asarray(X, dtype=np.int64)
    span = g.dm if g.VP == 1 else g.R                         # rows per lane
    lines = []
    for t in range(g.n):
        for q in range(g.LOAD):
            v = 0
            for lane in range(g.VP):
                for j in range(EPS):
                    e, row = lane * span + 5 * q + j, 5 * q + j
                    if row < span and e < g.dm:
                        v |= (int(X[t, e]) & 0xFF) << (8 * (EPS * lane + j))
            lines.append(f"{v:0{10 * g.VP}x}")
    (case_dir / "x.mem").write_text("\n".join(lines) + "\n")
    Wp = np.zeros((g.VP * g.R, g.C), dtype=np.int64)
    Wp[:g.dm, :g.d] = np.asarray(W, dtype=np.int64)
    strobes = [Wp[v * g.R:(v + 1) * g.R].reshape(-1) for v in range(g.VP)]
    if g.VP > 1:
        strobes.append(identity_weights(g).astype(np.int64).reshape(-1))
    (case_dir / "w.mem").write_text("".join(f"{int(v) & 0xFF:02x}\n" for v in np.concatenate(strobes)))
    (case_dir / "xgo.txt").write_text("".join(f"{x}\n" for x in x_go))

    def hexw(bs):
        v = 0
        for i, b in enumerate(bs):
            v |= (b & 0xFF) << (8 * i)
        return f"{v:x}"
    (case_dir / "words.mem").write_text("".join(f"{s:x} {k:x} {w:x} {c:x} {hexw(bs)}\n" for s, k, w, c, bs in r["words"]))
    case = dict(kind=g.kind, n=g.n, dm=g.dm, d=g.d, R=g.R, C=g.C, S=g.S, bkv=g.bkv, I=g.I, G=g.G, NB=g.NB, W_K=g.W_K,
                A_NW=g.A_NW, LOAD=g.LOAD, VP=g.VP,
                OUT=g.OUT, cls=cls, n_words=len(r["words"]), exp_writes=cm["writes"], exp_done=cm["done"],
                exp_drain_end=[x["drain_end"] for x in cm["tokens"]])
    (case_dir / "case.json").write_text(json.dumps(case, indent=1) + "\n")
    return case


# ---------------------------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/simulator/kernels/proj_sim.py
# ---------------------------------------------------------------------------------------------
def _operands(rng, n, dm, d, cls):
    if cls == "extremes":
        X = rng.choice(np.array([-128, -1, 0, 1, 127]), size=(n, dm))
        W = rng.choice(np.array([-128, -1, 0, 1, 127]), size=(dm, d))
    elif cls == "zeros":
        X = np.zeros((n, dm), dtype=np.int64)
        W = rng.integers(-128, 128, size=(dm, d))
    else:
        X = rng.integers(-128, 128, size=(n, dm))
        W = rng.integers(-128, 128, size=(dm, d))
    return X.astype(np.int8), W.astype(np.int8)


def _self_test() -> None:
    rng = np.random.default_rng(2026_10_08)
    n_chk = 0
    for kind, extra in (("q", 0), ("k", 0), ("v", 0), ("v", 2), ("vr", 0)):
        for n, dm, d, R, C, bkv in ((32, 24, 16, 64, 64, 8), (64, 64, 32, 64, 64, 16), (16, 40, 8, 128, 64, 8),
                                    (128, 128, 64, 256, 256, 16),
                                    (32, 96, 16, 64, 64, 8), (64, 160, 32, 128, 64, 16), (16, 200, 8, 64, 128, 8),
                                    (256, 192, 64, 128, 128, 16)):            # V > 1 (VP 2, 2, 4, 2)
            a_nw = 1
            if kind == "v":                                    # the smallest A_NW that keeps up, plus `extra`
                I = min(R, C); nb = I // min(n, I); wk = d // nb
                ts = int(prim.cycle_model(1, R, C, P, BUF, rows=dm if dm <= R else R, cols=d).t_steady)
                a_nw = min(wk, -(-wk // ts) + extra)
                if a_nw > 1:
                    try:
                        geometry("v", n, dm, d, R, C, bkv, a_nw - 1 if extra == 0 else 1)
                        assert extra or -(-wk // (a_nw - 1)) <= ts, "guard must reject a writer that falls behind"
                    except ValueError:
                        n_chk += 1
            pj = NldpeProj(kind, n, dm, d, R, C, bkv, a_nw)
            for cls in ("random", "extremes", "zeros"):
                X, W = _operands(rng, n, dm, d, cls)
                r = pj.gate1(X, W)
                # word layout == the tile operands (attention_ref.head_from_qkv / dimm_tile_sim formats)
                g = pj.g
                L = r["L"]
                seen = {}
                for sel, k, w, cnt, bs in r["words"]:
                    if kind == "q":                            # QK tile A: LA = Q rows of run sel, word (k, w)
                        assert bs == [int(L[sel * g.S + w * g.G + i, k]) for i in range(g.G)]
                    elif kind == "k":                          # QK tile B: LB = K^T of block sel, word (k)
                        assert bs == [int(L[sel * g.bkv + i, k]) for i in range(g.G)]
                    elif kind == "vr":                         # PV tile B (untransposed): LB = V of block sel, word (k)
                        assert w == 0 and cnt == 1 and bs == [int(L[sel * g.bkv + k, c]) for c in range(g.d)]
                        seen[(sel, k, 0)] = True
                        continue
                    else:                                      # PV tile A: LA = V^T of block sel, words (k, w..w+cnt-1)
                        LA = L[sel * g.bkv:(sel + 1) * g.bkv].T                     # [d, Bkv]
                        assert bs == [int(LA[(w + j) * g.NB + i, k]) for j in range(cnt) for i in range(g.NB)]
                        for j in range(cnt):
                            seen[(sel, k, w + j)] = True
                        continue
                    seen[(sel, k, w)] = True
                assert len(seen) == g.n * g.d // g.G // (g.NB if kind in ("v", "vr") else 1)
                n_chk += 1
            # cycles: back-to-back stream == T_fill + t*T_steady (certified), and monotone in x_go
            g = pj.g
            cm = proj_cycle_model(g, list(range(g.n)))
            t0 = GO_LAT + PF_LAT
            for t, ev in enumerate(cm["tokens"]):
                assert ev["drain_end"] + 1 - t0 == g.T_fill + t * g.T_steady, (kind, t)
            if g.VP == 1:
                ref = schedule_pass_sequence([t0] * g.n, R, C, BUF, P, rows=dm, cols=d)
                assert [e["drain_end"] for e in ref] == [e["drain_end"] for e in cm["tokens"]]
            else:                                              # gemm tiles == the certified GEMM timeline, converter
                gt = gemm_sim.NldpeGemm(g.n, dm, d, R, C, BUF, P, pass_extent="used")      # == the pass engine
                gt.program_weights(np.zeros((dm, d), dtype=np.int8))
                tl = gt.run(np.zeros((g.n, dm), dtype=np.int8)).timeline
                assert [x.array_drain_start + t0 + 1 for x in tl] == [e["conv_load"] for e in cm["tokens"]], kind
                ref = schedule_pass_sequence([e["conv_load"] for e in cm["tokens"]], R, C, BUF, P, rows=d, cols=d)
                assert [e["drain_end"] for e in ref] == [e["drain_end"] for e in cm["tokens"]]
                assert [e["load_start"] for e in ref] == [e["conv_load"] for e in cm["tokens"]], "converter waited"
                n_chk += 1
            late = proj_cycle_model(g, [5 * t for t in range(g.n)])
            assert all(b["drain_end"] >= a["drain_end"] for a, b in zip(cm["tokens"], late["tokens"]))
            n_chk += 1
    for bad in (("q", 32, 24, 128, 64, 64, 8), ("v", 64, 64, 32, 64, 64, 16), ("k", 24, 24, 16, 64, 64, 8),
                ("k", 32, 24, 16, 64, 64, 4), ("q", 32, 160, 128, 64, 128, 8)):   # d > C; V>1 with d > R
        try:
            geometry(*bad)
        except ValueError:
            n_chk += 1
        else:
            raise AssertionError(f"guard missed {bad}")
    print(f"proj_sim: ALL PASS ({n_chk} checks: log bytes == attention_ref V=1 and GEMM paths through the certified "
          f"LOG pass, Q / K / V word layouts == the tile operands, writer-rate guard, stream == certified T_fill + t*T_steady, guards)")


if __name__ == "__main__":
    _self_test()
