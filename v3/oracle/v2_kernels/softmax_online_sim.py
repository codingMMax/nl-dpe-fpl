#!/usr/bin/env python3
"""softmax_online_sim.py — behavior + cycle simulator of the NL online softmax
(log-domain carry, FlashAttention interface).

Models `v2/sim/rtl/softmax_online_top.v` (charter `v2/sim/spec/softmax_online.md`):

* Values: every ACAM goes through the certified primitive model
  (`nldpe_sim.NldpeDpe`, identity weights, packed windows via
  `softmax_sim._convert_packed_stream`) — an independent path from the
  closed-form oracle `v2/sim/oracle/softmax_online_ref.py`; GATE 1 = sim values
  == oracle on every stage, checked per case in `dump_case` before any file is
  written.
* Cycles: data-independent event model of the RTL structure. Each crossbar
  places its passes with the certified pass grammar
  (`pass_engine.schedule_pass_sequence`, anchored at the first accepted word);
  the CLB stages add the structural offsets below (pipeline depths of the RTL,
  each named after the register stages it counts). Cycles are counted from the
  `start` edge to the `done` edge (the TB's measured cycles; GATE 2 requires
  Delta_impl = 0).

Structural offsets (softmax_online_top.v):
  MAX_LAT  = 4  last word of block b accepted -> max_cnt counts b
                (softmax_online_loadmax M0, M1a, M1b, M2, M3 write)
  FEED_LAT = 9  last word of block b -> earliest first EXP pop of its windows
                (max_cnt visible, registered block gate, producer issue, I1 group
                select + RAM address, I2 row select + RAM read, I3 -> FIFO, pop)
  SP_LAT   = 2  row-completing drain word -> sp count (row_acc stage A + event)
  WAIT_LAT = 1  condition visible -> carry sequencer transition
  GO_LAT   = 4  sequencer trigger -> first pop of a carry pass
                (go register, wfeed start, issue + registered select, FIFO)
Stream: `start` is registered (re-arm), so word 0 is accepted at edge 2, then one
word per edge; block b >= 2 waits for retire(b-2) (its slot); a window's last row
sits in drain word WPW-1, a carry window's in drain word WL-1 (WL = ceil(S/EPS));
the LOG drain word with row S-1 retires the block; `done` = retire(B-1) + 1.
Shapes: S rows per run, N keys in B = N/Bkv blocks (default N = S). Every run
(re-armed by `start` after `done`) has the same cycles.

Domain (the RTL's elaboration guards, enforced here too): S, N, Bkv,
I = min(R, C), n_exp powers of two; 8 <= S <= I; 8 <= Bkv <= min(N, I); BUF = 40.

Run:  python3 v2/sim/simulator/kernels/softmax_online_sim.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "oracle"))
import nldpe_sim as prim  # noqa: E402
from pass_engine import identity_extent, schedule_pass_sequence  # noqa: E402
from softmax_sim import _convert_packed_stream  # noqa: E402
import nldpe_ref as nref  # noqa: E402
import softmax_online_ref as oref  # noqa: E402

MAX_LAT, FEED_LAT, SP_LAT, WAIT_LAT, GO_LAT = 4, 9, 2, 1, 4
EPS = 5

# Stage keys certified by GATE 1 (sim == oracle).
STAGE_KEYS = ("m_blk", "m_new", "log_a", "logp", "sp", "t", "e", "lprime", "lq",
              "log_l", "m", "log_l_final")


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def check_domain(S: int, Bkv: int, R: int, C: int, BUF: int, n_exp: int,
                 N: int | None = None) -> None:
    """The RTL's elaboration-time guards."""
    N = S if N is None else N
    I = min(R, C)
    pow2 = lambda x: x > 0 and x & (x - 1) == 0  # noqa: E731
    if not (pow2(S) and pow2(N) and pow2(Bkv) and pow2(I) and pow2(n_exp)):
        raise ValueError("S, N, Bkv, I = min(R, C), n_exp must be powers of two")
    if not 8 <= S <= I:
        raise ValueError(f"need 8 <= S <= I (S={S}, I={I})")
    if not 8 <= Bkv <= min(N, I):
        raise ValueError(f"need 8 <= Bkv <= min(N, I) (Bkv={Bkv}, N={N}, I={I})")
    if BUF != 40:
        raise ValueError("BUF must be 40 (EPS = 5)")


@dataclass
class SoftmaxOnlineCycleModel:
    """Measured-cycle model of softmax_online_top for one run (start -> done)."""

    S: int
    N: int
    Bkv: int
    n_exp: int
    total: int                  # `done` edge = compute cycles of one run
    stream_end: int             # edge the last score word is accepted
    stall: int                  # edges the stream waits for a free block slot
    exp_passes: list            # EXP passes per crossbar
    carry_exp_passes: int       # B - 1
    carry_log_passes: int       # B
    blocks: list = field(default_factory=list)   # per-block event edges
    pass_extent: str = "full"   # "full" (certified, RTL-witnessed) | "used" (P29: I x I passes, RTL revision pending)


def softmax_online_cycle_model(S: int, Bkv: int, R: int = 256, C: int = 256, BUF: int = 40,
                               P: int = 8, n_exp: int = 1,
                               N: int | None = None, *,
                               pass_extent: str = "full") -> SoftmaxOnlineCycleModel:
    """Cycles of the softmax_online_top RTL for one run (data-independent).

    `pass_extent` (keyword-only, dpe_nldpe.md P29): "full" (default) = the certified
    RTL schedule; "used" = every EXP / carry-EXP / carry-LOG identity pass loads and
    drains only I = min(R, C) (model-level only until the RTL revision).
    """
    N = S if N is None else N
    check_domain(S, Bkv, R, C, BUF, n_exp, N)
    rows, cols = identity_extent(R, C, pass_extent)
    B, BLKEL = N // Bkv, S * Bkv
    STREAM = _cdiv(BLKEL, EPS)
    NWEL = min(BLKEL, min(R, C))
    NWB = BLKEL // NWEL
    WPW, WL = _cdiv(NWEL, EPS), _cdiv(S, EPS)

    ready = [[] for _ in range(n_exp)]
    cexp_ready, clog_ready = [], []
    a_last, ret, wait, blocks = [], [], [], []
    for b in range(B):
        first = 2 if b == 0 else a_last[b - 1] + 1
        if b >= 2:
            first = max(first, ret[b - 2] + 1)
        a_last.append(first + STREAM - 1)
        sp_done = 0
        for w in range(NWB):
            g = b * NWB + w
            c = g % n_exp
            ready[c].append(a_last[b] + FEED_LAT)
            ev = schedule_pass_sequence(ready[c], R, C, BUF, P, rows=rows, cols=cols)[-1]
            sp_done = max(sp_done, ev["drain_start"] + (WPW - 1) + SP_LAT)
        cond = max(sp_done, a_last[b] + MAX_LAT, ret[b - 1] if b else 0)
        wait.append(cond + WAIT_LAT)
        if b == 0:
            clog_ready.append(wait[b] + WL + GO_LAT)
            x = None
        else:
            cexp_ready.append(wait[b] + GO_LAT)
            evx = schedule_pass_sequence(cexp_ready, R, C, BUF, P, rows=rows, cols=cols)[-1]
            x = evx["drain_start"] + (WL - 1)
            clog_ready.append(x + GO_LAT)
        evl = schedule_pass_sequence(clog_ready, R, C, BUF, P, rows=rows, cols=cols)[-1]
        ret.append(evl["drain_start"] + (WL - 1))
        blocks.append(dict(first_word=first, last_word=a_last[b], sp_full=sp_done,
                           carry_start=wait[b], cexp_last=x, retire=ret[b]))
    # the TB counts the stream from edge 1, so block 0 contributes its one-edge start
    stall = sum(blk["first_word"] - (a_last[i - 1] + 1 if i else 1) for i, blk in enumerate(blocks))
    return SoftmaxOnlineCycleModel(S=S, N=N, Bkv=Bkv, n_exp=n_exp, total=ret[-1] + 1,
                                   stream_end=a_last[-1], stall=stall,
                                   exp_passes=[len(r) for r in ready],
                                   carry_exp_passes=len(cexp_ready),
                                   carry_log_passes=len(clog_ready), blocks=blocks,
                                   pass_extent=pass_extent)


@dataclass
class SoftmaxOnlineResult:
    values: dict                         # per-stage values (keys: STAGE_KEYS + "logp")
    compute_cycles: int                  # `done` edge of one run
    load_words: int                      # score words streamed (overlapped with compute)
    cycle_model: SoftmaxOnlineCycleModel


class NldpeSoftmaxOnline:
    """NL online softmax: values via the certified primitive, cycles via the model."""

    def __init__(self, S: int, R: int = 256, C: int = 256, BUF: int = 40, P: int = 8,
                 n_exp: int = 1, Bkv: int = 16, N: int | None = None) -> None:
        self.N = S if N is None else N
        check_domain(S, Bkv, R, C, BUF, n_exp, self.N)
        self.S, self.Bkv, self.R, self.C, self.BUF, self.P, self.n_exp = S, Bkv, R, C, BUF, P, n_exp
        self.B = self.N // Bkv
        eye = np.eye(R, C, dtype=np.int8)

        def _eye():
            d = prim.NldpeDpe(R, C, BUF, P)
            d.program_weights(eye)
            return d
        self._exp = [_eye() for _ in range(n_exp)]
        self._cexp = [_eye()]
        self._clog = [_eye()]

    def _conv(self, flat, mode, xbars):
        v, _ = _convert_packed_stream(np.asarray(flat, dtype=np.int8), mode, self.R, self.C,
                                      self.BUF, self.P, xbars)
        return v

    def cycle_model(self) -> SoftmaxOnlineCycleModel:
        return softmax_online_cycle_model(self.S, self.Bkv, self.R, self.C, self.BUF, self.P,
                                          self.n_exp, self.N)

    def stream_words(self) -> int:
        return self.B * _cdiv(self.S * self.Bkv, EPS)

    def values(self, scores: np.ndarray) -> dict:
        s = np.asarray(scores, dtype=np.int64)
        S, Bkv, B = self.S, self.Bkv, self.B
        assert s.shape == (S, self.N), f"scores must be [{S}, {self.N}]"
        LS = self.N.bit_length() - 1            # mean shift over the N keys
        out = {k: np.zeros((B, S), dtype=np.int64)
               for k in ("m_blk", "m_new", "log_a", "sp", "t", "e", "lprime", "lq", "log_l")}
        out["logp"] = np.zeros((B, S, Bkv), dtype=np.int64)
        m = ll = None
        for b in range(B):
            blk = s[:, b * Bkv:(b + 1) * Bkv]
            mb = blk.max(axis=1)
            mn = mb if b == 0 else np.maximum(m, mb)
            la = np.zeros(S, dtype=np.int64) if b == 0 else m - mn
            lp = np.maximum(blk - mn[:, None], -128)
            p = self._conv(lp.reshape(-1), nref.MODE_EXP, self._exp).view(np.uint8).astype(np.int64)
            sp = p.reshape(S, Bkv).sum(axis=1)
            if b == 0:
                t = np.zeros(S, dtype=np.int64)
                e = np.zeros(S, dtype=np.int64)
                lprime = sp
            else:
                t = np.clip(la + ll, -128, 127)
                e = self._conv(t, nref.MODE_EXP, self._cexp).view(np.uint8).astype(np.int64)
                lprime = (e << LS) + sp
            lq = np.minimum(lprime >> LS, 127)
            ll = self._conv(lq, nref.MODE_LOG, self._clog).astype(np.int64)
            m = mn
            for k, v in (("m_blk", mb), ("m_new", mn), ("log_a", la), ("sp", sp), ("t", t),
                         ("e", e), ("lprime", lprime), ("lq", lq), ("log_l", ll)):
                out[k][b] = v
            out["logp"][b] = lp
        out["m"], out["log_l_final"] = m, ll
        return out

    def run(self, scores: np.ndarray) -> SoftmaxOnlineResult:
        cm = self.cycle_model()
        return SoftmaxOnlineResult(values=self.values(scores), compute_cycles=cm.total,
                                   load_words=self.stream_words(), cycle_model=cm)

    def gate1(self, scores: np.ndarray) -> dict:
        """GATE 1: sim values == oracle on every stage (raises on mismatch)."""
        v = self.values(scores)
        o = oref.logcarry_stage_values(scores, self.Bkv)
        for k in STAGE_KEYS:
            if not np.array_equal(np.asarray(v[k]), np.asarray(o[k])):
                raise AssertionError(f"GATE 1 mismatch on {k} (S={self.S}, N={self.N}, Bkv={self.Bkv})")
        return v

    # -- stimulus dump (GATE 1 + GATE 2 expected files) ---------------------
    def dump_case(self, case_dir: Path, scores: np.ndarray, cls: str = "") -> None:
        """GATE 1 first (sim == oracle on every stage), then write the case.

        File contract (consumed by `v2/sim/tb/tb_softmax_online_top.v`):
          scores.mem     block-major score stream, one 40-bit word per line (10 hex);
                         block b = scores[:, b*Bkv:(b+1)*Bkv] row-major, padded to
                         whole words
          exp_logp.mem   log|P| per element, block-major element order (S*N bytes)
          exp_la.mem     log_a per (block, row), 9-bit two's complement (3 hex)
          exp_sp.mem     sp per (block, row) (8 hex)
          exp_e.mem      carry EXP byte per (block, row) (0 at block 0)
          exp_lq.mem     lq per (block, row)
          exp_ll.mem     log_l per (block, row)
          exp_bmax.mem   block max per (block, row)
          exp_m.mem      final running max per row
          exp_llf.mem    final log_l per row
          scores.npy     the int8 scores (re-certification)
          case.json      S, N, BKV, R, C, N_EXP, B, STREAM, NWORDS, cls,
                         compute_cycles (= expected_cycles), stall, load_words
        """
        scores = np.asarray(scores, dtype=np.int64)
        v = self.gate1(scores)                       # raises before any write
        S, N, Bkv, B = self.S, self.N, self.Bkv, self.B
        stream = _cdiv(S * Bkv, EPS)
        parts = []
        for b in range(B):
            blk = np.asarray(scores[:, b * Bkv:(b + 1) * Bkv], dtype=np.int8).reshape(-1)
            parts.append(np.concatenate([blk, np.zeros(stream * EPS - blk.size, dtype=np.int8)]))
        words = nref.pack_act_stream(np.concatenate(parts))
        assert len(words) == B * stream
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        with open(case_dir / "scores.mem", "w") as f:
            for w in words:
                f.write(f"{w:010x}\n")

        def write(name: str, vals, width: int = 2) -> None:
            mask = (1 << (4 * width)) - 1
            with open(case_dir / name, "w") as f:
                for x in np.asarray(vals).reshape(-1):
                    f.write(f"{int(x) & mask:0{width}x}\n")

        write("exp_logp.mem", v["logp"])                 # [B, S, Bkv] = block-major
        write("exp_la.mem", v["log_a"], width=3)
        write("exp_sp.mem", v["sp"], width=8)
        write("exp_e.mem", v["e"])
        write("exp_lq.mem", v["lq"])
        write("exp_ll.mem", v["log_l"])
        write("exp_bmax.mem", v["m_blk"])
        write("exp_m.mem", v["m"])
        write("exp_llf.mem", v["log_l_final"])
        np.save(case_dir / "scores.npy", np.asarray(scores, dtype=np.int8))
        cm = self.cycle_model()
        (case_dir / "case.json").write_text(json.dumps(dict(
            S=S, N=N, BKV=Bkv, R=self.R, C=self.C, N_EXP=self.n_exp, B=B, STREAM=stream,
            NWORDS=B * stream, cls=cls, compute_cycles=cm.total, expected_cycles=cm.total,
            stall=cm.stall, load_words=B * stream), indent=1) + "\n")


# ---------------------------------------------------------------------------
# Self-test — run:  python3 v2/sim/simulator/kernels/softmax_online_sim.py
# ---------------------------------------------------------------------------
# Cycles measured on the RTL (strict GATE 2, Delta_impl = 0): (S, N, Bkv, n_exp) -> done edge.
_RTL_MEASURED_PINS = {
    (8, 8, 8, 1): 170,
    (64, 64, 16, 2): 1299,
    (128, 128, 16, 4): 3649,
    (64, 256, 16, 2): 4575,          # also every run of the 3-run back-to-back case
    (256, 1024, 16, 4): 53021,
}


def _self_test() -> None:
    rng = np.random.default_rng(7)
    n = 0
    for S, N in ((16, 16), (32, 32), (64, 64), (16, 64), (8, 128), (32, 16), (64, 256)):
        for Bkv in [k for k in (8, 16, 32, 64) if k <= N]:
            for ne in (1, 2, 4):
                sim = NldpeSoftmaxOnline(S, n_exp=ne, Bkv=Bkv, N=N)
                for sc in (rng.integers(-128, 128, size=(S, N)), rng.integers(-16, 17, size=(S, N)),
                           np.zeros((S, N), dtype=np.int64)):
                    sim.gate1(sc)
                    n += 1
                cm = sim.cycle_model()
                assert cm.total > cm.stream_end and cm.carry_log_passes == N // Bkv
                assert cm.carry_exp_passes == N // Bkv - 1
                assert sum(cm.exp_passes) == (N // Bkv) * max(1, S * Bkv // 256)
                assert cm.blocks[0]["first_word"] == 2
    # Bkv = N: one block, the conventional softmax on N-key rows (oracle gate)
    for S, N in ((16, 16), (32, 64)):
        sc = rng.integers(-128, 128, size=(S, N))
        v = NldpeSoftmaxOnline(S, Bkv=N, N=N).gate1(sc)
        g = oref.softmax_rect(sc)
        assert np.array_equal(v["log_l_final"], g["log_output"]) and np.array_equal(v["m"], g["row_max"])
    for (S, N, Bkv, ne), want in _RTL_MEASURED_PINS.items():
        got = softmax_online_cycle_model(S, Bkv, n_exp=ne, N=N).total
        assert got == want, f"S={S} N={N} Bkv={Bkv} n_exp={ne}: model {got} != RTL {want}"
    for bad in ((16, 4), (16, 32), (24, 8), (4, 8), (512, 16)):
        try:
            softmax_online_cycle_model(*bad)
        except ValueError:
            continue
        raise AssertionError(f"domain guard missed {bad}")
    _test_pass_extent()
    print(f"softmax_online_sim self-test: ALL PASS ({n} GATE-1 runs: primitive-path values == oracle; "
          f"Bkv = N == conventional softmax; {len(_RTL_MEASURED_PINS)} RTL-measured cycle pins; "
          "cycle model structure; domain guards)")


def _test_pass_extent() -> None:
    """P29 extent: default == certified (RTL pins untouched); 'used' == the same run on an I x I
    array; never slower; pass counts unchanged."""
    for (S, N, Bkv, ne) in _RTL_MEASURED_PINS:
        assert softmax_online_cycle_model(S, Bkv, n_exp=ne, N=N, pass_extent="full").total == \
            _RTL_MEASURED_PINS[(S, N, Bkv, ne)]
    for R, C in [(64, 64), (128, 64), (128, 256), (256, 256), (512, 128), (1024, 64)]:
        I = min(R, C)
        for S in [s for s in (8, 16, 64, 128) if s <= I]:
            for N in (S, 4 * S):
                for Bkv in [k for k in (8, 16) if k <= min(N, I)]:
                    for ne in (1, 4):
                        full = softmax_online_cycle_model(S, Bkv, R, C, n_exp=ne, N=N)
                        used = softmax_online_cycle_model(S, Bkv, R, C, n_exp=ne, N=N, pass_extent="used")
                        small = softmax_online_cycle_model(S, Bkv, I, I, n_exp=ne, N=N)
                        assert used.total == small.total and used.total <= full.total
                        assert (used.exp_passes, used.carry_exp_passes, used.carry_log_passes) == \
                            (full.exp_passes, full.carry_exp_passes, full.carry_log_passes)
                        if R == C:
                            assert used.total == full.total
    try:
        softmax_online_cycle_model(16, 8, pass_extent="half")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown pass_extent must be rejected")


if __name__ == "__main__":
    _self_test()
