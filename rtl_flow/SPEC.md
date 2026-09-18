# SPEC — Behavioral ground truth (v2 clean-room flow)

> **Status: v2 clean-room flow active.** First live spec:
> [`v2/spec/dpe_nldpe.md`](../v2/spec/dpe_nldpe.md) — **v2.0.1 integer
> dataflow + interface freeze, 2026-09-15** (clean rewrite; supersedes the
> v1.1 fp32 amendment per advisor consultation): int8 weights/activations,
> exact integer MAC (int32 accumulator), integer ACAM forms + `trunc8` low
> byte, WEIGHT strobe (int8). Decisions P1–P13, P16 and P21–P27 are live;
> P14–P20 retired. Amendments require a new revision + decision-log entry +
> oracle re-transcription.
>
> **Verification chain (trust flows downward only; the spec decides all
> disagreements):**
>
> ```
> spec v2.0.1 → oracle v2/oracle/nldpe_ref.py
>             →(GATE 1: per case, inside NldpeDpe.dump_case — sim ≡ oracle
>               bit-exact on int32 y, output bytes and §5.3 cycles; a case
>               that disagrees is never dumped)
>             → sim v2/sim/nldpe_sim.py
>             →(GATE 2: RTL ≡ dumped expected bits, dual compare P26; cycles
>               = §5.3 total + one invariant Δ_impl, gated by
>               v2/smoke/run_dpe_rtl.py + v2/tb/tb_dpe_nldpe.v)
>             → RTL v2/rtl/dpe_nldpe.v
> ```
>
> The RTL is **integer-only** — no fp; transcendentals/high precision stay in
> the oracle, the sim owns the quantize/`trunc8` boundary and dumps expected
> bits. The same policy extends to later stages (softmax, DIMM).
>
> The legacy generated RTL (`rtl_flow/rtl/`, `rtl_flow/tb/`, `rtl_flow/smoke/`)
> is frozen as reference — read-only, never copied into v2 mid-flight. The
> legacy Python oracle (`smoke/oracles/nldpe_mac_oracle.py`) is an independent
> integer witness (different code lineage); its values are expected to match
> (MAC, REGULAR/LOG), while schedule/buffer, weight-interface and ACAM-mode
> differences are reported, not gated.
>
> Ladder position: **Stage 1.5 COMPLETE (2026-09-17)** — hand-written integer
> RTL (`v2/rtl/dpe_nldpe.v`, structure/event-driven control channels) passes
> GATE 2 on 48/48 default cases + M∈{1,2,4,8} sweep + micro-geometries
> (Δ_impl = 0, T_steady steps 10/16/60/104 exact). The independent witness
> (`v2/smoke/legacy_witness.py`) confirms the legacy oracle and the frozen
> legacy RTL on identical stimulus (mode 0; legacy cadence 52 vs v2 60 is a
> documented differing witness). Next: Stage 2 — fc_top (VMM/projection).
