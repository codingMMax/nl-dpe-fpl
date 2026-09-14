# SPEC — Behavioral ground truth (v2 clean-room flow)

> **Status: v2 clean-room flow active.** First live spec:
> [`v2/spec/dpe_nldpe.md`](../v2/spec/dpe_nldpe.md) — **v2.0 integer dataflow,
> 2026-09-14** (clean rewrite; supersedes the v1.1 fp32 amendment per advisor
> consultation): int8 weights/activations, exact integer MAC (int32
> accumulator), integer ACAM forms + `trunc8` low byte, WEIGHT strobe (int8).
> Decisions P1–P13, P16 and P21–P26 are live; P14–P20 retired. Amendments
> require a new revision + decision-log entry + oracle re-transcription.
>
> The legacy generated RTL (`rtl_flow/rtl/`, `rtl_flow/tb/`, `rtl_flow/smoke/`)
> is frozen as reference — read-only, never copied into v2 mid-flight. The
> legacy Python oracle (`smoke/oracles/nldpe_mac_oracle.py`) is a structural
> reference only; the v2 oracle is a fresh transcription of the frozen spec.
> With integer semantics the legacy arithmetic is numerically comparable again
> (values expected to match); schedule/buffer, weight-interface and
> ACAM-mode differences are reported, not gated.
>
> Ladder position: Stage 1.2 (v2 oracle + sim, integer v2.0) DONE → hand-written
> RTL → cross-check (v2 ≡ oracle, legacy as witness).
