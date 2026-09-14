# SPEC — Behavioral ground truth (v2 clean-room flow)

> **Status: v2 clean-room flow active.** First live spec:
> [`v2/spec/dpe_nldpe.md`](../v2/spec/dpe_nldpe.md) — **v1.1 amended 2026-09-12**
> (fp32 dataflow: fp32 weights/crossbar, structural fp32 MAC, trunc8 ACAM,
> WEIGHT strobe; decisions P1–P19 closed; supersedes v1.0 FROZEN 2026-08-29).
> Amendments require a new revision + decision-log entry + oracle
> re-transcription.
>
> The legacy generated RTL (`rtl_flow/rtl/`, `rtl_flow/tb/`, `rtl_flow/smoke/`)
> is frozen as reference — read-only, never copied into v2 mid-flight. The
> legacy Python oracle (`smoke/oracles/nldpe_mac_oracle.py`) encodes *legacy*
> semantics (double-buffer cycles, no weight interface, Taylor ACAM modes) and
> is a structural reference only — the v2 oracle is a fresh transcription of
> the frozen spec.
>
> Ladder position: Stage 1.2 (v2 NumPy oracle, re-transcribed to v1.1) next →
> skeleton → user hand-writes RTL → cross-check (v2 ≡ oracle, legacy as
> witness).
