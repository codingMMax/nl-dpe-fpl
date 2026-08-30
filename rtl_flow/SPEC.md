# SPEC — Behavioral Charter (primitive ground truth)

> **Status: PLACEHOLDER — Stage 1.1 in progress.**
> This file will become the normative behavioral charter for every primitive:
> FSM, LOAD/COMPUTE/OUTPUT cadence, double-buffer semantics, precision/ACAM
> behavior, port contracts, and I/O formats (log-domain conventions,
> K-identity patterns).
>
> Until the charter lands, the descriptive reference is
> `docs/DPE_PRIMITIVE_WALKTHROUGH.md` + `docs/DPE_NLDPE_FAITHFUL_WALKTHROUGH.md`
> — **descriptive, not yet normative** (they describe what the code does today,
> which is exactly what is under audit).

## Open decisions (owner: project lead) — to be resolved before the charter is normative

| # | Decision | Default proposal | Status |
|---|----------|------------------|--------|
| D1 | Which model tier is ground truth | Faithful primitives (`dpe_*_faithful.v`); `dpe_*.v` stubs are VTR interface contracts only | open |
| D2 | Intended LOAD/COMPUTE/OUTPUT cadence | LCYC/CCYC/OCYC from `specs/*.json`; CCYC structural via double-buffer | open |
| D3 | Double-buffer overlap policy | Pass-(k+1) LOAD fills substrate B while pass-k COMPUTE reads A; no LOAD gate | open |
| D4 | Output-domain convention | VMM: linear integers; softmax: log-domain `log p_i`; DIMM: log-domain in/out | open |
| D5 | Precision defaults | W=16 substrate, INT8 activations default, PRECISION-parameterized compute depth | open |

Each decision gets an explicit rationale line when closed; the charter cites
them as invariants.
