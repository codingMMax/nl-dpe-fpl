# Exploration scripts behind architecture.md (kept as evidence)

`pb_*.py` (the PV B credit and row-register sizing), `nep_*.py` (the PV farm size), `compute_cycles.py` and
`latency_terms.py` were written on 2026-10-14/15 against the pre-refactor simulator (git `816807b`: `sim/`,
`ip/*/model.py`). They import modules that no longer exist, so they do not run on the current tree; `architecture.md`
cites them for the numbers it quotes. To rerun one, check out `816807b`, or port it to `simulator/` the way
`../page_data.py` was ported (`headParams`, `attentionHead`, the dataclass `.d` / `.p`).
