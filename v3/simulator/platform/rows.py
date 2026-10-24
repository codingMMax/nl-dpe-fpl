"""rows.py — a run priced into a DSE row (docs/experiment.md §5): cycles from the run, energy from the Report's counts
x the platform's prices, area from the crossbar count and the row's VTR metrics.

The column names are the DSE's (dse/results/fc_priced_results.csv, dse/results/attn_priced_results.csv; the energy
terms e_*_pj are the same in both): they are the data format of the results, not Python names.
"""
from __future__ import annotations

from simulator.platform.breakdown import energyByRole
from simulator.core.engine import run
from simulator.platform import area, prices
from simulator.platform.platform import Platform
from simulator.models.attentionHead import attentionHead, headParams
from simulator.models.fc import fc

UM2_PER_MWTA = 0.033864
ATTENTION_ROLES = ("gemm", "qk", "sm", "pv", "acc")      # the roles priced apart in an attention row


def fcRow(R: int, C: int, shape: tuple, platform: Platform, vtr: dict | None = None, passExtent: str = "used",
          tileRule: str = "row_parallel") -> dict:
    """One FC DSE row (timing only: the row's cycles and counts do not depend on the values, I4). tileRule: the crossbar
    tile's rule (area.footprint; "legacy" for the v2 regression tests)."""
    M, K, N = shape
    g = fc(K, N, R, C, M=M, platform=platform, passExtent=passExtent, withValues=False)
    rep = run(g)
    gemm = next(m for m in g.modules if getattr(m, "NAME", "") == "gemm_top")
    e = prices.energy(rep.totalCounts, platform)
    res = rep.resources["gemm"]
    row = dict(config=f"{R}x{C}", R=R, C=C, family="gemm", M=M, K=K, N=N, V=gemm.d.V, H=gemm.d.H,
               dpe_count=res.nCrossbars, T_cycles=rep.cycles, pass_extent=passExtent,
               passes=rep.totalCounts.nPasses, e_total_pj=e["e_total_pj"], e_pass_pj=e["e_pass_pj"],
               e_traffic_pj=e["e_traffic_pj"], e_adds_pj=e["e_adds_pj"], e_cmp_pj=e["e_cmp_pj"],
               e_mem_pj=e["e_mem_pj"], e_setup_pj=prices.setupEnergyPj(platform, res.nCrossbars),
               area_wc_mwta=area.areaWcMwta(res.nCrossbars, R, C, tileRule))
    m = (vtr or {}).get((row["config"], "gemm", M, K, N))
    if m is not None:
        row.update(area.withVtr(rep.cycles, R, C, m, tileRule))
        row["area_total_mm2"] = round(row["area_total_mwta"] * UM2_PER_MWTA / 1e6, 4)
        row["edap"] = round(e["e_total_pj"] * row["latency_ns"] * row["area_total_mm2"], 1)   # pJ x ns x mm^2
    return row


def attentionRow(name: str, nseq: int, dModel: int, d: int, R: int, C: int, vtr: dict | None = None,
                 tileRule: str = "row_parallel", nE: int | None = None) -> dict:
    """One attention DSE row at the fixed mapping (headParams defaults): the head's compute cycles (done - start),
    energy in total and per role, crossbars, the memory inventory and its predicted blocks; with VTR metrics also the
    total area, latency and EDAP (experiment.md §5). tileRule as fcRow."""
    hp = headParams(nseq, dModel, d, R, C, N_E=nE)       # nE None: the fixed mapping (N_E = d since 2026-10-23)
    g, parts = attentionHead(hp, withValues=False)
    rep = run(g)
    platform = Platform("nl_dpe", rows=R, cols=C)
    ctrl = parts["ctrl"]
    T = ctrl.done - ctrl.tStart
    e = prices.energy(rep.totalCounts, platform)
    nXbar = rep.totalResources.nCrossbars
    memories = [(words, bits, k) for r in rep.resources.values() for (words, bits), k in r.memories.items()]
    row = dict(shape=name, n=nseq, d_model=dModel, d=d, config=f"{R}x{C}", R=R, C=C, R_b=hp.RB, V=hp.V,
               P_PROJ=hp.P_PROJ, N_E=hp.N_E, N_EP=hp.N_EP, N_EXP=hp.N_EXP, N_OC=hp.N_OC, B_kv=hp.BKV,
               T_cycles=T, crossbars=nXbar, passes=rep.totalCounts.nPasses,
               e_total_pj=round(e["e_total_pj"], 2), e_pass_pj=round(e["e_pass_pj"], 2),
               e_traffic_pj=round(e["e_traffic_pj"], 2), e_adds_pj=round(e["e_adds_pj"], 2),
               e_cmp_pj=round(e["e_cmp_pj"], 2), e_mem_pj=round(e["e_mem_pj"], 2),
               e_setup_pj=round(prices.setupEnergyPj(platform, nXbar), 2),
               area_wc_mwta=area.areaWcMwta(nXbar, R, C, tileRule),
               mem_bits_model=sum(words * bits * k for words, bits, k in memories),
               mem_max_words=max(words for words, _, _ in memories),
               mem_blocks_model=area.predictedMemBlocks(rep.resources.values(), R, C))
    row["area_mem_model_mwta"] = row["mem_blocks_model"] * area.MEM_TILE_CELLS * area.MWTA_PER_CELL
    byRole = energyByRole(rep, platform)
    for role in ATTENTION_ROLES:
        row[f"e_{role}_pj"] = round(byRole.get(role, {}).get("e_total_pj", 0), 2)
    if vtr and (name, R, C) in vtr:
        m = vtr[(name, R, C)]
        row.update(area.withVtr(T, R, C, m, tileRule))
        row["area_total_mm2"] = round(row["area_total_mwta"] * UM2_PER_MWTA / 1e6, 4)
        row["edap"] = round(e["e_total_pj"] * row["latency_ns"] * row["area_total_mm2"], 1)   # pJ x ns x mm^2
        row["vtr_seeds"] = m.get("seeds", "")
        row.update({f"fmax_s{s}": f for s, f in m.get("fmax_seed", {}).items()})       # for the seed resamples
        row["mem_check"] = "equal" if row["mem_blocks"] == row["mem_blocks_model"] else "DIFFERS"
    return row
