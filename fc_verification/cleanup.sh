#!/usr/bin/env bash
#
# cleanup.sh — Delete deprecated artifacts and archive historical docs
#               under FIDELITY_METHODOLOGY.md re-org.
#
# Status:  drafted, NOT YET EXECUTED.
# Anchor:  fc_verification/FIDELITY_METHODOLOGY.md §7, §10 step 1
#
# What this does:
#   - Moves historical docs to fc_verification/archive/
#   - Deletes deprecated alignment JSONs, runners, TBs, RTL, generators
#   - Uses git mv / git rm so changes stage for review (git status to inspect)
#   - Does NOT commit; user runs `git status` and commits when satisfied
#
# What this does NOT do:
#   - Submodule revert (separate procedure, §9)
#   - Touch block_comp_apr_11/ (deferred — possibly separate cleanup pass)
#   - Touch the simulator submodule (azurelily/) — that's the revert step
#   - Touch any KEEP-list artifact in §7
#
# Usage:
#   bash fc_verification/cleanup.sh             # dry-run (default; prints plan)
#   bash fc_verification/cleanup.sh --execute   # actually run the operations
#

set -u
set -o pipefail

# ───────────────────────────────────────────────────────────────────────────
# Sanity checks
# ───────────────────────────────────────────────────────────────────────────
REPO_ROOT="/mnt/vault0/jiajunh5/nl-dpe-fpl"
if [[ "$(pwd)" != "$REPO_ROOT" ]]; then
    echo "ERROR: must be run from $REPO_ROOT (got $(pwd))" >&2
    exit 1
fi
if [[ ! -f fc_verification/FIDELITY_METHODOLOGY.md ]]; then
    echo "ERROR: FIDELITY_METHODOLOGY.md not found — wrong dir?" >&2
    exit 1
fi
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "ERROR: not in a git repo" >&2
    exit 1
fi

DRY_RUN=1
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=0
fi

run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] $*"
    else
        echo "[run]     $*"
        eval "$@"
    fi
}

ARCHIVE_DIR="fc_verification/archive"

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$ARCHIVE_DIR"
fi

# ───────────────────────────────────────────────────────────────────────────
# 1. ARCHIVE — historical docs (move to fc_verification/archive/)
# ───────────────────────────────────────────────────────────────────────────
echo "=== 1. ARCHIVE historical docs ==="
for f in \
    fc_verification/DIMM_pipeline_model_vs_rtl.md \
    fc_verification/VERIFICATION.md \
    fc_verification/AH_pipeline_schematic.md
do
    if [[ -f "$f" ]]; then
        run "git mv '$f' '$ARCHIVE_DIR/$(basename $f)'"
    fi
done

# ───────────────────────────────────────────────────────────────────────────
# 2. DELETE — alignment JSONs (known_deltas, expected_cycles, etc.)
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== 2. DELETE alignment JSONs ==="
for f in \
    fc_verification/phase2_known_deltas.json \
    fc_verification/phase3_known_deltas.json \
    fc_verification/phase5_known_deltas.json \
    fc_verification/phase7_known_deltas.json \
    fc_verification/expected_cycles.json \
    fc_verification/expected_counters.json \
    fc_verification/per_stage_targets.json \
    fc_verification/known_count_deltas.json \
    fc_verification/known_cycle_deltas.json \
    fc_verification/functional_whitelist.json
do
    if [[ -f "$f" ]]; then
        run "git rm -f '$f'"
    fi
done

# ───────────────────────────────────────────────────────────────────────────
# 3. DELETE — alignment-era runners and helpers
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== 3. DELETE alignment runners and helpers ==="
for f in \
    fc_verification/run_checks.py \
    fc_verification/run_fc_phase2.py \
    fc_verification/run_alignment.sh \
    fc_verification/gen_expected_cycles.py \
    fc_verification/check_vtr_resources.py \
    fc_verification/gen_dimm_vtr.py \
    fc_verification/gen_dimm_wrapper.py \
    fc_verification/gen_azurelily_dimm_top_vtr.py \
    fc_verification/gen_azurelily_dimm_vtr.py \
    fc_verification/gen_nldpe_dimm_top_vtr.py
do
    if [[ -f "$f" ]]; then
        run "git rm -f '$f'"
    fi
done

# ───────────────────────────────────────────────────────────────────────────
# 4. DELETE — testbenches (rebuilt under new methodology)
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== 4. DELETE testbenches ==="
# Glob: fc_verification/tb_*.v
for f in fc_verification/tb_*.v; do
    if [[ -f "$f" ]]; then
        run "git rm -f '$f'"
    fi
done

# ───────────────────────────────────────────────────────────────────────────
# 5. DELETE — current dpe_stub.v (replaced by generator)
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== 5. DELETE current dpe_stub.v (will be regenerated) ==="
if [[ -f fc_verification/dpe_stub.v ]]; then
    run "git rm -f 'fc_verification/dpe_stub.v'"
fi

# ───────────────────────────────────────────────────────────────────────────
# 6. DELETE — alignment-era RTL in fc_verification/rtl/
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== 6. DELETE alignment-era RTL ==="
# Top-level fc_verification/rtl/*.v
for f in \
    fc_verification/rtl/azurelily_attn_head_d64_c128.v \
    fc_verification/rtl/azurelily_dimm_int_sop_4_stub.v \
    fc_verification/rtl/azurelily_dimm_mac_qk_d64_c128.v \
    fc_verification/rtl/azurelily_dimm_mac_sv_d64_c128.v \
    fc_verification/rtl/azurelily_dimm_softmax_d64_n128.v \
    fc_verification/rtl/azurelily_dimm_top_d64_c128.v \
    fc_verification/rtl/dimm_pipeline_d64_c128.v \
    fc_verification/rtl/dimm_score_n4d4.v \
    fc_verification/rtl/dimm_score_n4d4_single.v \
    fc_verification/rtl/dimm_score_n4d5_single.v \
    fc_verification/rtl/dimm_score_n4d8_single.v \
    fc_verification/rtl/fc_128_128_1024x128_acam_dw40.v \
    fc_verification/rtl/nldpe_attn_head_d64_c128.v \
    fc_verification/rtl/nldpe_dimm_mac_sv_d64_c128.v \
    fc_verification/rtl/nldpe_dimm_softmax_norm_d64_c128.v \
    fc_verification/rtl/nldpe_dimm_top_d64_c128.v
do
    if [[ -f "$f" ]]; then
        run "git rm -f '$f'"
    fi
done

# fc_verification/rtl/azurelily/*.v
for f in fc_verification/rtl/azurelily/*.v; do
    if [[ -f "$f" ]]; then
        run "git rm -f '$f'"
    fi
done

# fc_verification/rtl/nldpe/*.v
for f in fc_verification/rtl/nldpe/*.v; do
    if [[ -f "$f" ]]; then
        run "git rm -f '$f'"
    fi
done

# ───────────────────────────────────────────────────────────────────────────
# 7. DELETE — alignment-era generators in nl_dpe/
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== 7. DELETE alignment-era generators in nl_dpe/ ==="
for f in \
    nl_dpe/gen_attention_wrapper.py \
    nl_dpe/gen_attention_gemm_wrapper.py \
    nl_dpe/gen_nldpe_attn_head_top.py \
    nl_dpe/gen_azurelily_attn_head_top.py \
    nl_dpe/gen_dimm_nldpe_top.py \
    nl_dpe/gen_dimm_nldpe_full.py \
    nl_dpe/gen_dimm_azurelily_top.py \
    nl_dpe/gen_dimm_azurelily_full.py \
    nl_dpe/gen_gemv_wrappers.py \
    nl_dpe/gen_gemm_wrapper.py \
    nl_dpe/gen_azurelily_fc_wrapper.py \
    nl_dpe/gen_fc_softmax_wrapper.py
do
    if [[ -f "$f" ]]; then
        run "git rm -f '$f'"
    fi
done

# ───────────────────────────────────────────────────────────────────────────
# 8. SUMMARY
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== Summary ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN — no changes made. Re-run with --execute to apply."
else
    echo "Cleanup complete. Review with:"
    echo "  git status"
    echo "  git diff --stat HEAD"
    echo
    echo "If satisfied, commit with:"
    echo "  git commit -m 'FIDELITY_METHODOLOGY: cleanup deprecated artifacts'"
    echo
    echo "If unhappy with anything, restore individual files with:"
    echo "  git checkout HEAD -- <path>"
    echo "  git restore --staged <path>"
fi

# ───────────────────────────────────────────────────────────────────────────
# Deferred / not handled by this script
# ───────────────────────────────────────────────────────────────────────────
echo
echo "=== Not handled by this script (separate ops) ==="
echo "  - Submodule revert: see FIDELITY_METHODOLOGY.md §9"
echo "  - block_comp_apr_11/rtl/setup{0..5}/fc_*.v: deferred,"
echo "    decide separately whether to delete or archive"
echo "  - nl_dpe/gen_bert_*.py and gen_dsp_gemv_wrapper.py: kept"
echo "    (BERT-Tiny benchmarks + Azure-Lily DSE flow continue)"
