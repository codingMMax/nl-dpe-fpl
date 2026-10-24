#!/usr/bin/env bash
# cleanup_workdir.sh — free v3/work/ before the bit-plane DSE sweep (2026-10-23, user: "clean up the rundir and get
# more space first"). Dry run by default: prints every target with its size. `--go` removes them.
#
# Removed:
#   work/vtr/dse/*                 the previous DSE run's 248 build folders (2026-10-15/16, pre-bit-plane RTL; their metrics
#                                  are dse/results/attn_vtr_results.csv and fc_vtr_results.csv) and the three stopped
#                                  256x128 smokes of 2026-10-23 (no result). The sweep writes new folders (new hashes).
#   work/vtr/smoke/<stopped runs>  the 2026-10-23 smokes that were stopped or aborted (the swappable-pin runs, the 64x64
#                                  starts). The three *_none folders (QK, PV, head at 128x64: the reference smokes) stay.
#   work/gate2, gate2_trim, gate2_trim2   earlier GATE-2 work folders (compiled testbenches, logs); gate2_qkd (the latest) stays.
#   work/tmp/vtr_dse_test_*        the driver's self-test scratch folders.
# Kept: work/corpus (the GATE-1/2 corpora; the v2 dpe/gemm corpora are F2's value witnesses), work/equivalence (the
# references), work/gate2_qkd, work/dse (run logs), work/vtr/smoke/*_none, work/tmp/*.py (today's patch scripts).
set -euo pipefail
cd "$(dirname "$0")/.."                      # v3/
GO=0; [ "${1:-}" = "--go" ] && GO=1
targets=(
  work/vtr/dse
  work/vtr/smoke/dimm_tile_top_qk_bert_tiny_128x64
  work/vtr/smoke/dimm_tile_top_pv_bert_tiny_128x64
  work/vtr/smoke/att_head_bert_tiny_128x64
  work/vtr/smoke/dimm_tile_top_qk_bert_tiny_64
  work/vtr/smoke/dimm_tile_top_pv_bert_tiny_64
  work/vtr/smoke/att_head_bert_tiny_64
  work/vtr/smoke/head_bert_tiny_64.out
  work/vtr/smoke/head_bert_tiny_128x64.out
  work/vtr/smoke/dimm_qk_bert_tiny_64.out
  work/vtr/smoke/dimm_pv_bert_tiny_64.out
  work/vtr/smoke/dimm_qk_bert_tiny_128x64.out
  work/vtr/smoke/dimm_pv_bert_tiny_128x64.out
  work/gate2
  work/gate2_trim
  work/gate2_trim2
)
total=0
for t in "${targets[@]}" work/tmp/vtr_dse_test_*; do
  [ -e "$t" ] || continue
  case "$t" in work/*) ;; *) echo "refusing $t (outside work/)"; exit 1;; esac
  kb=$(du -sk "$t" | cut -f1); total=$((total + kb))
  if [ $GO = 1 ]; then rm -rf "$t"; echo "removed  $(numfmt --to=iec --from-unit=1024 "$kb")  $t"
  else echo "would remove  $(numfmt --to=iec --from-unit=1024 "$kb")  $t"; fi
done
echo "total $(numfmt --to=iec --from-unit=1024 "$total")  ($([ $GO = 1 ] && echo removed || echo 'dry run; add --go'))"
[ $GO = 1 ] && mkdir -p work/vtr/dse && df -h /mnt/vault0 | tail -1 | awk '{print "disk free now", $4}'
exit 0
