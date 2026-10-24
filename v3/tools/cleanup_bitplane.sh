#!/usr/bin/env bash
# cleanup_bitplane.sh — remove the files the bit-plane pass made stale (2026-10-23; LEDGER log). The agent cannot
# delete files; you run this. Without --go it only lists what it would remove (a dry run).
#   stale since F3/F4:   att_layer (both twins, its testbench, runner, baseline), att_ocarry_top (both twins, its
#                        testbench, scripts, baseline), the obsolete v2-interface softmax harness scripts
#   retired 2026-10-23:  the crossbar's pre-bit-plane test (tb_dpe_nldpe.v, run_dpe_rtl.py, gen_cases.py) and the GEMM's v2
#                        harness scripts (run_gemm_rtl.py, gen_gemm_cases.py); the GEMM's GATE-2 harness replaces them
# Their tools/copies.py entries are already in FORKED (no copy check needs the files), so the lock gate passes
# before and after this script. Usage: bash v3/tools/cleanup_bitplane.sh [--go]
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES=(
  v3/top/rtl/sim/att_layer.v v3/top/rtl/vtr/att_layer.v v3/top/tb/tb_att_layer.v v3/top/tb/run_layer_rtl.py
  v3/tools/twin_baseline/att_layer.diff
  v3/ip/softmax/rtl/sim/att_ocarry_top.v v3/ip/softmax/rtl/vtr/att_ocarry_top.v v3/ip/softmax/tb/tb_att_ocarry_top.v
  v3/ip/softmax/tb/gen_ocarry_cases.py v3/ip/softmax/tb/run_ocarry_rtl.py v3/tools/twin_baseline/att_ocarry.diff
  v3/ip/softmax/tb/gen_softmax_online_cases.py v3/ip/softmax/tb/run_softmax_online_rtl.py
  v3/ip/xbar/tb/tb_dpe_nldpe.v v3/ip/xbar/tb/run_dpe_rtl.py v3/ip/xbar/tb/gen_cases.py
  v3/ip/gemm/tb/run_gemm_rtl.py v3/ip/gemm/tb/gen_gemm_cases.py
)
GO=0
[[ "${1:-}" == "--go" ]] && GO=1
for f in "${FILES[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "absent   $f"
  elif git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    if (( GO )); then git rm -q "$f" && echo "git rm   $f"; else echo "would git rm  $f"; fi
  else
    if (( GO )); then rm -f "$f" && echo "rm       $f"; else echo "would rm      $f"; fi
  fi
done
(( GO )) || echo "dry run: nothing removed; run again with --go"
