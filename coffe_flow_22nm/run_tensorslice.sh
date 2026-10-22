#!/bin/bash
# run_tensorslice.sh — launch the `tensorslice` COFFE run (interface-only `-hi` sizing) in the
# pristine clone. Output: work/tensorslice/report.txt (+ sizing/SPICE artifacts); stdout/meta in
# work/tensorslice/launch.log. COFFE deletes only the subdirectories of the output folder and
# rewrites report.txt, so config.yaml, README.md and launch.log survive a re-run.
set -u

FLOW=/mnt/vault0/jiajunh5/nl-dpe-fpl/coffe_flow_22nm
CLEAN=/mnt/vault0/jiajunh5/coffe_pristine
OUT="$FLOW/work/tensorslice"

source /mnt/vault0/jiajunh5/set_cad.sh
export EDI_HOME=/home/tools/cadence/INNOVUS211

{
  echo "=== command: python2 coffe.py -a 1 -d 2 -hi 100 $OUT/config.yaml"
  echo "=== cwd: $CLEAN"
  echo "=== coffe_commit: $(git -C "$CLEAN" rev-parse HEAD 2>&1)"
  echo "=== erf_line: $(grep -m1 'ERF_MAX_ITERATIONS' "$CLEAN/coffe/tran_sizing.py")"
  echo "=== start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/launch.log"

cd "$CLEAN"
python2 coffe.py -a 1 -d 2 -hi 100 "$OUT/config.yaml" >> "$OUT/launch.log" 2>&1
rc=$?

# COFFE's parameter checker exits 0 on validation errors -> treat leading ERROR as failure
if grep -q '^ERROR' "$OUT/launch.log"; then
  rc=1
fi
echo "=== exit_code: $rc | end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/launch.log"
exit "$rc"
