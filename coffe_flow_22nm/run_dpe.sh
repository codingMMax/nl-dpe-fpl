#!/bin/bash
# run_dpe.sh <R> [fullr|bw] — launch a dpe COFFE run (interface-only `-hi` sizing) for crossbar
# row count R, representative C=64.
#
# Models:
#   fullr (default) — row-parallel interface: one data pin per crossbar row
#       num_gen_inputs  = R + 7 (7 fixed controls)
#       num_gen_outputs = C + 5 = 69 (status; inert — COFFE never reads it)
#       load/drain = ceil(8R/R) = 8 / ceil(8C/C) = 8 cycles (int8 bit-serial)
#   bw — proportional-bandwidth interface (earlier family)
#       B_in  = 40*R/128  -> num_gen_inputs  = B_in + 7
#       B_out = 40*C/64   -> num_gen_outputs = 45 (status; inert)
#       load/drain = 26 / 13 cycles
#
# The run dir + config.yaml are created if missing, seeded from the real 47-pin block
# (work/dpe_R128_C64/config.yaml); only name / num_gen_inputs / num_gen_outputs / paths change.
# COFFE deletes only the subdirectories of the output folder and rewrites report.txt, so
# config.yaml, README.md and launch.log survive a re-run.
#
# NOTE: hard-block names are all-lowercase (HSPICE folds netlist/measure names to lowercase in
# .lis output while COFFE looks them up with the configured case; uppercase breaks the sizing).
#
# usage: run_dpe.sh <R> [fullr|bw]      e.g. run_dpe.sh 256   |   run_dpe.sh 512 bw
set -u

usage() { echo "usage: run_dpe.sh <R> [fullr|bw]   (default model: fullr)" >&2; exit 2; }
[ $# -ge 1 ] && [ $# -le 2 ] || usage
R="$1"; MODEL="${2:-fullr}"
C=64

FLOW=/mnt/vault0/jiajunh5/nl-dpe-fpl/coffe_flow_22nm
CLEAN=/mnt/vault0/jiajunh5/coffe_pristine
SEED="$FLOW/work/dpe_R128_C64"

case "$MODEL" in
  fullr)
    OUT="$FLOW/work/dpe_full_R${R}_C64"
    N_IN=$(( R + 7 )); N_OUT=$(( C + 5 ))
    NAME="dpe_full_r${R}"
    DIRNAME="dpe_full_R${R}_C64"
    LOAD_CYC=8; DRAIN_CYC=8
    ;;
  bw)
    B_IN=$(( 40 * R / 128 ))
    if [ $(( B_IN * 128 )) -ne $(( 40 * R )) ]; then
      echo "ERROR: 40*$R/128 is not an integer (R=$R)" >&2; exit 2
    fi
    OUT="$FLOW/work/dpe_R${R}_C64"
    N_IN=$(( B_IN + 7 )); N_OUT=$(( 40 * C / 64 + 5 ))
    NAME="dpe_r${R}"
    DIRNAME="dpe_R${R}_C64"
    LOAD_CYC=$(( (8 * R + B_IN - 1) / B_IN ))
    DRAIN_CYC=$(( (8 * C + (40 * C / 64) - 1) / (40 * C / 64) ))
    ;;
  *)
    usage
    ;;
esac

# generate the run config from the 47-pin seed if missing (identity + paths only)
if [ ! -f "$OUT/config.yaml" ]; then
  mkdir -p "$OUT"
  sed -e "s|/work/dpe_R128_C64|/work/${DIRNAME}|g" \
      -e "s|^    name: dpe_r128$|    name: ${NAME}|" \
      -e "s|^    num_gen_inputs: 47$|    num_gen_inputs: ${N_IN}|" \
      -e "s|^    num_gen_outputs: 45$|    num_gen_outputs: ${N_OUT}|" \
      "$SEED/config.yaml" > "$OUT/config.yaml"
  echo "generated $OUT/config.yaml (model=${MODEL}, name=${NAME}, n_in=${N_IN}, n_out=${N_OUT})"
fi

source /mnt/vault0/jiajunh5/set_cad.sh
export EDI_HOME=/home/tools/cadence/INNOVUS211

{
  echo "=== run: ${DIRNAME} (model=${MODEL}, R=${R}, C=${C}, n_in=${N_IN}, n_out=${N_OUT}, load/drain=${LOAD_CYC}/${DRAIN_CYC})"
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
