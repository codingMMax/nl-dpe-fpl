#!/bin/bash
# run_all.sh — the DSE pipeline of v3/dse/README.md, from the priced rows to the pages. The two VTR tables
# (results/fc_vtr_results.csv, results/attn_vtr_results.csv) are inputs. Stops at the first failing step.
set -euo pipefail
cd "$(dirname "$0")/../.."                                 # the repository root
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/v3/work/tmp
mkdir -p "$TMPDIR"
MPL=${MPL_PYTHON:-/mnt/vault0/jiajunh5/miniconda3/bin/python}   # the Python that has matplotlib
python3 v3/dse/fc_rows.py v3/dse/results/fc_priced_results.csv --vtr v3/dse/results/fc_vtr_results.csv
python3 v3/dse/attn_rows.py v3/dse/results/attn_priced_results.csv --vtr v3/dse/results/attn_vtr_results.csv
# python3 v3/dse/check_energy.py                      # the two priced tables against an independent pricing of the ledger
python3 v3/dse/rank.py
python3 v3/dse/breakdown.py
"$MPL" v3/dse/plot_dse.py
python3 v3/dse/results_page.py
echo "run_all: done -> v3/dse/results/"
