#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# "NLME with the wrong structure" row for the scenario-4 benchmark (Table 3 of ndm.tex), 2026-10-02.
# Same 100 training patients as Table 3 (confound_vc00_s4_pnoise_win_n100, 2 visits each). Monolix fits with the
# modeller covariate model (fit_s4_select.R, variant-l1 covariates, published starting values):
#   LINEAR elimination  -> results/monolix/sel_lin_pnoise_win_n100/estimates.json   (the new row)
#   Michaelis-Menten    -> results/monolix/sel_mm_pnoise_win_n100/estimates.json    (AIC/BIC: would a modeller pick linear?)
# then MAP-BE with the linear model on the 1100 test patients (ebe_popmodel_lin.py), V1 + V2 in / lo / hi ->
# results/s4/ebe/popebe_mlx_lin_pnoise_win_n1100.csv. Log logs/chain_lin_s4.log, marker LIN_S4_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; CAP=${CAP:-1000}; D=results/monolix/data/confound_vc00_s4_pnoise_win_n100.csv
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
gate () { while [ $(( $(cpu) + 200 )) -gt "$CAP" ]; do sleep 60; done; }
say () { echo "[$(date '+%F %T')] $*"; }
for s in lin mm; do
  O=results/monolix/sel_${s}_pnoise_win_n100
  [ -f $O/estimates.json ] && { say "$s fit exists"; continue; }
  gate; mkdir -p $O
  nice -n 5 Rscript scripts/monolix/fit_s4_select.R $D $O 2 $s > $O/fit.log 2>&1 &
  eval "P_$s=$!"; say "$s fit launched (pid $!)"; sleep 60
done
wait ${P_lin:-}; grep -q FIT_DONE results/monolix/sel_lin_pnoise_win_n100/fit.log || { say "LINEAR FIT FAILED"; exit 1; }
say "linear fit done; MAP-BE on 1100 test patients"
EBE_COHORT=confound_vc00_s4_pnoise_win OMP_NUM_THREADS=1 nice -n 5 $PY scripts/monolix/ebe_popmodel_lin.py \
  results/monolix/sel_lin_pnoise_win_n100/estimates.json 1100 0 results/s4/ebe/popebe_mlx_lin_pnoise_win_n1100.csv \
  > logs/popebe_mlx_lin_pnoise_win_n1100.log 2>&1 && say "MAP-BE done" || say "MAP-BE FAILED (logs/popebe_mlx_lin_pnoise_win_n1100.log)"
wait; say "mm fit: $(grep -c FIT_DONE results/monolix/sel_mm_pnoise_win_n100/fit.log 2>/dev/null) FIT_DONE"
say "LIN_S4_DONE"
