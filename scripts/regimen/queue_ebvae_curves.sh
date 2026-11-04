#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# EB-VAE reruns with predicted-curve dumps (DUMP_CURVES=1) for the patient-level figures (scripts/regimen/plot_examples.py):
# N100 sparse seed 1, randomised + titrated (rho 0.8). Deterministic: the JSONs must equal res_titr/{ctrl,titr0.8}/N100_sparse_s1.json.
# Same gate as queue_fair.sh (<= 10 heavy python jobs incl. other sessions, CPU < 950 %). Log logs/queue_ebvae_curves.log, marker EBCURVES_DONE.
cd $REPO_ROOT; PY=${PYTHON:-python3}; EB=$REPO_ROOT/third_party/baaz_ebvae; O=$EB/ours/regimen/res_curves
mkdir -p $O; LOG=logs/queue_ebvae_curves.log; say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|train_|dh_regimen|lu_regimen|run_regimen|scripts/theory/"; }
cpu () { ps -A -o pcpu= | awk '{s+=$1} END {printf "%d", s}'; }
gate () { while [ $(running) -ge 10 ] || [ $(( $(cpu) + 100 )) -gt 950 ]; do sleep 20; done; }
P=""
for rho in none 0.8; do
  gate
  ( cd $EB && DUMP_CURVES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg nohup nice -n 5 $PY -u ours/regimen/run_regimen_titrated.py \
      100 sparse 1 $rho $O/N100_sparse_s1_$rho.json > $OLDPWD/logs/regimen_dh/ebcurves_$rho.log 2>&1 ) &
  P="$P $!"; say "start rho $rho (pid $!)"; sleep 20
done
wait $P; say "EBCURVES_DONE"
