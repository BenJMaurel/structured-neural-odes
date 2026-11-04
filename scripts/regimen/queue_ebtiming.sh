#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# EB-VAE at UNSEEN DOSING TIMES: run_regimen_timing.py on
# N100 dense s0 + sparse s0-2, randomised arms. The 6 original regimens must equal res_titr/ctrl/*.json (same training).
# Gate: < 6 heavy python jobs incl. other sessions (CPU cap). Log logs/queue_ebtiming.log, marker EBTIMING_DONE.
cd $REPO_ROOT; PY=${PYTHON:-python3}; EB=$REPO_ROOT/third_party/baaz_ebvae; O=$EB/ours/regimen/res_timing
mkdir -p $O logs/regimen_dh; LOG=logs/queue_ebtiming.log; say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|train_|dh_regimen|lu_regimen|run_regimen|mapbe|scripts/theory/"; }
gate () { while [ $(running) -ge 6 ]; do sleep 20; done; }
P=""
for cfg in "dense 0" "sparse 0" "sparse 1" "sparse 2"; do
  set -- $cfg; gate
  ( cd $EB && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg nohup nice -n 5 $PY -u ours/regimen/run_regimen_timing.py \
      100 $1 $2 none $O/N100_$1_s$2.json > $OLDPWD/logs/regimen_dh/ebtiming_$1_s$2.log 2>&1 ) &
  P="$P $!"; say "start $1 s$2 (pid $!)"; sleep 20
done
wait $P; say "EBTIMING_DONE"
