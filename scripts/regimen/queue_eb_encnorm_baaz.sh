#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# NORMALISATION TEST ON THE DESIGN OF BAAZ ET AL.: the EB-VAE with
# its stage-2 encoder input divided by the patient's first amount (and the global mean concentration) instead of by the stage-1 population
# prediction under the patient's own regimen (BAAZ_ENC_NORM=dose; opt-in switch in $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/run_regimen_titrated.py,
# backup .bak_encnorm); prior unchanged (dose-conditioned). Sparse, linear: randomised s0-4, titrated (rho 0.8) s0-2 -> res_titr/encnorm_dose/.
# Same command as queue_eb_priortest_baaz.sh. Gate <= 8 python3 jobs. Table scripts/regimen/eb_priortest_baaz_table.py.
# Marker in logs/queue_eb_encnorm_baaz.log: EBENCNORM_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; EB=$REPO_ROOT/third_party/baaz_ebvae; RT=$EB/ours/regimen/res_titr; L=logs/regimen_dh; mkdir -p $L
CAP=8
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge $CAP ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt $CAP ] && return; done; }
busy () { [ -f "$1" ] || pgrep -f -- "$1" > /dev/null; }
run () {   # seed rho out log
  if busy $3; then say "skip $3"; return; fi
  gate; mkdir -p $(dirname $3)
  ( cd $EB && env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BAAZ_ENC_NORM=dose nohup nice -n 5 $PY -u ours/regimen/run_regimen_titrated.py \
      100 sparse $1 $2 $3 > $REPO_ROOT/$L/$4.log 2>&1 < /dev/null & )
  say "start $4"; sleep 20
}
for s in 0 1 2; do run $s 0.8 $RT/encnorm_dose/titr0.8/N100_sparse_s$s.json eb_encnorm_baaz_titr_s$s; done   # titrated first: the decisive cells
for s in 0 1 2 3 4; do run $s none $RT/encnorm_dose/ctrl/N100_sparse_s$s.json eb_encnorm_baaz_ctrl_s$s; done
sleep 60
while pgrep -f "run_regimen_titrated.py 100 sparse .*res_titr/encnorm_dose/" > /dev/null; do sleep 60; done
$PY scripts/regimen/eb_priortest_baaz_table.py > /dev/null 2>&1; say "table -> results/regimen_dh/eb_priortest_baaz_table.txt"
say "EBENCNORM_DONE"
