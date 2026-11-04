#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# PRIOR TEST ON THE DESIGN OF BAAZ ET AL.: the EB-VAE with a prior mean that ignores the
# first amount (BAAZ_PRIOR_NO_DOSE=1, both training stages; opt-in switch added to $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/run_regimen_titrated.py,
# backup .bak_priornodose), sparse sampling, linear simulator: randomised arms s0-4 and titrated arms (rho 0.8) s0-2, paired with the
# Table 3 runs (res_titr/{ctrl,titr0.8}); then the control sparse s1 rerun UNCHANGED (default path + run-to-run noise) ->
# res_titr/repro/ctrl. Command = queue_claim5.sh's (cwd $REPO_ROOT/third_party/baaz_ebvae, run_regimen_titrated.py 100 sparse <s> <none|0.8> <out>).
# Gate <= 8 python3 jobs (cap 800 %), re-checked after 41 s. Table scripts/regimen/eb_priortest_baaz_table.py ->
# results/regimen_dh/eb_priortest_baaz_table.txt. Marker in logs/queue_eb_priortest_baaz.log: EBPRIORBAAZ_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; EB=$REPO_ROOT/third_party/baaz_ebvae; RT=$EB/ours/regimen/res_titr; L=logs/regimen_dh; mkdir -p $L
CAP=8
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge $CAP ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt $CAP ] && return; done; }
busy () { [ -f "$1" ] || pgrep -f -- "$1" > /dev/null; }
run () {   # seed rho out log env
  if busy $3; then say "skip $3"; return; fi
  gate; mkdir -p $(dirname $3)
  ( cd $EB && env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 $5 nohup nice -n 5 $PY -u ours/regimen/run_regimen_titrated.py \
      100 sparse $1 $2 $3 > $REPO_ROOT/$L/$4.log 2>&1 < /dev/null & )
  say "start $4"; sleep 20
}
for s in 0 1 2 3 4; do run $s none $RT/prior_nodose/ctrl/N100_sparse_s$s.json eb_priornodose_baaz_ctrl_s$s BAAZ_PRIOR_NO_DOSE=1; done
for s in 0 1 2; do run $s 0.8 $RT/prior_nodose/titr0.8/N100_sparse_s$s.json eb_priornodose_baaz_titr_s$s BAAZ_PRIOR_NO_DOSE=1; done
run 1 none $RT/repro/ctrl/N100_sparse_s1.json eb_repro_baaz_ctrl_s1 ""
sleep 60
while pgrep -f "run_regimen_titrated.py 100 sparse .*res_titr/(prior_nodose|repro)/" > /dev/null; do sleep 60; done
$PY scripts/regimen/eb_priortest_baaz_table.py > /dev/null 2>&1; say "table -> results/regimen_dh/eb_priortest_baaz_table.txt"
say "EBPRIORBAAZ_DONE"
