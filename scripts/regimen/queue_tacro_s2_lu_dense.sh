#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Lu et al. (regimen model 'is' and two-port 'nmi') on the LINEAR tacrolimus benchmark, 61 samples (dense), seeds 0-2: the
# supplementary patient figure must show an experiment where the dose-event model is best -> 61-sample version; these runs also fill the
# missing Lu rows of Table S5, linear 61). Same command as queue_tacro_s2_lu.sh (script defaults, Euler step 0.25 h: the 0.75-h samples are
# on the grid). Marker TACROS2LUDENSE_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh/tacro_s2; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 8 ]; do sleep 60; done; sleep 23; [ "$(njobs)" -lt 8 ] && return; done; }
for m in is nmi; do
  for s in 0 1 2; do
    o=$R/lu$m/N100_dense_s$s.json
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate; mkdir -p $R/lu$m
    env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py $m 100 dense $s $o --iters 6000 --sim tacro \
        > $L/tacro_s2_lu${m}_dense_s$s.log 2>&1 < /dev/null &
    say "start tacro_s2_lu${m}_dense_s$s"; sleep 5
  done
done
sleep 30; while pgrep -f "lu_regimen.py .*100 dense" > /dev/null; do sleep 30; done
say "TACROS2LUDENSE_DONE"
