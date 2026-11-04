#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Lu et al. on the LINEAR tacrolimus regimen benchmark (scenario 2), intensive-PK-visit design (12 samples), seeds 0-4:
# Fig. 6 with the 12-sample instead of the 6-sample regimen). iScience: script defaults. NMI: --euler-dt 1/12 (opt-in added 19/10,
# backup lu_regimen.py.bak_eulerdt; default 0.25-h path bit-identical) because the 20/40-min samples are off the 0.25-h grid.
# -> results/regimen_dh/tacro_s2/{luis,lunmi}/N100_visit_s*.json. Gate <= 8 python3 jobs. Marker TACROS2LUVISIT_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh/tacro_s2; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 8 ]; do sleep 60; done; sleep 23; [ "$(njobs)" -lt 8 ] && return; done; }
for m in is nmi; do
  extra=""; [ $m = nmi ] && extra="--euler-dt 0.0833333333333333"
  for s in 0 1 2 3 4; do
    o=$R/lu$m/N100_visit_s$s.json
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate; mkdir -p $R/lu$m
    env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py $m 100 visit $s $o --iters 6000 --sim tacro $extra \
        > $L/tacro_s2_lu${m}_visit_s$s.log 2>&1 < /dev/null &
    say "start tacro_s2_lu${m}_visit_s$s"; sleep 5
  done
done
sleep 30; while pgrep -f "lu_regimen.py .*100 visit" > /dev/null; do sleep 30; done
say "TACROS2LUVISIT_DONE"
