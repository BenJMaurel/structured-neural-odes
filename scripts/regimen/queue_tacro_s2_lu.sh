#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Lu et al. on the LINEAR tacrolimus regimen benchmark (scenario 2), sparse (6 samples), seeds 0-4: a main-text figure
# comparing patient predictions of the dose-event model, the EB-VAE and Lu for the 6-sample linear tacrolimus design; Lu had only been
# run on scenario 4). Command = queue_tacro_versatility.sh's (lu_regimen.py --sim tacro, script defaults) with TACRO_SCENARIO=2, as in
# queue_tacro_s2.sh -> results/regimen_dh/tacro_s2/{luis,lunmi}/. Gate <= 8 python3 jobs (cap 800 %). Marker TACROS2LU_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh/tacro_s2; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 8 ]; do sleep 60; done; sleep 23; [ "$(njobs)" -lt 8 ] && return; done; }
for m in is nmi; do
  for s in 0 1 2 3 4; do
    o=$R/lu$m/N100_sparse_s$s.json
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate; mkdir -p $R/lu$m
    env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py $m 100 sparse $s $o --iters 6000 --sim tacro \
        > $L/tacro_s2_lu${m}_sparse_s$s.log 2>&1 < /dev/null &
    say "start tacro_s2_lu${m}_sparse_s$s"; sleep 5
  done
done
sleep 30; while pgrep -f "lu_regimen.py .*regimen_dh/tacro_s2/lu" > /dev/null; do sleep 30; done
say "TACROS2LU_DONE"
