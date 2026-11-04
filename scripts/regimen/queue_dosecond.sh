#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Table 4 restructure: EB-VAE vs free-flow B (iv) vs a DOSE-CONDITIONED field, linear simulator.
#   dose-cond: dh_regimen.py --dose-cond --obsrv-mgl 20 (same encoder / latents / decoder / noise / 6000 it as (iv), no kick)
#              sparse seeds 0-4, dense 0-2, titrated (rho 0.8) sparse 0-2      -> results/regimen_dh/dosecond[/titr0.8]/
#   free-flow B (iv) on titrated arms: --obsrv-mgl 20 --kick-phys 32 --titrate 0.8, sparse 0-2 -> results/regimen_dh/titr0.8/Bfixkp20/
# Gate: <= 6 running python3 jobs on the machine (600 % cap), idle multiprocessing helpers not counted.
# Log logs/queue_dosecond.log, marker DOSECOND_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}
R=results/regimen_dh; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while [ "$(njobs)" -ge 6 ]; do sleep 60; done; }
J=()
for s in 0 1 2 3 4; do J+=("sparse $s $R/dosecond/N100_sparse_s$s.json dc_sparse_s$s --obsrv-mgl 20 --dose-cond"); done
for s in 0 1 2; do J+=("sparse $s $R/dosecond/titr0.8/N100_sparse_s$s.json dc_titr_sparse_s$s --obsrv-mgl 20 --dose-cond --titrate 0.8"); done
for s in 0 1 2; do J+=("sparse $s $R/titr0.8/Bfixkp20/N100_sparse_s$s.json Bfree_titr_sparse_s$s --obsrv-mgl 20 --kick-phys 32 --titrate 0.8"); done
for s in 0 1 2; do J+=("dense $s $R/dosecond/N100_dense_s$s.json dc_dense_s$s --obsrv-mgl 20 --dose-cond"); done
for j in "${J[@]}"; do
    set -- $j; d=$1 s=$2 o=$3 lg=$4; shift 4
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate; mkdir -p "$(dirname $o)"
    env OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s "$o" --iters 6000 "$@" > "$L/$lg.log" 2>&1 < /dev/null &
    say "start $lg"; sleep 20
done
while pgrep -f "dh_regimen.py 100 .*(dosecond|Bfixkp20)" > /dev/null; do sleep 60; done
say "DOSECOND_DONE"
