#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# EB-VAE on scenario 2 of the tacrolimus regimen benchmark. queue_tacro_s2.sh never started these: its log path was
# RELATIVE (logs/regimen_dh) inside `cd $EB`, so every launch failed with "No such file or directory". Same command, absolute log path.
# Sparse s0-4, dense s0-2, curves dumped. Gate <= 6 python3 jobs (re-check after 41 s). Marker TACROS2EB_DONE.
EB=$REPO_ROOT/third_party/baaz_ebvae; PY=${PYTHON:-python3}; ER=$EB/ours/regimen/res_tacro_s2/ctrl; L=$REPO_ROOT/logs/regimen_dh
mkdir -p $ER $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
for job in "sparse 0" "sparse 1" "sparse 2" "sparse 3" "sparse 4" "dense 0" "dense 1" "dense 2"; do
    set -- $job; o=$ER/N100_$1_s$2.json
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate
    ( cd $EB && env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 DUMP_CURVES=1 nohup nice -n 5 $PY -u ours/regimen/run_regimen_tacro.py 100 $1 $2 none $o \
        > $L/tacro_s2_ebvae_$1_s$2.log 2>&1 < /dev/null & )
    say "start EB-VAE s2 $1 s$2"; sleep 20
done
sleep 30; while pgrep -f "res_tacro_s2/" > /dev/null; do sleep 60; done
say "TACROS2EB_DONE"
