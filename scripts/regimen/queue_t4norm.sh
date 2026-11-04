#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Table 4: the dose-event model = free-flow B, jump a*b(p), noise fixed 20 mg/L, DOSE-NORMALISED
# encoder (Bfixkp20n; sparse s0-4 already exist). Fill its dense s0-2 and titrated (rho 0.8) sparse s0-2, and the dose-conditioned
# field with the SAME encoder (--enc-dose norm): sparse s0-4, dense s0-2, titrated sparse s0-2. Linear simulator only.
# Gate <= 6 python3 jobs on the machine (600 % cap), idle multiprocessing helpers not counted. Marker T4NORM_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while [ "$(njobs)" -ge 6 ]; do sleep 60; done; }
B="--obsrv-mgl 20 --kick-phys 32 --enc-dose norm"; DC="--obsrv-mgl 20 --dose-cond --enc-dose norm"
J=()
for s in 0 1 2 3 4; do J+=("sparse $s $R/dosecond_n/N100_sparse_s$s.json dcn_sparse_s$s $DC"); done
for s in 0 1 2; do J+=("dense $s $R/Bfixkp20n/N100_dense_s$s.json Bn_dense_s$s $B"); done
for s in 0 1 2; do J+=("sparse $s $R/titr0.8/Bfixkp20n/N100_sparse_s$s.json Bn_titr_sparse_s$s $B --titrate 0.8"); done
for s in 0 1 2; do J+=("dense $s $R/dosecond_n/N100_dense_s$s.json dcn_dense_s$s $DC"); done
for s in 0 1 2; do J+=("sparse $s $R/dosecond_n/titr0.8/N100_sparse_s$s.json dcn_titr_sparse_s$s $DC --titrate 0.8"); done
for j in "${J[@]}"; do
    set -- $j; d=$1 s=$2 o=$3 lg=$4; shift 4
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate; mkdir -p "$(dirname $o)"
    env OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s "$o" --iters 6000 "$@" > "$L/$lg.log" 2>&1 < /dev/null &
    say "start $lg"; sleep 20
done
while pgrep -f "dh_regimen.py 100 .*(dosecond_n|Bfixkp20n)" > /dev/null; do sleep 60; done
say "T4NORM_DONE"
