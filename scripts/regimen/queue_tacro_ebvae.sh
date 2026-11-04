#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# EB-VAE on the tacrolimus transposition of the regimen benchmark. $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/run_regimen_tacro.py = run_regimen_titrated.py on gen_tacro_regimens.py data
# (same patients / arms / noise as dh_regimen.py --sim tacro of the same seed, checked), covariates CYP3A5 + Ht/35 (patch P2),
# pulse width 1.5 h. Their training (100 + 300 epochs), unmodified. Sparse randomised s0-4, dense s0-2 (titrated runs dropped 13/10 12:00).
# Gate <= 6 python3 jobs, shared with queue_tacro_regimen.sh (re-checked after a pause so the two queues do not both fill the
# last slot). DUMP_CURVES=1: predicted test curves saved (<out>_curves.npz) for the shape comparison (0.00zz42). Marker TACROEB_DONE.
EB=$REPO_ROOT/third_party/baaz_ebvae; PY=${PYTHON:-python3}; R=$EB/ours/regimen/res_tacro; L=$REPO_ROOT/logs/regimen_dh
mkdir -p $L $R/ctrl $R/titr0.8
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 37; [ "$(njobs)" -lt 6 ] && return; done; }
J=()
for s in 0 1 2 3 4; do J+=("sparse $s none $R/ctrl/N100_sparse_s$s.json"); done
for s in 0 1 2; do J+=("dense $s none $R/ctrl/N100_dense_s$s.json"); done
for j in "${J[@]}"; do
    set -- $j; d=$1 s=$2 rho=$3 o=$4; tag=$(basename $(dirname $o))_$(basename $o .json)
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate
    ( cd $EB && env OMP_NUM_THREADS=1 DUMP_CURVES=1 nohup nice -n 5 $PY -u ours/regimen/run_regimen_tacro.py 100 $d $s $rho $o > $L/tacro_ebvae_$tag.log 2>&1 < /dev/null & )
    say "start EB-VAE $tag"; sleep 20
done
sleep 30; while pgrep -f "run_regimen_tacro.py 100" > /dev/null; do sleep 60; done
say "TACROEB_DONE"
