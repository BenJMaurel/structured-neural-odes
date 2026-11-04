#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Tacrolimus regimen benchmark, the other designs: does the data-driven absorption -- no absorption
# with sparse samples, the true absorption peak with dense ones -- also appear in Lu, dose-cond or the EB-VAE?). Same patients as the
# dose-event runs of each seed. Lu iScience and Lu NMI (lu_regimen.py --sim tacro: script defaults, Adam 1e-3, full batch, 6000 it,
# no sweep; CYP3A5 + Ht/35 as constant encoder channels), then the dose-conditioned field with the dose-event model's encoder, readout,
# likelihood and training (dh_regimen.py --sim tacro --dose-cond --enc-dose norm --obsrv-mgl 4). Sparse s0-4, dense s0-2.
# Gate <= 6 python3 jobs shared with the other queues (re-checked after 53 s). Marker TACROVERS_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh/tacro; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 53; [ "$(njobs)" -lt 6 ] && return; done; }
J=()
for m in is nmi; do
  for s in 0 1 2 3 4; do J+=("lu_regimen.py $R/lu$m/N100_sparse_s$s.json tacro_lu${m}_sparse_s$s $m 100 sparse $s"); done
  for s in 0 1 2; do J+=("lu_regimen.py $R/lu$m/N100_dense_s$s.json tacro_lu${m}_dense_s$s $m 100 dense $s"); done
done
for s in 0 1 2 3 4; do J+=("dh_regimen.py $R/dosecond_n/N100_sparse_s$s.json tacro_dcn_sparse_s$s 100 sparse $s"); done
for s in 0 1 2; do J+=("dh_regimen.py $R/dosecond_n/N100_dense_s$s.json tacro_dcn_dense_s$s 100 dense $s"); done
for j in "${J[@]}"; do
    set -- $j; scr=$1 o=$2 lg=$3; shift 3
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    if [ $scr = dh_regimen.py ]; then extra="--iters 6000 --sim tacro --dose-cond --enc-dose norm --obsrv-mgl 4"; else extra="--iters 6000 --sim tacro"; fi
    gate; mkdir -p "$(dirname $o)"
    env OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/$scr "$@" "$o" $extra > "$L/$lg.log" 2>&1 < /dev/null &
    say "start $lg"; sleep 20
done
sleep 30; while pgrep -f "regimen.py .*regimen_dh/tacro/(lu|dosecond)" > /dev/null; do sleep 60; done
say "TACROVERS_DONE"
