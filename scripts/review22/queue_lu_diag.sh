#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Job 3 diagnostic: Lu et al. two-port model on the tacrolimus regimen benchmark (linear, seed 0) with their InitState = 0
# (--nmi-x0 zero) instead of the encoded initial concentration; patients start drug-free on this design. Compare training-regimen
# fit and unseen-regimen error with the published runs (results/regimen_dh/tacro_s2/lunmi). Gate <= 8 jobs. Marker LUDIAG_DONE.
P=$REPO_ROOT; PY=${PYTHON:-python3}; O=$P/results/review22/lu_diag; L=$P/logs/review22
cd $P
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 8 ]; do sleep 30; done; sleep 23; [ "$(njobs)" -lt 8 ] && return; done; }
for d in visit sparse visit1 dense; do
  o=$O/lunmi_x0zero_${d}_s0.json; [ -f $o ] && continue; x=""; case $d in visit|visit1) x="--euler-dt 0.0833333333333333";; esac
  gate
  env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py nmi 100 $d 0 $o --iters 6000 --sim tacro \
      --nmi-x0 zero $x > $L/lu_diag_x0zero_$d.log 2>&1 < /dev/null &
  sleep 10
done
sleep 30; while pgrep -f "lunmi_x0zero_" > /dev/null; do sleep 30; done
echo LUDIAG_DONE
