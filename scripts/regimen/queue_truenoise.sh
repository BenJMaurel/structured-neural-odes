#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Improved B with the TRUE observation noise (10 mg/L) on the saturable simulator.
# Hypothesis: the fixed noise 0.05*YMAX (81 mg/L sparse / 73 dense) hides the curve-shape misfit (~20 mg/L) that lets NLME identify
# saturation (dBIC ~1000). --obsrv-std = 10 / YMAX (YMAX 1611.8 sparse s1, 1462.9 dense s0; printed by the run, check it).
# The data term is ~64x stronger than at 81 mg/L, so the penalty is swept wider: 0, 1e3, 1e4, 1e5 (1e5 ~ the old 1000 rescaled).
# B2 (all corrections off) at the true noise = baseline. One seed per test (sparse s1, dense s0).
# Log logs/queue_truenoise.log, marker TRUENOISE_DONE. Outputs results/regimen_dh/v3/mm500_truenoise/{lam<L>,B2}/.
cd $REPO_ROOT
source scripts/regimen/gate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 REGIMEN_KM=500
PY=${PYTHON:-python3}; O=results/regimen_dh/v3/mm500_truenoise; L=logs/regimen_dh; LOG=logs/queue_truenoise.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
say "start"
for d in sparse dense; do
  if [ $d = sparse ]; then s=1; sd=0.0062043; else s=0; sd=0.0068357; fi
  for lam in B2 0 1000 10000 100000; do
    if [ $lam = B2 ]; then sub=B2; extra=""; else sub=lam$lam; extra="--resid-dyn $lam --resid-read $lam"; fi
    out=$O/$sub/N100_${d}_s${s}.json; [ -e $out ] && continue
    gate
    $PY scripts/regimen/dh_regimen_v3.py 100 $d $s $out --iters 6000 --obsrv-std $sd $extra > $L/v3_mm500_truenoise_${sub}_${d}_s${s}.log 2>&1 &
    say "start $sub $d s$s (pid $!)"; sleep 10
  done
done
wait
$PY scripts/regimen/truenoise_table.py > results/regimen_dh/truenoise_table.txt 2> logs/truenoise_table.err
say "TRUENOISE_DONE"
