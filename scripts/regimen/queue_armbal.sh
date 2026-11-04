#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Physiology arm-balance test and p(p | 400 mg), on the physiology p ONLY,
# randomised arms; "test it as you suggested after the runs finish"). Improved B on the saturable simulator, sparse s1 / dense s0,
# balance weight 0.1 / 1 / 10; + titrated training arms (rho 0.8, sparse s1) with balance applied naively (weight 1) and its
# unbalanced reference. Noise and correction penalty are chosen from the true-noise runs (queue_truenoise.sh):
#   queue_armbal.sh <true|default> <correction penalty>
# Log logs/queue_armbal.log, marker ARMBAL_DONE. Outputs results/regimen_dh/v3/mm500_armbal_<noise>_lam<L>/{w<W>,titr0.8/w<W>}/.
cd $REPO_ROOT
source scripts/regimen/gate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 REGIMEN_KM=500
NOISE=$1; LAM=$2; [ -n "$LAM" ] || { echo "usage: queue_armbal.sh <true|default> <penalty>"; exit 1; }
PY=${PYTHON:-python3}; O=results/regimen_dh/v3/mm500_armbal_${NOISE}_lam${LAM}; L=logs/regimen_dh; LOG=logs/queue_armbal.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
sdflag () { [ $NOISE = true ] && { [ $1 = sparse ] && echo "--obsrv-std 0.0062043" || echo "--obsrv-std 0.0068357"; }; }
run () {   # design seed weight titrate(""|0.8)
  local d=$1 s=$2 w=$3 t=$4 sub=w$3; [ -n "$t" ] && sub=titr$t/w$w
  local out=$O/$sub/N100_${d}_s${s}.json; [ -e $out ] && return
  gate
  $PY scripts/regimen/dh_regimen_v3.py 100 $d $s $out --iters 6000 $(sdflag $d) --resid-dyn $LAM --resid-read $LAM --arm-balance $w \
      ${t:+--titrate $t} > $L/armbal_${NOISE}_lam${LAM}_${sub//\//_}_${d}_s${s}.log 2>&1 &
  say "start $sub $d s$s (pid $!)"; sleep 10
}
say "start: noise $NOISE, correction penalty $LAM"
for w in 0.1 1 10; do run dense 0 $w; done          # dense first (longest)
for w in 0.1 1 10; do run sparse 1 $w; done
run sparse 1 1 0.8; run sparse 1 0 0.8                # titrated: naive balance, and its unbalanced reference
wait
say "ARMBAL_DONE"
