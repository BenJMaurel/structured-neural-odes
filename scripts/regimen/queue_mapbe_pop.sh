#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Realistic MAP-BE on Baaz's regimen test: SAEM (Monolix) on the 200 training patients, then
# MAP-BE with the estimated population; linear model on linear AND saturable data, saturable model on saturable data, and the
# linear model with the TRUE population on saturable data. One seed per test (sparse s1, dense s0).
# Log logs/queue_mapbe_pop.log, marker MAPBE_POP_DONE. Outputs results/regimen_dh/mapbe_pop/.
cd $REPO_ROOT
source scripts/regimen/gate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}; D=results/regimen_dh/mapbe_pop; L=logs/regimen_dh; LOG=logs/queue_mapbe_pop.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
km () { [ "$1" = mm500 ] && echo 500 || echo ""; }
pipeline () {   # sim design seed struct
  local sim=$1 d=$2 s=$3 st=$4 K; K=$(km $sim); local tag=${sim}_${d}_s${s}
  local fit=$D/saem/${tag}_${st}/estimates.json out=$D/$sim/saem_$st/N100_${d}_s${s}.json
  [ -e $out ] && return
  [ -e $D/data/$tag.csv ] || REGIMEN_KM=$K $PY scripts/regimen/mapbe_pop.py export $d $s $D/data/$tag.csv >> $L/mapbe_pop_${tag}.log 2>&1
  [ -e $fit ] || Rscript scripts/monolix/fit_regimen.R $D/data/$tag.csv $D/saem/${tag}_${st} $st > $L/saem_${tag}_${st}.log 2>&1
  [ -e $fit ] || { say "FIT FAILED $tag $st"; return; }
  REGIMEN_KM=$K $PY scripts/regimen/mapbe_pop.py fit $d $s $out --struct $st --pop $fit > $L/mapbe_pop_${tag}_saem_${st}.log 2>&1
  say "done $tag saem_$st"
}
say "start"
for job in "lin sparse 1 lin" "lin dense 0 lin" "mm500 sparse 1 lin" "mm500 dense 0 lin" "mm500 sparse 1 mm" "mm500 dense 0 mm"; do
  gate; pipeline $job & sleep 5
done
for ds in "sparse 1" "dense 0"; do
  set -- $ds; o=$D/mm500/true_lin/N100_$1_s$2.json
  [ -e $o ] || { gate; REGIMEN_KM=500 $PY scripts/regimen/mapbe_pop.py fit $1 $2 $o --struct lin --pop true > $L/mapbe_pop_mm500_$1_s$2_true_lin.log 2>&1 & }
  sleep 5
done
wait
say "MAPBE_POP_DONE"
