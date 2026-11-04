#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Two hypotheses as NLME models. PATIENT route = linear model + arm covariate (800 vs 400 mg) on ke, or on ka/ke/V; DRUG route =
# saturable model (shared Km, no arm effect; fitted by queue_mapbe_pop.sh); BOTH = saturable + arm covariate on ka/ke/V.
# Negative control: linear data, linear + arm covariate on ka/ke/V (beta should be ~0). SAEM (Monolix) then MAP-BE (mapbe_pop.py).
# Log logs/queue_nlme_hyp.log, marker NLME_HYP_DONE.
cd $REPO_ROOT
source scripts/regimen/gate.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}; D=results/regimen_dh/mapbe_pop; L=logs/regimen_dh; LOG=logs/queue_nlme_hyp.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
pipeline () {   # sim design seed struct arm_on variant
  local sim=$1 d=$2 s=$3 st=$4 on=$5 v=$6 K=""; [ $sim = mm500 ] && K=500; local tag=${sim}_${d}_s${s}
  local fit=$D/saem/${tag}_${v}/estimates.json out=$D/$sim/saem_$v/N100_${d}_s${s}.json
  [ -e $out ] && return
  [ -e $D/data/${tag}_arm.csv ] || REGIMEN_KM=$K $PY scripts/regimen/mapbe_pop.py export $d $s $D/data/${tag}_arm.csv --arm-col >> $L/mapbe_pop_${tag}_arm.log 2>&1
  [ -e $fit ] || Rscript scripts/monolix/fit_regimen.R $D/data/${tag}_arm.csv $D/saem/${tag}_${v} $st $on > $L/saem_${tag}_${v}.log 2>&1
  [ -e $fit ] || { say "FIT FAILED $tag $v"; return; }
  REGIMEN_KM=$K $PY scripts/regimen/mapbe_pop.py fit $d $s $out --struct $st --pop $fit > $L/mapbe_pop_${tag}_saem_${v}.log 2>&1
  say "done $tag $v"
}
say "start"
# exports first (sequential: the same file is shared by several fits)
for ds in "lin sparse 1" "lin dense 0" "mm500 sparse 1" "mm500 dense 0"; do
  set -- $ds; K=""; [ $1 = mm500 ] && K=500
  [ -e $D/data/$1_$2_s$3_arm.csv ] || REGIMEN_KM=$K $PY scripts/regimen/mapbe_pop.py export $2 $3 $D/data/$1_$2_s$3_arm.csv --arm-col >> $L/mapbe_pop_export_arm.log 2>&1
done
for job in "mm500 sparse 1 mm ka,ke,V mm_armall" "mm500 dense 0 mm ka,ke,V mm_armall" \
           "mm500 sparse 1 lin ke lin_armke" "mm500 dense 0 lin ke lin_armke" \
           "mm500 sparse 1 lin ka,ke,V lin_armall" "mm500 dense 0 lin ka,ke,V lin_armall" \
           "lin sparse 1 lin ka,ke,V lin_armall" "lin dense 0 lin ka,ke,V lin_armall"; do
  gate; pipeline $job & sleep 5
done
wait
say "NLME_HYP_DONE"
