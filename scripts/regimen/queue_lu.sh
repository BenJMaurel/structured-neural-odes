#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Lu comparison on Baaz's regimen test: configs N100 sparse seeds 0-2 + N100
# dense seed 0; titration strength rho = 0.8 (corr(log ke, 800 mg) ~ +0.65).
#   controls (alternating arms): Lu iScience + Lu NMI (scripts/regimen/lu_regimen.py), EB-VAE through its titrated copy with
#            rho 'none' (must reproduce run_regimen.py, and stores per-patient AUCs); B / B2 controls already exist.
#   titrated: Lu iScience, Lu NMI, B (dh_regimen.py), B2 (dh_regimen_v2.py), EB-VAE (run_regimen_titrated.py).
# Gate (600 % cap): a job starts only while fewer than SLOTS heavy python jobs run AND total CPU < CAP %.
# Then scripts/regimen/lu_table.py -> results/regimen_dh/lu_table.txt. Log logs/queue_lu.log, marker LUQ_DONE.
# Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}; EB=$REPO_ROOT/third_party/baaz_ebvae
SLOTS=${SLOTS:-6}; CAP=${CAP:-550}; RHO=0.8; R=results/regimen_dh; T=$R/titr$RHO
LOG=logs/queue_lu.log; mkdir -p $R/luis $R/lunmi $T/luis $T/lunmi $T/B $T/B2 logs/regimen_dh $EB/ours/regimen/res_titr/ctrl $EB/ours/regimen/res_titr/titr$RHO
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|dh_regimen.py|dh_regimen_v2.py|lu_regimen.py|run_regimen.py|run_regimen_titrated.py|scripts/theory/"; }
cpu () { ps -A -o pcpu= | awk '{s+=$1} END {printf "%d", s}'; }
gate () { while [ $(running) -ge $SLOTS ] || [ $(( $(cpu) + 100 )) -gt $CAP ]; do sleep 15; done; }
CONFIGS="100:sparse:0 100:sparse:1 100:sparse:2 100:dense:0"
JOBS=""
for c in $CONFIGS; do JOBS="$JOBS ebvae:ctrl:$c ebvae:titr:$c"; done          # ~50 min each: start first
for c in $CONFIGS; do JOBS="$JOBS B:titr:$c B2:titr:$c"; done                   # 10-60 min
for c in $CONFIGS; do JOBS="$JOBS luis:ctrl:$c luis:titr:$c lunmi:ctrl:$c lunmi:titr:$c"; done   # minutes
say "QUEUE START: $(echo $JOBS | wc -w | tr -d ' ') jobs, slots $SLOTS, cap $CAP %"
PIDS=""
for j in $JOBS; do
  IFS=: read m cond n d s <<< "$j"
  tag=N${n}_${d}_s$s; lg=logs/regimen_dh/lu_${m}_${cond}_$tag.log
  case $m in
    ebvae) out=$EB/ours/regimen/res_titr/$([ $cond = ctrl ] && echo ctrl || echo titr$RHO)/$tag.json ;;
    *)     out=$([ $cond = ctrl ] && echo $R/$m || echo $T/$m)/$tag.json ;;
  esac
  [ -e $out ] && { say "skip $j (exists)"; continue; }
  gate
  case $m in
    ebvae) ( cd $EB && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg nohup nice -n 5 $PY -u ours/regimen/run_regimen_titrated.py \
               $n $d $s $([ $cond = ctrl ] && echo none || echo $RHO) $out > $OLDPWD/$lg 2>&1 ) & ;;
    B)     OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py $n $d $s $out --iters 6000 --titrate $RHO > $lg 2>&1 & ;;
    B2)    OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen_v2.py $n $d $s $out --iters 6000 --titrate $RHO > $lg 2>&1 & ;;
    luis|lunmi)
           OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py ${m#lu} $n $d $s $out --iters 6000 \
             $([ $cond = titr ] && echo --titrate $RHO) > $lg 2>&1 & ;;
  esac
  PIDS="$PIDS $!"; say "start $j (pid $!)"; sleep 20
done
wait $PIDS
for j in $JOBS; do
  IFS=: read m cond n d s <<< "$j"; lg=logs/regimen_dh/lu_${m}_${cond}_N${n}_${d}_s$s.log
  grep -qE "REGIMEN_LU_DONE|REGIMEN_DH_DONE|REGIMEN_DONE" $lg && say "done $j" || say "FAILED $j ($lg)"
done
$PY scripts/regimen/lu_table.py > $R/lu_table.txt 2> logs/lu_table.err
say "table -> $R/lu_table.txt"
say "LUQ_DONE"
