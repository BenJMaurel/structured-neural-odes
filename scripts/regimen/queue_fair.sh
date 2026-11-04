#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Fairness checks for the Lu comparison: N100 sparse seeds 0-2 + N100 dense seed 0.
#   A. tuning sweep of Lu iScience on Baaz's counterfactual test: lr {1e-4, 3e-4, 1e-3, 3e-3} x weight decay {0, 1e-3, 0.1}
#      (lr 1e-3 / wd 0 = the existing runs in results/regimen_dh/luis) + THEIR optimiser (batch 1, lr 5e-5, L2 0.1, 300 epochs)
#      -> results/regimen_dh/luis_sweep/<cfg>/N100_<design>_s<seed>.json
#   B. the iScience FORMAT for every model (--protocol firstint: the encoder sees the first dosing interval of a regimen, the
#      model predicts the rest of the same regimen; test patients observed under every regimen; scored on [t2, 30 h]):
#      Lu iScience, Lu NMI, B, B2 -> results/regimen_dh/firstint/<model>/; EB-VAE (run_regimen_firstint.py, patch P3) ->
#      $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/res_firstint/
# Gate:allowed up to 1000 % until 07/10 ~16:00 -> at most SLOTS heavy python jobs (other sessions' included) AND
# total CPU < CAP %. Then scripts/regimen/fair_table.py -> results/regimen_dh/fair_table.txt. Log logs/queue_fair.log,
# marker FAIRQ_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}; EB=$REPO_ROOT/third_party/baaz_ebvae
SLOTS=${SLOTS:-10}; CAP=${CAP:-950}; R=results/regimen_dh; SW=$R/luis_sweep; FI=$R/firstint
LOG=logs/queue_fair.log; mkdir -p $SW $FI/luis $FI/lunmi $FI/B $FI/B2 logs/regimen_dh $EB/ours/regimen/res_firstint
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|train_|dh_regimen|lu_regimen|run_regimen|scripts/theory/"; }
cpu () { ps -A -o pcpu= | awk '{s+=$1} END {printf "%d", s}'; }
gate () { while [ $(running) -ge $SLOTS ] || [ $(( $(cpu) + 100 )) -gt $CAP ]; do sleep 15; done; }
CONFIGS="100:sparse:0 100:sparse:1 100:sparse:2 100:dense:0"
JOBS=""
for c in $CONFIGS; do JOBS="$JOBS fi:ebvae:$c fi:B:$c fi:B2:$c"; done                     # ~10-60 min each: first
for c in $CONFIGS; do JOBS="$JOBS sw:b1:$c"; done                                          # their optimiser, ~30 min
for c in $CONFIGS; do for lr in 1e-4 3e-4 1e-3 3e-3; do for wd in 0 1e-3 0.1; do
  [ "$lr:$wd" = "1e-3:0" ] && continue; JOBS="$JOBS sw:lr${lr}_wd${wd}:$c"; done; done; done  # minutes each
for c in $CONFIGS; do JOBS="$JOBS fi:luis:$c fi:lunmi:$c"; done
say "QUEUE START: $(echo $JOBS | wc -w | tr -d ' ') jobs, slots $SLOTS, cap $CAP %"
PIDS=""
for j in $JOBS; do
  IFS=: read part m n d s <<< "$j"
  tag=N${n}_${d}_s$s; lg=logs/regimen_dh/fair_${part}_${m}_$tag.log
  if [ $part = sw ]; then out=$SW/$m/$tag.json; mkdir -p $SW/$m
  elif [ $m = ebvae ]; then out=$EB/ours/regimen/res_firstint/$tag.json
  else out=$FI/$m/$tag.json; fi
  [ -e $out ] && { say "skip $j (exists)"; continue; }
  gate
  if [ $part = sw ]; then
    if [ $m = b1 ]; then opt="--batch 1 --lr 5e-5 --wd 0.1 --iters 60000"
    else lr=${m#lr}; lr=${lr%%_wd*}; wd=${m##*_wd}; opt="--lr $lr --wd $wd --iters 6000"; fi
    OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py is $n $d $s $out $opt > $lg 2>&1 &
  else
    case $m in
      ebvae) ( cd $EB && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg nohup nice -n 5 $PY -u ours/regimen/run_regimen_firstint.py \
                 $n $d $s $out > $OLDPWD/$lg 2>&1 ) & ;;
      B)     OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py $n $d $s $out --iters 6000 --protocol firstint > $lg 2>&1 & ;;
      B2)    OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen_v2.py $n $d $s $out --iters 6000 --protocol firstint > $lg 2>&1 & ;;
      luis|lunmi) OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py ${m#lu} $n $d $s $out --iters 6000 --protocol firstint > $lg 2>&1 & ;;
    esac
  fi
  PIDS="$PIDS $!"; say "start $j (pid $!)"; sleep 10
done
wait $PIDS
for j in $JOBS; do
  IFS=: read part m n d s <<< "$j"; lg=logs/regimen_dh/fair_${part}_${m}_N${n}_${d}_s$s.log
  grep -qE "REGIMEN_LU_DONE|REGIMEN_DH_DONE|REGIMEN_DONE" $lg && say "done $j" || say "FAILED $j ($lg)"
done
$PY scripts/regimen/fair_table.py > $R/fair_table.txt 2> logs/fair_table.err
say "table -> $R/fair_table.txt"
say "FAIRQ_DONE"
