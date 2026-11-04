#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Score the Table 2 extra seeds as soon as each training run ends ,
# instead of waiting for queue_weekend.sh to have launched every regimen run. Same commands and outputs as queue_weekend.sh's scoring
# block (which then skips the files that exist). Gate: total python3 jobs <= 8, polled every 5 s so that a freed slot is taken
# before the queue's 41-s re-check; at most 4 scoring jobs at once. At the end: scripts/weekend/table2_more.py. Marker T2EARLY_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; L=logs/weekend
T=confound_vc00_s4_pnoise_win_n100; BC=confound_vc00_s4_pnoise_win; R2=results/s4_pnoise_win_more; RC=results/recal/paper_win_more
CAP=8; KEEP=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
mine () { pgrep -f "seeded_eval.py .*t1winmore_" | wc -l; }
slot () { while [ "$(njobs)" -ge $CAP ] || [ "$(mine)" -ge $KEEP ]; do sleep 5; done; }
mkdir -p $R2/eval $RC
say "start"
while true; do
  left=0
  for arch in film dc; do
    [ $arch = film ] && { H=test_film_matched.py; sub=exp_film_run; } || { H=test_dose_cond.py; sub=exp_dosecond_run; }
    for s in 1 4 5 6 7 8 9 10; do
      D=$R2/${arch}_n100_s$s
      CK=$(ls $D/$sub/$T/traj/*_ep006000.ckpt 2>/dev/null)
      if [ "$(echo "$CK" | grep -c ckpt)" != 1 ] || pgrep -f -- "--save ./$D/" > /dev/null; then left=$((left + 1)); continue; fi
      for reg in in lo hi all; do
        if [ $reg = all ]; then o=$RC/${arch}_s${s}_all.json; ex=$T; split=all; lab=t1winmore_${arch}_all_s$s; lg=$L/recal_${arch}_s$s.log
        else o=$R2/eval/ep006000_${arch}_n100_s${s}_on_$reg.json; ex=$BC; [ $reg != in ] && ex=${BC}_$reg; split=test
             lab=t1winmore_${arch}_s${s}_$reg; lg=$L/eval_${arch}_s${s}_$reg.log; fi
        if [ -f $o ]; then continue; fi
        if [ -f $lg ] && grep -q Traceback $lg && ! pgrep -f -- "--out-json $o" > /dev/null; then say "FAILED $o (see $lg)"; continue; fi
        if pgrep -f -- "--out-json $o" > /dev/null; then left=$((left + 1)); continue; fi
        slot
        nohup nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $ex --data-dir ./results/exp_film_run \
          --ckpt $CK --scale-from $T --eval-split $split --label $lab --out-json $o > $lg 2>&1 < /dev/null &
        say "score $arch s$s $reg"; left=$((left + 1)); sleep 2
      done
    done
  done
  [ $left = 0 ] && break
  sleep 20
done
say "all scored: $(ls $R2/eval/*.json | wc -l | tr -d ' ') eval + $(ls $RC/*.json | wc -l | tr -d ' ') recal files"
$PY -W ignore scripts/weekend/table2_more.py > $L/table2_more.out 2>&1
say "T2EARLY_DONE"
