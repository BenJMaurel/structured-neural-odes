#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Null test of the a-posteriori physiology tracker: FiLM / A (osplit1-5) / fixed B (dh7pg), seeds 1-3,
# ep6000 traj checkpoints, all 1100 in-range test patients of the windowed paper-noise s4 cohort, sigma eb + ckpt;
# plus the true-model MAP-BE reference. At most 6 heavy python jobs on the box (600 % cap), 1 thread each.
#   outputs  results/tracking/null/{film,a,b}_s<seed>.json, results/tracking/null/truemodel.json
#   logs     logs/tracking/null_<kind>_s<seed>.log, logs/tracking/null_truemodel.log; this log logs/tracking/queue_null.log
#   marker   TRACK_NULL_DONE in logs/tracking/queue_null.log. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
T=confound_vc00_s4_pnoise_win_n100
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
R=results/tracking/null
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
ckpt () {
  case $1 in
    film) echo results/s4_pnoise_win/film_n100_s$2/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_ep006000.ckpt;;
    a)    echo results/s4_orth/osplit1-5_n100_s$2/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_osplit1-5_ep006000.ckpt;;
    b)    echo results/s4_doseh/dh7pg_osplit1-5_n100_s$2/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_osplit1-5_dh7_taupg12_ep006000.ckpt;;
  esac
}
heavy () { ps -eo pcpu=,command= | awk '$1 > 50 && /python/' | wc -l | tr -d ' '; }
gate () { sleep 20; while [ "$(heavy)" -ge 6 ]; do sleep 60; done; }
mkdir -p $R logs/tracking
JOBS="b:1 b:2 b:3 true:0 film:1 film:2 film:3 a:1 a:2 a:3"
for j in $JOBS; do
  k=${j%%:*}; s=${j##*:}
  gate
  if [ $k = true ]; then
    [ -f $R/truemodel.json ] && { say "skip truemodel (exists)"; continue; }
    EBE_COHORT=confound_vc00_s4_pnoise_win nohup nice -n 5 $PY scripts/tracking/track_null_truemodel.py $R/truemodel.json \
      > logs/tracking/null_truemodel.log 2>&1 &
  else
    C=$(ckpt $k $s); [ -f "$C" ] || { say "MISSING $C"; continue; }
    [ -f $R/${k}_s$s.json ] && { say "skip $k s$s (exists)"; continue; }
    nohup nice -n 5 $PY scripts/tracking/track_null.py $k $C $R/${k}_s$s.json --sigmas eb,ckpt --iters 40 \
      > logs/tracking/null_${k}_s$s.log 2>&1 &
  fi
  say "start $k s$s (pid $!)"
done
wait
say "all jobs finished; table"
$PY scripts/tracking/null_table.py > $R/null_table.txt 2> logs/tracking/null_table.err
say "table -> $R/null_table.txt"
say "TRACK_NULL_DONE"
