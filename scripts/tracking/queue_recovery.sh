#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Recovery test of the a-posteriori physiology tracker: the frozen FiLM / A / B checkpoints of the null
# test and the true-model reference on confound_vc00_s4_pnoise_win_shift70 = the same patients, doses and
# visit-1 data as the windowed paper-noise cohort, with visit-2 Vmax x exp(delta), delta ~ U(-0.7, 0.7), for half of the
# patients. Waits for the cohort (gen_tacro_confound.py --v2-shift, log logs/tracking/gen_shift70.log), then runs the 10
# jobs under the total-CPU gate (< 500 % before each 1-thread job; 600 % cap), then the table.
#   outputs  results/tracking/recovery/{film,a,b}_s<seed>.json, truemodel.json, recovery_table.txt
#   log      logs/tracking/queue_recovery.log, marker TRACK_RECOVERY_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
C=confound_vc00_s4_pnoise_win_shift70
T=confound_vc00_s4_pnoise_win_n100
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
R=results/tracking/recovery
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
ckpt () {
  case $1 in
    film) echo results/s4_pnoise_win/film_n100_s$2/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_ep006000.ckpt;;
    a)    echo results/s4_orth/osplit1-5_n100_s$2/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_osplit1-5_ep006000.ckpt;;
    b)    echo results/s4_doseh/dh7pg_osplit1-5_n100_s$2/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_osplit1-5_dh7_taupg12_ep006000.ckpt;;
  esac
}
cpu () { ps -eo pcpu=,command= | awk '/python/ {s += $1} END {print int(s)}'; }
gate () { sleep 30; while [ "$(cpu)" -ge 500 ]; do sleep 60; done; }
mkdir -p $R logs/tracking
say "waiting for results/exp_film_run/$C"
while [ ! -f results/exp_film_run/$C/noise.json ]; do
  pgrep -f "gen_tacro_confound.py --exp $C" > /dev/null || { sleep 5; [ -f results/exp_film_run/$C/noise.json ] || { say "GENERATOR DIED"; exit 1; }; }
  sleep 60
done
say "cohort ready: $(tail -n 6 logs/tracking/gen_shift70.log | tr '\n' ' ')"
for j in true:0 b:1 b:2 b:3 film:1 film:2 film:3 a:1 a:2 a:3; do
  k=${j%%:*}; s=${j##*:}
  gate
  if [ $k = true ]; then
    EBE_COHORT=$C nohup nice -n 5 $PY scripts/tracking/track_null_truemodel.py $R/truemodel.json > logs/tracking/recovery_truemodel.log 2>&1 &
  else
    CK=$(ckpt $k $s); [ -f "$CK" ] || { say "MISSING $CK"; continue; }
    nohup nice -n 5 $PY scripts/tracking/track_null.py $k $CK $R/${k}_s$s.json --cohort $C --sigmas eb,ckpt --iters 40 \
      > logs/tracking/recovery_${k}_s$s.log 2>&1 &
  fi
  say "start $k s$s (pid $!)"
done
while pgrep -f "scripts/tracking/track_null" > /dev/null; do sleep 60; done
say "all jobs finished; table"
$PY scripts/tracking/recovery_table.py > $R/recovery_table.txt 2> logs/tracking/recovery_table.err
say "table -> $R/recovery_table.txt"
say "TRACK_RECOVERY_DONE"
