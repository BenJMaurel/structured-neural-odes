#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Takes over from queue_null.sh (killed 19:35 with b s1-3, the true model and film s1-2 running): that queue gated on the
# NUMBER of heavy jobs, but the true-model job runs 2 threads (ebe_s4_batch sets torch.set_num_threads(2)), so 6 jobs
# could exceed 600 % cap. This one gates on the TOTAL python CPU (< 500 % before adding a 1-thread job).
# Remaining jobs: film s3, a s1-3; skips a job whose output exists or whose process is running. Then waits for every
# track_null job (also the ones queue_null.sh started) and writes the table.
#   marker TRACK_NULL_DONE in logs/tracking/queue_null.log (appended). Do not edit while running.
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
  esac
}
cpu () { ps -eo pcpu=,command= | awk '/python/ {s += $1} END {print int(s)}'; }
gate () { sleep 30; while [ "$(cpu)" -ge 500 ]; do sleep 60; done; }
say "queue_null2 takes over (cpu now $(cpu) %)"
for j in film:3 a:1 a:2 a:3; do
  k=${j%%:*}; s=${j##*:}
  [ -f $R/${k}_s$s.json ] && { say "skip $k s$s (exists)"; continue; }
  pgrep -f "track_null.py $k .*_s$s/" > /dev/null && { say "skip $k s$s (running)"; continue; }
  gate
  C=$(ckpt $k $s); [ -f "$C" ] || { say "MISSING $C"; continue; }
  nohup nice -n 5 $PY scripts/tracking/track_null.py $k $C $R/${k}_s$s.json --sigmas eb,ckpt --iters 40 \
    > logs/tracking/null_${k}_s$s.log 2>&1 &
  say "start $k s$s (pid $!)"
done
while pgrep -f "scripts/tracking/track_null" > /dev/null; do sleep 60; done
say "all jobs finished; table"
$PY scripts/tracking/null_table.py > $R/null_table.txt 2> logs/tracking/null_table.err
say "table -> $R/null_table.txt"
say "TRACK_NULL_DONE"
