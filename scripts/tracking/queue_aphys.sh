#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# A's physiology block only : track_null.py --phys-only on
# the frozen A (orth-split 1,5) checkpoints of the null test and of the recovery test, seeds 1-3, sigma eb
# and 0.05, then scripts/tracking/phys_table.py (FiLM / A full / A physiology-only / B / true model side by side).
# Gate: total python CPU < 500 % before each 1-thread job (600 % cap; other sessions' jobs count).
#   outputs  results/tracking/{null,recovery}/aphys_s<seed>.json, results/tracking/phys_table.txt
#   log      logs/tracking/queue_aphys.log, marker APHYS_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
T=confound_vc00_s4_pnoise_win_n100
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -eo pcpu=,command= | awk '/python/ {s += $1} END {print int(s)}'; }
gate () { sleep 20; while [ "$(cpu)" -ge 500 ]; do sleep 30; done; }
mkdir -p logs/tracking
for j in null:1 recovery:1 null:2 recovery:2 null:3 recovery:3; do
  t=${j%%:*}; s=${j##*:}
  OUT=results/tracking/$t/aphys_s$s.json
  [ -f $OUT ] && { say "skip $t s$s (exists)"; continue; }
  CK=results/s4_orth/osplit1-5_n100_s$s/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_osplit1-5_ep006000.ckpt
  [ -f "$CK" ] || { say "MISSING $CK"; continue; }
  if [ $t = recovery ]; then CO="--cohort confound_vc00_s4_pnoise_win_shift70"; else CO=""; fi
  gate
  nohup nice -n 5 $PY scripts/tracking/track_null.py a $CK $OUT $CO --sigmas eb,ckpt --iters 40 --phys-only \
    > logs/tracking/aphys_${t}_s$s.log 2>&1 &
  say "start $t s$s (pid $!, cpu before $(cpu) %)"
done
while pgrep -f "track_null.py a .*--phys-only" > /dev/null; do sleep 30; done
say "all jobs finished; failures: $(grep -l Traceback logs/tracking/aphys_*.log 2>/dev/null | tr '\n' ' ')"
$PY scripts/tracking/phys_table.py > results/tracking/phys_table.txt 2> logs/tracking/phys_table.err
say "table -> results/tracking/phys_table.txt"
say "APHYS_DONE"
