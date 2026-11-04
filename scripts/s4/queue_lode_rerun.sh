#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Rerun of the 6 plain latent ODE jobs of queue_l14.sh: they crashed at their first checkpoint save because
# arch_job.sh did not create the experiment subfolder (fixed 04/10 19:05; failed folders moved to results/_failed_runs/).
# Same 600 % gate. Then scripts/s4/l14_table.py -> results/s4_latent/l14_table.txt. Log logs/queue_l14.log (shared),
# marker LODE_RERUN_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
SLOTS=${SLOTS:-6}
LOG=logs/queue_l14.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|dh_regimen.py|dh_regimen_v2.py|eval_lode.py|scripts/theory/"; }
say "LODE RERUN START"
PIDS=""
for j in 10:1 14:1 10:2 14:2 10:3 14:3; do
  IFS=: read L s <<< "$j"
  while [ $(running) -ge $SLOTS ]; do sleep 30; done
  nohup bash scripts/s4/arch_job.sh $L lode $s > /dev/null 2>&1 &
  PIDS="$PIDS $!"; sleep 20
done
wait $PIDS
$PY scripts/s4/l14_table.py > results/s4_latent/l14_table.txt 2> logs/l14_table.err
say "table -> results/s4_latent/l14_table.txt"
say "LODE_RERUN_DONE"
