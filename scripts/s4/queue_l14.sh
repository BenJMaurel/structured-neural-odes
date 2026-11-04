#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Latent-14 study: variant A splits at -l 14 (1,5 / 1,7 / 1,9 / 1,11 = physiology 8 / 6 / 4 / 2) x seeds
# 1-3, plain latent ODE (paper model) on the windowed cohort at -l 10 and 14 x seeds 1-3, FiLM -l 14 seeds 4-6 (-> 6 with
# the latent sweep's 1-3). 21 jobs of arch_job.sh; a job starts only when fewer than SLOTS (6) heavy python jobs run
# (600 % cap; shares slots with any other queue). Then scripts/s4/l14_table.py -> results/s4_latent/l14_table.txt.
# Log logs/queue_l14.log, marker L14_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
SLOTS=${SLOTS:-6}
LOG=logs/queue_l14.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|dh_regimen.py|dh_regimen_v2.py|eval_lode.py|scripts/theory/"; }
JOBS=""
for s in 1 2 3; do JOBS="$JOBS 14:osplit1-7:$s 14:osplit1-9:$s 14:lode:$s 14:osplit1-5:$s 14:osplit1-11:$s 10:lode:$s"; done
for s in 4 5 6; do JOBS="$JOBS 14:film:$s"; done
say "QUEUE START: $(echo $JOBS | wc -w | tr -d ' ') jobs, slots $SLOTS"
PIDS=""
for j in $JOBS; do
  IFS=: read L k s <<< "$j"
  while [ $(running) -ge $SLOTS ]; do sleep 30; done
  nohup bash scripts/s4/arch_job.sh $L $k $s > /dev/null 2>&1 &
  PIDS="$PIDS $!"; sleep 20
done
wait $PIDS
say "QUEUE JOBS FINISHED: $(grep -c ': DONE$' $LOG) done, $(grep -cE 'FAILED|REFUSING' $LOG) failed/refused"
$PY scripts/s4/l14_table.py > results/s4_latent/l14_table.txt 2> logs/l14_table.err
say "table -> results/s4_latent/l14_table.txt"
say "L14_DONE"
