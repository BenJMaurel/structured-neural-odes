#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Variant A --orth-split 1,9 at -l 14, seeds 4-10"yes go ahead"; ): seeds 1-3 put
# all three V2 medians below every FiLM seed (12.1 / 12.1 / 12.5 vs FiLM l14 12.7-13.2), the pattern A 1,5 at l10 showed at
# 3 seeds before falling back at 10 -> 10 seeds decide. 7 jobs of arch_job.sh, 600 % gate. Then scripts/s4/l14_table.py ->
# results/s4_latent/l14_table.txt. Log logs/queue_l14.log (shared), marker A19_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
SLOTS=${SLOTS:-6}
LOG=logs/queue_l14.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|dh_regimen.py|dh_regimen_v2.py|eval_lode.py|scripts/theory/"; }
say "A 1,9 l14 SEEDS 4-10 START"
PIDS=""
for s in 4 5 6 7 8 9 10; do
  while [ $(running) -ge $SLOTS ]; do sleep 30; done
  nohup bash scripts/s4/arch_job.sh 14 osplit1-9 $s > /dev/null 2>&1 &
  PIDS="$PIDS $!"; sleep 20
done
wait $PIDS
$PY scripts/s4/l14_table.py > results/s4_latent/l14_table.txt 2> logs/l14_table.err
say "table -> results/s4_latent/l14_table.txt"
say "A19_DONE"
