#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Latent-size sweep; ): Table 3 FiLM main config, -l 6 8 12 14 16 18 20 (l = 10 = the
# existing 10 FiLM seeds), 3 seeds each = 21 jobs of latent_job.sh, at most MAXJ at once (default 6 = the 600 % cap: one
# core per job, evals run sequentially inside a job). Interleaved so every size gets a seed early. Then
# scripts/s4/latent_table.py -> results/s4_latent/latent_table.txt. Log logs/queue_latent.log, marker LATENT_DONE.
# Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
MAXJ=${MAXJ:-6}
LOG=logs/queue_latent.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
ORDER=""
for s in 1 2 3; do for L in 6 20 8 18 12 16 14; do ORDER="$ORDER $L:$s"; done; done
say "QUEUE START: $(echo $ORDER | wc -w | tr -d ' ') jobs, MAXJ $MAXJ"
echo $ORDER | tr ' ' '\n' | tr ':' ' ' | xargs -P $MAXJ -L 1 bash scripts/s4/latent_job.sh
say "QUEUE JOBS FINISHED: $(grep -c ': DONE$' $LOG) done, $(grep -cE 'FAILED|REFUSING' $LOG) failed/refused"
$PY scripts/s4/latent_table.py > results/s4_latent/latent_table.txt 2> logs/latent_table.err
say "table -> results/s4_latent/latent_table.txt"
say "LATENT_DONE"
