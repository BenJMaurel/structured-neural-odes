#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Overnight regimen queue, replacing queue_fair.sh / queue_b2diag.sh / queue_ebvae_curves.sh (stopped before any of
# their pending jobs started). Job lists from scripts/regimen/jobs_night.py; a job is skipped when its output exists or a process
# already writes it. Gate: scripts/regimen/gate.sh (10 jobs / 950 % until 2026-10-07 16:00, then 6 / 550 %; other sessions count).
# List 1, then barrier, then list 2 (titrated improved B at the validation-chosen penalty), then tables.
# Log logs/queue_regimen_night.log, markers LIST1_DONE, NIGHT_DONE. Do not edit while running.
cd $REPO_ROOT
source scripts/regimen/gate.sh
PY=${PYTHON:-python3}; LOG=logs/queue_regimen_night.log; mkdir -p logs/regimen_dh
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
run_list () {
  local file=$1 P="" n=0
  while IFS=$'\t' read -r out log wd env cmd; do
    [ -z "$out" ] && continue
    [ -e "$out" ] && continue
    ps -A -o command= | grep -v grep | grep -qF -- "$out" && { say "already running: $out"; continue; }
    gate
    mkdir -p "$(dirname "$out")"
    [ "$env" = "-" ] && env=""
    ( cd "$wd" && env OMP_NUM_THREADS=1 $env nohup nice -n 5 $cmd > "$log" 2>&1 ) &
    P="$P $!"; n=$((n + 1)); say "start $(basename "$log" .log) (pid $!)"; sleep 10
  done < "$file"
  say "list $file: $n jobs started; waiting"
  wait $P
}
$PY scripts/regimen/jobs_night.py list1 > logs/queue_regimen_night_list1.tsv
say "QUEUE START: list 1 = $(grep -c . logs/queue_regimen_night_list1.tsv) jobs (existing outputs skipped)"
run_list logs/queue_regimen_night_list1.tsv
say "LIST1_DONE"
$PY scripts/regimen/jobs_night.py list2 > logs/queue_regimen_night_list2.tsv
say "list 2 = $(grep -c . logs/queue_regimen_night_list2.tsv) jobs; penalties chosen on validation: $(cat results/regimen_dh/v3/selected_lambda.json | tr -d '\n ')"
run_list logs/queue_regimen_night_list2.tsv
for t in fair_table lu_table; do $PY scripts/regimen/$t.py > results/regimen_dh/$t.txt 2> logs/$t.err; done
[ -e scripts/regimen/v3_table.py ] && $PY scripts/regimen/v3_table.py > results/regimen_dh/v3_table.txt 2> logs/v3_table.err
say "tables written"
say "NIGHT_DONE"
