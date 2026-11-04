#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# B VERSION 2 on Baaz's regimen test: scripts/regimen/dh_regimen_v2.py (fixed noise sd 0.05, linear
# compartmental drug dynamics + readout, exact matrix-exponential solution), the same 12 configurations as the EB-VAE:
# N 25/100/250 per arm x dense/sparse seed 0, + sparse seeds 1-2. Same 600 % gate as queue_dh_regimen.sh (a job starts
# only when fewer than SLOTS heavy python jobs run; polls every 10 s so the short v2 jobs get freed slots first).
# Then scripts/regimen/regimen_table.py -> results/regimen_dh/regimen_table.txt. Log logs/queue_dh_regimen_v2.log,
# marker DHREG2_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
SLOTS=${SLOTS:-6}; ITERS=${ITERS:-6000}
LOG=logs/queue_dh_regimen_v2.log; R=results/regimen_dh/B2
mkdir -p $R logs/regimen_dh
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|dh_regimen.py|dh_regimen_v2.py|scripts/theory/"; }
JOBS=""
for c in "25 dense" "25 sparse" "100 sparse" "100 dense" "250 sparse" "250 dense"; do JOBS="$JOBS ${c/ /:}:0"; done
for s in 1 2; do for n in 25 100 250; do JOBS="$JOBS $n:sparse:$s"; done; done
say "QUEUE START: $(echo $JOBS | wc -w | tr -d ' ') jobs, slots $SLOTS, $ITERS iterations"
PIDS=""
for j in $JOBS; do
  IFS=: read n d s <<< "$j"
  out=$R/N${n}_${d}_s$s.json
  [ -e $out ] && { say "skip $j (exists)"; continue; }
  while [ $(running) -ge $SLOTS ]; do sleep 10; done
  OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen_v2.py $n $d $s $out --iters $ITERS \
    > logs/regimen_dh/B2_N${n}_${d}_s$s.log 2>&1 &
  PIDS="$PIDS $!"; say "start $j (pid $!)"; sleep 15
done
wait $PIDS
for j in $JOBS; do
  IFS=: read n d s <<< "$j"
  grep -q REGIMEN_DH_DONE logs/regimen_dh/B2_N${n}_${d}_s$s.log && say "done $j" || say "FAILED $j (logs/regimen_dh/B2_N${n}_${d}_s$s.log)"
done
$PY scripts/regimen/regimen_table.py > results/regimen_dh/regimen_table.txt 2> logs/regimen_table.err
say "table -> results/regimen_dh/regimen_table.txt"
say "DHREG2_DONE"
