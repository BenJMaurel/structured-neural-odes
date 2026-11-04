#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Baaz's own regimen test with our model: scripts/regimen/dh_regimen.py, 18 runs in priority order
#   1. B (dose events, orth-split 1,5)  N 25/100/250 per arm x dense/sparse, seed 0      (= the EB-VAE seed-0 runs)
#   2. plain latent ODE + dose kicks     same 6 configurations, seed 0                     (does the B structure matter?)
#   3. B                                 sparse N 25/100/250, seeds 1-2                    (= the EB-VAE extra sparse seeds)
# 600 % CPU cap (fan): a job starts only when fewer than SLOTS (6) heavy python jobs of ANY queue run, so this
# queue fills the slots the latent sweep frees. 6000 iterations each. Then scripts/regimen/regimen_table.py ->
# results/regimen_dh/regimen_table.txt. Log logs/queue_dh_regimen.log, marker DHREG_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
SLOTS=${SLOTS:-6}; ITERS=${ITERS:-6000}
LOG=logs/queue_dh_regimen.log; R=results/regimen_dh
mkdir -p $R/B $R/plain logs/regimen_dh
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|dh_regimen.py|scripts/theory/"; }
JOBS=""
for c in "25 dense" "25 sparse" "100 sparse" "100 dense" "250 sparse" "250 dense"; do JOBS="$JOBS B:${c/ /:}:0"; done
for c in "25 dense" "25 sparse" "100 sparse" "100 dense" "250 sparse" "250 dense"; do JOBS="$JOBS plain:${c/ /:}:0"; done
for s in 1 2; do for n in 25 100 250; do JOBS="$JOBS B:$n:sparse:$s"; done; done
say "QUEUE START: $(echo $JOBS | wc -w | tr -d ' ') jobs, slots $SLOTS, $ITERS iterations"
PIDS=""
for j in $JOBS; do
  IFS=: read m n d s <<< "$j"
  out=$R/$m/N${n}_${d}_s$s.json
  [ -e $out ] && { say "skip $j (exists)"; continue; }
  while [ $(running) -ge $SLOTS ]; do sleep 30; done
  X=""; [ $m = plain ] && X="--plain"
  OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py $n $d $s $out --iters $ITERS $X \
    > logs/regimen_dh/${m}_N${n}_${d}_s$s.log 2>&1 &
  PIDS="$PIDS $!"; say "start $j (pid $!)"; sleep 20
done
wait $PIDS
for j in $JOBS; do
  IFS=: read m n d s <<< "$j"
  grep -q REGIMEN_DH_DONE logs/regimen_dh/${m}_N${n}_${d}_s$s.log && say "done $j" || say "FAILED $j (logs/regimen_dh/${m}_N${n}_${d}_s$s.log)"
done
$PY scripts/regimen/regimen_table.py > $R/regimen_table.txt 2> logs/regimen_table.err
say "table -> $R/regimen_table.txt"
say "DHREG_DONE"
