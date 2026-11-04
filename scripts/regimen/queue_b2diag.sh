#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Why does B2 do badly in the iScience format? N100 sparse s0-2 + dense s0, --protocol firstint:
#   ceiling: MAP-BE oracle (true model + true population, first interval of each regimen) -> results/regimen_dh/firstint/mapbe_oracle/
#   B2 with what the ENCODER is told about the dose changed (--enc-dose norm | none; dynamics keep the true doses)
#     -> results/regimen_dh/firstint/B2_encnorm/, B2_encnone/   (hypothesis: the encoder uses the dose as a 2-valued patient
#     label -- only 400/800 in training -- and an unseen dose, e.g. 600 mg, breaks it: +51/+53 % bias at amt600, seeds 1-2)
# Gate as queue_fair.sh (<= 10 heavy python jobs incl. other sessions, CPU < 950 %). Log logs/queue_b2diag.log, marker B2DIAG_DONE.
cd $REPO_ROOT; PY=${PYTHON:-python3}; R=results/regimen_dh/firstint
mkdir -p $R/mapbe_oracle $R/B2_encnorm $R/B2_encnone logs/regimen_dh; LOG=logs/queue_b2diag.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|train_|dh_regimen|lu_regimen|run_regimen|mapbe_regimen|scripts/theory/"; }
cpu () { ps -A -o pcpu= | awk '{s+=$1} END {printf "%d", s}'; }
gate () { while [ $(running) -ge 10 ] || [ $(( $(cpu) + 100 )) -gt 950 ]; do sleep 15; done; }
JOBS=""
for c in 100:sparse:0 100:sparse:2 100:dense:0; do JOBS="$JOBS mapbe:$c"; done
for c in 100:sparse:0 100:sparse:1 100:sparse:2 100:dense:0; do JOBS="$JOBS norm:$c none:$c"; done
say "QUEUE START: $(echo $JOBS | wc -w | tr -d ' ') jobs"; P=""
for j in $JOBS; do
  IFS=: read m n d s <<< "$j"; tag=N${n}_${d}_s$s
  if [ $m = mapbe ]; then out=$R/mapbe_oracle/$tag.json; else out=$R/B2_enc$m/$tag.json; fi
  [ -e $out ] && { say "skip $j"; continue; }
  gate
  if [ $m = mapbe ]; then OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/mapbe_regimen.py $n $d $s $out firstint > logs/regimen_dh/b2diag_${m}_$tag.log 2>&1 &
  else OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen_v2.py $n $d $s $out --iters 6000 --protocol firstint --enc-dose $m > logs/regimen_dh/b2diag_${m}_$tag.log 2>&1 &
  fi
  P="$P $!"; say "start $j (pid $!)"; sleep 10
done
wait $P; say "B2DIAG_DONE"
