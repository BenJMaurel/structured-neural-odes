#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Why does the EB-VAE lose on our tacrolimus benchmarks?. Check 1 (training
# vs test patients, saved models) is already done -> results/baaz_check/trainvstest/. This queue runs checks 2 and 3 on
# scenario 3 (same 10 cohorts, same budgets as results/baaz_scen3/runs = the published configuration):
#   big   BAAZ_DIMS=2,4,4            bigger bottleneck (2 IC + 4 dynamic latents, 4-D state; published 1,2,2)
#   true  BAAZ_FIXED_NOISE=0.9967,0.1389   noise of both stages FIXED at the closest combined-2 fit to the simulator's true
#                                     sd = 0.71 + 0.113 C (within +-11 % over the 5-95 % concentration range)
# then scores every saved model of both variants on its TRAINING patients (eval_saved.py), then the table.
# Gate: total python CPU < 500 % before each 1-thread job (600 % cap; other sessions' jobs count).
#   outputs  results/baaz_scen3/runs_{big,truenoise}/, results/baaz_check/trainvstest/, results/baaz_check/ebcheck_table.txt
#   log      logs/queue_ebcheck.log, per-run logs logs/baaz/eb{big,true}_s<seed>.log, marker EBCHECK_DONE. Do not edit while running.
R=$REPO_ROOT
EB=$REPO_ROOT/third_party/baaz_ebvae
PY=${PYTHON:-python3}
D=$R/results/baaz_scen3/data
cd $R
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -eo pcpu=,command= | awk '/python/ {s += $1} END {print int(s)}'; }
gate () { sleep 30; while [ "$(cpu)" -ge 500 ]; do sleep 60; done; }
mkdir -p logs/baaz results/baaz_scen3/runs_big results/baaz_scen3/runs_truenoise results/baaz_check/trainvstest
envof () { [ $1 = big ] && echo "BAAZ_DIMS=2,4,4" || echo "BAAZ_FIXED_NOISE=0.9967,0.1389"; }
dirof () { [ $1 = big ] && echo runs_big || echo runs_truenoise; }
for s in $(seq 1 10); do for v in big true; do
  O=$R/results/baaz_scen3/$(dirof $v)
  [ -f $O/eval_test_s${s}_seed${s}_ep1000.json ] && { say "skip $v s$s (done)"; continue; }
  gate
  ( cd $EB && env $(envof $v) nohup nice -n 5 $PY -u ours/tacro/run_scen_baaz.py $s $D/train_s$s.csv $O 100 100,200,300,500,700,1000 \
      $D/test_s$s.csv > $R/logs/baaz/eb${v}_s$s.log 2>&1 < /dev/null & echo $! > $R/logs/baaz/eb${v}_s$s.pid )
  say "start $v s$s (pid $(cat logs/baaz/eb${v}_s$s.pid), cpu before $(cpu) %)"
done; done
while pgrep -f "run_scen_baaz.py" > /dev/null; do sleep 60; done
say "trainings finished; failures: $(grep -L SCEN_BAAZ_DONE logs/baaz/ebbig_s*.log logs/baaz/ebtrue_s*.log | tr '\n' ' ')"
for v in big true; do for s in $(seq 1 10); do for E in 300 1000; do
  M=$R/results/baaz_scen3/$(dirof $v)/models_seed${s}_ep${E}.pt; [ -f $M ] || continue
  for w in train test; do
    ( cd $EB && env $(envof $v) $PY ours/tacro/eval_saved.py scen $M $D/train_s$s.csv $D/test_s$s.csv $D/${w}_s$s.csv \
        $R/results/baaz_check/trainvstest/scen3${v}_s${s}_ep${E}_on_${w}.json >> $R/logs/baaz/ebcheck_eval.log 2>&1 )
  done
done; done; done
say "training-patient scores done"
$PY scripts/baaz/ebcheck_table.py > results/baaz_check/ebcheck_table.txt 2> logs/baaz/ebcheck_table.err
say "table -> results/baaz_check/ebcheck_table.txt"
say "EBCHECK_DONE"
