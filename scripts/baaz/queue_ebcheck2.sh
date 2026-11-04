#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Takes over from queue_ebcheck.sh, whose launcher was stopped 08/10 ~17:50: its total-%CPU gate reads
# macOS's DECAYING ps %cpu, which under-counts freshly started jobs, so it had 7 jobs running (> 600 % cap).
# 1. waits for the 4 Bfix20 runs (fixed noise 20 mg/L, given priority), rebuilds results/regimen_dh/bfix_table.txt (BFIX20_DONE);
# 2. resumes the 5 EB-VAE runs paused for them (logs/baaz/paused_ebcheck_pids.txt);
# 3. launches the remaining big / true-noise scenario-3 runs, gating on the NUMBER of running (non-stopped) heavy python
#    jobs (<= 6, other sessions' jobs count); skips runs that are done or whose process exists;
# 4. waits, scores every saved model on its training patients, writes the table.  Log logs/queue_ebcheck.log (appended),
#    marker EBCHECK_DONE. Do not edit while running.
R=$REPO_ROOT
EB=$REPO_ROOT/third_party/baaz_ebvae
PY=${PYTHON:-python3}
D=$R/results/baaz_scen3/data
cd $R
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
say () { echo "[$(date '+%F %T')] $*"; }
# count only real python processes: the FIRST word of the command line is the executable (macOS truncates comm to 16
# chars; shells whose command line merely mentions python are not counted); stopped ('T') excluded
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && /run_scen_baaz|dh_regimen|run_models|run_tacro_baaz|track_null|lu_regimen|run_regimen|eval_saved|mapbe/' | wc -l | tr -d ' '; }
gate () { sleep 20; while [ "$(njobs)" -ge 6 ]; do sleep 60; done; }
envof () { [ $1 = big ] && echo "BAAZ_DIMS=2,4,4" || echo "BAAZ_FIXED_NOISE=0.9967,0.1389"; }
dirof () { [ $1 = big ] && echo runs_big || echo runs_truenoise; }
say "queue_ebcheck2 starts; waiting for the Bfix20 runs"
while pgrep -f "Bfix20/N100|Bfixkp20/N100" > /dev/null; do sleep 60; done
$PY scripts/regimen/bfix_table.py > results/regimen_dh/bfix_table.txt 2> logs/bfix_table.err
say "Bfix20 runs finished -> results/regimen_dh/bfix_table.txt; BFIX20_DONE"
P=$(cat logs/baaz/paused_ebcheck_pids.txt); bash -c "kill -CONT $P"; say "resumed paused EB-VAE runs: $P"
for s in $(seq 1 10); do for v in big true; do
  O=$R/results/baaz_scen3/$(dirof $v)
  [ -f $O/eval_test_s${s}_seed${s}_ep1000.json ] && continue
  pgrep -f "run_scen_baaz.py $s .*/$(dirof $v) " > /dev/null && continue
  gate
  ( cd $EB && env $(envof $v) nohup nice -n 5 $PY -u ours/tacro/run_scen_baaz.py $s $D/train_s$s.csv $O 100 100,200,300,500,700,1000 \
      $D/test_s$s.csv > $R/logs/baaz/eb${v}_s$s.log 2>&1 < /dev/null & )
  say "start $v s$s (running heavy jobs before: $(njobs))"
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
