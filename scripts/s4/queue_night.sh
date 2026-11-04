#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Overnight queue 03->04/10
# 23 jobs of night_job.sh, at most MAXJ at once, in priority order:
#   P1  windowed pnoise: FiLM and A (--orth-split 1,5) seeds 7-10 -> 10 vs 10 seeds for "is A better in range"
#   P2  windowed tslow (2nd cohort, Student-t + 30 % slow absorbers; N=100 subset made 03/10 22:10): FiLM and A 1,5, seeds 1-3
#   P3  split sweep on windowed pnoise: A 1,4 / 1,6 / 1,7, seeds 1-3 (with 1,3 and 1,5: cost vs flow-subspace size)
# then tables (orth15_table.py, night_tables.py tslow / sweep) and the latent geometry of A 1,5 seeds 1-10.
# Log: logs/queue_night.log (one line per job event), marker NIGHT_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
MAXJ=${MAXJ:-8}
LOG=logs/queue_night.log
say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
# interleaved so every question gets early seeds: P1 seed 7, P2 seed 1, P3 seed 1, P1 seed 8, ...
ORDER="pnoise:film:7 pnoise:osplit1-5:7 tslow:film:1 tslow:osplit1-5:1 pnoise:osplit1-4:1 pnoise:osplit1-6:1 pnoise:osplit1-7:1
pnoise:film:8 pnoise:osplit1-5:8 tslow:film:2 tslow:osplit1-5:2 pnoise:osplit1-4:2 pnoise:osplit1-6:2 pnoise:osplit1-7:2
pnoise:film:9 pnoise:osplit1-5:9 tslow:film:3 tslow:osplit1-5:3 pnoise:osplit1-4:3 pnoise:osplit1-6:3 pnoise:osplit1-7:3
pnoise:film:10 pnoise:osplit1-5:10"
n=$(echo $ORDER | wc -w | tr -d ' ')
say "QUEUE START: $n jobs, MAXJ $MAXJ"
echo $ORDER | tr ' ' '\n' | tr ':' ' ' | xargs -P $MAXJ -L 1 bash scripts/s4/night_job.sh
say "QUEUE JOBS FINISHED: $(grep -c ': DONE$' $LOG) done, $(grep -cE 'FAILED|REFUSING' $LOG) failed/refused"
$PY scripts/s4/orth15_table.py > results/s4_orth/orth15_table.txt 2> logs/orth15_table.err
$PY scripts/s4/night_tables.py tslow > results/s4_tslow_win/tslow_table.txt 2> logs/night_tables_tslow.err
$PY scripts/s4/night_tables.py sweep > results/s4_orth/sweep_table.txt 2> logs/night_tables_sweep.err
say "tables written"
CK=""
for s in 1 2 3 4 5 6 7 8 9 10; do
  for r in results/s4_orth/osplit1-5_n100_s$s; do
    c=$r/exp_film_run/confound_vc00_s4_pnoise_win_n100/traj/experiment_film_confound_vc00_s4_pnoise_win_n100__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres_osplit1-5_ep006000.ckpt
    [ -f $c ] && CK="$CK $c"
  done
done
OMP_NUM_THREADS=2 $PY scripts/theory/latent_geometry.py confound_vc00_s4_pnoise_win confound_vc00_s4_pnoise_win_n100 \
  results/theory/geometry/osplit1-5_allseeds $CK > logs/geometry_osplit1-5_allseeds.log 2>&1
say "geometry -> results/theory/geometry/osplit1-5_allseeds.txt"
say "NIGHT_DONE"
