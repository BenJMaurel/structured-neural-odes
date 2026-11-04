#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# EB-VAE convergence + scenario 3 (2026-10-07 and
# "try testing it against the latent ODE on scenario 3").
#  (1) Part 2 cohort, seeds 1-3: $REPO_ROOT/third_party/baaz_ebvae/ours/tacro/run_tacro_baaz_curve.py, stage 2 to 1000 epochs, test_in
#      scored at 300 (must equal results/baaz_part2/baaz/eval_test_in_seed<s>.json), 400, ..., 1000 -> results/baaz_part2/curve/
#  (2) NMI 3-scenario study, scenario 3, seeds 1-10 (data = the rerun's own cohorts, converted by convert_scen.py):
#      run_scen_baaz.py, scored at stage-2 epochs 100, 200, 300 (their default), 500, 700, 1000 -> results/baaz_scen3/runs/
# Gate: < 6 heavy python jobs incl. other sessions. Log logs/queue_ebconv.log, marker EBCONV_DONE.
cd $REPO_ROOT; PY=${PYTHON:-python3}; EB=$REPO_ROOT/third_party/baaz_ebvae; R=$(pwd)
mkdir -p results/baaz_part2/curve results/baaz_scen3/runs logs/baaz; LOG=logs/queue_ebconv.log; say () { echo "[$(date '+%F %T')] $*" >> $LOG; }
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|train_|dh_regimen|lu_regimen|run_regimen|run_tacro_baaz|run_scen_baaz|mapbe|scripts/theory/"; }
gate () { while [ $(running) -ge 6 ]; do sleep 30; done; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
P=""; D=$R/results/baaz_part2/data
for s in 1 2 3; do
  gate
  ( cd $EB && nohup nice -n 5 $PY -u ours/tacro/run_tacro_baaz_curve.py $s $D/train.csv $R/results/baaz_part2/curve 100 \
      300,400,500,600,700,800,900,1000 $D/test_in.csv > $R/logs/baaz/ebcurve_s$s.log 2>&1 ) &
  P="$P $!"; say "start Part 2 curve seed $s (pid $!)"; sleep 30
done
D=$R/results/baaz_scen3/data
for s in $(seq 1 10); do
  gate
  ( cd $EB && nohup nice -n 5 $PY -u ours/tacro/run_scen_baaz.py $s $D/train_s$s.csv $R/results/baaz_scen3/runs 100 \
      100,200,300,500,700,1000 $D/test_s$s.csv > $R/logs/baaz/ebscen3_s$s.log 2>&1 ) &
  P="$P $!"; say "start scenario-3 seed $s (pid $!)"; sleep 30
done
wait $P; say "EBCONV_DONE"
