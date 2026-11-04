#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Reruns of "improved B" (dh_regimen_v3.py, lam 1000) on the LINEAR simulator for the claim-5 cells whose queue_claim5.sh job
# died at the data check. dh_regimen_v3.py checks its simulated patients against the EB-VAE data in
# $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/res/N100_<design>_s<seed>_data, which only exists for sparse s0-s2 / dense s0; the EB-VAE
# runs of the claim-5 queue write theirs to res_titr/ctrl/. res/ and res_titr/ctrl/ files are byte-identical where both exist
# (sparse s0, s1, s2: train.csv and test_factual.csv), so res/<tag>_data is symlinked to res_titr/ctrl/<tag>_data and the
# check runs unchanged against the EB-VAE run of the same seed. Cells: sparse s3, sparse s4, dense s1, dense s2.
# A cell is rerun only when its output is missing, no run of it is alive, its queue log ends in a traceback and the EB-VAE data
# exist; one rerun per cell (log c5_impB_lin_<cell>_rerun.log). Slots: total <= 6 python jobs and, while the Vc queue runs,
# claim-5 jobs < 3 (the arbiter's share). At the end, after CLAIM5_DONE, the claim-5 table is regenerated.
# Log logs/rerun_impB_lin.log; marker IMPB_RERUN_DONE. Do not edit while running.
R=$REPO_ROOT; EB=$REPO_ROOT/third_party/baaz_ebvae/ours/regimen; PY=${PYTHON:-python3}
cd $R
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && /run_models|train_dose_cond|seeded_eval|test_film_matched|test_dose_cond|train_lu_pk|run_scen_baaz|dh_regimen|run_tacro_baaz|track_null|lu_regimen|run_regimen|eval_saved|map_refine|eval_lode|mapbe|dh_multivisit|gen_tacro/' | wc -l | tr -d ' '; }
c5 () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && /mapbe_regimen|run_regimen_titrated|dh_regimen/' | wc -l | tr -d ' '; }
slot () { [ "$(njobs)" -lt 6 ] && { ! pgrep -f "bash scripts/confound_eval/queue_vcw_seeds.sh" > /dev/null || [ "$(c5)" -lt 3 ]; }; }
CELLS="sparse:3 sparse:4 dense:1 dense:2"
for cell in $CELLS; do d=${cell%%:*}; s=${cell##*:}; L=$EB/res/N100_${d}_s${s}_data
  [ -e $L ] || [ -L $L ] || { ln -s ../res_titr/ctrl/N100_${d}_s${s}_data $L; say "linked res/N100_${d}_s${s}_data -> res_titr/ctrl/"; }
done
say "rerun_impB_lin starts"
while true; do
  left=0
  for cell in $CELLS; do d=${cell%%:*}; s=${cell##*:}
    o=$R/results/regimen_dh/v3/lin/lam1000/N100_${d}_s$s.json; Q=logs/regimen_dh/c5_impB_lin_${d}_s$s.log; RL=${Q%.log}_rerun.log
    [ -f $o ] && continue
    [ -f $RL ] && ! pgrep -f "dh_regimen_v3.py 100 $d $s .*/v3/lin/" > /dev/null && continue   # rerun already ended (without output)
    left=$((left + 1))
    pgrep -f "dh_regimen_v3.py 100 $d $s .*/v3/lin/" > /dev/null && continue                   # a run is alive
    [ -f $Q ] && grep -q Traceback $Q || continue                                               # the queue has not failed it (yet)
    [ -f $EB/res_titr/ctrl/N100_${d}_s${s}_data/test_factual.csv ] || continue
    slot || continue
    nohup nice -n 5 $PY scripts/regimen/dh_regimen_v3.py 100 $d $s $o --iters 6000 --resid-dyn 1000 --resid-read 1000 > $RL 2>&1 < /dev/null &
    say "rerun improved B lin_${d}_s$s (pid $!, python jobs now $(njobs), claim-5 jobs $(c5))"; sleep 20
  done
  [ $left = 0 ] && break
  sleep 60
done
for cell in $CELLS; do d=${cell%%:*}; s=${cell##*:}
  o=results/regimen_dh/v3/lin/lam1000/N100_${d}_s$s.json; say "lin_${d}_s$s: $([ -f $o ] && echo output || echo MISSING) $(grep -h 'data check' logs/regimen_dh/c5_impB_lin_${d}_s${s}*.log 2>/dev/null | tr '\n' ' ')"
done
[ -n "${SKIP_WAIT:-}" ] || until grep -q CLAIM5_DONE logs/queue_claim5.log 2>/dev/null || ! pgrep -f "bash scripts/regimen/queue_claim5.sh" > /dev/null; do sleep 120; done
while pgrep -f "mapbe_regimen.py 100|run_regimen_titrated.py 100|dh_regimen_v3.py 100|dh_regimen.py 100" > /dev/null; do sleep 60; done
$PY scripts/regimen/claim5_table.py > results/regimen_dh/claim5_table.txt 2> logs/claim5_table.err
say "table regenerated -> results/regimen_dh/claim5_table.txt"
say "IMPB_RERUN_DONE"
