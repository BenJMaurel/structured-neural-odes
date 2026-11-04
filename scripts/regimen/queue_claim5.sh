#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Claim 5 with seeds: our dose-event models vs the EB-VAE on Baaz's regimen
# test (N100, randomised arms), linear and saturable (Km 500) simulators, sparse seeds 0-4 and dense seeds 0-2. Models:
#   MAP-BE oracle (true model + population; ceiling)        scripts/regimen/mapbe_regimen.py
#   EB-VAE (their pipeline, unpatched)                      $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/run_regimen_titrated.py ... none
#   improved B, correction penalty 1000 (fixed a priori)    scripts/regimen/dh_regimen_v3.py --resid-dyn 1000 --resid-read 1000
#   free-flow B + physiology-dependent jump, noise 20 mg/L  scripts/regimen/dh_regimen.py --obsrv-mgl 20 --kick-phys 32
# Outputs in the existing layouts (a job is skipped when its output exists): results/regimen_dh/[mm500/]mapbe_oracle/,
# $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/res_{titr/ctrl,mm500/ctrl}/, results/regimen_dh/v3/{lin,mm500}/lam1000/,
# results/regimen_dh/[mm500/]Bfixkp20/. Table: scripts/regimen/claim5_table.py -> results/regimen_dh/claim5_table.txt.
# Gate: <= 6 running python jobs (executable-path counter; shares the machine with queue_vcw_seeds.sh).
# Log logs/queue_claim5.log, per-job logs logs/regimen_dh/c5_*.log, marker CLAIM5_DONE. Do not edit while running.
R=$REPO_ROOT; EB=$REPO_ROOT/third_party/baaz_ebvae; PY=${PYTHON:-python3}
cd $R
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
say () { echo "[$(date '+%F %T')] $*"; }
# Cap: 6 jobs, or 10 until the epoch in QCAP_UNTIL (; set at relaunch). Load = running python jobs of every queue + 2 per Monolix fit.
cap () { if [ "$(date +%s)" -lt "${QCAP_UNTIL:-0}" ]; then echo 10; else echo 6; fi; }
njobs () { echo $(( $(ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && /run_models|train_dose_cond|seeded_eval|test_film_matched|test_dose_cond|train_lu_pk|run_scen_baaz|dh_regimen|run_tacro_baaz|track_null|lu_regimen|run_regimen|eval_saved|map_refine|eval_lode|mapbe|dh_multivisit|gen_tacro|build_monolix/' | wc -l) + 2 * $(pgrep -f fit_mv_pop.R | wc -l) )); }
gate () { sleep 15; while [ "$(njobs)" -ge "$(cap)" ]; do sleep 60; done; }
todo () { [ ! -f "$1" ] && ! pgrep -f "$1" > /dev/null; }   # restart-safe: output missing and no live run writing it
CELLS="sparse:1 sparse:0 sparse:2 sparse:3 sparse:4 dense:0 dense:1 dense:2"
for cell in $CELLS; do d=${cell%%:*}; s=${cell##*:}
  for sim in lin mm; do
    if [ $sim = mm ]; then ENV="REGIMEN_KM=500"; M=mm500/; EBD=res_mm500/ctrl; V=mm500; NC="--no-check"; else ENV=""; M=""; EBD=res_titr/ctrl; V=lin; NC="--no-check"; fi
    tag=${sim}_${d}_s$s
    o=$R/results/regimen_dh/${M}mapbe_oracle/N100_${d}_s$s.json
    if todo $o; then gate; mkdir -p $(dirname $o); env $ENV nohup nice -n 5 $PY scripts/regimen/mapbe_regimen.py 100 $d $s $o > logs/regimen_dh/c5_oracle_$tag.log 2>&1 < /dev/null & say "start oracle $tag"; fi
    o=$EB/ours/regimen/$EBD/N100_${d}_s$s.json
    if todo $o; then gate; mkdir -p $(dirname $o); ( cd $EB && env $ENV nohup nice -n 5 $PY -u ours/regimen/run_regimen_titrated.py 100 $d $s none $o > $R/logs/regimen_dh/c5_ebvae_$tag.log 2>&1 < /dev/null & ); say "start EB-VAE $tag"; fi
    o=$R/results/regimen_dh/v3/$V/lam1000/N100_${d}_s$s.json
    if todo $o; then gate; env $ENV nohup nice -n 5 $PY scripts/regimen/dh_regimen_v3.py 100 $d $s $o --iters 6000 --resid-dyn 1000 --resid-read 1000 > logs/regimen_dh/c5_impB_$tag.log 2>&1 < /dev/null & say "start improved B $tag"; fi
    o=$R/results/regimen_dh/${M}Bfixkp20/N100_${d}_s$s.json
    if todo $o; then gate; mkdir -p $(dirname $o); env $ENV nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s $o --iters 6000 --obsrv-mgl 20 --kick-phys 32 $NC > logs/regimen_dh/c5_Bfree_$tag.log 2>&1 < /dev/null & say "start free-flow B + b(p) $tag"; fi
  done
done
sleep 30; while pgrep -f "mapbe_regimen.py 100|run_regimen_titrated.py 100|dh_regimen_v3.py 100|dh_regimen.py 100" > /dev/null; do sleep 60; done
say "all runs finished; logs without a completion line: $(grep -L -E 'DONE|wrote|saved' logs/regimen_dh/c5_*.log | tr '\n' ' ')"
$PY scripts/regimen/claim5_table.py > results/regimen_dh/claim5_table.txt 2> logs/claim5_table.err
say "table -> results/regimen_dh/claim5_table.txt"
say "CLAIM5_DONE"
