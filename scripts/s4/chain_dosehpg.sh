#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Variant B FIXED: the first B run gave every patient 24 h dosing intervals, but
# ~40 % of the windowed s4 patients are Prograf, dosed every 12 h. --dose-interval-prograf 12 gives each patient the
# interval of its formulation (static[:,1]). Everything else as chain_doseh.sh (main FiLM config + --orth-split 1,5
# --dose-history 7), seeds 1-3, own roots results/s4_doseh/dh7pg_osplit1-5_n100_s*.
#   score   ep6000 on in / lo / hi -> results/s4_doseh/eval/ep006000_dh7pg_osplit1-5_n100_s<s>_on_<reg>.json
#   check   scripts/theory/doseh_check.py -> results/s4_doseh/check_dh7pg.txt
#   table   scripts/s4/doseh_table.py -> results/s4_doseh/doseh_table.txt (both B versions vs FiLM and A)
# Markers in logs/chain_dosehpg.log: DOSEHPG_TRAINED, DOSEHPG_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_doseh
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres_osplit1-5_dh7_taupg12
SEEDS="1 2 3"
for s in $SEEDS; do
  D=$R/dh7pg_osplit1-5_n100_s$s
  [ -e $D ] && { say "REFUSING: $D exists"; exit 1; }
done
for s in $SEEDS; do
  D=$R/dh7pg_osplit1-5_n100_s$s; mkdir -p $D/exp_film_run/$T
  nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --orth-split 1,5 --dose-history 7 --dose-interval 24 --dose-interval-prograf 12 --save ./$D/ \
    --eval-every 300 --ckpt-every 300 --log-suffix __dh7pg_osplit1-5_s$s > logs/train_dh7pg_osplit1-5_s$s.log 2>&1 &
  say "launched dh7pg osplit 1-5 seed $s (pid $!)"; sleep 5
done
ck () { echo $R/dh7pg_osplit1-5_n100_s$1/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_ep006000.ckpt; }
for s in $SEEDS; do
  while [ ! -f "$(ck $s)" ]; do
    pgrep -f "log-suffix __dh7pg_osplit1-5_s$s" >/dev/null || { sleep 30; [ -f "$(ck $s)" ] || { say "TRAINING DIED dh7pg s$s"; break; }; }
    sleep 60
  done
done
say "DOSEHPG_TRAINED"
for s in $SEEDS; do
  C=$(ck $s); [ -f "$C" ] || continue
  for reg in in lo hi; do
    case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
      --ckpt $C --scale-from $T --eval-split test --label dh7pg_osplit1-5_s${s}_$reg --out-json $R/eval/ep006000_dh7pg_osplit1-5_n100_s${s}_on_$reg.json \
      > logs/eval_dh7pg_osplit1-5_s${s}_$reg.log 2>&1 &
  done
  wait
  say "scored dh7pg s$s"
done
OMP_NUM_THREADS=2 $PY scripts/theory/doseh_check.py $B $T $(for s in $SEEDS; do ck $s; done) > $R/check_dh7pg.txt 2> logs/doseh_check_pg.err
say "check -> $R/check_dh7pg.txt"
$PY scripts/s4/doseh_table.py > $R/doseh_table.txt 2> logs/doseh_table.err
say "table -> $R/doseh_table.txt"
say "DOSEHPG_DONE"
