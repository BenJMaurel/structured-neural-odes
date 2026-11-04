#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Variant A, seeds 2-3 of --orth-split 1,5"yes do that"). Seed 1 had a median
# |AUC err| 0.5-0.8 pt below FiLM in range, within seed noise; seeds 2-3 decide whether A is better or merely equal.
# Exactly chain_orth.sh's command for 1-5 with --seed 2/3 and their own save roots. The code has since gained
# --dose-history; the osplit1-5 path was verified unchanged (seed-1 ckpt eval byte-identical, 3-iteration
# training losses identical).
#   train   windowed s4 N=100, 6000 epochs -> results/s4_orth/osplit1-5_n100_s{2,3}/
#   score   ep6000 on in / lo / hi, scaler pinned, seeded -> results/s4_orth/eval/ep006000_osplit1-5_n100_s<s>_on_<reg>.json
#   table   scripts/s4/orth15_table.py -> results/s4_orth/orth15_table.txt (in range, seed-level + paired)
# Markers in logs/chain_orth15_seeds.log: ORTH15_TRAINED, ORTH15_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_orth
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
SEEDS="2 3"
for s in $SEEDS; do
  D=$R/osplit1-5_n100_s$s
  [ -e $D ] && { say "REFUSING: $D exists (would overwrite a run)"; exit 1; }
  mkdir -p $D/exp_film_run/$T
  nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --orth-split 1,5 --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __osplit1-5_s$s \
    > logs/train_osplit1-5_s$s.log 2>&1 &
  say "launched osplit 1-5 seed $s (pid $!)"; sleep 5
done
ck () { echo $R/osplit1-5_n100_s$1/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_osplit1-5_ep006000.ckpt; }
for s in $SEEDS; do
  while [ ! -f "$(ck $s)" ]; do
    pgrep -f "log-suffix __osplit1-5_s$s" >/dev/null || { sleep 30; [ -f "$(ck $s)" ] || { say "TRAINING DIED osplit 1-5 s$s"; break; }; }
    sleep 60
  done
done
say "ORTH15_TRAINED"
for s in $SEEDS; do
  C=$(ck $s); [ -f "$C" ] || continue
  for reg in in lo hi; do
    case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
      --ckpt $C --scale-from $T --eval-split test --label osplit1-5_s${s}_$reg --out-json $R/eval/ep006000_osplit1-5_n100_s${s}_on_$reg.json \
      > logs/eval_osplit1-5_s${s}_$reg.log 2>&1 &
  done
  wait
  say "scored osplit 1-5 s$s"
done
$PY scripts/s4/orth15_table.py > $R/orth15_table.txt 2> logs/orth15_table.err
say "table -> $R/orth15_table.txt"
say "ORTH15_DONE"
