#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# One overnight job (queue_night.sh): train the main FiLM config (optionally + --orth-split 1,K) on a
# windowed s4 N=100 cohort for 6000 epochs, then score ep6000 on in / lo / hi (scaler pinned to the training cohort,
# seeded). Refuses to start if its save root exists.
#   night_job.sh <pnoise|tslow> <film|osplit1-K> <seed>
cd $REPO_ROOT
coh=$1; kind=$2; s=$3
PY=${PYTHON:-python3}
B=confound_vc00_s4_${coh}_win; T=${B}_n100
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
LOG=logs/queue_night.log
say () { echo "[$(date '+%F %T')] $coh $kind s$s: $*" >> $LOG; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
case $coh:$kind in
  pnoise:film) D=results/s4_seeds/film_n100_s$s; E=results/s4_seeds/eval;;
  pnoise:*)    D=results/s4_orth/${kind}_n100_s$s; E=results/s4_orth/eval;;
  tslow:*)     D=results/s4_tslow_win/${kind}_n100_s$s; E=results/s4_tslow_win/eval;;
  *) say "UNKNOWN job"; exit 1;;
esac
X=""; t=""
if [ $kind != film ]; then k=${kind#osplit}; X="--orth-split ${k/-/,}"; t="_$kind"; fi
[ -e $D ] && { say "REFUSING: $D exists"; exit 1; }
mkdir -p $D/exp_film_run/$T $E
say "train start ($D)"
nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 \
  --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 --film-self-consistency 141.27 \
  --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 --static-dim 4 $X --save ./$D/ \
  --eval-every 300 --ckpt-every 300 --log-suffix __night_${coh}_${kind}_s$s > logs/train_night_${coh}_${kind}_s$s.log 2>&1
C=$D/exp_film_run/$T/traj/experiment_film_${T}__${TAG}${t}_ep006000.ckpt
[ -f "$C" ] || { say "TRAINING FAILED (no ep6000 ckpt; see logs/train_night_${coh}_${kind}_s$s.log)"; exit 1; }
say "trained; scoring"
for reg in in lo hi; do
  case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
  nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex \
    --data-dir ./results/exp_film_run --ckpt $C --scale-from $T --eval-split test --label ${kind}_s${s}_$reg \
    --out-json $E/ep006000_${kind}_n100_s${s}_on_$reg.json > logs/eval_night_${coh}_${kind}_s${s}_$reg.log 2>&1 &
done
wait
for reg in in lo hi; do [ -f $E/ep006000_${kind}_n100_s${s}_on_$reg.json ] || say "EVAL FAILED on $reg"; done
say "DONE"
