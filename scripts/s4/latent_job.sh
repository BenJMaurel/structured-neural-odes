#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# One job of the latent-size sweep
# ): the Table 3 FiLM main config with -l <L> on windowed pnoise N=100, 6000 epochs, then ep6000 scored on
# in / lo / hi ONE AFTER ANOTHER (scaler pinned, seeded), so a job never uses more than one core (600 % cap = MAXJ 6).
# Own root per (L, seed): the checkpoint tag does not encode -l (README.md gotcha 4). Refuses existing roots.
#   latent_job.sh <L> <seed>
cd $REPO_ROOT
L=$1; s=$2
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
D=results/s4_latent/film_l${L}_n100_s$s; E=results/s4_latent/eval
LOG=logs/queue_latent.log
say () { echo "[$(date '+%F %T')] l$L s$s: $*" >> $LOG; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
[ -e $D ] && { say "REFUSING: $D exists"; exit 1; }
mkdir -p $D/exp_film_run/$T $E
say "train start"
nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l $L --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 \
  --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 --film-self-consistency 141.27 \
  --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 --static-dim 4 --save ./$D/ \
  --eval-every 300 --ckpt-every 300 --log-suffix __latent_l${L}_s$s > logs/train_latent_l${L}_s$s.log 2>&1
C=$D/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_ep006000.ckpt
[ -f "$C" ] || { say "TRAINING FAILED (no ep6000 ckpt; see logs/train_latent_l${L}_s$s.log)"; exit 1; }
for reg in in lo hi; do
  case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
  nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex \
    --data-dir ./results/exp_film_run --ckpt $C --scale-from $T --eval-split test --label film_l${L}_s${s}_$reg \
    --out-json $E/ep006000_film_l${L}_n100_s${s}_on_$reg.json > logs/eval_latent_l${L}_s${s}_$reg.log 2>&1 \
    || say "EVAL FAILED on $reg"
done
say "DONE"
