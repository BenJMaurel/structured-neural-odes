#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# One job of the latent-14 study . Windowed pnoise N=100, 6000 epochs, one core at a time (600 % cap).
#   arch_job.sh <L> film       <seed>   Table 3 FiLM main config, -l L            -> results/s4_latent/film_l<L>_n100_s<s>
#   arch_job.sh <L> osplit1-K  <seed>   + --orth-split 1,K (variant A)             -> results/s4_latent/osplit1-K_l<L>_n100_s<s>
#   arch_job.sh <L> lode       <seed>   plain latent ODE (paper command, linear decoder, Table 3 ablation row's flags) on
#                                       the same windowed cohort -> results/lode_s4pnwin_l<L>/s<s>; V1 only
# film / osplit: ep6000 scored in / lo / hi sequentially (scaler pinned, seeded) -> results/s4_latent/eval/.
# lode: final ckpt scored by scripts/repro/eval_lode.py, paper and harness conventions -> results/lode_s4pnwin_l<L>/eval/.
# Refuses existing roots.
cd $REPO_ROOT
L=$1; kind=$2; s=$3
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
LOG=logs/queue_l14.log
say () { echo "[$(date '+%F %T')] l$L $kind s$s: $*" >> $LOG; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [ $kind = lode ]; then
  E=s4pnwin_lode_n100; D=results/lode_s4pnwin_l$L/s$s; EV=results/lode_s4pnwin_l$L/eval
  [ -e $D ] && { say "REFUSING: $D exists"; exit 1; }
  # run_models saves into $D/$E: it must exist (the 04/10 14:45-16:42 runs crashed at the first save without it)
  mkdir -p $D/$E $EV; say "train start"
  nice -n 5 $PY run_models.py --latent-ode --dataset PK_Tacro -n 200 --niters 6000 -l $L -b 512 -s 40 --noise-weight 0.01 \
    --static-dim 4 --n-train-series 200 --experiment $E --seed $s --save ./$D/ --log-suffix __lode_l${L}_s$s \
    > logs/train_l14q_lode_l${L}_s$s.log 2>&1
  C=$D/$E/experiment_$E.ckpt
  [ -f "$C" ] || { say "TRAINING FAILED (no final ckpt)"; exit 1; }
  nice -n 5 $PY scripts/repro/eval_lode.py $C $EV/s${s}_final.json > logs/eval_l14q_lode_l${L}_s$s.log 2>&1 || say "EVAL FAILED (paper conv)"
  nice -n 5 $PY scripts/repro/eval_lode.py $C $EV/s${s}_final_harness.json harness > logs/eval_l14q_lode_l${L}_s${s}_h.log 2>&1 || say "EVAL FAILED (harness)"
  say "DONE"; exit 0
fi
D=results/s4_latent/${kind}_l${L}_n100_s$s; EV=results/s4_latent/eval
X=""; t=""
if [ $kind != film ]; then k=${kind#osplit}; X="--orth-split ${k/-/,}"; t="_$kind"; fi
[ -e $D ] && { say "REFUSING: $D exists"; exit 1; }
mkdir -p $D/exp_film_run/$T $EV; say "train start"
nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l $L --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 \
  --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 --film-self-consistency 141.27 \
  --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 --static-dim 4 $X --save ./$D/ \
  --eval-every 300 --ckpt-every 300 --log-suffix __l14q_${kind}_l${L}_s$s > logs/train_l14q_${kind}_l${L}_s$s.log 2>&1
C=$D/exp_film_run/$T/traj/experiment_film_${T}__${TAG}${t}_ep006000.ckpt
[ -f "$C" ] || { say "TRAINING FAILED (no ep6000 ckpt)"; exit 1; }
for reg in in lo hi; do
  case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
  nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex \
    --data-dir ./results/exp_film_run --ckpt $C --scale-from $T --eval-split test --label ${kind}_l${L}_s${s}_$reg \
    --out-json $EV/ep006000_${kind}_l${L}_n100_s${s}_on_$reg.json > logs/eval_l14q_${kind}_l${L}_s${s}_$reg.log 2>&1 \
    || say "EVAL FAILED on $reg"
done
say "DONE"
