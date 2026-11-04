#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Seeds 4-6 of the unconstrained Table 3 FiLM AND of variant A (--orth-split 1,5): machine free
# until 23:00, "fill it the best possible"; ). With seeds 1-3 (A 2-3 from chain_orth15_seeds.sh) this
# gives 6 vs 6 seeds for "is A better than FiLM in range" (3 vs 3 cannot reach p < 0.1 at the seed level).
# Commands = the Table 3 FiLM args read from its ckpts (results/s4_pnoise_win/film_n100_s1) and chain_orth's 1-5 command.
# FiLM 4-6 go to results/s4_seeds/ so the Table 3 directory (read by ndm_tables.py) is untouched.
# Each seed is scored (in / lo / hi, scaler pinned, seeded) as soon as its ep6000 ckpt exists, then
# scripts/s4/orth15_table.py -> results/s4_orth/orth15_table.txt. Markers in logs/chain_seeds456.log: SEEDS456_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
COMMON="--niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 \
  --experiment $T --film-no-z0-cond --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual \
  --select-on mse_v2 --patience 1000000 --static-dim 4 --eval-every 300 --ckpt-every 300"
RUNS="film:4 osplit1-5:4 film:5 osplit1-5:5 film:6 osplit1-5:6"
root () { case $1 in film) echo results/s4_seeds/film_n100_s$2;; *) echo results/s4_orth/$1_n100_s$2;; esac; }
evdir () { case $1 in film) echo results/s4_seeds/eval;; *) echo results/s4_orth/eval;; esac; }
ck () { local t=""; [ $1 != film ] && t="_$1"; echo $(root $1 $2)/exp_film_run/$T/traj/experiment_film_${T}__${TAG}${t}_ep006000.ckpt; }
for run in $RUNS; do
  c=${run%%:*}; s=${run##*:}; D=$(root $c $s)
  [ -e $D ] && { say "REFUSING: $D exists (would overwrite a run)"; exit 1; }
done
mkdir -p results/s4_seeds/eval
for run in $RUNS; do
  c=${run%%:*}; s=${run##*:}; D=$(root $c $s); mkdir -p $D/exp_film_run/$T
  X=""; [ $c = osplit1-5 ] && X="--orth-split 1,5"
  nohup nice -n 5 $PY run_models.py $COMMON --seed $s $X --save ./$D/ --log-suffix __${c}_n100_s$s \
    > logs/train_seeds456_${c}_s$s.log 2>&1 &
  say "launched $c seed $s (pid $!)"; sleep 3
done
todo="$RUNS"
while [ -n "$todo" ]; do
  left=""
  for run in $todo; do
    c=${run%%:*}; s=${run##*:}; C=$(ck $c $s)
    if [ -f "$C" ]; then
      pids=""
      for reg in in lo hi; do
        case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
        nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
          --ckpt $C --scale-from $T --eval-split test --label ${c}_s${s}_$reg --out-json $(evdir $c)/ep006000_${c}_n100_s${s}_on_$reg.json \
          > logs/eval_seeds456_${c}_s${s}_$reg.log 2>&1 &
        pids="$pids $!"
      done
      # wait for THESE evals only: a bare `wait` would also wait for the trainings this script started
      wait $pids
      say "scored $c s$s"
    elif ! pgrep -f "log-suffix __${c}_n100_s$s" >/dev/null; then
      sleep 30; [ -f "$C" ] && left="$left $run" || say "TRAINING DIED $c s$s"
    else
      left="$left $run"
    fi
  done
  todo=$(echo $left)
  [ -n "$todo" ] && sleep 60
done
$PY scripts/s4/orth15_table.py > results/s4_orth/orth15_table.txt 2> logs/orth15_table.err
say "table -> results/s4_orth/orth15_table.txt"
say "SEEDS456_DONE"
