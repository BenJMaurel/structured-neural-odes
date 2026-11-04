#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Two-way comparison, Part 2 (2026-10-03): EB-VAE (Baaz et al. 2026 + opt-in patches P1-P3) vs our OT-FiLM on the
# VARIANT part2b (03/10 13:35: no below/above-range cohorts): in-range only; waits for the in-range generator (pid $GEN_PID) instead of GEN_SWITCH_DONE, then makes the _n100 subset itself.
# realistic dose-switch cohort (confound_vc00_s4k3_switch: k=3 saturation, visit 1 at exact steady state, switch to d2,
# visit 2 four days later; 100 training patients, ~1100 test; in / lo / hi). Waits for GEN_SWITCH_DONE.
#   ours  : run_models.py OT-FiLM main config (as chain_table1_window.sh), seeds 1-3, 6000 epochs, scored at ep6000 with
#           seeded_eval + test_film_matched (scaler pinned to the training cohort) -> results/baaz_part2/eval/
#   theirs: $REPO_ROOT/third_party/baaz_ebvae/ours/tacro/run_tacro_baaz.py, seeds 1-3, stages 100 + 300 epochs -> results/baaz_part2/baaz/
# Table: scripts/baaz/part2_table.py -> results/baaz_part2/part2_table.txt. Log logs/chain_part2b.log, marker PART2_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; B=confound_vc00_s4k3_switch; T=${B}_n100; R=results/baaz_part2; BZ=$REPO_ROOT/third_party/baaz_ebvae
mkdir -p $R/eval $R/baaz $R/data logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
say () { echo "[$(date '+%F %T')] $*"; }
GEN_PID=${GEN_PID:-17609}
while kill -0 $GEN_PID 2>/dev/null; do sleep 30; done
[ -f results/exp_film_run/$B/confound_truth.csv ] || { say "IN-RANGE COHORT MISSING"; exit 1; }
$PY scripts/nsweep/make_subsets.py $B "100" || { say "SUBSET FAILED"; exit 1; }
cp results/exp_film_run/$B/noise.json results/exp_film_run/$T/ 2>/dev/null
say "in-range cohort ready"; grep -E "switch design|exposure window|train .* / test" logs/gen_switch_in.log
# ---- data for the EB-VAE ----
( cd $BZ && $PY ours/tacro/convert_switch.py $OLDPWD/results/exp_film_run/$T $OLDPWD/$R/data/train.csv train \
  && $PY ours/tacro/convert_switch.py $OLDPWD/results/exp_film_run/$B $OLDPWD/$R/data/test_in.csv test ) || { say "CONVERT FAILED"; exit 1; }
# ---- launch: ours (3 FiLM) + theirs (3 EB-VAE) ----
PIDS=""
for s in 1 2 3; do
  D=$R/film_s$s; mkdir -p $D/exp_film_run/$T
  nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __part2_s$s > logs/train_part2_film_s$s.log 2>&1 &
  PIDS="$PIDS $!"; say "launched FiLM seed $s (pid $!)"; sleep 10
  ( cd $BZ && nohup nice -n 5 $PY -u ours/tacro/run_tacro_baaz.py $s $OLDPWD/$R/data/train.csv $OLDPWD/$R/baaz 100 300 \
      $OLDPWD/$R/data/test_in.csv ) > logs/train_part2_baaz_s$s.log 2>&1 &
  PIDS="$PIDS $!"; say "launched EB-VAE seed $s (pid $!)"; sleep 10
done
wait $PIDS; say "PART2_TRAINED"
# ---- score ours ----
EV=""
for s in 1 2 3; do
  CK=$(ls $R/film_s$s/exp_film_run/$T/traj/*_ep006000.ckpt 2>/dev/null)
  [ "$(echo "$CK" | grep -c ckpt)" = 1 ] || { say "MISSING ckpt FiLM s$s"; continue; }
  for reg in in; do ex=$B
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
      --ckpt $CK --scale-from $T --eval-split test --label part2_film_s${s}_$reg --out-json $R/eval/ep006000_film_n100_s${s}_on_$reg.json \
      > logs/eval_part2_film_s${s}_$reg.log 2>&1 &
    EV="$EV $!"
  done
done
wait $EV; say "scored ours: $(ls $R/eval | wc -l | tr -d ' ') files; EB-VAE evals: $(ls $R/baaz/eval_*.json 2>/dev/null | wc -l | tr -d ' ')"
$PY scripts/baaz/part2_table.py > $R/part2_table.txt 2>&1; say "table -> $R/part2_table.txt"
say "PART2_DONE"
