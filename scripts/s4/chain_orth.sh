#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Variant A of the unified dose-action design, "go ahead"): OT-FiLM main config + --orth-split
# k_f,k_d = learned orthonormal latent basis split into FiLM / dynamics / physiology subspaces (physiology conserved by
# the transport AND the flow). Question: does conserving physiology cost accuracy vs the unconstrained Table 3 FiLM?
#   train   windowed s4 N=100 (confound_vc00_s4_pnoise_win_n100), 6000 epochs, osplit 1,3 seeds 1-3 + osplit 1,5 seed 1
#   score   ep6000 on in / lo / hi, scaler pinned, seeded -> results/s4_orth/eval/ep006000_osplit<kf-kd>_n100_s<s>_on_<reg>.json
#   check   scripts/theory/latent_geometry.py on the new checkpoints -> results/theory/geometry/osplit*.{txt,json}
#   table   scripts/s4/orth_table.py -> results/s4_orth/orth_table.txt
# Markers in logs/chain_orth.log: ORTH_TRAINED, ORTH_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_orth
mkdir -p $R/eval logs results/theory/geometry
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres
RUNS="1-3:1 1-3:2 1-3:3 1-5:1"
for run in $RUNS; do
  k=${run%%:*}; s=${run##*:}; D=$R/osplit${k}_n100_s$s; mkdir -p $D/exp_film_run/$T
  nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --orth-split ${k/-/,} --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __osplit${k}_s$s \
    > logs/train_osplit${k}_s$s.log 2>&1 &
  say "launched osplit $k seed $s (pid $!)"; sleep 5
done
ck () { echo $R/osplit$1_n100_s$2/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_osplit$1_ep006000.ckpt; }
for run in $RUNS; do
  k=${run%%:*}; s=${run##*:}
  while [ ! -f "$(ck $k $s)" ]; do
    pgrep -f "log-suffix __osplit${k}_s$s" >/dev/null || { sleep 30; [ -f "$(ck $k $s)" ] || { say "TRAINING DIED osplit $k s$s"; break; }; }
    sleep 60
  done
done
say "ORTH_TRAINED"
for run in $RUNS; do
  k=${run%%:*}; s=${run##*:}; C=$(ck $k $s); [ -f "$C" ] || continue
  for reg in in lo hi; do
    case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
      --ckpt $C --scale-from $T --eval-split test --label osplit${k}_s${s}_$reg --out-json $R/eval/ep006000_osplit${k}_n100_s${s}_on_$reg.json \
      > logs/eval_osplit${k}_s${s}_$reg.log 2>&1 &
  done
  wait
  say "scored osplit $k s$s"
done
OMP_NUM_THREADS=2 $PY scripts/theory/latent_geometry.py $B $T results/theory/geometry/osplit1-3 $(ck 1-3 1) $(ck 1-3 2) $(ck 1-3 3) > logs/geometry_osplit1-3.log 2>&1
OMP_NUM_THREADS=2 $PY scripts/theory/latent_geometry.py $B $T results/theory/geometry/osplit1-5 $(ck 1-5 1) > logs/geometry_osplit1-5.log 2>&1
say "geometry done"
$PY scripts/s4/orth_table.py > $R/orth_table.txt 2> logs/orth_table.err
say "table -> $R/orth_table.txt"
say "ORTH_DONE"
