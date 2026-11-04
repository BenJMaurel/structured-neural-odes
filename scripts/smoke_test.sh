#!/bin/bash
# Checks the installation: each main step for a few iterations on a small cohort. The numbers mean nothing.
# Writes only in results/, out/ and third_party/baaz_ebvae/ours/regimen/res/, all ignored by git. Monolix and R are not needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
C=smoke_cohort; T=${C}_n20; O=$PWD/out/smoke; rm -rf "$O"; mkdir -p "$O" logs
say () { echo "[$(date '+%H:%M:%S')] $*"; }

say "1/9 two-visit cohort"
$PY gen_tacro_confound.py --exp $C --num_patients 150 --rho 0.0 --test-fraction 0.5 --seed 0 --dose-seed 1234 --confound-param Vc \
  --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 --prop-sd 0.113 --add-sd 0.71 --v1-cavg-window 3,25 > "$O/gen.log" 2>&1
$PY scripts/nsweep/make_subsets.py $C "20" >> "$O/gen.log" 2>&1

say "2/9 dose-shift model, training"
mkdir -p "$O/film/exp_film_run/$T"
$PY run_models.py --niters 4 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 64 \
  --seed 1 --experiment $T --film-no-z0-cond --film-gamma-pos --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 \
  --decoder-residual --select-on mse_v2 --patience 1000000 --static-dim 4 --save "$O/film/" --eval-every 4 --ckpt-every 4 > "$O/film.log" 2>&1
CK=$(find "$O/film" -name '*_ep000004.ckpt' | head -1); [ -n "$CK" ]

say "3/9 dose-shift model, evaluation"
$PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $C --data-dir ./results/exp_film_run \
  --ckpt "$CK" --scale-from $T --eval-split test --out-json "$O/film.json" > "$O/film_eval.log" 2>&1
[ -s "$O/film.json" ]

say "4/9 dose-in-the-dynamics model, training and evaluation"
mkdir -p "$O/dc/exp_dosecond_run/$T"
$PY train_dose_cond.py --experiment $T --data-dir ./results/exp_film_run --niters 4 -b 64 -l 10 --lr 1e-2 --seed 1 --patience 1000000 \
  --obsrv-std 0.05 --static-dim 4 --decoder-hidden 50 --decoder-residual --save "$O/dc/" --eval-every 4 --ckpt-every 4 > "$O/dc.log" 2>&1
CK=$(find "$O/dc" -name '*_ep000004.ckpt' | head -1); [ -n "$CK" ]
$PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_dose_cond.py --experiment $C --data-dir ./results/exp_film_run \
  --ckpt "$CK" --scale-from $T --eval-split test --out-json "$O/dc.json" > "$O/dc_eval.log" 2>&1
[ -s "$O/dc.json" ]

say "5/9 two-port model of Lu et al."
$PY -W ignore train_lu_pk.py --experiment $T --data-dir ./results/exp_film_run --niters 4 -b 64 --seed 1 --save "$O/lu/" --static-dim 4 \
  --use-static --train-mode counterfactual --ckpt-every 4 --eval-every 4 > "$O/lu.log" 2>&1
CK=$(find "$O/lu" -name '*_ep000004.ckpt' | head -1); [ -n "$CK" ]
$PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_lu_pk.py --experiment $C --data-dir ./results/exp_film_run \
  --ckpt "$CK" --scale-from $T --eval-split test --out-json "$O/lu.json" > "$O/lu_eval.log" 2>&1
[ -s "$O/lu.json" ]

say "6/9 dose-event model, design of Baaz et al."
REGIMENS_CONFIRM=1 $PY scripts/regimen/dh_regimen.py 100 sparse 0 "$O/de_baaz.json" --iters 4 --no-check \
  --obsrv-mgl 20 --kick-phys 32 --enc-dose norm > "$O/de_baaz.log" 2>&1
[ -s "$O/de_baaz.json" ]

say "7/9 empirical Bayes model, design of Baaz et al."
( cd third_party/baaz_ebvae && $PY -u ours/regimen/run_regimen_titrated.py 100 sparse 0 none "$O/eb_baaz.json" 1 1 ) > "$O/eb_baaz.log" 2>&1
[ -s "$O/eb_baaz.json" ]

say "8/9 dose-event model, tacrolimus regimen design"
TACRO_SCENARIO=2 $PY scripts/regimen/dh_regimen.py 100 visit 0 "$O/de_tacro.json" --iters 4 --sim tacro --no-check \
  --obsrv-mgl 3.1 --kick-phys 32 --enc-dose norm > "$O/de_tacro.log" 2>&1
[ -s "$O/de_tacro.json" ]

say "9/9 regimen model of Lu et al."
$PY scripts/regimen/lu_regimen.py is 100 sparse 0 "$O/lu_regimen.json" --iters 4 --no-check > "$O/lu_regimen.log" 2>&1
[ -s "$O/lu_regimen.json" ]

say "all steps ran"
