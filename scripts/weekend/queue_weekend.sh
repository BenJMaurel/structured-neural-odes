#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Weekend runs 17-18/10.
#  1. Confirmation of the regimen benchmark (Table 3) on FRESH seeds with five held-out regimens (REGIMENS_CONFIRM=1, opt-in in
#     $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/gen_regimens.py): frozen dose-event model (Bfixkp20n flags) vs EB-VAE (defaults);
#     sparse titrated (rho 0.8) s5-14, sparse randomised s5-14, dense randomised s3-7.
#     -> results/regimen_dh/confirm/{ctrl,titr0.8}/Bfixkp20n/  and  $REPO_ROOT/third_party/baaz_ebvae/ours/regimen/res_titr/confirm/{ctrl,titr0.8}/
#  2. Table 2 (two-visit task): OT-FiLM residual and dose-cond residual, the commands of scripts/s4/chain_table1_window.sh, init seeds
#     4-10 + a reproducibility rerun of seed 1 -> results/s4_pnoise_win_more/ (results/s4_pnoise_win/ untouched), scored at ep6000
#     like the chain (seeded_eval --eval-seed 0, --scale-from the training cohort, in/lo/hi + all).
# Gate <= 8 python3 jobs (cap 800 %), re-checked after 41 s. Tables at the end if their scripts exist. Marker WEEKEND_DONE.
# Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}; ROOT=$PWD; L=logs/weekend; mkdir -p $L
EBD=$REPO_ROOT/third_party/baaz_ebvae; RT=$EBD/ours/regimen/res_titr/confirm; RD=results/regimen_dh/confirm
T=confound_vc00_s4_pnoise_win_n100; BC=confound_vc00_s4_pnoise_win; R2=results/s4_pnoise_win_more; RC=results/recal/paper_win_more
BFLAGS="--obsrv-mgl 20 --kick-phys 32 --enc-dose norm"
CAP=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge $CAP ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt $CAP ] && return; done; }

eb () {   # design seed rho(none|0.8) sub
  local o=$RT/$4/N100_$1_s$2.json
  if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; return; fi
  gate; mkdir -p $RT/$4
  ( cd $EBD && env REGIMENS_CONFIRM=1 nohup nice -n 5 $PY -u ours/regimen/run_regimen_titrated.py 100 $1 $2 $3 $o \
      > $ROOT/$L/eb_$4_$1_s$2.log 2>&1 < /dev/null & )
  say "start eb $4 $1 s$2"; sleep 20
}
dh () {   # design seed sub [--titrate 0.8]
  local o=$RD/$3/Bfixkp20n/N100_$1_s$2.json d=$1 s=$2 sub=$3; shift 3
  if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; return; fi
  gate; mkdir -p $(dirname $o)
  env REGIMENS_CONFIRM=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s $o --iters 6000 --no-check $BFLAGS "$@" \
      > $L/dh_${sub}_${d}_s$s.log 2>&1 < /dev/null &
  say "start dh $sub $d s$s"; sleep 20
}
t2 () {   # arch(film|dc) seed
  local D=$R2/${1}_n100_s$2 sub
  [ $1 = film ] && sub=exp_film_run || sub=exp_dosecond_run
  if ls $D/$sub/$T/traj/*_ep006000.ckpt > /dev/null 2>&1 || pgrep -f -- "--save ./$D/" > /dev/null; then say "skip $D"; return; fi
  gate; mkdir -p $D/$sub/$T
  if [ $1 = film ]; then
    nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
      --noise-weight 0.01 --max-t 5. -b 512 --seed $2 --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
      --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
      --static-dim 4 --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __t1winmore_s$2 > $L/t2_film_s$2.log 2>&1 < /dev/null &
  else
    nohup nice -n 5 $PY train_dose_cond.py --experiment $T --data-dir ./results/exp_film_run --niters 6000 -b 512 \
      -l 10 --lr 1e-2 --seed $2 --patience 1000000 --obsrv-std 0.05 --static-dim 4 --decoder-hidden 50 \
      --decoder-residual --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __t1winmore_s$2 > $L/t2_dc_s$2.log 2>&1 < /dev/null &
  fi
  say "start t2 $1 s$2"; sleep 20
}

say "WEEKEND queue start"
for s in 5 6 7 8 9 10 11 12 13 14; do eb sparse $s 0.8 titr0.8; dh sparse $s titr0.8 --titrate 0.8; done
for s in 1 4 5 6 7 8 9 10; do t2 film $s; t2 dc $s; done
for s in 5 6 7 8 9 10 11 12 13 14; do eb sparse $s none ctrl; dh sparse $s ctrl; done
for s in 3 4 5 6 7; do eb dense $s none ctrl; dh dense $s ctrl; done
say "all training launched"

# ---- Table 2 scoring, once its training processes are gone ----
sleep 60
while pgrep -f -- "--save ./$R2/" > /dev/null; do sleep 120; done
mkdir -p $R2/eval $RC
for arch in film dc; do
  [ $arch = film ] && { H=test_film_matched.py; sub=exp_film_run; } || { H=test_dose_cond.py; sub=exp_dosecond_run; }
  for s in 1 4 5 6 7 8 9 10; do
    CK=$(ls $R2/${arch}_n100_s$s/$sub/$T/traj/*_ep006000.ckpt 2>/dev/null)
    [ "$(echo "$CK" | grep -c ckpt)" = 1 ] || { say "MISSING ckpt $arch s$s"; continue; }
    for reg in in lo hi; do
      ex=$BC; [ $reg != in ] && ex=${BC}_$reg
      o=$R2/eval/ep006000_${arch}_n100_s${s}_on_$reg.json; [ -f $o ] && continue
      gate
      nohup nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $ex --data-dir ./results/exp_film_run \
        --ckpt $CK --scale-from $T --eval-split test --label t1winmore_${arch}_s${s}_$reg --out-json $o \
        > $L/eval_${arch}_s${s}_$reg.log 2>&1 < /dev/null &
      sleep 5
    done
    o=$RC/${arch}_s${s}_all.json
    if [ ! -f $o ]; then
      gate
      nohup nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $T --data-dir ./results/exp_film_run \
        --ckpt $CK --scale-from $T --eval-split all --label t1winmore_${arch}_all_s$s --out-json $o > $L/recal_${arch}_s$s.log 2>&1 < /dev/null &
      sleep 5
    fi
  done
done
say "Table 2 scoring launched"

# ---- wait for everything, then the tables ----
sleep 60
while pgrep -f "res_titr/confirm/|regimen_dh/confirm/|s4_pnoise_win_more/|paper_win_more/" > /dev/null; do sleep 120; done
say "all runs finished: $(ls $RD/*/Bfixkp20n/*.json 2>/dev/null | wc -l | tr -d ' ') dose-event, $(ls $RT/*/*.json 2>/dev/null | wc -l | tr -d ' ') EB-VAE, $(ls $R2/eval/*.json 2>/dev/null | wc -l | tr -d ' ') Table-2 evals"
for f in scripts/weekend/confirm_table.py scripts/weekend/table2_more.py; do
  [ -f $f ] && { $PY $f > $L/$(basename $f .py).out 2>&1; say "ran $f (exit $?)"; }
done
say "WEEKEND_DONE"
