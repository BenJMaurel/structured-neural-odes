#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# OPTION (b) : ONE encoder for both benchmarks = dose-normalised samples + the dose value through weights
# that start at zero (the initial encoder is exactly the dose-normalised one; checked 15/10, scratch check_encdoseval.py).
#   two-visit tacrolimus: dh7pgkpnd = dh7pgkpn (Fig. 3) + --encoder-dose-value; seeds 1-3, ep6000, scored in / lo / hi with
#     --scale-from the training cohort -> results/s4_doseh/eval/ep006000_dh7pgkpnd_*; summary scripts/s4/dosehkpn_summary.py
#   regimen (Baaz linear, the three cells of main-text Table 3): Bfixkp20nd = Bfixkp20n with --enc-dose normd: sparse s0-4, titrated
#     (rho 0.8) sparse s0-2, dense s0-2 -> results/regimen_dh/[titr0.8/]Bfixkp20nd/; table scripts/regimen/encdoseval_table.py
# Two-visit runs first (the long ones, ~4 h), then the regimen runs (~1 h). Gate: <= 8 python3 jobs on the machine (cap 800 %),
# re-checked after 41 s. Markers in logs/chain_encdoseval.log: ENCDOSEVAL_TRAINED, ENCDOSEVAL_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_doseh; RR=results/regimen_dh; L=logs/regimen_dh; mkdir -p $L
CAP=8
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge $CAP ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt $CAP ] && return; done; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
SEEDS="1 2 3"
for s in $SEEDS; do D=$R/dh7pgkpnd_osplit1-5_n100_s$s; [ -e $D ] && { say "REFUSING: $D exists"; exit 1; }; done
for s in $SEEDS; do
  D=$R/dh7pgkpnd_osplit1-5_n100_s$s
  gate
  mkdir -p $D/exp_film_run/$T
  nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --orth-split 1,5 --dose-history 7 --dose-interval 24 --dose-interval-prograf 12 \
    --kick-phys 32 --encoder-dose-norm --encoder-dose-value \
    --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __dh7pgkpnd_osplit1-5_s$s \
    > logs/train_dh7pgkpnd_osplit1-5_s$s.log 2>&1 &
  say "launched dh7pgkpnd seed $s (pid $!)"; sleep 30
done
BN="--obsrv-mgl 20 --kick-phys 32 --enc-dose normd"
J=()
for s in 0 1 2 3 4; do J+=("sparse $s $RR/Bfixkp20nd/N100_sparse_s$s.json Bnd_sparse_s$s"); done
for s in 0 1 2; do J+=("sparse $s $RR/titr0.8/Bfixkp20nd/N100_sparse_s$s.json Bnd_titr_sparse_s$s --titrate 0.8"); done
for s in 0 1 2; do J+=("dense $s $RR/Bfixkp20nd/N100_dense_s$s.json Bnd_dense_s$s"); done
for j in "${J[@]}"; do
  set -- $j; d=$1 s=$2 o=$3 lg=$4; shift 4
  if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
  gate; mkdir -p "$(dirname $o)"
  env OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s "$o" --iters 6000 $BN "$@" > "$L/$lg.log" 2>&1 < /dev/null &
  say "start $lg (pid $!)"; sleep 20
done
ck () { ls $R/dh7pgkpnd_osplit1-5_n100_s$1/exp_film_run/$T/traj/*_ep006000.ckpt 2>/dev/null | head -1; }
for s in $SEEDS; do
  while [ -z "$(ck $s)" ]; do
    pgrep -f "log-suffix __dh7pgkpnd_osplit1-5_s$s" >/dev/null || { sleep 30; [ -n "$(ck $s)" ] || { say "TRAINING DIED dh7pgkpnd s$s"; break; }; }
    sleep 60
  done
done
while pgrep -f "dh_regimen.py 100 .*Bfixkp20nd" > /dev/null; do sleep 60; done
say "ENCDOSEVAL_TRAINED"
mkdir -p $R/eval
for s in $SEEDS; do
  C=$(ck $s); [ -n "$C" ] || continue
  for reg in in lo hi; do
    case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
    gate
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
      --ckpt $C --scale-from $T --eval-split test --label dh7pgkpnd_osplit1-5_s${s}_$reg \
      --out-json $R/eval/ep006000_dh7pgkpnd_osplit1-5_n100_s${s}_on_$reg.json > logs/eval_dh7pgkpnd_osplit1-5_s${s}_$reg.log 2>&1 &
    sleep 10
  done
done
wait
$PY scripts/s4/dosehkpn_summary.py > /dev/null 2>&1; say "summary -> results/s4_doseh/dosehkpn_summary.txt"
$PY scripts/regimen/encdoseval_table.py > /dev/null 2>&1; say "table -> results/regimen_dh/encdoseval_table.txt"
say "ENCDOSEVAL_DONE"
