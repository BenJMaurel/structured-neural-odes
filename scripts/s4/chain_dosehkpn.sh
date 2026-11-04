#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Table 3 "Dose events" row re-run with the Table 4 dose-event settings): chain_dosehpg.sh
# (main FiLM config + --orth-split 1,5 --dose-history 7 --dose-interval 24 --dose-interval-prograf 12, fixed noise --obsrv-std
# 0.05 like every Table 3 network) + --kick-phys 32 (physiology-dependent jump) + --encoder-dose-norm (encoder reads samples / dose,
# no dose value). Seeds 1-3, roots results/s4_doseh/dh7pgkpn_osplit1-5_n100_s*. Scored at ep6000 on in / lo / hi with
# --scale-from the training cohort -> results/s4_doseh/eval/ep006000_dh7pgkpn_osplit1-5_n100_s<s>_on_<reg>.json.
# doseh_check.py is NOT run (it reads dh.kick(), which ignores the physiology-dependent jump).
# Gate: <= 6 python3 jobs on the machine. Markers in logs/chain_dosehkpn.log: DOSEHKPN_TRAINED, DOSEHKPN_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_doseh
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while [ "$(njobs)" -ge 6 ]; do sleep 60; done; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres_osplit1-5_dh7_taupg12_kp32_encdosenorm
SEEDS="1 2 3"
for s in $SEEDS; do D=$R/dh7pgkpn_osplit1-5_n100_s$s; [ -e $D ] && { say "REFUSING: $D exists"; exit 1; }; done
for s in $SEEDS; do
  D=$R/dh7pgkpn_osplit1-5_n100_s$s; mkdir -p $D/exp_film_run/$T
  gate
  nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --orth-split 1,5 --dose-history 7 --dose-interval 24 --dose-interval-prograf 12 --kick-phys 32 \
    --encoder-dose-norm --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __dh7pgkpn_osplit1-5_s$s \
    > logs/train_dh7pgkpn_osplit1-5_s$s.log 2>&1 &
  say "launched dh7pgkpn seed $s (pid $!)"; sleep 30
done
ck () { echo $R/dh7pgkpn_osplit1-5_n100_s$1/exp_film_run/$T/traj/experiment_film_${T}__${TAG}_ep006000.ckpt; }
for s in $SEEDS; do
  while [ ! -f "$(ck $s)" ]; do
    pgrep -f "log-suffix __dh7pgkpn_osplit1-5_s$s" >/dev/null || { sleep 30; [ -f "$(ck $s)" ] || { say "TRAINING DIED dh7pgkpn s$s"; break; }; }
    sleep 60
  done
done
say "DOSEHKPN_TRAINED"
mkdir -p $R/eval
for s in $SEEDS; do
  C=$(ck $s); [ -f "$C" ] || continue
  for reg in in lo hi; do
    case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
    gate
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
      --ckpt $C --scale-from $T --eval-split test --label dh7pgkpn_osplit1-5_s${s}_$reg --out-json $R/eval/ep006000_dh7pgkpn_osplit1-5_n100_s${s}_on_$reg.json \
      > logs/eval_dh7pgkpn_osplit1-5_s${s}_$reg.log 2>&1 &
    sleep 10
  done
  wait
  say "scored dh7pgkpn s$s"
done
say "DOSEHKPN_DONE"
