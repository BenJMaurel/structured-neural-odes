#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# RERUN of chain_dosehkpn.sh on NEW seeds 4-6 (). IDENTICAL configuration (copied, only SEEDS / log / markers changed):
# dose-event model of Fig. 3 = --kick-phys 32 (b(p)) + --encoder-dose-norm, main FiLM config, --orth-split 1,5 --dose-history 7
# --dose-interval 24 --dose-interval-prograf 12, --obsrv-std 0.05, 6000 epochs. Model code unchanged since the seed 1-3 runs
# (lib/ and run_models.py last edited 12/10 13:58, runs launched from 14:04), so seeds 1-3 would reproduce identically; new seeds
# test whether 21.7 / 24.6 is systematic. Roots results/s4_doseh/dh7pgkpn_osplit1-5_n100_s{4,5,6} (refuses if they exist).
# Gate <= 6 python3 jobs (double-checked, as the other queues). Markers in logs/chain_dosehkpn_rerun.log: DOSEHKPNR_TRAINED,
# DOSEHKPNR_DONE. Summary: scripts/s4/dosehkpn_summary.py -> results/s4_doseh/dosehkpn_summary.txt.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_doseh
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TAG=noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres_osplit1-5_dh7_taupg12_kp32_encdosenorm
SEEDS="4 5 6"
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
say "DOSEHKPNR_TRAINED"
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
$PY scripts/s4/dosehkpn_summary.py > /dev/null 2>&1; say "summary -> results/s4_doseh/dosehkpn_summary.txt"
say "DOSEHKPNR_DONE"
