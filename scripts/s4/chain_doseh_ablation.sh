#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# ABLATION of the Fig. 3 dose-event model on the two-visit tacrolimus task): which of its two
# changes makes it worse than the Table 2 row (dh7pg 12.7 / 15.5 vs dh7pgkpn 21.7 / 24.6, three seeds)?
#   dh7pgkp  = dh7pg + --kick-phys 32        (physiology-dependent jump b(p); encoder still reads the samples AND the dose)
#   dh7pgen  = dh7pg + --encoder-dose-norm   (encoder reads samples / dose, no dose value; population jump)
# Base command = chain_dosehpg.sh (identical flags; chain_dosehkpn.sh = this base + both flags). Seeds 1-3, paired with the existing
# dh7pg and dh7pgkpn seeds. Roots results/s4_doseh/dh7pg{kp,en}_osplit1-5_n100_s* (refuses if they exist). Scored at ep6000 on
# in / lo / hi with --scale-from the training cohort; summary scripts/s4/dosehkpn_summary.py -> results/s4_doseh/dosehkpn_summary.txt.
# CPU: cap 800 %. The queues already running (queue_dc_phase2 / dc_visit / eb_priortest) gate at 6 jobs and cannot be
# edited while they run, so this chain takes the 2 extra slots while any of them is alive (<= 2 of its own jobs), and up to 8 jobs in
# total once they have all finished. Markers in logs/chain_doseh_ablation.log: DOSEHABL_TRAINED, DOSEHABL_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_doseh
CAP=8
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
others () { pgrep -f "queue_dc_phase2.sh|queue_dc_visit.sh|queue_eb_priortest.sh" >/dev/null; }
mine () { ps -A -o args= | grep -E "python3 .*(log-suffix __dh7pg(kp|en)_|label dh7pg(kp|en)_)" | grep -v grep | wc -l; }
blocked () { [ "$(njobs)" -ge $CAP ] && return 0; others && [ "$(mine)" -ge 2 ] && return 0; return 1; }
gate () { while true; do while blocked; do sleep 60; done; sleep 41; blocked || return; done; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
SEEDS="1 2 3"; CONFIGS="kp en"
flags () { case $1 in kp) echo "--kick-phys 32";; en) echo "--encoder-dose-norm";; esac; }
for c in $CONFIGS; do for s in $SEEDS; do D=$R/dh7pg${c}_osplit1-5_n100_s$s; [ -e $D ] && { say "REFUSING: $D exists"; exit 1; }; done; done
for s in $SEEDS; do for c in $CONFIGS; do   # interleaved: the two arms of a seed run side by side
  D=$R/dh7pg${c}_osplit1-5_n100_s$s
  gate
  mkdir -p $D/exp_film_run/$T
  nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --orth-split 1,5 --dose-history 7 --dose-interval 24 --dose-interval-prograf 12 $(flags $c) \
    --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __dh7pg${c}_osplit1-5_s$s \
    > logs/train_dh7pg${c}_osplit1-5_s$s.log 2>&1 &
  say "launched dh7pg$c seed $s ($(flags $c); pid $!)"; sleep 30
done; done
ck () { ls $R/dh7pg$1_osplit1-5_n100_s$2/exp_film_run/$T/traj/*_ep006000.ckpt 2>/dev/null | head -1; }
for c in $CONFIGS; do for s in $SEEDS; do
  while [ -z "$(ck $c $s)" ]; do
    pgrep -f "log-suffix __dh7pg${c}_osplit1-5_s$s" >/dev/null || { sleep 30; [ -n "$(ck $c $s)" ] || { say "TRAINING DIED dh7pg$c s$s"; break; }; }
    sleep 60
  done
done; done
say "DOSEHABL_TRAINED"
mkdir -p $R/eval
for s in $SEEDS; do for c in $CONFIGS; do
  C=$(ck $c $s); [ -n "$C" ] || continue
  for reg in in lo hi; do
    case $reg in in) ex=$B;; *) ex=${B}_$reg;; esac
    gate
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
      --ckpt $C --scale-from $T --eval-split test --label dh7pg${c}_osplit1-5_s${s}_$reg \
      --out-json $R/eval/ep006000_dh7pg${c}_osplit1-5_n100_s${s}_on_$reg.json > logs/eval_dh7pg${c}_osplit1-5_s${s}_$reg.log 2>&1 &
    sleep 10
  done
done; done
wait
$PY scripts/s4/dosehkpn_summary.py > /dev/null 2>&1; say "summary -> results/s4_doseh/dosehkpn_summary.txt"
say "DOSEHABL_DONE"
