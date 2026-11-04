#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# FALLBACK (a): the Table 2 dose-event row becomes dh7pgkp
# (jump b(p), encoder reads the samples AND the dose), which had 2 seeds -- its seed 3 was killed at ep5400 with the ablation.
# Retrain seed 3 with the command of chain_doseh_ablation.sh (c=kp, s=3), in a FRESH root (..._s3r) so the killed partial run in
# ..._s3 stays untouched. Reproducibility check: the rerun's ep300 checkpoint vs the killed run's (same seed, same code path, which
# 0.00zz74 verified unchanged) -> REPRO line in the log. Scoring (--scale-from the training cohort): s1 / s2 lo / hi right away, s3
# in / lo / hi after training -> results/s4_doseh/eval/ep006000_dh7pgkp_osplit1-5_n100_s*_on_*.json; then dosehkpn_summary.py.
# Gate <= 8 python3 jobs. Markers in logs/chain_dh7pgkp_s3.log: DH7PGKP3_TRAINED, DH7PGKP3_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_doseh
CAP=8
say () { echo "[$(date '+%F %T')] $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge $CAP ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt $CAP ] && return; done; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
D=$R/dh7pgkp_osplit1-5_n100_s3r
[ -e $D ] && { say "REFUSING: $D exists"; exit 1; }
gate
mkdir -p $D/exp_film_run/$T
nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
  --noise-weight 0.01 --max-t 5. -b 512 --seed 3 --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
  --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
  --static-dim 4 --orth-split 1,5 --dose-history 7 --dose-interval 24 --dose-interval-prograf 12 --kick-phys 32 \
  --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __dh7pgkp_osplit1-5_s3r \
  > logs/train_dh7pgkp_osplit1-5_s3r.log 2>&1 &
TP=$!; say "launched dh7pgkp seed 3 retrain (pid $TP)"
ev () {   # seed reg ckpt
  case $2 in in) ex=$B;; *) ex=${B}_$2;; esac
  gate
  nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex --data-dir ./results/exp_film_run \
    --ckpt $3 --scale-from $T --eval-split test --label dh7pgkp_osplit1-5_s$1_$2 \
    --out-json $R/eval/ep006000_dh7pgkp_osplit1-5_n100_s$1_on_$2.json > logs/eval_dh7pgkp_osplit1-5_s$1_$2.log 2>&1 &
  say "eval dh7pgkp s$1 $2 (pid $!)"; sleep 10
}
mkdir -p $R/eval
for s in 1 2; do
  C=$(ls $R/dh7pgkp_osplit1-5_n100_s$s/exp_film_run/$T/traj/*_ep006000.ckpt | head -1)
  for reg in lo hi; do [ -f $R/eval/ep006000_dh7pgkp_osplit1-5_n100_s${s}_on_$reg.json ] || ev $s $reg $C; done
done
ck () { ls $1/exp_film_run/$T/traj/*_ep$2.ckpt 2>/dev/null | head -1; }
while [ -z "$(ck $D 000300)" ] && kill -0 $TP 2>/dev/null; do sleep 30; done
sleep 20
$PY - "$(ck $R/dh7pgkp_osplit1-5_n100_s3 000300)" "$(ck $D 000300)" <<'EOF' 2>&1 | sed "s/^/[$(date '+%F %T')] /"
import sys, torch
a, b = (torch.load(p, map_location='cpu', weights_only=False)['state_dict'] for p in sys.argv[1:3])
same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
print(f'REPRO ep300 rerun vs killed run: state_dict identical = {same}' + ('' if same else
      f' (max |diff| {max(float((a[k].float() - b[k].float()).abs().max()) for k in a if k in b and a[k].shape == b[k].shape):.3e})'))
EOF
while [ -z "$(ck $D 006000)" ]; do
  kill -0 $TP 2>/dev/null || { sleep 30; [ -n "$(ck $D 006000)" ] || { say "TRAINING DIED dh7pgkp s3r"; exit 1; }; }
  sleep 60
done
say "DH7PGKP3_TRAINED"
for reg in in lo hi; do ev 3 $reg $(ck $D 006000); done
wait
$PY scripts/s4/dosehkpn_summary.py > /dev/null 2>&1; say "summary -> results/s4_doseh/dosehkpn_summary.txt"
say "DH7PGKP3_DONE"
