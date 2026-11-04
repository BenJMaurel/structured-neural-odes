#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Trough-only latent ODE (OT-FiLM, main config) on the windowed s4 benchmark, N=100"launch the
# trough-only latent-ODE"), to set against the trough-only oracle (results/s4/oracle_trough/oracle_trough_n1100.txt).
# Inputs via the opt-in environment switches of lib/read_tacro.py + lib/encoder_decoder.py (defaults untouched):
#   trough_hidecyp : PK_ENC_TIMES=0 PK_HIDE_CYP=1  (pre-dose trough only, CYP3A5 genotype unmeasured)
#   trough         : PK_ENC_TIMES=0                (pre-dose trough only, genotype given)
# Seeds 1-3, 6000 epochs, traj every 300; scoring at ep6000 on in / lo / hi with THE SAME switches, scaler pinned to the
# training cohort -> results/s4_trough/eval/ep006000_<cfg>_n100_s<seed>_on_<reg>.json. Table: scripts/s4/trough_table.py.
# Log logs/chain_trough.log, markers TROUGH_TRAINED, TROUGH_DONE. Machine freed by(02/10): CAP 1100 default.
cd $REPO_ROOT
PY=${PYTHON:-python3}; B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_trough
SEEDS="1 2 3"; NIT=6000; CAP=${CAP:-1100}; mkdir -p $R/eval logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
gate () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 30; done; }
envs () { case $1 in trough_hidecyp) echo "PK_ENC_TIMES=0 PK_HIDE_CYP=1" ;; trough) echo "PK_ENC_TIMES=0" ;; esac; }
TRAIN=""
for cfg in trough_hidecyp trough; do for s in $SEEDS; do
  D=$R/${cfg}_n100_s$s; mkdir -p $D/exp_film_run/$T; gate
  env $(envs $cfg) nohup nice -n 5 $PY run_models.py --niters $NIT -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
    --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
    --static-dim 4 --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __${cfg}_s$s > logs/train_${cfg}_s$s.log 2>&1 &
  TRAIN="$TRAIN $!"; say "launched $cfg seed $s (pid $!)"; sleep 20
done; done
wait $TRAIN; say "TROUGH_TRAINED"
EV=""
for cfg in trough_hidecyp trough; do for s in $SEEDS; do
  CK=$(ls $R/${cfg}_n100_s$s/exp_film_run/$T/traj/*_ep006000.ckpt 2>/dev/null)
  [ "$(echo "$CK" | grep -c ckpt)" = 1 ] || { say "MISSING ckpt $cfg s$s"; continue; }
  for reg in in lo hi; do ex=$B; [ $reg != in ] && ex=${B}_$reg; gate
    env $(envs $cfg) nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex \
      --data-dir ./results/exp_film_run --ckpt $CK --scale-from $T --eval-split test --label ${cfg}_s${s}_$reg \
      --out-json $R/eval/ep006000_${cfg}_n100_s${s}_on_$reg.json > logs/eval_${cfg}_s${s}_$reg.log 2>&1 &
    EV="$EV $!"
  done
done; done
wait $EV; say "scored: $(ls $R/eval | wc -l | tr -d ' ') eval files"
$PY scripts/s4/trough_table.py > $R/trough_table.txt 2>&1; say "table -> $R/trough_table.txt"
say "TROUGH_DONE"
