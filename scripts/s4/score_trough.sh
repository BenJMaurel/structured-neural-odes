#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Scoring + table of the trough-only latent ODEs (= the second half of chain_trough.sh), relaunched after the 02/10
# ~19:44 reboot killed that chain during scoring (training had finished: TROUGH_TRAINED 19:35). Skips existing outputs.
# Log logs/score_trough.log, marker TROUGH_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_trough; CAP=${CAP:-1100}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
gate () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 20; done; }
envs () { case $1 in trough_hidecyp) echo "PK_ENC_TIMES=0 PK_HIDE_CYP=1" ;; trough) echo "PK_ENC_TIMES=0" ;; esac; }
mkdir -p $R/eval; EV=""
for cfg in trough_hidecyp trough; do for s in 1 2 3; do
  CK=$(ls $R/${cfg}_n100_s$s/exp_film_run/$T/traj/*_ep006000.ckpt 2>/dev/null)
  [ "$(echo "$CK" | grep -c ckpt)" = 1 ] || { say "MISSING ckpt $cfg s$s"; continue; }
  for reg in in lo hi; do
    out=$R/eval/ep006000_${cfg}_n100_s${s}_on_$reg.json; [ -f $out ] && continue
    ex=$B; [ $reg != in ] && ex=${B}_$reg; gate
    env $(envs $cfg) nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment $ex \
      --data-dir ./results/exp_film_run --ckpt $CK --scale-from $T --eval-split test --label ${cfg}_s${s}_$reg \
      --out-json $out > logs/eval_${cfg}_s${s}_$reg.log 2>&1 &
    EV="$EV $!"; sleep 5
  done
done; done
[ -n "$EV" ] && wait $EV
say "scored: $(ls $R/eval | wc -l | tr -d ' ') eval files; tracebacks: $(grep -l Traceback logs/eval_trough*_s*_*.log 2>/dev/null | wc -l | tr -d ' ')"
$PY scripts/s4/trough_table.py > $R/trough_table.txt 2>&1; say "table -> $R/trough_table.txt"
say "TROUGH_DONE"
