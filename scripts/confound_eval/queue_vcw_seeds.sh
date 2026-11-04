#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Claim 3 (Vc shortcut test), more seeds + an N=400 point. Training commands IDENTICAL to scripts/confound_eval/chain_vcw200_matched.sh (FiLM: run_models.py -n 200 ...
# --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual; dose-cond: train_dose_cond.py
# --obsrv-std 0.05 --decoder-hidden 50 --decoder-residual; 6000 epochs; Lu dropped: unstable at N=200,:
#   N=200: seeds 4-10 (seeds 1-3 exist) -> results/vcw200_matched/s<seed>/, cohorts confound_vc0{9,0}_win_n200
#   N=400: seeds 1-3                    -> results/vcw400_matched/s<seed>/, cohorts confound_vc0{9,0}_win_n400 (nested:
#          n200 c n400 c n800, same test sets, made by make_subsets.py seed 0)
# Each finished run is scored at the ENDPOINT checkpoints ep 5400/5700/6000 on the 4 test sets (own vcw09 / vcw00 splits,
# vcw09big / vcw00big), seeded, scaler pinned to the training cohort (as the chain). Summaries via km_matched_did.py with
# KM_SEEDS: results/vcw200_matched/did_summary_ep6000_10seeds.txt (seeds 1-10; the 3-seed file is left untouched),
# results/vcw400_matched/did_summary_ep6000.txt. Gate: <= 6 running python jobs (executable path, non-stopped).
# Log logs/queue_vcw_seeds.log; markers VCW200_10SEEDS_DONE, VCW400_DONE, VCW_SEEDS_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
# Cap: 6 jobs, or 10 until the epoch in QCAP_UNTIL (; set at relaunch). Load = running python jobs of every queue + 2 per Monolix fit.
cap () { if [ "$(date +%s)" -lt "${QCAP_UNTIL:-0}" ]; then echo 10; else echo 6; fi; }
njobs () { echo $(( $(ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && /run_models|train_dose_cond|seeded_eval|test_film_matched|test_dose_cond|train_lu_pk|run_scen_baaz|dh_regimen|run_tacro_baaz|track_null|lu_regimen|run_regimen|eval_saved|map_refine|eval_lode|mapbe|dh_multivisit|gen_tacro|build_monolix/' | wc -l) + 2 * $(pgrep -f fit_mv_pop.R | wc -l) )); }
free () { [ "$(njobs)" -lt "$(cap)" ]; }
lab () { case $1 in *vc09_win_n*) echo vcw09 ;; *vc00_win_n*) echo vcw00 ;; confound_vc09_big_win) echo vcw09big ;; confound_vc00_big_win) echo vcw00big ;; esac; }
EPS="005400 005700 006000"

# job list: "N arch seed cohort"
JOBS=()
for s in 4 5 6 7 8 9 10; do for a in film dc; do for k in 09 00; do JOBS+=("200 $a $s confound_vc${k}_win_n200"); done; done; done
for s in 1 2 3; do for a in film dc; do for k in 09 00; do JOBS+=("400 $a $s confound_vc${k}_win_n400"); done; done; done
root () { echo results/vcw$1_matched/s$2; }
ckpt () {  # N arch seed cohort ep
  local R=$(root $1 $3); case $2 in film) ls $R/exp_film_run/$4/traj/*_ep$5.ckpt 2>/dev/null ;; dc) ls $R/exp_dosecond_run/$4/traj/*_ep$5.ckpt 2>/dev/null ;; esac; }
running () {  # N arch seed cohort
  local R=$(root $1 $3); case $2 in film) pgrep -f "run_models.py .*--experiment $4 .*--save ./$R/" > /dev/null ;; dc) pgrep -f "train_dose_cond.py --experiment $4 .*--save ./$R/" > /dev/null ;; esac; }
launch () {  # N arch seed cohort
  local N=$1 a=$2 s=$3 c=$4 R=$(root $1 $3) L=logs/train_vcw$1_$2_$(lab $4)_s$3.log
  if [ $a = film ]; then
    mkdir -p $R/exp_film_run/$c
    nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro \
      --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $c \
      --film-no-z0-cond --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual \
      --select-on mse_v2 --patience 1000000 --save ./$R/ --eval-every 300 --ckpt-every 300 \
      --log-suffix __vcw${N}_s$s > $L 2>&1 < /dev/null &
  else
    mkdir -p $R/exp_dosecond_run/$c
    nohup nice -n 5 $PY train_dose_cond.py --experiment $c --data-dir ./results/exp_film_run \
      --niters 6000 -b 512 -l 10 --lr 1e-2 --seed $s --patience 1000000 --obsrv-std 0.05 \
      --decoder-hidden 50 --decoder-residual --save ./$R/ --eval-every 300 --ckpt-every 300 \
      --log-suffix __vcw${N}_s$s > $L 2>&1 < /dev/null &
  fi
  say "launched N$N $a $(lab $c) s$s (pid $!, running python jobs now $(njobs))"
}
pending_evals () {  # N arch seed cohort -> prints "CK ev out" lines for missing evaluations of a finished run
  local N=$1 a=$2 s=$3 tr=$4 E=results/vcw$1_matched/eval
  for ep in $EPS; do
    local CK=$(ckpt $N $a $s $tr $ep); [ "$(echo "$CK" | grep -c ckpt)" = 1 ] || continue
    for ev in confound_vc09_win_n$N confound_vc00_win_n$N confound_vc09_big_win confound_vc00_big_win; do
      local out=$E/ep${ep}_${a}_s${s}_$(lab $tr)_on_$(lab $ev).json
      [ -f $out ] || pgrep -f "out-json $out" > /dev/null || echo "$CK $ev $out $ep"
    done
  done
}
summary () {  # N seeds out
  KM_PFX=vcw KM_EVAL=results/vcw$1_matched/eval KM_EPS="$EPS" KM_SEEDS="$2" $PY scripts/confound_eval/km_matched_did.py > $3 2>&1
}
mkdir -p results/vcw200_matched/eval results/vcw400_matched/eval
say "queue_vcw_seeds starts: ${#JOBS[@]} trainings"
next=0; done200=0
while true; do
  # 1. score finished runs first (short jobs; ep6000 checkpoint present, process gone); a failed evaluation is NOT retried
  todo=0
  for j in "${JOBS[@]}"; do
    set -- $j
    [ -n "$(ckpt $1 $2 $3 $4 006000)" ] && ! running $1 $2 $3 $4 || continue
    while read -r CK ev out ep; do
      [ -n "$CK" ] || continue
      EL=logs/eval_vcw$1_$2_s$3_$(lab $4)_on_$(lab $ev)_ep$ep.log
      grep -q Traceback $EL 2>/dev/null && continue
      todo=$((todo + 1)); free || continue
      case $2 in film) H=test_film_matched.py ;; dc) H=test_dose_cond.py ;; esac
      nohup nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $ev \
        --data-dir ./results/exp_film_run --ckpt $CK --scale-from $4 --eval-split test \
        --label vcw$1_$2_s$3_$(lab $4)_on_$(lab $ev)_ep$ep --out-json $out > $EL 2>&1 < /dev/null &
      sleep 2
    done < <(pending_evals $1 $2 $3 $4)
  done
  # 2. fill the remaining slots with trainings
  while [ $next -lt ${#JOBS[@]} ] && free; do
    set -- ${JOBS[$next]}
    if [ -n "$(ckpt $1 $2 $3 $4 006000)" ] || running $1 $2 $3 $4; then say "skip N$1 $2 $(lab $4) s$3 (done or running)"; else launch $1 $2 $3 $4; fi
    next=$((next + 1)); sleep 20
  done
  # 3. milestones
  n200=$(ls results/vcw200_matched/eval/ep00{5400,5700,6000}_*_s{4,5,6,7,8,9,10}_*.json 2>/dev/null | wc -l | tr -d ' ')
  if [ $done200 = 0 ] && [ "$n200" -ge 336 ]; then   # 7 seeds x 2 archs x 2 cohorts x 4 test sets x 3 ckpts
    sleep 30; summary 200 "1 2 3 4 5 6 7 8 9 10" results/vcw200_matched/did_summary_ep6000_10seeds.txt
    say "N=200 seeds 1-10 scored -> results/vcw200_matched/did_summary_ep6000_10seeds.txt; VCW200_10SEEDS_DONE"; done200=1
  fi
  n400=$(ls results/vcw400_matched/eval/ep00{5400,5700,6000}_*.json 2>/dev/null | wc -l | tr -d ' ')
  if [ $next -ge ${#JOBS[@]} ] && [ "$n400" -ge 144 ] && [ $done200 = 1 ] && ! pgrep -f "seeded_eval.py" > /dev/null; then
    summary 400 "1 2 3" results/vcw400_matched/did_summary_ep6000.txt
    say "N=400 scored -> results/vcw400_matched/did_summary_ep6000.txt; VCW400_DONE"
    say "eval failures: $(grep -l Traceback logs/eval_vcw200_*_s{4,5,6,7,8,9,10}_* logs/eval_vcw400_* 2>/dev/null | wc -l | tr -d ' ')"
    say "VCW_SEEDS_DONE"; break
  fi
  # 4. stall guard: everything launched, nothing running, evaluations still missing -> report and stop
  if [ $next -ge ${#JOBS[@]} ] && [ "$(njobs)" = 0 ] && [ $todo = 0 ] && [ $done200 = 0 -o "$n400" -lt 144 ]; then
    sleep 120
    if [ "$(njobs)" = 0 ]; then say "STALLED: n200 evals $n200/336, n400 evals $n400/144 -- check training logs"; break; fi
  fi
  sleep 60
done
