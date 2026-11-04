#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Score the ten gamma>0 OT-FiLM seeds on the 1100 in-range test patients (Table 2), same command as queue_review22.sh's final step
# (that step skips seeds whose json exists). One eval at a time, waiting while >= 8 single-thread jobs run. Marker GPOS_EVAL_DONE.
P=$REPO_ROOT; PY=${PYTHON:-python3}; G2=$P/results/s4_pnoise_win_gpos; T=confound_vc00_s4_pnoise_win_n100
L=$P/logs/review22; cd $P; mkdir -p $G2/eval
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && ($2 ~ /python3$/ || ($2 ~ /(Rscript|exec\/R)$/ && $0 ~ /scripts\/monolix/)) && $0 !~ /multiprocessing/' | wc -l; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for s in 1 2 3 4 5 6 7 8 9 10; do
  CK=$(ls $G2/film_n100_s$s/exp_film_run/$T/traj/*_ep006000.ckpt); o=$G2/eval/ep006000_film_n100_s${s}_on_in.json
  [ -f $o ] && continue
  while [ "$(njobs)" -ge 8 ]; do sleep 20; done
  nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py \
      --experiment confound_vc00_s4_pnoise_win --data-dir ./results/exp_film_run --ckpt $CK --scale-from $T --eval-split test \
      --label gpos_film_s${s}_in --out-json $o > $L/eval_film_gpos_s$s.log 2>&1 < /dev/null
  echo "$(date '+%H:%M') s$s $([ -f $o ] && echo ok || echo FAILED)"
done
echo GPOS_EVAL_DONE
