#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Our plain latent ODE's train / test gap on scenario 3: eval_lode.py with EVAL_SPLIT=train (and the default
# test split, to pair them) on the final and _best checkpoints of the 10 rerun cohorts, run inside each run's slot worktree
# (the paper-era loader reads ./results/<ID>/virtual_cohort_*.csv). <= 6 jobs (shared with map_refine). Marker LODESPLIT_DONE.
R=$REPO_ROOT; PY=${PYTHON:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && /run_scen_baaz|dh_regimen|run_models|run_tacro_baaz|track_null|lu_regimen|run_regimen|eval_saved|mapbe|map_refine|eval_lode/' | wc -l | tr -d ' '; }
for s in $(seq -w 1 10); do
  ID=300$s; W=$(dirname $(dirname $(ls -d $R/results/paper_repro/w*/results/$ID | head -1)))
  for c in final best; do
    f=results/$ID/experiment_$ID.ckpt; [ $c = best ] && f=results/$ID/experiment_${ID}_best.ckpt
    for sp in train test; do
      O=$R/results/baaz_check/lode_split/s3_seed0${s}_${c}_${sp}.json; [ -f $O ] && continue
      sleep 3; while [ "$(njobs)" -ge 6 ]; do sleep 15; done
      ( cd $W && EVAL_SPLIT=$sp nohup nice -n 5 $PY -W ignore $R/scripts/repro/eval_lode.py $f $O > $O.log 2>&1 < /dev/null & )
    done
  done
done
sleep 5; while pgrep -f eval_lode.py > /dev/null; do sleep 15; done
echo "[$(date '+%F %T')] LODESPLIT_DONE ($(ls $R/results/baaz_check/lode_split/*.json | wc -l | tr -d ' ') files)" >> $R/logs/baaz/lode_split.log
