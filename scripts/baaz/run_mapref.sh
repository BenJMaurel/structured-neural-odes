#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Amortisation-gap check: map_refine.py on the published EB-VAE ep300 models, scenario 3 seeds 1-10, training
# and test patients; <= 6 jobs at once (job-count gate, python executables only). Log logs/baaz/mapref.log, marker MAPREF_DONE.
R=$REPO_ROOT; EB=$REPO_ROOT/third_party/baaz_ebvae; PY=${PYTHON:-python3}; D=$R/results/baaz_scen3/data
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && /run_scen_baaz|dh_regimen|run_models|run_tacro_baaz|track_null|lu_regimen|run_regimen|eval_saved|mapbe|map_refine|eval_lode/' | wc -l | tr -d ' '; }
for s in $(seq 1 10); do for w in train test; do
  O=$R/results/baaz_check/mapref/scen3_s${s}_ep300_on_${w}.json; [ -f $O ] && continue
  sleep 3; while [ "$(njobs)" -ge 6 ]; do sleep 15; done
  ( cd $EB && nohup nice -n 5 $PY ours/tacro/map_refine.py $R/results/baaz_scen3/runs/models_seed${s}_ep300.pt $D/train_s$s.csv $D/test_s$s.csv \
      $D/${w}_s$s.csv $O >> $R/logs/baaz/mapref.log 2>&1 < /dev/null & )
done; done
sleep 5; while pgrep -f map_refine.py > /dev/null; do sleep 15; done
echo "[$(date '+%F %T')] MAPREF_DONE ($(ls $R/results/baaz_check/mapref/*.json | wc -l | tr -d ' ') files)" >> $R/logs/baaz/mapref.log
