#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Fig. 6 on the job-1 design (review 22/10): 12 samples around the first dose, dose-event model with the softplus readout (Bv1sp20),
# Lu regimen model with the factual encoder, other models clipped at zero; training seed 0 and the figure's fixed patient rule.
# Waits for scripts/review22/clip0_visit1.sh (same slot), then gates at 8 jobs (2-s poll).
cd $REPO_ROOT; PY=${PYTHON:-python3}
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && ($2 ~ /python3$/ || ($2 ~ /(Rscript|exec\/R)$/ && $0 ~ /scripts\/monolix/)) && $0 !~ /multiprocessing/' | wc -l; }
[ -n "${SKIP_WAIT:-}" ] || until grep -q CLIP0_VISIT1_DONE logs/review22/clip0_visit1.out 2>/dev/null; do sleep 2; done
while [ "$(njobs)" -ge 8 ]; do sleep 2; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
FIG_CLIP0=1 FIG_LUIS=factual FIG_OURS=Bv1sp20 nice -n 5 $PY scripts/ndm/fig_tacro_patients_data.py visit1 0 > logs/review22/fig6_visit1_data.log 2>&1 || { echo FAILED data; exit 1; }
$PY scripts/ndm/make_fig_tacro_patients.py visit1 Bv1sp20 clip0 luisfact > logs/review22/fig6_visit1_make.log 2>&1 || { echo FAILED make; exit 1; }
echo FIG6_DONE
