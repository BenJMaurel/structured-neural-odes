#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Part 2 cohorts (2026-10-03): realistic dose-switch design (gen_tacro_confound.py --switch-delay 4) at saturation k=3
# (stress test, otherwise Table 3's recipe: scenario 4, paper noise, Vc-confound rho 0 (unconfounded), dose
# grid 1-8 mg, visit-1 exposure window 3-25 ng/mL, seed 0 / dose-seed 1234, test fraction 0.904. Visit 1 = exact steady
# state on d1 (patients without one excluded and counted), switch to d2 24 h after the visit-1 dose, visit 2 = the
# interval starting 4 days later. In-range first (computes + caches the steady states), then _lo / _hi (same visit 1,
# --ood-keep-v1, reuse the cache), then the 100-patient training subset. Log logs/gen_switch.log, marker GEN_SWITCH_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; B=confound_vc00_s4k3_switch
COMMON="--num_patients 2800 --rho 0.0 --test-fraction 0.904 --seed 0 --dose-seed 1234 --confound-param Vc --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 --prop-sd 0.113 --add-sd 0.71 --v1-cavg-window 3,25 --sat-scale 3 --switch-delay 4"
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1
say "in-range cohort"; SWITCH_WORKERS=${W:-6} nice -n 5 $PY gen_tacro_confound.py --exp $B $COMMON > logs/gen_switch_in.log 2>&1 || { say "FAILED in"; exit 1; }
say "lo / hi cohorts (cached visit 1)"
SWITCH_WORKERS=3 nice -n 5 $PY gen_tacro_confound.py --exp ${B}_lo $COMMON --ood-v2-grid 0.25,0.5 --ood-keep-v1 > logs/gen_switch_lo.log 2>&1 &
SWITCH_WORKERS=3 nice -n 5 $PY gen_tacro_confound.py --exp ${B}_hi $COMMON --ood-v2-grid 10,12 --ood-keep-v1 > logs/gen_switch_hi.log 2>&1 &
wait
$PY scripts/nsweep/make_subsets.py $B "100" || { say "SUBSET FAILED"; exit 1; }
cp results/exp_film_run/$B/noise.json results/exp_film_run/${B}_n100/ 2>/dev/null
for f in logs/gen_switch_in.log logs/gen_switch_lo.log logs/gen_switch_hi.log; do grep -E "switch design|exposure window|train .* / test" $f; done
say "GEN_SWITCH_DONE"
