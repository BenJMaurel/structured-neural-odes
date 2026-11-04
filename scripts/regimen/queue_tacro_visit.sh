#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Intensive-PK-visit design (12 samples: troughs before doses 2-4 + the development cohort's profile after the last dose; project notes
# 0.00zz50). Dose-event model (Table 4
# architecture; noise 4 ng/mL in scenario 4, 3.1 in scenario 2, as the other designs) and EB-VAE (curves dumped), scenarios 4 and 2,
# seeds 0-4, interleaved per seed. Absolute log paths (the 0.00zz47 bug). Gate <= 6 python3 jobs (re-check 41 s). Marker TACROVISIT_DONE.
P=$REPO_ROOT; EB=$REPO_ROOT/third_party/baaz_ebvae; PY=${PYTHON:-python3}; L=$P/logs/regimen_dh
cd $P; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
busy () { [ -f "$1" ] || pgrep -f -- "$1" > /dev/null; }
for s in 0 1 2 3 4; do
  for sc in 4 2; do
    if [ $sc = 4 ]; then D=$P/results/regimen_dh/tacro/Bfixkp4n; NZ=4; ED=$EB/ours/regimen/res_tacro/ctrl; else D=$P/results/regimen_dh/tacro_s2/Bfixkp3.1n; NZ=3.1; ED=$EB/ours/regimen/res_tacro_s2/ctrl; fi
    mkdir -p $D $ED
    o=$D/N100_visit_s$s.json
    if busy $o; then say "skip $o"; else gate
      env TACRO_SCENARIO=$sc OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 visit $s $o --iters 6000 --sim tacro \
          --obsrv-mgl $NZ --kick-phys 32 --enc-dose norm > $L/tacro_sc${sc}_Bn_visit_s$s.log 2>&1 < /dev/null &
      say "start ours scenario $sc visit s$s"; sleep 20; fi
    o=$ED/N100_visit_s$s.json
    if busy $o; then say "skip $o"; else gate
      ( cd $EB && env TACRO_SCENARIO=$sc OMP_NUM_THREADS=1 DUMP_CURVES=1 nohup nice -n 5 $PY -u ours/regimen/run_regimen_tacro.py 100 visit $s none $o \
          > $L/tacro_sc${sc}_ebvae_visit_s$s.log 2>&1 < /dev/null & )
      say "start EB-VAE scenario $sc visit s$s"; sleep 20; fi
  done
done
sleep 30; while pgrep -f "N100_visit_s" > /dev/null; do sleep 60; done
say "TACROVISIT_DONE"
