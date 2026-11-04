#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Pilot ( after the 12-sample Fig. 6 showed negative concentrations at every dose): two opt-in
# variants of the dose-event model (dh_regimen.py, backup .bak_depot; default path bit-identical) on one seed per design, to choose
# the fix BEFORE any full rerun. --readout-relu: concentrations clipped at zero. --depot 1: each dose translates only one drug
# coordinate that the readout does not see (concentration continuous at a dose). Tacrolimus scenario 2, visit (12) and sparse (6) s0;
# design of Baaz et al., confirmation seed 5 (randomised, REGIMENS_CONFIRM=1). -> results/regimen_dh/pilot_depot/. Marker PILOTDEPOT_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh/pilot_depot; L=logs/regimen_dh; mkdir -p $L $R
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 8 ]; do sleep 60; done; sleep 23; [ "$(njobs)" -lt 8 ] && return; done; }
TB="--iters 6000 --no-check --obsrv-mgl 3.1 --kick-phys 32 --enc-dose norm --sim tacro"
BB="--iters 6000 --no-check --obsrv-mgl 20 --kick-phys 32 --enc-dose norm"
run () {   # out log env args...
  local o=$1 lg=$2 e=$3; shift 3
  if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; return; fi
  gate; local a=("$@"); env $e OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py ${a[0]} ${a[1]} ${a[2]} $o "${a[@]:3}" > $L/$lg.log 2>&1 < /dev/null &
  say "start $lg"; sleep 5
}
for v in relu depot; do
  [ $v = relu ] && opt="--readout-relu" || opt="--depot 1"
  run $R/tacro_s2_visit_${v}_s0.json pilot_tacro_visit_$v TACRO_SCENARIO=2 100 visit 0 $TB $opt
  run $R/tacro_s2_sparse_${v}_s0.json pilot_tacro_sparse_$v TACRO_SCENARIO=2 100 sparse 0 $TB $opt
  run $R/baaz_ctrl_sparse_${v}_s5.json pilot_baaz_sparse_$v REGIMENS_CONFIRM=1 100 sparse 5 $BB $opt
done
sleep 30; while pgrep -f "dh_regimen.py .*pilot_depot" > /dev/null; do sleep 30; done
say "PILOTDEPOT_DONE"
