#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Pilot of the LINEAR depot: --depot 1 hurt the design of Baaz et al. (s5 unseen 20.0 vs 10.8) because the
# depot-to-central transfer is the flow network, fitted through two dose levels as a convex curve (from a drug-free state, AUC per mg
# relative to 400 mg: 0.00 at 100 mg, 2.21 at 800 mg). --depot-linear (dh_regimen.py opt-in, backup .bak_deplin, default path checked
# identical): the depot empties at a first-order rate ka(p) into the other drug coordinates and the flow does not read it.
# Same seeds and flags as queue_pilot_depot.sh, plus Baaz s6 (a second seed against the existing current-model run).
# -> results/regimen_dh/pilot_depot/*_deplin_*. Marker PILOTDEPLIN_DONE.
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
opt="--depot 1 --depot-linear"
run $R/baaz_ctrl_sparse_deplin_s5.json pilot_baaz_sparse_deplin_s5 REGIMENS_CONFIRM=1 100 sparse 5 $BB $opt
run $R/tacro_s2_visit_deplin_s0.json pilot_tacro_visit_deplin TACRO_SCENARIO=2 100 visit 0 $TB $opt
run $R/tacro_s2_sparse_deplin_s0.json pilot_tacro_sparse_deplin TACRO_SCENARIO=2 100 sparse 0 $TB $opt
run $R/baaz_ctrl_sparse_deplin_s6.json pilot_baaz_sparse_deplin_s6 REGIMENS_CONFIRM=1 100 sparse 6 $BB $opt
sleep 30; while pgrep -f "dh_regimen.py .*_deplin_s" > /dev/null; do sleep 30; done
say "PILOTDEPLIN_DONE"
