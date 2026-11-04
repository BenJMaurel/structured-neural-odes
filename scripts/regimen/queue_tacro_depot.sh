#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Tacrolimus regimen benchmark rerun with the depot (, after the 12-sample Fig. 6 showed negative
# concentrations at every dose; pilot ). Same flags as the current runs + --depot 1 (dh_regimen.py opt-in, backup .bak_depot):
#   scenario 2 (linear): Bfixkp3.1n flags -> results/regimen_dh/tacro_s2/Bdepot3.1n/   (sparse s0 / visit s0 copied from the pilot, same config)
#   scenario 4 (saturable): Bfixkp4n flags -> results/regimen_dh/tacro/Bdepot4n/, titrated (rho 0.8) sparse s0-2 -> tacro/titr0.8/Bdepot4n/
# sparse and visit s0-4, dense s0-2. Gate <= 8 python3 jobs. Marker TACRODEPOT_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 8 ]; do sleep 60; done; sleep 23; [ "$(njobs)" -lt 8 ] && return; done; }
run () {   # scenario design seed outdir mgl [extra...]
  local sc=$1 d=$2 s=$3 od=$4 mgl=$5; shift 5
  local o=$od/N100_${d}_s$s.json
  if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; return; fi
  gate; mkdir -p $od
  env TACRO_SCENARIO=$sc OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s $o --iters 6000 --no-check \
      --obsrv-mgl $mgl --kick-phys 32 --enc-dose norm --sim tacro --depot 1 "$@" > $L/tacro_depot_sc${sc}_${d}_s$s$( [ $# -gt 0 ] && echo _titr ).log 2>&1 < /dev/null &
  say "start sc$sc $d s$s $*"; sleep 5
}
for s in 0 1 2; do run 2 dense $s $R/tacro_s2/Bdepot3.1n 3.1; run 4 dense $s $R/tacro/Bdepot4n 4; done
for s in 0 1 2 3 4; do run 2 visit $s $R/tacro_s2/Bdepot3.1n 3.1; run 4 visit $s $R/tacro/Bdepot4n 4; done
for s in 0 1 2 3 4; do run 2 sparse $s $R/tacro_s2/Bdepot3.1n 3.1; run 4 sparse $s $R/tacro/Bdepot4n 4; done
for s in 0 1 2; do run 4 sparse $s $R/tacro/titr0.8/Bdepot4n 4 --titrate 0.8; done
say "all launched"; sleep 60
while pgrep -f "dh_regimen.py .*(Bdepot3.1n|Bdepot4n)" > /dev/null; do sleep 60; done
say "TACRODEPOT_DONE"
