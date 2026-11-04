#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# -2LL of the published-init, default-init and auto-init Monolix fits on the same datasets (2026-10-02): is the
# degenerate solution the better fit of the (misspecified) model, or a worse local optimum? Unselected datasets:
# scenario 3 seeds 1-20, scenario 1 seeds 1-10 (override SET="sc:seeds ..."). NSLOT workers, cap-gated, skip done.
# Output results/scen_rerun/loglik/<ID>_<fit>.json; table scripts/scen_rerun/loglik_table.py. Marker LOGLIK_DONE.
cd $REPO_ROOT
CAP=${CAP:-1000}; NSLOT=${NSLOT:-6}; SET=${SET:-"3:$(seq -s, 1 20) 1:$(seq -s, 1 10)"}
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 30; done; }
say () { echo "[$(date '+%F %T')] $*"; }
JOBS=(); for g in $SET; do sc=${g%%:*}; for s in ${g#*:}; do :; done; for s in $(echo ${g#*:} | tr , ' '); do for f in published definit autoinit; do JOBS+=("$sc $s $f"); done; done; done
say "${#JOBS[@]} likelihood jobs over $NSLOT slots"
CL=results/scen_rerun/loglik/claims; mkdir -p $CL
worker () { local j; for j in "${JOBS[@]}"; do set -- $j
  mkdir $CL/$1_$2_$3 2>/dev/null || continue; throttle; bash scripts/scen_rerun/run_loglik.sh $1 $2 $3; done; }
for (( k = 0; k < NSLOT; k++ )); do worker & sleep 20; done; wait
say "$(ls results/scen_rerun/loglik/*.json 2>/dev/null | wc -l | tr -d ' ') json"; say "LOGLIK_DONE"
