#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 3-scenario MAP-BE with Monolix AUTO-INIT + combined1 (form-fixed) residual error: 300 runs,
# same datasets as Table 1 / the default-init redo, so all comparisons are paired. Per run scripts/scen_rerun/run_autoinit.sh
# (fit + MAP-BE, ~60 min at 1 thread: auto-init itself ~32 min on s3 seed 1, SAEM ~22 min). NSLOT workers, each with its
# own worktree results/paper_repro/wNN (the R script writes fixed-name files in cwd), cap-gated, restartable (DONE_autoinit),
# SCENS restricts the scenarios (default "1 2 3"; 01/10 launched with "1 3"). Order seed-major (seed 1 s1 s2 s3, seed 2 ...) so partial results are balanced. Interim summary any time:
#   ${PYTHON:-python3} scripts/scen_rerun/summarise.py   (rows "MAP-BE, auto init, FORM FIXED")
# Final summary -> results/scen_rerun/summary_autoinit.txt. Marker SCEN_AUTOINIT_DONE in logs/chain_scen_autoinit.log.
cd $REPO_ROOT
CAP=${CAP:-1000}; NSLOT=${NSLOT:-10}; SEEDS=${SEEDS:-$(seq 1 100)}; SCENS=${SCENS:-1 2 3}; PY=${PYTHON:-python3}
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; }
JOBS=(); for s in $SEEDS; do for sc in $SCENS; do JOBS+=("$sc $s"); done; done
say "${#JOBS[@]} jobs over $NSLOT slots (scenarios $SCENS; AUTOINIT_IDS=${AUTOINIT_IDS:-all})"
worker () {
  local k=$1 w=results/paper_repro/w$(printf %02d $(( $1 + 1 ))) i
  for (( i = k; i < ${#JOBS[@]}; i += NSLOT )); do
    set -- ${JOBS[$i]}
    [ -f results/scen_rerun/runs/s$1_seed$(printf %03d $2)/DONE_autoinit ] && continue
    throttle
    bash scripts/scen_rerun/run_autoinit.sh $w $1 $2
  done
}
for (( k = 0; k < NSLOT; k++ )); do worker $k & sleep 60; done
wait
say "finished: $(ls results/scen_rerun/runs/*/DONE_autoinit 2>/dev/null | wc -l | tr -d ' ') DONE_autoinit, $(ls results/scen_rerun/runs/*/FAILED_autoinit_* 2>/dev/null | wc -l | tr -d ' ') FAILED"
$PY scripts/scen_rerun/summarise.py > results/scen_rerun/summary_autoinit.txt 2>&1
say "summary -> results/scen_rerun/summary_autoinit.txt"
say "SCEN_AUTOINIT_DONE"
