#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Dose-cond at the INTENSIVE-PK-VISIT design (12 samples).
# Waits for DCPHASE2_DONE (logs/queue_dc_phase2.log), reads results/regimen_dh/dc_tune/selected.json, then tacrolimus scenario 4
# (--obsrv-mgl 4) and scenario 2 (TACRO_SCENARIO=2, --obsrv-mgl 3.1), visit s0-4, --dose-cond --enc-dose norm + the selected config.
# If the DEFAULT config was selected (phase 2 then reruns nothing), this queue ALSO runs the scenario-2 sparse s0-4 / dense s0-2 rows,
# which have never been run for dose-cond. Outputs: tacro[_s2]/dc_tuned_<cfg>/ (or tacro[_s2]/dosecond_n/ for default).
# A 3-iteration smoke run (visit + dose-cond, not yet combined before) aborts the queue if it fails. Gate <= 6 python3 jobs.
# Log logs/queue_dc_visit.log, marker DCVISIT_DONE.
P=$REPO_ROOT; PY=${PYTHON:-python3}; L=$P/logs/regimen_dh; LOG=$P/logs/queue_dc_visit.log
R=$P/results/regimen_dh; S=$P/scripts/regimen/dh_regimen.py
cd $P; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
say "waiting for DCPHASE2_DONE"
[ -n "${SKIP_WAIT:-}" ] || until grep -q DCPHASE2_DONE $P/logs/queue_dc_phase2.log 2>/dev/null; do
  grep -q ABORT $P/logs/queue_dc_phase2.log 2>/dev/null && { say "ABORT: phase 2 aborted"; exit 1; }; sleep 120; done
C=$($PY -c "import json; print(json.load(open('$R/dc_tune/selected.json'))['selected'])" 2>/dev/null)
case $C in z6) IT=6000; X="--dc-zero-init"; SUB=dc_tuned_z6;; z12) IT=12000; X="--dc-zero-init"; SUB=dc_tuned_z12;;
           d12) IT=12000; X=""; SUB=dc_tuned_d12;; default) IT=6000; X=""; SUB=dosecond_n;; *) say "ABORT: selected '$C'"; exit 1;; esac
say "selected: $C -> --iters $IT $X, outputs */$SUB/"
gate; mkdir -p $R/dc_tune/check
TACRO_SCENARIO=2 $PY $S 100 visit 0 $R/dc_tune/check/visit_smoke.json --iters 3 --sim tacro --obsrv-mgl 3.1 --dose-cond --enc-dose norm $X \
    > $R/dc_tune/check/visit_smoke.log 2>&1 || { say "ABORT: visit + dose-cond smoke run failed (see dc_tune/check/visit_smoke.log)"; exit 1; }
say "smoke run ok"
run () {   # scenario design seed
  local sc=$1 d=$2 s=$3 sub=tacro; local nz=4; [ $sc = 2 ] && { sub=tacro_s2; nz=3.1; }
  local o=$R/$sub/$SUB/N100_${d}_s$s.json; [ -e $o ] && return; mkdir -p $(dirname $o); gate
  env TACRO_SCENARIO=$sc OMP_NUM_THREADS=1 nohup nice -n 5 $PY $S 100 $d $s $o --iters $IT --sim tacro --obsrv-mgl $nz --dose-cond \
      --enc-dose norm $X > $L/dcvisit_${C}_tacro${sc}_${d}_s$s.log 2>&1 < /dev/null &
  say "start scenario $sc $d s$s"; sleep 20
}
for s in 0 1 2 3 4; do run 4 visit $s; run 2 visit $s; done
if [ $C = default ]; then for s in 0 1 2 3 4; do run 2 sparse $s; done; for s in 0 1 2; do run 2 dense $s; done; fi
sleep 30; while pgrep -f "$SUB/N100_" > /dev/null; do sleep 60; done
$PY scripts/regimen/tacro_design_table.py > $R/tacro/design_table.txt 2> $R/tacro/design_table.err
say "DCVISIT_DONE"
