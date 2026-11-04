#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Dose-cond PHASE 2. Waits for DCTUNE_DONE
# (logs/queue_dc_tune.log), reads results/regimen_dh/dc_tune/selected.json (rule: lowest training-regimen error, see
# dc_tune_table.py). If the default config is selected nothing is rerun (the existing runs stand). Otherwise the selected config on:
# Baaz linear dense s0-2 and titrated (rho 0.8) sparse s0-2 (--obsrv-mgl 20); tacrolimus scenario 4 (--obsrv-mgl 4) and scenario 2
# (TACRO_SCENARIO=2, --obsrv-mgl 3.1) sparse s0-4 + dense s0-2 = 22 runs; all --dose-cond --enc-dose norm (the dose-event model's
# encoder and noise). Baaz sparse s0-4 = the tuning runs themselves. Gate <= 6 python3 jobs. Table scripts/regimen/dc_phase2_table.py
# -> results/regimen_dh/dc_tuned/table.txt. Log logs/queue_dc_phase2.log, marker DCPHASE2_DONE.
P=$REPO_ROOT; PY=${PYTHON:-python3}; L=$P/logs/regimen_dh; LOG=$P/logs/queue_dc_phase2.log
R=$P/results/regimen_dh; S=$P/scripts/regimen/dh_regimen.py
cd $P; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
say "waiting for DCTUNE_DONE"
[ -n "${SKIP_WAIT:-}" ] || until grep -q DCTUNE_DONE $P/logs/queue_dc_tune.log 2>/dev/null; do
  grep -q ABORT $P/logs/queue_dc_tune.log 2>/dev/null && { say "ABORT: tuning queue aborted"; exit 1; }; sleep 120; done
C=$($PY -c "import json; print(json.load(open('$R/dc_tune/selected.json'))['selected'])" 2>/dev/null)
[ -n "$C" ] || { say "ABORT: no selected.json"; exit 1; }
if [ "$C" = default ]; then say "selected: default -> phase 2 skipped (existing dose-cond runs stand)"; say "DCPHASE2_DONE"; exit 0; fi
case $C in z6) IT=6000; X="--dc-zero-init";; z12) IT=12000; X="--dc-zero-init";; d12) IT=12000; X="";; *) say "ABORT: unknown $C"; exit 1;; esac
say "selected: $C (--iters $IT $X)"
run () {   # env out log args...
  local e=$1 o=$2 lg=$3; shift 3
  [ -e $o ] && return; mkdir -p $(dirname $o); gate
  env $e OMP_NUM_THREADS=1 nohup nice -n 5 $PY $S "$@" $o --iters $IT --dose-cond --enc-dose norm $X $EXTRA > $L/$lg.log 2>&1 < /dev/null &
  say "start $lg"; sleep 20
}
T4=$R/tacro/dc_tuned_$C; T2=$R/tacro_s2/dc_tuned_$C; B=$R/dc_tuned/$C
for s in 0 1 2 3 4; do   # tacrolimus sparse first (the longest), scenarios interleaved
  EXTRA="--sim tacro --obsrv-mgl 4";   run TACRO_SCENARIO=4 $T4/N100_sparse_s$s.json dcp2_${C}_tacro4_sparse_s$s 100 sparse $s
  EXTRA="--sim tacro --obsrv-mgl 3.1"; run TACRO_SCENARIO=2 $T2/N100_sparse_s$s.json dcp2_${C}_tacro2_sparse_s$s 100 sparse $s
done
for s in 0 1 2; do
  EXTRA="--sim tacro --obsrv-mgl 4";   run TACRO_SCENARIO=4 $T4/N100_dense_s$s.json dcp2_${C}_tacro4_dense_s$s 100 dense $s
  EXTRA="--sim tacro --obsrv-mgl 3.1"; run TACRO_SCENARIO=2 $T2/N100_dense_s$s.json dcp2_${C}_tacro2_dense_s$s 100 dense $s
  EXTRA="--obsrv-mgl 20";               run DCP2=1 $B/N100_dense_s$s.json dcp2_${C}_baaz_dense_s$s 100 dense $s
  EXTRA="--obsrv-mgl 20 --titrate 0.8"; run DCP2=1 $B/titr0.8/N100_sparse_s$s.json dcp2_${C}_baaz_titr_sparse_s$s 100 sparse $s
done
sleep 30; while pgrep -f "dc_tuned" > /dev/null; do sleep 60; done
mkdir -p $R/dc_tuned; $PY scripts/regimen/dc_phase2_table.py > $R/dc_tuned/table.txt 2> $R/dc_tuned/table.err
say "DCPHASE2_DONE"
