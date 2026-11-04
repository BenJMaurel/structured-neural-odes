#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Free-flow B with FIXED noise"we have not tested the free dynamic with fixed noise. Just
# test that"): scripts/regimen/dh_regimen.py (original B, unchanged otherwise) + --obsrv-std 0.05 (B2's value), 6000 its,
# N100 sparse s1 / dense s0, linear + saturable (REGIMEN_KM=500) simulators, randomised + titrated (rho 0.8) arms = 8 runs.
# Gate: total python CPU < 500 % before each 1-thread job (600 % cap; other sessions' jobs count).
#   outputs  results/regimen_dh/[mm500/][titr0.8/]Bfix/N100_<d>_s<s>.{json,pt}; table results/regimen_dh/bfix_table.txt
#   log      logs/queue_bfix.log, per-run logs logs/regimen_dh/bfix_*.log, marker BFIX_DONE. Do not edit while running.
cd $REPO_ROOT
PY=${PYTHON:-python3}
R=results/regimen_dh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -eo pcpu=,command= | awk '/python/ {s += $1} END {print int(s)}'; }
gate () { sleep 30; while [ "$(cpu)" -ge 500 ]; do sleep 60; done; }
mkdir -p logs/regimen_dh
# sim:design:seed:titrated -- saturable sparse first (the open question), then linear, then dense
for j in mm:sparse:1:0 mm:sparse:1:1 lin:sparse:1:0 lin:sparse:1:1 mm:dense:0:0 mm:dense:0:1 lin:dense:0:0 lin:dense:0:1; do
  IFS=: read sim d s ti <<< "$j"
  sub=""; X=""; ENV=""
  [ $sim = mm ] && { sub="mm500/"; X="--no-check"; ENV="REGIMEN_KM=500"; }
  [ $ti = 1 ] && { sub="${sub}titr0.8/"; X="$X --titrate 0.8"; }
  out=$R/${sub}Bfix/N100_${d}_s$s.json; mkdir -p $(dirname $out)
  [ -f $out ] && { say "skip $out (exists)"; continue; }
  gate
  lg=logs/regimen_dh/bfix_${sim}_${d}_s${s}_t${ti}.log
  env $ENV nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s $out --iters 6000 --obsrv-std 0.05 $X > $lg 2>&1 &
  say "start $sim $d s$s titrated=$ti (pid $!, cpu before $(cpu) %) -> $out"
done
while pgrep -f "dh_regimen.py 100 .*--obsrv-std" > /dev/null; do sleep 60; done
say "all runs finished; failures: $(grep -L REGIMEN_DH_DONE logs/regimen_dh/bfix_*.log | tr '\n' ' ')"
$PY scripts/regimen/bfix_table.py > $R/bfix_table.txt 2> logs/bfix_table.err
say "table -> $R/bfix_table.txt"
say "BFIX_DONE"
