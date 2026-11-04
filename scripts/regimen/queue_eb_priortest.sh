#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# PRIOR-GAUGE TEST of the EB-VAE. Waits for DCVISIT_DONE (the dose-cond
# chain), then the EB-VAE on tacrolimus scenario 2 (linear) DENSE s0-2 with BAAZ_PRIOR_NO_DOSE=1 (prior mean ignores the first amount,
# covariates kept, both training stages; everything else as res_tacro_s2/ctrl) -> res_tacro_s2/prior_nodose/. Prediction declared
# before the runs (Prop. priorgauge (iii) of docs/dose_injection_theory.tex): if the arm split of 0.00zz57 comes from the prior, the
# 4-mg vs 2-mg exposure ratio is recovered and the split shrinks (amt1.75 for 4-mg patients far below +32 %, amt4.25 for 2-mg patients
# far above -20 %); if it persists, the encoder channel (raw concentrations reveal the arm) or psi carries it.
# Gate <= 6 python3 jobs. Log logs/queue_eb_priortest.log, marker EBPRIOR_DONE. Then tacro_arm_split-style comparison.
P=$REPO_ROOT; EB=$REPO_ROOT/third_party/baaz_ebvae; PY=${PYTHON:-python3}; L=$P/logs/regimen_dh
LOG=$P/logs/queue_eb_priortest.log; D=$EB/ours/regimen/res_tacro_s2/prior_nodose
cd $P; mkdir -p $L $D
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
say "waiting for DCVISIT_DONE"
[ -n "${SKIP_WAIT:-}" ] || until grep -q DCVISIT_DONE $P/logs/queue_dc_visit.log 2>/dev/null; do
  grep -q ABORT $P/logs/queue_dc_visit.log 2>/dev/null && { say "dose-cond chain aborted: starting anyway"; break; }; sleep 120; done
for s in 0 1 2; do
  o=$D/N100_dense_s$s.json; [ -e $o ] && continue; gate
  ( cd $EB && env TACRO_SCENARIO=2 BAAZ_PRIOR_NO_DOSE=1 OMP_NUM_THREADS=1 DUMP_CURVES=1 nohup nice -n 5 $PY -u ours/regimen/run_regimen_tacro.py \
      100 dense $s none $o > $L/eb_priornodose_tacro2_dense_s$s.log 2>&1 < /dev/null & )
  say "start prior_nodose dense s$s"; sleep 20
done
sleep 30; while pgrep -f "prior_nodose/N100_" > /dev/null; do sleep 60; done
$PY scripts/regimen/eb_priortest_table.py > $D/table.txt 2> $D/table.err
say "EBPRIOR_DONE"
