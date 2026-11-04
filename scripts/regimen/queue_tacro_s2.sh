#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Scenario 2 of the tacrolimus regimen benchmark: TACRO_SCENARIO=2 = linear elimination, CL ~ Ht^-3.14 x 2^CYP3A5, same design / arms /
# noise / patients' random stream. Dose-event model (Table 4 architecture; fixed noise 3.1 ng/mL = 2 x RMS residual sd 1.55 on the
# sparse training samples, the rule of 4 ng/mL in scenario 4 and 20 mg/L on Baaz's) and EB-VAE (run_regimen_tacro.py, curves dumped),
# interleaved per seed: sparse s0-4 then dense s0-2. Then the 3 dose-cond DENSE runs of scenario 4 that queue_tacro_versatility.sh
# had not launched (stopped 13/10 ~12:50 to give scenario 2 priority); TACROVERS_DONE is written to its log when every scenario-4
# dose-cond run has finished. Gate <= 6 python3 jobs (re-check after 41 s). Marker TACROS2_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh; L=logs/regimen_dh; EB=$REPO_ROOT/third_party/baaz_ebvae; ER=$EB/ours/regimen/res_tacro_s2/ctrl
mkdir -p $L $ER $R/tacro_s2/Bfixkp3.1n
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
busy () { [ -f "$1" ] || pgrep -f -- "$1" > /dev/null; }
ours () {   # design seed
    o=$R/tacro_s2/Bfixkp3.1n/N100_$1_s$2.json; if busy $o; then say "skip $o"; return; fi; gate
    env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $1 $2 $o --iters 6000 --sim tacro \
        --obsrv-mgl 3.1 --kick-phys 32 --enc-dose norm > $L/tacro_s2_Bn_$1_s$2.log 2>&1 < /dev/null &
    say "start ours s2 $1 s$2"; sleep 20; }
ebvae () {
    o=$ER/N100_$1_s$2.json; if busy $o; then say "skip $o"; return; fi; gate
    ( cd $EB && env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 DUMP_CURVES=1 nohup nice -n 5 $PY -u ours/regimen/run_regimen_tacro.py 100 $1 $2 none $o \
        > $L/tacro_s2_ebvae_$1_s$2.log 2>&1 < /dev/null & )
    say "start EB-VAE s2 $1 s$2"; sleep 20; }
for s in 0 1 2 3 4; do ours sparse $s; ebvae sparse $s; done
for s in 0 1 2; do ours dense $s; ebvae dense $s; done
for s in 0 1 2; do   # scenario-4 dose-cond dense, left over from queue_tacro_versatility.sh
    o=$R/tacro/dosecond_n/N100_dense_s$s.json; if busy $o; then say "skip $o"; continue; fi; gate
    env OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 dense $s $o --iters 6000 --sim tacro --dose-cond \
        --enc-dose norm --obsrv-mgl 4 > $L/tacro_dcn_dense_s$s.log 2>&1 < /dev/null &
    say "start tacro_dcn_dense_s$s"; sleep 20
done
sleep 30; while pgrep -f "regimen_dh/tacro/dosecond_n" > /dev/null; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') TACROVERS_DONE (written by queue_tacro_s2.sh)" >> logs/queue_tacro_versatility.log
while pgrep -f "tacro_s2/|res_tacro_s2/" > /dev/null; do sleep 60; done
say "TACROS2_DONE"
