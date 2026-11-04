#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Tacrolimus transposition of the regimen benchmark. Model = Table 4's dose-event model (free drug flow, jump a*b(p), FIXED noise, dose-normalised
# encoder) on gen_tacro_regimens.py (paper's scenario-4 simulator, Prograf q12h, arms 2/4 mg, CYP3A5 + hematocrit as static inputs).
# Fixed noise 4 ng/mL = twice the simulator's RMS residual sd on the training samples (2.0), the rule that gave 20 mg/L on Baaz's (2 x 10).
# Sparse randomised s0-4, dense s0-2, titrated (rho 0.8 on log Vmax) sparse s0-2. Gate <= 6 python3 jobs. Marker TACROREG_DONE.
cd $REPO_ROOT
PY=${PYTHON:-python3}; R=results/regimen_dh/tacro; L=logs/regimen_dh; mkdir -p $L
say () { echo "$(date '+%m-%d %H:%M:%S') $*"; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while [ "$(njobs)" -ge 6 ]; do sleep 60; done; }
B="--sim tacro --obsrv-mgl 4 --kick-phys 32 --enc-dose norm"
J=()
for s in 0 1 2 3 4; do J+=("sparse $s $R/Bfixkp4n/N100_sparse_s$s.json tacro_Bn_sparse_s$s $B"); done
for s in 0 1 2; do J+=("dense $s $R/Bfixkp4n/N100_dense_s$s.json tacro_Bn_dense_s$s $B"); done
for s in 0 1 2; do J+=("sparse $s $R/titr0.8/Bfixkp4n/N100_sparse_s$s.json tacro_Bn_titr_sparse_s$s $B --titrate 0.8"); done
for j in "${J[@]}"; do
    set -- $j; d=$1 s=$2 o=$3 lg=$4; shift 4
    if [ -f "$o" ] || pgrep -f -- "$o" > /dev/null; then say "skip $o"; continue; fi
    gate; mkdir -p "$(dirname $o)"
    env OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 $d $s "$o" --iters 6000 "$@" > "$L/$lg.log" 2>&1 < /dev/null &
    say "start $lg"; sleep 20
done
while pgrep -f "dh_regimen.py 100 .*regimen_dh/tacro/" > /dev/null; do sleep 60; done
say "TACROREG_DONE"
