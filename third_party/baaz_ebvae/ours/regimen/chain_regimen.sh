#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# EB-VAE regimen-generalisation x data size (2026-10-03): N per arm {25, 100, 250} x sampling {dense (their 61), sparse (6)},
# seed 0, their full training (100 + 300 epochs). -> ours/regimen/res/N<n>_<design>.json (+ _data/). Marker REGIMEN_ALL_DONE.
cd $REPO_ROOT/third_party/baaz_ebvae; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg; mkdir -p ours/regimen/res ours/regimen/logs
for n in 25 100 250; do for d in sparse dense; do
  nohup nice -n 5 ${PYTHON:-python3} -u ours/regimen/run_regimen.py $n $d 0 ours/regimen/res/N${n}_$d.json \
    > ours/regimen/logs/N${n}_$d.log 2>&1 & sleep 5
done; done; wait; echo REGIMEN_ALL_DONE
