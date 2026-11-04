#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# Extra seeds 1-2 for the SPARSE arms of the regimen experiment (2026-10-03 15:15): seed 0 of N100 sparse failed to fit even
# its training regimen (bias -26 %), so one seed per arm cannot answer "how much data". -> res/N<n>_sparse_s<seed>.json.
# Marker REGIMEN_SEEDS_DONE.
cd $REPO_ROOT/third_party/baaz_ebvae; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
for s in 1 2; do for n in 25 100 250; do
  nohup nice -n 5 ${PYTHON:-python3} -u ours/regimen/run_regimen.py $n sparse $s ours/regimen/res/N${n}_sparse_s$s.json \
    > ours/regimen/logs/N${n}_sparse_s$s.log 2>&1 & sleep 5
done; done; wait; echo REGIMEN_SEEDS_DONE
