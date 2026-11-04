#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Job 1 scoring (review 22/10): clip0_eval on the 12-samples-around-the-first-dose design ('visit1', 5 seeds, both scenarios), dose-event
# model with the softplus(beta 20) readout (Bv1sp20) and the dose-input ablation dc_z12_v1sp20, then Lu's regimen model under the factual
# encoder protocol. Outputs results/regimen_dh/clip0/<model>_sc<sc>_visit1_s<seed>_v1{,fact}.json; table scripts/review22/visit1_table.py.
cd $REPO_ROOT; PY=${PYTHON:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CLIP_DESIGNS=visit1:5
CLIP_DE_SUB=Bv1sp20 CLIP_DC_SUB=dc_z12_v1sp20 CLIP_TAG_SFX=_v1 nice -n 5 $PY scripts/regimen/clip0_eval.py > logs/review22/clip0_visit1.log 2>&1 || echo "FAILED main"
CLIP_LUIS_PROTOCOL=factual CLIP_TAG_SFX=_v1fact nice -n 5 $PY scripts/regimen/clip0_eval.py luis > logs/review22/clip0_visit1_luisfact.log 2>&1 || echo "FAILED luis factual"
$PY scripts/review22/visit1_table.py > /dev/null 2>&1
echo CLIP0_VISIT1_DONE
