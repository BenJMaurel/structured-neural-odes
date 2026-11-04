#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Clinical baselines for the counterfactual tasks (review comment, 22/10: "The counterfactual task has no clinical baseline").
# TWO-VISIT (Table 2 cohort: confound_vc00_s4_pnoise_win, 100 training patients, 1100 test): MAP-BE under the two models that drop
#   hematocrit entirely (Table 1's omission), fitted by SAEM on the same 100 patients (fit_s4_select.R ... noht):
#     Michaelis-Menten without HT  -> results/monolix/sel_mm_noht_pnoise_win_n100 -> popebe_mlx_mm_noht_pnoise_win_n1100.csv
#     linear without HT (= Table 1's MAP-BE model) -> sel_lin_noht_pnoise_win_n100 -> popebe_mlx_lin_noht_pnoise_win_n1100.csv
#   (true population model, MM true covariates, MM without HT on Km and linear with HT already exist in results/s4/ebe).
# REGIMEN BENCHMARK (Table 3 confirmation sets, REGIMENS_CONFIRM=1: sparse seeds 5-14 randomised and titrated rho 0.8, dense 3-7):
#   MAP-BE with the simulator's structure (one compartment, linear), population fitted by SAEM on the 200 training patients
#   (fit_regimen.R lin), superposition over each regimen (mapbe_pop.py fit) -> results/review22/mapbe_regimen/{ctrl,titr0.8}/.
#   Data check: the exported training samples must equal the EB-VAE confirmation run's train.csv.
# One process at a time, Monolix on one thread, waiting while >= 8 python3 / R jobs run. Marker BASELINES_DONE.
P=$REPO_ROOT; PY=${PYTHON:-python3}; L=$P/logs/review22/baselines; LOG=$P/logs/queue_baselines.log
O=$P/results/review22/mapbe_regimen; EBC=$REPO_ROOT/third_party/baaz_ebvae/ours/regimen/res_titr/confirm
cd $P; mkdir -p $L $O/data $O/saem $O/ctrl $O/titr0.8
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && ($2 ~ /python3$/ || ($2 ~ /(Rscript|exec\/R)$/ && $0 ~ /scripts\/monolix/)) && $0 !~ /multiprocessing/' | wc -l; }   # only our Monolix fits count (an idle editor R session must not)
gate () { while [ "$(njobs)" -ge 8 ]; do sleep 30; done; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
say "BASELINES start"

# ---- two-visit: no-HT population models
D=results/monolix/data/confound_vc00_s4_pnoise_win_n100.csv
for s in lin mm; do
  F=results/monolix/sel_${s}_noht_pnoise_win_n100
  if ! grep -q FIT_DONE $F/fit.log 2>/dev/null; then
    gate; mkdir -p $F; say "SAEM $s without HT"
    nice -n 5 Rscript scripts/monolix/fit_s4_select.R $D $F 1 $s 1 noht > $F/fit.log 2>&1 || say "SAEM $s noht FAILED"
  fi
  E=results/s4/ebe/popebe_mlx_${s}_noht_pnoise_win_n1100.csv
  if [ ! -f $E ] && grep -q FIT_DONE $F/fit.log; then
    gate; say "MAP-BE $s without HT on 1100 test patients"
    S=scripts/monolix/ebe_popmodel.py; [ $s = lin ] && S=scripts/monolix/ebe_popmodel_lin.py
    EBE_COHORT=confound_vc00_s4_pnoise_win nice -n 5 $PY $S $F/estimates.json 1100 0 $E > $L/popebe_${s}_noht.log 2>&1 || say "MAP-BE $s noht FAILED"
  fi
done
say "two-visit baselines done"

# ---- regimen benchmark: linear MAP-BE with SAEM population on the confirmation training sets
run () {   # design seed sub
  local d=$1 s=$2 sub=$3 x=""; [ $sub = titr0.8 ] && x="--titrate 0.8"
  local tag=${sub}_${d}_s$s out=$O/$sub/N100_${d}_s$s.json
  [ -f $out ] && return
  gate
  [ -f $O/data/$tag.csv ] || REGIMENS_CONFIRM=1 nice -n 5 $PY scripts/regimen/mapbe_pop.py export $d $s $O/data/$tag.csv $x > $L/export_$tag.log 2>&1
  $PY - $O/data/$tag.csv $EBC/$sub/N100_${d}_s${s}_data/train.csv <<'EOF' >> $LOG 2>&1 || { say "DATA CHECK FAILED $tag"; return; }
import sys, numpy as np, pandas as pd
a = pd.read_csv(sys.argv[1]); a = a[a.DV != '.'].astype({'DV': float}).sort_values(['ID', 'TIME'])
b = pd.read_csv(sys.argv[2], sep=';'); b = b[b.EVID == 0].sort_values(['ID', 'TIME'])
assert len(a) == len(b) and np.allclose(a.DV.values, b.DV.values, atol=1e-5), 'training samples differ from the EB-VAE run'
print(f'data check {sys.argv[1].split("/")[-1]}: identical ({len(a)} samples)')
EOF
  [ -f $O/saem/${tag}_lin/estimates.json ] || nice -n 5 Rscript scripts/monolix/fit_regimen.R $O/data/$tag.csv $O/saem/${tag}_lin lin > $L/saem_$tag.log 2>&1
  REGIMENS_CONFIRM=1 nice -n 5 $PY scripts/regimen/mapbe_pop.py fit $d $s $out --struct lin --pop $O/saem/${tag}_lin/estimates.json $x \
      > $L/mapbe_$tag.log 2>&1 && say "MAP-BE regimen $tag done" || say "MAP-BE regimen $tag FAILED"
}
for s in 5 6 7 8 9 10 11 12 13 14; do run sparse $s ctrl; run sparse $s titr0.8; done
for s in 3 4 5 6 7; do run dense $s ctrl; done
say "BASELINES_DONE"
