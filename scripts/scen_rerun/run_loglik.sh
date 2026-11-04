#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# -2LL of one existing Monolix fit (published / definit / autoinit) of the 3-scenario rerun, on a scratch COPY of the
# project (originals untouched). Output results/scen_rerun/loglik/<ID>_<fit>.json (+ log in loglik/logs/).
#   run_loglik.sh <scenario> <seed> <published|definit|autoinit>
set -u
SC=$1; SEED=$2; FIT=$3; ROOT=$REPO_ROOT; cd $ROOT
ID=$((SC * 10000 + SEED)); SFX=; [ $FIT != published ] && SFX=_$FIT
L=results/scen_rerun/loglik; OUT=$L/${ID}_$FIT.json; mkdir -p $L/work $L/logs
[ -f $OUT ] && { echo "exists: $OUT"; exit 0; }
SRC=$(ls -d results/paper_repro/w*/results/$ID$SFX 2>/dev/null)
if [ "$(echo $SRC | wc -w | tr -d ' ')" -gt 1 ]; then
  # duplicates (a paused launch left an unfinished copy in another worktree): keep the one whose estimates are the
  # ones copied into the run dir by the COMPLETED run
  RUN=results/scen_rerun/runs/s${SC}_seed$(printf %03d $SEED); REF=$RUN/populationParameters${SFX}.txt; PICK=
  for c in $SRC; do [ -f $REF ] && cmp -s $c/2_test_tacro/populationParameters.txt $REF && PICK=$c; done
  [ -n "$PICK" ] || { echo "no unique fit dir for $ID$SFX (duplicates, none matches $REF)"; exit 1; }
  echo "duplicates for $ID$SFX: using $PICK (matches $REF)"; SRC=$PICK
fi
[ "$(echo $SRC | wc -w | tr -d ' ')" = 1 ] || { echo "no fit dir for $ID$SFX"; exit 1; }
[ -f $SRC/2_test_tacro/populationParameters.txt ] || { echo "no estimates in $SRC"; exit 1; }
[ -f $L/test_model.txt ] || cp results/paper_repro/w01/test_model.txt $L/test_model.txt    # = commit 2eaadf6; the project reads ../../test_model.txt
D=$L/work/${ID}_$FIT; rm -rf $D; mkdir -p $D
cp $SRC/2_test_tacro.mlxtran $SRC/virtual_cohort_train.csv $D/ && cp $SRC/2_test_tacro/populationParameters.txt $D/est.txt || exit 1
OMP_NUM_THREADS=1 MLX_THREADS=1 nice -n 5 Rscript scripts/scen_rerun/loglik_fit.r $ROOT/$D/2_test_tacro.mlxtran $ROOT/$D/est.txt $ROOT/$OUT \
  > $L/logs/${ID}_$FIT.log 2>&1 && rm -rf $D && echo "done $ID $FIT: $(grep -o 'OFV = .*' $L/logs/${ID}_$FIT.log)" || { echo "FAILED $ID $FIT (see $L/logs/${ID}_$FIT.log)"; exit 1; }
