#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Review 22/10.: (1) redo the 12-sample tacrolimus regimen benchmark with the intensive profile after the
# FIRST dose (design 'visit1': 20, 40 min, 1, 1.5, 2, 3, 4, 6, 9 h + troughs before doses 2-4) and a positive readout for our model;
# (OT) enforce gamma > 0 in OT-FiLM (--film-gamma-pos) and redo the Table 2 OT-FiLM row (10 seeds, same command as queue_weekend.sh).
# Phase A: readout pilot (seed 0, both scenarios: softplus beta 20 / 50 and the current identity readout) + EB-VAE + Lu + OT-FiLM.
# Phase B (after the pilot): our model (selected beta) seeds 1-4 and the dose-input ablation (tuned z12, same readout) seeds 0-4.
# PRE-STATED selection rule: the beta with the lower training-regimen median |AUC err| (mean of the two scenarios, seed 0); test regimens
# are never used. If both softplus pilots are > 3 points worse than the identity readout on the training regimen, phase B is skipped
# (ABORT_READOUT) anddecides. Gate: <= 8 python3 jobs (CPU cap 800 %). Marker REVIEW22_DONE.
P=$REPO_ROOT; EB=$REPO_ROOT/third_party/baaz_ebvae; PY=${PYTHON:-python3}; L=$P/logs/review22; LOG=$P/logs/queue_review22.log
R=$P/results/regimen_dh; G2=$P/results/s4_pnoise_win_gpos; T=confound_vc00_s4_pnoise_win_n100
cd $P; mkdir -p $L $G2/eval
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 8 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 8 ] && return; done; }
sub_of () { [ $1 = 2 ] && echo tacro_s2 || echo tacro; }
nz_of () { [ $1 = 2 ] && echo 3.1 || echo 4; }

ours () {   # scenario seed readout(id|beta)
  local sc=$1 s=$2 rd=$3 d x; if [ $rd = id ]; then d=Bv1id; x=""; else d=Bv1sp$rd; x="--readout-softplus $rd"; fi
  local o=$R/$(sub_of $sc)/$d/N100_visit1_s$s.json; [ -e $o ] && return; mkdir -p $(dirname $o); gate
  env TACRO_SCENARIO=$sc OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 visit1 $s $o --iters 6000 --sim tacro \
      --obsrv-mgl $(nz_of $sc) --kick-phys 32 --enc-dose norm $x > $L/ours_sc${sc}_${d}_s$s.log 2>&1 < /dev/null &
  say "start ours sc$sc $d s$s"; sleep 20
}
dc () {     # scenario seed beta: tuned dose-input ablation (z12: zero-init output layer, 12000 iterations), same readout as ours
  local sc=$1 s=$2 b=$3; local o=$R/$(sub_of $sc)/dc_z12_v1sp$b/N100_visit1_s$s.json; [ -e $o ] && return; mkdir -p $(dirname $o); gate
  env TACRO_SCENARIO=$sc OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/dh_regimen.py 100 visit1 $s $o --iters 12000 --sim tacro \
      --obsrv-mgl $(nz_of $sc) --dose-cond --enc-dose norm --dc-zero-init --readout-softplus $b > $L/dc_sc${sc}_s$s.log 2>&1 < /dev/null &
  say "start dose-input sc$sc s$s"; sleep 20
}
ebvae () {  # scenario seed (unmodified training; curves dumped for the figure)
  local sc=$1 s=$2 ed; [ $sc = 2 ] && ed=$EB/ours/regimen/res_tacro_s2/ctrl || ed=$EB/ours/regimen/res_tacro/ctrl
  local o=$ed/N100_visit1_s$s.json; [ -e $o ] && return; mkdir -p $ed; gate
  ( cd $EB && env TACRO_SCENARIO=$sc OMP_NUM_THREADS=1 DUMP_CURVES=1 nohup nice -n 5 $PY -u ours/regimen/run_regimen_tacro.py 100 visit1 $s none $o \
      > $L/ebvae_sc${sc}_s$s.log 2>&1 < /dev/null & )
  say "start EB-VAE sc$sc s$s"; sleep 20
}
lu () {     # model(is|nmi) seed; linear scenario only, as the published 12-sample Lu rows
  local m=$1 s=$2 x=""; [ $m = nmi ] && x="--euler-dt 0.0833333333333333"
  local o=$R/tacro_s2/lu$m/N100_visit1_s$s.json; [ -e $o ] && return; gate
  env TACRO_SCENARIO=2 OMP_NUM_THREADS=1 nohup nice -n 5 $PY scripts/regimen/lu_regimen.py $m 100 visit1 $s $o --iters 6000 --sim tacro $x \
      > $L/lu${m}_s$s.log 2>&1 < /dev/null &
  say "start Lu $m s$s"; sleep 20
}
film () {   # seed: Table 2 OT-FiLM command of queue_weekend.sh + --film-gamma-pos, own save root (filenames do not encode the flag)
  local s=$1; local D=$G2/film_n100_s$s; ls $D/exp_film_run/$T/traj/*_ep006000.ckpt > /dev/null 2>&1 && return
  pgrep -f -- "--save $D/" > /dev/null && return; mkdir -p $D/exp_film_run/$T/traj; gate
  nohup nice -n 5 env OMP_NUM_THREADS=1 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
      --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
      --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
      --static-dim 4 --save $D/ --eval-every 300 --ckpt-every 300 --film-gamma-pos --log-suffix __gpos_s$s > $L/film_gpos_s$s.log 2>&1 < /dev/null &
  say "start OT-FiLM gamma>0 s$s"; sleep 20
}

say "REVIEW22 queue start"
# ---- phase A
for sc in 2 4; do ours $sc 0 20; ours $sc 0 50; ours $sc 0 id; done
for s in 1 2; do film $s; done
for s in 0 1 2 3 4; do ebvae 2 $s; ebvae 4 $s; done
for s in 0 1 2 3 4; do lu is $s; lu nmi $s; done
for s in 3 4 5 6 7 8 9 10; do film $s; done
say "phase A launched"

# ---- readout selection (pre-stated rule above)
sleep 60; while pgrep -f "Bv1(sp|id)[0-9]*/N100_visit1_s0" > /dev/null; do sleep 60; done
SEL=$($PY - <<EOF
import json, numpy as np
R = '$R'; tr = lambda d, sc: json.load(open(f"{R}/{'tacro_s2' if sc == 2 else 'tacro'}/{d}/N100_visit1_s0.json"))['individual']['train']['auc_abs_median']
m = {d: np.mean([tr(d, sc) for sc in (2, 4)]) for d in ('Bv1sp20', 'Bv1sp50', 'Bv1id')}
best = min(('Bv1sp20', 'Bv1sp50'), key=lambda d: m[d])
ok = m[best] <= m['Bv1id'] + 3.0
json.dump(dict(train_err=m, selected=best[5:] if ok else None), open(f'{R}/tacro/visit1_readout_selected.json', 'w'), indent=1)
print(best[5:] if ok else 'ABORT')
EOF
)
say "readout pilot -> $SEL ($(cat $R/tacro/visit1_readout_selected.json | tr -d '\n '))"
if [ "$SEL" = ABORT ] || [ -z "$SEL" ]; then
  say "ABORT_READOUT: both softplus readouts > 3 points worse than identity on the training regimen; phase B skipped"
else
  # ---- phase B
  for s in 1 2 3 4; do ours 2 $s $SEL; ours 4 $s $SEL; done
  for s in 0 1 2 3 4; do dc 2 $s $SEL; dc 4 $s $SEL; done
  say "phase B launched (beta $SEL)"
fi

# ---- OT-FiLM gamma>0: score Table 2 (in-range test patients, 1100) once each training ends
sleep 60
while pgrep -f -- "--save $G2/" > /dev/null; do sleep 120; done
for s in 1 2 3 4 5 6 7 8 9 10; do
  CK=$(ls $G2/film_n100_s$s/exp_film_run/$T/traj/*_ep006000.ckpt 2>/dev/null)
  [ "$(echo "$CK" | grep -c ckpt)" = 1 ] || { say "MISSING ckpt film gpos s$s"; continue; }
  o=$G2/eval/ep006000_film_n100_s${s}_on_in.json; [ -f $o ] && continue; gate
  nohup nice -n 5 env OMP_NUM_THREADS=1 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py \
      --experiment confound_vc00_s4_pnoise_win --data-dir ./results/exp_film_run --ckpt $CK --scale-from $T --eval-split test \
      --label gpos_film_s${s}_in --out-json $o > $L/eval_film_gpos_s$s.log 2>&1 < /dev/null &
  sleep 5
done
sleep 60; while pgrep -f "N100_visit1_s|s4_pnoise_win_gpos|gpos_film_s" > /dev/null; do sleep 120; done
say "REVIEW22_DONE"
