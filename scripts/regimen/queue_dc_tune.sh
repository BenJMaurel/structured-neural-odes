#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Dose-cond tuning.
# 1. waits for TACROVISIT_DONE (logs/queue_tacro_visit.log); 2. installs scripts/regimen/staged/dh_regimen.py.dcinit (adds the
#    opt-in --dc-zero-init) only if dh_regimen.py is unchanged since staging (md5), backup dh_regimen.py.bak_dcinit; 3. checks that
#    the DEFAULT dose-cond path is bit-identical (3-iteration JSON, backup vs new, minutes_train excluded), restores the backup and
#    aborts otherwise; 4. Baaz linear sparse s0-4 x {z6: zero-init 6000 it, z12: zero-init 12000 it, d12: default init 12000 it},
#    all --dose-cond --enc-dose norm --obsrv-mgl 20 (= dosecond_n, Table 4); 5. scripts/regimen/dc_tune_table.py ->
#    results/regimen_dh/dc_tune/table.txt + selected.json (selection on the TRAINING-regimen error only). Gate <= 6 python3 jobs.
#    Log logs/queue_dc_tune.log, marker DCTUNE_DONE.
P=$REPO_ROOT; PY=${PYTHON:-python3}; L=$P/logs/regimen_dh; LOG=$P/logs/queue_dc_tune.log
D=$P/results/regimen_dh/dc_tune; ST=$P/scripts/regimen/staged; S=$P/scripts/regimen/dh_regimen.py
cd $P; mkdir -p $L $D
say () { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
njobs () { ps -A -o stat=,args= | awk '$1 !~ /^T/ && $2 ~ /python3$/ && $0 !~ /multiprocessing/' | wc -l; }
gate () { while true; do while [ "$(njobs)" -ge 6 ]; do sleep 60; done; sleep 41; [ "$(njobs)" -lt 6 ] && return; done; }
say "waiting for TACROVISIT_DONE"
[ -n "${SKIP_WAIT:-}" ] || until grep -q TACROVISIT_DONE $P/logs/queue_tacro_visit.log; do sleep 120; done
say "TACROVISIT_DONE seen"
C=$D/check; mkdir -p $C; F="--iters 3 --obsrv-mgl 20 --dose-cond --enc-dose norm"
gate; $PY $S.bak_dcinit 100 sparse 0 $C/old.json $F > $C/old.log 2>&1
gate; $PY $S 100 sparse 0 $C/new.json $F > $C/new.log 2>&1
gate; $PY $S 100 sparse 0 $C/zero.json $F --dc-zero-init > $C/zero.log 2>&1
$PY - $C <<'PYEOF' >> $LOG 2>&1
import json, sys
c = sys.argv[1]; a, b = json.load(open(f'{c}/old.json')), json.load(open(f'{c}/new.json'))
for j in (a, b): j.pop('minutes_train', None)
ok = a == b; z = json.load(open(f'{c}/zero.json'))
print(f'identity check (default dose-cond path, 3 iterations): {"IDENTICAL" if ok else "DIFFERENT"}; zero-init run ok: {z.get("dc_zero_init")}')
sys.exit(0 if ok else 1)
PYEOF
if [ $? -ne 0 ]; then cp $S.bak_dcinit $S; say "ABORT: default path changed; backup restored"; exit 1; fi
grep -h "^it     1" $C/old.log $C/zero.log | sed 's/^/  first iteration (default, zero-init): /' >> $LOG
for s in 0 1 2 3 4; do
  for cfg in z6 z12 d12; do
    case $cfg in z6) IT=6000; X="--dc-zero-init";; z12) IT=12000; X="--dc-zero-init";; d12) IT=12000; X="";; esac
    o=$D/$cfg/N100_sparse_s$s.json; mkdir -p $D/$cfg
    [ -e $o ] && continue
    gate
    env OMP_NUM_THREADS=1 nohup nice -n 5 $PY $S 100 sparse $s $o --iters $IT --obsrv-mgl 20 --dose-cond --enc-dose norm $X \
        > $L/dctune_${cfg}_sparse_s$s.log 2>&1 < /dev/null &
    say "start $cfg sparse s$s"; sleep 20
  done
done
sleep 30; while pgrep -f "results/regimen_dh/dc_tune/" > /dev/null; do sleep 60; done
$PY scripts/regimen/dc_tune_table.py > $D/table.txt 2> $D/table.err
say "DCTUNE_DONE"
