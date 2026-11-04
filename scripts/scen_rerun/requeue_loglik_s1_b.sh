#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Second hold (02/10 18:15): unstarted scenario-1 likelihood jobs pre-claimed so the trough-only oracle can start under
# the CPU cap. Once oracle_trough.py runs (or has finished), remove the held claims, wait for the loglik chain to exit,
# relaunch it for s1 (skips finished jobs). Log logs/requeue_loglik_s1_b.log.
cd $REPO_ROOT; L=results/scen_rerun/loglik
until pgrep -f "scripts/ident/oracle_trough.py" >/dev/null || grep -q ORACLE_DONE logs/oracle_trough.log 2>/dev/null; do sleep 30; done
for c in $(sed 's/^held://' $L/held_s1_b.txt); do rmdir $L/claims/$c 2>/dev/null; done; echo "[$(date '+%F %T')] held claims removed"
while pgrep -f "bash scripts/scen_rerun/chain_loglik.sh" >/dev/null; do sleep 60; done
NSLOT=6 SET="1:$(seq -s, 1 10)" bash scripts/scen_rerun/chain_loglik.sh > logs/chain_loglik_s1_b.log 2>&1
echo "[$(date '+%F %T')] REQUEUE_S1_B_DONE"
