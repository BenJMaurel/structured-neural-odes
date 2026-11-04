#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Re-queue the scenario-1 likelihood jobs held back on 02/10 14:35 (claims pre-created so the linear-NLME fits of
# chain_lin_s4.sh could get CPU): wait until both of those fits have launched (or that chain ended), remove the held
# claims, wait for the first loglik chain to exit, then relaunch it (it skips finished jobs). Log logs/requeue_loglik_s1.log.
cd $REPO_ROOT
L=results/scen_rerun/loglik
[ -n "${SKIP_WAIT:-}" ] || until grep -q "mm fit launched\|mm fit exists\|LIN_S4_DONE\|FAILED" logs/chain_lin_s4.log 2>/dev/null; do sleep 60; done
for c in $(sed 's/^held://' $L/held_s1.txt); do rmdir $L/claims/$c 2>/dev/null; done
echo "[$(date '+%F %T')] held claims removed"
while pgrep -f "bash scripts/scen_rerun/chain_loglik.sh" >/dev/null; do sleep 60; done
NSLOT=6 SET="1:$(seq -s, 1 10)" bash scripts/scen_rerun/chain_loglik.sh > logs/chain_loglik_s1.log 2>&1
echo "[$(date '+%F %T')] REQUEUE_S1_DONE"
