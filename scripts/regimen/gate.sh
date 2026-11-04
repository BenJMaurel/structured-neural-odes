# Shared CPU gate for the regimen queues (source it). cap: 600 % (<= 6 single-thread jobs) by default; he allowed
# up to 1000 % until 2026-10-07 16:00. Counts every heavy python job on the machine, other sessions' included.
running () { ps -A -o stat=,command= | grep "python3" | grep -v "multiprocessing\|grep" | grep -v "^T" \
  | grep -cE "run_models.py|seeded_eval.py|test_film_matched.py|train_|dh_regimen|lu_regimen|run_regimen|mapbe_regimen|scripts/theory/"; }
cpu () { ps -A -o pcpu= | awk '{s+=$1} END {printf "%d", s}'; }
limits () { if [ $(date +%s) -lt $(date -d "2026-10-07 16:00" +%s) ]; then echo "10 950"; else echo "6 550"; fi; }
gate () { while true; do read SL CP <<< "$(limits)"; [ $(running) -lt $SL ] && [ $(( $(cpu) + 100 )) -le $CP ] && return 0; sleep 20; done; }
