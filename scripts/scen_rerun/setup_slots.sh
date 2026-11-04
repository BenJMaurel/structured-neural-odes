#!/bin/bash
# Creates the slot folders of chain_scen_rerun.sh: copies of paper_era/, the code of the earlier pipeline, one per parallel run
# because that pipeline writes fixed file names in its working directory.
#   setup_slots.sh [number of slots, default 10]
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
for i in $(seq 1 "${1:-10}"); do
  w=results/paper_repro/w$(printf %02d "$i")
  [ -d "$w" ] || { mkdir -p results/paper_repro && cp -R paper_era "$w"; }
done
echo "slots ready in results/paper_repro/"
