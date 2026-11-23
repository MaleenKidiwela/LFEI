#!/usr/bin/env bash
# Second driver: smallest families first, alongside run_all_families.sh (largest first).
# run_family_full.sh takes a lock (data/locks/famF), so each family is run once.
# Logs go to logs/run_family${F}_full_b.log so the other driver's skip message can't clobber them.
set -uo pipefail
cd /home/seismic/LFE
NPAR=${NPAR:-2}
export PROCS=${PROCS:-6}
FAMS="11 9 3 4 5 26 16 13 10 2 17 7 15 12 25 14 24 1 20 21 18 22 6 19"
echo "[$(date '+%F %T')] smallest-first driver: $FAMS"
echo $FAMS | tr ' ' '\n' | xargs -P $NPAR -I{} bash -c \
  'bash scripts/run_family_full.sh {} > logs/run_family{}_full_b.log 2>&1; echo "[$(date "+%F %T")] fam {} finished (driver b): $(tail -1 logs/run_family{}_full_b.log)"'
echo "[$(date '+%F %T')] smallest-first driver finished"
