#!/usr/bin/env bash
# Second driver: smallest families first, alongside run_all_families.sh (largest first).
# run_family_full.sh takes a lock (data/locks/famF), so each family is run once.
# Logs go to logs/run_family${F}_full_c.log so the other driver's skip message can't clobber them.
set -uo pipefail
cd /home/seismic/LFE
NPAR=${NPAR:-2}
export PROCS=${PROCS:-6}
FAMS="${FAMS:-1 24 14 25 12 15 7 17 2 10 13 16 26 5 4 3 20 21 18 22 6 19}"
echo "[$(date '+%F %T')] middle-first driver: $FAMS"
echo $FAMS | tr ' ' '\n' | xargs -P $NPAR -I{} bash -c \
  'bash scripts/run_family_full.sh {} > logs/run_family{}_full_c.log 2>&1; echo "[$(date "+%F %T")] fam {} finished (driver c): $(tail -1 logs/run_family{}_full_c.log)"'
echo "[$(date '+%F %T')] middle-first driver finished"
