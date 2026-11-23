#!/usr/bin/env bash
# Driver: wait for the combined download, fill gaps, then run run_family_full.sh for all
# families except 23, NPAR at a time (largest first). If /home free space drops below
# MIN_FREE_GB after a family finishes, that family's HDF5 (rebuildable from raw) is removed.
set -uo pipefail
cd /home/seismic/LFE
PY=/home/seismic/anaconda3/envs/lfe/bin/python
NPAR=${NPAR:-2}
export PROCS=${PROCS:-12}
MIN_FREE_GB=${MIN_FREE_GB:-120}
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for download processes"
while ps -eo cmd | grep -q "[s]cripts/download_windows.py --catalog"; do sleep 60; done
log "download processes finished; filling missing/errored days (one pass)"
grep -h ERROR logs/download_allfam_chunk*.log > logs/download_allfam_errors.txt || true
for d in $(awk '{print $1}' logs/download_allfam_errors.txt); do
  rm -f data/waveforms/allfam_raw/${d:0:4}/$d/_log.json
done
$PY scripts/download_windows.py --catalog data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv --exclude-family 23 \
    --gap 300 --out data/waveforms/allfam_raw --start 2008-01-01 --end 2026-06-30 --workers 8 \
    > logs/download_allfam_fill.log 2>&1
log "fill pass done: $(grep -c ERROR logs/download_allfam_fill.log) errors remain"

FAMS=$($PY - <<'EOF'
import pandas as pd
d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
n = d.groupby('idnum').size().sort_values(ascending=False)
print(' '.join(str(i) for i in n.index if i != 23))
EOF
)
log "families (largest first): $FAMS"

one() {
  F=$1
  bash scripts/run_family_full.sh $F > logs/run_family${F}_full.log 2>&1
  free=$(df -BG --output=avail /home | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$MIN_FREE_GB" ] && [ "$F" != 23 ]; then
    rm -f data/waveforms/fam${F}.h5
    echo "[$(date '+%F %T')] fam $F done; free ${free} GB < ${MIN_FREE_GB}: removed data/waveforms/fam${F}.h5"
  else
    echo "[$(date '+%F %T')] fam $F done; free ${free} GB"
  fi
}
export -f one
echo $FAMS | tr ' ' '\n' | xargs -P $NPAR -I{} bash -c 'one {}'
log "all families done; building cross-family summary"
$PY scripts/summarize_families.py > logs/summarize_families.log 2>&1
log "finished"
