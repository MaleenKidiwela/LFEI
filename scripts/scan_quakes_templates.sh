#!/usr/bin/env bash
# Era-matched templates for the quake scans (Petrolia 2021-12-20, Ferndale 2022-12-20), all 27 families.
# Same recipe as run_family_full.sh for the M7: events from ~quake-3 months to the quake, extended
# back to include at least the last 300 events. Output data/templates/fam{F}_templates_era{TAG}.h5
# Usage: scripts/scan_quakes_templates.sh petrolia|ferndale  (NPAR parallel families, default 8)
set -uo pipefail
Q=${1:?quake}
case $Q in
  petrolia) TQ=2021-12-20T20:10:20; T3=2021-09-20; TAG=petrolia ;;
  ferndale) TQ=2022-12-20T10:34:24; T3=2022-09-20; TAG=ferndale ;;
  *) echo unknown; exit 1 ;;
esac
NPAR=${NPAR:-8}
PY=/home/seismic/anaconda3/envs/lfe/bin/python
cd /home/seismic/LFE
one() {
  F=$1
  TMIN=$($PY - <<PYEOF
import pandas as pd
c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv'); c = c[(c.source != 'v1_only') & (c.idnum == $F)]
t = pd.to_datetime(c.time_utc, format='mixed').sort_values()
t = t[t < pd.Timestamp('$TQ')]
print(min(pd.Timestamp('$T3'), t.iloc[-300] if len(t) >= 300 else t.iloc[0]).strftime('%Y-%m-%d'))
PYEOF
)
  echo "[$(date '+%F %T')] fam $F era $TAG from $TMIN to $TQ"
  nice -n 10 $PY scripts/templates.py --family $F --tmin $TMIN --tmax $TQ \
      --out data/templates/fam${F}_templates_era${TAG}.h5 > logs/scan_quakes_fam${F}_templates_${TAG}.log 2>&1
  echo "[$(date '+%F %T')] fam $F done ($(grep -c SNR logs/scan_quakes_fam${F}_templates_${TAG}.log) channels)"
}
export -f one; export PY TQ T3 TAG
seq 1 27 | xargs -P $NPAR -I{} bash -c 'one {}'
