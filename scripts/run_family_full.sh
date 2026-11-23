#!/usr/bin/env bash
# Complete per-family workflow = family-23 recipe:
#  run_family.sh (histogram, build, templates, 3D/1D location, station detections,
#  relocation bins/events, dv/v + core bins)  +  fine dv/v (iso 5 s, 150/bin) + fit + transient
#  + fixed-station event relocation + post-M7 silence (era templates, reverse-polarity scan,
#  coherence check, ideas 1/3/5 with control).
# Usage: PROCS=12 scripts/run_family_full.sh 8
set -uo pipefail
F=${1:?family}
# lock: two drivers may run concurrently (largest-first and smallest-first); skip if taken
mkdir -p /home/seismic/LFE/data/locks
mkdir /home/seismic/LFE/data/locks/fam$F 2>/dev/null || { echo "fam $F already taken by another driver; skipping"; exit 0; }
PROCS=${PROCS:-12}
PY=/home/seismic/anaconda3/envs/lfe/bin/python
cd /home/seismic/LFE
L=logs/fam${F}
step() { echo "[$(date '+%F %T')] fam $F: $*"; }

step "core pipeline (run_family.sh)"
PROCS=$PROCS bash scripts/run_family.sh $F || step "WARNING run_family.sh exited non-zero"

step "fine-bin dv/v (iso 5 s, 150 events/bin) + fit + transient"
$PY scripts/dvv_family.py --family $F --nbin 150 --iso 5 --tag _iso5_n150 > ${L}_dvv_iso5_n150.log 2>&1
for H in odd even; do
  $PY scripts/dvv_family.py --family $F --nbin 150 --iso 5 --subset $H --tag _iso5_n150_$H > ${L}_dvv_iso5_n150_$H.log 2>&1
done
$PY scripts/dvv_fit.py --family $F --tag _iso5_n150 > ${L}_dvv_fit.log 2>&1
$PY scripts/dvv_transient.py --family $F > ${L}_dvv_transient.log 2>&1
# joint all-channel inversion for one dv/v curve (+ same fit and transient test on it)
$PY scripts/dvv_invert.py --family $F --tag _iso5_n150 > ${L}_dvv_invert.log 2>&1
$PY scripts/dvv_transient.py --family $F --tag _iso5_n150_inv > ${L}_dvv_transient_inv.log 2>&1
$PY scripts/dvv_invert.py --family $F --tag _iso5_n150 --smooth auto --nboot 100 > ${L}_dvv_invert_smooth.log 2>&1

step "fixed-station relocation of individual events"
CORE=PB.B045,PB.B046,PB.B047,PB.B049,PB.B932,PB.B933,PB.B934,PB.B935,NC.KMR,NC.KMPB,NC.KBN,NC.KFP,NC.KIP,NC.KCR,NC.KSM
$PY scripts/relocate_relative.py --family $F --mode events --model 3d --stations $CORE --minsta 5 --tag _core \
    > ${L}_reloc_events_core.log 2>&1

step "era (pre-M7) templates"
# 2024-09-01 to the M7 (as family 23), extended back to include at least the last 300 events
TMIN=$($PY - <<EOF
import pandas as pd
c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv'); c = c[(c.source != 'v1_only') & (c.idnum == $F)]
t = pd.to_datetime(c.time_utc, format='mixed').sort_values()
t = t[t < pd.Timestamp('2024-12-05T18:44')]
print(min(pd.Timestamp('2024-09-01'), t.iloc[-300] if len(t) >= 300 else t.iloc[0]).strftime('%Y-%m-%d'))
EOF
)
step "era templates from $TMIN"
$PY scripts/templates.py --family $F --tmin $TMIN --tmax 2024-12-05T18:44 \
    --out data/templates/fam${F}_templates_era2024.h5 > ${L}_templates_era2024.log 2>&1

step "matched-filter scan 2024-11-10..2025-01-10 (normal + reversed)"
$PY scripts/scan_matched_filter.py --family $F --start 2024-11-10 --end 2025-01-10 \
    --templates data/templates/fam${F}_templates_era2024.h5 --procs $PROCS > ${L}_scan_silent.log 2>&1
$PY scripts/scan_postprocess.py --family $F > ${L}_scan_post.log 2>&1
$PY scripts/check_reversed.py --family $F > ${L}_coherence.log 2>&1

step "silent period ideas 1/3/5 + control"
$PY scripts/silent_period.py --family $F > ${L}_silent_period.log 2>&1
$PY scripts/silent_period.py --family $F --main 2024-08-19 --late-start 2024-10-20 --tag _control \
    > ${L}_silent_control.log 2>&1

step "report"
$PY scripts/family_report.py --family $F > ${L}_report.log 2>&1
step "all done"
