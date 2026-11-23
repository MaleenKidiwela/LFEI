#!/usr/bin/env bash
# Full single-family pipeline (after the waveform download for that family is complete).
# Usage: scripts/run_family.sh 23
set -uo pipefail   # no -e: a failed step must not skip the remaining steps (check logs)
F=${1:-23}
PY=/home/seismic/anaconda3/envs/lfe/bin/python
cd /home/seismic/LFE
mkdir -p logs
echo "[$(date)] family $F: availability + histogram"
$PY scripts/availability_and_histogram.py --family $F            > logs/fam${F}_hist.log 2>&1
# a shared one-pass build (scripts/build_all_families.py) may be producing fam$F.h5: wait for it
while [ -e data/locks/allfam_build ] && [ ! -e data/locks/allfam_build.finished ]; do sleep 60; done
if [ -e data/waveforms/fam${F}.h5.done ]; then
  echo "[$(date)] build dataset: reusing one-pass build ($(cat data/waveforms/fam${F}.h5.done))"
else
  echo "[$(date)] build dataset"
  $PY scripts/build_dataset.py --family $F --procs ${PROCS:-24}              > logs/fam${F}_build.log 2>&1
fi
echo "[$(date)] templates + yearly QC"
$PY scripts/templates.py --family $F                             > logs/fam${F}_templates.log 2>&1
for M in 3d 1d; do
  echo "[$(date)] location ($M, earthquake statics)"
  $PY scripts/locate_family.py --family $F --model $M --statics data/validation/station_residuals_$M.csv \
      > logs/fam${F}_loc_$M.log 2>&1
done
echo "[$(date)] per-station detections"
$PY scripts/station_detections.py --family $F                    > logs/fam${F}_stadet.log 2>&1
echo "[$(date)] relative relocation: bins + events (3d)"
$PY scripts/relocate_relative.py --family $F --mode bins   --model 3d > logs/fam${F}_reloc_bins.log 2>&1
$PY scripts/relocate_relative.py --family $F --mode events --model 3d > logs/fam${F}_reloc_events.log 2>&1
$PY scripts/plot_relocations.py --family $F --model 3d            > logs/fam${F}_reloc_plot.log 2>&1
echo "[$(date)] dv/v"
$PY scripts/dvv_family.py --family $F --nbin 200                 > logs/fam${F}_dvv.log 2>&1
for B in "2 4" "4 8"; do set -- $B
  $PY scripts/dvv_family.py --family $F --nbin 200 --fmin $1 --fmax $2 --tag _${1}-${2}Hz > logs/fam${F}_dvv_${1}-${2}Hz.log 2>&1
done
for H in odd even; do
  $PY scripts/dvv_family.py --family $F --nbin 200 --subset $H --tag _$H > logs/fam${F}_dvv_$H.log 2>&1
done
echo "[$(date)] done"
echo "[$(date)] fixed-station relocation check (stations running 2008-2026)"
CORE=PB.B045,PB.B046,PB.B047,PB.B049,PB.B932,PB.B933,PB.B934,PB.B935,NC.KMR,NC.KMPB,NC.KBN,NC.KFP,NC.KIP,NC.KCR,NC.KSM
$PY scripts/relocate_relative.py --family $F --mode bins --model 3d --stations $CORE --minsta 5 --tag _core \
    > logs/fam${F}_reloc_bins_core.log 2>&1
echo "[$(date)] all done"
$PY scripts/dvv_summary.py --family $F > logs/fam${F}_dvv_summary.log 2>&1
