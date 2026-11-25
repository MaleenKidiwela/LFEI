#!/bin/bash
# Full receiver-function pipeline (after the first download pass). nice 10, <=4 workers.
cd /home/seismic/LFE
PY="nice -n 10 $HOME/anaconda3/envs/lfe/bin/python"
while kill -0 1073785 2>/dev/null; do sleep 60; done
echo "download pass 1 done $(date)"
$PY scripts/sci_rf_download.py --workers 4 > logs/sci_rf_download_pass2.log 2>&1
echo "download pass 2 done $(date)"
$PY scripts/sci_rf_compute.py --workers 4 > logs/sci_rf_compute.log 2>&1 && $PY scripts/sci_rf_compute.py --gather >> logs/sci_rf_compute.log 2>&1
echo "compute done $(date)"
$PY scripts/sci_rf_static.py --stations all > logs/sci_rf_static.log 2>&1
echo "static done $(date)"
$PY scripts/sci_rf_timelapse.py --stations all > logs/sci_rf_timelapse.log 2>&1
echo "timelapse done $(date)"
$PY scripts/sci_rf_power.py --stations all > logs/sci_rf_power.log 2>&1
echo "power done $(date)"
$PY scripts/sci_rf_model.py > logs/sci_rf_model.log 2>&1
$PY scripts/sci_rf_summary.py > logs/sci_rf_summary.log 2>&1
echo "all done $(date)"
