#!/bin/bash
# Wait for the Ducellier matched-filter scan to finish, then run validation, time-lapse and tides-through-time.
cd /home/seismic/LFE
PY="nice -n 10 /home/seismic/anaconda3/envs/lfe/bin/python -I"
while ps -eo args | awk '/[s]ci_ducellier_mf.py --daylist/ {f=1} END {exit !f}'; do sleep 300; done
echo "scan done $(date); days: $(ls /data/wsd02/maleen_data/ducellier/mf/scan/ | grep -c '^det_')"
$PY scripts/sci_ducellier_validate.py scan > logs/sci_ducellier_validate_final.log 2>&1; echo "validate $? $(date)"
OMP_NUM_THREADS=8 $PY scripts/sci_ducellier_tides_time.py > logs/sci_ducellier_tides_time_final.log 2>&1; echo "tides $? $(date)"
OMP_NUM_THREADS=8 $PY scripts/sci_ducellier_timelapse.py scan > logs/sci_ducellier_timelapse_final.log 2>&1; echo "timelapse $? $(date)"
echo "all done $(date)"
