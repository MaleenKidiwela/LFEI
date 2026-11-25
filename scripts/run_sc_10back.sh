#!/bin/bash
# wait for the running SC plot stage, rerun dvv (adds 10-day backward stacks), then N-day back figures for 5 and 10
cd /home/seismic/LFE
PY="nice -n 10 /home/seismic/anaconda3/envs/lfe/bin/python"
while ps -eo args | awk '/[s]ci_noise_dvv_sc_full.py plot/ {f=1} END {exit !f}'; do sleep 30; done
echo "plot done $(date)"
$PY scripts/sci_noise_dvv_sc_full.py dvv --workers 4 --nboot 200 > logs/sci_sc_full_dvv_10back.log 2>&1 || { echo dvv failed; exit 1; }
echo "dvv done $(date)"
NBACK=5 $PY scripts/sci_noise_dvv_sc_5back_figs.py 2>&1 | grep -v Warn | tail -2
NBACK=10 $PY scripts/sci_noise_dvv_sc_5back_figs.py 2>&1 | grep -v Warn | tail -2
echo "all done $(date)"
