#!/bin/bash
# Wait for the SC full dv/v stage, then run powreg and plot (nice 10).
cd /home/seismic/LFE
PY="nice -n 10 /home/seismic/anaconda3/envs/lfe/bin/python"
while ps -eo args | awk '/[s]ci_noise_dvv_sc_full.py dvv/ {f=1} END {exit !f}'; do sleep 300; done
echo "dvv done $(date)"
$PY scripts/sci_noise_dvv_sc_full.py powreg > logs/sci_sc_full_powreg.log 2>&1 || { echo "powreg failed"; exit 1; }
$PY scripts/sci_noise_dvv_sc_full.py plot > logs/sci_sc_full_plot.log 2>&1 || { echo "plot failed"; exit 1; }
echo "all done $(date)"
