#!/bin/bash
# Finish the long-period inter-station noise dv/v after the download + correlation complete (nice 10).
cd /home/seismic/LFE
PY="nice -n 10 /home/seismic/anaconda3/envs/lfe/bin/python"
while pgrep -f "sci_noise_lp.py correlate" >/dev/null; do sleep 300; done
echo "correlate done $(date)"
$PY scripts/sci_noise_lp.py assemble > logs/sci_noise_lp_assemble.log 2>&1 || { echo "assemble failed"; exit 1; }
$PY scripts/sci_noise_lp_dvv.py pairs --workers 4 > logs/sci_noise_lp_dvv.log 2>&1 || { echo "dvv failed"; exit 1; }
$PY scripts/sci_noise_lp_net.py > logs/sci_noise_lp_net.log 2>&1 || { echo "net failed"; exit 1; }
$PY scripts/sci_noise_lp_figs.py > logs/sci_noise_lp_figs.log 2>&1 || { echo "figs failed"; exit 1; }
echo "all done $(date)"
