#!/bin/bash
# After run_sci_rf_all.sh: PyRaysum modelling (rf env), RfPy cross-check, summary. nice 10, <=4 workers.
cd /home/seismic/LFE
L="nice -n 10 $HOME/anaconda3/envs/lfe/bin/python"
R="nice -n 10 $HOME/anaconda3/envs/rf/bin/python"
$L scripts/sci_rf_raysum_prep.py > logs/sci_rf_raysum.log 2>&1
$R scripts/sci_rf_model_raysum.py >> logs/sci_rf_raysum.log 2>&1
echo "raysum done $(date)"
$L scripts/sci_rf_rfpy_check.py --dump NC.KMPB > logs/sci_rf_rfpy_check.log 2>&1
$R scripts/sci_rf_rfpy_check.py --check NC.KMPB >> logs/sci_rf_rfpy_check.log 2>&1
$L scripts/sci_rf_summary.py > logs/sci_rf_summary.log 2>&1
echo "post done $(date)"
