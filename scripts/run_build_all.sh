#!/usr/bin/env bash
# One-pass dataset build for the remaining families; run_family.sh waits on data/locks/allfam_build
# until data/locks/allfam_build.finished exists (then reuses fam{F}.h5 if fam{F}.h5.done exists,
# else falls back to build_dataset.py).
cd /home/seismic/LFE
touch data/locks/allfam_build
/home/seismic/anaconda3/envs/lfe/bin/python scripts/build_all_families.py --families "$@" --procs ${PROCS:-20} > logs/build_all_families.log 2>&1
touch data/locks/allfam_build.finished
