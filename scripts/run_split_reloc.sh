#!/usr/bin/env bash
# Relocate single events twice with disjoint, azimuth-interleaved station halves (A/B).
# Independent errors -> cov(A,B) estimates the real source spread, var(A-B)/2 the error.
cd /home/seismic/LFE
PY=/home/seismic/anaconda3/envs/lfe/bin/python
for F in "$@"; do
  for H in A B; do
    S=$($PY -c "import json;d=json.load(open('data/relocations/station_split_AB.json'));print(d['$F'][0 if '$H'=='A' else 1])")
    $PY scripts/relocate_relative.py --family $F --mode events --model 3d --stations $S --minsta 4 --minobs 6 --tag _half$H \
      > logs/fam${F}_reloc_events_half$H.log 2>&1 &
  done
  wait
done
