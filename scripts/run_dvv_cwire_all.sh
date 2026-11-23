#!/usr/bin/env bash
# CWIRE-style dv/v for every family whose pipeline is complete (report exists), 2 at a time:
# measurement (all channels) + inversions on all / borehole PB / surface BK+NC channels.
cd /home/seismic/LFE
export PY=/home/seismic/anaconda3/envs/lfe/bin/python
one() {
  F=$1; L=logs/fam${F}_cwire
  [ -e data/dvv/fam${F}_cwire_params.json ] && { echo "fam $F already done"; return; }
  $PY scripts/dvv_cwire.py --family $F --procs 6 --nboot 100 > ${L}.log 2>&1
  $PY scripts/dvv_cwire.py --family $F --reuse --nboot 100 --chan-regex '^PB' > ${L}_PB.log 2>&1
  $PY scripts/dvv_cwire.py --family $F --reuse --nboot 100 --chan-regex '^(BK|NC)' > ${L}_surf.log 2>&1
  echo "[$(date '+%F %T')] fam $F cwire done: $(grep -c 'CC ref' ${L}.log) channels measured"
}
export -f one
# wait for each family's pipeline report, then measure; order = completion order
todo="${*:-27 8 1 24 9 11 12 14 25 4 3 15 7 19 6 17 2 10 5 26 13 16 18 20 21 22}"
for F in $todo; do
  while [ ! -e data/reports/fam${F}_report.md ] || ! grep -q "all done" $(ls logs/run_family${F}_full*.log | xargs grep -L skipping | head -1) 2>/dev/null; do sleep 120; done
  echo $F
done | xargs -P 2 -I{} bash -c 'one {}'
echo "[$(date '+%F %T')] all cwire done"
