"""Day list for the targeted 2012-2026 main-band matched-filter scan (+ 2004-2011 validation days).
 - PNSN-tremor main-band episodes with >= 300 tremor windows (94% of tremor), from 1 d before start to end
 - quake windows +/-20 d: 2014-03-10 M6.8, 2021-12-20 Petrolia, 2022-12-20 Ferndale, 2024-12-05 M7
 - inter-ETS sample: every 14th day otherwise
 - validation 2004-2011: 40 days inside Ducellier-catalog ETS episodes + 40 random outside (seeded)
Writes data/sci/ducellier/scan_daylist.txt (ordered: validation, quake windows, then chronological)."""
import numpy as np, pandas as pd
OUT = '/home/seismic/LFE/data/sci/ducellier'
rng = np.random.default_rng(7)
e = pd.read_csv(f'{OUT}/tremor_episodes.csv', parse_dates=['start', 'end'])
days = set()
qdays = set()
e = e[e.n_win >= 300]
for a, b in zip(e.start, e.end):
    for d in pd.date_range(a - pd.Timedelta(days=1), b):
        days.add(d)
for q in ['2014-03-10', '2021-12-20', '2022-12-20', '2024-12-05']:
    for d in pd.date_range(pd.Timestamp(q) - pd.Timedelta(days=20), pd.Timestamp(q) + pd.Timedelta(days=20)):
        qdays.add(d)
        days.add(d)
allp = pd.date_range('2012-01-01', '2026-11-20')
for i, d in enumerate(allp):
    if i % 14 == 0:
        days.add(d)
scan = sorted(d for d in days if pd.Timestamp('2012-01-01') <= d <= pd.Timestamp('2026-11-20'))
de = pd.read_csv(f'{OUT}/compare_ets_episodes.csv', parse_dates=['start', 'end'])
inep = sorted({d for a, b in zip(de.start, de.end) for d in pd.date_range(a, b - pd.Timedelta(days=1))})
ine = set(inep)
outs = [d for d in pd.date_range('2004-02-01', '2011-12-30') if d not in ine]
val = sorted(set(rng.choice(inep, 40, replace=False)) | set(rng.choice(outs, 40, replace=False)))
with open(f'{OUT}/scan_daylist.txt', 'w') as f:
    qd = sorted(d for d in qdays)
    rest = [d for d in scan if d not in qdays]
    for d in list(val) + qd + rest:
        f.write(pd.Timestamp(d).strftime('%Y-%m-%d') + '\n')
print('validation', len(val), 'scan', len(scan))
