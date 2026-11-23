#!/usr/bin/env python3
"""Fixed channel set for a quake scan: template channels (union over the 27 families' era templates)
with data on >= MINFRAC of the days in BOTH the pre-quake and post-quake halves of the window
(a day counts if the channel has >= 80% of the day). Headers only are read.
Output data/scan_quakes/chanset_{tag}.txt (one channel id per line) + availability table.
"""
import argparse
import glob
import os

import pandas as pd
from obspy import read

QUAKES = {'petrolia': ('2021-12-20T20:10:20', 'petrolia'),
          'ferndale': ('2022-12-20T10:34:24', 'ferndale'),
          'm7': ('2024-12-05T18:44:21', '2024')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--minfrac', type=float, default=0.9)
    a = ap.parse_args()
    import h5py
    Q = pd.Timestamp(QUAKES[a.tag][0])
    era = QUAKES[a.tag][1]
    tcids = set()
    for F in range(1, 28):
        with h5py.File(f'data/templates/fam{F}_templates_era{era}.h5', 'r') as T:
            tcids |= set(T)
    rows = []
    for d in pd.date_range(a.start, a.end, freq='D'):
        day = d.strftime('%Y-%m-%d')
        for f in glob.glob(os.path.join('data/continuous', day, '*.mseed')):
            try:
                st = read(f, headonly=True)
            except Exception:
                continue
            cov = {}
            for tr in st:
                cov[tr.id] = cov.get(tr.id, 0) + tr.stats.npts / tr.stats.sampling_rate
            for cid, s in cov.items():
                if cid in tcids:
                    rows.append(dict(day=day, cid=cid, ok=s >= 0.8 * 86400))
    A = pd.DataFrame(rows)
    A['post'] = pd.to_datetime(A.day) >= Q.normalize()
    ndays = {False: len(pd.date_range(a.start, Q.normalize() - pd.Timedelta(days=1))),
             True: len(pd.date_range(Q.normalize(), a.end))}
    av = A[A.ok].groupby(['cid', 'post']).size().unstack(fill_value=0)
    for k in (False, True):
        if k not in av:
            av[k] = 0
        av[k] = av[k] / ndays[k]
    av.columns = ['pre' if not c else 'post' for c in av.columns]
    keep = av[(av.pre >= a.minfrac) & (av.post >= a.minfrac)].index.tolist()
    av.to_csv(f'data/scan_quakes/chanset_{a.tag}_availability.csv')
    with open(f'data/scan_quakes/chanset_{a.tag}.txt', 'w') as fh:
        fh.write('\n'.join(sorted(keep)) + '\n')
    print(f'{len(tcids)} template channels; {len(av)} with data; kept {len(keep)}')
    print(av[~av.index.isin(keep)].round(2).to_string())


if __name__ == '__main__':
    main()
