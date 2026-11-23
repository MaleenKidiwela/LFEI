#!/usr/bin/env python3
"""Combine a family's daily matched-filter scan files and flag catalog matches.

Output data/scan/fam{F}_scan_all.csv with columns of the daily files plus
  t, ts          detection time (datetime, epoch s)
  catfam         a v2 catalog event of THIS family within +-2 s
  catoth         a v2 catalog event of any OTHER family within +-2 s
and data/scan/fam{F}_silence_window.json: the family's post-M7 silence window
  start = M7 origin time; end = time of the family's 10th v2 event after the M7
  (start of sustained activity), capped at the scan end. Family 23 used a fixed
  Dec 5 - Dec 28 window (10th event after the M7: see the json written for it).
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

M7 = pd.Timestamp('2024-12-05T18:44')


def ep(x):
    return np.asarray((pd.DatetimeIndex(x) - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1))


def near(ts, ref, tol=2.0):
    ref = np.sort(ref)
    if len(ref) == 0:
        return np.zeros(len(ts), bool)
    i = np.clip(np.searchsorted(ref, ts), 1, max(len(ref) - 1, 1))
    return np.minimum(abs(ts - ref[i - 1]), abs(ts - ref[i])) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--scan-dir', default='data/scan')
    ap.add_argument('--start', default='2024-11-10')
    ap.add_argument('--end', default='2025-01-11')
    ap.add_argument('--nrestart', type=int, default=10)
    a = ap.parse_args()
    F = a.family
    fs = [f for f in sorted(glob.glob(f'{a.scan_dir}/fam{F}_20*.csv')) if open(f).read().strip()]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d['t'] = pd.to_datetime(d.time_utc.str.replace('Z', ''), format='mixed')
    d['ts'] = ep(d.t)
    c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv')
    c = c[c.source != 'v1_only']
    c['t'] = pd.to_datetime(c.time_utc, format='mixed')
    c['ts'] = ep(c.t)
    w = c[(c.t >= a.start) & (c.t < a.end)]
    d['catfam'] = near(d.ts.values, w[w.idnum == F].ts.values)
    d['catoth'] = near(d.ts.values, w[w.idnum != F].ts.values)
    d.to_csv(f'{a.scan_dir}/fam{F}_scan_all.csv', index=False)
    after = c[(c.idnum == F) & (c.t >= M7)].sort_values('t')
    end = pd.Timestamp(a.end)
    if len(after) >= a.nrestart:
        end = min(end, after.t.iloc[a.nrestart - 1])
    first = after.t.iloc[0] if len(after) else None
    info = dict(family=F, start=str(M7), end=str(end), first_after=str(first),
                gap_days=(first - M7).total_seconds() / 86400 if first is not None else None,
                silence_days=(end - M7).total_seconds() / 86400)
    json.dump(info, open(f'{a.scan_dir}/fam{F}_silence_window.json', 'w'), indent=1)
    print(info)
    for name, (t0, t1) in {'pre': (pd.Timestamp(a.start), M7), 'silence': (M7, end),
                           'post': (end, pd.Timestamp(a.end))}.items():
        x = d[(d.t >= t0) & (d.t < t1)]
        days = (t1 - t0).total_seconds() / 86400
        pos, neg = x[x.sign == 1], x[x.sign == -1]
        print(f'{name:8s} {days:5.1f} d  +{len(pos)} (catalog {pos.catfam.sum()}, other fam {(pos.catoth & ~pos.catfam).sum()}, '
              f'new {(~pos.catfam & ~pos.catoth).sum()})  -{len(neg)} (isolated {(~neg.catfam & ~neg.catoth).sum()}, '
              f'{(~neg.catfam & ~neg.catoth).sum() / days:.2f}/day)')
    cat_pre = w[(w.idnum == F) & (w.t < M7)]
    rec = d[(d.t < M7) & (d.sign == 1)].catfam.sum() / max(len(cat_pre), 1)
    print(f'catalog recovery pre-M7: {rec:.2f} ({len(cat_pre)} catalog events)')


if __name__ == '__main__':
    main()
