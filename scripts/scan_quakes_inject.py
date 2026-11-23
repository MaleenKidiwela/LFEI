#!/usr/bin/env python3
"""Build a synthetic-injection table for scan_quakes_mf.py --inject (detection-efficiency test).

Donors: per family, NDONOR catalog events from the pre-quake window (Q-30 d .. Q-2 d) with typical
strength (Shelly cc_div_thr_v2 between the family's 30th and 70th percentiles in that window;
no other catalog event within +-5 s).
For each donor, the raw continuous segment on every template channel (same demean/taper/resample
as the scan, NOT band-passed; includes the donor's own background noise) from min(w0)-3 s to
max(w0)+9 s around the catalog time, with 1-s cosine end tapers. Gap-free channels only.
Injection times: days Q-DB .. Q+DA, PER_DAY per family per day, families staggered by
7200/27 s so that injected events never overlap (+-20 s jitter, seeded).
Output: pickle {'table': DataFrame(day, fam, t_inj, donor), 'donors': {id: {'fam','t_cat','off0','segs'}}}
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

sys.path.insert(0, os.path.dirname(__file__))
from scan_matched_filter import SR, load_templates  # noqa: E402


def ep(x):
    return np.asarray((pd.DatetimeIndex(x) - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quake', required=True)
    ap.add_argument('--era', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--days-before', type=int, default=2)
    ap.add_argument('--days-after', type=int, default=3)
    ap.add_argument('--per-day', type=int, default=12)
    ap.add_argument('--ndonor', type=int, default=6)
    ap.add_argument('--cont', default='data/continuous')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--iso', type=float, default=5.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Q = pd.Timestamp(a.quake)
    c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv')
    c = c[c.source != 'v1_only']
    c['t'] = pd.to_datetime(c.time_utc, format='mixed')
    days_avail = {os.path.basename(p) for p in glob.glob(os.path.join(a.cont, '20*'))}
    donors, want = {}, []
    tpls = {}
    for F in range(1, 28):
        tpls[F] = load_templates(F, '3d', 10.0, f'data/templates/fam{F}_templates_era{a.era}.h5')
        w = c[(c.idnum == F) & (c.t >= Q - pd.Timedelta(days=30)) & (c.t < Q - pd.Timedelta(days=2))]
        w = w[w.t.dt.strftime('%Y-%m-%d').isin(days_avail)]
        # avoid events within ISO s of any other catalog event (cleaner donor segment)
        allt = np.sort(ep(c.t))   # unit-safe epoch seconds (pandas 3 datetime64[us])
        ts = ep(w.t)
        i = np.searchsorted(allt, ts)
        dprev = ts - allt[np.clip(i - 1, 0, len(allt) - 1)]
        dnext = allt[np.clip(i + 1, 0, len(allt) - 1)] - ts
        w = w[((dprev > a.iso) | (i == 0)) & (dnext > a.iso)]
        q40, q60 = w.cc_div_thr_v2.quantile([0.3, 0.7])
        cand = w[(w.cc_div_thr_v2 >= q40) & (w.cc_div_thr_v2 <= q60)]
        pick = cand.sample(min(a.ndonor * 4, len(cand)), random_state=a.seed + F)
        want += [(F, r.t) for r in pick.itertuples()]
    # read donor segments, grouped by day
    by_day = {}
    for F, t in want:
        by_day.setdefault(t.strftime('%Y-%m-%d'), []).append((F, t))
    nper = {F: 0 for F in range(1, 29)}
    for day, lst in sorted(by_day.items()):
        t0 = UTCDateTime(day)
        st_all = {}
        for f in glob.glob(os.path.join(a.cont, day, '*.mseed')):
            try:
                st = read(f)
            except Exception:
                continue
            st.merge(method=1, fill_value=0)
            for tr in st:
                tr.detrend('demean')
                tr.taper(max_percentage=None, max_length=5)
                if tr.stats.sampling_rate != SR:
                    fs = tr.stats.sampling_rate
                    tr.filter('lowpass', freq=0.4 * SR, corners=8, zerophase=True)
                    tr.resample(SR, no_filter=True) if fs % SR else tr.decimate(int(fs // SR), no_filter=True)
                st_all[tr.id] = tr
        for F, t in lst:
            if nper[F] >= a.ndonor:
                continue
            T = tpls[F]
            w0s = [w0 for _, w0 in T.values()]
            off0, off1 = min(w0s) - 3.0, max(w0s) + 9.0
            n = int(round((off1 - off0) * SR))
            tap = np.ones(n)
            m = int(SR)
            tap[:m] = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
            tap[-m:] = tap[:m][::-1]
            segs = {}
            for cid in T:
                tr = st_all.get(cid)
                if tr is None:
                    continue
                i0 = int(round((UTCDateTime(t.to_pydatetime()) + off0 - tr.stats.starttime) * SR))
                if i0 < 0 or i0 + n > tr.stats.npts:
                    continue
                x = tr.data[i0:i0 + n].astype(float)
                if np.any(x == 0):
                    continue
                segs[cid] = ((x - x.mean()) * tap).astype(np.float32)
            if len(segs) < 0.6 * len(T):
                continue
            did = f'{F}_{t:%Y%m%dT%H%M%S}'
            donors[did] = dict(fam=F, t_cat=str(t), off0=off0, segs=segs)
            nper[F] += 1
        print(day, sum(nper.values()), flush=True)
    print('candidates per family:', {F: sum(1 for f, _ in want if f == F) for F in range(1, 28)})
    rows = []
    days = pd.date_range(Q.normalize() - pd.Timedelta(days=a.days_before),
                         Q.normalize() + pd.Timedelta(days=a.days_after), freq='D')
    for d in days:
        for F in range(1, 28):
            ids = [k for k, v in donors.items() if v['fam'] == F]
            if not ids:
                continue
            for h in range(a.per_day):
                t = d + pd.Timedelta(seconds=h * 86400 / a.per_day + (F - 1) * 7200 / 27 + 60
                                     + rng.uniform(-20, 20))
                rows.append(dict(day=d.strftime('%Y-%m-%d'), fam=F, t_inj=str(t), donor=ids[h % len(ids)]))
    tab = pd.DataFrame(rows)
    with open(a.out, 'wb') as fh:
        pickle.dump(dict(table=tab, donors=donors, quake=str(Q)), fh)
    print(f'{len(donors)} donors, {len(tab)} injections over {len(days)} days -> {a.out}')
    print(pd.Series({F: nper[F] for F in range(1, 28)}).to_string())


if __name__ == '__main__':
    main()
