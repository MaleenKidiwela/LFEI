#!/usr/bin/env python3
"""Per-channel coherence check (check_reversed.py style) of quake-scan detections.

For each detection: every template channel's normalised CC at the detection time (+-0.1 s lag),
using the detecting family's era templates. Groups (rule fixed8, deduplicated detections):
  ref_real      pre-quake positives matching catalog LFEs (random 40)
  ref_noise     pre-quake reversed false-alarm candidates (fa=True; random 40)
  pre_new       pre-quake positives not in the catalog (random 40)
  post_cat_2d   0-2 d after the quake, positives matching the catalog
  post_new_2d   0-2 d after the quake, positives not in the catalog (all, up to 150)
  post_fa_2d    0-2 d after the quake, reversed false-alarm candidates (all, up to 150)
Also flags NCSN earthquakes (within 1.4 deg) in the preceding 120 s / following 30 s.
Output data/scan_quakes/{tag}_coherence.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read, read_events
from obspy.clients.fdsn import Client
from scipy.signal import butter

sys.path.insert(0, os.path.dirname(__file__))
from check_reversed import chan_cc  # noqa: E402
from scan_matched_filter import SR, load_templates  # noqa: E402
from scan_quakes_analysis import QUAKES  # noqa: E402


def ncsn(tag, t0, t1):
    qf = f'data/scan_quakes/ncsn_{tag}.xml'
    if os.path.exists(qf):
        return read_events(qf)
    cat = Client('NCEDC', timeout=180).get_events(starttime=UTCDateTime(str(t0)), endtime=UTCDateTime(str(t1)),
                                                  latitude=40.05, longitude=-123.56, maxradius=1.4)
    cat.write(qf, format='QUAKEML')
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--rule', default='fixed8')
    a = ap.parse_args()
    base = a.tag.split('_')[0]
    Q = pd.Timestamp(QUAKES[base][0])
    era = QUAKES[base][1]
    D = pd.read_csv(f'data/scan_quakes/{a.tag}_detections.csv')
    D = D[D.rule == a.rule].copy()
    D['t'] = pd.to_datetime(D.t, format='mixed')
    pre, post = D.t < Q, (D.t >= Q) & (D.t < Q + pd.Timedelta(days=2))
    P, N = D.sign == 1, (D.sign == -1) & (D.fa.astype(str) == 'True')
    rs = 7
    def samp(x, n):
        return x.sample(min(n, len(x)), random_state=rs)
    groups = {'ref_real': samp(D[pre & P & D.catany.astype(bool)], 40),
              'ref_noise': samp(D[pre & N], 40),
              'pre_new': samp(D[pre & P & ~D.catany.astype(bool)], 40),
              'post_cat_2d': D[post & P & D.catany.astype(bool)],
              'post_new_2d': samp(D[post & P & ~D.catany.astype(bool)], 150),
              'post_fa_2d': samp(D[post & N], 150)}
    allc = pd.concat([g.assign(grp=n) for n, g in groups.items()]).sort_values('t')
    sos = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
    tpl_cache = {}
    rows = []
    for r in allc.itertuples():
        F = int(r.fam)
        if F not in tpl_cache:
            tpl_cache[F] = load_templates(F, '3d', 10.0, f'data/templates/fam{F}_templates_era{era}.h5')
        tpls = tpl_cache[F]
        tc = UTCDateTime(str(r.time_utc).replace('Z', ''))
        st = Stream()
        for day in sorted({(tc + dt).strftime('%Y-%m-%d') for dt in (-60, 90)}):
            pat = os.path.join('data/continuous', day, '*.mseed')
            if glob.glob(pat):
                st += read(pat, starttime=tc - 60, endtime=tc + 90)
        st.merge(method=1, fill_value=0)
        for tr in st:
            if tr.id in tpls and tr.stats.sampling_rate != SR:
                tr.filter('lowpass', freq=0.4 * SR, corners=8, zerophase=True)
                tr.resample(SR, no_filter=True)
        cc = chan_cc(st, tpls, str(r.time_utc).replace('Z', ''), sos)
        v = np.array(list(cc.values()))
        if len(v) == 0:
            continue
        rows.append(dict(group=r.grp, fam=F, time_utc=r.time_utc, sign=r.sign, ccsum=r.ccsum, meancc=r.meancc,
                         n=len(v), frac_same_sign=float(np.mean(np.sign(v) == r.sign)),
                         median_abs_cc=float(np.median(np.abs(v))), frac_strong=float(np.mean(r.sign * v > 0.2))))
    res = pd.DataFrame(rows)
    try:
        cat = ncsn(base, Q - pd.Timedelta(days=35), Q + pd.Timedelta(days=52))
        eq = np.array([(e.preferred_origin() or e.origins[0]).time.timestamp for e in cat])
        mags = np.array([e.magnitudes[0].mag if e.magnitudes else np.nan for e in cat])

        def eqnear(t):
            ts = UTCDateTime(str(t).replace('Z', '')).timestamp
            k = np.where((eq > ts - 120) & (eq < ts + 30))[0]
            return ';'.join(f'M{mags[i]:.1f}@{eq[i] - ts:+.0f}s' for i in k)
        res['eq_nearby'] = res.time_utc.map(eqnear)
        print(f'NCSN events: {len(cat)}')
    except Exception as e:
        print('earthquake check failed:', repr(e)[:200])
    res.to_csv(f'data/scan_quakes/{a.tag}_coherence.csv', index=False)
    print(res.groupby('group')[['frac_same_sign', 'median_abs_cc', 'frac_strong']].describe()
          .loc[:, (slice(None), ['count', 'mean', '50%'])].round(3).to_string())


if __name__ == '__main__':
    main()
