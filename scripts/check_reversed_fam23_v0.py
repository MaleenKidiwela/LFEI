#!/usr/bin/env python3
"""Per-channel coherence check of scan detections (e.g. reversed-polarity candidates).

For each detection time, compute every template channel's normalised CC at the detection
(catalog-time convention, lag window +-0.1 s): a real event (normal or reversed) shows a
coherent sign across most channels (frac of channels with CC of the detection's sign >> 0.5,
median |CC| well above noise), noise shows ~50/50 signs. Also flags NCSN earthquakes
(any magnitude, within 150 km) in the preceding 120 s / following 30 s.
Compares candidate groups against references: catalog-matched normal detections (real LFEs)
and pre-M7 isolated reversed detections (noise baseline).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(__file__))
from scan_matched_filter import SR, load_templates  # noqa: E402


def chan_cc(st_day, templates, t_det, sos):
    out = {}
    for tr in st_day:
        if tr.id not in templates:
            continue
        tpl, w0 = templates[tr.id]
        n = len(tpl)
        t0 = UTCDateTime(t_det) + w0
        seg = tr.slice(t0 - 2.0, t0 + n / SR + 2.0)
        if seg.stats.npts < n + int(2 * SR):
            continue
        x = sosfiltfilt(sos, seg.data.astype(float))
        i0 = int(round((t0 - seg.stats.starttime) * SR))
        best = 0.0
        for L in range(-int(0.1 * SR), int(0.1 * SR) + 1):
            w = x[i0 + L:i0 + L + n]
            if len(w) < n:
                continue
            w = w - w.mean()
            c = float(w @ tpl / (np.linalg.norm(w) + 1e-20))
            if abs(c) > abs(best):
                best = c
        out[tr.id] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--templates', default='data/templates/fam23_templates_era2024.h5')
    ap.add_argument('--scan', default='data/scan/fam23_scan_all.csv')
    a = ap.parse_args()
    tpls = load_templates(a.family, '3d', 10.0, a.templates)
    d = pd.read_csv(a.scan)
    d['t'] = pd.to_datetime(d.time_utc.str.replace('Z', ''), format='mixed')
    M7, END = pd.Timestamp('2024-12-05T18:44'), pd.Timestamp('2024-12-28')
    groups = {
        'silence_reversed_isolated': d[(d.t >= M7) & (d.t < END) & (d.sign == -1) & ~d.cat23 & ~d.catoth],
        'silence_normal_new': d[(d.t >= M7) & (d.t < END) & (d.sign == 1) & ~d.cat23 & ~d.catoth],
        'ref_real_LFE (catalog, normal)': d[(d.t < M7) & (d.sign == 1) & d.cat23].nlargest(25, 'cc_div_thr'),
        'ref_noise (pre-M7 reversed isolated)': d[(d.t < M7) & (d.sign == -1) & ~d.cat23 & ~d.catoth],
        'ref_new_normal (pre-M7, not in catalog)': d[(d.t < M7) & (d.sign == 1) & ~d.cat23 & ~d.catoth],
    }
    sos = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
    cache = {}
    rows = []
    for gname, g in groups.items():
        for r in g.itertuples():
            day = r.t.strftime('%Y-%m-%d')
            if day not in cache:
                st = read(os.path.join('data/continuous', day, '*.mseed'))
                st.merge(method=1, fill_value=0)
                st = st.select(id='*')
                for tr in st:
                    if tr.id in tpls and tr.stats.sampling_rate != SR:
                        tr.filter('lowpass', freq=0.4 * SR, corners=8, zerophase=True)
                        tr.resample(SR, no_filter=True)
                cache = {day: st}
            cc = chan_cc(cache[day], tpls, r.time_utc.replace('Z', ''), sos)
            v = np.array(list(cc.values()))
            if len(v) == 0:
                continue
            rows.append(dict(group=gname, time_utc=r.time_utc, sign=r.sign, cc_div_thr=r.cc_div_thr,
                             n=len(v), frac_same_sign=float(np.mean(np.sign(v) == r.sign)),
                             median_abs_cc=float(np.median(np.abs(v))),
                             frac_strong=float(np.mean(r.sign * v > 0.2))))
    res = pd.DataFrame(rows)
    # nearby earthquakes (NCSN, within 150 km of family) for candidate times
    try:
        cl = Client('NCEDC', timeout=120)
        cat = cl.get_events(starttime=UTCDateTime('2024-11-10'), endtime=UTCDateTime('2025-01-11'),
                            latitude=40.05, longitude=-123.56, maxradius=1.4)
        eq = np.array([(e.preferred_origin() or e.origins[0]).time.timestamp for e in cat])
        mags = [e.magnitudes[0].mag if e.magnitudes else np.nan for e in cat]
        def eqnear(t):
            ts = UTCDateTime(t.replace('Z', '')).timestamp
            k = np.where((eq > ts - 120) & (eq < ts + 30))[0]
            return ';'.join(f'M{mags[i]:.1f}@{eq[i] - ts:+.0f}s' for i in k)
        res['eq_nearby'] = res.time_utc.map(eqnear)
        print(f'NCSN events within 1.4 deg, Nov 10 - Jan 10: {len(cat)}')
    except Exception as e:
        print('earthquake check failed:', repr(e)[:100])
    res.to_csv('data/scan/fam23_coherence_check.csv', index=False)
    print(res.groupby('group')[['frac_same_sign', 'median_abs_cc', 'frac_strong']].describe()
          .loc[:, (slice(None), ['count', 'mean', '50%'])].round(3).to_string())
    print('\nsilence candidates:')
    print(res[res.group.str.startswith('silence')].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
