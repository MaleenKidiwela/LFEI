#!/usr/bin/env python3
"""Event x station waveform availability for one family, plus a histogram of repeats
through time.

Outputs
  data/waveforms/fam{F}_availability.parquet  rows=events, cols=NET.STA (bool: a trace
                                               covers [t-5 s, t+40 s] on any component)
  figures/fam{F}_repeats_through_time.png      (a) detections per month by catalog source
                                               (b) number of stations with waveforms per event
                                               (c) station x month: fraction of events recorded
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from obspy import read

COVER = (-5.0, 40.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--wfdir', default=None)
    ap.add_argument('--include-v1-only', action='store_true',
                    help='also use v1-only detections (default: v2 catalog only)')
    ap.add_argument('--stations', default='data/stations/candidate_channels_65km_2008_2026.csv')
    a = ap.parse_args()
    F = a.family
    wfdir = a.wfdir or (f'data/waveforms/fam{F}_raw' if os.path.isdir(f'data/waveforms/fam{F}_raw') else 'data/waveforms/allfam_raw')

    cat = pd.read_csv(a.catalog)
    cat['t'] = pd.to_datetime(cat.time_utc)
    if not a.include_v1_only:          # project uses the v2 catalog only
        cat = cat[cat.source != 'v1_only']
    ev = cat[cat.idnum == F].sort_values('t').reset_index(drop=True)
    ev['day'] = ev.t.dt.normalize()

    # header spans of the shared all-family archive are cached (same for every family;
    # re-reading ~250k headers took ~16 min per family). Delete the cache if raw data change.
    cache = wfdir.rstrip('/') + '_spans.parquet' if wfdir.rstrip('/').endswith('allfam_raw') else None
    if cache and os.path.exists(cache):
        sp = pd.read_parquet(cache)
    else:
        spans = []   # (station, start, end) from mseed headers
        for d in sorted(glob.glob(os.path.join(wfdir, '*', '*'))):
            for f in glob.glob(os.path.join(d, '*.mseed')):
                try:
                    st = read(f, headonly=True)
                except Exception:
                    continue
                for tr in st:
                    spans.append((f'{tr.stats.network}.{tr.stats.station}',
                                  tr.stats.starttime.datetime, tr.stats.endtime.datetime))
        sp = pd.DataFrame(spans, columns=['sta', 'start', 'end'])
        if cache:
            sp.to_parquet(cache + f'.tmp{os.getpid()}')
            os.replace(cache + f'.tmp{os.getpid()}', cache)
    stations = sorted(sp.sta.unique())
    avail = pd.DataFrame(False, index=ev.index, columns=stations)
    t0 = (ev.t + pd.Timedelta(seconds=COVER[0])).values
    t1 = (ev.t + pd.Timedelta(seconds=COVER[1])).values
    for s, g in sp.groupby('sta'):
        g = g.sort_values('start')
        st_, en_ = g.start.values, g.end.values
        i = np.searchsorted(st_, t0, side='right') - 1
        ok = (i >= 0)
        ok[ok] = en_[i[ok]] >= t1[ok]
        avail[s] = ok
    avail.insert(0, 'time_utc', ev.time_utc.values)
    out = os.path.join('data/waveforms', f'fam{F}_availability.parquet')
    avail.to_parquet(out)

    # ---- figure
    dist = pd.read_csv(a.stations).assign(sta=lambda x: x.net + '.' + x.sta) \
        .groupby('sta').dist_km.first()
    order = sorted(stations, key=lambda s: dist.get(s, 999))
    nsta = avail[stations].sum(axis=1)
    month = ev.t.dt.to_period('M')
    fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=False,
                           gridspec_kw=dict(height_ratios=[1.2, 0.8, 2.2]))
    cnt = ev.groupby([month, 'source']).size().unstack(fill_value=0)
    cnt = cnt.reindex(pd.period_range(month.min(), month.max(), freq='M'), fill_value=0)
    x = cnt.index.to_timestamp()
    bottom = np.zeros(len(cnt))
    colors = {'both': '#4C72B0', 'v2_only': '#4C72B0', 'v1_only': '#DD8452'}
    labels = {'both': 'v2', 'v2_only': None, 'v1_only': 'v1 only'}
    for src in ['both', 'v2_only', 'v1_only']:
        if src in cnt:
            ax[0].bar(x, cnt[src], width=25, bottom=bottom, color=colors[src], label=labels[src], align='edge')
            bottom += cnt[src].values
    for q, lab in [('2021-12-20', 'M6.1 Petrolia'), ('2022-12-20', 'M6.4 Ferndale'),
                   ('2024-12-05', 'M7.0 Mendocino')]:
        for axx in ax[:2]:
            axx.axvline(pd.Timestamp(q), color='k', lw=0.8, ls='--')
        ax[0].text(pd.Timestamp(q), ax[0].get_ylim()[1] * 0.95, ' ' + lab, fontsize=8, va='top')
    ax[0].set_ylabel('detections / month')
    ax[0].set_title(f'Family {F}: {len(ev)} repeats, {ev.t.min():%Y-%m} to {ev.t.max():%Y-%m}')
    ax[0].legend(loc='upper left', fontsize=8)
    ax[1].scatter(ev.t, nsta, s=1, c='k', alpha=0.3)
    ax[1].set_ylabel('stations with\nwaveforms / event')
    frac = avail[order].groupby(month.to_numpy()).mean().T
    frac = frac.reindex(columns=cnt.index)
    im = ax[2].imshow(frac.values, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=1,
                      extent=[mdates.date2num(x[0]), mdates.date2num(x[-1] + pd.offsets.MonthEnd(1)), len(order) - 0.5, -0.5])
    ax[2].set_yticks(range(len(order)))
    ax[2].set_yticklabels([f'{s} ({dist.get(s, np.nan):.0f} km)' for s in order], fontsize=6)
    ax[2].xaxis_date()
    ax[2].set_title('fraction of events with waveforms (station sorted by distance)')
    plt.colorbar(im, ax=ax[2], fraction=0.02, pad=0.01)
    for axx in ax:
        axx.set_xlim(x[0], x[-1] + pd.offsets.MonthEnd(1))
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fn = f'figures/fam{F}_repeats_through_time.png'
    fig.savefig(fn, dpi=140)
    print(f'{len(ev)} events, {len(stations)} stations; median stations/event {nsta.median():.0f}; '
          f'events with >=5 stations: {(nsta >= 5).mean()*100:.0f}%')
    print('wrote', out, fn)


if __name__ == '__main__':
    main()
