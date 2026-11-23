#!/usr/bin/env python3
"""Which stations detected each repeat of a family?

For every event and every station (best channel per station: horizontals for S if present),
measure in the S window around the master S pick:
  cc    : max normalised CC with the station's master stack (lag <= 0.2 s)
  snr   : RMS(S window) / RMS(noise window -15..-5 s rel. catalog time)
  cc_null: same CC measured on a noise-only window (-14.5..-11.5 s rel. catalog time)
detected := cc > per-station 99th percentile of cc_null (1% false-alarm rate per station).
(A single LFE has S-window SNR ~1, so an amplitude-SNR criterion is not used; snr is stored.)
Outputs data/detections/fam{F}_station_detections.parquet (long: ev, time_utc, sta, cc, snr,
detected) and a figure of monthly detection fraction per station + stations-detected/event.
"""
import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

S_WIN, NOISE, MAXLAG = (-0.5, 2.5), (-15.0, -5.0), 0.2
NULL_START = -14.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--model', default='3d')
    ap.add_argument('--far', type=float, default=0.01, help='per-station false-alarm rate')
    a = ap.parse_args()
    F = a.family
    h = h5py.File(f'data/waveforms/fam{F}.h5', 'r')
    T = h5py.File(f'data/templates/fam{F}_templates.h5', 'r')
    picks = pd.read_csv(f'data/locations/fam{F}_{a.model}_picks.csv').set_index('sta')
    sr, pre = h.attrs['sr'], h.attrs['pre']
    sos = butter(4, [2.0, 8.0], btype='band', fs=sr, output='sos')
    t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    ml = int(MAXLAG * sr)
    ni = slice(int((NOISE[0] - pre) * sr), int((NOISE[1] - pre) * sr))

    by_sta = {}
    for cid in T:
        n, s, l, c = cid.split('.')
        by_sta.setdefault(f'{n}.{s}', []).append(cid)
    rows = []
    for sta, cids in sorted(by_sta.items()):
        if sta not in picks.index:
            continue
        S = picks.loc[sta, 'S_final'] if 'S_final' in picks.columns and np.isfinite(picks.loc[sta, 'S_final']) \
            else picks.loc[sta, 'S']
        if not np.isfinite(S):
            continue
        hor = [c for c in cids if c[-1] in 'EN12']
        use = hor if hor else cids
        ws, we = int((S + S_WIN[0] - pre) * sr), int((S + S_WIN[1] - pre) * sr)
        res = {}
        for cid in use:
            ev = h[f'chan/{cid}/ev'][:]
            D = sosfiltfilt(sos, h[f'chan/{cid}/data'][:].astype('float64'), axis=1)
            m = T[cid]['stack'][ws:we].astype(float)
            m = (m - m.mean()) / (np.linalg.norm(m - m.mean()) + 1e-20)
            def maxcc(a0, a1):
                b = np.full(len(ev), -1.0)
                for L in range(-ml, ml + 1):
                    seg = D[:, a0 + L:a1 + L]
                    seg = seg - seg.mean(1, keepdims=True)
                    b = np.maximum(b, seg @ m / (np.linalg.norm(seg, axis=1) + 1e-20))
                return b
            best = maxcc(ws, we)
            n0 = int((NULL_START - pre) * sr)
            null = maxcc(n0, n0 + (we - ws))
            snr = np.sqrt(np.mean(D[:, ws:we] ** 2, 1)) / (np.sqrt(np.mean(D[:, ni] ** 2, 1)) + 1e-20)
            for e, c, sn, cn in zip(ev, best, snr, null):
                if e not in res or c > res[e][0]:
                    res[e] = (c, sn, cn)      # best component per station
        for e, (c, sn, cn) in res.items():
            rows.append((int(e), sta, float(c), float(sn), float(cn)))
        print(f'{sta:10s} events {len(res):6d}', flush=True)
    d = pd.DataFrame(rows, columns=['ev', 'sta', 'cc', 'snr', 'cc_null'])
    d['time_utc'] = t_ev[d.ev].astype(str)
    thr = d.groupby('sta').cc_null.quantile(1 - a.far)
    d['cc_thr'] = d.sta.map(thr)
    d['detected'] = d.cc > d.cc_thr
    os.makedirs('data/detections', exist_ok=True)
    d.to_parquet(f'data/detections/fam{F}_station_detections.parquet')

    # summary + figure
    dist = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv').assign(
        sta=lambda x: x.net + '.' + x.sta).groupby('sta').dist_km.first()
    d['month'] = pd.to_datetime(d.time_utc, format='mixed').dt.to_period('M')
    frac = d.groupby(['sta', 'month']).detected.mean().unstack()
    order = sorted(frac.index, key=lambda s: dist.get(s, 999))
    frac = frac.loc[order]
    months = pd.period_range(d.month.min(), d.month.max(), freq='M')
    frac = frac.reindex(columns=months)
    nd = d.groupby('ev').detected.sum()
    nrec = d.groupby('ev').size()
    tt = t_ev[nd.index]
    fig, ax = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw=dict(height_ratios=[1, 2.4]))
    ax[0].scatter(tt, nd.values, s=1, c='k', alpha=0.25, label='stations detecting')
    m = pd.Series(nd.values, index=tt).resample('MS').mean()
    r = pd.Series(nrec.values, index=tt).resample('MS').mean()
    ax[0].plot(m.index, m.values, 'r', lw=1.2, label='monthly mean detecting')
    ax[0].plot(r.index, r.values, 'b', lw=1.0, label='monthly mean with data')
    ax[0].set_ylabel('stations / event')
    ax[0].legend(fontsize=8)
    ax[0].set_title(f'Family {F}: stations detecting each repeat (S-window CC above station null, FAR {a.far:.0%})')
    x0, x1 = mdates.date2num(months[0].to_timestamp()), mdates.date2num(months[-1].to_timestamp('M'))
    im = ax[1].imshow(frac.values, aspect='auto', interpolation='nearest', cmap='magma', vmin=0, vmax=1,
                      extent=[x0, x1, len(order) - 0.5, -0.5])
    ax[1].xaxis_date()
    ax[1].set_yticks(range(len(order)))
    ax[1].set_yticklabels([f'{s} ({dist.get(s, np.nan):.0f} km)' for s in order], fontsize=6)
    ax[1].set_title('fraction of repeats detected per station per month (blank = no data)')
    plt.colorbar(im, ax=ax[1], fraction=0.02, pad=0.01)
    ax[0].set_xlim(months[0].to_timestamp(), months[-1].to_timestamp('M'))
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_station_detections.png', dpi=130)
    s = d.groupby('sta').agg(n=('ev', 'size'), frac_detected=('detected', 'mean'),
                             median_cc=('cc', 'median'), cc_thr=('cc_thr', 'first'),
                             median_snr=('snr', 'median'))
    s['dist_km'] = [dist.get(x, np.nan) for x in s.index]
    s.sort_values('dist_km').round(3).to_csv(f'data/detections/fam{F}_station_summary.csv')
    print(s.sort_values('dist_km').round(2).to_string())
    print(f'median stations detecting per event: {nd.median():.0f} (of median {nrec.median():.0f} with data)')


if __name__ == '__main__':
    main()
