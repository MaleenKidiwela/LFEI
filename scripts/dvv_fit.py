#!/usr/bin/env python3
"""Network dv/v (fixed channel list from the main run) with split-half errors from the SAME
channels, then weighted LSQ fit: offset + linear trend + annual sinusoid + steps at the
Dec 2021 M6.1, Dec 2022 M6.4 and Dec 2024 M7.0. Bootstrap (over bins) step uncertainties.
Usage: dvv_fit.py --family 23 --tag _iso5_n150
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Q = {'M6.1 Petrolia': '2021-12-20T20:10', 'M6.4 Ferndale': '2022-12-20T10:34', 'M7.0 Mendocino': '2024-12-05T18:44'}
CC_MIN, NOISE_FAC = 0.7, 5.0


def network(F, tag, chans):
    rows = []
    for c in chans:
        f = f'data/dvv/fam{F}_{c}{tag}.csv'
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d['dvv'] = d.dvv_ref - d.dvv_ref.mean()
        d['w'] = d.cc_ref.clip(0, 1) ** 2
        d['chan'] = c
        rows.append(d)
    d = pd.concat(rows)
    d['t'] = pd.to_datetime(d.t, format='mixed')
    g = d.groupby('bin').apply(lambda x: pd.Series(dict(
        t=x.t.iloc[0], dvv=np.average(x.dvv, weights=x.w), n=len(x),
        se=x.dvv.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)), include_groups=False)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--tag', default='_iso5_n150')
    ap.add_argument('--min-chan', type=int, default=3)
    a = ap.parse_args()
    F, tag = a.family, a.tag
    sm = pd.read_csv(f'data/dvv/fam{F}_summary{tag}.csv')
    good = sm[(sm.mean_cc_coda >= CC_MIN) & (sm.dvv_noise_std >= NOISE_FAC * sm.dvv_ref_std)].chan.tolist()
    allb = network(F, tag, good)
    odd, even = network(F, tag + '_odd', good), network(F, tag + '_even', good)
    sh = (odd.dvv - even.dvv).abs().reindex(allb.index) / 2
    allb['err'] = np.sqrt(sh.fillna(0) ** 2 + allb.se.fillna(allb.se.median()) ** 2)
    allb.loc[sh.isna(), 'err'] = np.maximum(allb.loc[sh.isna(), 'err'], allb.err.median())
    allb = allb[allb.n >= a.min_chan].dropna(subset=['dvv']).sort_values('t')
    t = allb.t
    yr = (t - pd.Timestamp('2008-01-01')).dt.total_seconds() / (365.25 * 86400)
    G = np.array([np.ones(len(t)), yr - yr.mean(), np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)] +
                 [(t >= pd.Timestamp(v)).astype(float).values for v in Q.values()]).T
    d, w = allb.dvv.values, 1 / np.maximum(allb.err.values, 1e-5) ** 2
    W = np.sqrt(w)[:, None]
    m, *_ = np.linalg.lstsq(G * W, d * W[:, 0], rcond=None)
    r = d - G @ m
    chi2 = np.sum(w * r ** 2) / (len(d) - len(m))
    se = np.sqrt(np.diag(np.linalg.inv((G * W).T @ (G * W)) * max(chi2, 1)))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        mm, *_ = np.linalg.lstsq(G[i] * W[i], d[i] * W[i, 0], rcond=None)
        bs.append(mm)
    bs = np.array(bs)
    names = ['offset', 'trend per yr', 'annual sin', 'annual cos'] + [f'step {k}' for k in Q]
    print(f'channels: {len(good)}; bins: {len(d)} ({t.min():%Y-%m} .. {t.max():%Y-%m}); median err {np.median(allb.err)*100:.4f}%; reduced chi2 {chi2:.2f}')
    out = []
    for k, (n_, v, s) in enumerate(zip(names, m, se)):
        sb = bs[:, k].std()
        print(f'  {n_:22s} {v*100:+.4f} %  ± {s*100:.4f} (LSQ)  ± {sb*100:.4f} (bootstrap)   {v/max(s, sb):+.1f} sigma')
        out.append(dict(param=n_, value_pct=v * 100, se_lsq_pct=s * 100, se_boot_pct=sb * 100))
    amp = np.hypot(m[2], m[3])
    peak = ((np.pi / 2 - np.arctan2(m[3], m[2])) % (2 * np.pi)) / (2 * np.pi) * 365.25
    print(f'  annual amplitude {amp*100:.4f} %, fastest near day-of-year {peak:.0f}')
    pd.DataFrame(out).to_csv(f'data/dvv/fam{F}_dvv_fit{tag}.csv', index=False)
    allb.to_csv(f'data/dvv/fam{F}_network_dvv{tag}.csv')
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.errorbar(t, d * 100, yerr=allb.err * 100, fmt='o', ms=2.5, lw=0.6, color='k',
                label=f'network dv/v ({len(good)} channels)')
    tt = pd.date_range(t.min(), t.max(), freq='7D')
    yy = (tt - pd.Timestamp('2008-01-01')).total_seconds() / (365.25 * 86400)
    GG = np.array([np.ones(len(tt)), yy - yr.mean(), np.sin(2 * np.pi * yy), np.cos(2 * np.pi * yy)] +
                  [(tt >= pd.Timestamp(v)).astype(float) for v in Q.values()]).T
    ax.plot(tt, GG @ m * 100, 'r', lw=1, label='fit: trend + annual cycle + steps at mainshocks')
    for k, v in Q.items():
        ax.axvline(pd.Timestamp(v), color='C0', ls='--', lw=0.8)
        ax.text(pd.Timestamp(v), np.nanmax(d * 100), ' ' + k, fontsize=8, color='C0', va='top')
    ax.set_ylabel('dv/v (%)  [+ = faster]')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title(f'Family {F}: coda dv/v from isolated-LFE stacks (2-8 Hz)')
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_dvv_fit{tag}.png', dpi=130)


if __name__ == '__main__':
    main()
