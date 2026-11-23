#!/usr/bin/env python3
"""Network-average dv/v for a family: combine per-channel dv/v (vs full-period reference,
demeaned per channel) over channels passing quality cuts, with split-half (odd/even) errors.

Channel selection: mean coda CC >= CC_MIN and noise-control scatter >= 5 x coda scatter.
Per bin: weighted mean over channels (w = cc^2), error = |odd - even| / 2 (split halves)
combined with the channel-scatter standard error.
Also shows direct-S constant dt (clock/timing control) network median per bin.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUAKES = {'M6.1': '2021-12-20T20:10', 'M6.4': '2022-12-20T10:34', 'M7.0': '2024-12-05T18:44'}
CC_MIN = 0.7


def load(F, tag):
    fs = f'data/dvv/fam{F}_summary{tag}.csv'
    if not os.path.exists(fs):
        return None, []
    sm = pd.read_csv(fs)
    good = sm[(sm.mean_cc_coda >= CC_MIN) & (sm.dvv_noise_std >= 5 * sm.dvv_ref_std)].chan.tolist()
    rows = []
    for c in good:
        f = f'data/dvv/fam{F}_{c}{tag}.csv'
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d['dvv'] = d.dvv_ref - d.dvv_ref.mean()
        d['chan'] = c
        rows.append(d)
    if not rows:
        return None, good
    d = pd.concat(rows)
    d['t'] = pd.to_datetime(d.t, format='mixed')
    d['w'] = d.cc_ref.clip(0, 1) ** 2
    g = d.groupby('bin').apply(lambda x: pd.Series(dict(
        t=x.t.iloc[0], dvv=np.average(x.dvv, weights=x.w), n=len(x),
        se=x.dvv.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
        dt_directS=np.median(x.dt_directS))), include_groups=False)
    return g, good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    a = ap.parse_args()
    F = a.family
    allb, good = load(F, '')
    odd, _ = load(F, '_odd')
    even, _ = load(F, '_even')
    b24, _ = load(F, '_2-4Hz')
    if allb is None:
        print('no channels pass quality cuts')
        return
    if odd is not None and even is not None:
        sh = (odd.dvv - even.dvv).abs() / 2
        allb['err'] = np.sqrt(sh.reindex(allb.index) ** 2 + allb.se ** 2)
    else:
        allb['err'] = allb.se
    allb.to_csv(f'data/dvv/fam{F}_network_dvv.csv')
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1]))
    ax[0].errorbar(allb.t, allb.dvv * 100, yerr=allb.err * 100, fmt='o-', color='k', ms=3, lw=0.8,
                   label=f'2-8 Hz, {len(good)} channels (split-half errors)')
    if b24 is not None:
        ax[0].plot(b24.t, b24.dvv * 100, 's--', color='C0', ms=3, lw=0.6, label='2-4 Hz')
    if odd is not None:
        ax[1].plot(odd.t, odd.dvv * 100, '.-', color='C1', lw=0.6, label='odd events')
        ax[1].plot(even.t, even.dvv * 100, '.-', color='C2', lw=0.6, label='even events')
    ax[2].plot(allb.t, allb.dt_directS * 1000, 'k.-', lw=0.6)
    ax[0].set_ylabel('dv/v (%)')
    ax[1].set_ylabel('dv/v (%)')
    ax[2].set_ylabel('direct-S dt (ms)\n(clock control)')
    for axx in ax:
        for k, q in QUAKES.items():
            axx.axvline(pd.Timestamp(q), color='r', ls='--', lw=0.8)
        axx.axhline(0, color='0.7', lw=0.5)
    for k, q in QUAKES.items():
        ax[0].text(pd.Timestamp(q), ax[0].get_ylim()[1], ' ' + k, color='r', fontsize=8, va='top')
    ax[0].legend(fontsize=8, loc='lower left')
    ax[1].legend(fontsize=8, loc='lower left')
    ax[0].set_title(f'Family {F}: network-average coda dv/v from isolated-LFE stacks (~200 events/bin); '
                    'positive = faster')
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_dvv_network.png', dpi=130)
    print('channels used:', good)
    print(allb[['t', 'dvv', 'err', 'n', 'dt_directS']].assign(dvv=lambda x: x.dvv * 100, err=lambda x: x.err * 100)
          .round(4).to_string())


if __name__ == '__main__':
    main()
