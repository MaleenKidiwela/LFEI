#!/usr/bin/env python3
"""Instrument-cleaned per-station-channel dv/v curves (CWIRE layout, all families pooled).

Same inversion as dvv_cwire_pool.py (invert_channel), with the instrument problems found by the
shear-wave-splitting analysis handled as in sci_dvv_instrument_check.py (variant V2):
  - rows whose stack time (or the paired stack's time) falls in an excluded window (EXCL) are dropped;
  - at instrument breaks (BREAKS2: B933, BRIC, RVIT, BJES reinstalls, KMPB maintenance) the channel is
    split into separate instrument segments, each inverted on its own (its own per-family offsets),
    so a level jump at a reinstall cannot leak into the curve; pairs spanning a break are dropped.
Channels without windows or breaks are unchanged and copied from data/dvv/pool/.
Outputs: data/dvv/pool_clean/{chan}_curve.csv (or {chan}__seg{k}_curve.csv for split channels),
data/dvv/pool_clean/params_by_channel.csv, figures/dvv_pool_channels_clean.png (affected channels:
original grey, cleaned colour).
"""
import argparse
import os
import shutil
import sys
from multiprocessing import get_context

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from dvv_cwire import CC_MIN, PNAMES, QUAKES, epoch_s  # noqa: E402
from dvv_cwire_pool import invert_channel, load_all  # noqa: E402
from sci_dvv_instrument_check import BREAKS2, EXCL, bad_mask, sta_of  # noqa: E402

OUT = 'data/dvv/pool_clean'


def affected(ch):
    return any(ch.startswith(k) for k in EXCL) or sta_of(ch) in BREAKS2


def split_rows(ch, p, T):
    tb = np.array([T[F][b] for F, b in zip(p.fam, p.b)])
    has_a = (p.a >= 0).values
    ta = np.array([T[F][a] if a >= 0 else np.nan for F, a in zip(p.fam, p.a)])
    bad = bad_mask(ch, tb) | (has_a & bad_mask(ch, np.nan_to_num(ta, nan=-1e18)))
    br = sorted(epoch_s([pd.Timestamp(x) for x in BREAKS2.get(sta_of(ch), [])]))
    sb = np.searchsorted(br, tb, side='right')
    sa = np.where(has_a, np.searchsorted(br, np.nan_to_num(ta, nan=-1e18), side='right'), sb)
    keep = ~bad & (sa == sb)
    q = p[keep].assign(_seg=sb[keep])
    return {k: g.drop(columns='_seg').reset_index(drop=True) for k, g in q.groupby('_seg')}


def _job(args):
    name, p, T, nboot = args
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    try:
        return name, invert_channel(p, T, nboot, rng)
    except Exception as e:                      # too few rows in a segment
        return name, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-rows', type=int, default=1500)
    ap.add_argument('--nboot', type=int, default=60)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P, T = load_all(0.4)
    P = P[P.cc >= max(CC_MIN - 0.3, 0.2)]
    orig = pd.read_csv('data/dvv/pool/params_by_channel.csv').set_index('chan')
    jobs, copied = [], []
    for ch, p in P.groupby('chan'):
        if ch not in orig.index:
            continue                                    # was below min-rows in the original run
        if not affected(ch):
            shutil.copy(f'data/dvv/pool/{ch}_curve.csv', f'{OUT}/{ch}_curve.csv')
            copied.append(ch)
            continue
        segs = split_rows(ch, p.reset_index(drop=True), T)
        multi = len(segs) > 1 or sta_of(ch) in BREAKS2
        for k, q in segs.items():
            if len(q) >= a.min_rows:
                jobs.append((f'{ch}__seg{k}' if multi else ch, q, T, a.nboot))
    print(f'{len(copied)} channels unchanged (copied); {len(jobs)} cleaned inversions', flush=True)
    with get_context('spawn').Pool(a.procs) as pool:
        res = pool.map(_job, jobs)
    rows = [orig.loc[c].to_dict() | dict(chan=c, cleaned=False) for c in copied]
    for name, r in res:
        if isinstance(r, str):
            print('skip', name, r)
            continue
        curve, m, se, lam, nf, nr = r
        curve.to_csv(f'{OUT}/{name}_curve.csv', index=False)
        d = dict(chan=name, cleaned=True, n_fam=nf, n_rows=nr, lam=lam, annual_amp_pct=np.hypot(m[1], m[2]) * 100,
                 annual_peak_doy=((np.pi / 2 - np.arctan2(m[2], m[1])) % (2 * np.pi)) / (2 * np.pi) * 365.25)
        for k, n_ in enumerate(PNAMES):
            d[n_] = m[k] * 100
            d[n_ + '_se'] = se[k] * 100
        rows.append(d)
        print(f'{name:24s} rows {nr:7d} M7 {d["step_m7_pct"]:+.3f}±{d["step_m7_pct_se"]:.3f} '
              f'fer {d["step_fer_pct"]:+.3f}±{d["step_fer_pct_se"]:.3f}', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/params_by_channel.csv', index=False)
    # figure: affected channels, original vs cleaned
    aff = sorted({n.split('__')[0] for n, r in res if not isinstance(r, str)})
    fig, axs = plt.subplots(len(aff), 1, figsize=(13, 1.7 * len(aff) + 1), sharex=True, squeeze=False)
    for ax, ch in zip(axs[:, 0], aff):
        o = pd.read_csv(f'data/dvv/pool/{ch}_curve.csv', parse_dates=['t'])
        for g in o.segment.unique():
            q = o[o.segment == g]
            ax.plot(q.t, q.dvv_pct, color='0.65', lw=1, label='original' if g == 0 else None)
        for n, r in res:
            if isinstance(r, str) or n.split('__')[0] != ch:
                continue
            c = r[0]
            for g in c.segment.unique():
                q = c[c.segment == g]
                ax.plot(q.t, q.dvv_pct, lw=1.2, label=(n.split('__')[1] if '__' in n else 'cleaned') if g == 0 else None)
        for k, ww in EXCL.items():
            if ch.startswith(k):
                for x0, x1 in ww:
                    ax.axvspan(max(pd.Timestamp(x0), pd.Timestamp('2008-01-01')), min(pd.Timestamp(x1), pd.Timestamp('2026-07-01')),
                               color='r', alpha=0.08)
        for qt in QUAKES.values():
            ax.axvline(pd.Timestamp(qt), color='k', ls='--', lw=0.6)
        ax.set_ylabel(ch, fontsize=7)
        ax.legend(fontsize=6, loc='upper left', ncol=4)
    axs[0, 0].set_title('Instrument-cleaned dv/v (%) per affected channel: grey = original, colour = cleaned '
                        '(separate instrument segments, each with its own level); red = excluded windows')
    fig.tight_layout()
    fig.savefig('figures/dvv_pool_channels_clean.png', dpi=110)
    print('done')


if __name__ == '__main__':
    main()
