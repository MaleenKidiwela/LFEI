#!/usr/bin/env python3
"""'Breathing' LFE patches (Fable idea 6): true patch size tr(C_true) from split-station (A x B) relocations,
in windows of (a) time around the quakes, (b) M2 tidal phase, (c) season. Pooled over resolved families as
the ratio to each family's overall tr(C_true) (event-weighted), bootstrap over events.
C_true uses the window's own centroid, so it measures spread, not centroid shift.
Output data/patches/breathing.csv, figures/patch_breathing.png"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
Q = {'Petrolia': pd.Timestamp('2021-12-20T20:10'), 'Ferndale': pd.Timestamp('2022-12-20T10:34'), 'M7': pd.Timestamp('2024-12-05T18:44')}
M2_H = 12.4206012
import os
SDH = float(os.environ.get('SDH', 0.5))


def load(F):
    A = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfA.csv')
    B = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfB.csv')
    m = A.merge(B, on=['ev', 'time_utc'], suffixes=('_a', '_b'))
    # A x B is unbiased whatever the error size, so a looser cut only adds noise, not bias -> more events
    m = m[(m.sd_x_a < SDH) & (m.sd_y_a < SDH) & (m.sd_z_a < 2 * SDH) & (m.sd_x_b < SDH) & (m.sd_y_b < SDH) & (m.sd_z_b < 2 * SDH)]
    m = m.assign(t=pd.to_datetime(m.time_utc, format='mixed'), fam=F)
    return m


def trC(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return np.sum(a * b) / len(a)


def main():
    P = pd.read_csv('data/patches/patch_tensors.csv')
    fams = P[P.resolved == True].family.astype(int).tolist()  # noqa: E712
    D = pd.concat([load(F) for F in fams], ignore_index=True)
    tot = {F: trC(g[['dx_a', 'dy_a', 'dz_a']].values, g[['dx_b', 'dy_b', 'dz_b']].values) for F, g in D.groupby('fam')}
    hrs = (D.t - pd.Timestamp('2008-01-01')).dt.total_seconds() / 3600
    D['m2'] = ((hrs / M2_H) % 1) * 360
    D['doy'] = D.t.dt.dayofyear
    rows = []

    def ratio(sel, label, group):
        num, den, nn = 0.0, 0.0, 0
        parts = []
        for F, g in D[sel].groupby('fam'):
            if len(g) < 15:
                continue
            a, b = g[['dx_a', 'dy_a', 'dz_a']].values, g[['dx_b', 'dy_b', 'dz_b']].values
            parts.append((F, a, b))
            num += len(g) * trC(a, b) / tot[F]
            nn += len(g)
        if nn == 0:
            return
        bs = []
        for _ in range(200):
            s, w = 0.0, 0
            for F, a, b in parts:
                i = rng.integers(0, len(a), len(a))
                s += len(a) * trC(a[i], b[i]) / tot[F]
                w += len(a)
            bs.append(s / w)
        rows.append(dict(group=group, window=label, n=nn, ratio=num / nn, se=float(np.std(bs))))

    for k, q in Q.items():
        for lab, lo, hi in (('-365..0 d', -365, 0), ('0..30 d', 0, 30), ('30..365 d', 30, 365)):
            sel = (D.t >= q + pd.Timedelta(days=lo)) & (D.t < q + pd.Timedelta(days=hi))
            ratio(sel, f'{k} {lab}', 'quakes')
    for p0 in range(0, 360, 45):
        ratio((D.m2 >= p0) & (D.m2 < p0 + 45), f'M2 {p0}-{p0 + 45}', 'tide')
    for m0, lab in ((1, 'Jan-Mar'), (4, 'Apr-Jun'), (7, 'Jul-Sep'), (10, 'Oct-Dec')):
        ratio(D.t.dt.month.between(m0, m0 + 2), lab, 'season')
    R = pd.DataFrame(rows)
    R.to_csv(f'data/patches/breathing_sd{SDH}.csv', index=False)
    pd.set_option('display.width', 160)
    print(f'families {fams}; events {len(D)}')
    print(R.round(3).to_string(index=False))
    for g in ('tide', 'season'):
        r = R[R.group == g]
        chi = np.sum(((r.ratio - np.average(r.ratio, weights=1 / r.se ** 2)) / r.se) ** 2)
        from scipy.stats import chi2
        print(f'{g}: chi2 vs constant = {chi:.1f} for {len(r) - 1} dof, p={chi2.sf(chi, len(r) - 1):.3f}')
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a_, g in zip(ax, ('quakes', 'tide', 'season')):
        r = R[R.group == g]
        a_.errorbar(range(len(r)), r.ratio, yerr=r.se, fmt='o')
        a_.set_xticks(range(len(r)))
        a_.set_xticklabels(r.window, rotation=60, ha='right', fontsize=7)
        a_.axhline(1, color='k', lw=0.6)
        a_.set_ylabel('true patch size tr(C) / long-term')
        a_.set_title(f'Patch breathing: {g}')
    fig.tight_layout()
    fig.savefig(f'figures/patch_breathing_sd{SDH}.png', dpi=120)


if __name__ == '__main__':
    main()
