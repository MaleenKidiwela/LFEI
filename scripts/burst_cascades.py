#!/usr/bin/env python3
"""Do LFE bursts propagate across the 27-family array? (catalog only; Fable idea 1)

1. Family bursts: events of one family with gaps <= GAP (2 h) form a burst; onset = first event.
2. Network episodes: all family onsets clustered with gaps <= EP_GAP (6 h); keep episodes with >= MINFAM families.
3. Per episode: fit onset_k = t0 + s . x_k (x = family position in km, from our 3D locations, E/N),
   robust (Huber IRLS), giving a slowness vector s -> apparent speed and azimuth of propagation.
   Null: permute family positions within the episode (200x) -> p-value of the fit improvement.
4. Also: onset delay vs distance from the first family (linear = front, quadratic = diffusion).
Outputs data/cascades/episodes.csv, figures/burst_cascades.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GAP, EP_GAP, MINFAM, NPERM = 2 * 3600, 6 * 3600, 6, 200
QUAKES = {'Petrolia': '2021-12-20T20:10', 'Ferndale': '2022-12-20T10:34', 'M7': '2024-12-05T18:44'}


def fit(t, X, iters=5):
    A = np.c_[np.ones(len(t)), X]
    w = np.ones(len(t))
    for _ in range(iters):
        m, *_ = np.linalg.lstsq(A * w[:, None], t * w, rcond=None)
        r = t - A @ m
        s = 1.4826 * np.median(np.abs(r)) + 1.0
        w = np.sqrt(np.where(np.abs(r) / s <= 2, 1.0, 2 * s / np.abs(r)))
    r = t - A @ m
    return m, np.sum((w * r) ** 2), np.sum((w * (t - np.average(t, weights=w ** 2))) ** 2)


def main():
    os.makedirs('data/cascades', exist_ok=True)
    d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
    d['t'] = (pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) - pd.Timestamp('1970-01-01')).dt.total_seconds() + d.s_of_day
    lat0, lon0 = 40.06, -123.625
    pos = {}
    for F in range(1, 28):
        j = json.load(open(f'data/locations/fam{F}_3d.json'))
        pos[F] = ((j['lon'] - lon0) * 111.19 * np.cos(np.radians(lat0)), (j['lat'] - lat0) * 111.19)
    on = []
    for F, g in d.groupby('idnum'):
        t = np.sort(g.t.values)
        starts = np.r_[0, np.where(np.diff(t) > GAP)[0] + 1]
        ends = np.r_[starts[1:], len(t)]
        for a, b in zip(starts, ends):
            on.append((F, t[a], b - a))
    on = pd.DataFrame(on, columns=['fam', 'onset', 'n']).sort_values('onset').reset_index(drop=True)
    ep = np.r_[0, np.cumsum(np.diff(on.onset.values) > EP_GAP)]
    on['ep'] = ep
    rng = np.random.default_rng(0)
    rows = []
    for e, g in on.groupby('ep'):
        g = g.sort_values('onset').drop_duplicates('fam')       # first onset of each family in the episode
        if len(g) < MINFAM:
            continue
        X = np.array([pos[F] for F in g.fam])
        t = (g.onset.values - g.onset.min()) / 3600.0            # hours
        m, rss, tss = fit(t, X)
        r2 = 1 - rss / tss if tss > 0 else np.nan
        null = []
        for _ in range(NPERM):
            mp, rssp, tssp = fit(t, X[rng.permutation(len(X))], iters=2)
            null.append(1 - rssp / tssp if tssp > 0 else np.nan)
        p = (np.sum(np.array(null) >= r2) + 1) / (NPERM + 1)
        sx, sy = m[1], m[2]                                       # h/km
        smag = np.hypot(sx, sy)
        rows.append(dict(ep=e, t0=pd.to_datetime(g.onset.min(), unit='s'), n_fam=len(g), dur_h=t.max(),
                         n_events=int(g.n.sum()), sx_h_per_km=sx, sy_h_per_km=sy,
                         speed_km_per_day=24 / smag if smag > 0 else np.inf,
                         azimuth_deg=np.degrees(np.arctan2(sx, sy)) % 360, r2=r2, p_perm=p,
                         first_fam=int(g.fam.iloc[0])))
    E = pd.DataFrame(rows)
    E.to_csv('data/cascades/episodes.csv', index=False)
    sig = E[E.p_perm < 0.05]
    print(f'{len(on)} family bursts; {len(E)} network episodes with >= {MINFAM} families; '
          f'{len(sig)} ({len(sig) / max(len(E), 1):.0%}) with significant propagation (perm p<0.05; expected 5% by chance)')
    if len(sig):
        az = np.radians(sig.azimuth_deg)
        R = np.hypot(np.cos(az).mean(), np.sin(az).mean())
        mean_az = np.degrees(np.arctan2(np.sin(az).mean(), np.cos(az).mean())) % 360
        print(f'significant episodes: median speed {sig.speed_km_per_day.median():.1f} km/day '
              f'(IQR {sig.speed_km_per_day.quantile(.25):.1f}-{sig.speed_km_per_day.quantile(.75):.1f}); '
              f'mean azimuth {mean_az:.0f} deg, resultant R={R:.2f} (n={len(sig)}; random ~{1 / np.sqrt(len(sig)):.2f})')
        # axial (bidirectional) concentration
        R2 = np.hypot(np.cos(2 * az).mean(), np.sin(2 * az).mean())
        ax2 = (np.degrees(np.arctan2(np.sin(2 * az).mean(), np.cos(2 * az).mean())) / 2) % 180
        print(f'axial concentration R2={R2:.2f}, axis {ax2:.0f}/{ax2 + 180:.0f} deg (Shelly strike 293/113)')
        print('first family of significant episodes:', sig.first_fam.value_counts().head(8).to_dict())
        for k, q in QUAKES.items():
            tq = pd.Timestamp(q)
            pre = sig[(sig.t0 < tq) & (sig.t0 > tq - pd.Timedelta(days=730))]
            post = sig[(sig.t0 >= tq) & (sig.t0 < tq + pd.Timedelta(days=730))]
            print(f'  {k}: 2 yr before n={len(pre)} median speed {pre.speed_km_per_day.median():.1f} km/d, '
                  f'2 yr after n={len(post)} median speed {post.speed_km_per_day.median():.1f} km/d')
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].hist(E.p_perm, bins=20, color='0.5')
    ax[0].axhline(len(E) / 20, color='r', ls='--', label='uniform (no propagation)')
    ax[0].set_xlabel('permutation p-value per episode')
    ax[0].legend()
    if len(sig):
        a = plt.subplot(1, 3, 2, projection='polar')
        a.set_theta_zero_location('N')
        a.set_theta_direction(-1)
        a.hist(np.radians(sig.azimuth_deg), bins=24, color='C0')
        a.set_title('propagation azimuth (significant episodes)')
        ax[2].scatter(sig.t0, np.clip(sig.speed_km_per_day, 0, 200), s=6)
        ax[2].set_yscale('log')
        ax[2].set_ylabel('apparent speed (km/day)')
        for q in QUAKES.values():
            ax[2].axvline(pd.Timestamp(q), color='k', ls='--', lw=0.7)
    fig.tight_layout()
    fig.savefig('figures/burst_cascades.png', dpi=120)


if __name__ == '__main__':
    main()
