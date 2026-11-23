#!/usr/bin/env python3
"""CWIRE layout: one dv/v curve PER STATION-CHANNEL, pooling all LFE families
(as cwire/scripts/calculate.py + inversion.invert_and_output: one inversion per station/channel,
all clusters pooled). Uses the stack-pair measurements saved by dvv_cwire.py
(data/dvv/fam{F}_cwire_pairs.parquet + _stacks.csv).

Rows per channel: within-family stack pairs  m(t_b) - m(t_a);  stack vs that family's reference
m(t_b) - c_F (one offset per family). Same control nodes / breaks at mainshocks / 1st+2nd difference
damping / CC weights / Huber as dvv_cwire.py; lambda by leave-FAMILIES-out CV; parametric
trend+annual+steps+60-d transient fitted directly to the rows; block bootstrap over stacks.
Outputs: data/dvv/pool/{chan}_curve.csv, data/dvv/pool/params_by_channel.csv,
figures/dvv_pool_channels.png (curves), figures/dvv_pool_map.png (step & annual maps).
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dvv_cwire import (CC_MIN, PNAMES, QUAKES, build_G, epoch_s, nodes_for, param_G, reg_matrix,
                       solve, wls)


def load_all(min_ref_cc):
    P, T = [], {}
    for f in sorted(glob.glob('data/dvv/fam*_cwire_pairs.parquet')):
        F = int(re.search(r'fam(\d+)_cwire_pairs', f).group(1))
        st = f'data/dvv/fam{F}_cwire_stacks.csv'
        ch = f'data/dvv/fam{F}_cwire_channels.csv'
        if not (os.path.exists(st) and os.path.exists(ch)):
            continue
        s = pd.read_csv(st)
        T[F] = epoch_s(pd.to_datetime(s.t, format='mixed'))
        m = pd.read_csv(ch)
        good = set(m[m.median_cc_ref >= min_ref_cc].chan)
        p = pd.read_parquet(f)
        p = p[p.chan.isin(good)].assign(fam=F)
        P.append(p)
    return pd.concat(P, ignore_index=True), T


def invert_channel(p, T, nboot, rng):
    fams = sorted(p.fam.unique())
    fk = {F: k for k, F in enumerate(fams)}
    nk = len(fams)
    has_a = (p.a >= 0).values
    tb = np.array([T[F][b] for F, b in zip(p.fam, p.b)])
    ta = np.array([T[F][a] if a >= 0 else 0.0 for F, a in zip(p.fam, p.a)])
    gk = p.fam.map(fk).values
    d = p.eps.values
    cc = np.clip(p.cc.values, 0, 0.99)
    w0 = cc ** 2 / (1 - cc ** 2)
    w0 /= w0.mean()
    segs = nodes_for(tb.min() - 86400, tb.max() + 86400)
    G, nn = build_G(ta, tb, has_a, segs, gk, nk)
    R = reg_matrix(segs, nn, nk)
    # lambda: leave-families-out CV if >= 3 families, else leave-stack-blocks-out
    groups = gk if nk >= 3 else (p.b.values % 5)
    ug = np.unique(groups)
    fold = {g: i % min(5, len(ug)) for i, g in enumerate(rng.permutation(ug))}
    fo = np.array([fold[g] for g in groups])
    cv = []
    for lam in [1, 3, 10, 30, 100, 300, 1000]:
        num = den = 0.0
        for f in np.unique(fo):
            tr = fo != f
            x, _ = solve(G[tr], d[tr], w0[tr], R, lam, nn, iters=2)
            te = (~tr) & has_a
            if te.sum() == 0:
                continue
            r = d[te] - G[te][:, :nn] @ x[:nn]
            r = np.clip(r, -5 * r.std(), 5 * r.std())
            num += np.sum(w0[te] * r ** 2)
            den += np.sum(w0[te])
        cv.append((lam, num / max(den, 1e-12)))
    lam = min(cv, key=lambda x: x[1])[0]
    x, w = solve(G, d, w0, R, lam, nn)
    A = param_G(ta, tb, has_a, gk, nk)
    m = wls(A, d, w)
    # block bootstrap over (family, stack)
    sid_b = p.fam.values * 100000 + p.b.values
    sid_a = np.where(has_a, p.fam.values * 100000 + p.a.values, -1)
    ids = np.unique(sid_b)
    XB, MB = [], []
    for i in range(nboot):
        cnt = pd.Series(np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)), index=ids)
        mb = cnt.reindex(sid_b).values.astype(float)
        ma = np.where(sid_a >= 0, cnt.reindex(sid_a).fillna(0).values, 1.0)
        mult = mb * ma
        if i < 50:
            XB.append(solve(G, d, w0, R, lam, nn, iters=2, mult=mult)[0][:nn])
        MB.append(wls(A, d, w * mult)[:7])
    node_t = np.concatenate(segs)
    seg = np.concatenate([[s] * len(n) for s, n in enumerate(segs)])
    curve = pd.DataFrame(dict(t=pd.to_datetime(node_t, unit='s'), segment=seg, dvv_pct=x[:nn] * 100,
                              err_pct=np.std(XB, axis=0) * 100))
    se = np.std(MB, axis=0)
    return curve, m[:7], se, lam, nk, len(p)


def _job(args):
    c, p, T, nboot = args
    rng = np.random.default_rng(abs(hash(c)) % 2 ** 32)
    return c, invert_channel(p, T, nboot, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-ref-cc', type=float, default=0.4)
    ap.add_argument('--min-rows', type=int, default=3000)
    ap.add_argument('--nboot', type=int, default=100)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    os.makedirs('data/dvv/pool', exist_ok=True)
    P, T = load_all(a.min_ref_cc)
    P = P[P.cc >= max(CC_MIN - 0.3, 0.2)]
    print(f'{P.fam.nunique()} families, {P.chan.nunique()} channels, {len(P)} rows', flush=True)
    jobs = [(c, p.reset_index(drop=True), T, a.nboot) for c, p in P.groupby('chan') if len(p) >= a.min_rows]
    rows, curves = [], {}
    from multiprocessing import Pool
    with Pool(a.procs) as pool:
        results = pool.map(_job, jobs)
    for c, (curve, m, se, lam, nf, nr) in results:
        curve.to_csv(f'data/dvv/pool/{c}_curve.csv', index=False)
        curves[c] = curve
        r = dict(chan=c, n_fam=nf, n_rows=nr, lam=lam, annual_amp_pct=np.hypot(m[1], m[2]) * 100,
                 annual_peak_doy=((np.pi / 2 - np.arctan2(m[2], m[1])) % (2 * np.pi)) / (2 * np.pi) * 365.25)
        for k, n_ in enumerate(PNAMES):
            r[n_] = m[k] * 100
            r[n_ + '_se'] = se[k] * 100
        rows.append(r)
        print(f"{c:18s} fams {nf:2d} rows {nr:7d} lam {lam:5g}  annual {r['annual_amp_pct']:.3f}% d{r['annual_peak_doy']:.0f}  "
              f"steps pet {r['step_pet_pct']:+.3f}±{r['step_pet_pct_se']:.3f} fer {r['step_fer_pct']:+.3f}±{r['step_fer_pct_se']:.3f} "
              f"M7 {r['step_m7_pct']:+.3f}±{r['step_m7_pct_se']:.3f} tr {r['transient60_pct']:+.3f}±{r['transient60_pct_se']:.3f}",
              flush=True)
    out = pd.DataFrame(rows)
    st = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv').assign(sta=lambda x: x.net + '.' + x.sta) \
        .groupby('sta')[['lat', 'lon', 'dist_km']].first() if os.path.exists('data/stations/candidate_channels_65km_2008_2026.csv') else None
    if st is not None:
        out['sta'] = out.chan.str.split('.').str[:2].str.join('.')
        out = out.merge(st, left_on='sta', right_index=True, how='left')
    out.to_csv('data/dvv/pool/params_by_channel.csv', index=False)

    # curves figure (one panel per station, components overlaid)
    stas = sorted({c.rsplit('.', 2)[0] for c in curves}, key=lambda s: (s.split('.')[0] != 'PB', s))
    n = len(stas)
    fig, axs = plt.subplots(n, 1, figsize=(13, 1.6 * n + 1), sharex=True, squeeze=False)
    for ax, s in zip(axs[:, 0], stas):
        for c, cv in curves.items():
            if not c.startswith(s + '.'):
                continue
            for g in cv.segment.unique():
                q = cv[cv.segment == g]
                ax.plot(q.t, q.dvv_pct, lw=0.9, label=c.split('.')[-1] if g == 0 else None)
        for qt in QUAKES.values():
            ax.axvline(pd.Timestamp(qt), color='k', ls='--', lw=0.6)
        ax.set_ylabel(s, fontsize=8)
        ax.legend(fontsize=6, loc='upper left', ncol=3)
    axs[0, 0].set_title('dv/v (%) per station-channel, all LFE families pooled (CWIRE layout)')
    fig.tight_layout()
    fig.savefig('figures/dvv_pool_channels.png', dpi=110)
    if st is not None:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
        o = out.groupby('sta').agg(lat=('lat', 'first'), lon=('lon', 'first'), m7=('step_m7_pct', 'median'),
                                   fer=('step_fer_pct', 'median'), ann=('annual_amp_pct', 'median'))
        for k, (col, lab) in enumerate([('m7', 'M7 step (%)'), ('fer', 'Ferndale step (%)'), ('ann', 'annual amplitude (%)')]):
            v = o[col]
            lim = np.nanmax(np.abs(v)) if col != 'ann' else None
            sc = ax[k].scatter(o.lon, o.lat, c=v, cmap='RdBu' if col != 'ann' else 'viridis', s=90,
                               vmin=-lim if lim else None, vmax=lim, edgecolor='k')
            for sname, r in o.iterrows():
                ax[k].annotate(sname.split('.')[1], (r.lon, r.lat), fontsize=6, xytext=(3, 3), textcoords='offset points')
            ax[k].plot(-123.625, 40.06, 'r*', ms=14)
            plt.colorbar(sc, ax=ax[k], label=lab)
            ax[k].set_title(lab + ' per station (median over components)')
        fig.tight_layout()
        fig.savefig('figures/dvv_pool_map.png', dpi=120)


if __name__ == '__main__':
    main()
