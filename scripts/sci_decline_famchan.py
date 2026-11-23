#!/usr/bin/env python3
"""Per-family x per-channel yearly dv/v from the ORIGINAL CWIRE pair measurements (helper of sci_decline.py).

Rows (as dvv_cwire_pool.py): data/dvv/fam{F}_cwire_pairs.parquet, channels with median_cc_ref >= 0.4, pair cc >= 0.2.
Model per (family, channel): m(t) = x_year(t) + annual sin/cos;  pair rows d = m(tb) - m(ta), reference rows
d = m(tb) (the reference offset is absorbed by the yearly levels; only differences of x are used).
Weights cc^2/(1-cc^2), Huber IRLS (k = 2 sigma_MAD).  Calendar-year bins 2008..2026 (quakes all fall in December, so
the 2021 bin is pre-Petrolia, the 2022 bin is post-Petrolia/pre-Ferndale, 2024 is pre-M7 for 11/12 of the year).
Output: data/sci/decline/famchan_yearly.csv (fam, chan, year, x_pct, n_stacks)
"""
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import epoch_s  # noqa: E402

OUT = 'data/sci/decline'
Y0 = epoch_s(['2008-01-01'])[0]
YRS = np.arange(2008, 2027)


def fit_one(p, tyear, tfrac):
    has_a = (p.a.values >= 0)
    b = p.b.values
    a = np.where(has_a, p.a.values, b)
    yb, ya = tyear[b], tyear[a]
    used = np.unique(np.r_[yb, ya[has_a]])
    k = {y: i for i, y in enumerate(used)}
    n = len(used)
    G = np.zeros((len(p), n + 2))
    G[np.arange(len(p)), [k[y] for y in yb]] += 1
    ia = np.where(has_a)[0]
    G[ia, [k[y] for y in ya[has_a]]] -= 1
    for j, fn in enumerate((np.sin, np.cos)):
        G[:, n + j] = fn(2 * np.pi * tfrac[b]) - np.where(has_a, fn(2 * np.pi * tfrac[a]), 0.0)
    d = p.eps.values
    cc = np.clip(p.cc.values, 0, 0.99)
    w0 = cc ** 2 / (1 - cc ** 2)
    w0 /= w0.mean()
    # gauge: if no reference rows connect the levels, x is defined up to a constant -> tiny ridge on x
    R = np.zeros((1, n + 2))
    R[0, :n] = 1.0 / n
    w = w0.copy()
    for _ in range(4):
        sw = np.sqrt(w)[:, None]
        A = np.vstack([G * sw, R * 1e-3])
        y = np.r_[d * sw[:, 0], 0.0]
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        r = d - G @ x
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (2 * s)
        w = w0 * np.where(u <= 1, 1.0, 1.0 / u)
    nst = pd.Series(b).groupby(yb).nunique()
    return pd.DataFrame(dict(year=used, x_pct=x[:n] * 100, n_stacks=nst.reindex(used).fillna(0).values.astype(int)))


def fam_job(F):
    p = pd.read_parquet(f'data/dvv/fam{F}_cwire_pairs.parquet')
    m = pd.read_csv(f'data/dvv/fam{F}_cwire_channels.csv')
    good = set(m[m.median_cc_ref >= 0.4].chan)
    p = p[p.chan.isin(good) & (p.cc >= 0.2)]
    s = pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv')
    t = pd.to_datetime(s.t, format='mixed')
    tyear = np.full(s['stack'].max() + 1, -1)
    tfrac = np.zeros(s['stack'].max() + 1)
    tyear[s['stack'].values] = t.dt.year.values
    tfrac[s['stack'].values] = (epoch_s(t) - Y0) / (365.25 * 86400)
    out = []
    for c, q in p.groupby('chan'):
        if len(q) < 300:
            continue
        r = fit_one(q.reset_index(drop=True), tyear, tfrac)
        out.append(r.assign(fam=F, chan=c))
    return pd.concat(out) if out else None


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fams = sorted(int(re.search(r'fam(\d+)_cwire', f).group(1)) for f in glob.glob('data/dvv/fam*_cwire_pairs.parquet'))
    with Pool(4) as pool:
        res = pool.map(fam_job, fams)
    Y = pd.concat([r for r in res if r is not None], ignore_index=True)
    Y[['fam', 'chan', 'year', 'x_pct', 'n_stacks']].to_csv(f'{OUT}/famchan_yearly.csv', index=False)
    print(len(Y), Y.fam.nunique(), Y.chan.nunique(), Y.groupby(['fam', 'chan']).ngroups)
