#!/usr/bin/env python3
"""Unregularised MONTHLY dv/v per station-channel, all families pooled (helper of sci_decline.py).

Why: the pooled curves of dvv_cwire_pool.py are smoothed with roughness damping that is BROKEN at the three
mainshocks and has denser nodes after them, so noise can show up as steps at the real quake dates but is smoothed
away at placebo dates.  Here every calendar month is a free level (no smoothing, no breaks), so quake dates and
placebo dates are treated identically.

Rows as dvv_cwire_pool.py (data/dvv/fam{F}_cwire_pairs.parquet, channels with median_cc_ref >= 0.4, pair cc >= 0.2):
pair rows d = m[month(tb)] - m[month(ta)], reference rows d = m[month(tb)] - c_F (one offset per family).
Weights cc^2/(1-cc^2), Huber IRLS; formal SE from the weighted normal matrix (scaled by robust residual variance).
Output: data/sci/decline/monthly_bins.csv (chan, month, dvv_pct, se_pct, n_rows)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'          # 4 worker processes x 1 thread (shared machine)
import glob
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = 'data/sci/decline'
MONTHS = pd.date_range('2008-01-01', '2026-07-01', freq='MS')


def load():
    P = []
    for f in sorted(glob.glob('data/dvv/fam*_cwire_pairs.parquet')):
        F = int(re.search(r'fam(\d+)_cwire_pairs', f).group(1))
        m = pd.read_csv(f'data/dvv/fam{F}_cwire_channels.csv')
        s = pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv')
        t = pd.to_datetime(s.t, format='mixed')
        mi = np.full(s['stack'].max() + 1, -1)
        mi[s['stack'].values] = (t.dt.year.values - 2008) * 12 + t.dt.month.values - 1
        p = pd.read_parquet(f)
        p = p[p.chan.isin(set(m[m.median_cc_ref >= 0.4].chan)) & (p.cc >= 0.2)]
        has_a = p.a.values >= 0
        p = p.assign(fam=F, mb=mi[p.b.values], ma=np.where(has_a, mi[np.where(has_a, p.a.values, 0)], -1))
        P.append(p[['chan', 'fam', 'mb', 'ma', 'eps', 'cc']])
    return pd.concat(P, ignore_index=True)


def fit_channel(args):
    ch, p = args
    nm = len(MONTHS)
    fams = np.unique(p.fam.values)
    fk = np.searchsorted(fams, p.fam.values)
    n = len(p)
    has_a = p.ma.values >= 0
    rows = np.r_[np.arange(n), np.where(has_a)[0], np.where(~has_a)[0]]
    cols = np.r_[p.mb.values, p.ma.values[has_a], nm + fk[~has_a]]
    vals = np.r_[np.ones(n), -np.ones(has_a.sum()), -np.ones((~has_a).sum())]
    G = sp.csr_matrix((vals, (rows, cols)), shape=(n, nm + len(fams)))
    d = p.eps.values
    cc = np.clip(p.cc.values, 0, 0.99)
    w0 = cc ** 2 / (1 - cc ** 2)
    w0 /= w0.mean()
    w = w0.copy()
    used = np.zeros(nm, bool)
    used[np.unique(np.r_[p.mb.values, p.ma.values[has_a]])] = True
    ridge = np.diag(np.r_[np.where(used, 0.0, 1.0), np.zeros(len(fams))])
    u = np.r_[used.astype(float), np.zeros(len(fams))]
    ridge += np.outer(u, u) / used.sum()        # gauge: sum of used months = 0
    for _ in range(5):
        N = (G.T @ (sp.diags(w) @ G)).toarray() + ridge
        x = np.linalg.solve(N, G.T @ (w * d))
        r = d - G @ x
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        u = np.abs(r) / (2 * s)
        w = w0 * np.where(u <= 1, 1.0, 1.0 / u)
    s2 = np.sum(w * r ** 2) / max(n - used.sum() - len(fams), 1)
    # gauge: remove the mean of used months (constant shared with family offsets is not resolved)
    x0 = x[:nm] - x[:nm][used].mean()
    Ninv = np.linalg.inv(N)
    se = np.sqrt(np.clip(np.diag(Ninv)[:nm] * s2, 0, None))
    cnt = np.bincount(p.mb.values, minlength=nm)
    out = pd.DataFrame(dict(chan=ch, month=MONTHS[:nm], dvv_pct=np.where(used, x0 * 100, np.nan),
                            se_pct=np.where(used, se * 100, np.nan), n_rows=cnt[:nm]))
    return out


if __name__ == '__main__':
    P = load()
    P = P[(P.mb >= 0) & (P.mb < len(MONTHS))]
    jobs = [(c, q.reset_index(drop=True)) for c, q in P.groupby('chan') if len(q) >= 3000]
    with Pool(4) as pool:
        res = pool.map(fit_channel, jobs)
    M = pd.concat(res, ignore_index=True)
    M.to_csv(f'{OUT}/monthly_bins.csv', index=False)
    print(M.chan.nunique(), 'channels', M.dvv_pct.notna().sum(), 'channel-months')
