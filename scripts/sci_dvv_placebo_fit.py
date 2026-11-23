#!/usr/bin/env python3
"""Fitting helpers for sci_dvv_placebo.py (placebo calibration of the main full-coda dv/v claims).

Rows: the ORIGINAL dvv_cwire full-coda pair measurements (data/dvv/fam{F}_cwire_pairs.parquet), channels with
median_cc_ref >= 0.4, pair cc >= 0.2 (as dvv_cwire_pool.py).  Model: dvv_cwire.param_G (trend, annual sin/cos,
steps at Petrolia / Ferndale / M7, 60-d M7 transient) + one reference offset per key (channel within a family
fit; family within a per-channel fit), Huber IRLS (sci_transient_all_fit.Fit, identical to sci_dvv_depth.GroupFit).
Per fit, with the IRLS weights held fixed:
  placebo_step    : + step at each placebo date                 -> compare Petrolia / Ferndale steps
  placebo_steptr  : + step + 60-d transient at each placebo date -> compare the M7 step (model has a transient)
  placebo_annual  : annual sin/cos replaced by sin/cos of a random period (0.7-0.9 or 1.1-1.4 yr)
  stboot          : block bootstrap over (family, stack) (formal SE, as dvv_cwire_pool.py)
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import epoch_s  # noqa: E402
from sci_transient_all_fit import DAY, Fit  # noqa: E402

Y0 = epoch_s(['2008-01-01'])[0]


def periodic(f, P):
    """sin/cos columns of period P (yr) for the rows of Fit f (pair rows: f(tb) - f(ta); reference rows: f(tb))."""
    out = []
    for fn in (np.sin, np.cos):
        g = lambda t: fn(2 * np.pi * (t - Y0) / (365.25 * DAY) / P)  # noqa: E731
        out.append(g(f.tb) - np.where(f.has_a, g(np.where(f.has_a, f.ta, f.tb)), 0.0))
    return np.array(out).T


def safe(f, w, X, idx):
    """Solve with fixed weights; NaN for singular or absurd (|x| >= 5 %) solutions."""
    bad = np.nan if not isinstance(idx, slice) else np.full(len(range(*idx.indices(X.shape[1]))), np.nan)
    try:
        x = f.solve(w, X)[idx]
    except np.linalg.LinAlgError:
        return bad
    return x if np.all(np.abs(x) < 0.05) else bad


def run_fit(p, dates, periods, nboot, seed, block_ids):
    """p: rows with chan(key), a, b, eps, cc, ta, tb; block_ids: (sid_b, sid_a) for the stack bootstrap."""
    rng = np.random.default_rng(seed)
    try:
        f = Fit(p)
    except np.linalg.LinAlgError:                    # e.g. record starts after Petrolia: pet/fer steps collinear
        Fit.lstsq_fallback = True
        f = Fit(p)
        Fit.lstsq_fallback = False
        f.lstsq_fallback = True
    w = f.w
    X7 = f.X7
    r = dict(m=f.m.copy(), n_key=f.nk, n_rows=len(p), collinear=bool(getattr(f, 'lstsq_fallback', False)))
    tg = np.unique(f.tb)
    ps, pst = np.full(len(dates), np.nan), np.full(len(dates), np.nan)
    for i, t0 in enumerate(dates):
        if ((tg >= t0) & (tg < t0 + 60 * DAY)).sum() < 1 or ((tg >= t0 - 365 * DAY) & (tg < t0)).sum() < 2 \
                or (tg >= t0).sum() < 2:
            continue
        ps[i] = safe(f, w, f.extra([('step', t0)]), 7)
        pst[i] = safe(f, w, f.extra([('step', t0), ('tr', t0)]), 7)
    r['placebo_step'], r['placebo_steptr'] = ps, pst
    pa = np.full((len(periods), 2), np.nan)
    for i, P in enumerate(periods):
        X = X7.copy()
        X[:, 1:3] = periodic(f, P)
        pa[i] = safe(f, w, X, slice(1, 3))
    r['placebo_annual'] = pa
    sb, sa = block_ids
    ids = np.unique(np.r_[sb, sa[sa >= 0]])
    ib = np.searchsorted(ids, sb)
    ia = np.searchsorted(ids, np.where(sa >= 0, sa, ids[0]))
    B = []
    for _ in range(nboot):
        c = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        B.append(safe(f, w * c[ib] * np.where(sa >= 0, c[ia], 1.0), X7, slice(0, 7)))
    r['stboot'] = np.array(B)
    r['t_first'], r['t_last'] = float(tg.min()), float(tg.max())
    return r


def stack_times(F):
    return epoch_s(pd.to_datetime(pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv').t, format='mixed'))


def load_family(F):
    m = pd.read_csv(f'data/dvv/fam{F}_cwire_channels.csv')
    good = set(m[m.median_cc_ref >= 0.4].chan)
    p = pd.read_parquet(f'data/dvv/fam{F}_cwire_pairs.parquet')
    p = p[p.chan.isin(good) & (p.cc >= 0.2)].copy()
    T = stack_times(F)
    p['tb'] = T[p.b.values]
    p['ta'] = np.where(p.a >= 0, T[np.clip(p.a.values, 0, None)], np.nan)
    p['fam'] = F
    return p.reset_index(drop=True)


def family_job(args):
    F, grp, dates, periods, nboot = args
    p = load_family(F)
    p = p[p.chan.str.startswith('PB.') == (grp == 'borehole')].reset_index(drop=True)
    if p.chan.nunique() < 3:
        return ('fam', F, grp), None
    sb = p.b.values
    sa = np.where(p.a.values >= 0, p.a.values, -1)
    r = run_fit(p, dates, periods, nboot, 100 * F + (grp == 'borehole'), (sb, sa))
    r['stations'] = ';'.join(sorted(set(p.chan.str.split('.').str[:2].str.join('.'))))
    return ('fam', F, grp), r


def channel_job(args):
    chan, p, dates, periods, nboot = args
    p = p.copy()
    sb = p.fam.values * 100000 + p.b.values
    sa = np.where(p.a.values >= 0, p.fam.values * 100000 + p.a.values, -1)
    p['chan'] = p.fam.astype(str)          # one reference offset per family (as dvv_cwire_pool.param_G)
    r = run_fit(p.reset_index(drop=True), dates, periods, nboot, zlib.crc32(chan.encode()), (sb, sa))
    return ('chan', chan), r
