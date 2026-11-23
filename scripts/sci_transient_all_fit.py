#!/usr/bin/env python3
"""Per-family fitting stage of sci_transient_all.py (see that file's docstring).

For one family, groups surface (BK/NC...) and borehole (PB), windows early (b28_early, tau 3-8 s) and late
(b28_late, tau 8 s-end), using the same channel keep rule (early AND late reference CC >= 0.25), pair CC >= 0.2,
the dvv_cwire parametric model (trend, annual, steps at Petrolia/Ferndale/M7, 60-d M7 transient) with one offset
per channel for the reference rows and Huber IRLS (identical to sci_dvv_depth.GroupFit, checked):
  m7      : the 7-column model, IRLS                                    -> M7 transient (col 6), M7 step (col 5)
  pf      : 7 columns + 60-d transients after Petrolia and Ferndale (IRLS weights of m7 held fixed)
  placebo : 7 columns + step and 60-d transient at each common pre-M7 placebo date (fixed weights)
  stboot  : block bootstrap over stacks (formal SE, as in the earlier scripts)
  staboot : bootstrap over STATIONS with station counts drawn once for the union of all families' stations
            (shared draws across families -> pooled statistics keep the cross-family station sharing)
  stajk   : delete-one-station (union list) estimates
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import QUAKES, epoch_s, param_G  # noqa: E402

TAU = 60.0
DAY = 86400.0
GROUPS = {'surface': lambda c: ~c.str.startswith('PB.'), 'borehole': lambda c: c.str.startswith('PB.')}
CFG = {'early': 'b28_early', 'late': 'b28_late'}
MIN_CCREF, MIN_PAIRCC = 0.25, 0.2
T_PET, T_FER, T_M7 = (epoch_s([QUAKES[k]])[0] for k in ('pet', 'fer', 'm7'))


def pair_paths(F):
    if os.path.exists(f'data/sci/dvv_depth/fam{F}_pairs.parquet'):
        return f'data/sci/dvv_depth/fam{F}_pairs.parquet', f'data/sci/dvv_depth/fam{F}_channels.csv'
    return f'data/sci/transient_all/fam{F}_pairs.parquet', f'data/sci/transient_all/fam{F}_channels.csv'


def keep_channels(F):
    m = pd.read_csv(pair_paths(F)[1])
    if 'b28_late_ccref' not in m:
        return set()
    ok = (m.b28_early_ccref >= MIN_CCREF) & (m.b28_late_ccref >= MIN_CCREF)
    return set(m.chan[ok])


def sta_of(c):
    return c.str.split('.').str[:2].str.join('.')


def tcol(kind, t0, ta, tb, has_a):
    def f(t):
        dt = (t - t0) / DAY
        if kind == 'step':
            return (dt >= 0).astype(float)
        return np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / TAU), 0.0)
    fb = f(tb)
    fa = np.where(has_a, f(np.where(has_a, ta, tb)), 0.0)
    return fb - fa


class Fit:
    """Same model and IRLS as sci_dvv_depth.GroupFit; offsets eliminated by Schur complement (A22 diagonal)."""
    def __init__(self, p):
        self.has_a = (p.a >= 0).values
        self.ta, self.tb = np.nan_to_num(p.ta.values), p.tb.values
        self.k, self.keys = pd.factorize(p.chan)
        self.nk = len(self.keys)
        self.X7 = param_G(self.ta, self.tb, self.has_a, np.zeros(len(p), int), 1)[:, :7]
        self.d = p.eps.values
        cc = np.clip(p.cc.values, 0, 0.99)
        w0 = cc ** 2 / (1 - cc ** 2)
        self.w0 = w0 / w0.mean()
        self.ref = ~self.has_a
        self.sid_b = p.b.values
        self.sid_a = np.where(self.has_a, p.a.values, -1)
        self.sta = sta_of(p.chan).values
        self.w = self.w0.copy()
        self.m = self.solve(self.w0, self.X7, irls=3)

    def extra(self, spec):
        return np.c_[self.X7] if not spec else np.c_[self.X7, np.array([tcol(k, t0, self.ta, self.tb, self.has_a)
                                                                         for k, t0 in spec]).T]

    def solve(self, w, X, irls=0):
        d, k, ref, nk = self.d, self.k, self.ref, self.nk
        for it in range(irls + 1):
            p = X.shape[1]
            wr = np.where(ref, w, 0.0)
            A11 = (X * w[:, None]).T @ X
            A12 = np.array([-np.bincount(k, weights=wr * X[:, j], minlength=nk) for j in range(p)])
            A22 = np.bincount(k, weights=wr, minlength=nk) + 1e-12
            b1 = X.T @ (w * d)
            b2 = -np.bincount(k, weights=wr * d, minlength=nk)
            S = A11 - (A12 / A22) @ A12.T
            try:
                x = np.linalg.solve(S, b1 - A12 @ (b2 / A22))
            except np.linalg.LinAlgError:
                if not getattr(self, 'lstsq_fallback', False):
                    raise
                x = np.linalg.lstsq(S, b1 - A12 @ (b2 / A22), rcond=1e-10)[0]   # min-norm (collinear steps)
            if it < irls:
                off = (b2 - A12.T @ x) / A22                  # offsets c_k (reference rows get -c_k)
                r = d - X @ x + np.where(ref, off[k], 0.0)
                z = np.abs(r) * np.sqrt(self.w0) / (1.4826 * np.median(np.abs(r) * np.sqrt(self.w0)) + 1e-12)
                w = self.w0 * np.where(z <= 2.0, 1.0, 2.0 / np.maximum(z, 1e-12))
                self.w = w
        return x


def load_rows(F):
    pp, _ = pair_paths(F)
    keep = keep_channels(F)
    P = pd.read_parquet(pp, filters=[('cfg', 'in', list(CFG.values()))])
    P['cfg'] = P.cfg.astype(str)
    P = P[P.chan.isin(keep) & (P.cc >= MIN_PAIRCC)]
    T = epoch_s(pd.to_datetime(pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv').t, format='mixed'))
    P['tb'] = T[P.b.values]
    P['ta'] = np.where(P.a >= 0, T[np.clip(P.a.values, 0, None)], np.nan)
    return P.reset_index(drop=True), T


def family_job(args):
    F, dates, sta_list, sta_counts, nb_stack, seed = args
    rng = np.random.default_rng(seed)
    P, T = load_rows(F)
    sidx = {s: i for i, s in enumerate(sta_list)}
    out = {}
    for grp, gsel in GROUPS.items():
        for win, cfg in CFG.items():
            p = P[(P.cfg == cfg) & gsel(P.chan)].reset_index(drop=True)
            if p.chan.nunique() < 3:
                continue
            f = Fit(p)
            w = f.w
            X7 = f.X7
            Xpf = f.extra([('tr', T_PET), ('tr', T_FER)])
            r = dict(fam=F, group=grp, window=win, n_chan=f.nk, n_sta=len(set(f.sta)), n_rows=len(p),
                     stations=';'.join(sorted(set(f.sta))), m7=f.m.copy())
            r['pf'] = f.solve(w, Xpf)
            # number of stacks inside the first 60 d after each date (information content)
            Tg = T[np.unique(f.sid_b)]           # stacks actually present in this group's rows
            r['n_stacks_60d'] = {k: int(((Tg >= t0) & (Tg < t0 + 60 * DAY)).sum())
                                 for k, t0 in [('m7', T_M7), ('pet', T_PET), ('fer', T_FER)]}
            pl = np.full((len(dates), 2), np.nan)
            npl = np.zeros(len(dates), int)
            for i, t0 in enumerate(dates):
                npl[i] = int(((Tg >= t0) & (Tg < t0 + 60 * DAY)).sum())
                nbef = int(((Tg >= t0 - 365 * DAY) & (Tg < t0)).sum())
                if npl[i] < 1 or nbef < 2:     # placebo date outside this group's data coverage
                    continue
                try:
                    x = f.solve(w, f.extra([('step', t0), ('tr', t0)]))[7:9]
                    if np.all(np.abs(x) < 0.05):       # guard against (near-)singular placebo designs
                        pl[i] = x
                except np.linalg.LinAlgError:
                    pass
            r['placebo'] = pl                       # columns: step, transient
            r['placebo_nstk'] = npl
            # stack block bootstrap (within family)
            ids = np.unique(np.r_[f.sid_b, f.sid_a[f.sid_a >= 0]])
            ib = np.searchsorted(ids, f.sid_b)
            ia = np.searchsorted(ids, np.where(f.sid_a >= 0, f.sid_a, ids[0]))
            sb = []
            for _ in range(nb_stack):
                c = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
                try:
                    sb.append(f.solve(w * c[ib] * np.where(f.sid_a >= 0, c[ia], 1.0), X7)[5:7])
                except np.linalg.LinAlgError:
                    sb.append(np.full(2, np.nan))
            r['stboot'] = np.array(sb)
            # station bootstrap with globally shared draws
            js = np.array([sidx[s] for s in f.sta])
            bm7, bpf = np.full((len(sta_counts), 2), np.nan), np.full((len(sta_counts), 2), np.nan)
            for b, cnt in enumerate(sta_counts):
                mult = cnt[js].astype(float)
                if len(np.unique(f.k[mult > 0])) < 2:
                    continue
                try:
                    bm7[b] = f.solve(w * mult, X7)[5:7]
                    bpf[b] = f.solve(w * mult, Xpf)[7:9]
                except np.linalg.LinAlgError:
                    pass
            r['staboot_m7'], r['staboot_pf'] = bm7, bpf
            # delete-one-station (union list; NaN column where the station is absent -> estimate unchanged)
            jk = {}
            for s in sorted(set(f.sta)):
                m = f.sta != s
                if len(np.unique(f.k[m])) < 2:
                    continue
                try:
                    jk[s] = np.r_[f.solve(w * m, X7)[5:7], f.solve(w * m, Xpf)[7:9]]
                except np.linalg.LinAlgError:
                    pass
            r['stajk'] = jk
            out[(grp, win)] = r
    return F, out
