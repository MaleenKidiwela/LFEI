#!/usr/bin/env python3
"""Joint 3-D dv/v: coda / noise observables refitted with ONE common time model.

Time model (all data): annual sin/cos + Petrolia/Ferndale/M7 steps + 60-d windows after each quake + ramp from
2021-06 (sci_dvv_source_region.tcols).  Quantities: m7 / pet / fer = permanent steps (after the 60-d window),
D = mean 2025 - mean 2021 of steps + ramp (sci_dvv_source_region.dvec).

SC  single-station noise (sc_10day_back.csv, 5 bands): 10-day backward values dated at the window centre
    (d - 4.5 d); instrument windows removed and a free offset per instrument segment (breaks) as in
    sci_dvv_3d_data.py; optional log10 microseism-power regressor (PB.B045 EHZ, 10-day mean; variant 'pow').
    Errors: moving-block bootstrap of residuals (60-d blocks, 200 reps) and SEASON-MATCHED placebos = the whole
    fit on pre-2021-06 data with all model dates shifted back by whole years 5..12; honest = max(boot, placebo
    RMS); stations without >= 4 placebos: boot x band-median (placebo / boot) ratio.  The placebo values are
    simultaneous across stations (same shifted dates) and are stored for the network covariance.
LP  long-period inter-station: comp-averaged 10-d bin series per pair (QC as sci_noise_lp_net), averaged per
    band x distance bin (<40, 40-80, >=80 km), same model/errors.
LFE coda per station: stage-1 family x channel coda fits of sci_dvv_deep_shallow (stage1.pkl, 'full' window;
    model sin cos pet fer m7 60-d-exp ramp, D via its decline()), honest per family-channel SE = max(bootstrap,
    placebo RMS), IVW mean per station with SE inflated by sqrt(max(1, chi2r)) of the family-channel scatter.
Outputs data/sci/dvv_3d/coda_obs.csv (+ sc_placebo.npz).
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_3d'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_source_region as SR_  # noqa: E402
from sci_dvv_3d_data import load_series, BREAKS, BANDS  # noqa: E402

NJ = len(SR_.PN)
SHIFTS = [5, 6, 7, 8, 9, 10, 11, 12]
QS = ['m7', 'pet', 'fer', 'D']
DAY = 86400.0
rng = np.random.default_rng(3)


def qvec(q, s=0.0):
    v = np.zeros(NJ)
    for j in range(NJ):
        e = np.zeros(NJ)
        e[j] = 1
        v[j] = SR_.QUANT[q](e, s) - SR_.QUANT[q](np.zeros(NJ), s)
    return v


def epoch(ts):
    return (pd.DatetimeIndex(ts) - pd.Timestamp('1970-01-01')).total_seconds().values


def usable_cols(t, s):
    """columns identifiable: each step needs >= 30 values within 1 yr on both sides (same segment ignored)."""
    use = np.ones(NJ, bool)
    sh = s * SR_.YR
    for j, k in [(2, 'pet'), (3, 'fer'), (4, 'm7')]:
        tq = SR_.T_Q[k] - sh
        ok = ((t < tq) & (t > tq - SR_.YR)).sum() >= 30 and ((t >= tq + 60 * DAY) & (t < tq + SR_.YR)).sum() >= 30
        use[j] = ok
        use[j + 3] = ok and ((t >= tq) & (t < tq + 60 * DAY)).sum() >= 10
    use[8] = (t < SR_.T_RAMP - sh).sum() >= 60 and (t > SR_.T_RAMP - sh + SR_.YR).sum() >= 60
    return use


def design(t, seg, s=0.0, extra=None):
    F = SR_.tcols(t, s)
    use = usable_cols(t, s)
    segs = np.unique(seg)
    A = [F[:, use]] + [(seg == g).astype(float)[:, None] for g in segs]
    if extra is not None:
        A.append(extra[:, None])
    return np.hstack(A), use


def ols(A, y):
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return x


def fit_series(t, y, seg, extra=None, nboot=200, block_days=60):
    ok = np.isfinite(y) & (np.isfinite(extra) if extra is not None else True)
    t, y, seg = t[ok], y[ok], seg[ok]
    ex = extra[ok] if extra is not None else None
    if len(t) < 200:
        return None
    A, use = design(t, seg, 0.0, ex)
    x = ols(A, y)
    th = np.full(NJ, np.nan)
    th[use] = x[:use.sum()]
    res = y - A @ x
    # moving-block bootstrap of residuals (time-ordered)
    order = np.argsort(t)
    r = res[order]
    nblk = max(1, int(block_days / max(np.median(np.diff(t[order])) / DAY, 1e-3)))
    B = []
    for b in range(nboot):
        starts = rng.integers(0, max(1, len(r) - nblk), int(np.ceil(len(r) / nblk)) + 1)
        rb = np.concatenate([r[s_:s_ + nblk] for s_ in starts])[:len(r)]
        yb = (A @ x)[order] + rb
        xb = ols(A[order], yb)
        tb = np.full(NJ, np.nan)
        tb[use] = xb[:use.sum()]
        B.append(tb)
    B = np.array(B)
    P = {}
    pre = t < SR_.T_RAMP
    for s in SHIFTS:
        if pre.sum() < 200:
            break
        Ap, up = design(t[pre], seg[pre], s, ex[pre] if ex is not None else None)
        if not up[4] and not up[2]:
            continue
        xp = ols(Ap, y[pre])
        tp = np.full(NJ, np.nan)
        tp[up] = xp[:up.sum()]
        P[s] = tp
    return th, B, P


def summarize(th, B, P):
    rows = {}
    for q in QS:
        v = qvec(q)
        need = v != 0
        if not np.all(np.isfinite(th[need])):
            continue
        est = float(v[need] @ th[need])
        bs = np.array([v[need] @ b[need] for b in B])
        bs = bs[np.isfinite(bs)]
        pv = []
        for s, tp in P.items():
            vs = qvec(q, s)
            nd = vs != 0
            if np.all(np.isfinite(tp[nd])):
                pv.append(float(vs[nd] @ tp[nd]))
        rows[q] = dict(est=est, se_boot=float(np.std(bs)) if len(bs) > 10 else np.nan,
                       plac=dict(zip([s for s, tp in P.items()], pv)) if pv else {}, n_plac=len(pv),
                       se_plac=float(np.sqrt(np.mean(np.square(pv)))) if len(pv) >= 4 else np.nan)
    return rows


def power_series():
    p = pd.read_csv(f'{ROOT}/data/sci/noise_dvv_sc_full/power_B045_EHZ.csv')
    s = pd.DataFrame(np.log10(p.drop(columns=['day']).values), index=pd.to_datetime(p['day']),
                     columns=[c for c in p.columns if c != 'day'])
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='D'))
    return s.rolling(10, min_periods=5).mean()


def stage_sc(pow_=False):
    S = load_series()
    pw = power_series() if pow_ else None
    rows, plac = [], {}
    for sta, Y in S.items():
        tt = Y.index - pd.Timedelta(hours=108)
        t = epoch(tt)
        br = [pd.Timestamp(b) for b in BREAKS.get(sta, [])]
        seg = np.searchsorted(np.array(br, dtype='datetime64[ns]'), Y.index.values) if br else np.zeros(len(Y), int)
        for b in BANDS:
            pc = b.replace('Hz', '') if pw is not None and b.replace('Hz', '') in pw.columns else '0.1-1'
            ex = pw[pc].reindex(Y.index).values if pw is not None else None
            r = fit_series(t, Y[b].values, seg, ex)
            if r is None:
                continue
            sm = summarize(*r)
            for q, o in sm.items():
                rows.append(dict(dataset='SC', sta=sta, band=b, quantity=q, est=o['est'], se_boot=o['se_boot'],
                                 se_plac=o['se_plac'], n_plac=o['n_plac']))
                plac[(sta, b, q)] = o['plac']
    T = pd.DataFrame(rows)
    T = honest(T, ['band'])
    T.to_csv(f'{OUT}/coda_obs_sc{"_pow" if pow_ else ""}.csv', index=False, float_format='%.5g')
    pickle.dump(plac, open(f'{OUT}/sc_placebo{"_pow" if pow_ else ""}.pkl', 'wb'))
    print(T.groupby(['quantity', 'band']).agg(n=('est', 'size'), med=('est', 'median'), se=('se', 'median'),
                                              ratio=('ratio', 'first'), nz3=('z', lambda z: int((z.abs() > 3).sum()))).round(4))


def honest(T, by):
    T = T.copy()
    for key, g in T.groupby(['quantity'] + by):
        good = g[g.n_plac >= 4]
        ratio = float(np.nanmedian(good.se_plac / good.se_boot)) if len(good) >= 3 else 1.5
        T.loc[g.index, 'ratio'] = max(ratio, 1.0)
        for i in g.index:
            r = T.loc[i]
            T.loc[i, 'se'] = np.nanmax([r.se_boot, r.se_plac]) if r.n_plac >= 4 else r.se_boot * max(ratio, 1.0)
    T['z'] = T.est / T.se
    return T


def stage_lp():
    P = pd.read_csv(f'{ROOT}/data/sci/noise_lp/pairs.csv')
    sys.path.insert(0, f'{ROOT}/scripts')
    from sci_dvv_3d_obs import site_excl_mask, DAY0
    series = {}
    for fn in sorted(glob.glob('/data/wsd02/maleen_data/MTJ_LH/dvv/*.npz')):
        name = os.path.basename(fn)[:-4]
        a, b = name.split('_')
        pr = P[(P.a == a) & (P.b == b)].iloc[0]
        Z = np.load(fn)
        for band in ['0.05-0.1', '0.1-0.2', '0.2-0.3']:
            S = []
            for c in ['ZZ', 'RR', 'TT', 'ZR', 'RZ']:
                k = f'{c}|{band}|'
                if k + 'ndays' not in Z.files:
                    continue
                cc30 = Z[k + 'bin30_full_both_cc']
                med = np.nanmedian(cc30) if np.isfinite(cc30).any() else np.nan
                if int(Z[k + 'ndays']) < 120 or not (med >= 0.5) or (band == '0.05-0.1' and pr.kind != 'BB-BB'):
                    continue
                S.append(Z[k + 'bin10_full_both_dvv'].astype(float))
            if not S:
                continue
            with np.errstate(all='ignore'):
                s = np.nanmean(np.array(S), 0) * 100
            i0 = np.arange(len(s)) * 10
            bad = site_excl_mask(a, i0, i0 + 10) | site_excl_mask(b, i0, i0 + 10)
            s[bad] = np.nan
            s = s - np.nanmean(s)
            db = '<40' if pr.dist_km < 40 else ('40-80' if pr.dist_km < 80 else '>=80')
            series.setdefault((band, db), []).append(s)
    rows = []
    for (band, db), L in series.items():
        X = np.array(L)
        with np.errstate(all='ignore'):
            m = np.nanmean(X, 0)
            n = np.isfinite(X).sum(0)
        m[n < 3] = np.nan
        t = epoch(DAY0 + pd.to_timedelta(np.arange(len(m)) * 10 + 5, 'D'))
        r = fit_series(t, m, np.zeros(len(m), int), nboot=200, block_days=90)
        if r is None:
            continue
        for q, o in summarize(*r).items():
            rows.append(dict(dataset='LP', sta=f'LP{band}|{db}', band=band, dbin=db, n_pairs=len(L), quantity=q,
                             est=o['est'], se_boot=o['se_boot'], se_plac=o['se_plac'], n_plac=o['n_plac']))
    T = honest(pd.DataFrame(rows), [])
    T.to_csv(f'{OUT}/coda_obs_lp.csv', index=False, float_format='%.5g')
    print(T.round(4).to_string())


def stage_lfe():
    sys.path.insert(0, f'{ROOT}/scripts')
    import sci_dvv_deep_shallow as DS
    # project-internal pickle written by scripts/sci_dvv_deep_shallow.py (trusted)
    S1 = pickle.load(open(f'{ROOT}/data/sci/dvv_deep_shallow/stage1.pkl', 'rb'))
    rows = []
    for (F, obs), o in S1.items():
        if obs != 'full':
            continue
        for i, ch in enumerate(o['chans']):
            th = o['th'][i]
            for q in QS:
                if q == 'D':
                    if not np.isfinite(th[6]):
                        continue
                    est = DS.decline(th)
                    bs = np.array([DS.decline(b[i]) for b in o['boot'] if np.isfinite(b[i][6])])
                    pv = [DS.decline(p[i], s) for p, s in zip(o['plac'], DS.SHIFTS) if np.isfinite(p[i][6])]
                else:
                    j = DS.PNAMES.index(q)
                    if not np.isfinite(th[j]):
                        continue
                    est = th[j]
                    bs = o['boot'][:, i, j]
                    bs = bs[np.isfinite(bs)]
                    pv = o['plac'][:, i, j]
                    pv = list(pv[np.isfinite(pv)])
                se = np.nanmax([np.std(bs) if len(bs) > 5 else np.nan, np.sqrt(np.mean(np.square(pv))) if len(pv) >= 4 else np.nan])
                if np.isfinite(se) and se > 0:
                    rows.append(dict(F=F, chan=ch, sta=DS.sta_of(ch), quantity=q, est=est * 100, se=se * 100))
    R = pd.DataFrame(rows)
    R = R[(R.se < 0.5) & (R.est.abs() < 1.5)]
    out = []
    for (sta, q), g in R.groupby(['sta', 'quantity']):
        w = 1 / g.se ** 2
        m = float((w * g.est).sum() / w.sum())
        se = float(1 / np.sqrt(w.sum()))
        chi = float((w * (g.est - m) ** 2).sum() / max(len(g) - 1, 1))
        out.append(dict(dataset='LFEcoda', sta=sta, band='2-8Hz', quantity=q, est=m, se=se * np.sqrt(max(1.0, chi)),
                        n=len(g), chi2r=chi))
    T = pd.DataFrame(out)
    T['z'] = T.est / T.se
    T.to_csv(f'{OUT}/coda_obs_lfe.csv', index=False, float_format='%.5g')
    print(T.groupby('quantity').agg(n=('est', 'size'), med=('est', 'median'), se=('se', 'median'),
                                    nz3=('z', lambda z: int((z.abs() > 3).sum()))).round(4))


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'sc': lambda: stage_sc(False), 'scpow': lambda: stage_sc(True), 'lp': stage_lp, 'lfe': stage_lfe}[st]()
