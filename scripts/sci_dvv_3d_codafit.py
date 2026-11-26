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
LFE coda per station: raw family x channel x stack S-aligned coda stretching ('full' window) of
    sci_dvv_deep_shallow, refitted with the common model (offsets per family x channel x segment), cluster
    bootstrap over (family, stack), placebos.  REP: repeater coda pairs refitted with the common model as
    pair differences per station x depth class (coda_obs_rep.csv).
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
        pd_ = {}
        for s, tp in P.items():
            vs = qvec(q, s)
            nd = vs != 0
            if np.all(np.isfinite(tp[nd])):
                pd_[s] = float(vs[nd] @ tp[nd])          # label kept with its own shift (no zip misalignment)
        pv = list(pd_.values())
        rows[q] = dict(est=est, se_boot=float(np.std(bs)) if len(bs) > 10 else np.nan,
                       plac=pd_, n_plac=len(pv),
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


def wfit(t, y, w, grp, s=0.0, irls=2, minrows=10):
    """weighted LSQ of the common time model with free offsets per group (weighted within-group demeaning,
    Frisch-Waugh) and Huber IRLS.  Returns theta (NJ, NaN = not identifiable)."""
    F = SR_.tcols(t, s)
    use = usable_cols_n(t, s, minrows)
    th = np.full(NJ, np.nan)
    if not use.any():
        return th
    X = F[:, use]
    _, gi = np.unique(grp, return_inverse=True)
    w0 = w.copy()
    ww = w.copy()
    for it in range(irls + 1):
        sw = np.bincount(gi, ww)
        my = np.bincount(gi, ww * y) / sw
        mX = np.column_stack([np.bincount(gi, ww * X[:, j]) / sw for j in range(X.shape[1])])
        Xd, yd = X - mX[gi], y - my[gi]
        sq = np.sqrt(ww)
        x, *_ = np.linalg.lstsq(Xd * sq[:, None], yd * sq, rcond=None)
        r = yd - Xd @ x
        z = np.abs(r) * np.sqrt(w0)
        z = z / (1.4826 * np.median(z) + 1e-20)
        ww = w0 * np.where(z <= 2, 1.0, 2.0 / np.maximum(z, 1e-12))
    th[use] = x
    return th


def usable_cols_n(t, s, nmin):
    use = np.ones(NJ, bool)
    sh = s * SR_.YR
    for j, k in [(2, 'pet'), (3, 'fer'), (4, 'm7')]:
        tq = SR_.T_Q[k] - sh
        ok = ((t < tq) & (t > tq - SR_.YR)).sum() >= nmin and ((t >= tq + 60 * DAY) & (t < tq + SR_.YR)).sum() >= nmin
        use[j] = ok
        use[j + 3] = ok and ((t >= tq) & (t < tq + 60 * DAY)).sum() >= max(2, nmin // 5)
    use[8] = (t < SR_.T_RAMP - sh).sum() >= 2 * nmin and (t > SR_.T_RAMP - sh + SR_.YR).sum() >= 2 * nmin
    return use


def fit_clustered(t, y, w, grp, clus, nboot=100, nmin=10):
    """real fit + cluster bootstrap + season-matched whole-year-shift placebos (pre-2021-06 rows)."""
    th = wfit(t, y, w, grp, 0.0, minrows=nmin)
    cu, ci = np.unique(clus, return_inverse=True)
    B = []
    for b in range(nboot):
        cnt = np.bincount(rng.integers(0, len(cu), len(cu)), minlength=len(cu)).astype(float)
        m = cnt[ci]
        k = m > 0
        B.append(wfit(t[k], y[k], w[k] * m[k], grp[k], 0.0, irls=1, minrows=nmin))
    P = {}
    pre = t < SR_.T_RAMP
    for s in SHIFTS:
        if pre.sum() < 4 * nmin:
            break
        tp = wfit(t[pre], y[pre], w[pre], grp[pre], s, irls=1, minrows=nmin)
        if np.isfinite(tp[4]) or np.isfinite(tp[2]):
            P[s] = tp
    return th, np.array(B), P


def stage_lp():
    """LP: per band x distance bin, all pair series (comp-averaged 10-d bins, instrument windows removed) fitted
    jointly with free offsets per pair x instrument segment; cluster (pair) bootstrap; season-matched placebos."""
    P = pd.read_csv(f'{ROOT}/data/sci/noise_lp/pairs.csv')
    from sci_dvv_3d_obs import site_excl_mask, DAY0
    rows_by = {}
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
            # instrument segments of either site (breaks) -> separate offsets
            segk = np.zeros(len(s), int)
            for site in (a, b):
                for kk in [k for k in BREAKS if k.split('.')[1] == site]:
                    for bb in BREAKS[kk]:
                        segk += (i0 >= (pd.Timestamp(bb) - DAY0).days).astype(int)
            ok = np.isfinite(s) & ~bad
            db = '<40' if pr.dist_km < 40 else ('40-80' if pr.dist_km < 80 else '>=80')
            t = epoch(DAY0 + pd.to_timedelta(i0[ok] + 5, 'D'))
            rows_by.setdefault((band, db), []).append(pd.DataFrame(dict(t=t, y=s[ok], grp=[f'{name}|{g}' for g in segk[ok]],
                                                                        clus=name)))
    out = []
    for (band, db), L in rows_by.items():
        R = pd.concat(L, ignore_index=True)
        th, B, Pl = fit_clustered(R.t.values, R.y.values, np.ones(len(R)), R.grp.values, R.clus.values, nboot=200, nmin=20)
        for q, o in summarize(th, B, Pl).items():
            out.append(dict(dataset='LP', sta=f'LP{band}|{db}', band=band, dbin=db, n_pairs=len(L), quantity=q,
                            est=o['est'], se_boot=o['se_boot'], se_plac=o['se_plac'], n_plac=o['n_plac']))
    T = honest(pd.DataFrame(out), [])
    T.to_csv(f'{OUT}/coda_obs_lp.csv', index=False, float_format='%.5g')
    print(T.round(4).to_string())


def stage_lfe():
    """LFE coda (2-8 Hz, 'full' S-aligned coda window) from the raw per family x channel x stack stretching of
    sci_dvv_deep_shallow (EXCL + BREAKS2 applied by its loader), refitted with the COMMON time model per station
    with free offsets per family x channel x segment; cluster bootstrap over (family, stack); placebos."""
    import sci_dvv_deep_shallow as DS
    rows = []
    for F in range(1, 28):
        if not os.path.exists(f'{DS.MEAS}/fam{F}.parquet'):
            continue
        d, meta, loc = DS.load(F)
        r = DS.obs_rows(d, 'full')
        r['F'] = F
        r['sta'] = r.chan.map(DS.sta_of)
        rows.append(r)
    R = pd.concat(rows, ignore_index=True)
    out = []
    for sta, g in R.groupby('sta'):
        th, B, Pl = fit_clustered(g.t.values, g.y.values * 100, 1 / (g.se.values * 100) ** 2,
                                  (g.F.astype(str) + '|' + g.chan + '|' + g.seg.astype(str)).values,
                                  (g.F.astype(str) + '|' + g['stack'].astype(str)).values, nboot=100, nmin=8)
        for q, o in summarize(th, B, Pl).items():
            out.append(dict(dataset='LFEcoda', sta=sta, band='2-8Hz', quantity=q, est=o['est'], se_boot=o['se_boot'],
                            se_plac=o['se_plac'], n_plac=o['n_plac'], n=len(g)))
    T = honest(pd.DataFrame(out), [])
    T.to_csv(f'{OUT}/coda_obs_lfe.csv', index=False, float_format='%.5g')
    print(T.groupby('quantity').agg(n=('est', 'size'), med=('est', 'median'), se=('se', 'median'),
                                    nz3=('z', lambda z: int((z.abs() > 3).sum()))).round(4))


def stage_rep():
    """repeating-earthquake coda pairs (sci_dvv_deep_shallow_rep rep_pairs_dvv.csv, same QC), refitted with the
    COMMON time model as pair differences f(tj) - f(ti) per station x depth class; cluster bootstrap; placebos."""
    R = pd.read_csv(f'{ROOT}/data/sci/dvv_deep_shallow/rep/rep_pairs_dvv.csv', parse_dates=['ti', 'tj'])
    R = R[(R.cc >= 0.5) & (R.snr >= 2) & np.isfinite(R.eps) & (np.abs(R.eps) < 0.009)]
    R['cls'] = np.where(R.depth < 10, 'shallow', np.where(R.depth > 15, 'deep', 'mid'))
    R = R[R.cls != 'mid']
    ti, tj = epoch(R.ti), epoch(R.tj)
    R['ti_s'], R['tj_s'] = ti, tj
    cc = np.clip(R.cc.values, 0, 0.99)
    R['w'] = cc ** 2 / (1 - cc ** 2)

    def pfit(g, s=0.0, wm=None):
        A = SR_.tcols(g.tj_s.values, s) - SR_.tcols(g.ti_s.values, s)
        use = np.abs(A).sum(0) > 1e-9
        use[[5, 6, 7]] &= (np.abs(A[:, [5, 6, 7]]) > 0).sum(0) >= 5
        th = np.full(NJ, np.nan)
        if use.sum() == 0:
            return th
        A = A[:, use]
        y = g.eps.values * 100
        w0 = g.w.values * (wm if wm is not None else 1.0)
        w = w0.copy()
        for it in range(3):
            sq = np.sqrt(w)
            x, *_ = np.linalg.lstsq(A * sq[:, None], y * sq, rcond=None)
            r = y - A @ x
            z = np.abs(r) * np.sqrt(w0)
            z = z / (1.4826 * np.median(z) + 1e-20)
            w = w0 * np.where(z <= 2, 1, 2 / np.maximum(z, 1e-12))
        th[use] = x
        return th
    out = []
    for (sta, cls), g in R.groupby(['sta', 'cls']):
        if len(g) < 50:
            continue
        th = pfit(g)
        cu, ci = np.unique(g.cl95.values, return_inverse=True)
        B = []
        for b in range(100):
            cnt = np.bincount(rng.integers(0, len(cu), len(cu)), minlength=len(cu)).astype(float)
            m = cnt[ci]
            k = m > 0
            B.append(pfit(g[k], 0.0, m[k]))
        Pl = {}
        pre = (g.tj_s.values < SR_.T_RAMP)
        for s in SHIFTS:
            if pre.sum() < 50:
                break
            tp = pfit(g[pre], s)
            if np.isfinite(tp[4]):
                Pl[s] = tp
        for q, o in summarize(th, np.array(B), Pl).items():
            out.append(dict(dataset='REP', sta=sta, cls=cls, band='2-8Hz', quantity=q, est=o['est'], se_boot=o['se_boot'],
                            se_plac=o['se_plac'], n_plac=o['n_plac'], n=len(g)))
    T = honest(pd.DataFrame(out), ['cls'])
    T.to_csv(f'{OUT}/coda_obs_rep.csv', index=False, float_format='%.5g')
    print(T.round(4).to_string())


# ------------------------------------------------------------------------------------------ LP robustness tests
def load_lp_pairs():
    """per pair x band: comp x side 10-d bin series (%, instrument windows removed), segment ids, geometry."""
    from sci_dvv_3d_obs import site_excl_mask, DAY0
    P = pd.read_csv(f'{ROOT}/data/sci/noise_lp/pairs.csv')
    out = []
    for fn in sorted(glob.glob('/data/wsd02/maleen_data/MTJ_LH/dvv/*.npz')):
        name = os.path.basename(fn)[:-4]
        a, b = name.split('_')
        pr = P[(P.a == a) & (P.b == b)].iloc[0]
        Z = np.load(fn)
        for band in ['0.05-0.1', '0.1-0.2', '0.2-0.3']:
            ser = {}
            for c in ['ZZ', 'RR', 'TT', 'ZR', 'RZ']:
                k = f'{c}|{band}|'
                if k + 'ndays' not in Z.files:
                    continue
                cc30 = Z[k + 'bin30_full_both_cc']
                med = np.nanmedian(cc30) if np.isfinite(cc30).any() else np.nan
                if int(Z[k + 'ndays']) < 120 or not (med >= 0.5) or (band == '0.05-0.1' and pr.kind != 'BB-BB'):
                    continue
                for side in ('both', 'caus', 'acaus'):
                    ser[(c, side)] = Z[k + f'bin10_full_{side}_dvv'].astype(float) * 100
            if not ser:
                continue
            n = len(next(iter(ser.values())))
            i0 = np.arange(n) * 10
            bad = site_excl_mask(a, i0, i0 + 10) | site_excl_mask(b, i0, i0 + 10)
            segk = np.zeros(n, int)
            for site in (a, b):
                for kk in [k for k in BREAKS if k.split('.')[1] == site]:
                    for bb in BREAKS[kk]:
                        segk += (i0 >= (pd.Timestamp(bb) - DAY0).days).astype(int)
            out.append(dict(pair=name, a=a, b=b, band=band, dist=pr.dist_km, az=pr.az % 180, ser=ser, bad=bad, seg=segk,
                            t=epoch(DAY0 + pd.to_timedelta(i0 + 5, 'D'))))
    return out


def _lp_rows(L, comps, side, pw=None):
    rows = []
    for p in L:
        S = [p['ser'][(c, side)] for c in comps if (c, side) in p['ser']]
        if not S:
            continue
        with np.errstate(all='ignore'):
            s = np.nanmean(np.array(S), 0)
        ok = np.isfinite(s) & ~p['bad']
        if ok.sum() < 20:
            continue
        r = pd.DataFrame(dict(t=p['t'][ok], y=s[ok], grp=[f"{p['pair']}|{g}" for g in p['seg'][ok]], clus=p['pair'],
                              a=p['a'], b=p['b']))
        if pw is not None:
            r['pw'] = pw[ok]
        rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else None


def stage_lptests():
    L = load_lp_pairs()
    pw = power_series()['0.1-0.3']
    from sci_dvv_3d_obs import DAY0
    tq = SR_.T_Q['pet']
    out, yearly = [], []
    for band in ['0.05-0.1', '0.1-0.2', '0.2-0.3']:
        for db, (d0, d1) in {'<40': (0, 40), '40-80': (40, 80), '>=80': (80, 1e9), 'all': (0, 1e9)}.items():
            sel = [p for p in L if p['band'] == band and d0 <= p['dist'] < d1]
            if not sel:
                continue
            # pairs spanning Petrolia: >= 6 good bins within 1 yr before and after
            nspan = sum(1 for p in sel if ((p['t'] < tq) & (p['t'] > tq - SR_.YR) & ~p['bad']).sum() >= 6
                        and ((p['t'] > tq) & (p['t'] < tq + SR_.YR) & ~p['bad']).sum() >= 6)
            variants = {'all_comps': (['ZZ', 'RR', 'TT', 'ZR', 'RZ'], 'both', None), 'ZZ': (['ZZ'], 'both', None),
                        'TT': (['TT'], 'both', None), 'Rayl_RR_ZR_RZ': (['RR', 'ZR', 'RZ'], 'both', None),
                        'causal': (['ZZ', 'RR', 'TT', 'ZR', 'RZ'], 'caus', None),
                        'acausal': (['ZZ', 'RR', 'TT', 'ZR', 'RZ'], 'acaus', None),
                        'power_reg': (['ZZ', 'RR', 'TT', 'ZR', 'RZ'], 'both', 'pw')}
            for azb in range(4):
                variants[f'az{azb * 45}-{azb * 45 + 45}'] = (['ZZ', 'RR', 'TT', 'ZR', 'RZ'], 'both', ('az', azb))
            for vn, (comps, side, extra) in variants.items():
                s2 = sel
                if isinstance(extra, tuple):
                    s2 = [p for p in sel if int(p['az'] // 45) == extra[1]]
                pwx = None
                R = _lp_rows(s2, comps, side)
                if R is None or R.clus.nunique() < 3:
                    continue
                if extra == 'pw':
                    tt = pd.to_datetime(R.t.values, unit='s').normalize()
                    R['pwv'] = pw.reindex(tt).values
                    R = R[np.isfinite(R.pwv)]
                    # regress out power within pairs first (FWL): y ~ offsets + b*pw, then fit the time model
                    _, gi = np.unique(R.grp.values, return_inverse=True)
                    ym = R.y - R.groupby('grp').y.transform('mean')
                    pm = R.pwv - R.groupby('grp').pwv.transform('mean')
                    bpw = float((ym * pm).sum() / (pm * pm).sum())
                    R = R.assign(y=R.y - bpw * R.pwv)
                th, B, Pl = fit_clustered(R.t.values, R.y.values, np.ones(len(R)), R.grp.values, R.clus.values,
                                          nboot=100, nmin=20)
                # station-cluster bootstrap (pairs weighted by the product of their stations' draw counts)
                stas = sorted(set(R.a) | set(R.b))
                sidx = {s: i for i, s in enumerate(stas)}
                ia, ib = R.a.map(sidx).values, R.b.map(sidx).values
                SB = []
                for k in range(100):
                    cnt = np.bincount(rng.integers(0, len(stas), len(stas)), minlength=len(stas)).astype(float)
                    m = cnt[ia] * cnt[ib]
                    kk = m > 0
                    if kk.sum() < 50:
                        continue
                    SB.append(wfit(R.t.values[kk], R.y.values[kk], m[kk], R.grp.values[kk], 0.0, irls=1, minrows=20))
                SB = np.array(SB)
                for q, o in summarize(th, B, Pl).items():
                    v = qvec(q)
                    nd = v != 0
                    sbs = np.array([v[nd] @ b_[nd] for b_ in SB]) if len(SB) else np.array([])
                    sbs = sbs[np.isfinite(sbs)]
                    out.append(dict(band=band, dbin=db, variant=vn, quantity=q, n_pairs=R.clus.nunique(), n_span_pet=nspan,
                                    est=o['est'], se_pairboot=o['se_boot'], se_stationboot=float(np.std(sbs)) if len(sbs) > 10 else np.nan,
                                    se_plac=o['se_plac'], n_plac=o['n_plac'], flag_few_plac=o['n_plac'] < 4,
                                    b_power=bpw if extra == 'pw' else np.nan))
                if vn == 'all_comps':
                    # yearly series: yearly indicators with pair x segment offsets (FWL)
                    Fy = np.array([((R.t.values >= SR_._ep(f'{y}-01-01')) & (R.t.values < SR_._ep(f'{y + 1}-01-01'))).astype(float)
                                   for y in range(2009, 2027)]).T
                    _, gi = np.unique(R.grp.values, return_inverse=True)
                    sw = np.bincount(gi)
                    Xd = Fy - np.column_stack([np.bincount(gi, Fy[:, j]) / sw for j in range(Fy.shape[1])])[gi]
                    yd = R.y.values - (np.bincount(gi, R.y.values) / sw)[gi]
                    x, *_ = np.linalg.lstsq(Xd[:, 1:], yd, rcond=None)
                    for j, y in enumerate(range(2010, 2027)):
                        yearly.append(dict(band=band, dbin=db, year=y, est_rel2009=x[j]))
    T = pd.DataFrame(out)
    T['se'] = T[['se_pairboot', 'se_stationboot', 'se_plac']].max(axis=1)
    T['z'] = T.est / T.se
    T.to_csv(f'{OUT}/lp_tests.csv', index=False, float_format='%.5g')
    pd.DataFrame(yearly).to_csv(f'{OUT}/lp_yearly.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250):
        print(T[(T.dbin == '>=80') & T.quantity.isin(['pet', 'D', 'm7'])].pivot_table(index=['band', 'variant'], columns='quantity',
                                                                                     values=['est', 'z']).round(3).to_string())


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'sc': lambda: stage_sc(False), 'scpow': lambda: stage_sc(True), 'lp': stage_lp, 'lfe': stage_lfe,
         'rep': stage_rep, 'lptests': stage_lptests}[st]()
