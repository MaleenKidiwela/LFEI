#!/usr/bin/env python3
"""Analyse the multi-family quake scans (scan_quakes_mf.py) against the v2 catalog.

Per quake (tag): baseline noise b_F = median pre-quake hourly MAD of family F's CC sum.
Detection rules (positive = normal polarity; negative = reversed polarity = false-alarm control):
  daily   |ccsum| >= 8 x daily MAD           (the standard adaptive rule, as scan_matched_filter.py)
  hourly  |ccsum| >= 8 x hourly MAD          (adaptive to hourly noise)
  fixed   |ccsum| >= KF x b_F                (fixed absolute CC-sum threshold, KF = 8 and 10)
Cross-family duplicates: within +-2 s across families keep the detection with the largest meancc.
Outputs data/scan_quakes/{tag}_summary.json, {tag}_family_table.csv, {tag}_hourly_threshold.csv,
{tag}_detections.csv (deduplicated, with catalog flags), {tag}_injection.csv (if injected).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

QUAKES = {'petrolia': ('2021-12-20T20:10:20', 'petrolia'),
          'ferndale': ('2022-12-20T10:34:24', 'ferndale'),
          'm7': ('2024-12-05T18:44:21', '2024')}
CAT_REF = {'petrolia': dict(n24=4, fam2d=8, r30=1374 / 1459),
           'ferndale': dict(n24=23, fam2d=20, r30=695 / 1295),
           'm7': dict(n24=0, fam2d=0, r30=509 / 1221)}
TOL = 2.0


def ep(x):
    return np.asarray((pd.DatetimeIndex(x) - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1))


def near(ts, ref, tol=TOL):
    ref = np.r_[-np.inf, np.sort(np.asarray(ref, float)), np.inf]
    if len(ts) == 0:
        return np.zeros(len(ts), bool)
    i = np.searchsorted(ref, ts)
    return np.minimum(abs(ts - ref[i - 1]), abs(ts - ref[i])) <= tol


def dedup(d):
    """Group detections (one sign) closer than TOL s across families; keep max meancc."""
    if len(d) == 0:
        return d
    d = d.sort_values('ts').reset_index(drop=True)
    g = np.r_[0, np.cumsum(np.diff(d.ts.values) > TOL)]
    d['grp'] = g
    return d.loc[d.groupby('grp').meancc.apply(lambda s: s.abs().idxmax())].drop(columns='grp')


def load(tag, root='data/scan_quakes'):
    fs = sorted(glob.glob(f'{root}/{tag}/det_*.csv'))
    D = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 5], ignore_index=True)
    D['t'] = pd.to_datetime(D.time_utc.str.replace('Z', ''), format='mixed')
    D['ts'] = ep(D.t)
    H = pd.concat([pd.read_csv(f).assign(day=os.path.basename(f)[7:17])
                   for f in sorted(glob.glob(f'{root}/{tag}/hourly_*.csv')) if os.path.getsize(f) > 5],
                  ignore_index=True)
    H['t'] = pd.to_datetime(H.day) + pd.to_timedelta(H.hour, unit='h')
    R = pd.concat([pd.read_csv(f).assign(day=os.path.basename(f)[8:18])
                   for f in sorted(glob.glob(f'{root}/{tag}/chanrms_*.csv')) if os.path.getsize(f) > 5],
                  ignore_index=True)
    R['t'] = pd.to_datetime(R.day) + pd.to_timedelta(R.hour, unit='h')
    return D, H, R, [os.path.basename(f)[4:14] for f in fs]


def catalog():
    c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv')
    c = c[c.source != 'v1_only'].copy()
    c['t'] = pd.to_datetime(c.time_utc, format='mixed')
    c['ts'] = ep(c.t)
    return c


def fa_filter(neg, pos, cw, tol=3.0):
    """False-alarm control: reversed detections NOT within +-tol s of a catalog LFE or of a
    catalog-matched positive detection (those are CC side-lobes / cross-family responses of real LFEs)."""
    ref = np.r_[cw.ts.values, pos[pos.catany].ts.values]
    return neg[~near(neg.ts.values, ref, tol)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--pre', type=float, default=30.0)
    ap.add_argument('--post', type=float, default=30.0)
    a = ap.parse_args()
    tag = a.tag
    base = tag.split('_')[0]
    Q = pd.Timestamp(QUAKES[base][0])
    qs = ep([Q])[0]
    D, H, R, days = load(tag)
    t_lo = max(Q - pd.Timedelta(days=a.pre), pd.Timestamp(days[0]))
    t_hi = min(Q + pd.Timedelta(days=a.post), pd.Timestamp(days[-1]) + pd.Timedelta(days=1))
    T_pre = (Q - t_lo).total_seconds() / 86400
    T_post = (t_hi - Q).total_seconds() / 86400
    c = catalog()
    cw = c[(c.t >= t_lo) & (c.t < t_hi)]
    fams = sorted(D.fam.unique())

    # baseline noise and fixed thresholds
    Hpre = H[(H.t >= t_lo) & (H.t < Q - pd.Timedelta(hours=1))]
    b = Hpre.groupby('fam').mad.median()
    nch0 = Hpre.groupby('fam').nchan_med.median()
    H['mad_rel'] = H.mad / H.fam.map(b)
    H['nchan_rel'] = H.nchan_med / H.fam.map(nch0)
    floor_viol = (H.groupby(['fam', 'day']).floor.first() > 8 * b.reindex(
        H.groupby(['fam', 'day']).floor.first().index.get_level_values(0)).values).mean()

    D['b'] = D.fam.map(b)
    rules = {'daily': D.ccsum.abs() >= 8 * D.mad_day,
             'hourly': D.ccsum.abs() >= 8 * D.mad_hour,
             'fixed8': D.ccsum.abs() >= 8 * D.b,
             'fixed10': D.ccsum.abs() >= 10 * D.b,
             'fixed12': D.ccsum.abs() >= 12 * D.b}
    out = dict(tag=tag, quake=str(Q), window=[str(t_lo), str(t_hi)], T_pre_d=T_pre, T_post_d=T_post,
               n_days_scanned=len(days), floor_violation_frac=float(floor_viol),
               baseline_mad_median=float(b.median()))
    # catalog reference
    def cnt(x, t0, t1):
        return int(((x.t >= t0) & (x.t < t1)).sum())
    wins = {'24h': (Q, Q + pd.Timedelta(days=1)), '2d': (Q, Q + pd.Timedelta(days=2)),
            '30d': (Q, t_hi), 'd1_2': (Q + pd.Timedelta(days=1), Q + pd.Timedelta(days=2)),
            'd2_7': (Q + pd.Timedelta(days=2), Q + pd.Timedelta(days=7)),
            '6h': (Q, Q + pd.Timedelta(hours=6))}
    pre_w = (t_lo, Q)
    cat_pre = cnt(cw, *pre_w)
    # aftershock-coda association: an NCSN earthquake (within 1.4 deg) 15-65 s BEFORE the detection
    from obspy import read_events
    evc = read_events(f'data/scan_quakes/ncsn_{base}.xml')
    eqt = np.sort(np.array([(e.preferred_origin() or e.origins[0]).time.timestamp for e in evc]))

    def eqassoc(ts):
        ts = np.asarray(ts, float)
        return np.searchsorted(eqt, ts - 15) > np.searchsorted(eqt, ts - 65)

    def eq_prob_random(t0, t1):
        r = np.linspace(ep([t0])[0], ep([t1])[0], 20000, endpoint=False)
        return float(eqassoc(r).mean())
    cpre = cw[(cw.t >= t_lo) & (cw.t < Q)]
    cat_pre_clean = (len(cpre) - eqassoc(cpre.ts.values).sum()) / (1 - eq_prob_random(t_lo, Q))
    out['catalog'] = dict(pre=cat_pre, pre_rate=cat_pre / T_pre)
    for k, (t0, t1) in wins.items():
        dur = (t1 - t0).total_seconds() / 86400
        n = cnt(cw, t0, t1)
        cc_ = cw[(cw.t >= t0) & (cw.t < t1)]
        pr = eq_prob_random(t0, t1)
        na = int(eqassoc(cc_.ts.values).sum())
        nclean = (n - na) / max(1 - pr, 1e-9)       # genuine-LFE estimate: remove aftershock-coda associates
        out['catalog'][k] = dict(n=n, ratio=(n / dur) / (cat_pre / T_pre), nfam=int(cc_.idnum.nunique()),
                                 n_eqassoc=na, p_eqassoc_random=pr, n_clean=nclean,
                                 ratio_clean=(nclean / dur) / (cat_pre_clean / T_pre),
                                 nfam_clean=int(cc_[~eqassoc(cc_.ts.values)].idnum.nunique()))

    fam_rows = []
    dets_all = []
    for rn, m in rules.items():
        X = D[m].copy()
        pos, neg = dedup(X[X.sign == 1].copy()), dedup(X[X.sign == -1].copy())
        pos['catfam'] = False
        for F in fams:
            k = pos.fam == F
            pos.loc[k, 'catfam'] = near(pos.ts[k].values, cw[cw.idnum == F].ts.values)
        pos['catany'] = near(pos.ts.values, cw.ts.values)
        n_all = neg
        neg = fa_filter(neg, pos, cw)
        if rn in ('fixed8', 'fixed10', 'daily'):
            dets_all.append(pos.assign(rule=rn))
            dets_all.append(n_all.assign(rule=rn, fa=near(n_all.ts.values, neg.ts.values, 0.01)))
        r = dict(pre_pos=cnt(pos, *pre_w), pre_neg=cnt(neg, *pre_w), pre_neg_all=cnt(n_all, *pre_w),
                 pre_new=cnt(pos[~pos.catany], *pre_w), pre_catmatch=cnt(pos[pos.catany], *pre_w))
        r['pre_pos_rate'] = r['pre_pos'] / T_pre
        r['pre_neg_rate'] = r['pre_neg'] / T_pre
        net_pre = (r['pre_pos'] - r['pre_neg']) / T_pre
        for k, (t0, t1) in wins.items():
            dur = (t1 - t0).total_seconds() / 86400
            npos, nneg = cnt(pos, t0, t1), cnt(neg, t0, t1)
            pw = pos[(pos.t >= t0) & (pos.t < t1)]
            nw = neg[(neg.t >= t0) & (neg.t < t1)]
            pr = eq_prob_random(t0, t1)
            na_p, na_n = int(eqassoc(pw.ts.values).sum()), int(eqassoc(nw.ts.values).sum())
            r[k] = dict(pos=npos, neg=nneg, neg_all=cnt(n_all, t0, t1), catmatch=int(pw.catany.sum()),
                        pos_eqassoc=na_p, neg_eqassoc=na_n, p_eqassoc_random=pr,
                        net_clean=((npos - na_p) - (nneg - na_n)) / max(1 - pr, 1e-9),
                        ratio=(npos / dur) / max(r['pre_pos_rate'], 1e-9),
                        ratio_fa_corr=((npos - nneg) / dur) / max(net_pre, 1e-9),
                        neg_ratio=(nneg / dur) / max(r['pre_neg_rate'], 1e-9),
                        nfam_pos=int(pw.fam.nunique()), nfam_neg=int(nw.fam.nunique()),
                        nfam_pos_cat=int(pw[pw.catany].fam.nunique()))
        # recall of catalog events (any-family template within +-2 s, before dedup, positive)
        Xp = X[X.sign == 1]
        own = np.zeros(len(cw), bool)
        for F in fams:
            k = (cw.idnum == F).values
            own[k] = near(cw.ts.values[k], Xp[Xp.fam == F].ts.values)
        cw_ = cw.assign(own=own, anyf=near(cw.ts.values, Xp.ts.values))
        rec = {}
        for k, (t0, t1) in {'pre': pre_w, **wins}.items():
            cc = cw_[(cw_.t >= t0) & (cw_.t < t1)]
            rec[k] = dict(n=len(cc), own=float(cc.own.mean()) if len(cc) else None,
                          any=float(cc.anyf.mean()) if len(cc) else None)
        r['recall'] = rec
        out[rn] = r
        # per family table
        for F in fams:
            cf = cw_[cw_.idnum == F]
            pf, nf = pos[pos.fam == F], neg[neg.fam == F]
            row = dict(fam=F, rule=rn, b=b.get(F), nchan0=nch0.get(F),
                       cat_pre=cnt(cf, *pre_w), cat_24h=cnt(cf, *wins['24h']), cat_2d=cnt(cf, *wins['2d']),
                       cat_30d=cnt(cf, *wins['30d']),
                       recall_pre=float(cf[cf.t < Q].own.mean()) if (cf.t < Q).any() else np.nan,
                       recall_post30=float(cf[cf.t >= Q].own.mean()) if (cf.t >= Q).any() else np.nan,
                       pos_pre=cnt(pf, *pre_w), neg_pre=cnt(nf, *pre_w),
                       pos_24h=cnt(pf, *wins['24h']), neg_24h=cnt(nf, *wins['24h']),
                       pos_2d=cnt(pf, *wins['2d']), neg_2d=cnt(nf, *wins['2d']),
                       pos_30d=cnt(pf, *wins['30d']), neg_30d=cnt(nf, *wins['30d']),
                       new_pre=cnt(pf[~pf.catany], *pre_w), new_30d=cnt(pf[~pf.catany], *wins['30d']))
            fam_rows.append(row)
    ft = pd.DataFrame(fam_rows)
    ft.to_csv(f'data/scan_quakes/{tag}_family_table.csv', index=False)
    pd.concat(dets_all).drop(columns=['b']).to_csv(f'data/scan_quakes/{tag}_detections.csv', index=False)

    # threshold / noise through time (network medians over families / channels)
    hh = H.groupby('t').agg(mad_rel_med=('mad_rel', 'median'), mad_rel_p90=('mad_rel', lambda s: s.quantile(0.9)),
                            nchan_rel_med=('nchan_rel', 'median')).reset_index()
    Rpre = R[(R.t >= t_lo) & (R.t < Q - pd.Timedelta(hours=1))].groupby('cid').rms.median()
    R['rms_rel'] = R.rms / R.cid.map(Rpre)
    rr = R.groupby('t').rms_rel.median().rename('chan_rms_rel_med').reset_index()
    hh = hh.merge(rr, on='t', how='left')
    hh['hours_since'] = (hh.t - Q).dt.total_seconds() / 3600
    hh.to_csv(f'data/scan_quakes/{tag}_hourly_threshold.csv', index=False)
    th = {}
    for k, (t0, t1) in {'pre': pre_w, 'h0_6': (Q, Q + pd.Timedelta(hours=6)),
                        'h6_24': (Q + pd.Timedelta(hours=6), Q + pd.Timedelta(days=1)), **wins}.items():
        x = hh[(hh.t >= t0.floor('h')) & (hh.t < t1)]
        th[k] = dict(mad_rel_median=float(x.mad_rel_med.median()), mad_rel_max=float(x.mad_rel_med.max()),
                     hours_mad_gt_1p2=int((x.mad_rel_med > 1.2).sum()),
                     chan_rms_rel_median=float(x.chan_rms_rel_med.median()),
                     nchan_rel_median=float(x.nchan_rel_med.median()))
    out['threshold'] = th

    # injection test
    inj_pk = f'data/scan_quakes/inject/{base}_inject_v3.pkl'
    if not os.path.exists(inj_pk):
        inj_pk = f'data/scan_quakes/inject/{base}_inject.pkl'
    if os.path.exists(inj_pk) and glob.glob(f'data/scan_quakes/{tag}_inj/det_*.csv'):
        tab = pd.read_pickle(inj_pk)['table']   # our own file (scan_quakes_inject.py)
        DI, HI, _, idays = load(f'{tag}_inj')
        tab = tab[tab.day.isin(idays)].copy()
        tab['ts'] = ep(pd.to_datetime(tab.t_inj))
        DI['b'] = DI.fam.map(b)
        P = DI[DI.sign == 1]
        res = []
        for r in tab.itertuples():
            x = P[(P.fam == r.fam) & (abs(P.ts - r.ts) <= 1.5)]
            if len(x):
                x = x.loc[x.ccsum.idxmax()]
                res.append(dict(fam=r.fam, ts=r.ts, ccsum=x.ccsum, b=x.b, mad_day=x.mad_day, mad_hour=x.mad_hour))
            else:
                res.append(dict(fam=r.fam, ts=r.ts, ccsum=0.0, b=b.get(r.fam), mad_day=np.nan, mad_hour=np.nan))
        I = pd.DataFrame(res)
        I['h'] = (I.ts - qs) / 3600
        I['det_fixed8'] = I.ccsum >= 8 * I.b
        I['det_fixed10'] = I.ccsum >= 10 * I.b
        I['det_fixed12'] = I.ccsum >= 12 * I.b
        I['det_daily'] = I.ccsum >= 8 * I.mad_day
        I['det_hourly'] = I.ccsum >= 8 * I.mad_hour
        I['ccrel'] = I.ccsum / I.b
        I.to_csv(f'data/scan_quakes/{tag}_injection.csv', index=False)
        bins = {'pre': (-1e9, 0), 'h0_6': (0, 6), 'h6_24': (6, 24), 'h0_24': (0, 24), 'h24_48': (24, 48),
                'h0_48': (0, 48), 'h48_96': (48, 96), 'd4_7': (96, 168), 'd7_14': (168, 336), 'd14_30': (336, 1e9)}
        cols = ['det_fixed8', 'det_fixed10', 'det_fixed12', 'det_daily', 'det_hourly']
        inj = {}
        for k, (h0, h1) in bins.items():
            x = I[(I.h >= h0) & (I.h < h1)]
            if len(x):
                inj[k] = dict(n=len(x), **{c: float(x[c].mean()) for c in cols},
                              ccrel_median=float(x.ccrel.median()))
        out['injection'] = inj
        # relative efficiency e(h) (piecewise; a missing bin takes the previous bin's value)
        seq = [(0, 6, 'h0_6'), (6, 24, 'h6_24'), (24, 48, 'h24_48'), (48, 96, 'h48_96'),
               (96, 168, 'd4_7'), (168, 336, 'd7_14'), (336, 1e9, 'd14_30')]

        def eff_fun(col):
            """Relative efficiency e(h): fine bins over the first 4 d; after that, linear interpolation
            between the daily efficiencies of the injected days (h >= 72 h)."""
            e0 = inj['pre'][col]
            pieces, last = [], 1.0
            for h0, h1, k in seq[:4]:
                if k in inj:
                    last = inj[k][col] / max(e0, 1e-9)
                pieces.append((h0, h1, last))
            J = I[I.h >= 72].copy()
            J['dc'] = (np.floor((J.ts - (qs - (qs % 86400))) / 86400) * 86400 + (qs - (qs % 86400)) + 43200 - qs) / 3600
            dd = J.groupby('dc')[col].mean() / max(e0, 1e-9)
            return pieces, dd.index.values, dd.values

        def mean_eff(model, h0w, h1w):
            pieces, xc, yc = model
            hs = np.linspace(h0w, h1w, 2000, endpoint=False)
            e = np.ones_like(hs)
            for a0, a1, v in pieces:
                e[(hs >= a0) & (hs < a1)] = v
            if len(xc):
                late = hs >= 96
                e[late] = np.interp(hs[late], xc, yc)
            return float(e.mean())
        # observed net (positive - reversed false alarms) vs expected under "no rate change, only detection loss"
        out['corrected'] = {}
        for rn in ('fixed8', 'fixed10', 'daily'):
            pieces = eff_fun('det_' + rn)
            X = D[rules[rn]]
            pos_ = dedup(X[X.sign == 1].copy())
            pos_['catany'] = near(pos_.ts.values, cw.ts.values)
            neg_ = fa_filter(dedup(X[X.sign == -1].copy()), pos_, cw)
            net_pre = (out[rn]['pre_pos'] - out[rn]['pre_neg']) / T_pre
            # pre-quake sliding windows (1-h steps) of net counts, for an empirical (clustering-aware) p-value
            pts, nts = np.sort(pos_.ts.values), np.sort(neg_.ts.values)
            t_lo_s = ep([t_lo])[0]
            res = {}
            for k, (t0, t1) in wins.items():
                dur = (t1 - t0).total_seconds() / 86400
                h0w, h1w = (t0 - Q).total_seconds() / 3600, (t1 - Q).total_seconds() / 3600
                me = mean_eff(pieces, h0w, h1w)
                O = int(((pos_.t >= t0) & (pos_.t < t1)).sum() - ((neg_.t >= t0) & (neg_.t < t1)).sum())
                E = net_pre * dur * me
                r_ = dict(mean_rel_eff=me, observed_net=O, expected_if_unchanged=E,
                          ratio_eff_corr=O / max(E, 1e-9),
                          catalog_ratio_eff_corr=out['catalog'][k]['ratio'] / max(me, 1e-9))
                wl = dur * 86400
                starts = np.arange(t_lo_s, qs - wl + 1, 3600.0)
                if len(starts) >= 5 and dur <= 7:
                    npw = (np.searchsorted(pts, starts + wl) - np.searchsorted(pts, starts)) - \
                          (np.searchsorted(nts, starts + wl) - np.searchsorted(nts, starts))
                    r_['p_empirical'] = float(np.mean(npw * me <= max(O, 0)))   # FA noise can make O < 0
                    r_['pre_window_net_q05_q50'] = [float(np.quantile(npw * me, 0.05)), float(np.median(npw * me))]
                    # same test for the catalog (aftershock-associated events removed), assuming the catalog's
                    # efficiency loss equals this rule's measured loss
                    cts = np.sort(cpre.ts.values[~eqassoc(cpre.ts.values)])
                    ncw = np.searchsorted(cts, starts + wl) - np.searchsorted(cts, starts)
                    r_['catalog_p_empirical'] = float(np.mean(ncw * me <= out['catalog'][k]['n_clean']))
                    r_['catalog_expected_if_unchanged'] = float(cat_pre_clean / T_pre * dur * me)
                res[k] = r_
            out['corrected'][rn] = res
        # noise-model efficiency: logistic fit of injected detection vs log10(hourly median channel RMS / pre),
        # applied hour by hour over every window (uses the measured noise time series, not only injected days)
        from scipy.optimize import minimize
        hr_rms = hh.set_index('t').chan_rms_rel_med
        I['t_hour'] = (pd.Timestamp('1970-01-01') + pd.to_timedelta(I.ts, unit='s')).dt.floor('h')
        I['lrms'] = np.log10(I.t_hour.map(hr_rms).astype(float).clip(lower=0.1))
        out['corrected_noise_model'] = {}
        for rn in ('fixed8', 'fixed10', 'daily'):
            col = 'det_' + rn
            J = I.dropna(subset=['lrms'])
            y, xv = J[col].values.astype(float), J.lrms.values

            def nll(par):
                z = np.clip(par[0] + par[1] * xv, -30, 30)
                pz = 1 / (1 + np.exp(-z))
                return -np.sum(y * np.log(pz + 1e-12) + (1 - y) * np.log(1 - pz + 1e-12))
            fit = minimize(nll, [-1.0, -1.0], method='Nelder-Mead')
            a0, a1 = fit.x

            def pdet(lr):
                return 1 / (1 + np.exp(-np.clip(a0 + a1 * lr, -30, 30)))
            lr_all = np.log10(hr_rms.astype(float).clip(lower=0.1))
            t_all = hr_rms.index
            e_pre = float(np.nanmean(pdet(lr_all[(t_all >= t_lo) & (t_all < Q)])))
            net_pre = (out[rn]['pre_pos'] - out[rn]['pre_neg']) / T_pre
            res = dict(logit_a=float(a0), logit_b=float(a1), n_inj=int(len(J)))
            for k, (t0, t1) in wins.items():
                sel = (t_all >= t0.floor('h')) & (t_all < t1)
                me = float(np.nanmean(pdet(lr_all[sel])) / e_pre) if sel.any() else np.nan
                dur = (t1 - t0).total_seconds() / 86400
                O = out[rn][k]['pos'] - out[rn][k]['neg']
                E = net_pre * dur * me
                res[k] = dict(mean_rel_eff=me, observed_net=int(O), expected_if_unchanged=E,
                              ratio_eff_corr=O / max(E, 1e-9),
                              catalog_ratio_eff_corr=out['catalog'][k]['ratio'] / max(me, 1e-9),
                              catalog_clean_ratio_eff_corr=out['catalog'][k]['ratio_clean'] / max(me, 1e-9))
            out['corrected_noise_model'][rn] = res
    json.dump(out, open(f'data/scan_quakes/{tag}_summary.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
