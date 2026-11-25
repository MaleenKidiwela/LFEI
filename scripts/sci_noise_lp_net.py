#!/usr/bin/env python3
"""Long-period inter-station noise dv/v: network combination, event steps vs placebo dates and moving-block
bootstrap, injection-based detection limits, long-term trend + seasonal fit, depth conversion, figures.

Input: per-pair results of sci_noise_lp_dvv.py (/data/wsd02/maleen_data/MTJ_LH/dvv/*.npz), kernel fractions of
sci_noise_lp_kernels.py. Output: data/sci/noise_lp/{pair_qc.csv, steps.csv, detection.csv, trend.csv,
network_series_30d.csv, network_series_10d.csv, summary.json}; figures/sci_noise_lp_*.png.

Entity = (pair, component) in one band. Pair QC: >= 120 good days and median CC (30-d stacks vs the full
reference, both coda sides) >= CCMIN. Groups: components ZZ, RR, TT, ZR+RZ, 'Rayl' (ZZ+RR+ZR+RZ), 'all';
station kinds 'BB' (broadband pairs only) and 'all' (incl. PB borehole geophones; not used at 0.05-0.1 Hz).
Steps: s = dvv(post W days) - dvv(pre W days) (event day excluded). Network step = sum w s / sum w over the
entities valid at the event, w = 1/sigma^2, sigma = 1.4826 MAD of the entity's own placebo steps. The placebo
distribution uses the SAME entities and weights at every 3rd day (>= 70 % of the event weight available,
>= W+10 days from any of the 5 events); 'seasonal' placebos are restricted to +/-45 days of the event's
day of year. Bootstrap: same weights on 300 moving-block (5-day) replicates of the pre/post windows.
Detection: post window stretched by dv/v = -0.02..-0.2 % at every placebo date; P(detect) = fraction with the
network step below the 5th percentile of the placebo steps (one-sided, velocity drop); the 90 % detection limit
is interpolated. Depth: a change confined to [z1, z2] appears as delta * F(z1, z2), F = kernel fraction
(Rayleigh for ZZ/RR/ZR, Love for TT, mean for 'all').
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.path.join(ROOT, 'data/sci/noise_lp')
FIG = os.path.join(ROOT, 'figures')
DVVD = '/data/wsd02/maleen_data/MTJ_LH/dvv'
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from sci_noise_lp_dvv import (BANDS, COMPS, DAY0, EVENTS, GRID, INJ, NDAYS, WS)  # noqa: E402

CCMIN = 0.5
GROUPS = {'ZZ': ['ZZ'], 'RR': ['RR'], 'TT': ['TT'], 'ZR': ['ZR', 'RZ'], 'Rayl': ['ZZ', 'RR', 'ZR', 'RZ'],
          'all': COMPS}
EVD = {k: (pd.Timestamp(v) - DAY0).days for k, v in EVENTS.items()}


def load():
    P = pd.read_csv(os.path.join(OUT, 'pairs.csv'))
    ent = []
    data = {}
    for fn in sorted(glob.glob(os.path.join(DVVD, '*.npz'))):
        name = os.path.basename(fn)[:-4]
        a, b = name.split('_')
        pr = P[(P.a == a) & (P.b == b)].iloc[0]
        Z = np.load(fn)
        keys = {tuple(k.split('|')[:2]) for k in Z.files if '|' in k}
        for comp, band in keys:
            g = lambda f: Z[f'{comp}|{band}|{f}']  # noqa: E731
            cc30 = g('bin30_full_both_cc')
            medcc = float(np.nanmedian(cc30)) if np.isfinite(cc30).any() else np.nan
            e = dict(pair=name, a=a, b=b, comp=comp, band=band, dist=pr.dist_km, kind=pr.kind,
                     mid_lat=pr.mid_lat, mid_lon=pr.mid_lon, ndays=int(g('ndays')), medcc30=medcc,
                     t1=float(g('t1')), t2=float(g('t2')))
            d = {k.split('|')[2]: Z[k] for k in Z.files if k.startswith(f'{comp}|{band}|')}
            d['dates'] = Z['dates']
            ent.append(e)
            data[(name, comp, band)] = d
    E = pd.DataFrame(ent)
    E['ok'] = (E.ndays >= 120) & (E.medcc30 >= CCMIN)
    E.loc[(E.band == '0.05-0.1') & (E.kind != 'BB-BB'), 'ok'] = False
    return E, data


def robust_sd(x):
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def near_event(dates, W, pad=10):
    m = np.zeros(len(dates), bool)
    for d in EVD.values():
        m |= np.abs(dates - d) <= W + pad
    return m


def doy(i):
    return (DAY0 + pd.Timedelta(days=int(i))).dayofyear


def select(E, band, kind, comps):
    sel = E[E.ok & (E.band == band) & E.comp.isin(comps)]
    if kind == 'BB':
        sel = sel[sel.kind == 'BB-BB']
    return sel


def event_steps(E, data):
    rows, boots = [], {}
    for band in BANDS:
        for kind in ['BB', 'all']:
            if band == '0.05-0.1' and kind == 'all':
                continue
            for gname, comps in GROUPS.items():
                sel = select(E, band, kind, comps)
                if sel.empty:
                    continue
                for W in WS:
                    for side in ['both', 'caus', 'acaus']:
                        ents = []
                        for _, e in sel.iterrows():
                            d = data[(e.pair, e.comp, band)]
                            dates = d['dates']
                            s = d[f'step{W}_{side}'].astype(float)
                            plac = ~near_event(dates, W) & (np.arange(len(dates)) < len(GRID))
                            sd = robust_sd(s[plac])
                            if not np.isfinite(sd) or sd <= 0:
                                continue
                            ents.append(dict(pair=e.pair, s=s, w=1 / sd ** 2,
                                             inj=d[f'inj{W}'].astype(float) if side == 'both' else None,
                                             boot=d[f'boot{W}'].astype(float) if side == 'both' else None))
                        if not ents:
                            continue
                        S = np.array([x['s'] for x in ents])
                        Wt = np.array([x['w'] for x in ents])
                        dates = data[(sel.iloc[0].pair, sel.iloc[0].comp, band)]['dates']
                        plac = ~near_event(dates, W) & (np.arange(len(dates)) < len(GRID))
                        for ie, (ev, di) in enumerate(EVD.items()):
                            jj = len(GRID) + ie
                            v = np.isfinite(S[:, jj])
                            if v.sum() == 0:
                                continue
                            w = Wt * v
                            net = float(np.sum(w * np.nan_to_num(S[:, jj])) / w.sum())
                            avail = np.isfinite(S) & v[:, None]
                            wsum = (w[:, None] * avail).sum(axis=0)
                            okd = plac & (wsum >= 0.7 * w.sum())
                            pn = (w[:, None] * np.where(avail, S, 0)).sum(axis=0)[okd] / wsum[okd]
                            pdates = dates[okd]
                            dd = np.array([abs(doy(x) - doy(di)) for x in pdates])
                            sea = np.minimum(dd, 365 - dd) <= 45
                            pairs_v = sorted({ents[k]['pair'] for k in np.where(v)[0]})
                            rec = dict(band=band, kind=kind, group=gname, W=W, side=side, event=ev,
                                       n_ent=int(v.sum()), n_pairs=len(pairs_v), step=net,
                                       n_plac=int(okd.sum()), n_plac_seas=int(sea.sum()),
                                       pairs=';'.join(pairs_v) if len(pairs_v) <= 12 else '')
                            if okd.sum() >= 20:
                                med = np.median(pn)
                                rec.update(plac_med=float(med), plac_sd=robust_sd(pn),
                                           plac_p2_5=float(np.percentile(pn, 2.5)),
                                           plac_p97_5=float(np.percentile(pn, 97.5)),
                                           pct_signed=float(np.mean(pn < net) * 100),
                                           p_two=float(np.mean(np.abs(pn - med) >= abs(net - med))))
                            if sea.sum() >= 10:
                                ps = pn[sea]
                                rec.update(seas_sd=float(np.std(ps)), seas_p2_5=float(np.percentile(ps, 2.5)),
                                           seas_p97_5=float(np.percentile(ps, 97.5)),
                                           seas_pct_signed=float(np.mean(ps < net) * 100))
                            if side == 'both':
                                bb = np.array([ents[k]['boot'][ie] for k in np.where(v)[0]])
                                ww = Wt[v]
                                vb = np.isfinite(bb)
                                den = (ww[:, None] * vb).sum(axis=0)
                                B = (ww[:, None] * np.where(vb, bb, 0)).sum(axis=0) / np.where(den > 0, den, np.nan)
                                if np.isfinite(B).sum() > 50:
                                    rec.update(boot_sd=float(np.nanstd(B)), boot_p2_5=float(np.nanpercentile(B, 2.5)),
                                               boot_p97_5=float(np.nanpercentile(B, 97.5)))
                                    boots[(band, kind, gname, W, ev)] = B
                                if okd.sum() >= 20:
                                    thr = np.percentile(pn, 5)
                                    J = np.array([x['inj'] for x in ents])          # (nent, ninj, ndates)
                                    for ii, dl in enumerate(INJ):
                                        Ji = J[:, ii, :]
                                        av = np.isfinite(Ji) & v[:, None]
                                        ws_ = (w[:, None] * av).sum(axis=0)[okd]
                                        ni = (w[:, None] * np.where(av, Ji, 0)).sum(axis=0)[okd] / np.maximum(ws_, 1e-30)
                                        rec[f'P_det_{dl * 100:.2f}'] = float(np.mean(ni < thr))
                                        rec[f'recov_{dl * 100:.2f}'] = float(np.median(ni - pn) / dl)
                                    P = [rec[f'P_det_{dl * 100:.2f}'] for dl in INJ]
                                    mags = np.abs(INJ) * 100
                                    Pm = np.maximum.accumulate(P)
                                    rec['det90_pct'] = float(np.interp(0.9, Pm, mags)) if Pm[-1] >= 0.9 else np.nan
                                    rec['det90_reached'] = bool(max(P) >= 0.9)
                            rows.append(rec)
    return pd.DataFrame(rows), boots


def backfit(D, Wt, n_iter=60):
    """D (nent, nt) with NaN, weights per entity -> common series s_t, offsets o_e, effective err."""
    o = np.zeros(D.shape[0])
    v = np.isfinite(D)
    for _ in range(n_iter):
        R = np.where(v, D - o[:, None], 0)
        wt = (Wt[:, None] * v).sum(axis=0)
        s = (Wt[:, None] * R).sum(axis=0) / np.where(wt > 0, wt, np.nan)
        s = s - np.nanmean(s)
        o = np.nansum(np.where(v, D - s[None, :], 0), axis=1) / np.maximum(v.sum(axis=1), 1)
    wt = (Wt[:, None] * v).sum(axis=0)
    err = 1 / np.sqrt(np.where(wt > 0, wt, np.nan))
    return s, o, err, v.sum(axis=0)


def network_series(E, data, nb=30, ref='full', side='both'):
    out = []
    for band in BANDS:
        for kind in ['BB', 'all']:
            if band == '0.05-0.1' and kind == 'all':
                continue
            for gname, comps in GROUPS.items():
                sel = select(E, band, kind, comps)
                Ds, Wt = [], []
                for _, e in sel.iterrows():
                    d = data[(e.pair, e.comp, band)]
                    k = f'bin{nb}_{ref}_{side}_dvv'
                    if k not in d:
                        continue
                    x = d[k].astype(float)
                    cc = d[f'bin{nb}_{ref}_{side}_cc'].astype(float)
                    x[cc < 0.3] = np.nan
                    sd = robust_sd(np.diff(x[np.isfinite(x)])) / np.sqrt(2)
                    if not np.isfinite(sd) or sd <= 0:
                        continue
                    Ds.append(x)
                    Wt.append(1 / sd ** 2)
                if not Ds:
                    continue
                s, o, err, n = backfit(np.array(Ds), np.array(Wt))
                t = DAY0 + pd.to_timedelta(np.arange(len(s)) * nb + nb / 2, unit='D')
                for i in range(len(s)):
                    if n[i] > 0:
                        out.append(dict(band=band, kind=kind, group=gname, ref=ref, side=side, t=t[i], dvv=s[i],
                                        err=err[i], n_ent=int(n[i])))
    return pd.DataFrame(out)


def fit_trend(df, nboot=500, block=12, seed=0):
    """WLS: c + b*yr + annual + semiannual + Heaviside steps (events) on a 30-d network series."""
    t = (pd.to_datetime(df.t) - pd.Timestamp('2008-01-01')).dt.days.values / 365.25
    y = df.dvv.values
    e = df.err.values
    cols = [np.ones_like(t), t - t.mean(), np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), np.sin(4 * np.pi * t),
            np.cos(4 * np.pi * t)]
    names = ['c', 'trend_per_yr', 's1', 'c1', 's2', 'c2']
    for ev, d in EVD.items():
        td = d / 365.25
        if (t < td).sum() >= 3 and (t > td).sum() >= 3:
            cols.append((t > td).astype(float))
            names.append(ev)
    G = np.array(cols).T
    w = 1 / e ** 2
    sw = np.sqrt(w)
    m, *_ = np.linalg.lstsq(G * sw[:, None], y * sw, rcond=None)
    r = y - G @ m
    rng = np.random.default_rng(seed)
    n = len(y)
    M = []
    for _ in range(nboot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, max(n - block, 1), n // block + 1)])[:n]
        yb = G @ m + r[idx % n]
        mb, *_ = np.linalg.lstsq(G * sw[:, None], yb * sw, rcond=None)
        M.append(mb)
    M = np.array(M)
    res = {}
    for i, nm in enumerate(names):
        res[nm] = float(m[i])
        res[nm + '_se'] = float(M[:, i].std())
    res['annual_amp'] = float(np.hypot(m[2], m[3]))
    res['semiannual_amp'] = float(np.hypot(m[4], m[5]))
    res['rms_resid'] = float(np.sqrt(np.mean(r ** 2)))
    res['n'] = int(n)
    res['t0'] = str(pd.to_datetime(df.t).min().date())
    res['t1'] = str(pd.to_datetime(df.t).max().date())
    return res


def main():
    E, data = load()
    E.to_csv(os.path.join(OUT, 'pair_qc.csv'), index=False)
    print('entities', len(E), 'ok', int(E.ok.sum()))
    print(E[E.ok].groupby(['band', 'kind', 'comp']).size().to_string())
    S, boots = event_steps(E, data)
    S.to_csv(os.path.join(OUT, 'steps.csv'), index=False)
    ser = []
    for nb in (30, 10):
        for side in ['both', 'caus', 'acaus']:
            ser.append(network_series(E, data, nb=nb, side=side).assign(nb=nb))
        ser.append(network_series(E, data, nb=nb, ref='pre').assign(nb=nb))
    SER = pd.concat(ser)
    SER[SER.nb == 30].to_csv(os.path.join(OUT, 'network_series_30d.csv'), index=False)
    SER[SER.nb == 10].to_csv(os.path.join(OUT, 'network_series_10d.csv'), index=False)
    TR = []
    for (band, kind, group, ref, side), g in SER[SER.nb == 30].groupby(['band', 'kind', 'group', 'ref', 'side']):
        g = g.sort_values('t')
        if len(g) < 24:
            continue
        r = fit_trend(g)
        r.update(band=band, kind=kind, group=group, ref=ref, side=side)
        # 2025 mean - 2021 mean
        yr = pd.to_datetime(g.t).dt.year
        if (yr == 2021).sum() >= 6 and (yr == 2025).sum() >= 6:
            r['d2025_2021'] = float(g.dvv[yr == 2025].mean() - g.dvv[yr == 2021].mean())
        TR.append(r)
    TR = pd.DataFrame(TR)
    TR.to_csv(os.path.join(OUT, 'trend.csv'), index=False)
    print(TR[(TR.side == 'both') & (TR.ref == 'full')][['band', 'kind', 'group', 'n', 't0', 'trend_per_yr', 'trend_per_yr_se', 'annual_amp']].to_string())


if __name__ == '__main__':
    main()
