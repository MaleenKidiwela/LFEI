#!/usr/bin/env python3
"""Stage C of the joint 3-D dv/v inversion: long-period inter-station (LP) and LFE-family observables.

Quantities: M7 (2024-12-05), Petrolia (2021-12-20), Ferndale (2022-12-20) steps, and D = 2021->2025 total change
(mean over 2025 minus mean over 2021, including the quake steps; same definition as the single-station SC data
of sci_dvv_3d_data.py, the direct-S and repeater analyses (decline() in sci_dvv_deep_shallow*.py)).

LP (N2): per pair x band (0.05-0.1, 0.1-0.2, 0.2-0.3 Hz; /data/wsd02/maleen_data/MTJ_LH/dvv/*.npz from
  sci_noise_lp_dvv.py).  Components that pass the LP QC (ndays >= 120, median 30-d CC >= 0.5; 0.05-0.1 Hz BB-BB
  only; as sci_noise_lp_net.py) are averaged.  Step = step30_both (30-d post stack minus 30-d pre stack, W = 30,
  the LP primary).  Season-matched null: the same statistic on the 3-day grid dates within +-30 days of the event's
  calendar date in all other years, >= W+10 d from any regional quake, no instrument window/break of either site in
  [d-W-1, d+W+1].  Datum = step - placebo mean, sigma = placebo SD x sqrt(1 + 1/n_years) (pooled band ratio if
  < 3 years).  D from the 30-d non-overlapping series (bin30_full_both_dvv): matched-bin 2025 - 2021, null = all
  (Y, Y+4) pairs with Y+4 <= 2020.
  Error correlation: per band, the mean correlation of placebo step series between pairs that share a site and
  between pairs that do not (network-common term) -> parametric covariance used by the inversion.  The same is
  computed for the single-station SC series (between stations).
LFE (L1-L6): see PLAN.md; values and honest errors read from the existing analyses (nothing re-measured).
Outputs: data/sci/dvv_3d/obs_lp.csv, obs_lfe.csv, err_corr.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_3d'
LPD = '/data/wsd02/maleen_data/MTJ_LH/dvv'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_dvv_3d_data import EXCL, BREAKS, load_series, all_dates, step_stat, BANDS as SC_BANDS  # noqa: E402

DAY0 = pd.Timestamp('2008-01-01')
NDAYS = (pd.Timestamp('2026-11-23') - DAY0).days + 1
GRID = np.arange(0, NDAYS, 3)
LP_EVENTS = ['M6.5_2010', 'M6.8_2014', 'Petrolia', 'Ferndale', 'M7']
EVD = {'M6.5_2010': '2010-01-10', 'M6.8_2014': '2014-03-10', 'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20',
       'M7': '2024-12-05', 'M6.5_2016': '2016-12-08'}
EVI = {k: (pd.Timestamp(v) - DAY0).days for k, v in EVD.items()}
W = 30
COMPS = ['ZZ', 'RR', 'TT', 'ZR', 'RZ']
LPB = ['0.05-0.1', '0.1-0.2', '0.2-0.3']
QUANT = ['M7', 'Petrolia', 'Ferndale']


def site_excl_mask(site, lo, hi):
    """True where [lo, hi] (day indices, arrays) overlaps an instrument window or break of the site."""
    key = [k for k in EXCL if k.split('.')[1] == site]
    m = np.zeros(np.broadcast(lo, hi).shape, bool)
    for k in key:
        for a, b in EXCL[k]:
            ia, ib = (pd.Timestamp(a) - DAY0).days, (pd.Timestamp(b) - DAY0).days
            m |= (hi >= ia) & (lo <= ib)
    for k in [k for k in BREAKS if k.split('.')[1] == site]:
        for b in BREAKS[k]:
            ib = (pd.Timestamp(b) - DAY0).days
            m |= (lo <= ib) & (hi >= ib)
    return m


def lp_obs():
    P = pd.read_csv(f'{ROOT}/data/sci/noise_lp/pairs.csv')
    rows, series = [], {b: {} for b in LPB}
    gdates = GRID
    cal = (DAY0 + pd.to_timedelta(gdates, 'D'))
    near = np.zeros(len(gdates), bool)
    for v in EVI.values():
        near |= np.abs(gdates - v) <= W + 10
    for fn in sorted(glob.glob(f'{LPD}/*.npz')):
        name = os.path.basename(fn)[:-4]
        a, b = name.split('_')
        pr = P[(P.a == a) & (P.b == b)].iloc[0]
        Z = np.load(fn)
        bad = site_excl_mask(a, gdates - W - 1, gdates + W + 1) | site_excl_mask(b, gdates - W - 1, gdates + W + 1)
        for band in LPB:
            S, S30 = [], []
            for c in COMPS:
                k = f'{c}|{band}|'
                if k + 'ndays' not in Z.files:
                    continue
                cc30 = Z[k + 'bin30_full_both_cc']
                medcc = np.nanmedian(cc30) if np.isfinite(cc30).any() else np.nan
                if int(Z[k + 'ndays']) < 120 or not (medcc >= 0.5):
                    continue
                if band == '0.05-0.1' and pr.kind != 'BB-BB':
                    continue
                S.append(Z[k + f'step{W}_both'].astype(float))
                S30.append(Z[k + 'bin30_full_both_dvv'].astype(float))
            if not S:
                continue
            with np.errstate(all='ignore'):
                s = np.nanmean(np.array(S), 0) * 100            # %
                s30 = np.nanmean(np.array(S30), 0) * 100
            sg = s[:len(gdates)].copy()
            sg[bad] = np.nan
            plc = sg.copy()
            plc[near] = np.nan
            series[band][name] = plc
            base = dict(pair=name, a=a, b=b, band=band, dist_km=pr.dist_km, kind=pr.kind, ncomp=len(S))
            for ie, ev in enumerate(LP_EVENTS):
                if ev not in QUANT:
                    continue
                v = s[len(gdates) + ie]
                ei = EVI[ev]
                if not np.isfinite(v) or site_excl_mask(a, ei - W - 1, ei + W + 1) or site_excl_mask(b, ei - W - 1, ei + W + 1):
                    continue
                ed = pd.Timestamp(EVD[ev])
                vals, yrs = [], []
                for yr in range(2008, 2027):
                    if yr == ed.year:
                        continue
                    try:
                        c0 = ed.replace(year=yr)
                    except ValueError:
                        continue
                    m = np.abs((cal - c0).days) <= 30
                    x = plc[m]
                    x = x[np.isfinite(x)]
                    vals += list(x)
                    yrs += [yr] * len(x)
                allsd = 1.4826 * np.nanmedian(np.abs(plc - np.nanmedian(plc))) if np.isfinite(plc).sum() > 20 else np.nan
                rows.append(dict(**base, quantity=ev, step=v, n_season=len(vals), n_years=len(set(yrs)),
                                 mu_season=np.mean(vals) if len(vals) > 5 else np.nan,
                                 sd_season=np.std(vals) if len(vals) > 5 else np.nan, sd_all=allsd))
            # decline from 30-d bins
            nb = len(s30)
            bt = DAY0 + pd.to_timedelta(np.arange(nb) * 30 + 15, 'D')
            bad30 = site_excl_mask(a, np.arange(nb) * 30, np.arange(nb) * 30 + 30) | site_excl_mask(b, np.arange(nb) * 30, np.arange(nb) * 30 + 30)
            s30[bad30] = np.nan

            def mdiff(y1, y2):
                i1 = np.where(bt.year == y1)[0]
                i2 = np.where(bt.year == y2)[0]
                n = min(len(i1), len(i2))
                x1, x2 = s30[i1[:n]], s30[i2[:n]]
                ok = np.isfinite(x1) & np.isfinite(x2)
                if ok.sum() < 4:
                    return np.nan
                # breaks between the years are excluded
                if site_excl_mask(a, (pd.Timestamp(f'{y1}-01-01') - DAY0).days, (pd.Timestamp(f'{y2}-12-31') - DAY0).days) & False:
                    return np.nan
                return x2[ok].mean() - x1[ok].mean()
            spans = any(pd.Timestamp('2021-01-01') <= pd.Timestamp(bb) <= pd.Timestamp('2025-12-31')
                        for k in BREAKS if k.split('.')[1] in (a, b) for bb in BREAKS[k])
            v = mdiff(2021, 2025)
            if np.isfinite(v) and not spans:
                nul = np.array([mdiff(y, y + 4) for y in range(2008, 2017)])
                nul = nul[np.isfinite(nul)]
                rows.append(dict(**base, quantity='D', step=v, n_season=len(nul), n_years=len(nul),
                                 mu_season=0.0, sd_season=np.std(nul) if len(nul) >= 4 else np.nan, sd_all=np.nan))
    T = pd.DataFrame(rows)
    for q in QUANT + ['D']:
        for band in LPB:
            m = (T.quantity == q) & (T.band == band)
            good = m & (T.n_years >= (3 if q != 'D' else 4)) & np.isfinite(T.sd_season)
            if q != 'D':
                ratio = np.nanmedian(T.loc[good, 'sd_season'] / T.loc[good, 'sd_all']) if good.sum() >= 3 else 1.0
                mu_med = np.nanmedian(T.loc[good, 'mu_season']) if good.sum() >= 3 else 0.0
                for i in T.index[m]:
                    r = T.loc[i]
                    if r.n_years >= 3 and np.isfinite(r.sd_season):
                        T.loc[i, ['sigma', 'mu', 'null_src']] = [r.sd_season * np.sqrt(1 + 1 / r.n_years), r.mu_season, 'own']
                    else:
                        T.loc[i, ['sigma', 'mu', 'null_src']] = [np.nanmax([ratio * r.sd_all, r.sd_season]), mu_med, 'pooled']
            else:
                med = np.nanmedian(T.loc[good, 'sd_season']) if good.any() else np.nan
                for i in T.index[m]:
                    r = T.loc[i]
                    T.loc[i, ['sigma', 'mu', 'null_src']] = ([r.sd_season, 0.0, 'own'] if (r.n_years >= 4 and np.isfinite(r.sd_season))
                                                             else [1.5 * med, 0.0, 'pooled'])
    T['d'] = T.step - T.mu
    T = T[np.isfinite(T.d) & np.isfinite(T.sigma) & (T.sigma > 0)]
    T.to_csv(f'{OUT}/obs_lp.csv', index=False, float_format='%.5g')
    print(T.groupby(['quantity', 'band']).agg(n=('pair', 'size'), med_d=('d', 'median'), med_sig=('sigma', 'median'),
                                              nz2=('d', lambda x: int((np.abs(x / T.loc[x.index, 'sigma']) > 2).sum()))).round(4))
    # correlation between pairs' placebo series
    corr = {}
    rng = np.random.default_rng(0)
    for band in LPB:
        names = list(series[band])
        X = np.array([series[band][n] for n in names])
        st = [set(n.split('_')) for n in names]
        sh, ns_ = [], []
        for _ in range(6000):
            i, j = rng.choice(len(names), 2, replace=False)
            ok = np.isfinite(X[i]) & np.isfinite(X[j])
            if ok.sum() < 200:
                continue
            r = np.corrcoef(X[i, ok], X[j, ok])[0, 1]
            (sh if st[i] & st[j] else ns_).append(r)
        corr[band] = dict(rho_share=float(np.median(sh)) if sh else 0.0, rho_net=float(np.median(ns_)) if ns_ else 0.0,
                          n_share=len(sh), n_net=len(ns_))
    return corr


def sc_corr():
    """median correlation between stations' all-season placebo step series per SC band (network-common error)."""
    S = load_series()
    dates = pd.date_range('2008-12-01', '2026-10-01', freq='5D')
    M = {}
    for sta, Y in S.items():
        X = np.full((len(dates), len(SC_BANDS)), np.nan)
        ok_dates = set(all_dates(sta, Y))
        for i, d in enumerate(dates):
            if d in ok_dates:
                X[i] = step_stat(Y, d)
        M[sta] = X
    out = {}
    names = list(M)
    for j, b in enumerate(SC_BANDS):
        rs = []
        for i1 in range(len(names)):
            for i2 in range(i1 + 1, len(names)):
                x, y = M[names[i1]][:, j], M[names[i2]][:, j]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 60:
                    rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
        out[b] = dict(rho_net=float(np.median(rs)) if rs else 0.0, n=len(rs))
    return out


def lfe_obs():
    rows = []
    fam = pd.read_csv(f'{ROOT}/data/families/shelly_family_locations.csv')
    ST = pd.read_csv(f'{OUT}/stations.csv').set_index('ns')
    # ---- L1 channel coda (family-averaged), honest SE
    C = pd.read_csv(f'{ROOT}/data/sci/dvv_source_receiver/chanfit_calibrated.csv')
    for r in C.itertuples():
        base = dict(dataset='L1', id=r.chseg, sta=r.sta, chan=r.chan, panel=bool(r.panel))
        vals = {}
        if np.isfinite(r.step_m7) and 0 < r.step_m7_se < 0.5:
            vals['M7'] = (r.step_m7, r.step_m7_se)
        for q, v, s in [('Petrolia', r.step_pet, r.step_pet_se_boot), ('Ferndale', r.step_fer, r.step_fer_se_boot)]:
            if np.isfinite(v) and 0 < s < 0.5 and abs(v) < 2:
                vals[q] = (v, 2.0 * s)            # placebo/bootstrap factor ~2 measured at the M7
        if all(q in vals for q in ['M7', 'Petrolia', 'Ferndale']) and np.isfinite(r.decl4yr) and 0 < r.decl_se < 1:
            v = 0.955 * r.decl4yr + vals['M7'][0] + vals['Petrolia'][0] + vals['Ferndale'][0]
            s = np.sqrt((0.955 * r.decl_se) ** 2 + vals['M7'][1] ** 2 + vals['Petrolia'][1] ** 2 + vals['Ferndale'][1] ** 2)
            vals['D'] = (v, s)
        for q, (v, s) in vals.items():
            rows.append(dict(**base, quantity=q, d=v, sigma=s))
    # ---- L2 family deviations (balanced panel)
    F = pd.read_csv(f'{ROOT}/data/sci/dvv_source_receiver/family_table_balanced.csv')
    for r in F.itertuples():
        base = dict(dataset='L2', id=f'F{r.family}', family=int(r.family))
        rows.append(dict(**base, quantity='M7', d=r.step_m7, sigma=r.step_m7_se))
        rows.append(dict(**base, quantity='Petrolia', d=r.step_pet, sigma=r.step_pet_se))
        rows.append(dict(**base, quantity='Ferndale', d=r.step_fer, sigma=r.step_fer_se))
        v = 0.955 * r.decl4yr + r.step_m7 + r.step_pet + r.step_fer
        s = np.sqrt((0.955 * r.decl4yr_se) ** 2 + r.step_m7_se ** 2 + r.step_pet_se ** 2 + r.step_fer_se ** 2)
        rows.append(dict(**base, quantity='D', d=v, sigma=s))
    # ---- L3 direct S per station (ms; origin term removed -> nuisance offset in the inversion)
    Dd = pd.read_csv(f'{ROOT}/data/sci/dvv_deep_shallow/station_directS_coda.csv')
    for r in Dd[Dd.X.isin(['m7', 'D'])].itertuples():
        se = np.nanmax([r.dS_se_boot_ms, r.dS_se_plac_ms])
        if not np.isfinite(r.dS_ms) or not (0 < se < 20) or r.n < 3:
            continue
        rows.append(dict(dataset='L3', id=r.sta, sta=r.sta, quantity='M7' if r.X == 'm7' else 'D', d=r.dS_ms, sigma=se,
                         T_S=r.T_S, sensor_km=r.sensor_km, n=r.n))
    # ---- L4 near-source S-P eps, 10-km sphere, receiver-corrected (primary set all_corr_ds_P0.5)
    T4 = pd.read_csv(f'{ROOT}/data/sci/dvv_source_region/timefit_reccorr.csv')
    T4 = T4[(T4.set == 'all_corr_ds_P0.5') & (T4.model == 'sph10') & (T4.param == 'eps0')]
    for q, k in [('M7', 'm7'), ('Petrolia', 'pet'), ('Ferndale', 'fer'), ('D', 'D')]:
        r = T4[T4.quantity == k].iloc[0]
        rows.append(dict(dataset='L4', id='sph10', quantity=q, d=r.val, sigma=r.se))
    # ---- L5 LFE-minus-earthquake 22-31 km layer (joint)
    s5 = json.load(open(f'{ROOT}/data/sci/eq_above/summary.json'))['diff']['joint']
    for q, k in [('M7', 'm7'), ('Petrolia', 'pet'), ('Ferndale', 'fer'), ('D', 'D')]:
        rows.append(dict(dataset='L5', id='layer22_31', quantity=q, d=s5[k]['val'], sigma=s5[k]['se']))
    # ---- L6 repeaters
    R = pd.read_csv(f'{ROOT}/data/sci/dvv_deep_shallow/rep/rep_fit.csv')
    for r in R.itertuples():
        base = dict(dataset='L6', id=f'{r.sta}|{r.cls}', sta=r.sta, cls=r.cls, depth_med=r.depth_med)
        if np.isfinite(r.m7_pct) and r.n_span_m7 >= 5:
            rows.append(dict(**base, quantity='M7', d=r.m7_pct, sigma=1.5 * r.m7_se_boot))
        if np.isfinite(r.D_pct):
            rows.append(dict(**base, quantity='D', d=r.D_pct, sigma=1.5 * r.D_se_boot))
    T = pd.DataFrame(rows)
    T = T[np.isfinite(T.d) & np.isfinite(T.sigma) & (T.sigma > 0)]
    T.to_csv(f'{OUT}/obs_lfe.csv', index=False, float_format='%.5g')
    print(T.groupby(['dataset', 'quantity']).agg(n=('d', 'size'), med=('d', 'median'), sig=('sigma', 'median')).round(4))


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    corr = json.load(open(f'{OUT}/err_corr.json')) if os.path.exists(f'{OUT}/err_corr.json') else {}
    if st in ('lfe', 'all'):
        lfe_obs()
    if st in ('lp', 'all'):
        corr['LP'] = lp_obs()
    if st in ('sccorr', 'all'):
        corr['SC'] = sc_corr()
    json.dump(corr, open(f'{OUT}/err_corr.json', 'w'), indent=1)
    print(json.dumps(corr, indent=1))
