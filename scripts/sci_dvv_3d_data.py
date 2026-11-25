#!/usr/bin/env python3
"""Data preparation for the 3-D relative dv/v inversion (scripts/sci_dvv_3d*.py).

Input: data/sci/noise_dvv_sc_full/sc_10day_back.csv (single-station cross-component (SC) dv/v, 10-day BACKWARD
moving stacks, 1-day step; value at day d = CCFs of days d-9..d), combo 'SC', bands 0.1-0.3, 0.3-0.6, 0.5-1, 1-3,
3-5 Hz (the full 0.1-1 Hz band is ignored: it follows ocean-microseism power).  dv/v is NOT recomputed.

1. Instrument-problem windows (NOTES.md: instrument check / shear-wave splitting; scripts/sci_dvv_instrument_check.py
   EXCL) are removed from the series: a 10-day value is dropped if its window [d-9, d] overlaps an excluded window.
   B045 EH1 (2009-10) is not removed because the SC value mixes all components; this only affects 2009-10 placebos.
   Instrument BREAKS (B933 2021-11-01, BRIC 2024-05-22, RVIT 2024-06-21, BJES 2024-05-23, KMPB 2023-10-17): no step
   or placebo statistic may span a break; long-term differences may not span one (KMPB kept as a variant).
2. Event step statistic (same as sc_steps.csv method '10day_back'): mean(days +1..+15) - mean(days -30..-1) of the
   10-day backward series (>=10 pre and >=5 post values).  Note: days +1..+9 of a backward series still contain
   pre-event days, so for a permanent step at day 0 the statistic is ~0.76 x the step.
3. SEASON-MATCHED NULL: placebo dates = event calendar date +-30 days (1-day step) in every OTHER year of the record,
   excluding dates whose statistic window [td-40, td+16] contains a real regional quake (2010-01-10 M6.5,
   2014-03-10 M6.8, 2016-12-08 M6.5 offshore, Petrolia, Ferndale, M7), an instrument window or a break.
   sigma_season = SD of the placebo statistics; mu_season = their mean (systematic Nov-Dec seasonal drift).
   Primary datum = step - mu_season ("season-corrected") with error sigma_season * sqrt(1 + 1/n_years).
   Stations with < 3 placebo years: sigma = max(own season SD, band ratio (season/all-season SD, median of the
   >=3-year stations) x own all-season SD); mu = band median of mu_season over >=3-year stations, its across-station
   SD added in quadrature.  The all-season null (every 5 days, all months) is also stored.
4. Long-term: mean over 2025 minus mean over 2021 using only calendar days (day of year) present in both years
   (removes the annual cycle for partial years).  Null: the same statistic for every (Y, Y+4) pair with
   Y+4 <= 2020; stations without >= 4 null pairs get the band median null SD scaled by their all-season placebo SD
   relative to the band median.  Pre-2021 stability observable: 2020 minus 2016 (null from the other pairs).
5. Placebo matrices (dates x bands) per station are stored so the inter-band error correlation can be estimated.

Outputs (small): data/sci/dvv_3d/obs_steps.csv, obs_longterm.csv, obs_frames.csv, placebo_corr.json, stations.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_3d'
SRC = f'{ROOT}/data/sci/noise_dvv_sc_full'
BANDS = ['0.1-0.3Hz', '0.3-0.6Hz', '0.5-1Hz', '1-3Hz', '3-5Hz']
EVENTS = {'M7': '2024-12-05', 'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20'}
OTHER_Q = {'M6.5_2010': '2010-01-10', 'M6.8_2014': '2014-03-10', 'M6.5_2016': '2016-12-08'}
ALLQ = {**EVENTS, **OTHER_Q}
EXCL = {'PB.B933': [('2017-01-01', '2021-11-01'), ('2024-08-01', '2030-01-01')],
        'PB.B935': [('2024-05-01', '2030-01-01')],
        'PB.B047': [('2000-01-01', '2009-03-01')],
        'NC.KMPB': [('2000-01-01', '2010-11-03')],
        'NC.KMR': [('2000-01-01', '2009-09-17'), ('2025-09-17', '2030-01-01')],
        'BK.BRIC': [('2022-12-01', '2024-05-22')],
        'BK.RVIT': [('2022-12-20', '2024-06-21')],
        'BK.BJES': [('2022-12-20', '2024-05-23')]}
BREAKS = {'PB.B933': ['2021-11-01'], 'BK.BRIC': ['2024-05-22'], 'BK.RVIT': ['2024-06-21'],
          'BK.BJES': ['2024-05-23'], 'NC.KMPB': ['2023-10-17']}
PRE, POST = (-30, -1), (1, 15)
POST_FULL = (10, 24)        # variant: post window whose 10-day stacks are entirely post-event


def stations():
    c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv')
    c['ns'] = c.net + '.' + c.sta
    c = c[c.cha.str[:2].isin(['HH', 'EH', 'BH'])]
    # sensor depth used 2008-2025 (KMR posthole 71 m only from 2026)
    c = c[c.start < '2025-09-01']
    g = c.groupby('ns').agg(lat=('lat', 'median'), lon=('lon', 'median'), elev=('elev', 'median'),
                            depth_m=('depth', 'median')).reset_index()
    g['z0_km'] = np.where(g.ns.str.startswith('PB.'), g.depth_m / 1000.0, 0.0)
    return g


def load_series():
    x = pd.read_csv(f'{SRC}/sc_10day_back.csv', usecols=['t', 'band', 'combo', 'dvv_pct', 'sta'])
    x = x[(x.combo == 'SC') & x.band.isin(BANDS)]
    x['t'] = pd.to_datetime(x.t)
    P = x.pivot_table(index=['sta', 't'], columns='band', values='dvv_pct')
    S = {}
    for sta, g in P.groupby(level=0):
        g = g.droplevel(0).reindex(columns=BANDS)
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq='D'))
        for a, b in EXCL.get(sta, []):
            m = (g.index >= pd.Timestamp(a)) & (g.index <= pd.Timestamp(b) + pd.Timedelta(days=9))
            g.loc[m] = np.nan
        S[sta] = g
    return S


def _blocked(sta, td, lo=-40, hi=16, quakes=ALLQ):
    t0, t1 = td + pd.Timedelta(days=lo), td + pd.Timedelta(days=hi)
    if any(t0 <= pd.Timestamp(q) <= t1 for q in quakes.values()):
        return True
    if any(t0 <= pd.Timestamp(b) <= t1 for b in BREAKS.get(sta, [])):
        return True
    return False


def step_stat(Y, td, post=POST):
    """Y: DataFrame (days x bands); returns (nb,) step statistic (NaN where insufficient data)."""
    i = Y.index.get_indexer([td])[0]
    if i < 0:
        rel = (Y.index - td).days
    else:
        rel = np.arange(len(Y)) - i
    pre = Y.values[(rel >= PRE[0]) & (rel <= PRE[1])]
    po = Y.values[(rel >= post[0]) & (rel <= post[1])]
    npre, npo = np.isfinite(pre).sum(0), np.isfinite(po).sum(0)
    with np.errstate(all='ignore'):
        s = np.nanmean(po, 0) - np.nanmean(pre, 0)
    s[(npre < 10) | (npo < 5)] = np.nan
    return s


def season_dates(sta, Y, ev, date):
    d0 = pd.Timestamp(date)
    out = []
    for yr in range(Y.index.min().year - 1, Y.index.max().year + 2):
        if yr == d0.year:
            continue
        try:
            c = d0.replace(year=yr)
        except ValueError:
            continue
        for k in range(-30, 31):
            td = c + pd.Timedelta(days=k)
            if td - pd.Timedelta(days=40) < Y.index.min() or td + pd.Timedelta(days=16) > Y.index.max():
                continue
            if _blocked(sta, td):
                continue
            out.append((td, yr))
    return out


def all_dates(sta, Y):
    ds = pd.date_range(Y.index.min() + pd.Timedelta(days=41), Y.index.max() - pd.Timedelta(days=17), freq='5D')
    return [d for d in ds if not _blocked(sta, d)]


def matched_diff(Y, y1, y2):
    a = Y[Y.index.year == y1].copy()
    b = Y[Y.index.year == y2].copy()
    a.index = np.minimum(a.index.day_of_year, 365)
    b.index = np.minimum(b.index.day_of_year, 365)
    a = a.groupby(level=0).mean()
    b = b.groupby(level=0).mean()
    J = a.index.intersection(b.index)
    out, n = np.full(len(BANDS), np.nan), np.zeros(len(BANDS), int)
    for j in range(len(BANDS)):
        m = np.isfinite(a.values[a.index.get_indexer(J), j]) & np.isfinite(b.values[b.index.get_indexer(J), j])
        n[j] = m.sum()
        if m.sum() >= 90:
            out[j] = b.values[b.index.get_indexer(J), j][m].mean() - a.values[a.index.get_indexer(J), j][m].mean()
    return out, n


def spans_break(sta, y1, y2):
    return any(pd.Timestamp(f'{y1}-01-01') <= pd.Timestamp(b) <= pd.Timestamp(f'{y2}-12-31') for b in BREAKS.get(sta, []))


def main():
    os.makedirs(OUT, exist_ok=True)
    ST = stations()
    S = load_series()
    ST = ST[ST.ns.isin(S)].reset_index(drop=True)
    ST.to_csv(f'{OUT}/stations.csv', index=False, float_format='%.6g')
    rows, plac_store = [], {}
    for sta, Y in S.items():
        dall = all_dates(sta, Y)
        Pall = np.array([step_stat(Y, d) for d in dall]) if dall else np.full((0, 5), np.nan)
        sd_all = np.nanstd(Pall, 0) if len(Pall) else np.full(5, np.nan)
        for ev, date in EVENTS.items():
            td = pd.Timestamp(date)
            if td < Y.index.min() + pd.Timedelta(days=40) or td > Y.index.max() - pd.Timedelta(days=16):
                continue
            blocked_break = any(td - pd.Timedelta(days=40) <= pd.Timestamp(b) <= td + pd.Timedelta(days=16)
                                for b in BREAKS.get(sta, []))
            st = step_stat(Y, td)
            stf = step_stat(Y, td, POST_FULL)
            dsy = season_dates(sta, Y, ev, date)
            ds = [d for d, _ in dsy]
            Ps = np.array([step_stat(Y, d) for d in ds]) if ds else np.full((0, 5), np.nan)
            yrs = np.array([y for _, y in dsy])
            Psf = np.array([step_stat(Y, d, POST_FULL) for d in ds]) if ds else np.full((0, 5), np.nan)
            plac_store[(sta, ev)] = (ds, Ps)
            for j, b in enumerate(BANDS):
                ok = np.isfinite(Ps[:, j]) if len(Ps) else np.zeros(0, bool)
                ny = len(np.unique(yrs[ok])) if ok.any() else 0
                okf = np.isfinite(Psf[:, j]) if len(Psf) else np.zeros(0, bool)
                rows.append(dict(sta=sta, event=ev, band=b, step_pct=st[j], step_fullpost_pct=stf[j],
                                 blocked_break=blocked_break,
                                 mu_season=np.mean(Ps[ok, j]) if ok.sum() > 5 else np.nan,
                                 sd_season=np.std(Ps[ok, j]) if ok.sum() > 5 else np.nan,
                                 mu_season_fp=np.mean(Psf[okf, j]) if okf.sum() > 5 else np.nan,
                                 sd_season_fp=np.std(Psf[okf, j]) if okf.sum() > 5 else np.nan,
                                 n_season=int(ok.sum()), n_years=ny, sd_all=sd_all[j],
                                 n_all=int(np.isfinite(Pall[:, j]).sum()) if len(Pall) else 0))
    T = pd.DataFrame(rows)
    T = T[np.isfinite(T.step_pct) & ~T.blocked_break].copy()
    # fallbacks for short records
    for ev in EVENTS:
        for b in BANDS:
            m = (T.event == ev) & (T.band == b)
            good = m & (T.n_years >= 3)
            ratio = np.nanmedian(T.loc[good, 'sd_season'] / T.loc[good, 'sd_all']) if good.sum() >= 3 else 1.0
            mu_med = np.nanmedian(T.loc[good, 'mu_season']) if good.sum() >= 3 else 0.0
            mu_sd = np.nanstd(T.loc[good, 'mu_season']) if good.sum() >= 3 else 0.0
            ratio_fp = np.nanmedian(T.loc[good, 'sd_season_fp'] / T.loc[good, 'sd_all']) if good.sum() >= 3 else 1.0
            mu_med_fp = np.nanmedian(T.loc[good, 'mu_season_fp']) if good.sum() >= 3 else 0.0
            T.loc[m, 'season_ratio_band'] = ratio
            for i in T.index[m]:
                r = T.loc[i]
                if r.n_years >= 3:
                    sig = r.sd_season * np.sqrt(1 + 1.0 / r.n_years)
                    mu, src = r.mu_season, 'own'
                    sig_fp = r.sd_season_fp * np.sqrt(1 + 1.0 / r.n_years)
                    mu_fp = r.mu_season_fp
                else:
                    cand = [ratio * r.sd_all]
                    if np.isfinite(r.sd_season):
                        cand.append(r.sd_season)
                    sig = np.sqrt(np.nanmax(cand) ** 2 + mu_sd ** 2)
                    mu, src = mu_med, 'pooled'
                    cand = [ratio_fp * r.sd_all] + ([r.sd_season_fp] if np.isfinite(r.sd_season_fp) else [])
                    sig_fp = np.sqrt(np.nanmax(cand) ** 2 + mu_sd ** 2)
                    mu_fp = mu_med_fp
                T.loc[i, ['sigma', 'mu', 'null_src', 'sigma_fp', 'mu_fp']] = [sig, mu, src, sig_fp, mu_fp]
    T['d_corr'] = T.step_pct - T.mu
    T['d_corr_fp'] = T.step_fullpost_pct - T.mu_fp
    T['z_corr'] = T.d_corr / T.sigma
    T.to_csv(f'{OUT}/obs_steps.csv', index=False, float_format='%.5g')
    # inter-band correlation of season placebos (pooled over stations & events)
    Z = []
    for (sta, ev), (ds, Ps) in plac_store.items():
        if len(Ps) > 20:
            Zs = (Ps - np.nanmean(Ps, 0)) / np.nanstd(Ps, 0)
            Z.append(Zs)
    Z = np.concatenate(Z)
    C = pd.DataFrame(Z, columns=BANDS).corr(min_periods=100).values
    json.dump(dict(bands=BANDS, corr=C.round(4).tolist()), open(f'{OUT}/placebo_corr.json', 'w'), indent=1)

    # ---------------------------------------------------------------- long-term
    L = []
    for sta, Y in S.items():
        nulls = []
        for y in range(2008, 2017):
            if spans_break(sta, y, y + 4):
                continue
            v, n = matched_diff(Y, y, y + 4)
            nulls.append(v)
        nulls = np.array(nulls) if nulls else np.full((0, 5), np.nan)
        for (y1, y2, kind) in [(2021, 2025, 'LT'), (2016, 2020, 'PRE')]:
            v, n = matched_diff(Y, y1, y2)
            brk = spans_break(sta, y1, y2)
            for j, b in enumerate(BANDS):
                if kind == 'PRE':
                    nv = nulls[:, j] if len(nulls) else np.array([])
                    # leave out the 2016-2020 pair itself
                    nv = np.array([nulls[k, j] for k, y in enumerate(range(2008, 2017)) if k < len(nulls) and y != 2016]) if len(nulls) else nv
                else:
                    nv = nulls[:, j] if len(nulls) else np.array([])
                nv = nv[np.isfinite(nv)]
                L.append(dict(sta=sta, kind=kind, band=b, d_pct=v[j], n_days=n[j], spans_break=brk,
                              null_sd=np.std(nv) if len(nv) >= 4 else np.nan, null_n=len(nv)))
    L = pd.DataFrame(L)
    sdall = T.groupby(['sta', 'band']).sd_all.first()
    L['sd_all'] = [sdall.get((s, b), np.nan) for s, b in zip(L.sta, L.band)]
    for kind in ['LT', 'PRE']:
        for b in BANDS:
            m = (L.kind == kind) & (L.band == b)
            good = m & np.isfinite(L.null_sd)
            med = np.nanmedian(L.loc[good, 'null_sd'])
            medall = np.nanmedian(L.loc[good, 'sd_all'])
            for i in L.index[m]:
                if np.isfinite(L.loc[i, 'null_sd']):
                    L.loc[i, ['sigma', 'null_src']] = [L.loc[i, 'null_sd'], 'own']
                else:
                    sc = L.loc[i, 'sd_all'] / medall if np.isfinite(L.loc[i, 'sd_all']) else 1.5
                    L.loc[i, ['sigma', 'null_src']] = [med * max(sc, 1.0), 'pooled']
    L = L[np.isfinite(L.d_pct)]
    L.to_csv(f'{OUT}/obs_longterm.csv', index=False, float_format='%.5g')

    # ---------------------------------------------------------------- time frames around the M7
    # frame k = mean of the 10-day series over [d_k, d_k+29] minus mean over the pre window 2024-10-26..11-24
    # (all fully pre-event; 10-day stacks of the pre window end 2024-11-24); null = same calendar windows shifted
    # by whole years (season-matched by construction), break/quake-free only.
    pre0, pre1 = pd.Timestamp('2024-10-26'), pd.Timestamp('2024-11-24')
    starts = pd.date_range('2024-09-01', '2025-11-01', freq='MS')
    F = []
    for sta, Y in S.items():
        for s0 in starts:
            s1 = s0 + pd.Timedelta(days=29)
            vals = []
            for sh in range(-16, 3):
                a0, a1 = pre0 + pd.DateOffset(years=sh), pre1 + pd.DateOffset(years=sh)
                b0, b1 = s0 + pd.DateOffset(years=sh), s1 + pd.DateOffset(years=sh)
                lo, hi = min(a0, b0) - pd.Timedelta(days=10), max(a1, b1)
                if sh != 0 and (any(lo <= pd.Timestamp(q) <= hi for q in ALLQ.values())
                                or any(lo <= pd.Timestamp(q) <= hi for q in BREAKS.get(sta, []))):
                    continue
                if sh == 0 and any(lo <= pd.Timestamp(q) <= hi for q in BREAKS.get(sta, [])):
                    vals.append((sh, np.full(5, np.nan)))
                    continue
                A = Y[(Y.index >= a0) & (Y.index <= a1)]
                B = Y[(Y.index >= b0) & (Y.index <= b1)]
                if len(A) < 10 or len(B) < 10:
                    continue
                with np.errstate(all='ignore'):
                    v = B.mean().values - A.mean().values
                v[(A.notna().sum().values < 10) | (B.notna().sum().values < 10)] = np.nan
                vals.append((sh, v))
            obs = [v for sh, v in vals if sh == 0]
            if not obs:
                continue
            nul = np.array([v for sh, v in vals if sh != 0]) if any(sh != 0 for sh, _ in vals) else np.full((0, 5), np.nan)
            for j, b in enumerate(BANDS):
                nv = nul[:, j][np.isfinite(nul[:, j])] if len(nul) else np.array([])
                F.append(dict(sta=sta, frame=s0.strftime('%Y-%m'), band=b, d_pct=obs[0][j],
                              null_mu=nv.mean() if len(nv) >= 3 else np.nan, null_sd=nv.std() if len(nv) >= 3 else np.nan,
                              null_n=len(nv)))
    F = pd.DataFrame(F)
    F = F[np.isfinite(F.d_pct)]
    for fr in F.frame.unique():
        for b in BANDS:
            m = (F.frame == fr) & (F.band == b)
            good = m & np.isfinite(F.null_sd)
            med_sd = np.nanmedian(F.loc[good, 'null_sd']) if good.any() else np.nan
            med_mu = np.nanmedian(F.loc[good, 'null_mu']) if good.any() else 0.0
            F.loc[m, 'sigma'] = np.where(np.isfinite(F.loc[m, 'null_sd']), F.loc[m, 'null_sd'], 1.5 * med_sd)
            F.loc[m, 'mu'] = np.where(np.isfinite(F.loc[m, 'null_mu']), F.loc[m, 'null_mu'], med_mu)
    F['d_corr'] = F.d_pct - F.mu
    F.to_csv(f'{OUT}/obs_frames.csv', index=False, float_format='%.5g')

    # ---------------------------------------------------------------- check against sc_steps.csv
    ref = pd.read_csv(f'{SRC}/sc_steps.csv')
    ref = ref[ref.method.str.startswith('10day_back')][['sta', 'band', 'event', 'step_pct', 'placebo_sd_pct']]
    M = T.merge(ref, on=['sta', 'band', 'event'], suffixes=('', '_ref'))
    dif = (M.step_pct - M.step_pct_ref).abs()
    print('check vs sc_steps 10day_back: n', len(M), 'max |diff|', dif.max(), 'n diff>1e-4', (dif > 1e-4).sum())
    print(M[dif > 1e-4][['sta', 'band', 'event', 'step_pct', 'step_pct_ref']].to_string())
    print(T.groupby(['event', 'band']).agg(n=('sta', 'size'), med_step=('step_pct', 'median'),
                                           med_mu=('mu', 'median'), med_sig=('sigma', 'median'),
                                           med_sd_all=('sd_all', 'median'), ratio=('season_ratio_band', 'first'),
                                           nz3=('z_corr', lambda z: int((z.abs() > 3).sum()))).round(4).to_string())
    print(L.groupby(['kind', 'band']).agg(n=('sta', 'size'), med=('d_pct', 'median'), sig=('sigma', 'median')).round(4))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
