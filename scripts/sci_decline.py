#!/usr/bin/env python3
"""What causes the post-2021 coda dv/v decline?  Discriminating tests, all placebo-calibrated.

Inputs: data/sci/decline/famchan_yearly.csv (scripts/sci_decline_famchan.py: per family x channel yearly levels
refit from the original CWIRE pairs), pooled per-channel curves data/dvv/pool/*_curve.csv, station metadata
data/stations/candidate_channels_65km_2008_2026.csv + NCEDC channel text (data/sci/decline/raw/meta_*.txt),
NOAA GHCN-Daily PRCP data/sci/decline/raw/ghcn_*.csv, family locations data/families/shelly_family_locations.csv.

 1. Station vs family (path/source): D = x(2025) - x(2021) per family x channel; robust two-way decomposition
    D = mu + station + family; F statistics, variance components, split-half reproducibility of family effects
    (PB stations A vs B) and station effects (odd vs even families), within-station slope vs family-station
    distance; all compared with the same statistics for the 8 pre-2021 4-yr windows (y -> y+4, y = 2009..2016).
 2. Spatial: per-channel decline (pooled curves, yearly means, as sci_dvv_placebo.longterm) vs station attributes,
    Spearman vs the same correlations in pre-2021 placebo windows.
 3. Timing: deseasoned monthly group-median curves; hinge onset scan; step-only vs gradual models; half-year means.
 4. Hydrology: exponentially filtered precipitation anomaly h_tau; fits on monthly group medians; pre-2021-only fits,
    and the coefficient REQUIRED to explain the decline -> its predicted pre-2021 4-yr changes vs observed.
 5. Instrument metadata epochs 2019-2026 for the contributing stations; step tests at epoch dates.
Outputs data/sci/decline/*.csv, summary.json; figures/sci_decline_{family,timing,hydro}.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = 'data/sci/decline'
Q = {'pet': pd.Timestamp('2021-12-20T20:10'), 'fer': pd.Timestamp('2022-12-20T10:34'), 'm7': pd.Timestamp('2024-12-05T18:44')}
QLOC = {'pet': (40.3902, -124.2980), 'fer': (40.5250, -124.4230), 'm7': (40.3740, -125.0217)}
PLACEBO_Y = list(range(2009, 2017))
LONG_SURF = {'NC.KMPB', 'NC.KMR'}
rng = np.random.default_rng(20261122)
S = {}


def km(lat1, lon1, lat2, lon2):
    la = np.radians((lat1 + lat2) / 2)
    return 111.19 * np.hypot(lat1 - lat2, (lon1 - lon2) * np.cos(la))


def sta_of(ch):
    return ch.rsplit('.', 2)[0]


def zp(x, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return dict(value=float(x), null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                z=float((x - null.mean()) / null.std(ddof=1)), n_null=int(len(null)),
                null_max=float(null.max()), null_min=float(null.min()))


# ----------------------------------------------------------------------------------------------- 1 station vs family
def twoway(d):
    """Huber-IRLS additive fit D = mu + s_sta + f_fam (sum-to-zero by lstsq min-norm then centring)."""
    sta = d.sta.to_numpy(dtype=str)
    fam = d.fam.to_numpy()
    us, uf = np.unique(sta), np.unique(fam)
    Xs = (sta[:, None] == us[None]).astype(float)
    Xf = (fam[:, None] == uf[None]).astype(float)
    y = d.D.values
    w = np.ones(len(y))

    def fit(X, w):
        sw = np.sqrt(w)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        return b, y - X @ b

    X = np.hstack([np.ones((len(y), 1)), Xs, Xf])
    for _ in range(6):
        b, r = fit(X, w)
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        u = np.abs(r) / (2 * s)
        w = np.where(u <= 1, 1.0, 1 / u)
    rss_full = np.sum(w * r ** 2)
    rss_s = np.sum(w * fit(np.hstack([np.ones((len(y), 1)), Xs]), w)[1] ** 2)
    rss_f = np.sum(w * fit(np.hstack([np.ones((len(y), 1)), Xf]), w)[1] ** 2)
    rss_0 = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    dfr = len(y) - len(us) - len(uf) + 1
    F_fam = ((rss_s - rss_full) / (len(uf) - 1)) / (rss_full / dfr)
    F_sta = ((rss_f - rss_full) / (len(us) - 1)) / (rss_full / dfr)
    se = b[1:1 + len(us)]
    fe = b[1 + len(us):]
    se, fe = se - se.mean(), fe - fe.mean()
    return dict(F_fam=F_fam, F_sta=F_sta, sd_sta=float(se.std(ddof=1)), sd_fam=float(fe.std(ddof=1)),
                resid_sd=float(np.sqrt(rss_full / dfr)), frac_fam=(rss_s - rss_full) / rss_0, frac_sta=(rss_f - rss_full) / rss_0,
                n=len(y), n_sta=len(us), n_fam=len(uf)), pd.Series(se, us), pd.Series(fe, uf)


def window_D(W, N, y0, y1, chans_ok=None, min_st=2):
    ok = W[y0].notna() & W[y1].notna() & (N[y0] >= min_st) & (N[y1] >= min_st)
    d = (W.loc[ok, y1] - W.loc[ok, y0]).rename('D').reset_index()
    d['sta'] = d.chan.map(sta_of)
    if chans_ok is not None:
        d = d[d.chan.isin(chans_ok)]
    # keep stations with >= 3 families and families with >= 3 stations (iterate once)
    for _ in range(2):
        d = d[d.groupby('sta').fam.transform('nunique') >= 3]
        d = d[d.groupby('fam').sta.transform('nunique') >= 3]
    return d


def family_station_test():
    Y = pd.read_csv(f'{OUT}/famchan_yearly.csv')
    W = Y.pivot_table(index=['fam', 'chan'], columns='year', values='x_pct')
    N = Y.pivot_table(index=['fam', 'chan'], columns='year', values='n_stacks').fillna(0)
    fl = pd.read_csv('data/families/shelly_family_locations.csv').set_index('idnum')
    stc = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv').assign(ns=lambda x: x.net + '.' + x.sta) \
        .groupby('ns')[['lat', 'lon']].first()
    PBA, PBB = {'PB.B045', 'PB.B046', 'PB.B047', 'PB.B049'}, {'PB.B932', 'PB.B933', 'PB.B934', 'PB.B935'}
    allch = W.index.get_level_values('chan').unique()
    long_ch = [c for c in allch if c.startswith('PB.') or sta_of(c) in LONG_SURF]
    res, per = [], {}
    for setname, chs in [('long_record', long_ch), ('PB_only', [c for c in allch if c.startswith('PB.')]), ('all', None)]:
        for y0 in PLACEBO_Y + [2021]:
            y1 = y0 + 4
            d = window_D(W, N, y0, y1, chs)
            if len(d) < 30:
                continue
            r, se, fe = twoway(d)
            # split-half family effects: PB group A vs B
            fa = d[d.sta.isin(PBA)].groupby('fam').D.median()
            fb = d[d.sta.isin(PBB)].groupby('fam').D.median()
            j = fa.index.intersection(fb.index)
            r['rho_fam_PBA_PBB'] = stats.spearmanr(fa[j], fb[j])[0] if len(j) >= 5 else np.nan
            # split-half station effects: odd vs even families
            so = d[d.fam % 2 == 1].groupby('sta').D.median()
            sv = d[d.fam % 2 == 0].groupby('sta').D.median()
            j = so.index.intersection(sv.index)
            r['rho_sta_odd_even'] = stats.spearmanr(so[j], sv[j])[0] if len(j) >= 5 else np.nan
            # within-station slope vs family-station distance (station fixed effects)
            dist = km(fl.loc[d.fam, 'lat'].values, fl.loc[d.fam, 'lon'].values, stc.loc[d.sta, 'lat'].values, stc.loc[d.sta, 'lon'].values)
            dd = d.D - d.groupby('sta').D.transform('median')
            xx = dist - pd.Series(dist, index=d.index).groupby(d.sta).transform('mean').values
            m = np.abs(dd) < 5 * 1.4826 * np.median(np.abs(dd))
            r['slope_dist_pct_per_10km'] = float(np.polyfit(xx[m], dd[m], 1)[0] * 10)
            # family effect vs family location
            for k in ['lat', 'lon', 'depth_km']:
                r[f'rho_fam_{k}'] = stats.spearmanr(fe.values, fl.loc[fe.index, k].values)[0]
            r['median_D'] = float(d.D.median())
            r.update(set=setname, y0=y0, y1=y1)
            res.append(r)
            if y0 == 2021:
                per[setname] = (d, se, fe)
    R = pd.DataFrame(res)
    R.to_csv(f'{OUT}/famsta_decomposition.csv', index=False)
    out = {}
    stats_cols = ['F_fam', 'F_sta', 'sd_fam', 'sd_sta', 'frac_fam', 'frac_sta', 'resid_sd', 'rho_fam_PBA_PBB', 'rho_sta_odd_even',
                  'slope_dist_pct_per_10km', 'rho_fam_lat', 'rho_fam_lon', 'rho_fam_depth_km', 'median_D']
    for s_, g in R.groupby('set'):
        c = g[g.y0 == 2021]
        p = g[g.y0 < 2021]
        if len(c) == 0 or len(p) < 4:
            continue
        out[s_] = {k: zp(c[k].iloc[0], p[k].values) for k in stats_cols}
        out[s_]['n_rows'] = int(c.n.iloc[0])
        out[s_]['n_sta'] = int(c.n_sta.iloc[0])
        out[s_]['n_fam'] = int(c.n_fam.iloc[0])
    # detection limit: inject random family effects N(0, sigma) into the claim-window rows; power to exceed the
    # largest placebo F_fam and the largest placebo PB-A vs PB-B family split-half rho
    inj = []
    for s_ in ['long_record', 'PB_only']:
        d, _, _ = per[s_]
        g = R[(R.set == s_) & (R.y0 < 2021)]
        fmax, rmax = g.F_fam.max(), g.rho_fam_PBA_PBB.max()
        for sig in [0.01, 0.02, 0.03, 0.04, 0.06, 0.08]:
            hitF = hitR = 0
            for _ in range(60):
                eff = pd.Series(rng.normal(0, sig, d.fam.nunique()), index=np.unique(d.fam))
                dd = d.assign(D=d.D + eff.reindex(d.fam).values)
                r_, _, _ = twoway(dd)
                fa = dd[dd.sta.isin(PBA)].groupby('fam').D.median()
                fb = dd[dd.sta.isin(PBB)].groupby('fam').D.median()
                j = fa.index.intersection(fb.index)
                hitF += r_['F_fam'] > fmax
                hitR += stats.spearmanr(fa[j], fb[j])[0] > rmax
            inj.append(dict(set=s_, sigma_fam_pct=sig, power_F=hitF / 60, power_rho=hitR / 60))
    inj = pd.DataFrame(inj)
    inj.to_csv(f'{OUT}/famsta_injection_power.csv', index=False)
    out['injection_power'] = inj.to_dict('records')
    # per-station and per-family effects in the claim window (long-record set)
    d, se, fe = per['long_record']
    sta_tab = d.groupby('sta').D.agg(['median', 'count', lambda x: x.std(ddof=1)]).rename(columns={'<lambda_0>': 'sd'})
    sta_tab.to_csv(f'{OUT}/famsta_station_effects.csv')
    fam_tab = d.groupby('fam').D.agg(['median', 'count']).join(fl)
    fam_tab['effect'] = fe
    fam_tab.to_csv(f'{OUT}/famsta_family_effects.csv')
    S['family_vs_station'] = out
    return R, per


# ----------------------------------------------------------------------------------------------- pooled curves
def load_curves():
    C = {}
    for f in sorted(glob.glob('data/dvv/pool/*_curve.csv')):
        ch = f.split('/')[-1].replace('_curve.csv', '')
        c = pd.read_csv(f)
        c['t'] = pd.to_datetime(c.t)
        C[ch] = c[np.isfinite(c.dvv_pct)].reset_index(drop=True)
    return C


def yearly(C):
    Y = {ch: c[c.t.dt.year.between(2009, 2025)].groupby(c.t.dt.year).dvv_pct.mean() for ch, c in C.items()}
    return pd.DataFrame(Y).T


# ----------------------------------------------------------------------------------------------- 2 spatial pattern
def spatial(C):
    Y = yearly(C)
    meta = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv').assign(ns=lambda x: x.net + '.' + x.sta)
    st = meta.groupby('ns')[['lat', 'lon', 'elev', 'depth', 'dist_km']].first()
    pgv = pd.read_csv('data/sci/dvv_depth/pgv_by_station.csv').pivot_table(index='sta', columns='quake', values='pgv_gm_cms')
    rows = []
    for ch in Y.index:
        s = sta_of(ch)
        r = dict(chan=ch, sta=s, group='borehole' if ch.startswith('PB.') else 'surface',
                 D_2021_2025=Y.loc[ch, 2025] - Y.loc[ch, 2021] if 2021 in Y and np.isfinite(Y.loc[ch, 2021]) else np.nan,
                 D_1920_2025=Y.loc[ch, 2025] - Y.loc[ch, [2019, 2020]].mean(),
                 rec_start=C[ch].t.min().year + C[ch].t.min().dayofyear / 365.25, comp=ch[-1])
        r.update(st.loc[s].to_dict())
        for k, (la, lo) in QLOC.items():
            r[f'dist_{k}_km'] = km(st.loc[s, 'lat'], st.loc[s, 'lon'], la, lo)
        r['dist_coast_proxy_lon'] = st.loc[s, 'lon']
        r['log_pgv_sum'] = np.log10(pgv.loc[s].sum()) if s in pgv.index else np.nan
        for y0 in PLACEBO_Y:
            r[f'P_{y0}'] = Y.loc[ch, y0 + 4] - Y.loc[ch, y0]
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/spatial_per_channel.csv', index=False)
    attrs = ['lat', 'lon', 'elev', 'depth', 'dist_km', 'dist_pet_km', 'dist_fer_km', 'dist_m7_km', 'log_pgv_sum', 'rec_start']
    res = []
    # station level (median over components) to avoid pseudo-replication
    Ts = T.groupby('sta').agg({**{a: 'first' for a in attrs + ['group']}, 'D_2021_2025': 'median',
                               **{f'P_{y}': 'median' for y in PLACEBO_Y}})
    for grp in ['borehole', 'surface', 'all']:
        q = Ts if grp == 'all' else Ts[Ts.group == grp]
        q = q[q.D_2021_2025.notna()]
        for a in attrs:
            x = q[a]
            if x.nunique() < 4:
                continue
            rho = stats.spearmanr(x, q.D_2021_2025)[0]
            nulls = []
            for y in PLACEBO_Y:
                m = q[f'P_{y}'].notna()
                if m.sum() >= 5:
                    nulls.append(stats.spearmanr(x[m], q.loc[m, f'P_{y}'])[0])
            row = dict(group=grp, attr=a, n_sta=len(q), rho=rho, p_formal=stats.spearmanr(x, q.D_2021_2025)[1])
            if len(nulls) >= 4:
                row.update(rho_null_mean=np.mean(nulls), rho_null_sd=np.std(nulls, ddof=1), n_null=len(nulls),
                           n_sta_null=int(q[f'P_{PLACEBO_Y[-1]}'].notna().sum()))
            res.append(row)
    R = pd.DataFrame(res)
    R.to_csv(f'{OUT}/spatial_correlations.csv', index=False)
    Ts.to_csv(f'{OUT}/spatial_per_station.csv')
    # surface vs borehole contrast (placebo SEs from data/sci/dvv_placebo/longterm.csv)
    lt = pd.read_csv('data/sci/dvv_placebo/longterm.csv')
    a = lt[(lt.metric == '4yr: 2021->2025')].set_index('group')
    diff = a.loc['surface', 'median_change_pct'] - a.loc['borehole', 'median_change_pct']
    se = np.hypot(a.loc['surface', 'se_placebo_pct'], a.loc['borehole', 'se_placebo_pct'])
    # station heterogeneity of the (low-noise) pooled-curve decline vs the same spread in pre-2021 windows
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))  # noqa: E731
    het = {}
    for lab, q in [('PB', T[T.group == 'borehole']), ('PB+KMR+KMPB', T[(T.group == 'borehole') | T.sta.isin(LONG_SURF)])]:
        s = q.groupby('sta')[['D_2021_2025'] + [f'P_{y}' for y in PLACEBO_Y]].median()
        het[lab] = dict(n_sta=len(s), sd=zp(s.D_2021_2025.std(), [s[f'P_{y}'].std() for y in PLACEBO_Y]),
                        mad=zp(mad(s.D_2021_2025), [mad(s[f'P_{y}'].dropna()) for y in PLACEBO_Y]))
    # same-station-set check: long-record surface (KMR, KMPB) only
    ls = T[T.sta.isin(LONG_SURF)]
    S['spatial'] = dict(station_heterogeneity=het, surface_minus_borehole_pct=diff, se_placebo=se, z=diff / se,
                        ratio_surface_over_borehole=a.loc['surface', 'median_change_pct'] / a.loc['borehole', 'median_change_pct'],
                        long_surface_KMR_KMPB_D2021_2025=ls.set_index('chan').D_2021_2025.round(4).to_dict(),
                        per_station_D=Ts.D_2021_2025.round(4).to_dict(),
                        notable=R[(R.n_sta >= 5)].assign(z=lambda x: (x.rho - x.rho_null_mean) / x.rho_null_sd)
                        .round(3).to_dict('records'))
    return T, Ts, R


# ----------------------------------------------------------------------------------------------- 3 timing
def load_bins():
    """unregularised monthly bins (scripts/sci_decline_monthly.py) in the same layout as the pooled curves"""
    B = pd.read_csv(f'{OUT}/monthly_bins.csv', parse_dates=['month'])
    B = B[np.isfinite(B.dvv_pct) & (B.n_rows >= 20)]
    C = {}
    for ch, b in B.groupby('chan'):
        # drop wild bins (> 6 robust SD from the channel's 13-month running median)
        v = b.set_index('month').dvv_pct
        dev = v - v.rolling(13, center=True, min_periods=5).median()
        ok = np.abs(dev) <= 6 * 1.4826 * np.nanmedian(np.abs(dev))
        bb = b[ok.values]
        C[ch] = pd.DataFrame(dict(t=bb.month + pd.Timedelta(days=14), dvv_pct=bb.dvv_pct.values, err_pct=bb.se_pct.values)).reset_index(drop=True)
    return C


def yearly_check(Cb):
    """the 2021->2025 group-median change and pre-2021 4-yr placebo windows from the UNREGULARISED monthly bins"""
    Y = yearly(Cb)
    res = {}
    for g, m in [('borehole', Y.index.str.startswith('PB.')), ('surface', ~Y.index.str.startswith('PB.')),
                 ('surface_long', Y.index.map(sta_of).isin(LONG_SURF))]:
        Yg = Y[m]
        c = (Yg[2025] - Yg[2021]).dropna()
        nl = [np.median((Yg[y + 4] - Yg[y]).reindex(c.index).dropna()) for y in PLACEBO_Y]
        res[g] = dict(n_chan=len(c), **zp(np.median(c), nl))
    S['yearly_check_bins'] = res


MONTHS = pd.date_range('2008-01-01', '2026-07-01', freq='MS')


def monthly_matrix(C, deseason=True):
    """channel x month matrix of (deseasoned) dv/v; annual sin/cos fitted on pre-Petrolia nodes of each channel
    (channels starting after 2021 are deseasoned with their full record + quake steps)."""
    M = {}
    for ch, c in C.items():
        t = c.t
        yr = (t - pd.Timestamp('2008-01-01')).dt.total_seconds().values / (365.25 * 86400)
        v = c.dvv_pct.values.copy()
        if deseason:
            pre = (t < Q['pet']).values
            cols = [np.ones(len(t)), np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
            if pre.sum() >= 24:
                A = np.array(cols).T
                m = np.linalg.lstsq(A[pre], v[pre], rcond=None)[0]
            else:
                # short record: annual + steps + linear trend
                ex = [(t >= q).astype(float).values for q in Q.values()] + [yr]
                A = np.array(cols + ex).T
                m = np.linalg.lstsq(A, v, rcond=None)[0]
            v = v - A[:, 1:3] @ m[1:3]
        s = pd.Series(v, index=t)
        M[ch] = s.groupby(s.index.to_period('M')).mean()
    M = pd.DataFrame(M).T
    M.columns = M.columns.to_timestamp()
    return M


def group_median(M, chans, ref=('2019-01-01', '2021-06-30')):
    X = M.loc[chans]
    base = X.loc[:, (X.columns >= ref[0]) & (X.columns <= ref[1])].mean(axis=1)
    X = X.sub(base, axis=0)
    X = X[base.notna()]
    return X.median(axis=0), X.notna().sum(axis=0), X


def design_time(t, kind, t0=None):
    ty = (t - pd.Timestamp('2008-01-01')).total_seconds().values / (365.25 * 86400)
    cols, names = [np.ones(len(t))], ['c']
    if kind in ('steps', 'steps+ramps', 'hinge+steps'):
        for k, q in Q.items():
            cols.append((t >= q).astype(float))
            names.append(f'step_{k}')
    if kind == 'steps+ramps':
        for k, q in Q.items():
            cols.append(np.clip((t - q).total_seconds().values / (365.25 * 86400), 0, None))
            names.append(f'ramp_{k}')
    if kind in ('hinge', 'hinge+steps'):
        cols.append(np.clip((t - t0).total_seconds().values / (365.25 * 86400), 0, None))
        names.append('ramp')
    return np.array(cols).T, names


def ar1_ll(r):
    """Gaussian AR(1) concentrated log-likelihood of residuals (monthly series)."""
    phi = np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0, 0.95)
    e = r[1:] - phi * r[:-1]
    s2 = np.mean(e ** 2)
    return -0.5 * len(e) * np.log(s2), phi


def timing(C, key='timing'):
    M = monthly_matrix(C)
    early = [ch for ch, c in C.items() if c.t.min() < pd.Timestamp('2020-07-01')]
    groups = {'borehole': [c for c in early if c.startswith('PB.')], 'surface': [c for c in early if not c.startswith('PB.')],
              'surface_long': [c for c in early if sta_of(c) in LONG_SURF]}
    out, curves = {}, {}
    win = (MONTHS >= '2017-01-01') & (MONTHS <= '2026-06-01')
    t0s = pd.date_range('2019-01-01', '2023-12-01', freq='MS')
    for g, chs in groups.items():
        med, n, X = group_median(M, chs)
        curves[g] = (med, n, X)
        y = med.reindex(MONTHS[win])
        ok = y.notna().values
        t = MONTHS[win][ok]
        yv = y.values[ok]
        # pre-2021 noise: monthly scatter of deseasoned median 2009-2020 (AR(1))
        pre = med[(med.index >= '2010-01-01') & (med.index < '2021-01-01')].dropna()
        r_pre = pre - pre.rolling(12, center=True, min_periods=6).mean()
        res = {'n_chan': len(X), 'pre2021_monthly_sd_pct': float(pre.std()), 'pre2021_detrended_sd_pct': float(r_pre.std())}
        fits = {}
        for kind in ['steps', 'steps+ramps']:
            A, nm = design_time(t, kind)
            m = np.linalg.lstsq(A, yv, rcond=None)[0]
            ll, phi = ar1_ll(yv - A @ m)
            fits[kind] = dict(aic=-2 * ll + 2 * A.shape[1], phi=phi, **{k: float(v) for k, v in zip(nm, m)})
        for kind in ['hinge', 'hinge+steps']:
            best = None
            prof = []
            for t0 in t0s:
                A, nm = design_time(t, kind, t0)
                m = np.linalg.lstsq(A, yv, rcond=None)[0]
                ll, phi = ar1_ll(yv - A @ m)
                prof.append((t0, ll))
                if best is None or ll > best[1]:
                    best = (t0, ll, m, nm, phi)
            t0, ll, m, nm, phi = best
            prof = pd.DataFrame(prof, columns=['t0', 'll'])
            ci = prof[prof.ll >= ll - 1.92].t0      # profile-likelihood 95 % (approx; AR(1)-whitened)
            fits[kind] = dict(aic=-2 * ll + 2 * (A.shape[1] + 1), phi=phi, t0=str(t0.date()), t0_ci=[str(ci.min().date()), str(ci.max().date())],
                              **{k: float(v) for k, v in zip(nm, m)})
            if kind == 'hinge':
                # channel bootstrap of onset
                bt = []
                for _ in range(200):
                    idx = rng.integers(0, len(X), len(X))
                    mb = X.iloc[idx].median(0).reindex(MONTHS[win]).values[ok]
                    lls = []
                    for t0b in t0s:
                        A, _ = design_time(t, kind, t0b)
                        mm = np.linalg.lstsq(A, mb, rcond=None)[0]
                        lls.append(ar1_ll(mb - A @ mm)[0])
                    bt.append(t0s[int(np.argmax(lls))])
                bt = pd.Series(bt)
                fits[kind]['t0_boot_p05_p50_p95'] = [str(bt.quantile(q).date()) for q in (0.05, 0.5, 0.95)]
                fits[kind]['frac_boot_t0_before_pet'] = float((bt < Q['pet']).mean())
                fits[kind]['frac_boot_t0_before_2021_07'] = float((bt < pd.Timestamp('2021-07-01')).mean())
        res['fits'] = fits
        # half-year means relative to 2019-2020 and placebo SD of half-year-to-baseline changes
        hy = med.groupby([med.index.year, (med.index.month > 6).astype(int)]).mean()
        hy.index = [f'{a}H{b + 1}' for a, b in hy.index]
        res['halfyear_mean_pct'] = hy.loc['2019H1':].round(4).to_dict()
        # placebo: change of a half-year mean vs the preceding 2-yr mean, pre-2021
        pl = []
        for y in range(2011, 2021):
            for h in (0, 1):
                a = med[(med.index >= pd.Timestamp(y, 1 + 6 * h, 1)) & (med.index < pd.Timestamp(y, 1 + 6 * h, 1) + pd.DateOffset(months=6))].mean()
                b = med[(med.index >= pd.Timestamp(y - 2, 1 + 6 * h, 1)) & (med.index < pd.Timestamp(y, 1 + 6 * h, 1))].mean()
                pl.append(a - b)
        pl = np.array(pl)
        res['placebo_halfyear_vs_prev2yr_sd'] = float(np.nanstd(pl, ddof=1))
        for lab, a0 in [('2021H1', '2021-01-01'), ('2021H2', '2021-07-01')]:
            a = med[(med.index >= a0) & (med.index < pd.Timestamp(a0) + pd.DateOffset(months=6))]
            if lab == '2021H2':
                a = a[a.index < Q['pet']]
            b = med[(med.index >= pd.Timestamp(a0) - pd.DateOffset(years=2)) & (med.index < a0)].mean()
            res[f'{lab}_minus_prev2yr'] = zp(a.mean() - b, pl)
        # pre-Petrolia slope 2021-01..2021-11 vs pre-2021 11-month slopes (placebo)
        def slope(s):
            s = s.dropna()
            return np.polyfit((s.index - s.index[0]).days / 365.25, s.values, 1)[0]
        sl_claim = slope(med['2021-01-01':'2021-11-01'])
        sl_null = [slope(med[f'{y}-01-01':f'{y}-11-01']) for y in range(2010, 2021)]
        res['slope_2021Jan_Nov_pct_per_yr'] = zp(sl_claim, sl_null)
        # inter-quake slopes (deseasoned median): Petrolia->Ferndale, Ferndale->M7 (excluding first 30 d), after M7
        for lab, a0, a1 in [('pet_to_fer', Q['pet'] + pd.Timedelta(days=30), Q['fer']),
                            ('fer_to_m7', Q['fer'] + pd.Timedelta(days=30), Q['m7']),
                            ('after_m7', Q['m7'] + pd.Timedelta(days=30), pd.Timestamp('2026-07-01'))]:
            s = med[(med.index >= a0) & (med.index < a1)]
            L = len(s)
            nl = []
            for y in range(2009, 2021):
                for mo in (1, 7):
                    st0 = pd.Timestamp(y, mo, 1)
                    ss = med[(med.index >= st0) & (med.index < st0 + pd.DateOffset(months=L))]
                    if ss.notna().sum() >= max(4, L - 2) and ss.index.max() < pd.Timestamp('2021-07-01'):
                        nl.append(slope(ss))
            res[f'slope_{lab}_pct_per_yr'] = zp(slope(s), nl)
        # persistent level change at each quake (11 monthly bins after vs 11 before, quake month excluded) and its
        # split into early (first 2 months) and later (months 3-11) parts; placebo: every month centre 2010-01..2019-06
        def lev(q):
            k = med.index.get_indexer([pd.Timestamp(q.year, q.month, 1)])[0]
            b, a = med.iloc[k - 11:k], med.iloc[k + 1:k + 12]
            if b.notna().sum() < 8 or a.notna().sum() < 8:
                return np.nan, np.nan, np.nan
            e = med.iloc[k + 1:k + 3].mean()
            return a.mean() - b.mean(), e - b.mean(), med.iloc[k + 3:k + 12].mean() - e
        nulls = np.array([lev(t_) for t_ in pd.date_range('2010-01-01', '2019-06-01', freq='MS')])
        for k_, q in Q.items():
            L, E, La = lev(q)
            res[f'level_change_{k_}'] = zp(L, nulls[:, 0])
            res[f'early2mo_{k_}'] = zp(E, nulls[:, 1])
            res[f'later_3to11mo_vs_early_{k_}'] = zp(La, nulls[:, 2])
        out[g] = res
    S[key] = out
    pd.DataFrame({g: curves[g][0] for g in curves}).assign(**{f'n_{g}': curves[g][1] for g in curves}) \
        .to_csv(f'{OUT}/{key}_monthly_group_median.csv', index_label='month')
    return curves, M


# ----------------------------------------------------------------------------------------------- 4 hydrology
TAUS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


def precip():
    d = pd.read_csv(f'{OUT}/raw/ghcn_USW00024213.csv', parse_dates=['DATE'])
    a = pd.read_csv(f'{OUT}/raw/ghcn_USW00024283.csv', parse_dates=['DATE'])
    p = d.set_index('DATE').PRCP.astype(float) / 10.0                     # tenths of mm -> mm
    pa = a.set_index('DATE').PRCP.astype(float) / 10.0
    idx = pd.date_range('2008-01-01', '2026-06-30', freq='D')
    p = p.reindex(idx)
    p = p.fillna(pa.reindex(idx))                                          # fill Eureka gaps with Arcata
    p = p.fillna(0.0)
    return p


def hproxy(p, tau_yr):
    """h(t) = sum_{s<=t} exp(-(t-s)/tau) (P(s) - Pbar), daily; returned in mm (anomaly storage)."""
    a = np.exp(-1 / (tau_yr * 365.25))
    x = p.values - p.values.mean()
    h = np.zeros(len(x))
    acc = 0.0
    for i, v in enumerate(x):
        acc = acc * a + v
        h[i] = acc
    return pd.Series(h, index=p.index)


def hydrology(curves, C):
    p = precip()
    wy = p.groupby(p.index.year + (p.index.month >= 10)).sum()
    H = pd.DataFrame({tau: hproxy(p, tau) for tau in TAUS})
    Hm = H.groupby(H.index.to_period('M')).mean()
    Hm.index = Hm.index.to_timestamp()
    Hm.to_csv(f'{OUT}/hydro_proxies_monthly.csv', index_label='month')
    # RAW (not deseasoned) group medians (relative to 2019-2021H1), for testing the annual cycle too
    Mraw = monthly_matrix(C, deseason=False)
    res = {'water_year_precip_mm': wy.round(0).to_dict()}
    figdat = {}
    for g in ['borehole', 'surface_long']:
        chs = curves[g][2].index
        med, n, X = group_median(Mraw, chs)
        y = med['2009-01-01':'2026-06-01'].dropna()
        t = y.index
        ty = (t - pd.Timestamp('2008-01-01')).total_seconds().values / (365.25 * 86400)
        sc = np.c_[np.sin(2 * np.pi * ty), np.cos(2 * np.pi * ty)]
        post = t >= Q['pet']
        stp = np.array([(t >= q).astype(float) for q in Q.values()]).T
        out = {}
        # (a) annual cycle: sin/cos vs short-tau proxy, pre-2021 only
        pre = ~post
        for lab, X_ in [('sincos', sc)] + [(f'h{tau}', Hm.loc[t, [tau]].values) for tau in (0.1, 0.25, 0.5)]:
            A = np.c_[np.ones(len(t)), X_][pre]
            m = np.linalg.lstsq(A, y.values[pre], rcond=None)[0]
            r = y.values[pre] - A @ m
            out[f'annual_pre2021_R2_{lab}'] = float(1 - r.var() / y.values[pre].var())
        # (b) long-tau proxies: fit annual(sin/cos) + b*h_tau(long) on pre-2021 only -> predict post-2021 change
        obs_post = y[(t >= '2025-01-01') & (t < '2026-01-01')].mean() - y[(t >= '2021-01-01') & (t < '2022-01-01')].mean()
        # observed pre-2021 4-yr changes of yearly means (deseasoned by annual averaging)
        yy = y.groupby(y.index.year).mean()
        obs_pl = np.array([yy[y0 + 4] - yy[y0] for y0 in PLACEBO_Y if y0 in yy and y0 + 4 in yy])
        hy = Hm.groupby(Hm.index.year).mean()
        rows = []
        for tau in TAUS:
            h = Hm.loc[t, tau].values
            # pre-2021 fit
            A = np.c_[np.ones(len(t)), sc, h]
            m_pre = np.linalg.lstsq(A[pre], y.values[pre], rcond=None)[0]
            pred = A @ m_pre
            pred_change = pd.Series(pred, t)
            pc = pred_change['2025'].mean() - pred_change['2021'].mean()
            # full-record fit with quake steps
            Af = np.c_[np.ones(len(t)), sc, h, stp]
            m_f = np.linalg.lstsq(Af, y.values, rcond=None)[0]
            rf = y.values - Af @ m_f
            A0 = np.c_[np.ones(len(t)), sc, stp]
            r0 = y.values - A0 @ np.linalg.lstsq(A0, y.values, rcond=None)[0]
            ll1, _ = ar1_ll(rf)
            ll0, _ = ar1_ll(r0)
            # coefficient REQUIRED to explain the 2021->2025 change by h alone, and its pre-2021 4-yr predictions
            dh_claim = hy.loc[2025, tau] - hy.loc[2021, tau]
            b_req = obs_post / dh_claim if abs(dh_claim) > 1e-9 else np.nan
            dh_pl = np.array([hy.loc[y0 + 4, tau] - hy.loc[y0, tau] for y0 in PLACEBO_Y])
            pred_pl = b_req * dh_pl
            rows.append(dict(group=g, tau_yr=tau, b_pre2021_pct_per_m=m_pre[3] * 1000, pred_change_2021_2025_from_pre_fit=pc,
                             obs_change_2021_2025=obs_post, dh_2021_2025_mm=dh_claim, b_required_pct_per_m=b_req * 1000,
                             pred_pre2021_4yr_max_abs_with_b_req=float(np.max(np.abs(pred_pl))),
                             pred_pre2021_4yr_rms_with_b_req=float(np.sqrt(np.mean(pred_pl ** 2))),
                             obs_pre2021_4yr_max_abs=float(np.max(np.abs(obs_pl))), obs_pre2021_4yr_rms=float(np.sqrt(np.mean(obs_pl ** 2))),
                             worst_placebo_window=f'{PLACEBO_Y[int(np.argmax(np.abs(pred_pl)))]}->{PLACEBO_Y[int(np.argmax(np.abs(pred_pl)))] + 4}',
                             full_fit_b_pct_per_m=m_f[3] * 1000, full_fit_dAIC_vs_steps_only=float(-2 * (ll1 - ll0) + 2)))
        out['tau_table'] = rows
        res[g] = out
        figdat[g] = (y, Hm)
    HT = pd.DataFrame([r for g in ['borehole', 'surface_long'] for r in res[g]['tau_table']])
    HT.to_csv(f'{OUT}/hydro_tau_table.csv', index=False)
    S['hydrology'] = {k: (v if k == 'water_year_precip_mm' else {kk: vv for kk, vv in v.items() if kk != 'tau_table'})
                      for k, v in res.items()}
    return HT, figdat, wy, p


# ----------------------------------------------------------------------------------------------- 5 metadata
def metadata(C):
    rows = []
    for f in glob.glob(f'{OUT}/raw/meta_*.txt'):
        m = pd.read_csv(f, sep='|')
        m.columns = [c.strip().lstrip('#').strip() for c in m.columns]
        rows.append(m)
    M = pd.concat(rows)
    M['ns'] = M.Network + '.' + M.Station
    M['StartTime'] = pd.to_datetime(M.StartTime)
    M['EndTime'] = pd.to_datetime(M.EndTime.str.replace('3000-01-01', '2100-01-01'))
    ep = M[(M.StartTime >= '2019-01-01') & (M.StartTime <= '2026-07-01')]
    keep = ['ns', 'Location', 'Channel', 'Instrument', 'Scale', 'SampleRate', 'Depth', 'Elevation', 'StartTime']
    ep = ep[[k for k in keep if k in ep]].sort_values(['ns', 'StartTime', 'Channel'])
    ep.to_csv(f'{OUT}/metadata_epochs_2019_2026.csv', index=False)
    # dv/v step at each epoch date (per channel): mean of nodes 120 d after minus 120 d before, deseasoned curve,
    # vs the same statistic at 50 random dates in that channel's record avoiding quakes
    Md = monthly_matrix(C)
    tests = []
    for (ns, t0), g in ep.groupby(['ns', 'StartTime']):
        sens = ';'.join(sorted(set(g.Instrument.astype(str))))
        for ch, c in C.items():
            if sta_of(ch) != ns:
                continue
            s = Md.loc[ch].dropna()

            def jump(tt):
                a = s[(s.index >= tt) & (s.index < tt + pd.Timedelta(days=150))]
                b = s[(s.index < tt) & (s.index >= tt - pd.Timedelta(days=150))]
                if len(a) < 3 or len(b) < 3:
                    return np.nan
                return a.mean() - b.mean()
            j = jump(t0)
            nul = []
            cand = s.index[(s.index > s.index.min() + pd.Timedelta(days=200)) & (s.index < s.index.max() - pd.Timedelta(days=200))]
            for tt in cand:
                if min(abs((tt - q).days) for q in Q.values()) > 200 and abs((tt - t0).days) > 200:
                    nul.append(jump(tt))
            nul = np.array(nul)
            nul = nul[np.isfinite(nul)]
            if np.isfinite(j) and len(nul) >= 8:
                tests.append(dict(chan=ch, epoch_start=str(t0.date()), sensor=sens, jump_pct=j, null_sd=nul.std(ddof=1),
                                  z=(j - nul.mean()) / nul.std(ddof=1), n_null=len(nul)))
    T = pd.DataFrame(tests)
    T.to_csv(f'{OUT}/metadata_epoch_steps.csv', index=False)
    S['metadata'] = dict(epochs=ep.assign(StartTime=ep.StartTime.astype(str)).drop_duplicates(['ns', 'StartTime', 'Instrument'])
                         [['ns', 'StartTime', 'Instrument', 'Depth']].to_dict('records'),
                         epoch_steps=T.round(4).to_dict('records'))
    return ep, T


# ----------------------------------------------------------------------------------------------- figures
def fig_family(R, per):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    d, se, fe = per['long_record']
    order = d.groupby('sta').D.median().sort_values().index
    for i, s in enumerate(order):
        v = d[d.sta == s].D
        ax[0].scatter(np.full(len(v), i) + rng.uniform(-0.2, 0.2, len(v)), v, s=8, alpha=0.5)
        ax[0].plot([i - 0.3, i + 0.3], [v.median()] * 2, 'k-', lw=2)
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels([s.split('.')[1] for s in order], rotation=60, fontsize=8)
    ax[0].axhline(0, color='grey', lw=0.6)
    ax[0].set_ylabel('dv/v change 2021 -> 2025 (%)')
    ax[0].set_title('per family x channel decline, by station')
    g = R[R.set == 'long_record']
    for k, (col, lab) in enumerate([('F_fam', 'family F-stat (given station)'), ('F_sta', 'station F-stat (given family)')]):
        pl = g[g.y0 < 2021]
        ax[1].scatter(np.full(len(pl), k) + rng.uniform(-0.1, 0.1, len(pl)), pl[col], c='grey', label='pre-2021 4-yr windows' if k == 0 else None)
        ax[1].scatter([k], g[g.y0 == 2021][col], c='r', s=80, marker='*', label='2021 -> 2025' if k == 0 else None)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(['family | station', 'station | family'])
    ax[1].set_ylabel('F statistic')
    ax[1].legend(fontsize=8)
    ax[1].set_title('two-way decomposition vs placebo windows')
    fa = d[d.sta.isin({'PB.B045', 'PB.B046', 'PB.B047', 'PB.B049'})].groupby('fam').D.median()
    fb = d[d.sta.isin({'PB.B932', 'PB.B933', 'PB.B934', 'PB.B935'})].groupby('fam').D.median()
    j = fa.index.intersection(fb.index)
    ax[2].scatter(fa[j], fb[j])
    for f_ in j:
        ax[2].annotate(str(f_), (fa[f_], fb[f_]), fontsize=7)
    r = g[g.y0 == 2021].rho_fam_PBA_PBB.iloc[0]
    rn = g[g.y0 < 2021].rho_fam_PBA_PBB
    ax[2].set_title(f'family decline: PB B045-49 vs B932-35\nrho={r:.2f} (placebo {rn.mean():.2f}+/-{rn.std():.2f})', fontsize=9)
    ax[2].set_xlabel('median over B045,B046,B047,B049 (%)')
    ax[2].set_ylabel('median over B932-B935 (%)')
    fig.tight_layout()
    fig.savefig('figures/sci_decline_family.png', dpi=120)


def fig_timing(curves, key='timing'):
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    for k, g in enumerate(['borehole', 'surface']):
        med, n, X = curves[g]
        for _, row in X.iterrows():
            ax[k].plot(row.index, row.values, color='0.85', lw=0.5)
        m12 = med.rolling(3, center=True, min_periods=2).mean()
        ax[k].plot(med.index, med.values, 'k.', ms=3)
        ax[k].plot(m12.index, m12.values, 'k-', lw=1.5, label=f'{g} median ({len(X)} ch), 3-month mean')
        fit = S[key][g]['fits']['hinge']
        t0 = pd.Timestamp(fit['t0'])
        tt = med.index[med.index >= '2017-01-01']
        ax[k].plot(tt, fit['c'] + fit['ramp'] * np.clip((tt - t0).days / 365.25, 0, None), 'r-', lw=1.2,
                   label=f"hinge fit, onset {fit['t0']} (boot 90%: {fit['t0_boot_p05_p50_p95'][0]}..{fit['t0_boot_p05_p50_p95'][2]})")
        for q in Q.values():
            ax[k].axvline(q, color='b', ls='--', lw=0.8)
        ax[k].axvline(pd.Timestamp('2023-01-01'), color='m', ls=':', lw=0.8)
        lim = np.nanpercentile(np.abs(med.values), 99) * 2.2
        ax[k].set_ylim(-lim, lim * 0.6)
        ax[k].set_ylabel('deseasoned dv/v (%)')
        ax[k].legend(fontsize=8, loc='lower left')
        ax[k].axhline(0, color='grey', lw=0.5)
    ax[0].set_title('monthly deseasoned dv/v (re 2019-2021H1); blue: Petrolia, Ferndale, M7; magenta: PB archive->NCEDC switch')
    ax[1].set_xlim(pd.Timestamp('2009-01-01'), pd.Timestamp('2026-07-01'))
    fig.tight_layout()
    fig.savefig('figures/sci_decline_timing.png', dpi=120)


def fig_hydro(HT, figdat, wy, p):
    fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=False)
    w = wy.loc[2009:2026]
    ax[0].bar([pd.Timestamp(y - 1, 10, 1) + pd.Timedelta(days=182) for y in w.index], w.values, width=300, color='tab:blue', alpha=0.6)
    ax[0].set_ylabel('water-year precip (mm)\nEureka USW00024213')
    a2 = ax[0].twinx()
    Hm = figdat['borehole'][1]
    for tau in (1.0, 3.0):
        a2.plot(Hm.index, Hm[tau] / 1000, label=f'h, tau={tau:g} yr')
    a2.set_ylabel('filtered precip anomaly (m)')
    a2.legend(fontsize=8, loc='upper left')
    for k, g in enumerate(['borehole', 'surface_long']):
        y, Hm = figdat[g]
        yy = y.rolling(12, center=True, min_periods=9).mean()
        ax[k + 1].plot(y.index, y.values, color='0.7', lw=0.7, label=f'{g} raw monthly median')
        ax[k + 1].plot(yy.index, yy.values, 'k', lw=1.6, label='12-month mean')
        for tau, c in [(1.0, 'tab:orange'), (3.0, 'tab:green')]:
            r = HT[(HT.group == g) & (HT.tau_yr == tau)].iloc[0]
            h = Hm[tau].reindex(y.index)
            b = r.b_required_pct_per_m / 1000
            pr = b * (h - h['2021'].mean()) + yy['2021'].mean()
            ax[k + 1].plot(pr.rolling(12, center=True, min_periods=9).mean(), color=c, lw=1.2,
                           label=f'hydrology with b needed for decline, tau={tau:g} yr (12-mo mean)')
        for q in Q.values():
            ax[k + 1].axvline(q, color='b', ls='--', lw=0.8)
        ax[k + 1].set_ylabel('dv/v (%)')
        ax[k + 1].legend(fontsize=7, loc='lower left')
    for a in ax:
        a.set_xlim(pd.Timestamp('2008-07-01'), pd.Timestamp('2026-07-01'))
    ax[0].set_title('Hydrology test: a precipitation-storage model scaled to produce the post-2021 decline predicts large pre-2021 swings')
    fig.tight_layout()
    fig.savefig('figures/sci_decline_hydro.png', dpi=120)


def main():
    os.makedirs(OUT, exist_ok=True)
    R, per = family_station_test()
    print(pd.DataFrame({k: {kk: (vv['value'], vv['z']) if isinstance(vv, dict) else vv for kk, vv in v.items()}
                        for k, v in S['family_vs_station'].items() if k != 'injection_power'}).to_string())
    C = load_curves()
    Cb = load_bins()
    T, Ts, SR = spatial(Cb)
    _, _, SRp = spatial(C)
    S['spatial_poolcurves_heterogeneity'] = S['spatial']['station_heterogeneity']
    T, Ts, SR = spatial(Cb)
    print(SR.round(3).to_string())
    timing(C, 'timing_poolcurves')
    yearly_check(Cb)
    print(json.dumps(S['yearly_check_bins'], indent=1, default=str))
    yb = S['yearly_check_bins']
    for g in ['surface', 'surface_long']:
        dd = yb[g]['value'] - yb['borehole']['value']
        se_ = np.hypot(yb[g]['null_sd'], yb['borehole']['null_sd'])
        S['spatial'][f'bins_{g}_minus_borehole'] = dict(diff_pct=dd, se_placebo_pct=se_, z=dd / se_,
                                                         ratio=yb[g]['value'] / yb['borehole']['value'])
    curves, M = timing(Cb, 'timing')
    print(json.dumps(S['timing'], indent=1, default=str)[:6000])
    HT, figdat, wy, p = hydrology(curves, Cb)
    print(HT.round(4).to_string())
    print(json.dumps(S['hydrology'], indent=1, default=str)[:3000])
    ep, ET = metadata(C)
    print(ET.round(4).to_string())
    fig_family(R, per)
    fig_timing(curves, 'timing')
    fig_hydro(HT, figdat, wy, p)
    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


if __name__ == '__main__':
    main()
