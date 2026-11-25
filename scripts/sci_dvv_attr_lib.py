#!/usr/bin/env python
"""Engine for the dv/v attribution: daily driver regressors, design matrices, robust LSQ,
water-year cross-validation, BIC with effective n, moving-block bootstrap, placebos.

All regressors live on the daily grid DAYS.  A dv/v sample that is an average over a window
(10-day backward stack, 30-day centred month, ...) is matched by averaging every regressor over the
same window ("smoothing operator" SM = ('back', n) or ('centre', n) or ('point', 1)).
Units: dv/v in %; time in years for trends; quake doses in cm/s (PGV) or nanostrain (static strain).
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy.signal import lfilter, fftconvolve

ROOT = '/home/seismic/LFE'
BULK = '/data/wsd02/maleen_data/dvv_attr'
DRV = f'{BULK}/drivers'
RAW = f'{BULK}/raw'
DAYS = pd.date_range('2007-01-01', '2026-11-24', freq='D')
ND = len(DAYS)
TY = (DAYS - pd.Timestamp('2008-01-01')).days.values / 365.25          # years since 2008
BIG3 = {'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20', 'M7': '2024-12-05'}
HINGE = '2021-07-01'
PGV_MIN = 0.5                    # cm/s (BSSA14, Vs30 760): smaller doses ignored (a permanent term for every
                                 # M3.5 would accumulate into a pseudo-trend)
# instrument windows (copied from scripts/sci_dvv_3d_data.py / sci_dvv_instrument_check.py, not edited there)
EXCL = {'PB.B933': [('2017-01-01', '2021-11-01'), ('2024-08-01', '2030-01-01')],
        'PB.B935': [('2024-05-01', '2030-01-01')],
        'PB.B047': [('2000-01-01', '2009-03-01')],
        'PB.B045': [('2000-01-01', '2011-01-01')],          # EH1 polarity problem 2009-10 (all SC pairs use EH1/EH2)
        'NC.KMPB': [('2000-01-01', '2010-11-03')],
        'NC.KMR': [('2000-01-01', '2009-09-17'), ('2025-09-17', '2030-01-01')],
        'BK.BRIC': [('2022-12-01', '2024-05-22')],
        'BK.RVIT': [('2022-12-20', '2024-06-21')],
        'BK.BJES': [('2022-12-20', '2024-05-23')]}
BREAKS = {'PB.B933': ['2021-11-01'], 'BK.BRIC': ['2024-05-22'], 'BK.RVIT': ['2024-06-21'],
          'BK.BJES': ['2024-05-23'], 'NC.KMPB': ['2023-10-17']}
GAUGES = {'11469000': (40.3132, -124.2834), '11475000': (40.2182, -123.6325), '11475800': (39.8746, -123.7206),
          '11476500': (40.1818, -123.7761), '11477000': (40.4915, -124.0997), '11478500': (40.4804, -123.8908)}


def dayidx(t):
    return np.clip(((pd.DatetimeIndex(t).normalize() - DAYS[0]).days.values).astype(int), 0, ND - 1)


# ------------------------------------------------------------------------------------------------ smoothing
def smooth(x, sm):
    """Average daily series x (ND,) or (ND,k) over the sample window.  NaNs in x propagate as mean of finite."""
    kind, n = sm
    x = np.asarray(x, float)
    one = x.ndim == 1
    X = x[:, None] if one else x
    if kind == 'point' or n <= 1:
        return x
    f = np.isfinite(X)
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(np.where(f, X, 0.0), 0)])
    cn = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(f, 0)])
    i = np.arange(ND)
    if kind == 'back':
        lo, hi = np.clip(i - n + 1, 0, ND), i + 1
    else:
        lo, hi = np.clip(i - n // 2, 0, ND), np.clip(i - n // 2 + n, 0, ND)
    s = (c[hi] - c[lo]) / np.maximum(cn[hi] - cn[lo], 1)
    s[(cn[hi] - cn[lo]) == 0] = np.nan
    return s[:, 0] if one else s


# ------------------------------------------------------------------------------------------------ drivers
_cache = {}


def _pq(name):
    if name not in _cache:
        _cache[name] = pd.read_parquet(f'{DRV}/{name}')
    return _cache[name]


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(np.asarray(lon2) - lon1), p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def station_coords():
    c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv')
    c['ns'] = c.net + '.' + c.sta
    c = c[c.start < '2025-09-01']
    g = c.groupby('ns').agg(lat=('lat', 'median'), lon=('lon', 'median'), depth_m=('depth', 'median'))
    g['depth_m'] = np.where(g.index.str.startswith('PB.'), g.depth_m, 0.0)
    return g


def streamflow():
    if 'flow' not in _cache:
        out = {}
        for s in GAUGES:
            j = json.load(open(f'{RAW}/usgs/daily_{s}.json'))
            d = pd.DataFrame([f['properties'] for f in j['features']])
            d['time'] = pd.to_datetime(d.time); d['value'] = pd.to_numeric(d.value, errors='coerce')
            out[s] = d.groupby('time').value.mean().reindex(DAYS)
        _cache['flow'] = pd.DataFrame(out)
    return _cache['flow']


def drivers(sta, lat=None, lon=None):
    """Daily raw driver series for one station (dict name -> (ND,) array).  sta may be 'REGION' (network mean)."""
    if sta == 'REGION':
        D = {}
        stas = [s for s in station_coords().index if os.path.exists(f'{DRV}/climate_{s}.parquet')]
        ds = [drivers(s) for s in stas]
        for k in ds[0]:
            if k.startswith('_') or k in ('bStr',):
                continue
            D[k] = np.nanmean(np.vstack([d[k] for d in ds if k in d]), 0)
        reg = _pq('gnss_regional.parquet').reindex(DAYS)
        D['gU'] = reg.gnss_u_regional_mm.values
        D['_gauge'] = 'mean of all'
        return D
    C = _pq(f'climate_{sta}.parquet').reindex(DAYS)
    D = {}
    P = C['om_precipitation_sum'].values.astype(float)
    D['P_era5'] = np.where(np.isfinite(P), P, 0.0)               # reanalysis at the nearest downloaded site
    g = C['ghcn_prcp'].values.astype(float) if 'ghcn_prcp' in C else np.full(ND, np.nan)
    D['P'] = np.where(np.isfinite(g), g, D['P_era5'])             # primary: nearest GHCN gauges, gaps from ERA5
    D['T'] = C['om_temperature_2m_mean'].interpolate(limit_direction='both').values
    for k, col in (('SM0_7', 'el_soil_moisture_0_to_7cm_mean'), ('SM28_100', 'el_soil_moisture_28_to_100cm_mean'),
                   ('SM100_255', 'el_soil_moisture_100_to_255cm_mean'), ('SM0_100', 'om_soil_moisture_0_to_100cm_mean')):
        if col in C:
            D[k] = C[col].interpolate(limit_direction='both').values
    D['Ps'] = C['om_surface_pressure_mean'].interpolate(limit_direction='both').values
    if 'sealevel_m' in C:
        D['SL'] = C['sealevel_m'].values * 100.0                       # cm
    # GNSS
    G = _pq('gnss_station.parquet')
    if sta in G.index.get_level_values(0):
        g = G.loc[sta].reindex(DAYS)
        D['gU'] = g.gnss_u_mm.values
        D['gDil'] = g.gnss_dil_ns.values
        D['gDilPost'] = g.gnss_dil_post_ns.values
    # borehole strain
    B = _pq('bstrain_areal.parquet').reindex(DAYS)
    if f'{sta}_hp' in B:
        D['bStr'] = B[f'{sta}_hp'].values
    # streamflow: nearest gauge, log10 Q
    cc = station_coords()
    if sta in cc.index:
        la, lo = cc.loc[sta, 'lat'], cc.loc[sta, 'lon']
        k = min(GAUGES, key=lambda s_: hav(la, lo, *GAUGES[s_]))
        q = streamflow()[k]
        D['logQ'] = np.log10(q.interpolate(limit=5).clip(lower=0.1)).values
        D['_gauge'] = k
    return D


def power_band(band):
    P = _pq('power.parquet').reindex(DAYS)
    b = band.replace('Hz', '')
    col = b if b in P else ('0.5-1' if b in ('1-3', '3-5') else None)
    if col is None:
        return None, None
    return np.log10(P[col].values), col


# ------------------------------------------------------------------------------------------------ physical transforms
def anom(x):
    x = np.asarray(x, float)
    return x - np.nanmean(x)


def hyd_exp(P, tau):
    """Sens-Schoenfelder & Wegler (2006)-type groundwater proxy: sum_t' P(t') exp(-(t-t')/tau), demeaned (mm)."""
    a = np.exp(-1.0 / tau)
    return anom(lfilter([1.0], [1.0, -a], P))


def hyd_diff(P, tau_d):
    """Talwani-type pore-pressure diffusion: sum_t' (P(t') - Pbar) erfc(sqrt(tau_d / (t - t'))); tau_d = z^2/(4D) (days)."""
    lag = np.arange(1, ND + 1, dtype=float)
    k = erfc(np.sqrt(tau_d / lag))
    x = P - P[TY >= 1.0].mean()
    out = fftconvolve(x, k)[:ND]
    return anom(out)


def rain_lags(P):
    """Empirical distributed lag: rain summed over [0,10), [10,30), [30,90), [90,180), [180,365) days before t."""
    c = np.concatenate([[0.0], np.cumsum(P)])
    i = np.arange(ND)
    cols = []
    for a, b in ((0, 10), (10, 30), (30, 90), (90, 180), (180, 365)):
        hi, lo = np.clip(i + 1 - a, 0, ND), np.clip(i + 1 - b, 0, ND)
        cols.append(anom(c[hi] - c[lo]))
    return np.column_stack(cols)


def thermal(T, kind, par):
    T = anom(T)
    if kind == 'lag':
        out = np.full(ND, np.nan); L = int(par)
        out[L:] = T[:ND - L] if L > 0 else T
        out[:L] = T[0]
        return out
    a = np.exp(-1.0 / par)                    # exponential (diffusive) low-pass, unit DC gain
    return lfilter([1 - a], [1.0, -a], T, zi=[a * T[0]])[0]


# ------------------------------------------------------------------------------------------------ quakes
_Q = None


def quakes_for(sta):
    global _Q
    if _Q is None:
        _Q = pd.read_csv(f'{DRV}/quakes_station.csv.gz', parse_dates=['t'])
    if sta == 'REGION':
        q = _Q.groupby('qid').agg(t=('t', 'first'), mag=('mag', 'first'), kind=('kind', 'first'),
                                  pgv_cms=('pgv_cms', 'median'), vol_strain=('vol_strain', 'median'),
                                  pgv_meas_cms=('pgv_meas_cms', 'median')).reset_index()
        return q
    return _Q[_Q.sta == sta].copy()


def big3_day(name):
    return int(dayidx([pd.Timestamp(BIG3[name])])[0])


def quake_cols(sta, tau, mode='free3', with_trans=True, tau_by=None):
    """Coseismic regressors.  mode:
       'free3'   : big three free (perm step + exp transient each) + shared PGV dose for all other quakes
       'pgv_all' : one shared PGV dose (perm + trans) for all quakes incl. big three
       'strain_all': one shared static-volumetric-strain coefficient (signed, nanostrain), perm + trans
       'free3_only': big three free, no dose for others
    Returns (X (ND,k), names)."""
    q = quakes_for(sta)
    d = dayidx(q.t)
    i = np.arange(ND)
    big_days = [big3_day(n) for n in BIG3]
    near_big = np.array([any(abs(x - b) <= 1 for b in big_days) for x in d])
    cols, names = [], []
    tr = lambda x0: np.where(i >= x0, np.exp(-(i - x0) / tau), 0.0)
    st = lambda x0: (i >= x0).astype(float)
    if mode in ('free3', 'free3_only'):
        for n, b in zip(BIG3, big_days):
            cols.append(st(b)); names.append(f'step_{n}')
            if with_trans:
                tq = (tau_by or {}).get(n, tau)
                cols.append(np.where(i >= b, np.exp(-(i - b) / tq), 0.0)); names.append(f'trans_{n}')
    if mode in ('free3', 'pgv_all', 'strain_all'):
        if mode == 'strain_all':
            g = q.vol_strain.values * 1e9
            use = np.abs(g) >= 0.5                         # >= 0.5 nanostrain
        else:
            g = q.pgv_cms.values
            use = g >= PGV_MIN
        if mode == 'free3':
            use &= ~near_big
        Dp, Dt = np.zeros(ND), np.zeros(ND)
        for x0, gg in zip(d[use], g[use]):
            Dp += gg * st(x0)
            Dt += gg * tr(x0)
        tag = 'strain' if mode == 'strain_all' else 'pgv'
        cols.append(Dp); names.append(f'dose_{tag}_perm')
        if with_trans:
            cols.append(Dt); names.append(f'dose_{tag}_trans')
    return (np.column_stack(cols) if cols else np.zeros((ND, 0))), names


# ------------------------------------------------------------------------------------------------ design
def base_cols(sta, segs_breaks=None):
    """constant + per-segment offsets at instrument breaks."""
    cols, names = [np.ones(ND)], ['const']
    for k, b in enumerate(segs_breaks if segs_breaks is not None else BREAKS.get(sta, [])):
        cols.append((DAYS >= pd.Timestamp(b)).astype(float)); names.append(f'offset_{b[:10]}')
    return np.column_stack(cols), names


def long_cols():
    return {'trend': (TY - 10.0)[:, None], 'ramp21': np.clip(TY - (pd.Timestamp(HINGE) - pd.Timestamp('2008-01-01')).days / 365.25, 0, None)[:, None]}


def annual_cols():
    w = 2 * np.pi * TY
    return np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


def candidate_groups(sta, band=None, D=None):
    """Seasonal / environmental candidate groups: name -> list of (param_label, X (ND,k))."""
    D = D if D is not None else drivers(sta)
    G = {}
    G['annual'] = [('sin+cos 1+2/yr', annual_cols())]
    G['hyd_exp'] = [(f'tau_h={t}d', hyd_exp(D['P'], t)[:, None]) for t in (10, 30, 60, 120, 240, 480)]
    G['hyd_diff'] = [(f'tau_d={t}d', hyd_diff(D['P'], t)[:, None]) for t in (3, 10, 30, 100, 300, 1000)]
    G['hyd_exp_era5'] = [(f'tau_h={t}d (ERA5 precip)', hyd_exp(D['P_era5'], t)[:, None]) for t in (30, 60, 120, 240)]
    G['rain_lags'] = [('0-10,10-30,30-90,90-180,180-365d', rain_lags(D['P']))]
    sm = [(k, anom(D[k])[:, None]) for k in ('SM0_7', 'SM28_100', 'SM100_255', 'SM0_100') if k in D]
    if sm:
        G['soilm'] = sm
    if 'logQ' in D:
        G['streamflow'] = [(f'log10Q gauge {D["_gauge"]}, exp-smoothed {t}d', thermal(D['logQ'], 'exp', t)[:, None] if t > 1 else anom(D['logQ'])[:, None])
                           for t in (1, 30, 90)]
    G['thermal'] = ([(f'exp tau_T={t}d', thermal(D['T'], 'exp', t)[:, None]) for t in (1, 15, 30, 60, 120)] +
                    [(f'lag={L}d', thermal(D['T'], 'lag', L)[:, None]) for L in (30, 60, 90, 120)])
    G['pressure'] = [('surface pressure anomaly', anom(D['Ps'])[:, None])]
    if 'SL' in D:
        G['sealevel'] = [('North Spit monthly MSL', anom(D['SL'])[:, None])]
    if 'gU' in D and np.isfinite(D['gU']).sum() > 1000:
        G['gnss_up'] = [('local GNSS vertical (seasonal kept)', anom(D['gU'])[:, None])]
    if 'gDil' in D and np.isfinite(D['gDil']).sum() > 1000:
        G['gnss_dil'] = [('GNSS areal strain (non-tectonic)', anom(D['gDil'])[:, None])]
    if 'bStr' in D:
        G['bstrain'] = [('borehole areal strain, high-passed', anom(D['bStr'])[:, None])]
    if band is not None:
        p, col = power_band(band)
        if p is not None:
            G['micro'] = [(f'log10 power {col} Hz (B045 EHZ)', anom(p)[:, None])]
    return G


HYDRO = ('hyd_exp', 'hyd_diff', 'hyd_exp_era5', 'rain_lags', 'soilm', 'streamflow')
GEOD = ('gnss_up', 'gnss_dil', 'bstrain')


# ------------------------------------------------------------------------------------------------ fitting
def huber_lsq(X, y, w=None, nit=4, k=1.5):
    """Robust (Huber) weighted LSQ.  Returns beta, weights."""
    w0 = np.ones(len(y)) if w is None else w
    ww = w0.copy()
    for _ in range(nit):
        s = np.sqrt(ww)
        beta, *_ = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)
        r = (y - X @ beta) * np.sqrt(w0)
        sc = 1.4826 * np.median(np.abs(r)) + 1e-12
        z = np.abs(r) / sc
        ww = w0 * np.where(z <= k, 1.0, k / np.maximum(z, 1e-12))
    return beta, ww


def water_year(idx):
    d = DAYS[idx]
    return d.year + (d.month >= 10).astype(int)


def cv_score_mask(idx, pre=15, post=365):
    """Samples scored in CV: outside [t_q - pre, t_q + post] of the big three (quake terms are not
    predictable when their own year is held out; the same mask is used for every model)."""
    m = np.ones(len(idx), bool)
    for n in BIG3:
        b = big3_day(n)
        m &= ~((idx >= b - pre) & (idx <= b + post))
    return m


def cv_rms(X, y, idx, w=None, min_train=0.5):
    """Leave-one-water-year-out CV RMS (robust fit on the rest), scored outside the big-3 windows."""
    wy = water_year(idx)
    score = cv_score_mask(idx)
    err = np.full(len(y), np.nan)
    for g in np.unique(wy):
        te = wy == g
        tr = ~te
        if tr.sum() < min_train * len(y):
            continue
        # columns without support in training are dropped for this fold
        sup = np.abs(X[tr]).sum(0) > 0
        b, _ = huber_lsq(X[tr][:, sup], y[tr], None if w is None else w[tr], nit=2)
        err[te] = y[te] - X[te][:, sup] @ b
    e = err[np.isfinite(err) & score]
    return float(np.sqrt(np.mean(e ** 2))), float(np.median(np.abs(e)) * 1.4826)


def neff_factor(r, maxlag=200):
    """Decorrelation length (samples) of residuals on a regular daily grid: first lag with acf < 1/e (>= 1)."""
    r = r - r.mean()
    v = np.dot(r, r) / len(r)
    for L in range(1, min(maxlag, len(r) // 3)):
        if np.dot(r[:-L], r[L:]) / (len(r) - L) / v < np.exp(-1):
            return L
    return maxlag


def bic(sse, n, k, L):
    ne = max(n / L, k + 2)
    return ne * np.log(sse / n) + k * np.log(ne)


class Series:
    """One dv/v series on the daily grid: y (%), sample day index, optional weights, smoothing operator."""

    def __init__(self, dataset, sta, chan, t, y, sm, w=None, breaks=None, pair_i=None):
        self.dataset, self.sta, self.chan, self.sm = dataset, sta, chan, sm
        self.idx = dayidx(t)
        self.y = np.asarray(y, float)
        self.w = None if w is None else np.asarray(w, float)
        self.breaks = BREAKS.get(sta, []) if breaks is None else breaks
        self.pair_i = pair_i               # for pairwise (repeater) data: day index of the first event

    def design(self, X):
        """Map daily regressor matrix (ND,k) to samples (n,k): smooth then sample (or difference for pairs)."""
        Xs = smooth(X, self.sm)
        if self.pair_i is not None:
            return Xs[self.idx] - Xs[self.pair_i]
        return Xs[self.idx]


def assemble(S, blocks):
    """blocks: list of (group, names, X daily).  Returns sample design, names, group labels; drops all-zero/NaN cols."""
    Xs, names, groups = [], [], []
    for g, nm, X in blocks:
        A = S.design(X)
        Xs.append(A); names += list(nm); groups += [g] * A.shape[1]
    A = np.column_stack(Xs)
    return A, names, groups


def finite_rows(A, y):
    return np.isfinite(y) & np.isfinite(A).all(1)
