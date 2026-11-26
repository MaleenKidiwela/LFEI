#!/usr/bin/env python
"""Engine for the dv/v attribution: daily driver regressors, design matrices, sign-constrained robust LSQ,
water-year cross-validation, BIC with effective n, moving-block bootstrap, water-year permutation placebos.

All regressors live on the daily grid DAYS.  A dv/v sample that is an average over a window
(10-day backward stack, 30-day centred month, ...) is matched by averaging every regressor over the
same window ("smoothing operator" SM = ('back', n) or ('centre', n) or ('point', 1)).
Units: dv/v in %; time in years for trends; quake doses in cm/s (PGV) or nanostrain (static strain).

Revision after code review (2026-11-25): M1 precipitation from GHCN gauges + NASA POWER (MERRA-2) gap fill and
reanalysis alternative (Open-Meteo best_match dropped); M2 coseismic step/transient pairs sign-constrained
(same sign, i.e. drop0 + recovered fraction in [0, 1]); M6 PGV dose transient-only for non-big-3 local events,
hypocentral distance + site factor, 30-day aftershock window; n_eff from the integrated ACF; quake day = day of
(origin + 12 h); hyd_diff tau_d <= 100 d; water-year permutation placebo.
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy.signal import lfilter, fftconvolve
from scipy.optimize import lsq_linear

ROOT = '/home/seismic/LFE'
BULK = '/data/wsd02/maleen_data/dvv_attr'
DRV = f'{BULK}/drivers'
RAW = f'{BULK}/raw'
DAYS = pd.date_range('2007-01-01', '2026-11-24', freq='D')
ND = len(DAYS)
TY = (DAYS - pd.Timestamp('2008-01-01')).days.values / 365.25          # years since 2008
BIG3 = {'Petrolia': '2021-12-20 20:10', 'Ferndale': '2022-12-20 10:34', 'M7': '2024-12-05 18:44'}   # UTC origin times
HINGE = '2021-07-01'
AFTERSHOCK_WIN = (-1, 30)        # days around a big-3 quake whose other events are absorbed by the big-3 terms
CV_ALL = os.environ.get('ATTR_CV_ALL', '0') == '1'      # sensitivity: score CV on all samples
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
HYDRO = ('hyd_exp', 'hyd_diff', 'hyd_exp_rean', 'rain_lags', 'soilm', 'streamflow')
GEOD = ('gnss_up', 'gnss_dil', 'bstrain')
CORE = ('big3', 'dose', 'trend', 'ramp21')


def dayidx(t):
    return np.clip(((pd.DatetimeIndex(t).normalize() - DAYS[0]).days.values).astype(int), 0, ND - 1)


def quake_day(t):
    """Daily-grid index of the first day whose data are (mostly) post-event: day of (origin + 12 h)."""
    return dayidx(pd.DatetimeIndex(t) + pd.Timedelta(hours=12))


def big3_day(name):
    return int(quake_day([pd.Timestamp(BIG3[name])])[0])


# ------------------------------------------------------------------------------------------------ smoothing
def smooth(x, sm):
    """Average daily series x (ND,) or (ND,k) over the sample window.  NaN-aware mean of the finite days."""
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
    if 'coords' not in _cache:
        c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv')
        c['ns'] = c.net + '.' + c.sta
        c = c[c.start < '2025-09-01']
        g = c.groupby('ns').agg(lat=('lat', 'median'), lon=('lon', 'median'), depth_m=('depth', 'median'))
        g['depth_m'] = np.where(g.index.str.startswith('PB.'), g.depth_m, 0.0)
        _cache['coords'] = g
    return _cache['coords']


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


def _interp(s):
    return pd.Series(np.asarray(s, float), index=DAYS).interpolate(limit_direction='both').values


def drivers(sta):
    """Daily raw driver series for one station (dict name -> (ND,) array).  sta may be 'REGION' (network mean)."""
    key = f'drv_{sta}'
    if key in _cache:
        return _cache[key]
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
        _cache[key] = D
        return D
    C = _pq(f'climate_{sta}.parquet').reindex(DAYS)
    D = {}
    pr = C['pw_prcp'].values.astype(float)
    D['P_rean'] = np.where(np.isfinite(pr), pr, 0.0)                 # NASA POWER (MERRA-2, gauge-corrected)
    g = C['ghcn_prcp'].values.astype(float)                          # GHCN gauges, gaps = POWER x local ratio
    D['P'] = np.where(np.isfinite(g), g, D['P_rean'])
    D['T'] = _interp(C['pw_t2m'])
    D['Ps'] = _interp(C['pw_ps_hpa'])
    for k, col in (('SM0_7', 'el_soil_moisture_0_to_7cm_mean'), ('SM7_28', 'el_soil_moisture_7_to_28cm_mean'),
                   ('SM28_100', 'el_soil_moisture_28_to_100cm_mean'), ('SM100_255', 'el_soil_moisture_100_to_255cm_mean'),
                   ('GWETROOT', 'pw_gwetroot'), ('GWETPROF', 'pw_gwetprof')):
        if col in C and C[col].notna().mean() > 0.9:
            D[k] = _interp(C[col])
    if 'sealevel_m' in C:
        D['SL'] = C['sealevel_m'].values * 100.0                       # cm
    G = _pq('gnss_station.parquet')
    if sta in G.index.get_level_values(0):
        gg = G.loc[sta].reindex(DAYS)
        D['gU'] = gg.gnss_u_mm.values
        D['gDil'] = gg.gnss_dil_ns.values
        D['gDilPost'] = gg.gnss_dil_post_ns.values
    B = _pq('bstrain_areal.parquet').reindex(DAYS)
    if f'{sta}_hp' in B:
        D['bStr'] = B[f'{sta}_hp'].values
    cc = station_coords()
    if sta in cc.index:
        la, lo = cc.loc[sta, 'lat'], cc.loc[sta, 'lon']
        k = min(GAUGES, key=lambda s_: hav(la, lo, *GAUGES[s_]))
        q = streamflow()[k]
        D['logQ'] = np.log10(q.interpolate(limit=5).clip(lower=0.1)).values
        D['_gauge'] = k
    _cache[key] = D
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
    """Talwani-type pore-pressure diffusion: sum_t' (P(t') - Pbar) erfc(sqrt(tau_d / (t - t'))); tau_d = z^2/(4D) (days).
    Only short tau_d are used (<= 100 d): long ones turn the cumulative rain departure into a quasi-trend (review)."""
    lag = np.arange(1, ND + 1, dtype=float)
    k = erfc(np.sqrt(tau_d / lag))
    x = P - P[TY >= 1.0].mean()
    return anom(fftconvolve(x, k)[:ND])


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
        L = int(par)
        out = np.empty(ND)
        out[L:] = T[:ND - L] if L > 0 else T
        out[:L] = T[0]
        return out
    a = np.exp(-1.0 / par)                    # exponential (diffusive) low-pass, unit DC gain
    return lfilter([1 - a], [1.0, -a], T, zi=[a * T[0]])[0]


def annual_cols():
    w = 2 * np.pi * TY
    return np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


RAWKEY = {'hyd_exp': ['P'], 'hyd_diff': ['P'], 'rain_lags': ['P'], 'hyd_exp_rean': ['P_rean'], 'thermal': ['T'],
          'soilm': ['SM0_7', 'SM7_28', 'SM28_100', 'SM100_255', 'GWETROOT', 'GWETPROF'], 'streamflow': ['logQ'],
          'gnss_up': ['gU'], 'gnss_dil': ['gDil'], 'bstrain': ['bStr'], 'pressure': ['Ps'], 'sealevel': ['SL'], 'micro': ['_micro']}


def group_variants(g, D, band=None):
    """List of (label, X (ND,k)) for one candidate group built from raw drivers D (None if unavailable)."""
    if g == 'annual':
        return [('sin+cos 1+2/yr', annual_cols())]
    if g == 'hyd_exp':
        return [(f'tau_h={t}d', hyd_exp(D['P'], t)[:, None]) for t in (10, 30, 60, 120, 240, 480)]
    if g == 'hyd_diff':
        return [(f'tau_d={t}d', hyd_diff(D['P'], t)[:, None]) for t in (3, 10, 30, 100)]
    if g == 'hyd_exp_rean':
        return [(f'tau_h={t}d (POWER precip)', hyd_exp(D['P_rean'], t)[:, None]) for t in (30, 60, 120, 240)]
    if g == 'rain_lags':
        return [('0-10,10-30,30-90,90-180,180-365d', rain_lags(D['P']))]
    if g == 'soilm':
        v = [(k, anom(D[k])[:, None]) for k in RAWKEY['soilm'] if k in D]
        return v or None
    if g == 'streamflow':
        if 'logQ' not in D:
            return None
        return [(f'log10Q gauge {D["_gauge"]}, exp-smoothed {t}d', thermal(D['logQ'], 'exp', t)[:, None] if t > 1 else anom(D['logQ'])[:, None])
                for t in (1, 30, 90)]
    if g == 'thermal':
        return ([(f'exp tau_T={t}d', thermal(D['T'], 'exp', t)[:, None]) for t in (1, 15, 30, 60, 120)] +
                [(f'lag={L}d', thermal(D['T'], 'lag', L)[:, None]) for L in (30, 60, 90, 120)])
    if g == 'pressure':
        return [('surface pressure anomaly (POWER)', anom(D['Ps'])[:, None])]
    if g == 'sealevel':
        return [('North Spit monthly MSL', anom(D['SL'])[:, None])] if 'SL' in D else None
    if g == 'gnss_up':
        return [('local GNSS vertical (seasonal kept)', anom(D['gU'])[:, None])] if 'gU' in D and np.isfinite(D['gU']).sum() > 1000 else None
    if g == 'gnss_dil':
        return [('GNSS areal strain (non-tectonic)', anom(D['gDil'])[:, None])] if 'gDil' in D and np.isfinite(D['gDil']).sum() > 1000 else None
    if g == 'bstrain':
        return [('borehole areal strain, high-passed', anom(D['bStr'])[:, None])] if 'bStr' in D else None
    if g == 'micro':
        if '_micro' in D:
            return [(D['_micro_lab'], anom(D['_micro'])[:, None])]
        return None
    raise KeyError(g)


ALLGROUPS = ('annual', 'hyd_exp', 'hyd_diff', 'hyd_exp_rean', 'rain_lags', 'soilm', 'streamflow', 'thermal',
             'pressure', 'sealevel', 'gnss_up', 'gnss_dil', 'bstrain', 'micro')


def candidate_groups(sta, band=None, D=None):
    D = dict(D if D is not None else drivers(sta))
    if band is not None:
        p, col = power_band(band)
        if p is not None:
            D['_micro'] = p; D['_micro_lab'] = f'log10 power {col} Hz (B045 EHZ)'
    G = {}
    for g in ALLGROUPS:
        v = group_variants(g, D)
        if v:
            G[g] = v
    return G, D


def wy_permute(x, perm):
    """Rebuild a daily series with complete water years (Oct-Sep, WY2008..WY2026) re-ordered by perm
    (perm[k] = source WY for target WY k); seasonality within a year is kept, interannual timing scrambled."""
    out = np.array(x, float, copy=True)
    starts = [int(dayidx([pd.Timestamp(f'{y - 1}-10-01')])[0]) for y in range(2008, 2027)]
    ends = [int(dayidx([pd.Timestamp(f'{y}-09-30')])[0]) for y in range(2008, 2027)]
    for k, src in enumerate(perm):
        n = min(ends[k] - starts[k], ends[src] - starts[src]) + 1
        out[starts[k]:starts[k] + n] = x[starts[src]:starts[src] + n]
        if ends[k] - starts[k] + 1 > n:                   # leap-day mismatch: repeat last day
            out[starts[k] + n:ends[k] + 1] = x[starts[src] + n - 1]
    return out


# ------------------------------------------------------------------------------------------------ quakes
_Q = None


def quakes_for(sta):
    global _Q
    if _Q is None:
        _Q = pd.read_csv(f'{DRV}/quakes_station.csv.gz', parse_dates=['t'])
        _Q = _Q[_Q.kind == 'local']                     # teleseisms not used (measured LP PGV <= 0.1 cm/s)
    if sta == 'REGION':
        return _Q.groupby('qid').agg(t=('t', 'first'), mag=('mag', 'first'), modelled=('modelled', 'first'),
                                     pgv_site_cms=('pgv_site_cms', 'median'), vol_strain=('vol_strain', 'median')).reset_index()
    return _Q[_Q.sta == sta]


def quake_cols(sta, tau, mode='free3', tau_by=None):
    """Coseismic regressors.  Columns come in (perm, trans) pairs that are sign-constrained in the fit
    (same sign => drop0 = perm + trans, recovered fraction trans/drop0 in [0, 1]).
       'free3'      : big three free (perm + trans each) + transient-only PGV dose of all other local events
       'free3_log'  : as free3, dose uses max(0, log10(PGV / 0.05 cm/s))
       'free3_only' : big three only, no dose
       'pgv_all'    : the 13 events with source models (incl. big three): one shared PGV coefficient pair (perm, trans)
                      + transient-only PGV dose of the other local events
       'strain_all' : as pgv_all but the modelled events scale with signed static volumetric strain (nanostrain)
    PGV = BSSA14 (Rjb for finite sources, hypocentral distance otherwise) x station site factor, measured where available."""
    q = quakes_for(sta)
    d = quake_day(q.t)
    i = np.arange(ND)
    big_days = [big3_day(n) for n in BIG3]
    near_big = np.array([any(AFTERSHOCK_WIN[0] <= x - b <= AFTERSHOCK_WIN[1] for b in big_days) for x in d])
    st = lambda x0: (i >= x0).astype(float)
    tr = lambda x0, tq: np.where(i >= x0, np.exp(-(i - x0) / tq), 0.0)
    cols, names = [], []
    if mode in ('free3', 'free3_log', 'free3_only'):
        for n, b in zip(BIG3, big_days):
            cols += [st(b), tr(b, (tau_by or {}).get(n, tau))]
            names += [f'step_{n}', f'trans_{n}']
    pgv = q.pgv_site_cms.values
    if mode in ('pgv_all', 'strain_all'):
        mod = q.modelled.values.astype(bool)
        g = (q.vol_strain.values * 1e9) if mode == 'strain_all' else pgv
        Dp, Dt = np.zeros(ND), np.zeros(ND)
        for x0, gg in zip(d[mod], g[mod]):
            Dp += gg * st(x0); Dt += gg * tr(x0, tau)
        tag = 'strain' if mode == 'strain_all' else 'pgvmod'
        cols += [Dp, Dt]; names += [f'dose_{tag}_perm', f'dose_{tag}_trans']
        other = ~mod & ~near_big
    else:
        other = ~near_big
    if mode != 'free3_only':
        g = np.maximum(0.0, np.log10(np.maximum(pgv, 1e-6) / 0.05)) if mode == 'free3_log' else pgv
        Dt = np.zeros(ND)
        for x0, gg in zip(d[other], g[other]):
            if gg > 0:
                Dt += gg * tr(x0, tau)
        cols.append(Dt); names.append('dose_pgv_trans' if mode != 'free3_log' else 'dose_logpgv_trans')
    return np.column_stack(cols), names


def sign_pairs(names):
    """Index pairs (perm, trans) that must share a sign."""
    out = []
    for i, n in enumerate(names):
        if n.startswith('step_') and f'trans_{n[5:]}' in names:
            out.append((i, names.index(f'trans_{n[5:]}')))
        if n.endswith('_perm') and n.replace('_perm', '_trans') in names:
            out.append((i, names.index(n.replace('_perm', '_trans'))))
        if n == 'fake_step' and 'fake_trans' in names:
            out.append((i, names.index('fake_trans')))
    return out


# ------------------------------------------------------------------------------------------------ design
def base_cols(breaks):
    cols, names = [np.ones(ND)], ['const']
    for b in breaks:
        cols.append((DAYS >= pd.Timestamp(b)).astype(float)); names.append(f'offset_{str(b)[:10]}')
    return np.column_stack(cols), names


def long_cols():
    return {'trend': (TY - 10.0)[:, None],
            'ramp21': np.clip(TY - (pd.Timestamp(HINGE) - pd.Timestamp('2008-01-01')).days / 365.25, 0, None)[:, None]}


# ------------------------------------------------------------------------------------------------ fitting
def _solve(Xs, ys, pairs):
    """Least squares with same-sign constraints on column pairs.  If any pair of the unconstrained solution
    disagrees in sign, every sign combination of the disagreeing pairs is tried (bounded LSQ, <= 16 fits; agreeing
    pairs are bounded to their own sign) and the combination with the lowest SSE is kept."""
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    if not pairs:
        return beta
    bad = [(a, b) for a, b in pairs if beta[a] * beta[b] < 0]
    if not bad:
        return beta
    good = [(a, b) for a, b in pairs if beta[a] * beta[b] >= 0]
    sc = np.sqrt((Xs ** 2).sum(0)) + 1e-12                      # column scaling for the bounded solver
    best, best_sse = None, np.inf
    for signs in itertools.product((-1.0, 1.0), repeat=len(bad)):
        lb = np.full(len(beta), -np.inf); ub = np.full(len(beta), np.inf)
        for (a, b), sg in list(zip(bad, signs)) + [((a, b), np.sign(beta[a] + beta[b]) or -1.0) for a, b in good]:
            if sg > 0:
                lb[a] = lb[b] = 0.0
            else:
                ub[a] = ub[b] = 0.0
        r = lsq_linear(Xs / sc, ys, bounds=(lb * sc, ub * sc), method='bvls', lsmr_tol='auto')
        x = r.x / sc
        sse = float(np.sum((ys - Xs @ x) ** 2))
        if sse < best_sse:
            best, best_sse = x, sse
    return best


def huber_lsq(X, y, w=None, nit=4, k=1.5, pairs=None):
    """Robust (Huber IRLS) weighted LSQ with optional sign-pair constraints.  Returns beta, weights."""
    w0 = np.ones(len(y)) if w is None else w
    ww = w0.copy()
    for _ in range(nit):
        s = np.sqrt(ww)
        beta = _solve(X * s[:, None], y * s, pairs)
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
    if CV_ALL:
        return m
    for n in BIG3:
        b = big3_day(n)
        m &= ~((idx >= b - pre) & (idx <= b + post))
    return m


def cv_rms(X, y, idx, w=None, names=None, min_train=0.5):
    """Leave-one-water-year-out CV RMS (robust fit on the rest), scored outside the big-3 windows."""
    wy = water_year(idx)
    score = cv_score_mask(idx)
    err = np.full(len(y), np.nan)
    for g in np.unique(wy):
        te = wy == g
        tr = ~te
        if tr.sum() < min_train * len(y):
            continue
        sup = np.abs(X[tr]).sum(0) > 0
        nm = [n for n, s in zip(names, sup) if s] if names is not None else None
        b, _ = huber_lsq(X[tr][:, sup], y[tr], None if w is None else w[tr], nit=2, pairs=sign_pairs(nm) if nm else None)
        err[te] = y[te] - X[te][:, sup] @ b
    e = err[np.isfinite(err) & score]
    return float(np.sqrt(np.mean(e ** 2))), float(np.median(np.abs(e)) * 1.4826)


def decorr_length(r, spacing=1, maxlag=400):
    """Integrated decorrelation length in samples: 1 + 2 sum rho_k up to the first non-positive rho,
    at least the number of samples spanning 10 days (overlapping 10-day stacks)."""
    r = r - r.mean()
    v = np.dot(r, r) / len(r)
    s = 0.0
    for L in range(1, min(maxlag, len(r) // 3)):
        rho = np.dot(r[:-L], r[L:]) / (len(r) - L) / v
        if rho <= 0:
            break
        s += rho
    return max(max(1, int(np.ceil(10 / max(spacing, 1)))), 1 + 2 * s)


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
        self.cluster = None

    def design(self, X):
        Xs = smooth(X, self.sm)
        if self.pair_i is not None:
            return Xs[self.idx] - Xs[self.pair_i]
        return Xs[self.idx]

    def subset(self, m):
        S = Series(self.dataset, self.sta, self.chan, DAYS[self.idx[m]], self.y[m], self.sm,
                   None if self.w is None else self.w[m], self.breaks, None if self.pair_i is None else self.pair_i[m])
        if self.cluster is not None:
            S.cluster = self.cluster[m]
        return S


def assemble(S, blocks):
    Xs, names, groups = [], [], []
    for g, nm, X in blocks:
        A = S.design(X)
        Xs.append(A); names += list(nm); groups += [g] * A.shape[1]
    return np.column_stack(Xs), names, groups


def finite_rows(A, y):
    return np.isfinite(y) & np.isfinite(A).all(1)
