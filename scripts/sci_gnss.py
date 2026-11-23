#!/usr/bin/env python
"""Regional GNSS search for aseismic slip associated with LFE activity (Mendocino triple junction).

Data: Nevada Geodetic Laboratory (NGL) daily IGS20 tenv3 time series + NGL steps file (public, no token).
Stages (run in order; each caches its products in data/sci/gnss/):
  download  station selection (near <=110 km of the LFE centroid; far 150-400 km and >=200 km from the
            M7 epicentre for the common-mode estimate) and tenv3 download
  prep      per-station trajectory models (offsets from the NGL steps file, seasonal, quake log terms),
            common-mode error (CME) from far stations, cleaned near-station series
  q1        LFE-rate peaks (multi-week) stacks vs random-date placebos; continuous regression of an
            LFE-patch potency series on cumulative excess LFE activity (circular-shift placebo)
  q2        M7 (2024-12-05) postseismic: time-dependent inversion with M7 afterslip basis + LFE patch,
            placebo patch positions, timing vs LFE silence/restart and 60-90 d moment repayment
  q3        long-term velocity changes after 2021 (Petrolia/Ferndale/M7) vs an empirical break-date null,
            LFE-patch slip-rate change and its detection limit
  figs      figures/sci_gnss_*.png
Okada Green's functions come from scripts/sci_dose_response.py (imported, not modified).
"""
import sys, os, json, time, glob
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/gnss'
RAW = f'{OUT}/raw'
TEN = f'{RAW}/tenv3'
FIG = f'{ROOT}/figures'
UA = 'Mozilla/5.0 (compatible; generic-research-client)'
NGL = 'https://geodesy.unr.edu'

LFE_LAT, LFE_LON, LFE_Z = 40.06, -123.625, 28.0      # centroid (as in sci_strain_stack.py)
STRIKE, DIP, RAKE = 293.0, 40.0, 180.0               # NE-dipping right-lateral LFE plane
M7_LAT, M7_LON = 40.374, -125.0217
QUAKES = {'petrolia': ('2021-12-20', 6.2), 'ferndale': ('2022-12-20', 6.4), 'm7': ('2024-12-05', 7.0)}
MU = 30e9
KM = 111.195


def mjd2dt(m):
    return pd.Timestamp('1858-11-17') + pd.to_timedelta(np.asarray(m), unit='D')


def dt2mjd(d):
    return (pd.Timestamp(d) - pd.Timestamp('1858-11-17')).days


def en_lfe(lat, lon):
    """km east/north of the LFE centroid."""
    return ((np.asarray(lon) - LFE_LON) * KM * np.cos(np.radians(LFE_LAT)), (np.asarray(lat) - LFE_LAT) * KM)


def mw_from_potency(P):
    P = np.asarray(P, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log10(np.abs(P) * MU) - 9.1) / 1.5


# ------------------------------------------------------------------------------------------------
# download
# ------------------------------------------------------------------------------------------------
def holdings():
    d = pd.read_csv(f'{RAW}/DataHoldings.txt', sep=r'\s+', usecols=range(11), skiprows=1, header=None,
                    names='sta lat lon h x y z beg end mod n'.split())
    d['lon'] = ((d.lon + 180) % 360) - 180
    e, n = en_lfe(d.lat, d.lon)
    d['dist'] = np.hypot(e, n)
    d['dist_m7'] = np.hypot((d.lon - M7_LON) * KM * np.cos(np.radians(M7_LAT)), (d.lat - M7_LAT) * KM)
    d['beg'] = pd.to_datetime(d.beg, errors='coerce')
    d['end'] = pd.to_datetime(d.end, errors='coerce')
    return d


def fetch(url, fn, tries=3):
    import subprocess
    for k in range(tries):
        r = subprocess.run(['curl', '-s', '-f', '-A', UA, '--max-time', '300', '-o', fn + '.part', url])
        if r.returncode == 0 and os.path.getsize(fn + '.part') > 1000:
            os.replace(fn + '.part', fn)
            return True
        time.sleep(3)
    if os.path.exists(fn + '.part'):
        os.remove(fn + '.part')
    return False


def stage_download():
    os.makedirs(TEN, exist_ok=True)
    for f in ('DataHoldings.txt', 'steps.txt'):
        if not os.path.exists(f'{RAW}/{f}'):
            fetch(f'{NGL}/NGLStationPages/{f}', f'{RAW}/{f}')
    d = holdings()
    near = d[(d.dist <= 110) & (d.n >= 1500) & (d.end >= '2015-01-01')].copy()
    near['set'] = 'near'
    far = d[(d.dist >= 150) & (d.dist <= 400) & (d.dist_m7 >= 200) & (d.n >= 4000) &
            (d.beg <= '2011-01-01') & (d.end >= '2026-01-01')].copy()
    # thin far stations on a 0.35 deg grid, keeping the longest record per cell
    far['cell'] = (np.floor(far.lat / 0.35)).astype(int).astype(str) + '_' + (np.floor(far.lon / 0.35)).astype(int).astype(str)
    far = far.sort_values('n', ascending=False).groupby('cell').head(1)
    far['set'] = 'far'
    sel = pd.concat([near, far.drop(columns='cell')])
    sel.to_csv(f'{OUT}/stations.csv', index=False)
    print(f'near {len(near)}  far {len(far)}', flush=True)
    bad = []
    for s in sel.sta:
        fn = f'{TEN}/{s}.tenv3'
        if os.path.exists(fn) and os.path.getsize(fn) > 1000:
            continue
        ok = fetch(f'{NGL}/gps_timeseries/IGS20/tenv3/IGS20/{s}.tenv3', fn)
        if not ok:
            bad.append(s)
        time.sleep(0.5)
    print('failed:', bad, flush=True)


# ------------------------------------------------------------------------------------------------
# loading, steps, trajectory model
# ------------------------------------------------------------------------------------------------
def load_tenv3(sta):
    a = pd.read_csv(f'{TEN}/{sta}.tenv3', sep=r'\s+', skiprows=1, header=None)
    df = pd.DataFrame({'mjd': a[3].astype(int), 'e': a[7] + a[8], 'n': a[9] + a[10], 'u': a[11] + a[12],
                       'se': a[14], 'sn': a[15], 'su': a[16]})
    df = df.drop_duplicates('mjd').sort_values('mjd').reset_index(drop=True)
    return df


_STEPS = None


def steps_for(sta):
    """NGL steps: equipment (type 1) and earthquake (type 2) dates as MJD (deduplicated)."""
    global _STEPS
    if _STEPS is None:
        rows = []
        for l in open(f'{RAW}/steps.txt'):
            p = l.split()
            if len(p) < 3:
                continue
            try:
                d = pd.to_datetime(p[1], format='%y%b%d')
            except Exception:
                continue
            rows.append((p[0], dt2mjd(d), int(p[2]), p[-1] if p[2] == '2' else ' '.join(p[3:])))
        _STEPS = pd.DataFrame(rows, columns=['sta', 'mjd', 'typ', 'info'])
    s = _STEPS[_STEPS.sta == sta]
    return s.drop_duplicates('mjd').sort_values('mjd')


def design(mjd, step_mjd, logs=(), vbreaks=(), seasonal=True, t0=55000.0):
    """Columns: offset, velocity (per yr), [annual, semiannual], Heaviside steps, log(1+dt/tau) terms,
    velocity changes (ramps) at vbreaks.  logs = [(mjd_q, tau_d)].  Returns G, names."""
    t = (mjd - t0) / 365.25
    cols, names = [np.ones_like(t), t], ['off', 'vel']
    if seasonal:
        for k in (1, 2):
            cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
            names += [f's{k}', f'c{k}']
    for m in step_mjd:
        cols.append((mjd >= m).astype(float))
        names.append(f'step_{int(m)}')
    for m, tau in logs:
        dt = np.clip(mjd - m, 0, None)
        cols.append(np.log1p(dt / tau))
        names.append(f'log_{int(m)}_{tau:g}')
    for m in vbreaks:
        cols.append(np.clip(mjd - m, 0, None) / 365.25)
        names.append(f'dv_{int(m)}')
    return np.column_stack(cols), names


def usable_steps(mjd, smjd, nmin=10):
    """Keep steps with >= nmin observations on each side within 365 d; merge steps closer than 5 d."""
    keep = []
    for m in smjd:
        if keep and m - keep[-1] < 5:
            continue
        nb = ((mjd < m) & (mjd >= m - 365)).sum()
        na = ((mjd >= m) & (mjd < m + 365)).sum()
        if nb >= nmin and na >= nmin and m > mjd.min() and m <= mjd.max():
            keep.append(m)
    return keep


def robust_fit(G, y, nit=4, k=5.0):
    ok = np.isfinite(y)
    for _ in range(nit):
        x, *_ = np.linalg.lstsq(G[ok], y[ok], rcond=None)
        r = y - G @ x
        s = 1.4826 * np.nanmedian(np.abs(r[ok] - np.nanmedian(r[ok])))
        ok2 = np.isfinite(y) & (np.abs(r) < k * max(s, 1e-4))
        if (ok2 == ok).all():
            break
        ok = ok2
    return x, r, ok


def quake_mjd():
    return {k: dt2mjd(v[0]) for k, v in QUAKES.items()}


def base_model(df, sta, tau_m7=30.0, tau_m6=30.0, start=None):
    """Full trajectory model used for cleaning: offsets (equipment + all NGL quake steps), velocity,
    seasonal, log transients after the 3 local quakes, velocity breaks at the 3 quakes."""
    mjd = df.mjd.values.astype(float)
    st = steps_for(sta)
    sm = usable_steps(mjd, st.mjd.values.astype(float))
    Q = quake_mjd()
    logs, vb = [], []
    for k, m in Q.items():
        if mjd.max() > m + 60 and mjd.min() < m - 60:
            logs.append((m, tau_m7 if k == 'm7' else tau_m6))
            vb.append(m)
    G, names = design(mjd, sm, logs, vb)
    return G, names


# ------------------------------------------------------------------------------------------------
# prep: trajectory models, CME, cleaned series
# ------------------------------------------------------------------------------------------------
def stage_prep():
    S = pd.read_csv(f'{OUT}/stations.csv')
    S = S[[os.path.exists(f'{TEN}/{s}.tenv3') for s in S.sta]]
    day0, day1 = dt2mjd('2006-01-01'), dt2mjd('2026-11-15')
    days = np.arange(day0, day1)
    res = {c: {} for c in 'enu'}
    meta = []
    for _, r in S.iterrows():
        df = load_tenv3(r.sta)
        df = df[(df.mjd >= day0) & (df.mjd < day1)]
        if len(df) < 500:
            continue
        sig = df[['se', 'sn', 'su']].values
        okq = (sig < 4 * np.nanmedian(sig, axis=0)).all(axis=1)
        df = df[okq]
        G, names = base_model(df, r.sta)
        row = dict(sta=r.sta, set=r.set, lat=r.lat, lon=r.lon, dist=r.dist, n=len(df),
                   mjd0=int(df.mjd.min()), mjd1=int(df.mjd.max()), nsteps=sum(n.startswith('step') for n in names))
        for c in 'enu':
            y = df[c].values * 1e3        # mm
            x, rr, ok = robust_fit(G, y)
            rr[~ok] = np.nan
            v = np.full(len(days), np.nan)
            v[(df.mjd.values - day0).astype(int)] = rr
            res[c][r.sta] = v
            row[f'vel_{c}'] = x[names.index('vel')]
            row[f'wrms_{c}'] = float(np.nanstd(rr))
        meta.append(row)
        print(r.sta, r.set, len(df), {c: round(row[f'wrms_{c}'], 2) for c in 'enu'}, flush=True)
    M = pd.DataFrame(meta)
    far = M[M.set == 'far'].sta.tolist()
    cme = {}
    for c in 'enu':
        R = np.vstack([res[c][s] for s in far])
        n = np.isfinite(R).sum(0)
        med = np.nanmedian(R, axis=0)
        med[n < 8] = np.nan
        cme[c] = med
    C = pd.DataFrame({'mjd': days, **{f'cme_{c}': cme[c] for c in 'enu'},
                      'nfar': np.isfinite(np.vstack([res['e'][s] for s in far])).sum(0)})
    C.to_csv(f'{OUT}/cme.csv.gz', index=False)
    # cleaned near-station series: raw data minus CME, then refit the base model; store
    # (a) "clean" = data - steps - seasonal - CME (velocity, log terms and vbreaks retained) and
    # (b) residual after the full base model
    cl, rs = [], []
    for _, r in M[M.set == 'near'].iterrows():
        df = load_tenv3(r.sta)
        df = df[(df.mjd >= day0) & (df.mjd < day1)]
        sig = df[['se', 'sn', 'su']].values
        df = df[(sig < 4 * np.nanmedian(sig, axis=0)).all(axis=1)]
        idx = (df.mjd.values - day0).astype(int)
        G, names = base_model(df, r.sta)
        out = pd.DataFrame({'sta': r.sta, 'mjd': df.mjd.values})
        out2 = out.copy()
        for c in 'enu':
            y = df[c].values * 1e3 - cme[c][idx]
            x, rr, ok = robust_fit(G, y)
            nuis = [i for i, nm in enumerate(names) if nm.startswith('step') or nm in ('off', 's1', 'c1', 's2', 'c2')]
            yc = y - G[:, nuis] @ x[nuis]
            yc[~ok] = np.nan
            rr[~ok] = np.nan
            out[c] = yc
            out2[c] = rr
            M.loc[M.sta == r.sta, f'wrms_cme_{c}'] = float(np.nanstd(rr))
            for k, nm in enumerate(names):
                if nm.startswith('log_') or nm.startswith('dv_') or nm == 'vel':
                    M.loc[M.sta == r.sta, f'{c}_{nm}'] = x[k]
        cl.append(out)
        rs.append(out2)
    pd.concat(cl).to_csv(f'{OUT}/near_clean.csv.gz', index=False)
    pd.concat(rs).to_csv(f'{OUT}/near_resid.csv.gz', index=False)
    M.to_csv(f'{OUT}/station_meta.csv', index=False)
    nr = M[M.set == 'near']
    print('median WRMS near before/after CME (mm):',
          {c: (round(nr[f'wrms_{c}'].median(), 2), round(nr[f'wrms_cme_{c}'].median(), 2)) for c in 'enu'})


# ------------------------------------------------------------------------------------------------
# shared helpers for q1-q3
# ------------------------------------------------------------------------------------------------
DAY0 = dt2mjd('2006-01-01')
CAT = f'{ROOT}/data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv'
MP = 1e6            # potency unit (m^3); 1e6 m^3 = M0 3e16 N m = Mw 4.92
LFE_PATCH = dict(L=16.0, W=15.6, dep=27.0)          # LFE zone: 15-16 km along strike, 22-32 km depth on dip 40
BIG_PATCH = dict(L=40.0, W=31.0, dep=30.0)          # broader slow-slip zone (20-40 km depth)


def okada():
    sys.path.insert(0, f'{ROOT}/scripts')
    import sci_dose_response as dr
    return dr


def near_cube(kind='clean', maxdist=110.0):
    """Returns days (mjd), stations DataFrame, Y (ndays, nsta, 3) in mm."""
    M = pd.read_csv(f'{OUT}/station_meta.csv')
    M = M[(M.set == 'near') & (M.dist <= maxdist)].reset_index(drop=True)
    D = pd.read_csv(f'{OUT}/near_{kind}.csv.gz')
    days = np.arange(DAY0, dt2mjd('2026-11-15'))
    Y = np.full((len(days), len(M), 3), np.nan)
    for j, s in enumerate(M.sta):
        d = D[D.sta == s]
        idx = (d.mjd.values - DAY0).astype(int)
        for k, c in enumerate('enu'):
            Y[idx, j, k] = d[c].values
    return days, M, Y


def green(M, srcs=None, e0=0.0, n0=0.0, patch=LFE_PATCH, strike=STRIKE, dip=DIP, rake=RAKE):
    """Displacement (nsta, 3) in mm per MP (1e6 m^3) potency on the patch (centre shifted by e0, n0 km)."""
    dr = okada()
    e, n = en_lfe(M.lat.values, M.lon.values)
    P = np.column_stack([e, n, np.zeros(len(e))])
    if srcs is None:
        slip = MP / (patch['L'] * patch['W'] * 1e6)
        srcs = [dr.Rect(e0, n0, patch['dep'], strike, dip, rake, patch['L'], patch['W'], slip)]
    return dr.disp_sum(srcs, P) * 1e3


def station_sigma(M):
    """Daily noise (mm) per station/component from the full-model residuals."""
    R = pd.read_csv(f'{OUT}/near_resid.csv.gz')
    s = R.groupby('sta')[['e', 'n', 'u']].agg(lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))
    return s.loc[M.sta].values


def lfe_daily():
    c = pd.read_csv(CAT)
    c.columns = [x.strip('%') for x in c.columns]
    t = pd.to_datetime(dict(year=c.year, month=c.month, day=c.day))
    d = t.value_counts().sort_index().asfreq('D', fill_value=0)
    return d


def wls_scalar(g, a, s):
    """Weighted LS scalar fit a = P g; returns P, formal sigma."""
    ok = np.isfinite(a) & np.isfinite(s) & (s > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    w = 1 / s[ok] ** 2
    den = np.sum(w * g[ok] ** 2)
    return np.sum(w * g[ok] * a[ok]) / den, 1 / np.sqrt(den)


# ------------------------------------------------------------------------------------------------
# Q1: multi-week LFE-rate peaks and continuous regression
# ------------------------------------------------------------------------------------------------
def window_ramps(days, Y, t0, half=90, r0=-7.5, r1=7.5, nmin=40):
    """Per station/component ramp amplitude (mm) over [t0+r0, t0+r1] in a +-half day window with a
    local offset + trend.  Returns amp, sig (nsta,3)."""
    i0 = int(t0 - half - days[0]); i1 = int(t0 + half - days[0]) + 1
    if i0 < 0 or i1 > len(days):
        return None, None
    tt = days[i0:i1] - t0
    ramp = np.clip((tt - r0) / (r1 - r0), 0, 1)
    G = np.column_stack([np.ones_like(tt), tt / 365.25, ramp])
    W = Y[i0:i1]
    amp = np.full(W.shape[1:], np.nan); sig = np.full(W.shape[1:], np.nan)
    for j in range(W.shape[1]):
        for k in range(3):
            y = W[:, j, k]
            ok = np.isfinite(y)
            if (ok & (tt < r0)).sum() < nmin or (ok & (tt > r1)).sum() < nmin:
                continue
            x, res, *_ = np.linalg.lstsq(G[ok], y[ok], rcond=None)
            r = y[ok] - G[ok] @ x
            s2 = np.sum(r ** 2) / max(ok.sum() - 3, 1)
            C = np.linalg.inv(G[ok].T @ G[ok]) * s2
            amp[j, k] = x[2]; sig[j, k] = np.sqrt(C[2, 2])
    return amp, sig


def stage_q1():
    days, M, Y = near_cube('clean')
    g = green(M); gb = green(M, patch=BIG_PATCH)
    d = lfe_daily()
    d = d['2009-01-01':]
    r15 = d.rolling(15, center=True).mean()
    base = d.rolling(365, center=True, min_periods=180).median()
    rel = (r15 / base).dropna()
    Q = {k: pd.Timestamp(v[0]) for k, v in QUAKES.items()}
    def excluded(t, pad=60):
        if any(abs((t - q).days) < pad for q in Q.values()):
            return True
        if Q['m7'] - pd.Timedelta('30D') <= t <= Q['m7'] + pd.Timedelta('200D'):
            return True
        return t > pd.Timestamp('2026-06-10') or t < pd.Timestamp('2009-03-01')
    loc = rel[(rel == rel.rolling(121, center=True, min_periods=30).max())]
    loc = loc[[not excluded(t) for t in loc.index]].sort_values(ascending=False)
    peaks = []
    for t, v in loc.items():
        if all(abs((t - p).days) >= 120 for p, _ in peaks):
            peaks.append((t, v))
        if len(peaks) == 20:
            break
    P = pd.DataFrame(peaks, columns=['date', 'rel15'])
    P['excess_lfe'] = [float((d[t - pd.Timedelta('7D'):t + pd.Timedelta('7D')] -
                              base[t - pd.Timedelta('7D'):t + pd.Timedelta('7D')]).sum()) for t in P.date]
    P['base_rate'] = [float(base[t]) for t in P.date]
    P['mjd'] = [dt2mjd(t) for t in P.date]
    P = P.sort_values('date').reset_index(drop=True)
    print(P.to_string(), flush=True)
    # candidate dates for the placebo: every 3 d, not within 20 d of a peak or excluded
    cand = [t for t in pd.date_range('2009-03-01', '2026-06-10', freq='3D')
            if not excluded(t) and all(abs((t - p).days) > 20 for p in P.date)]
    res = {}
    for var, (r0, r1) in {'ramp15': (-7.5, 7.5), 'ramp30': (-15, 15)}.items():
        rows = []
        for lab, dates in (('peak', list(P.date)), ('cand', cand)):
            for t in dates:
                m = dt2mjd(t)
                amp, sig = window_ramps(days, Y, m, r0=r0, r1=r1)
                if amp is None:
                    continue
                sig = np.where(np.isfinite(sig), np.maximum(sig, 0.3), np.nan)   # floor (mm)
                p, sp = wls_scalar(g.ravel(), amp.ravel(), sig.ravel())
                pb, spb = wls_scalar(gb.ravel(), amp.ravel(), sig.ravel())
                ph, sph = wls_scalar(g[:, :2].ravel(), amp[:, :2].ravel(), sig[:, :2].ravel())
                rows.append(dict(kind=lab, date=t, mjd=m, P=p, sP=sp, P_big=pb, P_hor=ph,
                                 nobs=int(np.isfinite(amp).sum())))
                if lab == 'peak':
                    res.setdefault(f'amps_{var}', []).append(amp)
        R = pd.DataFrame(rows)
        R.to_csv(f'{OUT}/q1_window_potency_{var}.csv', index=False)
        pk = R[R.kind == 'peak'].merge(P[['mjd', 'excess_lfe', 'rel15']], on='mjd')
        cd = R[R.kind == 'cand'].dropna(subset=['P'])
        rng = np.random.default_rng(1)
        n = len(pk)
        out = {}
        for col in ('P', 'P_big', 'P_hor'):
            obs = pk[col].mean()
            null = np.array([cd[col].values[rng.choice(len(cd), n, replace=False)].mean() for _ in range(5000)])
            out[col] = dict(obs_mean_MP=float(obs), obs_Mw_equiv=float(mw_from_potency(obs * MP)),
                            null_sd_MP=float(null.std()), null_q025=float(np.quantile(null, .025)),
                            null_q975=float(np.quantile(null, .975)),
                            p_two_sided=float(np.mean(np.abs(null - null.mean()) >= abs(obs - null.mean()))),
                            z=float((obs - null.mean()) / null.std()),
                            bound95_MP=float(abs(obs) + 1.96 * null.std()) if False else float(max(obs, 0) + 1.96 * null.std()),
                            single_window_sd_MP=float(cd[col].std()),
                            single_window_robust_sd_MP=float(1.4826 * np.median(np.abs(cd[col] - cd[col].median()))))
            out[col]['bound95_Mw'] = float(mw_from_potency(out[col]['bound95_MP'] * MP))
            # season-matched placebo: each peak replaced by a random date within +-30 d of its day of year
            doy_c = pd.to_datetime(cd.date).dt.dayofyear.values
            pools = []
            for t in pk.date:
                dd = np.abs(doy_c - pd.Timestamp(t).dayofyear); dd = np.minimum(dd, 365 - dd)
                pools.append(np.where(dd <= 30)[0])
            nullS = np.array([np.mean([cd[col].values[rng.choice(pl)] for pl in pools]) for _ in range(3000)])
            out[col]['season_matched'] = dict(null_mean=float(nullS.mean()), null_sd=float(nullS.std()),
                                              z=float((obs - nullS.mean()) / nullS.std()),
                                              p_two_sided=float(np.mean(np.abs(nullS - nullS.mean()) >= abs(obs - nullS.mean()))))
            out[col]['single_window_2sd_Mw'] = float(mw_from_potency(2 * out[col]['single_window_robust_sd_MP'] * MP))
        # does window potency scale with LFE excess across peaks, and across all candidate dates?
        from scipy import stats
        out['peaks_spearman_P_vs_excess'] = list(map(float, stats.spearmanr(pk.P, pk.excess_lfe)))
        res[var] = out
        print(var, json.dumps(out, indent=1), flush=True)
    # per-station stacked ramp amplitudes for the figure
    A = np.array(res.pop('amps_ramp15')); res.pop('amps_ramp30', None)
    st = pd.DataFrame({'sta': M.sta, 'lat': M.lat, 'lon': M.lon,
                       **{f'stack_{c}': np.nanmean(A[:, :, k], 0) for k, c in enumerate('enu')},
                       **{f'g_{c}_perMP': g[:, k] for k, c in enumerate('enu')}})
    st.to_csv(f'{OUT}/q1_station_stack_ramp15.csv', index=False)
    P.to_csv(f'{OUT}/q1_peaks.csv', index=False)

    # ---- continuous regression: network LFE-patch projection vs cumulative excess LFE count ----
    Yr = near_cube('resid')[2]
    sg = station_sigma(M)
    w = 1 / sg ** 2
    num = np.nansum(np.where(np.isfinite(Yr), Yr * g[None] * w[None], 0), axis=(1, 2))
    den = np.nansum(np.where(np.isfinite(Yr), g[None] ** 2 * w[None], 0), axis=(1, 2))
    nst = np.isfinite(Yr[:, :, 0]).sum(1)
    p = np.where(nst >= 15, num / np.where(den > 0, den, np.nan), np.nan)     # MP, daily
    ser = pd.DataFrame({'mjd': days, 'p_MP': p, 'nsta': nst})
    ser.to_csv(f'{OUT}/q1_daily_patch_projection.csv.gz', index=False)
    dfull = lfe_daily()
    dd = pd.Series(0.0, index=mjd2dt(days))
    dd.loc[dfull.index.intersection(dd.index)] = dfull.loc[dfull.index.intersection(dd.index)].values
    ok_cat = (days >= dt2mjd('2009-01-01')) & (days <= dt2mjd('2026-06-30'))
    C = np.cumsum(np.where(ok_cat, dd.values, 0.0))
    reg = {}
    for lab, mask_extra in (('all', None), ('excl_M7_year', (dt2mjd('2024-11-20'), dt2mjd('2025-12-31')))):
        okt = ok_cat & np.isfinite(p)
        if mask_extra:
            okt &= ~((days >= mask_extra[0]) & (days <= mask_extra[1]))
        # residualise C against the common trajectory basis (offset, velocity, seasonal, breaks, logs)
        Qm = quake_mjd()
        G, _ = design(days.astype(float), [], [(m, 30.0) for m in Qm.values()], list(Qm.values()))
        G = np.column_stack([G] + [(days >= m).astype(float) for m in Qm.values()])
        xc, *_ = np.linalg.lstsq(G[okt], C[okt], rcond=None)
        Cr = C - G @ xc
        xp, *_ = np.linalg.lstsq(G[okt], p[okt], rcond=None)
        pr = p - G @ xp
        # 10-day bins
        b = ((days - days[0]) // 10).astype(int)
        dfb = pd.DataFrame({'b': b[okt], 'p': pr[okt], 'C': Cr[okt]}).groupby('b').mean()
        k = np.sum(dfb.p * dfb.C) / np.sum(dfb.C ** 2)
        rng = np.random.default_rng(2)
        nb = len(dfb)
        null = []
        for _ in range(3000):
            sh = rng.integers(37, nb - 37)
            cs = np.roll(dfb.C.values, sh)
            null.append(np.sum(dfb.p.values * cs) / np.sum(cs ** 2))
        null = np.array(null)
        r = np.corrcoef(dfb.p, dfb.C)[0, 1]
        reg[lab] = dict(k_m3_per_LFE=float(k * MP), null_sd_m3_per_LFE=float(null.std() * MP),
                        p_two_sided=float(np.mean(np.abs(null) >= abs(k))), corr=float(r),
                        bound95_m3_per_LFE=float((max(k, 0) + 1.96 * null.std()) * MP),
                        n_bins=int(nb), C_res_sd_LFE=float(dfb.C.std()))
        reg[lab]['bound95_ratio_to_LFE_potency'] = reg[lab]['bound95_m3_per_LFE'] / 7.5
        if lab == 'all':
            pd.DataFrame({'mjd': days[0] + dfb.index * 10 + 5, 'p_res_MP': dfb.p, 'C_res': dfb.C}).to_csv(
                f'{OUT}/q1_continuous_bins.csv', index=False)
    res['continuous'] = reg
    res['notes'] = dict(potency_unit='MP = 1e6 m^3 (M0 3e16 N m, Mw 4.92)',
                        lfe_patch=LFE_PATCH, big_patch=BIG_PATCH, geometry=dict(strike=STRIKE, dip=DIP, rake=RAKE),
                        lfe_potency_per_event_m3='7.5 (median LFE Mw 1.5, mu 30 GPa; NOTES moments)',
                        n_peaks=len(P), n_placebo_candidates=len(cand))
    print(json.dumps(reg, indent=1))
    json.dump(res, open(f'{OUT}/q1_summary.json', 'w'), indent=1, default=str)


# ------------------------------------------------------------------------------------------------
# Q2: M7 postseismic
# ------------------------------------------------------------------------------------------------
EPOCH_EDGES = np.array([3, 5, 7, 10, 14, 18, 25, 35, 45, 60, 75, 90, 120, 150, 180, 240, 300, 365, 450, 540, 630])


def m7_basis(M):
    """Displacement (nsta,3) mm per MP for M7 afterslip basis sources (right-lateral):
    'ffm' = coseismic FFM slip pattern scaled; 'deep' = FFM plane 24-40 km; 'east' = MFZ extension
    from the FFM east end 25 km further east (0-25 km)."""
    dr = okada()
    srcs, hdr, a = dr.ffm_rects('m7.fsp')
    e, n = dr.ll2en(M.lat.values, M.lon.values)
    P = np.column_stack([e, n, np.zeros(len(e))])
    pot = sum(r.slip * r.L * r.W * 1e6 for r in srcs)
    B = {'ffm': dr.disp_sum(srcs, P) * 1e3 * MP / pot}
    st = np.radians(hdr['strike'])
    ux = np.array([np.sin(st), np.cos(st)])
    E = np.array([[r.e, r.n] for r in srcs])
    s = E @ ux
    c = E[np.argmin(s)] + 0.5 * (s.max() - s.min()) * ux          # mid-trace
    L = s.max() - s.min() + hdr['dx']
    dip, strike = hdr['dip'], hdr['strike']
    for name, (cen, dep, LL, WW) in {'deep': (c, 32.0, L, 16.0),
                                     'east': (E[np.argmax(s)] + (hdr['dx'] / 2 + 12.5) * ux, 12.5, 25.0, 25.0)}.items():
        slip = MP / (LL * WW * 1e6)
        B[name] = dr.Rect(float(cen[0]), float(cen[1]), dep, strike, dip, 180.0, LL, WW, slip).disp(P) * 1e3
    return B


def post_epochs(days, Y, tq, pre=640, nmin_pre=200):
    """Displacement after tq relative to a linear pre-trend [tq-pre, tq-1] and referenced to days +1..+2.
    Returns (nepoch, nsta, 3) bin means over EPOCH_EDGES intervals (bin i = [edge_i, edge_i+1))."""
    i = int(tq - days[0])
    tt = days - tq
    out = np.full((len(EPOCH_EDGES) - 1, Y.shape[1], 3), np.nan)
    pm = (tt >= -pre) & (tt < 0)
    for j in range(Y.shape[1]):
        for k in range(3):
            y = Y[:, j, k]
            ok = pm & np.isfinite(y)
            if ok.sum() < nmin_pre or np.isfinite(y[(tt >= 1) & (tt <= 2)]).sum() < 1:
                continue
            A = np.column_stack([np.ones(ok.sum()), tt[ok]])
            x, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
            u = y - (x[0] + x[1] * tt)
            u = u - np.nanmean(u[(tt >= 1) & (tt <= 2)])
            for b in range(len(EPOCH_EDGES) - 1):
                m = (tt >= EPOCH_EDGES[b]) & (tt < EPOCH_EDGES[b + 1])
                if np.isfinite(u[m]).sum() >= max(1, 0.4 * m.sum()):
                    out[b, j, k] = np.nanmean(u[m])
    return out


def invert_epoch(u, sig, cols, wup=1.0, robust=False):
    """u, sig (nsta,3); cols: list of (nsta,3) basis arrays.  Returns x, formal sd, chi2, dof.
    robust: Huber reweighting (k = 2.5 normalised residual) to limit single-station anomalies."""
    d = u.ravel(); s = sig.ravel().copy()
    s[2::3] /= wup
    G = np.column_stack([c.ravel() for c in cols])
    ok = np.isfinite(d) & np.isfinite(s) & (s > 0) & np.isfinite(G).all(1)
    if ok.sum() <= G.shape[1] + 2:
        return np.full(G.shape[1], np.nan), np.full(G.shape[1], np.nan), np.nan, 0
    Gw = G[ok] / s[ok, None]; dw = d[ok] / s[ok]
    x, *_ = np.linalg.lstsq(Gw, dw, rcond=None)
    if robust:
        for _ in range(10):
            r = np.abs(dw - Gw @ x)
            h = np.where(r <= 2.5, 1.0, 2.5 / np.maximum(r, 1e-9))
            x, *_ = np.linalg.lstsq(Gw * np.sqrt(h)[:, None], dw * np.sqrt(h), rcond=None)
        Gw = Gw * np.sqrt(h)[:, None]; dw = dw * np.sqrt(h)
    C = np.linalg.pinv(Gw.T @ Gw)
    chi2 = float(np.sum((dw - Gw @ x) ** 2))
    return x, np.sqrt(np.diag(C)), chi2, int(ok.sum() - G.shape[1])


def nuisance_cols(M, kind):
    e, n = en_lfe(M.lat.values, M.lon.values)
    cols = []
    if 'trans' in kind:
        for k in range(3):
            c = np.zeros((len(M), 3)); c[:, k] = 1; cols.append(c)
    if 'grad' in kind:
        for k in range(2):
            for v in (e, n):
                c = np.zeros((len(M), 3)); c[:, k] = v / 50.0; cols.append(c)
    return cols


Q2_VARIANTS = {'lfe+trans': ('trans',), 'lfe+m7': ('m7',), 'lfe+m7+trans': ('m7', 'trans'),
               'lfe+m7+trans+grad': ('m7', 'trans', 'grad')}


def q2_solve(U, sig, g, B, M, variant):
    kind = Q2_VARIANTS[variant]
    cols = [g]
    if 'm7' in kind:
        cols += [B['ffm'], B['deep'], B['east']]
    cols += nuisance_cols(M, kind)
    X = []; SD = []; CH = []
    for b in range(U.shape[0]):
        x, sd, chi2, dof = invert_epoch(U[b], sig[b], cols)
        X.append(x); SD.append(sd); CH.append((chi2, dof))
    return np.array(X), np.array(SD), CH


def stage_q2():
    days, M, Y = near_cube('clean', maxdist=100.0)
    tq = dt2mjd(QUAKES['m7'][0])
    U = post_epochs(days, Y, tq)
    # placebo dates: every 30 d, post windows end before Petrolia
    fake = [dt2mjd(t) for t in pd.date_range('2011-06-01', '2020-02-01', freq='30D')]
    UF = np.array([post_epochs(days, Y, t) for t in fake])
    print('epochs done', UF.shape, flush=True)
    # empirical per-epoch/station/component noise from the placebo dates (robust)
    med = np.nanmedian(UF, axis=0)
    sig = 1.4826 * np.nanmedian(np.abs(UF - med), axis=0)
    sig = np.where(np.isfinite(sig), np.maximum(sig, 0.3), np.nan)
    g = green(M); B = m7_basis(M)
    mid = np.sqrt(EPOCH_EDGES[:-1] * EPOCH_EDGES[1:])
    out = {'epoch_mid_days': mid.tolist(), 'n_placebo_dates': len(fake), 'n_stations': int(len(M))}
    rows = []
    for var in Q2_VARIANTS:
        X, SD, CH = q2_solve(U, sig, g, B, M, var)
        XF = np.array([q2_solve(UF[i], sig, g, B, M, var)[0][:, 0] for i in range(len(fake))])
        nm = np.nanmean(XF, 0); ns = np.nanstd(XF, 0)
        for b in range(len(mid)):
            rows.append(dict(variant=var, epoch_lo=EPOCH_EDGES[b], epoch_hi=EPOCH_EDGES[b + 1], mid=mid[b],
                             P_lfe_MP=X[b, 0], formal_sd=SD[b, 0], null_mean=nm[b], null_sd=ns[b],
                             z=(X[b, 0] - nm[b]) / ns[b],
                             p_emp=float(np.mean(np.abs(XF[:, b] - nm[b]) >= abs(X[b, 0] - nm[b]))),
                             bound95_MP=max(X[b, 0], 0) + 1.96 * ns[b],
                             chi2=CH[b][0], dof=CH[b][1],
                             **({f'm7_{k}_MP': X[b, i + 1] for i, k in enumerate(['ffm', 'deep', 'east'])}
                                if 'm7' in Q2_VARIANTS[var] else {})))
    R = pd.DataFrame(rows)
    R['bound95_Mw'] = mw_from_potency(R.bound95_MP * MP)
    R.to_csv(f'{OUT}/q2_lfe_patch_timeseries.csv', index=False)
    # model comparison: chi2 improvement from adding the LFE patch (preferred variant) vs placebo dates
    var = 'lfe+m7+trans'
    def chi_drop(Ue):
        k = Q2_VARIANTS[var]
        base = [B['ffm'], B['deep'], B['east']] + nuisance_cols(M, k)
        out = []
        for b in range(Ue.shape[0]):
            c1 = invert_epoch(Ue[b], sig[b], [g] + base)[2]
            c0 = invert_epoch(Ue[b], sig[b], base)[2]
            out.append(c0 - c1)
        return np.array(out)
    dchi = chi_drop(U); dchiF = np.array([chi_drop(UF[i]) for i in range(len(fake))])
    # total postseismic signal: chi2 of the raw epoch field vs placebo
    chi_raw = np.nansum((U / sig) ** 2, axis=(1, 2)); chi_rawF = np.nansum((UF / sig[None]) ** 2, axis=(2, 3))
    # placebo patch positions at the real date
    grid = []
    for e0 in np.arange(-60, 61, 10):
        for n0 in np.arange(-60, 61, 10):
            gg = green(M, e0=e0, n0=n0)
            X, SD, CH = q2_solve(U, sig, gg, B, M, var)
            grid.append(dict(e0=e0, n0=n0, **{f'P_{int(EPOCH_EDGES[b])}_{int(EPOCH_EDGES[b+1])}': X[b, 0] for b in range(len(mid))},
                             **{f'z_{int(EPOCH_EDGES[b])}_{int(EPOCH_EDGES[b+1])}': X[b, 0] / SD[b, 0] for b in range(len(mid))}))
    Gd = pd.DataFrame(grid); Gd.to_csv(f'{OUT}/q2_patch_position_placebo.csv', index=False)
    # LFE timing (catalog): cumulative count deficit after the M7 relative to the pre-M7 90-d rate
    dl = lfe_daily()
    t0 = pd.Timestamp(QUAKES['m7'][0])
    pre_rate = dl[t0 - pd.Timedelta('365D'):t0 - pd.Timedelta('1D')].mean()
    post = dl[t0:t0 + pd.Timedelta('630D')]
    deficit = (post - pre_rate).cumsum()
    pd.DataFrame({'days': (post.index - t0).days, 'count': post.values, 'cum_excess': deficit.values}).to_csv(
        f'{OUT}/q2_lfe_cum_excess_after_m7.csv', index=False)
    # per-station observed displacements (for maps)
    pd.DataFrame([dict(sta=s, lat=M.lat[j], lon=M.lon[j], epoch_lo=EPOCH_EDGES[b], epoch_hi=EPOCH_EDGES[b + 1],
                       **{f'u_{c}': U[b, j, k] for k, c in enumerate('enu')},
                       **{f'sig_{c}': sig[b, j, k] for k, c in enumerate('enu')},
                       **{f'g_{c}': g[j, k] for k, c in enumerate('enu')},
                       **{f'ffm_{c}': B['ffm'][j, k] for k, c in enumerate('enu')})
                  for b in range(len(mid)) for j, s in enumerate(M.sta)]).to_csv(f'{OUT}/q2_station_epochs.csv', index=False)
    np.save(f'{OUT}/q2_placebo_epochs.npy', UF)
    lab = [f'{EPOCH_EDGES[b]}-{EPOCH_EDGES[b+1]}d' for b in range(len(mid))]
    out['dchi2_lfe_patch'] = {lab[b]: dict(obs=float(dchi[b]), placebo_q95=float(np.nanquantile(dchiF[:, b], .95)),
                                           p=float(np.mean(dchiF[:, b] >= dchi[b]))) for b in range(len(mid))}
    out['raw_field_chi2'] = {lab[b]: dict(obs=float(chi_raw[b]), placebo_median=float(np.median(chi_rawF[:, b])),
                                          placebo_q95=float(np.quantile(chi_rawF[:, b], .95)),
                                          p=float(np.mean(chi_rawF[:, b] >= chi_raw[b]))) for b in range(len(mid))}
    for b in range(len(mid)):
        col = f'P_{EPOCH_EDGES[b]}_{EPOCH_EDGES[b+1]}'
        v = Gd[col].values; lfe = Gd[(Gd.e0 == 0) & (Gd.n0 == 0)][col].iloc[0]
        out.setdefault('patch_position_rank', {})[lab[b]] = dict(
            P_lfe=float(lfe), frac_positions_with_larger_abs=float(np.mean(np.abs(v) > abs(lfe))),
            best_e0=float(Gd.e0[np.argmax(Gd[col.replace('P_', 'z_')])]), best_n0=float(Gd.n0[np.argmax(Gd[col.replace('P_', 'z_')])]))
    out['pre_m7_lfe_rate_per_day'] = float(pre_rate)
    out['lfe_cum_excess_at'] = {str(d): float(deficit.iloc[d]) for d in (7, 18, 25, 60, 90, 180, 365) if d < len(deficit)}
    summ = R[R.variant == 'lfe+m7+trans'][['epoch_lo', 'epoch_hi', 'P_lfe_MP', 'null_sd', 'z', 'p_emp', 'bound95_Mw']]
    print(summ.round(3).to_string())
    print(json.dumps(out, indent=1, default=float)[:4000])
    json.dump(out, open(f'{OUT}/q2_summary.json', 'w'), indent=1, default=float)


# ------------------------------------------------------------------------------------------------
# Q3: long-term velocity changes after 2021
# ------------------------------------------------------------------------------------------------
def q3_fit_station(df, sta, qdates, start, end, tau):
    """Velocity-change fit for one station on [start, end] with velocity breaks + steps + optional
    log terms at qdates.  tau None: no log terms, data within 60 d after each break are dropped."""
    m = df[(df.mjd >= start) & (df.mjd <= end)]
    mjd = m.mjd.values.astype(float)
    if tau is None:
        keep = np.ones(len(mjd), bool)
        for q in qdates:
            keep &= ~((mjd >= q) & (mjd < q + 60))
        m = m[keep]; mjd = mjd[keep]
    if len(mjd) < 400 or (mjd < qdates[0]).sum() < 365 or (mjd > qdates[-1]).sum() < 200:
        return None
    st = steps_for(sta)
    sm = usable_steps(mjd, st.mjd.values.astype(float))
    sm = sorted(set(sm) | {q for q in qdates if mjd.min() < q < mjd.max()})
    logs = [] if tau is None else [(q, tau) for q in qdates]
    G, names = design(mjd, sm, logs, qdates)
    if len(mjd) < 400:
        return None
    out = {}
    for c in 'enu':
        x, r, ok = robust_fit(G, m[c].values)
        dv = [x[names.index(f'dv_{int(q)}')] for q in qdates]
        out[c] = dict(v0=x[names.index('vel')], d1=dv[0], d2=dv[0] + dv[1], d3=dv[0] + dv[1] + dv[2])
    return out


def stage_q3():
    days, M, _ = near_cube('clean', maxdist=110.0)
    cme = pd.read_csv(f'{OUT}/cme.csv.gz')
    cm = {c: cme[f'cme_{c}'].values for c in 'enu'}
    real = [float(dt2mjd(v[0])) for v in QUAKES.values()]
    end_real = float(dt2mjd('2026-11-10')); start_real = float(dt2mjd('2010-06-01'))
    span_pre = real[0] - start_real; span_post = end_real - real[0]
    shifts = np.arange(5.0, 10.01, 0.25)
    rows = []
    cached = os.path.exists(f'{OUT}/q3_velocity_changes.csv') and '--refit' not in sys.argv
    for j, s in (M.iloc[:0] if cached else M).iterrows():
        df = load_tenv3(s.sta)
        df = df[df.mjd >= DAY0]
        sig = df[['se', 'sn', 'su']].values
        df = df[(sig < 4 * np.nanmedian(sig, axis=0)).all(axis=1)].copy()
        idx = (df.mjd.values - cme.mjd.values[0]).astype(int)
        for c in 'enu':
            df[c] = df[c].values * 1e3 - np.nan_to_num(cm[c][idx])
        for tau in (None, 10.0, 30.0, 100.0):
            r = q3_fit_station(df, s.sta, real, start_real, end_real, tau)
            if r is not None:
                rows.append(dict(sta=s.sta, kind='real', shift=0.0, tau=tau or 0, **{f'{c}_{k}': r[c][k] for c in 'enu' for k in r[c]}))
            for sh in shifts:
                q = [x - sh * 365.25 for x in real]
                r = q3_fit_station(df, s.sta, q, max(q[0] - span_pre, DAY0), q[0] + span_post, tau)
                if r is not None:
                    rows.append(dict(sta=s.sta, kind='fake', shift=sh, tau=tau or 0, **{f'{c}_{k}': r[c][k] for c in 'enu' for k in r[c]}))
        print(s.sta, flush=True)
    if cached:
        R = pd.read_csv(f'{OUT}/q3_velocity_changes.csv')
    else:
        R = pd.DataFrame(rows)
        R.to_csv(f'{OUT}/q3_velocity_changes.csv', index=False)
    g = green(M)
    res = {}
    intervals = {'d1': 'Petrolia->Ferndale minus pre-2021', 'd2': 'Ferndale->M7 minus pre-2021', 'd3': 'post-M7 minus pre-2021'}
    sta_rows = []
    for tau in (0, 10.0, 30.0, 100.0):
        Rt = R[R.tau == tau]
        for iv in intervals:
            # per-station null sd from the fake break dates
            piv = {c: Rt.pivot_table(index='shift', columns='sta', values=f'{c}_{iv}') for c in 'enu'}
            real_v = np.full((len(M), 3), np.nan); null_sd = np.full((len(M), 3), np.nan)
            for k, c in enumerate('enu'):
                rr = Rt[Rt.kind == 'real'].set_index('sta')[f'{c}_{iv}']
                fk = piv[c].drop(index=0.0, errors='ignore')
                for jj, s in enumerate(M.sta):
                    if s in rr.index:
                        real_v[jj, k] = rr[s]
                    if s in fk.columns and fk[s].notna().sum() >= 8:
                        null_sd[jj, k] = 1.4826 * np.nanmedian(np.abs(fk[s] - np.nanmedian(fk[s])))
            null_sd = np.maximum(null_sd, 0.2)
            if tau == 30.0:
                for jj, s in enumerate(M.sta):
                    sta_rows.append(dict(sta=s, lat=M.lat[jj], lon=M.lon[jj], dist=M.dist[jj], interval=iv,
                                         **{f'dv_{c}': real_v[jj, k] for k, c in enumerate('enu')},
                                         **{f'nullsd_{c}': null_sd[jj, k] for k, c in enumerate('enu')},
                                         **{f'g_{c}': g[jj, k] for k, c in enumerate('enu')}))
            fk_all = {}
            for nk in ('none', 'trans', 'trans+grad'):
                cols = [g] + nuisance_cols(M, nk)
                x, sd, chi2, dof = invert_epoch(real_v, null_sd, cols, robust=True)
                xf = []
                for sh in shifts:
                    fv = np.full((len(M), 3), np.nan)
                    for k, c in enumerate('enu'):
                        row = piv[c].loc[sh] if sh in piv[c].index else None
                        if row is not None:
                            fv[:, k] = [row.get(s, np.nan) for s in M.sta]
                    xf.append(invert_epoch(fv, null_sd, cols, robust=True)[0][0])
                    if nk == 'none':
                        fk_all.setdefault(sh, fv)
                xf = np.array(xf)
                nsd = 1.4826 * np.nanmedian(np.abs(xf - np.nanmedian(xf)))
                nsd = max(nsd, np.nanstd(xf) * 0.0) if np.isfinite(nsd) else np.nan
                area = LFE_PATCH['L'] * LFE_PATCH['W'] * 1e6
                key = f'tau{int(tau)}_{iv}_{nk}'
                res[key] = dict(interval=intervals[iv], Pdot_MP_per_yr=float(x[0]), formal_sd=float(sd[0]),
                                null_sd_MP_per_yr=float(nsd), null_sd_std=float(np.nanstd(xf)), z=float(x[0] / nsd),
                                slip_rate_change_mm_yr=float(x[0] * MP / area * 1e3),
                                bound95_slip_rate_mm_yr=float((abs(x[0]) + 1.96 * nsd) * MP / area * 1e3),
                                bound95_Mw_per_yr=float(mw_from_potency((abs(x[0]) + 1.96 * nsd) * MP)),
                                chi2=chi2, dof=dof, n_fake=int(np.isfinite(xf).sum()))
            # cluster statistic: median velocity change of the stations within 30 km of the LFEs vs
            # the network median of stations beyond 50 km (local minus regional), against the fake dates
            cl = (M.dist.values <= 30); fa = (M.dist.values >= 50)
            def clus(v):
                return np.nanmedian(v[cl], 0), np.nanmedian(v[cl], 0) - np.nanmedian(v[fa], 0)
            ob = clus(real_v); nf = [clus(v) for v in fk_all.values()]
            for lab, i in (('cluster', 0), ('cluster_minus_far', 1)):
                arr = np.array([z[i] for z in nf])
                res[f'tau{int(tau)}_{iv}_{lab}'] = dict(
                    interval=intervals[iv], stations=list(M.sta[cl]),
                    obs_mm_yr={c: float(ob[i][k]) for k, c in enumerate('enu')},
                    null_sd={c: float(np.nanstd(arr[:, k])) for k, c in enumerate('enu')},
                    z={c: float(ob[i][k] / np.nanstd(arr[:, k])) for k, c in enumerate('enu')},
                    p_emp={c: float(np.mean(np.abs(arr[:, k]) >= abs(ob[i][k]))) for k, c in enumerate('enu')})
    pd.DataFrame(sta_rows).to_csv(f'{OUT}/q3_station_dv_tau30.csv', index=False)
    # rolling 2-yr velocity anomaly field -> LFE-patch slip-rate anomaly (translation nuisance, robust)
    _, M2, Yc = near_cube('clean', maxdist=110.0)
    starts = pd.date_range('2010-01-01', '2024-11-15', freq='60D')
    VS = np.full((len(starts), len(M2), 3), np.nan)
    for ii, t in enumerate(starts):
        m = dt2mjd(t)
        sel = (days >= m) & (days < m + 730)
        tt = (days[sel] - m) / 365.25
        for jj in range(len(M2)):
            for k in range(3):
                y = Yc[sel, jj, k]; ok = np.isfinite(y)
                if ok.sum() < 300:
                    continue
                VS[ii, jj, k] = np.polyfit(tt[ok], y[ok], 1)[0]
    mids = starts + pd.Timedelta('365D')
    pre = mids < pd.Timestamp('2020-12-20')            # windows ending before Petrolia
    ref = np.nanmedian(VS[pre], 0)
    An = VS - ref
    sdv = 1.4826 * np.nanmedian(np.abs(An[pre] - np.nanmedian(An[pre], 0)), 0)
    sdv = np.where(np.isfinite(sdv), np.maximum(sdv, 0.2), np.nan)
    gg = green(M2)
    cols = [gg] + nuisance_cols(M2, 'trans')
    rr = []
    for ii in range(len(starts)):
        x, sd, chi2, dof = invert_epoch(An[ii], sdv, cols, robust=True)
        rr.append(dict(start=starts[ii], mid=mids[ii], Pdot_MP_per_yr=x[0], formal_sd=sd[0],
                       nsta=int(np.isfinite(An[ii, :, 0]).sum())))
    RR = pd.DataFrame(rr)
    area = LFE_PATCH['L'] * LFE_PATCH['W'] * 1e6
    RR['slip_rate_anom_mm_yr'] = RR.Pdot_MP_per_yr * MP / area * 1e3
    dl = lfe_daily()['2009-01-01':'2026-06-30']
    RR['lfe_rate_2yr'] = [float(dl[s:s + pd.Timedelta('729D')].mean()) for s in RR.start]
    RR.to_csv(f'{OUT}/q3_rolling_patch_rate.csv', index=False)
    ok = RR.slip_rate_anom_mm_yr.notna() & RR.lfe_rate_2yr.notna() & (RR.start + pd.Timedelta('730D') <= pd.Timestamp('2026-06-30'))
    a_ = RR.slip_rate_anom_mm_yr[ok].values; b_ = RR.lfe_rate_2yr[ok].values
    r_obs = np.corrcoef(a_, b_)[0, 1]
    rng = np.random.default_rng(5)
    nul = np.array([np.corrcoef(a_, np.roll(b_, rng.integers(12, len(b_) - 12)))[0, 1] for _ in range(3000)])
    pre_sd = float(np.nanstd(RR.slip_rate_anom_mm_yr[RR.mid < pd.Timestamp('2020-12-20')]))
    res['rolling'] = dict(note='2-yr windows every 60 d; anomaly vs pre-2021 median velocity; LFE patch + translation; robust',
                          corr_with_lfe_rate=float(r_obs), p_circular=float(np.mean(np.abs(nul) >= abs(r_obs))),
                          pre2021_sd_mm_yr=pre_sd,
                          post2021_mean_mm_yr=float(RR.slip_rate_anom_mm_yr[RR.start >= pd.Timestamp('2022-01-01')].mean()),
                          post2023_mean_mm_yr=float(RR.slip_rate_anom_mm_yr[RR.start >= pd.Timestamp('2023-02-20')].mean()))
    print(json.dumps({k: v for k, v in res.items() if 'tau30' in k or 'tau0' in k}, indent=1))
    json.dump(res, open(f'{OUT}/q3_summary.json', 'w'), indent=1)


def stage_figs():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dr = okada()
    M = pd.read_csv(f'{OUT}/station_meta.csv')
    near = M[M.set == 'near']
    fams_lon, fams_lat = [-123.71, -123.53], [40.03, 40.09]
    # FFM trace
    srcs, hdr, _ = dr.ffm_rects('m7.fsp')
    fe = np.array([r.e for r in srcs]); fn = np.array([r.n for r in srcs])
    flon = fe / (KM * np.cos(np.radians(dr.LAT0))) + dr.LON0; flat = fn / KM + dr.LAT0

    def lfe_box(ax):
        ax.add_patch(plt.Rectangle((fams_lon[0], fams_lat[0]), fams_lon[1] - fams_lon[0], fams_lat[1] - fams_lat[0],
                                   fc='none', ec='C3', lw=2, zorder=5))

    def base_map(ax, title):
        ax.scatter(near.lon, near.lat, s=12, c='0.4', zorder=3)
        lfe_box(ax)
        ax.plot(flon, flat, '.', ms=1, c='C1', alpha=0.5)
        ax.set_xlim(-125.2, -122.4); ax.set_ylim(39.0, 41.05)
        ax.set_aspect(1 / np.cos(np.radians(40)))
        ax.set_title(title, fontsize=9)

    # ---------------- map + Green's functions ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    base_map(ax[0], 'NGL stations (grey = near, <=110 km; blue = far CME set)\nred box = LFE families, orange = M7 FFM')
    fr = M[M.set == 'far']
    ax[0].scatter(fr.lon, fr.lat, s=8, c='C0')
    ax[0].set_xlim(-126, -119); ax[0].set_ylim(37.5, 43.5)
    for _, r in near.iterrows():
        if r.dist < 30:
            ax[0].text(r.lon, r.lat, r.sta, fontsize=6)
    Mg = near[near.dist <= 110].reset_index(drop=True)
    g = green(Mg) * 10
    base_map(ax[1], 'Surface displacement for 1e7 m^3 potency (Mw 5.6) RL slip on LFE patch\n(arrows horizontal, colour vertical, mm)')
    sc = ax[1].scatter(Mg.lon, Mg.lat, c=g[:, 2], cmap='RdBu_r', vmin=-1, vmax=1, s=40, zorder=4)
    ax[1].quiver(Mg.lon, Mg.lat, g[:, 0], g[:, 1], scale=5, width=0.004, zorder=6)
    ax[1].quiver([-125.0], [39.2], [1.0], [0], scale=5, width=0.004); ax[1].text(-125.0, 39.27, '1 mm', fontsize=8)
    plt.colorbar(sc, ax=ax[1], shrink=0.7)
    plt.tight_layout(); plt.savefig(f'{FIG}/sci_gnss_map.png', dpi=130); plt.close()

    # ---------------- Q1 ----------------
    S1 = json.load(open(f'{OUT}/q1_summary.json'))
    P = pd.read_csv(f'{OUT}/q1_peaks.csv', parse_dates=['date'])
    W = pd.read_csv(f'{OUT}/q1_window_potency_ramp15.csv', parse_dates=['date'])
    ser = pd.read_csv(f'{OUT}/q1_daily_patch_projection.csv.gz')
    CB = pd.read_csv(f'{OUT}/q1_continuous_bins.csv')
    d = lfe_daily()['2009-01-01':]
    rel = d.rolling(15, center=True).mean() / d.rolling(365, center=True, min_periods=180).median()
    fig = plt.figure(figsize=(14, 10))
    a = fig.add_subplot(3, 1, 1)
    a.plot(rel.index, rel.values, lw=0.6, c='0.3')
    a.plot(P.date, P.rel15, 'v', c='C3', label='selected peaks (top 20, >=120 d apart)')
    for k, v in QUAKES.items():
        a.axvline(pd.Timestamp(v[0]), c='C1', ls='--', lw=0.8)
    a.set_ylabel('15-d LFE rate / 1-yr median'); a.legend(fontsize=8)
    a.set_title('Q1: largest multi-week LFE-rate peaks', fontsize=10)
    # stacked daily projection around peaks
    a = fig.add_subplot(3, 3, 4)
    p = ser.p_MP.values; mj = ser.mjd.values
    def seg(m):
        i = int(m - mj[0]); s = p[i - 90:i + 91].copy()
        tt = np.arange(-90, 91)
        ok = np.isfinite(s) & (tt < -8)
        if ok.sum() < 40:
            return None
        x = np.polyfit(tt[ok], s[ok], 1)
        return s - np.polyval(x, tt)
    segs = [seg(m) for m in P.mjd]; segs = np.array([s for s in segs if s is not None])
    cand = W[W.kind == 'cand'].mjd.values
    rng = np.random.default_rng(3)
    allc = [seg(m) for m in cand]; allc = np.array([s for s in allc if s is not None])
    pl = np.array([np.nanmean(allc[rng.choice(len(allc), len(segs), replace=False)], 0) for _ in range(1000)])
    tt = np.arange(-90, 91)
    def smooth(x, k=7):
        return pd.Series(x).rolling(k, center=True, min_periods=3).mean().values
    lo, hi = np.nanquantile([smooth(x) for x in pl], [0.025, 0.975], axis=0)
    a.fill_between(tt, lo, hi, color='0.8', label='placebo 95%')
    a.plot(tt, smooth(np.nanmean(segs, 0)), c='C3', label=f'stack of {len(segs)} peaks')
    a.axvspan(-7.5, 7.5, color='C3', alpha=0.1)
    a.set_xlabel('days from peak'); a.set_ylabel('LFE-patch potency (1e6 m^3)'); a.legend(fontsize=7)
    a.set_title('daily network projection (7-d smooth)', fontsize=9)
    a = fig.add_subplot(3, 3, 5)
    cd = W[W.kind == 'cand'].P.dropna().values
    null = np.array([cd[rng.choice(len(cd), len(P), replace=False)].mean() for _ in range(5000)])
    a.hist(null, 60, color='0.7', density=True)
    a.axvline(S1['ramp15']['P']['obs_mean_MP'], c='C3', lw=2)
    a.set_xlabel('mean ramp potency over 20 windows (1e6 m^3)')
    a.set_title(f"peaks vs random dates: p={S1['ramp15']['P']['p_two_sided']:.2f}; 95% bound Mw {S1['ramp15']['P']['bound95_Mw']:.2f}", fontsize=9)
    a = fig.add_subplot(3, 3, 6)
    pk = W[W.kind == 'peak'].merge(P[['mjd', 'excess_lfe']], on='mjd')
    a.errorbar(pk.excess_lfe, pk.P, yerr=pk.sP, fmt='o', c='C3', ms=4)
    a.axhline(0, c='k', lw=0.5)
    a.set_xlabel('excess LFEs in the 15-d peak'); a.set_ylabel('window potency (1e6 m^3)')
    a.set_title('per-peak estimates (formal 1-sigma)', fontsize=9)
    a = fig.add_subplot(3, 1, 3)
    tb = mjd2dt(CB.mjd.values)
    k = S1['continuous']['all']['k_m3_per_LFE'] / MP
    a.plot(tb, CB.p_res_MP, c='0.4', lw=0.7, label='LFE-patch projection (10-d bins, model residual)')
    a2 = a.twinx()
    a2.plot(tb, CB.C_res, c='C3', lw=1, label='cumulative excess LFE count (same basis removed)')
    a.set_ylabel('potency (1e6 m^3)'); a2.set_ylabel('LFEs', color='C3')
    c = S1['continuous']['all']
    a.set_title(f"continuous regression: k = {c['k_m3_per_LFE']:.0f} +/- {c['null_sd_m3_per_LFE']:.0f} m^3 per LFE "
                f"(p={c['p_two_sided']:.2f}); 95% bound {c['bound95_m3_per_LFE']:.0f} m^3 = {c['bound95_ratio_to_LFE_potency']:.0f}x LFE potency", fontsize=9)
    a.legend(loc='upper left', fontsize=7); a2.legend(loc='upper right', fontsize=7)
    plt.tight_layout(); plt.savefig(f'{FIG}/sci_gnss_q1_rate_peaks.png', dpi=130); plt.close()

    # ---------------- Q2 ----------------
    R = pd.read_csv(f'{OUT}/q2_lfe_patch_timeseries.csv')
    S2 = json.load(open(f'{OUT}/q2_summary.json'))
    E = pd.read_csv(f'{OUT}/q2_station_epochs.csv')
    L = pd.read_csv(f'{OUT}/q2_lfe_cum_excess_after_m7.csv')
    Gd = pd.read_csv(f'{OUT}/q2_patch_position_placebo.csv')
    fig = plt.figure(figsize=(15, 10))
    a = fig.add_subplot(2, 2, 1)
    for i, var in enumerate(Q2_VARIANTS):
        r = R[R.variant == var]
        a.errorbar(r['mid'] * (1 + 0.03 * i), r.P_lfe_MP, yerr=r.null_sd, fmt='o-', ms=3, lw=0.8, capsize=0, label=var)
    a.axhline(0, c='k', lw=0.5)
    a.axvspan(0.5, 18, color='C3', alpha=0.08); a.axvspan(60, 90, color='C2', alpha=0.1)
    a.set_xscale('log'); a.set_xlabel('days after M7'); a.set_ylabel('LFE-patch potency (1e6 m^3)')
    a.set_title('Q2: LFE-patch slip after the M7 (error bars = placebo-date sd)\nred = LFE silence (t50 ~18 d), green = 60-90 d moment repayment', fontsize=9)
    a.legend(fontsize=7)
    a2 = a.twinx()
    a2.plot(L.days.clip(lower=0.5), L.cum_excess, c='C3', lw=1)
    a2.set_ylabel('cumulative LFE excess vs pre-M7 rate', color='C3')
    a = fig.add_subplot(2, 2, 2)
    lab = list(S2['raw_field_chi2'].keys())
    mid = np.array(S2['epoch_mid_days'])
    a.plot(mid, [S2['raw_field_chi2'][k]['obs'] for k in lab], 'o-', c='k', label='raw post-M7 field chi2')
    a.plot(mid, [S2['raw_field_chi2'][k]['placebo_q95'] for k in lab], '--', c='0.5', label='placebo-date 95%')
    a.plot(mid, [S2['dchi2_lfe_patch'][k]['obs'] for k in lab], 's-', c='C3', label='chi2 drop from adding LFE patch')
    a.plot(mid, [S2['dchi2_lfe_patch'][k]['placebo_q95'] for k in lab], ':', c='C3', label='placebo 95% (LFE patch)')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('days after M7'); a.legend(fontsize=7)
    a.set_title('postseismic signal is strong; LFE patch adds nothing beyond placebo', fontsize=9)
    a = fig.add_subplot(2, 2, 3)
    e = E[E.epoch_lo == 300]
    base_map(a, 'observed postseismic displacement, mean of days 300-365 (mm)')
    sc = a.scatter(e.lon, e.lat, c=e.u_u, cmap='RdBu_r', vmin=-8, vmax=8, s=30, zorder=4)
    a.quiver(e.lon, e.lat, e.u_e, e.u_n, scale=80, width=0.004, zorder=6)
    a.quiver([-125.0], [39.2], [10.0], [0], scale=80, width=0.004); a.text(-125.0, 39.27, '10 mm', fontsize=8)
    plt.colorbar(sc, ax=a, shrink=0.7, label='up (mm)')
    a = fig.add_subplot(2, 2, 4)
    col = 'z_60_75'
    Z = Gd.pivot(index='n0', columns='e0', values=col)
    im = a.pcolormesh(Z.columns, Z.index, Z.values, cmap='RdBu_r', vmin=-4, vmax=4, shading='nearest')
    a.plot(0, 0, 's', mfc='none', mec='k', ms=12)
    e_, n_ = en_lfe(near.lat, near.lon); a.plot(e_, n_, '.', c='0.3', ms=3)
    a.set_xlim(-65, 65); a.set_ylim(-65, 65); a.set_aspect(1)
    plt.colorbar(im, ax=a, label='formal z of patch potency (60-75 d)')
    a.set_xlabel('patch centre east of LFE centroid (km)'); a.set_ylabel('north (km)')
    a.set_title('placebo patch positions at 27 km (model lfe+m7+trans); square = LFEs', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{FIG}/sci_gnss_q2_postseismic.png', dpi=130); plt.close()

    # ---------------- Q3 ----------------
    S3 = json.load(open(f'{OUT}/q3_summary.json'))
    V = pd.read_csv(f'{OUT}/q3_station_dv_tau30.csv')
    RR = pd.read_csv(f'{OUT}/q3_rolling_patch_rate.csv', parse_dates=['mid', 'start'])
    fig = plt.figure(figsize=(15, 10))
    for i, iv in enumerate(['d2', 'd3']):
        a = fig.add_subplot(2, 2, i + 1)
        v = V[V.interval == iv]
        base_map(a, {'d2': 'velocity change Ferndale->M7 minus 2010-2021 (tau 30 d), mm/yr',
                     'd3': 'velocity change post-M7 minus 2010-2021 (tau 30 d), mm/yr'}[iv])
        zz = np.hypot(v.dv_e / v.nullsd_e, v.dv_n / v.nullsd_n)
        sc = a.scatter(v.lon, v.lat, c=v.dv_u, cmap='RdBu_r', vmin=-4, vmax=4, s=30, zorder=4)
        a.quiver(v.lon, v.lat, v.dv_e, v.dv_n, scale=30, width=0.004, zorder=6)
        a.quiver([-125.0], [39.2], [2.0], [0], scale=30, width=0.004); a.text(-125.0, 39.27, '2 mm/yr', fontsize=8)
        plt.colorbar(sc, ax=a, shrink=0.7, label='up (mm/yr)')
    a = fig.add_subplot(2, 2, 3)
    keys = [k for k in S3 if k.endswith('trans')]
    ys = np.arange(len(keys))
    a.errorbar([S3[k]['slip_rate_change_mm_yr'] for k in keys], ys,
               xerr=[1.96 * S3[k]['null_sd_MP_per_yr'] * MP / (LFE_PATCH['L'] * LFE_PATCH['W'] * 1e6) * 1e3 for k in keys],
               fmt='o', c='C0')
    a.set_yticks(ys); a.set_yticklabels([k.replace('_trans', '') for k in keys], fontsize=7)
    a.axvline(0, c='k', lw=0.5)
    a.axvspan(-25, 25, color='C2', alpha=0.08)
    a.set_xlabel('LFE-patch slip-rate change (mm/yr, RL), 95% from placebo break dates')
    a.set_title('Q3: LFE-patch loading-rate change (with translation nuisance); green = +/- plate rate 2.5 cm/yr', fontsize=9)
    a = fig.add_subplot(2, 2, 4)
    sdp = S3['rolling']['pre2021_sd_mm_yr']
    a.fill_between(RR['mid'], -2 * sdp, 2 * sdp, color='0.85', label='+/-2 sd of pre-2021 windows')
    a.plot(RR['mid'], RR.slip_rate_anom_mm_yr, 'o-', ms=3, label='LFE-patch slip-rate anomaly (2-yr windows)')
    a.set_ylabel('slip-rate anomaly on LFE patch (mm/yr, RL)')
    a2 = a.twinx(); a2.plot(RR['mid'], RR.lfe_rate_2yr, c='C3', lw=1); a2.set_ylabel('LFE/day (same 2-yr window)', color='C3')
    for k, v in QUAKES.items():
        a.axvline(pd.Timestamp(v[0]), c='C1', ls='--', lw=0.8)
    a.legend(fontsize=7, loc='upper left')
    a.set_title(f"rolling 2-yr velocity anomaly projected on LFE patch (incl. postseismic); corr with LFE rate "
                f"r={S3['rolling']['corr_with_lfe_rate']:.2f}, p={S3['rolling']['p_circular']:.2f}", fontsize=9)
    plt.tight_layout(); plt.savefig(f'{FIG}/sci_gnss_q3_velocity.png', dpi=130); plt.close()
    print('figures written')


def stage_summary():
    """Collect headline numbers from q1-q3 into data/sci/gnss/summary.json."""
    S1 = json.load(open(f'{OUT}/q1_summary.json')); S2 = json.load(open(f'{OUT}/q2_summary.json'))
    S3 = json.load(open(f'{OUT}/q3_summary.json'))
    M = pd.read_csv(f'{OUT}/station_meta.csv')
    R2 = pd.read_csv(f'{OUT}/q2_lfe_patch_timeseries.csv')
    r2 = R2[R2.variant == 'lfe+m7+trans']
    E = pd.read_csv(f'{OUT}/q2_station_epochs.csv'); UF = np.load(f'{OUT}/q2_placebo_epochs.npy')
    stas = E.sta.unique(); dist = M.set_index('sta').loc[stas, 'dist'].values; cl = dist <= 30
    clus = {}
    for b, e0 in enumerate(sorted(E.epoch_lo.unique())):
        x = E[E.epoch_lo == e0].set_index('sta').loc[stas]
        o = np.array([np.nanmean(x[c].values[cl]) for c in ('u_e', 'u_n', 'u_u')])
        sd = np.nanstd(np.nanmean(UF[:, b][:, cl], axis=1), 0)
        clus[f'{e0}d'] = dict(mean_enu_mm=o.round(2).tolist(), placebo_sd=sd.round(2).tolist(), z=(o / sd).round(1).tolist())
    out = dict(
        data=dict(source='NGL IGS20 daily tenv3 + NGL steps.txt (public)', n_near=int((M.set == 'near').sum()),
                  n_far_cme=int((M.set == 'far').sum()), near_wrms_mm_before_after_cme={
                      c: [float(M[M.set == 'near'][f'wrms_{c}'].median()), float(M[M.set == 'near'][f'wrms_cme_{c}'].median())] for c in 'enu'}),
        q1_peaks=dict(n=S1['notes']['n_peaks'], ramp15=S1['ramp15']['P'], ramp30=S1['ramp30']['P'],
                      big_patch_ramp15=S1['ramp15']['P_big']),
        q1_continuous=S1['continuous'],
        q2_lfe_patch_m7_trans={f"{int(r.epoch_lo)}-{int(r.epoch_hi)}d": dict(P_MP=round(r.P_lfe_MP, 2), null_sd=round(r.null_sd, 2),
                                                                           z=round(r.z, 2), p=round(r.p_emp, 3), bound95_Mw=round(r.bound95_Mw, 2))
                               for r in r2.itertuples()},
        q2_dchi2=S2['dchi2_lfe_patch'], q2_raw_field=S2['raw_field_chi2'], q2_patch_position=S2['patch_position_rank'],
        q2_near_lfe_cluster_displacement=clus,
        q3=S3)
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    print('summary written')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for st in ['download', 'prep', 'q1', 'q2', 'q3', 'figs', 'summary']:
        if stage in (st, 'all'):
            globals()[f'stage_{st}']()
