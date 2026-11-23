#!/usr/bin/env python
"""PBO borehole strainmeter stack on LFE network episodes (Mendocino triple junction).

Stages (python scripts/sci_strain_stack.py <stage>):
  download  - 10-min RS1-4 gauge strain (loc T0) + 30-min RDO barometric pressure (loc TS)
              for B045/B933/B934/B935 from the EarthScope FDSN service (public, no token),
              one request per station-year, cached in data/sci/strain/raw/.
              Also a regional NCEDC earthquake catalogue (M>=3, 200 km) for contamination flags.
  analyse   - per-window tide/pressure/trend correction, step estimates, stacks, null, bounds.

Only B045, B933, B934, B935 have Gladwin tensor strainmeters (B046/B047/B049/B932 do not:
FDSN metadata list no LS*/RS* channels for them).
"""
import sys, os, json, glob, time
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/strain'
RAW = f'{OUT}/raw'
FIG = f'{ROOT}/figures'
os.makedirs(RAW, exist_ok=True)

STATIONS = ['B045', 'B933', 'B934', 'B935']
LFE_LAT, LFE_LON, LFE_Z = 40.06, -123.625, 28.0e3


def stage_download():
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime as U
    c = Client('EARTHSCOPE')
    log = open(f'{OUT}/download.log', 'a')
    for sta in STATIONS:
        for yr in range(2008, 2027):
            fn = f'{RAW}/{sta}_{yr}.mseed'
            if os.path.exists(fn) or os.path.exists(fn + '.none'):
                continue
            t1, t2 = U(f'{yr}-01-01'), U(f'{yr + 1}-01-01')
            ok = False
            for attempt in range(3):
                try:
                    st = c.get_waveforms('PB', sta, 'T0,TS', 'RS1,RS2,RS3,RS4,RDO', t1, t2)
                    st.write(fn, format='MSEED')
                    msg = f'{sta} {yr} ok {len(st)} traces ' + ' '.join(
                        f'{tr.stats.channel}:{tr.stats.npts}' for tr in st[:12])
                    ok = True
                    break
                except Exception as e:
                    msg = f'{sta} {yr} {type(e).__name__} {str(e)[:80]}'
                    if 'No data' in str(e) or 'NoData' in type(e).__name__:
                        open(fn + '.none', 'w').close()
                        break
                    time.sleep(10)
            print(msg, flush=True)
            log.write(msg + '\n'); log.flush()
            time.sleep(1)
    # regional catalogue for coseismic-step flags
    fn = f'{OUT}/raw/ncedc_events_M3_200km.csv'
    if not os.path.exists(fn):
        cn = Client('NCEDC')
        rows = []
        for y in range(2008, 2027):
            try:
                cat = cn.get_events(starttime=U(f'{y}-01-01'), endtime=U(f'{y + 1}-01-01'),
                                    latitude=LFE_LAT, longitude=LFE_LON, maxradius=1.8,
                                    minmagnitude=3.0)
            except Exception as e:
                print('events', y, type(e).__name__, flush=True)
                continue
            for ev in cat:
                o = ev.preferred_origin() or ev.origins[0]
                m = ev.preferred_magnitude() or ev.magnitudes[0]
                rows.append(dict(time=str(o.time), lat=o.latitude, lon=o.longitude,
                                 depth_km=(o.depth or 0) / 1e3, mag=m.mag))
            time.sleep(1)
        pd.DataFrame(rows).to_csv(fn, index=False)
        print('events', len(rows))


# ======================================================================================
# Analysis
# ======================================================================================
DT = 600.0                                   # 10-min RS samples
GRID0 = pd.Timestamp('2008-09-01').value / 1e9
NGRID = int((pd.Timestamp('2026-12-01').value / 1e9 - GRID0) / DT)
G_OVER_D = 1e-4 / 0.087                      # Gladwin gap / diameter (UNAVCO linearisation)
TIDES = {'M2': 1.9322736, 'S2': 2.0, 'N2': 1.8959820, 'K1': 1.0027379, 'O1': 0.9295357}  # cpd
HALF_FIT, HALF_CORE = 15.0, 2.0              # days
MU = 30e9
MW_MED = 1.5
M0_MED = 10 ** (1.5 * MW_MED + 9.1)
STRIKE, DIP, RAKE = 293.0, 40.0, 180.0       # adopted NE-dipping right-lateral plane
COMPS = ['areal', 'gam1', 'gam2']            # areal = eEE+eNN, gam1 = eEE-eNN, gam2 = 2eEN


def load_station(sta):
    """Return dict channel -> float array on the global 10-min grid (gauges in nanostrain,
    extension positive by the UNAVCO convention; pressure in hPa)."""
    from obspy import read, Stream
    st = Stream()
    for fn in sorted(glob.glob(f'{RAW}/{sta}_*.mseed')):
        st += read(fn)
    out = {}
    for ch in ['RS1', 'RS2', 'RS3', 'RS4', 'RDO']:
        a = np.full(NGRID, np.nan)
        for tr in st.select(channel=ch):
            t = tr.stats.starttime.timestamp + np.arange(tr.stats.npts) * tr.stats.delta
            d = tr.data.astype(float)
            if ch == 'RDO':
                v = d * 1e-3                                     # counts -> hPa (1e-4 kPa/count)
                ok = (v > 800) & (v < 1100)
            else:
                R = d / 1e8
                ok = (R > 0.02) & (R < 0.98)
                v = R / (1 - R) * G_OVER_D * 1e9                 # linearised gauge strain, nanostrain
            if ch == 'RDO':                                      # 30-min -> place at grid, interp later
                i = np.round((t - GRID0) / DT).astype(int)
            else:
                i = np.round((t - GRID0) / DT).astype(int)
            k = ok & (i >= 0) & (i < NGRID)
            a[i[k]] = v[k]
        if ch == 'RDO':                                          # fill 30-min to 10-min (gaps <= 1 h)
            s = pd.Series(a).interpolate(limit=6, limit_area='inside')
            a = s.values
        out[ch] = a
    # gauge azimuths from FDSN metadata (deg E of N), fetched once and cached
    return out


GAUGE_AZ = {'B045': [227.7, 167.7, 107.7, 77.7], 'B933': [271.0, 211.0, 151.0, 121.0],
            'B934': [319.0, 259.0, 199.0, 169.0], 'B935': [263.1, 203.1, 143.1, 113.1]}
STA_LL = {'B045': (40.436001, -123.996498), 'B933': (40.060001, -123.969002),
          'B934': (39.844501, -123.691399), 'B935': (40.478699, -123.573196)}


def gauge_matrix(sta):
    """Unit-coupling ('uncalibrated, isotropic') relation e_i = 1/2 eA - 1/2 g1 cos2th + 1/2 g2 sin2th
    (th = gauge azimuth E of N; extension along the gauge).  Returns (4,3) matrix."""
    th = np.radians(GAUGE_AZ[sta])
    return np.column_stack([0.5 * np.ones(4), -0.5 * np.cos(2 * th), 0.5 * np.sin(2 * th)])


def destep(x, core_mask, thr_abs=3.0, k=12.0):
    """Remove isolated offsets (gauge resets / maintenance) outside the core by zeroing
    anomalous first differences.  Returns (x_fixed, n_steps_outside, has_step_in_core)."""
    d = np.diff(x)
    ok = np.isfinite(d)
    if ok.sum() < 50:
        return x, 0, True
    mad = np.median(np.abs(d[ok] - np.median(d[ok]))) * 1.4826 + 1e-6
    bad = ok & (np.abs(d) > max(thr_abs, k * mad))
    core_d = core_mask[1:] | core_mask[:-1]
    in_core = bool((bad & core_d).any())
    fix = bad & ~core_d
    if fix.any():
        off = np.where(fix, d, 0.0)
        x = x - np.concatenate([[0.0], np.cumsum(off)])
    return x, int(fix.sum()), in_core


def window_design(tday, p):
    """Columns: Legendre deg-3 trend, 5 tidal constituents (cos,sin), pressure."""
    u = tday / HALF_FIT
    cols = [np.ones_like(u), u, 0.5 * (3 * u ** 2 - 1), 0.5 * (5 * u ** 3 - 3 * u)]
    return np.column_stack(cols), np.column_stack(
        [f(2 * np.pi * fr * tday) for fr in TIDES.values() for f in (np.cos, np.sin)]), p[:, None]


def process_window(G, t0, dur_d, want_series=False):
    """G: dict channel->grid array for one station.  Returns dict with per-gauge step estimates
    (nanostrain), pressure coefficient, residual RMS and optional core residual series."""
    i0 = int(round((t0 - GRID0) / DT))
    nh = int(HALF_FIT * 86400 / DT)
    a, b = i0 - nh, i0 + nh + 1
    if a < 0 or b > NGRID:
        return None
    tday = (np.arange(a, b) - i0) * DT / 86400.0 + (i0 * DT + GRID0 - t0) / 86400.0
    nc = int(HALF_CORE * 86400 / DT)
    core = np.abs(np.arange(a, b) - i0) <= nc
    p = G['RDO'][a:b].copy()
    pok = np.isfinite(p)
    if pok[core].mean() < 0.9 or pok.mean() < 0.7:
        use_p = False
        p = np.zeros_like(tday)
    else:
        use_p = True
        p = p - np.nanmean(p)
        p = np.where(pok, p, 0.0)
    Ltr, Htd, Pp = window_design(tday, p)
    tc = tday[core]
    # core model: a + b t + c ramp(0..dur) ; second model: 1-day ramp
    ramp = np.clip(tc / max(dur_d, DT / 86400), 0, 1)
    ramp1 = np.clip(tc / 1.0, 0, 1)
    res = dict(steps=np.full(4, np.nan), steps1d=np.full(4, np.nan), rms=np.full(4, np.nan),
               pcoef=np.full(4, np.nan), M2amp=np.full(4, np.nan), use_p=use_p,
               series=np.full((4, core.sum()), np.nan) if want_series else None)
    for g in range(4):
        x = G[f'RS{g + 1}'][a:b].copy()
        ok = np.isfinite(x) & (pok if use_p else np.ones_like(pok))
        if ok[core].mean() < 0.9 or ok[~core].mean() < 0.6:
            continue
        x, nfix, step_core = destep(x, core)
        if step_core:
            continue
        ok = ok & np.isfinite(x)
        X = np.column_stack([Ltr, Htd] + ([Pp] if use_p else []))
        fit = ok & ~core
        Xf, xf = X[fit], x[fit]
        w = np.ones(fit.sum(), bool)
        for _ in range(2):
            beta, *_ = np.linalg.lstsq(Xf[w], xf[w], rcond=None)
            r = xf - Xf @ beta
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            w = np.abs(r) < 5 * s + 1e-9
        nt = Ltr.shape[1]
        tide_p = X[:, nt:] @ beta[nt:]
        y = (x - tide_p)[core]
        okc = ok[core]
        D = np.column_stack([np.ones(core.sum()), tc, ramp])
        bc, *_ = np.linalg.lstsq(D[okc], y[okc], rcond=None)
        D1 = np.column_stack([np.ones(core.sum()), tc, ramp1])
        b1, *_ = np.linalg.lstsq(D1[okc], y[okc], rcond=None)
        rr = y[okc] - D[okc] @ bc
        res['steps'][g] = bc[2]
        res['steps1d'][g] = b1[2]
        res['rms'][g] = np.std(np.diff(rr))      # high-frequency residual noise per 10 min
        res['pcoef'][g] = beta[-1] if use_p else np.nan
        res['M2amp'][g] = np.hypot(beta[nt], beta[nt + 1])
        if want_series:
            pre = okc & (tc < 0)
            if pre.sum() > 50:
                bp = np.polyfit(tc[pre], y[pre], 1)
                ser = y - np.polyval(bp, tc)
                ser[~okc] = np.nan
                res['series'][g] = ser
    return res


def to_tensor(sta, v):
    """Gauge values (4,...) -> areal, gam1, gam2 by least squares (needs >=3 finite gauges)."""
    A = gauge_matrix(sta)
    v = np.asarray(v, float)
    ok = np.isfinite(v) if v.ndim == 1 else np.isfinite(v).any(axis=1)
    if ok.sum() < 3:
        return np.full((3,) + v.shape[1:], np.nan)
    return np.linalg.pinv(A[ok]) @ v[ok]


def greens(strike=None, dip=None, rake=None, depth_m=None, e0=0.0, n0=0.0):
    """Strain (nanostrain) at each station per unit potency (1 m^3) of a point source at the LFE
    centroid (28 km) on the strike 293 / dip 40 right-lateral plane, via the project's Okada DC3D
    port (scripts/sci_dose_response.py, validated there).  Returns {sta: [areal, gam1, gam2, g1..g4]}."""
    sys.path.insert(0, f'{ROOT}/scripts')
    from sci_dose_response import Rect, strain_fd
    Lk = 0.2                                              # km, small rectangle ~ point source
    slip = 1.0 / (Lk * 1e3) ** 2                          # 1 m^3 potency
    src = Rect(e0, n0, (depth_m or LFE_Z) / 1e3, strike if strike is not None else STRIKE,
               dip if dip is not None else DIP, rake if rake is not None else RAKE, Lk, Lk, slip)
    out = {}
    for sta, (la, lo) in STA_LL.items():
        e = (lo - LFE_LON) * 111.195 * np.cos(np.radians(LFE_LAT))
        n = (la - LFE_LAT) * 111.195
        eps = strain_fd([src], np.array([[e, n, -0.15]]))[0] * 1e9
        areal = eps[0, 0] + eps[1, 1]
        g1 = eps[0, 0] - eps[1, 1]
        g2 = 2 * eps[0, 1]
        gauges = gauge_matrix(sta) @ np.array([areal, g1, g2])
        out[sta] = dict(areal=areal, gam1=g1, gam2=g2, dist_km=float(np.hypot(e, n)),
                        gauges=gauges.tolist())
    return out


def null_candidates(t0, busy, rng, nmax=60):
    """Integer-day shifts (exact time of day) with M2 phase within +-18 deg, |shift| 10..500 d,
    excluding times within 3 h of any catalogued episode start and within 2.5 d of a selected one."""
    n = np.arange(10, 501)
    n = np.concatenate([n, -n])
    ph = (n * 24.0 / 12.4206012) % 1.0
    ph = np.minimum(ph, 1 - ph)
    n = n[ph < 0.05]
    tc = t0 + n * 86400.0
    keep = np.ones(len(tc), bool)
    for arr, tol in busy:
        j = np.searchsorted(arr, tc)
        for jj in (j - 1, j):
            jj = np.clip(jj, 0, len(arr) - 1)
            keep &= np.abs(arr[jj] - tc) > tol
    tc = tc[keep & (tc > GRID0 + 20 * 86400) & (tc < GRID0 + (NGRID - 3000) * DT)]
    rng.shuffle(tc)
    return tc[:nmax]


def episode_lfe_moment(eps_df):
    ev = pd.read_csv(f'{ROOT}/data/sci/moments/event_amplitudes.csv.gz', usecols=['fam', 'time', 'A', 'qc_bad'])
    ev = ev[~ev.qc_bad]
    t = pd.to_datetime(ev.time).astype('datetime64[ns]').astype('int64').values / 1e9
    o = np.argsort(t)
    t, A = t[o], ev.A.values[o]
    cA = np.concatenate([[0], np.cumsum(A)])
    a = np.searchsorted(t, eps_df.t0.values - 60)
    b = np.searchsorted(t, eps_df.t0.values + eps_df.dur_h.values * 3600 + 600)
    return (cA[b] - cA[a]) * M0_MED, b - a


def near(sorted_arr, times, tol):
    times = np.asarray(times, float)
    if len(sorted_arr) == 0:
        return np.zeros(len(times), bool)
    j = np.searchsorted(sorted_arr, times)
    a = np.abs(sorted_arr[np.clip(j - 1, 0, len(sorted_arr) - 1)] - times)
    b = np.abs(sorted_arr[np.clip(j, 0, len(sorted_arr) - 1)] - times)
    return np.minimum(a, b) < tol


def quake_flags(eps_df, half=HALF_CORE):
    return quake_flags_t(eps_df.t0.values, half)


def quake_flags_t(times, half=HALF_CORE):
    """Flag station-windows where a regional M>=3 event could produce >0.3 nanostrain static strain
    (crude M0/(mu r^3)), or a dyn_trigger catalogue P arrival falls in [t0-2 d, t0+2 d] at all."""
    fq = f'{RAW}/ncedc_events_M3_200km.csv'
    q = pd.read_csv(fq)
    q['t'] = pd.to_datetime(q.time.str.replace('Z', '')).astype('datetime64[ns]').astype('int64') / 1e9
    M0 = 10 ** (1.5 * q.mag.values + 9.1)
    flags = {}
    for sta, (la, lo) in STA_LL.items():
        dx = (q.lon.values - lo) * 111.195 * np.cos(np.radians(la))
        dy = (q.lat.values - la) * 111.195
        r = np.sqrt(dx ** 2 + dy ** 2 + q.depth_km.values ** 2) * 1e3
        est = M0 / (MU * r ** 3) * 1e9
        big = q.t.values[est > 0.3]
        big = np.sort(big)
        flags[sta] = near(big, times, half * 86400)
    tp = np.sort(pd.read_csv(f'{ROOT}/data/sci/dyn_trigger/quakes_ptime.csv').t_p.values)
    tele = near(tp, times, half * 86400)
    return flags, tele


def analyse():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20261122)
    S = {}
    # ---------------- episodes ----------------
    th = pd.read_csv(f'{ROOT}/data/sci/cascades2/episodes_thin.csv')
    allep = pd.read_csv(f'{ROOT}/data/cascades/episodes.csv')
    allep_t = np.sort(pd.to_datetime(allep.t0).astype('datetime64[ns]').astype('int64').values / 1e9)
    th = th[th.t0 > pd.Timestamp('2008-11-01').value / 1e9]
    sel = th.sort_values(['n_events', 'n_fam', 'dur_h'], ascending=False).head(200).sort_values('t0').reset_index(drop=True)
    sel['M0_lfe'], sel['n_lfe_all'] = episode_lfe_moment(sel)
    flags, tele = quake_flags(sel)
    sel['tele_in_window'] = tele
    for s_ in STATIONS:
        sel[f'eq_{s_}'] = flags[s_]
    S['episodes'] = dict(n_selected=len(sel), rank='thin catalogue, top 200 by n_events (then n_fam, dur_h), t0 >= 2008-11-01',
                         n_events_median=float(sel.n_events.median()), n_fam_median=float(sel.n_fam.median()),
                         dur_h_median=float(sel.dur_h.median()), min_n_events=int(sel.n_events.min()),
                         M0_lfe_median_Nm=float(sel.M0_lfe.median()), n_tele_flag=int(tele.sum()))
    print(S['episodes'], flush=True)
    busy = [(allep_t, 3 * 3600.0), (np.sort(sel.t0.values), 2.5 * 86400)]
    GR = greens()
    S['greens_nanostrain_per_m3'] = GR
    rows, nrows, series = [], [], {s_: [] for s_ in STATIONS}
    for sta in STATIONS:
        G = load_station(sta)
        for k, r in sel.iterrows():
            dur_d = r.dur_h / 24
            w = process_window(G, r.t0, dur_d, want_series=True)
            if w is None:
                continue
            ten = to_tensor(sta, w['steps']); ten1 = to_tensor(sta, w['steps1d'])
            row = dict(sta=sta, ep=int(r.ep), k=k, t0=r.t0, use_p=w['use_p'], eqflag=bool(r[f'eq_{sta}']),
                       tele=bool(r.tele_in_window))
            for g in range(4):
                row[f'g{g + 1}'] = w['steps'][g]; row[f'g{g + 1}_1d'] = w['steps1d'][g]
                row[f'rms{g + 1}'] = w['rms'][g]; row[f'pcoef{g + 1}'] = w['pcoef'][g]; row[f'M2_{g + 1}'] = w['M2amp'][g]
            for c, v, v1 in zip(COMPS, ten, ten1):
                row[c] = v; row[c + '_1d'] = v1
            rows.append(row)
            if np.isfinite(ten[0]):
                ser = to_tensor(sta, w['series'])
                series[sta].append((k, ser))
            # null windows
            for tn in null_candidates(r.t0, busy, rng, nmax=40):
                wn = process_window(G, tn, dur_d)
                if wn is None:
                    continue
                tn_ = to_tensor(sta, wn['steps']); tn1 = to_tensor(sta, wn['steps1d'])
                nr = dict(sta=sta, k=k, tn=tn, rmsmax=np.nanmax(wn['rms']) if np.isfinite(wn['rms']).any() else np.nan)
                for g in range(4):
                    nr[f'g{g + 1}'] = wn['steps'][g]
                for c, v, v1 in zip(COMPS, tn_, tn1):
                    nr[c] = v; nr[c + '_1d'] = v1
                nrows.append(nr)
        print(sta, 'done', sum(1 for x in rows if x['sta'] == sta), flush=True)
    W = pd.DataFrame(rows); N = pd.DataFrame(nrows)
    W.to_csv(f'{OUT}/episode_window_steps.csv', index=False)
    N.to_csv(f'{OUT}/null_window_steps.csv.gz', index=False)
    sel.to_csv(f'{OUT}/selected_episodes.csv', index=False)
    np.save(f'{OUT}/core_series.npy', {s_: [(k, v.astype(np.float32)) for k, v in series[s_]] for s_ in STATIONS},
            allow_pickle=True)
    summarise(sel, W, N, series, GR, S, rng, plt)


def robust_sigma(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def summarise(sel, W, N, series, GR, S, rng, plt):
    NB = 2000
    out_rows = []
    # QC: reject windows with outlying residual noise (per station, any gauge > 3x median)
    for sta in STATIONS:
        m = W.sta == sta
        rms = W.loc[m, ['rms1', 'rms2', 'rms3', 'rms4']].max(axis=1)
        W.loc[m, 'noisy'] = rms > 3 * np.nanmedian(rms)
    W['clean'] = ~W.noisy.astype(bool) & ~W.eqflag & ~W.tele
    # same QC for null windows
    fl, te = quake_flags_t(N.tn.values)
    N['clean'] = ~te
    for sta in STATIONS:
        m = (W.sta == sta).values
        thr = 3 * np.nanmedian(W.loc[m, ['rms1', 'rms2', 'rms3', 'rms4']].max(axis=1))
        mn = (N.sta == sta).values
        N.loc[mn, 'clean'] = N.loc[mn, 'clean'] & ~fl[sta][mn] & (N.loc[mn, 'rmsmax'] <= thr)
    N['clean'] = N.clean.astype(bool)
    # winsorise every step column at +-4 robust sigma (sigma from clean null windows of that
    # station); applied identically to episode and null windows, so the null test stays exact
    for sta in STATIONS:
        mw, mn = (W.sta == sta).values, (N.sta == sta).values
        for c in COMPS + ['g1', 'g2', 'g3', 'g4']:
            for suf in ['', '_1d']:
                col = c + suf
                if col not in N:
                    continue
                s_ = robust_sigma(N.loc[mn & N.clean.values, col].values)
                W.loc[mw, col] = W.loc[mw, col].clip(-4 * s_, 4 * s_)
                N.loc[mn, col] = N.loc[mn, col].clip(-4 * s_, 4 * s_)
    N_all = N
    S['null_windows'] = dict(n_total=int(len(N)), n_clean=int(N.clean.sum()),
                             note='step values winsorised at +-4 robust sigma (clean-null sigma per station/component)')
    # per-station barometric response and M2 amplitude (sanity)
    S['station_qc'] = {}
    for sta in STATIONS:
        m = W.sta == sta
        pc = to_tensor(sta, np.nanmedian(W.loc[m, ['pcoef1', 'pcoef2', 'pcoef3', 'pcoef4']].values, axis=0))
        S['station_qc'][sta] = dict(n_windows=int(m.sum()), n_valid_areal=int(np.isfinite(W.loc[m, 'areal']).sum()),
                                    n_clean=int((m & W.clean & np.isfinite(W.areal)).sum()),
                                    baro_areal_nstrain_per_hPa=float(pc[0]),
                                    M2_gauge_amp_nstrain=[float(x) for x in np.nanmedian(W.loc[m, ['M2_1', 'M2_2', 'M2_3', 'M2_4']].values, axis=0)],
                                    dist_km=GR[sta]['dist_km'])
    # ---------- per station/component stacks ----------
    variants = {'clean': W.clean, 'all': np.ones(len(W), bool)}
    for vname, vm in variants.items():
        for sta in STATIONS:
            for c in COMPS + ['g1', 'g2', 'g3', 'g4']:
                for suf in ['', '_1d']:
                    col = c + suf
                    if col not in W or col not in N:
                        continue
                    m = (W.sta == sta) & vm & np.isfinite(W[col])
                    x = W.loc[m, col].values; ks = W.loc[m, 'k'].values
                    if len(x) < 10:
                        continue
                    obs_mean, obs_med = x.mean(), np.median(x)
                    bs = np.array([rng.choice(x, len(x)).mean() for _ in range(NB)])
                    NN = N_all[N_all.clean] if vname == 'clean' else N_all
                    Nn = NN[(NN.sta == sta) & np.isfinite(NN[col])]
                    groups = {k: g[col].values for k, g in Nn.groupby('k')}
                    ks2 = [k for k in ks if k in groups]
                    nul = np.array([np.mean([groups[k][rng.integers(len(groups[k]))] for k in ks2]) for _ in range(NB)])
                    sig1 = robust_sigma(Nn[col].values)
                    out_rows.append(dict(variant=vname, sta=sta, comp=col, n=len(x), mean_nstrain=obs_mean,
                                         median_nstrain=obs_med, ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5),
                                         null_mean=nul.mean(), null_2p5=np.percentile(nul, 2.5), null_97p5=np.percentile(nul, 97.5),
                                         p_two_sided=float((np.abs(nul - nul.mean()) >= abs(obs_mean - nul.mean())).mean()),
                                         single_window_sigma_null=sig1,
                                         pred_per_1e3m3=(GR[sta][c] * 1e3 if c in COMPS else GR[sta]['gauges'][int(c[1]) - 1] * 1e3)))
    ST = pd.DataFrame(out_rows)
    ST.to_csv(f'{OUT}/stack_by_station_component.csv', index=False)
    # ---------- joint potency estimate per episode (weighted LS over stations x components) ----------
    def potency(df, col_suffix=''):
        """per-row potency numerator/denominator, weights from null sigma per station-component."""
        num = np.zeros(len(df)); den = np.zeros(len(df))
        for sta in STATIONS:
            for c in COMPS:
                sig = sigmas[(sta, c + col_suffix)]
                m = (df.sta == sta).values & np.isfinite(df[c + col_suffix].values)
                gg = GR[sta][c]
                num[m] += df[c + col_suffix].values[m] * gg / sig ** 2
                den[m] += gg ** 2 / sig ** 2
        return num, den
    joint = {}
    for suf in ['', '_1d']:
        NC = N_all[N_all.clean]
        sigmas = {(sta, c + suf): robust_sigma(NC.loc[NC.sta == sta, c + suf].values) for sta in STATIONS for c in COMPS}
        for vname, vm in variants.items():
            Wv = W[vm & np.isfinite(W.areal)]
            nu, de = potency(Wv, suf)
            E = pd.DataFrame(dict(k=Wv.k.values, nu=nu, de=de)).groupby('k').sum()
            P_ep = E.nu / E.de                                  # potency per episode, m^3
            sigP_ep = 1 / np.sqrt(E.de)
            # stacked: inverse-variance mean over episodes = sum nu / sum de
            Pst = E.nu.sum() / E.de.sum()
            idx = E.index.values
            bs = []
            for _ in range(NB):
                j = rng.choice(len(idx), len(idx))
                bs.append(E.nu.values[j].sum() / E.de.values[j].sum())
            bs = np.array(bs)
            # null: one null window per episode per station (same station set)
            NN = N_all[N_all.clean] if vname == 'clean' else N_all
            Nn = NN[np.isfinite(NN.areal)].copy()
            nn_, nd_ = potency(Nn, suf)
            Nn['nu'], Nn['de'] = nn_, nd_
            G_ = {(k, s_): g[['nu', 'de']].values for (k, s_), g in Nn.groupby(['k', 'sta'])}
            pairs = list(zip(Wv.k.values, Wv.sta.values))
            nul = []
            for _ in range(NB):
                a = 0.0; b = 0.0
                for pr in pairs:
                    if pr in G_:
                        arr = G_[pr]; v = arr[rng.integers(len(arr))]
                        a += v[0]; b += v[1]
                nul.append(a / b)
            nul = np.array(nul)
            nul_c = nul - nul.mean()
            ub95 = (Pst - nul.mean()) + np.percentile(np.abs(nul_c), 95)
            joint[f'{vname}{suf}'] = dict(
                n_episodes=int(len(E)), potency_stack_m3=float(Pst), M0_stack_Nm=float(Pst * MU),
                boot95_m3=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                null_mean_m3=float(nul.mean()), null95_m3=[float(np.percentile(nul, 2.5)), float(np.percentile(nul, 97.5))],
                p_two_sided=float((np.abs(nul_c) >= abs(Pst - nul.mean())).mean()),
                upper95_abs_potency_m3=float(abs(Pst - nul.mean()) + np.percentile(np.abs(nul_c), 95)),
                single_episode_sigma_m3_median=float(np.median(sigP_ep)))
            if vname == 'clean' and suf == '':
                ep_tab = pd.DataFrame(dict(k=idx, potency_m3=P_ep.values, sigma_m3=sigP_ep.values))
    S['joint_potency'] = joint
    # ---------- slip conversions ----------
    Pub = joint['clean']['upper95_abs_potency_m3']
    Pub1 = joint['clean_1d']['upper95_abs_potency_m3']
    M0l = sel.M0_lfe.values
    conv = {}
    for L in (1, 3, 5):
        A = (L * 1e3) ** 2
        conv[f'L{L}km'] = dict(slip_upper95_mm=Pub / A * 1e3, slip_upper95_1d_ramp_mm=Pub1 / A * 1e3,
                               slip_stack_mm=joint['clean']['potency_stack_m3'] / A * 1e3,
                               slip_stack_boot95_mm=[x / A * 1e3 for x in joint['clean']['boot95_m3']],
                               lfe_slip_median_mm=float(np.median(M0l) / MU / A * 1e3),
                               lfe_slip_mean_mm=float(np.mean(M0l) / MU / A * 1e3))
    S['slip_conversion_square_patch'] = conv
    ep_per_yr = len(pd.read_csv(f'{ROOT}/data/sci/cascades2/episodes_thin.csv')) / ((1.7826e9 - 1.2309e9) / 3.156e7)
    S['detection_limit'] = dict(
        upper95_potency_m3=Pub, upper95_M0_Nm=Pub * MU, upper95_Mw=float((np.log10(Pub * MU) - 9.1) / 1.5),
        upper95_potency_1d_ramp_m3=Pub1, upper95_Mw_1d_ramp=float((np.log10(Pub1 * MU) - 9.1) / 1.5),
        lfe_M0_per_episode_median_Nm=float(np.median(M0l)), lfe_M0_per_episode_mean_Nm=float(np.mean(M0l)),
        lfe_Mw_equiv_median=float((np.log10(np.median(M0l)) - 9.1) / 1.5),
        ratio_upper_to_lfe_median=float(Pub * MU / np.median(M0l)),
        single_episode_sigma_potency_m3=joint['clean']['single_episode_sigma_m3_median'],
        single_episode_sigma_Mw=float((np.log10(2 * joint['clean']['single_episode_sigma_m3_median'] * MU) - 9.1) / 1.5),
        loading_budget=dict(note='if 2.5 cm/yr on a 15x5 km array-wide zone were released only in thin-catalogue episodes',
                            episodes_per_yr=float(ep_per_yr),
                            potency_per_episode_m3=float(0.025 * 15e3 * 5e3 / ep_per_yr)),
        assumptions=f'median LFE Mw {MW_MED} (M0 {M0_MED:.2e} N m), M0 proportional to relative amplitude A (per-family median 1), '
                    f'mu 30 GPa; point source at {LFE_LAT}N {LFE_LON}E {LFE_Z / 1e3:.0f} km, strike {STRIKE} dip {DIP} rake {RAKE} '
                    '(Okada half-space, nu 0.25); unit-coupling (uncalibrated) gauge-to-tensor conversion; upper bound = |stack - null mean| + 95th pct of |null|.')
    S['tide_pressure_method'] = ('per window: LS fit over t0+-15 d with the core t0+-2 d masked: Legendre deg-3 trend + M2,S2,N2,K1,O1 + '
                                 'barometric (RDO, 30-min interpolated); tides+pressure predicted into the core and removed; core model '
                                 'a + b t + c ramp(t0..t0+dur) gives the step c (and a 1-day ramp variant). Offsets outside the core removed; '
                                 'windows with offsets inside the core rejected. Body-tide model not used: it lacks ocean loading at these coastal sites.')
    S['sign_convention'] = ('gauge strain linearised as (R/(1-R))*(G/D), R=counts/1e8, G/D=1e-4/0.087 (UNAVCO), extension positive; '
                            'areal = eEE+eNN, gam1 = eEE-eNN, gam2 = 2eEN from e_i = eA/2 - g1 cos(2th)/2 + g2 sin(2th)/2 (th = gauge az E of N), '
                            'unit coupling (no in-situ calibration).')
    with open(f'{OUT}/summary.json', 'w') as f:
        json.dump(S, f, indent=1, default=float)
    # per-episode table
    ep_tab = ep_tab.merge(sel[['ep', 't0', 'n_fam', 'n_events', 'dur_h', 'M0_lfe']].reset_index().rename(columns={'index': 'k'}), on='k')
    ep_tab['potency_lfe_m3'] = ep_tab.M0_lfe / MU
    ep_tab.to_csv(f'{OUT}/episode_potency.csv', index=False)
    print(json.dumps(S['joint_potency'], indent=1)); print(json.dumps(S['detection_limit'], indent=1))
    figures(sel, W, ST, series, GR, S, plt)


def figures(sel, W, ST, series, GR, S, plt):
    # Fig 1: time-series stacks per station (areal, gam1, gam2), clean windows
    clean_k = {sta: set(W[(W.sta == sta) & W.clean].k) for sta in STATIONS}
    tc = np.arange(-int(HALF_CORE * 144), int(HALF_CORE * 144) + 1) / 144.0
    fig, ax = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True)
    for j, sta in enumerate(STATIONS):
        arr = np.array([v for k, v in series[sta] if k in clean_k[sta]])
        if len(arr) == 0:
            continue
        for i, c in enumerate(COMPS):
            X = arr[:, i, :len(tc)]
            mu = np.nanmean(X, axis=0)
            n = np.sum(np.isfinite(X), axis=0)
            se = np.nanstd(X, axis=0) / np.sqrt(np.maximum(n, 1))
            a = ax[i, j]
            a.fill_between(tc, mu - 2 * se, mu + 2 * se, color='0.8')
            a.plot(tc, mu, 'k', lw=1)
            pred = GR[sta][c] * 1e4
            a.plot(tc, pred * np.clip(tc / (np.median(sel.dur_h) / 24), 0, 1), 'r--', lw=1,
                   label='1e4 m$^3$ potency' if (i == 0 and j == 0) else None)
            a.axvline(0, color='b', lw=0.6)
            a.set_title(f'{sta} {c} (n={len(arr)}, {GR[sta]["dist_km"]:.0f} km)', fontsize=9)
            if j == 0:
                a.set_ylabel('nanostrain')
            if i == 2:
                a.set_xlabel('days from episode t0')
    ax[0, 0].legend(fontsize=8)
    fig.suptitle('PBO strain stacked on the 200 largest LFE episodes (tides, pressure removed; pre-event trend removed); '
                 'band = +-2 s.e.; red = prediction for 1e4 m^3 (Mw~3.6) RL slip at 28 km', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{FIG}/sci_strain_stacks.png', dpi=130); plt.close(fig)
    # Fig 2: per-station-component step vs null band, and the joint potency bound
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    T = ST[(ST.variant == 'clean') & ST.comp.isin(COMPS)].reset_index(drop=True)
    x = np.arange(len(T))
    ax[0].fill_between(x, T.null_2p5, T.null_97p5, step='mid', color='0.85', label='null 95% (matched ToD + M2 phase)')
    ax[0].errorbar(x, T.mean_nstrain, yerr=[T.mean_nstrain - T.ci95_lo, T.ci95_hi - T.mean_nstrain], fmt='ko', ms=4,
                   label='stack mean, bootstrap 95%')
    ax[0].plot(x, T.pred_per_1e3m3 * 10, 'r^', ms=5, label='predicted for 1e4 m$^3$ potency')
    ax[0].set_xticks(x); ax[0].set_xticklabels([f'{a}\n{b}' for a, b in zip(T.sta, T.comp)], fontsize=7)
    ax[0].axhline(0, color='k', lw=0.5); ax[0].set_ylabel('step at episode (nanostrain)'); ax[0].legend(fontsize=8)
    ax[0].set_title('Strain step over the episode duration (ramp t0..t0+dur), clean windows')
    J = S['joint_potency']
    labs = ['clean', 'all', 'clean_1d', 'all_1d']
    y = np.arange(len(labs))
    for i, l in enumerate(labs):
        d = J[l]
        ax[1].plot(d['null95_m3'], [i, i], color='0.6', lw=6, solid_capstyle='butt')
        ax[1].errorbar(d['potency_stack_m3'], i, xerr=[[d['potency_stack_m3'] - d['boot95_m3'][0]], [d['boot95_m3'][1] - d['potency_stack_m3']]],
                       fmt='ko')
    lfe = np.median(sel.M0_lfe) / MU
    ax[1].axvline(lfe, color='C1', ls='--', label=f'median LFE potency/episode ({lfe:.0f} m$^3$)')
    ax[1].axvline(S['detection_limit']['loading_budget']['potency_per_episode_m3'], color='C2', ls=':',
                  label='2.5 cm/yr x 75 km$^2$ / episodes per yr')
    ax[1].set_yticks(y); ax[1].set_yticklabels(['step, clean', 'step, all', '1-d ramp, clean', '1-d ramp, all'])
    ax[1].set_xlabel('stacked potency per episode (m$^3$); grey = null 95%'); ax[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13))
    ax[1].set_title('Joint 4-station estimate (RL, strike 293 dip 40, 28 km)')
    ax[1].axvline(0, color='k', lw=0.5)
    fig.tight_layout(); fig.savefig(f'{FIG}/sci_strain_null_potency.png', dpi=130); plt.close(fig)


def stage_geometry():
    """Sensitivity of the joint potency bound to the assumed source geometry/location, re-using
    the saved window steps (same QC + winsorisation as summarise)."""
    rng = np.random.default_rng(7)
    W = pd.read_csv(f'{OUT}/episode_window_steps.csv'); N = pd.read_csv(f'{OUT}/null_window_steps.csv.gz')
    for sta in STATIONS:
        m = W.sta == sta
        rms = W.loc[m, ['rms1', 'rms2', 'rms3', 'rms4']].max(axis=1)
        W.loc[m, 'noisy'] = rms > 3 * np.nanmedian(rms)
    W['clean'] = ~W.noisy.astype(bool) & ~W.eqflag & ~W.tele
    fl, te = quake_flags_t(N.tn.values)
    N['clean'] = ~te
    for sta in STATIONS:
        m = (W.sta == sta).values
        thr = 3 * np.nanmedian(W.loc[m, ['rms1', 'rms2', 'rms3', 'rms4']].max(axis=1))
        mn = (N.sta == sta).values
        N.loc[mn, 'clean'] = N.loc[mn, 'clean'] & ~fl[sta][mn] & (N.loc[mn, 'rmsmax'] <= thr)
    N['clean'] = N.clean.astype(bool)
    for sta in STATIONS:
        mw, mn = (W.sta == sta).values, (N.sta == sta).values
        for col in COMPS:
            s_ = robust_sigma(N.loc[mn & N.clean.values, col].values)
            W.loc[mw, col] = W.loc[mw, col].clip(-4 * s_, 4 * s_)
            N.loc[mn, col] = N.loc[mn, col].clip(-4 * s_, 4 * s_)
    W = W[W.clean & np.isfinite(W.areal)]; N = N[N.clean & np.isfinite(N.areal)]
    sig = {(s_, c): robust_sigma(N.loc[N.sta == s_, c].values) for s_ in STATIONS for c in COMPS}
    cases = [('RL 293/40 28 km (adopted)', {}), ('RL 293/90 28 km', dict(dip=90.0)),
             ('RL 293/40 25 km', dict(depth_m=25e3)), ('RL 293/40 31 km', dict(depth_m=31e3)),
             ('RL 293/40 shifted 7 km W', dict(e0=-7.0)), ('RL 293/40 shifted 7 km E', dict(e0=7.0)),
             ('RL 293/40 shifted 7 km N', dict(n0=7.0)), ('RL 293/40 shifted 7 km S', dict(n0=-7.0)),
             ('reverse 293/40 (rake 90)', dict(rake=90.0))]
    rows = []
    for name, kw in cases:
        GR = greens(**kw)
        def nd(df):
            nu = np.zeros(len(df)); de = np.zeros(len(df))
            for s_ in STATIONS:
                m = (df.sta == s_).values
                for c in COMPS:
                    g = GR[s_][c]
                    nu[m] += df[c].values[m] * g / sig[(s_, c)] ** 2
                    de[m] += g ** 2 / sig[(s_, c)] ** 2
            return nu, de
        nu, de = nd(W); Pst = nu.sum() / de.sum()
        nn, ndn = nd(N); N2 = pd.DataFrame(dict(k=N.k.values, sta=N.sta.values, nu=nn, de=ndn))
        G_ = {key: g[['nu', 'de']].values for key, g in N2.groupby(['k', 'sta'])}
        pairs = [pr for pr in zip(W.k.values, W.sta.values) if pr in G_]
        nul = []
        for _ in range(2000):
            v = np.array([G_[pr][rng.integers(len(G_[pr]))] for pr in pairs])
            nul.append(v[:, 0].sum() / v[:, 1].sum())
        nul = np.array(nul); c0 = nul - nul.mean()
        ub = abs(Pst - nul.mean()) + np.percentile(np.abs(c0), 95)
        rows.append(dict(geometry=name, potency_stack_m3=Pst, null_mean_m3=nul.mean(),
                         p_two_sided=float((np.abs(c0) >= abs(Pst - nul.mean())).mean()),
                         upper95_potency_m3=ub, upper95_Mw=(np.log10(ub * MU) - 9.1) / 1.5,
                         upper95_slip_mm_3km=ub / 9e6 * 1e3,
                         max_abs_strain_per_1e4m3_nstrain=max(abs(GR[s_][c]) for s_ in STATIONS for c in COMPS) * 1e4))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/geometry_sensitivity.csv', index=False)


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'analyse'
    if st == 'download':
        stage_download()
    elif st == 'analyse':
        analyse()
    elif st == 'geometry':
        stage_geometry()
