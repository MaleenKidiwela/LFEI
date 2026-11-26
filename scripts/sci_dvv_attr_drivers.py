#!/usr/bin/env python
"""Build daily driver series for the dv/v attribution (one table per seismic station + regional tables).

Stages:  quakes   catalogue (ComCat M>=3.5, 250 km) + teleseisms (data/sci/dyn_trigger) -> per-station PGV (BSSA14,
                  Vs30 760; Rjb from finite rupture where available) and static volumetric strain (Okada, imported
                  from scripts/sci_dose_response.py) -> drivers/quakes_station.csv
         climate  Open-Meteo (ERA5 / ERA5-Land) + GHCN nearest station + CO-OPS sea level -> drivers/climate_<sta>.parquet
         geod     NGL GNSS: vertical loading proxy (steps/velocity/quake terms removed, seasonal KEPT, no CME),
                  areal strain from local gradients, M7 postseismic curve -> drivers/gnss_*.parquet
         bstrain  PB borehole areal strain (daily, destepped, baro-regressed, >3-yr trend removed) -> drivers/bstrain.parquet
         power    microseism log10 power (PB.B045 EHZ proxy) -> drivers/power.parquet
Bulk output: /data/wsd02/maleen_data/dvv_attr/drivers/.  Raw input files are parsed as data only.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
BULK = '/data/wsd02/maleen_data/dvv_attr'
RAW = f'{BULK}/raw'
DRV = f'{BULK}/drivers'
OUT = f'{ROOT}/data/sci/dvv_attr'
os.makedirs(DRV, exist_ok=True)
DAYS = pd.date_range('2007-01-01', '2026-11-24', freq='D')
KM = 111.195
BIG3 = {'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20', 'M7': '2024-12-05'}


def seis_stations():
    c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv')
    c['ns'] = c.net + '.' + c.sta
    c = c[c.start < '2025-09-01']
    g = c.groupby('ns').agg(lat=('lat', 'median'), lon=('lon', 'median'), depth_m=('depth', 'median')).reset_index()
    g['z_km'] = np.where(g.ns.str.startswith('PB.'), g.depth_m / 1000.0, 0.0)
    return g


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(np.asarray(lon2) - lon1), p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------------------------------------ quakes
def stage_quakes():
    import sci_dose_response as dr
    S = seis_stations()
    C = pd.read_csv(f'{RAW}/comcat/local_M3.5_250km.csv')
    C = C[C.type == 'earthquake'].copy()
    C['t'] = pd.to_datetime(C.time).dt.tz_localize(None)
    C = C[C.t >= '2007-06-01'].reset_index(drop=True)
    print('local quakes', len(C))
    # finite / rectangle sources for the events with mechanisms (as in the Coulomb analysis)
    src = {}
    for ev in dr.MECH:
        r = C[C.id == ev]
        if len(r) == 0:
            print('MECH event not in ComCat list', ev); continue
        r = r.iloc[0]
        V, pl, alt = dr.source_variants(ev, r.mag, r.latitude, r.longitude, r.depth)
        src[ev] = V['primary']
    se, sn = dr.ll2en(S.lat.values, S.lon.values)
    P = np.column_stack([se, sn, -np.maximum(S.z_km.values, 0.02)])
    rows = []
    for _, q in C.iterrows():
        repi = hav(q.latitude, q.longitude, S.lat.values, S.lon.values)
        if q.id in src:
            pts = dr.rupture_points(src[q.id])
            pe, pn = pts[:, 0], pts[:, 1]
            rjb = np.array([np.min(np.hypot(pe - a, pn - b)) for a, b in zip(se, sn)])
            eps = dr.strain_fd(src[q.id], P)
            vol = np.trace(eps, axis1=1, axis2=2)
            areal = eps[:, 0, 0] + eps[:, 1, 1]
            srcflag = 'finite/rect'
        else:
            rjb = np.sqrt(repi ** 2 + max(q.depth, 0.0) ** 2)     # point source: hypocentral distance (review M6)
            vol = np.zeros(len(S)); areal = np.zeros(len(S))
            if q.mag >= 4.5:        # generic strike-slip rectangle (MFZ style) for unmodelled M>=4.5 events
                rr = dr.wc_rect(q.latitude, q.longitude, max(q.depth, 5.0), (290, 85, 180), q.mag, 'SS')
                eps = dr.strain_fd(rr, P)
                vol = np.trace(eps, axis1=1, axis2=2); areal = eps[:, 0, 0] + eps[:, 1, 1]
            srcflag = 'point' if q.mag < 4.5 else 'generic_rect'
        pgv = dr.bssa14_pgv(q.mag, rjb, 'SS')
        for j, s in enumerate(S.ns):
            rows.append(dict(qid=q.id, t=q.t, mag=q.mag, qlat=q.latitude, qlon=q.longitude, qdep=q.depth, sta=s,
                             repi_km=repi[j], rjb_km=rjb[j], pgv_cms=pgv[j], vol_strain=vol[j], areal_strain=areal[j],
                             src=srcflag, kind='local'))
    # teleseisms: measured long-period PGV at JCC/HOPS (same value assigned to every station)
    T = pd.read_csv(f'{ROOT}/data/sci/dyn_trigger/per_quake_pgv_stress_triggering.csv')
    col = 'JCC_bb_pgv3' if 'JCC_bb_pgv3' in T else None
    pg = T[col].values * 100.0 if col else T.pgv_proxy_um_s.values * 1e-4     # m/s -> cm/s
    pg = np.where(np.isfinite(pg), pg, T.pgv_proxy_um_s.values * 1e-4)
    T['t'] = pd.to_datetime(T.time).dt.tz_localize(None)
    for (_, q), p in zip(T.iterrows(), pg):
        for s in S.ns:
            rows.append(dict(qid=q.id, t=q.t, mag=q.mag, qlat=q.latitude, qlon=q.longitude, qdep=q.depth, sta=s,
                             repi_km=q.dist_km, rjb_km=q.dist_km, pgv_cms=p, vol_strain=0.0, areal_strain=0.0,
                             src='tele_measured', kind='tele'))
    Q = pd.DataFrame(rows)
    # measured PGV (NCEDC HN/HH) for the big three where available
    M = pd.read_csv(f'{ROOT}/data/sci/dvv_depth/pgv_by_station.csv')
    key = {'pet': 'Petrolia', 'fer': 'Ferndale', 'm7': 'M7'}
    M['qname'] = M.quake.map(key)
    Q['pgv_meas_cms'] = np.nan
    for _, m in M.iterrows():
        if not np.isfinite(m.pgv_gm_cms):
            continue
        d = pd.Timestamp(BIG3[m.qname])
        k = (Q.sta == m.sta) & (Q.t.dt.normalize() == d) & (Q.kind == 'local')
        if k.any():
            i = Q[k].mag.idxmax()
            Q.loc[i, 'pgv_meas_cms'] = m.pgv_gm_cms
    # station site factor = median(measured / BSSA14) over the big-three events with measured PGV (review M6);
    # stations without measurements get the network median.  pgv_site_cms = BSSA14 x factor (measured where available).
    loc = Q[(Q.kind == 'local') & Q.pgv_meas_cms.notna()]
    # station term after removing each event's median (log) residual (event/source misfit of the GMPE), plus the
    # overall median event term as the global bias (code re-check item 6)
    lr = np.log10(loc.pgv_meas_cms / loc.pgv_cms)
    ev = lr.groupby(loc.qid).transform('median')
    sf = 10 ** ((lr - ev).groupby(loc.sta).median() + float(lr.groupby(loc.qid).median().median()))
    sf_med = float(sf.median()) if len(sf) else 1.0
    isbh = lambda x: x.str.startswith('PB.')
    med_bh = float(sf[isbh(pd.Series(sf.index, index=sf.index))].median())
    med_sf = float(sf[~isbh(pd.Series(sf.index, index=sf.index))].median())
    Q['site_factor'] = Q.sta.map(sf)
    Q.loc[Q.site_factor.isna() & isbh(Q.sta), 'site_factor'] = med_bh
    Q.loc[Q.site_factor.isna(), 'site_factor'] = med_sf
    Q['site_factor_src'] = np.where(Q.sta.isin(sf.index), 'measured big-3', 'class median (borehole/surface)')
    print('class medians borehole', round(med_bh, 2), 'surface', round(med_sf, 2))
    Q['pgv_site_cms'] = np.where(Q.kind == 'local', Q.pgv_cms * Q.site_factor, Q.pgv_cms)
    Q['pgv_site_cms'] = np.where(Q.pgv_meas_cms.notna(), Q.pgv_meas_cms, Q.pgv_site_cms)
    Q['modelled'] = Q.src.eq('finite/rect')
    print('site factors:', sf.round(2).to_dict(), 'network median', round(sf_med, 2))
    Q.to_csv(f'{DRV}/quakes_station.csv.gz', index=False)
    big = Q[(Q.kind == 'local') & (Q.pgv_cms > 0.5)]
    print('station-quake pairs with BSSA PGV > 0.5 cm/s:', len(big), 'quakes:', big.qid.nunique())
    s = Q[Q.kind == 'local'].groupby('qid').agg(t=('t', 'first'), mag=('mag', 'first'), pgv_max=('pgv_cms', 'max'),
                                                pgv_med=('pgv_cms', 'median'), vol_med=('vol_strain', 'median'))
    s.sort_values('pgv_max', ascending=False).head(40).to_csv(f'{OUT}/quakes_top40_pgv.csv')
    print(s.sort_values('pgv_max', ascending=False).head(25).to_string())


# ------------------------------------------------------------------------------------------------ climate
def load_openmeteo(sta, lat, lon):
    """Open-Meteo daily series from the nearest downloaded site (per product)."""
    out = []
    S = seis_stations().set_index('ns')
    for tag in ('best', 'era5land'):
        sites = [os.path.basename(f).rsplit('_', 1)[0] for f in glob.glob(f'{RAW}/openmeteo/*_{tag}.json')]
        if not sites:
            continue
        site = min(sites, key=lambda s_: float(hav(lat, lon, S.loc[s_, 'lat'], S.loc[s_, 'lon'])))
        fn = f'{RAW}/openmeteo/{site}_{tag}.json'
        j = json.load(open(fn))
        d = pd.DataFrame(j['daily'])
        d['time'] = pd.to_datetime(d.time)
        d = d.set_index('time')
        d.columns = [f'om_{c}' if tag == 'best' else f'el_{c}' for c in d.columns]
        d[f'site_{tag}'] = site
        out.append(d)
    return pd.concat(out, axis=1) if out else None


def load_power(sta):
    cells = pd.read_csv(f'{RAW}/power/station_cells.csv').set_index('ns')
    cell = cells.loc[sta, 'cell']
    j = json.load(open(f'{RAW}/power/cell_{cell}.json'))
    P = j['properties']['parameter']
    D = pd.DataFrame({k: pd.Series(v) for k, v in P.items()})
    D.index = pd.to_datetime(D.index, format='%Y%m%d')
    D = D.replace(-999.0, np.nan).reindex(DAYS)
    out = pd.DataFrame({'pw_prcp': D.PRECTOTCORR, 'pw_t2m': D.T2M, 'pw_ps_hpa': D.PS * 10.0,
                        'pw_gwettop': D.GWETTOP, 'pw_gwetroot': D.GWETROOT, 'pw_gwetprof': D.GWETPROF}, index=DAYS)
    out.attrs['cell'] = cell
    return out


def ghcn_table():
    d = f'{RAW}/ghcn'
    st = pd.read_fwf(f'{d}/ghcnd-stations.txt', header=None, colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (41, 71)],
                     names=['id', 'lat', 'lon', 'elev', 'name'])
    ids = [os.path.basename(f)[:-4] for f in glob.glob(f'{d}/*.csv') if 'inventory' not in f]
    st = st[st.id.isin(ids)].reset_index(drop=True)
    series = {}
    for i in st.id:
        try:
            x = pd.read_csv(f'{d}/{i}.csv', usecols=lambda c: c in ('DATE', 'PRCP', 'TMAX', 'TMIN', 'TAVG'), low_memory=False)
        except Exception:
            continue
        x['DATE'] = pd.to_datetime(x.DATE, errors='coerce')
        x = x.dropna(subset=['DATE']).set_index('DATE').reindex(DAYS)
        for c in ('PRCP', 'TMAX', 'TMIN', 'TAVG'):
            if c in x:
                x[c] = pd.to_numeric(x[c], errors='coerce') / 10.0     # tenths of mm / degC
        series[i] = x
    cov = {i: float(series[i]['PRCP'].loc['2008':'2026-10'].notna().mean()) if 'PRCP' in series[i] else 0.0 for i in series}
    st['prcp_cov'] = st.id.map(cov)
    return st, series


def stage_climate():
    S = seis_stations()
    st, G = ghcn_table()
    st.to_csv(f'{OUT}/ghcn_stations_used.csv', index=False)
    print('GHCN stations', len(st), 'with PRCP cov>0.7:', (st.prcp_cov > 0.7).sum())
    good = st[st.prcp_cov > 0.7].reset_index(drop=True)
    # sea level (monthly MSL -> daily linear interpolation)
    sl = []
    for f in sorted(glob.glob(f'{RAW}/coops/9418767_*.csv')):
        x = pd.read_csv(f, skipinitialspace=True)
        x.columns = [c.strip() for c in x.columns]
        if 'MSL' in x:
            x['t'] = pd.to_datetime(dict(year=x.Year, month=x.Month, day=15))
            sl.append(x[['t', 'MSL']])
    SL = pd.concat(sl).drop_duplicates('t').set_index('t').MSL.astype(float) if sl else None
    meta = []
    for _, r in S.iterrows():
        om = load_openmeteo(r.ns, r.lat, r.lon)
        D = pd.DataFrame(index=DAYS)
        if om is not None:       # only ERA5-Land soil layers are used (Open-Meteo 'best_match' is inhomogeneous)
            om = om.reindex(DAYS)
            for c in om.columns:
                if c.startswith('el_') or c == 'site_era5land':
                    D[c] = om[c]
        pw = load_power(r.ns)
        for c in pw.columns:
            D[c] = pw[c]
        d = hav(r.lat, r.lon, good.lat.values, good.lon.values)
        o = np.argsort(d)
        # nearest GHCN precipitation; gaps filled from next-nearest stations scaled by ratio of means
        p = pd.Series(np.nan, index=DAYS)
        used = []
        for k in o[:4]:
            if d[k] > 60:
                break
            g = G[good.id[k]]['PRCP']
            if p.isna().all():
                p = g.copy(); used.append(good.id[k])
            else:
                both = p.notna() & g.notna()
                sc = p[both].mean() / max(g[both].mean(), 1e-3) if both.sum() > 365 else 1.0
                fill = p.isna() & g.notna()
                p[fill] = g[fill] * sc
                used.append(good.id[k])
        D['ghcn_prcp_raw'] = p
        # gap-fill with POWER precipitation scaled by the LOCAL gauge/POWER ratio (2-yr centred window)
        e = D['pw_prcp']
        both = p.notna() & e.notna()
        gs = p.where(both, 0.0).rolling(731, center=True, min_periods=1).sum()
        es = e.where(both, 0.0).rolling(731, center=True, min_periods=1).sum()
        nb = both.astype(float).rolling(731, center=True, min_periods=1).sum()
        glob_ratio = p[both].sum() / max(e[both].sum(), 1e-3)
        ratio = (gs / es.where(es > 0)).where(nb >= 180).fillna(glob_ratio)
        D['ghcn_prcp'] = p.where(p.notna(), e * ratio)
        D['prcp_ratio_local'] = ratio
        # nearest GHCN temperature
        for k in o[:6]:
            g = G[good.id[k]]
            if 'TMAX' in g and g['TMAX'].loc['2008':'2026'].notna().mean() > 0.7:
                D['ghcn_tmean'] = (g['TMAX'] + g['TMIN']) / 2; D.attrs['tsta'] = good.id[k]; break
        if SL is not None:
            D['sealevel_m'] = SL.reindex(SL.index.union(DAYS)).interpolate('time').reindex(DAYS).values
        D.index.name = 't'
        D.to_parquet(f'{DRV}/climate_{r.ns}.parquet')
        rr = (p[both].groupby(p[both].index.year).sum() / e[both].groupby(e[both].index.year).sum()).round(2)
        meta.append(dict(sta=r.ns, power_cell=pw.attrs.get('cell', ''), el_site=D['site_era5land'].iloc[0] if 'site_era5land' in D else '', ghcn_prcp=';'.join(used), ghcn_dist_km=float(d[o[0]]),
                         r_power_vs_ghcn_10d=float(D.pw_prcp.rolling(10).sum().corr(D.ghcn_prcp_raw.rolling(10).sum())),
                         gauge_over_power_by_year=json.dumps(rr.to_dict()), frac_gapfilled=float(p.loc['2008':'2026-10'].isna().mean()),
                         prcp_power_mm_yr=float(D.pw_prcp.loc['2008':'2025'].mean() * 365.25),
                         prcp_ghcn_mm_yr=float(D.ghcn_prcp.loc['2008':'2025'].mean() * 365.25)))
        print(meta[-1], flush=True)
    pd.DataFrame(meta).to_csv(f'{OUT}/climate_station_meta.csv', index=False)


# ------------------------------------------------------------------------------------------------ geodesy
def stage_geod():
    import sci_gnss as gn
    import sci_dose_response as dr
    M = pd.read_csv(f'{ROOT}/data/sci/gnss/stations.csv')
    M = M[(M.set == 'near')].reset_index(drop=True)
    day0, day1 = gn.dt2mjd('2007-01-01'), gn.dt2mjd('2026-11-24')
    mjd_all = np.arange(day0, day1)
    Q = gn.quake_mjd()
    comp = {c: {} for c in 'enu'}
    post = {c: {} for c in 'enu'}
    cos = []
    for _, r in M.iterrows():
        fn = f'{gn.TEN}/{r.sta}.tenv3'
        if not os.path.exists(fn):
            continue
        df = gn.load_tenv3(r.sta)
        df = df[(df.mjd >= day0) & (df.mjd < day1)]
        if len(df) < 700:
            continue
        sig = df[['se', 'sn', 'su']].values
        df = df[(sig < 4 * np.nanmedian(sig, axis=0)).all(axis=1)]
        mjd = df.mjd.values.astype(float)
        st = gn.steps_for(r.sta)
        sm = list(gn.usable_steps(mjd, st.mjd.values.astype(float)))
        logs, vb, qsteps = [], [], []
        for k, m in Q.items():
            if mjd.max() > m + 60 and mjd.min() < m - 60:
                logs.append((m, 30.0)); vb.append(m)
                if not any(abs(m - s) < 3 for s in sm):
                    sm.append(m)
        sm = sorted(sm)
        G, names = gn.design(mjd, sm, logs, vb, seasonal=False)
        idx = (mjd - day0).astype(int)
        for c in 'enu':
            y = df[c].values * 1e3
            x, rr, ok = gn.robust_fit(G, y)
            rr = rr.copy(); rr[~ok] = np.nan
            v = np.full(len(mjd_all), np.nan); v[idx] = rr          # residual incl. seasonal (non-tectonic part)
            comp[c][r.sta] = v
            # postseismic (log terms) + velocity breaks, evaluated on all days
            Gall, _ = gn.design(mjd_all.astype(float), sm, logs, vb, seasonal=False)
            ip = [i for i, nm in enumerate(names) if nm.startswith('log_') or nm.startswith('dv_')]
            post[c][r.sta] = Gall[:, ip] @ x[ip]
            for k, m in Q.items():
                nm = f'step_{int(m)}'
                if nm in names:
                    cos.append(dict(sta=r.sta, quake=k, comp=c, off_mm=x[names.index(nm)]))
        print(r.sta, len(df), flush=True)
    stas = list(comp['u'])
    Mg = M.set_index('sta').loc[stas].reset_index()
    t = pd.to_datetime(mjd_all - 0, unit='D', origin=pd.Timestamp('1858-11-17'))
    U = pd.DataFrame({s: comp['u'][s] for s in stas}, index=t)
    E = pd.DataFrame({s: comp['e'][s] for s in stas}, index=t)
    N = pd.DataFrame({s: comp['n'][s] for s in stas}, index=t)
    U.to_parquet(f'{DRV}/gnss_U_resid.parquet'); E.to_parquet(f'{DRV}/gnss_E_resid.parquet'); N.to_parquet(f'{DRV}/gnss_N_resid.parquet')
    PE = pd.DataFrame({s: post['e'][s] for s in stas}, index=t); PN = pd.DataFrame({s: post['n'][s] for s in stas}, index=t)
    PU = pd.DataFrame({s: post['u'][s] for s in stas}, index=t)
    PE.to_parquet(f'{DRV}/gnss_E_post.parquet'); PN.to_parquet(f'{DRV}/gnss_N_post.parquet'); PU.to_parquet(f'{DRV}/gnss_U_post.parquet')
    CO = pd.DataFrame(cos); CO.to_csv(f'{OUT}/gnss_coseismic_offsets.csv', index=False)
    # per seismic station: vertical loading (Gaussian-weighted mean of nearby U, 30 km), areal strain from gradients
    S = seis_stations()
    out = {}
    for _, r in S.iterrows():
        d = hav(r.lat, r.lon, Mg.lat.values, Mg.lon.values)
        x = (Mg.lon.values - r.lon) * KM * np.cos(np.radians(r.lat)); y = (Mg.lat.values - r.lat) * KM
        w = np.exp(-(d / 30.0) ** 2); sel = d < 60
        Uv = U.values[:, sel]; ws = w[sel]
        num = np.nansum(Uv * ws, 1); den = np.nansum(np.isfinite(Uv) * ws, 1)
        u_loc = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
        # strain: weighted LSQ plane fit per day for E and N (needs >= 5 stations); strain = mm/km = 1e-6
        selg = d < 50
        A = np.column_stack([np.ones(selg.sum()), x[selg], y[selg]])
        wg = np.exp(-(d[selg] / 25.0) ** 2)
        Ev, Nv = E.values[:, selg], N.values[:, selg]
        PEv, PNv = PE.values[:, selg], PN.values[:, selg]
        dil = np.full(len(t), np.nan); dilp = np.full(len(t), np.nan)
        # group days by availability pattern for speed
        mask = np.isfinite(Ev) & np.isfinite(Nv)
        keys = {}
        for i in range(len(t)):
            if mask[i].sum() >= 5:
                keys.setdefault(mask[i].tobytes(), []).append(i)
        for kk, ii in keys.items():
            m = np.frombuffer(kk, dtype=bool)
            Aw = A[m] * np.sqrt(wg[m])[:, None]
            pinv = np.linalg.pinv(Aw)
            ii = np.array(ii)
            ce = pinv @ (Ev[ii][:, m] * np.sqrt(wg[m])).T
            cn = pinv @ (Nv[ii][:, m] * np.sqrt(wg[m])).T
            dil[ii] = (ce[1] + cn[2]) * 1e-6 * 1e9     # nanostrain
        ok = np.isfinite(PEv).all(0) & np.isfinite(PNv).all(0)
        if ok.sum() >= 5:
            m = ok
            Aw = A[m] * np.sqrt(wg[m])[:, None]; pinv = np.linalg.pinv(Aw)
            ce = pinv @ (PEv[:, m] * np.sqrt(wg[m])).T; cn = pinv @ (PNv[:, m] * np.sqrt(wg[m])).T
            dilp = (ce[1] + cn[2]) * 1e3
        out[r.ns] = pd.DataFrame({'gnss_u_mm': u_loc, 'gnss_dil_ns': dil, 'gnss_dil_post_ns': dilp}, index=t)
        print(r.ns, 'nU', int((d < 60).sum()), 'nE', int(selg.sum()), 'dil sd', np.nanstd(dil), flush=True)
    pd.concat(out, names=['sta', 't']).to_parquet(f'{DRV}/gnss_station.parquet')
    # regional vertical stack (all near stations): median
    reg = U.median(axis=1, skipna=True).where(U.notna().sum(1) >= 8)
    reg.to_frame('gnss_u_regional_mm').to_parquet(f'{DRV}/gnss_regional.parquet')
    # coseismic GNSS strain at seismic stations (gradient of offsets) vs Okada
    rows = []
    for k in Q:
        c = CO[CO.quake == k].pivot_table(index='sta', columns='comp', values='off_mm')
        if 'e' not in c:
            continue
        c = c.join(Mg.set_index('sta')[['lat', 'lon']], how='inner')
        for _, r in S.iterrows():
            d = hav(r.lat, r.lon, c.lat.values, c.lon.values); sel = d < 50
            if sel.sum() < 5:
                continue
            x = (c.lon.values[sel] - r.lon) * KM * np.cos(np.radians(r.lat)); y = (c.lat.values[sel] - r.lat) * KM
            A = np.column_stack([np.ones(sel.sum()), x, y]); w = np.sqrt(np.exp(-(d[sel] / 25) ** 2))
            ce = np.linalg.lstsq(A * w[:, None], c.e.values[sel] * w, rcond=None)[0]
            cn = np.linalg.lstsq(A * w[:, None], c.n.values[sel] * w, rcond=None)[0]
            rows.append(dict(sta=r.ns, quake=k, gnss_dil_ns=(ce[1] + cn[2]) * 1e3, n=int(sel.sum())))
    pd.DataFrame(rows).to_csv(f'{OUT}/gnss_coseismic_dilatation.csv', index=False)


# ------------------------------------------------------------------------------------------------ borehole strain
def stage_bstrain():
    import sci_strain_stack as ss
    out = {}
    for sta in ss.STATIONS:
        ch = ss.load_station(sta)
        t = pd.to_datetime(ss.GRID0 + np.arange(ss.NGRID) * ss.DT, unit='s')
        D = pd.DataFrame({k: v for k, v in ch.items()}, index=t).resample('D').median()
        n = pd.DataFrame({k: np.isfinite(v) for k, v in ch.items()}, index=t).resample('D').sum()
        for c in ['RS1', 'RS2', 'RS3', 'RS4']:
            D.loc[n[c] < 100, c] = np.nan
            x = D[c].values.copy()
            # destep: anomalous day-to-day differences (resets) removed
            x2, nfix, _ = ss.destep(x, np.zeros(len(x), bool), thr_abs=20.0, k=10.0)
            D[c] = x2
        V = D[['RS1', 'RS2', 'RS3', 'RS4']].values.T
        A = ss.gauge_matrix(sta)
        areal = np.full(len(D), np.nan)
        for i in range(len(D)):
            ok = np.isfinite(V[:, i])
            if ok.sum() >= 3:
                areal[i] = (np.linalg.pinv(A[ok]) @ V[ok, i])[0]
        s = pd.Series(areal, index=D.index)
        # remove long-term drift: cubic spline-like smooth trend with ~3-yr knots (piecewise-linear basis + log/exp)
        tt = (s.index - s.index[0]).days.values.astype(float)
        ok = np.isfinite(s.values)
        knots = np.arange(0, tt.max(), 3 * 365.25)
        B = np.column_stack([np.ones_like(tt), tt] + [np.clip(tt - k, 0, None) for k in knots[1:]] +
                            [np.log1p(tt / 30.0), 1 - np.exp(-tt / 300.0)])
        # baro admittance on the high-passed series
        p = D['RDO'].values if 'RDO' in D else np.full(len(D), np.nan)
        okp = ok & np.isfinite(p)
        G = np.column_stack([B, np.where(np.isfinite(p), p - np.nanmean(p), 0.0)])
        x, *_ = np.linalg.lstsq(G[okp], s.values[okp], rcond=None)
        res = s.values - B @ x[:-1] - x[-1] * np.where(np.isfinite(p), p - np.nanmean(p), np.nan)
        # robust clip of wild days
        med = np.nanmedian(res); mad = 1.4826 * np.nanmedian(np.abs(res - med))
        res[np.abs(res - med) > 8 * mad] = np.nan
        # spike removal on day-to-day differences, then high-pass (minus 731-d running median): periods <~1.5 yr kept
        r = pd.Series(res, index=D.index)
        dd = r.diff(); md = 1.4826 * np.nanmedian(np.abs(dd - np.nanmedian(dd)))
        r[(dd.abs() > 10 * md) | (dd.abs().shift(-1) > 10 * md)] = np.nan
        out[f'PB.{sta}'] = r
        out[f'PB.{sta}_hp'] = r - r.rolling(731, center=True, min_periods=200).median()
        print(sta, 'baro admittance nstrain/hPa', round(x[-1], 2), 'resid sd', round(np.nanstd(res), 1), 'n', int(np.isfinite(res).sum()), flush=True)
    pd.DataFrame(out).to_parquet(f'{DRV}/bstrain_areal.parquet')


# ------------------------------------------------------------------------------------------------ power
def stage_power():
    P = pd.read_csv(f'{ROOT}/data/sci/noise_dvv_sc_full/power_B045_EHZ.csv', parse_dates=['day']).set_index('day')
    P.to_parquet(f'{DRV}/power.parquet')
    print(P.describe())


if __name__ == '__main__':
    for s in sys.argv[1:]:
        print('== stage', s, flush=True)
        globals()[f'stage_{s}']()
