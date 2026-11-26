#!/usr/bin/env python
"""Download public environmental + catalogue data for the dv/v attribution study.

Raw files go to /data/wsd02/maleen_data/dvv_attr/raw/<source>/ (treated as untrusted data; parsed as text only).
Stages: ghcn, openmeteo, usgs, coops, comcat, all.  Generic user agent; no personal information sent.
"""
import os, sys, time, io, json
import numpy as np
import pandas as pd
import requests

ROOT = '/home/seismic/LFE'
RAW = '/data/wsd02/maleen_data/dvv_attr/raw'
UA = {'User-Agent': 'Mozilla/5.0 (research-client)'}
BOX = dict(lat0=39.3, lat1=41.0, lon0=-124.5, lon1=-122.8)
T0, T1 = '2007-01-01', '2026-11-24'
SITES = ['BK.BJES', 'BK.BLAS', 'BK.BRIC', 'BK.ETSL', 'BK.GHOP', 'NC.KCPC', 'PB.B046']


def get(url, fn, tries=4, timeout=180, params=None):
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        return fn
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, params=params)
            if r.status_code == 200:
                open(fn, 'wb').write(r.content)
                return fn
            print('HTTP', r.status_code, url, flush=True)
            if r.status_code in (400, 404):
                return None
        except Exception as e:
            print('ERR', e, url, flush=True)
        time.sleep(10 * (k + 1))
    return None


def seis_stations():
    c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv')
    c['ns'] = c.net + '.' + c.sta
    return c.groupby('ns').agg(lat=('lat', 'median'), lon=('lon', 'median')).reset_index()


def stage_ghcn():
    d = f'{RAW}/ghcn'; os.makedirs(d, exist_ok=True)
    get('https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt', f'{d}/ghcnd-stations.txt')
    get('https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt', f'{d}/ghcnd-inventory.txt')
    inv = pd.read_fwf(f'{d}/ghcnd-inventory.txt', header=None, colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
                      names=['id', 'lat', 'lon', 'el', 'y0', 'y1'])
    sel = inv[(inv.lat.between(BOX['lat0'], BOX['lat1'])) & (inv.lon.between(BOX['lon0'], BOX['lon1'])) &
              inv.el.isin(['PRCP', 'TMAX', 'TMIN', 'TAVG']) & (inv.y1 >= 2012) & (inv.y0 <= 2020)]
    ids = sorted(sel.id.unique())
    print('GHCN candidate stations', len(ids), flush=True)
    sel.to_csv(f'{d}/selected_inventory.csv', index=False)
    for i in ids:
        get(f'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{i}.csv', f'{d}/{i}.csv')
        time.sleep(0.5)


def stage_openmeteo():
    d = f'{RAW}/openmeteo'; os.makedirs(d, exist_ok=True)
    # Open-Meteo archive requests over 20 yr weigh ~500 'calls' each (10k/day limit): only a few representative
    # sites are fetched; other stations use the nearest site (ERA5 0.25 deg / ERA5-Land 0.1 deg fields are smooth).
    S = seis_stations()
    S = S[S.ns.isin(SITES)]
    v1 = 'precipitation_sum,rain_sum,snowfall_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,surface_pressure_mean,et0_fao_evapotranspiration,soil_moisture_0_to_100cm_mean'
    v2 = 'soil_moisture_0_to_7cm_mean,soil_moisture_7_to_28cm_mean,soil_moisture_28_to_100cm_mean,soil_moisture_100_to_255cm_mean,soil_temperature_0_to_100cm_mean,snow_depth_mean'
    for _, r in S.iterrows():
        # 'best_match' mixes model systems over time (inhomogeneous precipitation, code review M1): only the
        # ERA5-Land soil layers are fetched now; precipitation/temperature/pressure come from NASA POWER (MERRA-2).
        for tag, v, extra in (('era5land', v2, {'models': 'era5_land'}),):
            fn = f'{d}/{r.ns}_{tag}.json'
            p = dict(latitude=round(r.lat, 4), longitude=round(r.lon, 4), start_date='2007-01-01', end_date='2026-11-20',
                     daily=v, timezone='GMT', **extra)
            ok = get('https://archive-api.open-meteo.com/v1/archive', fn, params=p)
            print(r.ns, tag, 'ok' if ok else 'FAIL', flush=True)
            time.sleep(75)


def stage_power():
    """NASA POWER daily point data (MERRA-2 based; single frozen reanalysis system), one request per
    0.5 x 0.625 deg cell containing a seismic station.  No key."""
    d = f'{RAW}/power'; os.makedirs(d, exist_ok=True)
    S = seis_stations()
    S['cell'] = [f'{round(la * 2) / 2:.1f}_{round(lo / 0.625) * 0.625:.3f}' for la, lo in zip(S.lat, S.lon)]
    for cell, g in S.groupby('cell'):
        la, lo = g.lat.mean(), g.lon.mean()
        fn = f'{d}/cell_{cell}.json'
        ok = get('https://power.larc.nasa.gov/api/temporal/daily/point', fn,
                 params=dict(parameters='PRECTOTCORR,T2M,PS,GWETROOT,GWETPROF,GWETTOP', community='AG', longitude=round(lo, 4),
                             latitude=round(la, 4), start='20070101', end='20261124', format='JSON'))
        print(cell, list(g.ns), 'ok' if ok else 'FAIL', flush=True)
        time.sleep(2)
    S[['ns', 'lat', 'lon', 'cell']].to_csv(f'{d}/station_cells.csv', index=False)


def stage_usgs():
    d = f'{RAW}/usgs'; os.makedirs(d, exist_ok=True)
    bb = f"{BOX['lon0']},{BOX['lat0']},{BOX['lon1']},{BOX['lat1']}"
    # sites with daily values (continuous recorders) and with any field/periodic gw levels
    get(f'https://waterservices.usgs.gov/nwis/site/?format=rdb&bBox={bb}&siteType=GW&hasDataTypeCd=dv&siteOutput=expanded',
        f'{d}/sites_gw_dv.rdb')
    get(f'https://waterservices.usgs.gov/nwis/site/?format=rdb&bBox={bb}&siteType=GW&hasDataTypeCd=gw&startDT=2008-01-01',
        f'{d}/sites_gw_gw.rdb')
    get(f'https://waterservices.usgs.gov/nwis/site/?format=rdb&bBox={bb}&siteType=GW&hasDataTypeCd=iv&startDT=2008-01-01',
        f'{d}/sites_gw_iv.rdb')
    for kind in ('dv', 'gw', 'iv'):
        fn = f'{d}/sites_gw_{kind}.rdb'
        if not os.path.exists(fn):
            continue
        L = [l for l in open(fn, errors='replace') if not l.startswith('#')]
        if len(L) < 3:
            print(kind, 'no sites'); continue
        df = pd.read_csv(io.StringIO(''.join(L)), sep='\t', dtype=str).iloc[1:]
        print(kind, len(df), 'sites', flush=True)
        df.to_csv(f'{d}/sites_{kind}.csv', index=False)
        for s in df.site_no.unique()[:60]:
            if kind == 'dv':
                get(f'https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={s}&startDT=2007-01-01&endDT=2026-11-24&siteStatus=all',
                    f'{d}/dv_{s}.rdb')
            elif kind == 'gw':
                get(f'https://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&sites={s}&startDT=2007-01-01&endDT=2026-11-24&siteStatus=all',
                    f'{d}/gw_{s}.rdb')
            time.sleep(0.5)


def stage_coops():
    d = f'{RAW}/coops'; os.makedirs(d, exist_ok=True)
    for st in ('9418767', '9419750', '9416841'):
        for y0 in range(2007, 2027, 10):
            y1 = min(y0 + 9, 2026)
            get('https://api.tidesandcurrents.noaa.gov/api/prod/datagetter', f'{d}/{st}_{y0}.csv',
                params=dict(product='monthly_mean', station=st, begin_date=f'{y0}0101', end_date=f'{y1}1231', datum='MSL',
                            units='metric', time_zone='gmt', format='csv', application='research'))


def stage_comcat():
    d = f'{RAW}/comcat'; os.makedirs(d, exist_ok=True)
    get('https://earthquake.usgs.gov/fdsnws/event/1/query', f'{d}/local_M3.5_250km.csv',
        params=dict(format='csv', starttime='2007-06-01', endtime='2026-11-25', latitude=40.1, longitude=-123.8,
                    maxradiuskm=250, minmagnitude=3.5, orderby='time-asc'))
    get('https://earthquake.usgs.gov/fdsnws/event/1/query', f'{d}/local_M2.5_60km.csv',
        params=dict(format='csv', starttime='2007-06-01', endtime='2026-11-25', latitude=40.1, longitude=-123.8,
                    maxradiuskm=60, minmagnitude=2.5, orderby='time-asc'))


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for s in (['comcat', 'coops', 'usgs', 'ghcn', 'openmeteo'] if st == 'all' else [st]):
        print('== stage', s, flush=True)
        globals()[f'stage_{s}']()
