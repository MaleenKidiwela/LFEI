#!/usr/bin/env python3
"""Measured PGV at the dv/v stations for the 3 mainshocks (NCEDC FDSN), plus BA08-type GMPE PGV.

For each quake: one bulk request (origin-30 s .. origin+300 s) of HN?/HH? (BK, NC) and EH? (PB) for
the stations in data/dvv/pool/params_by_channel.csv; instrument response removed to velocity
(pre_filt 0.05-0.1 .. 20-25 Hz); clip check on raw counts; PGV = geometric mean of the two
horizontal peak |v| (and max-horizontal), cm/s. Accelerometer (HN) preferred when present and
unclipped; HH/EH used otherwise (flagged).
GMPE: Boore & Atkinson (2008) PGV, strike-slip, Vs30=760 (no site term), distance = sqrt(Repi^2+depth^2)
used in place of sqrt(Rjb^2+h^2) (point source; deep slab events).
Output: data/sci/dvv_depth/pgv_by_station.csv
"""
import os
import time

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

OUT = 'data/sci/dvv_depth'
QUAKES = {  # verified in data/catalogs/comcat_M5.5_200km_2008_2026.csv (ComCat nc ids)
    'pet': dict(t='2021-12-20T20:10:31.31', lat=40.3902, lon=-124.2980, dep=27.0, M=6.2),
    'fer': dict(t='2022-12-20T10:34:24.77', lat=40.5250, lon=-124.4230, dep=17.9, M=6.4),
    'm7': dict(t='2024-12-05T18:44:21.11', lat=40.3740, lon=-125.0217, dep=10.0, M=7.0),
}


def ba08_pgv(M, R, mech='SS'):
    """Boore & Atkinson (2008) PGV (cm/s), Vs30 = 760 m/s (reference, no site term)."""
    e1, e2, e3, e4, e5, e6, e7, Mh = 5.00121, 5.04727, 4.63188, 5.08210, 0.18322, -0.12736, 0.0, 8.5
    c1, c2, c3, h, Mref, Rref = -0.87370, 0.10060, -0.00334, 2.54, 4.5, 1.0
    eM = {'U': e1, 'SS': e2, 'N': e3, 'R': e4}[mech]
    FM = eM + e5 * (M - Mh) + e6 * (M - Mh) ** 2 if M <= Mh else eM + e7 * (M - Mh)
    Rr = np.sqrt(R ** 2 + h ** 2)
    FD = (c1 + c2 * (M - Mref)) * np.log(Rr / Rref) + c3 * (Rr - Rref)
    return np.exp(FM + FD)


def main():
    os.makedirs(OUT, exist_ok=True)
    par = pd.read_csv('data/dvv/pool/params_by_channel.csv')
    stas = sorted(par.sta.unique())
    st = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv')
    st['sid'] = st.net + '.' + st.sta
    coords = st.groupby('sid')[['lat', 'lon', 'elev', 'depth']].first()
    cl = Client('NCEDC', timeout=120)
    rows = []
    for q, Q in QUAKES.items():
        t0 = UTCDateTime(Q['t'])
        bulk = []
        for s in stas:
            n, sta = s.split('.')
            for cha in (['EH?'] if n == 'PB' else ['HN?', 'HH?']):
                bulk.append((n, sta, '*', cha, t0 - 30, t0 + 300))
        try:
            raw = cl.get_waveforms_bulk(bulk, attach_response=True)
        except Exception as e:  # noqa: BLE001
            print(q, 'bulk failed', e, flush=True)
            continue
        time.sleep(2)
        raw.merge(method=1, fill_value=0)
        print(q, len(raw), 'traces', flush=True)
        for s in stas:
            n, sta = s.split('.')
            la, lo = coords.loc[s, 'lat'], coords.loc[s, 'lon']
            repi = gps2dist_azimuth(Q['lat'], Q['lon'], la, lo)[0] / 1e3
            R = np.hypot(repi, Q['dep'])
            r = dict(quake=q, sta=s, repi_km=repi, rhyp_km=R, pgv_gmpe_cms=ba08_pgv(Q['M'], R))
            best = None
            for band in (['EH'] if n == 'PB' else ['HN', 'HH']):
                tr3 = raw.select(network=n, station=sta, channel=band + '?')
                if len(tr3) == 0:
                    continue
                locs = sorted({t.stats.location for t in tr3})
                tr3 = tr3.select(location=locs[0])
                hz = {}
                clipped = False
                for tr in tr3:
                    d = tr.data.astype(float)
                    amax = np.abs(d).max()
                    nmax = np.sum(np.abs(d) >= 0.999 * amax)
                    if amax >= 0.98 * 2 ** 23 or (nmax >= 5 and amax > 1e5):
                        clipped = True
                    t = tr.copy()
                    try:
                        t.detrend('demean'); t.taper(0.05)          # noqa: E702
                        nyq = t.stats.sampling_rate / 2
                        t.remove_response(output='VEL', pre_filt=(0.05, 0.1, 0.8 * nyq * 0.5, nyq * 0.5),
                                          water_level=None if band == 'HN' else 60)
                        t.trim(t0 - 5, t0 + 290)
                    except Exception as e:  # noqa: BLE001
                        print('  resp fail', tr.id, e)
                        continue
                    hz[tr.stats.channel[-1]] = np.abs(t.data).max() * 100.0     # cm/s
                hor = [hz[c] for c in hz if c not in 'Z']
                if len(hor) < 2:
                    continue
                cand = dict(band=band, clipped=clipped, pgv_gm_cms=float(np.sqrt(hor[0] * hor[1])),
                            pgv_maxh_cms=float(max(hor)), pgv_z_cms=float(hz.get('Z', np.nan)))
                r[f'pgv_{band}_cms'] = cand['pgv_gm_cms']
                r[f'clip_{band}'] = clipped
                if best is None and not clipped:
                    best = cand
            if best is None and any(k.startswith('pgv_') and k.endswith('_cms') and k != 'pgv_gmpe_cms' for k in r):
                best = dict(band='clipped', clipped=True, pgv_gm_cms=np.nan, pgv_maxh_cms=np.nan, pgv_z_cms=np.nan)
            if best:
                r.update(best)
            rows.append(r)
            print(f"  {q} {s:8s} R={R:6.1f} km gmpe {r['pgv_gmpe_cms']:.2f}  obs {r.get('pgv_gm_cms', np.nan):.3f} "
                  f"({r.get('band', '-')})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f'{OUT}/pgv_by_station.csv', index=False)


if __name__ == '__main__':
    main()
