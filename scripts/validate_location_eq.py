#!/usr/bin/env python3
"""Validate the travel-time grids + grid-search locator on NCSN earthquakes inside the
location sub-volume: fetch NCSN picks (QuakeML with arrivals), relocate with our grids
(P and S, L1, free origin) and compare with the NCSN catalog hypocentres.
Also returns mean station residuals (candidate station statics).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from locate_family import locate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='3d')
    ap.add_argument('--events', default='data/validation/ncsn_events_box.txt')
    ap.add_argument('--s-scale', type=float, default=1.0)
    a = ap.parse_args()
    g = json.load(open(f'data/velocity/tt_{a.model}/grid.json'))
    gx, gy, gz = map(np.array, (g['sub_x'], g['sub_y'], g['sub_z']))
    tr = Transformer.from_crs('EPSG:4326', g['proj'], always_xy=True)
    ev = pd.read_csv(a.events, sep='|', comment=None)
    ev.columns = [c.strip().lstrip('#') for c in ev.columns]
    cl = Client('NCEDC', timeout=120)
    cache = 'data/validation/quakeml'
    os.makedirs(cache, exist_ok=True)
    TT = {}
    rows, resid = [], []
    for r in ev.itertuples():
        f = os.path.join(cache, f'{r.EventID}.xml')
        if not os.path.exists(f):
            try:
                cat = cl.get_events(eventid=str(r.EventID), includearrivals=True)
                cat.write(f, format='QUAKEML')
            except Exception as e:
                print('skip', r.EventID, repr(e)[:80])
                continue
        from obspy import read_events
        e = read_events(f)[0]
        o = e.preferred_origin() or e.origins[0]
        obs = []
        for p in e.picks:
            ph = (p.phase_hint or '').upper()[:1]
            if ph not in 'PS':
                continue
            sta = f'{p.waveform_id.network_code}.{p.waveform_id.station_code}'
            fn = f'data/velocity/tt_{a.model}/{sta}.npz'
            if not os.path.exists(fn):
                continue
            if sta not in TT:
                z = np.load(fn)
                TT[sta] = {'P': z['tp'], 'S': z['ts']}
            key = (ph, sta)
            if any(k == key for k, _, _ in obs):
                continue
            sig = 0.05 if ph == 'P' else 0.10
            obs.append((key, float(p.time - o.time), sig))
        if len(obs) < 6 or sum(1 for k, _, _ in obs if k[0] == 'S') < 1:
            continue
        TTd = {(ph, s): (TT[s][ph] * (a.s_scale if ph == 'S' else 1.0)) for (ph, s), _, _ in obs}
        best, mis, t0, boots, res = locate(obs, TTd, nboot=50)
        cx, cy = tr.transform(o.longitude, o.latitude)
        bx, by, bz = gx[best[0]], gy[best[1]], gz[best[2]]
        edge = (best[0] in (0, len(gx) - 1)) or (best[1] in (0, len(gy) - 1)) or (best[2] in (0, len(gz) - 1))
        rows.append(dict(id=r.EventID, mag=r.Magnitude, cat_depth=o.depth / 1000, our_depth=bz,
                         dh_km=float(np.hypot(bx - cx, by - cy)), dz_km=float(bz - o.depth / 1000),
                         dt0=float(t0), n_P=sum(k[0] == 'P' for k, _, _ in obs),
                         n_S=sum(k[0] == 'S' for k, _, _ in obs), l1=float(mis[best]), on_edge=edge))
        for (k, _, _), rr in zip(obs, res):
            resid.append(dict(id=r.EventID, phase=k[0], sta=k[1], res=float(rr)))
    df = pd.DataFrame(rows)
    df.to_csv(f'data/validation/eq_relocation_{a.model}.csv', index=False)
    rs = pd.DataFrame(resid)
    st = rs.groupby(['phase', 'sta']).res.agg(['median', 'count']).reset_index()
    st.to_csv(f'data/validation/station_residuals_{a.model}.csv', index=False)
    ok = df[~df.on_edge]
    print(f'{a.model}: {len(df)} events located ({len(ok)} not on grid edge)')
    print(ok[['dh_km', 'dz_km', 'dt0', 'l1', 'n_P', 'n_S']].describe().round(2).to_string())
    print('\nmedian station residuals (count>=5):')
    print(st[st['count'] >= 5].sort_values(['phase', 'median']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
