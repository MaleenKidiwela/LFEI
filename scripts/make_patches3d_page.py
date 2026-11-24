#!/usr/bin/env python3
"""Interactive 3D page of the sharpened LFE patches with slab interfaces (one Plotly figure).
Reads data/patches/sharpened/sharpened_patches_geo.csv and the slab models via plot_sharpened_patches.slab_models();
writes report/patches3d/index.html (local E/N km about the mean family centre, depth km)."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from plot_sharpened_patches import centre, kx, ky, lat0, lon0, slab_models  # noqa: E402

P = pd.read_csv('data/patches/sharpened/sharpened_patches_geo.csv')
P['E'] = (P.lon - lon0) * kx
P['N'] = (P.lat - lat0) * ky
fams = sorted(P.family.unique())
C = {F: centre(F) for F in fams}
c0 = np.mean([C[F][:2] for F in fams], axis=0)
P['E'] -= c0[0]; P['N'] -= c0[1]
P = P.sort_values('density')
pat = dict(x=P.E.round(3).tolist(), y=P.N.round(3).tolist(), z=P.depth_km.round(3).tolist(),
           d=P.density.round(3).tolist(), f=P.family.astype(int).tolist())
lab = dict(x=[round(C[F][0] - c0[0], 3) for F in fams], y=[round(C[F][1] - c0[1], 3) for F in fams],
           z=[round(C[F][2], 3) for F in fams], f=[int(F) for F in fams])
S = slab_models()
ge = np.arange(-80, 31, 1.0); gn = np.arange(-36, 46, 1.0)
# map layer at the surface: Natural Earth 10 m coastline, stations used for dv/v, place names
LONW, LONE, LATS, LATN = lon0 + (ge[0] + c0[0]) / kx, lon0 + (ge[-1] + c0[0]) / kx, lat0 + (gn[0] + c0[1]) / ky, lat0 + (gn[-1] + c0[1]) / ky
# Natural Earth 10 m coastline clipped to the region (data/maps/coast_ne10m_mtj.csv, exported from the cartopy cache)
co = pd.read_csv('data/maps/coast_ne10m_mtj.csv')
co = co[co.lon.between(LONW - 0.05, LONE + 0.05) & co.lat.between(LATS - 0.05, LATN + 0.05)]
cx, cy = [], []
for _, g in co.groupby('seg'):
    cx += ((g.lon - lon0) * kx - c0[0]).round(2).tolist() + [None]
    cy += ((g.lat - lat0) * ky - c0[1]).round(2).tolist() + [None]
chans = pd.read_csv('data/dvv/pool/params_by_channel.csv').chan
used = {'.'.join(c.split('.')[:2]) for c in chans}
st = pd.concat([pd.read_csv(f'data/stations/{f}', sep='|', comment='#', header=None, usecols=[0, 1, 2, 3], names=['net', 'sta', 'lat', 'lon'])
                for f in ['sta_ncedc.txt', 'sta_iris.txt']]).drop_duplicates(['net', 'sta'])
st = st[(st.net + '.' + st.sta).isin(used)]
st = st.assign(E=(st.lon - lon0) * kx - c0[0], N=(st.lat - lat0) * ky - c0[1])
st = st[st.E.between(ge[0], ge[-1]) & st.N.between(gn[0], gn[-1])]
stations = dict(x=st.E.round(2).tolist(), y=st.N.round(2).tolist(), name=(st.net + '.' + st.sta).tolist(), bh=(st.net == 'PB').tolist())
places = [('Cape Mendocino', 40.440, -124.409), ('Petrolia', 40.325, -124.287), ('Ferndale', 40.576, -124.264),
          ('Honeydew', 40.243, -124.122), ('Garberville', 40.100, -123.795), ('Shelter Cove', 40.030, -124.073)]
places = dict(x=[round((lo - lon0) * kx - c0[0], 2) for _, la, lo in places], y=[round((la - lat0) * ky - c0[1], 2) for _, la, lo in places],
              name=[n for n, _, _ in places])
# lon/lat tick positions
lonticks = [v for v in np.arange(-124.6, -123.0, 0.2) if LONW <= v <= LONE]
latticks = [v for v in np.arange(39.6, 40.8, 0.1) if LATS <= v <= LATN]
ticks = dict(xv=[round((v - lon0) * kx - c0[0], 2) for v in lonticks], xt=[f'{-v:.1f}°W' for v in lonticks],
             yv=[round((v - lat0) * ky - c0[1], 2) for v in latticks], yt=[f'{v:.1f}°N' for v in latticks])
GE, GN = np.meshgrid(ge, gn)
slabs = {}
for name, fn in S.items():
    Z = fn(GE + c0[0], GN + c0[1])
    slabs[name] = [[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in Z]
data = dict(pat=pat, lab=lab, ge=ge.round(2).tolist(), gn=gn.round(2).tolist(), slabs=slabs, coast=dict(x=cx, y=cy),
            stations=stations, places=places, ticks=ticks,
            lon0=lon0 + c0[0] / kx, lat0=lat0 + c0[1] / ky)
tpl = open('scripts/patches3d_template.html').read()
open('report/patches3d/index.html', 'w').write(tpl.replace('/*DATA*/null', json.dumps(data, separators=(',', ':'))))
print(len(cx), 'coast pts', len(stations['x']), 'stations', len(P), 'pixels', len(fams), 'families', {k: int(np.isfinite(np.array(v, dtype=float)).sum()) for k, v in slabs.items()})
