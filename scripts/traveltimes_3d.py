#!/usr/bin/env python3
"""P and S travel-time grids from every candidate station, by eikonal (PyKonal) in the
Furlong et al. (2024) 3D model (Benz 2024, doi 10.5066/P14QGYIY) or its 1D starting model.

Coordinates: transverse Mercator (lon_0=-122.7, lat_0=40.4, km) = the model's own frame.
Depth z: km below sea level (negative above). Station z = -elev + sensor depth (boreholes).
Reciprocity: source at the station; the saved sub-volume around the family is the
station->hypocentre travel time for any trial hypocentre.

Output: data/velocity/tt_{model}/{NET.STA}.npz with tp, ts (float32, sub-volume) and grid
axes; plus grid.json.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pykonal
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

TM = "+proj=tmerc +lon_0=-122.7 +lat_0=40.4 +k=1 +units=km +ellps=WGS84"
MODEL_DIR = 'data/velocity/furlong2024'
VPVS_1D = 1.78


def load_model(kind):
    tr = Transformer.from_crs('EPSG:4326', TM, always_xy=True)
    out = {}
    for ph, fn in [('p', 'pm.llzv.gz'), ('s', 'sm.llzv.gz')]:
        d = pd.read_csv(os.path.join(MODEL_DIR, fn), sep=r'\s+', header=None,
                        names=['lon', 'lat', 'z', 'v'])
        x, y = tr.transform(d.lon.values, d.lat.values)
        d['x'], d['y'] = np.round(x, 1), np.round(y, 1)
        xs, ys, zs = np.unique(d.x), np.unique(d.y), np.unique(d.z)
        cube = d.set_index(['x', 'y', 'z']).v.unstack('z').reindex(
            pd.MultiIndex.from_product([xs, ys])).values.reshape(len(xs), len(ys), len(zs))
        out[ph] = (xs, ys, zs, cube)
    if kind == '1d':
        lay = np.loadtxt(os.path.join(MODEL_DIR, 'vel_p.1d'))
        xs, ys, zs, _ = out['p']
        vp = np.interp(zs, lay[:, 0], lay[:, 1])      # step model sampled at node depths
        for ph, scale in [('p', 1.0), ('s', 1 / VPVS_1D)]:
            out[ph] = (xs, ys, zs, np.broadcast_to(vp * scale, (len(xs), len(ys), len(zs))).copy())
    return out


def solve_one(args):
    sta, sx, sy, sz, grid, vel, sub, outdir = args
    res = {}
    for ph in ['p', 's']:
        solver = pykonal.solver.PointSourceSolver(coord_sys='cartesian')
        solver.velocity.min_coords = grid['x0'], grid['y0'], grid['z0']
        solver.velocity.node_intervals = grid['d'], grid['d'], grid['d']
        solver.velocity.npts = grid['nx'], grid['ny'], grid['nz']
        solver.velocity.values = vel[ph]
        solver.src_loc = np.array([sx, sy, sz])
        solver.solve()
        tt = solver.tt.values
        res['t' + ph] = tt[sub[0]:sub[1], sub[2]:sub[3], sub[4]:sub[5]].astype('float32')
    np.savez_compressed(os.path.join(outdir, f'{sta}.npz'), **res)
    return sta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['3d', '1d'], default='3d')
    ap.add_argument('--lat', type=float, default=40.053434)      # family 23 (Shelly)
    ap.add_argument('--lon', type=float, default=-123.556055)
    ap.add_argument('--half', type=float, default=80.0, help='half width of solver grid, km')
    ap.add_argument('--d', type=float, default=0.5, help='grid spacing, km')
    ap.add_argument('--zmin', type=float, default=-2.0)
    ap.add_argument('--zmax', type=float, default=45.0)
    ap.add_argument('--sub-half', type=float, default=15.0, help='saved sub-volume half width, km')
    ap.add_argument('--sub-z', type=float, nargs=2, default=[12.0, 42.0])
    ap.add_argument('--stations', default='data/stations/candidate_channels_65km_2008_2026.csv')
    ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args()

    tr = Transformer.from_crs('EPSG:4326', TM, always_xy=True)
    fx, fy = tr.transform(a.lon, a.lat)
    d = a.d
    x0, y0 = np.floor((fx - a.half) / d) * d, np.floor((fy - a.half) / d) * d
    nx = ny = int(round(2 * a.half / d)) + 1
    z0 = a.zmin
    nz = int(round((a.zmax - a.zmin) / d)) + 1
    gx, gy, gz = x0 + d * np.arange(nx), y0 + d * np.arange(ny), z0 + d * np.arange(nz)

    model = load_model(a.model)
    vel = {}
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel(), np.clip(Z.ravel(), -3.75, 70.75)]
    for ph in ['p', 's']:
        xs, ys, zs, cube = model[ph]
        f = RegularGridInterpolator((xs, ys, zs), cube, bounds_error=False, fill_value=None)
        vel[ph] = f(pts).reshape(nx, ny, nz)
        assert np.isfinite(vel[ph]).all() and vel[ph].min() > 0.5, f'bad {ph} velocities'

    ix = lambda v, g: int(np.argmin(np.abs(g - v)))
    sub = (ix(fx - a.sub_half, gx), ix(fx + a.sub_half, gx) + 1,
           ix(fy - a.sub_half, gy), ix(fy + a.sub_half, gy) + 1,
           ix(a.sub_z[0], gz), ix(a.sub_z[1], gz) + 1)
    outdir = f'data/velocity/tt_{a.model}'
    os.makedirs(outdir, exist_ok=True)
    grid = dict(x0=float(x0), y0=float(y0), z0=float(z0), d=d, nx=nx, ny=ny, nz=nz,
                sub=list(map(int, sub)),
                sub_x=gx[sub[0]:sub[1]].tolist(), sub_y=gy[sub[2]:sub[3]].tolist(),
                sub_z=gz[sub[4]:sub[5]].tolist(), proj=TM, model=a.model,
                family_xy=[fx, fy])
    json.dump(grid, open(os.path.join(outdir, 'grid.json'), 'w'))

    st = pd.read_csv(a.stations).groupby(['net', 'sta']).agg(
        lat=('lat', 'first'), lon=('lon', 'first'), elev=('elev', 'first'),
        depth=('depth', 'max')).reset_index()
    jobs = []
    for _, r in st.iterrows():
        sx, sy = tr.transform(r.lon, r.lat)
        sz = -r.elev / 1000.0 + r.depth / 1000.0
        jobs.append((f'{r.net}.{r.sta}', sx, sy, sz, grid, vel, sub, outdir))
    with Pool(a.procs) as p:
        for s in p.imap_unordered(solve_one, jobs):
            print('done', s, flush=True)


if __name__ == '__main__':
    main()
