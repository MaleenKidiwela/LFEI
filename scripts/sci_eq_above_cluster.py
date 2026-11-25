#!/usr/bin/env python3
"""EQ-above step 1b: load event waveforms, predict P/S, pairwise waveform similarity, cluster into families.

load  : data/sci/eq_above/wf/{evid}.mseed -> 2-8 Hz (4-pole zero-phase), cubic-spline resampled to 40 Hz on the
        common grid origin-5 .. origin+45 s (exact sample times; PB sub-sample offsets honoured)
        -> wf40.npy (event x station x comp[Z,1/N,2/E] x 2000), events.csv
tt    : P, S predicted from the 3D model tables data/velocity/tt_3d/{sta}.npz (sub-grid, trilinear) at the catalog
        hypocentre -> ttP.npy, ttS.npy (s after origin)
pairs : catalog neighbours (hypocentral distance <= 6 km) -> per station CC (mean of components with data in both)
        in [P-1, S+4] s, lag +-1.5 s (predicted moveout difference removed first); network CC = mean of the best
        min(6, n) stations with |sta CC| finite, n >= 3 -> pairs.csv
clust : average-linkage clustering of 1-netCC (missing = 0 similarity), cut at 1-CC_CUT -> families.csv
Usage: nice -n 10 python scripts/sci_eq_above_cluster.py --stage load|tt|pairs|clust [--procs 4]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/eq_above'
BIG = '/data/wsd02/maleen_data/eq_above'          # bulk waveforms (kept off /home)
sys.path.insert(0, f'{ROOT}/scripts')
from sci_eq_above_dl import CH, select  # noqa: E402

STAS = [f'{n}.{s}' for n, s, _, _ in CH]
SR = 40.0
PRE = 5.0
NS = 2000
FMIN, FMAX = 2.0, 8.0
CC_CUT = 0.55


def load_one(args):
    evid, t = args
    from obspy import read, UTCDateTime
    A = np.full((len(STAS), 3, NS), np.nan, np.float32)
    f = f'{BIG}/wf/{evid}.mseed'
    if not os.path.exists(f):
        return A
    try:
        st = read(f)
    except Exception:                                  # noqa: BLE001
        return A
    t0 = UTCDateTime(str(t)) - PRE
    tg = np.arange(NS) / SR
    st.merge(method=1, fill_value=None)
    for tr in st:
        sta = f'{tr.stats.network}.{tr.stats.station}'
        if sta not in STAS:
            continue
        ci = {'Z': 0, 'N': 1, '1': 1, 'E': 2, '2': 2}.get(tr.stats.channel[-1])
        if ci is None:
            continue
        x = tr.data
        if np.ma.isMaskedArray(x):
            if x.mask.any():
                continue
            x = x.data
        x = x.astype(float)
        sr = tr.stats.sampling_rate
        if len(x) < 0.9 * 50 * sr or np.std(x) == 0 or sr < 20:
            continue
        x = x - np.mean(x)
        w = np.ones(len(x))
        nt = int(0.5 * sr)
        w[:nt] = np.hanning(2 * nt)[:nt]
        w[-nt:] = np.hanning(2 * nt)[nt:]
        sos = butter(4, [FMIN, min(FMAX, 0.45 * sr)], btype='band', fs=sr, output='sos')
        y = sosfiltfilt(sos, x * w)
        ts = (tr.stats.starttime - t0) + np.arange(len(y)) / sr
        ok = (tg >= ts[0]) & (tg <= ts[-1])
        if ok.sum() < 0.9 * NS:
            continue
        v = np.full(NS, np.nan)
        v[ok] = CubicSpline(ts, y)(tg[ok])
        if np.isfinite(A[STAS.index(sta), ci, 0]):
            continue                                    # keep first location code
        A[STAS.index(sta), ci] = np.nan_to_num(v).astype(np.float32)
    return A


def stage_load(a):
    c = select()
    with Pool(a.procs) as p:
        W = np.array(p.map(load_one, list(zip(c.EventID.astype(str), c.Time)), chunksize=8))
    np.save(f'{BIG}/wf40.npy', W)
    c.to_csv(f'{OUT}/events.csv', index=False)
    print(W.shape, 'events with data per station:', dict(zip(STAS, np.isfinite(W[:, :, 0, 100]).sum(0))))


# ----------------------------------------------------------------------------------------------- travel times
_TT = {}


def tt_grid():
    g = json.load(open(f'{ROOT}/data/velocity/tt_3d/grid.json'))
    return g


def proj(lat, lon):
    from pyproj import Proj
    p = Proj(tt_grid()['proj'])
    return p(np.asarray(lon), np.asarray(lat))


def tt_at(sta, x, y, z, phase):
    """trilinear interpolation of the 3D tables (sub grid) at x, y (km, model projection), z (km)."""
    key = sta
    if key not in _TT:
        f = f'{ROOT}/data/velocity/tt_3d/{sta}.npz'
        if not os.path.exists(f):
            _TT[key] = None
        else:
            z_ = np.load(f)
            _TT[key] = {'P': z_['tp'], 'S': z_['ts']}
    if _TT[key] is None:
        return np.full(np.shape(x), np.nan)
    from scipy.interpolate import RegularGridInterpolator
    g = tt_grid()
    G = _TT[key][phase]
    fn = RegularGridInterpolator((np.array(g['sub_x']), np.array(g['sub_y']), np.array(g['sub_z'])), G,
                                 bounds_error=False, fill_value=np.nan)
    return fn(np.c_[np.ravel(x), np.ravel(y), np.ravel(z)]).reshape(np.shape(x))


def stage_tt(a):
    c = pd.read_csv(f'{OUT}/events.csv')
    x, y = proj(c.Latitude.values, c.Longitude.values)
    z = np.clip(c.depth.values, 12.0, 42.0)
    P = np.full((len(c), len(STAS)), np.nan)
    S = np.full((len(c), len(STAS)), np.nan)
    for j, s in enumerate(STAS):
        P[:, j] = tt_at(s, x, y, z, 'P')
        S[:, j] = tt_at(s, x, y, z, 'S')
    # fallback outside the 3D sub-grid: straight rays, Vp 6.2 km/s, Vp/Vs 1.78 (windowing only)
    sc = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv').groupby(['net', 'sta']).first()
    for j, s in enumerate(STAS):
        n_, s_ = s.split('.')
        if (n_, s_) not in sc.index:
            continue
        la, lo = sc.loc[(n_, s_), ['lat', 'lon']]
        r = np.hypot(np.hypot((c.Latitude.values - la) * 111.2, (c.Longitude.values - lo) * 111.2 * np.cos(np.radians(la))),
                     c.depth.values)
        bad = ~np.isfinite(P[:, j])
        P[bad, j] = r[bad] / 6.2
        S[bad, j] = r[bad] / 6.2 * 1.78
    np.save(f'{OUT}/ttP.npy', P)
    np.save(f'{OUT}/ttS.npy', S)
    print('events outside grid:', np.isnan(P[:, STAS.index('NC.KMR')]).sum())


# ----------------------------------------------------------------------------------------------- pairs
_G = {}


def ccmax(a, b, ml):
    """max normalised CC of b vs a over lags +-ml samples (a, b same length, b extracted with margin)."""
    from scipy.signal import fftconvolve
    n = len(a) - 2 * ml
    r = a[ml:ml + n] - a[ml:ml + n].mean()
    rn = np.linalg.norm(r) + 1e-20
    num = fftconvolve(b, r[::-1], mode='valid')
    c1 = np.concatenate([[0.0], np.cumsum(b)])
    c2 = np.concatenate([[0.0], np.cumsum(b * b)])
    S1 = c1[n:] - c1[:-n]
    S2 = c2[n:] - c2[:-n]
    v = num / (rn * np.sqrt(np.clip(S2 - S1 ** 2 / n, 1e-30, None)))
    k = int(np.argmax(v))
    return float(v[k]), k - ml


def pair_job(ij):
    i, j = ij
    W, P, S = _G['W'], _G['P'], _G['S']
    ml = int(1.5 * SR)
    out = np.full(len(STAS), np.nan)
    lag = np.full(len(STAS), np.nan)
    for k in range(len(STAS)):
        if not (np.isfinite(P[i, k]) and np.isfinite(P[j, k])):
            continue
        i0 = int((PRE + P[i, k] - 1.0) * SR)
        i1 = int((PRE + S[i, k] + 4.0) * SR)
        sh = int(round((P[j, k] - P[i, k]) * SR))
        if i0 - ml + sh < 0 or i1 + ml + sh > NS or i0 - ml < 0 or i1 + ml > NS:
            continue
        ccs, lgs = [], []
        for ci in range(3):
            a = W[i, k, ci, i0 - ml:i1 + ml].astype(float)
            b = W[j, k, ci, i0 - ml + sh:i1 + ml + sh].astype(float)
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and a.std() > 0 and b.std() > 0):
                continue
            cc, lg = ccmax(a, b, ml)
            ccs.append(cc)
            lgs.append(lg)
        if ccs:
            out[k] = np.mean(ccs)
            lag[k] = np.median(lgs) / SR
    return i, j, out, lag


def stage_pairs(a):
    c = pd.read_csv(f'{OUT}/events.csv')
    W = np.load(f'{BIG}/wf40.npy', mmap_mode='r')
    _G['W'] = np.asarray(W)
    _G['P'] = np.load(f'{OUT}/ttP.npy')
    _G['S'] = np.load(f'{OUT}/ttS.npy')
    x, y = proj(c.Latitude.values, c.Longitude.values)
    X = np.c_[x, y, c.depth.values]
    from scipy.spatial import cKDTree
    pr = cKDTree(X).query_pairs(6.0, output_type='ndarray')
    print(len(pr), 'pairs', flush=True)
    with Pool(a.procs) as p:
        res = p.map(pair_job, [tuple(r) for r in pr], chunksize=50)
    rows = []
    for i, j, cc, lg in res:
        v = cc[np.isfinite(cc)]
        n = len(v)
        net = np.mean(np.sort(v)[::-1][:6]) if n >= 3 else np.nan
        rows.append(dict(i=i, j=j, n=n, netcc=net, **{f'cc_{s}': cc[k] for k, s in enumerate(STAS)}))
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/pairs.csv', index=False)
    print(d.netcc.describe())


def stage_clust(a):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    c = pd.read_csv(f'{OUT}/events.csv')
    d = pd.read_csv(f'{OUT}/pairs.csv')
    n = len(c)
    D = np.ones((n, n))
    ok = np.isfinite(d.netcc)
    D[d.i[ok], d.j[ok]] = 1 - d.netcc[ok]
    D[d.j[ok], d.i[ok]] = 1 - d.netcc[ok]
    np.fill_diagonal(D, 0)
    D = np.clip(D, 0, 2)
    Z = linkage(squareform(D, checks=False), 'average')
    lab = fcluster(Z, 1 - a.cut, 'distance')
    c['fam_raw'] = lab
    cnt = c.fam_raw.value_counts()
    big = cnt[cnt >= 5].index
    rank = {f: k + 1 for k, f in enumerate(big)}
    c['efam'] = c.fam_raw.map(rank).fillna(0).astype(int)
    c.to_csv(f'{OUT}/families.csv', index=False)
    g = c[c.efam > 0].groupby('efam').agg(n=('EventID', 'size'), lat=('Latitude', 'median'), lon=('Longitude', 'median'),
                                          z=('depth', 'median'), zsd=('depth', 'std'), M=('Magnitude', 'median'),
                                          t0=('Time', 'min'), t1=('Time', 'max'), r=('r', 'median'))
    print(g.to_string())
    print('in families:', (c.efam > 0).sum(), 'of', n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--cut', type=float, default=CC_CUT)
    a = ap.parse_args()
    {'load': stage_load, 'tt': stage_tt, 'pairs': stage_pairs, 'clust': stage_clust}[a.stage](a)


if __name__ == '__main__':
    main()
