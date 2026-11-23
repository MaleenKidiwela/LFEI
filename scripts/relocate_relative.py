#!/usr/bin/env python3
"""Relative relocation of a family against its master stacks.

dt_i (arrival in sub-stack/event minus arrival in master, at station-channel i) is measured
by cross-correlation in phase windows around the master picks (S: horizontals, P: vertical),
with parabolic sub-sample interpolation. Linearised model about the master location x0:
    dt_i = grad(T_i)(x0) . dx + dt0
(grad of the station's travel-time field = source-side slowness, by reciprocity; S times
scaled by the chosen s_scale). Weighted least squares (w = CC^2/sigma^2), iteratively
reweighted (Huber), with bootstrap errors.

Modes:
  bins   : equal-count bins of N consecutive events (catalog-aligned stacks, no shifts)
  events : individual events; keep obs with CC >= CCMIN, need >= MINOBS obs and >= MINSTA
           stations, azimuthal gap < 180 deg
Output: data/relocations/fam{F}_{mode}_{model}.csv (+ per-obs dt file for bins).
"""
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import butter, sosfiltfilt

S_WIN = (-0.5, 2.5)     # s around master S pick
P_WIN = (-0.3, 1.2)     # s around master P pick
MAXLAG = 0.25           # s


def cc_lag(master_w, X_w, maxlag):
    """Normalised CC of master window with rows of X (same length, padded by maxlag on both
    sides in X). Returns lag (s.samples, + = X later), CC, with parabolic interpolation."""
    n = len(master_w)
    L = X_w.shape[1] - n          # = 2*maxlag
    ccs = np.empty((X_w.shape[0], L + 1))
    mw = (master_w - master_w.mean()) / (np.linalg.norm(master_w - master_w.mean()) + 1e-20)
    for k in range(L + 1):
        seg = X_w[:, k:k + n]
        seg = seg - seg.mean(1, keepdims=True)
        ccs[:, k] = seg @ mw / (np.linalg.norm(seg, axis=1) + 1e-20)
    j = np.argmax(ccs, axis=1)
    c = ccs[np.arange(len(j)), j]
    jm, jp = np.clip(j - 1, 0, L), np.clip(j + 1, 0, L)
    y0, y1, y2 = ccs[np.arange(len(j)), jm], c, ccs[np.arange(len(j)), jp]
    den = (y0 - 2 * y1 + y2)
    frac = np.where((j > 0) & (j < L) & (den != 0), 0.5 * (y0 - y2) / den, 0.0)
    return (j + frac) - L // 2, c


def invert(G, d, w, nboot=200, rng=None):
    """Weighted LSQ for m=[dx,dy,dz,dt0] with Huber IRLS; bootstrap sd."""
    def solve(G, d, w):
        ww = w.copy()
        for _ in range(5):
            W = np.sqrt(ww)[:, None]
            m, *_ = np.linalg.lstsq(G * W, d * W[:, 0], rcond=None)
            r = d - G @ m
            s = max(1.4826 * np.median(np.abs(r)), 0.005)   # floor: 5 ms (avoid /0 on exact fits)
            hub = np.where(np.abs(r) <= 1.5 * s, 1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
            ww = w * hub
        return m, r
    m, r = solve(G, d, w)
    rng = rng or np.random.default_rng(0)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(d), len(d))
        if len(np.unique(i)) < 6:
            continue
        bs.append(solve(G[i], d[i], w[i])[0])
    bs = np.array(bs)
    return m, (bs.std(0) if len(bs) else np.full(4, np.nan)), r


def azimuth_gap(az):
    a = np.sort(np.mod(az, 360))
    return np.max(np.diff(np.r_[a, a[0] + 360])) if len(a) > 1 else 360.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--mode', choices=['bins', 'events'], default='bins')
    ap.add_argument('--model', default='3d')
    ap.add_argument('--h5', default=None)
    ap.add_argument('--templates', default=None)
    ap.add_argument('--loc', default=None, help='location json (master location, picks, s_scale)')
    ap.add_argument('--nbin', type=int, default=150)
    ap.add_argument('--ccmin', type=float, default=0.5)
    ap.add_argument('--minobs', type=int, default=8)
    ap.add_argument('--minsta', type=int, default=6)
    ap.add_argument('--fmin', type=float, default=2.0)
    ap.add_argument('--fmax', type=float, default=8.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--stations', default=None,
                    help='comma list of NET.STA to use (fixed station set test); default all')
    a = ap.parse_args()
    F = a.family
    h = h5py.File(a.h5 or f'data/waveforms/fam{F}.h5', 'r')
    T = h5py.File(a.templates or f'data/templates/fam{F}_templates.h5', 'r')
    locf = a.loc or f'data/locations/fam{F}_{a.model}.json'
    loc = json.load(open(locf))
    picks = pd.read_csv(locf.replace('.json', '_picks.csv')).set_index('sta')
    g = json.load(open(f'data/velocity/tt_{a.model}/grid.json'))
    gx, gy, gz = map(np.array, (g['sub_x'], g['sub_y'], g['sub_z']))
    tr = Transformer.from_crs('EPSG:4326', g['proj'], always_xy=True)
    x0, y0 = tr.transform(loc['lon'], loc['lat'])
    z0 = loc['depth_km']
    i0, j0, k0 = np.argmin(abs(gx - x0)), np.argmin(abs(gy - y0)), np.argmin(abs(gz - z0))
    d = g['d']
    sr, pre = h.attrs['sr'], h.attrs['pre']
    sos = butter(4, [a.fmin, a.fmax], btype='band', fs=sr, output='sos')
    t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    nev = len(t_ev)
    ml = int(round(MAXLAG * sr))

    # observation definitions: (channel, phase, window start index, slowness vector)
    keep = set(a.stations.split(',')) if a.stations else None
    defs = []
    for cid in T:
        n, s, l, c = cid.split('.')
        sta = f'{n}.{s}'
        if keep is not None and sta not in keep:
            continue
        if sta not in picks.index or not os.path.exists(f'data/velocity/tt_{a.model}/{sta}.npz'):
            continue
        z = np.load(f'data/velocity/tt_{a.model}/{sta}.npz')
        r = picks.loc[sta]
        for ph, col, win, comp_ok, scale in [('S', 'S_final', S_WIN, c[-1] in 'EN12', loc.get('s_scale', 1.0)),
                                              ('P', 'P', P_WIN, c[-1] == 'Z', 1.0)]:
            if not comp_ok or col not in r or not np.isfinite(r[col]):
                continue
            V = z['t' + ph.lower()] * scale
            grad = np.array([(V[i0 + 1, j0, k0] - V[i0 - 1, j0, k0]) / (2 * d),
                             (V[i0, j0 + 1, k0] - V[i0, j0 - 1, k0]) / (2 * d),
                             (V[i0, j0, k0 + 1] - V[i0, j0, k0 - 1]) / (2 * d)])
            az = np.degrees(np.arctan2(-grad[0], -grad[1]))   # direction source->station
            ws = int(round((r[col] + win[0] - pre) * sr))
            we = int(round((r[col] + win[1] - pre) * sr))
            defs.append(dict(cid=cid, sta=sta, phase=ph, ws=ws, we=we, grad=grad, az=az,
                             sigma=0.02 if ph == 'S' else 0.03))
    print(f'{len(defs)} phase-channel observations defined', flush=True)

    # measure dt for each obs: per bin stack or per event
    if a.mode == 'bins':
        order = np.argsort(t_ev.values)
        bins = [order[i:i + a.nbin] for i in range(0, nev - a.nbin // 2, a.nbin)]
    rows = []
    meas = {}   # (unit, obs index) -> (dt, cc)
    for oi, o in enumerate(defs):
        ev = h[f'chan/{o["cid"]}/ev'][:]
        D = sosfiltfilt(sos, h[f'chan/{o["cid"]}/data'][:].astype('float64'), axis=1)
        D /= (np.sqrt(np.mean(D[:, o['ws']:o['we']] ** 2, axis=1)) + 1e-20)[:, None]
        mw = T[o['cid']]['stack'][o['ws']:o['we']].astype(float)
        if a.mode == 'bins':
            pos = {e: k for k, e in enumerate(ev)}
            units, X = [], []
            for b, members in enumerate(bins):
                kk = [pos[e] for e in members if e in pos]
                if len(kk) < max(20, a.nbin // 4):
                    continue
                units.append(b)
                X.append(D[kk].mean(0))
            if not X:
                continue
            X = np.array(X)
        else:
            units, X = list(ev), D
        Xw = X[:, o['ws'] - ml:o['we'] + ml]
        lag, cc = cc_lag(mw, Xw, ml)
        for u, L, c in zip(units, lag / sr, cc):
            meas.setdefault(u, []).append((oi, L, c))

    out = []
    for u, lst in meas.items():
        lst = [(oi, L, c) for oi, L, c in lst if c >= (a.ccmin if a.mode == 'events' else 0.3)]
        stas = {defs[oi]['sta'] for oi, _, _ in lst}
        if len(lst) < a.minobs or len(stas) < a.minsta:
            continue
        gap = azimuth_gap(np.array([defs[oi]['az'] for oi, _, _ in lst]))
        if gap >= 180:
            continue
        G = np.array([np.r_[defs[oi]['grad'], 1.0] for oi, _, _ in lst])
        dd = np.array([L for _, L, _ in lst])
        w = np.array([c ** 2 / defs[oi]['sigma'] ** 2 for oi, _, c in lst])
        m, sd, r = invert(G, dd, w, nboot=100 if a.mode == 'bins' else 30)
        if a.mode == 'bins':
            members = bins[u]
            tmid = t_ev[members].sort_values()[len(members) // 2]
            out.append(dict(bin=u, t_mid=tmid, t_start=t_ev[members].min(), t_end=t_ev[members].max(),
                            n_ev=len(members), n_obs=len(lst), n_sta=len(stas), gap=gap,
                            dx=m[0], dy=m[1], dz=m[2], dt0=m[3], sd_x=sd[0], sd_y=sd[1], sd_z=sd[2],
                            rms=np.sqrt(np.mean(r ** 2)), mean_cc=np.mean([c for _, _, c in lst])))
            for (oi, L, c), rr in zip(lst, r):
                rows.append(dict(bin=u, cid=defs[oi]['cid'], phase=defs[oi]['phase'], dt=L, cc=c, res=rr))
        else:
            out.append(dict(ev=int(u), time_utc=str(t_ev[u]), n_obs=len(lst), n_sta=len(stas), gap=gap,
                            dx=m[0], dy=m[1], dz=m[2], dt0=m[3], sd_x=sd[0], sd_y=sd[1], sd_z=sd[2],
                            rms=np.sqrt(np.mean(r ** 2)), mean_cc=np.mean([c for _, _, c in lst])))
    os.makedirs('data/relocations', exist_ok=True)
    tag = f'fam{F}_{a.mode}_{a.model}{a.tag}'
    o = pd.DataFrame(out)
    o.to_csv(f'data/relocations/{tag}.csv', index=False)
    if rows:
        pd.DataFrame(rows).to_csv(f'data/relocations/{tag}_obs.csv', index=False)
    print(f'{a.mode}: relocated {len(o)} units (of {len(meas)} measured)')
    if len(o):
        print(o[['dx', 'dy', 'dz', 'dt0', 'sd_x', 'sd_y', 'sd_z', 'rms', 'n_sta']].describe().round(3).to_string())


if __name__ == '__main__':
    main()
