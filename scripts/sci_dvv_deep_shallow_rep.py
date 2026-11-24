#!/usr/bin/env python3
"""Deep/shallow dv/v, item 2: ordinary repeating earthquakes near the MTJ as a receiver reference.

1. load   : event waveforms (scripts/sci_dvv_deep_shallow_repdl.py) -> 100 Hz arrays aligned on origin (-10..+60 s)
            per station x component -> rep/wf_arrays.npz
2. pairs  : catalog neighbours (<= 5 km horiz., <= 5 km depth, |dM| <= 0.5) -> CC (1-12 Hz, window P-1 .. S+5 s,
            lag +-1 s) per station (mean over components); repeater if CC >= 0.95 at >= 2 stations
            (relaxed variant CC >= 0.90 reported too) -> clusters (connected components)
3. dvv    : per cluster pair x station x component (2-8 Hz): align on direct S (lag of CC in S-0.5..S+2.5 after
            the P-S window alignment), stretch about S in the coda S+3 .. S+20 s (or data end): eps_ij = dvv(j)-dvv(i)
4. fit    : per station and depth class (shallow < 10 km, deep > 15 km): eps_ij = f(t_j) - f(t_i),
            f = annual + Petrolia/Ferndale/M7 steps + ramp from 2021-06 (as for the LFEs), CC weights + Huber;
            bootstrap over clusters; placebo M7 dates (random dates 2016-2021, M7 column only).
Outputs data/sci/dvv_deep_shallow/rep/: pairs_cc.csv, clusters.csv, rep_pairs_dvv.csv, rep_fit.csv
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.spatial import cKDTree

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_deep_shallow/rep'
sys.path.insert(0, f'{ROOT}/scripts')
STAS = {'NC.KMR': (40.202175, -123.708633), 'NC.KMPB': (40.417194, -124.120758), 'PB.B045': (40.436001, -123.996498),
        'PB.B046': (40.321972, -124.284302), 'PB.B047': (40.249933, -124.169848), 'PB.B049': (40.237499, -123.819702),
        'PB.B932': (40.275002, -124.221199), 'BK.BRIC': (40.11223, -123.8965), 'BK.HAYF': (40.5227, -123.23198)}
SL = list(STAS)
COMPS = ['Z', 'N', 'E']
SR = 100.0
NS = 7000
VP, VS = 6.0, 3.5
EPS = np.round(np.arange(-0.01, 0.01 + 1e-9, 0.0002), 6)


def cat():
    c = pd.read_csv(f'{OUT}/candidates.csv')
    c['t'] = pd.to_datetime(c.Time)
    c['depth'] = c['Depth/km']
    return c.reset_index(drop=True)


def load_one(evid):
    from obspy import read, UTCDateTime  # noqa: F401
    f = f'{OUT}/wf/{evid}.mseed'
    A = np.full((len(SL), 3, NS), np.nan, np.float32)
    if not os.path.exists(f):
        return evid, A
    try:
        st = read(f)
    except Exception:                                  # noqa: BLE001
        return evid, A
    t0 = min(tr.stats.starttime for tr in st)
    for tr in st:
        sta = f'{tr.stats.network}.{tr.stats.station}'
        if sta not in STAS or tr.stats.sampling_rate != SR:
            continue
        ch = tr.stats.channel[-1]
        ci = {'Z': 0, 'N': 1, '1': 1, 'E': 2, '2': 2}.get(ch)
        if ci is None:
            continue
        x = tr.data.astype(float)
        if len(x) < NS * 0.9 or np.std(x) == 0:
            continue
        x = x - np.mean(x)
        k0 = int(round((tr.stats.starttime - t0) * SR))
        n = min(NS - k0, len(x))
        if k0 < 0 or n < NS * 0.9:
            continue
        A[SL.index(sta), ci, k0:k0 + n] = x[:n]
    return evid, A


def stage_load(a):
    c = cat()
    with Pool(a.procs) as p:
        res = dict(p.map(load_one, c.EventID.tolist(), chunksize=20))
    W = np.array([res[e] for e in c.EventID])
    np.save(f'{OUT}/wf_raw.npy', W)
    print(W.shape, np.isfinite(W[:, :, 0, 100]).sum(0))


def dist_km(lat1, lon1, lat2, lon2):
    return np.hypot((lat1 - lat2) * 111.2, (lon1 - lon2) * 111.2 * np.cos(np.radians(40.3)))


def arrivals(c):
    """predicted P, S times (s after window start = origin - 10 s) per event x station."""
    P = np.zeros((len(c), len(SL)))
    S = np.zeros((len(c), len(SL)))
    for j, s in enumerate(SL):
        la, lo = STAS[s]
        r = np.hypot(dist_km(c.Latitude.values, c.Longitude.values, la, lo), c.depth.values)
        P[:, j] = 10 + r / VP
        S[:, j] = 10 + r / VS
    return P, S


def filt(W, f1, f2):
    sos = butter(4, [f1, f2], btype='band', fs=SR, output='sos')
    X = np.nan_to_num(W.astype(float))
    return sosfiltfilt(sos, X, axis=-1)


def cc_lag(a, b, maxlag):
    """max normalised CC of b shifted vs a; a, b same length windows extracted with margin maxlag (FFT)."""
    from scipy.signal import fftconvolve
    n = len(a) - 2 * maxlag
    r = a[maxlag:maxlag + n]
    r = r - r.mean()
    rn = np.linalg.norm(r) + 1e-20
    num = fftconvolve(b, r[::-1], mode='valid')          # length 2*maxlag+1, lag L = k - maxlag
    c1 = np.concatenate([[0.0], np.cumsum(b)])
    c2 = np.concatenate([[0.0], np.cumsum(b * b)])
    S1 = c1[n:] - c1[:-n]
    S2 = c2[n:] - c2[:-n]
    var = np.clip(S2 - S1 ** 2 / n, 1e-30, None)
    v = num / (rn * np.sqrt(var))                         # r is zero-mean -> mean of s drops out of num
    k = int(np.argmax(v))
    return float(v[k]), k - maxlag


_G = {}


def pair_job(args):
    i, j = args
    Xi, Xj, Pi, Si = _G['X'][i], _G['X'][j], _G['P'][i], _G['S'][i]
    out = []
    for k in range(len(SL)):
        ccs, lags = [], []
        for ci in range(3):
            a, b = Xi[k, ci], Xj[k, ci]
            if not (np.any(a) and np.any(b)):
                continue
            w0 = int((Pi[k] - 1.0) * SR)
            w1 = int((Si[k] + 5.0) * SR)
            ml = int(1.0 * SR)
            if w0 - ml < 0 or w1 + ml > NS:
                continue
            v, L = cc_lag(a[w0 - ml:w1 + ml], b[w0 - ml:w1 + ml], ml)
            ccs.append(v)
            lags.append(L)
        if ccs:
            out.append((i, j, SL[k], float(np.mean(ccs)), float(np.median(lags)) / SR, len(ccs)))
    return out


def stage_pairs(a):
    c = cat()
    W = np.load(f'{OUT}/wf_raw.npy', mmap_mode='r')
    X = filt(np.asarray(W), 1.0, 12.0).astype(np.float32)
    np.save(f'{OUT}/wf_1_12.npy', X)
    P, S = arrivals(c)
    x = (c.Longitude + 124) * 85.0
    y = (c.Latitude - 40.3) * 111.2
    pr = cKDTree(np.c_[x, y]).query_pairs(5.0, output_type='ndarray')
    i, j = pr[:, 0], pr[:, 1]
    ok = (np.abs(c.depth.values[i] - c.depth.values[j]) <= 5) & (np.abs(c.Magnitude.values[i] - c.Magnitude.values[j]) <= 0.5)
    i, j = i[ok], j[ok]
    print(len(i), 'candidate pairs', flush=True)
    _G.update(X=X, P=P, S=S)
    jobs = list(zip(i, j))
    rows = []
    with Pool(a.procs) as p:
        for k, r in enumerate(p.imap_unordered(pair_job, jobs, chunksize=50)):
            rows += r
            if k % 5000 == 0:
                print(k, flush=True)
    D = pd.DataFrame(rows, columns=['i', 'j', 'sta', 'cc', 'lag', 'ncomp'])
    D.to_csv(f'{OUT}/pairs_cc.csv', index=False)
    clusters(c, D)


def clusters(c, D):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    res = {}
    for thr in (0.95, 0.90):
        g = D[D.cc >= thr].groupby(['i', 'j']).size()
        g = g[g >= 2]
        n = len(c)
        M = sp.coo_matrix((np.ones(len(g)), (g.index.get_level_values(0), g.index.get_level_values(1))), shape=(n, n))
        nc, lab = connected_components(M, directed=False)
        sizes = np.bincount(lab)
        lab = np.where(sizes[lab] >= 2, lab, -1)
        res[thr] = lab
        cl = pd.Series(lab[lab >= 0]).value_counts()
        print(f'CC>={thr}: {len(g)} repeater pairs, {len(cl)} clusters, {cl.sum()} events', flush=True)
    C = c[['EventID', 'Time', 'Latitude', 'Longitude', 'depth', 'Magnitude']].copy()
    C['cl95'] = res[0.95]
    C['cl90'] = res[0.90]
    C.to_csv(f'{OUT}/clusters.csv', index=False)
    return C


def stretch(ref, cur, tau, win):
    """eps maximising CC of ref(tau(1+eps)) vs cur(tau) in win (bool)."""
    tc = tau[win]
    c = cur[win] - cur[win].mean()
    cn = np.linalg.norm(c) + 1e-20
    M = np.array([np.interp(tc * (1 + e), tau, ref) for e in EPS])
    M -= M.mean(1, keepdims=True)
    v = (M @ c) / (np.linalg.norm(M, axis=1) * cn + 1e-20)
    k = int(np.argmax(v))
    if 0 < k < len(EPS) - 1:
        y0, y1, y2 = v[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        f = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return EPS[k] + f * (EPS[1] - EPS[0]), y1
    return np.nan, -1.0


def dvv_job(args):
    i, j, lagij = args
    Xi, Xj, Pi, Si = _G['X'][i], _G['X'][j], _G['P'][i], _G['S'][i]
    t = np.arange(NS) / SR
    out = []
    for k in range(len(SL)):
        L0 = lagij.get(SL[k])
        if L0 is None:
            continue
        for ci in range(3):
            a, b = Xi[k, ci], Xj[k, ci]
            if not (np.any(a) and np.any(b)):
                continue
            # b shifted by L0 (from the P-S window), then S-window refinement
            bb = np.interp(t + L0, t, b)
            s0, s1 = int((Si[k] - 0.5) * SR), int((Si[k] + 2.5) * SR)
            ml = int(0.2 * SR)
            if s0 - ml < 0 or s1 + ml > NS:
                continue
            v, L = cc_lag(a[s0 - ml:s1 + ml], bb[s0 - ml:s1 + ml], ml)
            bb = np.interp(t + L / SR, t, bb)
            tau = t - Si[k]
            tend = min(20.0, (NS / SR - 1.0) / 1.01 - Si[k])
            win = (tau >= 3.0) & (tau <= tend)
            if tend - 3.0 < 4:
                continue
            # noise level check: coda rms above 2x pre-P noise
            nz = np.sqrt(np.mean(a[int(1 * SR):int((Pi[k] - 1) * SR)] ** 2)) if Pi[k] > 3 else np.nan
            snr = np.sqrt(np.mean(a[win] ** 2)) / (nz + 1e-20)
            e, cc = stretch(a, bb, tau, win)
            out.append((i, j, SL[k], 'ZNE'[ci], e, cc, v, snr))
    return out


def stage_dvv(a):
    c = cat()
    C = pd.read_csv(f'{OUT}/clusters.csv')
    D = pd.read_csv(f'{OUT}/pairs_cc.csv')
    W = np.load(f'{OUT}/wf_raw.npy', mmap_mode='r')
    X = filt(np.asarray(W), 2.0, 8.0).astype(np.float32)
    P, S = arrivals(c)
    lab = C.cl95.values
    jobs = []
    lagd = {(r.i, r.j, r.sta): r.lag for r in D.itertuples()}
    for cl in np.unique(lab[lab >= 0]):
        mem = np.where(lab == cl)[0]
        mem = mem[np.argsort(c.t.values[mem])]
        for ii in range(len(mem)):
            for jj in range(ii + 1, len(mem)):
                i, j = mem[ii], mem[jj]
                key = (min(i, j), max(i, j))
                lags = {s: (lagd[(key[0], key[1], s)] if key == (i, j) else -lagd[(key[0], key[1], s)])
                        for s in SL if (key[0], key[1], s) in lagd}
                if lags:
                    jobs.append((i, j, lags))
    _G.update(X=X, P=P, S=S)
    print(len(jobs), 'pairs for dv/v', flush=True)
    rows = []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(dvv_job, jobs, chunksize=20):
            rows += r
    R = pd.DataFrame(rows, columns=['i', 'j', 'sta', 'comp', 'eps', 'cc', 'ccS', 'snr'])
    R['ti'] = c.t.values[R.i]
    R['tj'] = c.t.values[R.j]
    R['cl'] = lab[R.i]
    R['cl95'] = C.cl95.values[R.i]
    R['depth'] = 0.5 * (c.depth.values[R.i] + c.depth.values[R.j])
    R['mag'] = 0.5 * (c.Magnitude.values[R.i] + c.Magnitude.values[R.j])
    R.to_csv(f'{OUT}/rep_pairs_dvv.csv', index=False)


def fcols(t, m7=None):
    from sci_dvv_deep_shallow import cols
    from dvv_cwire import epoch_s
    ts = epoch_s(pd.DatetimeIndex(t))
    C = cols(ts)
    if m7 is not None:
        C[:, 4] = (ts >= epoch_s([m7])[0]).astype(float)
    return C[:, [0, 1, 2, 3, 4, 6]]               # sin cos pet fer m7 ramp (no transient)


def fit_pairs(R, m7=None):
    A = fcols(R.tj, m7) - fcols(R.ti, m7)
    use = np.abs(A).sum(0) > 0
    A = A[:, use]
    y = R.eps.values
    cc = np.clip(R.cc.values, 0, 0.99)
    w0 = cc ** 2 / (1 - cc ** 2)
    w = w0.copy()
    for it in range(3):
        W = np.sqrt(w)[:, None]
        x, *_ = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)
        r = y - A @ x
        z = np.abs(r) * np.sqrt(w0)
        z = z / (1.4826 * np.median(z) + 1e-20)
        w = w0 * np.where(z <= 2, 1, 2 / np.maximum(z, 1e-12))
    out = np.full(6, np.nan)
    out[use] = x
    return out


def stage_fit(a):
    R = pd.read_csv(f'{OUT}/rep_pairs_dvv.csv', parse_dates=['ti', 'tj'])
    R = R[(R.cc >= 0.5) & (R.snr >= 2) & np.isfinite(R.eps) & (np.abs(R.eps) < 0.009)]
    from dvv_cwire import epoch_s
    G21 = pd.DatetimeIndex([f'2021-{m:02d}-15' for m in range(1, 13)])
    G25 = pd.DatetimeIndex([f'2025-{m:02d}-15' for m in range(1, 13)])
    Cd = fcols(G25).mean(0) - fcols(G21).mean(0)
    Cd[[0, 1]] = 0
    rng = np.random.default_rng(5)
    rows = []
    R['cls'] = np.where(R.depth < 10, 'shallow', np.where(R.depth > 15, 'deep', 'mid'))
    for (sta, cls), g in R.groupby(['sta', 'cls']):
        if cls == 'mid':
            continue
        span_m7 = ((g.ti < '2024-12-05') & (g.tj > '2024-12-06')).sum()
        span_21 = ((g.ti < '2021-06-01') & (g.tj > '2023-01-01')).sum()
        if len(g) < 20:
            continue
        x = fit_pairs(g)
        cls_ids = g.cl.unique()
        B = []
        for b in range(100):
            pick = rng.choice(cls_ids, len(cls_ids))
            gb = pd.concat([g[g.cl == k] for k in pick])
            B.append(fit_pairs(gb))
        B = np.array(B)
        # placebo M7 dates
        PL = []
        for k in range(40):
            d = pd.Timestamp('2016-01-01') + pd.Timedelta(days=int(rng.integers(0, 365 * 5)))
            gp = g[(g.tj < '2021-06-01')]
            if len(gp) < 20:
                break
            PL.append(fit_pairs(gp, m7=d)[4])
        D = np.nan_to_num(x) @ Cd
        DB = np.nan_to_num(B) @ Cd
        rows.append(dict(sta=sta, cls=cls, n_pairs=len(g), n_clusters=len(cls_ids), n_span_m7=int(span_m7),
                         n_span_2021_2023=int(span_21), depth_med=g.depth.median(),
                         m7_pct=x[4] * 100, m7_se_boot=np.nanstd(B[:, 4]) * 100,
                         m7_se_plac=np.nanstd(PL) * 100 if PL else np.nan,
                         D_pct=D * 100, D_se_boot=np.std(DB) * 100, ramp_pct_yr=x[5] * 100,
                         annual_amp_pct=np.hypot(x[0], x[1]) * 100))
        print(rows[-1], flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(f'{OUT}/rep_fit.csv', index=False)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['load', 'pairs', 'dvv', 'fit', 'all'])
    ap.add_argument('--procs', type=int, default=10)
    a = ap.parse_args()
    for s, f in [('load', stage_load), ('pairs', stage_pairs), ('dvv', stage_dvv), ('fit', stage_fit)]:
        if a.stage in (s, 'all'):
            f(a)


if __name__ == '__main__':
    main()
