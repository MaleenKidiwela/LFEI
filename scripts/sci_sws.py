#!/usr/bin/env python3
"""Time-lapse shear-wave splitting (SWS) with repeating LFE stacks as fixed sources.

Stages (run in order):
  meta     station table: horizontal channel epochs (azimuth, gain), coordinates, borehole flag,
           1D-model incidence angle per family-station  -> data/sci/sws/stations.csv, incidence.csv
  measure  per station (parallel over stations, <=4 procs, nice): for every family
           - isolated events (no other catalog LFE within +-5 s, as dvv_cwire), both horizontals present
           - per event: gain-correct (epoch scale), rotate to N/E with the EPOCH azimuth, band-pass,
             normalise by the event's horizontal RMS over the whole 60-s window (inverse-noise weight)
           - family reference stack (all isolated events) -> S window [t_on-0.15, t_on+WIN] s, t_on from
             the horizontal envelope (first >25% of peak near the S pick)
           - global time bins (calendar months; months containing a mainshock split at the mainshock)
             family-bin stack (>= MIN_EV events) aligned to the reference by a common N/E lag (+-0.1 s)
           - Silver & Chan (1991) eigenvalue method: grid phi 0..178 (2 deg) x dt 0..0.30 s (5 ms, 200 Hz),
             symmetric shift, lambda2 surface normalised by the stack's pre-origin noise energy
             (so surfaces from different families can be summed, Wolfe & Silver 1998)
           - bootstrap over events (B=40) and odd/even halves of each family-bin
           - per-family measurement + station-pooled measurement (sum of normalised surfaces over families)
           - fixed-phi dt: dt along the reference fast direction (family's own / station pooled)
           - polarisation of the corrected S wave (orientation monitor: a sensor rotation moves pol and phi
             together, leaves dt unchanged)
           outputs data/sci/sws/meas/{STA}_{family,pooled,ref}.parquet, {STA}_refsurf.npz
  analyze  time-series fits (offset + trend + annual + steps at Petrolia/Ferndale/M7), random-date
           placebo steps, annual placebo periods, long-term (2025-26 minus 2019-21) vs pre-2021 windows,
           odd/even half consistency, metadata-epoch steps, family-vs-station decomposition
           -> data/sci/sws/*.csv, summary.json, figures/sci_sws_*.png
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import os
import sys
import time
import zlib

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/sws'
FIG = f'{ROOT}/figures'
QUAKES = {'pet': '2021-12-20T20:10', 'fer': '2022-12-20T10:34', 'm7': '2024-12-05T18:44'}
BANDS = {'b28': (2.0, 8.0), 'b25': (2.0, 5.0)}
UP = 4                       # 50 Hz -> 200 Hz
PHI = np.arange(0, 180, 2.0)  # deg
NDT = 61                      # 0..0.30 s at 5 ms
DT = np.arange(NDT) / 200.0
WIN = 1.0                     # s after onset
PRE_ON = 0.15
SEG0, SEG1 = -1.5, 4.5        # signal segment relative to S pick (s)
NOISE = (-14.5, -9.5)         # catalog-relative noise window (before the origin, ~cat-8.9 s)
ISO = 5.0
MIN_EV = 8
NBOOT = 40


def epoch_s(x):
    x = pd.DatetimeIndex(pd.to_datetime(x))
    return np.asarray((x - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1), dtype=float)


QT = epoch_s(list(QUAKES.values()))


# ------------------------------------------------------------------ metadata
def read_meta():
    cols = ['net', 'sta', 'loc', 'cha', 'lat', 'lon', 'elev', 'depth', 'az', 'dip', 'inst', 'scale',
            'sfreq', 'sunit', 'sr', 'start', 'end']
    a = pd.read_csv(f'{ROOT}/data/stations/cha_ncedc_2008_2026.txt', sep='|', comment='#', names=cols,
                    dtype={'loc': str})
    a['loc'] = a['loc'].fillna('')
    a = a[a.net.isin(['BK', 'NC', 'PB']) & a.cha.str.match(r'^(HH[ENZ]|EH[12Z])$')]
    a['start'] = pd.to_datetime(a.start.str[:19])
    a['end'] = pd.to_datetime(a.end.str[:19].replace('3000-01-01T00:00:00', '2100-01-01T00:00:00'))
    a['cid'] = a.net + '.' + a.sta + '.' + a['loc'] + '.' + a.cha
    return a


def vs_model():
    v = np.loadtxt(f'{ROOT}/data/velocity/furlong2024/vel_p.1d')
    z = v[::2, 0]
    vp = v[::2, 1]
    return z, vp / 1.78   # layer tops (km, positive down, relative to sea level), Vs


def incidence(dist_km, zsrc, zrec):
    """1D flat-layer ray: incidence angle (deg from vertical) at the receiver; zrec, zsrc km below sea level."""
    ztop, vs = vs_model()
    zb = np.r_[ztop[1:], 1e3]
    # layer thicknesses between receiver and source
    h, v = [], []
    for z0, z1, vv in zip(ztop, zb, vs):
        lo, hi = max(z0, zrec), min(z1, zsrc)
        if hi > lo:
            h.append(hi - lo)
            v.append(vv)
    h, v = np.array(h), np.array(v)
    pmax = 1 / v.max()
    ps = np.linspace(0, pmax * 0.999999, 20000)
    X = np.array([np.sum(h * p * v / np.sqrt(1 - (p * v) ** 2)) for p in ps])
    p = np.interp(dist_km, X, ps)
    return np.degrees(np.arcsin(p * v[0])), np.degrees(np.arctan2(dist_km, zsrc - zrec))


def stage_meta(a):
    m = read_meta()
    h = m[m.cha.str.match(r'^(HH[EN]|EH[12])$')]
    rows = []
    for (net, sta), g in h.groupby(['net', 'sta']):
        g0 = g.iloc[-1]
        z_rec = -g0.elev / 1000 + g0.depth / 1000
        ep = []
        for cid, gg in g.groupby('cid'):
            for _, r in gg.iterrows():
                ep.append(f"{r.cha}:{r.start:%Y-%m-%d}..{r.end:%Y-%m-%d}:az{r.az:g}:sc{r.scale:.4g}:{r.inst.split(',')[0]}")
        rows.append(dict(sta=f'{net}.{sta}', lat=g0.lat, lon=g0.lon, elev=g0.elev, depth=g0.depth, z_rec_km=z_rec,
                         borehole=bool(net == 'PB' or g0.depth > 50), start=g.start.min(), end=g.end.max(),
                         n_epochs=len(g) // 2, epochs=' | '.join(ep)))
    st = pd.DataFrame(rows)
    st.to_csv(f'{OUT}/stations.csv', index=False)
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    inc = []
    for F in range(1, 28):
        L = json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json'))
        for _, s in st.iterrows():
            az, baz, d = geod.inv(L['lon'], L['lat'], s.lon, s.lat)
            i1, i0 = incidence(d / 1000, L['depth_km'], s.z_rec_km)
            inc.append(dict(family=F, sta=s.sta, dist_km=d / 1000, baz=baz % 360, src_depth=L['depth_km'],
                            inc_1d=i1, inc_straight=i0))
    pd.DataFrame(inc).to_csv(f'{OUT}/incidence.csv', index=False)
    print(st[['sta', 'borehole', 'start', 'end', 'n_epochs']].to_string())


# ------------------------------------------------------------------ splitting core
def sc_surface(X, i0, i1):
    """X: (R, 2, n) [E, N] at 200 Hz (R replicates). Window [i0, i1); symmetric shift k = k1 + k2
    (fast taken k1 samples earlier, slow k2 later). Returns lam2, lam1, cff, css, cfs, each (R, nphi, ndt)."""
    W = i1 - i0
    k = np.arange(NDT)
    k1 = k // 2
    k2 = k - k1
    ia = (i0 - k1)[:, None] + np.arange(W)[None, :]
    ib = (i0 + k2)[:, None] + np.arange(W)[None, :]
    A = X[:, :, ia].transpose(0, 2, 1, 3)   # (R, ndt, 2, W)
    B = X[:, :, ib].transpose(0, 2, 1, 3)
    AA = A @ A.transpose(0, 1, 3, 2)          # (R, ndt, 2, 2)
    BB = B @ B.transpose(0, 1, 3, 2)
    AB = A @ B.transpose(0, 1, 3, 2)
    ph = np.radians(PHI)
    u = np.stack([np.sin(ph), np.cos(ph)])        # (2, nphi) fast (E, N)
    v = np.stack([np.cos(ph), -np.sin(ph)])       # slow, azimuth phi+90
    cff = np.einsum('ip,rkij,jp->rpk', u, AA, u, optimize=True)
    css = np.einsum('ip,rkij,jp->rpk', v, BB, v, optimize=True)
    cfs = np.einsum('ip,rkij,jp->rpk', u, AB, v, optimize=True)
    m = 0.5 * (cff + css)
    r = np.sqrt(0.25 * (cff - css) ** 2 + cfs ** 2)
    return m - r, m + r, cff, css, cfs


def parab(y, k):
    if 0 < k < len(y) - 1:
        d = y[k - 1] - 2 * y[k] + y[k + 1]
        if d > 0:
            return float(np.clip(0.5 * (y[k - 1] - y[k + 1]) / d, -0.5, 0.5))
    return 0.0


def best_from_surface(L2):
    """Return phi (deg), dt (s) at the minimum with parabolic refinement (phi periodic)."""
    ip, ik = np.unravel_index(np.argmin(L2), L2.shape)
    col = L2[:, ik]
    yp = np.array([col[(ip - 1) % len(PHI)], col[ip], col[(ip + 1) % len(PHI)]])
    fp = parab(yp, 1)
    fk = parab(L2[ip], ik)
    phi = (PHI[ip] + fp * (PHI[1] - PHI[0])) % 180
    dt = DT[ik] + fk * (DT[1] - DT[0])
    return phi, dt, ip, ik


LOC_PHI, LOC_DT = 30.0, 0.05     # local search half-widths around the reference (time-lapse tracking)


def dt_at_phi(L2, phi0, dtc=None):
    """dt minimising lambda2 along the fixed fast direction phi0 (restricted to |dt-dtc|<=LOC_DT)."""
    j = int(np.round(phi0 / (PHI[1] - PHI[0]))) % len(PHI)
    row = L2[j].astype(float).copy()
    if dtc is not None:
        row[np.abs(DT - dtc) > LOC_DT + 1e-9] = np.inf
    k = int(np.argmin(row))
    rr = np.where(np.isfinite(row), row, L2[j])
    return DT[k] + parab(rr, k) * (DT[1] - DT[0])


def phi_at_dt(L2, dt0, phic=None):
    k = int(np.clip(np.round(dt0 / (DT[1] - DT[0])), 0, NDT - 1))
    col = L2[:, k].astype(float).copy()
    if phic is not None:
        col[np.abs(wrap90(PHI - phic)) > LOC_PHI + 1e-9] = np.inf
    j = int(np.argmin(col))
    c0 = L2[:, k]
    yp = np.array([c0[(j - 1) % len(PHI)], c0[j], c0[(j + 1) % len(PHI)]])
    return (PHI[j] + parab(yp, 1) * (PHI[1] - PHI[0])) % 180


def local_min(L2, phic, dtc):
    """2-D minimum restricted to |phi-phic|<=LOC_PHI, |dt-dtc|<=LOC_DT. Returns phi, dt, edge flag."""
    M = L2.astype(float).copy()
    M[np.abs(wrap90(PHI - phic)) > LOC_PHI + 1e-9, :] = np.inf
    M[:, np.abs(DT - dtc) > LOC_DT + 1e-9] = np.inf
    ip, ik = np.unravel_index(np.argmin(M), M.shape)
    edge = (abs(abs(wrap90(PHI[ip] - phic)) - LOC_PHI) < 1.5) or (abs(abs(DT[ik] - dtc) - LOC_DT) < 0.003) \
        or ik in (0, NDT - 1)
    col = L2[:, ik]
    yp = np.array([col[(ip - 1) % len(PHI)], col[ip], col[(ip + 1) % len(PHI)]])
    return (PHI[ip] + parab(yp, 1) * (PHI[1] - PHI[0])) % 180, DT[ik] + parab(L2[ip], ik) * (DT[1] - DT[0]), edge


RW = 12                                   # re-centred pooling: dt offsets -60..+60 ms
DPHI = np.r_[PHI[45:] - 180, PHI[:45]]    # -90..88
DDT = np.arange(-RW, RW + 1) / 200.0


def recenter(L2s, ipr, ikr):
    """Shift surfaces (R, nphi, ndt) so that the family reference minimum sits at (dphi, ddt) = (0, 0)."""
    ip = (ipr + np.arange(-45, 45)) % len(PHI)
    ik = np.clip(ikr + np.arange(-RW, RW + 1), 0, NDT - 1)
    return L2s[:, ip][:, :, ik]


def rel_measure(Q):
    """Q: re-centred pooled surface (nphi, 2RW+1). Returns dphi_loc, ddt_loc, ddt_fix, dphi_fix, edge."""
    M = Q.astype(float).copy()
    M[np.abs(DPHI) > LOC_PHI + 1e-9, :] = np.inf
    M[:, np.abs(DDT) > LOC_DT + 1e-9] = np.inf
    ip, ik = np.unravel_index(np.argmin(M), M.shape)
    edge = abs(abs(DPHI[ip]) - LOC_PHI) < 1.5 or abs(abs(DDT[ik]) - LOC_DT) < 0.003
    dphi_loc = DPHI[ip] + (parab(Q[:, ik], ip) if 0 < ip < len(DPHI) - 1 else 0) * 2.0
    ddt_loc = DDT[ik] + parab(Q[ip], ik) * 0.005
    row = M[45].copy()
    kk = int(np.argmin(row))
    ddt_fix = DDT[kk] + parab(Q[45], kk) * 0.005
    col = M[:, RW].copy()
    jj = int(np.argmin(col))
    dphi_fix = DPHI[jj] + (parab(Q[:, RW], jj) if 0 < jj < len(DPHI) - 1 else 0) * 2.0
    return dphi_loc, ddt_loc, ddt_fix, dphi_fix, edge


def measure(X, i0, i1, noiseE):
    """X: (R, 2, n) or (2, n). Returns list of dicts and normalised surfaces (R, nphi, ndt)."""
    single = X.ndim == 2
    if single:
        X = X[None]
    noiseE = np.broadcast_to(np.atleast_1d(noiseE), (X.shape[0],))
    L2, L1, cff, css, cfs = sc_surface(X, i0, i1)
    rc = np.abs(cfs) / np.sqrt(cff * css + 1e-30)
    out = []
    for r in range(X.shape[0]):
        phi, dt, ip, ik = best_from_surface(L2[r])
        th = 0.5 * np.degrees(np.arctan2(2 * cfs[r, ip, ik], cff[r, ip, ik] - css[r, ip, ik]))
        jp, jk = np.unravel_index(np.argmax(rc[r]), rc[r].shape)
        out.append(dict(phi=phi, dt=dt, pol=(PHI[ip] + th) % 180, l2l1=L2[r, ip, ik] / L1[r, ip, ik],
                        l2min=L2[r, ip, ik] / noiseE[r], phi_rc=PHI[jp], dt_rc=DT[jk], rc=rc[r, jp, jk]))
    S = L2 / noiseE[:, None, None]
    if single:
        return out[0], S[0]
    return out, S


def axial_mean(a, w=None):
    a = np.radians(np.asarray(a) * 2)
    w = np.ones_like(a) if w is None else np.asarray(w)
    c, s = np.sum(w * np.cos(a)), np.sum(w * np.sin(a))
    R = np.hypot(c, s) / np.sum(w)
    return (np.degrees(np.arctan2(s, c)) / 2) % 180, np.degrees(np.sqrt(max(-2 * np.log(max(R, 1e-12)), 0))) / 2


def wrap90(d):
    return (np.asarray(d) + 90) % 180 - 90


# ------------------------------------------------------------------ data
def month_bins(t0='2008-01-01', t1='2026-08-01'):
    e = list(epoch_s(pd.date_range(t0, t1, freq='MS')))
    e = np.sort(np.r_[e, QT])
    return e


def load_family_station(F, sta, meta, allt, bands, inject=None):
    """Returns dict band -> (seg (n,2,nseg) [E,N] 50 Hz, noise (n,2,nn)), event times, S pick, sr."""
    h5f = f'{ROOT}/data/waveforms/fam{F}.h5'
    picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    if sta not in picks.index or not np.isfinite(picks.loc[sta, 'S_final']):
        return None
    S = float(picks.loc[sta, 'S_final'])
    with h5py.File(h5f, 'r') as h:
        sr, pre = float(h.attrs['sr']), float(h.attrs['pre'])
        chans = sorted(c for c in h['chan'] if c.startswith(sta + '.') and c[-1] in 'EN12')
        locs = sorted(set(c.rsplit('.', 1)[0] for c in chans))
        out = []
        te_all = epoch_s(h['events/time_utc'][:].astype(str))
        for lc in locs:           # e.g. NC.KMR. or PB.B049.
            cc = [c for c in chans if c.rsplit('.', 1)[0] == lc]
            if len(cc) != 2:
                continue
            e1, e2 = h[f'chan/{cc[0]}/ev'][:], h[f'chan/{cc[1]}/ev'][:]
            com, i1, i2 = np.intersect1d(e1, e2, return_indices=True)
            te = te_all[com]
            iso = (np.searchsorted(allt, te + ISO, side='right') - np.searchsorted(allt, te - ISO)) == 1
            if iso.sum() < 50:
                continue
            com, i1, i2, te = com[iso], i1[iso], i2[iso], te[iso]
            D1 = h[f'chan/{cc[0]}/data']
            D2 = h[f'chan/{cc[1]}/data']
            # read contiguous blocks then index (h5py fancy indexing is slow)
            X1 = D1[i1.min():i1.max() + 1][i1 - i1.min()]
            X2 = D2[i2.min():i2.max() + 1][i2 - i2.min()]
            out.append((cc, com, te, X1, X2))
    if not out:
        return None
    t = np.arange(3000) / sr + pre
    js = np.where((t >= S + SEG0) & (t < S + SEG1))[0]
    jn = np.where((t >= NOISE[0]) & (t < NOISE[1]))[0]
    res = {b: [[], []] for b in bands}
    TE, EV = [], []
    for cc, com, te, X1, X2 in out:
        # epoch azimuth & scale per event
        az = np.zeros((2, len(te)))
        sc = np.ones((2, len(te)))
        for k, c in enumerate(cc):
            ep = meta[meta.cid == c]
            if len(ep) == 0:
                raise RuntimeError(f'no metadata for {c}')
            st_, en_ = epoch_s(ep.start), epoch_s(ep.end)
            for e in range(len(ep)):
                m = (te >= st_[e]) & (te < en_[e])
                az[k, m] = ep.az.values[e]
                sc[k, m] = ep.scale.values[e]
            miss = ~np.any([(te >= st_[e]) & (te < en_[e]) for e in range(len(ep))], axis=0)
            if miss.any():   # outside listed epochs: nearest epoch
                for j in np.where(miss)[0]:
                    e = int(np.argmin(np.minimum(abs(te[j] - st_), abs(te[j] - en_))))
                    az[k, j], sc[k, j] = ep.az.values[e], ep.scale.values[e]
        for b, (f0, f1) in bands.items():
            sos = butter(4, [f0, f1], btype='band', fs=sr, output='sos')
            for c0 in range(0, len(te), 2000):
                sl = slice(c0, c0 + 2000)
                y1 = sosfiltfilt(sos, X1[sl].astype(np.float64), axis=1) / sc[0, sl, None]
                y2 = sosfiltfilt(sos, X2[sl].astype(np.float64), axis=1) / sc[1, sl, None]
                a1, a2 = np.radians(az[0, sl])[:, None], np.radians(az[1, sl])[:, None]
                N = y1 * np.cos(a1) + y2 * np.cos(a2)
                E = y1 * np.sin(a1) + y2 * np.sin(a2)
                if inject is not None:
                    N, E = apply_injection(N, E, te[sl], sr, inject)
                nrm = np.sqrt(N.var(1) + E.var(1)) + 1e-30
                N /= nrm[:, None]
                E /= nrm[:, None]
                res[b][0].append(np.stack([E[:, js], N[:, js]], 1).astype(np.float32))
                res[b][1].append(np.stack([E[:, jn], N[:, jn]], 1).astype(np.float32))
        TE.append(te)
        EV.append(com)
    te = np.concatenate(TE)
    o = np.argsort(te)
    data = {b: (np.concatenate(res[b][0])[o], np.concatenate(res[b][1])[o]) for b in bands}
    return data, te[o], S, sr


def apply_injection(N, E, te, sr, inj):
    """Synthetic change for events after inj['t']: extra delay inj['ddt'] (s) of the component along
    inj['phi']+90 (slow), then a sensor rotation by inj['rot'] deg (clockwise, as a misoriented sensor)."""
    m = te >= inj['t']
    if not m.any():
        return N, E
    N, E = N.copy(), E.copy()
    ph = np.radians(inj['phi'])
    f = np.sin(ph) * E[m] + np.cos(ph) * N[m]
    s = np.cos(ph) * E[m] - np.sin(ph) * N[m]
    if inj.get('ddt', 0):
        fr = np.fft.rfftfreq(s.shape[1], 1 / sr)
        s = np.fft.irfft(np.fft.rfft(s, axis=1) * np.exp(-2j * np.pi * fr * inj['ddt']), n=s.shape[1], axis=1)
    e2 = np.sin(ph) * f + np.cos(ph) * s
    n2 = np.cos(ph) * f - np.sin(ph) * s
    if inj.get('rot', 0):
        r = np.radians(inj['rot'])     # recorded components of a sensor rotated by +r: E' = E cos r - N sin r ...
        e2, n2 = e2 * np.cos(r) - n2 * np.sin(r), e2 * np.sin(r) + n2 * np.cos(r)
    E[m], N[m] = e2, n2
    return N, E


def up(x):
    return resample_poly(x, UP, 1, axis=-1)


def onset_window(ref):
    """ref: (2, nseg*UP). Onset where horizontal envelope first exceeds 25% of its peak (search -1..+2.5 s of pick)."""
    srU = 50.0 * UP
    tt = np.arange(ref.shape[1]) / srU + SEG0
    env = np.sqrt(ref[0] ** 2 + ref[1] ** 2)
    k = int(0.1 * srU)
    env = np.convolve(env, np.ones(k) / k, mode='same')
    m = (tt > -1.0) & (tt < 2.0)
    pk = env[m].max()
    on = tt[m][np.argmax(env[m] > 0.25 * pk)]
    i0 = int(round((on - PRE_ON - SEG0) * srU))
    i1 = i0 + int(round((PRE_ON + WIN) * srU))
    i0 = max(i0, NDT)
    i1 = min(i1, ref.shape[1] - NDT - 1)
    return i0, i1, on


def align_lag(xU, refU, i0, i1, maxlag=20):
    r = refU[:, i0:i1]
    best, bl = -2, 0
    for L in range(-maxlag, maxlag + 1):
        c = xU[:, i0 + L:i1 + L]
        v = np.sum(r * c) / (np.linalg.norm(r) * np.linalg.norm(c) + 1e-30)
        if v > best:
            best, bl = v, L
    return bl, best


def station_job(args):
    sta, fams, bands, nboot, test = args[:5]
    inject = args[5] if len(args) > 5 else None
    pass  # niceness set by the caller (nice -n 10)
    t0 = time.time()
    meta = read_meta()
    cat = pd.read_csv(f'{ROOT}/data/merged/LFEcat_MTJ_merged_v1v2.csv', usecols=['time_utc', 'source'])
    allt = np.sort(epoch_s(cat[cat.source != 'v1_only'].time_utc))
    edges = month_bins()
    nb = len(edges) - 1
    rng = np.random.default_rng(zlib.crc32(sta.encode()))
    nrep = 1 + nboot + 2
    pooled = {b: np.zeros((nb, nrep, len(PHI), NDT), np.float32) for b in bands}
    pooled_n = {b: np.zeros((nb, 2)) for b in bands}       # n events, n families
    pooled_t = {b: np.zeros(nb) for b in bands}
    pooled_dpol = {b: [[] for _ in range(nb)] for b in bands}
    pooled_rel = {b: np.zeros((nb, nrep, len(PHI), 2 * RW + 1), np.float32) for b in bands}
    pooled_n_rel = {b: np.zeros(nb) for b in bands}
    refsum = {b: None for b in bands}
    fam_rows, ref_rows = [], []
    for F in fams:
        try:
            r = load_family_station(F, sta, meta, allt, bands, inject)
        except Exception as e:  # noqa
            print(f'{sta} fam{F}: load error {e}', flush=True)
            continue
        if r is None:
            continue
        data, te, S, sr = r
        for b in bands:
            seg, noi = data[b]
            ref = seg.mean(0)
            refU = up(ref)
            i0, i1, on = onset_window(refU)
            nref = noi.mean(0)
            noiseE = np.mean(nref ** 2) * (i1 - i0)   # per-sample noise power (both comps) x window length
            m0, L2ref = measure(refU, i0, i1, noiseE)
            sig = np.sqrt(np.mean(refU[:, i0:i1] ** 2) / np.mean(nref ** 2))
            # tracking reference: best minimum with dt inside [RW, NDT-1-RW] samples (interior of the grid)
            Lint = np.full_like(L2ref, np.inf)
            Lint[:, RW:NDT - RW] = L2ref[:, RW:NDT - RW]
            ipr, ikr = np.unravel_index(np.argmin(Lint), Lint.shape)
            phi_f = (PHI[ipr] + parab(np.array([L2ref[(ipr - 1) % len(PHI), ikr], L2ref[ipr, ikr],
                                                 L2ref[(ipr + 1) % len(PHI), ikr]]), 1) * 2.0) % 180
            dt_f = DT[ikr] + parab(L2ref[ipr], ikr) * 0.005
            m0['phi_int'], m0['dt_int'] = phi_f, dt_f
            m0['l2_int_over_min'] = L2ref[ipr, ikr] / L2ref.min()
            # bootstrap the reference (events resampled)
            xb = up(np.stack([seg[rng.integers(0, len(te), len(te))].mean(0) for _ in range(20)]))
            mb, _ = measure(xb, i0, i1, noiseE)
            bs = np.array([(x['phi'], x['dt'], x['pol']) for x in mb])
            ref_rows.append(dict(sta=sta, family=F, band=b, n=len(te), snr=sig, onset_rel_S=on, **m0,
                                 se_phi=axial_mean(bs[:, 0])[1], se_dt=bs[:, 1].std(),
                                 se_pol=axial_mean(bs[:, 2])[1],
                                 null_angle=abs(wrap90(m0['pol'] - m0['phi'])),
                                 t0=pd.to_datetime(te.min(), unit='s'), t1=pd.to_datetime(te.max(), unit='s')))
            refsum[b] = L2ref if refsum[b] is None else refsum[b] + L2ref
            # bins
            ib = np.searchsorted(edges, te, side='right') - 1
            for k in np.unique(ib):
                if k < 0 or k >= nb:
                    continue
                mem = np.where(ib == k)[0]
                n = len(mem)
                if n < MIN_EV:
                    continue
                W = np.zeros((nrep, n), np.float32)
                W[0] = 1.0 / n
                for j in range(nboot):
                    W[1 + j] = np.bincount(rng.integers(0, n, n), minlength=n) / n
                W[1 + nboot, 0::2] = 1.0 / len(mem[0::2])
                W[2 + nboot, 1::2] = 1.0 / max(len(mem[1::2]), 1)
                if n < 2 * 4:
                    W[1 + nboot:] = 0
                Sx = np.einsum('rn,nct->rct', W, seg[mem])
                Nx = np.einsum('rn,nct->rct', W, noi[mem])
                SU = up(Sx)
                lag, cS = align_lag(SU[0], refU, i0, i1)
                if cS < 0.3:
                    lag = 0
                nE = np.mean(Nx ** 2, axis=(1, 2)) * (i1 - i0)
                res, L2s = measure(SU, i0 + lag, i1 + lag, nE)
                for rr, mm in enumerate(res):
                    L2 = L2s[rr]
                    mm['dt_fix'] = dt_at_phi(L2, phi_f, dt_f)
                    mm['phi_fix'] = phi_at_dt(L2, dt_f, phi_f)
                    mm['phi_loc'], mm['dt_loc'], mm['edge'] = local_min(L2, phi_f, dt_f)
                    mm['snr'] = np.sqrt(np.mean(SU[rr][:, i0 + lag:i1 + lag] ** 2) / np.mean(Nx[rr] ** 2))
                pooled[b][k] += L2s.astype(np.float32)
                if RW <= ikr <= NDT - 1 - RW and sig >= 8:     # reference well inside the grid, good SNR
                    pooled_rel[b][k] += recenter(L2s, ipr, ikr).astype(np.float32)
                    pooled_n_rel[b][k] += 1
                m = res[0]
                bt = [x for x in res[1:1 + nboot]]
                row = dict(sta=sta, family=F, band=b, bin=k, t=te[mem].mean(), n=n, lag=lag / (50.0 * UP), ccS=cS,
                           **{kk: m[kk] for kk in ['phi', 'dt', 'pol', 'l2l1', 'l2min', 'phi_rc', 'dt_rc', 'rc',
                                                   'dt_fix', 'phi_fix', 'phi_loc', 'dt_loc', 'edge', 'snr']},
                           se_phi=axial_mean([x['phi'] for x in bt])[1], se_dt=np.std([x['dt'] for x in bt]),
                           se_pol=axial_mean([x['pol'] for x in bt])[1],
                           se_dt_fix=np.std([x['dt_fix'] for x in bt]),
                           se_phi_fix=axial_mean([x['phi_fix'] for x in bt])[1],
                           se_phi_loc=axial_mean([x['phi_loc'] for x in bt])[1],
                           se_dt_loc=np.std([x['dt_loc'] for x in bt]),
                           phi_ref=phi_f, dt_ref=dt_f, pol_ref=m0['pol'])
                for hname, rr in (('h0', 1 + nboot), ('h1', 2 + nboot)):
                    x = res[rr]
                    for kk in ['phi', 'dt', 'pol', 'dt_fix', 'phi_fix', 'phi_loc', 'dt_loc']:
                        row[f'{kk}_{hname}'] = np.nan if x is None else x[kk]
                fam_rows.append(row)
                pooled_n[b][k] += (n, 1)
                pooled_t[b][k] += te[mem].sum()
                pooled_dpol[b][k].append((wrap90(m['pol'] - m0['pol']),
                                           1.0 / max(row['se_pol'], 1.0) ** 2,
                                           [wrap90(x['pol'] - m0['pol']) for x in bt]))
        print(f'{sta} fam{F}: {len(te)} iso events, {time.time() - t0:.0f} s', flush=True)
        if test:
            break
    if not ref_rows:
        return sta, 0
    # ---- pooled
    prow = []
    ref_out = {}
    for b in bands:
        if refsum[b] is None:
            continue
        phi0, dt0, _, _ = best_from_surface(refsum[b])
        ref_out[f'{b}_surf'] = refsum[b]
        ref_out[f'{b}_phi0'] = phi0
        ref_out[f'{b}_dt0'] = dt0
        for k in range(nb):
            if pooled_n[b][k, 1] == 0:
                continue
            P = pooled[b][k]
            vals = []
            for rr in range(nrep):
                if not np.any(P[rr]):
                    vals.append((np.nan,) * 7)
                    continue
                ph, dt, _, _ = best_from_surface(P[rr])
                pl, dl, ed = local_min(P[rr], phi0, dt0)
                vals.append((ph, dt, dt_at_phi(P[rr], phi0, dt0), phi_at_dt(P[rr], dt0, phi0), pl, dl, ed))
            vals = np.array(vals)
            bt = vals[1:1 + nboot]
            if pooled_n_rel[b][k] > 0:
                rv = np.array([rel_measure(pooled_rel[b][k, rr]) for rr in range(nrep)], dtype=float)
            else:
                rv = np.full((nrep, 5), np.nan)
            rb = rv[1:1 + nboot]
            relcols = dict(nfam_rel=pooled_n_rel[b][k], rel_dphi_loc=rv[0, 0], rel_ddt_loc=rv[0, 1], rel_ddt_fix=rv[0, 2], rel_dphi_fix=rv[0, 3],
                           rel_edge=rv[0, 4], se_rel_dphi_loc=rb[:, 0].std(), se_rel_ddt_loc=rb[:, 1].std(),
                           se_rel_ddt_fix=rb[:, 2].std(), se_rel_dphi_fix=rb[:, 3].std())
            for hn, rr in (('h0', 1 + nboot), ('h1', 2 + nboot)):
                for j, c in enumerate(['rel_dphi_loc', 'rel_ddt_loc', 'rel_ddt_fix', 'rel_dphi_fix']):
                    relcols[f'{c}_{hn}'] = rv[rr, j]
            dp = pooled_dpol[b][k]
            w = np.array([x[1] for x in dp])
            dpol = np.sum(w * np.array([x[0] for x in dp])) / w.sum()
            dpb = np.array([[x[2][j] for x in dp] for j in range(nboot)])   # (nboot, nfam)
            dpol_b = (dpb * w[None, :]).sum(1) / w.sum()
            prow.append(dict(sta=sta, band=b, bin=k, t=pooled_t[b][k] / pooled_n[b][k, 0], n=pooled_n[b][k, 0],
                             nfam=pooled_n[b][k, 1], phi=vals[0, 0], dt=vals[0, 1], dt_fix=vals[0, 2],
                             phi_fix=vals[0, 3], se_phi=axial_mean(bt[:, 0])[1], se_dt=np.nanstd(bt[:, 1]),
                             se_dt_fix=np.nanstd(bt[:, 2]), se_phi_fix=axial_mean(bt[:, 3])[1],
                             phi_h0=vals[1 + nboot, 0], dt_h0=vals[1 + nboot, 1], dt_fix_h0=vals[1 + nboot, 2],
                             phi_fix_h0=vals[1 + nboot, 3], phi_h1=vals[2 + nboot, 0], dt_h1=vals[2 + nboot, 1],
                             dt_fix_h1=vals[2 + nboot, 2], phi_fix_h1=vals[2 + nboot, 3],
                             phi_loc=vals[0, 4], dt_loc=vals[0, 5], edge=vals[0, 6],
                             se_phi_loc=axial_mean(bt[:, 4])[1], se_dt_loc=np.nanstd(bt[:, 5]),
                             phi_loc_h0=vals[1 + nboot, 4], dt_loc_h0=vals[1 + nboot, 5],
                             phi_loc_h1=vals[2 + nboot, 4], dt_loc_h1=vals[2 + nboot, 5],
                             dpol=dpol, se_dpol=np.std(dpol_b), phi0=phi0, dt0=dt0, **relcols))
        ref_out[f'{b}_relsum'] = pooled_rel[b][:, 0].sum(0)
    tag = ('_test' if test else '') + (f"_inj{inject['name']}" if inject else '')
    os.makedirs(f'{OUT}/meas', exist_ok=True)
    pd.DataFrame(fam_rows).to_parquet(f'{OUT}/meas/{sta}{tag}_family.parquet')
    pd.DataFrame(prow).to_parquet(f'{OUT}/meas/{sta}{tag}_pooled.parquet')
    pd.DataFrame(ref_rows).to_parquet(f'{OUT}/meas/{sta}{tag}_ref.parquet')
    np.savez(f'{OUT}/meas/{sta}{tag}_refsurf.npz', **ref_out)
    print(f'{sta} DONE {time.time() - t0:.0f} s', flush=True)
    return sta, len(fam_rows)


def stage_measure(a):
    from multiprocessing import Pool
    st = pd.read_csv(f'{OUT}/stations.csv')
    stas = a.stations.split(',') if a.stations else st.sta.tolist()
    fams = [int(x) for x in a.families.split(',')] if a.families else list(range(1, 28))
    if a.skip_done:
        stas = [s for s in stas if not os.path.exists(f'{OUT}/meas/{s}_pooled.parquet')]
    jobs = [(s, fams, BANDS, a.nboot, a.test) for s in stas]
    print(f'{len(jobs)} stations, {len(fams)} families', flush=True)
    with Pool(a.procs) as p:
        for s, n in p.imap_unordered(station_job, jobs):
            print(f'== {s}: {n} family-bin rows', flush=True)


# ------------------------------------------------------------------ analysis
YR = 365.25 * 86400
T_MID = epoch_s(['2017-01-01'])[0]
QNAME = ['pet', 'fer', 'm7']
# known instrument problems excluded from the time-series fits (NOTES 2026-11-22: B045 EH1 2009-2010
# shifted by ~half a period, probably an undocumented polarity reversal; SWS sees it as ddt -50 ms, dphi +30 deg)
EXCLUDE = {'PB.B045': ('2000-01-01', '2011-01-01')}


def design(t, steps, period=1.0, extra=()):
    cols = [np.ones_like(t), (t - T_MID) / YR,
            np.cos(2 * np.pi * (t - T_MID) / (period * YR)), np.sin(2 * np.pi * (t - T_MID) / (period * YR))]
    cols += [(t >= q).astype(float) for q in steps]
    cols += [(t >= q).astype(float) for q in extra]
    return np.stack(cols, 1)


def robust_wls(G, y, e, iters=5, c=2.5):
    """Huber IRLS with weights 1/e^2; returns coef, cov scaled by reduced chi2 (>=1)."""
    w0 = 1 / e ** 2
    w = w0.copy()
    for _ in range(iters):
        Gw = G * w[:, None]
        try:
            m = np.linalg.solve(G.T @ Gw, Gw.T @ y)
        except np.linalg.LinAlgError:
            return None, None, None
        r = (y - G @ m) * np.sqrt(w0)
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        u = np.abs(r) / (c * s)
        w = w0 * np.where(u <= 1, 1, 1 / u)
    Gw = G * w[:, None]
    C = np.linalg.inv(G.T @ Gw)
    dof = max(len(y) - G.shape[1], 1)
    chi = np.sum(w * (y - G @ m) ** 2) / dof
    return m, C * max(chi, 1.0), chi


def valid_steps(t, qs=QT, need=3, span=YR):
    return [q for q in qs if np.sum((t < q) & (t > q - span)) >= need and np.sum((t >= q) & (t < q + span)) >= need]


def series_tests(t, y, e, placebo_dates, rng=None, epochs=()):
    """Fit y(t) = c + trend + annual + steps at mainshocks (+ placebo / epoch steps). Returns dict."""
    out = {}
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    t, y, e = t[ok], y[ok], e[ok]
    if len(t) < 12:
        return None
    e = np.maximum(e, np.median(e) * 0.3)
    qs = valid_steps(t)
    G = design(t, qs)
    m, C, chi = robust_wls(G, y, e)
    if m is None:
        return None
    out.update(n=len(t), t0=t.min(), t1=t.max(), chi2=chi, trend=m[1], se_trend=np.sqrt(C[1, 1]),
               ann_amp=np.hypot(m[2], m[3]), ann_peak_doy=(np.degrees(np.arctan2(m[3], m[2])) % 360) / 360 * 365.25
               + pd.Timestamp('2017-01-01').dayofyear - 1)
    for j, q in enumerate(qs):
        name = QNAME[list(QT).index(q)]
        out[f'step_{name}'] = m[4 + j]
        out[f'se_step_{name}'] = np.sqrt(C[4 + j, 4 + j])
    # placebo steps (common dates across stations): +one extra step, real steps kept
    pl = {}
    for d in placebo_dates:
        if np.any(np.abs(d - QT) < 0.25 * YR):
            continue
        if np.sum((t < d) & (t > d - YR)) < 3 or np.sum((t >= d) & (t < d + YR)) < 3:
            continue
        mm, _, _ = robust_wls(design(t, qs, extra=[d]), y, e, iters=3)
        if mm is not None:
            pl[d] = mm[-1]
    out['placebo'] = pl
    vals = np.array(list(pl.values()))
    if len(vals) >= 20:
        sd = 1.4826 * np.median(np.abs(vals - np.median(vals)))
        out['placebo_sd'] = sd
        out['placebo_n'] = len(vals)
        for q in QNAME:
            if f'step_{q}' in out:
                out[f'z_{q}'] = out[f'step_{q}'] / sd
                out[f'p_{q}'] = float(np.mean(np.abs(vals) >= abs(out[f'step_{q}'])))
    # annual placebo periods
    if t.max() - t.min() > 3 * YR:
        amps = []
        for P in np.r_[np.linspace(0.7, 0.9, 20), np.linspace(1.1, 1.4, 30)]:
            mm, _, _ = robust_wls(design(t, qs, period=P), y, e, iters=3)
            if mm is not None:
                amps.append(np.hypot(mm[2], mm[3]))
        amps = np.array(amps)
        out['ann_placebo_mean'] = amps.mean()
        out['ann_placebo_sd'] = amps.std()
        out['ann_z'] = (out['ann_amp'] - amps.mean()) / amps.std()
        out['ann_p'] = float(np.mean(amps >= out['ann_amp']))
    # long-term: mean(2025-01..2026-07) - mean(2019-07..2021-07); placebo: same geometry, pre-2021 windows
    def lt(T0):
        a = (t >= T0) & (t < T0 + 2 * YR)
        b = (t >= T0 + 5.5 * YR) & (t < T0 + 7 * YR)
        if a.sum() < 8 or b.sum() < 6:
            return np.nan
        return np.median(y[b]) - np.median(y[a])     # robust to cycle-skip outliers
    T0 = epoch_s(['2019-07-01'])[0]
    out['longterm'] = lt(T0)
    pls = [lt(T0 - k * YR / 12) for k in range(24, 140)]   # windows ending before mid-2021 + earlier
    pls = np.array([v for v, k in zip(pls, range(24, 140)) if np.isfinite(v)
                    and T0 - k * YR / 12 + 7 * YR <= epoch_s(['2021-07-01'])[0] + 1])
    if len(pls) >= 10:
        out['lt_placebo_sd'] = pls.std()
        out['lt_placebo_mean'] = pls.mean()
        out['lt_z'] = (out['longterm'] - pls.mean()) / pls.std()
        out['lt_n'] = len(pls)
    # instrument / metadata epochs
    for name, d in epochs:
        if np.sum((t < d) & (t > d - YR)) < 3 or np.sum((t >= d) & (t < d + YR)) < 3:
            continue
        mm, CC, _ = robust_wls(design(t, qs, extra=[d]), y, e)
        if mm is not None:
            out[f'epoch_{name}'] = mm[-1]
            if 'placebo_sd' in out:
                out[f'epoch_z_{name}'] = mm[-1] / out['placebo_sd']
    return out


def meta_epochs(sta):
    m = read_meta()
    g = m[(m.net + '.' + m.sta == sta) & m.cha.str.match(r'^(HH[EN]|EH[12])$')].sort_values('start')
    ev = []
    prev = {}
    for _, r in g.iterrows():
        key = r.cha
        if key in prev:
            p = prev[key]
            ch = []
            if p.az != r.az:
                ch.append(f'az{p.az:g}->{r.az:g}')
            if p.inst.split(',')[0] != r.inst.split(',')[0]:
                ch.append('sensor')
            if abs(p.scale / r.scale - 1) > 0.002:
                ch.append('gain')
            if ch and r.cha[-1] in 'N2':
                ev.append((f"{r.start:%Y-%m-%d}:{'+'.join(ch)}", epoch_s([r.start])[0]))
        prev[key] = r
    return ev


def load_meas():
    P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{OUT}/meas/*_pooled.parquet')) if '_test' not in f and '_inj' not in f])
    Fm = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{OUT}/meas/*_family.parquet')) if '_test' not in f and '_inj' not in f])
    R = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{OUT}/meas/*_ref.parquet')) if '_test' not in f and '_inj' not in f])
    return P, Fm, R


def obs_columns(df, ref_phi=None, ref_dt=None, pooled=True):
    """Return dict observable -> (y, e). dt in ms, angles in deg.
    pooled: from the re-centred pooled surfaces (each family's surface shifted to its own reference
    minimum, so the family mix cannot move the estimate); family rows: relative to the family reference."""
    d = {}
    if pooled:
        d['ddt'] = (df.rel_ddt_fix * 1e3, df.se_rel_ddt_fix * 1e3)
        d['dphi'] = (df.rel_dphi_fix, df.se_rel_dphi_fix)
        d['ddt_loc'] = (df.rel_ddt_loc * 1e3, df.se_rel_ddt_loc * 1e3)
        d['dphi_loc'] = (df.rel_dphi_loc, df.se_rel_dphi_loc)
        d['dpol'] = (df.dpol, df.se_dpol)
    else:
        d['ddt'] = ((df.dt_fix - ref_dt) * 1e3, df.se_dt_fix * 1e3)
        d['dphi'] = (wrap90(df.phi_fix - ref_phi), df.se_phi_fix)
    return d


HALFCOL = {'ddt': ('rel_ddt_fix', 1e3), 'dphi': ('rel_dphi_fix', 1), 'ddt_loc': ('rel_ddt_loc', 1e3),
           'dphi_loc': ('rel_dphi_loc', 1)}


def stage_analyze(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P, Fm, R = load_meas()
    st = pd.read_csv(f'{OUT}/stations.csv').set_index('sta')
    inc = pd.read_csv(f'{OUT}/incidence.csv')
    rng = np.random.default_rng(7)
    pdates = np.sort(rng.uniform(epoch_s(['2009-01-01'])[0], epoch_s(['2026-01-01'])[0], 300))
    summary = {}
    # ---------- reference splitting per station
    refrows = []
    for (sta, b), g in R.groupby(['sta', 'band']):
        pp = P[(P.sta == sta) & (P.band == b)]
        if len(pp) == 0:
            continue
        phi0, dt0 = pp.phi0.iloc[0], pp.dt0.iloc[0]
        good = g[(g.snr >= 8) & (g.n >= 300)]
        mphi, sphi = axial_mean(good.phi) if len(good) else (np.nan, np.nan)
        ii = inc[inc.sta == sta]
        refrows.append(dict(sta=sta, band=b, borehole=sta.startswith('PB.'), phi0=phi0, dt0_ms=dt0 * 1e3,
                            n_fam=len(good), fam_phi_mean=mphi, fam_phi_sd=sphi,
                            fam_dt_med_ms=good.dt.median() * 1e3 if len(good) else np.nan,
                            fam_dt_iqr_ms=(good.dt.quantile(.75) - good.dt.quantile(.25)) * 1e3 if len(good) else np.nan,
                            frac_fam_within20=float(np.mean(np.abs(wrap90(good.phi - phi0)) <= 20)) if len(good) else np.nan,
                            frac_fam_dt_within30ms=float(np.mean(np.abs(good.dt - dt0) <= 0.03)) if len(good) else np.nan,
                            null_angle_med=good.null_angle.median() if len(good) else np.nan,
                            l2l1_med=good.l2l1.median() if len(good) else np.nan,
                            snr_med=good.snr.median() if len(good) else np.nan,
                            dist_km=ii.dist_km.median(), inc_1d=ii.inc_1d.median(), inc_straight=ii.inc_straight.median(),
                            sensor_depth_m=st.loc[sta, 'depth'] if sta in st.index else np.nan,
                            t0=pp.t.min(), t1=pp.t.max(), n_bins=len(pp)))
    REF = pd.DataFrame(refrows)
    REF['t0'] = pd.to_datetime(REF.t0, unit='s')
    REF['t1'] = pd.to_datetime(REF.t1, unit='s')
    REF.to_csv(f'{OUT}/reference_splitting.csv', index=False)
    print(REF.round(2).to_string())
    # ---------- pooled per-station time series tests
    rows, plac_store = [], {}
    for (sta, b), g in P.groupby(['sta', 'band']):
        g = g.sort_values('t')
        g = g[(g.n >= 20)]
        if len(g) < 12:
            continue
        ep = meta_epochs(sta)
        g = g[g.nfam_rel > 0]
        if sta in EXCLUDE:
            x0, x1 = epoch_s(list(EXCLUDE[sta]))
            g = g[(g.t < x0) | (g.t >= x1)]
        if len(g) < 12:
            continue
        for ob, (y, e) in obs_columns(g).items():
            r = series_tests(g.t.values, np.asarray(y, float), np.asarray(e, float), pdates, epochs=ep)
            if r is None:
                continue
            plac_store[(sta, b, ob)] = r.pop('placebo')
            # odd/even halves
            hh = {}
            if ob in HALFCOL:
                col, sc_ = HALFCOL[ob]
                for hn in ('h0', 'h1'):
                    yh = g[f'{col}_{hn}'].values * sc_
                    eh = np.asarray(e, float) * np.sqrt(2)
                    rh = series_tests(g.t.values, np.asarray(yh, float), eh, [], epochs=())
                    if rh is not None:
                        hh[hn] = rh
                        hh[hn + '_y'] = yh
            if 'h0' in hh and 'h1' in hh:
                for q in QNAME:
                    if f'step_{q}' in hh['h0'] and f'step_{q}' in hh['h1']:
                        r[f'h0_step_{q}'] = hh['h0'][f'step_{q}']
                        r[f'h1_step_{q}'] = hh['h1'][f'step_{q}']
                a0, a1 = np.asarray(hh['h0_y'], float), np.asarray(hh['h1_y'], float)
                ok = np.isfinite(a0) & np.isfinite(a1)
                # correlation of half-series anomalies (a common real signal makes them correlate)
                if ok.sum() > 10:
                    r['half_r'] = float(np.corrcoef(a0[ok] - np.median(a0[ok]), a1[ok] - np.median(a1[ok]))[0, 1])
                    r['half_n'] = int(ok.sum())
            yv = np.asarray(y, float)
            r.update(sta=sta, band=b, obs=ob, borehole=sta.startswith('PB.'), n_bins=len(g),
                     med_se=float(np.nanmedian(np.asarray(e, float))), rms=float(np.nanstd(yv)),
                     epochs=';'.join(x[0] for x in ep))
            rows.append(r)
    TS = pd.DataFrame(rows)
    TS.to_csv(f'{OUT}/station_tests.csv', index=False)
    # ---------- network pooling (surface / borehole / all) with common placebo dates
    net_rows = []
    for (b, ob), g in TS.groupby(['band', 'obs']):
        for grp, gg in (('all', g), ('borehole', g[g.borehole]), ('surface', g[~g.borehole])):
            for q in QNAME:
                h = gg[gg[f'step_{q}'].notna() & gg.placebo_sd.notna()] if f'step_{q}' in gg else gg.iloc[0:0]
                if len(h) == 0:
                    continue
                w = 1 / h.placebo_sd ** 2
                est = np.sum(w * h[f'step_{q}']) / w.sum()
                # placebo: same weighted mean at each common placebo date
                pv = []
                for d in pdates:
                    vals, ws = [], []
                    for _, rr in h.iterrows():
                        pl = plac_store[(rr.sta, b, ob)]
                        if d in pl:
                            vals.append(pl[d])
                            ws.append(1 / rr.placebo_sd ** 2)
                    if len(vals) >= max(1, int(0.6 * len(h))):
                        pv.append(np.sum(np.array(ws) * np.array(vals)) / np.sum(ws))
                pv = np.array(pv)
                sd = 1.4826 * np.median(np.abs(pv - np.median(pv))) if len(pv) >= 20 else np.nan
                hz = h[[f'h0_step_{q}', f'h1_step_{q}']].dropna() if f'h0_step_{q}' in h else pd.DataFrame()
                wh = (1 / h.loc[hz.index].placebo_sd ** 2) if len(hz) else None
                net_rows.append(dict(band=b, obs=ob, group=grp, quake=q, n_sta=len(h), step=est,
                                     placebo_sd=sd, z=est / sd if sd else np.nan,
                                     p=float(np.mean(np.abs(pv) >= abs(est))) if len(pv) else np.nan, n_placebo=len(pv),
                                     h0=np.sum(wh * hz.iloc[:, 0]) / wh.sum() if len(hz) else np.nan,
                                     h1=np.sum(wh * hz.iloc[:, 1]) / wh.sum() if len(hz) else np.nan,
                                     stations=','.join(h.sta)))
            # annual & long-term pooled
            h = gg[gg.ann_z.notna()] if 'ann_z' in gg else gg.iloc[0:0]
            if len(h):
                net_rows.append(dict(band=b, obs=ob, group=grp, quake='annual', n_sta=len(h),
                                     step=h.ann_amp.median(), z=h.ann_z.sum() / np.sqrt(len(h)),
                                     p=np.nan, stations=','.join(h.sta)))
            h = gg[gg.lt_z.notna()] if 'lt_z' in gg else gg.iloc[0:0]
            if len(h):
                w = 1 / h.lt_placebo_sd ** 2
                net_rows.append(dict(band=b, obs=ob, group=grp, quake='longterm', n_sta=len(h),
                                     step=np.sum(w * h.longterm) / w.sum(), z=h.lt_z.sum() / np.sqrt(len(h)),
                                     p=np.nan, stations=','.join(h.sta)))
    NET = pd.DataFrame(net_rows)
    NET.to_csv(f'{OUT}/network_tests.csv', index=False)
    print(NET[NET.band == 'b28'].drop(columns='stations').round(3).to_string())
    # ---------- family x station decomposition (receiver vs source side) for each quake
    fs_rows = []
    for b in BANDS:
        Fb = Fm[Fm.band == b]
        okref = set(map(tuple, R[(R.band == b) & (R.snr >= 8)][['sta', 'family']].values))
        Fb = Fb[[(x, y) in okref for x, y in zip(Fb.sta, Fb.family)]]
        for ob in ['ddt', 'dphi']:
            est = []
            for (sta, F), g in Fb.groupby(['sta', 'family']):
                g = g.sort_values('t')
                g = g[g.n >= 15]
                if sta in EXCLUDE:
                    x0, x1 = epoch_s(list(EXCLUDE[sta]))
                    g = g[(g.t < x0) | (g.t >= x1)]
                if len(g) < 24:
                    continue
                y, e = obs_columns(g, g.phi_ref.iloc[0], g.dt_ref.iloc[0], pooled=False)[ob]
                r = series_tests(g.t.values, np.asarray(y, float), np.asarray(e, float), pdates[::3])
                if r is None or 'placebo_sd' not in r:
                    continue
                d = dict(sta=sta, family=F, placebo_sd=r['placebo_sd'])
                for q in QNAME:
                    d[q] = r.get(f'step_{q}', np.nan)
                for dd, v in r['placebo'].items():
                    d[f'pl_{dd:.0f}'] = v
                est.append(d)
            E = pd.DataFrame(est)
            if len(E) == 0:
                continue
            E.drop(columns=[c for c in E if c.startswith('pl_')]).assign(band=b, obs=ob).to_csv(
                f'{OUT}/family_station_steps_{b}_{ob}.csv', index=False)

            def fstats(col):
                h = E[E[col].notna()]
                z = h[col] / h.placebo_sd
                if len(h) < 6:
                    return np.nan, np.nan, len(h)
                ms = z.groupby(h.sta).agg(['mean', 'count'])
                mf = z.groupby(h.family).agg(['mean', 'count'])
                Fs = np.sum(ms['count'] * ms['mean'] ** 2) / max(len(ms), 1)
                Ff = np.sum(mf['count'] * mf['mean'] ** 2) / max(len(mf), 1)
                return Fs, Ff, len(h)
            plc = [c for c in E if c.startswith('pl_')]
            for q in QNAME:
                Fs, Ff, nn = fstats(q)
                if not np.isfinite(Fs):
                    continue
                PS, PF = [], []
                for c in plc:
                    a_, b_, n_ = fstats(c)
                    if np.isfinite(a_) and n_ >= 0.6 * nn:
                        PS.append(a_)
                        PF.append(b_)
                PS, PF = np.array(PS), np.array(PF)
                fs_rows.append(dict(band=b, obs=ob, quake=q, n_famsta=nn, n_sta=E[E[q].notna()].sta.nunique(),
                                    n_fam=E[E[q].notna()].family.nunique(), F_station=Fs, F_family=Ff,
                                    p_station=float(np.mean(PS >= Fs)) if len(PS) else np.nan,
                                    p_family=float(np.mean(PF >= Ff)) if len(PF) else np.nan, n_placebo=len(PS)))
    FS = pd.DataFrame(fs_rows)
    FS.to_csv(f'{OUT}/family_vs_station.csv', index=False)
    print(FS.round(3).to_string())
    summary['reference'] = REF[REF.band == 'b28'][['sta', 'phi0', 'dt0_ms', 'fam_phi_sd', 'n_fam', 'inc_1d',
                                                   'borehole']].round(2).to_dict('records')
    summary['network'] = NET.drop(columns='stations').round(4).to_dict('records')
    summary['family_vs_station'] = FS.round(4).to_dict('records')
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    make_figures(P, R, REF, TS, NET, plt)


def make_figures(P, R, REF, TS, NET, plt):
    qd = pd.to_datetime(list(QUAKES.values()))
    fl = [json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json')) for F in range(1, 28)]
    st = pd.read_csv(f'{OUT}/stations.csv').set_index('sta')
    # 1. reference map
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.5))
    for ax, b in zip(axs[:2], BANDS):
        ax.plot([x['lon'] for x in fl], [x['lat'] for x in fl], 'k.', ms=3)
        for _, r in REF[REF.band == b].iterrows():
            if r.sta not in st.index:
                continue
            lo, la = st.loc[r.sta, 'lon'], st.loc[r.sta, 'lat']
            L = 0.08 * r.dt0_ms / 100
            ph = np.radians(r.phi0)
            dx, dy = L * np.sin(ph) / np.cos(np.radians(la)), L * np.cos(ph)
            c = 'tab:red' if r.borehole else 'tab:blue'
            ax.plot([lo - dx, lo + dx], [la - dy, la + dy], c=c, lw=3 if r.fam_phi_sd < 20 else 1.2)
            ax.plot(lo, la, 'k^', ms=4)
            ax.text(lo, la, ' ' + r.sta.split('.')[1], fontsize=6)
        ax.set_title(f'{b}: station fast direction (bar length ~ dt; red = PB borehole)\nthick: families agree (sd<20 deg)',
                     fontsize=9)
        ax.set_aspect(1 / np.cos(np.radians(40.1)))
    ax = axs[2]
    g = REF[REF.band == 'b28']
    for bh, c in ((True, 'tab:red'), (False, 'tab:blue')):
        h = g[g.borehole == bh]
        ax.errorbar(h.dist_km, h.dt0_ms, yerr=h.fam_dt_iqr_ms / 2, fmt='o', c=c,
                    label='PB borehole' if bh else 'surface')
        for _, r in h.iterrows():
            ax.text(r.dist_km, r.dt0_ms, ' ' + r.sta.split('.')[1], fontsize=6)
    ax.set_xlabel('epicentral distance (km)')
    ax.set_ylabel('pooled reference dt (ms), bar = family IQR/2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG}/sci_sws_reference.png', dpi=110)
    plt.close()
    # 2. time series (b28): dt_fix, dphi_fix, dpol for stations with >= 36 bins
    g = P[(P.band == 'b28') & (P.n >= 20) & (P.nfam_rel > 0)]
    stas = [s for s, h in g.groupby('sta') if len(h) >= 36]
    stas = sorted(stas, key=lambda s: (not s.startswith('PB'), s))
    fig, axs = plt.subplots(len(stas), 3, figsize=(18, 1.7 * len(stas)), sharex=True, squeeze=False)
    for i, s in enumerate(stas):
        h = g[g.sta == s].sort_values('t')
        tt = pd.to_datetime(h.t, unit='s')
        obs = obs_columns(h)
        for j, ob in enumerate(['ddt', 'dphi', 'dpol']):
            y, e = obs[ob]
            ax = axs[i, j]
            ax.errorbar(tt, y, yerr=e, fmt='.', ms=2, lw=0.4, c='0.4')
            yy = pd.Series(np.asarray(y, float), index=tt).rolling(6, center=True, min_periods=3).median()
            ax.plot(tt, yy, c='tab:orange', lw=1.2)
            for q in qd:
                ax.axvline(q, c='r', lw=0.7, ls='--')
            lim = np.nanpercentile(np.abs(np.asarray(y, float)), 97) * 1.2
            ax.set_ylim(-lim, lim)
            ax.text(0.01, 0.85, f'{s} {ob}' + (' (ms)' if 'dt' in ob else ' (deg)'), transform=ax.transAxes, fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{FIG}/sci_sws_timeseries.png', dpi=90)
    plt.close()
    # 3. step z-scores per station and network
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for i, ob in enumerate(['ddt', 'dphi']):
        for j, q in enumerate(QNAME):
            ax = axs[i, j]
            h = TS[(TS.band == 'b28') & (TS.obs == ob)]
            if f'z_{q}' not in h:
                continue
            h = h[h[f'z_{q}'].notna()].sort_values('sta')
            c = ['tab:red' if x else 'tab:blue' for x in h.borehole]
            ax.barh(range(len(h)), h[f'z_{q}'], color=c)
            ax.set_yticks(range(len(h)))
            ax.set_yticklabels(h.sta, fontsize=6)
            ax.axvline(-2, c='k', ls=':')
            ax.axvline(2, c='k', ls=':')
            nn = NET[(NET.band == 'b28') & (NET.obs == ob) & (NET.quake == q)]
            ttl = '; '.join(f"{r.group} {r.step:+.2f} (z {r.z:+.1f})" for _, r in nn.iterrows())
            ax.set_title(f'{ob} step at {q}: z vs placebo\n{ttl}', fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{FIG}/sci_sws_steps.png', dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['meta', 'measure', 'analyze', 'inject'])
    ap.add_argument('--stations', default=None)
    ap.add_argument('--families', default=None)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--nboot', type=int, default=NBOOT)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--skip-done', action='store_true')
    ap.add_argument('--inj-date', default='2016-07-01')
    ap.add_argument('--ddt', type=float, default=0.0, help='injected extra slow delay (s)')
    ap.add_argument('--rot', type=float, default=0.0, help='injected sensor rotation (deg)')
    ap.add_argument('--name', default='x')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage == 'meta':
        stage_meta(a)
    elif a.stage == 'measure':
        stage_measure(a)
    elif a.stage == 'inject':
        from multiprocessing import Pool
        jobs = []
        for sta in a.stations.split(','):
            z = np.load(f'{OUT}/meas/{sta}_refsurf.npz')
            inj = dict(t=epoch_s([a.inj_date])[0], ddt=a.ddt, rot=a.rot, phi=float(z['b28_phi0']), name=a.name)
            jobs.append((sta, [int(x) for x in a.families.split(',')], {'b28': BANDS['b28']}, a.nboot, False, inj))
        with Pool(a.procs) as p:
            for s_, n in p.imap_unordered(station_job, jobs):
                print(f'== {s_}: {n}', flush=True)
    else:
        stage_analyze(a)


if __name__ == '__main__':
    main()
