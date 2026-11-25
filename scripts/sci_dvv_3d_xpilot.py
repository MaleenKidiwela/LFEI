#!/usr/bin/env python3
"""PILOT: inter-station ambient-noise cross-correlations (pairs < 40 km) for the 3-D dv/v inversion.

Pilot period 2023-06-01 .. 2025-06-30, all stations of the 20-Hz archive /data/wsd02/maleen_data/MTJ_20Hz
(manifest.csv).  Conventions of the single-station SC pipeline (scripts/sci_noise_dvv_sc_full.py = S,
scripts/sci_noise_dvv_jewt.py = J; both imported, NOT modified):

  per station-day (archive day file, 20 Hz int32, dominant location/instrument, Z/N/E or Z/1/2 = S._cmap):
    * sub-sample alignment: each trace is shifted (FFT phase shift) onto the UTC 0.05-s grid (the archive
      start times are offset 0-13 ms from the grid and the offset can change at gaps; harmless for
      single-station autos/SC, but a time shift between two stations biases inter-station stretching);
      the 5-Hz branch starts on the 0.2-s grid before J._to_low (2-Hz low-pass, decimate 4).
    * 'day': 00-24 UTC, 120-s segments, 20 Hz, bands 1-3 / 3-5 Hz:  demean, 5% taper, 4-pole bandpass,
      transient rejection (segment RMS < 4 x median RMS, per component), one-bit, pure-phase whitening
      with the J._cos_taper_band window   (= S.corr_sc cross-component branch)
    * 'low': 600-s segments at 5 Hz, 0.1-1 Hz, same steps (= S.corr_sc on J._to_low)
    whitened spectra are kept in memory only (never written).
  per pair-day (both stations present): all 9 component pairs (A comp x B comp, ZZ ZN ZE NZ NN NE EZ EN EE,
    N/E = 1/2 for the PB boreholes: mixed orientation is fine because each component pair is stretched
    separately and only dv/v values are combined), segment-mean cross-spectrum (mask = both components
    good) -> irfft -> lags +-60 s (day, 20 Hz) / +-150 s (low, 5 Hz).  Positive lag = A later than B
    (wave B -> A), with A < B alphabetically.  float16 / trace + float32 scale + segment counts, one npz
    per day:  /data/wsd02/maleen_data/dvv_3d/xpilot/days/YYYY-MM-DD.npz  (keys 'A|B|day', ...).

  dv/v per pair (stage dvv): >= 20 segments per day (as S.load_sc); persistent spectral lines
  (S.detect_lines_sc on the 9 component CCFs, >= 2 of the 3 pilot years); J.notch_g; sub-bands
  0.1-0.3 / 0.3-0.6 / 0.5-1 Hz by S.subband of the 0.1-1 Hz CCFs; reference = mean of 2023-06-01..
  2024-11-30; 10-day BACKWARD moving stacks with 1-day step (>= 5 days); stretching on the S.Stretcher
  numerics (M.EPS +-2 %, 4x upsampling, parabolic refinement) on ONE side of the lag axis: causal
  (lag > 0), acausal (lag < 0, time-reversed) and symmetric (average of both sides); Weaver errors;
  component-pair quality = median CC of non-overlapping 10-day stacks vs the reference in the reference
  period >= 0.3; inverse-variance combination of the good component pairs (= SC rule).
  Coda window per pair and band: t1 = max(2 * dist / v_group, base t1), t2 = min(t1 + base length,
  0.92 * maxlag); band dropped for the pair if t2 - t1 < 0.4 x base length.

Stages:  corr [--ndays N --workers 2]  |  vel  |  dvv [--workers 2]  |  summary
Small outputs: data/sci/dvv_3d/xpilot/ ; figure figures/sci_dvv_3d_xpilot.png
"""
import argparse
import itertools
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, '/home/seismic/LFE/scripts')
import sci_noise_dvv_sc_full as S      # noqa: E402  (NOT modified)
J = S.J
M = S.M

ROOT = '/home/seismic/LFE'
ARCH = '/data/wsd02/maleen_data/MTJ_20Hz'
BULK = '/data/wsd02/maleen_data/dvv_3d/xpilot'
DAYS_DIR = os.path.join(BULK, 'days')
SER_DIR = os.path.join(BULK, 'series')
OUT = os.path.join(ROOT, 'data/sci/dvv_3d/xpilot')
FIG = os.path.join(ROOT, 'figures')
D0, D1 = '2023-06-01', '2025-06-30'
REF0, REF1 = pd.Timestamp('2023-06-01'), pd.Timestamp('2024-11-30')
M7 = pd.Timestamp('2024-12-05')
MAXDIST = 40.0
CP = [a + b for a in 'ZNE' for b in 'ZNE']          # 9 component pairs (A comp, B comp)
DAY_BANDS = J.DAY20_BANDS                            # (1,3), (3,5)
LOW_BANDS = J.LOW_BANDS                              # (0.1,1)
FS_D, SEG_D, ML_D = M.FS, M.SEG, M.MAXLAG            # 20 Hz, 120 s, 60 s
FS_L, SEG_L, ML_L = J.LOW_FS, J.LOW_SEG, J.LOW_MAXLAG  # 5 Hz, 600 s, 150 s

# base coda windows = single-station SC conventions (J.CODA_NEW, S.CODA_SUB)
BASE = {'0.1-1Hz': (10.0, 60.0), '0.1-0.3Hz': (20.0, 120.0), '0.3-0.6Hz': (10.0, 60.0), '0.5-1Hz': (5.0, 40.0),
        '1-3Hz': (3.0, 30.0), '3-5Hz': (1.0, 8.0)}
# group velocities (km/s) for the ballistic exclusion t1 >= 2 dist / v (conservative; checked by stage vel)
VG = {'0.1-1Hz': 2.0, '0.1-0.3Hz': 2.5, '0.3-0.6Hz': 2.0, '0.5-1Hz': 1.8, '1-3Hz': 1.5, '3-5Hz': 1.5}
BANDS = [('low', '0.1-1Hz'), ('low_a', '0.1-0.3Hz'), ('low_b', '0.3-0.6Hz'), ('low_c', '0.5-1Hz'),
         ('day', '1-3Hz'), ('day', '3-5Hz')]
SIDES = ['c', 'a', 's']


def prods():
    """J.Prod objects on the stored lag axes (9 component pairs)."""
    P = {'day': J.Prod('day', 'day', [0, 1], DAY_BANDS, FS_D, ML_D, BASE, mute0=0.75),
         'low': J.Prod('low', 'low', [0], LOW_BANDS, FS_L, ML_L, BASE, mute0=0.75, nf=4096,
                       line_med_hz=0.05, line_dil_hz=0.005)}
    for k, b in S.SUB.items():
        P[k] = J.Prod(k, 'low', [0], [b], FS_L, ML_L, BASE, mute0=0.75, nf=4096, line_med_hz=0.05, line_dil_hz=0.005)
    return P


# ----------------------------------------------------------------------------- stations / pairs
def stations():
    s = pd.read_csv(os.path.join(ROOT, 'data/sci/dvv_3d/stations.csv'))
    return s.set_index('ns')[['lat', 'lon']]


def dist_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def all_pairs(maxd=None):
    st = stations()
    rows = []
    for a, b in itertools.combinations(sorted(st.index), 2):
        d = dist_km(st.loc[a].values, st.loc[b].values)
        if maxd is None or d < maxd:
            rows.append((a, b, d))
    return pd.DataFrame(rows, columns=['a', 'b', 'dist_km'])


def manifest():
    m = pd.read_csv(os.path.join(ARCH, 'manifest.csv'))
    return m[m.status == 'ok']


# ----------------------------------------------------------------------------- correlation
def align(st, day, fs):
    """shift each trace onto the UTC grid of spacing 1/fs (FFT phase shift of the sub-sample offset)."""
    from obspy import UTCDateTime
    t0 = UTCDateTime(day)
    for tr in st:
        off = (tr.stats.starttime - t0) * fs
        n0 = int(np.round(off))
        frac = off - n0
        if abs(frac) > 1e-4:
            from scipy import fft as sfft
            x = tr.data.astype('float64')
            n = len(x)
            nf = sfft.next_fast_len(n + 64, real=True)        # zero-padded (no wrap-around of the ends)
            f = np.fft.rfftfreq(nf, 1.0)
            tr.data = sfft.irfft(sfft.rfft(x - x.mean(), nf) * np.exp(-2j * np.pi * f * frac), nf)[:n] + x.mean()
        tr.stats.starttime = t0 + n0 / fs
    return st


def to_low(st20, day):
    """J._to_low after trimming each trace to start on the 0.2-s grid."""
    from obspy import UTCDateTime
    t0 = UTCDateTime(day)
    st = st20.copy()
    for tr in st:
        n0 = int(np.round((tr.stats.starttime - t0) * FS_D))
        k = (-n0) % int(FS_D / FS_L)
        if k:
            tr.data = tr.data[k:]
            tr.stats.starttime = t0 + (n0 + k) / FS_D
    st.traces = [tr for tr in st if tr.stats.npts > 100]
    for tr in st:
        tr.data = np.asarray(tr.data, dtype='float64')
    return J._to_low(st)


def whiten(arr, bands, fs):
    """S.corr_sc preprocessing (cross branch) -> per band (Y[3, nseg, nbin] complex64 on band bins, good[3, nseg],
    bin index, nfft)."""
    from scipy.signal import butter, sosfiltfilt
    ns = arr.shape[-1]
    nfft = 2 * ns
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    good = np.isfinite(arr).all(axis=2) & (np.nanstd(arr, axis=2) > 0)
    x = np.where(good[..., None], arr, 0.0)
    x = x - x.mean(axis=2, keepdims=True)
    tap = np.ones(ns)
    nt = int(0.05 * ns)
    tap[:nt] = 0.5 * (1 - np.cos(np.pi * np.arange(nt) / nt))
    tap[-nt:] = tap[:nt][::-1]
    out = []
    for f1, f2 in bands:
        sos = butter(4, [f1, f2], btype='band', fs=fs, output='sos')
        y = sosfiltfilt(sos, x * tap, axis=2)
        rms = np.sqrt((y ** 2).mean(axis=2))
        gb = good.copy()
        for ic in range(3):
            if gb[ic].any():
                gb[ic] &= rms[ic] < 4 * np.median(rms[ic, gb[ic]])
        y = np.sign(y) * tap
        W = J._cos_taper_band(freqs, f1, f2, fs)
        idx = np.where(W > 0)[0]
        Y = np.fft.rfft(y, nfft, axis=2)[..., idx]
        Y = (W[idx] * Y / (np.abs(Y) + 1e-10)).astype('complex64')
        out.append((Y, gb, idx, nfft))
        del y
    return out


def prep_station(path, day):
    from obspy import read
    st = read(path)
    if len(st) == 0:
        return None
    loc, cm = S._cmap(st)
    st = st.select(location=loc)
    st = align(st, day, FS_D)
    hours = sum(tr.stats.npts / FS_D for tr in st.select(channel=cm['Z'])) / 3600
    wd = whiten(S._segments(st, day, cm, FS_D, SEG_D), DAY_BANDS, FS_D)
    stl = to_low(st, day)
    wl = whiten(S._segments(stl, day, cm, FS_L, SEG_L), LOW_BANDS, FS_L)
    return dict(day=wd, low=wl, chans=f'{loc}.' + ','.join(cm[c] for c in 'ZNE'), hours=hours)


def xcorr(wa, wb, maxlag, fs):
    """-> CCF (B, 9, 2 nl + 1) float32, n (B, 9)."""
    nl = int(maxlag * fs)
    B = len(wa)
    out = np.full((B, 9, 2 * nl + 1), np.nan, dtype='float32')
    nn = np.zeros((B, 9), dtype='int16')
    for b in range(B):
        Ya, ga, idx, nfft = wa[b]
        Yb, gb, _, _ = wb[b]
        full = np.zeros(nfft // 2 + 1, dtype='complex128')
        for k, (ca, cb) in enumerate(CP):
            ia, ib = 'ZNE'.index(ca), 'ZNE'.index(cb)
            m = ga[ia] & gb[ib]
            if m.sum() < 10:
                continue
            full[idx] = (Ya[ia, m] * np.conj(Yb[ib, m])).mean(axis=0)
            C = np.fft.irfft(full, nfft)
            out[b, k] = np.concatenate([C[-nl:], C[:nl + 1]])
            nn[b, k] = m.sum()
    return out, nn


def _day_job(job):
    dstr, avail, pairs = job
    fn = os.path.join(DAYS_DIR, f'{dstr}.npz')
    if os.path.exists(fn):
        return dstr, 'have', [], []
    day = pd.Timestamp(dstr)
    need = sorted({s for a, b in pairs for s in (a, b) if a in avail and b in avail})
    W, tprep = {}, []
    for s in need:
        c0, w0 = time.process_time(), time.time()
        try:
            r = prep_station(avail[s], day)
        except Exception as e:
            r = None
            print('PREPFAIL', s, dstr, str(e)[:80], flush=True)
        if r is not None:
            W[s] = r
        tprep.append((s, time.process_time() - c0, time.time() - w0, r['hours'] if r else 0.0))
    data, tpair = {}, []
    for a, b in pairs:
        if a not in W or b not in W:
            continue
        c0 = time.process_time()
        cd, nd = xcorr(W[a]['day'], W[b]['day'], ML_D, FS_D)
        cl, nl_ = xcorr(W[a]['low'], W[b]['low'], ML_L, FS_L)
        qd, sd = S._q16(cd)
        ql, sl = S._q16(cl)
        k = f'{a}|{b}'
        data.update({k + '|day': qd, k + '|day_s': sd, k + '|day_n': nd, k + '|low': ql, k + '|low_s': sl,
                     k + '|low_n': nl_})
        tpair.append((a, b, time.process_time() - c0))
    data['chans'] = np.array([f'{s}:{W[s]["chans"]}' for s in sorted(W)])
    tmp = fn + '.tmp.npz'
    np.savez(tmp, **data)
    os.replace(tmp, fn)
    return dstr, f'ok {len(W)} sta {len(tpair)} pairs', tprep, tpair


def stage_corr(args):
    from multiprocessing import Pool
    os.makedirs(DAYS_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    m = manifest()
    m = m[(m.day >= D0) & (m.day <= D1)]
    P = all_pairs(MAXDIST)
    pairs = [(a, b) for a, b in zip(P.a, P.b)]
    byday = {d: dict(zip(g.station, g.path)) for d, g in m.groupby('day')}
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(D0, D1)]
    if args.ndays:
        # spread test days over the pilot
        days = days[::max(1, len(days) // args.ndays)][:args.ndays]
    jobs = [(d, byday.get(d, {}), pairs) for d in days]
    tlog = os.path.join(OUT, 'cost_station_day.csv')
    plog = os.path.join(OUT, 'cost_pair_day.csv')
    new = not os.path.exists(tlog)
    ft, fp = open(tlog, 'a'), open(plog, 'a')
    if new:
        ft.write('day,sta,cpu_s,wall_s,hours\n')
        fp.write('day,a,b,cpu_s\n')
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for i, (d, st_, tp, tq) in enumerate(pool.imap_unordered(_day_job, jobs)):
            for s, c, w, h in tp:
                ft.write(f'{d},{s},{c:.3f},{w:.3f},{h:.2f}\n')
            for a, b, c in tq:
                fp.write(f'{d},{a},{b},{c:.4f}\n')
            ft.flush()
            fp.flush()
            sz = os.path.getsize(os.path.join(DAYS_DIR, f'{d}.npz')) / 1e6 if os.path.exists(os.path.join(DAYS_DIR, f'{d}.npz')) else 0
            print(f'{d} {st_} {sz:.1f} MB  [{i + 1}/{len(jobs)}, {(time.time() - t0) / 60:.1f} min]', flush=True)


# ----------------------------------------------------------------------------- load per pair
def load_pair(a, b, days):
    """-> ccf_day (D, 2, 9, 2401), ccf_low (D, 1, 9, 1501) float32, n < 20 segments -> NaN."""
    P = prods()
    cd = np.full((len(days), 2, 9, len(P['day'].lags)), np.nan, dtype='float32')
    cl = np.full((len(days), 1, 9, len(P['low'].lags)), np.nan, dtype='float32')
    k = f'{a}|{b}'
    for i, d in enumerate(days):
        fn = os.path.join(DAYS_DIR, f'{d:%Y-%m-%d}.npz')
        if not os.path.exists(fn):
            continue
        z = np.load(fn)
        if k + '|day' not in z.files:
            continue
        for key, arr in (('day', cd), ('low', cl)):
            c = z[k + '|' + key].astype('float32') * z[k + '|' + key + '_s'][..., None]
            c[z[k + '|' + key + '_n'] < 20] = np.nan
            arr[i] = c
    return cd, cl


# ----------------------------------------------------------------------------- stage vel
def stage_vel(args):
    """envelope-peak lag of the reference-period stacks vs distance (both sides, ZZ and all 9 summed in
    envelope) -> apparent group velocity per band; used to check the VG choice."""
    from scipy.signal import hilbert
    files = sorted(f for f in os.listdir(DAYS_DIR) if f.endswith('.npz') and '.tmp' not in f)
    days = pd.DatetimeIndex([f[:10] for f in files])
    days = days[days <= REF1]
    PR = prods()
    rows = []
    for _, r in all_pairs(MAXDIST).iterrows():
        cd, cl = load_pair(r.a, r.b, days)
        if np.isfinite(cl[:, 0, 0, 0]).sum() < 20:
            continue
        for key, bn in BANDS:
            P = PR[key]
            if key == 'day':
                b = P.names.index(bn)
                X = cd[:, b:b + 1]
            elif key == 'low':
                X = cl
            else:
                X = S.subband(cl, *S.SUB[key])
            with np.errstate(invalid='ignore'):
                ref = np.nanmean(X[:, 0], axis=0)           # (9, L)
            if not np.isfinite(ref).all():
                continue
            env = np.abs(hilbert(ref, axis=-1))
            env = env / np.median(env, axis=-1, keepdims=True)
            e = env.sum(axis=0)
            lags = P.lags
            for side, sel in (('c', lags > 0.5), ('a', lags < -0.5)):
                i = np.argmax(np.where(sel, e, 0))
                rows.append(dict(a=r.a, b=r.b, dist_km=r.dist_km, band=bn, side=side, peak_lag_s=abs(lags[i]),
                                 peak_over_median=e[i] / np.median(e), ndays=int(np.isfinite(X[:, 0, 0, 0]).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'vel_check.csv'), index=False)
    for bn, g in df.groupby('band'):
        gg = g[(g.peak_over_median > 4) & (g.dist_km > 8)]
        v = gg.dist_km / gg.peak_lag_s
        print(f'{bn:10s} n {len(gg):3d}/{len(g)}  apparent v median {v.median():.2f} km/s, 10-90% {v.quantile(.1):.2f}-{v.quantile(.9):.2f}'
              f'  (VG used {VG[bn]})', flush=True)


# ----------------------------------------------------------------------------- dv/v
class SideStretcher(S.Stretcher):
    """S.Stretcher numerics, coda mask on POSITIVE lags only (the side is selected by time reversal)."""
    def stretch(self, P, coda, ref, X, ck):
        from scipy.signal import resample
        X = np.atleast_2d(X)
        out_e = np.full(len(X), np.nan)
        out_c = np.full(len(X), np.nan)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(ref).all()
        if not ok.any():
            return out_e, out_c
        mu = (P.lags_u >= coda[0]) & (P.lags_u <= coda[1])
        if ck not in self.cache:
            t = P.lags_u[mu]
            refu = resample(ref, len(P.lags_u))
            R = np.array([np.interp(t * (1 + e), P.lags_u, refu) for e in M.EPS])
            R -= R.mean(axis=1, keepdims=True)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            self.cache[ck] = R
        R = self.cache[ck]
        Y = resample(X[ok].astype('float64'), len(P.lags_u), axis=1)[:, mu]
        Y = Y - Y.mean(axis=1, keepdims=True)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        C = R @ Y.T
        j = np.argmax(C, axis=0)
        jj = np.clip(j, 1, len(M.EPS) - 2)
        k = np.arange(C.shape[1])
        y0, y1, y2 = C[jj - 1, k], C[jj, k], C[jj + 1, k]
        den = y0 - 2 * y1 + y2
        off = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0), -1, 1)
        e = M.EPS[jj] + off * (M.EPS[1] - M.EPS[0])
        e[(j == 0) | (j == len(M.EPS) - 1)] = np.nan
        out_e[ok] = e
        out_c[ok] = y1 - 0.25 * (y0 - y2) * off
        return out_e, out_c


def coda_window(bn, dist, maxlag):
    b1, b2 = BASE[bn]
    L = b2 - b1
    t1 = max(2 * dist / VG[bn], b1)
    t2 = min(t1 + L, 0.92 * maxlag)
    return (round(t1, 2), round(t2, 2)) if t2 - t1 >= 0.4 * L else None


def side_view(X, side):
    if side == 'c':
        return X
    if side == 'a':
        return X[..., ::-1]
    return 0.5 * (X + X[..., ::-1])


def back_stacks(X, nwin=10, min_days=5):
    """10-day backward moving stacks, 1-day step: value at day d = nanmean of days d-9..d (>= min_days)."""
    fin = np.isfinite(X[..., 0])
    B = np.where(np.isfinite(X), X, 0.0).astype('float64')
    cs = np.concatenate([np.zeros((1,) + B.shape[1:]), np.cumsum(B, axis=0)])
    cn = np.concatenate([np.zeros((1,) + fin.shape[1:]), np.cumsum(fin, axis=0)])
    D = len(X)
    out = np.full(B.shape, np.nan)
    i = np.arange(D)
    lo = np.clip(i - nwin + 1, 0, None)
    ssum = cs[i + 1] - cs[lo]
    scnt = cn[i + 1] - cn[lo]
    ok = (scnt >= min_days) & (i >= nwin - 1)[(...,) + (None,) * (scnt.ndim - 1)]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(ok[..., None], ssum / np.maximum(scnt, 1)[..., None], np.nan)
    return out.astype('float32'), scnt


def step_stat(y, te):
    """mean(d+1..+15) - mean(d-30..-1) of a daily series (pd.Series on dates); >= 10 pre, >= 5 post."""
    rel = (y.index - te).days
    pre = y[(rel >= -30) & (rel <= -1)].dropna()
    post = y[(rel >= 1) & (rel <= 15)].dropna()
    if len(pre) < 10 or len(post) < 5:
        return np.nan
    return post.mean() - pre.mean()


def placebo_dates(days):
    first, last = days[0] + pd.Timedelta(days=31), days[-1] - pd.Timedelta(days=16)
    return [d for d in pd.date_range(first, last, freq='5D') if not (-40 <= (d - M7).days <= 16)]


def series_metrics(y):
    """y: pd.Series (daily, %), -> scatter metrics over the pre-M7 reference period."""
    pre = y[(y.index >= REF0) & (y.index <= REF1)]
    dd = pre.diff().dropna()
    return dict(n_days=int(y.notna().sum()), n_pre=int(pre.notna().sum()),
                sd_daydiff_pct=float(dd.std()) if len(dd) > 20 else np.nan,
                sd_series_pct=float(pre.std()) if pre.notna().sum() > 20 else np.nan)


def _dvv_pair(job):
    a, b, dist = job
    fn_out = os.path.join(SER_DIR, f'{a}_{b}.npz')
    if os.path.exists(fn_out):
        return a, b, 'have', None
    c0 = time.process_time()
    days = pd.date_range(D0, D1)
    PR = prods()
    cd, cl = load_pair(a, b, days)
    valid = np.isfinite(cl[:, 0, :, 0]).any(axis=1) | np.isfinite(cd[:, 0, :, 0]).any(axis=1)
    rmask = np.asarray((days >= REF0) & (days <= REF1))
    if (valid & rmask).sum() < 60 or (valid & (days > M7)).sum() < 30:
        np.savez(fn_out, skipped=True)
        return a, b, f'skip valid {valid.sum()} ref {(valid & rmask).sum()}', None
    SR = SideStretcher()
    nyr = sum(int((valid & (days.year == y)).sum() >= 60) for y in set(days.year))
    keepD, linesD = S.detect_lines_sc(PR['day'], days, cd, 2 if nyr >= 2 else 1)
    keepL, linesL = S.detect_lines_sc(PR['low'], days, cl, 2 if nyr >= 2 else 1)
    XD = J.notch_g(PR['day'], cd, keepD)
    XL0 = J.notch_g(PR['low'], cl, keepL)
    del cd, cl
    rows, ser = [], {}
    pl = placebo_dates(days)
    for key, bn in BANDS:
        P = PR[key]
        if key == 'day':
            bi = P.names.index(bn)
            X = XD[:, bi:bi + 1]
        elif key == 'low':
            X = XL0
        else:
            X = S.subband(XL0, *S.SUB[key])
        cw = coda_window(bn, dist, P.maxlag)
        if cw is None:
            rows.append(dict(a=a, b=b, dist_km=dist, band=bn, side='-', status='window_too_short'))
            continue
        with np.errstate(invalid='ignore'):
            ref_all = np.nanmean(X[rmask], axis=0, dtype='float64')      # (1, 9, L)
        Xb, cnt = back_stacks(X)                                        # (D, 1, 9, L)
        k10 = ((days - days[0]).days // 10).values
        for side in SIDES:
            Xs, Rs, Xbs = side_view(X, side), side_view(ref_all, side), side_view(Xb, side)
            # component-pair quality: median CC of non-overlapping 10-day stacks in the reference period
            labs, S10, N10 = S.stack_groups(days[rmask], Xs[rmask], k10[rmask], 5)
            dv10, cc10, er10 = SR.measure(P, Rs, S10, codas=[cw])
            with np.errstate(invalid='ignore'):
                q = np.nanmedian(cc10, axis=0)[0]                       # (9,)
            good = q >= 0.3
            dv, cc, er = SR.measure(P, Rs, Xbs, codas=[cw])                # (D, 1, 9)
            m, e = M.combine(dv[:, 0], np.where(good[None], er[:, 0], np.nan))
            y = pd.Series(m * 100, index=days)
            ccm = np.nanmean(np.where(good[None], cc[:, 0], np.nan), axis=1) if good.any() else np.full(len(days), np.nan)
            ser[(bn, side)] = np.stack([m * 100, e * 100, ccm]).astype('float32')
            # also ZZ only (for reference)
            yzz = pd.Series(dv[:, 0, 0] * 100 if q[0] >= 0.3 else np.full(len(days), np.nan), index=days)
            st = step_stat(y, M7)
            pv = np.array([step_stat(y, t) for t in pl])
            pv = pv[np.isfinite(pv)]
            sdp = float(np.std(pv)) if len(pv) > 5 else np.nan
            with np.errstate(invalid='ignore'):
                ccq = float(np.nanmedian(cc10[:, 0][:, good])) if good.any() else np.nan
            r = dict(a=a, b=b, dist_km=dist, band=bn, side=side, status='ok' if good.any() else 'no_good_comp',
                     t1=cw[0], t2=cw[1], n_good_comp=int(good.sum()), good_comps=' '.join(np.array(CP)[good]),
                     cc_med_good=ccq, cc_med_best=float(np.nanmax(q)) if np.isfinite(q).any() else np.nan,
                     cc_med_ZZ=float(q[0]), step_M7_pct=st, placebo_sd_pct=sdp, n_placebo=len(pv),
                     z_M7=st / sdp if sdp and np.isfinite(sdp) else np.nan,
                     step_M7_ZZ_pct=step_stat(yzz, M7), mean_err_pct=float(np.nanmedian(e * 100)) if good.any() else np.nan)
            r.update(series_metrics(y))
            r.update({'zz_' + k_: v for k_, v in series_metrics(yzz).items() if k_.startswith('sd')})
            rows.append(r)
    cpu = time.process_time() - c0
    np.savez(fn_out, days=np.array(days.strftime('%Y-%m-%d')),
             keys=np.array([f'{k[0]}|{k[1]}' for k in ser]), vals=np.array(list(ser.values())),
             lines_day=json.dumps(linesD), lines_low=json.dumps(linesL))
    for r in rows:
        r['cpu_dvv_s'] = cpu
        r['n_valid_days'] = int(valid.sum())
    return a, b, 'ok', rows


def stage_dvv(args):
    from multiprocessing import Pool
    os.makedirs(SER_DIR, exist_ok=True)
    P = all_pairs(MAXDIST)
    jobs = [(r.a, r.b, float(r.dist_km)) for _, r in P.iterrows()]
    if args.pairs:
        jobs = [j for j in jobs if f'{j[0]}|{j[1]}' in args.pairs]
    fn = os.path.join(OUT, 'pair_band_metrics.csv')
    allrows = []
    if os.path.exists(fn) and not args.pairs:
        allrows = pd.read_csv(fn).to_dict('records')
    done = {(r['a'], r['b']) for r in allrows}
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for i, (a, b, st_, rows) in enumerate(pool.imap_unordered(_dvv_pair, jobs)):
            if rows:
                allrows += rows
                pd.DataFrame(allrows).to_csv(fn, index=False)
            print(f'{a}-{b} {st_} [{i + 1}/{len(jobs)} {(time.time() - t0) / 60:.1f} min]', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['corr', 'vel', 'dvv'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--ndays', type=int, default=0)
    ap.add_argument('--pairs', nargs='*')
    a = ap.parse_args()
    if a.workers > 2:
        sys.exit('at most 2 workers (shared machine)')
    {'corr': stage_corr, 'vel': stage_vel, 'dvv': stage_dvv}[a.stage](a)


if __name__ == '__main__':
    main()
