#!/usr/bin/env python3
"""Single-station ambient-noise dv/v at BK.JEWT (Jewett Creek; directly above the LFE families),
2023-08-23 .. 2026-11-23, reusing scripts/sci_noise_dvv.py (imported as M, NOT modified).

Same algorithm as the method (one-bit + whitening, 1-Hz-smoothed whitening for autos, 6 pairs
ZZ NN EE EN EZ NZ, transient rejection, persistent-line notching, stretching on the method's EPS
grid with 4x upsampling, Weaver errors, inverse-variance pair combination ALL/SC/AC, 30-d pre/post
event steps with a 5-d moving-block bootstrap, placebo dates).  Products (per day, from one 100-Hz
NCEDC day that is never written to disk; 20-Hz night window and 5-Hz day cached in raw/):
  M      method bands 1-4 / 2-8 Hz, 09-11 UTC, coda 2-20 s  (directly comparable with KMPB/BRIC)
  day    requested 1-3 Hz (coda 3-30 s) and 3-5 Hz (coda 1-8 s), 24 h, 120-s segments, 20 Hz
  night  1-3 / 3-5 Hz in the method's 09-11 UTC window (comparison)
  low    requested 0.1-1 Hz, 24 h, 600-s segments, 5 Hz, lags +-150 s, coda 10-60 s, SC pairs only
JEWT-specific choices: reference = mean 2023-08-23..2024-11-30 (pre-M7; --ref all variant);
pair quality from 10-d stacks in that period; placebo dates every 5 d excluding M7-31..M7+120 d;
free offset at the 2024-02-28 StationXML epoch (HH response identical in both epochs).

Stages: local | download | m7local (62-day interim) | dvv | power | compare | plot
Outputs data/sci/noise_dvv_jewt/, figures figures/sci_noise_dvv_jewt_*.png
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, '/home/seismic/LFE/scripts')
import sci_noise_dvv as M          # noqa: E402  (method module; NOT modified)

ROOT = M.ROOT
OUT = os.path.join(ROOT, 'data/sci/noise_dvv_jewt')
RAW = os.path.join(OUT, 'raw')
DAYS = os.path.join(OUT, 'days')
FIG = M.FIG
NAME = 'JEWT'
NET, STA, LOC, CP = 'BK', 'JEWT', '00', 'HH?'
CMAP = {'Z': 'HHZ', 'N': 'HHN', 'E': 'HHE'}
START = '2023-08-23'
M7 = pd.Timestamp('2024-12-05')
LOCAL0, LOCAL1 = '2024-11-10', '2025-01-10'
SILENCE_END = pd.Timestamp('2024-12-30')
MIN_FREE_GB = 40.0


# ----------------------------------------------------------------------------- products
# Per-day products (all from the same 100-Hz day, which is never written to disk):
#   night : method window 09-11 UTC, 20 Hz, 120-s segments, lags +-60 s, bands NIGHT_BANDS
#           (method bands 1-4, 2-8 Hz for comparability + requested 1-3, 3-5 Hz), M._correlate_day
#   day20 : 00-24 UTC, 20 Hz, 120-s segments, lags +-60 s, bands 1-3, 3-5 Hz (M._correlate_day)
#   low   : 00-24 UTC, 5 Hz, 600-s segments, lags +-150 s, band 0.1-1 Hz (same algorithm,
#           _correlate_low: bandpass -> transient rejection -> one-bit -> whitening; autos with a
#           0.1-Hz smoothed amplitude instead of 1 Hz, i.e. the same "remove shape at lags < coda
#           start" rule scaled to the 10-s coda start)
# STS-5A response is flat (>=0.997 of 1-Hz gain) 0.05-10 Hz with phase 6.7 deg at 0.1 Hz and is
# identical in both StationXML epochs, so no response removal (one-bit + whitening anyway).
NIGHT_BANDS = [(1.0, 4.0), (2.0, 8.0), (1.0, 3.0), (3.0, 5.0)]
DAY20_BANDS = [(1.0, 3.0), (3.0, 5.0)]
LOW_BANDS = [(0.1, 1.0)]
LOW_FS, LOW_SEG, LOW_MAXLAG, LOW_SMOOTH = 5.0, 600.0, 150.0, 0.1


def _bn(b):
    return f'{b[0]:g}-{b[1]:g}Hz'


def _segments(st, day, fs, seg, h0, h1):
    """like M._day_segments but with explicit fs / segment / hours."""
    from obspy import UTCDateTime
    ns = int(seg * fs)
    t0 = UTCDateTime(day + pd.Timedelta(hours=h0))
    nseg = int((h1 - h0) * 3600 / seg)
    arr = np.full((3, nseg, ns), np.nan)
    for ic, c in enumerate('ZNE'):
        for tr in st.select(channel=CMAP[c]):
            i0 = int(round((tr.stats.starttime - t0) * fs))
            d = tr.data.astype('float64')
            for k in range(nseg):
                a = k * ns - i0
                if a >= 0 and a + ns <= len(d):
                    arr[ic, k] = d[a:a + ns]
    return arr


def _cos_taper_band(freqs, f1, f2, fs):
    w = np.zeros_like(freqs)
    lo0, hi0 = 0.5 * f1, min(1.25 * f2, fs / 2 - 0.2)
    w[(freqs >= f1) & (freqs <= f2)] = 1
    m = (freqs > lo0) & (freqs < f1)
    w[m] = 0.5 * (1 - np.cos(np.pi * (freqs[m] - lo0) / (f1 - lo0)))
    m = (freqs > f2) & (freqs < hi0)
    w[m] = 0.5 * (1 + np.cos(np.pi * (freqs[m] - f2) / (hi0 - f2)))
    return w


def _correlate_low(arr, bands, fs, maxlag, smooth_hz):
    """M._correlate_day with fs / maxlag / auto-smoothing as parameters (identical steps)."""
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import uniform_filter1d
    ns = arr.shape[-1]
    nfft = 2 * ns
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    nl = int(maxlag * fs)
    out = np.full((len(bands), 6, 2 * nl + 1), np.nan, dtype='float32')
    nused = np.zeros((len(bands), 6), dtype='int16')
    good = np.isfinite(arr).all(axis=2) & (np.nanstd(arr, axis=2) > 0)
    x = np.where(good[..., None], arr, 0.0)
    x = x - x.mean(axis=2, keepdims=True)
    tap = np.ones(ns)
    nt = int(0.05 * ns)
    tap[:nt] = 0.5 * (1 - np.cos(np.pi * np.arange(nt) / nt))
    tap[-nt:] = tap[:nt][::-1]
    for ib, (f1, f2) in enumerate(bands):
        sos = butter(4, [f1, f2], btype='band', fs=fs, output='sos')
        y = sosfiltfilt(sos, x * tap, axis=2)
        rms = np.sqrt((y ** 2).mean(axis=2))
        gb = good.copy()
        for ic in range(3):
            if gb[ic].any():
                gb[ic] &= rms[ic] < 4 * np.median(rms[ic, gb[ic]])
        y = np.sign(y) * tap
        Y = np.fft.rfft(y, nfft, axis=2)
        W = _cos_taper_band(freqs, f1, f2, fs)
        A = np.abs(Y) + 1e-10
        k = int(round(smooth_hz / (freqs[1] - freqs[0])))
        As = uniform_filter1d(A, k, axis=2, mode='nearest')
        Ya = W * Y / As
        Y = W * Y / A
        for ip, (ca, cb) in enumerate(M.PAIRS):
            ia, ibb = 'ZNE'.index(ca), 'ZNE'.index(cb)
            m = gb[ia] & gb[ibb]
            if m.sum() < 10:
                continue
            YY = Ya if ca == cb else Y
            C = np.fft.irfft(YY[ia, m] * np.conj(YY[ibb, m]), nfft, axis=1)
            C = np.concatenate([C[:, -nl:], C[:, :nl + 1]], axis=1)
            out[ib, ip] = C.mean(axis=0)
            nused[ib, ip] = m.sum()
    return out, nused


def _to_low(st20):
    st = st20.copy()
    for tr in st:
        tr.data = tr.data.astype('float64')
        tr.filter('lowpass', freq=2.0, corners=6, zerophase=True)
        tr.decimate(int(round(tr.stats.sampling_rate / LOW_FS)), no_filter=True)
        tr.data = np.round(tr.data).astype('int32')
    return st


def _process_day(st, day, cache=True):
    """100-Hz day stream -> per-day npz with the three products; caches 20-Hz night window
    and 5-Hz day (small) in RAW."""
    from obspy import UTCDateTime
    fn = os.path.join(DAYS, f'{day:%Y-%m-%d}.npz')
    st = st.select(network=NET, station=STA, location=LOC)
    st20 = M._decimate(st)                 # method's decimation (8-Hz low-pass, 20 Hz, int32)
    if len(st20) == 0:
        open(fn + '.none', 'w').close()
        return 0, 0
    a = UTCDateTime(day + pd.Timedelta(hours=M.H0))
    night = st20.slice(a - 30, a + (M.H1 - M.H0) * 3600 + 30)
    M.BANDS = NIGHT_BANDS
    cn, nn = M._correlate_day(M._day_segments(night, CMAP, day)) if len(night) else (None, None)
    if cn is None:
        cn = np.full((len(NIGHT_BANDS), 6, int(2 * M.MAXLAG * M.FS) + 1), np.nan, 'float32')
        nn = np.zeros((len(NIGHT_BANDS), 6), 'int16')
    M.BANDS = DAY20_BANDS
    cd, nd = M._correlate_day(_segments(st20, day, M.FS, M.SEG, 0, 24))
    stl = _to_low(st20)
    cl, nl_ = _correlate_low(_segments(stl, day, LOW_FS, LOW_SEG, 0, 24), LOW_BANDS, LOW_FS, LOW_MAXLAG, LOW_SMOOTH)
    np.savez_compressed(fn, night=cn, night_n=nn, day20=cd, day20_n=nd, low=cl, low_n=nl_,
                        chans=' '.join(sorted({t.stats.channel for t in st20})))
    if cache:
        os.makedirs(os.path.join(RAW, f'{day:%Y}'), exist_ok=True)
        if len(night):
            night.write(os.path.join(RAW, f'{day:%Y}', f'{day:%Y-%m-%d}_night20.mseed'), format='MSEED', encoding='STEIM2')
        stl.write(os.path.join(RAW, f'{day:%Y}', f'{day:%Y-%m-%d}_day5.mseed'), format='MSEED', encoding='STEIM2')
    return int(nn[0].max()), int(nl_[0].max())


def _local_day(dstr):
    from obspy import read
    day = pd.Timestamp(dstr)
    st = read(os.path.join(ROOT, 'data/continuous', dstr, 'BK.JEWT.mseed'))
    return (dstr,) + _process_day(st, day)


def stage_local(args):
    os.makedirs(DAYS, exist_ok=True)
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(LOCAL0, LOCAL1)]
    days = [d for d in days if os.path.exists(os.path.join(ROOT, 'data/continuous', d, 'BK.JEWT.mseed'))]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_local_day, days):
            print('local', *r, flush=True)


def _free_gb():
    s = os.statvfs('/home')
    return s.f_bavail * s.f_frsize / 1e9


def _download_day(dstr):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    day = pd.Timestamp(dstr)
    fday = os.path.join(DAYS, f'{dstr}.npz')
    if os.path.exists(fday) or os.path.exists(fday + '.none'):
        return dstr, 'have', 0
    if _free_gb() < MIN_FREE_GB:
        return dstr, 'lowdisk', 0
    t0 = time.time()
    a = UTCDateTime(day)
    client = Client('NCEDC', timeout=300)
    st = None
    for attempt in range(5):
        try:
            st = client.get_waveforms(NET, STA, LOC, CP, a - 60, a + 86400 + 60)
            break
        except Exception as e:
            if 'No data' in str(e) or '204' in str(e):
                st = None
                break
            time.sleep(5 * 2 ** attempt)
    else:
        return dstr, 'FAILED', time.time() - t0
    if st is None or len(st) == 0:
        open(fday + '.none', 'w').close()
        return dstr, 'nodata', time.time() - t0
    try:
        n1, n2 = _process_day(st, day)
    except Exception as e:
        return dstr, f'PROCFAIL {str(e)[:60]}', time.time() - t0
    nsec = sum(tr.stats.npts / tr.stats.sampling_rate for tr in st) / 3
    del st
    return dstr, f'ok night_seg{n1} low_seg{n2} hours{nsec / 3600:.1f}', time.time() - t0


def stage_download(args):
    os.makedirs(DAYS, exist_ok=True)
    end = args.end or (pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(args.start or START, end)]
    if args.reverse:
        days = days[::-1]
    logf = open(os.path.join(OUT, 'download_log.csv'), 'a')
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for d, status, dt in ex.map(_download_day, days):
            if status == 'have':
                continue
            logf.write(f'{d},{status},{dt:.1f}\n')
            logf.flush()
            print(d, status, f'{dt:.1f}s', f'free {_free_gb():.1f} GB', flush=True)
            if status == 'lowdisk':
                print('free space below', MIN_FREE_GB, 'GB: stopping', flush=True)
                ex.shutdown(cancel_futures=True)
                break


def stage_assemble(args):
    files = sorted(f for f in os.listdir(DAYS) if f.endswith('.npz'))
    d0 = min(pd.Timestamp(START), pd.Timestamp(files[0][:10]))
    d1 = pd.Timestamp(files[-1][:10])
    days = pd.date_range(d0, d1)
    nl = int(M.MAXLAG * M.FS)
    ccf = np.full((len(days), len(M.BANDS), len(M.PAIRS), 2 * nl + 1), np.nan, dtype='float32')
    nused = np.zeros((len(days), len(M.BANDS), len(M.PAIRS)), dtype='int16')
    for i, d in enumerate(days):
        fn = os.path.join(DAYS, f'{d:%Y-%m-%d}.npz')
        if os.path.exists(fn):
            z = np.load(fn)
            ccf[i], nused[i] = z['ccf'], z['nused']
    np.savez_compressed(os.path.join(OUT, f'ccf_{NAME}.npz'), days=np.array(days.strftime('%Y-%m-%d'), dtype='U10'),
                        ccf=ccf, nused=nused, pairs=M.PAIR_NAMES, bands=M.BAND_NAMES, fs=M.FS, maxlag=M.MAXLAG)
    print('assembled', len(days), 'days,', int((nused[:, 0].max(axis=1) >= 20).sum()), 'usable', flush=True)


# ----------------------------------------------------------------------------- generalized dv/v
class Prod:
    """one analysis product = a subset (sel) of the bands of a stored per-day correlation
    product (src: night / day20 / low), with its lag axis, per-band coda windows, mute and
    line-detector scales. 'M' (method bands, night window) reproduces sci_noise_dvv exactly
    (same FS, lags, MUTE 1.5-2 s, NF, line windows 30-45/45-60 s, 0.5-Hz median, EPS, UP=4)."""
    def __init__(self, key, src, sel, bands, fs, maxlag, coda, mute0=None, nf=8192,
                 line_med_hz=0.5, line_dil_hz=0.03, label='', allowed=None):
        self.key, self.src, self.sel, self.bands, self.fs, self.maxlag = key, src, list(sel), list(bands), fs, maxlag
        self.names = [_bn(b) for b in self.bands]
        self.coda = dict(coda)                    # band name -> (t1, t2)
        self.lags = np.arange(-int(maxlag * fs), int(maxlag * fs) + 1) / fs
        m0 = mute0 if mute0 is not None else 0.75 * min(c[0] for c in self.coda.values())
        m1 = m0 / 0.75
        self.mute = np.sin(np.pi / 2 * np.clip((np.abs(self.lags) - m0) / (m1 - m0), 0, 1)) ** 2
        self.nf = nf
        self.freq = np.fft.rfftfreq(nf, 1 / fs)
        self.linew = ((0.5 * maxlag, 0.75 * maxlag), (0.75 * maxlag, maxlag))
        self.line_med_hz, self.line_dil_hz = line_med_hz, line_dil_hz
        self.lags_u = self.lags[0] + np.arange(len(self.lags) * M.UP) / (fs * M.UP)
        self.label = label
        self.allowed = np.ones(6, bool) if allowed is None else np.array([p in allowed for p in M.PAIR_NAMES])


# coda windows: start >= ~3 dominant periods (T = 1/sqrt(f1 f2)) after zero lag, i.e. past the
# muted zero-lag peak, end where 5-d auto stacks keep median CC >~0.3-0.8 (tested on the 62 local
# days: scripts output m7_62d_coda_test.txt): 0.1-1 Hz 10-100 s (3-30 T), 1-3 Hz 3-30 s (5-50 T),
# 3-5 Hz 1-8 s (4-30 T); method bands 1-4 / 2-8 Hz keep the method's 2-20 s.
# 0.1-1 Hz: cross-component pairs ONLY. The 0.1-1 Hz autocorrelations at 10-30 s lag are within a
# factor ~3 of the deterministic response of the whitening taper (FT of W^2) plus the microseism
# peak shape, and 80% of daily stretches hit the +-2% grid edge; their SC coda is above the
# 24-h noise floor (~1.5e-4) to ~40-60 s, hence 10-60 s.
CODA_NEW = {'0.1-1Hz': (10.0, 60.0), '1-3Hz': (3.0, 30.0), '3-5Hz': (1.0, 8.0)}
PRODS = {
    'low': Prod('low', 'low', [0], LOW_BANDS, LOW_FS, LOW_MAXLAG, CODA_NEW, nf=4096,
                line_med_hz=0.05, line_dil_hz=0.005, label='24 h, 600-s segments, 5 Hz, SC pairs only',
                allowed=['EN', 'EZ', 'NZ']),
    'day': Prod('day', 'day20', [0, 1], DAY20_BANDS, M.FS, M.MAXLAG, CODA_NEW, mute0=0.75,
                label='24 h, 120-s segments, 20 Hz'),
    'night': Prod('night', 'night', [2, 3], NIGHT_BANDS[2:], M.FS, M.MAXLAG, CODA_NEW, mute0=0.75,
                  label='09-11 UTC (method window)'),
    'M': Prod('M', 'night', [0, 1], NIGHT_BANDS[:2], M.FS, M.MAXLAG, {'1-4Hz': M.CODA, '2-8Hz': M.CODA}, mute0=1.5,
              label='method bands, 09-11 UTC'),
}
MAIN = [('low', '0.1-1Hz'), ('day', '1-3Hz'), ('day', '3-5Hz')]


def load_prod(key, path=None, d0=None, d1=None):
    """-> days, ccf (D,B,P,L) with the method's >=20-segment rule (night/day20: >=40 min;
    low: >= 20 x 600 s = 3.3 h)."""
    if path is None:
        files = sorted(f for f in os.listdir(DAYS) if f.endswith('.npz'))
        d0 = pd.Timestamp(d0 or START)
        d1 = pd.Timestamp(d1 or files[-1][:10])
        days = pd.date_range(d0, d1)
        P = PRODS[key]
        ccf = np.full((len(days), len(P.bands), 6, len(P.lags)), np.nan)
        for i, d in enumerate(days):
            fn = os.path.join(DAYS, f'{d:%Y-%m-%d}.npz')
            if os.path.exists(fn):
                z = np.load(fn)
                c = z[P.src][P.sel].astype('float64')
                c[z[P.src + '_n'][P.sel] < 20] = np.nan
                ccf[i] = c
        return days, ccf
    z = np.load(path)
    return pd.to_datetime(z['days']), z['ccf']


def detect_lines_g(P, days, ccf, min_periods=2):
    """M.detect_lines generalized to the product's lag axis (line windows at 0.5-0.75 and
    0.75-1.0 x maxlag; 30-45/45-60 s for the 20-Hz products as in the method). Years with
    >= 60 days; a line must appear in >= 2 years (or in the single chunk if min_periods=1)."""
    from scipy.ndimage import median_filter, binary_dilation, gaussian_filter1d
    a = np.abs(P.lags)
    taps = [np.where((a >= lo) & (a <= hi), np.sin(np.pi * (a - lo) / (hi - lo)) ** 2, 0) for lo, hi in P.linew]
    k = int(round(P.line_med_hz / (P.freq[1] - P.freq[0])))
    if min_periods == 1:
        groups = [np.ones(len(days), bool)]
        need = 1
    else:
        groups = [days.year == yr for yr in sorted(set(days.year))]
        need = 2
    count = np.zeros(len(P.freq), int)
    for m in groups:
        if np.isfinite(ccf[m, 0, 0, 0]).sum() < (60 if min_periods > 1 else 20):
            continue
        with np.errstate(invalid='ignore'):
            S = np.nanmean(ccf[m], axis=0)
        line = np.zeros(len(P.freq), bool)
        for b, (f1, f2) in enumerate(P.bands):
            inb = (P.freq >= f1) & (P.freq <= f2)
            for p in range(6):
                if not np.isfinite(S[b, p]).all():
                    continue
                hit = inb.copy()
                for tap in taps:
                    A = np.abs(np.fft.rfft(S[b, p] * tap, P.nf))
                    r = A / (median_filter(A, size=k, mode='nearest') + 1e-20)
                    hit &= np.maximum.reduce([r, np.roll(r, 1), np.roll(r, -1)]) > M.LINE_THR
                line |= hit
        count += line
    line = count >= need
    line = binary_dilation(line, iterations=max(1, int(round(P.line_dil_hz / (P.freq[1] - P.freq[0])))))
    keep = 1 - np.clip(gaussian_filter1d(line.astype(float), 2), 0, 1)
    keep[line] = 0
    lab = np.diff(np.r_[0, line.astype(int), 0])
    i0, i1 = np.where(lab == 1)[0], np.where(lab == -1)[0] - 1
    return keep, [(round(float(P.freq[i]), 3), round(float(P.freq[j]), 3)) for i, j in zip(i0, i1)]


def notch_g(P, ccf, keep, chunk=200):
    out = np.empty(ccf.shape, dtype='float32')
    for i in range(0, len(ccf), chunk):
        c = ccf[i:i + chunk]
        X = np.fft.rfft(np.nan_to_num(c) * P.mute, P.nf, axis=-1) * keep
        o = np.fft.irfft(X, P.nf, axis=-1)[..., :ccf.shape[-1]]
        o[~np.isfinite(c)] = np.nan
        out[i:i + chunk] = o
    return out


def stretch_g(P, coda, ref, X):
    """M.stretch on the product's lag axis and coda window (same EPS grid, UP, refinement)."""
    from scipy.signal import resample
    X = np.atleast_2d(X)
    out_e = np.full(len(X), np.nan)
    out_c = np.full(len(X), np.nan)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(ref).all()
    if not ok.any():
        return out_e, out_c
    mu = (np.abs(P.lags_u) >= coda[0]) & (np.abs(P.lags_u) <= coda[1])
    t = P.lags_u[mu]
    refu = resample(ref, len(P.lags_u))
    R = np.array([np.interp(t * (1 + e), P.lags_u, refu) for e in M.EPS])
    R -= R.mean(axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    Y = resample(X[ok], len(P.lags_u), axis=1)[:, mu]
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


def weaver_err_g(cc, band, coda):
    f1, f2 = band
    T = 1.0 / (f2 - f1)
    wc = 2 * np.pi * np.sqrt(f1 * f2)
    t1, t2 = coda
    cc = np.clip(cc, 1e-3, 0.999)
    return np.sqrt(1 - cc ** 2) / (2 * cc) * np.sqrt(6 * np.sqrt(np.pi / 2) * T / (wc ** 2 * (t2 ** 3 - t1 ** 3)))


def measure_g(P, ref, S, codas=None):
    """S (G,B,P,L) -> dvv, cc, err (G,B,P) fractions; codas: list per band (default P.coda)."""
    G, B, NP, L = S.shape
    dv = np.full((G, B, NP), np.nan)
    cc = np.full((G, B, NP), np.nan)
    er = np.full((G, B, NP), np.nan)
    for b in range(B):
        coda = codas[b] if codas else P.coda[P.names[b]]
        for p in range(NP):
            dv[:, b, p], cc[:, b, p] = stretch_g(P, coda, ref[b, p], S[:, b, p])
        er[:, b] = weaver_err_g(cc[:, b], P.bands[b], coda)
    return dv, cc, er


def event_step_g(P, days, ccf, ref, te, good, half=30, skip=1, nboot=0, rng=None, codas=None):
    """M.event_step generalized (same pre/post definition, 5-d moving-block bootstrap)."""
    te = pd.Timestamp(te)
    pre = (days >= te - pd.Timedelta(days=half)) & (days < te)
    post = (days > te + pd.Timedelta(days=skip - 1)) & (days <= te + pd.Timedelta(days=half + skip - 1))
    stacks = []
    for msk in (pre, post):
        w = ccf[msk]
        if np.isfinite(w[:, 0, :, 0]).any(axis=1).sum() < half // 3:
            return None
        with np.errstate(invalid='ignore'):
            base = np.nanmean(w, axis=0)[None]
        if nboot:
            base = np.concatenate([base, M.mbb_window(w, nboot, 5, rng)], axis=0)
        stacks.append(base)
    out = []
    for S in stacks:
        dv, cc, er = measure_g(P, ref, S, codas)
        er = np.where(good[None], er, np.nan)
        out.append({c: M.combine(dv[:, :, ix], er[:, :, ix])[0] for c, ix in M.COMBOS.items()})
    return {c: (out[1][c] - out[0][c]) * 100 for c in M.COMBOS}


# ----------------------------------------------------------------------------- interim (62 d)
def good_pairs(P, days, ccf, ref, mask, block=10, min_days=5, min_cc=0.3):
    """method's pair_quality: median CC of block-day stacks vs reference inside `mask`."""
    k = ((days - days[0]).days // block).values
    labs, S, N = M.stack_groups(days[mask], ccf[mask], k[mask], min_days)
    dv, cc, er = measure_g(P, ref, S)
    with np.errstate(invalid='ignore'):
        q = np.nanmedian(cc, axis=0)
    return q, (q >= min_cc) & P.allowed[None]


def series_g(P, days, ccf, ref, good, blk, t0, min_days):
    """block stacks aligned on t0 -> dict combo/pair -> DataFrame(t, dvv_pct, err_pct, cc)."""
    k = ((days - t0).days // blk).values
    labs, S, N = M.stack_groups(days, ccf, k, min_days)
    dv, cc, er = measure_g(P, ref, S)
    erg = np.where(good[None], er, np.nan)
    tt = t0 + pd.to_timedelta(labs * blk + (blk - 1) / 2, unit='D')
    out = {}
    for c, ix in M.COMBOS.items():
        m, e = M.combine(dv[:, :, ix], erg[:, :, ix])
        for b in range(len(P.bands)):
            out[(c, b)] = pd.DataFrame(dict(t=tt, k=labs, dvv_pct=m[:, b] * 100, err_pct=e[:, b] * 100,
                                            ndays=N[:, b, ix].max(axis=1)))
    for p in range(6):
        for b in range(len(P.bands)):
            out[(M.PAIR_NAMES[p], b)] = pd.DataFrame(dict(t=tt, k=labs, dvv_pct=dv[:, b, p] * 100,
                                                          err_pct=er[:, b, p] * 100, cc=cc[:, b, p], ndays=N[:, b, p]))
    return out


def stage_m7local(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)
    rows, srows, prow, ser, coda_lines, meta = [], [], [], [], [], {}
    keep_series = {}
    for key, P in PRODS.items():
        days, ccf = load_prod(key, d0=LOCAL0, d1=LOCAL1)
        pre = days < M7
        keep, lines = detect_lines_g(P, days, ccf, 1)
        nf = {P.names[b]: round(float(1 - keep[(P.freq > f1) & (P.freq < f2)].mean()), 3) for b, (f1, f2) in enumerate(P.bands)}
        meta[key] = dict(label=P.label, lines_hz=lines, notched_frac=nf, coda=P.coda,
                         usable_days=int(np.isfinite(ccf[:, 0, :, 0]).any(axis=1).sum()))
        for variant in ('notch', 'nonotch'):
            X = notch_g(P, ccf, keep) if variant == 'notch' else (ccf * P.mute).astype('float32')
            with np.errstate(invalid='ignore'):
                ref = np.nanmean(X[pre], axis=0)
            q, good = good_pairs(P, days, X, ref, pre, block=5, min_days=3)
            if variant == 'notch':    # coda-window coherence test (5-d stacks, pre-M7)
                k = ((days - days[0]).days // 5).values
                labs, S, N = M.stack_groups(days[pre], X[pre], k[pre], 3)
                for b, bn in enumerate(P.names):
                    tests = {'0.1-1Hz': [(5, 40), (10, 40), (10, 60), (10, 100), (20, 120)],
                             '1-3Hz': [(2, 20), (3, 30), (5, 40), (10, 60)], '3-5Hz': [(0.5, 5), (1, 8), (2, 12)],
                             '1-4Hz': [(2, 20)], '2-8Hz': [(2, 20), (1, 8)]}[bn]
                    for cd in tests:
                        dv, cc, er = measure_g(P, ref, S, [cd] * len(P.bands))
                        coda_lines.append(f'{key:6s} {bn:8s} coda {str(cd):11s} median CC(5-d) '
                                          + ' '.join(f'{p}:{v:.2f}' for p, v in zip(M.PAIR_NAMES, np.nanmedian(cc[:, b], 0)))
                                          + '  | sd(5-d dvv) % ' + ' '.join(f'{v:.3f}' for v in np.nanstd(dv[:, b], 0) * 100))
            r = event_step_g(P, days, X, ref, M7, good, half=30, nboot=args.nboot, rng=rng)
            ss = {blk: series_g(P, days, X, ref, good, blk, M7, 1 if blk == 1 else 2) for blk in (1, 5)}
            keep_series[(key, variant)] = (ss, good, q)
            for (c, b), df in ss[1].items():
                ser.append(df.assign(prod=key, band=P.names[b], combo=c, block_days=1, variant=variant))
            for (c, b), df in ss[5].items():
                ser.append(df.assign(prod=key, band=P.names[b], combo=c, block_days=5, variant=variant))
            for c in M.COMBOS:
                for b, bn in enumerate(P.names):
                    s5, s1 = ss[5][(c, b)], ss[1][(c, b)]
                    p5, q5 = s5[s5.k < 0].dvv_pct.dropna(), s5[s5.k >= 0].dvv_pct.dropna()
                    sd5 = p5.std(ddof=1)
                    se_sc = sd5 * np.sqrt(1 / max(len(p5), 1) + 1 / max(len(q5), 1))
                    st_ = r[c][0, b] if r else np.nan
                    rows.append(dict(prod=key, band=bn, coda=str(P.coda[bn]), variant=variant, combo=c,
                                     step30_pct=st_, se_mbb_pct=np.nanstd(r[c][1:, b]) if r else np.nan,
                                     sd_pre_5d_pct=sd5, n_pre_5d=len(p5), n_post_5d=len(q5),
                                     se_prescatter_pct=se_sc, z_prescatter=st_ / se_sc,
                                     sd_pre_daily_pct=s1[s1.k < 0].dvv_pct.std(ddof=1),
                                     pairs_used=' '.join(np.array(M.PAIR_NAMES)[good[b]]),
                                     pair_q=' '.join(f'{p}:{v:.2f}' for p, v in zip(M.PAIR_NAMES, q[b]))))
                    # silence-period windows relative to the pre-M7 mean (daily values)
                    pm = s1[s1.k < 0].dvv_pct
                    for lo, hi, lab in [(1, 5, 'd1-5'), (6, 10, 'd6-10'), (11, 15, 'd11-15'), (16, 20, 'd16-20'),
                                        (21, 25, 'd21-25'), (1, 25, 'silence d1-25 (Dec 6-30)'),
                                        (26, 36, 'd26-36 (Dec 31-Jan 10)')]:
                        w = s1[(s1.k >= lo) & (s1.k <= hi)].dvv_pct.dropna()
                        # SE: pre-M7 5-d scatter scaled to the window length (accounts for day-to-day
                        # correlation better than the daily white-noise SE, also given)
                        nb = max(len(w) / 5, 1)
                        srows.append(dict(prod=key, band=bn, variant=variant, combo=c, window=lab, n_days=len(w),
                                          mean_minus_pre_pct=w.mean() - pm.mean(),
                                          se_5dscatter_pct=sd5 * np.sqrt(1 / nb + 1 / max(len(p5), 1)),
                                          se_daily_white_pct=pm.std(ddof=1) * np.sqrt(1 / max(len(w), 1) + 1 / pm.notna().sum())))
            # placebo: steps with shorter windows at dates inside the 25-d pre-M7 period
            for half in (8, 12):
                vals = []
                for td in pd.date_range(pd.Timestamp(LOCAL0) + pd.Timedelta(days=half), M7 - pd.Timedelta(days=half), freq='1D'):
                    rr = event_step_g(P, days, X, ref, td, good, half=half)
                    if rr is not None:
                        vals.append(rr['ALL'][0])
                vals = np.array(vals).reshape(-1, len(P.bands))
                rr = event_step_g(P, days, X, ref, M7, good, half=half)
                for b, bn in enumerate(P.names):
                    v = vals[:, b]
                    prow.append(dict(prod=key, band=bn, variant=variant, half=half, m7_step_pct=rr['ALL'][0, b],
                                     n_placebo_dates=len(v), n_nonoverlapping=int(25 // (2 * half)),
                                     placebo_sd_pct=v.std() if len(v) else np.nan,
                                     placebo_absmax_pct=np.abs(v).max() if len(v) else np.nan))
    S = pd.DataFrame(rows)
    W = pd.DataFrame(srows)
    PL = pd.DataFrame(prow)
    S.to_csv(os.path.join(OUT, 'm7_62d_steps.csv'), index=False)
    W.to_csv(os.path.join(OUT, 'm7_62d_silence_windows.csv'), index=False)
    PL.to_csv(os.path.join(OUT, 'm7_62d_placebo.csv'), index=False)
    pd.concat(ser).to_csv(os.path.join(OUT, 'm7_62d_series.csv'), index=False)
    open(os.path.join(OUT, 'm7_62d_coda_test.txt'), 'w').write('\n'.join(coda_lines) + '\n')
    json.dump(meta, open(os.path.join(OUT, 'm7_62d_meta.json'), 'w'), indent=1, default=str)
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        print('\n'.join(coda_lines))
        print(json.dumps(meta, default=str))
        print(S[S.combo == 'ALL'].drop(columns=['pair_q']).round(3).to_string())
        print(S[(S.combo != 'ALL') & (S.variant == 'notch')].drop(columns=['pair_q', 'pairs_used']).round(3).to_string())
        print(W[(W.combo == 'ALL') & (W.variant == 'notch')].round(3).to_string())
        print(PL.round(3).to_string())
    # ---------------- figure
    panels = [('low', 0), ('day', 0), ('day', 1), ('night', 0), ('M', 0), ('M', 1)]
    fig, axs = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)), sharex=True)
    for ax, (key, b) in zip(axs, panels):
        P = PRODS[key]
        ss, good, q = keep_series[(key, 'notch')]
        bn = P.names[b]
        s1 = ss[1][('ALL', b)]
        ax.plot(s1.t, s1.dvv_pct, '.', color='0.6', ms=4, label='daily, all pairs')
        for c, colr, off in (('ALL', 'C3', 0), ('AC', 'C0', -0.25), ('SC', 'C2', 0.25)):
            s5 = ss[5][(c, b)]
            ax.errorbar(s5.t + pd.Timedelta(days=off), s5.dvv_pct, yerr=s5.err_pct, fmt='o-', ms=4, lw=1.2 if c == 'ALL' else 0.8,
                        color=colr, label=f'5-d {c}', alpha=1 if c == 'ALL' else 0.75)
        st = S[(S['prod'] == key) & (S.band == bn) & (S.variant == 'notch') & (S.combo == 'ALL')].iloc[0]
        wsil = W[(W['prod'] == key) & (W.band == bn) & (W.variant == 'notch') & (W.combo == 'ALL') & W.window.str.startswith('silence')].iloc[0]
        ax.set_title(f'{bn} ({P.label}; coda {P.coda[bn][0]:g}-{P.coda[bn][1]:g} s; pairs {st.pairs_used or "none"}):  '
                     f'30-d step {st.step30_pct:+.3f} +/- {st.se_mbb_pct:.3f} (MBB) / {st.se_prescatter_pct:.3f} (pre scatter); '
                     f'silence mean {wsil.mean_minus_pre_pct:+.3f}', fontsize=7.5, loc='left')
        ax.axvline(pd.Timestamp('2024-12-05 18:44'), color='r', lw=1)
        ax.axvspan(M7, SILENCE_END, color='#fde0dd', zorder=0)
        ax.axhline(0, color='0.5', lw=0.5)
        lo, hi = np.nanpercentile(s1.dvv_pct, [3, 97])
        pad = 0.3 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel('dv/v (%)')
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=4, loc='lower left')
    fig.suptitle('BK.JEWT ambient-noise dv/v, 62 local days (ref = pre-M7 mean; shaded = LFE silence Dec 5-30)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_jewt_m7_62d.png'), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- full record
BREAKS = {'xml_epoch_2024-02-28': '2024-02-28'}   # StationXML epoch change (identical HH response)
EVENTS = {'M7': '2024-12-05', 'M4.8_2023-10-16_46km': '2023-10-16', 'M5.25_2024-12-15_138km': '2024-12-15',
          'M5.1_2026-06-03_123km': '2026-06-03', 'M5.6_2026-06-24_85km': '2026-06-24'}
EXCL_PLACEBO = (-31, 120)        # placebo dates excluded within M7 + [-31, +120] d
REC_WINDOWS = [(1, 25, 'silence d1-25'), (1, 10, 'd1-10'), (11, 25, 'd11-25'), (26, 60, 'd26-60'),
               (61, 120, 'd61-120'), (121, 240, 'd121-240'), (241, 365, 'd241-365'), (366, 730, 'd366-730')]
LAPSE = {'0.1-1Hz': [(10, 30), (30, 60)], '1-3Hz': [(3, 10), (10, 30)], '3-5Hz': [(1, 3), (3, 8)],
         '1-4Hz': [(2, 8), (8, 20)], '2-8Hz': [(2, 8), (8, 20)]}


def _design(tt, t_end_fit=None, tau=None):
    """c + trend + annual + free offset at BREAKS + M7 step [+ M7 exp recovery (tau d)]."""
    ty = (tt - M7).days.values / 365.25
    cols = [np.ones_like(ty), ty, np.cos(2 * np.pi * tt.day_of_year.values / 365.25),
            np.sin(2 * np.pi * tt.day_of_year.values / 365.25)]
    names = ['c', 'trend_per_yr', 'acos', 'asin']
    for k, d in BREAKS.items():
        cols.append((tt >= pd.Timestamp(d)).astype(float))
        names.append('off_' + k)
    dt = (tt - M7).days.values
    cols.append((dt > 0).astype(float))
    names.append('step_M7_perm')
    if tau:
        cols.append(np.where(dt > 0, np.exp(-np.clip(dt, 0, None) / tau), 0.0))
        names.append('rec_M7_amp')
    return np.array(cols).T, names


def fit_series(tt, y, rng, nb=500):
    """least squares on 10-d series; tau chosen from a grid by RSS (None = permanent step only);
    moving-block (9) residual bootstrap as in the method."""
    ok = np.isfinite(y)
    tt, y = tt[ok], y[ok]
    if len(y) < 30:
        return dict(n=len(y)), None
    best = None
    for tau in [None, 10, 20, 40, 80, 160, 320]:
        X, names = _design(tt, tau=tau)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rss = ((y - X @ beta) ** 2).sum()
        bic = len(y) * np.log(rss / len(y)) + X.shape[1] * np.log(len(y))
        if best is None or bic < best[0]:
            best = (bic, tau, X, names, beta, rss)
    bic, tau, X, names, beta, rss = best
    resid = y - X @ beta
    bb = []
    for _ in range(nb):
        st0 = rng.integers(0, len(resid) - 9 + 1, size=int(np.ceil(len(resid) / 9)))
        rr = np.concatenate([resid[s_:s_ + 9] for s_ in st0])[:len(resid)]
        bb.append(np.linalg.lstsq(X, X @ beta + rr, rcond=None)[0])
    bb = np.array(bb)
    out = dict(n=len(y), tau_days=tau, rms_resid=float(np.std(resid)))
    for n_, v, e in zip(names, beta, bb.std(0)):
        out[n_], out[n_ + '_se'] = v, e
    amp = np.hypot(beta[2], beta[3])
    ampb = np.hypot(bb[:, 2], bb[:, 3])
    out['annual_amp_pct'], out['annual_amp_se'] = amp, ampb.std()
    out['annual_peak_doy'] = (np.degrees(np.arctan2(beta[3], beta[2])) % 360) / 360 * 365.25
    out['annual_peak_doy_se'] = np.degrees(np.std(np.angle(np.exp(1j * (np.arctan2(bb[:, 3], bb[:, 2]) - np.arctan2(beta[3], beta[2])))))) / 360 * 365.25
    if tau:
        out['coseis_M7'] = beta[names.index('step_M7_perm')] + beta[names.index('rec_M7_amp')]
        out['coseis_M7_se'] = (bb[:, names.index('step_M7_perm')] + bb[:, names.index('rec_M7_amp')]).std()
    # 'method' fit for comparability: c + annual + M7 step (+ break offset), no trend
    Xm = np.delete(X[:, :len(names) - (1 if tau else 0)], 1, axis=1)
    bm = np.linalg.lstsq(Xm, y, rcond=None)[0]
    out['methodfit_step_M7'] = bm[-1]
    out['methodfit_annual_amp'] = np.hypot(bm[1], bm[2])
    return out, (tt, y, X @ beta)


def _window_stat(s1, te, lo, hi, pre_days=30):
    """mean daily dv/v in te+[lo,hi] minus mean over te-[pre_days,1]."""
    k = (s1.t - te).dt.days
    a = s1.dvv_pct[(k >= -pre_days) & (k < 0)].dropna()
    w = s1.dvv_pct[(k >= lo) & (k <= hi)].dropna()
    if len(a) < 10 or len(w) < max(3, (hi - lo + 1) // 3):
        return np.nan
    return w.mean() - a.mean()


def _dvv_prod(job):
    key, nboot, refmode, seed = job
    rng = np.random.default_rng(seed)
    P = PRODS[key]
    days, ccf = load_prod(key)
    keep, lines = detect_lines_g(P, days, ccf, 2)
    nf = {P.names[b]: round(float(1 - keep[(P.freq > f1) & (P.freq < f2)].mean()), 3) for b, (f1, f2) in enumerate(P.bands)}
    X = notch_g(P, ccf, keep).astype('float64')
    del ccf
    valid = np.isfinite(X[:, 0, :, 0]).any(axis=1)
    rmask = (days < pd.Timestamp('2024-11-30')) if refmode == 'pre' else np.ones(len(days), bool)
    with np.errstate(invalid='ignore'):
        ref = np.nanmean(X[rmask], axis=0)
    q, good = good_pairs(P, days, X, ref, rmask, block=10, min_days=5)
    print(key, 'lines', lines, 'notched', nf, 'pair q', np.round(q, 2), flush=True)
    res = dict(meta=dict(key=key, label=P.label, lines_hz=lines, notched_frac=nf, coda={n: P.coda[n] for n in P.names},
                         n_days=int(valid.sum()), first=str(days[valid][0].date()), last=str(days[valid][-1].date()),
                         n_ref_days=int((valid & rmask).sum()),
                         pair_q={P.names[b]: dict(zip(M.PAIR_NAMES, np.round(q[b], 3).tolist())) for b in range(len(P.bands))},
                         pairs_used={P.names[b]: list(np.array(M.PAIR_NAMES)[good[b]]) for b in range(len(P.bands))}))
    rows10, rowsM, rowsP, rows1 = [], [], [], []
    # ---- series: daily, 5-d, 10-d (aligned on the M7 day), monthly
    ser = {}
    for blk, md in ((1, 1), (5, 2), (10, 5)):
        ser[blk] = series_g(P, days, X, ref, good, blk, M7, md)
    km = days.to_period('M').astype(str).values
    labsM, SM, NM = M.stack_groups(days, X, km, 8)
    dvM, ccM, erM = measure_g(P, ref, SM)
    erMg = np.where(good[None], erM, np.nan)
    for c, ix in M.COMBOS.items():
        m, e = M.combine(dvM[:, :, ix], erMg[:, :, ix])
        for b in range(len(P.bands)):
            for g in range(len(labsM)):
                rowsM.append(dict(prod=key, band=P.names[b], combo=c, month=labsM[g], dvv_pct=m[g, b] * 100,
                                  err_pct=e[g, b] * 100, ndays=int(NM[g, b, ix].max())))
    for (c, b), df in ser[10].items():
        (rows10 if c in M.COMBOS else rowsP).append(df.assign(prod=key, band=P.names[b], combo=c))
    for blk in (1, 5):
        for (c, b), df in ser[blk].items():
            if c in M.COMBOS:
                rows1.append(df.assign(prod=key, band=P.names[b], combo=c, block_days=blk))
    # ---- steps: MBB at events, placebo dates, same-date other years
    steps, plac_out = [], {}
    first, last = days[valid][0], days[valid][-1]
    pl_dates = [d for d in pd.date_range(first + pd.Timedelta(days=31), last - pd.Timedelta(days=31), freq='5D')
                if not (EXCL_PLACEBO[0] <= (d - M7).days <= EXCL_PLACEBO[1])]
    plac = []
    for td in pl_dates:
        r = event_step_g(P, days, X, ref, td, good)
        plac.append(np.stack([r[c][0] for c in M.COMBOS]) if r is not None else np.full((3, len(P.bands)), np.nan))
    plac = np.array(plac)                                    # (ndates, combo, band)
    plac_out = dict(dates=np.array([d.strftime('%Y-%m-%d') for d in pl_dates]), steps=plac, combos=list(M.COMBOS), bands=P.names)
    for ev, te in list(EVENTS.items()) + [(k, v) for k, v in BREAKS.items()]:
        half = 30 if ev != 'M5.25_2024-12-15_138km' else 9          # inside the M7 sequence: 9-d windows
        r = event_step_g(P, days, X, ref, te, good, half=half, nboot=nboot, rng=rng)
        if r is None:
            continue
        # placebo for 9-d windows
        if half != 30:
            pv9 = []
            for td in pl_dates[::2]:
                rr = event_step_g(P, days, X, ref, td, good, half=half)
                if rr is not None:
                    pv9.append(np.stack([rr[c][0] for c in M.COMBOS]))
            pv_all = np.array(pv9)
        else:
            pv_all = plac
        seas = []
        for yr in range(2023, 2027):
            td = pd.Timestamp(te).replace(year=yr)
            if td == pd.Timestamp(te) or abs((td - M7).days) < 45:
                continue
            rr = event_step_g(P, days, X, ref, td, good, half=half)
            if rr is not None:
                seas.append((yr, rr))
        for ic, c in enumerate(M.COMBOS):
            for b in range(len(P.bands)):
                pv = pv_all[:, ic, b]
                pv = pv[np.isfinite(pv)]
                st_ = r[c][0, b]
                sdp = np.std(pv) if len(pv) > 5 else np.nan
                steps.append(dict(prod=key, band=P.names[b], combo=c, event=ev, date=te, half_days=half, step_pct=st_,
                                  se_mbb_pct=np.nanstd(r[c][1:, b]), placebo_sd_pct=sdp, n_placebo=len(pv),
                                  z_placebo=st_ / sdp, frac_placebo_le=np.mean(pv <= st_) if len(pv) else np.nan,
                                  placebo_robust_sd_pct=1.4826 * np.median(np.abs(pv - np.median(pv))) if len(pv) else np.nan,
                                  same_date_other_years=';'.join(f'{yr}:{rr[c][0, b]:+.3f}' for yr, rr in seas)))
    # ---- recovery windows (daily series) with placebo-date null for each window
    rec = []
    for ic, c in enumerate(M.COMBOS):
        for b in range(len(P.bands)):
            s1 = ser[1][(c, b)]
            for lo, hi, lab in REC_WINDOWS:
                v = _window_stat(s1, M7, lo, hi)
                nul = np.array([_window_stat(s1, td, lo, hi) for td in pl_dates
                                if not (-hi - 31 <= (td - M7).days <= EXCL_PLACEBO[1])])
                nul = nul[np.isfinite(nul)]
                rec.append(dict(prod=key, band=P.names[b], combo=c, window=lab, lo=lo, hi=hi, minus_pre30_pct=v,
                                null_sd_pct=np.std(nul) if len(nul) > 5 else np.nan, null_mean_pct=np.mean(nul) if len(nul) else np.nan,
                                n_null=len(nul), z=(v - np.mean(nul)) / np.std(nul) if len(nul) > 5 else np.nan))
    # ---- long-term fit on 10-d series: trend, annual, break offset, M7 step + recovery
    fits = []
    for c in M.COMBOS:
        for b in range(len(P.bands)):
            s10 = ser[10][(c, b)]
            o, _ = fit_series(pd.DatetimeIndex(s10.t), s10.dvv_pct.values, rng)
            fits.append(dict(prod=key, band=P.names[b], combo=c, **o))
    # ---- lapse dependence: M7 step in early / late coda windows (+ placebo sd)
    lap = []
    for b, bn in enumerate(P.names):
        for cd in LAPSE.get(bn, []):
            codas = [cd if i == b else P.coda[P.names[i]] for i in range(len(P.bands))]
            r = event_step_g(P, days, X, ref, M7, good, nboot=min(nboot, 100), rng=rng, codas=codas)
            pv = []
            for td in pl_dates[::2]:
                rr = event_step_g(P, days, X, ref, td, good, codas=codas)
                if rr is not None:
                    pv.append(np.stack([rr[c][0] for c in M.COMBOS]))
            pv = np.array(pv)
            for ic, c in enumerate(M.COMBOS):
                lap.append(dict(prod=key, band=bn, combo=c, coda=f'{cd[0]}-{cd[1]}s', step_pct=r[c][0, b],
                                se_mbb_pct=np.nanstd(r[c][1:, b]), placebo_sd_pct=np.nanstd(pv[:, ic, b])))
    import pickle
    out = dict(meta=res['meta'], rows10=pd.concat(rows10), rowsP=pd.concat(rowsP), rowsM=pd.DataFrame(rowsM),
                rows1=pd.concat(rows1), steps=pd.DataFrame(steps), rec=pd.DataFrame(rec), fits=pd.DataFrame(fits),
                lapse=pd.DataFrame(lap), plac=plac_out)
    pickle.dump(out, open(os.path.join(OUT, f'_dvv_{key}_{refmode}.pkl'), 'wb'))
    return out


def stage_dvv(args):
    tag = args.tag
    keys = args.prods or list(PRODS)
    jobs = [(k, args.nboot, args.ref, 11 + i) for i, k in enumerate(keys)]
    acc = {k: [] for k in ['rows10', 'rowsP', 'rowsM', 'rows1', 'steps', 'rec', 'fits', 'lapse']}
    meta = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for k, r in zip(keys, ex.map(_dvv_prod, jobs)):
            for a in acc:
                acc[a].append(r[a])
            meta[k] = r['meta']
            np.savez(os.path.join(OUT, f'placebo_{k}{tag}.npz'), **r['plac'])
    names = dict(rows10='dvv_10day', rowsP='dvv_10day_pairs', rowsM='dvv_monthly', rows1='dvv_daily_5day',
                 steps='steps', rec='recovery_windows', fits='longterm_fit', lapse='lapse_steps')
    out = {}
    for a, n in names.items():
        out[a] = pd.concat(acc[a], ignore_index=True)
        out[a].to_csv(os.path.join(OUT, f'jewt_noise_{n}{tag}.csv'), index=False)
    json.dump(dict(meta=meta, ref=args.ref, events=EVENTS, breaks=BREAKS, excl_placebo=EXCL_PLACEBO),
              open(os.path.join(OUT, f'jewt_noise_meta{tag}.json'), 'w'), indent=1, default=str)
    with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 500):
        S = out['steps']
        print(S.drop(columns='same_date_other_years').round(3).to_string())
        print(out['rec'].round(3).to_string())
        F = out['fits']
        print(F.round(3).to_string())
        print(out['lapse'].round(3).to_string())


# ----------------------------------------------------------------------------- noise power (source check)
def _power_day(dstr):
    from obspy import read
    from scipy.signal import butter, sosfiltfilt
    day = pd.Timestamp(dstr)
    out = dict(day=dstr)
    for kind, fs, seg, bands in (('day5', LOW_FS, 600.0, [(0.1, 0.3), (0.3, 1.0), (0.1, 1.0), (1.0, 2.0)]),
                                 ('night20', M.FS, 120.0, [(1.0, 3.0), (3.0, 5.0), (1.0, 4.0), (2.0, 8.0)])):
        fn = os.path.join(RAW, f'{day:%Y}', f'{dstr}_{kind}.mseed')
        if not os.path.exists(fn):
            loc = os.path.join(ROOT, 'data/continuous', dstr, 'BK.JEWT.mseed')
            if not os.path.exists(loc):
                continue
        try:
            st = read(fn)
        except Exception:
            continue
        h0, h1 = (0, 24) if kind == 'day5' else (M.H0, M.H1)
        arr = _segments(st, day, fs, seg, h0, h1)
        good = np.isfinite(arr).all(axis=2)
        for f1, f2 in bands:
            sos = butter(4, [f1, f2], btype='band', fs=fs, output='sos')
            for ic, c in enumerate('ZNE'):
                x = arr[ic][good[ic]]
                if len(x) < 5:
                    continue
                y = sosfiltfilt(sos, (x - x.mean(1, keepdims=True)), axis=1)
                out[f'{kind}_{f1:g}-{f2:g}_{c}'] = float(np.median(np.sqrt((y ** 2).mean(1))))
    return out


def stage_power(args):
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(START, args.end or '2026-12-31')]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(_power_day, days, chunksize=8) if len(r) > 1]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'jewt_noise_power.csv'), index=False)
    print('power days', len(rows))


# ----------------------------------------------------------------------------- comparisons
def _lfe_jewt():
    out = {}
    for ch in 'ENZ':
        d = pd.read_csv(os.path.join(ROOT, f'data/dvv/pool_clean/BK.JEWT.00.HH{ch}_curve.csv'), parse_dates=['t'])
        out['HH' + ch] = d
    return out


def stage_compare(args):
    """JEWT noise vs LFE-coda dv/v at JEWT and vs KMPB/BRIC noise (existing method outputs);
    noise-power regression (source check)."""
    Mo = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_monthly.csv'))
    Mo['month'] = pd.to_datetime(Mo.month)
    T10 = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_10day.csv'), parse_dates=['t'])
    S = pd.read_csv(os.path.join(OUT, 'jewt_noise_steps.csv'))
    rows = []
    # --- LFE coda at JEWT (monthly curve points; step at the M7 between segments 0 and 1)
    lfe = _lfe_jewt()
    lf = pd.concat([d.assign(chan=k) for k, d in lfe.items()])
    lf['month'] = lf.t.dt.to_period('M').dt.to_timestamp()
    lfm = lf.groupby(['month']).dvv_pct.mean()
    lfm_ch = lf.groupby(['chan', 'month']).dvv_pct.mean().unstack(0)
    # --- other noise stations (method outputs, 2019-2026/06, reference 2019-2021)
    oth = pd.read_csv(os.path.join(ROOT, 'data/sci/noise_dvv/noise_dvv_monthly.csv'), parse_dates=['month'])
    othS = pd.read_csv(os.path.join(ROOT, 'data/sci/noise_dvv/noise_dvv_steps.csv'))
    for (prod, band, combo), g in Mo.groupby(['prod', 'band', 'combo']):
        g = g.set_index('month').dvv_pct
        d = dict(prod=prod, band=band, combo=combo)
        for name, ref in [('LFEcoda_JEWT_mean', lfm)] + [(f'LFEcoda_JEWT_{c}', lfm_ch[c]) for c in lfm_ch]:
            j = pd.concat([g, ref], axis=1, keys=['a', 'b']).dropna()
            d[f'r_{name}'] = j.a.corr(j.b) if len(j) > 8 else np.nan
            d[f'n_{name}'] = len(j)
            # deseasonalised (calendar-month means removed)
            if len(j) > 18:
                jm = j - j.groupby(j.index.month).transform('mean')
                d[f'rdes_{name}'] = jm.a.corr(jm.b)
        for sta in ['KMPB', 'BRIC', 'B045', 'HAYF']:
            for ob in ['1-4Hz', '2-8Hz']:
                if band in ('1-4Hz', '2-8Hz') and ob != band:
                    continue
                o = oth[(oth.sta == sta) & (oth.band == ob) & (oth.combo == combo)].set_index('month').dvv_pct
                j = pd.concat([g, o], axis=1, keys=['a', 'b']).dropna()
                d[f'r_{sta}_{ob}'] = j.a.corr(j.b) if len(j) > 8 else np.nan
                d[f'n_{sta}_{ob}'] = len(j)
                if len(j) > 18:
                    jm = j - j.groupby(j.index.month).transform('mean')
                    d[f'rdes_{sta}_{ob}'] = jm.a.corr(jm.b)
        rows.append(d)
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, 'jewt_noise_correlations.csv'), index=False)
    # --- step table: JEWT noise vs LFE coda vs KMPB/BRIC noise
    pbc = pd.read_csv(os.path.join(ROOT, 'data/dvv/pool_clean/params_by_channel.csv'))
    pbc = pbc[pbc.sta == 'BK.JEWT'][['chan', 'step_m7_pct', 'step_m7_pct_se', 'annual_amp_pct', 'annual_peak_doy',
                                       'trend_pct_per_yr', 'trend_pct_per_yr_se']]
    pbc.to_csv(os.path.join(OUT, 'lfe_coda_jewt_params.csv'), index=False)
    st = S[S.event == 'M7'][['prod', 'band', 'combo', 'step_pct', 'se_mbb_pct', 'placebo_sd_pct', 'z_placebo']].assign(sta='JEWT')
    o = othS[(othS.event == 'M7') & othS.sta.isin(['KMPB', 'BRIC', 'B045', 'HAYF'])][['sta', 'band', 'combo', 'step_pct', 'se_mbb_pct', 'placebo_sd_pct', 'z_placebo']].assign(prod='M(2019-21 ref)')
    pd.concat([st, o]).to_csv(os.path.join(OUT, 'jewt_vs_others_m7_steps.csv'), index=False)
    # --- KMPB/BRIC/B045/HAYF over the JEWT period: annual amplitude + trend with the same fit
    fr = []
    oth10 = pd.read_csv(os.path.join(ROOT, 'data/sci/noise_dvv/noise_dvv_10day.csv'), parse_dates=['t'])
    rng = np.random.default_rng(3)
    for (sta, band, combo), g in oth10.groupby(['sta', 'band', 'combo']):
        g = g[g.t >= pd.Timestamp(START)]
        if g.dvv_pct.notna().sum() < 30:
            continue
        saveb = dict(BREAKS)
        BREAKS.clear()
        if sta == 'BRIC':
            BREAKS['reinstall_2024-05-21'] = '2024-05-21'
        if sta == 'KMPB':
            BREAKS['maint_2023-10-17'] = '2023-10-17'
        o_, _ = fit_series(pd.DatetimeIndex(g.t), g.dvv_pct.values, rng, nb=300)
        BREAKS.clear()
        BREAKS.update(saveb)
        fr.append(dict(sta=sta, band=band, combo=combo, period=f'{START}..{g.t.max().date()}', **o_))
    pd.DataFrame(fr).to_csv(os.path.join(OUT, 'others_longterm_fit_same_period.csv'), index=False)
    # --- noise power (source check): add log10 power as a regressor in the 10-d fit
    pw = pd.read_csv(os.path.join(OUT, 'jewt_noise_power.csv'), parse_dates=['day']).set_index('day')
    pr = []
    pcol = {'0.1-1Hz': 'day5_0.1-0.3', '1-3Hz': 'night20_1-3', '3-5Hz': 'night20_3-5', '1-4Hz': 'night20_1-4', '2-8Hz': 'night20_2-8'}
    for (prod, band, combo), g in T10.groupby(['prod', 'band', 'combo']):
        base = pcol[band]
        lp = np.log10(pw[[c for c in pw.columns if c.startswith(base + '_')]].mean(axis=1))
        t = pd.DatetimeIndex(g.t)
        # 10-d mean log power over the stack's days (stack centre +-4.5 d)
        lpp = np.array([lp[(lp.index >= tt - pd.Timedelta(days=4.5)) & (lp.index <= tt + pd.Timedelta(days=4.5))].mean() for tt in t])
        y = g.dvv_pct.values
        ok = np.isfinite(y) & np.isfinite(lpp)
        X, names = _design(t[ok])
        X2 = np.c_[X, lpp[ok] - np.nanmean(lpp[ok])]
        b1 = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        b2 = np.linalg.lstsq(X2, y[ok], rcond=None)[0]
        r = np.corrcoef(y[ok] - X @ b1 + b1[-1] * X[:, -1] * 0, lpp[ok])[0, 1]
        res = y[ok] - X @ b1
        pr.append(dict(prod=prod, band=band, combo=combo, power_band=base, n=int(ok.sum()),
                       r_resid_vs_logpower=np.corrcoef(res, lpp[ok])[0, 1], r_raw_vs_logpower=np.corrcoef(y[ok], lpp[ok])[0, 1],
                       coef_pct_per_decade=b2[-1], step_M7_nopower=b1[names.index('step_M7_perm')],
                       step_M7_withpower=b2[names.index('step_M7_perm')],
                       annual_amp_nopower=np.hypot(b1[2], b1[3]), annual_amp_withpower=np.hypot(b2[2], b2[3]),
                       trend_nopower=b1[1], trend_withpower=b2[1]))
    PR = pd.DataFrame(pr)
    PR.to_csv(os.path.join(OUT, 'jewt_noise_power_regression.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 60, 'display.max_rows', 300):
        print(C.round(2).to_string())
        print(pbc.round(3).to_string())
        print(pd.DataFrame(fr)[['sta', 'band', 'combo', 'n', 'trend_per_yr', 'trend_per_yr_se', 'annual_amp_pct', 'annual_amp_se',
                                'annual_peak_doy', 'step_M7_perm', 'step_M7_perm_se', 'tau_days']].round(3).to_string())
        print(PR.round(3).to_string())


# ----------------------------------------------------------------------------- figures
PANELS = [('low', '0.1-1Hz'), ('day', '1-3Hz'), ('day', '3-5Hz'), ('M', '1-4Hz'), ('M', '2-8Hz')]
CCOL = {'ALL': 'C3', 'AC': 'C0', 'SC': 'C2'}


def stage_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_10day.csv'), parse_dates=['t'])
    Mo = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_monthly.csv'), parse_dates=['month'])
    D = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_daily_5day.csv'), parse_dates=['t'])
    S = pd.read_csv(os.path.join(OUT, 'jewt_noise_steps.csv'))
    R = pd.read_csv(os.path.join(OUT, 'jewt_noise_recovery_windows.csv'))
    L = pd.read_csv(os.path.join(OUT, 'jewt_noise_lapse_steps.csv'))
    F = pd.read_csv(os.path.join(OUT, 'jewt_noise_longterm_fit.csv'))
    meta = json.load(open(os.path.join(OUT, 'jewt_noise_meta.json')))['meta']
    lfe = _lfe_jewt()
    oth = pd.read_csv(os.path.join(ROOT, 'data/sci/noise_dvv/noise_dvv_monthly.csv'), parse_dates=['month'])
    evl = [pd.Timestamp(v) for v in EVENTS.values()]

    def deco(ax):
        for e in evl:
            ax.axvline(e, color='r' if e == M7 else 'salmon', lw=1 if e == M7 else 0.7, ls='-' if e == M7 else '--')
        for v in BREAKS.values():
            ax.axvline(pd.Timestamp(v), color='0.4', ls=':', lw=1)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.grid(alpha=0.25)

    # 1) full time series
    fig, axs = plt.subplots(len(PANELS), 1, figsize=(12, 2.7 * len(PANELS)), sharex=True)
    for ax, (pk, bn) in zip(axs, PANELS):
        for c in (['SC'] if pk == 'low' else ['AC', 'SC', 'ALL']):
            g = T[(T['prod'] == pk) & (T.band == bn) & (T.combo == c)]
            ax.plot(g.t, g.dvv_pct, '.', ms=3, color=CCOL[c], alpha=0.45)
            gm = Mo[(Mo['prod'] == pk) & (Mo.band == bn) & (Mo.combo == c)]
            ax.plot(gm.month + pd.Timedelta(days=15), gm.dvv_pct, '-', color=CCOL[c], lw=1.3, label=f'{c} monthly')
        if bn in ('1-3Hz', '1-4Hz'):
            for ch, d in lfe.items():
                dd = d[d.t >= pd.Timestamp(START) - pd.Timedelta(days=30)]
                ax.plot(dd.t, dd.dvv_pct - dd.dvv_pct[dd.t < M7].mean(), '-', color='k', lw=0.9,
                        alpha={'HHE': 1, 'HHN': 0.6, 'HHZ': 0.35}[ch], label=f'LFE coda {ch}')
        f = F[(F['prod'] == pk) & (F.band == bn) & (F.combo == ('SC' if pk == 'low' else 'ALL'))].iloc[0]
        cod = meta[pk]['coda'][bn]
        ax.set_title(f'{bn}  [{meta[pk]["label"]}; coda {cod[0]:g}-{cod[1]:g} s; pairs {",".join(meta[pk]["pairs_used"][bn])}]  '
                     f'fit ({"SC" if pk == "low" else "ALL"}): annual {f.annual_amp_pct:.3f}+/-{f.annual_amp_se:.3f}% peak doy {f.annual_peak_doy:.0f}; '
                     f'trend {f.trend_per_yr:+.3f}+/-{f.trend_per_yr_se:.3f} %/yr', fontsize=8, loc='left')
        lo, hi = np.nanpercentile(T[(T['prod'] == pk) & (T.band == bn)].dvv_pct, [1, 99])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_ylabel('dv/v (%)')
        deco(ax)
        ax.legend(fontsize=6.5, ncol=6, loc='lower left')
    axs[-1].set_xlim(pd.Timestamp(START), T.t.max() + pd.Timedelta(days=10))
    fig.suptitle('BK.JEWT single-station ambient-noise dv/v (10-d points, monthly lines; ref = 2023-08-23..2024-11-30; '
                 'red = M7, salmon = M4.8-5.6, dotted = StationXML epoch)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_jewt_timeseries.png'), dpi=140)
    plt.close(fig)

    # 2) M7 zoom
    fig, axs = plt.subplots(len(PANELS), 1, figsize=(11, 2.5 * len(PANELS)), sharex=True)
    for ax, (pk, bn) in zip(axs, PANELS):
        for c in (['SC'] if pk == 'low' else ['AC', 'SC']):
            g = D[(D['prod'] == pk) & (D.band == bn) & (D.combo == c) & (D.block_days == 5)]
            ax.errorbar(g.t, g.dvv_pct, yerr=g.err_pct, fmt='o-', ms=3, lw=0.9, color=CCOL[c], label=f'{c} 5-d')
            g1 = D[(D['prod'] == pk) & (D.band == bn) & (D.combo == c) & (D.block_days == 1)]
            ax.plot(g1.t, g1.dvv_pct, '.', ms=2, color=CCOL[c], alpha=0.35)
        for c in (['SC'] if pk == 'low' else ['AC', 'SC']):
            st = S[(S['prod'] == pk) & (S.band == bn) & (S.combo == c) & (S.event == 'M7')].iloc[0]
            rs = R[(R['prod'] == pk) & (R.band == bn) & (R.combo == c) & (R.window == 'silence d1-25')].iloc[0]
            ax.text(0.005, 0.04 + (0.12 if c == 'SC' else 0), f'{c}: 30-d step {st.step_pct:+.3f} (MBB {st.se_mbb_pct:.3f}, placebo sd '
                    f'{st.placebo_sd_pct:.3f}, z {st.z_placebo:+.1f}); silence {rs.minus_pre30_pct:+.3f} (null sd {rs.null_sd_pct:.3f})',
                    transform=ax.transAxes, fontsize=7, color=CCOL[c])
        ax.axvspan(M7, SILENCE_END, color='#fde0dd', zorder=0)
        ax.set_title(bn + ' ' + meta[pk]['label'], fontsize=8, loc='left')
        g = D[(D['prod'] == pk) & (D.band == bn) & (D.block_days == 5) & D.combo.isin(['SC', 'AC'])
              & (D.t > pd.Timestamp('2024-09-01')) & (D.t < pd.Timestamp('2025-07-01'))]
        lo, hi = np.nanpercentile(g.dvv_pct, [2, 98])
        ax.set_ylim(lo - 0.4 * (hi - lo), hi + 0.15 * (hi - lo))
        ax.set_ylabel('dv/v (%)')
        deco(ax)
        ax.legend(fontsize=7, loc='upper right')
    axs[-1].set_xlim(pd.Timestamp('2024-09-01'), pd.Timestamp('2025-07-01'))
    fig.suptitle('BK.JEWT noise dv/v around the 2024-12-05 M7 (shaded: LFE silence to Dec 30)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_jewt_m7.png'), dpi=140)
    plt.close(fig)

    # 3) steps vs frequency & lapse; recovery windows; comparison with other stations / LFE coda
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    fc = {'0.1-1Hz': 0.32, '1-3Hz': 1.73, '3-5Hz': 3.87, '1-4Hz': 2.0, '2-8Hz': 4.0}
    for c, dx in (('AC', 0.97), ('SC', 1.03), ('ALL', 1.0)):
        for pk, bn in PANELS:
            r = S[(S['prod'] == pk) & (S.band == bn) & (S.combo == c) & (S.event == 'M7')]
            if r.empty or not np.isfinite(r.step_pct.iloc[0]):
                continue
            r = r.iloc[0]
            mk = 's' if pk == 'M' else 'o'
            ax.errorbar(fc[bn] * dx, r.step_pct, yerr=r.placebo_sd_pct, fmt=mk, color=CCOL[c], ms=5, capsize=2,
                        label=c if (pk, bn) == ('day', '1-3Hz') else None)
    lp = pd.read_csv(os.path.join(ROOT, 'data/dvv/pool_clean/params_by_channel.csv'))
    lp = lp[lp.sta == 'BK.JEWT']
    for i, (_, r) in enumerate(lp.iterrows()):
        ax.errorbar(5.5 + 0.4 * i, r.step_m7_pct, yerr=r.step_m7_pct_se, fmt='D', color='k', ms=4)
        ax.text(5.5 + 0.4 * i, r.step_m7_pct + 0.06, r.chan[-1], fontsize=6, ha='center')
    ax.set_xscale('log')
    ax.set_xticks([0.3, 1, 2, 4, 6])
    ax.set_xticklabels(['0.3', '1', '2', '4', 'LFE'])
    ax.minorticks_off()
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xlabel('band centre (Hz); squares: method bands 09-11 UTC;\ndiamonds: LFE-coda steps HHE/HHN/HHZ', fontsize=8)
    ax.set_ylabel('M7 30-d step (%)  +/- placebo sd')
    ax.legend(fontsize=7)
    ax.set_title('M7 step vs frequency', fontsize=9)
    ax = axs[1]
    for c in ['AC', 'SC']:
        for i, (pk, bn) in enumerate(PANELS):
            g = L[(L['prod'] == pk) & (L.band == bn) & (L.combo == c)]
            for j, (_, r) in enumerate(g.iterrows()):
                x = i + (j - 0.5) * 0.3 + (0.06 if c == 'SC' else -0.06)
                ax.errorbar(x, r.step_pct, yerr=r.placebo_sd_pct, fmt='o' if j == 0 else 's', color=CCOL[c], ms=4,
                            mfc='none' if j else CCOL[c], label=f'{c} {"early" if j == 0 else "late"}' if i == 1 else None)
    ax.set_xticks(range(len(PANELS)))
    ax.set_xticklabels([f'{bn}\n{"/".join(L[(L.band == bn) & (L["prod"] == pk)].coda.unique())}' for pk, bn in PANELS], fontsize=6.5)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.legend(fontsize=7)
    ax.set_title('M7 step: early vs late coda (lapse)', fontsize=9)
    ax = axs[2]
    wins = [w for _, _, w in REC_WINDOWS if w not in ('d1-10', 'd11-25')]
    for c, ls in (('AC', '-'), ('SC', '--')):
        for pk, bn in PANELS:
            g = R[(R['prod'] == pk) & (R.band == bn) & (R.combo == c) & R.window.isin(wins)].set_index('window').loc[wins]
            if g.minus_pre30_pct.isna().all():
                continue
            ax.errorbar(np.arange(len(wins)) + 0.03 * PANELS.index((pk, bn)), g.minus_pre30_pct, yerr=g.null_sd_pct,
                        fmt='o' + ls, ms=3, lw=0.9, label=f'{bn} {c}' + (' (M)' if pk == 'M' else ''))
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels(wins, fontsize=6.5, rotation=30)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_ylabel('mean - 30-d pre mean (%)  +/- null sd')
    ax.legend(fontsize=5.5, ncol=2)
    ax.set_title('Post-M7 windows (not deseasoned; null = same stat at placebo dates)', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_jewt_steps.png'), dpi=140)
    plt.close(fig)

    # 4) comparison with KMPB / BRIC / B045 / HAYF and LFE coda (monthly, same period)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (band, combo) in zip(axs, [('1-4Hz', 'ALL'), ('2-8Hz', 'SC')]):
        for sta, col in [('KMPB', 'C1'), ('BRIC', 'C4'), ('B045', 'C5'), ('HAYF', 'C7')]:
            o = oth[(oth.sta == sta) & (oth.band == band) & (oth.combo == combo) & (oth.month >= pd.Timestamp('2023-06-01'))]
            ax.plot(o.month + pd.Timedelta(days=15), o.dvv_pct - o.dvv_pct[(o.month < M7) & (o.month >= pd.Timestamp('2023-08-01'))].mean(),
                    '-', color=col, lw=1.2, label=f'{sta} noise {band} {combo}')
        for pk, bn, colr in [('M', band, 'C3'), ('day', '1-3Hz', 'C2')]:
            j = Mo[(Mo['prod'] == pk) & (Mo.band == bn) & (Mo.combo == combo)]
            ax.plot(j.month + pd.Timedelta(days=15), j.dvv_pct - j.dvv_pct[j.month < M7].mean(), 'o-', color=colr, lw=2, ms=3,
                    label=f'JEWT noise {bn} {combo} ({meta[pk]["label"]})')
        d = lfe['HHE']
        d = d[d.t >= pd.Timestamp('2023-07-01')]
        ax.plot(d.t, d.dvv_pct - d.dvv_pct[d.t < M7].mean(), 'k-', lw=1.5, label='JEWT LFE coda HHE')
        deco(ax)
        ax.legend(fontsize=6.5, ncol=3, loc='lower left')
        ax.set_ylabel('dv/v (%), pre-M7 mean removed')
    axs[-1].set_xlim(pd.Timestamp('2023-08-01'), pd.Timestamp('2026-12-01'))
    fig.suptitle('JEWT vs other noise stations (method outputs end 2026-06) and LFE coda, monthly', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_jewt_compare.png'), dpi=140)
    plt.close(fig)

    # 5) source check: 0.1-1 Hz dv/v vs microseism power
    fp = os.path.join(OUT, 'jewt_noise_power.csv')
    if os.path.exists(fp):
        pw = pd.read_csv(fp, parse_dates=['day']).set_index('day')
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        g = D[(D['prod'] == 'low') & (D.combo == 'SC') & (D.block_days == 5)]
        ax = axs[0]
        ax.plot(g.t, g.dvv_pct, 'o-', ms=2, lw=0.8, color='C2', label='0.1-1 Hz SC dv/v (5-d)')
        ax.set_ylabel('dv/v (%)')
        ax2 = ax.twinx()
        lp = np.log10(pw[[c for c in pw.columns if c.startswith('day5_0.1-0.3_')]].mean(axis=1))
        ax2.plot(lp.rolling(5, center=True).mean(), '-', color='0.3', lw=0.8, label='log10 0.1-0.3 Hz RMS (5-d)')
        ax2.invert_yaxis()
        ax2.set_ylabel('log10 microseism RMS (inverted)')
        ax.legend(loc='lower left', fontsize=7)
        ax2.legend(loc='lower right', fontsize=7)
        deco(ax)
        ax = axs[1]
        for c, colr in (('AC', 'C0'), ('SC', 'C2')):
            g = D[(D['prod'] == 'day') & (D.band == '1-3Hz') & (D.combo == c) & (D.block_days == 5)]
            ax.plot(g.t, g.dvv_pct, 'o-', ms=2, lw=0.8, color=colr, label=f'1-3 Hz {c} (5-d)')
        ax2 = ax.twinx()
        lp = np.log10(pw[[c for c in pw.columns if c.startswith('night20_1-3_')]].mean(axis=1))
        ax2.plot(lp.rolling(5, center=True).mean(), '-', color='0.3', lw=0.8, label='log10 1-3 Hz RMS 09-11 UTC (5-d)')
        ax2.invert_yaxis()
        ax.legend(loc='lower left', fontsize=7)
        ax2.legend(loc='lower right', fontsize=7)
        deco(ax)
        fig.suptitle('Source check: dv/v vs noise power at JEWT', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, 'sci_noise_dvv_jewt_power.png'), dpi=140)
        plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['local', 'download', 'assemble', 'm7local', 'dvv', 'plot', 'power', 'compare'])
    ap.add_argument('--prods', nargs='*')
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--min-cc', type=float, default=0.3)
    ap.add_argument('--no-notch', action='store_true')
    ap.add_argument('--ref', default='pre', choices=['pre', 'all'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--reverse', action='store_true')
    a = ap.parse_args()
    os.makedirs(DAYS, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    g = globals()
    g['stage_' + a.stage](a)
