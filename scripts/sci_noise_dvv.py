#!/usr/bin/env python3
"""Ambient-noise single-station dv/v (SC cross-component + autocorrelation) at a few
stations, as an independent test of the post-2021 receiver-side decline seen in LFE-coda dv/v.

Stages (run in order):
  download   2 h/day (09:00-11:00 UTC, night local), 2019-01-01..2026-06-30, sequential NCEDC
             bulk requests (PB.B045 2019-2022 from local PNWstore). Low-pass + decimate to
             20 Hz, cache as int32 Steim2 under data/sci/noise_dvv/raw/YYYY/YYYY-MM-DD.mseed
  correlate  per station-day: 120-s segments, per band (1-4, 2-8 Hz) bandpass -> one-bit ->
             spectral whitening; 6 pairs (ZZ,NN,EE autos; EN,EZ,NZ cross), lags +-25 s.
             -> data/sci/noise_dvv/ccf_<STA>.npz   (<=4 processes)
  dvv        reference = mean of daily CCFs 2019-01-01..2021-11-30; stretching in the coda
             (|lag| 2-20 s); 10-day stacks; event steps (30-d pre/post) with moving-block
             bootstrap and placebo dates; annual cycle; 2025-2021 decline; LFE comparison.
  plot       figures/sci_noise_dvv_timeseries.png, figures/sci_noise_dvv_steps.png
  robust     collects sensitivity runs (dvv --no-notch --tag _nonotch; dvv --coda 2 10 --tag
             _coda2-10; dvv --line-thr 2.5 --tag _strictnotch) into noise_dvv_robustness.csv
  (fillpb    optional: PB days missing from PNWstore 2019-2022 re-requested from NCEDC)
Spectral lines: narrow persistent lines (machinery) are notched in the CCFs (see detect_lines).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'          # shared machine: <= 4 single-threaded processes
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.path.join(ROOT, 'data/sci/noise_dvv')
RAW = os.path.join(OUT, 'raw')
FIG = os.path.join(ROOT, 'figures')
START, END = '2019-01-01', '2026-06-30'
H0, H1 = 9, 11            # UTC hours of the daily window
FS = 20.0
STATIONS = {               # name: (net, sta, loc, channel pattern, component map)
    'KMPB': ('NC', 'KMPB', '*', 'HH?', {'Z': 'HHZ', 'N': 'HHN', 'E': 'HHE'}),
    'BRIC': ('BK', 'BRIC', '00', 'HH?', {'Z': 'HHZ', 'N': 'HHN', 'E': 'HHE'}),
    'HAYF': ('BK', 'HAYF', '00', 'HH?', {'Z': 'HHZ', 'N': 'HHN', 'E': 'HHE'}),
    'B045': ('PB', 'B045', '*', 'EH?', {'Z': 'EHZ', 'N': 'EH1', 'E': 'EH2'}),
}
LFE_CHAN_PREFIX = {'KMPB': 'NC.KMPB.', 'BRIC': 'BK.BRIC.', 'HAYF': 'BK.HAYF.', 'B045': 'PB.B045.'}
PAIRS = [('Z', 'Z'), ('N', 'N'), ('E', 'E'), ('E', 'N'), ('E', 'Z'), ('N', 'Z')]
PAIR_NAMES = [a + b for a, b in PAIRS]
BANDS = [(1.0, 4.0), (2.0, 8.0)]
BAND_NAMES = ['1-4Hz', '2-8Hz']
SEG = 120.0               # s
MAXLAG = 60.0             # s (stored lags; dv/v uses the 2-20 s coda, 30-60 s is used to find lines)
LINE_THR = 3.0            # spectral-line threshold (late-lag spectrum / 0.5-Hz running median)
CODA = (2.0, 20.0)        # |lag| window, s
EVENTS = {'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20', 'M7': '2024-12-05'}
REF_END = '2021-11-30'
TAG = ''                  # output-name suffix for sensitivity runs


# ----------------------------------------------------------------------------- download
def _snap_sr(st):
    for tr in st:
        sr = tr.stats.sampling_rate
        if abs(sr - round(sr)) < 1e-4 * sr:
            tr.stats.sampling_rate = float(round(sr))


def _decimate(st):
    """merge, split at gaps, low-pass 8 Hz (zero-phase), decimate 100->20 Hz, int32."""
    from obspy import Stream
    _snap_sr(st)
    st.merge(method=1, fill_value=None)
    st = st.split()
    out = Stream()
    for tr in st:
        if tr.stats.npts < 60 * tr.stats.sampling_rate:
            continue
        tr.data = tr.data.astype('float64')
        tr.detrend('demean')
        tr.taper(0.002, max_length=2.0)
        if abs(tr.stats.sampling_rate - 100.0) < 1e-6:
            tr.filter('lowpass', freq=8.0, corners=6, zerophase=True)
            tr.decimate(5, no_filter=True)
        elif abs(tr.stats.sampling_rate - FS) > 1e-6:
            tr.filter('lowpass', freq=8.0, corners=6, zerophase=True)
            tr.resample(FS)
        tr.data = np.round(tr.data).astype('int32')
        out += tr
    return out


def stage_download(args):
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from download_windows import read_pnwstore
    from obspy import Stream, UTCDateTime
    from obspy.clients.fdsn import Client
    client = Client('NCEDC', timeout=120)
    days = pd.date_range(args.start or START, args.end or END, freq='D')
    logf = open(os.path.join(OUT, 'download_log.csv'), 'a')
    for day in days:
        fn = os.path.join(RAW, f'{day:%Y}', f'{day:%Y-%m-%d}.mseed')
        if os.path.exists(fn) or os.path.exists(fn + '.none'):
            continue
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        t0 = time.time()
        a = day + pd.Timedelta(hours=H0)
        b = day + pd.Timedelta(hours=H1)
        st = Stream()
        remote = []
        for name, (n, s, l, cp, cmap) in STATIONS.items():
            if n == 'PB' and day.year <= 2022:
                try:
                    st += read_pnwstore([(n, s, '', ch) for ch in cmap.values()],
                                        [(a.to_pydatetime(), b.to_pydatetime())], day.year)
                except Exception as e:
                    print('pnwstore fail', day.date(), e, flush=True)
            else:
                remote.append((n, s, l, cp, UTCDateTime(a), UTCDateTime(b)))
        for attempt in range(5):
            try:
                st += client.get_waveforms_bulk(remote)
                break
            except Exception as e:
                if 'No data' in str(e) or '204' in str(e):
                    break
                time.sleep(5 * 2 ** attempt)
        else:
            print('FAILED', day.date(), flush=True)
            continue
        st = _decimate(st)
        if len(st) == 0:
            open(fn + '.none', 'w').close()
        else:
            st.write(fn, format='MSEED', encoding='STEIM2')
        got = sorted({tr.stats.station + '.' + tr.stats.channel for tr in st})
        logf.write(f'{day:%Y-%m-%d},{len(got)},{" ".join(got)},{time.time() - t0:.1f}\n')
        logf.flush()
        time.sleep(0.3)   # be gentle


def stage_fillpb(args):
    """PB days missing in the local PNWstore (2019-2022): try NCEDC instead and add them."""
    from obspy import read, Stream, UTCDateTime
    from obspy.clients.fdsn import Client
    client = Client('NCEDC', timeout=120)
    n, s, l, cp, cmap = STATIONS['B045']
    for day in pd.date_range(START, '2022-12-31', freq='D'):
        fn = os.path.join(RAW, f'{day:%Y}', f'{day:%Y-%m-%d}.mseed')
        st = read(fn) if os.path.exists(fn) else Stream()
        if len(st.select(station=s)) >= 3:
            continue
        a = UTCDateTime(day + pd.Timedelta(hours=H0))
        try:
            new = client.get_waveforms(n, s, '*', cp, a, a + (H1 - H0) * 3600)
        except Exception as e:
            print(day.date(), 'none', str(e)[:40], flush=True)
            time.sleep(0.3)
            continue
        new = _decimate(new)
        if len(new):
            st = Stream([tr for tr in st if tr.stats.station != s]) + new
            st.write(fn, format='MSEED', encoding='STEIM2')
            if os.path.exists(fn + '.none'):
                os.remove(fn + '.none')
            print(day.date(), 'filled', len(new), flush=True)
        time.sleep(0.3)


# ----------------------------------------------------------------------------- correlate
def _cos_taper_band(freqs, f1, f2):
    """flat 1 between f1 and f2, cosine roll-off over 0.5*f1 below and 0.25*f2 above."""
    w = np.zeros_like(freqs)
    lo0, hi0 = 0.5 * f1, min(1.25 * f2, FS / 2 - 0.2)
    m = (freqs >= f1) & (freqs <= f2)
    w[m] = 1
    m = (freqs > lo0) & (freqs < f1)
    w[m] = 0.5 * (1 - np.cos(np.pi * (freqs[m] - lo0) / (f1 - lo0)))
    m = (freqs > f2) & (freqs < hi0)
    w[m] = 0.5 * (1 + np.cos(np.pi * (freqs[m] - f2) / (hi0 - f2)))
    return w


def _day_segments(st, cmap, day):
    """Return array (3, nseg, nseg_samples) for components Z,N,E; NaN rows if missing."""
    from obspy import UTCDateTime
    ns = int(SEG * FS)
    t0 = UTCDateTime(day + pd.Timedelta(hours=H0))
    nseg = int((H1 - H0) * 3600 / SEG)
    arr = np.full((3, nseg, ns), np.nan)
    for ic, c in enumerate('ZNE'):
        for tr in st.select(channel=cmap[c]):
            i0 = int(round((tr.stats.starttime - t0) * FS))
            d = tr.data.astype('float64')
            for k in range(nseg):
                a = k * ns - i0
                if a >= 0 and a + ns <= len(d):
                    arr[ic, k] = d[a:a + ns]
    return arr


def _correlate_day(arr):
    """arr (3, nseg, ns). Returns ccf (nband, npair, nlag), nseg_used (nband, npair)."""
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import uniform_filter1d
    ns = arr.shape[-1]
    nfft = 2 * ns
    freqs = np.fft.rfftfreq(nfft, 1 / FS)
    nl = int(MAXLAG * FS)
    out = np.full((len(BANDS), len(PAIRS), 2 * nl + 1), np.nan, dtype='float32')
    nused = np.zeros((len(BANDS), len(PAIRS)), dtype='int16')
    good = np.isfinite(arr).all(axis=2) & (np.nanstd(arr, axis=2) > 0)      # (3, nseg)
    x = np.where(good[..., None], arr, 0.0)
    x = x - x.mean(axis=2, keepdims=True)
    tap = np.ones(ns)
    nt = int(0.05 * ns)
    tap[:nt] = 0.5 * (1 - np.cos(np.pi * np.arange(nt) / nt))
    tap[-nt:] = tap[:nt][::-1]
    for ib, (f1, f2) in enumerate(BANDS):
        sos = butter(4, [f1, f2], btype='band', fs=FS, output='sos')
        y = sosfiltfilt(sos, x * tap, axis=2)
        # reject transients: segments whose band RMS > 4x the daily median of that component
        rms = np.sqrt((y ** 2).mean(axis=2))
        gb = good.copy()
        for ic in range(3):
            if gb[ic].any():
                med = np.median(rms[ic, gb[ic]])
                gb[ic] &= rms[ic] < 4 * med
        y = np.sign(y) * tap                                    # one-bit
        Y = np.fft.rfft(y, nfft, axis=2)
        W = _cos_taper_band(freqs, f1, f2)
        A = np.abs(Y) + 1e-10
        # autocorrelations: pure-phase whitening would make them deterministic (|Y|^2 = W^2),
        # so autos use a smoothed amplitude (1-Hz running mean): removes the broad spectral
        # shape (lags < ~1 s) but keeps the fine structure that carries the 2-20 s coda.
        k = int(round(1.0 / (freqs[1] - freqs[0])))
        As = uniform_filter1d(A, k, axis=2, mode='nearest')
        Ya = W * Y / As
        Y = W * Y / A                                           # pure-phase whitening (SC pairs)
        for ip, (ca, cb) in enumerate(PAIRS):
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


def _correlate_station(name):
    from obspy import read
    n, s, l, cp, cmap = STATIONS[name]
    days = pd.date_range(START, END, freq='D')
    nl = int(MAXLAG * FS)
    fn_out = os.path.join(OUT, f'ccf_{name}.npz')
    ccf = np.full((len(days), len(BANDS), len(PAIRS), 2 * nl + 1), np.nan, dtype='float32')
    nused = np.zeros((len(days), len(BANDS), len(PAIRS)), dtype='int16')
    for i, day in enumerate(days):
        fn = os.path.join(RAW, f'{day:%Y}', f'{day:%Y-%m-%d}.mseed')
        if not os.path.exists(fn):
            continue
        try:
            st = read(fn).select(network=n, station=s)
        except Exception:
            continue
        if len(st) == 0:
            continue
        arr = _day_segments(st, cmap, day)
        ccf[i], nused[i] = _correlate_day(arr)
    np.savez_compressed(fn_out, days=np.array(days.strftime('%Y-%m-%d'), dtype='U10'), ccf=ccf, nused=nused,
                        pairs=PAIR_NAMES, bands=BAND_NAMES, fs=FS, maxlag=MAXLAG)
    return name, int(np.isfinite(ccf[:, 0, 0, 0]).sum())


def stage_correlate(args):
    names = args.stations or list(STATIONS)
    with ProcessPoolExecutor(max_workers=min(4, len(names))) as ex:
        for r in ex.map(_correlate_station, names):
            print('correlated', r, flush=True)


# ----------------------------------------------------------------------------- dv/v
EPS = np.arange(-0.02, 0.02 + 1e-12, 0.0001)     # dv/v grid (fraction)
LAGS = np.arange(-int(MAXLAG * FS), int(MAXLAG * FS) + 1) / FS
CODA_MASK = (np.abs(LAGS) >= CODA[0]) & (np.abs(LAGS) <= CODA[1])
UP = 4                                           # FFT upsampling before stretching (20 -> 80 Hz)
LAGS_U = LAGS[0] + np.arange(len(LAGS) * UP) / (FS * UP)
MASK_U = (np.abs(LAGS_U) >= CODA[0]) & (np.abs(LAGS_U) <= CODA[1])
_STRETCH_CACHE = {}


def _stretched_refs(ref):
    """R_eps(t) = ref(t (1+eps)) on the coda mask; rows normalised (zero mean, unit norm)."""
    from scipy.signal import resample
    t = LAGS_U[MASK_U]
    refu = resample(ref, len(LAGS_U))
    R = np.array([np.interp(t * (1 + e), LAGS_U, refu) for e in EPS])
    R -= R.mean(axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return R


def stretch(ref, X):
    """X (K, L) current traces. Model cur(t) = ref(t(1+e)), e = dv/v.
    Returns dvv (K,), cc (K,) with parabolic refinement."""
    X = np.atleast_2d(X)
    out_e = np.full(len(X), np.nan)
    out_c = np.full(len(X), np.nan)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(ref).all()
    if not ok.any():
        return out_e, out_c
    R = _stretched_refs(ref)
    from scipy.signal import resample
    Y = resample(X[ok], len(LAGS_U), axis=1)[:, MASK_U]
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    C = R @ Y.T                                   # (neps, K)
    j = np.argmax(C, axis=0)
    jj = np.clip(j, 1, len(EPS) - 2)
    k = np.arange(C.shape[1])
    y0, y1, y2 = C[jj - 1, k], C[jj, k], C[jj + 1, k]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    off = np.clip(off, -1, 1)
    e = EPS[jj] + off * (EPS[1] - EPS[0])
    e[(j == 0) | (j == len(EPS) - 1)] = np.nan    # hit the grid edge
    out_e[ok] = e
    out_c[ok] = y1 - 0.25 * (y0 - y2) * off
    return out_e, out_c


def weaver_err(cc, band):
    """Weaver et al. (2011) rms error of stretching (fraction)."""
    f1, f2 = band
    T = 1.0 / (f2 - f1)
    wc = 2 * np.pi * np.sqrt(f1 * f2)
    t1, t2 = CODA
    cc = np.clip(cc, 1e-3, 0.999)
    return np.sqrt(1 - cc ** 2) / (2 * cc) * np.sqrt(6 * np.sqrt(np.pi / 2) * T / (wc ** 2 * (t2 ** 3 - t1 ** 3)))


def combine(dvv, err, axis=-1):
    """inverse-variance mean over pairs (fraction); returns mean, err, scatter-based err."""
    w = np.where(np.isfinite(dvv) & np.isfinite(err), 1 / err ** 2, 0.0)
    d = np.where(w > 0, dvv, 0.0)
    sw = w.sum(axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = (w * d).sum(axis=axis) / sw
        e = 1 / np.sqrt(sw)
    m[sw == 0] = np.nan
    e[sw == 0] = np.nan
    return m, e


COMBOS = {'ALL': list(range(6)), 'SC': [3, 4, 5], 'AC': [0, 1, 2]}


def load_ccf(name):
    z = np.load(os.path.join(OUT, f'ccf_{name}.npz'))
    days = pd.to_datetime(z['days'])
    ccf = z['ccf'].astype('float64')
    nused = z['nused']
    ccf[nused < 20] = np.nan                     # need >= 40 min of usable data that day
    return days, ccf


NF = 8192
FREQ_N = np.fft.rfftfreq(NF, 1 / FS)
# mute |lag| < 1.5 s (cosine ramp to 2 s) before notching so the zero-lag peak of the
# autocorrelations does not leak notch ringing into the coda window
MUTE = np.sin(np.pi / 2 * np.clip((np.abs(LAGS) - 1.5) / 0.5, 0, 1)) ** 2


def detect_lines(days, ccf):
    """Persistent narrow spectral lines (machinery, PBO line noise) make CCF codas ring
    sinusoidally and bias stretching. A line keeps a fixed frequency at all lags, whereas
    random spectral peaks of the (decayed) coda / noise floor do not repeat between disjoint
    lag windows. Test: in yearly CCF stacks, amplitude spectra of the 30-45 s and 45-60 s
    lag windows, each divided by its 0.5-Hz running median; a bin is a line if BOTH ratios
    exceed LINE_THR (false-alarm ~4e-6 per bin for Rayleigh noise) for any pair, in at
    least 2 years. Returns a smooth notch weight (1 = keep) on FREQ_N and the intervals."""
    from scipy.ndimage import median_filter, binary_dilation, gaussian_filter1d
    a = np.abs(LAGS)
    w1, w2 = (30.0, 45.0), (45.0, 60.0)
    taps = [np.where((a >= lo) & (a <= hi), np.sin(np.pi * (a - lo) / (hi - lo)) ** 2, 0) for lo, hi in (w1, w2)]
    k = int(round(0.5 / (FREQ_N[1] - FREQ_N[0])))
    count = np.zeros(len(FREQ_N), int)
    for yr in sorted(set(days.year)):
        m = days.year == yr
        if np.isfinite(ccf[m, 0, 0, 0]).sum() < 60:
            continue
        with np.errstate(invalid='ignore'):
            S = np.nanmean(ccf[m], axis=0)
        line = np.zeros(len(FREQ_N), bool)
        for b, (f1, f2) in enumerate(BANDS):
            inb = (FREQ_N >= f1) & (FREQ_N <= f2)
            for p in range(len(PAIRS)):
                if not np.isfinite(S[b, p]).all():
                    continue
                hit = inb.copy()
                for tap in taps:
                    A = np.abs(np.fft.rfft(S[b, p] * tap, NF))
                    med = median_filter(A, size=k, mode='nearest')
                    r = A / (med + 1e-20)
                    # allow +-1 bin frequency jitter between the two windows
                    hit &= np.maximum.reduce([r, np.roll(r, 1), np.roll(r, -1)]) > LINE_THR
                line |= hit
        count += line
    line = count >= 2
    line = binary_dilation(line, iterations=int(round(0.03 / (FREQ_N[1] - FREQ_N[0]))))
    keep = 1 - np.clip(gaussian_filter1d(line.astype(float), 2), 0, 1)
    keep[line] = 0
    lab = np.diff(np.r_[0, line.astype(int), 0])
    a, b_ = np.where(lab == 1)[0], np.where(lab == -1)[0] - 1
    return keep, [(round(float(FREQ_N[i]), 2), round(float(FREQ_N[j]), 2)) for i, j in zip(a, b_)]


def notch(ccf, keep, chunk=200):
    """apply the central-lag mute and notch weights along the lag axis (chunked over days)."""
    out = np.empty(ccf.shape, dtype='float32')
    for i in range(0, len(ccf), chunk):
        c = ccf[i:i + chunk]
        X = np.fft.rfft(np.nan_to_num(c) * MUTE, NF, axis=-1) * keep
        o = np.fft.irfft(X, NF, axis=-1)[..., :ccf.shape[-1]]
        o[~np.isfinite(c)] = np.nan
        out[i:i + chunk] = o
    return out


def reference(days, ccf):
    m = days <= pd.Timestamp(REF_END)
    with np.errstate(invalid='ignore'):
        return np.nanmean(ccf[m], axis=0), int(np.isfinite(ccf[m, 0, 0, 0]).sum())


def stack_groups(days, ccf, key, min_days):
    """key: array of group labels per day. Returns labels, stacks (G,B,P,L), ndays (G,B,P)."""
    labs = pd.unique(key[pd.notna(key)])
    S = np.full((len(labs),) + ccf.shape[1:], np.nan)
    N = np.zeros((len(labs),) + ccf.shape[1:3], dtype=int)
    for g, lab in enumerate(labs):
        sel = ccf[key == lab]
        n = np.isfinite(sel[..., 0]).sum(axis=0)
        with np.errstate(invalid='ignore'):
            m = np.nanmean(sel, axis=0)
        m[n < min_days] = np.nan
        S[g], N[g] = m, n
    return labs, S, N


def measure_stacks(ref, S):
    """S (G,B,P,L) -> dvv, cc, err (G,B,P) as fractions."""
    G, B, P, L = S.shape
    dv = np.full((G, B, P), np.nan)
    cc = np.full((G, B, P), np.nan)
    for b in range(B):
        for p in range(P):
            dv[:, b, p], cc[:, b, p] = stretch(ref[b, p], S[:, b, p])
    er = np.stack([weaver_err(cc[:, b], BANDS[b]) for b in range(B)], axis=1)
    return dv, cc, er


def pair_quality(cc10, days10):
    """median CC of 10-day stacks vs reference within the reference period, per (B,P)."""
    m = days10 <= pd.Timestamp(REF_END)
    with np.errstate(invalid='ignore'):
        return np.nanmedian(cc10[m], axis=0)


def mbb_window(ccf_win, nboot, block, rng):
    """moving-block bootstrap stacks of a (D,B,P,L) window -> (nboot,B,P,L)."""
    D = len(ccf_win)
    nb = int(np.ceil(D / block))
    starts = rng.integers(0, D - block + 1, size=(nboot, nb))
    idx = (starts[..., None] + np.arange(block)).reshape(nboot, -1)[:, :D]
    with np.errstate(invalid='ignore'):
        return np.nanmean(ccf_win[idx], axis=1)


def event_step(days, ccf, ref, te, good, half=30, skip=1, nboot=0, rng=None):
    """dv/v(post) - dv/v(pre) in %, per band and combo, using pairs with good[b,p].
    pre = [te-half, te-1], post = [te+1+skip-1 .. te+half+skip-1] (event day excluded)."""
    te = pd.Timestamp(te)
    pre = (days >= te - pd.Timedelta(days=half)) & (days < te)
    post = (days > te + pd.Timedelta(days=skip - 1)) & (days <= te + pd.Timedelta(days=half + skip - 1))
    res = {}
    stacks = []
    for msk in (pre, post):
        w = ccf[msk]
        if np.isfinite(w[:, 0, :, 0]).any(axis=1).sum() < half // 3:
            return None
        base = np.nanmean(w, axis=0)[None]
        if nboot:
            base = np.concatenate([base, mbb_window(w, nboot, 5, rng)], axis=0)
        stacks.append(base)
    out = []
    for S in stacks:
        dv, cc, er = measure_stacks(ref, S)
        er = np.where(good[None], er, np.nan)
        out.append({c: combine(dv[:, :, ix], er[:, :, ix])[0] for c, ix in COMBOS.items()})
    for c in COMBOS:
        d = (out[1][c] - out[0][c]) * 100          # (1+nboot, B) in %
        res[c] = d
    return res


def _dvv_station(job):
    name, min_cc, nboot, no_notch, seed = job
    rng = np.random.default_rng(seed)
    lfe = pd.read_csv(os.path.join(ROOT, 'data/sci/decline/monthly_bins.csv'))
    lfe['month'] = pd.to_datetime(lfe.month)
    rows10, rowsM, rowsP, steps, longt, summ = [], [], [], [], [], {}
    days, ccf = load_ccf(name)
    keep, lines = detect_lines(days, ccf)
    print(name, 'notched lines (Hz):', lines, flush=True)
    if not no_notch:
        ccf = notch(ccf, keep)
    ref, nref = reference(days, ccf)
    print(name, 'reference days', nref, flush=True)
    # ---- 10-day stacks
    k10 = ((days - pd.Timestamp(START)).days // 10).values
    labs, S10, N10 = stack_groups(days, ccf, k10, 5)
    t10 = pd.Timestamp(START) + pd.to_timedelta(labs * 10 + 5, unit='D')
    dv10, cc10, er10 = measure_stacks(ref, S10)
    q = pair_quality(cc10, t10)
    good = q >= min_cc
    print(name, 'pair median CC (10-d, ref period):\n', np.round(q, 2), flush=True)
    er10g = np.where(good[None], er10, np.nan)
    for b in range(len(BANDS)):
        for p in range(len(PAIRS)):
            for g in range(len(t10)):
                rowsP.append(dict(sta=name, band=BAND_NAMES[b], pair=PAIR_NAMES[p], t=t10[g],
                                  dvv_pct=dv10[g, b, p] * 100, cc=cc10[g, b, p],
                                  err_pct=er10[g, b, p] * 100, ndays=N10[g, b, p],
                                  pair_ref_cc=q[b, p], used=bool(good[b, p])))
    for c, ix in COMBOS.items():
        m, e = combine(dv10[:, :, ix], er10g[:, :, ix])
        for b in range(len(BANDS)):
            for g in range(len(t10)):
                rows10.append(dict(sta=name, band=BAND_NAMES[b], combo=c, t=t10[g],
                                   dvv_pct=m[g, b] * 100, err_pct=e[g, b] * 100,
                                   ndays=int(N10[g, b, ix].max())))
    # ---- monthly stacks
    km = days.to_period('M').astype(str).values
    labsM, SM, NM = stack_groups(days, ccf, km, 8)
    tM = pd.to_datetime(labsM)
    dvM, ccM, erM = measure_stacks(ref, SM)
    erMg = np.where(good[None], erM, np.nan)
    lf = lfe[lfe.chan.str.startswith(LFE_CHAN_PREFIX[name])]
    lfm = lf.groupby('month').dvv_pct.median()
    for c, ix in COMBOS.items():
        m, e = combine(dvM[:, :, ix], erMg[:, :, ix])
        for b in range(len(BANDS)):
            for g in range(len(tM)):
                rowsM.append(dict(sta=name, band=BAND_NAMES[b], combo=c, month=tM[g],
                                  dvv_pct=m[g, b] * 100, err_pct=e[g, b] * 100,
                                  lfe_dvv_pct=lfm.get(tM[g], np.nan)))
    # ---- event steps with moving-block bootstrap, placebo and same-season placebo
    ev_res = {}
    for ev, te in EVENTS.items():
        r = event_step(days, ccf, ref, te, good, nboot=nboot, rng=rng)
        ev_res[ev] = r
    first = days[np.isfinite(ccf[:, 0, :, 0]).any(axis=1)][0]
    pl_dates = pd.date_range(first + pd.Timedelta(days=31), pd.Timestamp('2021-11-15'), freq='5D')
    plac = {c: [] for c in COMBOS}
    for td in pl_dates:
        r = event_step(days, ccf, ref, td, good)
        if r is not None:
            for c in COMBOS:
                plac[c].append(r[c][0])
    plac = {c: np.array(v) for c, v in plac.items()}
    np.savez(os.path.join(OUT, f'placebo_{name}{TAG}.npz'), dates=np.array(pl_dates.strftime('%Y-%m-%d'), dtype='U10'),
             **{c: v for c, v in plac.items()})
    for ev, te in EVENTS.items():
        r = ev_res[ev]
        te = pd.Timestamp(te)
        seas = []
        for yr in range(2019, 2027):
            td = te.replace(year=yr)
            if any(abs((td - pd.Timestamp(x)).days) < 45 for x in EVENTS.values()):
                continue
            rr = event_step(days, ccf, ref, td, good)
            if rr is not None:
                seas.append((yr, rr))
        for c in COMBOS:
            for b in range(len(BANDS)):
                if r is None:
                    continue
                boot = r[c][1:, b]
                pv = plac[c][:, b] if len(plac[c]) else np.array([np.nan])
                pv = pv[np.isfinite(pv)]
                sd_p = np.std(pv) if len(pv) > 5 else np.nan
                st_ = r[c][0, b]
                steps.append(dict(sta=name, band=BAND_NAMES[b], combo=c, event=ev, step_pct=st_,
                                  se_mbb_pct=np.nanstd(boot), placebo_sd_pct=sd_p,
                                  n_placebo=len(pv), z_placebo=st_ / sd_p,
                                  frac_placebo_le=(np.mean(pv <= st_) if len(pv) else np.nan),
                                  same_season=';'.join(f'{yr}:{rr[c][0, b]:+.3f}' for yr, rr in seas)))
    # ---- long-term: annual means (as LFE D_2021_2025), fit annual + 3 steps
    for c, ix in COMBOS.items():
        m, e = combine(dv10[:, :, ix], er10g[:, :, ix])
        for b in range(len(BANDS)):
            y = m[:, b] * 100
            ok = np.isfinite(y)
            tt, yy = t10[ok], y[ok]
            yrs = tt.year.values
            ann = pd.Series(yy).groupby(yrs).mean()
            nper = pd.Series(yy).groupby(yrs).size()
            # block bootstrap of the annual-mean difference (blocks of 9 stacks ~ 90 d)
            def boot_diff(y1, y2, nb=500):
                a, bb = yy[yrs == y1], yy[yrs == y2]
                if len(a) < 9 or len(bb) < 9:
                    return np.nan, np.nan
                out = []
                for _ in range(nb):
                    rs = []
                    for v in (a, bb):
                        st0 = rng.integers(0, len(v) - 9 + 1, size=int(np.ceil(len(v) / 9)))
                        rs.append(np.concatenate([v[s:s + 9] for s in st0])[:len(v)].mean())
                    out.append(rs[1] - rs[0])
                return ann[y2] - ann[y1], np.std(out)
            D, Dse = boot_diff(2021, 2025)
            P1, P1se = boot_diff(2019, 2020)
            P2, P2se = boot_diff(2020, 2021)
            ref_mean = yy[tt <= pd.Timestamp(REF_END)].mean()
            late = yy[(tt >= pd.Timestamp('2025-01-01')) & (tt < pd.Timestamp('2026-07-01'))].mean()
            # regression: c + annual sin/cos + steps (annual fitted on all data)
            ty = (tt - pd.Timestamp('2019-01-01')).days.values / 365.25
            Xc = [np.ones_like(ty), np.cos(2 * np.pi * ty), np.sin(2 * np.pi * ty)]
            names_ = ['c', 'acos', 'asin']
            for ev, te in EVENTS.items():
                Xc.append((tt > pd.Timestamp(te)).astype(float))
                names_.append('step_' + ev)
            X = np.array(Xc).T
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            resid = yy - X @ beta
            bb_ = []
            for _ in range(500):   # moving-block bootstrap of residuals, block 9
                st0 = rng.integers(0, len(resid) - 9 + 1, size=int(np.ceil(len(resid) / 9)))
                rr = np.concatenate([resid[s:s + 9] for s in st0])[:len(resid)]
                bb_.append(np.linalg.lstsq(X, X @ beta + rr, rcond=None)[0])
            bse = np.std(bb_, axis=0)
            amp = np.hypot(beta[1], beta[2])
            # phase: day-of-year of maximum dv/v (t=0 at Jan 1)
            peak_doy = (np.degrees(np.arctan2(beta[2], beta[1])) % 360) / 360 * 365.25
            # annual cycle from pre-Petrolia data only (with a linear trend)
            pm = tt <= pd.Timestamp(REF_END)
            Xp = np.array([np.ones(pm.sum()), ty[pm], np.cos(2 * np.pi * ty[pm]), np.sin(2 * np.pi * ty[pm])]).T
            bp = np.linalg.lstsq(Xp, yy[pm], rcond=None)[0]
            lfe_D = np.nan
            lfy = lf.assign(y=lf.month.dt.year).groupby(['chan', 'y']).dvv_pct.mean().unstack()
            if 2021 in lfy and 2025 in lfy:
                lfe_D = float((lfy[2025] - lfy[2021]).median())
            longt.append(dict(sta=name, band=BAND_NAMES[b], combo=c, n_10d=len(yy),
                              D_2021_2025_pct=D, D_se_pct=Dse, z_D=D / Dse,
                              placebo_D_2019_2020=P1, placebo_D_2020_2021=P2,
                              late_minus_ref_pct=late - ref_mean,
                              sd10_ref_pct=np.std(yy[pm]),
                              fit_amp_pct=amp, fit_peak_doy=peak_doy,
                              pre_amp_pct=np.hypot(bp[2], bp[3]),
                              pre_peak_doy=(np.degrees(np.arctan2(bp[3], bp[2])) % 360) / 360 * 365.25,
                              **{f'fit_{n_}': v for n_, v in zip(names_[3:], beta[3:])},
                              **{f'fit_{n_}_se': v for n_, v in zip(names_[3:], bse[3:])},
                              lfe_D_2021_2025_pct=lfe_D,
                              annual_means=';'.join(f'{k}:{v:+.3f}(n{nper[k]})' for k, v in ann.items())))
    summ[name] = dict(n_ref_days=nref, coda_s=CODA, line_thr=LINE_THR,
                      notched_frac={BAND_NAMES[b]: round(float(1 - keep[(FREQ_N > f1) & (FREQ_N < f2)].mean()), 3)
                                    for b, (f1, f2) in enumerate(BANDS)}, notched_lines_hz=lines, notch_applied=not no_notch, pair_ref_cc={BAND_NAMES[b]: dict(zip(PAIR_NAMES, np.round(q[b], 3).tolist()))
                                                     for b in range(len(BANDS))})
    return rows10, rowsM, rowsP, steps, longt, summ


def stage_dvv(args):
    rows10, rowsM, rowsP, steps, longt, summ = [], [], [], [], [], {}
    names = args.stations or list(STATIONS)
    jobs = [(n, args.min_cc, args.nboot, args.no_notch, 42 + i) for i, n in enumerate(names)]
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        for r in ex.map(_dvv_station, jobs):
            for acc, part in zip((rows10, rowsM, rowsP, steps, longt), r[:5]):
                acc.extend(part)
            summ.update(r[5])
    pd.DataFrame(rows10).to_csv(os.path.join(OUT, f'noise_dvv_10day{TAG}.csv'), index=False)
    pd.DataFrame(rowsP).to_csv(os.path.join(OUT, f'noise_dvv_10day_pairs{TAG}.csv'), index=False)
    M = pd.DataFrame(rowsM)
    M.to_csv(os.path.join(OUT, f'noise_dvv_monthly{TAG}.csv'), index=False)
    S = pd.DataFrame(steps)
    S.to_csv(os.path.join(OUT, f'noise_dvv_steps{TAG}.csv'), index=False)
    L = pd.DataFrame(longt)
    L.to_csv(os.path.join(OUT, f'noise_dvv_longterm{TAG}.csv'), index=False)
    # LFE comparison on monthly data (2019-2026): correlation and month-binned steps
    comp = []
    for (sta, band, c), g in M.groupby(['sta', 'band', 'combo']):
        g = g.set_index('month').sort_index()
        ok = g.dvv_pct.notna() & g.lfe_dvv_pct.notna()
        r = np.corrcoef(g.dvv_pct[ok], g.lfe_dvv_pct[ok])[0, 1] if ok.sum() > 10 else np.nan
        # deseasonalised correlation (remove calendar-month means)
        gm = g[ok].copy()
        mo = gm.index.month
        rd = np.corrcoef(gm.dvv_pct - gm.dvv_pct.groupby(mo).transform('mean'),
                         gm.lfe_dvv_pct - gm.lfe_dvv_pct.groupby(mo).transform('mean'))[0, 1] if ok.sum() > 24 else np.nan
        def mstep(col, te):
            te = pd.Timestamp(te).to_period('M').to_timestamp()
            pre = g[col].loc[te - pd.DateOffset(months=2):te - pd.DateOffset(months=1)].mean()
            post = g[col].loc[te + pd.DateOffset(months=1):te + pd.DateOffset(months=2)].mean()
            return post - pre
        # annual cycle of noise and LFE monthly series over the same pre-Petrolia months
        def annual(col):
            q = g[col].loc[:REF_END].dropna()
            ty = (q.index + pd.Timedelta(days=15) - pd.Timestamp('2019-01-01')).days.values / 365.25
            X = np.array([np.ones_like(ty), ty, np.cos(2 * np.pi * ty), np.sin(2 * np.pi * ty)]).T
            bb = np.linalg.lstsq(X, q.values, rcond=None)[0]
            return np.hypot(bb[2], bb[3]), (np.degrees(np.arctan2(bb[3], bb[2])) % 360) / 360 * 365.25
        na, npk = annual('dvv_pct')
        la, lpk = annual('lfe_dvv_pct')
        comp.append(dict(sta=sta, band=band, combo=c, n_months=int(ok.sum()), r_monthly=r, r_deseason=rd,
                         noise_annual_amp_pct=na, noise_annual_peak_doy=npk,
                         lfe_annual_amp_pct=la, lfe_annual_peak_doy=lpk,
                         **{f'noise_mstep_{ev}': mstep('dvv_pct', te) for ev, te in EVENTS.items()},
                         **{f'lfe_mstep_{ev}': mstep('lfe_dvv_pct', te) for ev, te in EVENTS.items()}))
    C = pd.DataFrame(comp)
    C.to_csv(os.path.join(OUT, f'noise_vs_lfe{TAG}.csv'), index=False)
    def rec(df):
        return json.loads(df.to_json(orient='records', date_format='iso'))
    summ = dict(stations=summ, params=dict(window_utc=f'{H0:02d}-{H1:02d}', fs=FS, seg_s=SEG, bands=BAND_NAMES,
                                          coda_s=CODA, ref_end=REF_END, min_pair_cc=args.min_cc,
                                          step_window_days=30, mbb_block_days=5, nboot=args.nboot,
                                          events=EVENTS),
                steps=rec(S[S.combo == 'ALL']) if len(S) else [], longterm=rec(L[L.combo == 'ALL']),
                noise_vs_lfe=rec(C[C.combo == 'ALL']))
    json.dump(summ, open(os.path.join(OUT, f'summary{TAG}.json'), 'w'), indent=1, default=float)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        if len(S):
            print(S[S.combo == 'ALL'].drop(columns='same_season').round(3).to_string())
            print(S[S.combo == 'ALL'][['sta', 'band', 'event', 'same_season']].to_string())
        print(L[L.combo == 'ALL'].drop(columns='annual_means').round(3).T.to_string())
        print(L[L.combo == 'ALL'][['sta', 'band', 'annual_means']].to_string())
        print(C.round(3).to_string())


# ----------------------------------------------------------------------------- plots
def stage_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.read_csv(os.path.join(OUT, f'noise_dvv_10day{TAG}.csv'), parse_dates=['t'])
    M = pd.read_csv(os.path.join(OUT, f'noise_dvv_monthly{TAG}.csv'), parse_dates=['month'])
    S = pd.read_csv(os.path.join(OUT, f'noise_dvv_steps{TAG}.csv'))
    L = pd.read_csv(os.path.join(OUT, f'noise_dvv_longterm{TAG}.csv'))
    names = list(STATIONS)
    col = {'1-4Hz': '#1f77b4', '2-8Hz': '#ff7f0e'}
    fig, axs = plt.subplots(len(names), 1, figsize=(11, 2.6 * len(names)), sharex=True)
    for ax, n in zip(axs, names):
        for band in BAND_NAMES:
            g = T[(T.sta == n) & (T.band == band) & (T.combo == 'ALL')]
            ax.plot(g.t, g.dvv_pct, '.', ms=2.5, color=col[band], alpha=0.5)
            gm = M[(M.sta == n) & (M.band == band) & (M.combo == 'ALL')]
            ax.plot(gm.month + pd.Timedelta(days=15), gm.dvv_pct, '-', color=col[band], lw=1.2,
                    label=f'noise {band} (monthly)')
        gm = M[(M.sta == n) & (M.band == '1-4Hz') & (M.combo == 'ALL')]
        ax.plot(gm.month + pd.Timedelta(days=15), gm.lfe_dvv_pct, 'k-', lw=1.6, label='LFE coda (monthly)')
        for ev, te in EVENTS.items():
            ax.axvline(pd.Timestamp(te), color='r', ls='--', lw=0.8)
        ax.axvspan(pd.Timestamp(START), pd.Timestamp(REF_END), color='0.92', zorder=0)
        ax.axhline(0, color='0.5', lw=0.5)
        lo = np.nanpercentile(T[T.sta == n].dvv_pct, [1, 99])
        ax.set_ylim(min(lo[0], -0.35) - 0.05, max(lo[1], 0.2) + 0.05)
        ax.set_ylabel('dv/v (%)')
        ax.set_title(f'{n}: ambient-noise single-station dv/v (all pairs) vs LFE-coda dv/v', fontsize=9, loc='left')
    axs[0].legend(fontsize=7, ncol=3, loc='lower left')
    axs[-1].set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    axs[0].text(pd.Timestamp('2019-02-01'), axs[0].get_ylim()[1] * 0.85, 'reference', fontsize=7, color='0.4')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_timeseries.png'), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    ax = axs[0]
    evs = list(EVENTS)
    xt, xl = [], []
    for i, n in enumerate(names):
        for j, ev in enumerate(evs):
            x0 = i * 4 + j
            xt.append(x0)
            xl.append(f'{n}\n{ev}')
            for k, band in enumerate(BAND_NAMES):
                r = S[(S.sta == n) & (S.event == ev) & (S.band == band) & (S.combo == 'ALL')]
                if r.empty:
                    continue
                r = r.iloc[0]
                xx = x0 + (k - 0.5) * 0.3
                ax.bar(xx, 2 * r.placebo_sd_pct, bottom=-r.placebo_sd_pct, width=0.28, color=col[band], alpha=0.18)
                ax.errorbar(xx, r.step_pct, yerr=r.se_mbb_pct, fmt='o', color=col[band], ms=4,
                            label=band if (i == 0 and j == 0) else None)
                if isinstance(r.same_season, str) and r.same_season:   # same calendar date, other years
                    ss = [float(v.split(':')[1]) for v in r.same_season.split(';')]
                    ax.plot([xx] * len(ss), ss, 'x', color='0.45', ms=3.5,
                            label='same date, other years' if (i == 0 and j == 0 and k == 0) else None)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6, rotation=60)
    ax.set_ylabel('30-d post - 30-d pre dv/v (%)')
    ax.set_title('30-d steps, all pairs (error: block bootstrap; bars: +-1 sd placebo dates 2019-21)', fontsize=8)
    ax.legend(fontsize=7)
    ax = axs[1]
    for k, band in enumerate(BAND_NAMES):
        g = L[(L.band == band) & (L.combo == 'ALL')]
        ax.errorbar(g.lfe_D_2021_2025_pct, g.D_2021_2025_pct, yerr=g.D_se_pct, fmt='o', color=col[band], label=band)
        for _, r in g.iterrows():
            ax.annotate(r.sta, (r.lfe_D_2021_2025_pct, r.D_2021_2025_pct), fontsize=7, xytext=(3, 3),
                        textcoords='offset points')
    lim = [-0.4, 0.15]
    ax.plot(lim, lim, 'k:', lw=0.8, label='1:1')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.axvline(0, color='0.6', lw=0.5)
    ax.set_xlabel('LFE-coda D(2025-2021) (%)')
    ax.set_ylabel('noise D(2025-2021) (%)')
    ax.set_title('Long-term change', fontsize=9)
    ax.legend(fontsize=7)
    ax = axs[2]
    for k, band in enumerate(BAND_NAMES):
        g = L[(L.band == band) & (L.combo == 'ALL')]
        for i, (_, r) in enumerate(g.iterrows()):
            ang = 2 * np.pi * r.fit_peak_doy / 365.25
            ax.plot([0, r.fit_amp_pct * np.sin(ang)], [0, r.fit_amp_pct * np.cos(ang)], '-', color=col[band])
            ax.text(r.fit_amp_pct * np.sin(ang), r.fit_amp_pct * np.cos(ang), r.sta, fontsize=7)
    m = max(L.fit_amp_pct.max(), 0.05) * 1.2
    for mo, lab in [(0, 'Jan'), (90.25, 'Apr'), (181.5, 'Jul'), (273, 'Oct')]:
        a = 2 * np.pi * mo / 365.25
        ax.text(1.05 * m * np.sin(a), 1.05 * m * np.cos(a), lab, ha='center', va='center', fontsize=7, color='0.4')
    ax.add_patch(plt.Circle((0, 0), m * 0.5, fill=False, color='0.8'))
    ax.set_xlim(-1.2 * m, 1.2 * m)
    ax.set_ylim(-1.2 * m, 1.2 * m)
    ax.set_aspect('equal')
    ax.set_title(f'Annual cycle (all data):\namplitude & time of max dv/v\n(circle = {m * 0.5:.3f}%)', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_steps.png'), dpi=150)
    plt.close(fig)


def stage_robust(args):
    """Collect default + sensitivity runs (tags) into one table and add it to summary.json."""
    variants = {'default': '', 'no_notch': '_nonotch', 'coda_2-10s': '_coda2-10', 'strict_notch': '_strictnotch'}
    rows = []
    for v, t in variants.items():
        fl, fs = os.path.join(OUT, f'noise_dvv_longterm{t}.csv'), os.path.join(OUT, f'noise_dvv_steps{t}.csv')
        if not (os.path.exists(fl) and os.path.exists(fs)):
            continue
        L, S = pd.read_csv(fl), pd.read_csv(fs)
        sm = json.load(open(os.path.join(OUT, f'summary{t}.json')))['stations']
        for _, r in L.iterrows():
            d = dict(variant=v, sta=r.sta, band=r.band, combo=r.combo, D_2021_2025_pct=r.D_2021_2025_pct,
                     z_D=r.z_D, lfe_D_2021_2025_pct=r.lfe_D_2021_2025_pct,
                     notched_frac=(sm[r.sta].get('notched_frac') or {}).get(r.band, 0.0))
            for ev in EVENTS:
                q = S[(S.sta == r.sta) & (S.band == r.band) & (S.combo == r.combo) & (S.event == ev)]
                d[f'step_{ev}_pct'] = q.step_pct.iloc[0] if len(q) else np.nan
                d[f'z_{ev}'] = q.z_placebo.iloc[0] if len(q) else np.nan
            rows.append(d)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, 'noise_dvv_robustness.csv'), index=False)
    fs = os.path.join(OUT, 'summary.json')
    summ = json.load(open(fs))
    g = R[R.combo == 'ALL'].groupby(['sta', 'band'])
    summ['robustness_ALL'] = {f'{k[0]} {k[1]}': {c: dict(min=round(float(x[c].min()), 3), max=round(float(x[c].max()), 3),
                                                          by_variant=dict(zip(x.variant, x[c].round(3))))
                                                  for c in ['D_2021_2025_pct', 'step_M7_pct']} for k, x in g}
    json.dump(summ, open(fs, 'w'), indent=1, default=float)
    with pd.option_context('display.width', 250):
        print(R[R.combo == 'ALL'].pivot_table(index=['sta', 'band'], columns='variant',
                                             values=['D_2021_2025_pct', 'step_M7_pct']).round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['download', 'fillpb', 'correlate', 'dvv', 'plot', 'robust'])
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--stations', nargs='*')
    ap.add_argument('--min-cc', type=float, default=0.3)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--no-notch', action='store_true')
    ap.add_argument('--line-thr', type=float, help='sensitivity run: spectral-line threshold')
    ap.add_argument('--coda', type=float, nargs=2, help='sensitivity run: |lag| window (s)')
    ap.add_argument('--tag', default='', help='suffix for dvv output files (sensitivity runs)')
    a = ap.parse_args()
    TAG = a.tag
    if a.line_thr:
        LINE_THR = a.line_thr
    if a.coda:     # worker processes are forked, so they inherit these module globals
        CODA = tuple(a.coda)
        CODA_MASK = (np.abs(LAGS) >= CODA[0]) & (np.abs(LAGS) <= CODA[1])
        MASK_U = (np.abs(LAGS_U) >= CODA[0]) & (np.abs(LAGS_U) <= CODA[1])
    os.makedirs(RAW, exist_ok=True)
    {'download': stage_download, 'fillpb': stage_fillpb, 'correlate': stage_correlate,
     'dvv': lambda a: stage_dvv(a), 'plot': lambda a: stage_plot(a), 'robust': lambda a: stage_robust(a)}[a.stage](a)
