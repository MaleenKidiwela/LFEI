#!/usr/bin/env python3
"""Single-station CROSS-COMPONENT (EN, EZ, NZ = "SC") ambient-noise dv/v over each station's full
record, bands 0.1-1, 1-3, 3-5 Hz, for the PB boreholes and the NC/BK stations around the LFE
families.  Generalises the BK.JEWT method (scripts/sci_noise_dvv_jewt.py, imported as J, NOT
modified; it in turn imports scripts/sci_noise_dvv.py as M) to any station:

  per station-day (raw 100-Hz day read from PNWstore (PB 2008-2022) or NCEDC FDSN, processed in
  memory, never written to disk):
    day : M._decimate (8-Hz low-pass, 20 Hz) -> 00-24 UTC, 120-s segments, per band 1-3 / 3-5 Hz
          bandpass -> transient rejection (4x median RMS) -> one-bit -> pure-phase whitening ->
          segment-mean CCF, lags +-60 s                                  (= J 'day20' / M._correlate_day)
    low : J._to_low (2-Hz low-pass, 5 Hz) -> 600-s segments, 0.1-1 Hz, lags +-150 s   (= J 'low')
  Only the three cross-component pairs are computed/stored (autos dropped), float16 normalised by
  a float32 per-trace scale, full stored lag range of the method (needed for its line detector at
  0.5-1.0 x maxlag), ~40 kB per station-day.  Days without data -> <day>.npz.none.

  dv/v (identical algorithm to J._dvv_prod, SC pairs only): persistent spectral lines notched
  (J.detect_lines_g rule: in >=2 years; 1 if the record has <2 years with >=60 days), reference =
  mean of all days before 2024-11-30 (whole record if <60 such days), pair quality = median CC of
  10-d stacks in the reference period >= 0.3, stretching on M.EPS with 4x upsampling, Weaver
  errors, inverse-variance SC combination; codas 0.1-1 Hz 10-60 s, 1-3 Hz 3-30 s, 3-5 Hz 1-8 s
  (J.CODA_NEW).  5-d / 10-d stacks aligned on the M7 day, monthly stacks (>=8 d).  Steps (30 d
  pre/post) at M7 2024-12-05, Petrolia 2021-12-20, Ferndale 2022-12-20 with a 5-d moving-block
  bootstrap (200) and placebo dates every 5 d (excluding M7-31..+120 d and Petrolia/Ferndale
  -31..+60 d).  Long-term fit on 10-d values: c + trend + annual + free offsets at StationXML
  epoch boundaries + M7 step [+ exp recovery, tau by BIC]  (J.fit_series with per-station breaks),
  plus a pre-M7 trend, 2025-minus-2021 change and its pre-2021 4-yr placebo spread.

Stages:  download | verify | dvv | plot        (download is resumable; run under nice, <=8 workers)
Outputs: data/sci/noise_dvv_sc_full/{NET.STA}/days/*.npz,  sc_10day.csv, sc_5day.csv,
         sc_monthly.csv, sc_steps.csv, sc_longterm_fit.csv, sc_meta.json;
         figures/sci_noise_dvv_sc_full_{STA}.png, figures/sci_noise_dvv_sc_full_all.png
"""
import argparse
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, '/home/seismic/LFE/scripts')
import sci_noise_dvv as M           # noqa: E402  (method module; NOT modified)
import sci_noise_dvv_jewt as J      # noqa: E402  (JEWT generalisation; NOT modified)

ROOT = M.ROOT
OUT = os.path.join(ROOT, 'data/sci/noise_dvv_sc_full')
FIG = M.FIG
MIN_FREE_GB = 20.0
LAST_DAY = '2026-11-23'
# (net, sta, first day, last day or None=open) in processing priority order
STATIONS = [('PB', 'B049', '2008-11-01', None), ('PB', 'B934', '2008-10-01', None),
            ('PB', 'B933', '2008-09-01', None), ('PB', 'B045', '2008-09-01', None),
            ('PB', 'B047', '2008-09-01', None), ('PB', 'B046', '2008-08-01', None),
            ('PB', 'B932', '2008-10-01', None), ('PB', 'B935', '2008-10-01', None),
            ('NC', 'KMR', '2016-04-01', '2025-09-17'), ('NC', 'KMPB', '2016-04-01', '2026-07-01'),
            ('NC', 'KCPB', '2016-04-01', '2022-11-15'), ('NC', 'KCPC', '2023-06-01', None),
            ('BK', 'BRIC', '2019-02-01', None), ('BK', 'RVIT', '2019-10-01', None),
            ('BK', 'HALS', '2020-06-01', None), ('BK', 'RBOW', '2020-06-01', None),
            ('BK', 'HAYF', '2019-11-01', None), ('BK', 'BJES', '2019-08-01', None),
            ('BK', 'SPRL', '2023-09-01', None), ('BK', 'USAL', '2024-03-01', None),
            ('BK', 'BLAS', '2022-06-01', None), ('BK', 'GHOP', '2022-05-01', None),
            ('BK', 'HUNT', '2022-06-01', None), ('BK', 'PRDS', '2022-09-01', None),
            ('BK', 'SHWD', '2023-02-01', None), ('BK', 'ETSL', '2024-07-01', '2025-08-31')]
SC_PAIRS = [('E', 'N'), ('E', 'Z'), ('N', 'Z')]
PN = ['EN', 'EZ', 'NZ']
PRODS = {'low': J.PRODS['low'], 'day': J.PRODS['day']}       # lag axes, mutes, codas, line scales
MAIN = [('low', '0.1-1Hz'), ('day', '1-3Hz'), ('day', '3-5Hz')]
M7 = pd.Timestamp('2024-12-05')
EVENTS = {'M7': '2024-12-05', 'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20'}
EXCL = {'M7': (-31, 120), 'Petrolia': (-31, 60), 'Ferndale': (-31, 60)}
REF_END = pd.Timestamp('2024-11-30')


def sdir(net, sta):
    return os.path.join(OUT, f'{net}.{sta}')


def _free_gb():
    s = os.statvfs('/home')
    return s.f_bavail * s.f_frsize / 1e9


# ----------------------------------------------------------------------------- correlation
def _segments(st, day, cmap, fs, seg):
    """J._segments for 00-24 UTC with an explicit component map."""
    from obspy import UTCDateTime
    ns = int(seg * fs)
    t0 = UTCDateTime(day)
    nseg = int(86400 / seg)
    arr = np.full((3, nseg, ns), np.nan)
    for ic, c in enumerate('ZNE'):
        for tr in st.select(channel=cmap[c]):
            i0 = int(round((tr.stats.starttime - t0) * fs))
            d = tr.data.astype('float64')
            for k in range(nseg):
                a = k * ns - i0
                if a >= 0 and a + ns <= len(d):
                    arr[ic, k] = d[a:a + ns]
    return arr


def auto_idx(fs, maxlag):
    """lag indices kept for the autocorrelations: |lag| >= 0.5 maxlag, i.e. exactly the support
    of the method's line-detector windows (0.5-0.75 and 0.75-1.0 x maxlag)."""
    lags = np.arange(-int(maxlag * fs), int(maxlag * fs) + 1) / fs
    return np.where(np.abs(lags) >= 0.5 * maxlag - 1e-9)[0]


def corr_sc(arr, bands, fs, maxlag, smooth_hz):
    """J._correlate_low / M._correlate_day (identical steps: demean, 5% taper, bandpass, transient
    rejection, one-bit, pure-phase whitening for cross pairs, smooth_hz-smoothed amplitude for
    autos). Cross pairs EN EZ NZ on the full lag axis; autos ZZ NN EE only at |lag| >= maxlag/2
    (used only to find persistent spectral lines, as in the method)."""
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import uniform_filter1d
    ns = arr.shape[-1]
    nfft = 2 * ns
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    nl = int(maxlag * fs)
    ai = auto_idx(fs, maxlag)
    out = np.full((len(bands), 3, 2 * nl + 1), np.nan, dtype='float32')
    nused = np.zeros((len(bands), 3), dtype='int16')
    aut = np.full((len(bands), 3, len(ai)), np.nan, dtype='float32')
    anused = np.zeros((len(bands), 3), dtype='int16')
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
        W = J._cos_taper_band(freqs, f1, f2, fs)
        A = np.abs(Y) + 1e-10
        k = int(round(smooth_hz / (freqs[1] - freqs[0])))
        As = uniform_filter1d(A, k, axis=2, mode='nearest')
        Ya = W * Y / As
        Y = W * Y / A
        for ic in range(3):
            m = gb[ic]
            if m.sum() < 10:
                continue
            C = np.fft.irfft(Ya[ic, m] * np.conj(Ya[ic, m]), nfft, axis=1)
            C = np.concatenate([C[:, -nl:], C[:, :nl + 1]], axis=1)
            aut[ib, ic] = C.mean(axis=0)[ai]
            anused[ib, ic] = m.sum()
        del Ya
        for ip, (ca, cb) in enumerate(SC_PAIRS):
            ia, ibb = 'ZNE'.index(ca), 'ZNE'.index(cb)
            m = gb[ia] & gb[ibb]
            if m.sum() < 10:
                continue
            C = np.fft.irfft(Y[ia, m] * np.conj(Y[ibb, m]), nfft, axis=1)
            C = np.concatenate([C[:, -nl:], C[:, :nl + 1]], axis=1)
            out[ib, ip] = C.mean(axis=0)
            nused[ib, ip] = m.sum()
    return out, nused, aut, anused


def _q16(c):
    """float16 of c / max|c| per trace + float32 scale (NaN traces kept NaN)."""
    s = np.nanmax(np.abs(c), axis=-1)
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0).astype('float32')
    return (c / s[..., None]).astype('float16'), s


def _cmap(st):
    """Z/N/E -> channel codes of the dominant location and instrument (N=1, E=2 for PB)."""
    tot = {}
    for tr in st:
        k = (tr.stats.location, tr.stats.channel[:2])
        tot[k] = tot.get(k, 0) + tr.stats.npts
    loc, pre = max(tot, key=tot.get)
    chans = {tr.stats.channel for tr in st if tr.stats.location == loc and tr.stats.channel[:2] == pre}
    cm = {'Z': pre + 'Z', 'N': pre + ('N' if pre + 'N' in chans else '1'), 'E': pre + ('E' if pre + 'E' in chans else '2')}
    return loc, cm


def process_day(st, day, fn):
    st20 = M._decimate(st)
    if len(st20) == 0:
        open(fn + '.none', 'w').close()
        return 'nodata'
    loc, cm = _cmap(st20)
    st20 = st20.select(location=loc)
    cd, nd, ad, and_ = corr_sc(_segments(st20, day, cm, M.FS, M.SEG), J.DAY20_BANDS, M.FS, M.MAXLAG, 1.0)
    stl = J._to_low(st20)
    cl, nl_, al, anl = corr_sc(_segments(stl, day, cm, J.LOW_FS, J.LOW_SEG), J.LOW_BANDS, J.LOW_FS, J.LOW_MAXLAG,
                               J.LOW_SMOOTH)
    qd, sd = _q16(cd)
    ql, sl = _q16(cl)
    qad, sad = _q16(ad)
    qal, sal = _q16(al)
    hours = sum(tr.stats.npts / tr.stats.sampling_rate for tr in st20.select(channel=cm['Z'])) / 3600
    np.savez_compressed(fn, day=qd, day_s=sd, day_n=nd, low=ql, low_s=sl, low_n=nl_,
                        dayA=qad, dayA_s=sad, dayA_n=and_, lowA=qal, lowA_s=sal, lowA_n=anl,
                        chans=f'{loc}.' + ','.join(cm[c] for c in 'ZNE'), hours=np.float32(hours))
    return f'ok day_seg{int(nd[0].max())} low_seg{int(nl_[0].max())} h{hours:.1f}'


# ----------------------------------------------------------------------------- data access
def _read_pnwstore(sta, day):
    import sqlite3
    from obspy import read, Stream
    from download_windows import pnw_path
    con = sqlite3.connect(f'file:/data/wsd01/PNWstore_sqlite/{day.year}.sqlite?mode=ro&immutable=1', uri=True)
    a, b = day.strftime('%Y-%m-%dT00:00:00'), day.strftime('%Y-%m-%dT23:59:59.99')
    files = {r[0] for r in con.execute("select filename from tsindex where network='PB' and station=? and channel like 'EH_' "
                                       "and starttime<=? and endtime>=?", (sta, b, a))}
    con.close()
    st = Stream()
    for f in sorted(files):
        try:
            st += read(pnw_path(f), sourcename=f'PB.{sta}.*.EH?').select(channel='EH?')
        except Exception:
            pass
    from obspy import UTCDateTime
    return st.trim(UTCDateTime(day) - 60, UTCDateTime(day) + 86460)


def _fetch(net, sta, day):
    """-> Stream (may be empty) or raises RuntimeError on repeated failures."""
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    pst = None
    if net == 'PB' and day.year <= 2022:
        pst = _read_pnwstore(sta, day)
        if len({tr.stats.channel for tr in pst}) >= 3 and sum(tr.stats.npts for tr in pst) > 3 * 12 * 3600 * 100:
            return pst, 'pnwstore'
    a = UTCDateTime(day)
    client = Client('NCEDC', timeout=300)
    for cp in (['EH?'] if net == 'PB' else ['HH?', 'BH?']):
        for attempt in range(5):
            try:
                st = client.get_waveforms(net, sta, '*', cp, a - 60, a + 86400 + 60)
                if len(st):
                    return st, 'ncedc'
                break
            except Exception as e:
                if 'No data' in str(e) or '204' in str(e):
                    break
                time.sleep(5 * 2 ** attempt)
        else:
            raise RuntimeError('fdsn failed')
    if pst is not None and len(pst):
        return pst, 'pnwstore-partial'
    return None, 'none'


def _day_fn(job):
    net, sta, dstr = job[:3]
    return os.path.join(job[3] if len(job) > 3 else os.path.join(sdir(net, sta), 'days'), f'{dstr}.npz')


def _fetch_part(job):
    t0 = time.time()
    if _free_gb() < MIN_FREE_GB:
        return None, 'lowdisk', 0.0
    try:
        st, src = _fetch(job[0], job[1], pd.Timestamp(job[2]))
    except Exception as e:
        return None, f'FAILED {str(e)[:50]}', time.time() - t0
    return st, src, time.time() - t0


def _proc_part(job, fetched):
    net, sta, dstr = job[:3]
    fn = _day_fn(job)
    st, src, t_dl = fetched
    if src == 'lowdisk' or src.startswith('FAILED'):
        return net, sta, dstr, src, t_dl, 0.0, 0
    if st is None:
        open(fn + '.none', 'w').close()
        return net, sta, dstr, 'nodata', t_dl, 0.0, 0
    t1 = time.time()
    try:
        status = process_day(st, pd.Timestamp(dstr), fn) + ' ' + src
    except Exception as e:
        return net, sta, dstr, f'PROCFAIL {str(e)[:60]}', t_dl, time.time() - t1, 0
    sz = os.path.getsize(fn) if os.path.exists(fn) else 0
    return net, sta, dstr, status, t_dl, time.time() - t1, sz


def _do_day(job):
    if os.path.exists(_day_fn(job)) or os.path.exists(_day_fn(job) + '.none'):
        return job[0], job[1], job[2], 'have', 0.0, 0.0, 0
    return _proc_part(job, _fetch_part(job))


def _do_chunk(jobs):
    """several station-days in one worker process; the next day's waveforms are fetched in a
    thread while the current day is processed (I/O overlaps CPU; still one process)."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [j for j in jobs if not (os.path.exists(_day_fn(j)) or os.path.exists(_day_fn(j) + '.none'))]
    out = []
    with ThreadPoolExecutor(max_workers=1) as tp:
        fut = tp.submit(_fetch_part, todo[0]) if todo else None
        for i, j in enumerate(todo):
            fetched = fut.result()
            fut = tp.submit(_fetch_part, todo[i + 1]) if i + 1 < len(todo) else None
            out.append(_proc_part(j, fetched))
            del fetched
    return out


def _station_days(net, sta, d0, d1, start=None, end=None):
    d1 = d1 or LAST_DAY
    a = max(pd.Timestamp(d0), pd.Timestamp(start)) if start else pd.Timestamp(d0)
    b = min(pd.Timestamp(d1), pd.Timestamp(end)) if end else pd.Timestamp(d1)
    return [(net, sta, d.strftime('%Y-%m-%d')) for d in pd.date_range(a, b)]


def stage_download(args):
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    jobs = []
    sel = [s for s in STATIONS if not args.stations or s[1] in args.stations]
    for net, sta, d0, d1 in sel:
        os.makedirs(os.path.join(sdir(net, sta), 'days'), exist_ok=True)
        js = _station_days(net, sta, d0, d1, args.start, args.end)
        if args.days:
            js = [j for j in js if j[2] in set(args.days)]
        jobs += js
    jobs = [j for j in jobs if not (os.path.exists(os.path.join(sdir(j[0], j[1]), 'days', j[2] + '.npz'))
                                    or os.path.exists(os.path.join(sdir(j[0], j[1]), 'days', j[2] + '.npz.none')))]
    print(f'{len(jobs)} station-days to do, {args.workers} workers, free {_free_gb():.1f} GB', flush=True)
    logs = {}
    t_start = time.time()
    n_done = n_ok = 0
    bytes_ok = 0
    tdl = tpr = 0.0
    stop = False
    chunks = [jobs[i:i + args.chunk] for i in range(0, len(jobs), args.chunk)]
    it = iter(chunks)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        pend = set()
        while True:
            while not stop and len(pend) < 2 * args.workers:
                j = next(it, None)
                if j is None:
                    break
                pend.add(ex.submit(_do_chunk, j))
            if not pend:
                break
            done, pend = wait(pend, return_when=FIRST_COMPLETED)
            for f in [r for f in done for r in f.result()]:
                net, sta, d, status, t_dl, t_pr, sz = f
                if status == 'have':
                    continue
                key = f'{net}.{sta}'
                if key not in logs:
                    logs[key] = open(os.path.join(sdir(net, sta), 'download_log.csv'), 'a')
                logs[key].write(f'{d},{status},{t_dl:.1f},{t_pr:.1f},{sz}\n')
                logs[key].flush()
                n_done += 1
                tdl += t_dl
                tpr += t_pr
                if status.startswith('ok'):
                    n_ok += 1
                    bytes_ok += sz
                if status == 'lowdisk' and not stop:
                    print(f'HARD STOP: free space {_free_gb():.1f} GB < {MIN_FREE_GB} GB', flush=True)
                    stop = True
                if n_done % args.report == 0 or args.verbose:
                    el = time.time() - t_start
                    print(f'{key} {d} {status} dl {t_dl:.1f}s proc {t_pr:.1f}s | done {n_done} ok {n_ok} '
                          f'wall {el / n_done:.2f} s/station-day ({args.workers} workers), mean dl {tdl / n_done:.1f}s '
                          f'proc {tpr / n_done:.1f}s per day/worker, {bytes_ok / max(n_ok, 1) / 1e6:.3f} MB/day, '
                          f'free {_free_gb():.1f} GB', flush=True)
                if status.startswith('FAILED') or status.startswith('PROCFAIL'):
                    print(key, d, status, flush=True)
            if stop:
                for f in pend:
                    f.cancel()
    el = time.time() - t_start
    print(f'FINISHED: {n_done} station-days in {el / 3600:.2f} h ({el / max(n_done, 1):.2f} s each), {n_ok} ok, '
          f'{bytes_ok / max(n_ok, 1) / 1e6:.3f} MB/day, free {_free_gb():.1f} GB', flush=True)


# ----------------------------------------------------------------------------- loading
def load_sc(key, days_dir, d0=None, d1=None):
    """-> days (DatetimeIndex), ccf (D, B, 3, L) float32 (n < 20 segments -> NaN, as J.load_prod),
    aut (D, B, 3, La) autocorrelations ZZ NN EE at |lag| >= maxlag/2 (for line detection only)."""
    P = PRODS[key]
    files = sorted(f for f in os.listdir(days_dir) if f.endswith('.npz'))
    if not files:
        return None, None, None
    d0 = pd.Timestamp(d0 or files[0][:10])
    d1 = pd.Timestamp(d1 or files[-1][:10])
    days = pd.date_range(d0, d1)
    ccf = np.full((len(days), len(P.bands), 3, len(P.lags)), np.nan, dtype='float32')
    aut = np.full((len(days), len(P.bands), 3, len(auto_idx(P.fs, P.maxlag))), np.nan, dtype='float32')
    have = set(files)
    for i, d in enumerate(days):
        f = f'{d:%Y-%m-%d}.npz'
        if f in have:
            z = np.load(os.path.join(days_dir, f))
            c = z[key].astype('float32') * z[key + '_s'][..., None]
            c[z[key + '_n'] < 20] = np.nan
            ccf[i] = c
            if key + 'A' in z.files:
                c = z[key + 'A'].astype('float32') * z[key + 'A_s'][..., None]
                c[z[key + 'A_n'] < 20] = np.nan
                aut[i] = c
    return days, ccf, aut


def load_jewt_sc(key, d0=None, d1=None):
    """existing JEWT day files (6 pairs, float32) -> SC-only array in the same layout."""
    P = PRODS[key]
    days, ccf = J.load_prod(key, d0=d0, d1=d1)
    return days, ccf[:, :, 3:6].astype('float32')


# ----------------------------------------------------------------------------- dv/v (SC only)
class Stretcher:
    """J.stretch_g with the stretched-reference matrix cached per (product, band, pair, coda)
    (the reference is fixed within a station run; identical numerics)."""
    def __init__(self):
        self.cache = {}

    def stretch(self, P, coda, ref, X, ck):
        from scipy.signal import resample
        X = np.atleast_2d(X)
        out_e = np.full(len(X), np.nan)
        out_c = np.full(len(X), np.nan)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(ref).all()
        if not ok.any():
            return out_e, out_c
        mu = (np.abs(P.lags_u) >= coda[0]) & (np.abs(P.lags_u) <= coda[1])
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

    def measure(self, P, ref, S, codas=None):
        G, B, NP, L = S.shape
        dv = np.full((G, B, NP), np.nan)
        cc = np.full((G, B, NP), np.nan)
        er = np.full((G, B, NP), np.nan)
        for b in range(B):
            coda = tuple(codas[b]) if codas else tuple(P.coda[P.names[b]])
            for p in range(NP):
                dv[:, b, p], cc[:, b, p] = self.stretch(P, coda, ref[b, p], S[:, b, p], (P.key, b, p, coda))
            er[:, b] = J.weaver_err_g(cc[:, b], P.bands[b], coda)
        return dv, cc, er


def detect_lines_sc(P, days, ccf, min_periods, aut=None):
    """J.detect_lines_g (identical rule) on the SC pairs plus, if given, the autocorrelations
    stored at |lag| >= maxlag/2 (zero elsewhere; the detector windows are zero there anyway)."""
    ai = auto_idx(P.fs, P.maxlag)
    from scipy.ndimage import median_filter, binary_dilation, gaussian_filter1d
    a = np.abs(P.lags)
    taps = [np.where((a >= lo) & (a <= hi), np.sin(np.pi * (a - lo) / (hi - lo)) ** 2, 0) for lo, hi in P.linew]
    k = int(round(P.line_med_hz / (P.freq[1] - P.freq[0])))
    if min_periods == 1:
        groups, need = [np.ones(len(days), bool)], 1
    else:
        groups, need = [days.year == yr for yr in sorted(set(days.year))], 2
    count = np.zeros(len(P.freq), int)
    for m in groups:
        chk = aut if aut is not None else ccf          # J checks pair 0 (= ZZ)
        if np.isfinite(chk[m, 0, 0, 0]).sum() < (60 if min_periods > 1 else 20):
            continue
        with np.errstate(invalid='ignore'):
            S = np.nanmean(ccf[m], axis=0, dtype='float64')
            if aut is not None:
                Sa = np.zeros(S.shape[:2] + (len(P.lags),))
                Sa[..., ai] = np.nanmean(aut[m], axis=0, dtype='float64')
                S = np.concatenate([Sa, S], axis=1)
        line = np.zeros(len(P.freq), bool)
        for b, (f1, f2) in enumerate(P.bands):
            inb = (P.freq >= f1) & (P.freq <= f2)
            for p in range(S.shape[1]):
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


def stack_groups(days, ccf, key, min_days):
    """M.stack_groups with float64 accumulation."""
    labs = pd.unique(key[pd.notna(key)])
    S = np.full((len(labs),) + ccf.shape[1:], np.nan)
    N = np.zeros((len(labs),) + ccf.shape[1:3], dtype=int)
    for g, lab in enumerate(labs):
        sel = ccf[key == lab]
        n = np.isfinite(sel[..., 0]).sum(axis=0)
        with np.errstate(invalid='ignore'):
            m = np.nanmean(sel, axis=0, dtype='float64')
        m[n < min_days] = np.nan
        S[g], N[g] = m, n
    return labs, S, N


def good_pairs_sc(SR, P, days, ccf, ref, mask, block=10, min_days=5, min_cc=0.3):
    k = ((days - days[0]).days // block).values
    labs, S, N = stack_groups(days[mask], ccf[mask], k[mask], min_days)
    dv, cc, er = SR.measure(P, ref, S)
    with np.errstate(invalid='ignore'):
        q = np.nanmedian(cc, axis=0)
    return q, q >= min_cc


def event_step_sc(SR, P, days, ccf, ref, te, good, half=30, skip=1, nboot=0, rng=None):
    """J.event_step_g, SC combination only -> (1+nboot, B) in %."""
    te = pd.Timestamp(te)
    pre = (days >= te - pd.Timedelta(days=half)) & (days < te)
    post = (days > te + pd.Timedelta(days=skip - 1)) & (days <= te + pd.Timedelta(days=half + skip - 1))
    out = []
    for msk in (pre, post):
        w = ccf[msk]
        if np.isfinite(w[:, 0, :, 0]).any(axis=1).sum() < half // 3:
            return None
        with np.errstate(invalid='ignore'):
            base = np.nanmean(w, axis=0, dtype='float64')[None]
        if nboot:
            base = np.concatenate([base, M.mbb_window(w.astype('float64'), nboot, 5, rng)], axis=0)
        dv, cc, er = SR.measure(P, ref, base)
        er = np.where(good[None], er, np.nan)
        out.append(M.combine(dv, er)[0])
    return (out[1] - out[0]) * 100


def series_sc(SR, P, days, ccf, ref, good, blk, t0, min_days):
    k = ((days - t0).days // blk).values
    labs, S, N = stack_groups(days, ccf, k, min_days)
    dv, cc, er = SR.measure(P, ref, S)
    erg = np.where(good[None], er, np.nan)
    tt = t0 + pd.to_timedelta(labs * blk + (blk - 1) / 2, unit='D')
    m, e = M.combine(dv, erg)
    out = []
    for b in range(len(P.bands)):
        out.append(pd.DataFrame(dict(t=tt, k=labs, dvv_pct=m[:, b] * 100, err_pct=e[:, b] * 100, cc=np.nan,
                                     ndays=N[:, b].max(axis=1), band=P.names[b], combo='SC')))
        for p in range(3):
            out.append(pd.DataFrame(dict(t=tt, k=labs, dvv_pct=dv[:, b, p] * 100, err_pct=er[:, b, p] * 100,
                                         cc=cc[:, b, p], ndays=N[:, b, p], band=P.names[b], combo=PN[p])))
    return pd.concat(out, ignore_index=True)


def _design(tt, breaks, m7terms, tau=None):
    """J._design with per-station breaks; M7 terms only if the record spans the M7."""
    ty = (tt - M7).days.values / 365.25
    cols = [np.ones_like(ty), ty, np.cos(2 * np.pi * tt.day_of_year.values / 365.25),
            np.sin(2 * np.pi * tt.day_of_year.values / 365.25)]
    names = ['c', 'trend_per_yr', 'acos', 'asin']
    for k, d in breaks.items():
        cols.append((tt >= pd.Timestamp(d)).astype(float))
        names.append('off_' + k)
    if m7terms:
        dt = (tt - M7).days.values
        cols.append((dt > 0).astype(float))
        names.append('step_M7_perm')
        if tau:
            cols.append(np.where(dt > 0, np.exp(-np.clip(dt, 0, None) / tau), 0.0))
            names.append('rec_M7_amp')
    return np.array(cols).T, names


def fit_series(tt, y, rng, breaks, nb=500):
    """J.fit_series (BIC tau grid, 9-point moving-block residual bootstrap) with breaks."""
    ok = np.isfinite(y)
    tt, y = tt[ok], y[ok]
    if len(y) < 30:
        return dict(n=len(y))
    m7terms = (tt < M7 - pd.Timedelta(days=30)).sum() >= 6 and (tt > M7 + pd.Timedelta(days=30)).sum() >= 6
    breaks = {k: v for k, v in breaks.items() if (tt < pd.Timestamp(v)).sum() >= 6 and (tt >= pd.Timestamp(v)).sum() >= 6}
    best = None
    for tau in ([None, 10, 20, 40, 80, 160, 320] if m7terms else [None]):
        X, names = _design(tt, breaks, m7terms, tau=tau)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rss = ((y - X @ beta) ** 2).sum()
        bic = len(y) * np.log(rss / len(y)) + X.shape[1] * np.log(len(y))
        if best is None or bic < best[0]:
            best = (bic, tau, X, names, beta)
    bic, tau, X, names, beta = best
    resid = y - X @ beta
    bb = []
    for _ in range(nb):
        st0 = rng.integers(0, len(resid) - 9 + 1, size=int(np.ceil(len(resid) / 9)))
        rr = np.concatenate([resid[s_:s_ + 9] for s_ in st0])[:len(resid)]
        bb.append(np.linalg.lstsq(X, X @ beta + rr, rcond=None)[0])
    bb = np.array(bb)
    out = dict(n=len(y), first=str(tt.min().date()), last=str(tt.max().date()), tau_days=tau,
               rms_resid=float(np.std(resid)), breaks=';'.join(f'{k}' for k in breaks))
    for n_, v, e in zip(names, beta, bb.std(0)):
        out[n_], out[n_ + '_se'] = v, e
    out['annual_amp_pct'] = np.hypot(beta[2], beta[3])
    out['annual_amp_se'] = np.hypot(bb[:, 2], bb[:, 3]).std()
    out['annual_peak_doy'] = (np.degrees(np.arctan2(beta[3], beta[2])) % 360) / 360 * 365.25
    if tau:
        i1, i2 = names.index('step_M7_perm'), names.index('rec_M7_amp')
        out['coseis_M7'] = beta[i1] + beta[i2]
        out['coseis_M7_se'] = (bb[:, i1] + bb[:, i2]).std()
    return out


def _pre_m7_trend(tt, y, rng, breaks):
    m = np.isfinite(y) & (tt < M7)
    if m.sum() < 30:
        return {}
    o = fit_series(tt[m], y[m], rng, breaks, nb=300)
    return dict(trend_preM7_per_yr=o.get('trend_per_yr'), trend_preM7_per_yr_se=o.get('trend_per_yr_se'),
                n_preM7=o.get('n'))


def _decline_stats(s10):
    """mean(2025) - mean(2021) of the SC 10-d values; null = pre-2021 4-yr differences."""
    s = s10.set_index('t').dvv_pct.dropna()
    yr = s.groupby(s.index.year).agg(['mean', 'size'])
    yr = yr[yr['size'] >= 12]['mean']
    out = {}
    if 2021 in yr and 2025 in yr:
        out['d2025_minus_2021_pct'] = yr[2025] - yr[2021]
    nul = [yr[y + 4] - yr[y] for y in yr.index if y + 4 <= 2020 and y + 4 in yr]
    if 2021 in yr:
        nul += [yr[2021] - yr[2017]] if 2017 in yr else []
    if len(nul) >= 3:
        out['null_4yr_sd_pct'] = float(np.std(nul))
        out['null_4yr_n'] = len(nul)
        out['null_4yr_vals'] = ';'.join(f'{v:+.3f}' for v in nul)
    return out


def epoch_breaks(net, sta):
    """StationXML epoch start dates (HH/BH/EH Z) -> {name: date}; from the cached NCEDC inventory."""
    from obspy import read_inventory
    fn = os.path.join(OUT, 'stations_ncedc.xml')
    inv = read_inventory(fn).select(network=net, station=sta)
    ds = set()
    for n in inv:
        for s in n:
            for c in s:
                if c.code[-1] == 'Z' and c.code[:2] in ('HH', 'EH'):
                    ds.add(c.start_date.date)
    return {f'xml_epoch_{d}': str(d) for d in sorted(ds)}


def run_dvv(name, key, days, ccf, nboot, seed, breaks, ref_override=None, keep_override=None, steps=True, aut=None):
    """the J._dvv_prod pipeline for one station/product, SC pairs only."""
    rng = np.random.default_rng(seed)
    P = PRODS[key]
    SR = Stretcher()
    valid = np.isfinite(ccf[:, 0, :, 0]).any(axis=1)
    nyr = sum(int((valid & (days.year == y)).sum() >= 60) for y in set(days.year))
    if keep_override is not None:
        keep, lines = keep_override, 'override'
    else:
        keep, lines = detect_lines_sc(P, days, ccf, 2 if nyr >= 2 else 1, aut)
    del aut
    nf = {P.names[b]: round(float(1 - keep[(P.freq > f1) & (P.freq < f2)].mean()), 3) for b, (f1, f2) in enumerate(P.bands)}
    X = J.notch_g(P, ccf, keep)
    del ccf
    rmask = days < REF_END
    if (valid & rmask).sum() < 60:
        rmask = np.ones(len(days), bool)
    if ref_override is not None:
        ref = ref_override
    else:
        with np.errstate(invalid='ignore'):
            ref = np.nanmean(X[rmask], axis=0, dtype='float64')
    q, good = good_pairs_sc(SR, P, days, X, ref, rmask)
    meta = dict(sta=name, prod=key, lines_hz=lines, notched_frac=nf, n_days=int(valid.sum()),
                first=str(days[valid][0].date()), last=str(days[valid][-1].date()), n_ref_days=int((valid & rmask).sum()),
                ref_period=f'{days[rmask][0].date()}..{days[rmask][-1].date()}',
                pair_q={P.names[b]: dict(zip(PN, np.round(q[b], 3).tolist())) for b in range(len(P.bands))},
                pairs_used={P.names[b]: list(np.array(PN)[good[b]]) for b in range(len(P.bands))})
    print(name, key, 'lines', lines, 'notched', nf, 'pair q', np.round(q, 2).tolist(), flush=True)
    s10 = series_sc(SR, P, days, X, ref, good, 10, M7, 5).assign(sta=name, prod=key)
    s5 = series_sc(SR, P, days, X, ref, good, 5, M7, 2).assign(sta=name, prod=key)
    km = days.to_period('M').astype(str).values
    labsM, SM, NM = stack_groups(days, X, km, 8)
    dvM, ccM, erM = SR.measure(P, ref, SM)
    mM, eM = M.combine(dvM, np.where(good[None], erM, np.nan))
    rowsM = [dict(sta=name, prod=key, band=P.names[b], combo='SC', month=labsM[g], dvv_pct=mM[g, b] * 100,
                  err_pct=eM[g, b] * 100, ndays=int(NM[g, b].max())) for b in range(len(P.bands)) for g in range(len(labsM))]
    res = dict(meta=meta, s10=s10, s5=s5, monthly=pd.DataFrame(rowsM), steps=pd.DataFrame(), fits=pd.DataFrame())
    if not steps:
        return res
    # steps + placebo
    first, last = days[valid][0], days[valid][-1]
    pl_dates = [d for d in pd.date_range(first + pd.Timedelta(days=31), last - pd.Timedelta(days=31), freq='5D')
                if not any(EXCL[e][0] <= (d - pd.Timestamp(t)).days <= EXCL[e][1] for e, t in EVENTS.items())]
    plac = []
    for td in pl_dates:
        r = event_step_sc(SR, P, days, X, ref, td, good)
        plac.append(r[0] if r is not None else np.full(len(P.bands), np.nan))
    plac = np.array(plac).reshape(-1, len(P.bands))
    srows = []
    for ev, te in EVENTS.items():
        r = event_step_sc(SR, P, days, X, ref, te, good, nboot=nboot, rng=rng)
        if r is None:
            continue
        for b in range(len(P.bands)):
            pv = plac[:, b][np.isfinite(plac[:, b])]
            sdp = np.std(pv) if len(pv) > 5 else np.nan
            st_ = r[0, b]
            srows.append(dict(sta=name, prod=key, band=P.names[b], combo='SC', event=ev, date=te, half_days=30,
                              step_pct=st_, se_mbb_pct=np.nanstd(r[1:, b]), placebo_sd_pct=sdp, n_placebo=len(pv),
                              z_placebo=st_ / sdp, z_mbb=st_ / np.nanstd(r[1:, b]),
                              frac_placebo_le=np.mean(pv <= st_) if len(pv) else np.nan,
                              placebo_robust_sd_pct=1.4826 * np.median(np.abs(pv - np.median(pv))) if len(pv) else np.nan,
                              pairs_used=' '.join(meta['pairs_used'][P.names[b]])))
    fits = []
    for b, bn in enumerate(P.names):
        g = s10[(s10.band == bn) & (s10.combo == 'SC')]
        tt = pd.DatetimeIndex(g.t)
        o = fit_series(tt, g.dvv_pct.values, rng, breaks)
        o.update(_pre_m7_trend(tt, g.dvv_pct.values, rng, breaks))
        o.update(_decline_stats(g))
        fits.append(dict(sta=name, prod=key, band=bn, combo='SC', **o))
    res['steps'] = pd.DataFrame(srows)
    res['fits'] = pd.DataFrame(fits)
    res['placebo'] = dict(dates=[d.strftime('%Y-%m-%d') for d in pl_dates], steps=plac)
    return res


def _dvv_job(job):
    net, sta, key, nboot, seed = job
    import pickle          # our own intermediate results (trusted, written by this script)
    name = f'{net}.{sta}'
    days, ccf, aut = load_sc(key, os.path.join(sdir(net, sta), 'days'))
    if days is None or np.isfinite(ccf[:, 0, :, 0]).any(axis=1).sum() < 60:
        return name, key, None
    r = run_dvv(name, key, days, ccf, nboot, seed, epoch_breaks(net, sta), aut=aut)
    pickle.dump(r, open(os.path.join(sdir(net, sta), f'_dvv_{key}.pkl'), 'wb'))
    return name, key, 'ok'


def stage_dvv(args):
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    sel = [s for s in STATIONS if not args.stations or s[1] in args.stations]
    jobs = [(n, s, k, args.nboot, 100 + i) for i, (n, s, _, _) in enumerate(sel) for k in (args.prods or ['day', 'low'])]
    if not args.collect_only:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            for r in ex.map(_dvv_job, jobs):
                print('dvv', *r, flush=True)
    acc = {k: [] for k in ('s10', 's5', 'monthly', 'steps', 'fits')}
    meta = {}
    for n, s, _, _ in STATIONS:
        for k in ('low', 'day'):
            fn = os.path.join(sdir(n, s), f'_dvv_{k}.pkl')
            if not os.path.exists(fn):
                continue
            r = pickle.load(open(fn, 'rb'))
            for a in acc:
                acc[a].append(r[a])
            meta[f'{n}.{s}:{k}'] = r['meta']
    names = dict(s10='sc_10day', s5='sc_5day', monthly='sc_monthly', steps='sc_steps', fits='sc_longterm_fit')
    for a, nm in names.items():
        D = pd.concat(acc[a], ignore_index=True)
        D.to_csv(os.path.join(OUT, f'{nm}.csv'), index=False)
    json.dump(dict(meta=meta, events=EVENTS, excl_placebo=EXCL, ref_end=str(REF_END.date())),
              open(os.path.join(OUT, 'sc_meta.json'), 'w'), indent=1, default=str)
    S = pd.concat(acc['steps'], ignore_index=True)
    F = pd.concat(acc['fits'], ignore_index=True)
    with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 500):
        print(S[['sta', 'band', 'event', 'step_pct', 'se_mbb_pct', 'placebo_sd_pct', 'z_placebo', 'pairs_used']].round(3).to_string())
        cols = [c for c in ['sta', 'band', 'n', 'first', 'last', 'trend_per_yr', 'trend_per_yr_se', 'trend_preM7_per_yr',
                            'trend_preM7_per_yr_se', 'annual_amp_pct', 'annual_amp_se', 'annual_peak_doy', 'step_M7_perm',
                            'tau_days', 'coseis_M7', 'd2025_minus_2021_pct', 'null_4yr_sd_pct'] if c in F]
        print(F[cols].round(3).to_string())


# ----------------------------------------------------------------------------- JEWT verification
def stage_verify(args):
    """(1) dv/v code: rerun this script's SC pipeline on the existing JEWT day files and compare
    with jewt_noise_dvv_10day.csv (combo SC);  (2) correlation + float16 storage: process a 60-day
    window from NCEDC with this script, substitute those days and compare the 10-day SC values."""
    vdir = os.path.join(OUT, '_verify_jewt', 'days')
    os.makedirs(vdir, exist_ok=True)
    v0, v1 = pd.Timestamp('2025-03-05'), pd.Timestamp('2025-05-03')      # 6 M7-aligned 10-d blocks
    from concurrent.futures import ProcessPoolExecutor
    jobs = [('BK', 'JEWT', d.strftime('%Y-%m-%d'), vdir) for d in pd.date_range(v0, v1)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_do_day, jobs))
    print(f'verify: processed {len(res)} JEWT days in {time.time() - t0:.0f} s;',
          sum(r[3].startswith('ok') for r in res), 'ok', flush=True)
    T = pd.read_csv(os.path.join(J.OUT, 'jewt_noise_dvv_10day.csv'), parse_dates=['t'])
    T = T[T.combo == 'SC']
    rows = []
    for key in ('day', 'low'):
        P = PRODS[key]
        days, ccf = load_jewt_sc(key)
        # same line notch as J (detected on the 6-pair JEWT record)
        _, c6 = J.load_prod(key)
        keep, lines = J.detect_lines_g(P, days, c6, 2)
        ai = auto_idx(P.fs, P.maxlag)
        keep2, lines2 = detect_lines_sc(P, days, c6[:, :, 3:6].astype('float32'), 2,
                                        c6[:, :, 0:3][..., ai].astype('float32'))
        print(f'{key}: line detector J (6 pairs) {lines}\n   this script (SC + stored auto windows) {lines2}; '
              f'max|keep diff| {np.abs(keep - keep2).max():.2e}', flush=True)
        del c6
        rA = run_dvv('BK.JEWT', key, days, ccf.copy(), 0, 1, {}, keep_override=keep, steps=False)
        # (2) substitute the newly processed days
        dn, cn, _ = load_sc(key, vdir)
        ccf2 = ccf.copy()
        for i, d in enumerate(dn):
            j = days.get_loc(d)
            ccf2[j] = cn[i]
        # reference from the ORIGINAL record so only the 60 days differ
        Xo = J.notch_g(P, ccf, keep)
        with np.errstate(invalid='ignore'):
            ref = np.nanmean(Xo[days < REF_END], axis=0, dtype='float64')
        rB = run_dvv('BK.JEWT', key, days, ccf2, 0, 1, {}, ref_override=ref, keep_override=keep, steps=False)
        for bn in P.names:
            t = T[(T['prod'] == key) & (T.band == bn)].set_index('t')
            a = rA['s10'][(rA['s10'].band == bn) & (rA['s10'].combo == 'SC')].set_index('t')
            b = rB['s10'][(rB['s10'].band == bn) & (rB['s10'].combo == 'SC')].set_index('t')
            j = pd.concat([t.dvv_pct.rename('jewt'), t.err_pct.rename('jewt_err'), a.dvv_pct.rename('rerun_existing_days'),
                           b.dvv_pct.rename('new_days')], axis=1)
            w = j[(j.index >= v0) & (j.index <= v1)]
            for tt, r in w.iterrows():
                rows.append(dict(prod=key, band=bn, t=tt, **r.to_dict()))
            dd = (j.jewt - j.rerun_existing_days).abs()
            print(f'{key} {bn}: whole record, rerun(existing days, SC code) vs JEWT csv: max|diff| {dd.max():.4f}% '
                  f'(median err {j.jewt_err.median():.3f}%), n={dd.notna().sum()}', flush=True)
    V = pd.DataFrame(rows)
    V['diff_new_minus_jewt'] = V.new_days - V.jewt
    V['z'] = V.diff_new_minus_jewt / V.jewt_err
    V.to_csv(os.path.join(OUT, 'verify_jewt_60d.csv'), index=False)
    with pd.option_context('display.width', 200):
        print(V.round(4).to_string())


# ----------------------------------------------------------------------------- figures
PANELS = [('low', '0.1-1Hz', '0.1–1 Hz (microseism band)'), ('day', '1-3Hz', '1–3 Hz'), ('day', '3-5Hz', '3–5 Hz')]
PCOL = {'EN': '#1f77b4', 'EZ': '#ff7f0e', 'NZ': '#9467bd'}
QCOL = {'M7': 'r', 'Petrolia': 'salmon', 'Ferndale': 'salmon'}


def _families_center():
    f = pd.read_csv(os.path.join(ROOT, 'data/families/shelly_family_locations.csv'))
    return f.lat.mean(), f.lon.mean()


def _dist_km(lat, lon):
    la0, lo0 = _families_center()
    return 111.2 * np.hypot(lat - la0, (lon - lo0) * np.cos(np.radians(la0)))


def _coords():
    from obspy import read_inventory
    inv = read_inventory(os.path.join(OUT, 'stations_ncedc.xml'))
    return {f'{n.code}.{s.code}': (s.latitude, s.longitude) for n in inv for s in n}


def stage_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.read_csv(os.path.join(OUT, 'sc_10day.csv'), parse_dates=['t'])
    D = pd.read_csv(os.path.join(OUT, 'sc_5day.csv'), parse_dates=['t'])
    Mo = pd.read_csv(os.path.join(OUT, 'sc_monthly.csv'), parse_dates=['month'])
    S = pd.read_csv(os.path.join(OUT, 'sc_steps.csv'))
    F = pd.read_csv(os.path.join(OUT, 'sc_longterm_fit.csv'))
    meta = json.load(open(os.path.join(OUT, 'sc_meta.json')))['meta']
    coords = _coords()
    for n, s, _, _ in STATIONS:
        name = f'{n}.{s}'
        if name not in set(T.sta):
            continue
        fig, axs = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)
        for ax, (pk, bn, lab) in zip(axs, PANELS):
            sel = lambda X: X[(X.sta == name) & (X['prod'] == pk) & (X.band == bn)]
            t10 = sel(T)
            for pr, col in PCOL.items():
                g = t10[t10.combo == pr]
                ax.plot(g.t, g.dvv_pct, '.-', ms=2.5, lw=0.4, color=col, alpha=0.5, label=f'{pr} (10-day)')
            g5 = sel(D)[sel(D).combo == 'SC']
            ax.plot(g5.t, g5.dvv_pct, '-', lw=0.4, color='0.55', alpha=0.6, label='combined, 5-day')
            g = t10[t10.combo == 'SC']
            ax.errorbar(g.t, g.dvv_pct, yerr=g.err_pct, fmt='o', ms=2.4, lw=0.5, color='k', label='combined, 10-day')
            gm = sel(Mo)
            gm = gm.set_index('month').asfreq('MS') if len(gm) else gm
            ax.plot(gm.index + pd.Timedelta(days=15), gm.dvv_pct, '-', lw=1.6, color='C3', label='combined, monthly')
            for ev, te in EVENTS.items():
                ax.axvline(pd.Timestamp(te), color=QCOL[ev], lw=1.1 if ev == 'M7' else 0.9, ls='-' if ev == 'M7' else '--')
            for k, v in epoch_breaks(n, s).items():
                if pd.Timestamp(v) > t10.t.min():
                    ax.axvline(pd.Timestamp(v), color='0.4', ls=':', lw=0.9)
            ax.axhline(0, color='0.6', lw=0.5)
            ax.grid(alpha=0.25)
            txt = lab
            mk = meta.get(f'{name}:{pk}', {})
            pu = mk.get('pairs_used', {}).get(bn, [])
            txt += f'  [pairs used: {",".join(pu) or "none"}]'
            st = S[(S.sta == name) & (S['prod'] == pk) & (S.band == bn)]
            if len(st):
                txt += '   steps (30 d): ' + '; '.join(f'{r.event} {r.step_pct:+.3f}% (z {r.z_placebo:+.1f})' for r in st.itertuples())
            f = F[(F.sta == name) & (F['prod'] == pk) & (F.band == bn)]
            if len(f) and 'annual_amp_pct' in f and np.isfinite(f.annual_amp_pct.iloc[0]):
                f = f.iloc[0]
                txt += f'\nseasonal amplitude {f.annual_amp_pct:.3f}%, trend {f.trend_per_yr:+.3f} ± {f.trend_per_yr_se:.3f} %/yr'
                if 'trend_preM7_per_yr' in f and np.isfinite(f.trend_preM7_per_yr):
                    txt += f' (pre-M7 {f.trend_preM7_per_yr:+.3f} ± {f.trend_preM7_per_yr_se:.3f})'
                if 'd2025_minus_2021_pct' in f and np.isfinite(f.d2025_minus_2021_pct):
                    txt += f'; 2025−2021 {f.d2025_minus_2021_pct:+.3f}%'
                    if np.isfinite(f.get('null_4yr_sd_pct', np.nan)):
                        txt += f' (pre-2021 4-yr sd {f.null_4yr_sd_pct:.3f})'
            ax.set_title(txt, fontsize=8, loc='left')
            v = t10[t10.combo.isin(list(PCOL))].dvv_pct
            if v.notna().sum() > 10:
                lo, hi = np.nanpercentile(v, [1, 99])
                ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            ax.set_ylabel('dv/v (%)')
        axs[0].legend(fontsize=7, ncol=6, loc='lower left')
        tt = T[T.sta == name].t
        axs[-1].set_xlim(tt.min() - pd.Timedelta(days=20), tt.max() + pd.Timedelta(days=20))
        mk = meta.get(f'{name}:day', {})
        la, lo_ = coords.get(name, (np.nan, np.nan))
        fig.suptitle(f'{name} single-station cross-component (EN, EZ, NZ) noise dv/v, {mk.get("first", "")} – {mk.get("last", "")}'
                     f' ({_dist_km(la, lo_):.0f} km from the LFE families)\nReference = {mk.get("ref_period", "")}; red = 2024-12-05 M7, '
                     'dashed = Petrolia 2021-12-20 / Ferndale 2022-12-20, dotted = StationXML epochs', fontsize=9.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f'sci_noise_dvv_sc_full_{s}.png'), dpi=130)
        plt.close(fig)
    # overview
    names = [f'{n}.{s}' for n, s, _, _ in STATIONS if f'{n}.{s}' in set(Mo.sta)]
    dist = {nm: _dist_km(*coords[nm]) for nm in names}
    names = sorted(names, key=dist.get)
    fig, axs = plt.subplots(1, 2, figsize=(15, 1.0 + 0.62 * len(names)), sharey=True)
    for ax, bn in zip(axs, ['1-3Hz', '3-5Hz']):
        g = Mo[(Mo['prod'] == 'day') & (Mo.band == bn)]
        sp = 0.5 if bn == '1-3Hz' else 0.8
        for i, nm in enumerate(names):
            h = g[g.sta == nm].set_index('month').dvv_pct
            if h.notna().sum() == 0:
                continue
            h = h.asfreq('MS')
            y0 = -i * sp
            ax.plot(h.index + pd.Timedelta(days=15), y0 + h - np.nanmedian(h), '-', lw=0.9,
                    color='C0' if nm.startswith('PB') else ('C2' if nm.startswith('NC') else 'C1'))
            ax.axhline(y0, color='0.85', lw=0.5, zorder=0)
            ax.text(pd.Timestamp('2008-01-01'), y0 + 0.08 * sp, f'{nm.split(".")[1]} {dist[nm]:.0f} km', fontsize=7)
        for ev, te in EVENTS.items():
            ax.axvline(pd.Timestamp(te), color=QCOL[ev], lw=1 if ev == 'M7' else 0.8, ls='-' if ev == 'M7' else '--')
        ax.plot([pd.Timestamp('2009-01-01')] * 2, [-(len(names) - 0.3) * sp, -(len(names) - 0.3) * sp + sp * 0.5], 'k-', lw=2)
        ax.text(pd.Timestamp('2009-03-01'), -(len(names) - 0.3) * sp, f'{0.5 * sp:.2f} %', fontsize=7)
        ax.set_title(f'{bn.replace("-", "–").replace("Hz", " Hz")} SC dv/v, monthly (offset by {sp} % per station; '
                     'blue PB borehole, green NC, orange BK)', fontsize=9)
        ax.set_xlim(pd.Timestamp('2007-10-01'), pd.Timestamp('2027-01-01'))
        ax.grid(alpha=0.2, axis='x')
        ax.set_yticks([])
    fig.suptitle('Single-station cross-component ambient-noise dv/v, all stations, ordered by distance to the LFE families '
                 '(median removed; red = M7, dashed = Petrolia/Ferndale)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_dvv_sc_full_all.png'), dpi=130)
    plt.close(fig)
    print('figures written', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['download', 'verify', 'dvv', 'plot'])
    ap.add_argument('--stations', nargs='*')
    ap.add_argument('--prods', nargs='*')
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--days', nargs='*')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--report', type=int, default=50)
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--collect-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    globals()['stage_' + a.stage](a)
