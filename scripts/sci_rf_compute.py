#!/usr/bin/env python3
"""Stage 3: P receiver functions per station-event (resumable by event chunk).
Per station-event:
  - 3 components present, no gaps over P-30..P+90 s (P = iasp91 at the station)
  - gain correction (remove_sensitivity, epoch-aware StationXML) -- components of one station share the sensor
    response, which cancels in the deconvolution; detrend, taper, anti-alias 8 Hz, decimate to 20 Hz
  - rotate 1/2/Z -> Z/N/E with StationXML azimuths/dips, N/E -> R/T with the station back-azimuth
  - high-pass 0.05 Hz (broadband) / 0.1 Hz (PB 2-Hz borehole geophones)
  - P signal-to-noise ratio on Z and R (Gaussian a=1 band, signal P-2..P+18 s, noise P-28..P-5 s)
  - iterative time-domain deconvolution (Ligorria & Ammon 1999) of R and T by Z, window P-15..P+75 s,
    Gaussian a = 1, 2.5, 5, 10 (low-pass ~0.5, 1.2, 2.4, 4.7 Hz), <=400 spikes at lags -5..30 s,
    fit = variance reduction of the filtered numerator.  RF normalised so a unit spike has unit peak.
  - modified Wiener deconvolution (Audet 2010; pre-event-noise regularised, unfiltered to 4 Hz) of R and T
    for the Herath & Audet (2024) spectral-power analysis (sci_rf_power.py)
  - stored for Z SNR >= 1.5 (selection later)
-> data/sci/rf_timelapse/chunks/c{k:03d}.pkl  ->  (gather) data/sci/rf_timelapse/rf_{NET.STA}.npz
Usage: sci_rf_compute.py [--workers 4] [--chunk 100] ; sci_rf_compute.py --gather
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT, STATIONS, PRE, POST, coords, ev_dir, excluded, inventory  # noqa: E402

warnings.filterwarnings('ignore')
FS = 20.0
DT = 1.0 / FS
NT = int((PRE + POST) * FS)              # samples in the P-30..P+90 window
T_DATA = -PRE + np.arange(NT) * DT
GAUSS = (1.0, 2.5, 5.0, 10.0)
LAG0, LAG1 = -5.0, 30.0
T_RF = np.arange(int(round(LAG0 * FS)), int(round(LAG1 * FS)) + 1) * DT
DEC_WIN = (-15.0, 75.0)
NMAX = 400
CHUNK_DIR = os.path.join(OUT, 'chunks')


def gauss_filter(nf, dt, a):
    f = np.fft.rfftfreq(nf, dt)
    return np.exp(-(2 * np.pi * f) ** 2 / (4 * a * a))


def itdecon(num, den, a, nmax=NMAX, minderr=1e-4):
    """Iterative time-domain deconvolution. -> rf on T_RF, fit (%) , nspikes."""
    n = len(num)
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    G = gauss_filter(nf, DT, a)
    NUM = np.fft.rfft(num, nf) * G
    DEN = np.fft.rfft(den, nf) * G
    pw = np.sum(np.fft.irfft(NUM, nf) ** 2)
    c = np.fft.irfft(NUM * np.conj(DEN), nf)          # c[k] = sum_t w[t] u[t-k]
    ac = np.fft.irfft(DEN * np.conj(DEN), nf)
    ac0 = ac[0]
    if pw <= 0 or ac0 <= 0:
        return None, 0.0, 0
    k0, k1 = int(round(LAG0 * FS)), int(round(LAG1 * FS))
    idx = np.arange(k0, k1 + 1) % nf
    spikes = np.zeros(nf)
    err = pw
    fit_old = 0.0
    it = 0
    for it in range(1, nmax + 1):
        j = idx[np.argmax(np.abs(c[idx]))]
        amp = c[j] / ac0
        spikes[j] += amp
        err -= amp * c[j]
        c -= amp * np.roll(ac, j)
        fit = 1 - err / pw
        if fit - fit_old < minderr:
            break
        fit_old = fit
    g = np.fft.irfft(G, nf)
    rf = np.fft.irfft(np.fft.rfft(spikes) * G, nf) / g[0]
    return rf[idx].astype(np.float32), 100 * fit, it


def wiener_mod(x):
    """Modified Wiener deconvolution (Audet 2010 BSSA, as in RfPy 'wiener-mod'): spectral division regularised by
    the pre-event noise power of the vertical, RF = R Z* / (Z Z* + S_nn); no Gaussian filter, only a cosine
    low-pass 4-6 Hz, so the 1.25-3 Hz band used by the power analysis is unbiased.  Signal window P-10..P+60 s,
    noise window P-30..P-3 s (power scaled to the signal-window taper energy, smoothed over ~0.1 Hz).
    Normalised by the peak of the vertical P 'RF' (Z Z* / (Z Z* + S_nn)).  -> (R, T) RFs on T_RF."""
    from scipy.signal.windows import tukey
    from scipy.ndimage import uniform_filter1d
    sig = (T_DATA >= -10) & (T_DATA < 60)
    noi = (T_DATA >= -30) & (T_DATA < -3)
    nf = 2048
    ts, tn = tukey(sig.sum(), 0.1), tukey(noi.sum(), 0.2)
    Z = np.fft.rfft(x['Z'][sig] * ts, nf)
    N = np.fft.rfft(x['Z'][noi] * tn, nf)
    Snn = uniform_filter1d(np.abs(N) ** 2 * (np.sum(ts ** 2) / np.sum(tn ** 2)), 9, mode='nearest')
    Spp = np.abs(Z) ** 2
    f = np.fft.rfftfreq(nf, DT)
    lp = np.clip((6.0 - f) / 2.0, 0, 1)
    lp = 0.5 - 0.5 * np.cos(np.pi * lp)
    den = Spp + Snn + 1e-30
    rz = np.fft.irfft(lp * Spp / den, nf)
    norm = rz.max()
    k = np.arange(int(round(LAG0 * FS)), int(round(LAG1 * FS)) + 1) % nf
    out = []
    for c in 'RT':
        X = np.fft.rfft(x[c][sig] * ts, nf)
        out.append(np.fft.irfft(lp * X * np.conj(Z) / den, nf)[k] / norm)
    return np.array(out, np.float32)


def gfilt(x, a):
    nf = 1 << int(np.ceil(np.log2(2 * len(x))))
    return np.fft.irfft(np.fft.rfft(x, nf) * gauss_filter(nf, DT, a), nf)[:len(x)]


def snr(x, a=1.0):
    y = gfilt(x, a)
    s = (T_DATA >= -2) & (T_DATA <= 18)
    nn = (T_DATA >= -28) & (T_DATA <= -5)
    return float(np.sqrt(np.mean(y[s] ** 2) / max(np.mean(y[nn] ** 2), 1e-30)))


def prep_station(st, code, tp, baz, inv):
    """-> dict(Z,R,T arrays on T_DATA) or (None, reason)."""
    from obspy import Stream
    from scipy.signal import butter, sosfiltfilt
    net, sta = code.split('.')
    s = st.select(network=net, station=sta)
    if not len(s):
        return None, 'nodata'
    locs = sorted({tr.stats.location for tr in s}, key=lambda l: (l not in ('', '00'), l))
    s = s.select(location=locs[0])
    s = Stream([tr for tr in s if tr.stats.channel[:2] in ('HH', 'EH')])
    try:
        s.merge(method=1, fill_value=None)
    except Exception:
        return None, 'merge'
    if len(s) != 3 or any(np.ma.is_masked(tr.data) for tr in s):
        return None, 'gap'
    t1, t2 = tp - PRE - 1.0, tp + POST + 1.0
    if any(tr.stats.starttime > t1 or tr.stats.endtime < t2 for tr in s):
        return None, 'short'
    if max(np.abs(tr.data).max() for tr in s) > 0.95 * 2 ** 23 and code.startswith('PB'):
        clip = True
    else:
        clip = False
    for tr in s:
        tr.data = tr.data.astype(np.float64)
        if abs(tr.stats.sampling_rate - round(tr.stats.sampling_rate)) < 1e-3:
            tr.stats.sampling_rate = round(tr.stats.sampling_rate)
    try:
        s.remove_sensitivity(inv)
    except Exception:
        return None, 'resp'
    s.detrend('linear'); s.taper(0.05)
    for tr in s:
        tr.filter('lowpass', freq=8.0, corners=8, zerophase=True)
        sr = tr.stats.sampling_rate
        if abs(sr / FS - round(sr / FS)) < 1e-6:
            tr.decimate(int(round(sr / FS)), no_filter=True)
        else:
            tr.resample(FS, no_filter=True)
    try:
        s.rotate('->ZNE', inventory=inv)
    except Exception:
        return None, 'rot'
    s.rotate('NE->RT', back_azimuth=baz)
    out = {}
    for c in 'ZRT':
        tr = s.select(component=c)
        if len(tr) != 1:
            return None, 'comp'
        tr = tr[0]
        i0 = int(round((tp - PRE - tr.stats.starttime) * FS))
        x = tr.data[i0:i0 + NT]
        if i0 < 0 or len(x) != NT:
            return None, 'short'
        out[c] = x - x.mean()
    hp = 0.1 if code.startswith('PB') else 0.05
    sos = butter(2, hp, 'highpass', fs=FS, output='sos')
    w = np.ones(NT)
    m = int(5 * FS)
    w[:m] = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m)); w[-m:] = w[:m][::-1]
    for c in out:
        out[c] = sosfiltfilt(sos, out[c] * w)
    out['clip'] = clip
    return out, 'ok'


def do_chunk(job):
    k, evs, C = job
    fn = os.path.join(CHUNK_DIR, f'c{k:03d}.pkl')
    if os.path.exists(fn):
        return k, 'cached'
    from obspy import read, Stream, UTCDateTime
    from obspy.geodetics import gps2dist_azimuth, locations2degrees
    inv = inventory()
    rows, rfs, rfws = [], [], []
    nwin = DEC_WIN
    wmask = (T_DATA >= nwin[0]) & (T_DATA <= nwin[1])
    m = int(3 * FS)
    tap = np.ones(wmask.sum()); tap[:m] = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m)); tap[-m:] = tap[:m][::-1]
    for ev in evs:
        d = ev_dir(ev['evid'])
        sfn = os.path.join(d, 'status.json')
        if not os.path.exists(sfn):
            continue
        status = json.load(open(sfn))
        st = Stream()
        for name in ('ncedc', 'pnw'):
            f = os.path.join(d, f'{name}.mseed')
            if os.path.exists(f):
                try:
                    st += read(f)
                except Exception:
                    pass
        t0 = UTCDateTime(pd.Timestamp(ev['time']).to_pydatetime())
        for code, src in status.items():
            if src not in ('ncedc', 'pnw') or code not in C:
                continue
            la, lo = C[code][:2]
            dist = locations2degrees(la, lo, ev['lat'], ev['lon'])
            baz = gps2dist_azimuth(la, lo, ev['lat'], ev['lon'])[1]   # azimuth station->event = back-azimuth
            tp = t0 + ev['tP'] + ev['p_sdeg'] * (dist - ev['dist'])
            rec = dict(evid=ev['evid'], sta=code, time=ev['time'], mag=ev['mag'], depth=ev['depth'], dist=dist,
                       baz=baz, p=ev['p_skm'], src=src, excl=excluded(code, ev['time']))
            try:
                x, why = prep_station(st, code, tp, baz, inv)
            except Exception as e:
                x, why = None, f'exc:{type(e).__name__}'
            rec['status'] = why
            if x is None:
                rows.append(rec); continue
            rec['clip'] = x['clip']
            rec['snr_z'] = snr(x['Z']); rec['snr_r'] = snr(x['R']); rec['snr_t'] = snr(x['T'])
            rec['snr_z25'] = snr(x['Z'], 2.5)
            if rec['snr_z'] < 1.5:
                rec['status'] = 'lowsnr'; rows.append(rec); continue
            Z, R, T = (x[c][wmask] * tap for c in 'ZRT')
            out = np.zeros((len(GAUSS), 2, len(T_RF)), np.float32)
            for i, a in enumerate(GAUSS):
                for j, num in enumerate((R, T)):
                    rf, fit, nsp = itdecon(num, Z, a)
                    if rf is None:
                        break
                    out[i, j] = rf
                    rec[f'fit_{"rt"[j]}{a:g}'] = fit
                    if j == 0:
                        rec[f'nsp{a:g}'] = nsp
            rec['irf'] = len(rfs)
            rfs.append(out)
            rfws.append(wiener_mod(x))
            rows.append(rec)
    tmp = fn + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(dict(rows=rows, rfs=np.array(rfs, np.float32) if rfs else np.zeros((0, len(GAUSS), 2, len(T_RF)), np.float32),
                         rfw=np.array(rfws, np.float32) if rfws else np.zeros((0, 2, len(T_RF)), np.float32)), f)
    os.replace(tmp, fn)
    return k, f'{len(rows)} rows {len(rfs)} rfs'


def gather():
    rows, rfs, rfws, off = [], [], [], 0
    for fn in sorted(glob.glob(os.path.join(CHUNK_DIR, 'c*.pkl'))):
        with open(fn, 'rb') as f:          # our own intermediate files (trusted pickle written by do_chunk)
            d = pickle.load(f)
        for r in d['rows']:
            if 'irf' in r:
                r['irf'] = r['irf'] + off
            rows.append(r)
        rfs.append(d['rfs']); rfws.append(d['rfw']); off += len(d['rfs'])
    D = pd.DataFrame(rows)
    RF = np.concatenate(rfs) if rfs else None
    RFW = np.concatenate(rfws) if rfws else None
    D.drop(columns=['irf']).to_csv(os.path.join(OUT, 'qc_all.csv'), index=False)
    for code, g in D[D.irf.notna()].groupby('sta'):
        g = g.sort_values('time')
        ii = g.irf.astype(int).values
        meta = g.drop(columns=['irf']).reset_index(drop=True)
        np.savez_compressed(os.path.join(OUT, f'rf_{code}.npz'), rf=RF[ii], rfw=RFW[ii], t=T_RF, gauss=np.array(GAUSS),
                            meta=meta.to_json(orient='records'))
        print(code, len(g), flush=True)
    print(D.groupby('sta').status.value_counts().unstack(fill_value=0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=100)
    ap.add_argument('--gather', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CHUNK_DIR, exist_ok=True)
    if a.gather:
        return gather()
    E = pd.read_csv(os.path.join(OUT, 'events.csv'))
    if a.limit:
        E = E.iloc[:a.limit]
    C = coords()
    evs = E.to_dict('records')
    jobs = [(k, evs[i:i + a.chunk], C) for k, i in enumerate(range(0, len(evs), a.chunk))]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(do_chunk, j) for j in jobs]):
            k, msg = f.result()
            print(k, msg, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
