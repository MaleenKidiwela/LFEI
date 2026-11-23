#!/usr/bin/env python3
"""Multi-family matched-filter scan around a quake, with hourly noise/threshold statistics.

Derived from scan_matched_filter.py (same templates, preprocessing, CC normalisation, catalog-time
indexing, peak rules), but
  * all families are scanned in one pass per day (data read and filtered once);
  * detections are saved down to a low floor (6 x the day's minimum hourly MAD) so that
    any threshold can be applied afterwards: the standard daily 8xMAD, an hourly 8xMAD, or a FIXED
    absolute CC-sum threshold;
  * per family and hour: MAD of the network CC sum, median channel count, and the floor;
  * per channel and hour: RMS of the 2-8 Hz filtered data (noise level through time).
Optional --inject: add donor waveforms (raw continuous segments of typical pre-quake catalog
events of the same family) at known times, to measure detection efficiency directly.

Output dir data/scan_quakes/{tag}/:
  det_{day}.csv     fam, time_utc, sign, ccsum, nchan, meancc, mad_day, mad_hour
  hourly_{day}.csv  fam, hour (UTC start), mad, nchan_med, floor
  chanrms_{day}.csv cid, hour, rms
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from scipy.ndimage import maximum_filter1d
from scipy import fft as sfft
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, os.path.dirname(__file__))
from scan_matched_filter import SR, load_templates  # noqa: E402

NPTS = int(86400 * SR)
HR = int(3600 * SR)
BL = 16384          # overlap-save FFT block length
FLOOR_K = 6.0
TEMPLATES = {}   # filled in workers via initializer
INJ = None


def init(tpl_paths, model, min_snr, inj_path):
    global TEMPLATES, INJ
    for F, p in tpl_paths.items():
        TEMPLATES[F] = load_templates(F, model, min_snr, p)
    if inj_path:   # injection table is our own file written by scan_quakes_inject.py
        INJ = pd.read_pickle(inj_path)


def moving_stats(x, n):
    cs = np.cumsum(np.r_[0.0, x])
    cs2 = np.cumsum(np.r_[0.0, x * x])
    s1 = cs[n:] - cs[:-n]
    s2 = cs2[n:] - cs2[:-n]
    var = s2 - s1 * s1 / n
    pos = var > 0
    good = var > 1e-3 * np.median(var[pos]) if np.any(pos) else np.zeros_like(var, bool)
    den = np.zeros_like(var)
    den[good] = 1.0 / np.sqrt(var[good])
    return den


def scan_day(args):
    day, cont_dir, out_dir, inject = args
    os.makedirs(out_dir, exist_ok=True)
    fdet = os.path.join(out_dir, f'det_{day}.csv')
    if os.path.exists(fdet):
        return day, -1, 0
    t0 = UTCDateTime(day)
    sos = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
    need = set().union(*[set(t) for t in TEMPLATES.values()])
    data, valid = {}, {}
    for f in glob.glob(os.path.join(cont_dir, day, '*.mseed')):
        try:
            st = read(f)
        except Exception:
            continue
        st.merge(method=1, fill_value=0)
        for tr in st:
            if tr.id not in need:
                continue
            tr.detrend('demean')
            tr.taper(max_percentage=None, max_length=5)
            if tr.stats.sampling_rate != SR:
                fs = tr.stats.sampling_rate
                tr.filter('lowpass', freq=0.4 * SR, corners=8, zerophase=True)
                tr.resample(SR, no_filter=True) if fs % SR else tr.decimate(int(fs // SR), no_filter=True)
            x = np.zeros(NPTS)
            i = int(round((tr.stats.starttime - t0) * SR))
            d = tr.data.astype(float)
            a0, b0 = max(0, i), min(NPTS, i + len(d))
            if b0 <= a0:
                continue
            x[a0:b0] = d[a0 - i:b0 - i]
            v = np.zeros(NPTS, bool)
            v[a0:b0] = d[a0 - i:b0 - i] != 0
            data[tr.id], valid[tr.id] = x, v
    if not data:
        pd.DataFrame().to_csv(fdet, index=False)
        return day, 0, 0
    # injection of donor raw segments (already demeaned, tapered, resampled to SR) at known times
    if inject and INJ is not None:
        tab = INJ['table']
        for _, r in tab[tab.day == day].iterrows():
            don = INJ['donors'][r.donor]
            i0 = int(round((UTCDateTime(r.t_inj) + don['off0'] - t0) * SR))
            for cid, seg in don['segs'].items():
                if cid in data:
                    a0, b0 = max(0, i0), min(NPTS, i0 + len(seg))
                    if b0 > a0:
                        data[cid][a0:b0] += seg[a0 - i0:b0 - i0] * valid[cid][a0:b0]
    rms_rows = []
    for cid in data:
        x = sosfiltfilt(sos, data[cid])
        data[cid] = x.astype(np.float32)
        for h in range(24):
            vv = valid[cid][h * HR:(h + 1) * HR]
            if vv.mean() > 0.5:
                rms_rows.append(dict(cid=cid, hour=h, rms=float(np.sqrt(np.mean(x[h * HR:(h + 1) * HR][vv] ** 2)))))
    pd.DataFrame(rms_rows).to_csv(os.path.join(out_dir, f'chanrms_{day}.csv'), index=False)
    # correlation: channel-outer, overlap-save with one forward FFT per channel shared by all families
    fam_list = list(TEMPLATES)
    ccsums = {F: np.zeros(NPTS, np.float32) for F in fam_list}
    nchs = {F: np.zeros(NPTS, np.float32) for F in fam_list}
    mmax = max(len(t[0]) for T in TEMPLATES.values() for t in T.values())
    step = BL - mmax + 1
    for cid in data:
        users = [(F, TEMPLATES[F][cid]) for F in fam_list if cid in TEMPLATES[F]]
        if not users:
            continue
        x = data[cid]
        nb = int(np.ceil((NPTS - mmax + 1) / step))
        pad = np.zeros((nb - 1) * step + BL, np.float32)
        pad[:NPTS] = x
        X = sfft.rfft(np.lib.stride_tricks.sliding_window_view(pad, BL)[::step], axis=1)
        dens, vcs = {}, {}
        for F, (tpl, w0) in users:
            n = len(tpl)
            if n not in dens:
                dens[n] = moving_stats(x.astype(float), n).astype(np.float32)
                vcs[n] = valid[cid][n // 2:n // 2 + NPTS - n + 1].astype(np.float32)
            Tk = np.conj(sfft.rfft(tpl.astype(np.float32), n=BL))
            num = sfft.irfft(X * Tk, n=BL, axis=1)[:, :step].ravel()[:NPTS - n + 1]
            cc = np.clip(num * dens[n], -1.0, 1.0) * vcs[n]
            shift = int(round(w0 * SR))
            j0, j1 = max(0, shift), min(NPTS - n + 1, NPTS + shift)
            if j1 <= j0:
                continue
            ccsums[F][j0 - shift:j1 - shift] += cc[j0:j1]
            nchs[F][j0 - shift:j1 - shift] += vcs[n][j0:j1]
    det_rows, hr_rows = [], []
    for F in fam_list:
        ccsum = ccsums[F].astype(float)
        nch = nchs[F]
        if not nch.any():
            continue
        mad_day = np.median(np.abs(ccsum - np.median(ccsum)))
        mad_h = np.full(24, np.nan)
        for h in range(24):
            seg = ccsum[h * HR:(h + 1) * HR]
            ns = nch[h * HR:(h + 1) * HR]
            if (ns > 0).mean() > 0.5:
                seg = seg[ns > 0]
                mad_h[h] = np.median(np.abs(seg - np.median(seg)))
            hr_rows.append(dict(fam=F, hour=h, mad=mad_h[h], nchan_med=float(np.median(ns)), mad_day=mad_day))
        floor = FLOOR_K * np.nanmin(mad_h) if np.isfinite(mad_h).any() else FLOOR_K * mad_day
        for r in hr_rows[-24:]:
            r['floor'] = floor
        env = maximum_filter1d(np.abs(ccsum), size=int(2 * SR) + 1)
        for sign in (1, -1):
            y = sign * ccsum
            pk, _ = find_peaks(y, height=floor, distance=int(4 * SR))
            pk = pk[np.abs(ccsum[pk]) >= env[pk] - 1e-9]
            for p in pk:
                det_rows.append(dict(fam=F, time_utc=str(t0 + p / SR), sign=sign, ccsum=ccsum[p],
                                     nchan=int(nch[p]), meancc=ccsum[p] / max(nch[p], 1),
                                     mad_day=mad_day, mad_hour=mad_h[p // HR]))
    pd.DataFrame(hr_rows).to_csv(os.path.join(out_dir, f'hourly_{day}.csv'), index=False)
    pd.DataFrame(det_rows).to_csv(fdet + '.tmp', index=False)
    os.replace(fdet + '.tmp', fdet)
    return day, len(data), len(det_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--tag', required=True, help='petrolia | ferndale | m7 (output subdir)')
    ap.add_argument('--era', required=True, help='template era tag: petrolia | ferndale | 2024')
    ap.add_argument('--families', default='1-27')
    ap.add_argument('--model', default='3d')
    ap.add_argument('--min-snr', type=float, default=10.0)
    ap.add_argument('--cont', default='data/continuous')
    ap.add_argument('--out', default='data/scan_quakes')
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--inject', default=None, help='injection table (pickle); output to {tag}_inj')
    a = ap.parse_args()
    lo, hi = map(int, a.families.split('-')) if '-' in a.families else (int(a.families),) * 2
    fams = list(range(lo, hi + 1))
    paths = {F: f'data/templates/fam{F}_templates_era{a.era}.h5' for F in fams}
    out = os.path.join(a.out, a.tag + ('_inj' if a.inject else ''))
    init(paths, a.model, a.min_snr, None)
    for F in fams:
        print(f'fam {F}: {len(TEMPLATES[F])} template channels', flush=True)
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(a.start, a.end, freq='D')]
    if a.inject:
        inj = pd.read_pickle(a.inject)
        days = sorted(set(days) & set(inj['table'].day))
    with ProcessPoolExecutor(a.procs, initializer=init, initargs=(paths, a.model, a.min_snr, a.inject)) as ex:
        for day, nu, nd in ex.map(scan_day, [(d, a.cont, out, bool(a.inject)) for d in days]):
            print(f'{day}: {nu:3d} channels  {nd:5d} candidate peaks', flush=True)


if __name__ == '__main__':
    main()
