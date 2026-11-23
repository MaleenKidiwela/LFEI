#!/usr/bin/env python3
"""Matched-filter scan of continuous data with a family's master templates, for both
normal (positive network CC) and reversed (negative network CC) polarity.

Template per channel: master stack (2-8 Hz, 50 sps) cut in [S-1, S+5] s around the master
S pick (horizontals; and verticals of vertical-only stations), or [P-0.5, P+3.5] on verticals
of 3-component stations if a P pick exists. Offsets are relative to catalog time, so the
network CC sum is indexed by catalog-convention time (comparable with Shelly's catalog).
Per day: demean/detrend/taper, 2-8 Hz, resample 50 sps; gaps zero-filled and masked.
Detection: ccsum > +K*MAD (normal) or ccsum < -K*MAD (reversed), MAD of the day's ccsum;
peaks separated by >= 4 s and required to be the absolute |ccsum| extremum within +-1 s
(rejects opposite-sign CC side-lobes of real events); meancc = ccsum/nchan reported.
Output data/scan/fam{F}_{YYYY-MM-DD}.csv and a daily summary.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from scipy.signal import butter, fftconvolve, find_peaks, sosfiltfilt

SR = 50.0
K = 8.0
EXCLUDE = {'NC.KPP..SHZ', 'NC.KCPB..HHN', 'NC.KCPB..HHE', 'NC.KBS..SHZ'}


def load_templates(F, model, min_snr, path=None):
    T = h5py.File(path or f'data/templates/fam{F}_templates.h5', 'r')
    pre = T.attrs['pre']
    pk = pd.read_csv(f'data/locations/fam{F}_{model}_picks.csv').set_index('sta')
    out = {}
    for cid in T:
        if cid in EXCLUDE or T[cid].attrs['snr'] < min_snr:
            continue
        n, s, l, c = cid.split('.')
        sta = f'{n}.{s}'
        if sta not in pk.index:
            continue
        r = pk.loc[sta]
        S = r['S_final'] if 'S_final' in r and np.isfinite(r['S_final']) else r['S']
        has_h = any(k.split('.')[0] + '.' + k.split('.')[1] == sta and k[-1] in 'EN12' for k in T)
        if c[-1] == 'Z' and has_h and 'P' in r and np.isfinite(r['P']):
            w0, w1 = r['P'] - 0.5, r['P'] + 3.5
        elif np.isfinite(S):
            w0, w1 = S - 1.0, S + 5.0
        else:
            continue
        st = T[cid]['stack'][:].astype(float)
        i0, i1 = int(round((w0 - pre) * SR)), int(round((w1 - pre) * SR))
        tpl = st[i0:i1] - st[i0:i1].mean()
        out[cid] = (tpl / np.linalg.norm(tpl), w0)
    return out


def norm_cc(x, tpl):
    """Normalised CC of unit-norm, zero-mean template with x at every lag (valid part)."""
    n = len(tpl)
    num = fftconvolve(x, tpl[::-1], mode='valid')
    cs = np.cumsum(np.r_[0.0, x])
    cs2 = np.cumsum(np.r_[0.0, x * x])
    s1 = cs[n:] - cs[:-n]
    s2 = cs2[n:] - cs2[:-n]
    var = s2 - s1 * s1 / n
    good = var > 1e-3 * np.median(var[var > 0]) if np.any(var > 0) else np.zeros_like(var, bool)
    cc = np.zeros_like(num)
    cc[good] = num[good] / np.sqrt(var[good])
    return np.clip(cc, -1.0, 1.0)       # near-zero variance (gaps, zero-fill) -> 0


def scan_day(args):
    day, F, templates, cont_dir, out_dir = args
    files = glob.glob(os.path.join(cont_dir, day, '*.mseed'))
    t0 = UTCDateTime(day)
    npts = int(86400 * SR)
    sos = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
    ccsum = np.zeros(npts)
    nch = np.zeros(npts)
    used = []
    for f in files:
        try:
            st = read(f)
        except Exception:
            continue
        st.merge(method=1, fill_value=0)
        for tr in st:
            if tr.id not in templates:
                continue
            tpl, w0 = templates[tr.id]
            tr.detrend('demean')
            tr.taper(max_percentage=None, max_length=5)
            if tr.stats.sampling_rate != SR:
                fs = tr.stats.sampling_rate
                tr.filter('lowpass', freq=0.4 * SR, corners=8, zerophase=True)
                tr.resample(SR, no_filter=True) if fs % SR else tr.decimate(int(fs // SR), no_filter=True)
            x = np.zeros(npts)
            i = int(round((tr.stats.starttime - t0) * SR))
            d = tr.data.astype(float)
            a0, b0 = max(0, i), min(npts, i + len(d))
            if b0 <= a0:
                continue
            x[a0:b0] = d[a0 - i:b0 - i]
            valid = np.zeros(npts, bool)
            valid[a0:b0] = d[a0 - i:b0 - i] != 0
            x = sosfiltfilt(sos, x)
            cc = norm_cc(x, tpl)
            # index k of cc = window starting at sample k = catalog time + w0
            shift = int(round(w0 * SR))
            k = np.arange(len(cc)) - shift            # catalog-time sample index
            ok = (k >= 0) & (k < npts)
            vv = valid[np.clip(np.arange(len(cc)) + len(tpl) // 2, 0, npts - 1)][ok]
            ccv = np.where(vv, cc[ok], 0.0)
            ccsum[k[ok]] += ccv
            nch[k[ok]] += vv
            used.append(tr.id)
    if not used:
        return day, 0, 0, 0
    mad = np.median(np.abs(ccsum - np.median(ccsum)))
    rows = []
    from scipy.ndimage import maximum_filter1d
    env = maximum_filter1d(np.abs(ccsum), size=int(2 * SR) + 1)   # max |ccsum| within +-1 s
    for sign in (1, -1):
        y = sign * ccsum
        pk, _ = find_peaks(y, height=K * mad, distance=int(4 * SR))
        # keep only peaks that are the absolute extremum within +-1 s: rejects the opposite-sign
        # side-lobes of oscillatory CC (half a dominant period ~0.1-0.15 s away from the true peak)
        pk = pk[np.abs(ccsum[pk]) >= env[pk] - 1e-12]
        for p in pk:
            rows.append(dict(time_utc=str(t0 + p / SR), sign=sign, ccsum=ccsum[p], mad=mad,
                             cc_div_thr=abs(ccsum[p]) / (K * mad), nchan=int(nch[p]),
                             meancc=ccsum[p] / max(nch[p], 1)))
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f'fam{F}_{day}.csv'), index=False)
    npos = int((df.sign == 1).sum()) if len(df) else 0
    nneg = int((df.sign == -1).sum()) if len(df) else 0
    return day, len(used), npos, nneg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--model', default='3d')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--min-snr', type=float, default=10.0)
    ap.add_argument('--cont', default='data/continuous')
    ap.add_argument('--out', default='data/scan')
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--templates', default=None, help='templates h5 (e.g. era-matched)')
    a = ap.parse_args()
    tpls = load_templates(a.family, a.model, a.min_snr, a.templates)
    print(f'{len(tpls)} template channels', flush=True)
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(a.start, a.end, freq='D')]
    rows = []
    with ProcessPoolExecutor(a.procs) as ex:
        for day, nu, npos, nneg in ex.map(scan_day, [(d, a.family, tpls, a.cont, a.out) for d in days]):
            print(f'{day}: {nu:3d} channels  +{npos:4d}  -{nneg:4d}', flush=True)
            rows.append(dict(day=day, n_chan=nu, n_pos=npos, n_neg=nneg))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f'fam{a.family}_daily_summary.csv'), index=False)


if __name__ == '__main__':
    main()
