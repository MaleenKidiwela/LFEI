#!/usr/bin/env python3
"""EQ-above step 1c: matched-filter detection of small earthquakes of each earthquake family in the 20 Hz archive.

Archive (read only): /data/wsd02/maleen_data/MTJ_20Hz/{NET.STA}/{YYYY}/{NET.STA}.{YYYY}.{DDD}.mseed (or wsd03 root).
tmpl : per family E (>= MINMEM catalog members) and archive station: catalog-member windows cut from the archive
       (origin -5 .. +45 s, 2-8 Hz), aligned by CC and stacked (sci_eq_above_measure.align_stack; mean member lag = 0,
       so the template is on the members' mean catalog-origin frame).  Template channels: Z [P-0.5, P+2.5] s,
       horizontals [S-0.5, S+3.5] s (P, S = 3D-model family-median prediction + measured stack offset); kept if the
       template SNR >= 3.  -> mf/templates.npz, mf/templates.csv
scan : per day (2009-01-01 .. 2026-11-15): every template channel with data that day: normalised CC (FFT), shifted to
       origin time, summed -> network mean CC per family.  Detection: mean CC > max(K x MAD(day), CCMIN), >= 3 s
       apart, >= NCHMIN channels, and the family with the highest mean CC within +-2 s wins (cross-family dedup).
       Amplitude: median over channels of log10(window RMS / template RMS) -> relative magnitude.
       -> mf/det_{YYYY}.csv (per year), mf/days_{YYYY}.csv (channels used, MAD per family)
Usage: nice -n 10 python scripts/sci_eq_above_mf.py --stage tmpl|scan [--procs 4] [--years 2009 2010 ...]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/eq_above'
MF = f'{OUT}/mf'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_eq_above_cluster import STAS  # noqa: E402

ARCH = ['/data/wsd02/maleen_data/MTJ_20Hz', '/data/wsd03/maleen_MTJ_20Hz']
SR = 20.0
PRE = 5.0
NS = int(50 * SR)
K = 8.0
CCMIN = 0.12
NCHMIN = 4
MINMEM = 6
PW = (-0.5, 2.5)
SW = (-0.5, 3.5)
SOS = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')


def day_file(sta, day):
    rel = f'{sta}/{day:%Y}/{sta}.{day:%Y}.{day:%j}.mseed'
    for r in ARCH:
        f = f'{r}/{rel}'
        if os.path.exists(f) and os.path.getsize(f) > 0:
            return f
    return None


def read_day(sta, day):
    """-> dict comp -> (t0 epoch s, filtered float array at 20 Hz) for one UTC day; gaps -> zeros (masked)."""
    from obspy import read
    f = day_file(sta, day)
    if f is None:
        return {}
    try:
        st = read(f)
    except Exception:                                  # noqa: BLE001
        return {}
    out = {}
    d0 = pd.Timestamp(day).value / 1e9
    n = int(86400 * SR)
    for tr in st:
        c = {'Z': 'Z', 'N': '1', '1': '1', 'E': '2', '2': '2'}.get(tr.stats.channel[-1])
        if c is None or abs(tr.stats.sampling_rate - SR) > 1e-6:
            continue
        x = np.zeros(n)
        m = np.zeros(n, bool)
        k0 = int(round((tr.stats.starttime.timestamp - d0) * SR))
        y = np.asarray(tr.data, float)
        a, b = max(k0, 0), min(k0 + len(y), n)
        if b - a < SR * 600:
            continue
        x[a:b] = y[a - k0:b - k0]
        m[a:b] = True
        if c in out:
            x0, m0 = out[c]
            x = np.where(m0, x0, x)
            m = m0 | m
        out[c] = (x, m)
    res = {}
    for c, (x, m) in out.items():
        if m.sum() < SR * 3600:
            continue
        mu = x[m].mean()
        x = np.where(m, x - mu, 0.0)
        y = sosfiltfilt(SOS, x)
        y[~m] = 0.0
        s = np.std(y[m])
        if s == 0:
            continue
        # clip glitches (>20 sigma) so they do not dominate the normalisation
        y = np.clip(y / s, -20, 20)
        res[c] = (y, m)
    return res


# ------------------------------------------------------------------------------------------------ templates
def archive_stations():
    """stations with archive files (any year)."""
    s = []
    for st in STAS:
        if any(os.path.isdir(f'{r}/{st}') for r in ARCH):
            s.append(st)
    return s


def cut_event(sta, t):
    """20 Hz 2-8 Hz window origin-5 .. +45 s from the archive day file(s); -> (3, NS) or None."""
    t = pd.Timestamp(t)
    t0 = t - pd.Timedelta(seconds=PRE)
    day = t0.normalize()
    D = read_day(sta, day)
    if not D:
        return None
    k0 = int(round((t0 - day).total_seconds() * SR))
    if k0 + NS > int(86400 * SR):
        return None
    A = np.full((3, NS), np.nan)
    for c, (y, m) in D.items():
        ci = {'Z': 0, '1': 1, '2': 2}[c]
        if m[k0:k0 + NS].all():
            A[ci] = y[k0:k0 + NS]
    return A if np.isfinite(A).any() else None


def tmpl_job(args):
    E, sta, times, P, S = args
    from sci_eq_above_measure import align_stack
    W = []
    for t in times:
        a = cut_event(sta, t)
        W.append(a if a is not None else np.full((3, NS), np.nan))
    W = np.array(W)
    r = align_stack(W, P, S, SR, PRE)
    if r is None:
        return None
    ref, lag, cc, good = r
    return E, sta, ref, int(good.sum())


def stage_tmpl(a):
    os.makedirs(MF, exist_ok=True)
    c = pd.read_csv(f'{OUT}/families.csv')
    TP = np.load(f'{OUT}/ttP.npy')
    TS = np.load(f'{OUT}/ttS.npy')
    arch = archive_stations()
    print('archive stations:', arch, flush=True)
    jobs = []
    fams = c[c.efam > 0].efam.value_counts()
    fams = sorted(fams[fams >= MINMEM].index)
    for E in fams:
        idx = np.where(c.efam.values == E)[0]
        times = c.Time.values[idx]
        for sta in arch:
            k = STAS.index(sta)
            jobs.append((E, sta, times, np.nanmedian(TP[idx, k]), np.nanmedian(TS[idx, k])))
    with Pool(a.procs) as p:
        res = p.map(tmpl_job, jobs, chunksize=1)
    rows, arrs = [], {}
    for (E, sta, times, P, S), r in zip(jobs, res):
        if r is None:
            continue
        _, _, ref, ng = r
        t = np.arange(NS) / SR - PRE
        noise = ref[:, (t > -4.5) & (t < P - 1.0)]
        for ci, comp in enumerate('Z12'):
            if not np.isfinite(ref[ci]).all() or np.std(ref[ci]) == 0:
                continue
            a0, a1 = (P + PW[0], P + PW[1]) if comp == 'Z' else (S + SW[0], S + SW[1])
            w = (t >= a0) & (t < a1)
            snr = np.sqrt(np.mean(ref[ci, w] ** 2)) / (np.sqrt(np.mean(noise[ci] ** 2)) + 1e-12)
            key = f'E{E}|{sta}|{comp}'
            arrs[key] = ref[ci, w].astype(np.float32)
            rows.append(dict(efam=E, sta=sta, comp=comp, off=a0, n_used=ng, snr=snr, key=key,
                             use=bool(snr >= 3 and ng >= 3)))
    T = pd.DataFrame(rows)
    T.to_csv(f'{MF}/templates.csv', index=False)
    np.savez(f'{MF}/templates.npz', **arrs)
    print(T[T.use].groupby('efam').agg(nch=('key', 'size'), snr=('snr', 'median')).to_string())


# ------------------------------------------------------------------------------------------------ scan
_T = {}


def load_T():
    if not _T:
        T = pd.read_csv(f'{MF}/templates.csv')
        T = T[T.use].reset_index(drop=True)
        Z = np.load(f'{MF}/templates.npz')
        _T['T'] = T
        _T['arr'] = {k: Z[k].astype(float) for k in T.key}
    return _T['T'], _T['arr']


MF_FAMS = [1, 2, 3, 4, 9, 10, 12, 13, 18, 22, 23]     # largest of the ~20 km W cluster + every family within 15 km
MF_STAS = ['PB.B047', 'PB.B049', 'PB.B932', 'PB.B933', 'PB.B934', 'NC.KMR', 'BK.BRIC', 'BK.SPRL', 'BK.JEWT']
LAGTOL = 6                                             # per-channel lag tolerance (samples, +-0.3 s): running max
NFFT = 1728720                                        # 5-smooth >= 86400*20 + 80
_TF = {}


def scan_day(day):
    """FFT-reused normalised CC; per channel running max over +-LAGTOL (family stacks are spatially extended)."""
    import scipy.fft as sfft
    from scipy.ndimage import maximum_filter1d
    T, arr = load_T()
    T = T[T.efam.isin(MF_FAMS) & T.sta.isin(MF_STAS)]
    day = pd.Timestamp(day)
    nday = int(86400 * SR)
    data = {}
    for sta in T.sta.unique():
        D = read_day(sta, day)
        for c, v in D.items():
            y, m = v
            y = y.astype(np.float32)
            Y = sfft.rfft(y, NFFT)
            data[(sta, c)] = (y, m, Y, {})
    S, NCH, AMP = {}, {}, {}
    for E in sorted(T.efam.unique()):
        tE = T[T.efam == E]
        acc = np.zeros(nday)
        cnt = np.zeros(nday)
        amps = []
        stacc = {}
        for r in tE.itertuples():
            if (r.sta, r.comp) not in data:
                continue
            y, m, Y, cache = data[(r.sta, r.comp)]
            tm = arr[r.key]
            n = len(tm)
            if r.key not in _TF:
                t = tm - tm.mean()
                t = t / (np.linalg.norm(t) + 1e-20)
                _TF[r.key] = np.conj(sfft.rfft(t.astype(np.float32), NFFT))
            num = sfft.irfft(Y * _TF[r.key], NFFT)[:nday - n + 1]
            if n not in cache:
                yd = y.astype(float)
                c1 = np.concatenate([[0.0], np.cumsum(yd)])
                c2 = np.concatenate([[0.0], np.cumsum(yd * yd)])
                cm = np.concatenate([[0], np.cumsum(~m)])
                var = (c2[n:] - c2[:-n]) - (c1[n:] - c1[:-n]) ** 2 / n
                bad = (cm[n:] - cm[:-n]) > 0
                inv = np.where((var > 1e-6 * n) & ~bad, 1.0 / np.sqrt(np.clip(var, 1e-20, None)), 0.0)
                cache[n] = (inv.astype(np.float32), bad)
            inv, bad = cache[n]
            v = num * inv
            ok = ~bad
            sh = int(round(r.off * SR))
            j0 = -sh
            a_, b_ = max(j0, 0), min(j0 + len(v), nday)
            v = maximum_filter1d(v, 2 * LAGTOL + 1)    # per-channel lag tolerance
            acc[a_:b_] += v[a_ - j0:b_ - j0]
            cnt[a_:b_] += ok[a_ - j0:b_ - j0]
            amps.append((r.key, sh, n, np.linalg.norm(tm - tm.mean()) / np.sqrt(n), y))
        if not amps:
            continue
        S[E] = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        NCH[E] = cnt
        AMP[E] = amps
    rows = []
    meta = dict(day=str(day.date()), nchan=len(data))
    for E, s in S.items():
        ok = NCH[E] >= max(NCHMIN, 0.6 * NCH[E].max())
        if ok.sum() < SR * 3600:
            continue
        med = np.median(s[ok])
        mad = np.median(np.abs(s[ok] - med))
        thr = max(med + K * mad, CCMIN)
        meta[f'mad_E{E}'] = mad
        meta[f'thr_E{E}'] = thr
        pk, _ = find_peaks(np.where(ok, s, 0), height=thr, distance=int(3 * SR))
        for j in pk:
            la = []
            for key, sh, n, trms, y in AMP[E]:
                a_ = j + sh
                if 0 <= a_ and a_ + n <= nday:
                    w = y[a_:a_ + n]
                    if np.any(w):
                        la.append(np.log10(np.sqrt(np.mean(w.astype(float) ** 2)) / (trms + 1e-20) + 1e-20))
            rows.append(dict(efam=E, t=day.value / 1e9 + j / SR, cc=float(s[j]), nch=int(NCH[E][j]),
                             thr=thr, mad=mad, lamp=float(np.median(la)) if la else np.nan))
    return rows, meta


def stage_scan(a):
    os.makedirs(MF, exist_ok=True)
    for yr in a.years:
        f = f'{MF}/det_{yr}.csv'
        if os.path.exists(f):
            print(yr, 'done', flush=True)
            continue
        days = pd.date_range(f'{yr}-01-01', f'{yr}-12-31', freq='D')
        days = days[days <= pd.Timestamp('2026-11-15')]
        t0 = time.time()
        with Pool(a.procs) as p:
            res = p.map(scan_day, days, chunksize=2)
        rows = [r for rr, _ in res for r in rr]
        d = pd.DataFrame(rows, columns=['efam', 't', 'cc', 'nch', 'thr', 'mad', 'lamp'])
        # cross-family dedup: keep the family with the largest CC/thr within +-2 s
        if len(d):
            d['score'] = d.cc / d.thr
            d = d.sort_values('t').reset_index(drop=True)
            keep = np.ones(len(d), bool)
            tt = d.t.values
            sc = d.score.values
            for i in range(len(d)):
                j = np.where(np.abs(tt - tt[i]) <= 2.0)[0]
                if sc[i] < sc[j].max():
                    keep[i] = False
            d = d[keep]
        d.to_csv(f, index=False)
        pd.DataFrame([m for _, m in res]).to_csv(f'{MF}/days_{yr}.csv', index=False)
        print(yr, len(d), 'detections', f'{time.time() - t0:.0f} s', flush=True)


def stage_eval(a):
    """detections vs catalog: recall, magnitude calibration, effective stack gain (sum of amplitude^2)."""
    import glob
    import json
    fs = sorted(glob.glob(f'{MF}/det_*.csv'))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    yrs = sorted({int(os.path.basename(f)[4:8]) for f in fs})
    c = pd.read_csv(f'{OUT}/families.csv')
    c['te'] = (pd.to_datetime(c.Time) - pd.Timestamp('1970-01-01')).dt.total_seconds()
    c['yr'] = pd.to_datetime(c.Time).dt.year
    cy = c[c.yr.isin(yrs)]
    # catalog match: any catalog event (selected volume) within +-3 s
    ct = np.sort(cy.te.values)
    k = np.searchsorted(ct, d.t.values)
    dt = np.minimum(np.abs(d.t.values - ct[np.clip(k - 1, 0, len(ct) - 1)]), np.abs(ct[np.clip(k, 0, len(ct) - 1)] - d.t.values))
    d['cat'] = dt <= 3.0
    fam = cy[cy.efam.isin(MF_FAMS)]
    ft = fam.te.values
    rec = np.array([np.any(np.abs(d.t.values - x) <= 3.0) for x in ft])
    # magnitude calibration on matched detections
    m = []
    for r in d[d.cat].itertuples():
        j = np.argmin(np.abs(cy.te.values - r.t))
        m.append((r.lamp, cy.Magnitude.values[j]))
    m = np.array(m)
    off = float(np.median(m[:, 1] - m[:, 0])) if len(m) else np.nan
    d['Mest'] = d.lamp + off
    out = {'years': yrs, 'n_det': int(len(d)), 'n_det_q12': int((d.cc / d.thr >= 1.2).sum()),
           'n_catalog_matched': int(d.cat.sum()), 'catalog_family_events': int(len(ft)),
           'recall_family_catalog': float(rec.mean()) if len(rec) else np.nan, 'M_offset': off,
           'M_slope_check': float(np.polyfit(m[:, 0], m[:, 1], 1)[0]) if len(m) > 3 else np.nan,
           'Mest_median_new': float(d[~d.cat].Mest.median())}
    gain = {}
    for lab, sel in (('all', np.ones(len(d), bool)), ('cc/thr>=1.2', (d.cc / d.thr >= 1.2).values)):
        a2 = 10 ** (2 * d.lamp.values)
        g_all = np.nansum(a2[sel])
        g_cat = np.nansum(a2[sel & d.cat.values])
        gain[lab] = float(g_all / g_cat) if g_cat > 0 else np.nan
        out[f'n_new_{lab}'] = int((sel & ~d.cat.values).sum())
    out['gain'] = gain
    # per family-year counts
    d['yr'] = pd.to_datetime(d.t, unit='s').dt.year
    out['per_family_year'] = d.groupby(['efam', 'yr']).size().unstack(fill_value=0).to_dict()
    json.dump(out, open(f'{MF}/eval.json', 'w'), indent=1, default=str)
    d.to_csv(f'{MF}/detections_eval.csv', index=False)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_family_year'}, indent=1, default=str))
    print(d.groupby(['efam', 'yr']).size().unstack(fill_value=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--years', type=int, nargs='*', default=list(range(2009, 2027)))
    a = ap.parse_args()
    {'tmpl': stage_tmpl, 'scan': stage_scan, 'eval': stage_eval}[a.stage](a)


if __name__ == '__main__':
    main()
