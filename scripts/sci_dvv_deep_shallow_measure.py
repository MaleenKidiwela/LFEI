#!/usr/bin/env python3
"""Deep/shallow dv/v separation -- MEASUREMENT stage (one pass over the family waveform files).

For every family F and channel (template SNR >= 15, S pick in data/locations/fam{F}_3d_picks.csv):
  stacks: isolated LFEs (no other v2 detection within +-5 s), consecutive stacks of N=100
          (N=50 within 182 d after each mainshock), never straddling a mainshock (dvv_cwire.make_stacks).
  reference = mean of all isolated events.  2-8 Hz, traces std-normalised (as dvv_cwire), 4x upsampled (200 Hz).
  Per stack vs reference:
   * direct S delay dtS: CC lag in [S-0.5, S+2.5] s (+-0.1 s, parabolic sub-sample); positive = stack LATE.
   * direct P delay dtP: CC lag in [P-0.3, P+1.5] s (if a P pick exists).
   * coda stretching about S AFTER aligning the stack on direct S (as dvv_cwire) in fixed lapse windows
     tau = t - S in [3, 8], [8, 15], [15, 25] s and the classic [3, tau_end] window (tau_end where the
     reference envelope drops below 2x noise).  eps = dv/v(stack) - dv/v(ref), positive = faster.
   * joint shift+stretch in each fixed window: grid eps x lag (lag +-60 ms), lag after the S alignment.
   * bootstrap (B=20 member resamples) standard errors of dtS, dtP and every stretch-only eps.
   * window SNR = rms(ref in window)/rms(ref noise -15..-5 s).
Outputs data/sci/dvv_deep_shallow/meas/fam{F}.parquet (+ fam{F}_channels.csv).
Usage: nice -n 10 python scripts/sci_dvv_deep_shallow_measure.py --procs 10 [--fams 23 8 ...]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import epoch_s, make_stacks  # noqa: E402

ROOT = '/home/seismic/LFE'
SONLY = os.environ.get('DS_SONLY', '0') == '1'      # only direct-S sub-windows (onset / late) -> meas_swin/
OUT = f'{ROOT}/data/sci/dvv_deep_shallow/' + ('meas_swin' if SONLY else 'meas')
UP = 4
EPS = np.round(np.arange(-0.01, 0.01 + 1e-9, 0.0002), 6)
WINS = {'w3_8': (3.0, 8.0), 'w8_15': (8.0, 15.0), 'w15_25': (15.0, 25.0)}
NB = 20
LAG2 = 0.10          # s, joint shift+stretch lag range
FMIN, FMAX = 2.0, 8.0


def subpk(y, k):
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        f = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(np.clip(f, -0.5, 0.5)), float(y1 - 0.25 * (y0 - y2) * f)
    return 0.0, float(y[k])


def lag_cc(refw, X, i0, maxlag):
    """refw: (n,) ref window (demeaned); X: (m, N) traces; i0: window indices; returns lag (samples, float), cc."""
    r = refw / (np.linalg.norm(refw) + 1e-20)
    L = np.arange(-maxlag, maxlag + 1)
    C = np.empty((X.shape[0], len(L)))
    for j, l in enumerate(L):
        seg = X[:, i0 + l]
        seg = seg - seg.mean(1, keepdims=True)
        C[:, j] = seg @ r / (np.linalg.norm(seg, axis=1) + 1e-20)
    k = np.argmax(C, axis=1)
    lag, cc = np.empty(len(k)), np.empty(len(k))
    for i in range(len(k)):
        f, c = subpk(C[i], k[i])
        lag[i], cc[i] = L[k[i]] + f, c
        if k[i] in (0, len(L) - 1):
            cc[i] = -1.0
    return lag, cc


def channel_job(args):
    (F, h5f, cid, S, P, stacks, stack_t, iso_idx, seed) = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    with h5py.File(h5f, 'r') as h:
        sr, pre = float(h.attrs['sr']), float(h.attrs['pre'])
        ev = h[f'chan/{cid}/ev'][:]
        X = h[f'chan/{cid}/data'][:].astype('float64')
    pos = {e: k for k, e in enumerate(ev)}
    sos = butter(4, [FMIN, FMAX], btype='band', fs=sr, output='sos')
    X = sosfiltfilt(sos, X, axis=1)
    X /= (np.std(X, axis=1) + 1e-20)[:, None]
    t = np.arange(X.shape[1]) / sr + pre
    rk = [pos[e] for e in iso_idx if e in pos]
    if len(rk) < 100:
        return None, dict(fam=F)
    ref = X[rk].mean(0)
    env = np.convolve(np.abs(hilbert(ref)), np.ones(int(sr)) / sr, mode='same')
    noise = np.sqrt(np.mean(ref[(t > -15) & (t < -5)] ** 2))
    after = np.where((t > S + 3.0) & (env < 2.0 * noise))[0]
    t_end = min(t[after[0]] if len(after) else t[-1] - 1, t[-1] - 1.0)
    srU = sr * UP
    tU = np.arange(X.shape[1] * UP) / srU + pre
    up = lambda x: resample_poly(x, UP, 1, axis=-1)  # noqa: E731
    refU = up(ref)
    tau = tU - S
    # windows
    sw = np.where((tU >= S - 0.5) & (tU <= S + 2.5))[0]
    mlS = int(0.1 * srU)
    havP = np.isfinite(P) and (P - 0.3 - 0.1) > pre + 1
    pw = np.where((tU >= P - 0.3) & (tU <= P + 1.5))[0] if havP else None
    wins = dict(WINS)
    wins['full'] = (3.0, float(t_end - S))
    tmax = tU[-1] - 0.2 - LAG2
    wmask, wsnr = {}, {}
    for k, (a, b) in wins.items():
        b2 = min(b, (tmax - S) / 1.0101)
        if b2 - a < 2.0:
            continue
        m = (tau >= a) & (tau <= b2)
        wmask[k] = m
        wsnr[k] = float(np.sqrt(np.mean(refU[m] ** 2)) / (noise + 1e-20))
    # stretched reference matrices (per window), demeaned + normalised
    Mref = {}
    for k, m in wmask.items():
        tc = tau[m]
        M = np.array([np.interp(tc * (1 + e), tau, refU) for e in EPS])
        M -= M.mean(1, keepdims=True)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-20
        Mref[k] = M
    ml2 = int(LAG2 * srU)
    rows = []
    for b, mem in enumerate(stacks):
        kk = np.array([pos[e] for e in mem if e in pos])
        if len(kk) < max(5, len(mem) // 2):
            continue
        Xm = X[kk]
        n = len(kk)
        W = np.vstack([np.full(n, 1.0 / n)] + [np.bincount(rng.integers(0, n, n), minlength=n) / n for _ in range(NB)])
        Sk = up(W @ Xm)                                        # (1+NB, NU)
        lagS, ccS = lag_cc(refU[sw] - refU[sw].mean(), Sk, sw, mlS)
        r = dict(fam=F, chan=cid, stack=b, t=stack_t[b], n=n,
                 dtS=lagS[0] / srU, ccS=ccS[0], dtS_se=np.std(lagS[1:]) / srU)
        if havP:
            lagP, ccP = lag_cc(refU[pw] - refU[pw].mean(), Sk, pw, mlS)
            r.update(dtP=lagP[0] / srU, ccP=ccP[0], dtP_se=np.std(lagP[1:]) / srU)
        else:
            r.update(dtP=np.nan, ccP=np.nan, dtP_se=np.nan)
        # direct-S sub-windows: onset [S-0.4, S+0.4] (before any borehole surface reflection) and [S+0.4, S+2.5]
        for nm, (a_, b_) in (('Son', (-0.4, 0.4)), ('Slate', (0.4, 2.5))):
            wi = np.where((tU >= S + a_) & (tU <= S + b_))[0]
            lg, cg = lag_cc(refU[wi] - refU[wi].mean(), Sk, wi, mlS)
            r.update({f'dt{nm}': lg[0] / srU, f'cc{nm}': cg[0], f'dt{nm}_se': np.std(lg[1:]) / srU})
        if SONLY:
            rows.append(r)
            continue
        # align every (bootstrap) stack on its own direct-S lag
        Al = np.array([np.interp(tU + lagS[i] / srU, tU, Sk[i]) for i in range(Sk.shape[0])])
        for k, m in wmask.items():
            B = Al[:, m]
            B = B - B.mean(1, keepdims=True)
            B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-20
            C = Mref[k] @ B.T                                  # (neps, 1+NB)
            es = []
            for i in range(C.shape[1]):
                j = int(np.argmax(C[:, i]))
                f, c = subpk(C[:, i], j)
                es.append(EPS[j] + f * (EPS[1] - EPS[0]) if 0 < j < len(EPS) - 1 else np.nan)
                if i == 0:
                    r[f'cc_{k}'] = c if 0 < j < len(EPS) - 1 else -1.0
            es = np.array(es)
            r[f'eps_{k}'] = es[0]
            r[f'eps_{k}_se'] = np.nanstd(es[1:]) if np.isfinite(es[1:]).sum() >= NB // 2 else np.nan
            if k == 'full':
                continue
            # joint shift + stretch on the main stack
            idx = np.where(m)[0]
            Cl = np.empty((len(EPS), 2 * ml2 + 1))
            for jl, l in enumerate(range(-ml2, ml2 + 1)):
                seg = Al[0, idx + l]
                seg = seg - seg.mean()
                Cl[:, jl] = Mref[k] @ seg / (np.linalg.norm(seg) + 1e-20)
            je, jl = np.unravel_index(np.argmax(Cl), Cl.shape)
            fe, _ = subpk(Cl[:, jl], je)
            fl, c2 = subpk(Cl[je, :], jl)
            ok = 0 < je < len(EPS) - 1 and 0 < jl < 2 * ml2
            r[f'eps2_{k}'] = EPS[je] + fe * (EPS[1] - EPS[0]) if ok else np.nan
            r[f'lag2_{k}'] = (jl - ml2 + fl) / srU if ok else np.nan
            r[f'cc2_{k}'] = c2 if ok else -1.0
        rows.append(r)
    if len(rows) < 6:
        return None, dict(fam=F)
    meta = dict(fam=F, chan=cid, S=S, P=P, t_end=float(t_end), n_stacks=len(rows), secs=time.time() - t0,
                **{f'snr_{k}': v for k, v in wsnr.items()})
    return pd.DataFrame(rows), meta


def family_jobs(F, a):
    h5f = f'{ROOT}/data/waveforms/fam{F}.h5'
    with h5py.File(h5f, 'r') as h:
        t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    te = epoch_s(t_ev)
    cat = pd.read_csv(f'{ROOT}/data/merged/LFEcat_MTJ_merged_v1v2.csv', usecols=['time_utc', 'source'])
    cat = cat[cat.source != 'v1_only']
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
    idx = np.where(iso)[0]
    idx = idx[np.argsort(te[idx])]
    stacks = make_stacks(te, idx, a.n, a.n_post)
    stack_t = np.array([te[m].mean() for m in stacks])
    T = h5py.File(f'{ROOT}/data/templates/fam{F}_templates.h5', 'r')
    picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    jobs = []
    for i, cid in enumerate(sorted(T)):
        sta = '.'.join(cid.split('.')[:2])
        if T[cid].attrs['snr'] < 15.0 or sta not in picks.index or not np.isfinite(picks.loc[sta, 'S_final']):
            continue
        P = float(picks.loc[sta, 'P']) if 'P' in picks.columns and np.isfinite(picks.loc[sta, 'P']) else np.nan
        jobs.append((F, h5f, cid, float(picks.loc[sta, 'S_final']), P, stacks, stack_t, idx, 1000 * F + i))
    T.close()
    stk = pd.DataFrame(dict(stack=range(len(stacks)), t=stack_t, n=[len(m) for m in stacks]))
    return jobs, stk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fams', type=int, nargs='*', default=list(range(1, 28)))
    ap.add_argument('--procs', type=int, default=10)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--n-post', type=int, default=50)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    alljobs, stks = [], {}
    for F in a.fams:
        if os.path.exists(f'{OUT}/fam{F}.parquet'):
            print(f'fam {F}: done, skip', flush=True)
            continue
        if not os.path.exists(f'{ROOT}/data/waveforms/fam{F}.h5'):
            continue
        j, s = family_jobs(F, a)
        print(f'fam {F}: {len(j)} channels, {len(s)} stacks', flush=True)
        alljobs += j
        stks[F] = s
    res = {F: [] for F in stks}
    metas = {F: [] for F in stks}
    left = {F: sum(1 for j in alljobs if j[0] == F) for F in stks}
    t0 = time.time()
    with Pool(a.procs) as p:
        for k, r in enumerate(p.imap_unordered(channel_job, alljobs, chunksize=1)):
            F = r[1]['fam']
            if r[0] is not None:
                res[F].append(r[0])
                metas[F].append(r[1])
            left[F] -= 1
            done_fams = []
            for F in list(left):
                if left[F] == 0 and res[F]:
                    pd.concat(res[F], ignore_index=True).to_parquet(f'{OUT}/fam{F}.parquet')
                    pd.DataFrame(metas[F]).to_csv(f'{OUT}/fam{F}_channels.csv', index=False)
                    stks[F].to_csv(f'{OUT}/fam{F}_stacks.csv', index=False)
                    done_fams.append(F)
            for F in done_fams:
                del left[F]
                res[F], metas[F] = [], []
                print(f'== fam {F} written ({time.time() - t0:.0f} s)', flush=True)
            if k % 50 == 0:
                print(f'{k}/{len(alljobs)} jobs, {time.time() - t0:.0f} s', flush=True)
    for F in list(left):                       # families with None results
        if res[F]:
            pd.concat(res[F], ignore_index=True).to_parquet(f'{OUT}/fam{F}.parquet')
            pd.DataFrame(metas[F]).to_csv(f'{OUT}/fam{F}_channels.csv', index=False)
            stks[F].to_csv(f'{OUT}/fam{F}_stacks.csv', index=False)
            print(f'== fam {F} written at end', flush=True)


if __name__ == '__main__':
    main()
