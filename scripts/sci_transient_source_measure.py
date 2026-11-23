#!/usr/bin/env python3
"""Waveform measurements for the early-coda transient vs source-location test.

Same machinery as dvv_cwire / sci_dvv_depth_measure (2-8 Hz, traces normalised, 4x upsampling,
every stack S-aligned to the all-isolated reference stack by CC in [S-0.5, S+2.5] s, stretching
about S on the EPS grid with parabolic refinement), applied to two kinds of stacks:

 (2) location-selected pre-M7 stacks: isolated, non-overlapping (no other-family LFE within
     +-10 s), located events before the M7, in time blocks of BLOCK consecutive events; inside
     every block the events are split at the block median of one fault-frame coordinate
     (time-matched halves).  eps(Hi vs Lo) = dv/v(Hi) - dv/v(Lo) per lapse window, plus the
     direct-S lag difference dS.  Selection on the full-network location ('F*') or on the A-half
     location ('A*', true separation then measured with the independent B half).  Null: random
     splits inside the same blocks.
 (3) dv/v pair measurements (b28_early tau 3-8 s, b28_late tau 8 s-end) for re-built stacks:
     'loc'   = stacks (50 / 25 post-M7) of isolated non-overlapping located events (unweighted)
     'match' = same stacks, events in the first 182 d after the M7 weighted so that each post-M7
               sub-period has the (s, d) location histogram of the reference events
Outputs (data/sci/transient_source/): fam{F}_split_eps.parquet, fam{F}_pairs_{variant}.parquet,
fam{F}_channels_meas.csv
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import CODA_START, EPS, MAXSEP, SNR_END, UP, subsample_peak  # noqa: E402
from sci_transient_source_common import OUT  # noqa: E402

SPLIT_WINDOWS = {'early': (3.0, 8.0), 'late': (8.0, None), 'w3_5': (3.0, 5.0), 'w5_8': (5.0, 8.0),
                 'w8_11': (8.0, 11.0), 'w11_16': (11.0, 16.0)}
PAIR_WINDOWS = {'b28_early': (3.0, 8.0), 'b28_late': (8.0, None)}
MINLEN = {'early': 2.5, 'late': 2.5, 'b28_early': 2.5, 'b28_late': 2.5}


def best(C):
    k = np.argmax(C, axis=0)
    e, c = np.empty(C.shape[1]), np.empty(C.shape[1])
    for j in range(C.shape[1]):
        f, c[j] = subsample_peak(C[:, j], k[j])
        e[j] = EPS[k[j]] + f * (EPS[1] - EPS[0])
        if k[j] in (0, len(EPS) - 1):
            c[j] = -1.0
    return e, c


def channel_job(args):
    h5f, cid, S, iso_idx, splits, variants = args
    with h5py.File(h5f, 'r') as h:
        sr, pre = h.attrs['sr'], h.attrs['pre']
        ev = h[f'chan/{cid}/ev'][:]
        X = h[f'chan/{cid}/data'][:].astype('float64')
    pos = np.full(ev.max() + 2 if len(ev) else 1, -1)
    pos[ev] = np.arange(len(ev))
    getpos = lambda m: pos[m[m < len(pos)]][pos[m[m < len(pos)]] >= 0]  # noqa: E731
    sos = butter(4, [2.0, 8.0], btype='band', fs=sr, output='sos')
    X = sosfiltfilt(sos, X, axis=1)
    X /= (np.std(X, axis=1) + 1e-20)[:, None]
    t = np.arange(X.shape[1]) / sr + pre
    rk = getpos(np.asarray(iso_idx))
    if len(rk) < 100:
        return None
    ref = X[rk].mean(0)
    env = np.convolve(np.abs(hilbert(ref)), np.ones(int(sr)) / sr, mode='same')
    noise = np.sqrt(np.mean(ref[(t > -15) & (t < -5)] ** 2))
    after = np.where((t > S + CODA_START) & (env < SNR_END * noise))[0]
    t_end = min(t[after[0]] if len(after) else t[-1] - 1, t[-1] - 1.0)
    if t_end - (S + CODA_START) < 3.0:
        return None
    tau_end = t_end - S
    srU = sr * UP
    tU = np.arange(X.shape[1] * UP) / srU + pre
    tau = tU - S
    up = lambda x: resample_poly(x, UP, 1)  # noqa: E731
    refU = up(ref)
    sw = (tU >= S - 0.5) & (tU <= S + 2.5)
    maxlag = int(0.1 * srU)
    i0 = np.where(sw)[0]
    r0 = refU[sw] - refU[sw].mean()

    def aligned(kk, w=None):
        x = X[kk].mean(0) if w is None else (w[:, None] * X[kk]).sum(0) / w.sum()
        xU = up(x)
        ccs = []
        for L in range(-maxlag, maxlag + 1):
            c = xU[i0 + L]
            c = c - c.mean()
            ccs.append(r0 @ c / (np.linalg.norm(r0) * np.linalg.norm(c) + 1e-20))
        ccs = np.array(ccs)
        k = int(np.argmax(ccs))
        f, _ = subsample_peak(ccs, k)
        lag = (k - maxlag + f) / srU
        return np.interp(tU + lag, tU, xU), lag

    def win(w0, w1):
        hi = tau_end if w1 is None else min(w1, tau_end)
        return (tau >= w0) & (tau <= hi), hi - w0

    def stretched(xU, tauc):
        M = np.array([np.interp(tauc * (1 + e), tau, xU) for e in EPS])
        M -= M.mean(1, keepdims=True)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-20
        return M

    def norm(v):
        v = v - v.mean()
        return v / (np.linalg.norm(v) + 1e-20)

    meta = dict(chan=cid, tau_end=float(tau_end), S=S)
    # ---------------- (2) location splits
    srows = []
    wins = {k: win(*v) for k, v in SPLIT_WINDOWS.items()}
    for name, hi_m, lo_m in splits:
        kh, kl = getpos(hi_m), getpos(lo_m)
        if len(kh) < 30 or len(kl) < 30:
            continue
        xh, lh = aligned(kh)
        xl, ll = aligned(kl)
        for wn, (m, L) in wins.items():
            if L < MINLEN.get(wn, 1.9):
                continue
            e, c = best(stretched(xl, tau[m]) @ norm(xh[m])[:, None])
            srows.append((cid, name, wn, e[0], c[0], lh - ll, len(kh), len(kl), L))
    SP = pd.DataFrame(srows, columns=['chan', 'split', 'window', 'eps', 'cc', 'dS', 'n_hi', 'n_lo', 'win_len'])
    # ---------------- (3) pair measurements for re-built stacks
    prows = []
    for vname, (stacks, weights, stack_t) in variants.items():
        keep, SL = [], []
        for b, (mem, w) in enumerate(zip(stacks, weights)):
            mem = np.asarray(mem)
            ok = (mem < len(pos))
            ok[ok] = pos[mem[ok]] >= 0
            if ok.sum() < max(5, len(mem) // 2):
                continue
            kk = pos[mem[ok]]
            ww = None if w is None else np.asarray(w)[ok]
            keep.append(b)
            SL.append(aligned(kk, ww)[0])
        if len(keep) < 6:
            continue
        tk = np.array([stack_t[b] for b in keep])
        for cfg, (w0, w1) in PAIR_WINDOWS.items():
            m, L = win(w0, w1)
            if L < MINLEN[cfg]:
                continue
            tauc = tau[m]
            B = np.array([norm(x[m]) for x in SL])
            e, c = best(stretched(refU, tauc) @ B.T)
            meta[f'{vname}_{cfg}_ccref'] = float(np.median(c))
            for j, b in enumerate(keep):
                prows.append((cid, vname, cfg, -1, b, e[j], c[j]))
            for i, b in enumerate(keep):
                J = np.where((tk > tk[i]) & (tk - tk[i] <= MAXSEP * 86400))[0]
                if len(J) == 0:
                    continue
                e, c = best(stretched(SL[i], tauc) @ B[J].T)
                for jj, j in enumerate(J):
                    prows.append((cid, vname, cfg, b, keep[j], e[jj], c[jj]))
    PR = pd.DataFrame(prows, columns=['chan', 'variant', 'cfg', 'a', 'b', 'eps', 'cc'])
    return SP, PR, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--only', default=None)
    ap.add_argument('--prefix', default='', help='output/jobdef file prefix (e.g. quality_)')
    ap.add_argument('--tag', default='', help='jobdef / output tag (e.g. _qc for the event-quality control splits)')
    a = ap.parse_args()
    F = a.family
    # own intermediate file written by sci_transient_source.py --prep in this project (trusted)
    job = pickle.load(open(f'{OUT}/{a.prefix}fam{F}_jobdef{a.tag}.pkl', 'rb'))     # written by sci_transient_source.py --prep
    chans = pd.read_csv(f'data/sci/dvv_depth/fam{F}_channels.csv').chan.tolist()
    if a.only:
        chans = [c for c in chans if c in a.only.split(',')]
    picks = pd.read_csv(f'data/locations/fam{F}_3d_picks.csv').set_index('sta')
    jobs = [(f'data/waveforms/fam{F}.h5', c, float(picks.loc['.'.join(c.split('.')[:2]), 'S_final']),
             job['iso_idx'], job['splits'], job['variants']) for c in chans]
    print(f'family {F}: {len(jobs)} channels, {len(job["splits"])} splits, variants {list(job["variants"])}', flush=True)
    sps, prs, metas = [], [], []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(channel_job, jobs):
            if r is None:
                continue
            sps.append(r[0])
            prs.append(r[1])
            metas.append(r[2])
            print('  done', r[2]['chan'], len(r[0]), len(r[1]), flush=True)
    sfx = a.tag + ('' if not a.only else '_test')
    pd.concat(sps, ignore_index=True).to_parquet(f'{OUT}/{a.prefix}fam{F}_split_eps{sfx}.parquet')
    PR = pd.concat(prs, ignore_index=True)
    for v in PR.variant.unique() if len(PR) else []:
        PR[PR.variant == v].reset_index(drop=True).to_parquet(f'{OUT}/{a.prefix}fam{F}_pairs_{v}{sfx}.parquet')
    pd.DataFrame(metas).to_csv(f'{OUT}/{a.prefix}fam{F}_channels_meas{sfx}.csv', index=False)


if __name__ == '__main__':
    main()
