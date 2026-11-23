#!/usr/bin/env python3
"""Lapse-time and frequency split of the CWIRE-style LFE-coda dv/v measurement (dvv_cwire.py),
for depth partitioning. Same stacks, same S alignment, same stretching about S, same pair set as
dvv_cwire.channel_job, but several measurement configurations from ONE read of each channel:

  b28_full  : 2-8 Hz, tau = t - S in [3 s, tau_end]          (reproduces dvv_cwire; sanity check)
  b28_early : 2-8 Hz, tau in [3, 8] s
  b28_late  : 2-8 Hz, tau in [8, tau_end]   (needs tau_end - 8 >= MINLEN)
  b24_full  : 2-4 Hz, tau in [3, tau_end]
  b48_full  : 4-8 Hz, tau in [3, tau_end]
tau_end is ALWAYS taken from the 2-8 Hz reference envelope (< 2x noise), and the S-alignment lag of
every stack is ALWAYS measured on the 2-8 Hz stack and applied to all bands, so the frequency
configurations differ only in pass band and the lapse-time ones only in window.
Only channels that passed the original QC (median reference CC >= 0.4 in fam{F}_cwire_channels.csv)
are measured.  --inject DATE DVV applies a uniform synthetic dv/v (stretch about S) after DATE.
Output: data/sci/dvv_depth/fam{F}_pairs{tag}.parquet (chan, cfg, a, b, eps, cc), _channels{tag}.csv
"""
import argparse
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import CODA_START, EPS, MAXSEP, SNR_END, UP, epoch_s, make_stacks, subsample_peak  # noqa: E402

CONFIGS = [('b28_full', (2, 8), 3.0, None), ('b28_early', (2, 8), 3.0, 8.0), ('b28_late', (2, 8), 8.0, None),
           ('b24_full', (2, 4), 3.0, None), ('b48_full', (4, 8), 3.0, None)]
MINLEN = 2.5
OUT = 'data/sci/dvv_depth'


def channel_job(args):
    h5f, cid, S, stacks, stack_t, iso_idx, inject = args
    with h5py.File(h5f, 'r') as h:
        sr, pre = h.attrs['sr'], h.attrs['pre']
        ev = h[f'chan/{cid}/ev'][:]
        X0 = h[f'chan/{cid}/data'][:].astype('float64')
    pos = {e: k for k, e in enumerate(ev)}
    t = np.arange(X0.shape[1]) / sr + pre
    rk = [pos[e] for e in iso_idx if e in pos]
    if len(rk) < 100:
        return None
    srU = sr * UP
    tU = np.arange(X0.shape[1] * UP) / srU + pre
    tau = tU - S
    up = lambda x: resample_poly(x, UP, 1)  # noqa: E731

    def band(fmin, fmax):
        sos = butter(4, [fmin, fmax], btype='band', fs=sr, output='sos')
        X = sosfiltfilt(sos, X0, axis=1)
        X /= (np.std(X, axis=1) + 1e-20)[:, None]
        return X

    Xb = {(2, 8): band(2, 8)}
    ref = Xb[(2, 8)][rk].mean(0)
    env = np.convolve(np.abs(hilbert(ref)), np.ones(int(sr)) / sr, mode='same')
    noise = np.sqrt(np.mean(ref[(t > -15) & (t < -5)] ** 2))
    after = np.where((t > S + CODA_START) & (env < SNR_END * noise))[0]
    t_end = min(t[after[0]] if len(after) else t[-1] - 1, t[-1] - 1.0)
    if t_end - (S + CODA_START) < 3.0:
        return None
    tau_end = t_end - S
    # --- S alignment on 2-8 Hz (identical to dvv_cwire)
    refU = up(ref)
    sw = (tU >= S - 0.5) & (tU <= S + 2.5)
    maxlag = int(0.1 * srU)
    i0 = np.where(sw)[0]
    r = refU[sw] - refU[sw].mean()

    def lag_of(xU):
        ccs = []
        for L in range(-maxlag, maxlag + 1):
            c = xU[i0 + L]
            c = c - c.mean()
            ccs.append(r @ c / (np.linalg.norm(r) * np.linalg.norm(c) + 1e-20))
        ccs = np.array(ccs)
        k = int(np.argmax(ccs))
        f, _ = subsample_peak(ccs, k)
        return (k - maxlag + f) / srU

    keep, lags, members = [], [], []
    for b, mem in enumerate(stacks):
        kk = [pos[e] for e in mem if e in pos]
        if len(kk) < max(5, len(mem) // 2):
            continue
        keep.append(b)
        members.append(kk)
        lags.append(lag_of(up(Xb[(2, 8)][kk].mean(0))))
    if len(keep) < 6:
        return None
    tk = np.array([stack_t[b] for b in keep])
    rows, meta = [], dict(chan=cid, tau_end=float(tau_end), n_stacks=len(keep))
    for bnd in sorted({c[1] for c in CONFIGS}):
        if bnd not in Xb:
            Xb[bnd] = band(*bnd)
        X = Xb[bnd]
        refB = up(X[rk].mean(0))
        SL = []
        for j, kk in enumerate(members):
            xU = np.interp(tU + lags[j], tU, up(X[kk].mean(0)))
            if inject is not None and tk[j] >= inject[0]:
                xU = np.interp(tau * (1 + inject[1]), tau, xU)
            SL.append(xU)
        for name, bb, w0, w1 in CONFIGS:
            if bb != bnd:
                continue
            hi = tau_end if w1 is None else min(w1, tau_end)
            if hi - w0 < MINLEN:
                meta[f'{name}_len'] = float(max(hi - w0, 0))
                continue
            coda = (tau >= w0) & (tau <= hi)
            tauc = tau[coda]
            B = np.array([x[coda] for x in SL])
            B = B - B.mean(1, keepdims=True)
            B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-20

            def stretched(xU):
                M = np.array([np.interp(tauc * (1 + e), tau, xU) for e in EPS])
                M -= M.mean(1, keepdims=True)
                M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-20
                return M

            def best(C):
                k = np.argmax(C, axis=0)
                e, c = np.empty(C.shape[1]), np.empty(C.shape[1])
                for j in range(C.shape[1]):
                    f, c[j] = subsample_peak(C[:, j], k[j])
                    e[j] = EPS[k[j]] + f * (EPS[1] - EPS[0])
                    if k[j] in (0, len(EPS) - 1):
                        c[j] = -1.0
                return e, c
            e, c = best(stretched(refB) @ B.T)
            for j, b in enumerate(keep):
                rows.append((cid, name, -1, b, e[j], c[j]))
            ccr = np.median(c)
            for i, b in enumerate(keep):
                J = np.where((tk > tk[i]) & (tk - tk[i] <= MAXSEP * 86400))[0]
                if len(J) == 0:
                    continue
                e, c = best(stretched(SL[i]) @ B[J].T)
                for jj, j in enumerate(J):
                    rows.append((cid, name, b, keep[j], e[jj], c[jj]))
            meta[f'{name}_len'] = float(hi - w0)
            meta[f'{name}_ccref'] = float(ccr)
    P = pd.DataFrame(rows, columns=['chan', 'cfg', 'a', 'b', 'eps', 'cc'])
    P['cfg'] = P.cfg.astype('category')
    return P, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--min-ref-cc', type=float, default=0.4)
    ap.add_argument('--inject', nargs=2, default=None)
    ap.add_argument('--only', default=None, help='comma list of channels (testing)')
    a = ap.parse_args()
    F = a.family
    os.makedirs(OUT, exist_ok=True)
    tag = '' if a.inject is None else f'_inj{a.inject[0][:10]}_{a.inject[1]}'
    h5f = f'data/waveforms/fam{F}.h5'
    with h5py.File(h5f, 'r') as h:
        te = epoch_s(pd.to_datetime(h['events/time_utc'][:].astype(str)))
    cat = pd.read_csv(a.catalog)
    cat = cat[cat.source != 'v1_only']
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
    idx = np.where(iso)[0]
    idx = idx[np.argsort(te[idx])]
    stacks = make_stacks(te, idx, 50, 25)
    stack_t = np.array([te[m].mean() for m in stacks])
    old = pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv')
    assert len(old) == len(stacks), 'stack definition differs from dvv_cwire run'
    meta0 = pd.read_csv(f'data/dvv/fam{F}_cwire_channels.csv')
    chans = meta0[meta0.median_cc_ref >= a.min_ref_cc].chan.tolist()
    if a.only:
        chans = [c for c in chans if c in a.only.split(',')]
    picks = pd.read_csv(f'data/locations/fam{F}_3d_picks.csv').set_index('sta')
    inj = None if a.inject is None else (epoch_s([a.inject[0]])[0], float(a.inject[1]))
    jobs = [(h5f, c, float(picks.loc['.'.join(c.split('.')[:2]), 'S_final']), stacks, stack_t, idx, inj) for c in chans]
    print(f'family {F}: {len(stacks)} stacks, {len(jobs)} channels', flush=True)
    res, metas = [], []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(channel_job, jobs):
            if r is None:
                continue
            res.append(r[0])
            metas.append(r[1])
            m = r[1]
            print('  ' + m['chan'] + ' ' + ' '.join(f"{k}={v:.2f}" for k, v in m.items() if k.endswith(('_len', '_ccref', 'tau_end'))), flush=True)
    P = pd.concat(res, ignore_index=True)
    P['cfg'] = P.cfg.astype(str)
    suffix = '' if not a.only else '_test'
    P.to_parquet(f'{OUT}/fam{F}_pairs{tag}{suffix}.parquet')
    pd.DataFrame(metas).to_csv(f'{OUT}/fam{F}_channels{tag}{suffix}.csv', index=False)


if __name__ == '__main__':
    main()
