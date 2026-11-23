#!/usr/bin/env python3
"""Coda-wave dv/v through time from a family's repeats (stacked in time bins).

1. Events: v2 detections of the family that are ISOLATED (no detection of any family within
   +-ISO s, using the full v2 catalog).
2. Bins: consecutive equal-count bins of NBIN isolated events; bin edges forced at the
   Dec 2021 M6.1, Dec 2022 M6.4 and Dec 2024 M7.0 mainshocks (bins never straddle them).
3. Per channel: bin stacks (catalog-aligned, 2-8 Hz by default) and the full-period stack.
   Coda window: [S + CODA_START, S + L], L = where the master stack envelope falls below
   SNR_END x noise (capped by data end). Lapse time measured from the located origin time.
4. Stretching: eps grid; u_ref(t (1+eps)) vs u_bin(t) in the coda window; dv/v = eps
   (positive = faster). Also direct-S window constant-shift dt (clock/timing control).
5. All bin pairs (within MAXPAIR bins of each other) measured -> CWIRE-style least-squares
   inversion eps_ij = m_j - m_i (weights CC^2), mean-zero; plus vs-reference series.
6. Controls: (a) same pipeline on the pre-event noise window; (b) synthetic stretch of
   INJECT applied to the second half of bins must be recovered.
Outputs data/dvv/fam{F}_{chan}.csv and summary data/dvv/fam{F}_summary.csv; figure.
"""
import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

ISO = 15.0
CODA_START, SNR_END = 3.0, 2.0
EPS = np.arange(-0.01, 0.01 + 1e-9, 0.0002)           # +-1 %, 0.02 % step (refined after)
MAXPAIR = 12
INJECT = 0.001
QUAKES = ['2021-12-20T20:10', '2022-12-20T10:34', '2024-12-05T18:44']


def epoch_s(x):
    """Epoch seconds from datetime-like (any datetime64 unit)."""
    x = pd.DatetimeIndex(x)
    return np.asarray((x - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1), dtype=float)


def stretch_cc(ref, cur, t, win, eps_grid):
    """Best eps maximizing CC between ref(t(1+eps)) and cur(t) inside win (index mask)."""
    tw = t[win]
    c = cur[win] - cur[win].mean()
    cn = np.linalg.norm(c) + 1e-20
    best = (-2, 0.0)
    ccs = []
    for e in eps_grid:
        r = np.interp(tw * (1 + e), t, ref)
        r = r - r.mean()
        cc = float(r @ c / (np.linalg.norm(r) * cn + 1e-20))
        ccs.append(cc)
    ccs = np.array(ccs)
    k = int(np.argmax(ccs))
    if 0 < k < len(eps_grid) - 1:                       # parabolic refinement
        y0, y1, y2 = ccs[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return eps_grid[k] + frac * (eps_grid[1] - eps_grid[0]), y1
    return eps_grid[k], ccs[k]


def shift_dt(ref, cur, sr, win, maxlag):
    r, c = ref[win], cur[win]
    lags = np.arange(-maxlag, maxlag + 1)
    i0 = np.where(win)[0]
    ccs = []
    for L in lags:
        cs = cur[i0 + L] if (i0 + L).min() >= 0 and (i0 + L).max() < len(cur) else c
        ccs.append(np.corrcoef(r, cs)[0, 1])
    ccs = np.array(ccs)
    k = int(np.argmax(ccs))
    frac = 0.0
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = ccs[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return (lags[k] + frac) / sr, ccs[k]


def invert_pairs(nb, pairs):
    """pairs: list of (i, j, eps_ij, cc). Solve eps_ij = m_j - m_i, mean(m)=0."""
    if not pairs:
        return np.full(nb, np.nan)
    G = np.zeros((len(pairs) + 1, nb))
    d = np.zeros(len(pairs) + 1)
    w = np.ones(len(pairs) + 1)
    for k, (i, j, e, c) in enumerate(pairs):
        G[k, i], G[k, j], d[k], w[k] = -1, 1, e, max(c, 0) ** 2
    G[-1, :] = 1.0
    w[-1] = 1e3
    W = np.sqrt(w)[:, None]
    m, *_ = np.linalg.lstsq(G * W, d * W[:, 0], rcond=None)
    used = np.zeros(nb, bool)
    for i, j, _, _ in pairs:
        used[i] = used[j] = True
    m[~used] = np.nan
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--h5', default=None)
    ap.add_argument('--templates', default=None)
    ap.add_argument('--loc', default=None)
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--nbin', type=int, default=200)
    ap.add_argument('--fmin', type=float, default=2.0)
    ap.add_argument('--fmax', type=float, default=8.0)
    ap.add_argument('--min-snr', type=float, default=15.0, help='min master SNR to use channel')
    ap.add_argument('--tag', default='')
    ap.add_argument('--iso', type=float, default=ISO, help='isolation window (s) vs any-family detection')
    ap.add_argument('--subset', choices=['all', 'odd', 'even'], default='all',
                    help='use alternate isolated events only (split-half error estimate)')
    a = ap.parse_args()
    F = a.family
    h = h5py.File(a.h5 or f'data/waveforms/fam{F}.h5', 'r')
    T = h5py.File(a.templates or f'data/templates/fam{F}_templates.h5', 'r')
    locf = a.loc or f'data/locations/fam{F}_3d.json'
    loc = json.load(open(locf))
    picks = pd.read_csv(locf.replace('.json', '_picks.csv')).set_index('sta')
    sr, pre = h.attrs['sr'], h.attrs['pre']
    sos = butter(4, [a.fmin, a.fmax], btype='band', fs=sr, output='sos')
    t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    npts = h[f'chan/{next(iter(h["chan"]))}/data'].shape[1]
    t = np.arange(npts) / sr + pre                       # rel. catalog time
    t_lapse = t - loc['origin_rel_catalog_s']            # rel. origin time

    # 1. isolation using the full v2 catalog (all families)
    cat = pd.read_csv(a.catalog)
    cat = cat[cat.source != 'v1_only']
    # unit-safe epoch seconds (pandas 3 parses to datetime64[us], not ns)
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    te = epoch_s(t_ev)
    lo = np.searchsorted(allt, te - a.iso)
    hi = np.searchsorted(allt, te + a.iso, side='right')
    isolated = (hi - lo) == 1                            # only itself
    print(f'isolated events: {isolated.sum()} of {len(te)}', flush=True)

    # 2. bins with forced edges
    idx = np.where(isolated)[0]
    idx = idx[np.argsort(te[idx])]
    if a.subset != 'all':
        idx = idx[(0 if a.subset == 'even' else 1)::2]
        a.nbin = a.nbin // 2            # keep the same number of bins (same time sampling)
    edges_t = [float(epoch_s(pd.DatetimeIndex([q]))[0]) for q in QUAKES]
    seg_id = np.searchsorted(edges_t, te[idx])
    bins = []
    for sgm in np.unique(seg_id):
        ii = idx[seg_id == sgm]
        nb = max(1, int(round(len(ii) / a.nbin)))
        for chunk in np.array_split(ii, nb):
            bins.append(chunk)
    nb = len(bins)
    bin_t = [t_ev[b].min() + (t_ev[b].max() - t_ev[b].min()) / 2 for b in bins]
    print(f'{nb} bins (~{a.nbin} isolated events each)', flush=True)

    os.makedirs('data/dvv', exist_ok=True)
    summary = []
    series = {}
    for cid in sorted(T):
        n, s, l, c = cid.split('.')
        sta = f'{n}.{s}'
        if T[cid].attrs['snr'] < a.min_snr or sta not in picks.index:
            continue
        S = picks.loc[sta, 'S_final'] if np.isfinite(picks.loc[sta, 'S_final']) else np.nan
        if not np.isfinite(S):
            continue
        ev = h[f'chan/{cid}/ev'][:]
        pos = {e: k for k, e in enumerate(ev)}
        D = sosfiltfilt(sos, h[f'chan/{cid}/data'][:].astype('float64'), axis=1)
        D /= (np.std(D, axis=1) + 1e-20)[:, None]
        ref = D[[pos[e] for e in idx if e in pos]].mean(0) if any(e in pos for e in idx) else None
        if ref is None:
            continue
        env = np.abs(hilbert(ref))
        noise_lvl = np.sqrt(np.mean(ref[(t > -15) & (t < -5)] ** 2))
        envs = np.convolve(env, np.ones(int(sr)) / sr, mode='same')
        after = np.where((t > S + CODA_START) & (envs < SNR_END * noise_lvl))[0]
        t_end = min(t[after[0]] if len(after) else t[-1] - 1, t[-1] - 1)
        if t_end - (S + CODA_START) < 3.0:
            continue
        coda = (t >= S + CODA_START) & (t <= t_end)
        sdir = (t >= S - 0.5) & (t <= S + 2.5)
        noise = (t >= -15) & (t <= -15 + (t_end - S - CODA_START))
        stacks = [None] * nb
        for b, members in enumerate(bins):
            kk = [pos[e] for e in members if e in pos]
            if len(kk) >= a.nbin // 3:
                stacks[b] = D[kk].mean(0)
        have = [b for b in range(nb) if stacks[b] is not None]
        if len(have) < 4:
            continue
        rows = []
        for b in have:
            e_ref, cc_ref = stretch_cc(ref, stacks[b], t_lapse, coda, EPS)
            dt_s, cc_s = shift_dt(ref, stacks[b], sr, sdir, int(0.1 * sr))
            e_noise, cc_noise = stretch_cc(ref, stacks[b], t_lapse, noise, EPS)
            inj = stacks[b] if b < have[len(have) // 2] else \
                np.interp(t_lapse / (1 + INJECT), t_lapse, stacks[b])   # arrivals later: dv/v = -INJECT
            e_inj, _ = stretch_cc(ref, inj, t_lapse, coda, EPS)
            rows.append(dict(bin=b, t=bin_t[b], dvv_ref=e_ref, cc_ref=cc_ref, dt_directS=dt_s,
                             cc_directS=cc_s, dvv_noise=e_noise, cc_noise=cc_noise, dvv_inj=e_inj))
        pairs = []
        for ii_, bi in enumerate(have):
            for bj in have[ii_ + 1:ii_ + 1 + MAXPAIR]:
                e, cc = stretch_cc(stacks[bi], stacks[bj], t_lapse, coda, EPS)
                pairs.append((bi, bj, e, cc))
        m = invert_pairs(nb, pairs)
        # keep the pair measurements for the joint (all-channel) inversion in dvv_invert.py
        pd.DataFrame(pairs, columns=['bi', 'bj', 'eps', 'cc']).to_csv(
            f'data/dvv/fam{F}_{cid}{a.tag}_pairs.csv', index=False)
        df = pd.DataFrame(rows)
        df['dvv_inv'] = [m[b] for b in df.bin]
        df.to_csv(f'data/dvv/fam{F}_{cid}{a.tag}.csv', index=False)
        series[cid] = df
        second = df.bin >= have[len(have) // 2]
        inj_rec = (df.dvv_inj[second].mean() - df.dvv_ref[second].mean())
        summary.append(dict(chan=cid, snr=T[cid].attrs['snr'], coda_s=float(t_end - S - CODA_START),
                            n_bins=len(have), mean_cc_coda=df.cc_ref.mean(),
                            dvv_ref_std=df.dvv_ref.std(), dvv_inv_std=df.dvv_inv.std(),
                            dvv_noise_std=df.dvv_noise.std(), directS_dt_std=df.dt_directS.std(),
                            injected=-INJECT, recovered=inj_rec))
        print(f'{cid:18s} coda {t_end - S - CODA_START:5.1f}s bins {len(have):3d} '
              f'CC {df.cc_ref.mean():.2f} std(dv/v) ref {df.dvv_ref.std()*100:.3f}% '
              f'inv {df.dvv_inv.std()*100:.3f}% noise {df.dvv_noise.std()*100:.3f}% '
              f'inj {-INJECT*100:.2f}%->{inj_rec*100:.3f}%', flush=True)
    sm = pd.DataFrame(summary)
    sm.to_csv(f'data/dvv/fam{F}_summary{a.tag}.csv', index=False)

    # figure: dv/v (inversion) per channel through time, best channels
    if len(sm):
        top = sm.sort_values('mean_cc_coda', ascending=False).chan[:12]
        fig, ax = plt.subplots(len(top), 1, figsize=(12, 1.6 * len(top) + 1), sharex=True)
        ax = np.atleast_1d(ax)
        for k, cid in enumerate(top):
            df = series[cid]
            ax[k].plot(df.t, df.dvv_inv * 100, 'k.-', lw=0.7, ms=3, label='all-pairs inversion')
            ax[k].plot(df.t, (df.dvv_ref - df.dvv_ref.mean()) * 100, '.', color='C1', ms=3,
                       label='vs full-period ref')
            for q in QUAKES:
                ax[k].axvline(pd.Timestamp(q), color='r', lw=0.7, ls='--')
            ax[k].set_ylabel('dv/v %')
            ax[k].text(0.01, 0.85, cid, transform=ax[k].transAxes, fontsize=8)
        ax[0].legend(fontsize=7, loc='upper right')
        ax[0].set_title(f'Family {F}: coda dv/v from {a.nbin}-event isolated-LFE stacks ({a.fmin}-{a.fmax} Hz)')
        fig.tight_layout()
        fig.savefig(f'figures/fam{F}_dvv{a.tag}.png', dpi=130)


if __name__ == '__main__':
    main()
