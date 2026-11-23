#!/usr/bin/env python3
"""CWIRE-style dv/v for one LFE family (Hotovec-Ellis CWIRE 1.0.0 adapted to LFE stacks;
design per Fable review 2026-11-22, see NOTES.md).

1. "Events" = stacks of N isolated LFEs (N=50; N=25 within 182 d after each mainshock), never
   straddling a mainshock; stack time = mean member time. Same stack membership for all channels.
2. Per channel (2-8 Hz, traces normalised before stacking), at 200 Hz (4x upsampled):
   - align every stack on direct S to the reference (all isolated events) stack: lag from CC in
     [S-0.5, S+2.5] s (+-0.1 s, sub-sample) -> removes catalog-time / alignment jitter
     (stretching about origin time without a shift maps a timing error d into dv/v ~ d/t).
   - coda window tau = t - S in [3 s, tau_end], tau_end where the reference envelope < 2x noise.
   - stretch about S: a(tau(1+eps)) vs b(tau), eps grid +-1 % at 0.02 % + parabolic refinement.
     eps_ab = dv/v(b) - dv/v(a) (positive = faster).
   - pairs: all stack pairs within MAXSEP days (730), plus every stack vs the reference.
3. Joint inversion over channels for one curve (CWIRE design matrix): piecewise-linear dv/v on
   control nodes (30 d; 10 d within 182 d after each mainshock), each segment between mainshocks
   has its own nodes (no continuity or smoothing across a mainshock -> steps not smeared);
   1st+2nd difference damping (0.5/0.5 lambda, as CWIRE) inside segments; per-channel offsets
   for the reference rows; weights CC^2/(1-CC^2) with Huber IRLS; mean(curve) = 0.
   lambda from leave-channels-out 5-fold cross-validation (CWIRE uses an L-curve).
4. Hypothesis test WITHOUT the curve/regularisation: f(t) = trend + annual + steps at the 3
   mainshocks + A exp(-(t-M7)/60 d), fitted directly to all pair + reference rows.
5. Uncertainties: block bootstrap over stacks (each stack resampled with its pairs), and
   delete-one-channel jackknife; formal LSQ errors are reported but are too small (pairs share stacks).
Outputs: data/dvv/fam{F}_cwire{tag}_pairs.parquet, _stacks.csv, _curve.csv, _params.csv/.json,
figures/fam{F}_dvv_cwire{tag}.png
"""
import argparse
import json
import os
from multiprocessing import Pool

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt
from scipy.sparse import coo_matrix, diags

QUAKES = {'pet': '2021-12-20T20:10', 'fer': '2022-12-20T10:34', 'm7': '2024-12-05T18:44'}
UP = 4
CODA_START, SNR_END = 3.0, 2.0
EPS = np.round(np.arange(-0.01, 0.01 + 1e-9, 0.0002), 6)
MAXSEP = 730.0
CC_MIN = 0.5
HUBER = 2.0
TAU_TR = 60.0


def epoch_s(x):
    x = pd.DatetimeIndex(x)
    return np.asarray((x - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1), dtype=float)


def make_stacks(te, idx, n_norm, n_post, post_days=182.0):
    """te: epoch s of all events; idx: isolated event indices. Returns list of member arrays."""
    idx = idx[np.argsort(te[idx])]
    qs = sorted(epoch_s(list(QUAKES.values())))
    seg = np.searchsorted(qs, te[idx])
    out = []
    for s in np.unique(seg):
        ii = idx[seg == s]
        if s > 0:   # split the first post_days after the mainshock that opens this segment
            post = te[ii] < qs[s - 1] + post_days * 86400
            parts = [(ii[post], n_post), (ii[~post], n_norm)]
        else:
            parts = [(ii, n_norm)]
        for jj, n in parts:
            if len(jj) == 0:
                continue
            k = max(1, int(round(len(jj) / n)))
            out += list(np.array_split(jj, k))
    return out


def subsample_peak(y, k):
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        return (0.5 * (y0 - y2) / den if den != 0 else 0.0), y1
    return 0.0, y[k]


def channel_job(args):
    (h5f, cid, S, stacks, stack_t, iso_idx, fmin, fmax, inject, window) = args
    with h5py.File(h5f, 'r') as h:
        sr, pre = h.attrs['sr'], h.attrs['pre']
        ev = h[f'chan/{cid}/ev'][:]
        X = h[f'chan/{cid}/data'][:].astype('float64')
    pos = {e: k for k, e in enumerate(ev)}
    sos = butter(4, [fmin, fmax], btype='band', fs=sr, output='sos')
    X = sosfiltfilt(sos, X, axis=1)
    X /= (np.std(X, axis=1) + 1e-20)[:, None]
    t = np.arange(X.shape[1]) / sr + pre
    rk = [pos[e] for e in iso_idx if e in pos]
    if len(rk) < 100:
        return None
    ref = X[rk].mean(0)
    # coda end from reference envelope
    env = np.convolve(np.abs(hilbert(ref)), np.ones(int(sr)) / sr, mode='same')
    noise = np.sqrt(np.mean(ref[(t > -15) & (t < -5)] ** 2))
    after = np.where((t > S + CODA_START) & (env < SNR_END * noise))[0]
    t_end = min(t[after[0]] if len(after) else t[-1] - 1, t[-1] - 1.0)
    if t_end - (S + CODA_START) < 3.0:
        return None
    srU = sr * UP
    tU = np.arange(X.shape[1] * UP) / srU + pre
    up = lambda x: resample_poly(x, UP, 1)  # noqa: E731
    refU = up(ref)
    sw = (tU >= S - 0.5) & (tU <= S + 2.5)
    maxlag = int(0.1 * srU)
    i0 = np.where(sw)[0]

    def align(xU):
        r = refU[sw] - refU[sw].mean()
        ccs = []
        for L in range(-maxlag, maxlag + 1):
            c = xU[i0 + L]
            c = c - c.mean()
            ccs.append(r @ c / (np.linalg.norm(r) * np.linalg.norm(c) + 1e-20))
        ccs = np.array(ccs)
        k = int(np.argmax(ccs))
        f, cmax = subsample_peak(ccs, k)
        lag = (k - maxlag + f) / srU                 # x is late by `lag` s relative to ref
        return np.interp(tU + lag, tU, xU), lag, cmax

    tau = tU - S
    coda = (tau >= CODA_START) & (tau <= t_end - S)
    if window == 'noise':          # control: same length, pre-event noise (stretch still about S)
        L = t_end - S - CODA_START
        coda = (tU >= -14.0) & (tU <= -14.0 + L) & (tU < S - 1.0)
    tauc = tau[coda]
    S_list, keep, lags, ccS, nmem = [], [], [], [], []
    for b, mem in enumerate(stacks):
        kk = [pos[e] for e in mem if e in pos]
        if len(kk) < max(5, len(mem) // 2):
            continue
        xU, lag, cs = align(up(X[kk].mean(0)))
        if inject is not None and stack_t[b] >= inject[0]:
            # synthetic dv/v = inject[1] after the date: arrivals at tau/(1+dvv) -> x_new(tau) = x(tau (1+dvv))
            xU = np.interp(tau * (1 + inject[1]), tau, xU)
        S_list.append(xU)
        keep.append(b)
        lags.append(lag)
        ccS.append(cs)
        nmem.append(len(kk))
    if len(keep) < 6:
        return None
    # unstretched, normalised coda of each stack (the "b" side)
    B = np.array([x[coda] for x in S_list])
    B = B - B.mean(1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-20
    rc = refU[coda] - refU[coda].mean()
    rc /= np.linalg.norm(rc) + 1e-20

    def stretched(xU):
        M = np.array([np.interp(tauc * (1 + e), tau, xU) for e in EPS])
        M -= M.mean(1, keepdims=True)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-20
        return M

    def best(C):
        """C: (n_eps, n) CC matrix -> eps, cc per column (parabolic refinement)."""
        k = np.argmax(C, axis=0)
        e, c = np.empty(C.shape[1]), np.empty(C.shape[1])
        for j in range(C.shape[1]):
            f, c[j] = subsample_peak(C[:, j], k[j])
            e[j] = EPS[k[j]] + f * (EPS[1] - EPS[0])
            if k[j] in (0, len(EPS) - 1):
                c[j] = -1.0                              # at the edge of the grid: reject
        return e, c

    tk = np.array([stack_t[b] for b in keep])
    rows = []
    Mref = stretched(refU)
    e, c = best(Mref @ B.T)                              # reference stretched to each stack
    for j, b in enumerate(keep):
        rows.append((cid, -1, b, e[j], c[j]))
    for i, b in enumerate(keep):
        J = np.where((tk > tk[i]) & (tk - tk[i] <= MAXSEP * 86400))[0]
        if len(J) == 0:
            continue
        e, c = best(stretched(S_list[i]) @ B[J].T)
        for jj, j in enumerate(J):
            rows.append((cid, b, keep[j], e[jj], c[jj]))
    pairs = pd.DataFrame(rows, columns=['chan', 'a', 'b', 'eps', 'cc'])
    st = pd.DataFrame(dict(chan=cid, stack=keep, n=nmem, dtS=lags, ccS=ccS))
    meta = dict(chan=cid, coda_s=float(t_end - S - CODA_START), n_stacks=len(keep),
                median_cc_ref=float(pairs[pairs.a < 0].cc.median()),
                median_cc_pair=float(pairs[pairs.a >= 0].cc.median()) if (pairs.a >= 0).any() else np.nan,
                dtS_std_ms=float(np.std(lags) * 1e3))
    return pairs, st, meta


# ---------------------------------------------------------------- inversion
def nodes_for(t0, t1, post_days=182.0, d_norm=30.0, d_post=10.0):
    """Control nodes per segment (segments split at mainshocks). Returns list of arrays (epoch s)."""
    qs = sorted(epoch_s(list(QUAKES.values())))
    bounds = [t0] + [q for q in qs if t0 < q < t1] + [t1]
    segs = []
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        if s0 in qs:
            p = min(s0 + post_days * 86400, s1)
            n1 = np.arange(s0, p, d_post * 86400)
            n2 = np.arange(p, s1, d_norm * 86400) if p < s1 else np.array([])
            nd = np.r_[n1, n2, s1]
        else:
            nd = np.r_[np.arange(s0, s1, d_norm * 86400), s1]
        segs.append(np.unique(nd))
    return segs


def build_G(ta, tb, has_a, segs, chan_k, nk):
    """Rows: m(tb) - m(ta) [pairs] or m(tb) - c_k [reference rows, has_a False]."""
    offs = np.cumsum([0] + [len(s) for s in segs])
    nn = offs[-1]
    allq = [s[0] for s in segs[1:]]

    def interp(t):
        sg = np.searchsorted(allq, t, side='right')     # segment index (event at quake time -> after)
        r, c, v = [], [], []
        for n, (tt, s) in enumerate(zip(t, sg)):
            nd = segs[s]
            j = np.clip(np.searchsorted(nd, tt) - 1, 0, len(nd) - 2)
            w = (tt - nd[j]) / (nd[j + 1] - nd[j])
            w = min(max(w, 0.0), 1.0)
            r += [n, n]
            c += [offs[s] + j, offs[s] + j + 1]
            v += [1 - w, w]
        return np.array(r), np.array(c), np.array(v)

    n = len(tb)
    rb, cb, vb = interp(tb)
    ra, ca, va = interp(np.where(has_a, ta, tb))
    va = np.where(has_a[ra], va, 0.0)
    rows = np.r_[rb, ra, np.where(~has_a)[0]]
    cols = np.r_[cb, ca, nn + chan_k[~has_a]]
    vals = np.r_[vb, -va, -np.ones((~has_a).sum())]
    return coo_matrix((vals, (rows, cols)), shape=(n, nn + nk)).tocsr(), nn


def reg_matrix(segs, nn, nk):
    r, c, v = [], [], []
    offs = np.cumsum([0] + [len(s) for s in segs])
    k = 0
    for s, nd in enumerate(segs):
        o = offs[s]
        for j in range(len(nd) - 1):
            r += [k, k]; c += [o + j, o + j + 1]; v += [-0.5, 0.5]; k += 1        # noqa: E702
        for j in range(1, len(nd) - 1):
            r += [k] * 3; c += [o + j - 1, o + j, o + j + 1]; v += [0.5, -1.0, 0.5]; k += 1   # noqa: E702
    return coo_matrix((v, (r, c)), shape=(k, nn + nk)).tocsr()


def solve(G, d, w0, R, lam, nn, iters=4, mult=None):
    """Damped weighted LSQ with Huber IRLS; few unknowns (~300) -> normal equations, direct solve."""
    w0 = w0 if mult is None else w0 * mult
    n = G.shape[1]
    RtR = (R.T @ R).toarray() * lam ** 2
    mrow = np.zeros(n)
    mrow[:nn] = 1.0 / nn
    MtM = np.outer(mrow, mrow) * 1e4                     # mean(curve) = 0
    Gt = G.T.tocsr()
    w = w0.copy()
    x = np.zeros(n)
    for _ in range(iters):
        N = (Gt @ diags(w) @ G).toarray() + RtR + MtM
        x = np.linalg.lstsq(N, Gt @ (w * d), rcond=None)[0]
        r = d - G @ x
        ok = w0 > 0
        z = np.abs(r) * np.sqrt(w0) / (1.4826 * np.median(np.abs(r[ok]) * np.sqrt(w0[ok])) + 1e-12)
        w = w0 * np.where(z <= HUBER, 1.0, HUBER / np.maximum(z, 1e-12))
    return x, w


def param_G(ta, tb, has_a, chan_k, nk):
    """Parametric f(t): trend, annual sin/cos, steps, M7 transient; reference rows get -c_k."""
    def f(t):
        yr = (t - epoch_s(['2008-01-01'])[0]) / (365.25 * 86400)
        cols = [yr, np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
        for q in QUAKES.values():
            cols.append((t >= epoch_s([q])[0]).astype(float))
        dt = (t - epoch_s([QUAKES['m7']])[0]) / 86400
        cols.append(np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / TAU_TR), 0.0))
        return np.array(cols).T
    Fb, Fa = f(tb), f(np.where(has_a, ta, tb))
    Fa[~has_a] = 0.0
    C = np.zeros((len(tb), nk))
    C[np.where(~has_a)[0], chan_k[~has_a]] = -1.0
    return np.c_[Fb - Fa, C]


PNAMES = ['trend_pct_per_yr', 'annual_sin_pct', 'annual_cos_pct', 'step_pet_pct', 'step_fer_pct',
          'step_m7_pct', 'transient60_pct']


def wls(A, d, w):
    W = np.sqrt(w)
    m, *_ = np.linalg.lstsq(A * W[:, None], d * W, rcond=None)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--iso', type=float, default=5.0)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--n-post', type=int, default=25)
    ap.add_argument('--fmin', type=float, default=2.0)
    ap.add_argument('--fmax', type=float, default=8.0)
    ap.add_argument('--min-snr', type=float, default=15.0)
    ap.add_argument('--subset', choices=['all', 'odd', 'even'], default='all')
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--tag', default='')
    ap.add_argument('--reuse', action='store_true', help='reuse saved pair measurements')
    ap.add_argument('--inject', nargs=2, default=None, metavar=('DATE', 'DVV'),
                    help='synthetic test: apply dv/v=DVV (fraction) to all stacks after DATE')
    ap.add_argument('--window', choices=['coda', 'noise'], default='coda')
    ap.add_argument('--chan-regex', default=None, help='use only channels matching (inversion stage)')
    ap.add_argument('--qc-cc', type=float, default=None,
                    help='min median stack-vs-reference CC for a channel (default 0.7*sqrt(n/150), i.e. scaled '
                         'with stack size: CC of an N-stack grows ~sqrt(N) at low SNR)')
    ap.add_argument('--pair-cc', type=float, default=None, help='min CC per measurement (default qc_cc - 0.2)')
    a = ap.parse_args()
    F = a.family
    tag = a.tag + ('' if a.subset == 'all' else f'_{a.subset}')
    meas_tag = tag
    if a.inject:
        tag += f'_inj{a.inject[0][:10]}_{a.inject[1]}'
        meas_tag = tag
    if a.window == 'noise':
        tag += '_noise'
        meas_tag = tag
    if a.chan_regex:
        tag += '_' + ''.join(ch for ch in a.chan_regex if ch.isalnum())
    base = f'data/dvv/fam{F}_cwire{tag}'
    mbase = f'data/dvv/fam{F}_cwire{meas_tag}'
    h5f = f'data/waveforms/fam{F}.h5'
    with h5py.File(h5f, 'r') as h:
        t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    te = epoch_s(t_ev)
    cat = pd.read_csv(a.catalog)
    cat = cat[cat.source != 'v1_only']
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    iso = (np.searchsorted(allt, te + a.iso, side='right') - np.searchsorted(allt, te - a.iso)) == 1
    idx = np.where(iso)[0]
    idx = idx[np.argsort(te[idx])]
    if a.subset != 'all':
        idx = idx[(0 if a.subset == 'even' else 1)::2]
        a.n, a.n_post = a.n // 2, max(a.n_post // 2, 10)
    stacks = make_stacks(te, idx, a.n, a.n_post)
    stack_t = np.array([te[m].mean() for m in stacks])
    print(f'family {F}: {iso.sum()} isolated of {len(te)}; {len(stacks)} stacks', flush=True)

    if a.reuse and os.path.exists(mbase + '_pairs.parquet'):
        P = pd.read_parquet(mbase + '_pairs.parquet')
        meta = pd.read_csv(mbase + '_channels.csv')
    else:
        T = h5py.File(f'data/templates/fam{F}_templates.h5', 'r')
        loc = f'data/locations/fam{F}_3d'
        picks = pd.read_csv(loc + '_picks.csv').set_index('sta')
        jobs = []
        for cid in sorted(T):
            sta = '.'.join(cid.split('.')[:2])
            if T[cid].attrs['snr'] < a.min_snr or sta not in picks.index or not np.isfinite(picks.loc[sta, 'S_final']):
                continue
            inj = None if a.inject is None else (epoch_s([a.inject[0]])[0], float(a.inject[1]))
            jobs.append((h5f, cid, float(picks.loc[sta, 'S_final']), stacks, stack_t, idx, a.fmin, a.fmax,
                         inj, a.window))
        T.close()
        print(f'{len(jobs)} channels to measure', flush=True)
        res, metas, sts = [], [], []
        with Pool(a.procs) as p:
            for r in p.imap_unordered(channel_job, jobs):
                if r is None:
                    continue
                res.append(r[0])
                sts.append(r[1])
                metas.append(r[2])
                print(f"  {r[2]['chan']:18s} stacks {r[2]['n_stacks']:4d} coda {r[2]['coda_s']:5.1f}s "
                      f"CC ref {r[2]['median_cc_ref']:.2f} pair {r[2]['median_cc_pair']:.2f} "
                      f"dtS std {r[2]['dtS_std_ms']:.1f} ms", flush=True)
        P = pd.concat(res, ignore_index=True)
        meta = pd.DataFrame(metas)
        P.to_parquet(base + '_pairs.parquet')
        meta.to_csv(base + '_channels.csv', index=False)
        pd.concat(sts).to_csv(base + '_stack_shifts.csv', index=False)
    pd.DataFrame(dict(stack=range(len(stacks)), t=pd.to_datetime(stack_t, unit='s'),
                      n=[len(m) for m in stacks])).to_csv(base + '_stacks.csv', index=False)

    # ---- QC channels: median reference CC >= 0.7 (as before)
    qc = a.qc_cc if a.qc_cc is not None else round(min(0.7, 0.7 * np.sqrt(a.n / 150)), 2)
    pcc = a.pair_cc if a.pair_cc is not None else max(qc - 0.2, 0.2)
    good = meta[meta.median_cc_ref >= qc].chan.tolist()
    if a.chan_regex:
        import re
        good = [c for c in good if re.search(a.chan_regex, c)]
    P = P[P.chan.isin(good) & (P.cc >= pcc)].reset_index(drop=True)
    print(f'QC: channel median ref CC >= {qc}, measurement CC >= {pcc}', flush=True)
    if len(good) == 0 or len(P) == 0:
        print('no channels/measurements pass QC; nothing to invert')
        json.dump(dict(family=F, n_channels=0, note='no channels pass QC'), open(base + '_params.json', 'w'))
        return
    ck = {c: k for k, c in enumerate(good)}
    nk = len(good)
    chan_k = P.chan.map(ck).values
    has_a = (P.a >= 0).values
    ta = np.where(has_a, stack_t[np.clip(P.a.values, 0, None)], np.nan)
    tb = stack_t[P.b.values]
    d = P.eps.values
    cc = np.clip(P.cc.values, 0, 0.99)
    w0 = cc ** 2 / (1 - cc ** 2)
    w0 /= w0.mean()
    print(f'{nk} channels pass; {has_a.sum()} pair rows + {(~has_a).sum()} reference rows', flush=True)

    # ---- curve: CWIRE-style regularised inversion with breaks at mainshocks
    segs = nodes_for(stack_t.min() - 86400, stack_t.max() + 86400)
    G, nn = build_G(np.nan_to_num(ta), tb, has_a, segs, chan_k, nk)
    R = reg_matrix(segs, nn, nk)
    rng = np.random.default_rng(3)
    fold = rng.permutation(nk) % 5
    cv = []
    for lam in [1, 3, 10, 30, 100, 300, 1000, 3000]:
        num = den = 0.0
        for f in range(5):
            tr = fold[chan_k] != f
            x, _ = solve(G[tr], d[tr], w0[tr], R, lam, nn, iters=2)
            te_ = (~tr) & has_a
            r = d[te_] - G[te_][:, :nn] @ x[:nn]
            r = np.clip(r, -5 * r.std(), 5 * r.std())
            num += np.sum(w0[te_] * r ** 2)
            den += np.sum(w0[te_])
        cv.append((lam, num / den))
    lam = min(cv, key=lambda x: x[1])[0]
    print('lambda CV: ' + ', '.join(f'{l:g}:{e:.3e}' for l, e in cv) + f' -> {lam:g}', flush=True)
    x, w = solve(G, d, w0, R, lam, nn)
    node_t = np.concatenate(segs)
    node_seg = np.concatenate([[s] * len(nd) for s, nd in enumerate(segs)])
    # data support per node (CWIRE "Density"/"Crosses")
    Gn = abs(G[:, :nn]).tocsc()
    density = np.asarray((Gn > 0).sum(0)).ravel()
    # block bootstrap over stacks for the curve
    sid_a, sid_b = np.where(has_a, P.a.values, -1), P.b.values
    ns = len(stacks)
    XB, MB = [], []
    A_par = param_G(np.nan_to_num(ta), tb, has_a, chan_k, nk)
    m_par = wls(A_par, d, w)
    for i in range(a.nboot):
        cnt = np.bincount(rng.integers(0, ns, ns), minlength=ns).astype(float)
        mult = cnt[sid_b] * np.where(sid_a >= 0, cnt[np.clip(sid_a, 0, None)], 1.0)
        if i < min(a.nboot, 100):
            XB.append(solve(G, d, w0, R, lam, nn, iters=2, mult=mult)[0][:nn])
        MB.append(wls(A_par, d, w * mult))
    XB, MB = np.array(XB), np.array(MB)
    MJ = []
    for k in range(nk):
        s = chan_k != k
        MJ.append(wls(A_par[s][:, [i for i in range(A_par.shape[1]) if i < 7 or i - 7 != k]], d[s], w[s])[:7])
    MJ = np.array(MJ)
    se_j = np.sqrt((nk - 1) / nk * np.sum((MJ - MJ.mean(0)) ** 2, axis=0))
    se_b = MB[:, :7].std(0)
    Aw = A_par * np.sqrt(w)[:, None]
    r = d - A_par @ m_par
    red = np.sum(w * r ** 2) / (len(d) - A_par.shape[1])
    se_l = np.sqrt(np.diag(np.linalg.pinv(Aw.T @ Aw))[:7] * red)
    curve = pd.DataFrame(dict(t=pd.to_datetime(node_t, unit='s'), segment=node_seg, dvv_pct=x[:nn] * 100,
                              err_boot_pct=XB.std(0) * 100, density=density))
    curve.to_csv(base + '_curve.csv', index=False)
    out = []
    for k, n_ in enumerate(PNAMES):
        se = max(se_b[k], se_j[k])
        out.append(dict(param=n_, value=m_par[k] * 100, se_boot=se_b[k] * 100, se_jack=se_j[k] * 100,
                        se_lsq=se_l[k] * 100, sigma=m_par[k] / se if se > 0 else np.nan))
        print(f'  {n_:18s} {m_par[k]*100:+.4f}  boot {se_b[k]*100:.4f}  jack {se_j[k]*100:.4f}  '
              f'lsq {se_l[k]*100:.4f}  -> {m_par[k]/se:+.1f} sigma', flush=True)
    pd.DataFrame(out).to_csv(base + '_params.csv', index=False)
    amp = np.hypot(m_par[1], m_par[2]) * 100
    peak = ((np.pi / 2 - np.arctan2(m_par[2], m_par[1])) % (2 * np.pi)) / (2 * np.pi) * 365.25
    summ = dict(family=F, n_stacks=ns, n_channels=nk, n_pair_rows=int(has_a.sum()), n_ref_rows=int((~has_a).sum()),
                lam=lam, annual_amp_pct=amp, annual_peak_doy=peak,
                n_stacks_0_120d_after_m7=int((((stack_t - epoch_s([QUAKES['m7']])[0]) / 86400 >= 0) & ((stack_t - epoch_s([QUAKES['m7']])[0]) / 86400 < 120)).sum()),
                **{r_['param']: dict(value=r_['value'], se=max(r_['se_boot'], r_['se_jack'])) for r_ in out})
    json.dump(summ, open(base + '_params.json', 'w'), indent=1, default=float)
    print(f"  annual amplitude {amp:.4f} %, fastest ~day {peak:.0f}", flush=True)

    # ---- figure
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for s in np.unique(node_seg):
        c = curve[curve.segment == s]
        ax[0].plot(c.t, c.dvv_pct, 'k-', lw=1.2)
        ax[0].fill_between(c.t, c.dvv_pct - c.err_boot_pct, c.dvv_pct + c.err_boot_pct, color='0.7', alpha=0.6, lw=0)
    tt = np.linspace(stack_t.min(), stack_t.max(), 3000)
    Ft = param_G(np.full_like(tt, np.nan), tt, np.zeros(len(tt), bool), np.zeros(len(tt), int), 1)[:, :7]
    fp = Ft @ m_par[:7]
    ax[0].plot(pd.to_datetime(tt, unit='s'), (fp - fp.mean() + curve.dvv_pct.mean() / 100) * 100, 'r', lw=0.8,
               label='parametric fit to pair data (trend+annual+steps+60-d M7 transient)')
    for q in QUAKES.values():
        for x_ in ax:
            x_.axvline(pd.Timestamp(q), color='C0', ls='--', lw=0.8)
    ax[0].set_ylabel('dv/v (%)  [+ = faster]')
    ax[0].legend(fontsize=8, loc='lower left')
    ax[0].set_title(f'Family {F}: CWIRE-style joint inversion ({nk} channels, {ns} stacks of ~{a.n} LFEs, '
                    f'lambda={lam:g} by CV; band = stack bootstrap)')
    ax[1].bar(pd.to_datetime(node_t, unit='s'), density, width=8, color='0.4')
    ax[1].set_ylabel('measurements per node')
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_dvv_cwire{tag}.png', dpi=130)


if __name__ == '__main__':
    main()
