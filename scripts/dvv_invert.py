#!/usr/bin/env python3
"""Joint network inversion for ONE dv/v(t) curve per family (CWIRE / Brenguier et al. 2014 style).

Data, for every QC-passing channel k (same QC as dvv_fit.py: coda CC >= 0.7 and noise-window
scatter >= 5x signal scatter):
  * all bin pairs (i, j) within MAXPAIR bins:  eps_ij^k = m_j - m_i
  * every bin vs the channel's full-period reference stack:  eps_ref_b^k = m_b - c_k
Unknowns: m_b (one network curve) and c_k (one offset per channel). Weights from the stretching
correlation, w = CC^2 / (1 - CC^2) (pairs with CC < CC_PAIR_MIN dropped). Iteratively reweighted
(Huber) to down-weight outliers; mean(m) = 0. Optional first-difference smoothing (--smooth)
that is never applied across the Dec 2021 / Dec 2022 / Dec 2024 mainshocks (default 0 = none, so
steps and transients are not biased).
Uncertainty per bin: max(LSQ posterior scaled by reduced chi2, bootstrap over channels).

Outputs (tag defaults to _iso5_n150; the inverted products get suffix _inv):
  data/dvv/fam{F}_network_dvv{tag}_inv.csv  (bin, t, dvv, err, n)  -> dvv_transient.py --tag {tag}_inv
  data/dvv/fam{F}_dvv_fit{tag}_inv.csv      same model as dvv_fit.py (offset, trend, annual, 3 steps)
  figures/fam{F}_dvv_inv{tag}.png           inverted curve vs the reference-stack network curve
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import lsqr

Q = {'M6.1 Petrolia': '2021-12-20T20:10', 'M6.4 Ferndale': '2022-12-20T10:34', 'M7.0 Mendocino': '2024-12-05T18:44'}
CC_MIN, NOISE_FAC = 0.7, 5.0
CC_PAIR_MIN = 0.5
HUBER = 2.0


def load(F, tag, chans):
    P, R, T = [], [], {}
    for k, c in enumerate(chans):
        fs, fp = f'data/dvv/fam{F}_{c}{tag}.csv', f'data/dvv/fam{F}_{c}{tag}_pairs.csv'
        if not (os.path.exists(fs) and os.path.exists(fp)):
            continue
        s = pd.read_csv(fs)
        for b, t in zip(s.bin, s.t):
            T[b] = t
        R.append(s.assign(k=k)[['k', 'bin', 'dvv_ref', 'cc_ref']])
        P.append(pd.read_csv(fp).assign(k=k))
    return (pd.concat(P) if P else pd.DataFrame()), (pd.concat(R) if R else pd.DataFrame()), T


def solve(P, R, nb, nk, smooth, edges_bins, iters=5):
    """Return m (nb), c (nk), per-row residuals; IRLS with Huber weights."""
    cc_w = lambda c: np.clip(c, 0, 0.99) ** 2 / (1 - np.clip(c, 0, 0.99) ** 2)  # noqa: E731
    nP, nR = len(P), len(R)
    rows = np.r_[np.arange(nP), np.arange(nP), nP + np.arange(nR), nP + np.arange(nR)]
    cols = np.r_[P.bi.values, P.bj.values, R.bin.values, nb + R.k.values]
    vals = np.r_[-np.ones(nP), np.ones(nP), np.ones(nR), -np.ones(nR)]
    G = coo_matrix((vals, (rows, cols)), shape=(nP + nR, nb + nk)).tocsr()
    d = np.r_[P.eps.values, R.dvv_ref.values]
    w0 = np.r_[cc_w(P.cc.values), cc_w(R.cc_ref.values)]
    # constraint mean(m)=0 (only over bins that have data) + optional smoothing
    used = np.zeros(nb, bool)
    used[np.r_[P.bi.values, P.bj.values, R.bin.values].astype(int)] = True
    C = coo_matrix((used / used.sum(), (np.zeros(nb, int), np.arange(nb))), shape=(1, nb + nk)).tocsr()
    extra, dext = [C * 1e3 * np.sqrt(w0.mean())], [0.0]
    if smooth > 0:
        ub = np.where(used)[0]
        rr, cc_, vv = [], [], []
        for n_, (i, j) in enumerate(zip(ub[:-1], ub[1:])):
            if any(i < e <= j for e in edges_bins):      # never smooth across a mainshock
                continue
            rr += [n_, n_]
            cc_ += [i, j]
            vv += [-1.0, 1.0]
        if rr:
            D = coo_matrix((vv, (rr, cc_)), shape=(max(rr) + 1, nb + nk)).tocsr()
            extra.append(D * smooth * np.sqrt(w0.mean()))
            dext += [0.0] * D.shape[0]
    w = w0.copy()
    for _ in range(iters):
        A = vstack([diags(np.sqrt(w)) @ G] + extra).tocsr()
        b = np.r_[d * np.sqrt(w), dext[:1], np.zeros(A.shape[0] - len(d) - 1)]
        x = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
        r = d - G @ x
        s = 1.4826 * np.median(np.abs(r * np.sqrt(w0))) + 1e-12
        z = np.abs(r * np.sqrt(w0)) / s
        w = w0 * np.where(z <= HUBER, 1.0, HUBER / z)
    m = x[:nb].copy()
    m[~used] = np.nan
    chi2 = float(np.sum(w * r ** 2) / max(len(d) - used.sum() - nk, 1))
    # posterior variance of m (dense; nb + nk is small)
    A = (diags(np.sqrt(w)) @ G).toarray()
    Ae = np.vstack([A] + [e.toarray() for e in extra])
    cov = np.linalg.pinv(Ae.T @ Ae)
    se = np.sqrt(np.clip(np.diag(cov)[:nb], 0, None) * max(chi2, 1))
    se[~used] = np.nan
    return m, x[nb:], se, chi2


def choose_smooth(P, R, nb, nk, edges_bins, grid=(0, 1, 3, 10, 30, 100, 300, 1000, 3000), nfold=5):
    """Leave-channels-out CV: fit on 4/5 of the channels, predict the held-out channels' pair
    measurements (pairs only: they do not depend on the unknown channel offsets). Weighted MSE."""
    rng = np.random.default_rng(2)
    fold = rng.permutation(nk) % nfold
    cc_w = lambda c: np.clip(c, 0, 0.99) ** 2 / (1 - np.clip(c, 0, 0.99) ** 2)  # noqa: E731
    tab = []
    for lam in grid:
        num = den = 0.0
        for f in range(nfold):
            tr = fold[P.k.values] != f
            trR = fold[R.k.values] != f
            m = solve(P[tr], R[trR], nb, nk, lam, edges_bins, iters=3)[0]
            te = P[~tr]
            pred = m[te.bj.values.astype(int)] - m[te.bi.values.astype(int)]
            ok = np.isfinite(pred)
            w = cc_w(te.cc.values[ok])
            r = te.eps.values[ok] - pred[ok]
            r = np.clip(r, -5 * np.std(r), 5 * np.std(r))       # robust to outliers
            num += np.sum(w * r ** 2)
            den += np.sum(w)
        tab.append((lam, num / den))
    best = min(tab, key=lambda x: x[1])[0]
    return best, tab


def model_fit(t, d, err):
    yr = (t - pd.Timestamp('2008-01-01')).dt.total_seconds().values / (365.25 * 86400)
    G = np.array([np.ones(len(t)), yr - yr.mean(), np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)] +
                 [(t >= pd.Timestamp(v)).astype(float).values for v in Q.values()]).T
    w = 1 / np.maximum(err, 1e-5) ** 2
    W = np.sqrt(w)[:, None]
    m, *_ = np.linalg.lstsq(G * W, d * W[:, 0], rcond=None)
    r = d - G @ m
    chi2 = np.sum(w * r ** 2) / (len(d) - len(m))
    se = np.sqrt(np.diag(np.linalg.pinv((G * W).T @ (G * W))) * max(chi2, 1))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        mm, *_ = np.linalg.lstsq(G[i] * W[i], d[i] * W[i, 0], rcond=None)
        bs.append(mm)
    return m, se, np.array(bs).std(0), chi2, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--tag', default='_iso5_n150')
    ap.add_argument('--smooth', default='0', help="first-difference damping: 0 = none (tests), number, or 'auto' "
                    "(leave-channels-out 5-fold cross-validation)")
    ap.add_argument('--suffix', default=None, help='output suffix (default _inv, or _inv_smooth if smoothing)')
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--min-chan', type=int, default=3)
    a = ap.parse_args()
    F, tag = a.family, a.tag
    sm = pd.read_csv(f'data/dvv/fam{F}_summary{tag}.csv')
    good = sm[(sm.mean_cc_coda >= CC_MIN) & (sm.dvv_noise_std >= NOISE_FAC * sm.dvv_ref_std)].chan.tolist()
    P, R, T = load(F, tag, good)
    if P.empty:
        print('no pair files (rerun dvv_family.py with this tag)')
        return
    P = P[P.cc >= CC_PAIR_MIN]
    nb, nk = int(max(T) + 1), len(good)
    tb = pd.Series({b: pd.Timestamp(t) for b, t in T.items()}).sort_index()
    edges_bins = [int(tb.index[np.searchsorted(tb.values, np.datetime64(pd.Timestamp(q)))])
                  for q in Q.values() if pd.Timestamp(q) < tb.max()]
    if a.smooth == 'auto':
        lam, cvtab = choose_smooth(P, R, nb, nk, edges_bins)
        print('cross-validation (leave-channels-out): ' + ', '.join(f'{l:g}:{e:.3g}' for l, e in cvtab) + f' -> {lam:g}')
    else:
        lam = float(a.smooth)
    a.smooth = lam
    suf = a.suffix or ('_inv' if lam == 0 else '_inv_smooth')
    m, c, se, chi2 = solve(P, R, nb, nk, a.smooth, edges_bins)
    # bootstrap over channels
    rng = np.random.default_rng(1)
    B = []
    for _ in range(a.nboot):
        ks = rng.integers(0, nk, nk)
        Pb = pd.concat([P[P.k == k].assign(k=n) for n, k in enumerate(ks)])
        Rb = pd.concat([R[R.k == k].assign(k=n) for n, k in enumerate(ks)])
        if Pb.empty:
            continue
        B.append(solve(Pb, Rb, nb, nk, a.smooth, edges_bins, iters=3)[0])
    B = np.array(B)
    se_b = np.nanstd(B, axis=0)
    nch = R.groupby('bin').k.nunique().reindex(range(nb)).fillna(0).astype(int).values
    out = pd.DataFrame(dict(bin=range(nb), t=[T.get(b) for b in range(nb)], dvv=m,
                            err=np.fmax(se, se_b), err_lsq=se, err_boot=se_b, n=nch))
    out = out[(out.n >= a.min_chan) & out.dvv.notna()]
    out.to_csv(f'data/dvv/fam{F}_network_dvv{tag}{suf}.csv', index=False)
    print(f'family {F} (smooth={lam:g}): {nk} channels, {len(P)} pairs (CC>={CC_PAIR_MIN}), {len(R)} ref measurements, '
          f'{len(out)} bins; reduced chi2 {chi2:.2f}; median err {out.err.median()*100:.4f}% '
          f'(LSQ {out.err_lsq.median()*100:.4f}, boot {out.err_boot.median()*100:.4f})')

    t = pd.to_datetime(out.t, format='mixed')
    mm, se_l, se_bs, chi2f, G = model_fit(t, out.dvv.values, out.err.values)
    names = ['offset', 'trend per yr', 'annual sin', 'annual cos'] + [f'step {k}' for k in Q]
    rows = []
    for k, n_ in enumerate(names):
        s = max(se_l[k], se_bs[k])
        print(f'  {n_:22s} {mm[k]*100:+.4f} %  ± {se_l[k]*100:.4f} (LSQ)  ± {se_bs[k]*100:.4f} (bootstrap)   {mm[k]/s:+.1f} sigma')
        rows.append(dict(param=n_, value_pct=mm[k] * 100, se_lsq_pct=se_l[k] * 100, se_boot_pct=se_bs[k] * 100))
    amp = np.hypot(mm[2], mm[3])
    peak = ((np.pi / 2 - np.arctan2(mm[3], mm[2])) % (2 * np.pi)) / (2 * np.pi) * 365.25
    print(f'  annual amplitude {amp*100:.4f} %, fastest near day-of-year {peak:.0f}; fit reduced chi2 {chi2f:.2f}')
    pd.DataFrame(rows).to_csv(f'data/dvv/fam{F}_dvv_fit{tag}{suf}.csv', index=False)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    fr = f'data/dvv/fam{F}_network_dvv{tag}.csv'
    if os.path.exists(fr):
        r = pd.read_csv(fr)
        ax.errorbar(pd.to_datetime(r.t, format='mixed'), (r.dvv - r.dvv.mean()) * 100, yerr=r.err * 100, fmt='o',
                    ms=2, lw=0.5, color='0.6', label='vs-reference network average (previous primary)')
    ax.errorbar(t, out.dvv * 100, yerr=out.err * 100, fmt='o', ms=2.5, lw=0.6, color='k',
                label=f'joint inversion, {nk} channels')
    ax.plot(t, G @ mm * 100, 'r', lw=1, label='fit: trend + annual + steps')
    for k, v in Q.items():
        ax.axvline(pd.Timestamp(v), color='C0', ls='--', lw=0.8)
    ax.set_ylabel('dv/v (%)  [+ = faster]')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title(f'Family {F}: joint all-pairs dv/v inversion (2-8 Hz coda, isolated-LFE stacks)')
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_dvv{suf}{tag}.png', dpi=130)


if __name__ == '__main__':
    main()
