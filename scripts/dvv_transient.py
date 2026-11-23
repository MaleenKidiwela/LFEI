#!/usr/bin/env python3
"""Post-M7 dv/v transient test on a family's network dv/v (output of dvv_fit.py).

Model: offset + trend + annual + steps at Petrolia (2021) and Ferndale (2022)
       + A * exp(-(t - t_M7) / tau) for t >= t_M7.
1) Pre-specified tau = 60 d (the value found on family 23): amplitude A +- se (no look-elsewhere
   for families other than 23) -> independent test per family, combinable across families.
2) tau scan (20..365 d and a permanent step) for reference: best tau, A, delta chi2 vs no M7 term.
Output: data/dvv/fam{F}_dvv_transient{tag}.json
"""
import argparse
import json

import numpy as np
import pandas as pd

M7 = pd.Timestamp('2024-12-05T18:44')
STEPS = ['2021-12-20T20:10', '2022-12-20T10:34']


def fit(t, d, err, extra=None):
    yr = (t - pd.Timestamp('2008-01-01')).dt.total_seconds().values / (365.25 * 86400)
    cols = [np.ones(len(t)), yr - yr.mean(), np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)] + \
        [(t >= pd.Timestamp(s)).astype(float).values for s in STEPS]
    if extra is not None:
        cols.append(extra)
    G = np.array(cols).T
    w = 1 / np.maximum(err, 1e-5) ** 2
    W = np.sqrt(w)[:, None]
    m, *_ = np.linalg.lstsq(G * W, d * W[:, 0], rcond=None)
    r = d - G @ m
    chi2 = float(np.sum(w * r ** 2))
    red = chi2 / max(len(d) - len(m), 1)
    cov = np.linalg.pinv((G * W).T @ (G * W)) * max(red, 1)
    return m, np.sqrt(np.diag(cov)), chi2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--tag', default='_iso5_n150')
    ap.add_argument('--tau', type=float, default=60.0)
    a = ap.parse_args()
    F = a.family
    b = pd.read_csv(f'data/dvv/fam{F}_network_dvv{a.tag}.csv')
    b['t'] = pd.to_datetime(b.t, format='mixed')
    b = b.dropna(subset=['dvv', 'err']).sort_values('t')
    t, d, err = b.t, b.dvv.values, b.err.values
    dtd = (t - M7).dt.total_seconds().values / 86400
    n_post = int((dtd >= 0).sum())
    n_post60 = int(((dtd >= 0) & (dtd < 120)).sum())
    out = dict(family=F, n_bins=len(b), n_bins_after_M7=n_post, n_bins_0_120d=n_post60)
    if n_post == 0 or len(b) < 10:
        out['note'] = 'no bins after the M7 (or too few bins)'
        json.dump(out, open(f'data/dvv/fam{F}_dvv_transient{a.tag}.json', 'w'), indent=1)
        print(out)
        return
    _, _, chi0 = fit(t, d, err)
    m, se, chi = fit(t, d, err, np.where(dtd >= 0, np.exp(-np.clip(dtd, 0, None) / a.tau), 0.0))
    out.update(tau_fixed_d=a.tau, A_pct=m[-1] * 100, A_se_pct=se[-1] * 100, A_sigma=m[-1] / se[-1],
               dchi2_fixed=chi0 - chi)
    scan = []
    for tau in [20, 30, 45, 60, 90, 120, 180, 365, np.inf]:
        x = np.where(dtd >= 0, 1.0 if np.isinf(tau) else np.exp(-np.clip(dtd, 0, None) / tau), 0.0)
        mm, ss, cc = fit(t, d, err, x)
        scan.append(dict(tau=str(tau), A_pct=mm[-1] * 100, se_pct=ss[-1] * 100, dchi2=chi0 - cc))
    best = max(scan, key=lambda s: s['dchi2'])
    out.update(best_tau=best['tau'], best_A_pct=best['A_pct'], best_se_pct=best['se_pct'], best_dchi2=best['dchi2'],
               scan=scan)
    json.dump(out, open(f'data/dvv/fam{F}_dvv_transient{a.tag}.json', 'w'), indent=1)
    print(f"family {F}: {len(b)} bins ({n_post} after M7, {n_post60} within 120 d)")
    print(f"  fixed tau {a.tau:.0f} d: A = {out['A_pct']:+.4f} +- {out['A_se_pct']:.4f} % ({out['A_sigma']:+.1f} sigma), "
          f"dchi2 {out['dchi2_fixed']:.1f}")
    print(f"  best tau {best['tau']}: A = {best['A_pct']:+.4f} +- {best['se_pct']:.4f} %, dchi2 {best['dchi2']:.1f}")


if __name__ == '__main__':
    main()
