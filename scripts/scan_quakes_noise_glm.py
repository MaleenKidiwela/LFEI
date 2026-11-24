#!/usr/bin/env python3
"""Catalog daily LFE counts vs daily noise: does the post-quake drop survive a noise covariate?

Poisson regression  log E[n_d] = a + b*log10(RMS_d) + c*post_d  over days -30..+29 (catalog v2, events
with an NCSN earthquake 15-65 s before them removed), RMS_d = daily median of the hourly median-channel
2-8 Hz RMS relative to pre-quake (fixed channel set scan). 95% CI by bootstrap over days (300 reps).
Output data/scan_quakes/noise_glm.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from obspy import read_events
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from scan_quakes_analysis import QUAKES, catalog, ep  # noqa: E402


def fit(n, X):
    def nll(b):
        eta = np.clip(X @ b, -30, 30)
        return np.sum(np.exp(eta) - n * eta)
    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(max(n.mean(), 0.1))
    return minimize(nll, b0, method='BFGS').x


rows = []
c = catalog()
rng = np.random.default_rng(0)
for base in ['petrolia', 'ferndale', 'm7']:
    Q = pd.Timestamp(QUAKES[base][0])
    h = pd.read_csv(f'data/scan_quakes/{base}_fc_hourly_threshold.csv')
    h['d'] = np.floor(h.hours_since / 24).astype(int)
    r = h.groupby('d').chan_rms_rel_med.median()
    evc = read_events(f'data/scan_quakes/ncsn_{base}.xml')
    eqt = np.sort([(e.preferred_origin() or e.origins[0]).time.timestamp for e in evc])
    cc = c[(c.t >= Q - pd.Timedelta(days=30)) & (c.t < Q + pd.Timedelta(days=30))].copy()
    cc = cc[~(np.searchsorted(eqt, cc.ts.values - 15) > np.searchsorted(eqt, cc.ts.values - 65))]
    cc['d'] = np.floor((cc.t - Q).dt.total_seconds() / 86400).astype(int)
    n = cc.groupby('d').size().reindex(range(-30, 30), fill_value=0)
    df = pd.DataFrame(dict(n=n, lr=np.log10(r.reindex(n.index)))).dropna()
    for lab, sub in [('all days', df), ('excluding days 0-1', df[(df.index < 0) | (df.index >= 2)]),
                     ('days 2-29 vs pre', df[(df.index < 0) | (df.index >= 2)])]:
        if lab == 'days 2-29 vs pre' and base != 'm7':
            continue
        post = (sub.index >= 0).astype(float)
        X1 = np.c_[np.ones(len(sub)), post]
        X2 = np.c_[np.ones(len(sub)), sub.lr.values, post]
        b1, b2 = fit(sub.n.values, X1), fit(sub.n.values, X2)
        bs1, bs2 = [], []
        for _ in range(300):
            k = rng.integers(0, len(sub), len(sub))
            if post[k].sum() == 0 or post[k].sum() == len(k):
                continue
            bs1.append(fit(sub.n.values[k], X1[k])[1])
            bs2.append(fit(sub.n.values[k], X2[k]))
        bs2 = np.array(bs2)
        # noise slope from PRE-quake days only (avoids confounding with the quake), then observed/predicted post
        pre_ = post == 0
        bp = fit(sub.n.values[pre_], np.c_[np.ones(pre_.sum()), sub.lr.values[pre_]])
        pred_post = np.exp(bp[0] + bp[1] * sub.lr.values[~pre_]).sum()
        pre_mean = np.exp(bp[0] + bp[1] * sub.lr.values[pre_]).sum() / pre_.sum()
        rows.append(dict(quake=base, days=lab, n_days=len(sub),
                         pre_only_noise_slope=np.exp(bp[1]),
                         obs_over_pred_post_preslope=sub.n.values[~pre_].sum() / pred_post,
                         noise_only_pred_post_factor=pred_post / (~pre_).sum() / pre_mean,
                         post_factor_no_noise=np.exp(b1[1]),
                         post_factor_no_noise_ci=np.exp(np.percentile(bs1, [2.5, 97.5])).round(2).tolist(),
                         post_factor_with_noise=np.exp(b2[2]),
                         post_factor_with_noise_ci=np.exp(np.percentile(bs2[:, 2], [2.5, 97.5])).round(2).tolist(),
                         rate_factor_per_decade_noise=np.exp(b2[1]),
                         rate_factor_per_decade_noise_ci=np.exp(np.percentile(bs2[:, 1], [2.5, 97.5])).round(2).tolist(),
                         median_rms_pre=float(10 ** sub.lr[sub.index < 0].median()),
                         median_rms_post=float(10 ** sub.lr[sub.index >= 0].median())))
out = pd.DataFrame(rows)
out.to_csv('data/scan_quakes/noise_glm.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(2).to_string(index=False))
