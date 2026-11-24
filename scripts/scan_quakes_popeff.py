#!/usr/bin/env python3
"""Population-matched detection efficiency from the injection runs.

The injected donors are typical catalog LFEs; many sit below the scan threshold, and events near the
threshold lose relatively more detections in noise than the stronger events that make up the scan's
detected population. Here only donors that are usually detected before the quake (pre-quake injected
detection fraction >= MINPRE under the given rule) are used: their post/pre detection ratio approximates
the efficiency loss for the scan's detected population. Efficiency vs time is interpolated between
injection days (fine bins in the first 4 d) as in scan_quakes_analysis.py, and the efficiency-corrected
rate ratios (FA-corrected net counts) are recomputed.
Output data/scan_quakes/{tag}_popeff.json
"""
import argparse
import json

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--minpre', type=float, default=0.5)
a = ap.parse_args()
base = a.tag.split('_')[0]
import os
pk = f'data/scan_quakes/inject/{base}_inject_v3.pkl'
if not os.path.exists(pk):
    pk = f'data/scan_quakes/inject/{base}_inject.pkl'
tab = pd.read_pickle(pk)['table']          # our own file (scan_quakes_inject.py)
I = pd.read_csv(f'data/scan_quakes/{a.tag}_injection.csv')
tab['ts'] = (pd.to_datetime(tab.t_inj, format='mixed') - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1)
I = I.merge(tab[['ts', 'donor']], on='ts', how='left')
S = json.load(open(f'data/scan_quakes/{a.tag}_summary.json'))
Q = pd.Timestamp(S['quake'])
qs = (Q - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1)
d0 = qs - (qs % 86400)
out = dict(tag=a.tag, minpre=a.minpre)
wins = {'24h': (0, 24), '2d': (0, 48), 'd1_2': (24, 48), 'd2_7': (48, 168), '30d': (0, 720)}
for rn in ['fixed10', 'fixed8', 'daily']:
    col = 'det_' + rn
    pre = I[I.h < 0].groupby('donor')[col].mean()
    keep = pre[pre >= a.minpre].index
    J = I[I.donor.isin(keep)].copy()
    e0 = J[J.h < 0][col].mean()
    fine = []
    for h0, h1 in [(0, 6), (6, 24), (24, 48), (48, 96)]:
        x = J[(J.h >= h0) & (J.h < h1)]
        fine.append((h0, h1, x[col].mean() / e0 if len(x) else np.nan, len(x)))
    L = J[J.h >= 72].copy()
    L['dc'] = (np.floor((L.ts - d0) / 86400) * 86400 + d0 + 43200 - qs) / 3600
    dd = L.groupby('dc')[col].mean() / e0

    def mean_eff(h0w, h1w):
        hs = np.linspace(h0w, h1w, 2000, endpoint=False)
        e = np.ones_like(hs)
        for b0, b1, v, _ in fine:
            e[(hs >= b0) & (hs < b1)] = v
        late = hs >= 96
        if len(dd):
            e[late] = np.interp(hs[late], dd.index.values, dd.values)
        return float(np.nanmean(e))
    r = dict(n_donors=int(len(keep)), n_inj=int(len(J)), pre_eff=float(e0),
             fine_rel_eff={f'{b0}-{b1}h': [round(v, 3), n] for b0, b1, v, n in fine},
             daily_rel_eff={f'{k / 24:.1f}d': round(v, 3) for k, v in dd.items()})
    x = S[rn]
    net_pre = (x['pre_pos'] - x['pre_neg']) / S['T_pre_d']
    for k, (h0, h1) in wins.items():
        me = mean_eff(h0, min(h1, S['T_post_d'] * 24))
        dur = (min(h1, S['T_post_d'] * 24) - h0) / 24
        O = x[k]['pos'] - x[k]['neg']
        E = net_pre * dur * me
        c = S['catalog'][k]
        r[k] = dict(mean_rel_eff=me, observed_net=O, expected_if_unchanged=E, ratio_eff_corr=O / max(E, 1e-9),
                    catalog_ratio_eff_corr=c['ratio'] / max(me, 1e-9),
                    catalog_clean_ratio_eff_corr=c['ratio_clean'] / max(me, 1e-9))
    out[rn] = r
json.dump(out, open(f'data/scan_quakes/{a.tag}_popeff.json', 'w'), indent=1, default=float)
for rn in ['fixed10', 'daily']:
    r = out[rn]
    print(a.tag, rn, 'donors', r['n_donors'], 'pre_eff %.2f' % r['pre_eff'], r['fine_rel_eff'])
    print('   late daily rel eff', r['daily_rel_eff'])
    for k in wins:
        print('  ', k, {kk: round(vv, 2) for kk, vv in r[k].items()})
