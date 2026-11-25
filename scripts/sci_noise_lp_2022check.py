#!/usr/bin/env python
"""Check of the 2021->2023 level drop seen in the LP network series (0.05-0.1 Hz Rayleigh ~-0.1%).
Constant-pair subset: usable BB-BB entities with 30-d bins covering both 2019-01..2021-11 (A) and 2023-01..2024-11 (B).
Per entity: mean dv/v in B minus A. Distribution across entities, per station (leave-one-station-out and
per-station median), vs distance, causal vs acausal (source-direction check), and per band.
Outputs data/sci/noise_lp/check2022/{per_entity.csv, per_station.csv, summary.csv}"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import sci_noise_lp_net as N  # noqa: E402

E, data = N.load()
OUT = 'data/sci/noise_lp/check2022'
A = (pd.Timestamp('2019-01-01'), pd.Timestamp('2021-11-30'))
B = (pd.Timestamp('2023-01-01'), pd.Timestamp('2024-11-30'))
rows = []
for _, e in E[E.ok & (E.kind == 'BB-BB')].iterrows():
    d = data[(e.pair, e.comp, e.band)]
    dates = N.DAY0 + pd.to_timedelta(d['dates'], 'D')
    nb = len(d['bin30_full_both_dvv'])
    t30 = pd.date_range(N.DAY0, periods=nb, freq='30D') + pd.Timedelta('15D')
    r = dict(pair=e.pair, a=e.a, b=e.b, comp=e.comp, band=e.band, dist=e.dist)
    for side in ('both', 'caus', 'acaus'):
        v = d[f'bin30_full_{side}_dvv']
        ia = (t30 >= A[0]) & (t30 <= A[1]) & np.isfinite(v)
        ib = (t30 >= B[0]) & (t30 <= B[1]) & np.isfinite(v)
        r[f'nA'], r['nB'] = int(ia.sum()), int(ib.sum())
        r[f'd_{side}'] = (np.nanmean(v[ib]) - np.nanmean(v[ia])) * 100 if ia.sum() >= 15 and ib.sum() >= 10 else np.nan
    rows.append(r)
P = pd.DataFrame(rows).dropna(subset=['d_both'])
P.to_csv(f'{OUT}/per_entity.csv', index=False)
S = []
for band, g in P.groupby('band'):
    for grp, comps in (('Rayl', ['ZZ', 'RR', 'ZR', 'RZ']), ('TT', ['TT'])):
        h = g[g.comp.isin(comps)]
        if len(h) < 3:
            continue
        pm = h.groupby('pair').d_both.mean()
        boot = [np.median(np.random.default_rng(i).choice(pm.values, len(pm))) for i in range(2000)]
        sts = sorted(set(h.a) | set(h.b))
        loo = {s: h[(h.a != s) & (h.b != s)].groupby('pair').d_both.mean().median() for s in sts}
        S.append(dict(band=band, group=grp, n_pairs=len(pm), median=pm.median(), boot_lo=np.percentile(boot, 2.5),
                      boot_hi=np.percentile(boot, 97.5), frac_neg=(pm < 0).mean(),
                      caus=h.groupby('pair').d_caus.mean().median(), acaus=h.groupby('pair').d_acaus.mean().median(),
                      loo_min=min(loo.values()), loo_max=max(loo.values()),
                      r_dist=np.corrcoef(h.groupby('pair').dist.first(), pm)[0, 1]))
S = pd.DataFrame(S)
S.to_csv(f'{OUT}/summary.csv', index=False)
print(S.round(3).to_string())
st = []
g = P[(P.band == '0.05-0.1') & P.comp.isin(['ZZ', 'RR', 'ZR', 'RZ'])]
for s in sorted(set(g.a) | set(g.b)):
    h = g[(g.a == s) | (g.b == s)].groupby('pair').d_both.mean()
    st.append(dict(sta=s, n=len(h), med=h.median()))
st = pd.DataFrame(st).sort_values('med')
st.to_csv(f'{OUT}/per_station.csv', index=False)
print(st.round(3).to_string())
