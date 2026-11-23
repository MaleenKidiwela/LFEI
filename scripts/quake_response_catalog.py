#!/usr/bin/env python3
"""Catalog-only LFE response (Shelly v2, all 27 families) to every M>=5.5 within 200 km, 2008-2026.
For each quake and family: gap spanning the quake (days) and its rank among all inter-event gaps
since 2009; 30-d count after vs expected from the preceding 365 d. Output data/reports/quake_response_catalog.csv"""
import numpy as np
import pandas as pd

d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
d['t'] = pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) + pd.to_timedelta(d.s_of_day, unit='s')
q = pd.read_csv('data/catalogs/comcat_M5.5_200km_2008_2026.csv')
q['t'] = pd.to_datetime(q.time).dt.tz_localize(None)
q['dist_km'] = np.hypot((q.latitude - 40.06) * 111.2, (q.longitude + 123.625) * 111.2 * np.cos(np.radians(40.06)))
q = q[(q.t > '2008-02-01') & (q.t < '2026-06-01')].drop_duplicates('t')
rows = []
for _, e in q.iterrows():
    for F, g in d.groupby('idnum'):
        t = g.t.sort_values().reset_index(drop=True)
        t9 = t[t > '2009-01-01'].reset_index(drop=True)
        gaps = t9.diff().dt.total_seconds() / 86400
        i = t9.searchsorted(e.t)
        if i == 0 or i >= len(t9):
            continue
        gap = (t9[i] - t9[i - 1]).total_seconds() / 86400
        rank = int((gaps > gap).sum()) + 1
        pre = ((t > e.t - pd.Timedelta(days=365)) & (t < e.t)).sum()
        post = ((t >= e.t) & (t < e.t + pd.Timedelta(days=30))).sum()
        exp = pre * 30 / 365
        rows.append(dict(quake=f'{e.t:%Y-%m-%d} M{e.mag:.1f}', dist_km=round(e.dist_km), family=F, gap_d=gap,
                         gap_rank=rank, pre365=pre, post30=post, ratio30=post / exp if exp > 0 else np.nan))
r = pd.DataFrame(rows)
r.to_csv('data/reports/quake_response_catalog.csv', index=False)
s = r.groupby(['quake', 'dist_km'], sort=False).agg(
    n_fam=('family', 'size'), fam_gap_top10=('gap_rank', lambda x: int((x <= 10).sum())),
    median_gap_d=('gap_d', 'median'), max_gap_d=('gap_d', 'max'),
    median_ratio30=('ratio30', 'median'), all_post30=('post30', 'sum'), all_pre365=('pre365', 'sum'))
s['all_ratio30'] = s.all_post30 / (s.all_pre365 * 30 / 365)
pd.set_option('display.width', 200)
print(s.drop(columns=['all_post30', 'all_pre365']).round(2).to_string())
