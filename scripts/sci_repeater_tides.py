#!/usr/bin/env python
"""Tidal (M2, N2, O1, K1, S2) modulation of ordinary repeating earthquakes near the MTJ (353 clusters, NCSN M>=1.5,
2015-2026; data/sci/dvv_deep_shallow/rep/clusters.csv) vs all regional events in the same catalogue and vs the MTJ LFEs
(Shelly v2, declustered per family > 1 h), with the same signed, day-block-debiased estimator as
sci_ducellier_tides_time.py (network and per-cluster/per-family phase-free versions; off-tide 95% null from 50 periods;
day-block bootstrap). Astronomical tidal phase only (no stress model), so it is a rate-modulation test.
Outputs data/sci/repeater_tides/tides.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/seismic/LFE/scripts')
import sci_ducellier_compare as C
import sci_ducellier_tides_time as T
OUT = 'data/sci/repeater_tides'; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)

def est(t, fam, const, perfam):
    ts = (t - C.TREF).dt.total_seconds().to_numpy()
    a = T.amp_signed(*T.parts(ts, T.PER_H[const], fam.to_numpy() if perfam else None))
    nz = [T.amp_signed(*T.parts(ts, p, fam.to_numpy() if perfam else None)) for p in T.OFF]
    # day-block bootstrap (simple: resample days)
    day = np.floor(ts / 86400).astype(np.int64); ud, inv = np.unique(day, return_inverse=True)
    bs = []
    for _ in range(300):
        w = rng.poisson(1.0, len(ud)); idx = np.repeat(np.arange(len(ts)), w[inv])
        if len(idx) < 20: continue
        bs.append(T.amp_signed(*T.parts(ts[idx] + 1e-3 * np.arange(len(idx)) * 0, T.PER_H[const], fam.to_numpy()[idx] if perfam else None)))
    phi = 2 * np.pi * np.mod(ts / 3600.0 / T.PER_H[const], 1.0)
    return dict(const=const, perfam=perfam, n=len(ts), a=a, boot_sd=np.nanstd(bs), null95=np.nanpercentile(np.abs(nz), 95),
                phase_deg=float(np.rad2deg(np.angle(np.exp(1j * phi).sum())) % 360))

c = pd.read_csv('data/sci/dvv_deep_shallow/rep/clusters.csv'); c['t'] = pd.to_datetime(c.Time)
rep = c[c.cl95 >= 0].copy(); rep['fam'] = rep.cl95.astype(str)
allq = c.copy(); allq['fam'] = 'all'
sets = {'repeaters (all)': rep, 'repeaters deep (>=15 km)': rep[rep.depth >= 15], 'repeaters shallow (<15 km)': rep[rep.depth < 15],
        'all catalogue events': allq}
s = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv'); s.columns = [k.lstrip('%') for k in s.columns]
s['t'] = pd.to_datetime(dict(year=s.year, month=s.month, day=s.day)) + pd.to_timedelta(s.s_of_day, unit='s')
fc = 'idnum'
s['fam'] = s[fc].astype(str); s['tsec'] = (s.t - C.TREF).dt.total_seconds()
lfe = C.decluster(s[(s.t >= '2015-01-01')], 'fam', 3600.0)
sets['MTJ LFEs 2015-2026 (declustered)'] = lfe
rows = []
for nm, d in sets.items():
    for const in ('M2', 'N2', 'O1', 'K1', 'S2'):
        for pf in ((False, True) if nm != 'all catalogue events' else (False,)):
            r = est(d.t.reset_index(drop=True), d.fam.reset_index(drop=True), const, pf); r['set'] = nm; rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/tides.csv', index=False)
print(R[['set', 'const', 'perfam', 'n', 'a', 'boot_sd', 'null95', 'phase_deg']].round(3).to_string())
