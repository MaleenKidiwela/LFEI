#!/usr/bin/env python3
"""Long-term dv/v model per station-channel (pooled CWIRE curves, data/dvv/pool/*_curve.csv):
 A: offset + annual + steps at Petrolia/Ferndale/M7 + log-recovery after each quake  [R_q * log(1 + dt/10 d)]
 B: A + linear ramp starting 2021-06-01 (a slow process independent of the quakes)
Compare by AIC; then relate the per-station decline (total 2021->2025 change) and its parts to measured
PGV (data/sci/dvv_depth/pgv_by_station.csv). Output data/dvv/pool/longterm_model.csv, figures/dvv_longterm_model.png"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

Q = {'pet': pd.Timestamp('2021-12-20T20:10'), 'fer': pd.Timestamp('2022-12-20T10:34'), 'm7': pd.Timestamp('2024-12-05T18:44')}
RAMP0 = pd.Timestamp('2021-06-01')


def design(t, ramp):
    yr = (t - pd.Timestamp('2008-01-01')).dt.total_seconds().values / (365.25 * 86400)
    cols, names = [np.ones(len(t)), np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)], ['off', 'as', 'ac']
    for k, q in Q.items():
        dt = (t - q).dt.total_seconds().values / 86400
        cols += [(dt >= 0).astype(float), np.where(dt >= 0, np.log1p(np.clip(dt, 0, None) / 10), 0.0)]
        names += [f'step_{k}', f'rec_{k}']
    if ramp:
        cols.append(np.clip((t - RAMP0).dt.total_seconds().values / (365.25 * 86400), 0, None))
        names.append('ramp_per_yr')
    return np.array(cols).T, names


rows = []
for f in sorted(glob.glob('data/dvv/pool/*_curve.csv')):
    ch = f.split('/')[-1].replace('_curve.csv', '')
    c = pd.read_csv(f)
    c['t'] = pd.to_datetime(c.t)
    c = c[(c.t >= '2009-01-01') & np.isfinite(c.dvv_pct) & (c.err_pct > 0)]
    if c.t.min() > pd.Timestamp('2020-06-01') or c.t.max() < pd.Timestamp('2025-06-01') or len(c) < 40:
        continue
    w = 1 / np.maximum(c.err_pct.values, 0.005) ** 2
    out = dict(chan=ch, type='PB' if ch.startswith('PB') else 'surface', n=len(c))
    for ramp in (False, True):
        G, names = design(c.t, ramp)
        W = np.sqrt(w)[:, None]
        m, *_ = np.linalg.lstsq(G * W, c.dvv_pct.values * W[:, 0], rcond=None)
        r = c.dvv_pct.values - G @ m
        chi = np.sum(w * r ** 2)
        red = chi / (len(c) - len(m))
        cov = np.linalg.pinv((G * W).T @ (G * W)) * max(red, 1)
        out[f'aic_{"B" if ramp else "A"}'] = len(c) * np.log(chi / len(c)) + 2 * len(m)
        if ramp:
            for n_, v, s in zip(names, m, np.sqrt(np.diag(cov))):
                out[n_], out[n_ + '_se'] = v, s
    yrs = c.t.dt.year
    out['decline_2021_2025'] = c[yrs == 2025].dvv_pct.mean() - c[yrs.between(2019, 2020)].dvv_pct.mean()
    rows.append(out)
M = pd.DataFrame(rows)
M['dAIC_B_minus_A'] = M.aic_B - M.aic_A
pgv = pd.read_csv('data/sci/dvv_depth/pgv_by_station.csv').pivot_table(index='sta', columns='quake', values='pgv_gm_cms')
M['sta'] = M.chan.str.rsplit('.', n=2).str[0]
M = M.merge(pgv.add_prefix('pgv_'), left_on='sta', right_index=True, how='left')
M['pgv_sum'] = M[[c for c in M.columns if c.startswith('pgv_') and c != 'pgv_sum']].sum(axis=1, min_count=1)
M.to_csv('data/dvv/pool/longterm_model.csv', index=False)


def wm(g, c):
    w = 1 / g[c + '_se'] ** 2
    m = (g[c] * w).sum() / w.sum()
    chi = ((g[c] - m) ** 2 * w).sum() / max(len(g) - 1, 1)
    return m, np.sqrt(max(chi, 1)) / np.sqrt(w.sum())


print(f'{len(M)} channels with records spanning 2020-2025')
for t, g in M.groupby('type'):
    print(f'== {t} ({len(g)} ch): ramp model preferred (dAIC<-2) in {(g.dAIC_B_minus_A < -2).sum()}; median decline 2019/20->2025 {g.decline_2021_2025.median():+.3f}%')
    for c in ('ramp_per_yr', 'step_pet', 'step_fer', 'step_m7', 'rec_pet', 'rec_fer', 'rec_m7'):
        m, s = wm(g, c)
        print(f'   {c:12s} {m:+.4f} ± {s:.4f}')
ok = M.dropna(subset=['pgv_sum'])
for y in ('decline_2021_2025', 'ramp_per_yr'):
    rr, p = stats.spearmanr(np.log10(ok.pgv_sum), ok[y])
    print(f'{y} vs log10(sum PGV): spearman {rr:+.2f} p={p:.3f} (n={len(ok)})')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for t, c in (('PB', 'C0'), ('surface', 'C3')):
    g = ok[ok.type == t]
    ax[0].scatter(g.pgv_sum, g.decline_2021_2025, color=c, label=t)
    ax[1].errorbar(g.pgv_sum, g.ramp_per_yr, yerr=g.ramp_per_yr_se, fmt='o', color=c, label=t)
for a, lab in zip(ax, ('change 2019/20 -> 2025 (%)', 'post-2021 ramp (%/yr) after steps + recovery')):
    a.set_xscale('log')
    a.set_xlabel('sum of measured PGV at the station, 3 quakes (cm/s)')
    a.set_ylabel(lab)
    a.axhline(0, color='k', lw=0.5)
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig('figures/dvv_longterm_model.png', dpi=130)
