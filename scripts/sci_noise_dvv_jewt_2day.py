#!/usr/bin/env python3
"""JEWT ambient-noise dv/v from 2-day moving stacks (CCFs of days d and d+1 averaged, step 1 day) in the three bands
0.1-1 Hz (cross-components only), 1-3 Hz and 3-5 Hz (24-h products), 2024-11-10 to 2025-01-10, reference = pre-M7 mean.
Reuses sci_noise_dvv_jewt.py functions (per-day CCFs in data/sci/noise_dvv_jewt/days/, notch, stretching, pair selection).
Outputs data/sci/noise_dvv_jewt/m7_2day_moving.csv, figures/sci_noise_dvv_jewt_m7_2day.png"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import sci_noise_dvv as M                      # noqa: E402
import sci_noise_dvv_jewt as J                 # noqa: E402

PANELS = [('low', 0, '0.1-1 Hz (cross-components only)'), ('day', 0, '1-3 Hz'), ('day', 1, '3-5 Hz')]
rows = []
for key, b, lab in PANELS:
    P = J.PRODS[key]
    days, ccf = J.load_prod(key, d0=J.LOCAL0, d1=J.LOCAL1)
    keep, _ = J.detect_lines_g(P, days, ccf, 1)
    X = J.notch_g(P, ccf, keep)
    pre = days < J.M7
    with np.errstate(invalid='ignore'):
        ref = np.nanmean(X[pre], axis=0)
    q, good = J.good_pairs(P, days, X, ref, pre, block=5, min_days=3)
    S = np.full((len(days) - 1,) + X.shape[1:], np.nan)
    for i in range(len(days) - 1):
        with np.errstate(invalid='ignore'):
            S[i] = np.nanmean(X[i:i + 2], axis=0)
    dv, cc, er = J.measure_g(P, ref, S)
    erg = np.where(good[None], er, np.nan)
    tt = days[:-1] + pd.Timedelta(hours=24)                 # centre of the 2-day window
    for c, ix in M.COMBOS.items():
        m, e = M.combine(dv[:, :, ix], erg[:, :, ix])
        rows.append(pd.DataFrame(dict(t=tt, band=lab, combo=c, dvv_pct=m[:, b] * 100, err_pct=e[:, b] * 100)))
    for p_ in (3, 4, 5):
        if good[b, p_]:
            rows.append(pd.DataFrame(dict(t=tt, band=lab, combo=M.PAIR_NAMES[p_], dvv_pct=dv[:, b, p_] * 100, err_pct=er[:, b, p_] * 100)))
D = pd.concat(rows)
D.to_csv('data/sci/noise_dvv_jewt/m7_2day_moving.csv', index=False)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, (key, b, lab) in zip(axs, PANELS):
    for c, col, lw in (('EN', 'C0', 0.7), ('EZ', 'C1', 0.7), ('NZ', 'C4', 0.7), ('SC', 'k', 1.6)):
        g = D[(D.band == lab) & (D.combo == c)]
        if g.dvv_pct.notna().sum() == 0:
            continue
        g = g.assign(dvv_pct=g.dvv_pct - g[g.t < J.M7].dvv_pct.mean())
        if c == 'SC':
            ax.errorbar(g.t, g.dvv_pct, g.err_pct, fmt='o-', ms=3.5, lw=lw, color=col, label='cross-components combined', zorder=5)
        else:
            ax.plot(g.t, g.dvv_pct, '.-', ms=2, lw=lw, color=col, alpha=0.8, label=f'{c[0]}-{c[1]}')
    ax.axvline(pd.Timestamp('2024-12-05 18:44'), color='r', lw=1)
    ax.axvspan(J.M7, J.SILENCE_END, color='#fde0dd', zorder=0)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_title(f'BK.JEWT {lab.split(" (")[0]}: single-station cross-component dv/v, 2-day moving stacks', fontsize=9, loc='left')
    ax.set_ylabel('dv/v (%)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=4, loc='lower left')
fig.suptitle('JEWT single-station cross-component (E-N, E-Z, N-Z) ambient-noise dv/v; reference = pre-M7 mean; '
             'red line = M7; shaded = LFE silence', fontsize=9)
fig.tight_layout()
fig.savefig('figures/sci_noise_dvv_jewt_m7_2day_sc.png', dpi=140)
print(D[D.combo == 'SC'].groupby('band').dvv_pct.describe().round(3))
