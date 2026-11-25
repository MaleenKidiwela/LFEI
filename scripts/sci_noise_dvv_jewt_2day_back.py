#!/usr/bin/env python3
"""JEWT single-station CROSS-COMPONENT (E-N, E-Z, N-Z) ambient-noise dv/v from 2-day BACKWARD (trailing) moving stacks:
the value at day d is from the CCFs of days d-1 and d averaged (step 1 day), so no later data leaks into day d.
Whole JEWT record (2023-08-23 onward), bands 0.1-1 Hz, 1-3 Hz, 3-5 Hz; reference = mean of all pre-M7 days.
Reuses sci_noise_dvv_jewt.py functions (per-day CCFs in data/sci/noise_dvv_jewt/days/, notch, stretching, pair selection).
Outputs data/sci/noise_dvv_jewt/jewt_2day_back.csv, figures/sci_noise_dvv_jewt_2day_back_full.png and _m7.png"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import sci_noise_dvv as M                      # noqa: E402
import sci_noise_dvv_jewt as J                 # noqa: E402

PANELS = [('low', 0, '0.1-1 Hz'), ('day', 0, '1-3 Hz'), ('day', 1, '3-5 Hz')]
PAIRS = {3: 'EN', 4: 'EZ', 5: 'NZ'}
M7T = pd.Timestamp('2024-12-05 18:44')
rows = []
for key, b, lab in PANELS:
    P = J.PRODS[key]
    days, ccf = J.load_prod(key)                      # whole record
    keep, _ = J.detect_lines_g(P, days, ccf, 1)
    X = J.notch_g(P, ccf, keep)
    pre = days < J.M7
    with np.errstate(invalid='ignore'):
        ref = np.nanmean(X[pre], axis=0)
    q, good = J.good_pairs(P, days, X, ref, pre, block=10, min_days=5)
    S = np.full(X.shape, np.nan)
    for i in range(1, len(days)):
        with np.errstate(invalid='ignore'):
            S[i] = np.nanmean(X[i - 1:i + 1], axis=0)     # days d-1 and d
    dv, cc, er = J.measure_g(P, ref, S)
    sc = [p for p in PAIRS if good[b, p]]
    erg = np.where(good[None], er, np.nan)
    m, e = M.combine(dv[:, b, sc], erg[:, b, sc])
    rows.append(pd.DataFrame(dict(t=days, band=lab, combo='SC', dvv_pct=m * 100, err_pct=e * 100)))
    for p_ in sc:
        rows.append(pd.DataFrame(dict(t=days, band=lab, combo=PAIRS[p_], dvv_pct=dv[:, b, p_] * 100,
                                      err_pct=er[:, b, p_] * 100, cc=cc[:, b, p_])))
    print(lab, 'pairs used', [PAIRS[p] for p in sc], flush=True)
D = pd.concat(rows)
D.to_csv('data/sci/noise_dvv_jewt/jewt_2day_back.csv', index=False)

EVENTS = {'M4.8 (46 km)': '2023-10-16', 'M5.1 (123 km)': '2026-06-03', 'M5.6 (85 km)': '2026-06-24'}


def panel(ax, lab, x0, x1, full):
    for c, col in (('EN', 'C0'), ('EZ', 'C1'), ('NZ', 'C4')):
        g = D[(D.band == lab) & (D.combo == c) & (D.t >= x0) & (D.t <= x1)]
        if g.dvv_pct.notna().any():
            ax.plot(g.t, g.dvv_pct, '-', lw=0.5 if full else 0.8, color=col, alpha=0.6, label=f'{c[0]}-{c[1]}')
    g = D[(D.band == lab) & (D.combo == 'SC') & (D.t >= x0) & (D.t <= x1)]
    if full:
        ax.plot(g.t, g.dvv_pct, '-', lw=0.8, color='k', label='combined (2-day)')
        r = g.set_index('t').dvv_pct.rolling('30D', min_periods=20).median()    # trailing (backward) 30-day median
        ax.plot(r.index, r.values, '-', lw=1.8, color='C3', label='combined, trailing 30-day median')
    else:
        ax.errorbar(g.t, g.dvv_pct, g.err_pct, fmt='o-', ms=3, lw=1.2, color='k', label='combined (2-day)', zorder=5)
        ax.axvspan(J.M7, J.SILENCE_END, color='#fde0dd', zorder=0)
    ax.axvline(M7T, color='r', lw=1.2)
    for v in EVENTS.values():
        if x0 <= pd.Timestamp(v) <= x1:
            ax.axvline(pd.Timestamp(v), color='salmon', ls='--', lw=0.8)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.grid(alpha=0.3)
    v = D[(D.band == lab) & D.combo.isin(['SC', 'EN', 'EZ', 'NZ']) & (D.t >= x0) & (D.t <= x1)].dvv_pct
    lo, hi = np.nanpercentile(v, [1, 99])
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_ylabel('dv/v (%)')
    note = ' (microseism band: follows ocean-noise power; not reliable)' if lab.startswith('0.1') else ''
    ax.set_title(f'BK.JEWT {lab}{note}', fontsize=9, loc='left')


for name, x0, x1, full in (('full', D.t.min(), D.t.max(), True),
                            ('m7', pd.Timestamp('2024-10-01'), pd.Timestamp('2025-03-01'), False)):
    fig, axs = plt.subplots(3, 1, figsize=(13 if full else 11, 8.5), sharex=True)
    for ax, (_, _, lab) in zip(axs, PANELS):
        panel(ax, lab, x0, x1, full)
    axs[0].legend(fontsize=7, ncol=6, loc='lower left')
    axs[-1].set_xlim(x0, x1)
    fig.suptitle('BK.JEWT single-station cross-component noise dv/v, 2-day backward moving stacks (day d = days d-1 and d)\n'
                 'reference = mean of all days before the M7; red line = 2024-12-05 M7'
                 + ('; dashed = smaller distant quakes' if full else '; shaded = LFE silence'), fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f'figures/sci_noise_dvv_jewt_2day_back_{name}.png', dpi=140)
    plt.close(fig)
g = D[D.combo == 'SC'].copy()
g['rel'] = (g.t - J.M7).dt.days
for lab, h in g.groupby('band'):
    pre, post = h[(h.rel >= -30) & (h.rel <= -1)].dvv_pct, h[(h.rel >= 1) & (h.rel <= 15)].dvv_pct
    print(f'{lab}: step (days 1..15 minus -30..-1) {post.mean() - pre.mean():+.3f}%  daily scatter pre {pre.std():.3f}%')
