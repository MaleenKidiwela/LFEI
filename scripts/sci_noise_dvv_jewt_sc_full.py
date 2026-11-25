"""BK.JEWT single-station cross-component (EN, EZ, NZ) ambient-noise dv/v over the whole record
(2023-08-23 onward; JEWT has no earlier data), three bands. Reads the existing products of
sci_noise_dvv_jewt.py (no reprocessing): 10-day stacks per pair, combined SC 10-day, monthly, 5-day.
Output: figures/sci_noise_dvv_jewt_sc_full.png, data/sci/noise_dvv_jewt/jewt_sc_full_summary.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data/sci/noise_dvv_jewt')
FIG = os.path.join(ROOT, 'figures')
PANELS = [('low', '0.1-1Hz', '0.1–1 Hz (microseism band; tracks noise power, not reliable)'),
          ('day', '1-3Hz', '1–3 Hz'), ('day', '3-5Hz', '3–5 Hz')]
PAIRS = {'EN': '#1f77b4', 'EZ': '#ff7f0e', 'NZ': '#9467bd'}
M7 = pd.Timestamp('2024-12-05')
OTHER = {'M4.8 2023-10-16 (46 km)': '2023-10-16', 'M5.1 2026-06-03 (123 km)': '2026-06-03',
         'M5.6 2026-06-24 (85 km)': '2026-06-24'}

T = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_10day.csv'), parse_dates=['t'])
P = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_10day_pairs.csv'), parse_dates=['t'])
Mo = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_monthly.csv'), parse_dates=['month'])
D = pd.read_csv(os.path.join(OUT, 'jewt_noise_dvv_daily_5day.csv'), parse_dates=['t'])
S = pd.read_csv(os.path.join(OUT, 'jewt_noise_steps.csv'))
F = pd.read_csv(os.path.join(OUT, 'jewt_noise_longterm_fit.csv'))

fig, axs = plt.subplots(len(PANELS), 1, figsize=(12, 3.0 * len(PANELS)), sharex=True)
rows = []
for ax, (pk, bn, lab) in zip(axs, PANELS):
    sel = lambda X: X[(X['prod'] == pk) & (X.band == bn)]
    for pr, col in PAIRS.items():
        g = sel(P)[sel(P).combo == pr]
        ax.plot(g.t, g.dvv_pct, '.-', ms=3, lw=0.5, color=col, alpha=0.55, label=f'{pr} (10-day)')
    g5 = sel(D)[(sel(D).combo == 'SC') & (sel(D).block_days == 5)]
    ax.plot(g5.t, g5.dvv_pct, '-', lw=0.4, color='0.55', alpha=0.6, label='combined, 5-day')
    g = sel(T)[sel(T).combo == 'SC']
    ax.errorbar(g.t, g.dvv_pct, yerr=g.err_pct, fmt='o', ms=3.2, lw=0.7, color='k', label='combined, 10-day')
    gm = sel(Mo)[sel(Mo).combo == 'SC']
    ax.plot(gm.month + pd.Timedelta(days=15), gm.dvv_pct, '-', lw=1.8, color='C3', label='combined, monthly')
    ax.axvline(M7, color='r', lw=1.2)
    for v in OTHER.values():
        ax.axvline(pd.Timestamp(v), color='salmon', lw=0.8, ls='--')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.grid(alpha=0.25)
    st = S[(S['prod'] == pk) & (S.band == bn) & (S.combo == 'SC') & (S.event == 'M7')]
    f = F[(F['prod'] == pk) & (F.band == bn) & (F.combo == 'SC')]
    txt = lab
    r = {'band': bn}
    if len(st):
        st = st.iloc[0]
        txt += f'   |   M7 step (30 d after − 30 d before): {st.step_pct:+.3f}% (placebo sd {st.placebo_sd_pct:.3f}, z {st.z_placebo:+.1f})'
        r.update(m7_step_pct=st.step_pct, placebo_sd_pct=st.placebo_sd_pct, z=st.z_placebo)
    if len(f):
        f = f.iloc[0]
        txt += f'\nseasonal amplitude {f.annual_amp_pct:.3f}%, trend {f.trend_per_yr:+.3f} ± {f.trend_per_yr_se:.3f} %/yr'
        r.update(annual_amp_pct=f.annual_amp_pct, trend_pct_per_yr=f.trend_per_yr, trend_se=f.trend_per_yr_se)
    rows.append(r)
    ax.set_title(txt, fontsize=8.5, loc='left')
    lo, hi = np.nanpercentile(sel(P)[sel(P).combo.isin(list(PAIRS))].dvv_pct, [1, 99])
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_ylabel('dv/v (%)')
axs[0].legend(fontsize=7, ncol=6, loc='lower left')
axs[-1].set_xlim(T.t.min() - pd.Timedelta(days=10), T.t.max() + pd.Timedelta(days=10))
fig.suptitle('BK.JEWT single-station cross-component noise dv/v, whole record (station installed Aug 2023)\n'
             'Reference = Aug 2023 – Nov 2024; red line = 2024-12-05 M7, dashed = smaller distant quakes', fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'sci_noise_dvv_jewt_sc_full.png'), dpi=140)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'jewt_sc_full_summary.csv'), index=False)
print(pd.DataFrame(rows).to_string())
