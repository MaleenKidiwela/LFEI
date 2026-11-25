#!/usr/bin/env python
"""Figures for the N-day BACKWARD (env NBACK, default 5; 5 or 10) moving stacks (day d = CCFs of days d-4..d) of single-station cross-component noise dv/v,
all 27 stations, bands 0.1-0.3, 0.3-0.6, 0.5-1, 1-3, 3-5 Hz.
(1) figures/sci_noise_dvv_sc_5back_all.png: full record, stations ordered by distance to the LFEs (5-day series, thin, + trailing 30-d median)
(2) figures/sci_noise_dvv_sc_5back_m7.png: 2024-10-01..2025-03-01 zoom (5-day series per station + network median)
(3) figures/sci_noise_dvv_sc_5back_{STA}.png: per station, 5 band panels, 5-day (black) over 2-day (grey)
Also prints the M7 step table (5-day backward, days 1..15 minus -30..-1) from sc_steps.csv."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = 'data/sci/noise_dvv_sc_full'
N = int(os.environ.get('NBACK', 5))
BANDS = ['0.1-0.3Hz', '0.3-0.6Hz', '0.5-1Hz', '1-3Hz', '3-5Hz']
OFF = {'0.1-0.3Hz': 0.4, '0.3-0.6Hz': 0.4, '0.5-1Hz': 0.6, '1-3Hz': 0.25, '3-5Hz': 0.5}
M7 = pd.Timestamp('2024-12-05 18:44'); PET = pd.Timestamp('2021-12-20'); FER = pd.Timestamp('2022-12-20')
D5 = pd.read_csv(f'{OUT}/sc_{N}day_back.csv', parse_dates=['t'])
D5 = D5[D5.combo == 'SC']
D2 = pd.read_csv(f'{OUT}/sc_2day_back.csv', parse_dates=['t'], usecols=['t', 'band', 'combo', 'dvv_pct', 'sta'])
D2 = D2[D2.combo == 'SC']
C = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv').groupby(['net', 'sta'])[['lat', 'lon']].first()
LFE = (40.05, -123.56)


def dist(s):
    n, c = s.split('.')
    if (n, c) not in C.index:
        return 999
    la, lo = C.loc[(n, c)]
    return 111.2 * np.hypot(la - LFE[0], (lo - LFE[1]) * np.cos(np.radians(LFE[0])))


stas = sorted(D5.sta.unique(), key=dist)
col = lambda s: 'C0' if s.startswith('PB') else ('C2' if s.startswith('NC') else 'C1')


def marks(ax):
    ax.axvline(M7, color='r', lw=1)
    for t in (PET, FER):
        ax.axvline(t, color='r', ls='--', lw=0.6, alpha=0.6)


# (1) overview
fig, axs = plt.subplots(1, 5, figsize=(26, 16))
for ax, b in zip(axs, BANDS):
    for i, s in enumerate(stas):
        g = D5[(D5.sta == s) & (D5.band == b)].set_index('t').dvv_pct
        ax.text(pd.Timestamp('2008-02-01'), -i * OFF[b] + 0.3 * OFF[b], f'{s.split(".")[1]} {dist(s):.0f} km', fontsize=6.5)
        if g.notna().sum() < 20:
            continue
        g = g - g.median()
        ax.plot(g.index, -i * OFF[b] + g, '-', lw=0.3, color=col(s), alpha=0.35)
        r = g.rolling('30D', min_periods=4).median()
        ax.plot(r.index, -i * OFF[b] + r, '-', lw=1.1, color=col(s))
    marks(ax); ax.grid(alpha=0.3); ax.relim(); ax.autoscale(axis='y')
    ax.set_title(f'{b} (offset {OFF[b]}% per station)', fontsize=9)
    ax.set_yticks([])
    ax.plot([pd.Timestamp('2026-06-01')] * 2, [-(len(stas)) * OFF[b], -(len(stas) - 0.5) * OFF[b]], 'k-', lw=2)
    ax.text(pd.Timestamp('2025-01-01'), -(len(stas) - 0.25) * OFF[b], f'{OFF[b] / 2:g} %', fontsize=7, ha='right')
fig.suptitle(f'Single-station cross-component noise dv/v, {N}-day BACKWARD moving stacks (day d = days d-{N-1}..d): thin = {N}-day, thick = trailing 30-day median; NOT clipped (traces may overlap)\n'
             'stations ordered by distance to the LFEs; blue PB borehole, green NC, orange BK; red = M7, dashed = Petrolia/Ferndale; median removed', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f'figures/sci_noise_dvv_sc_{N}back_all.png', dpi=110); plt.close(fig)

# (2) M7 zoom
x0, x1 = pd.Timestamp('2024-10-01'), pd.Timestamp('2025-03-01')
fig, axs = plt.subplots(1, 5, figsize=(24, 7))
for ax, b in zip(axs, BANDS):
    G = []
    for s in stas:
        g = D5[(D5.sta == s) & (D5.band == b) & (D5.t >= x0) & (D5.t <= x1)].set_index('t').dvv_pct
        pre = g[(g.index >= M7 - pd.Timedelta('30D')) & (g.index < M7)]
        if g.notna().sum() < 30 or pre.notna().sum() < 10:
            continue
        g = g - pre.mean()
        ax.plot(g.index, g, '-', lw=0.6, color=col(s), alpha=0.45)
        G.append(g.rename(s))
    if G:
        med = pd.concat(G, axis=1).median(axis=1)
        ax.plot(med.index, med, 'k-', lw=2.2, label=f'median of {len(G)} stations')
    ax.axvspan(M7, M7 + pd.Timedelta('18D'), color='#fde0dd', zorder=0)
    ax.axvline(M7, color='r', lw=1.2); ax.axhline(0, color='0.5', lw=0.6)
    ax.grid(alpha=0.3); ax.set_title(b, fontsize=10); ax.legend(fontsize=8, loc='lower left')
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
axs[0].set_ylabel('dv/v (%) relative to the 30 days before the M7')
fig.suptitle(f'M7 (red line; shaded = LFE silence): {N}-day backward single-station cross-component dv/v, all stations (blue PB, green NC, orange BK)', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f'figures/sci_noise_dvv_sc_{N}back_m7.png', dpi=120); plt.close(fig)

# (3) per station
for s in stas:
    fig, axs = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
    for ax, b in zip(axs, BANDS):
        g2 = D2[(D2.sta == s) & (D2.band == b)]
        g5 = D5[(D5.sta == s) & (D5.band == b)]
        ax.plot(g2.t, g2.dvv_pct, '-', lw=0.4, color='0.7', label='2-day backward')
        ax.plot(g5.t, g5.dvv_pct, '-', lw=0.7, color='k', label=f'{N}-day backward')
        r = g5.set_index('t').dvv_pct.rolling('30D', min_periods=4).median()
        ax.plot(r.index, r.values, '-', lw=1.6, color='C3', label=f'{N}-day, trailing 30-d median')
        v = g5.dvv_pct.dropna()
        if len(v) > 20:
            lo, hi = v.min(), v.max(); ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        marks(ax); ax.grid(alpha=0.3); ax.set_ylabel('dv/v (%)'); ax.set_title(f'{s} {b}', fontsize=9, loc='left')
    axs[0].legend(fontsize=7, ncol=3, loc='lower left')
    fig.suptitle(f'{s} single-station cross-component noise dv/v: {N}-day backward moving stacks (day d = days d-{N-1}..d)', fontsize=10)
    fig.tight_layout(); fig.savefig(f'figures/sci_noise_dvv_sc_{N}back_{s}.png', dpi=110); plt.close(fig)

S = pd.read_csv(f'{OUT}/sc_steps.csv')
for m in ('2day_back', f'{N}day_back'):
    g = S[(S.event == 'M7') & S.method.str.startswith(m)]
    t = g.pivot_table(index='sta', columns='band', values='step_pct').reindex(columns=BANDS)
    z = g.pivot_table(index='sta', columns='band', values='z_placebo').reindex(columns=BANDS)
    print(m, 'M7 step median %:', t.median().round(3).to_dict(), '| n |z|>3:', (z.abs() > 3).sum().to_dict(),
          '| median placebo sd %:', g.groupby('band').placebo_sd_pct.median().reindex(BANDS).round(3).to_dict())
