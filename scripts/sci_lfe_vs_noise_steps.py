#!/usr/bin/env python
"""Same-station comparison of LFE-coda dv/v (pooled over families, instrument-cleaned, 2-8 Hz; data/dvv/pool_clean)
and single-station cross-component ambient-noise dv/v (10-day backward stacks; 0.1-0.3, 1-3, 3-5 Hz) around the
three quakes and over 2021->2025, with IDENTICAL step definitions: mean(t+30..t+150 d) - mean(t-150..t-10 d),
each within one LFE segment (instrument epoch); placebo SD from the same statistic at 60 season-matched dates
(+-30 d of the event day in other years, both series).
Outputs data/sci/lfe_vs_noise/steps.csv, figures/sci_lfe_vs_noise_{steps,series}.png"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = 'data/sci/lfe_vs_noise'; os.makedirs(OUT, exist_ok=True)
Q = {'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20', 'M7': '2024-12-05'}
rng = np.random.default_rng(3)

# LFE pooled curves (regularised, with breaks at the quakes): station mean over channel files
LFE_SRC = sys.argv[1] if len(sys.argv) > 1 else 'pool'
L = []
if LFE_SRC == 'monthly':   # unregularised monthly bins (no smoothing, no breaks): quake and placebo dates treated identically
    M = pd.read_csv('data/sci/decline/monthly_bins.csv'); M['t'] = pd.to_datetime(M.month) + pd.Timedelta(days=14)
    M['sta'] = M.chan.str.split('.').str[:2].str.join('.'); M['ch'] = M.chan; M['file'] = M.chan
    # instrument windows: drop known bad windows / split at breaks (same as pool_clean V2)
    BRK = {'PB.B933': '2021-11-01', 'BK.BRIC': '2024-05-22', 'BK.RVIT': '2024-06-21', 'BK.BJES': '2024-05-23', 'NC.KMPB': '2023-10-17'}
    for st, d0 in BRK.items():
        m = M.sta == st; M.loc[m & (M.t >= d0), 'file'] = M.loc[m & (M.t >= d0), 'chan'] + '_seg1'
    L = [M[['t', 'dvv_pct', 'sta', 'ch', 'file']]]
for f in (glob.glob('data/dvv/pool_clean/*_curve.csv') if LFE_SRC == 'pool' else []):
    ch = os.path.basename(f).split('_')[0]; sta = '.'.join(ch.split('.')[:2])
    d = pd.read_csv(f, parse_dates=['t']); d['sta'] = sta; d['ch'] = ch; d['file'] = os.path.basename(f); L.append(d)
L = pd.concat(L)
SC = pd.read_csv('data/sci/noise_dvv_sc_full/sc_10day_back.csv', parse_dates=['t'], usecols=['t', 'band', 'combo', 'dvv_pct', 'sta'])
SC = SC[(SC.combo == 'SC') & SC.band.isin(['0.1-0.3Hz', '1-3Hz', '3-5Hz'])]

def step(t, y, te, pre=(-150, -10), post=(30, 150)):
    r = (t - te).dt.days
    a = y[(r >= pre[0]) & (r <= pre[1])]; b = y[(r >= post[0]) & (r <= post[1])]
    return b.mean() - a.mean() if len(a) >= 3 and len(b) >= 3 else np.nan

def series_steps(t, y, te, seg=None):
    s = step(t, y, te)
    pl = []
    for yr in range(2009, 2026):
        for dd in (-30, -15, 0, 15, 30):
            td = pd.Timestamp(f'{yr}-{te.month:02d}-{te.day:02d}') + pd.Timedelta(days=dd)
            if any(abs((td - pd.Timestamp(v)).days) < 200 for v in Q.values()):
                continue
            pl.append(step(t, y, td))
    pl = np.array([p for p in pl if np.isfinite(p)])
    sd = pl.std() if len(pl) >= 5 else np.nan
    return s, sd, len(pl)

rows = []
for sta, g in L.groupby('sta'):
    for q, te in Q.items():
        te = pd.Timestamp(te)
        # LFE: channel-mean per segment, use the segment spanning the event window
        vals = []
        for fn, h in g.groupby('file'):   # one file = one channel/instrument epoch, common reference (segments = quake breaks)
            s, sd, n = series_steps(h.t.reset_index(drop=True), h.dvv_pct.reset_index(drop=True), te)
            if np.isfinite(s): vals.append((s, sd, n))
        if vals:
            s = np.mean([v[0] for v in vals]); sd = np.nanmean([v[1] for v in vals]) / np.sqrt(len(vals))
            rows.append(dict(sta=sta, quake=q, data='LFE coda 2-8 Hz', step=s, plac_sd=sd, z=s / sd if sd > 0 else np.nan))
        for b, h in SC[SC.sta == sta].groupby('band'):
            s, sd, n = series_steps(h.t, h.dvv_pct, te)
            if np.isfinite(s):
                rows.append(dict(sta=sta, quake=q, data=f'noise {b}', step=s, plac_sd=sd, z=s / sd if sd > 0 else np.nan))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/steps{"_monthly" if LFE_SRC == "monthly" else ""}.csv', index=False)
P = R.pivot_table(index=['quake', 'sta'], columns='data', values='step').round(3)
Z = R.pivot_table(index=['quake', 'sta'], columns='data', values='z').round(1)
print(P.to_string()); print(); print(Z.to_string())
for q in Q:
    m = R[R.quake == q].pivot_table(index='sta', columns='data', values='step').dropna()
    for b in ['noise 1-3Hz', 'noise 3-5Hz', 'noise 0.1-0.3Hz']:
        if b in m and len(m) > 3:
            print(f'{q}: LFE vs {b}: n {len(m)} median LFE {m["LFE coda 2-8 Hz"].median():+.3f} median noise {m[b].median():+.3f} r {np.corrcoef(m["LFE coda 2-8 Hz"], m[b])[0,1]:.2f}')

# figure 1: scatter LFE vs noise steps per quake
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, q in zip(axs, Q):
    m = R[R.quake == q].pivot_table(index='sta', columns='data', values='step')
    for b, c in (('noise 1-3Hz', 'C0'), ('noise 3-5Hz', 'C3'), ('noise 0.1-0.3Hz', 'C2')):
        if b in m and 'LFE coda 2-8 Hz' in m:
            mm = m[['LFE coda 2-8 Hz', b]].dropna(); ax.scatter(mm['LFE coda 2-8 Hz'], mm[b], c=c, label=b, s=30)
            for s, r in mm.iterrows():
                ax.annotate(s.split('.')[1], (r['LFE coda 2-8 Hz'], r[b]), fontsize=6)
    lim = 0.6; ax.plot([-lim, lim], [-lim, lim], 'k:', lw=.8); ax.axhline(0, c='.6', lw=.5); ax.axvline(0, c='.6', lw=.5)
    ax.set_xlim(-lim, .2); ax.set_ylim(-lim, .25); ax.set_xlabel('LFE coda step (%)'); ax.set_ylabel('ambient-noise step (%)'); ax.set_title(q); ax.grid(alpha=.3)
axs[0].legend(fontsize=8)
fig.suptitle('Same stations, same step definition (mean 30–150 d after minus 150–10 d before): LFE coda vs ambient noise', fontsize=10)
fig.tight_layout(); fig.savefig(f'figures/sci_lfe_vs_noise_steps{"_monthly" if LFE_SRC == "monthly" else ""}.png', dpi=130)

# figure 2: overlay series at stations with both
stas = [s for s in ['NC.KMPB', 'BK.BRIC', 'PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'NC.KMR', 'BK.JEWT', 'BK.HAYF'] if s in set(L.sta)]
fig, axs = plt.subplots(len(stas), 1, figsize=(13, 2.1 * len(stas)), sharex=True)
for ax, s in zip(np.atleast_1d(axs), stas):
    g = L[L.sta == s]
    for fn, h in g.groupby('file'):
        ax.plot(h.t, h.dvv_pct - h.dvv_pct.median(), 'k-', lw=1, alpha=.6)
    for b, c in (('1-3Hz', 'C0'), ('0.1-0.3Hz', 'C2')):
        h = SC[(SC.sta == s) & (SC.band == b)].set_index('t').dvv_pct
        if len(h): r = h.rolling('60D', min_periods=10).median(); ax.plot(r.index, r - r.median(), c=c, lw=1, label=f'noise {b} (60-d median)')
    for v in Q.values(): ax.axvline(pd.Timestamp(v), c='r', ls='--', lw=.8)
    ax.set_ylabel(s.split('.')[1]); ax.grid(alpha=.3); ax.set_xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2026-12-01'))
np.atleast_1d(axs)[0].plot([], [], 'k-', label='LFE coda 2-8 Hz (' + ('unsmoothed monthly' if LFE_SRC == 'monthly' else 'pooled, smoothed') + ')'); np.atleast_1d(axs)[0].legend(fontsize=7, ncol=3, loc='lower left')
fig.suptitle('LFE-coda vs ambient-noise dv/v at the same stations (median removed; dashed = Petrolia, Ferndale, M7)', fontsize=10)
fig.tight_layout(); fig.savefig(f'figures/sci_lfe_vs_noise_series{"_monthly" if LFE_SRC == "monthly" else ""}.png', dpi=120)
