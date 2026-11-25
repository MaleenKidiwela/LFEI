#!/usr/bin/env python3
"""Figures + summary for the long-period inter-station noise dv/v (sci_noise_lp*.py).

figures/sci_noise_lp_map.png         stations, usable pairs (0.1-0.2 Hz ZZ), LFE area
figures/sci_noise_lp_ccf.png         reference CCFs (ZZ, TT) vs distance per band, coda windows
figures/sci_noise_lp_timeseries.png  network 30-d dv/v (Rayleigh comps, TT) per band, 10-d overlay
figures/sci_noise_lp_steps.png       event steps vs placebo distributions + bootstrap CI (W = 30 d)
figures/sci_noise_lp_depth.png       depth kernels, detection probability vs injected step, depth-converted limits
data/sci/noise_lp/depth_limits.csv, summary.json
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = '/home/seismic/LFE'
OUT = os.path.join(ROOT, 'data/sci/noise_lp')
FIG = os.path.join(ROOT, 'figures')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from sci_noise_lp_dvv import BANDS, EVENTS, DVVD, INJ  # noqa: E402

EVC = {'M6.5_2010': '#7f7f7f', 'M6.8_2014': '#9467bd', 'Petrolia_2021': '#1f77b4', 'Ferndale_2022': '#2ca02c',
       'M7_2024': '#d62728'}


def kern_frac(band, group, col='f_10_30', variant='furlong'):
    F = pd.read_csv(os.path.join(OUT, 'kernel_fractions.csv'))
    F = F[(F.variant == variant) & (F.band == band)].set_index('wave')[col]
    if group == 'TT':
        return F['love']
    if group in ('ZZ', 'RR', 'ZR', 'Rayl'):
        return F['rayleigh']
    return (4 * F['rayleigh'] + F['love']) / 5


def fig_map():
    S = pd.read_csv(os.path.join(OUT, 'sites.csv'))
    Q = pd.read_csv(os.path.join(OUT, 'pair_qc.csv'))
    P = pd.read_csv(os.path.join(OUT, 'pairs.csv'))
    C = pd.read_csv(os.path.join(ROOT, 'data/maps/coast_ne10m_mtj.csv'))
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for _, g in C.groupby('seg'):
        ax.plot(g.lon, g.lat, 'k-', lw=0.6)
    q = Q[Q.ok & (Q.band == '0.1-0.2') & (Q.comp == 'ZZ')]
    for _, r in q.iterrows():
        p = P[(P.a == r.a) & (P.b == r.b)].iloc[0]
        sa, sb = S[S.site == r.a].iloc[0], S[S.site == r.b].iloc[0]
        ax.plot([sa.lon, sb.lon], [sa.lat, sb.lat], '-', lw=0.4, alpha=0.5,
                color='tab:blue' if p.kind == 'BB-BB' else 'tab:orange')
    for _, s in S.iterrows():
        bb = s.kind == 'broadband'
        ax.plot(s.lon, s.lat, '^' if bb else 'v', ms=7, color='k' if bb else 'tab:orange', mec='w')
        ax.text(s.lon + 0.02, s.lat + 0.02, s.site, fontsize=6.5)
    ax.plot(-123.62, 40.06, '*', ms=16, color='tab:red', mec='k', label='LFE families (~28 km depth)')
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(-123.62 + 80 / 85.1 * np.cos(th), 40.06 + 80 / 111.2 * np.sin(th), 'k:', lw=0.8, label='80 km')
    ax.plot([], [], '-', color='tab:blue', label='usable BB-BB pair (0.1-0.2 Hz ZZ)')
    ax.plot([], [], '-', color='tab:orange', label='usable pair with PB geophone')
    ax.set_xlim(-124.7, -122.5)
    ax.set_ylim(39.3, 41.0)
    ax.set_aspect(1 / np.cos(np.radians(40)))
    ax.legend(loc='lower left', fontsize=7)
    ax.set_title('Long-period inter-station noise dv/v: stations and pairs')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_lp_map.png'), dpi=150)
    plt.close(fig)


def fig_ccf():
    Q = pd.read_csv(os.path.join(OUT, 'pair_qc.csv'))
    fig, axs = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    lag = np.arange(-300, 301)
    for j, band in enumerate(BANDS):
        for i, comp in enumerate(['ZZ', 'TT']):
            ax = axs[i, j]
            q = Q[Q.ok & (Q.band == band) & (Q.comp == comp) & (Q.kind == 'BB-BB')].sort_values('dist')
            for _, r in q.iterrows():
                Z = np.load(os.path.join(DVVD, r.pair + '.npz'))
                x = Z[f'{comp}|{band}|ref_full'].astype(float)
                x = x / (np.abs(x).max() + 1e-30)
                ax.plot(lag, r.dist + 3.0 * x, 'k-', lw=0.3)
                ax.plot([r.t1, r.t2], [r.dist, r.dist], '-', color='tab:red', lw=1.0, alpha=0.4)
                ax.plot([-r.t2, -r.t1], [r.dist, r.dist], '-', color='tab:red', lw=1.0, alpha=0.4)
            ax.set_title(f'{comp} {band} Hz (n={len(q)})', fontsize=9)
            ax.set_xlim(-300, 300)
            if j == 0:
                ax.set_ylabel('distance (km)')
            if i == 1:
                ax.set_xlabel('lag (s)')
    fig.suptitle('Reference CCFs (full-record mean, usable BB pairs); red = coda windows')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_lp_ccf.png'), dpi=130)
    plt.close(fig)


def fig_timeseries():
    S30 = pd.read_csv(os.path.join(OUT, 'network_series_30d.csv'), parse_dates=['t'])
    S10 = pd.read_csv(os.path.join(OUT, 'network_series_10d.csv'), parse_dates=['t'])
    fig, axs = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, band in zip(axs, BANDS):
        for grp, col, off in [('Rayl', 'tab:blue', 0.0), ('TT', 'tab:green', -0.3)]:
            kind = 'BB'
            for nb, S, lw, a in [(10, S10, 0.6, 0.35), (30, S30, 1.4, 1.0)]:
                g = S[(S.band == band) & (S.group == grp) & (S.kind == kind) & (S.ref == 'full') & (S.side == 'both')]
                g = g.sort_values('t')
                if g.empty:
                    continue
                y = g.dvv * 100 + off
                ax.plot(g.t, y, '-', color=col, lw=lw, alpha=a,
                        label=f'{grp} {nb}-d' + (f' (offset {off:+.1f}%)' if off else ''))
                if nb == 30:
                    ax.fill_between(g.t, y - 100 * g.err, y + 100 * g.err, color=col, alpha=0.2, lw=0)
        for ev, d in EVENTS.items():
            ax.axvline(pd.Timestamp(d), color=EVC[ev], lw=1, ls='--')
        ax.set_ylabel(f'dv/v (%)  {band} Hz')
        ax.set_ylim(-0.75, 0.45)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=4, loc='lower left')
    axs[0].set_title('Inter-station LP noise dv/v (broadband pairs, coda, ref = full mean); dashed = M6.5 2010, '
                     'M6.8 2014, Petrolia, Ferndale, M7')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_lp_timeseries.png'), dpi=130)
    plt.close(fig)


def fig_steps(boots=None):
    T = pd.read_csv(os.path.join(OUT, 'steps.csv'))
    T = T[(T.W == 30) & (T.side == 'both') & (T.group == 'all') & (T.kind == 'BB')]
    fig, axs = plt.subplots(len(BANDS), len(EVENTS), figsize=(15, 8), sharex=True)
    for i, band in enumerate(BANDS):
        for j, ev in enumerate(EVENTS):
            ax = axs[i, j]
            r = T[(T.band == band) & (T.event == ev)]
            if r.empty or not np.isfinite(r.iloc[0].get('plac_sd', np.nan)):
                ax.text(0.5, 0.5, 'no usable pairs', ha='center', transform=ax.transAxes, fontsize=8)
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(ev, fontsize=9)
                continue
            r = r.iloc[0]
            # placebo distribution approximated by stored percentiles -> draw normal of plac_sd for display
            xs = np.linspace(-0.5, 0.5, 400)
            ax.plot(xs, np.exp(-0.5 * ((xs - 100 * r.plac_med) / (100 * r.plac_sd)) ** 2), '-', color='0.5')
            ax.axvspan(100 * r.plac_p2_5, 100 * r.plac_p97_5, color='0.85')
            ax.axvline(100 * r.step, color=EVC[ev], lw=2)
            if np.isfinite(r.get('boot_p2_5', np.nan)):
                ax.plot([100 * r.boot_p2_5, 100 * r.boot_p97_5], [0.5, 0.5], '-', color=EVC[ev], lw=4, alpha=0.5)
            ax.text(0.02, 0.95, f'{100 * r.step:+.3f}%\npct {r.pct_signed:.0f}\nn={r.n_pairs} pairs',
                    transform=ax.transAxes, va='top', fontsize=7)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(ev, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{band} Hz')
            if i == len(BANDS) - 1:
                ax.set_xlabel('30-d step dv/v (%)')
    fig.suptitle('Event steps (BB pairs, all components, W=30 d) vs placebo 95% band (grey) and bootstrap 95% CI (bar)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_lp_steps.png'), dpi=130)
    plt.close(fig)


def depth_limits():
    T = pd.read_csv(os.path.join(OUT, 'steps.csv'))
    T = T[(T.side == 'both')]
    rows = []
    for _, r in T.iterrows():
        if not np.isfinite(r.get('plac_sd', np.nan)):
            continue
        ub = abs(r.step) + 1.96 * r.plac_sd
        rec = dict(band=r.band, kind=r.kind, group=r.group, W=r.W, event=r.event, n_pairs=r.n_pairs, step=r.step,
                   plac_sd=r.plac_sd, pct_signed=r.pct_signed, bound95_obs=ub, det90_obs=r.get('det90_pct', np.nan) / 100)
        for col, lab in [('f_10_30', '10_30'), ('f_0_10', '0_10'), ('f_5_30', '5_30'), ('f_10_20', '10_20'),
                         ('f_20_30', '20_30')]:
            f = kern_frac(r.band, r.group, col)
            rec[f'F_{lab}'] = f
            rec[f'bound95_{lab}'] = ub / f
            rec[f'det90_{lab}'] = rec['det90_obs'] / f
        rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, 'depth_limits.csv'), index=False)
    return D


def fig_depth(D):
    K = pd.read_csv(os.path.join(OUT, 'kernels.csv'))
    K = K[K.variant == 'furlong']
    T = pd.read_csv(os.path.join(OUT, 'steps.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
    ax = axs[0]
    cols = {'0.05-0.1': 'tab:red', '0.1-0.2': 'tab:blue', '0.2-0.3': 'tab:green'}
    for band in BANDS:
        for wave, ls in [('rayleigh', '-'), ('love', '--')]:
            g = K[(K.band == band) & (K.wave == wave)]
            ax.plot(g.K, 0.5 * (g.z_top + g.z_bot), ls, color=cols[band], label=f'{wave} {band} Hz')
    ax.axhspan(10, 30, color='0.9')
    ax.set_ylim(60, 0)
    ax.set_xlabel('coda dv/v kernel (per km)')
    ax.set_ylabel('depth (km)')
    ax.legend(fontsize=7)
    ax.set_title('Depth sensitivity (Furlong 1-D, fundamental modes)')
    ax = axs[1]
    t = T[(T.W == 30) & (T.side == 'both') & (T.group == 'all') & (T.kind == 'BB')]
    for _, r in t.iterrows():
        P = [r.get(f'P_det_{d * 100:.2f}', np.nan) for d in INJ]
        ax.plot(np.abs(INJ) * 100, P, 'o-', color=cols[r.band], alpha=0.8, ms=3,
                label=f'{r.band} {r.event}')
    ax.axhline(0.9, color='k', ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('injected 30-d velocity drop (%)')
    ax.set_ylabel('P(detect) vs placebo 5th pct')
    ax.legend(fontsize=5.5, ncol=2)
    ax.set_title('Injection tests (real data, placebo dates)')
    ax = axs[2]
    d = D[(D.W == 30) & (D.group == 'all') & (D.kind == 'BB')]
    evs = list(EVENTS)
    for k, band in enumerate(BANDS):
        g = d[d.band == band].set_index('event').reindex(evs)
        x = np.arange(len(evs)) + (k - 1) * 0.25
        ax.bar(x, 100 * g.bound95_10_30, width=0.25, color=cols[band], alpha=0.7, label=f'{band} Hz: 95% bound')
        ax.plot(x, 100 * g.det90_10_30, 'k_', ms=10)
    ax.set_xticks(np.arange(len(evs)))
    ax.set_xticklabels(evs, rotation=20, fontsize=7)
    ax.set_yscale('log')
    ax.set_ylabel('|dv/v| in 10-30 km (%)')
    ax.legend(fontsize=7)
    ax.set_title('Limits on a step confined to 10-30 km (bars: 95% bound, -: 90% detection)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_noise_lp_depth.png'), dpi=130)
    plt.close(fig)


def joint_depth():
    """2-layer separation per event: s_b = F_b(0-10) d_shallow + F_b(10-30) d_deep (+ F_b(>30) * 0), WLS over the
    bands with errors = placebo SD; also for the long-term trend. Uses group 'all' BB (mixed kernel) and 'Rayl'."""
    T = pd.read_csv(os.path.join(OUT, 'steps.csv'))
    TR = pd.read_csv(os.path.join(OUT, 'trend.csv'))
    rows = []
    for group in ['all', 'Rayl', 'TT']:
        for W in [30, 60]:
            for ev in EVENTS:
                t = T[(T.W == W) & (T.side == 'both') & (T.group == group) & (T.kind == 'BB') & (T.event == ev)]
                t = t[np.isfinite(t.plac_sd)] if 'plac_sd' in t else t.iloc[0:0]
                if len(t) < 2:
                    continue
                G = np.array([[kern_frac(b, group, 'f_0_10'), kern_frac(b, group, 'f_10_30')] for b in t.band])
                y = t.step.values
                e = t.plac_sd.values
                A = G / e[:, None]
                C = np.linalg.inv(A.T @ A)
                m = C @ A.T @ (y / e)
                rows.append(dict(kind='step', group=group, W=W, event=ev, bands=';'.join(t.band), d_shallow=m[0],
                                 d_shallow_se=np.sqrt(C[0, 0]), d_deep=m[1], d_deep_se=np.sqrt(C[1, 1]),
                                 corr=C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])))
        t = TR[(TR.group == group) & (TR.kind == 'BB') & (TR.ref == 'full') & (TR.side == 'both')]
        if len(t) >= 2:
            G = np.array([[kern_frac(b, group, 'f_0_10'), kern_frac(b, group, 'f_10_30')] for b in t.band])
            e = t.trend_per_yr_se.values
            A = G / e[:, None]
            C = np.linalg.inv(A.T @ A)
            m = C @ A.T @ (t.trend_per_yr.values / e)
            rows.append(dict(kind='trend_per_yr', group=group, bands=';'.join(t.band), d_shallow=m[0],
                             d_shallow_se=np.sqrt(C[0, 0]), d_deep=m[1], d_deep_se=np.sqrt(C[1, 1]),
                             corr=C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])))
    J = pd.DataFrame(rows)
    J.to_csv(os.path.join(OUT, 'joint_depth.csv'), index=False)
    return J


def summary(D):
    T = pd.read_csv(os.path.join(OUT, 'steps.csv'))
    TR = pd.read_csv(os.path.join(OUT, 'trend.csv'))
    Q = pd.read_csv(os.path.join(OUT, 'pair_qc.csv'))
    F = pd.read_csv(os.path.join(OUT, 'kernel_fractions.csv'))
    out = dict(
        n_sites=int(pd.read_csv(os.path.join(OUT, 'sites.csv')).shape[0]),
        usable_entities={f'{b}|{c}': int(n) for (b, c), n in Q[Q.ok].groupby(['band', 'comp']).size().items()},
        usable_pairs_BB={b: int(Q[Q.ok & (Q.band == b) & (Q.kind == 'BB-BB')].pair.nunique()) for b in BANDS},
        usable_pairs_all={b: int(Q[Q.ok & (Q.band == b)].pair.nunique()) for b in BANDS},
        kernel_fractions=F[F.variant == 'furlong'].round(3).to_dict(orient='records'),
        steps_W30_all_BB=T[(T.W == 30) & (T.side == 'both') & (T.group == 'all') & (T.kind == 'BB')].round(5).to_dict(orient='records'),
        depth_limits_W30_all_BB=D[(D.W == 30) & (D.group == 'all') & (D.kind == 'BB')].round(5).to_dict(orient='records'),
        joint_depth=pd.read_csv(os.path.join(OUT, 'joint_depth.csv')).round(5).to_dict(orient='records'),
        trend=TR[(TR.side == 'both') & (TR.ref == 'full') & (TR.group.isin(['all', 'Rayl', 'TT']))].round(5).to_dict(orient='records'),
    )
    json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    fig_map()
    fig_ccf()
    fig_timeseries()
    fig_steps()
    D = depth_limits()
    fig_depth(D)
    joint_depth()
    summary(D)
