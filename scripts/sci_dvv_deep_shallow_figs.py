#!/usr/bin/env python3
"""Figures for the deep/shallow dv/v separation (scripts/sci_dvv_deep_shallow.py outputs).
figures/sci_dvv_deep_shallow_{directS,deep,clock,lapse,repeaters}.png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/scripts')
OUT = 'data/sci/dvv_deep_shallow'
FIG = 'figures'


def fig_directS():
    import sci_dvv_deep_shallow as m
    ST = pd.read_csv(f'{OUT}/station_directS_coda.csv')
    O = pd.read_csv(f'{OUT}/stage2_deep.csv')
    geo = m.get_geo('ref', '100')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, X, lab in zip(axs, ['D', 'm7'], ['decline 2021->2025', 'M7 step']):
        S = ST[(ST.X == X) & np.isfinite(ST.coda_full_pct)]
        se_d = np.fmax(S.dS_se_boot_ms, S.dS_se_plac_ms.fillna(0))
        se_c = np.fmax(S.coda_full_se_boot_pct, S.coda_full_se_plac_pct.fillna(0))
        bh = S.sensor_km > 0
        for msk, c, nm in [(~bh, 'C0', 'surface'), (bh & (S.sensor_km < 0.3), 'C1', 'borehole <0.3 km'),
                           (bh & (S.sensor_km >= 0.3), 'C3', 'borehole 0.41-0.59 km')]:
            ax.errorbar(S.coda_full_pct[msk], S.dS_ms[msk], xerr=se_c[msk], yerr=se_d[msk], fmt='o', color=c, label=nm,
                        ms=5, lw=0.8)
            for _, r in S[msk].iterrows():
                ax.annotate(r.sta.split('.')[1], (r.coda_full_pct, r.dS_ms), fontsize=7, xytext=(3, 3),
                            textcoords='offset points')
        x = np.linspace(min(-0.35, S.coda_full_pct.min()), max(0.15, S.coda_full_pct.max()), 10)
        r_s = geo.tsh('NC.KMR', 0, 0.3) / geo.Ks('NC.KMR', 0, 0.3)
        ax.plot(x, -r_s * x * 10, 'C0--', lw=1, label=f'surface, change in top 0.3 km (body-wave kernel, {r_s:.1f} s per unit)')
        ax.plot(x, 0 * x, 'C3:', lw=1, label='borehole, change above sensor')
        q = O[(O.X == X) & (O.model == 'ratio')]
        if len(q) and 'beta' in q:
            b = q.beta.values[0]
            ax.plot(x, b * x * 10, 'k-', lw=1, label=f'fit: {-b:.1f} s per unit coda dv/v')
        ax.axhline(0, color='0.7', lw=0.5)
        ax.axvline(0, color='0.7', lw=0.5)
        ax.set_xlabel('coda dv/v change (%, S-aligned, 2-8 Hz, 3 s - noise)')
        ax.set_ylabel('relative direct-S delay change (ms; + = later)')
        ax.set_title(lab)
        ax.legend(fontsize=7, loc='best')
    fig.suptitle('Direct-S delay vs coda dv/v per station (relative to the long-record-station mean; all families)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_deep_shallow_directS.png', dpi=130)
    plt.close(fig)


def fig_deep():
    E = pd.read_csv(f'{OUT}/deep_estimates.csv')
    Z = pd.read_csv(f'{OUT}/synthetic_recovery.csv') if os.path.exists(f'{OUT}/synthetic_recovery.csv') else None
    fig, axs = plt.subplots(1, 3 if Z is not None else 2, figsize=(16, 5.5))
    for ax, X in zip(axs[:2], ['m7', 'D']):
        q = E[(E.X == X) & (E.model.isin(['DD', 'ratio', 'ratio2', 'ratio_bh']) |
                            ((E.model == 'joint') & (E.variant == 'ref/l100')))].reset_index(drop=True)
        lab = [f"joint kernel ref h={r.h_km}" if r.model == 'joint' else
               {'DD': 'DD (station terms free)', 'ratio': 'empirical ratio', 'ratio2': 'empirical ratio surf/bh',
                'ratio_bh': 'boreholes only'}[r.model] for r in q.itertuples()]
        y = np.arange(len(q))
        ax.errorbar(q.dd_pct, y, xerr=1.96 * q.se_honest, fmt='o', ms=4, color='k', lw=0.8,
                    label='95% (honest SE = max(bootstrap, robust placebo))')
        ax.errorbar(q.dd_pct, y + 0.2, xerr=1.96 * q.se_plac_all, fmt='none', color='0.6', lw=0.6,
                    label='95% with all 15 placebos')
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=7)
        ax.axvline(0, color='r', lw=0.6)
        ref = {'m7': (-0.039, 0.029, 'coda common mode (source-receiver)'), 'D': (-0.067, 0.028, 'coda common mode')}[X]
        ax.axvspan(ref[0] - ref[1], ref[0] + ref[1], color='C1', alpha=0.15, label=ref[2])
        ax.set_xlabel('deep (z >= 1 km) dv/v (%)')
        ax.set_title({'m7': 'M7 step: deep layer from direct S', 'D': 'decline 2021->2025: deep layer from direct S'}[X])
        ax.set_xlim(-0.2, 0.15)
        ax.legend(fontsize=6, loc='upper left')
    if Z is not None:
        ax = axs[2]
        Z = Z[Z.scenario != 'baseline_no_injection']
        mk = {'DD': 's', 'ratio': '^', 'ratio2': 'v', 'joint': 'o'}
        for i, (k, g) in enumerate(Z.groupby('scenario')):
            for r in g.itertuples():
                xoff = {'m7': -0.004, 'D': 0.004}[r.X]
                ax.plot(r.true_dd_pct + xoff, r.rec_minus_baseline_pct, mk[r.model], color=f'C{i % 10}', ms=4,
                        mfc='none' if r.X == 'm7' else f'C{i % 10}')
            ax.plot([], [], 'o', color=f'C{i % 10}', label=k)
        lim = [-0.16, 0.02]
        ax.plot(lim, lim, 'k-', lw=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(-0.3, 0.15)
        ax.set_xlabel('true deep dv/v (%)  (open = M7, filled = decline)')
        ax.set_ylabel('recovered minus no-injection baseline (%)')
        ax.legend(fontsize=6)
        ax.set_title('synthetic recovery [s DD, ^ ratio, v ratio2, o joint]', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_deep_shallow_deep.png', dpi=130)
    plt.close(fig)


def fig_clock():
    M = pd.read_csv(f'{OUT}/clock_monthly.csv', parse_dates=['mon'])
    stas = ['NC.KMR', 'NC.KMPB', 'PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B934', 'BK.BRIC',
            'BK.RBOW', 'BK.HAYF', 'BK.RVIT']
    fig, axs = plt.subplots(len(stas), 1, figsize=(11, 1.4 * len(stas)), sharex=True)
    for ax, s in zip(axs, stas):
        for obs, c in (('S', 'k'), ('P', 'C3')):
            g = M[(M.sta == s) & (M.obs == obs) & (M.n >= 5)]
            ax.plot(g.mon, g.med * 1e3, '.-', color=c, ms=2, lw=0.5, label=obs)
        ax.set_ylim(-15, 15)
        ax.set_ylabel(s.split('.')[1], fontsize=8)
        for q in ['2021-12-20', '2022-12-20', '2024-12-05']:
            ax.axvline(pd.Timestamp(q), color='b', lw=0.5)
    axs[0].legend(fontsize=7, ncol=2)
    axs[0].set_title('Monthly median direct-S (black) and direct-P (red) delay, origin/stack term removed (ms)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_deep_shallow_clock.png', dpi=120)
    plt.close(fig)


def fig_lapse():
    ST = pd.read_csv(f'{OUT}/station_directS_coda.csv')
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    wins = ['w3_8', 'w8_15', 'w15_25']
    for ax, X in zip(axs, ['D', 'm7']):
        S = ST[ST.X == X].sort_values('sensor_km').reset_index(drop=True)
        x = np.arange(len(S))
        for k, w in enumerate(wins):
            ax.errorbar(x + (k - 1) * 0.2, S[f'coda_{w}_pct'], yerr=S[f'coda_{w}_se_pct'], fmt='o', ms=3,
                        label=f'stretch {w}', color=f'C{k}')
            ax.errorbar(x + (k - 1) * 0.2 + 0.07, S[f'coda_j{w[1:]}_pct'], yerr=S[f'coda_j{w[1:]}_se_pct'], fmt='x',
                        ms=3, color=f'C{k}', alpha=0.5, label=f'shift+stretch {w}')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s.split('.')[1]}\n{d * 1000:.0f}m" for s, d in zip(S.sta, S.sensor_km)], fontsize=6,
                           rotation=90)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_ylabel('coda dv/v (%)')
        ax.set_ylim(-0.6, 0.4)
        ax.set_title({'D': 'decline 2021->2025', 'm7': 'M7 step'}[X] + ' by lapse window (tau after S)')
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_deep_shallow_lapse.png', dpi=130)
    plt.close(fig)


def fig_rep():
    f = f'{OUT}/rep/rep_fit.csv'
    if not os.path.exists(f):
        return
    F = pd.read_csv(f)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (v, se, lab) in zip(axs, [('m7_pct', 'm7_se', 'M7 step'), ('D_pct', 'D_se_boot', 'decline 2021->2025')]):
        for k, cls in enumerate(['shallow', 'deep']):
            g = F[F.cls == cls].reset_index(drop=True)
            if se == 'm7_se':
                e = g.m7_se_boot          # placebo unreliable for sparse pre-2021 repeater pairs
            else:
                e = g[se]
            ax.errorbar(np.arange(len(g)) + 0.15 * k, g[v], yerr=e, fmt='o', label=cls, color=f'C{k}')
            for i, r in g.iterrows():
                ax.annotate(f"{r.sta.split('.')[1]} ({int(r.n_clusters)})", (i + 0.15 * k, r[v]), fontsize=6)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_ylabel('coda dv/v (%)')
        ax.set_title(f'repeating earthquakes: {lab}')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_deep_shallow_repeaters.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    for fn in [fig_directS, fig_deep, fig_clock, fig_lapse, fig_rep]:
        try:
            fn()
            print('ok', fn.__name__)
        except Exception as e:                          # noqa: BLE001
            print('FAIL', fn.__name__, repr(e))
