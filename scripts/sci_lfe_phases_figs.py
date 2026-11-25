#!/usr/bin/env python3
"""Figures for the LFE secondary-phase / attenuation study.
  atten   : figures/sci_lfe_phases_atten.png   (yearly dt*_S, dt*_P, d(t*_P - t*_S); quake steps vs placebo; Qp/Qs)
  scan    : figures/sci_lfe_phases_scan.png    (z vs reflector offset per phase / subset; injection recovery)
  sections: figures/sci_lfe_phases_sections.png (record sections of the matched-filter detection function, families
            sorted by height above the Bloch slab Moho, predicted SmS / PmP times)
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_lfe_phases_common import OUT, FIG, HGRID, geometry  # noqa: E402

QY = {'Petrolia': 2021.97, 'Ferndale': 2022.97, 'M7': 2024.93}


def atten():
    Y = pd.read_csv(f'{OUT}/atten_yearly.csv')
    T = pd.read_csv(f'{OUT}/atten_tests.csv')
    Q = pd.read_csv(f'{OUT}/atten_qpqs.csv')
    qj = json.load(open(f'{OUT}/atten_qpqs.json'))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, (q, lab) in enumerate((('S', 'dt*_S'), ('P', 'dt*_P'), ('PS', 'd(t*_P - t*_S)'))):
        a = ax[0, j]
        for grp, c in (('all', 'k'), ('PB', 'tab:blue'), ('surface', 'tab:orange')):
            g = Y[(Y.q == q) & (Y.group == grp)].copy()
            g['yr'] = g.test.str[5:].astype(int) + 0.5
            g = g.sort_values('yr')
            a.plot(g.yr, g.value * 1e3, 'o-', color=c, ms=3, label=grp)
        pre = Y[(Y.q == q) & (Y.group == 'all')].copy()
        pre['yr'] = pre.test.str[5:].astype(int)
        sd = pre[pre.yr <= 2020].value.std()
        a.axhspan(-2e3 * sd, 2e3 * sd, color='0.85', zorder=0, label='+-2 sd (2008-20 years)')
        for k, v in QY.items():
            a.axvline(v, color='r', lw=0.8, ls='--')
        a.set_title(f'yearly {lab} vs all-time (ms)')
        a.set_ylabel('ms')
        if j == 0:
            a.legend(fontsize=7)
        # quake steps
        b = ax[1, j] if j < 2 else None
        if b is not None:
            g = T[(T.q == q) & (T.group == 'all')]
            y = np.arange(len(g))
            b.errorbar(g.value * 1e3, y, xerr=2e3 * g.placebo_sd, fmt='o', color='k', capsize=2)
            b.set_yticks(y)
            b.set_yticklabels(g.test, fontsize=7)
            b.axvline(0, color='0.5')
            b.set_xlabel(f'{lab} change (ms), +-2 placebo sd')
            b.set_title(f'{q}: steps / restart / long-term (network)')
    g = T[(T.q == 'PS') & (T.group == 'all')]
    b = ax[1, 2]
    x = Q.TS - Q.groupby('F').TS.transform('mean')
    yv = Q.dtPS - Q.groupby('F').dtPS.transform('mean')
    b.scatter(x, yv * 1e3, s=6, c='0.4')
    xx = np.linspace(x.min(), x.max(), 10)
    b.plot(xx, qj['slope_dtPS_per_sTS'] * xx * 1e3, 'r')
    b.set_xlabel('T_S - family mean (s)')
    b.set_ylabel('t*_P - t*_S, family-demeaned (ms)')
    b.set_title(f"long-term P/S slope: {qj['slope_dtPS_per_sTS'] * 1e3:.2f} +- {qj['se_station_boot'] * 1e3:.2f} ms/s")
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_lfe_phases_atten.png', dpi=120)


def scan(tag=''):
    from sci_lfe_phases_summary import load_scans
    D = load_scans()
    phases = ['SmS', 'PmS', 'PmP', 'SmP', 'Sp_above', 'Ps_above']
    kinds = ['c', 'a', 'e']
    kname = {'c': 'matched filter', 'a': 'envelope/S', 'e': 'detrended log env'}
    fig, ax = plt.subplots(len(kinds) + 1, len(phases), figsize=(3.2 * len(phases), 2.6 * (len(kinds) + 1)),
                           squeeze=False)
    for i, k in enumerate(kinds):
        for j, ph in enumerate(phases):
            a = ax[i, j]
            for sub, c in (('all', 'k'), ('boreholes', 'tab:blue'), ('surface', 'tab:orange'), ('fam_odd', 'tab:green'),
                           ('fam_even', 'tab:olive')):
                g = D[(D.phase == ph) & (D.kind == k) & (D.subset == sub) & (D.null == 'famclass')].sort_values('dz')
                if len(g):
                    a.plot(g.dz, g.z, color=c, lw=1.4 if sub == 'all' else 0.7, label=sub)
            g = D[(D.phase == ph) & (D.kind == k) & (D.subset == 'all') & (D.null == 'fam')].sort_values('dz')
            a.plot(g.dz, g.z, color='k', ls='--', lw=0.8, label='all, all-family null')
            a.axhline(3, color='0.5', ls=':')
            a.axhline(-3, color='0.5', ls=':')
            a.axvline(0, color='r', lw=0.6)
            a.set_ylim(-5, 6)
            a.set_title(f'{ph} [{kname[k]}]', fontsize=9)
            if j == 0:
                a.set_ylabel('z (within-class null)')
            a.set_xlabel('offset from Bloch ' + ('t' if 'above' in ph else 'm') + ' (km)', fontsize=8)
    ax[0, 0].legend(fontsize=6)
    # injection recovery
    try:
        I = pd.read_csv(f'{OUT}/search_injection.csv')
        a = ax[-1, 0]
        for (k, null), g in I[(I.subset == 'all') & I.null.isin(['fam', 'famclass'])].groupby(['kind', 'null']):
            g = g.sort_values('amp')
            a.plot(g.amp, g.z_at_true, 'o-', label=f'{kname[k]} / {null}')
        a.axhline(3, color='0.5', ls=':')
        a.set_xscale('log')
        a.set_xlabel('injected SmS amplitude / direct S')
        a.set_ylabel('z at true dz (all)')
        a.set_title('injection recovery (SmS, Bloch m)', fontsize=9)
        a.legend(fontsize=6)
    except FileNotFoundError:
        pass
    for a in ax[-1, 1:]:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_lfe_phases_scan.png', dpi=110)


def sections():
    from sci_lfe_phases import load_fs, TAU
    G, tabs = geometry()
    recs, _ = load_fs()
    stas = ['BK.JEWT', 'NC.KZE', 'NC.KMR', 'BK.RVIT', 'BK.SPRL', 'PB.B934', 'PB.B049', 'PB.B933']
    fig, ax = plt.subplots(2, len(stas), figsize=(2.6 * len(stas), 11), sharex=True)
    for row, (comp, ref, key, tm) in enumerate((('H', 'S', 'SmS', 'S'), ('Z', 'P', 'PmP', 'P'))):
        for a, s in zip(ax[row], stas):
            rr = [r for r in recs if r['sta'] == s and f'c{comp}{tm}_{ref}' in r]
            rr = sorted(rr, key=lambda r: G.iloc[r['gi']].H_m)
            for j, r in enumerate(rr):
                c = r[f'c{comp}{tm}_{ref}']
                a.plot(TAU, j + np.clip(c, 0, 1.5) * 0.9, 'k', lw=0.6)
                g = G.iloc[r['gi']]
                d = np.interp(g.H_m, HGRID, tabs[key][r['gi']])
                a.plot([d, d], [j, j + 1], 'r', lw=1.5)
                a.text(-2.9, j + 0.1, f"{r['F']} {g.H_m:.1f}", fontsize=5)
                other = r['P'] - r['S'] if ref == 'S' else r['S'] - r['P']
                if np.isfinite(other):
                    a.plot([other] * 2, [j, j + 1], 'b', lw=1)
            a.set_xlim(-3, 6)
            a.set_title(f'{s} {comp} rel {ref}', fontsize=8)
            a.set_yticks([])
            if row == 1:
                a.set_xlabel(f'time after direct {ref} (s)')
    ax[0, 0].set_ylabel('families sorted by height above Bloch m (km) ->')
    fig.suptitle('Matched-filter detection function (self-correlation envelope, 2-8 Hz); red = predicted SmS (top) / '
                 'PmP (bottom) from the Bloch slab Moho; blue = other direct phase', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_lfe_phases_sections.png', dpi=110)


def timelapse():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    modes = (('real', 'real data (predicted SmS window = S coda)', 'k'),
             ('inject0', 'synthetic SmS amp 0.3, no change', 'tab:gray'),
             ('inject', 'synthetic SmS amp 0.3, +1% delay after restart', 'tab:red'),
             ('inject_amp1', 'synthetic SmS amp 1.0, +1% delay after restart', 'tab:purple'))
    for m, lab, c in modes:
        try:
            R = pd.read_csv(f'{OUT}/timelapse_{m}.csv')
        except FileNotFoundError:
            continue
        R = R[R.group == 'all']
        y = R[R.kind == 'year'].copy()
        y['yr'] = y.test.str[5:].astype(int) + 0.5
        y = y.sort_values('yr')
        ax[0].plot(y.yr, 100 * y.eps, 'o-', color=c, ms=3, label=lab)
        pl = R[R.kind == 'placebo'].eps * 100
        ax[1].hist(pl, bins=30, histtype='step', color=c, density=True)
        for _, r in R[R.kind == 'real'].iterrows():
            ax[2].plot(100 * r.eps, ['step_petrolia', 'step_ferndale', 'step_m7', 'restart_0_60d'].index(r.test)
                       + 0.1 * [mm[0] for mm in modes].index(m), 'o', color=c)
        ax[2].errorbar([0] * 4, np.arange(4) + 0.15, xerr=2 * pl.std(), fmt='none', color=c, alpha=0.4)
    for k, v in QY.items():
        ax[0].axvline(v, color='r', lw=0.6, ls='--')
    ax[0].set_ylabel('phase-minus-direct delay change / tau (%)')
    ax[0].set_title('yearly (vs all-time)')
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel('placebo step eps (%)')
    ax[1].set_title('placebo distributions (+-180 d)')
    ax[2].set_yticks(range(4))
    ax[2].set_yticklabels(['Petrolia', 'Ferndale', 'M7 (restart+180 d)', 'restart 0-60 d'])
    ax[2].axvline(0, color='0.5')
    ax[2].axvline(1, color='tab:red', ls=':', lw=0.8)
    ax[2].set_xlim(-1, 1.5)
    ax[2].set_xlabel('eps (%) = -dv/v in layer; bars +-2 placebo sd')
    ax[2].set_title('quake tests')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_lfe_phases_timelapse.png', dpi=120)


if __name__ == '__main__':
    for w in sys.argv[1:]:
        {'atten': atten, 'scan': scan, 'sections': sections, 'timelapse': timelapse}[w]()
