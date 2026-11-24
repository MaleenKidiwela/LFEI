#!/usr/bin/env python3
"""Figures + summary.json for the source-region S-P analysis (scripts/sci_dvv_source_region*.py).
figures/sci_dvv_source_region_{timeseries,estimates,synth,family}.png ; data/sci/dvv_source_region/summary.json
"""
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_source_region as M  # noqa: E402

OUT = M.OUT
FIG = M.FIG
Q4 = ['m7', 'restart', 'fer', 'D']
QLAB = {'m7': 'M7 step\n(>60 d)', 'restart': 'M7 restart\nwindow (0-60 d)', 'fer': 'Ferndale\nstep', 'D': 'decline\n2025-2021'}


def binned(o, p, edges):
    t = pd.to_datetime(o.t, unit='s')
    out = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        g = o[(t >= a_) & (t < b_) & np.isfinite(o[p])]
        if len(g) < 3:
            continue
        w = 1 / g[p + '_se'] ** 2
        out.append((a_ + (b_ - a_) / 2, np.sum(w * g[p]) / w.sum(), 1 / np.sqrt(w.sum()) * np.sqrt(max(
            np.sum(w * (g[p] - np.sum(w * g[p]) / w.sum()) ** 2) / max(len(g) - 1, 1), 1)), len(g)))
    return pd.DataFrame(out, columns=['t', 'y', 'se', 'n'])


def main():
    res = pickle.load(open(f'{OUT}/inv_all.pkl', 'rb'))   # our own outputs
    S = pd.read_csv(f'{OUT}/timefit_network.csv')
    # ---------------- time series: family-offset-removed eps (sph5, sph10) and dz
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    edges = list(pd.date_range('2008-01-01', '2021-01-01', freq='12MS')) + \
        list(pd.date_range('2021-07-01', '2024-12-01', freq='6MS')) + \
        [pd.Timestamp('2024-12-05T18:44'), pd.Timestamp('2025-02-03'), pd.Timestamp('2025-08-01'),
         pd.Timestamp('2026-02-01'), pd.Timestamp('2026-08-01')]
    for a_, (m, p, sc, lab) in zip(ax, [('sph5', 'eps0', 100, 'eps, 5-km sphere (%)'),
                                         ('sph10', 'eps0', 100, 'eps, 10-km sphere (%)'),
                                         ('sph5', 'dz', 1e3, 'centroid dz (m, +down)')]):
        o = res[m].copy()
        o = o[np.isfinite(o[p]) & (o[p + '_se'] < 20 * o[p + '_se'].median())]
        o[p] = o[p] - o.groupby('F')[p].transform(lambda x: np.average(x, weights=1 / o.loc[x.index, p + '_se'] ** 2))
        a_.scatter(pd.to_datetime(o.t, unit='s'), o[p] * sc, s=4, c='0.75', zorder=1)
        B = binned(o, p, edges)
        a_.errorbar(B.t, B.y * sc, B.se * sc, fmt='o', color='C0', ms=4, zorder=3)
        for k, c in (('pet', 'C2'), ('fer', 'C1'), ('m7', 'C3')):
            a_.axvline(pd.to_datetime(M.T_Q[k], unit='s'), color=c, lw=1)
        a_.axhline(0, color='k', lw=0.5)
        a_.set_ylabel(lab)
        lim = np.nanpercentile(np.abs(o[p] * sc), 97)
        a_.set_ylim(-lim, lim)
    ax[0].set_title('Source-region S-P inversion, network (27 families, family offsets removed); '
                    'grey = epoch stacks, blue = weighted bins')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_region_timeseries.png', dpi=130)
    plt.close(fig)

    # ---------------- estimates
    sets = [('all_stations', 'sph5', 'eps0', 'sph5 eps (Vp/Vs const)'), ('all_stations', 'sph10', 'eps0', 'sph10 eps'),
            ('all_stations', 'sph5S', 'eps0', 'sph5 eps_S only'), ('all_stations', 'sph5P', 'eps0', 'sph5 eps_P only'),
            ('all_stations', 'slab5SP', 'eps0', 'slab5 eps_S'), ('all_stations', 'slab5SP', 'eps1', 'slab5 eps_P'),
            ('all_stations', 'slab5SP', 'vpvs', 'slab5 dln(Vp/Vs)')]
    extra = []
    for f, lab in ((f'{OUT}/timefit_reccorr.csv', None), (f'{OUT}/placebo_repeaters_as_sources.csv', None)):
        if os.path.exists(f):
            extra.append(pd.read_csv(f))
    E = pd.concat([S] + extra, ignore_index=True)
    for s_, lab in (('all_corr_ds_P0.5', 'sph5, rec.-corr (deep_shallow S terms)'), ('boreholes_raw', 'sph5, boreholes only'),
                    ('rep9_corr_deep', 'sph5, 9 sta, 19-km repeater corr'),
                    ('placebo_shallow_as_source_corr_deep', 'PLACEBO 8-km repeaters as sources'),
                    ('placebo_deep_as_source_corr_shallow', 'PLACEBO 19-km repeaters as sources')):
        if (E.set == s_).any():
            sets.append((s_, 'sph5', 'eps0', lab))
    fig, ax = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
    for j, q in enumerate(Q4):
        for i, (s_, m, p, lab) in enumerate(sets):
            g = E[(E.set == s_) & (E.model == m) & (E.param == p) & (E.quantity == q)]
            if len(g) == 0:
                continue
            g = g.iloc[0]
            se = g.se_total if 'se_total' in g and np.isfinite(g.get('se_total', np.nan)) else g.se
            c = 'C3' if 'PLACEBO' in lab else ('C0' if s_ == 'all_stations' else 'C2')
            lim = 1.2
            if abs(g.val) + se > lim:          # off-scale (repeater placebos): print the value instead
                ax[j].text(0, -i, f'{g.val:+.2f} +/- {se:.2f} (off scale)', color=c, fontsize=7, ha='center', va='center')
                continue
            ax[j].errorbar(g.val, -i, xerr=se, fmt='o', color=c, capsize=3)
            ax[j].errorbar(g.val, -i, xerr=2 * se, fmt='none', color=c, alpha=0.35)
        ax[j].set_xlim(-1.2, 1.2)
        ax[j].axvline(0, color='k', lw=0.6)
        ax[j].set_title(QLAB[q].replace('\n', ' '))
        ax[j].set_xlabel('dv/v (%)  [1, 2 x honest SE]')
    ax[0].set_yticks([-i for i in range(len(sets))])
    ax[0].set_yticklabels([s[3] for s in sets], fontsize=8)
    fig.suptitle('Source-region dv/v from S-P (honest SE = max(bootstrap, placebo))')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_region_estimates.png', dpi=130)
    plt.close(fig)

    # ---------------- synthetics
    Z = pd.read_csv(f'{OUT}/synthetic_recovery.csv')
    Z = Z[(Z.quantity == 'm7') & Z.model.isin(['sph5', 'sph10', 'sph5_noz'])]
    piv = Z.pivot_table(index='scenario', columns='model', values='rec_minus_base_pct')
    piv = piv.drop(index='baseline', errors='ignore')
    fig, ax = plt.subplots(figsize=(10, 5))
    piv.plot.barh(ax=ax)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('recovered M7-step eps minus baseline (%)')
    ax.set_title('Delay-level injections into the real epoch-stack delays (M7)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_region_synth.png', dpi=130)
    plt.close(fig)

    # ---------------- family terms
    T = pd.read_csv(f'{OUT}/family_terms_sph5.csv')
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for a_, q in zip(ax, ['m7', 'restart', 'D']):
        g = T[np.isfinite(T[q])]
        a_.errorbar(g[q], g.F, xerr=g[q + '_se'], fmt='o')
        a_.axvline(0, color='k', lw=0.6)
        a_.set_title(f'{q}: per family (sph5)')
        a_.set_xlabel('%')
    ax[0].set_ylabel('family')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_region_family.png', dpi=130)
    plt.close(fig)

    # ---------------- summary.json
    summ = {}
    for s_, m, p, lab in sets:
        for q in Q4:
            g = E[(E.set == s_) & (E.model == m) & (E.param == p) & (E.quantity == q)]
            if len(g):
                g = g.iloc[0]
                summ[f'{s_}|{m}|{p}|{q}'] = {k: (float(g[k]) if k in g and pd.notna(g[k]) else None)
                                              for k in ('val', 'se_boot', 'se_plac', 'se', 'z', 'se_total')}
    for f in ('family_terms_Q.json',):
        if os.path.exists(f'{OUT}/{f}'):
            summ['family_heterogeneity_Q'] = json.load(open(f'{OUT}/{f}'))
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1)
    print('figures + summary.json written')


if __name__ == '__main__':
    main()
