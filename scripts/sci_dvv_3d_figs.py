#!/usr/bin/env python3
"""Figures + summary.json for the joint 3-D relative dv/v study (scripts/sci_dvv_3d_*.py).

figures/sci_dvv_3d_deep.png        stage-1 deep estimates (clean primary, all-rows secondary), honest 95 % t-CI, placebos
figures/sci_dvv_3d_validation.png  injection recovery (cross-geometry deep truths, mid-crust leakage, shallow leakage),
                                   variance inflation by the nuisance terms, noise SD
figures/sci_dvv_3d_ak.png          averaging kernels (layer sums) of the deep estimators, own geometry
figures/sci_dvv_3d_petrolia.png    Petrolia / decline checks: yearly layer series, jackknives, S-only / P-only, LP tests
figures/sci_dvv_3d_shallow.png     stage-2 per-station shallow terms (0-0.1, 0.1-1, 1-3 km) for M7 and D (map)
figures/sci_dvv_3d_fit.png         stage-2 chi2 per dataset / quantity
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_3d'
FIG = f'{ROOT}/figures'
sys.path.insert(0, f'{ROOT}/scripts')
QL = {'m7': 'M7 step', 'pet': 'Petrolia step', 'fer': 'Ferndale step', 'D': '2021→2025 change', 'ramp': 'ramp /yr'}


def rd(f):
    p = f'{OUT}/{f}'
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def fig_deep(T):
    fig, axs = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
    runs = [r for r in T.run.unique() if r.startswith(('clean_', 'joint_')) and '_yearly' not in r]
    order = sorted(runs, key=lambda r: (not r.startswith('clean'), r))
    for ax, q in zip(axs, ['m7', 'pet', 'fer', 'D']):
        g = T[(T.quantity == q) & T.run.isin(order) & ~T.vol.str.startswith('R')].copy()
        g['lab'] = g.run + ' : ' + g.vol
        g = g.sort_values('lab')
        y = np.arange(len(g))
        col = np.where(g.run.str.startswith('clean'), '#1f5fbf', '#999999')
        ax.errorbar(g.est_pct, y, xerr=[g.est_pct - g.ci95_lo, g.ci95_hi - g.est_pct], fmt='none', ecolor=col, lw=1.2)
        ax.scatter(g.est_pct, y, c=col, s=18, zorder=3)
        for yi, (_, r) in zip(y, g.iterrows()):
            if isinstance(r.plac_vals, str) and r.plac_vals:
                pv = np.array([float(v) for v in r.plac_vals.split(';')])
                ax.scatter(pv, np.full(len(pv), yi + 0.25), s=4, c='#d08000', alpha=0.6)
        ax.axvline(0, c='k', lw=0.6)
        ax.set_title(QL[q])
        ax.set_xlabel('dv/v (%)')
        ax.set_yticks(y)
        ax.set_yticklabels(g.lab, fontsize=6)
        ax.set_xlim(-1.5, 1.5)
    fig.suptitle('Stage-1 deep terms (free receiver terms; honest SE; 95 % t-CI). Blue = clean primary subset, grey = all rows '
                 '(amplitude calibration x2-3 uncertain). Orange = season-matched placebo values', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_deep.png', dpi=130)
    plt.close(fig)


def fig_valid(I, V):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
    ax = axs[0]
    g = I[(I.stage == 'TT') & I.kind.isin(['deep', 'mid'])]
    piv = g.pivot_table(index=['run', 'vol'], columns=['truth', 'geom'], values='rec_pct')
    im = ax.imshow(piv.values / -0.1, cmap='RdBu_r', vmin=-1.2, vmax=1.2, aspect='auto')
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([f'{a}:{b}' for a, b in piv.index], fontsize=6)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f'{a}|{b}' for a, b in piv.columns], rotation=90, fontsize=6)
    plt.colorbar(im, ax=ax, label='recovered / (-0.1 %) [sph, slab truths are -0.3 %]')
    ax.set_title('Stage-1 response to deep / mid-crust truths (real estimator)', fontsize=9)
    ax = axs[1]
    s = I[~I.kind.isin(['deep', 'mid', 'noise'])].groupby(['run', 'stage', 'kind', 'coda']).rec_pct.agg(lambda x: np.sqrt(np.mean(x ** 2))).reset_index()
    for k, (kind, gg) in enumerate(s.groupby('kind')):
        ax.scatter(gg.rec_pct, np.full(len(gg), k) + np.where(gg.stage == 'TT', -0.15, 0.15), s=12,
                   c=np.where(gg.stage == 'TT', '#1f5fbf', '#c03030'))
    ax.set_yticks(range(s.kind.nunique()))
    ax.set_yticklabels(sorted(s.kind.unique()))
    ax.set_xlabel('RMS spurious deep value (%)')
    ax.set_title('Shallow / clock truths: leakage into deep terms\n(blue stage 1, red stage 2 with adversarial coda kernels)', fontsize=9)
    ax = axs[2]
    if len(V):
        y = np.arange(len(V))
        ax.barh(y - 0.25, V.se_naive, 0.25, label='naive one-column')
        ax.barh(y, V.se_no_receiver, 0.25, label='no receiver terms')
        ax.barh(y + 0.25, V.se_free_receiver, 0.25, label='free receiver terms (used)')
        ax.set_yticks(y)
        ax.set_yticklabels(V.run + ':' + V.vol, fontsize=7)
        ax.set_xscale('log')
        ax.set_xlabel('formal SE of the M7 deep term (%)')
        ax.legend(fontsize=7)
        ax.set_title('Information absorbed by nuisance terms', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_validation.png', dpi=130)
    plt.close(fig)


def fig_ak(A):
    import sci_dvv_3d_G as GG
    cols = [f'L{n}' for n in GG.LAYNAME]
    zc = [0.5 * (GG.LAY[i] + min(GG.LAY[i + 1], 40)) for i in range(GG.NL)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, q in zip(axs, ['m7', 'D']):
        for _, r in A[A.quantity == q].iterrows():
            ls = '-' if r.run.startswith('clean') else ':'
            ax.plot([r[c] for c in cols], zc, ls, marker='o', ms=3, label=f'{r.run}:{r.vol} (sum {r.ak_total:+.2f})')
        ax.axvline(0, c='k', lw=0.5)
        ax.axhspan(20, 31, color='0.9')
        ax.set_ylim(40, 0)
        ax.set_xlabel('averaging kernel, summed per layer')
        ax.set_title(QL[q])
    axs[0].set_ylabel('depth (km)')
    axs[1].legend(fontsize=6, loc='lower right')
    fig.suptitle('Averaging kernels of the stage-1 deep estimators (response to a unit dv/v in each layer, own geometry)', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_ak.png', dpi=130)
    plt.close(fig)


def fig_pet(T, J, L, LY):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax = axs[0, 0]
    for run in [r for r in T.run.unique() if r.endswith('_yearly')]:
        g = T[T.run == run]
        yr = g.quantity.str[1:].astype(int)
        ax.errorbar(yr + (0.1 if '3d' in run else -0.1), g.est_pct, g.se_formal, fmt='o-', ms=3, capsize=2, label=run)
    for d in (2021.97, 2022.97, 2024.93):
        ax.axvline(d, c='r', lw=0.6)
    ax.axhline(0, c='k', lw=0.5)
    ax.set_title('Layer (20-31 km) yearly series rel. 2008-09 (formal SE)', fontsize=9)
    ax.legend(fontsize=6)
    ax = axs[0, 1]
    if len(J):
        J = J.copy()
        J['v'] = J.run.str.split('_').str[1]
        J['kind'] = J.run.str.split('_').str[2]
        for k, (key, g) in enumerate(J[J.quantity.isin(['pet', 'D', 'm7'])].groupby(['v', 'quantity'])):
            ax.scatter(g.est_pct, np.full(len(g), k) + np.where(g.kind == 'sta', -0.15, 0.15), s=8,
                       c=np.where(g.kind == 'sta', '#1f5fbf', '#c03030'))
        labs = [f'{a}:{b}' for a, b in J[J.quantity.isin(['pet', 'D', 'm7'])].groupby(['v', 'quantity']).groups]
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=7)
        ax.axvline(0, c='k', lw=0.5)
        ax.set_title('Jackknife (clean subset): drop one station (blue) / family (red)', fontsize=9)
    ax = axs[1, 0]
    sel = T[T.run.str.contains('Sonly|Ponly|eqboth|lvz|clean_lay_|clean_x40_lay|clean_lfe_lay') & T.quantity.isin(['pet', 'D'])
            & ~T.run.str.contains('yearly')]
    sel = sel.sort_values(['quantity', 'run'])
    y = np.arange(len(sel))
    ax.errorbar(sel.est_pct, y, xerr=1.96 * sel.se, fmt='o', ms=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sel.run + ':' + sel.quantity, fontsize=6)
    ax.axvline(0, c='k', lw=0.5)
    ax.set_title('Layer term: Petrolia and D under subsets (±1.96 SE)', fontsize=9)
    ax = axs[1, 1]
    if len(L):
        g = L[(L.dbin == '>=80') & L.quantity.isin(['pet', 'D'])].sort_values(['band', 'quantity', 'variant'])
        y = np.arange(len(g))
        ax.errorbar(g.est, y, xerr=1.96 * g.se, fmt='o', ms=3, c='#444')
        ax.set_yticks(y)
        ax.set_yticklabels(g.band + ' ' + g.quantity + ' ' + g.variant, fontsize=5)
        ax.axvline(0, c='k', lw=0.5)
        ax.set_title('LP pairs >= 80 km: Petrolia / D robustness', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_petrolia.png', dpi=130)
    plt.close(fig)


def fig_shallow(S):
    ST = pd.read_csv(f'{OUT}/stations.csv').set_index('ns')
    run = 'clean_R_lay_3d' if (S.run == 'clean_R_lay_3d').any() else S.run.iloc[0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for i, q in enumerate(['m7', 'D']):
        for k, zb in enumerate(['0-0.1 km', '0.1-1 km', '1-3 km']):
            ax = axs[i, k]
            g = S[(S.run == run) & (S.quantity == q) & S.param.str.startswith('s|') & S.param.str.endswith(f'|{k}')].copy()
            g['sta'] = g.param.str.split('|').str[1]
            g = g[g.sta.isin(ST.index)]
            z = g.est / g.sd
            sc = ax.scatter(ST.loc[g.sta, 'lon'], ST.loc[g.sta, 'lat'], c=g.est, cmap='RdBu', vmin=-np.nanpercentile(np.abs(g.est), 95),
                            vmax=np.nanpercentile(np.abs(g.est), 95), s=30 + 30 * np.clip(np.abs(z), 0, 4), edgecolors='k', lw=0.4)
            for _, r in g.iterrows():
                ax.annotate(r.sta.split('.')[1], (ST.loc[r.sta, 'lon'], ST.loc[r.sta, 'lat']), fontsize=5)
            ax.plot(-123.56, 40.05, 'k*', ms=10)
            plt.colorbar(sc, ax=ax, label='dv/v (%)')
            ax.set_title(f'{QL[q]}: station term {zb} (size ~ |z|)', fontsize=9)
    fig.suptitle(f'Stage-2 shallow attribution ({run}); conditional on coda kernels and priors', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_shallow.png', dpi=130)
    plt.close(fig)


def fig_fit(C):
    g = C[(C.cfg == 'primary') & C.run.str.startswith('clean_R_lay')]
    piv = g.pivot_table(index=['run', 'quantity'], columns='dataset', values='chi2r')
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(piv.values, cmap='viridis', vmin=0, vmax=3, aspect='auto')
    for (i, j), v in np.ndenumerate(piv.values):
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='w', fontsize=7)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([f'{a}:{b}' for a, b in piv.index], fontsize=7)
    plt.colorbar(im, ax=ax, label='chi2 / n')
    ax.set_title('Stage-2 data fit per dataset (primary configuration)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_fit.png', dpi=130)
    plt.close(fig)


def summary(T, Jk, I, V, A, G, LT, TEN):
    out = {}
    prim = T[T.run.str.startswith('clean_') & ~T.run.str.contains('yearly')]
    out['stage1_primary_clean'] = prim[['run', 'vol', 'quantity', 'est_pct', 'se', 'z', 'p_t', 'ci95_lo', 'ci95_hi',
                                        'n_plac', 'plac_zrms', 'chi2r', 'n']].round(5).to_dict('records')
    sec = T[T.run.str.startswith('joint_') & ~T.run.str.contains('yearly')]
    out['stage1_secondary_allrows'] = sec[['run', 'vol', 'quantity', 'est_pct', 'se', 'z', 'p_t']].round(5).to_dict('records')
    ds = T[T.run.str.startswith('ds_')]
    out['directS_honest_equivalent'] = ds[['run', 'vol', 'quantity', 'est_pct', 'se', 'p_t']].round(5).to_dict('records')
    tests = T[T.run.str.startswith(('clean_', 'joint_')) & T.quantity.isin(['m7', 'pet', 'fer', 'D'])]
    out['look_elsewhere'] = dict(n_tests=int(len(tests)), n_p_lt_005=int((tests.p_t < 0.05).sum()),
                                 expected_by_chance=round(0.05 * len(tests), 1))
    if len(Jk):
        out['jackknife'] = Jk[Jk.quantity.isin(['m7', 'pet', 'D'])].assign(v=Jk.run.str.split('_').str[1]).groupby(['v', 'quantity']).est_pct.agg(
            ['min', 'max', 'mean', 'std']).round(4).reset_index().to_dict('records')
    if len(V):
        out['variance_inflation'] = V.round(5).to_dict('records')
    if len(I):
        d = I[(I.stage == 'TT') & I.kind.isin(['deep', 'mid', 'noise'])]
        out['injections_stage1'] = d[['run', 'truth', 'geom', 'vol', 'rec_pct', 'frac', 'noise_sd_pct']].round(5).to_dict('records')
        sh = I[~I.kind.isin(['deep', 'mid', 'noise'])].groupby(['stage', 'kind']).rec_pct.agg(lambda x: float(np.sqrt(np.mean(x ** 2))))
        out['shallow_clock_leakage_rms'] = {f'{a}|{b}': round(v, 5) for (a, b), v in sh.items()}
    if len(A):
        out['averaging_kernels'] = A.round(4).to_dict('records')
    if len(G):
        out['stage2_conditional'] = G[G.cfg == 'primary'][['run', 'quantity', 'vol', 'stage2_conditional_est_pct',
                                                            'stage2_conditional_sd_pct', 'dchi2_shallow_only', 'dchi2_u', 'kappa']].round(4).to_dict('records')
    if len(TEN):
        out['tt_lp_tension'] = TEN.round(4).to_dict('records')
    if len(LT):
        out['lp_tests_ge80'] = LT[(LT.dbin == '>=80') & LT.quantity.isin(['pet', 'D', 'm7'])][
            ['band', 'variant', 'quantity', 'est', 'se', 'z', 'n_pairs', 'n_span_pet', 'n_plac', 'flag_few_plac']].round(4).to_dict('records')
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)


def main():
    T = pd.concat([rd('tt_results.csv')], ignore_index=True)
    Jk = rd('tt_results_jack.csv')
    I, V, A = rd('inj_results.csv'), rd('var_inflation.csv'), rd('ak_summary.csv')
    G, C, S = rd('gls_results.csv'), rd('gls_dataset_chi2.csv'), rd('gls_station_terms.csv')
    LT, LY, TEN = rd('lp_tests.csv'), rd('lp_yearly.csv'), rd('gls_tt_lp_tension.csv')
    if len(T):
        fig_deep(T[~T.run.str.contains('yearly')])
    if len(I):
        fig_valid(I, V)
    if len(A):
        fig_ak(A)
    if len(T):
        fig_pet(T, Jk, LT, LY)
    if len(S):
        fig_shallow(S)
    if len(C):
        fig_fit(C)
    summary(T, Jk, I, V, A, G, LT, TEN)


if __name__ == '__main__':
    main()
