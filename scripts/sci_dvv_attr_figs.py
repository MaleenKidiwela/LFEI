#!/usr/bin/env python
"""Figures for the dv/v attribution (reads data/sci/dvv_attr/*.csv and the bulk component series).
python scripts/sci_dvv_attr_figs.py [tag]  -> figures/sci_dvv_attr_{examples,selection,bands,decline,drivers,geodesy,oos}.png
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_attr_lib as L

OUT = f'{ROOT}/data/sci/dvv_attr'
FIG = f'{ROOT}/figures'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'main'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']   # fixed order
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BANDS = ['0.1-0.3Hz', '0.3-0.6Hz', '0.5-1Hz', '1-3Hz', '3-5Hz']
FC = {'0.1-0.3Hz': 0.2, '0.3-0.6Hz': 0.45, '0.5-1Hz': 0.75, '1-3Hz': 2.0, '3-5Hz': 4.0}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False})

C = pd.read_csv(f'{OUT}/components.csv')
S = pd.read_csv(f'{OUT}/model_scores.csv')
T = pd.read_csv(f'{OUT}/candidate_tests.csv')
C['cls'] = np.where(C.sta.str.startswith('PB.'), 'borehole', np.where(C.sta == 'REGION', 'network', 'surface'))


def comp(df, name):
    return df[df.component == name].set_index(['dataset', 'sta', 'chan_band'])


# ------------------------------------------------------------------------------------------------ 1 examples
def fig_examples():
    Z = pd.read_parquet(f'{L.BULK}/results/component_series_{TAG}.parquet')
    want = [('SC10', 'PB.B045', '1-3Hz'), ('SC10', 'NC.KMPB', '1-3Hz'), ('SC10', 'BK.BRIC', '1-3Hz'),
            ('SC10', 'PB.B049', '0.1-0.3Hz'), ('LFE', 'NC.KMPB', 'NC.KMPB..HHE'), ('LP10', 'REGION', '0.05-0.1Hz_BB')]
    sys.path.insert(0, f'{ROOT}/scripts')
    import sci_dvv_attr_fit as F
    series = {}
    for ds, loader in (('SC10', F.load_sc), ('LFE', F.load_lfe), ('LP10', F.load_lp)):
        if any(w[0] == ds for w in want):
            for s in loader():
                if (s.dataset, s.sta, s.chan) in want:
                    series[(s.dataset, s.sta, s.chan)] = s
    want = [w for w in want if w in series and f'{w[0]}|{w[1]}|{w[2]}|FIT' in Z]
    fig, axs = plt.subplots(len(want), 1, figsize=(11, 2.3 * len(want)), sharex=True)
    for ax, w in zip(np.atleast_1d(axs), want):
        s = series[w]
        t = L.DAYS[s.idx]
        fit = Z[f'{w[0]}|{w[1]}|{w[2]}|FIT']
        off = np.nanmedian(s.y - fit.values[s.idx])
        ax.plot(t, s.y - off, '.', ms=1.2, color='#9a9994', label='data')
        span = (L.DAYS >= t.min()) & (L.DAYS <= t.max())
        ax.plot(L.DAYS[span], fit[span], color=INK, lw=1.0, label='model')
        for k, (g, col) in enumerate((('QUAKES', CAT[7]), ('LONGTERM', CAT[6]), ('ENV', CAT[0]))):
            key = f'{w[0]}|{w[1]}|{w[2]}|{g}'
            if key in Z:
                v = Z[key][span]
                ax.plot(L.DAYS[span], v - np.nanmean(v) - 0.25 * (k + 1) * np.nanstd(s.y) * 2, color=col, lw=1.2, label=g.lower())
        sel = S[(S.dataset == w[0]) & (S.sta == w[1]) & (S.chan_band == w[2])]
        txt = f"{w[1]} {w[2]} ({w[0]})  selected: {sel.selected.fillna('none').iloc[0] if len(sel) else '?'};  R2 {sel.final_r2.iloc[0]:.2f}" if len(sel) else ''
        ax.set_title(txt, loc='left', fontsize=9, color=INK)
        ax.set_ylabel('dv/v (%)')
    np.atleast_1d(axs)[0].legend(ncol=5, loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_attr_examples.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 2 selection matrix
def fig_selection():
    sc = S[S.dataset == 'SC10'].copy()
    stas = sorted(sc.sta.unique(), key=lambda s: (not s.startswith('PB.'), s))
    groups = ['hyd_exp', 'hyd_exp_era5', 'hyd_diff', 'rain_lags', 'soilm', 'streamflow', 'thermal', 'gnss_up', 'gnss_dil',
              'bstrain', 'micro', 'pressure', 'sealevel']
    col = {g: CAT[i % 8] for i, g in enumerate(['hyd_exp', 'thermal', 'gnss_up', 'micro', 'pressure', 'bstrain', 'sealevel', 'gnss_dil'])}
    for g in ('hyd_exp_era5', 'hyd_diff', 'rain_lags', 'soilm', 'streamflow'):
        col[g] = col['hyd_exp']
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(stas) + 1.5))
    for i, s in enumerate(stas):
        for j, b in enumerate(BANDS):
            r = sc[(sc.sta == s) & (sc.chan_band == b)]
            if not len(r):
                continue
            sel = [x for x in str(r.selected.iloc[0]).split(';') if x and x != 'nan']
            for k, g in enumerate(sel):
                ax.add_patch(plt.Rectangle((j + 0.05 + 0.25 * k, i + 0.1), 0.23, 0.8, color=col.get(g, '#999'), lw=0))
                pv = r.get(f'p_wy_perm_{g}', pd.Series([np.nan])).iloc[0]
                ax.text(j + 0.165 + 0.25 * k, i + 0.5, '' if pv <= 0.05 else '?', ha='center', va='center', fontsize=7, color=INK)
    ax.set_xlim(0, len(BANDS)); ax.set_ylim(len(stas), 0)
    ax.set_xticks(np.arange(len(BANDS)) + 0.5); ax.set_xticklabels(BANDS)
    ax.set_yticks(np.arange(len(stas)) + 0.5); ax.set_yticklabels(stas, fontsize=7)
    ax.grid(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=col[g]) for g in ('hyd_exp', 'thermal', 'gnss_up', 'micro', 'pressure', 'bstrain', 'sealevel', 'gnss_dil')]
    ax.legend(h, ['hydrology (any form)', 'thermal', 'GNSS vertical', 'microseism power', 'pressure', 'borehole strain', 'sea level', 'GNSS areal strain'],
              ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.04))
    ax.set_title('Selected environmental drivers per station x band (? = water-year permutation p > 0.05: seasonal, cause unresolved)', loc='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_attr_selection.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 3 band dependence
def fig_bands():
    sc = C[C.dataset == 'SC10']
    items = [('annual_amp[ENV]', 'annual amplitude of environmental model (%)'), ('annual_amp[TOTAL]', 'annual amplitude, total model (%)'),
             ('drop0_M7', 'M7 coseismic drop (%)'), ('level1yr_M7', 'M7 level after 1 yr (%)'),
             ('chg_2125[TOTAL]', '2025-2021 change, total model (%)'), ('chg_2125[POST21]', '2025-2021 change, quake terms + ramp (%)'),
             ('chg_2125[LONGTERM]', '2025-2021 change from trend/ramp (%)'), ('chg_2125[ENV]', '2025-2021 change from environment (%)')]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (k, lab) in zip(axs.ravel(), items):
        d = sc[sc.component == k]
        if k.startswith('chg_2125'):
            d = d[d.chg_identifiable.astype(bool)]          # free instrument offset inside 2021-2025 -> excluded
        for ci, cls in enumerate(('borehole', 'surface')):
            dd = d[d.cls == cls]
            x = dd.chan_band.map(FC) * (1.06 if ci else 0.94)
            ax.scatter(x, dd.amp, s=10, color=CAT[ci], alpha=0.5, lw=0)
            med = dd.groupby('chan_band').amp.median().reindex(BANDS)
            ax.plot([FC[b] for b in BANDS], med.values, '-o', ms=4, color=CAT[ci], lw=2, label=cls)
        ax.set_xscale('log'); ax.set_xticks([FC[b] for b in BANDS]); ax.set_xticklabels([b.replace('Hz', '') for b in BANDS], fontsize=7)
        ax.minorticks_off()
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(lab, fontsize=8, loc='left')
        lo, hi = np.nanpercentile(d.amp, [3, 97]) if len(d) else (0, 1)
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    axs[0, 0].legend(fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('band (Hz); higher = shallower')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_attr_bands.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 4 decline attribution
def fig_decline():
    rows = []
    for (ds, band) in [('SC10', b) for b in BANDS] + [('LFE', None), ('REP', None), ('LP10', None)]:
        d = C[(C.dataset == ds) & ((C.chan_band == band) if band else True)]
        for (sta, ch), g in d.groupby(['sta', 'chan_band']):
            if not g.chg_identifiable.astype(bool).all():
                continue                       # free instrument offset inside 2021-2025: decline not identifiable
            v = g.set_index('component').amp
            rg = g.set_index('component')
            if 'obs_chg_2125' not in v or 'chg_2125[TOTAL]' not in v:
                continue
            rows.append(dict(ds=ds, band=band or ch, sta=sta, obs=v['obs_chg_2125'], tot=v['chg_2125[TOTAL]'],
                             Q=v.get('chg_2125[QUAKES]', 0.0), Lt=v.get('chg_2125[LONGTERM]', 0.0), E=v.get('chg_2125[ENV]', 0.0),
                             Qlo=rg.range_lo.get('chg_2125[QUAKES]', np.nan), Qhi=rg.range_hi.get('chg_2125[QUAKES]', np.nan),
                             Llo=rg.range_lo.get('chg_2125[LONGTERM]', np.nan), Lhi=rg.range_hi.get('chg_2125[LONGTERM]', np.nan)))
    R = pd.DataFrame(rows)
    if R.empty:
        return
    R = R.fillna(0.0)
    sets = [('SC10 1-3Hz', R[(R.ds == 'SC10') & (R.band == '1-3Hz')]), ('SC10 0.3-0.6Hz', R[(R.ds == 'SC10') & (R.band == '0.3-0.6Hz')]),
            ('SC10 0.1-0.3Hz', R[(R.ds == 'SC10') & (R.band == '0.1-0.3Hz')]), ('LFE coda / REP / LP', R[R.ds.isin(['LFE', 'REP', 'LP10'])])]
    fig, axs = plt.subplots(len(sets), 1, figsize=(12, 3.0 * len(sets)))
    for ax, (title, d) in zip(axs, sets):
        d = d.sort_values('obs').reset_index(drop=True)
        x = np.arange(len(d))
        wdt = 0.26
        ax.bar(x - wdt, d.Q, wdt * 0.92, color=CAT[7], label='quake terms')
        ax.bar(x, d.Lt, wdt * 0.92, color=CAT[6], label='trend / ramp')
        ax.bar(x + wdt, d.E, wdt * 0.92, color=CAT[0], label='environment')
        ax.vlines(x - wdt, d.Qlo, d.Qhi, color=INK2, lw=1.0)
        ax.vlines(x, d.Llo, d.Lhi, color=INK2, lw=1.0, label='range over core-model subsets (dBIC<6)')
        ax.plot(x, d.obs, 'D', ms=5, color=INK, label='observed 2025-2021')
        ax.plot(x, d.tot, '_', ms=12, mew=2, color=CAT[1], label='model total')
        ax.set_xticks(x); ax.set_xticklabels([f'{a}\n{b}' if t.startswith('LFE') else a for a, b, t in zip(d.sta, d.band, [title] * len(d))], fontsize=6, rotation=90)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_ylabel('% change'); ax.set_title(title, loc='left', fontsize=9)
    axs[0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_attr_decline.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 5 single-driver tests
def fig_drivers():
    t = T[T.dataset == 'SC10'].copy()
    groups = ['annual', 'hyd_exp', 'hyd_exp_era5', 'hyd_diff', 'rain_lags', 'soilm', 'streamflow', 'thermal', 'gnss_up', 'gnss_dil', 'bstrain', 'micro', 'pressure', 'sealevel']
    M = t.pivot_table(index='group', columns='chan_band', values='dcv_rel', aggfunc='median').reindex(index=groups, columns=BANDS) * 100
    W = t.assign(win=(t.dcv_rel < -0.005) & (t.dbic < -2)).pivot_table(index='group', columns='chan_band', values='win', aggfunc='mean').reindex(index=groups, columns=BANDS)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    v = np.nanmax(np.abs(M.values)) if np.isfinite(M.values).any() else 1
    im = ax.imshow(M.values, cmap='RdBu', vmin=-v, vmax=v, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M.values[i, j]):
                ax.text(j, i, f'{M.values[i, j]:.0f}%\n{W.values[i, j] * 100:.0f}%', ha='center', va='center', fontsize=6.5, color=INK)
    ax.set_xticks(range(len(BANDS))); ax.set_xticklabels(BANDS); ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups)
    ax.grid(False)
    plt.colorbar(im, ax=ax, label='median change in CV RMS when added alone to the core model (%)')
    ax.set_title('Each driver alone (SC noise): median dCV; second line = fraction of stations where it passes', fontsize=8, loc='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_attr_drivers.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 6 geodesy
def fig_geodesy():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ax = axs[0]
    for k, (nm, col) in enumerate((('eta_gnss_dil', CAT[0]), ('eta_bstrain', CAT[1]))):
        d = C[(C.component == nm) & (C.dataset == 'SC10')]
        x = d.chan_band.map(FC) * (0.94 if k == 0 else 1.06)
        ax.errorbar(x, d.amp, d.sd, fmt='o', ms=3, color=col, alpha=0.6, lw=0.8, label=nm.replace('eta_', 'eta: '))
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=100); ax.axhline(0, color=INK2, lw=0.6); ax.minorticks_off()
    ax.set_xticks([FC[b] for b in BANDS]); ax.set_xticklabels([b.replace('Hz', '') for b in BANDS], fontsize=7)
    ax.set_title('strain sensitivity eta = (dv/v)/strain, single-driver fits', fontsize=8, loc='left'); ax.legend(fontsize=7)
    ax = axs[1]
    d = S[S.dataset == 'SC10'].copy()
    d['dS'] = d.q_strain_all_bic - d.q_pgv_all_bic
    for k, b in enumerate(BANDS):
        dd = d[d.chan_band == b]
        ax.scatter(np.full(len(dd), k) + np.random.default_rng(k).uniform(-0.2, 0.2, len(dd)), dd.dS, s=10, color=CAT[0], alpha=0.6)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(BANDS))); ax.set_xticklabels(BANDS, fontsize=7)
    ax.set_title('BIC(static strain) - BIC(PGV) for all-quake models\n(<0: static strain preferred)', fontsize=8, loc='left')
    ax = axs[2]
    js = f'{OUT}/afterslip_tau.json'
    if os.path.exists(js):
        A = json.load(open(js))
        for form, col in (('log', CAT[2]), ('exp', CAT[3])):
            p = np.array(A[form]['sse']); ax.plot(A['taus'], (p - p.min()) / (p.max() - p.min() + 1e-12), '-o', ms=3, color=col, label=f'GNSS afterslip ({form})')
    pr = C[(C.component == 'drop0_M7')].param.fillna('')
    iv = pr.str.extract(r'95% (\d+)-(\d+) d').astype(float).dropna()
    ylev = np.linspace(0.05, 0.95, max(len(iv), 1))
    for yv, (_, (a_, b_)) in zip(ylev, iv.iterrows()):
        ax.plot([a_, b_], [yv, yv], color=CAT[7], lw=1.5, alpha=0.7)
    ax.plot([], [], color=CAT[7], label=f'dv/v M7 recovery tau, 95% interval (constrained: {len(iv)} of {len(pr)} series)')
    ax.set_xscale('log'); ax.set_xlabel('time constant (days)'); ax.legend(fontsize=7)
    ax.set_title('M7 recovery time constants: GNSS afterslip vs dv/v', fontsize=8, loc='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_attr_geodesy.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 7 out-of-sample
def fig_oos():
    fn = f'{OUT}/out_of_sample.csv'
    if not os.path.exists(fn):
        return
    O = pd.read_csv(fn)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, v in enumerate(('env_only', 'pgv_dose+env')):
        d = O[O.variant == v]
        if 'obs_chg' in d:
            axs[0].scatter(d.obs_chg, d.pred_chg, s=14, color=CAT[k], label=v, alpha=0.7)
    lim = np.nanpercentile(np.abs(O[['obs_chg', 'pred_chg']].values), 97) if 'obs_chg' in O else 0.5
    axs[0].plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8); axs[0].set_xlim(-lim, lim); axs[0].set_ylim(-lim, lim)
    axs[0].set_xlabel('observed 2025 - 2021H1 (%)'); axs[0].set_ylabel('predicted from pre-2021-07 fit (%)'); axs[0].legend(fontsize=7)
    axs[0].set_title('out-of-sample long-term change', fontsize=8, loc='left')
    for k, v in enumerate(('env_only', 'pgv_dose+env')):
        d = O[O.variant == v]
        axs[1].scatter(d.rms_post_const, d.rms_post_offset, s=14, color=CAT[k], label=v, alpha=0.7)
    m = np.nanmax(O[['rms_post_const', 'rms_post_offset']].values)
    axs[1].plot([0, m], [0, m], color=INK2, lw=0.8)
    axs[1].set_xlabel('post-2021 RMS about pre-2021 median (%)'); axs[1].set_ylabel('post-2021 RMS of prediction, free offset (%)')
    axs[1].set_title('out-of-sample short-term skill (below line = skill)', fontsize=8, loc='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_attr_oos.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig_examples, fig_selection, fig_bands, fig_decline, fig_drivers, fig_geodesy, fig_oos):
        try:
            f(); print('ok', f.__name__, flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
