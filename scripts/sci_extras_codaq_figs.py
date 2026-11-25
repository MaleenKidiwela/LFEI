#!/usr/bin/env python3
"""Figures for the coda-Q-through-time test (scripts/sci_extras_codaq.py outputs).
-> figures/sci_extras_codaq_timeseries.png, figures/sci_extras_codaq_steps.png"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/extras/codaq'
FIG = f'{ROOT}/figures'
QUAKES = {'2010 M6.5': '2010-01-10', '2014 M6.8': '2014-03-10', 'Petrolia': '2021-12-20', 'Ferndale': '2022-12-20',
          'M7': '2024-12-05'}
COL = {'all_both': '#1f4e79', 'borehole_both': '#c55a11', 'surface_both': '#548235'}


def timeseries():
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, mode, title in ((axs[0], 'coh', 'coherent coda of linear LFE stacks (quarterly)'),
                            (axs[1], 'coh', 'coherent coda, yearly + trend'),
                            (axs[2], 'env', 'incoherent single-LFE energy coda (quarterly; low SNR)')):
        try:
            if ax is axs[1]:
                d = pd.read_csv(f'{OUT}/{mode}_yearly.csv')
                d['t'] = pd.to_datetime(d.year.astype(str) + '-07-01')
            else:
                d = pd.read_csv(f'{OUT}/{mode}_quarterly.csv')
                d['t'] = pd.to_datetime(d.t0) + pd.Timedelta(days=45)
        except FileNotFoundError:
            continue
        for g, c in COL.items():
            x = d[d.group == g]
            ax.errorbar(x.t, 100 * x.dq_rel, yerr=100 * x.se_formal, fmt='o-', ms=3, lw=0.8, color=c,
                         label=g.replace('_both', ''), capsize=0, alpha=0.85)
        if ax is axs[1]:
            try:
                d2 = pd.read_csv(f'{OUT}/coh137_yearly.csv')
                d2['t'] = pd.to_datetime(d2.year.astype(str) + '-07-01')
                for g, c in COL.items():
                    x = d2[d2.group == g]
                    ax.plot(x.t, 100 * x.dq_rel, '--', color=c, lw=1.2, alpha=0.8,
                            label=g.replace('_both', '') + ' (meancc>=0.137 events)')
            except FileNotFoundError:
                pass
            lt = pd.read_csv(f'{OUT}/{mode}_longterm.csv')
            r = lt[(lt.group == 'all_both') & (lt.kind == 'trend_per_yr_2009_2026')]
            if len(r):
                ax.set_title(f'{title}: trend all = {100 * r.value.iloc[0]:+.2f} +/- {100 * r.se_scatter.iloc[0]:.2f} %/yr')
            else:
                ax.set_title(title)
        else:
            ax.set_title(title)
        for k, v in QUAKES.items():
            ax.axvline(pd.Timestamp(v), color='k', ls=':', lw=0.8)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_ylabel('dQc^-1 / Qc^-1 (%)')
        ax.legend(fontsize=8, ncol=3)
    lim = axs[0].get_ylim()
    for k, v in QUAKES.items():
        axs[0].text(pd.Timestamp(v), lim[1], k, fontsize=7, rotation=90, va='top', ha='right')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_extras_codaq_timeseries.png', dpi=130)


def steps():
    tests = ['eureka2010', 'mendo2014', 'petrolia', 'ferndale', 'm7_restart60', 'm7_1yr']
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mode in zip(axs, ['coh', 'env']):
        try:
            S = pd.read_csv(f'{OUT}/{mode}_steps.csv')
            P = pd.read_csv(f'{OUT}/{mode}_placebo_steps.csv.gz')
        except FileNotFoundError:
            continue
        for gi, g in enumerate(COL):
            for ti, t in enumerate(tests):
                x = ti + (gi - 1) * 0.25
                pv = P[(P.test == t) & (P.group == g)].dq_rel.dropna().values * 100
                if len(pv) > 5:
                    lo, hi = np.percentile(pv, [2.5, 97.5])
                    q1, q3 = np.percentile(pv, [16, 84])
                    ax.plot([x, x], [lo, hi], color=COL[g], lw=1, alpha=0.4)
                    ax.plot([x, x], [q1, q3], color=COL[g], lw=5, alpha=0.3)
                r = S[(S.test == t) & (S.group == g)]
                if len(r) and np.isfinite(r.dq_rel.iloc[0]):
                    ax.plot(x, 100 * r.dq_rel.iloc[0], 'D', color=COL[g], ms=6,
                            label=g.replace('_both', '') if ti == 0 else None)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_xticks(range(len(tests)))
        ax.set_xticklabels(tests, rotation=25, fontsize=8)
        ax.set_ylabel('step in Qc^-1 (%)')
        ax.set_title(f'{mode}: real steps (diamonds) vs placebo 68/95 % ranges')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_extras_codaq_steps.png', dpi=130)


if __name__ == '__main__':
    timeseries()
    steps()


def summary():
    """combined summary: per mode, steps (placebo), long-term contrasts with a yearly-scatter SE (honest: the
    split-date placebos overlap heavily), and per-station post-2021 contrasts."""
    import json
    out = {}
    for mode in ('coh', 'coh137', 'env'):
        try:
            S = pd.read_csv(f'{OUT}/{mode}_steps.csv')
            Y = pd.read_csv(f'{OUT}/{mode}_yearly.csv')
            LT = pd.read_csv(f'{OUT}/{mode}_longterm.csv')
            SY = pd.read_csv(f'{OUT}/{mode}_station_yearly.csv')
            hdr = json.load(open(f'{OUT}/summary_{mode}.json'))
        except FileNotFoundError:
            continue
        g3 = ['all_both', 'borehole_both', 'surface_both']
        m = dict(n_used=hdr['n_used'], n_stations=hdr['n_stations'], n_families=hdr['n_families'],
                 Qc_ref_median=hdr['Qc_ref_median'])
        m['steps'] = S[S.group.isin(g3)][['test', 'group', 'dq_rel', 'se_formal', 'nchan', 'placebo_sd', 'z_placebo',
                                         'pct_placebo', 'inject_recovery_placebo', 'detect_2sigma_rel']] \
            .round(4).replace({np.nan: None}).to_dict('records')
        lt = []
        for g in g3:
            y = Y[(Y.group == g) & Y.dq_rel.notna()]
            post = y[y.year >= 2022].dq_rel.values
            pre = y[(y.year >= 2015) & (y.year <= 2021)].dq_rel.values
            resid = np.r_[post - post.mean(), pre - pre.mean()]
            sd = np.sqrt(np.sum(resid ** 2) / max(len(resid) - 2, 1))
            se = sd * np.sqrt(1 / len(post) + 1 / len(pre))
            r = LT[(LT.group == g)].set_index('kind')
            lt.append(dict(group=g, yearly_2022_26_minus_2015_21=float(post.mean() - pre.mean()), se_yearly_scatter=se,
                           z_yearly=float((post.mean() - pre.mean()) / se),
                           contrast_split_placebo=r.loc['post2021_vs_7yr_before'].replace({np.nan: None}).to_dict()
                           if 'post2021_vs_7yr_before' in r.index else None,
                           trend_per_yr=float(r.loc['trend_per_yr_2009_2026', 'value']),
                           trend_se=float(r.loc['trend_per_yr_2009_2026', 'se_scatter'])))
        m['longterm'] = lt
        st = []
        for s_, y in SY.groupby('station'):
            y = y[y.dq_rel.notna()]
            post, pre = y[y.year >= 2022].dq_rel.values, y[(y.year >= 2015) & (y.year <= 2021)].dq_rel.values
            if len(post) >= 2 and len(pre) >= 3:
                resid = np.r_[post - post.mean(), pre - pre.mean()]
                sd = np.sqrt(np.sum(resid ** 2) / max(len(resid) - 2, 1))
                se = sd * np.sqrt(1 / len(post) + 1 / len(pre))
                st.append(dict(station=s_, d=float(post.mean() - pre.mean()), se=float(se),
                               z=float((post.mean() - pre.mean()) / se)))
        stdf = pd.DataFrame(st)
        stdf.to_csv(f'{OUT}/{mode}_station_longterm.csv', index=False, float_format='%.5g')
        m['station_longterm_frac_z_lt_-2'] = float(np.mean(stdf.z < -2)) if len(stdf) else None
        m['station_longterm_n'] = int(len(stdf))
        out[mode] = m
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    for mode, m in out.items():
        print(mode, json.dumps(m['longterm'], default=float)[:1500])
        print('  stations z<-2 frac', m['station_longterm_frac_z_lt_-2'], 'n', m['station_longterm_n'])


if __name__ == '__main__':
    summary()
