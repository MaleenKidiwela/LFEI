#!/usr/bin/env python3
"""Figures and summary.json for sci_transient_source.py (reads the CSVs written by --prep / --analyze)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_transient_source_common import AX, DAY, FAMS, M7, OUT  # noqa: E402

COL = {23: 'C0', 8: 'C1', 27: 'C2'}


def binned_track(T, width=60.0, t0=-540, t1=570):
    m = T.iso.values & ~T.ovl10.values & T['loc'].values
    rows = []
    for a in np.arange(t0, t1, width):
        k = m & (T.day.values >= a) & (T.day.values < a + width)
        if k.sum() < 15:
            continue
        x = T.loc[k, AX].values
        rows.append(dict(day=a + width / 2, n_ev=k.sum(), **{c: x[:, i].mean() for i, c in enumerate(AX)},
                         **{c + '_se': x[:, i].std() / np.sqrt(k.sum()) for i, c in enumerate(AX)}))
    return pd.DataFrame(rows)


def qc_control():
    """Apparent dv/v of time-matched splits by event quality (control), vs the along-strike location split."""
    rows = []
    S = pd.read_csv(f'{OUT}/task2_sensitivity.csv')
    for F in FAMS:
        f = f'{OUT}/fam{F}_split_eps_qc.parquet'
        if not os.path.exists(f):
            continue
        Q = pd.read_parquet(f)
        E = pd.read_parquet(f'{OUT}/fam{F}_split_eps.parquet')
        keep = pd.read_csv(f'data/sci/dvv_depth/fam{F}_channels.csv')
        keep = set(keep.chan[(keep.b28_early_ccref >= 0.25) & (keep.b28_late_ccref >= 0.25)])
        sd = E[E.split.str.startswith('Fnull')].groupby(['chan', 'window']).eps.std().rename('sd')
        for grp, sel in [('surface', lambda c: ~c.str.startswith('PB.')), ('borehole', lambda c: c.str.startswith('PB.'))]:
            q = Q[sel(Q.chan) & Q.chan.isin(keep) & (Q.cc > 0.2)].join(sd, on=['chan', 'window'])
            q = q[np.isfinite(q.sd) & (q.sd > 0)]
            for (sp, win), g in q.groupby(['split', 'window']):
                w = 1 / g.sd ** 2
                s_row = S[(S.fam == F) & (S.group == grp) & (S.window == win)]
                rows.append(dict(fam=F, group=grp, split=sp, window=win, eps_pct=np.sum(w * g.eps) / w.sum() * 100,
                                 null_sd_pct=s_row.null_sd_F_pct.iloc[0] if len(s_row) else np.nan,
                                 eps_Fs_pct=s_row.eps_Fs_pct.iloc[0] if len(s_row) else np.nan))
    Q = pd.DataFrame(rows)
    Q.to_csv(f'{OUT}/task2_quality_split_control.csv', index=False)
    return Q


def pooled_g(S, col, grp, win):
    d = S[(S.group == grp) & (S.window == win) & np.isfinite(S[col]) & np.isfinite(S[col + '_se'])]
    if d.empty:
        return np.nan, np.nan
    w = 1 / d[col + '_se'] ** 2
    m = np.sum(w * d[col]) / w.sum()
    se = np.sqrt(1 / w.sum())
    chi = np.sum(w * (d[col] - m) ** 2) / max(len(d) - 1, 1)
    return m, se * np.sqrt(max(chi, 1.0))


def report():
    T1 = pd.read_csv(f'{OUT}/task1_centroid_shifts.csv')
    SP = pd.read_csv(f'{OUT}/task1_spread_AxB.csv')
    PL = pd.read_csv(f'{OUT}/task1_placebo.csv')
    S = pd.read_csv(f'{OUT}/task2_sensitivity.csv')
    INFO = pd.read_csv(f'{OUT}/task2_true_vs_measured_spread.csv')
    R = pd.read_csv(f'{OUT}/task3_matched_transient.csv')
    SY = pd.read_csv(f'{OUT}/task4_predicted_transient.csv')
    LS = pd.read_csv(f'{OUT}/task1_location_transient.csv')
    Q = qc_control()
    plsd = PL.groupby('fam')[['ds', 'dd', 'dn']].std()

    # ------------------------------------------------ figure 1: location
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for k, c in enumerate(['s', 'd', 'n']):
        a = ax[0, k]
        for F in FAMS:
            T = pd.read_parquet(f'{OUT}/fam{F}_events.parquet')
            tr = binned_track(T)
            ref = T.iso & ~T.ovl10 & T['loc'] & (T.day >= -365) & (T.day < 0)
            y = tr[c] - T.loc[ref, c].mean()
            a.errorbar(tr.day, y, tr[c + '_se'], fmt='o-', ms=3, color=COL[F], label=f'fam {F}', capsize=2)
        a.axvline(0, color='k', ls='--', lw=0.8)
        a.axvspan(11, 85, color='0.9', zorder=0, label='restart..restart+60 d')
        a.axhline(0, color='0.5', lw=0.5)
        a.set_xlabel('days after the M7 (60-d bins)')
        a.set_ylabel(f'centroid {c} - pre-M7 year mean (m)')
        a.set_title({'s': 'along strike (+ WNW, 290 deg)', 'd': 'down-dip (+ NNE / deeper)', 'n': 'plane normal'}[c], fontsize=9)
        if k == 0:
            a.legend(fontsize=7)
    a = ax[1, 0]
    x0 = 0
    for F in FAMS:
        for j, c in enumerate(AX):
            for i_, (src, mk) in enumerate([('full', 'o'), ('A', '^'), ('B', 'v')]):
                r = T1[(T1.fam == F) & (T1.events == 'iso_nonovl') & (T1.comp == 'post_early60-ref') & (T1['loc'] == src)]
                if r.empty:
                    continue
                r = r.iloc[0]
                a.errorbar(x0 + j + 0.12 * (i_ - 1), r[f'd{c}'], r[f'd{c}_se'], fmt=mk, color=COL[F], ms=5,
                           label=(f'{src} network' if F == 23 and j == 0 else None))
            a.fill_between([x0 + j - 0.3, x0 + j + 0.3], -plsd.loc[F, f'd{c}'], plsd.loc[F, f'd{c}'], color=COL[F], alpha=0.15, lw=0)
        x0 += 3.5
    a.set_xticks([j + 3.5 * i for i in range(3) for j in range(3)])
    a.set_xticklabels([f'{c}\nfam{F}' for F in FAMS for c in AX], fontsize=7)
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel('centroid shift, first 60 d after restart - reference (m)')
    a.set_title('Task 1: shift vs pre-M7 year + >182 d (bars: bootstrap SE; band: placebo 1 sd)', fontsize=8)
    a.legend(fontsize=7)
    a = ax[1, 1]
    for F in FAMS:
        d = SP[SP.fam == F]
        order = {'pre1y': 0, 'post_early60': 1, 'post_60_182': 2, 'post_late': 3}
        xx = d.period.map(order) + 0.1 * FAMS.index(F)
        a.errorbar(xx, d.var_true_s, d.var_true_s_se, fmt='o', color=COL[F], label=f'fam {F} (s)')
        a.errorbar(xx + 0.04, d.var_true_n, d.var_true_n_se, fmt='x', color=COL[F], alpha=0.6)
    a.set_xticks(range(4))
    a.set_xticklabels(['pre-M7 yr', '0-60 d\nafter restart', '60-182 d', '>182 d'], fontsize=8)
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel('true variance from A x B (m$^2$); o = s, x = n')
    a.set_title('Patch spread (A x B cross-covariance)', fontsize=9)
    a.legend(fontsize=7)
    a = ax[1, 2]
    ls = LS[(LS.stacks == 'orig') & (LS.group == 'surface') & (LS.cfg == 'b28_early')]
    a.errorbar(range(3), ls.loc_transient_m, ls.se_m, fmt='ks', capsize=3)
    a.set_xticks(range(3))
    a.set_xticklabels([f'{c}' for c in ls.axis])
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel('60-d transient amplitude of the stack centroid (m)')
    a.set_title('Location "transient": same parametric fit as dv/v,\napplied to per-stack centroids (3 families, surface rows)', fontsize=8)
    fig.tight_layout()
    fig.savefig('figures/sci_transient_source_location.png', dpi=130)
    plt.close(fig)

    # ------------------------------------------------ figure 2: sensitivity and prediction
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    wins = ['w3_5', 'w5_8', 'w8_11', 'w11_16']
    a = ax[0]
    for grp, mk in [('surface', 'o'), ('borehole', 's')]:
        for F in FAMS:
            d = S[(S.fam == F) & (S.group == grp) & S.window.isin(wins)].sort_values('tau_mid')
            a.errorbar(d.tau_mid + 0.15 * FAMS.index(F) + (0.5 if grp == 'borehole' else 0), d.gemp_s_F * 1e3,
                       d.gemp_s_F_se * 1e3, fmt=mk + '-', color=COL[F], ms=4, mfc='none' if grp == 'borehole' else COL[F],
                       label=f'fam {F} {grp}', lw=0.8)
    tt = np.linspace(3.5, 14, 50)
    a.axhline(0, color='0.5', lw=0.5)
    a.set_xlabel('lapse time after S, window centre (s)')
    a.set_ylabel('apparent dv/v per 10 m along-strike source offset (%)')
    a.set_title('Task 2: empirical sensitivity (time-matched pre-M7 location splits,\ntrue offset from A x B)', fontsize=9)
    a.legend(fontsize=6, ncol=2)
    a = ax[1]
    for grp, mk in [('surface', 'o'), ('borehole', 's')]:
        for F in FAMS:
            d = S[(S.fam == F) & (S.group == grp) & S.window.isin(wins)].sort_values('tau_mid')
            a.errorbar(d.tau_mid + 0.15 * FAMS.index(F), d.c_per_s, d.c_se, fmt=mk, color=COL[F], ms=4,
                       mfc='none' if grp == 'borehole' else COL[F])
    tt = np.linspace(3, 16, 50)
    a.plot([4, 6.5, 9.5, 13.5], [S[S.window == w].tau_eff_inv.median() for w in wins], 'k--', label='1/tau_eff (coda fixed, S moves)')
    a.axhline(0, color='0.5', lw=0.5)
    a.set_xlabel('lapse time after S, window centre (s)')
    a.set_ylabel('apparent dv/v per second of differential direct-S delay (1/s)')
    a.set_title('S-alignment mechanism: measured c vs the 1/tau bound', fontsize=9)
    a.legend(fontsize=7)
    a = ax[2]
    x = 0
    ticks = []
    for grp in ['surface', 'borehole']:
        for cfg in ['b28_early', 'b28_late']:
            for v, col in [('orig', 'k'), ('loc', '0.4'), ('match', 'C3')]:
                r = R[(R.group == grp) & (R.cfg == cfg) & (R.variant == v)].iloc[0]
                a.errorbar(x, r.transient60_pct, r.se, fmt='o', color=col, label=f'observed: {v} stacks' if not ticks else None, capsize=2)
                x += 0.25
            for gt, col, mk in [('emp_s_F', 'C0', 'D'), ('emp_s_A', 'C9', 'D'), ('hyb_model', 'C2', '^'), ('hyb_bound', 'C1', 'v'),
                                ('bound', 'C5', 'x')]:
                r = SY[(SY.group == grp) & (SY.cfg == cfg) & (SY.stacks == 'orig') & (SY.g_type == gt)]
                if r.empty:
                    continue
                r = r.iloc[0]
                a.errorbar(x, r.pred_transient_pct, r.pred_se, fmt=mk, color=col, capsize=2,
                           label={'emp_s_F': 'pred.: along-strike g (F split)', 'emp_s_A': 'pred.: along-strike g (A split, B-true)',
                                  'hyb_model': 'pred.: + S-delay model for d, n', 'hyb_bound': 'pred.: + 1/tau bound for d, n',
                                  'bound': 'pred.: 1/tau bound all axes (s falsified)'}[gt] if not ticks else None)
                x += 0.25
            ticks.append((x - 0.9, f'{grp}\n{cfg.replace("b28_", "")}'))
            x += 0.6
    a.set_xticks([t[0] for t in ticks])
    a.set_xticklabels([t[1] for t in ticks], fontsize=8)
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel('60-d M7 transient amplitude (%)')
    a.set_title('Tasks 3-4: observed transient (orig / located-only / location-matched)\nvs transient predicted from the location change', fontsize=9)
    a.legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig('figures/sci_transient_source_prediction.png', dpi=130)
    plt.close(fig)

    # ------------------------------------------------ summary
    summ = dict(
        frame='strike 290, dip 45 NNE; s along strike (+WNW), d down-dip, n normal; metres',
        families=FAMS,
        task1_shift_post_early60_vs_ref=T1[(T1.events == 'iso_nonovl') & (T1.comp == 'post_early60-ref')].to_dict('records'),
        task1_placebo_sd_m=plsd.round(1).to_dict('index'),
        task1_spread_AxB=SP.to_dict('records'),
        task1_location_transient_m=LS.to_dict('records'),
        task2_true_vs_measured=INFO.to_dict('records'),
        task2_pooled_gemp_s_pct_per_10m={f'{g}_{w}_{p}': [v * 1e3 for v in pooled_g(S, f'gemp_s_{p}', g, w)]
                                         for g in ['surface', 'borehole'] for w in ['early', 'late', 'w3_5', 'w5_8', 'w8_11', 'w11_16']
                                         for p in ['F', 'A']},
        task2_c_per_s=S[['fam', 'group', 'window', 'c_per_s', 'c_se', 'tau_eff_inv', 'kappa', 'kappa_se']].to_dict('records'),
        task2_quality_control=Q.to_dict('records'),
        task3=R.to_dict('records'),
        task4=SY.to_dict('records'),
        observed_reference=dict(surface_early=-0.177, surface_early_se=0.035, surface_late=-0.017, surface_late_se=0.029),
    )
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))
    print(json.dumps(summ['task2_pooled_gemp_s_pct_per_10m'], indent=0, default=float))
    print(Q.round(4).to_string())
    return summ
