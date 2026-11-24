#!/usr/bin/env python3
"""Summary of the deep/shallow dv/v separation -> data/sci/dvv_deep_shallow/summary.json (+ printed digest).
Reads stage2_deep.csv, stage2_replicates.pkl, station_directS_coda.csv, swin_station_terms.csv, clock_steps.csv,
synthetic_recovery.csv, rep/rep_fit.csv.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/scripts')
OUT = 'data/sci/dvv_deep_shallow'
BH = ['PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B934']
SF = ['NC.KMR', 'NC.KMPB', 'BK.RBOW', 'BK.HAYF', 'BK.BRIC']
LONG = BH + ['NC.KMR', 'NC.KMPB']


def r4(x):
    return None if x is None or not np.isfinite(x) else float(np.round(x, 4))


def deep_table():
    O = pd.read_csv(f'{OUT}/stage2_deep.csv')
    R = pickle.load(open(f'{OUT}/stage2_replicates.pkl', 'rb'))
    rows = []
    for (X, model, var, h), v in R.items():
        if model == 'joint' and not (var == 'ref/l100' and h in (0.1, 0.3, 1.0)) and not (h == 0.3):
            continue
        p = v['plac'][np.isfinite(v['plac'])]
        b = v['boot'][np.isfinite(v['boot'])]
        # robust placebo: drop the single most extreme shift (the 2008-2012 era with known instrument problems)
        pr = np.delete(p, np.argmax(np.abs(p - np.median(p)))) if len(p) > 3 else p
        se_b, se_p, se_pr = np.std(b), np.std(p), np.std(pr)
        se_h = max(se_b, se_pr)
        rows.append(dict(X=X, model=model, variant=var, h_km=h, dd_pct=v['real'] * 100, se_boot=se_b * 100,
                         se_plac_all=se_p * 100, se_plac_robust=se_pr * 100, plac_mean_robust=np.mean(pr) * 100,
                         se_honest=se_h * 100, lo95=(v['real'] - 1.96 * se_h) * 100, hi95=(v['real'] + 1.96 * se_h) * 100,
                         z=v['real'] / se_h if se_h > 0 else np.nan))
    T = pd.DataFrame(rows)
    extra = O[['X', 'model', 'variant', 'h_km', 'beta', 'beta_se_boot', 'beta_se_plac', 'beta_surf', 'beta_surf_se_boot',
               'chi_s', 'chi_c']]
    T = T.merge(extra, on=['X', 'model', 'variant', 'h_km'], how='left')
    T.to_csv(f'{OUT}/deep_estimates.csv', index=False)
    return T


def ratio_fit(g, col='d_ms', se_cols=('se_boot_ms', 'se_plac_ms')):
    se = np.fmax(g[se_cols[0]], g[se_cols[1]].fillna(0))
    w = 1 / se ** 2
    A = np.c_[np.ones(len(g)), g.coda_full_pct]
    W = np.sqrt(w.values)[:, None]
    x, *_ = np.linalg.lstsq(A * W, g[col].values * W[:, 0], rcond=None)
    r = g[col].values - A @ x
    chi = np.sum(w * r ** 2) / max(len(g) - 2, 1)
    cov = np.linalg.inv((A * W).T @ (A * W)) * max(chi, 1)
    return -x[1] * 0.1, np.sqrt(cov[1, 1]) * 0.1, chi        # s per unit coda dv/v (+ = delay when slower)


def tension_tables():
    T = pd.read_csv(f'{OUT}/swin_station_terms.csv')
    ST = pd.read_csv(f'{OUT}/station_directS_coda.csv')
    out = {}
    for X in ['D', 'm7']:
        for cls, L in [('borehole', BH), ('surface', SF), ('all', BH + SF)]:
            for obs in ['S', 'Son', 'Slate']:
                g = T[(T.X == X) & (T.obs == obs) & T.sta.isin(L)].merge(ST[ST.X == X][['sta', 'coda_full_pct']], on='sta')
                g = g[np.isfinite(g.d_ms) & np.isfinite(g.coda_full_pct)]
                if len(g) < 3:
                    continue
                b, se, chi = ratio_fit(g)
                loo = []
                for s_ in g.sta:
                    if len(g) > 3:
                        loo.append(ratio_fit(g[g.sta != s_])[0])
                out[f'{X}|{cls}|{obs}'] = dict(n=len(g), ratio_s=r4(b), ratio_se_s=r4(se), chi2r=r4(chi),
                                               loo_min=r4(min(loo)) if loo else None, loo_max=r4(max(loo)) if loo else None)
    return out


def shallow_from_directS():
    """implied near-surface dv/v at surface stations from their relative direct-S delay minus the borehole mean
    (boreholes: direct S does not cross the layer above the sensor)."""
    import sci_dvv_deep_shallow as m
    geo = m.get_geo('ref', '100')
    geo_none = m.get_geo('none', '100')
    ST = pd.read_csv(f'{OUT}/station_directS_coda.csv')
    rows = []
    for X in ['D', 'm7']:
        S = ST[ST.X == X].copy()
        S['se'] = np.fmax(S.dS_se_boot_ms, S.dS_se_plac_ms.fillna(0))
        b = S[S.sta.isin(BH)]
        wb = 1 / b.se ** 2
        bm = np.sum(wb * b.dS_ms) / wb.sum()
        bse = np.sqrt(max(1 / wb.sum(), np.var(b.dS_ms) / len(b)))
        for r in S[S.sta.isin(SF + ['BK.SPRL', 'BK.RVIT'])].itertuples():
            dt = r.dS_ms - bm
            se = np.hypot(r.se, bse)
            row = dict(X=X, sta=r.sta, delay_rel_borehole_ms=dt, se_ms=se, coda_full_pct=r.coda_full_pct)
            for h in (0.03, 0.1, 0.3):
                for nm, g in (('ref', geo), ('noLVZ', geo_none)):
                    t = g.tsh(r.sta, 0, h)
                    row[f'dvv_top{int(h * 1000)}m_{nm}_pct'] = -dt * 1e-3 / t * 100
                    row[f'dvv_top{int(h * 1000)}m_{nm}_se_pct'] = se * 1e-3 / t * 100
            rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/shallow_from_directS.csv', index=False)
    return T


def lapse_tables():
    ST = pd.read_csv(f'{OUT}/station_directS_coda.csv')
    out = {}
    for X in ['D', 'm7']:
        for cls, L in [('surface', SF), ('borehole', BH), ('deep_borehole', ['PB.B932', 'PB.B934', 'PB.B049']),
                       ('long_record_all', LONG)]:
            g = ST[(ST.X == X) & ST.sta.isin(L)]
            res = {}
            for w in ['w3_8', 'w8_15', 'w15_25', 'full']:
                v = g[f'coda_{w}_pct']
                se = (g[f'coda_{w}_se_pct'] if w != 'full'
                      else np.fmax(g.coda_full_se_boot_pct, g.coda_full_se_plac_pct.fillna(0)))
                ok = np.isfinite(v) & np.isfinite(se) & (se > 0)
                if ok.sum() == 0:
                    continue
                wt = 1 / se[ok] ** 2
                mu = np.sum(wt * v[ok]) / wt.sum()
                chi = np.sum(wt * (v[ok] - mu) ** 2) / max(ok.sum() - 1, 1)
                res[w] = dict(mean_pct=r4(mu), se_pct=r4(np.sqrt(max(chi, 1) / wt.sum())), n=int(ok.sum()),
                              birge=r4(np.sqrt(chi)))
            out[f'{X}|{cls}'] = res
    return out


def main():
    S = {}
    T = deep_table()
    key = T[T.model.isin(['DD', 'ratio', 'ratio2', 'ratio_bh']) | ((T.model == 'joint') & (T.variant == 'ref/l100'))]
    S['deep_direct_S'] = {f'{r.X}|{r.model}|{r.variant}|{r.h_km}': dict(
        dd_pct=r4(r.dd_pct), se_honest_pct=r4(r.se_honest), se_boot=r4(r.se_boot), se_plac_all=r4(r.se_plac_all),
        se_plac_robust=r4(r.se_plac_robust), lo95=r4(r.lo95), hi95=r4(r.hi95)) for r in key.itertuples()
        if r.X in ('m7', 'D', 'sin', 'cos', 'tr')}
    S['borehole_tension_ratios'] = tension_tables()
    SH = shallow_from_directS()
    S['shallow_from_directS'] = SH.round(4).to_dict('records')
    S['lapse'] = lapse_tables()
    if os.path.exists(f'{OUT}/clock_steps.csv'):
        C = pd.read_csv(f'{OUT}/clock_steps.csv')
        S['clock_steps_absz_ge4'] = C[C.max_step_z.abs() >= 4].round(3).to_dict('records')
    if os.path.exists(f'{OUT}/synthetic_recovery.csv'):
        S['synthetic_recovery'] = pd.read_csv(f'{OUT}/synthetic_recovery.csv').round(4).to_dict('records')
    if os.path.exists(f'{OUT}/rep/rep_fit.csv'):
        F = pd.read_csv(f'{OUT}/rep/rep_fit.csv')
        S['repeaters'] = F.round(4).to_dict('records')
        P = F.pivot(index='sta', columns='cls')
        rs = {}
        for q, se in [('m7_pct', 'm7_se_boot'), ('D_pct', 'D_se_boot')]:
            for nm, v, e in [('deep_sources', P[q]['deep'], P[se]['deep']), ('shallow_sources', P[q]['shallow'], P[se]['shallow']),
                             ('deep_minus_shallow', P[q]['deep'] - P[q]['shallow'], np.hypot(P[se]['deep'], P[se]['shallow']))]:
                w = 1 / e ** 2
                mu = np.sum(w * v) / w.sum()
                chi = np.sum(w * (v - mu) ** 2) / (len(v) - 1)
                rs[f'{q}|{nm}'] = dict(mean_pct=r4(mu), se_pct=r4(np.sqrt(max(chi, 1) / w.sum())), birge=r4(np.sqrt(chi)),
                                       n_sta=int(len(v)))
        S['repeaters_network'] = rs
        print(json.dumps(rs, indent=0))
    if os.path.exists(f'{OUT}/synthetic_recovery.csv'):
        Z = pd.read_csv(f'{OUT}/synthetic_recovery.csv')
        d = Z[Z.scenario == 'deep_only']
        S['synthetic_recovery_factor_deep_only'] = {f'{r.X}|{r.model}|{r.h_assumed}': r4(r.rec_minus_baseline_pct / r.true_dd_pct)
                                                    for r in d.itertuples()}
        sh = Z[Z.scenario.str.startswith('shallow_obs')]
        S['synthetic_max_shallow_leakage_pct'] = {f'{k[0]}|{k[1]}|{k[2]}': r4(g.rec_minus_baseline_pct.abs().max())
                                                  for k, g in sh.groupby(['X', 'model', sh.h_assumed.fillna(-1)])}
    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
    pd.set_option('display.width', 250)
    print(key[key.X.isin(['m7', 'D', 'sin', 'cos'])][['X', 'model', 'variant', 'h_km', 'dd_pct', 'se_boot', 'se_plac_all',
                                                     'se_plac_robust', 'se_honest', 'lo95', 'hi95']].round(4).to_string())
    print(json.dumps(S['borehole_tension_ratios'], indent=0))
    print(SH.round(3).to_string())
    print(json.dumps(S['lapse'], indent=0))


if __name__ == '__main__':
    main()
