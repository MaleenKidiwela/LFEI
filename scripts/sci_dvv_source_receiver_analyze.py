#!/usr/bin/env python3
"""Analysis + figures for scripts/sci_dvv_source_receiver.py (stage `analyze`).  Reads only this
investigation's own caches in data/sci/dvv_source_receiver/ (trusted pickles) and existing project tables.
Writes summary.json, family_table.csv, regressions.csv, common_bounds.csv and
figures/sci_dvv_source_receiver_{curves,family,nulls,common,synth,coherence}.png"""
import json
import os
import pickle          # own caches only (trusted)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sci_dvv_source_receiver as SR

OUT, FIG = SR.OUT, SR.FIG
PN = SR.PNAMES
iP = {p: k for k, p in enumerate(PN)}
STRIKE = 293.0
M7_EPI = (40.374, -125.022)          # USGS epicentre of the 2024-12-05 M7.0 (offshore Cape Mendocino)
SUMMARY = {}


def jl(x):
    if isinstance(x, dict):
        return {str(k): jl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jl(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return jl(x.tolist())
    return x


def ld(name):
    f = f'{OUT}/{name}'
    return pickle.load(open(f, 'rb')) if os.path.exists(f) else None


def family_covariates(fams):
    d = pd.read_csv('data/sci/dose_response/m7_per_family_stress_vs_gap.csv')
    d = d[d.quake.str.contains('M7')].set_index('family')
    c = pd.read_csv('data/patches/m7_cessation_vs_patch.csv').set_index('family')
    X = pd.DataFrame(index=fams)
    X['first_d'] = c.first_d.reindex(fams)
    X['tenth_d'] = c.tenth_d.reindex(fams)
    for k in ('dp', 'dtau', 'dsn', 'dsn_eff', 'dCFS04', 'dist_km', 'lon', 'lat', 'depth_km'):
        X[k] = d[k].reindex(fams)
    lat0, lon0 = X.lat.mean(), X.lon.mean()
    e = (X.lon - lon0) * 111.2 * np.cos(np.radians(lat0))
    n = (X.lat - lat0) * 111.2
    X['along_strike_km'] = e * np.sin(np.radians(STRIKE)) + n * np.cos(np.radians(STRIKE))
    X['dist_m7_epi_km'] = np.hypot((X.lon - M7_EPI[1]) * 111.2 * np.cos(np.radians(lat0)), (X.lat - M7_EPI[0]) * 111.2)
    return X


def wslope(x, y, w):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[m], y[m], w[m]
    xm = np.sum(w * x) / np.sum(w)
    return np.sum(w * (x - xm) * (y - np.sum(w * y) / np.sum(w))) / np.sum(w * (x - xm) ** 2)


# ============================================================================== family deviations
def ml_tau(d, se):
    """ML intrinsic scatter tau for d ~ N(mu, se^2 + tau^2); returns tau_hat, 95% upper (profile LR)."""
    grid = np.linspace(0, max(4 * np.std(d), 1e-4), 400)

    def nll(t):
        v = se ** 2 + t ** 2
        mu = np.sum(d / v) / np.sum(1 / v)
        return 0.5 * np.sum(np.log(v) + (d - mu) ** 2 / v)
    L = np.array([nll(t) for t in grid])
    i = int(np.argmin(L))
    ok = np.where((grid >= grid[i]) & (L - L[i] <= 1.92))[0]
    return grid[i], grid[ok.max()]


def family_tests(panel='balanced'):
    """Family deviations dS_F (identifiable).  Error calibration: bootstrap over stacks (validated by the
    pseudo-family split test for well-measured families) vs placebo dates vs family-label permutation;
    honest SE = max(bootstrap, placebo, permutation).  The equal-weight gauge sum_F dS_F = 0 lets poorly
    measured families shift every deviation, so deviations are re-centred on the precision-weighted mean
    (another valid gauge) and tested for homogeneity (Cochran Q, dof NF-1)."""
    pr, pl = ld(f'param_{panel}.pkl'), ld(f'placebo_{panel}.pkl')
    if pr is None or pl is None:
        return None
    fams = pr['fams']
    s, B = pr['s'] * 100, pr['boot'] * 100                    # % units; B (nboot, NF, NP)
    m7p = pl['m7_placebo'] * 100                              # (nd, NF, 2) step, tr
    dpl = pl['decl_placebo'] * 100                            # (nw, NF)
    pe = pl['perm'] * 100
    sd_ = pl['split_diff'] * 100
    Q = {'step_m7': (s[:, iP['step_m7']], B[:, :, iP['step_m7']], m7p[:, :, 0], pe[:, :, iP['step_m7']], sd_[:, :, iP['step_m7']]),
         'tr60_m7': (s[:, iP['tr60_m7']], B[:, :, iP['tr60_m7']], m7p[:, :, 1], pe[:, :, iP['tr60_m7']], sd_[:, :, iP['tr60_m7']]),
         'total_m7': (s[:, iP['step_m7']] + s[:, iP['tr60_m7']], B[:, :, iP['step_m7']] + B[:, :, iP['tr60_m7']],
                      m7p[:, :, 0] + m7p[:, :, 1], pe[:, :, iP['step_m7']] + pe[:, :, iP['tr60_m7']],
                      sd_[:, :, iP['step_m7']] + sd_[:, :, iP['tr60_m7']]),
         'decl4yr': (s[:, iP['decl4yr']], B[:, :, iP['decl4yr']], dpl, pe[:, :, iP['decl4yr']], sd_[:, :, iP['decl4yr']]),
         'step_fer': (s[:, iP['step_fer']], B[:, :, iP['step_fer']], m7p[:, :, 0], pe[:, :, iP['step_fer']], sd_[:, :, iP['step_fer']]),
         'step_pet': (s[:, iP['step_pet']], B[:, :, iP['step_pet']], m7p[:, :, 0], pe[:, :, iP['step_pet']], sd_[:, :, iP['step_pet']])}
    T = pd.DataFrame(index=fams)
    out = {}
    from scipy import stats
    for nm, (obs, bo, nul, per, spl) in Q.items():
        se_b, se_p, se_q = bo.std(0), nul.std(0), per.std(0)
        se = np.maximum.reduce([se_b, se_p, se_q])
        w = 1 / se ** 2
        mu_w = np.sum(w * obs) / np.sum(w)
        dev = obs - mu_w
        sdev = np.sqrt(np.clip(se ** 2 - 1 / np.sum(w), 1e-12, None))
        z = dev / sdev
        T[nm] = obs
        T[nm + '_wc'] = dev                                   # re-centred on the precision-weighted mean
        T[nm + '_se'] = se
        T[nm + '_se_boot'], T[nm + '_sd_plc'], T[nm + '_sd_perm'] = se_b, se_p, se_q
        T[nm + '_z'] = z
        Qc = float(np.sum(w * dev ** 2))
        tau, tau95 = ml_tau(obs, se)
        good = se_b < np.percentile(se_b, 50)
        split_ratio = np.sqrt(np.mean(spl ** 2, axis=0)) / (2 * se_b)
        # bootstrap distribution of Q (centred replicates) -> calibration check of Q itself
        Qb = np.array([np.sum(w * ((bo[k] - bo.mean(0)) - np.sum(w * (bo[k] - bo.mean(0))) / np.sum(w)) ** 2) for k in range(len(bo))])
        Qp = np.array([np.sum(w * (per[k] - np.sum(w * per[k]) / np.sum(w)) ** 2) for k in range(len(per))])
        out[nm] = dict(weighted_mean_pct=mu_w, Q=Qc, dof=len(fams) - 1, p_chi2=float(stats.chi2.sf(Qc, len(fams) - 1)),
                       Q_boot_median=float(np.median(Qb)), p_Q_boot=float((1 + np.sum(Qb >= Qc)) / (len(Qb) + 1)),
                       Q_perm=Qp.tolist(), p_Q_perm=float((1 + np.sum(Qp >= Qc)) / (len(Qp) + 1)),
                       intrinsic_tau_pct=tau, intrinsic_tau_95_upper_pct=tau95,
                       median_se_boot=float(np.median(se_b)), median_sd_placebo=float(np.median(se_p)),
                       median_sd_perm=float(np.median(se_q)),
                       split_ratio_rms_diff_over_2se_boot_wellmeasured=float(np.median(split_ratio[good])),
                       split_ratio_all=float(np.median(split_ratio)),
                       max_abs_z=float(np.max(np.abs(z))), fam_max_abs_z=int(fams[int(np.argmax(np.abs(z)))]),
                       p_bonferroni_max=float(min(1, 2 * stats.norm.sf(np.max(np.abs(z))) * len(fams))),
                       n_abs_z_gt2=int(np.sum(np.abs(z) > 2)), n_abs_z_gt3=int(np.sum(np.abs(z) > 3)),
                       expected_n_abs_z_gt2=0.0455 * len(fams))
    X = family_covariates(fams)
    T = T.join(X)
    # regressions (PRIMARY: M7 step / transient / total vs silence duration and modelled stress)
    regs = []
    covs = ['tenth_d', 'first_d', 'dp', 'dtau', 'dsn_eff', 'dCFS04', 'along_strike_km', 'depth_km', 'dist_m7_epi_km']
    rng = np.random.default_rng(3)
    for nm in ('step_m7', 'tr60_m7', 'total_m7', 'decl4yr'):
        obs, bo, nul, per, _ = Q[nm]
        se = T[nm + '_se'].values
        w = 1 / se ** 2
        for cv in covs:
            x = T[cv].values.astype(float)
            if np.sum(np.isfinite(x)) < 8 or np.nanstd(x) == 0:
                continue
            b = wslope(x, obs, w)
            bb = np.array([wslope(x, bo[k], w) for k in range(len(bo))])
            bn = np.array([wslope(x, nul[j], w) for j in range(len(nul))])
            bp = np.array([wslope(rng.permutation(x), obs, w) for _ in range(2000)])
            sd_b = bb.std()
            m = np.isfinite(x)
            regs.append(dict(quantity=nm, covariate=cv,
                             primary=(nm in ('step_m7', 'tr60_m7', 'total_m7') and cv in ('tenth_d', 'first_d', 'dp')),
                             slope=b, slope_sd_boot=sd_b, z_boot=b / sd_b, p_boot=float(2 * stats.norm.sf(abs(b / sd_b))),
                             slope_sd_placebo=bn.std(), p_perm=(1 + np.sum(np.abs(bp) >= abs(b))) / (len(bp) + 1),
                             spearman=pd.Series(obs[m]).corr(pd.Series(x[m]), method='spearman'),
                             slope_95_abs_bound=abs(b) + 1.96 * max(sd_b, bn.std()), x_range=np.nanmax(x) - np.nanmin(x),
                             n=int(m.sum())))
    R = pd.DataFrame(regs)
    if len(R):
        R['p_use'] = np.maximum(R.p_boot, R.p_perm)
        pm = R[R.primary]
        o = np.argsort(pm.p_use.values)
        adj = np.empty(len(pm))
        run = 0
        for r_, i in enumerate(o):
            run = max(run, (len(pm) - r_) * pm.p_use.values[i])
            adj[i] = min(1, run)
        R.loc[pm.index, 'p_holm_primary'] = adj
    R.to_csv(f'{OUT}/regressions_{panel}.csv', index=False)
    T.rename_axis('family').to_csv(f'{OUT}/family_table_{panel}.csv')
    out['regressions_primary'] = R[R.primary].to_dict('records') if len(R) else []
    out['regressions_min_p'] = R.sort_values('p_use').head(8).to_dict('records') if len(R) else []
    return T, R, out


# ============================================================================== common term bounds
def comp_inflation(D, col, bcol):
    """components of one station sample (nearly) the same receiver change: their scatter about the station mean
    vs the bootstrap SD gives an empirical noise-inflation factor (>= 1)."""
    g = D.groupby('sta')
    num, den = 0.0, 0.0
    for _, q in g:
        if len(q) < 2:
            continue
        num += np.sum((q[col] - q[col].mean()) ** 2)
        den += np.sum(q[bcol] ** 2) * (len(q) - 1) / len(q)
    return float(max(1.0, np.sqrt(num / den))) if den > 0 else 2.0

def common_bounds():
    cf = ld('chanfit_placebo.pkl')
    if cf is None or not os.path.exists(f'{OUT}/chanfit.csv'):
        return None, None
    C = pd.read_csv(f'{OUT}/chanfit.csv')
    ch = pd.read_csv(f'{OUT}/channels.csv', index_col=0)
    C['chan'] = C.chseg.str.split('__').str[0]
    C['panel'] = C.chan.map(ch.panel).fillna(False).astype(bool)
    C['borehole'] = C.chseg.str.startswith('PB.')
    dates, starts = cf['dates'], cf['starts']
    # placebo SD per channel only over dates inside the record (>= 1 yr of data on both sides)
    sdm, sdd, PM, PDm = [], [], {}, {}
    for _, r in C.iterrows():
        a = cf['arr'][r.chseg]
        okm = (dates > r.t_first + SR.YR) & (dates < r.t_last - SR.YR)
        okd = (starts > r.t_first + 0.5 * SR.YR) & (starts + 4 * SR.YR < r.t_last)
        pm, pdl = a['m7'] * 100, a['decl'] * 100
        sdm.append(pm[okm].std() if okm.sum() >= 15 else np.nan)
        sdd.append(pdl[okd].std() if okd.sum() >= 8 else np.nan)
        PM[r.chseg] = np.where(okm, pm, np.nan)
        PDm[r.chseg] = np.where(okd, pdl, np.nan)
    # a quantity is only defined if the record covers it: decline needs data before 2020-07, M7 step >= 6 months
    # of pre-M7 data and data after the M7
    C.loc[C.t_first > SR.epoch_s(['2020-07-01'])[0], ['decl4yr', 'decl4yr_se_boot']] = np.nan
    C.loc[(C.t_first > SR.TQ['m7'] - 0.5 * SR.YR) | (C.t_last < SR.TQ['m7'] + 0.25 * SR.YR),
          ['step_m7', 'step_m7_se_boot']] = np.nan
    C['step_m7_sd_plc_valid'] = sdm
    C['decl_sd_plc_valid'] = sdd
    # inflation of bootstrap SE where placebo is not available (short BK records): median ratio on panel
    inf_m = np.nanmedian(C.step_m7_sd_plc_valid / C.step_m7_se_boot)
    inf_d = np.nanmedian(C.decl_sd_plc_valid / C.decl4yr_se_boot)
    C['step_m7_se'] = np.fmax(C.step_m7_sd_plc_valid, C.step_m7_se_boot)
    C['decl_se'] = np.fmax(C.decl_sd_plc_valid, C.decl4yr_se_boot)
    # the parametric decline needs a long pre-2021 baseline (trend, 3 log-healing terms and the ramp trade off)
    C.loc[C.t_first > SR.epoch_s(['2016-07-01'])[0], ['decl4yr', 'decl_se']] = np.nan
    C.to_csv(f'{OUT}/chanfit_calibrated.csv', index=False)
    # station averages (components share the site; average values, SE of mean with full correlation as bound)
    S = C.groupby('sta').agg(step_m7=('step_m7', 'mean'), step_m7_se=('step_m7_se', 'mean'), decl=('decl4yr', 'mean'),
                             decl_se=('decl_se', 'mean'), panel=('panel', 'max'), borehole=('borehole', 'max'),
                             n_ch=('chseg', 'size'), t_first=('t_first', 'min'))
    res = dict(inflation_placebo_over_boot=dict(step=inf_m, decl=inf_d))
    rows = []
    for q, col, sec, P_ in (('step_m7', 'step_m7', 'step_m7_se', PM), ('decline_4yr', 'decl', 'decl_se', PDm)):
        pan = S[S.panel & S[col].notna()]
        # (1) cross-station common mode (median and IVW mean over panel stations); null from the SAME
        #     statistic on the placebo dates (keeps the cross-channel correlation of the noise)
        chs = C[C.panel & C[{'step_m7': 'step_m7', 'decl': 'decl4yr'}[col]].notna()]
        arr = np.array([P_[c] for c in chs.chseg])                     # (nch, ndates)
        sta_of = chs.sta.values
        ust = sorted(set(sta_of))
        sta_null = np.array([np.nanmean(arr[sta_of == u], axis=0) for u in pan.index])     # (nsta, nd), same order
        okst = np.isfinite(sta_null).sum(1) >= 15
        sta_null, pan = sta_null[okst], pan[okst]
        okd = np.all(np.isfinite(sta_null), axis=0)
        cm_null_med = np.nanmedian(sta_null[:, okd], axis=0)
        w = 1 / pan[sec].values ** 2
        cm_null_ivw = np.sum(sta_null[:, okd] * w[:, None], axis=0) / np.sum(w)
        cm_med = np.median(pan[col])
        cm_ivw = np.sum(pan[col] * w) / np.sum(w)
        res[f'{q}_parametric_cm_stations'] = list(pan.index)
        for nm, v, nl in (('common_mode_median_panel', cm_med, cm_null_med), ('common_mode_ivw_panel', cm_ivw, cm_null_ivw)):
            sd = nl.std()
            sd = max(sd, np.sqrt(np.mean(pan[sec] ** 2)) / np.sqrt(len(pan)))
            rows.append(dict(quantity=q, constraint=nm + '_param', value_pct=v - nl.mean(), sd_pct=sd, lo95=v - nl.mean() - 1.96 * sd,
                             hi95=v - nl.mean() + 1.96 * sd, abs95=abs(v - nl.mean()) + 1.96 * sd, n=len(pan),
                             note='parametric receiver model (step + log-healing); max-common attribution'))
        # (2) smallest-signal station(s): HAYF and the panel station with the smallest |value|
        cand = [('HAYF', S.loc['BK.HAYF']) if 'BK.HAYF' in S.index else None]
        ps = pan.iloc[np.argsort(np.abs(pan[col].values))[:1]]
        if 'BK.HAYF' in S.index and not np.isfinite(S.loc['BK.HAYF', col]):
            cand = [None]
        cand.append((ps.index[0], ps.iloc[0]))
        for item in cand:
            if item is None:
                continue
            nm, r = item
            rows.append(dict(quantity=q, constraint=f'smallest_signal_{nm}_param', value_pct=r[col], sd_pct=r[sec],
                             lo95=r[col] - 1.96 * r[sec], hi95=r[col] + 1.96 * r[sec], abs95=abs(r[col]) + 1.96 * r[sec],
                             n=int(r.n_ch), note='assumes no fine-tuned cancellation of receiver and common terms at this station'))
        # (3) deepest boreholes (>= 400 m: B932, B934, B049) - "max-deep" reading
        deep = S.loc[[x for x in ('PB.B932', 'PB.B934', 'PB.B049') if x in S.index]]
        if len(deep):
            v = deep[col].mean()
            sd = np.sqrt(np.mean(deep[sec] ** 2))            # fully correlated (conservative)
            rows.append(dict(quantity=q, constraint='deep_boreholes_412_592m_param', value_pct=v, sd_pct=sd, lo95=v - 1.96 * sd,
                             hi95=v + 1.96 * sd, abs95=abs(v) + 1.96 * sd, n=len(deep),
                             note='if receiver change were zero at >=400 m depth (contradicts dvv_depth2; max-deep reading)'))
        # (4) all-receiver reading
        rows.append(dict(quantity=q, constraint='all_receiver_S0', value_pct=0.0, sd_pct=np.nan, lo95=np.nan, hi95=np.nan,
                         abs95=np.nan, n=np.nan, note='gauge choice Sbar = 0: everything common is near-surface'))
    # (2b) curve-based decline (mean 2025 - mean 2019-07..2021-06 of the non-parametric R_C + Sbar curves from
    #      the all-channel joint fit; stack-bootstrap SD x the panel placebo/boot inflation) - robust for short records
    pa = ld('np_all_lam300.pkl')
    if pa is not None and pa.get('Rboot') is not None:
        nd = pd.to_datetime(pa['nodes'], unit='s')
        m0, m1 = (nd >= '2019-07-01') & (nd < '2021-07-01'), (nd >= '2025-01-01') & (nd < '2026-01-01')
        val = (pa['R'][:, m1].mean(1) - pa['R'][:, m0].mean(1)) * 100
        bsd = (pa['Rboot'][:, :, m1].mean(2) - pa['Rboot'][:, :, m0].mean(2)).std(0) * 100
        q0, q1 = (nd >= pd.to_datetime(SR.TQ['m7'] - 120 * SR.DAY, unit='s')) & (nd < pd.to_datetime(SR.TQ['m7'], unit='s')), \
            (nd >= pd.to_datetime(SR.TQ['m7'], unit='s')) & (nd <= pd.to_datetime(SR.TQ['m7'] + 90 * SR.DAY, unit='s'))
        m7v = (pa['R'][:, q1].mean(1) - pa['R'][:, q0].mean(1)) * 100
        m7b = (pa['Rboot'][:, :, q1].mean(2) - pa['Rboot'][:, :, q0].mean(2)).std(0) * 100
        Dm = pd.DataFrame(dict(chseg=pa['chans'], m7_np=m7v, m7_np_bsd=m7b))
        Dm['sta'] = Dm.chseg.str.split('__').str[0].str.rsplit('.', n=2).str[0]
        tf = Dm.chseg.map(C.set_index('chseg').t_first)
        tl = Dm.chseg.map(C.set_index('chseg').t_last)
        Dm = Dm[(tf < SR.TQ['m7'] - 0.5 * SR.YR) & (tl > SR.TQ['m7'] + 0.25 * SR.YR)]
        fac_m = comp_inflation(Dm, 'm7_np', 'm7_np_bsd')
        res['m7_npcurve_component_inflation'] = fac_m
        Ms = Dm.groupby('sta').agg(m7_np=('m7_np', 'mean'), bsd=('m7_np_bsd', 'mean'))
        Ms['se'] = Ms.bsd * fac_m
        Ms['panel'] = [S.loc[x, 'panel'] if x in S.index else False for x in Ms.index]
        Ms.to_csv(f'{OUT}/station_m7_npcurve.csv')
        res['m7_90d_npcurve_by_station'] = Ms.reset_index().to_dict('records')
        if 'BK.HAYF' in Ms.index:
            r = Ms.loc['BK.HAYF']
            rows.append(dict(quantity='step_m7', constraint='smallest_signal_BK.HAYF_npcurve_90d', value_pct=r.m7_np, sd_pct=r.se,
                             lo95=r.m7_np - 1.96 * r.se, hi95=r.m7_np + 1.96 * r.se, abs95=abs(r.m7_np) + 1.96 * r.se, n=1,
                             note='mean(M7..M7+90 d) - mean(M7-120 d..M7) of the joint-fit curve; bootstrap SD x inflation'))
        pm_ = Ms[Ms.panel]
        if len(pm_):
            v = np.median(pm_.m7_np)
            sd = 1.2533 * np.sqrt(np.mean(pm_.se ** 2))
            rows.append(dict(quantity='step_m7', constraint='common_mode_median_panel_npcurve_90d', value_pct=v, sd_pct=sd,
                             lo95=v - 1.96 * sd, hi95=v + 1.96 * sd, abs95=abs(v) + 1.96 * sd, n=len(pm_),
                             note='curve-based 90-d change; errors treated as fully correlated across stations'))
            ps_ = pm_.iloc[int(np.argmin(np.abs(pm_.m7_np.values)))]
            rows.append(dict(quantity='step_m7', constraint=f'smallest_signal_panel_{ps_.name}_npcurve_90d', value_pct=ps_.m7_np,
                             sd_pct=ps_.se, lo95=ps_.m7_np - 1.96 * ps_.se, hi95=ps_.m7_np + 1.96 * ps_.se,
                             abs95=abs(ps_.m7_np) + 1.96 * ps_.se, n=1, note='smallest |90-d change| among panel stations'))
        Dn = pd.DataFrame(dict(chseg=pa['chans'], decl_np=val, decl_np_bsd=bsd))
        Dn['sta'] = Dn.chseg.str.split('__').str[0].str.rsplit('.', n=2).str[0]
        Dn = Dn[(Dn.chseg.map(C.set_index('chseg').t_first) < SR.epoch_s(['2019-12-31'])[0])
                & (Dn.chseg.map(C.set_index('chseg').t_last) > SR.epoch_s(['2025-06-01'])[0])]
        Dn = Dn[~Dn.chseg.str.contains('__seg')]                    # no decline across an instrument break (free offset)
        fac_d = comp_inflation(Dn, 'decl_np', 'decl_np_bsd')
        res['decline_npcurve_component_inflation'] = fac_d
        Ds = Dn.groupby('sta').agg(decl_np=('decl_np', 'mean'), bsd=('decl_np_bsd', 'mean'))
        Ds['se'] = Ds.bsd * fac_d
        Ds.to_csv(f'{OUT}/station_decline_npcurve.csv')
        res['decline_npcurve_by_station'] = Ds.reset_index().to_dict('records')
        for nm in ('BK.HAYF',):
            if nm in Ds.index:
                r = Ds.loc[nm]
                rows.append(dict(quantity='decline_4yr', constraint=f'smallest_signal_{nm}_npcurve', value_pct=r.decl_np, sd_pct=r.se,
                                 lo95=r.decl_np - 1.96 * r.se, hi95=r.decl_np + 1.96 * r.se, abs95=abs(r.decl_np) + 1.96 * r.se,
                                 n=1, note='2025 minus 2019H2-2021H1 of the joint-fit curve; bootstrap SD x panel inflation'))
        pan_ = Ds.loc[[x for x in Ds.index if x in S.index and S.loc[x, 'panel']]]
        pan_ = pan_[pan_.index != 'NC.KMPB'] if 'NC.KMPB' in pan_.index and not np.isfinite(pan_.loc['NC.KMPB', 'decl_np']) else pan_
        if len(pan_):
            ps_ = pan_.iloc[int(np.argmin(np.abs(pan_.decl_np.values)))]
            rows.append(dict(quantity='decline_4yr', constraint=f'smallest_signal_panel_{ps_.name}_npcurve', value_pct=ps_.decl_np,
                             sd_pct=ps_.se, lo95=ps_.decl_np - 1.96 * ps_.se, hi95=ps_.decl_np + 1.96 * ps_.se,
                             abs95=abs(ps_.decl_np) + 1.96 * ps_.se, n=1, note='smallest |decline| among panel stations'))
            v = np.median(pan_.decl_np)
            sd = 1.2533 * np.sqrt(np.mean(pan_.se ** 2)) / np.sqrt(len(pan_)) * np.sqrt(len(pan_))   # fully correlated (conservative)
            rows.append(dict(quantity='decline_4yr', constraint='common_mode_median_panel_npcurve', value_pct=v, sd_pct=sd,
                             lo95=v - 1.96 * sd, hi95=v + 1.96 * sd, abs95=abs(v) + 1.96 * sd, n=len(pan_),
                             note='curve-based; errors treated as fully correlated across stations'))
    # (2c) 'damage-only' reading: if near-surface changes can only LOWER velocity (damage, decline), a common
    #      term must satisfy w_S*Sbar >= obs_C for every station C -> lower bound = max_C (obs_C - 1.96 se_C)
    for q, tab, vc in (('step_m7', 'station_m7_npcurve.csv', 'm7_np'), ('decline_4yr', 'station_decline_npcurve.csv', 'decl_np')):
        f = f'{OUT}/{tab}'
        if os.path.exists(f):
            Z = pd.read_csv(f, index_col=0)
            lb = Z[vc] - 1.96 * Z.se
            k = lb.idxmax()
            rows.append(dict(quantity=q, constraint=f'damage_only_lower_bound({k})', value_pct=Z.loc[k, vc], sd_pct=Z.loc[k, 'se'],
                             lo95=lb.max(), hi95=np.nan, abs95=np.nan, n=len(Z),
                             note='one-sided: a deep velocity DROP larger than -lo95 is excluded if receivers only decline'))
    # (5) PGV intercept for the M7 step (receiver damage -> 0 at low shaking)
    try:
        pg = pd.read_csv('data/sci/dvv_depth/pgv_by_station.csv')
        pg = pg[pg.quake == 'm7'].set_index('sta')['pgv_gm_cms']
        Sx = S.join(pg.rename('pgv'), how='inner').dropna(subset=['pgv', 'step_m7'])
        x = np.log10(Sx.pgv.values)
        w = 1 / Sx.step_m7_se.values ** 2
        A = np.c_[np.ones(len(x)), x]
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        b = cov @ (A.T @ (w * Sx.step_m7.values))
        r_ = Sx.step_m7.values - A @ b
        chi = np.sum(w * r_ ** 2) / max(len(x) - 2, 1)
        cov *= max(chi, 1)
        rows.append(dict(quantity='step_m7', constraint='PGV_intercept_at_1cm_s', value_pct=b[0], sd_pct=np.sqrt(cov[0, 0]),
                         lo95=b[0] - 1.96 * np.sqrt(cov[0, 0]), hi95=b[0] + 1.96 * np.sqrt(cov[0, 0]),
                         abs95=abs(b[0]) + 1.96 * np.sqrt(cov[0, 0]), n=len(x),
                         note=f'slope {b[1]:+.3f}+/-{np.sqrt(cov[1, 1]):.3f} %/decade, chi2r {chi:.2f}; PGV range '
                              f'{Sx.pgv.min():.1f}-{Sx.pgv.max():.1f} cm/s'))
        res['pgv_fit'] = dict(intercept=b[0], slope=b[1], cov=cov, chi2r=chi, stations=list(Sx.index))
    except Exception as e:                                 # noqa: BLE001
        res['pgv_fit_error'] = repr(e)
    # (6) direct S (path-averaged, receiver-insensitive; from sci_dvv_depth direct_wave_check)
    dw = json.load(open('data/sci/dvv_depth/summary.json'))['direct_wave_check']
    for k, nm in (('all_S', 'direct_S_all'), ('S_without_KMPB', 'direct_S_noKMPB'), ('S_PB_only', 'direct_S_PB')):
        v, sd = dw[k]['dvv_pct'], dw[k]['se_pct']
        rows.append(dict(quantity='step_m7', constraint=nm, value_pct=v, sd_pct=sd, lo95=v - 1.96 * sd, hi95=v + 1.96 * sd,
                         abs95=abs(v) + 1.96 * sd, n=dw[k]['n'],
                         note='path-averaged direct-S dv/v at the M7 (whole 28-km path; ~0.4-0.5 of travel time below 15 km)'))
    B = pd.DataFrame(rows)
    B.to_csv(f'{OUT}/common_bounds.csv', index=False)
    S.to_csv(f'{OUT}/station_receiver_plus_common.csv')
    return B, (C, S, res)


# ============================================================================== coherence -> w_S
def coherence_ws():
    f = f'{OUT}/coherence.csv'
    if not os.path.exists(f):
        return None
    D = pd.read_csv(f)
    X = family_covariates(sorted(set(D.famA) | set(D.famB)))
    D['sep_km'] = [np.hypot((X.loc[a, 'lon'] - X.loc[b, 'lon']) * 111.2 * np.cos(np.radians(40.06)),
                            (X.loc[a, 'lat'] - X.loc[b, 'lat']) * 111.2) for a, b in zip(D.famA, D.famB)]
    D['sep_km'] = np.hypot(D.sep_km, [X.loc[a, 'depth_km'] - X.loc[b, 'depth_km'] for a, b in zip(D.famA, D.famB)])
    # chance-corrected, reliability-normalised coherence
    D['g'] = ((D.cc - D.cc_null) / (1 - D.cc_null)).clip(-0.5, 1.5) / np.sqrt(D.relA * D.relB).clip(0.05, None)
    D['ok'] = (D.relA > 0.3) & (D.relB > 0.3)
    D.to_csv(f, index=False)
    G = D[D.ok].groupby('tau').agg(g_med=('g', 'median'), cc_med=('cc', 'median'), null_med=('cc_null', 'median'),
                                   n=('g', 'size'), g_q25=('g', lambda x: np.percentile(x, 25)), g_q75=('g', lambda x: np.percentile(x, 75)))
    direct = G.loc[G.index < 2.5, 'g_med'].mean()
    coda = D[D.ok & (D.tau >= 4) & (D.tau <= 12)]
    gc = coda.g.median()
    near = coda[coda.sep_km < 3].g.median()
    far = coda[coda.sep_km >= 3].g.median()
    out = dict(gamma_direct_S=direct, gamma_coda_4_12s=gc, gamma_coda_sep_lt3km=near, gamma_coda_sep_ge3km=far,
               wS_est_1_minus_gamma=1 - gc, wS_est_relative_to_direct=1 - gc / direct if direct > 0 else np.nan,
               n_pairs=int(D[D.ok][['chan', 'famA', 'famB']].drop_duplicates().shape[0]),
               by_tau=G.reset_index().to_dict('records'),
               interpretation='inter-family coda coherence is near chance at all lapse times for separations >= 3 km '
                              '(and only ~0.15 below 2 km), and the direct S itself is only ~0.2 coherent: scattered waves '
                              'from scatterers beyond ~lambda*L/separation (~1 km) of the receiver also decorrelate, so '
                              '1-gamma is NOT the source-side weight; it only gives a loose upper limit w_S <~0.5-0.9. '
                              'The kernel values (0.05-0.12 within 10 km of the source) are used for the conversion.')
    return out


# ============================================================================== synthetics
def synth_summary(T):
    sy = ld('synth_balanced.pkl')
    if sy is None:
        return None
    fams = sy['fams']
    out = {}
    s0 = sy['cases']['S0'] * 100
    leak_mean = s0.mean(0)                                 # (NF, NP)
    for p in ('step_m7', 'tr60_m7', 'decl4yr', 'step_fer', 'step_pet'):
        out[f'S0_{p}'] = dict(rms_mean_leak_pct=np.sqrt(np.mean(leak_mean[:, iP[p]] ** 2)),
                              max_abs_mean_leak_pct=np.max(np.abs(leak_mean[:, iP[p]])),
                              rms_scatter_pct=np.sqrt(np.mean(s0[:, :, iP[p]].var(0))),
                              note='receiver-only truth (observed R_C) + station-epoch-correlated noise')
    if T is not None:
        out['S0_scatter_over_placebo_sd_step_m7'] = np.median(s0[:, :, iP['step_m7']].std(0) / T.step_m7_sd_plc.reindex(fams).values)
    s1 = sy['cases']['S1'] * 100
    out['S1_common_step_-0.03pct_leak_into_dS'] = dict(
        rms_pct=np.sqrt(np.mean((s1[:, :, iP['step_m7']] - s0[:, :, iP['step_m7']]).mean(0) ** 2)),
        note='difference S1-S0 with the same noise seeds would be exactly 0 by linearity; reported with independent noise')
    s2 = sy['cases']['S2'] * 100
    inj = np.array([sy['fam_inj'].get(F, 0.0) for F in fams]) * 100
    inj_g = inj - inj.mean()                                # gauge-projected truth
    rec = s2[:, :, iP['step_m7']].mean(0) - leak_mean[:, iP['step_m7']]
    msk = inj != 0
    slope = np.sum(rec[msk] * inj_g[msk]) / np.sum(inj_g[msk] ** 2)
    out['S2_family_steps'] = dict(injected_pct=dict(zip([int(F) for F in fams], inj)), recovered_pct=dict(zip([int(F) for F in fams], rec)),
                                  recovery_slope=slope, leak_to_noninjected_rms_pct=np.sqrt(np.mean((rec[~msk] - inj_g[~msk]) ** 2)),
                                  per_rep_sd_pct=dict(zip([int(F) for F in fams], s2[:, :, iP['step_m7']].std(0))))
    out['S2_detail'] = dict(fams=fams, inj=inj, inj_gauge=inj_g, rec=rec, s0_mean=leak_mean[:, iP['step_m7']])
    return out


# ============================================================================== stress sensitivity
LIT = [  # order-of-magnitude literature values of d ln v / d sigma (per MPa); approximate, from memory of the papers
    ('Lab, crystalline rock, high effective pressure (>~100 MPa, cracks closed)', 1e-4, 1e-3),
    ('Lab, crystalline rock, low effective pressure (<~20 MPa, open cracks)', 1e-3, 3e-2),
    ('Field, SAFOD ~1 km depth, barometric (Silver et al. 2007)', 1e-1, 3e-1),
    ('Field, tidal / near-surface coda and noise dv/v (e.g. Takano et al.; Hillers et al.)', 1e-2, 1e0),
    ('Fable guidance for the LFE zone (local dv/v 2e-5..2e-4 per ~2 kPa)', 1e-2, 1e-1)]


def stress_bounds(B, T, coh, fb=None):
    """apparent coda bound X (fraction) -> local dv/v = X / w_S -> per kPa."""
    out = {}
    dr = pd.read_csv('data/sci/dose_response/m7_per_family_stress_vs_gap.csv')
    dr = dr[dr.quake.str.contains('M7')]
    stress = dict(dp_kPa=dr.dp.median(), dtau_kPa=dr.dtau.median(), dsn_kPa=dr.dsn.median(), dsn_eff_kPa=dr.dsn_eff.median(),
                  dp_range=[dr.dp.min(), dr.dp.max()])
    out['stress_M7_median'] = stress
    wsets = {'kernel_within_10km_of_source(0.05-0.12)': (0.05, 0.12), 'kernel_below_15km(0.35-0.8)': (0.35, 0.8)}
    if coh is not None and np.isfinite(coh.get('wS_est_1_minus_gamma', np.nan)):
        w = float(np.clip(coh['wS_est_1_minus_gamma'], 0.01, 1))
        wsets['empirical_upper_limit_1_minus_gamma(not diagnostic)'] = (w, w)
    wsets['Fable_range(0.01-0.1)'] = (0.01, 0.1)
    rows = []
    cands = []
    if B is not None:
        for q in ('step_m7', 'decline_4yr'):
            sub = B[(B.quantity == q) & B.abs95.notna()]
            for _, r in sub.iterrows():
                cands.append((q, 'common:' + r.constraint, r.abs95))
    if T is not None:
        # family-specific: 95% upper bound on the intrinsic family scatter (ML), and |dS_F|+1.96 se over the
        # better-measured half of the families (median and max)
        for q, c in (('step_m7', 'step_m7'), ('total_m7', 'total_m7'), ('tr60_m7', 'tr60_m7'), ('decline_4yr', 'decl4yr')):
            good = T[c + '_se'] <= T[c + '_se'].median()
            v = np.abs(T.loc[good, c + '_wc']) + 1.96 * T.loc[good, c + '_se']
            cands.append((q, 'family_wellmeasured_median_abs95', float(np.median(v))))
            cands.append((q, 'family_wellmeasured_max_abs95', float(np.max(v))))
            if fb is not None and c in fb:
                cands.append((q, 'family_intrinsic_tau95', float(fb[c]['intrinsic_tau_95_upper_pct'])))
    wfam = {'kernel_within_~2km_of_patch(0.01-0.024; 10-km sphere value x r/10km)': (0.01, 0.024),
            'Fable_range(0.01-0.1)': (0.01, 0.1)}
    for q, src, X in cands:
        for wn, (w0, w1) in (wfam if src.startswith('family') else wsets).items():
            loc_lo, loc_hi = X / w1, X / w0                  # local dv/v bound (%)
            dsig = abs(stress['dp_kPa'])
            rows.append(dict(quantity=q, source=src, apparent_abs95_pct=X, wS=wn, local_dvv_bound_pct_lo=loc_lo,
                             local_dvv_bound_pct_hi=loc_hi,
                             sens_bound_per_kPa_lo=loc_lo / 100 / dsig if q not in ('decline_4yr', 'tr60_m7') else np.nan,
                             sens_bound_per_kPa_hi=loc_hi / 100 / dsig if q not in ('decline_4yr', 'tr60_m7') else np.nan,
                             sens_bound_per_MPa_lo=loc_lo / 100 / dsig * 1e3 if q not in ('decline_4yr', 'tr60_m7') else np.nan,
                             sens_bound_per_MPa_hi=loc_hi / 100 / dsig * 1e3 if q not in ('decline_4yr', 'tr60_m7') else np.nan))
    Bd = pd.DataFrame(rows)
    Bd.to_csv(f'{OUT}/stress_sensitivity_bounds.csv', index=False)
    out['literature_dlnv_dsigma_per_MPa'] = [dict(source=a, lo=b, hi=c) for a, b, c in LIT]
    # what the M7 should produce: local = sens * 2.3 kPa; apparent = local * w_S
    out['expected_apparent_M7_pct'] = dict(lo=1e-2 * 2.3e-3 * 0.01 * 100, hi=1e-1 * 2.3e-3 * 0.12 * 100,
                                           note='sens 0.01-0.1 /MPa x 2.3 kPa x w_S 0.01-0.12')
    return Bd, out


# ============================================================================== figures
def fig_curves(T):
    pk = ld(f'np_balanced_lam{int(SR.LAM)}.pkl')
    if pk is None:
        return
    nodes = pd.to_datetime(pk['nodes'], unit='s')
    fams = pk['fams']
    s = pk['s'] * 100
    sb = pk['sboot'] * 100 if pk['sboot'] is not None and len(pk['sboot']) else None
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    seg = np.concatenate([[k] * len(x) for k, x in enumerate(pk['segs'])])
    cmap = plt.get_cmap('viridis')
    X = family_covariates(fams)
    xs = X.along_strike_km.values
    norm = (xs - np.nanmin(xs)) / (np.nanmax(xs) - np.nanmin(xs))
    for i, F in enumerate(fams):
        for g in np.unique(seg):
            m = seg == g
            ax[0].plot(nodes[m], s[i, m], color=cmap(norm[i]), lw=0.8, alpha=0.8)
    if sb is not None:
        rms_b = np.median(sb.std(0), axis=0)
        for g in np.unique(seg):
            m = seg == g
            ax[0].fill_between(nodes[m], -2 * rms_b[m], 2 * rms_b[m], color='0.85', zorder=0)
    ax[0].set_ylabel('dS_F = S_F - mean S (%)')
    ax[0].set_title('Identifiable family deviations (gauge sum_F S_F = 0; colour = along-strike position); '
                    'grey = +/-2x median bootstrap SD')
    R = pk['R'] * 100
    chans = pk['chans']
    rw = pd.read_parquet(f'{OUT}/rows_v2.parquet', columns=['chseg', 'tb']).groupby('chseg').tb.agg(['min', 'max'])
    span = {c: (pd.to_datetime(rw.loc[c, 'min'] - 31 * SR.DAY, unit='s'), pd.to_datetime(rw.loc[c, 'max'] + 31 * SR.DAY, unit='s'))
            for c in rw.index}
    R = np.array([np.where((nodes >= span[c][0]) & (nodes <= span[c][1]), r, np.nan) for c, r in zip(chans, R)])
    for c, r in zip(chans, R):
        col = 'C0' if c.startswith('PB.') else 'C3'
        for g in np.unique(seg):
            m = seg == g
            ax[1].plot(nodes[m], r[m] - np.nanmean(r[(nodes > '2012') & (nodes < '2020')]), color=col, lw=0.6, alpha=0.6)
    ax[1].set_ylabel('R_C + Sbar (%)')
    ax[1].set_title('Receiver curves (carry the unidentified common term); blue PB boreholes, red NC (balanced panel)')
    # candidate common terms
    sta = np.array([c.split('__')[0].rsplit('.', 2)[0] for c in chans])
    Rz = np.array([r - np.nanmean(r[(nodes > '2012') & (nodes < '2020')]) if np.isfinite(r[(nodes > '2012') & (nodes < '2020')]).any()
                   else r - np.nanmean(r[(nodes > '2024-01') & (nodes < '2024-11')]) for r in R])
    Rz[np.array(['__seg1' in c for c in chans])] = np.nan     # free offset after a break: level unknown
    sta_mean = np.array([np.nanmean(Rz[sta == u], axis=0) for u in sorted(set(sta))])
    cm = np.nanmedian(sta_mean, axis=0)
    deep = np.nanmean(np.array([np.nanmean(Rz[sta == u], axis=0) for u in ('PB.B932', 'PB.B934', 'PB.B049') if u in sta]), axis=0)
    for g in np.unique(seg):
        m = seg == g
        ax[2].plot(nodes[m], cm[m], 'k', lw=1.5, label='cross-station common mode (median, max-common reading)' if g == 0 else None)
        ax[2].plot(nodes[m], deep[m], 'C1', lw=1.2, label='deep boreholes >=400 m (max-deep reading)' if g == 0 else None)
        ax[2].plot(nodes[m], 0 * cm[m], 'C2', lw=1.2, label='all-receiver reading (Sbar = 0)' if g == 0 else None)
    pa = ld('np_all_lam300.pkl')
    if pa is not None:
        na = pd.to_datetime(pa['nodes'], unit='s')
        hay = [r for c, r in zip(pa['chans'], pa['R']) if 'HAYF' in c]
        if hay:
            h = np.mean(hay, axis=0) * 100
            ref = (na > '2020-03') & (na < '2021-06')
            h = h - h[ref].mean() + cm[(nodes > '2020-03') & (nodes < '2021-06')].mean()
            for g in np.unique(seg):
                m = (seg == g) & (na > '2019-12')
                ax[2].plot(na[m], h[m], 'C4', lw=1.2, label='HAYF (smallest-signal station)' if g == 0 else None)
    ax[2].set_ylabel('w_S * Sbar candidates (%)')
    ax[2].set_title('Common term is NOT identified: candidate readings under different assumptions')
    ax[2].legend(fontsize=7, loc='lower left')
    for a in ax:
        for q in SR.TQ.values():
            a.axvline(pd.to_datetime(q, unit='s'), color='k', ls='--', lw=0.6)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_receiver_curves.png', dpi=110)
    plt.close(fig)


def fig_family(T, R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fams = T.index.values
    o = np.argsort(T.along_strike_km.values)
    for k, (c, lab) in enumerate((('step_m7', 'M7 step'), ('tr60_m7', '60-d post-M7 transient'), ('decl4yr', 'post-2021 4-yr decline'))):
        a = ax[0, k]
        y, e = T[c + '_wc'].values[o], T[c + '_se'].values[o]
        a.errorbar(np.arange(len(fams)), y, 1.96 * e, fmt='o', ms=4, capsize=2)
        a.axhline(0, color='k', lw=0.6)
        a.set_xticks(np.arange(len(fams)))
        a.set_xticklabels(fams[o], fontsize=7)
        a.set_ylim(-0.5, 0.5) if c == 'tr60_m7' else a.set_ylim(-0.3, 0.3)
        a.set_xlabel('family (ordered along strike, ESE -> WNW)')
        a.set_ylabel('dS_F - weighted mean (%)')
        a.set_title(lab + ': family deviation +/-95% (honest SE)', fontsize=9)
    for k, (cv, c) in enumerate((('tenth_d', 'total_m7'), ('dp', 'tr60_m7'), ('along_strike_km', 'tr60_m7'))):
        a = ax[1, k]
        good = T[c + '_se'] < 0.15
        a.errorbar(T[cv][good], T[c + '_wc'][good], 1.96 * T[c + '_se'][good], fmt='o', ms=4, capsize=2)
        for F, x, y in zip(fams[good.values], T[cv][good], T[c + '_wc'][good]):
            a.annotate(str(F), (x, y), fontsize=6)
        r = R[(R.quantity == c) & (R.covariate == cv)]
        if len(r):
            r = r.iloc[0]
            a.set_title(f'{c} vs {cv}: slope {r.slope:+.4f} (z_boot {r.z_boot:+.1f}, p_perm {r.p_perm:.2f})', fontsize=8)
        a.axhline(0, color='k', lw=0.6)
        a.set_xlabel(cv)
        a.set_ylabel(c + ' deviation (%) [SE<0.15% shown]')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_receiver_family.png', dpi=110)
    plt.close(fig)


def fig_nulls(out):
    fb = out.get('family_balanced')
    pl = ld('placebo_balanced.pkl')
    if fb is None or pl is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    names = ['step_m7', 'tr60_m7', 'total_m7', 'decl4yr', 'step_fer', 'step_pet']
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [fb[n]['Q'] for n in names], 0.4, label='observed Q (honest SE)')
    ax[0].bar(x + 0.2, [np.median(fb[n]['Q_perm']) for n in names], 0.4, label='family-label permutations (median)')
    ax[0].axhline(fb['step_m7']['dof'], color='k', ls='--', lw=0.8, label='dof')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=30, fontsize=8)
    ax[0].set_ylabel('Cochran Q of family deviations')
    ax[0].legend(fontsize=7)
    for k, (lab, key) in enumerate((('bootstrap', 'median_se_boot'), ('placebo dates', 'median_sd_placebo'), ('label permutation', 'median_sd_perm'))):
        ax[1].bar(x + (k - 1) * 0.27, [fb[n][key] for n in names], 0.27, label=lab)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=30, fontsize=8)
    ax[1].set_ylabel('median per-family SE (%)')
    ax[1].legend(fontsize=7)
    ax[1].set_title('error calibration: placebo dates under-state noise at the M7 (fewer post-event data)', fontsize=8)
    ax[2].bar(x, [fb[n]['split_ratio_rms_diff_over_2se_boot_wellmeasured'] for n in names], color='C2')
    ax[2].axhline(1, color='k', ls='--')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(names, rotation=30, fontsize=8)
    ax[2].set_ylabel('RMS(half A - half B) / (2 x bootstrap SE)')
    ax[2].set_title('pseudo-family split test (well-measured half of families): ~1 => bootstrap SE calibrated', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_receiver_nulls.png', dpi=110)
    plt.close(fig)


def fig_common(B, extra):
    if B is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    pg = pd.read_csv('data/sci/dvv_depth/pgv_by_station.csv')
    pg = pg[pg.quake == 'm7'].set_index('sta')['pgv_gm_cms']
    M = pd.read_csv(f'{OUT}/station_m7_npcurve.csv', index_col=0).join(pg.rename('pgv'), how='left')
    for pan, col in ((True, 'C0'), (False, 'C3')):
        q = M[M.panel == pan]
        ax[0].errorbar(q.pgv, q.m7_np, 1.96 * q.se, fmt='o', color=col, capsize=2,
                       label='balanced panel' if pan else 'other (2019+ BK etc.)')
        for n_, r in q.iterrows():
            ax[0].annotate(n_.split('.')[1], (r.pgv, r.m7_np), fontsize=7, xytext=(3, 2), textcoords='offset points')
    ax[0].set_xscale('log')
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('M7 PGV (cm/s)')
    ax[0].set_ylabel('90-d M7 change of R_C + w_S Sbar (%)')
    ax[0].set_title('receiver curves (joint fit, all channels) at the M7', fontsize=9)
    ax[0].legend(fontsize=7)
    D = pd.read_csv(f'{OUT}/station_decline_npcurve.csv', index_col=0)
    ax[1].errorbar(np.arange(len(D)), D.decl_np, 1.96 * D.se, fmt='o', capsize=2)
    ax[1].set_xticks(np.arange(len(D)))
    ax[1].set_xticklabels([x.split('.')[1] for x in D.index], rotation=90, fontsize=8)
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set_ylabel('2025 minus 2019H2-2021H1, R_C + w_S Sbar (%)')
    ax[1].set_title('post-2021 decline per station (records spanning 2019-2025, no break)', fontsize=9)
    sub = B[~B.constraint.str.contains('_param|all_receiver')].reset_index(drop=True)
    for k, q in enumerate(('step_m7', 'decline_4yr')):
        z = sub[sub.quantity == q]
        y = z.index.values
        lo = np.where(np.isfinite(z.lo95), z.value_pct - z.lo95, 0)
        hi = np.where(np.isfinite(z.hi95), z.hi95 - z.value_pct, 0)
        ax[2].errorbar(z.value_pct, y, xerr=[lo, hi], fmt='o' if k == 0 else 's', color=f'C{k}', capsize=2, label=q)
        for yy, (_, r) in zip(y, z.iterrows()):
            ax[2].annotate(r.constraint, (0.105, yy), fontsize=6.5, va='center')
    ax[2].axvline(0, color='k', lw=0.5)
    ax[2].set_xlim(-0.16, 0.30)
    ax[2].set_yticks([])
    ax[2].set_xlabel('apparent common term w_S * Sbar (%, 95% interval)')
    ax[2].legend(fontsize=7, loc='upper left')
    ax[2].set_title('common deep term: bounds under stated assumptions (not identified)', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_receiver_common.png', dpi=110)
    plt.close(fig)


def fig_synth(sy):
    if sy is None:
        return
    d = sy['S2_detail']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(np.arange(len(d['fams'])), d['s0_mean'], color='0.6')
    ax[0].set_xticks(np.arange(len(d['fams'])))
    ax[0].set_xticklabels(d['fams'], fontsize=7)
    ax[0].set_ylabel('recovered M7 dS_F (%)')
    ax[0].set_title('Leakage floor: observed R_C injected only (mean over noise realisations)', fontsize=9)
    ax[1].plot(d['inj_gauge'], d['rec'], 'o')
    lim = np.max(np.abs(d['inj_gauge'])) * 1.3
    ax[1].plot([-lim, lim], [-lim, lim], 'k--', lw=0.7)
    ax[1].set_xlabel('injected family M7 step (gauge-projected, %)')
    ax[1].set_ylabel('recovered (minus leakage floor, %)')
    ax[1].set_title(f"recovery slope {sy['S2_family_steps']['recovery_slope']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_receiver_synth.png', dpi=110)
    plt.close(fig)


def fig_coh(coh):
    f = f'{OUT}/coherence.csv'
    if coh is None or not os.path.exists(f):
        return
    D = pd.read_csv(f)
    D = D[D.ok]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for lab, m, c in (('separation < 3 km', D.sep_km < 3, 'C0'), ('>= 3 km', D.sep_km >= 3, 'C3')):
        g = D[m].groupby('tau').g.median()
        ax[0].plot(g.index, g.values, 'o-', color=c, label=lab)
    ax[0].axhline(1, color='k', lw=0.5)
    ax[0].set_xlabel('lapse time after S (s)')
    ax[0].set_ylabel('normalised inter-family coherence gamma')
    ax[0].legend(fontsize=8)
    ax[0].set_title('inter-family coda coherence (chance-corrected, reliability-normalised): near zero for all lapse\n'
                    'times, even direct S ~0.3 -> decorrelation is not specific to source-side scattering (w_S not resolved)', fontsize=8)
    cod = D[(D.tau >= 4) & (D.tau <= 12)]
    ax[1].scatter(cod.sep_km, cod.g, s=4, alpha=0.3)
    ax[1].set_xlabel('family separation (km)')
    ax[1].set_ylabel('gamma (coda 4-12 s)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_source_receiver_coherence.png', dpi=110)
    plt.close(fig)


# ============================================================================== main
def lam_sensitivity():
    out = {}
    base = ld('np_balanced_lam300.pkl')
    if base is None:
        return out
    for lam in (100, 1000):
        o = ld(f'np_balanced_lam{lam}.pkl')
        if o is None:
            continue
        out[f'rms_dS_diff_lam{lam}_vs_300_pct'] = float(np.sqrt(np.mean((o['s'] - base['s']) ** 2)) * 100)
    out['rms_dS_lam300_pct'] = float(np.sqrt(np.mean(base['s'] ** 2)) * 100)
    if base.get('sboot') is not None and len(base['sboot']):
        out['rms_boot_sd_dS_pct'] = float(np.sqrt(np.mean(base['sboot'].std(0) ** 2)) * 100)
    # node-curve M7 step of dS_F (break at the M7): value at first post-M7 node minus last pre-M7 node
    segs = base['segs']
    i_post = sum(len(s) for s in segs[:3])
    st = (base['s'][:, i_post] - base['s'][:, i_post - 1]) * 100
    out['np_curve_m7_step_dS'] = dict(zip([int(F) for F in base['fams']], st))
    if base.get('sboot') is not None and len(base['sboot']):
        out['np_curve_m7_step_dS_bootsd'] = dict(zip([int(F) for F in base['fams']],
                                                      (base['sboot'][:, :, i_post] - base['sboot'][:, :, i_post - 1]).std(0) * 100))
    return out


def interaction():
    f = f'{OUT}/interaction_residual_balanced_lam300.csv'
    if not os.path.exists(f):
        return None
    D = pd.read_csv(f)
    return dict(n_cells=len(D), rms_res_all_pct=float(np.sqrt(np.nanmean(D.res_all ** 2))),
                rms_post_minus_pre_pct=float(np.sqrt(np.nanmean(D.res_post_minus_pre ** 2))),
                top5_post_minus_pre=D.reindex(D.res_post_minus_pre.abs().sort_values(ascending=False).index).head(5)[
                    ['sta', 'chseg', 'fam', 'n', 'res_post_minus_pre']].to_dict('records'))


def main():
    out = {}
    T = R = None
    ft = family_tests('balanced')
    if ft is not None:
        T, R, out['family_balanced'] = ft
    fa = family_tests('all') if os.path.exists(f'{OUT}/placebo_all.pkl') else None
    if fa is not None:
        out['family_all'] = fa[2]
    else:
        pa = ld('param_all.pkl')
        if pa is not None and T is not None:
            sa = pd.DataFrame(pa['s'] * 100, index=pa['fams'], columns=PN)
            sb = pd.DataFrame(pa['boot'].std(0) * 100, index=pa['fams'], columns=PN)
            j = T[['step_m7', 'tr60_m7', 'decl4yr']].join(sa[['step_m7', 'tr60_m7', 'decl4yr']], rsuffix='_all', how='inner')
            out['family_all_vs_balanced'] = dict(
                corr=dict((c, float(j[c].corr(j[c + '_all']))) for c in ('step_m7', 'tr60_m7', 'decl4yr')),
                rms_all=dict((c, float(np.sqrt(np.mean((sa[c] - sa[c].mean()) ** 2)))) for c in ('step_m7', 'tr60_m7', 'decl4yr')),
                median_boot_se_all=dict((c, float(sb[c].median())) for c in ('step_m7', 'tr60_m7', 'decl4yr')),
                note='all channels incl. 2019+ BK stations (coverage-unbalanced sensitivity test); bootstrap SE only')
            pd.concat([sa, sb.add_suffix('_se_boot')], axis=1).rename_axis('family').to_csv(f'{OUT}/param_dS_all_table.csv')
            from scipy import stats
            qa = {}
            for c in ('step_m7', 'tr60_m7', 'decl4yr', 'step_fer', 'step_pet'):
                v, e = sa[c].values, sb[c].values
                tot = (sa[c] if c != 'tr60_m7' else sa[c]).values
                w = 1 / e ** 2
                dev = v - np.sum(w * v) / np.sum(w)
                Qc = float(np.sum(w * dev ** 2))
                z = dev / np.sqrt(np.clip(e ** 2 - 1 / np.sum(w), 1e-12, None))
                tau, tau95 = ml_tau(v, e)
                qa[c] = dict(Q=Qc, dof=len(v) - 1, p_chi2=float(stats.chi2.sf(Qc, len(v) - 1)), max_abs_z=float(np.max(np.abs(z))),
                             fam_max_abs_z=int(sa.index[int(np.argmax(np.abs(z)))]), intrinsic_tau95=tau95,
                             p_bonferroni_max=float(min(1, 2 * stats.norm.sf(np.max(np.abs(z))) * len(v))))
            out['family_all_vs_balanced']['bootstrap_Q_tests_all'] = qa
    B, extra = common_bounds()
    if B is not None:
        out['common_bounds'] = B.to_dict('records')
        out['common_misc'] = extra[2]
    coh = coherence_ws()
    out['coherence_wS'] = coh
    sy = synth_summary(T)
    out['synthetics'] = {k: v for k, v in (sy or {}).items() if k != 'S2_detail'}
    Bd, sb_ = stress_bounds(B, T, coh, out.get('family_balanced'))
    out['stress'] = sb_
    out['stress_bounds_table'] = Bd.to_dict('records')
    out['lambda_sensitivity'] = lam_sensitivity()
    out['interaction'] = interaction()
    out['headline'] = dict(
        identifiability='common source term in the null space; gauge sum_F S_F(t)=0 (hard); family deviations identified; '
                        'common term only bounded (cross-station common mode, smallest-signal station, damage-only, direct S)',
        error_calibration='bootstrap over stacks validated by pseudo-family splits (ratio ~0.9-1.1); placebo dates under-state '
                          'noise at the M7 by ~2x (fewer post-event data); honest SE = max(bootstrap, placebo, permutation)',
        family_deviations_null={k: dict(Q=out['family_balanced'][k]['Q'], dof=out['family_balanced'][k]['dof'],
                                        p=out['family_balanced'][k]['p_chi2'], tau95_pct=out['family_balanced'][k]['intrinsic_tau_95_upper_pct'],
                                        max_abs_z=out['family_balanced'][k]['max_abs_z'], fam=out['family_balanced'][k]['fam_max_abs_z'])
                                for k in ('step_m7', 'tr60_m7', 'total_m7', 'decl4yr', 'step_fer', 'step_pet')} if 'family_balanced' in out else None,
        candidate='family 27 post-M7 60-d transient deviation (see family_table_balanced.csv); not significant after multiplicity',
        caveat_windows='coda windows end at the noise level, so window length depends on family amplitude and season; '
                       'fixed time-after-S windows would be cleaner (not redone)')
    json.dump(jl(out), open(f'{OUT}/summary.json', 'w'), indent=1)
    if T is not None:
        fig_family(T, R)
        fig_nulls(out)
    fig_curves(T)
    fig_common(B, extra)
    fig_synth(sy)
    fig_coh(coh)
    fb = out.get('family_balanced', {})
    for k in ('step_m7', 'tr60_m7', 'total_m7', 'decl4yr', 'step_fer', 'step_pet'):
        if k in fb:
            print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in fb[k].items() if kk != 'Q_perm'})
    for r in fb.get('regressions_min_p', []):
        print('REG', r['quantity'], r['covariate'], 'slope %.4f z_boot %.2f p_boot %.3f p_perm %.3f' % (r['slope'], r['z_boot'], r['p_boot'], r['p_perm']))


if __name__ == '__main__':
    main()
