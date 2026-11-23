#!/usr/bin/env python3
"""The early-coda 60-d M7 velocity transient in ALL families with enough data, pooled; spatial pattern;
Petrolia / Ferndale control.

Families: >= 20 channels with median_cc_ref >= 0.4 in data/dvv/fam{F}_cwire_channels.csv.
Pairs: early (b28_early, tau = t - S in 3-8 s) and late (b28_late, tau 8 s - end) windows, measured exactly as
sci_dvv_depth_measure.py (reused for 23, 8, 27; new families via scripts/sci_transient_all_measure.py).
Per family, group (surface / borehole) and window: the dvv_cwire parametric model (sci_transient_all_fit.py).
Amplitudes are divided by the injection recoveries (data/sci/dvv_depth/injection_recovery.csv).

Uncertainty / nulls
  * placebo: the same fit + (step, 60-d transient) at 200 COMMON random pre-M7 dates (only those inside each fit's data coverage are used) (2009-06 .. M7-200 d,
    excluding -60..+240 d around Petrolia and Ferndale); z = (A - mean_pl) / sd_pl, empirical two-sided p.
  * pooled = inverse-placebo-variance weighted mean over families; null = the same weighted mean of the
    placebo amplitudes at each common date (keeps cross-family correlation from shared stations / shared time).
  * station bootstrap (200 draws of the union station list, shared by all families), station and family
    jackknife.
  * spatial regressions: WLS slope on each standardized covariate; null = the same slope for the placebo
    amplitudes at each date (+ family permutation); Holm over the covariate x quantity family of tests.
Stages: --fit (per-family fits, 4 processes; cache data/sci/transient_all/fits.pkl), --analyze.
Outputs: data/sci/transient_all/*.csv, summary.json; figures/sci_transient_all_{families,spatial}.png
"""
import json
import os
import pickle
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_transient_all_fit import (DAY, T_FER, T_M7, T_PET, family_job, keep_channels, pair_paths,  # noqa: E402
                                   sta_of)
from dvv_cwire import epoch_s  # noqa: E402

OUT = 'data/sci/transient_all'
NPL, NB_STA, NB_STK = 200, 200, 100
ORIG = [23, 8, 27]
MIN_PL = 20
GRPS, WINS = ['surface', 'borehole'], ['early', 'late']


def families():
    fams = []
    for F in range(1, 40):
        f = f'data/dvv/fam{F}_cwire_channels.csv'
        if not os.path.exists(f):
            continue
        n = int((pd.read_csv(f).median_cc_ref >= 0.4).sum())
        if n >= 20:
            fams.append((F, n))
    return fams


def placebo_dates():
    rng = np.random.default_rng(20261122)
    lo, hi = epoch_s(['2009-06-01'])[0], T_M7 - 200 * DAY
    out = []
    while len(out) < NPL:
        t = rng.uniform(lo, hi)
        if any(q - 60 * DAY <= t <= q + 240 * DAY for q in (T_PET, T_FER)):
            continue
        out.append(t)
    return np.sort(np.array(out))


def fit_stage():
    fams = [F for F, _ in families() if os.path.exists(pair_paths(F)[0])]
    print('families with pairs:', fams, flush=True)
    dates = placebo_dates()
    stas = sorted({s for F in fams for s in sta_of(pd.Series(sorted(keep_channels(F))))})
    rng = np.random.default_rng(7)
    counts = rng.multinomial(len(stas), np.ones(len(stas)) / len(stas), size=NB_STA)
    jobs = [(F, dates, stas, counts, NB_STK, 1000 + F) for F in fams]
    res = {}
    with Pool(4) as p:
        for F, r in p.imap_unordered(family_job, jobs):
            res[F] = r
            print(F, {k: (v['n_chan'], round(v['m7'][6] * 100, 3)) for k, v in r.items()}, flush=True)
    # own cache, read only by --analyze (trusted)
    pickle.dump(dict(fams=fams, dates=dates, stations=stas, counts=counts, res=res), open(f'{OUT}/fits.pkl', 'wb'))


# =========================================================================== analysis helpers
def recoveries():
    I = pd.read_csv('data/sci/dvv_depth/injection_recovery.csv')
    return {(r.group, 'early' if r.cfg == 'b28_early' else 'late'): r.recovery for r in I.itertuples()
            if r.cfg in ('b28_early', 'b28_late')}


def emp_p(x, null):
    null = null[np.isfinite(null)]
    c = np.mean(null)
    return (1 + np.sum(np.abs(null - c) >= abs(x - c))) / (len(null) + 1)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def covariates(fams):
    D = pd.read_csv('data/sci/dose_response/m7_per_family_stress_vs_gap.csv').set_index('family')
    rows = []
    for F in fams:
        j = json.load(open(f'data/locations/fam{F}_3d.json'))
        rows.append(dict(fam=F, lon=j['lon'], lat=j['lat'], depth=j['depth_km'], dp=D.loc[F, 'dp'], gap_d=D.loc[F, 'gap_d']))
    C = pd.DataFrame(rows).set_index('fam')
    lat0, lon0 = C.lat.mean(), C.lon.mean()
    x = (C.lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (C.lat - lat0) * 110.57
    az = np.radians(290.0)
    C['s_along'] = x * np.sin(az) + y * np.cos(az)          # km, + toward WNW (strike 290)
    C['d_dip'] = x * np.sin(az + np.pi / 2) + y * np.cos(az + np.pi / 2)
    return C


def wslope(x, y, w):
    X = np.c_[np.ones_like(x), x]
    W = np.sqrt(w)
    return np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0][1]


# =========================================================================== analysis
def analyze():
    # own cache from --fit (trusted)
    Z = pickle.load(open(f'{OUT}/fits.pkl', 'rb'))
    res, dates, stas, fams = Z['res'], Z['dates'], Z['stations'], Z['fams']
    REC = recoveries()
    C = covariates(fams)
    # ---------------- per family / group table
    fam_rows, Q = [], {}   # Q[(grp, quantity)] = dict fam -> dict(val, pl (NPL,), sb (NB,), jk{sta: val})

    def put(grp, qn, F, val, pl, sb, jk, se_formal=np.nan):
        Q.setdefault((grp, qn), {})[F] = dict(val=val, pl=pl, sb=sb, jk=jk, se_formal=se_formal)

    for F in fams:
        for grp in GRPS:
            if (grp, 'early') not in res[F] or (grp, 'late') not in res[F]:
                continue
            E, L = res[F][(grp, 'early')], res[F][(grp, 'late')]
            re_, rl = REC[(grp, 'early')], REC[(grp, 'late')]
            e, l = E['m7'][6] / re_, L['m7'][6] / rl
            pe, pl_ = E['placebo'][:, 1] / re_, L['placebo'][:, 1] / rl
            if np.sum(np.isfinite(pe - pl_)) < MIN_PL:      # too few placebo dates inside the data coverage
                print(f'fam {F} {grp}: only {np.sum(np.isfinite(pe - pl_))} valid placebo dates -> excluded')
                continue
            for R_ in (E, L):        # near-singular resampled designs (station/stack sets losing a period) -> NaN
                for key in ('staboot_m7', 'staboot_pf', 'stboot'):
                    R_[key] = np.where(np.abs(R_[key]) < 0.05, R_[key], np.nan)
                R_['stajk'] = {s_: np.where(np.abs(v) < 0.05, v, np.nan) for s_, v in R_['stajk'].items()}
            sbe, sbl = E['staboot_m7'][:, 1] / re_, L['staboot_m7'][:, 1] / rl
            jke = {s: v[1] / re_ for s, v in E['stajk'].items() if np.isfinite(v[1])}
            jkl = {s: v[1] / rl for s, v in L['stajk'].items() if np.isfinite(v[1])}
            jkd = {s: jke.get(s, e) - jkl.get(s, l) for s in set(jke) | set(jkl)}
            put(grp, 'early', F, e, pe, sbe, jke, np.nanstd(E['stboot'][:, 1]) / re_)
            put(grp, 'late', F, l, pl_, sbl, jkl, np.nanstd(L['stboot'][:, 1]) / rl)
            put(grp, 'diff', F, e - l, pe - pl_, sbe - sbl, jkd,
                np.nanstd(E['stboot'][:, 1] / re_ - L['stboot'][:, 1] / rl))   # stack draws not paired; approx
            for ev, idx in [('pet', 7), ('fer', 8)]:
                ae, al = E['pf'][idx] / re_, L['pf'][idx] / rl
                k = 0 if ev == 'pet' else 1
                be, bl = E['staboot_pf'][:, k] / re_, L['staboot_pf'][:, k] / rl
                je = {s: v[2 + k] / re_ for s, v in E['stajk'].items() if np.isfinite(v[2 + k])}
                jl = {s: v[2 + k] / rl for s, v in L['stajk'].items() if np.isfinite(v[2 + k])}
                put(grp, f'{ev}_early', F, ae, pe, be, je)
                put(grp, f'{ev}_late', F, al, pl_, bl, jl)
                put(grp, f'{ev}_diff', F, ae - al, pe - pl_, be - bl, {s: je.get(s, ae) - jl.get(s, al) for s in set(je) | set(jl)})
            r = dict(fam=F, group=grp, n_chan_early=E['n_chan'], n_chan_late=L['n_chan'], n_sta=E['n_sta'],
                     stations=E['stations'], n_stacks_60d_m7=E['n_stacks_60d']['m7'],
                     n_stacks_60d_pet=E['n_stacks_60d']['pet'], n_stacks_60d_fer=E['n_stacks_60d']['fer'],
                     placebo_median_nstacks_60d=float(np.median(E['placebo_nstk'])),
                     early_raw_pct=E['m7'][6] * 100, late_raw_pct=L['m7'][6] * 100,
                     step_m7_early_raw_pct=E['m7'][5] * 100, step_m7_late_raw_pct=L['m7'][5] * 100)
            for qn in ['early', 'late', 'diff', 'pet_early', 'pet_diff', 'fer_early', 'fer_diff']:
                q = Q[(grp, qn)][F]
                pl = q['pl'][np.isfinite(q['pl'])]
                r[f'{qn}_corr_pct'] = q['val'] * 100
                r[f'{qn}_placebo_mean_pct'] = pl.mean() * 100
                r[f'{qn}_placebo_sd_pct'] = pl.std() * 100
                r[f'{qn}_z'] = (q['val'] - pl.mean()) / pl.std()
                r[f'{qn}_p_emp'] = emp_p(q['val'], pl)
                if qn in ('early', 'late', 'diff'):
                    mm = np.isfinite(q['pl']) & (E['placebo_nstk'] >= E['n_stacks_60d']['m7'])
                    plm = q['pl'][mm]
                    r[f'{qn}_n_placebo_matched'] = int(mm.sum())
                    r[f'{qn}_placebo_sd_matched_pct'] = plm.std() * 100 if mm.sum() >= 10 else np.nan
                    r[f'{qn}_z_matched'] = (q['val'] - plm.mean()) / plm.std() if mm.sum() >= 10 else np.nan
                r[f'{qn}_n_placebo'] = len(pl)
                if qn in ('early', 'late', 'diff'):
                    r[f'{qn}_se_stackboot_pct'] = q['se_formal'] * 100
            fam_rows.append(r)
    FT = pd.DataFrame(fam_rows)
    for grp in GRPS:
        for qn in ['early', 'diff', 'pet_early', 'fer_early']:
            m = FT.group == grp
            FT.loc[m, f'{qn}_q_bh'] = bh(FT.loc[m, f'{qn}_p_emp'].values)
    FT = FT.join(C, on='fam')
    FT.to_csv(f'{OUT}/per_family.csv', index=False)
    show = ['fam', 'group', 'n_chan_early', 'n_sta', 'n_stacks_60d_m7', 'early_corr_pct', 'early_placebo_sd_pct',
            'early_se_stackboot_pct', 'early_z', 'early_p_emp', 'late_corr_pct', 'late_z', 'diff_corr_pct', 'diff_z',
            'pet_early_z', 'fer_early_z']
    print(FT[show].round(3).to_string())

    # ---------------- pooling
    def pool(grp, qn, subset=None, weights='placebo'):
        d = Q[(grp, qn)]
        fl = [F for F in d if subset is None or F in subset]
        if not fl:
            return None
        base = qn.split('_')[-1] if qn.startswith(('pet', 'fer')) else qn
        sd = np.array([np.nanstd(Q[(grp, base)][F]['pl']) for F in fl])
        w = 1 / sd ** 2 if weights == 'placebo' else np.ones(len(fl))
        w = w / w.sum()
        val = np.array([d[F]['val'] for F in fl])
        real = np.sum(w * val)
        PL = np.array([d[F]['pl'] for F in fl])                 # (nf, NPL)
        ok = np.isfinite(PL)
        plp = np.nansum(w[:, None] * np.where(ok, PL, 0), 0) / np.sum(w[:, None] * ok, 0)
        SB = np.array([d[F]['sb'] for F in fl])
        okb = np.isfinite(SB)
        sbp = np.sum(w[:, None] * np.where(okb, SB, 0), 0) / np.sum(w[:, None] * okb, 0)
        # station jackknife over the union station list
        allst = sorted({s for F in fl for s in d[F]['jk']})
        jk = np.array([np.sum(w * np.array([d[F]['jk'].get(s, d[F]['val']) for F in fl])) for s in allst])
        n = len(jk)
        se_jk = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2)) if n > 2 else np.nan
        # family jackknife
        if len(fl) > 2:
            fj = np.array([np.sum(np.delete(w, i) * np.delete(val, i)) / np.sum(np.delete(w, i)) for i in range(len(fl))])
            se_fj = np.sqrt((len(fl) - 1) / len(fl) * np.sum((fj - fj.mean()) ** 2))
        else:
            se_fj = np.nan
        # heterogeneity: Q statistic vs placebo Q
        sdq = np.array([np.nanstd(d[F]['pl']) for F in fl])
        Qr = np.sum((val - real) ** 2 / sdq ** 2)
        Qp = np.array([np.nansum((PL[:, i] - plp[i]) ** 2 / sdq ** 2) for i in range(PL.shape[1])])
        plv = plp[np.isfinite(plp)]
        return dict(group=grp, quantity=qn, subset='all' if subset is None else ('orig3' if set(subset) == set(ORIG) else 'new'),
                    weights=weights, n_fam=len(fl), pooled_pct=real * 100, placebo_mean_pct=plv.mean() * 100,
                    placebo_sd_pct=plv.std() * 100, z_placebo=(real - plv.mean()) / plv.std(), p_emp_placebo=emp_p(real, plv),
                    se_station_boot_pct=np.nanstd(sbp) * 100, z_station_boot=real / np.nanstd(sbp),
                    se_station_jk_pct=se_jk * 100, se_family_jk_pct=se_fj * 100, n_sta_union=len(allst),
                    Q_het=Qr, Q_het_placebo_p=(1 + np.sum(Qp >= Qr)) / (len(Qp) + 1), Q_dof=len(fl) - 1,
                    _plp=plp, _sbp=sbp)
    PR, PLP = [], {}
    new = [F for F in fams if F not in ORIG]
    for grp in GRPS:
        for qn in ['early', 'late', 'diff', 'pet_early', 'pet_late', 'pet_diff', 'fer_early', 'fer_late', 'fer_diff']:
            for sub in [None, ORIG, new]:
                for wt in (['placebo', 'equal'] if sub is None else ['placebo']):
                    r = pool(grp, qn, sub, wt)
                    if r is None:
                        continue
                    PLP[(grp, qn, r['subset'], wt)] = (r.pop('_plp'), r.pop('_sbp'))
                    PR.append(r)
    PR = pd.DataFrame(PR)
    PR.to_csv(f'{OUT}/pooled.csv', index=False)
    print(PR.round(3).to_string())

    # ---------------- spatial pattern
    covs = ['dp', 'gap_d', 'depth', 's_along']
    CC = C[covs + ['lon', 'd_dip']].corr()
    CC.to_csv(f'{OUT}/covariate_correlations.csv')
    rng = np.random.default_rng(5)
    SR = []
    for grp in GRPS:
        for qn in ['early', 'diff']:
            d = Q[(grp, qn)]
            fl = [F for F in fams if F in d]
            y = np.array([d[F]['val'] for F in fl])
            PL = np.array([d[F]['pl'] for F in fl])
            sd = np.nanstd(PL, 1)
            w = 1 / sd ** 2
            for cv in covs + ['lon']:
                x = C.loc[fl, cv].values
                xs = (x - x.mean()) / x.std()
                b = wslope(xs, y, w)
                nb = []
                for i in range(PL.shape[1]):
                    ok = np.isfinite(PL[:, i])
                    if ok.sum() >= 5:
                        nb.append(wslope(xs[ok], PL[ok, i], w[ok]))
                nb = np.array(nb)
                perm = np.array([wslope(xs[rng.permutation(len(xs))], y, w) for _ in range(2000)])
                rho = stats.spearmanr(x, y)[0]
                SR.append(dict(group=grp, quantity=qn, covariate=cv, n_fam=len(fl), slope_pct_per_sd=b * 100,
                               null_sd_pct=nb.std() * 100, z_placebo=(b - nb.mean()) / nb.std(), p_placebo=emp_p(b, nb),
                               p_perm=(1 + np.sum(np.abs(perm) >= abs(b))) / (len(perm) + 1), spearman=rho,
                               cov_sd=x.std()))
            # partial: dp with s_along regressed out (and vice versa), WLS 2-covariate
            for a_, b_ in [('dp', 's_along'), ('s_along', 'dp'), ('dp', 'depth'), ('gap_d', 'dp')]:
                xa, xb = C.loc[fl, a_].values, C.loc[fl, b_].values
                xa = (xa - xa.mean()) / xa.std()
                xb = (xb - xb.mean()) / xb.std()
                X = np.c_[np.ones(len(fl)), xa, xb]
                W = np.sqrt(w)

                def coef(yy, X=X, W=W):
                    return np.linalg.lstsq(X * W[:, None], yy * W, rcond=None)[0][1]
                b = coef(y)
                nb = np.array([coef(np.where(np.isfinite(PL[:, i]), PL[:, i], 0)) for i in range(PL.shape[1])])
                SR.append(dict(group=grp, quantity=qn, covariate=f'{a_}|{b_}', n_fam=len(fl), slope_pct_per_sd=b * 100,
                               null_sd_pct=nb.std() * 100, z_placebo=(b - nb.mean()) / nb.std(), p_placebo=emp_p(b, nb),
                               p_perm=np.nan, spearman=np.nan, cov_sd=np.nan))
    SR = pd.DataFrame(SR)
    prim = SR.covariate.isin(covs)
    for grp in GRPS:
        m = prim & (SR.group == grp)
        SR.loc[m, 'p_placebo_holm'] = holm(SR.loc[m, 'p_placebo'].values)
        SR.loc[m, 'p_perm_holm'] = holm(SR.loc[m, 'p_perm'].values)
    SR.to_csv(f'{OUT}/spatial_regressions.csv', index=False)
    print(CC.round(2).to_string())
    print(SR.round(3).to_string())

    summary = dict(families=fams, n_families=len(fams), placebo_dates=[str(pd.to_datetime(t, unit='s').date()) for t in dates],
                   recoveries={f'{k[0]}_{k[1]}': v for k, v in REC.items()}, pooled=PR.to_dict('records'),
                   spatial=SR.to_dict('records'), covariate_corr=CC.round(3).to_dict(),
                   per_family=FT.drop(columns=['stations']).to_dict('records'),
                   joint_fit_crosscheck=(pd.read_csv(f'{OUT}/joint_fit.csv').to_dict('records')
                                         if os.path.exists(f'{OUT}/joint_fit.csv') else None))
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    figures(FT, PR, PLP, SR, C, Q, fams)


def figures(FT, PR, PLP, SR, C, Q, fams):
    # ---------------- figure 1: per family + pooled placebo distributions
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4)
    for gi, grp in enumerate(GRPS):
        a = fig.add_subplot(gs[gi, :2])
        d = FT[FT.group == grp].sort_values('s_along')
        x = np.arange(len(d))
        for off, qn, col, lab in [(-0.2, 'early', 'C3', 'early (S+3..8 s)'), (0.0, 'late', 'C0', 'late (S+8 s..end)'),
                                  (0.2, 'diff', 'k', 'early - late')]:
            a.errorbar(x + off, d[f'{qn}_corr_pct'] - d[f'{qn}_placebo_mean_pct'], d[f'{qn}_placebo_sd_pct'], fmt='o',
                       color=col, ms=4, capsize=1.5, label=lab, mfc='none' if qn == 'diff' else col)
        for j, F in enumerate(d.fam):
            if F in ORIG:
                a.axvspan(j - 0.4, j + 0.4, color='y', alpha=0.15, lw=0)
        a.set_xticks(x)
        a.set_xticklabels([f'{F}' for F in d.fam], fontsize=7)
        a.axhline(0, color='0.5', lw=0.5)
        a.set_xlabel('family (sorted along strike, ESE -> WNW; shaded = discovery families 23, 8, 27)', fontsize=8)
        a.set_ylabel('M7 60-d transient, recovery-corrected,\nminus placebo mean (%) +- placebo sd', fontsize=8)
        a.set_title(f'{grp}: per-family M7 transient', fontsize=10)
        a.legend(fontsize=7, ncol=3)
        for k, qn in enumerate(['early', 'diff']):
            a = fig.add_subplot(gs[gi, 2 + k])
            for sub, col in [('all', 'k'), ('new', 'C2'), ('orig3', 'C1')]:
                key = (grp, qn, sub, 'placebo')
                if key not in PLP:
                    continue
                plp = PLP[key][0] * 100
                r = PR[(PR.group == grp) & (PR.quantity == qn) & (PR.subset == sub) & (PR.weights == 'placebo')].iloc[0]
                a.hist(plp[np.isfinite(plp)], 25, histtype='step', color=col, density=True)
                a.axvline(r.pooled_pct, color=col, lw=2, label=f'{sub} ({r.n_fam}): {r.pooled_pct:+.3f}%, z={r.z_placebo:+.1f}')
            for ev, ls in [('pet', ':'), ('fer', '--')]:
                r = PR[(PR.group == grp) & (PR.quantity == f'{ev}_{qn}') & (PR.subset == 'all') & (PR.weights == 'placebo')].iloc[0]
                a.axvline(r.pooled_pct, color='C4', ls=ls, lw=1.5, label=f'{ev} (all): {r.pooled_pct:+.3f}%, z={r.z_placebo:+.1f}')
            a.set_title(f'{grp} pooled {qn}: M7 vs placebo dates', fontsize=9)
            a.set_xlabel('pooled transient (%)', fontsize=8)
            a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig('figures/sci_transient_all_families.png', dpi=130)
    plt.close(fig)

    # ---------------- figure 2: spatial pattern + controls
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    labels = {'dp': 'modelled pore-pressure change dp', 'gap_d': 'silence length gap_d (d)', 'depth': 'depth (km, 3-D loc.)',
              's_along': 'along-strike position (km, + WNW)'}
    for gi, grp in enumerate(GRPS):
        d = FT[FT.group == grp]
        for k, cv in enumerate(['dp', 'gap_d', 'depth', 's_along']):
            a = ax[gi, k]
            a.errorbar(d[cv], d.early_corr_pct - d.early_placebo_mean_pct, d.early_placebo_sd_pct, fmt='o', color='C3', ms=4,
                       capsize=1.5)
            for r in d.itertuples():
                a.annotate(str(r.fam), (getattr(r, cv), r.early_corr_pct - r.early_placebo_mean_pct), fontsize=6,
                           xytext=(2, 2), textcoords='offset points')
            s = SR[(SR.group == grp) & (SR.quantity == 'early') & (SR.covariate == cv)].iloc[0]
            a.axhline(0, color='0.5', lw=0.5)
            a.set_title(f'{grp} early: slope {s.slope_pct_per_sd:+.3f}%/sd, z={s.z_placebo:+.1f}\n'
                        f'p_placebo={s.p_placebo:.2f} (Holm {s.p_placebo_holm:.2f}), p_perm={s.p_perm:.2f}', fontsize=8)
            a.set_xlabel(labels[cv], fontsize=8)
            if k == 0:
                a.set_ylabel('early M7 transient (corr.) - placebo mean (%)', fontsize=8)
    fig.tight_layout()
    fig.savefig('figures/sci_transient_all_spatial.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    if '--fit' in sys.argv:
        fit_stage()
    if '--analyze' in sys.argv:
        analyze()
