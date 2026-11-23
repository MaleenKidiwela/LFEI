#!/usr/bin/env python3
"""Placebo calibration of the main full-coda dv/v claims (steps, annual cycle, post-2021 decline).

1) Coseismic steps (Petrolia, Ferndale, M7), surface / borehole:
   a) per family x group fits of the original dvv_cwire full-coda pairs; placebo steps at the 200 common random
      pre-M7 dates of sci_transient_all.py (step-only placebo for Petrolia / Ferndale, step + 60-d transient
      placebo for the M7 step, mirroring the model); pooled = inverse-placebo-variance mean, null = the same mean
      of the placebo steps at the same dates.
   b) per station-channel fits pooling all families (dvv_cwire_pool.py layout, one offset per family), same
      placebos; then the station step-vs-log10 PGV regression of sci_dvv_depth.py part A (surface, measured PGV,
      WLS + intrinsic scatter) re-done with placebo-calibrated SEs and against the slope of the PLACEBO steps.
2) Annual cycle: the same fits with the annual sin/cos replaced by sin/cos of a random period in 0.7-0.9 or
   1.1-1.4 yr (100 common periods); amplitude (and coherent pooled amplitude) vs the placebo amplitudes.
3) Post-2021 decline: yearly means of the pooled per-channel curves (data/dvv/pool/*_curve.csv, as
   dvv_longterm.py); 2021 -> 2025 change and the 2019/20 -> 2025 change of dvv_longterm_model.py vs the same
   changes in all fully pre-2021 windows (2009..2020), per channel and group median; Spearman with summed PGV.
Stages --fit (4 processes, cache data/sci/dvv_placebo/fits.pkl), --analyze.
Outputs: data/sci/dvv_placebo/*.csv, summary.json, figures/sci_dvv_placebo.png
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
from dvv_cwire import epoch_s  # noqa: E402
from sci_dvv_placebo_fit import channel_job, family_job  # noqa: E402
from sci_transient_all import emp_p, placebo_dates  # noqa: E402
from sci_transient_all_fit import T_PET  # noqa: E402

OUT = 'data/sci/dvv_placebo'
NPER, NBOOT = 100, 100
GRPS = ['surface', 'borehole']
os.makedirs(OUT, exist_ok=True)


def periods():
    r = np.random.default_rng(31)
    P = []
    while len(P) < NPER:
        x = r.uniform(0.7, 1.4)
        if not 0.9 <= x <= 1.1:
            P.append(x)
    return np.array(P)


def fit_stage():
    import glob
    import re
    from dvv_cwire_pool import load_all
    dates, pers = placebo_dates(), periods()
    fams = sorted(int(re.search(r'fam(\d+)_cwire_pairs', f).group(1)) for f in glob.glob('data/dvv/fam*_cwire_pairs.parquet')
                  if os.path.exists(f.replace('_pairs.parquet', '_channels.csv')) and os.path.exists(f.replace('_pairs.parquet', '_stacks.csv')))
    jobs = [(family_job, (F, g, dates, pers, NBOOT)) for F in fams for g in GRPS]
    P, T = load_all(0.4)
    P = P[P.cc >= 0.2]
    for c, p in P.groupby('chan'):
        if len(p) < 3000:
            continue
        p = p.reset_index(drop=True)
        p['tb'] = np.array([T[F][b] for F, b in zip(p.fam, p.b)])
        p['ta'] = np.array([T[F][a] if a >= 0 else np.nan for F, a in zip(p.fam, p.a)])
        jobs.append((channel_job, (c, p, dates, pers, NBOOT)))
    print(f'{len(fams)} families, {len(jobs)} jobs', flush=True)
    res = {}
    with Pool(4) as pool:
        for k, r in pool.imap_unordered(_run, jobs):
            res[k] = r
            if r is not None:
                print(k, 'm7 step %.4f  annual %.4f' % (r['m'][5] * 100, np.hypot(r['m'][1], r['m'][2]) * 100), flush=True)
    # own cache, read only by --analyze (trusted)
    pickle.dump(dict(dates=dates, periods=pers, res=res), open(f'{OUT}/fits.pkl', 'wb'))


def _run(job):
    fn, args = job
    return fn(args)


# ============================================================== analysis
QUAKE_COL = {'pet': (3, 'placebo_step'), 'fer': (4, 'placebo_step'), 'm7': (5, 'placebo_steptr')}


def unit_rows(res, kind):
    """Per fit unit (family x group, or channel): steps, annual, formal + placebo stats."""
    rows, PL = [], {}
    for k, r in res.items():
        if r is None or k[0] != kind:
            continue
        uid = k[1:] if kind == 'fam' else k[1]
        grp = k[2] if kind == 'fam' else ('borehole' if k[1].startswith('PB.') else 'surface')
        row = dict(unit=str(uid), group=grp, n_key=r['n_key'], n_rows=r['n_rows'],
                   t_first=str(pd.to_datetime(r['t_first'], unit='s').date()))
        B = r['stboot']
        for q, (j, pk) in QUAKE_COL.items():
            if q in ('pet', 'fer') and r['t_first'] > T_PET - 365 * 86400:
                continue                               # Petrolia / Ferndale steps not separable (late data start)
            v, pl = r['m'][j], r[pk]
            ok = np.isfinite(pl)
            if ok.sum() < 20:
                continue
            row[f'step_{q}_pct'] = v * 100
            row[f'step_{q}_se_formal_pct'] = np.nanstd(B[:, j]) * 100
            row[f'step_{q}_se_placebo_pct'] = pl[ok].std() * 100
            row[f'step_{q}_z'] = (v - pl[ok].mean()) / pl[ok].std()
            row[f'step_{q}_p_emp'] = emp_p(v, pl[ok])
            PL[(grp, f'step_{q}', str(uid))] = (v, pl, np.nanstd(B[:, j]))
        s, c = r['m'][1], r['m'][2]
        amp = np.hypot(s, c)
        pa = np.hypot(r['placebo_annual'][:, 0], r['placebo_annual'][:, 1])
        ok = np.isfinite(pa)
        row['annual_amp_pct'] = amp * 100
        row['annual_peak_doy'] = ((np.pi / 2 - np.arctan2(c, s)) % (2 * np.pi)) / (2 * np.pi) * 365.25
        row['annual_amp_se_formal_pct'] = np.nanstd(np.hypot(B[:, 1], B[:, 2])) * 100
        row['annual_amp_placebo_median_pct'] = np.median(pa[ok]) * 100
        row['annual_amp_placebo_sd_pct'] = pa[ok].std() * 100
        row['annual_amp_z'] = (amp - pa[ok].mean()) / pa[ok].std()
        row['annual_amp_p_emp'] = (1 + np.sum(pa[ok] >= amp)) / (ok.sum() + 1)
        PL[(grp, 'annual', str(uid))] = ((s, c), r['placebo_annual'], B[:, 1:3])
        rows.append(row)
    return pd.DataFrame(rows), PL


def pool_steps(PL, grp, q):
    keys = [k for k in PL if k[0] == grp and k[1] == f'step_{q}']
    if len(keys) < 2:
        return None
    v = np.array([PL[k][0] for k in keys])
    P = np.array([PL[k][1] for k in keys])
    se_f = np.array([PL[k][2] for k in keys])
    sd = np.nanstd(P, 1)
    w = 1 / sd ** 2
    w /= w.sum()
    real = np.sum(w * v)
    ok = np.isfinite(P)
    plp = np.sum(w[:, None] * np.where(ok, P, 0), 0) / np.sum(w[:, None] * ok, 0)
    plp = plp[np.isfinite(plp)]
    fj = np.array([np.sum(np.delete(w, i) * np.delete(v, i)) / np.sum(np.delete(w, i)) for i in range(len(v))])
    return dict(group=grp, quantity=f'step_{q}', n_units=len(v), pooled_pct=real * 100,
                se_formal_pct=np.sqrt(np.sum(w ** 2 * se_f ** 2)) * 100,
                se_placebo_pct=plp.std() * 100, z_placebo=(real - plp.mean()) / plp.std(), p_emp=emp_p(real, plp),
                placebo_mean_pct=plp.mean() * 100, se_unit_jackknife_pct=np.sqrt((len(v) - 1) / len(v) * np.sum((fj - fj.mean()) ** 2)) * 100,
                median_unit_pct=np.median(v) * 100, frac_units_absz_gt2=np.mean(np.abs((v - np.nanmean(P, 1)) / sd) > 2))


def pool_annual(PL, grp):
    keys = [k for k in PL if k[0] == grp and k[1] == 'annual']
    sc = np.array([PL[k][0] for k in keys])                     # (n, 2)
    PA = np.array([PL[k][1] for k in keys])                     # (n, NPER, 2)
    B = [PL[k][2] for k in keys]
    var = np.nanmean(np.nanvar(PA, 1), 1)
    w = 1 / var
    w /= w.sum()
    coh = np.sum(w[:, None] * sc, 0)
    amp = np.hypot(*coh)
    ok = np.isfinite(PA[:, :, 0])
    pc = np.array([np.sum(w[ok[:, i]] * PA[ok[:, i], i].T, 1) / w[ok[:, i]].sum() for i in range(PA.shape[1])])
    pamp = np.hypot(pc[:, 0], pc[:, 1])
    nb = min(len(b) for b in B)
    bc = np.array([np.nansum([w[j] * B[j][i] for j in range(len(B))], 0) for i in range(nb)])
    med = np.median(np.hypot(sc[:, 0], sc[:, 1]))
    pmed = np.array([np.nanmedian(np.hypot(PA[:, i, 0], PA[:, i, 1])) for i in range(PA.shape[1])])
    return dict(group=grp, quantity='annual', n_units=len(keys), pooled_pct=amp * 100,
                se_formal_pct=np.nanstd(np.hypot(bc[:, 0], bc[:, 1])) * 100,
                se_placebo_pct=pamp.std() * 100, z_placebo=(amp - pamp.mean()) / pamp.std(),
                p_emp=(1 + np.sum(pamp >= amp)) / (len(pamp) + 1), placebo_mean_pct=pamp.mean() * 100,
                pooled_peak_doy=((np.pi / 2 - np.arctan2(coh[1], coh[0])) % (2 * np.pi)) / (2 * np.pi) * 365.25,
                median_unit_pct=med * 100, median_unit_placebo_mean_pct=pmed.mean() * 100,
                median_unit_placebo_sd_pct=pmed.std() * 100, median_unit_z=(med - pmed.mean()) / pmed.std(),
                median_unit_p_emp=(1 + np.sum(pmed >= med)) / (len(pmed) + 1))


def pgv_regression(CH, PL, dates):
    """sci_dvv_depth part A, M7 surface step vs log10 measured PGV, with original / placebo SEs and placebo slopes."""
    from sci_dvv_depth import combine, wls_scatter
    orig = pd.read_csv('data/dvv/pool/params_by_channel.csv')
    S0 = pd.read_csv('data/sci/dvv_depth/station_steps_pgv.csv').set_index('sta')
    orig['sta'] = orig.chan.str.split('.').str[:2].str.join('.')
    plsd = {k[2]: np.nanstd(PL[k][1]) for k in PL if k[1] == 'step_m7'}
    plv = {k[2]: PL[k] for k in PL if k[1] == 'step_m7'}
    rng = np.random.default_rng(4)
    out = []

    def station_table(val_of, se_of):
        rows = []
        for sta, g in orig.groupby('sta'):
            if len(g) == 0 or sta.startswith('PB.') or sta not in S0.index or S0.loc[sta, 'pgv_m7_src'] != 'measured':
                continue
            vv = np.array([val_of(c) for c in g.chan], float)
            ss = np.array([se_of(c) for c in g.chan], float)
            k = np.isfinite(vv) & np.isfinite(ss)
            if k.sum() == 0:
                continue
            v, s = combine(vv[k], ss[k])
            rows.append((sta, v, s, np.log10(S0.loc[sta, 'pgv_m7'])))
        return pd.DataFrame(rows, columns=['sta', 'y', 's', 'lp'])

    om = orig.set_index('chan')
    sur = [c for c in plsd if not c.startswith('PB.') and c in om.index]
    ratio = float(np.median([plsd[c] * 100 / om.loc[c, 'step_m7_pct_se'] for c in sur]))

    def cal_se(c):          # placebo SE where the record allows it, else formal SE x median surface ratio
        return plsd[c] * 100 if c in plsd and np.isfinite(plsd[c]) else om.loc[c, 'step_m7_pct_se'] * ratio
    variants = {
        'original values, original SE': (lambda c: om.loc[c, 'step_m7_pct'], lambda c: om.loc[c, 'step_m7_pct_se']),
        'original values, calibrated SE (all stations)': (lambda c: om.loc[c, 'step_m7_pct'], cal_se),
        'original values, placebo SE (stations with pre-2021 records only)':
            (lambda c: om.loc[c, 'step_m7_pct'] if c in plsd else np.nan, lambda c: plsd.get(c, np.nan) * 100)}
    for name, (vf, sf) in variants.items():
        d = station_table(vf, sf)
        X = np.c_[np.ones(len(d)), d.lp]
        b, se, si, chi0, _ = wls_scatter(d.y.values, d.s.values, X)
        perm = np.array([wls_scatter(d.y.values, d.s.values, np.c_[np.ones(len(d)), rng.permutation(d.lp.values)])[0][1]
                         for _ in range(1000)])
        r = dict(variant=name, n_sta=len(d), slope_pct_per_decade=b[1], slope_se=se[1], sig_int=si, chi2r_noscatter=chi0,
                 p_perm=float(np.mean(np.abs(perm) >= abs(b[1]))))
        # placebo-step slopes: the same regression on each date's placebo steps (keeps station correlation)
        nb = []
        for i in range(len(dates)):
            vals = {c: plv[c][1][i] * 100 for c in plv}
            dp_ = station_table(lambda c: vals.get(c, np.nan), sf)
            if len(dp_) < 0.7 * len(d):            # too few stations covered at this date
                continue
            nb.append(wls_scatter(dp_.y.values, dp_.s.values, np.c_[np.ones(len(dp_)), dp_.lp])[0][1])
        nb = np.array(nb)
        r['se_ratio_used'] = ratio
        if len(nb) >= 10:
            r.update(placebo_slope_sd=nb.std(), placebo_slope_mean=nb.mean(), n_placebo=len(nb),
                     z_placebo=(b[1] - nb.mean()) / nb.std(), p_placebo=emp_p(b[1], nb))
        else:
            r.update(n_placebo=len(nb), z_placebo=np.nan, p_placebo=np.nan, placebo_slope_sd=np.nan)
        out.append(r)
    return pd.DataFrame(out)


def longterm():
    import glob
    lt = pd.read_csv('data/dvv/pool/longterm_model.csv')[['chan', 'pgv_sum']].set_index('chan')
    Y = {}
    for f in glob.glob('data/dvv/pool/*_curve.csv'):
        ch = f.split('/')[-1].replace('_curve.csv', '')
        c = pd.read_csv(f)
        c['y'] = pd.to_datetime(c.t).dt.year
        Y[ch] = c[c.y.between(2009, 2025)].groupby('y').dvv_pct.mean()
    Y = pd.DataFrame(Y).T                                       # channel x year
    grp = np.where(Y.index.str.startswith('PB.'), 'borehole', 'surface')
    rows, rng = [], np.random.default_rng(8)
    defs = {'4yr: 2021->2025': [(y, y + 4) for y in range(2009, 2017)] + [(2021, 2025)],
            '2019/20->2025': [(y, y + 5) for y in range(2009, 2015)] + [(2019, 2025)]}
    for name, wins in defs.items():
        for g in GRPS:
            Yg = Y[grp == g]
            vals = []
            for y0, y1 in wins:
                base = Yg[[y0, y0 + 1]].mean(1) if name.startswith('2019') else Yg[y0]
                d = (Yg[y1] - base).dropna()
                vals.append(d)
            claim, null = vals[-1], vals[:-1]
            chans = claim.index
            nm = np.array([np.median(n.reindex(chans).dropna()) for n in null])
            bs = [np.median(claim.values[rng.integers(0, len(claim), len(claim))]) for _ in range(1000)]
            # per channel: claim vs that channel's own placebo windows
            Zc = []
            for ch in chans:
                nn = np.array([n.get(ch, np.nan) for n in null])
                nn = nn[np.isfinite(nn)]
                if len(nn) >= 4:
                    Zc.append((claim[ch] - nn.mean()) / nn.std(ddof=1))
            Zc = np.array(Zc)
            # Spearman with summed PGV, claim and placebo windows
            ok = [c for c in chans if c in lt.index and np.isfinite(lt.loc[c, 'pgv_sum'])]
            rho = stats.spearmanr(np.log10(lt.loc[ok, 'pgv_sum']), claim[ok])[0] if len(ok) > 4 else np.nan
            rnull = [stats.spearmanr(np.log10(lt.loc[ok, 'pgv_sum']), n.reindex(ok))[0] for n in null] if len(ok) > 4 else []
            rows.append(dict(metric=name, group=g, n_chan=len(chans), median_change_pct=np.median(claim) * 1,
                             se_formal_channel_boot_pct=np.std(bs), n_placebo_windows=len(nm),
                             placebo_median_changes=';'.join(f'{x:+.3f}' for x in nm),
                             se_placebo_pct=nm.std(ddof=1), z_placebo=(np.median(claim) - nm.mean()) / nm.std(ddof=1),
                             max_abs_placebo_pct=np.max(np.abs(nm)),
                             per_channel_median_z=np.median(Zc), frac_channels_z_lt_m2=np.mean(Zc < -2),
                             spearman_pgv=rho, spearman_pgv_placebo=';'.join(f'{x:+.2f}' for x in rnull)))
    # pooled over both groups for the Spearman (as the original claim, all channels with PGV)
    return pd.DataFrame(rows), Y


def spearman_all(Y):
    lt = pd.read_csv('data/dvv/pool/longterm_model.csv').set_index('chan')
    ok = lt.index[np.isfinite(lt.pgv_sum)]
    x = np.log10(lt.loc[ok, 'pgv_sum'])
    r0, p0 = stats.spearmanr(x, lt.loc[ok, 'decline_2021_2025'])
    nulls = []
    for y in range(2009, 2015):
        d = Y.reindex(ok)[y + 5] - Y.reindex(ok)[[y, y + 1]].mean(1)
        m = d.notna()
        nulls.append(stats.spearmanr(x[m], d[m])[0])
    return dict(n=len(ok), rho=r0, p_formal=p0, placebo_rhos=nulls, placebo_rho_sd=float(np.std(nulls, ddof=1)),
                z_placebo=(r0 - np.mean(nulls)) / np.std(nulls, ddof=1))


def analyze():
    # own cache from --fit (trusted)
    Z = pickle.load(open(f'{OUT}/fits.pkl', 'rb'))
    res, dates = Z['res'], Z['dates']
    FAM, PLf = unit_rows(res, 'fam')
    CH, PLc = unit_rows(res, 'chan')
    FAM.to_csv(f'{OUT}/per_family_group.csv', index=False)
    CH.to_csv(f'{OUT}/per_channel.csv', index=False)
    # check: per-channel refit vs dvv_cwire_pool
    orig = pd.read_csv('data/dvv/pool/params_by_channel.csv').set_index('chan')
    cmp = CH.set_index('unit').join(orig[['step_m7_pct', 'step_m7_pct_se', 'annual_amp_pct']], rsuffix='_pool')
    print('refit vs pool: r(step_m7)=%.3f  r(annual)=%.3f  median se_placebo/se_pool(M7)=%.2f' % (
        cmp[['step_m7_pct', 'step_m7_pct_pool']].corr().iloc[0, 1], cmp[['annual_amp_pct', 'annual_amp_pct_pool']].corr().iloc[0, 1],
        np.nanmedian(cmp.step_m7_se_placebo_pct / cmp.step_m7_pct_se)))
    PR = []
    for lvl, PL in [('family', PLf), ('channel', PLc)]:
        for g in GRPS:
            for q in ['pet', 'fer', 'm7']:
                r = pool_steps(PL, g, q)
                if r:
                    PR.append(dict(level=lvl, **r))
            PR.append(dict(level=lvl, **pool_annual(PL, g)))
    PR = pd.DataFrame(PR)
    PR.to_csv(f'{OUT}/pooled.csv', index=False)
    print(PR.round(4).to_string())
    REG = pgv_regression(CH, PLc, dates)
    REG.to_csv(f'{OUT}/pgv_regression.csv', index=False)
    print(REG.round(4).to_string())
    LT, Y = longterm()
    LT.to_csv(f'{OUT}/longterm.csv', index=False)
    SP = spearman_all(Y)
    print(LT.round(4).to_string())
    print(SP)
    for lvl, D in [('family', FAM), ('channel', CH)]:
        for g in GRPS:
            d = D[D.group == g]
            for q in ['pet', 'fer', 'm7']:
                if f'step_{q}_se_placebo_pct' in d:
                    print(lvl, g, q, 'median placebo/formal SE ratio %.2f' % np.nanmedian(d[f'step_{q}_se_placebo_pct'] / d[f'step_{q}_se_formal_pct']))
    json.dump(dict(pooled=PR.to_dict('records'), pgv_regression=REG.to_dict('records'), longterm=LT.to_dict('records'),
                   longterm_spearman_all=SP, n_dates=len(dates), periods=list(Z['periods']),
                   se_ratio={f'{lvl}_{g}_{q}': float(np.nanmedian(D[D.group == g][f'step_{q}_se_placebo_pct'] / D[D.group == g][f'step_{q}_se_formal_pct']))
                             for lvl, D in [('family', FAM), ('channel', CH)] for g in GRPS for q in ['pet', 'fer', 'm7']
                             if f'step_{q}_se_placebo_pct' in D}),
              open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    figure(PR, REG, LT, PLc, Y)


def figure(PR, REG, LT, PLc, Y):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a = ax[0]
    x = 0
    ticks = []
    for lvl in ['family', 'channel']:
        for g in GRPS:
            for q in ['step_pet', 'step_fer', 'step_m7', 'annual']:
                r = PR[(PR.level == lvl) & (PR.group == g) & (PR.quantity == q)]
                if not len(r):
                    continue
                r = r.iloc[0]
                a.bar(x, r.z_placebo, color={'step_pet': 'C0', 'step_fer': 'C1', 'step_m7': 'C3', 'annual': 'C2'}[q])
                a.text(x, r.z_placebo, f'{r.pooled_pct:+.3f}', fontsize=6, ha='center', va='bottom' if r.z_placebo > 0 else 'top')
                ticks.append((x, f'{lvl[:4]} {g[:4]}\n{q.replace("step_", "")}'))
                x += 1
            x += 0.5
    a.set_xticks([t[0] for t in ticks])
    a.set_xticklabels([t[1] for t in ticks], fontsize=6, rotation=90)
    for h in (-2, 2):
        a.axhline(h, color='0.5', ls='--', lw=0.6)
    a.set_ylabel('z against placebo (pooled)')
    a.set_title('Pooled steps / annual amplitude vs placebo (labels: effect %)', fontsize=9)
    a = ax[1]
    plv = {k[2]: PLc[k] for k in PLc if k[1] == 'step_m7' and not k[2].startswith('PB.')}
    S0 = pd.read_csv('data/sci/dvv_depth/station_steps_pgv.csv').set_index('sta')
    for c, (v, pl, sf) in plv.items():
        sta = '.'.join(c.split('.')[:2])
        if sta in S0.index and S0.loc[sta, 'pgv_m7_src'] == 'measured':
            lp = np.log10(S0.loc[sta, 'pgv_m7'])
            a.errorbar(lp, v * 100, np.nanstd(pl) * 100, fmt='o', color='C3', ms=4, alpha=0.8)
            a.errorbar(lp + 0.01, v * 100, sf * 100, fmt='none', color='k', lw=2)
    a.set_xlabel('log10 measured M7 PGV (cm/s)')
    a.set_ylabel('M7 step per surface channel (%)')
    t = REG.set_index('variant')
    a.set_title('channels with pre-2021 records; red: placebo SE, black: formal SE\n' + '\n'.join(
        f'{k[:40]}: n={r.n_sta} {r.slope_pct_per_decade:+.3f}+-{r.slope_se:.3f}, p_perm={r.p_perm:.2f}, z_pl={r.z_placebo:+.1f}' for k, r in t.iterrows()), fontsize=7)
    a = ax[2]
    for g, col in [('borehole', 'C0'), ('surface', 'C3')]:
        Yg = Y[Y.index.str.startswith('PB.') == (g == 'borehole')]
        med = (Yg.sub(Yg.mean(1), axis=0)).median()
        a.plot(med.index, med.values, 'o-', color=col, label=g)
        r = LT[(LT.metric == '4yr: 2021->2025') & (LT.group == g)].iloc[0]
        a.text(0.02, 0.12 if g == 'surface' else 0.05, transform=a.transAxes, s=
               f'{g}: 2021->25 {r.median_change_pct:+.3f}%, placebo 4-yr sd {r.se_placebo_pct:.3f}, z={r.z_placebo:+.1f}',
               fontsize=7, color=col)
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel('median yearly dv/v (%), channel-demeaned')
    a.set_title('Long-term: post-2021 change vs 4-yr changes 2009-2021', fontsize=9)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('figures/sci_dvv_placebo.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    if '--fit' in sys.argv:
        fit_stage()
    if '--analyze' in sys.argv:
        analyze()
