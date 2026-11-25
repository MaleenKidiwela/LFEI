#!/usr/bin/env python3
"""Extra-quakes step tests: repeat the key LFE-based tests for the other strong nearby/offshore quakes
(2010-01-10 M6.5 offshore Eureka, 114 km; 2014-03-10 M6.8 offshore Mendocino, 156 km; these are the only
M>=6 within ~160 km besides Petrolia 2021, Ferndale 2022 and the M7 2024 in comcat_M5.5_200km_2008_2026.csv),
with the three known quakes as references.  Placebo-calibrated (random dates >= 90 d from any M>=6 quake).

Stages (each writes data/sci/extras/quakes2/<stage>*.csv|json):
  dose   : static stress / undrained dp at the LFEs from data/sci/dose_response/per_quake_drivers_responses.csv
           (sci_dose_response.py already modelled all 12 M>=5.5 dates incl. 2010 and 2014 -> reused, not recomputed)
  rate   : LFE rate ratio 7/30 d, longest network gap in 30 d, families silent beyond their own 99th-pct wait,
           for all 12 M>=5.5 dates; 500 placebo dates.
  dvv    : full-coda LFE dv/v (data/dvv/fam*_cwire_pairs.parquet; sci_dvv_placebo_fit/sci_transient_all_fit Fit,
           same keep rules) + steps at 2010 and 2014; per family x surface/borehole; 200 placebo steps; pooled by
           inverse placebo variance.
  sp     : S-P source-region dv/v (sph10, sph5) and d ln(Vp/Vs) (slab5SP) per-stack family inversions
           (data/sci/dvv_source_region/inv_all.pkl); network time model with extra 2010/2014 steps; placebo steps.
  tides  : M2/O1 tidal modulation 1 yr before vs 1 yr after (sci_tides functions); 120 placebo dates (M2).
  amp    : LFE amplitude (data/sci/moments/event_amplitudes.csv.gz, above T, qc ok): mean log10 A, +/-180 d and
           first 30 d vs 365 d before; per family, pooled; placebo dates.
  spec   : source spectra +/-180 d step (sci_source_spectra functions: load_family/step_sets/compare) at 2010/2014,
           pooled with the existing per-family placebo rows (dates within 180 d of 2010/2014 dropped).
  figs   : figures/sci_extras_quakes2.png ; summary : summary.json (key numbers)
Run: nice -n 10 python scripts/sci_extras_quakes2.py --stage all  (lfe env; <=2 processes)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import pickle
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/scripts')
OUT = f'{ROOT}/data/sci/extras/quakes2'
FIG = f'{ROOT}/figures'
os.makedirs(OUT, exist_ok=True)
DAY = 86400.0
YR = 365.25 * DAY


def ep(x):
    return (pd.Timestamp(x) - pd.Timestamp('1970-01-01')).total_seconds()


# main quakes (UTC)
Q = {'eur2010': '2010-01-10T00:27:39', 'mend2014': '2014-03-10T05:18:13',
     'pet2021': '2021-12-20T20:10:20', 'fer2022': '2022-12-20T10:34:24', 'm7_2024': '2024-12-05T18:44:21'}
NEW = ['eur2010', 'mend2014']
TQ = {k: ep(v) for k, v in Q.items()}
# all M>=5.5 within 200 km (dose-response dates) for the rate table
ALLQ = {'2010-01-10 M6.5': '2010-01-10T00:27:39', '2010-02-04 M5.9': '2010-02-04T20:20:21',
        '2012-02-13 M5.6': '2012-02-13T21:07:02', '2014-03-10 M6.8': '2014-03-10T05:18:13',
        '2015-01-28 M5.7': '2015-01-28T21:08:53', '2016-09-03 M5.6': '2016-09-03T03:27:57',
        '2019-06-23 M5.6': '2019-06-23T03:53:02', '2020-03-09 M5.8': '2020-03-09T02:59:08',
        '2021-12-20 M6.2': '2021-12-20T20:10:20', '2022-12-20 M6.4': '2022-12-20T10:34:24',
        '2023-05-21 M5.6': '2023-05-21T18:44:01', '2024-12-05 M7.0': '2024-12-05T18:44:21'}


def placebo_dates(n, seed, lo='2009-03-01', hi='2024-06-01', excl=90.0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        t = rng.uniform(ep(lo), ep(hi))
        if all(abs(t - q) > excl * DAY for q in TQ.values()):
            out.append(t)
    return np.sort(np.array(out))


def pctl(x, null):
    null = np.asarray(null)[np.isfinite(null)]
    return float(100.0 * np.mean(null < x)) if len(null) and np.isfinite(x) else np.nan


def pool_inv(vals, P, minp=50, cover=0.6):
    """vals: dict unit->value; P: dict unit->placebo array (common dates). Inverse-placebo-variance pooling;
    returns value, placebo mean/sd, z, pctl, n_units, and the pooled placebo array."""
    w = {u: 1.0 / max(np.nanvar(P[u]), 1e-12) for u in vals
         if np.isfinite(vals[u]) and np.isfinite(P[u]).sum() >= minp}
    if not w:
        return None
    U = list(w)
    ws = np.array([w[u] for u in U])
    X = float(np.sum(ws * np.array([vals[u] for u in U])) / ws.sum())
    M = np.array([P[u] for u in U]).T
    W = ws[None] * np.isfinite(M)
    Pk = np.nansum(np.nan_to_num(M) * W, 1) / np.maximum(W.sum(1), 1e-12)
    Pk = Pk[W.sum(1) / ws.sum() > cover]
    mu, sd = float(np.mean(Pk)), float(np.std(Pk))
    return dict(value=X, placebo_mean=mu, placebo_sd=sd, z=(X - mu) / sd, pctl=pctl(X, Pk),
                p2=float(np.mean(np.abs(Pk - mu) >= abs(X - mu))), n_units=len(U), n_placebo=len(Pk),
                det2sig=2 * sd), Pk


# ======================================================================================== dose
def stage_dose(a):
    d = pd.read_csv('data/sci/dose_response/per_quake_drivers_responses.csv')
    cols = ['quake', 'dist_km', 'mag', 'dtau', 'dsn', 'dCFS04', 'dCFS06', 'dp', 'dp_ens_lo', 'dp_ens_hi',
            'dp_fam_lo', 'dp_fam_hi', 'dsn_eff', 'dCFS_ud', 'pgv_rrup', 'dyn_strain_ue', 'ratio30_all', 'n_long_gap',
            'source_model']
    t = d[[c for c in cols if c in d]].copy()
    t = t[t.quake != '2021-12-20 M5.7']
    t.to_csv(f'{OUT}/dose.csv', index=False)
    m7 = t[t.quake.str.contains('M7.0')].iloc[0]
    oth = t[~t.quake.str.contains('M7.0') & (t.quake != '2021-12-20 M6.2')]
    s = dict(source='data/sci/dose_response/per_quake_drivers_responses.csv (sci_dose_response.py; reused, '
                    'already contains 2010-01-10 M6.5 and 2014-03-10 M6.8)',
             units='kPa (dp: undrained, B=0.7, compression positive; dsn>0 unclamping); PGV cm/s',
             m7_dp_ens_hi=float(m7.dp_ens_hi),
             max_dp_ens_lo_other=float(oth.dp_ens_lo.min()),
             quake_with_most_negative_other=str(oth.loc[oth.dp_ens_lo.idxmin(), 'quake']),
             dp_negative_unique_to_m7=bool(m7.dp_ens_hi < oth.dp_ens_lo.min()))
    for k in ('2010-01-10 M6.5', '2014-03-10 M6.8'):
        r = t[t.quake == k].iloc[0]
        s[k] = {c: (float(r[c]) if isinstance(r[c], (float, int, np.floating)) else str(r[c])) for c in t.columns
                if c != 'quake'}
    json.dump(s, open(f'{OUT}/dose_summary.json', 'w'), indent=1)
    print(t[['quake', 'dist_km', 'dCFS04', 'dp', 'dp_ens_lo', 'dp_ens_hi', 'pgv_rrup', 'ratio30_all']].round(3))
    print(json.dumps({k: v for k, v in s.items() if not isinstance(v, dict)}, indent=1))


# ======================================================================================== rate
def load_cat():
    from sci_tides import load
    d, _ = load()
    d['ts'] = (d.t - pd.Timestamp('1970-01-01')).dt.total_seconds()
    return d


def rate_stats(ts, fam, t0, famwaits):
    """ts sorted (s), fam ids. Returns r7, r30, maxgap30_h, n_fam_silent99."""
    def n(a, b):
        return np.searchsorted(ts, b) - np.searchsorted(ts, a)
    base = n(t0 - 365 * DAY, t0) / 365.0
    if base <= 0:
        return dict(r7=np.nan, r30=np.nan, gap30_h=np.nan, nsil=np.nan)
    r7 = n(t0, t0 + 7 * DAY) / 7.0 / base
    r30 = n(t0, t0 + 30 * DAY) / 30.0 / base
    i0, i1 = np.searchsorted(ts, t0), np.searchsorted(ts, t0 + 30 * DAY)
    seg = np.r_[t0, ts[i0:i1 + 1]]
    gap = np.max(np.diff(seg)) / 3600.0 if len(seg) > 1 else 30 * 24.0
    # families silent beyond the 99th pct of their own waits in the preceding 365 d
    nsil = 0
    for f, (ft) in famwaits.items():
        k0 = np.searchsorted(ft, t0 - 365 * DAY)
        k1 = np.searchsorted(ft, t0)
        if k1 - k0 < 20:
            continue
        w99 = np.percentile(np.diff(ft[k0:k1]), 99)
        nxt = ft[k1] if k1 < len(ft) else np.inf
        nsil += (nxt - t0) > w99
    return dict(r7=r7, r30=r30, gap30_h=gap, nsil=nsil)


def stage_rate(a):
    d = load_cat()
    rows = []
    P = placebo_dates(500, 501, lo='2009-01-15', hi='2025-09-01')
    e = pd.read_csv('data/sci/moments/event_amplitudes.csv.gz', usecols=['fam', 'time', 'qc_bad', 'above_T'])
    e = e[(~e.qc_bad) & e.above_T].copy()
    e['ts'] = (pd.to_datetime(e.time, format='mixed') - pd.Timestamp('1970-01-01')).dt.total_seconds()
    # aboveT: events above the completeness amplitude of the noisiest epoch (sci_moments) -> robust to detection loss
    for lab, sub in [('all', d), ('meancc0.137', d[d.meancc >= 0.137]), ('aboveT', e)]:
        ts = np.sort(sub.ts.values)
        fw = {f: np.sort(g.ts.values) for f, g in sub.groupby('fam')}
        pl = pd.DataFrame([rate_stats(ts, None, t, fw) for t in P])
        pl.insert(0, 'date', pd.to_datetime(P, unit='s'))
        pl.to_csv(f'{OUT}/rate_placebo_{lab}.csv', index=False)
        for k, v in ALLQ.items():
            r = rate_stats(ts, None, ep(v), fw)
            row = dict(subset=lab, quake=k)
            for s_, x in r.items():
                row[s_] = x
                row[s_ + '_pctl'] = pctl(x, pl[s_].values)
            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/rate.csv', index=False)
    print(R.round(3).to_string())


# ======================================================================================== dvv
DV_DATES = None


def dvv_job(args):
    F, grp, dates = args
    from sci_dvv_placebo_fit import load_family
    from sci_transient_all_fit import Fit
    p = load_family(F)
    p = p[p.chan.str.startswith('PB.') == (grp == 'borehole')].reset_index(drop=True)
    if p.chan.nunique() < 3:
        return (F, grp), None
    try:
        f = Fit(p)
    except np.linalg.LinAlgError:
        Fit.lstsq_fallback = True
        f = Fit(p)
        Fit.lstsq_fallback = False
        f.lstsq_fallback = True
    tg = np.unique(f.tb)

    def ok(t0):
        return ((tg >= t0) & (tg < t0 + 60 * DAY)).sum() >= 1 and ((tg >= t0 - 365 * DAY) & (tg < t0)).sum() >= 2 \
            and (tg >= t0).sum() >= 2

    spec = [('step', TQ[k]) for k in NEW if ok(TQ[k])]
    keys = [k for k in NEW if ok(TQ[k])]
    w = f.w

    def solve(X):
        try:
            x = f.solve(w, X)
        except np.linalg.LinAlgError:
            return None
        return x if np.all(np.abs(x) < 0.05) else None

    X = f.extra(spec)
    x = solve(X)
    r = dict(F=F, grp=grp, n_chan=int(p.chan.nunique()), n_rows=len(p), t_first=float(tg.min()),
             n_stacks=len(tg))
    for k in NEW:
        r[k] = np.nan
    if x is not None:
        for j, k in enumerate(keys):
            r[k] = x[7 + j]
        r['m7'], r['fer'], r['pet'] = x[5], x[4], x[3]
    pls = np.full(len(dates), np.nan)
    for i, t0 in enumerate(dates):
        if not ok(t0):
            continue
        xp = solve(f.extra(spec + [('step', t0)]))
        if xp is not None:
            pls[i] = xp[-1]
    r['placebo'] = pls
    return (F, grp), r


def stage_dvv(a):
    dates = placebo_dates(200, 202)
    fams = sorted(int(re.search(r'fam(\d+)_cwire_pairs', f).group(1))
                  for f in glob.glob('data/dvv/fam*_cwire_pairs.parquet'))
    jobs = [(F, g, dates) for F in fams for g in ('surface', 'borehole')]
    res = {}
    with Pool(a.procs) as pool:
        for k, r in pool.imap_unordered(dvv_job, jobs):
            res[k] = r
            if r is not None:
                print(k, {q: round(r[q] * 100, 4) for q in NEW}, int(np.isfinite(r['placebo']).sum()), flush=True)
    pickle.dump(dict(dates=dates, res=res), open(f'{OUT}/dvv_fits.pkl', 'wb'))
    analyze_dvv()


def analyze_dvv():
    Dd = pickle.load(open(f'{OUT}/dvv_fits.pkl', 'rb'))
    res = Dd['res']
    rows, pooled = [], []
    for (F, g), r in sorted(res.items()):
        if r is None:
            continue
        row = dict(F=F, group=g, n_chan=r['n_chan'], t_first=str(pd.to_datetime(r['t_first'], unit='s'))[:10])
        pl = r['placebo'] * 100
        row['placebo_sd_pct'] = float(np.nanstd(pl))
        row['n_placebo'] = int(np.isfinite(pl).sum())
        for q in NEW + ['pet', 'fer', 'm7']:
            v = r.get(q, np.nan) * 100
            row[q + '_pct'] = v
            row[q + '_z'] = (v - np.nanmean(pl)) / np.nanstd(pl) if np.isfinite(v) else np.nan
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/dvv_per_family.csv', index=False)
    for g in ('surface', 'borehole'):
        sub = {(F, gg): r for (F, gg), r in res.items() if gg == g and r is not None}
        for q in NEW + ['pet', 'fer', 'm7']:
            vals = {k: r.get(q, np.nan) * 100 for k, r in sub.items()}
            P = {k: r['placebo'] * 100 for k, r in sub.items()}
            out = pool_inv(vals, P)
            if out is None:
                continue
            o, _ = out
            # seasonal control: placebo dates in winter only (doy >= 320 or <= 60; all of Petrolia/Ferndale/M7/2010
            # are Dec/Jan)
            doy = pd.to_datetime(Dd['dates'], unit='s').dayofyear.values
            wint = (doy >= 320) | (doy <= 60)
            o2 = pool_inv(vals, {k: v[wint] for k, v in P.items()}, minp=20)
            if o2 is not None:
                o.update(winter_placebo_mean=o2[0]['placebo_mean'], winter_placebo_sd=o2[0]['placebo_sd'],
                         winter_z=o2[0]['z'], winter_pctl=o2[0]['pctl'], winter_n=o2[0]['n_placebo'])
            pooled.append(dict(group=g, quake=q, **o, recovery_note='waveform-level full-coda recovery ~0.7 '
                               '(NOTES 2026-11-23); det2sig_corr = det2sig / 0.7', det2sig_corr=o['det2sig'] / 0.7))
    Pd = pd.DataFrame(pooled)
    Pd.to_csv(f'{OUT}/dvv_pooled.csv', index=False)
    print(Pd.round(4).to_string())


# ======================================================================================== S-P
SP_PN = ['sin', 'cos', 'eur2010', 'mend2014', 'pet', 'fer', 'm7', 'epet', 'efer', 'em7', 'ramp']
T_RAMP = ep('2021-06-01')


def sp_cols(t, extra=None):
    yr = (t - ep('2008-01-01')) / YR
    c = [np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    for k in ['eur2010', 'mend2014', 'pet2021', 'fer2022', 'm7_2024']:
        c.append((t >= TQ[k]).astype(float))
    for k in ['pet2021', 'fer2022', 'm7_2024']:
        c.append(((t >= TQ[k]) & (t < TQ[k] + 60 * DAY)).astype(float))
    c.append(np.clip((t - T_RAMP) / YR, 0, None))
    if extra is not None:
        c.append((t >= extra).astype(float))
    return np.array(c).T


def sp_fit(d, col, extra=None, irls=2, use=None):
    t = d.t.values
    C = sp_cols(t, extra)
    if use is None:
        use = np.abs(C - C.mean(0)).sum(0) > 0
        for j, k in [(2, 'eur2010'), (3, 'mend2014')]:
            tq = TQ[k]
            use[j] = ((t < tq) & (t > tq - 2 * YR)).sum() >= 2 and ((t >= tq) & (t < tq + 2 * YR)).sum() >= 2
        if extra is not None:
            use[-1] = ((t < extra) & (t > extra - 2 * YR)).sum() >= 2 and ((t >= extra) & (t < extra + 2 * YR)).sum() >= 2
            if not use[-1]:
                return None, None
    Fs, fi = np.unique(d.F.values, return_inverse=True)
    AF = np.zeros((len(d), len(Fs)))
    AF[np.arange(len(d)), fi] = 1
    A = np.c_[AF, C[:, use]]
    y = d[col].values
    w0 = 1 / d[col + '_se'].values ** 2
    w = w0.copy()
    for it in range(irls + 1):
        W = np.sqrt(w)[:, None]
        x, *_ = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)
        r = y - A @ x
        if it < irls:
            z = np.abs(r) * np.sqrt(w0)
            z = z / (1.4826 * np.median(z) + 1e-20)
            w = w0 * np.where(z <= 2.0, 1.0, 2.0 / np.maximum(z, 1e-12))
    th = np.full(C.shape[1], np.nan)
    th[use] = x[len(Fs):]
    chi = float(np.sum(w * r ** 2) / max(len(d) - A.shape[1], 1))
    cov = np.linalg.pinv((A * w[:, None]).T @ A) * max(chi, 1.0)
    se = np.full(C.shape[1], np.nan)
    se[use] = np.sqrt(np.diag(cov))[len(Fs):]
    return th, se


def stage_sp(a):
    res = pickle.load(open('data/sci/dvv_source_region/inv_all.pkl', 'rb'))   # project cache (trusted)
    o = res['slab5SP'].copy()
    o['vpvs'] = o.eps1 - o.eps0
    o['vpvs_se'] = np.hypot(o.eps0_se, o.eps1_se)
    sets = [('sph10', res['sph10'], 'eps0'), ('sph5', res['sph5'], 'eps0'), ('slab5SP', o, 'vpvs')]
    dates = placebo_dates(150, 303, lo='2009-03-01', hi='2024-03-01')
    rows, prow = [], []
    for name, df, col in sets:
        d = df[['F', 't', col, col + '_se']].copy()
        d = d[np.isfinite(d[col]) & (d[col + '_se'] > 0)]
        d = d[d[col + '_se'] < 20 * d[col + '_se'].median()].reset_index(drop=True)
        th, se = sp_fit(d, col)
        pl = []
        for t0 in dates:
            thp, _ = sp_fit(d, col, extra=t0)
            pl.append(thp[-1] * 100 if thp is not None else np.nan)
        pl = np.array(pl)
        for t0, v in zip(dates, pl):
            prow.append(dict(model=name, date=str(pd.to_datetime(t0, unit='s'))[:10], step_pct=v))
        sd = float(np.nanstd(pl))
        mu = float(np.nanmean(pl))
        for j, q in enumerate(SP_PN):
            if q in ('sin', 'cos', 'epet', 'efer', 'em7', 'ramp'):
                continue
            v = th[j] * 100
            t = d.t.values
            tq = TQ[{'pet': 'pet2021', 'fer': 'fer2022', 'm7': 'm7_2024'}.get(q, q)]
            rows.append(dict(model=name, quantity=('eps_pct' if col == 'eps0' else 'dlnVpVs_pct'), quake=q,
                             val=v, se_formal=se[j] * 100, placebo_mean=mu, placebo_sd=sd,
                             z_placebo=(v - mu) / sd if np.isfinite(v) else np.nan, pctl=pctl(v, pl),
                             det2sig=2 * sd, n_placebo=int(np.isfinite(pl).sum()),
                             n_stacks_2yr_before=int(((t < tq) & (t > tq - 2 * YR)).sum()),
                             n_stacks_2yr_after=int(((t >= tq) & (t < tq + 2 * YR)).sum()),
                             n_fam=int(d.F.nunique())))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/sp.csv', index=False)
    pd.DataFrame(prow).to_csv(f'{OUT}/sp_placebo.csv', index=False)
    print(R.round(3).to_string())


# ======================================================================================== tides
def stage_tides(a):
    import sci_tides as st
    d, off = st.load()
    units = st.decluster(d[d.meancc >= st.MEANCC_THR])
    yr = pd.Timedelta(days=365)

    def comp(t0, c, B=1000, M=2000):
        tq = pd.Timestamp(t0, unit='s') if not isinstance(t0, pd.Timestamp) else t0
        db, da = st.win(units, tq - yr, tq), st.win(units, tq, tq + yr)
        return st.compare(st.tidal_block(db, c, B=B, M=M), st.tidal_block(da, c, B=B, M=M), '', '', '')

    rows = []
    dates = placebo_dates(120, 404, lo='2009-01-20', hi='2024-11-01')
    pl = []
    for t0 in dates:
        r = comp(pd.to_datetime(t0, unit='s'), 'M2', B=100, M=100)
        pl.append(dict(date=str(pd.to_datetime(t0, unit='s'))[:10], dA=r['dA'], ratio=r['ratio'],
                       A_before=r['A_before'], n_fam=r['n_fam']))
    pl = pd.DataFrame(pl)
    pl.to_csv(f'{OUT}/tides_placebo.csv', index=False)
    for k, v in Q.items():
        for c in ('M2', 'O1'):
            r = comp(pd.Timestamp(v), c)
            row = dict(quake=k, const=c, n_fam=r['n_fam'], n_before=r['n_before'], n_after=r['n_after'],
                       A_before=r['A_before'], A_after=r['A_after'], dA=r['dA'], dA_se_boot=r['dA_se'],
                       ratio=r['ratio'], ratio_lo=r['ratio_lo'], ratio_hi=r['ratio_hi'])
            if c == 'M2':
                row.update(dA_pctl=pctl(r['dA'], pl.dA), ratio_pctl=pctl(r['ratio'], pl.ratio),
                           dA_placebo_sd=float(pl.dA.std()), ratio_placebo_sd=float(pl.ratio.std()),
                           det2sig_frac_drop=float(2 * pl.ratio.std()))
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/tides.csv', index=False)


# ======================================================================================== amplitude
def stage_amp(a):
    e = pd.read_csv('data/sci/moments/event_amplitudes.csv.gz', usecols=['fam', 'time', 'A', 'qc_bad', 'above_T'])
    e = e[(~e.qc_bad) & e.above_T & (e.A > 0)].copy()
    e['ts'] = (pd.to_datetime(e.time, format='mixed') - pd.Timestamp('1970-01-01')).dt.total_seconds()
    e['la'] = np.log10(e.A)
    fams = {F: (np.sort(g.ts.values), g.sort_values('ts').la.values) for F, g in e.groupby('fam')}
    dates = placebo_dates(300, 505, lo='2009-01-15', hi='2025-06-01')
    tests = {'step180': (0, 180, -180, 0), 'first30': (0, 30, -365, 0)}
    nmins = {'step180': 15, 'first30': 5}

    def stat(t, la, t0, w, nmin=15):
        a0, a1, b0, b1 = w
        A = la[(t >= t0 + a0 * DAY) & (t < t0 + a1 * DAY)]
        B = la[(t >= t0 + b0 * DAY) & (t < t0 + b1 * DAY)]
        if len(A) < nmin or len(B) < nmin:
            return np.nan
        return A.mean() - B.mean()

    rows = []
    for tn, w in tests.items():
        nm = nmins[tn]
        P = {F: np.array([stat(t, la, d, w, nm) for d in dates]) for F, (t, la) in fams.items()}
        for k in Q:
            vals = {F: stat(t, la, TQ[k], w, nm) for F, (t, la) in fams.items()}
            out = pool_inv(vals, P, minp=100)
            if out is None:
                rows.append(dict(test=tn, quake=k, n_units=0))
                continue
            o, Pk = out
            rows.append(dict(test=tn, quake=k, **o, ratio=10 ** o['value'],
                             n_fam_with_data=int(np.isfinite(list(vals.values())).sum())))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/amp.csv', index=False)
    print(R.round(4).to_string())


# ======================================================================================== spectra
def spec_job(F):
    import sci_source_spectra as ss
    cat = ss.load_catalog()
    fd = ss.load_family(F, cat, ss.meta_epochs())
    rng = np.random.default_rng(900 + F)
    out = []
    for k in NEW:
        A, B = ss.step_sets(fd, TQ[k])
        r = ss.compare(fd, A * 1.0, B * 1.0, 100, rng)
        if r is not None:
            out.append(dict(F=F, quake=k, **ss.flat(r)))
    return out


def stage_spec(a):
    import sci_source_spectra as ss
    fams = sorted(int(re.search(r'fam(\d+)_mf', f).group(1)) for f in glob.glob(f'{ss.OUT}/raw/fam*_mf.h5'))
    rows = []
    with Pool(a.procs) as pool:
        for out in pool.imap_unordered(spec_job, fams):
            rows += out
            print(len(rows), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/spec_per_family.csv', index=False)
    plc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{ss.OUT}/fam/fam*_placebo.csv'))], ignore_index=True)
    plc = plc[(plc.test == 'step') & (plc.variant == 'raw')]
    for k in NEW:
        plc = plc[np.abs(plc.date - TQ[k]) > 180 * DAY]
    pooled = []
    for k in NEW:
        Rk = R[R.quake == k]
        for stat in ss.STATS:
            o = ss.pool_one(Rk, plc, stat)
            if o is not None:
                o['pctl'] = np.nan
                pooled.append(dict(quake=k, **o))
    P = pd.DataFrame(pooled)
    # percentile vs pooled placebo distribution (recompute like pool_one)
    for i, r in P.iterrows():
        Rk = R[(R.quake == r.quake) & np.isfinite(R[r.stat])]
        pv = plc.pivot_table(index='k', columns='F', values=r.stat)
        w = {F: 1 / max(np.nanvar(pv[F]), 1e-8) for F in Rk.F if F in pv and pv[F].count() >= 20}
        M = pv[list(w)].values
        W = np.array([w[F] for F in w])[None] * np.isfinite(M)
        Pk = np.nansum(np.nan_to_num(M) * W, 1) / np.maximum(W.sum(1), 1e-12)
        Pk = Pk[W.sum(1) / sum(w.values()) > 0.6]
        P.loc[i, 'pctl'] = pctl(r.value, Pk)
        P.loc[i, 'det2sig'] = 2 * r.placebo_sd
    P.to_csv(f'{OUT}/spec_pooled.csv', index=False)
    print(P[['quake', 'stat', 'value', 'placebo_sd', 'z', 'pctl', 'n_fam', 'n_A', 'n_B']].round(4).to_string())


# ======================================================================================== summary + figs
def stage_figs(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qn = {'eur2010': '2010 M6.5', 'mend2014': '2014 M6.8', 'pet2021': 'Petrolia', 'fer2022': 'Ferndale',
          'm7_2024': 'M7', 'pet': 'Petrolia', 'fer': 'Ferndale', 'm7': 'M7'}
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    # dose
    D = pd.read_csv(f'{OUT}/dose.csv')
    R = pd.read_csv(f'{OUT}/rate.csv')
    R = R[R.subset == 'meancc0.137'].set_index('quake')
    x = D.dp.values
    y = R.reindex(D.quake.str.replace('M6.2', 'M6.2')).r30.values
    ax[0, 0].errorbar(x, y, xerr=[x - D.dp_ens_lo, D.dp_ens_hi - x], fmt='o', color='#2a78d6', ecolor='#bbb')
    for xi, yi, l in zip(x, y, D.quake):
        ax[0, 0].annotate(l, (xi, yi), fontsize=7, xytext=(3, 3), textcoords='offset points',
                          color='#eb6834' if ('2010-01' in l or '2014' in l) else 'k')
    ax[0, 0].axvline(0, color='k', lw=0.5)
    ax[0, 0].set_xlabel('undrained dp at LFEs (kPa, B=0.7)')
    ax[0, 0].set_ylabel('LFE rate ratio, 30 d after / 365 d before')
    ax[0, 0].set_title('Dose-response (meancc>=0.137 units)')
    # rate pctl table
    R2 = pd.read_csv(f'{OUT}/rate.csv')
    R2 = R2[R2.subset == 'meancc0.137']
    ax[0, 1].barh(R2.quake, R2.r7_pctl, color='#2a78d6', alpha=0.7, label='r7 pctl')
    ax[0, 1].plot(R2.gap30_h_pctl, R2.quake, 'o', color='#eb6834', label='max gap pctl')
    ax[0, 1].axvline(2.5, color='k', lw=0.5)
    ax[0, 1].axvline(97.5, color='k', lw=0.5)
    ax[0, 1].set_xlabel('placebo percentile')
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].tick_params(axis='y', labelsize=7)
    ax[0, 1].set_title('Rate / silence vs 500 placebo dates')
    # dvv pooled
    P = pd.read_csv(f'{OUT}/dvv_pooled.csv')
    for i, g in enumerate(['surface', 'borehole']):
        s = P[P.group == g]
        xx = np.arange(len(s)) + (i - 0.5) * 0.3
        ax[0, 2].errorbar(xx, s.value - s.placebo_mean, yerr=2 * s.placebo_sd, fmt='o', label=g,
                          color=['#2a78d6', '#1baf7a'][i])
    ax[0, 2].set_xticks(np.arange(len(s)))
    ax[0, 2].set_xticklabels([qn[q] for q in s.quake], fontsize=7)
    ax[0, 2].axhline(0, color='k', lw=0.5)
    ax[0, 2].set_ylabel('full-coda dv/v step (%) +/- 2 placebo SD')
    ax[0, 2].legend(fontsize=7)
    ax[0, 2].set_title('LFE coda dv/v steps (pooled families)')
    # S-P
    S = pd.read_csv(f'{OUT}/sp.csv')
    for i, m in enumerate(['sph10', 'slab5SP']):
        s = S[S.model == m]
        xx = np.arange(len(s)) + (i - 0.5) * 0.3
        ax[1, 0].errorbar(xx, s.val, yerr=2 * s.placebo_sd, fmt='o', label=m + (' eps' if m == 'sph10' else ' dlnVp/Vs'))
    ax[1, 0].set_xticks(np.arange(len(s)))
    ax[1, 0].set_xticklabels([qn[q] for q in s.quake], fontsize=7)
    ax[1, 0].axhline(0, color='k', lw=0.5)
    ax[1, 0].set_ylabel('% (+/- 2 placebo SD)')
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title('Source-region S-P dv/v and Vp/Vs steps')
    # tides
    T = pd.read_csv(f'{OUT}/tides.csv')
    t = T[T.const == 'M2']
    ax[1, 1].errorbar(np.arange(len(t)), t.ratio, yerr=[t.ratio - t.ratio_lo, t.ratio_hi - t.ratio], fmt='o',
                      color='#2a78d6')
    pl = pd.read_csv(f'{OUT}/tides_placebo.csv')
    ax[1, 1].axhspan(np.percentile(pl.ratio, 2.5), np.percentile(pl.ratio, 97.5), color='#ddd', label='placebo 95%')
    ax[1, 1].set_xticks(np.arange(len(t)))
    ax[1, 1].set_xticklabels([qn[q] for q in t.quake], fontsize=7)
    ax[1, 1].axhline(1, color='k', lw=0.5)
    ax[1, 1].set_ylabel('M2 modulation ratio (1 yr after / before)')
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title('Tidal sensitivity')
    # amplitude + spectra
    A = pd.read_csv(f'{OUT}/amp.csv')
    a1 = A[A.test == 'step180']
    ax[1, 2].errorbar(np.arange(len(a1)) - 0.1, a1.value, yerr=2 * a1.placebo_sd, fmt='o', label='log10 A (+/-180 d)')
    Sp = pd.read_csv(f'{OUT}/spec_pooled.csv')
    s1 = Sp[Sp.stat == 'HFI_all']
    xs = [list(a1.quake).index(q) + 0.1 for q in s1.quake]
    ax[1, 2].errorbar(xs, s1.value, yerr=2 * s1.placebo_sd, fmt='s', label='HFI spectral index (+/-180 d)')
    ax[1, 2].set_xticks(np.arange(len(a1)))
    ax[1, 2].set_xticklabels([qn[q] for q in a1.quake], fontsize=7)
    ax[1, 2].axhline(0, color='k', lw=0.5)
    ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_title('Amplitude and spectra (+/- 2 placebo SD)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_extras_quakes2.png', dpi=130)
    print('fig written')


def stage_summary(a):
    D = json.load(open(f'{OUT}/dose_summary.json'))
    R = pd.read_csv(f'{OUT}/rate.csv')
    V = pd.read_csv(f'{OUT}/dvv_pooled.csv')
    S = pd.read_csv(f'{OUT}/sp.csv')
    T = pd.read_csv(f'{OUT}/tides.csv')
    A = pd.read_csv(f'{OUT}/amp.csv')
    Sp = pd.read_csv(f'{OUT}/spec_pooled.csv')
    rk = {'eur2010': '2010-01-10 M6.5', 'mend2014': '2014-03-10 M6.8'}
    out = dict(dose={k: v for k, v in D.items() if not isinstance(v, dict)}, quakes={})
    for q in NEW:
        o = dict(dose={k: D[rk[q]][k] for k in ('dist_km', 'dtau', 'dsn', 'dCFS04', 'dp', 'dp_ens_lo', 'dp_ens_hi',
                                                  'pgv_rrup')})
        o['rate'] = R[R.quake == rk[q]].drop(columns='quake').set_index('subset').to_dict('index')
        o['dvv'] = V[V.quake == q][['group', 'value', 'placebo_mean', 'placebo_sd', 'z', 'pctl', 'winter_z',
                                     'det2sig', 'det2sig_corr']].set_index('group').to_dict('index')
        o['sp'] = S[S.quake == q][['model', 'val', 'placebo_sd', 'z_placebo', 'pctl', 'det2sig']].set_index(
            'model').to_dict('index')
        o['tides'] = T[(T.quake == q)][['const', 'ratio', 'ratio_lo', 'ratio_hi', 'dA_pctl']].set_index(
            'const').to_dict('index')
        o['amp'] = A[A.quake == q][['test', 'value', 'placebo_sd', 'z', 'pctl', 'det2sig']].set_index(
            'test').to_dict('index')
        o['spec'] = Sp[(Sp.quake == q) & Sp.stat.isin(['LO_all', 'MID_all', 'HFI_all', 'dD_all'])][
            ['stat', 'value', 'placebo_sd', 'z', 'pctl', 'det2sig']].set_index('stat').to_dict('index')
        out['quakes'][q] = o
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    print('summary written')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    st = ['dose', 'rate', 'sp', 'tides', 'amp', 'dvv', 'spec', 'figs', 'summary'] if a.stage == 'all' else a.stage.split(',')
    for s in st:
        t0 = time.time()
        print(f'=== {s}', flush=True)
        if s == 'dvv_analyze':
            analyze_dvv()
        else:
            globals()[f'stage_{s}'](a)
        print(f'=== {s} done {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
