#!/usr/bin/env python3
"""sci_cascades2: cascade statistics through time, family interaction network, post-M7 restart,
and remote dynamic triggering for the 27 MTJ LFE families (Shelly 2026 v2 catalog).

Builds on scripts/burst_cascades.py (same burst / episode definitions: family bursts = gaps <= 2 h,
network episodes = onsets clustered within 6 h with >= 6 families; robust planar onset fit + position
permutation test).

Constant-completeness catalog ("thin"): the catalog's per-event detection floor in mean channel CC is
thresh/NSta; its 99th percentile is <= 0.13 in every year 2009-2026 (2008: 0.19, sparse network), so
we keep events with meancc >= 0.13 and drop 2008.  Variants 'all' (full catalog, 2009+) and 'top15'
(15 most productive families, full catalog, 2009+) are robustness checks.

Usage: python scripts/sci_cascades2.py [task ...]   tasks: t1 t2 t3 t4 fig  (default all)
Outputs: data/sci/cascades2/*.csv, summary_*.json, figures/sci_cascades2_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = 'data/sci/cascades2'
os.makedirs(OUT, exist_ok=True)
GAP, EP_GAP, MINFAM, NPERM = 2 * 3600, 6 * 3600, 6, 200
MEANCC_CUT, T_START = 0.13, '2009-01-01'
EPOCH = pd.Timestamp('1970-01-01')
QUAKES = {'Petrolia': '2021-12-20 20:10:00', 'Ferndale': '2022-12-20 10:34:00', 'M7': '2024-12-05 18:44:00'}
STRIKE = 293.0
UA = np.array([np.sin(np.radians(STRIKE)), np.cos(np.radians(STRIKE))])     # along strike (toward WNW)
UC = np.array([np.sin(np.radians(STRIKE + 90)), np.cos(np.radians(STRIKE + 90))])
LAT0, LON0 = 40.06, -123.625
M2 = 12.4206012 * 3600
YR = 365.25 * 86400


def ts(s):
    return (pd.Timestamp(s) - EPOCH).total_seconds()


TQ = {k: ts(v) for k, v in QUAKES.items()}
T_END = None


# ---------------------------------------------------------------- data
def load():
    global T_END
    d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
    d['t'] = (pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) - EPOCH).dt.total_seconds() + d.s_of_day
    d = d.sort_values('t').reset_index(drop=True)
    T_END = d.t.max()
    P = {}
    for F in range(1, 28):
        j = json.load(open(f'data/locations/fam{F}_3d.json'))
        x, y = (j['lon'] - LON0) * 111.19 * np.cos(np.radians(LAT0)), (j['lat'] - LAT0) * 111.19
        P[F] = dict(x=x, y=y, z=j['depth_km'], a=x * UA[0] + y * UA[1], c=x * UC[0] + y * UC[1])
    P = pd.DataFrame(P).T
    P.index.name = 'fam'
    return d, P


def catalog_variant(d, v):
    d = d[d.t >= ts(T_START)]
    if v == 'thin':
        return d[d.meancc >= MEANCC_CUT]
    if v == 'top15':
        top = d.groupby('idnum').size().sort_values(ascending=False).index[:15]
        return d[d.idnum.isin(top)]
    return d


def make_bursts(d, gap=GAP):
    rows = []
    for F, g in d.groupby('idnum'):
        t = np.sort(g.t.values)
        s = np.r_[0, np.where(np.diff(t) > gap)[0] + 1]
        e = np.r_[s[1:], len(t)]
        rows.append(pd.DataFrame({'fam': F, 'onset': t[s], 'end': t[e - 1], 'n': e - s}))
    return pd.concat(rows).sort_values('onset', kind='stable').reset_index(drop=True)


def fit_batch(t, X, iters=5):
    """Huber-IRLS planar fit t = t0 + s.X for a batch of position sets X (P,n,2), same as burst_cascades.fit."""
    Pn, n, _ = X.shape
    A = np.concatenate([np.ones((Pn, n, 1)), X], axis=2)
    w = np.ones((Pn, n))
    for _ in range(iters):
        Aw = A * w[..., None]
        m = np.linalg.solve(np.einsum('pni,pnj->pij', Aw, Aw) + 1e-9 * np.eye(3),
                            np.einsum('pni,pn->pi', Aw, t[None] * w)[..., None])[..., 0]
        r = t[None] - np.einsum('pni,pi->pn', A, m)
        s = 1.4826 * np.median(np.abs(r), axis=1, keepdims=True) + 1.0
        ar = np.maximum(np.abs(r), 1e-12)
        w = np.sqrt(np.where(ar / s <= 2, 1.0, 2 * s / ar))
    r = t[None] - np.einsum('pni,pi->pn', A, m)
    w2 = w ** 2
    tm = (w2 * t[None]).sum(1) / w2.sum(1)
    rss = ((w * r) ** 2).sum(1)
    tss = ((w * (t[None] - tm[:, None])) ** 2).sum(1)
    return m, np.where(tss > 0, 1 - rss / np.where(tss > 0, tss, 1), np.nan)


def make_episodes(B, P, rng, nperm=NPERM):
    B = B.copy()
    B['ep'] = np.r_[0, np.cumsum(np.diff(B.onset.values) > EP_GAP)]
    rows, inep = [], np.zeros(len(B), bool)
    for e, g in B.groupby('ep', sort=False):
        g1 = g.drop_duplicates('fam')
        if len(g1) < MINFAM:
            continue
        inep[g.index.values] = True
        n = len(g1)
        X = P.loc[g1.fam.values, ['x', 'y']].values.astype(float)
        t = (g1.onset.values - g1.onset.values[0]) / 3600.0
        idx = np.vstack([np.arange(n), np.argsort(rng.random((nperm, n)), axis=1)])
        m, r2 = fit_batch(t, X[idx])
        p = (np.sum(r2[1:] >= r2[0]) + 1) / (nperm + 1)
        sx, sy = m[0, 1], m[0, 2]
        smag = np.hypot(sx, sy)
        a = P.loc[g1.fam.values, 'a'].values.astype(float)
        rows.append(dict(ep=e, t0=g1.onset.values[0], n_fam=n, dur_h=t.max(), n_events=int(g.n.sum()),
                         sx_h_per_km=sx, sy_h_per_km=sy, s_along_h_per_km=sx * UA[0] + sy * UA[1],
                         speed_km_per_day=24 / smag if smag > 0 else np.inf,
                         azimuth_deg=np.degrees(np.arctan2(sx, sy)) % 360, r2=r2[0], p_perm=p,
                         first_fam=int(g1.fam.values[0]), lead_a_rel=a[0] - a.mean(),
                         lead_end=float(abs(a[0] - np.median(a)) >= np.quantile(np.abs(a - np.median(a)), 2 / 3))))
    E = pd.DataFrame(rows)
    E['sig'] = E.p_perm < 0.05
    E['west_led_fit'] = E.s_along_h_per_km < 0      # onsets later toward ESE => started in WNW
    E['west_led_leader'] = E.lead_a_rel > 0          # first family is WNW of the participants' centroid
    B['in_ep'] = inep
    return E, B


# ---------------------------------------------------------------- task 1
def ep_metrics(E, B, span_s):
    yrs = span_s / YR
    sig = E[E.sig]
    return dict(ep_rate_per_yr=len(E) / yrs,
                ep_per_1000_bursts=1000 * len(E) / max(len(B), 1),
                frac_bursts_in_ep=B.in_ep.mean() if len(B) else np.nan,
                frac_sig=E.sig.mean() if len(E) else np.nan,
                med_log10_speed_sig=np.log10(sig.speed_km_per_day).median() if len(sig) else np.nan,
                frac_westled_sig=sig.west_led_fit.mean() if len(sig) else np.nan,
                frac_westled_leader=E.west_led_leader.mean() if len(E) else np.nan,
                frac_endled=E.lead_end.mean() if len(E) else np.nan,
                mean_nfam=E.n_fam.mean() if len(E) else np.nan,
                med_dur_h=E.dur_h.median() if len(E) else np.nan,
                med_events_per_ep=E.n_events.median() if len(E) else np.nan)


def window(E, B, t1, t2):
    return E[(E.t0 >= t1) & (E.t0 < t2)], B[(B.onset >= t1) & (B.onset < t2)]


def block_resample(E, B, t1, t2, blocks, rng):
    """Monthly-block bootstrap: resample 12 blocks of the window with replacement."""
    L = (t2 - t1) / 12
    be = ((E.t0.values - t1) // L).astype(int)
    bb = ((B.onset.values - t1) // L).astype(int)
    ie = np.concatenate([np.where(be == k)[0] for k in blocks])
    ib = np.concatenate([np.where(bb == k)[0] for k in blocks])
    return E.iloc[ie], B.iloc[ib]


def quake_tests(E, B, rng, nboot=500, nplac=300):
    out = []
    keys = list(ep_metrics(E.iloc[:5], B.iloc[:5], YR).keys())

    def diff(tq):
        e1, b1 = window(E, B, tq - YR, tq)
        e2, b2 = window(E, B, tq, tq + YR)
        m1, m2 = ep_metrics(e1, b1, YR), ep_metrics(e2, b2, YR)
        return m1, m2
    plac_t = rng.uniform(ts('2010-01-01'), ts('2025-06-30'), nplac)
    plac = pd.DataFrame([{k: b[k] - a[k] for k in keys} for a, b in (diff(t) for t in plac_t)])
    for q, tq in TQ.items():
        m1, m2 = diff(tq)
        e1, b1 = window(E, B, tq - YR, tq)
        e2, b2 = window(E, B, tq, tq + YR)
        bt = []
        for _ in range(nboot):
            a = ep_metrics(*block_resample(e1, b1, tq - YR, tq, rng.integers(0, 12, 12), rng), YR)
            b = ep_metrics(*block_resample(e2, b2, tq, tq + YR, rng.integers(0, 12, 12), rng), YR)
            bt.append({k: b[k] - a[k] for k in keys})
        bt = pd.DataFrame(bt)
        # block-label permutation: 24 monthly blocks shuffled between pre and post
        eall, ball = window(E, B, tq - YR, tq + YR)
        pm = []
        for _ in range(nboot):
            perm = rng.permutation(24)
            L = YR / 12
            be = ((eall.t0.values - (tq - YR)) // L).astype(int)
            bb = ((ball.onset.values - (tq - YR)) // L).astype(int)
            pre_b = perm[:12]
            ea, ba = eall[np.isin(be, pre_b)], ball[np.isin(bb, pre_b)]
            eb, bb_ = eall[~np.isin(be, pre_b)], ball[~np.isin(bb, pre_b)]
            ma, mb = ep_metrics(ea, ba, YR), ep_metrics(eb, bb_, YR)
            pm.append({k: mb[k] - ma[k] for k in keys})
        pm = pd.DataFrame(pm)
        for k in keys:
            dobs = m2[k] - m1[k]
            out.append(dict(quake=q, metric=k, pre=m1[k], post=m2[k], diff=dobs,
                            ci_lo=bt[k].quantile(.025), ci_hi=bt[k].quantile(.975),
                            p_blockperm=float((np.sum(np.abs(pm[k].dropna()) >= abs(dobs)) + 1) / (pm[k].notna().sum() + 1)),
                            p_placebo=float((np.sum(np.abs(plac[k].dropna() - plac[k].median()) >= abs(dobs - plac[k].median())) + 1)
                                            / (plac[k].notna().sum() + 1)),
                            n_pre=len(e1), n_post=len(e2)))
    return pd.DataFrame(out)


def fam_recurrence(d, rng, gap=GAP, nplac=300):
    """Item 5: per-family burst recurrence, events/burst, burstiness (CV of inter-burst times) 1 yr pre/post."""
    B = make_bursts(d, gap)
    byf = {F: g for F, g in B.groupby('fam')}

    def fm(g, t1, t2):
        g = g[(g.onset >= t1) & (g.onset < t2)]
        if len(g) < 5:
            return dict(rate=len(g), rec_h=np.nan, ev=np.nan, cv=np.nan)
        dt = np.diff(g.onset.values) / 3600
        return dict(rate=len(g), rec_h=np.median(dt), ev=g.n.mean(), cv=dt.std() / dt.mean())

    def logratios(tq):
        rows = []
        for F, g in byf.items():
            a, b = fm(g, tq - YR, tq), fm(g, tq, tq + YR)
            rows.append({k: np.log10(b[k] / a[k]) if (a[k] and b[k] and np.isfinite(a[k]) and np.isfinite(b[k]) and a[k] > 0 and b[k] > 0) else np.nan
                         for k in a} | {'fam': F, **{f'pre_{k}': a[k] for k in a}, **{f'post_{k}': b[k] for k in a}})
        return pd.DataFrame(rows)
    plac_t = rng.uniform(ts('2010-01-01'), ts('2025-06-30'), nplac)
    plac = pd.DataFrame([logratios(t)[['rate', 'rec_h', 'ev', 'cv']].median() for t in plac_t])
    res, per = [], []
    for q, tq in TQ.items():
        L = logratios(tq)
        L['quake'] = q
        per.append(L)
        for k in ['rate', 'rec_h', 'ev', 'cv']:
            v = L[k].dropna()
            med = v.median()
            boot = [np.median(rng.choice(v.values, len(v))) for _ in range(2000)]
            res.append(dict(quake=q, gap_h=gap / 3600, metric=k, n_fam=len(v), median_log10_ratio=med,
                            ci_lo=np.quantile(boot, .025), ci_hi=np.quantile(boot, .975),
                            frac_fam_up=float((v > 0).mean()),
                            p_wilcoxon=float(stats.wilcoxon(v).pvalue) if len(v) > 5 else np.nan,
                            p_placebo=float((np.sum(np.abs(plac[k] - plac[k].median()) >= abs(med - plac[k].median())) + 1) / (len(plac) + 1)),
                            placebo_median=float(plac[k].median()), placebo_sd=float(plac[k].std())))
    return pd.DataFrame(res), pd.concat(per)


def task1(d, P, summ):
    rng = np.random.default_rng(11)
    allq, yearly, recs = [], [], []
    for v in ['thin', 'all', 'top15']:
        dv = catalog_variant(d, v)
        B = make_bursts(dv)
        E, B = make_episodes(B, P, rng)
        E['t0_utc'] = pd.to_datetime(E.t0, unit='s')
        E.to_csv(f'{OUT}/episodes_{v}.csv', index=False)
        for y in range(2009, 2027):
            t1, t2 = ts(f'{y}-01-01'), min(ts(f'{y + 1}-01-01'), T_END)
            e, b = window(E, B, t1, t2)
            m = ep_metrics(e, b, t2 - t1)
            m.update(variant=v, year=y, n_episodes=len(e), n_bursts=len(b),
                     n_events=int(((dv.t >= t1) & (dv.t < t2)).sum()))
            yearly.append(m)
        Q = quake_tests(E, B, rng)
        Q['variant'] = v
        allq.append(Q)
        R, per = fam_recurrence(dv, rng)
        R['variant'] = v
        recs.append(R)
        if v == 'thin':
            R24, per24 = fam_recurrence(dv, rng, gap=24 * 3600)
            R24['variant'] = v
            recs.append(R24)
            per.to_csv(f'{OUT}/family_recurrence_perfamily_thin.csv', index=False)
        sig = E[E.sig]
        summ[f't1_{v}'] = dict(n_episodes=len(E), frac_sig=float(E.sig.mean()),
                               median_speed_sig_km_day=float(sig.speed_km_per_day.median()),
                               speed_iqr=[float(sig.speed_km_per_day.quantile(.25)), float(sig.speed_km_per_day.quantile(.75))],
                               frac_westled_sig=float(sig.west_led_fit.mean()), n_sig=len(sig))
        print(v, summ[f't1_{v}'], flush=True)
    pd.DataFrame(yearly).to_csv(f'{OUT}/yearly_episode_stats.csv', index=False)
    Q = pd.concat(allq)
    Q.to_csv(f'{OUT}/quake_window_tests.csv', index=False)
    R = pd.concat(recs)
    R.to_csv(f'{OUT}/family_recurrence_tests.csv', index=False)
    pd.set_option('display.width', 250)
    print(Q[Q.variant == 'thin'].round(3).to_string())
    print(R.round(3).to_string())
    summ['t1_quake_tests_thin'] = Q[Q.variant == 'thin'].round(4).to_dict('records')


# ---------------------------------------------------------------- task 2
def tidal_shifts(lo_d=3, hi_d=60, tod_tol_h=1.0):
    """Shifts = k * M2 period whose time-of-day offset is within +-tod_tol_h (keeps tidal + diurnal phase)."""
    out = []
    for k in range(1, int(hi_d * 86400 / M2) + 1):
        s = k * M2
        tod = (s / 3600) % 24
        if s >= lo_d * 86400 and min(tod, 24 - tod) <= tod_tol_h:
            out += [s, -s]
    return np.array(out)


SHIFTS = tidal_shifts()
DTS = {'10min': 600, '1h': 3600, '6h': 21600}
MINLAG = 15.0


def count_follow(ti, tj, dt):
    """number of i onsets with a j onset in [t+MINLAG, t+dt]"""
    k = np.searchsorted(tj, ti + MINLAG, 'left')
    ok = k < len(tj)
    return int(np.sum(tj[np.minimum(k, len(tj) - 1)][ok] <= ti[ok] + dt))


def network(on, t1, t2, rng, K=100):
    """on: dict fam -> sorted onset times within [t1,t2). returns obs/null arrays (F,F,len(DTS))."""
    fams = sorted(on)
    nF, nD = len(fams), len(DTS)
    obs = np.zeros((nF, nF, nD))
    null = np.zeros((K, nF, nF, nD))
    L = t2 - t1

    def counts(o):
        c = np.zeros((nF, nF, nD))
        for a, i in enumerate(fams):
            for b, j in enumerate(fams):
                if a != b and len(o[i]) and len(o[j]):
                    for q, dt in enumerate(DTS.values()):
                        c[a, b, q] = count_follow(o[i], o[j], dt)
        return c
    obs = counts(on)
    for k in range(K):
        sh = {F: np.sort(t1 + (on[F] - t1 + rng.choice(SHIFTS)) % L) for F in fams}
        null[k] = counts(sh)
    return fams, obs, null


def lead_sign(on, fams, dt=3600):
    """nearest-neighbour lag sign test: for each i onset, nearest j onset with MINLAG<=|lag|<=dt."""
    nF = len(fams)
    lead = np.zeros((nF, nF))
    for a, i in enumerate(fams):
        for b, j in enumerate(fams):
            if a >= b or not len(on[i]) or not len(on[j]):
                continue
            ti, tj = on[i], on[j]
            k = np.searchsorted(tj, ti)
            lags = []
            for kk in (k - 1, k):
                ok = (kk >= 0) & (kk < len(tj))
                l = np.full(len(ti), np.nan)
                l[ok] = tj[kk[ok]] - ti[ok]
                lags.append(l)
            l = np.where(np.abs(np.nan_to_num(lags[0], nan=1e18)) < np.abs(np.nan_to_num(lags[1], nan=1e18)), lags[0], lags[1])
            l = l[np.isfinite(l) & (np.abs(l) >= MINLAG) & (np.abs(l) <= dt)]
            lead[a, b], lead[b, a] = np.sum(l > 0), np.sum(l < 0)   # i leads j / j leads i
    return lead


def pair_table(fams, obs, null, P):
    rows = []
    mu, sd = null.mean(0), null.std(0)
    for a, i in enumerate(fams):
        for b, j in enumerate(fams):
            if a == b:
                continue
            dvec = P.loc[j, ['x', 'y']].values.astype(float) - P.loc[i, ['x', 'y']].values.astype(float)
            r = dict(i=i, j=j, dist_h_km=float(np.hypot(*dvec)), d_along_km=float(abs(dvec @ UA)),
                     d_across_km=float(abs(dvec @ UC)), dz_km=float(abs(P.loc[j, 'z'] - P.loc[i, 'z'])))
            for q, k in enumerate(DTS):
                r[f'N_{k}'] = obs[a, b, q]
                r[f'null_{k}'] = mu[a, b, q]
                r[f'G_{k}'] = (obs[a, b, q] + 0.5) / (mu[a, b, q] + 0.5)
                r[f'z_{k}'] = (obs[a, b, q] - mu[a, b, q]) / max(sd[a, b, q], 1.0)
            rows.append(r)
    T = pd.DataFrame(rows)
    # symmetric gain per unordered pair
    for k in DTS:
        key = T[['i', 'j']].min(axis=1).astype(str) + '-' + T[['i', 'j']].max(axis=1).astype(str)
        Ns = T.groupby(key)[f'N_{k}'].transform('sum')
        Ms = T.groupby(key)[f'null_{k}'].transform('sum')
        T[f'Gsym_{k}'] = (Ns + 0.5) / (Ms + 0.5)
    return T


def distance_tests(T, P, rng, k, nperm=2000):
    U = T[T.i < T.j].copy()
    y = np.log2(U[f'Gsym_{k}'].values)
    fams = sorted(P.index)
    pos = P.loc[fams, ['x', 'y', 'z']].values.astype(float)
    fi = np.searchsorted(fams, U.i.values)
    fj = np.searchsorted(fams, U.j.values)

    def feats(pp):
        dv = pp[fj, :2] - pp[fi, :2]
        return np.hypot(dv[:, 0], dv[:, 1]), np.abs(dv @ UA), np.abs(dv @ UC), np.abs(pp[fj, 2] - pp[fi, 2])

    def stat(pp):
        dh, da, dc, dz = feats(pp)
        rho = stats.spearmanr(y, dh)[0]
        A = np.c_[np.ones(len(y)), da, dc, dz]
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        sl = np.polyfit(dh, y, 1)[0]
        return rho, sl, b[1], b[2], b[3]
    s0 = stat(pos)
    null = np.array([stat(pos[rng.permutation(len(pos))]) for _ in range(nperm)])
    # jackknife over families for SE of coefficients
    jk = []
    for f in fams:
        m = (U.i.values != f) & (U.j.values != f)
        dh, da, dc, dz = [v[m] for v in feats(pos)]
        A = np.c_[np.ones(m.sum()), da, dc, dz]
        b = np.linalg.lstsq(A, y[m], rcond=None)[0]
        jk.append([np.polyfit(dh, y[m], 1)[0], b[1], b[2], b[3], b[1] - b[2]])
    jk = np.array(jk)
    se = np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean(0)) ** 2).sum(0))
    return dict(dt=k, n_pairs=len(y), median_log2_Gsym=float(np.median(y)),
                spearman_logG_vs_dist=float(s0[0]),
                p_mantel=float((np.sum(null[:, 0] <= s0[0]) + 1) / (nperm + 1)),
                slope_log2G_per_km=float(s0[1]), slope_se_jk=float(se[0]),
                b_along=float(s0[2]), b_along_se=float(se[1]), b_across=float(s0[3]), b_across_se=float(se[2]),
                b_dz=float(s0[4]), b_dz_se=float(se[3]),
                along_minus_across=float(s0[2] - s0[3]), along_minus_across_se=float(se[4]),
                p_along_gt_across_perm=float((np.sum((null[:, 2] - null[:, 3]) >= (s0[2] - s0[3])) + 1) / (nperm + 1)))


def onsets_dict(B, t1, t2):
    return {F: np.sort(g.onset.values[(g.onset.values >= t1) & (g.onset.values < t2)]) for F, g in B.groupby('fam')}


def window_net_metrics(B, P, t1, t2, rng, K=30):
    on = onsets_dict(B, t1, t2)
    fams, obs, null = network(on, t1, t2, rng, K)
    T = pair_table(fams, obs, null, P)
    U = T[T.i < T.j]
    m = {}
    for k in DTS:
        y = np.log2(U[f'Gsym_{k}'])
        m[f'med_log2G_{k}'] = float(y.median())
        m[f'frac_z3_{k}'] = float((T[f'z_{k}'] > 3).mean())
        m[f'slope_{k}'] = float(np.polyfit(U.dist_h_km, y, 1)[0])
    return m, T


def task2(d, P, summ):
    rng = np.random.default_rng(22)
    dv = catalog_variant(d, 'thin')
    B = make_bursts(dv)
    t1, t2 = ts(T_START), T_END + 1
    on = onsets_dict(B, t1, t2)
    fams, obs, null = network(on, t1, t2, rng, K=100)
    T = pair_table(fams, obs, null, P)
    lead = lead_sign(on, fams, 3600)
    T['lead_ij_1h'] = [lead[fams.index(i), fams.index(j)] for i, j in zip(T.i, T.j)]
    T['lead_ji_1h'] = [lead[fams.index(j), fams.index(i)] for i, j in zip(T.i, T.j)]
    T['p_lead_binom'] = [stats.binomtest(int(a), int(a + b)).pvalue if a + b > 0 else 1.0 for a, b in zip(T.lead_ij_1h, T.lead_ji_1h)]
    # BH over unordered pairs
    U = T[T.i < T.j].copy()
    p = U.p_lead_binom.values
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    U['q_lead'] = q
    T = T.merge(U[['i', 'j', 'q_lead']], on=['i', 'j'], how='left')
    T.to_csv(f'{OUT}/network_pairs_thin_all.csv', index=False)
    dres = [distance_tests(T, P, rng, k) for k in DTS]
    # directed: net leadership per family (number of significant pairs led minus followed)
    sigU = U[U.q_lead < 0.05]
    net = pd.Series(0, index=fams)
    for _, r in sigU.iterrows():
        w = 1 if r.lead_ij_1h > r.lead_ji_1h else -1
        net[r.i] += w
        net[r.j] -= w
    lead_frac = pd.Series({f: U.loc[U.i == f, 'lead_ij_1h'].sum() + U.loc[U.j == f, 'lead_ji_1h'].sum() for f in fams}) / \
        pd.Series({f: U.loc[(U.i == f) | (U.j == f), ['lead_ij_1h', 'lead_ji_1h']].values.sum() for f in fams})
    a = P.loc[fams, 'a'].values.astype(float)
    summ['t2_network'] = dict(
        null='circular shifts by multiples of M2 (12.42 h) with time-of-day offset <=1 h, |shift| 3-60 d, K=100; lags <15 s excluded',
        distance_tests=dres,
        frac_pairs_z3={k: float((T[f'z_{k}'] > 3).mean()) for k in DTS},
        median_G={k: float(T[f'G_{k}'].median()) for k in DTS},
        n_pairs_sig_lead_q05=int(len(sigU)), n_pairs=int(len(U)),
        lead_frac_vs_along_spearman=[float(x) for x in stats.spearmanr(lead_frac.values, a)],
        lead_frac_vs_absalong_spearman=[float(x) for x in stats.spearmanr(lead_frac.values, np.abs(a - np.median(a)))],
        lead_frac=lead_frac.round(3).to_dict(), net_lead_degree=net.to_dict())
    print(json.dumps(summ['t2_network'], indent=1, default=float)[:4000], flush=True)
    # change after M7 vs year-to-year placebo
    rows = []
    for y in range(2009, 2026):
        m, _ = window_net_metrics(B, P, ts(f'{y}-01-01'), ts(f'{y + 1}-01-01'), rng)
        m.update(window=str(y), t1=ts(f'{y}-01-01'))
        rows.append(m)
    tq = TQ['M7']
    for lab, a1, a2 in [('pre_M7_1yr', tq - YR, tq), ('post_M7_1yr', tq, tq + YR), ('post_M7_all', tq, T_END + 1)]:
        m, Tw = window_net_metrics(B, P, a1, a2, rng, K=50)
        m.update(window=lab, t1=a1)
        rows.append(m)
        Tw.to_csv(f'{OUT}/network_pairs_thin_{lab}.csv', index=False)
    W = pd.DataFrame(rows)
    # placebo: differences between consecutive 1-yr windows at random dates
    pl = []
    for tp in rng.uniform(ts('2010-01-01'), ts('2024-06-30'), 40):
        ma, _ = window_net_metrics(B, P, tp - YR, tp, rng, K=20)
        mb, _ = window_net_metrics(B, P, tp, tp + YR, rng, K=20)
        pl.append({k: mb[k] - ma[k] for k in ma})
    pl = pd.DataFrame(pl)
    pre, post = W[W.window == 'pre_M7_1yr'].iloc[0], W[W.window == 'post_M7_1yr'].iloc[0]
    ch = {}
    for k in pl.columns:
        dd = post[k] - pre[k]
        ch[k] = dict(pre=float(pre[k]), post=float(post[k]), diff=float(dd), placebo_sd=float(pl[k].std()),
                     p_placebo=float((np.sum(np.abs(pl[k] - pl[k].median()) >= abs(dd - pl[k].median())) + 1) / (len(pl) + 1)))
    # pairwise matrix similarity pre vs post (Spearman of log Gsym 1h)
    Ta = pd.read_csv(f'{OUT}/network_pairs_thin_pre_M7_1yr.csv')
    Tb = pd.read_csv(f'{OUT}/network_pairs_thin_post_M7_1yr.csv')
    ch['pairwise_spearman_logGsym_1h_pre_vs_post'] = float(stats.spearmanr(np.log(Ta.Gsym_1h), np.log(Tb.Gsym_1h))[0])
    yy = W[W.window.str.len() == 4]
    cons = [stats.spearmanr(*[np.log(window_net_metrics(B, P, ts(f'{y}-01-01'), ts(f'{y + 1}-01-01'), rng, K=20)[1].Gsym_1h)
                              for y in (y0, y0 + 1)])[0] for y0 in range(2010, 2023, 3)]
    ch['pairwise_spearman_consecutive_years'] = [float(c) for c in cons]
    W.to_csv(f'{OUT}/network_by_window.csv', index=False)
    summ['t2_M7_change'] = ch
    print(json.dumps(ch, indent=1), flush=True)


# ---------------------------------------------------------------- task 3
def order_stats(tt, P, fams, rng, nperm=5000):
    """tt: times (days) per family. Robust planar fit + permutation, Spearman with along/|along| coordinate."""
    tt = np.asarray(tt, float)
    X = P.loc[fams, ['x', 'y']].values.astype(float)
    a = P.loc[fams, 'a'].values.astype(float)
    idx = np.vstack([np.arange(len(tt)), np.argsort(rng.random((nperm, len(tt))), axis=1)])
    # planar fit in days (scale term +1.0 in fit_batch is in time units; use hours for consistency)
    m, r2 = fit_batch(tt * 24, X[idx])
    sx, sy = m[0, 1], m[0, 2]
    smag = np.hypot(sx, sy)
    rho_a = stats.spearmanr(tt, a)[0]
    rho_abs = stats.spearmanr(tt, np.abs(a - np.median(a)))[0]
    rho_a_null = np.array([stats.spearmanr(tt, a[p])[0] for p in idx[1:1001]])
    rho_abs_null = np.array([stats.spearmanr(tt, np.abs(a - np.median(a))[p])[0] for p in idx[1:1001]])
    return dict(r2=float(r2[0]), p_perm_r2=float((np.sum(r2[1:] >= r2[0]) + 1) / nperm),
                speed_km_per_day=float(24 / smag), azimuth_deg=float(np.degrees(np.arctan2(sx, sy)) % 360),
                s_along_h_per_km=float(sx * UA[0] + sy * UA[1]),
                spearman_t_vs_along=float(rho_a), p_along=float((np.sum(np.abs(rho_a_null) >= abs(rho_a)) + 1) / 1001),
                spearman_t_vs_absalong=float(rho_abs), p_absalong=float((np.sum(np.abs(rho_abs_null) >= abs(rho_abs)) + 1) / 1001))


def task3(d, P, summ):
    rng = np.random.default_rng(33)
    tq = TQ['M7']
    fams = list(range(1, 28))
    B = make_bursts(d)
    rows = []
    for F in fams:
        t = d.t.values[d.idnum.values == F]
        post, pre = t[t >= tq], t[t < tq]
        bF = B[(B.fam == F) & (B.onset >= tq)]
        b3 = bF[bF.n >= 3]
        pre_rate = np.sum(pre >= tq - YR) / 365.25
        rows.append(dict(fam=F, x=P.loc[F, 'x'], y=P.loc[F, 'y'], z=P.loc[F, 'z'], along_km=P.loc[F, 'a'],
                         last_before_d=(pre.max() - tq) / 86400, first_after_d=(post.min() - tq) / 86400,
                         third_after_d=(np.sort(post)[2] - tq) / 86400,
                         first_burst3_d=(b3.onset.min() - tq) / 86400, pre_rate_per_day=pre_rate))
    R = pd.DataFrame(rows)
    res = {}
    for col in ['first_after_d', 'third_after_d', 'first_burst3_d', 'last_before_d']:
        res[col] = order_stats(R[col].values, P, fams, rng)
        bs = []
        for _ in range(1000):
            s = rng.choice(27, 27)
            if len(set(s)) < 6:
                continue
            m, _ = fit_batch(R[col].values[s] * 24, P.loc[np.array(fams)[s], ['x', 'y']].values.astype(float)[None], iters=5)
            bs.append(24 / np.hypot(m[0, 1], m[0, 2]))
        res[col]['speed_boot_ci95'] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        res[col]['spearman_vs_pre_rate'] = float(stats.spearmanr(R[col], R.pre_rate_per_day)[0])
        res[col]['span_days'] = float(R[col].max() - R[col].min())
    # placebo 1: random reference times (rate confound): first event after / last before
    tt_all = {F: d.t.values[d.idnum.values == F] for F in fams}
    pl, FA = [], []
    for tr in rng.uniform(ts('2014-01-01'), ts('2026-03-01'), 300):
        if abs(tr - tq) < 60 * 86400:
            continue
        fa = np.array([(t[np.searchsorted(t, tr)] - tr) / 86400 if np.searchsorted(t, tr) < len(t) else np.nan for t in tt_all.values()])
        lb = np.array([(t[np.searchsorted(t, tr) - 1] - tr) / 86400 for t in tt_all.values()])
        if np.isnan(fa).any():
            continue
        FA.append(fa)
        o1, o2 = order_stats(fa, P, fams, rng, nperm=200), order_stats(lb, P, fams, rng, nperm=200)
        pl.append(dict(kind='random', r2_first=o1['r2'], rho_along_first=o1['spearman_t_vs_along'], rho_abs_first=o1['spearman_t_vs_absalong'],
                       speed_first=o1['speed_km_per_day'], span_first=np.ptp(fa),
                       r2_last=o2['r2'], rho_along_last=o2['spearman_t_vs_along'], rho_abs_last=o2['spearman_t_vs_absalong']))
    # placebo 2: restarts after other network-wide quiet gaps >= 4 d (no event in any family)
    t = d.t.values
    gi = np.where(np.diff(t) > 1.5 * 86400)[0]
    gaps = []
    for g in gi:
        ge = t[g + 1]
        if abs(t[g] - tq) < 40 * 86400 or ge < ts('2009-06-01'):
            continue
        fa = np.array([(x[np.searchsorted(x, t[g] + 1)] - t[g]) / 86400 if np.searchsorted(x, t[g] + 1) < len(x) else np.nan for x in tt_all.values()])
        if np.isnan(fa).any():
            continue
        o1 = order_stats(fa, P, fams, rng, nperm=500)
        gaps.append(dict(gap_start=str(pd.to_datetime(t[g], unit='s')), gap_days=(ge - t[g]) / 86400,
                         span_days=float(np.ptp(fa)), **{k: o1[k] for k in ['r2', 'p_perm_r2', 'speed_km_per_day', 'spearman_t_vs_along', 'spearman_t_vs_absalong']}))
    pl = pd.DataFrame(pl)
    G = pd.DataFrame(gaps)
    G.to_csv(f'{OUT}/restart_quiet_gaps.csv', index=False)
    pl.to_csv(f'{OUT}/restart_random_time_placebo.csv', index=False)
    for col, pref in [('first_after_d', 'first'), ('last_before_d', 'last')]:
        o = res[col]
        res[col]['p_placebo_r2'] = float((np.sum(pl[f'r2_{pref}'] >= o['r2']) + 1) / (len(pl) + 1))
        res[col]['p_placebo_rho_along'] = float((np.sum(np.abs(pl[f'rho_along_{pref}']) >= abs(o['spearman_t_vs_along'])) + 1) / (len(pl) + 1))
        res[col]['p_placebo_rho_absalong'] = float((np.sum(np.abs(pl[f'rho_abs_{pref}']) >= abs(o['spearman_t_vs_absalong'])) + 1) / (len(pl) + 1))
    if len(G):
        o = res['first_after_d']
        res['first_after_d']['p_quietgap_r2'] = float((np.sum(G.r2 >= o['r2']) + 1) / (len(G) + 1))
        res['first_after_d']['p_quietgap_rho_absalong'] = float((np.sum(np.abs(G.spearman_t_vs_absalong) >= abs(o['spearman_t_vs_absalong'])) + 1) / (len(G) + 1))
    FA = np.array(FA)
    R['first_after_pctl_vs_random'] = [(FA[:, k] < R.first_after_d.values[k]).mean() for k in range(27)]
    R['first_after_over_median_wait'] = R.first_after_d.values / np.median(FA, 0)
    res['rate_normalised'] = dict(
        median_wait_after_random_time_d={int(F): float(v) for F, v in zip(fams, np.median(FA, 0))},
        spearman_ratio_vs_along=[float(x) for x in stats.spearmanr(R.first_after_over_median_wait, R.along_km)],
        spearman_ratio_vs_absalong=[float(x) for x in stats.spearmanr(R.first_after_over_median_wait, np.abs(R.along_km - R.along_km.median()))],
        frac_fams_beyond_99pct=float((R.first_after_pctl_vs_random >= 0.99).mean()))
    # placebo for the (post hoc) rate-normalised gradient: same statistic at random times and after other quiet gaps
    med = np.median(FA, 0)
    av = P.loc[fams, 'a'].values.astype(float)
    obs = stats.spearmanr(R.first_after_d.values / med, av)[0]
    rr = np.array([stats.spearmanr(f / med, av)[0] for f in FA])
    rg = []
    for g in gi:
        if abs(t[g] - tq) < 40 * 86400 or t[g + 1] < ts('2009-06-01'):
            continue
        fa = np.array([(x[np.searchsorted(x, t[g] + 1)] - t[g]) / 86400 if np.searchsorted(x, t[g] + 1) < len(x) else np.nan for x in tt_all.values()])
        if not np.isnan(fa).any():
            rg.append(stats.spearmanr(fa / med, av)[0])
    rg = np.array(rg)
    res['rate_normalised'].update(rho_obs=float(obs), p_random_times=float((np.sum(np.abs(rr) >= abs(obs)) + 1) / (len(rr) + 1)),
                                  p_quiet_gaps=float((np.sum(np.abs(rg) >= abs(obs)) + 1) / (len(rg) + 1)),
                                  quiet_gap_rho_5_50_95=[float(x) for x in np.percentile(rg, [5, 50, 95])],
                                  random_rho_2p5_50_97p5=[float(x) for x in np.percentile(rr, [2.5, 50, 97.5])],
                                  caveat='post hoc statistic (chosen after seeing raw-time results)')
    res['n_random_placebo'] = len(pl)
    res['n_quiet_gaps'] = len(G)
    res['quiet_gaps_median_r2'] = float(G.r2.median()) if len(G) else None
    res['stop_restart_rank_corr'] = float(stats.spearmanr(R.last_before_d, R.first_after_d)[0])
    res['stop_restart_rank_corr_burst3'] = float(stats.spearmanr(R.last_before_d, R.first_burst3_d)[0])
    R.to_csv(f'{OUT}/restart_M7.csv', index=False)
    summ['t3_restart'] = res
    print(R.sort_values('first_after_d').round(2).to_string())
    print(json.dumps(res, indent=1), flush=True)
    print(G.round(3).to_string())


# ---------------------------------------------------------------- task 4
def gc_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return 6371.0 * np.arccos(np.clip(c, -1, 1))


def task4(d, P, summ):
    rng = np.random.default_rng(44)
    Q = pd.read_csv(f'{OUT}/usgs/usgs_M6.5_2008_2026.csv')
    Q = Q[Q.type == 'earthquake'].copy()
    Q['to'] = (pd.to_datetime(Q.time, utc=True).dt.tz_localize(None) - EPOCH).dt.total_seconds()
    Q['dist_km'] = gc_km(LAT0, LON0, Q.latitude.values, Q.longitude.values)
    Q['delta_deg'] = Q.dist_km / 111.19
    Q['t_surf'] = Q.to + Q.dist_km / 4.0          # Love-wave onset (4 km/s); Rayleigh ~3.5 km/s follows
    # Ms-type PGV proxy (Lay & Wallace: Ms = log10(A/T) + 1.66 log10(delta) + 3.3, A in um), PGV ~ 2*pi*A/T
    Q['pgv_proxy_um_s'] = 2 * np.pi * 10 ** (Q.mag - 1.66 * np.log10(np.maximum(Q.delta_deg, 1.0)) - 3.3)
    Q['dyn_strain_proxy'] = Q.pgv_proxy_um_s * 1e-6 / 3500.0
    Q = Q[(Q.dist_km > 300) & (Q.t_surf > ts(T_START)) & (Q.t_surf < T_END - 2 * 86400)].reset_index(drop=True)
    res = {}
    Hs = [0.5, 1, 3, 6, 12, 24]
    for v in ['all', 'thin']:
        dv = catalog_variant(d, v)
        te = np.sort(dv.t.values)
        Bo = np.sort(make_bursts(dv).onset.values)

        def cnt(arr, a, b):
            return np.searchsorted(arr, b) - np.searchsorted(arr, a)
        # control times: same quake shifted by k*M2 with time-of-day offset <=1 h, |shift| 3-60 d
        ctrl = Q.t_surf.values[:, None] + rng.choice(SHIFTS, (len(Q), 300))
        ctrl = np.where((ctrl > ts(T_START)) & (ctrl < T_END - 86400), ctrl, Q.t_surf.values[:, None] - rng.choice(np.abs(SHIFTS), (len(Q), 300)))
        for H in Hs:
            for kind, arr in [('events', te), ('onsets', Bo)]:
                o = cnt(arr, Q.t_surf.values, Q.t_surf.values + H * 3600)
                pre = cnt(arr, Q.to.values - H * 3600, Q.to.values)
                c = cnt(arr, ctrl, ctrl + H * 3600)
                Q[f'{v}_{kind}_obs_{H}h'] = o
                Q[f'{v}_{kind}_pre_{H}h'] = pre
                Q[f'{v}_{kind}_exp_{H}h'] = c.mean(1)
                # randomised PIT (continuous under the null even for discrete counts)
                Q[f'{v}_{kind}_pctl_{H}h'] = (c < o[:, None]).mean(1) + rng.random(len(o)) * (c == o[:, None]).mean(1)
        for sub, sel in [('all_tele', np.ones(len(Q), bool)),
                         ('top30_pgv', Q.pgv_proxy_um_s.rank(ascending=False).values <= 30),
                         ('pgv_gt_100um_s', Q.pgv_proxy_um_s.values > 100)]:
            for H in Hs:
                for kind in ['events', 'onsets']:
                    o = Q.loc[sel, f'{v}_{kind}_obs_{H}h'].sum()
                    # stack null: sum over quakes of one random control window each (2000 draws) -> exact matched null
                    arr = te if kind == 'events' else Bo
                    cs = ctrl[sel]
                    jj = rng.integers(0, cs.shape[1], (2000, len(cs)))
                    cc = cnt(arr, cs, cs + H * 3600)            # (nq, 300)
                    nul = cc[np.arange(len(cs))[None, :], jj].sum(1)
                    # fraction of quakes with any activity
                    fo = (Q.loc[sel, f'{v}_{kind}_obs_{H}h'] > 0).mean()
                    fc = (cc > 0).mean()
                    res[f'{v}|{sub}|{kind}|{H}h'] = dict(n_quakes=int(sel.sum()), obs=int(o), expected=float(nul.mean()),
                                                         null_ci95=[float(np.quantile(nul, .025)), float(np.quantile(nul, .975))],
                                                         ratio=float(o / nul.mean()) if nul.mean() > 0 else np.nan,
                                                         p_upper=float((np.sum(nul >= o) + 1) / (len(nul) + 1)),
                                                         p_lower=float((np.sum(nul <= o) + 1) / (len(nul) + 1)),
                                                         frac_quakes_active=float(fo), frac_ctrl_active=float(fc))
        # per-quake percentile uniformity (KS) for 6 h onsets
        for H in [1, 6]:
            res[f'{v}|KS_randPIT_onsets_{H}h'] = float(stats.kstest(Q[f'{v}_onsets_pctl_{H}h'], 'uniform').pvalue)
            res[f'{v}|mean_randPIT_onsets_{H}h'] = float(Q[f'{v}_onsets_pctl_{H}h'].mean())
            res[f'{v}|spearman_PIT_vs_pgv_{H}h'] = [float(x) for x in stats.spearmanr(Q[f'{v}_onsets_pctl_{H}h'], Q.pgv_proxy_um_s)]
    named = {'Maule2010': '2010-02-27', 'Tohoku2011': '2011-03-11', 'Sumatra2012': '2012-04-11',
             'HaidaGwaii2012': '2012-10-28', 'GulfAlaska2018': '2018-01-23', 'Ridgecrest2019': '2019-07-06',
             'Chignik2021': '2021-07-29', 'Turkey2023': '2023-02-06', 'Kamchatka2025': '2025-07-29'}
    Q['name'] = ''
    for nm, dte in named.items():
        m = (pd.to_datetime(Q.time).dt.strftime('%Y-%m-%d') == dte)
        if m.any():
            Q.loc[Q[m].mag.idxmax(), 'name'] = nm
    Q.to_csv(f'{OUT}/remote_quakes.csv', index=False)
    cols = ['name', 'time', 'mag', 'dist_km', 'pgv_proxy_um_s', 'all_events_obs_6h', 'all_events_exp_6h', 'all_events_pctl_6h',
            'all_onsets_obs_6h', 'all_onsets_exp_6h', 'all_onsets_pctl_6h']
    top = Q.sort_values('pgv_proxy_um_s', ascending=False).head(20)[cols]
    print(top.round(2).to_string())
    print(Q[Q.name != ''][cols].round(2).to_string())
    summ['t4_remote'] = dict(n_quakes=len(Q), stack=res,
                             top20_pgv=top.round(3).to_dict('records'),
                             named=Q[Q.name != ''][cols].round(3).to_dict('records'),
                             note='control windows = same quake time shifted by k*12.42h with time-of-day offset <=1 h, 3-60 d; 300 per quake')
    for k, v in res.items():
        if '6h' in k or '1h' in k or 'KS' in k:
            print(k, v)


# ---------------------------------------------------------------- figures
def figures(d, P):
    Y = pd.read_csv(f'{OUT}/yearly_episode_stats.csv')
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for a, (k, lab) in zip(ax.flat, [('ep_per_1000_bursts', 'episodes per 1000 family bursts'),
                                     ('frac_sig', 'fraction propagating (perm p<0.05)'),
                                     ('med_log10_speed_sig', 'median log10 speed (km/day), propagating'),
                                     ('frac_westled_sig', 'fraction WNW-led (propagating)'),
                                     ('mean_nfam', 'mean families per episode'),
                                     ('med_dur_h', 'median episode duration (h)')]):
        for v, c in [('thin', 'k'), ('all', 'C0'), ('top15', 'C1')]:
            y = Y[Y.variant == v]
            a.plot(y.year + 0.5, y[k], 'o-', color=c, ms=3, lw=1, label=v)
        for q in QUAKES.values():
            t = pd.Timestamp(q)
            a.axvline(t.year + t.day_of_year / 365.25, color='r', ls='--', lw=0.8)
        a.set_title(lab, fontsize=10)
    ax[1, 2].set_xlabel('year')
    ax[0, 0].legend(fontsize=8)
    fig.suptitle('Network-episode statistics by year (red: Petrolia M6.2, Ferndale M6.4, M7.0); thin = meancc>=0.13')
    fig.tight_layout()
    fig.savefig('figures/sci_cascades2_episodes_by_year.png', dpi=120)
    plt.close(fig)

    T = pd.read_csv(f'{OUT}/network_pairs_thin_all.csv')
    W = pd.read_csv(f'{OUT}/network_by_window.csv')
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    U = T[T.i < T.j]
    for k, c in zip(DTS, ['C0', 'C1', 'C2']):
        ax[0].scatter(U.dist_h_km, np.log2(U[f'Gsym_{k}']), s=6, alpha=0.5, color=c, label=f'dt={k}')
        bins = np.arange(0, 18, 2)
        mb = [np.median(np.log2(U[f'Gsym_{k}'][(U.dist_h_km >= b) & (U.dist_h_km < b + 2)])) for b in bins]
        ax[0].plot(bins + 1, mb, '-', color=c, lw=2)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('horizontal separation (km)')
    ax[0].set_ylabel('log2 gain vs tidal-phase-preserving shift null')
    ax[0].legend(fontsize=8)
    ax[0].set_title('pairwise triggering gain vs distance')
    a = ax[1]
    sig = T[(T.i < T.j) & (T.q_lead < 0.05)]
    for _, r in sig.iterrows():
        src, dst = (r.i, r.j) if r.lead_ij_1h > r.lead_ji_1h else (r.j, r.i)
        p0, p1 = P.loc[src], P.loc[dst]
        a.annotate('', xy=(p1.x, p1.y), xytext=(p0.x, p0.y),
                   arrowprops=dict(arrowstyle='->', color='0.4', lw=0.6, alpha=0.7, shrinkA=4, shrinkB=4))
    sc = a.scatter(P.x, P.y, c=P.z, s=60, cmap='viridis', zorder=3, edgecolor='k')
    for F, r in P.iterrows():
        a.text(r.x + 0.15, r.y + 0.15, str(F), fontsize=7, zorder=4)
    plt.colorbar(sc, ax=a, label='depth (km)')
    a.set_xlabel('E (km)')
    a.set_ylabel('N (km)')
    a.set_aspect('equal')
    a.set_title(f'directed lead (1 h, BH q<0.05): {len(sig)} edges')
    yy = W[W.window.str.len() == 4]
    for k, c in zip(DTS, ['C0', 'C1', 'C2']):
        ax[2].plot(yy.window.astype(int) + 0.5, yy[f'med_log2G_{k}'], 'o-', color=c, label=k)
        for lab, mk in [('pre_M7_1yr', 's'), ('post_M7_1yr', 'D')]:
            r = W[W.window == lab].iloc[0]
            ax[2].plot(2024.93 + (0.5 if 'post' in lab else -0.5), r[f'med_log2G_{k}'], mk, color=c, mec='k', ms=8)
    for q in QUAKES.values():
        t = pd.Timestamp(q)
        ax[2].axvline(t.year + t.day_of_year / 365.25, color='r', ls='--', lw=0.8)
    ax[2].set_ylabel('median pair log2 gain')
    ax[2].set_title('coupling by year (sq: 1 yr pre-M7, diamond: post)')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('figures/sci_cascades2_network.png', dpi=120)
    plt.close(fig)

    R = pd.read_csv(f'{OUT}/restart_M7.csv')
    Q = pd.read_csv(f'{OUT}/remote_quakes.csv')
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a = ax[0]
    a.scatter(R.along_km, R.first_after_d, c='C3', label='first event after M7')
    a.scatter(R.along_km, R.first_burst3_d, c='C0', marker='s', label='first burst (>=3 ev)')
    a.scatter(R.along_km, R.last_before_d, c='0.5', marker='v', label='last event before M7')
    for _, r in R.iterrows():
        a.text(r.along_km + 0.1, r.first_after_d + 0.2, str(int(r.fam)), fontsize=7)
    a.axhline(0, color='k', lw=0.8)
    a.set_xlabel('along-strike position (km, + = WNW, strike 293)')
    a.set_ylabel('days relative to M7 (2024-12-05 18:44)')
    a.legend(fontsize=8)
    a.set_title('post-M7 restart vs position')
    pl = pd.read_csv(f'{OUT}/restart_random_time_placebo.csv')
    G = pd.read_csv(f'{OUT}/restart_quiet_gaps.csv')
    s = json.load(open(f'{OUT}/summary_t3.json'))['t3_restart']
    ax[1].hist(pl.rho_abs_first, bins=30, color='0.7', density=True, label='random times')
    if len(G):
        ax[1].hist(G.spearman_t_vs_absalong, bins=15, color='C0', alpha=0.5, density=True, label='other network quiet gaps >=1.5 d')
    ax[1].axvline(s['first_after_d']['spearman_t_vs_absalong'], color='C3', lw=2, label='M7 restart')
    ax[1].set_xlabel('Spearman(first-event time, |along-strike distance from centre|)')
    ax[1].legend(fontsize=8)
    ax[1].set_title('ends-first restart? placebo comparison')
    # remote stack: obs/expected per lag-hour window
    lags = np.arange(-24, 48, 1)
    te = np.sort(catalog_variant(d, 'all').t.values)
    sel = Q.pgv_proxy_um_s.rank(ascending=False) <= 30
    for subsel, c, lab in [(np.ones(len(Q), bool), 'k', f'all {len(Q)} M>=6.5 >300 km'), (sel.values, 'C3', 'top 30 PGV proxy')]:
        ts_ = Q.t_surf.values[subsel]
        cnt = np.array([np.sum(np.searchsorted(te, ts_ + (h + 1) * 3600) - np.searchsorted(te, ts_ + h * 3600)) for h in lags])
        exp = Q.loc[subsel, 'all_events_exp_1h'].sum()
        ax[2].step(lags, cnt / exp, where='post', color=c, label=lab)
    ax[2].axvline(0, color='b', ls='--', lw=0.8)
    ax[2].axhline(1, color='0.5', lw=0.5)
    ax[2].set_xlabel('hours after surface-wave arrival (4 km/s)')
    ax[2].set_ylabel('stacked LFE count / tidal-matched expectation (1 h)')
    ax[2].legend(fontsize=8)
    ax[2].set_title('remote triggering stack')
    fig.tight_layout()
    fig.savefig('figures/sci_cascades2_restart_remote.png', dpi=120)
    plt.close(fig)


def main():
    tasks = sys.argv[1:] or ['t1', 't2', 't3', 't4', 'fig']
    d, P = load()
    for tk in tasks:
        summ = {}
        if tk == 't1':
            task1(d, P, summ)
        elif tk == 't2':
            task2(d, P, summ)
        elif tk == 't3':
            task3(d, P, summ)
        elif tk == 't4':
            task4(d, P, summ)
        elif tk == 'fig':
            figures(d, P)
            allsum = {}
            for t in ['t1', 't2', 't3', 't4']:
                f = f'{OUT}/summary_{t}.json'
                if os.path.exists(f):
                    allsum.update(json.load(open(f)))
            allsum['catalog_note'] = (f'constant-completeness thinning: meancc >= {MEANCC_CUT} (detection floor thresh/NSta 99th pct '
                                      '<= 0.13 in 2009-2026), 2008 excluded')
            json.dump(allsum, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
            continue
        json.dump(summ, open(f'{OUT}/summary_{tk}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
