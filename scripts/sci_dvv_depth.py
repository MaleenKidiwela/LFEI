#!/usr/bin/env python3
"""Where in depth is the LFE-coda dv/v change?  (analysis; inputs made by the helpers)

Inputs
  data/dvv/pool/params_by_channel.csv          pooled per-channel steps (27 families, dvv_cwire_pool.py)
  data/sci/dvv_depth/pgv_by_station.csv        measured + BA08 PGV     (sci_dvv_depth_pgv.py)
  data/sci/dvv_depth/fam{23,8,27}_pairs.parquet  lapse-time / band split (sci_dvv_depth_measure.py)
  data/sci/dvv_depth/fam23_pairs_inj2024-12-05_-0.0005.parquet  injection control
  data/sci/dvv_depth/kernel_windows.csv, kernel_mc.csv   MC sensitivity (sci_dvv_depth_kernel.py)
  data/sci/centroids/station_delay_change_M7_by_station.csv   direct-wave delays (other agent)
A. station-by-station step vs PGV / borehole / kernel predictors (WLS + intrinsic scatter,
   permutation and station-bootstrap nulls)
B. lapse-time and frequency split: joint group fits (surface BK/NC, borehole PB) of the dvv_cwire
   parametric model to the pair rows of 3 families, block bootstrap over (family, stack) shared by
   all configurations (paired differences) and delete-one-station jackknife; injection recovery.
C. kernel scenarios: required deep / source-region / shallow change for the observed signals.
Outputs: data/sci/dvv_depth/*.csv, summary.json; figures/sci_dvv_depth_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import PNAMES, epoch_s, param_G  # noqa: E402

OUT = 'data/sci/dvv_depth'
FAMS = [23, 8, 27]
CFGS = ['b28_full', 'b28_early', 'b28_late', 'b24_full', 'b48_full']
MIN_CCREF, MIN_PAIRCC = 0.25, 0.2
NBOOT = 200
rng = np.random.default_rng(11)


# ------------------------------------------------------------------ helpers
def wls_scatter(y, s, X):
    """WLS with intrinsic scatter sig_int (chosen so chi2/dof = 1, >= 0). Returns beta, se, sig_int, chi2r0."""
    def fit(si):
        w = 1 / (s ** 2 + si ** 2)
        A = X * np.sqrt(w)[:, None]
        b = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        cov = np.linalg.pinv(A.T @ A)
        chi = np.sum(w * (y - X @ b) ** 2) / max(len(y) - X.shape[1], 1)
        return b, cov, chi
    b, cov, chi0 = fit(0.0)
    si = 0.0
    if chi0 > 1:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if fit(mid)[2] > 1:
                lo = mid
            else:
                hi = mid
        si = hi
        b, cov, _ = fit(si)
    w = 1 / (s ** 2 + si ** 2)
    ll = -0.5 * np.sum(w * (y - X @ b) ** 2 + np.log(2 * np.pi / w))
    return b, np.sqrt(np.diag(cov)), si, chi0, ll


def combine(v, s):
    w = 1 / s ** 2
    m = np.sum(w * v) / w.sum()
    se = np.sqrt(1 / w.sum())
    chi = np.sum(w * (v - m) ** 2) / max(len(v) - 1, 1)
    return m, se * np.sqrt(max(1.0, chi))


# ------------------------------------------------------------------ A: station regression
def part_a(K):
    P = pd.read_csv('data/dvv/pool/params_by_channel.csv')
    G = pd.read_csv(f'{OUT}/pgv_by_station.csv')
    rows = []
    for sta, g in P.groupby('sta'):
        r = dict(sta=sta, bore=sta.startswith('PB.'), dist_km=g.dist_km.iloc[0], lat=g.lat.iloc[0], lon=g.lon.iloc[0],
                 n_chan=len(g))
        coll = np.isclose(g.step_pet_pct, g.step_fer_pct)          # pet/fer collinear (data start after 2021)
        for q in ['pet', 'fer', 'm7']:
            gg = g[~coll] if q in ('pet', 'fer') else g
            if len(gg) == 0:
                r[f'step_{q}'], r[f'step_{q}_se'] = np.nan, np.nan
                continue
            r[f'step_{q}'], r[f'step_{q}_se'] = combine(gg[f'step_{q}_pct'].values, gg[f'step_{q}_pct_se'].values)
        r['tr60'], r['tr60_se'] = combine(g.transient60_pct.values, g.transient60_pct_se.values)
        rows.append(r)
    S = pd.DataFrame(rows)
    # PGV: measured (unclipped) where available; else GMPE x empirical ratio (station median, else group median)
    G['ratio'] = G.pgv_gm_cms / G.pgv_gmpe_cms
    G['bore'] = G.sta.str.startswith('PB.')
    grp_ratio = G.groupby('bore').ratio.median()
    sta_ratio = G.groupby('sta').ratio.median()
    G['pgv_use'] = G.pgv_gm_cms
    G['pgv_src'] = np.where(G.pgv_gm_cms.notna(), 'measured', '')
    m = G.pgv_use.isna()
    sr = G.sta.map(sta_ratio)
    G.loc[m & sr.notna(), 'pgv_src'] = 'gmpe*station_ratio'
    G.loc[m & sr.isna(), 'pgv_src'] = 'gmpe*group_ratio'
    G.loc[m, 'pgv_use'] = G.pgv_gmpe_cms[m] * np.where(sr[m].notna(), sr[m], G.bore[m].map(grp_ratio))
    site = G[G.pgv_src == 'measured'].groupby('sta').ratio.apply(lambda x: np.log10(x).mean())
    S['site_log10'] = S.sta.map(site)
    for q in ['pet', 'fer', 'm7']:
        gq = G[G.quake == q].set_index('sta')
        S[f'pgv_{q}'] = S.sta.map(gq.pgv_use)
        S[f'pgv_{q}_src'] = S.sta.map(gq.pgv_src)
        S[f'pgv_{q}_gmpe'] = S.sta.map(gq.pgv_gmpe_cms)
    # kernel predictors at each station distance (l = 50 km uniform, full window)
    kw = K[(K.l_km == '50') & (K.window == 'full')].sort_values('dist_km')
    S['K_sphere'] = np.interp(S.dist_km, kw.dist_km, kw.K_sphere)
    S['K_deep15'] = np.interp(S.dist_km, kw.dist_km, kw.K_deep15)
    S.to_csv(f'{OUT}/station_steps_pgv.csv', index=False)

    # regressions
    reg = []
    for q in ['m7', 'fer', 'pet', 'tr60']:
        yc, sc = (f'step_{q}', f'step_{q}_se') if q != 'tr60' else ('tr60', 'tr60_se')
        pq = 'm7' if q == 'tr60' else q
        for subset in ['all', 'surface']:
            d = S[np.isfinite(S[yc]) & np.isfinite(S[f'pgv_{pq}'])]
            if subset == 'surface':
                d = d[~d.bore]
            if subset == 'surface' and q == 'm7':
                d = d[d[f'pgv_{pq}_src'] == 'measured']
            y, s = d[yc].values, d[sc].values
            lp = np.log10(d[f'pgv_{pq}'].values)
            one = np.ones(len(d))
            models = {'const': np.c_[one], 'logPGV': np.c_[one, lp], 'K_sphere': np.c_[one, d.K_sphere.values],
                      'K_deep15': np.c_[one, d.K_deep15.values], 'dist': np.c_[one, d.dist_km.values]}
            if subset == 'all':
                models['bore'] = np.c_[one, d.bore.values.astype(float)]
                models['logPGV+bore'] = np.c_[one, lp, d.bore.values.astype(float)]
            for mn, X in models.items():
                b, se, si, chi0, ll = wls_scatter(y, s, X)
                r = dict(target=q, subset=subset, model=mn, n=len(d), intercept=b[0], intercept_se=se[0],
                         sig_int=si, chi2r_noscatter=chi0, logL=ll, AIC=2 * X.shape[1] + 2 - 2 * ll)
                if X.shape[1] > 1:
                    r['slope'], r['slope_se'] = b[1], se[1]
                    # permutation null for the first predictor (shuffle within surface/borehole)
                    perm = []
                    for _ in range(1000):
                        Xp = X.copy()
                        if mn == 'bore':                    # permute the indicator itself
                            Xp[:, 1] = rng.permutation(X[:, 1])
                        else:                               # permute within surface / borehole groups
                            for bb in [True, False]:
                                ii = np.where(d.bore.values == bb)[0]
                                Xp[ii, 1] = X[rng.permutation(ii), 1]
                        perm.append(wls_scatter(y, s, Xp)[0][1])
                    perm = np.array(perm)
                    r['slope_perm_p'] = float(np.mean(np.abs(perm) >= abs(b[1])))
                    # bootstrap over stations
                    bs = []
                    for _ in range(500):
                        ii = rng.integers(0, len(d), len(d))
                        if np.ptp(X[ii, 1]) == 0:
                            continue
                        bs.append(wls_scatter(y[ii], s[ii], X[ii])[0][1])
                    r['slope_boot_se'] = float(np.std(bs))
                if X.shape[1] > 2:
                    r['coef2'], r['coef2_se'] = b[2], se[2]
                reg.append(r)
    R = pd.DataFrame(reg)
    R.to_csv(f'{OUT}/station_regressions.csv', index=False)
    return S, R


# ------------------------------------------------------------------ B: lapse time / frequency
def load_pairs(tag=''):
    out = []
    for F in FAMS:
        f = f'{OUT}/fam{F}_pairs{tag}.parquet'
        if not os.path.exists(f):
            continue
        p = pd.read_parquet(f)
        m = pd.read_csv(f'{OUT}/fam{F}_channels{tag}.csv')
        T = epoch_s(pd.to_datetime(pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv').t, format='mixed'))
        p = p.assign(fam=F)
        p['tb'] = T[p.b.values]
        p['ta'] = np.where(p.a >= 0, T[np.clip(p.a.values, 0, None)], np.nan)
        out.append((p, m))
    P = pd.concat([o[0] for o in out], ignore_index=True)
    M = pd.concat([o[1].assign(fam=F) for o, F in zip(out, FAMS[:len(out)])], ignore_index=True)
    return P, M


class GroupFit:
    """Parametric model (dvv_cwire.param_G columns) + one offset per (family, channel) for the reference
    rows; normal equations built with one-hot offsets (fast for 1e6 rows); Huber IRLS."""
    def __init__(self, p):
        self.has_a = (p.a >= 0).values
        key = p.fam.astype(str) + ':' + p.chan.astype(str)
        self.k, self.keys = pd.factorize(key)
        self.nk = len(self.keys)
        F = param_G(np.nan_to_num(p.ta.values), p.tb.values, self.has_a, np.zeros(len(p), int), 1)[:, :7]
        self.F = F
        self.d = p.eps.values
        cc = np.clip(p.cc.values, 0, 0.99)
        w0 = cc ** 2 / (1 - cc ** 2)
        self.w0 = w0 / w0.mean()
        self.ref = ~self.has_a
        self.sid_b = (p.fam.values * 100000 + p.b.values)
        self.sid_a = np.where(self.has_a, p.fam.values * 100000 + p.a.values, -1)
        self.sta = p.chan.str.split('.').str[:2].str.join('.').values
        self.w = self.w0.copy()
        self.m = self.solve(self.w0, irls=3)

    def solve(self, w, irls=0):
        F, d, k, ref, nk = self.F, self.d, self.k, self.ref, self.nk
        for it in range(irls + 1):
            A11 = (F * w[:, None]).T @ F
            wr = np.where(ref, w, 0.0)
            A12 = np.zeros((7, nk))
            for j in range(7):
                A12[j] = -np.bincount(k, weights=wr * F[:, j], minlength=nk)
            A22 = np.diag(np.bincount(k, weights=wr, minlength=nk)) + 1e-12 * np.eye(nk)
            b1 = F.T @ (w * d)
            b2 = -np.bincount(k, weights=wr * d, minlength=nk)
            N = np.block([[A11, A12], [A12.T, A22]])
            x = np.linalg.lstsq(N, np.r_[b1, b2], rcond=None)[0]
            if it < irls:
                r = d - F @ x[:7] + np.where(ref, x[7:][k], 0.0)
                z = np.abs(r) * np.sqrt(self.w0) / (1.4826 * np.median(np.abs(r) * np.sqrt(self.w0)) + 1e-12)
                w = self.w0 * np.where(z <= 2.0, 1.0, 2.0 / np.maximum(z, 1e-12))
                self.w = w
        return x[:7]


def part_b():
    P, M = load_pairs()
    Pi, Mi = load_pairs('_inj2024-12-05_-0.0005')
    P = P[P.cc >= MIN_PAIRCC]
    Pi = Pi[Pi.cc >= MIN_PAIRCC]
    M['sta'] = M.chan.str.split('.').str[:2].str.join('.')
    ok = {c: set(zip(M.fam[M[f'{c}_ccref'] >= MIN_CCREF], M.chan[M[f'{c}_ccref'] >= MIN_CCREF]))
          for c in CFGS if f'{c}_ccref' in M}
    comps = {'lapse': ('b28_early', 'b28_late'), 'band': ('b24_full', 'b48_full'), 'full': ('b28_full', 'b28_full')}
    groups = {'surface': lambda c: ~c.str.startswith('PB.'), 'borehole': lambda c: c.str.startswith('PB.')}
    res, chan_used = [], []
    for gname, gsel in groups.items():
        for comp, (c1, c2) in comps.items():
            keep = ok[c1] & ok[c2]
            fits = {}
            for c in dict.fromkeys([c1, c2]):
                p = P[(P.cfg == c) & gsel(P.chan)]
                p = p[pd.MultiIndex.from_arrays([p.fam, p.chan]).isin(list(keep))].reset_index(drop=True)
                fits[c] = GroupFit(p)
                chan_used.append(dict(group=gname, comparison=comp, cfg=c, n_chan_fam=len(keep & set(zip(p.fam, p.chan))),
                                      n_rows=len(p), stations=';'.join(sorted(set(fits[c].sta)))))
            # paired block bootstrap over (family, stack), same draws for both configs
            ids = np.unique(np.concatenate([np.r_[f.sid_b, f.sid_a[f.sid_a >= 0]] for f in fits.values()]))
            ix = {c: (np.searchsorted(ids, f.sid_b), np.searchsorted(ids, np.where(f.sid_a >= 0, f.sid_a, ids[0])))
                  for c, f in fits.items()}
            boots = {c: [] for c in fits}
            for i in range(NBOOT):
                cnt = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
                for c, f in fits.items():
                    mb = cnt[ix[c][0]]
                    ma = np.where(f.sid_a >= 0, cnt[ix[c][1]], 1.0)
                    boots[c].append(f.solve(f.w * mb * ma))
            # delete-one-station jackknife
            stas = sorted(set(fits[c1].sta) & set(fits[c2].sta))
            jk = {c: [] for c in fits}
            for s in stas:
                for c, f in fits.items():
                    jk[c].append(f.solve(f.w * (f.sta != s)))
            nj = len(stas)
            for c, f in fits.items():
                B, J = np.array(boots[c]), np.array(jk[c])
                seb = B.std(0)
                sej = np.sqrt((nj - 1) / nj * np.sum((J - J.mean(0)) ** 2, 0)) if nj > 1 else np.zeros(7)
                for j, pn in enumerate(PNAMES):
                    res.append(dict(group=gname, comparison=comp, cfg=c, param=pn, value=f.m[j] * 100,
                                    se_boot=seb[j] * 100, se_jack=sej[j] * 100, se=max(seb[j], sej[j]) * 100,
                                    n_sta=nj))
            if c1 != c2:
                B1, B2 = np.array(boots[c1]), np.array(boots[c2])
                J1, J2 = np.array(jk[c1]), np.array(jk[c2])
                for j, pn in enumerate(PNAMES):
                    dv = (fits[c1].m[j] - fits[c2].m[j]) * 100
                    seb = np.std(B1[:, j] - B2[:, j]) * 100
                    dj = J1[:, j] - J2[:, j]
                    sej = np.sqrt((nj - 1) / nj * np.sum((dj - dj.mean()) ** 2)) * 100
                    rb = B1[:, j] / np.where(B2[:, j] == 0, np.nan, B2[:, j])
                    res.append(dict(group=gname, comparison=comp, cfg=f'{c1}-{c2}', param=pn, value=dv, se_boot=seb,
                                    se_jack=sej, se=max(seb, sej), n_sta=nj,
                                    ratio=fits[c1].m[j] / fits[c2].m[j],
                                    ratio_lo=np.nanpercentile(rb, 16), ratio_hi=np.nanpercentile(rb, 84)))
            print(gname, comp, 'done', flush=True)
    R = pd.DataFrame(res)
    R.to_csv(f'{OUT}/lapse_band_group_params.csv', index=False)
    pd.DataFrame(chan_used).to_csv(f'{OUT}/lapse_band_channels_used.csv', index=False)
    # injection recovery (family 23 only; identical stacks/channels with and without injection)
    inj = []
    for gname, gsel in groups.items():
        for c in CFGS:
            keep = {k for k in ok.get(c, set()) if k[0] == 23}
            a = P[(P.fam == 23) & (P.cfg == c) & gsel(P.chan)]
            b = Pi[(Pi.cfg == c) & gsel(Pi.chan)]
            a = a[pd.MultiIndex.from_arrays([a.fam, a.chan]).isin(list(keep))].reset_index(drop=True)
            b = b[pd.MultiIndex.from_arrays([b.fam, b.chan]).isin(list(keep))].reset_index(drop=True)
            if len(a) < 100 or len(b) < 100:
                continue
            fa, fb = GroupFit(a), GroupFit(b)
            inj.append(dict(group=gname, cfg=c, injected_pct=-0.05, step_m7_diff_pct=(fb.m[5] - fa.m[5]) * 100,
                            recovery=(fb.m[5] - fa.m[5]) * 100 / -0.05, transient_diff_pct=(fb.m[6] - fa.m[6]) * 100))
    I = pd.DataFrame(inj)
    I.to_csv(f'{OUT}/injection_recovery.csv', index=False)
    return R, I


# ------------------------------------------------------------------ C: kernel scenarios
def part_c(K, obs):
    rows = []
    for l in K.l_km.unique():
        for wn in ['early', 'late', 'full']:
            k = K[(K.l_km == l) & (K.window == wn) & (K.dist_km == 40)].iloc[0]
            for tL in [0.25, 0.5]:
                rows.append(dict(l_km=l, window=wn, dist_km=40, t_layer_s=tL, K_deep15=k.K_deep15, K_sphere10=k.K_sphere,
                                 K_upper5=k.K_upper5, K_layer=k.K_layer_per_tL * tL,
                                 req_deep15_for_bore_pct=obs['bore_m7'] / k.K_deep15,
                                 req_sphere10_for_bore_pct=obs['bore_m7'] / k.K_sphere,
                                 req_layer_for_bore_pct=obs['bore_m7'] / (k.K_layer_per_tL * tL),
                                 req_layer_for_surface_excess_pct=obs['surf_excess_m7'] / (k.K_layer_per_tL * tL)))
    C = pd.DataFrame(rows)
    C.to_csv(f'{OUT}/kernel_scenarios.csv', index=False)
    return C


# ------------------------------------------------------------------ figures
def figures(S, R, LB, I, K, KM):
    # 1: drop vs PGV
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for a, q in zip(ax, ['m7', 'fer', 'pet', 'tr60']):
        pq = 'm7' if q == 'tr60' else q
        yc, sc = (f'step_{q}', f'step_{q}_se') if q != 'tr60' else ('tr60', 'tr60_se')
        for bore, col, lab in [(False, 'C3', 'surface BK/NC'), (True, 'C0', 'borehole PB (~200 m)')]:
            d = S[(S.bore == bore) & np.isfinite(S[yc])]
            meas = d[f'pgv_{pq}_src'] == 'measured'
            a.errorbar(d[f'pgv_{pq}'][meas], d[yc][meas], d[sc][meas], fmt='o', color=col, label=lab + ' (PGV measured)')
            a.errorbar(d[f'pgv_{pq}'][~meas], d[yc][~meas], d[sc][~meas], fmt='s', mfc='none', color=col,
                       label=lab + ' (PGV from GMPE x ratio)')
            for _, r in d.iterrows():
                a.annotate(r.sta.split('.')[1], (r[f'pgv_{pq}'], r[yc]), fontsize=6, xytext=(3, 2), textcoords='offset points')
        rr = R[(R.target == q) & (R.subset == 'all') & (R.model == 'logPGV')]
        if len(rr):
            rr = rr.iloc[0]
            xx = np.logspace(np.log10(np.nanmin(S[f'pgv_{pq}'])), np.log10(np.nanmax(S[f'pgv_{pq}'])), 50)
            a.plot(xx, rr.intercept + rr.slope * np.log10(xx), 'k--', lw=1,
                   label=f'all: {rr.slope:+.3f}±{rr.slope_se:.3f} %/dex (perm p={rr.slope_perm_p:.3f})')
        rs = R[(R.target == q) & (R.subset == 'surface') & (R.model == 'logPGV')]
        if len(rs):
            rs = rs.iloc[0]
            a.plot(xx, rs.intercept + rs.slope * np.log10(xx), 'C3:', lw=1,
                   label=f'surface: {rs.slope:+.3f}±{rs.slope_se:.3f} (p={rs.slope_perm_p:.3f})')
        a.axhline(0, color='0.5', lw=0.5)
        a.set_xscale('log')
        a.set_xlabel('PGV (cm/s, geometric-mean horizontal)')
        a.set_ylabel({'m7': 'M7 step', 'fer': 'Ferndale step', 'pet': 'Petrolia step', 'tr60': 'M7 60-d transient'}[q] + ' dv/v (%)')
        a.legend(fontsize=6)
    ax[0].set_title('Per-station coseismic dv/v vs local PGV (27 families pooled)', fontsize=9, loc='left')
    fig.tight_layout()
    fig.savefig('figures/sci_dvv_depth_pgv.png', dpi=130)
    plt.close(fig)

    # 2: lapse/band comparison
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, pn, lab in [(ax[0], 'step_m7_pct', 'M7 step (%)'), (ax[1], 'transient60_pct', 'M7 60-d transient amplitude (%)')]:
        x = 0
        ticks = []
        for comp, cf in [('full', ['b28_full']), ('lapse', ['b28_early', 'b28_late']), ('band', ['b24_full', 'b48_full'])]:
            for c in cf:
                for gi, (g, col) in enumerate([('surface', 'C3'), ('borehole', 'C0')]):
                    r = LB[(LB.group == g) & (LB.comparison == comp) & (LB.cfg == c) & (LB.param == pn)]
                    if len(r):
                        r = r.iloc[0]
                        rec = I[(I.group == g) & (I.cfg == c)].recovery
                        a.errorbar(x + 0.15 * (gi * 2 - 1), r.value, r.se, fmt='o', color=col, label=g if x == 0 else None)
                        if len(rec) and pn == 'step_m7_pct':
                            a.plot(x + 0.15 * (gi * 2 - 1), r.value / rec.iloc[0], 'x', color=col, ms=7,
                                   label=f'{g}: / injection recovery' if x == 0 else None)
                ticks.append((x, c.replace('b28_', '2-8 ').replace('b24_full', '2-4 Hz').replace('b48_full', '4-8 Hz')
                              .replace('early', 'S+3-8 s').replace('late', 'S+8 s-end')))
                x += 1
        a.set_xticks([t[0] for t in ticks])
        a.set_xticklabels([t[1] for t in ticks], fontsize=8)
        a.axhline(0, color='0.5', lw=0.5)
        a.set_ylabel(lab)
        a.legend(fontsize=7)
        a.set_title(f'Families {FAMS}: {lab}', fontsize=9)
    # kernel panel
    a = ax[2]
    for l, ls in [('50', '-'), ('10/5km/100', '--')]:
        k = KM[(KM.l_km == l) & (KM.dist_km == 40)]
        a.plot(k.tau, k.K_deep15, 'C2' + ls, label=f'z>15 km (l={l})')
        a.plot(k.tau, k.K_sphere, 'C1' + ls, label=f'within 10 km of source (l={l})')
        a.plot(k.tau, k.K_upper5, 'C4' + ls, label=f'z<5 km (l={l})')
        a.plot(k.tau, 2 * 0.5 * k.nbounce / k.tau, 'C5' + ls, label=f'0-0.5 km layer, t_L=0.5 s (l={l})')
    a.axvspan(3, 8, color='0.9')
    a.set_xlim(1, 20)
    a.set_ylim(-0.05, 1)
    a.set_xlabel('lapse time after S, tau (s)')
    a.set_ylabel('sensitivity K (apparent dv/v per unit local dv/v)')
    a.set_title('MC sensitivity of S-aligned coda stretching (source 28 km, station 40 km)', fontsize=8)
    a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig('figures/sci_dvv_depth_lapse_band.png', dpi=130)
    plt.close(fig)


def recovery_corrected(LB, I):
    """Divide M7 steps by the injection recovery of the same group/config (family 23 injection).
    Differences: SE combined as independent (conservative; the paired bootstrap SE of the raw
    difference is smaller). The same factor is applied to the transient (not separately tested)."""
    rows = []
    for (g, comp), d in LB[LB.param.isin(['step_m7_pct', 'transient60_pct'])].groupby(['group', 'comparison']):
        cfgs = [c for c in d.cfg.unique() if '-' not in c]
        for pn in ['step_m7_pct', 'transient60_pct']:
            vals = {}
            for c in cfgs:
                r = d[(d.cfg == c) & (d.param == pn)].iloc[0]
                rec = I[(I.group == g) & (I.cfg == c)].recovery.iloc[0]
                vals[c] = (r.value / rec, r.se / rec, rec)
                rows.append(dict(group=g, comparison=comp, cfg=c, param=pn, value_raw=r.value, se_raw=r.se, recovery=rec,
                                 value_corr=vals[c][0], se_corr=vals[c][1]))
            if len(cfgs) == 2:
                (v1, s1, _), (v2, s2, _) = vals[cfgs[0]], vals[cfgs[1]]
                rows.append(dict(group=g, comparison=comp, cfg=f'{cfgs[0]}-{cfgs[1]}', param=pn,
                                 value_corr=v1 - v2, se_corr=np.hypot(s1, s2), ratio_corr=v1 / v2 if v2 else np.nan,
                                 z=(v1 - v2) / np.hypot(s1, s2)))
    C = pd.DataFrame(rows)
    C.to_csv(f'{OUT}/lapse_band_recovery_corrected.csv', index=False)
    return C


def kernel_figure(KW):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    regs = [('K_upper5', 'z<5 km'), ('K_z5_10', '5-10 km'), ('K_z10_15', '10-15 km'), ('K_deep15', 'z>15 km'),
            ('K_sphere', '<=10 km from LFE source (subset of z>15)')]
    for a, l in zip(ax, ['50', '10/5km/100']):
        for j, wn in enumerate(['early', 'late']):
            k = KW[(KW.l_km == l) & (KW.window == wn) & (KW.dist_km == 40)].iloc[0]
            vals = [k[c] for c, _ in regs] + [k.K_layer_per_tL * 0.5]
            a.bar(np.arange(len(vals)) + 0.38 * j, vals, 0.38, label=f'{wn} window (tau {"3-8" if wn == "early" else "8-15"} s)')
        a.set_xticks(np.arange(len(vals)) + 0.19)
        a.set_xticklabels([r for _, r in regs] + ['0-0.5 km layer\n(t_L=0.5 s)'], fontsize=7, rotation=20)
        a.set_title(f'S-aligned coda sensitivity, station 40 km, mean free path {l} km', fontsize=9)
        a.legend(fontsize=8)
    ax[0].set_ylabel('apparent dv/v per unit dv/v in region')
    fig.tight_layout()
    fig.savefig('figures/sci_dvv_depth_kernel.png', dpi=130)
    plt.close(fig)


def direct_wave_check():
    f = 'data/sci/centroids/station_delay_change_M7_by_station.csv'
    if not os.path.exists(f):
        return {}
    d = pd.read_csv(f)
    d = d[(d.phase == 'S') & np.isfinite(d.se_ms)]
    out = {}
    for lab, sel in [('all_S', np.ones(len(d), bool)), ('S_without_KMPB', d.sta != 'NC.KMPB'),
                     ('S_PB_only', d.sta.str.startswith('PB.'))]:
        q = d[sel]
        X = np.c_[np.ones(len(q)), q.T_mod.values]
        b, se, si, chi0, _ = wls_scatter(q.ddt_ms.values, q.se_ms.values, X)
        # ddt = dt0 - dvv * T  ->  dvv = -slope (ms/s -> fraction 1e-3)
        out[lab] = dict(n=len(q), dvv_pct=-b[1] * 1e-3 * 100, se_pct=se[1] * 1e-3 * 100, dt0_ms=b[0], sig_int_ms=si)
    return out


def main():
    K = pd.read_csv(f'{OUT}/kernel_windows.csv', dtype={'l_km': str})
    KM = pd.read_csv(f'{OUT}/kernel_mc.csv', dtype={'l_km': str})
    S, R = part_a(K)
    print(R[['target', 'subset', 'model', 'n', 'intercept', 'intercept_se', 'slope', 'slope_se', 'slope_perm_p',
             'slope_boot_se', 'coef2', 'coef2_se', 'sig_int', 'AIC']].round(4).to_string(), flush=True)
    if '--reuse-b' in sys.argv and os.path.exists(f'{OUT}/lapse_band_group_params.csv'):
        LB, I = pd.read_csv(f'{OUT}/lapse_band_group_params.csv'), pd.read_csv(f'{OUT}/injection_recovery.csv')
    else:
        LB, I = part_b()
    LC = recovery_corrected(LB, I)
    print(LC.round(4).to_string())
    print(LB[LB.param.isin(['step_m7_pct', 'transient60_pct', 'step_fer_pct', 'step_pet_pct'])].round(4).to_string())
    print(I.round(3).to_string())
    # observed numbers for the kernel scenarios (pooled 27 families, NOTES 2026-11-22)
    obs = dict(bore_m7=-0.016, bore_m7_se=0.005, surf_m7=-0.065, surf_m7_se=0.011, surf_excess_m7=-0.049)
    C = part_c(K, obs)
    print(C.round(3).to_string())
    dw = direct_wave_check()
    print(json.dumps(dw, indent=1))
    figures(S, R, LB, I, K, KM)
    kernel_figure(K)
    summ = dict(observed_pooled=obs, direct_wave_check=dw,
                regressions=R.replace({np.nan: None}).to_dict('records'),
                lapse_band=LB[LB.param.isin(['step_m7_pct', 'transient60_pct', 'step_fer_pct', 'step_pet_pct',
                                             'annual_sin_pct', 'annual_cos_pct'])].replace({np.nan: None}).to_dict('records'),
                injection=I.to_dict('records'),
                lapse_band_recovery_corrected=LC.replace({np.nan: None}).to_dict('records'),
                kernel_scenarios=C.to_dict('records'))
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
