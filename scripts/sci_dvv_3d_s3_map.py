#!/usr/bin/env python3
"""Stage 3 lateral maps (revised after the map-script review): Gaussian-process / universal-kriging maps of the
APPARENT shallow dv/v per component, with the lateral correlation length ESTIMATED from the data, and an explicit
network mean.

Per component k, station rows only (SC 5 bands, LFE coda channels, repeater classes; the same amplitudes and
covariance as the station model; LP pairs are NOT used in the fit - they are held out for validation):
   d_j = kappa_j [ fA_j zA(s_j) + fB_j zB(s_j) + fD_j u_g ] + e_j
   zA = muA + gA + nA,   zB = muB + gB + nB   (apparent dv/v in 0-0.3 / 0.3-3 km, parameter units)
   g ~ GP(0, sg^2 exp(-|x-x'|/L)) (coherent field; gB has SD r sg), n ~ iid N(0, sn^2) per station (station-specific;
   nB SD r sn), u ~ N(0, s_k^2) per dataset x band (coda mass below 3 km), e ~ data covariance + 5 % model-error
   variance per row.  The means (muA, muB) are FIXED effects marginalised analytically (REML / universal kriging);
   the network mean is reported as one number +- SE (kernel-weighted 1-3 Hz apparent value).
Hyper-parameters: L on a grid (2.5 ... 80 km), r on {0.25, 0.5, 1, 2, 4}, (sg, sn) optimised continuously (log scale)
   by restricted maximum likelihood; profile 95 % intervals for L, r and for the coherent fraction phi = sg^2 /
   (sg^2 + sn^2).  An optimum at a grid / bound edge -> 'L not bounded'.
Gate ('coherent field'): L bounded, L_lo >= median nearest-station spacing, profile lower bound of phi >= 0.25, and an
   error-weighted permutation test (station values permuted WITHIN sensor class, borehole vs surface) of the
   semivariance for pairs < 20 km with p < 0.05 and >= 10 pairs.  ~1 chance pass in 20 tests is expected.  A failed
   gate means 'station-specific: a coherent field is not demonstrated (not shown to be absent)'.
Maps (ALWAYS exported, per the user's request): PRIMARY = kernel-weighted apparent quantity (1-3 Hz functional fA gA + fB gB
   with the network-median fractions, x kappa_SC of the run, i.e. the same units as the station 'amp'); mean = network
   mean + anomaly; SD from universal kriging (incl. the mean uncertainty) evaluated with the LOWER bound L_lo; masked
   where SD / prior SD > 0.7 or farther than L_lo from the nearest station.  Separate depth-bin maps only if the r
   interval excludes the grid ends and at least one station has a resolved depth verdict for the component; otherwise
   'depth split prior-dominated' (bins fully masked).  Failed gate: no gridded mean (fully masked), plus net_mean,
   net_se and text in the map object.
Date maps: sum over ALL components (station-specific ones listed); SD assumes independent components (ignores
   drop0/level and trend/ramp correlations and stations lacking drop0 rows).
LP validation: every LP pair step (0.1-0.3 Hz, level fields) is predicted from the station-only field (path average,
   LP kernel fractions, network mean included); correlation and chi2 reported.
Outputs: data/sci/dvv_3d/stage3/maps_summary.csv, lp_validation.csv; /data/wsd02/.../stage3/maps.npz; figures.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_3d_s3_inv as S3  # noqa: E402

OUT, BIG = S3.OUT, S3.BIG
LAT0, LON0 = 40.05, -123.56
KMX, KMY = 111.195 * np.cos(np.radians(LAT0)), 111.195
LGRID = [2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0]
RGRID = [0.25, 0.5, 1.0, 2.0, 4.0]
PHIGRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.98]
GLAT = np.round(np.arange(39.45, 40.6501, 0.0225), 4)
GLON = np.round(np.arange(-124.45, -122.9999, 0.03), 4)
LP_MAP = {'M7': 'level_M7', 'Petrolia': 'level_Petrolia', 'Ferndale': 'level_Ferndale'}


def xy(lat, lon):
    return np.c_[(np.asarray(lon) - LON0) * KMX, (np.asarray(lat) - LAT0) * KMY]


class CompData:
    def __init__(self, dat, KF, kappa, comp, ST):
        D = dat.D
        idx = [i for i in np.where((D.amp.values == comp) & (D.dataset.values != 'LP10'))[0] if D.sta.iloc[i] in ST.index]
        rows = []
        for i in idx:
            r = D.iloc[i]
            f = S3.frac_row(KF, r.dataset, r.sta, r.chan, 'default')
            if f is None:
                continue
            rows.append((i, r, f))
        self.ii = np.array([i for i, _, _ in rows])
        self.d = D.est.values[self.ii]
        C = dat.C[np.ix_(self.ii, self.ii)].copy()
        C[np.diag_indices_from(C)] += 0.05 * np.median(np.diag(C))           # model error between bands
        self.Cd = C
        kap = np.array([S3.KAPPA[kappa][r.dataset] for _, r, _ in rows])
        self.a = kap * np.array([f['fA'] for _, _, f in rows])
        self.b = kap * np.array([f['fB'] for _, _, f in rows])
        self.c = kap * np.array([f['fD'] for _, _, f in rows])
        self.sta = np.array([r.sta for _, r, _ in rows], dtype=object)
        self.ust = sorted(set(self.sta))
        self.X = np.c_[self.a, self.b]
        P = xy(ST.loc[self.ust, 'lat'].values, ST.loc[self.ust, 'lon'].values)
        self.sp = P
        si = {s: i for i, s in enumerate(self.ust)}
        self.row_s = np.array([si[s] for s in self.sta])
        self.Dss = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        self.same = self.row_s[:, None] == self.row_s[None, :]
        ug = np.array([f'{r.dataset}|{r.chan if r.dataset == "SC10" else "2-8Hz"}' for _, r, _ in rows], dtype=object)
        self.same_u = ug[:, None] == ug[None, :]
        num = np.abs(self.d)
        den = self.a + self.b
        self.s_k = float(np.median(num[den > 0] / den[den > 0])) if (den > 0).any() else 0.1
        self.n = len(self.d)
        self.kap_sc = S3.KAPPA[kappa]['SC10']
        self.borehole = np.array([s.startswith('PB.') for s in self.ust])

    def S(self, L, sg, sn, r):
        Rss = np.exp(-self.Dss / L)
        R = Rss[np.ix_(self.row_s, self.row_s)]
        KA = sg ** 2 * R + sn ** 2 * self.same
        KB = (r * sg) ** 2 * R + (r * sn) ** 2 * self.same
        return (np.outer(self.a, self.a) * KA + np.outer(self.b, self.b) * KB
                + np.outer(self.c, self.c) * self.s_k ** 2 * self.same_u + self.Cd)


def reml(cd, L, sg, sn, r):
    S = cd.S(L, sg, sn, r)
    try:
        c = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return -np.inf, None
    Si_X = np.linalg.solve(c.T, np.linalg.solve(c, cd.X))
    Si_d = np.linalg.solve(c.T, np.linalg.solve(c, cd.d))
    XtSX = cd.X.T @ Si_X
    try:
        cx = np.linalg.cholesky(XtSX)
    except np.linalg.LinAlgError:
        return -np.inf, None
    beta = np.linalg.solve(XtSX, cd.X.T @ Si_d)
    res = cd.d - cd.X @ beta
    q = float(res @ np.linalg.solve(c.T, np.linalg.solve(c, res)))
    ll = -0.5 * q - np.log(np.diag(c)).sum() - np.log(np.diag(cx)).sum()
    return ll, dict(beta=beta, cov_beta=np.linalg.inv(XtSX))


def opt_sigmas(cd, L, r, phi=None):
    lo, hi = np.log(1e-3 * cd.s_k), np.log(10 * cd.s_k)
    if phi is None:
        def f(p):
            return -reml(cd, L, np.exp(p[0]), np.exp(p[1]), r)[0]
        best = None
        for p0 in ([np.log(cd.s_k), np.log(cd.s_k)], [np.log(0.1 * cd.s_k), np.log(cd.s_k)], [np.log(cd.s_k), np.log(0.1 * cd.s_k)]):
            o = minimize(lambda p: f(np.clip(p, lo, hi)), p0, method='Nelder-Mead', options=dict(xatol=0.02, fatol=0.01, maxiter=300))
            if best is None or o.fun < best.fun:
                best = o
        p = np.clip(best.x, lo, hi)
        edge = bool(p[0] <= lo + 0.05 or p[0] >= hi - 0.05)
        return -best.fun, np.exp(p[0]), np.exp(p[1]), edge
    # fixed coherent fraction phi: optimise the total SD only
    lo_t, hi_t = np.log(1e-3 * cd.s_k), np.log(10 * cd.s_k)

    def g(t):
        tot = np.exp(np.clip(t[0], lo_t, hi_t))
        return -reml(cd, L, tot * np.sqrt(phi), tot * np.sqrt(1 - phi), r)[0]
    o = minimize(g, [np.log(cd.s_k)], method='Nelder-Mead', options=dict(xatol=0.02, fatol=0.01, maxiter=200))
    return -o.fun


def fit_comp(cd):
    tab = {}
    for L in LGRID:
        for r in RGRID:
            ll, sg, sn, edge = opt_sigmas(cd, L, r)
            tab[(L, r)] = (ll, sg, sn, edge)
    (Lb, rb), (llb, sgb, snb, edge_s) = max(tab.items(), key=lambda kv: kv[1][0])
    profL = {L: max(v[0] for (l_, r_), v in tab.items() if l_ == L) for L in LGRID}
    profR = {r: max(v[0] for (l_, r_), v in tab.items() if r_ == r) for r in RGRID}
    okL = [L for L, v in profL.items() if v >= llb - 1.92]
    okR = [r for r, v in profR.items() if v >= llb - 1.92]
    profP = {ph: opt_sigmas(cd, Lb, rb, phi=ph) for ph in PHIGRID}
    okP = [p for p, v in profP.items() if v >= llb - 1.92]
    L_edge = Lb in (LGRID[0], LGRID[-1]) or edge_s
    return dict(L=Lb, r=rb, sg=sgb, sn=snb, ll=llb, L_lo=min(okL), L_hi=max(okL), r_lo=min(okR), r_hi=max(okR),
                phi=sgb ** 2 / (sgb ** 2 + snb ** 2), phi_lo=min(okP) if okP else 0.0, phi_hi=max(okP) if okP else 1.0,
                L_not_bounded=bool(L_edge), profL=profL, profR=profR)


def perm_gate(cd, dmax=20.0, nperm=2000):
    """error-weighted semivariance of per-station apparent values (station-mean residual about the network mean) for
    pairs < dmax, against permutations WITHIN sensor class (borehole / surface)."""
    v = np.zeros(len(cd.ust))
    w = np.zeros(len(cd.ust))
    for i in range(len(cd.ust)):
        m = cd.row_s == i
        sd = np.sqrt(np.diag(cd.Cd)[m])
        den = cd.a[m] + cd.b[m]
        ok = den > 0
        if not ok.any():
            v[i] = np.nan
            continue
        z = cd.d[m][ok] / den[ok]
        ww = (den[ok] / sd[ok]) ** 2
        v[i] = float((ww * z).sum() / ww.sum())
        w[i] = float(ww.sum())
    ok = np.isfinite(v)
    iu = np.triu_indices(ok.sum(), 1)
    Dm = cd.Dss[np.ix_(ok, ok)]
    close = Dm[iu] < dmax
    if close.sum() < 1:
        return dict(p=1.0, n_pairs=0)
    vv, ww, bh = v[ok], w[ok], cd.borehole[ok]

    def stat(x):
        W = 1.0 / (1.0 / ww[:, None] + 1.0 / ww[None, :])
        q = (x[:, None] - x[None, :]) ** 2
        return float((W[iu][close] * q[iu][close]).sum() / W[iu][close].sum())
    s0 = stat(vv)
    rng = np.random.default_rng(3)
    ns = []
    for _ in range(nperm):
        x = vv.copy()
        for cls in (True, False):
            idx = np.where(bh == cls)[0]
            x[idx] = vv[rng.permutation(idx)]
        ns.append(stat(x))
    return dict(p=float(np.mean(np.array(ns) <= s0)), n_pairs=int(close.sum()))


def krige(cd, par, gpts, fA, fB, L_use):
    """universal kriging of the apparent functional kap_sc (fA zA + fB zB) at grid points (mean marginalised)."""
    sg, sn, r = par['sg'], par['sn'], par['r']
    S = cd.S(L_use, sg, sn, r)
    Si = np.linalg.inv(S)
    Rgs = np.exp(-np.sqrt(((gpts[:, None] - cd.sp[None]) ** 2).sum(-1)) / L_use)       # (g, stations)
    Rg = Rgs[:, cd.row_s]                                                               # (g, rows)
    k = cd.kap_sc * (fA * sg ** 2 * Rg * cd.a[None, :] + fB * (r * sg) ** 2 * Rg * cd.b[None, :])
    prior = cd.kap_sc ** 2 * (fA ** 2 * sg ** 2 + fB ** 2 * (r * sg) ** 2)
    XtSi = cd.X.T @ Si
    A = np.linalg.inv(XtSi @ cd.X)
    beta = A @ XtSi @ cd.d
    xs = cd.kap_sc * np.array([fA, fB])
    mean = xs @ beta + k @ Si @ (cd.d - cd.X @ beta)
    u = xs[None, :] - k @ Si @ cd.X
    var = prior - np.einsum('ij,jk,ik->i', k, Si, k) + np.einsum('ij,jk,ik->i', u, A, u)
    sd = np.sqrt(np.maximum(var, 0))
    return mean, sd, np.sqrt(prior), beta, A


def lp_validation(cd, par, comp, ktag, fA13, fB13):
    LP = pd.read_csv(f'{ROOT}/data/sci/dvv_3d/obs_lp.csv')
    q = [k for k, v in LP_MAP.items() if v == comp]
    if not q:
        return []
    LP = LP[LP.band.isin(['0.1-0.2', '0.2-0.3']) & (LP.quantity == q[0])]
    S = pd.read_csv(f'{ROOT}/data/sci/noise_lp/sites.csv').set_index('site')
    LPK = json.load(open(f'{ROOT}/data/sci/dvv_3d/lp_depth_kernels.json'))
    sg, sn, r, L = par['sg'], par['sn'], par['r'], par['L']
    Sm = cd.S(L, sg, sn, r)
    Si = np.linalg.inv(Sm)
    XtSi = cd.X.T @ Si
    beta = np.linalg.solve(XtSi @ cd.X, XtSi @ cd.d)
    resid = Si @ (cd.d - cd.X @ beta)
    out = []
    for rr in LP.itertuples():
        if rr.a not in S.index or rr.b not in S.index:
            continue
        lf = np.array(LPK[rr.band]['layer_frac'])
        fa, fb = lf[0] + lf[1], lf[2] + lf[3]
        p0, p1 = xy(S.loc[rr.a, 'lat'], S.loc[rr.a, 'lon'])[0], xy(S.loc[rr.b, 'lat'], S.loc[rr.b, 'lon'])[0]
        n = max(2, int(np.hypot(*(p1 - p0)) / 3) + 1)
        P = p0[None] + np.linspace(0, 1, n)[:, None] * (p1 - p0)[None]
        Rg = np.exp(-np.sqrt(((P[:, None] - cd.sp[None]) ** 2).sum(-1)) / L).mean(0)[cd.row_s]
        k = fa * sg ** 2 * Rg * cd.a + fb * (r * sg) ** 2 * Rg * cd.b
        pred = fa * beta[0] + fb * beta[1] + k @ resid
        out.append(dict(kappa=ktag, comp=comp, pair=rr.pair, band=rr.band, obs=rr.d, sigma=rr.sigma, pred=pred))
    return out


def run(smoke=False):
    sfx = '_smoke' if smoke else ''
    KF = pd.read_csv(f'{OUT}/kernel_fracs.csv').set_index(['dataset', 'sta', 'band', 'variant'])
    dat = S3.Data(smoke=smoke, calib=False)
    taus = json.load(open(f'{OUT}/tau_net{sfx}.json'))['tau_q']
    ST = pd.read_csv(f'{ROOT}/data/sci/dvv_3d/stations.csv').set_index('ns')
    V = pd.read_csv(f'{OUT}/depth_verdicts{sfx}.csv')
    sp = xy(ST.lat, ST.lon)
    dm = np.sqrt(((sp[:, None] - sp[None]) ** 2).sum(-1)) + np.eye(len(sp)) * 1e9
    nn_med = float(np.median(dm.min(1)))
    fr = pd.read_csv(f'{OUT}/kernel_fracs.csv')
    fr = fr[(fr.dataset == 'SC10') & (fr.band == '1-3Hz') & (fr.variant == 'default')]
    fA13, fB13 = float(fr.fA.median()), float(fr.fB.median())
    GLA, GLO = np.meshgrid(GLAT, GLON, indexing='ij')
    gp = xy(GLA.ravel(), GLO.ravel())
    dist_near = np.sqrt(((gp[:, None] - sp[None]) ** 2).sum(-1)).min(1)
    summ, maps, lpv = [], {}, []
    for ktag in ('k1', 'kprof'):
        for comp in S3.AMPS:
            cd = CompData(dat, KF, ktag, comp, ST)
            if cd.n < 6 or len(cd.ust) < 5:
                continue
            par = fit_comp(cd)
            gate = perm_gate(cd)
            coherent = (not par['L_not_bounded'] and par['L_lo'] >= nn_med and par['phi_lo'] >= 0.25
                        and gate['n_pairs'] >= 10 and gate['p'] < 0.05)
            mean, sd, psd, beta, Ab = krige(cd, par, gp, fA13, fB13, par['L_lo'])
            xs = cd.kap_sc * np.array([fA13, fB13])
            net_mean, net_se = float(xs @ beta), float(np.sqrt(xs @ Ab @ xs))
            text = ('coherent field: surface = network mean + kriged anomaly' if coherent else
                    ('station-specific (L not constrained): surface ~ network mean, anomaly collapsed' if par['L_not_bounded'] else
                     'station-specific: a coherent field is not demonstrated (not shown to be absent); surface ~ network mean + weak anomaly'))
            split_ok = (coherent and par['r_lo'] > RGRID[0] and par['r_hi'] < RGRID[-1]
                        and ((V.amp == comp) & (V.verdict.isin(['top ~0.3 km', 'uniform 0-3 km']))).any())
            layers = {}
            # the APPARENT surface is ALWAYS exported: network mean (marginalised constant) + GP anomaly posterior mean,
            # with its SD; mask only = farther than L_lo from any station (the viewer fades by SD).  For a
            # station-specific component the anomaly collapses and the surface is ~ the network mean.
            mk = dist_near > par['L_lo']
            layers['apparent'] = (mean, sd, mk)
            for lay, (wa, wb) in {'0-0.3': (1.0, 0.0), '0.3-3': (0.0, 1.0)}.items():
                if split_ok:
                    m2, s2, p2, _, _ = krige(cd, par, gp, wa, wb, par['L_lo'])
                    layers[lay] = (m2, s2, dist_near > par['L_lo'])
                else:
                    layers[lay] = (np.full(len(gp), np.nan), np.full(len(gp), np.nan), np.ones(len(gp), bool))
            for lay, (m_, s_, k_) in layers.items():
                maps[f'{ktag}|{comp}|{lay}|mean'] = m_.reshape(GLA.shape).astype(np.float32)
                maps[f'{ktag}|{comp}|{lay}|sd'] = s_.reshape(GLA.shape).astype(np.float32)
                maps[f'{ktag}|{comp}|{lay}|mask'] = k_.reshape(GLA.shape).astype(np.float32)
            maps[f'{ktag}|{comp}|apparent|prior_sd'] = np.float32(np.median(psd))
            summ.append(dict(kappa=ktag, comp=comp, n_rows=cd.n, n_sta=len(cd.ust), L_km=par['L'], L_lo=par['L_lo'], L_hi=par['L_hi'],
                             L_not_bounded=par['L_not_bounded'], r=par['r'], r_lo=par['r_lo'], r_hi=par['r_hi'],
                             sg=par['sg'], sn=par['sn'], phi=par['phi'], phi_lo=par['phi_lo'], phi_hi=par['phi_hi'],
                             perm_p=gate['p'], perm_npairs=gate['n_pairs'], nn_spacing_km=nn_med, coherent=coherent,
                             split_maps=bool(split_ok), net_mean=net_mean, net_se=net_se, text=text,
                             frac_within_Llo=float(1 - layers['apparent'][2].mean()),
                             profL=json.dumps({str(k): round(v, 2) for k, v in par['profL'].items()})))
            lpv += lp_validation(cd, par, comp, ktag, fA13, fB13)
            print(ktag, comp, f"L {par['L']} [{par['L_lo']}-{par['L_hi']}] phi {par['phi']:.2f} [{par['phi_lo']}-{par['phi_hi']}] "
                  f"perm p {gate['p']:.3f} ({gate['n_pairs']} pairs) net {net_mean:+.4f} +- {net_se:.4f} -> {text}", flush=True)
        # date maps: gate-passing components only
        S_ = pd.DataFrame(summ)
        dates = {'2024-11-30': '2024-11-30', '2024-12-20': '2024-12-20', '2025-06-01': '2025-06-01', '2025-12-01': '2025-12-01',
                 'wet (Feb)': '2025-02-15', 'dry (Sep)': '2025-09-15'}
        tf = S3.time_functions(pd.DatetimeIndex(list(dates.values())), taus)
        spec = set(S_[(S_.kappa == ktag) & ~S_.coherent].comp)
        for j, lab in enumerate(dates):
            w = tf[j].copy()
            if lab.startswith(('wet', 'dry')):
                w[2:] = 0.0
            mu = np.zeros(GLA.shape)
            var = np.zeros(GLA.shape)
            used, ss = [], []
            for k, comp in enumerate(S3.AMPS):
                key = f'{ktag}|{comp}|apparent'
                if w[k] == 0 or f'{key}|mean' not in maps:
                    continue
                mu += w[k] * maps[f'{key}|mean'].astype(float)
                var += (w[k] * maps[f'{key}|sd'].astype(float)) ** 2
                used.append(comp)
                if comp in spec:
                    ss.append(comp)
            mask = np.zeros(GLA.shape, bool)
            for comp in used:
                mask |= maps[f'{ktag}|{comp}|apparent|mask'] > 0
            maps[f'{ktag}|date:{lab}|apparent|mean'] = mu.astype(np.float32)
            maps[f'{ktag}|date:{lab}|apparent|sd'] = np.sqrt(var).astype(np.float32)
            maps[f'{ktag}|date:{lab}|apparent|mask'] = mask.astype(np.float32)
            maps[f'{ktag}|date:{lab}|used'] = np.array(used if used else ['none'])
            maps[f'{ktag}|date:{lab}|station_specific'] = np.array(ss if ss else ['none'])
    S_ = pd.DataFrame(summ)
    S_.to_csv(f'{OUT}/maps_summary{sfx}.csv', index=False, float_format='%.5g')
    LV = pd.DataFrame(lpv)
    if len(LV):
        LV.to_csv(f'{OUT}/lp_validation{sfx}.csv', index=False, float_format='%.5g')
        rows = []
        for (k, c, b), g in LV.groupby(['kappa', 'comp', 'band']):
            rows.append(dict(kappa=k, comp=c, band=b, n=len(g), corr=float(np.corrcoef(g.obs, g.pred)[0, 1]) if len(g) > 2 else np.nan,
                             chi2_pred=float((((g.obs - g.pred) / g.sigma) ** 2).sum()), chi2_zero=float(((g.obs / g.sigma) ** 2).sum())))
        pd.DataFrame(rows).to_csv(f'{OUT}/lp_validation_summary{sfx}.csv', index=False, float_format='%.4g')
        print(pd.DataFrame(rows).round(3).to_string())
    np.savez_compressed(f'{BIG}/maps{sfx}.npz', lat=GLAT, lon=GLON, **maps)
    figs(maps, S_, sfx)


def figs(maps, summ, sfx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ST = pd.read_csv(f'{ROOT}/data/sci/dvv_3d/stations.csv')
    P = pd.read_csv(f'{OUT}/predictions{sfx}.csv')
    for ktag in ('k1', 'kprof'):
        comps = [c for c in S3.AMPS if f'{ktag}|{c}|apparent|mean' in maps]
        if not comps:
            continue
        fig, axs = plt.subplots(2, (len(comps) + 1) // 2, figsize=(3.4 * ((len(comps) + 1) // 2), 7.5), squeeze=False)
        for j, c in enumerate(comps):
            ax = axs.ravel()[j]
            m = maps[f'{ktag}|{c}|apparent|mean'].astype(float)
            s = summ[(summ.kappa == ktag) & (summ.comp == c)].iloc[0]
            g = P[(P.kappa == ktag) & (P.date == f'amp:{c}')].merge(ST, left_on='sta', right_on='ns')
            vm = np.nanpercentile(np.abs(np.r_[g['mean'].values, m[np.isfinite(m)]]), 95) if len(g) else 1e-3
            if np.isfinite(m).any():
                ax.pcolormesh(GLON, GLAT, np.ma.masked_invalid(m), cmap='RdBu', vmin=-vm, vmax=vm, shading='auto')
            sc = ax.scatter(g.lon, g.lat, c=g['mean'], cmap='RdBu', vmin=-vm, vmax=vm, s=25 + 20 * np.clip(np.abs(g['mean'] / g['sd']), 0, 4),
                            edgecolors='k', lw=0.4)
            ax.plot(-123.56, 40.05, 'k*', ms=8)
            plt.colorbar(sc, ax=ax, fraction=0.04)
            ax.set_title(f"{c}: {s.text[:40]}\nnet {s.net_mean:+.3f}+-{s.net_se:.3f}; L {s.L_km:g} [{s.L_lo:g}-{s.L_hi:g}] km", fontsize=6)
            ax.tick_params(labelsize=5)
        fig.suptitle(f'Stage-3 apparent 0-3 km dv/v (1-3 Hz functional), kappa {ktag}: station symbols; gridded only where a coherent '
                     f'field passes the gate', fontsize=8)
        fig.tight_layout()
        fig.savefig(f'{ROOT}/figures/sci_dvv_3d_s3_maps_{ktag}{sfx}.png', dpi=110)
        plt.close(fig)


if __name__ == '__main__':
    run(len(sys.argv) > 1 and sys.argv[1] == 'smoke')
