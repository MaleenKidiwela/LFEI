#!/usr/bin/env python
"""Rate-recovery modelling of the post-M7 (2024-12-05) LFE quiescence, per family.

Task 1  per-family fits to daily counts (thin catalog, meancc >= 0.13), -60 d .. +120 d:
        M0   constant rate                                   (1 par: R0)
        BOX  rate eps*R0 for t<T then R0                     (3 par: R0, eps, T)  phenomenological control
        RS   Dieterich (1994) negative step:  R/R0 = 1/[1+(e^D-1) e^{-t/ta}],  D = -Delta/(a sigma) > 0   (3 par)
        DX   diffusion, exp sensitivity: R/R0 = exp(phi(t)), phi = -D erf(sqrt(tau_d/t))                     (3 par)
        RSD  rate-state with the diffusing perturbation phi(t) (Dieterich ODE integrated exactly)              (4 par)
        Diffusion geometry (stated choice): 1-D re-pressurisation from a reservoir at distance r held at the
        pre-quake pressure; the stress deficit decays as Delta0*[1-erfc(r/sqrt(4ct))] = Delta0*erf(sqrt(tau_d/t)),
        tau_d = r^2/(4c).
        Poisson likelihood (R0 profiled analytically).  Overdispersion c_hat from daily counts of the year
        before (Pearson chi2/dof of a constant rate) -> QAIC and quasi-profile-likelihood CIs.
Task 2  spatial organisation of recovery times; r vs r^2 vs none with permutation and quiet-gap placebo;
        joint (shared) diffusivity fits from per-family profile likelihoods; bootstrap over families.
Task 3  t_a = a sigma / tau_dot consistency.
Outputs data/sci/recovery_model/, figures/sci_recovery_model_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal, special, stats

OUT = 'data/sci/recovery_model'
FIG = 'figures'
os.makedirs(OUT, exist_ok=True)
TQ = pd.Timestamp('2024-12-05 18:44:00')
EPOCH = pd.Timestamp('1970-01-01')
MEANCC = 0.13
PRE, POST = 60, 120
H = 0.02                                    # sub-step (d) for integrating rate over daily bins
STRIKE = 293.0
UA = np.array([np.sin(np.radians(STRIKE)), np.cos(np.radians(STRIKE))])   # +WNW
UC = np.array([np.sin(np.radians(STRIKE + 90)), np.cos(np.radians(STRIKE + 90))])  # +NNE (down-dip side)
LAT0, LON0 = 40.06, -123.625
EPI = (40.37, -125.02)
RNG = np.random.default_rng(2024)
FAMS = list(range(1, 28))
DAYS2S = 86400.0


# ------------------------------------------------------------------ data
def load():
    d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
    d['t'] = ((pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) - TQ).dt.total_seconds() + d.s_of_day) / DAYS2S
    d = d.sort_values('t').reset_index(drop=True)
    P = {}
    for F in FAMS:
        j = json.load(open(f'data/locations/fam{F}_3d.json'))
        x, y = (j['lon'] - LON0) * 111.19 * np.cos(np.radians(LAT0)), (j['lat'] - LAT0) * 111.19
        # horizontal great-circle-ish distance to the M7 epicentre (east end of the westward rupture)
        dx = (j['lon'] - EPI[1]) * 111.19 * np.cos(np.radians(0.5 * (j['lat'] + EPI[0])))
        dy = (j['lat'] - EPI[0]) * 111.19
        P[F] = dict(lon=j['lon'], lat=j['lat'], x=x, y=y, z=j['depth_km'], a=x * UA[0] + y * UA[1],
                    c=x * UC[0] + y * UC[1], r_epi=np.hypot(dx, dy))
    P = pd.DataFrame(P).T
    P.index.name = 'fam'
    return d, P


def decluster(th, gap_h=2.0):
    keep = []
    for F in FAMS:
        g = th[th.idnum == F]
        t = g.t.values
        k = np.concatenate([[True], np.diff(t) > gap_h / 24])
        keep.append(g[k])
    return pd.concat(keep).sort_values('t')


def daily_counts(d, declust=False):
    th = d[d.meancc >= MEANCC]
    if declust:
        th = decluster(th)
    edges = np.arange(-PRE, POST + 1e-9, 1.0)
    N = {F: np.histogram(th.t.values[th.idnum.values == F], edges)[0] for F in FAMS}
    # overdispersion from the year before the fit window (-425..-60 d), independent of the models
    e2 = np.arange(-425, -60 + 1e-9, 1.0)
    chat = {}
    for F in FAMS:
        n = np.histogram(th.t.values[th.idnum.values == F], e2)[0]
        m = n.mean()
        chat[F] = float(max(1.0, np.sum((n - m) ** 2 / max(m, 1e-9)) / (len(n) - 1)))
    return N, chat, th


# ------------------------------------------------------------------ models (normalised rate on fine grid)
TF = np.arange(H / 2, POST, H)               # midpoints of sub-steps after the quake
NSUB = int(round(1 / H))


def phi_diff(D, td):
    return -D * special.erf(np.sqrt(td / TF))


def f_rs(D, ta):
    return 1.0 / (1.0 + np.expm1(D) * np.exp(-TF / ta))


def f_dx(D, td):
    return np.exp(phi_diff(D, td))


def slab(t, ts):
    """mid-plane perturbation of a slab (thickness L, faces held at the reservoir value), initial value 1;
    ts = L^2/(pi^2 c).  Short-time erfc series for u<1, eigen-series for u>=1."""
    u = np.asarray(t, float) / ts
    out = np.empty_like(u)
    sm = u < 1
    us = u[sm]
    acc = np.ones_like(us)
    for n in range(12):
        acc -= 2 * (-1) ** n * special.erfc((2 * n + 1) * np.pi / (4 * np.sqrt(us)))
    out[sm] = acc
    ul = u[~sm]
    acc = np.zeros_like(ul)
    for k in range(6):
        acc += (-1) ** k / (2 * k + 1) * np.exp(-(2 * k + 1) ** 2 * ul)
    out[~sm] = 4 / np.pi * acc
    return out


def halfspace(t, td):
    return special.erf(np.sqrt(td / np.asarray(t, float)))


def f_dxs(D, ts):
    return np.exp(-D * slab(TF, ts))


def rs_with_phi(D, ta, shape):
    """Dieterich state evolution with S(t) = Sdot*t + a*sigma*phi(t), phi = -D*shape(t), shape(0+) = 1.
    g = gamma*Sdot; R/R0 = 1/g;  g(t) = e^{-t/ta - phi(t)} [1 + (1/ta) int_0^t e^{t'/ta + phi(t')} dt']
    written as g = e^{-t/ta - phi} + e^{-phi} I(t)/ta,  I_n = e^{-h/ta} I_{n-1} + ta(1-e^{-h/ta}) e^{phi_n}."""
    ph = -D * shape(TF)
    a = np.exp(-H / ta)
    b = ta * (1 - a) * np.exp(ph)
    I = signal.lfilter([1.0], [1.0, -a], b)
    tend = TF + H / 2
    phe = -D * shape(tend)
    g = np.exp(-tend / ta - phe) + np.exp(-phe) * I / ta
    gm = np.concatenate([[np.exp(D)], g])          # g at sub-step edges; average 1/g on the two edges
    return 0.5 * (1 / gm[:-1] + 1 / gm[1:])


def f_rsd(D, ta, td):
    return rs_with_phi(D, ta, lambda t: halfspace(t, td))


def f_rss(D, ta, ts):
    return rs_with_phi(D, ta, lambda t: slab(t, ts))


def binned(f):
    return f.reshape(-1, NSUB).mean(1)        # mean normalised rate per day bin (bin width 1 d)


def loglik_profile(n, fpost):
    """Poisson LL with R0 profiled out. n: counts (PRE+POST bins); fpost: normalised rate per post bin."""
    g = np.concatenate([np.ones(PRE), fpost])
    g = np.maximum(g, 1e-300)
    R0 = n.sum() / g.sum()
    mu = R0 * g
    ll = np.sum(n * np.log(mu) - mu - special.gammaln(n + 1))
    return ll, R0


MODELS = {'RS': (f_rs, ['D', 'ta']), 'DX': (f_dx, ['D', 'td']), 'RSD': (f_rsd, ['D', 'ta', 'td']),
          'DXS': (f_dxs, ['D', 'ts']), 'RSS': (f_rss, ['D', 'ta', 'ts'])}
BOUNDS = {'D': (0.01, 300.0), 'ta': (0.02, 3000.0), 'td': (1e-5, 1e5), 'ts': (0.01, 1e4)}


def nll_factory(n, fn):
    def nll(lp):
        p = np.exp(lp)
        return -loglik_profile(n, binned(fn(*p)))[0]
    return nll


def fit_model(n, name, fixed=None):
    """ML fit; fixed: dict param->value (for profiles)."""
    fn, pars = MODELS[name]
    fixed = fixed or {}
    free = [p for p in pars if p not in fixed]
    lb = np.log([BOUNDS[p][0] for p in free])
    ub = np.log([BOUNDS[p][1] for p in free])

    def full(lp):
        vals, k = [], 0
        for p in pars:
            if p in fixed:
                vals.append(fixed[p])
            else:
                vals.append(np.exp(np.clip(lp[k], lb[k], ub[k])))
                k += 1
        return vals

    def nll(lp):
        return -loglik_profile(n, binned(fn(*full(lp))))[0]

    if not free:
        ll, R0 = loglik_profile(n, binned(fn(*full([]))))
        return dict(ll=ll, R0=R0, **dict(zip(pars, full([]))))
    # coarse grid start
    grids = {'D': np.log([0.3, 1, 3, 6, 10, 20, 60]), 'ta': np.log([0.5, 2, 5, 12, 30, 100, 500]),
             'td': np.log([1e-4, 1e-2, 0.1, 1, 5, 20, 100, 1000, 1e4]),
             'ts': np.log([0.1, 0.5, 1, 2, 4, 8, 16, 50, 300])}
    mesh = np.array(np.meshgrid(*[grids[p] for p in free], indexing='ij')).reshape(len(free), -1).T
    vals = np.array([nll(m) for m in mesh])
    best = None
    for i0 in np.argsort(vals)[:3]:
        r = optimize.minimize(nll, mesh[i0], method='Nelder-Mead',
                              options=dict(xatol=1e-4, fatol=1e-6, maxiter=3000))
        if best is None or r.fun < best.fun:
            best = r
    p = dict(zip(pars, full(best.x)))
    ll, R0 = loglik_profile(n, binned(fn(*full(best.x))))
    return dict(ll=ll, R0=R0, **p)


def fit_box(n):
    best, curve = None, []
    tot = n.sum()
    for T in range(1, 61):
        nin = n[PRE:PRE + T].sum()
        nout = tot - nin
        R0 = nout / (PRE + POST - T)
        rin = max(nin / T, 1e-12)
        mu = np.concatenate([np.full(PRE, R0), np.full(T, rin), np.full(POST - T, R0)])
        ll = np.sum(n * np.log(np.maximum(mu, 1e-300)) - mu - special.gammaln(n + 1))
        curve.append(ll)
        if best is None or ll > best['ll']:
            best = dict(ll=ll, R0=R0, eps=rin / max(R0, 1e-12), T=T)
    best['curve'] = np.array(curve)
    return best


def fit_const(n):
    R0 = n.mean()
    return dict(ll=np.sum(n * np.log(max(R0, 1e-300)) - R0 - special.gammaln(n + 1)), R0=R0)


def profile(n, name, par, grid):
    out = []
    for v in grid:
        out.append(fit_model(n, name, fixed={par: v})['ll'])
    return np.array(out)


def ci_from_profile(grid, ll, chat, llmax):
    ok = 2 * (llmax - ll) / chat <= 3.84
    if not ok.any():
        return [np.nan, np.nan]
    lo, hi = grid[ok].min(), grid[ok].max()
    lo = lo if lo > grid[0] else 0.0          # open interval -> report bound
    hi = hi if hi < grid[-1] else np.inf
    return [float(lo), float(hi)]


# ------------------------------------------------------------------ task 1
TA_GRID = np.exp(np.linspace(np.log(0.1), np.log(2000), 49))
TD_GRID = np.exp(np.linspace(np.log(1e-4), np.log(3e4), 57))
TS_GRID = np.exp(np.linspace(np.log(0.03), np.log(3000), 41))
D_GRID = np.exp(np.linspace(np.log(0.1), np.log(200), 45))


def t_half(name, p):
    """time after the M7 at which the model rate first reaches 0.5 R0."""
    fn, pars = MODELS[name]
    f = fn(*[p[k] for k in pars])
    k = np.where(f >= 0.5)[0]
    return float(TF[k[0]]) if len(k) else np.inf


def fit_family(args):
    F, n, ch = args
    r = dict(fam=F, n_pre=int(n[:PRE].sum()), n_post=int(n[PRE:].sum()), c_hat=ch)
    m0 = fit_const(n)
    bx = fit_box(n)
    fits = {k: fit_model(n, k) for k in MODELS}
    r.update(ll_M0=m0['ll'], ll_BOX=bx['ll'], box_T=bx['T'], box_eps=bx['eps'])
    for k, v in fits.items():
        r[f'll_{k}'] = v['ll']
        r[f'R0_{k}'] = v['R0']
        for p in MODELS[k][1]:
            r[f'{k}_{p}'] = v[p]
        r[f't50_{k}'] = t_half(k, v)
    npar = dict(M0=1, BOX=3, RS=3, DX=3, RSD=4, DXS=3, RSS=4)
    for k in npar:
        r[f'AIC_{k}'] = -2 * r[f'll_{k}'] + 2 * npar[k]
        r[f'QAIC_{k}'] = -2 * r[f'll_{k}'] / ch + 2 * (npar[k] + 1)
    f = binned(f_rs(r['RS_D'], r['RS_ta']))
    mu = r['R0_RS'] * np.concatenate([np.ones(PRE), f])
    r['c_hat_fitwindow_RS'] = float(np.sum((n - mu) ** 2 / np.maximum(mu, 1e-9)) / (len(n) - 3))
    pr = dict(RS_ta=profile(n, 'RS', 'ta', TA_GRID), RS_D=profile(n, 'RS', 'D', D_GRID),
              DX_td=profile(n, 'DX', 'td', TD_GRID), RSD_td=profile(n, 'RSD', 'td', TD_GRID),
              DXS_ts=profile(n, 'DXS', 'ts', TS_GRID), RSS_ts=profile(n, 'RSS', 'ts', TS_GRID),
              BOX_T=bx['curve'])
    r['RS_ta_ci'] = ci_from_profile(TA_GRID, pr['RS_ta'], ch, r['ll_RS'])
    r['RS_D_ci'] = ci_from_profile(D_GRID, pr['RS_D'], ch, r['ll_RS'])
    r['DX_td_ci'] = ci_from_profile(TD_GRID, pr['DX_td'], ch, r['ll_DX'])
    r['RSD_td_ci'] = ci_from_profile(TD_GRID, pr['RSD_td'], ch, r['ll_RSD'])
    r['DXS_ts_ci'] = ci_from_profile(TS_GRID, pr['DXS_ts'], ch, r['ll_DXS'])
    r['RSS_ts_ci'] = ci_from_profile(TS_GRID, pr['RSS_ts'], ch, r['ll_RSS'])
    r['box_T_ci'] = ci_from_profile(np.arange(1, 61, dtype=float), pr['BOX_T'], ch, r['ll_BOX'])
    print(F, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
              if k.startswith(('RS_', 'DX_', 'RSD_', 't50', 'QAIC'))}, flush=True)
    return r, pr


def task1(N, chat, tag=''):
    from multiprocessing import Pool
    with Pool(4) as pool:
        res = pool.map(fit_family, [(F, N[F], chat[F]) for F in FAMS])
    T = pd.DataFrame([r for r, _ in res])
    np.savez(f'{OUT}/profiles{tag}.npz', TA_GRID=TA_GRID, TD_GRID=TD_GRID, D_GRID=D_GRID, TS_GRID=TS_GRID,
             **{f'{k}_{r["fam"]}': v for r, pr in res for k, v in pr.items()})
    return T


# ------------------------------------------------------------------ task 2: space
KAPPA50 = special.erfcinv(0.5) ** 2          # erfc(r/sqrt(4ct)) = 0.5  ->  t = r^2 / (4 c KAPPA50)
KM2D_TO_M2S = 1e6 / DAYS2S


def fam_first_after(d, t0, fams=FAMS):
    out = []
    for F in fams:
        x = d.t.values[d.idnum.values == F]
        k = np.searchsorted(x, t0, side='right')
        out.append(x[k] - t0 if k < len(x) else np.nan)
    return np.array(out)


def design_1d(a, grid):
    """basis f(r) for source at a0 (grid) along strike: r and r^2, shape (ngrid, nfam)."""
    r = np.abs(a[None, :] - grid[:, None])
    return {'r': r, 'r2': r ** 2}


def design_2d(x, y, gx, gy):
    X0, Y0 = np.meshgrid(gx, gy, indexing='ij')
    r = np.hypot(x[None, :] - X0.ravel()[:, None], y[None, :] - Y0.ravel()[:, None])
    return {'r': r, 'r2': r ** 2}, X0.ravel(), Y0.ravel()


def fit_family_of_models(Y, B1, B2, a):
    """Y: (m, n) metric rows (obs or permutations).  Returns dict model -> (AIC (m,), argbest (m,), slope (m,), t0 (m,)).
    Gaussian AIC = n ln(RSS/n) + 2k (k incl. sigma)."""
    m, n = Y.shape
    out = {}
    Yc = Y - Y.mean(1, keepdims=True)
    Syy = (Yc ** 2).sum(1)
    out['const'] = (n * np.log(Syy / n) + 2 * 2, np.zeros(m, int), np.zeros(m), Y.mean(1))
    ac = a - a.mean()
    b = Yc @ ac / (ac @ ac)
    rss = Syy - b ** 2 * (ac @ ac)
    out['linear_along'] = (n * np.log(rss / n) + 2 * 3, np.zeros(m, int), b, Y.mean(1) - b * a.mean())
    for lab, B, npar_src in [('1d', B1, 1), ('2d', B2, 2)]:
        if B is None:
            continue
        for kind in ['r', 'r2']:
            Fm = B[kind]                                   # (g, n)
            # pure (through origin): y = k f
            Sff = (Fm ** 2).sum(1)                         # (g,)
            Syf = Y @ Fm.T                                 # (m, g)
            rss = (Y ** 2).sum(1)[:, None] - Syf ** 2 / Sff[None]
            rss = np.where(Syf > 0, rss, np.inf)           # source, not sink: delay must grow with distance
            j = np.argmin(rss, 1)
            k = Syf[np.arange(m), j] / Sff[j]
            out[f'pure_{kind}_{lab}'] = (n * np.log(rss[np.arange(m), j] / n) + 2 * (2 + npar_src), j, k, np.zeros(m))
            # with delay: y = t0 + k f
            Fc = Fm - Fm.mean(1, keepdims=True)
            Sff = (Fc ** 2).sum(1)
            Syf = Yc @ Fc.T
            rss = Syy[:, None] - Syf ** 2 / Sff[None]
            rss = np.where(Syf > 0, rss, np.inf)
            j = np.argmin(rss, 1)
            k = Syf[np.arange(m), j] / Sff[j]
            t0 = Y.mean(1) - k * Fm[j].mean(1)
            out[f'delay_{kind}_{lab}'] = (n * np.log(rss[np.arange(m), j] / n) + 2 * (3 + npar_src), j, k, t0)
    return out


def spatial_tests(y, P, nperm, rng, G1, B1, B2, ok=None):
    ok = np.isfinite(y) if ok is None else ok
    a = P.a.values.astype(float)[ok]
    yy = y[ok]
    B1o = {k: v[:, ok] for k, v in B1.items()}
    B2o = {k: v[:, ok] for k, v in B2.items()}
    Y = np.vstack([yy] + [rng.permutation(yy) for _ in range(nperm)])
    res = fit_family_of_models(Y, B1o, B2o, a)
    base = res['const'][0]
    summ = {}
    for k, (aic, j, slope, t0) in res.items():
        d_obs = aic[0] - base[0]
        d_null = aic[1:] - base[1:]
        summ[k] = dict(dAIC_vs_const=float(d_obs), p_perm=float((np.sum(d_null <= d_obs) + 1) / (nperm + 1)),
                       slope=float(slope[0]), t0=float(t0[0]), src_index=int(j[0]))
    return summ, res


def task2(d, P, T, chat):
    rng = np.random.default_rng(7)
    R = pd.read_csv('data/sci/cascades2/restart_M7.csv').set_index('fam').loc[FAMS]
    s3 = json.load(open('data/sci/cascades2/summary_t3.json'))['t3_restart']['rate_normalised']
    mw = np.array([s3['median_wait_after_random_time_d'][str(F)] for F in FAMS])
    a = P.a.values.astype(float)
    G1 = np.arange(a.min() - 40, a.max() + 40.01, 0.25)
    B1 = design_1d(a, G1)
    gx = np.arange(P.x.min() - 30, P.x.max() + 30.01, 1.0)
    gy = np.arange(P.y.min() - 30, P.y.max() + 30.01, 1.0)
    B2, X0, Y0 = design_2d(P.x.values.astype(float), P.y.values.astype(float), gx, gy)
    Tm = T.set_index('fam').loc[FAMS]
    metrics = {'t50_RS_thin': Tm.t50_RS.values, 'box_T_thin': Tm.box_T.values.astype(float),
               'first_after_full': R.first_after_d.values, 'first_after_over_median_wait': R.first_after_d.values / mw,
               'RS_ta_upper95': np.array([c[1] for c in Tm.RS_ta_ci]), 'DXS_ts': Tm.DXS_ts.values}
    coords = {'along_km(+WNW)': P.a.values, 'lon': P.lon.values, 'depth_km': P.z.values, 'cross_km(+NNE,downdip)': P.c.values,
              'r_epi_km': P.r_epi.values}
    out = {'coord_collinearity_spearman': {
        'along_vs_lon': float(stats.spearmanr(P.a, P.lon)[0]), 'r_epi_vs_lon': float(stats.spearmanr(P.r_epi, P.lon)[0]),
        'depth_vs_cross': float(stats.spearmanr(P.z, P.c)[0])}}
    # rank correlations with permutation p
    rows = []
    for mk, y in metrics.items():
        ok = np.isfinite(y)
        for ck, x in coords.items():
            rho = stats.spearmanr(y[ok], x[ok])[0]
            null = np.array([stats.spearmanr(rng.permutation(y[ok]), x[ok])[0] for _ in range(2000)])
            rows.append(dict(metric=mk, coord=ck, n=int(ok.sum()), spearman=float(rho),
                             p_perm=float((np.sum(np.abs(null) >= abs(rho)) + 1) / 2001)))
    C = pd.DataFrame(rows)
    C.to_csv(f'{OUT}/spatial_rank_correlations.csv', index=False)
    print(C.round(3).to_string())
    # rate confound of the normalised delay
    out['rate_confound'] = dict(
        rho_median_wait_vs_along=float(stats.spearmanr(mw, P.a)[0]),
        rho_first_after_full_vs_along=float(stats.spearmanr(R.first_after_d, P.a)[0]),
        rho_ratio_vs_along=float(stats.spearmanr(R.first_after_d / mw, P.a)[0]),
        rho_ratio_vs_1_over_median_wait=float(stats.spearmanr(R.first_after_d / mw, 1 / mw)[0]),
        n_restart_in_17p5_18p8_d_full=int(((R.first_after_d > 17.5) & (R.first_after_d < 18.8)).sum()),
        n_restart_in_17p5_18p8_d_thin=None)
    # structural model comparison (const / linear / front r / diffusion r^2), permutation null
    model_rows = []
    allres = {}
    for mk in ['t50_RS_thin', 'first_after_full', 'first_after_over_median_wait']:
        y = metrics[mk]
        summ, res = spatial_tests(y, P, 1999, rng, G1, B1, B2)
        allres[mk] = res
        for k, v in summ.items():
            row = dict(metric=mk, model=k, **v)
            if k.endswith('1d'):
                row['src_along_km'] = float(G1[v['src_index']])
                row['src_at_grid_edge'] = bool(v['src_index'] in (0, len(G1) - 1))
            elif k.endswith('2d'):
                row['src_x_km'], row['src_y_km'] = float(X0[v['src_index']]), float(Y0[v['src_index']])
            if mk == 'first_after_over_median_wait':
                pass                                       # dimensionless metric: no diffusivity / speed
            elif '_r2_' in k and v['slope'] > 0:
                row['c_m2_s_50pct'] = float(1 / (4 * KAPPA50 * v['slope']) * KM2D_TO_M2S)
            if '_r_' in k and v['slope'] > 0 and mk != 'first_after_over_median_wait':
                row['front_speed_km_d'] = float(1 / v['slope'])
            model_rows.append(row)
    M = pd.DataFrame(model_rows)
    # placebo: other network-wide quiet gaps (same first-after metric, full catalog)
    G = pd.read_csv('data/sci/cascades2/restart_quiet_gaps.csv')
    tgap = ((pd.to_datetime(G.gap_start) - TQ).dt.total_seconds() / DAYS2S).values
    PL = []
    for tg in tgap:
        fa = fam_first_after(d, tg + 1e-6)
        if np.isnan(fa).any():
            continue
        for lab, y in [('first_after_full', fa), ('first_after_over_median_wait', fa / mw)]:
            Y = y[None]
            res = fit_family_of_models(Y, B1, B2, a)
            base = res['const'][0][0]
            for k, v in res.items():
                PL.append(dict(gap_t_rel_M7_d=tg, metric=lab, model=k, dAIC_vs_const=float(v[0][0] - base),
                               slope=float(v[2][0]), rho_along=float(stats.spearmanr(y, a)[0])))
    PL = pd.DataFrame(PL)
    PL.to_csv(f'{OUT}/placebo_quiet_gaps_models.csv', index=False)

    def nsync(fa, w=1.3):
        fa = np.sort(fa[np.isfinite(fa)])
        return int(max(np.searchsorted(fa, x + w, side='right') - i for i, x in enumerate(fa)))
    ns_gaps = []
    for tg in tgap:
        fa = fam_first_after(d, tg + 1e-6)
        if not np.isnan(fa).any():
            ns_gaps.append(dict(n=nsync(fa), n_rel=nsync(fa / np.median(fa), 1.3 / np.median(R.first_after_d.values))))
    ns_gaps = pd.DataFrame(ns_gaps)
    ns_m7 = nsync(R.first_after_d.values)
    out['restart_synchrony'] = dict(
        definition='max number of families whose first post-gap event falls in one 1.3-d window (full catalog)',
        M7=ns_m7, quiet_gaps_median=float(ns_gaps.n.median()), quiet_gaps_range=[int(ns_gaps.n.min()), int(ns_gaps.n.max())],
        p_quietgaps_ge=float((np.sum(ns_gaps.n >= ns_m7) + 1) / (len(ns_gaps) + 1)),
        window_scaled_to_gap_median=dict(M7=ns_m7, gaps_median=float(ns_gaps.n_rel.median()),
                                         p=float((np.sum(ns_gaps.n_rel >= ns_m7) + 1) / (len(ns_gaps) + 1))),
        note='first-after times after quiet gaps are ~10x shorter; the scaled variant uses a window of 1.3 d x (gap median/M7 median)')
    M['p_placebo_quietgaps'] = np.nan
    M['n_placebo'] = 0
    for i, r in M.iterrows():
        q = PL[(PL.metric == r.metric) & (PL.model == r.model)]
        if len(q):
            M.loc[i, 'p_placebo_quietgaps'] = (np.sum(q.dAIC_vs_const <= r.dAIC_vs_const) + 1) / (len(q) + 1)
            M.loc[i, 'n_placebo'] = len(q)
    M.to_csv(f'{OUT}/spatial_model_comparison.csv', index=False)
    print(M.round(3).to_string())
    # bootstrap (over families) of the single diffusivity from the pure-r^2 1-D model on t50
    boots = {}
    for mk in ['t50_RS_thin', 'first_after_full']:
        y = metrics[mk]
        cs, v_s, a0s = [], [], []
        for _ in range(2000):
            s = rng.integers(0, 27, 27)
            Bs = {k: v[:, s] for k, v in B1.items()}
            res = fit_family_of_models(y[s][None], Bs, None, a[s])
            k2 = res['pure_r2_1d'][2][0]
            cs.append(1 / (4 * KAPPA50 * k2) * KM2D_TO_M2S if k2 > 0 else np.nan)
            a0s.append(G1[res['pure_r2_1d'][1][0]])
            k1 = res['pure_r_1d'][2][0]
            v_s.append(1 / k1 if k1 > 0 else np.nan)
        boots[mk] = dict(c_m2_s_pure_r2_1d_median=float(np.nanmedian(cs)),
                         c_m2_s_ci95=[float(np.nanpercentile(cs, 2.5)), float(np.nanpercentile(cs, 97.5))],
                         src_along_km_ci95=[float(np.percentile(a0s, 2.5)), float(np.percentile(a0s, 97.5))],
                         front_speed_km_d_ci95=[float(np.nanpercentile(v_s, 2.5)), float(np.nanpercentile(v_s, 97.5))])
    out['diffusivity_from_restart_geometry_bootstrap'] = boots
    out['metrics_table'] = pd.DataFrame(dict(fam=FAMS, along_km=P.a.values, depth_km=P.z.values, **metrics)).round(4).to_dict('records')
    return out, C, M, PL, metrics, allres, (G1, X0, Y0)


# ------------------------------------------------------------------ joint (shared-parameter) fits from profiles
def joint_profiles(T, chat):
    Z = np.load(f'{OUT}/profiles.npz')
    out = {}
    rng = np.random.default_rng(11)
    Tm = T.set_index('fam')
    for key, grid, llcol, npar in [('RS_ta', Z['TA_GRID'], 'll_RS', 3), ('RS_D', Z['D_GRID'], 'll_RS', 3),
                                   ('DX_td', Z['TD_GRID'], 'll_DX', 3), ('DXS_ts', Z['TS_GRID'], 'll_DXS', 3),
                                   ('RSS_ts', Z['TS_GRID'], 'll_RSS', 4)]:
        Q = np.array([Z[f'{key}_{F}'] / chat[F] for F in FAMS])            # quasi-LL profiles (27, g)
        qmax = np.array([Tm.loc[F, llcol] / chat[F] for F in FAMS])
        qmax = np.maximum(qmax, Q.max(1))
        Qs = Q
        tot = Qs.sum(0)
        j = np.argmax(tot)
        ok = 2 * (tot[j] - tot) <= 3.84
        # QAIC: per-family parameter (27 values) vs one common value
        qaic_free = -2 * qmax.sum() + 2 * 27 * npar
        qaic_common = -2 * tot[j] + 2 * (27 * (npar - 1) + 1)
        bs = []
        for _ in range(2000):
            s = rng.integers(0, 27, 27)
            bs.append(grid[np.argmax(Qs[s].sum(0))])
        out[key] = dict(common_best=float(grid[j]), common_ci95_profile=[float(grid[ok].min()), float(grid[ok].max())],
                        common_ci95_boot_fam=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        QAIC_common_minus_free=float(qaic_common - qaic_free),
                        dev_common_minus_free=float(2 * (qmax.sum() - tot[j])), dof=int(26))
        out[key]['p_chi2_common'] = float(stats.chi2.sf(out[key]['dev_common_minus_free'], 26))
        out[key]['profile_sum'] = tot.tolist()
        out[key]['grid'] = grid.tolist()
    return out


# ------------------------------------------------------------------ task 3: consistency
def task3(T, J, N, chat):
    MU = 30e9
    YRD = 365.25
    Tm = T.set_index('fam').loc[FAMS]
    t50 = Tm.t50_RS.values
    out = {}
    # (a) quiescence duration in RS: t* ~ ta ln(e^D - 1) ~ |Delta|/tau_dot  (independent of a sigma)
    for lab, dl in [('Delta_eff_clamp_3p5kPa', 3.5), ('Delta_range_lo_2kPa', 2.0), ('Delta_range_hi_4p6kPa', 4.6)]:
        td = dl / t50                                                      # kPa/day
        out[f'tau_dot_required_{lab}'] = dict(median_kPa_per_yr=float(np.median(td) * YRD),
                                              iqr_kPa_per_yr=[float(np.percentile(td, 25) * YRD), float(np.percentile(td, 75) * YRD)])
    # (b) loading-rate estimates from 2-3 cm/yr slip
    load = {}
    for v_cm in [2.0, 3.0]:
        v = v_cm / 100
        for W in [5e3, 15e3, 30e3]:
            load[f'zone_mu_v_over_W_v{v_cm:g}cm_W{W / 1e3:g}km_kPa_yr'] = MU * v / W / 1e3
        for Rp in [250.0, 500.0]:
            load[f'patch_crack_7pi16_mu_v_over_R_v{v_cm:g}cm_R{Rp:g}m_kPa_yr'] = 7 * np.pi / 16 * MU * v / Rp / 1e3
    out['loading_rate_estimates'] = load
    out['t_star_predicted_d_for_3p5kPa'] = {k: 3.5 / (v / YRD) for k, v in load.items()}
    # (c) a sigma from ta (=a sigma/tau_dot) and from D (=|Delta|/a sigma)
    ta_up = np.array([c[1] for c in Tm.RS_ta_ci])
    D_lo = np.array([c[0] for c in Tm.RS_D_ci])
    ta_common = J['RS_ta']
    D_common = J['RS_D']
    asig = {}
    for tdot_lab, tdot_yr in [('tau_dot_50kPa_yr(zone,15km,2.5cm/yr)', 50.0), ('tau_dot_required_~70kPa_yr', 3.5 / np.median(t50) * YRD),
                              ('tau_dot_4MPa_yr(patch)', 4000.0)]:
        tdot = tdot_yr / YRD
        asig[tdot_lab] = dict(
            a_sigma_kPa_from_common_ta_best=float(tdot * ta_common['common_best']),
            a_sigma_kPa_upper95_from_common_ta=float(tdot * ta_common['common_ci95_profile'][1]),
            a_sigma_kPa_upper95_per_family_median=float(np.median(tdot * ta_up[np.isfinite(ta_up)])))
    asig['from_D_with_Delta_3p5kPa'] = dict(
        a_sigma_kPa_common_D_best=float(3.5 / D_common['common_best']),
        a_sigma_kPa_upper95_common_D=float(3.5 / D_common['common_ci95_profile'][0]),
        a_sigma_kPa_upper95_per_family_median=float(np.median(3.5 / D_lo[D_lo > 0])),
        fam23_two_sided=[float(3.5 / Tm.loc[23, 'RS_D_ci'][1]), float(3.5 / Tm.loc[23, 'RS_D_ci'][0])])
    out['a_sigma'] = asig
    a_up = asig['from_D_with_Delta_3p5kPa']['a_sigma_kPa_upper95_per_family_median']
    out['implied'] = dict(a_sigma_upper_used_kPa=float(a_up), a_upper_if_sigma_eff_ge_6kPa=float(a_up / 6.0),
                          sigma_eff_kPa_upper_if_a_0p005_0p01_0p02=[float(a_up / a) for a in [0.005, 0.01, 0.02]],
                          tidal_a_sigma_0p46_sigma_eff_if_a_0p005_0p01_0p02=[0.4623 / a for a in [0.005, 0.01, 0.02]],
                          note=('per-family median 95% upper bound (pooled common-D bound is driven by the abrupt, episode-led restart '
                                'and is not used); sigma_eff <= a_sigma_upper / a'))
    tid = 'data/sci/tidal_stress'
    out['tidal_stress_dir_present'] = os.path.isdir(tid)
    if os.path.exists(f'{tid}/run.log'):
        for line in open(f'{tid}/run.log'):
            if line.startswith("tau {"):
                out['tidal_a_sigma_line'] = line.strip()
    # RS with D fixed by the tidal a sigma (0.46 kPa, shear-stress model in data/sci/tidal_stress/run.log)
    asig_tide = 0.4623
    tide = {}
    for dl in [2.0, 3.5, 4.6]:
        Dfix = dl / asig_tide
        rows = []
        for F in FAMS:
            fz = fit_model(N[F], 'RS', fixed={'D': Dfix})
            pr = np.array([fit_model(N[F], 'RS', fixed={'D': Dfix, 'ta': x})['ll'] for x in TA_GRID]) / chat[F]
            rows.append((fz['ta'], fz['ll'], pr))
        tas = np.array([r[0] for r in rows])
        llfix = np.array([r[1] for r in rows])
        Q = np.array([r[2] for r in rows])
        tot = Q.sum(0)
        j = np.argmax(tot)
        ok = 2 * (tot[j] - tot) <= 3.84
        bs = [TA_GRID[np.argmax(Q[rng_s].sum(0))] for rng_s in np.random.default_rng(5).integers(0, 27, (2000, 27))]
        dq = float(np.sum((-2 * llfix / np.array([chat[F] for F in FAMS]) + 2 * 3) - Tm.QAIC_RS.values))
        ta_c = float(TA_GRID[j])
        tide[f'Delta_{dl:g}kPa'] = dict(D_fixed=Dfix, ta_per_family_median=float(np.median(tas)),
                                         ta_common=ta_c, ta_common_ci95_profile=[float(TA_GRID[ok].min()), float(TA_GRID[ok].max())],
                                         ta_common_ci95_boot_fam=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                         tau_dot_kPa_yr_common=asig_tide / ta_c * YRD,
                                         sum_QAIC_minus_freeRS=dq)
    out['RS_with_tidal_a_sigma'] = tide
    if os.path.isdir(tid):
        out['tidal_stress_files'] = sorted(os.listdir(tid))
    # (d) overshoot: observed vs RS-predicted counts from t50 to t50+30 d (pooled)
    obs, pred, var = 0.0, 0.0, 0.0
    per = []
    for F in FAMS:
        r = Tm.loc[F]
        f = binned(f_rs(r.RS_D, r.RS_ta))
        mu = r.R0_RS * f
        k0 = int(np.ceil(r.t50_RS))
        o, p_ = N[F][PRE + k0:PRE + k0 + 30].sum(), mu[k0:k0 + 30].sum()
        obs += o
        pred += p_
        var += chat[F] * p_
        per.append(o / p_)
    out['post_restart_30d_obs_over_RSpred'] = dict(pooled=float(obs / pred), se=float(np.sqrt(var) / pred),
                                                   median_family=float(np.median(per)), frac_fams_gt1=float(np.mean(np.array(per) > 1)))
    # pre-M7 baseline for the same comparison: R0 fitted includes post data, so also compare to pre-60-d rate only
    o2 = sum(N[F][PRE + int(np.ceil(Tm.loc[F, 't50_RS'])):PRE + int(np.ceil(Tm.loc[F, 't50_RS'])) + 30].sum() for F in FAMS)
    p2 = sum(N[F][:PRE].sum() / PRE * 30 for F in FAMS)
    out['post_restart_30d_obs_over_pre60d_rate'] = float(o2 / p2)
    return out


# ------------------------------------------------------------------ figures
def figures(T, N, P, metrics, M, PL, J, geo):
    Tm = T.set_index('fam').loc[FAMS]
    order = np.argsort(-P.a.values)                       # WNW -> ESE
    fig, axs = plt.subplots(6, 5, figsize=(17, 17), sharex=True)
    tb = np.arange(-PRE, POST) + 0.5
    for ax, i in zip(axs.ravel(), order):
        F = FAMS[i]
        r = Tm.loc[F]
        n = N[F]
        k = 3                                             # display in 3-day bins
        nb = n.reshape(-1, k).sum(1) / k
        ax.bar(tb.reshape(-1, k).mean(1), nb, width=k, color='0.75', edgecolor='none')
        for lab, fn, pars, col, ls in [('RS', f_rs, ['D', 'ta'], 'C3', '-'), ('DXS', f_dxs, ['D', 'ts'], 'C2', '--'),
                                       ('DX', f_dx, ['D', 'td'], 'C0', ':')]:
            f = binned(fn(*[r[f'{lab}_{p}'] for p in pars]))
            ax.plot(np.concatenate([[-PRE, 0], tb[PRE:]]), r[f'R0_{lab}'] * np.concatenate([[1, 1], f]), color=col, ls=ls, lw=1.3,
                    label=f'{lab} dQAIC={r[f"QAIC_{lab}"] - r.QAIC_M0:+.0f}')
        ax.axvline(0, color='k', lw=0.6)
        ax.set_title(f'fam {F}  along {P.a.values[i]:+.1f} km  t50={r.t50_RS:.1f} d', fontsize=8)
        ax.legend(fontsize=6, loc='upper left', frameon=False)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[27:]:
        ax.axis('off')
    axs.ravel()[27].text(0, 0.5, 'grey: daily counts (meancc>=0.13), 3-d means\nRS: Dieterich negative step\n'
                         'DXS: slab diffusion, exp sensitivity\nDX: half-space (erf) diffusion, exp sensitivity\n'
                         'dQAIC vs constant rate (c_hat from prior year)\npanels ordered WNW -> ESE', fontsize=9, va='center')
    fig.supxlabel('days after 2024-12-05 M7')
    fig.supylabel('LFEs per day')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_recovery_model_fits.png', dpi=110)
    plt.close(fig)

    # recovery time vs position
    G1, X0, Y0 = geo
    a = P.a.values.astype(float)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ci = np.array([c for c in Tm.box_T_ci])
    for j, (mk, xk, xl) in enumerate([('t50_RS_thin', 'a', 'along strike (km, + WNW, toward M7)'),
                                      ('t50_RS_thin', 'z', 'depth (km)'),
                                      ('first_after_full', 'a', 'along strike (km, + WNW)')]):
        A = ax[0, j]
        x = P[xk].values.astype(float)
        y = metrics[mk]
        if mk == 't50_RS_thin':
            A.errorbar(x, y, yerr=[np.clip(y - ci[:, 0], 0, None), np.clip(ci[:, 1] - y, 0, None)], fmt='o', color='C3', ms=4,
                       ecolor='0.6', lw=0.8, label='t50 (RS) with BOX switch-time 95% CI')
        else:
            A.plot(x, y, 'o', color='C1', ms=4, label=mk)
        for F, xi, yi in zip(FAMS, x, y):
            A.text(xi, yi, f' {F}', fontsize=6)
        if xk == 'a':
            mm = M[(M.metric == mk)].set_index('model')
            xs = np.linspace(a.min() - 1, a.max() + 1, 200)
            for mod, col in [('delay_r_1d', 'C0'), ('delay_r2_1d', 'C2'), ('linear_along', 'k')]:
                r = mm.loc[mod]
                if mod == 'linear_along':
                    ys = r.t0 + r.slope * xs
                else:
                    rr = np.abs(xs - r.src_along_km)
                    ys = r.t0 + r.slope * (rr if '_r_' in mod else rr ** 2)
                A.plot(xs, ys, color=col, lw=1, label=f'{mod}: dAIC={r.dAIC_vs_const:+.1f}, p_perm={r.p_perm:.2f}, p_plac={r.p_placebo_quietgaps:.2f}')
        A.set_xlabel(xl)
        A.set_ylabel(f'{mk} (d)')
        A.legend(fontsize=6)
    # ratio vs along
    A = ax[1, 0]
    y = metrics['first_after_over_median_wait']
    A.scatter(a, y, c=1 / np.array([1.0] * 27), s=0)
    sc = A.scatter(a, y, c=np.log10(metrics['first_after_full'] / y), cmap='viridis', s=25)
    plt.colorbar(sc, ax=A, label='log10 median wait (d)')
    A.set_xlabel('along strike (km, + WNW)')
    A.set_ylabel('first_after / median wait')
    A.set_title('normalised delay: dominated by the wait (rate) term', fontsize=9)
    # placebo distributions
    A = ax[1, 1]
    for mod, col in [('delay_r_1d', 'C0'), ('delay_r2_1d', 'C2'), ('linear_along', 'k')]:
        q = PL[(PL.metric == 'first_after_full') & (PL.model == mod)].dAIC_vs_const
        q = q[np.isfinite(q)]
        A.hist(q, bins=15, histtype='step', color=col, label=f'quiet gaps: {mod}')
        o = M[(M.metric == 'first_after_full') & (M.model == mod)].dAIC_vs_const.values[0]
        A.axvline(o, color=col, lw=2)
    A.set_xlabel('dAIC vs constant (first_after_full); lines = M7')
    A.legend(fontsize=7)
    # joint profiles
    A = ax[1, 2]
    for key, col in [('RS_ta', 'C3'), ('DXS_ts', 'C2'), ('DX_td', 'C0')]:
        g = np.array(J[key]['grid'])
        p = np.array(J[key]['profile_sum'])
        A.semilogx(g, 2 * (p - p.max()), color=col, label=f'{key}: common {J[key]["common_best"]:.2g} d, '
                   f'QAIC(common-free)={J[key]["QAIC_common_minus_free"]:+.0f}')
    A.axhline(-3.84, color='0.5', ls=':')
    A.set_ylim(-40, 2)
    A.set_xlabel('time constant shared by all families (d)')
    A.set_ylabel('2 x summed quasi-profile LL')
    A.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_recovery_model_space.png', dpi=110)
    plt.close(fig)

    # model comparison summary
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    labs = ['BOX', 'RS', 'DX', 'RSD', 'DXS', 'RSS']
    dq = np.array([[Tm.loc[F, f'QAIC_{k}'] - Tm.loc[F, 'QAIC_RS'] for k in labs] for F in FAMS])
    ax[0].boxplot(dq, tick_labels=labs)
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_ylabel('QAIC - QAIC(RS) per family')
    ax[0].set_title('sum: ' + ', '.join(f'{k} {v:+.0f}' for k, v in zip(labs, dq.sum(0))), fontsize=8)
    ax[1].scatter(Tm.t50_RS, Tm.RS_ta.clip(upper=100), s=15, label='RS ta (best)')
    up = np.array([c[1] for c in Tm.RS_ta_ci])
    ax[1].scatter(Tm.t50_RS, np.clip(up, None, 100), marker='v', color='C3', s=15, label='ta 95% upper (inf clipped to 100)')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('t50 (d)')
    ax[1].set_ylabel('ta (d)')
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_recovery_model_aic.png', dpi=110)
    plt.close(fig)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run_analysis(d, P, N, chat, th):
    T = pd.read_pickle(f'{OUT}/_fits.pkl')
    # per-family table
    cols = [c for c in T.columns]
    Tout = T.copy()
    for c in Tout.columns:
        if Tout[c].dtype == object:
            Tout[c] = Tout[c].apply(lambda v: ';'.join(f'{x:.4g}' for x in v) if isinstance(v, (list, tuple)) else v)
    Tout.insert(1, 'along_km', P.a.values)
    Tout.insert(2, 'depth_km', P.z.values)
    Tout.to_csv(f'{OUT}/per_family_fits.csv', index=False)
    summ = {'setup': dict(catalog='LFEcatV2_MTJ_2008-2026, meancc>=0.13', window_days=[-PRE, POST], bins='1 d',
                          likelihood='Poisson, R0 profiled; c_hat = Pearson dispersion of daily counts, -425..-60 d; QAIC = -2LL/c_hat + 2(k+1)',
                          diffusion_geometry=('DX/RSD: half-space, reservoir at distance r: Delta(t)=Delta0*erf(sqrt(td/t)), td=r^2/(4c). '
                                              'DXS/RSS: slab of thickness L with faces at reservoir pressure, mid-plane, ts=L^2/(pi^2 c)'),
                          D='D = -Delta/(a sigma) > 0 (negative step)')}
    labs = ['M0', 'BOX', 'RS', 'DX', 'RSD', 'DXS', 'RSS']
    summ['model_comparison'] = {k: dict(sum_QAIC_minus_RS=float((T[f'QAIC_{k}'] - T.QAIC_RS).sum()),
                                        sum_AIC_minus_RS=float((T[f'AIC_{k}'] - T.AIC_RS).sum()),
                                        n_fam_best_QAIC=int((T[[f'QAIC_{x}' for x in labs]].idxmin(1) == f'QAIC_{k}').sum()),
                                        n_fam_better_than_RS_by_2=int(((T[f'QAIC_{k}'] - T.QAIC_RS) < -2).sum()),
                                        n_fam_worse_than_RS_by_2=int(((T[f'QAIC_{k}'] - T.QAIC_RS) > 2).sum()))
                                for k in labs}
    summ['c_hat'] = dict(median=float(T.c_hat.median()), range=[float(T.c_hat.min()), float(T.c_hat.max())],
                         fitwindow_RS_median=float(T.c_hat_fitwindow_RS.median()))
    summ['RS'] = dict(t50_median_d=float(T.t50_RS.median()), t50_range=[float(T.t50_RS.min()), float(T.t50_RS.max())],
                      n_ta_upper_finite=int(np.sum([np.isfinite(c[1]) for c in T.RS_ta_ci])),
                      ta_upper95_median=float(np.median([c[1] for c in T.RS_ta_ci])),
                      n_ta_lower_gt0=int(np.sum([c[0] > 0 for c in T.RS_ta_ci])),
                      D_lower95_median=float(np.median([c[0] for c in T.RS_D_ci])),
                      n_D_at_upper_bound=int((T.RS_D > 299).sum()))
    if os.path.exists(f'{OUT}/_fits_declust.pkl'):
        Td = pd.read_pickle(f'{OUT}/_fits_declust.pkl')
        Zd = np.load(f'{OUT}/profiles_declust.npz')
        Qd = np.array([Zd[f'RS_ta_{F}'] / Td.set_index('fam').loc[F, 'c_hat'] for F in FAMS]).sum(0)
        jd = np.argmax(Qd)
        okd = 2 * (Qd[jd] - Qd) <= 3.84
        summ['declustered_robustness'] = dict(
            definition='first event of each family burst (gap > 2 h), same window/models',
            c_hat_median=float(Td.c_hat.median()),
            sum_QAIC_minus_RS={k: float((Td[f'QAIC_{k}'] - Td.QAIC_RS).sum()) for k in labs},
            t50_RS_median=float(Td.t50_RS.median()),
            rho_t50_declust_vs_events=float(stats.spearmanr(Td.t50_RS, T.t50_RS)[0]),
            ta_upper95_median=float(np.median([c[1] for c in Td.RS_ta_ci])),
            common_ta_best=float(Zd['TA_GRID'][jd]), common_ta_ci95=[float(Zd['TA_GRID'][okd].min()), float(Zd['TA_GRID'][okd].max())],
            n_D_at_upper_bound=int((Td.RS_D > 299).sum()))
    J = joint_profiles(T, chat)
    summ['joint_shared_parameter'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('profile_sum', 'grid')} for k, v in J.items()}
    for k in ['DXS_ts', 'RSS_ts']:
        ts = J[k]['common_best']
        lo, hi = J[k]['common_ci95_boot_fam']
        summ['joint_shared_parameter'][k]['c_m2_s_for_L'] = {
            f'L={L:g}m': [L ** 2 / (np.pi ** 2 * x * DAYS2S) for x in (hi, ts, lo)] for L in [10, 100, 500, 1000]}
    out2, C, M, PL, metrics, allres, geo = task2(d, P, T, chat)
    thin_first = fam_first_after(th, 0.0)
    out2['rate_confound']['n_restart_in_17p5_18p8_d_thin'] = int(((thin_first > 17.5) & (thin_first < 18.8)).sum())
    summ['spatial'] = out2
    summ['spatial']['model_comparison_brief'] = M[['metric', 'model', 'dAIC_vs_const', 'p_perm', 'p_placebo_quietgaps', 'n_placebo']
                                                  + [c for c in ['src_along_km', 'c_m2_s_50pct', 'front_speed_km_d'] if c in M]].round(4).to_dict('records')
    t50m = float(T.t50_RS.median())
    span = float(np.ptp(P.a.values.astype(float)))
    summ['diffusion_scales'] = dict(
        diffusion_length_km_sqrt_4ct50={f'c={c:g}': float(np.sqrt(4 * c * t50m * DAYS2S) / 1e3) for c in [1e-3, 1e-2, 1e-1, 1.0]},
        c_needed_m2_s_to_cross_network_span_in_t50=float((span * 1e3) ** 2 / (4 * KAPPA50 * t50m * DAYS2S)),
        network_along_span_km=span, t50_median_d=t50m)
    summ['consistency'] = task3(T, J, N, chat)
    json.dump(jsonable(summ), open(f'{OUT}/summary.json', 'w'), indent=1)
    figures(T, N, P, metrics, M, PL, J, geo)
    return summ


if __name__ == '__main__':
    d, P = load()
    N, chat, th = daily_counts(d)
    if 'fit' in sys.argv:
        T = task1(N, chat)
        T.to_pickle(f'{OUT}/_fits.pkl')
    if 'declust' in sys.argv:
        Nd, chatd, _ = daily_counts(d, declust=True)
        Td = task1(Nd, chatd, tag='_declust')
        Td.to_pickle(f'{OUT}/_fits_declust.pkl')
    if 'analyse' in sys.argv or 'fit' in sys.argv:
        summ = run_analysis(d, P, N, chat, th)
