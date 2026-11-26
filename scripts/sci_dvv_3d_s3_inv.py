#!/usr/bin/env python3
"""Stage 3, step 2: station-local shallow dv/v model + deep bound (revised plan, data/sci/dvv_3d/stage3/PLAN.md).

Model (one joint GLS over all components and series; PRIMARY parametrisation = per-station terms):
  amplitude d(series i, component k) = kappa_ds * ( fA_i a[s,k] + fB_i b[s,k] + fD_i u[k, ds, band] )
    a = apparent dv/v in 0-0.3 km beneath station s, b = in 0.3-3 km (two parameters per station, M3a);
    fA, fB, fD = fractions of the series' coda kernel in 0-0.3 km, 0.3-3 km and below 3 km (coda does not vote
    below 3 km: fD multiplies a deep-coda nuisance u per component x dataset x band, prior 1 x s_k);
    LP network series: mean over stations of (fA a + fB b) + fD u_LP.
  kappa: coupling of coda to its kernels per dataset; runs 'k1' (kappa = 1) and 'kprof' (stage-2 profile: SC 10,
    LFE 5, REP 2.5).  Amplitudes are 'apparent' dv/v.
  Priors (parameter units): a ~ N(0, s_k), b ~ N(0, 0.5 s_k), u ~ N(0, s_k), s_k = median |d| / (kappa (fA + fB))
    (so kappa does not act as a prior scale); variants x0.3, x3 and INVERTED (b ~ N(0, 2 s_k)).
  Covariance (PRIMARY): per-series bootstrap blocks + explicit low-rank common mode per dataset x amplitude type
    (factors kept only above a shuffled-draw null), x placebo factor per dataset x amplitude type; variant: 0.7 S +
    0.3 blockdiag.  Held-out-draw calibration chi2/p reported.  Repeaters reuse the LFE kernel fractions.
  Deep (>= 3 km): NOT estimated here - the coda's deep mass only feeds u; the deep statement is the stage-1
    travel-time bound (tt_results.csv, clean subset), printed alongside.
Depth statement: per station and component the penalised objective of H0 (a = b = 0), top 0.3 km, uniform 0-3 km,
  >= 1 km (1-3 km only; a LOWER limit) and below 3 km (station term x fD) and of the free (a, b) fit.  Verdict = best of
  {top, uniform, deep (>= 1 km and below-3 collapsed)} if it beats H0 by >= 9, is not rejected against the free fit
  (<= 4), beats the runner-up by >= 4, and is the same in ALL six runs (kappa 1 / profile x prior x1, x0.3, x3);
  else 'not resolved' / 'no significant signal'.  Verdicts rank shallow hypotheses only.
Validation: injected truths (top 0.3, uniform 0-3, 1-3 km only, increasing with depth, below 3 km -0.3/-0.5/-1 %),
  generated with default and adversarial kernels (pure surface, l* 5, l* 50, no LVZ, coda 4x near-surface), with
  noise realisations from the data covariance, through the FULL verdict rule; the smallest deep amplitude that
  produces a false shallow verdict is reported.
Products: stage3/station_terms.csv, depth_hypotheses.csv, predictions.csv, series_fit.csv, validation.csv,
  variogram.json, summary.json, viewer.json (spec of the coordinator); bulk npz on /data/wsd02/.../stage3/.
Stages: kern | main [smoke]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
OUT = f'{ROOT}/data/sci/dvv_3d/stage3'
BIG = '/data/wsd02/maleen_data/dvv_3d/stage3'
AMPS = ['ENV_cos', 'ENV_sin', 'drop0_Petrolia', 'level_Petrolia', 'drop0_Ferndale', 'level_Ferndale', 'drop0_M7',
        'level_M7', 'ramp21', 'trend']
KAPPA = {'k1': {'SC10': 1.0, 'LFE': 1.0, 'REP': 1.0, 'LP10': 1.0}, 'kprof': {'SC10': 10.0, 'LFE': 5.0, 'REP': 2.5, 'LP10': 1.0}}
SC_BANDS = ['0.1-0.3Hz', '0.3-0.6Hz', '0.5-1Hz', '1-3Hz', '3-5Hz']
EDGES = (0.3, 1.0, 3.0)


# ------------------------------------------------------------------------------------------ kernel fractions
def fracs_from(K, ze):
    zt, zb = ze[:-1], ze[1:]
    K = K / max(K.sum(), 1e-30)
    fA = K[zb <= 0.3 + 1e-9].sum()
    f31 = K[(zt >= 1 - 1e-9) & (zb <= 3 + 1e-9)].sum()
    fB = K[(zt >= 0.3 - 1e-9) & (zb <= 3 + 1e-9)].sum()
    fD = K[zt >= 3 - 1e-9].sum()
    f2031 = K[(zt >= 20 - 1e-9) & (zb <= 31 + 1e-9)].sum()
    return dict(fA=fA, fB=fB, f13=f31, fD=fD, f2031=f2031)


def stage_kern():
    import sci_dvv_3d_kernels as KK
    import sci_dvv_depth2_kernels as K2
    ST = pd.read_csv(f'{ROOT}/data/sci/dvv_3d/stations.csv')
    Kn = KK.Kernels()
    ze = KK.Z_E
    var = {'default': {}, 'surf': dict(alpha=1.0), 'l5': dict(lstar='5'), 'lvz_none': dict(lvz='none'), 'l50': dict(lstar='50')}
    rows = []
    for r in ST.itertuples():
        for b in SC_BANDS:
            for v, o in var.items():
                Krz, _ = Kn.station(r.ns, r.z0_km, b, **o)
                rows.append(dict(dataset='SC10', sta=r.ns, band=b, variant=v, **fracs_from(Krz.sum(0), ze)))
        vm = KK.station_vmodel(r.lat, r.lon, 'ref')
        body = K2.lfe_body_kernel(vm, ze, (3.0, 20.0), l_km='20', dist=40, bh_depth=r.z0_km, f_shift=0.2 if r.z0_km > 0 else 0.0)
        Kz = 0.5 * (Kn.sw[f'{r.ns}|ref|1-3Hz|Kz'] + Kn.sw[f'{r.ns}|ref|3-5Hz|Kz'])
        rr = 0.5 * (float(Kn.sw[f'{r.ns}|ref|1-3Hz|r']) + float(Kn.sw[f'{r.ns}|ref|3-5Hz|r'])) if r.z0_km > 0 else 1.0
        for v, a in (('default', 0.25), ('surf', 1.0), ('l5', 0.25), ('lvz_none', 0.25), ('l50', 0.25)):
            ae = a * rr / (a * rr + 1 - a + 1e-12) if a < 1 else 1.0
            K = (1 - ae) * body / body.sum() + ae * Kz
            rows.append(dict(dataset='LFE', sta=r.ns, band='2-8Hz', variant=v, **fracs_from(K, ze)))
    LP = json.load(open(f'{ROOT}/data/sci/dvv_3d/lp_depth_kernels.json'))
    for b, o in LP.items():
        lf = np.array(o['layer_frac'])          # layers 0,0.1,0.3,1,3,6,10,15,20,25,31,35,250
        f = dict(fA=lf[0] + lf[1], fB=lf[2] + lf[3], f13=lf[3], fD=lf[4:].sum(), f2031=lf[8] + lf[9])
        for v in var:
            rows.append(dict(dataset='LP10', sta='REGION', band=b, variant=v, **f))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/kernel_fracs.csv', index=False, float_format='%.5g')
    print(T[T.variant == 'default'].groupby(['dataset', 'band'])[['fA', 'fB', 'f13', 'fD', 'f2031']].median().round(3))


def frac_row(KF, ds, sta, chan, variant):
    if ds == 'SC10':
        key = ('SC10', sta, chan, variant)
    elif ds in ('LFE', 'REP'):
        key = ('LFE', sta, '2-8Hz', variant)
    else:
        key = ('LP10', 'REGION', chan.split('Hz')[0], variant)
    if key not in KF.index:
        return None
    f = KF.loc[key]
    out = dict(fA=f.fA, fB=f.fB, f13=f.f13, fD=f.fD, f2031=f.f2031)
    return out


def adv_fracs(f, kind):
    """adversarial modification: coda 4x more near-surface sensitive (top 1 km x4, renormalised)."""
    if kind != 'x4':
        return f
    a, b, d = 4 * f['fA'], f['fB'] + 3 * (f['fB'] - f['f13']), f['fD']
    s = a + b + d
    return dict(fA=a / s, fB=b / s, f13=f['f13'] / s, fD=d / s, f2031=f['f2031'] / s)


# ------------------------------------------------------------------------------------------ data
def _psd(C, rel=1e-6):
    ev, V = np.linalg.eigh(C)
    ev = np.maximum(ev, rel * ev.max())
    return (V * ev) @ V.T


def build_cov(D, B, mode='lowrank', nshuffle=10, rng=None):
    """B: (n_draws, n_rows) bootstrap amplitudes.  'lowrank' (PRIMARY): empirical per-series blocks + a low-rank common
    mode per dataset x amplitude type (leading factors of the bootstrap correlation kept only if their eigenvalue
    exceeds the 95th percentile of the same eigenvalue for independently shuffled draws; at most 3), applied to the
    cross-series entries.  'shrink0.3': 0.7 S + 0.3 blockdiag (variant)."""
    rng = rng or np.random.default_rng(5)
    S = np.cov(B.T)
    sv = np.asarray(D.series.astype(object))
    same = sv[:, None] == sv[None, :]
    # per-series blocks: eigenvalue floor at 2 % of the block's mean variance (near-collinear amplitudes, e.g. drop0
    # vs level with a small transient, give near-zero bootstrap directions that do not generalise)
    for key in np.unique(sv):
        ii = np.where(sv == key)[0]
        Bk = S[np.ix_(ii, ii)]
        ev, V = np.linalg.eigh(Bk)
        ev = np.maximum(ev, 0.02 * np.mean(np.diag(Bk)))
        S[np.ix_(ii, ii)] = (V * ev) @ V.T
    if mode.startswith('shrink'):
        lam = float(mode[6:])
        return (1 - lam) * S + lam * np.where(same, S, 0.0), {}
    info = {}
    Lsum = np.zeros_like(S)
    grp = (D.dataset.astype(str) + '|' + D.amp.astype(str)).values
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        if len(idx) < 3:
            continue
        X = B[:, idx]
        sd = X.std(0)
        Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
        R = np.corrcoef(Z.T)
        ev, V = np.linalg.eigh(R)
        ev, V = ev[::-1], V[:, ::-1]
        null = []
        for _ in range(nshuffle):
            Zs = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
            null.append(np.linalg.eigvalsh(np.corrcoef(Zs.T))[::-1][:3])
        thr = np.percentile(np.array(null), 95, axis=0)
        nf = 0
        for f in range(min(3, len(ev))):
            if ev[f] > thr[f]:
                nf += 1
            else:
                break
        info[g] = dict(n=len(idx), n_factors=nf, eig=[round(float(e), 2) for e in ev[:3]], null95=[round(float(t), 2) for t in thr])
        if nf:
            # common-mode part above the null level (eigenvalue - 1, as in a factor model with unit-variance residuals)
            lam = np.maximum(ev[:nf] - 1.0, 0.0)
            Lc = (V[:, :nf] * lam) @ V[:, :nf].T
            Lsum[np.ix_(idx, idx)] += Lc * np.outer(sd, sd)
    # factor model C = L (common mode, PSD) + Psi (per-series residual blocks, PSD-projected): PSD by construction
    C = Lsum.copy()
    for key in np.unique(sv):
        ii = np.where(sv == key)[0]
        Rb = S[np.ix_(ii, ii)] - Lsum[np.ix_(ii, ii)]
        ev, V = np.linalg.eigh(Rb)
        ev = np.maximum(ev, 0.02 * np.mean(np.diag(S[np.ix_(ii, ii)])))
        C[np.ix_(ii, ii)] += (V * ev) @ V.T
    return _psd(C), info


class Data:
    def __init__(self, tag='', mode='lowrank', smoke=False, calib=True, drop_chans=()):
        sfx = tag + ('_smoke' if smoke else '')
        A = pd.read_csv(f'{OUT}/amplitudes{sfx}.csv')
        A = A[~A.chan.astype(str).str.endswith('_all')]          # LP: keep the BB network series only
        if drop_chans:
            A = A[~A.chan.isin(drop_chans) & ~((A.dataset == 'LP10') & ('LP' in drop_chans))]
        Z = np.load(f'{BIG}/boot{sfx}.npz', allow_pickle=False)
        ps = json.load(open(f'{OUT}/placebo_scale{"_smoke" if smoke else ""}.json'))['scale']
        rows, Bcols = [], []
        for key in sorted(A.series.unique()):
            g = A[A.series == key].set_index('amp')
            Bk = Z[key]
            for j, nm in enumerate(AMPS):
                if nm.startswith('drop0_') and np.allclose(Bk[:, j], Bk[:, AMPS.index('level_' + nm[6:])]):
                    continue            # no transient term supported: drop0 == level (duplicate, singular)
                if nm in g.index and np.std(Bk[:, j]) > 0:
                    rows.append(dict(series=key, dataset=g.dataset.iloc[0], sta=g.sta.iloc[0], chan=g.chan.iloc[0], amp=nm,
                                     est=g.loc[nm, 'est'],
                                     no_transient=bool(nm.startswith('level_') and 'drop0_' + nm[6:] not in g.index)))
                    Bcols.append(Bk[:, j])
        self.D = pd.DataFrame(rows)
        B = np.column_stack(Bcols)
        C, self.cov_info = build_cov(self.D, B, mode)
        f = np.array([ps.get(f'{d}|{a}', {}).get('applied', 1.0) if isinstance(ps.get(f'{d}|{a}'), dict) else 1.0
                      for d, a in zip(self.D.dataset, self.D.amp)])
        C = C * np.outer(f, f)
        dg = np.diag(C).copy()
        for nm in AMPS:
            m = self.D.amp.values == nm
            if m.any():
                dg[m] = np.maximum(dg[m], 0.05 * np.median(dg[m]))
        C[np.diag_indices_from(C)] = dg * 1.02
        self.C = _psd(C)
        self.d = self.D.est.values
        self.sd = np.sqrt(np.diag(self.C))
        self.calib = None
        if calib:
            # held-out-draw calibration: covariance from the odd draws, chi2/p of the even draws (should be ~1)
            h1, h2 = B[0::2], B[1::2]
            C1, _ = build_cov(self.D, h1, mode)
            C1 = _psd(C1)
            Ci = np.linalg.inv(C1)
            z = h2 - h1.mean(0)
            self.calib = float(np.mean(np.einsum('ij,jk,ik->i', z, Ci, z)) / B.shape[1])
            # (bootstrap-only calibration: placebo factors are not applied to either half)


# ------------------------------------------------------------------------------------------ model
class Model:
    """d = A x ; x = [a, b per station x component, u per component x group]; Gaussian priors (see header)."""

    def __init__(self, data, kappa='k1', variant='default', prior_scale=1.0, inverted=False, KF=None):
        self.dat = data
        D = data.D
        self.kap = KAPPA[kappa]
        self.STA = sorted(s for s in D.sta.unique() if s != 'REGION')
        si = {s: i for i, s in enumerate(self.STA)}
        nk = len(AMPS)
        groups = sorted(set((r.dataset, r.chan if r.dataset in ('SC10', 'LP10') else '2-8Hz') for r in D.itertuples()))
        self.ug = {g: i for i, g in enumerate(groups)}
        self.ng = len(groups)
        ns = len(self.STA)
        self.npar = ns * nk * 2 + nk * len(groups)
        self.o_u = ns * nk * 2
        A = np.zeros((len(D), self.npar))
        F, KD, SI = [], np.zeros(len(D)), -np.ones(len(D), int)
        for i, r in enumerate(D.itertuples()):
            k = AMPS.index(r.amp)
            f = frac_row(KF, r.dataset, r.sta, r.chan, variant)
            F.append(f)
            if f is None:
                continue
            kap = self.kap[r.dataset]
            KD[i] = kap * f['fD']
            if r.dataset == 'LP10':
                for s in self.STA:
                    A[i, self.ia(si[s], k)] += kap * f['fA'] / ns
                    A[i, self.ib(si[s], k)] += kap * f['fB'] / ns
            elif r.sta in si:
                SI[i] = si[r.sta]
                A[i, self.ia(si[r.sta], k)] = kap * f['fA']
                A[i, self.ib(si[r.sta], k)] = kap * f['fB']
            g = (r.dataset, r.chan if r.dataset in ('SC10', 'LP10') else '2-8Hz')
            A[i, self.o_u + k * len(groups) + self.ug[g]] = kap * f['fD']
        self.A, self.F, self.KD, self.SI = A, F, KD, SI
        # prior scale per component in PARAMETER units: median |d| / (kappa (fA + fB))
        s_k = []
        for nm in AMPS:
            m = (D.amp.values == nm)
            num = np.abs(D.est.values[m])
            den = np.array([A[i, :self.o_u].sum() for i in np.where(m)[0]])
            ok = den > 0
            s_k.append(np.median(num[ok] / den[ok]) if ok.any() else 0.1)
        self.s_k = np.maximum(np.array(s_k), 1e-4) * prior_scale
        P = np.zeros(self.npar)
        for s in range(ns):
            for k in range(nk):
                P[self.ia(s, k)] = 1 / self.s_k[k] ** 2
                P[self.ib(s, k)] = 1 / ((2.0 if inverted else 0.5) * self.s_k[k]) ** 2
        for k in range(nk):
            P[self.o_u + k * len(groups):self.o_u + (k + 1) * len(groups)] = 1 / self.s_k[k] ** 2
        self.P = P
        self.Ci = np.linalg.inv(data.C)
        self.AtCi = A.T @ self.Ci
        self.N0 = self.AtCi @ A

    def ia(self, s, k):
        return (s * len(AMPS) + k) * 2

    def ib(self, s, k):
        return (s * len(AMPS) + k) * 2 + 1

    def solve(self, d=None, fix=None, extraP=None, col=None, want_cov=True):
        """fix: reparametrisation matrix T (x = T z); col = (j, c): replace design column j by c."""
        d = self.dat.d if d is None else d
        P = self.P if extraP is None else extraP
        N = self.N0.copy()
        rhs = self.AtCi @ d
        A = self.A
        if col is not None:
            j, c = col
            Cic = self.Ci @ c
            v = self.AtCi @ c
            N[:, j] = v
            N[j, :] = v
            N[j, j] = float(c @ Cic)
            rhs[j] = float(Cic @ d)
            A = A.copy()
            A[:, j] = c
        N = N + np.diag(P)
        C = None
        if fix is not None:
            T = fix
            Nz = T.T @ N @ T
            z = np.linalg.solve(Nz, T.T @ rhs)
            x = T @ z
            if want_cov:
                C = T @ np.linalg.inv(Nz) @ T.T
        else:
            if want_cov:
                C = np.linalg.inv(N)
                x = C @ rhs
            else:
                x = np.linalg.solve(N, rhs)
        r = d - A @ x
        chi = float(r @ self.Ci @ r)
        return x, C, chi + float(x @ (P * x)), chi


# ------------------------------------------------------------------------------------------ depth hypotheses
HYPS = ('H0', 'top0.3', 'uniform0_3', 'ge1km', 'below3', 'free')


def _T(M, keep_cols, newcol=None):
    n = M.npar
    keep = [j for j in range(n) if j not in keep_cols]
    T = np.eye(n)[:, keep]
    if newcol is not None:
        T = np.column_stack([T, newcol])
    return T


def depth_hypotheses(M, tag, d=None, amps=None, stations=None):
    """per station x component: penalised objective (tot) of one-parameter depth hypotheses:
    H0 (a = b = 0), top0.3 (b = 0), uniform0_3 (a = b), ge1km (1-3 km only: f13 instead of fB; a LOWER depth limit),
    below3 (station-specific term with the kernel mass below 3 km, fD), free (a, b free).  The tested terms get a weak
    prior (1e-6) so the comparison is data-driven; everything else keeps its prior."""
    out = []
    for s, sta in enumerate(M.STA):
        if stations is not None and sta not in stations:
            continue
        rows_s = M.SI == s
        for k, nm in enumerate(AMPS):
            if amps is not None and nm not in amps:
                continue
            ia, ib = M.ia(s, k), M.ib(s, k)
            if not M.A[:, ia].any():
                continue
            P = M.P.copy()
            P[ia] = P[ib] = 1e-6
            res = {}
            for hyp in HYPS:
                col, T = None, None
                if hyp == 'H0':
                    T = _T(M, (ia, ib))
                elif hyp == 'top0.3':
                    T = _T(M, (ib,))
                elif hyp == 'uniform0_3':
                    v = np.zeros(M.npar)
                    v[ia] = v[ib] = 1.0
                    T = _T(M, (ia, ib), v)
                elif hyp == 'ge1km':
                    c = M.A[:, ib].copy()
                    for i in np.where(c != 0)[0]:
                        f = M.F[i]
                        c[i] *= f['f13'] / max(f['fB'], 1e-12)
                    col = (ib, c)
                    T = _T(M, (ia,))
                elif hyp == 'below3':
                    c = np.where(rows_s & (M.A[:, ia] != 0), M.KD, 0.0)
                    col = (ia, c)
                    T = _T(M, (ib,))
                _, _, tot, chi = M.solve(d=d, fix=T, extraP=P, col=col, want_cov=False)
                res[hyp] = tot
            out.append(dict(run=tag, sta=sta, amp=nm, **{f'tot_{h}': v for h, v in res.items()}))
    return pd.DataFrame(out)


def classify(row, dmin_h0=9.0, dfree=4.0, margin=4.0):
    """verdict for one run: shallow hypotheses ranked; deep ones collapsed unless separated by >= 4."""
    t = {h: row[f'tot_{h}'] for h in HYPS}
    deep = min(t['ge1km'], t['below3'])
    cand = {'top ~0.3 km': t['top0.3'], 'uniform 0-3 km': t['uniform0_3'], '>=1 km (lower limit only)': deep}
    best = min(cand, key=cand.get)
    srt = sorted(cand.values())
    if t['H0'] - cand[best] < dmin_h0:
        return 'no significant signal'
    if cand[best] - t['free'] > dfree:
        return 'not resolved'
    if srt[1] - srt[0] < margin:
        return 'not resolved'
    return best


def verdicts(H, runs_required):
    """final verdict per station x component: the same verdict in ALL required runs (2 kappa x 3 prior scales)."""
    H = H.copy()
    H['v'] = H.apply(classify, axis=1)
    out = []
    for (sta, nm), g in H.groupby(['sta', 'amp']):
        vs = set(g.v)
        ok = set(g.run) >= set(runs_required) and len(vs) == 1
        out.append(dict(sta=sta, amp=nm, verdict=list(vs)[0] if ok else 'not resolved', per_run=';'.join(f'{a}:{b}' for a, b in zip(g.run, g.v))))
    return pd.DataFrame(out), H


# ------------------------------------------------------------------------------------------ predictions
def time_functions(t, taus):
    """daily time functions of each amplitude type at dates t: y = level (H - exp) + drop0 exp + ramp + trend + ENV."""
    out = np.zeros((len(t), len(AMPS)))
    doy = 2 * np.pi * (t.dayofyear.values - 1) / 365.25
    out[:, 0], out[:, 1] = np.cos(doy), np.sin(doy)
    tq = {'Petrolia': pd.Timestamp('2021-12-20 20:10'), 'Ferndale': pd.Timestamp('2022-12-20 10:34'), 'M7': pd.Timestamp('2024-12-05 18:44')}
    for q, t0 in tq.items():
        dt = (t - t0).days.values.astype(float)
        st = (dt >= 0).astype(float)
        tr = np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / taus[q]), 0.0)
        out[:, AMPS.index(f'level_{q}')] = st - tr
        out[:, AMPS.index(f'drop0_{q}')] = tr
    out[:, AMPS.index('ramp21')] = np.clip((t - pd.Timestamp('2021-07-01')).days.values / 365.25, 0, None)
    out[:, AMPS.index('trend')] = (t - pd.Timestamp('2018-01-01')).days.values / 365.25
    return out


def station_pred_vectors(M, sta, KF):
    """parameter-space vectors (per amplitude type) giving the apparent dv/v of the station's 1-3 Hz SC series
    (fallback: its LFE coda), with that dataset's kappa; drop0 tied to level where the station has no transient."""
    f = frac_row(KF, 'SC10', sta, '1-3Hz', 'default')
    ds = 'SC10'
    if f is None:
        f, ds = frac_row(KF, 'LFE', sta, '2-8Hz', 'default'), 'LFE'
    if f is None:
        return None, None
    s = M.STA.index(sta)
    kap = M.kap[ds]
    V = np.zeros((len(AMPS), M.npar))
    for k, nm in enumerate(AMPS):
        kk = k
        if nm.startswith('drop0_') and not M.A[:, M.ia(s, k)].any():
            kk = AMPS.index('level_' + nm[6:])
        V[k, M.ia(s, kk)] = kap * f['fA']
        V[k, M.ib(s, kk)] = kap * f['fB']
    return V, ds


# ------------------------------------------------------------------------------------------ main
RUNS = [(k, ps) for k in ('k1', 'kprof') for ps in (1.0, 0.3, 3.0)]


def _inj_job(args):
    kind, truth, amp_t, noise_i, smoke = args
    KF = pd.read_csv(f'{OUT}/kernel_fracs.csv').set_index(['dataset', 'sta', 'band', 'variant'])
    dat = _DAT['d']
    Ms = {(k, ps): Model(dat, k, KF=KF, prior_scale=ps) for k, ps in RUNS}
    rng = np.random.default_rng(1000 + noise_i)
    allH = []
    # truths in APPARENT units (kappa = 1 kernels); the same synthetic data are inverted by every run
    d0 = np.zeros(len(dat.D))
    for i, r in enumerate(dat.D.itertuples()):
        if r.amp != 'level_M7':
            continue
        f = frac_row(KF, r.dataset, r.sta, r.chan, 'default' if kind == 'x4' else kind)
        if f is None:
            continue
        f = adv_fracs(f, kind)
        kap = 1.0
        if truth == 'top0.3':
            d0[i] = kap * f['fA'] * amp_t
        elif truth == 'uniform0_3':
            d0[i] = kap * (f['fA'] + f['fB']) * amp_t
        elif truth == 'ge1km_1-3':
            d0[i] = kap * f['f13'] * amp_t
        elif truth == 'increasing':
            d0[i] = kap * (f['fA'] * amp_t / 3 + f['fB'] * amp_t)
        elif truth == 'below3':
            d0[i] = kap * f['fD'] * amp_t
    if noise_i > 0:
        Ls = np.linalg.cholesky(dat.C)
        d0 = d0 + Ls @ rng.standard_normal(len(d0))
    for (k, ps), M in Ms.items():
        d = d0
        H = depth_hypotheses(M, f'{k}|x{ps:g}', d=d, amps=['level_M7'])
        x, C, _, _ = M.solve(d=d)
        H['a_rec'] = [x[M.ia(M.STA.index(s), AMPS.index('level_M7'))] for s in H.sta]
        H['b_rec'] = [x[M.ib(M.STA.index(s), AMPS.index('level_M7'))] for s in H.sta]
        allH.append(H)
    H = pd.concat(allH, ignore_index=True)
    V, H = verdicts(H, [f'{k}|x{ps:g}' for k, ps in RUNS])
    V['kernel'], V['truth'], V['amp_true'], V['noise'] = kind, truth, amp_t, noise_i
    rec = H[H.run == 'k1|x1'][['sta', 'a_rec', 'b_rec']]
    return V.merge(rec, on='sta')


_DAT = {}


def run_all(smoke=False, tag=''):
    from multiprocessing import Pool
    t0 = time.time()
    KF = pd.read_csv(f'{OUT}/kernel_fracs.csv').set_index(['dataset', 'sta', 'band', 'variant'])
    sfx = tag + ('_smoke' if smoke else '')
    taus = json.load(open(f'{OUT}/tau_net{"_smoke" if smoke else ""}.json'))['tau_q']
    if tag.startswith('_tau'):
        taus = {q: max(2.0, t * float(tag[4:])) for q, t in taus.items()}
    dat = Data(tag=tag, smoke=smoke)
    _DAT['d'] = dat
    terms, preds, fits, allH = [], [], [], []
    sols = {}
    cfgs = [('primary', {}, {}), ('prior_x0.3', dict(prior_scale=0.3), {}), ('prior_x3', dict(prior_scale=3.0), {}),
            ('inverted', dict(inverted=True), {}), ('shrink0.3', {}, dict(mode='shrink0.3')),
            ('sc_surf', dict(variant='surf'), {}), ('sc_l5', dict(variant='l5'), {}), ('lvz_none', dict(variant='lvz_none'), {})]
    for ktag in ('k1', 'kprof'):
        for cfg, kw, dkw in cfgs:
            if (smoke or tag) and cfg not in ('primary', 'prior_x0.3', 'prior_x3', 'inverted'):
                continue
            dd = Data(tag=tag, smoke=smoke, calib=False, **dkw) if dkw else dat
            M = Model(dd, ktag, KF=KF, **kw)
            x, C, tot, chi = M.solve()
            sols[(ktag, cfg)] = (M, x, C)
            for s, sta in enumerate(M.STA):
                for k, nm in enumerate(AMPS):
                    ia, ib = M.ia(s, k), M.ib(s, k)
                    if not M.A[:, ia].any():
                        continue
                    terms.append(dict(kappa=ktag, cfg=cfg, sta=sta, amp=nm, a_0_0p3=x[ia], a_sd=np.sqrt(C[ia, ia]),
                                      b_0p3_3=x[ib], b_sd=np.sqrt(C[ib, ib]), corr_ab=C[ia, ib] / np.sqrt(C[ia, ia] * C[ib, ib]),
                                      a_post_over_prior=np.sqrt(C[ia, ia] * M.P[ia]), b_post_over_prior=np.sqrt(C[ib, ib] * M.P[ib]),
                                      n_series=int((M.A[:, ia] != 0).sum()),
                                      level_without_transient=bool(nm.startswith('level_') and dd.D[(dd.D.sta == sta) & (dd.D.amp == nm)].no_transient.any())))
            r = dd.d - M.A @ x
            for nm in AMPS:
                m = dd.D.amp.values == nm
                fits.append(dict(kappa=ktag, cfg=cfg, amp=nm, n=int(m.sum()),
                                 chi2=float(r[m] @ np.linalg.solve(dd.C[np.ix_(m, m)], r[m])),
                                 rms_z=float(np.sqrt(np.mean((r[m] / dd.sd[m]) ** 2)))))
            if cfg in ('primary', 'prior_x0.3', 'prior_x3'):
                ps = {'primary': 1.0, 'prior_x0.3': 0.3, 'prior_x3': 3.0}[cfg]
                allH.append(depth_hypotheses(M, f'{ktag}|x{ps:g}'))
    T = pd.DataFrame(terms)
    V, H = verdicts(pd.concat(allH, ignore_index=True), [f'{k}|x{ps:g}' for k, ps in RUNS])
    # export rule (MF6): the (a, b) split is reported only if both posterior/prior < 0.7 and the inverted-prior run
    # agrees (same sign, within 1 SD); otherwise only the kernel-weighted 0-3 km amplitude ('split prior-dominated')
    prim = T[T.cfg == 'primary'].merge(T[T.cfg == 'inverted'][['kappa', 'sta', 'amp', 'a_0_0p3', 'b_0p3_3']],
                                       on=['kappa', 'sta', 'amp'], suffixes=('', '_inv'))
    prim['split_ok'] = ((prim.a_post_over_prior < 0.7) & (prim.b_post_over_prior < 0.7)
                        & (np.sign(prim.a_0_0p3) == np.sign(prim.a_0_0p3_inv)) & (np.sign(prim.b_0p3_3) == np.sign(prim.b_0p3_3_inv))
                        & ((prim.a_0_0p3 - prim.a_0_0p3_inv).abs() < prim.a_sd) & ((prim.b_0p3_3 - prim.b_0p3_3_inv).abs() < prim.b_sd))
    T = T.merge(prim[['kappa', 'sta', 'amp', 'split_ok']], on=['kappa', 'sta', 'amp'], how='left')
    # predictions at stations (apparent; relative 0-3 km part; ENV annual only; no dose / microseism)
    dates = {'2024-11-30': '2024-11-30', '2024-12-20': '2024-12-20', '2025-06-01': '2025-06-01', '2025-12-01': '2025-12-01',
             'wet (Feb)': '2025-02-15', 'dry (Sep)': '2025-09-15'}
    for ktag in ('k1', 'kprof'):
        M, x, C = sols[(ktag, 'primary')]
        tf = time_functions(pd.DatetimeIndex(list(dates.values())), taus)
        for sta in M.STA:
            Vv, ds = station_pred_vectors(M, sta, KF)
            if Vv is None:
                continue
            amp_vec = np.array([Vv[k] @ x for k in range(len(AMPS))])
            for j, lab in enumerate(dates):
                w = tf[j].copy()
                if lab.startswith(('wet', 'dry')):
                    w[2:] = 0.0
                v = w @ Vv
                preds.append(dict(kappa=ktag, sta=sta, date=lab, mean=float(v @ x), sd=float(np.sqrt(v @ C @ v)),
                                  kernel_dataset=ds, tau_sensitive=lab == '2024-12-20'))
            for k, nm in enumerate(AMPS):
                preds.append(dict(kappa=ktag, sta=sta, date=f'amp:{nm}', mean=float(amp_vec[k]),
                                  sd=float(np.sqrt(Vv[k] @ C @ Vv[k])), kernel_dataset=ds, tau_sensitive=False))
    P = pd.DataFrame(preds)
    T.to_csv(f'{OUT}/station_terms{sfx}.csv', index=False, float_format='%.5g')
    H.to_csv(f'{OUT}/depth_hypotheses{sfx}.csv', index=False, float_format='%.4g')
    V.to_csv(f'{OUT}/depth_verdicts{sfx}.csv', index=False)
    P.to_csv(f'{OUT}/predictions{sfx}.csv', index=False, float_format='%.5g')
    pd.DataFrame(fits).to_csv(f'{OUT}/series_fit{sfx}.csv', index=False, float_format='%.4g')
    json.dump(dict(cov_info=dat.cov_info, heldout_chi2_per_dim=dat.calib, taus=taus), open(f'{OUT}/cov_info{sfx}.json', 'w'), indent=1)
    np.savez_compressed(f'{BIG}/solutions{sfx}.npz', **{f'{a}|{b}|x': v[1] for (a, b), v in sols.items()},
                        **{f'{a}|{b}|C': v[2] for (a, b), v in sols.items()})
    print('main done', f'{time.time() - t0:.0f} s; held-out chi2/p {dat.calib:.2f}', flush=True)
    if tag:
        return
    # ---------------------------------------------------------------- low-band test (coordinator): are the '>= 1 km' ENV
    # verdicts carried by the 0.1-0.3 / 0.3-0.6 Hz SC bands (seasonal microseism-source change) ?  Refit without them
    # (and without LP); SC 0.5-5 Hz + LFE + REP only; full verdict rule.
    dlow = Data(tag=tag, smoke=smoke, calib=False, drop_chans=('0.1-0.3Hz', '0.3-0.6Hz', 'LP'))
    allH2 = []
    for k_, ps_ in RUNS:
        M2 = Model(dlow, k_, KF=KF, prior_scale=ps_)
        allH2.append(depth_hypotheses(M2, f'{k_}|x{ps_:g}', amps=['ENV_cos', 'ENV_sin']))
    V2, _ = verdicts(pd.concat(allH2, ignore_index=True), [f'{k}|x{ps:g}' for k, ps in RUNS])
    V2 = V2.merge(V[['sta', 'amp', 'verdict']], on=['sta', 'amp'], suffixes=('_no_lowbands', '_all_bands'))
    V2.to_csv(f'{OUT}/depth_verdicts_env_nolowbands{sfx}.csv', index=False)
    print(V2[V2.verdict_all_bands != V2.verdict_no_lowbands][['sta', 'amp', 'verdict_all_bands', 'verdict_no_lowbands']].to_string(), flush=True)
    # ---------------------------------------------------------------- validation (injections, full verdict rule)
    jobs = []
    for kind in ('default', 'surf', 'l5', 'lvz_none', 'x4', 'l50'):
        for truth, amps_ in (('top0.3', (-0.5,)), ('uniform0_3', (-0.3,)), ('ge1km_1-3', (-0.3,)), ('increasing', (-0.3,)),
                             ('below3', (-0.3, -0.5, -1.0))):
            for a_ in amps_:
                for ni in range(0, 1 if smoke else 4):
                    jobs.append((kind, truth, a_, ni, smoke))
    with Pool(int(os.environ.get('NPROC', '6'))) as pool:
        res = pool.map(_inj_job, jobs, chunksize=1)
    Vd = pd.concat(res, ignore_index=True)
    Vd.to_csv(f'{OUT}/validation{sfx}.csv', index=False, float_format='%.5g')
    S = Vd.groupby(['kernel', 'truth', 'amp_true']).verdict.agg(lambda s: s.value_counts(normalize=True).round(3).to_dict()).reset_index()
    S.to_csv(f'{OUT}/validation_summary{sfx}.csv', index=False)
    deep = Vd[Vd.truth == 'below3']
    false_sh = deep[deep.verdict.isin(['top ~0.3 km', 'uniform 0-3 km'])]
    smallest = float(false_sh.amp_true.abs().min()) if len(false_sh) else None
    json.dump(dict(smallest_deep_amp_false_shallow_verdict=smallest,
                   note='verdicts rank shallow hypotheses only; a deep change >= X % below 3 km is indistinguishable from '
                        '">= 1 km"; the deep crust is bounded by stage 1 only'),
              open(f'{OUT}/validation_note{sfx}.json', 'w'), indent=1)
    with pd.option_context('display.width', 220, 'display.max_colwidth', 120):
        print(pd.DataFrame(fits)[lambda x: x.cfg == 'primary'].round(3).to_string())
        print(V.verdict.value_counts().to_string())
        print(S.to_string())
        print('smallest deep amplitude giving a false shallow verdict:', smallest, ' held-out chi2/p', dat.calib)


if __name__ == '__main__':
    st = sys.argv[1]
    smoke = 'smoke' in sys.argv[2:]
    tag = next((a for a in sys.argv[2:] if a.startswith('_tau')), '')
    if st == 'kern':
        stage_kern()
    elif st == 'main':
        run_all(smoke, tag)
