#!/usr/bin/env python3
"""Joint 3-D dv/v: PRIMARY small physically parametrised GLS (second stage) + validation.

Stage 1 (scripts/sci_dvv_3d_tt.py) fits the raw LFE/earthquake S and P delays with origin/location nuisances,
free receiver terms beta(station, phase, time function) and deep volume terms gamma.  Its global estimate and
covariance are a sufficient (Gaussian) summary, so this second stage is equivalent to one joint fit:

  parameters per quantity q (m7, pet, fer, D):
    s[st, k]   per-station dv/v (%) in 0-0.1, 0.1-1, 1-3 km  (Gaussian prior SD 5 / 1 / 0.5 %, independent)
    clock[st]  per-station timing step (s), S = P            (Gaussian prior SD 3 ms)
    gamma[v]   deep / regional volumes of the stage-1 run (%) (FLAT prior)
    c[band]    SC per-band common mode (variant 'cm')
  data:
    TT   b[st, ph] = -sum_k T[st, ph, k] s[st, k] / 100 + clock[st];  gamma_hat[v] = gamma[v]
         (joint covariance from stage 1, formal x max(1, chi2r) and x (honest / formal) of the stage-1 gamma)
    SC   d[st, band] = kappa_SC  sum_k Ksc[st, band, k] s[st, k] (+ c[band])
    LFE  d[st]       = kappa_L   sum_k Klfe[st, k] s[st, k]          (LFE coda, 2-8 Hz)
    REP  d[st, cls]  = kappa_R   sum_k Klfe[st, k] s[st, k]          (repeater coda)
    LP   d[band, dist] = f_shallow * mean_st(s) + f_1_10 g_R1_10 + f_10_20 g_R10_20 + f_deep g_lay  (runs with R)
  Revised after code review: (M2) the PRIMARY deep numbers are the stage-1 values (free receiver terms, placebo-
  calibrated SE); stage-2 deep values are CONDITIONAL on the shallow priors, which are varied (x1/x3/x10), and a
  flat network-mean receiver term per phase (rcm) is added.  (M3) the coda kernel mass below 3 km is not dropped:
  it multiplies a deep-coda nuisance u per dataset x band (prior 1 %), dchi2 for u is reported.  (M4) SC covariance
  = factor model x_ib = sqrt(rho_b) u_b + sqrt(1 - rho_b) e_ib with Corr(u) = Corr(e) = R_band (PSD by
  construction), rho_b from simultaneous season-matched placebo fits; variants rho = 0 / 0.5.  One coupling kappa
  per coda dataset shared by all quantities, profiled on 0.1-5 and re-profiled for the shallow-only null.  LP:
  0-1 km -> mean station terms, 1-10 -> R1_10, 10-20 -> R10_20, 20-31 -> layer, > 31 km -> nuisance lp31.
Outputs: data/sci/dvv_3d/gls_results.csv, gls_station_terms.csv, gls_dataset_chi2.csv, ak_*.csv, inj_*.csv,
         summary.json (key 'gls').
Function invert_component(df) maps any per-station x band amplitude vector to depth-binned station terms.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_3d'
BIG = '/data/wsd02/maleen_data/dvv_3d'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_source_region as SR_  # noqa: E402
import sci_dvv_3d_kernels as KK  # noqa: E402

NJ = len(SR_.PN)
QS = ['m7', 'pet', 'fer', 'D']
BINS = [(0.0, 0.1), (0.1, 1.0), (1.0, 3.0)]
PRIOR_S = np.array([5.0, 1.0, 0.5])
PRIOR_CLOCK = 0.003
SC_BANDS = KK.BANDS


def qvec(q, s=0.0):
    v = np.zeros(NJ)
    for j in range(NJ):
        e = np.zeros(NJ)
        e[j] = 1
        v[j] = SR_.QUANT[q](e, s) - SR_.QUANT[q](np.zeros(NJ), s)
    return v


# ------------------------------------------------------------------------------------------ coda depth kernels
def depth_fracs(K):
    ze = KK.Z_E
    out = []
    for a, b in BINS:
        m = (ze[:-1] >= a - 1e-9) & (ze[1:] <= b + 1e-9)
        out.append(K[m].sum())
    out.append(K[ze[:-1] >= 3 - 1e-9].sum())
    for a, b in ((1, 10), (10, 20), (20, 31)):
        out.append(K[(ze[:-1] >= a - 1e-9) & (ze[1:] <= b + 1e-9)].sum())
    return np.array(out)


def stage_kern():
    """depth fractions (0-0.1, 0.1-1, 1-3, >3 km) of SC (variants) and LFE-coda kernels per station."""
    import sci_dvv_depth2_kernels as K2
    ST = pd.read_csv(f'{OUT}/stations.csv')
    Kn = KK.Kernels()
    var = {'default': {}, 'surf': dict(alpha=1.0), 'body': dict(alpha=0.0), 'l5': dict(lstar='5'),
           'l50': dict(lstar='50'), 'lvz_soft': dict(lvz='soft'), 'lvz_none': dict(lvz='none')}
    rows = []
    for r in ST.itertuples():
        for b in SC_BANDS:
            for v, o in var.items():
                Krz, _ = Kn.station(r.ns, r.z0_km, b, **o)
                f = depth_fracs(Krz.sum(0))
                rows.append(dict(dataset='SC', sta=r.ns, band=b, variant=v, f0=f[0], f1=f[1], f2=f[2], fdeep=f[3],
                                 f1_10=f[4], f10_20=f[5], f20_31=f[6]))
        vm = KK.station_vmodel(r.lat, r.lon, 'ref')
        body = K2.lfe_body_kernel(vm, KK.Z_E, (3.0, 20.0), l_km='20', dist=40, bh_depth=r.z0_km,
                                  f_shift=0.2 if r.z0_km > 0 else 0.0)
        Kz = 0.5 * (Kn.sw[f'{r.ns}|ref|1-3Hz|Kz'] + Kn.sw[f'{r.ns}|ref|3-5Hz|Kz'])
        rr = 0.5 * (float(Kn.sw[f'{r.ns}|ref|1-3Hz|r']) + float(Kn.sw[f'{r.ns}|ref|3-5Hz|r'])) if r.z0_km > 0 else 1.0
        for v, a in (('default', 0.25), ('a0', 0.0), ('a50', 0.5)):
            ae = a * rr / (a * rr + 1 - a + 1e-12)
            f = depth_fracs((1 - ae) * body / body.sum() + ae * Kz)
            rows.append(dict(dataset='LFE', sta=r.ns, band='2-8Hz', variant=v, f0=f[0], f1=f[1], f2=f[2], fdeep=f[3],
                             f1_10=f[4], f10_20=f[5], f20_31=f[6]))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/coda_depth_frac.csv', index=False, float_format='%.4g')
    print(T[T.variant == 'default'].groupby(['dataset', 'band'])[['f0', 'f1', 'f2', 'fdeep']].median().round(3))


# ------------------------------------------------------------------------------------------ data assembly
KGRID = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5, 5.0, 10.0]
PRIOR_U = 1.0          # % ; deep-coda nuisance (coda kernel mass below 3 km), per dataset x band
PRIOR_LP31 = 1.0       # % ; LP kernel mass below 31 km


def _psd_corr(R, floor=0.05):
    ev, V = np.linalg.eigh(R)
    R2 = (V * np.maximum(ev, floor)) @ V.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


class Problem:
    """second-stage GLS for one stage-1 run.  CONDITIONAL on the shallow priors and coda kernels: the primary deep
    numbers are the stage-1 values (free receiver terms, placebo-calibrated SE; tt_results.csv)."""

    def __init__(self, run, use=('TT', 'SC', 'LFE', 'REP', 'LP'), cm=False, sc_var='default', lfe_var='default',
                 prior_scale=1.0, rho='est', unuis=True, sc_file='coda_obs_sc.csv', lp_bands=('0.1-0.2', '0.2-0.3')):
        self.run, self.use, self.cm = run, set(use), cm
        self.prior_scale, self.rho_mode, self.unuis = prior_scale, rho, unuis
        self.lp_bands = tuple(lp_bands)    # 0.05-0.1 Hz LP kept out of the 10-31 km terms unless its checks pass
        R = pd.read_csv(os.environ.get('TT_RESULTS', f'{OUT}/tt_results.csv'))
        R = R[R.run == run]
        if R.empty:
            raise ValueError(f'run {run} missing from tt_results.csv')
        self.tt_res = R
        self.hon = {(r.vol, r.quantity): max(1.0, r.se / r.se_formal) for r in R.itertuples() if r.quantity in QS}
        Z = np.load(f'{BIG}/tt_{run}.npz', allow_pickle=False)
        self.njx = int(Z['njx']) if 'njx' in Z.files else NJ
        self.sp = [tuple(x.split('|')) for x in Z['sp']]
        self.vols = list(Z['vols'])
        self.xg, self.cov = Z['xg'], Z['cov']
        self.nb = len(self.sp) * self.njx
        G = pd.read_csv(f'{OUT}/tt_geometry.csv')
        G = G[G.lvz == 'ref']
        self.Tsh = G.groupby(['sta', 'phase'])[['T0_0.1', 'T0.1_1', 'T1_3']].mean()
        self.kf = pd.read_csv(f'{OUT}/coda_depth_frac.csv').set_index(['dataset', 'sta', 'band', 'variant'])
        self.sc = pd.read_csv(f'{OUT}/{sc_file}')
        self.lfe = pd.read_csv(f'{OUT}/coda_obs_lfe.csv')
        self.rep = pd.read_csv(f'{OUT}/coda_obs_rep.csv')
        self.lp = pd.read_csv(f'{OUT}/coda_obs_lp.csv')
        self.sc_var, self.lfe_var = sc_var, lfe_var
        self.corr = self._sc_corr(sc_file)
        self.lpk = json.load(open(f'{OUT}/lp_depth_kernels.json'))
        self.layname = next((v for v in self.vols if v.startswith('lay')), None)
        self.r1 = next((v for v in self.vols if v.startswith('R1_10')), None)
        self.r2 = next((v for v in self.vols if v.startswith('R10_20')), None)
        self._cache = {}

    def qv(self, q):
        v = qvec(q)
        return np.r_[v, np.zeros(self.njx - NJ)]

    def _sc_corr(self, sc_file):
        pf = f'{OUT}/sc_placebo{"_pow" if "pow" in sc_file else ""}.pkl'
        # project-internal pickle written by scripts/sci_dvv_3d_codafit.py (trusted)
        P = pickle.load(open(pf, 'rb'))
        out = {}
        for q in QS:
            for b in SC_BANDS:
                stas = [k[0] for k in P if k[1] == b and k[2] == q and len(P[k]) >= 4]
                rs = []
                for i in range(len(stas)):
                    for j in range(i + 1, len(stas)):
                        a, c = P[(stas[i], b, q)], P[(stas[j], b, q)]
                        sh = sorted(set(a) & set(c))
                        if len(sh) >= 5:
                            rs.append(np.corrcoef([a[s] for s in sh], [c[s] for s in sh])[0, 1])
                out[(q, b)] = float(np.clip(np.nanmedian(rs), 0, 0.95)) if rs else 0.0
            M = np.eye(len(SC_BANDS))
            for i, b1 in enumerate(SC_BANDS):
                for j, b2 in enumerate(SC_BANDS):
                    if j <= i:
                        continue
                    rs = []
                    for k in P:
                        if k[1] == b1 and k[2] == q and (k[0], b2, q) in P:
                            a, c = P[k], P[(k[0], b2, q)]
                            sh = sorted(set(a) & set(c))
                            if len(sh) >= 5:
                                rs.append(np.corrcoef([a[s] for s in sh], [c[s] for s in sh])[0, 1])
                    M[i, j] = M[j, i] = float(np.clip(np.nanmedian(rs), -0.95, 0.95)) if rs else 0.0
            out[(q, 'bands')] = _psd_corr(M)
        return out

    def stations(self):
        st = set(s for s, _ in self.sp) if 'TT' in self.use else set()
        for k, df in (('SC', self.sc), ('LFE', self.lfe), ('REP', self.rep)):
            if k in self.use:
                st |= set(df.sta)
        return sorted(st)

    def has_lp(self):
        return 'LP' in self.use and self.r1 is not None and self.r2 is not None and self.layname is not None

    def labels(self):
        STA = self.stations()
        ttsta = sorted(set(s for s, _ in self.sp)) if 'TT' in self.use else []
        lab = [f's|{s}|{k}' for s in STA for k in range(3)] + [f'clock|{s}' for s in ttsta]
        if 'TT' in self.use:
            lab += ['rcm|S', 'rcm|P']
        if 'TT' in self.use or self.has_lp():
            lab += [f'g|{v}' for v in self.vols]
        if self.unuis:
            lab += [f'u|SC|{b}' for b in SC_BANDS if 'SC' in self.use] + (['u|LFE'] if 'LFE' in self.use else []) + \
                   (['u|REP'] if 'REP' in self.use else [])
        if self.has_lp():
            lab += ['lp31']
        if self.cm and 'SC' in self.use:
            lab += [f'cm|{b}' for b in SC_BANDS]
        return lab, STA, ttsta

    def build(self, q, tt_override=None, coda_override=None):
        """blocks (A, d, C, name) at kappa = 1, prior precision P, labels.  The coda columns that scale with kappa are
        recorded in self._kcols[name]."""
        lab, STA, ttsta = self.labels()
        ix = {l: i for i, l in enumerate(lab)}
        npar = len(lab)
        si = {s: i for i, s in enumerate(STA)}
        blocks = []
        kcols = {}
        co = coda_override or {}
        if 'TT' in self.use:
            xg = self.xg if tt_override is None else tt_override
            nsp, nv = len(self.sp), len(self.vols)
            Qv = self.qv(q)
            Tm = np.zeros((nsp + nv, len(xg)))
            for k in range(nsp):
                Tm[k, k * self.njx:(k + 1) * self.njx] = Qv
            for v in range(nv):
                Tm[nsp + v, self.nb + v * self.njx:self.nb + (v + 1) * self.njx] = Qv * 100.0
            z = Tm @ xg
            Cz = Tm @ self.cov @ Tm.T
            f = np.ones(nsp + nv)
            f[:nsp] = max([self.hon.get((v, q), 1.0) for v in self.vols] + [1.0])
            for v in range(nv):
                f[nsp + v] = self.hon.get((self.vols[v], q), 1.0)
            Cz = Cz * np.outer(f, f)
            A = np.zeros((nsp + nv, npar))
            for k, (s, ph) in enumerate(self.sp):
                T = self.Tsh.loc[(s, ph)].values if (s, ph) in self.Tsh.index else np.zeros(3)
                for b in range(3):
                    A[k, ix[f's|{s}|{b}']] = -T[b] / 100.0
                A[k, ix[f'clock|{s}']] = 1.0
                A[k, ix[f'rcm|{ph}']] = 1.0          # network-mean receiver term per phase (flat prior)
            for v in range(nv):
                A[nsp + v, ix[f'g|{self.vols[v]}']] = 1.0
            blocks.append((A, z, Cz, 'TT'))

        def coda_block(name, g, keyfun, ucol):
            A = np.zeros((len(g), npar))
            cols = set()
            keep = []
            for i, r in enumerate(g.itertuples()):
                key = keyfun(r)
                if key not in self.kf.index:
                    continue
                fr = self.kf.loc[key, ['f0', 'f1', 'f2', 'fdeep']].values.astype(float)
                for b in range(3):
                    A[i, ix[f's|{r.sta}|{b}']] = fr[b]
                    cols.add(ix[f's|{r.sta}|{b}'])
                uc = ucol(r)
                if self.unuis and uc in ix:
                    A[i, ix[uc]] = fr[3]
                    cols.add(ix[uc])
                keep.append(i)
            kcols[name] = sorted(cols)
            return A[keep], np.array(keep)
        if 'SC' in self.use:
            g = co.get('SC', self.sc[(self.sc.quantity == q)])
            g = g[g.sta.isin(si) & np.isfinite(g.se) & (g.se > 0)].reset_index(drop=True)
            A, keep = coda_block('SC', g, lambda r: ('SC', r.sta, r.band, self.sc_var), lambda r: f'u|SC|{r.band}')
            g = g.iloc[keep].reset_index(drop=True)
            if self.cm:
                for i, r in enumerate(g.itertuples()):
                    A[i, ix[f'cm|{r.band}']] = 1.0
            se = g.se.values
            bi = np.array([SC_BANDS.index(b) for b in g.band])
            if self.rho_mode == 'est':
                rho = np.array([self.corr[(q, b)] for b in SC_BANDS])
            else:
                rho = np.full(len(SC_BANDS), float(self.rho_mode))
            Rb = self.corr[(q, 'bands')]
            stv = np.asarray(g.sta.astype(object))
            same = stv[:, None] == stv[None, :]
            r_ = np.sqrt(rho[bi])
            N = np.outer(r_, r_) + same * np.outer(np.sqrt(1 - rho[bi]), np.sqrt(1 - rho[bi]))
            # factor model x_ib = sqrt(rho_b) u_b + sqrt(1 - rho_b) e_ib, Corr(u) = Corr(e) = R_band: PSD by construction
            C = Rb[np.ix_(bi, bi)] * N * np.outer(se, se)
            blocks.append((A, g.est.values, C, 'SC'))
        if 'LFE' in self.use:
            g = co.get('LFE', self.lfe[(self.lfe.quantity == q)])
            g = g[g.sta.isin(si) & np.isfinite(g.se) & (g.se > 0)].reset_index(drop=True)
            A, keep = coda_block('LFE', g, lambda r: ('LFE', r.sta, '2-8Hz', self.lfe_var), lambda r: 'u|LFE')
            g = g.iloc[keep]
            blocks.append((A, g.est.values, np.diag(g.se.values ** 2), 'LFE'))
        if 'REP' in self.use:
            g = co.get('REP', self.rep[(self.rep.quantity == q)])
            g = g[g.sta.isin(si) & np.isfinite(g.se) & (g.se > 0)].reset_index(drop=True)
            A, keep = coda_block('REP', g, lambda r: ('LFE', r.sta, '2-8Hz', self.lfe_var), lambda r: 'u|REP')
            g = g.iloc[keep]
            blocks.append((A, g.est.values, np.diag(g.se.values ** 2), 'REP'))
        if self.has_lp():
            g = co.get('LP', self.lp[(self.lp.quantity == q) & self.lp.band.isin(self.lp_bands)])
            g = g[np.isfinite(g.se) & (g.se > 0)].reset_index(drop=True)
            A = np.zeros((len(g), npar))
            for i, r in enumerate(g.itertuples()):
                lf = np.array(self.lpk[r.band]['layer_frac'])   # layers 0,0.1,0.3,1,3,6,10,15,20,25,31,35,250
                for s in STA:
                    A[i, ix[f's|{s}|0']] += lf[0] / len(STA)
                    A[i, ix[f's|{s}|1']] += (lf[1] + lf[2]) / len(STA)
                A[i, ix[f'g|{self.r1}']] = lf[3] + lf[4] + lf[5]
                A[i, ix[f'g|{self.r2}']] = lf[6] + lf[7]
                A[i, ix[f'g|{self.layname}']] = lf[8] + lf[9]
                A[i, ix['lp31']] = lf[10] + lf[11]
            blocks.append((A, g.est.values, np.diag(g.se.values ** 2), 'LP'))
        P = np.zeros(npar)
        ps = self.prior_scale
        for s in STA:
            for b in range(3):
                P[ix[f's|{s}|{b}']] = 1 / (PRIOR_S[b] * ps) ** 2
        for s in ttsta:
            P[ix[f'clock|{s}']] = 1 / (PRIOR_CLOCK * ps) ** 2
        for l in lab:
            if l.startswith('u|'):
                P[ix[l]] = 1 / PRIOR_U ** 2
            if l == 'lp31':
                P[ix[l]] = 1 / PRIOR_LP31 ** 2
        self._kcols = kcols
        return blocks, P, lab, dict(STA=STA, ix=ix)

    def _blocks(self, q, kappa, **kw):
        if kw:
            blocks, P, lab, info = self.build(q, **kw)
            kc = dict(self._kcols)
            out = [(A, d, C, nm, np.linalg.inv(C)) for A, d, C, nm in blocks]
        else:
            if q not in self._cache:
                blocks, P, lab, info = self.build(q)
                self._cache[q] = ([(A, d, C, nm, np.linalg.inv(C)) for A, d, C, nm in blocks], P, lab, info, dict(self._kcols))
            out, P, lab, info, kc = self._cache[q]
        res = []
        for A, d, C, nm, Ci in out:
            k = {'SC': kappa[0], 'LFE': kappa[1], 'REP': kappa[2]}.get(nm, 1.0)
            if k != 1.0 and nm in kc:
                A = A.copy()
                A[:, kc[nm]] *= k
            res.append((A, d, C, nm, Ci))
        return res, P, lab, info

    def solve(self, q, kappa=(1.0, 1.0, 1.0), fix=(), **kw):
        """fix: label prefixes held at 0 (e.g. ('g|',) = shallow-only model, ('u|',) = no deep-coda nuisance)."""
        blocks, P, lab, info = self._blocks(q, kappa, **kw)
        npar = len(lab)
        keep = np.array([not any(l.startswith(f) for f in fix) for l in lab], dtype=bool)
        N = np.diag(P).astype(float)
        rhs = np.zeros(npar)
        for A, d, C, nm, Ci in blocks:
            N += A.T @ Ci @ A
            rhs += A.T @ Ci @ d
        Nk = N[np.ix_(keep, keep)]
        Cx = np.linalg.inv(Nk + np.eye(keep.sum()) * 1e-12 * np.median(np.diag(Nk)))
        x = np.zeros(npar)
        x[keep] = Cx @ rhs[keep]
        Cf = np.zeros((npar, npar))
        Cf[np.ix_(keep, keep)] = Cx
        chi = {}
        tot = float(x @ (P * x))
        for A, d, C, nm, Ci in blocks:
            r = d - A @ x
            chi[nm] = (float(r @ Ci @ r), len(d))
            tot += chi[nm][0]
        chi['prior'] = (float(x @ (P * x)), 0)
        return dict(x=x, C=Cf, lab=lab, chi=chi, tot=tot, info=info, ndata=sum(len(b[1]) for b in blocks),
                    npar=int(keep.sum()))


def _normal_pieces(pr, q):
    key = ('np', q)
    if key not in pr._cache:
        pr.solve(q)                                     # fills the kappa = 1 block cache
        out, P, lab, info, kc = pr._cache[q]
        pieces = [(nm, A.T @ Ci @ A, A.T @ Ci @ d, float(d @ Ci @ d), len(d), kc.get(nm, [])) for A, d, C, nm, Ci in out]
        pr._cache[key] = (pieces, P, lab)
    return pr._cache[key]


def solve_tot(pr, q, kappa, fix=()):
    """fast total chi2 (data + prior) for the kappa profile, from cached normal-equation pieces (exact)."""
    pieces, P, lab = _normal_pieces(pr, q)
    npar = len(lab)
    keep = np.array([not any(l.startswith(f) for f in fix) for l in lab], dtype=bool)
    N = np.diag(P).astype(float)
    rhs = np.zeros(npar)
    sc = []
    for nm, M, v, c, n, kcol in pieces:
        sv = np.ones(npar)
        k = {'SC': kappa[0], 'LFE': kappa[1], 'REP': kappa[2]}.get(nm, 1.0)
        sv[kcol] = k
        sc.append(sv)
        N += sv[:, None] * M * sv[None, :]
        rhs += sv * v
    Nk = N[np.ix_(keep, keep)]
    x = np.zeros(npar)
    x[keep] = np.linalg.solve(Nk + np.eye(keep.sum()) * 1e-12 * np.median(np.diag(Nk)), rhs[keep])
    tot = float(x @ (P * x))
    for (nm, M, v, c, n, kcol), sv in zip(pieces, sc):
        tot += c - 2 * float((sv * v) @ x) + float(x @ ((sv[:, None] * M * sv[None, :]) @ x))
    return tot


def profile_kappa(pr, qs=QS, fix=()):
    """ONE coupling factor per coda dataset, shared by all quantities; profiled on the summed chi2."""
    active = [k in pr.use for k in ('SC', 'LFE', 'REP')]
    grids = [KGRID if a else [1.0] for a in active]
    best = None
    for a in grids[0]:
        for b in grids[1]:
            for c in grids[2]:
                tot = sum(solve_tot(pr, q, (a, b, c), fix=fix) for q in qs)
                if best is None or tot < best[1]:
                    best = ((a, b, c), tot)
    return best[0]


def deep_report(sol, q, pr, extra=None):
    lab, ix = sol['lab'], sol['info']['ix']
    STA = sol['info']['STA']
    ms = np.zeros(len(lab))
    for s in STA:
        ms[ix[f's|{s}|0']] = 1.0 / len(STA)
    rows = []
    for l in lab:
        if not l.startswith('g|'):
            continue
        i = ix[l]
        e, sd = sol['x'][i], np.sqrt(sol['C'][i, i])
        v = l[2:]
        s1 = pr.tt_res[(pr.tt_res.vol == v) & (pr.tt_res.quantity == q)]
        corr = float(sol['C'][i] @ ms) / np.sqrt(sd ** 2 * (ms @ sol['C'] @ ms) + 1e-30)
        rows.append(dict(quantity=q, vol=v, stage2_conditional_est_pct=e, stage2_conditional_sd_pct=sd,
                         stage2_conditional_z=e / sd, stage2_note='conditional on shallow priors/coda kernels; not for inference',
                         stage1_est_pct=float(s1.est_pct.iloc[0]) if len(s1) else np.nan,
                         stage1_se_pct=float(s1.se.iloc[0]) if len(s1) else np.nan,
                         stage1_p_t=float(s1.p_t.iloc[0]) if len(s1) else np.nan,
                         corr_with_mean_s0=corr, **(extra or {})))
    return rows


CONFIGS = [('primary', {}), ('lp_all', dict(lp_bands=('0.05-0.1', '0.1-0.2', '0.2-0.3'))),
           ('LPonly', dict(use=('LP',))), ('prior_x3', dict(prior_scale=3.0)), ('prior_x10', dict(prior_scale=10.0)),
           ('rho0', dict(rho=0.0)), ('rho0.5', dict(rho=0.5)), ('cm', dict(cm=True)), ('no_u', dict(unuis=False)),
           ('TTonly', dict(use=('TT',))), ('noSC', dict(use=('TT', 'LFE', 'REP', 'LP'))),
           ('noLFE', dict(use=('TT', 'SC', 'REP', 'LP'))), ('noREP', dict(use=('TT', 'SC', 'LFE', 'LP'))),
           ('noLP', dict(use=('TT', 'SC', 'LFE', 'REP'))), ('sc_surf', dict(sc_var='surf')),
           ('sc_body', dict(sc_var='body')), ('sc_l5', dict(sc_var='l5')), ('sc_lvznone', dict(sc_var='lvz_none')),
           ('lfe_a0', dict(lfe_var='a0')), ('lfe_a50', dict(lfe_var='a50')), ('sc_pow', dict(sc_file='coda_obs_sc_pow.csv'))]
RUNS_PRIMARY = ['clean_R_lay_3d', 'clean_R_lay_1d', 'clean_lay_3d', 'clean_lay_1d', 'clean_box_3d', 'clean_box_1d',
                'clean_sph_3d', 'clean_sph_1d', 'clean_slab_3d', 'clean_slab_1d', 'clean_R_only_1d',
                'joint_R_lay_3d', 'joint_lay_3d', 'joint_lay_1d']
RUNS_VARIANTS = ['clean_R_lay_3d', 'clean_R_lay_1d', 'clean_lay_3d']


def _cfg_job(args):
    run, cfg, kw = args
    os.environ['OMP_NUM_THREADS'] = '1'
    pr = Problem(run, **kw)
    kap = profile_kappa(pr)
    kap0 = profile_kappa(pr, fix=('g|',))              # re-profiled for the shallow-only (null) model
    rows, chis, stt = [], [], []
    for q in QS:
        sol = pr.solve(q, kap)
        s0 = pr.solve(q, kap0, fix=('g|',))
        su = pr.solve(q, kap, fix=('u|',)) if pr.unuis else sol
        ex = dict(run=run, cfg=cfg, kappa=str(kap), kappa_null=str(kap0), chi2_total=sol['tot'], ndata=sol['ndata'],
                  npar=sol['npar'], dchi2_shallow_only=s0['tot'] - sol['tot'], dchi2_u=su['tot'] - sol['tot'])
        rows += deep_report(sol, q, pr, ex)
        for k, (c, n) in sol['chi'].items():
            chis.append(dict(run=run, cfg=cfg, quantity=q, dataset=k, chi2=c, n=n, chi2r=c / max(n, 1)))
        if cfg == 'primary':
            for i, l in enumerate(sol['lab']):
                if l.startswith(('s|', 'clock|', 'u|', 'rcm|', 'lp31')):
                    stt.append(dict(run=run, quantity=q, param=l, est=sol['x'][i], sd=np.sqrt(sol['C'][i, i])))
    return rows, chis, stt


def stage_main():
    from multiprocessing import Pool
    jobs = [(r, 'primary', {}) for r in RUNS_PRIMARY]
    jobs += [(r, c, kw) for r in RUNS_VARIANTS for c, kw in CONFIGS if c != 'primary']
    have = set(pd.read_csv(os.environ.get('TT_RESULTS', f'{OUT}/tt_results.csv')).run)
    jobs = [j for j in jobs if os.path.exists(f'{BIG}/tt_{j[0]}.npz') and j[0] in have]
    with Pool(int(os.environ.get('NPROC', '8'))) as pool:
        res = pool.map(_cfg_job, jobs, chunksize=1)
    R = pd.DataFrame([r for a, b, c in res for r in a])
    # TT-vs-LP tension on the regional terms: stage-1 (free receiver terms, honest SE) vs LP-only stage-2
    lpo = R[R.cfg == 'LPonly'].set_index(['run', 'quantity', 'vol'])
    tens = []
    for (run, q, v), r in lpo.iterrows():
        a = R[(R.run == run) & (R.cfg == 'primary') & (R.quantity == q) & (R.vol == v)]
        if len(a):
            d = a.stage1_est_pct.iloc[0] - r.stage2_conditional_est_pct
            v2 = a.stage1_se_pct.iloc[0] ** 2 + r.stage2_conditional_sd_pct ** 2
            tens.append(dict(run=run, quantity=q, vol=v, tt_stage1=a.stage1_est_pct.iloc[0], tt_se=a.stage1_se_pct.iloc[0],
                             lp_only=r.stage2_conditional_est_pct, lp_sd=r.stage2_conditional_sd_pct, dchi2_tension=d * d / v2))
    pd.DataFrame(tens).to_csv(f'{OUT}/gls_tt_lp_tension.csv', index=False, float_format='%.5g')
    R.to_csv(f'{OUT}/gls_results.csv', index=False, float_format='%.5g')
    pd.DataFrame([r for a, b, c in res for r in b]).to_csv(f'{OUT}/gls_dataset_chi2.csv', index=False, float_format='%.4g')
    pd.DataFrame([r for a, b, c in res for r in c]).to_csv(f'{OUT}/gls_station_terms.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250):
        print(R[R.cfg == 'primary'][['run', 'quantity', 'vol', 'stage1_est_pct', 'stage1_se_pct', 'stage1_p_t',
                                     'stage2_conditional_est_pct', 'stage2_conditional_sd_pct', 'dchi2_shallow_only', 'dchi2_u',
                                     'kappa']].round(4).to_string())
        print(pd.DataFrame(tens).round(4).to_string())


# ------------------------------------------------------------------------------------------ generic mapping
def invert_component(df, kappa=1.0, sc_var='default', lfe_var='default', corr_q='D'):
    """df: rows dataset ('SC' | 'LFEcoda' | 'REP'), sta, band (SC band name or '2-8Hz'), amp (%), sd (%).
    Per station: amp = kappa (f0 s0 + f1 s1 + f2 s2 + fdeep u), s = dv/v (%) in 0-0.1 / 0.1-1 / 1-3 km (priors
    5 / 1 / 0.5 %), u = deep-coda nuisance (prior 1 %; the coda does not resolve depth below 3 km).  The SC band
    errors are correlated with the inter-band correlation of the season placebos (quantity corr_q).
    Returns station x depth-bin estimates, SD, averaging-kernel rows, and dchi2 for u."""
    kf = pd.read_csv(f'{OUT}/coda_depth_frac.csv').set_index(['dataset', 'sta', 'band', 'variant'])
    # project-internal pickle written by scripts/sci_dvv_3d_codafit.py (trusted)
    Pl = pickle.load(open(f'{OUT}/sc_placebo.pkl', 'rb'))
    Rb = np.eye(5)
    for i, b1 in enumerate(SC_BANDS):
        for j, b2 in enumerate(SC_BANDS):
            if j > i:
                rs = []
                for k in Pl:
                    if k[1] == b1 and k[2] == corr_q and (k[0], b2, corr_q) in Pl:
                        a, c = Pl[k], Pl[(k[0], b2, corr_q)]
                        sh = sorted(set(a) & set(c))
                        if len(sh) >= 5:
                            rs.append(np.corrcoef([a[s] for s in sh], [c[s] for s in sh])[0, 1])
                Rb[i, j] = Rb[j, i] = np.nanmedian(rs) if rs else 0.0
    Rb = _psd_corr(Rb)
    out = []
    pri = np.r_[1 / PRIOR_S ** 2, 1 / PRIOR_U ** 2]
    for sta, g in df.groupby('sta'):
        A, d, se, bi = [], [], [], []
        for r in g.itertuples():
            key = ('SC', sta, r.band, sc_var) if r.dataset == 'SC' else ('LFE', sta, '2-8Hz', lfe_var)
            if key not in kf.index or not np.isfinite(r.amp) or not (r.sd > 0):
                continue
            A.append(kappa * kf.loc[key, ['f0', 'f1', 'f2', 'fdeep']].values.astype(float))
            d.append(r.amp)
            se.append(r.sd)
            bi.append(SC_BANDS.index(r.band) if r.dataset == 'SC' else -1)
        if not A:
            continue
        A, d, se, bi = np.array(A), np.array(d), np.array(se), np.array(bi)
        C = np.diag(se ** 2)
        for i in range(len(d)):
            for j in range(len(d)):
                if i != j and bi[i] >= 0 and bi[j] >= 0:
                    C[i, j] = Rb[bi[i], bi[j]] * se[i] * se[j]
        Ci = np.linalg.inv(C)
        N = A.T @ Ci @ A + np.diag(pri)
        Cx = np.linalg.inv(N)
        x = Cx @ (A.T @ Ci @ d)
        Rm = Cx @ (A.T @ Ci @ A)
        chi = float((d - A @ x) @ Ci @ (d - A @ x))
        N3 = A[:, :3].T @ Ci @ A[:, :3] + np.diag(pri[:3])
        x3 = np.linalg.solve(N3, A[:, :3].T @ Ci @ d)
        chi3 = float((d - A[:, :3] @ x3) @ Ci @ (d - A[:, :3] @ x3)) + float(x3 @ (pri[:3] * x3))
        dchi_u = chi3 - (chi + float(x @ (pri * x)))
        names = [f'{a:g}-{b:g}' for a, b in BINS] + ['u_deep']
        for k in range(4):
            out.append(dict(sta=sta, zbin=names[k], est=x[k], sd=np.sqrt(Cx[k, k]), res_diag=Rm[k, k],
                            ak=';'.join(f'{v:.3f}' for v in Rm[k]), n=len(d), chi2=chi, dchi2_u=dchi_u))
    return pd.DataFrame(out)


def stage_components():
    """map the attribution components (data/sci/dvv_attr/components.csv, rows flagged for_depth_mapping) to depth:
    SC10 rows (bands 0.1-0.3 ... 3-5 Hz; the full 0.1-1 Hz band is skipped) and LFE / REP coda rows (2-8 Hz kernel)
    per station -> dv/v in 0-0.1 / 0.1-1 / 1-3 km + deep-coda nuisance u (invert_component).  Annual amplitudes are
    mapped as if in phase across bands (approximation)."""
    f = f'{ROOT}/data/sci/dvv_attr/components.csv'
    if not os.path.exists(f):
        print('components.csv not found')
        return
    C = pd.read_csv(f)
    C = C[C.for_depth_mapping.astype(str) == 'True']
    C = C[C.dataset.isin(['SC10', 'LFE', 'REP']) & (C.chan_band != '0.1-1Hz') & np.isfinite(C.amp) & (C.sd > 0)]
    res = []
    for comp, g in C.groupby('component'):
        df = pd.DataFrame(dict(dataset=np.where(g.dataset == 'SC10', 'SC', 'LFEcoda'), sta=g.sta.values,
                               band=np.where(g.dataset == 'SC10', g.chan_band, '2-8Hz'), amp=g.amp.values, sd=g.sd.values))
        r = invert_component(df)
        if len(r):
            r.insert(0, 'component', comp)
            res.append(r)
    R = pd.concat(res, ignore_index=True)
    R['z'] = R.est / R.sd
    R.to_csv(f'{OUT}/component_depth.csv', index=False, float_format='%.5g')
    S = R.groupby(['component', 'zbin']).apply(lambda g: pd.Series(dict(
        n=len(g), median=g.est.median(), ivw_mean=float((g.est / g.sd ** 2).sum() / (1 / g.sd ** 2).sum()),
        ivw_se=float(1 / np.sqrt((1 / g.sd ** 2).sum())), n_z_gt2=int((g.z.abs() > 2).sum()), med_res=g.res_diag.median(),
        med_dchi2_u=g.dchi2_u.median()))).reset_index()
    S.to_csv(f'{OUT}/component_depth_summary.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250):
        print(S.round(4).to_string())


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'kern': stage_kern, 'main': stage_main, 'components': stage_components}[st]()
