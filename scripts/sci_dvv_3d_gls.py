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
  Coda kernels are truncated at 3 km (coda does not vote on depth below 3 km); kappa (coda/kernel coupling) is
  profiled over 0.25-4 per coda dataset.  SC covariance: inter-band correlation (season placebos) and network
  correlation per band from SIMULTANEOUS season-matched placebo fits (same shifted dates at all stations).
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
KGRID = [0.25, 0.5, 1.0, 2.0, 4.0]
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
                rows.append(dict(dataset='SC', sta=r.ns, band=b, variant=v, f0=f[0], f1=f[1], f2=f[2], fdeep=f[3]))
        vm = KK.station_vmodel(r.lat, r.lon, 'ref')
        body = K2.lfe_body_kernel(vm, KK.Z_E, (3.0, 20.0), l_km='20', dist=40, bh_depth=r.z0_km,
                                  f_shift=0.2 if r.z0_km > 0 else 0.0)
        Kz = 0.5 * (Kn.sw[f'{r.ns}|ref|1-3Hz|Kz'] + Kn.sw[f'{r.ns}|ref|3-5Hz|Kz'])
        rr = 0.5 * (float(Kn.sw[f'{r.ns}|ref|1-3Hz|r']) + float(Kn.sw[f'{r.ns}|ref|3-5Hz|r'])) if r.z0_km > 0 else 1.0
        for v, a in (('default', 0.25), ('a0', 0.0), ('a50', 0.5)):
            ae = a * rr / (a * rr + 1 - a + 1e-12)
            f = depth_fracs((1 - ae) * body / body.sum() + ae * Kz)
            rows.append(dict(dataset='LFE', sta=r.ns, band='2-8Hz', variant=v, f0=f[0], f1=f[1], f2=f[2], fdeep=f[3]))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/coda_depth_frac.csv', index=False, float_format='%.4g')
    print(T[T.variant == 'default'].groupby(['dataset', 'band'])[['f0', 'f1', 'f2', 'fdeep']].median().round(3))


# ------------------------------------------------------------------------------------------ data assembly
class Problem:
    def __init__(self, run='joint_R_lay', sc_var='default', lfe_var='default', use=('TT', 'SC', 'LFE', 'REP', 'LP'),
                 cm=False, sc_file='coda_obs_sc.csv'):
        self.run, self.use, self.cm = run, set(use), cm
        Z = np.load(f'{BIG}/tt_{run}.npz', allow_pickle=False)
        self.sp = [tuple(x.split('|')) for x in Z['sp']]
        self.vols = list(Z['vols'])
        self.xg, self.cov = Z['xg'], Z['cov']
        self.nb = len(self.sp) * NJ
        R = pd.read_csv(f'{OUT}/tt_results.csv')
        R = R[R.run == run]
        self.hon = {(r.vol, r.quantity): max(1.0, r.se / r.se_formal) for r in R.itertuples()}
        self.tt_res = R
        G = pd.read_csv(f'{OUT}/tt_geometry.csv')
        G = G[G.lvz == 'ref']
        self.Tsh = G.groupby(['sta', 'phase'])[['T0_0.1', 'T0.1_1', 'T1_3']].mean()
        self.kf = pd.read_csv(f'{OUT}/coda_depth_frac.csv')
        self.sc = pd.read_csv(f'{OUT}/{sc_file}')
        self.lfe = pd.read_csv(f'{OUT}/coda_obs_lfe.csv')
        self.lp = pd.read_csv(f'{OUT}/coda_obs_lp.csv')
        self.rep = pd.read_csv(f'{ROOT}/data/sci/dvv_deep_shallow/rep/rep_fit.csv')
        self.sc_var, self.lfe_var = sc_var, lfe_var
        self.corr = self._sc_corr(sc_file)
        lpk = json.load(open(f'{OUT}/lp_depth_kernels.json')) if os.path.exists(f'{OUT}/lp_depth_kernels.json') else None
        self.lpk = lpk
        self._cache = {}

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
                out[(q, b)] = float(np.clip(np.nanmedian(rs), 0, 0.9)) if rs else 0.0
            # inter-band (same station)
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
                    M[i, j] = M[j, i] = float(np.clip(np.nanmedian(rs), -0.9, 0.9)) if rs else 0.0
            out[(q, 'bands')] = M
        return out

    def stations(self):
        st = set(s for s, _ in self.sp) if 'TT' in self.use else set()
        if 'SC' in self.use:
            st |= set(self.sc.sta)
        if 'LFE' in self.use:
            st |= set(self.lfe.sta)
        if 'REP' in self.use:
            st |= set(self.rep.sta)
        return sorted(st)

    def build(self, q, kappa=(1.0, 1.0, 1.0), tt_override=None, coda_override=None):
        """returns design A, data d, covariance C, prior precision P (diag), labels."""
        STA = self.stations()
        si = {s: i for i, s in enumerate(STA)}
        ttsta = sorted(set(s for s, _ in self.sp)) if 'TT' in self.use else []
        ci = {s: i for i, s in enumerate(ttsta)}
        nv = len(self.vols) if 'TT' in self.use else 0
        lab = [f's|{s}|{k}' for s in STA for k in range(3)] + [f'clock|{s}' for s in ttsta] + \
              [f'g|{v}' for v in self.vols[:nv]] + ([f'cm|{b}' for b in SC_BANDS] if self.cm and 'SC' in self.use else [])
        npar = len(lab)
        o_clk, o_g = 3 * len(STA), 3 * len(STA) + len(ttsta)
        o_cm = o_g + nv
        blocks = []          # (A, d, C, name)
        Qv = qvec(q)
        if 'TT' in self.use:
            xg = self.xg if tt_override is None else tt_override
            nsp = len(self.sp)
            Tm = np.zeros((nsp + nv, len(xg)))
            for k in range(nsp):
                Tm[k, k * NJ:(k + 1) * NJ] = Qv
            for v in range(nv):
                Tm[nsp + v, self.nb + v * NJ:self.nb + (v + 1) * NJ] = Qv * 100.0     # -> %
            z = Tm @ xg
            Cz = Tm @ self.cov @ Tm.T
            f = np.ones(nsp + nv)
            fmax = max([self.hon.get((v, q), 1.0) for v in self.vols] + [1.0])
            f[:nsp] = fmax
            for v in range(nv):
                f[nsp + v] = self.hon.get((self.vols[v], q), 1.0)
            Cz = Cz * np.outer(f, f)
            A = np.zeros((nsp + nv, npar))
            for k, (s, ph) in enumerate(self.sp):
                T = self.Tsh.loc[(s, ph)].values if (s, ph) in self.Tsh.index else np.zeros(3)
                for b in range(3):
                    A[k, 3 * si[s] + b] = -T[b] / 100.0
                A[k, o_clk + ci[s]] = 1.0
            for v in range(nv):
                A[nsp + v, o_g + v] = 1.0
            blocks.append((A, z, Cz, 'TT'))
        kf = self.kf.set_index(['dataset', 'sta', 'band', 'variant'])
        if 'SC' in self.use:
            g = self.sc[(self.sc.quantity == q)].copy()
            g = g[g.sta.isin(si) & np.isfinite(g.se) & (g.se > 0)]
            if coda_override is not None and 'SC' in coda_override:
                g = coda_override['SC']
            A = np.zeros((len(g), npar))
            for i, r in enumerate(g.itertuples()):
                fr = kf.loc[('SC', r.sta, r.band, self.sc_var), ['f0', 'f1', 'f2']].values
                A[i, 3 * si[r.sta]:3 * si[r.sta] + 3] = kappa[0] * fr
                if self.cm:
                    A[i, o_cm + SC_BANDS.index(r.band)] = 1.0
            se = g.se.values
            rho = self.corr[(q, 'bands')]
            C = np.zeros((len(g), len(g)))
            bands = g.band.values
            stas = g.sta.values
            for i in range(len(g)):
                for j in range(len(g)):
                    if i == j:
                        C[i, j] = se[i] ** 2
                    elif stas[i] == stas[j]:
                        C[i, j] = rho[SC_BANDS.index(bands[i]), SC_BANDS.index(bands[j])] * se[i] * se[j]
                    elif bands[i] == bands[j]:
                        C[i, j] = self.corr[(q, bands[i])] * se[i] * se[j]
            # the two correlation models (inter-band within a station, network within a band) need not combine into
            # a positive-definite matrix: project onto the nearest PSD matrix (eigenvalue floor 5 % of the mean var)
            ev, V = np.linalg.eigh(C)
            ev = np.maximum(ev, 0.05 * np.mean(se ** 2))
            C = (V * ev) @ V.T
            blocks.append((A, g.est.values, C, 'SC'))
        if 'LFE' in self.use:
            g = self.lfe[(self.lfe.quantity == q) & self.lfe.sta.isin(si)]
            if coda_override is not None and 'LFE' in coda_override:
                g = coda_override['LFE']
            A = np.zeros((len(g), npar))
            for i, r in enumerate(g.itertuples()):
                if ('LFE', r.sta, '2-8Hz', self.lfe_var) in kf.index:
                    A[i, 3 * si[r.sta]:3 * si[r.sta] + 3] = kappa[1] * kf.loc[('LFE', r.sta, '2-8Hz', self.lfe_var), ['f0', 'f1', 'f2']].values
            blocks.append((A, g.est.values, np.diag(g.se.values ** 2), 'LFE'))
        if 'REP' in self.use and q in ('m7', 'D'):
            col, secol = ('m7_pct', 'm7_se_boot') if q == 'm7' else ('D_pct', 'D_se_boot')
            g = self.rep[np.isfinite(self.rep[col]) & self.rep.sta.isin(si)]
            if coda_override is not None and 'REP' in coda_override:
                g = coda_override['REP']
            A = np.zeros((len(g), npar))
            for i, r in enumerate(g.itertuples()):
                if ('LFE', r.sta, '2-8Hz', self.lfe_var) in kf.index:
                    A[i, 3 * si[r.sta]:3 * si[r.sta] + 3] = kappa[2] * kf.loc[('LFE', r.sta, '2-8Hz', self.lfe_var), ['f0', 'f1', 'f2']].values
            blocks.append((A, g[col].values, np.diag((1.5 * g[secol].values) ** 2), 'REP'))
        if 'LP' in self.use and self.lpk is not None and all(v in self.vols for v in ('R1_10', 'R10_20', 'lay')):
            g = self.lp[self.lp.quantity == q]
            if coda_override is not None and 'LP' in coda_override:
                g = coda_override['LP']
            A = np.zeros((len(g), npar))
            for i, r in enumerate(g.itertuples()):
                lf = np.array(self.lpk[r.band]['layer_frac'])   # layers of sci_dvv_3d_G.LAY
                # LAY = 0,0.1,0.3,1,3,6,10,15,20,25,31,35,250
                f0, f1, f2 = lf[0], lf[1] + lf[2], lf[3]
                fR = lf[4] + lf[5]
                fR2 = lf[6] + lf[7]
                fD = lf[8:].sum()
                for s in STA:
                    A[i, 3 * si[s]:3 * si[s] + 3] = np.array([f0, f1, f2]) / len(STA)
                A[i, o_g + self.vols.index('R1_10')] = fR
                A[i, o_g + self.vols.index('R10_20')] = fR2
                A[i, o_g + self.vols.index('lay')] = fD
            blocks.append((A, g.est.values, np.diag(g.se.values ** 2), 'LP'))
        P = np.zeros(npar)
        for s in STA:
            P[3 * si[s]:3 * si[s] + 3] = 1 / PRIOR_S ** 2
        P[o_clk:o_g] = 1 / PRIOR_CLOCK ** 2
        return blocks, P, lab, dict(o_g=o_g, nv=nv, STA=STA)

    def _blocks(self, q, kappa, **kw):
        """blocks with inverse covariances; without overrides they are built once per quantity (kappa = 1) and the
        coda blocks' station-term columns are rescaled by kappa (C does not depend on kappa)."""
        if kw:
            blocks, P, lab, info = self.build(q, kappa, **kw)
            return [(A, d, C, nm, np.linalg.pinv(C, rcond=1e-10, hermitian=True)) for A, d, C, nm in blocks], P, lab, info
        if q not in self._cache:
            blocks, P, lab, info = self.build(q, (1.0, 1.0, 1.0))
            self._cache[q] = ([(A, d, C, nm, np.linalg.pinv(C, rcond=1e-10, hermitian=True)) for A, d, C, nm in blocks],
                              P, lab, info)
        blocks, P, lab, info = self._cache[q]
        ns = 3 * len(info['STA'])
        out = []
        for A, d, C, nm, Ci in blocks:
            k = {'SC': kappa[0], 'LFE': kappa[1], 'REP': kappa[2]}.get(nm, 1.0)
            if k != 1.0:
                A = A.copy()
                A[:, :ns] *= k
            out.append((A, d, C, nm, Ci))
        return out, P, lab, info

    def solve(self, q, kappa=(1.0, 1.0, 1.0), fix_g0=False, **kw):
        blocks5, P, lab, info = self._blocks(q, kappa, **kw)
        blocks = [b[:4] for b in blocks5]
        npar = len(lab)
        N = np.diag(P)
        rhs = np.zeros(npar)
        keep = np.ones(npar, bool)
        if fix_g0:
            keep[info['o_g']:info['o_g'] + info['nv']] = False
        Wb = []
        for (A, d, C, name), Ci in zip(blocks, [b[4] for b in blocks5]):
            N += A.T @ Ci @ A
            rhs += A.T @ Ci @ d
            Wb.append(Ci)
        Nk = N[np.ix_(keep, keep)]
        Nk += np.eye(keep.sum()) * 1e-12 * np.median(np.diag(Nk))
        Cx = np.linalg.inv(Nk)
        x = np.zeros(npar)
        x[keep] = Cx @ rhs[keep]
        Cfull = np.zeros((npar, npar))
        Cfull[np.ix_(keep, keep)] = Cx
        chi = {}
        tot = float(x @ (P * x))
        for (A, d, C, name), Ci in zip(blocks, Wb):
            r = d - A @ x
            chi[name] = (float(r @ Ci @ r), len(d))
            tot += chi[name][0]
        chi['prior'] = (float(x @ (P * x)), 0)
        return dict(x=x, C=Cfull, lab=lab, chi=chi, tot=tot, info=info, ndata=sum(len(b[1]) for b in blocks),
                    npar=int(keep.sum()))


def deep_report(sol, q, extra=None):
    rows = []
    lab = sol['lab']
    o_g, nv = sol['info']['o_g'], sol['info']['nv']
    STA = sol['info']['STA']
    ms = np.zeros(len(lab))
    for s in STA:
        ms[lab.index(f's|{s}|0')] = 1.0 / len(STA)
    for v in range(nv):
        i = o_g + v
        e, sd = sol['x'][i], np.sqrt(sol['C'][i, i])
        cov_ms = float(sol['C'][i] @ ms)
        corr_ms = cov_ms / np.sqrt(sd ** 2 * (ms @ sol['C'] @ ms) + 1e-30)
        rows.append(dict(quantity=q, vol=lab[i][2:], est_pct=e, sd_pct=sd, z=e / sd, corr_with_mean_s0=corr_ms, **(extra or {})))
    return rows


def best_kappa(pr, q, **kw):
    best = None
    for a in KGRID:
        for b in KGRID:
            for c in KGRID:
                s = pr.solve(q, (a, b, c), **kw)
                if best is None or s['tot'] < best[1]['tot']:
                    best = ((a, b, c), s)
    return best


# ------------------------------------------------------------------------------------------ main runs
def stage_main():
    rows, chis, stt, comp = [], [], [], []
    for run in ['joint_R_lay', 'joint_lay', 'joint_R_box', 'joint_box', 'joint_R_sph', 'joint_sph', 'joint_R_slab', 'joint_slab']:
        if not os.path.exists(f'{BIG}/tt_{run}.npz'):
            continue
        for cfg, kw in [('primary', {}), ('cm', dict(cm=True)), ('TTonly', dict(use=('TT',))),
                        ('noTT_coda', dict(use=('SC', 'LFE', 'REP'))),
                        ('noSC', dict(use=('TT', 'LFE', 'REP', 'LP'))), ('noLFE', dict(use=('TT', 'SC', 'REP', 'LP'))),
                        ('noREP', dict(use=('TT', 'SC', 'LFE', 'LP'))), ('noLP', dict(use=('TT', 'SC', 'LFE', 'REP'))),
                        ('sc_surf', dict(sc_var='surf')), ('sc_body', dict(sc_var='body')), ('sc_l5', dict(sc_var='l5')),
                        ('sc_lvznone', dict(sc_var='lvz_none')), ('lfe_a0', dict(lfe_var='a0')), ('lfe_a50', dict(lfe_var='a50')),
                        ('sc_pow', dict(sc_file='coda_obs_sc_pow.csv'))]:
            if cfg != 'primary' and run not in ('joint_R_lay', 'joint_sph', 'joint_box', 'joint_lay'):
                continue
            if cfg == 'sc_pow' and not os.path.exists(f'{OUT}/coda_obs_sc_pow.csv'):
                continue
            pr = Problem(run, **kw)
            for q in QS:
                kap, sol = best_kappa(pr, q)
                s0 = pr.solve(q, kap, fix_g0=True)
                ex = dict(run=run, cfg=cfg, kappa=str(kap), chi2_total=sol['tot'], ndata=sol['ndata'], npar=sol['npar'],
                          dchi2_shallow_only=s0['tot'] - sol['tot'],
                          dBIC=(s0['tot'] - sol['tot']) - sol['info']['nv'] * np.log(max(sol['ndata'], 2)))
                # deep estimate range over the kappa grid
                gr = [pr.solve(q, (a, b, c)) for a in (0.25, 1, 4) for b in (0.25, 1, 4) for c in (0.25, 1, 4)]
                rr = deep_report(sol, q, ex)
                for i, r in enumerate(rr):
                    vals = [g_['x'][sol['info']['o_g'] + i] for g_ in gr]
                    r['est_min_kgrid'], r['est_max_kgrid'] = float(np.min(vals)), float(np.max(vals))
                rows += rr
                for k, (c, n) in sol['chi'].items():
                    chis.append(dict(run=run, cfg=cfg, quantity=q, dataset=k, chi2=c, n=n, chi2r=c / max(n, 1)))
                if cfg == 'primary':
                    for i, l in enumerate(sol['lab']):
                        if l.startswith('s|') or l.startswith('clock|'):
                            stt.append(dict(run=run, quantity=q, param=l, est=sol['x'][i], sd=np.sqrt(sol['C'][i, i])))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/gls_results.csv', index=False, float_format='%.5g')
    pd.DataFrame(chis).to_csv(f'{OUT}/gls_dataset_chi2.csv', index=False, float_format='%.4g')
    pd.DataFrame(stt).to_csv(f'{OUT}/gls_station_terms.csv', index=False, float_format='%.5g')
    show = R[(R.cfg == 'primary')]
    print(show[['run', 'quantity', 'vol', 'est_pct', 'sd_pct', 'z', 'dchi2_shallow_only', 'dBIC', 'kappa', 'est_min_kgrid', 'est_max_kgrid']].round(4).to_string())
    print(R[(R.run == 'joint_R_lay') & (R.vol == 'lay')][['cfg', 'quantity', 'est_pct', 'sd_pct', 'z']].round(4).to_string())


# ------------------------------------------------------------------------------------------ generic mapping
def invert_component(df, kappa=1.0, sc_var='default', lfe_var='default'):
    """df: rows dataset ('SC' | 'LFEcoda' | 'REP'), sta, band (SC band name / '2-8Hz'), amp (%), sd (%).
    Maps a per-station x band amplitude vector to per-station dv/v in 0-0.1, 0.1-1, 1-3 km (coda kernels truncated
    at 3 km, as for the primary model; priors 5 / 1 / 0.5 %).  Returns station terms with SD and, per station,
    the 3x3 resolution (averaging-kernel) matrix over the depth bins."""
    kf = pd.read_csv(f'{OUT}/coda_depth_frac.csv').set_index(['dataset', 'sta', 'band', 'variant'])
    out = []
    for sta, g in df.groupby('sta'):
        A, d, w = [], [], []
        for r in g.itertuples():
            key = ('SC', sta, r.band, sc_var) if r.dataset == 'SC' else ('LFE', sta, '2-8Hz', lfe_var)
            if key not in kf.index or not np.isfinite(r.amp) or not (r.sd > 0):
                continue
            A.append(kappa * kf.loc[key, ['f0', 'f1', 'f2']].values.astype(float))
            d.append(r.amp)
            w.append(1 / r.sd ** 2)
        if not A:
            continue
        A, d, w = np.array(A), np.array(d), np.array(w)
        N = A.T @ (A * w[:, None]) + np.diag(1 / PRIOR_S ** 2)
        Cx = np.linalg.inv(N)
        x = Cx @ (A.T @ (w * d))
        Rm = Cx @ (A.T @ (A * w[:, None]))
        chi = float(np.sum(w * (d - A @ x) ** 2))
        for k, (a, b) in enumerate(BINS):
            out.append(dict(sta=sta, zbin=f'{a:g}-{b:g}', est=x[k], sd=np.sqrt(Cx[k, k]), res_diag=Rm[k, k],
                            ak=';'.join(f'{v:.3f}' for v in Rm[k]), n=len(d), chi2=chi))
    return pd.DataFrame(out)


def stage_components():
    f = f'{ROOT}/data/sci/dvv_attr/components.csv'
    if not os.path.exists(f):
        print('components.csv not found')
        return
    C = pd.read_csv(f)
    print(C.columns.tolist(), len(C))
    C = C.rename(columns={c: c for c in C.columns})
    bandcol = 'band' if 'band' in C.columns else ('chan' if 'chan' in C.columns else None)
    res = []
    for comp, g in C.groupby('component'):
        df = pd.DataFrame(dict(dataset=np.where(g.dataset.astype(str).str.upper().str.startswith('SC'), 'SC', 'LFEcoda'),
                               sta=g.sta.values, band=g[bandcol].values if bandcol else '', amp=g.amp.values, sd=g.sd.values))
        r = invert_component(df)
        if len(r):
            r.insert(0, 'component', comp)
            res.append(r)
    if res:
        R = pd.concat(res, ignore_index=True)
        R.to_csv(f'{OUT}/component_depth.csv', index=False, float_format='%.5g')
        print(R.groupby(['component', 'zbin']).agg(n=('est', 'size'), med=('est', 'median'), med_sd=('sd', 'median'),
                                                   res=('res_diag', 'median')).round(3).to_string())


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'kern': stage_kern, 'main': stage_main, 'components': stage_components}[st]()
