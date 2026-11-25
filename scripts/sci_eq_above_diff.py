#!/usr/bin/env python3
"""EQ-above step 2b: LFE-minus-earthquake differential time-lapse inversion for the layer between the earthquake
cluster depth (~22 km) and the LFEs (~25-31 km).

Data
  LFE : epoch-stack P and S delays per family x channel (data/sci/dvv_source_region/meas, loaded with
        sci_dvv_source_region.load_rows: EXCL dropped, BREAKS2 segments).
  EQ  : per-event P and S delays per channel of the catalog earthquake families (meas_cat.parquet, pass-2,
        cc >= 0.6 S / 0.5 P), rows flagged as outliers by the relocation (|res| > 4 sigma) dropped;
        se = robust per station-phase scatter from the relocation.
Model (all sources s = LFE stack or EQ event; channel c at station k; phase X):
  y = tau_s + g^X_{s,k} . dx_s + off_{fam,c,X,seg} + sum_j beta_{k,X,j} f_j(t) - K^X_{s,k} sum_j gamma_j f_j(t)
  f_j = time functions of sci_dvv_source_region (annual sin/cos, Petrolia/Ferndale/M7 steps, 60-d early windows,
        ramp from 2021-06);  beta = receiver / shared-path terms per STATION and phase (common to LFE and EQ rays);
  K^X (s) = time the LFE ray spends in the layer between Z_TOP and the family depth (straight ray along the 3D-model
        take-off direction; 0 for earthquakes, which sit at/above Z_TOP).  gamma = dv/v of that layer.
  Local parameters (tau, dx per source; offsets) are eliminated per family (Schur complement); IRLS Huber.
Variants: joint (beta free) | lfe_nobeta (beta = 0: LFE-only, uncorrected) | joint_steps (beta, gamma: steps + ramp
          + annual only).
Errors: formal; bootstrap over sources within families (B=40); placebo: all model dates shifted back 5.25..12.25 yr,
        fit on pre-2021-06 data (15 shifts).  Honest SE = max(boot, placebo).
Injections: (A) layer step -0.3% at the M7 into the LFE rows; (B) common receiver step (S +5 ms, P +2 ms at every
        station, M7) into LFE and EQ rows; (C) as B but EQ rows get 0.7x (non-common receiver term);
        (D) EQ sources moved dz +50 m after the M7.
Outputs data/sci/eq_above/diff_results.csv, diff_beta.csv, geom_layer.csv; summary in summary.json (stage summary).
Usage: nice -n 10 python scripts/sci_eq_above_diff.py --stage geom|fit [--ztop 22]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "2")
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
OUT = f'{ROOT}/data/sci/eq_above'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_source_region as SR_  # noqa: E402
import sci_eq_above_lapse as LA  # noqa: E402

Z_TOP = 22.0
GEOM_TAG = ''
T_Q = SR_.T_Q
YR = SR_.YR
DAY = SR_.DAY
NJ = len(SR_.PN)
RNG = np.random.default_rng(5)


# ------------------------------------------------------------------------------------------------ geometry
def stage_geom(a):
    grid = SR_.Grid('3d')
    G = pd.read_csv('data/sci/dvv_source_region/geometry_3d.csv')
    sh = pd.read_csv('data/families/shelly_family_locations.csv').set_index('idnum')
    rows = []
    for r in G.itertuples():
        src = np.array([r.x, r.y, r.z])
        if a.zsrc == 'shelly':
            src[2] = float(sh.loc[r.F, 'depth_km'])          # Shelly depths (~2.5 km shallower than ours)
        d = dict(F=r.F, sta=r.sta)
        for ph in ('S', 'P'):
            I = grid.interp(r.sta, ph.lower())
            g = np.array([getattr(r, f'g{ph}{c}') for c in 'xyz'])
            u = -g / np.linalg.norm(g)
            cosi = max(abs(u[2]), 0.05)
            L = max(src[2] - a.ztop, 0.0) / cosi
            p = src + L * u
            T0 = float(I(src)[0])
            if grid.inside(p) and np.isfinite(float(I(p)[0])):
                K = T0 - float(I(p)[0])
            else:
                K = L * np.linalg.norm(g)
            d[f'K{ph}_lay'] = K
            d[f'L{ph}_lay'] = L
        rows.append(d)
    K = pd.DataFrame(rows)
    K.to_csv(f'{OUT}/geom_layer{"_shelly" if a.zsrc == "shelly" else ""}.csv', index=False)
    print(K.groupby('F')[['KS_lay', 'KP_lay', 'LS_lay']].median().round(2).to_string())
    print('median K_S - K_P:', (K.KS_lay - K.KP_lay).median())


# ------------------------------------------------------------------------------------------------ data
def load_lfe():
    G = pd.read_csv('data/sci/dvv_source_region/geometry_3d.csv')
    K = pd.read_csv(f'{OUT}/geom_layer{GEOM_TAG}.csv')
    G = G.merge(K, on=['F', 'sta'])
    out = []
    for F in range(1, 28):
        L = SR_.load_rows(F)
        if L is None:
            continue
        R, _ = L
        R = R.merge(G[G.F == F][['sta', 'gSx', 'gSy', 'gSz', 'gPx', 'gPy', 'gPz', 'KS_lay', 'KP_lay']], on='sta')
        R['fam'] = f'L{F}'
        R['src'] = f'L{F}_' + R['stack'].astype(str)
        out.append(R)
    R = pd.concat(out, ignore_index=True)
    ph = R.phase.values
    R['gx'] = np.where(ph == 'S', R.gSx, R.gPx)
    R['gy'] = np.where(ph == 'S', R.gSy, R.gPy)
    R['gz'] = np.where(ph == 'S', R.gSz, R.gPz)
    R['K'] = np.where(ph == 'S', R.KS_lay, R.KP_lay)
    R['kind'] = 'L'
    return R[['fam', 'src', 'kind', 't', 'chan', 'sta', 'phase', 'seg', 'y', 'se', 'gx', 'gy', 'gz', 'K']]


def load_eq():
    d = LA.load_meas()
    rr = pd.read_parquet(f'{OUT}/reloc_resid.parquet')[['efam', 'ev', 'chan', 'phase', 'res', 'sig']]
    d = d.merge(rr, on=['efam', 'ev', 'chan', 'phase'], how='inner')
    d = d[np.abs(d.res) <= 4 * d.sig]
    G = pd.read_csv(f'{OUT}/geom_eq.csv')
    d = d.merge(G[['efam', 'sta', 'gSx', 'gSy', 'gSz', 'gPx', 'gPy', 'gPz']], on=['efam', 'sta'])
    ph = d.phase.values
    d['gx'] = np.where(ph == 'S', d.gSx, d.gPx)
    d['gy'] = np.where(ph == 'S', d.gSy, d.gPy)
    d['gz'] = np.where(ph == 'S', d.gSz, d.gPz)
    d['K'] = 0.0
    d['fam'] = 'E' + d.efam.astype(str)
    d['src'] = d.fam + '_' + d.ev.astype(str)
    d['se'] = np.sqrt(d.sig ** 2 + 0.001 ** 2)
    d['kind'] = 'E'
    d['t'] = d.t.astype(float)
    return d[['fam', 'src', 'kind', 't', 'chan', 'sta', 'phase', 'seg', 'y', 'se', 'gx', 'gy', 'gz', 'K']]


# ------------------------------------------------------------------------------------------------ solver
YEARS = list(range(2010, 2027))
_YE = [SR_._ep(f'{y}-01-01') for y in YEARS + [2027]]


def ycols(t, s=0.0):
    """yearly-bin indicators 2010..2026 (2008-2009 = reference)."""
    return np.array([((t >= _YE[i]) & (t < _YE[i + 1])).astype(float) for i in range(len(YEARS))]).T


class Joint:
    def __init__(self, D, beta=True, cols=None, yearly=False):
        self.D = D.reset_index(drop=True)
        self.beta = beta
        self.yearly = yearly
        if yearly:
            cols = np.ones(len(YEARS), bool)
        self.cols = np.ones(NJ, bool) if cols is None else cols
        self.sp = sorted(set(zip(self.D.sta, self.D.phase)))
        self.spi = {k: i for i, k in enumerate(self.sp)}
        self.nj = int(self.cols.sum())
        self.nb = len(self.sp) * self.nj if beta else 0
        self.ng = self.nb + self.nj
        self.fams = [g.index.values for _, g in self.D.groupby('fam')]

    def global_rows(self, d, s=0.0):
        F = (ycols(d.t.values, s) if self.yearly else SR_.tcols(d.t.values, s))[:, self.cols]
        A = np.zeros((len(d), self.ng))
        if self.beta:
            k = np.array([self.spi[(a, b)] for a, b in zip(d.sta.values, d.phase.values)])
            for j in range(self.nj):
                A[np.arange(len(d)), k * self.nj + j] = F[:, j]
        A[:, self.nb:] = -d.K.values[:, None] * F
        return A

    def local_rows(self, d):
        srcs, si = np.unique(d.src.values, return_inverse=True)
        key = d.chan.values + '|' + d.phase.values + '|' + d.seg.astype(str).values
        ks, ki = np.unique(key, return_inverse=True)
        n = len(d)
        A = np.zeros((n, 4 * len(srcs) + len(ks)))
        A[np.arange(n), 4 * si] = 1.0
        A[np.arange(n), 4 * si + 1] = d.gx.values
        A[np.arange(n), 4 * si + 2] = d.gy.values
        A[np.arange(n), 4 * si + 3] = d.gz.values
        A[np.arange(n), 4 * len(srcs) + ki] = 1.0
        return A

    def solve(self, y, w, s=0.0, sel=None, idx_override=None):
        """y, w full-length arrays; sel boolean mask of rows to use.  Returns x_global, cov_global, residuals."""
        D = self.D
        Sg = np.zeros((self.ng, self.ng))
        rg = np.zeros(self.ng)
        cache = []
        fams = self.fams if idx_override is None else idx_override
        for ii in fams:
            if sel is not None:
                ii = ii[sel[ii]]
            if len(ii) < 8:
                continue
            d = D.iloc[ii]
            Al = self.local_rows(d)
            Ag = self.global_rows(d, s)
            ww = w[ii]
            Nll = (Al * ww[:, None]).T @ Al
            sc = np.median(np.diag(Nll)[np.diag(Nll) > 0])
            Nll += np.eye(len(Nll)) * 1e-6 * sc
            Nlg = (Al * ww[:, None]).T @ Ag
            Ngg = (Ag * ww[:, None]).T @ Ag
            bl = Al.T @ (ww * y[ii])
            bg = Ag.T @ (ww * y[ii])
            Linv_lg = np.linalg.solve(Nll, np.c_[Nlg, bl])
            Sg += Ngg - Nlg.T @ Linv_lg[:, :-1]
            rg += bg - Nlg.T @ Linv_lg[:, -1]
            cache.append((ii, Al, Ag, Nll, Nlg, bl))
        sc = np.median(np.diag(Sg)[np.diag(Sg) > 0])
        Sreg = Sg + np.eye(self.ng) * 1e-7 * sc
        xg = np.linalg.solve(Sreg, rg)
        cov = np.linalg.inv(Sreg)
        res = np.full(len(D), np.nan)
        for ii, Al, Ag, Nll, Nlg, bl in cache:
            xl = np.linalg.solve(Nll, bl - Nlg @ xg)
            res[ii] = y[ii] - Al @ xl - Ag @ xg
        return xg, cov, res

    def fit(self, y=None, s=0.0, sel=None, irls=3, huber=2.0, idx_override=None):
        y = self.D.y.values if y is None else y
        w0 = 1.0 / self.D.se.values ** 2
        w = w0.copy()
        for it in range(irls + 1):
            xg, cov, res = self.solve(y, w, s, sel, idx_override)
            if it < irls:
                z = np.abs(res) * np.sqrt(w0)
                ok = np.isfinite(z)
                z = z / (1.4826 * np.nanmedian(z[ok]) + 1e-20)
                w = w0 * np.where(np.isfinite(z) & (z > huber), huber / np.maximum(np.nan_to_num(z, nan=1.0), 1e-9), 1.0)
        ok = np.isfinite(res)
        chi = float(np.sum((w * res ** 2)[ok]) / max(ok.sum() - self.ng, 1))
        return xg, cov * max(chi, 1.0), res, chi

    def gamma(self, xg):
        if self.yearly:
            return xg[self.nb:]
        th = np.full(NJ, np.nan)
        th[self.cols] = xg[self.nb:]
        return th


def quantities(th, s=0.0):
    t2 = np.nan_to_num(th)
    return {q: f(t2, s) for q, f in SR_.QUANT.items()}


def qvec():
    """linear combination vectors (over the NJ time-function coefficients) for each quantity."""
    out = {}
    for q in SR_.QUANT:
        v = np.zeros(NJ)
        for j in range(NJ):
            e = np.zeros(NJ)
            e[j] = 1.0
            v[j] = SR_.QUANT[q](e) - SR_.QUANT[q](np.zeros(NJ))
        out[q] = v
    return out


def run_variant(J, name, nboot=40, inj=None):
    y = J.D.y.values.copy()
    if inj is not None:
        y = y + inj
    xg, cov, res, chi = J.fit(y)
    th = J.gamma(xg)
    covg = np.zeros((NJ, NJ))
    ci = np.where(J.cols)[0]
    covg[np.ix_(ci, ci)] = cov[J.nb:, J.nb:]
    Q = qvec()
    real = {q: float(Q[q] @ np.nan_to_num(th)) for q in Q}
    formal = {q: float(np.sqrt(Q[q] @ covg @ Q[q])) for q in Q}
    out = dict(variant=name, chi2r=chi, n=len(J.D), nL=int((J.D.kind == 'L').sum()), nE=int((J.D.kind == 'E').sum()))
    boot = {q: [] for q in Q}
    plac = {q: [] for q in Q}
    if nboot:
        # bootstrap over sources within families
        src_by_fam = [(J.D.src.values[ii], ii) for ii in J.fams]
        for b in range(nboot):
            idxs = []
            for srcs, ii in src_by_fam:
                u = np.unique(srcs)
                pick = RNG.choice(u, len(u))
                cnt = pd.Series(pick).value_counts()
                sub = ii[np.isin(srcs, cnt.index)]
                rep = pd.Series(srcs[np.isin(srcs, cnt.index)]).map(cnt).values
                idxs.append(np.repeat(sub, rep))
            # duplicated sources must stay distinct: emulate by weighting (repeat rows; local params shared ->
            # equivalent to weight multiplication)
            xb, _, _, _ = J.fit(y, irls=1, idx_override=idxs)
            tb = J.gamma(xb)
            for q in Q:
                boot[q].append(float(Q[q] @ np.nan_to_num(tb)))
        pre = J.D.t.values < SR_.T_RAMP
        for s in SR_.SHIFTS:
            xp, _, _, _ = J.fit(y, s=s, sel=pre, irls=1)
            tp = J.gamma(xp)
            for q in Q:
                plac[q].append(float(Q[q] @ np.nan_to_num(tp)))
    for q in Q:
        sb = float(np.std(boot[q])) if boot[q] else np.nan
        sp = float(np.sqrt(np.mean(np.square(plac[q])))) if plac[q] else np.nan
        out[f'{q}'] = real[q] * 100
        out[f'{q}_se_formal'] = formal[q] * 100
        out[f'{q}_se_boot'] = sb * 100
        out[f'{q}_se_plac'] = sp * 100
        out[f'{q}_se'] = np.nanmax([formal[q], sb, sp]) * 100
    return out, xg, cov


def stage_fit(a):
    L = load_lfe()
    E = load_eq()
    D = pd.concat([L, E], ignore_index=True)
    D = D[np.isfinite(D.y) & np.isfinite(D.se) & np.isfinite(D.gx) & np.isfinite(D.K)].reset_index(drop=True)
    # stations used by both kinds only matter for beta; keep everything
    both = set(L.sta) & set(E.sta)
    print(f'LFE rows {len(L)}, EQ rows {len(E)}, stations with both: {len(both)}', flush=True)
    results = []
    steps = np.zeros(NJ, bool)
    steps[[0, 1, 2, 3, 4, 8]] = True
    variants = [('joint', dict(beta=True)), ('lfe_beta', dict(beta=True)), ('lfe_nobeta', dict(beta=False)),
                ('joint_steps', dict(beta=True, cols=steps)), ('joint_bothsta', dict(beta=True))]
    variants += [('joint_Kmean', dict(beta=True)), ('lfe_beta_Kmean', dict(beta=True))]
    if a.variants:
        variants = [v for v in variants if v[0] in a.variants]
    if a.quick:
        variants = variants[:1]
    J0 = None
    for name, kw in variants:
        Dv = D
        if name.startswith('lfe'):
            Dv = D[D.kind == 'L']
        if name == 'joint_bothsta':
            Dv = D[D.sta.isin(both)]
        if name.endswith('_Kmean'):
            # remove the inter-family leverage: every LFE family gets the station-mean kernel, so only the
            # LFE-vs-earthquake contrast can identify the layer term
            Dv = Dv.copy()
            isL = Dv.kind == 'L'
            Dv.loc[isL, 'K'] = Dv[isL].groupby(['sta', 'phase']).K.transform('mean')
        J = Joint(Dv, **kw)
        r, xg, cov = run_variant(J, name, nboot=a.nboot)
        results.append(r)
        print(pd.Series(r).to_string(), flush=True)
        if name == 'joint':
            J0 = J
            # beta table (receiver terms per station-phase) for the steps and ramp
            rows = []
            for (sta, ph), i in J.spi.items():
                rr = dict(sta=sta, phase=ph, nL=int(((J.D.sta == sta) & (J.D.phase == ph) & (J.D.kind == 'L')).sum()),
                          nE=int(((J.D.sta == sta) & (J.D.phase == ph) & (J.D.kind == 'E')).sum()))
                for j, nm in enumerate(SR_.PN):
                    rr[nm] = xg[i * J.nj + j] * 1e3
                    rr[nm + '_se'] = np.sqrt(cov[i * J.nj + j, i * J.nj + j]) * 1e3
                rows.append(rr)
            pd.DataFrame(rows).to_csv(f'{OUT}/diff_beta.csv', index=False)
    # injections (joint)
    if J0 is not None and not a.quick and not a.noinj:
        D0 = J0.D
        t = D0.t.values
        post = (t >= T_Q['m7']).astype(float)
        isL = (D0.kind == 'L').values
        isS = (D0.phase == 'S').values
        injs = {
            'injA_layer_-0.3pct_m7': -(-0.003) * D0.K.values * post * isL,       # delay = -eps*K
            'injB_common_receiver': post * np.where(isS, 0.005, 0.002),
            'injC_noncommon_receiver_EQ0.7': post * np.where(isS, 0.005, 0.002) * np.where(isL, 1.0, 0.7),
            'injD_EQ_dz+50m': post * (~isL) * D0.gz.values * 0.05,
        }
        for nm, v in injs.items():
            r, _, _ = run_variant(J0, nm, nboot=0, inj=v)
            results.append(r)
            print(nm, 'm7 =', round(r['m7'], 4), '+-', round(r['m7_se_formal'], 4), 'D =', round(r['D'], 4), flush=True)
    R = pd.DataFrame(results)
    R.to_csv(f'{OUT}/diff_results{a.tag}.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['variant'] + [c for q in ('m7', 'restart', 'fer', 'pet', 'D') for c in (q, f'{q}_se_formal', f'{q}_se_boot', f'{q}_se_plac')]
    print(R[cols].round(4).to_string())

def stage_yearly(a):
    L = load_lfe()
    E = load_eq()
    D = pd.concat([L, E], ignore_index=True)
    D = D[np.isfinite(D.y) & np.isfinite(D.se) & np.isfinite(D.gx) & np.isfinite(D.K)].reset_index(drop=True)
    # yearly series of the layer dv/v (joint and LFE-only-with-beta)
    yrows = []
    for name, Dv in (('joint', D), ('lfe_beta', D[D.kind == 'L'])):
        J = Joint(Dv, beta=True, yearly=True)
        xg, cov, res, chi = J.fit()
        g = J.gamma(xg)
        se = np.sqrt(np.diag(cov)[J.nb:])
        pre = np.array(YEARS) < 2021
        emp = float(np.sqrt(np.mean(np.square(g[pre]))))
        for y, v, e in zip(YEARS, g, se):
            yrows.append(dict(variant=name, year=y, eps_pct=v * 100, se_formal_pct=e * 100, rms_pre2021_pct=emp * 100,
                              chi2r=chi))
        print(name, 'yearly RMS pre-2021 (%):', round(emp * 100, 4), 'median formal se (%):',
              round(float(np.median(se)) * 100, 4), flush=True)
    pd.DataFrame(yrows).to_csv(f'{OUT}/diff_yearly{a.tag}.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--ztop', type=float, default=Z_TOP)
    ap.add_argument('--nboot', type=int, default=40)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--variants', nargs='*', default=None)
    ap.add_argument('--noinj', action='store_true')
    ap.add_argument('--zsrc', default='3d', help='3d (our relocations) | shelly (layer kernel from Shelly depths)')
    a = ap.parse_args()
    global GEOM_TAG
    if a.zsrc == 'shelly' and a.stage != 'geom':
        GEOM_TAG = '_shelly'
    {'geom': stage_geom, 'fit': stage_fit, 'yearly': stage_yearly}[a.stage](a)


if __name__ == '__main__':
    main()
