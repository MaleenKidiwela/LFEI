#!/usr/bin/env python3
"""Joint 3-D dv/v, travel-time block: raw LFE epoch-stack and earthquake S and P delays, ingested once.

Data (as sci_eq_above_diff.load_lfe / load_eq, imported unchanged): LFE epoch-stack P and S delays per family x
channel (data/sci/dvv_source_region/meas; EXCL dropped, BREAKS2 segments) and per-event P and S delays of the
catalog earthquake families above the LFEs (data/sci/eq_above/meas_cat.parquet, relocation outliers dropped).

Model (extends the validated sci_eq_above_diff.Joint design):
  y = tau_src + g . dx_src + off_{fam,chan,phase,seg} + sum_j beta_{sta,phase,j} f_j(t)
      - sum_v K_v(src, sta, phase) sum_j gamma_{v,j} f_j(t)
  f_j: sci_dvv_source_region.tcols (annual, Petrolia/Ferndale/M7 steps, 60-d windows after each, ramp 2021-06).
  tau (origin, common to S and P) and dx/dy/dz per source (LFE stack or earthquake) and the channel offsets are
  eliminated per family (Schur complement).  beta = free receiver terms per station and phase (absorb every
  receiver-side change: shallow velocity, clock, site).  gamma_v = dv/v of volume v (flat prior):
    R1_10  1-10 km (everywhere)         R10_20  10-20 km (everywhere)
    lay    20-31 km, uniform layer       box     20-31 km inside the 20-km box around 40.05N -123.56W
    sph    union of 10-km spheres around the LFE families
    slab   5-km thick slab through the families (plane fitted to the family hypocentres), 20-km box
  K_v = S (or P) travel time (s) spent in v along a 1-D ray in the receiver's Furlong(2024)+LVZ column
  (scripts/sci_dvv_3d_kernels.station_vmodel); shallow path times T_0_0.1, T_0.1_1, T_1_3 per row are stored for
  the second-stage (GLS) mapping of beta onto per-station shallow terms.
Errors: formal x max(1, chi2r) (Huber IRLS); bootstrap over sources within families; season-matched placebo =
  whole fit on pre-2021-06 rows with all model dates shifted back by whole years 5..12 (8 values).
  Honest SE = max(formal, bootstrap, placebo RMS).
Linear gains: the final IRLS weights define a linear estimator gamma = H y; H is stored (per gamma) so that
  averaging kernels on the 3-D grid (scripts/sci_dvv_3d_G.py) and injections through the REAL estimator
  (origin/location removal included) are exact.
Outputs: data/sci/dvv_3d/tt_results.csv, tt_geometry.csv; /data/wsd02/maleen_data/dvv_3d/tt_{run}.npz.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_eq_above_diff as ED  # noqa: E402
import sci_dvv_source_region as SR_  # noqa: E402
import sci_dvv_3d_kernels as KK  # noqa: E402
import sci_dvv_3d_G as GG  # noqa: E402

OUT = f'{ROOT}/data/sci/dvv_3d'
BIG = '/data/wsd02/maleen_data/dvv_3d'
NPROC = int(os.environ.get('NPROC', '2'))
NJ = len(SR_.PN)
VOLS = ['R1_10', 'R10_20', 'lay', 'box', 'sph', 'slab']
SHALLOW = ['T0_0.1', 'T0.1_1', 'T1_3']
PSHIFTS = [5, 6, 7, 8, 9, 10, 11, 12]
QUANTS = ['m7', 'pet', 'fer', 'D']
BOX = 10.0          # half-width km; centred on the LFE family centroid (BOXC, set in sources())
BOXC = [-5.98, 1.30]


def qvecs(s=0.0):
    out = {}
    for q in QUANTS:
        v = np.zeros(NJ)
        for j in range(NJ):
            e = np.zeros(NJ)
            e[j] = 1.0
            v[j] = SR_.QUANT[q](e, s) - SR_.QUANT[q](np.zeros(NJ), s)
        out[q] = v
    return out


# --------------------------------------------------------------------------------------------- geometry
def stations_all():
    c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv')
    c['ns'] = c.net + '.' + c.sta
    c = c[c.start < '2025-09-01']
    g = c.groupby('ns').agg(lat=('lat', 'median'), lon=('lon', 'median'), depth_m=('depth', 'median')).reset_index()
    g['z0'] = np.where(g.ns.str.startswith('PB.'), g.depth_m / 1000.0, 0.0)
    return g.set_index('ns')


def sources():
    rows = []
    for F in range(1, 28):
        f = f'{ROOT}/data/locations/fam{F}_3d.json'
        if os.path.exists(f):
            d = json.load(open(f))
            rows.append(dict(fam=f'L{F}', lat=d['lat'], lon=d['lon'], z=d['depth_km']))
    E = pd.read_csv(f'{ROOT}/data/sci/eq_above/families.csv')
    for k, g in E.groupby('efam'):
        rows.append(dict(fam=f'E{k}', lat=g.Latitude.mean(), lon=g.Longitude.mean(), z=g.depth.mean()))
    S = pd.DataFrame(rows).set_index('fam')
    S['x'], S['y'] = GG.xy(S.lat.values, S.lon.values)
    return S


def slab_plane(S):
    L = S[S.index.str.startswith('L')]
    A = np.c_[np.ones(len(L)), L.x, L.y]
    c, *_ = np.linalg.lstsq(A, L.z.values, rcond=None)
    return c


def ray_phase(vm, zs, z0, X, phase, n=800):
    ze = np.linspace(z0, zs, n + 1)
    zc = 0.5 * (ze[1:] + ze[:-1])
    dz = np.diff(ze)
    vp, vs, _ = vm(zc)
    v = vs if phase == 'S' else vp
    pmax = 1 / v.max() * (1 - 1e-9)

    def Xp(p):
        return np.sum(p * v * dz / np.sqrt(np.clip(1 - (p * v) ** 2, 1e-12, None)))
    lo, hi = 0.0, pmax
    if Xp(hi) < X:
        p = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if Xp(mid) < X else (lo, mid)
        p = 0.5 * (lo + hi)
    q = np.sqrt(np.clip(1 - (p * v) ** 2, 1e-12, None))
    dx = p * v * dz / q
    dt = dz / (v * q)
    hx = np.cumsum(dx[::-1]) - 0.5 * dx[::-1]          # horizontal distance from the source
    sc = X / max(dx.sum(), 1e-9)
    return zc[::-1], hx * sc, dt[::-1], v[::-1]


def _geom_job(args):
    fam, src, sta, st, plane, famxyz, phases = args
    vm = KK.station_vmodel(st['lat'], st['lon'], st['lvz'])
    xr, yr = GG.xy(st['lat'], st['lon'])
    X = float(np.hypot(xr - src['x'], yr - src['y']))
    out = []
    for ph in phases:
        zc, h, dt, v = ray_phase(vm, src['z'], st['z0'], X, ph)
        ux, uy = ((xr - src['x']) / X, (yr - src['y']) / X) if X > 0 else (1.0, 0.0)
        px, py = src['x'] + ux * h, src['y'] + uy * h
        inbox = (np.abs(px - BOXC[0]) <= BOX) & (np.abs(py - BOXC[1]) <= BOX)
        d2 = np.min((px[:, None] - famxyz[None, :, 0]) ** 2 + (py[:, None] - famxyz[None, :, 1]) ** 2
                    + (zc[:, None] - famxyz[None, :, 2]) ** 2, axis=1)
        zpl = plane[0] + plane[1] * px + plane[2] * py
        r = dict(fam=fam, sta=sta, phase=ph, T=dt.sum(), X=X, lvz=st['lvz'],
                 **{'T0_0.1': dt[zc < 0.1].sum(), 'T0.1_1': dt[(zc >= 0.1) & (zc < 1)].sum(),
                    'T1_3': dt[(zc >= 1) & (zc < 3)].sum(),
                    'R1_10': dt[(zc >= 1) & (zc < 10)].sum(), 'R10_20': dt[(zc >= 10) & (zc < 20)].sum(),
                    'lay': dt[(zc >= 20) & (zc < 31)].sum(), 'box': dt[(zc >= 20) & (zc < 31) & inbox].sum(),
                    'sph': dt[d2 <= 100.0].sum(), 'slab': dt[(np.abs(zc - zpl) <= 2.5) & inbox].sum(),
                    'deep1': dt[zc >= 1].sum(),
                    'cos_src1d': float(np.sqrt(max(0.0, 1 - (h[1] - h[0]) ** 2 / max((h[1] - h[0]) ** 2 + (zc[0] - zc[1]) ** 2, 1e-12))))})
        out.append(r)
    return out


def stage_geom(lvzs=('ref', 'soft', 'none')):
    S = sources()
    ST = stations_all()
    D = load_rows()
    combos = D[['fam', 'sta']].drop_duplicates()
    plane = slab_plane(S)
    famxyz = S[S.index.str.startswith('L')][['x', 'y', 'z']].values
    jobs = []
    for r in combos.itertuples():
        if r.sta not in ST.index:
            continue
        for lvz in lvzs:
            st = dict(lat=ST.loc[r.sta, 'lat'], lon=ST.loc[r.sta, 'lon'], z0=ST.loc[r.sta, 'z0'], lvz=lvz)
            jobs.append((r.fam, S.loc[r.fam].to_dict(), r.sta, st, plane, famxyz, ('S', 'P')))
    with Pool(NPROC) as pool:
        res = pool.map(_geom_job, jobs, chunksize=16)
    G = pd.DataFrame([x for r in res for x in r])
    G.to_csv(f'{OUT}/tt_geometry.csv', index=False, float_format='%.5g')
    json.dump(dict(plane=list(plane)), open(f'{OUT}/tt_slab_plane.json', 'w'))
    print(G[G.lvz == 'ref'].groupby('phase')[['T', 'T0_0.1', 'T0.1_1', 'T1_3', 'R1_10', 'R10_20', 'lay', 'box', 'sph', 'slab']].median().round(3))


# --------------------------------------------------------------------------------------------- data
_ROWS = {}


def load_rows():
    if 'D' not in _ROWS:
        L = ED.load_lfe()
        E = ED.load_eq()
        D = pd.concat([L, E], ignore_index=True)
        D = D[np.isfinite(D.y) & np.isfinite(D.se) & np.isfinite(D.gx)].reset_index(drop=True)
        _ROWS['D'] = D
    return _ROWS['D']


def attach(D, G, lvz='ref'):
    g = G[G.lvz == lvz].drop(columns=['lvz'])
    D = D.drop(columns=[c for c in g.columns if c in D.columns and c not in ('fam', 'sta', 'phase')])
    D = D.merge(g, on=['fam', 'sta', 'phase'], how='inner')
    return D.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- solver
class Joint:
    """y = local(src, offsets) + beta(sta, phase) x f + sum_v (-K_v) gamma_v x f; local eliminated per family."""

    def __init__(self, D, vols, beta=True, sel_sta=None):
        self.D = D.reset_index(drop=True)
        self.vols = list(vols)
        self.beta = beta
        self.sp = sorted(set(zip(self.D.sta, self.D.phase)))
        self.spi = {k: i for i, k in enumerate(self.sp)}
        self.nb = len(self.sp) * NJ if beta else 0
        self.ng = self.nb + len(self.vols) * NJ
        self.fams = [g.index.values for _, g in self.D.groupby('fam')]
        self.Kv = self.D[self.vols].values if self.vols else np.zeros((len(self.D), 0))

    def global_rows(self, ii, s=0.0):
        d = self.D.iloc[ii]
        F = SR_.tcols(d.t.values, s)
        A = np.zeros((len(ii), self.ng))
        if self.beta:
            k = np.array([self.spi[(a, b)] for a, b in zip(d.sta.values, d.phase.values)])
            for j in range(NJ):
                A[np.arange(len(ii)), k * NJ + j] = F[:, j]
        for v in range(len(self.vols)):
            A[:, self.nb + v * NJ:self.nb + (v + 1) * NJ] = -self.Kv[ii, v][:, None] * F
        return A

    def local_rows(self, ii):
        d = self.D.iloc[ii]
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

    def solve(self, Y, w, s=0.0, sel=None, idx_override=None, want_gain=False):
        """Y (n,) or (n, k).  Returns xg (ng[,k]), cov, residuals (of first column), gain (n_vol*NJ x n) if asked."""
        Y = np.asarray(Y)
        one = Y.ndim == 1
        Y2 = Y[:, None] if one else Y
        Sg = np.zeros((self.ng, self.ng))
        rg = np.zeros((self.ng, Y2.shape[1]))
        cache = []
        fams = self.fams if idx_override is None else idx_override
        for ii in fams:
            if sel is not None:
                ii = ii[sel[ii]]
            if len(ii) < 8:
                continue
            Al = self.local_rows(ii)
            Ag = self.global_rows(ii, s)
            ww = w[ii]
            Nll = (Al * ww[:, None]).T @ Al
            dg = np.diag(Nll)
            Nll += np.eye(len(Nll)) * 1e-6 * np.median(dg[dg > 0])
            Nlg = (Al * ww[:, None]).T @ Ag
            Ngg = (Ag * ww[:, None]).T @ Ag
            X = np.linalg.solve(Nll, np.c_[Nlg, (Al * ww[:, None]).T])     # [Nll^-1 Nlg, Nll^-1 Al^T W]
            NiNlg, NiAlW = X[:, :self.ng], X[:, self.ng:]
            Sg += Ngg - Nlg.T @ NiNlg
            M = (Ag * ww[:, None]).T - Nlg.T @ NiAlW                       # (ng, n_F): rg = M y
            rg += M @ Y2[ii]
            cache.append((ii, Al, Ag, NiNlg, NiAlW, M))
        dg = np.diag(Sg)
        Sreg = Sg + np.eye(self.ng) * 1e-7 * np.median(dg[dg > 0])
        cov = np.linalg.inv(Sreg)
        xg = cov @ rg
        res = np.full(len(self.D), np.nan)
        gain = None
        if want_gain:
            gain = np.zeros((self.ng - self.nb, len(self.D)))
        for ii, Al, Ag, NiNlg, NiAlW, M in cache:
            xl = NiAlW @ Y2[ii, 0] - NiNlg @ xg[:, 0]
            res[ii] = Y2[ii, 0] - Al @ xl - Ag @ xg[:, 0]
            if want_gain:
                gain[:, ii] = cov[self.nb:, :] @ M
        return (xg[:, 0] if one else xg), cov, res, gain

    def fit(self, y=None, s=0.0, sel=None, irls=3, huber=2.0, idx_override=None, want_gain=False):
        y = self.D.y.values if y is None else y
        w0 = 1.0 / self.D.se.values ** 2
        w = w0.copy()
        for it in range(irls + 1):
            xg, cov, res, gain = self.solve(y, w, s, sel, idx_override, want_gain=(want_gain and it == irls))
            if it < irls:
                z = np.abs(res) * np.sqrt(w0)
                ok = np.isfinite(z)
                z = z / (1.4826 * np.nanmedian(z[ok]) + 1e-20)
                w = w0 * np.where(np.isfinite(z) & (z > huber), huber / np.maximum(np.nan_to_num(z, nan=1.0), 1e-9), 1.0)
        ok = np.isfinite(res)
        chi = float(np.sum((w * res ** 2)[ok]) / max(ok.sum() - self.ng, 1))
        self.w_final = w
        return xg, cov * max(chi, 1.0), res, chi, gain

    def gam(self, xg):
        return xg[self.nb:].reshape(len(self.vols), NJ)


def boot_indices(J, rng):
    idxs = []
    for ii in J.fams:
        srcs = J.D.src.values[ii]
        u = np.unique(srcs)
        cnt = pd.Series(rng.choice(u, len(u))).value_counts()
        keep = np.isin(srcs, cnt.index)
        rep = pd.Series(srcs[keep]).map(cnt).values
        idxs.append(np.repeat(ii[keep], rep))
    return idxs


def run(name, D, vols, nboot=20, nplac=True, save=False, beta=True):
    t0 = time.time()
    J = Joint(D, vols, beta=beta)
    xg, cov, res, chi, gain = J.fit(want_gain=save)
    Q = qvecs()
    G = J.gam(xg)
    nv = len(vols)
    covg = cov[J.nb:, J.nb:]
    out = []
    est = {(v, q): float(Q[q] @ G[iv]) for iv, v in enumerate(vols) for q in QUANTS}
    fse = {(v, q): float(np.sqrt(np.kron(np.eye(nv)[iv], Q[q]) @ covg @ np.kron(np.eye(nv)[iv], Q[q]))) for iv, v in enumerate(vols) for q in QUANTS}
    boot = {k: [] for k in est}
    plac = {k: [] for k in est}
    rng = np.random.default_rng(11)
    for b in range(nboot):
        xb, *_ = J.fit(irls=1, idx_override=boot_indices(J, rng))
        Gb = J.gam(xb)
        for iv, v in enumerate(vols):
            for q in QUANTS:
                boot[(v, q)].append(float(Q[q] @ Gb[iv]))
    if nplac:
        pre = J.D.t.values < SR_.T_RAMP
        for s in PSHIFTS:
            Qs = qvecs(s)
            xp, *_ = J.fit(s=s, sel=pre, irls=1)
            Gp = J.gam(xp)
            for iv, v in enumerate(vols):
                for q in QUANTS:
                    plac[(v, q)].append(float(Qs[q] @ Gp[iv]))
    if not vols:
        out.append(dict(run=name, vol='none', quantity='-', chi2r=chi, n=len(J.D), chi2=float(np.nansum(J.w_final * res ** 2)),
                        ndof=int(np.isfinite(res).sum() - J.ng)))
    for (v, q), e in est.items():
        sb = np.std(boot[(v, q)]) if boot[(v, q)] else np.nan
        sp = np.sqrt(np.mean(np.square(plac[(v, q)]))) if plac[(v, q)] else np.nan
        out.append(dict(run=name, vol=v, quantity=q, est_pct=e * 100, se_formal=fse[(v, q)] * 100,
                        se_boot=sb * 100, se_plac=sp * 100, n_plac=len(plac[(v, q)]),
                        se=np.nanmax([fse[(v, q)], sb, sp]) * 100, chi2r=chi, n=len(J.D),
                        nL=int((J.D.kind == 'L').sum()), nE=int((J.D.kind == 'E').sum()),
                        chi2=float(np.nansum(J.w_final * res ** 2)), ndof=int(np.isfinite(res).sum() - J.ng),
                        plac_vals=';'.join(f'{x * 100:+.4f}' for x in plac[(v, q)])))
    if save:
        # honest/formal variance ratio per (vol, quantity) for the second stage
        np.savez_compressed(f'{BIG}/tt_{name}.npz', xg=xg, cov=cov, sp=np.array([f'{a}|{b}' for a, b in J.sp]),
                            vols=np.array(vols), gain=gain.astype(np.float32), w=J.w_final.astype(np.float32),
                            fam=J.D.fam.values, sta=J.D.sta.values, phase=J.D.phase.values, t=J.D.t.values,
                            T=J.D['T'].values, res=res.astype(np.float32))
    print(f'{name}: {len(J.D)} rows, chi2r {chi:.2f}, {time.time() - t0:.0f} s', flush=True)
    return pd.DataFrame(out)


def _run_job(args):
    name, sub, vols, nboot, save, lvz = args
    D = attach(load_rows(), pd.read_csv(f'{OUT}/tt_geometry.csv'), lvz)
    if sub == 'lfe':
        D = D[D.kind == 'L']
    elif sub.startswith('dropsta:'):
        D = D[D.sta != sub.split(':')[1]]
    elif sub.startswith('dropfam:'):
        D = D[D.fam != sub.split(':')[1]]
    return run(name, D, vols, nboot=nboot, save=save)


def stage_fit(jobs_kind='main'):
    jobs = []
    if jobs_kind == 'main':
        for tgt in ['lay', 'box', 'sph', 'slab']:
            jobs.append((f'joint_{tgt}', 'all', [tgt], 20, True, 'ref'))
            jobs.append((f'joint_R_{tgt}', 'all', ['R1_10', 'R10_20', tgt], 20, True, 'ref'))
            jobs.append((f'lfe_{tgt}', 'lfe', [tgt], 10, False, 'ref'))
        jobs.append(('joint_lay_lvzsoft', 'all', ['lay'], 0, False, 'soft'))
        jobs.append(('joint_lay_lvznone', 'all', ['lay'], 0, False, 'none'))
        jobs.append(('joint_R_only', 'all', ['R1_10', 'R10_20'], 10, False, 'ref'))
        jobs.append(('joint_shallowonly', 'all', [], 0, False, 'ref'))
        # eq_above geometry: straight rays along the 3-D take-off direction, layer 22 km..family depth (K column)
        jobs.append(('joint_layED', 'all', ['K'], 20, False, 'ref'))
        jobs.append(('lfe_layED', 'lfe', ['K'], 10, False, 'ref'))
    elif jobs_kind == 'jack':
        D = load_rows()
        for s in sorted(D.sta.unique()):
            jobs.append((f'jk_sta_{s}', f'dropsta:{s}', ['lay', 'sph'], 0, False, 'ref'))
        for f in sorted(D.fam.unique()):
            jobs.append((f'jk_fam_{f}', f'dropfam:{f}', ['lay', 'sph'], 0, False, 'ref'))
    with Pool(NPROC) as pool:
        res = pool.map(_run_job, jobs, chunksize=1)
    R = pd.concat(res, ignore_index=True)
    fn = f'{OUT}/tt_results{"" if jobs_kind == "main" else "_" + jobs_kind}.csv'
    R.to_csv(fn, index=False, float_format='%.5g')
    print(R[R.quantity.isin(['m7', 'D'])][['run', 'vol', 'quantity', 'est_pct', 'se_formal', 'se_boot', 'se_plac', 'se', 'chi2r']].round(4).to_string())


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'geom':
        stage_geom()
    elif st == 'fit':
        stage_fit('main')
    elif st == 'jack':
        stage_fit('jack')
    elif st == 'time':
        D = attach(load_rows(), pd.read_csv(f'{OUT}/tt_geometry.csv'))
        print(run('timing', D, ['lay'], nboot=0, nplac=False).round(4).to_string())
