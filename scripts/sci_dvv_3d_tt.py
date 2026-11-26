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
                    'R10_20W': dt[(zc >= 10) & (zc < 20) & (px < BOXC[0])].sum(),
                    'R1_10W': dt[(zc >= 1) & (zc < 10) & (px < BOXC[0])].sum(),
                    'cos_src1d': float(np.sqrt(max(0.0, 1 - (h[1] - h[0]) ** 2 / max((h[1] - h[0]) ** 2 + (zc[0] - zc[1]) ** 2, 1e-12))))})
        out.append(r)
    return out


def stage_geom(lvzs=('ref', 'soft', 'none')):
    S = sources()
    ST = stations_all()
    D = load_rows()
    combos = pd.concat([D[['fam', 'sta']], load_rows_ds()[['fam', 'sta']]]).drop_duplicates()
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
T_REF = SR_._ep('2016-01-01')
NJX = NJ + 1          # + linear secular term (absorbs slow drifts; not part of any reported quantity)


YEARS = list(range(2010, 2027))
_YE = [SR_._ep(f'{y}-01-01') for y in YEARS + [2027]]


def ycols(t):
    """yearly indicators 2010..2026 (2008-2009 = reference level)."""
    return np.array([((t >= _YE[i]) & (t < _YE[i + 1])).astype(float) for i in range(len(YEARS))]).T


def tcolsx(t, s=0.0):
    return np.c_[SR_.tcols(t, s), (t - T_REF) / SR_.YR]


def load_rows():
    if 'D' not in _ROWS:
        L = ED.load_lfe()
        E = ED.load_eq()
        D = pd.concat([L, E], ignore_index=True)
        D = D[np.isfinite(D.y) & np.isfinite(D.se) & np.isfinite(D.gx)].reset_index(drop=True)
        _ROWS['D'] = D
    return _ROWS['D']


def attach(D, lvz='ref', g3=True):
    """merge 1-D ray geometry (LVZ variant) and the 3-D traced geometry onto the rows (inner join)."""
    G = pd.read_csv(f'{OUT}/tt_geometry.csv')
    g = G[G.lvz == lvz].drop(columns=['lvz'])
    D = D.drop(columns=[c for c in g.columns if c in D.columns and c not in ('fam', 'sta', 'phase')])
    D = D.merge(g, on=['fam', 'sta', 'phase'], how='inner')
    if g3 and os.path.exists(f'{OUT}/tt_geometry3d.csv'):
        G3 = pd.read_csv(f'{OUT}/tt_geometry3d.csv')
        D = D.merge(G3, on=['fam', 'sta', 'phase'], how='left')
        for c in ('lay3', 'box3', 'sph3', 'slab3', 'R1_10_3', 'R10_20_3', 'R1_10_3W', 'R10_20_3W'):
            D[c] = D[c].fillna(0.0)
    return D.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- solver
class Joint:
    """y = local(src: origin, dx, dy, dz; chan/phase/segment offsets) + beta(sta, phase) x f + sum_v (-K_v) gamma_v x f.
    f = tcolsx (time functions + linear secular term).  Local parameters are eliminated per family (Schur)."""

    def __init__(self, D, vols, beta=True, vol_tf='std'):
        self.D = D.reset_index(drop=True)
        self.vols = list(vols)
        self.beta = beta
        self.vol_tf = vol_tf                     # 'std' (time functions) | 'yearly' (one value per calendar year)
        self.nvt = NJX if vol_tf == 'std' else len(YEARS)
        self.sp = sorted(set(zip(self.D.sta, self.D.phase)))
        self.spi = {k: i for i, k in enumerate(self.sp)}
        self.nb = len(self.sp) * NJX if beta else 0
        self.ng = self.nb + len(self.vols) * self.nvt
        self.fams = [g.index.values for _, g in self.D.groupby('fam')]
        self.Kv = self.D[self.vols].values if self.vols else np.zeros((len(self.D), 0))
        self.spk = np.array([self.spi[(a, b)] for a, b in zip(self.D.sta.values, self.D.phase.values)])

    def global_rows(self, ii, s=0.0):
        F = tcolsx(self.D.t.values[ii], s)
        A = np.zeros((len(ii), self.ng))
        if self.beta:
            k = self.spk[ii]
            for j in range(NJX):
                A[np.arange(len(ii)), k * NJX + j] = F[:, j]
        Fv = F if self.vol_tf == 'std' else ycols(self.D.t.values[ii])
        for v in range(len(self.vols)):
            A[:, self.nb + v * self.nvt:self.nb + (v + 1) * self.nvt] = -self.Kv[ii, v][:, None] * Fv
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
        """Y (n,) or (n, k).  Returns xg (ng[,k]), cov (unscaled), residuals (first column), gain (gamma rows x n),
        number of local parameters."""
        Y = np.asarray(Y)
        one = Y.ndim == 1
        Y2 = Y[:, None] if one else Y
        Sg = np.zeros((self.ng, self.ng))
        rg = np.zeros((self.ng, Y2.shape[1]))
        cache = []
        nloc = 0
        fams = self.fams if idx_override is None else idx_override
        for ii in fams:
            if sel is not None:
                ii = ii[sel[ii]]
            if len(ii) < 8:
                continue
            Al = self.local_rows(ii)
            nloc += Al.shape[1]
            Ag = self.global_rows(ii, s)
            ww = w[ii]
            Nll = (Al * ww[:, None]).T @ Al
            dg = np.diag(Nll)
            Nll += np.eye(len(Nll)) * 1e-6 * np.median(dg[dg > 0])
            Nlg = (Al * ww[:, None]).T @ Ag
            Ngg = (Ag * ww[:, None]).T @ Ag
            X = np.linalg.solve(Nll, np.c_[Nlg, (Al * ww[:, None]).T])
            NiNlg, NiAlW = X[:, :self.ng], X[:, self.ng:]
            Sg += Ngg - Nlg.T @ NiNlg
            M = (Ag * ww[:, None]).T - Nlg.T @ NiAlW
            rg += M @ Y2[ii]
            cache.append((ii, Al, Ag, NiNlg, NiAlW, M))
        dg = np.diag(Sg)
        Sreg = Sg + np.eye(self.ng) * 1e-7 * np.median(dg[dg > 0])
        cov = np.linalg.inv(Sreg)
        xg = cov @ rg
        res = np.full(len(self.D), np.nan)
        gain = np.zeros((self.ng - self.nb, len(self.D))) if want_gain else None
        for ii, Al, Ag, NiNlg, NiAlW, M in cache:
            xl = NiAlW @ Y2[ii, 0] - NiNlg @ xg[:, 0]
            res[ii] = Y2[ii, 0] - Al @ xl - Ag @ xg[:, 0]
            if want_gain:
                gain[:, ii] = cov[self.nb:, :] @ M
        self._nloc = nloc
        return (xg[:, 0] if one else xg), cov, res, gain

    def fit(self, y=None, s=0.0, sel=None, irls=3, huber=2.0, idx_override=None, want_gain=False):
        """returns xg, cov scaled by max(1, chi2r) (chi2r with ALL fitted parameters incl. local), res, chi2r, gain, w."""
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
        chi = float(np.sum((w * res ** 2)[ok]) / max(ok.sum() - self.ng - self._nloc, 1))
        return xg, cov * max(chi, 1.0), res, chi, gain, w

    def gam(self, xg):
        return xg[self.nb:].reshape(len(self.vols), self.nvt)


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


def _se(cov, nb, nv, iv, qv, njx=None):
    njx = njx or NJX
    e = np.kron(np.eye(nv)[iv], qv)
    return float(np.sqrt(e @ cov[nb:nb + nv * njx, nb:nb + nv * njx] @ e))


QALL = QUANTS + ['ramp']


def qx(q, s=0.0):
    if q == 'ramp':
        v = np.zeros(NJX)
        v[8] = 1.0
        return v
    return np.r_[qvecs(s)[q], 0.0]


def run(name, D, vols, nboot=100, nplac=True, save=False, beta=True, vol_tf='std'):
    """Honest SE = max(formal, bootstrap over sources, formal x placebo z-RMS), where the placebo z of each
    whole-year shift is its estimate / its own formal SE (shifts whose formal SE > 3 x the real-fit formal SE are
    dropped: their shifted ramp/steps are nearly collinear with the secular term).  Shifts share dates, so
    max(n_plac - 1, 4) dof are used for the t-quantiles."""
    from scipy import stats
    t0 = time.time()
    J = Joint(D, vols, beta=beta, vol_tf=vol_tf)
    xg, cov, res, chi, gain, w_main = J.fit(want_gain=save)
    nv = len(vols)
    out = []
    if vol_tf == 'yearly':
        G = J.gam(xg)
        for iv, v in enumerate(vols):
            ci = J.nb + iv * J.nvt
            for k, yr in enumerate(YEARS):
                out.append(dict(run=name, vol=v, quantity=f'y{yr}', est_pct=G[iv, k] * 100,
                                se_formal=np.sqrt(cov[ci + k, ci + k]) * 100, chi2r=chi, n=len(J.D)))
        R = pd.DataFrame(out)
        os.makedirs(f'{OUT}/tt_parts', exist_ok=True)
        R.to_csv(f'{OUT}/tt_parts/{name}.csv', index=False, float_format='%.6g')
        print(f'{name}: yearly, {len(J.D)} rows, {time.time() - t0:.0f} s', flush=True)
        return R
    G = J.gam(xg)
    est = {(v, q): float(qx(q) @ G[iv]) for iv, v in enumerate(vols) for q in QALL}
    fse = {(v, q): _se(cov, J.nb, nv, iv, qx(q)) for iv, v in enumerate(vols) for q in QALL}
    # correlation of the Petrolia step with the ramp (same volume)
    cpr = {}
    for iv, v in enumerate(vols):
        a, b = np.kron(np.eye(nv)[iv], qx('pet')), np.kron(np.eye(nv)[iv], qx('ramp'))
        Cg = cov[J.nb:, J.nb:]
        cpr[v] = float(a @ Cg @ b / np.sqrt((a @ Cg @ a) * (b @ Cg @ b)))
    boot = {k: [] for k in est}
    plac, pfse = {k: [] for k in est}, {k: [] for k in est}
    rng = np.random.default_rng(11)
    for b in range(nboot):
        xb, *_ = J.fit(irls=1, idx_override=boot_indices(J, rng))
        Gb = J.gam(xb)
        for iv, v in enumerate(vols):
            for q in QALL:
                boot[(v, q)].append(float(qx(q) @ Gb[iv]))
    if nplac and vols:
        pre = J.D.t.values < SR_.T_RAMP
        for s in PSHIFTS:
            xp, cp, *_ = J.fit(s=s, sel=pre, irls=1)
            Gp = J.gam(xp)
            for iv, v in enumerate(vols):
                for q in QALL:
                    plac[(v, q)].append(float(qx(q, s) @ Gp[iv]))
                    pfse[(v, q)].append(_se(cp, J.nb, nv, iv, qx(q, s)))
    ok = np.isfinite(res)
    chi2 = float(np.sum((w_main * res ** 2)[ok]))
    ndof = int(ok.sum() - J.ng - J._nloc)
    if not vols:
        out.append(dict(run=name, vol='none', quantity='-', chi2r=chi, n=len(J.D), chi2=chi2, ndof=ndof))
    for (v, q), e in est.items():
        sb = np.std(boot[(v, q)]) if boot[(v, q)] else np.nan
        pv, pf = np.array(plac[(v, q)]), np.array(pfse[(v, q)])
        keep = pf <= 3 * fse[(v, q)] if len(pv) else np.zeros(0, bool)
        zr = np.sqrt(np.mean((pv[keep] / pf[keep]) ** 2)) if keep.sum() >= 3 else np.nan
        sp_scaled = fse[(v, q)] * zr if np.isfinite(zr) else np.nan
        se = np.nanmax([fse[(v, q)], sb, sp_scaled])
        npl = int(keep.sum())
        dof = max(npl - 1, 4) if npl else 30
        z = e / se
        out.append(dict(run=name, vol=v, quantity=q, est_pct=e * 100, se_formal=fse[(v, q)] * 100,
                        se_boot=sb * 100, plac_zrms=zr, se_plac_scaled=sp_scaled * 100, n_plac=npl,
                        n_plac_dropped=int(len(pv) - npl), se=se * 100, z=z, p_t=2 * stats.t.sf(abs(z), dof),
                        ci95_lo=(e - stats.t.ppf(0.975, dof) * se) * 100, ci95_hi=(e + stats.t.ppf(0.975, dof) * se) * 100,
                        corr_pet_ramp=cpr[v] if q in ('pet', 'ramp') else np.nan, chi2r=chi, n=len(J.D),
                        nL=int((J.D.kind == 'L').sum()), nE=int((J.D.kind == 'E').sum()), nsta=J.D.sta.nunique(),
                        chi2=chi2, ndof=ndof, plac_vals=';'.join(f'{x * 100:+.4f}' for x in pv),
                        plac_formal=';'.join(f'{x * 100:.4f}' for x in pf)))
    if save:
        np.savez_compressed(f'{BIG}/tt_{name}.npz', xg=xg, cov=cov, sp=np.array([f'{a}|{b}' for a, b in J.sp]),
                            vols=np.array(vols), gain=gain.astype(np.float32), w=w_main.astype(np.float32),
                            fam=np.asarray(J.D.fam, dtype='U'), sta=np.asarray(J.D.sta, dtype='U'),
                            phase=np.asarray(J.D.phase, dtype='U'), t=J.D.t.values, kind=np.asarray(J.D.kind, dtype='U'),
                            T=J.D['T'].values, res=res.astype(np.float32), njx=NJX)
    R = pd.DataFrame(out)
    os.makedirs(f'{OUT}/tt_parts', exist_ok=True)
    R.to_csv(f'{OUT}/tt_parts/{name}.csv', index=False, float_format='%.6g')
    print(f'{name}: {len(J.D)} rows, chi2r {chi:.3f}, {time.time() - t0:.0f} s', flush=True)
    return R


def subset(D, sub):
    for part in sub.split('+'):
        if part == 'all':
            continue
        elif part == 'lfe':
            D = D[D.kind == 'L']
        elif part == 'x40':
            D = D[D.X < 40]
        elif part == 'cos02':
            D = D[D.cos_src.fillna(0.0) >= 0.2]
        elif part == 'noexit':
            D = D[~D.exit_deep.fillna(True).astype(bool)]
        elif part in ('S', 'P'):
            D = D[D.phase == part]
        elif part == 'eqboth':
            # earthquake families need events on both sides of Petrolia (2021-12-20)
            tq = SR_.T_Q['pet']
            ok = D[D.kind == 'E'].groupby('fam').t.agg(lambda t: (t < tq).any() and (t > tq).any())
            D = D[(D.kind == 'L') | D.fam.isin(ok[ok].index)]
        elif part.startswith('dropsta:'):
            D = D[D.sta != part.split(':')[1]]
        elif part.startswith('dropfam:'):
            D = D[D.fam != part.split(':')[1]]
    return D


def _run_job(args):
    name, data, sub, vols, nboot, save, lvz = args[:7]
    vol_tf = args[7] if len(args) > 7 else 'std'
    D0 = load_rows_ds() if data == 'ds' else load_rows()
    D = subset(attach(D0, lvz), sub)
    return run(name, D, vols, nboot=nboot, save=save, vol_tf=vol_tf)


PRIMARY = {'lay': ('lay', 'lay3'), 'box': ('box', 'box3'), 'sph': ('sph', 'sph3'), 'slab': ('slab', 'slab3')}
RTERMS = {'1d': ['R1_10', 'R10_20'], '3d': ['R1_10_3', 'R10_20_3']}
CLEAN = 'cos02+noexit'


def stage_fit(jobs_kind='main'):
    jobs = []
    if jobs_kind == 'main':
        for tgt, (v1, v3) in PRIMARY.items():
            for gname, v in (('1d', v1), ('3d', v3)):
                # PRIMARY: clean subset (take-off cos >= 0.2, no lateral exit of the 3-D sub-grid below 20 km)
                jobs.append((f'clean_{tgt}_{gname}', 'tt', CLEAN, [v], 100, True, 'ref'))
                jobs.append((f'clean_R_{tgt}_{gname}', 'tt', CLEAN, RTERMS[gname] + [v], 100, True, 'ref'))
                # secondary (all rows; amplitude calibration uncertain by x2-3)
                jobs.append((f'joint_{tgt}_{gname}', 'tt', 'all', [v], 100, True, 'ref'))
                jobs.append((f'joint_R_{tgt}_{gname}', 'tt', 'all', RTERMS[gname] + [v], 100, True, 'ref'))
                jobs.append((f'clean_lfe_{tgt}_{gname}', 'tt', CLEAN + '+lfe', [v], 50, False, 'ref'))
                jobs.append((f'clean_x40_{tgt}_{gname}', 'tt', CLEAN + '+x40', [v], 50, False, 'ref'))
        for gname, v in (('1d', 'lay'), ('3d', 'lay3')):
            for ph in ('S', 'P'):
                jobs.append((f'clean_lay_{gname}_{ph}only', 'tt', CLEAN + f'+{ph}', [v], 50, False, 'ref'))
            jobs.append((f'clean_lay_{gname}_eqboth', 'tt', CLEAN + '+eqboth', [v], 50, False, 'ref'))
            jobs.append((f'clean_lay_{gname}_yearly', 'tt', CLEAN, [v], 0, False, 'ref', 'yearly'))
            jobs.append((f'joint_lay_{gname}_yearly', 'tt', 'all', [v], 0, False, 'ref', 'yearly'))
        jobs.append(('clean_lay_1d_lvzsoft', 'tt', CLEAN, ['lay'], 0, False, 'soft'))
        jobs.append(('clean_lay_1d_lvznone', 'tt', CLEAN, ['lay'], 0, False, 'none'))
        jobs.append(('clean_R_only_1d', 'tt', CLEAN, RTERMS['1d'], 100, True, 'ref'))
        jobs.append(('clean_shallowonly', 'tt', CLEAN, [], 0, True, 'ref'))
        # honest stage-1 equivalent of the earlier direct-S deep bound (sci_dvv_deep_shallow stacks of 100 LFEs)
        jobs.append(('ds_deep1', 'ds', 'all', ['deep1'], 100, False, 'ref'))
        jobs.append(('ds_lay3_clean', 'ds', CLEAN, ['lay3'], 100, False, 'ref'))
        jobs.append(('ds_lay_clean', 'ds', CLEAN, ['lay'], 100, False, 'ref'))
    elif jobs_kind == 'jack':
        D = subset(attach(load_rows(), 'ref'), CLEAN)
        for v in ('lay', 'lay3', 'sph3'):
            for s in sorted(D.sta.unique()):
                jobs.append((f'jk_{v}_sta_{s}', 'tt', f'{CLEAN}+dropsta:{s}', [v], 0, False, 'ref'))
            for f in sorted(D.fam.unique()):
                jobs.append((f'jk_{v}_fam_{f}', 'tt', f'{CLEAN}+dropfam:{f}', [v], 0, False, 'ref'))
    with Pool(NPROC) as pool:
        res = pool.map(_run_job, jobs, chunksize=1)
    R = pd.concat(res, ignore_index=True)
    fn = f'{OUT}/tt_results{"" if jobs_kind == "main" else "_" + jobs_kind}.csv'
    R.to_csv(fn, index=False, float_format='%.6g')
    with pd.option_context('display.width', 250):
        print(R[R.quantity.isin(['m7', 'pet', 'D'])][['run', 'vol', 'quantity', 'est_pct', 'se_formal', 'se_boot', 'se_plac_scaled', 'se', 'z', 'p_t', 'chi2r']].round(4).to_string())


# --------------------------------------------------------------------------------------------- 3-D ray geometry
# Rays traced through the 3-D travel-time fields of sci_dvv_source_region (data/velocity/tt_3d/{sta}.npz: S and P
# travel time from every node of the stored source-region sub-grid to the station; 0.5 km nodes, TM projection,
# x/y +-20 km around the sources, z 12-42 km).  From each source the ray follows -grad T (steepest descent of the
# station's travel-time field = the reciprocal ray) in 0.25-km steps until it leaves the sub-grid; the time spent in
# a volume is the sum of the T decrements of the steps inside it (K <= T by construction).  The rest of the path,
# from the exit point to the receiver (top exit at 12 km, or lateral exit), is the 1-D ray of the receiver column
# (Furlong + LVZ).  All volume times of the '3d' geometry (lay3, box3, sph3, slab3, R1_10_3, R10_20_3) and the
# 3-D grid rows used for the averaging kernels come from this combined path.  Rays that leave the sub-grid
# laterally while still below 20 km are flagged (exit_deep): their deep segment beyond the sub-grid is 1-D.
# cos_src = |dT/dz| / |grad T| at the source.  The full-width travel-time grid is not stored (only the sub-grid),
# so extending it would need the eikonal solver to be rerun - not done.
def _path_volumes(mx, my, z, dt, plane, famxyz, boxc):
    inl = (z >= 20) & (z < 31)
    inbox = (np.abs(mx - boxc[0]) <= BOX) & (np.abs(my - boxc[1]) <= BOX)
    d2 = np.min((mx[:, None] - famxyz[None, :, 0]) ** 2 + (my[:, None] - famxyz[None, :, 1]) ** 2
                + (z[:, None] - famxyz[None, :, 2]) ** 2, axis=1)
    zpl = plane[0] + plane[1] * mx + plane[2] * my
    west = mx < boxc[0]
    return {'lay3': dt[inl].sum(), 'box3': dt[inl & inbox].sum(), 'sph3': dt[d2 <= 100.0].sum(),
            'slab3': dt[(np.abs(z - zpl) <= 2.5) & inbox].sum(), 'R1_10_3': dt[(z >= 1) & (z < 10)].sum(),
            'R10_20_3': dt[(z >= 10) & (z < 20)].sum(), 'R10_20_3W': dt[(z >= 10) & (z < 20) & west].sum(),
            'R1_10_3W': dt[(z >= 1) & (z < 10) & west].sum(), 'Tpath3': dt.sum()}


def _geom3d_job(args):
    sta, fams, src_tm, plane, famxyz, boxc, st = args
    from scipy.interpolate import RegularGridInterpolator as RGI
    from pyproj import Transformer
    g = json.load(open(f'{ROOT}/data/velocity/tt_3d/grid.json'))
    ax = (np.array(g['sub_x']), np.array(g['sub_y']), np.array(g['sub_z']))
    inv = Transformer.from_crs(SR_.TM, 'EPSG:4326', always_xy=True)
    f = f'{ROOT}/data/velocity/tt_3d/{sta}.npz'
    if not os.path.exists(f):
        return [], {}
    Z = np.load(f)
    vm = KK.station_vmodel(st['lat'], st['lon'], 'ref')
    xr, yr = GG.xy(st['lat'], st['lon'])
    out, rows = [], {}
    lo = np.array([a[0] for a in ax])
    hi = np.array([a[-1] for a in ax])
    for ph in ('S', 'P'):
        T = Z['t' + ph.lower()].astype(float)
        I = RGI(ax, T, bounds_error=False, fill_value=np.nan)
        grads = np.gradient(T, 0.5, 0.5, 0.5)
        Ig = [RGI(ax, gg, bounds_error=False, fill_value=np.nan) for gg in grads]
        P = src_tm.copy()
        T0 = I(P)
        g0 = np.column_stack([gi(P) for gi in Ig])
        cos_src = np.abs(g0[:, 2]) / np.linalg.norm(g0, axis=1)
        alive = np.isfinite(T0)
        Tp = T0.copy()
        pexit = P.copy()
        path = [[] for _ in range(len(P))]
        for it in range(800):
            if not alive.any():
                break
            gv = np.column_stack([gi(P) for gi in Ig])
            nrm = np.linalg.norm(gv, axis=1)
            Pn = P - 0.25 * gv / np.where(nrm > 0, nrm, 1)[:, None]
            Tn = I(Pn)
            ok = alive & np.all((Pn >= lo) & (Pn <= hi), axis=1) & np.isfinite(Tn) & (nrm > 0)
            mid = 0.5 * (P + Pn)
            lon, lat = inv.transform(mid[:, 0], mid[:, 1])
            mx, my = GG.xy(lat, lon)
            for i in np.where(ok)[0]:
                path[i].append((mx[i], my[i], mid[i, 2], Tp[i] - Tn[i]))
            alive = ok
            P = np.where(ok[:, None], Pn, P)
            Tp = np.where(ok, Tn, Tp)
        elon, elat = inv.transform(P[:, 0], P[:, 1])
        ex, ey = GG.xy(elat, elon)
        for i, fam in enumerate(fams):
            if not np.isfinite(T0[i]):
                continue
            a = np.array(path[i]) if path[i] else np.zeros((0, 4))
            zex = float(P[i, 2])
            # 1-D continuation from the exit point to the receiver
            X = float(np.hypot(xr - ex[i], yr - ey[i]))
            zc, h, dt, v = ray_phase(vm, zex, st['z0'], X, ph, n=400)
            ux, uy = ((xr - ex[i]) / X, (yr - ey[i]) / X) if X > 0 else (1.0, 0.0)
            cx, cy = ex[i] + ux * h, ey[i] + uy * h
            mxa = np.r_[a[:, 0], cx]
            mya = np.r_[a[:, 1], cy]
            za = np.r_[a[:, 2], zc]
            dta = np.r_[a[:, 3], dt]
            vols = _path_volumes(mxa, mya, za, dta, plane, famxyz, boxc)
            exit_deep = bool(zex >= 20.0)
            out.append(dict(fam=fam, sta=sta, phase=ph, T3=T0[i], cos_src=cos_src[i], zexit=zex, exit_deep=exit_deep,
                            T3_part=float(a[:, 3].sum()) if len(a) else 0.0, Tcont=float(dt.sum()), **vols))
            rows[(fam, sta, ph)] = GG.points_to_row(mxa, mya, za, dta).astype(np.float32)   # seconds per cell
    return out, rows


def stage_geom3d():
    from pyproj import Transformer
    S = sources()
    tr = Transformer.from_crs('EPSG:4326', SR_.TM, always_xy=True)
    tx, ty = tr.transform(S.lon.values, S.lat.values)
    S['tx'], S['ty'] = tx, ty
    plane = slab_plane(S)
    famxyz = S[S.index.str.startswith('L')][['x', 'y', 'z']].values
    ST = stations_all()
    D = load_rows()
    combos = pd.concat([D[['fam', 'sta']], load_rows_ds()[['fam', 'sta']]]).drop_duplicates()
    jobs = []
    for sta, g in combos.groupby('sta'):
        if sta not in ST.index:
            continue
        fams = [f for f in g.fam if f in S.index]
        st = dict(lat=float(ST.loc[sta, 'lat']), lon=float(ST.loc[sta, 'lon']), z0=float(ST.loc[sta, 'z0']))
        jobs.append((sta, fams, S.loc[fams, ['tx', 'ty', 'z']].values.astype(float), plane, famxyz, BOXC, st))
    with Pool(NPROC) as pool:
        res = pool.map(_geom3d_job, jobs, chunksize=1)
    G = pd.DataFrame([x for r, _ in res for x in r])
    G.to_csv(f'{OUT}/tt_geometry3d.csv', index=False, float_format='%.5g')
    keys, arr = [], []
    for _r, rw in res:
        for k, v in rw.items():
            keys.append('|'.join(k))
            arr.append(v)
    np.savez_compressed(f'{BIG}/grid_rows_3d.npz', keys=np.array(keys), rows=np.array(arr, np.float32))
    G1 = pd.read_csv(f'{OUT}/tt_geometry.csv')
    G1 = G1[G1.lvz == 'ref']
    M = G.merge(G1[['fam', 'sta', 'phase', 'T', 'lay', 'box', 'sph', 'slab', 'R1_10', 'R10_20']], on=['fam', 'sta', 'phase'])
    M['kind'] = M.fam.str[0]
    with pd.option_context('display.width', 220):
        print(M.groupby(['kind', 'phase'])[['T3', 'Tpath3', 'T', 'lay3', 'lay', 'box3', 'box', 'sph3', 'sph', 'slab3', 'slab',
                                            'R10_20_3', 'R10_20', 'R1_10_3', 'R1_10', 'cos_src']].median().round(3))
        print('exit_deep fraction', M.groupby('kind').exit_deep.mean().round(3).to_dict(), ' cos_src<0.2',
              M.groupby('kind').cos_src.apply(lambda c: (c < 0.2).mean()).round(3).to_dict())
        print('Tpath3/T3 median', float((M.Tpath3 / M.T3).median()))


# --------------------------------------------------------------------------------------------- direct-S stacks
def load_rows_ds():
    """deep_shallow direct-S/P stacks (100 LFEs; sci_dvv_deep_shallow.load / obs_rows, EXCL + BREAKS2 applied),
    in the same row format; gradients from the source-region 3-D geometry."""
    if 'DS' in _ROWS:
        return _ROWS['DS']
    import sci_dvv_deep_shallow as DS
    G3 = pd.read_csv(f'{ROOT}/data/sci/dvv_source_region/geometry_3d.csv')
    out = []
    for F in range(1, 28):
        if not os.path.exists(f'{DS.MEAS}/fam{F}.parquet'):
            continue
        d, meta, loc = DS.load(F)
        for ph in ('S', 'P'):
            r = DS.obs_rows(d, ph)
            if not len(r):
                continue
            r['sta'] = r.chan.map(DS.sta_of)
            g = G3[G3.F == F].set_index('sta')
            r = r[r.sta.isin(g.index)].copy()
            for c in 'xyz':
                r[f'g{c}'] = g.loc[r.sta, f'g{ph}{c}'].values
            r['phase'] = ph
            r['fam'] = f'L{F}'
            r['src'] = f'D{F}_' + r['stack'].astype(str)
            r['kind'] = 'L'
            r['K'] = 0.0
            out.append(r[['fam', 'src', 'kind', 't', 'chan', 'sta', 'phase', 'seg', 'y', 'se', 'gx', 'gy', 'gz', 'K']])
    R = pd.concat(out, ignore_index=True)
    R = R[np.isfinite(R.y) & np.isfinite(R.se) & np.isfinite(R.gx)].reset_index(drop=True)
    _ROWS['DS'] = R
    return R


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'geom':
        stage_geom()
    elif st == 'geom3d':
        stage_geom3d()
    elif st == 'fit':
        stage_fit('main')
    elif st == 'jack':
        stage_fit('jack')
