#!/usr/bin/env python3
"""Model grid and forward (sensitivity) matrices G for the joint 3-D relative dv/v inversion.

Grid: local flat coordinates (km) about the LFE centroid 40.05 N, -123.56 W; box x -100..80, y -90..90 km.
Layers (km): 0-0.1, 0.1-0.3, 0.3-1, 1-3, 3-6, 6-10 (5-km cells, 36 x 36), 10-15, 15-20, 20-25, 25-31, 31-35,
35-250 (absorbing; 10-km cells, 18 x 18); plus one EXTERIOR parameter per layer (kernel mass outside the box).
Model m = dv/v (%).  Every row satisfies d = G m; for stretching data each row sums to 1 (uniform change recovered).

Datasets (rows):
  SC  single-station coda (sci_dvv_3d_kernels.Kernels.station radial kernels mapped through ring/cell overlaps)
  LP  inter-station long-period coda: 2-D separated-station diffusion kernel (Pacheco & Snieder 2005, D = U l*/2,
      numerical time integral, lapse window of the LP measurement, w ~ t^0.5) x LP depth kernel
      (data/sci/noise_lp/kernels.csv, 0.8 Rayleigh + 0.2 Love, matching ZZ/RR/ZR/RZ + TT averaging)
  L1  LFE coda per channel = mean over the 25 balanced-panel families of the source-receiver kernel
  L2  family deviation = panel-channel mean of K_F,C minus its mean over families (gauge sum dS_F = 0)
  L6  repeater coda, deep / shallow cluster sets (cluster centroids within 60 km of the station)
      Source-receiver body kernel: half-space diffusion with free-surface images for source and receiver,
      exact time integral (a+b)/(4 pi D a b) G3(a+b, t), normalised by the direct intensity, D = c l*/3,
      c = 3 km/s, l* = 5 km (2-8 Hz; variant 20 km), lapse t = T_S + tau, tau 3-20 s (w ~ tau^0.5), truncated at
      a + b <= c t.  Depth profile re-weighted to sci_dvv_depth2_kernels.lfe_body_kernel (Monte-Carlo radiative
      transfer, kernel_mc.csv, slowness-weighted in the station's 1-D model incl. LVZ) for LFEs, and by local S
      slowness for repeaters.  Plus a near-receiver surface-wave share a_L (default 0.25; buried sensors reduced by
      the surface-wave amplitude ratio) with the SC lateral/depth kernel of the 1-5 Hz bands.
  L3  direct-S delay (ms) per station: 1-D ray (station Furlong + LVZ model) from each family to the sensor,
      Fresnel volume (Gaussian, sigma = 0.5 sqrt(lambda L1 L2 / L), 3 Hz); d = -10 T sum(frac m), averaged over
      families; one free origin offset per quantity (nuisance column).
  L4  near-source S-P eps: time-weighted average over ray segments within 10 km of the sources.
  L5  LFE-minus-earthquake layer: time-weighted average over ray segments between 22 km and the family depth.
Output: /data/wsd02/maleen_data/dvv_3d/G_{dataset}_{variant}.npz (+ row tables in data/sci/dvv_3d/rows_*.csv).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sci_dvv_3d_kernels as KK  # noqa: E402
import sci_dvv_depth2_kernels as K2  # noqa: E402

ROOT = KK.ROOT
OUT = KK.OUT
BIG = KK.BIG
NPROC = int(os.environ.get('NPROC', '2'))
LAT0, LON0 = 40.05, -123.56
KMX = 111.195 * np.cos(np.radians(LAT0))
KMY = 111.195
X0, X1, Y0, Y1 = -100.0, 80.0, -90.0, 90.0
LAY = np.array([0, 0.1, 0.3, 1, 3, 6, 10, 15, 20, 25, 31, 35, 250.0])
NL = len(LAY) - 1
CELL = np.array([5.0] * 6 + [10.0] * 6)
LAYNAME = [f'{LAY[i]:g}-{LAY[i + 1]:g}' for i in range(NL)]


def xy(lat, lon):
    return (np.asarray(lon) - LON0) * KMX, (np.asarray(lat) - LAT0) * KMY


class Grid:
    def __init__(self):
        self.nx = {cs: int(round((X1 - X0) / cs)) for cs in (5.0, 10.0)}
        self.ny = {cs: int(round((Y1 - Y0) / cs)) for cs in (5.0, 10.0)}
        self.ncell = {cs: self.nx[cs] * self.ny[cs] for cs in (5.0, 10.0)}
        self.off = np.zeros(NL + 1, int)
        for l in range(NL):
            self.off[l + 1] = self.off[l] + self.ncell[CELL[l]]
        self.next0 = self.off[-1]
        self.npar = self.next0 + NL                     # + exterior per layer
        self.zl = np.searchsorted(LAY, 0.5 * (KK.Z_E[1:] + KK.Z_E[:-1])) - 1   # fine-z cell -> layer

    def centers(self, cs):
        xc = X0 + cs * (np.arange(self.nx[cs]) + 0.5)
        yc = Y0 + cs * (np.arange(self.ny[cs]) + 0.5)
        X, Y = np.meshgrid(xc, yc, indexing='ij')
        return X.ravel(), Y.ravel()

    def cell_of(self, cs, x, y):
        ix = np.floor((np.asarray(x) - X0) / cs).astype(int)
        iy = np.floor((np.asarray(y) - Y0) / cs).astype(int)
        ok = (ix >= 0) & (ix < self.nx[cs]) & (iy >= 0) & (iy < self.ny[cs])
        return np.where(ok, ix * self.ny[cs] + iy, -1)

    def param_info(self):
        rows = []
        for l in range(NL):
            X, Y = self.centers(CELL[l])
            for i in range(len(X)):
                rows.append((l, X[i], Y[i], CELL[l]))
        for l in range(NL):
            rows.append((l, np.nan, np.nan, np.nan))
        return pd.DataFrame(rows, columns=['layer', 'x', 'y', 'cell'])


GR = Grid()


# --------------------------------------------------------------------------------------- radial kernels -> grid
_PIX = {}


def ring_overlap(xs, ys):
    """fraction of each RHO_E ring falling in each cell (per lateral grid) and outside the box."""
    if 'x' not in _PIX:
        h = 0.25
        px = np.arange(X0 + h / 2, X1, h)
        py = np.arange(Y0 + h / 2, Y1, h)
        PX, PY = np.meshgrid(px, py, indexing='ij')
        _PIX['x'], _PIX['y'], _PIX['a'] = PX.ravel(), PY.ravel(), h * h
        _PIX['c'] = {cs: GR.cell_of(cs, _PIX['x'], _PIX['y']) for cs in (5.0, 10.0)}
    r = np.hypot(_PIX['x'] - xs, _PIX['y'] - ys)
    nr = len(KK.RHO_E) - 1
    rb = np.searchsorted(KK.RHO_E, r) - 1
    ok = (rb >= 0) & (rb < nr)
    ring_area = np.pi * (KK.RHO_E[1:] ** 2 - KK.RHO_E[:-1] ** 2)
    dist_edge = min(xs - X0, X1 - xs, ys - Y0, Y1 - ys)
    inside = KK.RHO_E[1:] <= dist_edge
    out = {}
    for cs in (5.0, 10.0):
        nc = GR.ncell[cs]
        A = np.bincount(GR.cell_of(cs, _PIX['x'][ok], _PIX['y'][ok]) * nr + rb[ok], weights=np.full(ok.sum(), _PIX['a']),
                        minlength=nc * nr).reshape(nc, nr)
        s = A.sum(0)
        F = A / ring_area[None, :]
        # rings entirely inside: renormalise sampling; tiny rings with no pixel -> station cell
        F[:, inside & (s > 0)] = A[:, inside & (s > 0)] / s[inside & (s > 0)][None, :]
        miss = inside & (s == 0)
        if miss.any():
            F[GR.cell_of(cs, xs, ys), miss] = 1.0
        ext = np.clip(1 - F.sum(0), 0, 1)
        out[cs] = (F, ext)
    return out


def radial_to_row(Krz, ov):
    """Krz (n_rho, n_zfine) masses -> G row (npar)."""
    g = np.zeros(GR.npar)
    for l in range(NL):
        kl = Krz[:, GR.zl == l].sum(1)
        F, ext = ov[CELL[l]]
        g[GR.off[l]:GR.off[l + 1]] = F @ kl
        g[GR.next0 + l] = ext @ kl
    return g


# --------------------------------------------------------------------------------------- SC rows
SC_VARIANTS = {'default': {}, 'l5': dict(lstar='5'), 'l20': dict(lstar='20'), 'l50': dict(lstar='50'),
               'surf': dict(alpha=1.0), 'body': dict(alpha=0.0), 'half': dict(alpha=0.5),
               'lvz_soft': dict(lvz='soft'), 'lvz_none': dict(lvz='none')}


def build_sc():
    ST = pd.read_csv(f'{OUT}/stations.csv')
    Kn = KK.Kernels()
    rows, G = [], {v: [] for v in SC_VARIANTS}
    for r in ST.itertuples():
        xs, ys = xy(r.lat, r.lon)
        ov = ring_overlap(float(xs), float(ys))
        for b in KK.BANDS:
            for v, o in SC_VARIANTS.items():
                Krz, info = Kn.station(r.ns, r.z0_km, b, **o)
                G[v].append(radial_to_row(Krz, ov))
                if v == 'default':
                    rows.append(dict(sta=r.ns, band=b, **info))
    for v in SC_VARIANTS:
        np.savez_compressed(f'{BIG}/G_SC_{v}.npz', G=np.array(G[v], np.float32))
    pd.DataFrame(rows).to_csv(f'{OUT}/rows_SC.csv', index=False, float_format='%.4g')
    print('SC rows', len(rows), flush=True)


# --------------------------------------------------------------------------------------- lateral sample points
def lattice(cs, sub):
    X, Y = GR.centers(cs)
    o = (np.arange(sub) + 0.5) / sub * cs - cs / 2
    ox, oy = np.meshgrid(o, o, indexing='ij')
    px = (X[:, None] + ox.ravel()[None, :]).ravel()
    py = (Y[:, None] + oy.ravel()[None, :]).ravel()
    return px, py, np.full(px.size, (cs / sub) ** 2), np.repeat(np.arange(len(X)), sub * sub)


def refined_points(cs, sub, centres, fine, nb=1):
    """lattice with the (2nb+1)^2 cells around each centre re-sampled at `fine` points per cell side."""
    px, py, pa, pc = lattice(cs, sub)
    cells = set()
    for (x, y) in centres:
        c = GR.cell_of(cs, x, y)
        if c < 0:
            continue
        ix, iy = divmod(int(c), GR.ny[cs])
        for dx in range(-nb, nb + 1):
            for dy in range(-nb, nb + 1):
                if 0 <= ix + dx < GR.nx[cs] and 0 <= iy + dy < GR.ny[cs]:
                    cells.add((ix + dx) * GR.ny[cs] + iy + dy)
    keep = ~np.isin(pc, list(cells))
    X, Y = GR.centers(cs)
    o = (np.arange(fine) + 0.5) / fine * cs - cs / 2
    ox, oy = np.meshgrid(o, o, indexing='ij')
    add_x, add_y, add_c = [], [], []
    for c in sorted(cells):
        add_x.append(X[c] + ox.ravel()); add_y.append(Y[c] + oy.ravel()); add_c.append(np.full(fine * fine, c))  # noqa: E702
    if cells:
        px = np.r_[px[keep], np.concatenate(add_x)]
        py = np.r_[py[keep], np.concatenate(add_y)]
        pa = np.r_[pa[keep], np.full(len(cells) * fine * fine, (cs / fine) ** 2)]
        pc = np.r_[pc[keep], np.concatenate(add_c)]
    return px, py, pa, pc


ZC = 0.5 * (KK.Z_E[1:] + KK.Z_E[:-1])
DZ = np.diff(KK.Z_E)


# --------------------------------------------------------------------------------------- source-receiver body kernel
def sr_body(src, rec, T_S, ls=5.0, c=KK.C_BODY, taus=np.linspace(3, 20, 8), zweight=None):
    """3-D source-receiver coda kernel (masses per fine-z cell x lateral cell, per lateral grid) -> G row.
    src, rec: (x, y, z) km.  zweight: (n_zfine,) multiplicative depth re-weighting (or None)."""
    D = c * ls / 3.0
    w = taus ** 0.5
    w = w / w.sum()
    xs, ys, zs = src
    xr, yr, zr = rec
    rsr = np.sqrt((xs - xr) ** 2 + (ys - yr) ** 2 + (zs - zr) ** 2)
    rsr2 = np.sqrt((xs - xr) ** 2 + (ys - yr) ** 2 + (zs + zr) ** 2)
    g = np.zeros(GR.npar)
    tot = 0.0
    per_layer_mass = np.zeros(NL)
    for cs, zmask in ((5.0, ZC < 10), (10.0, ZC >= 10)):
        if cs == 5.0:
            px, py, pa, pc = refined_points(cs, 5, [(xr, yr)], 25)
        else:
            px, py, pa, pc = refined_points(cs, 5, [(xs, ys)], 10)
        z = ZC[zmask]
        dz = DZ[zmask]
        zl = GR.zl[zmask]
        dxs2 = (px - xs) ** 2 + (py - ys) ** 2
        dxr2 = (px - xr) ** 2 + (py - yr) ** 2
        a1 = np.sqrt(dxs2[:, None] + (z[None, :] - zs) ** 2)
        a2 = np.sqrt(dxs2[:, None] + (z[None, :] + zs) ** 2)
        b1 = np.maximum(np.sqrt(dxr2[:, None] + (z[None, :] - zr) ** 2), 0.05)
        b2 = np.maximum(np.sqrt(dxr2[:, None] + (z[None, :] + zr) ** 2), 0.05)
        a1 = np.maximum(a1, 0.05)
        acc = np.zeros(a1.shape)
        for tau, wt in zip(taus, w):
            t = T_S + tau
            f4 = 4 * D * t
            P0 = np.exp(-rsr ** 2 / f4) + np.exp(-(rsr2 ** 2 - rsr ** 2) / f4) * np.exp(-rsr ** 2 / f4)
            k = np.zeros(a1.shape)
            for a in (a1, a2):
                for b in (b1, b2):
                    k += (a + b) / (4 * np.pi * D * a * b) * np.exp(-((a + b) ** 2 - rsr ** 2) / f4)
            k /= (P0 * np.exp(rsr ** 2 / f4))
            k *= (a1 + b1) <= c * t
            acc += wt * k / t                       # fraction of lapse time per unit volume
        if zweight is not None:
            acc *= zweight[zmask][None, :]
        M = acc * pa[:, None] * dz[None, :]
        for l in np.unique(zl):
            ml = M[:, zl == l].sum(1)
            gl = np.bincount(pc, weights=ml, minlength=GR.ncell[cs])
            g[GR.off[l]:GR.off[l + 1]] += gl
            per_layer_mass[l] += gl.sum()
        tot += M.sum()
    return g, tot


def finish_sr(g, tot_unweighted_inbox, zw_applied):
    """exterior mass: 1 - in-box mass (diffusion kernel integrates to 1 over the half-space);
    if a depth re-weighting was applied, rescale the in-box part to the unweighted in-box total."""
    inbox = g[:GR.next0].sum()
    if zw_applied and inbox > 0:
        g[:GR.next0] *= tot_unweighted_inbox / inbox
        inbox = tot_unweighted_inbox
    ext = max(0.0, 1.0 - inbox)
    # put exterior mass in proportion to the in-box layer distribution
    lm = np.array([g[GR.off[l]:GR.off[l + 1]].sum() for l in range(NL)])
    if lm.sum() > 0:
        g[GR.next0:] = ext * lm / lm.sum()
    return g


# --------------------------------------------------------------------------------------- rays (direct S)
def ray_1d(vm, zs, z0, X, n=600):
    """upgoing S ray in a flat 1-D model from depth zs to z0 at horizontal distance X.
    Returns z (centres), h (horizontal distance from source), dt (s per segment), T."""
    ze = np.linspace(z0, zs, n + 1)
    zc = 0.5 * (ze[1:] + ze[:-1])
    dz = np.diff(ze)
    _, v, _ = vm(zc)
    pmax = 1 / v.max() * (1 - 1e-9)

    def Xp(p):
        q = np.sqrt(np.clip(1 - (p * v) ** 2, 1e-12, None))
        return np.sum(p * v * dz / q)
    lo, hi = 0.0, pmax
    if Xp(hi) < X:
        p = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if Xp(mid) < X:
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
    q = np.sqrt(np.clip(1 - (p * v) ** 2, 1e-12, None))
    dx = p * v * dz / q
    dt = dz / (v * q)
    # from source upward: reverse order
    zc, dx, dt, dl = zc[::-1], dx[::-1], dt[::-1], (dz / q)[::-1]
    h = np.cumsum(dx) - 0.5 * dx
    hscale = X / max(dx.sum(), 1e-9)                 # head-wave cases: stretch horizontally
    return zc, h * hscale, dt, dl, float(dt.sum()), v[::-1]


def ray_points(src, rec, vm, f=3.0, fres=1.0):
    xs, ys, zs = src
    xr, yr, zr = rec
    X = float(np.hypot(xr - xs, yr - ys))
    zc, h, dt, dl, T, v = ray_1d(vm, zs, zr, X)
    ux, uy = ((xr - xs) / X, (yr - ys) / X) if X > 0 else (1.0, 0.0)
    L1 = np.cumsum(dl) - 0.5 * dl
    L = dl.sum()
    lam = v / f
    sig = 0.5 * fres * np.sqrt(np.clip(lam * L1 * (L - L1) / L, 0, None))
    offs = np.array([-2, -1, 0, 1, 2.0])
    wo = np.exp(-offs ** 2 / 2)
    wo /= wo.sum()
    px = xs + ux * h[:, None] + (-uy) * sig[:, None] * offs[None, :]
    py = ys + uy * h[:, None] + ux * sig[:, None] * offs[None, :]
    wt = (dt / T)[:, None] * wo[None, :]
    zz = np.repeat(zc[:, None], len(offs), 1)
    return px.ravel(), py.ravel(), zz.ravel(), wt.ravel(), T, np.repeat(np.hypot(zc - zs, h)[:, None] * 0 + np.sqrt((h) ** 2 + (zc - zs) ** 2)[:, None], len(offs), 1).ravel()


def points_to_row(px, py, pz, wt):
    g = np.zeros(GR.npar)
    lz = np.clip(np.searchsorted(LAY, pz) - 1, 0, NL - 1)
    for l in np.unique(lz):
        m = lz == l
        c = GR.cell_of(CELL[l], px[m], py[m])
        ins = c >= 0
        g[GR.off[l]:GR.off[l + 1]] += np.bincount(c[ins], weights=wt[m][ins], minlength=GR.ncell[CELL[l]])
        g[GR.next0 + l] += wt[m][~ins].sum()
    return g


# --------------------------------------------------------------------------------------- LFE / repeater jobs
_VM = {}


def vmodel(sta, lat, lon, lvz='ref'):
    k = (sta, lvz)
    if k not in _VM:
        _VM[k] = KK.station_vmodel(lat, lon, lvz)
    return _VM[k]


def _sr_job(args):
    """one (source, station) pair: body kernel rows for l* variants, ray rows for Fresnel/LVZ variants."""
    os.environ['OMP_NUM_THREADS'] = '1'
    kind, sid, src, sta, lat, lon, z0, do_ray = args
    xr, yr = xy(lat, lon)
    rec = (float(xr), float(yr), z0)
    vm = vmodel(sta, lat, lon)
    X = float(np.hypot(rec[0] - src[0], rec[1] - src[1]))
    zc, h, dt, dl, T_S, _ = ray_1d(vm, src[2], z0, X)
    # depth re-weighting
    if kind == 'lfe':
        dists = np.array([5, 20, 30, 40, 50, 60])
        dist = int(dists[np.argmin(np.abs(dists - X))])
        tgt = K2.lfe_body_kernel(vm, KK.Z_E, (3.0, 20.0), l_km='20', dist=dist)
    else:
        _, vs, _ = vm(ZC)
        tgt = None
        sw = (1 / vs) / np.mean(1 / vs)
    out = {}
    for ls in (5.0, 20.0):
        g0, tot = sr_body(src, rec, T_S, ls=ls)
        if kind == 'lfe':
            # marginal of the unweighted kernel per fine-z cell
            marg = np.zeros(len(ZC))
            # approximate marginal from layer masses spread uniformly by dz within layers
            lm = np.array([g0[GR.off[l]:GR.off[l + 1]].sum() for l in range(NL)])
            for l in range(NL):
                m = GR.zl == l
                marg[m] = lm[l] * DZ[m] / DZ[m].sum()
            tl = np.array([tgt[GR.zl == l].sum() for l in range(NL)])
            ml = lm / max(lm.sum(), 1e-12)
            tl = tl / max(tl.sum(), 1e-12)
            ratio = np.where(ml > 1e-9, tl / np.maximum(ml, 1e-12), 1.0)
            g = g0.copy()
            for l in range(NL):
                g[GR.off[l]:GR.off[l + 1]] *= ratio[l]
        else:
            g = g0.copy()
            lsw = np.array([(sw[GR.zl == l] * DZ[GR.zl == l]).sum() / DZ[GR.zl == l].sum() for l in range(NL)])
            for l in range(NL):
                g[GR.off[l]:GR.off[l + 1]] *= lsw[l]
        out[f'body_l{ls:g}'] = finish_sr(g, g0[:GR.next0].sum(), True).astype(np.float32)
    if do_ray:
        for lvz in ('ref', 'soft', 'none'):
            vmv = vmodel(sta, lat, lon, lvz)
            for fr in ((1.0, 0.5, 2.0) if lvz == 'ref' else (1.0,)):
                px, py, pz, wt, T, dsrc = ray_points(src, rec, vmv, fres=fr)
                out[f'ray_{lvz}_f{fr:g}'] = points_to_row(px, py, pz, wt).astype(np.float32)
                out[f'T_{lvz}_f{fr:g}'] = T
                if lvz == 'ref' and fr == 1.0:
                    # L4 sphere (within 10 km of the source) and L5 layer (22 km .. source depth)
                    m4 = dsrc <= 10.0
                    out['sph10'] = points_to_row(px[m4], py[m4], pz[m4], wt[m4] / max(wt[m4].sum(), 1e-12)).astype(np.float32)
                    m5 = pz >= 22.0
                    out['lay22'] = points_to_row(px[m5], py[m5], pz[m5], wt[m5] / max(wt[m5].sum(), 1e-12)).astype(np.float32)
    out['T_S'] = T_S
    out['X'] = X
    return (kind, sid, sta), out


def build_sr():
    ST = pd.read_csv(f'{OUT}/stations.csv').set_index('ns')
    O = pd.read_csv(f'{OUT}/obs_lfe.csv')
    fam = pd.read_csv(f'{ROOT}/data/families/shelly_family_locations.csv').set_index('idnum')
    FT = pd.read_csv(f'{ROOT}/data/sci/dvv_source_receiver/family_table_balanced.csv')
    fams = [int(f) for f in FT.family]
    sta_lfe = sorted(set(O[O.dataset.isin(['L1', 'L3'])].sta.dropna()) | {'NC.KMPB', 'NC.KMR', 'PB.B045', 'PB.B046', 'PB.B047',
                                                                         'PB.B049', 'PB.B932', 'PB.B934'})
    sta_lfe = [s for s in sta_lfe if s in ST.index]
    jobs = []
    for f in fams:
        fx, fy = xy(fam.loc[f, 'lat'], fam.loc[f, 'lon'])
        src = (float(fx), float(fy), float(fam.loc[f, 'depth_km']))
        for s in sta_lfe:
            r = ST.loc[s]
            jobs.append(('lfe', f, src, s, r.lat, r.lon, r.z0_km, True))
    # repeater cluster centroids per class
    C = pd.read_csv(f'{ROOT}/data/sci/dvv_deep_shallow/rep/clusters.csv')
    C = C[C.cl95 >= 0]
    cen = C.groupby('cl95').agg(lat=('Latitude', 'mean'), lon=('Longitude', 'mean'), depth=('depth', 'mean'),
                                n=('EventID', 'size'))
    cen['cls'] = np.where(cen.depth >= 14, 'deep', 'shallow')
    rep = O[O.dataset == 'L6']
    for s in sorted(rep.sta.unique()):
        r = ST.loc[s]
        sx, sy = xy(r.lat, r.lon)
        for cls in ('deep', 'shallow'):
            cc = cen[cen.cls == cls].copy()
            cx, cy = xy(cc.lat.values, cc.lon.values)
            cc['dist'] = np.hypot(cx - sx, cy - sy)
            cc = cc[cc.dist < 60].sort_values('n', ascending=False).head(6)
            for k, q in cc.iterrows():
                qx, qy = xy(q.lat, q.lon)
                jobs.append(('rep', f'{cls}|{k}', (float(qx), float(qy), float(q.depth)), s, r.lat, r.lon, r.z0_km, False))
    print('sr jobs', len(jobs), flush=True)
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(_sr_job, jobs, chunksize=2)
    print('sr done', round(time.time() - t0), 's', flush=True)
    flat = {}
    meta = []
    for (kind, sid, sta), o in res:
        for k, v in o.items():
            if isinstance(v, np.ndarray):
                flat[f'{kind}|{sid}|{sta}|{k}'] = v
        meta.append(dict(kind=kind, sid=sid, sta=sta, T_S=o['T_S'], X=o['X'], T_ray=o.get('T_ref_f1')))
    np.savez_compressed(f'{BIG}/sr_kernels.npz', **flat)
    pd.DataFrame(meta).to_csv(f'{OUT}/sr_meta.csv', index=False, float_format='%.4g')


# --------------------------------------------------------------------------------------- LP rows
def lp_lateral(xa, ya, xb, yb, U, ls, t1, t2, nt=8, ns=48):
    """2-D separated-station coda kernel on both lateral grids -> {cs: (ncell,) masses}, plus exterior mass."""
    D = U * ls / 2.0
    ts = np.linspace(t1, t2, nt)
    w = ts ** 0.5
    w /= w.sum()
    xg, wg = np.polynomial.legendre.leggauss(ns)
    rab = np.hypot(xa - xb, ya - yb)
    out = {}
    inbox = 0.0
    for cs, sub in ((5.0, 3), (10.0, 3)):
        px, py, pa, pc = lattice(cs, sub)
        ra2 = (px - xa) ** 2 + (py - ya) ** 2
        rb2 = (px - xb) ** 2 + (py - yb) ** 2
        acc = np.zeros(px.size)
        for t, wt in zip(ts, w):
            th = 0.5 * np.pi * (xg + 1)                                   # s = t (1 - cos th) / 2
            s = t * (1 - np.cos(th)) / 2
            ds = t * np.sin(th) / 2 * 0.5 * np.pi * wg
            k = np.zeros(px.size)
            for si, dsi in zip(s, ds):
                ex = -ra2 / (4 * D * si) - rb2 / (4 * D * (t - si)) + rab ** 2 / (4 * D * t)
                k += dsi * np.exp(np.minimum(ex, 50)) / (4 * np.pi * D * si) / (4 * np.pi * D * (t - si)) * (4 * np.pi * D * t)
            k *= (np.sqrt(ra2) + np.sqrt(rb2)) <= U * t
            acc += wt * k / t
        m = np.bincount(pc, weights=acc * pa, minlength=GR.ncell[cs])
        out[cs] = m
        if cs == 5.0:
            inbox = m.sum()
    return out, inbox


def _lp_job(args):
    os.environ['OMP_NUM_THREADS'] = '1'
    key, xa, ya, xb, yb, U, ls, t1, t2 = args
    m, inbox = lp_lateral(xa, ya, xb, yb, U, ls, t1, t2)
    return key, m[5.0].astype(np.float32), m[10.0].astype(np.float32), inbox


def build_lp():
    L = pd.read_csv(f'{OUT}/obs_lp.csv')
    S = pd.read_csv(f'{ROOT}/data/sci/noise_lp/sites.csv').set_index('site')
    Kz = pd.read_csv(f'{ROOT}/data/sci/noise_lp/kernels.csv')
    Kz = Kz[Kz.variant == 'furlong']
    disp = pd.read_csv(f'{ROOT}/data/sci/noise_lp/dispersion.csv')
    disp = disp[disp.variant == 'furlong']
    win = {'0.05-0.1': (30.0, 150.0), '0.1-0.2': (22.02, 112.02), '0.2-0.3': (22.02, 82.02)}
    fr = {'0.05-0.1': (0.05, 0.1), '0.1-0.2': (0.1, 0.2), '0.2-0.3': (0.2, 0.3)}
    lay_frac = {}
    for b in win:
        U = disp[(disp.f >= fr[b][0] - 1e-9) & (disp.f <= fr[b][1] + 1e-9)].U.mean()
        kk = []
        for wave, wgt in (('rayleigh', 0.8), ('love', 0.2)):
            q = Kz[(Kz.band == b) & (Kz.wave == wave)]
            lf = np.zeros(NL)
            for r in q.itertuples():
                for l in range(NL):
                    ov = max(0.0, min(r.z_bot, LAY[l + 1]) - max(r.z_top, LAY[l]))
                    lf[l] += r.K * ov / (r.z_bot - r.z_top)
            kk.append(wgt * lf)
        lay_frac[b] = (np.sum(kk, 0), U)
    pairs = L[['pair', 'a', 'b', 'band']].drop_duplicates()
    jobs = []
    for r in pairs.itertuples():
        xa, ya = xy(S.loc[r.a, 'lat'], S.loc[r.a, 'lon'])
        xb, yb = xy(S.loc[r.b, 'lat'], S.loc[r.b, 'lon'])
        lz, U = lay_frac[r.band]
        for lv, ls in (('default', float(np.clip(20.0 / np.sqrt(fr[r.band][0] * fr[r.band][1]), 5, 150))), ('l50', 50.0)):
            jobs.append(((r.pair, r.band, lv), float(xa), float(ya), float(xb), float(yb), float(U), ls, *win[r.band]))
    print('lp jobs', len(jobs), flush=True)
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(_lp_job, jobs, chunksize=8)
    print('lp done', round(time.time() - t0), 's', flush=True)
    rows = {}
    for (pair, band, lv), m5, m10, inbox in res:
        lz, _ = lay_frac[band]
        g = np.zeros(GR.npar, np.float32)
        ext = max(0.0, 1.0 - inbox)
        for l in range(NL):
            mm = m5 if CELL[l] == 5.0 else m10
            mm = mm / max(mm.sum(), 1e-12) * (1 - ext)
            g[GR.off[l]:GR.off[l + 1]] = lz[l] * mm
            g[GR.next0 + l] = lz[l] * ext
        rows[f'{pair}|{band}|{lv}'] = g
    np.savez_compressed(f'{BIG}/lp_rows.npz', **rows)
    json.dump({b: dict(layer_frac=lay_frac[b][0].round(4).tolist(), U=lay_frac[b][1]) for b in lay_frac},
              open(f'{OUT}/lp_depth_kernels.json', 'w'), indent=1)


# --------------------------------------------------------------------------------------- LFE surface share rows
def build_lfe_surface():
    """near-receiver surface-wave kernel for LFE / repeater coda (2-8 Hz, tau 3-20 s) per station (radial)."""
    ST = pd.read_csv(f'{OUT}/stations.csv')
    Kn = KK.Kernels()
    U = np.mean([json.loads(str(Kn.lat['Uband']))[b] for b in ('1-3Hz', '3-5Hz')])
    taus = np.linspace(3, 20, 8)
    w = taus ** 0.5
    w /= w.sum()
    L = sum(wt * KK.klat2(U, 5.0, t) for t, wt in zip(taus, w))
    rows = {}
    for r in ST.itertuples():
        Kz = 0.5 * (Kn.sw[f'{r.ns}|ref|1-3Hz|Kz'] + Kn.sw[f'{r.ns}|ref|3-5Hz|Kz'])
        rr = 0.5 * (float(Kn.sw[f'{r.ns}|ref|1-3Hz|r']) + float(Kn.sw[f'{r.ns}|ref|3-5Hz|r'])) if r.z0_km > 0 else 1.0
        xs, ys = xy(r.lat, r.lon)
        ov = ring_overlap(float(xs), float(ys))
        rows[r.ns] = radial_to_row(L[:, None] * Kz[None, :], ov).astype(np.float32)
        rows[r.ns + '|r'] = np.array(rr)
    np.savez_compressed(f'{BIG}/lfe_surface_rows.npz', **rows)


if __name__ == '__main__':
    st = sys.argv[1:] or ['sc', 'surf', 'sr', 'lp']
    pd.DataFrame(GR.param_info()).to_csv(f'{OUT}/grid_params.csv', index=False, float_format='%.4g')
    for s in st:
        t0 = time.time()
        {'sc': build_sc, 'surf': build_lfe_surface, 'sr': build_sr, 'lp': build_lp}[s]()
        print(s, 'done', round(time.time() - t0), 's', flush=True)
