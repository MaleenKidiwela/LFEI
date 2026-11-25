#!/usr/bin/env python3
"""3-D sensitivity kernels for single-station cross-component (SC) coda dv/v (helper of scripts/sci_dvv_3d.py).

Per station s and band b:
  K_s,b(rho, z) = sum_t w(t) [ a_eff(t) * Klat2(rho, t) * Kz_sw,s(z)  +  (1 - a_eff(t)) * Kbody(rho, z, t | z0_s) ]
over the band's lapse window (w(t) ~ t^0.5: least-squares stretching weight t^2 I(t) with I ~ t^-1.5, as in
scripts/sci_dvv_depth2_kernels.noise_body_kernel), normalised so that a uniform change is recovered exactly.

* Kz_sw,s(z): fundamental-mode coda-stretching depth kernel, mean of Rayleigh and Love, averaged over 3 frequencies
  in the band; thin-layer method of scripts/sci_dvv_depth2_kernels.py (imported, not modified; its module constant
  ZMAX_GRID is raised to 150 km at run time so 0.1-Hz modes are not truncated).  STATION-SPECIFIC 1-D model: the
  Furlong et al. (2024) 3-D model column nearest the station (data/velocity/furlong2024/{pm,sm}.llzv.gz; depth below
  the first valid node, down to 60 km) plus the shallow low-velocity zone of the chosen LVZ variant (VModel).
* Klat2(rho, t): lateral kernel of a coincident source/receiver in 2-D (surface waves, energy speed = group velocity
  U of the band): single-scattering part (time spent uniformly along the straight paths to the scattering circle
  rho = U t / 2, density ~ 1/rho) with weight exp(-U t / l*) (= exact single-scattering share of the 2-D return
  intensity for isotropic scattering), plus the diffusion part (Pacheco & Snieder 2005):
  K = (1 / 2 pi D) exp(-rho^2 / 2Dt) K0(rho^2 / 2Dt), D = U l* / 2, truncated at rho <= U t / 2.
* Kbody(rho, z, t | z0): body waves in a half-space with a free surface (image source), receiver at depth z0
  (PB boreholes): diffusion part (exact time integral for 3-D Gaussians:
  int G(a,s) G(b,t-s) ds = (a+b)/(4 pi D a b) G(a+b, t), summed over direct/image combinations, D = c l* / 3,
  c = 3.0 km/s, truncated at distance <= c t / 2) plus single-scattering part (density ~ 1/r1^2 + 1/r2^2 inside
  the c t / 2 sphere) with weight exp(-c t / l*).  Its rho-integral reduces to the 1-D depth kernel of
  sci_dvv_depth2_kernels.noise_body_kernel (checked in tests()).
* Surface/body partition (Obermann et al. 2013, 2016: surface waves dominate early, body waves late; the ratio is a
  function of lapse time / mean free time): a(t) = a_inf + (1 - a_inf) exp(-t / tau), tau = l* / U, a_inf = 0.3
  (default).  Buried receiver: a_eff = a r / (a r + 1 - a), r = surface-wave energy at the sensor depth relative
  to the surface (as in sci_dvv_depth2.G_of_alpha).
* Mean free path default: l*(f) = 20 km x (1 Hz / f_c), clipped to 5-150 km (0.2 Hz: 100 km; 2 Hz: 10 km;
  4 Hz: 5 km), i.e. within the 10-100 km crustal range of coda modelling (e.g. Margerin et al. 1999) and
  shorter at high frequency in the strongly heterogeneous Franciscan.  Alternatives: constant 5, 20, 50 km.

Stages: sw (surface-wave depth kernels, 4 workers) -> data/sci/dvv_3d/sw_kernels.npz;
        body (lateral + body kernels) -> /data/wsd02/maleen_data/dvv_3d/kbody.npz (+ small klat in sw file dir);
        tests.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import k0e

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sci_dvv_depth2_kernels as K2  # noqa: E402

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_3d'
BIG = '/data/wsd02/maleen_data/dvv_3d'
NPROC = int(os.environ.get("NPROC", "2"))
K2.ZMAX_GRID = 150.0
BANDS = ['0.1-0.3Hz', '0.3-0.6Hz', '0.5-1Hz', '1-3Hz', '3-5Hz']
BAND_F = {'0.1-0.3Hz': (0.1, 0.3), '0.3-0.6Hz': (0.3, 0.6), '0.5-1Hz': (0.5, 1.0), '1-3Hz': (1.0, 3.0), '3-5Hz': (3.0, 5.0)}
# coda lapse windows (s): sub-bands from scripts/sci_noise_dvv_sc_full.py CODA_SUB, 1-3/3-5 Hz from
# scripts/sci_noise_dvv_jewt.py CODA_NEW (PRODS['day'])
WIN = {'0.1-0.3Hz': (20.0, 120.0), '0.3-0.6Hz': (10.0, 60.0), '0.5-1Hz': (5.0, 40.0), '1-3Hz': (3.0, 30.0), '3-5Hz': (1.0, 8.0)}
LVZS = ['ref', 'soft', 'none']
LSTARS = ['f', '5', '20', '50']
C_BODY = 3.0
NT = 24
# fine kernel grids (edges, km)
RHO_E = np.unique(np.r_[np.arange(0, 5, 0.25), np.arange(5, 20, 0.5), np.arange(20, 60, 1.0), np.arange(60, 200.1, 2.0)])
Z_E = np.unique(np.r_[np.arange(0, 0.1, 0.01), np.arange(0.1, 0.3, 0.025), np.arange(0.3, 1.0, 0.05), np.arange(1, 3, 0.1),
                      np.arange(3, 10, 0.25), np.arange(10, 31, 0.5), np.arange(31, 60, 1.0), [60.0, 250.0]])


def fc(band):
    a, b = BAND_F[band]
    return np.sqrt(a * b)


def lstar_of(band, var):
    if var == 'f':
        return float(np.clip(20.0 / fc(band), 5.0, 150.0))
    return float(var)


def band_freqs(band, n=3):
    a, b = BAND_F[band]
    e = np.geomspace(a, b, n + 1)
    return np.sqrt(e[1:] * e[:-1])


def lapse(band):
    t1, t2 = WIN[band]
    t = np.linspace(t1, t2, NT)
    w = t ** 0.5
    return t, w / w.sum()


# ----------------------------------------------------------------------------------------- station 1-D models
_FUR = {}


def furlong_column(lat, lon, zmax=60.0):
    if not _FUR:
        for nm in ['pm', 'sm']:
            _FUR[nm] = pd.read_csv(f'{ROOT}/data/velocity/furlong2024/{nm}.llzv.gz', sep=r'\s+', header=None,
                                   names=['lon', 'lat', 'z', 'v'])
    out = {}
    for nm, d in _FUR.items():
        dd = (d.lon - lon) ** 2 * np.cos(np.radians(lat)) ** 2 + (d.lat - lat) ** 2
        i = dd.idxmin()
        c = d[(d.lon == d.lon[i]) & (d.lat == d.lat[i]) & (d.v > 0)].sort_values('z')
        zz = c.z.values - c.z.values[0]
        out[nm] = np.interp(np.arange(0, zmax + 0.01, 0.5), zz, c.v.values)
    return np.arange(0, zmax + 0.01, 0.5), out['pm'], out['sm']


def station_vmodel(lat, lon, variant):
    vm = K2.VModel(variant)
    z, vp, vs = furlong_column(lat, lon)
    vm.zf, vm.vpf, vm.vsf = z, vp, vs
    return vm


def _sw_job(args):
    sta, lat, lon, z0, variant = args
    os.environ['OMP_NUM_THREADS'] = '1'
    vm = station_vmodel(lat, lon, variant)
    res = {}
    for b in BANDS:
        KR, KL, aR, aL, UR, UL, cR, cL = [], [], [], [], [], [], [], []
        for f in band_freqs(b):
            r = K2.rayleigh(vm, f)
            lv = K2.love(vm, f)
            KR.append(K2.on_grid(r, Z_E)); KL.append(K2.on_grid(lv, Z_E))  # noqa: E702
            aR.append(K2.amp_at(r, z0)); aL.append(K2.amp_at(lv, z0))      # noqa: E702
            UR.append(r['U']); UL.append(lv['U']); cR.append(r['c']); cL.append(lv['c'])  # noqa: E702
        res[b] = dict(Kz=0.5 * (np.mean(KR, 0) + np.mean(KL, 0)), KzR=np.mean(KR, 0), KzL=np.mean(KL, 0),
                      r=float(0.5 * (np.mean(aR) + np.mean(aL))), U=float(0.5 * (np.mean(UR) + np.mean(UL))),
                      UR=float(np.mean(UR)), UL=float(np.mean(UL)), cR=float(np.mean(cR)), cL=float(np.mean(cL)))
    _, vs, _ = vm(0.5 * (Z_E[1:] + Z_E[:-1]))
    return sta, variant, res, vs


def stage_sw():
    ST = pd.read_csv(f'{OUT}/stations.csv')
    jobs = [(r.ns, r.lat, r.lon, r.z0_km, v) for v in LVZS for r in ST.itertuples()]
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(_sw_job, jobs, chunksize=1)
    flat = {'Z_E': Z_E}
    rows = []
    for sta, v, r, vs in res:
        flat[f'{sta}|{v}|vs'] = vs
        for b, o in r.items():
            for k, x in o.items():
                flat[f'{sta}|{v}|{b}|{k}'] = np.asarray(x)
            c = np.cumsum(o['Kz']) / o['Kz'].sum()
            rows.append(dict(sta=sta, lvz=v, band=b, U=o['U'], r_sensor=o['r'], ksum=o['Kz'].sum(),
                             z50_km=np.interp(0.5, c, Z_E[1:]), z90_km=np.interp(0.9, c, Z_E[1:]),
                             f_0_0p3=o['Kz'][Z_E[1:] <= 0.3 + 1e-9].sum(), f_0_1=o['Kz'][Z_E[1:] <= 1 + 1e-9].sum(),
                             f_0_3=o['Kz'][Z_E[1:] <= 3 + 1e-9].sum(), f_gt10=o['Kz'][Z_E[:-1] >= 10 - 1e-9].sum(),
                             f_20_31=o['Kz'][(Z_E[:-1] >= 20 - 1e-9) & (Z_E[1:] <= 31 + 1e-9)].sum()))
    np.savez_compressed(f'{OUT}/sw_kernels.npz', **flat)
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/sw_kernel_summary.csv', index=False, float_format='%.4g')
    print(T[T.lvz == 'ref'].groupby('band')[['U', 'r_sensor', 'z50_km', 'z90_km', 'f_0_0p3', 'f_0_1', 'f_0_3', 'f_gt10']].median().round(3))
    print('sw stage', round(time.time() - t0), 's', flush=True)


# ----------------------------------------------------------------------------------------- lateral / body
def _gl(edges, n=3):
    x, w = np.polynomial.legendre.leggauss(n)
    a, b = edges[:-1], edges[1:]
    pts = (0.5 * (b - a))[:, None] * x[None, :] + (0.5 * (a + b))[:, None]
    wts = (0.5 * (b - a))[:, None] * w[None, :]
    return pts, wts            # (ncell, n)


RP, RW = _gl(RHO_E)
ZP, ZW = _gl(Z_E)


def klat2(U, ls, t):
    """lateral 2-D kernel, ring masses on RHO_E cells (sums to 1)."""
    D = U * ls / 2.0
    rho = RP
    u = rho ** 2 / (2 * D * t)
    dens = (1 / (2 * np.pi * D)) * k0e(np.maximum(u, 1e-12)) * np.exp(-2 * u + 0.0) * 1.0
    # k0e(u) = exp(u) K0(u): exp(-u) K0(u) = k0e(u) exp(-2u)
    dens = dens * (rho <= U * t / 2)
    md = (dens * 2 * np.pi * rho * RW).sum(1)
    md /= max(md.sum(), 1e-30)
    R = U * t / 2
    lo, hi = RHO_E[:-1], RHO_E[1:]
    ms = np.clip(np.minimum(hi, R) - lo, 0, None) / R          # uniform per unit radius
    ps = np.exp(-U * t / ls)
    return ps * ms + (1 - ps) * md


def kbody(ls, t, z0, c=C_BODY, ss=True):
    """3-D body kernel masses on (RHO_E x Z_E) cells (sums to 1), receiver at depth z0, free surface by image."""
    D = c * ls / 3.0
    rho = RP.reshape(-1, 1, 3, 1)
    z = ZP.reshape(1, -1, 1, 3)
    wv = (2 * np.pi * RP * RW).reshape(-1, 1, 3, 1) * ZW.reshape(1, -1, 1, 3)
    r1 = np.sqrt(rho ** 2 + (z - z0) ** 2)
    r2 = np.sqrt(rho ** 2 + (z + z0) ** 2)
    four = 4 * D * t

    def G(R):
        return np.exp(-R ** 2 / four) / (np.pi * four) ** 1.5

    def I(a, b):
        return (a + b) / (4 * np.pi * D * a * b) * G(a + b)
    P0 = G(0.0) + G(2 * z0)
    Kd = (I(r1, r1) + 2 * I(r1, r2) + I(r2, r2)) / P0
    Kd = Kd * (r1 <= c * t / 2)
    md = (Kd * wv).sum(axis=(2, 3))
    md /= max(md.sum(), 1e-30)
    if not ss:
        return md
    R = c * t / 2
    Ks = (1 / r1 ** 2) * (r1 <= R) + (1 / r2 ** 2) * (r2 <= R)
    ms = (Ks * wv).sum(axis=(2, 3))
    ms /= max(ms.sum(), 1e-30)
    ps = np.exp(-c * t / ls)
    return ps * ms + (1 - ps) * md


def _body_job(args):
    b, lv, z0 = args
    ls = lstar_of(b, lv)
    t, _ = lapse(b)
    return (b, lv, z0), np.stack([kbody(ls, tt, z0) for tt in t]).astype('float32')


def stage_body():
    os.makedirs(BIG, exist_ok=True)
    ST = pd.read_csv(f'{OUT}/stations.csv')
    SW = np.load(f'{OUT}/sw_kernels.npz')
    z0s = sorted(set(np.round(ST.z0_km.values, 4)))
    t0 = time.time()
    # lateral surface-wave kernels: U = median group velocity over stations (ref LVZ) per band
    lat = {'RHO_E': RHO_E, 'Z_E': Z_E}
    Uband = {}
    for b in BANDS:
        Uband[b] = float(np.median([SW[f'{s}|ref|{b}|U'] for s in ST.ns]))
        t, w = lapse(b)
        for lv in LSTARS:
            lat[f'{b}|{lv}'] = np.stack([klat2(Uband[b], lstar_of(b, lv), tt) for tt in t])
        lat[f'{b}|t'] = t
        lat[f'{b}|w'] = w
    lat['Uband'] = json.dumps(Uband)
    np.savez_compressed(f'{OUT}/klat.npz', **lat)
    jobs = [(b, lv, z0) for b in BANDS for lv in LSTARS for z0 in z0s]
    with Pool(NPROC) as pool:
        res = pool.map(_body_job, jobs, chunksize=1)
    out = {'RHO_E': RHO_E, 'Z_E': Z_E, 'z0s': np.array(z0s)}
    for (b, lv, z0), x in res:
        out[f'{b}|{lv}|{z0:.4f}'] = x
    np.savez(f'{BIG}/kbody.npz', **out)
    print('body stage', round(time.time() - t0), 's; U per band', Uband, flush=True)


# ----------------------------------------------------------------------------------------- station kernels
class Kernels:
    """assemble station kernels K(rho, z) (masses on RHO_E x Z_E cells) for a variant."""

    def __init__(self):
        self.sw = np.load(f'{OUT}/sw_kernels.npz')
        self.lat = np.load(f'{OUT}/klat.npz')
        self.body = np.load(f'{BIG}/kbody.npz')
        self.z0s = self.body['z0s']

    def station(self, sta, z0, band, lvz='ref', lstar='f', ainf=0.3, alpha=None, sw_kind='Kz'):
        """alpha: None -> a(t) = ainf + (1-ainf) exp(-t/tau); float -> constant surface-wave fraction."""
        Kz = self.sw[f'{sta}|{lvz}|{band}|{sw_kind}']
        r = float(self.sw[f'{sta}|{lvz}|{band}|r']) if z0 > 0 else 1.0
        U = json.loads(str(self.lat['Uband']))[band]
        t, w = self.lat[f'{band}|t'], self.lat[f'{band}|w']
        L = self.lat[f'{band}|{lstar}']
        zk = self.z0s[np.argmin(np.abs(self.z0s - z0))]
        B = self.body[f'{band}|{lstar}|{zk:.4f}'].astype('float64')
        ls = lstar_of(band, lstar)
        a = (ainf + (1 - ainf) * np.exp(-t / (ls / U))) if alpha is None else np.full(len(t), float(alpha))
        ae = a * r / (a * r + 1 - a + 1e-12)
        ae = np.where(a >= 1, 1.0, ae)
        Ksw = np.einsum('t,tr->r', w * ae, L)[:, None] * Kz[None, :]
        Kb = np.einsum('t,trz->rz', w * (1 - ae), B)
        return Ksw + Kb, dict(r=r, a_mean=float((w * a).sum()), ae_mean=float((w * ae).sum()))


# ----------------------------------------------------------------------------------------- tests
def tests():
    out = {}
    # 2-D diffusion closed form vs numerical time integral
    U, ls, t = 3.0, 20.0, 30.0
    D = U * ls / 2
    rho = np.array([0.5, 2.0, 10.0, 30.0])
    s = np.linspace(0, t, 200001)[1:-1]
    G2 = lambda r, s: np.exp(-r ** 2 / (4 * D * s)) / (4 * np.pi * D * s)  # noqa: E731
    num = np.array([np.trapezoid(G2(r, s) * G2(r, t - s), s) for r in rho]) / G2(0, t)
    from scipy.special import k0
    ana = (1 / (2 * np.pi * D)) * np.exp(-rho ** 2 / (2 * D * t)) * k0(rho ** 2 / (2 * D * t))
    out['klat2_closed_form_rel_err'] = (num / ana - 1).round(4).tolist()
    # 3-D: rho-integral of the diffusion body kernel (no ss) vs depth2 noise_body_kernel (same D, weights)
    for z0 in [0.0, 0.2176]:
        win = (2.0, 20.0)
        tt = np.linspace(win[0], win[1], 60)
        ww = tt ** 0.5
        acc = sum(w_ * kbody(5.0, t_, z0, c=2.6, ss=False).sum(0) for t_, w_ in zip(tt, ww)) / ww.sum()
        ref = K2.noise_body_kernel(Z_E, win, lstar=5.0, v=2.6, z0=z0)
        c1, c2 = np.cumsum(acc), np.cumsum(ref) / ref.sum()
        out[f'body_depth_marginal_z50_this_vs_depth2_z0={z0}'] = [float(np.interp(0.5, c1, Z_E[1:])), float(np.interp(0.5, c2, Z_E[1:]))]
        out[f'body_depth_marginal_frac_top1km_this_vs_depth2_z0={z0}'] = [float(acc[Z_E[1:] <= 1].sum()), float(ref[Z_E[1:] <= 1].sum() / ref.sum())]
    return out


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('tests', 'all'):
        print(json.dumps(tests(), indent=1), flush=True)
    if st in ('sw', 'all'):
        stage_sw()
    if st in ('body', 'all'):
        stage_body()
