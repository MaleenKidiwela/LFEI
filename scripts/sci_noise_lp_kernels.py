#!/usr/bin/env python3
"""Depth sensitivity of long-period inter-station noise-coda dv/v (helper of sci_noise_lp_dvv.py).

Model: 1-D average of the Furlong et al. (2024) Vp/Vs model (data/velocity/furlong2024/{pm,sm}.llzv.gz) over all
columns within 60 km of 40.06N -123.62 (0-70 km below sea level; the uppermost node -0.25 km is the surface),
constant mantle (Vs 4.54, Vp 8.09) below 70 km, Gardner density. Variant 'lvz' adds the shallow low-velocity zone
of sci_dvv_depth2_kernels ('ref': Vs 0.4 km/s at the surface -> Furlong at 1 km) - irrelevant at these periods but
reported. Fundamental-mode Rayleigh and Love waves with the project's validated thin-layer solver
(scripts/sci_dvv_depth2_kernels.py: rayleigh(), love(); rigid base at 3 wavelengths, here allowed to 300 km).
Coda-stretching kernel K(z) = (U/c) dln c/dln Vs(z) (Vp/Vs fixed): sums to 1 for a uniform change, so a change
delta confined to a depth range [z1, z2] produces an apparent coda dv/v = delta * sum_{z1..z2} K. Kernels are
averaged over 11 frequencies across each band.
Outputs: data/sci/noise_lp/kernels.csv (per km), kernel_fractions.csv, dispersion.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.path.join(ROOT, 'data/sci/noise_lp')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import sci_dvv_depth2_kernels as K2  # noqa: E402

K2.ZMAX_GRID = 300.0
BANDS = {'0.05-0.1': (0.05, 0.1), '0.1-0.2': (0.1, 0.2), '0.2-0.3': (0.2, 0.3)}
DEPTH_RANGES = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (10, 30), (0, 10), (30, 300), (0, 5), (5, 30)]


def regional_profile():
    p = os.path.join(OUT, 'furlong_regional_profile.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    out = {}
    for nm in ['pm', 'sm']:
        d = pd.read_csv(os.path.join(ROOT, f'data/velocity/furlong2024/{nm}.llzv.gz'), sep=r'\s+', header=None,
                        names=['lon', 'lat', 'z', 'v'])
        r = 111.19 * np.hypot(d.lat - 40.06, (d.lon + 123.62) * np.cos(np.radians(40.06)))
        q = d[(r < 60) & (d.v > 0) & (d.z >= -0.25)]
        g = q.groupby('z').v.mean()
        out[nm] = g
    P = pd.DataFrame(dict(z_km=out['sm'].index.values + 0.25, vp=out['pm'].values, vs=out['sm'].values))
    P.to_csv(p, index=False)
    return P


class DeepModel:
    def __init__(self, lvz=False):
        F = regional_profile()
        self.z, self.vp, self.vs = F.z_km.values, F.vp.values, F.vs.values
        self.lvz = lvz

    def __call__(self, z):
        z = np.asarray(z, float)
        vs = np.interp(z, self.z, self.vs)
        vp = np.interp(z, self.z, self.vp)
        if self.lvz:
            vs0, zL, n = 0.40, 1.0, 0.5
            vsL, vpL = np.interp(zL, self.z, self.vs), np.interp(zL, self.z, self.vp)
            x = np.clip(z / zL, 0, 1) ** n
            m = z < zL
            vsl = vs0 + (vsL - vs0) * x
            vs = np.where(m, vsl, vs)
            vp = np.where(m, vsl * (2.0 + (vpL / vsL - 2.0) * x), vp)
        rho = np.maximum(1.74 * vp ** 0.25, 1.9)
        return vp, vs, rho


def main():
    zg = np.arange(0, 301, 1.0)
    rows, disp = [], []
    fr = []
    for variant in ['furlong', 'furlong_lvz']:
        vm = DeepModel(lvz=(variant == 'furlong_lvz'))
        for bname, (f1, f2) in BANDS.items():
            for wave, fn in [('rayleigh', K2.rayleigh), ('love', K2.love)]:
                ks = []
                for f in np.linspace(f1, f2, 11):
                    r = fn(vm, f)
                    k = K2.on_grid(r, zg)
                    ks.append(k)
                    disp.append(dict(variant=variant, wave=wave, f=f, c=r['c'], U=r['U'], ksum=float(k.sum())))
                k = np.mean(ks, axis=0)
                for i in range(len(zg) - 1):
                    rows.append(dict(variant=variant, band=bname, wave=wave, z_top=zg[i], z_bot=zg[i + 1], K=k[i]))
                rec = dict(variant=variant, band=bname, wave=wave, ksum=float(k.sum()),
                           z50=float(np.interp(0.5, np.cumsum(k) / k.sum(), zg[1:])))
                for a, b in DEPTH_RANGES:
                    rec[f'f_{a}_{b}'] = float(k[(zg[:-1] >= a) & (zg[1:] <= b)].sum())
                fr.append(rec)
                print(variant, bname, wave, {kk: round(v, 3) for kk, v in rec.items() if kk.startswith('f_') or kk in ('ksum', 'z50')}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'kernels.csv'), index=False)
    pd.DataFrame(fr).to_csv(os.path.join(OUT, 'kernel_fractions.csv'), index=False)
    pd.DataFrame(disp).to_csv(os.path.join(OUT, 'dispersion.csv'), index=False)


if __name__ == '__main__':
    main()
