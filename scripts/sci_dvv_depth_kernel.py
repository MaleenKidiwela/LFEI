#!/usr/bin/env python3
"""Depth sensitivity of S-ALIGNED coda stretching for a deep (28 km) source: Monte Carlo
radiative transfer (scalar energy particles, isotropic scattering, mean free path l, constant
Vs = 3.6 km/s half-space with a specular free surface; intrinsic Q only reweights lapse times).

Travel-time perturbation of a coda path = -sum_A dv/v_A * T_A, T_A = time the path spends in region A
(paths fixed to first order; Pacheco & Snieder 2005 / Mayor et al. 2014 style kernels).
Our measurement aligns every stack on direct S and stretches about S, so for a wavelet arriving at
lapse tau = t - t_S the apparent dv/v is   sum_A dv/v_A * K_A(tau),  K_A = (T_A(t) - T_A^dirS)/tau,
and sum_A K_A = 1 exactly (a uniform change is recovered).  A stretching fit over a window is
an intensity * tau^2 weighted mean of K_A (LSQ of dt = eps*tau).
Shallow low-velocity layer (0-h, one-way vertical time t_L = h/Vs_L): not in the MC geometry; every
free-surface bounce adds 2 t_L (waves near-vertical in the slow layer) and the final up-going pass
cancels against direct S, so K_layer(tau) = 2 t_L <N_bounce(tau)> / tau.  Surface-wave / in-layer
scattered energy (not modelled) would add to K_layer at surface sensors only.
Outputs: data/sci/dvv_depth/kernel_mc.csv (per l, distance, tau bin), kernel_windows.csv/json.
"""
import json
import os

import numpy as np
import pandas as pd

OUT = 'data/sci/dvv_depth'
V = 3.6
ZS = 28.0
ZB = np.array([0, 2, 5, 10, 15, 20, 25, 35, 50, 1e9])   # depth bins (km)
RSPH = 10.0                                                   # 'LFE source region' sphere radius (km)
F0, QI = 4.0, 1000.0                                          # intrinsic attenuation weighting
DIST = [5, 20, 30, 40, 50, 60]
HALF = 2.5
TAU_BINS = np.arange(0, 25.01, 1.0)


def seg_depth_time(z1, z2, s):
    """time spent in each depth bin along straight segments z1->z2 of length s (vectorised)."""
    lo, hi = np.minimum(z1, z2), np.maximum(z1, z2)
    dz = hi - lo
    out = np.zeros((len(z1), len(ZB) - 1))
    for k in range(len(ZB) - 1):
        ov = np.clip(np.minimum(hi, ZB[k + 1]) - np.maximum(lo, ZB[k]), 0, None)
        flat = dz < 1e-9
        frac = np.where(flat, ((lo >= ZB[k]) & (lo < ZB[k + 1])).astype(float), ov / np.where(flat, 1, dz))
        out[:, k] = frac * s / V
    return out


def seg_sphere_time(p, u, s, c=np.array([0, 0, ZS])):
    """time inside sphere(c, RSPH) along p + u*x, x in [0, s]."""
    w = p - c
    b = np.einsum('ij,ij->i', w, u)
    cc = np.einsum('ij,ij->i', w, w) - RSPH ** 2
    disc = b * b - cc
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0))
    x1, x2 = np.clip(-b - sq, 0, s), np.clip(-b + sq, 0, s)
    return np.where(ok, np.clip(x2 - x1, 0, None), 0) / V


def run(l, n=1000000, tmax=45.0, seed=0):
    """l: mean free path (km), or (l_shallow, z_c, l_deep) for depth-dependent scattering
    (delta tracking: tentative collisions at rate 1/l_min, real with probability l_min/l(z))."""
    rng = np.random.default_rng(seed)
    if np.isscalar(l):
        lmin, lof = float(l), (lambda z: np.full(len(z), float(l)))
    else:
        ls, zc, ld = l
        lmin, lof = min(ls, ld), (lambda z: np.where(z < zc, ls, ld))
    p = np.zeros((n, 3)); p[:, 2] = ZS                      # noqa: E702
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]   # noqa: E702
    t = np.zeros(n); Td = np.zeros((n, len(ZB) - 1)); Ts = np.zeros(n); nb = np.zeros(n)   # noqa: E702
    rec = []
    alive = np.ones(n, bool)
    while alive.any():
        i = np.where(alive)[0]
        s = rng.exponential(lmin, len(i))
        s = np.minimum(s, (tmax - t[i]) * V)
        # does the segment reach the surface?
        up = u[i, 2] < 0
        s_hit = np.where(up, p[i, 2] / np.maximum(-u[i, 2], 1e-12), np.inf)
        hit = s_hit < s
        ss = np.where(hit, s_hit, s)
        p_new = p[i] + u[i] * ss[:, None]
        Td[i] += seg_depth_time(p[i, 2], p_new[:, 2], ss)
        Ts[i] += seg_sphere_time(p[i], u[i], ss)
        t[i] += ss / V
        p[i] = p_new
        # surface hits: record, reflect, continue the remaining path without scattering
        h = i[hit]
        if len(h):
            r = np.hypot(p[h, 0], p[h, 1])
            for d in DIST:
                m = np.abs(r - d) < HALF
                if m.any():
                    hh = h[m]
                    rec.append(np.c_[np.full(len(hh), d), t[hh], nb[hh], Td[hh], Ts[hh]])
            nb[h] += 1
            u[h, 2] *= -1
            p[h, 2] = 0.0
            # remaining distance
            rem = (s - ss)[hit]
            p2 = p[h] + u[h] * rem[:, None]
            Td[h] += seg_depth_time(p[h, 2], p2[:, 2], rem)
            Ts[h] += seg_sphere_time(p[h], u[h], rem)
            t[h] += rem / V
            p[h] = p2
            # (a second hit within the remaining straight leg is impossible: it now travels down)
        # scatter those that did not die
        sc = i[t[i] < tmax - 1e-9]
        sc = sc[rng.random(len(sc)) < lmin / lof(p[sc, 2])]      # real (not null) collisions
        nu = rng.normal(size=(len(sc), 3))
        u[sc] = nu / np.linalg.norm(nu, axis=1)[:, None]
        alive[:] = False
        alive[i[t[i] < tmax - 1e-9]] = True
    return np.vstack(rec)


def direct(d):
    p0, p1 = np.array([[0, 0, ZS]]), np.array([[d, 0, 0.0]])
    L = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / L
    return L / V, seg_depth_time(np.array([ZS]), np.array([0.0]), np.array([L]))[0], seg_sphere_time(p0, u, np.array([L]))[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    names = [f'z{int(ZB[k])}_{int(ZB[k+1]) if ZB[k+1] < 1e8 else "inf"}' for k in range(len(ZB) - 1)]
    for l in [20, 50, 100, 200, (10, 5.0, 100), (5, 3.0, 100)]:
        R = np.vstack([run(l, seed=s) for s in range(4)])
        for d in DIST:
            tS, TdS, TsS = direct(d)
            A = R[R[:, 0] == d]
            tau = A[:, 1] - tS
            ok = tau > 0.5          # exclude the ballistic direct arrival itself
            A, tau = A[ok], tau[ok]
            kb = np.digitize(tau, TAU_BINS) - 1
            for k in range(len(TAU_BINS) - 1):
                m = kb == k
                if m.sum() < 30:
                    continue
                tc = 0.5 * (TAU_BINS[k] + TAU_BINS[k + 1])
                Td = A[m, 3:3 + len(names)]
                r = dict(l_km=(str(l) if np.isscalar(l) else f'{l[0]}/{l[1]:g}km/{l[2]}'), dist_km=d, tS=tS, tau=tc, n=int(m.sum()),
                         # intensity per unit time (flux counts) x intrinsic attenuation
                         I=m.sum() * np.exp(-2 * np.pi * F0 * (tc + tS) / QI),
                         nbounce=A[m, 2].mean(), nbounce_se=A[m, 2].std() / np.sqrt(m.sum()),
                         K_sphere=(A[m, -1].mean() - TsS) / tau[m].mean())
                for j, nm in enumerate(names):
                    r['K_' + nm] = (Td[:, j].mean() - TdS[j]) / tau[m].mean()
                rows.append(r)
        print('l', l, 'done', flush=True)
    K = pd.DataFrame(rows)
    K['K_deep15'] = K[[c for c in K if c.startswith('K_z') and int(c.split('_')[1][1:]) >= 15]].sum(axis=1)
    K['K_upper5'] = K[['K_z0_2', 'K_z2_5']].sum(axis=1)
    K.to_csv(f'{OUT}/kernel_mc.csv', index=False)
    # window averages: weight I * tau^2 (LSQ stretch); t_L for the shallow layer
    W = []
    for (l, d), g in K.groupby(['l_km', 'dist_km']):
        for wn, (a, b) in dict(early=(3, 8), late=(8, 15), full=(3, 12)).items():
            q = g[(g.tau >= a) & (g.tau <= b)]
            w = q.I * q.tau ** 2
            r = dict(l_km=l, dist_km=d, window=wn)
            for c in ['K_deep15', 'K_upper5', 'K_sphere', 'K_z5_10', 'K_z10_15']:
                r[c] = float(np.sum(w * q[c]) / w.sum())
            # layer kernel per second of one-way layer time: 2 <N>/tau
            r['K_layer_per_tL'] = float(np.sum(w * 2 * q.nbounce / q.tau) / w.sum())
            W.append(r)
    W = pd.DataFrame(W)
    W.to_csv(f'{OUT}/kernel_windows.csv', index=False)
    print(W.round(3).to_string())


if __name__ == '__main__':
    main()
