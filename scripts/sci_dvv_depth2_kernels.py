#!/usr/bin/env python3
"""Depth sensitivity kernels for the dv/v depth study (helper of sci_dvv_depth2.py).

1. 1-D velocity models: Furlong et al. (2024) 3-D model (data/velocity/furlong2024/{pm,sm}.llzv.gz) averaged
   beneath the stations (0-12 km), plus a shallow low-velocity zone (LVZ) that the 0.5-km grid cannot resolve:
   Vs(z) = Vs0 + (Vs_F(zL) - Vs0) * (z/zL)^n for z < zL, Vp/Vs 2.0 at the surface -> 1.78 at zL, Gardner density.
2. Surface waves: fundamental-mode Rayleigh and Love waves by the thin-layer method (Lysmer 1970; Kausel &
   Roesset 1981): linear finite elements in z, rigid base several wavelengths deep, exact eigenproblem
   (k^2 A + k B + G - w^2 M) u = 0 (vertical displacement scaled by i so all matrices are real symmetric).
   Kernels by first-order perturbation of the eigenvalue (no finite differences):
       dk = -u^T (k^2 dA + k dB + dG - w^2 dM) u / u^T (2kA + B) u,   U = u^T(2kA+B)u / (2 w u^T M u).
   Coda-stretching kernel of a surface wave = (U/c) * dln c/dln Vs (z): a phase-velocity change dc/c delays
   a wave arriving at lapse t = L/U by dt/t = -(U/c) dc/c, so a uniform change is recovered exactly
   (sum of kernel = 1 when Vp scales with Vs; checked numerically).
3. Body waves:
   LFE coda (S-aligned stretching, 28-km source): depth profile from the earlier Monte Carlo radiative-transfer
   kernels (data/sci/dvv_depth/kernel_mc.csv: per lapse bin, time-in-depth-bin minus direct-S time / tau),
   distributed within each bin in proportion to vertical S slowness of the layered model (slow near-surface
   layers hold more of each free-surface bounce) and renormalised to sum 1.
   Ambient noise (coincident source/receiver, stretching about zero lag): diffusion depth kernel with a
   reflecting free surface, K(z,t) = 1/t int_0^t P(z0->z,s) P(z->z0,t-s) ds / P(z0->z0,t), lateral factor
   cancels; truncated at the single-scattering depth v t/2.
   Buried receiver (z0 = sensor depth): handled by z0 in the diffusion kernel (noise) and by the
   free-surface reflected (down-going) wave term for the S-aligned LFE coda (see lfe_body_kernel).
"""
import os

import numpy as np
import pandas as pd
import scipy.linalg as sla

ROOT = '/home/seismic/LFE'
ZMAX_GRID = 40.0


# ------------------------------------------------------------------------------------------- velocity models
def furlong_profile(cache='data/sci/dvv_depth2/furlong_profile.csv'):
    """mean Vp, Vs (km/s) beneath the dv/v stations, z = 0..12 km below the local surface (0.5-km nodes)."""
    p = os.path.join(ROOT, cache)
    if os.path.exists(p):
        return pd.read_csv(p)
    st = {'KMPB': (40.4157, -124.1212), 'KMR': (40.4167, -123.8735), 'BRIC': (40.2585, -123.8355),
          'HAYF': (40.4720, -123.6696), 'RBOW': (40.3890, -123.5650), 'B045': (40.436, -123.9965),
          'B046': (40.322, -124.2843), 'B047': (40.2499, -124.1698), 'B049': (40.2375, -123.8197),
          'B932': (40.275, -124.2212), 'B933': (40.060, -123.969), 'B934': (39.8445, -123.6914),
          'B935': (40.4787, -123.5732)}
    out = {}
    for nm in ['pm', 'sm']:
        d = pd.read_csv(os.path.join(ROOT, f'data/velocity/furlong2024/{nm}.llzv.gz'), sep=r'\s+', header=None,
                        names=['lon', 'lat', 'z', 'v'])
        prof = []
        for s, (la, lo) in st.items():
            dd = (d.lon - lo) ** 2 * np.cos(np.radians(la)) ** 2 + (d.lat - la) ** 2
            i = dd.idxmin()
            c = d[(d.lon == d.lon[i]) & (d.lat == d.lat[i]) & (d.v > 0)].sort_values('z')
            c = c[c.z >= c.z.min()]
            zz = c.z.values - c.z.values[np.argmax(c.v.values > 0)]          # depth below the first valid node
            prof.append(np.interp(np.arange(0, 12.01, 0.5), zz, c.v.values))
        out[nm] = np.mean(prof, axis=0)
    P = pd.DataFrame(dict(z_km=np.arange(0, 12.01, 0.5), vp=out['pm'], vs=out['sm']))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    P.to_csv(p, index=False)
    return P


LVZ_VARIANTS = {            # name: (Vs0 km/s, zL km, exponent)
    'ref': (0.40, 1.0, 0.5),
    'soft': (0.25, 0.5, 0.5),
    'stiff': (0.70, 1.5, 0.5),
    'thin': (0.40, 0.3, 0.5),
    'none': (None, None, None),
}


class VModel:
    def __init__(self, variant='ref'):
        F = furlong_profile()
        self.zf, self.vpf, self.vsf = F.z_km.values, F.vp.values, F.vs.values
        self.variant = variant
        self.vs0, self.zL, self.n = LVZ_VARIANTS[variant]

    def _furlong(self, z):
        zc = np.clip(z, 0, self.zf[-1])
        vs = np.interp(zc, self.zf, self.vsf) + 0.012 * np.clip(z - self.zf[-1], 0, None)
        vp = np.interp(zc, self.zf, self.vpf) + 0.02 * np.clip(z - self.zf[-1], 0, None)
        return vp, vs

    def __call__(self, z):
        z = np.asarray(z, float)
        vp, vs = self._furlong(z)
        if self.vs0 is not None:
            vpL, vsL = self._furlong(np.full_like(z, self.zL))
            m = z < self.zL
            x = np.clip(z / self.zL, 0, 1) ** self.n
            vs_l = self.vs0 + (vsL - self.vs0) * x
            r = 2.0 + (vpL / vsL - 2.0) * x                              # Vp/Vs 2.0 -> Furlong
            vs = np.where(m, vs_l, vs)
            vp = np.where(m, vs_l * r, vp)
        rho = 1.74 * vp ** 0.25                                         # Gardner (g/cc, km/s)
        rho = np.maximum(rho, 1.9)
        return vp, vs, rho

    def vertical_time(self, z0, z1, p=0.0, n=400):
        """S one-way vertical-ish time between depths (ray parameter p s/km)."""
        z = np.linspace(z0, z1, n)
        _, vs, _ = self(z)
        q = np.sqrt(np.clip(1 / vs ** 2 - p ** 2, 1e-6, None))
        return np.trapz(q, z)


# ------------------------------------------------------------------------------------------- thin-layer method
def make_mesh(vm, f, zmax_lam=3.0, ppw=16, hmin=0.0015):
    """element boundaries (km): h <= Vs(z)/(f ppw); base at ~zmax_lam wavelengths of the local Vs."""
    z = [0.0]
    while True:
        _, vs, _ = vm(np.array([z[-1]]))
        h = max(min(vs[0] / (f * ppw), 0.5), hmin)
        z.append(z[-1] + h)
        _, vsb, _ = vm(np.array([z[-1]]))
        if z[-1] >= zmax_lam * vsb[0] / f or z[-1] >= ZMAX_GRID:
            break
    return np.array(z)


def element_props(vm, zb):
    zc = 0.5 * (zb[1:] + zb[:-1])
    vp, vs, rho = vm(zc)
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2 * mu
    return zc, np.diff(zb), lam, mu, rho, vp, vs


def _love_mats(h, mu, rho):
    ne = len(h)
    n = ne + 1
    A, G, M = (np.zeros((n, n)) for _ in range(3))
    E1 = np.array([[2, 1], [1, 2]]) / 6.0
    E2 = np.array([[1, -1], [-1, 1]])
    for e in range(ne):
        s = slice(e, e + 2)
        A[s, s] += mu[e] * h[e] * E1
        G[s, s] += mu[e] / h[e] * E2
        M[s, s] += rho[e] * h[e] * E1
    return A, G, M


def _rayl_elem(h, lam, mu, rho):
    """element matrices, dof order (U1, V1, U2, V2), V = -i W (all real symmetric)."""
    l2m = lam + 2 * mu
    A = np.zeros((4, 4)); G = np.zeros((4, 4)); M = np.zeros((4, 4)); B = np.zeros((4, 4))  # noqa: E702
    E1 = np.array([[2, 1], [1, 2]]) / 6.0
    E2 = np.array([[1, -1], [-1, 1]])
    iu, iv = [0, 2], [1, 3]
    A[np.ix_(iu, iu)] = l2m * h * E1
    A[np.ix_(iv, iv)] = mu * h * E1
    G[np.ix_(iu, iu)] = mu / h * E2
    G[np.ix_(iv, iv)] = l2m / h * E2
    M[np.ix_(iu, iu)] = rho * h * E1
    M[np.ix_(iv, iv)] = rho * h * E1
    # coupling: int(-lam U V' + mu U' V) for U_i, V_j
    Cuv = np.array([[lam / 2 - mu / 2, -lam / 2 - mu / 2], [lam / 2 + mu / 2, -lam / 2 + mu / 2]])
    B[np.ix_(iu, iv)] = Cuv
    B[np.ix_(iv, iu)] = Cuv.T
    return A, B, G, M


def _rayl_mats(h, lam, mu, rho):
    ne = len(h)
    n = 2 * (ne + 1)
    A, B, G, M = (np.zeros((n, n)) for _ in range(4))
    for e in range(ne):
        a, b, g, m = _rayl_elem(h[e], lam[e], mu[e], rho[e])
        s = slice(2 * e, 2 * e + 4)
        A[s, s] += a; B[s, s] += b; G[s, s] += g; M[s, s] += m   # noqa: E702
    return A, B, G, M


def love(vm, f, vp_fixed=False, zb=None):
    w = 2 * np.pi * f
    zb = make_mesh(vm, f) if zb is None else zb
    zc, h, lam, mu, rho, vp, vs = element_props(vm, zb)
    A, G, M = _love_mats(h, mu, rho)
    A, G, M = A[:-1, :-1], G[:-1, :-1], M[:-1, :-1]                   # rigid base
    # (w^2 M - G) y = k^2 A y ; A positive definite
    ev, V = sla.eigh(w ** 2 * M - G, A)
    j = np.argmax(ev)
    k = np.sqrt(ev[j])
    y = V[:, j]
    num = 2 * k * (y @ A @ y)
    U = num / (2 * w * (y @ M @ y))
    c = w / k
    K = np.zeros(len(h))
    E1 = np.array([[2, 1], [1, 2]]) / 6.0
    E2 = np.array([[1, -1], [-1, 1]])
    yy = np.r_[y, 0.0]
    for e in range(len(h)):
        ye = yy[e:e + 2]
        dmu = 2 * mu[e]                                               # d mu / d ln Vs
        dL = k ** 2 * dmu * h[e] * (ye @ E1 @ ye) + dmu / h[e] * (ye @ E2 @ ye)
        dk = -dL / num
        K[e] = -dk / k                                                 # d ln c / d ln Vs_e
    amp = yy ** 2
    return dict(c=c, U=U, k=k, zb=zb, zc=zc, h=h, Kc=K, Kcoda=K * U / c, amp_nodes=amp / amp[0])


def rayleigh(vm, f, vp_fixed=False, zb=None):
    w = 2 * np.pi * f
    zb = make_mesh(vm, f) if zb is None else zb
    zc, h, lam, mu, rho, vp, vs = element_props(vm, zb)
    A, B, G, M = _rayl_mats(h, lam, mu, rho)
    A, B, G, M = A[:-2, :-2], B[:-2, :-2], G[:-2, :-2], M[:-2, :-2]   # rigid base
    n = A.shape[0]
    # shift-invert around the slowest possible wavenumber
    kmax = w / (0.80 * vs.min())
    Z = np.zeros((n, n))
    I = np.eye(n)
    L0 = G - w ** 2 * M
    # [u; k u]: [[0, I], [-L0, -B]] x = k [[I, 0], [0, A]] x
    Aa = np.block([[Z, I], [-L0, -B]])
    Bb = np.block([[I, Z], [Z, A]])
    ev, V = sla.eig(Aa, Bb)
    ok = np.isfinite(ev) & (np.abs(ev.imag) < 1e-6 * np.abs(ev)) & (ev.real > 0) & (ev.real < kmax * 1.05)
    if not ok.any():
        raise RuntimeError('no Rayleigh root')
    cand = np.where(ok)[0]
    j = cand[np.argmax(ev.real[cand])]
    k = ev.real[j]
    u = V[:n, j].real
    if np.abs(u).max() < 1e-12:
        u = V[:n, j].imag
    # refine with inverse iteration on the symmetric quadratic problem
    for _ in range(3):
        L = k ** 2 * A + k * B + L0
        dLk = 2 * k * A + B
        k = k - (u @ L @ u) / (u @ dLk @ u)
        try:
            u = np.linalg.solve(L + 1e-14 * np.eye(n), dLk @ u)
        except np.linalg.LinAlgError:
            break
        u /= np.linalg.norm(u)
    num = u @ (2 * k * A + B) @ u
    U = num / (2 * w * (u @ M @ u))
    c = w / k
    uu = np.r_[u, 0.0, 0.0]
    K = np.zeros(len(h))
    for e in range(len(h)):
        ue = uu[2 * e:2 * e + 4]
        dmu = 2 * mu[e]
        dlam = -2 * dmu if vp_fixed else 2 * lam[e]                     # Vp fixed or Vp/Vs fixed
        a1, b1, g1, _ = _rayl_elem(h[e], 1.0, 0.0, 0.0)                  # d/d lambda
        a2, b2, g2, _ = _rayl_elem(h[e], 0.0, 1.0, 0.0)                  # d/d mu
        dL = ue @ ((k ** 2 * a1 + k * b1 + g1) * dlam + (k ** 2 * a2 + k * b2 + g2) * dmu) @ ue
        K[e] = (dL / num) / k
    amp = uu[0::2] ** 2 + uu[1::2] ** 2
    return dict(c=c, U=U, k=k, zb=zb, zc=zc, h=h, Kc=K, Kcoda=K * U / c, amp_nodes=amp / amp[0])


def on_grid(res, zg):
    """kernel density (per km) on grid zg (cell edges) by integrating element kernels."""
    cum = np.r_[0, np.cumsum(res['Kcoda'])]
    C = np.interp(zg, res['zb'], cum, right=cum[-1])
    return np.diff(C)


def amp_at(res, d):
    zn = res['zb']
    return float(np.interp(d, zn, res['amp_nodes'], right=0.0))


# ------------------------------------------------------------------------------------------- body kernels
def lfe_body_kernel(vm, zg, window, l_km='20', dist=40, bh_depth=0.0, f_shift=0.0, p=0.15):
    """S-aligned LFE-coda body-wave kernel on grid edges zg (km); returns per-cell sensitivity (sums to 1).
    window: (tau1, tau2) s.  MC bins of kernel_mc.csv are redistributed in proportion to vertical S slowness of
    the layered model, scaled by (model time / MC 3.6 km/s time) within each bin, then renormalised.
    Buried receiver: the S alignment cancels the up-going final pass; the free-surface reflected (down-going)
    wave adds 2 passes through [0, min(z, d)] for every arrival, also in direct S, so it cancels to first order.
    f_shift = (down-going energy fraction in the coda) - (in the S-alignment window): residual constant shift
    2 T(0..d) on that fraction -> apparent stretch f_shift * 2 s(z) <tau>/<tau^2> for z < d."""
    K = pd.read_csv(os.path.join(ROOT, 'data/sci/dvv_depth/kernel_mc.csv'))
    K = K[(K.l_km == l_km) & (K.dist_km == dist)]
    q = K[(K.tau >= window[0]) & (K.tau <= window[1])]
    w = q.I * q.tau ** 2
    edges = [0, 2, 5, 10, 15, 20, 25, 35, 50, 1e9]
    names = ['K_z0_2', 'K_z2_5', 'K_z5_10', 'K_z10_15', 'K_z15_20', 'K_z20_25', 'K_z25_35', 'K_z35_50', 'K_z50_inf']
    kb = np.array([np.sum(w * q[c]) / w.sum() for c in names])
    zc = 0.5 * (zg[1:] + zg[:-1])
    dz = np.diff(zg)
    _, vs, _ = vm(zc)
    s = np.sqrt(np.clip(1 / vs ** 2 - p ** 2, 1e-6, None))               # vertical slowness
    s36 = np.sqrt(1 / 3.6 ** 2 - p ** 2)
    out = np.zeros(len(zc))
    for i in range(len(names)):
        m = (zc >= edges[i]) & (zc < edges[i + 1])
        if not m.any():
            continue
        tb = np.sum(s[m] * dz[m])
        t36 = s36 * min(edges[i + 1], zg[-1]) - s36 * edges[i]
        out[m] = kb[i] * (s[m] * dz[m]) / tb * (tb / max(t36, 1e-9))
    out /= out.sum()
    if bh_depth > 0 and f_shift != 0:
        tau = q.tau.values
        g = float(np.sum(w * tau) / np.sum(w * tau ** 2))
        m = zc < bh_depth
        out = out + np.where(m, f_shift * 2 * s * dz * g, 0.0)
    return out


def noise_body_kernel(zg, window, lstar=5.0, v=2.6, z0=0.0, nt=60, ns=200):
    """diffusion depth kernel for coincident source/receiver at depth z0 (km), reflecting surface,
    window (t1, t2) s, LSQ-stretch weights ~ t^2 x I(t), I ~ t^-1.5; truncated at v t/2 (+ z0)."""
    D = v * lstar / 3.0
    zc = 0.5 * (zg[1:] + zg[:-1])
    dz = np.diff(zg)

    def P(za, zb, t):
        g = lambda x: np.exp(-x ** 2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)  # noqa: E731
        return g(za - zb) + g(za + zb)
    ts = np.linspace(window[0], window[1], nt)
    acc = np.zeros(len(zc))
    wsum = 0.0
    for t in ts:
        s = (np.arange(ns) + 0.5) / ns * t
        Kt = np.zeros(len(zc))
        for si in s:
            Kt += P(z0, zc, si) * P(zc, z0, t - si)
        Kt *= (t / ns) / (P(z0, z0, t) * t)
        Kt *= (zc <= v * t / 2 + z0)
        Kt = Kt * dz
        Kt /= max(Kt.sum(), 1e-12)
        wt = t ** 0.5
        acc += wt * Kt
        wsum += wt
    return acc / wsum


# ------------------------------------------------------------------------------------------- tests
def tests():
    out = {}

    class HS:                                       # homogeneous Poisson solid
        def __call__(self, z):
            z = np.asarray(z, float)
            return np.full(z.shape, 3.0 * np.sqrt(3)), np.full(z.shape, 3.0), np.full(z.shape, 2.5)
    r = rayleigh(HS(), 2.0)
    out['rayleigh_halfspace_c_over_vs'] = r['c'] / 3.0                 # exact 0.919402
    out['rayleigh_halfspace_kernel_sum'] = float(r['Kcoda'].sum())

    class LOH:                                      # layer 1 km Vs 2 over Vs 3.5 (Love)
        def __call__(self, z):
            z = np.asarray(z, float)
            vs = np.where(z < 1.0, 2.0, 3.5)
            return vs * 1.8, vs, np.where(z < 1.0, 2.4, 2.7)
    f = 1.0
    L = love(LOH(), f, zb=np.unique(np.r_[np.linspace(0, 1, 101), np.linspace(1, 15, 701)]))
    # analytic Love dispersion: tan(k h sqrt(c^2/b1^2-1)) = mu2 sqrt(1-c^2/b2^2) / (mu1 sqrt(c^2/b1^2-1))
    from scipy.optimize import brentq
    w = 2 * np.pi * f
    b1, b2, r1, r2 = 2.0, 3.5, 2.4, 2.7
    mu1, mu2 = r1 * b1 ** 2, r2 * b2 ** 2

    def fn(c):
        q1 = np.sqrt(c ** 2 / b1 ** 2 - 1)
        q2 = np.sqrt(1 - c ** 2 / b2 ** 2)
        return np.tan(w / c * 1.0 * q1) - mu2 * q2 / (mu1 * q1)
    cs = np.linspace(b1 * 1.0001, b2 * 0.9999, 4000)
    v = fn(cs)
    roots = [brentq(fn, cs[i], cs[i + 1]) for i in range(len(cs) - 1) if np.sign(v[i]) != np.sign(v[i + 1]) and abs(v[i]) < 10 and abs(v[i + 1]) < 10]
    out['love_LOH_c_tlm'] = L['c']
    out['love_LOH_c_exact_fund'] = min(roots) if roots else np.nan
    out['love_LOH_kernel_sum'] = float(L['Kcoda'].sum())
    vm = VModel('ref')
    for fn_, nm in [(rayleigh, 'R'), (love, 'L')]:
        for fq in [1.5, 6.0]:
            res = fn_(vm, fq)
            out[f'{nm}_{fq}Hz_c'] = res['c']
            out[f'{nm}_{fq}Hz_U'] = res['U']
            out[f'{nm}_{fq}Hz_kernel_sum'] = float(res['Kcoda'].sum())
            # finite-difference check of the kernel for the top 100 m
            zb = res['zb']

            class Pert:
                def __init__(s, eps):
                    s.eps = eps

                def __call__(s, z):
                    vp, vs, rho = vm(z)
                    m = np.asarray(z) < 0.1
                    return vp * np.where(m, 1 + s.eps, 1), vs * np.where(m, 1 + s.eps, 1), rho
            cp = fn_(Pert(1e-3), fq, zb=zb)['c']
            cm = fn_(Pert(-1e-3), fq, zb=zb)['c']
            out[f'{nm}_{fq}Hz_dlnc_top100m_fd'] = (np.log(cp) - np.log(cm)) / 2e-3
            out[f'{nm}_{fq}Hz_dlnc_top100m_kernel'] = float(res['Kc'][res['zc'] < 0.1].sum())
    return out


if __name__ == '__main__':
    import json
    import time
    t0 = time.time()
    print(json.dumps(tests(), indent=1))
    print('time', time.time() - t0)
