#!/usr/bin/env python3
"""Sharper images of the LFE slip patches on the fault plane (along strike x down dip), validated on synthetics.

Frame: patch_tensors.shelly_frame(293, 40): s = along strike (+ toward az 293, WNW), d = down dip (+ deeper).
Units: metres. Image grid +-1000 m, 25 m pixels. Events with any other-family catalog LFE within +-10 s are
excluded everywhere (overlap pull, NOTES 2026-11-22).

Event sets
  H = events good in BOTH split-station halves (sd_x,sd_y<0.5 km, sd_z<1 km in A and B): M=(A+B)/2, D=(A-B)/2.
      If the half errors eA, eB are independent and eB is symmetric, M's error has exactly the distribution of D
      (model-free kernel), including any per-event heteroscedasticity that is independent of position.
  G = events good in the all-station solution (same cuts). Many more events and a smaller error than M, but the
      error is not observed directly; it is calibrated by the AxB moment identity  cov(G) = C_true + <lambda> Sigma_f0.

Image methods (all deconvolutions = Richardson-Lucy / EM on the grid, FFT convolutions, iteration count chosen by
2-fold cross-validated held-out likelihood, start from a flat image)
  raw_G   : Gaussian KDE (40 m) of the full-solution locations (what plot_cloud_density shows, sharper bandwidth)
  C_sel   : consistency subset: H events with Mahalanobis |D| below the median, KDE (40 m) of M
  A_emp   : M deconvolved with the empirical, symmetrised D kernel (homoscedastic, fully model-free)
  A_het   : M deconvolved with per-event kernels sqrt(lambda_i) L0 u, lambda_i=((qA^2+qB^2)/2/med)^p,
            q = formal in-plane sd; p and Sigma0=L0 L0^T fitted to D by maximum likelihood; u = empirical whitened,
            lambda-normalised D shapes (Gaussian kernels were tried first: the real errors have a sharper core and
            heavier tails, which drove the CV to 140 iterations and spurious sharpness; empirical shapes fix this)
  B_full  : G deconvolved with per-event kernels sqrt(lambda_i) Lf u, lambda_i=(q_f^2/med)^p (p from the halves),
            Lf Lf^T = Sigma_f0 = (cov(G) - C_true)/<lambda>, C_true = AxB covariance from H; ASSUMES the full-solution
            error has the same normalised shape u as the half-difference error.
  CV folds and the G bootstrap keep 1-h bursts together (events in bursts are spatially correlated).
Model-free moment diagnostic
  AxB excess kurtosis of the true distribution along s and d:
            E[x^4] = (E[a^3 b]+E[a b^3])/2 - 1.5 E[ab](var a + var b - 2E[ab]),  robust to a per-event error scale
            shared by the two halves.  Gaussian 0, uniform disc -1, uniform streak -1.2, two blobs ~-1.6.

Stages (python scripts/sci_patch_images.py STAGE):
  synth  : synthetic shapes passed through three families' fitted error models -> data/sci/patch_images/synth_*.csv
  real   : per-family images, bootstrap, Gaussian-null parametric p-values -> real_*.csv, images_real.npz
  time   : before/after M7 and early/late (pre-M7 halves) comparisons vs random contiguous-block nulls -> time_*.csv
  plot   : figures/sci_patch_images_*.png
Environment: NCPU (default 8), NREAL (synthetic realisations, 16), NNULL (20), NBOOT (16), NTNULL (40).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import ndimage

sys.path.insert(0, 'scripts')
from patch_tensors import shelly_frame  # noqa: E402

OUT = 'data/sci/patch_images'
FIG = 'figures'
M7 = pd.Timestamp('2024-12-05T18:44')
R2 = shelly_frame(293.0, 40.0)[:2]           # rows: along-strike, down-dip (E, N, Down)
L, PIX = 1000.0, 25.0
NG = int(round(2 * L / PIX)) + 1              # 81
NK = 2 * NG - 1                               # kernel grid 161
NF = 256                                      # FFT size >= NG+NK-1
GX = np.arange(NG) * PIX - L
KX = np.arange(NK) * PIX - (NG - 1) * PIX
WIN = L - PIX / 2
NCPU = int(os.environ.get('NCPU', 8))
NREAL = int(os.environ.get('NREAL', 16))
NNULL = int(os.environ.get('NNULL', 20))
NBOOT = int(os.environ.get('NBOOT', 16))
NTNULL = int(os.environ.get('NTNULL', 40))
CKPT = [3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140]
NCLASS = 5
OVL = os.environ.get('OVL', 'incl')           # 'incl' or 'excl' (drop events with other-family LFEs within +-10 s)
OUT = f'{OUT}/ovl_{OVL}'
FTAG = '' if OVL == 'incl' else f'_{OVL}'
METHODS = ['raw_G', 'C_sel', 'A_emp', 'A_het', 'B_full']
DECON = ['A_emp', 'A_het', 'B_full']


# =========================================================================================== FFT RL machinery
def kfft(kimg):
    pad = np.zeros((NF, NF))
    pad[:NK, :NK] = kimg
    return sfft.rfft2(pad, workers=1)


def conv(f, KF):
    pad = np.zeros((NF, NF))
    pad[:NG, :NG] = f
    out = sfft.irfft2(sfft.rfft2(pad, workers=1) * KF, s=(NF, NF), workers=1)
    return out[NG - 1:2 * NG - 1, NG - 1:2 * NG - 1]


def gauss_kernel(S):
    S = np.asarray(S, float)
    Si = np.linalg.inv(S)
    X, Y = np.meshgrid(KX, KX, indexing='ij')
    q = Si[0, 0] * X ** 2 + 2 * Si[0, 1] * X * Y + Si[1, 1] * Y ** 2
    k = np.exp(-0.5 * q)
    return k / k.sum()


def emp_kernel(Dxy, bw=20.0):
    """Symmetrised empirical kernel from D samples (N,2), lightly smoothed."""
    P = np.vstack([Dxy, -Dxy])
    edges = np.r_[KX - PIX / 2, KX[-1] + PIX / 2]
    h, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=[edges, edges])
    h = ndimage.gaussian_filter(h, bw / PIX)
    return h / h.sum()


def emp_shape_kernel(U, Lm, lam, bw0=20.0):
    """Kernel of e = sqrt(lam) Lm u, u = whitened, lambda-normalised D samples (empirical shape: sharp core and
    heavy tails of the real errors), symmetrised; smoothing 20 m or 15% of the kernel sd (slight under-deconvolution)."""
    X = np.sqrt(lam) * U @ Lm.T
    X = np.vstack([X, -X])
    e = np.r_[KX - PIX / 2, KX[-1] + PIX / 2]
    h, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[e, e])
    bw = max(bw0, 0.15 * np.sqrt(lam * np.mean(np.linalg.eigvalsh(Lm @ Lm.T))))
    h = ndimage.gaussian_filter(h, bw / PIX)
    return h / h.sum()


def binidx(x):
    return np.clip(np.round((x + L) / PIX).astype(int), 0, NG - 1)


def hist(x, w=None):
    i, j = binidx(x[:, 0]), binidx(x[:, 1])
    y = np.zeros((NG, NG))
    np.add.at(y, (i, j), 1.0 if w is None else w)
    return y


class Decon:
    def __init__(self, kernels):
        self.KF = [kfft(k) for k in kernels]
        one = np.ones((NG, NG))
        self.norm1 = [conv(one, K) for K in self.KF]

    def run(self, ys, nits, tests=None):
        n = [y.sum() for y in ys]
        den = sum(nc * nm for nc, nm in zip(n, self.norm1))
        den = np.maximum(den, 1e-300)
        f = np.full((NG, NG), 1.0 / NG ** 2)
        ll = {}
        for it in range(1, nits + 1):
            num = np.zeros_like(f)
            for y, K, nc in zip(ys, self.KF, n):
                if nc == 0:
                    continue
                pred = conv(f, K)
                num += conv(np.where(y > 0, y / np.maximum(pred, 1e-300), 0.0), K)
            f = np.maximum(f * num / den, 0)
            f /= f.sum()
            if tests is not None and it in CKPT:
                s = 0.0
                for yt, K in zip(tests, self.KF):
                    if yt.sum() == 0:
                        continue
                    p = np.maximum(conv(f, K), 1e-300)
                    s += (yt * np.log(p / p.sum())).sum()
                ll[it] = s
        return f, ll


def time_groups(t, gap=3600.0):
    """burst ids (events separated by <= gap s share an id); all-equal times -> one id per event."""
    t = np.asarray(t, float)
    if np.ptp(t) == 0:
        return np.arange(len(t))
    o = np.argsort(t)
    g = np.empty(len(t), int)
    g[o] = np.r_[0, np.cumsum(np.diff(t[o]) > gap)]
    return g


def decon_cv(kernels, cls, x, rng, kfix=None, groups=None):
    """cls: class index per event (into kernels); x: (N,2) positions. 2-fold CV with whole bursts (groups) kept
    in one fold. Returns image, k*."""
    D = Decon(kernels)
    nc = len(kernels)
    if kfix is None:
        groups = np.arange(len(x)) if groups is None else groups
        ug = rng.permutation(np.unique(groups))
        inA = np.isin(groups, ug[::2])
        folds = [np.where(inA)[0], np.where(~inA)[0]]
        tot = {}
        for a in range(2):
            tr, te = folds[a], folds[1 - a]
            ys = [hist(x[tr][cls[tr] == c]) for c in range(nc)]
            yt = [hist(x[te][cls[te] == c]) for c in range(nc)]
            _, ll = D.run(ys, CKPT[-1], yt)
            for k, v in ll.items():
                tot[k] = tot.get(k, 0) + v
        kfix = max(tot, key=tot.get)
    ys = [hist(x[cls == c]) for c in range(nc)]
    f, _ = D.run(ys, kfix)
    return f, kfix


def kde_img(x, bw=40.0):
    h = hist(x)
    h = ndimage.gaussian_filter(h, bw / PIX)
    return h / h.sum()


# =========================================================================================== image statistics
def img_stats(f, prom=0.1, hmin=0.2):
    f = f / f.max()
    X, Y = np.meshgrid(GX, GX, indexing='ij')
    w = f / f.sum()
    cs, cd = (w * X).sum(), (w * Y).sum()
    C = np.array([[(w * (X - cs) ** 2).sum(), (w * (X - cs) * (Y - cd)).sum()],
                  [0, (w * (Y - cd) ** 2).sum()]])
    C[1, 0] = C[0, 1]
    # prominent peaks via superlevel-set components
    npk, pk = 1, []
    for t in np.arange(0.05, 0.95, 0.025):
        lab, n = ndimage.label(f >= t)
        if n < 2:
            continue
        mx = ndimage.maximum(f, lab, index=np.arange(1, n + 1))
        good = [i for i in range(n) if mx[i] >= max(t + prom, hmin)]
        if len(good) > npk:
            npk = len(good)
            pk = [np.unravel_index(np.argmax(np.where(lab == i + 1, f, -1)), f.shape) for i in good]
    sep = 0.0
    if len(pk) >= 2:
        P = np.array([[GX[i], GX[j]] for i, j in pk])
        sep = float(np.max(np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))))
    i0, j0 = binidx(np.array([cs])), binidx(np.array([cd]))
    dip = 1.0 - f[i0[0], j0[0]]
    # non-Gaussianity: L1 between image and the Gaussian with its moments
    Ci = np.linalg.inv(C + np.eye(2) * 1e-6)
    g = np.exp(-0.5 * (Ci[0, 0] * (X - cs) ** 2 + 2 * Ci[0, 1] * (X - cs) * (Y - cd) + Ci[1, 1] * (Y - cd) ** 2))
    g /= g.sum()
    nong = 0.5 * np.abs(w - g).sum()
    ev = np.linalg.eigvalsh(C)
    return dict(npk=npk, sep=sep, dip=dip, nong=nong, std_s=np.sqrt(C[0, 0]), std_d=np.sqrt(C[1, 1]),
                aspect=np.sqrt(ev[1] / max(ev[0], 1e-9)))


def fidelity(f, truth):
    return 1 - 0.5 * np.abs(f / f.sum() - truth / truth.sum()).sum()


# =========================================================================================== data
def tsec(x):
    return (pd.to_datetime(x, format='mixed') - pd.Timestamp('1970-01-01')).dt.total_seconds().values


_CAT = None


def catalog():
    global _CAT
    if _CAT is None:
        d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
        t = (pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) - pd.Timestamp('1970-01-01')).dt.total_seconds() + d.s_of_day
        o = np.argsort(t.values)
        _CAT = (t.values[o], d.idnum.values[o])
    return _CAT


def n_overlap(F, t):
    ct, cf = catalog()
    own = np.sort(ct[cf == F])
    j = np.clip(np.searchsorted(own, t), 1, len(own) - 1)
    j = np.where(np.abs(own[j - 1] - t) < np.abs(own[j] - t), j - 1, j)
    tc = own[j]
    oth = ct[cf != F]                                    # OTHER-family catalog LFEs within +-10 s
    return np.searchsorted(oth, tc + 10, side='right') - np.searchsorted(oth, tc - 10)


def inplane_q(df, suf=''):
    """formal in-plane sd (m): sqrt(var_s + var_d) of diag(sx^2, sy^2, sz^2) rotated."""
    v = np.vstack([df[f'sd_x{suf}'] ** 2, df[f'sd_y{suf}'] ** 2, df[f'sd_z{suf}'] ** 2]).T * 1e6
    return np.sqrt(v @ (R2[0] ** 2) + v @ (R2[1] ** 2))


def load_family(F):
    E = pd.read_csv(f'data/relocations/fam{F}_events_3d.csv')
    E['t'] = tsec(E.time_utc)
    E['nov'] = n_overlap(F, E.t.values)
    good = (E.sd_x < 0.5) & (E.sd_y < 0.5) & (E.sd_z < 1.0)
    G = E[good].copy()
    xg = G[['dx', 'dy', 'dz']].values * 1000 @ R2.T
    A = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfA.csv')
    B = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfB.csv')
    m = A.merge(B.drop(columns='time_utc'), on='ev', suffixes=('_a', '_b')).merge(E[['ev', 't', 'nov']], on='ev')
    m = m[(m.sd_x_a < 0.5) & (m.sd_y_a < 0.5) & (m.sd_z_a < 1) & (m.sd_x_b < 0.5) & (m.sd_y_b < 0.5) & (m.sd_z_b < 1)].copy()
    xa = m[['dx_a', 'dy_a', 'dz_a']].values * 1000 @ R2.T
    xb = m[['dx_b', 'dy_b', 'dz_b']].values * 1000 @ R2.T
    return dict(F=F, xg=xg, qg=inplane_q(G), tg=G.t.values, novg=G.nov.values, xa=xa, xb=xb,
                qa=inplane_q(m, '_a'), qb=inplane_q(m, '_b'), th=m.t.values, novh=m.nov.values)


# =========================================================================================== pipeline
def axb_cov(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return (a.T @ b + b.T @ a) / (2 * len(a))


def psd(C, floor=0.0):
    w, V = np.linalg.eigh(C)
    return (V * np.maximum(w, floor)) @ V.T


def axb_kurt(a, b):
    a, b = a - a.mean(), b - b.mean()
    m2 = np.mean(a * b)
    x4 = 0.5 * (np.mean(a ** 3 * b) + np.mean(a * b ** 3)) - 1.5 * m2 * (np.var(a) + np.var(b) - 2 * m2)
    return x4 / m2 ** 2 - 3 if m2 > 0 else np.nan


def fit_scale(D, mu):
    """D_i ~ N(0, (mu_i/med)^p Sigma0). ML over p grid. Returns p, Sigma0, tau."""
    best = None
    r = mu / np.median(mu)
    for p in np.arange(0, 1.51, 0.125):
        lam = r ** p
        S0 = (D.T * (1 / lam)) @ D / len(D)
        Si = np.linalg.inv(S0)
        w = np.einsum('ij,jk,ik->i', D, Si, D) / lam
        ll = -0.5 * np.sum(np.log(np.linalg.det(S0) * lam ** 2) + w)
        if best is None or ll > best[0]:
            best = (ll, p, S0, w)
    _, p, S0, w = best
    tau = np.sqrt(max(np.log(np.mean(w ** 2) / (2 * np.mean(w) ** 2)) / 4, 0.0))
    return p, S0, tau


def classes(lam):
    qs = np.quantile(lam, np.linspace(0, 1, NCLASS + 1))
    c = np.clip(np.searchsorted(qs[1:-1], lam, side='right'), 0, NCLASS - 1)
    lc = np.array([np.median(lam[c == k]) if np.any(c == k) else 1.0 for k in range(NCLASS)])
    return c, lc


def prepare(dat):
    """Centre, window, fit the error models. Returns dict used by images().
    Calibration (C_true, p, Sigma0, tau, D kernel) uses ALL H events; the imaged sets (M of H, G) drop events with
    other-family overlaps when OVL=excl."""
    xa, xb, xg = dat['xa'].copy(), dat['xb'].copy(), dat['xg'].copy()
    xa -= np.median(xa, 0)
    xb -= np.median(xb, 0)
    M, D = (xa + xb) / 2, (xa - xb) / 2
    okH = np.all(np.abs(M) < WIN, 1) & np.all(np.abs(D) < WIN, 1)
    xa, xb, M, D = xa[okH], xb[okH], M[okH], D[okH]
    novh = dat['novh'][okH]
    muH = (dat['qa'][okH] ** 2 + dat['qb'][okH] ** 2) / 2
    p, S0, tau = fit_scale(D, muH)
    Ct = psd(axb_cov(xa, xb))
    lamH = (muH / np.median(muH)) ** p
    imH = (novh == 0) if OVL == 'excl' else np.ones(len(M), bool)
    selG = (dat['novg'] == 0) if OVL == 'excl' else np.ones(len(xg), bool)
    xg, qg, tg = xg[selG], dat['qg'][selG], dat['tg'][selG]
    xg -= np.median(xg, 0)
    okG = np.all(np.abs(xg) < WIN, 1)
    xg_, qg = xg[okG], qg[okG]
    lamG = (qg ** 2 / np.median(qg ** 2)) ** p
    Sf0 = psd((np.cov(xg_.T) - Ct) / lamG.mean(), floor=30.0 ** 2)
    L0, Lf = np.linalg.cholesky(S0), np.linalg.cholesky(Sf0)
    U = np.linalg.solve(L0, D.T).T / np.sqrt(lamH)[:, None]      # whitened, lambda-normalised error shape samples
    tHo = dat['th'][okH]
    return dict(U=U, L0=L0, Lf=Lf, gG=time_groups(tg[okG]), gHimg=time_groups(tHo[imH]),xa=xa, xb=xb, M=M, D=D, muH=muH, lamH=lamH, novh=novh, Mimg=M[imH] - np.median(M[imH], 0),
                lamHimg=lamH[imH], p=p, S0=S0, tau=tau, Ct=Ct, xg=xg_, qg=qg, lamG=lamG,
                Sf0=Sf0, tH=dat['th'][okH], tG=tg[okG])


def images(P, rng, kfix=None, methods=METHODS):
    """kfix: dict method->iterations (skip CV)."""
    out, ks = {}, {}
    kfix = kfix or {}
    if 'raw_G' in methods:
        out['raw_G'] = kde_img(P['xg'])
    if 'C_sel' in methods:
        Si = np.linalg.inv(np.cov(P['D'].T))
        r = np.einsum('ij,jk,ik->i', P['D'], Si, P['D'])
        out['C_sel'] = kde_img(P['M'][r < np.median(r)])        # consistency subset of ALL H events
    if 'A_emp' in methods:
        out['A_emp'], ks['A_emp'] = decon_cv([emp_kernel(P['D'])], np.zeros(len(P['Mimg']), int), P['Mimg'], rng,
                                             kfix.get('A_emp'), P['gHimg'])
    if 'A_het' in methods:
        c, lc = classes(P['lamHimg'])
        out['A_het'], ks['A_het'] = decon_cv([emp_shape_kernel(P['U'], P['L0'], l) for l in lc], c, P['Mimg'], rng,
                                             kfix.get('A_het'), P['gHimg'])
    if 'B_full' in methods:
        c, lc = classes(P['lamG'])
        out['B_full'], ks['B_full'] = decon_cv([emp_shape_kernel(P['U'], P['Lf'], l) for l in lc], c, P['xg'], rng,
                                               kfix.get('B_full'), P['gG'])
    return out, ks


def moment_diag(P):
    a, b = P['xa'], P['xb']
    return dict(kurt_s=axb_kurt(a[:, 0], b[:, 0]), kurt_d=axb_kurt(a[:, 1], b[:, 1]),
                ct_s=np.sqrt(P['Ct'][0, 0]), ct_d=np.sqrt(P['Ct'][1, 1]))


# =========================================================================================== synthetics
SHAPES = ['gauss', 'disc', 'blobs150', 'blobs200', 'blobs300', 'blobs400', 'blobs500', 'ring', 'streak', 'gradient',
          'core_halo']


def truth_sample(shape, n, rng):
    if shape == 'gauss':
        return rng.normal(size=(n, 2)) * [150, 100]
    if shape in ('disc', 'gradient'):
        r = 280 * np.sqrt(rng.uniform(size=4 * n))
        a = rng.uniform(0, 2 * np.pi, 4 * n)
        x = np.c_[r * np.cos(a), r * np.sin(a)]
        if shape == 'gradient':           # density rises 1 -> 4 from -s edge to +s edge
            keep = rng.uniform(size=len(x)) < (1 + 3 * (x[:, 0] + 280) / 560) / 4
            x = x[keep]
        return x[:n]
    if shape.startswith('blobs'):
        sep = float(shape[5:])
        sgn = rng.choice([-1, 1], n)
        return rng.normal(size=(n, 2)) * 50 + np.c_[sgn * sep / 2, np.zeros(n)]
    if shape == 'ring':
        a = rng.uniform(0, 2 * np.pi, n)
        r = 250 + rng.normal(size=n) * 30
        return np.c_[r * np.cos(a), r * np.sin(a)]
    if shape == 'streak':
        return np.c_[rng.uniform(-300, 300, n), rng.normal(size=n) * 30]
    if shape == 'core_halo':               # 40% tight core (60 m) + 60% broad disc 300 m
        k = rng.uniform(size=n) < 0.4
        x = rng.normal(size=(n, 2)) * 60
        r = 300 * np.sqrt(rng.uniform(size=n))
        a = rng.uniform(0, 2 * np.pi, n)
        x[~k] = np.c_[r * np.cos(a), r * np.sin(a)][~k]
        return x
    if shape.startswith('fam'):           # Gaussian with a given covariance: 'famC:<c00>,<c01>,<c11>'
        c = [float(v) for v in shape.split(':')[1].split(',')]
        C = np.array([[c[0], c[1]], [c[1], c[2]]])
        return rng.multivariate_normal([0, 0], C, n)
    raise ValueError(shape)


def truth_image(shape):
    rng = np.random.default_rng(0)
    return hist(truth_sample(shape, 400000, rng))


def synth_data(P, shape, rng, nH=None, nG=None):
    """Synthetic data with the family's fitted error model (P from prepare(real))."""
    nH = nH or len(P['M'])
    nG = nG or len(P['xg'])
    # errors = sqrt(lambda) * L * u, u resampled (random sign) from the real whitened D shapes (heavy tails kept);
    # the two halves of an event share lambda (per-event heteroscedasticity)
    U = P['U']

    def u(n):
        return U[rng.integers(0, len(U), n)] * rng.choice([-1.0, 1.0], (n, 1))
    j = rng.integers(0, len(P['muH']), nH)
    lam = P['lamH'][j]
    x = truth_sample(shape, nH, rng)
    L2 = np.sqrt(2) * P['L0']
    xa = x + np.sqrt(lam)[:, None] * (u(nH) @ L2.T)
    xb = x + np.sqrt(lam)[:, None] * (u(nH) @ L2.T)
    jg = rng.integers(0, len(P['qg']), nG)
    lg = P['lamG'][jg]
    xg = truth_sample(shape, nG, rng) + np.sqrt(lg)[:, None] * (u(nG) @ P['Lf'].T)
    q = np.sqrt(P['muH'][j])
    novh = (np.arange(nH) >= len(P['Mimg'])).astype(int)     # imaged subset size as in the real data
    return dict(xa=xa, xb=xb, qa=q, qb=q, xg=xg, qg=P['qg'][jg], th=np.zeros(nH), tg=np.zeros(nG), novh=novh,
                novg=np.zeros(nG, int))


_PREP = {}


def real_prep(F):
    if F not in _PREP:
        _PREP[F] = prepare(load_family(F))
    return _PREP[F]


def synth_task(args):
    F, shape, r = args
    rng = np.random.default_rng(1000 * F + 97 * r + SHAPES.index(shape))
    P0 = real_prep(F)
    P = prepare(synth_data(P0, shape, rng))
    imgs, ks = images(P, rng)
    T = TRUTH[shape]
    rows = []
    for m, f in imgs.items():
        s = img_stats(f)
        s.update(F=F, shape=shape, real=r, method=m, k=ks.get(m, 0), fid=fidelity(f, T))
        rows.append(s)
    md = moment_diag(P)
    rows.append(dict(F=F, shape=shape, real=r, method='axb_moments', **md))
    keep = imgs if r == 0 else None
    return rows, (F, shape, keep)


TRUTH = {}


def stage_synth():
    fams = [int(v) for v in os.environ.get('SYNFAMS', '23,8,24').split(',')]
    for s in SHAPES:
        TRUTH[s] = truth_image(s)
    for F in fams:
        real_prep(F)
    tasks = [(F, s, r) for F in fams for s in SHAPES for r in range(NREAL)]
    rows, ex = [], {}
    with Pool(NCPU, initializer=os.nice, initargs=(5,)) as pool:
        for i, (rr, (F, s, keep)) in enumerate(pool.imap_unordered(synth_task, tasks, chunksize=1)):
            rows += rr
            if keep is not None:
                ex[(F, s)] = keep
            if i % 50 == 0:
                print(f'synth {i}/{len(tasks)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/synth_runs.csv', index=False)
    np.savez_compressed(f'{OUT}/synth_examples.npz', **{f'{F}_{s}_{m}': v for (F, s), d in ex.items() for m, v in d.items()},
                        **{f'truth_{s}': TRUTH[s] for s in SHAPES})
    summarize_synth(df)


def summarize_synth(df):
    im = df[df.method != 'axb_moments']
    g = im.groupby(['F', 'shape', 'method'])
    S = g.agg(n=('npk', 'size'), p_multi=('npk', lambda v: np.mean(v >= 2)), dip=('dip', 'median'),
              nong=('nong', 'median'), fid=('fid', 'median'), std_s=('std_s', 'median'), std_d=('std_d', 'median'),
              sep=('sep', 'median'), aspect=('aspect', 'median'), k=('k', 'median')).reset_index()
    # thresholds from the Gaussian null (95th percentile) per family and method
    thr = im[im['shape'] == 'gauss'].groupby(['F', 'method']).agg(dip95=('dip', lambda v: np.quantile(v, 0.95)),
                                                                  nong95=('nong', lambda v: np.quantile(v, 0.95)),
                                                                  multi0=('npk', lambda v: np.mean(v >= 2))).reset_index()
    im = im.merge(thr, on=['F', 'method'])
    det = im.assign(det_dip=im.dip > im.dip95, det_nong=im.nong > im.nong95).groupby(['F', 'shape', 'method']).agg(
        pow_dip=('det_dip', 'mean'), pow_nong=('det_nong', 'mean')).reset_index()
    S = S.merge(det, on=['F', 'shape', 'method']).merge(thr, on=['F', 'method'])
    mo = df[df.method == 'axb_moments'].groupby(['F', 'shape']).agg(
        kurt_s_med=('kurt_s', 'median'), kurt_s_lo=('kurt_s', lambda v: np.nanquantile(v, 0.16)),
        kurt_s_hi=('kurt_s', lambda v: np.nanquantile(v, 0.84)), kurt_d_med=('kurt_d', 'median'),
        ct_s=('ct_s', 'median'), ct_d=('ct_d', 'median')).reset_index()
    S.to_csv(f'{OUT}/synth_summary.csv', index=False)
    mo.to_csv(f'{OUT}/synth_moments.csv', index=False)
    print(S.to_string())
    print(mo.to_string())


# =========================================================================================== real data
def real_task(args):
    F, kind, r = args
    P0 = real_prep(F)
    rng = np.random.default_rng(7919 * F + 31 * r + (0 if kind == 'boot' else 5))
    if kind == 'main':
        imgs, ks = images(P0, rng)
        md = moment_diag(P0)
        st = {m: img_stats(f) for m, f in imgs.items()}
        return F, kind, r, imgs, ks, st, md
    if kind == 'boot':                      # resample events (H and G independently), refit everything
        Q = dict(P0)
        jh = rng.integers(0, len(P0['M']), len(P0['M']))
        ug = np.unique(P0['gG'])                                  # G: block bootstrap over 1-h bursts
        pick = rng.choice(ug, len(ug))
        idx = {g: np.where(P0['gG'] == g)[0] for g in ug} if 'gidx' not in P0 else P0['gidx']
        P0['gidx'] = idx
        jg = np.concatenate([idx[g] for g in pick])
        dat = dict(xa=P0['xa'][jh], xb=P0['xb'][jh], qa=np.sqrt(P0['muH'][jh]), qb=np.sqrt(P0['muH'][jh]),
                   xg=P0['xg'][jg], qg=P0['qg'][jg], th=P0['tH'][jh], tg=P0['tG'][jg], novh=P0['novh'][jh],
                   novg=np.zeros(len(jg), int))
        Q = prepare(dat)
        imgs, ks = images(Q, rng, methods=DECON)
        md = moment_diag(Q)
        return F, kind, r, imgs, ks, {m: img_stats(f) for m, f in imgs.items()}, md
    if kind == 'null':                      # Gaussian truth with the AxB covariance, same error model and N
        C = P0['Ct'] + np.eye(2) * 25.0
        sh = f'famC:{C[0, 0]},{C[0, 1]},{C[1, 1]}'
        Q = prepare(synth_data(P0, sh, rng))
        imgs, ks = images(Q, rng, methods=DECON)
        md = moment_diag(Q)
        return F, kind, r, None, ks, {m: img_stats(f) for m, f in imgs.items()}, md


def real_families():
    fams = []
    for F in range(1, 28):
        try:
            P = real_prep(F)
        except Exception as e:                      # too few events (e.g. families 4, 9: H < 15)
            print('skip family', F, repr(e))
            continue
        if len(P['M']) >= 80 and len(P['Mimg']) >= 40 and len(P['xg']) >= 300:
            fams.append(F)
    return fams


def stage_real():
    fams = real_families()
    print('families', fams, flush=True)
    tasks = [(F, 'main', 0) for F in fams] + [(F, 'boot', r) for F in fams for r in range(NBOOT)] + \
            [(F, 'null', r) for F in fams for r in range(NNULL)]
    rows, mains, boots = [], {}, {}
    with Pool(NCPU, initializer=os.nice, initargs=(5,)) as pool:
        for i, (F, kind, r, imgs, ks, st, md) in enumerate(pool.imap_unordered(real_task, tasks, chunksize=1)):
            for m, s in st.items():
                rows.append(dict(F=F, kind=kind, r=r, method=m, k=ks.get(m, 0), **s))
            rows.append(dict(F=F, kind=kind, r=r, method='axb_moments', **md))
            if kind == 'main':
                mains[F] = imgs
            if kind == 'boot':
                for m, f in imgs.items():
                    boots.setdefault((F, m), []).append(f / f.sum())
            if i % 25 == 0:
                print(f'real {i}/{len(tasks)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/real_runs.csv', index=False)
    np.savez_compressed(f'{OUT}/images_real.npz', **{f'{F}_{m}': v for F, d in mains.items() for m, v in d.items()},
                        **{f'{F}_{m}_bootmean': np.mean(v, 0) for (F, m), v in boots.items()},
                        **{f'{F}_{m}_bootsd': np.std(v, 0) for (F, m), v in boots.items()})
    meta = {}
    for F in fams:
        P = real_prep(F)
        meta[F] = dict(nH=len(P['M']), nG=len(P['xg']), p=P['p'], tau=P['tau'], S0=P['S0'].tolist(),
                       Ct=P['Ct'].tolist(), Sf0=P['Sf0'].tolist(), lamG_mean=float(P['lamG'].mean()))
    json.dump(meta, open(f'{OUT}/real_error_models.json', 'w'), indent=1)
    summarize_real(df)


def summarize_real(df):
    out = []
    for F, g in df.groupby('F'):
        mm = g[(g.kind == 'main') & (g.method == 'axb_moments')].iloc[0]
        bm = g[(g.kind == 'boot') & (g.method == 'axb_moments')]
        nm = g[(g.kind == 'null') & (g.method == 'axb_moments')]
        row = dict(F=F, kurt_s=mm.kurt_s, kurt_s_bootsd=bm.kurt_s.std(), kurt_d=mm.kurt_d, kurt_d_bootsd=bm.kurt_d.std(),
                   kurt_s_null_lo=nm.kurt_s.quantile(0.025), kurt_s_null_hi=nm.kurt_s.quantile(0.975),
                   kurt_d_null_lo=nm.kurt_d.quantile(0.025), kurt_d_null_hi=nm.kurt_d.quantile(0.975),
                   ct_s=mm.ct_s, ct_d=mm.ct_d)
        for m in METHODS:
            a = g[(g.kind == 'main') & (g.method == m)]
            if not len(a):
                continue
            a = a.iloc[0]
            row.update({f'{m}_npk': a.npk, f'{m}_sep': a.sep, f'{m}_dip': a.dip, f'{m}_nong': a.nong,
                        f'{m}_std_s': a.std_s, f'{m}_std_d': a.std_d, f'{m}_k': a.k})
            if m in DECON:
                b = g[(g.kind == 'boot') & (g.method == m)]
                n = g[(g.kind == 'null') & (g.method == m)]
                row.update({f'{m}_boot_multi': np.mean(b.npk >= 2), f'{m}_null_multi': np.mean(n.npk >= 2),
                            f'{m}_p_dip': (1 + np.sum(n.dip >= a.dip)) / (1 + len(n)),
                            f'{m}_p_nong': (1 + np.sum(n.nong >= a.nong)) / (1 + len(n))})
        out.append(row)
    S = pd.DataFrame(out)
    S.to_csv(f'{OUT}/real_summary.csv', index=False)
    print(S.round(3).to_string())


# =========================================================================================== time changes
def shape_mmd(x1, x2, bw=60.0, align=True):
    if align:
        x1, x2 = x1 - np.median(x1, 0), x2 - np.median(x2, 0)
    h1 = ndimage.gaussian_filter(hist(x1), bw / PIX)
    h2 = ndimage.gaussian_filter(hist(x2), bw / PIX)
    return 1e4 * np.sum((h1 / h1.sum() - h2 / h2.sum()) ** 2)


def group_image(P, sel, kB):
    """B_full image of a subset of G, kernel calibrated on that subset (cov(G_sub) - C_true)."""
    xg, lam = P['xg'][sel], P['lamG'][sel]
    xg = xg - np.median(xg, 0)
    Sf0 = psd((np.cov(xg.T) - P['Ct']) / lam.mean(), floor=30.0 ** 2)
    c, lc = classes(lam)
    D = Decon([emp_shape_kernel(P['U'], np.linalg.cholesky(Sf0), l) for l in lc])
    f, _ = D.run([hist(xg[c == k]) for k in range(NCLASS)], kB)
    return f / f.sum(), Sf0


def img_l1(f1, f2, align=True):
    if align:
        out = []
        for f in (f1, f2):
            X, Y = np.meshgrid(GX, GX, indexing='ij')
            cs, cd = (f * X).sum(), (f * Y).sum()
            out.append(ndimage.shift(f, (-cs / PIX, -cd / PIX), order=1, mode='constant'))
        f1, f2 = out
    return 0.5 * np.abs(f1 / f1.sum() - f2 / f2.sum()).sum()


def splits(P):
    t = P['tG']
    o = np.argsort(t)
    tm7 = (M7 - pd.Timestamp('1970-01-01')).total_seconds()
    post = t >= tm7
    pre_idx = np.where(~post)[0]
    tmed = np.median(t[pre_idx])
    early = (~post) & (t < tmed)
    late = (~post) & (t >= tmed)
    return dict(m7=(~post, post), earlylate=(early, late)), o


def time_task(args):
    F, which, r, kB = args
    P = real_prep(F)
    rng = np.random.default_rng(5003 * F + 17 * r + (0 if which == 'm7' else 1))
    sp, o = splits(P)
    g1, g2 = sp[which]
    if r > 0:                                  # random contiguous block of the same size
        if which == 'm7':
            pool_idx = o                       # all events, time-sorted
            n2 = g2.sum()
            st = rng.integers(0, len(pool_idx) - n2 + 1)
            blk = pool_idx[st:st + n2]
            g2 = np.zeros(len(o), bool); g2[blk] = True
            g1 = ~g2
        else:                                  # pre-M7 only, circular block of half size
            pre = o[~sp['m7'][1][o]]
            n2 = sp['earlylate'][1].sum()
            st = rng.integers(0, len(pre))
            blk = pre[(st + np.arange(n2)) % len(pre)]
            g2 = np.zeros(len(o), bool); g2[blk] = True
            g1 = np.zeros(len(o), bool); g1[pre] = True; g1 &= ~g2
    x1, x2 = P['xg'][g1], P['xg'][g2]
    f1, S1 = group_image(P, g1, kB)
    f2, S2 = group_image(P, g2, kB)
    st1, st2 = img_stats(f1), img_stats(f2)
    return dict(F=F, which=which, r=r, n1=g1.sum(), n2=g2.sum(),
                mmd_shape=shape_mmd(x1, x2), mmd_full=shape_mmd(x1, x2, align=False),
                dcen_s=np.median(x2[:, 0]) - np.median(x1[:, 0]), dcen_d=np.median(x2[:, 1]) - np.median(x1[:, 1]),
                l1_img=img_l1(f1, f2), q1=np.median(P['qg'][g1]), q2=np.median(P['qg'][g2]),
                sf1_s=np.sqrt(S1[0, 0]), sf2_s=np.sqrt(S2[0, 0]), sf1_d=np.sqrt(S1[1, 1]), sf2_d=np.sqrt(S2[1, 1]),
                std1_s=st1['std_s'], std2_s=st2['std_s'], std1_d=st1['std_d'], std2_d=st2['std_d'],
                npk1=st1['npk'], npk2=st2['npk']), ((F, which), (f1, f2)) if r == 0 else None


def axb_time(F, rng, nnull=200):
    """AxB C_true per group (model-free second moments), with contiguous-block nulls."""
    P = real_prep(F)
    t = P['tH']
    o = np.argsort(t)
    tm7 = (M7 - pd.Timestamp('1970-01-01')).total_seconds()
    post = t >= tm7
    rows = []
    tmed = np.median(t[~post])
    defs = dict(m7=(~post, post), earlylate=((~post) & (t < tmed), (~post) & (t >= tmed)))
    for which, (g1, g2) in defs.items():
        if g2.sum() < 25 or g1.sum() < 25:
            continue

        def stat(a, b):
            C1, C2 = axb_cov(P['xa'][a], P['xb'][a]), axb_cov(P['xa'][b], P['xb'][b])
            return np.trace(C2) - np.trace(C1), C2[0, 0] - C1[0, 0], C2[1, 1] - C1[1, 1]
        real = stat(g1, g2)
        nul = []
        for _ in range(nnull):
            if which == 'm7':
                n2 = g2.sum(); st = rng.integers(0, len(o) - n2 + 1); blk = o[st:st + n2]
                b = np.zeros(len(o), bool); b[blk] = True; a = ~b
            else:
                pre = o[~post[o]]; n2 = g2.sum(); st = rng.integers(0, len(pre))
                blk = pre[(st + np.arange(n2)) % len(pre)]
                b = np.zeros(len(o), bool); b[blk] = True; a = np.zeros(len(o), bool); a[pre] = True; a &= ~b
            nul.append(stat(a, b))
        nul = np.array(nul)
        rows.append(dict(F=F, which=which, n1=g1.sum(), n2=g2.sum(), dtr=real[0], dss=real[1], ddd=real[2],
                         p_tr=np.mean(np.abs(nul[:, 0]) >= abs(real[0])), p_ss=np.mean(np.abs(nul[:, 1]) >= abs(real[1])),
                         p_dd=np.mean(np.abs(nul[:, 2]) >= abs(real[2])), null_sd_tr=nul[:, 0].std()))
    return rows


def stage_time():
    fams = real_families()
    rr = pd.read_csv(f'{OUT}/real_runs.csv')
    kB = rr[(rr.kind == 'main') & (rr.method == 'B_full')].set_index('F').k.to_dict()
    tasks = []
    for F in fams:
        sp, _ = splits(real_prep(F))
        for which, (g1, g2) in sp.items():
            if g2.sum() >= 150 and g1.sum() >= 150:
                tasks += [(F, which, r, int(kB[F])) for r in range(NTNULL + 1)]
    rows, ex = [], {}
    with Pool(NCPU, initializer=os.nice, initargs=(5,)) as pool:
        for i, (row, e) in enumerate(pool.imap_unordered(time_task, tasks, chunksize=2)):
            rows.append(row)
            if e is not None:
                ex[e[0]] = e[1]
            if i % 100 == 0:
                print(f'time {i}/{len(tasks)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/time_runs.csv', index=False)
    np.savez_compressed(f'{OUT}/images_time.npz', **{f'{F}_{w}_{k}': f for (F, w), fs in ex.items() for k, f in zip('12', fs)})
    S = []
    for (F, w), g in df.groupby(['F', 'which']):
        a, n = g[g.r == 0].iloc[0], g[g.r > 0]
        S.append(dict(F=F, which=w, n1=a.n1, n2=a.n2, mmd_shape=a.mmd_shape, p_mmd_shape=(1 + np.sum(n.mmd_shape >= a.mmd_shape)) / (1 + len(n)),
                      mmd_full=a.mmd_full, p_mmd_full=(1 + np.sum(n.mmd_full >= a.mmd_full)) / (1 + len(n)),
                      l1_img=a.l1_img, p_l1_img=(1 + np.sum(n.l1_img >= a.l1_img)) / (1 + len(n)),
                      dcen_s=a.dcen_s, z_dcen_s=(a.dcen_s - n.dcen_s.mean()) / n.dcen_s.std(),
                      dcen_d=a.dcen_d, z_dcen_d=(a.dcen_d - n.dcen_d.mean()) / n.dcen_d.std(),
                      q1=a.q1, q2=a.q2, q_ratio_null_sd=(n.q2 / n.q1).std(),
                      std1_s=a.std1_s, std2_s=a.std2_s, std1_d=a.std1_d, std2_d=a.std2_d, npk1=a.npk1, npk2=a.npk2,
                      sf1_s=a.sf1_s, sf2_s=a.sf2_s, sf1_d=a.sf1_d, sf2_d=a.sf2_d))
    S = pd.DataFrame(S)
    rng = np.random.default_rng(3)
    ax = pd.DataFrame([r for F in fams for r in axb_time(F, rng)])
    ax.to_csv(f'{OUT}/time_axb.csv', index=False)
    S.to_csv(f'{OUT}/time_summary.csv', index=False)
    print(S.round(3).to_string())
    print(ax.round(3).to_string())


# =========================================================================================== plots
def stage_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    ext = [-L - PIX / 2, L + PIX / 2, L + PIX / 2, -L - PIX / 2]       # x = s, y = d (down-dip downward)
    lim = 700

    def show(ax, f, title=None, cont=True):
        f = f / f.max()
        ax.imshow(f.T, extent=ext, cmap='magma_r', vmin=0, vmax=1, interpolation='bilinear')
        if cont:
            ax.contour(GX, GX, f.T, levels=[0.25, 0.5, 0.75], colors='w', linewidths=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(lim, -lim)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        if title:
            ax.set_title(title, fontsize=7)

    # ---- synthetic examples + resolution summary (only where the synth stage was run)
    SYN = OUT if os.path.exists(f'{OUT}/synth_summary.csv') else OUT.replace(f'ovl_{OVL}', 'ovl_incl')
    if FTAG == '' and os.path.exists(f'{SYN}/synth_examples.npz'):
        ex = np.load(f'{SYN}/synth_examples.npz')
        S = pd.read_csv(f'{SYN}/synth_summary.csv')
        for F in sorted({int(k.split('_')[0]) for k in ex.files if not k.startswith('truth')}):
            shapes = [s for s in SHAPES]
            cols = ['truth'] + METHODS
            fig, axs = plt.subplots(len(shapes), len(cols), figsize=(1.9 * len(cols), 1.9 * len(shapes)))
            for i, s in enumerate(shapes):
                for j, m in enumerate(cols):
                    key = f'truth_{s}' if m == 'truth' else f'{F}_{s}_{m}'
                    ax = axs[i, j]
                    if key in ex.files:
                        f = ex[key]
                        if m == 'truth':
                            f = ndimage.gaussian_filter(f, 0.7)
                        t = ''
                        if m != 'truth':
                            r = S[(S.F == F) & (S['shape'] == s) & (S.method == m)]
                            if len(r):
                                r = r.iloc[0]
                                t = f'fid {r.fid:.2f} P2pk {r.p_multi:.2f}'
                        show(ax, f, (s if j == 0 else '') + (f'{m}\n' if i == 0 and j > 0 else '') + t)
                    else:
                        ax.axis('off')
            fig.suptitle(f'Synthetic patches through family {F} error model (one realisation; fid = median overlap with truth; '
                         f'P2pk = fraction of {NREAL} runs with >=2 prominent peaks).\nColumns: truth, {", ".join(METHODS)}. '
                         f'x: along strike (m, + WNW), y: down dip (m, down = deeper)', fontsize=8)
            fig.tight_layout(rect=[0, 0, 1, 0.975])
            fig.savefig(f'{FIG}/sci_patch_images{FTAG}_synthetic_fam{F}.png', dpi=130)
            plt.close(fig)

        # ---- resolution summary: two-blob separation sweep, ring/disc detection, fidelity
        fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
        sty = dict(raw_G=('0.5', '-'), C_sel=('tab:orange', '-'), A_emp=('tab:green', '-'), A_het=('tab:blue', '-'), B_full=('tab:red', '-'))
        ls = {23: '-', 8: '--', 24: ':'}
        for F in sorted(S.F.unique()):
            for m in METHODS:
                s = S[(S.F == F) & (S.method == m)].set_index('shape')
                seps = [150, 200, 300, 400, 500]
                axs[0].plot(seps, [s.loc[f'blobs{v}', 'p_multi'] for v in seps], color=sty[m][0], ls=ls.get(F, '-'),
                            marker='o', ms=3, label=f'{m} fam{F}' if F == 23 else None)
                axs[0].axhline(s.loc['gauss', 'p_multi'], color=sty[m][0], ls=ls.get(F, '-'), lw=0.5)
                axs[1].plot(seps, [s.loc[f'blobs{v}', 'sep'] for v in seps], color=sty[m][0], ls=ls.get(F, '-'), marker='o', ms=3)
                sh = ['gauss', 'disc', 'ring', 'streak', 'gradient', 'core_halo', 'blobs300']
                axs[2].plot(range(len(sh)), [s.loc[v, 'pow_dip'] for v in sh], color=sty[m][0], ls=ls.get(F, '-'), marker='o', ms=3)
                axs[3].plot(range(len(sh)), [s.loc[v, 'fid'] for v in sh], color=sty[m][0], ls=ls.get(F, '-'), marker='o', ms=3)
        axs[0].set_xlabel('true blob separation (m; blobs sigma 50 m)'); axs[0].set_ylabel('fraction with >=2 prominent peaks')
        axs[0].legend(fontsize=6); axs[0].set_title('Two-blob detection (thin lines: Gaussian-null false rate)', fontsize=8)
        axs[1].plot([150, 500], [150, 500], 'k:', lw=0.8)
        axs[1].set_xlabel('true separation (m)'); axs[1].set_ylabel('recovered peak separation (median, m)')
        for a in axs[2:]:
            a.set_xticks(range(len(sh))); a.set_xticklabels(sh, rotation=40, fontsize=7)
        axs[2].set_ylabel('central-dip detection power (5% FPR)'); axs[2].set_title('Ring / central-depression detection', fontsize=8)
        axs[3].set_ylabel('image fidelity (overlap with truth)')
        fig.suptitle('Synthetic resolution tests (solid fam 23, dashed fam 8, dotted fam 24 error models)', fontsize=9)
        fig.tight_layout()
        fig.savefig(f'{FIG}/sci_patch_images{FTAG}_resolution.png', dpi=130)
        plt.close(fig)

    # ---- real families
    im = np.load(f'{OUT}/images_real.npz')
    RS = pd.read_csv(f'{OUT}/real_summary.csv').set_index('F')
    em = json.load(open(f'{OUT}/real_error_models.json'))
    fams = list(RS.index)
    for best, tag in [('B_full', 'B_full'), ('A_het', 'A_het')]:
        nc = 7
        nr = int(np.ceil(len(fams) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.7 * nr))
        for ax in axs.flat:
            ax.axis('off')
        for ax, F in zip(axs.flat, fams):
            ax.axis('on')
            f = im[f'{F}_{best}']
            r = RS.loc[F]
            show(ax, f, f'fam {F} nG={em[str(F)]["nG"]} nH={em[str(F)]["nH"]}\n'
                        f'pk={int(r[f"{best}_npk"])} boot2pk={r[f"{best}_boot_multi"]:.2f} null2pk={r[f"{best}_null_multi"]:.2f}\n'
                        f'p_dip={r[f"{best}_p_dip"]:.2f} p_nonG={r[f"{best}_p_nong"]:.2f} k_s={r.kurt_s:+.1f}+/-{r.kurt_s_bootsd:.1f}')
            other = 'A_het' if best == 'B_full' else 'B_full'
            g = im[f'{F}_{other}']
            ax.contour(GX, GX, (g / g.max()).T, levels=[0.5], colors='c', linewidths=0.8)
            C = np.array(em[str(F)]['Ct'])
            w, V = np.linalg.eigh(C)
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(max(w[1], 0)), 2 * np.sqrt(max(w[0], 0)), angle=-ang, fill=False, ec='lime', lw=0.8, ls='--'))
        fig.suptitle(f'Deconvolved patch images ({best}; cyan = 50% contour of {"A_het" if best == "B_full" else "B_full"}; '
                     f'green dashed = AxB 1-sigma ellipse). x along strike (m, + WNW), y down dip (m, down = deeper).',
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(f'{FIG}/sci_patch_images{FTAG}_families_{tag}.png', dpi=130)
        plt.close(fig)

    # ---- AxB kurtosis (model-free): real families vs Gaussian null band and synthetic shape references
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.8))
    x = np.arange(len(fams))
    ax.fill_between(x, RS['kurt_s_null_lo'], RS['kurt_s_null_hi'], color='0.85', label='Gaussian-truth null 95% (same errors, N)')
    ax.errorbar(x, RS['kurt_s'], yerr=RS['kurt_s_bootsd'], fmt='o', color='k', ms=4, label='AxB estimate +/- event-bootstrap sd')
    for v, lab, col in [(-1.0, 'uniform disc (-1.0)', 'tab:blue'), (-1.2, 'uniform streak (-1.2)', 'tab:green'),
                        (-1.62, 'two blobs 300 m, sigma 50 (-1.6)', 'tab:red')]:
        ax.axhline(v, color=col, lw=0.8, ls='--', label=lab)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=7)
    ax.set_xlabel('family'); ax.set_ylabel('excess kurtosis (Gaussian 0)')
    ax.set_title('Model-free AxB excess kurtosis of the true along-strike distribution (analytic shape values dashed)', fontsize=9)
    ax.set_ylim(-4, 6)
    ax.legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_patch_images{FTAG}_kurtosis.png', dpi=130)
    plt.close(fig)

    # ---- time comparisons
    if os.path.exists(f'{OUT}/time_summary.csv'):
        T = pd.read_csv(f'{OUT}/time_summary.csv')
        it = np.load(f'{OUT}/images_time.npz')
        for which, lab in [('m7', ('pre-M7', 'post-M7')), ('earlylate', ('early pre-M7', 'late pre-M7'))]:
            t = T[T.which == which]
            if not len(t):
                continue
            fl = list(t.F)
            nc = 6
            nr = int(np.ceil(len(fl) / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.8 * nr))
            for ax in np.atleast_1d(axs).flat:
                ax.axis('off')
            for ax, F in zip(np.atleast_1d(axs).flat, fl):
                ax.axis('on')
                r = t[t.F == F].iloc[0]
                f1, f2 = it[f'{F}_{which}_1'], it[f'{F}_{which}_2']
                show(ax, f1, f'fam {F}: n {int(r.n1)}/{int(r.n2)}\np_img={r.p_l1_img:.2f} p_shape={r.p_mmd_shape:.2f}\n'
                             f'dcen s {r.dcen_s:+.0f} ({r.z_dcen_s:+.1f}z) d {r.dcen_d:+.0f} ({r.z_dcen_d:+.1f}z)', cont=False)
                ax.contour(GX, GX, (f1 / f1.max()).T, levels=[0.25, 0.5, 0.75], colors='w', linewidths=0.6)
                ax.contour(GX, GX, (f2 / f2.max()).T, levels=[0.25, 0.5, 0.75], colors='c', linewidths=0.8)
            fig.suptitle(f'{lab[0]} (image, white contours) vs {lab[1]} (cyan contours); B_full deconvolution with each group\'s '
                         f'own calibrated kernel; p from {NTNULL} random contiguous blocks of the same size', fontsize=9)
            fig.tight_layout()
            fig.savefig(f'{FIG}/sci_patch_images{FTAG}_time_{which}.png', dpi=130)
            plt.close(fig)


# =========================================================================================== consistency-selection bias
def stage_selbias(nrep=60):
    """Simulate approach 1 (keep the 50% of H events with the smallest Mahalanobis |A-B|): does it bias the truth,
    the AxB covariance, or the apparent error?  Gaussian truth with each family's C_true; heteroscedastic errors."""
    rows = []
    for F in [int(v) for v in os.environ.get('SYNFAMS', '23,8,24').split(',')]:
        P0 = real_prep(F)
        C = P0['Ct'] + np.eye(2) * 25.0
        sh = f'famC:{C[0, 0]},{C[0, 1]},{C[1, 1]}'
        for kind in ['synthetic'] * nrep + ['real']:
            rng = np.random.default_rng(len(rows))
            if kind == 'real':
                xa, xb = P0['xa'], P0['xb']
                xt = None
            else:
                d = synth_data(P0, sh, rng)
                xa, xb = d['xa'], d['xb']
            M, D = (xa + xb) / 2, (xa - xb) / 2
            Si = np.linalg.inv(np.cov(D.T))
            r = np.einsum('ij,jk,ik->i', D, Si, D)
            sel = r < np.median(r)
            Ca, Cs = axb_cov(xa, xb), axb_cov(xa[sel], xb[sel])
            row = dict(F=F, kind=kind, true_s=np.sqrt(C[0, 0]), true_d=np.sqrt(C[1, 1]),
                       ct_all_s=np.sqrt(max(Ca[0, 0], 0)), ct_all_d=np.sqrt(max(Ca[1, 1], 0)),
                       ct_sel_s=np.sqrt(max(Cs[0, 0], 0)), ct_sel_d=np.sqrt(max(Cs[1, 1], 0)),
                       stdM_all_s=M[:, 0].std(), stdM_sel_s=M[sel, 0].std(), stdM_all_d=M[:, 1].std(), stdM_sel_d=M[sel, 1].std(),
                       stdD_sel_s=D[sel, 0].std(), stdD_sel_d=D[sel, 1].std())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/selection_bias_runs.csv', index=False)
    S = df.groupby(['F', 'kind']).median().reset_index()
    # implied error of M in the selected subset = stdM_sel^2 - true^2 (synthetic: truth known)
    S['errM_sel_s'] = np.sqrt(np.maximum(S.stdM_sel_s ** 2 - S.true_s ** 2, 0))
    S['errM_sel_d'] = np.sqrt(np.maximum(S.stdM_sel_d ** 2 - S.true_d ** 2, 0))
    S.to_csv(f'{OUT}/selection_bias.csv', index=False)
    print(S.round(1).to_string())


# =========================================================================================== population combination
def stage_combine():
    """Across families: are deconvolved patches more structured (central dip / non-Gaussian) than the per-family
    Gaussian-truth null?  Per family rank p (one-sided) and z vs the null; Stouffer combination; binomial count of
    p<=0.05 (floor 1/(NNULL+1)).  Run for each OVL variant present."""
    from scipy import stats as sst
    rows = []
    base = OUT.rsplit('/', 1)[0]
    for var in ['incl', 'excl']:
        f = f'{base}/ovl_{var}/real_runs.csv'
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['kind'] = df['kind'].fillna('null')          # pandas reads the string 'null' as NaN
        for m in DECON + ['axb_moments']:
            for stat in (['kurt_s'] if m == 'axb_moments' else ['dip', 'nong']):
                zs, ps, fl = [], [], []
                for F, g in df.groupby('F'):
                    a = g[(g.kind == 'main') & (g.method == m)]
                    n = g[(g.kind == 'null') & (g.method == m)][stat].dropna()
                    if not len(a) or len(n) < 5 or not np.isfinite(a[stat].iloc[0]):
                        continue
                    v = a[stat].iloc[0]
                    sgn = -1 if stat == 'kurt_s' else 1          # platykurtic = flat-topped / multi-lobed
                    pr = (1 + np.sum(sgn * n >= sgn * v)) / (1 + len(n))
                    mad = 1.4826 * np.median(np.abs(n - n.median()))
                    zs.append(sgn * (v - n.median()) / max(mad, 1e-9)); ps.append(pr); fl.append(F)
                zs, ps = np.clip(np.array(zs), -10, 10), np.array(ps)
                k = int(np.sum(ps <= 0.05 + 1e-9))
                pfloor = 1.0 / (NNULL + 1)
                rows.append(dict(variant=var, method=m, stat=stat, nfam=len(zs), stouffer_z=zs.sum() / np.sqrt(len(zs)),
                                 n_p05=k, binom_p=sst.binom.sf(k - 1, len(zs), max(0.05, pfloor)),
                                 median_rank_p=np.median(ps), fams_p05=' '.join(str(f) for f, p in zip(fl, ps) if p <= 0.05 + 1e-9)))
    S = pd.DataFrame(rows)
    S.to_csv(f'{base}/population_structure.csv', index=False)
    print(S.round(4).to_string())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('synth', 'all'):
        stage_synth()
    if stage in ('real', 'all'):
        stage_real()
    if stage in ('time', 'all'):
        stage_time()
    if stage in ('combine', 'all'):
        stage_combine()
    if stage in ('selbias', 'all'):
        stage_selbias()
    if stage in ('plot', 'all'):
        stage_plot()
