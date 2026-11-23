#!/usr/bin/env python3
"""Dose-response of the MTJ LFE families to the 13 M>=5.5 earthquakes (2010-2024).

Part A  Okada (1992) DC3D displacement (numpy port, displacement terms only) for a finite
        rectangular dislocation in an elastic half-space; strains by 4th-order central finite
        differences of the displacement field.  Validated against (i) the 2-D buried screw
        dislocation (analytic), (ii) displacement jump = slip vector across the fault,
        (iii) traction-free surface, (iv) static equilibrium, (v) stress drop on the source.
        Together (ii)-(iv) + decay at infinity uniquely determine the elastic solution.
Part B  Source models for each quake (USGS FFM for 2022 Ferndale and 2024 M7; otherwise
        Wells & Coppersmith 1994 uniform-slip rectangles on USGS moment-tensor nodal planes),
        static stress at the 27 family centroids resolved on the LFE fault (strike 293,
        dip 40 NE, right-lateral), undrained pore pressure, BSSA14 PGV.
Part C  Dose-response statistics (per quake, n=12 independent dates) and within-M7 per-family test.

Sign conventions: stress tension-positive.  dsn > 0 = unclamping.  dtau > 0 = promotes slip in
the receiver rake direction.  dP = -tr(dsigma)/3 (compression positive).  Undrained pore
pressure dp = B*dP = -B*tr(dsigma)/3.  Effective normal-stress change, unclamping positive:
dsn_eff = dsn + dp  (equivalently -(d sigma_n^compr - dp)).  dCFS = dtau + mu'*dsn (effective
friction form); dCFS_ud = dtau + mu*(dsn + dp) (isotropic undrained form).

Outputs: data/sci/dose_response/*, figures/sci_dose_response_*.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dose_response'
RAW = f'{OUT}/usgs_raw'
FIG = f'{ROOT}/figures'
os.makedirs(OUT, exist_ok=True)

MU = 30e9          # Pa
NU = 0.25
LAM = 2 * MU * NU / (1 - 2 * NU)
ALPHA = (LAM + MU) / (LAM + 2 * MU)   # = 2/3 for nu = 0.25

# ----------------------------------------------------------------------------------------
# Part A. Okada (1992) DC3D, displacement only
# ----------------------------------------------------------------------------------------
EPS = 1e-6


def _dccon2(xi, et, q, sd, cd, kxi, ket):
    xi = np.where(np.abs(xi) < EPS, 0.0, xi)
    et = np.where(np.abs(et) < EPS, 0.0, et)
    q = np.where(np.abs(q) < EPS, 0.0, q)
    xi2, et2, q2 = xi * xi, et * et, q * q
    r2 = xi2 + et2 + q2
    r = np.sqrt(r2)
    r3 = r * r2
    y = et * cd + q * sd
    d = et * sd - q * cd
    with np.errstate(divide='ignore', invalid='ignore'):
        tt = np.where(q == 0, 0.0, np.arctan(xi * et / (q * r)))
        rxi = r + xi
        alx = np.where(kxi, -np.log(r - xi), np.log(rxi))
        x11 = np.where(kxi, 0.0, 1.0 / (r * rxi))
        x32 = np.where(kxi, 0.0, (r + rxi) * x11 * x11 / r)
        ret = r + et
        ale = np.where(ket, -np.log(r - et), np.log(ret))
        y11 = np.where(ket, 0.0, 1.0 / (r * ret))
        y32 = np.where(ket, 0.0, (r + ret) * y11 * y11 / r)
    return dict(xi=xi, et=et, q=q, xi2=xi2, et2=et2, q2=q2, r=r, r2=r2, r3=r3, r5=r3 * r2,
                y=y, d=d, tt=tt, alx=alx, x11=x11, x32=x32, ale=ale, y11=y11, y32=y32)


def _ua(c, d1, d2, d3, a1, a2):
    xi, q, et, r, tt = c['xi'], c['q'], c['et'], c['r'], c['tt']
    qx, qy = q * c['x11'], q * c['y11']
    u = np.zeros((3,) + np.shape(xi))
    if d1:
        u += d1 * np.array([tt / 2 + a2 * xi * qy, a2 * q / r, a1 * c['ale'] - a2 * q * qy])
    if d2:
        u += d2 * np.array([a2 * q / r, tt / 2 + a2 * et * qx, a1 * c['alx'] - a2 * q * qx])
    if d3:
        u += d3 * np.array([-a1 * c['ale'] - a2 * q * qy, -a1 * c['alx'] - a2 * q * qx,
                            tt / 2 - a2 * (et * qx + xi * qy)])
    return u / (2 * np.pi)


def _ub(c, d1, d2, d3, a3, sd, cd):
    xi, q, et, r, tt, y, d = c['xi'], c['q'], c['et'], c['r'], c['tt'], c['y'], c['d']
    xi2, q2 = c['xi2'], c['q2']
    rd = r + d
    d11 = 1.0 / (r * rd)
    aj2 = xi * y / rd * d11
    aj5 = -(d + y * y / rd) * d11
    if cd != 0:
        cdcd, sdcd = cd * cd, sd * cd
        x = np.sqrt(xi2 + q2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ai4 = np.where(xi == 0, 0.0, 1.0 / cdcd * (xi / rd * sdcd + 2 * np.arctan(
                (et * (x + q * cd) + x * (r + x) * sd) / (xi * (r + x) * cd))))
        ai3 = (y * cd / rd - c['ale'] + sd * np.log(rd)) / cdcd
        ak1 = xi * (d11 - c['y11'] * sd) / cd
        ak3 = (q * c['y11'] - y * d11) / cd
        aj3 = (ak1 - aj2 * sd) / cd
        aj6 = (ak3 - aj5 * sd) / cd
    else:
        rd2 = rd * rd
        ai3 = (et / rd + y * q / rd2 - c['ale']) / 2
        ai4 = xi * y / rd2 / 2
    ai1 = -xi / rd * cd - ai4 * sd
    ai2 = np.log(rd) + ai3 * sd
    qx, qy = q * c['x11'], q * c['y11']
    sdcd, sdsd = sd * cd, sd * sd
    u = np.zeros((3,) + np.shape(xi))
    if d1:
        u += d1 * np.array([-xi * qy - tt - a3 * ai1 * sd, -q / r + a3 * y / rd * sd,
                            q * qy - a3 * ai2 * sd])
    if d2:
        u += d2 * np.array([-q / r + a3 * ai3 * sdcd, -et * qx - tt - a3 * xi / rd * sdcd,
                            q * qx + a3 * ai4 * sdcd])
    if d3:
        u += d3 * np.array([q * qy - a3 * ai3 * sdsd, q * qx + a3 * xi / rd * sdsd,
                            et * qx + xi * qy - tt - a3 * ai4 * sdsd])
    return u / (2 * np.pi)


def _uc(c, z, d1, d2, d3, a4, a5, sd, cd):
    xi, q, et, r, y, d = c['xi'], c['q'], c['et'], c['r'], c['y'], c['d']
    xi2, q2, r3 = c['xi2'], c['q2'], c['r3']
    x11, x32, y11, y32 = c['x11'], c['x32'], c['y11'], c['y32']
    cc = d + z
    h = q * cd - z
    z32 = sd / r3 - h * y32
    qy, qx, xy = q * y11, q * x11, xi * y11
    u = np.zeros((3,) + np.shape(xi))
    if d1:
        u += d1 * np.array([a4 * xy * cd - a5 * xi * q * z32,
                            a4 * (cd / r + 2 * qy * sd) - a5 * cc * q / r3,
                            a4 * qy * cd - a5 * (cc * et / r3 - z * y11 + xi2 * z32)])
    if d2:
        u += d2 * np.array([a4 * cd / r - qy * sd - a5 * cc * q / r3,
                            a4 * y * x11 - a5 * cc * et * q * x32,
                            -d * x11 - xy * sd - a5 * cc * (x11 - q2 * x32)])
    if d3:
        u += d3 * np.array([-a4 * (sd / r + qy * cd) - a5 * (z * y11 - q2 * z32),
                            a4 * 2 * xy * sd + d * x11 - a5 * cc * (x11 - q2 * x32),
                            a4 * (qy * sd + y * x11) + a5 * q * (cc * et * x32 + xi * z32)])
    return u / (2 * np.pi)


def dc3d_disp(x, y, z, depth, dip, al1, al2, aw1, aw2, d1, d2, d3, alpha=ALPHA):
    """Okada (1992) DC3D displacement.  Fault coords: x along strike, z up (z<=0 in medium),
    reference point (0,0,-depth), fault spans x in [al1,al2], up-dip coordinate in [aw1,aw2].
    d1 strike-slip (+ = left-lateral), d2 dip-slip (+ = reverse), d3 tensile.
    Returns (3, N) displacement in the same units as d1..d3."""
    x, y, z = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, z))
    a1, a2 = (1 - alpha) / 2, alpha / 2
    a3, a4, a5 = (1 - alpha) / alpha, 1 - alpha, alpha
    sd, cd = np.sin(np.radians(dip)), np.cos(np.radians(dip))
    if abs(cd) < EPS:
        cd = 0.0
        sd = 1.0 if sd > 0 else -1.0
    xi = [x - al1, x - al2]
    u = np.zeros((3,) + x.shape)
    for img in (False, True):
        dd = depth - z if img else depth + z
        p = y * cd + dd * sd
        q = y * sd - dd * cd
        et = [p - aw1, p - aw2]
        # singular-branch flags as in DC3D
        r = [[np.sqrt(xi[j] ** 2 + et[k] ** 2 + q ** 2) for k in range(2)] for j in range(2)]
        kxi = [(xi[0] < 0) & (r[1][k] + xi[1] < EPS) for k in range(2)]
        ket = [(et[0] < 0) & (r[j][1] + et[1] < EPS) for j in range(2)]
        for k in range(2):
            for j in range(2):
                c = _dccon2(xi[j], et[k], q, sd, cd, kxi[k], ket[j])
                ua = _ua(c, d1, d2, d3, a1, a2)
                if not img:
                    du = np.array([-ua[0], -ua[1] * cd + ua[2] * sd, -ua[1] * sd - ua[2] * cd])
                else:
                    ub = _ub(c, d1, d2, d3, a3, sd, cd)
                    uc = _uc(c, z, d1, d2, d3, a4, a5, sd, cd)
                    du = np.array([ua[0] + ub[0] + z * uc[0],
                                   (ua[1] + ub[1] + z * uc[1]) * cd - (ua[2] + ub[2] + z * uc[2]) * sd,
                                   (ua[1] + ub[1] - z * uc[1]) * sd + (ua[2] + ub[2] - z * uc[2]) * cd])
                u += du if (j + k) != 1 else -du
    return u


# ---------- rectangular source in a global ENU frame (km), slip in m ----------
class Rect:
    """Rectangle given by centroid (e,n,depth km, depth positive down), strike, dip, rake (deg),
    length L, width W (km) and slip (m).  Rake: Aki-Richards (0 left-lateral, 180 right-lateral)."""

    def __init__(self, e, n, dep, strike, dip, rake, L, W, slip):
        self.e, self.n, self.dep = e, n, dep
        self.strike, self.dip, self.rake = strike, dip, rake
        self.L, self.W, self.slip = L, W, slip
        s = np.radians(strike)
        self.ex = np.array([np.sin(s), np.cos(s), 0.0])      # along strike
        self.ey = np.array([-np.cos(s), np.sin(s), 0.0])     # fault-local y (dip goes to -y)
        self.ez = np.array([0.0, 0.0, 1.0])
        self.R = np.vstack([self.ex, self.ey, self.ez])      # global -> local

    def disp(self, P):
        """P: (N,3) ENU km with U = -depth.  Returns (N,3) ENU displacement (m)."""
        rel = P - np.array([self.e, self.n, 0.0])      # horizontal shift only; z stays = U
        loc = rel @ self.R.T
        r = np.radians(self.rake)
        u = dc3d_disp(loc[:, 0], loc[:, 1], loc[:, 2], self.dep, self.dip,
                      -self.L / 2, self.L / 2, -self.W / 2, self.W / 2,
                      self.slip * np.cos(r), self.slip * np.sin(r), 0.0)
        return (self.R.T @ u).T


def disp_sum(srcs, P):
    return sum(s.disp(P) for s in srcs)


def strain_fd(srcs, P, h=0.05):
    """4th-order central differences, h in km.  Returns (N,3,3) strain (dimensionless)."""
    P = np.atleast_2d(P)
    G = np.zeros((len(P), 3, 3))
    for j in range(3):
        dv = np.zeros(3)
        dv[j] = h
        up1, um1 = disp_sum(srcs, P + dv), disp_sum(srcs, P - dv)
        up2, um2 = disp_sum(srcs, P + 2 * dv), disp_sum(srcs, P - 2 * dv)
        G[:, :, j] = (-up2 + 8 * up1 - 8 * um1 + um2) / (12 * h * 1000.0)   # m / m
    return 0.5 * (G + np.transpose(G, (0, 2, 1)))


def stress_from_strain(eps, mu=MU, lam=LAM):
    tr = np.trace(eps, axis1=1, axis2=2)
    return lam * tr[:, None, None] * np.eye(3) + 2 * mu * eps


def receiver_vectors(strike, dip, rake):
    s, d, r = np.radians(strike), np.radians(dip), np.radians(rake)
    sv = np.array([np.sin(s), np.cos(s), 0.0])
    h = np.array([np.cos(s), -np.sin(s), 0.0])              # dip direction (horizontal)
    dv = np.cos(d) * h + np.array([0, 0, -np.sin(d)])        # down-dip
    n = np.sin(d) * h + np.array([0, 0, np.cos(d)])          # normal into hanging wall
    u = np.cos(r) * sv - np.sin(r) * dv                      # hanging-wall slip direction
    return n, u


def resolve(sig, strike, dip, rake):
    n, u = receiver_vectors(strike, dip, rake)
    t = sig @ n
    return t @ u, t @ n          # dtau (drives slip along rake), dsn (tension +)


def sdr_from_vectors(n, u):
    """Inverse of receiver_vectors (n: hanging-wall normal, u: slip).  Returns strike, dip, rake."""
    if n[2] < 0:
        n, u = -n, -u
    dip = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    dipdir = np.degrees(np.arctan2(n[0], n[1]))
    strike = (dipdir - 90) % 360
    s = np.radians(strike)
    sv = np.array([np.sin(s), np.cos(s), 0.0])
    _, dv0 = None, np.cos(np.radians(dip)) * np.array([np.cos(s), -np.sin(s), 0]) + \
        np.array([0, 0, -np.sin(np.radians(dip))])
    rake = np.degrees(np.arctan2(-u @ dv0, u @ sv))
    return strike, dip, rake


def aux_plane(strike, dip, rake):
    n, u = receiver_vectors(strike, dip, rake)
    return sdr_from_vectors(u.copy(), n.copy())


# ----------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------
def validate():
    res = {}
    # (i) 2-D buried screw dislocation: vertical strike-slip, slip 1 m from depth D=10 km to Db
    E = np.linspace(-100, 100, 41)
    Db, D = 1e5, 10.0
    src = Rect(0, 0, (Db + D) / 2, 0, 90, 0, 4e6, Db - D, 1.0)
    errs = []
    for zeta in (0.0, 5.0):
        P = np.c_[E, np.zeros_like(E), -zeta * np.ones_like(E)]
        u = src.disp(P)[:, 1]
        ana = (np.arctan(E / (D - zeta)) + np.arctan(E / (D + zeta)) - np.arctan(E / (Db - zeta))
               - np.arctan(E / (Db + zeta))) / (2 * np.pi)
        errs.append(float(np.abs(u - ana).max()))
    res['screw_dislocation_max_abs_err_m_per_m_slip'] = max(errs)
    # (ii) displacement jump = slip vector (dip 37 deg, strike, dip and tensile)
    dep, dip = 12., 37.
    sd, cd = np.sin(np.radians(dip)), np.cos(np.radians(dip))
    n = np.array([0, -sd, cd])
    jerr, offj = [], []
    for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        exp = d[0] * np.array([1, 0, 0]) + d[1] * np.array([0, cd, sd]) + d[2] * n
        for xi, eta in [(1., 2.), (-3., -1.), (4., 3.5)]:
            p0 = np.array([xi, eta * cd, -dep + eta * sd])
            e = 1e-5
            du = (dc3d_disp(*(p0 + e * n), dep, dip, -5, 5, -4, 4, *d)
                  - dc3d_disp(*(p0 - e * n), dep, dip, -5, 5, -4, 4, *d)).ravel()
            jerr.append(float(np.abs(du - exp).max()))
        p0 = np.array([8., cd, -dep + sd])
        offj.append(float(np.abs(dc3d_disp(*(p0 + e * n), dep, dip, -5, 5, -4, 4, *d)
                                 - dc3d_disp(*(p0 - e * n), dep, dip, -5, 5, -4, 4, *d)).max()))
    res['jump_minus_slip_max_abs_err'] = max(jerr)
    res['jump_outside_fault_max_abs'] = max(offj)
    # (ii-b) the global wrapper: jump on a strike-293 / dip-40 / rake-150 rectangle
    r = Rect(10, -20, 15, 293, 40, 150, 20, 10, 2.0)
    n2, u2 = receiver_vectors(293, 40, 150)
    P0 = np.array([10, -20, -15.0]) + 3 * r.ex
    jump = r.disp((P0 + 1e-5 * n2)[None]) - r.disp((P0 - 1e-5 * n2)[None])
    res['global_wrapper_jump_err'] = float(np.abs(jump[0] - 2 * u2).max())
    # (iii) traction-free surface, (iv) equilibrium, (v) stress drop sign on the source
    srcs = [Rect(0, 0, 8, 293, 40, 150, 20, 10, 2.0)]
    h = 0.002
    Ps = np.array([[5, 3, 0], [-12, 7, 0], [20, -15, 0.]])
    G = np.zeros((3, 3, 3))
    for j in range(2):
        dv = np.zeros(3); dv[j] = h
        G[:, :, j] = (disp_sum(srcs, Ps + dv) - disp_sum(srcs, Ps - dv)) / (2 * h * 1000)
    dz = np.array([0, 0, h])
    G[:, :, 2] = (3 * disp_sum(srcs, Ps) - 4 * disp_sum(srcs, Ps - dz) + disp_sum(srcs, Ps - 2 * dz)) / (2 * h * 1000)
    S = stress_from_strain(0.5 * (G + G.transpose(0, 2, 1)))
    res['surface_traction_rel'] = float(np.abs(S[:, :, 2]).max() / np.abs(S).max())
    Pi = np.array([[5, 8, -12], [-15, 4, -25], [30, -10, -20.]])
    H = 0.2
    st = lambda P: stress_from_strain(strain_fd(srcs, P))
    div = np.zeros((3, 3))
    for j in range(3):
        dv = np.zeros(3); dv[j] = H
        div += ((-st(Pi + 2 * dv) + 8 * st(Pi + dv) - 8 * st(Pi - dv) + st(Pi - 2 * dv)) / (12 * H))[:, :, j]
    dist = np.linalg.norm(Pi - np.array([0, 0, -8]), axis=1)
    res['equilibrium_rel'] = float((np.abs(div).max(1) / (np.abs(st(Pi)).max((1, 2)) / dist)).max())
    Pc = np.array([[0, 0, -8.]]) + 0.05 * n2
    tau, sn = resolve(stress_from_strain(strain_fd(srcs, Pc, h=0.005))[0], 293, 40, 150)
    res['source_centre_dtau_MPa_2m_slip_20x10km'] = float(tau / 1e6)
    # (vi) aux-plane routine
    res['aux_plane_check_(323,69,169)->'] = [round(v, 1) for v in aux_plane(323, 69, 169)]
    ok = (res['screw_dislocation_max_abs_err_m_per_m_slip'] < 1e-5 and res['jump_minus_slip_max_abs_err'] < 1e-4
          and res['surface_traction_rel'] < 1e-5 and res['equilibrium_rel'] < 1e-3 and tau < 0)
    res['all_pass'] = bool(ok)
    return res


if __name__ == '__main__' and '--validate' in sys.argv:
    print(json.dumps(validate(), indent=1))


# ----------------------------------------------------------------------------------------
# Part B. Sources, receivers, stresses
# ----------------------------------------------------------------------------------------
LAT0, LON0 = 40.3, -124.3
KMDEG = 111.195


def ll2en(lat, lon):
    return (np.asarray(lon) - LON0) * KMDEG * np.cos(np.radians(LAT0)), (np.asarray(lat) - LAT0) * KMDEG


# Mechanisms: (primary plane, alternative plane, source of MT, tectonic rationale)
# Primary plane chosen by tectonic setting: Mendocino fault zone (MFZ) events -> WNW/E-W right-lateral
# plane; Gorda-slab events -> NE-striking left-lateral plane (Gorda internal deformation style, as in
# the USGS Ferndale FFM); normal event: steeper plane.
MECH = {
    'nc71338066': dict(pl=(57, 80, 20), src='USGS WCMT (pref.)', set='Gorda slab'),
    'nc71348851': dict(pl=(307, 57, 175), src='USGS WCMT (pref.)', set='MFZ'),
    'nc71734741': dict(pl=(226, 71, -76), src='USGS WCMT (pref.)', set='Gorda slab, normal'),
    'nc72182046': dict(pl=(228, 82, -14), src='USGS Mww', set='Gorda plate'),
    'nc72387946': dict(pl=(105, 89, -173), src='NC/USGS Mww', set='MFZ'),
    'nc72689331': dict(pl=(96, 85, 176), src='USGS Mww', set='MFZ'),
    'nc73201181': dict(pl=(295.66, 88.05, -164.45), src='USGS Mww', set='MFZ/MTJ onshore'),
    'nc73351710': dict(pl=(94.09, 89.17, -169.61), src='USGS Mww', set='MFZ'),
    'nc71127029': dict(pl=None, fm=(185, 60, -10), src='NC first-motion only (no MT)', set='MFZ'),
    'nc73666231': dict(pl=(23.86, 77.55, -0.99), src='USGS Mww', set='Gorda slab'),
    'nc73821036': dict(pl=(71.09, 83.24, 9.5), src='USGS Mww + USGS FFM', set='Gorda slab', ffm='ferndale.fsp'),
    'nc73890906': dict(pl=(273.43, 74.3, -175.49), src='USGS Mww', set='MFZ'),
    'nc75095651': dict(pl=(279.59, 83.75, -178.87), src='USGS Mww + USGS FFM', set='MFZ', ffm='m7.fsp'),
}
MT_DEPTH = {'nc72689331': 15.5, 'nc73890906': 11.5, 'nc75095651': 11.5}   # catalog depth fixed/unconstrained


def wc_dims(M, kind):
    if kind == 'N':
        return 10 ** (-1.88 + 0.50 * M), 10 ** (-1.14 + 0.35 * M)
    return 10 ** (-2.57 + 0.62 * M), 10 ** (-0.76 + 0.27 * M)   # strike-slip (RLD, RW)


def wc_rect(lat, lon, dep, sdr, M, kind='SS'):
    L, W = wc_dims(M, kind)
    M0 = 10 ** (1.5 * M + 9.1)
    slip = M0 / (MU * L * W * 1e6)
    dep = max(dep, W / 2 * np.sin(np.radians(sdr[1])) + 0.5)
    e, n = ll2en(lat, lon)
    return [Rect(float(e), float(n), dep, sdr[0], sdr[1], sdr[2], L, W, slip)]


def ffm_rects(fname):
    txt = open(f'{RAW}/{fname}').read().splitlines()
    hdr = {}
    for l in txt:
        if 'STRK' in l:
            p = l.replace('=', ' ').split()
            hdr['strike'], hdr['dip'] = float(p[p.index('STRK') + 1]), float(p[p.index('DIP') + 1])
        if 'Dx' in l and 'Dz' in l:
            p = l.replace('=', ' ').split()
            hdr['dx'], hdr['dz'] = float(p[p.index('Dx') + 1]), float(p[p.index('Dz') + 1])
    a = np.array([list(map(float, l.split())) for l in txt if l.strip() and not l.startswith('%')])
    out = []
    for lat, lon, x, y, z, slip, rake, trup, rise, m0 in a:
        if m0 <= 0:
            continue
        e, n = ll2en(lat, lon)
        dep = max(z, hdr['dz'] / 2 * np.sin(np.radians(hdr['dip'])) + 0.01)
        s_mc = m0 / (MU * hdr['dx'] * hdr['dz'] * 1e6)       # moment-consistent slip for mu=30 GPa
        out.append(Rect(float(e), float(n), dep, hdr['strike'], hdr['dip'], rake, hdr['dx'], hdr['dz'], s_mc))
    return out, hdr, a


def source_variants(ev, mag, lat, lon, dep):
    """Return dict variant -> list[Rect]."""
    mech = MECH[ev]
    pl = mech['pl'] if mech['pl'] is not None else aux_plane(*mech['fm'])
    alt = aux_plane(*pl)
    kind = 'N' if 'normal' in mech['set'] else 'SS'
    dep = MT_DEPTH.get(ev, dep)
    V = {}
    if 'ffm' in mech:
        V['primary'] = ffm_rects(mech['ffm'])[0]
        V['wc_plane'] = wc_rect(lat, lon, dep, pl, mag, kind)
    else:
        V['primary'] = wc_rect(lat, lon, dep, pl, mag, kind)
    V['alt_plane'] = wc_rect(lat, lon, dep, alt, mag, kind)
    V['depth+5'] = wc_rect(lat, lon, dep + 5, pl, mag, kind)
    V['depth-5'] = wc_rect(lat, lon, max(dep - 5, 1), pl, mag, kind)
    return V, pl, alt


# BSSA14 PGV (Boore, Stewart, Seyhan & Atkinson 2014), Vs30 = 760 m/s (F_S = 0)
BSSA = dict(e0=5.037, e1=5.078, e2=4.849, e3=5.033, e4=1.073, e5=-0.1536, e6=0.2252, Mh=6.2,
            c1=-1.243, c2=0.1489, c3=-0.00344, h=5.3, Mref=4.5, Rref=1.0)


def bssa14_pgv(M, R, kind='SS'):
    c = BSSA
    e = c['e1'] if kind == 'SS' else c['e2']
    FE = e + c['e4'] * (M - c['Mh']) + c['e5'] * (M - c['Mh']) ** 2 if M <= c['Mh'] else e + c['e6'] * (M - c['Mh'])
    Rr = np.sqrt(R ** 2 + c['h'] ** 2)
    FP = (c['c1'] + c['c2'] * (M - c['Mref'])) * np.log(Rr / c['Rref']) + c['c3'] * (Rr - c['Rref'])
    return np.exp(FE + FP)     # cm/s


def rupture_points(rects, frac=0.1):
    """Sample points over rectangles carrying >= frac of the max slip."""
    smax = max(r.slip for r in rects)
    pts = []
    for r in rects:
        if r.slip < frac * smax:
            continue
        sd, cd = np.sin(np.radians(r.dip)), np.cos(np.radians(r.dip))
        for a in np.linspace(-r.L / 2, r.L / 2, 7):
            for b in np.linspace(-r.W / 2, r.W / 2, 5):
                # up-dip coordinate b: local y = b*cd, z = -dep + b*sd
                p = np.array([r.e, r.n, -r.dep]) + a * r.ex + b * cd * r.ey + np.array([0, 0, b * sd])
                pts.append(p)
    return np.array(pts)


RECEIVERS = {'s293_d40_rl': (293, 40, 180), 's293_d45_rl': (293, 45, 180), 's283_d40_rl': (283, 40, 180),
             's303_d40_rl': (303, 40, 180)}
RECV_LL = (293, 40, 0)          # left-lateral slip-sense test
B_LIST = (0.5, 0.7, 0.9)
B0 = 0.7
VS_DYN = 3.5                    # km/s, phase velocity for dynamic strain = PGV / c


def drivers_from_stress(S, recv, B=B0):
    """S: (N,3,3) stress change (Pa, tension +).  Returns dict of arrays in kPa (PGV handled elsewhere)."""
    tau, sn = resolve(S, *recv)
    mean = np.trace(S, axis1=1, axis2=2) / 3
    dP = -mean                                     # compression positive
    dp = B * dP                                    # undrained pore pressure
    dev = S - mean[:, None, None] * np.eye(3)
    J2 = 0.5 * np.einsum('nij,nij->n', dev, dev)
    k = 1e-3
    return dict(dtau=tau * k, dsn=sn * k, dCFS04=(tau + 0.4 * sn) * k, dCFS06=(tau + 0.6 * sn) * k,
                dP=dP * k, dp=dp * k, dsn_eff=(sn + dp) * k, dCFS_ud=(tau + 0.6 * (sn + dp)) * k,
                tau_oct=np.sqrt(J2) * k)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    val = validate()
    print('Okada validation:', json.dumps(val))
    assert val['all_pass']

    fam = pd.read_csv(f'{ROOT}/data/families/shelly_family_locations.csv').sort_values('idnum')
    fe, fn = ll2en(fam.lat.values, fam.lon.values)
    PF = np.c_[fe, fn, -fam.depth_km.values]

    resp = pd.read_csv(f'{ROOT}/data/reports/quake_response_catalog.csv')
    cat = pd.read_csv(f'{ROOT}/data/catalogs/comcat_M5.5_200km_2008_2026.csv')
    cat['t'] = pd.to_datetime(cat.time).dt.tz_localize(None)
    cat['quake'] = [f'{t:%Y-%m-%d} M{m:.1f}' for t, m in zip(cat.t, cat.mag)]
    cat = cat[cat.quake.isin(resp.quake.unique())].reset_index(drop=True)
    assert len(cat) == 13

    # ---- stresses per quake, variant ----
    STRESS, PGV, SRCINFO = {}, {}, {}
    for _, q in cat.iterrows():
        ev = q.id
        V, pl, alt = source_variants(ev, q.mag, q.latitude, q.longitude, q.depth)
        STRESS[q.quake] = {k: stress_from_strain(strain_fd(r, PF)) for k, r in V.items()}
        kind = 'N' if 'normal' in MECH[ev]['set'] else 'SS'
        pts = rupture_points(V['primary'])
        dh = np.sqrt(((PF[:, None, :2] - pts[None, :, :2]) ** 2).sum(-1)).min(1)
        d3 = np.sqrt(((PF[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1)
        M = q.mag
        PGV[q.quake] = dict(Rjb=dh, Rrup=d3, pgv_rjb=bssa14_pgv(M, dh, kind), pgv_rrup=bssa14_pgv(M, d3, kind))
        prim = V['primary']
        SRCINFO[q.quake] = dict(
            id=ev, mag=float(M), mech_source=MECH[ev]['src'], setting=MECH[ev]['set'],
            plane_primary=[round(float(v), 1) for v in pl], plane_alt=[round(float(v), 1) for v in alt],
            model='USGS FFM (%d subfaults, moment-consistent slip for mu=30 GPa)' % len(prim) if 'ffm' in MECH[ev]
            else 'W&C94 uniform rectangle L=%.1f W=%.1f km slip=%.2f m centroid depth %.1f km' % (
                prim[0].L, prim[0].W, prim[0].slip, prim[0].dep))
        print(q.quake, SRCINFO[q.quake]['model'], 'planes', SRCINFO[q.quake]['plane_primary'],
              SRCINFO[q.quake]['plane_alt'])

    # merged 2021-12-20 event (M5.65 + M6.2, 11 s apart: one response)
    a, b = '2021-12-20 M5.7', '2021-12-20 M6.2'
    m = '2021-12-20 M6.2+M5.7'
    STRESS[m] = {k: STRESS[a][k] + STRESS[b][k] for k in STRESS[b]}
    PGV[m] = {k: (np.maximum(PGV[a][k], PGV[b][k]) if k.startswith('pgv') else np.minimum(PGV[a][k], PGV[b][k]))
              for k in PGV[b]}

    # ---- per-family driver table ----
    rows = []
    for qk in STRESS:
        for var, S in STRESS[qk].items():
            for rn, recv in list(RECEIVERS.items()) + [('s293_d40_LL', RECV_LL)]:
                for B in B_LIST:
                    if B != B0 and not (var == 'primary' and rn == 's293_d40_rl'):
                        continue
                    D = drivers_from_stress(S, recv, B)
                    for i, F in enumerate(fam.idnum.values):
                        r = dict(quake=qk, variant=var, receiver=rn, B=B, family=int(F))
                        r.update({k: float(v[i]) for k, v in D.items()})
                        r.update(pgv_rjb=float(PGV[qk]['pgv_rjb'][i]), pgv_rrup=float(PGV[qk]['pgv_rrup'][i]),
                                 Rjb=float(PGV[qk]['Rjb'][i]), Rrup=float(PGV[qk]['Rrup'][i]))
                        rows.append(r)
    fd = pd.DataFrame(rows)
    fd['dyn_strain_ue'] = fd.pgv_rrup / 100 / (VS_DYN * 1000) * 1e6     # microstrain
    fd.to_csv(f'{OUT}/per_family_drivers_all_variants.csv', index=False)

    DRV = ['dtau', 'dsn', 'dCFS04', 'dCFS06', 'dP', 'dp', 'dsn_eff', 'dCFS_ud', 'tau_oct', 'pgv_rjb',
           'pgv_rrup', 'dyn_strain_ue']
    prim = fd[(fd.variant == 'primary') & (fd.receiver == 's293_d40_rl') & (fd.B == B0)]
    ens = fd[(fd.receiver != 's293_d40_LL') & (fd.B == B0)]
    med = prim.groupby('quake')[DRV].median()
    lo = ens.groupby(['quake', 'variant', 'receiver'])[DRV].median().groupby('quake').min()
    hi = ens.groupby(['quake', 'variant', 'receiver'])[DRV].median().groupby('quake').max()
    fam_lo = prim.groupby('quake')[DRV].min()
    fam_hi = prim.groupby('quake')[DRV].max()
    # B range for dp and dsn_eff
    pB = fd[(fd.variant == 'primary') & (fd.receiver == 's293_d40_rl')].groupby(['quake', 'B'])[
        ['dp', 'dsn_eff', 'dCFS_ud']].median().unstack('B')
    ll = fd[(fd.variant == 'primary') & (fd.receiver == 's293_d40_LL')].groupby('quake')[['dtau', 'dCFS04', 'dCFS06']].median()

    # ---- responses ----
    g = resp.groupby('quake')
    R = pd.DataFrame(dict(
        dist_km=g.dist_km.first(),
        ratio30_all=g.apply(lambda x: x.post30.sum() / (x.pre365.sum() * 30 / 365)),
        n_long_gap=g.apply(lambda x: int((x.gap_rank <= 10).sum())),
        median_gap_d=g.gap_d.median(),
        median_norm_gap=g.apply(lambda x: np.median(x.gap_d / (365 / x.pre365.clip(lower=1)))),
    ))
    R.loc[m] = R.loc[b]
    null_p = {'2024-12-05 M7.0': '<0.001', '2022-12-20 M6.4': '0.001', '2010-01-10 M6.5': '0.015'}

    tab = R.join(med, how='left')
    for k in DRV:
        tab[k + '_ens_lo'] = lo[k]
        tab[k + '_ens_hi'] = hi[k]
        tab[k + '_fam_lo'] = fam_lo[k]
        tab[k + '_fam_hi'] = fam_hi[k]
    for B in B_LIST:
        tab[f'dp_B{B}'] = pB[('dp', B)]
        tab[f'dsn_eff_B{B}'] = pB[('dsn_eff', B)]
    tab['dtau_leftlateral'] = ll.dtau
    tab['dCFS04_leftlateral'] = ll.dCFS04
    tab['random_date_null_p'] = [null_p.get(q, '') for q in tab.index]
    tab['mag'] = [SRCINFO[q]['mag'] if q in SRCINFO else 6.2 for q in tab.index]
    tab['source_model'] = [SRCINFO[q]['model'] if q in SRCINFO else 'sum of M6.2 + M5.65 models' for q in tab.index]
    tab.index.name = 'quake'
    tab.to_csv(f'{OUT}/per_quake_drivers_responses.csv')

    # ---- dose-response statistics (n = 12 independent dates: merged 2021) ----
    Q12 = [q for q in tab.index if q not in (a, b)]
    T = tab.loc[Q12]
    rng = np.random.default_rng(1)
    RESP = ['ratio30_all', 'n_long_gap', 'median_gap_d', 'median_norm_gap']
    stat = {}
    for d in DRV:
        stat[d] = {}
        x = T[d].values
        for rsp in RESP:
            y = T[rsp].values
            rho = stats.spearmanr(x, y).statistic
            perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(5000)])
            p = (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (len(perm) + 1)
            bs = []
            for _ in range(2000):
                ii = rng.integers(0, len(x), len(x))
                if len(np.unique(x[ii])) > 2 and len(np.unique(y[ii])) > 2:
                    bs.append(stats.spearmanr(x[ii], y[ii]).statistic)
            stat[d][rsp] = dict(rho=round(float(rho), 3), p_perm=round(float(p), 4),
                                ci95=[round(float(np.nanpercentile(bs, 2.5)), 2), round(float(np.nanpercentile(bs, 97.5)), 2)])
        # rank of the M7 (1 = most extreme in that direction)
        rk_low = int((T[d] < T.loc['2024-12-05 M7.0', d]).sum()) + 1
        rk_high = int((T[d] > T.loc['2024-12-05 M7.0', d]).sum()) + 1
        stat[d]['M7_rank_lowest'] = rk_low
        stat[d]['M7_rank_highest'] = rk_high
        stat[d]['M7_value'] = float(T.loc['2024-12-05 M7.0', d])
        others = T.drop('2024-12-05 M7.0')[d]
        stat[d]['others_min_max'] = [float(others.min()), float(others.max())]
        # robustness of M7 extremeness across ensemble variants (per variant x receiver)
        if d in fd.columns and d not in ('pgv_rjb', 'pgv_rrup', 'dyn_strain_ue'):
            em = ens.groupby(['variant', 'receiver', 'quake'])[d].median().unstack('quake')
            em[m] = em.get(m, np.nan)
            em = em[[c for c in Q12 if c in em.columns]]
            # cross-combination: M7 variant vs all others at their extreme-adverse values
            m7lo, m7hi = lo.loc['2024-12-05 M7.0', d], hi.loc['2024-12-05 M7.0', d]
            olo, ohi = lo.drop([a, b, '2024-12-05 M7.0'])[d], hi.drop([a, b, '2024-12-05 M7.0'])[d]
            stat[d]['M7_ens_range'] = [float(m7lo), float(m7hi)]
            stat[d]['M7_below_all_others_robust'] = bool(m7hi < olo.min())
            stat[d]['M7_above_all_others_robust'] = bool(m7lo > ohi.max())
            stat[d]['frac_variants_M7_lowest'] = float((em.idxmin(axis=1) == '2024-12-05 M7.0').mean())
            stat[d]['frac_variants_M7_highest'] = float((em.idxmax(axis=1) == '2024-12-05 M7.0').mean())
    # top-3 suppressed (random-date p <= 0.015): M7, Ferndale, 2010 M6.5. Rank-sum of those 3 per driver.
    sup = ['2024-12-05 M7.0', '2022-12-20 M6.4', '2010-01-10 M6.5']
    for d in DRV:
        rk = T[d].rank()
        stat[d]['suppressed3_ranks_low_to_high'] = [int(rk[s]) for s in sup]

    # ---- 2-driver combination: magnitude of stress x sign? simple: PGV-normalised clamping ----
    # ---- within-M7, across families ----
    m7 = resp[resp.quake == '2024-12-05 M7.0'].set_index('family')
    pf = prim[prim.quake == '2024-12-05 M7.0'].set_index('family')
    m7 = m7.join(pf[['dtau', 'dsn', 'dp', 'dsn_eff', 'dCFS04', 'dCFS_ud', 'dP', 'pgv_rrup']])
    m7['norm_gap'] = m7.gap_d / (365 / m7.pre365.clip(lower=1))
    m7 = m7.join(fam.set_index('idnum')[['lon', 'lat', 'depth_km']])
    within = {}
    for d in ['dtau', 'dsn', 'dp', 'dsn_eff', 'dCFS04', 'dCFS_ud', 'pgv_rrup', 'lon', 'depth_km']:
        within[d] = {}
        for rsp in ['gap_d', 'norm_gap']:
            r_, p_ = stats.spearmanr(m7[d], m7[rsp])
            within[d][rsp] = dict(rho=round(float(r_), 3), p=round(float(p_), 4))
        within[d]['range_across_families'] = [float(m7[d].min()), float(m7[d].max())]
    # partial: stress vs gap controlling longitude (rank residuals)
    rr = lambda v: stats.rankdata(v)
    def resid(y, x):
        X = np.c_[np.ones(len(x)), rr(x)]
        return rr(y) - X @ np.linalg.lstsq(X, rr(y), rcond=None)[0]
    for d in ['dsn', 'dp', 'dsn_eff']:
        r_, p_ = stats.pearsonr(resid(m7[d].values, m7.lon.values), resid(m7.gap_d.values, m7.lon.values))
        within[d]['gap_d_partial_lon'] = dict(rho=round(float(r_), 3), p=round(float(p_), 4))
    rlon = stats.spearmanr(m7.lon, m7.dsn_eff).statistic
    within['note_lon_vs_dsn_eff_rho'] = round(float(rlon), 3)
    # how large is the across-family variation compared with model uncertainty?
    m7ens = ens[ens.quake == '2024-12-05 M7.0'].groupby(['variant', 'receiver'])[['dsn', 'dp', 'dsn_eff']]
    within['M7_family_spread_vs_model_spread_kPa'] = {
        k: dict(family_range=float(m7[k].max() - m7[k].min()),
                model_range_of_median=float(m7ens.median()[k].max() - m7ens.median()[k].min())) for k in ['dsn', 'dp', 'dsn_eff']}
    m7.to_csv(f'{OUT}/m7_per_family_stress_vs_gap.csv')

    # ---- figure: response vs drivers ----
    plotd = [('dCFS04', r"$\Delta$CFS ($\mu'$=0.4) [kPa]"), ('dtau', r'$\Delta\tau$ (RL) [kPa]'),
             ('dsn', r'$\Delta\sigma_n$ (+unclamp) [kPa]'), ('dp', r'$\Delta p$ undrained B=0.7 [kPa]'),
             ('dsn_eff', r"$\Delta\sigma_n'=\Delta\sigma_n+\Delta p$ [kPa]"), ('tau_oct', r'$|\Delta\tau_{oct}|$ [kPa]'),
             ('pgv_rrup', 'PGV BSSA14 (R$_{rup}$ to LFE) [cm/s]')]
    resp_plot = [('ratio30_all', 'all-family 30-d rate ratio'), ('n_long_gap', 'families with top-10 gap'),
                 ('median_norm_gap', 'median gap / expected interval')]
    fig, axs = plt.subplots(len(resp_plot), len(plotd), figsize=(3.0 * len(plotd), 3.0 * len(resp_plot)),
                            constrained_layout=True)
    for j, (d, lab) in enumerate(plotd):
        for i, (rsp, rl) in enumerate(resp_plot):
            ax = axs[i, j]
            x = T[d]
            xerr = np.vstack([x - T[d + '_ens_lo'], T[d + '_ens_hi'] - x]) if d + '_ens_lo' in T else None
            xerr = np.clip(xerr, 0, None) if xerr is not None else None
            col = ['C3' if q == '2024-12-05 M7.0' else 'C1' if q == '2022-12-20 M6.4' else
                   'C2' if q.startswith('2021-12-20') else 'C0' for q in T.index]
            ax.errorbar(x, T[rsp], xerr=xerr, fmt='none', ecolor='0.7', lw=0.8, zorder=1)
            ax.scatter(x, T[rsp], c=col, s=18 + 8 * (T.mag - 5.5) ** 2 * 6, zorder=2, edgecolor='k', lw=0.4)
            if d.startswith('pgv'):
                ax.set_xscale('log')
                ax.set_xticks([0.2, 0.5, 1, 2, 3])
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            elif d != 'tau_oct':
                ax.axvline(0, color='0.5', lw=0.6, ls=':')
            s = stat[d][rsp]
            ax.set_title(f"rho={s['rho']:+.2f} p={s['p_perm']:.2f}", fontsize=8)
            if i == len(resp_plot) - 1:
                ax.set_xlabel(lab, fontsize=8)
            if j == 0:
                ax.set_ylabel(rl, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle('LFE response vs static/dynamic driver at the LFE fault (n=12 dates; red M7.0, orange Ferndale M6.4, '
                 'green Petrolia 2021). Bars: model-variant range of family-median', fontsize=9)
    fig.savefig(f'{FIG}/sci_dose_response_drivers.png', dpi=150)
    plt.close(fig)

    # figure 2: M7 within-family
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
    for ax, d in zip(axs, ['dsn', 'dp', 'dsn_eff', 'dCFS04']):
        ax.scatter(m7[d], m7.gap_d, c=m7.lon, cmap='viridis', s=25, edgecolor='k', lw=0.3)
        ax.set_xlabel(d + ' [kPa]')
        ax.set_title(f"rho={within[d]['gap_d']['rho']:+.2f} p={within[d]['gap_d']['p']:.2f}", fontsize=9)
    axs[0].set_ylabel('M7 gap (d)')
    fig.suptitle('Within the M7: per-family static stress vs gap length (colour = longitude)', fontsize=9)
    fig.savefig(f'{FIG}/sci_dose_response_m7_families.png', dpi=150)
    plt.close(fig)

    # ---- friction / Skempton sweep of the isotropic-undrained Coulomb stress ----
    mus = np.round(np.arange(0.0, 1.01, 0.05), 2)
    sweep = {}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, B in zip(axs, B_LIST):
        sub = fd[(fd.variant == 'primary') & (fd.receiver == 's293_d40_rl')]
        S0 = sub[sub.B == B0].groupby('quake')[['dtau', 'dsn', 'dP']].median().loc[Q12]
        cfs = pd.DataFrame({mu_: S0.dtau + mu_ * (S0.dsn + B * S0.dP) for mu_ in mus})
        rows_ = []
        for mu_ in mus:
            c = cfs[mu_]
            oth = c.drop('2024-12-05 M7.0')
            rows_.append(dict(mu=float(mu_), M7=float(c['2024-12-05 M7.0']), others_min=float(oth.min()),
                              others_min_quake=oth.idxmin(), M7_lowest=bool(c.idxmin() == '2024-12-05 M7.0')))
        sweep[f'B{B}'] = rows_
        for qq in Q12:
            colr = 'C3' if qq == '2024-12-05 M7.0' else 'C1' if qq == '2022-12-20 M6.4' else 'C2' if qq.startswith('2021') else '0.6'
            ax.plot(mus, cfs.loc[qq], color=colr, lw=2 if colr != '0.6' else 0.8, label=qq if colr != '0.6' else None)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel("friction $\\mu$")
        ax.set_title(f'$\\Delta CFS = \\Delta\\tau + \\mu(\\Delta\\sigma_n + B\\Delta P)$, B={B}', fontsize=9)
    axs[0].set_ylabel('family-median kPa')
    axs[0].legend(fontsize=7)
    fig.savefig(f'{FIG}/sci_dose_response_cfs_mu_sweep.png', dpi=150)
    plt.close(fig)
    crit = {}
    for B in B_LIST:
        sub = fd[(fd.variant == 'primary') & (fd.receiver == 's293_d40_rl')]
        S0 = sub[sub.B == B0].groupby('quake')[['dtau', 'dsn', 'dP']].median().loc[Q12]
        eff = S0.dsn + B * S0.dP
        crit[f'B{B}'] = {q_: (round(float(-S0.dtau[q_] / eff[q_]), 3) if eff[q_] < 0 else None) for q_ in Q12}
    stat['_mu_B_sweep'] = sweep
    stat['_critical_mu_for_negative_dCFS_ud'] = crit

    summary = dict(
        okada_validation=val,
        conventions=__doc__.split('Sign conventions:')[1].split('Outputs')[0].strip(),
        receiver_primary='strike 293, dip 40 NE, rake 180 (right-lateral; Shelly et al. 2026 Science per NOTES.md)',
        elastic=dict(mu_Pa=MU, nu=NU), pgv='BSSA14 PGV (Vs30=760, strike-slip/normal term), distance = closest '
        'distance from the LFE family (at depth, Rrup) or its epicentre (Rjb) to the slip>10%max part of the rupture; '
        'no depth attenuation correction beyond using Rrup; dynamic strain = PGV(Rrup)/3.5 km/s',
        sources=SRCINFO, n_independent_dates=len(Q12), dose_response=stat, within_M7=within,
        table=tab.loc[Q12, RESP + DRV].round(4).reset_index().to_dict(orient='records'))
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(tab.loc[Q12, RESP + DRV].round(2).to_string())
    print(json.dumps({d: {k: v for k, v in stat[d].items()} for d in DRV}, indent=0, default=float)[:6000])
    print(json.dumps(within, indent=0))


if __name__ == '__main__' and '--validate' not in sys.argv:
    main()
