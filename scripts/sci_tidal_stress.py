#!/usr/bin/env python
"""sci_tidal_stress.py - Solid-earth tidal stress on the MTJ LFE fault and the LFE-rate response.

Stages (all in one run, ~10-20 min on 4 threads):
  1. Ephemeris: Meeus "Astronomical Algorithms" Moon (ch. 47, main ~60 lon/dist and 30 lat terms)
     and Sun (ch. 25 low precision); GMST (ch. 12).  Validated against Meeus worked examples.
  2. Degree-2 tidal potential W2 (Moon + Sun) on a spherical Earth, analytic derivatives;
     validation: harmonic analysis of W2/g at the site vs Doodson/Cartwright-Tayler equilibrium
     amplitudes, Greenwich phases vs Schureman arguments, gravity factor delta=1.16 tidal gravity.
  3. Body-tide surface strain with Love/Shida h=0.6, l=0.08:
        e_tt = (h W + l W_tt)/(g a);  e_ll = (h W + l(cot t W_t + W_ll/sin^2 t))/(g a);
        e_tl = l/(g a sin t) (W_tl - cot t W_l)
     check: areal strain = (2h - 6l) W/(g a).  Stress: plane stress (sigma_zz = sigma_xz = sigma_yz = 0),
     mu = 30 GPa, nu = 0.25, assumed CONSTANT WITH DEPTH to ~25-30 km (first-order assumption).
  4. Resolve onto strike 293, dip 40 (and 45) NE-dipping plane, rake 180 (right-lateral) and 0 (LL):
     shear tau in slip direction, normal sigma_n (tension/unclamping positive), CFS = tau + mu' sigma_n.
  5. ROUGH ocean loading (optional, clearly labelled): uniform-amplitude/phase ocean tide west of an
     approximate coastline, Boussinesq point loads on an elastic half-space, stress at each family depth.
     M2 only (primary rough variant) and M2+K1+O1 (rougher).
  6. LFE response: declustered units (meancc>=0.137 then first event per family burst, gap>1 h),
     exactly as scripts/sci_tides.py.  Conditional Poisson likelihood: within each UTC day the units
     are distributed over 72 x 20-min bins with probability ~ exp(D_hour + beta X), i.e. day fixed
     effects (removes all slow rate variation) + a 24-h time-of-day detection pattern.
     rate ~ exp(X / (a sigma)) -> a sigma = 1/beta.  Cluster-robust (day and week) sandwich SEs,
     day-block Poisson bootstrap, score-test nulls: random whole-solar-day shifts and
     Fourier-phase-scrambled stress series.
Outputs: data/sci/tidal_stress/*; figures/sci_tidal_stress_*.png
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import json
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/seismic/LFE"
OUT = f"{ROOT}/data/sci/tidal_stress"
FIG = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261123)

# ---------------------------------------------------------------- constants
LAT0, LON0 = 40.06, -123.625
A_E = 6.371e6            # mean Earth radius (m)
G0 = 9.81
GM_MOON = 4.9028e12
GM_SUN = 1.32712440018e20
AU = 1.495978707e11
H_LOVE, L_SHIDA, DELTA_GRAV = 0.6, 0.08, 1.16
MU, NU = 30e9, 0.25
LAMP = 2 * MU * NU / (1 - NU)      # plane-stress lambda'
LAM3 = 2 * MU * NU / (1 - 2 * NU)  # 3-D lambda
DT_TT = 67.6                        # TT-UTC (s), 2008-2026 mean (65.2-69.2)
STRIKE, DIP = 293.0, 40.0
RHO_W = 1025.0
T0 = pd.Timestamp("2008-01-01")
T_END = pd.Timestamp("2026-07-01")
BIN_MIN = 20
J = 24 * 60 // BIN_MIN               # bins per day
NF = 27
MEANCC_THR, GAP_S = 0.137, 3600.0
TREF = pd.Timestamp("2000-01-01T12:00:00")
MUP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
C1, C2, C3, C4, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#52514e"

# constituent speeds (deg/h) and astronomical-phase periods (h) as in sci_tides.py
SPEED = dict(M2=28.9841042, S2=30.0, N2=28.4397295, K2=30.0821373, K1=15.0410686, O1=13.9430356,
             P1=14.9589314, Q1=13.3986609, Mf=1.0980331, Mm=0.5443747, Ssa=0.0821373, Sa=0.0410686)
PER_H = {k: 360.0 / v for k, v in SPEED.items()}
PER_H.update({"M2": 12.4206012, "N2": 12.65834751, "O1": 25.81933871, "S2": 12.0, "K1": 23.93447213})
# Doodson / Schwiderski equilibrium amplitudes (m), lat factors cos^2(phi) (SD), sin(2phi) (D)
EQ_AMP = dict(M2=0.242334, S2=0.112841, N2=0.046398, K2=0.030704, K1=0.141565, O1=0.100514,
              P1=0.046843, Q1=0.019256)
# observed LFE phases (astronomical convention, network) from data/sci/tides/tidal_whole_period.csv
LFE_PHASE = dict(M2=259.08, O1=124.49, K1=233.51, N2=11.09)
LFE_AMP = dict(M2=0.280, O1=0.237, N2=0.058)

# ---------------------------------------------------------------- ephemeris (Meeus)
MOON_LR = np.array([
    (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111), (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925), (0, 1, 0, 0, -185116, 48888), (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158), (2, -1, -1, 0, 57066, -152138), (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586), (0, 1, -1, 0, -40923, -129620), (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755), (2, 0, 0, -2, 15327, 10321), (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661), (4, 0, -1, 0, 10675, -34782), (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636), (2, 1, -1, 0, -7888, 24208), (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379), (1, 1, 0, 0, 4987, -16675), (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445), (4, 0, 0, 0, 3861, -11650), (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003), (2, 0, -1, 2, -2602, 0), (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322), (2, -2, 0, 0, 2236, -9884), (0, 1, 2, 0, -2120, 5751),
    (0, 2, 0, 0, -2069, 0), (2, -2, -1, 0, 2048, -4950), (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0), (4, -1, -1, 0, 1215, -3958), (0, 0, 2, 2, -1110, 0),
    (3, 0, -1, 0, -892, 3258), (2, 1, 1, 0, -810, 2616), (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117), (2, 2, -1, 0, -700, 2354), (2, 1, -2, 0, 691, 0),
    (2, -1, 0, -2, 596, 0), (4, 0, 1, 0, 549, -1423), (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571), (1, 0, -2, 0, -487, -1739), (2, 1, 0, -2, -399, 0),
    (0, 0, 2, -2, -381, -4421), (1, 1, 1, 0, 351, 0), (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0), (2, -1, 2, 0, 327, 0), (0, 2, 1, 0, -323, 1165),
    (1, 1, -1, 0, 299, 0), (2, 0, 3, 0, 294, 0), (2, 0, -1, -2, 0, 8752)], float)
MOON_B = np.array([
    (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693), (2, 0, 0, -1, 173237),
    (2, 0, -1, 1, 55413), (2, 0, -1, -1, 46271), (2, 0, 0, 1, 32573), (0, 0, 2, 1, 17198),
    (2, 0, 1, -1, 9266), (0, 0, 2, -1, 8822), (2, -1, 0, -1, 8216), (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200), (2, 1, 0, -1, -3359), (2, -1, -1, 1, 2463), (2, -1, 0, 1, 2211),
    (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870), (4, 0, -1, -1, 1828), (0, 1, 0, 1, -1794),
    (0, 0, 0, 3, -1749), (0, 1, -1, 1, -1565), (1, 0, 0, 1, -1491), (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410), (0, 1, 0, -1, -1344), (1, 0, 0, -1, -1335), (0, 0, 3, 1, 1107),
    (4, 0, 0, -1, 1021), (4, 0, -1, 1, 833)], float)


def moon_ecl(T):
    """Geometric ecliptic lon, lat (deg, mean equinox of date) and distance (km). T: Julian centuries TT."""
    T = np.asarray(T, float)
    Lp = 218.3164477 + 481267.88123421 * T - 0.0015786 * T ** 2
    D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T ** 2
    M = 357.5291092 + 35999.0502909 * T - 0.0001536 * T ** 2
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T ** 2
    F = 93.2720950 + 483202.0175233 * T - 0.0036539 * T ** 2
    E = 1 - 0.002516 * T - 0.0000074 * T ** 2
    A1 = 119.75 + 131.849 * T
    A2 = 53.09 + 479264.290 * T
    A3 = 313.45 + 481266.484 * T
    r = np.deg2rad
    arg = np.outer(MOON_LR[:, 0], D) + np.outer(MOON_LR[:, 1], M) + np.outer(MOON_LR[:, 2], Mp) + np.outer(MOON_LR[:, 3], F)
    ef = E[None, :] ** np.abs(MOON_LR[:, 1])[:, None]
    sl = (MOON_LR[:, 4][:, None] * ef * np.sin(r(arg))).sum(0)
    sr = (MOON_LR[:, 5][:, None] * ef * np.cos(r(arg))).sum(0)
    argb = np.outer(MOON_B[:, 0], D) + np.outer(MOON_B[:, 1], M) + np.outer(MOON_B[:, 2], Mp) + np.outer(MOON_B[:, 3], F)
    efb = E[None, :] ** np.abs(MOON_B[:, 1])[:, None]
    sb = (MOON_B[:, 4][:, None] * efb * np.sin(r(argb))).sum(0)
    sl += 3958 * np.sin(r(A1)) + 1962 * np.sin(r(Lp - F)) + 318 * np.sin(r(A2))
    sb += (-2235 * np.sin(r(Lp)) + 382 * np.sin(r(A3)) + 175 * np.sin(r(A1 - F)) + 175 * np.sin(r(A1 + F))
           + 127 * np.sin(r(Lp - Mp)) - 115 * np.sin(r(Lp + Mp)))
    lam = (Lp + sl / 1e6) % 360
    bet = sb / 1e6
    dist = 385000.56 + sr / 1000.0
    return lam, bet, dist


def sun_ecl(T):
    """Meeus ch. 25 low precision: true lon (deg), distance (AU)."""
    T = np.asarray(T, float)
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T ** 2
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T ** 2
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T ** 2
    Mr = np.deg2rad(M)
    C = ((1.914602 - 0.004817 * T - 0.000014 * T ** 2) * np.sin(Mr) + (0.019993 - 0.000101 * T) * np.sin(2 * Mr)
         + 0.000289 * np.sin(3 * Mr))
    lam = (L0 + C) % 360
    nu = Mr + np.deg2rad(C)
    R = 1.000001018 * (1 - e ** 2) / (1 + e * np.cos(nu))
    return lam, R


def gmst_deg(jd_ut):
    T = (jd_ut - 2451545.0) / 36525.0
    return (280.46061837 + 360.98564736629 * (jd_ut - 2451545.0) + 0.000387933 * T ** 2 - T ** 3 / 38710000.0) % 360


def ecl2eq(lam, bet, T):
    eps = np.deg2rad(23.439291 - 0.0130042 * T)
    l, b = np.deg2rad(lam), np.deg2rad(bet)
    x = np.cos(b) * np.cos(l)
    y = np.cos(b) * np.sin(l) * np.cos(eps) - np.sin(b) * np.sin(eps)
    z = np.cos(b) * np.sin(l) * np.sin(eps) + np.sin(b) * np.cos(eps)
    return np.c_[x, y, z]


def jd_from_ts(ts):
    """ts: DatetimeIndex/array (UTC) -> JD(UT)."""
    return (pd.DatetimeIndex(ts) - TREF).total_seconds().to_numpy() / 86400.0 + 2451545.0


def bodies_ef(jd_ut):
    """Earth-fixed unit vectors (n,3) and distances (m) of Moon and Sun."""
    T = (jd_ut + DT_TT / 86400.0 - 2451545.0) / 36525.0
    lm, bm, dm = moon_ecl(T)
    ls, rs = sun_ecl(T)
    um = ecl2eq(lm, bm, T)
    us = ecl2eq(ls, np.zeros_like(ls), T)
    th = np.deg2rad(gmst_deg(jd_ut))
    c, s = np.cos(th), np.sin(th)

    def rot(u):
        return np.c_[c * u[:, 0] + s * u[:, 1], -s * u[:, 0] + c * u[:, 1], u[:, 2]]
    return rot(um), dm * 1e3, rot(us), rs * AU


def potential_derivs(lat, lon, jd_ut):
    """Degree-2 potential W (J/kg) at r=a and analytic angular derivatives (theta=colatitude)."""
    th, la = np.deg2rad(90 - lat), np.deg2rad(lon)
    st, ct, sl, cl = np.sin(th), np.cos(th), np.sin(la), np.cos(la)
    n = np.array([st * cl, st * sl, ct])
    nt = np.array([ct * cl, ct * sl, -st])
    nl = np.array([-st * sl, st * cl, 0.0])
    nll = np.array([-st * cl, -st * sl, 0.0])
    ntl = np.array([-ct * sl, ct * cl, 0.0])
    um, dm, us, ds = bodies_ef(jd_ut)
    out = dict(W=0, Wt=0, Wl=0, Wtt=0, Wll=0, Wtl=0, Wm=0, Ws=0)
    for u, dist, GM, key in ((um, dm, GM_MOON, "Wm"), (us, ds, GM_SUN, "Ws")):
        K = GM * A_E ** 2 / dist ** 3
        c = u @ n
        ct_, cl_, cll, ctl = u @ nt, u @ nl, u @ nll, u @ ntl
        W = K * (1.5 * c ** 2 - 0.5)
        out[key] = W
        out["W"] = out["W"] + W
        out["Wt"] = out["Wt"] + 3 * K * c * ct_
        out["Wl"] = out["Wl"] + 3 * K * c * cl_
        out["Wtt"] = out["Wtt"] + 3 * K * (ct_ ** 2 - c ** 2)
        out["Wll"] = out["Wll"] + 3 * K * (cl_ ** 2 + c * cll)
        out["Wtl"] = out["Wtl"] + 3 * K * (ct_ * cl_ + c * ctl)
    out["theta"] = th
    return out


def body_strain(P):
    th = P["theta"]
    ga = G0 * A_E
    ett = (H_LOVE * P["W"] + L_SHIDA * P["Wtt"]) / ga
    ell = (H_LOVE * P["W"] + L_SHIDA * (P["Wt"] / np.tan(th) + P["Wll"] / np.sin(th) ** 2)) / ga
    etl = L_SHIDA / (ga * np.sin(th)) * (P["Wtl"] - P["Wl"] / np.tan(th))
    return dict(enn=ett, eee=ell, ene=-etl)  # north = -theta


def plane_stress(e):
    tr = e["enn"] + e["eee"]
    return dict(snn=2 * MU * e["enn"] + LAMP * tr, see=2 * MU * e["eee"] + LAMP * tr, sne=2 * MU * e["ene"])


def fault_vectors(strike, dip, rake):
    """Aki-Richards, coords (N, E, Down). n: footwall->hanging-wall normal; d: hanging-wall slip."""
    f, dl, r = np.deg2rad([strike, dip, rake])
    n = np.array([-np.sin(dl) * np.sin(f), np.sin(dl) * np.cos(f), -np.cos(dl)])
    d = np.array([np.cos(r) * np.cos(f) + np.sin(r) * np.cos(dl) * np.sin(f),
                  np.cos(r) * np.sin(f) - np.sin(r) * np.cos(dl) * np.cos(f), -np.sin(r) * np.sin(dl)])
    s = np.array([np.cos(f), np.sin(f), 0.0])
    return n, d, s


def resolve(S, strike, dip, rake=180.0):
    """S: dict of tensor components in N,E,D (arrays).  Returns tau (slip dir), sigma_n (tension +),
    sigma_ss (along-strike normal), mean stress."""
    n, d, s = fault_vectors(strike, dip, rake)
    keys = [["snn", "sne", "snd"], ["sne", "see", "sed"], ["snd", "sed", "sdd"]]

    def comp(a, b):
        out = 0
        for i in range(3):
            for k in range(3):
                if a[i] == 0 or b[k] == 0:
                    continue
                out = out + a[i] * b[k] * S.get(keys[i][k], 0.0)
        return out
    return dict(tau=comp(d, n), sn=comp(n, n), sss=comp(s, s),
                mean=(S["snn"] + S["see"] + S.get("sdd", 0.0)) / 3.0)


# ---------------------------------------------------------------- validation
def harmonic_fit(t_h, y, consts, V=None):
    """Least-squares amplitudes/phases.  If V (dict of argument series, deg) is given, phases are
    Greenwich-type lags G: y = A cos(V - G).  Else astronomical convention ph=2pi frac(t/P)."""
    cols = []
    for c in consts:
        ph = np.deg2rad(V[c]) if V is not None else 2 * np.pi * np.mod(t_h / PER_H[c], 1.0)
        cols += [np.cos(ph), np.sin(ph)]
    X = np.c_[np.ones_like(y), np.array(cols).T]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = {}
    for i, c in enumerate(consts):
        a_, s_ = b[1 + 2 * i], b[2 + 2 * i]
        res[c] = (float(np.hypot(a_, s_)), float(np.rad2deg(np.arctan2(s_, a_)) % 360))
    return res


def doodson_V(jd_ut):
    """Schureman equilibrium arguments (deg) at Greenwich; T (mean-sun hour angle) = GMST - h."""
    T = (jd_ut - 2451545.0) / 36525.0
    s = 218.3164477 + 481267.88123421 * T
    h = 280.46646 + 36000.76983 * T
    p = 83.3532465 + 4069.0137287 * T
    tg = gmst_deg(jd_ut)
    return dict(M2=2 * tg - 2 * s, S2=2 * tg - 2 * h, N2=2 * tg - 3 * s + p, K2=2 * tg,
                K1=tg - 90, O1=tg - 2 * s + 90, P1=tg - 2 * h + 90, Q1=tg - 3 * s + p + 90,
                Mf=2 * s, Mm=s - p)


def validate():
    v = {}
    # Meeus Example 47.a: 1992-04-12 0h TD
    lm, bm, dm = moon_ecl(np.array([(2448724.5 - 2451545.0) / 36525]))
    v["moon_meeus47a"] = dict(lam=float(lm[0]), lam_ref=133.162655, beta=float(bm[0]), beta_ref=-3.229126,
                              dist_km=float(dm[0]), dist_ref=368409.7)
    ls, rs = sun_ecl(np.array([(2448908.5 - 2451545.0) / 36525]))
    v["sun_meeus25a"] = dict(lam=float(ls[0]), lam_ref=199.90988, R_au=float(rs[0]), R_ref=0.99766)
    v["gmst_meeus12a"] = dict(gmst=float(gmst_deg(np.array([2446895.5]))[0]), ref=197.693195)
    print(json.dumps(v, indent=1))
    return v


# ---------------------------------------------------------------- Boussinesq ocean load
COAST = np.array([  # (lat, lon) rough coastline, N. California - Oregon
    (35.5, -121.0), (36.0, -121.5), (36.6, -121.9), (37.0, -122.2), (37.5, -122.5), (37.8, -122.5),
    (38.0, -123.0), (38.3, -123.07), (38.77, -123.53), (38.95, -123.74), (39.45, -123.81),
    (39.64, -123.78), (39.83, -123.85), (40.0, -124.03), (40.03, -124.08), (40.25, -124.35),
    (40.44, -124.41), (40.6, -124.35), (40.77, -124.24), (41.06, -124.14), (41.75, -124.20),
    (42.0, -124.21), (42.84, -124.56), (43.5, -124.25), (44.0, -124.12), (44.6, -124.08)])


def ocean_cells(lat_min=35.5, lat_max=44.6, lon_min=-131.0, dd=0.02):
    lats = np.arange(lat_min + dd / 2, lat_max, dd)
    lons = np.arange(lon_min + dd / 2, -120.9, dd)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    coast = np.interp(LA, COAST[:, 0], COAST[:, 1])
    m = LO < coast
    la, lo = LA[m], LO[m]
    area = (dd * 111.195e3) ** 2 * np.cos(np.deg2rad(la))
    return la, lo, area


def bouss_disp(x, y, z, P):
    """Displacement (ux east, uy north, uz down) at (x,y,z>0) from downward point forces P at origin(s)."""
    r = np.hypot(x, y)
    R = np.sqrt(r ** 2 + z ** 2)
    k = P / (4 * np.pi * MU)
    ur = k * (r * z / R ** 3 - (1 - 2 * NU) * r / (R * (R + z)))
    uz = k * (z ** 2 / R ** 3 + 2 * (1 - NU) / R)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx, cy = np.where(r > 0, x / r, 0), np.where(r > 0, y / r, 0)
    return ur * cx, ur * cy, uz


def bouss_stress(xr, yr, zr, xs, ys, P, h=5.0):
    """Stress tensor (N,E,D keys; tension +) at receiver from point loads P (N) at (xs,ys) on surface."""
    def U(dx, dy, dz):
        ux, uy, uz = bouss_disp(xr + dx - xs, yr + dy - ys, zr + dz, P)
        return np.array([ux.sum(), uy.sum(), uz.sum()])
    gx = (U(h, 0, 0) - U(-h, 0, 0)) / (2 * h)
    gy = (U(0, h, 0) - U(0, -h, 0)) / (2 * h)
    gz = (U(0, 0, h) - U(0, 0, -h)) / (2 * h)
    Gm = np.array([gx, gy, gz]).T  # Gm[i,j] = du_i/dx_j, coords (E, N, D)
    eps = 0.5 * (Gm + Gm.T)
    sig = LAM3 * np.trace(eps) * np.eye(3) + 2 * MU * eps
    # reorder E,N,D -> N,E,D
    o = [1, 0, 2]
    s = sig[np.ix_(o, o)]
    return dict(snn=s[0, 0], see=s[1, 1], sdd=s[2, 2], sne=s[0, 1], snd=s[0, 2], sed=s[1, 2])


def bouss_check():
    z, x = 25e3, 15e3
    s = bouss_stress(x, 0.0, z, np.array([0.0]), np.array([0.0]), np.array([1e12]))
    R = np.hypot(x, z)
    ref_zz = -3 * 1e12 * z ** 3 / (2 * np.pi * R ** 5)
    ref_xz = -3 * 1e12 * x * z ** 2 / (2 * np.pi * R ** 5)
    return dict(szz=float(s["sdd"]), szz_ref=float(ref_zz), sxz=float(s["sed"]), sxz_ref=float(ref_xz))


def ocean_green(lat, lon, depth_km, lon_cut=None):
    """Stress tensor (Pa) at the receiver per 1 m of uniform in-phase ocean tide."""
    la, lo, area = ocean_cells()
    if lon_cut is not None:
        keep = lo > lon_cut
        la, lo, area = la[keep], lo[keep], area[keep]
    xs = (lo - lon) * 111.195e3 * np.cos(np.deg2rad(0.5 * (la + lat)))
    ys = (la - lat) * 111.195e3
    P = RHO_W * G0 * 1.0 * area
    return bouss_stress(0.0, 0.0, depth_km * 1e3, xs, ys, P)


# ocean tide constants (ROUGH; open-coast values off Humboldt/Shelter Cove; Greenwich phase lag)
OCEAN = dict(M2=(0.70, 220.0), K1=(0.37, 235.0), O1=(0.23, 220.0))


# ---------------------------------------------------------------- data
def load_units():
    d = pd.read_csv(f"{ROOT}/data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv")
    d.columns = [c.lstrip("%") for c in d.columns]
    d["t"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day)) + pd.to_timedelta(d.s_of_day, unit="s")
    d["fam"] = d.idnum.astype(int) - 1
    d["tsec"] = (d.t - TREF).dt.total_seconds()
    d = d[d.meancc >= MEANCC_THR].sort_values(["fam", "t"])
    g = d.groupby("fam").tsec.diff()
    u = d[(g.isna() | (g > GAP_S)).to_numpy()].sort_values("t").reset_index(drop=True)
    s = (u.t - T0).dt.total_seconds().to_numpy()
    u["day"] = (s // 86400).astype(int)
    u["bin"] = ((s % 86400) // (BIN_MIN * 60)).astype(int)
    u = u[(u.t >= T0) & (u.t < T_END)].reset_index(drop=True)
    return u


# ---------------------------------------------------------------- conditional likelihood fits
HOD = (np.arange(J) * BIN_MIN // 60).astype(int)   # hour of day of each bin
FH = np.zeros((J, 23))
for _j in range(J):
    if HOD[_j] > 0:
        FH[_j, HOD[_j] - 1] = 1.0


def cfit(C, Xs, w=None, offset=None, theta0=None, iters=40, hod=True):
    """Conditional (multinomial-within-day) Poisson fit.
    C (nr,J) counts; Xs list of (nr,J) covariates; w (nr,) row weights; offset (nr,J).
    Returns dict(theta, beta, LL, H (neg Hessian), score_rows)."""
    nr = C.shape[0]
    m = len(Xs)
    parts = []
    if hod:
        parts.append(np.broadcast_to(FH, (nr, J, 23)))
    if m:
        parts.append(np.stack(Xs, -1))
    F = np.concatenate(parts, -1) if parts else np.zeros((nr, J, 0))
    K = F.shape[-1]
    N = C.sum(1)
    w = np.ones(nr) if w is None else w
    off = 0.0 if offset is None else offset
    th = np.zeros(K) if theta0 is None else theta0.copy()

    Fr = F.reshape(nr * J, K)

    def ll_of(th_):
        eta = (Fr @ th_).reshape(nr, J) + off
        mx = eta.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(eta - mx).sum(1))
        return float((w * ((C * eta).sum(1) - N * lse)).sum()), eta, lse

    def grad_hess(eta, lse):
        p = np.exp(eta - lse[:, None])
        resid = C - N[:, None] * p
        g = Fr.T @ (w[:, None] * resid).ravel()
        mbar = np.matmul(p[:, None, :], F)[:, 0, :]
        wn = w * N
        H = Fr.T @ (Fr * (wn[:, None] * p).ravel()[:, None]) - mbar.T @ (mbar * wn[:, None])
        return p, resid, g, mbar, H
    LL, eta, lse = ll_of(th)
    for it in range(iters):
        p, resid, g, mbar, H = grad_hess(eta, lse)
        step = np.linalg.solve(H + 1e-12 * np.eye(K), g)
        lam = 1.0
        while True:
            LLn, etan, lsen = ll_of(th + lam * step)
            if LLn >= LL - 1e-9 or lam < 1e-4:
                break
            lam /= 2
        th, LL, eta, lse = th + lam * step, LLn, etan, lsen
        if np.max(np.abs(lam * step)) < 1e-7:
            break
    p, resid, g, mbar, H = grad_hess(eta, lse)
    srow = np.matmul(resid[:, None, :], F)[:, 0, :]
    nh = 23 if hod else 0
    return dict(theta=th, beta=th[nh:], LL=LL, H=H, srow=srow, nh=nh, p=p)


def robust_se(fit, clusters):
    """Cluster-robust sandwich covariance for the beta block."""
    Hi = np.linalg.inv(fit["H"])
    uc, inv = np.unique(clusters, return_inverse=True)
    S = np.zeros((len(uc), fit["srow"].shape[1]))
    np.add.at(S, inv, fit["srow"])
    V = Hi @ (S.T @ S) @ Hi
    nh = fit["nh"]
    Vb = V[nh:, nh:]
    Vn = Hi[nh:, nh:]
    return Vb, Vn


# ---------------------------------------------------------------- main
def main():
    summ = {}
    summ["validation"] = validate()
    summ["validation"]["boussinesq_point_load"] = bouss_check()

    # ---- time grid (20-min bin centres) and stresses at centroid
    ND = (T_END - T0).days
    tg = T0 + pd.to_timedelta((np.arange(ND * J) + 0.5) * BIN_MIN, unit="min")
    jd = jd_from_ts(tg)
    th_h = (tg - TREF).total_seconds().to_numpy() / 3600.0   # hours since J2000 (sci_tides convention)
    P = potential_derivs(LAT0, LON0, jd)
    e = body_strain(P)
    S = plane_stress(e)
    areal_fd = e["enn"] + e["eee"]
    areal_an = (2 * H_LOVE - 6 * L_SHIDA) * P["W"] / (G0 * A_E)
    summ["validation"]["areal_strain_formula_maxrel"] = float(np.max(np.abs(areal_fd - areal_an)) / np.max(np.abs(areal_an)))

    # equilibrium harmonic analysis at the site (astronomical) and at Greenwich (Schureman V)
    consts = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "Mf", "Mm"]
    sub = slice(None, None, 3)   # hourly
    hW = harmonic_fit(th_h[sub], P["W"][sub] / G0, consts)
    clat2 = np.cos(np.deg2rad(LAT0)) ** 2
    s2lat = np.sin(np.deg2rad(2 * LAT0))
    eqcmp = {}
    for c in ["M2", "S2", "N2", "K2"]:
        eqcmp[c] = dict(model_m=hW[c][0], textbook_m=EQ_AMP[c] * clat2)
    for c in ["K1", "O1", "P1", "Q1"]:
        eqcmp[c] = dict(model_m=hW[c][0], textbook_m=EQ_AMP[c] * s2lat)
    summ["validation"]["equilibrium_tide_site"] = eqcmp
    jd_h = jd[sub]
    Pg = potential_derivs(51.4779, 0.0, jd_h)
    Vg = doodson_V(jd_h)
    hg = harmonic_fit(None, Pg["W"] / G0, ["M2", "S2", "N2", "K1", "O1", "P1", "Q1"], V=Vg)
    summ["validation"]["greenwich_eq_phase_deg_expect0"] = {k: round(v[1], 2) for k, v in hg.items()}
    dg = -DELTA_GRAV * 2 * P["W"] / A_E * 1e8   # microGal, downward-positive gravity change
    hdg = harmonic_fit(th_h[sub], dg[sub], ["M2", "S2", "O1", "K1"])
    summ["validation"]["gravity_uGal"] = dict(M2=hdg["M2"][0], S2=hdg["S2"][0], O1=hdg["O1"][0], K1=hdg["K1"][0],
                                              max=float(dg.max()), min=float(dg.min()),
                                              expected_M2=EQ_AMP["M2"] * clat2 * 2 * G0 / A_E * DELTA_GRAV * 1e8)
    print("validation:", json.dumps(summ["validation"], indent=1))

    # ---- fault resolution (body tide, plane stress, constant with depth)
    Sk = {k: v / 1e3 for k, v in S.items()}   # kPa
    R40 = resolve(Sk, STRIKE, 40.0)
    R45 = resolve(Sk, STRIKE, 45.0)
    X = dict(tau=R40["tau"], sn=R40["sn"], sss=R40["sss"], mean=R40["mean"],
             tau45=R45["tau"], sn45=R45["sn"])
    for mu in MUP:
        X[f"cfs{mu:.1f}"] = X["tau"] + mu * X["sn"]

    # ---- ocean loading (ROUGH)
    fam = pd.read_csv(f"{ROOT}/data/families/shelly_family_locations.csv").sort_values("idnum").reset_index(drop=True)
    cen = dict(lat=fam.lat.mean(), lon=fam.lon.mean(), depth=fam.depth_km.mean())
    Gc = ocean_green(cen["lat"], cen["lon"], cen["depth"])
    Gc300 = ocean_green(cen["lat"], cen["lon"], cen["depth"], lon_cut=cen["lon"] - 300 / 85.0)
    Vt = doodson_V(jd)

    def ocean_series(G, strike=STRIKE, dip=40.0, consts_=("M2",), amp_scale=1.0, dphase=0.0):
        Rg = resolve({k: v / 1e3 for k, v in G.items()}, strike, dip)  # kPa per m of tide
        out = {k: 0.0 for k in ("tau", "sn", "sss", "mean")}
        for c in consts_:
            A, Gph = OCEAN[c]
            eta = amp_scale * A * np.cos(np.deg2rad(Vt[c] - Gph - dphase))
            for k in out:
                out[k] = out[k] + Rg[k] * eta
        return out, Rg
    OLm2, Rg_c = ocean_series(Gc)
    _, Rg_c300 = ocean_series(Gc300)
    OLd, _ = ocean_series(Gc, consts_=("M2", "K1", "O1"))
    summ["ocean_load"] = dict(
        note="ROUGH: uniform ocean tide west of an approximate coastline, Boussinesq half-space, no SAL/"
             "ocean-bottom topography; constants (amp m, Greenwich phase deg) assumed: " + json.dumps(OCEAN),
        centroid=cen, kPa_per_m_tide={k: float(v) for k, v in Rg_c.items()},
        kPa_per_m_tide_cut300km={k: float(v) for k, v in Rg_c300.items()},
        coast_distance_km=float((cen["lon"] - np.interp(cen["lat"], COAST[:, 0], COAST[:, 1])) * 85.2))
    X["tau_OL"] = X["tau"] + OLm2["tau"]
    X["sn_OL"] = X["sn"] + OLm2["sn"]
    X["tau_OLd"] = X["tau"] + OLd["tau"]
    X["sn_OLd"] = X["sn"] + OLd["sn"]
    X["ol_tau"] = OLm2["tau"]
    X["ol_sn"] = OLm2["sn"]

    # ---- harmonic analysis of stress components (sci_tides astronomical convention)
    hrows = []
    for k in ["tau", "sn", "sss", "mean", "cfs0.1", "cfs0.2", "cfs0.4", "cfs0.6", "tau_OL", "sn_OL", "tau_OLd", "sn_OLd", "ol_tau", "ol_sn"]:
        h = harmonic_fit(th_h[sub], X[k][sub], consts)
        for c in consts:
            row = dict(series=k, const=c, amp_kPa=h[c][0], phase_peak_deg=h[c][1])
            if c in LFE_PHASE:
                lag = (LFE_PHASE[c] - h[c][1] + 180) % 360 - 180
                row.update(lfe_phase_deg=LFE_PHASE[c], lfe_minus_stress_deg=lag, lag_hours=lag / 360 * PER_H[c])
            hrows.append(row)
    for k in ["snn", "see", "sne"]:
        h = harmonic_fit(th_h[sub], Sk[k][sub], consts)
        for c in consts:
            hrows.append(dict(series=k, const=c, amp_kPa=h[c][0], phase_peak_deg=h[c][1]))
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(f"{OUT}/stress_harmonics.csv", index=False)
    print(hdf[hdf.const.isin(["M2", "O1", "K1", "N2"])].to_string())

    # ---- hourly output
    hr = slice(1, None, 3)   # bin centred on hh:30 ... use the bins at hh:30
    hourly = pd.DataFrame(dict(time_utc=(tg[hr] - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M"),
                               W2_over_g_m=P["W"][hr] / G0, dg_uGal=dg[hr],
                               e_nn=e["enn"][hr], e_ee=e["eee"][hr], e_ne=e["ene"][hr],
                               s_nn_kPa=Sk["snn"][hr], s_ee_kPa=Sk["see"][hr], s_ne_kPa=Sk["sne"][hr],
                               tau_RL_d40_kPa=X["tau"][hr], sn_d40_kPa=X["sn"][hr],
                               tau_RL_d45_kPa=X["tau45"][hr], sn_d45_kPa=X["sn45"][hr],
                               **{f"cfs{mu:.1f}_d40_kPa": X[f"cfs{mu:.1f}"][hr] for mu in MUP},
                               OLM2_tau_kPa_rough=X["ol_tau"][hr], OLM2_sn_kPa_rough=X["ol_sn"][hr]))
    # NOTE: the hh:30 bin centre is used as the value for hour hh:00-hh:59 (labelled hh:00 + 30 min)
    hourly["time_utc"] = (tg[hr]).strftime("%Y-%m-%dT%H:%M")
    hourly.to_csv(f"{OUT}/tidal_stress_hourly.csv.gz", index=False, float_format="%.6g")
    print("hourly rows", len(hourly))

    # ---- LFE units -> count matrices
    u = load_units()
    summ["n_units"] = int(len(u))
    Cnet = np.zeros((ND, J))
    np.add.at(Cnet, (u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    Cfam = np.zeros((NF, ND, J), np.float32)
    np.add.at(Cfam, (u.fam.to_numpy(), u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    rows_net = np.where(Cnet.sum(1) > 0)[0]
    week = rows_net // 7

    def M(k):
        return X[k].reshape(ND, J)

    def fitnet(keys, rows=rows_net, C=None, **kw):
        C = Cnet[rows] if C is None else C
        return cfit(C, [M(k)[rows] for k in keys], **kw)

    def report(name, keys, fit, rows=rows_net):
        Vd, Vn = robust_se(fit, rows)
        Vw, _ = robust_se(fit, rows // 7)
        out = dict(model=name, covariates="+".join(keys), LL=fit["LL"])
        for i, k in enumerate(keys):
            out[f"beta_{i}"] = float(fit["beta"][i])
            out[f"se_day_{i}"] = float(np.sqrt(Vd[i, i]))
            out[f"se_week_{i}"] = float(np.sqrt(Vw[i, i]))
            out[f"se_poisson_{i}"] = float(np.sqrt(Vn[i, i]))
        return out

    f0 = fitnet([])
    LL0 = f0["LL"]
    p_hod = f0["p"][0]           # null time-of-day pattern (same in every row)
    summ["LL_null"] = LL0
    res = []
    models = {"tau": ["tau"], "sn": ["sn"], "sss": ["sss"], "mean": ["mean"],
              **{f"cfs{mu:.1f}": [f"cfs{mu:.1f}"] for mu in MUP},
              "tau+sn": ["tau", "sn"], "tau+sn+sss": ["tau", "sn", "sss"],
              "tau45": ["tau45"], "sn45": ["sn45"], "tau45+sn45": ["tau45", "sn45"],
              "tau_OL": ["tau_OL"], "sn_OL": ["sn_OL"], "tau_OL+sn_OL": ["tau_OL", "sn_OL"],
              "tau_OLd": ["tau_OLd"], "sn_OLd": ["sn_OLd"], "tau_OLd+sn_OLd": ["tau_OLd", "sn_OLd"]}
    fits = {}
    for name, keys in models.items():
        f = fitnet(keys)
        fits[name] = f
        r = report(name, keys, f)
        r["dLL"] = f["LL"] - LL0
        if len(keys) == 1:
            b, se = r["beta_0"], r["se_week_0"]
            r["a_sigma_kPa"] = 1 / b
            r["a_sigma_lo"] = 1 / (b + 1.96 * se) if b + 1.96 * se > 0 else np.nan
            r["a_sigma_hi"] = 1 / (b - 1.96 * se) if b - 1.96 * se > 0 else np.inf
            r["std_X_kPa"] = float(X[keys[0]].std())
            r["rate_mod_amp"] = float(b * X[keys[0]].std() * np.sqrt(2))   # ~ a for a sinusoid
        if len(keys) >= 2:
            Vw, _ = robust_se(f, rows_net // 7)
            b0, b1 = f["beta"][0], f["beta"][1]
            mu_e = b1 / b0
            g_ = np.array([-b1 / b0 ** 2, 1 / b0])
            r["mu_eff"] = mu_e
            r["mu_eff_se_week"] = float(np.sqrt(g_ @ Vw[:2, :2] @ g_))
        res.append(r)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # mu' profile
    prof = []
    for mu in np.round(np.arange(-0.6, 1.21, 0.05), 3):
        for tag, (kt, kn) in {"body": ("tau", "sn"), "body_d45": ("tau45", "sn45"),
                              "OL_M2": ("tau_OL", "sn_OL"), "OL_M2K1O1": ("tau_OLd", "sn_OLd")}.items():
            xx = M(kt)[rows_net] + mu * M(kn)[rows_net]
            f = cfit(Cnet[rows_net], [xx])
            prof.append(dict(variant=tag, mu=mu, beta=float(f["beta"][0]), LL=f["LL"], dLL=f["LL"] - LL0))
    prof = pd.DataFrame(prof)
    prof.to_csv(f"{OUT}/mu_profile_network.csv", index=False)
    best_mu = {}
    for tag, g in prof.groupby("variant"):
        i = g.LL.idxmax()
        ok = g[g.LL >= g.LL.max() - 1.92]
        best_mu[tag] = dict(mu_best=float(g.mu[i]), beta=float(g.beta[i]), a_sigma_kPa=float(1 / g.beta[i]),
                            mu_95_profile=[float(ok.mu.min()), float(ok.mu.max())],
                            dLL_best=float(g.dLL[i]),
                            dLL_mu0=float(g[np.isclose(g.mu, 0)].dLL.iloc[0]),
                            dLL_mu0p4=float(g[np.isclose(g.mu, 0.4)].dLL.iloc[0]))
    summ["mu_profile_best"] = best_mu
    print("best mu:", best_mu)

    # ocean-load scale/phase profile: X = body + s * OL(phase shifted), tau and sigma_n both free (mu free)
    olp = []
    for consts_, tagc in ((("M2",), "M2"), (("M2", "K1", "O1"), "M2K1O1")):
        for dph in (-90, -60, -30, 0, 30, 60, 90):
            OLx, _ = ocean_series(Gc, consts_=consts_, dphase=dph)
            for s_ in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
                xt = (X["tau"] + s_ * OLx["tau"]).reshape(ND, J)[rows_net]
                xn = (X["sn"] + s_ * OLx["sn"]).reshape(ND, J)[rows_net]
                f = cfit(Cnet[rows_net], [xt, xn])
                f1 = cfit(Cnet[rows_net], [xt])
                olp.append(dict(ocean=tagc, dphase=dph, scale=s_, dLL_tau_sn=f["LL"] - LL0, beta_tau=float(f["beta"][0]),
                                beta_sn=float(f["beta"][1]), mu_eff=float(f["beta"][1] / f["beta"][0]),
                                dLL_tau_only=f1["LL"] - LL0, beta_tau_only=float(f1["beta"][0])))
    olp = pd.DataFrame(olp)
    olp.to_csv(f"{OUT}/ocean_load_scale_profile.csv", index=False)
    summ["ocean_load_profile_best"] = olp.loc[olp.dLL_tau_sn.idxmax()].to_dict()
    summ["ocean_load_profile_best_tau_only"] = olp.loc[olp.dLL_tau_only.idxmax()].to_dict()
    print(olp.sort_values("dLL_tau_sn", ascending=False).head(10).to_string())

    # opposite sense (left-lateral) = beta sign flip on tau; strike scan for strike-slip shear
    scan = []
    for stv in np.arange(0, 360, 10):
        Rs = resolve(Sk, stv, 40.0)
        f = cfit(Cnet[rows_net], [Rs["tau"].reshape(ND, J)[rows_net]])
        scan.append(dict(strike=stv, beta=float(f["beta"][0]), dLL=f["LL"] - LL0))
    scan = pd.DataFrame(scan)
    scan.to_csv(f"{OUT}/strike_scan_rl_shear_network.csv", index=False)
    summ["strike_scan_best"] = scan.loc[scan.dLL.idxmax()].to_dict()

    # phase lag: Hilbert quadrature model and time-lag scan
    lagres = {}
    for k in ["tau", "sn", "cfs0.4", "tau_OL", "sn_OL"]:
        xs = X[k] - X[k].mean()
        hx = np.imag(hilbert(xs))
        f = cfit(Cnet[rows_net], [M(k)[rows_net], hx.reshape(ND, J)[rows_net]])
        Vw, _ = robust_se(f, rows_net // 7)
        bc, bs = f["beta"]
        ph = np.degrees(np.arctan2(bs, bc))
        gph = np.array([-bs, bc]) / (bc ** 2 + bs ** 2)
        ph_se = np.degrees(np.sqrt(gph @ Vw @ gph))
        lags = []
        for sh in range(-18, 19):
            xr = np.roll(X[k], sh).reshape(ND, J)    # X(t - sh*20min): LFE responds sh bins after stress
            fl = cfit(Cnet[rows_net], [xr[rows_net]])
            lags.append((sh * BIN_MIN / 60.0, fl["LL"] - LL0, float(fl["beta"][0])))
        lags = np.array(lags)
        ib = np.argmax(lags[:, 1])
        okl = lags[lags[:, 1] >= lags[ib, 1] - 1.92, 0]
        lagres[k] = dict(phase_lag_deg=float(ph), phase_lag_se_deg=float(ph_se),
                         beta_amp=float(np.hypot(bc, bs)), a_sigma_amp_kPa=float(1 / np.hypot(bc, bs)),
                         dLL_quadrature_model=f["LL"] - LL0,
                         time_lag_best_h=float(lags[ib, 0]), time_lag_95_h=[float(okl.min()), float(okl.max())],
                         time_lag_profile=lags.tolist())
        print(k, "lag", lagres[k]["phase_lag_deg"], "+/-", ph_se, "time lag", lags[ib, 0], okl.min(), okl.max())
    summ["phase_lag"] = {k: {kk: vv for kk, vv in v.items() if kk != "time_lag_profile"} for k, v in lagres.items()}

    # linear vs exponential (profile over gamma) for tau and best body CFS
    linres = {}
    for k in ["tau", f"cfs{best_mu['body']['mu_best']:.1f}" if f"cfs{best_mu['body']['mu_best']:.1f}" in X else "tau"]:
        xx = M(k)[rows_net]
        gmax = 0.98 / max(xx.max(), -xx.min())
        best = (-np.inf, 0)
        for gm in np.linspace(-gmax, gmax, 81):
            f = cfit(Cnet[rows_net], [], offset=np.log1p(gm * xx))
            if f["LL"] > best[0]:
                best = (f["LL"], gm)
        fe = cfit(Cnet[rows_net], [xx])
        linres[k] = dict(gamma_per_kPa=best[1], dLL_linear=best[0] - LL0, dLL_exp=fe["LL"] - LL0,
                         beta_exp=float(fe["beta"][0]), gmax=gmax)
    summ["linear_vs_exp"] = linres
    print("lin vs exp", linres)

    # bootstrap (day Poisson weights) for main 1-param models
    B = 100
    bootb = {k: [] for k in ["tau", "sn", "cfs0.4", "tau_OL"]}
    bootmu = []
    W = rng.poisson(1.0, (B, len(rows_net))).astype(float)
    for b in range(B):
        for k in bootb:
            f = cfit(Cnet[rows_net], [M(k)[rows_net]], w=W[b], theta0=fits[k]["theta"], iters=8)
            bootb[k].append(f["beta"][0])
        f = cfit(Cnet[rows_net], [M("tau")[rows_net], M("sn")[rows_net]], w=W[b], theta0=fits["tau+sn"]["theta"], iters=8)
        bootmu.append(f["beta"][1] / f["beta"][0])
    summ["bootstrap"] = {k: dict(beta_mean=float(np.mean(v)), beta_sd=float(np.std(v)),
                                 a_sigma_2p5_97p5=[float(1 / np.percentile(v, 97.5)), float(1 / np.percentile(v, 2.5))])
                         for k, v in bootb.items()}
    summ["bootstrap"]["mu_eff_tau+sn"] = dict(median=float(np.median(bootmu)), p2p5=float(np.percentile(bootmu, 2.5)),
                                              p97p5=float(np.percentile(bootmu, 97.5)))
    print("bootstrap", summ["bootstrap"])

    # ---- nulls (score test, D fixed at the null time-of-day pattern)
    Cr = Cnet[rows_net]
    Nr = Cr.sum(1)
    resid0 = Cr - Nr[:, None] * p_hod[None, :]

    def score_beta(Xr):
        U = (resid0 * Xr).sum()
        mX = Xr @ p_hod
        I = (Nr * ((Xr ** 2) @ p_hod - mX ** 2)).sum()
        return U / I, U / np.sqrt(I)
    NNULL = 1000
    nulls = {}
    alldays = np.arange(ND)
    for k in ["tau", "sn", "cfs0.4", "tau_OL"]:
        Xm = M(k)
        b_obs, z_obs = score_beta(Xm[rows_net])
        bs_shift, bs_scr = [], []
        for _ in range(NNULL):
            kk = rng.integers(1, 401, len(rows_net)) * rng.choice([-1, 1], len(rows_net))
            tgt = rows_net + kk
            tgt = np.where((tgt < 0) | (tgt >= ND), rows_net - kk, tgt)
            bs_shift.append(score_beta(Xm[tgt])[0])
        xs = X[k] - X[k].mean()
        Fx = np.fft.rfft(xs)
        for _ in range(NNULL // 4):
            ph = np.exp(2j * np.pi * rng.random(len(Fx)))
            ph[0] = 1
            xsur = np.fft.irfft(np.abs(Fx) * ph, n=len(xs)).reshape(ND, J)
            bs_scr.append(score_beta(xsur[rows_net])[0])
        bs_shift, bs_scr = np.array(bs_shift), np.array(bs_scr)
        nulls[k] = dict(beta_onestep=float(b_obs), z_poisson=float(z_obs),
                        dayshift_null_sd=float(bs_shift.std()), dayshift_null_95abs=float(np.percentile(np.abs(bs_shift), 95)),
                        dayshift_p=float((1 + np.sum(np.abs(bs_shift) >= abs(b_obs))) / (NNULL + 1)),
                        dayshift_z=float(b_obs / bs_shift.std()),
                        scramble_null_sd=float(bs_scr.std()), scramble_null_95abs=float(np.percentile(np.abs(bs_scr), 95)),
                        scramble_p=float((1 + np.sum(np.abs(bs_scr) >= abs(b_obs))) / (len(bs_scr) + 1)),
                        scramble_z=float(b_obs / bs_scr.std()))
        print("null", k, nulls[k])
    summ["nulls"] = nulls
    # null for the tau-vs-sn model discrimination: dLL difference under day shifts is not needed; use bootstrap of dLL
    dd = []
    for b in range(50):
        ft = cfit(Cnet[rows_net], [M("tau")[rows_net]], w=W[b], theta0=fits["tau"]["theta"], iters=8)
        fn = cfit(Cnet[rows_net], [M("sn")[rows_net]], w=W[b], theta0=fits["sn"]["theta"], iters=8)
        dd.append(ft["LL"] - fn["LL"])
    summ["dLL_tau_minus_sn_bootstrap"] = dict(obs=float(fits["tau"]["LL"] - fits["sn"]["LL"]),
                                              sd=float(np.std(dd)), frac_le0=float(np.mean(np.array(dd) <= 0)))

    # ---- deciles
    dec = []
    pex = Nr[:, None] * p_hod[None, :]
    for k in ["tau", "sn", "cfs0.4", f"cfs{best_mu['body']['mu_best']:.1f}" if f"cfs{best_mu['body']['mu_best']:.1f}" in X else "cfs0.1", "tau_OL", "sn_OL"]:
        xr = M(k)[rows_net]
        edges = np.quantile(xr, np.linspace(0, 1, 11), method="linear", weights=None) if False else np.quantile(xr, np.linspace(0, 1, 11))
        q = np.clip(np.searchsorted(edges, xr, side="right") - 1, 0, 9)
        obs = np.array([[Cr[r][q[r] == i].sum() for i in range(10)] for r in range(len(rows_net))])
        exp_ = np.array([[pex[r][q[r] == i].sum() for i in range(10)] for r in range(len(rows_net))])
        xm = np.array([(pex * xr * (q == i)).sum() / (pex * (q == i)).sum() for i in range(10)])
        ratio = obs.sum(0) / exp_.sum(0)
        bt = np.array([(W[b] @ obs) / (W[b] @ exp_) for b in range(B)])
        lo, hi = np.percentile(bt, [16, 84], axis=0)
        beta = fits[k]["beta"][0] if k in fits else cfit(Cr, [xr])["beta"][0]
        for i in range(10):
            dec.append(dict(series=k, decile=i + 1, x_lo=edges[i], x_hi=edges[i + 1], x_mean_kPa=xm[i],
                            obs=obs.sum(0)[i], exp=exp_.sum(0)[i], ratio=ratio[i], ratio_lo68=lo[i], ratio_hi68=hi[i],
                            ml_exp_curve=np.exp(beta * xm[i]) / np.sum(np.exp(beta * xm) * exp_.sum(0) / exp_.sum())))
    dec = pd.DataFrame(dec)
    dec.to_csv(f"{OUT}/decile_rates_network.csv", index=False)

    netdf = pd.DataFrame(res)
    netdf.to_csv(f"{OUT}/network_model_fits.csv", index=False)

    # ---- per family
    famrows = []
    for f_ in range(NF):
        rows = np.where(Cfam[f_].sum(1) > 0)[0]
        if len(rows) < 50:
            continue
        Cf = Cfam[f_][rows].astype(float)
        fr = fam.iloc[f_]
        Gf = ocean_green(fr.lat, fr.lon, fr.depth_km)
        OLf, Rgf = ocean_series(Gf)
        f0f = cfit(Cf, [])
        r = dict(family=f_ + 1, lat=fr.lat, lon=fr.lon, depth_km=fr.depth_km, n_units=int(Cf.sum()), n_days=len(rows),
                 OL_tau_per_m=Rgf["tau"], OL_sn_per_m=Rgf["sn"])
        for name, xs_ in {"tau": [M("tau")], "sn": [M("sn")], "cfs0.4": [M("cfs0.4")],
                          "tau_OL": [(X["tau"] + OLf["tau"]).reshape(ND, J)],
                          "tau+sn": [M("tau"), M("sn")]}.items():
            ff = cfit(Cf, [x_[rows] for x_ in xs_])
            Vw, _ = robust_se(ff, rows // 7)
            r[f"{name}_beta"] = float(ff["beta"][0])
            r[f"{name}_se"] = float(np.sqrt(Vw[0, 0]))
            r[f"{name}_dLL"] = ff["LL"] - f0f["LL"]
            if name == "tau+sn":
                r["tau+sn_beta_sn"] = float(ff["beta"][1])
                r["tau+sn_se_sn"] = float(np.sqrt(Vw[1, 1]))
                r["mu_eff"] = float(ff["beta"][1] / ff["beta"][0])
        # Hilbert phase lag vs tau
        xs = X["tau"] - X["tau"].mean()
        hx = np.imag(hilbert(xs)).reshape(ND, J)
        ff = cfit(Cf, [M("tau")[rows], hx[rows]])
        Vw, _ = robust_se(ff, rows // 7)
        bc, bs = ff["beta"]
        gph = np.array([-bs, bc]) / (bc ** 2 + bs ** 2)
        r["tau_phase_lag_deg"] = float(np.degrees(np.arctan2(bs, bc)))
        r["tau_phase_lag_se"] = float(np.degrees(np.sqrt(gph @ Vw @ gph)))
        r["tau_amp_beta"] = float(np.hypot(bc, bs))
        # day-shift score null for tau
        p_f = f0f["p"][0]
        Nf = Cf.sum(1)
        rf = Cf - Nf[:, None] * p_f[None, :]

        def sb(Xr):
            mX = Xr @ p_f
            return (rf * Xr).sum() / (Nf * ((Xr ** 2) @ p_f - mX ** 2)).sum()
        Xm = M("tau")
        bo = sb(Xm[rows])
        nb = []
        for _ in range(300):
            kk = rng.integers(1, 401, len(rows)) * rng.choice([-1, 1], len(rows))
            tgt = rows + kk
            tgt = np.where((tgt < 0) | (tgt >= ND), rows - kk, tgt)
            nb.append(sb(Xm[tgt]))
        nb = np.array(nb)
        r["tau_dayshift_p"] = float((1 + np.sum(np.abs(nb) >= abs(bo))) / 301)
        r["tau_dayshift_z"] = float(bo / nb.std())
        r["a_sigma_tau_kPa"] = 1 / r["tau_beta"] if r["tau_beta"] > 0 else np.nan
        r["a_sigma_cfs0.4_kPa"] = 1 / r["cfs0.4_beta"] if r["cfs0.4_beta"] > 0 else np.nan
        famrows.append(r)
        print("fam", f_ + 1, r["n_units"], round(r["tau_beta"], 3), round(r["tau_se"], 3), round(r["tau_phase_lag_deg"], 1),
              round(r["mu_eff"], 2), r["tau_dayshift_p"])
    fdf = pd.DataFrame(famrows)
    # along-strike / down-dip coordinates
    xE = (fdf.lon - cen["lon"]) * 111.195 * np.cos(np.deg2rad(cen["lat"]))
    yN = (fdf.lat - cen["lat"]) * 111.195
    fs = np.deg2rad(STRIKE)
    fdf["along_strike_km"] = xE * np.sin(fs) + yN * np.cos(fs)
    fdf["horiz_downdip_km"] = xE * np.sin(fs + np.pi / 2) + yN * np.cos(fs + np.pi / 2)
    fdf.to_csv(f"{OUT}/family_fits.csv", index=False)
    good = fdf[fdf.n_units >= 300]
    cor = {}
    for var in ["depth_km", "along_strike_km", "horiz_downdip_km", "lon"]:
        for y in ["tau_beta", "cfs0.4_beta", "tau_phase_lag_deg"]:
            rho, p = spearmanr(good[var], good[y])
            w = 1 / good[y.replace("_beta", "_se") if y.endswith("_beta") else "tau_phase_lag_se"] ** 2
            A_ = np.c_[np.ones(len(good)), good[var]]
            Wm = np.diag(w)
            cov = np.linalg.inv(A_.T @ Wm @ A_)
            coef = cov @ A_.T @ Wm @ good[y]
            chi2 = float(((good[y] - A_ @ coef) ** 2 * w).sum())
            cor[f"{y}_vs_{var}"] = dict(spearman=float(rho), p=float(p), slope=float(coef[1]),
                                        slope_se=float(np.sqrt(cov[1, 1])), slope_se_scaled=float(np.sqrt(cov[1, 1] * max(chi2 / (len(good) - 2), 1))),
                                        chi2_red=chi2 / (len(good) - 2))
    w = 1 / good.tau_se ** 2
    mb = float((good.tau_beta * w).sum() / w.sum())
    chi2 = float(((good.tau_beta - mb) ** 2 * w).sum())
    summ["family"] = dict(n_fam=int(len(good)), tau_beta_wmean=mb, tau_beta_chi2_const=chi2, dof=int(len(good) - 1),
                          a_sigma_tau_median_kPa=float(np.nanmedian(good.a_sigma_tau_kPa)),
                          a_sigma_tau_IQR=[float(np.nanpercentile(good.a_sigma_tau_kPa, 25)), float(np.nanpercentile(good.a_sigma_tau_kPa, 75))],
                          n_tau_beta_pos=int((good.tau_beta > 0).sum()), n_dayshift_p05=int((good.tau_dayshift_p < 0.05).sum()),
                          lag_circmean_deg=float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(good.tau_phase_lag_deg)))))),
                          mu_eff_median=float(np.median(good.mu_eff)), correlations=cor)
    print(json.dumps(summ["family"], indent=1))

    summ["network_fits"] = netdf.replace({np.nan: None, np.inf: None}).to_dict(orient="records")
    summ["assumptions"] = [
        "Degree-2 potential only (degree-3 lunar ~1.7% omitted); spherical Earth, Love h=0.6, Shida l=0.08 (no latitude/frequency dependence, no FCN).",
        "Body-tide stress = plane stress at the surface (sigma_zz=sigma_xz=sigma_yz=0), assumed constant with depth to ~28 km (first-order; tidal sigma_zz at 25 km is not zero in reality, order 0.3 kPa).",
        "mu=30 GPa, nu=0.25; no pore-pressure coupling (drained), no poroelastic B.",
        "Fault strike 293, dip 40 NE (Shelly et al. 2026), rake 180 = right-lateral; LL = sign flip of tau. With body-tide-only plane stress the dip-slip shear is collinear with sigma_n, and dip only rescales tau (sin d) and sigma_n (sin^2 d).",
        "Ocean loading NOT in the primary model; rough Boussinesq estimate with assumed uniform ocean constants " + json.dumps(OCEAN) + " (amplitude/phase uncertain by ~0.1 m / ~30 deg); no SAL, no ocean pore-pressure diffusion.",
        "Event times UTC; TT-UTC=67.6 s constant; UT1=UTC.",
        "LFE units as in sci_tides.py (meancc>=0.137, then first event per family burst with gap>1 h), 2008-01-01 to 2026-06-30.",
        "Conditional likelihood (day fixed effects + 24 h time-of-day term) removes slow rate changes and the cultural-noise detection cycle; S2 stress is absorbed by the time-of-day term."]
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summ, fh, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8})
    plot_model()

    # Fig 2: LFE response
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    cols = {"tau": C1, "sn": C2, "cfs0.4": C3, "tau_OL": C4}
    for k, c in cols.items():
        g = dec[dec.series == k]
        ax[0, 0].errorbar(g.x_mean_kPa, g.ratio, yerr=[g.ratio - g.ratio_lo68, g.ratio_hi68 - g.ratio], fmt="o", ms=3,
                          color=c, label=k)
        xx = np.linspace(g.x_mean_kPa.min(), g.x_mean_kPa.max(), 50)
        b = netdf.set_index("model").loc[k, "beta_0"]
        norm = np.sum(np.exp(b * g.x_mean_kPa) * g.exp) / g.exp.sum()
        ax[0, 0].plot(xx, np.exp(b * xx) / norm, color=c, lw=1)
    ax[0, 0].axhline(1, color=GREY, lw=0.5)
    ax[0, 0].set_xlabel("stress decile mean (kPa)")
    ax[0, 0].set_ylabel("LFE rate / expected")
    ax[0, 0].legend(frameon=False)
    ax[0, 0].set_title("Rate vs stress deciles (lines: ML exp fits)")
    for tag, c in [("body", C1), ("body_d45", GREY), ("OL_M2", C4), ("OL_M2K1O1", C2)]:
        g = prof[prof.variant == tag]
        ax[0, 1].plot(g.mu, g.dLL, color=c, label=tag)
    ax[0, 1].set_xlabel("μ' in CFS = τ + μ'σn")
    ax[0, 1].set_ylabel("ΔlogL vs no-tide")
    ax[0, 1].legend(frameon=False)
    ax[0, 1].set_title("Friction profile (network)")
    for k, c in [("tau", C1), ("sn", C2), ("cfs0.4", C3), ("tau_OL", C4)]:
        lp = np.array(lagres[k]["time_lag_profile"])
        ax[1, 0].plot(lp[:, 0], lp[:, 1], color=c, label=k)
    ax[1, 0].set_xlabel("LFE delay after stress (h)")
    ax[1, 0].set_ylabel("ΔlogL")
    ax[1, 0].legend(frameon=False)
    ax[1, 0].set_title("Time-lag scan")
    ax[1, 1].plot(scan.strike, scan.dLL, color=C1)
    ax[1, 1].axvline(STRIKE, color=C2, ls="--", lw=0.8)
    ax[1, 1].axvline(STRIKE - 180, color=C2, ls="--", lw=0.8)
    ax[1, 1].set_xlabel("strike of RL plane (deg), body tide")
    ax[1, 1].set_ylabel("ΔlogL")
    ax[1, 1].set_title("Strike scan, RL shear (sign of β flips at +180°)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tidal_stress_response.png", dpi=150)
    plt.close(fig)

    plot_families()
    print("done")


def plot_model():
    """Fig 1 from the hourly CSV: tidal gravity and resolved stresses, 2020-01-01..02-15."""
    plt.rcParams.update({"font.size": 8})
    h = pd.read_csv(f"{OUT}/tidal_stress_hourly.csv.gz")
    h["t"] = pd.to_datetime(h.time_utc)
    h = h[(h.t >= "2020-01-01") & (h.t < "2020-02-15")]
    s = json.load(open(f"{OUT}/summary.json"))["validation"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    ax[0].plot(h.t, h.dg_uGal, color=C1, lw=0.8)
    ax[0].set_ylabel("tidal gravity (µGal, δ=1.16)")
    eq = s["equilibrium_tide_site"]["M2"]
    ax[0].set_title(f"Body tide at {LAT0}N {-LON0}W: M2 gravity {s['gravity_uGal']['M2']:.1f} µGal; "
                    f"M2 W2/g {eq['model_m']:.4f} m (textbook {eq['textbook_m']:.4f} m)", fontsize=8)
    ax[1].plot(h.t, h.tau_RL_d40_kPa, color=C1, lw=0.8, label="τ (RL, slip direction)")
    ax[1].plot(h.t, h.sn_d40_kPa, color=C2, lw=0.8, label="σn (unclamping +)")
    ax[1].plot(h.t, h["cfs0.1_d40_kPa"], color=C3, lw=0.8, label="CFS μ'=0.1")
    ax[1].set_ylabel("body-tide stress (kPa)")
    ax[1].legend(ncol=3, frameon=False, loc="upper right")
    ax[1].set_title("Strike 293°, dip 40° NE; plane stress, assumed constant with depth", fontsize=8)
    hh = h[h.t < "2020-01-08"]
    ax[2].plot(hh.t, hh.tau_RL_d40_kPa, color=C1, lw=1.0, label="body τ")
    ax[2].plot(hh.t, hh.OLM2_tau_kPa_rough, color=C1, lw=0.8, ls="--", label="ocean-load τ (M2 only, ROUGH)")
    ax[2].plot(hh.t, hh.OLM2_sn_kPa_rough, color=C2, lw=0.8, ls="--", label="ocean-load σn (M2 only, ROUGH)")
    ax[2].set_ylabel("kPa")
    ax[2].legend(ncol=3, frameon=False, loc="upper right")
    ax[2].set_title("First week: rough ocean-load M2 stress is comparable to and nearly opposite the body-tide τ", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tidal_stress_model.png", dpi=150)
    plt.close(fig)


def plot_families():
    plt.rcParams.update({"font.size": 8})
    g = pd.read_csv(f"{OUT}/family_fits.csv")
    g = g[g.n_units >= 300]
    pure = g.mu_eff < 0.15
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, var, lab in [(ax[0], "depth_km", "depth (km)"), (ax[1], "along_strike_km", "along strike (km, + toward 293°)")]:
        for m, c, l in [(pure, C1, "μ_eff<0.15 (shear only)"), (~pure, C2, "μ_eff≥0.15")]:
            a.errorbar(g[var][m], g.tau_beta[m], yerr=g.tau_se[m], fmt="o", ms=3, color=c, label=l)
        a.set_xlabel(lab)
        a.set_ylabel("β = 1/(aσ) from τ (1/kPa)")
        sec = a.secondary_yaxis("right", functions=(lambda b: 1 / np.maximum(b, 1e-3), lambda x: 1 / np.maximum(x, 1e-3)))
        sec.set_ylabel("aσ (kPa)")
        sec.set_ticks([0.25, 0.33, 0.5, 1.0, 2.0])
        a.axhline(0, color=GREY, lw=0.5)
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False)
    for m, c in [(pure, C1), (~pure, C2)]:
        ax[2].errorbar(g.along_strike_km[m], g.tau_phase_lag_deg[m], yerr=g.tau_phase_lag_se[m], fmt="o", ms=3, color=c)
    ax[2].axhline(0, color=GREY, lw=0.5)
    ax[2].set_xlabel("along strike (km)")
    ax[2].set_ylabel("phase lag of LFE rate after body τ (deg)")
    ax[2].grid(alpha=0.3)
    fig.suptitle("Per-family tidal-shear sensitivity (families with ≥300 units; colour = fitted μ_eff from τ+σn)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tidal_stress_families.png", dpi=150)
    plt.close(fig)


def ol_fine():
    """Supplementary stage (python sci_tidal_stress.py --ol-fine): finer ocean-load scale/phase profile."""
    ND = (T_END - T0).days
    tg = T0 + pd.to_timedelta((np.arange(ND * J) + 0.5) * BIN_MIN, unit="min")
    jd = jd_from_ts(tg)
    P = potential_derivs(LAT0, LON0, jd)
    Sk = {k: v / 1e3 for k, v in plane_stress(body_strain(P)).items()}
    R40 = resolve(Sk, STRIKE, 40.0)
    fam = pd.read_csv(f"{ROOT}/data/families/shelly_family_locations.csv")
    Gc = ocean_green(fam.lat.mean(), fam.lon.mean(), fam.depth_km.mean())
    Rg = resolve({k: v / 1e3 for k, v in Gc.items()}, STRIKE, 40.0)
    Vt = doodson_V(jd)
    u = load_units()
    Cnet = np.zeros((ND, J))
    np.add.at(Cnet, (u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    rows = np.where(Cnet.sum(1) > 0)[0]
    C = Cnet[rows]
    LL0 = cfit(C, [])["LL"]
    out = []
    grids = [(("M2",), np.round(np.arange(0, 0.61, 0.05), 3), np.arange(-40, 41, 10)),
             (("M2", "K1", "O1"), np.round(np.arange(0, 1.51, 0.125), 3), np.arange(-40, 41, 20))]
    for consts_, scales, dphs in grids:
        for dph in dphs:
            eta = sum(OCEAN[c][0] * np.cos(np.deg2rad(Vt[c] - OCEAN[c][1] - dph)) for c in consts_)
            for s_ in scales:
                xt = (R40["tau"] + s_ * Rg["tau"] * eta).reshape(ND, J)[rows]
                xn = (R40["sn"] + s_ * Rg["sn"] * eta).reshape(ND, J)[rows]
                f2 = cfit(C, [xt, xn])
                f1 = cfit(C, [xt])
                Vw, _ = robust_se(f2, rows // 7)
                out.append(dict(ocean="+".join(consts_), dphase=int(dph), scale=float(s_), dLL_tau_sn=f2["LL"] - LL0,
                                beta_tau=float(f2["beta"][0]), beta_sn=float(f2["beta"][1]),
                                se_tau=float(np.sqrt(Vw[0, 0])), se_sn=float(np.sqrt(Vw[1, 1])),
                                mu_eff=float(f2["beta"][1] / f2["beta"][0]),
                                dLL_tau_only=f1["LL"] - LL0, beta_tau_only=float(f1["beta"][0]),
                                a_sigma_tau_only_kPa=float(1 / f1["beta"][0]) if f1["beta"][0] > 0 else None))
            print(consts_, dph, "done", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(f"{OUT}/ocean_load_scale_profile_fine.csv", index=False)
    res = {}
    for oc, g in df.groupby("ocean"):
        i = g.dLL_tau_sn.idxmax()
        ok = g[g.dLL_tau_sn >= g.dLL_tau_sn.max() - 3.0]   # ~95% joint (2 extra params)
        res[oc] = dict(best=g.loc[i].to_dict(), scale_range_dLL3=[float(ok.scale.min()), float(ok.scale.max())],
                       dphase_range_dLL3=[int(ok.dphase.min()), int(ok.dphase.max())],
                       mu_eff_range_dLL3=[float(ok.mu_eff.min()), float(ok.mu_eff.max())],
                       a_sigma_tau_only_range=[float(1 / ok.beta_tau_only.max()), float(1 / ok.beta_tau_only.min())],
                       dLL_scale0=float(g[g.scale == 0].dLL_tau_sn.iloc[0]),
                       dLL_max_scale_dphase0={str(float(g.scale.max())): float(g[(g.scale == g.scale.max()) & (g.dphase == 0)].dLL_tau_sn.iloc[0])})
    print(json.dumps(res, indent=1, default=float))
    sp = f"{OUT}/summary.json"
    summ = json.load(open(sp))
    summ["ocean_load_fine_profile"] = res
    with open(sp, "w") as fh:
        json.dump(summ, fh, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))


if __name__ == "__main__":
    import sys
    if "--ol-fine" in sys.argv:
        ol_fine()
    elif "--figs" in sys.argv:
        plot_model()
        plot_families()
    else:
        main()
