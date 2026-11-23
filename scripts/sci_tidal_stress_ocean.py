#!/usr/bin/env python
"""sci_tidal_stress_ocean.py - Ocean-tide loading stress at the MTJ LFEs constrained by NOAA tide gauges,
added to the body-tide stress of sci_tidal_stress.py, and the refit of the LFE response.

Ocean tide field (assumptions, stated):
  * NOAA CO-OPS harmonic constants (raw JSON in data/sci/tidal_stress/noaa/).  phase_GMT is the Greenwich
    phase lag G (checked: phase_local = G - speed*8 h, i.e. kappa' referred to UTC-8), so
    eta(t) = f A cos(V(t) + u - G) with Schureman V at Greenwich (validated in sci_tidal_stress.py).
  * Constants for M2 S2 N2 K2 K1 O1 P1 Q1 from 11 open-coast / bay-entrance gauges 32.9-48.4 N
    (Toke Point excluded: inside Willapa Bay).  Complex (A e^{-iG}) interpolation in latitude along
    the coast; constant beyond the end gauges.
  * Offshore: field constant along each latitude (primary, "uniform"); variant "decay": amplitude
    decreasing linearly to 70% at 500 km offshore (phase unchanged); variant "near150": loads within
    150 km of each family only.
  * Coastline: polygon of known coastal points 32.5-49 N (lon interpolated in latitude).
  * Load: Boussinesq point loads on a homogeneous half-space (mu 30 GPa, nu 0.25), full 3-D stress at each
    family's depth (interior solution, via displacement gradients).  Domain out to 1200 km; the half-space
    far field of a one-sided load diverges logarithmically, so the radial convergence is reported and the
    primary cut is R = 1000 km (sphericity and the real deep-ocean phase rotation are NOT modelled).
  * Nodal corrections f, u (Schureman / Pugh approximations) applied to the ocean constituents.
  * Body tide: as sci_tidal_stress.py (plane stress, constant with depth).
Outputs: data/sci/tidal_stress/ocean_*; figures/sci_tidal_stress_ocean_*.png
"""
import os
import sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sci_tidal_stress as T  # noqa: E402

OUT = T.OUT
FIG = T.FIG
NOAA = f"{OUT}/noaa"
CONS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
EXCLUDE = {"9440910"}  # Toke Point (Willapa Bay, estuarine)
RCUT = 1000e3
rng = np.random.default_rng(20261124)

COAST = np.array([
    (32.5, -117.12), (32.87, -117.26), (33.2, -117.4), (33.6, -117.9), (33.75, -118.4), (34.0, -118.5),
    (34.4, -119.7), (34.45, -120.47), (34.6, -120.64), (35.17, -120.76), (35.7, -121.3), (36.3, -121.9),
    (36.6, -121.95), (37.0, -122.2), (37.5, -122.5), (37.8, -122.52), (38.0, -123.02), (38.3, -123.07),
    (38.51, -123.24), (38.77, -123.53), (38.95, -123.74), (39.31, -123.80), (39.44, -123.81),
    (39.64, -123.78), (39.74, -123.82), (39.83, -123.85), (39.92, -123.93), (39.94, -123.98),
    (40.02, -124.07), (40.13, -124.20), (40.25, -124.35), (40.44, -124.41), (40.52, -124.39),
    (40.64, -124.31), (40.77, -124.24), (41.06, -124.15), (41.29, -124.07), (41.55, -124.08),
    (41.75, -124.20), (41.78, -124.25), (42.0, -124.21), (42.84, -124.56), (43.4, -124.3),
    (44.0, -124.12), (44.6, -124.07), (45.5, -123.96), (46.25, -124.07), (47.0, -124.17),
    (47.9, -124.65), (48.38, -124.72), (49.0, -125.7)])


def coast_lon(lat):
    return np.interp(lat, COAST[:, 0], COAST[:, 1])


def load_gauges():
    rows = []
    for f in sorted(glob.glob(f"{NOAA}/harcon_*.json")):
        sid = os.path.basename(f)[7:14]
        st = json.load(open(f"{NOAA}/station_{sid}.json"))["stations"][0]
        h = {c["name"]: c for c in json.load(open(f))["HarmonicConstituents"]}
        r = dict(id=sid, name=st["name"], lat=float(st["lat"]), lon=float(st["lng"]), used=sid not in EXCLUDE)
        for c in CONS:
            r[f"{c}_A"] = h[c]["amplitude"]
            r[f"{c}_G"] = h[c]["phase_GMT"]
            r[f"{c}_local"] = h[c]["phase_local"]
            r[f"{c}_speed"] = h[c]["speed"]
        rows.append(r)
    g = pd.DataFrame(rows).sort_values("lat").reset_index(drop=True)
    # convention check: local = G - speed*8 (UTC-8 standard meridian)
    chk = [((r[f"{c}_G"] - r[f"{c}_speed"] * 8 - r[f"{c}_local"] + 180) % 360 - 180) for _, r in g.iterrows() for c in CONS]
    return g, float(np.max(np.abs(chk)))


def tide_field(g, lat, dist_off, variant):
    """Complex tide Z = A e^{-iG} (m) for each constituent at cells (lat, distance offshore m)."""
    gu = g[g.used]
    out = {}
    for c in CONS:
        z = gu[f"{c}_A"].to_numpy() * np.exp(-1j * np.deg2rad(gu[f"{c}_G"].to_numpy()))
        zr = np.interp(lat, gu.lat, z.real)
        zi = np.interp(lat, gu.lat, z.imag)
        Z = zr + 1j * zi
        if variant == "decay":
            Z = Z * (1 - 0.3 * np.minimum(dist_off / 500e3, 1.0))
        out[c] = Z
    return out


def cells():
    """Variable-resolution ocean cells: 0.01 deg near the LFEs, 0.05 deg elsewhere (31-49 N, to 140 W)."""
    out = []
    fine = (38.6, 41.6, -126.2)
    for dd, la0, la1, lo0 in ((0.01, fine[0], fine[1], fine[2]), (0.05, 31.0, 49.0, -140.0)):
        lats = np.arange(la0 + dd / 2, la1, dd)
        lons = np.arange(lo0 + dd / 2, -117.0, dd)
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        m = LO < coast_lon(LA)
        if dd == 0.05:
            m &= ~((LA > fine[0]) & (LA < fine[1]) & (LO > fine[2]))
        la, lo = LA[m], LO[m]
        out.append((la, lo, (dd * 111.195e3) ** 2 * np.cos(np.deg2rad(la))))
    la = np.concatenate([o[0] for o in out])
    lo = np.concatenate([o[1] for o in out])
    ar = np.concatenate([o[2] for o in out])
    doff = (coast_lon(la) - lo) * 111.195e3 * np.cos(np.deg2rad(la))
    return la, lo, ar, doff


def cell_tensors(lat0, lon0, depth_km, la, lo, ar, h=5.0):
    """Per-cell stress tensor (N,E,D; Pa per Pa of surface pressure) at the receiver.  Returns dict of arrays + R."""
    p1, p2 = np.deg2rad(lat0), np.deg2rad(la)
    dl = np.deg2rad(lo - lon0)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    R = 2 * T.A_E * np.arcsin(np.sqrt(a))
    az = np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))
    xs, ys = R * np.sin(az), R * np.cos(az)   # east, north of receiver
    z = depth_km * 1e3
    P = ar  # 1 Pa pressure * area

    def U(dx, dy, dz):
        return T.bouss_disp(dx - xs, dy - ys, z + dz, P)
    G = np.empty((3, 3, len(la)))
    for j, (dx, dy, dz) in enumerate(((h, 0, 0), (0, h, 0), (0, 0, h))):
        up = U(dx, dy, dz)
        um = U(-dx, -dy, -dz)
        for i in range(3):
            G[i, j] = (up[i] - um[i]) / (2 * h)
    eps = 0.5 * (G + G.transpose(1, 0, 2))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = 2 * T.MU * eps
    for i in range(3):
        sig[i, i] += T.LAM3 * tr
    # E,N,D -> N,E,D
    S = dict(snn=sig[1, 1], see=sig[0, 0], sdd=sig[2, 2], sne=sig[1, 0], snd=sig[1, 2], sed=sig[0, 2])
    return S, R


def nodal(jd):
    Tc = (jd - 2451545.0) / 36525.0
    N = np.deg2rad(125.0445 - 1934.1363 * Tc)
    c, s = np.cos, np.sin
    fu = {}
    fM2 = 1.0004 - 0.0373 * c(N) + 0.0002 * c(2 * N)
    uM2 = -2.14 * s(N)
    fK1 = 1.0060 + 0.1150 * c(N) - 0.0088 * c(2 * N) + 0.0006 * c(3 * N)
    uK1 = -8.86 * s(N) + 0.68 * s(2 * N) - 0.07 * s(3 * N)
    fO1 = 1.0089 + 0.1871 * c(N) - 0.0147 * c(2 * N) + 0.0014 * c(3 * N)
    uO1 = 10.80 * s(N) - 1.34 * s(2 * N) + 0.19 * s(3 * N)
    fK2 = 1.0241 + 0.2863 * c(N) + 0.0083 * c(2 * N) - 0.0015 * c(3 * N)
    uK2 = -17.74 * s(N) + 0.68 * s(2 * N) - 0.04 * s(3 * N)
    one, zero = np.ones_like(N), np.zeros_like(N)
    fu = dict(M2=(fM2, uM2), N2=(fM2, uM2), S2=(one, zero), K2=(fK2, uK2), K1=(fK1, uK1),
              O1=(fO1, uO1), Q1=(fO1, uO1), P1=(one, zero))
    return fu


def ol_complex(Scell, R, Z, rcut=RCUT):
    """Complex stress tensor (kPa) per constituent: sum over cells with R<rcut of rho g Z * tensor."""
    m = R < rcut
    out = {}
    for c in CONS:
        q = T.RHO_W * T.G0 * Z[c][m] / 1e3
        out[c] = {k: np.sum(v[m] * q) for k, v in Scell.items()}
    return out


def ol_series(Sc, Vt, fu, strike=T.STRIKE, dip=40.0):
    """Time series of resolved OL stresses (kPa)."""
    acc = {k: 0.0 for k in ("tau", "sn", "sss", "mean")}
    for c in CONS:
        Rc = T.resolve(Sc[c], strike, dip)
        f, u = fu[c]
        ph = np.exp(1j * np.deg2rad(Vt[c] + u))
        for k in acc:
            acc[k] = acc[k] + np.real(Rc[k] * f * ph)
    return acc


def main():
    summ = {}
    g, chk = load_gauges()
    g.to_csv(f"{OUT}/ocean_noaa_constants.csv", index=False)
    summ["noaa_phase_convention_check_maxdev_deg"] = chk
    summ["noaa_phase_convention"] = "phase_GMT = Greenwich lag G; phase_local = G - speed*8h (UTC-8). Used G."
    print(g[["id", "name", "lat", "lon", "used", "M2_A", "M2_G", "K1_A", "K1_G", "O1_A", "O1_G"]].to_string(), "\nconv check", chk)

    la, lo, ar, doff = cells()
    print("cells", len(la))
    fam = pd.read_csv(f"{T.ROOT}/data/families/shelly_family_locations.csv").sort_values("idnum").reset_index(drop=True)
    cen = dict(lat=fam.lat.mean(), lon=fam.lon.mean(), depth=fam.depth_km.mean())
    Zs = {v: tide_field(g, la, doff, v) for v in ("uniform", "decay")}

    # ---- convergence at the centroid (M2, resolved)
    Scen, Rcen = cell_tensors(cen["lat"], cen["lon"], cen["depth"], la, lo, ar)
    conv = []
    for rc in (50e3, 100e3, 150e3, 200e3, 300e3, 500e3, 700e3, 1000e3, 1200e3, 1500e3):
        for v in ("uniform", "decay"):
            Sc = ol_complex(Scen, Rcen, Zs[v], rc)
            for c in ("M2", "K1", "O1"):
                Rr = T.resolve(Sc[c], T.STRIKE, 40.0)
                conv.append(dict(variant=v, R_km=rc / 1e3, const=c,
                                 **{f"{k}_amp": float(np.abs(Rr[k])) for k in ("tau", "sn", "mean")},
                                 **{f"{k}_Gphase": float(np.rad2deg(-np.angle(Rr[k])) % 360) for k in ("tau", "sn", "mean")}))
    conv = pd.DataFrame(conv)
    conv.to_csv(f"{OUT}/ocean_convergence.csv", index=False)
    print(conv[conv.const == "M2"].round(3).to_string())

    # ---- time grid, body stress
    ND = (T.T_END - T.T0).days
    J = T.J
    tg = T.T0 + pd.to_timedelta((np.arange(ND * J) + 0.5) * T.BIN_MIN, unit="min")
    jd = T.jd_from_ts(tg)
    th_h = (tg - T.TREF).total_seconds().to_numpy() / 3600.0
    P = T.potential_derivs(T.LAT0, T.LON0, jd)
    Sk = {k: v / 1e3 for k, v in T.plane_stress(T.body_strain(P)).items()}
    body = T.resolve(Sk, T.STRIKE, 40.0)
    body45 = T.resolve(Sk, T.STRIKE, 45.0)
    Vt = T.doodson_V(jd)
    fu = nodal(jd)

    variants = {"uniform_R1000": ("uniform", RCUT), "decay_R1000": ("decay", RCUT),
                "uniform_R150": ("uniform", 150e3), "uniform_R500": ("uniform", 500e3)}
    OLc = {name: ol_series(ol_complex(Scen, Rcen, Zs[v], rc), Vt, fu) for name, (v, rc) in variants.items()}
    OLc45 = ol_series(ol_complex(Scen, Rcen, Zs["uniform"], RCUT), Vt, fu, dip=45.0)

    # harmonics of total stress vs LFE phases
    sub = slice(None, None, 3)
    hrows = []
    series = {"body_tau": body["tau"], "body_sn": body["sn"]}
    for name in variants:
        series[f"OL_{name}_tau"] = OLc[name]["tau"]
        series[f"OL_{name}_sn"] = OLc[name]["sn"]
        series[f"tot_{name}_tau"] = body["tau"] + OLc[name]["tau"]
        series[f"tot_{name}_sn"] = body["sn"] + OLc[name]["sn"]
    for k, x in series.items():
        h = T.harmonic_fit(th_h[sub], x[sub], ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "Mf", "Mm"])
        for c in ("M2", "N2", "O1", "K1"):
            lag = (T.LFE_PHASE[c] - h[c][1] + 180) % 360 - 180
            hrows.append(dict(series=k, const=c, amp_kPa=h[c][0], phase_peak_deg=h[c][1],
                              lfe_minus_stress_deg=lag, lag_h=lag / 360 * T.PER_H[c]))
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(f"{OUT}/ocean_stress_harmonics.csv", index=False)
    print(hdf[hdf.series.str.contains("body|uniform_R1000")].round(3).to_string())
    summ["harmonics_vs_LFE"] = hdf[hdf.series.str.contains("body|uniform_R1000|decay")].to_dict(orient="records")

    # ---- LFE units
    u = T.load_units()
    Cnet = np.zeros((ND, J))
    np.add.at(Cnet, (u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    Cfam = np.zeros((T.NF, ND, J), np.float32)
    np.add.at(Cfam, (u.fam.to_numpy(), u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    rows = np.where(Cnet.sum(1) > 0)[0]
    C = Cnet[rows]
    f0 = T.cfit(C, [])
    LL0 = f0["LL"]

    def M(x, rr=rows):
        return x.reshape(ND, J)[rr]

    def fit1(xs, CC=C, rr=rows):
        f = T.cfit(CC, [M(x, rr) for x in xs])
        Vw, _ = T.robust_se(f, rr // 7)
        return f, Vw

    res, prof = [], []
    tot = {"body": (body, None)}
    for name in variants:
        tot[name] = ({k: body[k] + OLc[name][k] for k in ("tau", "sn", "sss", "mean")}, OLc[name])
    tot["uniform_R1000_dip45"] = ({k: body45[k] + OLc45[k] for k in ("tau", "sn", "sss", "mean")}, OLc45)
    for name, (X, _) in tot.items():
        for model, keys in (("tau", ["tau"]), ("sn", ["sn"]), ("mean", ["mean"]), ("tau+sn", ["tau", "sn"]),
                            ("tau+sn+sss", ["tau", "sn", "sss"])):
            f, Vw = fit1([X[k] for k in keys])
            r = dict(stress=name, model=model, dLL=f["LL"] - LL0)
            for i, k in enumerate(keys):
                r[f"beta_{k}"] = float(f["beta"][i])
                r[f"se_{k}"] = float(np.sqrt(Vw[i, i]))
            if model == "tau":
                r["a_sigma_kPa"] = 1 / r["beta_tau"] if r["beta_tau"] > 0 else None
                r["sense"] = "RL" if r["beta_tau"] > 0 else "LL"
            if model == "tau+sn":
                b0, b1 = f["beta"][:2]
                gr = np.array([-b1 / b0 ** 2, 1 / b0])
                r["mu_eff"] = float(b1 / b0)
                r["mu_eff_se"] = float(np.sqrt(gr @ Vw[:2, :2] @ gr))
                r["sense"] = "RL" if b0 > 0 else "LL"
                r["a_sigma_cfs_kPa"] = float(1 / b0) if b0 > 0 else None
            res.append(r)
        for mu in np.round(np.arange(-0.6, 1.21, 0.05), 3):
            for sense, sg in (("RL", 1.0), ("LL", -1.0)):
                f = T.cfit(C, [M(sg * X["tau"] + mu * X["sn"])])
                prof.append(dict(stress=name, sense=sense, mu=mu, beta=float(f["beta"][0]), dLL=f["LL"] - LL0))
        # phase lag (Hilbert) vs tau and vs best CFS
        xs = X["tau"] - X["tau"].mean()
        f = T.cfit(C, [M(X["tau"]), M(np.imag(hilbert(xs)))])
        Vw, _ = T.robust_se(f, rows // 7)
        bc, bs = f["beta"]
        gph = np.array([-bs, bc]) / (bc ** 2 + bs ** 2)
        res.append(dict(stress=name, model="tau_hilbert", dLL=f["LL"] - LL0, lag_deg=float(np.degrees(np.arctan2(bs, bc))),
                        lag_se=float(np.degrees(np.sqrt(gph @ Vw @ gph))), beta_amp=float(np.hypot(bc, bs))))
        print(name, [ (r["model"], round(r["dLL"], 1), r.get("beta_tau"), r.get("mu_eff"), r.get("lag_deg")) for r in res if r["stress"] == name])
    res = pd.DataFrame(res)
    res.to_csv(f"{OUT}/ocean_network_fits.csv", index=False)
    prof = pd.DataFrame(prof)
    prof.to_csv(f"{OUT}/ocean_mu_profile.csv", index=False)
    best = {}
    for (name, sense), gg in prof.groupby(["stress", "sense"]):
        i = gg.dLL.idxmax()
        best[f"{name}|{sense}"] = dict(mu=float(gg.mu[i]), dLL=float(gg.dLL[i]), a_sigma_kPa=float(1 / gg.beta[i]) if gg.beta[i] > 0 else None)
    summ["mu_profile_best"] = best
    print(json.dumps(best, indent=1))

    # day-shift score null for primary total tau and best CFS
    p_hod = f0["p"][0]
    Nr = C.sum(1)
    resid0 = C - Nr[:, None] * p_hod[None, :]

    def sb(Xr):
        mX = Xr @ p_hod
        return (resid0 * Xr).sum() / (Nr * ((Xr ** 2) @ p_hod - mX ** 2)).sum()
    nulls = {}
    Xp = tot["uniform_R1000"][0]
    bU = best["uniform_R1000|RL"]["mu"]
    for k, x in (("tau", Xp["tau"]), ("sn", Xp["sn"]), (f"cfs{bU}", Xp["tau"] + bU * Xp["sn"])):
        Xm = x.reshape(ND, J)
        bo = sb(Xm[rows])
        nb = []
        for _ in range(500):
            kk = rng.integers(1, 401, len(rows)) * rng.choice([-1, 1], len(rows))
            tgt = rows + kk
            tgt = np.where((tgt < 0) | (tgt >= ND), rows - kk, tgt)
            nb.append(sb(Xm[tgt]))
        nb = np.array(nb)
        nulls[k] = dict(beta1=float(bo), null_sd=float(nb.std()), z=float(bo / nb.std()),
                        p=float((1 + np.sum(np.abs(nb) >= abs(bo))) / 501))
    summ["dayshift_nulls_uniform_R1000"] = nulls
    print(nulls)

    # ---- per family (body vs body + family-specific OL, uniform R1000)
    frows = []
    for f_ in range(T.NF):
        rr = np.where(Cfam[f_].sum(1) > 0)[0]
        Cf = Cfam[f_][rr].astype(float)
        if Cf.sum() < 300:
            continue
        fr = fam.iloc[f_]
        Sf, Rf = cell_tensors(fr.lat, fr.lon, fr.depth_km, la, lo, ar)
        Scf = ol_complex(Sf, Rf, Zs["uniform"], RCUT)
        OLf = ol_series(Scf, Vt, fu)
        M2r = T.resolve(Scf["M2"], T.STRIKE, 40.0)
        f0f = T.cfit(Cf, [])
        r = dict(family=f_ + 1, lat=fr.lat, lon=fr.lon, depth_km=fr.depth_km, n_units=int(Cf.sum()),
                 OL_M2_tau_amp=float(np.abs(M2r["tau"])), OL_M2_sn_amp=float(np.abs(M2r["sn"])),
                 OL_M2_tau_G=float(np.rad2deg(-np.angle(M2r["tau"])) % 360))
        for tag, X in (("body", body), ("tot", {k: body[k] + OLf[k] for k in ("tau", "sn")})):
            for model, keys in (("tau", ["tau"]), ("sn", ["sn"]), ("tausn", ["tau", "sn"])):
                f, Vw = fit1([X[k] for k in keys], Cf, rr)
                r[f"{tag}_{model}_dLL"] = f["LL"] - f0f["LL"]
                r[f"{tag}_{model}_beta"] = float(f["beta"][0])
                r[f"{tag}_{model}_se"] = float(np.sqrt(Vw[0, 0]))
                if model == "tausn":
                    r[f"{tag}_mu_eff"] = float(f["beta"][1] / f["beta"][0])
                    r[f"{tag}_beta_sn_in_tausn"] = float(f["beta"][1])
                    r[f"{tag}_se_sn_in_tausn"] = float(np.sqrt(Vw[1, 1]))
            xs = X["tau"] - X["tau"].mean()
            f = T.cfit(Cf, [M(X["tau"], rr), M(np.imag(hilbert(xs)), rr)])
            Vw, _ = T.robust_se(f, rr // 7)
            bc, bs = f["beta"]
            gph = np.array([-bs, bc]) / (bc ** 2 + bs ** 2)
            r[f"{tag}_lag_deg"] = float(np.degrees(np.arctan2(bs, bc)))
            r[f"{tag}_lag_se"] = float(np.degrees(np.sqrt(gph @ Vw @ gph)))
        frows.append(r)
        print(f_ + 1, {k: round(v, 3) for k, v in r.items() if isinstance(v, float) and ("mu" in k or "lag_deg" in k or "tau_beta" in k or "OL_M2" in k)})
    fd = pd.DataFrame(frows)
    xE = (fd.lon - cen["lon"]) * 111.195 * np.cos(np.deg2rad(cen["lat"]))
    yN = (fd.lat - cen["lat"]) * 111.195
    s_ = np.deg2rad(T.STRIKE)
    fd["along_strike_km"] = xE * np.sin(s_) + yN * np.cos(s_)
    fd["coast_dist_km"] = (fd.lon - coast_lon(fd.lat)) * 111.195 * np.cos(np.deg2rad(fd.lat))
    fd.to_csv(f"{OUT}/ocean_family_fits.csv", index=False)
    fam_s = {}
    for tag in ("body", "tot"):
        b, se = fd[f"{tag}_tau_beta"], fd[f"{tag}_tau_se"]
        w = 1 / se ** 2
        mb = float((b * w).sum() / w.sum())
        bs_, ses_ = fd[f"{tag}_beta_sn_in_tausn"], fd[f"{tag}_se_sn_in_tausn"]
        w2 = 1 / ses_ ** 2
        ms = float((bs_ * w2).sum() / w2.sum())
        lag = fd[f"{tag}_lag_deg"]
        fam_s[tag] = dict(
            tau_beta_wmean=mb, tau_beta_chi2=float(((b - mb) ** 2 * w).sum()),
            sn_beta_wmean=ms, sn_beta_chi2=float(((bs_ - ms) ** 2 * w2).sum()),
            lag_wmean=float((lag / fd[f"{tag}_lag_se"] ** 2).sum() / (1 / fd[f"{tag}_lag_se"] ** 2).sum()),
            lag_chi2=float((((lag - (lag / fd[f"{tag}_lag_se"] ** 2).sum() / (1 / fd[f"{tag}_lag_se"] ** 2).sum()) / fd[f"{tag}_lag_se"]) ** 2).sum()),
            n_tau_pos=int((b > 0).sum()), n=int(len(fd)),
            mu_eff_median=float(np.median(fd[f"{tag}_mu_eff"])),
            a_sigma_median=float(np.median(1 / b[b > 0])),
            spearman_lag_vs_along=[float(x) for x in spearmanr(fd.along_strike_km, lag)],
            spearman_mu_vs_along=[float(x) for x in spearmanr(fd.along_strike_km, fd[f"{tag}_mu_eff"])],
            spearman_lag_vs_coastdist=[float(x) for x in spearmanr(fd.coast_dist_km, lag)],
            spearman_snbeta_vs_OLsn=[float(x) for x in spearmanr(fd.OL_M2_sn_amp, bs_)],
            sumdLL_tausn=float(fd[f"{tag}_tausn_dLL"].sum()))
    fam_s["OL_M2_tau_amp_range"] = [float(fd.OL_M2_tau_amp.min()), float(fd.OL_M2_tau_amp.max())]
    fam_s["OL_M2_sn_amp_range"] = [float(fd.OL_M2_sn_amp.min()), float(fd.OL_M2_sn_amp.max())]
    fam_s["coast_dist_range_km"] = [float(fd.coast_dist_km.min()), float(fd.coast_dist_km.max())]
    summ["family"] = fam_s
    print(json.dumps(fam_s, indent=1))
    summ["network_fits"] = res.replace({np.nan: None}).to_dict(orient="records")
    with open(f"{OUT}/ocean_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))

    # ---- figures
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    gu = g[g.used]
    latg = np.linspace(31, 49, 200)
    Zl = tide_field(g, latg, np.zeros_like(latg), "uniform")
    ax[0].plot(np.abs(Zl["M2"]), latg, color=T.C1, label="M2 amp (m)")
    ax[0].plot(np.abs(Zl["K1"]), latg, color=T.C2, label="K1 amp (m)")
    ax[0].plot(np.abs(Zl["O1"]), latg, color=T.C3, label="O1 amp (m)")
    ax[0].scatter(gu.M2_A, gu.lat, color=T.C1, s=10)
    ax[0].scatter(gu.K1_A, gu.lat, color=T.C2, s=10)
    ax[0].scatter(gu.O1_A, gu.lat, color=T.C3, s=10)
    a2 = ax[0].twiny()
    a2.plot(np.rad2deg(-np.angle(Zl["M2"])) % 360, latg, color=T.GREY, ls="--")
    a2.scatter(gu.M2_G, gu.lat, color=T.GREY, s=8)
    a2.set_xlabel("M2 Greenwich phase G (deg, dashed)")
    ax[0].axhline(T.LAT0, color="k", lw=0.5)
    ax[0].set_ylabel("latitude")
    ax[0].set_xlabel("amplitude (m)")
    ax[0].legend(frameon=False, loc="lower right")
    for v, ls in (("uniform", "-"), ("decay", "--")):
        cc = conv[(conv.variant == v) & (conv.const == "M2")]
        ax[1].plot(cc.R_km, cc.tau_amp, color=T.C1, ls=ls, label=f"τ {v}")
        ax[1].plot(cc.R_km, cc.sn_amp, color=T.C2, ls=ls, label=f"σn {v}")
    ax[1].axhline(float(hdf[(hdf.series == "body_tau") & (hdf.const == "M2")].amp_kPa.iloc[0]), color=T.C1, lw=0.6, ls=":", label="body τ M2")
    ax[1].axvline(RCUT / 1e3, color=T.GREY, lw=0.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("load radius included (km)")
    ax[1].set_ylabel("M2 ocean-load stress amplitude (kPa)")
    ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("Convergence (centroid, 27 km)")
    for name, c in (("body", T.GREY), ("uniform_R1000", T.C1), ("decay_R1000", T.C3), ("uniform_R150", T.C2)):
        for sense, ls in (("RL", "-"), ("LL", ":")):
            gg = prof[(prof.stress == name) & (prof.sense == sense)]
            ax[2].plot(gg.mu, gg.dLL, color=c, ls=ls, label=f"{name} {sense}" if sense == "RL" else None)
    ax[2].set_xlabel("μ'")
    ax[2].set_ylabel("ΔlogL (CFS; solid RL, dotted LL)")
    ax[2].legend(frameon=False, fontsize=7)
    ax[2].set_title("LFE fit vs friction, with NOAA-constrained load")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tidal_stress_ocean_load.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for tag, c, lab in (("body", T.GREY, "body only"), ("tot", T.C1, "body + ocean load")):
        ax[0].errorbar(fd.along_strike_km, fd[f"{tag}_lag_deg"], yerr=fd[f"{tag}_lag_se"], fmt="o", ms=3, color=c, label=lab)
        ax[1].errorbar(fd.along_strike_km, fd[f"{tag}_beta_sn_in_tausn"], yerr=fd[f"{tag}_se_sn_in_tausn"], fmt="o", ms=3, color=c)
        ax[2].errorbar(fd.along_strike_km, fd[f"{tag}_tau_beta"], yerr=fd[f"{tag}_tau_se"], fmt="o", ms=3, color=c)
    ax[0].set_ylabel("phase lag after τ (deg)")
    ax[1].set_ylabel("β_σn in τ+σn fit (1/kPa)")
    ax[2].set_ylabel("β_τ = 1/(aσ) (1/kPa)")
    ax[0].legend(frameon=False)
    for a in ax:
        a.set_xlabel("along strike (km, + toward 293°)")
        a.axhline(0, color="k", lw=0.4)
        a.grid(alpha=0.3)
    fig.suptitle("Per-family response: body tide vs body + NOAA-constrained ocean load (uniform, R≤1000 km)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tidal_stress_ocean_families.png", dpi=150)
    plt.close(fig)
    print("done")


def extra():
    """--extra: joint shear + mean-stress models and an undrained Coulomb variant (network)."""
    g, _ = load_gauges()
    la, lo, ar, doff = cells()
    fam = pd.read_csv(f"{T.ROOT}/data/families/shelly_family_locations.csv")
    Scen, Rcen = cell_tensors(fam.lat.mean(), fam.lon.mean(), fam.depth_km.mean(), la, lo, ar)
    ND, J = (T.T_END - T.T0).days, T.J
    tg = T.T0 + pd.to_timedelta((np.arange(ND * J) + 0.5) * T.BIN_MIN, unit="min")
    jd = T.jd_from_ts(tg)
    th_h = (tg - T.TREF).total_seconds().to_numpy() / 3600.0
    Sk = {k: v / 1e3 for k, v in T.plane_stress(T.body_strain(T.potential_derivs(T.LAT0, T.LON0, jd))).items()}
    body = T.resolve(Sk, T.STRIKE, 40.0)
    Vt, fu = T.doodson_V(jd), nodal(jd)
    u = T.load_units()
    Cn = np.zeros((ND, J))
    np.add.at(Cn, (u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    rows = np.where(Cn.sum(1) > 0)[0]
    C = Cn[rows]
    LL0 = T.cfit(C, [])["LL"]
    out = []
    for name, (v, rc) in {"uniform_R1000": ("uniform", RCUT), "decay_R1000": ("decay", RCUT),
                          "uniform_R500": ("uniform", 500e3)}.items():
        OL = ol_series(ol_complex(Scen, Rcen, tide_field(g, la, doff, v), rc), Vt, fu)
        tot = {k: body[k] + OL[k] for k in OL}
        h = T.harmonic_fit(th_h[::3], tot["mean"][::3], ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"])
        for c in ("M2", "O1", "K1"):
            out.append(dict(stress=name, model=f"harm_mean_{c}", amp_kPa=h[c][0], phase=h[c][1],
                            lfe_minus=(T.LFE_PHASE[c] - h[c][1] + 180) % 360 - 180))
        mods = {"tot_tau+tot_mean": [tot["tau"], tot["mean"]], "body_tau+tot_mean": [body["tau"], tot["mean"]],
                "body_tau+body_sn+tot_mean": [body["tau"], body["sn"], tot["mean"]],
                "tot_tau+tot_sn+tot_mean": [tot["tau"], tot["sn"], tot["mean"]]}
        for B in (0.5, 0.9):
            mods[f"undrained_B{B}_tau+sn_eff"] = [tot["tau"], tot["sn"] - B * tot["mean"]]
        for m, xs in mods.items():
            f = T.cfit(C, [x.reshape(ND, J)[rows] for x in xs])
            Vw, _ = T.robust_se(f, rows // 7)
            r = dict(stress=name, model=m, dLL=f["LL"] - LL0)
            for i in range(len(xs)):
                r[f"b{i}"] = float(f["beta"][i])
                r[f"se{i}"] = float(np.sqrt(Vw[i, i]))
            out.append(r)
            print(r, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(f"{OUT}/ocean_network_fits_extra.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    if "--extra" in sys.argv:
        extra()
    else:
        main()
