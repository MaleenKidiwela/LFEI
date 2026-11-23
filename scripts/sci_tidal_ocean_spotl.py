#!/usr/bin/env python
"""sci_tidal_ocean_spotl.py - Converged ocean-tide-loading stress at LFE depth (SPOTL ocean models + Green's
functions), added to the body tide of sci_tidal_stress.py, validated against PBO strainmeters and NOAA tide
gauges, and the refit of the LFE-rate response (slip sense, mu', a*sigma; network, per constituent, per family).

Stages:  python scripts/sci_tidal_ocean_spotl.py <stage>
  tests    SPOTL example scripts, nloadf vs our Python re-implementation, Boussinesq closed form.
  stress   ocean-load stress phasors (8 constituents x 9 ocean-model combinations) at every LFE family
           centroid (data/locations/fam{F}_3d.json) + the network centroid + the strainmeters:
             near field  (great-circle distance < R0): SPOTL's own ocean load (ocpts helper = SPOTL ocmodl:
                         1/64-deg land-sea database, coastal interpolation, sea-water density) on a regular
                         1/64-deg grid (1/128 for the convergence test), Boussinesq point-load stress at depth;
             far field   (> R0): SPOTL nloadf surface strain (Farrell 1972 Gutenberg-Bullen Green's functions,
                         spherical, self-gravitating), with a polygon that excludes the R0 circle, converted to
                         stress with plane stress (assumed depth-independent for distant loads, like the body
                         tide).
           Convergence in R0, grid spacing and outer radius; 'hybrid' variant adds the surface layered-
           sphere-minus-half-space difference inside R0 (Farrell minus Boussinesq at the surface).
  strainobs  harmonic analysis of the raw PBO gauge data (data/sci/strain/raw) -> complex M2,O1,K1,N2,S2.
  validate   strainmeters (body + load, unit coupling and calibrated) and NOAA gauges vs each ocean model.
  fit        LFE-rate refits (imports sci_tidal_stress: cfit, robust_se, load_units, ...).
  figs       figures/sci_tidal_spotl_*.png
Outputs: data/sci/tidal_spotl/
"""
import os
import sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import json
import time
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sci_tidal_spotl_lib as L  # noqa: E402
import sci_tidal_stress as T  # noqa: E402

ROOT = T.ROOT
OUT = f"{ROOT}/data/sci/tidal_spotl"
FIG = T.FIG
os.makedirs(OUT, exist_ok=True)
NPROC = 8
CONS8 = L.CONS8
LAT_C, LON_C = 40.06, -123.625
MU_CONV = 30e9          # shear modulus used to turn large-scale strain (body tide, far-field load) into stress
MU_F = 34.69e9          # effective near-surface modulus of the Farrell GB Green's functions (fitted, tests)
NU = 0.25
R0_LIST = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
R0_MAIN = 1.0
GRID_BOX = (34.9, 45.2, -130.3, -116.9)   # near-field grid box (covers R0 <= 5 deg around the LFEs)
STRAIN_STA = {"B045": (40.436001, -123.996498), "B933": (40.060001, -123.969002),
              "B934": (39.844501, -123.691399), "B935": (40.478699, -123.573196)}


def receivers():
    rows = [dict(name="centroid", lat=LAT_C, lon=LON_C, depth_km=28.0)]
    for F in range(1, 28):
        d = json.load(open(f"{ROOT}/data/locations/fam{F}_3d.json"))
        rows.append(dict(name=f"fam{F}", lat=d["lat"], lon=d["lon"], depth_km=d["depth_km"]))
    for s, (la, lo) in STRAIN_STA.items():
        rows.append(dict(name=s, lat=la, lon=lo, depth_km=0.2))
    rows.append(dict(name="centroid_surface", lat=LAT_C, lon=LON_C, depth_km=0.2))
    # network centroid at the mean of the 3-D family locations
    df = pd.DataFrame(rows)
    fm = df[df.name.str.startswith("fam")]
    rows.append(dict(name="famcentroid", lat=fm.lat.mean(), lon=fm.lon.mean(), depth_km=fm.depth_km.mean()))
    return pd.DataFrame(rows)


# ===================================================================== tests
def stage_tests():
    res = {}
    wd = L.WD
    env = dict(os.environ, PATH=os.environ["PATH"] + ":.")
    for i in range(1, 7):
        r = subprocess.run(["nice", "sh", f"ex{i}.scr"], cwd=wd, capture_output=True, text=True, env=env)
        res[f"ex{i}_rc"] = r.returncode
        res[f"ex{i}_stderr"] = r.stderr[:200]
    res["ex4_equals_ex5"] = open(f"{wd}/ex4.f11").read() == open(f"{wd}/ex5.f11").read()
    res["ex1_PFO_M2_strain_E_N_EN"] = open(f"{wd}/ex1.f3").read().splitlines()[-1]
    # ex3: SPOTL usawest ocean tide at Scripps pier vs NOAA La Jolla 9410230
    ex3 = open(f"{wd}/ex3.f2").read().splitlines()
    res["ex3_usawest_scripps_M2_amp_m_localphase"] = [l for l in ex3 if l.startswith(" 2 0 0 0 0 0")]
    # Python re-implementation of nloadf (B933, M2, TPXO7.2)
    lat, lon = STRAIN_STA["B933"]
    gr = L.read_green("gr.gbaver.wef.p02.ce")
    c = L.ring_cells(lat, lon, gr)
    la, lo = L.dest_point(lat, lon, c["dist"], c["az"])
    amp = np.zeros(len(la), complex)
    for fl in ["F", "C"]:
        m = c["flag"] == fl
        amp[m] = L.ocpts("m2.osu.tpxo72.2010", la[m], lo[m], flag=fl)
    py = L.py_surface_strain(c, amp)
    sp, _ = L.nloadf(lat, lon, 0, "M2", "osu.tpxo72.2010")
    res["python_vs_nloadf_B933_M2"] = dict(nloadf=[[abs(z), float(np.angle(z, deg=True))] for z in sp["s"]],
                                           python=[[abs(py[k]), float(np.angle(py[k], deg=True))] for k in ["eEE", "eNN", "eEN"]])
    # Boussinesq closed form vs the finite-difference version in sci_tidal_stress
    s1 = L.bouss_point_stress(np.array([15e3]), np.array([7e3]), 25e3, 1e12)
    s2 = T.bouss_stress(15e3, 7e3, 25e3, np.array([0.0]), np.array([0.0]), np.array([1e12]))
    res["boussinesq_closed_vs_fd_maxrel"] = float(max(abs(s1[k][0] - s2[k]) / abs(s2[k]) for k in L.TENSOR_KEYS))
    # Farrell (GB average Earth) surface strain Green's function vs Boussinesq: apparent shear modulus vs distance
    rows = []
    for rg in gr:
        G = rg["G"]
        for ii in range(G.shape[0]):
            d = rg["beg"] + ii * rg["spc"]
            d1, d2 = np.deg2rad(d - rg["spc"] / 2), np.deg2rad(d + rg["spc"] / 2)
            if d > 20:
                break
            muapp = T.G0 * (1 - 2 * NU) / (4 * np.pi) * np.log(d2 / d1) / G[ii, 4]
            rows.append(dict(dist_deg=d, dist_km=d * 111.195, ett=G[ii, 4], ell=G[ii, 5],
                             mu_apparent_GPa=muapp / 1e9, ell_over_ett=G[ii, 5] / G[ii, 4]))
    gdf = pd.DataFrame(rows)
    gdf.to_csv(f"{OUT}/farrell_vs_boussinesq_gf.csv", index=False)
    res["farrell_mu_apparent_GPa_at_km"] = {f"{k:.0f}": float(np.interp(k, gdf.dist_km, gdf.mu_apparent_GPa))
                                            for k in [0.2, 1, 5, 20, 40, 60, 100, 150, 200, 300, 500, 1000]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/spotl_tests.json", "w"), indent=1)


# ===================================================================== stress
def _grid(res_deg, box=GRID_BOX):
    la0, la1, lo0, lo1 = box
    # cell centres aligned with the 1/64-deg land-sea pixels
    lats = np.arange(np.floor(la0 * 64) / 64 + res_deg / 2, la1, res_deg)
    lons = np.arange(np.floor(lo0 * 64) / 64 + res_deg / 2, lo1, res_deg)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return LA.ravel(), LO.ravel()


def _ocpts_job(args):
    mfile, spec, la, lo = args
    polys = L.poly_list(spec) if spec else None
    return L.ocpts(mfile, la, lo, flag="F", polys=polys, tag=f"{mfile}{len(la)}{la[0]:.3f}")


def near_loads(res_deg, models, cons=CONS8, box=GRID_BOX):
    """Return grid lat, lon, area and dict (model, cons) -> complex load (kg/m^2) restricted to ocean cells."""
    la, lo = _grid(res_deg, box)
    # first find ocean cells (nonzero load) with M2 of the primary regional model and TPXO (any model nonzero)
    jobs, keys = [], []
    nch = int(np.ceil(len(la) / NPROC))
    for mname in models:
        for mfile, spec in L.MODELS[mname]:
            for c in cons:
                for k in range(NPROC):
                    sl = slice(k * nch, (k + 1) * nch)
                    jobs.append((f"{c.lower()}.{mfile}", spec, la[sl], lo[sl]))
                    keys.append((mname, c, k))
    t0 = time.time()
    with ProcessPoolExecutor(NPROC) as ex:
        outs = list(ex.map(_ocpts_job, jobs))
    print(f"ocpts {len(jobs)} jobs, {len(la)} pts each grid, {time.time() - t0:.0f} s", flush=True)
    acc = {}
    for (mname, c, k), v in zip(keys, outs):
        acc.setdefault((mname, c), [None] * NPROC)
        if acc[(mname, c)][k] is None:
            acc[(mname, c)][k] = v
        else:
            acc[(mname, c)][k] = acc[(mname, c)][k] + v
    full = {key: np.concatenate(v) for key, v in acc.items()}
    ocean = np.zeros(len(la), bool)
    for v in full.values():
        ocean |= v != 0
    area = (res_deg * np.pi / 180 * L.A_E) ** 2 * np.cos(np.deg2rad(la[ocean]))
    return la[ocean], lo[ocean], area, {k: v[ocean] for k, v in full.items()}


def near_kernels(rec, la, lo, area):
    """Per-cell Boussinesq stress (Pa per kg/m^2 of load) at the receiver; plus distance (deg)."""
    d, az = L.gc_dist_az(rec["lat"], rec["lon"], la, lo)
    r = np.deg2rad(d) * L.A_E
    sE, sN = r * np.sin(np.deg2rad(az)), r * np.cos(np.deg2rad(az))   # source relative to receiver
    P = T.G0 * area
    z = rec["depth_km"] * 1e3
    K = L.bouss_point_stress(-sE, -sN, z, P, NU)
    Ks = L.bouss_surface_strain(-sE, -sN, P, MU_F, NU)   # surface strain (for the hybrid / surface check)
    return d, K, Ks


def far_field(rec, mname, c, r0, rout=None, green="gr.gbaver.wef.p02.ce"):
    """nloadf surface strain (Z convention, nanostrain E,N,EN) for loads outside r0 (and inside rout).
    SPOTL polygon logic: a point is rejected if it is inside any '-' polygon or outside ALL '+' polygons."""
    tot = np.zeros(3, complex)
    for mfile, spec in L.MODELS[mname]:
        polys = L.poly_list(spec)
        if r0 and r0 > 0:
            polys += L.poly_list(None, ("-", rec["lat"], rec["lon"], r0))
        if rout is not None:
            assert spec is None or spec.startswith("-")
            polys += L.poly_list(None, ("+", rec["lat"], rec["lon"], rout))
        z, _ = L.nloadf(rec["lat"], rec["lon"], 0.0, c, mfile, green=green, polys=polys or None)
        tot += z["s"]
    return tot


def near_field_inside(rec, mname, c, r0, green="gr.gbaver.wef.p02.ce"):
    """nloadf surface strain for loads INSIDE r0 only (for the hybrid correction / surface check).
    The R0 <= 5 deg circles lie inside the usawest polygon (its land part is irrelevant)."""
    tot = np.zeros(3, complex)
    for mfile, spec in L.MODELS[mname]:
        circ = L.poly_list(None, ("+", rec["lat"], rec["lon"], r0))
        polys = circ if spec == "+usawest" else L.poly_list(spec) + circ
        z, _ = L.nloadf(rec["lat"], rec["lon"], 0.0, c, mfile, green=green, polys=polys)
        tot += z["s"]
    return tot


def _far_job(a):
    kind, rec, mname, c, r0, rout = a
    if kind == "far":
        return far_field(rec, mname, c, r0, rout)
    if kind == "full":
        return far_field(rec, mname, c, 0.0)
    return near_field_inside(rec, mname, c, r0)


def strain_to_stress_Z(eZ, mu=MU_CONV):
    """eZ: complex (E, N, EN) nanostrain -> dict of complex stress (Pa), plane stress."""
    return L.plane_stress_from_strain(eZ[0] * 1e-9, eZ[1] * 1e-9, eZ[2] * 1e-9, mu, NU)


def stage_stress():
    recs = receivers()
    recs.to_csv(f"{OUT}/receivers.csv", index=False)
    models = list(L.MODELS)
    t0 = time.time()
    la, lo, area, loads = near_loads(1 / 64, models)
    print("near grid ocean cells", len(la), f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"{OUT}/near_grid_1_64.npz", la=la, lo=lo, area=area,
                        **{f"{m}|{c}": v.astype(np.complex64) for (m, c), v in loads.items()})
    # ---- far-field jobs
    jobs = []
    for _, rec in recs.iterrows():
        rd = rec.to_dict()
        r0s = R0_LIST if rec["name"] in ("centroid", "B933", "centroid_surface") else [R0_MAIN, 2.0]
        for m in models:
            for c in CONS8:
                for r0 in r0s:
                    if m != "usawest+tpxo72" and r0 not in (R0_MAIN, 2.0):
                        continue
                    jobs.append(("far", rd, m, c, r0, None))
                    if rec["depth_km"] < 1 or rec["name"] in ("centroid", "famcentroid"):
                        jobs.append(("in", rd, m, c, r0, None))
                if rec["depth_km"] < 1 or rec["name"] in ("centroid", "famcentroid"):
                    jobs.append(("full", rd, m, c, 0.0, None))
    # outer-radius convergence (centroid, primary model)
    rd = recs[recs.name == "centroid"].iloc[0].to_dict()
    for c in CONS8:
        for rout in [2, 3, 5, 10, 20, 30, None]:
            jobs.append(("far", rd, "tpxo72atlas", c, R0_MAIN, None if rout is None else float(rout)))
    print("nloadf jobs", len(jobs), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(NPROC) as ex:
        fz = list(ex.map(_far_job, jobs, chunksize=8))
    print(f"nloadf done {time.time() - t0:.0f}s", flush=True)
    far = {}
    for j, z in zip(jobs, fz):
        kind, rd_, m, c, r0, rout = j
        far[(kind, rd_["name"], m, c, r0, rout)] = z
    # ---- assemble phasors
    rows = []
    for _, rec in recs.iterrows():
        d, K, Ks = near_kernels(rec, la, lo, area)
        for m in models:
            for c in CONS8:
                q = loads[(m, c)]
                for r0 in R0_LIST:
                    key = ("far", rec["name"], m, c, r0, None)
                    if key not in far:
                        continue
                    sel = d < r0
                    nearS = {k: np.conj(np.sum(K[k][sel] * q[sel])) for k in L.TENSOR_KEYS}
                    nearE = {k: np.conj(np.sum(Ks[k][sel] * q[sel])) * 1e9 for k in ("eEE", "eNN", "eEN")}
                    farS = strain_to_stress_Z(far[key])
                    row = dict(receiver=rec["name"], model=m, cons=c, R0=r0, variant="bouss")
                    for k in L.TENSOR_KEYS:
                        row[f"{k}_near"] = nearS[k]
                        row[f"{k}_far"] = farS[k]
                        row[k] = nearS[k] + farS[k]
                    row.update({f"near_{k}": v for k, v in nearE.items()})
                    row.update(dict(far_eEE=far[key][0], far_eNN=far[key][1], far_eEN=far[key][2]))
                    kin = ("in", rec["name"], m, c, r0, None)
                    if kin in far:
                        row.update(dict(spotl_in_eEE=far[kin][0], spotl_in_eNN=far[kin][1], spotl_in_eEN=far[kin][2]))
                        # hybrid: add (Farrell - Boussinesq) surface difference of the near field as plane stress
                        dS = L.plane_stress_from_strain(*(1e-9 * (far[kin] - np.array([nearE["eEE"], nearE["eNN"], nearE["eEN"]]))),
                                                        MU_F, NU)
                        for k in ("snn", "see", "sne"):
                            row[f"{k}_hybrid"] = row[k] + dS[k]
                    kf = ("full", rec["name"], m, c, 0.0, None)
                    if kf in far:
                        row.update(dict(spotl_full_eEE=far[kf][0], spotl_full_eNN=far[kf][1], spotl_full_eEN=far[kf][2]))
                    rows.append(row)
        print("assembled", rec["name"], flush=True)
    ph = pd.DataFrame(rows)
    ph.to_pickle(f"{OUT}/stress_phasors.pkl")
    # outer radius rows
    orow = []
    for c in CONS8:
        for rout in [2, 3, 5, 10, 20, 30, None]:
            z = far[("far", "centroid", "tpxo72atlas", c, R0_MAIN, None if rout is None else float(rout))]
            S = strain_to_stress_Z(z)
            orow.append(dict(cons=c, Rout_deg=180 if rout is None else rout, **{f"far_{k}": S[k] for k in ("snn", "see", "sne")}))
    pd.DataFrame(orow).to_pickle(f"{OUT}/outer_radius.pkl")
    # ---- grid-resolution convergence (primary model, centroid + 3 families, R0_MAIN)
    conv = []
    for res in [1 / 32, 1 / 64, 1 / 128]:
        box = (LAT_C - R0_MAIN - 0.2, LAT_C + R0_MAIN + 0.2, LON_C - R0_MAIN / 0.76 - 0.3, LON_C + R0_MAIN / 0.76 + 0.3)
        la2, lo2, ar2, ld2 = near_loads(res, ["usawest+tpxo72"], cons=["M2", "K1", "O1"], box=box)
        for nm in ["centroid", "fam23", "fam26", "fam3"]:
            rec = recs[recs.name == nm].iloc[0]
            d, K, _ = near_kernels(rec, la2, lo2, ar2)
            sel = d < R0_MAIN
            for c in ["M2", "K1", "O1"]:
                q = ld2[("usawest+tpxo72", c)]
                conv.append(dict(res_deg=res, receiver=nm, cons=c,
                                 **{k: np.conj(np.sum(K[k][sel] * q[sel])) for k in L.TENSOR_KEYS}))
    pd.DataFrame(conv).to_pickle(f"{OUT}/grid_convergence.pkl")
    print("stress stage done", flush=True)


# ===================================================================== strainmeter observations
TIDE_OBS = ["M2", "N2", "S2", "K1", "O1", "Q1"]
INFER = {"K1": ("P1", 0.331), "S2": ("K2", 0.272)}   # equilibrium-ratio inference (same phase lag)


def tide_columns(jd, consts=TIDE_OBS, infer=True):
    """Design columns for x = Re{Z f e^{i(V+u)}} -> [f cos(V+u), -f sin(V+u)] per constituent (Z convention)."""
    import sci_tidal_stress_ocean as TO
    V = T.doodson_V(jd)
    fu = TO.nodal(jd)
    cols = []
    for c in consts:
        f, u = fu[c]
        ph = np.deg2rad(V[c] + u)
        cc, ss = f * np.cos(ph), -f * np.sin(ph)
        if infer and c in INFER:
            c2, r = INFER[c]
            f2, u2 = fu[c2]
            ph2 = np.deg2rad(V[c2] + u2)
            cc, ss = cc + r * f2 * np.cos(ph2), ss - r * f2 * np.sin(ph2)
        cols += [cc, ss]
    return np.array(cols).T


def stage_strainobs():
    import sci_strain_stack as SS
    rows = []
    for sta in SS.STATIONS:
        d = SS.load_station(sta)
        n = len(d["RS1"])
        t = SS.GRID0 + np.arange(n) * SS.DT
        # hourly means (require all 6 samples)
        nh = n // 6
        th = t[:nh * 6].reshape(nh, 6).mean(1)
        jd = th / 86400.0 + 2440587.5
        X = tide_columns(jd)
        P = d["RDO"][:nh * 6].reshape(nh, 6).mean(1)
        win = 30 * 24
        for g in range(4):
            y = d[f"RS{g + 1}"][:nh * 6].reshape(nh, 6).mean(1)
            for w0 in range(0, nh - win, win):
                sl = slice(w0, w0 + win)
                yy, XX, pp = y[sl], X[sl], P[sl]
                dy = np.diff(yy)
                dX = np.diff(XX, axis=0)
                dp = np.diff(pp)
                ok = np.isfinite(dy) & np.isfinite(dp)
                if ok.sum() < 0.7 * win:
                    continue
                med = np.median(dy[ok])
                mad = 1.4826 * np.median(np.abs(dy[ok] - med))
                ok &= np.abs(dy - med) < 6 * mad
                A = np.c_[np.ones(ok.sum()), np.arange(win - 1)[ok] / win, dp[ok], dX[ok]]
                try:
                    b, *_ = np.linalg.lstsq(A, dy[ok], rcond=None)
                except np.linalg.LinAlgError:
                    continue
                r = dy[ok] - A @ b
                row = dict(station=sta, gauge=g + 1, t0=float(th[w0]), n=int(ok.sum()), rms_diff=float(r.std()),
                           baro=float(b[2]))
                for i, c in enumerate(TIDE_OBS):
                    row[f"{c}_re"], row[f"{c}_im"] = b[3 + 2 * i], b[4 + 2 * i]
                rows.append(row)
        print(sta, "windows", sum(1 for r in rows if r["station"] == sta), flush=True)
    W = pd.DataFrame(rows)
    W.to_csv(f"{OUT}/strainmeter_tide_windows.csv.gz", index=False)
    summ = []
    for (sta, g), grp in W.groupby(["station", "gauge"]):
        # robust: drop windows whose M2 amplitude is far from the median
        a = np.hypot(grp.M2_re, grp.M2_im)
        ma = np.median(a)
        keep = (a > 0.5 * ma) & (a < 1.5 * ma)
        gg = grp[keep]
        r = dict(station=sta, gauge=g, n_windows=int(len(grp)), n_used=int(keep.sum()))
        for c in TIDE_OBS:
            re, im = np.median(gg[f"{c}_re"]), np.median(gg[f"{c}_im"])
            sre = 1.2533 * 1.4826 * np.median(np.abs(gg[f"{c}_re"] - re)) / np.sqrt(len(gg))
            sim = 1.2533 * 1.4826 * np.median(np.abs(gg[f"{c}_im"] - im)) / np.sqrt(len(gg))
            r.update({f"{c}_re": re, f"{c}_im": im, f"{c}_amp": np.hypot(re, im),
                      f"{c}_phase": float(np.degrees(np.arctan2(im, re))), f"{c}_se": float(np.hypot(sre, sim) / np.sqrt(2))})
        summ.append(r)
    S = pd.DataFrame(summ)
    S.to_csv(f"{OUT}/strainmeter_tides_observed.csv", index=False)
    print(S[["station", "gauge", "n_used", "M2_amp", "M2_phase", "M2_se", "O1_amp", "O1_phase", "O1_se", "K1_amp", "K1_phase"]].to_string())


# ===================================================================== validation
VCONS = ["M2", "N2", "O1", "K1"]


def body_phasors(lat, lon, consts=CONS8, years=("2008-01-01", "2026-07-01")):
    """Z-convention phasors of the body-tide surface strain (enn, eee, ene) and plane stress (Pa, MU_CONV)
    at a site, from the sci_tidal_stress potential (h=0.6, l=0.08), hourly LSQ with the 8 constituents."""
    tg = pd.date_range(years[0], years[1], freq="1h")[:-1] + pd.Timedelta(minutes=30)
    jd = T.jd_from_ts(tg)
    e = T.body_strain(T.potential_derivs(lat, lon, jd))
    X = tide_columns(jd, consts, infer=False)
    A = np.c_[np.ones(len(jd)), X]
    out = {}
    for k, y in e.items():
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        out[k] = {c: b[1 + 2 * i] + 1j * b[2 + 2 * i] for i, c in enumerate(consts)}
    return out


def gauge_proj(az_deg, eEE, eNN, eEN):
    th = np.deg2rad(az_deg)
    return eEE * np.sin(th) ** 2 + eNN * np.cos(th) ** 2 + eEN * np.sin(2 * th)


def stage_validate():
    import sci_strain_stack as SS
    import sci_tidal_stress_ocean as TO
    ph = pd.read_pickle(f"{OUT}/stress_phasors.pkl")
    obs = pd.read_csv(f"{OUT}/strainmeter_tides_observed.csv")
    models = list(L.MODELS)
    res = {}
    # ---------------- strainmeters
    rows, cal = [], []
    for sta, (la, lo) in STRAIN_STA.items():
        bp = body_phasors(la, lo, VCONS)
        az = np.array(SS.GAUGE_AZ[sta])
        ob = obs[obs.station == sta].sort_values("gauge")
        good = ob.M2_amp.to_numpy() > 2.0           # B045 gauge 2 is dead (M2 0.5 nstrain)
        for m in ["body_only"] + models:
            pred = {}
            for c in VCONS:
                eE, eN, eEN = bp["eee"][c] * 1e9, bp["enn"][c] * 1e9, bp["ene"][c] * 1e9
                if m != "body_only":
                    r = ph[(ph.receiver == sta) & (ph.model == m) & (ph.cons == c)].iloc[0]
                    eE, eN, eEN = eE + r.spotl_full_eEE, eN + r.spotl_full_eNN, eEN + r.spotl_full_eEN
                pred[c] = (eE, eN, eEN)
                zg = gauge_proj(az, eE, eN, eEN)
                zo = ob[f"{c}_re"].to_numpy() + 1j * ob[f"{c}_im"].to_numpy()
                for g in range(4):
                    rows.append(dict(station=sta, gauge=g + 1, model=m, cons=c, good=bool(good[g]),
                                     obs_amp=abs(zo[g]), obs_phase=np.angle(zo[g], deg=True),
                                     pred_amp=abs(zg[g]), pred_phase=np.angle(zg[g], deg=True),
                                     amp_ratio=abs(zo[g]) / abs(zg[g]),
                                     dphase=(np.angle(zo[g] / zg[g], deg=True))))
            # calibrations: (i) isotropic (areal coupling ca, shear coupling cs, orientation offset dth)
            #               (ii) general real 3-coefficient coupling per gauge
            ZO = np.array([[ob[f"{c}_re"].to_numpy()[g] + 1j * ob[f"{c}_im"].to_numpy()[g] for c in VCONS] for g in range(4)])
            best = None
            for dth in np.arange(-30, 30.1, 1.0):
                th = np.deg2rad(az + dth)
                A, y = [], []
                for g in range(4):
                    if not good[g]:
                        continue
                    for ci, c in enumerate(VCONS):
                        eE, eN, eEN = pred[c]
                        a1 = 0.5 * (eE + eN)
                        a2 = -0.5 * (eE - eN) * np.cos(2 * th[g]) + eEN * np.sin(2 * th[g])
                        for part in (np.real, np.imag):
                            A.append([part(a1), part(a2)])
                            y.append(part(ZO[g, ci]))
                A, y = np.array(A), np.array(y)
                b = np.linalg.lstsq(A, y, rcond=None)[0]
                rr = float(np.sum((y - A @ b) ** 2) / np.sum(y ** 2))
                if dth == 0:
                    r0 = (rr, b)
                if best is None or rr < best[0]:
                    best = (rr, b, dth)
            A, y = [], []
            resg = 0.0
            for g in range(4):
                if not good[g]:
                    continue
                Ag, yg = [], []
                for ci, c in enumerate(VCONS):
                    eE, eN, eEN = pred[c]
                    for part in (np.real, np.imag):
                        Ag.append([part(eE), part(eN), part(eEN)])
                        yg.append(part(ZO[g, ci]))
                Ag, yg = np.array(Ag), np.array(yg)
                bg = np.linalg.lstsq(Ag, yg, rcond=None)[0]
                resg += np.sum((yg - Ag @ bg) ** 2)
                y += list(yg)
            # areal strain phase: unit-coupling inversion of the good gauges
            Gm = SS.gauge_matrix(sta)[good]
            ar = {}
            for c in VCONS:
                zo = ZO[good, VCONS.index(c)]
                x = np.linalg.lstsq(Gm, zo, rcond=None)[0]   # eA, g1, g2 (complex)
                eE, eN, eEN = pred[c]
                ar[c] = (x[0], eE + eN)
            cal.append(dict(station=sta, model=m, n_good=int(good.sum()),
                            iso_resid_frac_dth0=r0[0], iso_ca_dth0=r0[1][0], iso_cs_dth0=r0[1][1],
                            iso_resid_frac=best[0], iso_ca=best[1][0], iso_cs=best[1][1], iso_dth=best[2],
                            gen_resid_frac=float(resg / np.sum(np.array(y) ** 2)),
                            **{f"areal_{c}_obs_amp": abs(ar[c][0]) for c in VCONS},
                            **{f"areal_{c}_pred_amp": abs(ar[c][1]) for c in VCONS},
                            **{f"areal_{c}_dphase": float(np.angle(ar[c][0] / ar[c][1], deg=True)) for c in VCONS}))
    SV = pd.DataFrame(rows)
    SV.to_csv(f"{OUT}/validation_strainmeter_gauges.csv", index=False)
    CV = pd.DataFrame(cal)
    CV.to_csv(f"{OUT}/validation_strainmeter_calibration.csv", index=False)
    print(CV[["station", "model", "iso_resid_frac_dth0", "iso_resid_frac", "iso_ca", "iso_cs", "iso_dth", "gen_resid_frac",
              "areal_M2_dphase", "areal_O1_dphase", "areal_M2_obs_amp", "areal_M2_pred_amp"]].to_string())
    agg = CV.groupby("model")[["iso_resid_frac_dth0", "iso_resid_frac", "gen_resid_frac"]].mean().sort_values("iso_resid_frac")
    print(agg.to_string())
    res["strainmeter_mean_resid"] = agg.to_dict(orient="index")
    # ---------------- NOAA gauges
    g, _ = TO.load_gauges()
    g = g[g.used]
    nrows = []
    for m in models:
        for c in ["M2", "S2", "N2", "K1", "O1"]:
            for _, r in g.iterrows():
                dl = np.arange(-0.25, 0.2501, 1 / 64)
                LA, LO = np.meshgrid(r.lat + dl, r.lon + dl, indexing="ij")
                v = np.zeros(LA.size, complex)
                for mfile, spec in L.MODELS[m]:
                    v += L.ocpts(f"{c.lower()}.{mfile}", LA.ravel(), LO.ravel(), "F", L.poly_list(spec) if spec else None)
                dd = np.hypot((LA.ravel() - r.lat) * 111.2, (LO.ravel() - r.lon) * 85.0)
                dd[v == 0] = np.inf
                i = int(np.argmin(dd))
                H = v[i] / 1025.0            # ~sea-water density used by SPOTL near this coast
                Zm = np.conj(H)               # A e^{-iG}
                Zo = r[f"{c}_A"] * np.exp(-1j * np.deg2rad(r[f"{c}_G"]))
                nrows.append(dict(model=m, cons=c, gauge=r["name"], id=r["id"], lat=r.lat, dist_to_model_km=dd[i],
                                  noaa_amp=r[f"{c}_A"], noaa_G=r[f"{c}_G"], model_amp=abs(Zm),
                                  model_G=(-np.angle(Zm, deg=True)) % 360, amp_ratio=abs(Zm) / r[f"{c}_A"],
                                  dG=float(np.angle(Zo / Zm, deg=True)), cmisfit_m=abs(Zm - Zo)))
    NV = pd.DataFrame(nrows)
    NV.to_csv(f"{OUT}/validation_noaa.csv", index=False)
    near = ["9418767", "9416841", "9419750"]
    nagg = NV.groupby(["model", "cons"]).apply(lambda d: pd.Series(dict(
        rms_cmisfit_cm=100 * np.sqrt(np.mean(d.cmisfit_m ** 2) / 2),
        rms_cmisfit_near3_cm=100 * np.sqrt(np.mean(d[d.id.isin(near)].cmisfit_m ** 2) / 2),
        med_amp_ratio=np.median(d.amp_ratio), med_dG=np.median(d.dG)))).reset_index()
    print(nagg.to_string())
    nagg.to_csv(f"{OUT}/validation_noaa_summary.csv", index=False)
    res["noaa_M2_rms_cm"] = nagg[nagg.cons == "M2"].set_index("model").rms_cmisfit_cm.to_dict()
    json.dump(res, open(f"{OUT}/validation_summary.json", "w"), indent=1, default=float)


# ===================================================================== LFE-rate refits
FCONS = ["M2", "N2", "K1", "O1", "S2"]    # S2 is exactly 24-h periodic -> absorbed by the time-of-day term
MUGRID = np.round(np.arange(-0.4, 1.21, 0.1), 3)
rng = np.random.default_rng(20261125)


class Ctx:
    pass


def setup_time():
    import sci_tidal_stress_ocean as TO
    cx = Ctx()
    cx.ND = (T.T_END - T.T0).days
    tg = T.T0 + pd.to_timedelta((np.arange(cx.ND * T.J) + 0.5) * T.BIN_MIN, unit="min")
    cx.jd = T.jd_from_ts(tg)
    V = T.doodson_V(cx.jd)
    fu = TO.nodal(cx.jd)
    cx.E = {c: fu[c][0] * np.exp(1j * np.deg2rad(V[c] + fu[c][1])) for c in CONS8}
    P = T.potential_derivs(T.LAT0, T.LON0, cx.jd)
    cx.bstrain = T.body_strain(P)                          # enn, eee, ene (strain)
    sub = slice(0, None, 3)
    X = tide_columns(cx.jd[sub], CONS8, infer=False)
    A = np.c_[np.ones(X.shape[0]), X]
    cx.bstrain_ph = {}
    for k, y in cx.bstrain.items():
        b = np.linalg.lstsq(A, y[sub], rcond=None)[0]
        cx.bstrain_ph[k] = {c: b[1 + 2 * i] + 1j * b[2 + 2 * i] for i, c in enumerate(CONS8)}
    cx.body_cache = {}
    # LFE units
    u = T.load_units()
    cx.n_units = len(u)
    cx.Cnet = np.zeros((cx.ND, T.J))
    np.add.at(cx.Cnet, (u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    cx.Cfam = np.zeros((T.NF, cx.ND, T.J), np.float32)
    np.add.at(cx.Cfam, (u.fam.to_numpy(), u.day.to_numpy(), u.bin.to_numpy()), 1.0)
    cx.rows = np.where(cx.Cnet.sum(1) > 0)[0]
    return cx


def body_stress(cx, mu, nu):
    """Body-tide plane stress (kPa) from the body-tide strain with moduli (mu, nu): series + phasors."""
    key = (round(mu / 1e6), round(nu, 4))
    if key not in cx.body_cache:
        e = cx.bstrain
        S = L.plane_stress_from_strain(e["eee"], e["enn"], e["ene"], mu, nu)
        ser = {k: v / 1e3 for k, v in S.items() if k in ("snn", "see", "sne")}
        ph_ = {}
        for k in ("snn", "see", "sne"):
            ph_[k] = {}
        for c in CONS8:
            Sp = L.plane_stress_from_strain(cx.bstrain_ph["eee"][c], cx.bstrain_ph["enn"][c], cx.bstrain_ph["ene"][c], mu, nu)
            for k in ("snn", "see", "sne"):
                ph_[k][c] = Sp[k] / 1e3
        cx.body_cache[key] = dict(ser=ser, ph=ph_)
    return cx.body_cache[key]


def ocean_phasors(ph, receiver, model="usawest+tpxo72", variant="layered_RL1"):
    """dict comp -> {cons: complex Pa} (Z convention) of the ocean-load stress tensor (stress_phasors_final)."""
    d = ph[(ph.receiver == receiver) & (ph.model == model) & (ph.variant == variant)]
    assert len(d) == len(CONS8), (receiver, model, variant, len(d))
    return {k: {r.cons: r[k] for _, r in d.iterrows()} for k in L.TENSOR_KEYS}


def series_from_ph(cx, phk, consts=CONS8):
    x = 0.0
    for c in consts:
        x = x + np.real(phk.get(c, 0) * cx.E[c])
    return x


def stress_set(cx, oph, body=None, body_scale=1.0, ocean_scale=1.0, strike=T.STRIKE, dip=40.0, dphase=0.0):
    """Resolved total stress series (kPa): tau (RL), sn (unclamping +), mean; + per-constituent phasors.
    body: output of body_stress (None -> no body tide)."""
    rot = np.exp(1j * np.deg2rad(dphase))
    S = {}
    for k in L.TENSOR_KEYS:
        oc = series_from_ph(cx, {c: v * rot / 1e3 for c, v in oph[k].items()}) if oph is not None else 0.0
        bd = body["ser"].get(k, 0.0) if body is not None else 0.0
        S[k] = body_scale * bd + ocean_scale * oc
    R = T.resolve(S, strike, dip)
    Sp = {k: {c: (body_scale * body["ph"].get(k, {}).get(c, 0.0) if body is not None else 0.0) +
              (ocean_scale * oph[k][c] * rot / 1e3 if oph is not None else 0.0) for c in CONS8} for k in L.TENSOR_KEYS}
    Rp = {c: T.resolve({k: Sp[k][c] for k in L.TENSOR_KEYS}, strike, dip) for c in CONS8}
    return dict(tau=R["tau"], sn=R["sn"], mean=R["mean"], ph=Rp)


def M_(cx, x):
    return x.reshape(cx.ND, T.J)


def fit_block(cx, X, C=None, rows=None, w=None, full=False):
    """Core fits for one stress model.  Returns dict."""
    rows = cx.rows if rows is None else rows
    C = cx.Cnet[rows] if C is None else C
    out = {}
    f0 = T.cfit(C, [], w=w)
    LL0 = f0["LL"]
    tau, sn, mean = (M_(cx, X[k])[rows] for k in ("tau", "sn", "mean"))
    ft = T.cfit(C, [tau], w=w)
    Vw, _ = T.robust_se(ft, rows // 7)
    out.update(beta_tau=float(ft["beta"][0]), se_tau=float(np.sqrt(Vw[0, 0])), dLL_tau=ft["LL"] - LL0,
               a_sigma_tau=float(1 / ft["beta"][0]) if ft["beta"][0] > 0 else np.nan)
    f2 = T.cfit(C, [tau, sn], w=w)
    Vw, _ = T.robust_se(f2, rows // 7)
    b0, b1 = f2["beta"]
    gg = np.array([-b1 / b0 ** 2, 1 / b0])
    out.update(beta2_tau=float(b0), beta2_sn=float(b1), se2_tau=float(np.sqrt(Vw[0, 0])), se2_sn=float(np.sqrt(Vw[1, 1])),
               mu_eff=float(b1 / b0), mu_eff_se=float(np.sqrt(gg @ Vw @ gg)), dLL_tau_sn=f2["LL"] - LL0)
    fs = T.cfit(C, [sn], w=w)
    fm = T.cfit(C, [mean], w=w)
    out.update(dLL_sn=fs["LL"] - LL0, beta_sn=float(fs["beta"][0]), dLL_mean=fm["LL"] - LL0, beta_mean=float(fm["beta"][0]))
    f3 = T.cfit(C, [tau, sn, mean], w=w)
    Vw3, _ = T.robust_se(f3, rows // 7)
    out.update(dLL_tau_sn_mean=f3["LL"] - LL0, beta3=[float(v) for v in f3["beta"]],
               se3=[float(np.sqrt(Vw3[i, i])) for i in range(3)])
    # mu' profiles for both slip senses (beta free sign -> a negative beta = opposite sense)
    prof = []
    for mu in MUGRID:
        for sense, sg in (("RL", 1.0), ("LL", -1.0)):
            x = sg * tau + mu * sn
            f = T.cfit(C, [x], w=w)
            prof.append(dict(sense=sense, mu=mu, beta=float(f["beta"][0]), dLL=f["LL"] - LL0))
    prof = pd.DataFrame(prof)
    for sense in ("RL", "LL"):
        g = prof[(prof.sense == sense) & (prof.beta > 0)]
        if len(g):
            i = g.dLL.idxmax()
            ok = g[g.dLL >= g.dLL.max() - 1.92]
            out[f"{sense}_mu_best"] = float(g.mu[i])
            out[f"{sense}_dLL_best"] = float(g.dLL[i])
            out[f"{sense}_a_sigma_best"] = float(1 / g.beta[i])
            out[f"{sense}_mu_95prof"] = [float(ok.mu.min()), float(ok.mu.max())]
            gp = g[(g.mu >= 0) & (g.mu <= 1.0)]
            if len(gp):
                j = gp.dLL.idxmax()
                out[f"{sense}_phys_mu_best"] = float(gp.mu[j])
                out[f"{sense}_phys_dLL"] = float(gp.dLL[j])
        else:
            out[f"{sense}_dLL_best"] = 0.0
            out[f"{sense}_phys_dLL"] = 0.0
    out["dLL_RL_minus_LL_phys"] = out.get("RL_phys_dLL", 0.0) - out.get("LL_phys_dLL", 0.0)
    out["prof"] = prof
    out["LL0"] = LL0
    return out


def lfe_phasors(cx, consts=("M2", "N2", "K1", "O1", "Q1"), C=None, rows=None, hod=True):
    rows = cx.rows if rows is None else rows
    C = cx.Cnet[rows] if C is None else C
    Xs = []
    for c in consts:
        Xs += [M_(cx, np.real(cx.E[c]))[rows], M_(cx, -np.imag(cx.E[c]))[rows]]
    f = T.cfit(C, Xs, hod=hod)
    Vw, _ = T.robust_se(f, rows // 7)
    b = f["beta"]
    return {c: b[2 * i] + 1j * b[2 * i + 1] for i, c in enumerate(consts)}, Vw, consts


def phasor_model_fit(bph, Vw, consts, Rp, mus=MUGRID, allc=("M2", "N2", "K1", "O1", "Q1")):
    """Fit LFE phasors b_c = beta (T_c + mu N_c), with the week-cluster covariance; profile over mu."""
    y = np.concatenate([[bph[c].real, bph[c].imag] for c in consts])
    idx = np.concatenate([[2 * allc.index(c), 2 * allc.index(c) + 1] for c in consts])
    Wi = np.linalg.inv(Vw[np.ix_(idx, idx)])
    best = None
    rows = []
    for mu in mus:
        a = np.concatenate([[(Rp[c]["tau"] + mu * Rp[c]["sn"]).real, (Rp[c]["tau"] + mu * Rp[c]["sn"]).imag] for c in consts])
        beta = float(a @ Wi @ y / (a @ Wi @ a))
        r = y - beta * a
        chi2 = float(r @ Wi @ r)
        rows.append(dict(mu=mu, beta=beta, chi2=chi2, se_beta=float(1 / np.sqrt(a @ Wi @ a))))
    d = pd.DataFrame(rows)
    chi0 = float(y @ Wi @ y)
    i = d.chi2.idxmin()
    ok = d[d.chi2 <= d.chi2.min() + 3.84]
    return dict(chi2_null=chi0, chi2_best=float(d.chi2[i]), dof=len(y) - 2, mu_best=float(d.mu[i]),
                beta_best=float(d.beta[i]), beta_se=float(d.se_beta[i]), mu_95=[float(ok.mu.min()), float(ok.mu.max())],
                sense="RL" if d.beta[i] > 0 else "LL", prof=d)


def per_constituent_td(cx, X, consts=("M2", "N2", "K1", "O1"), C=None, rows=None, hod=True, extra=()):
    """Time-domain fit with separate tau_c, sn_c covariates per constituent (+ remainder)."""
    rows = cx.rows if rows is None else rows
    C = cx.Cnet[rows] if C is None else C
    Xs, names = [], []
    rest_t, rest_n = X["tau"].copy(), X["sn"].copy()
    for c in tuple(consts) + tuple(extra):
        tc = np.real(X["ph"][c]["tau"] * cx.E[c])
        nc = np.real(X["ph"][c]["sn"] * cx.E[c])
        rest_t -= tc
        rest_n -= nc
        Xs += [M_(cx, tc)[rows], M_(cx, nc)[rows]]
        names += [(c, "tau"), (c, "sn")]
    Xs += [M_(cx, rest_t)[rows], M_(cx, rest_n)[rows]]
    names += [("rest", "tau"), ("rest", "sn")]
    f = T.cfit(C, Xs, hod=hod)
    Vw, _ = T.robust_se(f, rows // 7)
    out = []
    for i in range(0, len(names), 2):
        c = names[i][0]
        bt, bn = f["beta"][i], f["beta"][i + 1]
        Vs = Vw[i:i + 2, i:i + 2]
        gg = np.array([-bn / bt ** 2, 1 / bt])
        amp_t = abs(X["ph"][c]["tau"]) if c != "rest" else float(np.std(rest_t) * np.sqrt(2))
        amp_n = abs(X["ph"][c]["sn"]) if c != "rest" else float(np.std(rest_n) * np.sqrt(2))
        out.append(dict(cons=c, beta_tau=float(bt), se_tau=float(np.sqrt(Vs[0, 0])), beta_sn=float(bn),
                        se_sn=float(np.sqrt(Vs[1, 1])), mu_eff=float(bn / bt), mu_eff_se=float(np.sqrt(gg @ Vs @ gg)),
                        a_sigma_tau=float(1 / bt) if bt > 0 else np.nan, tau_amp_kPa=amp_t, sn_amp_kPa=amp_n,
                        z_tau=float(bt / np.sqrt(Vs[0, 0]))))
    return pd.DataFrame(out)


def stage_fit():
    t00 = time.time()
    ph = pd.read_pickle(f"{OUT}/stress_phasors_final.pkl")
    mods = pd.read_csv(f"{OUT}/receiver_local_moduli.csv").set_index("receiver")
    cx = setup_time()
    print("setup", f"{time.time() - t00:.0f}s", "units", cx.n_units, flush=True)
    NET = "famcentroid"
    mu_net, nu_net = mods.loc[NET, "mu_local"], mods.loc[NET, "nu_local"]
    body_loc = body_stress(cx, mu_net, nu_net)
    body_30 = body_stress(cx, 30e9, 0.25)
    summ = dict(n_units=cx.n_units, network_receiver=NET, mu_local_GPa=mu_net / 1e9, nu_local=nu_net)
    # ---------------- configurations (network): (name, ocean model, ocean variant, body, body_scale)
    PV = "layered_RL2"
    configs = [("body_only_mu30", None, None, body_30), ("body_only_local", None, None, body_loc),
               ("primary", "usawest+tpxo72", PV, body_loc)]
    for m in L.MODELS:
        if m != "usawest+tpxo72":
            configs.append((f"model_{m}", m, PV, body_loc))
    for v in ("layered_RL0.5", "layered_RL1", "layered_RL1.5", "layered_RL2.5"):
        configs.append((v, "usawest+tpxo72", v, body_loc))
    configs += [("homog_RB1_body30", "usawest+tpxo72", "homog_RB1", body_30),
                ("homog_RB5_body30", "usawest+tpxo72", "homog_RB5", body_30),
                ("layered_body30", "usawest+tpxo72", PV, body_30),
                ("ocean_only", "usawest+tpxo72", PV, None)]
    cres, profs = [], []
    Xprim = None
    for name, m, var, bdy in configs:
        oph = ocean_phasors(ph, NET, m, var) if m else None
        X = stress_set(cx, oph, body=bdy)
        if name == "primary":
            Xprim, oph_prim = X, oph
        r = fit_block(cx, X)
        pr = r.pop("prof")
        pr["config"] = name
        profs.append(pr)
        # phasor-level fit (all constituents jointly)
        if name == "body_only_mu30":
            bph, Vb, cs = lfe_phasors(cx)
            summ["lfe_phasors"] = {c: [float(abs(bph[c])), float(np.angle(bph[c], deg=True))] for c in cs}
        pf = phasor_model_fit(bph, Vb, ("M2", "N2", "K1", "O1"), X["ph"])
        r.update(name=name, phasor_mu_best=pf["mu_best"], phasor_beta=pf["beta_best"], phasor_beta_se=pf["beta_se"],
                 phasor_chi2=pf["chi2_best"], phasor_chi2_null=pf["chi2_null"], phasor_mu_95=pf["mu_95"],
                 phasor_sense=pf["sense"],
                 tau_M2_amp=abs(X["ph"]["M2"]["tau"]), tau_M2_ph=float(np.angle(X["ph"]["M2"]["tau"], deg=True)),
                 sn_M2_amp=abs(X["ph"]["M2"]["sn"]), sn_M2_ph=float(np.angle(X["ph"]["M2"]["sn"], deg=True)),
                 tau_O1_amp=abs(X["ph"]["O1"]["tau"]), tau_K1_amp=abs(X["ph"]["K1"]["tau"]))
        cres.append(r)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, dict))}, flush=True)
    CR = pd.DataFrame(cres)
    CR.to_csv(f"{OUT}/network_config_fits.csv", index=False)
    pd.concat(profs).to_csv(f"{OUT}/network_mu_profiles.csv", index=False)
    summ["configs"] = CR.replace({np.nan: None}).to_dict(orient="records")
    X = Xprim
    rows = cx.rows
    C = cx.Cnet[rows]
    LL0 = T.cfit(C, [])["LL"]
    # ---------------- stress harmonics table (network receiver, primary + body)
    hrows = []
    for c in CONS8:
        bodyR = T.resolve({k: body_loc["ph"].get(k, {}).get(c, 0.0) for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
        ocR = T.resolve({k: oph_prim[k][c] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
        for comp in ("tau", "sn", "mean"):
            b_, o_, t_ = bodyR[comp], ocR[comp], X["ph"][c][comp]
            hrows.append(dict(cons=c, comp=comp, body_amp=abs(b_), body_phase=float(np.angle(b_, deg=True)),
                              ocean_amp=abs(o_), ocean_phase=float(np.angle(o_, deg=True)),
                              total_amp=abs(t_), total_phase=float(np.angle(t_, deg=True)),
                              ocean_over_body=abs(o_) / abs(b_) if abs(b_) > 0 else np.nan,
                              ocean_minus_body_phase=float(np.angle(o_ / b_, deg=True)) if abs(b_) > 0 else np.nan))
    H = pd.DataFrame(hrows)
    H.to_csv(f"{OUT}/stress_harmonics_network.csv", index=False)
    print(H.to_string(), flush=True)
    # ---------------- per constituent (primary): phasors vs stress, time-domain per-constituent fits
    bph, Vb, cs = lfe_phasors(cx)
    pc = []
    for i, c in enumerate(cs):
        Vc = Vb[2 * i:2 * i + 2, 2 * i:2 * i + 2]
        for comp in ("tau", "sn"):
            for tag, Zs in (("total", X["ph"][c][comp]), ("body", T.resolve({k: body_loc["ph"].get(k, {}).get(c, 0.0) for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)[comp])):
                adm = bph[c] / Zs
                pc.append(dict(cons=c, comp=comp, stress=tag, lfe_amp=abs(bph[c]), lfe_phase=float(np.angle(bph[c], deg=True)),
                               lfe_se=float(np.sqrt(np.trace(Vc) / 2)), stress_amp_kPa=abs(Zs),
                               stress_phase=float(np.angle(Zs, deg=True)),
                               lfe_minus_stress_deg=float(np.angle(adm, deg=True)),
                               lfe_minus_stress_se_deg=float(np.degrees(np.sqrt(np.trace(Vc) / 2) / abs(bph[c]))),
                               beta_proj=float(abs(bph[c]) * np.cos(np.angle(adm)) / abs(Zs))))
    PC = pd.DataFrame(pc)
    PC.to_csv(f"{OUT}/per_constituent_phasors.csv", index=False)
    print(PC.to_string(), flush=True)
    TD = per_constituent_td(cx, X)
    TD["model"] = "primary_hod"
    TDb = per_constituent_td(cx, stress_set(cx, None, body=body_loc))
    TDb["model"] = "body_hod"
    TDs = per_constituent_td(cx, X, consts=("M2", "N2", "K1", "O1"), extra=("S2",), hod=False)
    TDs["model"] = "primary_noHOD_S2_confounded"
    TDall = pd.concat([TD, TDb, TDs])
    TDall.to_csv(f"{OUT}/per_constituent_timedomain.csv", index=False)
    print(TDall.to_string(), flush=True)
    # ---------------- ocean-scale / phase profile (consistency test)
    osp = []
    for sc in np.arange(0, 2.01, 0.25):
        for dphs in (-30, -15, 0, 15, 30):
            Xs_ = stress_set(cx, oph_prim, body=body_loc, ocean_scale=sc, dphase=dphs)
            f2 = T.cfit(C, [M_(cx, Xs_["tau"])[rows], M_(cx, Xs_["sn"])[rows]])
            f1 = T.cfit(C, [M_(cx, Xs_["tau"])[rows]])
            osp.append(dict(scale=sc, dphase=dphs, dLL_tau_sn=f2["LL"] - LL0, dLL_tau=f1["LL"] - LL0,
                            beta_tau=float(f2["beta"][0]), mu_eff=float(f2["beta"][1] / f2["beta"][0]),
                            beta_tau_only=float(f1["beta"][0])))
    OS = pd.DataFrame(osp)
    OS.to_csv(f"{OUT}/ocean_scale_profile.csv", index=False)
    summ["ocean_scale_best"] = OS.loc[OS.dLL_tau_sn.idxmax()].to_dict()
    summ["ocean_scale_best_tau_only"] = OS.loc[OS.dLL_tau.idxmax()].to_dict()
    print("ocean scale best", summ["ocean_scale_best"], flush=True)
    # ---------------- strike scan (total stress, dip 40, RL rake; beta<0 = opposite sense)
    best = [r for r in cres if r["name"] == "primary"][0]
    mu_b = best.get("RL_phys_mu_best", 0.1)
    scan = []
    for stv in np.arange(0, 360, 10):
        Xs_ = stress_set(cx, oph_prim, body=body_loc, strike=stv)
        f1 = T.cfit(C, [M_(cx, Xs_["tau"])[rows]])
        f2 = T.cfit(C, [M_(cx, Xs_["tau"] + mu_b * Xs_["sn"])[rows]])
        scan.append(dict(strike=stv, beta_tau=float(f1["beta"][0]), dLL_tau=f1["LL"] - LL0,
                         beta_cfs=float(f2["beta"][0]), dLL_cfs=f2["LL"] - LL0))
    SC = pd.DataFrame(scan)
    SC.to_csv(f"{OUT}/strike_scan_total.csv", index=False)
    summ["strike_scan_best_tau"] = SC.loc[SC.dLL_tau.idxmax()].to_dict()
    # ---------------- Hilbert phase lag vs total tau and total CFS
    from scipy.signal import hilbert
    lag = {}
    for k, x in (("tau", X["tau"]), ("cfs_best", X["tau"] + mu_b * X["sn"])):
        hx = np.imag(hilbert(x - x.mean()))
        f = T.cfit(C, [M_(cx, x)[rows], M_(cx, hx)[rows]])
        Vw, _ = T.robust_se(f, rows // 7)
        bc, bs = f["beta"]
        gph = np.array([-bs, bc]) / (bc ** 2 + bs ** 2)
        lag[k] = dict(phase_lag_deg=float(np.degrees(np.arctan2(bs, bc))), se_deg=float(np.degrees(np.sqrt(gph @ Vw @ gph))))
    summ["phase_lag_total"] = lag
    print("lag", lag, flush=True)
    # ---------------- nulls (score tests at the null time-of-day pattern)
    f0 = T.cfit(C, [])
    p_hod = f0["p"][0]
    Nr = C.sum(1)
    resid0 = C - Nr[:, None] * p_hod[None, :]

    def score_beta(Xr):
        U = (resid0 * Xr).sum()
        mX = Xr @ p_hod
        I = (Nr * ((Xr ** 2) @ p_hod - mX ** 2)).sum()
        return U / I
    nulls = {}
    for k, x in (("tau", X["tau"]), ("cfs_best", X["tau"] + mu_b * X["sn"]), ("sn", X["sn"])):
        Xm = M_(cx, x)
        b_obs = score_beta(Xm[rows])
        bsh, bsc = [], []
        for _ in range(1000):
            kk = rng.integers(1, 401, len(rows)) * rng.choice([-1, 1], len(rows))
            tgt = rows + kk
            tgt = np.where((tgt < 0) | (tgt >= cx.ND), rows - kk, tgt)
            bsh.append(score_beta(Xm[tgt]))
        Fx = np.fft.rfft(x - x.mean())
        for _ in range(250):
            phs = np.exp(2j * np.pi * rng.random(len(Fx)))
            phs[0] = 1
            bsc.append(score_beta(M_(cx, np.fft.irfft(np.abs(Fx) * phs, n=len(x)))[rows]))
        bsh, bsc = np.array(bsh), np.array(bsc)
        nulls[k] = dict(beta_onestep=float(b_obs), dayshift_z=float(b_obs / bsh.std()),
                        dayshift_p=float((1 + np.sum(np.abs(bsh) >= abs(b_obs))) / 1001),
                        scramble_z=float(b_obs / bsc.std()), scramble_p=float((1 + np.sum(np.abs(bsc) >= abs(b_obs))) / 251))
    summ["nulls"] = nulls
    print("nulls", nulls, flush=True)
    # ---------------- bootstrap (day Poisson weights)
    B = 100
    W = rng.poisson(1.0, (B, len(rows))).astype(float)
    tau, sn = M_(cx, X["tau"])[rows], M_(cx, X["sn"])[rows]
    ft = T.cfit(C, [tau])
    f2 = T.cfit(C, [tau, sn])
    mus_b = np.round(np.arange(0, 1.01, 0.1), 2)
    bt, bmu, dd, bmub, bsc_ = [], [], [], [], []
    OSx = {sc: stress_set(cx, oph_prim, body=body_loc, ocean_scale=sc) for sc in (0.0, 0.5, 1.0, 1.5, 2.0)}
    for b in range(B):
        fb = T.cfit(C, [tau], w=W[b], theta0=ft["theta"], iters=8)
        bt.append(fb["beta"][0])
        fb2 = T.cfit(C, [tau, sn], w=W[b], theta0=f2["theta"], iters=8)
        bmu.append(fb2["beta"][1] / fb2["beta"][0])
        llr, lll = [], []
        for mu in mus_b:
            fr = T.cfit(C, [tau + mu * sn], w=W[b], iters=8)
            fl = T.cfit(C, [-tau + mu * sn], w=W[b], iters=8)
            llr.append(fr["LL"] if fr["beta"][0] > 0 else -np.inf)
            lll.append(fl["LL"] if fl["beta"][0] > 0 else -np.inf)
        f0b = T.cfit(C, [], w=W[b], iters=8)
        dd.append(max(max(llr), f0b["LL"]) - max(max(lll), f0b["LL"]))
        bmub.append(mus_b[int(np.argmax(llr))])
        lls = []
        for sc, Xs_ in OSx.items():
            fs_ = T.cfit(C, [M_(cx, Xs_["tau"])[rows], M_(cx, Xs_["sn"])[rows]], w=W[b], iters=8)
            lls.append(fs_["LL"])
        bsc_.append(list(OSx)[int(np.argmax(lls))])
        if b % 25 == 0:
            print("boot", b, f"{time.time() - t00:.0f}s", flush=True)
    bt = np.array(bt)
    summ["bootstrap"] = dict(B=B, beta_tau_mean=float(bt.mean()), beta_tau_sd=float(bt.std()),
                             a_sigma_tau_95=[float(1 / np.percentile(bt, 97.5)), float(1 / np.percentile(bt, 2.5))],
                             mu_eff_median=float(np.median(bmu)), mu_eff_95=[float(np.percentile(bmu, 2.5)), float(np.percentile(bmu, 97.5))],
                             mu_RL_best_95=[float(np.percentile(bmub, 2.5)), float(np.percentile(bmub, 97.5))],
                             mu_RL_best_median=float(np.median(bmub)),
                             dLL_RL_minus_LL_median=float(np.median(dd)), dLL_RL_minus_LL_95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                             frac_LL_preferred=float(np.mean(np.array(dd) < 0)),
                             ocean_scale_best_counts={str(k): int(np.sum(np.array(bsc_) == k)) for k in OSx})
    print("bootstrap", summ["bootstrap"], flush=True)
    # ---------------- per family (family-specific ocean load at the 3-D location)
    frs = []
    for f_ in range(T.NF):
        rr = np.where(cx.Cfam[f_].sum(1) > 0)[0]
        if len(rr) < 50:
            continue
        Cf = cx.Cfam[f_][rr].astype(float)
        rec = f"fam{f_ + 1}"
        row = dict(family=f_ + 1, n_units=int(Cf.sum()))
        bf = body_stress(cx, mods.loc[rec, "mu_local"], mods.loc[rec, "nu_local"])
        for tag, oph in (("total", ocean_phasors(ph, rec, "usawest+tpxo72", PV)), ("body", None)):
            Xf = stress_set(cx, oph, body=bf)
            r = fit_block(cx, Xf, C=Cf, rows=rr)
            r.pop("prof")
            for k in ("beta_tau", "se_tau", "dLL_tau", "a_sigma_tau", "mu_eff", "mu_eff_se", "dLL_tau_sn", "dLL_mean",
                      "RL_phys_mu_best", "RL_phys_dLL", "LL_phys_dLL", "dLL_RL_minus_LL_phys", "RL_a_sigma_best"):
                row[f"{tag}_{k}"] = r.get(k, np.nan)
            hx = np.imag(hilbert(Xf["tau"] - Xf["tau"].mean()))
            ff = T.cfit(Cf, [M_(cx, Xf["tau"])[rr], M_(cx, hx)[rr]])
            Vw, _ = T.robust_se(ff, rr // 7)
            bc, bs = ff["beta"]
            gph = np.array([-bs, bc]) / (bc ** 2 + bs ** 2)
            row[f"{tag}_lag_deg"] = float(np.degrees(np.arctan2(bs, bc)))
            row[f"{tag}_lag_se"] = float(np.degrees(np.sqrt(gph @ Vw @ gph)))
            if tag == "total":
                row.update(tau_M2_amp=abs(Xf["ph"]["M2"]["tau"]), tau_M2_ph=float(np.angle(Xf["ph"]["M2"]["tau"], deg=True)),
                           sn_M2_amp=abs(Xf["ph"]["M2"]["sn"]), sn_M2_ph=float(np.angle(Xf["ph"]["M2"]["sn"], deg=True)),
                           mu_local_GPa=mods.loc[rec, "mu_local"] / 1e9, ocean_tau_M2_amp=abs(T.resolve({k: oph[k]["M2"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)["tau"]),
                           ocean_sn_M2_amp=abs(T.resolve({k: oph[k]["M2"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)["sn"]))
        frs.append(row)
        print("fam", f_ + 1, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    FR = pd.DataFrame(frs)
    recs = receivers().set_index("name")
    FR["lat"] = [recs.loc[f"fam{f}", "lat"] for f in FR.family]
    FR["lon"] = [recs.loc[f"fam{f}", "lon"] for f in FR.family]
    FR["depth_km"] = [recs.loc[f"fam{f}", "depth_km"] for f in FR.family]
    FR.to_csv(f"{OUT}/family_fits.csv", index=False)
    good = FR[FR.n_units >= 300]
    from scipy.stats import spearmanr
    summ["family"] = dict(
        n=int(len(good)),
        total_beta_tau_pos=int((good.total_beta_tau > 0).sum()), body_beta_tau_pos=int((good.body_beta_tau > 0).sum()),
        total_RL_preferred=int((good.total_dLL_RL_minus_LL_phys > 0).sum()),
        total_a_sigma_median=float(np.nanmedian(good.total_a_sigma_tau)),
        total_a_sigma_IQR=[float(np.nanpercentile(good.total_a_sigma_tau, 25)), float(np.nanpercentile(good.total_a_sigma_tau, 75))],
        total_mu_eff_median=float(np.median(good.total_mu_eff)), body_mu_eff_median=float(np.median(good.body_mu_eff)),
        total_lag_vs_lon_spearman=[float(v) for v in spearmanr(good.lon, good.total_lag_deg)],
        body_lag_vs_lon_spearman=[float(v) for v in spearmanr(good.lon, good.body_lag_deg)],
        total_mu_vs_lon_spearman=[float(v) for v in spearmanr(good.lon, good.total_mu_eff)],
        chi2_beta_tau_const_total=float((((good.total_beta_tau - np.average(good.total_beta_tau, weights=good.total_se_tau ** -2)) / good.total_se_tau) ** 2).sum()))
    print(json.dumps(summ["family"], indent=1), flush=True)
    json.dump(summ, open(f"{OUT}/fit_summary.json", "w"), indent=1,
              default=lambda o: float(o) if np.isscalar(o) else (o.tolist() if hasattr(o, "tolist") else str(o)))
    print("fit done", f"{time.time() - t00:.0f}s", flush=True)


# ===================================================================== convergence summary + figures
def stage_converge():
    """Tables of the ocean-load stress phasors at the network receiver vs R0, grid spacing and outer radius."""
    ph = pd.read_pickle(f"{OUT}/stress_phasors.pkl")
    rows = []
    for rec in ("famcentroid", "centroid", "fam23", "fam26"):
        for c in ("M2", "K1", "O1"):
            d = ph[(ph.receiver == rec) & (ph.model == "usawest+tpxo72") & (ph.cons == c)].sort_values("R0")
            for _, r in d.iterrows():
                S = {k: r[k] / 1e3 for k in L.TENSOR_KEYS}
                R = T.resolve(S, T.STRIKE, 40.0)
                Rn = T.resolve({k: r[f"{k}_near"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                rows.append(dict(receiver=rec, cons=c, R0=r.R0, tau_amp=abs(R["tau"]), tau_ph=np.angle(R["tau"], deg=True),
                                 sn_amp=abs(R["sn"]), sn_ph=np.angle(R["sn"], deg=True), mean_amp=abs(R["mean"]),
                                 mean_ph=np.angle(R["mean"], deg=True), near_tau_amp=abs(Rn["tau"]), near_sn_amp=abs(Rn["sn"])))
    R0T = pd.DataFrame(rows)
    R0T.to_csv(f"{OUT}/convergence_R0.csv", index=False)
    print(R0T[R0T.receiver.isin(["famcentroid", "centroid"])].to_string())
    gc = pd.read_pickle(f"{OUT}/grid_convergence.pkl")
    gr = []
    for _, r in gc.iterrows():
        R = T.resolve({k: r[k] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
        gr.append(dict(res_deg=r.res_deg, receiver=r.receiver, cons=r.cons, tau_amp=abs(R["tau"]), tau_ph=np.angle(R["tau"], deg=True),
                       sn_amp=abs(R["sn"]), sn_ph=np.angle(R["sn"], deg=True), mean_amp=abs(R["mean"])))
    GR = pd.DataFrame(gr)
    GR.to_csv(f"{OUT}/convergence_grid.csv", index=False)
    print(GR.to_string())
    orr = pd.read_pickle(f"{OUT}/outer_radius.pkl")
    oo = []
    for _, r in orr.iterrows():
        R = T.resolve({k: r[f"far_{k}"] / 1e3 if f"far_{k}" in r else 0.0 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
        oo.append(dict(cons=r.cons, Rout_deg=r.Rout_deg, far_tau_amp=abs(R["tau"]), far_tau_ph=np.angle(R["tau"], deg=True),
                       far_sn_amp=abs(R["sn"]), far_sn_ph=np.angle(R["sn"], deg=True)))
    OO = pd.DataFrame(oo)
    OO.to_csv(f"{OUT}/convergence_outer_radius.csv", index=False)
    print(OO.to_string())
    # surface check: Boussinesq (mu_F) near field + nloadf far field vs full nloadf at surface receivers
    sc = []
    for rec in ("B045", "B933", "B934", "B935", "centroid_surface"):
        for c in ("M2", "O1"):
            d = ph[(ph.receiver == rec) & (ph.model == "usawest+tpxo72") & (ph.cons == c)]
            for _, r in d.iterrows():
                full = np.array([r.spotl_full_eEE, r.spotl_full_eNN, r.spotl_full_eEN])
                hyb = np.array([r.near_eEE + r.far_eEE, r.near_eNN + r.far_eNN, r.near_eEN + r.far_eEN])
                inn = np.array([r.spotl_in_eEE, r.spotl_in_eNN, r.spotl_in_eEN])
                nb = np.array([r.near_eEE, r.near_eNN, r.near_eEN])
                sc.append(dict(receiver=rec, cons=c, R0=r.R0, full_norm=np.linalg.norm(full),
                               rel_diff_bouss_plus_far_vs_full=np.linalg.norm(hyb - full) / np.linalg.norm(full),
                               near_bouss_over_near_farrell=np.linalg.norm(nb) / np.linalg.norm(inn),
                               far_plus_in_vs_full=np.linalg.norm(inn + np.array([r.far_eEE, r.far_eNN, r.far_eEN]) - full) / np.linalg.norm(full)))
    SC = pd.DataFrame(sc)
    SC.to_csv(f"{OUT}/surface_check.csv", index=False)
    print(SC.to_string())


# ===================================================================== stress, converged formulation
RB_LIST = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
RB_MAIN = 4.0


def _full_job(a):
    rd, m, c = a
    return far_field(rd, m, c, 0.0)


def stage_stress2():
    """Converged ocean-load stress at depth (additive formulation):
         S(z) = sum_{r<RB} [B(z) - B(0)] q  +  PlaneStress( e_SPOTL(full globe, surface); mu_F )
    B(z) = Boussinesq stress at depth z, B(0) = Boussinesq surface stress (both closed form, independent of
    the shear modulus), e_SPOTL = nloadf surface strain (spherical, layered Gutenberg-Bullen Earth, Farrell
    Green's functions) from the whole ocean.  The SPOTL term carries the sphericity/layering of the surface
    field; the Boussinesq difference carries the depth dependence (incl. sigma_zz, sigma_xz, and the mean
    stress at depth that is absent at the surface).  The difference kernel decays as z/r^3, so the sum
    converges with RB (tested 0.25-5 deg)."""
    recs = receivers()
    models = list(L.MODELS)
    z = np.load(f"{OUT}/near_grid_1_64.npz")
    la, lo, area = z["la"], z["lo"], z["area"]
    loads = {(m, c): z[f"{m}|{c}"].astype(complex) for m in models for c in CONS8}
    jobs = [(rec.to_dict(), m, c) for _, rec in recs.iterrows() for m in models for c in CONS8]
    t0 = time.time()
    with ProcessPoolExecutor(NPROC) as ex:
        fz = list(ex.map(_full_job, jobs, chunksize=8))
    full = {(j[0]["name"], j[1], j[2]): v for j, v in zip(jobs, fz)}
    print(f"nloadf full {len(jobs)} runs {time.time() - t0:.0f}s", flush=True)
    rows = []
    for _, rec in recs.iterrows():
        d, az = L.gc_dist_az(rec["lat"], rec["lon"], la, lo)
        r = np.deg2rad(d) * L.A_E
        sE, sN = r * np.sin(np.deg2rad(az)), r * np.cos(np.deg2rad(az))
        P = T.G0 * area
        Kz = L.bouss_point_stress(-sE, -sN, rec["depth_km"] * 1e3, P, NU)
        K0 = L.bouss_point_stress(-sE, -sN, 1e-3, P, NU)
        Kd = {k: Kz[k] - K0[k] for k in L.TENSOR_KEYS}
        for m in models:
            for c in CONS8:
                q = loads[(m, c)]
                eF = full[(rec["name"], m, c)]
                SF = strain_to_stress_Z(eF, MU_F)
                SF45 = strain_to_stress_Z(eF, 45e9)
                for RB in RB_LIST:
                    sel = d < RB
                    corr = {k: np.conj(np.sum(Kd[k][sel] * q[sel])) for k in L.TENSOR_KEYS}
                    row = dict(receiver=rec["name"], model=m, cons=c, RB=RB, eE=eF[0], eN=eF[1], eEN=eF[2])
                    for k in L.TENSOR_KEYS:
                        row[k] = corr[k] + SF[k]
                        row[f"{k}_corr"] = corr[k]
                        row[f"{k}_spotl"] = SF[k]
                        row[f"{k}_mu45"] = corr[k] + SF45[k]
                    rows.append(row)
        print("assembled", rec["name"], flush=True)
    ph = pd.DataFrame(rows)
    ph.to_pickle(f"{OUT}/stress_phasors_v2.pkl")
    cv = []
    for rec in ("famcentroid", "fam23", "fam26", "fam3", "centroid"):
        for c in ("M2", "K1", "O1"):
            for m in ("usawest+tpxo72",):
                d = ph[(ph.receiver == rec) & (ph.model == m) & (ph.cons == c)].sort_values("RB")
                for _, r in d.iterrows():
                    R = T.resolve({k: r[k] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                    Rs = T.resolve({k: r[f"{k}_spotl"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                    cv.append(dict(receiver=rec, cons=c, RB=r.RB, tau_amp=abs(R["tau"]), tau_ph=np.angle(R["tau"], deg=True),
                                   sn_amp=abs(R["sn"]), sn_ph=np.angle(R["sn"], deg=True), mean_amp=abs(R["mean"]),
                                   mean_ph=np.angle(R["mean"], deg=True), spotl_surface_tau_amp=abs(Rs["tau"]),
                                   spotl_surface_sn_amp=abs(Rs["sn"]), spotl_surface_mean_amp=abs(Rs["mean"]),
                                   spotl_surface_mean_ph=np.angle(Rs["mean"], deg=True)))
    CV = pd.DataFrame(cv)
    CV.to_csv(f"{OUT}/convergence_RB_v2.csv", index=False)
    pd.set_option("display.width", 220)
    print(CV.round(3).to_string())


# ===================================================================== far-field depth correction (spherical, layered)
RB3 = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
RB3_MAIN = 1.0


def _farcorr_job(a):
    rec, mfile, spec, c = a
    gr = L.read_green("gr.gbaver.wef.p02.ce")
    cells = L.ring_cells(rec["lat"], rec["lon"], gr, dmin=min(RB3) - 1e-9)
    la, lo = L.dest_point(rec["lat"], rec["lon"], cells["dist"], cells["az"])
    amp = np.zeros(len(la), complex)
    polys = L.poly_list(spec) if spec else None
    for fl in ("F", "C"):
        m = cells["flag"] == fl
        if m.any():
            amp[m] = L.ocpts(f"{c.lower()}.{mfile}", la[m], lo[m], flag=fl, polys=polys)
    dv = L.vertical_gf_spline(gr)
    out = {}
    for RB in RB3:
        sel = cells["dist"] >= RB
        sub = {k: v[sel] for k, v in cells.items()}
        out[RB] = L.depth_strain_correction(sub, amp[sel], dv, z=rec["depth_km"] * 1e3)
        out[("surf", RB)] = L.py_surface_strain(sub, amp[sel])
    return out


def stage_farcorr():
    """S(z) = sum_{r<RB}[B(z)-B(0)] + PlaneStress(e_SPOTL full) + PlaneStress(z * Hessian(u_SPOTL, r>RB))."""
    recs = receivers()
    v2 = pd.read_pickle(f"{OUT}/stress_phasors_v2.pkl")
    jobs, keys = [], []
    for _, rec in recs.iterrows():
        for m in L.MODELS:
            for mfile, spec in L.MODELS[m]:
                for c in CONS8:
                    jobs.append((rec.to_dict(), mfile, spec, c))
                    keys.append((rec["name"], m, c))
    t0 = time.time()
    with ProcessPoolExecutor(NPROC) as ex:
        res = list(ex.map(_farcorr_job, jobs, chunksize=4))
    print(f"farcorr {len(jobs)} jobs {time.time() - t0:.0f}s", flush=True)
    accs = {}
    for k, r in zip(keys, res):
        for RB in RB3:
            v = np.array([r[("surf", RB)]["eEE"], r[("surf", RB)]["eNN"], r[("surf", RB)]["eEN"]])
            accs[k + (RB,)] = accs.get(k + (RB,), 0) + v
    acc = {}
    for k, r in zip(keys, res):
        if k not in acc:
            acc[k] = {RB: np.array([r[RB]["eEE"], r[RB]["eNN"], r[RB]["eEN"]]) for RB in RB3}
        else:
            for RB in RB3:
                acc[k][RB] = acc[k][RB] + np.array([r[RB]["eEE"], r[RB]["eNN"], r[RB]["eEN"]])
    rows = []
    for _, r in v2.iterrows():
        if r.RB not in RB3:
            continue
        e = acc[(r.receiver, r.model, r.cons)][r.RB]
        Sfc = L.plane_stress_from_strain(e[0], e[1], e[2], MU_F, NU)
        Sfc45 = L.plane_stress_from_strain(e[0], e[1], e[2], 45e9, NU)
        row = dict(receiver=r.receiver, model=r.model, cons=r.cons, RB=r.RB)
        for k in L.TENSOR_KEYS:
            row[f"{k}_corrB"] = r[f"{k}_corr"]
            row[f"{k}_spotl"] = r[f"{k}_spotl"]
            row[f"{k}_farcorr"] = Sfc[k]
            row[k] = r[f"{k}_corr"] + r[f"{k}_spotl"] + Sfc[k]
            row[f"{k}_halfspace"] = r[k]      # half-space depth correction to RB, nothing beyond
            row[f"{k}_mu45"] = r[f"{k}_corr"] + (r[f"{k}_mu45"] - r[f"{k}_corr"]) + Sfc45[k]
        rows.append(row)
    ph = pd.DataFrame(rows)
    ph.to_pickle(f"{OUT}/stress_phasors_v3.pkl")
    pd.to_pickle(dict(curv=acc, surf=accs), f"{OUT}/farfield_surface_strain.pkl")
    cv = []
    for rec in ("famcentroid", "fam23", "fam26", "fam8", "fam27"):
        for c in ("M2", "K1", "O1"):
            d = ph[(ph.receiver == rec) & (ph.model == "usawest+tpxo72") & (ph.cons == c)].sort_values("RB")
            for _, r in d.iterrows():
                R = T.resolve({k: r[k] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                Rb = T.resolve({k: r[f"{k}_corrB"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                Rf = T.resolve({k: r[f"{k}_farcorr"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                Rs = T.resolve({k: r[f"{k}_spotl"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                cv.append(dict(receiver=rec, cons=c, RB=r.RB, tau_amp=abs(R["tau"]), tau_ph=np.angle(R["tau"], deg=True),
                               sn_amp=abs(R["sn"]), sn_ph=np.angle(R["sn"], deg=True), mean_amp=abs(R["mean"]),
                               mean_ph=np.angle(R["mean"], deg=True),
                               bouss_tau=abs(Rb["tau"]), bouss_mean=abs(Rb["mean"]), farcorr_tau=abs(Rf["tau"]),
                               farcorr_mean=abs(Rf["mean"]), surf_tau=abs(Rs["tau"]), surf_mean=abs(Rs["mean"])))
    CV = pd.DataFrame(cv)
    CV.to_csv(f"{OUT}/convergence_RB_v3.csv", index=False)
    pd.set_option("display.width", 220)
    print(CV.round(3).to_string())


# ===================================================================== layered (FE) near field + Farrell far field
RL_LIST = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
RL_MAIN = 1.0


def stage_fe():
    import sci_tidal_spotl_fe as FE
    out = {}
    for model in ("furlong1d", "uniform"):
        t0 = time.time()
        fe = FE.solve(model, rmax=6000.0, zmax=6000.0, r_fine=300.0)
        j = (fe["zc"] > 15) & (fe["zc"] < 40)
        out[model] = dict(rc=fe["rc"], zc=fe["zc"][j], **{k: fe[k][:, j] for k in ("srr", "stt", "szz", "srz")},
                          mu_c=fe["mu_c"][j], lam_c=fe["lam_c"][j], surf_r=fe["surf"]["r"], surf_ur=fe["surf"]["ur"],
                          surf_uz=fe["surf"]["uz"])
        print(model, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"{OUT}/fe_kernels.npz", **{f"{m}|{k}": v for m, d in out.items() for k, v in d.items()})
    # validation table: uniform FE vs Boussinesq, layered FE vs Boussinesq, at 28 km
    rows = []
    for model in ("uniform", "furlong1d"):
        d = out[model]
        for zq in (22.0, 28.0, 32.0):
            jj = np.searchsorted(d["zc"], zq) - 1
            w = (zq - d["zc"][jj]) / (d["zc"][jj + 1] - d["zc"][jj])
            for rq in (25, 40, 60, 100, 150, 200, 300):
                i = int(np.argmin(abs(d["rc"] - rq)))
                B = L.bouss_point_stress(np.array([d["rc"][i] * 1e3]), np.array([0.0]), zq * 1e3, 1.0)
                K = {k: (1 - w) * d[k][i, jj] + w * d[k][i, jj + 1] for k in ("srr", "stt", "szz", "srz")}
                rows.append(dict(model=model, z_km=zq, r_km=d["rc"][i], srr_over_B=K["srr"] / B["see"][0],
                                 stt_over_B=K["stt"] / B["snn"][0], szz_over_B=K["szz"] / B["sdd"][0],
                                 srz_over_B=K["srz"] / B["sed"][0],
                                 mean_over_B=(K["srr"] + K["stt"] + K["szz"]) / (B["see"][0] + B["snn"][0] + B["sdd"][0])))
    V = pd.DataFrame(rows)
    V.to_csv(f"{OUT}/fe_validation.csv", index=False)
    print(V.round(3).to_string())


def local_moduli(z_km):
    import sci_tidal_spotl_fe as FE
    mu, lam = FE.elastic_model("furlong1d")(np.array([z_km]))
    nu = lam[0] / (2 * (lam[0] + mu[0]))
    return float(mu[0]), float(nu)


def stage_stress4():
    """Final ocean-load stress phasors.  Primary ('layered'): FE layered half-space (Furlong 1-D) for loads within
    RL, SPOTL/Farrell surface strain + first-order depth (curvature) correction for loads beyond RL, converted
    with the local moduli at the receiver depth.  Also carries the homogeneous variants of v3."""
    recs = receivers()
    z = np.load(f"{OUT}/near_grid_1_64.npz")
    la, lo, area = z["la"], z["lo"], z["area"]
    models = list(L.MODELS)
    loads = {(m, c): z[f"{m}|{c}"].astype(complex) for m in models for c in CONS8}
    fk = np.load(f"{OUT}/fe_kernels.npz")
    rc, zc = fk["furlong1d|rc"], fk["furlong1d|zc"]
    ff = pd.read_pickle(f"{OUT}/farfield_surface_strain.pkl")
    v3 = pd.read_pickle(f"{OUT}/stress_phasors_v3.pkl")
    rows = []
    mods = []
    for _, rec in recs.iterrows():
        if rec["depth_km"] < 15:
            continue
        zq = rec["depth_km"]
        mu_l, nu_l = local_moduli(zq)
        mods.append(dict(receiver=rec["name"], depth_km=zq, mu_local=mu_l, nu_local=nu_l))
        jj = int(np.clip(np.searchsorted(zc, zq) - 1, 0, len(zc) - 2))
        w = (zq - zc[jj]) / (zc[jj + 1] - zc[jj])
        Kr = {k: (1 - w) * fk[f"furlong1d|{k}"][:, jj] + w * fk[f"furlong1d|{k}"][:, jj + 1] for k in ("srr", "stt", "szz", "srz")}
        d, az = L.gc_dist_az(rec["lat"], rec["lon"], la, lo)
        rkm = np.deg2rad(d) * L.A_E / 1e3
        sel0 = d < max(RL_LIST) + 0.01
        P = T.G0 * area[sel0]
        cE, cN = -np.sin(np.deg2rad(az[sel0])), -np.cos(np.deg2rad(az[sel0]))    # unit vector source -> receiver
        srr = np.interp(rkm[sel0], rc, Kr["srr"]) * P
        stt = np.interp(rkm[sel0], rc, Kr["stt"]) * P
        szz = np.interp(rkm[sel0], rc, Kr["szz"]) * P
        srz = np.interp(rkm[sel0], rc, Kr["srz"]) * P
        K = dict(snn=srr * cN ** 2 + stt * cE ** 2, see=srr * cE ** 2 + stt * cN ** 2, sne=(srr - stt) * cE * cN,
                 sdd=szz, snd=srz * cN, sed=srz * cE)
        dd = d[sel0]
        for m in models:
            for c in CONS8:
                q = loads[(m, c)][sel0]
                for RLv in RL_LIST:
                    sel = dd < RLv
                    near = {k: np.conj(np.sum(K[k][sel] * q[sel])) for k in L.TENSOR_KEYS}
                    es = ff["surf"][(rec["name"], m, c, RLv)]
                    ec = ff["curv"][(rec["name"], m, c)][RLv]
                    # es is in nanostrain (py_surface_strain), ec in strain (depth_strain_correction)
                    Sf = L.plane_stress_from_strain(es[0] * 1e-9 + ec[0], es[1] * 1e-9 + ec[1], es[2] * 1e-9 + ec[2], mu_l, nu_l)
                    row = dict(receiver=rec["name"], model=m, cons=c, variant=f"layered_RL{RLv:g}")
                    for k in L.TENSOR_KEYS:
                        row[k] = near[k] + Sf[k]
                        row[f"{k}_near"] = near[k]
                        row[f"{k}_far"] = Sf[k]
                    rows.append(row)
        print("assembled", rec["name"], flush=True)
    # homogeneous variants from v3 (Boussinesq to RB + SPOTL surface + curvature beyond RB, mu_F)
    for RB in (1.0, 5.0):
        d3 = v3[v3.RB == RB]
        for _, r in d3.iterrows():
            row = dict(receiver=r.receiver, model=r.model, cons=r.cons, variant=f"homog_RB{RB:g}")
            for k in L.TENSOR_KEYS:
                row[k] = r[k]
            rows.append(row)
    ph = pd.DataFrame(rows)
    ph.to_pickle(f"{OUT}/stress_phasors_final.pkl")
    pd.DataFrame(mods).to_csv(f"{OUT}/receiver_local_moduli.csv", index=False)
    cv = []
    for rec in ("famcentroid", "fam23", "fam26", "fam8", "fam27", "fam3"):
        for c in ("M2", "K1", "O1", "N2", "S2"):
            dsel = ph[(ph.receiver == rec) & (ph.model == "usawest+tpxo72") & (ph.cons == c)]
            for _, r in dsel.iterrows():
                R = T.resolve({k: r[k] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
                cv.append(dict(receiver=rec, cons=c, variant=r.variant, tau_amp=abs(R["tau"]), tau_ph=np.angle(R["tau"], deg=True),
                               sn_amp=abs(R["sn"]), sn_ph=np.angle(R["sn"], deg=True), mean_amp=abs(R["mean"]),
                               mean_ph=np.angle(R["mean"], deg=True)))
    CV = pd.DataFrame(cv)
    CV.to_csv(f"{OUT}/convergence_final.csv", index=False)
    pd.set_option("display.width", 220)
    print(CV[CV.receiver.isin(["famcentroid", "fam26", "fam27"]) & CV.cons.isin(["M2", "K1", "O1"])].round(3).to_string())


# ===================================================================== undrained poroelastic Coulomb (Shelly & Materna 2026 form)
B_GRID = np.round(np.arange(0.0, 1.001, 0.1), 2)
MU_GRID = np.round(np.arange(0.0, 0.801, 0.05), 3)


def cfs_poro(tau, sn, mean, B, mu):
    """Shelly & Materna (2026) eqs. 1-2, tension positive throughout:
       p = -B * sigma_kk/3 (pore-pressure change, compression of the rock raises p);
       d sigma_eff (unclamping +) = d sigma_n + p = d sigma_n - B * mean ;  CFS = tau + mu * d sigma_eff."""
    return tau + mu * (sn - B * mean)


def poro_sign_tests():
    out = {}
    # (1) isotropic compression P=1 kPa, B=1: sigma_n = mean = -1 -> effective normal stress change 0
    out["isotropic_B1_dseff"] = float(-1.0 - 1.0 * (-1.0))
    # (2) uniaxial vertical load under the ocean (Boussinesq directly below a point load): mean < 0 -> p > 0
    S = L.bouss_point_stress(np.array([1.0]), np.array([0.0]), 28e3, 1e9)
    mean = float((S["snn"] + S["see"] + S["sdd"])[0] / 3)
    out["bouss_below_load_mean_Pa"] = mean
    out["bouss_below_load_pore_pressure_B1_Pa"] = -mean
    # (3) reproduce the dose-response convention: dp = -B tr/3, dsn_eff = dsn + dp
    d = pd.read_csv(f"{ROOT}/data/sci/dose_response/m7_per_family_stress_vs_gap.csv").iloc[0]
    out["dose_response_check_dsn_eff"] = [float(d.dsn + 0.7 * d.dP), float(d.dsn_eff)]
    out["dose_response_check_dCFS_ud_mu0.6"] = [float(d.dtau + 0.6 * (d.dsn + 0.7 * d.dP)), float(d.dCFS_ud)]
    assert abs(out["isotropic_B1_dseff"]) < 1e-12 and mean < 0
    assert abs(out["dose_response_check_dsn_eff"][0] - out["dose_response_check_dsn_eff"][1]) < 1e-9
    return out


def poro_surface(cx, X, C, rows, W=None, Bg=B_GRID, mug=MU_GRID):
    """Grid over (B, mu): conditional-Poisson fit of rate ~ exp(beta * CFS_poro).  Returns DataFrame."""
    tau, sn, mean = (M_(cx, X[k])[rows] for k in ("tau", "sn", "mean"))
    f0 = T.cfit(C, [], w=W)
    LL0 = f0["LL"]
    out = []
    for B in Bg:
        for mu in mug:
            x = tau + mu * (sn - B * mean)
            f = T.cfit(C, [x], w=W, iters=10, theta0=np.r_[f0["theta"], 0.0])
            out.append(dict(B=B, mu=mu, beta=float(f["beta"][0]), dLL=f["LL"] - LL0))
    return pd.DataFrame(out)


def stage_poro():
    t00 = time.time()
    res = dict(sign_tests=poro_sign_tests())
    print(res["sign_tests"], flush=True)
    ph = pd.read_pickle(f"{OUT}/stress_phasors_final.pkl")
    mods = pd.read_csv(f"{OUT}/receiver_local_moduli.csv").set_index("receiver")
    cx = setup_time()
    NET = "famcentroid"
    body_loc = body_stress(cx, mods.loc[NET, "mu_local"], mods.loc[NET, "nu_local"])
    body_30 = body_stress(cx, 30e9, 0.25)
    rows = cx.rows
    C = cx.Cnet[rows]
    cfgs = {"primary": (ocean_phasors(ph, NET, "usawest+tpxo72", "layered_RL2"), body_loc),
            "layered_RL1": (ocean_phasors(ph, NET, "usawest+tpxo72", "layered_RL1"), body_loc),
            "layered_body30": (ocean_phasors(ph, NET, "usawest+tpxo72", "layered_RL2"), body_30),
            "fes2004": (ocean_phasors(ph, NET, "fes2004", "layered_RL2"), body_loc),
            "homog_RB1_body30": (ocean_phasors(ph, NET, "usawest+tpxo72", "homog_RB1"), body_30),
            "body_only_local": (None, body_loc)}
    surf = []
    Xs = {}
    reuse = "--reuse" in sys.argv and os.path.exists(f"{OUT}/poro_surface_network.csv")
    S_old = pd.read_csv(f"{OUT}/poro_surface_network.csv") if reuse else None
    for name, (oph, bdy) in cfgs.items():
        X = stress_set(cx, oph, body=bdy)
        Xs[name] = X
        if reuse and name in set(S_old.config):
            d = S_old[S_old.config == name][["B", "mu", "beta", "dLL"]].reset_index(drop=True)
        else:
            d = poro_surface(cx, X, C, rows)
        d["config"] = name
        surf.append(d)
        i = d.dLL.idxmax()
        ref = d[(d.B == 1.0) & (np.isclose(d.mu, 0.6))].iloc[0]
        print(name, "best", d.loc[i].to_dict(), "B1 mu0.6", ref.to_dict(), f"{time.time() - t00:.0f}s", flush=True)
        # amplitude of the pieces (M2)
        Rp = X["ph"]["M2"]
        res.setdefault("M2_amplitudes_kPa", {})[name] = dict(tau=abs(Rp["tau"]), sn=abs(Rp["sn"]), mean=abs(Rp["mean"]),
                                                            sn_minus_mean=abs(Rp["sn"] - Rp["mean"]),
                                                            phase_tau=float(np.angle(Rp["tau"], deg=True)),
                                                            phase_sn=float(np.angle(Rp["sn"], deg=True)),
                                                            phase_mean=float(np.angle(Rp["mean"], deg=True)))
    S = pd.concat(surf)
    S.to_csv(f"{OUT}/poro_surface_network.csv", index=False)
    # overdispersion factor (week-cluster vs Poisson variance of the tau coefficient, primary)
    X = Xs["primary"]
    ft = T.cfit(C, [M_(cx, X["tau"])[rows]])
    Vw, Vn = T.robust_se(ft, rows // 7)
    phi = float(Vw[0, 0] / Vn[0, 0])
    res["overdispersion_phi"] = phi
    # bootstrap of dLL(best) - dLL(B=1, mu=0.6), and of the best (B, mu)
    S["dLL_pos"] = np.where(S.beta > 0, S.dLL, 0.0)
    S.to_csv(f"{OUT}/poro_surface_network.csv", index=False)
    Sp = S[S.config == "primary"].copy()
    Sp["dLL"] = Sp["dLL_pos"]
    ib = Sp.dLL.idxmax()
    bestB, bestmu = float(Sp.B[ib]), float(Sp.mu[ib])
    tau, sn, mean = (M_(cx, X[k])[rows] for k in ("tau", "sn", "mean"))
    NB = int(os.environ.get("PORO_NB", "40"))
    Wb = rng.poisson(1.0, (NB, len(rows))).astype(float)
    bd, bb = [], []
    for b in range(NB):
        d = poro_surface(cx, X, C, rows, W=Wb[b], Bg=np.round(np.arange(0, 1.01, 0.2), 2), mug=np.round(np.arange(0, 0.81, 0.1), 2))
        j = d.dLL.idxmax()
        bb.append((float(d.B[j]), float(d.mu[j])))
        ref = float(d[(d.B == 1.0) & np.isclose(d.mu, 0.6)].dLL.iloc[0])
        bd.append(float(d.dLL[j]) - ref)
        if b % 10 == 0:
            print("poro boot", b, bb[-1], bd[-1], f"{time.time() - t00:.0f}s", flush=True)
    bb = np.array(bb) if len(bb) else np.full((1, 2), np.nan)
    bd = bd if len(bd) else [np.nan]
    res["primary"] = dict(best_B=bestB, best_mu=bestmu, best_dLL=float(Sp.dLL[ib]), best_beta=float(Sp.beta[ib]),
                          a_sigma_best=float(1 / Sp.beta[ib]) if Sp.beta[ib] > 0 else None,
                          dLL_B1_mu06=float(Sp[(Sp.B == 1.0) & np.isclose(Sp.mu, 0.6)].dLL.iloc[0]),
                          beta_B1_mu06=float(Sp[(Sp.B == 1.0) & np.isclose(Sp.mu, 0.6)].beta.iloc[0]),
                          delta_LL_best_minus_B1mu06=float(Sp.dLL[ib] - Sp[(Sp.B == 1.0) & np.isclose(Sp.mu, 0.6)].dLL.iloc[0]),
                          delta_LL_over_phi=float((Sp.dLL[ib] - Sp[(Sp.B == 1.0) & np.isclose(Sp.mu, 0.6)].dLL.iloc[0]) / phi),
                          boot_best_B_median=float(np.median(bb[:, 0])), boot_best_B_95=[float(np.percentile(bb[:, 0], 2.5)), float(np.percentile(bb[:, 0], 97.5))],
                          boot_best_mu_median=float(np.median(bb[:, 1])), boot_best_mu_95=[float(np.percentile(bb[:, 1], 2.5)), float(np.percentile(bb[:, 1], 97.5))],
                          boot_dLL_best_minus_B1mu06_95=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                          boot_frac_B1mu06_within_best_le0=float(np.mean(np.array(bd) <= 0)))
    # profile: best mu at each B (and the implied effective friction mu*(1-B))
    prof = Sp.loc[Sp.groupby("B").dLL.idxmax()][["B", "mu", "dLL", "beta"]]
    prof["mu_times_1_minus_B"] = prof.mu * (1 - prof.B)
    res["primary_profile_over_B"] = prof.to_dict(orient="records")
    print(prof.to_string(), flush=True)
    # ---------------- joint test with the M7 (dose_response): CFS_poro(B, mu) per family at the M7
    m7 = pd.read_csv(f"{ROOT}/data/sci/dose_response/m7_per_family_stress_vs_gap.csv")
    jt = []
    for B in B_GRID:
        for mu in MU_GRID:
            cfs = m7.dtau + mu * (m7.dsn + B * m7.dP)      # dP = -tr/3 (compression +): dsn_eff = dsn + B dP
            row_t = Sp[(Sp.B == B) & np.isclose(Sp.mu, mu)].iloc[0]
            beta = row_t.beta
            pred = np.exp(beta * cfs)                      # instantaneous rate ratio implied by the tidal beta
            jt.append(dict(B=B, mu=mu, tide_dLL=row_t.dLL, tide_dLL_minus_best=row_t.dLL - Sp.dLL.max(),
                           tide_dLL_minus_best_over_phi=(row_t.dLL - Sp.dLL.max()) / phi,
                           m7_mu_crit_median=float(np.median(-m7.dtau / (m7.dsn + B * m7.dP))), tide_beta=beta, m7_cfs_median=float(np.median(cfs)),
                           m7_frac_families_negative=float(np.mean(cfs < 0)),
                           m7_pred_rate_ratio_median=float(np.median(pred)), m7_obs_ratio30_median=float(np.median(m7.ratio30)),
                           m7_corr_cfs_vs_gap=float(np.corrcoef(cfs, m7.gap_d)[0, 1])))
    JT = pd.DataFrame(jt)
    JT.to_csv(f"{OUT}/poro_joint_tide_m7.csv", index=False)
    res["m7_note"] = ("M7 stresses from scripts/sci_dose_response.py (Okada, mu 30 GPa homogeneous half-space, USGS FFM, "
                      "family centroids); M7 cessation needs CFS_poro < 0 at most families.")
    ok = JT[(JT.m7_frac_families_negative >= 0.8)]
    res["joint"] = dict(
        tide_best_among_M7_consistent=ok.loc[ok.tide_dLL.idxmax()].to_dict() if len(ok) else None,
        tide_global_best=JT.loc[JT.tide_dLL.idxmax()].to_dict(),
        B1_mu06=JT[(JT.B == 1.0) & np.isclose(JT.mu, 0.6)].iloc[0].to_dict())
    print(json.dumps(res["joint"], indent=1, default=float), flush=True)
    json.dump(res, open(f"{OUT}/poro_summary.json", "w"), indent=1, default=float)
    print("poro done", f"{time.time() - t00:.0f}s", flush=True)


# ===================================================================== figures
def stage_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    C1, C2, C3, C4, GREY = T.C1, T.C2, T.C3, T.C4, T.GREY
    # ---- Fig 1: validation
    NV = pd.read_csv(f"{OUT}/validation_noaa.csv")
    CV = pd.read_csv(f"{OUT}/validation_strainmeter_calibration.csv")
    SV = pd.read_csv(f"{OUT}/validation_strainmeter_gauges.csv")
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    for m, col in (("usawest+tpxo72", C1), ("tpxo72", C2), ("fes2004", C3), ("got4p7", C4)):
        d = NV[(NV.model == m) & (NV.cons == "M2")].sort_values("lat")
        ax[0, 0].plot(d.lat, d.amp_ratio, "o-", color=col, ms=3, label=m)
        ax[0, 1].plot(d.lat, d.dG, "o-", color=col, ms=3, label=m)
    ax[0, 0].axhline(1, color=GREY, lw=0.5); ax[0, 1].axhline(0, color=GREY, lw=0.5)
    ax[0, 0].set_ylabel("M2 amplitude model / NOAA"); ax[0, 1].set_ylabel("M2 phase NOAA - model (deg)")
    for a in ax[0]:
        a.set_xlabel("gauge latitude"); a.axvline(40.06, color=GREY, ls=":", lw=0.8)
    ax[0, 0].legend(frameon=False); ax[0, 0].set_title("SPOTL ocean models vs 11 NOAA tide gauges")
    ms = ["body_only", "usawest+tpxo72", "tpxo72", "fes2004", "got4p7", "eot11a", "hamtide11a", "dtu10", "nao99b", "tpxo72atlas"]
    agg = CV.groupby("model").iso_resid_frac.mean().reindex(ms)
    ax[1, 0].bar(range(len(ms)), agg.values, color=[GREY] + [C1] * (len(ms) - 1))
    ax[1, 0].set_xticks(range(len(ms))); ax[1, 0].set_xticklabels(ms, rotation=60, ha="right")
    ax[1, 0].set_ylabel("residual fraction (isotropic calibration)")
    ax[1, 0].set_title("PBO strainmeters, M2+N2+O1+K1: body vs body+load")
    for sta, mk in zip(STRAIN_STA, "osD^"):
        for m, col in (("body_only", GREY), ("usawest+tpxo72", C1)):
            d = SV[(SV.station == sta) & (SV.model == m) & SV.good & SV.cons.isin(["M2", "O1"])]
            ax[1, 1].plot(d.pred_phase, d.obs_phase, mk, color=col, ms=4, mfc="none" if m == "body_only" else col)
    ax[1, 1].plot([-180, 180], [-180, 180], color=GREY, lw=0.5)
    ax[1, 1].set_xlabel("predicted gauge phase (deg)"); ax[1, 1].set_ylabel("observed gauge phase (deg)")
    ax[1, 1].set_title("Gauge phases M2/O1 (open: body only; filled: + SPOTL load)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_tidal_spotl_validation.png", dpi=150); plt.close(fig)
    # ---- Fig 2: stress at depth: convergence + phasors
    CF = pd.read_csv(f"{OUT}/convergence_final.csv")
    H = pd.read_csv(f"{OUT}/stress_harmonics_network.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    d = CF[(CF.receiver == "famcentroid") & CF.variant.str.startswith("layered")].copy()  # primary = RL 2 deg
    d["RL"] = d.variant.str.replace("layered_RL", "").astype(float)
    for comp, col in (("tau", C1), ("sn", C2), ("mean", C3)):
        for c, ls in (("M2", "-"), ("O1", "--")):
            g = d[d.cons == c].sort_values("RL")
            ax[0].plot(g.RL, g[f"{comp}_amp"], ls, color=col, marker="o", ms=3, label=f"{comp} {c}")
    ax[0].set_xlabel("RL (deg): layered FE inside, SPOTL/Farrell outside"); ax[0].set_ylabel("ocean-load amplitude (kPa)")
    ax[0].legend(frameon=False, ncol=2); ax[0].set_title("Ocean load at LFE depth vs RL", fontsize=7)
    for k, (comp, col) in enumerate((("tau", C1), ("sn", C2), ("mean", C3))):
        for c, mk in (("M2", "o"), ("K1", "s"), ("O1", "^")):
            r = H[(H.cons == c) & (H.comp == comp)].iloc[0]
            for tag, a_, p_, fc in (("body", r.body_amp, r.body_phase, "none"), ("ocean", r.ocean_amp, r.ocean_phase, col),
                                    ("total", r.total_amp, r.total_phase, "k")):
                z = a_ * np.exp(1j * np.deg2rad(p_))
                ax[1 if comp != "mean" else 2].plot([0, z.real], [0, z.imag], color=col if tag != "total" else "k",
                                                    lw=1 if tag != "total" else 1.8, ls="-" if tag != "body" else ":")
                ax[1 if comp != "mean" else 2].plot(z.real, z.imag, mk, color=col, mfc=fc, ms=5)
    ax[1].set_title("M2 o K1 s O1 ^: tau blue, sigma_n orange\nbody dotted, ocean solid, total black", fontsize=7)
    ax[2].set_title("mean stress (tension +)", fontsize=7)
    for a in ax[1:]:
        a.axhline(0, color=GREY, lw=0.5); a.axvline(0, color=GREY, lw=0.5); a.set_aspect("equal")
        a.set_xlabel("Re (kPa)"); a.set_ylabel("Im (kPa)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_tidal_spotl_stress.png", dpi=150); plt.close(fig)
    # ---- Fig 3: fits
    P = pd.read_csv(f"{OUT}/network_mu_profiles.csv")
    OS = pd.read_csv(f"{OUT}/ocean_scale_profile.csv")
    SC = pd.read_csv(f"{OUT}/strike_scan_total.csv")
    PC = pd.read_csv(f"{OUT}/per_constituent_phasors.csv")
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    for cfg, col in (("body_only_mu30", GREY), ("primary", C1), ("homog_RB1_body30", C2), ("layered_RL1", C3), ("model_fes2004", C4)):
        for sense, ls in (("RL", "-"), ("LL", "--")):
            g = P[(P.config == cfg) & (P.sense == sense)]
            g = g.assign(dLLp=np.where(g.beta > 0, g.dLL, np.nan))
            ax[0, 0].plot(g.mu, g.dLLp, ls, color=col, label=f"{cfg} {sense}" if sense == "RL" else None)
    ax[0, 0].set_xlabel("mu' in CFS = (+/-)tau + mu' sigma_n  (solid RL, dashed LL)"); ax[0, 0].set_ylabel("dlogL")
    ax[0, 0].legend(frameon=False, fontsize=6)
    for dph, col in ((0, C1), (-30, C2), (30, C3)):
        g = OS[OS.dphase == dph]
        ax[0, 1].plot(g.scale, g.dLL_tau_sn, "o-", color=col, ms=3, label=f"phase shift {dph}")
    ax[0, 1].set_xlabel("ocean-load scale factor"); ax[0, 1].set_ylabel("dlogL (tau + sigma_n)")
    ax[0, 1].legend(frameon=False); ax[0, 1].set_title("Is the modelled load the one the LFEs feel?")
    ax[1, 0].plot(SC.strike, SC.dLL_tau, color=C1, label="tau")
    ax[1, 0].plot(SC.strike, SC.dLL_cfs, color=C2, label="CFS best mu'")
    ax[1, 0].axvline(T.STRIKE, color=GREY, ls="--"); ax[1, 0].set_xlabel("strike (dip 40, RL rake)")
    ax[1, 0].set_ylabel("dlogL"); ax[1, 0].legend(frameon=False)
    g = PC[(PC.comp == "tau")]
    for tag, mk in (("total", "o"), ("body", "s")):
        gg = g[g.stress == tag]
        ax[1, 1].errorbar(range(len(gg)), gg.lfe_minus_stress_deg, yerr=1.96 * gg.lfe_minus_stress_se_deg, fmt=mk,
                          color=C1 if tag == "total" else GREY, label=f"vs {tag} RL shear")
        ax[1, 1].set_xticks(range(len(gg))); ax[1, 1].set_xticklabels(gg.cons)
    ax[1, 1].axhline(0, color=GREY, lw=0.5); ax[1, 1].axhline(180, color=C2, lw=0.5); ax[1, 1].axhline(-180, color=C2, lw=0.5)
    ax[1, 1].set_ylabel("LFE phase - stress phase (deg); 0 = RL, +-180 = LL"); ax[1, 1].legend(frameon=False)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_tidal_spotl_fits.png", dpi=150); plt.close(fig)
    # ---- Fig 4: families
    FR = pd.read_csv(f"{OUT}/family_fits.csv")
    FR = FR[FR.n_units >= 300]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for tag, col in (("body", GREY), ("total", C1)):
        ax[0].errorbar(FR.lon, FR[f"{tag}_mu_eff"], yerr=FR[f"{tag}_mu_eff_se"], fmt="o", color=col, ms=3, label=tag)
        ax[1].errorbar(FR.lon, FR[f"{tag}_lag_deg"], yerr=FR[f"{tag}_lag_se"], fmt="o", color=col, ms=3, label=tag)
        ax[2].plot(FR.lon, FR[f"{tag}_a_sigma_tau"], "o", color=col, ms=3, label=tag)
    ax[0].set_ylabel("mu' (tau + sigma_n fit)"); ax[1].set_ylabel("phase lag vs tau (deg)"); ax[2].set_ylabel("a sigma (kPa), tau only")
    for a in ax:
        a.set_xlabel("family longitude"); a.legend(frameon=False); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_tidal_spotl_families.png", dpi=150); plt.close(fig)
    # ---- Fig 5: poroelastic surface
    if os.path.exists(f"{OUT}/poro_surface_network.csv"):
        S = pd.read_csv(f"{OUT}/poro_surface_network.csv")
        JT = pd.read_csv(f"{OUT}/poro_joint_tide_m7.csv")
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
        for a, cfg in zip(ax[:2], ("primary", "layered_body30")):
            g = S[S.config == cfg].pivot(index="mu", columns="B", values="dLL")
            gb = S[S.config == cfg].pivot(index="mu", columns="B", values="beta")
            Z = np.where(gb.values > 0, g.values, np.nan)
            Z = Z - np.nanmax(Z)
            im = a.pcolormesh(g.columns, g.index, np.maximum(Z, -400), shading="nearest", cmap="viridis")
            a.contour(g.columns, g.index, Z, levels=[-8 * 3, -8], colors="w", linewidths=0.8)
            a.plot(1.0, 0.6, "r*", ms=10)
            a.set_xlabel("Skempton B"); a.set_ylabel("intrinsic mu"); a.set_title(f"tides: dlogL - max ({cfg})")
            plt.colorbar(im, ax=a)
        g = JT.pivot(index="mu", columns="B", values="m7_frac_families_negative")
        im = ax[2].pcolormesh(g.columns, g.index, g.values, shading="nearest", cmap="RdBu_r")
        ax[2].plot(1.0, 0.6, "r*", ms=10)
        ax[2].set_xlabel("Skempton B"); ax[2].set_ylabel("intrinsic mu"); ax[2].set_title("M7: fraction of families with CFS_poro < 0")
        plt.colorbar(im, ax=ax[2])
        fig.tight_layout(); fig.savefig(f"{FIG}/sci_tidal_spotl_poro.png", dpi=150); plt.close(fig)
    print("figs done")


# ===================================================================== Shelly et al. (2026) dip x rake scan
def tensor_series(cx, oph, body):
    S = {}
    for k in L.TENSOR_KEYS:
        oc = series_from_ph(cx, {c: v / 1e3 for c, v in oph[k].items()}) if oph is not None else 0.0
        bd = body["ser"].get(k, 0.0) if body is not None else 0.0
        S[k] = bd + oc
    return S


def binned_slope(x, C, p_hod, nb=10):
    """Shelly-style metric: ratio of actual to expected counts in shear-stress deciles, slope of a weighted
    linear fit of the ratio vs bin-mean stress (per kPa).  Expected = day total x time-of-day pattern."""
    Nr = C.sum(1)
    pex = Nr[:, None] * p_hod[None, :]
    edges = np.quantile(x, np.linspace(0, 1, nb + 1))
    q = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    obs = np.bincount(q.ravel(), C.ravel(), nb)
    exp_ = np.bincount(q.ravel(), pex.ravel(), nb)
    xm = np.bincount(q.ravel(), (pex * x).ravel(), nb) / exp_
    ratio = obs / exp_
    w = exp_ / ratio.clip(1e-9)          # ~1/var of the ratio (Poisson)
    A = np.c_[np.ones(nb), xm]
    coef = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * ratio))
    rmse = float(np.sqrt(np.mean((ratio - A @ coef) ** 2)))
    return float(coef[1]), rmse, ratio, xm


def stage_dipscan():
    t00 = time.time()
    ph = pd.read_pickle(f"{OUT}/stress_phasors_final.pkl")
    mods = pd.read_csv(f"{OUT}/receiver_local_moduli.csv").set_index("receiver")
    cx = setup_time()
    NET = "famcentroid"
    body_loc = body_stress(cx, mods.loc[NET, "mu_local"], mods.loc[NET, "nu_local"])
    body_30 = body_stress(cx, 30e9, 0.25)
    rows = cx.rows
    C = cx.Cnet[rows]
    f0 = T.cfit(C, [])
    p_hod, LL0 = f0["p"][0], f0["LL"]
    cfgs = {"body_only_mu30": (None, body_30), "body_only_local": (None, body_loc),
            "total_primary": (ocean_phasors(ph, NET, "usawest+tpxo72", "layered_RL2"), body_loc),
            "total_layered_body30": (ocean_phasors(ph, NET, "usawest+tpxo72", "layered_RL2"), body_30),
            "total_homog_RB5_body30": (ocean_phasors(ph, NET, "usawest+tpxo72", "homog_RB5"), body_30),
            "total_fes2004": (ocean_phasors(ph, NET, "fes2004", "layered_RL2"), body_loc)}
    out = []
    for name, (oph, bdy) in cfgs.items():
        S = tensor_series(cx, oph, bdy)
        for dip in (35.0, 40.0, 45.0, 50.0, 55.0):
            for rake in np.arange(0, 360, 10.0):
                R = T.resolve(S, T.STRIKE, dip, rake)
                x = M_(cx, R["tau"])[rows]
                slope, rmse, _, _ = binned_slope(x, C, p_hod)
                f = T.cfit(C, [x], iters=10, theta0=np.r_[f0["theta"], 0.0])
                out.append(dict(config=name, dip=dip, rake=rake, slope_per_kPa=slope, rmse=rmse, beta=float(f["beta"][0]),
                                dLL=f["LL"] - LL0, tau_M2_amp=float(np.std(R["tau"]) * np.sqrt(2))))
        d = pd.DataFrame([o for o in out if o["config"] == name])
        i, j = d.slope_per_kPa.idxmax(), d.dLL.idxmax()
        print(name, "max slope", d.loc[i, ["dip", "rake", "slope_per_kPa"]].to_dict(),
              "max dLL", d.loc[j, ["dip", "rake", "dLL", "beta"]].to_dict(), f"{time.time() - t00:.0f}s", flush=True)
    D = pd.DataFrame(out)
    D.to_csv(f"{OUT}/dip_rake_scan.csv", index=False)
    summ = {}
    for name, d in D.groupby("config"):
        i = d.slope_per_kPa.idxmax()
        j = d.dLL.idxmax()
        ll = d[(d.rake >= 330) | (d.rake <= 30)]     # left-lateral (rake ~0)
        rl = d[(d.rake >= 150) & (d.rake <= 210)]
        summ[name] = dict(max_slope=d.loc[i, ["dip", "rake", "slope_per_kPa", "dLL"]].to_dict(),
                          max_dLL=d.loc[j, ["dip", "rake", "dLL", "slope_per_kPa"]].to_dict(),
                          d40_r180=d[(d.dip == 40) & (d.rake == 180)][["slope_per_kPa", "dLL", "beta"]].iloc[0].to_dict(),
                          best_LL_slope=ll.loc[ll.slope_per_kPa.idxmax(), ["dip", "rake", "slope_per_kPa", "dLL"]].to_dict(),
                          best_RL_slope=rl.loc[rl.slope_per_kPa.idxmax(), ["dip", "rake", "slope_per_kPa", "dLL"]].to_dict(),
                          best_LL_dLL=ll.loc[ll.dLL.idxmax(), ["dip", "rake", "slope_per_kPa", "dLL"]].to_dict(),
                          best_RL_dLL=rl.loc[rl.dLL.idxmax(), ["dip", "rake", "slope_per_kPa", "dLL"]].to_dict())
    json.dump(summ, open(f"{OUT}/dip_rake_scan_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    names = list(cfgs)
    fig, ax = plt.subplots(2, len(names), figsize=(3.0 * len(names), 6), squeeze=False)
    cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for k, name in enumerate(names):
        d = D[D.config == name]
        for dip, col in zip((35.0, 40.0, 45.0, 50.0, 55.0), cols):
            g = d[d.dip == dip]
            ax[0, k].plot(g.rake, g.slope_per_kPa, color=col, label=f"dip {dip:.0f}")
            ax[1, k].plot(g.rake, g.dLL, color=col)
        ax[0, k].set_title(name, fontsize=7)
        for a in ax[:, k]:
            a.axvline(180, color="r", lw=0.5); a.axvline(0, color="b", lw=0.5); a.grid(alpha=0.3); a.set_xlabel("rake")
    ax[0, 0].set_ylabel("slope of actual/expected vs tau (per kPa)")
    ax[1, 0].set_ylabel("dlogL (exp model)")
    ax[0, 0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_tidal_spotl_dip_rake.png", dpi=150); plt.close(fig)


# ===================================================================== post-processing (sign-constrained summaries)
def stage_post():
    out = {}
    # dip x rake: an orientation is only 'encouraging' if beta > 0 (rate up when tau in the slip direction is up)
    D = pd.read_csv(f"{OUT}/dip_rake_scan.csv")
    D["dLL_pos"] = np.where(D.beta > 0, D.dLL, 0.0)
    for name, d in D.groupby("config"):
        i, j = d.slope_per_kPa.idxmax(), d.dLL_pos.idxmax()
        rl = d[(d.rake >= 150) & (d.rake <= 210)]
        ll = d[(d.rake >= 330) | (d.rake <= 30)]
        nf = d[(d.rake >= 240) & (d.rake <= 300)]
        rv = d[(d.rake >= 60) & (d.rake <= 120)]
        out[f"diprake_{name}"] = dict(
            max_slope=d.loc[i, ["dip", "rake", "slope_per_kPa", "dLL_pos"]].to_dict(),
            max_dLL=d.loc[j, ["dip", "rake", "dLL_pos", "slope_per_kPa", "beta"]].to_dict(),
            d40_r180=d[(d.dip == 40) & (d.rake == 180)][["slope_per_kPa", "dLL_pos", "beta"]].iloc[0].to_dict(),
            RL_best_slope=rl.loc[rl.slope_per_kPa.idxmax(), ["dip", "rake", "slope_per_kPa"]].to_dict(),
            LL_best_slope=ll.loc[ll.slope_per_kPa.idxmax(), ["dip", "rake", "slope_per_kPa"]].to_dict(),
            normal_best_slope=nf.loc[nf.slope_per_kPa.idxmax(), ["dip", "rake", "slope_per_kPa"]].to_dict(),
            reverse_best_slope=rv.loc[rv.slope_per_kPa.idxmax(), ["dip", "rake", "slope_per_kPa"]].to_dict(),
            RL_best_dLL=rl.loc[rl.dLL_pos.idxmax(), ["dip", "rake", "dLL_pos"]].to_dict(),
            LL_best_dLL=ll.loc[ll.dLL_pos.idxmax(), ["dip", "rake", "dLL_pos"]].to_dict(),
            normal_best_dLL=nf.loc[nf.dLL_pos.idxmax(), ["dip", "rake", "dLL_pos"]].to_dict(),
            reverse_best_dLL=rv.loc[rv.dLL_pos.idxmax(), ["dip", "rake", "dLL_pos"]].to_dict())
    # poro surface with the physical sign (beta > 0)
    if os.path.exists(f"{OUT}/poro_surface_network.csv"):
        S = pd.read_csv(f"{OUT}/poro_surface_network.csv")
        S["dLL_pos"] = np.where(S.beta > 0, S.dLL, 0.0)
        S.to_csv(f"{OUT}/poro_surface_network.csv", index=False)
        for name, d in S.groupby("config"):
            i = d.dLL_pos.idxmax()
            r = d[(d.B == 1.0) & np.isclose(d.mu, 0.6)].iloc[0]
            # best mu at B = 1 (physical sign)
            b1 = d[d.B == 1.0]
            k = b1.dLL_pos.idxmax()
            out[f"poro_{name}"] = dict(best=d.loc[i, ["B", "mu", "beta", "dLL_pos"]].to_dict(),
                                       B1_mu06=dict(beta=r.beta, dLL=r.dLL, dLL_pos=r.dLL_pos),
                                       best_at_B1=d.loc[k, ["mu", "beta", "dLL_pos"]].to_dict(),
                                       dLL_best_minus_B1mu06=float(d.dLL_pos[i] - r.dLL_pos),
                                       max_mu_with_beta_pos_at_B1=float(b1[b1.beta > 0].mu.max()) if (b1.beta > 0).any() else None)
    json.dump(out, open(f"{OUT}/post_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


# ===================================================================== fit part 2 (parallel bootstrap + per family)
_G = {}


def _init_fit2():
    os.environ["OMP_NUM_THREADS"] = "1"
    ph = pd.read_pickle(f"{OUT}/stress_phasors_final.pkl")
    mods = pd.read_csv(f"{OUT}/receiver_local_moduli.csv").set_index("receiver")
    cx = setup_time()
    _G.update(ph=ph, mods=mods, cx=cx)


def _boot_job(seed):
    cx, ph, mods = _G["cx"], _G["ph"], _G["mods"]
    NET = "famcentroid"
    bl = body_stress(cx, mods.loc[NET, "mu_local"], mods.loc[NET, "nu_local"])
    oph = ocean_phasors(ph, NET, "usawest+tpxo72", "layered_RL2")
    rows = cx.rows
    C = cx.Cnet[rows]
    w = np.random.default_rng(seed).poisson(1.0, len(rows)).astype(float)
    out = {}
    for tag, sc in (("ocean1", 1.0), ("ocean0", 0.0)):
        X = stress_set(cx, oph, body=bl, ocean_scale=sc)
        tau, sn = M_(cx, X["tau"])[rows], M_(cx, X["sn"])[rows]
        f0 = T.cfit(C, [], w=w)
        ft = T.cfit(C, [tau], w=w)
        f2 = T.cfit(C, [tau, sn], w=w)
        llr = max([T.cfit(C, [tau + mu * sn], w=w)["LL"] if True else 0 for mu in (0.0, 0.1, 0.2)])
        lll = []
        for mu in (0.0, 0.1, 0.2, 0.4):
            fl = T.cfit(C, [-tau + mu * sn], w=w)
            lll.append(fl["LL"] if fl["beta"][0] > 0 else f0["LL"])
        out[tag] = dict(beta_tau=float(ft["beta"][0]), mu_eff=float(f2["beta"][1] / f2["beta"][0]),
                        LL_tau_sn=f2["LL"] - f0["LL"], dLL_RL_minus_LL=llr - max(lll))
    out["dLL_ocean0_minus_ocean1"] = out["ocean0"]["LL_tau_sn"] - out["ocean1"]["LL_tau_sn"]
    return out


def _fam_job(f_):
    from scipy.signal import hilbert
    cx, ph, mods = _G["cx"], _G["ph"], _G["mods"]
    rr = np.where(cx.Cfam[f_].sum(1) > 0)[0]
    if len(rr) < 50:
        return None
    Cf = cx.Cfam[f_][rr].astype(float)
    rec = f"fam{f_ + 1}"
    bf = body_stress(cx, mods.loc[rec, "mu_local"], mods.loc[rec, "nu_local"])
    row = dict(family=f_ + 1, n_units=int(Cf.sum()), mu_local_GPa=mods.loc[rec, "mu_local"] / 1e9)
    oph = ocean_phasors(ph, rec, "usawest+tpxo72", "layered_RL2")
    for tag, o in (("total", oph), ("body", None)):
        Xf = stress_set(cx, o, body=bf)
        r = fit_block(cx, Xf, C=Cf, rows=rr)
        r.pop("prof")
        for k in ("beta_tau", "se_tau", "dLL_tau", "a_sigma_tau", "mu_eff", "mu_eff_se", "dLL_tau_sn", "dLL_mean",
                  "RL_phys_mu_best", "RL_phys_dLL", "LL_phys_dLL", "dLL_RL_minus_LL_phys", "RL_a_sigma_best"):
            row[f"{tag}_{k}"] = r.get(k, np.nan)
        hx = np.imag(hilbert(Xf["tau"] - Xf["tau"].mean()))
        ff = T.cfit(Cf, [M_(cx, Xf["tau"])[rr], M_(cx, hx)[rr]])
        Vw, _ = T.robust_se(ff, rr // 7)
        bc, bs = ff["beta"]
        gph = np.array([-bs, bc]) / (bc ** 2 + bs ** 2)
        row[f"{tag}_lag_deg"] = float(np.degrees(np.arctan2(bs, bc)))
        row[f"{tag}_lag_se"] = float(np.degrees(np.sqrt(gph @ Vw @ gph)))
        # poroelastic check per family: (B=1, mu=0.6) vs best of a coarse grid
        if tag == "total":
            tau, sn, mean = (M_(cx, Xf[k])[rr] for k in ("tau", "sn", "mean"))
            LL0 = T.cfit(Cf, [])["LL"]
            best = (-np.inf, None)
            for B in (0.0, 0.5, 1.0):
                for mu in (0.0, 0.1, 0.2, 0.4, 0.6):
                    f = T.cfit(Cf, [tau + mu * (sn - B * mean)])
                    d = f["LL"] - LL0 if f["beta"][0] > 0 else 0.0
                    if (B, mu) == (1.0, 0.6):
                        row["poro_B1_mu06_dLL"] = d
                        row["poro_B1_mu06_beta"] = float(f["beta"][0])
                    if d > best[0]:
                        best = (d, (B, mu))
            row["poro_best_dLL"], row["poro_best_B"], row["poro_best_mu"] = best[0], best[1][0], best[1][1]
            Rm = T.resolve({k: oph[k]["M2"] / 1e3 for k in L.TENSOR_KEYS}, T.STRIKE, 40.0)
            row.update(tau_M2_amp=abs(Xf["ph"]["M2"]["tau"]), sn_M2_amp=abs(Xf["ph"]["M2"]["sn"]),
                       ocean_tau_M2_amp=abs(Rm["tau"]), ocean_sn_M2_amp=abs(Rm["sn"]), ocean_mean_M2_amp=abs(Rm["mean"]))
    return row


def stage_fit2():
    from scipy.stats import spearmanr
    t00 = time.time()
    summ = json.load(open(f"{OUT}/fit_summary.json")) if os.path.exists(f"{OUT}/fit_summary.json") else {}
    with ProcessPoolExecutor(NPROC, initializer=_init_fit2) as ex:
        fams = [r for r in ex.map(_fam_job, range(T.NF)) if r is not None]
        print("families done", f"{time.time() - t00:.0f}s", flush=True)
        boots = list(ex.map(_boot_job, range(1000, 1048)))
    print("bootstrap done", f"{time.time() - t00:.0f}s", flush=True)
    bt = np.array([b["ocean1"]["beta_tau"] for b in boots])
    bm = np.array([b["ocean1"]["mu_eff"] for b in boots])
    rl = np.array([b["ocean1"]["dLL_RL_minus_LL"] for b in boots])
    d01 = np.array([b["dLL_ocean0_minus_ocean1"] for b in boots])
    bm0 = np.array([b["ocean0"]["mu_eff"] for b in boots])
    summ["bootstrap"] = dict(B=len(boots), beta_tau_mean=float(bt.mean()), beta_tau_sd=float(bt.std()),
                             a_sigma_tau_95=[float(1 / np.percentile(bt, 97.5)), float(1 / np.percentile(bt, 2.5))],
                             mu_eff_median=float(np.median(bm)), mu_eff_95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                             mu_eff_body_only_95=[float(np.percentile(bm0, 2.5)), float(np.percentile(bm0, 97.5))],
                             dLL_RL_minus_LL_95=[float(np.percentile(rl, 2.5)), float(np.percentile(rl, 97.5))],
                             frac_LL_preferred=float(np.mean(rl < 0)),
                             dLL_bodyonly_minus_withocean_95=[float(np.percentile(d01, 2.5)), float(np.percentile(d01, 97.5))],
                             frac_ocean_preferred=float(np.mean(d01 < 0)))
    print(json.dumps(summ["bootstrap"], indent=1), flush=True)
    FR = pd.DataFrame(fams)
    recs = receivers().set_index("name")
    for k in ("lat", "lon", "depth_km"):
        FR[k] = [recs.loc[f"fam{f}", k] for f in FR.family]
    FR.to_csv(f"{OUT}/family_fits.csv", index=False)
    good = FR[FR.n_units >= 300]
    summ["family"] = dict(
        n=int(len(good)), total_beta_tau_pos=int((good.total_beta_tau > 0).sum()), body_beta_tau_pos=int((good.body_beta_tau > 0).sum()),
        total_RL_preferred=int((good.total_dLL_RL_minus_LL_phys > 0).sum()),
        total_a_sigma_median=float(np.nanmedian(good.total_a_sigma_tau)),
        total_a_sigma_IQR=[float(np.nanpercentile(good.total_a_sigma_tau, 25)), float(np.nanpercentile(good.total_a_sigma_tau, 75))],
        body_a_sigma_median=float(np.nanmedian(good.body_a_sigma_tau)),
        total_mu_eff_median=float(np.median(good.total_mu_eff)), body_mu_eff_median=float(np.median(good.body_mu_eff)),
        total_mu_eff_IQR=[float(np.percentile(good.total_mu_eff, 25)), float(np.percentile(good.total_mu_eff, 75))],
        total_lag_vs_lon_spearman=[float(v) for v in spearmanr(good.lon, good.total_lag_deg)],
        body_lag_vs_lon_spearman=[float(v) for v in spearmanr(good.lon, good.body_lag_deg)],
        total_mu_vs_lon_spearman=[float(v) for v in spearmanr(good.lon, good.total_mu_eff)],
        body_mu_vs_lon_spearman=[float(v) for v in spearmanr(good.lon, good.body_mu_eff)],
        n_body_better_than_total=int((good.body_dLL_tau_sn > good.total_dLL_tau_sn).sum()),
        poro_B1_mu06_rejected=int((good.poro_best_dLL - good.poro_B1_mu06_dLL > 8 * 3.84 / 2).sum()),
        poro_best_counts=good.groupby(["poro_best_B", "poro_best_mu"]).size().to_dict(),
        ocean_tau_M2_range=[float(good.ocean_tau_M2_amp.min()), float(good.ocean_tau_M2_amp.max())],
        ocean_sn_M2_range=[float(good.ocean_sn_M2_amp.min()), float(good.ocean_sn_M2_amp.max())],
        ocean_mean_M2_range=[float(good.ocean_mean_M2_amp.min()), float(good.ocean_mean_M2_amp.max())])
    summ["family"]["poro_best_counts"] = {str(k): v for k, v in summ["family"]["poro_best_counts"].items()}
    print(json.dumps(summ["family"], indent=1), flush=True)
    json.dump(summ, open(f"{OUT}/fit_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "tests"
    {"tests": stage_tests, "stress": stage_stress, "strainobs": stage_strainobs, "validate": stage_validate, "fit": stage_fit, "converge": stage_converge, "stress2": stage_stress2, "farcorr": stage_farcorr, "fe": stage_fe, "stress4": stage_stress4, "poro": stage_poro, "figs": stage_figs, "dipscan": stage_dipscan, "post": stage_post, "fit2": stage_fit2}[st]()
