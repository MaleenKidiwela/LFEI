"""sci_tidal_spotl_lib.py - helpers for scripts/sci_tidal_ocean_spotl.py (SPOTL ocean-tide loading at LFE depth).

SPOTL 3.3.0.2 (Agnew 2012) is installed in /home/seismic/LFE/tools/spotl (built with gfortran 8.5; see
tools/spotl/LFE_INSTALL.txt).  This module wraps:
  * nloadf (surface load: gravity, displacement, tilt, strain) with optional polygon files (regional+global
    model merging, and a circle around the receiver to split near and far field);
  * ocpts (our helper, scripts/sci_tidal_spotl_ocpts.f linked to SPOTL's ocmodl): SPOTL's own ocean-load
    value (tidal height x sea-water density, kg/m^2) at arbitrary points, i.e. the same 1/64-deg land-sea
    database and coastal bilinear interpolation that nloadf uses in its near field;
  * a Python re-implementation of nloadf's convolution (validated against nloadf) used to test the
    Green-function tables;
  * Boussinesq point-load stress (closed form, homogeneous half-space) at depth.

Phase convention (as SPOTL nloadf 'g' output): a quantity x(t) = Re{ Z f exp(i (V(t) + u)) } with V the
Greenwich equilibrium argument (sci_tidal_stress.doodson_V) and Z = A exp(i phi), phi = Greenwich phase with
LAGS NEGATIVE.  Ocean-model files store H = A exp(+i G) (G = Greenwich lag), so Z_height = conj(H).
"""
import os
import subprocess
import numpy as np

SP = "/home/seismic/LFE/tools/spotl"
WD = f"{SP}/working"
BIN = f"{SP}/bin"
A_E = 6.371e6
G0 = 9.81
CONS8 = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
NICE = ["nice", "-n", "10"]

# ocean-model combinations: list of (model-suffix, polygon spec)  polygon spec: None | '+usawest' | '-usawest'
MODELS = {
    "usawest+tpxo72": [("osu.usawest.2010", "+usawest"), ("osu.tpxo72.2010", "-usawest")],
    "tpxo72atlas": [("osu.tpxo72atlas.2011", None)],
    "tpxo72": [("osu.tpxo72.2010", None)],
    "fes2004": [("fes.2004", None)],
    "got4p7": [("got4p7.2004", None)],
    "eot11a": [("eot11a.2011", None)],
    "hamtide11a": [("hamtide11a.2011", None)],
    "dtu10": [("dtu10.tr.2010", None)],
    "nao99b": [("naoglobal.1999", None)],
}
USAWEST_POLY = np.array([
    (-131.017, 50.517), (-131.017, 30.483), (-116.016, 30.483), (-115.983, 30.683), (-115.983, 30.716),
    (-116.016, 30.816), (-122.183, 48.084), (-122.417, 48.650), (-122.483, 48.784), (-123.217, 49.617),
    (-124.817, 50.150), (-128.083, 50.517), (-131.017, 50.517)])


# ------------------------------------------------------------------ geometry
def dest_point(lat, lon, dist_deg, az_deg):
    """Great-circle destination (deg)."""
    p1, l1 = np.deg2rad(lat), np.deg2rad(lon)
    d, a = np.deg2rad(dist_deg), np.deg2rad(az_deg)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(a))
    l2 = l1 + np.arctan2(np.sin(a) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.rad2deg(p2), np.rad2deg(l2)


def gc_dist_az(lat0, lon0, lat, lon):
    """Great-circle distance (deg) and azimuth (deg E of N) from (lat0,lon0) to points."""
    p1, p2 = np.deg2rad(lat0), np.deg2rad(lat)
    dl = np.deg2rad(lon - lon0)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    d = np.arccos(np.clip(c, -1, 1))
    az = np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))
    return np.rad2deg(d), np.rad2deg(az)


def circle_lonlat(lat0, lon0, rdeg, n=400):
    az = np.linspace(0, 360, n, endpoint=False)
    la, lo = dest_point(lat0, lon0, rdeg, az)
    return np.r_[lo, lo[0]], np.r_[la, la[0]]


def write_poly(path, polys, title="LFE polygon file"):
    """polys: list of (name, sign '+'/'-', lon array, lat array)."""
    with open(path, "w") as f:
        f.write(title + "\n")
        f.write(f"{len(polys):3d}\n")
        for name, sgn, lo, la in polys:
            assert len(lo) <= 500
            f.write(f"{name}\n{len(lo):3d}\n{sgn}\n")
            for x, y in zip(lo, la):
                f.write(f"{x:10.4f}{y:10.4f}\n")


def poly_list(spec, circle=None):
    """spec None|'+usawest'|'-usawest'; circle None or (sign, lat0, lon0, rdeg)."""
    out = []
    if spec is not None:
        out.append(("osu.usawest.2010", spec[0], USAWEST_POLY[:, 0], USAWEST_POLY[:, 1]))
    if circle is not None:
        sgn, la0, lo0, r = circle
        lo, la = circle_lonlat(la0, lo0, r)
        out.append((f"circle{r:.3f}", sgn, lo, la))
    return out


# ------------------------------------------------------------------ SPOTL wrappers
_CNT = [0]


def _tmpname():
    """Short relative polygon-file name (SPOTL file names are character*80)."""
    _CNT[0] += 1
    return f"p{os.getpid()}_{_CNT[0]}.ply"


def _run(cmd, inp=None):
    r = subprocess.run(NICE + cmd, cwd=WD, input=inp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}: {r.stderr[:500]}")
    return r.stdout


def parse_load(txt):
    """Parse an nloadf/loadcomb load file -> dict of complex arrays (Z = amp exp(i phase))."""
    out = {}
    for line in txt.splitlines():
        if not line or line[0] not in "gpdts":
            continue
        v = np.array(line[10:].split(), float)
        z = v[0::2] * np.exp(1j * np.deg2rad(v[1::2]))
        out[line[0]] = z
    return out


def nloadf(lat, lon, ht, cons, model, green="gr.gbaver.wef.p02.ce", polys=None, tag="x", tmpdir=None):
    """Run nloadf (Greenwich phase, lags negative).  polys: list for write_poly or None."""
    args = ["../bin/nloadf", "STA", f"{lat:.5f}", f"{lon:.5f}", f"{ht:.1f}", f"{cons.lower()}.{model}", green, "g"]
    if polys:
        pf = _tmpname()
        write_poly(os.path.join(WD, pf), polys)
        args.append(pf)
    txt = _run(args)
    if polys:
        os.remove(os.path.join(WD, pf))
    if "s" not in parse_load(txt):
        raise RuntimeError("nloadf failed: " + txt[:300])
    return parse_load(txt), txt


def ocpts(model_file, lat, lon, flag="F", polys=None, tag="o"):
    """SPOTL ocean load (kg/m^2, complex H*rho with H = A exp(+iG)) at points.  Returns complex array."""
    args = ["../bin/ocpts", model_file, flag]
    pf = None
    if polys:
        pf = _tmpname()
        write_poly(os.path.join(WD, pf), polys)
        args.append(pf)
    inp = "\n".join(f"{a:.6f} {b:.6f}" for a, b in zip(np.ravel(lat), np.ravel(lon))) + "\n"
    txt = _run(args, inp)
    if pf:
        os.remove(os.path.join(WD, pf))
    v = np.array(txt.split(), float).reshape(-1, 2)
    return (v[:, 0] + 1j * v[:, 1]).reshape(np.shape(lat))


# ------------------------------------------------------------------ Green-function tables
def read_green(fn):
    """Return list of ranges: dict(beg, end, spc, flag, G (n,7)) from a SPOTL integrated-GF file."""
    lines = open(os.path.join(WD, fn)).read().splitlines()
    i = 1
    out = []
    while i < len(lines):
        h = lines[i]
        ngr = int(h[8:12])
        beg, end, spc = float(h[12:22]), float(h[22:32]), float(h[32:42])
        flag = h[47:48]
        vals = []
        i += 1
        while len(vals) < 7 * ngr:
            vals += [float(lines[i][k:k + 13]) for k in range(0, len(lines[i].rstrip()), 13)]
            i += 1
        out.append(dict(beg=beg, end=end, spc=spc, flag=flag, G=np.array(vals).reshape(ngr, 7)))
    return out


def ring_cells(lat0, lon0, green, dmax=None, dmin=0.0, minaz=150):
    """nloadf's station-centred cells: arrays dist (deg), az (deg), azstp (rad), ring GF row, flag."""
    rows = []
    for rg in green:
        for ii in range(rg["G"].shape[0]):
            dist = rg["beg"] + ii * rg["spc"]
            if (dmax is not None and dist > dmax) or dist < dmin:
                continue
            naz = int(360.0 * np.sin(np.deg2rad(dist)))
            if dist < 180.0:
                naz = max(naz, minaz)
            rows.append((dist, naz, rg["spc"], rg["flag"], rg["G"][ii]))
    D, AZ, AS, GR, FL, SP_ = [], [], [], [], [], []
    for dist, naz, spc, flag, g in rows:
        azs = 360.0 / naz
        az = azs / 2 + azs * np.arange(naz)
        D.append(np.full(naz, dist)); AZ.append(az); AS.append(np.full(naz, np.deg2rad(azs)))
        GR.append(np.repeat(g[None, :], naz, 0)); FL.append(np.full(naz, flag)); SP_.append(np.full(naz, spc))
    return dict(dist=np.concatenate(D), az=np.concatenate(AZ), azstp=np.concatenate(AS),
                G=np.concatenate(GR), flag=np.concatenate(FL), spc=np.concatenate(SP_))


def py_surface_strain(cells, amp):
    """Replicates nloadf strain convolution (eEE, eNN, eEN) in nloadf's internal complex convention
    (returns Z-convention values, nanostrain): Z = conj(sum)."""
    az = np.deg2rad(cells["az"])
    caz, saz = np.cos(az), np.sin(az)
    h = cells["azstp"]
    ett, ell = cells["G"][:, 4], cells["G"][:, 5]
    s2 = 2 * np.sin(h / 2)
    aa = 0.5 * (caz ** 2 - saz ** 2)
    bb = 0.5 * h + aa * s2
    cc = 0.5 * h - aa * s2
    eNN = np.sum((ett * bb + ell * cc) * amp)
    eEE = np.sum((ell * bb + ett * cc) * amp)
    eEN = np.sum((ett - ell) * saz * caz * np.sin(h) * np.cos(h) * amp)
    return dict(eEE=np.conj(eEE) * 1e9, eNN=np.conj(eNN) * 1e9, eEN=np.conj(eEN) * 1e9)


# ------------------------------------------------------------------ Boussinesq
def bouss_point_stress(dE, dN, z, P, nu=0.25):
    """Stress (Pa, tension +, keys in N,E,D) at a receiver at depth z (m) offset (dE, dN) (m, receiver minus
    load point) from a downward point force P (N) on the surface of a homogeneous half-space (closed form,
    e.g. Johnson 1985 eq. 3.?; independent of the shear modulus)."""
    r2 = dE ** 2 + dN ** 2
    r = np.sqrt(r2)
    R = np.sqrt(r2 + z ** 2)
    k = P / (2 * np.pi)
    # (1 - z/R)/r^2 = 1/(R (R + z))  (stable form)
    q = 1.0 / (R * (R + z))
    srr = k * ((1 - 2 * nu) * q - 3 * z * r2 / R ** 5)
    stt = -k * (1 - 2 * nu) * (q - z / R ** 3)
    szz = -3 * k * z ** 3 / R ** 5
    srz = -3 * k * r * z ** 2 / R ** 5
    with np.errstate(invalid="ignore", divide="ignore"):
        cE = np.where(r > 0, dE / r, 0.0)
        cN = np.where(r > 0, dN / r, 0.0)
    return dict(snn=srr * cN ** 2 + stt * cE ** 2, see=srr * cE ** 2 + stt * cN ** 2,
                sne=(srr - stt) * cE * cN, sdd=szz, snd=srz * cN, sed=srz * cE)


def bouss_surface_strain(dE, dN, P, mu, nu=0.25):
    """Surface strain (eEE, eNN, eEN) at offset (dE, dN) from a point force P (half-space)."""
    r2 = dE ** 2 + dN ** 2
    e = (1 - 2 * nu) * P / (4 * np.pi * mu * r2)   # e_rr = +e, e_tt = -e
    cE, cN = dE / np.sqrt(r2), dN / np.sqrt(r2)
    return dict(eEE=e * (cE ** 2 - cN ** 2), eNN=e * (cN ** 2 - cE ** 2), eEN=2 * e * cE * cN)


def plane_stress_from_strain(eEE, eNN, eEN, mu, nu=0.25):
    lamp = 2 * mu * nu / (1 - nu)
    tr = eEE + eNN
    return dict(snn=2 * mu * eNN + lamp * tr, see=2 * mu * eEE + lamp * tr, sne=2 * mu * eEN,
                sdd=0 * tr, snd=0 * tr, sed=0 * tr)


TENSOR_KEYS = ["snn", "see", "sdd", "sne", "snd", "sed"]


# ------------------------------------------------------------------ depth correction from distant loads
def vertical_gf_spline(green):
    """Point-load vertical-displacement Green's function u(D) (m per kg, up +) from the integrated table
    (column 1: G_i = a^2 int u sinD dD over the ring) and its first two derivatives w.r.t. D (radians),
    via a cubic spline of D*u(D) in log D."""
    from scipy.interpolate import CubicSpline
    D, U = [], []
    for rg in green:
        for ii in range(rg["G"].shape[0]):
            d = rg["beg"] + ii * rg["spc"]
            dr, wr = np.deg2rad(d), np.deg2rad(rg["spc"])
            D.append(dr)
            U.append(rg["G"][ii, 0] / (A_E ** 2 * np.sin(dr) * wr))
    D, U = np.array(D), np.array(U)
    o = np.argsort(D)
    D, U = D[o], U[o]
    keep = np.r_[True, np.diff(D) > 0]
    D, U = D[keep], U[keep]
    x = np.log(D)
    cs = CubicSpline(x, D * U)

    def derivs(Dq):
        xq = np.log(Dq)
        f, f1, f2 = cs(xq), cs(xq, 1), cs(xq, 2)       # f = D u(D), derivatives w.r.t. x = ln D
        # u = f/D ; du/dD = (f1 - f)/D^2 ; d2u/dD2 = (f2 - 3 f1 + 2 f)/D^3
        return f / Dq, (f1 - f) / Dq ** 2, (f2 - 3 * f1 + 2 * f) / Dq ** 3
    return derivs


def depth_strain_correction(cells, amp, derivs, z=1.0):
    """First-order depth correction of horizontal strain at depth z below a free surface:
        e_ab(z) - e_ab(0) = + z d_a d_b u_up   (from sigma_az = 0 at the surface)
    with the Hessian of the spherical-Earth vertical-displacement field of the loads in `cells`
    (point GF, covariant Hessian on the sphere: (1/a^2)[u'' n n + u' cotD t t]).
    Returns Z-convention complex strains (eEE, eNN, eEN) per metre of depth if z=1."""
    Dr = np.deg2rad(cells["dist"])
    az = np.deg2rad(cells["az"])
    nE, nN = np.sin(az), np.cos(az)
    _, u1, u2 = derivs(Dr)
    w = cells["G"][:, 0] * 0 + A_E ** 2 * np.sin(Dr) * np.deg2rad(cells["spc"]) * cells["azstp"]   # cell area
    a = u2 / A_E ** 2
    b = u1 / np.tan(Dr) / A_E ** 2
    hEE = a * nE * nE + b * (1 - nE * nE)
    hNN = a * nN * nN + b * (1 - nN * nN)
    hEN = (a - b) * nE * nN
    q = amp * w
    return dict(eEE=np.conj(z * np.sum(hEE * q)), eNN=np.conj(z * np.sum(hNN * q)), eEN=np.conj(z * np.sum(hEN * q)))
