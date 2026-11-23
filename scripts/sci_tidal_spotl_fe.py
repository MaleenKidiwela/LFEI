"""sci_tidal_spotl_fe.py - axisymmetric finite-element Green's function for a vertical point load on the surface of a
LAYERED elastic half-space (helper for scripts/sci_tidal_ocean_spotl.py).

Linear elasticity, axisymmetric (u_r, u_z; z positive down), 4-node bilinear quads with 2x2 Gauss integration,
graded mesh (0.25 km near the load, 0.5 km to 60 km depth/150 km range, geometric growth to the far
boundary, fixed far boundaries).  The point load (1 N, downward) is applied as a uniform pressure on a disk of
radius 1 km (irrelevant at the >= 25 km source-receiver distances used).  Stresses (tension +) are evaluated at
element centres and tabulated vs r for the requested depths.
Elastic models:  'furlong1d'  local P model of Furlong et al. (2024) (data/velocity/furlong2024/vel_p.1d),
                 Vs = Vp/1.78 (as the starting model), rho from Brocher (2005); below 71 km a PREM-like mantle.
                 'uniform'    mu = 30 GPa, nu = 0.25 (validation against Boussinesq).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = "/home/seismic/LFE"


def brocher_rho(vp):
    vp = np.asarray(vp)
    return 1.6612 * vp - 0.4721 * vp ** 2 + 0.0671 * vp ** 3 - 0.0043 * vp ** 4 + 0.000106 * vp ** 5   # g/cc


def furlong_profile():
    d = np.loadtxt(f"{ROOT}/data/velocity/furlong2024/vel_p.1d")
    z, vp = d[:, 0], d[:, 1]
    return z, vp


def elastic_model(name):
    """Return function z_km -> (mu Pa, lambda Pa)."""
    if name == "uniform":
        mu = 30e9
        lam = 2 * mu * 0.25 / (1 - 2 * 0.25)
        return lambda z: (np.full_like(z, mu, dtype=float), np.full_like(z, lam, dtype=float))
    zf, vpf = furlong_profile()

    def f(z):
        z = np.asarray(z, float)
        vp = np.interp(z, zf[1::2], vpf[1::2])
        vs = vp / 1.78
        rho = brocher_rho(vp) * 1e3
        deep = z > 71.0
        # PREM-like mantle below the model (Vp 8.1 -> 9.0 km/s at 400 km, Vp/Vs 1.80, rho 3.38 -> 3.55)
        t = np.clip((z - 71.0) / 330.0, 0, 1)
        vp = np.where(deep, 8.10 + 0.9 * t, vp)
        vs = np.where(deep, vp / 1.80, vs)
        rho = np.where(deep, 3380 + 170 * t, rho)
        mu = rho * (vs * 1e3) ** 2
        lam = rho * (vp * 1e3) ** 2 - 2 * mu
        return mu, lam
    return f


def graded(h0, x_fine, x_max, growth=1.06, start=0.0, h_start=None):
    """Node coordinates: spacing h_start up to 2 km (if given), h0 to x_fine, then geometric growth to x_max."""
    x = [start]
    while x[-1] < x_max:
        cur = x[-1]
        if h_start is not None and cur < 2.0:
            h = h_start
        elif cur < x_fine:
            h = h0
        else:
            h = (x[-1] - x[-2]) * growth
        x.append(cur + h)
    return np.array(x)


def solve(model="furlong1d", rmax=4000.0, zmax=4000.0, a0=1.0, r_fine=150.0, z_fine=60.0):
    r = graded(0.5, r_fine, rmax, 1.06, h_start=0.25)
    z = graded(0.5, z_fine, zmax, 1.06, h_start=0.25)
    nr, nz = len(r), len(z)
    R, Z = np.meshgrid(r * 1e3, z * 1e3, indexing="ij")      # (nr, nz) metres
    nid = np.arange(nr * nz).reshape(nr, nz)
    # elements
    e_n = np.stack([nid[:-1, :-1], nid[1:, :-1], nid[1:, 1:], nid[:-1, 1:]], -1).reshape(-1, 4)
    rc = 0.25 * (R.ravel()[e_n].sum(1))
    zc = 0.25 * (Z.ravel()[e_n].sum(1))
    mu, lam = elastic_model(model)(zc / 1e3)
    ne = len(e_n)
    xr = R.ravel()[e_n]
    xz = Z.ravel()[e_n]
    gp = np.array([-1, 1]) / np.sqrt(3)
    rows, cols, vals = [], [], []
    for xi in gp:
        for eta in gp:
            N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            J11 = xr @ dNx
            J12 = xz @ dNx
            J21 = xr @ dNe
            J22 = xz @ dNe
            det = J11 * J22 - J12 * J21
            dNr = (J22[:, None] * dNx[None, :] - J12[:, None] * dNe[None, :]) / det[:, None]
            dNz = (-J21[:, None] * dNx[None, :] + J11[:, None] * dNe[None, :]) / det[:, None]
            rg = xr @ N
            B = np.zeros((ne, 4, 8))
            B[:, 0, 0::2] = dNr                 # e_rr
            B[:, 1, 0::2] = N[None, :] / rg[:, None]   # e_tt
            B[:, 2, 1::2] = dNz                 # e_zz
            B[:, 3, 0::2] = dNz                 # g_rz
            B[:, 3, 1::2] = dNr
            D = np.zeros((ne, 4, 4))
            D[:, :3, :3] = lam[:, None, None]
            for i in range(3):
                D[:, i, i] += 2 * mu
            D[:, 3, 3] = mu
            w = 2 * np.pi * rg * det
            Ke = np.einsum("eki,ekl,elj->eij", B, D, B) * w[:, None, None]
            dof = np.stack([2 * e_n, 2 * e_n + 1], -1).reshape(ne, 8)
            rows.append(np.repeat(dof, 8, 1).ravel())
            cols.append(np.tile(dof, (1, 8)).ravel())
            vals.append(Ke.ravel())
    ndof = 2 * nr * nz
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)).tocsr()
    # load: pressure p on r <= a0 at z = 0, total 1 N downward (+z)
    f = np.zeros(ndof)
    p = 1.0 / (np.pi * (a0 * 1e3) ** 2)
    rs = r * 1e3
    for i in range(nr - 1):
        r1, r2 = rs[i], rs[i + 1]
        if r1 >= a0 * 1e3:
            break
        r2c = min(r2, a0 * 1e3)
        # linear shape functions on [r1, r2]; integrate N p 2 pi r dr over [r1, r2c] (Gauss 4 pts)
        xg, wg = np.polynomial.legendre.leggauss(4)
        rr = 0.5 * (r1 + r2c) + 0.5 * (r2c - r1) * xg
        ww = 0.5 * (r2c - r1) * wg
        N1 = (r2 - rr) / (r2 - r1)
        N2 = (rr - r1) / (r2 - r1)
        f[2 * nid[i, 0] + 1] += np.sum(ww * N1 * p * 2 * np.pi * rr)
        f[2 * nid[i + 1, 0] + 1] += np.sum(ww * N2 * p * 2 * np.pi * rr)
    # boundary conditions: u_r = 0 on the axis; u = 0 on the far boundaries
    fixed = np.zeros(ndof, bool)
    fixed[2 * nid[0, :]] = True
    fixed[2 * nid[-1, :]] = True
    fixed[2 * nid[-1, :] + 1] = True
    fixed[2 * nid[:, -1]] = True
    fixed[2 * nid[:, -1] + 1] = True
    free = ~fixed
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    # stresses at element centres
    xi = eta = 0.0
    N = 0.25 * np.ones(4)
    dNx = 0.25 * np.array([-1, 1, 1, -1.0])
    dNe = 0.25 * np.array([-1, -1, 1, 1.0])
    J11, J12, J21, J22 = xr @ dNx, xz @ dNx, xr @ dNe, xz @ dNe
    det = J11 * J22 - J12 * J21
    dNr = (J22[:, None] * dNx[None, :] - J12[:, None] * dNe[None, :]) / det[:, None]
    dNz = (-J21[:, None] * dNx[None, :] + J11[:, None] * dNe[None, :]) / det[:, None]
    ur = u[2 * e_n]
    uz = u[2 * e_n + 1]
    err = (dNr * ur).sum(1)
    ett = (N * ur).sum(1) / rc
    ezz = (dNz * uz).sum(1)
    grz = (dNz * ur).sum(1) + (dNr * uz).sum(1)
    tr = err + ett + ezz
    S = dict(srr=lam * tr + 2 * mu * err, stt=lam * tr + 2 * mu * ett, szz=lam * tr + 2 * mu * ezz, srz=mu * grz)
    # surface strain (top row of element centres is at z = 0.125 km; use nodal displacement at z = 0)
    ur0 = u[2 * nid[:, 0]]
    surf = dict(r=rs, ur=ur0, uz=u[2 * nid[:, 0] + 1])
    shape = (nr - 1, nz - 1)
    out = dict(rc=rc.reshape(shape)[:, 0] / 1e3, zc=zc.reshape(shape)[0, :] / 1e3,
               **{k: v.reshape(shape) for k, v in S.items()}, surf=surf, model=model,
               mu_c=mu.reshape(shape)[0, :], lam_c=lam.reshape(shape)[0, :])
    return out


def kernel_at_depth(fe, zq_km):
    """Interpolate the element-centre stress tables to depth zq (km): returns r (km) and dict of arrays."""
    zc = fe["zc"]
    j = np.searchsorted(zc, zq_km) - 1
    j = np.clip(j, 0, len(zc) - 2)
    w = (zq_km - zc[j]) / (zc[j + 1] - zc[j])
    return fe["rc"], {k: (1 - w) * fe[k][:, j] + w * fe[k][:, j + 1] for k in ("srr", "stt", "szz", "srz")}
