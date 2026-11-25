#!/usr/bin/env python3
"""Shared geometry / 1D ray tracing for the LFE secondary-phase study (scripts/sci_lfe_phases*.py).

Geometry: Shelly family locations (data/families/shelly_family_locations.csv); Bloch et al. 2023 interfaces t (top of
the low-velocity layer), c, m (slab Moho) interpolated at each family (data/slab/bloch2023/grids_masked).
1D layered model per family (horizontal layers):
   surface .. t_F : Furlong 2024 1D Vp (data/velocity/furlong2024/vel_p.1d), Vs = Vp / 1.78 (as traveltimes_3d.py)
   t_F .. m_F     : Bloch layer-2 (oceanic crust below c) Vs = 3.80 km/s, Vp/Vs = 1.88 (control points ME38/ME39/
                    O01C/KMR: Vs 3.62-3.98, Vp/Vs 1.86-1.91)
   below m_F      : mantle Vp 8.0, Vs 4.5 (only the reflection point is used)
Phase delays (s) relative to the direct phase, computed with exact layered ray tracing (ray parameter solved by
bisection for the epicentral distance):
   SmS-S, PmP-P, PmS-S, SmP-P : reflection from a reflector a height H BELOW the source (H grid, km)
   S-Sp (Sp precursor lead before S), Ps-P (P-to-S conversion at a reflector a height h ABOVE the source, after P),
   S-Ps lead (before S), Sp-P (after P)
"""
import os
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/lfe_phases'
FIG = f'{ROOT}/figures'
VPVS_UP = 1.78
VS2, VPVS2 = 3.80, 1.88
QUAKES = {'petrolia': '2021-12-20T20:10', 'ferndale': '2022-12-20T10:34', 'm7': '2024-12-05T18:44'}
RESTART = '2024-12-28'   # full LFE restart ~23 d after the M7 (NOTES 2026-11-21)
HGRID = np.round(np.arange(0.0, 15.01, 0.05), 3)


def epoch_s(x):
    x = pd.DatetimeIndex(x)
    return np.asarray((x - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1), dtype=float)


def stations():
    c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv')
    s = c.groupby(['net', 'sta']).agg(lat=('lat', 'first'), lon=('lon', 'first'), elev=('elev', 'first'),
                                      depth=('depth', 'max')).reset_index()
    s['code'] = s.net + '.' + s.sta
    return s.set_index('code')


def bloch_grid():
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    I = {}
    for k in 'tcm':
        with h5py.File(f'{ROOT}/data/slab/bloch2023/grids_masked/{k}_depth_masked.nc', 'r') as f:
            I[k] = RegularGridInterpolator((f['lat'][:], f['lon'][:]), f['z'][:], bounds_error=False)
    return I


def families():
    F = pd.read_csv(f'{ROOT}/data/families/shelly_family_locations.csv').rename(columns={'idnum': 'F'})
    I = bloch_grid()
    for k in 'tcm':
        F[f'z_{k}'] = [float(I[k]((a, b))) for a, b in zip(F.lat, F.lon)]
    F['H_m'] = F.z_m - F.depth_km          # source height above slab Moho
    F['h_t'] = F.depth_km - F.z_t          # depth of source below t
    return F.set_index('F')


def furlong_layers():
    a = np.loadtxt(f'{ROOT}/data/velocity/furlong2024/vel_p.1d')
    tops = a[0::2, 0]
    vp = a[0::2, 1]
    return tops, vp


class Model1D:
    """layered model for one family: depth below sea level of layer tops, Vp, Vs."""

    def __init__(self, z_t, z_m, top=-1.0):
        tops, vp = furlong_layers()
        zt, vpl = [], []
        for i in range(len(tops)):
            z0 = tops[i]
            z1 = tops[i + 1] if i + 1 < len(tops) else 1e9
            if z1 <= top or z0 >= z_t:
                continue
            zt.append(max(z0, top))
            vpl.append(vp[i])
        self.tops = np.array(zt + [z_t])
        self.vp = np.array(vpl + [VS2 * VPVS2])
        self.vs = np.r_[np.array(vpl) / VPVS_UP, VS2]
        self.z_t, self.z_m = z_t, z_m

    def legs(self, z0, z1, wave):
        """layer thicknesses and velocities between depths z0<z1."""
        z0, z1 = min(z0, z1), max(z0, z1)
        bot = np.r_[self.tops[1:], 1e9]
        h = np.clip(np.minimum(bot, z1) - np.maximum(self.tops, z0), 0, None)
        v = self.vp if wave == 'P' else self.vs
        k = h > 0
        return h[k], v[k]


def trace(X, legs):
    """legs: list of (h, v); h may be (n,) per layer or (nB, n) batch. Solve sum X(p) = X (km) by vectorised
    bisection; return T (nB,) or scalar."""
    hs, vs = [], []
    for h, v in legs:
        h = np.atleast_2d(np.asarray(h, float))
        hs.append(h)
        vs.append(np.broadcast_to(np.asarray(v, float), h.shape[-1:]))
    nB = max(h.shape[0] for h in hs)
    h = np.concatenate([np.broadcast_to(x, (nB, x.shape[1])) for x in hs], axis=1)
    v = np.concatenate(vs)[None, :]
    pmax = 1.0 / v.max() * (1 - 1e-10)

    def xt(p):
        c = np.sqrt(1 - (p[:, None] * v) ** 2)
        return np.sum(h * p[:, None] * v / c, 1), np.sum(h / (v * c), 1)
    lo, hi = np.zeros(nB), np.full(nB, pmax)
    bad = xt(hi)[0] < X
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        less = xt(mid)[0] < X
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    T = xt(0.5 * (lo + hi))[1]
    T[bad] = np.nan
    return T if nB > 1 else T[0]


def trace_v(X, h, V):
    """fixed slices h (n,), per-batch velocities V (nB, n) -> T (nB,)"""
    pmax = 1.0 / V.max(1) * (1 - 1e-10)

    def xt(p):
        c = np.sqrt(1 - (p[:, None] * V) ** 2)
        return np.sum(h * p[:, None] * V / c, 1), np.sum(h / (V * c), 1)
    lo, hi = np.zeros(len(V)), pmax.copy()
    bad = xt(hi)[0] < X
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        less = xt(mid)[0] < X
        lo = np.where(less, mid, lo)
        hi = np.where(less, hi, mid)
    T = xt(0.5 * (lo + hi))[1]
    T[bad] = np.nan
    return T


def phase_tables(X, zs, M, zr_elev=0.0):
    """delay tables vs H (HGRID) for one family-station. zs source depth (km bsl), receiver at -zr_elev km."""
    top = -zr_elev
    up = {w: M.legs(top, zs, w) for w in 'PS'}
    TS = float(trace(X, [up['S']]))
    TP = float(trace(X, [up['P']]))
    out = dict(TS=TS, TP=TP)
    Hc = np.maximum(HGRID, 1e-4)[:, None]
    lay = {'P': M.vp[-1], 'S': M.vs[-1]}
    for nm, wd, wu in (('SmS', 'S', 'S'), ('PmP', 'P', 'P'), ('PmS', 'P', 'S'), ('SmP', 'S', 'P')):
        T = trace(X, [(Hc, [lay[wd]]), (Hc, [lay[wu]]), up[wu]])
        out[nm] = T - (TS if wu == 'S' else TP)
        out[nm][HGRID <= 0] = np.nan
    # converting interface a height H ABOVE the source: fine slices (0.05 km) between the surface and the source;
    # slices above the converter carry the transmitted wave type, below it the incident one (vectorised over H)
    dzs = 0.05
    zb = np.arange(top, zs, dzs)
    hsl = np.diff(np.r_[zb, zs])
    zc = zb + hsl / 2
    lay = np.searchsorted(M.tops, zc, side='right') - 1
    vpl, vsl = M.vp[lay], M.vs[lay]
    for nm, wb, wa in (('Sp_above', 'S', 'P'), ('Ps_above', 'P', 'S')):
        za = zs - HGRID
        valid = (HGRID > 0) & (za > top + 0.5)
        vb = vsl if wb == 'S' else vpl
        va = vsl if wa == 'S' else vpl
        V = np.where(zc[None, :] < za[valid, None], va[None, :], vb[None, :])
        res = np.full(len(HGRID), np.nan)
        res[valid] = trace_v(X, hsl, V) - TP
        out[nm] = res
    return out


def geometry(recompute=False):
    """per family-station table + delay tables -> OUT/geometry.csv, OUT/delay_tables.npz"""
    fn = f'{OUT}/delay_tables.npz'
    if os.path.exists(fn) and not recompute:
        z = np.load(fn)
        return pd.read_csv(f'{OUT}/geometry.csv'), {k: z[k] for k in z.files}
    os.makedirs(OUT, exist_ok=True)
    Fm = families()
    S = stations()
    rows, tabs = [], {k: [] for k in ('SmS', 'PmP', 'PmS', 'SmP', 'Sp_above', 'Ps_above')}
    for F, f in Fm.iterrows():
        M = Model1D(f.z_t, f.z_m)
        for code, s in S.iterrows():
            X = np.hypot((s.lat - f.lat) * 111.19, (s.lon - f.lon) * 111.19 * np.cos(np.radians(f.lat)))
            zrec = s.elev / 1000.0 - s.depth / 1000.0
            o = phase_tables(X, f.depth_km, M, zrec)
            rows.append(dict(F=F, sta=code, X=X, zs=f.depth_km, z_t=f.z_t, z_m=f.z_m, H_m=f.H_m, h_t=f.h_t,
                             TS=o['TS'], TP=o['TP']))
            for k in tabs:
                tabs[k].append(o[k])
    G = pd.DataFrame(rows)
    G.to_csv(f'{OUT}/geometry.csv', index=False)
    tabs = {k: np.array(v) for k, v in tabs.items()}
    np.savez(fn, H=HGRID, **tabs)
    return G, tabs
