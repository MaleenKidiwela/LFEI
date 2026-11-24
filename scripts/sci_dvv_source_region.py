#!/usr/bin/env python3
"""Source-region dv/v from S-P times of LFE epoch stacks (detect or bound a velocity change within ~5-10 km of
the LFEs at the M7 2024-12-05, Ferndale 2022-12-20 and over the post-2021 decline).

Input: data/sci/dvv_source_region/meas/fam{F}.parquet (scripts/sci_dvv_source_region_measure.py): per family x
channel x epoch stack (~400 isolated LFEs; quake..+60 d windows one stack) the S and P delays (s, stack minus
all-event reference, bootstrap SE).  EXCL rows dropped; free channel offsets after BREAKS2 (sci_dvv_instrument_check).

Geometry (scripts/traveltimes_3d.py grids, Furlong 3D; 1D for sensitivity): at each family's 3D location
  g_S, g_P = grad T (s/km, x east, y north, z down) = source-position derivatives of S, P travel times.
  Volume kernels K (s): time the ray spends inside the volume:
    sph5 / sph10 : sphere of radius 5 / 10 km about the family  K = T(src) - T(src + R u)   (u: ray direction)
    slab5        : 5-km layer centred on the family (up-going ray crosses its upper 2.5 km)   K = T(src) - T(src + 2.5/cos(i) u)
  dv/v eps in a volume => delay -eps*K on that phase.
Per-family joint inversion over all stacks k (IRLS Huber):
  S_kc = tau_k + g_S,c . dx_k - sum_e eps_e,k K^S_e,c + off_c,seg     (channel offsets only after BREAKS2 breaks)
  P_kc = tau_k + g_P,c . dx_k - sum_e eps_e,k K^P_e,c + off_c,seg
  models: 'loc' (no eps), 'sph5' / 'sph10' (eps_S = eps_P = eps: only S-P constrains it), 'sph5S' (eps_P = 0),
          'sph5P' (eps_S = 0), 'slab5SP' (eps_S and eps_P separately via the 1/cos(i) pattern), 'sph5_noz' (dz = 0).
  Receiver correction (optional): station S, P delay corrections c_S(t), c_P(t) subtracted before the inversion.
Time model, network level (all families, offsets per family): y_Fk = a_F + sin + cos + steps (Petrolia, Ferndale, M7)
  + early 60-d windows after each quake + ramp from 2021-06; D = mean 2025 - mean 2021 (steps + ramp).
  Reported: M7 step (60 d - ...), restart window (M7 step + early, i.e. first 60 d incl. silence/restart), Ferndale,
  decline D.  Errors: bootstrap over stacks (B=200) and placebo (all model dates shifted back 5.25..12.25 yr, fitted on
  pre-2021-06 stacks; 15 shifts); honest SE = max(boot, placebo).
Stages: geom | invert | timefit | synth | placebo_rep | figs | all
Outputs data/sci/dvv_source_region/ ; figures/sci_dvv_source_region_*.png
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
os.chdir(ROOT)
from dvv_cwire import epoch_s  # noqa: E402
from sci_dvv_instrument_check import EXCL, BREAKS2  # noqa: E402

OUT = 'data/sci/dvv_source_region'
MEAS = f'{OUT}/meas'
FIG = 'figures'
DAY = 86400.0
YR = 365.25 * DAY
TM = "+proj=tmerc +lon_0=-122.7 +lat_0=40.4 +k=1 +units=km +ellps=WGS84"
SENSOR = {'PB.B045': 0.2176, 'PB.B046': 0.1875, 'PB.B047': 0.1753, 'PB.B049': 0.592, 'PB.B932': 0.412,
          'PB.B933': 0.139, 'PB.B934': 0.492, 'PB.B935': 0.1524}
REP9 = ['NC.KMR', 'NC.KMPB', 'PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'BK.BRIC', 'BK.HAYF']


def _ep(x):
    return epoch_s([pd.Timestamp(x)])[0]


T_Q = {'pet': _ep('2021-12-20T20:10'), 'fer': _ep('2022-12-20T10:34'), 'm7': _ep('2024-12-05T18:44')}
T_RAMP = _ep('2021-06-01')
Y08 = _ep('2008-01-01')
PN = ['sin', 'cos', 'pet', 'fer', 'm7', 'epet', 'efer', 'em7', 'ramp']
SHIFTS = [5.25 + 0.5 * i for i in range(15)]


def sta_of(ch):
    return ch.rsplit('.', 2)[0]


# ------------------------------------------------------------------------------------------------ time model
def tcols(t, s=0.0):
    sh = s * YR
    yr = (t - Y08) / YR
    c = [np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    for k in ['pet', 'fer', 'm7']:
        c.append((t >= T_Q[k] - sh).astype(float))
    for k in ['pet', 'fer', 'm7']:
        tq = T_Q[k] - sh
        c.append(((t >= tq) & (t < tq + 60 * DAY)).astype(float))
    c.append(np.clip((t - (T_RAMP - sh)) / YR, 0, None))
    return np.array(c).T


G21 = np.array([_ep(f'2021-{m:02d}-15') for m in range(1, 13)])
G25 = np.array([_ep(f'2025-{m:02d}-15') for m in range(1, 13)])


def dvec(s=0.0):
    C = tcols(G25 - s * YR, s).mean(0) - tcols(G21 - s * YR, s).mean(0)
    C[[0, 1, 5, 6, 7]] = 0.0
    return C


QUANT = {'m7': lambda th, s=0: th[4], 'restart': lambda th, s=0: th[4] + th[7], 'fer': lambda th, s=0: th[3],
         'fer60': lambda th, s=0: th[3] + th[6], 'pet': lambda th, s=0: th[2], 'D': lambda th, s=0: th @ dvec(s),
         'ramp': lambda th, s=0: th[8]}


# ------------------------------------------------------------------------------------------------ geometry
class Grid:
    def __init__(self, model='3d'):
        g = json.load(open(f'data/velocity/tt_{model}/grid.json'))
        self.ax = (np.array(g['sub_x']), np.array(g['sub_y']), np.array(g['sub_z']))
        self.model = model
        self.I = {}

    def interp(self, sta, ph):
        key = (sta, ph)
        if key not in self.I:
            f = f'data/velocity/tt_{self.model}/{sta}.npz'
            if not os.path.exists(f):
                self.I[key] = None
            else:
                z = np.load(f)
                self.I[key] = RegularGridInterpolator(self.ax, z['t' + ph].astype(float), bounds_error=False,
                                                      fill_value=np.nan)
        return self.I[key]

    def inside(self, p):
        return all(self.ax[i][0] <= p[i] <= self.ax[i][-1] for i in range(3))


def fam_xyz(F, model='3d'):
    from pyproj import Transformer
    L = json.load(open(f'data/locations/fam{F}_{model}.json'))
    tr = Transformer.from_crs('EPSG:4326', TM, always_xy=True)
    x, y = tr.transform(L['lon'], L['lat'])
    return np.array([x, y, L['depth_km']])


def station_geom(grid, sta, src, h=0.25):
    """-> dict with gS, gP (3,), T_S, T_P, K kernels for sph5, sph10, slab5 (S and P), cos_i (S)."""
    out = {}
    for ph in ('s', 'p'):
        I = grid.interp(sta, ph)
        if I is None:
            return None
        T0 = float(I(src)[0])
        g = np.array([(I(src + h * e)[0] - I(src - h * e)[0]) / (2 * h) for e in np.eye(3)])
        if not np.all(np.isfinite(g)) or not np.isfinite(T0):
            return None
        sl = np.linalg.norm(g)
        u = -g / sl                                       # ray direction leaving the source toward the station
        cosi = abs(u[2])
        P = ph.upper()
        out[f'T_{P}'] = T0
        out[f'g{P}'] = g
        out[f'cos_{P}'] = cosi

        def K(L):
            p = src + L * u
            if grid.inside(p):
                v = float(I(p)[0])
                if np.isfinite(v):
                    return T0 - v
            return L * sl
        out[f'K{P}_sph5'] = K(5.0)
        out[f'K{P}_sph10'] = K(10.0)
        out[f'K{P}_slab5'] = K(2.5 / max(cosi, 0.2))
    return out


def stage_geom(a=None, model='3d'):
    grid = Grid(model)
    stas = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'data/velocity/tt_{model}/*.npz'))
    rows = []
    for F in range(1, 28):
        src = fam_xyz(F, '3d')
        for s in stas:
            g = station_geom(grid, s, src)
            if g is None:
                continue
            r = dict(F=F, sta=s, x=src[0], y=src[1], z=src[2])
            for k, v in g.items():
                if isinstance(v, np.ndarray):
                    for i, c in enumerate('xyz'):
                        r[f'{k}{c}'] = v[i]
                else:
                    r[k] = v
            rows.append(r)
    G = pd.DataFrame(rows)
    G.to_csv(f'{OUT}/geometry_{model}.csv', index=False)
    print(G.groupby('F')[['KS_sph5', 'KP_sph5', 'KS_sph10', 'KP_sph10', 'KS_slab5', 'cos_S']].median().round(3).to_string())
    return G


# ------------------------------------------------------------------------------------------------ data
def excluded(ch, t):
    m = np.zeros(len(t), bool)
    for k, wins in EXCL.items():
        if ch == k or ch.startswith(k + '.'):
            for a_, b_ in wins:
                m |= (t >= _ep(a_)) & (t < _ep(b_))
    return m


def segment(ch, t):
    br = BREAKS2.get(sta_of(ch), [])
    return np.searchsorted(np.array([_ep(b) for b in br]), t, side='right') if br else np.zeros(len(t), int)


def load_rows(F, meas=MEAS):
    """long table: stack, t, n, chan, sta, phase, y, se, seg."""
    f = f'{meas}/fam{F}.parquet'
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f)
    out = []
    for ph, ccmin, semax in (('S', 0.6, 0.02), ('P', 0.4, 0.04)):
        y, se, cc = d[f'dt{ph}'], d[f'dt{ph}_se'], d[f'cc{ph}']
        ok = np.isfinite(y) & np.isfinite(se) & (se > 0) & (cc > ccmin) & (se < semax) & (np.abs(y) < 0.1)
        r = pd.DataFrame(dict(stack=d['stack'], t=d.t, n=d.n, chan=d.chan, phase=ph, y=y,
                              se=np.sqrt(se ** 2 + 0.0005 ** 2)))[ok]
        out.append(r)
    R = pd.concat(out, ignore_index=True)
    R['sta'] = R.chan.map(sta_of)
    keep = []
    for ch, g in R.groupby('chan'):
        g = g[~excluded(ch, g.t.values)].copy()
        g['seg'] = segment(ch, g.t.values)
        keep.append(g)
    R = pd.concat(keep, ignore_index=True)
    # channel needs >= 4 stacks
    R = R[R.groupby(['chan', 'phase']).y.transform('size') >= 4].reset_index(drop=True)
    stk = pd.read_csv(f'{meas}/fam{F}_stacks.csv')
    return R, stk


# ------------------------------------------------------------------------------------------------ inversion
MODELS = {
    'loc': [],
    'sph5': [('sph5', 'SP')],
    'sph10': [('sph10', 'SP')],
    'sph5S': [('sph5', 'S')],
    'sph5P': [('sph5', 'P')],
    'slab5': [('slab5', 'SP')],
    'slab5SP': [('slab5', 'S'), ('slab5', 'P')],
    'sph5_noz': [('sph5', 'SP')],
    'sph5_clk': [('sph5', 'SP')],      # clock-free: one time term per (stack, station) instead of per stack
    'sph10_clk': [('sph10', 'SP')],
    'sph5_net': [('sph5', 'SP')],      # one time term per (stack, network PB/NC/BK): robust to network clock offsets
    'sph10_net': [('sph10', 'SP')],
}


def design_rows(R, geo, model):
    """per row: coefficient vector over per-stack params [tau, dx, dy, dz, eps...]."""
    ph = R.phase.values
    G = geo.set_index('sta')
    stas = R.sta.values
    ok = np.isin(stas, G.index)
    n_eps = len(MODELS[model])
    A = np.zeros((len(R), 4 + n_eps))
    A[:, 0] = 1.0
    for i, c in enumerate('xyz'):
        gS = G[f'gS{c}'].reindex(stas).values
        gP = G[f'gP{c}'].reindex(stas).values
        A[:, 1 + i] = np.where(ph == 'S', gS, gP)
    for e, (vol, which) in enumerate(MODELS[model]):
        KS = G[f'KS_{vol}'].reindex(stas).values
        KP = G[f'KP_{vol}'].reindex(stas).values
        cS = -KS if 'S' in which else 0.0 * KS
        cP = -KP if 'P' in which else 0.0 * KP
        A[:, 4 + e] = np.where(ph == 'S', cS, cP)
    if model.endswith('_noz'):
        A[:, 3] = 0.0
    return A, ok


def invert_family(R, geo, model, w_extra=None, irls=3, huber=2.0):
    """joint inversion over stacks; returns DataFrame per stack (params, formal se, chi2r)."""
    A0, ok = design_rows(R, geo, model)
    R = R[ok].reset_index(drop=True)
    A0 = A0[ok]
    if w_extra is not None:
        w_extra = w_extra[ok]
    npar = A0.shape[1]
    stacks, si = np.unique(R['stack'].values, return_inverse=True)
    ns = len(stacks)
    # break offsets: (chan, phase, seg>0)
    key = R.chan.values + '|' + R.phase.values + '|' + R.seg.astype(str).values
    has_off = R.seg.values > 0
    okeys, oi = np.unique(key[has_off], return_inverse=True)
    nO = len(okeys)
    nC = 0
    if model.endswith('_clk') or model.endswith('_net'):
        grp = R.sta.values if model.endswith('_clk') else np.array([x.split('.')[0] for x in R.sta.values])
        ckey = R['stack'].astype(str).values + '|' + grp
        ckeys, ci = np.unique(ckey, return_inverse=True)
        nC = len(ckeys)
    ncol = ns * npar + nO + nC
    nrow = len(R)
    A = np.zeros((nrow, ncol))
    for j in range(npar):
        A[np.arange(nrow), si * npar + j] = A0[:, j]
    if nO:
        A[np.where(has_off)[0], ns * npar + oi] = 1.0
    if nC:
        A[np.arange(nrow), ns * npar + nO + ci] = 1.0
    if model.endswith('_noz'):
        for k in range(ns):
            A[:, k * npar + 3] = 0.0
    y = R.y.values
    w0 = 1.0 / R.se.values ** 2
    if w_extra is not None:
        w0 = w0 * w_extra
    w = w0.copy()
    reg = 1e-8 * np.eye(ncol)
    if model.endswith('_noz'):
        for k in range(ns):
            reg[k * npar + 3, k * npar + 3] = 1e6
    if model.endswith('_clk') or model.endswith('_net'):
        for k in range(ns):
            reg[k * npar, k * npar] = 1e6
    for it in range(irls + 1):
        N = (A * w[:, None]).T @ A
        sc = np.median(np.diag(N)[np.diag(N) > 0])
        N2 = N + reg * sc
        x = np.linalg.solve(N2, A.T @ (w * y))
        res = y - A @ x
        if it < irls:
            z = np.abs(res) * np.sqrt(w0)
            s_ = 1.4826 * np.median(z) + 1e-20
            z = z / s_
            w = w0 * np.where(z <= huber, 1.0, huber / np.maximum(z, 1e-12))
    dof = max(nrow - ncol, 1)
    chi = float(np.sum(w * res ** 2) / dof)
    cov = np.linalg.inv(N2)
    se = np.sqrt(np.clip(np.diag(cov), 0, None)) * np.sqrt(max(chi, 1.0))
    names = ['tau', 'dx', 'dy', 'dz'] + [f'eps{e}' for e in range(npar - 4)]
    out = pd.DataFrame({'stack': stacks})
    for j, nm in enumerate(names):
        out[nm] = x[np.arange(ns) * npar + j]
        out[nm + '_se'] = se[np.arange(ns) * npar + j]
    nS = np.bincount(si, weights=(R.phase.values == 'S').astype(float), minlength=ns)
    nP = np.bincount(si, weights=(R.phase.values == 'P').astype(float), minlength=ns)
    out['nS'], out['nP'] = nS, nP
    out['chi2r'] = chi
    return out, dict(res=res, w=w, R=R)


def family_job(args):
    F, models, stations, rec, inj, meas = args
    L = load_rows(F, meas)
    if L is None:
        return F, None
    R, stk = L
    if stations is not None:
        R = R[R.sta.isin(stations)].reset_index(drop=True)
    geo = GEO.get(F)
    if geo is None or len(R) < 50:
        return F, None
    if rec is not None:                   # receiver correction: dict sta -> callable(t) -> (cS, cP) in s
        corr = np.zeros(len(R))
        for s_, f in rec.items():
            m = R.sta.values == s_
            if m.any():
                cS, cP = f(R.t.values[m])
                corr[m] = np.where(R.phase.values[m] == 'S', cS, cP)
        R = R.assign(y=R.y - corr)
    if inj is not None:                   # synthetic injection at the delay level
        R = R.assign(y=R.y + inj(F, R, geo))
    out = {}
    for m in models:
        try:
            o, _ = invert_family(R, geo, m)
        except np.linalg.LinAlgError:
            continue
        o = o.merge(stk[['stack', 't', 'n', 'post60']], on='stack', how='left')
        o['F'] = F
        out[m] = o
    return F, out


GEO = {}


def load_geo(model='3d'):
    f = f'{OUT}/geometry_{model}.csv'
    if not os.path.exists(f):
        stage_geom(model=model)
    G = pd.read_csv(f)
    for F, g in G.groupby('F'):
        GEO[F] = g.reset_index(drop=True)
    return G


def run_invert(models, stations=None, rec=None, inj=None, meas=MEAS, procs=10, fams=None):
    fams = fams or sorted(int(os.path.basename(f)[3:-8]) for f in glob.glob(f'{meas}/fam*.parquet'))
    jobs = [(F, models, stations, rec, inj, meas) for F in fams]
    res = {m: [] for m in models}
    if procs > 1:
        with Pool(procs) as p:
            it = p.imap_unordered(family_job, jobs)
            for F, out in it:
                if out:
                    for m, o in out.items():
                        res[m].append(o)
    else:
        for j in jobs:
            F, out = family_job(j)
            if out:
                for m, o in out.items():
                    res[m].append(o)
    return {m: pd.concat(v, ignore_index=True) for m, v in res.items() if v}


# ------------------------------------------------------------------------------------------------ time model
def usable_cols(t, s=0.0, nmin=2):
    sh = s * YR
    use = np.ones(len(PN), bool)
    for j, k in [(2, 'pet'), (3, 'fer'), (4, 'm7')]:
        tq = T_Q[k] - sh
        use[j] = ((t < tq) & (t > tq - 2 * YR)).sum() >= nmin and ((t >= tq + 60 * DAY) & (t < tq + 2 * YR)).sum() >= nmin
        use[j + 3] = ((t >= tq) & (t < tq + 60 * DAY)).sum() >= 1
    use[8] = (t < T_RAMP - sh).sum() >= nmin and (t > T_RAMP - sh + YR).sum() >= nmin
    return use


def fit_time(df, col, s=0.0, fam_offsets=True, w_mult=None, use=None, irls=2):
    """df rows (F, t, col, col_se) -> theta (len PN) nan if unused, formal se, chi2r."""
    d = df[np.isfinite(df[col]) & np.isfinite(df[col + '_se']) & (df[col + '_se'] > 0)]
    if len(d) < 8:
        return np.full(len(PN), np.nan), np.full(len(PN), np.nan), np.nan
    t = d.t.values
    C = tcols(t, s)
    if use is None:
        use = usable_cols(t, s)
    Fs, fi = np.unique(d.F.values, return_inverse=True)
    AF = np.zeros((len(d), len(Fs)))
    AF[np.arange(len(d)), fi] = 1
    A = np.c_[AF if fam_offsets else np.ones((len(d), 1)), C[:, use]]
    y = d[col].values
    w0 = 1 / d[col + '_se'].values ** 2
    if w_mult is not None:
        w0 = w0 * w_mult[d.index.values] if len(w_mult) == len(df) else w0
    w = w0.copy()
    for it in range(irls + 1):
        W = np.sqrt(w)[:, None]
        x, *_ = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)
        r = y - A @ x
        if it < irls:
            z = np.abs(r) * np.sqrt(w0)
            z = z / (1.4826 * np.median(z) + 1e-20)
            w = w0 * np.where(z <= 2.0, 1.0, 2.0 / np.maximum(z, 1e-12))
    nO = A.shape[1] - use.sum()
    chi = float(np.sum(w * r ** 2) / max(len(d) - A.shape[1], 1))
    try:
        cov = np.linalg.inv((A * w[:, None]).T @ A) * max(chi, 1.0)
        se_ = np.sqrt(np.diag(cov))[nO:]
    except np.linalg.LinAlgError:
        se_ = np.full(use.sum(), np.nan)
    th = np.full(len(PN), np.nan)
    se = np.full(len(PN), np.nan)
    th[use] = x[nO:]
    se[use] = se_
    return th, se, chi


def quantities(th, s=0.0):
    t2 = np.nan_to_num(th)
    out = {}
    for q, f in QUANT.items():
        need = {'m7': [4], 'restart': [4, 7], 'fer': [3], 'fer60': [3, 6], 'pet': [2], 'D': [8], 'ramp': [8]}[q]
        out[q] = f(t2, s) if np.all(np.isfinite(th[need])) else np.nan
    return out


def timefit_errors(df, col, nboot=200, seed=0, fam_offsets=True):
    """real + bootstrap over stacks (within families) + placebo shifts -> per quantity value, se_boot, se_plac."""
    rng = np.random.default_rng(seed)
    d = df[np.isfinite(df[col]) & np.isfinite(df[col + '_se'])].reset_index(drop=True)
    th, se, chi = fit_time(d, col, fam_offsets=fam_offsets)
    use = np.isfinite(th)
    real = quantities(th)
    # formal se of quantities
    B = []
    grp = [np.where(d.F.values == F)[0] for F in np.unique(d.F.values)]
    for b in range(nboot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp])
        db = d.iloc[idx].reset_index(drop=True)
        thb, _, _ = fit_time(db, col, use=use, fam_offsets=fam_offsets, irls=1)
        B.append(quantities(thb))
    B = pd.DataFrame(B)
    P = []
    pre = d[d.t < T_RAMP].reset_index(drop=True)
    for s in SHIFTS:
        thp, _, _ = fit_time(pre, col, s=s, fam_offsets=fam_offsets, irls=1)
        P.append(quantities(thp, s))
    P = pd.DataFrame(P)
    out = {}
    for q in QUANT:
        v = real[q]
        sb = float(np.nanstd(B[q])) if q in B else np.nan
        sp = float(np.sqrt(np.nanmean(P[q] ** 2))) if q in P and np.isfinite(P[q]).sum() >= 4 else np.nan
        out[q] = dict(val=v, se_boot=sb, se_plac=sp, n_plac=int(np.isfinite(P[q]).sum()) if q in P else 0,
                      se=np.nanmax([sb, sp]) if np.isfinite([sb, sp]).any() else np.nan, chi2r=chi)
    return out, B, P


# ------------------------------------------------------------------------------------------------ stages
def stage_invert(a):
    load_geo('3d')
    models = ['loc', 'sph5', 'sph10', 'sph5S', 'sph5P', 'slab5', 'slab5SP', 'sph5_noz', 'sph5_net', 'sph10_net',
              'sph5_clk']
    res = run_invert(models, procs=a.procs)
    pickle.dump(res, open(f'{OUT}/inv_all.pkl', 'wb'))
    res9 = run_invert(['loc', 'sph5', 'sph10'], stations=REP9, procs=a.procs)
    pickle.dump(res9, open(f'{OUT}/inv_rep9.pkl', 'wb'))
    for m, o in res.items():
        print(m, len(o), o[[c for c in o.columns if c.endswith('_se')]].median().round(5).to_dict())


PARAM_SCALE = {'vpvs': 100.0, 'eps0': 100.0, 'eps1': 100.0, 'dz': 1e3, 'dx': 1e3, 'dy': 1e3, 'tau': 1e3}
PARAM_UNIT = {'vpvs': '%', 'eps0': '%', 'eps1': '%', 'dz': 'm', 'dx': 'm', 'dy': 'm', 'tau': 'ms'}


def summarize(res, tag, params=None, nboot=200, quiet=False):
    """network time-model results for each model/param -> rows."""
    rows = []
    for m, o in res.items():
        pars = params or [p for p in ['eps0', 'eps1', 'vpvs', 'dz', 'dx', 'dy', 'tau'] if p in o.columns]
        for p in pars:
            if p not in o.columns:
                continue
            d = o[['F', 't', 'n', p, p + '_se']].copy()
            d = d[np.isfinite(d[p]) & (d[p + '_se'] > 0)]
            # drop absurd stacks (se > 20x median)
            d = d[d[p + '_se'] < 20 * d[p + '_se'].median()]
            out, B, P = timefit_errors(d, p, nboot=nboot)
            for q, v in out.items():
                sc = PARAM_SCALE[p]
                rows.append(dict(set=tag, model=m, param=p, unit=PARAM_UNIT[p], quantity=q, val=v['val'] * sc,
                                 se_boot=v['se_boot'] * sc, se_plac=v['se_plac'] * sc, n_plac=v['n_plac'],
                                 se=v['se'] * sc, z=v['val'] / v['se'] if v['se'] > 0 else np.nan, chi2r=v['chi2r'],
                                 n_stacks=len(d), n_fam=d.F.nunique()))
            if not quiet:
                r_ = [r for r in rows if r['model'] == m and r['param'] == p]
                print(f"[{tag}] {m:9s} {p:5s} " + '  '.join(f"{r['quantity']}={r['val']:+.3f}+/-{r['se']:.3f}"
                                                            for r in r_ if r['quantity'] in ('m7', 'restart', 'fer', 'D')),
                      flush=True)
    return rows


def per_family(res, model='sph5', p='eps0'):
    """family-differential terms: per family M7 step / restart / Ferndale / D (time model fitted per family),
    and deviations from the weighted network mean (receiver-free).  Errors: per-family bootstrap (B=100)."""
    o = res[model]
    rows = []
    for F, g in o.groupby('F'):
        d = g[['F', 't', 'n', p, p + '_se']].copy()
        d = d[np.isfinite(d[p]) & (d[p + '_se'] > 0)].reset_index(drop=True)
        if len(d) < 6:
            continue
        out, B, P = timefit_errors(d, p, nboot=100, seed=F)
        r = dict(F=F, n_stacks=len(d))
        for q in ('m7', 'restart', 'fer', 'D'):
            r[q] = out[q]['val'] * 100
            r[q + '_se'] = out[q]['se'] * 100
        rows.append(r)
    T = pd.DataFrame(rows)
    for q in ('m7', 'restart', 'fer', 'D'):
        ok = np.isfinite(T[q]) & np.isfinite(T[q + '_se']) & (T[q + '_se'] > 0)
        w = 1 / T.loc[ok, q + '_se'] ** 2
        mu = np.sum(w * T.loc[ok, q]) / w.sum()
        T[q + '_dev'] = T[q] - mu
        Q = float(np.sum(w * (T.loc[ok, q] - mu) ** 2))
        from scipy.stats import chi2
        T.attrs[q] = dict(mean=mu, Q=Q, dof=int(ok.sum() - 1), p=float(chi2.sf(Q, max(ok.sum() - 1, 1))))
    return T


def stage_timefit(a):
    res = pickle.load(open(f'{OUT}/inv_all.pkl', 'rb'))
    res9 = pickle.load(open(f'{OUT}/inv_rep9.pkl', 'rb'))
    o = res['slab5SP']                      # d ln(Vp/Vs) = eps_P - eps_S (slab model separates them)
    o['vpvs'] = o.eps1 - o.eps0
    o['vpvs_se'] = np.hypot(o.eps0_se, o.eps1_se)
    rows = summarize(res, 'all_stations')
    rows += summarize(res9, 'rep9_raw')
    S = pd.DataFrame(rows)
    S.to_csv(f'{OUT}/timefit_network.csv', index=False)
    fam = {}
    for m in ('sph5', 'sph10'):
        T = per_family(res, m)
        T.to_csv(f'{OUT}/family_terms_{m}.csv', index=False)
        fam[m] = T.attrs
        print(m, json.dumps(T.attrs, indent=0, default=float))
    json.dump(fam, open(f'{OUT}/family_terms_Q.json', 'w'), indent=1, default=float)


class Inject:
    """delay-level injection after t_on: kind in sph5, sph10, slab5 (eps, both phases), sph5S (S only),
    dz / dx (centroid shift, km), recv (station S delays dict, s; P = pfac * S)."""

    def __init__(self, kind, amp, t_on=None, recv=None, pfac=0.5, t_off=None):
        self.kind, self.amp, self.t_on, self.recv, self.pfac = kind, amp, t_on or T_Q['m7'], recv, pfac
        self.t_off = t_off

    def __call__(self, F, R, geo):
        G = geo.set_index('sta')
        st = R.sta.values
        isS = R.phase.values == 'S'
        on = (R.t.values >= self.t_on) & ((R.t.values < self.t_off) if self.t_off else True)
        add = np.zeros(len(R))
        k = self.kind
        if k in ('sph5', 'sph10', 'slab5'):
            add = np.where(isS, G[f'KS_{k}'].reindex(st).values, G[f'KP_{k}'].reindex(st).values) * (-self.amp)
        elif k == 'sph5S':
            add = np.where(isS, G['KS_sph5'].reindex(st).values * (-self.amp), 0.0)
        elif k in ('dz', 'dx', 'dy'):
            c = k[1]
            add = np.where(isS, G[f'gS{c}'].reindex(st).values, G[f'gP{c}'].reindex(st).values) * self.amp
        elif k == 'recv':
            s_ = np.array([self.recv.get(x, 0.0) for x in st])
            add = np.where(isS, s_, self.pfac * s_)
        return np.nan_to_num(add) * on


def quick_q(res, p='eps0', scale=100.0):
    """real-only network time fit -> dict of quantities (scaled)."""
    out = {}
    for m, o in res.items():
        if p not in o.columns:
            continue
        d = o[np.isfinite(o[p]) & (o[p + '_se'] > 0)].copy()
        d = d[d[p + '_se'] < 20 * d[p + '_se'].median()]
        th, se, chi = fit_time(d.reset_index(drop=True), p)
        q = quantities(th)
        out[m] = {k: v * scale for k, v in q.items()}
    return out


def stage_synth(a):
    load_geo('3d')
    st = pd.read_csv('data/sci/dvv_deep_shallow/station_terms_S_P.csv')
    recv_m7 = {r.sta: r.d_ms / 1e3 for r in st[(st.obs == 'S') & (st.X == 'm7')].itertuples()}
    models = ['sph5', 'sph10', 'slab5SP', 'sph5_noz']
    scen = [('baseline', None),
            ('sph5_eps-0.2', Inject('sph5', -0.002)),
            ('sph10_eps-0.1', Inject('sph10', -0.001)),
            ('slab5_eps-0.2', Inject('slab5', -0.002)),
            ('sph5S_epsS-0.2', Inject('sph5S', -0.002)),
            ('dz+20m', Inject('dz', 0.020)),
            ('dz+10m', Inject('dz', 0.010)),
            ('dx+30m', Inject('dx', 0.030)),
            ('recv_m7_S_P0.5', Inject('recv', 1.0, recv=recv_m7, pfac=0.5)),
            ('recv_m7_S_P0', Inject('recv', 1.0, recv=recv_m7, pfac=0.0)),
            ('restart_only_sph5_-0.5', Inject('sph5', -0.005, t_off=T_Q['m7'] + 60 * DAY))]
    rows = []
    base = None
    for name, inj in scen:
        res = run_invert(models, inj=inj, procs=a.procs)
        q = quick_q(res)
        qz = quick_q(res, 'dz', 1e3)
        if name == 'baseline':
            base, basez = q, qz
        for m in q:
            for k in ('m7', 'restart', 'fer', 'D'):
                rows.append(dict(scenario=name, model=m, quantity=k, rec_eps_pct=q[m][k],
                                 rec_minus_base_pct=q[m][k] - base[m][k],
                                 rec_dz_minus_base_m=qz.get(m, {}).get(k, np.nan) - basez.get(m, {}).get(k, np.nan)))
            if 'slab5SP' == m:
                q1 = quick_q({m: res[m]}, 'eps1')
                for k in ('m7', 'restart'):
                    rows.append(dict(scenario=name, model=m + '_epsP', quantity=k, rec_eps_pct=q1[m][k],
                                     rec_minus_base_pct=np.nan))
        print(name, {m: round(q[m]['m7'] - base[m]['m7'], 4) for m in q},
              {m: round(qz[m]['m7'] - basez[m]['m7'], 2) for m in qz}, flush=True)
    Z = pd.DataFrame(rows)
    # slab5SP epsP baselines
    for m in ['slab5SP_epsP']:
        for k in ('m7', 'restart'):
            b = Z[(Z.scenario == 'baseline') & (Z.model == m) & (Z.quantity == k)].rec_eps_pct.values
            sel = (Z.model == m) & (Z.quantity == k)
            Z.loc[sel, 'rec_minus_base_pct'] = Z.loc[sel, 'rec_eps_pct'] - (b[0] if len(b) else np.nan)
    Z.to_csv(f'{OUT}/synthetic_recovery.csv', index=False)
    print(Z[Z.quantity == 'm7'].round(4).to_string())


# ------------------------------------------------------------------------------------------------ receiver terms
REP_PN = ['sin', 'cos', 'pet', 'fer', 'm7', 'early', 'ramp']
REP_MAP = [0, 1, 2, 3, 4, 7, 8]            # rep param -> column of tcols


class RecCorr:
    """station receiver S, P delay corrections from the repeater station fits (rep_station_fit_reps.pkl):
    cS(t) = f_{dS_rel}(t),  cP(t) = cS(t) - f_{dSP}(t)  (S-P exactly the repeater S-P; S pattern relative).
    replicate b (bootstrap) or None (real).  cls: 'deep' (19 km) or 'shallow' (8 km)."""

    def __init__(self, cls='deep', b=None, sp_only=False, stations=None):
        reps = pickle.load(open(f'{OUT}/rep/rep_station_fit_reps.pkl', 'rb'))  # our own output
        self.th = {}
        for (obs, sta, c), v in reps.items():
            if c != cls or (stations is not None and sta not in stations):
                continue
            x = v['x'] if b is None else v['B'][b]
            self.th.setdefault(sta, {})[obs] = np.nan_to_num(x)
        self.sp_only = sp_only
        self.th = {s: d for s, d in self.th.items() if 'dSP' in d and 'dS_rel' in d}

    def items(self):
        return [(s, self._f(s)) for s in self.th]

    def _f(self, sta):
        d = self.th[sta]

        def f(t):
            C = tcols(np.asarray(t, float))[:, REP_MAP]
            sp = C @ d['dSP']
            cS = np.zeros(len(t)) if self.sp_only else C @ d['dS_rel']
            return cS, cS - sp
        return f


class RecDict(dict):
    pass


def rec_dict(cls='deep', b=None, sp_only=False, stations=None):
    rc = RecCorr(cls, b, sp_only, stations)
    D = RecDict()
    for s, f in rc.items():
        D[s] = _RecF(rc.th[s], sp_only)
    return D


class _RecF:
    def __init__(self, d, sp_only):
        self.d, self.sp_only = d, sp_only

    def __call__(self, t):
        C = tcols(np.asarray(t, float))[:, REP_MAP]
        sp = C @ self.d['dSP']
        cS = np.zeros(len(t)) if self.sp_only else C @ self.d['dS_rel']
        return cS, cS - sp


class _RecDS:
    """receiver correction from the deep_shallow relative station S terms (station_terms_S_P.csv, gauge: mean of
    long-record stations = 0): cS(t) = m7 * H(t - M7) + D * ramp(t)/ramp_D,  cP = pfac * cS."""

    def __init__(self, m7, D, pfac):
        self.m7, self.D, self.pfac = m7, D, pfac
        self.rD = dvec()[8]

    def __call__(self, t):
        t = np.asarray(t, float)
        C = tcols(t)
        cS = self.m7 * C[:, 4] + (self.D - self.m7) / self.rD * C[:, 8]
        return cS, self.pfac * cS


def rec_ds(pfac=0.5):
    st = pd.read_csv('data/sci/dvv_deep_shallow/station_terms_S_P.csv')
    st = st[st.obs == 'S']
    m7 = st[st.X == 'm7'].set_index('sta').d_ms / 1e3
    D = st[st.X == 'D'].set_index('sta').d_ms / 1e3
    out = RecDict()
    for s_ in sorted(set(m7.index) | set(D.index)):
        out[s_] = _RecDS(float(m7.get(s_, 0.0)), float(D.get(s_, m7.get(s_, 0.0))), pfac)
    return out


PB = ['PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B933', 'PB.B934', 'PB.B935']


def stage_reccorr(a):
    """LFE inversion on the REP9 stations with the 19-km repeater receiver correction (real + 30 bootstrap
    replicates of the correction), and the 8-km-repeater correction as an alternative."""
    load_geo('3d')
    rows = []
    out = {}
    for tag, kw in [('rep9_corr_deep', dict(cls='deep')), ('rep9_corr_deep_SPonly', dict(cls='deep', sp_only=True)),
                    ('rep9_corr_shallow', dict(cls='shallow'))]:
        res = run_invert(['sph5', 'sph10'], stations=REP9, rec=rec_dict(**kw), procs=a.procs)
        out[tag] = res
        rows += summarize(res, tag, params=['eps0'])
    for tag, kw in [('all_corr_ds_P0.5', dict(rec=rec_ds(0.5))), ('all_corr_ds_P0', dict(rec=rec_ds(0.0))),
                    ('boreholes_raw', dict(stations=PB)), ('boreholes_corr_ds_P0.5', dict(stations=PB, rec=rec_ds(0.5))),
                    ('surface_raw', dict(stations=None, exclude=PB))]:
        if 'exclude' in kw:
            G = pd.concat(GEO.values())
            kw = dict(stations=sorted(set(G.sta) - set(PB)))
        res = run_invert(['sph5', 'sph10', 'loc'], procs=a.procs, **kw)
        out[tag] = res
        rows += summarize(res, tag, params=['eps0', 'dz'])
    pickle.dump(out, open(f'{OUT}/inv_rep9_corr.pkl', 'wb'))
    # propagate the correction's own uncertainty: bootstrap replicates of the repeater fits
    reps = []
    for b in range(20):
        res = run_invert(['sph5', 'sph10'], stations=REP9, rec=rec_dict('deep', b=b), procs=a.procs)
        q = quick_q(res)
        reps.append({f'{m}|{k}': v for m, d in q.items() for k, v in d.items()})
    R = pd.DataFrame(reps)
    R.to_csv(f'{OUT}/reccorr_replicates.csv', index=False)
    S = pd.DataFrame(rows)
    sd = R.std()
    S['se_reccorr'] = [sd.get(f'{r.model}|{r.quantity}', np.nan) if r.param == 'eps0' and r.set == 'rep9_corr_deep'
                       else np.nan for r in S.itertuples()]
    S['se_total'] = np.sqrt(S.se ** 2 + np.nan_to_num(S.se_reccorr) ** 2)
    S.to_csv(f'{OUT}/timefit_reccorr.csv', index=False)
    print(S[(S.param == 'eps0') & S.quantity.isin(['m7', 'restart', 'fer', 'D'])].round(4).to_string())


# ------------------------------------------------------------------------------------------------ placebo: repeaters as sources
def rep_rows_geo(cls):
    """repeater events as 'stacks', clusters as 'families' -> rows + straight-ray geometry (Vp 6, Vs 3.5)."""
    D = pd.read_parquet(f'{OUT}/rep/rep_event_meas.parquet')
    D['cls'] = np.where(D.depth < 10, 'shallow', np.where((D.depth > 15) & (D.depth <= 30), 'deep', 'other'))
    D = D[D.cls == cls].copy()
    D['t'] = epoch_s(pd.to_datetime(D.t))
    from sci_dvv_source_region_rep import STAS
    rows = []
    for ph in ('S', 'P'):
        cc = D['cc' + ph].clip(0, 0.999)
        ok = (cc > 0.7) & np.isfinite(D['d' + ph]) & (np.abs(D['d' + ph]) < 0.1)
        g = D[ok]
        rows.append(pd.DataFrame(dict(F=g.cl.values, stack=g.EventID.values, t=g.t.values, n=1, chan=g.sta.values + '.' + ph,
                                      sta=g.sta.values, phase=ph, y=g['d' + ph].values,
                                      se=0.0007 + 0.01 * (1 - cc[ok].values), seg=0)))
    R = pd.concat(rows, ignore_index=True)
    geo = {}
    for cl, g in D.groupby('cl'):
        la, lo, z = g.lat.mean(), g.lon.mean(), g.depth.mean()
        gr = []
        for s, (sla, slo) in STAS.items():
            v = np.array([(slo - lo) * 111.2 * np.cos(np.radians(40.3)), (sla - la) * 111.2, -z])
            L = np.linalg.norm(v)
            r = dict(sta=s)
            for P, vel in (('S', 3.5), ('P', 6.0)):
                gvec = -v / L / vel
                for i, c in enumerate('xyz'):
                    r[f'g{P}{c}'] = gvec[i]
                r[f'K{P}_sph5'] = 5 / vel
                r[f'K{P}_sph10'] = 10 / vel
            gr.append(r)
        geo[cl] = pd.DataFrame(gr)
    return R, geo


def stage_placebo_rep(a):
    rows = []
    for cls, corr_cls in (('shallow', 'deep'), ('deep', 'shallow')):
        R, geo = rep_rows_geo(cls)
        for tag, rec in (('raw', None), (f'corr_{corr_cls}', rec_dict(corr_cls))):
            res = []
            for cl, g in R.groupby('F'):
                if g['stack'].nunique() < 2:
                    continue
                g = g.reset_index(drop=True)
                if rec is not None:
                    corr = np.zeros(len(g))
                    for s_, f in rec.items():
                        m = g.sta.values == s_
                        if m.any():
                            cS, cP = f(g.t.values[m])
                            corr[m] = np.where(g.phase.values[m] == 'S', cS, cP)
                    g = g.assign(y=g.y - corr)
                if (g.phase == 'P').sum() < 6:
                    continue
                try:
                    o, _ = invert_family(g, geo[cl], 'sph5')
                except np.linalg.LinAlgError:
                    continue
                o['F'] = cl
                o = o.merge(g.groupby('stack').t.first().reset_index(), on='stack')
                o['n'] = 1
                res.append(o)
            o = pd.concat(res, ignore_index=True)
            o = o[(o.nP >= 4) & (o.nS >= 4)]
            r_ = summarize({'sph5': o}, f'placebo_{cls}_as_source_{tag}', params=['eps0', 'dz'], nboot=100)
            for r in r_:
                r['n_events'] = len(o)
            rows += r_
    S = pd.DataFrame(rows)
    S.to_csv(f'{OUT}/placebo_repeaters_as_sources.csv', index=False)
    print(S[S.quantity.isin(['m7', 'fer', 'D'])].round(4).to_string())


def stage_synth_wf(a):
    """waveform-level injection check: meas_inj_sph5m02 (stacked traces after the M7 shifted by the sph5 eps=-0.2%
    delays before the CC measurement, families 14, 23, 27) vs the real measurements; same bootstrap seeds."""
    load_geo('3d')
    rows = []
    for F in (14, 23, 27):
        for m in ('sph5', 'sph10', 'slab5SP', 'loc'):
            o = {}
            for tag, meas in (('real', MEAS), ('inj', f'{OUT}/meas_inj_sph5m02')):
                R, stk = load_rows(F, meas)
                oo, _ = invert_family(R, GEO[F], m)
                o[tag] = oo.merge(stk[['stack', 't', 'n', 'post60']], on='stack').assign(F=F)
            d = o['inj'].set_index('stack')
            r0 = o['real'].set_index('stack')
            post = d.t >= T_Q['m7']
            for p_ in [c for c in ('eps0', 'eps1', 'dz') if c in d.columns]:
                diff = (d[p_] - r0[p_])
                sc = 100 if p_.startswith('eps') else 1e3
                q = {}
                for tag in ('real', 'inj'):
                    th, _, _ = fit_time(o[tag], p_)
                    q[tag] = quantities(th)
                rows.append(dict(F=F, model=m, param=p_, stack_diff_post_mean=diff[post].mean() * sc,
                                 stack_diff_post_sd=diff[post].std() * sc, stack_diff_pre_mean=diff[~post].mean() * sc,
                                 m7_step_diff=(q['inj']['m7'] - q['real']['m7']) * sc,
                                 restart_diff=(q['inj']['restart'] - q['real']['restart']) * sc,
                                 n_post=int(post.sum())))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/synthetic_waveform_injection.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['geom', 'invert', 'timefit', 'synth', 'synth_wf', 'reccorr', 'placebo_rep', 'all'])
    ap.add_argument('--procs', type=int, default=10)
    a = ap.parse_args()
    if a.stage in ('geom', 'all'):
        stage_geom(a, '3d')
        stage_geom(a, '1d')
    if a.stage in ('invert', 'all'):
        stage_invert(a)
    if a.stage in ('timefit', 'all'):
        stage_timefit(a)
    if a.stage in ('synth', 'all'):
        stage_synth(a)
    if a.stage == 'synth_wf':
        stage_synth_wf(a)
    if a.stage in ('reccorr', 'all'):
        stage_reccorr(a)
    if a.stage in ('placebo_rep', 'all'):
        stage_placebo_rep(a)


if __name__ == '__main__':
    main()
