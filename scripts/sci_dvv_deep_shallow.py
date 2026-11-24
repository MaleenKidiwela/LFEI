#!/usr/bin/env python3
"""Deep vs shallow dv/v: direct-S (path-integrated) travel-time changes + coda in a joint layered model.

Input: data/sci/dvv_deep_shallow/meas/fam{F}.parquet (scripts/sci_dvv_deep_shallow_measure.py): per family x
channel x stack (100 isolated LFEs; 50 within 182 d after mainshocks) the direct-S / direct-P delay relative to the
all-isolated reference stack (bootstrap SE) and S-aligned coda stretching in fixed lapse windows.

Stage 1 (per family, per observable): parametric time model per channel
    y = o_b [origin/alignment term per stack, delays only] + offset(channel, instrument segment)
        + sin + cos + step_pet + step_fer + step_m7 + 60-d M7 transient + ramp from 2021-06
  EXCL rows dropped, free offsets at BREAKS2 (scripts/sci_dvv_instrument_check.py).  For delays the stack
  term o_b removes catalog/origin time and any change common to all channels; the per-channel parameters are
  RELATIVE (sum over channels = 0 per family) -> only differences between paths are used.
  Decline D = mean(2025) - mean(2021) of the steps + ramp part (same definition as sci_dvv_depth2).
  Errors: block bootstrap over stacks (B=40) and placebo = all model dates shifted back 5.25..12.25 yr fitted
  on pre-2021-06 rows (15 shifts).
Stage 2 (per quantity X = m7, D, sin, cos), all replicates (real / bootstrap / placebo):
  A  'DD'     : thS[F,c] = a_F + s_sta - dd * Td[F,c]                 (station terms free: clock- and
                site-free; dd from the double differences of travel time only)
  B  'joint'  : thS[F,c] = a_F - dd * Td[F,c] - dsh_sta * tsh_sta(h)  and  coda_sta = Ks_sta(h) dsh_sta + Kd dd
                (layer [0, h] + deep layer z >= 1 km; kernels from sci_dvv_depth2; boreholes: direct S does not
                cross the part of the layer above the sensor)
  C  'ratio'  : thS[F,c] = a_F + beta * coda_sta - dd * Td[F,c]         (empirical delay-per-coda ratio)
Clock check: per station pooled stage-1 residuals of S and P (o_b removed) -> 30-d medians -> step scan.
Synthetic recovery: injections into the real rows (delays and coda) -> full pipeline.
Usage: nice -n 10 python scripts/sci_dvv_deep_shallow.py [stage1|stage2|clock|synth|figs|all] --procs 10
Outputs data/sci/dvv_deep_shallow/ ; figures/sci_dvv_deep_shallow_*.png
"""
import argparse
import glob
import json
import os
import pickle
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
os.chdir(ROOT)
from dvv_cwire import epoch_s  # noqa: E402
from sci_dvv_instrument_check import EXCL, BREAKS2  # noqa: E402

OUT = 'data/sci/dvv_deep_shallow'
MEAS = f'{OUT}/meas'
FIG = 'figures'
DAY = 86400.0
YR = 365.25 * DAY


def _ep(x):
    return epoch_s([pd.Timestamp(x)])[0]


T_Q = {'pet': _ep('2021-12-20T20:10'), 'fer': _ep('2022-12-20T10:34'), 'm7': _ep('2024-12-05T18:44')}
T_RAMP = _ep('2021-06-01')
Y08 = _ep('2008-01-01')
PNAMES = ['sin', 'cos', 'pet', 'fer', 'm7', 'tr', 'ramp']
SHIFTS = [5.25 + 0.5 * i for i in range(15)]
NBOOT = 40
OBS = ['S', 'P', 'full', 'w3_8', 'w8_15', 'w15_25', 'j3_8', 'j8_15', 'j15_25']
CODA = ['full', 'w3_8', 'w8_15', 'w15_25', 'j3_8', 'j8_15', 'j15_25']
SENSOR = {'PB.B045': 0.2176, 'PB.B046': 0.1875, 'PB.B047': 0.1753, 'PB.B049': 0.592, 'PB.B932': 0.412,
          'PB.B933': 0.139, 'PB.B934': 0.492, 'PB.B935': 0.1524}


def sta_of(ch):
    return ch.rsplit('.', 2)[0]


def cols(t, s=0.0):
    sh = s * YR
    yr = (t - Y08) / YR
    c = [np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    for k in ['pet', 'fer', 'm7']:
        c.append((t >= T_Q[k] - sh).astype(float))
    dt = (t - (T_Q['m7'] - sh)) / DAY
    c.append(np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / 60.0), 0.0))
    c.append(np.clip((t - (T_RAMP - sh)) / YR, 0, None))
    return np.array(c).T


G21 = np.array([_ep(f'2021-{m:02d}-15') for m in range(1, 13)])
G25 = np.array([_ep(f'2025-{m:02d}-15') for m in range(1, 13)])


def decline(th, s=0.0):
    """th: (..., 7) params -> D = mean 2025 - mean 2021 of steps+ramp (nan params = not used)."""
    sh = s * YR
    C = cols(G25 - sh, s).mean(0) - cols(G21 - sh, s).mean(0)
    C[[0, 1, 5]] = 0.0
    th = np.nan_to_num(th)
    return th @ C


# ------------------------------------------------------------------------------------------------ loading
def excluded(ch, t):
    m = np.zeros(len(t), bool)
    for k, wins in EXCL.items():
        if ch == k or ch.startswith(k + '.'):
            for a, b in wins:
                m |= (t >= _ep(a)) & (t < _ep(b))
    return m


def segment(ch, t):
    br = BREAKS2.get(sta_of(ch), [])
    return np.searchsorted(np.array([_ep(b) for b in br]), t, side='right') if br else np.zeros(len(t), int)


def load(F, meas=None):
    meas = meas or MEAS
    d = pd.read_parquet(f'{meas}/fam{F}.parquet')
    meta = pd.read_csv(f'{meas}/fam{F}_channels.csv')
    loc = json.load(open(f'data/locations/fam{F}_3d.json'))
    meta['T_S'] = meta.S - loc['origin_rel_catalog_s']
    meta['T_P'] = meta.P - loc['origin_rel_catalog_s']
    meta['sta'] = meta.chan.map(sta_of)
    keep = []
    for ch, g in d.groupby('chan'):
        g = g[~excluded(ch, g.t.values)].copy()
        g['seg'] = segment(ch, g.t.values)
        keep.append(g)
    d = pd.concat(keep, ignore_index=True)
    return d, meta, loc


def obs_rows(d, obs):
    """-> DataFrame chan, stack, t, seg, y, se for one observable with QC."""
    if obs in ('S', 'P', 'Son', 'Slate'):
        y, se, cc = d[f'dt{obs}'], d[f'dt{obs}_se'], d[f'cc{obs}']
        ok = (cc > (0.5 if obs == 'P' else 0.6)) & np.isfinite(se) & (se > 0) & (np.abs(y) < 0.09) & (se < 0.02)
        se0 = 0.001
    elif obs.startswith('j'):          # joint shift+stretch (no bootstrap: SE = 2 x stretch-only bootstrap SE)
        w_ = 'w' + obs[1:]
        y, se, cc = d[f'eps2_{w_}'], 2 * d[f'eps_{w_}_se'], d[f'cc2_{w_}']
        ok = (cc > 0.15) & np.isfinite(y) & np.isfinite(se) & (se > 0) & (np.abs(y) < 0.0095) & (se < 0.012)
        se0 = 0.0003
    else:
        y, se, cc = d[f'eps_{obs}'], d[f'eps_{obs}_se'], d[f'cc_{obs}']
        ok = (cc > 0.15) & np.isfinite(y) & np.isfinite(se) & (se > 0) & (np.abs(y) < 0.0095) & (se < 0.006)
        se0 = 0.0003
    r = pd.DataFrame(dict(chan=d.chan, stack=d['stack'], t=d.t, seg=d.seg, y=y, se=np.sqrt(se ** 2 + se0 ** 2)))[ok]
    # channel QC: enough rows, robust scatter
    g = r.groupby('chan')
    n = g.y.transform('size')
    r = r[n >= 25]
    return r.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ stage 1
def usable(t_ch, seg_ch, s, nmin=3):
    """identifiability of the 7 params for one channel (rows t_ch); s = placebo shift (yr)."""
    sh = s * YR
    use = np.ones(7, bool)
    for j, k in [(2, 'pet'), (3, 'fer'), (4, 'm7')]:
        tq = T_Q[k] - sh
        ok = False
        for sg in np.unique(seg_ch):
            x = t_ch[seg_ch == sg]
            if ((x < tq) & (x > tq - YR)).sum() >= nmin and ((x >= tq) & (x < tq + YR)).sum() >= nmin:
                ok = True
        use[j] = ok
    use[5] = use[4]
    use[6] = (t_ch < T_RAMP - sh).sum() >= nmin and (t_ch > T_RAMP - sh + YR).sum() >= nmin
    if len(np.unique(np.round((t_ch - Y08) / YR % 1, 1))) < 6:
        use[0] = use[1] = False
    return use


def fit_stage1(r, origin, s=0.0, wmult=None, irls=2, use_in=None):
    """r: rows of one family & observable.  Returns dict chan -> (theta(7) with nan, formal se(7)), resid."""
    chans, ck = np.unique(r.chan.values, return_inverse=True)
    t, y = r.t.values, r.y.values
    w0 = 1.0 / r.se.values ** 2
    if wmult is not None:
        w0 = w0 * wmult
    segkey = ck * 10 + r.seg.values
    sk_u, sk = np.unique(segkey, return_inverse=True)
    C = cols(t, s)
    use = np.zeros((len(chans), 7), bool)
    for i in range(len(chans)):
        m = ck == i
        if use_in is not None:
            use[i] = use_in.get(chans[i], np.zeros(7, bool)) & usable(t[m], r.seg.values[m], s, nmin=1)
        else:
            use[i] = usable(t[m], r.seg.values[m], s)
    # columns: [o_b] + offsets + theta(c, X)
    col_th = -np.ones((len(chans), 7), int)
    nb = 0
    if origin:
        st_u, sti = np.unique(r['stack'].values, return_inverse=True)
        nb = len(st_u)
    n0 = nb + len(sk_u)
    k = n0
    for i in range(len(chans)):
        for j in range(7):
            if use[i, j]:
                col_th[i, j] = k
                k += 1
    ncol = k
    rows_i, cols_i, vals = [np.arange(len(y))], [nb + sk], [np.ones(len(y))]
    if origin:
        rows_i.append(np.arange(len(y)))
        cols_i.append(sti)
        vals.append(np.ones(len(y)))
    for j in range(7):
        cj = col_th[ck, j]
        m = cj >= 0
        rows_i.append(np.where(m)[0])
        cols_i.append(cj[m])
        vals.append(C[m, j])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))), shape=(len(y), ncol))
    # gauge constraints (origin case): sum_c theta_cX = 0 per X ; sum offsets = 0
    Cn = []
    if origin:
        for j in range(7):
            cc_ = col_th[:, j][col_th[:, j] >= 0]
            if len(cc_):
                v = np.zeros(ncol)
                v[cc_] = 1.0
                Cn.append(v)
        v = np.zeros(ncol)
        v[nb:n0] = 1.0
        Cn.append(v)
    Cn = np.array(Cn) if Cn else np.zeros((0, ncol))
    w = w0.copy()
    for it in range(irls + 1):
        N = (A.T @ A.multiply(w[:, None])).toarray()
        b = A.T @ (w * y)
        lamc = 1e3 * np.median(np.diag(N))
        N2 = N + lamc * Cn.T @ Cn + 1e-9 * np.median(np.diag(N)) * np.eye(ncol)
        x = np.linalg.solve(N2, b)
        res = y - A @ x
        if it < irls:
            z = np.abs(res) * np.sqrt(w0)
            sc = 1.4826 * np.median(z) + 1e-20
            z = z / sc
            w = w0 * np.where(z <= 2.0, 1.0, 2.0 / np.maximum(z, 1e-12))
    chi = np.sum(w * res ** 2) / max(len(y) - ncol, 1)
    try:
        cov = np.linalg.inv(N2) * max(chi, 1.0)
        fse = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        fse = np.full(ncol, np.nan)
    th = np.full((len(chans), 7), np.nan)
    se = np.full((len(chans), 7), np.nan)
    th[col_th >= 0] = x[col_th[col_th >= 0]]
    se[col_th >= 0] = fse[col_th[col_th >= 0]]
    o = None
    if origin:
        o = pd.Series(x[:nb], index=st_u)
    return dict(chans=chans, th=th, se=se, chi2r=chi, res=res, w=w, o=o, use=dict(zip(chans, use)))


def stage1_job(args):
    F, obs = args[:2]
    meas = args[2] if len(args) > 2 else None
    d, meta, loc = load(F, meas)
    r = obs_rows(d, obs)
    if len(r) < 200:
        return F, obs, None
    origin = obs in ('S', 'P', 'Son', 'Slate')
    rng = np.random.default_rng(zlib.crc32(f'{F}{obs}'.encode()))
    real = fit_stage1(r, origin)
    out = dict(chans=real['chans'], th=real['th'], se=real['se'], chi2r=real['chi2r'])
    # residual series for clock checks
    if origin:
        out['resid'] = pd.DataFrame(dict(chan=r.chan.values, t=r.t.values, res=real['res'],
                                         y_minus_o=r.y.values - real['o'].reindex(r['stack'].values).values))
    # bootstrap over stacks
    stacks = np.unique(r['stack'].values)
    pos = {s_: i for i, s_ in enumerate(stacks)}
    si = np.array([pos[s_] for s_ in r['stack'].values])
    B = []
    for b in range(NBOOT):
        cnt = np.bincount(rng.integers(0, len(stacks), len(stacks)), minlength=len(stacks)).astype(float)
        wm = cnt[si]
        keep = wm > 0
        rb = r[keep].reset_index(drop=True)
        fb = fit_stage1(rb, origin, wmult=wm[keep], irls=1, use_in=real['use'])
        thb = np.full_like(real['th'], np.nan)
        idx = np.searchsorted(real['chans'], fb['chans'])
        thb[idx] = fb['th']
        B.append(thb)
    out['boot'] = np.array(B)
    # placebo: pre-2021-06 rows, all model dates shifted back
    P = []
    pre = r[r.t < T_RAMP].reset_index(drop=True)
    for s in SHIFTS:
        thp = np.full_like(real['th'], np.nan)
        if len(pre) > 200:
            fp = fit_stage1(pre, origin, s=s, irls=1)
            idx = np.searchsorted(real['chans'], fp['chans'])
            thp[idx] = fp['th']
        P.append(thp)
    out['plac'] = np.array(P)
    return F, obs, out


def stage1(a):
    fams = sorted(int(os.path.basename(f)[3:-8]) for f in glob.glob(f'{MEAS}/fam*.parquet'))
    jobs = [(F, o) for F in fams for o in OBS]
    res = {}
    with Pool(a.procs) as p:
        for F, obs, out in p.imap_unordered(stage1_job, jobs):
            print(f'stage1 fam {F} {obs}: {"none" if out is None else len(out["chans"])}', flush=True)
            if out is not None:
                res[(F, obs)] = out
    pickle.dump(res, open(f'{OUT}/stage1.pkl', 'wb'))
    return res


# ------------------------------------------------------------------------------------------------ geometry
class Geo:
    """travel times, layer times and coda kernels."""

    def __init__(self, variant='ref', lmfp='100'):
        from sci_dvv_depth2_kernels import VModel
        self.vm = VModel(variant)
        k = np.load('data/sci/dvv_depth2/kernels.npz', allow_pickle=True)
        self.zg = k['ZG']
        self.zc = 0.5 * (self.zg[1:] + self.zg[:-1])
        self.K = {}
        for key in k.files:
            if key.startswith(f'{variant}|lfe_') and f'_{lmfp}' in key:
                self.K[key.split('|', 1)[1]] = k[key]
        self.lmfp = lmfp

    def ttime(self, z0, z1):
        if z1 <= z0:
            return 0.0
        z = np.linspace(z0, z1, 801)
        _, vs, _ = self.vm(z)
        return float(np.trapezoid(1.0 / vs, z))

    def kernel(self, sta, window='full'):
        d = SENSOR.get(sta, 0.0)
        key = f'lfe_{window}_{self.lmfp}'
        if d > 0:
            cands = [k for k in self.K if k.startswith(key + '_bh')]
            if cands:
                best = min(cands, key=lambda k: abs(float(k.split('_bh')[1]) - d))
                return self.K[best]
        return self.K[key]

    def Ks(self, sta, z0, z1, window='full'):
        K = self.kernel(sta, window)
        return float(K[(self.zc >= z0) & (self.zc < z1)].sum())

    def tsh(self, sta, z0, z1):
        """direct-S one-way time through [z0, z1] that the up-going wave crosses before the sensor."""
        d = SENSOR.get(sta, 0.0)
        return self.ttime(max(z0, d), z1)


_GEO = {}


def get_geo(variant='ref', lmfp='100'):
    if (variant, lmfp) not in _GEO:
        _GEO[(variant, lmfp)] = Geo(variant, lmfp)
    return _GEO[(variant, lmfp)]


# ------------------------------------------------------------------------------------------------ stage 2
def table(S1, obs):
    """long table (F, chan, sta, T, real theta(7), boot (B,7), plac (P,7))."""
    rows = []
    for (F, o), v in S1.items():
        if o != obs:
            continue
        meta = pd.read_csv(f'{MEAS}/fam{F}_channels.csv').set_index('chan')
        loc = json.load(open(f'data/locations/fam{F}_3d.json'))
        for i, ch in enumerate(v['chans']):
            Tcol = meta.loc[ch, 'P' if obs == 'P' else 'S'] - loc['origin_rel_catalog_s']
            rows.append(dict(F=F, chan=ch, sta=sta_of(ch), T=Tcol, th=v['th'][i], se=v['se'][i],
                             boot=v['boot'][:, i], plac=v['plac'][:, i]))
    return rows


def quantity(th, X, s=0.0):
    """th (..., 7) -> scalar quantity."""
    if X == 'D':
        return decline(th, s)
    j = PNAMES.index(X)
    return th[..., j]


def rows_to_arrays(rows, X):
    """-> DataFrame with q_real, q_boot (B), q_plac (P) per (F, chan)."""
    out = []
    for r in rows:
        qr = quantity(r['th'], X)
        if X == 'D':
            ok = np.isfinite(r['th'][6])
        else:
            ok = np.isfinite(r['th'][PNAMES.index(X)])
        if not ok:
            continue
        qb = np.array([quantity(b, X) for b in r['boot']])
        qp = np.array([quantity(p, X, s) if (np.isfinite(p[6]) if X == 'D' else np.isfinite(p[PNAMES.index(X)]))
                       else np.nan for p, s in zip(r['plac'], SHIFTS)])
        out.append(dict(F=r['F'], chan=r['chan'], sta=r['sta'], T=r['T'], q=float(qr), qb=qb, qp=qp))
    D = pd.DataFrame(out)
    if len(D):
        D['sta'] = np.asarray(D.sta, dtype=object)
        D['chan'] = np.asarray(D.chan, dtype=object)
    return D


def coda_station(S1, window, X, sta_list):
    """per-station coda quantity from stage 1 (all families, all components): real, boot, placebo."""
    R = rows_to_arrays(table(S1, window), X)
    out = {}
    for sta in sta_list:
        g = R[R.sta == sta]
        if len(g) == 0:
            continue
        qb = np.vstack(g.qb.values)
        v = np.nanvar(qb, axis=1)
        w = 1 / np.clip(v, np.nanmedian(v) * 0.2, None)
        w = np.where(np.isfinite(w), w, 0)
        q = np.sum(w * g.q.values) / w.sum()
        qbs = np.nansum(w[:, None] * np.nan_to_num(qb), 0) / w.sum()
        qp = np.vstack(g.qp.values)
        okp = np.isfinite(qp)
        wp = w[:, None] * okp
        qps = np.where(wp.sum(0) > 0, np.nansum(wp * np.nan_to_num(qp), 0) / np.maximum(wp.sum(0), 1e-30), np.nan)
        out[sta] = dict(q=q, qb=qbs, qp=qps, n=len(g))
    return out


def wls(A, y, w):
    if len(y) < A.shape[1] + 2:
        return np.full(A.shape[1], np.nan)
    W = np.sqrt(w)[:, None]
    x, *_ = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)
    return x


def stage2_fit(df, qcol, model, geo=None, coda=None, h=0.3, zdeep=1.0, wS=None, wC=None):
    """df rows (F, sta, T, Td, q) for direct S; returns dict (dd [fraction], extra)."""
    y = df[qcol].values
    ok = np.isfinite(y) & np.isfinite(wS)
    d = df[ok].copy()
    d['sta'] = np.asarray(d.sta, dtype=object)
    y = y[ok]
    w = wS[ok]
    fams = np.unique(d.F.values)
    stas = np.unique(d.sta.values)
    fi = np.searchsorted(fams, d.F.values)
    si = np.searchsorted(stas, d.sta.values)
    AF = np.zeros((len(d), len(fams)))
    AF[np.arange(len(d)), fi] = 1
    Td = d.Td.values
    if model == 'DD':
        AS = np.zeros((len(d), len(stas)))
        AS[np.arange(len(d)), si] = 1
        A = np.c_[AF, AS[:, 1:], -Td]
        x = wls(A, y, w)
        return dict(dd=x[-1])
    if model == 'ratio':
        cs = np.array([coda[s_] if s_ in coda else np.nan for s_ in d.sta.values])
        m = np.isfinite(cs)
        A = np.c_[AF[m], cs[m], -Td[m]]
        x = wls(A, y[m], w[m])
        return dict(dd=x[-1], beta=x[-2])
    if model in ('ratio2', 'ratio_bh'):
        # class-specific delay-per-coda ratios (surface / borehole); 'ratio_bh' = boreholes only
        cs = np.array([coda[s_] if s_ in coda else np.nan for s_ in d.sta.values])
        bh = np.array([SENSOR.get(s_, 0.0) > 0 for s_ in d.sta.values])
        m = np.isfinite(cs) & (bh if model == 'ratio_bh' else True)
        fk = np.unique(fi[m])
        AFm = AF[m][:, fk]
        if model == 'ratio2':
            A = np.c_[AFm, cs[m] * ~bh[m], cs[m] * bh[m], -Td[m]]
            x = wls(A, y[m], w[m])
            return dict(dd=x[-1], beta=x[-2], beta_surf=x[-3])
        A = np.c_[AFm, cs[m], -Td[m]]
        x = wls(A, y[m], w[m])
        return dict(dd=x[-1], beta=x[-2])
    if model == 'joint':
        # unknowns: a_F, dd, dsh_sta (stations with coda); delays [s], coda [fraction]
        cst = [s_ for s_ in stas if s_ in coda and np.isfinite(coda[s_])]
        m = np.isin(d.sta.values, cst)
        d2, y2, w2, fi2 = d[m], y[m], w[m], fi[m]
        ns = len(cst)
        cidx = {s_: i for i, s_ in enumerate(cst)}
        n = len(fams) + 1 + ns
        A1 = np.zeros((len(d2), n))
        A1[np.arange(len(d2)), fi2] = 1
        A1[:, len(fams)] = -d2.Td.values
        tsh = np.array([geo.tsh(s_, 0.0, h) for s_ in cst])
        for k_, s_ in enumerate(d2.sta.values):
            A1[k_, len(fams) + 1 + cidx[s_]] = -tsh[cidx[s_]]
        A2 = np.zeros((ns, n))
        y2c = np.zeros(ns)
        w2c = np.zeros(ns)
        for s_, i in cidx.items():
            A2[i, len(fams)] = geo.Ks(s_, zdeep, 1e9)
            A2[i, len(fams) + 1 + i] = geo.Ks(s_, 0.0, h)
            y2c[i] = coda[s_]
            w2c[i] = wC[s_]
        A = np.r_[A1, A2]
        yy, ww = np.r_[y2, y2c], np.r_[w2, w2c]
        x = wls(A, yy, ww)
        r_ = yy - A @ x
        chi_s = float(np.sum(ww[:len(y2)] * r_[:len(y2)] ** 2) / max(len(y2) - len(fams) - 1, 1))
        chi_c = float(np.sum(ww[len(y2):] * r_[len(y2):] ** 2) / max(ns, 1))
        return dict(dd=x[len(fams)], dsh=dict(zip(cst, x[len(fams) + 1:])), chi_s=chi_s, chi_c=chi_c)
    raise ValueError(model)


def station_S_table(S1, X, geo):
    R = rows_to_arrays(table(S1, 'S'), X)
    R['Td'] = [T - geo.tsh(s_, 0.0, 1.0) for T, s_ in zip(R['T'], R.sta)]
    return R


def stage2(a, S1=None, tag='', variants=None, verbose=True):
    S1 = S1 if S1 is not None else pickle.load(open(f'{OUT}/stage1.pkl', 'rb'))
    variants = variants or [('ref', '100'), ('ref', '20'), ('soft', '100'), ('thin', '100'), ('none', '100')]
    HS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    REPS = {}
    out = []
    station_rows = []
    for X in ['m7', 'D', 'sin', 'cos', 'tr']:
        geo0 = get_geo('ref', '100')
        R = station_S_table(S1, X, geo0)
        if len(R) == 0:
            continue
        qb = np.vstack(R.qb.values)
        qp = np.vstack(R.qp.values)
        vb = np.nanvar(qb, axis=1)
        vb = np.clip(vb, np.nanmedian(vb) * 0.2, None)
        wS = 1 / vb
        # station-level direct-S summary (relative delays; family offsets removed by DD-style fit later)
        stas = sorted(R.sta.unique())
        coda_src = {}
        for win in CODA:
            coda_src[win] = coda_station(S1, win, X, stas)

        def run(model, qv, wv, coda_q, geo, h, wC, full=False):
            df = R.assign(qq=qv)
            try:
                o_ = stage2_fit(df, 'qq', model, geo=geo, coda=coda_q, h=h, wS=wv, wC=wC)
                return o_ if full else o_['dd']
            except Exception:                    # noqa: BLE001
                return {} if full else np.nan

        cfull = coda_src['full']
        cq = {s_: v['q'] for s_, v in cfull.items()}
        cvar = {s_: np.nanvar(v['qb']) for s_, v in cfull.items()}
        cplac = {s_: np.nanstd(v['qp']) for s_, v in cfull.items()}
        wC = {s_: 1 / max(cvar[s_], (0.5 * cplac[s_]) ** 2 if np.isfinite(cplac[s_]) else 0, 1e-12) for s_ in cq}
        configs = [('DD', None, None), ('ratio', None, None), ('ratio2', None, None), ('ratio_bh', None, None)] + \
                  [('joint', v, h) for v in variants for h in HS]
        for model, var, h in configs:
            geo = geo0 if var is None else get_geo(*var)
            full_ = run(model, R.q.values, wS, cq, geo, h, wC, full=True)
            dd = full_.get('dd', np.nan)
            extra = {k_: v_ for k_, v_ in full_.items() if k_ in ('beta', 'beta_surf', 'chi_s', 'chi_c')}
            if model.startswith('ratio'):
                for kb in ('beta', 'beta_surf'):
                    if kb not in full_:
                        continue
                    bb = np.array([run(model, qb[:, b], wS, {s_: v['qb'][b] for s_, v in cfull.items()}, geo, h, wC,
                                       full=True).get(kb, np.nan) for b in range(qb.shape[1])])
                    bp = np.array([run(model, qp[:, p], wS, {s_: v['qp'][p] for s_, v in cfull.items()}, geo, h, wC,
                                       full=True).get(kb, np.nan) for p in range(qp.shape[1])])
                    extra[kb + '_se_boot'] = float(np.nanstd(bb))
                    extra[kb + '_se_plac'] = float(np.nanstd(bp))
            ddb = np.array([run(model, qb[:, b], wS, {s_: v['qb'][b] for s_, v in cfull.items()}, geo, h, wC)
                            for b in range(qb.shape[1])])
            ddp = np.array([run(model, qp[:, p], wS, {s_: v['qp'][p] for s_, v in cfull.items()}, geo, h, wC)
                            for p in range(qp.shape[1])])
            REPS[(X, model, '-' if var is None else f'{var[0]}/l{var[1]}', h)] = dict(boot=ddb, plac=ddp, real=dd)
            se_b, se_p = np.nanstd(ddb), np.nanstd(ddp)
            mad_p = 1.4826 * np.nanmedian(np.abs(ddp - np.nanmedian(ddp)))
            se = np.nanmax([se_b, se_p])
            out.append(dict(X=X, model=model, variant='-' if var is None else f'{var[0]}/l{var[1]}', h_km=h,
                            dd_pct=dd * 100, se_boot_pct=se_b * 100, se_plac_pct=se_p * 100,
                            plac_mean_pct=np.nanmean(ddp) * 100, se_plac_mad_pct=mad_p * 100, se_pct=se * 100, z=dd / se if se > 0 else np.nan,
                            n_plac=int(np.isfinite(ddp).sum()), n_rows=len(R), **extra))
            if verbose and (model != 'joint' or (var == ('ref', '100'))):
                o = out[-1]
                print(f"{X:4s} {model:6s} {o['variant']:10s} h={h} dd {o['dd_pct']:+.4f}% se_b {o['se_boot_pct']:.4f} "
                      f"se_p {o['se_plac_pct']:.4f} (pl mean {o['plac_mean_pct']:+.4f})", flush=True)
        # station table: family-offset-removed direct S per station (DD model with dd = 0 -> station terms)
        y = R.q.values
        fams = np.unique(R.F)
        AF = (R.F.values[:, None] == fams[None, :]).astype(float)
        AS = (np.asarray(R.sta, dtype=object)[:, None] == np.array(stas)[None, :]).astype(float)
        A = np.c_[AF, AS]
        BAL = ['PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B934', 'NC.KMR', 'NC.KMPB']
        Cn = np.r_[np.zeros(len(fams)), np.isin(stas, BAL).astype(float)]   # gauge: mean of long-record stations

        def st_terms(yv):
            m_ = np.isfinite(yv) & np.isfinite(wS)
            W = np.sqrt(wS[m_])[:, None]
            has = A[m_][:, len(fams):].sum(0) > 0
            Cm = Cn * np.r_[np.zeros(len(fams)), has]
            AA = np.r_[A[m_] * W, 1e3 * Cm[None, :]]
            x, *_ = np.linalg.lstsq(AA, np.r_[yv[m_] * W[:, 0], 0.0], rcond=None)
            x = x[len(fams):]
            return np.where(has, x, np.nan)
        st_r = st_terms(y)
        st_b = np.array([st_terms(qb[:, b]) for b in range(qb.shape[1])])
        st_p = np.array([st_terms(qp[:, p]) for p in range(qp.shape[1])])
        Tm = R.groupby('sta')['T'].mean()
        for i, s_ in enumerate(stas):
            cs = cfull.get(s_, {})
            row = dict(X=X, sta=s_, sensor_km=SENSOR.get(s_, 0.0), T_S=Tm[s_], n=int((R.sta == s_).sum()),
                       dS_ms=st_r[i] * 1e3, dS_se_boot_ms=np.nanstd(st_b[:, i]) * 1e3,
                       dS_se_plac_ms=np.nanstd(st_p[:, i]) * 1e3,
                       coda_full_pct=cs.get('q', np.nan) * 100,
                       coda_full_se_boot_pct=np.nanstd(cs['qb']) * 100 if cs else np.nan,
                       coda_full_se_plac_pct=np.nanstd(cs['qp']) * 100 if cs else np.nan)
            for win in CODA[1:]:
                c2 = coda_src[win].get(s_, {})
                row[f'coda_{win}_pct'] = c2.get('q', np.nan) * 100
                row[f'coda_{win}_se_pct'] = (max(np.nanstd(c2['qb']), np.nanstd(c2['qp'])) * 100) if c2 else np.nan
            station_rows.append(row)
    O = pd.DataFrame(out)
    O.to_csv(f'{OUT}/stage2_deep{tag}.csv', index=False)
    pickle.dump(REPS, open(f'{OUT}/stage2_replicates{tag}.pkl', 'wb'))
    ST = pd.DataFrame(station_rows)
    ST.to_csv(f'{OUT}/station_directS_coda{tag}.csv', index=False)
    return O, ST


def station_terms(S1, obs, X):
    """relative station delay terms (family offsets removed; gauge = mean of long-record stations)."""
    S1b = {(F, 'S'): v for (F, o), v in S1.items() if o == obs}
    R = rows_to_arrays(table(S1b, 'S'), X)
    qb, qp = np.vstack(R.qb.values), np.vstack(R.qp.values)
    vb = np.nanvar(qb, axis=1)
    wS = 1 / np.clip(vb, np.nanmedian(vb) * 0.2, None)
    fams = np.unique(R.F)
    stas = sorted(R.sta.unique())
    A = np.c_[(R.F.values[:, None] == fams[None, :]).astype(float),
              (np.asarray(R.sta, dtype=object)[:, None] == np.array(stas)[None, :]).astype(float)]
    BAL = ['PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B934', 'NC.KMR', 'NC.KMPB']
    Cn = np.r_[np.zeros(len(fams)), np.isin(stas, BAL).astype(float)]

    def st(yv):
        m_ = np.isfinite(yv) & np.isfinite(wS)
        W = np.sqrt(wS[m_])[:, None]
        has = A[m_][:, len(fams):].sum(0) > 0
        Cm = Cn * np.r_[np.zeros(len(fams)), has]
        x, *_ = np.linalg.lstsq(np.r_[A[m_] * W, 1e3 * Cm[None, :]], np.r_[yv[m_] * W[:, 0], 0.0], rcond=None)
        return np.where(has, x[len(fams):], np.nan)
    r0 = st(R.q.values.astype(float))
    rb = np.array([st(qb[:, b]) for b in range(qb.shape[1])])
    rp = np.array([st(qp[:, p]) for p in range(qp.shape[1])])
    return pd.DataFrame(dict(sta=stas, obs=obs, X=X, d_ms=r0 * 1e3, se_boot_ms=np.nanstd(rb, 0) * 1e3,
                             se_plac_ms=np.nanstd(rp, 0) * 1e3, n=[int((R.sta == s_).sum()) for s_ in stas]))


def swin(a):
    """direct-S onset vs late window (borehole surface-reflection test)."""
    meas = f'{OUT}/meas_swin'
    fams = sorted(int(os.path.basename(f)[3:-8]) for f in glob.glob(f'{meas}/fam*.parquet'))
    jobs = [(F, o, meas) for F in fams for o in ('S', 'Son', 'Slate')]
    S1 = {}
    with Pool(a.procs) as p:
        for F, obs, out in p.imap_unordered(stage1_job, jobs):
            if out is not None:
                S1[(F, obs)] = out
    pickle.dump(S1, open(f'{OUT}/stage1_swin.pkl', 'wb'))
    rows = [station_terms(S1, o, X) for o in ('S', 'Son', 'Slate') for X in ('D', 'm7', 'sin')]
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(f'{OUT}/swin_station_terms.csv', index=False)
    P = T.pivot_table(index=['X', 'sta'], columns='obs', values=['d_ms', 'se_boot_ms', 'se_plac_ms'])
    print(P.round(2).to_string())
    return T


# ------------------------------------------------------------------------------------------------ borehole tension
def tension(a=None):
    """Station-level test: relative direct-S delay vs coda change under layer hypotheses (body-wave kernel).
    pred delay_c = -coda_c / Ks_c(layer) * tS_c(layer part below the sensor) ; free common offset per class;
    chi2 with honest SE = max(bootstrap, placebo) per station."""
    ST = pd.read_csv(f'{OUT}/station_directS_coda.csv')
    geo = get_geo('ref', '100')
    rows, fits = [], []
    GOOD_BH = ['PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B934']
    GOOD_SF = ['NC.KMR', 'NC.KMPB', 'BK.RBOW', 'BK.HAYF', 'BK.BRIC']
    for X in ['D', 'm7']:
        S = ST[(ST.X == X) & ST.sta.isin(GOOD_BH + GOOD_SF)].copy()
        S['se_d'] = np.fmax(S.dS_se_boot_ms, S.dS_se_plac_ms.fillna(0))
        S['se_c'] = np.fmax(S.coda_full_se_boot_pct, S.coda_full_se_plac_pct.fillna(0))
        for r in S.itertuples():
            d = SENSOR.get(r.sta, 0.0)
            c = r.coda_full_pct / 100
            hyp = {'above_sensor': 0.0}
            for z0, z1, nm in [(0, 0.1, '0-0.1km'), (0, 0.3, '0-0.3km'), (0, 1.0, '0-1km'), (d, d + 0.3, 'sensor-+0.3km'),
                               (d, d + 1.0, 'sensor-+1km'), (1.0, 1e9, 'deep>1km')]:
                ks = geo.Ks(r.sta, z0, z1)
                tt = geo.tsh(r.sta, z0, min(z1, 30.0)) if z1 < 1e8 else r.T_S - geo.tsh(r.sta, 0, 1.0)
                hyp[nm] = -c / ks * tt * 1e3 if ks > 0 else np.nan
            rows.append(dict(X=X, sta=r.sta, sensor_km=d, T_S=r.T_S, coda_pct=r.coda_full_pct, coda_se_pct=r.se_c,
                             dS_ms=r.dS_ms, dS_se_ms=r.se_d, **{f'pred_{k}_ms': v for k, v in hyp.items()}))
    T = pd.DataFrame(rows)
    for X in ['D', 'm7']:
        for cls, sel in [('borehole', T.sensor_km > 0), ('surface', T.sensor_km == 0), ('all', T.sensor_km >= 0)]:
            g = T[(T.X == X) & sel]
            if len(g) < 3:
                continue
            w = 1 / g.dS_se_ms ** 2
            # empirical ratio fit: dS = a + beta * coda[%]  (ms per %)
            A = np.c_[np.ones(len(g)), g.coda_pct]
            W = np.sqrt(w.values)[:, None]
            x, *_ = np.linalg.lstsq(A * W, g.dS_ms.values * W[:, 0], rcond=None)
            res = g.dS_ms.values - A @ x
            chi = np.sum(w * res ** 2) / max(len(g) - 2, 1)
            cov = np.linalg.inv((A * W).T @ (A * W)) * max(chi, 1)
            fits.append(dict(X=X, cls=cls, hyp='free_ratio', n=len(g), chi2r=chi, offset_ms=x[0],
                             beta_s=-x[1] * 0.1, beta_se_s=np.sqrt(cov[1, 1]) * 0.1))
            for hcol in [c_ for c_ in g.columns if c_.startswith('pred_')]:
                p = g[hcol].values
                if not np.all(np.isfinite(p)):
                    continue
                # prediction error from the coda SE
                pe = np.abs(p) * (g.coda_se_pct / np.abs(g.coda_pct).clip(1e-3)).values
                ww = 1 / (g.dS_se_ms.values ** 2 + pe ** 2)
                a0 = np.sum(ww * (g.dS_ms.values - p)) / ww.sum()
                chi = np.sum(ww * (g.dS_ms.values - p - a0) ** 2) / max(len(g) - 1, 1)
                fits.append(dict(X=X, cls=cls, hyp=hcol[5:-3], n=len(g), chi2r=chi, offset_ms=a0))
    T.to_csv(f'{OUT}/tension_station_predictions.csv', index=False)
    Fz = pd.DataFrame(fits)
    Fz.to_csv(f'{OUT}/tension_hypothesis_fits.csv', index=False)
    print(T.round(3).to_string())
    print(Fz.round(3).to_string())
    return T, Fz


# ------------------------------------------------------------------------------------------------ clock check
def clock(a):
    S1 = pickle.load(open(f'{OUT}/stage1.pkl', 'rb'))
    rows = []
    for (F, o), v in S1.items():
        if o in ('S', 'P') and 'resid' in v:
            r = v['resid'].copy()
            r['F'] = F
            r['obs'] = o
            rows.append(r)
    R = pd.concat(rows, ignore_index=True)
    R['sta'] = R.chan.map(sta_of)
    # per (F, chan): demean y_minus_o (offset) -> pooled per station; monthly medians
    R['z'] = R.y_minus_o - R.groupby(['F', 'chan', 'obs']).y_minus_o.transform('median')
    R['mon'] = pd.to_datetime(R.t, unit='s').dt.to_period('M').dt.to_timestamp()
    M = R.groupby(['sta', 'obs', 'mon']).z.agg(['median', 'size', lambda x: 1.2533 * x.std() / np.sqrt(len(x))])
    M.columns = ['med', 'n', 'se']
    M = M.reset_index()
    M.to_csv(f'{OUT}/clock_monthly.csv', index=False)
    out = []
    for (sta, obs), g in M.groupby(['sta', 'obs']):
        g = g[g.n >= 5].sort_values('mon')
        if len(g) < 12:
            continue
        tt = g.mon.values
        y = g.med.values
        se = np.clip(g.se.values, 1e-4, None)
        best = None
        for k in range(3, len(g) - 3):
            a_, b_ = slice(max(0, k - 6), k), slice(k, min(len(g), k + 6))
            wa, wb = 1 / se[a_] ** 2, 1 / se[b_] ** 2
            ma, mb = np.sum(wa * y[a_]) / wa.sum(), np.sum(wb * y[b_]) / wb.sum()
            sa = np.sqrt(np.sum(wa * (y[a_] - ma) ** 2) / wa.sum() / max(len(wa) - 1, 1) + 1 / wa.sum())
            sb = np.sqrt(np.sum(wb * (y[b_] - mb) ** 2) / wb.sum() / max(len(wb) - 1, 1) + 1 / wb.sum())
            st = mb - ma
            z = st / np.hypot(sa, sb)
            if best is None or abs(z) > abs(best[2]):
                best = (tt[k], st, z)
        # dispersion of monthly medians
        out.append(dict(sta=sta, obs=obs, n_months=len(g), rms_ms=np.std(y) * 1e3,
                        max_step_date=str(pd.Timestamp(best[0]).date()), max_step_ms=best[1] * 1e3, max_step_z=best[2],
                        first=str(pd.Timestamp(tt[0]).date()), last=str(pd.Timestamp(tt[-1]).date())))
    C = pd.DataFrame(out)
    C['absz'] = C.max_step_z.abs()
    C = C.sort_values(['obs', 'absz'], ascending=[True, False]).drop(columns='absz')
    C.to_csv(f'{OUT}/clock_steps.csv', index=False)
    print(C.to_string())
    return C


# ------------------------------------------------------------------------------------------------ synthetics
def synth_job(args):
    F, scen, var = args
    d, meta, loc = load(F)
    geo = get_geo(*var)
    out = {}
    for obs in ['S', 'full']:
        r = obs_rows(d, obs)
        if len(r) < 200:
            continue
        sta = r.chan.map(sta_of).values
        C = cols(r.t.values)
        g_m7 = C[:, 4]
        g_ramp = C[:, 6]
        add = np.zeros(len(r))
        for X, g in (('m7', g_m7), ('ramp', g_ramp)):
            dd = scen['dd'][X]
            dsh = np.array([scen['dsh'][X].get(s_, 0.0) for s_ in sta])
            h = scen['h']
            if obs == 'S':
                Tm = meta.set_index('chan').loc[r.chan.values, 'T_S'].values
                tsh_ = np.array([geo.tsh(s_, 0.0, h) for s_ in sta])
                td = Tm - np.array([geo.tsh(s_, 0.0, 1.0) for s_ in sta])
                add += -(dd * td + dsh * tsh_) * g          # delay [s]
            else:
                ks = np.array([geo.Ks(s_, 0.0, h) for s_ in sta]) * scen.get('ks_mult', 1.0)
                kd = np.array([geo.Ks(s_, 1.0, 1e9) for s_ in sta])
                add += (ks * dsh + kd * dd) * g
        r = r.assign(y=r.y + add)
        origin = obs == 'S'
        real = fit_stage1(r, origin)
        rng = np.random.default_rng(F)
        se_ = np.nan_to_num(real['se'], nan=0.0)
        out[obs] = dict(chans=real['chans'], th=real['th'], se=real['se'], chi2r=real['chi2r'],
                        boot=real['th'][None] + se_[None] * rng.standard_normal((6,) + se_.shape),
                        plac=np.repeat(real['th'][None], 2, 0) * np.nan)
    return F, out


def synth(a):
    """inject scenarios (true layered models) into the real rows of all families; rerun stages 1-2 (real only)."""
    S1real = pickle.load(open(f'{OUT}/stage1.pkl', 'rb'))
    ST = pd.read_csv(f'{OUT}/station_directS_coda.csv')
    stas = sorted(ST.sta.unique())
    var = ('ref', '100')
    geo = get_geo(*var)
    scen = [dict(name='baseline_no_injection', h=0.3, dd={'m7': 0.0, 'ramp': 0.0}, dsh={'m7': {}, 'ramp': {}})]
    # 1 deep only
    scen.append(dict(name='deep_only', h=0.3, dd={'m7': -0.0004, 'ramp': -0.0002}, dsh={'m7': {}, 'ramp': {}}))
    # shallow scenarios with REALISTIC amplitudes: per station, the layer change that reproduces the OBSERVED coda
    # M7 step and decline (ramp part = (D - m7)/3.96 yr) through the kernel (x ks_mult)
    Cd = cols(G25).mean(0)[6] - cols(G21).mean(0)[6]
    obs_c = {X: ST[ST.X == X].set_index('sta').coda_full_pct.dropna() / 100 for X in ('m7', 'D')}

    def dsh_obs(h, ksm=1.0):
        m7 = {s_: obs_c['m7'][s_] / (geo.Ks(s_, 0, h) * ksm) for s_ in obs_c['m7'].index}
        rp = {s_: (obs_c['D'][s_] - obs_c['m7'].get(s_, 0.0)) / Cd / (geo.Ks(s_, 0, h) * ksm)
              for s_ in obs_c['D'].index}
        return {'m7': m7, 'ramp': rp}
    for h, ksm in [(0.1, 1.0), (0.3, 1.0), (1.0, 1.0), (0.3, 3.0), (0.1, 3.0)]:
        tag = f'h{h}' + ('' if ksm == 1 else f'_Ks{ksm:.0f}x')
        scen.append(dict(name=f'shallow_obs_{tag}', h=h, ks_mult=ksm, dd={'m7': 0.0, 'ramp': 0.0}, dsh=dsh_obs(h, ksm)))
    scen.append(dict(name='mixed_obs_h0.3_Ks3x', h=0.3, ks_mult=3.0, dd={'m7': -0.0004, 'ramp': -0.0002},
                     dsh=dsh_obs(0.3, 3.0)))
    res = []
    fams = sorted({F for F, o in S1real})
    for sc in scen:
        jobs = [(F, sc, var) for F in fams]
        S1 = {}
        with Pool(a.procs) as p:
            for F, out in p.imap_unordered(synth_job, jobs):
                for obs, v in out.items():
                    S1[(F, obs)] = v
        for X in ['m7', 'D']:
            R = station_S_table(S1, X, geo)
            vb = np.nanvar(np.vstack(R.qb.values), axis=1)
            wS = 1 / np.clip(vb, np.nanmedian(vb) * 0.2, None)
            cf = coda_station(S1, 'full', X, sorted(R.sta.unique()))
            cq = {s_: v['q'] for s_, v in cf.items()}
            wC = {s_: 1 / max(np.nanvar(v['qb']), 1e-10) for s_, v in cf.items()}
            true_dd = sc['dd']['m7'] if X == 'm7' else sc['dd']['ramp'] * Cd + sc['dd']['m7']
            for model, h in [('DD', None), ('ratio', None), ('ratio2', None), ('joint', 0.1), ('joint', 0.3), ('joint', 1.0)]:
                try:
                    dd = stage2_fit(R.assign(qq=R.q), 'qq', model, geo=geo, coda=cq, h=h or 0.3, wS=wS, wC=wC)['dd']
                except Exception:                    # noqa: BLE001
                    dd = np.nan
                base = next((r_['rec_dd_pct'] for r_ in res if r_['scenario'] == 'baseline_no_injection'
                             and r_['X'] == X and r_['model'] == model and r_['h_assumed'] == h), np.nan)
                res.append(dict(scenario=sc['name'], X=X, model=model, h_assumed=h, h_true=sc['h'],
                                ks_mult=sc.get('ks_mult', 1.0), true_dd_pct=true_dd * 100, rec_dd_pct=dd * 100,
                                rec_minus_baseline_pct=(dd * 100 - base) if sc['name'] != 'baseline_no_injection' else 0.0))
                print(res[-1], flush=True)
    Z = pd.DataFrame(res)
    Z.to_csv(f'{OUT}/synthetic_recovery.csv', index=False)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['stage1', 'stage2', 'tension', 'clock', 'synth', 'swin', 'all'])
    ap.add_argument('--procs', type=int, default=10)
    a = ap.parse_args()
    if a.stage in ('stage1', 'all'):
        stage1(a)
    if a.stage in ('stage2', 'all'):
        stage2(a)
    if a.stage in ('tension', 'all'):
        tension(a)
    if a.stage in ('clock', 'all'):
        clock(a)
    if a.stage in ('synth', 'all'):
        synth(a)
    if a.stage in ('swin', 'all'):
        swin(a)


if __name__ == '__main__':
    main()
