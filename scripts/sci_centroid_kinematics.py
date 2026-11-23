#!/usr/bin/env python3
"""Centroid kinematics of LFE families from fixed-station (core) bin relocations.

Tasks
  1. Common-mode (CM) removal: joint LSQ  d_fb = a_f + sum_k K_bk C_k  (K = fraction of the
     bin's 150 events in 60-day cell k; cell edges anchored on the three quakes so that no
     cell straddles a quake; weak first-difference smoothing except across quakes).
     Jackknife over families. Differential track of family f = d_f - K C^(-f) (leave-f-out CM).
     Steps at the 3 quakes + ramp since 2020 fitted to CM cells (2018+) and to differential
     tracks; spatial-gradient tests of differential steps with permutation nulls.
  2. Apparent shift predicted from a uniform velocity change: half-space straight rays with
     the actual core station/phase set, and the same with the 3-D gradients + observed times.
  3. Availability check: M7 step in core vs m7set vs all-station bins (fams 23, 8, 1).
  4. Differential migration since 2008 (block-bootstrap rates, fault-frame projection).
  Reviewer additions: placebo dates / windows, leave-one-station-out re-inversion from the
  per-bin obs files, per-station direct-phase delay changes at the M7, mean_cc reweighting.
Outputs: data/sci/centroids/*, figures/sci_centroid_*.png
"""
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import h5py  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

os.chdir('/home/seismic/LFE')
OUT = 'data/sci/centroids'
os.makedirs(OUT, exist_ok=True)
COMP = ['dx', 'dy', 'dz']
SDC = ['sd_x', 'sd_y', 'sd_z']
SEC_YR = 365.2425 * 86400
RNG = np.random.default_rng(42)
FLOOR = 0.003          # km, added to bootstrap sd
W_LOC = 0.75           # yr half-window of local step estimator
T2020 = None


def dyear(x):
    a = np.asarray(pd.to_datetime(np.atleast_1d(x)), dtype='datetime64[ns]').astype('int64')
    return 1970 + a / 1e9 / SEC_YR


QUAKES = {'Q2021': '2021-12-20T20:10', 'Q2022': '2022-12-20T10:34', 'M7_2024': '2024-12-05T18:44'}
TQ = {k: float(dyear(v)[0]) for k, v in QUAKES.items()}
QN = list(QUAKES)
T2020 = float(dyear('2020-01-01')[0])
EPI_M7 = (40.37, -125.0)
STRIKE, DIP = 290.0, 50.0
LAT0, LON0 = 40.06, -123.63
KM_DEG = 111.195


def en(lat, lon):
    return (np.asarray(lon) - LON0) * KM_DEG * np.cos(np.radians(LAT0)), (np.asarray(lat) - LAT0) * KM_DEG


# fault frame unit vectors in (E, N, down)
u_strike = np.array([np.sin(np.radians(STRIKE)), np.cos(np.radians(STRIKE)), 0.0])
dipaz = np.radians(STRIKE + 90)
u_dip = np.array([np.cos(np.radians(DIP)) * np.sin(dipaz), np.cos(np.radians(DIP)) * np.cos(dipaz),
                  np.sin(np.radians(DIP))])
u_norm = np.cross(u_strike, u_dip)

# ----------------------------------------------------------------------------- loading
EVT = {}


def event_times(F):
    if F not in EVT:
        with h5py.File(f'data/waveforms/fam{F}.h5', 'r') as h:
            t = pd.to_datetime(h['events/time_utc'][:].astype(str))
        EVT[F] = np.sort(dyear(t))
    return EVT[F]


def read_bins(F, suffix):
    p = f'data/relocations/fam{F}_bins_3d{suffix}.csv'
    try:
        b = pd.read_csv(p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    return b if len(b) >= 5 else None


FAMS = []
FINFO = {}
for F in range(1, 28):
    b = read_bins(F, '_core')
    if b is None:
        continue
    j = json.load(open(f'data/locations/fam{F}_3d.json'))
    e, n = en(j['lat'], j['lon'])
    ex, ny = en(EPI_M7[0], EPI_M7[1])
    FAMS.append(F)
    FINFO[F] = dict(lat=j['lat'], lon=j['lon'], depth=j['depth_km'], E=float(e), N=float(n),
                    dist_m7=float(np.hypot(e - ex, n - ny)))
for F in FAMS:
    p = np.array([FINFO[F]['E'], FINFO[F]['N'], FINFO[F]['depth']])
    FINFO[F]['along_strike'] = float(p @ u_strike)
    FINFO[F]['down_dip'] = float(p @ u_dip)
nF = len(FAMS)
FIDX = {F: i for i, F in enumerate(FAMS)}


def make_cells(t0=2008.35, t1=2026.55, step=60 / 365.2425, extra=()):
    brkt = np.array(sorted(list(TQ.values()) + list(extra)))
    anchors = [t0] + list(brkt) + [t1]
    edges = [t0]
    for a, b in zip(anchors[:-1], anchors[1:]):
        n = max(1, int(round((b - a) / step)))
        edges += list(np.linspace(a, b, n + 1)[1:])
    edges = np.array(edges)
    brk = np.array([np.min(np.abs(edges[1:-1][k] - brkt)) < 1e-6
                    for k in range(len(edges) - 2)])
    return edges, brk


EDGES, BREAKS = make_cells()
TC = 0.5 * (EDGES[1:] + EDGES[:-1])
nK = len(TC)


def build_table(bins_by_F):
    """Global bin table + per-bin event arrays + kernel."""
    rows, EV = [], []
    for F, b in bins_by_F.items():
        te = event_times(F)
        for r in b.itertuples(index=False):
            ev = te[int(r.bin) * 150:(int(r.bin) + 1) * 150]
            rows.append(dict(F=F, fi=FIDX[F], bin=int(r.bin), tmid=float(dyear(r.t_mid)[0]), tm=ev.mean(),
                             dx=r.dx, dy=r.dy, dz=r.dz, sd_x=r.sd_x, sd_y=r.sd_y, sd_z=r.sd_z,
                             mean_cc=r.mean_cc, n_sta=r.n_sta, span=ev.max() - ev.min()))
            EV.append(ev)
    T = pd.DataFrame(rows)
    L = max(len(e) for e in EV)
    EVm = np.full((len(EV), L), np.nan)
    for i, e in enumerate(EV):
        EVm[i, :len(e)] = e
    K = np.zeros((len(T), nK))
    for i, e in enumerate(EV):
        h, _ = np.histogram(e, EDGES)
        K[i] = h / len(e)
    T['ramp'] = np.nanmean(np.clip(EVm - T2020, 0, None), axis=1)
    for q in QN:
        T['fa_' + q] = np.nanmean(EVm > TQ[q], axis=1)
    return T, EVm, K


def frac_after(EVm, date):
    v = ~np.isnan(EVm)
    return (EVm > date).sum(1) / v.sum(1)


# ----------------------------------------------------------------------------- joint CM fit
def joint_fit(D, SIG, fi, K, use, lam=1 / 0.03 ** 2, huber=2.0, n_iter=5, breaks=None):
    breaks = BREAKS if breaks is None else breaks
    nK = K.shape[1]
    A = np.hstack([np.eye(nF)[fi], K])
    S = []
    for k in range(nK - 1):
        if not breaks[k]:
            s = np.zeros(nF + nK)
            s[nF + k], s[nF + k + 1] = 1, -1
            S.append(s)
    S = np.array(S)
    cnt = K[use].sum(0)
    g = np.zeros(nF + nK)
    g[nF:] = cnt / cnt.sum()
    w0 = np.where(use, 1 / SIG ** 2, 0.0)
    w = w0.copy()
    ridge = np.zeros(nF + nK)
    ridge[:nF] = 1e-6
    ridge[nF:] = 1e-6
    for _ in range(n_iter):
        N = A.T @ (A * w[:, None]) + lam * S.T @ S + 1e8 * np.outer(g, g) + np.diag(ridge)
        m = np.linalg.solve(N, A.T @ (w * D))
        r = D - A @ m
        z = r / SIG
        w = w0 * np.where(np.abs(z) <= huber, 1.0, huber / np.maximum(np.abs(z), 1e-12))
    cov = np.linalg.inv(N)
    # formal covariance excluding the gauge (approximation, gauge only shifts the mean)
    return m[:nF], m[nF:], np.diag(cov)[nF:], r, w


def family_scales(T, K, comp):
    SIG = np.sqrt(T[SDC[COMP.index(comp)]].to_numpy() ** 2 + FLOOR ** 2)
    D = T[comp].to_numpy()
    fi = T.fi.to_numpy()
    use = np.ones(len(D), bool)
    s = np.ones(nF)
    for _ in range(2):
        a, C, vC, r, w = joint_fit(D, SIG * s[fi], fi, K, use)
        z = r / (SIG * s[fi])
        for i in range(nF):
            zi = z[fi == i]
            if len(zi) > 3:
                s[i] *= 1.4826 * np.median(np.abs(zi))
    return SIG * s[fi], s


def cm_design(t):
    X = [np.ones_like(t), np.clip(t - T2020, 0, None)]
    for q in QN:
        X.append((t > TQ[q]).astype(float))
    return np.array(X).T


CM_PAR = ['const', 'rate2020_km_yr'] + ['step_' + q for q in QN]


def fit_cm_model(C, vC, cnt, tmin=2018.0):
    sel = (TC >= tmin) & (cnt > 0.5)
    X = cm_design(TC[sel])
    w = 1 / vC[sel]
    p, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], C[sel] * np.sqrt(w), rcond=None)
    r = C[sel] - X @ p
    chi2r = np.sum(w * r ** 2) / max(1, sel.sum() - X.shape[1])
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    return p, np.sqrt(np.diag(cov) * max(1, chi2r)), chi2r


def run_cm(T, K, comps=COMP, jack=True, extra_w=None):
    """Joint CM fit for each component; returns dict comp -> results."""
    res = {}
    fi = T.fi.to_numpy()
    for c in comps:
        SIG, s = family_scales(T, K, c)
        if extra_w is not None:
            SIG = SIG / np.sqrt(np.maximum(extra_w, 1e-12))
            SIG[extra_w <= 0] = 1e6
        D = T[c].to_numpy()
        use = np.ones(len(D), bool) if extra_w is None else extra_w > 0
        a, C, vC, r, w = joint_fit(D, SIG, fi, K, use)
        cnt = K[use].sum(0)
        p, pse_formal, chi2r = fit_cm_model(C, vC, cnt)
        out = dict(a=a, C=C, vC=vC, SIG=SIG, scale=s, cnt=cnt, p=p, pse_formal=pse_formal, chi2r=chi2r)
        if jack:
            CJ, PJ, AJ = [], [], []
            for i in range(nF):
                u2 = use & (fi != i)
                aj, Cj, vCj, _, _ = joint_fit(D, SIG, fi, K, u2)
                CJ.append(Cj)
                AJ.append(aj)
                PJ.append(fit_cm_model(Cj, vCj, K[u2].sum(0))[0])
            CJ, PJ = np.array(CJ), np.array(PJ)
            out['CJ'] = CJ
            out['AJ'] = np.array(AJ)
            out['C_se_jk'] = np.sqrt((nF - 1) / nF * np.sum((CJ - CJ.mean(0)) ** 2, 0))
            out['p_se_jk'] = np.sqrt((nF - 1) / nF * np.sum((PJ - PJ.mean(0)) ** 2, 0))
        res[c] = out
    return res


# ----------------------------------------------------------------------------- local estimators
def wlsq(X, y, sig):
    X, y, sig = np.asarray(X, float), np.asarray(y, float), np.asarray(sig, float)
    w = 1 / sig ** 2
    Xw = X * np.sqrt(w)[:, None]
    p, *_ = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)
    r = y - X @ p
    dof = len(y) - X.shape[1]
    chi2r = np.sum(w * r ** 2) / dof if dof > 0 else np.nan
    cov = np.linalg.pinv(X.T @ (X * w[:, None]))
    return p, np.sqrt(np.diag(cov) * max(1.0, chi2r if np.isfinite(chi2r) else 1.0)), chi2r


def local_steps(T, EVm, D, SIG, date, W=W_LOC, min_side=3):
    """Per family step at `date`: bins with tmid within +-W; model a + b(tm-date) + s*frac_after."""
    fa = frac_after(EVm, date)
    sel = np.abs(T.tmid.to_numpy() - date) <= W
    out = {}
    fi = T.fi.to_numpy()
    tm = T.tm.to_numpy()
    for i in range(nF):
        m = sel & (fi == i)
        if (fa[m] < 0.5).sum() < min_side or (fa[m] >= 0.5).sum() < min_side:
            continue
        X = np.c_[np.ones(m.sum()), tm[m] - date, fa[m]]
        p, se, _ = wlsq(X, D[m], SIG[m])
        out[FAMS[i]] = (p[2], se[2])
    return out


def pool(d):
    if len(d) < 2:
        return np.nan, np.nan, np.nan, 0
    v = np.array(list(d.values()))
    w = 1 / v[:, 1] ** 2
    m = np.sum(w * v[:, 0]) / np.sum(w)
    chi2r = np.sum(w * (v[:, 0] - m) ** 2) / (len(v) - 1)
    return m, np.sqrt(max(1.0, chi2r) / np.sum(w)), chi2r, len(v)


def local_trends(T, D, SIG, t0, t1, min_n=6):
    out = {}
    fi = T.fi.to_numpy()
    tm = T.tm.to_numpy()
    sel = (tm >= t0) & (tm <= t1)
    for i in range(nF):
        m = sel & (fi == i)
        if m.sum() < min_n or np.ptp(tm[m]) < 0.5 * (t1 - t0):
            continue
        X = np.c_[np.ones(m.sum()), tm[m] - 0.5 * (t0 + t1)]
        p, se, _ = wlsq(X, D[m], SIG[m])
        out[FAMS[i]] = (p[1], se[1])
    return out


def grad_stat(steps, key):
    """Weighted slope of per-family values vs FINFO[key], intercept free."""
    Fs = list(steps)
    if len(Fs) < 5:
        return np.nan, np.nan
    v = np.array([steps[F] for F in Fs])
    x = np.array([FINFO[F][key] for F in Fs])
    p, se, _ = wlsq(np.c_[np.ones(len(x)), x - x.mean()], v[:, 0], v[:, 1])
    return p[1], se[1]


def perm_test(steps, key, nperm=5000):
    Fs = list(steps)
    v = np.array([steps[F] for F in Fs])
    x = np.array([FINFO[F][key] for F in Fs])
    X0 = np.c_[np.ones(len(x)), x - x.mean()]
    p, se, _ = wlsq(X0, v[:, 0], v[:, 1])
    tobs = p[1] / se[1]
    cnt = 0
    for _ in range(nperm):
        xp = RNG.permutation(x)
        pp, sp, _ = wlsq(np.c_[np.ones(len(xp)), xp - xp.mean()], v[:, 0], v[:, 1])
        cnt += abs(pp[1] / sp[1]) >= abs(tobs)
    return p[1], se[1], (cnt + 1) / (nperm + 1)


def block_boot(X, y, sig, nboot=1000, blk=5):
    n = len(y)
    p0, se0, chi2r = wlsq(X, y, sig)
    B = []
    nb = int(np.ceil(n / blk))
    for _ in range(nboot):
        st = RNG.integers(0, max(1, n - blk + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + blk, n)) for s in st])[:n]
        Xb = X[idx]
        if np.linalg.matrix_rank(Xb) < X.shape[1]:
            continue
        B.append(wlsq(Xb, y[idx], sig[idx])[0])
    B = np.array(B)
    return p0, se0, (B.std(0) if len(B) > 50 else np.full(len(p0), np.nan)), chi2r


# ----------------------------------------------------------------------------- re-inversion from obs
GRID = json.load(open('data/velocity/tt_3d/grid.json'))
GX, GY, GZ = map(np.array, (GRID['sub_x'], GRID['sub_y'], GRID['sub_z']))
TRP = Transformer.from_crs('EPSG:4326', GRID['proj'], always_xy=True)
NPZ = {}


def obs_geometry(F):
    """3-D gradient (s/km) and observed travel time for each core cid/phase of family F."""
    loc = json.load(open(f'data/locations/fam{F}_3d.json'))
    picks = pd.read_csv(f'data/locations/fam{F}_3d_picks.csv').set_index('sta')
    x0, y0 = TRP.transform(loc['lon'], loc['lat'])
    i0, j0, k0 = (int(np.argmin(abs(GX - x0))), int(np.argmin(abs(GY - y0))),
                  int(np.argmin(abs(GZ - loc['depth_km']))))
    d = GRID['d']
    o = pd.read_csv(f'data/relocations/fam{F}_bins_3d_core_obs.csv')
    geo = {}
    for cid, ph in o[['cid', 'phase']].drop_duplicates().itertuples(index=False):
        sta = '.'.join(cid.split('.')[:2])
        if sta not in NPZ:
            NPZ[sta] = dict(np.load(f'data/velocity/tt_3d/{sta}.npz'))
        sc = loc.get('s_scale', 1.0) if ph == 'S' else 1.0
        V = NPZ[sta]['t' + ph.lower()] * sc
        g = np.array([(V[i0 + 1, j0, k0] - V[i0 - 1, j0, k0]) / (2 * d),
                      (V[i0, j0 + 1, k0] - V[i0, j0 - 1, k0]) / (2 * d),
                      (V[i0, j0, k0 + 1] - V[i0, j0, k0 - 1]) / (2 * d)])
        col = 'S_final' if ph == 'S' else 'P'
        tobs = picks.loc[sta, col] - loc['origin_rel_catalog_s'] if sta in picks.index else np.nan
        geo[(cid, ph)] = dict(sta=sta, g=g, az=np.degrees(np.arctan2(-g[0], -g[1])),
                              sigma=0.02 if ph == 'S' else 0.03, T=float(tobs), Tmod=float(V[i0, j0, k0]))
    return o, geo


def gapf(az):
    a = np.sort(np.mod(az, 360))
    return np.max(np.diff(np.r_[a, a[0] + 360])) if len(a) > 1 else 360.0


def solve_huber(G, d, w):
    ww = w.copy()
    for _ in range(5):
        Wt = np.sqrt(ww)[:, None]
        m, *_ = np.linalg.lstsq(G * Wt, d * Wt[:, 0], rcond=None)
        r = d - G @ m
        s = max(1.4826 * np.median(np.abs(r)), 0.005)
        ww = w * np.where(np.abs(r) <= 1.5 * s, 1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
    return m


def reinvert(o, geo, drop=None, minobs=8, minsta=5):
    out = []
    for b, grp in o.groupby('bin'):
        rows = [(geo[(c, p)], dt, cc) for c, p, dt, cc in grp[['cid', 'phase', 'dt', 'cc']].itertuples(index=False)
                if geo[(c, p)]['sta'] != drop]
        stas = {r[0]['sta'] for r in rows}
        if len(rows) < minobs or len(stas) < minsta or gapf(np.array([r[0]['az'] for r in rows])) >= 180:
            continue
        G = np.array([np.r_[r[0]['g'], 1.0] for r in rows])
        d = np.array([r[1] for r in rows])
        w = np.array([r[2] ** 2 / r[0]['sigma'] ** 2 for r in rows])
        m = solve_huber(G, d, w)
        out.append(dict(bin=b, dx=m[0], dy=m[1], dz=m[2], dt0=m[3]))
    return pd.DataFrame(out)


# ============================================================================= main
def main():
    S = {}   # JSON summary
    core = {F: read_bins(F, '_core') for F in FAMS}
    T, EVm, K = build_table(core)
    fi = T.fi.to_numpy()
    print(f'{nF} families, {len(T)} core bins, {nK} cells', flush=True)
    S['n_families'] = nF
    S['families'] = FAMS
    S['n_bins'] = int(len(T))

    # ---------------- Task 1: CM
    R = run_cm(T, K)
    cmrows = []
    for c in COMP:
        r = R[c]
        S.setdefault('common_mode_model_2018on', {})[c] = {
            n: dict(value_m=1e3 * r['p'][k], se_jackknife_m=1e3 * r['p_se_jk'][k],
                    se_formal_scaled_m=1e3 * r['pse_formal'][k]) for k, n in enumerate(CM_PAR)}
        S['common_mode_model_2018on'][c]['chi2r_cells'] = r['chi2r']
        S.setdefault('family_noise_scale', {})[c] = dict(zip(map(str, FAMS), np.round(r['scale'], 3).tolist()))
    for k in range(nK):
        row = dict(t0=EDGES[k], t1=EDGES[k + 1], t_center=TC[k], n_bins_equiv=R['dz']['cnt'][k])
        for c in COMP:
            row[f'C_{c}_m'] = 1e3 * R[c]['C'][k]
            row[f'C_{c}_se_jk_m'] = 1e3 * R[c]['C_se_jk'][k]
            row[f'C_{c}_se_formal_m'] = 1e3 * np.sqrt(R[c]['vC'][k])
        cmrows.append(row)
    pd.DataFrame(cmrows).to_csv(f'{OUT}/common_mode_track.csv', index=False)

    # differential tracks with leave-f-out CM
    for c in COMP:
        Cj = R[c]['CJ']
        cm_b = np.einsum('bk,bk->b', K, Cj[fi])
        diff = T[c].to_numpy() - cm_b
        # remove family static (weighted mean)
        SIG = R[c]['SIG']
        for i in range(nF):
            m = fi == i
            diff[m] -= np.sum(diff[m] / SIG[m] ** 2) / np.sum(1 / SIG[m] ** 2)
        T[f'diff_{c}'] = diff
        T[f'cm_{c}'] = cm_b
        T[f'sig_{c}'] = SIG
    T.to_csv(f'{OUT}/differential_bins.csv', index=False)

    # Pooled local step estimator (raw = ~CM) and differential local steps at quakes
    S['local_step_estimator'] = f'per-family a+b(t-T)+s*frac_after within +-{W_LOC} yr, >=3 bins each side'
    steprows = []
    for q in QN:
        for c in COMP:
            raw = local_steps(T, EVm, T[c].to_numpy(), T[f'sig_{c}'].to_numpy(), TQ[q])
            dif = local_steps(T, EVm, T[f'diff_{c}'].to_numpy(), T[f'sig_{c}'].to_numpy(), TQ[q])
            m, se, chi2r, n = pool(raw)
            S.setdefault('pooled_raw_local_step', {}).setdefault(q, {})[c] = dict(
                value_m=1e3 * m, se_m=1e3 * se, chi2r=chi2r, n_fam=n)
            # differential: heterogeneity + gradients
            v = np.array(list(dif.values()))
            chi2 = float(np.sum((v[:, 0] / v[:, 1]) ** 2)) if len(v) else np.nan
            gr = {}
            for key in ['E', 'depth', 'dist_m7', 'along_strike', 'down_dip']:
                sl, sse, pp = perm_test(dif, key, nperm=2000)
                gr[key] = dict(slope_m_per_km=1e3 * sl, se=1e3 * sse, p_perm=pp)
            S.setdefault('differential_local_step', {}).setdefault(q, {})[c] = dict(
                chi2=chi2, dof=len(v), gradients=gr)
            for F in raw:
                steprows.append(dict(quake=q, comp=c, F=F, raw_step_m=1e3 * raw[F][0], raw_se_m=1e3 * raw[F][1],
                                     diff_step_m=1e3 * dif[F][0] if F in dif else np.nan,
                                     diff_se_m=1e3 * dif[F][1] if F in dif else np.nan, **FINFO[F]))
    pd.DataFrame(steprows).to_csv(f'{OUT}/family_steps_local.csv', index=False)

    # global per-family differential model A: const + ramp2020 + steps (block bootstrap)
    famrows = []
    for F in FAMS:
        m = (fi == FIDX[F])
        Tf = T[m]
        row = dict(F=F, n_bins=int(m.sum()), **FINFO[F])
        XA = np.c_[np.ones(m.sum()), Tf.ramp.to_numpy()] if True else None
        XA = np.c_[XA, Tf[['fa_' + q for q in QN]].to_numpy()]
        XB = np.c_[np.ones(m.sum()), Tf.tm.to_numpy() - 2017.0]
        XB2 = np.c_[XB, Tf[['fa_' + q for q in QN]].to_numpy()]
        for c in COMP:
            y = Tf[f'diff_{c}'].to_numpy()
            sg = Tf[f'sig_{c}'].to_numpy()
            ok = all(((Tf['fa_' + q] > 0.5).sum() >= 2) and ((Tf['fa_' + q] < 0.5).sum() >= 2) for q in QN)
            if ok and m.sum() >= 12:
                pA, seA, bsA, chA = block_boot(XA, y, sg, nboot=500)
                row[f'{c}_rate2020_m_yr'] = 1e3 * pA[1]
                row[f'{c}_rate2020_se'] = 1e3 * np.nanmax([seA[1], bsA[1]])
                for k, q in enumerate(QN):
                    row[f'{c}_step_{q}_m'] = 1e3 * pA[2 + k]
                    row[f'{c}_step_{q}_se'] = 1e3 * np.nanmax([seA[2 + k], bsA[2 + k]])
            pB, seB, bsB, chB = block_boot(XB, y, sg, nboot=1000)
            row[f'{c}_rate_all_m_yr'] = 1e3 * pB[1]
            row[f'{c}_rate_all_se_formal'] = 1e3 * seB[1]
            row[f'{c}_rate_all_se_boot'] = 1e3 * bsB[1]
            row[f'{c}_rate_all_se'] = 1e3 * np.nanmax([seB[1], bsB[1]])
            row[f'{c}_chi2r'] = chB
            if np.linalg.matrix_rank(XB2) == XB2.shape[1] and m.sum() >= 12:
                pB2, seB2, bsB2, _ = block_boot(XB2, y, sg, nboot=500)
                row[f'{c}_rate_all_withsteps_m_yr'] = 1e3 * pB2[1]
                row[f'{c}_rate_all_withsteps_se'] = 1e3 * np.nanmax([seB2[1], bsB2[1]])
        famrows.append(row)
    FR = pd.DataFrame(famrows)

    # Task 4: fault-frame projection of differential rates (independent comps; ignore covariances)
    for nm, u in [('strike', u_strike), ('dip', u_dip), ('normal', u_norm)]:
        FR[f'rate_{nm}_m_yr'] = sum(u[k] * FR[f'{c}_rate_all_m_yr'] for k, c in enumerate(COMP))
        FR[f'rate_{nm}_se'] = np.sqrt(sum((u[k] * FR[f'{c}_rate_all_se']) ** 2 for k, c in enumerate(COMP)))
    FR.to_csv(f'{OUT}/family_differential_fits.csv', index=False)
    mig = {}
    for nm in ['strike', 'dip', 'normal'] + [f'{c}' for c in COMP]:
        col = f'rate_{nm}_m_yr' if nm in ['strike', 'dip', 'normal'] else f'{nm}_rate_all_m_yr'
        scol = f'rate_{nm}_se' if nm in ['strike', 'dip', 'normal'] else f'{nm}_rate_all_se'
        sub = FR[np.isfinite(FR[col]) & np.isfinite(FR[scol]) & (FR.n_bins >= 12)]
        v = sub[col].to_numpy()
        s = sub[scol].to_numpy()
        w = 1 / s ** 2
        mu = np.sum(w * v) / np.sum(w)
        chi2 = float(np.sum(w * (v - mu) ** 2))
        d = {int(F): (a / 1e3, b / 1e3) for F, a, b in zip(sub.F, v, s)}
        gr = {}
        for key in ['along_strike', 'down_dip', 'depth', 'E']:
            sl, sse, pp = perm_test(d, key, nperm=2000)
            gr[key] = dict(slope_m_per_yr_per_km=1e3 * sl, se=1e3 * sse, p_perm=pp)
        mig[nm] = dict(n_fam=len(v), chi2_about_mean=chi2, dof=len(v) - 1, max_abs_rate_m_yr=float(np.max(np.abs(v))),
                       median_se_m_yr=float(np.median(s)), rms_rate_m_yr=float(np.sqrt(np.mean(v ** 2))),
                       gradients=gr)
    # pairwise: fam 23/27 (east) vs western families
    S['migration_differential_since_2008'] = mig

    # ---------------- Placebo tests (reviewer 1)
    print('placebo', flush=True)
    qs = np.array(list(TQ.values()))
    dates = RNG.uniform(2009.3, 2025.7, 1000)
    pl = []
    for dte in dates:
        row = dict(date=dte, near_quake=float(np.min(np.abs(dte - qs))) < W_LOC)
        for c in COMP:
            raw = local_steps(T, EVm, T[c].to_numpy(), T[f'sig_{c}'].to_numpy(), dte)
            m, se, _, n = pool(raw)
            row[f'raw_{c}_m'] = 1e3 * m
            row[f'raw_{c}_z'] = m / se if se > 0 else np.nan
            if c == 'dz':
                dif = local_steps(T, EVm, T['diff_dz'].to_numpy(), T['sig_dz'].to_numpy(), dte)
                v = np.array(list(dif.values())) if dif else np.zeros((0, 2))
                row['diff_dz_chi2_per_dof'] = float(np.mean((v[:, 0] / v[:, 1]) ** 2)) if len(v) else np.nan
                for key in ['E', 'depth', 'dist_m7']:
                    sl, sse = grad_stat(dif, key)
                    row[f'diff_dz_grad_{key}_z'] = sl / sse if sse else np.nan
        pl.append(row)
    PL = pd.DataFrame(pl)
    PL.to_csv(f'{OUT}/placebo_steps.csv', index=False)
    plac = {}
    for q in QN:
        plac[q] = {}
        for c in COMP:
            o = S['pooled_raw_local_step'][q][c]
            for nm, sub in [('all', PL), ('far_from_quakes', PL[~PL.near_quake])]:
                v = sub[f'raw_{c}_m'].dropna().to_numpy()
                zv = sub[f'raw_{c}_z'].dropna().to_numpy()
                plac[q].setdefault(c, {})[nm] = dict(
                    n=len(v), pct_signed=float(np.mean(v < o['value_m']) * 100),
                    pct_abs=float(np.mean(np.abs(v) < abs(o['value_m'])) * 100),
                    pct_abs_z=float(np.mean(np.abs(zv) < abs(o['value_m'] / o['se_m'])) * 100),
                    placebo_sd_m=float(np.std(v)), placebo_p95_abs_m=float(np.percentile(np.abs(v), 95)))
        dif = local_steps(T, EVm, T['diff_dz'].to_numpy(), T['sig_dz'].to_numpy(), TQ[q])
        v = np.array(list(dif.values()))
        c2 = float(np.mean((v[:, 0] / v[:, 1]) ** 2))
        sub = PL[~PL.near_quake]
        plac[q]['diff_dz_chi2_per_dof'] = dict(value=c2, pct=float(np.mean(sub.diff_dz_chi2_per_dof.dropna() < c2) * 100))
        for key in ['E', 'depth', 'dist_m7']:
            sl, sse = grad_stat(dif, key)
            zz = sub[f'diff_dz_grad_{key}_z'].dropna().to_numpy()
            plac[q][f'diff_dz_grad_{key}'] = dict(z=sl / sse, pct_abs=float(np.mean(np.abs(zz) < abs(sl / sse)) * 100))
    S['placebo_steps'] = plac

    # trend placebo: pooled per-family linear trend in window
    obs_tr = {}
    for c in COMP:
        tr = local_trends(T, T[c].to_numpy(), T[f'sig_{c}'].to_numpy(), T2020, 2026.55)
        m, se, chi2r, n = pool(tr)
        obs_tr[c] = dict(value_m_yr=1e3 * m, se_m_yr=1e3 * se, chi2r=chi2r, n_fam=n)
    S['pooled_raw_trend_2020on'] = obs_tr
    ptr = []
    for _ in range(1000):
        L = RNG.uniform(4.0, 6.5)
        t0 = RNG.uniform(2008.5, 2026.55 - L)
        row = dict(t0=t0, L=L)
        for c in COMP:
            m, se, _, n = pool(local_trends(T, T[c].to_numpy(), T[f'sig_{c}'].to_numpy(), t0, t0 + L))
            row[f'{c}_m_yr'] = 1e3 * m
            row[f'{c}_z'] = m / se if se > 0 else np.nan
        ptr.append(row)
    PT = pd.DataFrame(ptr)
    PT.to_csv(f'{OUT}/placebo_trends.csv', index=False)
    pt = {}
    for c in COMP:
        o = obs_tr[c]
        for nm, sub in [('all', PT), ('end_before_2020', PT[PT.t0 + PT.L < T2020])]:
            v = sub[f'{c}_m_yr'].dropna().to_numpy()
            pt.setdefault(c, {})[nm] = dict(n=len(v), pct_signed=float(np.mean(v < o['value_m_yr']) * 100),
                                            pct_abs=float(np.mean(np.abs(v) < abs(o['value_m_yr'])) * 100),
                                            placebo_sd_m_yr=float(np.std(v)))
    S['placebo_trends'] = pt

    # ---------------- mean_cc reweighting (reviewer 4)
    print('cc reweight', flush=True)
    pre = T.tm.to_numpy() < T2020
    S['mean_cc'] = dict(median_pre2020=float(np.median(T.mean_cc[pre])), median_post2020=float(np.median(T.mean_cc[~pre])))
    corr = {}
    for c in COMP:
        # within-family correlation of position with mean_cc
        rr = []
        for i in range(nF):
            m = fi == i
            if m.sum() > 8:
                rr.append(np.corrcoef(T.mean_cc[m], T[c][m])[0, 1])
        corr[c] = dict(median_within_family_r=float(np.median(rr)), mean_r=float(np.mean(rr)))
    S['corr_position_meancc'] = corr
    wts = {}
    w_hist = np.zeros(len(T))
    w_band = np.zeros(len(T))
    for i in range(nF):
        m = fi == i
        cc = T.mean_cc.to_numpy()[m]
        p_ = pre[m]
        if p_.sum() >= 5 and (~p_).sum() >= 5:
            qe = np.quantile(cc[p_], [0, .2, .4, .6, .8, 1])
            qe[0], qe[-1] = -np.inf, np.inf
            hb = np.digitize(cc, qe[1:-1])
            ww = np.ones(m.sum())
            for k in range(5):
                npre = np.sum(p_ & (hb == k)) / p_.sum()
                npo = np.sum(~p_ & (hb == k)) / (~p_).sum()
                ww[~p_ & (hb == k)] = npre / npo if npo > 0 else 0
            w_hist[m] = ww
        lo, hi = np.quantile(cc, [0.25, 0.75])
        w_band[m] = ((cc >= lo) & (cc <= hi)).astype(float)
    wts['hist_matched'] = w_hist
    wts['cc_band_IQR'] = w_band
    ccres = {}
    for nm, ww in wts.items():
        Rw = run_cm(T, K, comps=['dz'], jack=True, extra_w=ww)
        SIGw = T.sig_dz.to_numpy() / np.sqrt(np.maximum(ww, 1e-12))
        Tw = T[ww > 0]
        tr = pool(local_trends(Tw, Tw.dz.to_numpy(), SIGw[ww > 0], T2020, 2026.55))
        st = pool(local_steps(Tw, EVm[ww > 0], Tw.dz.to_numpy(), SIGw[ww > 0], TQ['M7_2024']))
        ccres[nm] = dict(n_bins_used=int((ww > 0).sum()),
                         cm_rate2020_m_yr=1e3 * Rw['dz']['p'][1], cm_rate2020_se_jk=1e3 * Rw['dz']['p_se_jk'][1],
                         cm_step_M7_m=1e3 * Rw['dz']['p'][4], cm_step_M7_se_jk=1e3 * Rw['dz']['p_se_jk'][4],
                         pooled_trend2020_m_yr=1e3 * tr[0], pooled_trend2020_se=1e3 * tr[1],
                         pooled_M7_step_m=1e3 * st[0], pooled_M7_step_se=1e3 * st[1])
    # residualise dz on mean_cc and n_sta within family
    dzr = T.dz.to_numpy().copy()
    for i in range(nF):
        m = fi == i
        X = np.c_[np.ones(m.sum()), T.mean_cc[m] - T.mean_cc[m].mean(), T.n_sta[m] - T.n_sta[m].mean()]
        p, *_ = np.linalg.lstsq(X / T.sig_dz.to_numpy()[m, None], dzr[m] / T.sig_dz.to_numpy()[m], rcond=None)
        dzr[m] -= X[:, 1:] @ p[1:]
    T2 = T.copy()
    T2['dz'] = dzr
    Rr = run_cm(T2, K, comps=['dz'], jack=True)
    tr = pool(local_trends(T2, dzr, T.sig_dz.to_numpy(), T2020, 2026.55))
    st = pool(local_steps(T2, EVm, dzr, T.sig_dz.to_numpy(), TQ['M7_2024']))
    ccres['residualised_on_meancc_nsta'] = dict(
        cm_rate2020_m_yr=1e3 * Rr['dz']['p'][1], cm_rate2020_se_jk=1e3 * Rr['dz']['p_se_jk'][1],
        cm_step_M7_m=1e3 * Rr['dz']['p'][4], cm_step_M7_se_jk=1e3 * Rr['dz']['p_se_jk'][4],
        pooled_trend2020_m_yr=1e3 * tr[0], pooled_trend2020_se=1e3 * tr[1],
        pooled_M7_step_m=1e3 * st[0], pooled_M7_step_se=1e3 * st[1])
    ccres['baseline'] = dict(cm_rate2020_m_yr=1e3 * R['dz']['p'][1], cm_rate2020_se_jk=1e3 * R['dz']['p_se_jk'][1],
                             cm_step_M7_m=1e3 * R['dz']['p'][4], cm_step_M7_se_jk=1e3 * R['dz']['p_se_jk'][4],
                             pooled_trend2020_m_yr=obs_tr['dz']['value_m_yr'], pooled_trend2020_se=obs_tr['dz']['se_m_yr'],
                             pooled_M7_step_m=S['pooled_raw_local_step']['M7_2024']['dz']['value_m'],
                             pooled_M7_step_se=S['pooled_raw_local_step']['M7_2024']['dz']['se_m'])
    S['meancc_reweighting_dz'] = ccres

    # ---------------- Leave-one-station-out re-inversion (reviewer 2) + per-station delays (3)
    print('LOSO', flush=True)
    OBS, GEO = {}, {}
    for F in FAMS:
        OBS[F], GEO[F] = obs_geometry(F)
    # reproduction check
    rep = []
    for F in FAMS:
        rv = reinvert(OBS[F], GEO[F]).merge(core[F][['bin', 'dx', 'dy', 'dz']], on='bin', suffixes=('', '_f'))
        rep.append((rv[['dx', 'dy', 'dz']].to_numpy() - rv[['dx_f', 'dy_f', 'dz_f']].to_numpy()))
    rep = np.vstack(rep)
    S['reinversion_reproduction_rms_m'] = dict(zip(COMP, (1e3 * np.sqrt(np.mean(rep ** 2, 0))).tolist()))
    stations = sorted({g['sta'] for F in FAMS for g in GEO[F].values()})
    loso = []
    for sta in [None] + stations:
        newb = {}
        for F in FAMS:
            rv = reinvert(OBS[F], GEO[F], drop=sta)
            if len(rv) < 5:
                continue
            b = core[F].drop(columns=['dx', 'dy', 'dz', 'dt0']).merge(rv, on='bin')
            if len(b) >= 5:
                newb[F] = b
        Tl, EVl, Kl = build_table(newb)
        Rl = run_cm(Tl, Kl, comps=['dz'], jack=False)
        SIGl = Rl['dz']['SIG']
        tr = pool(local_trends(Tl, Tl.dz.to_numpy(), SIGl, T2020, 2026.55))
        st = pool(local_steps(Tl, EVl, Tl.dz.to_numpy(), SIGl, TQ['M7_2024']))
        loso.append(dict(dropped=sta or 'none', n_fam=len(newb), n_bins=len(Tl),
                         cm_step_M7_m=1e3 * Rl['dz']['p'][4], cm_rate2020_m_yr=1e3 * Rl['dz']['p'][1],
                         cm_step_Q2021_m=1e3 * Rl['dz']['p'][2], cm_step_Q2022_m=1e3 * Rl['dz']['p'][3],
                         pooled_M7_step_m=1e3 * st[0], pooled_M7_step_se=1e3 * st[1],
                         pooled_trend2020_m_yr=1e3 * tr[0], pooled_trend2020_se=1e3 * tr[1]))
        print(loso[-1], flush=True)
    LO = pd.DataFrame(loso)
    LO.to_csv(f'{OUT}/leave_one_station_out.csv', index=False)
    L1 = LO[LO.dropped != 'none']
    n = len(L1)
    S['leave_one_station_out_dz'] = {
        k: dict(all_stations=float(LO[k].iloc[0]), min=float(L1[k].min()), max=float(L1[k].max()),
                jackknife_se=float(np.sqrt((n - 1) / n * np.sum((L1[k] - L1[k].mean()) ** 2))),
                most_influential=str(L1.dropped.iloc[int(np.argmax(np.abs(L1[k] - LO[k].iloc[0])))]))
        for k in ['cm_step_M7_m', 'cm_rate2020_m_yr', 'pooled_M7_step_m', 'pooled_trend2020_m_yr']}

    # per-station direct-phase delay change at M7 (post - pre, +-W_LOC) stacked over families
    tq = TQ['M7_2024']
    drows = []
    for F in FAMS:
        o = OBS[F].merge(T[T.F == F][['bin', 'tmid', 'fa_M7_2024']], on='bin')
        o = o[np.abs(o.tmid - tq) <= W_LOC]
        for (cid, ph), g in o.groupby(['cid', 'phase']):
            pre_, post_ = g[g.fa_M7_2024 < 0.5], g[g.fa_M7_2024 >= 0.5]
            if len(pre_) < 2 or len(post_) < 2:
                continue
            def wm(x):
                w = x.cc ** 2
                mu = np.sum(w * x.dt) / np.sum(w)
                return mu, np.sqrt(np.sum(w * (x.dt - mu) ** 2) / np.sum(w) / max(1, len(x) - 1))
            a, sa = wm(pre_)
            b, sb = wm(post_)
            ge = GEO[F][(cid, ph)]
            drows.append(dict(F=F, cid=cid, sta=ge['sta'], phase=ph, ddt_ms=1e3 * (b - a),
                              se_ms=1e3 * np.hypot(max(sa, 1e-3), max(sb, 1e-3)), T_obs=ge['T'], T_mod=ge['Tmod'],
                              gz=ge['g'][2]))
    DD = pd.DataFrame(drows)
    DD.to_csv(f'{OUT}/station_delay_change_M7.csv', index=False)
    st_sum = DD.groupby(['sta', 'phase']).apply(
        lambda g: pd.Series(dict(ddt_ms=np.sum(g.ddt_ms / g.se_ms ** 2) / np.sum(1 / g.se_ms ** 2),
                                 se_ms=1 / np.sqrt(np.sum(1 / g.se_ms ** 2)), n_fam=len(g),
                                 T_mod=g.T_mod.mean())), include_groups=False).reset_index()
    st_sum.to_csv(f'{OUT}/station_delay_change_M7_by_station.csv', index=False)
    # apparent eps from regression ddt = a + b*T (per family, pooled)
    bl = []
    for F, g in DD.groupby('F'):
        if len(g) >= 6:
            p, se, _ = wlsq(np.c_[np.ones(len(g)), g.T_mod], g.ddt_ms / 1e3, g.se_ms / 1e3)
            bl.append((p[1], se[1]))
    bl = np.array(bl)
    w = 1 / bl[:, 1] ** 2
    S['station_delays_M7'] = dict(
        note='Per-station post-minus-pre delays of LFE direct phases (same data that drive the relocations; '
             'no independent direct-S delays from other sources are available here).',
        apparent_dvv_from_dt_vs_T_slope=dict(value=float(-np.sum(w * bl[:, 0]) / np.sum(w)),
                                             se=float(np.sqrt(max(1, np.sum(w * (bl[:, 0] - np.sum(w * bl[:, 0]) / np.sum(w)) ** 2) / (len(bl) - 1)) / np.sum(w))),
                                             n_fam=len(bl)),
        largest_station_changes=st_sum.reindex(st_sum.ddt_ms.abs().sort_values(ascending=False).index).head(6).round(3).to_dict('records'))

    # ---------------- Task 2: predicted apparent shift from uniform dv/v
    print('dvv prediction', flush=True)
    sc = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv')
    sc['ns'] = sc.net + '.' + sc.sta
    sc = sc.drop_duplicates('ns').set_index('ns')
    VP, VS = 6.5, 6.5 / 1.75
    eps = -1e-4
    prow = []
    for F in FAMS:
        o = OBS[F]
        use = o.groupby(['cid', 'phase']).agg(n=('dt', 'size'), cc=('cc', 'mean')).reset_index()
        xs, ys = en(FINFO[F]['lat'], FINFO[F]['lon'])
        zs = FINFO[F]['depth']
        G, wv, dT = [], [], {k: [] for k in ['whole', 'layer1km', 'layer3km', 'whole_3d']}
        G3 = []
        for r in use.itertuples(index=False):
            sta = '.'.join(r.cid.split('.')[:2])
            s = sc.loc[sta]
            xi, yi = en(s.lat, s.lon)
            zi = -(s.elev - s.depth) / 1e3
            v = VS if r.phase == 'S' else VP
            vec = np.array([xs - xi, ys - yi, zs - zi])
            R_ = np.linalg.norm(vec)
            G.append(np.r_[vec / (R_ * v), 1.0])
            ge = GEO[F][(r.cid, r.phase)]
            G3.append(np.r_[ge['g'], 1.0])
            wv.append(r.n * r.cc ** 2 / (0.02 if r.phase == 'S' else 0.03) ** 2)
            dT['whole'].append(-eps * R_ / v)
            dT['layer1km'].append(-eps * (1.0 * R_ / (zs - zi)) / v)
            dT['layer3km'].append(-eps * (3.0 * R_ / (zs - zi)) / v)
            dT['whole_3d'].append(-eps * ge['Tmod'])
        G, G3, wv = np.array(G), np.array(G3), np.array(wv)
        row = dict(F=F, **FINFO[F])
        for k, d in dT.items():
            Gu = G3 if k.endswith('3d') else G
            sw = np.sqrt(wv)
            m, *_ = np.linalg.lstsq(Gu * sw[:, None], np.array(d) * sw, rcond=None)
            for j, c in enumerate(['dx', 'dy', 'dz', 'dt0']):
                row[f'{k}_{c}_per_1e-4'] = 1e3 * m[j] if j < 3 else 1e3 * m[j]  # m (km->m) or ms
        prow.append(row)
    PR = pd.DataFrame(prow)
    PR.to_csv(f'{OUT}/dvv_predicted_shift.csv', index=False)
    pred = {}
    for k in ['whole', 'layer1km', 'layer3km', 'whole_3d']:
        dzp = PR[f'{k}_dz_per_1e-4']
        pred[k] = dict(dz_m_per_dvv_minus_0p01pct_median=float(dzp.median()), range=[float(dzp.min()), float(dzp.max())],
                       dx_median=float(PR[f'{k}_dx_per_1e-4'].median()), dy_median=float(PR[f'{k}_dy_per_1e-4'].median()),
                       dz_for_dvv_m0p02pct=float(2 * dzp.median()), dz_for_dvv_m0p065pct=float(6.5 * dzp.median()))
    S['dvv_predicted_shift'] = dict(note='m of apparent shift for dv/v = -0.01% (scale linearly); dz>0 = deeper. '
                                         'whole = uniform change along entire path; layerH = change confined to top H km below '
                                         'stations; whole_3d = 3-D gradients and model travel times',
                                    cases=pred)

    # ---------------- Task 3: availability-matched families
    print('m7set', flush=True)
    av = {}
    for F in [23, 8, 1]:
        av[F] = {}
        for suf in ['_core', '_m7set', '']:
            b = read_bins(F, suf)
            if b is None:
                continue
            Tb, EVb, Kb = build_table({F: b})
            i = FIDX[F]
            sig = np.sqrt(Tb.sd_z.to_numpy() ** 2 + FLOOR ** 2) * R['dz']['scale'][i]
            cmb = Kb @ R['dz']['CJ'][i]
            raw = local_steps(Tb, EVb, Tb.dz.to_numpy(), sig, TQ['M7_2024'])
            dif = local_steps(Tb, EVb, Tb.dz.to_numpy() - cmb, sig, TQ['M7_2024'])
            av[F][suf or '_all'] = dict(n_bins=len(Tb), raw_M7_dz_step_m=1e3 * raw[F][0] if F in raw else None,
                                        raw_se=1e3 * raw[F][1] if F in raw else None,
                                        diff_M7_dz_step_m=1e3 * dif[F][0] if F in dif else None,
                                        diff_se=1e3 * dif[F][1] if F in dif else None)
    S['availability_check_M7_dz'] = av
    S['common_mode_M7_dz_jk'] = dict(value_m=1e3 * R['dz']['p'][4], se_m=1e3 * R['dz']['p_se_jk'][4])


    # ---------------- CM-local step estimator (identical estimator at M7, quakes, random dates)
    print('cm placebo', flush=True)
    SIGz = R['dz']['SIG']
    Dz = T.dz.to_numpy()

    def kernel(edges):
        Kx = np.zeros((len(EVm), len(edges) - 1))
        for i in range(len(EVm)):
            e = EVm[i][~np.isnan(EVm[i])]
            Kx[i] = np.histogram(e, edges)[0] / len(e)
        return Kx

    def cm_local(date, W=1.5, use=None, return_C=False):
        extra = () if min(abs(date - v) for v in TQ.values()) < 1e-6 else (date,)
        ed, br = make_cells(extra=extra)
        tc = 0.5 * (ed[1:] + ed[:-1])
        Kx = kernel(ed)
        u = np.ones(len(Dz), bool) if use is None else use
        _, C, vC, _, _ = joint_fit(Dz, SIGz, fi, Kx, u, breaks=br)
        cnt = Kx[u].sum(0)
        sel = (np.abs(tc - date) <= W) & (cnt > 0.5)
        cols = [np.ones(sel.sum()), tc[sel] - date, (tc[sel] > date).astype(float)]
        for v in TQ.values():
            if abs(v - date) > 1e-6 and abs(v - date) < W:
                h = (tc[sel] > v).astype(float)
                if 0 < h.sum() < sel.sum():
                    cols.append(h)
        X = np.array(cols).T
        if (X[:, 2].sum() < 3) or ((1 - X[:, 2]).sum() < 3):
            return np.nan, np.nan
        p, se, _ = wlsq(X, C[sel], np.sqrt(vC[sel]))
        return p[2], se[2]

    cml = {}
    for q in QN:
        v, se = cm_local(TQ[q])
        jk = np.array([cm_local(TQ[q], use=(fi != i))[0] for i in range(nF)])
        cml[q] = dict(value_m=1e3 * v, se_formal_m=1e3 * se,
                      se_jackknife_fam_m=1e3 * float(np.sqrt((nF - 1) / nF * np.sum((jk - jk.mean()) ** 2))))
    qv = np.array(list(TQ.values()))
    pd_ = []
    while len(pd_) < 400:
        d = RNG.uniform(2009.9, 2025.0)
        if np.min(np.abs(d - qv)) < 0.3:
            continue
        v, se = cm_local(d)
        pd_.append(dict(date=d, step_m=1e3 * v, z=v / se if se else np.nan))
    PC = pd.DataFrame(pd_).dropna()
    PC.to_csv(f'{OUT}/placebo_cm_local_steps.csv', index=False)
    for q in QN:
        o = cml[q]
        cml[q]['placebo_pct_signed'] = float(np.mean(PC.step_m < o['value_m']) * 100)
        cml[q]['placebo_pct_abs'] = float(np.mean(np.abs(PC.step_m) < abs(o['value_m'])) * 100)
        cml[q]['placebo_pct_abs_z'] = float(np.mean(np.abs(PC.z) < abs(o['value_m'] / o['se_formal_m'])) * 100)
    S['cm_local_step_dz'] = dict(estimator='joint CM with extra cell edge at date; CM cells within +-1.5 yr: '
                                           'const + linear + step (+ nuisance steps at other quakes in window)',
                                 n_placebo=len(PC), placebo_sd_m=float(PC.step_m.std()),
                                 placebo_p95_abs_m=float(np.percentile(np.abs(PC.step_m), 95)),
                                 placebo_sd_z=float(PC.z.std()), results=cml)

    # CM linear trend in window: observed 2020-2026.55 vs random 4-6.5 yr windows
    def cm_trend(C, vC, cnt, t0, t1):
        sel = (TC >= t0) & (TC <= t1) & (cnt > 0.5)
        p, se, _ = wlsq(np.c_[np.ones(sel.sum()), TC[sel] - 0.5 * (t0 + t1)], C[sel], np.sqrt(vC[sel]))
        return p[1], se[1]
    rz = R['dz']
    ob, obse = cm_trend(rz['C'], rz['vC'], rz['cnt'], T2020, 2026.55)
    jk = np.array([cm_trend(rz['CJ'][i], rz['vC'], rz['cnt'], T2020, 2026.55)[0] for i in range(nF)])
    pt2 = []
    for _ in range(1000):
        L = RNG.uniform(4.0, 6.5)
        t0 = RNG.uniform(2008.5, 2026.55 - L)
        v, se = cm_trend(rz['C'], rz['vC'], rz['cnt'], t0, t0 + L)
        pt2.append(dict(t0=t0, L=L, rate_m_yr=1e3 * v))
    P2 = pd.DataFrame(pt2)
    P2.to_csv(f'{OUT}/placebo_cm_trends.csv', index=False)
    pre20 = P2[P2.t0 + P2.L < T2020]
    S['cm_linear_trend_2020on_dz'] = dict(
        value_m_yr=1e3 * ob, se_formal_m_yr=1e3 * obse,
        se_jackknife_fam=1e3 * float(np.sqrt((nF - 1) / nF * np.sum((jk - jk.mean()) ** 2))),
        placebo_sd_m_yr=float(P2.rate_m_yr.std()),
        placebo_pct_abs_all=float(np.mean(np.abs(P2.rate_m_yr) < abs(1e3 * ob)) * 100),
        placebo_pct_abs_end_before_2020=float(np.mean(np.abs(pre20.rate_m_yr) < abs(1e3 * ob)) * 100),
        placebo_sd_end_before_2020=float(pre20.rate_m_yr.std()))
    # predicted velocity shift vs depth (would a uniform dv/v produce a depth gradient?)
    S['dvv_predicted_shift']['corr_pred_dz_whole3d_vs_depth'] = float(np.corrcoef(PR['whole_3d_dz_per_1e-4'], PR.depth)[0, 1])
    S['dvv_predicted_shift']['slope_pred_dz_whole3d_vs_depth_m_per_km_per_0p01pct'] = float(np.polyfit(PR.depth, PR['whole_3d_dz_per_1e-4'], 1)[0])

    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    np.save(f'{OUT}/_cm_jack_dz.npy', R['dz']['CJ'])

    # ---------------- Figures
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for k, c in enumerate(COMP):
        r = R[c]
        ok = r['cnt'] > 0.5
        ax[k].fill_between(TC[ok], 1e3 * (r['C'] - r['C_se_jk'])[ok], 1e3 * (r['C'] + r['C_se_jk'])[ok],
                           color='C0', alpha=.3, label='common mode ±1 jackknife SE')
        ax[k].plot(TC[ok], 1e3 * r['C'][ok], 'C0.-', ms=3)
        tt = np.linspace(2018, 2026.5, 600)
        ax[k].plot(tt, 1e3 * cm_design(tt) @ r['p'], 'C3', lw=1.5, label='steps + ramp since 2020 (fit 2018+)')
        for q in QN:
            ax[k].axvline(TQ[q], color='k', ls=':', lw=1)
        ax[k].set_ylabel(f'{c} CM (m)')
        ax[k].grid(alpha=.3)
        ax[k].set_ylim(np.percentile(1e3 * r['C'][ok], [1, 99]) + np.array([-15, 15]))
    ax[0].legend(fontsize=8, loc='lower left')
    ax[0].set_title(f'Common-mode centroid track of {nF} families (core stations; dz>0 deeper)')
    ax[-1].set_xlabel('year')
    fig.tight_layout()
    fig.savefig('figures/sci_centroid_commonmode.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    o = S['pooled_raw_local_step']['M7_2024']['dz']
    sub = PL[~PL.near_quake]
    ax[0, 0].hist(PC.step_m, 40, color='0.6')
    for q, col in zip(QN, ['C1', 'C2', 'C3']):
        ax[0, 0].axvline(cml[q]['value_m'], color=col, lw=2, label=q)
    ax[0, 0].set_xlabel('common-mode dz step (m), random dates')
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title('CM step placebo (400 random dates)')
    ax[0, 1].hist(PT.dz_m_yr.dropna(), 40, color='0.6', alpha=.6, label='pooled per-family')
    ax[0, 1].hist(P2.rate_m_yr.dropna(), 40, color='C0', alpha=.4, label='CM cells')
    ax[0, 1].axvline(obs_tr['dz']['value_m_yr'], color='0.3', lw=2, label='pooled 2020-2026.5')
    ax[0, 1].axvline(1e3 * ob, color='C0', lw=2, label='CM 2020-2026.5')
    ax[0, 1].set_xlabel('dz trend (m/yr), random 4-6.5 yr windows')
    ax[0, 1].legend(fontsize=8)
    ax[0, 2].plot(PT.t0 + PT.L / 2, PT.dz_m_yr, '.', ms=3, color='0.5')
    ax[0, 2].axhline(obs_tr['dz']['value_m_yr'], color='C3')
    ax[0, 2].set_xlabel('window centre (yr)')
    ax[0, 2].set_ylabel('pooled dz trend (m/yr)')
    names = list(ccres)
    ax[1, 0].errorbar(range(len(names)), [ccres[n]['cm_rate2020_m_yr'] for n in names],
                      [ccres[n]['cm_rate2020_se_jk'] for n in names], fmt='o', label='CM ramp (jackknife)')
    ax[1, 0].errorbar(np.arange(len(names)) + .15, [ccres[n]['pooled_trend2020_m_yr'] for n in names],
                      [ccres[n]['pooled_trend2020_se'] for n in names], fmt='s', label='pooled linear')
    ax[1, 0].set_xticks(range(len(names)))
    ax[1, 0].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    ax[1, 0].axhline(0, color='k', lw=.5)
    ax[1, 0].set_ylabel('dz trend since 2020 (m/yr)')
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title('mean_cc reweighting')
    ax[1, 1].errorbar(range(len(names)), [ccres[n]['cm_step_M7_m'] for n in names],
                      [ccres[n]['cm_step_M7_se_jk'] for n in names], fmt='o', label='CM step')
    ax[1, 1].errorbar(np.arange(len(names)) + .15, [ccres[n]['pooled_M7_step_m'] for n in names],
                      [ccres[n]['pooled_M7_step_se'] for n in names], fmt='s', label='pooled local')
    ax[1, 1].set_xticks(range(len(names)))
    ax[1, 1].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    ax[1, 1].axhline(0, color='k', lw=.5)
    ax[1, 1].set_ylabel('dz M7 step (m)')
    ax[1, 1].legend(fontsize=7)
    x = np.arange(len(LO))
    ax[1, 2].plot(x, LO.cm_step_M7_m, 'o', label='CM M7 step (m)')
    ax[1, 2].plot(x, LO.cm_rate2020_m_yr, 's', label='CM ramp 2020+ (m/yr)')
    ax[1, 2].set_xticks(x)
    ax[1, 2].set_xticklabels(LO.dropped, rotation=90, fontsize=7)
    ax[1, 2].axhline(0, color='k', lw=.5)
    ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_title('leave-one-station-out (dz)')
    fig.tight_layout()
    fig.savefig('figures/sci_centroid_nulltests.png', dpi=130)
    plt.close(fig)

    SL = pd.read_csv(f'{OUT}/family_steps_local.csv')
    sl = SL[(SL.quake == 'M7_2024') & (SL.comp == 'dz')]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for k, key in enumerate(['E', 'depth', 'dist_m7']):
        ax[0, k].errorbar(sl[key], sl.diff_step_m, sl.diff_se_m, fmt='o')
        for r in sl.itertuples():
            ax[0, k].annotate(str(r.F), (getattr(r, key), r.diff_step_m), fontsize=7)
        g = S['differential_local_step']['M7_2024']['dz']['gradients'][key]
        ax[0, k].set_title(f"slope {g['slope_m_per_km']:.2f}±{g['se']:.2f} m/km, p_perm={g['p_perm']:.2f}", fontsize=9)
        ax[0, k].set_xlabel(key + ' (km)')
        ax[0, k].axhline(0, color='k', lw=.5)
    ax[0, 0].set_ylabel('differential M7 dz step (m)')
    fr = FR[FR.n_bins >= 12]
    for k, (nm, key) in enumerate([('strike', 'along_strike'), ('dip', 'down_dip'), ('normal', 'along_strike')]):
        ax[1, k].errorbar(fr[key], fr[f'rate_{nm}_m_yr'], fr[f'rate_{nm}_se'], fmt='o')
        for r in fr.itertuples():
            ax[1, k].annotate(str(r.F), (getattr(r, key), getattr(r, f'rate_{nm}_m_yr')), fontsize=7)
        ax[1, k].set_xlabel(key + ' position (km)')
        ax[1, k].set_ylabel(f'differential {nm} rate (m/yr)')
        ax[1, k].axhline(0, color='k', lw=.5)
    fig.suptitle('Differential (common-mode removed) M7 steps and 2008-2026 migration rates')
    fig.tight_layout()
    fig.savefig('figures/sci_centroid_spatial.png', dpi=130)
    plt.close(fig)
    print(json.dumps(S, indent=1, default=float)[:20000])


if __name__ == '__main__':
    main()
