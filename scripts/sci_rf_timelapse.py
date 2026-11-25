#!/usr/bin/env python3
"""Stage 5: time-lapse receiver functions at matched geometry.
Per station (QC-selected, Ps-moveout-corrected radial RFs, sci_rf_common.load_rf):
  Matched-geometry comparison of two event groups A (later) and B (earlier):
    geometry bins = 30-deg back-azimuth sectors x 2 ray-parameter classes (p < / >= 0.06 s/km);
    only bins sampled by both groups; group stacks = bin means combined with weights nA*nB/(nA+nB)
    (identical geometry weighting for A and B, so back-azimuth / ray-parameter sampling differences cancel).
  Measurements on the stacks (A relative to B), per window:
    P0   [-0.4, 0.4] s direct P: amplitude ratio (radial/vertical P amplitude -> shallow S velocity) and delay
    sh   [0.15, 1.0] s shallow part (a=2.5 and a=5): joint delay + amplitude ratio by Gauss-Newton least squares
         (a(t) ~ c b(t - tau), Fourier shifts; tau limited to +-150 ms)
    w1 [1.0, 2.5], w2 [2.5, 4.0], w3 [4.0, 6.0] s (intracrustal / slab-top low-velocity layer / slab Moho)
    phase windows +-0.5 s around the static picks of the slab phases (if picked)
    stretch over [0.15, 6] s: eps with A(t) ~ B(t (1+eps)); eps = -dv/v for a uniform velocity change
  Contrasts: step at each quake (B = 3 yr before, A = after until the next quake or 3 yr), long-term
    (after 2021-12-20 vs before), 2-year bins (each bin vs all other events) + linear trend.
  Uncertainty / nulls: bootstrap over events within group-bins (SE); permutation null (group labels shuffled
    within geometry bins, preserves the geometry); placebo dates (fake quakes, same window lengths, away from the
    real quakes). z = estimate / max(permutation SD, placebo SD). Detection limit = 2 x null SD.
  Injection: +20 ms delay of the RF after 0.25 s and +10% amplitude (whole RF) added to group A; recovery.
-> data/sci/rf_timelapse/timelapse_{STA}.csv, bins_{STA}.csv, figures/sci_rf_timelapse_{STA}.png
Usage: sci_rf_timelapse.py [--stations pilot|all|A,B] [--nperm 300] [--nboot 200] [--nplacebo 100]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import os
import sys
import warnings
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import (OUT, FIG, STATIONS, PILOT, QUAKES, QUAKE_LONG, BREAKS, SWS_EXCLUDE, load_rf,  # noqa: E402
                           P_REF)

warnings.filterwarnings('ignore')
YR = 365.25
WINDOWS = [('P0', 2.5, -0.4, 0.4), ('sh', 2.5, 0.15, 1.0), ('sh_a5', 5.0, 0.15, 1.0),
           ('w1', 2.5, 1.0, 2.5), ('w2', 2.5, 2.5, 4.0), ('w3', 2.5, 4.0, 6.0)]
WLABEL = {'P0': 'direct P (-0.4 to 0.4 s)', 'sh': 'shallow 0.15-1 s', 'sh_a5': 'shallow 0.15-1 s (a=5)',
          'w1': '1-2.5 s', 'w2': '2.5-4 s', 'w3': '4-6 s', 'ph_t': 'slab-top phase', 'ph_c': 'low-velocity-layer base phase',
          'ph_m': 'slab-Moho phase', 'stretch': 'stretch 0.15-6 s'}
STRETCH = (2.5, 0.15, 6.0)
MAXLAG = 0.15
UP = 10


class Station:
    def __init__(self, code):
        self.code = code
        m, rf, _, t, g, mall = load_rf(code)
        self.m, self.t, self.g, self.mall = m, t, g, mall
        self.dt = t[1] - t[0]
        self.R = {a: rf[:, g.index(a), 0].astype(np.float64) for a in g}
        self.T1 = rf[:, g.index(1.0), 1].astype(np.float64)
        self.time = m.time.values.astype('datetime64[s]').astype(np.int64) / 86400.0   # days
        self.geo = ((m.baz.values // 30).astype(int) * 2 + (m.p.values >= P_REF).astype(int))
        self.win = list(WINDOWS)
        sp = os.path.join(OUT, 'static_picks.csv')
        if os.path.exists(sp):
            P = pd.read_csv(sp)
            P = P[(P.sta == code) & P.phase.isin(['t', 'c', 'm'])]
            for r in P.itertuples():
                if np.isfinite(r.t_obs) and abs(r.amp) > 2 * r.amp_se and r.t_obs > 1.0:
                    self.win.append((f'ph_{r.phase}', 2.5, r.t_obs - 0.5, r.t_obs + 0.5))

    # ------------------------------------------------------------------ estimator
    def stacks(self, ia, ib, R=None):
        """matched-geometry stacks for index arrays ia, ib -> {a: (SA, SB)}, n_used_A, n_used_B"""
        ga, gb = self.geo[ia], self.geo[ib]
        common = np.intersect1d(ga, gb)
        if len(common) == 0:
            return None, 0, 0
        R = R or self.R
        out = {}
        na = nb = 0
        W = []
        for c in common:
            ka, kb = ia[ga == c], ib[gb == c]
            W.append((ka, kb, len(ka) * len(kb) / (len(ka) + len(kb))))
            na += len(ka); nb += len(kb)
        wsum = sum(w for _, _, w in W)
        for a in (1.0, 2.5, 5.0):
            SA = sum(w * R[a][ka].mean(0) for ka, kb, w in W) / wsum
            SB = sum(w * R[a][kb].mean(0) for ka, kb, w in W) / wsum
            out[a] = (SA, SB)
        out['T1'] = (sum(w * self.T1[ka].mean(0) for ka, kb, w in W) / wsum,
                     sum(w * self.T1[kb].mean(0) for ka, kb, w in W) / wsum)
        return out, na, nb

    def measure(self, S):
        t = self.t
        res = {}
        for name, a, t0, t1 in self.win:
            SA, SB = S[a]
            c, tau = shift_amp(SA, SB, t, t0, t1)
            res[f'{name}_amp'] = 100 * (c - 1)
            res[f'{name}_dt'] = 1000 * tau
        a, t0, t1 = STRETCH
        SA, SB = S[a]
        res['stretch_eps'] = 100 * stretch(SA, SB, t, t0, t1)
        # direct-P polarisation (a=1, |t|<=0.2 s): angle of the transverse vs radial P amplitude. A horizontal
        # sensor rotation by d changes it by ~d for every back-azimuth (orientation monitor)
        w = np.abs(t) <= 0.2
        (RA, RB), (TA, TB) = S[1.0], S['T1']
        res['Ppol_deg'] = np.degrees(np.arctan2(TA[w].mean(), RA[w].mean()) - np.arctan2(TB[w].mean(), RB[w].mean()))
        return res

    def compare(self, ia, ib, R=None):
        S, na, nb = self.stacks(ia, ib, R)
        if S is None or na < 3 or nb < 3:
            return None, na, nb
        return self.measure(S), na, nb

    # ------------------------------------------------------------------ nulls
    def bootstrap(self, ia, ib, rng, nb):
        out = []
        for _ in range(nb):
            ra = np.concatenate([rng.choice(ia[self.geo[ia] == c], (self.geo[ia] == c).sum()) for c in np.unique(self.geo[ia])])
            rb = np.concatenate([rng.choice(ib[self.geo[ib] == c], (self.geo[ib] == c).sum()) for c in np.unique(self.geo[ib])])
            r, _, _ = self.compare(ra, rb)
            if r:
                out.append(r)
        return pd.DataFrame(out)

    def permute(self, ia, ib, rng, nperm):
        allidx = np.concatenate([ia, ib]); lab = np.r_[np.ones(len(ia), bool), np.zeros(len(ib), bool)]
        geo = self.geo[allidx]
        out = []
        for _ in range(nperm):
            L = lab.copy()
            for c in np.unique(geo):
                k = np.where(geo == c)[0]
                L[k] = rng.permutation(L[k])
            r, _, _ = self.compare(allidx[L], allidx[~L])
            if r:
                out.append(r)
        return pd.DataFrame(out)


def fshift(x, tau, dt):
    """x(t - tau) by Fourier phase shift (tau > 0 delays)."""
    n = len(x)
    nf = 2 * n
    f = np.fft.rfftfreq(nf, dt)
    return np.fft.irfft(np.fft.rfft(x, nf) * np.exp(-2j * np.pi * f * tau), nf)[:n]


def fderiv(x, dt):
    n = len(x)
    nf = 2 * n
    f = np.fft.rfftfreq(nf, dt)
    return np.fft.irfft(np.fft.rfft(x, nf) * (2j * np.pi * f), nf)[:n]


def shift_amp(a, b, t, t0, t1, niter=5):
    """Joint amplitude ratio c and delay tau of a relative to b in [t0, t1]: a(t) ~ c b(t - tau).
    Gauss-Newton on the linearised model a = c b_s - c dtau b_s' (b_s = b shifted by the current tau);
    tau limited to +-MAXLAG."""
    dt = t[1] - t[0]
    w = (t >= t0) & (t <= t1)
    tau = 0.0
    c = 1.0
    for _ in range(niter):
        bs = fshift(b, tau, dt)
        bd = fderiv(bs, dt)
        G = np.c_[bs[w], -bd[w]]
        sol, *_ = np.linalg.lstsq(G, a[w], rcond=None)
        c = sol[0]
        if abs(c) < 1e-9:
            break
        dtau = sol[1] / c
        tau = float(np.clip(tau + dtau, -MAXLAG, MAXLAG))
        if abs(dtau) < 1e-5:
            break
    return c, tau


def stretch(a, b, t, t0, t1, emax=0.03, n=121):
    """eps such that a(t) ~ b(t / (1+eps)) (a later arrivals for eps>0 -> slower medium: dv/v ~ -eps)."""
    w = (t >= t0) & (t <= t1)
    eps = np.linspace(-emax, emax, n)
    cc = []
    for e in eps:
        bs = np.interp(t[w] / (1 + e), t, b)
        cc.append(np.dot(a[w], bs) / np.sqrt(np.dot(a[w], a[w]) * np.dot(bs, bs) + 1e-30))
    cc = np.array(cc)
    k = np.argmax(cc)
    if 0 < k < n - 1:
        den = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        dk = 0.5 * (cc[k - 1] - cc[k + 1]) / den if den != 0 else 0.0
    else:
        dk = 0.0
    return eps[k] + dk * (eps[1] - eps[0])


def day(s):
    return pd.Timestamp(s).value / 86400e9


def contrasts(S):
    """list of (name, A index, B index)"""
    tt = S.time
    t_end = tt.max()
    q = {k: day(v) for k, v in QUAKES.items()}
    qs = sorted(q.items(), key=lambda x: x[1])
    out = []
    for i, (k, tq) in enumerate(qs):
        nxt = qs[i + 1][1] if i + 1 < len(qs) else np.inf
        a_end = min(tq + 3 * YR, nxt)
        ia = np.where((tt >= tq) & (tt < a_end))[0]
        ib = np.where((tt >= tq - 3 * YR) & (tt < tq))[0]
        out.append((f'step_{k}', ia, ib, (tq, a_end)))
    tq = q['Petrolia']
    out.append(('longterm_post2021', np.where(tt >= tq)[0], np.where(tt < tq)[0], (tq, t_end)))
    return out


def placebo_dates(S, length_a, rng, n, exclude_days=365):
    tt = S.time
    q = np.array([day(v) for v in QUAKES.values()])
    lo, hi = tt.min() + YR, tt.max() - length_a
    if hi <= lo:
        return []
    cand = np.arange(lo, hi, 15.0)
    ok = [c for c in cand if np.all((np.abs(c - q) > exclude_days) & ~((q > c) & (q < c + length_a)) &
                                    ~((q < c) & (q > c - 3 * YR)))]
    if not ok:
        return []
    return list(rng.choice(ok, min(n, len(ok)), replace=False))


def inject(S, ia, kind):
    """copy of the RF dict with group A modified."""
    R = {a: x.copy() for a, x in S.R.items()}
    t = S.t
    if kind == 'delay20ms':
        ramp = np.clip((t - 0.05) / 0.2, 0, 1)
        ts = t - 0.020 * ramp
        for a in R:
            for k in ia:
                R[a][k] = np.interp(ts, t, S.R[a][k])
    elif kind == 'amp10pct':
        for a in R:
            R[a][ia] *= 1.10
    elif kind == 'stretch0.5pct':
        for a in R:
            for k in ia:
                R[a][k] = np.where(t > 0, np.interp(t / 1.005, t, S.R[a][k]), S.R[a][k])
    return R


def run_station(code, a):
    rng = np.random.default_rng(zlib.crc32(code.encode()))
    S = Station(code)
    n = len(S.m)
    print(code, 'n', n, 'windows', [w[0] for w in S.win], flush=True)
    rows = []
    for name, ia, ib, (ta0, ta1) in contrasts(S):
        r, na, nb = S.compare(ia, ib)
        base = dict(sta=code, contrast=name, nA=len(ia), nB=len(ib), nA_used=na, nB_used=nb)
        brk = [b for b in BREAKS.get(code, []) if (day(b) > ta0 - 3 * YR) and (day(b) < ta1)]
        base['breaks_in_window'] = ';'.join(brk)
        if r is None:
            rows.append(base); continue
        bs = S.bootstrap(ia, ib, rng, a.nboot)
        pm = S.permute(ia, ib, rng, a.nperm)
        # placebo dates
        pl = []
        if name.startswith('step'):
            for d in placebo_dates(S, ta1 - ta0, rng, a.nplacebo):
                ja = np.where((S.time >= d) & (S.time < d + (ta1 - ta0)))[0]
                jb = np.where((S.time >= d - 3 * YR) & (S.time < d))[0]
                rr, _, _ = S.compare(ja, jb)
                if rr:
                    pl.append(rr)
        pl = pd.DataFrame(pl)
        for q, v in r.items():
            row = dict(base, quantity=q, est=v, boot_se=bs[q].std() if len(bs) else np.nan,
                       perm_sd=pm[q].std() if len(pm) else np.nan, perm_mean=pm[q].mean() if len(pm) else np.nan,
                       placebo_sd=pl[q].std() if len(pl) > 10 else np.nan, n_placebo=len(pl))
            sd = np.nanmax([row['perm_sd'], row['placebo_sd']])
            row['null_sd'] = sd
            row['z'] = v / sd if sd > 0 else np.nan
            row['p_perm'] = (np.sum(np.abs(pm[q] - pm[q].mean()) >= abs(v)) + 1) / (len(pm) + 1) if len(pm) else np.nan
            row['det2sigma'] = 2 * sd
            rows.append(row)
        # injection tests (for the M7 step and the long-term contrast)
        if name in ('step_M7', 'longterm_post2021', 'step_Ferndale', 'step_Petrolia'):
            for kind in ('delay20ms', 'amp10pct', 'stretch0.5pct'):
                RI = inject(S, ia, kind)
                ri, _, _ = S.compare(ia, ib, RI)
                for row in rows:
                    if row.get('contrast') == name and 'quantity' in row and ri:
                        row[f'inj_{kind}'] = ri[row['quantity']] - row['est']
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, f'timelapse_{code}.csv'), index=False)
    # ---------------------------------------------------------------- 2-year bins vs rest + trend
    edges = [day(f'{y}-01-01') for y in range(2008, 2028, 2)]
    brow = []
    binstacks = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        ia = np.where((S.time >= e0) & (S.time < e1))[0]
        ib = np.where((S.time < e0) | (S.time >= e1))[0]
        if len(ia) < 5:
            continue
        r, na, nb = S.compare(ia, ib)
        if r is None:
            continue
        bs = S.bootstrap(ia, ib, rng, max(a.nboot // 2, 50))
        St, _, _ = S.stacks(ia, ib)
        binstacks.append((e0, e1, St[2.5][0], St[2.5][1], na))
        d = dict(sta=code, t0=pd.Timestamp(e0 * 86400e9), t1=pd.Timestamp(e1 * 86400e9), tmid=(e0 + e1) / 2, n=len(ia), nA_used=na)
        for q, v in r.items():
            d[q] = v; d[q + '_se'] = bs[q].std()
        brow.append(d)
    B = pd.DataFrame(brow)
    # trend (per year) with permutation null (event times shuffled within geometry bins)
    trend = {}
    if len(B) >= 3:
        qs = [c for c in B.columns if c + '_se' in B.columns]

        def slope(df):
            out = {}
            for q in qs:
                x = (df.tmid.values - df.tmid.mean()) / YR
                w = 1 / np.maximum(df[q + '_se'].values, 1e-9) ** 2 if q + '_se' in df else np.ones(len(df))
                out[q] = np.sum(w * x * (df[q].values - np.average(df[q].values, weights=w))) / np.sum(w * x * x)
            return out
        trend = slope(B)
        perm = []
        geo = S.geo
        for _ in range(max(a.nperm // 3, 60)):
            tp = S.time.copy()
            for c in np.unique(geo):
                k = np.where(geo == c)[0]
                tp[k] = rng.permutation(tp[k])
            rr = []
            for e0, e1 in zip(edges[:-1], edges[1:]):
                ia = np.where((tp >= e0) & (tp < e1))[0]
                ib = np.where((tp < e0) | (tp >= e1))[0]
                if len(ia) < 5:
                    continue
                x, _, _ = S.compare(ia, ib)
                if x:
                    x['tmid'] = (e0 + e1) / 2
                    rr.append(x)
            if len(rr) >= 3:
                D = pd.DataFrame(rr)
                for q in qs:
                    D[q + '_se'] = B[q + '_se'].median()
                perm.append(slope(D))
        perm = pd.DataFrame(perm)
        for q in qs:
            sd = perm[q].std() if len(perm) else np.nan
            T = pd.concat([T, pd.DataFrame([dict(sta=code, contrast='trend_per_yr', quantity=q, est=trend[q],
                                                 perm_sd=sd, null_sd=sd, z=trend[q] / sd if sd > 0 else np.nan,
                                                 det2sigma=2 * sd, nA=n)])], ignore_index=True)
        T.to_csv(os.path.join(OUT, f'timelapse_{code}.csv'), index=False)
    B.to_csv(os.path.join(OUT, f'bins_{code}.csv'), index=False)
    np.savez_compressed(os.path.join(OUT, f'binstacks_{code}.npz'), t=S.t,
                        e0=np.array([b[0] for b in binstacks]), e1=np.array([b[1] for b in binstacks]),
                        SA=np.array([b[2] for b in binstacks]), SB=np.array([b[3] for b in binstacks]),
                        n=np.array([b[4] for b in binstacks]))
    figure(S, T, B, binstacks)
    return T


SHOW_Q = ['P0_amp', 'Ppol_deg', 'sh_dt', 'sh_amp', 'sh_a5_dt', 'sh_a5_amp', 'w1_dt', 'w1_amp', 'w2_dt', 'w2_amp',
          'w3_dt', 'w3_amp', 'stretch_eps']
CONS = ['step_Petrolia', 'step_Ferndale', 'step_M7', 'longterm_post2021', 'trend_per_yr']
CONS_LABEL = ['Petrolia step', 'Ferndale step', 'M7 step', 'after vs before 2021-12-20', 'linear trend (per year)']
UNITS = {'dt': 'ms', 'amp': '%', 'eps': '%', 'deg': 'deg'}


def qlabel(q):
    base, kind = q.rsplit('_', 1)
    what = {'dt': 'delay', 'amp': 'amplitude', 'eps': '', 'deg': ''}[kind]
    if q == 'Ppol_deg':
        return 'direct-P polarisation angle (deg)'
    return f'{WLABEL.get(base, base)} {what} ({UNITS[kind]})'.replace('  ', ' ')


def figure(S, T, B, binstacks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    code = S.code
    qcol = {'Petrolia': 'darkorange', 'Ferndale': 'm', 'M7': 'r'}
    fig = plt.figure(figsize=(18, 15))
    tdt = pd.to_datetime(S.m.time.values)
    xlim = (pd.Timestamp('2008-01-01'), pd.Timestamp('2026-12-31'))

    def qlines(ax):
        for k, v in QUAKES.items():
            ax.axvline(pd.Timestamp(v), color=qcol[k], lw=1, ls='--')
        for b in BREAKS.get(code, []):
            ax.axvline(pd.Timestamp(b), color='0.5', lw=0.8, ls=':')
        for a0, a1 in SWS_EXCLUDE.get(code, []):
            ax.axvspan(max(pd.Timestamp(a0), xlim[0]), min(pd.Timestamp(a1), xlim[1]), color='0.9', zorder=0)
        ax.set_xlim(xlim)
    ax = fig.add_axes([0.05, 0.77, 0.40, 0.16])
    sc = ax.scatter(tdt, S.m.baz, c=S.m.p, s=8, cmap='viridis', vmin=0.04, vmax=0.08)
    plt.colorbar(sc, ax=ax, pad=0.01).set_label('ray parameter (s/km)')
    qlines(ax); ax.set_ylabel('back-azimuth (deg)'); ax.set_ylim(0, 360)
    ax.set_title(f'Accepted events (n={len(S.m)}). Dashed: Petrolia (orange), Ferndale (magenta), M7 (red);\n'
                 'dotted: instrument epochs; grey: excluded instrument-problem windows', fontsize=9)
    qsets = [('dt', 'delay (ms)', ['sh', 'sh_a5', 'w1', 'w2', 'w3']),
             ('amp', 'amplitude change (%)', ['P0', 'sh', 'w1', 'w2', 'w3'])]
    for i, (suf, lab, names) in enumerate(qsets):
        ax = fig.add_axes([0.05, 0.535 - 0.215 * i, 0.40, 0.16])
        if len(B):
            for j, nm in enumerate(names):
                q = f'{nm}_{suf}'
                if q not in B:
                    continue
                x = pd.to_datetime(B.tmid * 86400e9) + pd.Timedelta(days=25 * (j - 2))
                ax.errorbar(x, B[q], yerr=2 * B[q + '_se'], fmt='o-', ms=3, lw=0.8, capsize=2, label=WLABEL[nm])
        qlines(ax); ax.axhline(0, color='k', lw=0.5); ax.set_ylabel(lab)
        ax.set_ylim({'dt': (-200, 200), 'amp': (-100, 100)}[suf])
        ax.legend(fontsize=7, ncol=3, loc='lower left')
        ax.set_title(f'2-year bins vs all other events at matched geometry (2-sigma bootstrap): {lab}', fontsize=9)
    for k, (xr, sp, title) in enumerate([((-1, 6), 0.5, '2-year bin stacks (black) vs all other events at\n'
                                          'matched geometry (grey), radial, Gaussian a=2.5'),
                                         ((-0.5, 1.5), 0.3, 'zoom on -0.5 to 1.5 s')]):
        ax = fig.add_axes([0.52 + 0.25 * k, 0.32, 0.19, 0.61])
        for j, (e0, e1, SA, SB, na) in enumerate(binstacks):
            off = -sp * j
            w = (S.t >= xr[0]) & (S.t <= xr[1])
            ax.plot(S.t[w], off + SB[w], color='0.6', lw=1); ax.plot(S.t[w], off + SA[w], 'k', lw=1)
            if k == 0:
                ax.text(xr[1] + 0.05, off, f'{pd.Timestamp(e0 * 86400e9):%Y}-{pd.Timestamp(e1 * 86400e9 - 1):%Y}\nn={na}',
                        fontsize=7, va='center')
        ax.set_yticks([]); ax.set_xlabel('time after P (s)'); ax.set_xlim(xr); ax.set_title(title, fontsize=9)
    ax = fig.add_axes([0.17, 0.10, 0.70, 0.13])
    show = SHOW_Q + [q for q in T.quantity.dropna().unique() if q.startswith('ph_')]
    Z = np.full((len(CONS), len(show)), np.nan)
    for j, q in enumerate(show):
        for i, c in enumerate(CONS):
            r = T[(T.quantity == q) & (T.contrast == c)]
            if len(r) and np.isfinite(r.est.iloc[0]):
                Z[i, j] = r.z.iloc[0]
                ax.text(j, i, f'{r.est.iloc[0]:+.1f}\n(±{r.det2sigma.iloc[0]:.1f})', ha='center', va='center', fontsize=6.5)
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
    ax.set_yticks(range(len(CONS))); ax.set_yticklabels(CONS_LABEL, fontsize=8)
    ax.set_xticks(range(len(show))); ax.set_xticklabels([qlabel(q) for q in show], fontsize=7, rotation=25, ha='right')
    plt.colorbar(im, ax=ax, pad=0.01).set_label('z = estimate / null SD')
    ax.set_title('Estimate (± 2-sigma detection limit from permutation / placebo-date nulls); colour = z', fontsize=9)
    fig.suptitle(f'{code}: time-lapse P receiver functions compared at matched back-azimuth / ray-parameter sampling',
                 fontsize=12, y=0.985)
    fig.savefig(os.path.join(FIG, f'sci_rf_timelapse_{code.split(".")[1]}.png'), dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations', default='pilot')
    ap.add_argument('--nperm', type=int, default=300)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--nplacebo', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    sts = PILOT if a.stations == 'pilot' else (list(STATIONS) if a.stations == 'all' else a.stations.split(','))
    sts = [s for s in sts if os.path.exists(os.path.join(OUT, f'rf_{s}.npz'))]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(a.workers, len(sts))) as ex:
        res = list(ex.map(run_station, sts, [a] * len(sts)))
    print(pd.concat(res).round(3).to_string())


if __name__ == '__main__':
    main()
