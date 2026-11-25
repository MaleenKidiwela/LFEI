#!/usr/bin/env python3
"""Stage 7: spectral-power time-lapse receiver functions after Herath & Audet (2024, Research Square,
doi 10.21203/rs.3.rs-3953181/v1), adapted to the MTJ stations.
  RFs: modified Wiener deconvolution (Audet 2010 BSSA; sci_rf_compute.wiener_mod), radial and transverse.
  Event set ('power set'): Z SNR >= 3, outside instrument-problem windows, not clipped, positive direct P on
    the radial Wiener RF, RMS outliers (> median + 4 MAD) removed.  (No deconvolution-fit criterion: the Wiener RF
    has no fit; this is close to their 'all M>5 with SNR' selection.)
  Lag windows per station: slab = [t_Ps(top of low-velocity layer) - 0.5, t_Ps(slab Moho) + 0.5] s from the
    Bloch et al. 2023 model (static_picks.csv predictions; regional default 2-4 s if outside the model);
    control: crust [0.5, min(2.5, t_top - 0.5)] s (at least 1 s long); noise: pre-P lags [-4.5, -0.5] s.
  PSD: multitaper (scipy dpss, NW=2, K=3), averaged in 1/8-octave bands 0.25-5 Hz; band-summed 'total power'
    over 1.25-3 Hz (periods 0.33-0.8 s, the band of change of Herath & Audet).
  Time series: 1.5-yr windows with 95 % overlap (step 0.075 yr), PSD stacked with back-azimuth / ray-parameter
    reweighting to the station's all-time geometry distribution (and unweighted for comparison).
  Non-overlapping contrasts (steps at the three quakes, after vs before 2021-12-20, 2-yr bins, trend) at matched
    geometry (same bins/weights as sci_rf_timelapse), in dB; null = event-label permutation within geometry
    bins + placebo dates.
  External comparisons (overlapping series): LFE rate (merged Shelly catalogue, all families), LFE coda dv/v
    (data/dvv/pool_clean, station mean of components), detrended GNSS east displacement at the nearest NGL station
    (data/sci/gnss/near_clean.csv.gz).  Significance of Pearson r from the event-time permutation null (the
    overlapping-window autocorrelation is reproduced in each permutation).
  Modelling with PyRaysum (perturbing Vs of the slab low-velocity layer) is NOT done: PyRaysum is not installed.
-> data/sci/rf_timelapse/power_{STA}.csv (contrasts), power_series_{STA}.csv, power_corr.csv,
   figures/sci_rf_power_{STA}.png, figures/sci_rf_power_overview.png
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import (OUT, FIG, STATIONS, PILOT, QUAKES, BREAKS, SWS_EXCLUDE, P_REF, ROOT, coords)  # noqa: E402

YR = 365.25
FB = 0.25 * 2 ** (np.arange(0, 8 * np.log2(5 / 0.25) + 1) / 8)       # 1/8-octave band centres, 0.25-5 Hz
BAND = (1.25, 3.0)
WIN_LEN, WIN_STEP = 1.5 * YR, 0.075 * YR
MIN_EV = 8
COMPS = ['R', 'T']
WNAMES = ['slab', 'crust', 'noise']
WLAB = {'slab': 'slab window', 'crust': 'crustal control window', 'noise': 'pre-P noise window'}


def mt_psd(x, dt, nw=2.0, k=3, nfft=512):
    from scipy.signal.windows import dpss
    tap = dpss(len(x), nw, k)
    X = np.fft.rfft((x - x.mean())[None, :] * tap, nfft, axis=1)
    f = np.fft.rfftfreq(nfft, dt)
    return f, 2 * dt * np.mean(np.abs(X) ** 2, 0)


def band_avg(f, p):
    out = np.empty(len(FB))
    for i, c in enumerate(FB):
        k = (f >= c * 2 ** (-1 / 16)) & (f < c * 2 ** (1 / 16))
        out[i] = p[k].mean() if k.any() else np.interp(c, f, p)
    return out


def windows_for(code):
    sp = os.path.join(OUT, 'static_picks.csv')
    tt, tm = np.nan, np.nan
    if os.path.exists(sp):
        P = pd.read_csv(sp)
        r = P[(P.sta == code) & (P.phase == 't')]
        if len(r):
            tt = r.t_pred.iloc[0]
        r = P[(P.sta == code) & (P.phase == 'm')]
        if len(r):
            tm = r.t_pred.iloc[0]
    src = 'Bloch et al. 2023'
    if not np.isfinite(tt) or not np.isfinite(tm):
        tt, tm, src = 2.0, 4.0, 'regional default'
    slab = (max(tt - 0.5, 0.8), tm + 0.5)
    crust = (0.5, max(1.5, min(2.5, tt - 0.5)))
    return {'slab': slab, 'crust': crust, 'noise': (-4.5, -0.5)}, src


class PStation:
    def __init__(self, code):
        self.code = code
        d = np.load(os.path.join(OUT, f'rf_{code}.npz'))
        t = d['t']; rfw = d['rfw']
        m = pd.DataFrame(json.loads(str(d['meta'])))
        m['time'] = pd.to_datetime(m['time'])
        p0 = rfw[:, 0, (t >= -0.3) & (t <= 0.3)].max(1)
        rms = np.sqrt(np.mean(rfw[:, 0] ** 2, 1))
        keep = (m.snr_z >= 3) & (~m.excl.astype(bool)) & (p0 > 0.03)
        if 'clip' in m:
            keep &= ~m['clip'].fillna(False).astype(bool)
        med = np.median(rms[keep]); mad = 1.4826 * np.median(np.abs(rms[keep] - med))
        keep &= rms <= med + 4 * mad
        sel = np.where(keep.values)[0]
        self.m = m.iloc[sel].reset_index(drop=True)
        self.rfw = rfw[sel].astype(np.float64)
        self.t = t
        self.time = self.m.time.values.astype('datetime64[s]').astype(np.int64) / 86400.0
        self.geo = ((self.m.baz.values // 30).astype(int) * 2 + (self.m.p.values >= P_REF).astype(int))
        self.win, self.win_src = windows_for(code)
        dt = t[1] - t[0]
        n = len(sel)
        self.psd = np.zeros((n, len(WNAMES), 2, len(FB)))
        for i in range(n):
            for j, wn in enumerate(WNAMES):
                a, b = self.win[wn]
                w = (t >= a) & (t <= b)
                for c in range(2):
                    f, p = mt_psd(self.rfw[i, c, w], dt)
                    self.psd[i, j, c] = band_avg(f, p)
        self.bmask = (FB >= BAND[0]) & (FB <= BAND[1])

    def total(self, psd_stack):
        return psd_stack[..., self.bmask].sum(-1)

    # matched-geometry contrast in dB (A vs B) for all windows / comps
    def compare(self, ia, ib):
        ga, gb = self.geo[ia], self.geo[ib]
        common = np.intersect1d(ga, gb)
        if len(common) == 0:
            return None
        W = []
        for c in common:
            ka, kb = ia[ga == c], ib[gb == c]
            W.append((ka, kb, len(ka) * len(kb) / (len(ka) + len(kb))))
        if sum(len(ka) for ka, _, _ in W) < 3 or sum(len(kb) for _, kb, _ in W) < 3:
            return None
        ws = sum(w for *_, w in W)
        PA = sum(w * self.psd[ka].mean(0) for ka, kb, w in W) / ws
        PB = sum(w * self.psd[kb].mean(0) for ka, kb, w in W) / ws
        return 10 * np.log10(self.total(PA) / self.total(PB))        # (nwin, 2)

    def series(self, times=None, reweight=True):
        tt = self.time if times is None else times
        ref = np.bincount(self.geo, minlength=24) / len(self.geo)
        c0 = np.arange(tt.min() + WIN_LEN / 2, tt.max() - WIN_LEN / 2 + 1e-6, WIN_STEP)
        out = np.full((len(c0), len(WNAMES), 2), np.nan)
        nn = np.zeros(len(c0), int)
        for i, c in enumerate(c0):
            k = np.where(np.abs(tt - c) <= WIN_LEN / 2)[0]
            nn[i] = len(k)
            if len(k) < MIN_EV:
                continue
            g = self.geo[k]
            if reweight:
                cnt = np.bincount(g, minlength=24)
                w = ref[g] / cnt[g]
                if np.sum(ref[np.unique(g)]) < 0.5:          # window samples < half of the reference geometry
                    continue
            else:
                w = np.ones(len(k))
            P = np.tensordot(w / w.sum(), self.psd[k], axes=(0, 0))
            out[i] = self.total(P)
        allP = self.total(self.psd.mean(0))
        return c0, 10 * np.log10(out / allP[None]), nn


def day(s):
    return pd.Timestamp(s).value / 86400e9


def contrasts(S):
    tt = S.time
    q = sorted([(k, day(v)) for k, v in QUAKES.items()], key=lambda x: x[1])
    out = []
    for i, (k, tq) in enumerate(q):
        nxt = q[i + 1][1] if i + 1 < len(q) else np.inf
        ia = np.where((tt >= tq) & (tt < min(tq + 3 * YR, nxt)))[0]
        ib = np.where((tt >= tq - 3 * YR) & (tt < tq))[0]
        out.append((f'step_{k}', ia, ib, min(tq + 3 * YR, nxt, tt.max()) - tq))
    tq = day(QUAKES['Petrolia'])
    out.append(('longterm_post2021', np.where(tt >= tq)[0], np.where(tt < tq)[0], None))
    return out


def permute_times(S, rng):
    tp = S.time.copy()
    for c in np.unique(S.geo):
        k = np.where(S.geo == c)[0]
        tp[k] = rng.permutation(tp[k])
    return tp


# ----------------------------------------------------------------------------- external series
_EXT = {}


def lfe_rate():
    if 'lfe' not in _EXT:
        d = pd.read_csv(os.path.join(ROOT, 'data/merged/LFEcat_MTJ_merged_v1v2.csv'), usecols=['time_utc'])
        t = pd.to_datetime(d.time_utc)
        _EXT['lfe'] = (t.values.astype('datetime64[s]').astype(np.int64) / 86400.0)
    return _EXT['lfe']


def dvv_station(code):
    fs = glob.glob(os.path.join(ROOT, f'data/dvv/pool_clean/{code}.*curve.csv'))
    if not fs:
        return None
    S = []
    for f in fs:
        d = pd.read_csv(f, parse_dates=['t'])
        d['m'] = d.t.dt.to_period('M')
        S.append(d.groupby('m').dvv_pct.mean())
    s = pd.concat(S, axis=1).mean(1)
    return (s.index.to_timestamp() + pd.Timedelta(days=15)).values.astype('datetime64[s]').astype(np.int64) / 86400.0, s.values


def gnss_nearest(code, C):
    if 'gnss' not in _EXT:
        g = pd.read_csv(os.path.join(ROOT, 'data/sci/gnss/near_clean.csv.gz'))
        st = pd.read_csv(os.path.join(ROOT, 'data/sci/gnss/stations.csv'))
        _EXT['gnss'] = (g, st)
    g, st = _EXT['gnss']
    la, lo = C[code][:2]
    st = st[st.sta.isin(g.sta.unique())].copy()
    st['d'] = np.hypot((st.lat - la) * 111.2, (st.lon - lo) * 111.2 * np.cos(np.radians(la)))
    s0 = STATIONS[code][0]
    for r in st.sort_values('d').itertuples():
        x = g[(g.sta == r.sta) & np.isfinite(g.e)]
        tt = x.mjd.values - 40587.0                     # MJD -> days since 1970
        if tt.min() <= day(max(s0, '2008-06-01')) + 2 * YR and tt.max() >= day('2025-06-01'):
            return r.sta, r.d, tt, x.e.values
    return None


def window_mean(tc, tx, x):
    out = np.full(len(tc), np.nan)
    for i, c in enumerate(tc):
        k = np.abs(tx - c) <= WIN_LEN / 2
        if k.sum() >= 3:
            out[i] = np.nanmean(x[k])
    return out


def externals(S, tc, C):
    ext = {}
    lt = lfe_rate()
    ext['LFE rate (events/day)'] = np.array([np.sum(np.abs(lt - c) <= WIN_LEN / 2) / WIN_LEN for c in tc])
    d = dvv_station(S.code)
    if d is not None:
        ext['LFE coda dv/v (%)'] = window_mean(tc, d[0], d[1])
    gn = gnss_nearest(S.code, C)
    if gn is not None:
        sta, dist, tx, e = gn
        k = (tx >= S.time.min()) & (tx <= S.time.max())
        A = np.c_[np.ones(k.sum()), tx[k] - tx[k].mean()]
        cf, *_ = np.linalg.lstsq(A, e[k], rcond=None)
        ed = e - (cf[0] + cf[1] * (tx - tx[k].mean()))
        ext[f'GNSS {sta} east, detrended (mm; {dist:.0f} km)'] = window_mean(tc, tx, ed)
    return ext


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[k], b[k])[0, 1])


# ----------------------------------------------------------------------------- main per station
def run_station(code, a):
    rng = np.random.default_rng(zlib.crc32(code.encode()) + 7)
    C = coords()
    S = PStation(code)
    n = len(S.m)
    print(code, 'power set', n, 'windows', S.win, S.win_src, flush=True)
    rows = []
    for name, ia, ib, la in contrasts(S):
        est = S.compare(ia, ib)
        if est is None:
            continue
        perm = []
        allidx = np.r_[ia, ib]; lab = np.r_[np.ones(len(ia), bool), np.zeros(len(ib), bool)]
        for _ in range(a.nperm):
            L = lab.copy()
            for c in np.unique(S.geo[allidx]):
                k = np.where(S.geo[allidx] == c)[0]
                L[k] = rng.permutation(L[k])
            r = S.compare(allidx[L], allidx[~L])
            if r is not None:
                perm.append(r)
        perm = np.array(perm)
        pl = []
        if name.startswith('step') and la and la > 30:
            q = np.array([day(v) for v in QUAKES.values()])
            cand = np.arange(S.time.min() + YR, S.time.max() - la, 15.0)
            cand = [c for c in cand if np.all(np.abs(c - q) > 365) and not np.any((q > c - 3 * YR) & (q < c + la))]
            for c in (rng.choice(cand, min(len(cand), a.nplacebo), replace=False) if cand else []):
                r = S.compare(np.where((S.time >= c) & (S.time < c + la))[0], np.where((S.time >= c - 3 * YR) & (S.time < c))[0])
                if r is not None:
                    pl.append(r)
        pl = np.array(pl)
        for j, wn in enumerate(WNAMES):
            for c, cn in enumerate(COMPS):
                psd_ = perm[:, j, c].std() if len(perm) else np.nan
                plsd = pl[:, j, c].std() if len(pl) > 10 else np.nan
                sd = np.nanmax([psd_, plsd])
                rows.append(dict(sta=code, contrast=name, window=wn, comp=cn, nA=len(ia), nB=len(ib), est_dB=est[j, c],
                                 perm_sd=psd_, placebo_sd=plsd, n_placebo=len(pl), null_sd=sd,
                                 z=est[j, c] / sd if sd > 0 else np.nan, det2sigma=2 * sd))
    T = pd.DataFrame(rows)
    # overlapping series + externals + permutation null for correlations / envelopes
    tc, ser, nn = S.series()
    _, ser_u, _ = S.series(reweight=False)
    ext = externals(S, tc, C)
    nulls = []
    for _ in range(a.nperm_series):
        _, sp, _ = S.series(times=permute_times(S, rng))
        nulls.append(sp)
    nulls = np.array(nulls)
    crow = []
    for j, wn in enumerate(WNAMES):
        for c, cn in enumerate(COMPS):
            y = ser[:, j, c]
            for en, ev in ext.items():
                r0 = corr(y, ev)
                rn = np.array([corr(nulls[i][:, j, c], ev) for i in range(len(nulls))])
                rn = rn[np.isfinite(rn)]
                p = (np.sum(np.abs(rn) >= abs(r0)) + 1) / (len(rn) + 1) if len(rn) and np.isfinite(r0) else np.nan
                crow.append(dict(sta=code, window=wn, comp=cn, external=en, r=r0, p_perm=p,
                                 null_r_sd=rn.std() if len(rn) else np.nan, n_windows=int(np.isfinite(y).sum())))
            # range of the series vs null range
            rng_obs = np.nanmax(y) - np.nanmin(y)
            rng_null = np.array([np.nanmax(x[:, j, c]) - np.nanmin(x[:, j, c]) for x in nulls])
            crow.append(dict(sta=code, window=wn, comp=cn, external='series range (dB)', r=rng_obs,
                             p_perm=(np.sum(rng_null >= rng_obs) + 1) / (len(rng_null) + 1), null_r_sd=np.nanmean(rng_null),
                             n_windows=int(np.isfinite(y).sum())))
    CR = pd.DataFrame(crow)
    SER = pd.DataFrame({'tmid': pd.to_datetime(tc * 86400e9), 'n': nn})
    for j, wn in enumerate(WNAMES):
        for c, cn in enumerate(COMPS):
            SER[f'{wn}_{cn}_dB'] = ser[:, j, c]; SER[f'{wn}_{cn}_dB_unweighted'] = ser_u[:, j, c]
            SER[f'{wn}_{cn}_null_lo'] = np.nanpercentile(nulls[:, :, j, c], 2.5, axis=0)
            SER[f'{wn}_{cn}_null_hi'] = np.nanpercentile(nulls[:, :, j, c], 97.5, axis=0)
    for en, ev in ext.items():
        SER[en] = ev
    T.to_csv(os.path.join(OUT, f'power_{code}.csv'), index=False)
    SER.to_csv(os.path.join(OUT, f'power_series_{code}.csv'), index=False)
    CR.to_csv(os.path.join(OUT, f'power_corr_{code}.csv'), index=False)
    figure(S, T, SER, CR, ext)
    return T, CR


def figure(S, T, SER, CR, ext):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    code = S.code
    qcol = {'Petrolia': 'darkorange', 'Ferndale': 'm', 'M7': 'r'}
    xlim = (pd.Timestamp('2008-01-01'), pd.Timestamp('2026-12-31'))
    fig = plt.figure(figsize=(18, 13))

    def ql(ax):
        for k, v in QUAKES.items():
            ax.axvline(pd.Timestamp(v), color=qcol[k], ls='--', lw=1)
        for b in BREAKS.get(code, []):
            ax.axvline(pd.Timestamp(b), color='0.5', ls=':', lw=0.8)
        for a0, a1 in SWS_EXCLUDE.get(code, []):
            ax.axvspan(max(pd.Timestamp(a0), xlim[0]), min(pd.Timestamp(a1), xlim[1]), color='0.92', zorder=0)
        ax.set_xlim(xlim)
    # mean PSDs
    ax = fig.add_axes([0.05, 0.60, 0.22, 0.33])
    for j, wn in enumerate(WNAMES):
        for c, cn in enumerate(COMPS):
            ax.loglog(FB, S.psd[:, j, c].mean(0), ['C0', 'C1', 'C7'][j], ls=['-', '--'][c], label=f'{WLAB[wn]} {S.win[wn][0]:.1f}-{S.win[wn][1]:.1f} s, {"radial" if cn == "R" else "transverse"}')
    ax.axvspan(*BAND, color='y', alpha=0.2)
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4])); ax.xaxis.set_major_formatter(FixedFormatter(['0.25', '0.5', '1', '2', '4']))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('mean PSD of the receiver function (1/Hz)')
    ax.legend(fontsize=6.5); ax.set_title(f'Mean multitaper PSD, 1/8-octave bands (n={len(S.m)});\nyellow: 1.25-3 Hz total-power band', fontsize=9)
    # mean RF
    ax = fig.add_axes([0.33, 0.60, 0.22, 0.33])
    for c, cn in enumerate(COMPS):
        ax.plot(S.t, S.rfw[:, c].mean(0) - 0.25 * c, ['k', 'C3'][c], lw=1, label=['radial', 'transverse (offset)'][c])
    for wn, col in zip(WNAMES, ['C0', 'C1', 'C7']):
        ax.axvspan(*S.win[wn], color=col, alpha=0.15)
    ax.set_xlim(-5, 10); ax.set_xlabel('time after P (s)'); ax.legend(fontsize=7)
    ax.set_title(f'Mean modified-Wiener RF; shaded: slab ({S.win_src}), crustal and noise windows', fontsize=9)
    # contrasts table
    ax = fig.add_axes([0.62, 0.60, 0.33, 0.33])
    cons = ['step_Petrolia', 'step_Ferndale', 'step_M7', 'longterm_post2021']
    cols = [(w, c) for w in WNAMES for c in COMPS]
    Z = np.full((len(cons), len(cols)), np.nan)
    for i, cn in enumerate(cons):
        for j, (w, c) in enumerate(cols):
            r = T[(T.contrast == cn) & (T.window == w) & (T.comp == c)]
            if len(r):
                Z[i, j] = r.z.iloc[0]
                ax.text(j, i, f'{r.est_dB.iloc[0]:+.2f}\n(±{r.det2sigma.iloc[0]:.2f})', ha='center', va='center', fontsize=7)
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
    ax.set_yticks(range(len(cons))); ax.set_yticklabels(['Petrolia step', 'Ferndale step', 'M7 step', 'after vs before\n2021-12-20'], fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'{w}\n{"radial" if c == "R" else "transverse"}' for w, c in cols], fontsize=7)
    plt.colorbar(im, ax=ax, pad=0.01).set_label('z')
    ax.set_title('1.25-3 Hz power change (dB, ± 2-sigma null) at matched geometry', fontsize=9)
    # overlapping series
    tm = pd.to_datetime(SER.tmid)
    for k, wn in enumerate(['slab', 'crust']):
        ax = fig.add_axes([0.05, 0.33 - 0.0 * k, 0.42, 0.20]) if k == 0 else fig.add_axes([0.53, 0.33, 0.42, 0.20])
        for c, cn in enumerate(COMPS):
            col = ['k', 'C3'][c]
            ax.fill_between(tm, SER[f'{wn}_{cn}_null_lo'], SER[f'{wn}_{cn}_null_hi'], color=col, alpha=0.12, lw=0)
            ax.plot(tm, SER[f'{wn}_{cn}_dB'], col, lw=1.5, label=f'{"radial" if cn == "R" else "transverse"}')
            ax.plot(tm, SER[f'{wn}_{cn}_dB_unweighted'], col, lw=0.7, ls=':')
        ax.plot(tm, SER['noise_R_dB'], 'C7', lw=1, label='pre-P noise window, radial')
        ql(ax); ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('total power (dB re all-event mean)')
        ax.legend(fontsize=7, loc='lower left')
        ax.set_title(f'{WLAB[wn]} {S.win[wn][0]:.1f}-{S.win[wn][1]:.1f} s: 1.5-yr windows, 95% overlap, geometry-reweighted '
                     '(dotted: unweighted);\nshading: 95% range of event-time permutations', fontsize=8.5)
    # externals
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.20])
    y = SER['slab_R_dB'].values
    ax.plot(tm, (y - np.nanmean(y)) / np.nanstd(y), 'k', lw=2, label='slab-window radial power')
    for i, (en, ev) in enumerate(ext.items()):
        r = CR[(CR.window == 'slab') & (CR.comp == 'R') & (CR.external == en)]
        lab = f'{en}: r={r.r.iloc[0]:+.2f}, permutation p={r.p_perm.iloc[0]:.2f}' if len(r) else en
        ax.plot(tm, (ev - np.nanmean(ev)) / np.nanstd(ev), f'C{i}', lw=1, label=lab)
    ql(ax); ax.set_ylabel('standardised'); ax.legend(fontsize=7, loc='upper left')
    ax.set_title('Slab-window radial power vs LFE rate, LFE coda dv/v and GNSS east (all averaged in the same 1.5-yr windows)', fontsize=9)
    fig.suptitle(f'{code}: receiver-function spectral power through time (Herath & Audet 2024 method; '
                 f'modified Wiener deconvolution, {len(S.m)} events)', fontsize=12)
    fig.savefig(os.path.join(FIG, f'sci_rf_power_{code.split(".")[1]}.png'), dpi=100)
    plt.close(fig)


def overview():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'power_[A-Z]*.csv')))
                   if 'series' not in f and 'corr' not in f], ignore_index=True)
    CR = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'power_corr_*.csv')))], ignore_index=True)
    T.to_csv(os.path.join(OUT, 'power_contrasts_all.csv'), index=False)
    CR.to_csv(os.path.join(OUT, 'power_corr.csv'), index=False)
    stas = [s for s in STATIONS if s in set(T.sta)]
    fig, axs = plt.subplots(1, 2, figsize=(18, max(6, 0.4 * len(stas) + 3)), gridspec_kw=dict(width_ratios=[1.6, 1]))
    cons = ['step_Petrolia', 'step_Ferndale', 'step_M7', 'longterm_post2021']
    cols = [(w, c, cn) for w in ['slab', 'crust', 'noise'] for c in COMPS for cn in cons]
    Z = np.full((len(stas), len(cols)), np.nan)
    for i, s in enumerate(stas):
        for j, (w, c, cn) in enumerate(cols):
            r = T[(T.sta == s) & (T.window == w) & (T.comp == c) & (T.contrast == cn)]
            if len(r):
                Z[i, j] = r.z.iloc[0]
    ax = axs[0]
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(Z[i, j]) and abs(Z[i, j]) >= 2:
                ax.text(j, i, f'{Z[i, j]:.1f}', ha='center', va='center', fontsize=6)
    ax.set_yticks(range(len(stas))); ax.set_yticklabels(stas, fontsize=8)
    lab = {'step_Petrolia': 'Petrolia', 'step_Ferndale': 'Ferndale', 'step_M7': 'M7', 'longterm_post2021': 'after/before 2021-12'}
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([lab[cn] for w, c, cn in cols], rotation=90, fontsize=7)
    for k in range(6):
        ax.axvline(4 * k - 0.5, color='k', lw=1)
        w, c = [(w, c) for w in ['slab', 'crust', 'noise'] for c in COMPS][k]
        ax.text(4 * k + 1.5, -0.9, f'{WLAB[w]}\n{"radial" if c == "R" else "transverse"}', ha='center', fontsize=7)
    plt.colorbar(im, ax=ax, pad=0.01, fraction=0.03).set_label('z of 1.25-3 Hz power change')
    ax = axs[1]
    exts = ['LFE rate', 'LFE coda dv/v', 'GNSS']
    M = np.full((len(stas), 6), np.nan); txt = {}
    for i, s in enumerate(stas):
        for j, (w, c) in enumerate([('slab', 'R'), ('slab', 'T')]):
            for k, e in enumerate(exts):
                r = CR[(CR.sta == s) & (CR.window == w) & (CR.comp == c) & CR.external.str.startswith(e)]
                if len(r):
                    M[i, 3 * j + k] = r.r.iloc[0]
                    txt[(i, 3 * j + k)] = f'{r.r.iloc[0]:+.2f}\np={r.p_perm.iloc[0]:.2f}'
    im = ax.imshow(M, cmap='PuOr', vmin=-1, vmax=1, aspect='auto')
    for (i, j), tx in txt.items():
        ax.text(j, i, tx, ha='center', va='center', fontsize=6)
    ax.set_yticks(range(len(stas))); ax.set_yticklabels(stas, fontsize=8)
    ax.set_xticks(range(6)); ax.set_xticklabels([f'{e}\n({"radial" if c == "R" else "transverse"})' for c in COMPS for e in exts], fontsize=7, rotation=30)
    plt.colorbar(im, ax=ax, pad=0.01, fraction=0.05).set_label('Pearson r (overlapping 1.5-yr windows)')
    ax.set_title('Slab-window power vs external series; p from event-time permutations', fontsize=9)
    fig.suptitle('Receiver-function spectral power (Herath & Audet 2024 method): null-calibrated changes and correlations', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, 'sci_rf_power_overview.png'), dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations', default='pilot')
    ap.add_argument('--nperm', type=int, default=300)
    ap.add_argument('--nperm_series', type=int, default=200)
    ap.add_argument('--nplacebo', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--overview', action='store_true')
    a = ap.parse_args()
    if not a.overview:
        sts = PILOT if a.stations == 'pilot' else (list(STATIONS) if a.stations == 'all' else a.stations.split(','))
        sts = [s for s in sts if os.path.exists(os.path.join(OUT, f'rf_{s}.npz'))]
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(a.workers, len(sts))) as ex:
            for T, CR in ex.map(run_station, sts, [a] * len(sts)):
                print(T[T.window == 'slab'].round(3).to_string())
                print(CR[CR.window == 'slab'].round(3).to_string())
    overview()


if __name__ == '__main__':
    main()
