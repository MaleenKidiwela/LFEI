#!/usr/bin/env python3
"""Receiver-function anisotropy / dipping-structure harmonics through time (test 3 of the 'extras' round).

Back-azimuth harmonic decomposition (Bianchi et al. 2010; joint radial + transverse, k = 0, 1, 2) of the
QC-selected, Ps-moveout-corrected event RFs of data/sci/rf_timelapse (sci_rf_common.load_rf, Gaussian a=2.5).
Model per lag t:  R = c0 + c1 cos(b) + c2 sin(b) + c3 cos(2b) + c4 sin(2b)
                  T = s*( -c1 sin(b) + c2 cos(b) - c3 sin(2b) + c4 cos(2b) )      (b = back-azimuth)
(the transverse rows are the 90/45-deg phase-shifted versions of the radial ones; the sign s is the one chosen
by sci_rf_static.py for each station: static_picks.csv 'tsign').
k-th harmonic as a complex trace z_k(t) = c_cos + i c_sin; a rotation of the symmetry axis by dtheta multiplies
z_k by exp(i k dtheta).

Windows: noise [-4.5, -1.5] s (pre-P), P0 [-0.3, 0.3] s (direct P: tilt / sensor orientation),
shallow [0.15, 1.0] s, slab/LVZ [t_syn - 0.6, m_syn + 0.6] s per station (Bloch 2023 synthetic times from
data/sci/rf_lvz/stations_lvz.csv; clipped to [1.0, 5.5] s; default [1.5, 5.0]).

Quantities per contrast (A = later group, B = earlier group), per window, k = 0, 1, 2:
  dlnA_k   = ln(RMS|z_k(A)| / RMS|z_k(B)|) over the window
  dth_k    = arg(sum z_k(A) conj z_k(B)) / k  (deg; axis rotation; k >= 1)
  dnorm_k  = ||z_k(A) - z_k(B)|| / ||z_k(B)||  (waveform change of the harmonic)
Geometry matching: 30-deg back-azimuth sectors; only sectors sampled by both groups; each group's events are
weighted so that both have the same sector weights h_s = nA_s nB_s / (nA_s + nB_s) (weighted least squares).
Coverage rule: shared sectors must span >= 3 of the 4 back-azimuth quadrants, and the weighted design
(R+T, 5 parameters) condition number < 30.

Contrasts (as sci_rf_timelapse.py): steps at Petrolia / Ferndale / M7 (B = 3 yr before, A = after until the next
quake or 3 yr), post-2021 vs before, 2-yr bins vs rest + linear trend.
Nulls: bootstrap (events, within groups) -> SE; permutation (labels shuffled within sectors, sizes preserved)
-> null mean (noise bias of amplitudes) and SD; placebo dates (fake quakes / fake split dates, same window
lengths, >= 1 yr from the real quakes). z = (est - perm_mean) / max(perm SD, placebo SD); placebo percentile of
the debiased estimate (each placebo value debiased by its own 20-permutation mean). Network: mean of station
debiased estimates over stations with >=10 placebo dates; null from the SAME placebo dates applied to all
stations at once (keeps the shared-event correlation between stations).
Injection: (a) +20% of the station's own all-event k=1 and k=2 harmonics (slab window) added to group A
(expected dlnA = ln 1.2 = 0.182); (b) a 10-deg rotation of the k=1/k=2 harmonics of group A in the slab window.
Recovery -> detection limits (2 sigma) in % harmonic amplitude change and in degrees.
Excluded epochs: sci_rf_common.SWS_EXCLUDE (B933 2017-01..2021-11 and from 2024-08; B045 EH1 to 2011; ...)
plus NC.KMR from 2024-06-01 (orientation/sensor problem near the M7).
-> data/sci/extras/rf_aniso/{static_harmonics,contrasts,network,trend_bins,injection,coverage}.csv, summary.json
   figures/sci_extras_rf_aniso_{static,contrasts,timeseries}.png
Usage: sci_extras_rf_aniso.py [--stations all|A,B] [--nperm 200] [--nboot 150] [--nplacebo 100] [--nperm_placebo 20]
       sci_extras_rf_aniso.py --figs   (figures + summary from the CSVs)
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
from sci_rf_common import QUAKES, load_rf, OUT as RFOUT  # noqa: E402

warnings.filterwarnings('ignore')
ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/extras/rf_aniso'
FIG = f'{ROOT}/figures'
YR = 365.25
EXTRA_EXCL = {'NC.KMR': [('2024-06-01', '2030-01-01')]}
MIN_EVENTS = 40
TRANGE = (-5.0, 7.0)


def day(s):
    return pd.Timestamp(s).value / 86400e9


QDAY = {k: day(v) for k, v in QUAKES.items()}


def design(baz, tsign):
    ph = np.radians(baz)
    Gr = np.c_[np.ones_like(ph), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)]
    Gt = tsign * np.c_[np.zeros_like(ph), -np.sin(ph), np.cos(ph), -np.sin(2 * ph), np.cos(2 * ph)]
    return Gr, Gt


class Sta:
    def __init__(self, code, tsign):
        self.code = code
        m, rf, _, t, g, _ = load_rf(code)
        tt = m.time.values.astype('datetime64[ns]').astype(np.int64) / 86400e9
        keep = np.ones(len(m), bool)
        for a, b in EXTRA_EXCL.get(code, []):
            keep &= ~((tt >= day(a)) & (tt < day(b)))
        wt = (t >= TRANGE[0]) & (t <= TRANGE[1])
        ig = g.index(2.5)
        self.m = m[keep].reset_index(drop=True)
        self.time = tt[keep]
        self.t = t[wt]
        self.R = rf[keep][:, ig, 0][:, wt].astype(np.float64)
        self.T = rf[keep][:, ig, 1][:, wt].astype(np.float64)
        self.baz = self.m.baz.values
        self.sect = (self.baz // 30).astype(int) % 12
        self.tsign = tsign
        self.Gr, self.Gt = design(self.baz, tsign)
        self.win = {}

    def fit(self, idx, w, R=None, T=None):
        """weighted LS harmonic fit for events idx with weights w -> coeffs (5, nt), cond"""
        R = self.R if R is None else R
        T = self.T if T is None else T
        sw = np.sqrt(w)[:, None]
        G = np.r_[self.Gr[idx] * sw, self.Gt[idx] * sw]
        D = np.r_[R[idx] * sw, T[idx] * sw]
        c, *_ = np.linalg.lstsq(G, D, rcond=None)
        s = np.linalg.svd(G, compute_uv=False)
        return c, s[0] / max(s[-1], 1e-12)

    def weights(self, ia, ib, sect=None):
        sect = self.sect if sect is None else sect
        sa, sb = sect[ia], sect[ib]
        na = np.bincount(sa, minlength=12); nb = np.bincount(sb, minlength=12)
        shared = (na > 0) & (nb > 0)
        if shared.sum() == 0:
            return None
        quads = np.unique(np.where(shared)[0] // 3)
        if len(quads) < 3:
            return None
        h = np.where(shared, na * nb / np.maximum(na + nb, 1), 0.0)
        ka = shared[sa]; kb = shared[sb]
        wa = h[sa[ka]] / na[sa[ka]]; wb = h[sb[kb]] / nb[sb[kb]]
        return ia[ka], wa, ib[kb], wb, len(quads), int(shared.sum())

    def compare(self, ia, ib, R=None, T=None, sect=None, retc=False):
        ww = self.weights(np.asarray(ia), np.asarray(ib), sect)
        if ww is None:
            return None
        ja, wa, jb, wb, nq, ns = ww
        if len(ja) < 5 or len(jb) < 5:
            return None
        ca, conda = self.fit(ja, wa, R, T)
        cb, condb = self.fit(jb, wb, R, T)
        if max(conda, condb) > 30:
            return None
        out = measure(ca, cb, self.t, self.win)
        out['_n'] = (len(ja), len(jb), nq, ns, max(conda, condb))
        if retc:
            out['_c'] = (ca, cb)
        return out


def zk(c, k):
    if k == 0:
        return c[0] + 0j
    return c[2 * k - 1] + 1j * c[2 * k]


def measure(ca, cb, t, wins):
    out = {}
    for wn, (t0, t1) in wins.items():
        w = (t >= t0) & (t <= t1)
        for k in (0, 1, 2):
            za, zb = zk(ca, k)[w], zk(cb, k)[w]
            ra, rb = np.sqrt(np.mean(np.abs(za) ** 2)), np.sqrt(np.mean(np.abs(zb) ** 2))
            out[f'{wn}_k{k}_dlnA'] = np.log(max(ra, 1e-9) / max(rb, 1e-9))
            out[f'{wn}_k{k}_dnorm'] = np.sqrt(np.sum(np.abs(za - zb) ** 2) / max(np.sum(np.abs(zb) ** 2), 1e-18))
            if k >= 1:
                out[f'{wn}_k{k}_dth'] = np.degrees(np.angle(np.sum(za * np.conj(zb)))) / k
    return out


def wrap(x, k):
    p = 360.0 / k
    return (x + p / 2) % p - p / 2


def is_angle(q):
    return q.endswith('_dth')


def kof(q):
    return int(q.split('_k')[1][0])


def contrasts(S):
    tt = S.time
    qs = sorted(QDAY.items(), key=lambda x: x[1])
    out = []
    for i, (k, tq) in enumerate(qs):
        nxt = qs[i + 1][1] if i + 1 < len(qs) else np.inf
        a_end = min(tq + 3 * YR, nxt)
        out.append((f'step_{k}', np.where((tt >= tq) & (tt < a_end))[0], np.where((tt >= tq - 3 * YR) & (tt < tq))[0],
                    (tq, a_end)))
    tq = QDAY['Petrolia']
    out.append(('longterm_post2021', np.where(tt >= tq)[0], np.where(tt < tq)[0], (tq, tt.max())))
    return out


def placebo_global(rng, n=100):
    """common placebo dates for all stations (2009.5-2026), >=1 yr from each real quake."""
    q = np.array(list(QDAY.values()))
    cand = np.arange(day('2010-01-01'), day('2026-06-01'), 7.0)
    ok = cand[np.all(np.abs(cand[:, None] - q[None]) > YR, axis=1)]
    return np.sort(rng.choice(ok, min(n, len(ok)), replace=False))


def groups_for(S, name, d, la):
    tt = S.time
    if name.startswith('step'):
        ia = np.where((tt >= d) & (tt < d + la))[0]
        ib = np.where((tt >= d - 3 * YR) & (tt < d))[0]
    else:
        ia = np.where(tt >= d)[0]; ib = np.where(tt < d)[0]
    return ia, ib


def perm_null(S, ia, ib, rng, nperm):
    """labels shuffled within back-azimuth sectors (group sizes and geometry preserved)"""
    pm = []
    allx = np.r_[ia, ib]
    sec = S.sect[allx]
    for _ in range(nperm):
        lab = np.r_[np.ones(len(ia), bool), np.zeros(len(ib), bool)]
        for s in np.unique(sec):
            j = np.where(sec == s)[0]
            lab[j] = rng.permutation(lab[j])
        x = S.compare(allx[lab], allx[~lab])
        if x:
            x.pop('_n'); pm.append(x)
    return pd.DataFrame(pm)


def run_station(code, tsign, wins, a, pdates):
    rng = np.random.default_rng(zlib.crc32(code.encode()))
    S = Sta(code, tsign)
    S.win = wins
    n = len(S.time)
    print(code, 'n', n, 'win', {k: (round(v[0], 2), round(v[1], 2)) for k, v in wins.items()}, flush=True)
    rows, stat_rows, inj_rows, cov_rows, pl_rows, bin_rows = [], [], [], [], [], []
    if n < MIN_EVENTS:
        return None
    # ---------------- static harmonics (all events, uniform weights) + back-azimuth scramble null
    idx = np.arange(n)
    c_all, cond = S.fit(idx, np.ones(n))
    nq = len(np.unique(S.sect // 3))
    scr = []
    for _ in range(100):
        p = rng.permutation(n)
        G = np.r_[S.Gr[p], S.Gt[p]]
        c, *_ = np.linalg.lstsq(G, np.r_[S.R, S.T], rcond=None)
        scr.append(c)
    for wn, (t0, t1) in wins.items():
        w = (S.t >= t0) & (S.t <= t1)
        for k in (0, 1, 2):
            z = zk(c_all, k)[w]
            amp = np.sqrt(np.mean(np.abs(z) ** 2))
            ns = np.array([np.sqrt(np.mean(np.abs(zk(c, k)[w]) ** 2)) for c in scr])
            j = np.argmax(np.abs(z))
            ph = np.degrees(np.angle(z[j])) / k if k else np.nan
            # bootstrap SE of amplitude
            bs = []
            for _ in range(60):
                b = rng.integers(0, n, n)
                cb, _ = S.fit(b, np.ones(n))
                bs.append(np.sqrt(np.mean(np.abs(zk(cb, k)[w]) ** 2)))
            stat_rows.append(dict(sta=code, n=n, quadrants=nq, cond=cond, window=wn, t0=t0, t1=t1, k=k, rms_amp=amp,
                                  boot_se=np.std(bs), scramble_mean=ns.mean(), scramble_sd=ns.std(),
                                  z_vs_scramble=(amp - ns.mean()) / max(ns.std(), 1e-12),
                                  t_peak=S.t[w][j], phase_peak_deg=ph % (360.0 / k) if k else np.nan,
                                  peak_amp=np.abs(z[j])))
    # coverage table
    for y in range(2008, 2027):
        k = (S.time >= day(f'{y}-01-01')) & (S.time < day(f'{y + 1}-01-01'))
        cov_rows.append(dict(sta=code, year=y, n=int(k.sum()), n_quadrants=len(np.unique(S.sect[k] // 3)),
                             n_sectors=len(np.unique(S.sect[k]))))
    # ---------------- contrasts
    for name, ia, ib, (ta0, ta1) in contrasts(S):
        r = S.compare(ia, ib, retc=True)
        base = dict(sta=code, contrast=name, nA=len(ia), nB=len(ib))
        if r is None:
            rows.append(dict(base, quantity='NA', note='insufficient coverage/events')); continue
        nA_u, nB_u, nqd, nsec, cnd = r.pop('_n')
        ca, cb = r.pop('_c')
        base.update(nA_used=nA_u, nB_used=nB_u, quadrants=nqd, sectors=nsec, cond=cnd)
        # bootstrap
        bs = []
        for _ in range(a.nboot):
            ba = rng.choice(ia, len(ia)); bb = rng.choice(ib, len(ib))
            x = S.compare(ba, bb)
            if x:
                x.pop('_n'); bs.append(x)
        bs = pd.DataFrame(bs)
        # permutation within sectors
        pm = perm_null(S, ia, ib, rng, a.nperm)
        # placebo dates (common across stations); amplitude-type values debiased by their own permutation mean
        pl = []
        la = ta1 - ta0
        for d in pdates:
            ja, jb = groups_for(S, name, d, la)
            if name.startswith('longterm') and (min(len(ja), len(jb)) < 0.2 * n):
                continue
            if name.startswith('step') and (d + la > S.time.max() + 30 or d - 3 * YR < S.time.min() - 365):
                continue
            x = S.compare(ja, jb)
            if x:
                x.pop('_n')
                pmx = perm_null(S, ja, jb, rng, a.nperm_placebo)
                if len(pmx) < 5:
                    continue
                for q in list(x):
                    if not is_angle(q):
                        x[q] = x[q] - pmx[q].mean()
                    else:
                        x[q] = wrap(x[q], kof(q))
                x['pdate'] = d; pl.append(x)
        pl = pd.DataFrame(pl)
        for q, v in r.items():
            k = kof(q)
            if is_angle(q):
                v = wrap(v, k)
                pmv = wrap(pm[q].values, k) if len(pm) else np.array([])
                bsv = wrap(bs[q].values - v, k) if len(bs) else np.array([])
                pmean = 0.0
            else:
                pmv = pm[q].values if len(pm) else np.array([])
                bsv = bs[q].values if len(bs) else np.array([])
                pmean = np.mean(pmv) if len(pmv) else np.nan
            plv = pl[q].values if len(pl) else np.array([])
            psd = np.std(pmv) if len(pmv) > 10 else np.nan
            plsd = np.std(plv) if len(plv) > 10 else np.nan
            sd = np.nanmax([psd, plsd])
            est_db = v - pmean
            z = est_db / sd if sd > 0 else np.nan
            pct_pl = 100 * np.mean(plv < est_db) if len(plv) > 10 else np.nan
            pct_pm = 100 * np.mean(pmv - pmean < est_db) if len(pmv) > 10 else np.nan
            rows.append(dict(base, quantity=q, est=v, est_debiased=est_db, boot_se=np.std(bsv) if len(bsv) else np.nan,
                             perm_mean=pmean, perm_sd=psd, placebo_sd=plsd, n_placebo=len(plv), null_sd=sd, z=z,
                             pct_perm=pct_pm, pct_placebo=pct_pl, det2sigma=2 * sd))
        for _, prow in pl.iterrows():
            pl_rows.append(dict(sta=code, contrast=name, **prow.to_dict()))
        pl_rows.append(dict(sta=code, contrast=name, pdate=np.nan, real=True,
                            **{q: (wrap(v, kof(q)) if is_angle(q) else v - np.mean(pm[q]) if len(pm) else np.nan)
                               for q, v in r.items()}))
        # ---------------- injections (slab window)
        ws = wins['slab']
        wmask = ((S.t >= ws[0] - 0.3) & (S.t <= ws[1] + 0.3)).astype(float)
        taper = np.convolve(wmask, np.hanning(9) / np.hanning(9).sum(), 'same')
        for kind in ('amp20', 'rot10'):
            R2, T2 = S.R.copy(), S.T.copy()
            for k in (1, 2):
                zz = zk(c_all, k) * taper
                if kind == 'amp20':
                    dz = 0.2 * zz
                else:
                    dz = zz * (np.exp(1j * k * np.radians(10.0)) - 1)
                cc = np.zeros((5, len(S.t)))
                cc[2 * k - 1] = dz.real; cc[2 * k] = dz.imag
                R2[ia] += S.Gr[ia] @ cc
                T2[ia] += S.Gt[ia] @ cc
            x = S.compare(ia, ib, R2, T2)
            if x is None:
                continue
            for k in (1, 2):
                if kind == 'amp20':
                    q = f'slab_k{k}_dlnA'; expct = np.log(1.2)
                    rec = (x[q] - r[q]) / expct
                else:
                    q = f'slab_k{k}_dth'; expct = 10.0
                    rec = wrap(x[q] - r[q], k) / expct
                inj_rows.append(dict(sta=code, contrast=name, kind=kind, k=k, quantity=q, expected=expct,
                                     recovered=rec * expct, recovery=rec))
    # ---------------- 2-yr bins vs rest + trend (slab and shallow windows)
    edges = [day(f'{y}-01-01') for y in range(2008, 2028, 2)]

    def bins(time):
        out = []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            ia = np.where((time >= e0) & (time < e1))[0]
            ib = np.where((time < e0) | (time >= e1))[0]
            if len(ia) < 8:
                continue
            x = S.compare(ia, ib)
            if x:
                nn = x.pop('_n')
                x['tmid'] = (e0 + e1) / 2; x['n'] = nn[0]
                out.append(x)
        return pd.DataFrame(out)
    B = bins(S.time)
    if len(B) >= 3:
        qs = [c for c in B.columns if c not in ('tmid', 'n') and not c.endswith('dnorm')]

        def slope(D):
            x = (D.tmid.values - D.tmid.mean()) / YR
            w = D.n.values.astype(float)
            o = {}
            for q in qs:
                y = wrap(D[q].values, kof(q)) if is_angle(q) else D[q].values
                o[q] = np.sum(w * x * (y - np.average(y, weights=w))) / np.sum(w * x * x)
            return o
        tr = slope(B)
        perm = []
        for _ in range(max(a.nperm // 3, 60)):
            tp = S.time.copy()
            for s in np.unique(S.sect):
                j = np.where(S.sect == s)[0]
                tp[j] = rng.permutation(tp[j])
            D = bins(tp)
            if len(D) >= 3:
                perm.append(slope(D))
        perm = pd.DataFrame(perm)
        for q in qs:
            sd = perm[q].std() if len(perm) > 10 else np.nan
            rows.append(dict(sta=code, contrast='trend_per_yr', quantity=q, est=tr[q], est_debiased=tr[q], perm_sd=sd,
                             null_sd=sd, z=tr[q] / sd if sd > 0 else np.nan, det2sigma=2 * sd, nA=n,
                             pct_perm=100 * np.mean(perm[q].values < tr[q]) if len(perm) > 10 else np.nan))
        B.insert(0, 'sta', code)
        bin_rows = B.to_dict('records')
    return rows, stat_rows, inj_rows, cov_rows, pl_rows, bin_rows


def station_windows(code):
    L = pd.read_csv(f'{ROOT}/data/sci/rf_lvz/stations_lvz.csv').set_index('sta')
    t0, t1 = 1.5, 5.0
    if code in L.index:
        st, sm = L.loc[code, 'syn_t'], L.loc[code, 'syn_m']
        if np.isfinite(st):
            t0 = st - 0.6
            t1 = (sm + 0.6) if np.isfinite(sm) else st + 2.0
    t0, t1 = max(t0, 1.0), min(t1, 5.5)
    return {'noise': (-4.5, -1.5), 'P0': (-0.3, 0.3), 'shallow': (0.15, 1.0), 'slab': (t0, t1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations', default='all')
    ap.add_argument('--nperm', type=int, default=200)
    ap.add_argument('--nboot', type=int, default=150)
    ap.add_argument('--nplacebo', type=int, default=100)
    ap.add_argument('--nperm_placebo', type=int, default=20)
    ap.add_argument('--figs', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.figs:
        summarize(); figures(); return
    P = pd.read_csv(f'{RFOUT}/static_picks.csv')
    tsign = P.groupby('sta').tsign.first().to_dict()
    if a.stations == 'all':
        import glob
        codes = sorted(os.path.basename(f)[3:-4] for f in glob.glob(f'{RFOUT}/rf_*.npz'))
    else:
        codes = a.stations.split(',')
    pdates = placebo_global(np.random.default_rng(12345), a.nplacebo)
    res = {k: [] for k in ('contrasts', 'static_harmonics', 'injection', 'coverage', 'placebo_long', 'trend_bins')}
    for code in codes:
        try:
            out = run_station(code, tsign.get(code, 1.0), station_windows(code), a, pdates)
        except Exception as e:
            print(code, 'FAILED', repr(e), flush=True); continue
        if out is None:
            print(code, 'skipped (<%d events)' % MIN_EVENTS, flush=True); continue
        for k, v in zip(res, out):
            res[k] += v
        for k, v in res.items():
            pd.DataFrame(v).to_csv(f'{OUT}/{k}.csv', index=False)
    summarize(); figures()


# ----------------------------------------------------------------------------------------------- summary
MAINQ = [f'{w}_k{k}_{m}' for w in ('noise', 'P0', 'shallow', 'slab') for k in (1, 2) for m in ('dlnA', 'dth')] + \
        [f'{w}_k0_dlnA' for w in ('noise', 'shallow', 'slab')]


def summarize():
    C = pd.read_csv(f'{OUT}/contrasts.csv')
    C = C[C.quantity.isin(MAINQ + [q.replace('dlnA', 'dnorm') for q in MAINQ if 'dlnA' in q])]
    PL = pd.read_csv(f'{OUT}/placebo_long.csv')
    I = pd.read_csv(f'{OUT}/injection.csv') if os.path.exists(f'{OUT}/injection.csv') else pd.DataFrame()
    S = pd.read_csv(f'{OUT}/static_harmonics.csv')
    # network combination: mean of station debiased estimates; null = same placebo dates for all stations
    net = []
    for con, g in PL.groupby('contrast'):
        real = g[g.real == True]  # noqa: E712
        plc = g[g.real != True]  # noqa: E712
        npl = plc.groupby('sta').size()
        stas = sorted(s_ for s_ in real.sta.unique() if npl.get(s_, 0) >= 10)
        for q in MAINQ:
            if q not in g.columns:
                continue
            k = kof(q)
            vals = real.set_index('sta')[q].reindex(stas)
            est = np.nanmean(vals.values)
            pv = []
            for d, h in plc.groupby('pdate'):
                h = h[h.sta.isin(stas)].set_index('sta')
                if len(h) < max(2, 0.5 * len(stas)):
                    continue
                pv.append(np.nanmean(h[q].values.astype(float)))
            pv = np.array(pv)
            sd = np.std(pv) if len(pv) > 10 else np.nan
            net.append(dict(contrast=con, quantity=q, n_sta=len(stas), est=est, placebo_mean=np.mean(pv) if len(pv) else np.nan,
                            placebo_sd=sd, n_placebo=len(pv), z=(est - np.mean(pv)) / sd if sd > 0 else np.nan,
                            pct_placebo=100 * np.mean(pv < est) if len(pv) > 10 else np.nan, det2sigma=2 * sd))
    N = pd.DataFrame(net)
    N.to_csv(f'{OUT}/network.csv', index=False)
    # tests vs chance (station x contrast x quantity, dlnA/dth only, excluding noise window as a control)
    T = C[C.quantity.str.contains('dlnA|dth') & C.z.notna()]
    sig = T[~T.quantity.str.startswith('noise')]
    noi = T[T.quantity.str.startswith('noise')]
    summ = dict(
        n_station_tests=int(len(sig)), frac_absz_gt2=float((sig.z.abs() > 2).mean()), n_absz_gt3=int((sig.z.abs() > 3).sum()),
        expected_absz_gt3=float(len(sig) * 0.0027), noise_window_frac_absz_gt2=float((noi.z.abs() > 2).mean()),
        slab_only=dict(n=int(sig.quantity.str.startswith('slab').sum()),
                       frac_gt2=float((sig[sig.quantity.str.startswith('slab')].z.abs() > 2).mean())),
        top_station_tests=sig.reindex(sig.z.abs().sort_values(ascending=False).index).head(15)[
            ['sta', 'contrast', 'quantity', 'est_debiased', 'null_sd', 'z', 'pct_placebo', 'nA_used', 'nB_used']].to_dict('records'),
        network=N[N.quantity.str.startswith(('slab', 'shallow', 'P0'))].to_dict('records'),
    )
    # detection limits per station (slab window): 2 sigma, converted with injection recovery
    dl = []
    for (sta, con), g in C[C.quantity.str.startswith('slab') & C.contrast.ne('trend_per_yr')].groupby(['sta', 'contrast']):
        d = dict(sta=sta, contrast=con)
        for k in (1, 2):
            ga = g[g.quantity == f'slab_k{k}_dlnA']; gt = g[g.quantity == f'slab_k{k}_dth']
            if len(I):
                ra = I[(I.sta == sta) & (I.contrast == con) & (I.kind == 'amp20') & (I.k == k)].recovery
                rt = I[(I.sta == sta) & (I.contrast == con) & (I.kind == 'rot10') & (I.k == k)].recovery
            else:
                ra = rt = pd.Series([], dtype=float)
            reca = float(ra.iloc[0]) if len(ra) else np.nan
            rect = float(rt.iloc[0]) if len(rt) else np.nan
            if len(ga):
                d[f'k{k}_dlnA_det2s'] = float(ga.det2sigma.iloc[0])
                d[f'k{k}_amp_det2s_pct'] = 100 * (np.exp(float(ga.det2sigma.iloc[0]) / max(reca, 0.1)) - 1) if np.isfinite(reca) else np.nan
                d[f'k{k}_amp_recovery'] = reca
            if len(gt):
                d[f'k{k}_rot_det2s_deg'] = float(gt.det2sigma.iloc[0]) / max(rect, 0.1) if np.isfinite(rect) else np.nan
                d[f'k{k}_rot_recovery'] = rect
        dl.append(d)
    DL = pd.DataFrame(dl)
    DL.to_csv(f'{OUT}/detection_limits.csv', index=False)
    for con, g in DL.groupby('contrast'):
        summ[f'detlim_{con}'] = {c: dict(best=float(g[c].min()), median=float(g[c].median()),
                                         best_sta=str(g.loc[g[c].idxmin(), 'sta']) if g[c].notna().any() else '')
                                 for c in g.columns if c.endswith(('pct', 'deg')) and g[c].notna().any()}
    st = S[(S.window == 'slab')].pivot_table(index='sta', columns='k', values=['rms_amp', 'z_vs_scramble'])
    summ['static_slab'] = {s: {f'k{k}_amp': float(st.loc[s, ('rms_amp', k)]) for k in (0, 1, 2)} |
                           {f'k{k}_z': float(st.loc[s, ('z_vs_scramble', k)]) for k in (1, 2)} for s in st.index}
    # conversion of detection limits to LVL anisotropy with PyRaysum sensitivities (sci_extras_rf_aniso_raysum.py)
    rfn = f'{OUT}/raysum_sensitivity.csv'
    if os.path.exists(rfn):
        RS = pd.read_csv(rfn)
        conv = []
        for sta, g in RS.groupby('sta'):
            g = g.set_index('case')
            p0 = g.loc['iso', 'P0']
            sens2 = g.loc['lvl_h293_10', 'k2_diff_rms'] / 10 / p0      # k2 RMS change per % (fraction of direct P)
            sens1 = g.loc['lvl_p40_5', 'k1_diff_rms'] / 5 / p0         # k1 per % (axis plunging 40 deg)
            sp0 = S[(S.sta == sta) & (S.window == 'P0') & (S.k == 0)].peak_amp
            if not len(sp0):
                continue
            d0 = float(sp0.iloc[0])
            obs = {k: float(S[(S.sta == sta) & (S.window == 'slab') & (S.k == k)].rms_amp.iloc[0]) / d0 for k in (1, 2)}
            row = dict(sta=sta, obs_k1_over_P=obs[1], obs_k2_over_P=obs[2], syn_k2_per_pct=sens2, syn_k1_per_pct=sens1,
                       syn_iso_k2_over_P=g.loc['iso', 'k2_rms'] / p0, syn_iso_k1_over_P=g.loc['iso', 'k1_rms'] / p0,
                       obs_k2_equiv_ani_pct=obs[2] / sens2)
            for con in ('longterm_post2021', 'step_M7', 'step_Ferndale', 'step_Petrolia'):
                h = DL[(DL.sta == sta) & (DL.contrast == con)]
                if len(h):
                    for k, sens in ((1, sens1), (2, sens2)):
                        c = f'k{k}_amp_det2s_pct'
                        if c in h and np.isfinite(h[c].iloc[0]):
                            row[f'{con}_k{k}_det2s_ani_pct'] = h[c].iloc[0] / 100 * obs[k] / sens
            conv.append(row)
        CV = pd.DataFrame(conv)
        CV.to_csv(f'{OUT}/aniso_conversion.csv', index=False)
        summ['aniso_conversion'] = CV.to_dict('records')
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ('static_slab', 'network', 'top_station_tests')}, indent=1, default=float))


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C = pd.read_csv(f'{OUT}/contrasts.csv')
    S = pd.read_csv(f'{OUT}/static_harmonics.csv')
    B = pd.read_csv(f'{OUT}/trend_bins.csv')
    N = pd.read_csv(f'{OUT}/network.csv')
    # --- static: amplitudes per window and z vs scramble
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    stas = sorted(S.sta.unique())
    x = np.arange(len(stas))
    for j, (wn, mk) in enumerate([('slab', 'o'), ('shallow', 's'), ('noise', 'x')]):
        for k, col in ((1, 'C0'), (2, 'C3')):
            g = S[(S.window == wn) & (S.k == k)].set_index('sta').reindex(stas)
            ax[0].plot(x + 0.1 * (k - 1.5), g.rms_amp / S[(S.window == 'P0') & (S.k == 0)].set_index('sta').reindex(stas).rms_amp,
                       mk, color=col, alpha=0.8, label=f'{wn} k={k}')
            ax[1].plot(x + 0.1 * (k - 1.5), g.z_vs_scramble, mk, color=col, alpha=0.8, label=f'{wn} k={k}')
    ax[0].set_ylabel('harmonic RMS / direct-P k=0 RMS'); ax[0].set_yscale('log')
    ax[1].set_ylabel('z vs back-azimuth scramble'); ax[1].axhline(3, color='k', ls=':')
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels(stas, rotation=90, fontsize=8)
    ax[0].legend(fontsize=7, ncol=3)
    fig.suptitle('Static back-azimuth harmonics (all events, a=2.5): k=1 dipping/tilted-axis, k=2 horizontal-axis anisotropy')
    fig.tight_layout(); fig.savefig(f'{FIG}/sci_extras_rf_aniso_static.png', dpi=130); plt.close(fig)
    # --- contrasts: z per station
    cons = ['step_Petrolia', 'step_Ferndale', 'step_M7', 'longterm_post2021', 'trend_per_yr']
    qs = ['slab_k1_dlnA', 'slab_k2_dlnA', 'slab_k1_dth', 'slab_k2_dth', 'shallow_k1_dlnA', 'shallow_k2_dlnA',
          'noise_k1_dlnA', 'noise_k2_dlnA']
    fig, ax = plt.subplots(1, len(cons), figsize=(17, 5.5), sharey=True)
    for i, con in enumerate(cons):
        for j, q in enumerate(qs):
            g = C[(C.contrast == con) & (C.quantity == q)]
            ax[i].plot(g.z, np.full(len(g), j) + np.random.default_rng(j).uniform(-0.2, 0.2, len(g)), 'o', ms=4,
                       color='C3' if 'noise' in q else 'C0', alpha=0.7)
            n = N[(N.contrast == con) & (N.quantity == q)]
            if len(n):
                ax[i].plot(n.z, [j], 'k*', ms=12)
        ax[i].axvspan(-2, 2, color='0.9'); ax[i].set_title(con, fontsize=9); ax[i].set_xlim(-6, 6)
        ax[i].set_xlabel('z (vs max of permutation / placebo SD)')
    ax[0].set_yticks(range(len(qs))); ax[0].set_yticklabels(qs, fontsize=8)
    fig.suptitle('RF harmonic changes per station (dots) and network placebo-calibrated (stars); red = pre-P noise control')
    fig.tight_layout(); fig.savefig(f'{FIG}/sci_extras_rf_aniso_contrasts.png', dpi=130); plt.close(fig)
    # --- time series of slab k1/k2 dlnA for stations with most bins
    top = B.groupby('sta').size().sort_values(ascending=False).head(8).index
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for s in top:
        g = B[B.sta == s]
        tt = pd.to_datetime(g.tmid * 86400e9)
        for k in (1, 2):
            ax[k - 1].plot(tt, 100 * g[f'slab_k{k}_dlnA'], 'o-', label=s, ms=4)
    for k in (1, 2):
        ax[k - 1].set_ylabel(f'slab k={k}: 2-yr bin vs rest\nd ln amplitude (%)')
        for q in QUAKES.values():
            ax[k - 1].axvline(pd.Timestamp(q), color='k', ls=':')
    ax[0].legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig(f'{FIG}/sci_extras_rf_aniso_timeseries.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
