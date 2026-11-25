#!/usr/bin/env python3
"""Time-lapse of the slab low-velocity zone (LVZ) from the TOP (negative Ps, Bloch 2023 interface t) and BOTTOM /
slab-Moho (positive Ps, interface m) conversions in teleseismic P receiver functions.  (lfe env; needs the PyRaysum
synthetics from sci_rf_lvz_synth.py, rf env.)

Observable: the top-minus-bottom differential time dT = tau_m - tau_t (delay of the m conversion minus delay of
the t conversion, later group relative to earlier group) and the conversion amplitudes A_t, A_m and their ratio
A_t/A_m.  The differential cancels the shallow column (common to both conversions), which dominates the earlier
time-lapse measures (sci_rf_timelapse.py).  Free-surface multiples (PpPs, PpSs) are identified on raw
(not moveout-corrected) stacks and used where visible.

1. Identification (static): harmonic constant term A0 (a=2.5, Ps moveout-corrected; data/sci/rf_timelapse/static_*.npz)
   vs PyRaysum synthetics of the dipping Bloch et al. (2023) model at the station's event geometries.  t = negative
   extremum within +-0.6 s of the synthetic t-Ps, m = positive extremum within +-0.7 s of the synthetic m-Ps, each
   with |A0| >= 3 bootstrap SE and an interior extremum; usable pair if both and m - t >= 0.8 s.  Multiples: raw
   all-event mean stack vs synthetic multiples-only trace (base minus no-multiple model).
2. Measurements: matched-geometry stacks (30-deg back-azimuth x 2 ray-parameter bins, as sci_rf_timelapse), joint
   delay + amplitude ratio (sci_rf_timelapse.shift_amp) in +-0.35 s windows around each phase.
   Contrasts: steps at Petrolia / Ferndale / M7 (3 yr before vs after until the next quake or 3 yr), after vs before
   2021-12-20, LFE-activity state (prior-30-day catalog count / centred 2-yr median of that count >= 1 vs < 1;
   Shelly 2026 v2 catalogue, all families), annual phase (half-years centred on the peak of the annual LFE-rate
   cycle vs opposite; quadrature split), and 1.5-yr windows 2008-2026 (each vs all other events) with chi-square
   and linear trend.
3. Nulls (all generated GLOBALLY over the union of events, so every station sees the same resampled / permuted
   events and the network combination inherits the shared-event correlation): Poisson event bootstrap;
   label permutation within global geometry bins (restricted to the two groups); placebo dates (steps); circular
   shift of the LFE-state series (LFE contrast); time permutation for chi-square / trend.  Network estimate =
   inverse-bootstrap-variance weighted mean over stations, nulls built from the same replicates.
   Injections: +50 ms differential delay (warp ramp from t to m), and PyRaysum LVZ models (LVL Vs -0.10, -0.05,
   +0.05, +0.10 km/s; -4, -2, +2, +4 %), synthetic differences (warped onto the observed phase times, scaled to the
   observed t-amplitude) added to the later group; noise-free sensitivities -> detection limits in km/s and %.
-> data/sci/rf_lvz/{stations_lvz,contrasts_lvz,windows_lvz,variability_lvz,sensitivity_lvz,detection_lvz}.csv,
   summary.json; figures/sci_rf_lvz_{static,timeseries,contrasts,sensitivity}.png
Usage: sci_rf_lvz.py [--workers 4] [--nboot 300] [--nperm 500] [--nshift 300]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT as RFOUT, FIG, QUAKES, BREAKS, SWS_EXCLUDE, load_rf, P_REF  # noqa: E402
from sci_rf_timelapse import shift_amp  # noqa: E402
from sci_rf_static import pick  # noqa: E402

warnings.filterwarnings('ignore')
OUT = '/home/seismic/LFE/data/sci/rf_lvz'
LFECAT = '/home/seismic/LFE/data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv'
YR = 365.25
HW = 0.35                     # half-width of the phase windows (s), a = 2.5
IG = 1                        # index of a = 2.5 in the synthetic files (gauss 1, 2.5)
NMIN_TL = 60                  # minimum accepted RFs for time-lapse use
NET_EXCLUDE = {'PB.B933': 'instrument break 2021-11-01; problem windows 2017-01..2021-11 and 2024-08+'}
FLAG = {('NC.KMR', 'step_M7'): 'KMR orientation / polarisation change near the M7 (previous RF study)',
        ('PB.B933', '*'): 'B933 instrument break 2021-11-01',
        ('BK.RVIT', 'step_Ferndale'): 'RVIT problem window after Ferndale (excluded events)'}
MODELS_INJ = ['lvl_dvs-0.10', 'lvl_dvs-0.05', 'lvl_dvs+0.05', 'lvl_dvs+0.10', 'lvl_pct-4', 'lvl_pct-2', 'lvl_pct+2',
              'lvl_pct+4']
MODEL_UNIT = {'lvl_dvs-0.10': ('dVs_kms', -0.10), 'lvl_dvs-0.05': ('dVs_kms', -0.05), 'lvl_dvs+0.05': ('dVs_kms', 0.05),
              'lvl_dvs+0.10': ('dVs_kms', 0.10), 'lvl_pct-4': ('dVs_pct', -4), 'lvl_pct-2': ('dVs_pct', -2),
              'lvl_pct+2': ('dVs_pct', 2), 'lvl_pct+4': ('dVs_pct', 4)}


def day(s):
    return pd.Timestamp(s).value / 86400e9


# ============================================================================ static identification
def analytic_times(layers, p=P_REF):
    """flat-layer Ps, PpPs, PpSs times of the interfaces below layer i (vertical thicknesses)."""
    out = {}
    tps = tpp = tss = 0.0
    for i, l in enumerate(layers[:-1]):
        h = l['thick_km']
        qs = np.sqrt(1 / l['vs'] ** 2 - p * p); qp = np.sqrt(1 / l['vp'] ** 2 - p * p)
        tps += h * (qs - qp); tpp += h * (qs + qp); tss += 2 * h * qs
        out[i + 1] = (tps, tpp, tss)
    # interfaces: t = below layer index 1 (2-km layer above LVL), c = below LVL (2), m = below oceanic crust (3)
    return {'t': out[2], 'c': out[3], 'm': out[4]}


def boot_mean(X, rng, nb=200):
    n = len(X)
    B = np.array([X[rng.integers(0, n, n)].mean(0) for _ in range(nb)])
    return X.mean(0), B.std(0)


def identify(code, rng):
    st = np.load(os.path.join(RFOUT, f'static_{code}.npz'))
    S = np.load(os.path.join(OUT, f'synth_{code}.npz'))
    t = S['t']
    A0, A0se, n = st['A0'], st['A0se'], int(st['n'])
    models = list(S['models'])
    syn = S['R'][models.index('base'), IG].mean(0)
    synraw = S['Rraw'][models.index('base'), IG].mean(0)
    mult = synraw - S['Rraw'][models.index('nomult'), IG].mean(0)
    layers = json.loads(str(S['layers']))
    m, rf, rf_raw, t2, g, _ = load_rf(code)
    praw, praw_se = boot_mean(rf_raw[:, g.index(2.5), 0].astype(float), rng)
    pm = float(m.p.mean())
    T = analytic_times(layers, pm)
    row = dict(sta=code, n=n, z_t_km=sum(l['thick_km'] for l in layers[:2]), lvl_km=layers[2]['thick_km'],
               z_m_km=sum(l['thick_km'] for l in layers[:4]), dip=layers[1]['dip'], strike=layers[1]['strike'],
               first=str(m.time.min().date()), last=str(m.time.max().date()))
    for ph in 'tcm':
        row[f'an_Ps_{ph}'], row[f'an_PpPs_{ph}'], row[f'an_PpSs_{ph}'] = T[ph]
    # synthetic phase times (moveout-corrected all-event mean of the dipping model)
    row['syn_t'], row['syn_t_amp'] = pick(syn, t, T['t'][0] - 0.4, T['c'][0] + 0.2, -1)
    row['syn_m'], row['syn_m_amp'] = pick(syn, t, T['m'][0] - 0.5, T['m'][0] + 0.5, 1)
    row['syn_c'], row['syn_c_amp'] = pick(syn, t, T['c'][0] - 0.2, T['c'][0] + 0.4, 1)
    # observed
    for ph, sgn, half in (('t', -1, 0.6), ('m', 1, 0.7)):
        ts = row[f'syn_{ph}']
        if not np.isfinite(ts):
            ts = T[ph][0]
        lo = max(ts - half, 0.9)
        tt, aa = pick(A0, t, lo, ts + half, sgn)
        row[f'obs_{ph}'], row[f'obs_{ph}_amp'] = tt, aa
        row[f'obs_{ph}_z'] = abs(aa) / A0se[np.argmin(np.abs(t - tt))] if np.isfinite(tt) else np.nan
        row[f'ok_{ph}'] = bool(np.isfinite(tt) and row[f'obs_{ph}_z'] >= 3)
    row['pair'] = bool(row['ok_t'] and row['ok_m'] and row['obs_m'] - row['obs_t'] >= 0.8)
    # waveform similarity data vs synthetic, 1 s .. m + 1 s, best lag within +-0.5 s
    w = (t >= 1.0) & (t <= (row['syn_m'] if np.isfinite(row['syn_m']) else 4.0) + 1.0)
    best = (-2, 0)
    for lag in np.arange(-10, 11):
        a = A0[w]; b = np.roll(syn, lag)[w]
        c = np.corrcoef(a, b)[0, 1]
        if c > best[0]:
            best = (c, lag * (t[1] - t[0]))
    row['cc_syn'], row['cc_lag'] = best
    # multiples on raw stacks: PpPs_t (-), PpSs_t (+), PpPs_m (+), PpSs_m (-)
    mults = {}
    for ph, k, sgn in (('PpPs_t', 1, -1), ('PpSs_t', 2, 1), ('PpPs_m', 1, 1), ('PpSs_m', 2, -1)):
        it = ph[-1]
        ta = T[it][k]
        tsy, asy = pick(mult, t, ta - 0.6, ta + 0.6, sgn)
        row[f'syn_{ph}'], row[f'syn_{ph}_amp'] = tsy, asy
        if not np.isfinite(tsy):
            row[f'ok_{ph}'] = False; continue
        tt, aa = pick(praw, t, tsy - 0.6, tsy + 0.6, sgn)
        z = abs(aa) / praw_se[np.argmin(np.abs(t - tt))] if np.isfinite(tt) else np.nan
        row[f'obs_{ph}'], row[f'obs_{ph}_amp'], row[f'obs_{ph}_z'] = tt, aa, z
        row[f'ok_{ph}'] = bool(np.isfinite(tt) and z >= 3 and abs(asy) > 0.01)
        if row[f'ok_{ph}']:
            mults[ph] = tt
    row['multiples'] = ';'.join(mults)
    row['timelapse'] = bool(row['pair'] and n >= NMIN_TL)
    extra = dict(t=t, A0=A0, A0se=A0se, syn=syn, praw=praw, praw_se=praw_se, synraw=synraw, mult=mult)
    return row, extra


# ============================================================================ global events, labels, replicates
class Glob:
    def __init__(self, codes):
        evs = {}
        for c in codes:
            d = np.load(os.path.join(RFOUT, f'rf_{c}.npz'))
            meta = pd.DataFrame(json.loads(str(d['meta'])))
            for r in meta[['evid', 'time']].itertuples():
                evs[r.evid] = r.time
        E = pd.read_csv(os.path.join(RFOUT, 'events.csv'))
        E = E[E.evid.isin(evs)].reset_index(drop=True)
        self.evid = E.evid.values
        self.idx = {e: i for i, e in enumerate(self.evid)}
        self.time = pd.to_datetime(E.time).values.astype('datetime64[s]').astype(np.int64) / 86400.0
        self.geo = (E.baz.values // 30).astype(int) * 2 + (E.p_skm.values >= P_REF).astype(int)
        self.n = len(E)
        self.lfe_ratio, self.lfe_info = lfe_state(self.time)
        self.season = season_phase()

    # ---------------------------------------------------------------- labels: 1 = A (later / high), 0 = B, -1 none
    def step_labels(self, tt, tq, a_end):
        lab = np.full(len(tt), -1, np.int8)
        lab[(tt >= tq - 3 * YR) & (tt < tq)] = 0
        lab[(tt >= tq) & (tt < a_end)] = 1
        return lab

    def contrasts(self):
        q = sorted(((k, day(v)) for k, v in QUAKES.items()), key=lambda x: x[1])
        C = {}
        for i, (k, tq) in enumerate(q):
            nxt = q[i + 1][1] if i + 1 < len(q) else np.inf
            a_end = min(tq + 3 * YR, nxt)
            C[f'step_{k}'] = dict(lab=self.step_labels(self.time, tq, a_end), kind='step', tq=tq, lenA=a_end - tq)
        tq = day(QUAKES['Petrolia'])
        C['post2021'] = dict(lab=np.where(self.time >= tq, 1, 0).astype(np.int8), kind='long')
        C['lfe_high'] = dict(lab=self.lfe_label(self.lfe_ratio), kind='lfe')
        ph = 2 * np.pi * (self.doy() - self.season['peak_doy']) / YR
        C['season_cos'] = dict(lab=np.where(np.cos(ph) > 0, 1, 0).astype(np.int8), kind='season')
        C['season_sin'] = dict(lab=np.where(np.sin(ph) > 0, 1, 0).astype(np.int8), kind='season')
        return C

    def lfe_label(self, ratio):
        lab = np.full(len(ratio), -1, np.int8)
        ok = np.isfinite(ratio)
        lab[ok & (ratio >= 1)] = 1; lab[ok & (ratio < 1)] = 0
        return lab

    def doy(self, tt=None):
        tt = self.time if tt is None else tt
        return (tt - day('2000-01-01')) % YR

    def windows(self):
        e = np.arange(day('2008-01-01'), day('2026-12-31'), 1.5 * YR)
        return list(zip(e[:-1], e[1:])) + ([(e[-1], day('2026-12-31'))] if day('2026-12-31') - e[-1] > 0.5 * YR else [])

    def window_labels(self, tt):
        return [np.where((tt >= a) & (tt < b), 1, 0).astype(np.int8) for a, b in self.windows()]

    # ---------------------------------------------------------------- replicates
    def replicates(self, a, C):
        rng = np.random.default_rng(20261125)
        R = {}
        R['boot'] = rng.poisson(1.0, (a.nboot, self.n)).astype(np.float32)
        bins = [np.where(self.geo == c)[0] for c in np.unique(self.geo)]
        for name, c in C.items():
            lab = c['lab']
            P = np.empty((a.nperm, self.n), np.int8)
            for k in range(a.nperm):
                L = lab.copy()
                for b in bins:
                    j = b[L[b] >= 0]
                    L[j] = rng.permutation(L[j])
                P[k] = L
            c['perm'] = P
            if c['kind'] == 'step':
                c['placebo'] = np.array([self.step_labels(self.time, d, d + c['lenA']) for d in self.placebo_dates(c['lenA'])])
            if c['kind'] == 'lfe':
                c['shift'] = np.array([self.lfe_label(self.lfe_shift(s)) for s in
                                       rng.uniform(90, self.lfe_info['span_d'] - 90, a.nshift)])
        # unrestricted time permutation within geometry bins (window chi2 / trend null)
        TP = np.empty((a.nperm_win, self.n))
        for k in range(a.nperm_win):
            tt = self.time.copy()
            for b in bins:
                tt[b] = rng.permutation(tt[b])
            TP[k] = tt
        R['tperm'] = TP
        return R

    def placebo_dates(self, lenA, exclude=365):
        q = np.array([day(v) for v in QUAKES.values()])
        cand = np.arange(self.time.min() + 3 * YR, self.time.max() - lenA, 15.0)
        ok = [c for c in cand if np.all(np.abs(c - q) > exclude) and not np.any((q > c - 3 * YR) & (q < c + lenA))]
        return np.array(ok)

    def lfe_shift(self, s):
        I = self.lfe_info
        t0, span = I['t0'], I['span_d']
        tt = (self.time - t0 + s) % span + t0
        return np.interp(tt, I['tgrid'], I['ratio'], left=np.nan, right=np.nan)


def lfe_state(tt):
    """prior-30-day LFE count (all families) / median of that count within +-365 d, at times tt (days)."""
    c = pd.read_csv(LFECAT)
    ts = pd.to_datetime(dict(year=c['%year'], month=c.month, day=c.day, hour=c.hr, minute=c['min']))
    d = ts.values.astype('datetime64[s]').astype(np.int64) / 86400.0
    t0, t1 = np.floor(d.min()), np.ceil(d.max())
    daily = np.bincount((d - t0).astype(int), minlength=int(t1 - t0) + 1).astype(float)
    c30 = np.convolve(daily, np.ones(30), mode='full')[:len(daily)]          # days (i-29 .. i)
    c30[:30] = np.nan
    ref = pd.Series(c30).rolling(731, center=True, min_periods=365).median().values
    ratio = c30 / ref
    tgrid = t0 + np.arange(len(daily)) + 1.0
    out = np.interp(tt, tgrid, ratio, left=np.nan, right=np.nan)
    out[(tt < tgrid[0]) | (tt > tgrid[-1])] = np.nan
    return out, dict(t0=t0, span_d=float(len(daily)), tgrid=tgrid, ratio=ratio, daily=daily)


def season_phase():
    c = pd.read_csv(LFECAT)
    ts = pd.to_datetime(dict(year=c['%year'], month=c.month, day=c.day))
    dd = ts.dt.floor('D').value_counts().sort_index()
    s = dd.reindex(pd.date_range(dd.index.min(), dd.index.max()), fill_value=0).astype(float)
    rel = s / s.rolling(365, center=True, min_periods=180).mean()
    rel = rel.dropna()
    doy = ((rel.index.values.astype('datetime64[s]').astype(np.int64) / 86400.0) - day('2000-01-01')) % YR
    ph = 2 * np.pi * doy / YR
    G = np.c_[np.ones(len(ph)), np.cos(ph), np.sin(ph)]
    sol, *_ = np.linalg.lstsq(G, rel.values, rcond=None)
    amp = float(np.hypot(sol[1], sol[2]))
    peak = float((np.degrees(np.arctan2(sol[2], sol[1])) % 360) / 360 * YR)
    # yearly peaks (stability of the phase)
    yrs = {}
    for y in range(2009, 2026):
        k = (rel.index.year == y)
        if k.sum() > 300:
            s2, *_ = np.linalg.lstsq(G[k], rel.values[k], rcond=None)
            yrs[y] = float((np.degrees(np.arctan2(s2[2], s2[1])) % 360) / 360 * YR)
    return dict(peak_doy=peak, amp_rel=amp, yearly_peak_doy=yrs)


# ============================================================================ station time-lapse
class St:
    def __init__(self, code, ident, G):
        self.code = code
        m, rf, rf_raw, t, g, _ = load_rf(code)
        self.t = t
        ig = g.index(2.5)
        self.R = rf[:, ig, 0].astype(np.float64)
        self.Rraw = rf_raw[:, ig, 0].astype(np.float64)
        self.time = m.time.values.astype('datetime64[s]').astype(np.int64) / 86400.0
        self.gidx = np.array([G.idx[e] for e in m.evid.values])
        geo = ((m.baz.values // 30).astype(int) * 2 + (m.p.values >= P_REF).astype(int))
        self.M = np.zeros((len(m), 24)); self.M[np.arange(len(m)), geo] = 1.0
        self.ident = ident
        self.ph = [('t', 'mo', ident['obs_t']), ('m', 'mo', ident['obs_m'])]
        for p in str(ident.get('multiples') or '').split(';'):
            if p:
                self.ph.append((p, 'raw', ident[f'obs_{p}']))
        S = np.load(os.path.join(OUT, f'synth_{code}.npz'))
        assert np.all(S['evid'] == m.evid.values.astype(str))
        self.syn_models = list(S['models'])
        self.SR, self.SRraw = S['R'][:, IG].astype(np.float64), S['Rraw'][:, IG].astype(np.float64)
        self._warp_synth()

    def _warp_synth(self):
        """piecewise-linear time map synthetic -> observed phase times (t and m), applied to every synthetic trace."""
        t = self.t
        ts, tm = self.ident['syn_t'], self.ident['syn_m']
        ot, om = self.ident['obs_t'], self.ident['obs_m']
        src = np.interp(t, [ot, om], [ts, tm], left=np.nan, right=np.nan)
        src = np.where(t < ot, t + (ts - ot), np.where(t > om, t + (tm - om), src))
        self.SRw = np.array([[np.interp(src, t, x) for x in Rm] for Rm in self.SR])
        # amplitude scaling: observed t amplitude / synthetic t amplitude
        self.syn_scale = abs(self.ident['obs_t_amp'] / self.ident['syn_t_amp'])

    # ---------------------------------------------------------------- stacks / measurement
    def stacks(self, wA, wB, R, Rraw):
        sA, sB = self.M.T @ wA, self.M.T @ wB
        com = (sA > 0) & (sB > 0)
        nA = int(((self.M[:, com].sum(1) > 0) & (wA > 0)).sum()); nB = int(((self.M[:, com].sum(1) > 0) & (wB > 0)).sum())
        if nA < 3 or nB < 3:
            return None, nA, nB
        w = (sA * sB / (sA + sB))[com]
        w = w / w.sum()
        out = []
        for X in (R, Rraw):
            MA = (self.M[:, com].T @ (wA[:, None] * X)) / sA[com, None]
            MB = (self.M[:, com].T @ (wB[:, None] * X)) / sB[com, None]
            out.append((w @ MA, w @ MB))
        return out, nA, nB

    def measure(self, S):
        (SA, SB), (SAr, SBr) = S
        res = {}
        c = {}
        for name, kind, tc in self.ph:
            a, b = (SA, SB) if kind == 'mo' else (SAr, SBr)
            cc, tau = shift_amp(a, b, self.t, tc - HW, tc + HW)
            res[f'{name}_dt'] = 1000 * tau
            res[f'{name}_amp'] = 100 * (cc - 1)
            c[name] = cc
        res['dT_tm'] = res['m_dt'] - res['t_dt']
        res['ratio_tm'] = 100 * (c['t'] / c['m'] - 1)
        if 'PpPs_t' in c and 'PpPs_m' in c:
            res['dT_PpPs'] = res['PpPs_m_dt'] - res['PpPs_t_dt']
        if 'PpSs_t' in c and 'PpSs_m' in c:
            res['dT_PpSs'] = res['PpSs_m_dt'] - res['PpSs_t_dt']
        return res

    def est(self, lab, bw=None, R=None, Rraw=None):
        L = lab[self.gidx]
        bw = np.ones(len(L)) if bw is None else bw[self.gidx]
        wA = (L == 1) * bw; wB = (L == 0) * bw
        S, nA, nB = self.stacks(wA.astype(float), wB.astype(float), self.R if R is None else R,
                                self.Rraw if Rraw is None else Rraw)
        if S is None:
            return None, nA, nB
        return self.measure(S), nA, nB

    def absolute(self, lab):
        """absolute picks on the matched-geometry stack of group A (for the time series figure)."""
        L = lab[self.gidx]
        S, nA, nB = self.stacks((L == 1).astype(float), (L == 0).astype(float), self.R, self.Rraw)
        if S is None:
            return {}
        SA = S[0][0]
        out = {}
        for ph, sgn in (('t', -1), ('m', 1)):
            tc = self.ident[f'obs_{ph}']
            out[f'abs_{ph}'], out[f'abs_{ph}_amp'] = pick(SA, self.t, tc - 0.5, tc + 0.5, sgn)
        out['abs_dT_tm'] = out['abs_m'] - out['abs_t']
        out['abs_ratio_tm'] = out['abs_t_amp'] / out['abs_m_amp'] if out['abs_m_amp'] else np.nan
        return out

    # ---------------------------------------------------------------- injections
    def inj_delay(self, lab, X=0.050):
        t = self.t
        ot, om = self.ident['obs_t'], self.ident['obs_m']
        ts = t - X * np.clip((t - ot) / (om - ot), 0, 1)
        R = self.R.copy(); Rr = self.Rraw.copy()
        L = lab[self.gidx]
        for k in np.where(L == 1)[0]:
            R[k] = np.interp(ts, t, self.R[k]); Rr[k] = np.interp(ts, t, self.Rraw[k])
        return R, Rr

    def inj_model(self, lab, model):
        i0, i1 = self.syn_models.index('base'), self.syn_models.index(model)
        D = (self.SRw[i1] - self.SRw[i0]) * self.syn_scale
        L = lab[self.gidx]
        R = self.R.copy()
        R[L == 1] += D[L == 1]
        return R, self.Rraw

    def sensitivity(self):
        """noise-free: synthetic (warped, unscaled) model vs base, all events, plain mean; raw stacks unchanged."""
        i0 = self.syn_models.index('base')
        out = {}
        B = self.SRw[i0].mean(0)
        for mdl in MODELS_INJ:
            A = self.SRw[self.syn_models.index(mdl)].mean(0)
            out[mdl] = self.measure([(A, B), (self.Rraw.mean(0), self.Rraw.mean(0))])
        return out


def run_station(args):
    code, ident, a = args
    G = GLOBAL['G']; C = GLOBAL['C']; RP = GLOBAL['R']
    S = St(code, ident, G)
    out = dict(sta=code, contrasts={}, windows=[], sens=S.sensitivity(), syn_scale=S.syn_scale)
    boot = RP['boot']
    for name, c in C.items():
        lab = c['lab']
        e, nA, nB = S.est(lab)
        rec = dict(est=e, nA=nA, nB=nB)
        if e is None:
            out['contrasts'][name] = rec; continue
        q = list(e)

        def reps(L_list, bw_list=None):
            arr = np.full((len(L_list), len(q)), np.nan)
            for k, L in enumerate(L_list):
                r, _, _ = S.est(L, None if bw_list is None else bw_list[k])
                if r:
                    arr[k] = [r[x] for x in q]
            return arr
        rec['q'] = q
        rec['boot'] = reps([lab] * len(boot), boot)
        rec['perm'] = reps(c['perm'])
        if 'placebo' in c:
            rec['placebo'] = reps(c['placebo'])
        if 'shift' in c:
            rec['shift'] = reps(c['shift'])
        # injections
        inj = {}
        R, Rr = S.inj_delay(lab)
        r, _, _ = S.est(lab, R=R, Rraw=Rr)
        inj['dT50ms'] = {x: r[x] - e[x] for x in q} if r else {}
        for mdl in MODELS_INJ:
            R, Rr = S.inj_model(lab, mdl)
            r, _, _ = S.est(lab, R=R, Rraw=Rr)
            inj[mdl] = {x: r[x] - e[x] for x in q} if r else {}
        rec['inj'] = inj
        out['contrasts'][name] = rec
    # 1.5-yr windows: each vs rest; boot SE; chi2 / trend nulls by time permutation
    W = G.windows()
    labs = G.window_labels(G.time)
    wrows = []
    for (w0, w1), lab in zip(W, labs):
        e, nA, nB = S.est(lab)
        if e is None or nA < 5:
            continue
        bs = np.array([[r[x] for x in e] if r else [np.nan] * len(e)
                       for r in (S.est(lab, boot[k])[0] for k in range(min(len(boot), a.nboot_win)))])
        d = dict(sta=S.code, w0=w0, w1=w1, tmid=0.5 * (w0 + w1), nA=nA)
        for j, x in enumerate(e):
            d[x] = e[x]; d[x + '_se'] = rsd(bs[:, j])
        d.update(S.absolute(lab))
        wrows.append(d)
    out['windows'] = wrows
    # time-permutation replicates of the window series
    tp = []
    for tt in RP['tperm']:
        rr = []
        for (w0, w1), lab in zip(W, G.window_labels(tt)):
            e, nA, nB = S.est(lab)
            rr.append(dict(w0=w0, nA=nA, **(e or {})))
        tp.append(pd.DataFrame(rr))
    out['tperm'] = tp
    print(code, 'done', flush=True)
    return out


GLOBAL = {}


# ============================================================================ statistics / combination
def rsd(x):
    """robust SD (1.4826 x median absolute deviation): the shift/amplitude estimator has heavy tails when a weak
    phase lets the delay run to its +-150 ms limit or the amplitude ratio's denominator approaches zero."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def pct(v, arr):
    arr = arr[np.isfinite(arr)]
    if not len(arr) or not np.isfinite(v):
        return np.nan
    return 100 * np.mean(arr < v) + 50 * np.mean(arr == v)


def contrast_table(results, usable_net):
    rows = []
    allq = sorted({x for r in results for c in r['contrasts'].values() if c.get('q') for x in c['q']})
    names = list(results[0]['contrasts'])
    for name in names:
        per = {}
        for r in results:
            c = r['contrasts'][name]
            if c.get('est') is None:
                continue
            per[r['sta']] = c
            for j, x in enumerate(c['q']):
                d = dict(sta=r['sta'], contrast=name, quantity=x, est=c['est'][x], nA=c['nA'], nB=c['nB'],
                         boot_se=rsd(c['boot'][:, j]), perm_sd=rsd(c['perm'][:, j]),
                         perm_pct=pct(c['est'][x], c['perm'][:, j] - np.nanmean(c['perm'][:, j])))
                for nk in ('placebo', 'shift'):
                    if nk in c and len(c[nk]) and len(np.unique(np.round(c[nk][:, j][np.isfinite(c[nk][:, j])], 6))) >= 20:
                        d[f'{nk}_sd'] = rsd(c[nk][:, j]); d[f'n_{nk}'] = int(np.isfinite(c[nk][:, j]).sum())
                        d[f'{nk}_pct'] = pct(c['est'][x], c[nk][:, j])
                d['null_sd'] = np.nanmax([d['perm_sd'], d.get('placebo_sd', np.nan), d.get('shift_sd', np.nan)])
                d['z'] = d['est'] / d['null_sd'] if d['null_sd'] > 0 else np.nan
                d['det2sigma'] = 2 * d['null_sd']
                for k, v in c['inj'].items():
                    d[f'inj_{k}'] = v.get(x, np.nan)
                fl = [FLAG[k] for k in FLAG if k[0] == r['sta'] and k[1] in (name, '*')]
                d['flag'] = '; '.join(fl)
                rows.append(d)
        # network: inverse bootstrap-variance weighted mean over usable stations; same weights on every replicate
        for x in allq:
            sts = [s for s in per if s in usable_net and x in per[s]['q'] and not
                   any(k[0] == s and k[1] == name for k in FLAG)]
            if len(sts) < 2:
                continue
            j = {s: per[s]['q'].index(x) for s in sts}
            se = np.array([rsd(per[s]['boot'][:, j[s]]) for s in sts])
            w = 1 / np.maximum(se, 1e-6) ** 2; w = w / w.sum()
            est = float(np.sum(w * np.array([per[s]['est'][x] for s in sts])))

            def comb(key):
                if not all(key in per[s] and len(per[s][key]) for s in sts):
                    return None
                if key in ('placebo', 'shift'):     # stations without >= 20 distinct null values do not contribute
                    ok = [len(np.unique(np.round(per[s][key][:, j[s]][np.isfinite(per[s][key][:, j[s]])], 6))) >= 20
                          for s in sts]
                    if not any(ok):
                        return None
                    M = np.array([per[s][key][:, j[s]] if o else np.full(len(per[s][key]), np.nan)
                                  for s, o in zip(sts, ok)])
                    return np.nansum(w[:, None] * M, 0) / np.nansum(w[:, None] * np.isfinite(M), 0)
                M = np.array([per[s][key][:, j[s]] for s in sts])        # (nsta, nrep)
                M = np.where(np.isfinite(M), M, np.nan)
                return np.nansum(w[:, None] * M, 0) / np.nansum(w[:, None] * np.isfinite(M), 0)
            d = dict(sta='NETWORK', contrast=name, quantity=x, est=est, n_sta=len(sts), stations=','.join(sts))
            b = comb('boot'); p = comb('perm')
            d['boot_se'] = rsd(b); d['perm_sd'] = rsd(p); d['perm_pct'] = pct(est, p - np.nanmean(p))
            # naive (independent-station) SE for comparison with the shared-event bootstrap SE
            d['naive_se'] = float(np.sqrt(np.sum(w ** 2 * se ** 2)))
            for nk in ('placebo', 'shift'):
                v = comb(nk)
                if v is not None and len(np.unique(np.round(v[np.isfinite(v)], 6))) >= 20:
                    d[f'{nk}_sd'] = rsd(v); d[f'{nk}_pct'] = pct(est, v); d[f'n_{nk}'] = int(np.isfinite(v).sum())
            d['null_sd'] = np.nanmax([d['perm_sd'], d.get('placebo_sd', np.nan), d.get('shift_sd', np.nan)])
            d['z'] = est / d['null_sd'] if d['null_sd'] > 0 else np.nan
            d['det2sigma'] = 2 * d['null_sd']
            for k in per[sts[0]]['inj']:
                d[f'inj_{k}'] = float(np.sum(w * np.array([per[s]['inj'][k].get(x, np.nan) for s in sts])))
            rows.append(d)
    return pd.DataFrame(rows)


def variability(results, usable_net, quantities=('dT_tm', 't_amp', 'm_amp', 'ratio_tm', 't_dt', 'm_dt')):
    """chi2 of the 1.5-yr window series about their weighted mean and weighted linear trend; nulls from the
    time-permutation replicates (same SEs)."""
    rows = []
    for r in results:
        W = pd.DataFrame(r['windows'])
        if len(W) < 3:
            continue
        for x in quantities:
            if x not in W:
                continue
            se = W[x + '_se'].values
            ok = np.isfinite(W[x].values) & (se > 0)
            if ok.sum() < 3:
                continue

            def stats(v, tm, se):
                w = 1 / se ** 2
                mu = np.sum(w * v) / np.sum(w)
                chi2 = np.sum(w * (v - mu) ** 2)
                xx = (tm - np.average(tm, weights=w)) / YR
                sl = np.sum(w * xx * (v - mu)) / np.sum(w * xx * xx)
                return chi2, sl
            c2, sl = stats(W[x].values[ok], W.tmid.values[ok], se[ok])
            nc, ns = [], []
            for D in r['tperm']:
                if x not in D:
                    continue
                D2 = D.set_index('w0').reindex(W.w0.values[ok])
                v = D2[x].values
                k = np.isfinite(v)
                if k.sum() < 3:
                    continue
                a, b = stats(v[k], W.tmid.values[ok][k], se[ok][k])
                nc.append(a); ns.append(b)
            nc, ns = np.array(nc), np.array(ns)
            rows.append(dict(sta=r['sta'], quantity=x, n_win=int(ok.sum()), chi2=c2, dof=int(ok.sum() - 1),
                             chi2_perm_pct=pct(c2, nc), chi2_perm_median=np.median(nc) if len(nc) else np.nan,
                             trend_per_yr=sl, trend_perm_sd=rsd(ns) if len(ns) else np.nan,
                             trend_z=sl / rsd(ns) if len(ns) else np.nan, trend_perm_pct=pct(sl, ns)))
    return pd.DataFrame(rows)


# ============================================================================ figures
def fig_static(ids, extras):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    codes = list(ids.sta)
    nc = 3
    nr = int(np.ceil(len(codes) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(18, 2.6 * nr + 1), squeeze=False)
    for ax, code in zip(axs.ravel(), codes):
        r = ids.set_index('sta').loc[code]; X = extras[code]; t = X['t']
        ax.fill_between(t, X['A0'] - 2 * X['A0se'], X['A0'] + 2 * X['A0se'], color='0.8')
        ax.plot(t, X['A0'], 'k', lw=1.2, label='data A0 (moveout-corr.)')
        ax.plot(t, X['syn'], 'r', lw=0.9, label='PyRaysum Bloch-2023 model')
        sh = 6.0 + 0.0
        ax.plot(t, X['praw'] - 0.12, color='0.3', lw=0.8, label='data raw stack (offset)')
        ax.plot(t, X['mult'] - 0.12, 'm', lw=0.8, label='synthetic multiples only (offset)')
        for ph, col in (('t', 'royalblue'), ('m', 'navy')):
            ax.axvline(r[f'syn_{ph}'], color=col, ls=':', lw=1)
            if np.isfinite(r[f'obs_{ph}']):
                ax.plot(r[f'obs_{ph}'], r[f'obs_{ph}_amp'], 'o', color=col if r[f'ok_{ph}'] else 'w', mec=col, ms=6)
        for ph in ('PpPs_t', 'PpSs_t', 'PpPs_m', 'PpSs_m'):
            if np.isfinite(r.get(f'obs_{ph}', np.nan)):
                ax.plot(r[f'obs_{ph}'], r[f'obs_{ph}_amp'] - 0.12, 's', color='m' if r[f'ok_{ph}'] else 'w', mec='m', ms=5)
        ax.set_xlim(-0.5, 14); ax.set_ylim(-0.25, 0.2); ax.axhline(0, color='k', lw=0.4); ax.axhline(-0.12, color='k', lw=0.3)
        tag = 'USABLE PAIR' if r['pair'] else ('t only' if r['ok_t'] else ('m only' if r['ok_m'] else 'not identified'))
        ax.set_title(f"{code} n={r['n']}  t {r['z_t_km']:.1f} km, m {r['z_m_km']:.1f} km, dip {r['dip']:.0f}: {tag}"
                     f"  (cc {r['cc_syn']:.2f})", fontsize=8.5, color='g' if r['pair'] else 'k')
    for ax in axs.ravel()[len(codes):]:
        ax.axis('off')
    axs[0, 0].legend(fontsize=6.5, loc='upper right')
    fig.supxlabel('time after P (s); upper traces moveout-corrected to p=0.06 s/km, lower (offset -0.12) raw')
    fig.suptitle('LVZ top (t, negative Ps, blue) and slab Moho (m, positive Ps, navy) identification; dotted = synthetic '
                 'phase time; filled marker = |A0| >= 3 SE; magenta squares = multiples on raw stacks', fontsize=11)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig(os.path.join(FIG, 'sci_rf_lvz_static.png'), dpi=90)
    plt.close(fig)


def fig_timeseries(Wdf, net_codes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    codes = list(Wdf.sta.unique())
    qs = [('dT_tm', 'differential time m - t (ms)', (-200, 200)), ('ratio_tm', 'A_t / A_m change (%)', (-150, 150)),
          ('t_amp', 'A_t change (%)', (-150, 150)), ('m_amp', 'A_m change (%)', (-100, 100))]
    fig, axs = plt.subplots(len(qs), 1, figsize=(14, 13), sharex=True)
    for i, (q, lab, yl) in enumerate(qs):
        ax = axs[i]
        for j, c in enumerate(codes):
            D = Wdf[Wdf.sta == c]
            x = pd.to_datetime(D.tmid * 86400e9) + pd.Timedelta(days=20 * (j - len(codes) / 2))
            ax.errorbar(x, D[q], yerr=2 * D[q + '_se'], fmt='o-', ms=3, lw=0.8, capsize=2, label=c)
        for k, v in QUAKES.items():
            ax.axvline(pd.Timestamp(v), color={'Petrolia': 'darkorange', 'Ferndale': 'm', 'M7': 'r'}[k], ls='--', lw=1)
        ax.axhline(0, color='k', lw=0.5); ax.set_ylabel(lab); ax.set_ylim(yl)
        if i == 0:
            ax.legend(fontsize=7, ncol=6)
    axs[0].set_title('1.5-yr windows, each compared with all other events at matched geometry (2-sigma event bootstrap)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_rf_lvz_timeseries.png'), dpi=95)
    plt.close(fig)


def fig_contrasts(T):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cons = ['step_Petrolia', 'step_Ferndale', 'step_M7', 'post2021', 'lfe_high', 'season_cos', 'season_sin']
    clab = ['Petrolia step', 'Ferndale step', 'M7 step', 'after vs before 2021-12-20', 'high vs low LFE activity',
            'LFE-peak half-year vs opposite', 'quadrature half-years']
    qs = [('dT_tm', 'differential time m - t (ms)'), ('ratio_tm', 'A_t/A_m change (%)'),
          ('t_amp', 'A_t change (%)'), ('m_amp', 'A_m change (%)')]
    sts = [s for s in T.sta.unique() if s != 'NETWORK'] + ['NETWORK']
    fig, axs = plt.subplots(1, len(qs), figsize=(18, 9), sharey=True)
    for i, (q, lab) in enumerate(qs):
        ax = axs[i]
        for ci, c in enumerate(cons):
            for si, s in enumerate(sts):
                r = T[(T.sta == s) & (T.contrast == c) & (T.quantity == q)]
                if not len(r):
                    continue
                r = r.iloc[0]
                y = ci + 0.8 * (si / max(len(sts) - 1, 1) - 0.5)
                col = 'k' if s == 'NETWORK' else f'C{si % 10}'
                ax.plot([-r.det2sigma, r.det2sigma], [y, y], color='0.85', lw=4, solid_capstyle='butt')
                ax.errorbar(r.est, y, xerr=2 * r.boot_se, fmt='D' if s == 'NETWORK' else 'o', color=col, ms=4,
                            lw=1, label=s if (i == 0 and ci == 0) or (i == 0 and s not in [l for l in ax.get_legend_handles_labels()[1]]) else None)
        ax.axvline(0, color='k', lw=0.5); ax.set_xlabel(lab)
        lim = np.nanpercentile(np.abs(T[T.quantity == q].det2sigma), 90) * 1.3 if (T.quantity == q).any() else 100
        ax.set_xlim(-lim, lim)
    axs[0].set_yticks(range(len(cons))); axs[0].set_yticklabels(clab, fontsize=9)
    axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    fig.legend(list(seen.values()), list(seen.keys()), loc='lower center', ncol=8, fontsize=8)
    fig.suptitle('LVZ top/bottom conversions: later (or high-state) minus earlier (low-state) group, matched geometry.\n'
                 'Markers: estimate +- 2 bootstrap SE; grey bars: +- 2-sigma null (max of permutation / placebo / '
                 'circular-shift SD); diamonds: network (shared-event-aware nulls)', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(os.path.join(FIG, 'sci_rf_lvz_contrasts.png'), dpi=95)
    plt.close(fig)


def fig_sensitivity(results, DL, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    codes = [r['sta'] for r in results]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    ax = axs[0]
    for j, c in enumerate(codes):
        S = np.load(os.path.join(OUT, f'synth_{c}.npz'))
        mo = list(S['models']); t = S['t']
        b = S['R'][mo.index('base'), IG].mean(0)
        for mdl, ls in (('lvl_dvs-0.10', '-'), ('lvl_dvs+0.10', ':')):
            ax.plot(t, -0.12 * j + 5 * (S['R'][mo.index(mdl), IG].mean(0) - b), ls, color=f'C{j}', lw=1)
        ax.plot(t, -0.12 * j + b, color='0.6', lw=0.8)
        ax.text(6.2, -0.12 * j, c, fontsize=8)
    ax.set_xlim(0, 7); ax.set_yticks([]); ax.set_xlabel('time after P (s)')
    ax.set_title('Synthetic base RF (grey) and 5x (LVL Vs -0.10 km/s minus base) (solid; +0.10 dotted)', fontsize=9)
    ax = axs[1]
    D = DL[DL.unit == 'dVs_kms']
    for j, c in enumerate(D.sta.unique()):
        d = D[D.sta == c]
        ax.scatter(d.contrast, d.det_best, color=f'C{j}', label=c, s=25)
    ax.set_yscale('log'); ax.set_ylabel('2-sigma detectable |LVL Vs change| (km/s), best observable')
    ax.axhspan(0.05, 0.10, color='g', alpha=0.15, label='Gosselin et al. 2020 scale (0.05-0.1 km/s)')
    ax.tick_params(axis='x', rotation=30); ax.legend(fontsize=7, ncol=2)
    ax.set_title('Detection limits in model units (noise-free PyRaysum sensitivity, nulls from data)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'sci_rf_lvz_sensitivity.png'), dpi=95)
    plt.close(fig)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--nboot', type=int, default=300)
    ap.add_argument('--nboot_win', type=int, default=150)
    ap.add_argument('--nperm', type=int, default=500)
    ap.add_argument('--nperm_win', type=int, default=200)
    ap.add_argument('--nshift', type=int, default=300)
    ap.add_argument('--stations', default='')
    ap.add_argument('--from_pkl', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    codes = sorted(f[6:-4] for f in os.listdir(OUT) if f.startswith('synth_') and f.endswith('.npz'))
    codes = [c for c in codes if os.path.exists(os.path.join(RFOUT, f'static_{c}.npz'))]
    rows, extras = [], {}
    for c in codes:
        r, x = identify(c, rng)
        rows.append(r); extras[c] = x
    ids = pd.DataFrame(rows)
    ids.to_csv(os.path.join(OUT, 'stations_lvz.csv'), index=False)
    fig_static(ids, extras)
    print(ids[['sta', 'n', 'syn_t', 'obs_t', 'obs_t_amp', 'obs_t_z', 'syn_m', 'obs_m', 'obs_m_amp', 'obs_m_z', 'pair',
               'cc_syn', 'multiples', 'timelapse']].round(3).to_string(), flush=True)
    tl = ids[ids.timelapse]
    if a.stations:
        tl = ids[ids.sta.isin(a.stations.split(','))]
    G = Glob(list(tl.sta))
    C = G.contrasts()
    RP = G.replicates(a, C)
    GLOBAL.update(G=G, C=C, R=RP)
    print('global events', G.n, 'LFE season peak doy', round(G.season['peak_doy'], 1), 'amp', round(G.season['amp_rel'], 3),
          {k: int((v['lab'] == 1).sum()) for k, v in C.items()}, flush=True)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(r.sta, r._asdict(), a) for r in tl.itertuples(index=False)]
    import multiprocessing as mp
    if a.from_pkl:                       # re-derive tables / figures from saved replicates
        import pickle
        results = pickle.load(open(os.path.join(OUT, 'replicates.pkl'), 'rb'))['results']
    else:
        with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context('fork')) as ex:
            results = list(ex.map(run_station, jobs))
    # estimator check: the +50 ms differential-delay injection must be recovered (median over contrasts 25-75 ms)
    rec = {r['sta']: float(np.nanmedian([c['inj']['dT50ms'].get('dT_tm', np.nan) for c in r['contrasts'].values()
                                         if c.get('inj')])) for r in results}
    for r in results:
        if not 25 <= rec[r['sta']] <= 75:
            NET_EXCLUDE[r['sta']] = f"+50 ms injection recovered as {rec[r['sta']]:.0f} ms (t phase too weak; estimator fails)"
    usable_net = [r['sta'] for r in results if r['sta'] not in NET_EXCLUDE]
    import pickle
    with open(os.path.join(OUT, 'replicates.pkl'), 'wb') as f:
        pickle.dump(dict(results=results, contrasts={k: {kk: vv for kk, vv in v.items() if kk == 'lab' or kk == 'kind'}
                                                      for k, v in C.items()}, evid=G.evid, time=G.time), f)
    T = contrast_table(results, usable_net)
    T.to_csv(os.path.join(OUT, 'contrasts_lvz.csv'), index=False)
    Wdf = pd.DataFrame([w for r in results for w in r['windows']])
    Wdf['t0'] = pd.to_datetime(Wdf.w0 * 86400e9); Wdf['t1'] = pd.to_datetime(Wdf.w1 * 86400e9)
    Wdf.to_csv(os.path.join(OUT, 'windows_lvz.csv'), index=False)
    V = variability(results, usable_net)
    V.to_csv(os.path.join(OUT, 'variability_lvz.csv'), index=False)
    # sensitivities and detection limits in model units
    srows, drows = [], []
    for r in results:
        for mdl, d in r['sens'].items():
            unit, val = MODEL_UNIT[mdl]
            for q, v in d.items():
                srows.append(dict(sta=r['sta'], model=mdl, unit=unit, model_change=val, quantity=q, change=v,
                                  per_unit=v / val))
    Sdf = pd.DataFrame(srows)
    Sdf.to_csv(os.path.join(OUT, 'sensitivity_lvz.csv'), index=False)
    for (sta, con), d in T.groupby(['sta', 'contrast']):
        for unit, mlist in (('dVs_kms', ['lvl_dvs-0.10', 'lvl_dvs+0.10']), ('dVs_pct', ['lvl_pct-4', 'lvl_pct+4'])):
            row = dict(sta=sta, contrast=con, unit=unit)
            best = np.inf
            for q in ('dT_tm', 'ratio_tm', 't_amp', 'm_amp'):
                r = d[d.quantity == q]
                if not len(r):
                    continue
                sts = [sta] if sta != 'NETWORK' else r.stations.iloc[0].split(',')
                s = Sdf[(Sdf.sta.isin(sts)) & (Sdf.model.isin(mlist)) & (Sdf.quantity == q)]
                if not len(s):
                    continue
                slope = float(np.mean([np.abs(s[s.sta == x].per_unit).mean() for x in sts if (s.sta == x).any()]))
                det = r.det2sigma.iloc[0] / slope if slope > 0 else np.inf
                row[f'slope_{q}'] = slope; row[f'det_{q}'] = det
                row[f'est_{q}_units'] = r.est.iloc[0] / slope if slope > 0 else np.nan
                best = min(best, det)
            row['det_best'] = best
            row['sign_note'] = 'est_*_units > 0: S-velocity contrast across the LVZ top increased (LVL Vs decreased) by that amount'
            drows.append(row)
    DL = pd.DataFrame(drows)
    DL.to_csv(os.path.join(OUT, 'detection_lvz.csv'), index=False)
    # figures
    fig_timeseries(Wdf, usable_net)
    fig_contrasts(T)
    fig_sensitivity(results, DL, ids)
    # summary
    key = T[T.quantity.isin(['dT_tm', 'ratio_tm', 't_amp', 'm_amp'])]
    summ = dict(
        stations_identified=ids[['sta', 'n', 'ok_t', 'ok_m', 'pair', 'cc_syn', 'multiples', 'timelapse']].to_dict('records'),
        timelapse_stations=list(tl.sta), network_stations=usable_net, network_excluded=NET_EXCLUDE,
        inj50_recovery_ms=rec,
        n_global_events=int(G.n), lfe_season=G.season,
        lfe_state_fraction_high=float(np.nanmean(G.lfe_ratio >= 1)),
        max_abs_z=key.assign(az=key.z.abs()).sort_values('az', ascending=False).head(15)[
            ['sta', 'contrast', 'quantity', 'est', 'boot_se', 'det2sigma', 'z', 'perm_pct', 'flag']].to_dict('records'),
        network=T[T.sta == 'NETWORK'][['contrast', 'quantity', 'est', 'boot_se', 'naive_se', 'det2sigma', 'z', 'perm_pct']
                                     + [c for c in ('placebo_pct', 'shift_pct') if c in T]].to_dict('records'),
        n_tests=int(len(key)), n_absz_gt2=int((key.z.abs() > 2).sum()), n_absz_gt3=int((key.z.abs() > 3).sum()),
        variability=V.to_dict('records'),
        detection=DL.to_dict('records'))
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    pd.set_option('display.width', 250)
    print(T[T.quantity.isin(['dT_tm', 'ratio_tm', 't_amp', 'm_amp'])][
        ['sta', 'contrast', 'quantity', 'est', 'boot_se', 'det2sigma', 'z', 'perm_pct', 'inj_dT50ms', 'inj_lvl_dvs-0.10']].round(2).to_string())
    print(V.round(2).to_string())
    print(DL.round(3).to_string())


if __name__ == '__main__':
    main()
