#!/usr/bin/env python3
"""LFE attenuation through time: S spectral decay (dt*_S), P spectral decay (dt*_P) and the differential P/S
spectral ratio (d(t*_P - t*_S), a Qp/Qs proxy), per family-station, from per-event multitaper band powers
(OUT/raw/fam{F}.h5 from sci_lfe_phases_extract.py; incoherent power averages, so no stacking-jitter loss).

What is new relative to data/sci/source_spectra/ (S-window HFI/MID indices, pooled, null): P-window spectra and the
P/S ratio; per-station spectral-decay slopes expressed as dt* (s); a long-term P/S-ratio-vs-travel-time Qp/Qs fit.

Per window (event set) and channel: noise-corrected mean power  Pw(f) = mean 10^SP - mean 10^NP (bands with
Pw > 1x mean noise and f in 2-10 Hz).  Two windows A, B:  ln(PwA/PwB)(f) = c - 2 pi f dt*  (power, so 2 pi);
dt* > 0 = A more attenuated.  Station value: horizontals averaged for S, Z for P.  Families combined per station
by inverse bootstrap variance; network = mean over stations (equal weights), PB boreholes vs surface separately.
Events: isolated LFEs, size-matched between A and B (histogram reweighting on the event's network-median 2-3 Hz
S power) to remove source-size (corner-frequency) selection effects.
Tests: steps at Petrolia, Ferndale, M7 (windows -W..0 vs 0..W, W = 180 d and 60 d; the M7 'after' window starts at
the restart 2024-12-28), restart (M7 0-60 d) vs 180 d pre-M7, post-2021 vs pre-2021 long-term, yearly series vs
all-time reference.  Significance from >=200 placebo dates (2009-2024, |date - quake| > W + 30 d).
Long-term Qp/Qs: per family-station slope s_PS of ln(PwP/PwS) vs f (all-time); regress s_PS = c_F - 2 pi T_S
(1/(1.78 Qp) - 1/Qs) ... reported as the slope of s_PS vs T_S with family fixed effects.
Outputs OUT/atten_tests.csv, OUT/atten_yearly.csv, OUT/atten_station_steps.csv, OUT/atten_qpqs.csv,
figures/sci_lfe_phases_atten.png
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_lfe_phases_common import OUT, FIG, QUAKES, RESTART, epoch_s, geometry  # noqa: E402

EDGES = np.geomspace(1.0, 15.0, 13)
FC = np.sqrt(EDGES[:-1] * EDGES[1:])
FB = (FC >= 2.0) & (FC <= 10.0)
DAY = 86400.0
QT = dict(zip(QUAKES, epoch_s(list(QUAKES.values()))))
QT_RESTART = float(epoch_s([RESTART])[0])
NBOOT = 40
EXCL = {   # instrument windows (from the SWS/deep-shallow work): station -> list of (start, end)
    'PB.B933': [('2017-01-01', '2021-12-01'), ('2024-08-01', '2027-01-01')],
    'PB.B935': [('2024-05-01', '2027-01-01')],
    'BK.RVIT': [('2022-12-20', '2024-06-21')], 'BK.BRIC': [('2022-12-01', '2024-05-22')],
    'BK.BJES': [('2023-01-01', '2024-05-23')], 'NC.KMR': [('2025-09-01', '2027-01-01')]}


SIZE_BINS = np.arange(-2.0, 2.01, 0.25)
NSZ = len(SIZE_BINS) + 1


def bin_edges():
    e = epoch_s(pd.date_range('2008-01-01', '2027-01-01', freq='SMS'))     # 1st and 15th of each month
    return np.sort(np.r_[e, list(QT.values()), QT_RESTART])


def aggregate(F):
    """per family: sums over (time bin, channel, size bin) of linear S, P, noise power and squares."""
    with h5py.File(f'{OUT}/raw/fam{F}.h5', 'r') as h:
        PP, SP, NP = (h[k][:].astype(np.float32) for k in ('PP', 'SP', 'NP'))
        iso = h['iso'][:]
        te = epoch_s(pd.to_datetime(h['time_utc'][:].astype(str)))
        chan = [c.decode() for c in h['chan'][:]]
    sta = np.array(['.'.join(c.split('.')[:2]) for c in chan])
    isZ = np.array([c[-1] == 'Z' for c in chan])
    ok = np.ones(SP.shape[:2], bool)
    for k, s_ in enumerate(sta):
        for a, b in EXCL.get(s_, []):
            ok[(te >= epoch_s([a])[0]) & (te < epoch_s([b])[0]), k] = False
    lo = (FC >= 2.0) & (FC <= 3.1)
    sz = SP[:, :, lo].mean(2)
    sz = sz - np.nanmedian(sz, 0, keepdims=True)
    size = np.nanmedian(np.where(isZ[None, :], np.nan, sz), 1)
    E = bin_edges()
    keep = iso & np.isfinite(size)
    tb = np.searchsorted(E, te, side='right') - 1
    sb = np.digitize(size, SIZE_BINS)
    nb, nch = len(E) - 1, len(chan)
    A = {}
    nlin = 10 ** NP
    nbb = np.nanmean(NP[:, :, FB], 2)                         # broadband noise level (log10, rel. channel median)
    q80 = np.nanpercentile(nbb, 80, axis=0)
    wev = 10 ** (-np.clip(np.nan_to_num(nbb, nan=0.0), -2.0, 3.0)).astype(np.float64)                                         # event weight = 1 / broadband noise power
    for key, X in (('S', SP), ('P', PP)):
        lin = 10 ** X
        good = (keep[:, None] & ok & np.isfinite(lin).all(2) & np.isfinite(nlin).all(2) & (nbb <= q80[None, :]) & (nbb > -2.0)
                & (np.nanmax(X, 2) < 3.0))
        sw = np.zeros((nb, nch, NSZ))
        sw2 = np.zeros((nb, nch, NSZ))
        sx = np.zeros((nb, nch, NSZ, 12))
        sx2 = np.zeros((nb, nch, NSZ, 12))
        sn = np.zeros((nb, nch, NSZ, 12))
        for k in range(nch):
            g = np.where(good[:, k])[0]
            if not len(g):
                continue
            idx = (tb[g], np.full(len(g), k), sb[g])
            w = wev[g, k]
            np.add.at(sw, idx, w)
            np.add.at(sw2, idx, w ** 2)
            np.add.at(sx, idx, w[:, None] * lin[g, k])
            np.add.at(sx2, idx, (w ** 2)[:, None] * lin[g, k] ** 2)
            np.add.at(sn, idx, w[:, None] * nlin[g, k])
        cnt = np.zeros((nb, nch, NSZ))
        for k in range(nch):
            g = np.where(good[:, k])[0]
            np.add.at(cnt, (tb[g], np.full(len(g), k), sb[g]), 1)
        A[key] = (sw, sw2, sx, sx2, sn, cnt)
    return dict(A=A, E=E, chan=chan, sta=sta, isZ=isZ)


def wsum(arr, i0, i1):
    return arr[i0:i1].sum(0)


def window(Ag, i0, i1, wsz=None):
    """sum over time bins [i0, i1), size-reweighted (wsz (nch, NSZ) or None) ->
    noise-corrected weighted mean power Pw (nch, 12), se (nch, 12), weighted mean noise (nch, 12), n events (nch),
    size histogram (nch, NSZ)"""
    sw, sw2, sx, sx2, sn, cnt = (wsum(x, i0, i1) for x in Ag)
    hist = cnt.copy()
    if wsz is not None:
        sw, sx, sn, cnt = sw * wsz, sx * wsz[..., None], sn * wsz[..., None], cnt * wsz
        sw2, sx2 = sw2 * wsz ** 2, sx2 * wsz[..., None] ** 2
    W, W2 = sw.sum(1), sw2.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = sx.sum(1) / W[:, None]
        nm = sn.sum(1) / W[:, None]
        # var of weighted mean ~ sum w^2 (x - m)^2 / W^2
        v = (sx2.sum(1) - m ** 2 * W2[:, None]) / W[:, None] ** 2   # approx. (sum w^2 x ~ m sum w^2)
        se = np.sqrt(np.maximum(v, 0))
    return m - nm, se, nm, cnt.sum(1), hist


def dtstar(wA, wB, nmin=30):
    """-> dt* (nch), se (nch) from two windows"""
    PA, sA, nA, cA, _ = wA
    PB, sB, nB, cB, _ = wB
    nch = PA.shape[0]
    dt, se = np.full(nch, np.nan), np.full(nch, np.nan)
    for k in range(nch):
        if cA[k] < nmin or cB[k] < nmin:
            continue
        m = FB & (PA[k] > 0.3 * nA[k]) & (PB[k] > 0.3 * nB[k])
        if m.sum() < 4:
            continue
        y = np.log(PA[k, m] / PB[k, m])
        vy = (sA[k, m] / PA[k, m]) ** 2 + (sB[k, m] / PB[k, m]) ** 2
        w = 1 / np.maximum(vy, 1e-6)
        f = FC[m]
        fm = np.sum(w * f) / w.sum()
        sl = np.sum(w * (f - fm) * (y - np.sum(w * y) / w.sum())) / np.sum(w * (f - fm) ** 2)
        dt[k] = -sl / (2 * np.pi)
        se[k] = np.sqrt(1 / np.sum(w * (f - fm) ** 2)) / (2 * np.pi)
    return dt, se


def compare(G, i_a0, i_a1, i_b0, i_b1, match=True):
    """windows given as bin index ranges -> per-station rows (dtS from horizontals, dtP from Z, dtPS)"""
    out = {}
    for key in ('S', 'P'):
        Ag = G['A'][key]
        wA = window(Ag, i_a0, i_a1)
        wsz = None
        if match:
            hB = wsum(Ag[5], i_b0, i_b1)
            with np.errstate(invalid='ignore', divide='ignore'):
                wsz = np.where(hB > 0, wA[4] / hB, 0.0)
                wsz = wsz * (hB.sum(1) / np.maximum((wsz * hB).sum(1), 1e-9))[:, None]   # keep n_B
        wB = window(Ag, i_b0, i_b1, wsz)
        out[key] = dtstar(wA, wB)
    rows = []
    for s_ in np.unique(G['sta']):
        r = dict(sta=s_)
        for key, comp in (('S', False), ('P', True)):
            ks = [k for k in range(len(G['chan'])) if G['sta'][k] == s_ and G['isZ'][k] == comp]
            if key == 'S' and not ks:
                ks = [k for k in range(len(G['chan'])) if G['sta'][k] == s_]
            v, e = out[key][0][ks], out[key][1][ks]
            g = np.isfinite(v) & np.isfinite(e) & (e > 0)
            if g.any():
                r[f'dt{key}'] = float(v[g].mean())
                r[f'se{key}'] = float(np.sqrt(np.sum(e[g] ** 2)) / g.sum())
        if 'dtS' in r and 'dtP' in r:
            r['dtPS'] = r['dtP'] - r['dtS']
            r['sePS'] = float(np.hypot(r['seP'], r['seS']))
        if len(r) > 1:
            rows.append(r)
    return rows


def tests_for_family(args):
    F, n_placebo, seed = args
    rng = np.random.default_rng(seed)
    G = aggregate(F)
    E = G['E']
    out = []

    def run(name, a0, a1, b0, b1, kind, match=True):
        for r in compare(G, a0, a1, b0, b1, match):
            r.update(F=F, test=name, kind=kind)
            out.append(r)
    def ix(t):
        return int(np.argmin(np.abs(E - t)))
    ir = ix(QT_RESTART)
    for W in (180, 60):
        Ws = W * DAY
        for q in ('petrolia', 'ferndale', 'm7'):
            t0 = QT[q]
            ta = QT_RESTART if q == 'm7' else t0
            run(f'step_{q}_{W}d', ix(ta), ix(ta + Ws), ix(t0 - Ws), ix(t0), 'real')
        # placebo edges: 1st and 15th of each month 2009-01 .. 2024-06, > W + 30 d from every quake
        ms = epoch_s(pd.date_range('2009-01-01', '2024-06-01', freq='SMS'))
        ms = ms[np.all(np.abs(ms[:, None] - np.array(list(QT.values()))[None, :]) > (W + 30) * DAY, 1)]
        for t0 in ms:
            run(f'placebo_{W}d_{pd.to_datetime(t0, unit="s").strftime("%Y-%m-%d")}', ix(t0), ix(t0 + Ws),
                ix(t0 - Ws), ix(t0), f'placebo_{W}d')
    i0 = ix(QT['m7'])
    run('restart_0_60d', i0, ix(QT['m7'] + 60 * DAY), ix(QT['m7'] - 180 * DAY), i0, 'real')
    run('restart_0_60d_unmatched', i0, ix(QT['m7'] + 60 * DAY), ix(QT['m7'] - 180 * DAY), i0, 'real', match=False)
    ei = {q: ix(QT[q]) for q in QT}
    i22 = int(np.argmin(np.abs(E - epoch_s(['2022-01-01'])[0])))
    i15 = int(np.argmin(np.abs(E - epoch_s(['2015-01-01'])[0])))
    run('longterm_post2021', i22, ei['m7'], i15, ei['petrolia'], 'real')
    # placebo for the long-term test: same lengths, split points 2011-2018 (pre-Petrolia only)
    for y in range(2011, 2019):
        for mth in (1, 7):
            ic = int(np.argmin(np.abs(E - epoch_s([f'{y}-{mth:02d}-01'])[0])))
            run(f'placebo_long_{y}-{mth:02d}', ic, min(ic + 70, ei['petrolia']), max(ic - 166, 0), ic, 'placebo_long')
    for y in range(2008, 2027):
        a0 = int(np.argmin(np.abs(E - epoch_s([f'{y}-01-01'])[0])))
        a1 = int(np.argmin(np.abs(E - epoch_s([f'{y + 1}-01-01'])[0])))
        run(f'year_{y}', a0, a1, 0, len(E) - 1, 'year')
    # long-term P/S slope (all-time, all isolated) per station: Z P vs horizontal S
    qp = []
    wP = window(G['A']['P'], 0, len(E) - 1)
    wS = window(G['A']['S'], 0, len(E) - 1)
    for s_ in np.unique(G['sta']):
        kz = [k for k in range(len(G['chan'])) if G['sta'][k] == s_ and G['isZ'][k]]
        kh = [k for k in range(len(G['chan'])) if G['sta'][k] == s_ and not G['isZ'][k]]
        if not kz:
            continue
        for kk, comp in [(kz[0], 'Z')] + [(k, 'H') for k in kh]:
            wa = tuple(x[[kz[0]]] for x in wP)
            wb = tuple(x[[kk]] for x in wS)
            v, e = dtstar(wa, wb)
            qp.append(dict(F=F, sta=s_, Scomp=comp, dtPS_abs=v[0], se=e[0]))
    return pd.DataFrame(out), pd.DataFrame(qp)


def pool(R, key):
    """per test: per-station inverse-variance family mean; network = equal-weight mean over stations."""
    R = R[np.isfinite(R[f'dt{key}']) & np.isfinite(R[f'se{key}']) & (R[f'se{key}'] > 0)].copy()
    R['w'] = 1 / R[f'se{key}'] ** 2
    R['wv'] = R.w * R[f'dt{key}']
    S = R.groupby(['test', 'kind', 'sta'], as_index=False).agg(w=('w', 'sum'), wv=('wv', 'sum'))
    S['v'] = S.wv / S.w
    S['se'] = 1 / np.sqrt(S.w)
    S['bh'] = S.sta.str.startswith('PB.')
    rows = []
    for grp, s in (('all', S), ('PB', S[S.bh]), ('surface', S[~S.bh])):
        g = s.groupby(['test', 'kind']).agg(value=('v', 'mean'), ss=('se', lambda x: np.sqrt(np.sum(x ** 2))),
                                            n_sta=('v', 'size')).reset_index()
        g['boot_se'] = g.ss / g.n_sta
        g['group'] = grp
        g['q'] = key
        rows.append(g.drop(columns='ss'))
    return pd.concat(rows), S


def analyse():
    R = pd.read_csv(f'{OUT}/atten_station_steps.csv')
    tests, stas = [], []
    for key in ('S', 'P', 'PS'):
        T, S = pool(R, key)
        S['q'] = key
        tests.append(T)
        stas.append(S)
    T = pd.concat(tests)
    S = pd.concat(stas)
    rows = []
    for (q, grp), g in T.groupby(['q', 'group']):
        for W in (180, 60):
            pl = g[g.kind == f'placebo_{W}d'].value.values
            for _, r in g[(g.kind == 'real') & g.test.str.contains(f'_{W}d')].iterrows():
                rows.append(dict(test=r.test, q=q, group=grp, value=r.value, boot_se=r.boot_se, n_sta=r.n_sta,
                                 placebo_sd=np.std(pl), placebo_mean=np.mean(pl), n_placebo=len(pl),
                                 z=(r.value - np.mean(pl)) / np.std(pl),
                                 p=(np.sum(np.abs(pl - np.mean(pl)) >= abs(r.value - np.mean(pl))) + 1) / (len(pl) + 1)))
        pl = g[g.kind == 'placebo_long'].value.values
        for _, r in g[(g.kind == 'real') & (g.test == 'longterm_post2021')].iterrows():
            rows.append(dict(test=r.test, q=q, group=grp, value=r.value, boot_se=r.boot_se, n_sta=r.n_sta,
                             placebo_sd=np.std(pl), placebo_mean=np.mean(pl), n_placebo=len(pl),
                             z=(r.value - np.mean(pl)) / np.std(pl), p=np.nan))
    D = pd.DataFrame(rows)
    D.to_csv(f'{OUT}/atten_tests.csv', index=False)
    T[T.kind == 'year'].to_csv(f'{OUT}/atten_yearly.csv', index=False)
    S[S.kind.isin(['real', 'year'])].to_csv(f'{OUT}/atten_station_tests.csv', index=False)
    # per-station placebo z for the quake steps
    P = S[S.kind.str.startswith('placebo_') & (S.kind != 'placebo_long')]
    st = []
    for (q, sta), g in S[S.kind == 'real'].groupby(['q', 'sta']):
        for _, r in g.iterrows():
            W = 180 if '180d' in r.test else 60
            pl = P[(P.q == q) & (P.sta == sta) & (P.kind == f'placebo_{W}d')].v.values
            if len(pl) < 50:
                continue
            st.append(dict(q=q, sta=sta, test=r.test, value=r.v, boot_se=r.se, placebo_sd=np.std(pl),
                           z=(r.v - np.mean(pl)) / np.std(pl)))
    st = pd.DataFrame(st)
    st.to_csv(f'{OUT}/atten_station_placebo_z.csv', index=False)
    # Qp/Qs from long-term P/S slopes vs S travel time (family fixed effects)
    Q = pd.read_csv(f'{OUT}/atten_qpqs_raw.csv')
    Gm, _ = geometry()
    Q = Q.merge(Gm[['F', 'sta', 'TS', 'TP', 'X']], on=['F', 'sta'], how='left')
    Q = Q[np.isfinite(Q.dtPS_abs) & np.isfinite(Q.TS)]
    Qh = Q[Q.Scomp == 'H'].groupby(['F', 'sta'], as_index=False).agg(dtPS=('dtPS_abs', 'mean'), TS=('TS', 'first'),
                                                                      TP=('TP', 'first'))
    y = Qh.dtPS - Qh.groupby('F').dtPS.transform('mean')
    x = Qh.TS - Qh.groupby('F').TS.transform('mean')
    b = np.sum(x * y) / np.sum(x * x)
    res = y - b * x
    se_b = np.sqrt(np.sum(res ** 2) / (len(x) - Qh.F.nunique() - 1) / np.sum(x * x))
    # cluster (station) bootstrap
    rng = np.random.default_rng(3)
    stas = Qh.sta.unique()
    bb = []
    for _ in range(500):
        pick = rng.choice(stas, len(stas))
        Qb = pd.concat([Qh[Qh.sta == s_] for s_ in pick])
        yb = Qb.dtPS - Qb.groupby('F').dtPS.transform('mean')
        xb = Qb.TS - Qb.groupby('F').TS.transform('mean')
        bb.append(np.sum(xb * yb) / np.sum(xb * xb))
    Qh.to_csv(f'{OUT}/atten_qpqs.csv', index=False)
    out = dict(slope_dtPS_per_sTS=b, se_formal=se_b, se_station_boot=float(np.std(bb)), n=len(Qh),
               note='dtPS = t*_P - t*_S (from ln(P/S) slope); slope vs T_S = 1/(1.78 Qp) - 1/Qs (path-average, '
                    'site kappa differences add scatter)')
    json.dump(out, open(f'{OUT}/atten_qpqs.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    print(D[D.group == 'all'].round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['measure', 'analyse'])
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--fams', type=int, nargs='*')
    a = ap.parse_args()
    if a.stage == 'analyse':
        analyse()
        return
    jobs = [(F, 0, 100 + F) for F in (a.fams or range(1, 28))]
    with Pool(a.procs) as p:
        res = p.map(tests_for_family, jobs)
    R = pd.concat([r[0] for r in res if len(r[0])])
    Q = pd.concat([r[1] for r in res if len(r[1])])
    R.to_csv(f'{OUT}/atten_station_steps.csv', index=False)
    Q.to_csv(f'{OUT}/atten_qpqs_raw.csv', index=False)
    print('done', len(R), flush=True)


if __name__ == '__main__':
    main()
