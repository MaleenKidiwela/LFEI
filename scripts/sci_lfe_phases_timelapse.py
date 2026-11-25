#!/usr/bin/env python3
"""LFE secondary phases -- TIME-LAPSE machinery (task 2): phase-minus-direct delay through time from the monthly
isolated-LFE stacks (OUT/stacks/fam{F}.h5, sci_lfe_phases_extract.py).

Because the search (sci_lfe_phases.py) found no robust source-side phase, this is run in three modes:
  real    : the predicted SmS window (Bloch slab Moho, dz = 0) on the real stacks.  With no phase there this measures
            the S coda at lapse tau (a receiver/path-dominated quantity); reported only for completeness.
  inject  : a synthetic SmS (copy of each monthly stack's own direct-S window, amplitude ratio --amp) is added at the
            predicted delay tau, and after the M7 restart its delay is lengthened by --eps (fractional; an in-layer
            velocity DROP of eps).  Recovery of eps and the placebo scatter give the in-layer dv/v detection limit that
            a real phase of that amplitude would have.
  inject0 : the same synthetic phase with no change (placebo / false-alarm calibration of the injected data).
Family-stations: horizontal channels (template SNR >= 15) with predicted SmS delay tau >= 0.8 s (outside the
direct-S pulse).  Per window: count-weighted mean of the monthly stacks; 2-8 Hz; 200 Hz; CC lag vs the all-time
stack in the direct window [S-0.4, S+1.6] s and the phase window [S+tau-0.3, S+tau+0.7] s (+-0.1 s);
delta = lag_phase - lag_direct; eps = delta / tau (in-layer fractional delay change = -dv/v for a change confined
to the layer between the source and the reflector).  Network value = mean over family-stations of the channel means.
Tests: yearly 2008-2026 vs all-time; steps at Petrolia / Ferndale (180 d before vs after); M7 (180 d before vs
restart .. restart+180 d); restart (M7 .. M7+60 d vs 180 d before); 220+ placebo dates (1st and 15th of each month 2009-2024,
> 210 d from the quakes, same +-180 d design).
Outputs OUT/timelapse_{mode}.csv, OUT/timelapse_{mode}_summary.json
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
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_lfe_phases_common import OUT, QUAKES, RESTART, HGRID, epoch_s, geometry  # noqa: E402

SR, UP = 50.0, 4
DAY = 86400.0
QT = dict(zip(QUAKES, epoch_s(list(QUAKES.values()))))
QR = float(epoch_s([RESTART])[0])
TAU_MIN = 0.8
MAXLAG = 0.1


def subpk(y, k):
    if 0 < k < len(y) - 1:
        den = y[k - 1] - 2 * y[k] + y[k + 1]
        return float(np.clip(0.5 * (y[k - 1] - y[k + 1]) / den, -0.5, 0.5)) if den != 0 else 0.0
    return 0.0


def lag(ref, x, i0, i1, ml):
    r = ref[i0:i1] - ref[i0:i1].mean()
    r /= np.linalg.norm(r) + 1e-20
    L = np.arange(-ml, ml + 1)
    c = np.empty(len(L))
    for j, l in enumerate(L):
        s = x[i0 + l:i1 + l]
        s = s - s.mean()
        c[j] = s @ r / (np.linalg.norm(s) + 1e-20)
    k = int(np.argmax(c))
    if k in (0, len(L) - 1):
        return np.nan, c[k]
    return (L[k] + subpk(c, k)), c[k]


def windows():
    W = []
    for y in range(2008, 2027):
        W.append((f'year_{y}', 'year', epoch_s([f'{y}-01-01'])[0], epoch_s([f'{y + 1}-01-01'])[0], None, None))
    d = 180 * DAY
    for q in ('petrolia', 'ferndale'):
        W.append((f'step_{q}', 'real', QT[q], QT[q] + d, QT[q] - d, QT[q]))
    W.append(('step_m7', 'real', QR, QR + d, QT['m7'] - d, QT['m7']))
    W.append(('restart_0_60d', 'real', QT['m7'], QT['m7'] + 60 * DAY, QT['m7'] - d, QT['m7']))
    ms = epoch_s(pd.date_range('2009-01-01', '2024-06-01', freq='SMS'))
    ms = ms[np.all(np.abs(ms[:, None] - np.array(list(QT.values()))[None, :]) > 210 * DAY, 1)]
    for t0 in ms:
        W.append((f'placebo_{pd.to_datetime(t0, unit="s").strftime("%Y-%m-%d")}', 'placebo', t0, t0 + d, t0 - d, t0))
    return W


def family_job(args):
    F, mode, amp, eps = args
    G, tabs = geometry()
    g = G[G.F == F].set_index('sta')
    fn = f'{OUT}/stacks/fam{F}.h5'
    with h5py.File(fn, 'r') as h:
        chan = [c.decode() for c in h['chan'][:]]
        snr, Sp = h['snr'][:], h['S'][:]
        ME = h['medges'][:]
        MC = h['mcount'][:]
        t0 = h.attrs['t0']
        keep = [k for k, c in enumerate(chan) if c[-1] != 'Z' and snr[k] >= 15 and np.isfinite(Sp[k])
                and '.'.join(c.split('.')[:2]) in g.index]
        if not keep:
            return []
        MS = h['mstack'][:, keep, :].astype(np.float32)
    MC = MC[:, keep]
    chan = [chan[k] for k in keep]
    Sp = Sp[keep]
    tm = 0.5 * (ME[:-1] + ME[1:])
    nt = MS.shape[2]
    t = t0 + np.arange(nt) / SR
    sos = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
    rows = []
    W = windows()
    for k, c in enumerate(chan):
        sta = '.'.join(c.split('.')[:2])
        gi = int(np.where((G.F == F) & (G.sta == sta))[0][0])
        tau = float(np.interp(G.iloc[gi].H_m, HGRID, tabs['SmS'][gi]))
        if not np.isfinite(tau) or tau < TAU_MIN or Sp[k] + tau + 1.0 > t[-1]:
            continue
        X = MS[:, k, :].astype(np.float64)
        n = MC[:, k].astype(float)
        if mode.startswith('inject'):
            w = np.clip(np.minimum((t - (Sp[k] - 0.5)) / 0.2, ((Sp[k] + 1.5) - t) / 0.2), 0, 1)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            for m in range(len(X)):
                if n[m] == 0:
                    continue
                tt = tau * (1 + eps) if (mode == 'inject' and tm[m] >= QR) else tau
                X[m] = X[m] + amp * np.interp(t - tt, t, X[m] * w, left=0.0, right=0.0)
        X = sosfiltfilt(sos, X, axis=1)
        ref = resample_poly((X * n[:, None]).sum(0) / n.sum(), UP, 1)
        tU = t0 + np.arange(len(ref)) / (SR * UP)
        ml = int(MAXLAG * SR * UP)
        # direct window ends before the phase window starts (no overlap with a phase at tau)
        d0, d1 = np.searchsorted(tU, Sp[k] - 0.4), np.searchsorted(tU, Sp[k] + min(1.6, tau - 0.4))
        p0, p1 = np.searchsorted(tU, Sp[k] + tau - 0.3), np.searchsorted(tU, Sp[k] + tau + 0.7)
        if d0 - ml < 0 or p1 + ml >= len(ref):
            continue

        def wstack(a, b):
            m = (tm >= a) & (tm < b) & (n > 0)
            if n[m].sum() < 30:
                return None, 0
            return resample_poly((X[m] * n[m, None]).sum(0) / n[m].sum(), UP, 1), int(n[m].sum())
        cache = {}
        for name, kind, a0, a1, b0, b1 in W:
            A, nA = wstack(a0, a1)
            if A is None:
                continue
            if b0 is None:
                B, nB = ref, int(n.sum())
            else:
                B, nB = wstack(b0, b1)
                if B is None:
                    continue
            res = []
            for Y in (A, B):
                key = id(Y)
                ld, cd = lag(ref, Y, d0, d1, ml)
                lp, cp = lag(ref, Y, p0, p1, ml)
                res.append(((lp - ld) / (SR * UP), cd, cp))
            delta = res[0][0] - res[1][0]
            rows.append(dict(F=F, chan=c, sta=sta, tau=tau, test=name, kind=kind, delta=delta, eps=delta / tau,
                             ccA_dir=res[0][1], ccA_ph=res[0][2], nA=nA, nB=nB))
    return rows


def summarise(D, mode):
    D = D[np.isfinite(D.eps) & (D.ccA_dir > 0.5)]
    S = D.groupby(['test', 'kind', 'F', 'sta'], as_index=False).agg(eps=('eps', 'mean'), delta=('delta', 'mean'),
                                                                    tau=('tau', 'first'))
    S['dt'] = S.delta * S.tau
    S['tt'] = S.tau ** 2
    S['bh'] = S.sta.str.startswith('PB.')
    out = {}
    rows = []
    for grp, s in (('all', S), ('PB', S[S.bh]), ('surface', S[~S.bh])):
        # least-squares eps for delta = eps * tau over family-stations (long-tau stations dominate)
        N = s.groupby(['test', 'kind']).agg(dt=('dt', 'sum'), tt=('tt', 'sum'), eps_mean=('eps', 'mean'),
                                            delta_ms=('delta', lambda x: 1e3 * x.mean()),
                                            n_fs=('eps', 'size')).reset_index()
        N['eps'] = N.dt / N.tt
        N['group'] = grp
        pl = N[N.kind == 'placebo']
        mu, sd = pl.eps.mean(), pl.eps.std()
        N['z'] = (N.eps - mu) / sd
        N['placebo_sd'] = sd
        rows.append(N)
        out[grp] = dict(placebo_sd_eps=float(sd), placebo_mean_eps=float(mu), n_placebo=int(len(pl)),
                        dvv_limit_2sd_pct=float(200 * sd),
                        tests={r.test: dict(eps=float(r.eps), z=float(r.z), n_fs=int(r.n_fs))
                               for r in N[N.kind == 'real'].itertuples()})
    R = pd.concat(rows)
    R.to_csv(f'{OUT}/timelapse_{mode}.csv', index=False)
    json.dump(out, open(f'{OUT}/timelapse_{mode}_summary.json', 'w'), indent=1)
    print(mode, json.dumps(out['all'], indent=1)[:1500])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['real', 'inject', 'inject0'])
    ap.add_argument('--amp', type=float, default=0.3)
    ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    G, _ = geometry()
    with Pool(a.procs) as p:
        res = p.map(family_job, [(F, a.mode, a.amp, a.eps) for F in range(1, 28)])
    D = pd.DataFrame([r for rr in res for r in rr])
    mode = a.mode + a.tag
    D.to_csv(f'{OUT}/timelapse_{mode}_channels.csv.gz', index=False)
    summarise(D, mode)


if __name__ == '__main__':
    main()
