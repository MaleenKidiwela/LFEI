#!/usr/bin/env python3
"""EQ-above step 2 (catalog events): per-event P and S delays per channel relative to the earthquake-family reference.

Input : wf40.npy (event x station x comp x 2000; 40 Hz, 2-8 Hz, t = -5 s .. +45 s from catalog origin),
        ttP.npy / ttS.npy (3D-model predictions at catalog hypocentres), families.csv (efam > 0).
Per family E and station (>= 5 members with data):
  windows centred on the family-median predicted P and S; coarse per-event alignment by CC of [P-1, S+4] s on all
  components (lag +-1.0 s) vs an iterated reference stack (2 iterations, members with CC >= 0.5 enter the stack);
  fine delays per channel with 5x upsampling (200 Hz), parabolic sub-sample:
    dtS : horizontals (and Z of single-component stations) window [S-0.5, S+2.0] s, lag +-0.3 s about the coarse lag
    dtP : Z window [P-0.3, P+1.5] s, lag +-0.3 s about the coarse lag
  delay = event arrival minus reference arrival (s; positive = late), measured on the catalog origin time axis.
  snr = RMS(window) / RMS(pre-P noise -4.5 .. P-1 s).
Output data/sci/eq_above/meas_cat.parquet: efam, ev (index in events.csv), evid, t (epoch s), chan, phase, y, cc, snr
        and data/sci/eq_above/ref_cat.npz (reference stacks for templates).
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/eq_above'
BIG = '/data/wsd02/maleen_data/eq_above'          # bulk waveforms (kept off /home)
sys.path.insert(0, f'{ROOT}/scripts')
from sci_eq_above_cluster import NS, PRE, SR, STAS  # noqa: E402

UP = 5
PASS2_TOL = 0.08          # pass 2: lag search +-80 ms about the relocation-predicted delay (removes cycle skips)
COMPS = ['Z', '1', '2']


def subpk(y, k):
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        f = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(np.clip(f, -0.5, 0.5)), float(y1 - 0.25 * (y0 - y2) * f)
    return 0.0, float(y[k])


def lagcc(ref, x, i0, i1, l0, ml):
    """CC of ref[i0:i1] against x[i0+l : i1+l] for l in l0 +- ml -> (lag samples float, cc). Edge -> cc -1."""
    r = ref[i0:i1] - ref[i0:i1].mean()
    r = r / (np.linalg.norm(r) + 1e-20)
    L = np.arange(l0 - ml, l0 + ml + 1)
    C = np.full(len(L), -1.0)
    for j, l in enumerate(L):
        if i0 + l < 0 or i1 + l > len(x):
            continue
        s = x[i0 + l:i1 + l]
        s = s - s.mean()
        C[j] = s @ r / (np.linalg.norm(s) + 1e-20)
    k = int(np.argmax(C))
    f, c = subpk(C, k)
    if k in (0, len(L) - 1):
        c = -1.0
    return L[k] + f, c


def align_stack(W, P, S, sr, pre, maxlag=1.0, iters=3):
    """W (n, 3, ns) traces on the catalog-origin axis (t = -pre ..), P/S predicted (s after origin).
    -> ref (3, ns) std-normalised aligned stack on the members' MEAN origin frame, lag (n,) s, cc (n,), good (n,)
       or None if < 5 members have data.  Coarse alignment on [P-1, S+4] s, all components (mean CC)."""
    ns = W.shape[-1]
    have = np.array([np.isfinite(W[i, :, ns // 2]).any() and np.nanstd(W[i]) > 0 for i in range(len(W))])
    if have.sum() < 5:
        return None
    comps = [c for c in range(3) if np.isfinite(W[have][:, c, ns // 2]).sum() >= 5 and np.nanstd(W[have][:, c]) > 0]
    if not comps:
        return None
    X = np.nan_to_num(np.asarray(W, float))
    for i in range(len(X)):
        for c in range(3):
            sd = X[i, c].std()
            if sd > 0:
                X[i, c] /= sd
    i0 = int((pre + P - 1.0) * sr)
    i1 = int((pre + S + 4.0) * sr)
    ml = int(maxlag * sr)
    if i0 - ml < 0 or i1 + ml > ns:
        return None
    nz = slice(int(0.5 * sr), max(i0, int(0.6 * sr) + 1))
    snr0 = np.array([np.sqrt(np.mean(X[i, comps, i0:i1] ** 2)) / (np.sqrt(np.mean(X[i, comps, nz] ** 2)) + 1e-9)
                     if have[i] else 0 for i in range(len(X))])
    ref = X[int(np.argmax(snr0))].copy()
    lag = np.zeros(len(X))
    ccn = np.full(len(X), -1.0)
    L = np.arange(-ml, ml + 1)
    for it in range(iters):
        R = []
        for c in comps:
            r = ref[c, i0:i1] - ref[c, i0:i1].mean()
            R.append(r / (np.linalg.norm(r) + 1e-20))
        for i in range(len(X)):
            if not have[i]:
                continue
            C = np.zeros(len(L))
            for ci, c in enumerate(comps):
                for j, l in enumerate(L):
                    sg = X[i, c, i0 + l:i1 + l]
                    sg = sg - sg.mean()
                    C[j] += sg @ R[ci] / (np.linalg.norm(sg) + 1e-20)
            C /= len(comps)
            j = int(np.argmax(C))
            f, cv = subpk(C, j)
            lag[i], ccn[i] = L[j] + f, cv
        good = have & (ccn >= 0.5)
        if good.sum() < 3:
            good = have & (ccn >= np.sort(ccn[have])[-3])
        mu = int(round(np.mean(lag[good])))           # keep the stack on the members' mean origin frame
        new = np.zeros_like(ref)
        for i in np.where(good)[0]:
            new += np.roll(X[i], -(int(round(lag[i])) - mu), axis=-1)
        ref = new / good.sum()
    lag = lag - np.mean(lag[good])
    return ref, lag / sr, ccn, good


def fam_station(args):
    E, k, idx, P, S, W = args[:6]
    pred = args[6] if len(args) > 6 else None
    r = align_stack(W, P, S, SR, PRE)
    if r is None:
        return [], None
    ref, lag_s, ccn, good = r
    have = ccn > -1
    comps = [c for c in range(3) if np.any(ref[c])]
    X = np.nan_to_num(np.asarray(W, float))
    for i in range(len(X)):
        for c in range(3):
            sd = X[i, c].std()
            if sd > 0:
                X[i, c] /= sd
    lag = lag_s * SR
    # fine measurement at UP x
    refU = resample_poly(ref, UP, 1, axis=-1)
    srU = SR * UP
    rows = []
    sw = (int((PRE + S - 0.5) * srU), int((PRE + S + 2.0) * srU))
    pw = (int((PRE + P - 0.3) * srU), int((PRE + P + 1.5) * srU))
    nzU = (int(0.5 * srU), int((PRE + P - 1.0) * srU))
    mlU = int(0.3 * srU)
    single = comps == [0]
    for i in range(len(idx)):
        if not have[i]:
            continue
        xU = resample_poly(X[i], UP, 1, axis=-1)
        l0 = int(round(lag[i] * UP))
        for c in comps:
            nrm = np.sqrt(np.mean(xU[c, nzU[0]:nzU[1]] ** 2)) + 1e-12
            for ph, (a, b) in (('S', sw), ('P', pw)):
                if ph == 'S' and c == 0 and not single:
                    continue
                if ph == 'P' and c != 0:
                    continue
                key = (int(idx[i]), f'{STAS[k]}.{COMPS[c]}', ph)
                if pred is not None and key in pred:
                    lg, cc = lagcc(refU[c], xU[c], a, b, int(round(pred[key] * srU)), int(PASS2_TOL * srU))
                elif pred is not None:
                    lg, cc = lagcc(refU[c], xU[c], a, b, l0, mlU)
                    cc = min(cc, 0.0)                     # no relocation prediction: flag
                else:
                    lg, cc = lagcc(refU[c], xU[c], a, b, l0, mlU)
                snr = np.sqrt(np.mean(xU[c, a + l0:b + l0] ** 2)) / nrm if 0 <= a + l0 and b + l0 <= xU.shape[1] else np.nan
                rows.append(dict(efam=E, ev=int(idx[i]), chan=f'{STAS[k]}.{COMPS[c]}', sta=STAS[k], phase=ph,
                                 y=lg / srU, cc=cc, snr=snr, ccnet=ccn[i]))
    return rows, (E, k, ref.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--pass2', action='store_true', help='re-measure about relocation predictions (reloc_resid)')
    a = ap.parse_args()
    pred_all = None
    if a.pass2:
        R = pd.read_parquet(f'{OUT}/reloc_resid.parquet')
        # predicted delay for every channel of relocated events: tau + g.dx + offset (= y - res); all channels of a
        # relocated event get a prediction via the station geometry
        G = pd.read_csv(f'{OUT}/geom_eq.csv')
        ev = pd.read_csv(f'{OUT}/reloc_events.csv')
        off = R.groupby(['efam', 'chan', 'phase']).off.median()
        pred_all = {}
        for E, g in ev.groupby('efam'):
            gg = G[G.efam == E].set_index('sta')
            offE = off.loc[E] if E in off.index.get_level_values(0) else None
            for r in g.itertuples():
                for sta in gg.index:
                    for comp in COMPS:
                        ch = f'{sta}.{comp}'
                        for ph in ('P', 'S'):
                            gv = gg.loc[sta, [f'g{ph}x', f'g{ph}y', f'g{ph}z']].values.astype(float)
                            o = 0.0
                            if offE is not None and (ch, ph) in offE.index:
                                o = float(offE.loc[(ch, ph)])
                            pred_all.setdefault(E, {})[(int(r.ev), ch, ph)] = r.tau + gv @ np.array([r.dx, r.dy, r.dz]) + o
    c = pd.read_csv(f'{OUT}/families.csv')
    W = np.load(f'{BIG}/wf40.npy', mmap_mode='r')
    TP = np.load(f'{OUT}/ttP.npy')
    TS = np.load(f'{OUT}/ttS.npy')
    jobs = []
    for E in sorted(c.efam[c.efam > 0].unique()):
        idx = np.where(c.efam.values == E)[0]
        for k in range(len(STAS)):
            P = np.nanmedian(TP[idx, k])
            S = np.nanmedian(TS[idx, k])
            if not (np.isfinite(P) and np.isfinite(S)):
                continue
            jobs.append((E, k, idx, P, S, np.asarray(W[idx, k])) +
                        ((pred_all.get(E, {}),) if pred_all is not None else ()))
    print(len(jobs), 'family-station jobs', flush=True)
    with Pool(a.procs) as p:
        res = p.map(fam_station, jobs, chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    d = pd.DataFrame(rows)
    d['evid'] = c.EventID.values[d.ev.values]
    d['t'] = (pd.to_datetime(c.Time.values[d.ev.values]) - pd.Timestamp('1970-01-01')).total_seconds().values
    if a.pass2:
        os.replace(f'{OUT}/meas_cat.parquet', f'{OUT}/meas_cat_pass1.parquet') if not os.path.exists(
            f'{OUT}/meas_cat_pass1.parquet') else None
    d.to_parquet(f'{OUT}/meas_cat.parquet')
    refs = {f'E{E}_{STAS[k]}': r for _, x in res if x is not None for (E, k, r) in [x]}
    np.savez_compressed(f'{OUT}/ref_cat.npz', **refs)
    g = d[d.cc > 0.6].groupby(['efam', 'phase']).agg(n=('y', 'size'), nev=('ev', 'nunique'), cc=('cc', 'median'))
    print(g.to_string())


if __name__ == '__main__':
    main()
