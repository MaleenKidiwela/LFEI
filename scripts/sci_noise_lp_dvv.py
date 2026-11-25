#!/usr/bin/env python3
"""Long-period inter-station noise dv/v: stretching on the coda of daily ZRT cross-correlations
(scripts/sci_noise_lp.py -> /data/wsd02/maleen_data/MTJ_LH/pairs/*.npz).

Stage 'pairs' (<= 4 processes; per pair x band x component):
  day QC: nseg >= 16 of 47 (Z or H), finite, broadband RMS <= 4x the pair median.
  bands 0.05-0.1, 0.1-0.2, 0.2-0.3 Hz (zero-phase cosine-tapered FFT filter of every daily CCF).
  coda window |lag| in [t1, t2]: t1 = max(2 d / 2.5 km/s, 1.5/fmin), t2 = min(t1 + L_band, 290 s),
  L = 120 / 90 / 60 s; both sides jointly ('both') and separately ('caus' = positive lag a->b, 'acaus').
  stretching: reference upsampled 8x (band-limited FFT interpolation), grid dv/v -1.5..+1.5 % in 0.01 %,
  parabolic refinement; convention u(t) ~ ref(t (1 + dv/v)) (velocity increase -> earlier arrivals).
  references: 'full' = mean of all good days; 'pre' = mean of good days before 2021-12-01 (if >= 300 days).
  outputs per pair (/data/wsd02/maleen_data/MTJ_LH/dvv/<A>_<B>.npz):
    bin series: 10-d and 30-d non-overlapping bins from 2008-01-01 (dvv, cc, ndays, Weaver formal error);
    steps at every 3rd day (placebo grid) and at the 5 event dates: dvv(post) - dvv(pre) with
      pre = [E-W, E-1], post = [E+1, E+W] days, W = 15, 30, 60 (event day excluded), each vs 'full' ref;
    injections: post window stretched by dv/v = -0.02, -0.05, -0.1, -0.2, -0.5 % (band-limited) at every grid date;
    moving-block bootstrap (5-day blocks, 300 replicates, the same day design for every pair) at the events.
Stage 'network' combines pairs (see sci_noise_lp_net.py).
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.path.join(ROOT, 'data/sci/noise_lp')
BASE = '/data/wsd02/maleen_data/MTJ_LH'
PAIRD = os.path.join(BASE, 'pairs')
DVVD = os.path.join(BASE, 'dvv')
DAY0 = pd.Timestamp('2008-01-01')
NDAYS = (pd.Timestamp('2026-11-23') - DAY0).days + 1
COMPS = ['ZZ', 'RR', 'TT', 'ZR', 'RZ']
BANDS = {'0.05-0.1': (0.05, 0.1, 120.0), '0.1-0.2': (0.1, 0.2, 90.0), '0.2-0.3': (0.2, 0.3, 60.0)}
EVENTS = {'M6.5_2010': '2010-01-10', 'M6.8_2014': '2014-03-10', 'Petrolia_2021': '2021-12-20',
          'Ferndale_2022': '2022-12-20', 'M7_2024': '2024-12-05'}
WS = [15, 30, 60]
INJ = np.array([-0.02, -0.05, -0.10, -0.20, -0.50]) / 100
EPS = np.arange(-0.015, 0.015 + 1e-9, 0.0001)
LAG = np.arange(-300, 301).astype(float)
UP = 8
NBOOT = 300
BLOCK = 5
U_GROUP = 2.5
PRE_END = pd.Timestamp('2021-12-01')
GRID = np.arange(0, NDAYS, 3)
MINDAYS = 60


def ev_idx():
    return {k: (pd.Timestamp(v) - DAY0).days for k, v in EVENTS.items()}


def bandfilter(x, f1, f2, n=1024):
    """zero-phase cosine-tapered band-pass along the last axis (lag axis, 1 s)."""
    f = np.fft.rfftfreq(n, 1.0)
    t = np.zeros_like(f)
    w1, w2 = 0.3 * f1, 0.15 * f2
    t[(f >= f1) & (f <= f2)] = 1
    m = (f > f1 - w1) & (f < f1)
    t[m] = 0.5 * (1 - np.cos(np.pi * (f[m] - (f1 - w1)) / w1))
    m = (f > f2) & (f < f2 + w2)
    t[m] = 0.5 * (1 + np.cos(np.pi * (f[m] - f2) / w2))
    X = np.fft.rfft(x, n, axis=-1) * t
    return np.fft.irfft(X, n, axis=-1)[..., :x.shape[-1]]


def upsample(x, up=UP, nf=1024):
    """band-limited upsampling of traces on LAG (last axis) -> (lag_up, x_up)."""
    X = np.fft.rfft(x, nf, axis=-1)
    y = np.fft.irfft(X, nf * up, axis=-1) * up
    lu = LAG[0] + np.arange(nf * up) / up
    k = (lu <= LAG[-1] + 1e-9)
    return lu[k], y[..., k]


def windows(dist, band):
    f1, f2, L = BANDS[band]
    t1 = max(2 * dist / U_GROUP, 1.5 / f1)
    t2 = min(t1 + L, 290.0)
    if t2 - t1 < 40:
        return None
    return t1, t2


def stretch_matrix(ref, wmask):
    """ref (601,) -> S (neps, nw): ref evaluated at t(1+eps) for the lags in wmask."""
    lu, ru = upsample(ref[None, :])
    t = LAG[wmask]
    tt = t[None, :] * (1 + EPS[:, None])
    return np.interp(tt.ravel(), lu, ru[0]).reshape(tt.shape)


def measure(S, Snorm, U):
    """U (n, nw) -> dvv (n,), cc (n,) by grid + parabola."""
    Un = np.linalg.norm(U, axis=1)
    C = (U @ S.T) / (Un[:, None] * Snorm[None, :] + 1e-30)
    j = np.argmax(C, axis=1)
    j = np.clip(j, 1, len(EPS) - 2)
    r = np.arange(len(j))
    y0, y1, y2 = C[r, j - 1], C[r, j], C[r, j + 1]
    den = y0 - 2 * y1 + y2
    dx = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    dx = np.clip(dx, -1, 1)
    dvv = EPS[j] + dx * (EPS[1] - EPS[0])
    cc = y1 - 0.25 * (y0 - y2) * dx
    bad = ~np.isfinite(Un) | (Un == 0)
    dvv[bad] = np.nan
    cc[bad] = np.nan
    return dvv, cc


def weaver_err(cc, band, t1, t2):
    f1, f2, _ = BANDS[band]
    wc = 2 * np.pi * 0.5 * (f1 + f2)
    T = 1.0 / (f2 - f1)
    X = np.clip(cc, 1e-3, 0.999)
    return np.sqrt(1 - X ** 2) / (2 * X) * np.sqrt(6 * np.sqrt(np.pi / 2) * T / (wc ** 2 * (t2 ** 3 - t1 ** 3)))


def boot_design(W, seed):
    """offsets (NBOOT, W) for pre (negative) and post (positive) windows, 5-day moving blocks."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(W / BLOCK))
    pre, post = [], []
    for _ in range(NBOOT):
        for side, lst in ((-1, pre), (1, post)):
            st = rng.integers(0, W - BLOCK + 1, nb)
            o = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:W] + 1      # 1..W
            lst.append(side * o)
    return np.array(pre), np.array(post)


def process_pair(fn):
    name = os.path.basename(fn)[:-4]
    fo = os.path.join(DVVD, name + '.npz')
    if os.path.exists(fo):
        return name, 'exists'
    a, b = name.split('_')
    P = pd.read_csv(os.path.join(OUT, 'pairs.csv'))
    pr = P[(P.a == a) & (P.b == b)].iloc[0]
    Z = np.load(fn)
    days, ccf, nz, nh = Z['days'], Z['ccf'], Z['nseg_z'], Z['nseg_h']
    res = dict(days=days)
    evs = ev_idx()
    out = {}
    for ci, comp in enumerate(COMPS):
        ns = nz if comp == 'ZZ' else nh
        x = ccf[:, ci, :].astype(float)
        good = (ns >= 16) & np.isfinite(x).all(axis=1)
        if good.sum() < MINDAYS:
            continue
        x = np.where(good[:, None], x, 0.0)
        rms = np.sqrt((x ** 2).mean(axis=1))
        good &= rms <= 4 * np.median(rms[good])
        if good.sum() < MINDAYS:
            continue
        # dense daily array
        D = np.zeros((NDAYS, 601))
        cnt = np.zeros(NDAYS)
        D[days[good]] = x[good]
        cnt[days[good]] = 1
        for band in BANDS:
            w = windows(pr.dist_km, band)
            if w is None:
                continue
            t1, t2 = w
            f1, f2, _ = BANDS[band]
            Df = bandfilter(D, f1, f2)
            wm = {'both': (np.abs(LAG) >= t1) & (np.abs(LAG) <= t2), 'caus': (LAG >= t1) & (LAG <= t2),
                  'acaus': (LAG <= -t1) & (LAG >= -t2)}
            ref_full = Df[cnt > 0].mean(axis=0)
            pre_m = (cnt > 0) & (np.arange(NDAYS) < (PRE_END - DAY0).days)
            refs = {'full': ref_full}
            if pre_m.sum() >= 300:
                refs['pre'] = Df[pre_m].mean(axis=0)
            key = f'{comp}|{band}'
            rec = dict(t1=t1, t2=t2, ndays=int(cnt.sum()), ref_full=ref_full.astype(np.float32),
                       ref_bb=D[cnt > 0].mean(axis=0).astype(np.float32))
            cs = np.r_[np.zeros((1, 601)), np.cumsum(Df, axis=0)]
            cn = np.r_[0, np.cumsum(cnt)]
            # --- bin series
            for nb_ in (10, 30):
                e = np.arange(0, NDAYS + 1, nb_)
                e = np.r_[e, NDAYS] if e[-1] != NDAYS else e
                st = cs[e[1:]] - cs[e[:-1]]
                nd = cn[e[1:]] - cn[e[:-1]]
                keep = nd >= max(3, nb_ // 3)
                rec[f'bin{nb_}_nd'] = nd
                for rn, ref in refs.items():
                    for side, m in wm.items():
                        S = stretch_matrix(ref, m)
                        Sn = np.linalg.norm(S, axis=1)
                        dv, cc = measure(S, Sn, st[:, m])
                        dv[~keep] = np.nan
                        cc[~keep] = np.nan
                        rec[f'bin{nb_}_{rn}_{side}_dvv'] = dv.astype(np.float32)
                        rec[f'bin{nb_}_{rn}_{side}_cc'] = cc.astype(np.float32)
                rec[f'bin{nb_}_err'] = weaver_err(rec[f'bin{nb_}_full_both_cc'], band, t1, t2).astype(np.float32)
            # --- steps at grid dates and events (vs full ref)
            S = {side: stretch_matrix(ref_full, m) for side, m in wm.items()}
            Sn = {side: np.linalg.norm(S[side], axis=1) for side in wm}
            dates = np.r_[GRID, np.array(list(evs.values()))]
            lu, _ = upsample(np.zeros((1, 601)), nf=1024)
            for W in WS:
                ok = (dates - W >= 0) & (dates + W + 1 <= NDAYS)
                dd = np.where(ok, dates, W)
                pre = cs[dd] - cs[dd - W]
                post = cs[dd + W + 1] - cs[dd + 1]
                npre = cn[dd] - cn[dd - W]
                npost = cn[dd + W + 1] - cn[dd + 1]
                valid = ok & (npre >= W / 3) & (npost >= W / 3)
                for side, m in wm.items():
                    dpre, cpre = measure(S[side], Sn[side], pre[:, m])
                    dpost, cpost = measure(S[side], Sn[side], post[:, m])
                    stp = dpost - dpre
                    stp[~valid] = np.nan
                    rec[f'step{W}_{side}'] = stp.astype(np.float32)
                    rec[f'step{W}_{side}_cc'] = np.fmin(cpre, cpost).astype(np.float32)
                    if side == 'both':
                        # injection: post stack stretched by eps_inj, i.e. post(t(1+eps)) (band-limited)
                        inj = [np.full(len(dates), np.nan) for _ in INJ]
                        for c0 in range(0, len(dates), 256):
                            _, pu = upsample(post[c0:c0 + 256], nf=1024)
                            for ii, ei in enumerate(INJ):
                                tt = LAG[m] * (1 + ei)
                                k = np.clip(np.searchsorted(lu, tt) - 1, 0, len(lu) - 2)
                                fr = (tt - lu[k]) / (lu[k + 1] - lu[k])
                                pi_ = pu[:, k] * (1 - fr) + pu[:, k + 1] * fr
                                dpi, _ = measure(S[side], Sn[side], pi_)
                                inj[ii][c0:c0 + 256] = dpi - dpre[c0:c0 + 256]
                        for ii in range(len(INJ)):
                            inj[ii][~valid] = np.nan
                        rec[f'inj{W}'] = np.array(inj, np.float32)
                rec[f'step{W}_npre'] = npre.astype(np.int16)
                rec[f'step{W}_npost'] = npost.astype(np.int16)
            # --- moving-block bootstrap at the events (both sides)
            m = wm['both']
            for W in WS:
                bpre, bpost = boot_design(W, 1000 + W)
                B = np.full((len(evs), NBOOT), np.nan, np.float32)
                for ie, E in enumerate(evs.values()):
                    if E - W < 0 or E + W + 1 > NDAYS:
                        continue
                    if (cn[E] - cn[E - W]) < W / 3 or (cn[E + W + 1] - cn[E + 1]) < W / 3:
                        continue
                    sp = Df[E + bpre][:, :, m].sum(axis=1)
                    so = Df[E + bpost][:, :, m].sum(axis=1)
                    d1, _ = measure(S['both'], Sn['both'], sp)
                    d2, _ = measure(S['both'], Sn['both'], so)
                    B[ie] = d2 - d1
                rec[f'boot{W}'] = B
            out[key] = rec
    os.makedirs(DVVD, exist_ok=True)
    flat = {}
    for key, rec in out.items():
        for k, v in rec.items():
            flat[f'{key}|{k}'] = np.asarray(v)
    flat['dates'] = np.r_[GRID, np.array(list(evs.values()))]
    np.savez_compressed(fo, **flat)
    return name, len(out)


def stage_pairs(args):
    files = sorted(glob.glob(os.path.join(PAIRD, '*.npz')))
    if args.only:
        files = [f for f in files if any(o in f for o in args.only.split(','))]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(process_pair, files)):
            print(f'{time.strftime("%H:%M:%S")} {i + 1}/{len(files)} {r} {(time.time() - t0) / (i + 1):.1f} s/pair',
                  flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only')
    a = ap.parse_args()
    globals()['stage_' + a.stage](a)
