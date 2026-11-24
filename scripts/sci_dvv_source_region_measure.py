#!/usr/bin/env python3
"""Source-region dv/v -- MEASUREMENT stage: epoch stacks of 300-500 isolated LFEs, P and S delays per channel.

For every family F and channel with a template SNR >= 10 and a P and/or S pick (data/locations/fam{F}_3d_picks.csv):
  events : isolated LFEs (no other v2/merged detection within +-5 s).
  stacks : consecutive stacks of ~400 isolated events (sizes 267-533), never straddling Petrolia / Ferndale / M7 and
           breaking again at quake + 60 d.  The [quake, quake + 60 d) window (M7: silence + restart) is ONE stack per
           family (all its events, typically 5-230) unless it holds > 600 events.  Flag 'post60' marks them.
  ref    : mean of all isolated events of the family.  2-8 Hz, traces std-normalised, 4x upsampled (200 Hz).
  delays : CC lag of stack vs reference, parabolic sub-sample, positive = stack LATE
           dtS : window [S-0.4, S+1.6] s (lag +-0.12 s)  [all components of stations with an S pick]
           dtP : window [P-0.3, P+1.2] s (lag +-0.12 s)  [all components of stations with a P pick]
           bootstrap over members (B=30) -> dtS_se, dtP_se.
  per-event S fingerprint (composition control): every isolated event, horizontal channels with template SNR >= 25,
           50 Hz CC lag vs reference in [S-0.4, S+1.6] (lag +-0.16 s) -> fp_lag, fp_cc (event x channel).
  optional waveform-level injection (--inject JSON): stacks with t >= t_on get extra delays dS_c, dP_c
           (fractional shift of the stacked trace before the S / P measurement) -> meas_inj_{tag}/.
Outputs data/sci/dvv_source_region/meas/fam{F}.parquet, fam{F}_channels.csv, fam{F}_stacks.csv,
        fam{F}_members.npz (stack membership, event times), fam{F}_fingerprint.npz.
Usage: nice -n 10 python scripts/sci_dvv_source_region_measure.py --procs 12 [--fams 23 8 ...]
       composition-matched stacks: --stacks data/sci/dvv_source_region/comp/fam{F}_stacks_in.npz --out meas_comp
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from dvv_cwire import epoch_s  # noqa: E402

OUTBASE = f'{ROOT}/data/sci/dvv_source_region'
UP = 4
NB = 30
FMIN, FMAX = 2.0, 8.0
QUAKES = {'pet': '2021-12-20T20:10', 'fer': '2022-12-20T10:34', 'm7': '2024-12-05T18:44'}
QT = epoch_s(list(QUAKES.values()))
POST = 60.0 * 86400
NSTACK = 400
SWIN = (-0.4, 1.6)
PWIN = (-0.3, 1.2)
MAXLAG = 0.12
FP_SNR = 25.0


def subpk(y, k):
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        f = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(np.clip(f, -0.5, 0.5)), float(y1 - 0.25 * (y0 - y2) * f)
    return 0.0, float(y[k])


def lag_cc(refw, X, i0, maxlag):
    """refw (n,) demeaned ref window; X (m, N); i0 window indices -> lag (samples, float), cc (edge -> cc=-1)."""
    r = refw / (np.linalg.norm(refw) + 1e-20)
    L = np.arange(-maxlag, maxlag + 1)
    C = np.empty((X.shape[0], len(L)))
    for j, l in enumerate(L):
        seg = X[:, i0 + l]
        seg = seg - seg.mean(1, keepdims=True)
        C[:, j] = seg @ r / (np.linalg.norm(seg, axis=1) + 1e-20)
    k = np.argmax(C, axis=1)
    lag, cc = np.empty(len(k)), np.empty(len(k))
    for i in range(len(k)):
        f, c = subpk(C[i], k[i])
        lag[i], cc[i] = L[k[i]] + f, c
        if k[i] in (0, len(L) - 1):
            cc[i] = -1.0
    return lag, cc


def make_epoch_stacks(te, idx, nstack=NSTACK):
    """-> list of member arrays, list of post60 flags."""
    idx = idx[np.argsort(te[idx])]
    edges = np.sort(np.r_[QT, QT + POST])
    seg = np.searchsorted(edges, te[idx], side='right')
    out, flag = [], []
    for s in np.unique(seg):
        ii = idx[seg == s]
        post = s % 2 == 1                    # segments [q, q+60 d)
        if post and len(ii) <= 600:
            out.append(ii)
            flag.append(True)
            continue
        k = max(1, int(round(len(ii) / nstack)))
        for jj in np.array_split(ii, k):
            out.append(jj)
            flag.append(bool(post))
    return out, flag


def shift_trace(x, sh, sr):
    """delay x by sh seconds (positive = later)."""
    t = np.arange(len(x)) / sr
    return np.interp(t - sh, t, x)


def channel_job(args):
    (F, h5f, cid, S, P, stacks, stack_t, iso_idx, seed, do_fp, inj) = args
    rng = np.random.default_rng(seed)
    with h5py.File(h5f, 'r') as h:
        sr, pre = float(h.attrs['sr']), float(h.attrs['pre'])
        ev = h[f'chan/{cid}/ev'][:]
        X = h[f'chan/{cid}/data'][:].astype('float64')
    pos = {e: k for k, e in enumerate(ev)}
    sos = butter(4, [FMIN, FMAX], btype='band', fs=sr, output='sos')
    X = sosfiltfilt(sos, X, axis=1)
    X /= (np.std(X, axis=1) + 1e-20)[:, None]
    t = np.arange(X.shape[1]) / sr + pre
    rk = np.array([pos[e] for e in iso_idx if e in pos])
    if len(rk) < 100:
        return None, dict(fam=F, chan=cid), None
    ref = X[rk].mean(0)
    noise = np.sqrt(np.mean(ref[(t > -15) & (t < -5)] ** 2))
    srU = sr * UP
    tU = np.arange(X.shape[1] * UP) / srU + pre
    refU = resample_poly(ref, UP, 1)
    ml = int(MAXLAG * srU)
    havS = np.isfinite(S) and S + SWIN[0] - MAXLAG > pre + 0.5
    havP = np.isfinite(P) and P + PWIN[0] - MAXLAG > pre + 0.5
    sw = np.where((tU >= S + SWIN[0]) & (tU <= S + SWIN[1]))[0] if havS else None
    pw = np.where((tU >= P + PWIN[0]) & (tU <= P + PWIN[1]))[0] if havP else None
    snrS = float(np.sqrt(np.mean(refU[sw] ** 2)) / (noise + 1e-20)) if havS else np.nan
    snrP = float(np.sqrt(np.mean(refU[pw] ** 2)) / (noise + 1e-20)) if havP else np.nan
    rows = []
    for b, mem in enumerate(stacks):
        kk = np.array([pos[e] for e in mem if e in pos])
        if len(kk) < max(5, len(mem) // 2):
            continue
        n = len(kk)
        W = np.vstack([np.full(n, 1.0 / n)] + [np.bincount(rng.integers(0, n, n), minlength=n) / n for _ in range(NB)])
        Sk = resample_poly(W @ X[kk], UP, 1, axis=-1)
        r = dict(fam=F, chan=cid, stack=b, t=stack_t[b], n=n)
        for nm, have, wi, key in (('S', havS, sw, 'dS'), ('P', havP, pw, 'dP')):
            if not have:
                r.update({f'dt{nm}': np.nan, f'cc{nm}': np.nan, f'dt{nm}_se': np.nan})
                continue
            Sk2 = Sk
            if inj is not None and stack_t[b] >= inj['t_on'] and inj.get(key, 0.0) != 0.0:
                Sk2 = np.array([shift_trace(s_, inj[key], srU) for s_ in Sk])
            lg, cg = lag_cc(refU[wi] - refU[wi].mean(), Sk2, wi, ml)
            ok = cg[1:] > -1
            r.update({f'dt{nm}': lg[0] / srU, f'cc{nm}': cg[0],
                      f'dt{nm}_se': np.std(lg[1:][ok]) / srU if ok.sum() >= NB // 2 else np.nan})
        rows.append(r)
    if len(rows) < 3:
        return None, dict(fam=F, chan=cid), None
    meta = dict(fam=F, chan=cid, S=S, P=P, snrS=snrS, snrP=snrP, n_stacks=len(rows))
    fp = None
    if do_fp and havS:
        # per-event S lag at 50 Hz (no upsampling) vs reference
        swl = np.where((t >= S + SWIN[0]) & (t <= S + SWIN[1]))[0]
        mlf = int(round(0.16 * sr))
        if swl[0] - mlf >= 0 and swl[-1] + mlf < X.shape[1]:
            ek = np.array([pos[e] for e in iso_idx if e in pos])
            lg, cg = lag_cc(ref[swl] - ref[swl].mean(), X[ek], swl, mlf)
            fp = dict(ev=np.array([e for e in iso_idx if e in pos]), lag=(lg / sr).astype('float32'),
                      cc=cg.astype('float32'))
    return pd.DataFrame(rows), meta, fp


def family_jobs(F, a, inj_all=None):
    h5f = f'{ROOT}/data/waveforms/fam{F}.h5'
    with h5py.File(h5f, 'r') as h:
        t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    te = epoch_s(t_ev)
    cat = pd.read_csv(f'{ROOT}/data/merged/LFEcat_MTJ_merged_v1v2.csv', usecols=['time_utc', 'source'])
    cat = cat[cat.source != 'v1_only']
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
    idx = np.where(iso)[0]
    idx = idx[np.argsort(te[idx])]
    if getattr(a, 'stacks', None):
        z = np.load(a.stacks.format(F=F))
        b_ = z['bounds']
        stacks = [z['members'][b_[i]:b_[i + 1]] for i in range(len(b_) - 1)]
        post = [False] * len(stacks)
    else:
        stacks, post = make_epoch_stacks(te, idx, a.n)
    stack_t = np.array([te[m].mean() for m in stacks])
    T = h5py.File(f'{ROOT}/data/templates/fam{F}_templates.h5', 'r')
    picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    jobs = []
    for i, cid in enumerate(sorted(T)):
        sta = '.'.join(cid.split('.')[:2])
        if T[cid].attrs['snr'] < 10.0 or sta not in picks.index:
            continue
        pk = picks.loc[sta]
        S = float(pk.S_final) if np.isfinite(pk.S_final) else np.nan
        if not np.isfinite(S) and np.isfinite(pk.S) and pk.sig_S <= 0.1:
            S = float(pk.S)                  # vertical-only station with a clear S pick
        P = float(pk.P) if np.isfinite(pk.P) and pk.sig_P <= 0.25 else np.nan
        if not (np.isfinite(S) or np.isfinite(P)):
            continue
        do_fp = cid[-1] in 'NE12' and T[cid].attrs['snr'] >= FP_SNR and not getattr(a, 'stacks', None)
        inj = None
        if inj_all is not None:
            inj = dict(t_on=inj_all['t_on'], dS=inj_all['dS'].get(f'{F}|{cid}', 0.0),
                       dP=inj_all['dP'].get(f'{F}|{cid}', 0.0))
            do_fp = False
        jobs.append((F, h5f, cid, S, P, stacks, stack_t, idx, 1000 * F + i, do_fp, inj))
    T.close()
    stk = pd.DataFrame(dict(stack=range(len(stacks)), t=stack_t, n=[len(m) for m in stacks], post60=post,
                            t0=[te[m].min() for m in stacks], t1=[te[m].max() for m in stacks]))
    mem = dict(te=te, iso_idx=idx, members=np.concatenate(stacks), bounds=np.cumsum([0] + [len(m) for m in stacks]))
    return jobs, stk, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fams', type=int, nargs='*', default=list(range(1, 28)))
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--n', type=int, default=NSTACK)
    ap.add_argument('--inject', default=None, help='JSON file {tag, t_on, dS: {F|chan: s}, dP: {...}}')
    ap.add_argument('--stacks', default=None, help='custom stack membership npz pattern with {F} (members, bounds)')
    ap.add_argument('--out', default=None, help='output subdirectory name (with --stacks)')
    a = ap.parse_args()
    inj_all = None
    out = f'{OUTBASE}/meas'
    if a.inject:
        inj_all = json.load(open(a.inject))
        out = f'{OUTBASE}/meas_inj_{inj_all["tag"]}'
    if a.out:
        out = f'{OUTBASE}/{a.out}'
    os.makedirs(out, exist_ok=True)
    alljobs, stks, mems = [], {}, {}
    for F in a.fams:
        if a.stacks and not os.path.exists(a.stacks.format(F=F)):
            continue
        if os.path.exists(f'{out}/fam{F}.parquet'):
            print(f'fam {F}: done, skip', flush=True)
            continue
        j, s, m = family_jobs(F, a, inj_all)
        print(f'fam {F}: {len(j)} channels, {len(s)} stacks (post60 n: {s[s.post60].n.tolist()})', flush=True)
        alljobs += j
        stks[F], mems[F] = s, m
    # big families first (load balance)
    alljobs.sort(key=lambda j: -len(j[7]))
    res = {F: [] for F in stks}
    metas = {F: [] for F in stks}
    fps = {F: {} for F in stks}
    left = {F: sum(1 for j in alljobs if j[0] == F) for F in stks}
    t0 = time.time()

    def flush(F):
        if res[F]:
            pd.concat(res[F], ignore_index=True).to_parquet(f'{out}/fam{F}.parquet')
            pd.DataFrame(metas[F]).to_csv(f'{out}/fam{F}_channels.csv', index=False)
            stks[F].to_csv(f'{out}/fam{F}_stacks.csv', index=False)
            np.savez_compressed(f'{out}/fam{F}_members.npz', **mems[F])
            if fps[F]:
                chans = sorted(fps[F])
                evs = mems[F]['iso_idx']
                L = np.full((len(evs), len(chans)), np.nan, np.float32)
                C = np.full((len(evs), len(chans)), np.nan, np.float32)
                p_ = {e: k for k, e in enumerate(evs)}
                for j, ch in enumerate(chans):
                    f = fps[F][ch]
                    k = np.array([p_[e] for e in f['ev']])
                    L[k, j], C[k, j] = f['lag'], f['cc']
                np.savez_compressed(f'{out}/fam{F}_fingerprint.npz', ev=evs, chans=np.array(chans), lag=L, cc=C)
        print(f'== fam {F} written ({time.time() - t0:.0f} s)', flush=True)
        res[F], metas[F], fps[F] = [], [], {}

    with Pool(a.procs) as p:
        for k, r in enumerate(p.imap_unordered(channel_job, alljobs, chunksize=1)):
            F = r[1]['fam']
            if r[0] is not None:
                res[F].append(r[0])
                metas[F].append(r[1])
                if r[2] is not None:
                    fps[F][r[1]['chan']] = r[2]
            left[F] -= 1
            if left[F] == 0:
                flush(F)
                del left[F]
            if k % 50 == 0:
                print(f'{k}/{len(alljobs)} jobs, {time.time() - t0:.0f} s', flush=True)
    for F in list(left):
        flush(F)


if __name__ == '__main__':
    main()
