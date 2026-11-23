#!/usr/bin/env python3
"""Is the early-coda 60-d M7 transient a velocity change near the LFE sources, or an artefact of a
change in WHERE the LFEs occur inside their ~560 x 350 m patches after the M7 restart?

Families 23, 8, 27. Fault frame (Shelly): strike 290, dip 45 NNE; s along strike (+WNW),
d down-dip (+NNE/deeper), n normal.  Events with another family's LFE within +-10 s are excluded
from all location statistics and location-selected stacks (overlap pull ~50 m).

Stages
  --prep     event tables, task 1 (location shift at the restart), split / stack definitions for
             the waveform helper (data/sci/transient_source/fam{F}_jobdef.pkl)
  (run scripts/sci_transient_source_measure.py --family F)
  --analyze  task 2 (apparent dv/v per metre of TRUE source offset vs lapse time, from time-matched
             location-selected pre-M7 stacks), task 3 (location-matched transient, re-measured),
             task 4 (predicted transient from 2 x 1 through the same parametric group fit), figures.
Outputs: data/sci/transient_source/*.csv, summary.json; figures/sci_transient_source_*.png
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_transient_source_common import (AX, DAY, FAMS, M7, OUT, family_table, make_stacks,  # noqa: E402
                                         periods)

BLOCK = 40           # time block (consecutive pool events) for time-matched location splits
NNULL = 20           # random-split null realisations per pool
NBOOT = 500
rng = np.random.default_rng(2026)
os.makedirs(OUT, exist_ok=True)


# ============================================================ task 1: location shift
def centroid_diff(x1, x2, nb=NBOOT):
    """Mean(x1) - mean(x2) per column with event-bootstrap SE."""
    d = np.nanmean(x1, 0) - np.nanmean(x2, 0)
    bs = [np.nanmean(x1[rng.integers(0, len(x1), len(x1))], 0) - np.nanmean(x2[rng.integers(0, len(x2), len(x2))], 0)
          for _ in range(nb)]
    return d, np.std(bs, 0)


def cross_cov(xa, xb):
    a, b = xa - xa.mean(0), xb - xb.mean(0)
    C = (a.T @ b) / len(a)
    return 0.5 * (C + C.T)


def task1(F, T):
    P = periods(T)
    sel = T.iso.values & ~T.ovl10.values
    ref = P['pre1y'] | P['post_late']
    rows = []
    comps = [('post_early60', 'pre1y'), ('post_early60', 'post_late'), ('post_early60', 'ref'),
             ('post_60_182', 'ref'), ('post_late', 'pre1y')]
    for evsel_name, evsel in [('iso_nonovl', sel), ('all_nonovl', ~T.ovl10.values)]:
        for p1, p2 in comps:
            m1 = P[p1] & evsel
            m2 = (ref if p2 == 'ref' else P[p2]) & evsel
            for src, cols in [('full', AX), ('A', [c + 'A' for c in AX]), ('B', [c + 'B' for c in AX])]:
                x1 = T.loc[m1, cols].dropna().values
                x2 = T.loc[m2, cols].dropna().values
                if len(x1) < 10 or len(x2) < 10:
                    continue
                d, se = centroid_diff(x1, x2)
                r = dict(fam=F, events=evsel_name, comp=f'{p1}-{p2}', loc=src, n1=len(x1), n2=len(x2))
                for k, c in enumerate(AX):
                    r[f'd{c}'], r[f'd{c}_se'] = d[k], se[k]
                rows.append(r)
    # spread: AxB cross covariance per period (iso non-overlapping events with both halves)
    sp = []
    for pn in ['pre1y', 'post_early60', 'post_60_182', 'post_late']:
        m = P[pn] & ~T.ovl10.values & T.locAB.values
        xa = T.loc[m, [c + 'A' for c in AX]].values
        xb = T.loc[m, [c + 'B' for c in AX]].values
        if len(xa) < 15:
            continue
        C = cross_cov(xa, xb)
        bs = []
        for _ in range(300):
            i = rng.integers(0, len(xa), len(xa))
            bs.append(np.diag(cross_cov(xa[i], xb[i])))
        bs = np.array(bs)
        r = dict(fam=F, period=pn, n_AB=len(xa))
        for k, c in enumerate(AX):
            r[f'var_true_{c}'] = C[k, k]
            r[f'var_true_{c}_se'] = bs[:, k].std()
            r[f'std_true_{c}'] = np.sqrt(max(C[k, k], 0))
        # centroid by cross-estimate: average of the two halves
        r.update({f'cent_{c}': 0.5 * (xa[:, k].mean() + xb[:, k].mean()) for k, c in enumerate(AX)})
        sp.append(r)
    # placebo: 60-d windows starting at random pre-M7 dates vs the year before them (same event selection)
    t = T.t.values
    xs = T[AX].values
    ok = sel & T['loc'].values
    pl = []
    for t0 in rng.uniform(M7 - 13 * 365 * DAY, M7 - 120 * DAY, 400):
        m1 = ok & (t >= t0) & (t < t0 + 60 * DAY)
        m2 = ok & (t >= t0 - 365 * DAY) & (t < t0)
        if m1.sum() < 30 or m2.sum() < 100:
            continue
        pl.append(dict(fam=F, t0=t0, n1=m1.sum(), **{f'd{c}': xs[m1, k].mean() - xs[m2, k].mean() for k, c in enumerate(AX)}))
    return pd.DataFrame(rows), pd.DataFrame(sp), pd.DataFrame(pl)


# ============================================================ prep: split + stack definitions
def blocks_split(idx, coord, rng_=None):
    """Time-matched split: blocks of BLOCK consecutive events; Hi = above block median of coord
    (or a random half if rng_ is given)."""
    hi, lo = [], []
    for b0 in range(0, len(idx) - BLOCK // 2, BLOCK):
        ii = idx[b0:b0 + BLOCK]
        if len(ii) < 10:
            continue
        if rng_ is None:
            o = np.argsort(coord[b0:b0 + BLOCK])
        else:
            o = rng_.permutation(len(ii))
        h = len(ii) // 2
        lo += list(ii[o[:h]])
        hi += list(ii[o[len(ii) - h:]])
    return np.array(hi), np.array(lo)


def match_weights(T, stacks, stack_t):
    """Weights for events of stacks in the first 182 d after the M7: per sub-period (before restart+60 d,
    restart+60 d .. M7+182 d), w = p_ref(bin) / p_period(bin) on 3x3 (s, d) bins at the reference terciles.
    Reference = pre-M7 year + >182 d after the M7 (isolated non-overlapping located events)."""
    P = periods(T)
    base = T.iso.values & ~T.ovl10.values & T['loc'].values
    R = base & (P['pre1y'] | P['post_late'])
    es = np.quantile(T.s[R], [1 / 3, 2 / 3])
    ed = np.quantile(T.d[R], [1 / 3, 2 / 3])
    binid = np.digitize(T.s.values, es) * 3 + np.digitize(T.d.values, ed)
    pR = np.bincount(binid[R], minlength=9) / R.sum()
    w_ev = np.ones(len(T))
    d, dr = T.day.values, T.dr.values
    subp = {'P1': (d >= 0) & (dr < 60), 'P2': (dr >= 60) & (d < 182)}
    info = {}
    for k, m in subp.items():
        mm = base & m
        pP = np.bincount(binid[mm], minlength=9) / max(mm.sum(), 1)
        ratio = np.where(pP > 0, pR / np.where(pP > 0, pP, 1), 1.0)
        w_ev[mm] = np.clip(ratio[binid[mm]], 0.1, 10.0)
        xs = T.loc[mm, ['s', 'd']].values
        ww = w_ev[mm]
        xr = T.loc[R, ['s', 'd']].values.mean(0)
        info[k] = dict(n=int(mm.sum()), pP=pP.tolist(), pR=pR.tolist(),
                       meas_shift_unweighted=(xs.mean(0) - xr).tolist(),
                       meas_shift_weighted=((ww[:, None] * xs).sum(0) / ww.sum() - xr).tolist(),
                       n_eff=float(ww.sum() ** 2 / (ww ** 2).sum()))
    weights = [w_ev[np.asarray(m)] for m in stacks]
    return weights, w_ev, info


def prep():
    T1, SP, PL, jobinfo = [], [], [], {}
    for F in FAMS:
        T = family_table(F)
        T.to_parquet(f'{OUT}/fam{F}_events.parquet')
        a, b, c = task1(F, T)
        T1.append(a)
        SP.append(b)
        PL.append(c)
        te = T.t.values
        iso_idx = np.where(T.iso.values)[0]
        # ---- location-split pools (pre-M7, isolated, non-overlapping)
        pre = T.iso.values & ~T.ovl10.values & (T.day.values < 0)
        splits, members = [], {}
        poolF = np.where(pre & T['loc'].values)[0]
        poolF = poolF[np.argsort(te[poolF])]
        poolA = np.where(pre & np.isfinite(T.sA.values))[0]
        poolA = poolA[np.argsort(te[poolA])]
        for c_ in AX:
            members[f'F{c_}'] = blocks_split(poolF, T[c_].values[poolF])
        for c_ in ['s', 'd']:
            members[f'A{c_}'] = blocks_split(poolA, T[c_ + 'A'].values[poolA])
        r2 = np.random.default_rng(F)
        for i in range(NNULL):
            members[f'Fnull{i}'] = blocks_split(poolF, None, r2)
            members[f'Anull{i}'] = blocks_split(poolA, None, r2)
        splits = [(k, v[0], v[1]) for k, v in members.items()]
        # ---- re-built stacks for task 3
        idx_loc = np.where(T.iso.values & ~T.ovl10.values & T['loc'].values)[0]
        idx_loc = idx_loc[np.argsort(te[idx_loc])]
        stacks = make_stacks(te, idx_loc, 50, 25)
        stack_t = np.array([te[m].mean() for m in stacks])
        weights, w_ev, minfo = match_weights(T, stacks, stack_t)
        variants = {'loc': (stacks, [None] * len(stacks), stack_t), 'match': (stacks, weights, stack_t)}
        # own intermediate file, read only by sci_transient_source_measure.py (trusted)
        pickle.dump(dict(iso_idx=iso_idx, splits=splits, variants=variants), open(f'{OUT}/fam{F}_jobdef.pkl', 'wb'))
        pickle.dump(dict(members=members, stacks=stacks, stack_t=stack_t, w_ev=w_ev, match_info=minfo,
                         restart=T.attrs['restart']), open(f'{OUT}/fam{F}_defs.pkl', 'wb'))
        jobinfo[F] = dict(n_events=len(T), n_iso=int(T.iso.sum()), n_poolF=len(poolF), n_poolA=len(poolA),
                          n_loc_stacks=len(stacks), restart=str(pd.to_datetime(T.attrs['restart'], unit='s')),
                          restart_day_after_m7=(T.attrs['restart'] - M7) / DAY, match=minfo)
        print(F, json.dumps(jobinfo[F], default=float)[:600], flush=True)
    T1, SP, PL = pd.concat(T1), pd.concat(SP), pd.concat(PL)
    T1.to_csv(f'{OUT}/task1_centroid_shifts.csv', index=False)
    SP.to_csv(f'{OUT}/task1_spread_AxB.csv', index=False)
    PL.to_csv(f'{OUT}/task1_placebo.csv', index=False)
    json.dump(jobinfo, open(f'{OUT}/prep_info.json', 'w'), indent=1, default=float)
    show = T1[(T1.events == 'iso_nonovl')]
    print(show[['fam', 'comp', 'loc', 'n1', 'n2', 'ds', 'ds_se', 'dd', 'dd_se', 'dn', 'dn_se']].round(1).to_string())
    print(SP.round(1).to_string())
    print(PL.groupby('fam')[['ds', 'dd', 'dn']].std().round(1))


if __name__ == '__main__':
    if '--prep' in sys.argv:
        prep()
    if '--analyze' in sys.argv:
        from sci_transient_source_analysis import analyze  # noqa: E402
        analyze()
    if '--report' in sys.argv:
        from sci_transient_source_report import report  # noqa: E402
        report()
