#!/usr/bin/env python3
"""Joint-fit cross-check for sci_transient_all.py: ONE parametric fit over all families' rows of a group/window
(offset per family:channel, shared trend/annual/steps/transient), as in sci_dvv_depth.py part B (which gave the
established surface early -0.177 % for families 23, 8, 27).  Subsets: all families, discovery (23, 8, 27), new.
Null: the same joint fit + (step, 60-d transient) at the common placebo dates of sci_transient_all.py (fixed IRLS
weights).  Recovery correction as in sci_transient_all.py.
Output: data/sci/transient_all/joint_fit.csv
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_transient_all_fit import CFG, GROUPS, T_FER, T_PET, Fit, load_rows  # noqa: E402

OUT = 'data/sci/transient_all'
ORIG = [23, 8, 27]
NPL_JOINT = 100


def job(args):
    grp, win, sub, fams, dates = args
    ps = []
    for F in fams:
        P, _ = load_rows(F)
        p = P[(P.cfg == CFG[win]) & GROUPS[grp](P.chan)]
        if p.chan.nunique() < 3:
            continue
        p = p.assign(chan=str(F) + ':' + p.chan)
        ps.append(p)
    p = pd.concat(ps, ignore_index=True)
    f = Fit(p)
    pf = f.solve(f.w, f.extra([('tr', T_PET), ('tr', T_FER)]))
    pl = []
    for t0 in dates:
        try:
            x = f.solve(f.w, f.extra([('step', t0), ('tr', t0)]))[8]
            pl.append(x if abs(x) < 0.05 else np.nan)
        except np.linalg.LinAlgError:
            pl.append(np.nan)
    return dict(group=grp, window=win, subset=sub, n_fam=len(ps), n_chan=f.nk, n_rows=len(p), m7_tr=f.m[6],
                m7_step=f.m[5], pet_tr=pf[7], fer_tr=pf[8], placebo=np.array(pl))


def main():
    # own cache written by sci_transient_all.py --fit (trusted)
    Z = pickle.load(open(f'{OUT}/fits.pkl', 'rb'))
    fams, dates = Z['fams'], Z['dates']
    rng = np.random.default_rng(3)
    dates = np.sort(rng.choice(dates, NPL_JOINT, replace=False))
    subs = {'all': fams, 'orig3': ORIG, 'new': [F for F in fams if F not in ORIG]}
    jobs = [(g, w, s, fl, dates) for g in GROUPS for w in CFG for s, fl in subs.items()]
    with Pool(4) as p:
        res = p.map(job, jobs)
    I = pd.read_csv('data/sci/dvv_depth/injection_recovery.csv')
    rec = {(r.group, 'early' if r.cfg == 'b28_early' else 'late'): r.recovery for r in I.itertuples()}
    rows, byk = [], {}
    for r in res:
        byk[(r['group'], r['window'], r['subset'])] = r
    for (g, w, s), r in byk.items():
        c = rec[(g, w)]
        pl = r['placebo'] / c
        row = {k: v for k, v in r.items() if k != 'placebo'}
        row.update(m7_tr_corr_pct=r['m7_tr'] / c * 100, pet_tr_corr_pct=r['pet_tr'] / c * 100,
                   fer_tr_corr_pct=r['fer_tr'] / c * 100, placebo_mean_pct=np.nanmean(pl) * 100,
                   placebo_sd_pct=np.nanstd(pl) * 100, n_placebo=int(np.isfinite(pl).sum()))
        for k in ['m7', 'pet', 'fer']:
            row[f'{k}_z'] = (r[f'{k}_tr'] / c - np.nanmean(pl)) / np.nanstd(pl)
        rows.append(row)
    for g in GROUPS:
        for s in subs:
            e, l_ = byk[(g, 'early', s)], byk[(g, 'late', s)]
            ce, cl = rec[(g, 'early')], rec[(g, 'late')]
            pd_ = e['placebo'] / ce - l_['placebo'] / cl
            row = dict(group=g, window='early-late', subset=s, n_fam=e['n_fam'])
            for k in ['m7', 'pet', 'fer']:
                v = e[f'{k}_tr'] / ce - l_[f'{k}_tr'] / cl
                row[f'{k}_tr_corr_pct'] = v * 100
                row[f'{k}_z'] = (v - np.nanmean(pd_)) / np.nanstd(pd_)
            row.update(placebo_mean_pct=np.nanmean(pd_) * 100, placebo_sd_pct=np.nanstd(pd_) * 100,
                       n_placebo=int(np.isfinite(pd_).sum()))
            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/joint_fit.csv', index=False)
    print(R.drop(columns=[c for c in R if c in ('m7_tr', 'm7_step', 'pet_tr', 'fer_tr')]).round(3).to_string())


if __name__ == '__main__':
    main()
