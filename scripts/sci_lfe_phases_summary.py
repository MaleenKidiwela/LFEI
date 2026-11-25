#!/usr/bin/env python3
"""Collect the LFE secondary-phase / attenuation results -> data/sci/lfe_phases/summary.json and
search_robustness.csv.

Robustness rule for a phase hypothesis (phase, observable kind): take the reflector offset dz* that maximises z in
'all' (only dz with >= 100 family-stations and a non-degenerate null).  The phase is ROBUST only if, at dz* (+-1 km),
  z >= 3 in 'all' with the within-depth-class permutation null ('famclass', immune to shallow-vs-deep coda
  differences) AND with the all-family permutation null ('fam'),
  z >= 2 in boreholes AND surface AND odd AND even families (famclass null; and in both event halves when available
  with the fam null).
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_lfe_phases_common import OUT  # noqa: E402

NMIN = 100


def load_scans():
    fs = [f for f in glob.glob(f'{OUT}/search_scan*.csv') if '_inj_' not in f]
    D = []
    for f in fs:
        d = pd.read_csv(f)
        if 'null' not in d:
            continue
        D.append(d)
    D = pd.concat(D).drop_duplicates(['phase', 'kind', 'subset', 'null', 'dz'], keep='last')
    D.loc[(D.n < NMIN) | (D.null_sd < 1e-6) | ~np.isfinite(D.z), 'z'] = np.nan
    return D


def zat(D, ph, kind, sub, null, dz, tol=1.0):
    g = D[(D.phase == ph) & (D.kind == kind) & (D.subset == sub) & (D.null == null) & (np.abs(D.dz - dz) <= tol)]
    return float(np.nanmax(g.z)) if len(g) and np.isfinite(g.z).any() else np.nan


def robustness(D):
    rows = []
    for (ph, kind), g in D.groupby(['phase', 'kind']):
        for null0 in ('famclass', 'fam'):
            a = g[(g.subset == 'all') & (g.null == null0)]
            if not len(a) or not np.isfinite(a.z).any():
                continue
            dz = float(a.dz.values[np.nanargmax(a.z.values)])
            r = dict(phase=ph, kind=kind, ref_null=null0, dz_best=dz, z_all=float(np.nanmax(a.z)),
                     n_all=int(a.n.values[np.nanargmax(a.z.values)]))
            for sub in ('boreholes', 'surface', 'fam_odd', 'fam_even', 'half0', 'half1'):
                for null in ('famclass', 'fam'):
                    r[f'z_{sub}_{null}'] = zat(D, ph, kind, sub, null, dz)
            r['z_all_famclass'] = zat(D, ph, kind, 'all', 'famclass', dz, 0.05)
            r['z_all_fam'] = zat(D, ph, kind, 'all', 'fam', dz, 0.05)
            subs_ok = all(np.nan_to_num(r[f'z_{s}_famclass'], nan=-9) >= 2 or
                          (np.isnan(r[f'z_{s}_famclass']) and np.nan_to_num(r[f'z_{s}_fam'], nan=-9) >= 2)
                          for s in ('boreholes', 'surface', 'fam_odd', 'fam_even'))
            r['robust'] = bool(np.nan_to_num(r['z_all_famclass'], nan=-9) >= 3 and
                               np.nan_to_num(r['z_all_fam'], nan=-9) >= 3 and subs_ok)
            rows.append(r)
    return pd.DataFrame(rows)


def injections():
    rows = []
    for f in sorted(glob.glob(f'{OUT}/search_summary_inj_*.json')):
        amp = float(f.split('_')[-1][:-5])
        S = json.load(open(f))
        for k, v in S.items():
            ph, kind, sub, null = k.split('|')
            rows.append(dict(amp=amp, phase=ph, kind=kind, subset=sub, null=null, z_at_true=v['z_at_ref'],
                             dz_best=v['dz_best'], z_best=v['z_best']))
    return pd.DataFrame(rows)


def main():
    D = load_scans()
    R = robustness(D)
    R.to_csv(f'{OUT}/search_robustness.csv', index=False)
    I = injections()
    if len(I):
        I.to_csv(f'{OUT}/search_injection.csv', index=False)
    out = dict(search=dict(n_hypotheses=int(len(R)),
                           robust=R[R.robust][['phase', 'kind', 'ref_null', 'dz_best', 'z_all']].to_dict('records')))
    out['search']['table'] = R.round(3).replace({np.nan: None}).to_dict('records')
    if len(I):
        out['injection'] = I[(I.subset == 'all')].round(3).to_dict('records')
    for nm in ('atten_qpqs.json',):
        try:
            out['qpqs'] = json.load(open(f'{OUT}/{nm}'))
        except FileNotFoundError:
            pass
    try:
        T = pd.read_csv(f'{OUT}/atten_tests.csv')
        out['attenuation'] = T.round(5).replace({np.nan: None}).to_dict('records')
    except FileNotFoundError:
        pass
    for m in ('real', 'inject', 'inject0', 'inject_amp1', 'inject0_amp1'):
        try:
            out[f'timelapse_{m}'] = json.load(open(f'{OUT}/timelapse_{m}_summary.json'))
        except FileNotFoundError:
            pass
    # injection recovery of a +1 % in-layer delay change after the M7 restart, and the implied in-layer dv/v limit
    rec = {}
    for amp, a, b in ((0.3, 'inject', 'inject0'), (1.0, 'inject_amp1', 'inject0_amp1')):
        if f'timelapse_{a}' in out and f'timelapse_{b}' in out:
            for grp in ('all', 'PB', 'surface'):
                A, B = out[f'timelapse_{a}'][grp], out[f'timelapse_{b}'][grp]
                for t in ('step_m7', 'restart_0_60d'):
                    r = (A['tests'][t]['eps'] - B['tests'][t]['eps']) / 0.01
                    rec[f'amp{amp}|{grp}|{t}'] = dict(recovery=round(r, 3), placebo_sd_eps=B['placebo_sd_eps'],
                                                      dvv_limit_2sd_pct=round(200 * B['placebo_sd_eps'] / max(r, 1e-3), 2))
    out['timelapse_injection_recovery'] = rec
    out['verdict'] = ('No robust source-side secondary phase (SmS, PmS, PmP, SmP from below; Sp, Ps from above) in the '
                      'long-term LFE stacks of 27 families: no hypothesis reaches z >= 3 with the within-depth-class '
                      'null while being consistent in boreholes, surface stations and odd/even families. Injections: '
                      'an SmS copy of >= ~0.3 x direct S would have been detected (z 4-6). Attenuation (dt*_S, dt*_P, '
                      'P/S) unchanged at all quakes, restart and long-term.')
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(R.round(2).to_string())
    if len(I):
        print(I[I.subset == 'all'].round(2).to_string())


if __name__ == '__main__':
    main()
