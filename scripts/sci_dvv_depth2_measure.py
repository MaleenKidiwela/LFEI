#!/usr/bin/env python3
"""Sub-band x lapse-window LFE-coda pair measurements for the dv/v depth study (sci_dvv_depth2.py).

Re-uses the exact measurement of scripts/sci_dvv_depth_measure.channel_job (same stacks as dvv_cwire, same
S alignment on 2-8 Hz, same tau_end from the 2-8 Hz envelope, stretching about S), with an extended
configuration list (2-4 / 4-8 / 2-8 Hz  x  full / early 3-8 s / late 8 s-tau_end).  Optional uniform
synthetic dv/v injection after a date (recovery calibration).
Rows with pair cc < 0.2 are dropped (never used by the fits) and columns are down-cast to keep the
output small.  Output: data/sci/dvv_depth2/pairs/fam{F}_pairs{tag}.parquet, fam{F}_channels{tag}.csv
"""
import argparse
import os
import sys
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import h5py
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import sci_dvv_depth_measure as M  # noqa: E402
from dvv_cwire import epoch_s, make_stacks  # noqa: E402

M.CONFIGS = [('b28_full', (2, 8), 3.0, None), ('b28_early', (2, 8), 3.0, 8.0), ('b28_late', (2, 8), 8.0, None),
             ('b24_full', (2, 4), 3.0, None), ('b24_early', (2, 4), 3.0, 8.0), ('b24_late', (2, 4), 8.0, None),
             ('b48_full', (4, 8), 3.0, None), ('b48_early', (4, 8), 3.0, 8.0), ('b48_late', (4, 8), 8.0, None)]
OUT = 'data/sci/dvv_depth2/pairs'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--min-ref-cc', type=float, default=0.4)
    ap.add_argument('--inject', nargs=2, default=None)
    a = ap.parse_args()
    F = a.family
    os.makedirs(OUT, exist_ok=True)
    tag = '' if a.inject is None else '_inj'
    h5f = f'data/waveforms/fam{F}.h5'
    with h5py.File(h5f, 'r') as h:
        te = epoch_s(pd.to_datetime(h['events/time_utc'][:].astype(str)))
    cat = pd.read_csv(a.catalog)
    cat = cat[cat.source != 'v1_only']
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
    idx = np.where(iso)[0]
    idx = idx[np.argsort(te[idx])]
    stacks = make_stacks(te, idx, 50, 25)
    stack_t = np.array([te[m].mean() for m in stacks])
    old = pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv')
    assert len(old) == len(stacks), 'stack definition differs from dvv_cwire run'
    meta0 = pd.read_csv(f'data/dvv/fam{F}_cwire_channels.csv')
    chans = meta0[meta0.median_cc_ref >= a.min_ref_cc].chan.tolist()
    picks = pd.read_csv(f'data/locations/fam{F}_3d_picks.csv').set_index('sta')
    inj = None if a.inject is None else (epoch_s([a.inject[0]])[0], float(a.inject[1]))
    jobs = [(h5f, c, float(picks.loc['.'.join(c.split('.')[:2]), 'S_final']), stacks, stack_t, idx, inj) for c in chans
            if '.'.join(c.split('.')[:2]) in picks.index]
    print(f'family {F}{tag}: {len(stacks)} stacks, {len(jobs)} channels', flush=True)
    res, metas = [], []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(M.channel_job, jobs):
            if r is None:
                continue
            P = r[0]
            P = P[P.cc >= 0.2]
            res.append(pd.DataFrame(dict(chan=P.chan.astype(str), cfg=P.cfg.astype(str), a=P.a.astype('int16'),
                                         b=P.b.astype('int16'), eps=P.eps.astype('float32'), cc=P.cc.astype('float32'))))
            metas.append(r[1])
            print('  ' + r[1]['chan'], len(P), flush=True)
    P = pd.concat(res, ignore_index=True)
    P['chan'] = P.chan.astype('category')
    P['cfg'] = P.cfg.astype('category')
    P.to_parquet(f'{OUT}/fam{F}_pairs{tag}.parquet', compression='zstd')
    pd.DataFrame(metas).to_csv(f'{OUT}/fam{F}_channels{tag}.csv', index=False)
    print('done', F, tag, len(P), flush=True)


if __name__ == '__main__':
    main()
