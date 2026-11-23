#!/usr/bin/env python3
"""Measurement helper for sci_transient_all.py: early (tau 3-8 s) and late (tau 8 s - end) 2-8 Hz lapse-window
pairs for one family, with EXACTLY the procedure of sci_dvv_depth_measure.channel_job (same stacks, S alignment,
stretching about S, pair set), restricted to the 2-8 Hz configurations (b28_full, b28_early, b28_late) to save
time.  Families already measured in data/sci/dvv_depth (23, 8, 27) are reused there, not re-measured.
Output: data/sci/transient_all/fam{F}_pairs.parquet, fam{F}_channels.csv
"""
import argparse
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sci_dvv_depth_measure as M  # noqa: E402
from dvv_cwire import epoch_s, make_stacks  # noqa: E402

M.CONFIGS = [c for c in M.CONFIGS if c[1] == (2, 8)]     # forked workers inherit the patched global
OUT = 'data/sci/transient_all'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    F = a.family
    h5f = f'data/waveforms/fam{F}.h5'
    with h5py.File(h5f, 'r') as h:
        te = epoch_s(pd.to_datetime(h['events/time_utc'][:].astype(str)))
    cat = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv')
    cat = cat[cat.source != 'v1_only']
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
    idx = np.where(iso)[0]
    idx = idx[np.argsort(te[idx])]
    stacks = make_stacks(te, idx, 50, 25)
    stack_t = np.array([te[m].mean() for m in stacks])
    old = pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv')
    assert len(old) == len(stacks), 'stack definition differs from dvv_cwire run'
    told = epoch_s(pd.to_datetime(old.t, format='mixed'))
    print(f'max |stack_t - cwire t| = {np.max(np.abs(told - stack_t)):.1f} s', flush=True)
    meta0 = pd.read_csv(f'data/dvv/fam{F}_cwire_channels.csv')
    chans = meta0[meta0.median_cc_ref >= 0.4].chan.tolist()
    if a.only:
        chans = [c for c in chans if c in a.only.split(',')]
    picks = pd.read_csv(f'data/locations/fam{F}_3d_picks.csv').set_index('sta')
    jobs = [(h5f, c, float(picks.loc['.'.join(c.split('.')[:2]), 'S_final']), stacks, stack_t, idx, None) for c in chans]
    print(f'family {F}: {len(stacks)} stacks, {len(jobs)} channels', flush=True)
    res, metas = [], []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(M.channel_job, jobs):
            if r is None:
                continue
            res.append(r[0])
            metas.append(r[1])
    P = pd.concat(res, ignore_index=True)
    P['cfg'] = P.cfg.astype(str)
    sfx = '' if not a.only else '_test'
    P.to_parquet(f'{OUT}/fam{F}_pairs{sfx}.parquet')
    pd.DataFrame(metas).to_csv(f'{OUT}/fam{F}_channels{sfx}.csv', index=False)
    print(f'family {F}: done, {len(metas)} channels measured, {len(P)} rows', flush=True)


if __name__ == '__main__':
    main()
