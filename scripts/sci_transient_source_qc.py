#!/usr/bin/env python3
"""Control for task 2: time-matched splits of the same pre-M7 pool by event QUALITY instead of location
(catalog ccsum_v2, catalog meancc_v2, relocation mean_cc), measured with the same machinery
(sci_transient_source_measure.py --tag _qc). If such splits give apparent dv/v comparable to the location
splits, the location sensitivity would be confounded by event quality/SNR."""
import os
import pickle
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_transient_source import blocks_split  # noqa: E402
from sci_transient_source_common import FAMS, OUT  # noqa: E402

for F in FAMS:
    T = pd.read_parquet(f'{OUT}/fam{F}_events.parquet')
    full = pd.read_csv(f'data/relocations/fam{F}_events_3d.csv')[['ev', 'mean_cc']]
    T = T.merge(full, on='ev', how='left')
    with h5py.File(f'data/waveforms/fam{F}.h5', 'r') as h:
        T['ccsum'] = h['events/ccsum_v2'][:]
        T['meancc_cat'] = h['events/meancc_v2'][:]
    pre = T.iso.values & ~T.ovl10.values & (T.day.values < 0) & T['loc'].values
    pool = np.where(pre)[0]
    pool = pool[np.argsort(T.t.values[pool])]
    splits = [(f'Q{c}', *blocks_split(pool, T[c].values[pool])) for c in ['ccsum', 'meancc_cat', 'mean_cc']]
    iso_idx = np.where(T.iso.values)[0]
    # own intermediate file, read only by sci_transient_source_measure.py (trusted)
    pickle.dump(dict(iso_idx=iso_idx, splits=splits, variants={}), open(f'{OUT}/fam{F}_jobdef_qc.pkl', 'wb'))
    pickle.dump({s[0]: (s[1], s[2]) for s in splits}, open(f'{OUT}/fam{F}_defs_qc.pkl', 'wb'))
    print(F, [(s[0], len(s[1])) for s in splits])
