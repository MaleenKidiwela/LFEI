#!/usr/bin/env python3
"""Deep/shallow dv/v, item 2: download event waveforms (NCEDC FDSN, anonymous) for ordinary earthquakes near
the MTJ that have a catalog neighbour (<=2 km, |dM|<=0.5, >60 d apart) -> candidate repeaters.
Catalog: data/sci/dvv_deep_shallow/rep/ncedc_events.txt (NCEDC fdsnws/event, M>=1.5, 2015-2026,
39.8-40.8 N, 125.2-123.2 W).  Window origin -10 .. +60 s; stations = long-running LFE dv/v stations.
Output data/sci/dvv_deep_shallow/rep/wf/{evid}.mseed (+ candidates.csv).  Stops if /home free < 40 GB.
"""
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.spatial import cKDTree

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_deep_shallow/rep'
CH = [('NC', 'KMR', '*', 'HH?'), ('NC', 'KMPB', '*', 'HH?'), ('PB', 'B045', '*', 'EH?'), ('PB', 'B047', '*', 'EH?'),
      ('PB', 'B932', '*', 'EH?'), ('PB', 'B049', '*', 'EH?'), ('PB', 'B046', '*', 'EH?'), ('BK', 'BRIC', '*', 'HH?'),
      ('BK', 'HAYF', '*', 'HH?')]


def candidates():
    d = pd.read_csv(f'{OUT}/ncedc_events.txt', sep='|')
    d.columns = [c.strip().lstrip('#') for c in d.columns]
    d['t'] = pd.to_datetime(d.Time)
    x = (d.Longitude + 124) * 85.0
    y = (d.Latitude - 40.3) * 111.2
    z = d['Depth/km'].values
    m = d.Magnitude.values
    pr = cKDTree(np.c_[x, y, z / 1.5]).query_pairs(2.0, output_type='ndarray')
    i, j = pr[:, 0], pr[:, 1]
    ok = (np.abs(m[i] - m[j]) <= 0.5) & (np.abs((d.t.values[i] - d.t.values[j]) / np.timedelta64(1, 'D')) > 60)
    ev = np.unique(np.r_[i[ok], j[ok]])
    c = d.iloc[ev].copy()
    c.to_csv(f'{OUT}/candidates.csv', index=False)
    return c


def fetch(row, client):
    f = f'{OUT}/wf/{row.EventID}.mseed'
    if os.path.exists(f) or os.path.exists(f + '.none'):
        return 'skip'
    t0 = UTCDateTime(str(row.Time))
    bulk = [(n, s, l, c, t0 - 10, t0 + 60) for n, s, l, c in CH]
    for k in range(3):
        try:
            st = client.get_waveforms_bulk(bulk)
            st.write(f, format='MSEED')
            return 'ok'
        except Exception as e:                      # noqa: BLE001
            if 'No data' in str(e) or '204' in str(e):
                open(f + '.none', 'w').close()
                return 'none'
            time.sleep(5 * (k + 1))
    return 'fail'


def main():
    os.makedirs(f'{OUT}/wf', exist_ok=True)
    c = candidates()
    print(len(c), 'candidate events', flush=True)
    client = Client('NCEDC', timeout=120)
    rows = list(c.itertuples())
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for k in range(0, len(rows), 40):
            if shutil.disk_usage('/home').free / 1e9 < 40:
                print('STOP: /home free < 40 GB', flush=True)
                break
            res = list(ex.map(lambda r: fetch(r, client), rows[k:k + 40]))
            n += len(res)
            print(f'{n}/{len(rows)} {pd.Series(res).value_counts().to_dict()} {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
